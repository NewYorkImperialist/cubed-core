#!/usr/bin/env python3
"""Sequential windowed decode used by the canonical research pipeline.

The decoder advances from the supplied initial cube state, scores bounded
candidate windows from tracker evidence, and records commit, extension, or
abstention decisions.
"""
from __future__ import annotations

import base64
import bisect
import copy
import hashlib
import json
import math
import os
import time
from typing import Callable, NamedTuple, Optional

import numpy as np

from analysis import cubemodel as CM
import detect.bridge_search as BS
import detect.intraburst_motion as IBM
import detect.ll_completion_live as LLC
import detect.scrub_span_view as SSV

# Runner-visible end-to-end API marker; see scripts/run_decode.sh.
SCRUB_EXPERIMENT_API = frozenset({
    "dense-reads-v1", "dense-prefix-v1", "gate-drop-slots-v1",
    "intraburst-phase-slots-v1",
})

# ---- reused constants (KNOB-ZERO; sources documented above) ------------
DEPTH_CAP = BS.DEPTH_CAP          # 5: ball depth (memory envelope)
STRUCT_CAP = 2 * DEPTH_CAP        # exact/MITM + seeded/typed word cap
# exact-enumeration envelope: L_hi <= 2 => canonical ball <= 1+18+18*15 = 289
# (+ empty/committed) <= ~343 candidates -- the round-3 "~400" bar, derived
# from cube geometry. Module-level so the beam-vs-exact equivalence test can
# force either path.
EXACT_MAX_LHI = 2
_DEVICE_BEAM_WARNED = False
# The live hook passes the tracker's beam width. This default serves only direct
# callers and tests.
DEFAULT_BEAM_K = 8

_TERMINAL_DEDUP_POLICY_STATE_ONLY = "historical-state-only"
_TERMINAL_DEDUP_POLICY_PHYSICAL_HISTORY = (
    "physical-program-history-aware"
)
_TERMINAL_DEDUP_KEY_SCHEMA = (
    "packed-state-om-count[3]+word_len[1]+word[W]+timed_len[1]+"
    "timing[W]+slot_merge[4+2W]"
)
_DEVICE_DEDUP_BASE_KEY_WIDTH = 3
# Canonical live-action order: SKIP + 18 SINGLE + 18*15 DOUBLE rows.
_CANONICAL_COUNT_STREAMING_POLICY = (
    "SCORE_STATE_BOUND-exact-two-pass-rotation-aware-reject-no-top-k"
)
_CANONICAL_COUNT_STREAMING_PREREG_SHA256 = (
    "e3ff4690a00d7910cb3fbc919726a7ffee8716b067b37b0daffa2fc236cc70d8"
)
_UNOWNED_MOTION_CAPABILITY = (
    "unowned-motion-canonical-count-rotation-aware-streaming-v5a"
)
_V18_SHADOW_CAPACITY_PLANE_ROLES = frozenset((
    "nn-fused-primary",
    "nn-v18-terminal",
))
_PARTIAL_LEFT_NEUTRAL_PREFIX_CAPABILITY = (
    "partial-left-neutral-prefix-v1"
)
_PARTIAL_LEFT_NEUTRAL_PREFIX_KIND = "partial_left_neutral_prefix"
_PARTIAL_LEFT_NEUTRAL_PREFIX_REASON = (
    "attempt begins after the owned 16-frame grid starts"
)
_PARTIAL_LEFT_NEUTRAL_PREFIX_PREREG_SHA256 = (
    "a14e7ee9dd3d830e6a7462a589ba1e014c67d5616bf5705a065cbff37bc8e396"
)


def _sequential_payload_width(L_hi):
    """Word/timing width for one bounded chronological beam window.

    Preserve the historical ``STRUCT_CAP`` allocation for ordinary short
    windows so their checkpoint/layout contract stays byte-for-byte stable.
    Longer sequential windows own their declared physical length ceiling;
    actual expansion and materialization remain guarded by
    ``BS.SCORE_STATE_BOUND``.
    """
    width = int(L_hi)
    if width < 0:
        raise ValueError("sequential payload width must be non-negative")
    return max(int(STRUCT_CAP), width)


def _terminal_program_dedup_key_width(payload_width):
    """Fixed total CUDA dedup-key width for one terminal payload."""
    width = int(payload_width)
    if width < 0:
        raise ValueError("terminal payload width must be non-negative")
    return _DEVICE_DEDUP_BASE_KEY_WIDTH + 6 + 4 * width




def _device_beam_checkpoint_family_key(
        *, plane_source, a_span, state, start_ois, beam_k, band,
        late_evidence_band_active, om_nbrs, raw_burst_stream,
        payload_width):
    """Static identity for a resident chronological-beam frontier family.

    ``payload_width`` is allocation identity: a frontier created for a shorter
    extension cannot be resumed into wider word/timing tensors.  Per-step
    action/read/witness identities remain in the existing checkpoint key.
    """
    return (
        plane_source,
        int(a_span), np.asarray(state, dtype=np.int8).tobytes(),
        tuple(int(oi) for oi in start_ois), int(beam_k), float(band),
        bool(late_evidence_band_active),
        tuple(tuple(sorted(map(int, row))) for row in om_nbrs),
        bool(raw_burst_stream), int(payload_width),
    )


def _validated_ll_selected_mutation(
        *, decision, action_slots, orientations, onset_frame, final_frame,
        span_ends, prefix_windows, expected_prefix_moves, init_arr, ll_target):
    """Validate and build one selected LL replacement transaction.

    This is the final mutation boundary, downstream of the live scorer.  Treat
    its decision as untrusted: a partial/malformed receipt must not exploit
    ``zip`` truncation, manufacture timing/orientation rows, or set a solved
    endpoint for a word other than the one that will actually be emitted.
    Returns only after replaying that exact proposed word to both canonical
    SOLVED and the caller's terminal target.
    """

    def strict_int_tuple(raw, label, *, allow_equal):
        if type(raw) is not tuple or not raw:
            raise ValueError(f"selected {label} must be a nonempty tuple")
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer))
               for value in raw):
            raise ValueError(f"selected {label} contains a non-integer")
        values = tuple(int(value) for value in raw)
        if allow_equal:
            chronological = all(a <= b for a, b in zip(values, values[1:]))
        else:
            chronological = all(a < b for a, b in zip(values, values[1:]))
        if not chronological:
            raise ValueError(f"selected {label} is not chronological")
        if any(not int(onset_frame) < value <= int(final_frame)
               for value in values):
            raise ValueError(f"selected {label} lies outside the LL seam")
        return values

    if type(decision) is not LLC.LLLiveDecision:
        raise ValueError("selected LL decision has the wrong type")
    if decision.status != "selected":
        raise ValueError("selected LL decision status is not selected")
    selected = decision.selected
    if type(selected) is not LLC.LLLiveHypothesis:
        raise ValueError("selected LL hypothesis is missing or malformed")
    if selected.source != "table":
        raise ValueError("selected LL hypothesis is not table-owned")
    if (type(decision.retained) is not tuple
            or len(decision.retained) != 1
            or decision.retained[0] is not selected):
        raise ValueError("selected LL hypothesis is not uniquely retained")

    literal_moves = selected.literal_moves
    if (type(literal_moves) is not tuple or not literal_moves
            or any(type(move) is not str or not move or move not in CM.PERMS
                   for move in literal_moves)):
        raise ValueError("selected LL literal trace is empty or malformed")
    canonical_trace = LLC.canonicalize_ll_moves(literal_moves)
    if canonical_trace.literal_moves != literal_moves:
        raise ValueError("selected LL literal trace is not canonicalized")

    canonical_moves = selected.canonical_moves
    if (type(canonical_moves) is not tuple or not canonical_moves
            or any(type(move) is not str or move not in BS.MOVES
                   for move in canonical_moves)):
        raise ValueError("selected LL canonical trace is empty or malformed")
    if canonical_trace.canonical_moves != canonical_moves:
        raise ValueError("selected LL canonicalization receipt does not match")

    action_frames = strict_int_tuple(
        selected.action_frames, "action frames", allow_equal=False)
    move_frames = strict_int_tuple(
        selected.move_frames, "move frames", allow_equal=True)
    move_ois = selected.move_om_indices
    if (type(move_ois) is not tuple
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer))
                   for value in move_ois)):
        raise ValueError("selected move orientations are malformed")
    move_ois = tuple(int(value) for value in move_ois)
    if not (len(canonical_moves) == len(move_frames) == len(move_ois)):
        raise ValueError("selected canonical/timing/orientation lengths differ")
    if len(action_frames) != len(canonical_trace.steps):
        raise ValueError("selected literal/action lengths differ")

    if type(orientations) not in (list, tuple) or not orientations:
        raise ValueError("LL orientations are unavailable")
    if any(oi < 0 or oi >= len(orientations) for oi in move_ois):
        raise ValueError("selected move orientation index is out of range")

    if type(action_slots) is not tuple:
        raise ValueError("LL action slots are malformed")
    slot_by_frame = {}
    for slot in action_slots:
        if (type(slot) is not LLC.LLActionSlot
                or isinstance(slot.frame, (bool, np.bool_))
                or not isinstance(slot.frame, (int, np.integer))):
            raise ValueError("LL action slot is malformed")
        frame = int(slot.frame)
        if frame in slot_by_frame:
            raise ValueError("LL action slots share a frame")
        slot_by_frame[frame] = slot

    for step_index, (step, action_frame) in enumerate(
            zip(canonical_trace.steps, action_frames)):
        slot = slot_by_frame.get(action_frame)
        if slot is None:
            raise ValueError("selected action frame is absent from LL slots")
        step_kind = ("rotation" if CM.token_base(step.literal_move)
                     in CM.ROTATION_BASES else "move")
        compatible = (
            step_kind == "move" and slot.kind in ("move", "either")
        ) or (
            step_kind == "rotation" and slot.kind in ("rotation", "either")
        )
        if not compatible:
            raise ValueError("selected literal action has the wrong slot kind")
        start, end = step.canonical_start, step.canonical_end
        if tuple(move_frames[start:end]) != (action_frame,) * (end - start):
            raise ValueError("selected move frames do not bind to literal actions")
        if end - start > 1 and len(set(move_ois[start:end])) != 1:
            raise ValueError("one literal action has inconsistent move orientations")
        if step_index and action_frame <= action_frames[step_index - 1]:
            raise ValueError("selected literal actions are not strictly ordered")

    if (type(span_ends) not in (list, tuple) or not span_ends
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer))
                   for value in span_ends)):
        raise ValueError("LL span lattice is malformed")
    span_ends = tuple(int(value) for value in span_ends)
    if any(a >= b for a, b in zip(span_ends, span_ends[1:])):
        raise ValueError("LL span lattice is not chronological")

    if type(prefix_windows) is not list:
        raise ValueError("LL prefix windows are malformed")
    flattened_prefix = []
    for window in prefix_windows:
        if type(window) is not dict:
            raise ValueError("LL prefix window is malformed")
        tokens = window.get("tokens")
        oms = window.get("oms")
        land = window.get("land")
        if (type(tokens) not in (list, tuple)
                or any(type(move) is not str or move not in BS.MOVES
                       for move in tokens)
                or type(oms) not in (list, tuple)
                or len(oms) != len(tokens)
                or isinstance(land, (bool, np.bool_))
                or not isinstance(land, (int, np.integer))
                or not 0 <= int(land) < len(span_ends)):
            raise ValueError("LL prefix window payload is malformed")
        flattened_prefix.extend(tokens)

    if (type(expected_prefix_moves) is not tuple
            or any(type(move) is not str or move not in BS.MOVES
                   for move in expected_prefix_moves)
            or tuple(flattened_prefix) != expected_prefix_moves):
        raise ValueError("LL proposed prefix differs from the verified prefix")

    replacement = []
    for index in range(len(canonical_moves)):
        frame = move_frames[index]
        land = bisect.bisect_left(span_ends, frame)
        if land >= len(span_ends):
            raise ValueError("selected move frame is outside the span lattice")
        replacement.append({
            "tokens": [canonical_moves[index]],
            "land": int(land),
            "oms": [orientations[move_ois[index]]],
            # The selected suffix is established only by its final replay.
            # Its action frames remain internal search evidence, not
            # per-move timing authority.
            "checkpoint_frame": int(final_frame),
        })

    proposed_windows = list(prefix_windows) + replacement
    proposed_word = tuple(
        move for window in proposed_windows for move in window["tokens"])
    if proposed_word != expected_prefix_moves + canonical_moves:
        raise ValueError("LL proposed emitted word differs from its receipt")
    replay = CM.apply_seq(np.asarray(init_arr, np.int8), proposed_word)
    if (not np.array_equal(replay, CM.SOLVED)
            or not np.array_equal(replay, np.asarray(ll_target, np.int8))):
        raise ValueError("LL exact proposed emitted word does not solve target")
    return selected, proposed_windows, proposed_word, replay


def _reconstruction_checkpoint_groups(
        windows, tail_moves, *, final_checkpoint_frame):
    """Return ordered state checkpoints without manufacturing move timing.

    A normal scrub window establishes its whole emitted word at the terminal
    evidence span. Multiple moves at the same checkpoint are one atomic group.
    The legacy terminal bridge has only the final available checkpoint. This
    projection is receipt-only and never changes ``land`` or ``move_layer``.
    """
    if type(windows) is not list or type(tail_moves) is not list:
        raise ValueError("checkpoint inputs must be lists")
    if (isinstance(final_checkpoint_frame, (bool, np.bool_))
            or not isinstance(final_checkpoint_frame, (int, np.integer))
            or int(final_checkpoint_frame) < 0):
        raise ValueError("final checkpoint frame is malformed")

    groups = []

    def append_group(frame, move_count):
        if (isinstance(frame, (bool, np.bool_))
                or not isinstance(frame, (int, np.integer))
                or int(frame) < 0):
            raise ValueError("checkpoint frame is malformed")
        if (isinstance(move_count, (bool, np.bool_))
                or not isinstance(move_count, (int, np.integer))
                or int(move_count) <= 0):
            raise ValueError("checkpoint move count is malformed")
        frame = int(frame)
        move_count = int(move_count)
        if groups and frame < groups[-1]["frame"]:
            raise ValueError("checkpoint frames are not chronological")
        if groups and frame == groups[-1]["frame"]:
            groups[-1]["move_count"] += move_count
        else:
            groups.append({"frame": frame, "move_count": move_count})

    for window in windows:
        if type(window) is not dict:
            raise ValueError("checkpoint window is malformed")
        tokens = window.get("tokens")
        if (type(tokens) not in (list, tuple)
                or any(type(move) is not str or move not in BS.MOVES
                       for move in tokens)):
            raise ValueError("checkpoint window tokens are malformed")
        if tokens:
            append_group(window.get("checkpoint_frame"), len(tokens))

    if (any(type(move) is not str or move not in BS.MOVES
            for move in tail_moves)):
        raise ValueError("terminal bridge tokens are malformed")
    if tail_moves:
        append_group(final_checkpoint_frame, len(tail_moves))
    return groups


class TransitionSlot(NamedTuple):
    """One chronological action opportunity in the scrub prefix lattice."""

    kind: str                  # "motion" (hard-count) or a typed optional
    frame_lo: int
    frame_hi: int
    parent_frame: Optional[int] = None
    period_lo: Optional[int] = None
    period_hi: Optional[int] = None
    phase_index: Optional[int] = None
    phase_count: Optional[int] = None
































class _BeamBackpointer:
    """Persistent word/timing path used only by the typed visual CPU beam."""

    __slots__ = ("parent", "word_chunk", "timing_chunk", "word_len",
                 "timing_len", "_word", "_timing")

    def __init__(self, parent, word_chunk=(), timing_chunk=()):
        self.parent = parent
        self.word_chunk = tuple(int(value) for value in word_chunk)
        self.timing_chunk = tuple(int(value) for value in timing_chunk)
        self.word_len = ((0 if parent is None else parent.word_len)
                         + len(self.word_chunk))
        self.timing_len = ((0 if parent is None else parent.timing_len)
                           + len(self.timing_chunk))
        self._word = None
        self._timing = None

    def word(self):
        if self._word is None:
            prefix = (() if self.parent is None else self.parent.word())
            self._word = prefix + self.word_chunk
        return self._word

    def timing(self):
        if self._timing is None:
            prefix = (() if self.parent is None else self.parent.timing())
            self._timing = prefix + self.timing_chunk
        return self._timing


def visual_episodes_for_window(episodes, frame_start, frame_end):
    """Assign global episodes to exactly one half-open scrub window.

    Ownership is by the episode's terminal frame: ``start < hi <= end``.  A
    run that straddles a committed boundary therefore cannot be inserted again
    by the following window, and extension retries see the same immutable
    global episode identity.
    """
    start, end = int(frame_start), int(frame_end)
    normalized = sorted({(int(lo), int(hi)) for lo, hi in (episodes or ())
                         if int(lo) <= int(hi)})
    return tuple((lo, hi) for lo, hi in normalized
                 if start < hi <= end)


def typed_transition_slots(move_frames, visual_episodes, frame_start, frame_end,
                           dropped_intervals=(), intraburst_phase_slots=()):
    """Merge hard motion points with optional transition intervals.

    Exact interval overlap coalesces a visual episode with the existing motion
    slot.  Eligible final-gate records sharing the exact same structural gap
    are likewise one typed ``dropped`` slot; no proximity tolerance is used.
    Certified intra-burst phases are a third, separately-provenanced optional
    source.  Their authoritative parent event is never required to survive a
    later move gate: only an actually kept hard event inside the optional phase
    coalesces it.  Any malformed/overlapping external phase payload rejects the
    complete phase source and returns the exact pre-feature slot list.

    No optional source changes the hard motion count.  The returned order is
    chronological and deterministic; duplicate motion points retain legacy
    order.
    """
    start, end = int(frame_start), int(frame_end)
    motion = [int(frame) for frame in move_frames
              if start < int(frame) <= end]
    episodes = visual_episodes_for_window(
        visual_episodes, frame_start, frame_end)
    slots = [TransitionSlot("motion", frame, frame) for frame in motion]
    for lo, hi in episodes:
        if any(lo <= frame <= hi for frame in motion):
            continue
        slots.append(TransitionSlot("visual", lo, hi))
    baseline_slots = list(slots)
    phase_rows = []
    try:
        for row in (intraburst_phase_slots or ()):
            phase = IBM.intraburst_phase_slot_from_value(row)
            if not (phase.frame_lo <= phase.frame_hi
                    and phase.period_lo <= phase.frame_lo
                    and phase.frame_hi <= phase.period_hi
                    and not (phase.frame_lo <= phase.parent_frame
                             <= phase.frame_hi)):
                raise ValueError("invalid intra-burst phase provenance")
            if start < phase.frame_hi <= end:
                phase_rows.append(phase)
    except (KeyError, TypeError, ValueError, OverflowError):
        phase_rows = []
        slots = list(baseline_slots)
    phase_rows.sort(key=lambda row: (
        row.frame_hi, row.frame_lo, row.parent_frame, row.phase_index))
    if phase_rows:
        phase_conflict = False
        proposed = list(slots)
        for phase in phase_rows:
            if any(phase.frame_lo <= frame <= phase.frame_hi
                   for frame in motion):
                phase_conflict = True
                break
            overlaps = [
                slot for slot in proposed if slot.kind != "motion"
                and not (slot.frame_hi < phase.frame_lo
                         or phase.frame_hi < slot.frame_lo)
            ]
            if any((slot.frame_lo, slot.frame_hi)
                   != (phase.frame_lo, phase.frame_hi)
                   for slot in overlaps):
                phase_conflict = True
                break
            # Exact visual overlap is one structural opportunity, owned by the
            # more specific certified phase producer.  A later exact dropped
            # interval still replaces it with the gate's stronger semantics.
            proposed = [
                slot for slot in proposed
                if not (slot.kind == "visual"
                        and slot.frame_lo == phase.frame_lo
                        and slot.frame_hi == phase.frame_hi)
            ]
            proposed.append(TransitionSlot(
                "phase", phase.frame_lo, phase.frame_hi,
                parent_frame=phase.parent_frame,
                period_lo=phase.period_lo,
                period_hi=phase.period_hi,
                phase_index=phase.phase_index,
                phase_count=phase.phase_count,
            ))
        slots = list(baseline_slots) if phase_conflict else proposed
    dropped = sorted({(int(row["lo"]), int(row["hi"]))
                      for row in (dropped_intervals or ())
                      if int(row["lo"]) <= int(row["hi"])
                      and start < int(row.get("frame", row["hi"])) <= end})

    def add_dropped(source):
        output = list(source)
        phase_conflict = False
        for lo, hi in dropped:
            if any(lo <= frame <= hi for frame in motion):
                continue
            phase_conflict = phase_conflict or any(
                slot.kind == "phase"
                and not (slot.frame_hi < lo or hi < slot.frame_lo)
                for slot in output)
            # Exact overlap with an already-owned optional interval is one
            # action opportunity.  The final gate is the stronger ownership
            # source: it distinguishes re-grip/OM transition from a physical
            # layer move.
            output = [slot for slot in output
                      if not (slot.kind in ("visual", "phase")
                              and slot.frame_lo == lo
                              and slot.frame_hi == hi)]
            output.append(TransitionSlot("dropped", lo, hi))
        return output, phase_conflict

    slots, phase_drop_conflict = add_dropped(slots)
    if phase_drop_conflict:
        # Never keep a partial phase transaction when the stronger final-gate
        # source owns one of its intervals.
        slots, _unused = add_dropped(baseline_slots)
    # At equal terminal frames motion is processed first.  Exact overlap would
    # have coalesced, so this tie can only involve a visual interval ending next
    # to (not containing) a point under malformed external input; deterministic
    # ordering is preferable to an outcome-derived tolerance.
    slots.sort(key=lambda slot: (
        int(slot.frame_hi), 0 if slot.kind == "motion" else 1,
        int(slot.frame_lo)))
    return tuple(slots)


def monotone_pre_post_score(pre_scores, post_scores):
    """Maximum score for one latent pre→post boundary.

    ``boundary == j`` assigns rows ``[:j]`` to the pre-state and rows ``[j:]``
    to the post-state.  The transition may therefore precede every read or
    follow every read, but evidence can never alternate back to the pre-state.
    Ties keep the earliest boundary deterministically.
    """
    pre = np.asarray(pre_scores, dtype=np.float64)
    post = np.asarray(post_scores, dtype=np.float64)
    if pre.shape != post.shape or pre.ndim != 1:
        raise ValueError("pre/post score rows must be equal-length vectors")
    prefix = np.concatenate(([0.0], np.cumsum(pre)))
    suffix = np.concatenate((np.cumsum(post[::-1])[::-1], [0.0]))
    values = prefix + suffix
    boundary = int(np.argmax(values))
    return float(values[boundary]), boundary


def visual_slot_actions():
    """The fixed optional-visual action alphabet: SKIP + 18 SINGLEs."""
    return (None,) + tuple(range(18))


def phase_step_work(frontier_ois, om_nbrs):
    """Exact pre-dedup fanout for one certified phase transaction.

    A phase SKIP is the structural hypothesis that the unassigned motion was a
    whole-cube gesture.  It therefore emits stay plus every one-edge OM
    neighbor even when the gate has no separate rotation observation.  Each of
    the 18 SINGLE alternatives preserves its source OM.
    """
    ois = [int(oi) for oi in frontier_ois]
    skip_work = sum(
        len({int(oi)} | set(om_nbrs[int(oi)])) for oi in ois)
    return int(skip_work + len(BS.MOVES) * len(ois)), int(skip_work)


def phase_step_admissible(frontier_ois, om_nbrs, capacity=None):
    """Whether one complete phase SKIP/SINGLE fanout fits the shared bound."""
    bound = BS.SCORE_STATE_BOUND if capacity is None else int(capacity)
    work, skip_work = phase_step_work(frontier_ois, om_nbrs)
    return work <= bound, work, skip_work, bound


def hard_event_slot_identity(slot_source, window_slots, frame_start, frame_end):
    """Return the canonical non-phase structure owned by one window."""
    start = IBM.require_exact_integer(frame_start, "window frame start")
    end = IBM.require_exact_integer(frame_end, "window frame end")
    normalized_events = [
        IBM.require_exact_integer(frame, "hard event frame")
        for frame in slot_source
    ]
    hard_events = sorted(
        frame for frame in normalized_events if start < frame <= end)

    def slot_field(slot, field):
        value = getattr(slot, field)
        if value is None:
            return None
        return IBM.require_exact_integer(value, f"typed slot {field}")

    non_phase_slots = [
        {
            "kind": str(slot.kind),
            "frame_lo": slot_field(slot, "frame_lo"),
            "frame_hi": slot_field(slot, "frame_hi"),
            "parent_frame": slot_field(slot, "parent_frame"),
            "period_lo": slot_field(slot, "period_lo"),
            "period_hi": slot_field(slot, "period_hi"),
            "phase_index": slot_field(slot, "phase_index"),
            "phase_count": slot_field(slot, "phase_count"),
        }
        for slot in window_slots if slot.kind != "phase"
    ]
    return {
        "hard_events": hard_events,
        "non_phase_slots": non_phase_slots,
    }


def hard_event_slot_identity_digest(identity):
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phase_structure_comparisons(baseline, phased):
    """Compare every phase-invariant field before a result may be adopted."""
    comparisons = {}
    for field in (
            "hard_event_slot_digest", "L_est_eff", "L_lo", "L_hi"):
        baseline_value = baseline.get(field)
        phased_value = phased.get(field)
        comparisons[field] = {
            "baseline": baseline_value,
            "phased": phased_value,
            "match": baseline_value == phased_value,
        }
    comparisons["all_match"] = all(
        row["match"] for name, row in comparisons.items()
        if name != "all_match")
    return comparisons


def gate_drop_step_work(frontier_ois, om_nbrs=None, rotation_steps=0):
    """Exact pre-dedup work for one typed dropped-event action slot.

    Every SINGLE preserves the held orientation, contributing 18 outputs per
    source frontier row.  SKIP retains the legacy re-grip semantics and may
    reach every OM within the slot's actual rotation steps.  Counting that
    reachable set closes the old ``frontier * 19`` underestimate (even one OM
    neighbor makes the real fanout 20, not 19).  Reads may remove lanes and
    state/OM dedup may merge them later, so this is conservative but never an
    underestimate of the resident pre-dedup transaction.
    """
    if isinstance(frontier_ois, (int, np.integer)):
        ois = [0] * max(0, int(frontier_ois))
    else:
        ois = [int(oi) for oi in frontier_ois]
    steps = max(0, int(rotation_steps))
    if om_nbrs is None or steps == 0:
        skip_work = len(ois)
    else:
        skip_work = sum(len(om_reachable(
            om_nbrs, {int(oi)}, steps) or {int(oi)}) for oi in ois)
    return int(skip_work + len(BS.MOVES) * len(ois)), int(skip_work)


def gate_drop_step_admissible(frontier_ois, om_nbrs=None, rotation_steps=0,
                              capacity=None):
    """Whether one typed dropped-event transaction fits the state bound."""
    bound = BS.SCORE_STATE_BOUND if capacity is None else int(capacity)
    work, _skip_work = gate_drop_step_work(
        frontier_ois, om_nbrs, rotation_steps)
    return work <= bound, work, bound


def gate_drop_capped_step_work(frontier_oi_uses, om_nbrs=None,
                               rotation_steps=0, single_cap=1):
    """Exact work for one dropped slot under a transaction-wide SINGLE cap.

    ``frontier_oi_uses`` contains ``(orientation_index, singles_used)`` rows.
    Every row always retains the legacy SKIP/re-grip path, including its exact
    reachable-OM fanout.  Only a row below ``single_cap`` may emit the 18 layer
    SINGLE alternatives.  The cap is carried across *all* dropped gaps, so the
    work is linear in gaps rather than the old independent ``19**n`` lattice.
    """
    rows = [(int(oi), max(0, int(used)))
            for oi, used in frontier_oi_uses]
    steps = max(0, int(rotation_steps))
    cap = max(0, int(single_cap))
    if om_nbrs is None or steps == 0:
        skip_work = len(rows)
    else:
        skip_work = sum(len(om_reachable(
            om_nbrs, {oi}, steps) or {oi}) for oi, _used in rows)
    single_sources = sum(used < cap for _oi, used in rows)
    return (int(skip_work + len(BS.MOVES) * single_sources),
            int(skip_work))


def gate_drop_capped_step_admissible(frontier_oi_uses, om_nbrs=None,
                                     rotation_steps=0, single_cap=1,
                                     capacity=None):
    """Whether one globally capped dropped-slot step fits the state bound."""
    bound = BS.SCORE_STATE_BOUND if capacity is None else int(capacity)
    work, _skip_work = gate_drop_capped_step_work(
        frontier_oi_uses, om_nbrs, rotation_steps, single_cap)
    return work <= bound, work, bound


def optional_action_allows_om_transition(slot_kind, action):
    """Physical OM semantics for one typed optional action.

    An eligible alignment-veto proposal is either a re-grip (SKIP), which may
    stay or traverse the existing one-gesture OM neighbor edge, or a layer
    MOVE, which preserves physical orientation.  Alignment-owned visual
    intervals retain their historical behavior.
    """
    return action is None or str(slot_kind) not in ("dropped", "phase")

# SLACK provenance ("burst slack = movegt-corpus count-error quantiles,
# documented in-code"): on the only measured count-error datum for the
# event_frames burst source, summed event_frames over the audited gap
# = 13 detected vs 13 GT
# moves, i.e. COUNT-EXACT (error 0) on the one measured window; no per-window
# error-quantile corpus exists. SLACK = 1 therefore covers single-event
# boundary attribution at window edges (scrub windows are (rest, rest]
# envelopes, not the audited gap) on top of the measured zero count error.
# Data-derived ceiling, not a decode-outcome knob.
SLACK = 1


def _slot_slack_caps(n_slots, L_lo, L_hi):
    """Derived SKIP/DOUBLE caps spanning the declared count interval.

    When ``n_slots`` itself lies inside ``[L_lo, L_hi]``, every cap-respecting
    path is count-valid.  A producer-owned alternate stream may instead lie
    outside that interval; pair these caps with
    :func:`_physical_count_feasible_mask` so only prefixes with a legal
    terminal completion survive.
    """
    n_slots = int(n_slots)
    return (max(0, n_slots - int(L_lo)),
            max(0, int(L_hi) - n_slots))


def _physical_count_feasible_mask(
        *, step, n_slots, nskip, nins, skip_cap, insert_cap, L_lo, L_hi):
    """Return rows whose remaining slots can finish inside ``[L_lo,L_hi]``.

    ``nskip``/``nins`` may be scalars, NumPy arrays, or Torch tensors.  The
    calculation stays on their existing backend.  It is pure count algebra:
    no score, move identity, frame, tag, or truth input participates.
    """
    step = int(step)
    n_slots = int(n_slots)
    skip_cap = int(skip_cap)
    insert_cap = int(insert_cap)
    L_lo = int(L_lo)
    L_hi = int(L_hi)
    if not 0 <= step <= n_slots:
        raise ValueError("physical count step is outside the slot stream")
    if min(skip_cap, insert_cap, L_lo) < 0 or L_hi < L_lo:
        raise ValueError("invalid physical count interval or slack caps")

    remaining = n_slots - step
    current = step - nskip + nins
    skip_need = remaining - (skip_cap - nskip)
    insert_room = insert_cap - nins
    if hasattr(skip_need, "clamp"):
        skip_need = skip_need.clamp(min=0)
        insert_room = insert_room.clamp(min=0, max=remaining)
    else:
        skip_need = np.maximum(skip_need, 0)
        insert_room = np.clip(insert_room, 0, remaining)
    min_final = current + skip_need
    max_final = current + remaining + insert_room
    valid_usage = (
        (nskip >= 0) & (nskip <= skip_cap)
        & (nins >= 0) & (nins <= insert_cap)
        & (nskip + nins <= step))
    return valid_usage & (min_final <= L_hi) & (max_final >= L_lo)


def _canonical_count_slot_caps(n_slots, L_lo, L_hi):
    """Complete SKIP/DOUBLE caps for an HTM-counted physical slot stream.

    One non-SKIP slot can contribute at most two canonical actions (DOUBLE),
    so a terminal history with canonical length at least ``L_lo`` cannot use
    more than ``n-ceil(L_lo/2)`` SKIPs.  Conversely every DOUBLE contributes
    two canonical actions and cannot merge, so a history no longer than
    ``L_hi`` cannot use more than ``floor(L_hi/2)`` DOUBLE slots.  These are
    necessary maxima, not tuning knobs; the prefix feasibility mask below
    removes cap-respecting combinations that still cannot finish in-band.
    """
    n_slots = int(n_slots)
    L_lo = int(L_lo)
    L_hi = int(L_hi)
    if n_slots < 0 or L_lo < 0 or L_hi < L_lo:
        raise ValueError("invalid canonical count interval or slot count")
    return (
        max(0, n_slots - ((L_lo + 1) // 2)),
        min(n_slots, L_hi // 2),
    )


def _canonical_count_feasible_mask(
        *, step, n_slots, search_len, merge_pending, nskip, nins,
        skip_cap, insert_cap, L_lo, L_hi):
    """Return prefixes with a reachable terminal HTM count in the interval.

    ``search_len`` is the monotonically nondecreasing canonical HTM length;
    the physical ``word``/timing remains untouched.  A pending same-quarter
    SINGLE can make the next SINGLE contribute zero by completing a half-turn.
    With ``t`` future slots that cannot all be skipped, the exact minimum
    extra count is ``floor(t/2)`` with that pending quarter and ``ceil(t/2)``
    otherwise.  The exact maximum uses every remaining slot and as many
    remaining DOUBLE allocations as possible.

    All row-valued inputs may be NumPy arrays or Torch tensors.  Arithmetic
    remains on their existing backend and contains no score/frame/truth input.
    """
    step = int(step)
    n_slots = int(n_slots)
    skip_cap = int(skip_cap)
    insert_cap = int(insert_cap)
    L_lo = int(L_lo)
    L_hi = int(L_hi)
    if not 0 <= step <= n_slots:
        raise ValueError("canonical count step is outside the slot stream")
    if min(skip_cap, insert_cap, L_lo) < 0 or L_hi < L_lo:
        raise ValueError("invalid canonical count interval or slack caps")

    remaining = n_slots - step
    skip_room = skip_cap - nskip
    insert_room = insert_cap - nins
    if hasattr(skip_room, "clamp"):
        skip_room = skip_room.clamp(min=0, max=remaining)
        insert_room = insert_room.clamp(min=0, max=remaining)
        pending_i = merge_pending.to(dtype=search_len.dtype)
    else:
        skip_room = np.clip(skip_room, 0, remaining)
        insert_room = np.clip(insert_room, 0, remaining)
        pending_i = np.asarray(merge_pending, dtype=np.int64)
    mandatory_singles = remaining - skip_room
    min_add = (mandatory_singles + (1 - pending_i)) // 2
    max_add = remaining + insert_room
    min_final = search_len + min_add
    max_final = search_len + max_add
    valid_usage = (
        (nskip >= 0) & (nskip <= skip_cap)
        & (nins >= 0) & (nins <= insert_cap)
        & (nskip + nins <= step)
        & (search_len >= 0))
    return valid_usage & (min_final <= L_hi) & (max_final >= L_lo)


# ---- CUBED_SCRUB_SLOT_MERGE (L-currency bug; default OFF) ------------------
# `L_est`/`[L_lo,L_hi]` are burst-currency (SLACK/`_slot_slack_caps` above,
# UNTOUCHED by this section).  The search alphabet is HTM: a same-face 180deg
# is one `*2` token, while the detector can expose that turn as two successive
# physical quarter-action slots.  The CUDA controller must therefore carry
# TWO representations without ever rewriting one into the other:
#
# * `word`/`timed` are the unreduced PHYSICAL action program.  Every NN lane,
#   truth probe, trajectory trace, endpoint replay, and emitted move keeps it.
# * `search_len` plus merge provenance are the parallel HTM accounting used
#   only for incumbent-prefix matching and the slot-vs-token audit.
#
# A merge is legal only across adjacent physical hard slots; all observations
# between them remain in the physical timeline.  The existing SKIP/DOUBLE
# budgets remain in burst currency, and a merge changes no action or score.
# Flag absent/empty leaves the historical payload, keys, and controller calls
# literally untouched.
SLOT_MERGE_ENV = "CUBED_SCRUB_SLOT_MERGE"


def _slot_merge_active():
    """True iff CUBED_SCRUB_SLOT_MERGE is set truthy (default OFF)."""
    return bool(os.environ.get(SLOT_MERGE_ENV))


def _slot_merge_adjacency_audit(bursts, read_frames, rotation_frames):
    """Describe adjacent hard-slot pairs and intervening observations.

    The canonical search spelling may combine two adjacent equal quarters even
    when reads or rotations lie between them because the device controller
    never collapses the physical program: it still applies action one, scores
    every intermediate observation under that state/OM, then applies action
    two.  Only ``search_len`` and incumbent-prefix bookkeeping see the `*2`.
    Reads use ``[a,b)`` and rotations ``(a,b]`` here solely to audit which
    intermediate observations were preserved.  Tempo is deliberately absent.

    Every adjacent hard-slot pair is therefore structurally eligible; face and
    direction legality remain candidate-specific during expansion."""
    bursts = [int(f) for f in bursts]
    reads = sorted({int(f) for f in read_frames})
    rotations = sorted({int(f) for f in rotation_frames})
    rows = []
    for i in range(len(bursts) - 1):
        a, b = bursts[i], bursts[i + 1]
        reads_between = [f for f in reads if a <= f < b]
        rotations_between = [f for f in rotations if a < f <= b]
        rows.append(dict(
            slots=[i, i + 1], frames=[a, b], gap_frames=b - a,
            n_reads_between=len(reads_between),
            n_rotations_between=len(rotations_between),
            intermediate_observations_preserved=True,
            eligible=True))
    return rows


def _slot_merge_canonicalize_program(word, timing, merge_mask,
                                     eligible_adjacencies):
    """Return the HTM search spelling for one physical device program.

    ``word`` and ``timing`` remain the authoritative physical action program
    consumed by every NN, trajectory scorer, and final emission path.  A set
    bit at physical position ``i`` says that actions ``i-1,i`` were admitted
    by the device controller as one canonical half-turn search token.  This
    helper validates that claim against the window's physical-slot receipt and
    constructs the parallel HTM spelling only for reporting and incumbent-
    prefix accounting.

    The explicit mask matters: timing alone cannot distinguish a preceding
    ordinary SINGLE from the second action of a same-frame DOUBLE, and those
    programs have different merge legality."""
    physical = tuple(int(move) for move in word)
    frames = tuple(int(frame) for frame in timing)
    mask = int(merge_mask)
    if len(physical) != len(frames):
        raise ValueError("slot-merge physical word/timing length mismatch")
    if mask < 0 or mask >> len(physical):
        raise ValueError("slot-merge mask is outside the physical word")
    eligible = {
        (int(row["frames"][0]), int(row["frames"][1])): row
        for row in eligible_adjacencies if row.get("eligible")
    }
    search, search_timing, merges = [], [], []
    i = 0
    while i < len(physical):
        if mask & (1 << i):
            raise ValueError("slot-merge mask has an orphan second action")
        second = i + 1
        if second < len(physical) and mask & (1 << second):
            first_move, second_move = physical[i], physical[second]
            pair = (frames[i], frames[second])
            row = eligible.get(pair)
            if (row is None or first_move != second_move
                    or first_move % 3 == 2 or frames[i] == frames[second]):
                raise ValueError(
                    "slot-merge mask does not identify an eligible quarter pair")
            half = first_move - first_move % 3 + 2
            word_index = len(search)
            search.append(half)
            search_timing.append(frames[second])
            merges.append({
                "slots": [int(value) for value in row["slots"]],
                "frames": [frames[i], frames[second]],
                "word_index": word_index,
                "quarter": BS.MOVES[first_move],
                "token": BS.MOVES[half],
                "physical_positions": [i, second],
            })
            i += 2
            continue
        search.append(physical[i])
        search_timing.append(frames[i])
        i += 1
    return tuple(search), tuple(search_timing), merges


def _slot_merge_payload_identity(payload, torch):
    """Future-complete dedup key for the parallel physical/search payload.

    Non-merge rows deliberately retain the historical state/OM identity.
    Once a merge exists, both the physical actions and their timings are
    authoritative: the same actions assigned to different hard slots receive
    different NN/read evidence and must remain separate through every dedup.
    Unused ``-1`` payload cells map to zero, matching the move encoding.
    """
    if "search_len" not in payload:
        return None
    merged = payload["merge_count"] > 0
    mask = merged.reshape(-1, 1)
    physical = torch.where(
        mask,
        payload["word"].to(torch.int64) + 1,
        torch.zeros_like(payload["word"], dtype=torch.int64),
    )
    timing = torch.where(
        mask,
        payload["timed"].to(torch.int64) + 1,
        torch.zeros_like(payload["timed"], dtype=torch.int64),
    )
    return torch.cat((
        payload["search_len"].reshape(-1, 1),
        payload["merge_last"].reshape(-1, 1),
        payload["merge_reentry"].to(torch.int64).reshape(-1, 1),
        payload["merge_mask"].reshape(-1, 1),
        physical,
        timing,
    ), dim=1)


def _terminal_program_payload_identity(payload, torch):
    """Fixed-width physical-history identity for the V18 terminal plane.

    The ordinary/fused controller keeps its historical state identity.  This
    key is used only by the complete-band V18 survivor plane, whose downstream
    scorer consumes the retained physical word and timing.  A fixed zero slot-
    merge suffix keeps the schema and width identical when merging is inactive.
    """
    required = {"word", "timed", "word_len", "timed_len"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            "terminal physical-history payload is missing "
            + ", ".join(missing)
        )
    word = payload["word"]
    timed = payload["timed"]
    word_len = payload["word_len"]
    timed_len = payload["timed_len"]
    if (
        word.ndim != 2
        or timed.shape != word.shape
        or word_len.shape != (word.shape[0],)
        or timed_len.shape != (word.shape[0],)
        or word.dtype != torch.int8
        or timed.dtype != torch.int64
        or word_len.dtype != torch.int64
        or timed_len.dtype != torch.int64
        or any(
            value.device != word.device
            for value in (timed, word_len, timed_len)
        )
    ):
        raise ValueError(
            "terminal physical history must be typed word/timing tensors "
            "with shapes [N,W], [N,W], [N], [N] on one device"
        )
    base = torch.cat((
        word_len.reshape(-1, 1),
        word.to(torch.int64) + 1,
        timed_len.reshape(-1, 1),
        timed + 1,
    ), dim=1)
    merge = _slot_merge_payload_identity(payload, torch)
    if merge is None:
        merge = torch.zeros(
            (word.shape[0], 4 + 2 * word.shape[1]),
            dtype=torch.int64,
            device=word.device,
        )
    return torch.cat((base, merge), dim=1)


def _slot_has_endpoint_evidence(slot_frame, read_frames):
    """Whether a same-or-later sticker read can witness this transition."""
    return any(int(frame) >= int(slot_frame) for frame in read_frames)


def _cap_stateful_prefix_band(kept, *, endpoint_chunk, beam_k):
    """Keep the endpoint 2σ band whole; otherwise retain K states per OM."""
    if endpoint_chunk:
        return list(kept)
    strata = {}
    for kv in kept:
        strata.setdefault(kv[0][1], []).append(kv)
    return [kv for oi in sorted(strata) for kv in strata[oi][:int(beam_k)]]


def _longest_exact_beam_checkpoint(checkpoint_family, key_for_step,
                                   max_step):
    """Return the farthest frontier whose complete prefix key still matches.

    This is deliberately a lookup-only helper.  Callers own the key contract;
    a missing field therefore fails closed to a cold beam rather than accepting
    a partial match.  Keeping lookup separate also lets the CPU contract tests
    compare cached execution against the literal cold implementation without a
    production runtime switch.
    """
    for step in range(int(max_step), -1, -1):
        checkpoint = checkpoint_family.get(key_for_step(step))
        if checkpoint is not None:
            return step, checkpoint
    return -1, None


def _advance_device_beam_window_cache(
        checkpoint_cache, read_run_cache, runtime, a_span):
    """Scope resident CUDA retry artifacts to one monotone window start.

    Scrub never moves its committed cursor backwards.  Once ``a_span``
    changes, neither a retained frontier nor a compiled chronological read run
    from the prior start can participate in a future hypothesis.  Clearing the
    two caches is therefore an exact lifetime bound, not candidate pruning.
    Both control and dense planes call this helper, so a same-cursor plane
    switch retains each plane's independently namespaced artifacts.
    """
    a_span = int(a_span)
    current = runtime.get("a_span")
    if current == a_span:
        return False
    if current is not None:
        runtime["evictions"] = int(runtime.get("evictions", 0)) + 1
    checkpoint_cache.clear()
    read_run_cache.clear()
    runtime["a_span"] = a_span
    return True


def _dense_beam_checkpoint_key(step, *, skip_cap, insert_cap, slots,
                               chunk_signatures, witness_bits,
                               required_word):
    """Complete future-sensitive key for one retained dense-beam frontier.

    A later extension may append evidence, change whether an earlier action has
    an endpoint witness, or extend the incumbent word far enough that a prior
    DOUBLE lane becomes its prefix.  All three affect retention.  Only the
    chronological prefix that has actually executed belongs in this key;
    changes strictly after it remain reusable by design.
    """
    step = int(step)
    if step < 0 or step >= len(chunk_signatures):
        raise IndexError("dense checkpoint step lacks a chronological chunk")
    if step == 0:
        return (0, chunk_signatures[0])
    max_word_len = step + min(step, int(insert_cap))
    required_signature = (
        tuple(required_word[:max_word_len]),
        min(len(required_word), max_word_len))
    return (
        step, int(skip_cap), int(insert_cap),
        tuple((str(slot.kind), int(slot.frame_lo), int(slot.frame_hi))
              for slot in slots[:step]),
        tuple(chunk_signatures[:step + 1]),
        tuple(witness_bits[:step]), required_signature)


def _safe_exact_rescue_depth(max_depth):
    """Largest canonical word ball bounded by SCORE_STATE_BOUND paths."""
    total, width, depth = 1, 1, 0
    for d in range(1, int(max_depth) + 1):
        width = 18 if d == 1 else width * 15  # no adjacent same-face words
        if total + width > BS.SCORE_STATE_BOUND:
            break
        total += width
        depth = d
    return depth

# M2 FUSION: mid-motion static-cell
# reads carry a per-cell STABILITY confidence; a mid-motion segment is weighted by
# max(stability_conf, MM_CONF_FLOOR) L1-normalized -- the exact CONF_W formula
# (trellis_tracker.py:516 np.maximum(conf, 0.15); 0.15 is also the established
# trust-soft floor). Reused, not a new tuned constant.
# Applied ONLY to mid-motion segments so still-read scoring stays identical.
MM_CONF_FLOOR = 0.15


def _temporal_timing_sets(record):
    """Return normalized final-frontier timings keyed by terminal OM.

    ``timings`` is the original one-representative compatibility field.
    ``timing_sets`` extends it with every equal-best timing tuple that is still
    represented in the bounded final beam frontier.  It deliberately does not
    claim to recover alternatives discarded by an earlier state/OM dedup.
    """
    if not isinstance(record, dict):
        raise TypeError("temporal provenance must be a mapping")
    raw = record.get("timing_sets")
    if raw is None:
        raw = {oi: (timing,) for oi, timing
               in (record.get("timings") or {}).items()}
    if not isinstance(raw, dict):
        raise TypeError("temporal timing_sets must be a mapping")
    out = {}
    for raw_oi, values in raw.items():
        oi = int(raw_oi)
        if not isinstance(values, (list, tuple)):
            raise TypeError("temporal timing alternatives must be a sequence")
        normalized = set()
        for timing in values:
            if not isinstance(timing, (list, tuple)):
                raise TypeError("one temporal timing must be a sequence")
            normalized.add(tuple(int(frame) for frame in timing))
        if normalized:
            out[oi] = tuple(sorted(normalized))
    return out


def _record_temporal_frontier_path(
        record, oi, score, timing, *, retain_all_timings=False):
    """Retain bounded final-frontier timing provenance.

    The ordinary prefix contract keeps only equal-best timing alternatives for
    one ``(word, final_om)``.  A terminal complete-band survivor plane must keep
    every distinct bounded timing, because its learned prefix score is not the
    final comparison currency; the common exact scorer decides among them.
    ``om_scores`` and ``timings`` remain the best learned-score representative
    for compatibility in both modes.
    """
    oi = int(oi)
    score = float(score)
    timing = tuple(int(frame) for frame in timing)
    old = float(record["om_scores"][oi])
    retained = set(record["timing_sets"].get(oi, ()))
    representative = record["timings"].get(oi)
    if representative is not None:
        retained.add(tuple(representative))
    if score > old:
        record["om_scores"][oi] = score
        record["timings"][oi] = timing
        record["timing_sets"][oi] = (
            tuple(sorted(retained | {timing}))
            if retain_all_timings else (timing,))
        record["ambiguous_ois"].discard(oi)
    elif score == old:
        retained.add(timing)
        record["timing_sets"][oi] = tuple(sorted(retained))
        if representative is None:
            record["timings"][oi] = timing
        if len(retained) > 1:
            record["ambiguous_ois"].add(oi)
    elif retain_all_timings:
        retained.add(timing)
        record["timing_sets"][oi] = tuple(sorted(retained))
    if (retain_all_timings
            and len(record["timing_sets"].get(oi, ())) > 1):
        record["ambiguous_ois"].add(oi)


def _tag_prefix_temporal_record(record, sources, *, control_record=None):
    """Copy provenance while keeping prefix likelihood out of timing union."""
    tagged = dict(record)
    tagged["om_scores"] = np.asarray(record["om_scores"], float).copy()
    tagged["timings"] = {
        int(oi): tuple(int(frame) for frame in timing)
        for oi, timing in (record.get("timings") or {}).items()
    }
    tagged["timing_sets"] = _temporal_timing_sets(record)
    tagged["ambiguous_ois"] = set(record.get("ambiguous_ois") or ())
    tagged["_prefix_plane_sources"] = tuple(sorted(set(sources)))
    tagged["_control_prefit"] = control_record
    return tagged


def _union_prefix_plane_survivors(control_words, control_temporal,
                                  dense_words, dense_temporal, *,
                                  control_source="control",
                                  dense_source="dense"):
    """Union two independently pruned prefix planes by source identity.

    Candidate/observer metadata retains source identity: a shared word keeps
    the already-finalized control value and only a dense-only word gets dense
    metadata.  Timing alternatives are scoring provenance, however, so a shared
    word retains the union of control- and dense-frontier timings.  Prefix
    likelihoods are never compared or combined; final evidence scores every
    retained timing in one common program.

    Candidate values carry non-likelihood observer metadata (currently the
    centre-motion table attribution).  The same source-identity rule preserves
    control metadata for shared words and dense metadata for rescued words.
    """
    control_keys = set(control_words)
    dense_keys = set(dense_words)
    missing_control = control_keys - set(control_temporal)
    missing_dense = (dense_keys - control_keys) - set(dense_temporal)
    if missing_control or missing_dense:
        raise ValueError(
            "prefix survivor lacks typed temporal provenance: "
            f"control={len(missing_control)} dense={len(missing_dense)}")

    union_words = dict(control_words)
    control_source = str(control_source)
    dense_source = str(dense_source)
    if (not control_source or not dense_source
            or control_source == dense_source):
        raise ValueError("prefix plane source labels must be distinct")
    union_temporal = {
        word: _tag_prefix_temporal_record(
            control_temporal[word], (control_source,),
            control_record=control_temporal[word])
        for word in control_words
    }
    rescued = dense_keys - control_keys
    for word in dense_words:
        if word in control_keys:
            record = union_temporal[word]
            merged = _temporal_timing_sets(record)
            for oi, alternatives in _temporal_timing_sets(
                    dense_temporal[word]).items():
                merged[oi] = tuple(sorted(
                    set(merged.get(oi, ())) | set(alternatives)))
            record["timing_sets"] = merged
            for oi, alternatives in merged.items():
                record["timings"].setdefault(oi, alternatives[0])
                if len(alternatives) > 1:
                    record["ambiguous_ois"].add(int(oi))
            record["_prefix_plane_sources"] = tuple(sorted(
                (control_source, dense_source)))
            continue
        if word not in rescued:
            continue
        union_words[word] = dense_words[word]
        union_temporal[word] = _tag_prefix_temporal_record(
            dense_temporal[word], (dense_source,), control_record=None)
    timing_records = sum(
        len(alternatives)
        for record in union_temporal.values()
        for alternatives in _temporal_timing_sets(record).values())
    if timing_records > int(BS.SCORE_STATE_BOUND):
        raise RuntimeError(
            f"prefix timing records {timing_records} > "
            f"{int(BS.SCORE_STATE_BOUND)}")
    return union_words, union_temporal, {
        "control_keys": control_keys,
        "dense_keys": dense_keys,
        "union_keys": set(union_words),
        "rescued_keys": rescued,
        "timing_records": int(timing_records),
    }


def _dense_prefix_union_requires_control_rollback(
        seqs, candidate_prefit_om, control_candidate_metadata):
    """Whether a failed dense final score must restore control candidates.

    Normal raw-event windows carry explicit control/dense timing provenance.
    Cluster-fallback windows intentionally omit exact timing provenance, so a
    dense-only candidate must also be detected by set difference against the
    frozen control candidate map. Either signal makes the union one fail-closed
    transaction.
    """
    control_keys = set(control_candidate_metadata or {})
    has_dense_only_candidate = any(
        tuple(seq) not in control_keys for seq in seqs)
    has_dense_timing = any(
        "dense" in _prefix_plane_sources(record)
        for record in (candidate_prefit_om or {}).values())
    return bool(has_dense_only_candidate or has_dense_timing)


def _union_gate_tier_results(tier1, tier2, capacity=None):
    """Union two complete gate-tier candidate sets in final-score currency.

    Both inputs have already passed through the same window's final evidence
    scorer.  The union therefore retains their candidate scores verbatim and
    rebuilds the ordinary end-state and normal-form rankings over the combined
    population.  Equal words/states remain distinct physical hypotheses (their
    timings and OM traces can differ); the existing rankings fold duplicates by
    ``max`` exactly as a single search does.  Scores are never summed.

    Candidate-indexed OM traces are remapped to the union indices.  Any shape,
    score-currency, incumbent, or provenance mismatch raises so the caller can
    fail closed to the original primary result.  The existing state bound is a
    rejection bound, never a truncation rule.
    """
    limit = int(BS.SCORE_STATE_BOUND if capacity is None else capacity)
    if limit <= 0:
        raise ValueError("gate tier union capacity must be positive")

    def normalized_allowed(value):
        return (None if value is None
                else frozenset(int(oi) for oi in value))

    def normalized_common(result, field):
        value = result.get(field)
        if field in ("evidence_spans", "committed_word", "committed_nf"):
            return None if value is None else tuple(value)
        if field == "allowed_ois":
            return normalized_allowed(value)
        return value

    common_fields = (
        "L_lo", "L_hi", "L_est_eff", "slack_eff", "end_pinned",
        "evidence_spans", "allowed_ois",
        "committed_word", "committed_nf", "committed_in_budget",
    )
    for field in common_fields:
        if normalized_common(tier1, field) != normalized_common(tier2, field):
            raise ValueError(
                f"gate tier score-currency mismatch: {field}")

    def plane(result, source):
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ValueError(f"{source} gate result is not complete")
        seqs = [tuple(int(mi) for mi in seq)
                for seq in (result.get("seqs") or ())]
        cols = list(result.get("cols") or ())
        scores = np.asarray(
            result.get("sc") if result.get("sc") is not None else (),
            dtype=float)
        states = np.asarray(result.get("smat"))
        n_cand = len(seqs)
        if (not n_cand or len(cols) != n_cand or scores.shape != (n_cand,)
                or states.ndim != 2 or not len(states)):
            raise ValueError(f"{source} malformed candidate plane")
        if np.isnan(scores).any():
            raise ValueError(f"{source} candidate scores contain NaN")
        if np.isposinf(scores).any():
            raise ValueError(
                f"{source} candidate scores contain positive infinity")
        normalized_cols = []
        for ci, (seq, col) in enumerate(zip(seqs, cols)):
            if any(mi < 0 or mi >= len(BS.MOVES) for mi in seq):
                raise ValueError(f"{source} candidate {ci} has invalid move")
            try:
                path = tuple(int(index) for index in col)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} candidate {ci} has malformed state path") \
                    from exc
            if (len(path) != len(seq) + 1 or not path
                    or min(path) < 0 or max(path) >= len(states)):
                raise ValueError(
                    f"{source} candidate {ci} has invalid state path")
            normalized_cols.append(path)

        raw_modes = result.get("candidate_score_modes")
        modes = None if raw_modes is None else tuple(str(mode)
                                                     for mode in raw_modes)
        if modes is not None and len(modes) != n_cand:
            raise ValueError(f"{source} score-mode provenance is incomplete")

        raw_om = result.get("candidate_om_scores")
        om_scores = None if raw_om is None else np.asarray(raw_om, dtype=float)
        if (om_scores is not None
                and (om_scores.ndim != 2 or om_scores.shape[0] != n_cand
                     or np.isnan(om_scores).any()
                     or np.isposinf(om_scores).any())):
            raise ValueError(f"{source} OM score provenance is malformed")
        n_om = None if om_scores is None else int(om_scores.shape[1])

        def indexed_provenance(field, *, ambiguity=False):
            raw = result.get(field) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"{source} {field} must be a mapping")
            normalized = {}
            for raw_key, raw_value in raw.items():
                if (not isinstance(raw_key, tuple) or len(raw_key) != 2):
                    raise ValueError(
                        f"{source} {field} has malformed candidate key")
                ci, oi = (int(raw_key[0]), int(raw_key[1]))
                if (ci < 0 or ci >= n_cand or oi < 0
                        or (n_om is not None and oi >= n_om)):
                    raise ValueError(
                        f"{source} {field} key is out of range")
                try:
                    values = tuple(int(value) for value in raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{source} {field} value is malformed") from exc
                if ambiguity:
                    if any(pos < 0 or pos >= len(seqs[ci]) for pos in values):
                        raise ValueError(
                            f"{source} {field} move index is out of range")
                    normalized[(ci, oi)] = frozenset(values)
                else:
                    if len(values) != len(seqs[ci]):
                        raise ValueError(
                            f"{source} {field} trace length is invalid")
                    if n_om is not None and any(
                            value < 0 or value >= n_om for value in values):
                        raise ValueError(
                            f"{source} {field} OM index is out of range")
                    normalized[(ci, oi)] = values
            return normalized

        traces = indexed_provenance("move_om_traces")
        ambiguities = indexed_provenance(
            "move_om_trace_ambiguities", ambiguity=True)
        ambiguous_positions = frozenset(
            int(pos) for pos in
            (result.get("ambiguous_move_positions") or ()))
        if any(pos < 0 for pos in ambiguous_positions):
            raise ValueError(
                f"{source} global move ambiguity is malformed")

        committed_i = result.get("committed_index")
        represented = bool(result.get("committed_represented", False))
        if committed_i is not None:
            committed_i = int(committed_i)
            if committed_i < 0 or committed_i >= n_cand:
                raise ValueError(f"{source} incumbent index is out of range")
        if represented != (committed_i is not None):
            raise ValueError(
                f"{source} incumbent representation is inconsistent")
        if committed_i is not None:
            raw_committed_word = result.get("committed_word")
            if raw_committed_word is None:
                raise ValueError(
                    f"{source} represented incumbent lacks a literal word")
            try:
                committed_seq = tuple(
                    BS.MOVES.index(str(token))
                    for token in raw_committed_word)
            except ValueError as exc:
                raise ValueError(
                    f"{source} incumbent word is not in the move alphabet") \
                    from exc
            if seqs[committed_i] != committed_seq:
                raise ValueError(
                    f"{source} incumbent index does not identify the "
                    "committed word")
            committed_nf = result.get("committed_nf")
            actual_nf = tuple(BS.normal_form(BS.tokens_of(committed_seq)))
            if (committed_nf is None
                    or tuple(str(token) for token in committed_nf)
                    != actual_nf):
                raise ValueError(
                    f"{source} incumbent normal form is inconsistent")
        return dict(
            source=source, seqs=seqs, cols=normalized_cols,
            states=states, scores=scores, modes=modes,
            om_scores=om_scores, traces=traces,
            ambiguities=ambiguities,
            ambiguous_positions=ambiguous_positions,
            committed_i=committed_i)

    planes = (plane(tier1, "tier1"), plane(tier2, "tier2"))
    if planes[0]["states"].shape[1:] != planes[1]["states"].shape[1:]:
        raise ValueError("gate tier state geometry mismatch")
    if planes[0]["states"].dtype != planes[1]["states"].dtype:
        raise ValueError("gate tier state dtype mismatch")
    if ((planes[0]["modes"] is None)
            != (planes[1]["modes"] is None)):
        raise ValueError("gate tier score-mode provenance mismatch")
    if ((planes[0]["om_scores"] is None)
            != (planes[1]["om_scores"] is None)):
        raise ValueError("gate tier OM provenance mismatch")
    if (planes[0]["om_scores"] is not None
            and planes[0]["om_scores"].shape[1]
            != planes[1]["om_scores"].shape[1]):
        raise ValueError("gate tier OM geometry mismatch")
    ambiguous_position_mismatch = (
        planes[0]["ambiguous_positions"]
        != planes[1]["ambiguous_positions"])
    ambiguous_positions = (
        planes[0]["ambiguous_positions"]
        | planes[1]["ambiguous_positions"])

    # Compact the two state matrices exactly.  Candidate trajectories retain
    # their full ordered paths; only byte-identical state rows share storage.
    state_index = {}
    union_states = []
    seqs, cols, scores, sources = [], [], [], []
    modes = [] if planes[0]["modes"] is not None else None
    om_rows = [] if planes[0]["om_scores"] is not None else None
    source_maps = {}
    for item in planes:
        index_map = []
        for ci, (seq, source_col) in enumerate(
                zip(item["seqs"], item["cols"])):
            union_col = []
            for source_state_i in source_col:
                state_row = item["states"][source_state_i]
                state_key = state_row.tobytes()
                union_state_i = state_index.get(state_key)
                if union_state_i is None:
                    union_state_i = len(union_states)
                    if union_state_i >= limit:
                        raise RuntimeError(
                            "gate tier state union exceeds state bound")
                    state_index[state_key] = union_state_i
                    union_states.append(np.asarray(state_row).copy())
                union_col.append(union_state_i)
            union_i = len(seqs)
            index_map.append(union_i)
            seqs.append(seq)
            cols.append(tuple(union_col))
            scores.append(float(item["scores"][ci]))
            sources.append(item["source"])
            if modes is not None:
                modes.append(item["modes"][ci])
            if om_rows is not None:
                om_rows.append(item["om_scores"][ci].copy())
        source_maps[item["source"]] = tuple(index_map)

    smat = np.stack(union_states)
    sc = np.asarray(scores, dtype=float)
    om_scores = None if om_rows is None else np.stack(om_rows)
    traces, trace_ambiguities = {}, {}
    for item in planes:
        index_map = source_maps[item["source"]]
        for (ci, oi), trace in item["traces"].items():
            traces[(index_map[ci], oi)] = trace
        for (ci, oi), positions in item["ambiguities"].items():
            trace_ambiguities[(index_map[ci], oi)] = positions

    by_state = {}
    for i, col in enumerate(cols):
        key = smat[col[-1]].tobytes()
        oi_row = None
        if om_scores is not None and np.isfinite(om_scores[i]).any():
            oi_row = om_scores[i] + (
                float(sc[i]) - float(np.max(om_scores[i])))
        group = by_state.get(key)
        if group is None:
            by_state[key] = dict(
                state=key, best_i=i, best_score=float(sc[i]),
                om_scores=(oi_row.copy() if oi_row is not None else None))
        elif sc[i] > group["best_score"]:
            old_om = group.get("om_scores")
            group.update(best_i=i, best_score=float(sc[i]))
            if oi_row is not None:
                group["om_scores"] = (oi_row.copy() if old_om is None
                                      else np.maximum(old_om, oi_row))
        elif oi_row is not None:
            old_om = group.get("om_scores")
            group["om_scores"] = (oi_row.copy() if old_om is None
                                  else np.maximum(old_om, oi_row))
    state_groups = sorted(
        by_state.values(), key=lambda group: -group["best_score"])
    word_classes = BS.class_ranking(seqs, sc)

    incumbent_candidates = []
    for item in planes:
        if item["committed_i"] is not None:
            incumbent_candidates.append(
                source_maps[item["source"]][item["committed_i"]])
    committed_i = (max(incumbent_candidates,
                       key=lambda index: float(sc[index]))
                   if incumbent_candidates else None)

    out = dict(tier2)
    out.update(
        seqs=seqs, cols=cols, smat=smat, sc=sc,
        state_groups=state_groups, word_classes=word_classes,
        candidate_om_scores=om_scores,
        move_om_traces=traces,
        move_om_trace_ambiguities=trace_ambiguities,
        # This marker is global and suppresses exact per-move OM publication.
        # A tier mismatch therefore takes the conservative union rather than
        # discarding valid cube-state candidates or laundering one tier's
        # narrower trace into an exact physical attribution.
        ambiguous_move_positions=set(ambiguous_positions),
        candidate_score_modes=modes,
        committed_index=committed_i,
        committed_represented=committed_i is not None,
        committed_comparable=False,
        incumbent_selected=False,
        n_candidates=len(seqs), n_end_states=len(state_groups),
        n_word_classes=len(word_classes),
        gate_drop_candidate_sources=tuple(sources),
        gate_drop_tier_union=dict(
            status="union", score_currency="final-window-evidence",
            duplicate_semantics="state-and-normal-form-max",
            tier1_candidates=len(planes[0]["seqs"]),
            tier2_candidates=len(planes[1]["seqs"]),
            union_candidates=len(seqs),
            union_states=len(union_states),
            ambiguous_move_positions_mismatch=bool(
                ambiguous_position_mismatch),
            ambiguous_move_positions=sorted(ambiguous_positions),
            tier1_index_range=[0, len(planes[0]["seqs"])],
            tier2_index_range=[len(planes[0]["seqs"]), len(seqs)],
        ))
    return out


# -------------------------------------------------------------- small utils
def _cluster_events(event_frames):
    """Cluster raw event frames into burst GROUPS at the valley of the
    stream's own log inter-event-gap distribution. This is derived per capture
    from the camera event stream without teacher data or fixed timing constants;
    intra-burst re-fires sit left of the valley and true
    inter-move gaps right of it). Returns cluster-median representative
    frames. Degenerate streams (few gaps / no interior valley) return the
    raw frames unchanged, making the caller's fallback inert."""
    ev = sorted(int(f) for f in event_frames)
    if len(ev) < 12:
        return ev
    gaps = np.diff(np.asarray(ev, float))
    gaps = gaps[gaps > 0]
    if len(gaps) < 10:
        return ev
    lg = np.log(gaps)
    hist, edges = np.histogram(lg, bins=max(8, 2 * int(len(lg) ** 0.5)))
    vi, vbest = None, None
    for i in range(1, len(hist) - 1):
        left, right = hist[:i].max(), hist[i + 1:].max()
        if left > hist[i] < right and (vbest is None or hist[i] < vbest):
            vi, vbest = i, hist[i]
    if vi is None:
        return ev
    thr = float(np.exp(edges[vi + 1]))
    reps, cur = [], [ev[0]]
    for a, b in zip(ev, ev[1:]):
        if b - a <= thr:
            cur.append(b)
        else:
            reps.append(int(np.median(cur)))
            cur = [b]
    reps.append(int(np.median(cur)))
    return reps


def _events_in(event_frames, f_lo, f_hi):
    """# motion-event frames in the half-open frame interval (f_lo, f_hi]."""
    return sum(1 for f in event_frames if f_lo < int(f) <= f_hi)


def _frame_to_span(fr, span_lo, span_hi):
    """Span index whose [f0, f1] frame range contains frame `fr`, or None (a
    frame in a between-span gap is dropped honestly). span_lo ascending (M1
    mid-motion row -> span attachment)."""
    if fr < 0 or not span_lo:
        return None
    i = bisect.bisect_right(span_lo, fr) - 1
    if 0 <= i < len(span_hi) and span_lo[i] <= fr <= span_hi[i]:
        return i
    return None


def _mm_apply_weight(seg, cellconf):
    """Set seg._wv = max(stability_conf, MM_CONF_FLOOR) L1-normalized over the
    segment's admitted cells -- the CONF_W formula (trellis_tracker.py:516) on the
    mid-motion STABILITY conf, applied ONLY to mid-motion segments so still-read
    scoring is round-3-identical. Faithful for AbsSegment (its _face_names/_pos
    provenance); a no-op for a fake seg lacking that provenance (it carries its
    own weighting). Turning-layer cells (conf 0) -> weight MM_CONF_FLOOR (floored,
    minimal); static cells (conf ~1) dominate."""
    fns = getattr(seg, "_face_names", None)
    pos = getattr(seg, "_pos", None)
    if fns is None or pos is None or not cellconf:
        return
    w = np.array([max(float((cellconf.get(f) or [1.0] * 9)[p]), MM_CONF_FLOOR)
                  for f, p in zip(fns, list(pos))], np.float64)
    if w.size and w.sum() > 0:
        seg._wv = w / w.sum()


# ---- om-continuity geometry (derived from cube geometry; no constant) ------
_FACE_VEC = {"U": (0, 1, 0), "D": (0, -1, 0), "F": (0, 0, 1),
             "B": (0, 0, -1), "R": (1, 0, 0), "L": (-1, 0, 0)}
_VEC_FACE = {v: k for k, v in _FACE_VEC.items()}
_FACE_CODE = {
    "U": "U", "UP": "U",
    "D": "D", "DOWN": "D",
    "F": "F", "FRONT": "F",
    "B": "B", "BACK": "B",
    "R": "R", "RIGHT": "R",
    "L": "L", "LEFT": "L",
}


def om_key_of(om):
    """(up, front) key for an orientation object (dict or tuple form)."""
    if isinstance(om, dict):
        return (om["up"], om["front"])
    return (om[0], om[1])


def nn_verif_orientations(orientations):
    """Bridge decoder om objects to nn_verif's (up, front) FACE-LETTER pairs.

    The decoder's orientation objects carry direction WORDS ("up"/"front"/
    "down"/... -- the trellis_tracker om dict values), while nn_verif.CLASS24
    is keyed by face LETTERS ("U"/"F"/...).  Without this bridge every consume
    window silently fell back to scoring all 24 orientations.
    Reuses the SAME om_key_of + _FACE_CODE normalization as om_adjacency
    (normalize_names=True) -- no new mapping table.  An entry that does not
    normalize is passed through unchanged so nn_verif's fail-safe all-24
    fallback applies exactly as before.
    """
    if orientations is None:
        return None
    bridged = []
    for om in orientations:
        try:
            u, f = om_key_of(om)
            u = _FACE_CODE.get(str(u).upper())
            f = _FACE_CODE.get(str(f).upper())
        except Exception:
            u = f = None
        bridged.append((u, f) if u is not None and f is not None else om)
    return bridged








def _strict_receipt_int(value):
    """Return an integer receipt scalar, or ``None`` when it is malformed.

    Receipt validation is deliberately stricter than ``int(value)``: bools,
    floats, strings, missing values, and ``None`` cannot silently cross an
    adoption gate.  NumPy integer scalars are accepted because CUDA/NumPy
    accounting receipts may materialize them before JSON serialization.
    """
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        return None
    return int(value)


def _partial_left_neutral_prefix_receipt(
        *, plane_role, physical_frame, neutral_physical_frames, owned_span,
        attempt_first_solve_frame, cuda_control_parent_rows,
        cuda_control_resident):
    """Build one pending receipt for the frozen partial-left exception.

    The learned plane has no value on this prefix: no zero tensor is minted and
    no V18 delta is applied.  The row is appended immediately for diagnostics,
    but it is not admissible until a later complete owned grid reseeds the two
    learned siblings from the then-current CUDA control frontier.
    """
    if plane_role not in _V18_SHADOW_CAPACITY_PLANE_ROLES:
        raise ValueError("partial-left neutral plane role is invalid")
    frame = _strict_receipt_int(physical_frame)
    attempt = _strict_receipt_int(attempt_first_solve_frame)
    parent_rows = _strict_receipt_int(cuda_control_parent_rows)
    if frame is None or frame < 1:
        raise ValueError("partial-left neutral physical frame is invalid")
    if attempt is None or attempt < 1:
        raise ValueError("partial-left neutral attempt frame is invalid")
    if (not isinstance(owned_span, (list, tuple))
            or len(owned_span) != 2):
        raise ValueError("partial-left neutral owned span is invalid")
    owned_lo = _strict_receipt_int(owned_span[0])
    owned_hi = _strict_receipt_int(owned_span[1])
    if (owned_lo is None or owned_hi is None
            or owned_hi - owned_lo + 1 != 16
            or owned_lo != ((frame - 1) // 16) * 16 + 1
            or not owned_lo <= frame <= owned_hi):
        raise ValueError("partial-left neutral owned span is invalid")
    if not owned_lo < attempt <= frame:
        raise ValueError(
            "partial-left neutral attempt does not prove partial-left")
    if not isinstance(neutral_physical_frames, (list, tuple)):
        raise ValueError("partial-left neutral physical frames are invalid")
    neutral_frames = [
        _strict_receipt_int(value) for value in neutral_physical_frames
    ]
    if (not neutral_frames or any(value is None for value in neutral_frames)
            or neutral_frames != sorted(set(neutral_frames))
            or neutral_frames[0] != frame
            or any(not attempt <= value <= owned_hi
                   for value in neutral_frames)):
        raise ValueError("partial-left neutral physical frames are invalid")
    if parent_rows is None or parent_rows <= 0:
        raise ValueError("partial-left neutral CUDA parent rows are invalid")
    if type(cuda_control_resident) is not bool or not cuda_control_resident:
        raise ValueError("partial-left neutral control frontier is not CUDA")
    return {
        "kind": _PARTIAL_LEFT_NEUTRAL_PREFIX_KIND,
        "capability": _PARTIAL_LEFT_NEUTRAL_PREFIX_CAPABILITY,
        "preregistration_sha256": (
            _PARTIAL_LEFT_NEUTRAL_PREFIX_PREREG_SHA256),
        "status": "pending-later-complete-grid-reseed",
        "reason": _PARTIAL_LEFT_NEUTRAL_PREFIX_REASON,
        "plane_role": str(plane_role),
        "physical_frame": int(frame),
        "neutral_physical_frames": [int(value) for value in neutral_frames],
        "owned_span": [int(owned_lo), int(owned_hi)],
        "attempt_first_solve_frame": int(attempt),
        "cuda_control_parent_rows": int(parent_rows),
        "control_execution_backend": "cuda-resident",
        "learned_delta_applied": False,
        "v18_provider_output_presented": False,
        "calibrated_final_v18_likelihood_claimed": False,
        "execution_fallback": False,
        "reseed_frame": None,
        "reseed_owned_span": None,
        "reseed_cuda_control_parent_rows": None,
        "reseed_cuda_learned_parent_rows": None,
        "reseed_execution_backend": None,
        "reseed_from_current_cuda_control_frontier": False,
    }


def _complete_partial_left_neutral_prefix_receipt(
        receipt, *, reseed_frame, reseed_owned_span,
        reseed_cuda_control_parent_rows, reseed_cuda_learned_parent_rows,
        cuda_reseed_resident):
    """Complete a pending row only after an exact CUDA frontier reseed."""
    if (not isinstance(receipt, dict)
            or receipt.get("kind") != _PARTIAL_LEFT_NEUTRAL_PREFIX_KIND
            or receipt.get("capability")
            != _PARTIAL_LEFT_NEUTRAL_PREFIX_CAPABILITY
            or receipt.get("status")
            != "pending-later-complete-grid-reseed"):
        raise ValueError("partial-left neutral receipt is not pending")
    frame = _strict_receipt_int(reseed_frame)
    control_rows = _strict_receipt_int(reseed_cuda_control_parent_rows)
    learned_rows = _strict_receipt_int(reseed_cuda_learned_parent_rows)
    if (not isinstance(reseed_owned_span, (list, tuple))
            or len(reseed_owned_span) != 2):
        raise ValueError("partial-left neutral reseed span is invalid")
    owned_lo = _strict_receipt_int(reseed_owned_span[0])
    owned_hi = _strict_receipt_int(reseed_owned_span[1])
    prefix_span = receipt.get("owned_span")
    prefix_hi = (
        _strict_receipt_int(prefix_span[1])
        if isinstance(prefix_span, list) and len(prefix_span) == 2 else None
    )
    attempt = _strict_receipt_int(
        receipt.get("attempt_first_solve_frame"))
    if (frame is None or owned_lo is None or owned_hi is None
            or prefix_hi is None or attempt is None
            or owned_hi - owned_lo + 1 != 16
            or owned_lo != ((frame - 1) // 16) * 16 + 1
            or not owned_lo <= frame <= owned_hi
            or owned_lo <= prefix_hi
            or attempt > owned_lo):
        raise ValueError("partial-left neutral reseed span is invalid")
    if control_rows is None or control_rows <= 0:
        raise ValueError("partial-left neutral reseed control rows are invalid")
    if learned_rows is None or learned_rows <= 0:
        raise ValueError("partial-left neutral reseed learned rows are invalid")
    if type(cuda_reseed_resident) is not bool or not cuda_reseed_resident:
        raise ValueError("partial-left neutral reseed is not CUDA")
    receipt.update(
        status="completed",
        reseed_frame=int(frame),
        reseed_owned_span=[int(owned_lo), int(owned_hi)],
        reseed_cuda_control_parent_rows=int(control_rows),
        reseed_cuda_learned_parent_rows=int(learned_rows),
        reseed_execution_backend="cuda-resident",
        reseed_from_current_cuda_control_frontier=True,
    )
    return receipt


def _partial_left_neutral_prefix_receipt_failures(
        receipts, expected_frames, *, endpoint_history_capture_active=False):
    """Validate the complete two-plane neutral-prefix/reseed transaction.

    Returns ``(failures, coverage)``.  Coverage supplies each plane's exact
    neutral and V18-scorable physical frames so the enclosing unowned-motion
    gate can require V13 everywhere while requiring V18/capacity only where a
    complete owned grid exists.
    """
    failures = []
    try:
        raw_frames = list(expected_frames)
    except TypeError:
        raw_frames = []
        failures.append("partial-left-neutral-expected-frames-invalid")
    frames = [_strict_receipt_int(frame) for frame in raw_frames]
    if (not frames or any(frame is None for frame in frames)
            or frames != sorted(set(frames))):
        if "partial-left-neutral-expected-frames-invalid" not in failures:
            failures.append("partial-left-neutral-expected-frames-invalid")
        frames = []
    coverage = {
        role: {
            "neutral_frames": [],
            "scorable_frames": [int(frame) for frame in frames],
            "reseed_frames": [],
        }
        for role in sorted(_V18_SHADOW_CAPACITY_PLANE_ROLES)
    }
    if not isinstance(receipts, (list, tuple)):
        return [
            *failures,
            "partial-left-neutral-receipts-container-invalid",
        ], coverage
    rows = list(receipts)
    if not rows:
        return failures, coverage
    if endpoint_history_capture_active:
        failures.append(
            "partial-left-neutral-endpoint-history-capture-active")
    if any(not isinstance(row, dict) for row in rows):
        failures.append("partial-left-neutral-receipt-row-invalid")
    if any(
            isinstance(row, dict)
            and row.get("plane_role")
            not in _V18_SHADOW_CAPACITY_PLANE_ROLES
            for row in rows):
        failures.append("partial-left-neutral-plane-role-invalid")

    for role in sorted(_V18_SHADOW_CAPACITY_PLANE_ROLES):
        lane_rows = [
            row for row in rows
            if isinstance(row, dict) and row.get("plane_role") == role
        ]
        if len(lane_rows) != 1:
            failures.append(
                f"{role}-partial-left-neutral-cardinality-invalid")
            continue
        row = lane_rows[0]
        constants = (
            (row.get("kind") == _PARTIAL_LEFT_NEUTRAL_PREFIX_KIND,
             "kind-invalid"),
            (row.get("capability")
             == _PARTIAL_LEFT_NEUTRAL_PREFIX_CAPABILITY,
             "capability-invalid"),
            (row.get("preregistration_sha256")
             == _PARTIAL_LEFT_NEUTRAL_PREFIX_PREREG_SHA256,
             "preregistration-invalid"),
            (row.get("status") == "completed", "status-incomplete"),
            (row.get("reason") == _PARTIAL_LEFT_NEUTRAL_PREFIX_REASON,
             "reason-invalid"),
            (row.get("control_execution_backend") == "cuda-resident",
             "control-not-cuda"),
            (row.get("learned_delta_applied") is False,
             "learned-delta-applied"),
            (row.get("v18_provider_output_presented") is False,
             "v18-output-presented"),
            (row.get("calibrated_final_v18_likelihood_claimed") is False,
             "calibrated-v18-claimed"),
            (row.get("execution_fallback") is False,
             "execution-fallback"),
            (row.get("reseed_execution_backend") == "cuda-resident",
             "reseed-not-cuda"),
            (row.get("reseed_from_current_cuda_control_frontier") is True,
             "reseed-not-current-control"),
        )
        failures.extend(
            f"{role}-partial-left-neutral-{reason}"
            for valid, reason in constants if not valid
        )

        physical_frame = _strict_receipt_int(row.get("physical_frame"))
        attempt = _strict_receipt_int(
            row.get("attempt_first_solve_frame"))
        control_rows = _strict_receipt_int(
            row.get("cuda_control_parent_rows"))
        reseed_frame = _strict_receipt_int(row.get("reseed_frame"))
        reseed_control_rows = _strict_receipt_int(
            row.get("reseed_cuda_control_parent_rows"))
        reseed_learned_rows = _strict_receipt_int(
            row.get("reseed_cuda_learned_parent_rows"))
        owned_span = row.get("owned_span")
        reseed_span = row.get("reseed_owned_span")
        neutral_raw = row.get("neutral_physical_frames")
        owned = (
            [_strict_receipt_int(value) for value in owned_span]
            if isinstance(owned_span, list) and len(owned_span) == 2 else None
        )
        reseed_owned = (
            [_strict_receipt_int(value) for value in reseed_span]
            if isinstance(reseed_span, list) and len(reseed_span) == 2 else None
        )
        neutral = (
            [_strict_receipt_int(value) for value in neutral_raw]
            if isinstance(neutral_raw, list) else None
        )
        if (physical_frame is None or physical_frame not in frames):
            failures.append(
                f"{role}-partial-left-neutral-physical-frame-invalid")
        if (control_rows is None or control_rows <= 0):
            failures.append(
                f"{role}-partial-left-neutral-control-rows-invalid")
        if (reseed_control_rows is None or reseed_control_rows <= 0
                or reseed_learned_rows is None or reseed_learned_rows <= 0):
            failures.append(
                f"{role}-partial-left-neutral-reseed-rows-invalid")
        owned_valid = bool(
            owned is not None and all(value is not None for value in owned)
            and physical_frame is not None
            and owned[1] - owned[0] + 1 == 16
            and owned[0] == ((physical_frame - 1) // 16) * 16 + 1
            and owned[0] <= physical_frame <= owned[1]
            and attempt is not None and owned[0] < attempt <= physical_frame
        )
        if not owned_valid:
            failures.append(
                f"{role}-partial-left-neutral-owned-span-invalid")
        neutral_valid = bool(
            neutral is not None and neutral
            and all(value is not None for value in neutral)
            and neutral == sorted(set(neutral))
            and physical_frame is not None and neutral[0] == physical_frame
            and owned_valid
            and neutral == [
                int(frame) for frame in frames
                if attempt <= frame <= owned[1]
            ]
            and frames[:len(neutral)] == neutral
        )
        if not neutral_valid:
            failures.append(
                f"{role}-partial-left-neutral-prefix-frames-invalid")
            continue
        scorable = [frame for frame in frames if frame not in set(neutral)]
        reseed_valid = bool(
            scorable and reseed_frame == scorable[0]
            and reseed_owned is not None
            and all(value is not None for value in reseed_owned)
            and reseed_owned[1] - reseed_owned[0] + 1 == 16
            and reseed_owned[0] == ((reseed_frame - 1) // 16) * 16 + 1
            and reseed_owned[0] <= reseed_frame <= reseed_owned[1]
            and reseed_owned[0] > owned[1]
            and attempt <= reseed_owned[0]
        )
        if not reseed_valid:
            failures.append(
                f"{role}-partial-left-neutral-reseed-invalid")
            continue
        coverage[role] = {
            "neutral_frames": [int(frame) for frame in neutral],
            "scorable_frames": [int(frame) for frame in scorable],
            "reseed_frames": [int(reseed_frame)],
        }

    lane_coverages = list(coverage.values())
    if rows and len(lane_coverages) == 2:
        if (lane_coverages[0]["neutral_frames"]
                != lane_coverages[1]["neutral_frames"]):
            failures.append("partial-left-neutral-plane-prefix-mismatch")
        if (lane_coverages[0]["reseed_frames"]
                != lane_coverages[1]["reseed_frames"]):
            failures.append("partial-left-neutral-plane-reseed-mismatch")
    return failures, coverage


def _receipt_int_or_gate_failure(mapping, key, failures, reason):
    """Parse one strict integer field and append a named gate refusal."""
    value = mapping.get(key) if isinstance(mapping, dict) else None
    parsed = _strict_receipt_int(value)
    if parsed is None:
        failures.append(str(reason))
    return parsed


def _receipt_int_range_or_gate_failure(mapping, key, failures, reason):
    """Parse one two-integer list receipt and fail closed by field name."""
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if not isinstance(value, list) or len(value) != 2:
        failures.append(str(reason))
        return None
    lo = _strict_receipt_int(value[0])
    hi = _strict_receipt_int(value[1])
    if lo is None or hi is None:
        failures.append(str(reason))
        return None
    return [lo, hi]


def _terminal_dedup_receipt_failures(union, sequential_payload_width):
    """Validate the frozen terminal-history receipt without coercion."""
    failures = []
    if not isinstance(union, dict):
        return ["terminal-dedup-receipt-invalid"]
    if union.get("primary_terminal_dedup_policy") != (
            _TERMINAL_DEDUP_POLICY_STATE_ONLY):
        failures.append("primary-terminal-dedup-policy-drift")
    if union.get("terminal_dedup_policy") != (
            _TERMINAL_DEDUP_POLICY_PHYSICAL_HISTORY):
        failures.append("terminal-dedup-policy-drift")
    if union.get("terminal_dedup_key_schema") != _TERMINAL_DEDUP_KEY_SCHEMA:
        failures.append("terminal-dedup-key-schema-drift")
    key_width = _strict_receipt_int(union.get("terminal_dedup_key_width"))
    if key_width is None or key_width <= 0:
        failures.append("terminal-dedup-key-width-invalid")
    payload_width = _strict_receipt_int(sequential_payload_width)
    if payload_width is None or payload_width <= 0:
        failures.append("terminal-dedup-payload-width-invalid")
    elif (key_width is not None
          and key_width != _terminal_program_dedup_key_width(payload_width)):
        failures.append("terminal-dedup-key-width-drift")
    return failures




def _consume_receipt_provider_applied(receipt, provider_name):
    """Whether one consume receipt proves a provider actually ran.

    A V13 abstention deliberately returns an all-zero array, so non-null score
    storage alone is not an activation receipt.  Finite output with no reason
    is live; V13's explicit partial-coverage ``ok_with_*`` status is also live.
    """
    if not isinstance(receipt, dict):
        return False
    providers = receipt.get("providers", ())
    if not isinstance(providers, (list, tuple)):
        return False
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if provider.get("name") != provider_name:
            continue
        pre_clamp_l1 = provider.get("pre_clamp_l1")
        try:
            pre_clamp_l1 = float(pre_clamp_l1)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(pre_clamp_l1):
            continue
        reason = provider.get("reason")
        if reason is None:
            return True
        if (provider_name == "v13_side"
                and str(reason).startswith("ok_with_")):
            return True
    return False


def _consume_receipt_provider_cuda(receipt, provider_name):
    """Whether a live provider receipt binds its score work to CUDA."""
    if not isinstance(receipt, dict):
        return False
    providers = receipt.get("providers", ())
    if not isinstance(providers, (list, tuple)):
        return False
    matches = [
        provider for provider in providers
        if isinstance(provider, dict) and provider.get("name") == provider_name
    ]
    # One provider name represents one score tensor.  Reject duplicates so an
    # applied row and an unrelated CUDA-stamped row cannot jointly satisfy the
    # receipt contract.
    if len(matches) != 1:
        return False
    provider = matches[0]
    try:
        pre_clamp_l1 = float(provider.get("pre_clamp_l1"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not np.isfinite(pre_clamp_l1):
        return False
    reason = provider.get("reason")
    applied = reason is None or (
        provider_name == "v13_side"
        and str(reason).startswith("ok_with_")
    )
    return bool(
        applied
        and isinstance(provider.get("meta"), dict)
        and provider["meta"].get("execution_backend") == "cuda-resident"
    )


def _v13_cuda_proposal_receipt_valid(
        receipt, provider, *, frame_lo, frame_hi, candidate_rows):
    """Bind a V13 proposal tensor to the exact CUDA provider transaction."""
    if not isinstance(receipt, dict):
        return False
    try:
        from detect import nn_v13_side_evidence as v13_side
    except Exception:
        return False
    covered_frames = receipt.get("covered_frames")
    if (isinstance(covered_frames, bool)
            or not isinstance(covered_frames, (int, np.integer))):
        return False
    expected_span = [int(frame_lo), int(frame_hi)]
    expected_rows = int(candidate_rows)
    meta = getattr(provider, "meta", None) or {}
    return bool(
        receipt.get("schema_version") == v13_side.SCHEMA_VERSION
        and receipt.get("provider") == "v13_side"
        and receipt.get("currency")
        == "calibrated-one-vs-rest-llr-h18"
        and receipt.get("execution_backend") == "cuda-resident"
        and receipt.get("frame_span") == expected_span
        and int(covered_frames) >= int(provider.struct_min_coverage)
        and int(covered_frames) <= expected_span[1] - expected_span[0]
        and receipt.get("candidate_rows") == expected_rows
        and receipt.get("seed_count") == int(provider.n_seeds)
        and tuple(receipt.get("seed_order") or ())
        == tuple(meta.get("seed_order") or ())
        and receipt.get("probability_floor") == v13_side.PROB_FLOOR
        and receipt.get("window_policy") == "covered-frame-mean-log"
        and receipt.get("bundle_sha256") == meta.get("self_sha256")
        and receipt.get("orientations_sha256")
        == v13_side.ORIENTATIONS_SHA256
        and receipt.get("one_vs_rest_consumed") is True
        and receipt.get("viscal_consumed") is False
    )


def _nn_exact_history_score_enabled(
        *, nn_final_active, nn_endpoint_active, history_capture_active):
    """Whether proposal providers must use the strict exact-history lane.

    Capture is a consumer of the same runtime-native provider features as the
    final endpoint scorer.  Treating it as a legacy observer would silently
    materialize V13 proposal evidence on the host before serializing a row
    advertised as CUDA exact-history evidence.
    """
    return bool(
        nn_final_active or nn_endpoint_active or history_capture_active
    )


def _nn_proposal_provider_cuda_required(
        *, exact_history_active, unowned_motion_actions_active):
    """Whether proposal providers must remain device-native and receipted.

    Exact-history consumers already forbid host provider fusion.  The v3
    unowned-motion transaction has the same requirement even when no final
    endpoint calibration is installed: V13 must use ``window_delta_device``
    and V18 must validate its device tensor before either score can enter a
    proposal lane.  This does not activate final learned scoring.
    """
    return bool(exact_history_active or unowned_motion_actions_active)


def _v13_proposal_delta(
        provider, *, frame_lo, frame_hi, candidate_ois, device,
        exact_history, torch_module):
    """Route exact-history V13 proposals directly through the CUDA API.

    The host ``window_delta`` path remains available only to the historical
    proposal scorer.  Exact final scoring, endpoint scoring, and record-only
    history capture all use ``window_delta_device`` and its typed receipt.
    """
    if exact_history:
        return provider.window_delta_device(
            frame_lo, frame_hi, candidate_ois, device=device
        )
    candidate_oms = candidate_ois.detach().to("cpu").numpy()
    delta = provider.window_delta(frame_lo, frame_hi, candidate_oms)
    return (
        torch_module.as_tensor(
            delta, dtype=torch_module.float64, device=device
        ),
        None,
    )


def _v18_exact_history_provider_delta(
        delta, reason, *, exact_history, torch_module, device,
        expected_shape):
    """Validate and CUDA-stamp one V18 exact-history proposal tensor."""
    if not exact_history or delta is None:
        return delta, reason, None
    valid = bool(
        isinstance(delta, torch_module.Tensor)
        and delta.is_cuda
        and delta.device == device
        and delta.dtype == torch_module.float64
        and tuple(delta.shape) == tuple(expected_shape)
        and bool(torch_module.all(torch_module.isfinite(delta)))
    )
    if not valid:
        return (
            None,
            "nn-exact-history-v18-proposal-not-cuda",
            None,
        )
    return delta, reason, {"execution_backend": "cuda-resident"}


def _terminal_union_allows_incumbent_injection(
        comm_key, candidates, *, nn_terminal_survivor_union):
    """Whether a literal incumbent may enter this finalist currency.

    A complete NN terminal union may retain the incumbent only when it owns a
    physical timing/provider history already present in that union.  Legacy
    searches keep their historical explicit fallback injection.
    """
    return bool(
        comm_key is not None
        and (
            not nn_terminal_survivor_union
            or comm_key in candidates
        )
    )


def _missing_consume_provider_frames(receipts, frames, provider_names):
    """Physical opportunities lacking one complete provider receipt."""
    return [
        int(frame) for frame in frames
        if not any(
            (isinstance(receipt, dict)
             and _strict_receipt_int(receipt.get("burst_frame")) == int(frame))
            and all(_consume_receipt_provider_applied(receipt, name)
                    for name in provider_names)
            for receipt in receipts)
    ]


def _nn_capture_proposal_trigger_receipt(
        *, physical_frames, fused_receipts, v18_terminal_receipts):
    """Compact proof that both proposal lanes ran on CUDA at every slot.

    The full consume receipts are transaction-local and can be large.  Capture
    rows persist this deterministic projection only after independently
    checking exact one-row-per-frame cardinality, provider application, and
    CUDA residency.  The offline dataset assembler revalidates the projection
    before admitting any exact-history feature row.
    """
    raw_frames = list(physical_frames)
    frames = [_strict_receipt_int(frame) for frame in raw_frames]
    if (
        not frames
        or any(frame is None for frame in frames)
        or frames != sorted(set(frames))
    ):
        raise RuntimeError(
            "endpoint capture physical frames must be sorted and unique"
        )

    fused = list(fused_receipts)
    v18_terminal = list(v18_terminal_receipts)
    lane_identity_valid = bool(
        all(
            isinstance(row, dict)
            and row.get("plane_role") == "nn-fused-primary"
            and row.get("provider_mode") == "fused"
            for row in fused
        )
        and all(
            isinstance(row, dict)
            and row.get("plane_role") == "nn-v18-terminal"
            and row.get("provider_mode") == "v18-only"
            for row in v18_terminal
        )
    )

    def provider_frames(receipts, provider_name, *, require_cuda):
        return [
            frame for frame in frames
            if any(
                _strict_receipt_int(receipt.get("burst_frame")) == frame
                and (
                    _consume_receipt_provider_cuda(receipt, provider_name)
                    if require_cuda else
                    _consume_receipt_provider_applied(receipt, provider_name)
                )
                for receipt in receipts
                if isinstance(receipt, dict)
            )
        ]

    receipt = {
        "schema_version": 1,
        "status": "completed",
        "policy": "all-physical-frames-both-lanes-cuda-applied",
        "execution_backend": "cuda-only",
        "execution_fallback": False,
        "physical_frames": frames,
        "physical_frame_count": len(frames),
        "fused_lane": {
            "plane_role": "nn-fused-primary",
            "provider_mode": "fused",
            "receipt_count": len(fused),
            "v18_applied_frames": provider_frames(
                fused, "v18_action", require_cuda=False
            ),
            "v13_applied_frames": provider_frames(
                fused, "v13_side", require_cuda=False
            ),
            "cuda_v18_v13_frames": [
                frame for frame in frames
                if any(
                    _strict_receipt_int(row.get("burst_frame")) == frame
                    and _consume_receipt_provider_cuda(row, "v18_action")
                    and _consume_receipt_provider_cuda(row, "v13_side")
                    for row in fused
                    if isinstance(row, dict)
                )
            ],
        },
        "v18_only_lane": {
            "plane_role": "nn-v18-terminal",
            "provider_mode": "v18-only",
            "receipt_count": len(v18_terminal),
            "v18_applied_frames": provider_frames(
                v18_terminal, "v18_action", require_cuda=False
            ),
            "cuda_v18_frames": provider_frames(
                v18_terminal, "v18_action", require_cuda=True
            ),
        },
    }
    if (
        not lane_identity_valid
        or len(fused) != len(frames)
        or len(v18_terminal) != len(frames)
        or receipt["fused_lane"]["v18_applied_frames"] != frames
        or receipt["fused_lane"]["v13_applied_frames"] != frames
        or receipt["fused_lane"]["cuda_v18_v13_frames"] != frames
        or receipt["v18_only_lane"]["v18_applied_frames"] != frames
        or receipt["v18_only_lane"]["cuda_v18_frames"] != frames
    ):
        raise RuntimeError(
            "endpoint capture proposal trigger receipt is incomplete"
        )
    return receipt


def _bridge_pin_state_key(state):
    """Short content key for a 54-cell state (receipt/log identity only)."""
    return hashlib.sha256(
        np.asarray(state, np.int8).tobytes()).hexdigest()[:16]


def _load_bridge_pin(src, *, tag, n_orientations):
    """Load and validate a CUBED_SCRUB_BRIDGE_PIN payload. Returns
    ``(pin_dict, None)`` on success, ``(None, reason)`` on ANY failure --
    the caller keeps stock behavior (fail-soft, logged; never raises).

    Pin JSON schema (all fields required except ``om``):
      {"tag":   str    -- decode tag; must equal the run's tag,
       "gap":   [a, b] -- frame range of the certified post-desert rest,
       "state": [54 ints in 0..5] -- the externally certified state B,
       "word":  [tokens] -- bridge word for the occluded span (each token a
                canonical HTM move in BS.MOVES; may be empty),
       "om":    int|null -- carried orientation index at B.  null => the
                pinned window resolves om with the module's OWN om_ball /
                OM-CONTINUITY machinery (header) exactly as a normally
                committed window; an int is used as the carried om,
       "source": str -- receipt provenance (e.g. the MITM chain-cert
                receipt path/sha)}

    ENGAGEMENT (evaluated at the window-commit application site): the FIRST
    committed window whose frame range reaches/contains the pin gap --
    concretely ``f1_of(b_last) >= a and f1_of(cursor) < b`` -- one-shot.
    On engagement the window's emitted tokens are REPLACED by ``word``
    (spliced through the ordinary out_windows emission, the module's one
    _apply_bridges-style splice) and the carried committed state becomes
    ``state``, so every subsequent window continues from B.

    CARRY AUTHORITY: the downstream root chain is the
    loop's ``state`` variable, which engagement sets to pin["state"]
    VERBATIM; pin["word"] is emission-only and is NEVER replayed into the
    carry.  Two pins with identical "state" but different "word" therefore
    produce byte-identical downstream windows BY CONTRACT (that is the
    word/state separation, not a carry failure); a state discriminator must
    vary "state".  Armed runs receipt ``root_state_key`` (sha16 of the
    search-root state) on EVERY window row so the chain is auditable from
    the sidecar; the engagement row carries ``replaced_word`` (the beam or
    fallback tokens the pin displaced).

    SPAN SPLIT (armed pins only): the boundary vocabulary is
    lattice spans, so when ``b`` falls strictly inside a span the span is
    split at ``b`` BEFORE any span-derived structure is built
    (_bridge_pin_split_gap_span: policy + typed refusals + receipt in
    report/summary ``bridge_pin.span_split``), making the certified
    boundary exactly expressible; the synthetic rest then lands on it.

    FORCED BOUNDARY (armed pins only; overshoot fix): while the
    pin is armed and not yet engaged, a window under construction that
    reaches/crosses ``a`` has its target list CAPPED at the first rest
    whose end frame reaches the gap, so extension TERMINATES at the pin
    boundary regardless of commit-gate confidence -- the certified rest
    supersedes the gate's confidence requirement at that one boundary
    (the pin replaces the commit decision there anyway).  Moves after
    ``b`` therefore land in SUBSEQUENT windows, decoded from B.  The
    engagement receipt carries ``forced_boundary`` (the cap removed >= 1
    later target) plus the module-resolved om index, its (up, front) key,
    and ``n_orientations`` for offline cross-instrument om adjudication.

    SYNTHETIC REST (armed pins only): when the certified
    rest is ABSENT from the scrub's rest lattice (observed: candidacy
    never surfaced [5534,5555] through the desert, so the first lattice
    rest reaching the gap ended at 5728), a target covering the gap's
    lattice spans is synthesized -- rep = best-covered read-bearing gap
    span, last = the span whose end first reaches ``b`` -- and spliced
    into the window target sequence in frame order, so the forced-
    boundary cap truncates AT it and the window ends at ``b``-ish (the
    nearest lattice-expressible boundary).  The gap is externally
    certified as a rest (chain_cert receipt), so this imports no new
    inference; dedupe is by span overlap (a real rest ending at/inside
    the gap-b span suppresses insertion); a read-empty gap follows the
    read-empty tail-target precedent (no-evidence -> honest unresolved ->
    the pin overrides); an inexpressible boundary records the typed
    ``synthetic_rest_unavailable`` reason and keeps the lattice cap.
    Receipt: ``synthetic_rest`` on the engagement row / stats / ENGAGED
    stdout line.

    COMMITTED-STATE BOUNDARY: a pin carries an externally certified anchor
    from MITM bridge receipts,
    NEVER a state inherited from the committed trajectory.  Anchors that
    inherit committed states certify the committed projection and exclude
    truth from the candidate set by construction; that falsified pattern
    must not be re-enabled through this flag.
    """
    try:
        with open(src) as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None, "payload-not-object"
        pin_tag = payload.get("tag")
        if not isinstance(pin_tag, str) or not pin_tag:
            return None, "tag-missing"
        if pin_tag != tag:
            return None, f"tag-mismatch:{pin_tag}!={tag}"
        gap = payload.get("gap")
        if (not isinstance(gap, (list, tuple)) or len(gap) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in gap)
                or gap[0] < 0 or gap[0] > gap[1]):
            return None, "gap-invalid"
        state = payload.get("state")
        if (not isinstance(state, (list, tuple)) or len(state) != 54
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           and 0 <= v <= 5 for v in state)):
            return None, "state-invalid"
        word = payload.get("word")
        if (not isinstance(word, (list, tuple))
                or not all(isinstance(t, str) and t in BS.MOVES
                           for t in word)):
            return None, "word-invalid"
        om = payload.get("om")
        if om is not None and (isinstance(om, bool) or not isinstance(om, int)
                               or not 0 <= om < n_orientations):
            return None, "om-invalid"
        source = payload.get("source")
        if not isinstance(source, str) or not source:
            return None, "source-missing"
        return dict(tag=pin_tag, gap=[int(gap[0]), int(gap[1])],
                    state=[int(v) for v in state],
                    word=[str(t) for t in word],
                    om=(None if om is None else int(om)),
                    source=source), None
    except Exception as exc:               # fail-soft: caller keeps stock
        return None, f"{type(exc).__name__}: {exc}"


def _bridge_pin_split_gap_span(gap_b, *, meta, layer_reads, layer_sub_frames,
                               evidence_reads, evidence_sub_frames,
                               evidence_provenance, onset_clusters_per_span,
                               covT_per_span, purity_per_span,
                               gate_flags_per_span, margin_per_span,
                               move_layer, certified_rest_spans):
    """Split the lattice span containing ``gap_b`` at ``gap_b`` (armed
    bridge-pin path ONLY -- the certified boundary must be
    expressible in span vocabulary or the synthetic rest overshoots).

    Returns ``{"applied": bool, "receipt": {...}}`` plus, when applied,
    every per-span input REBUILT (inputs are never mutated -- S1 purity).
    Policy, all receipted:
      * first half keeps [f0, gap_b] + the span's fit rows; second half is
        [gap_b+1, f1] with an empty fit list (sigma population unchanged);
      * reads partition by ``layer_sub_frames`` when supplied; without
        sub-frames every read stays with the FIRST half (typed
        ``read_attribution`` -- the certified rest lies in the first half);
      * opaque per-span descriptors (onset clusters / covT / purity / gate
        flags / margin) are DUPLICATED into both halves (ordering-only
        consumers; ``descriptor_policy`` receipted);
      * committed-move layers at the split span stay with the first half
        (``move_attribution``); later span indices shift +1 everywhere
        (move_layer, certified_rest_spans);
      * a span already ending at ``gap_b`` => not applied
        (status="boundary-exists"); ``gap_b`` outside every span =>
        "gap-outside-spans"; unrecognized meta row shape =>
        "meta-row-shape-unknown" -- never invented semantics.
    """
    gap_b = int(gap_b)
    split_i = None
    for si, mrow in enumerate(meta):
        f0, f1 = int(mrow[0]), int(mrow[1])
        if f1 == gap_b:
            return dict(applied=False, receipt=dict(
                status="boundary-exists", span=int(si), at_frame=gap_b))
        if f0 <= gap_b < f1:
            split_i = si
            break
    if split_i is None:
        return dict(applied=False, receipt=dict(
            status="gap-outside-spans", at_frame=gap_b))
    mrow = meta[split_i]
    if len(mrow) not in (2, 3):
        return dict(applied=False, receipt=dict(
            status="meta-row-shape-unknown", span=int(split_i),
            at_frame=gap_b, row_len=int(len(mrow))))
    f0, f1 = int(mrow[0]), int(mrow[1])
    if len(mrow) == 3:
        first_row = (f0, gap_b, mrow[2])
        second_row = (gap_b + 1, f1, [])
    else:
        first_row = (f0, gap_b)
        second_row = (gap_b + 1, f1)
    new_meta = (list(meta[:split_i]) + [first_row, second_row]
                + list(meta[split_i + 1:]))

    def split_reads(reads_all, frames_all):
        reads_s = list(reads_all[split_i])
        frames_s = (list(frames_all[split_i])
                    if frames_all is not None else None)
        if frames_s is not None and len(frames_s) == len(reads_s):
            pairs_a = [(r, f) for r, f in zip(reads_s, frames_s)
                       if int(f) <= gap_b]
            pairs_b = [(r, f) for r, f in zip(reads_s, frames_s)
                       if int(f) > gap_b]
            attribution = "subframes"
        else:
            pairs_a = [(r, None) for r in reads_s]
            pairs_b = []
            attribution = ("all-first-no-subframes" if frames_s is None
                           else "all-first-subframe-mismatch")
        new_reads = (list(reads_all[:split_i])
                     + [[r for r, _f in pairs_a],
                        [r for r, _f in pairs_b]]
                     + list(reads_all[split_i + 1:]))
        new_frames = None
        if frames_all is not None:
            new_frames = (list(frames_all[:split_i])
                          + [[f for _r, f in pairs_a if f is not None],
                             [f for _r, f in pairs_b if f is not None]]
                          + list(frames_all[split_i + 1:]))
        return new_reads, new_frames, attribution, len(pairs_a), len(pairs_b)

    new_layer_reads, new_layer_sub, read_attr, n_first, n_second = (
        split_reads(layer_reads, layer_sub_frames))
    ev_attr = None
    new_ev_reads = evidence_reads
    new_ev_sub = evidence_sub_frames
    new_ev_prov = evidence_provenance
    if evidence_reads is not None:
        new_ev_reads, new_ev_sub, ev_attr, _na, _nb = (
            split_reads(evidence_reads, evidence_sub_frames))
        if evidence_provenance is not None:
            new_ev_prov = (list(evidence_provenance[:split_i])
                           + [evidence_provenance[split_i],
                              evidence_provenance[split_i]]
                           + list(evidence_provenance[split_i + 1:]))

    def dup(arr):
        if arr is None:
            return None
        return (list(arr[:split_i]) + [arr[split_i], arr[split_i]]
                + list(arr[split_i + 1:]))

    new_move_layer = [li if int(li) <= split_i else int(li) + 1
                      for li in move_layer]
    new_certified = [si if int(si) <= split_i else int(si) + 1
                     for si in certified_rest_spans]
    receipt = dict(status="split", span=int(split_i), at_frame=gap_b,
                   first=[f0, gap_b], second=[gap_b + 1, f1],
                   read_attribution=read_attr,
                   n_reads_first=int(n_first),
                   n_reads_second=int(n_second),
                   evidence_read_attribution=ev_attr,
                   descriptor_policy="duplicated",
                   move_attribution="layer-kept-first-half")
    return dict(applied=True, receipt=receipt, meta=new_meta,
                layer_reads=new_layer_reads,
                layer_sub_frames=new_layer_sub,
                evidence_reads=new_ev_reads,
                evidence_sub_frames=new_ev_sub,
                evidence_provenance=new_ev_prov,
                onset_clusters_per_span=dup(onset_clusters_per_span),
                covT_per_span=dup(covT_per_span),
                purity_per_span=dup(purity_per_span),
                gate_flags_per_span=dup(gate_flags_per_span),
                margin_per_span=dup(margin_per_span),
                move_layer=new_move_layer,
                certified_rest_spans=new_certified)


def _load_window_audit(src, *, tag):
    """Load + validate a CUBED_SCRUB_WINDOW_AUDIT payload.  Returns
    ``(audit_dict, None)`` on success, ``(None, reason)`` on ANY failure --
    the caller keeps stock behavior (fail-soft, the pin-loader pattern).

    Audit JSON schema (window-audit-v2 -- widened selector, MEMBERSHIP match):
      {"tag":  str -- decode tag; must equal the run's tag,
       "window_frames": [a, b] | [[a, b], ...] | absent -- match any scoring
                pass whose window frame range [f1_of(a_span), f1_of(b_span)]
                is a member of the (normalized-to-list-of-pairs) selector,
       "window_idx": int | [int, ...] | absent -- else match every scoring
                pass whose window index (attempt-tagged rows) is a member of
                the (normalized-to-list) selector; frames preferred when both
                present,
       "oracle_words": [[tokens...], ...] | null -- DEV-ONLY teacher words
                (labeled like --dev-gt): each is scored by EVERY matched
                window's audit scorer even if never enumerated there; a
                finalist rank is emitted only when its candidate-score ABI
                is identical,
       "out":  str -- JSONL path the audit appends to}

    Both ``window_frames`` and ``window_idx`` absent/null is a valid selector
    meaning "match every window" (v2 widening -- v1 required a selector).
    A bare pair (``[a, b]``) or bare int is accepted as legacy shorthand for
    a single-element list; internally both selectors normalize to a list (or
    ``None`` for "match all").

    RECEIPTS-ONLY contract (S1 class, like the bridge pin): env absent =>
    byte-identical; when armed the audit must not alter enumeration,
    scoring, or selection in any code path -- it dumps the materialized
    finalist decomposition at the final scoring seam and scores oracle
    words through the same materializer/span scorer/om marginalization/
    end-pin semantics in a snapshot-restored transaction.
    """
    try:
        with open(src) as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None, "payload-not-object"
        a_tag = payload.get("tag")
        if not isinstance(a_tag, str) or not a_tag:
            return None, "tag-missing"
        if a_tag != tag:
            return None, f"tag-mismatch:{a_tag}!={tag}"

        def _is_frame_pair(v):
            return (isinstance(v, (list, tuple)) and len(v) == 2
                    and all(isinstance(x, int) and not isinstance(x, bool)
                            for x in v))

        frames = payload.get("window_frames")
        if frames is not None:
            if _is_frame_pair(frames):
                frames = [[int(frames[0]), int(frames[1])]]
            elif (isinstance(frames, (list, tuple)) and frames
                    and all(_is_frame_pair(v) for v in frames)):
                frames = [[int(v[0]), int(v[1])] for v in frames]
            else:
                return None, "window-frames-invalid"

        idx = payload.get("window_idx")
        if idx is not None:
            def _is_idx(v):
                return (isinstance(v, int) and not isinstance(v, bool)
                        and v >= 0)
            if _is_idx(idx):
                idx = [int(idx)]
            elif (isinstance(idx, (list, tuple)) and idx
                    and all(_is_idx(v) for v in idx)):
                idx = [int(v) for v in idx]
            else:
                return None, "window-idx-invalid"
        words = payload.get("oracle_words")
        if words is not None:
            if not isinstance(words, (list, tuple)):
                return None, "oracle-words-invalid"
            for word in words:
                if (not isinstance(word, (list, tuple))
                        or not all(isinstance(t, str) and t in BS.MOVES
                                   for t in word)):
                    return None, "oracle-words-invalid"
            words = [[str(t) for t in word] for word in words]
        out_path = payload.get("out")
        if not isinstance(out_path, str) or not out_path:
            return None, "out-missing"
        return dict(tag=a_tag, window_frames=frames, window_idx=idx,
                    oracle_words=words, out=out_path), None
    except Exception as exc:               # fail-soft: caller keeps stock
        return None, f"{type(exc).__name__}: {exc}"


_ORPHAN_MOTION_PROVIDER_MODES = frozenset({
    "control", "v18-only", "v13-only", "fused",
})


def _orphan_motion_filter_provider_deltas(provider_deltas, mode):
    """Filter NN providers only for the snapshot-restored orphan probe.

    ``fused`` returns the input object unchanged so an omitted config field
    executes the already-measured microscope path exactly. Other modes retain
    provider receipt rows but replace excluded tensors with ``None`` and a
    typed reason before the existing assembler sees them.
    """
    if mode not in _ORPHAN_MOTION_PROVIDER_MODES:
        raise ValueError("record-only provider mode is invalid")
    if mode == "fused":
        return provider_deltas
    allowed = {
        "control": frozenset(),
        "v18-only": frozenset({"v18_action"}),
        "v13-only": frozenset({"v13_side"}),
    }[mode]
    return [
        (name, delta, reason, meta)
        if name in allowed else
        (name, None, "record-only-provider-excluded", meta)
        for name, delta, reason, meta in provider_deltas
    ]








def _window_audit_rank_receipt(score, finalist_scores, candidate_abi,
                               finalist_abis):
    """Rank one audit candidate only inside an identical score ABI.

    A numeric ``rank_among_finalists`` is meaningful only when *every*
    finalist has the candidate's exact ABI.  Mixed-mode windows instead get a
    rank over the compatible subset (when one exists) plus an explicit common-
    rank refusal.  This helper is receipts-only and never touches selection.
    """
    scores = [float(value) for value in finalist_scores]
    abis = list(finalist_abis)
    if len(scores) != len(abis):
        raise ValueError("window-audit finalist score/ABI length mismatch")
    comparable = [
        i for i, abi in enumerate(abis) if abi == candidate_abi]
    subset_rank = (1 + sum(scores[i] > float(score) for i in comparable)
                   if comparable else None)
    common_rank = (subset_rank
                   if comparable and len(comparable) == len(scores)
                   else None)
    mismatch_fields = sorted({
        key for abi in abis
        for key in set(candidate_abi) | set(abi)
        if candidate_abi.get(key) != abi.get(key)
    })
    return dict(
        rank_among_finalists=common_rank,
        rank_among_comparable_finalists=subset_rank,
        n_comparable_finalists=len(comparable),
        rank_refusal=(None if common_rank is not None
                      else "candidate-score-abi-mismatch"),
        abi_mismatch_fields=mismatch_fields,
    )


def _state_rank_truth_receipt(
        *, score, oi, kept_indices, selected_indices, truth_indices,
        rank_nats, rank_delta, effective_k):
    """Record-only truth receipt for the in-beam rank membership cut.

    ``kept_indices`` is already in the legacy global-score order emitted by
    ``scrub_device_beam.select_device``.  Stable sorting each OM slice by
    ``score + rank_delta`` therefore reproduces that selector's exact tie
    convention without touching its tensors or control flow.  ``rank_nats``
    is the unclamped accumulator; ``rank_delta`` is the clamp actually passed
    to selection.  Missing rank arrays mean the channel was inactive, so no
    receipt fields are returned at all.
    """

    if rank_nats is None or rank_delta is None:
        return {}

    scores = np.asarray(score, dtype=np.float64)
    oms = np.asarray(oi, dtype=np.int64)
    raw_rank = np.asarray(rank_nats, dtype=np.float64)
    clamped_rank = np.asarray(rank_delta, dtype=np.float64)
    kept = np.asarray(kept_indices, dtype=np.int64)
    selected = {
        int(index) for index in np.asarray(
            selected_indices, dtype=np.int64).reshape(-1)
    }
    truth = {
        int(index) for index in np.asarray(
            tuple(truth_indices), dtype=np.int64).reshape(-1)
    }
    n_rows = len(scores)
    if any(value.shape != (n_rows,)
           for value in (oms, raw_rank, clamped_rank)):
        raise ValueError("state-rank truth receipt arrays must be aligned 1-D rows")
    if np.any(kept < 0) or np.any(kept >= n_rows):
        raise ValueError("state-rank truth receipt kept index out of bounds")
    if any(index < 0 or index >= n_rows for index in truth | selected):
        raise ValueError("state-rank truth receipt row index out of bounds")
    k_eff = int(effective_k)
    if k_eff < 0:
        raise ValueError("state-rank truth receipt effective_k must be non-negative")

    adjusted_all = scores + clamped_rank
    kept_set = {int(index) for index in kept}
    truth_oms = sorted({int(oms[index]) for index in truth})
    ordered_by_om = {}
    rank_by_om = {}
    for om_index in truth_oms:
        stratum = kept[oms[kept] == om_index]
        # ``stratum`` inherits legacy kept order.  Stable adjusted sorting is
        # exactly select_device's (adjusted desc, legacy position asc) order.
        order = np.argsort(-adjusted_all[stratum], kind="stable")
        ordered = stratum[order]
        ordered_by_om[om_index] = ordered
        rank_by_om[om_index] = {
            int(row): int(position) for position, row in enumerate(ordered)
        }

    truth_rows = []
    for row in sorted(truth):
        om_index = int(oms[row])
        adjusted_rank = rank_by_om.get(om_index, {}).get(row)
        truth_rows.append({
            "row_index": int(row),
            "oi": om_index,
            "in_kept": bool(row in kept_set),
            "raw_rank_nats": float(raw_rank[row]),
            "clamped_rank_delta": float(clamped_rank[row]),
            "original_score": float(scores[row]),
            "adjusted_score": float(adjusted_all[row]),
            "adjusted_rank_in_om_zero_based": adjusted_rank,
            "survived_post_k": bool(row in selected),
        })

    om_receipts = []
    for om_index in truth_oms:
        ordered = ordered_by_om[om_index]
        top_rows = ordered[:k_eff] if k_eff else ordered[:0]
        kth_adjusted = (
            float(adjusted_all[ordered[k_eff - 1]])
            if k_eff and len(ordered) >= k_eff else None
        )
        eligible_truth = [
            row for row in truth if row in kept_set and int(oms[row]) == om_index
        ]
        best_truth_adjusted = (
            max(float(adjusted_all[row]) for row in eligible_truth)
            if eligible_truth else None
        )
        truth_margin = (
            best_truth_adjusted - kth_adjusted
            if best_truth_adjusted is not None and kth_adjusted is not None
            else None
        )
        best_truth_rank = min(
            (rank_by_om[om_index][row] for row in eligible_truth),
            default=None,
        )
        om_receipts.append({
            "oi": int(om_index),
            "effective_k": k_eff,
            "n_kept_in_om": int(len(ordered)),
            "kth_adjusted_score": kth_adjusted,
            "truth_margin_to_kth_adjusted": truth_margin,
            "truth_best_adjusted_rank_zero_based": best_truth_rank,
            "top_k": [
                {
                    "row_index": int(row),
                    "is_truth": bool(int(row) in truth),
                    "original_score": float(scores[row]),
                    "raw_rank_nats": float(raw_rank[row]),
                    "clamped_rank_delta": float(clamped_rank[row]),
                    "adjusted_score": float(adjusted_all[row]),
                    "adjusted_rank_in_om_zero_based": int(position),
                }
                for position, row in enumerate(top_rows)
            ],
        })

    return {
        "state_rank_effective_k": k_eff,
        "state_rank_truth_rows": truth_rows,
        "state_rank_om_receipts": om_receipts,
    }


def _state_dump_truth_receipt(
        *, score, oi, states, kept_indices, truth_indices,
        effective_k, rank_delta=None):
    """Record-only per-cell state dump for truth's OM stratum at a burst cut.

    Companion to ``_state_rank_truth_receipt`` (rank/margin bookkeeping
    only): this materializes the actual per-row state vectors for a small,
    bounded row set -- top-K kept, all truth rows, and the boundary pair
    around the K cut -- restricted to the OM(s) truth's own rows occupy, so
    a kill burst can be diffed cell-by-cell against the rivals that beat it.
    Reuses the exact ``score (+ rank_delta) -> stable per-om order``
    reconstruction of ``select_device``'s applied order (see
    ``_state_rank_truth_receipt``'s docstring), generalized to accept
    ``rank_delta=None`` (state-rank seam off, the common case) as an
    all-zero bias, so the reconstructed order is exactly the plain legacy
    score order actually used for that run's selection.

    ``is_kth_cutoff`` marks the last row that survived the cut (rank K,
    1-based -- the marginal winner, the same row
    ``_state_rank_truth_receipt`` scores as ``kth_adjusted_score``);
    ``is_first_cut`` marks the row immediately after it (rank K+1 -- the
    marginal loser).  Tagging both resolves which "cutoff row" a reader
    means without guessing; in practice one of them is very often also a
    truth row.  Empty ``truth_indices`` (nothing to probe this burst)
    returns ``{}``, mirroring the sibling receipt's inactive-channel shape.

    Deliberately excludes each row's emitted move key (action/move0/move1/
    word position): those live on ``batch``/``payload`` at the call site
    without any restructuring, but reading them would be a new
    device-to-host transfer beyond the truth-probe's already-established
    pulls -- see ``state_dump_move_key_note`` on the return value.
    """

    if not truth_indices:
        return {}

    scores = np.asarray(score, dtype=np.float64)
    oms = np.asarray(oi, dtype=np.int64)
    states_arr = np.asarray(states)
    kept = np.asarray(kept_indices, dtype=np.int64)
    truth = {
        int(index) for index in np.asarray(
            tuple(truth_indices), dtype=np.int64).reshape(-1)
    }
    n_rows = len(scores)
    if oms.shape != (n_rows,):
        raise ValueError("state dump truth receipt oi must be aligned 1-D rows")
    if states_arr.ndim != 2 or states_arr.shape[0] != n_rows:
        raise ValueError("state dump truth receipt states must be 2-D[N,*] rows")
    if np.any(kept < 0) or np.any(kept >= n_rows):
        raise ValueError("state dump truth receipt kept index out of bounds")
    if any(index < 0 or index >= n_rows for index in truth):
        raise ValueError("state dump truth receipt truth index out of bounds")
    k_eff = int(effective_k)
    if k_eff < 0:
        raise ValueError("state dump truth receipt effective_k must be non-negative")

    bias = (np.zeros(n_rows, dtype=np.float64) if rank_delta is None
            else np.asarray(rank_delta, dtype=np.float64))
    if bias.shape != (n_rows,):
        raise ValueError(
            "state dump truth receipt rank_delta must be aligned 1-D rows")
    adjusted_all = scores + bias
    kept_set = {int(index) for index in kept}

    def _row(index, *, is_top_k, is_truth, is_kth_cutoff, is_first_cut):
        return {
            "row_index": int(index),
            "oi": int(oms[index]),
            "in_kept": bool(int(index) in kept_set),
            "original_score": float(scores[index]),
            "state": states_arr[index].tolist(),
            "is_top_k": bool(is_top_k),
            "is_truth": bool(is_truth),
            "is_kth_cutoff": bool(is_kth_cutoff),
            "is_first_cut": bool(is_first_cut),
        }

    truth_oms = sorted({int(oms[index]) for index in truth})
    oms_out = []
    for om_index in truth_oms:
        stratum = kept[oms[kept] == om_index]
        # Stable sort by (score + rank_delta) reproduces select_device's own
        # tie convention -- the identical pattern as
        # _state_rank_truth_receipt.  An all-zero bias (seam off) collapses
        # this to the plain legacy score order.
        order = np.argsort(-adjusted_all[stratum], kind="stable")
        ordered = stratum[order]
        top_k = ordered[:k_eff] if k_eff else ordered[:0]
        kth_cutoff = (
            int(ordered[k_eff - 1])
            if k_eff and len(ordered) >= k_eff else None)
        first_cut = (
            int(ordered[k_eff])
            if k_eff and len(ordered) > k_eff else None)
        stratum_truth = sorted(
            row for row in truth if int(oms[row]) == om_index)

        rows_by_index = {}
        for row in top_k:
            row = int(row)
            rows_by_index[row] = _row(
                row, is_top_k=True, is_truth=row in truth,
                is_kth_cutoff=(row == kth_cutoff),
                is_first_cut=(row == first_cut))
        for row in stratum_truth:
            if row in rows_by_index:
                rows_by_index[row]["is_truth"] = True
            else:
                rows_by_index[row] = _row(
                    row, is_top_k=False, is_truth=True,
                    is_kth_cutoff=(row == kth_cutoff),
                    is_first_cut=(row == first_cut))
        if first_cut is not None and first_cut not in rows_by_index:
            rows_by_index[first_cut] = _row(
                first_cut, is_top_k=False, is_truth=False,
                is_kth_cutoff=False, is_first_cut=True)

        oms_out.append({
            "oi": int(om_index),
            "effective_k": k_eff,
            "n_kept_in_om": int(len(ordered)),
            "kth_cutoff_row_index": kth_cutoff,
            "first_cut_row_index": first_cut,
            "rows": [rows_by_index[i] for i in sorted(rows_by_index)],
        })

    return {
        "state_dump_effective_k": k_eff,
        "state_dump_oms": oms_out,
        # Per-row emitted move identity (action/move0/move1/word position)
        # is NOT included: batch.action/move0/move1 and payload["word_len"]
        # are real attributes already in scope at the call site (no
        # restructuring needed), but pulling them to host would be a new
        # D2H transfer beyond the truth-probe's already-established
        # states/score/oi/rank-index pulls, which the build hard rules
        # reserve.  The 54-cell state vector alone already answers "which
        # cells" a rival disagrees with truth on (see
        # scripts/probe_state_diff.py); a future lane can add the move key
        # by threading batch.action/move0/move1 + payload["word_len"]
        # through as extra arguments if that D2H is later authorized.
        "state_dump_move_key_note": (
            "omitted -- would require a new device-to-host transfer; "
            "see this key's comment in _state_dump_truth_receipt"),
    }


def _rot_gestures():
    """The 9 whole-cube rotation GESTURES: {x,y,z} axes x {90,180,270}
    degrees. One detected motion event = one axis rotation of any amount
    (the same one-gesture semantics as an HTM face move)."""
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], int)
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], int)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], int)
    out = []
    for m in (rx, ry, rz):
        a = m
        for _ in range(3):
            out.append(a)
            a = a @ m
    return out


def om_adjacency(orientations, *, normalize_names=False):
    """One-gesture neighbour sets over the SUPPLIED orientation list, derived
    by rotating each orientation's (up, front) face vectors. Returns None
    when the list is not a valid orientation set (non-face labels or
    non-perpendicular pairs) -- om continuity then disables honestly (full
    marginalization, recorded in the sidecar header)."""
    keys = []
    for o in orientations:
        try:
            u, f = om_key_of(o)
        except Exception:
            return None
        if normalize_names:
            u = _FACE_CODE.get(str(u).upper())
            f = _FACE_CODE.get(str(f).upper())
        if u not in _FACE_VEC or f not in _FACE_VEC:
            return None
        uv, fv = np.array(_FACE_VEC[u]), np.array(_FACE_VEC[f])
        if int(np.dot(uv, fv)) != 0:
            return None
        keys.append((u, f))
    idx = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, i)
    gestures = _rot_gestures()
    nbrs = []
    for u, f in keys:
        uv, fv = np.array(_FACE_VEC[u]), np.array(_FACE_VEC[f])
        ns = set()
        for m in gestures:
            nu = _VEC_FACE[tuple((m @ uv).tolist())]
            nf = _VEC_FACE[tuple((m @ fv).tolist())]
            j = idx.get((nu, nf))
            if j is not None and (nu, nf) != (u, f):
                ns.add(j)
        nbrs.append(ns)
    return nbrs


def om_reachable(nbrs, starts, radius):
    """Exact orientation-index set reachable from any index in ``starts``.

    Unlike :func:`om_ball`, saturation remains an explicit full set.  The
    stateful scrub needs to distinguish "all 24 are physically reachable"
    from "orientation continuity is unavailable".
    """
    if nbrs is None or starts is None:
        return None
    seen = {int(i) for i in starts}
    if any(i < 0 or i >= len(nbrs) for i in seen):
        return None
    frontier = set(seen)
    for _ in range(max(0, int(radius))):
        frontier = {j for i in frontier for j in nbrs[i]} - seen
        if not frontier:
            break
        seen |= frontier
    return seen


def _stateful_gate_streams(gate_events, accepted_events):
    """Validate and factor final move-gate records for stateful OM.

    Accepted records must reproduce the historical ``rot_event_frames``
    exactly; otherwise the richer stream is not authoritative and the feature
    fails soft.  Every dropped structural gap remains one possible whole-cube
    rotation interval.  Only a dropped record with authoritative boolean
    ``color_regrip is False`` and ``align_regrip is True`` is also a contested
    optional-action slot: direct color/state-change evidence says MOVE while
    alignment vetoes it.  The opposite disagreement remains rotation-only
    because color says no state change.  Missing/malformed verdict fields are
    locally fail-closed rather than invalidating the stateful gate stream.
    """
    if gate_events is None:
        return None, "gate-events-absent"
    try:
        records = list(gate_events)
        frames, accepted, dropped_groups = [], [], {}
        verdict_counts = dict(
            dropped_records=0,
            contested_records=0,
            color_regrip_only_records=0,
            align_regrip_only_records=0,
            regrip_agreement_records=0,
            move_agreement_dropped_records=0,
            invalid_or_missing_verdict_records=0,
        )
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("gate record is not a dict")
            if type(record.get("dropped")) is not bool:  # noqa: E721
                raise ValueError("gate record dropped is not bool")
            frame = IBM.require_exact_integer(
                record["frame"], "gate-event frame")
            frames.append(frame)
            if not record["dropped"]:
                accepted.append(frame)
                continue
            verdict_counts["dropped_records"] += 1
            color = record.get("color_regrip")
            align = record.get("align_regrip")
            contested = False
            direction = None
            if type(color) is bool and type(align) is bool:  # noqa: E721
                if color != align:
                    if color:
                        direction = "color-regrip-only"
                        verdict_counts["color_regrip_only_records"] += 1
                    else:
                        contested = True
                        direction = "align-regrip-only"
                        verdict_counts["contested_records"] += 1
                        verdict_counts["align_regrip_only_records"] += 1
                elif color:
                    verdict_counts["regrip_agreement_records"] += 1
                else:
                    verdict_counts[
                        "move_agreement_dropped_records"] += 1
            else:
                verdict_counts[
                    "invalid_or_missing_verdict_records"] += 1
            gap = record.get("gap")
            try:
                if not isinstance(gap, (list, tuple)) or len(gap) != 2:
                    raise ValueError("gate gap must have two endpoints")
                gap_lo = IBM.require_exact_integer(
                    gap[0], "gate gap lower frame")
                gap_hi = IBM.require_exact_integer(
                    gap[1], "gate gap upper frame")
                if not gap_lo <= frame <= gap_hi:
                    raise ValueError("gate frame lies outside gap")
                key = ("gap", gap_lo, gap_hi)
            except (TypeError, ValueError, OverflowError):
                key = ("frame", frame)
            dropped_groups.setdefault(key, []).append(dict(
                frame=frame, contested=contested, direction=direction))
        if len(frames) != len(set(frames)):
            raise ValueError("duplicate gate-event frame")
        expected_rows = tuple(
            IBM.require_exact_integer(frame, "accepted gate-event frame")
            for frame in accepted_events)
        if len(expected_rows) != len(set(expected_rows)):
            raise ValueError("duplicate accepted gate-event frame")
        expected = set(expected_rows)
        accepted_set = set(accepted)
        dropped_set = set(frames) - accepted_set
        if accepted_set - expected:
            raise ValueError("accepted gate event absent from rot_event_frames")
        if expected & dropped_set:
            raise ValueError("dropped gate event present in rot_event_frames")
        # Some final-kept events live outside every scored gate gap, so the
        # verdict loop has no rich record for them.  ``rot_event_frames`` is the
        # final-pass authority for those events; synthesize only the missing
        # accepted records.  A contradictory recorded verdict still fails.
        synthetic_moves = sorted(expected - set(frames))
        accepted.extend(synthetic_moves)
        frames.extend(synthetic_moves)
        rotation_intervals = []
        contested_intervals = []
        contested_gap_provenance = []
        for key, group in dropped_groups.items():
            if key[0] == "gap":
                lo, hi = int(key[1]), int(key[2])
            else:
                lo = hi = int(group[0]["frame"])
            interval = dict(
                frame=int(np.median([row["frame"] for row in group])),
                lo=lo, hi=hi)
            rotation_intervals.append(interval)
            contested_rows = [row for row in group if row["contested"]]
            if contested_rows:
                contested_intervals.append(dict(interval))
                contested_gap_provenance.append(dict(
                    interval=[int(lo), int(hi)],
                    representative_frame=int(interval["frame"]),
                    contested_source_frames=sorted(
                        int(row["frame"]) for row in contested_rows),
                    source_directions=sorted({
                        str(row["direction"]) for row in contested_rows}),
                ))
        rotation_intervals.sort(key=lambda row: (row["frame"], row["lo"],
                                                 row["hi"]))
        contested_intervals.sort(key=lambda row: (
            row["frame"], row["lo"], row["hi"]))
        contested_gap_provenance.sort(key=lambda row: (
            row["representative_frame"], row["interval"]))
        rotations = [row["frame"] for row in rotation_intervals]
        return dict(move=sorted(accepted), rotation=rotations,
                    rotation_intervals=rotation_intervals,
                    contested_intervals=contested_intervals,
                    gate_verdict_counts=verdict_counts,
                    contested_gap_provenance=contested_gap_provenance,
                    synthetic_move=synthetic_moves,
                    all=sorted(frames)), None
    except Exception as exc:  # noqa: BLE001 - feature-local fail-soft boundary
        return None, str(exc)


def om_ball(nbrs, start, radius):
    """Orientation indices reachable from `start` within `radius` rotation
    gestures. Returns None for 'unconstrained' (no prior, or the ball
    saturates the supplied orientation set)."""
    if start is None or nbrs is None:
        return None
    seen = {start}
    frontier = {start}
    for _ in range(max(0, int(radius))):
        frontier = {j for i in frontier for j in nbrs[i]} - seen
        if not frontier:
            break
        seen |= frontier
    if len(seen) >= len(nbrs):
        return None
    return seen


def _dp_scores_end(cols, T, end_pinned):
    """Monotone-alignment DP scores, adapted from BS.dp_scores (the same
    machinery). When `end_pinned`, the LAST row of T must align to the
    candidate's END state -- D[:, -1] -- instead of floating (max over end
    positions). Without the pin, a candidate PASSING THROUGH the pinned state
    but ending elsewhere ties the true word (trailing phantom moves ride for
    free); the pin is the scrub analogue of bridge_search's B-anchor."""
    N = len(cols)
    out = np.full(N, -np.inf)
    bylen = {}
    for i, ix in enumerate(cols):
        bylen.setdefault(len(ix), []).append(i)
    for _k, idxs in bylen.items():
        C = np.array([cols[i] for i in idxs])          # (n, k+1)
        F = T[:, C]                                    # (W, n, k+1)
        D = F[0]
        for w in range(1, T.shape[0]):
            D = F[w] + np.maximum.accumulate(D, axis=1)
        out[np.array(idxs)] = D[:, -1] if end_pinned else D.max(axis=1)
    return out


def _stateful_score_state_indices(n_states, cols, path_cis, endpoint_cis):
    """Sorted materialized-state indices needed by non-prefit OM scoring.

    Fallback and chronological candidates can visit every state in their
    materialized column path. Endpoint-only candidates consume only their final
    state. Prefit candidates consume neither and are deliberately absent from
    both sets.
    A dense boolean mask makes de-duplication linear in the already-materialized
    state count and ``flatnonzero`` supplies the required stable sorted order.
    """
    if not path_cis and not endpoint_cis:
        return []
    needed = np.zeros(int(n_states), dtype=bool)
    for ci in path_cis:
        needed[np.asarray(cols[ci], dtype=int)] = True
    if endpoint_cis:
        needed[np.asarray([cols[ci][-1] for ci in endpoint_cis], dtype=int)] = True
    return np.flatnonzero(needed).tolist()


def _score_stateful_segment_states(seg, states):
    """Single seam for the stateful final scorer's AbsSegment evaluation."""
    return seg.score_states(states)


def _score_stateful_segment_grid(segment_rows, states):
    """One shared-state scoring seam for every read/orientation segment."""
    from detect.trellis_tracker import score_state_segment_grid

    return score_state_segment_grid(segment_rows, states)


def _score_stateful_segment_grid_device(segment_rows, states):
    """CUDA-resident counterpart; ``None`` requests the exact host path."""
    from detect.trellis_tracker import score_state_segment_grid_device

    return score_state_segment_grid_device(segment_rows, states)


def _dense_grid_admissible(n_reads, n_states):
    """Deterministic dense-grid scaling guard in existing score-state units."""
    work = max(0, int(n_reads)) * max(0, int(n_states))
    bound = int(BS.SCORE_STATE_BOUND)
    return work <= bound, work, bound


class _DensePrefixFinalRollback(RuntimeError):
    """Dense final scoring failed after a two-plane candidate union."""


def _prefix_plane_sources(record):
    return frozenset(str(source) for source in
                     record.get("_prefix_plane_sources", ()))


def _prefit_timing_alternatives(record, word_len):
    """Return retained ``(final_om, timing)`` pairs without prefix scores."""
    alternatives = sorted({
        (int(oi), tuple(timing))
        for oi, values in _temporal_timing_sets(record).items()
        for timing in values
    })
    if not alternatives:
        raise ValueError("prefit candidate has no retained timing")
    if any(len(timing) != int(word_len)
           for _oi, timing in alternatives):
        raise ValueError(
            "prefit timing length does not match the physical word")
    return tuple(alternatives)


def _prefit_history_alternatives(record, word_len):
    """Return one unconstrained row for every retained timing."""
    return tuple(
        (int(final_oi), tuple(timing), (), None, (), ())
        for final_oi, timing in _prefit_timing_alternatives(record, word_len)
    )


def _truth_probe_terminal_union_receipt(record, expected_timing, word_len):
    """Observer-only exact timing/provenance check for a prefix-plane union."""
    expected_timing = tuple(int(frame) for frame in expected_timing)
    if record is None:
        return dict(
            truth_word_present=False,
            truth_exact_timing_submitted=False,
            truth_timing_ois=[],
            truth_word_sources=[])
    alternatives = _prefit_timing_alternatives(record, word_len)
    exact_ois = sorted({
        int(oi) for oi, timing in alternatives
        if tuple(timing) == expected_timing})
    return dict(
        truth_word_present=True,
        truth_exact_timing_submitted=bool(exact_ois),
        truth_timing_ois=exact_ois,
        truth_word_sources=sorted(_prefix_plane_sources(record)),
        retained_timing_records=len(alternatives))


def _truth_probe_common_score_receipt(
        *, seqs, cols, smat, scores, truth_word, truth_end_bytes):
    """Observer-only ranks in the common exact read-score currency.

    A prefix record may submit several fixed timings and the scorer retains the
    best result for their shared physical word.  This receipt therefore names
    a word score/rank, not an unsupported per-timing score claim.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(seqs),):
        raise ValueError("common truth-probe scores do not match candidates")
    ranked = np.argsort(-scores, kind="stable")
    rank_by_index = {int(index): rank
                     for rank, index in enumerate(ranked)}
    truth_word = tuple(int(move) for move in truth_word)
    word_indices = [
        index for index, seq in enumerate(seqs)
        if tuple(int(move) for move in seq) == truth_word]
    end_indices = [
        index for index, path in enumerate(cols)
        if np.asarray(smat[path[-1]], dtype=np.int8).tobytes()
        == truth_end_bytes]
    finite_word = [index for index in word_indices if np.isfinite(scores[index])]
    finite_end = [index for index in end_indices if np.isfinite(scores[index])]
    best_word = (max(finite_word, key=lambda index: scores[index])
                 if finite_word else None)
    best_end = (max(finite_end, key=lambda index: scores[index])
                if finite_end else None)
    return dict(
        truth_word_present=bool(word_indices),
        truth_word_finite=bool(finite_word),
        truth_word_score=(None if best_word is None
                          else float(scores[best_word])),
        truth_word_rank=(None if best_word is None
                         else int(rank_by_index[best_word])),
        truth_end_state_present=bool(end_indices),
        truth_end_state_finite=bool(finite_end),
        truth_end_state_score=(None if best_end is None
                               else float(scores[best_end])),
        truth_end_state_rank=(None if best_end is None
                              else int(rank_by_index[best_end])))


def _nn_common_state_support_receipt(
        *, seqs, cols, smat, scores, candidate_prefit_om,
        truth_end_bytes=None, top_n=2,
        currency="ordinary-read-fixed-timing-stateful"):
    """Aggregate record-only NN provenance by common-score endpoint.

    Candidate rank is the stable zero-based rank among finite physical-word
    candidates in the ordinary-read fixed-timing currency.  Endpoint rank is
    the corresponding rank of each state's best candidate among unique finite
    endpoints.  Different words reaching the same state remain separate
    support candidates; a shared word contributes to both source summaries
    and once to ``shared_program_candidates``.

    Only a short digest crosses the observer boundary.  Raw state bytes and
    sticker arrays are never returned.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(seqs),) or len(cols) != len(seqs):
        raise ValueError("common state-support candidates are misaligned")
    if not isinstance(candidate_prefit_om, dict):
        raise TypeError("common state-support provenance must be a mapping")
    top_n = int(top_n)
    if top_n < 0:
        raise ValueError("common state-support top_n must be non-negative")

    groups = {}
    candidate_sources = {}
    for index, (seq, path) in enumerate(zip(seqs, cols)):
        if not path:
            raise ValueError("common state-support path is empty")
        state_bytes = np.asarray(
            smat[path[-1]], dtype=np.int8).tobytes()
        groups.setdefault(state_bytes, []).append(index)
        record = candidate_prefit_om.get(tuple(seq))
        candidate_sources[index] = (
            _prefix_plane_sources(record)
            if isinstance(record, dict) else frozenset())

    finite_indices = [
        index for index, score in enumerate(scores)
        if np.isfinite(score)]
    ranked_candidates = sorted(
        finite_indices, key=lambda index: (-float(scores[index]), index))
    candidate_rank = {
        index: rank for rank, index in enumerate(ranked_candidates)}

    def best_index(indices):
        finite = [index for index in indices if index in candidate_rank]
        return (min(finite, key=lambda index: candidate_rank[index])
                if finite else None)

    best_by_state = {
        state_bytes: best_index(indices)
        for state_bytes, indices in groups.items()
    }
    ranked_states = sorted(
        (state_bytes for state_bytes, index in best_by_state.items()
         if index is not None),
        key=lambda state_bytes: candidate_rank[best_by_state[state_bytes]])
    state_rank = {
        state_bytes: rank for rank, state_bytes in enumerate(ranked_states)}
    top_states = ranked_states[:top_n]

    truth_key = (None if truth_end_bytes is None
                 else bytes(truth_end_bytes))
    selected = list(top_states)
    if truth_key is not None and truth_key not in selected:
        selected.append(truth_key)

    observed_sources = {
        source for sources in candidate_sources.values()
        for source in sources}
    source_names = sorted(
        observed_sources | {"fused", "v18-only"})

    def best_fields(indices):
        best = best_index(indices)
        return dict(
            candidate_count=len(indices),
            best_common_score=(None if best is None
                               else float(scores[best])),
            best_common_rank=(None if best is None
                              else int(candidate_rank[best])),
            best_physical_word=(None if best is None else
                list(BS.tokens_of(seqs[best]))))

    endpoints = []
    for order, state_bytes in enumerate(selected):
        indices = list(groups.get(state_bytes, ()))
        row = best_fields(indices)
        row.update(
            state_digest=hashlib.sha256(state_bytes).hexdigest()[:16],
            unique_state_rank=(None if state_bytes not in state_rank else
                               int(state_rank[state_bytes])),
            roles=(
                ([f"top-common-{order + 1}"]
                 if order < len(top_states) else [])
                + (["truth"] if state_bytes == truth_key else [])),
            truth_endpoint=bool(state_bytes == truth_key),
            prefix_plane_sources=sorted({
                source for index in indices
                for source in candidate_sources[index]}),
            shared_program_candidates=sum(
                {"fused", "v18-only"}.issubset(
                    candidate_sources[index])
                for index in indices),
            per_source={})
        for source in source_names:
            source_indices = [
                index for index in indices
                if source in candidate_sources[index]]
            row["per_source"][source] = best_fields(source_indices)
        endpoints.append(row)

    return dict(
        status="completed",
        currency=str(currency),
        candidate_count=len(seqs),
        finite_candidate_count=len(finite_indices),
        unique_end_state_count=len(groups),
        top_n=top_n,
        truth_endpoint_requested=bool(truth_key is not None),
        endpoints=endpoints)


def _nn_final_winner_runner_decomposition(
        receipt, state_groups, common_scores):
    """Bind learned provider contributions to the common state winner/runner."""
    if not isinstance(receipt, dict):
        raise TypeError("NN final receipt is unavailable")
    diagnostics = receipt.get("program_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("NN final program diagnostics are unavailable")
    scores = np.asarray(common_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("NN final common candidate scores are malformed")

    fields = (
        "total_score", "ordinary_read_score", "learned_score",
        "v18_component", "v13_component", "pool_normalizer",
    )
    best_by_candidate = {}
    for row in diagnostics:
        if not isinstance(row, dict):
            raise ValueError("NN final program diagnostic row is malformed")
        candidate_i = _strict_receipt_int(row.get("candidate_i"))
        program_id = _strict_receipt_int(row.get("program_id"))
        if (candidate_i is None or not 0 <= candidate_i < len(scores)
                or program_id is None):
            raise ValueError("NN final program diagnostic identity is invalid")
        values = {field: float(row[field]) for field in fields}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("NN final program diagnostic is non-finite")
        if not math.isclose(
                values["ordinary_read_score"] + values["learned_score"],
                values["total_score"], rel_tol=0.0, abs_tol=5e-10):
            raise ValueError("NN final common-score decomposition does not close")
        if not math.isclose(
                values["v18_component"] + values["v13_component"]
                + values["pool_normalizer"],
                values["learned_score"], rel_tol=0.0, abs_tol=5e-10):
            raise ValueError("NN final provider decomposition does not close")
        candidate = dict(
            candidate_i=int(candidate_i), program_id=int(program_id),
            timing=list(row.get("timing") or ()),
            best_final_oi=row.get("best_final_oi"),
            best_om_trace=list(row.get("best_om_trace") or ()),
            **values,
        )
        old = best_by_candidate.get(candidate_i)
        if (old is None
                or candidate["total_score"] > old["total_score"]
                or (candidate["total_score"] == old["total_score"]
                    and candidate["program_id"] < old["program_id"])):
            best_by_candidate[candidate_i] = candidate
    if set(best_by_candidate) != set(range(len(scores))):
        raise ValueError("NN final diagnostics do not cover every candidate")
    for candidate_i, row in best_by_candidate.items():
        if not math.isclose(
                row["total_score"], float(scores[candidate_i]),
                rel_tol=0.0, abs_tol=5e-10):
            raise ValueError("NN final diagnostic score differs from common score")
    if not state_groups:
        raise ValueError("NN final state ranking is empty")

    winner = dict(best_by_candidate[int(state_groups[0]["best_i"])])
    runner = (None if len(state_groups) < 2 else
              dict(best_by_candidate[int(state_groups[1]["best_i"])]))
    delta = None
    if runner is not None:
        delta = {
            field: float(winner[field] - runner[field])
            for field in fields
        }
    return dict(
        policy="common-winning-end-state-vs-runner-up-end-state",
        winner=winner,
        runner=runner,
        delta=delta,
    )


def _merge_trace_score(old, candidate):
    """Deterministically merge equal-score OM histories."""
    if old is None or candidate[0] > old[0]:
        return candidate
    if candidate[0] < old[0]:
        return old
    diff = {i for i, pair in enumerate(zip(old[1], candidate[1]))
            if pair[0] != pair[1]}
    representative = min((old, candidate), key=lambda value: tuple(value[1]))
    return (old[0], representative[1],
            frozenset(set(old[2]) | set(candidate[2]) | diff))


def _score_fixed_timing_candidate(cols_ci, state_col, score_grid, row_frames,
                                  timing, rotation_frames, om_nbrs,
                                  start_ois):
    """Score one retained physical timing on one common read/state grid."""
    timing = tuple(int(frame) for frame in timing)
    if len(timing) != len(cols_ci) - 1:
        raise ValueError("fixed timing does not cover every physical move")
    values = {int(oi): (0.0, (), frozenset())
              for oi in sorted(start_ois)}
    move_i = 0
    timeline = ([(int(frame), 0, "rotation", None)
                 for frame in rotation_frames]
                + [(int(frame), 1, "move", i)
                   for i, frame in enumerate(timing)]
                + [(int(frame), 2, "read", i)
                   for i, frame in enumerate(row_frames)])
    for _frame, _order, kind, payload in sorted(timeline):
        if kind == "rotation":
            nxt = {}
            for src in sorted(values):
                score, trace, ambiguity = values[src]
                for dst in sorted({src} | set(om_nbrs[src])):
                    nxt[int(dst)] = _merge_trace_score(
                        nxt.get(int(dst)), (score, trace, ambiguity))
            values = nxt
        elif kind == "move":
            values = {
                oi: (score, trace + (oi,), ambiguity)
                for oi, (score, trace, ambiguity) in values.items()
            }
            move_i += 1
        else:
            ri = int(payload)
            state_pos = min(move_i, len(cols_ci) - 1)
            state_idx = state_col[int(cols_ci[state_pos])]
            values = {
                oi: (score + float(score_grid[ri, oi, state_idx]),
                     trace, ambiguity)
                for oi, (score, trace, ambiguity) in values.items()
                if np.isfinite(score_grid[ri, oi, state_idx])
            }
    if move_i != len(timing):
        raise AssertionError("fixed-timing program did not consume every move")
    return values


def _score_fixed_timing_programs_transaction(
        programs, device_score_grid, host_score_grid, row_frames,
        rotation_frames, om_nbrs, start_ois):
    """Score an ordered fixed-timing batch on device, with exact host retry.

    ``programs`` already uses compact score-grid state columns.  A device
    exception, malformed result, or unavailable device grid restarts the whole
    batch through the historical scalar oracle.  No partial device result can
    mix with host currency.  ``BaseException`` is deliberately not swallowed.
    """
    device_error = None
    if device_score_grid is not None:
        try:
            from detect.scrub_stateful_cuda import (
                score_fixed_timing_programs_device,
            )
            results = score_fixed_timing_programs_device(
                programs,
                device_score_grid,
                row_frames=row_frames,
                rotation_frames=rotation_frames,
                om_neighbors=om_nbrs,
                start_ois=start_ois,
            )
            if len(results) != len(programs):
                raise RuntimeError(
                    "fixed-timing device result length mismatch")
            return results, {
                "execution_backend": "cuda-resident",
                "execution_fallback": False,
                "execution_fallback_error": None,
            }
        except Exception as exc:
            device_error = exc

    score_grid = host_score_grid()
    state_col = tuple(range(int(score_grid.shape[2])))
    results = [
        _score_fixed_timing_candidate(
            cols_ci, state_col, score_grid, row_frames, timing,
            rotation_frames, om_nbrs, start_ois)
        for cols_ci, timing in programs
    ]
    return results, {
        "execution_backend": "host-scalar",
        "execution_fallback": device_error is not None,
        "execution_fallback_error": (
            None if device_error is None else
            f"{type(device_error).__name__}: {device_error}"),
    }


def _merge_timing_score(old, candidate, timing):
    """Merge one timing's final-OM result and expose equal-best ambiguity."""
    row = (float(candidate[0]), tuple(candidate[1]),
           frozenset(candidate[2]), tuple(timing))
    if old is None or row[0] > old[0]:
        return row
    if row[0] < old[0]:
        return old
    trace_diff = {i for i, pair in enumerate(zip(old[1], row[1]))
                  if pair[0] != pair[1]}
    timing_diff = {i for i, pair in enumerate(zip(old[3], row[3]))
                   if pair[0] != pair[1]}
    representative = min((old, row), key=lambda value: (value[3], value[1]))
    return (old[0], representative[1],
            frozenset(set(old[2]) | set(row[2])
                      | trace_diff | timing_diff), representative[3])


def _dp_scores_stateful_om_device(cols, T, row_frames, start_frame,
                                  rotation_frames, nbrs, start_ois,
                                  end_pinned):
    """Device-resident stateful-OM DP around the same reachability tables.

    ``T`` is already the resident score grid. Only the final candidate and OM
    scores leave CUDA at the caller. Structural reachability stays on the host:
    it is tiny, integer-only, and identical to the NumPy oracle.
    """
    from detect.scrub_stateful_cuda import dp_scores_stateful_om_device

    n_om = int(T.shape[1])
    rotations = sorted(int(f) for f in rotation_frames)

    def nrot(lo, hi):
        return sum(1 for f in rotations if int(lo) < f <= int(hi))

    def sources(radius):
        reachable = [om_reachable(nbrs, {i}, radius) or set()
                     for i in range(n_om)]
        return [{src for src in range(n_om) if dst in reachable[src]}
                for dst in range(n_om)]

    first_radius = nrot(start_frame, row_frames[0])
    first_reach = om_reachable(nbrs, start_ois, first_radius) or set()
    step_sources = [None] + [
        sources(nrot(row_frames[w - 1], row_frames[w]))
        for w in range(1, len(row_frames))
    ]
    tail_radius = sum(1 for f in rotations if f > int(row_frames[-1]))
    tail_sources = sources(tail_radius) if tail_radius else None
    return dp_scores_stateful_om_device(
        cols, T, first_reach=first_reach, step_sources=step_sources,
        tail_sources=tail_sources, end_pinned=end_pinned)


def _dp_scores_stateful_om(cols, T, row_frames, start_frame,
                           rotation_frames, nbrs, start_ois, end_pinned):
    """Monotone state-prefix alignment with an explicit OM path.

    ``T`` is ``(reads, orientations, materialized_states)``.  State progress
    remains monotone exactly as in :func:`_dp_scores_end`; OM may remain fixed
    or traverse the physical rotation graph only when a dropped-event gap lies
    between consecutive aligned reads.  Returns the best score per candidate
    and the score of each possible final OM, so ambiguity can be carried.
    """
    n_cand, n_om = len(cols), T.shape[1]
    scores = np.full(n_cand, -np.inf)
    final_scores = np.full((n_cand, n_om), -np.inf)
    if n_cand == 0:
        return scores, final_scores
    rotations = sorted(int(f) for f in rotation_frames)

    def _nrot(lo, hi):
        return sum(1 for f in rotations if int(lo) < f <= int(hi))

    # The rotation-source tables and initial reachability are independent of
    # the candidate. The fast path computes them once per window; the legacy
    # path is retained behind a default-off env gate for clean live A/Bs.
    fast = os.environ.get("CUBED_SCRUB_FAST_OM", "0") == "1"
    sources_cache = {}

    def _sources(radius):
        radius = int(radius)
        if not fast or radius not in sources_cache:
            reachable = [om_reachable(nbrs, {i}, radius) or set()
                         for i in range(n_om)]
            value = [{src for src in range(n_om) if dst in reachable[src]}
                     for dst in range(n_om)]
            if fast:
                sources_cache[radius] = value
            return value
        return sources_cache[radius]

    W = T.shape[0]
    first_radius = _nrot(start_frame, row_frames[0])
    first_reach_fast = (om_reachable(nbrs, start_ois, first_radius) or set()
                        if fast else None)
    step_sources = ([None] + [
        _sources(_nrot(row_frames[w - 1], row_frames[w]))
        for w in range(1, W)] if fast else None)
    tail_radius = sum(1 for f in rotations if f > int(row_frames[-1]))
    tail_sources_fast = (_sources(tail_radius)
                         if fast and tail_radius else None)

    bylen = {}
    for i, ix in enumerate(cols):
        bylen.setdefault(len(ix), []).append(i)
    for _k, idxs in bylen.items():
        candidate_indices = np.asarray(idxs, dtype=int)
        C = np.asarray([cols[ci] for ci in idxs], dtype=int)
        n_batch, word_states = C.shape
        first_reach = (first_reach_fast if fast else
                       om_reachable(nbrs, start_ois, first_radius) or set())
        # Candidate is a leading batch axis; the OM/state operations below
        # retain the scalar path's reduction and addition order exactly.
        D = np.full((n_batch, n_om, word_states), -np.inf)
        for oi in first_reach:
            D[:, oi, :] = T[0, oi, C]
        for w in range(1, W):
            src_by_dst = (step_sources[w] if fast else
                          _sources(_nrot(row_frames[w - 1], row_frames[w])))
            nxt = np.full_like(D, -np.inf)
            for oi, sources in enumerate(src_by_dst):
                if not sources:
                    continue
                prev = np.max(D[:, list(sources), :], axis=1)
                nxt[:, oi, :] = (T[w, oi, C]
                                 + np.maximum.accumulate(prev, axis=1))
            D = nxt
        # Reads constrain OM only through their last timestamp. A later,
        # authoritative rotation still changes the returned posterior.
        if tail_radius:
            tail_sources = (tail_sources_fast if fast else
                            _sources(tail_radius))
            nxt = np.full_like(D, -np.inf)
            for oi, sources in enumerate(tail_sources):
                if sources:
                    nxt[:, oi, :] = np.max(
                        D[:, list(sources), :], axis=1)
            D = nxt
        fs = D[:, :, -1] if end_pinned else np.max(D, axis=2)
        final_scores[candidate_indices] = fs
        scores[candidate_indices] = np.max(fs, axis=1)
    return scores, final_scores


def _endpoint_scores_stateful_om(T, row_frames, start_frame,
                                 rotation_frames, nbrs, start_ois,
                                 final_state_indices):
    """Vectorized chronological OM scores for endpoint-only candidates.

    Every candidate transition precedes every read, so only its final cube
    state differs.  This is score-identical to the per-candidate timeline but
    processes all rescued endpoints as one ``(OM,candidate)`` matrix.
    """
    final_state_indices = np.asarray(final_state_indices, int)
    n_cand, n_om = len(final_state_indices), T.shape[1]
    if not n_cand or T.shape[0] == 0:
        return np.full((n_cand, n_om), -np.inf)
    rotations = sorted(int(f) for f in rotation_frames)

    def nrot(lo, hi):
        return sum(1 for f in rotations if int(lo) < f <= int(hi))

    first = om_reachable(
        nbrs, start_ois, nrot(start_frame, row_frames[0])) or set()
    D = np.full((n_om, n_cand), -np.inf)
    for oi in first:
        D[oi] = T[0, oi, final_state_indices]
    for w in range(1, T.shape[0]):
        radius = nrot(row_frames[w - 1], row_frames[w])
        reachable = [om_reachable(nbrs, {i}, radius) or set()
                     for i in range(n_om)]
        nxt = np.full_like(D, -np.inf)
        for dst in range(n_om):
            sources = [src for src in range(n_om)
                       if dst in reachable[src]]
            if sources:
                nxt[dst] = (np.max(D[sources], axis=0)
                            + T[w, dst, final_state_indices])
        D = nxt
    tail_radius = sum(1 for f in rotations if f > int(row_frames[-1]))
    if tail_radius:
        reachable = [om_reachable(nbrs, {i}, tail_radius) or set()
                     for i in range(n_om)]
        nxt = np.full_like(D, -np.inf)
        for dst in range(n_om):
            sources = [src for src in range(n_om)
                       if dst in reachable[src]]
            if sources:
                nxt[dst] = np.max(D[sources], axis=0)
        D = nxt
    return D.T






def _build_joint_split_frontier(res, band, capacity=None):
    """Return the complete in-band ``(state, OM, word)`` split frontier.

    A synthetic rest is a computational boundary, not a state decision.  The
    frontier therefore keeps state and OM coupled in the scorer's conditional
    currency and preserves distinct word provenance until the parent endpoint
    can decide.  With a calibrated endpoint posterior, the frontier is the
    complete 99% credible set of future-distinct history groups emitted by the
    CUDA reducer; no sigma-band or incumbent reinjection is applied.  The
    legacy currency retains the demoted incumbent as a scoreable control even
    outside its observation band.  Capacity is always a rejection bound,
    never a top-K prune.
    """
    limit = BS.SCORE_STATE_BOUND if capacity is None else int(capacity)
    seqs = list(res.get("seqs") or ())
    cols = list(res.get("cols") or ())
    smat = res.get("smat")
    sc = np.asarray(res.get("sc") if res.get("sc") is not None else (), float)
    om_sc = res.get("candidate_om_scores")
    if (not seqs or len(cols) != len(seqs) or smat is None
            or len(sc) != len(seqs) or om_sc is None):
        return dict(status="unavailable", entries=[], n_entries=0)
    om_sc = np.asarray(om_sc, float)
    if om_sc.ndim != 2 or om_sc.shape[0] != len(seqs):
        return dict(status="unavailable", entries=[], n_entries=0)
    modes = list(res.get("candidate_score_modes") or ())
    incumbent_i = res.get("committed_index")
    traces = res.get("move_om_traces") or {}
    ambiguities = res.get("move_om_trace_ambiguities") or {}

    joint_rows = []
    global_best = -np.inf
    for i, row in enumerate(om_sc):
        finite = np.where(np.isfinite(row))[0]
        if not len(finite):
            continue
        row_max = float(np.max(row[finite]))
        offset = float(sc[i]) - row_max
        for oi in finite:
            score = float(row[oi]) + offset
            global_best = max(global_best, score)
            joint_rows.append((i, int(oi), score))
    if not np.isfinite(global_best):
        return dict(status="unavailable", entries=[], n_entries=0)

    dedup = {}
    for i, oi, score in joint_rows:
        is_incumbent = bool(incumbent_i is not None and i == incumbent_i)
        if not is_incumbent and score < global_best - float(band):
            continue
        word = tuple(seqs[i])
        word_nf = tuple(BS.MOVES.index(t) for t in BS.normal_form(
            BS.tokens_of(word)))
        state = np.asarray(smat[cols[i][-1]], np.int8).tobytes()
        mode = modes[i] if i < len(modes) else None
        key = (state, oi, word_nf, mode)
        entry = dict(
            state=state, oi=oi, score=score,
            score_rel=score - global_best,
            candidate_i=i, word=word, word_nf=word_nf,
            score_mode=mode, is_incumbent=is_incumbent,
            incumbent_in_budget=bool(
                is_incumbent and res.get("committed_in_budget", False)),
            move_om_trace=traces.get((i, oi)),
            move_om_trace_ambiguity=frozenset(
                set(ambiguities.get((i, oi), ()))
                | set(res.get("ambiguous_move_positions") or ())))
        old = dedup.get(key)
        if old is None or score > old["score"]:
            dedup[key] = entry
        elif score == old["score"]:
            old_trace = old.get("move_om_trace")
            new_trace = entry.get("move_om_trace")
            diff = ({p for p, pair in enumerate(zip(old_trace, new_trace))
                     if pair[0] != pair[1]}
                    if old_trace is not None and new_trace is not None else
                    set(range(len(word))))
            old["move_om_trace_ambiguity"] = frozenset(
                set(old.get("move_om_trace_ambiguity") or ())
                | set(entry.get("move_om_trace_ambiguity") or ()) | diff)
        elif is_incumbent:
            old["is_incumbent"] = True
            old["incumbent_in_budget"] = bool(
                old.get("incumbent_in_budget")
                or entry.get("incumbent_in_budget"))
        if len(dedup) > limit:
            return dict(status=f"tripped(frontier {len(dedup)} > {limit})",
                        entries=[], n_entries=len(dedup),
                        global_best=global_best, capacity=limit)
    entries = sorted(dedup.values(),
                     key=lambda e: (-e["score"], e["oi"], e["word_nf"]))
    return dict(status="ok", entries=entries, n_entries=len(entries),
                global_best=global_best,
                capacity=limit,
                incumbent_entries=sum(e["is_incumbent"] for e in entries))


def _score_mode_contains(mode, token):
    """Recursive membership for score currencies composed across fixed lag."""
    if mode == token:
        return True
    return (isinstance(mode, tuple)
            and any(_score_mode_contains(part, token) for part in mode))


# ---- GT-TEACHER instruments (day-1 gates 2+3; EVAL-side, default OFF) -------
# Shared movegt loader + truth-trajectory builder for the candidate-containment
# audit (shadow) and the perfect-perception oracle. Both are TEACHER-fed by a
# walkthrough/movegt_<tag>.json ([{frame, move}] in the BS.MOVES HTM alphabet);
# absent/unloadable => the instrument never engages => byte-identical decode.
def _load_movegt_entries(src):
    """Normalize a movegt source to a sorted [(frame:int, move:str)] list.
    `src` may be a path to walkthrough/movegt_<tag>.json (list of
    {"frame","move"}) OR an already-loaded list of dicts / (frame, move)
    tuples (direct-caller / test injection). Returns [] on ANY failure --
    fail-soft: the audit/oracle then simply never engages (decode identical)."""
    try:
        data = json.load(open(src)) if isinstance(src, str) else src
        out = []
        for e in data:
            if isinstance(e, dict):
                out.append((int(e["frame"]), str(e["move"])))
            else:
                f, m = e
                out.append((int(f), str(m)))
        return sorted(out, key=lambda fm: fm[0])
    except Exception:                                  # noqa: BLE001 fail-soft
        return []


def _monotone_inject(a, b):
    """Order-preserving minimal-cost INJECTIVE assignment of sorted list `a`
    into sorted list `b` (len(a) <= len(b)): indices j_0 < j_1 < ... < j_{n-1}
    minimizing sum |a_i - b_{j_i}|. Classic O(len(a)*len(b)) DP with prefix
    minima; parameter-free (no gap/skip costs -- unmatched b entries are
    free, matching is exhaustive over a)."""
    n, m = len(a), len(b)
    inf = float("inf")
    prev = [inf] * m
    for j in range(0, m - n + 1):
        prev[j] = abs(a[0] - b[j])
    parents = [[-1] * m]
    for i in range(1, n):
        cur, par = [inf] * m, [-1] * m
        best, bidx = inf, -1
        for j in range(i, m - (n - 1 - i)):
            if prev[j - 1] < best:
                best, bidx = prev[j - 1], j - 1
            cur[j] = abs(a[i] - b[j]) + best
            par[j] = bidx
        prev = cur
        parents.append(par)
    j = min(range(m), key=lambda jj: prev[jj])
    idx = [0] * n
    for i in range(n - 1, -1, -1):
        idx[i] = j
        j = parents[i][j]
    return idx


def _snap_movegt_to_events(movegt, event_frames):
    """Re-attribute each movegt move to a detected motion-event frame by an
    order-preserving minimal-total-|Δframe| INJECTIVE assignment (each move
    gets its OWN event; parameter-free DP, `_monotone_inject`).

    BLE timestamps and visible video bursts may be offset. A simple nearest
    match can assign multiple teacher moves to the same burst, so this mapping
    is injective whenever enough events are available. The window search uses
    detected event frames as its move currency; teacher moves are therefore
    attributed to the window containing the corresponding burst.

    When moves OUTNUMBER events (missed bursts) the roles flip: events are
    injectively assigned to moves and the leftover moves ride their
    neighbouring matched move's event (previous preferred; the beam's own
    DOUBLE alternative). Empty event list => raw frames unchanged (unit
    fixtures place movegt frames ON the event frames => identity)."""
    ev = sorted(int(f) for f in event_frames)
    if not ev or not movegt:
        return list(movegt)
    mgf = [int(f) for f, _ in movegt]
    if len(ev) >= len(mgf):
        idx = _monotone_inject(mgf, ev)
        return [(ev[j], mv) for j, (_f, mv) in zip(idx, movegt)]
    # missed bursts: inject events into moves; unmatched moves share a
    # neighbour's event (previous matched preferred, else next -- DOUBLE).
    midx = _monotone_inject(ev, mgf)
    assign = {i: ev[j] for j, i in enumerate(midx)}
    out, cur = [], None
    for i in range(len(movegt)):
        if i in assign:
            cur = assign[i]
        out.append(cur)
    nxt = None
    for i in range(len(movegt) - 1, -1, -1):    # leading unmatched: backfill
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [(f, mv) for f, (_raw, mv) in zip(out, movegt)]


def _truth_trajectory(start_state, tokens, perms):
    """Prefix-state trajectory of GT `tokens` (HTM, BS.MOVES alphabet) applied
    to `start_state`. Returns (states, end_bytes, unmappable):
      states     = [P0=start, P1, ..., PL] int8 arrays (one per applied move);
      end_bytes  = PL.tobytes();
      unmappable = tokens not in BS.MOVES (wide/slice/rotation notation or an
                   om-frame token the pure permutation map cannot express -- the
                   caller then reports the notation ambiguity and the check
                   degrades to whatever end state the mappable prefix reached).
    Pure state permutation applied from the SCRUB's own window-start state, so
    it is om/notation-agnostic beyond the 18-move alphabet map (this is exactly
    the end-STATE-level containment the plan falls back to on om ambiguity)."""
    s = np.asarray(start_state, np.int8).copy()
    states = [s.copy()]
    unmappable = []
    for t in tokens:
        try:
            mi = BS.MOVES.index(t)
        except ValueError:
            unmappable.append(t)
            continue
        s = s[perms[mi]]
        states.append(s.copy())
    return states, s.tobytes(), unmappable


# ============================================================ truth-probe-v2
# Frozen spec: a chronological, ancestry-aware truth-containment
# matcher that replaces `_window_truth`'s unordered `traj_bytes` set
# membership (a stale earlier checkpoint could be relabeled a live
# continuation because ANY trajectory-depth state counted as "truth" at
# EVERY burst).  These are plain module-level functions -- unlike
# `_window_truth`/`_gt_split`/`_gt_prefix` (nested closures over `_run`'s
# many-variable scope) they take every input explicitly, so the matching
# ALGORITHM is unit-testable on bare numpy/lists without a live decode, and
# is equally callable from the CUDA-resident beam (device tensors pulled
# host-side first, exactly like the pre-existing burst-row pulls) or a CPU
# beam path that carries the same word/timed/word_len currency.  Nothing
# here is ever consulted for a candidate/score/prune/commit decision --
# record-only, per the roadmap #4 contract `_window_truth` already documents.
def _load_truth_boundaries_v2(src):
    """Load + validate a `truth-boundaries-v1` payload (frozen schema,
    `PLAN_DECODER_ROBUSTNESS_CENSUS.md`; `--truth-boundaries` /
    `CUBED_TRUTH_BOUNDARIES`).  `src` may be a path (str, env-var shape) or
    an already-loaded dict (test/direct-caller injection, mirrors
    `_load_movegt_entries`).  Returns `(windows_by_frame_span,
    movegt_sha256, reason)`: `windows_by_frame_span` is a dict keyed by
    `(start_frame, end_frame)` -> the window's own boundary entry, `{}` on
    ANY failure -- fail-soft, exactly like `_load_movegt_entries`: the probe
    then simply falls back to raw-timestamp membership for every window and
    never aborts the scrub.  `reason` is None on success, else a short
    string for the receipt."""
    try:
        data = json.load(open(src)) if isinstance(src, str) else src
        if not isinstance(data, dict):
            return {}, None, "not-a-dict"
        if data.get("schema") != "truth-boundaries-v1":
            return {}, None, f"bad-schema:{data.get('schema')!r}"
        out = {}
        for w in data.get("windows", []):
            key = (int(w["start_frame"]), int(w["end_frame"]))
            out[key] = w
        return out, data.get("movegt_sha256"), None
    except Exception as e:                              # noqa: BLE001 fail-soft
        return {}, None, f"{type(e).__name__}: {e}"


def _truth_boundaries_v2_state(value):
    """Decode one `truth-boundaries-v1` `*_state_b64` field to an int8[54]
    state array, or None if absent/malformed (fail-soft: the caller then
    falls back to its own replayed state)."""
    if not value:
        return None
    try:
        return np.frombuffer(base64.b64decode(value), dtype=np.int8).copy()
    except Exception:                                   # noqa: BLE001
        return None


def _window_truth_v2(w_state, a_frame, b_frame, raw_movegt, prefix_states,
                     perms, *, boundary_override=None):
    """CORRECTED per-window GT trajectory for truth-probe-v2.  Two
    defects found in `_window_truth`/v1 are fixed at the source, not
    patched downstream:

    (a) the boundary is direct half-open RAW-timestamp membership in
    `(a_frame, b_frame]` against `raw_movegt` -- NEVER the globally
    event-snapped stream `_window_truth` reads via `_gt_split` (that global
    re-snap is exactly the bug that turned the true `[43,56)` window-6 slice
    into `[39,53)`).  `boundary_override` (one `truth-boundaries-v1` window
    entry, already matched by the caller on `(start_frame, end_frame)`)
    supersedes raw membership when given; `boundary_source` in the return
    records which applied ("raw-timestamp" | "override").

    (b) the returned trajectory is over the UNREDUCED physical teacher
    tokens (`BS.reduce_word` is never applied) with each token's OWN raw
    frame carried alongside, so a burst-indexed chronological matcher can
    require exact per-progress-point state/word/timing agreement instead of
    unordered `state in trajectory` set membership.

    `prefix_states` is the caller's GLOBAL P_0..P_n array (e.g. `_run`'s own
    `_gt_prefix` result) -- index-aligned to `raw_movegt` and snap-invariant
    (snapping only ever rewrites frames, never move order/identity), reused
    as-is for the `start_matches_gt` drift check.  Returns None when
    `raw_movegt` is empty."""
    if not raw_movegt:
        return None
    if boundary_override is not None:
        k_lo = int(boundary_override["k_lo"])
        k_hi = int(boundary_override["k_hi"])
        boundary_source = "override"
        override_start = _truth_boundaries_v2_state(
            boundary_override.get("start_state_b64"))
    else:
        k_lo = sum(1 for f, _mv in raw_movegt if f <= a_frame)
        k_hi = sum(1 for f, _mv in raw_movegt if f <= b_frame)
        k_hi = max(k_lo, k_hi)
        boundary_source = "raw-timestamp"
        override_start = None
    if override_start is not None:
        start_state = override_start
    elif 0 <= k_lo < len(prefix_states):
        start_state = prefix_states[k_lo]
    else:
        start_state = np.asarray(w_state, np.int8)
    physical_tokens = [mv for _f, mv in raw_movegt[k_lo:k_hi]]
    physical_frames = [int(_f) for _f, _mv in raw_movegt[k_lo:k_hi]]
    phys_states, end_bytes, unmappable = _truth_trajectory(
        start_state, physical_tokens, perms)
    return dict(
        capability="truth-probe-v2.1", boundary_source=boundary_source,
        k_lo=k_lo, k_hi=k_hi,
        physical_tokens=list(physical_tokens),
        physical_word=[BS.MOVES.index(tok) for tok in physical_tokens],
        physical_frames=physical_frames,
        physical_word_len=len(physical_tokens),
        phys_state_bytes=[s.tobytes() for s in phys_states],
        end_bytes=end_bytes, unmappable=unmappable,
        start_matches_gt=bool(np.array_equal(
            np.asarray(w_state, np.int8), start_state)))


def _truth_probe_v2_progress(physical_frames, burst_frames):
    """Cumulative RAW teacher-token progress target per burst, plus each
    token's own expected burst frame (chronological matcher, stage-1
    input).

    `physical_frames` = raw per-token teacher frames for the window's exact
    (unreduced) physical-move slice (`_window_truth_v2`'s own currency,
    never globally event-snapped).  `burst_frames` = the window's
    own burst/action-opportunity frames (the beam's move-EMISSION currency;
    CUDA `timed` values are always one of these, never a raw teacher frame).

    Returns `(progress, due_frame)`:
      `progress[k]`  = expected cumulative RAW-token count after burst `k`,
                       for `k` in `0..len(burst_frames)` (`progress[0]==0`,
                       the window root, before any burst has fired).
      `due_frame[j]` = the burst frame at which teacher token `j` (0-based)
                       becomes due, or None if the window's OWN burst stream
                       is exhausted before token `j` is reached -- an
                       expressivity gap (T2/T8-shaped) the probe must
                       surface honestly, never silently drop or misreport
                       as a live loss.

    Monotone two-pointer walk, O(len(physical_frames) + len(burst_frames));
    both inputs are already sorted (teacher/burst chronology)."""
    bursts = [int(f) for f in burst_frames]
    frames = [int(f) for f in physical_frames]
    progress = [0] * (len(bursts) + 1)
    due_frame = [None] * len(frames)
    cnt = 0
    for k, bf in enumerate(bursts, start=1):
        while cnt < len(frames) and frames[cnt] <= bf:
            due_frame[cnt] = bf
            cnt += 1
        progress[k] = cnt
    return progress, due_frame


# -------------------------------------------------------- truth-probe-v2.1
# Burst-granular metric normalization.  The beam's OWN word
# alphabet is HTM-shaped and `expand_ball` FORBIDS two adjacent same-face
# quarter moves (the standing QTM verdict): a motion-merged 180
# is always ONE burst emitting ONE `X2`-class token, never two `X` tokens.
# `_window_truth_v2`'s trajectory is intentionally raw QTM (unreduced -- see
# its own docstring), so before v2.1 the matcher compared beam candidates
# against RAW teacher tokens directly -- whenever a same-face 180 landed
# inside one detected burst, the physically-true beam row could NEVER match
# (wrong token index, wrong word_len, wrong timed length; the
# "T2-multi" windows).  `_truth_probe_v2_merge_bursts` fixes this at the
# source, downstream of `_window_truth_v2` + `_truth_probe_v2_progress` (it
# needs both the raw tokens AND their resolved due bursts, neither of which
# `_window_truth_v2` alone has), so `_truth_probe_v2_match_stage` itself
# keeps matching EXACTLY as before -- only the expected word/timed/state
# currency it is fed changes, from raw-teacher to beam-alphabet.
def _truth_probe_v2_merge_bursts(physical_tokens, due_frame):
    """v2.1 burst-granular metric normalization: fold any ADJACENT identical
    quarter-turn pair (`X,X` or `X',X'`) that shares one due burst
    (`due_frame[i] == due_frame[i+1]`, i.e. the window's own detected burst
    stream only offered ONE action-opportunity for both) into the beam's own
    half-turn token -- the physically-true beam row carries exactly one `X2`
    word/timed entry there, never two `X` entries.

    Non-adjacent same-face repeats (different due bursts) are genuinely two
    separate physical turns and are left untouched (two quarter tokens,
    unchanged from today -- regression-tested).  A different-face pair in
    one due burst is also left untouched because the live beam represents it
    exactly as an ``ACTION_DOUBLE`` with two word/timing entries.  Same-face
    pairs that were not merged above, or 3+ tokens piled onto one burst, are
    genuine emission-capacity gaps; their duplicate timing survives into
    ``beam_due_frame`` for `_truth_probe_v2_match_stage` to report.

    Returns `(beam_tokens, beam_due_frame, merges)`:
      `beam_tokens`     = `physical_tokens` with each clean same-due-burst
                          identical pair replaced by its BS.MOVES half-turn
                          token (`["D", "D"] -> ["D2"]`).
      `beam_due_frame`  = `due_frame` re-indexed to `beam_tokens` (one
                          shared entry per merged pair; unchanged elsewhere,
                          including any surviving multi-token duplicate and
                          any trailing `None`).
      `merges`          = receipt list, one dict per merge:
                          `{"k_pair": [i, i+1], "token": "D2",
                            "burst_frame": F}` (`i, i+1` index the ORIGINAL
                          `physical_tokens`/`due_frame`)."""
    beam_tokens, beam_due_frame, merges = [], [], []
    i, n = 0, len(physical_tokens)
    while i < n:
        j = i + 1
        mergeable = (
            j < n
            and due_frame[i] is not None
            and due_frame[i] == due_frame[j]
            and physical_tokens[i] == physical_tokens[j]
            and physical_tokens[i][1:] in ("", "'"))     # quarter turn only
        if mergeable:
            half = physical_tokens[i][0] + "2"
            beam_tokens.append(half)
            beam_due_frame.append(due_frame[i])
            merges.append(dict(k_pair=[i, j], token=half,
                               burst_frame=due_frame[i]))
            i = j + 1
        else:
            beam_tokens.append(physical_tokens[i])
            beam_due_frame.append(due_frame[i])
            i += 1
    return beam_tokens, beam_due_frame, merges


def _truth_probe_v2_beam_progress(beam_due_frame, burst_frames):
    """Cumulative BEAM-currency progress target per burst -- the v2.1
    counterpart to `_truth_probe_v2_progress` (same `progress[k]` contract:
    `progress[0] == 0`, `progress[k]` = expected cumulative beam-token count
    due after burst `k`), computed directly from `beam_due_frame`
    (`_truth_probe_v2_merge_bursts`'s own output): every non-None entry
    there IS already one of `burst_frames` (copied straight from
    `_truth_probe_v2_progress`'s own resolved `due_frame`), so this is a
    direct cumulative count, not a second raw-frame two-pointer walk. `None`
    entries (burst-exhausted tail tokens) are never absorbed, exactly like
    the raw progress walk."""
    bursts = [int(f) for f in burst_frames]
    resolved = [int(f) for f in beam_due_frame if f is not None]
    progress = [0] * (len(bursts) + 1)
    cnt = 0
    for k, bf in enumerate(bursts, start=1):
        while cnt < len(resolved) and resolved[cnt] <= bf:
            cnt += 1
        progress[k] = cnt
    return progress


def _truth_probe_v2_timing_capacity_reason(physical_word_idx, due_frame):
    """Return why one exact beam-timing prefix exceeds the live grammar.

    ``None`` means the prefix is structurally representable by the beam's
    SKIP/SINGLE/DOUBLE alphabet.  The function intentionally does not inspect
    a window's configured skip/insert caps; those are search-budget policy,
    while this helper describes the branch grammar shared by the matcher and
    terminal-union receipt.
    """
    if len(physical_word_idx) != len(due_frame):
        raise ValueError("truth word/timing lengths differ")
    if any(frame is None for frame in due_frame):
        return "burst-exhausted"
    timing_i = 0
    while timing_i < len(due_frame):
        timing_j = timing_i + 1
        while (timing_j < len(due_frame)
               and due_frame[timing_j] == due_frame[timing_i]):
            timing_j += 1
        burst_width = timing_j - timing_i
        same_face_pair = bool(
            burst_width == 2
            and int(physical_word_idx[timing_i]) // 3
            == int(physical_word_idx[timing_i + 1]) // 3)
        if burst_width > 2 or same_face_pair:
            # Backward-compatible receipt name: historically this covered all
            # duplicate timings, including the different-face DOUBLE that is
            # in fact supported.  It now denotes only genuinely unsupported
            # same-face pairs or 3+ emissions at one burst.
            return "multi-face-burst"
        timing_i = timing_j
    return None


def _truth_probe_v2_match_stage(*, word, timed, word_len, states, target_m,
                                physical_word_idx, phys_state_bytes,
                                due_frame):
    """Pure, backend-agnostic chronological truth match at one
    pipeline stage.

    Row `i` is a truth match at this stage iff its OWN accumulated log --
    `word[i, :word_len[i]]`, `timed[i, :word_len[i]]`, `states[i]` -- is
    byte-identical to the teacher's prefix of the SAME length: the exact
    unreduced physical-move identities, the exact burst each was due at
    (`due_frame`, already expressed in the window's own burst currency --
    never the teacher's own independently-jittered raw frame), and the
    exact replayed state.  `target_m` is the SINGLE progress count every
    stage within one burst shares (the teacher's own cumulative RAW-token
    count due by that burst, `_truth_probe_v2_progress`'s `progress[k]`) --
    a row at any OTHER word_len has fallen behind or run ahead and is not
    truth AT THIS BURST, independent of whether it matches truth at some
    other progress depth.  Pooling every depth into one set regardless of
    burst is exactly the defect this replaces.

    `word`/`timed`/`word_len`/`states` are plain numpy arrays (already
    device-to-host if the caller is CUDA-resident, or the CPU beam's own
    arrays); `physical_word_idx`/`phys_state_bytes`/`due_frame` come from
    `_window_truth_v2` + `_truth_probe_v2_progress`, normalized into the
    beam's own word-alphabet currency by `_truth_probe_v2_merge_bursts` +
    `_truth_probe_v2_beam_progress` (v2.1) before this function
    ever sees them -- this matcher itself is currency-agnostic and would
    match equally correctly fed raw-teacher arrays directly, it is simply
    never called that way anymore.  Ancestry is a SEPARATE, explicit check
    (`_truth_probe_v2_ancestry_ok`) at the
    generation stage only -- this function alone is already chronology-exact
    because word/timed are append-only per-row logs (a row can only present
    this exact prefix if every ancestor also did; see `expand_payload`'s
    clone-then-append contract), but the frozen spec calls for a literal
    ancestry receipt too, so it is never silently assumed away.

    Returns a dict: `matched_rows` (sorted `list[int]`), `target_m`,
    `expected_state_bytes` (None when inexpressible), `expressible` (bool --
    False when `target_m` exceeds the teacher's own physical length, the
    window's own burst stream cannot carry it, or one burst requires an
    unsupported same-face pair or more than the live DOUBLE branch's two
    tokens, an honest expressivity gap, never a silent non-match), and (v2.1)
    `inexpressible_reason`: None when expressible, else one of
    "target-out-of-range" (`target_m` outside `[0, len(physical_word_idx)]`),
    "burst-exhausted" (the window's own burst stream is exhausted before
    this progress point -- a `None` in `due_frame`), or "multi-face-burst"
    (the backward-compatible receipt name for a due burst requiring an
    unsupported same-face pair or 3+ tokens after
    `_truth_probe_v2_merge_bursts`; an exact two-token different-face pair is
    the live beam's supported ``ACTION_DOUBLE`` and remains expressible)."""
    n = len(word_len)
    m = int(target_m)
    if not (0 <= m <= len(physical_word_idx)):
        return dict(matched_rows=[], target_m=m,
                    expected_state_bytes=None, expressible=False,
                    inexpressible_reason="target-out-of-range")
    expected_state = phys_state_bytes[m]
    expected_word = tuple(int(v) for v in physical_word_idx[:m])
    expected_timed = tuple(due_frame[:m])
    capacity_reason = _truth_probe_v2_timing_capacity_reason(
        expected_word, expected_timed)
    if capacity_reason is not None:
        return dict(matched_rows=[], target_m=m,
                    expected_state_bytes=expected_state, expressible=False,
                    inexpressible_reason=capacity_reason)
    matched = []
    for i in range(n):
        if int(word_len[i]) != m:
            continue
        if tuple(int(v) for v in word[i, :m]) != expected_word:
            continue
        if tuple(int(v) for v in timed[i, :m]) != expected_timed:
            continue
        if states[i].tobytes() != expected_state:
            continue
        matched.append(i)
    return dict(matched_rows=matched, target_m=m,
                expected_state_bytes=expected_state, expressible=True,
                inexpressible_reason=None)


def _truth_probe_v2_ancestry_ok(source, candidate_rows, prev_truth_rows):
    """Explicit ancestry receipt at the generation stage: which of
    `candidate_rows` (row indices into the just-expanded batch) descend, via
    the expansion `source` parent index, from a row this probe already
    validated as truth at the PRIOR burst (`prev_truth_rows`, index space =
    the pre-expansion batch, i.e. exactly what `source` is indexed against).

    Record-only cross-check: `_truth_probe_v2_match_stage`'s exact
    word/timed/state prefix equality already logically implies unbroken
    ancestry (the CUDA/CPU payload is an append-only per-row log, cloned
    then extended from `source` at every expansion), so this should never
    disagree with that result -- a disagreement would itself be a receipt-
    worthy signal, never something to silently reconcile.  Returns
    `{row: bool}` for every row in `candidate_rows`."""
    prev = {int(i) for i in prev_truth_rows}
    src = [int(v) for v in source]
    return {int(row): (src[int(row)] in prev) for row in candidate_rows}


def window_decision(state_groups, word_classes, at_cap, sigma,
                    z_sigma=SSV.Z_SIGMA):
    """The commit gate + state-first honesty as a PURE function (testable).

    state_groups: end-state ranking, best first (dicts with `best_score`).
    word_classes: normal-form class ranking, best first (BS.class_ranking
    convention, `best_score`). sigma: the solve fit-population std (the
    per-span fit population); z_sigma defaults to 2.0. Returns
    dict(decision=commit|extend|low_conf|unresolved, state_margin,
    margin_sigma, word_margin, state_unique, word_decisive).

    Gate (existing semantics only): commit iff the best END-STATE beats
    the runner-up end-state by >= z_sigma x sigma
    in solve-sigma units -- a ~0-sigma margin is the 0.72-sigma-class
    commit this mode refuses) AND margins are strictly positive AND the best
    word class is decisive (class_margin > 0). Sub-bar/ambiguous => extend,
    never commit. At the cap: state pinned (>= bar) but word not => low_conf
    commit (state-first honesty, the success metric); state below the
    bar => unresolved."""
    inf = float("inf")
    smargin = (inf if len(state_groups) <= 1 else
               float(state_groups[0]["best_score"]
                     - state_groups[1]["best_score"]))
    wmargin = (inf if len(word_classes) <= 1 else
               float(word_classes[0]["best_score"]
                     - word_classes[1]["best_score"]))
    sig = float(sigma) if sigma and sigma > 0 else 1.0
    margin_sigma = smargin / sig if np.isfinite(smargin) else inf
    state_ok = smargin > 0 and margin_sigma >= z_sigma
    word_ok = wmargin > 0
    if state_ok and word_ok:
        decision = "commit"
    elif not at_cap:
        decision = "extend"
    elif state_ok:
        decision = "low_conf"
    else:
        decision = "unresolved"
    return dict(decision=decision, state_margin=smargin,
                margin_sigma=margin_sigma, word_margin=wmargin,
                state_unique=state_ok, word_decisive=word_ok)


# ------------------------------------------------ seeded beam selection
def _seeded_lane_identity_device(
        batch, *, skip_cap, insert_cap):
    """Return lane IDs for exact seeded complete-band selection."""
    lane_width = int(insert_cap) + 1
    timing_lane_count = (int(skip_cap) + 1) * lane_width
    timing_lane = batch.nskip * lane_width + batch.nins
    return timing_lane, timing_lane_count


def _select_seeded_complete_band_device(
        batch, *, torch, DB, band, skip_cap, insert_cap):
    """Exact CUDA complete-band selector shared by live code and tests."""
    ranked = torch.argsort(batch.score, descending=True, stable=True)
    if not len(batch):
        return DB.SelectionResult(
            batch=DB.take_batch(batch, ranked),
            selected_indices=ranked,
            ranked_indices=ranked,
            kept_indices=ranked,
        )
    lane, lane_count = _seeded_lane_identity_device(
        batch,
        skip_cap=skip_cap,
        insert_cap=insert_cap,
    )
    lane_best = torch.full(
        (lane_count,),
        -torch.inf,
        dtype=torch.float64,
        device=batch.score.device,
    )
    lane_best.scatter_reduce_(
        0, lane, batch.score, reduce="amax", include_self=True
    )
    in_band = batch.score >= lane_best[lane] - float(band)
    ranked_in_band = in_band[ranked]
    ranked_required = batch.req[ranked]
    kept = torch.cat((
        ranked[ranked_in_band],
        ranked[~ranked_in_band & ranked_required],
    ))
    return DB.SelectionResult(
        batch=DB.take_batch(batch, kept),
        selected_indices=kept,
        ranked_indices=ranked,
        kept_indices=kept,
    )














# ------------------------------------------------------------- orchestrator
def run_scrub_decode(*, moves, move_layer, move_oms, meta, layer_reads,
                     layer_sub_frames=None, evidence_reads=None,
                     evidence_sub_frames=None, evidence_provenance=None,
                     seg_factory: Callable, init_arr, final_arr,
                     n_bridge: int, orientations, perms=None,
                     rot_event_frames=(), onset_clusters_per_span=None,
                     covT_per_span=None, purity_per_span=None,
                     gate_flags_per_span=None, margin_per_span=None,
                     z_sigma: float = SSV.Z_SIGMA,
                     beam_k: int = DEFAULT_BEAM_K,
                     midmotion_rows=None,
                     containment_audit=None, oracle_likelihood=None,
                     om_stateful: bool = False, gate_events=None,
                     certified_rest_spans=(), visual_transition_episodes=(),
                     intraburst_phase_slots=(), intraburst_phase_audit=None,
                     gate_drop_slots: bool = False,
                     dense_prefix: bool = False,
                     sidecar_path: Optional[str] = None, tag: str = ""):
    """Windowed-scrub post-pass. Returns (new_moves, new_move_layer,
    new_move_oms, report). report["emitted"] True => the caller replaces the
    timeline. Any runtime exception logs the error and returns the original
    lists untouched. A sidecar IO error alone never aborts the scrub.
    `beam_k` is
    the burst-aligned-beam width cap -- the tracker's own beam convention
    (TrellisTracker.__init__ K; the live hook passes self.K).

    ``layer_reads`` remains the finalized control plane for candidacy,
    opening-OM priors, carry, and rest coverage.  The optional ``evidence_*``
    plane is consumed only by stateful candidate-conditioned chronological
    scoring.  ``dense_prefix`` optionally runs that evidence as a second,
    independent prefix beam over the control beam's immutable structure and
    unions its survivors before final scoring.  It is deliberately
    all-or-nothing and requires ``om_stateful`` so exact-frame evidence cannot
    be scored under stateless per-read OM marginalization."""
    report = {"status": "ran", "emitted": False, "windows": 0, "committed": 0,
              "extended": 0, "low_conf": 0, "unresolved": 0,
              "final_endpoint_ok": None}
    try:
        return _run(moves=moves, move_layer=move_layer, move_oms=move_oms,
                    meta=meta, layer_reads=layer_reads,
                    layer_sub_frames=layer_sub_frames,
                    evidence_reads=evidence_reads,
                    evidence_sub_frames=evidence_sub_frames,
                    evidence_provenance=evidence_provenance,
                    seg_factory=seg_factory, init_arr=init_arr,
                    final_arr=final_arr, n_bridge=n_bridge,
                    orientations=orientations, perms=perms,
                    rot_event_frames=rot_event_frames,
                    onset_clusters_per_span=onset_clusters_per_span,
                    covT_per_span=covT_per_span,
                    purity_per_span=purity_per_span,
                    gate_flags_per_span=gate_flags_per_span,
                    margin_per_span=margin_per_span, z_sigma=z_sigma,
                    beam_k=beam_k, midmotion_rows=midmotion_rows,
                    containment_audit=containment_audit,
                    oracle_likelihood=oracle_likelihood,
                    om_stateful=om_stateful, gate_events=gate_events,
                    certified_rest_spans=certified_rest_spans,
                    visual_transition_episodes=visual_transition_episodes,
                    intraburst_phase_slots=intraburst_phase_slots,
                    intraburst_phase_audit=intraburst_phase_audit,
                    gate_drop_slots=gate_drop_slots,
                    dense_prefix=dense_prefix,
                    sidecar_path=sidecar_path, tag=tag, report=report)
    except Exception as exc:                       # preserve the stock decode
        print(f"  [scrub-decode] ERROR {type(exc).__name__}: {exc} -- "
              f"scrub abandoned, stock decode emitted", flush=True)
        report.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      emitted=False)
        # The streaming header is intentionally written before any window.
        # Dense scale-guard state is dynamic, so an exception (especially a
        # guard trip) must append the final state instead of leaving a stale
        # header that claims the guard never fired.  Sidecar IO remains
        # fail-soft and can never mask the stock-decode fallback.
        if (sidecar_path is not None
                and (report.get("dense_evidence") is not None
             or report.get("dense_prefix") is not None
             or report.get("gate_drop_slots") is not None
             or report.get("intraburst_phase_slots") is not None)
                and report.get("sidecar") == sidecar_path):
            try:
                with open(sidecar_path, "a") as fh:
                    error_row = {
                        "kind": "error", "tag": tag,
                        "status": report["status"],
                        "error": report["error"],
                    }
                    if report.get("dense_evidence") is not None:
                        error_row["dense_evidence"] = report[
                            "dense_evidence"]
                    if report.get("dense_prefix") is not None:
                        error_row["dense_prefix"] = report["dense_prefix"]
                    if report.get("gate_drop_slots") is not None:
                        error_row["gate_drop_slots"] = report[
                            "gate_drop_slots"]
                    if report.get("intraburst_phase_slots") is not None:
                        error_row["intraburst_phase_slots"] = report[
                            "intraburst_phase_slots"]
                    fh.write(json.dumps(
                        error_row, sort_keys=True, default=str) + "\n")
            except OSError as sidecar_exc:
                report["sidecar_error"] = (
                    f"{type(sidecar_exc).__name__}: {sidecar_exc}")
        return moves, move_layer, move_oms, report


def _run(*, moves, move_layer, move_oms, meta, layer_reads, layer_sub_frames,
         evidence_reads, evidence_sub_frames, evidence_provenance,
         seg_factory,
         init_arr, final_arr, n_bridge, orientations, perms,
         rot_event_frames, onset_clusters_per_span, covT_per_span,
         purity_per_span, gate_flags_per_span, margin_per_span, z_sigma,
         beam_k, midmotion_rows,
         containment_audit, oracle_likelihood,
         om_stateful, gate_events, certified_rest_spans,
         visual_transition_episodes, intraburst_phase_slots,
         intraburst_phase_audit, gate_drop_slots, dense_prefix,
         sidecar_path, tag, report):
    stateful_cuda_enabled = (
        os.environ.get("CUBED_GPU_SCRUB_STATEFUL_DP", "0") == "1")
    stateful_cuda_stats = dict(
        grid_calls=0,
        fused_dp_calls=0,
        fixed_timing_device_calls=0,
        fixed_timing_device_programs=0,
        fixed_timing_device_records=0,
        fixed_timing_device_fallbacks=0,
        fused_fallback_wall_ms=0.0,
        device_grid_bytes=0,
        fused_result_d2h_bytes=0,
        grid_d2h_calls=0,
        grid_d2h_bytes=0,
        fallback_candidates=0,
        chronological_candidates=0,
        endpoint_candidates=0,
        prefit_candidates=0,
        failures=0,
    )
    if perms is None:
        perms = BS.derive_perms()
    if not orientations:
        raise ValueError("orientations list required (om marginalization)")
    dense_evidence = any(value is not None for value in (
        evidence_reads, evidence_sub_frames, evidence_provenance))
    if dense_evidence and (evidence_reads is None
                           or evidence_sub_frames is None):
        raise ValueError("dense scrub evidence requires reads and exact frames")
    if dense_evidence and not om_stateful:
        raise ValueError("dense scrub evidence requires om_stateful=True")
    if dense_prefix and not dense_evidence:
        raise ValueError("dense prefix requires dense scrub evidence")
    if dense_prefix and not om_stateful:
        raise ValueError("dense prefix requires om_stateful=True")
    visual_transition_episodes = tuple(
        sorted({(int(lo), int(hi))
                for lo, hi in (visual_transition_episodes or ())
                if int(lo) <= int(hi)}))
    if visual_transition_episodes and not om_stateful:
        raise ValueError("visual transition slots require om_stateful=True")
    if gate_drop_slots and not om_stateful:
        raise ValueError("gate-drop slots require om_stateful=True")
    intraburst_phase_slots, intraburst_phase_receipt = (
        IBM.validate_intraburst_phase_transaction(
            intraburst_phase_slots, intraburst_phase_audit))
    if intraburst_phase_slots and not om_stateful:
        intraburst_phase_slots = ()
        intraburst_phase_receipt.update(
            enabled=False,
            admission_status="disabled",
            admission_reason="phase slots require stateful OM",
            slot_count=0,
            optional_action_rows=0,
        )
    intraburst_phase_receipt.update(
        producer_name="dense raw motion + authoritative event containment",
        action_alphabet="SKIP + 18 SINGLE; no DOUBLE",
        ownership="window_start < phase_hi <= window_end",
        coalescing="kept hard event inside exact phase interval only",
        om_semantics=(
            "SKIP is a latent stay-or-one-neighbor gesture even without a "
            "separate rotation observation; SINGLE preserves the held OM"),
        failure_policy=(
            "unavailable, malformed, ambiguous, over-bound, overlapping, or "
            "runtime-failed phase transaction returns the exact baseline "
            "lattice"),
    )
    report["intraburst_phase_slots"] = intraburst_phase_receipt
    if visual_transition_episodes:
        report["visual_transition_slots"] = dict(
            enabled=True,
            global_episode_count=len(visual_transition_episodes),
            global_episodes=[list(run)
                             for run in visual_transition_episodes],
            action_alphabet="SKIP + 18 SINGLE; no DOUBLE",
            hard_count_budget="unchanged",
            eligibility=(
                "tracker-certified transition spans only: exactly one "
                "strictly interior sustained break, no kept interior event, "
                "collapsed quality sample with existing emptygap threshold, "
                "and admitted pre/post rests"),
            ownership="window_start < episode_hi <= window_end",
            coalescing="exact motion-frame overlap only",
            evidence_policy=(
                "control by default; guarded dense rows only inside the "
                "exact owned episode; guard/unusable => control fallback"))
    # BRIDGE PIN (CUBED_SCRUB_BRIDGE_PIN; default absent/empty =>
    # byte-identical: no key, no branch, no
    # report field -- the state_rank surfacing pattern).  Schema/engagement/
    # Committed-state boundary: _load_bridge_pin docstring.  Load is fail-soft: any
    # malformed/mismatched payload logs a reason and keeps stock behavior.
    # Loaded HERE -- before the anchor-view build -- so an armed pin can
    # split the gap-b span before any span-derived structure exists.
    _bridge_pin_src = os.environ.get("CUBED_SCRUB_BRIDGE_PIN")
    bridge_pin = None
    bridge_pin_stats = {
        "requested": bool(_bridge_pin_src), "armed": False, "engaged": 0,
        "window_idx": None, "window_frames": None,
        "forced_boundary": None, "synthetic_rest": None,
        "synthetic_rest_unavailable": None, "span_split": None,
        "om_source": None, "om_index": None, "om_key": None,
        "n_orientations": None, "failsoft": None}
    if _bridge_pin_src:
        bridge_pin, _bp_reason = _load_bridge_pin(
            _bridge_pin_src, tag=tag, n_orientations=len(orientations))
        if bridge_pin is None:
            bridge_pin_stats["failsoft"] = f"load:{_bp_reason}"
            print(f"  [scrub-bridge-pin] FAILSOFT (stock scrub kept): "
                  f"{_bp_reason} src={_bridge_pin_src}", flush=True)
        else:
            bridge_pin_stats["armed"] = True
            bridge_pin_stats["pin"] = dict(
                tag=bridge_pin["tag"], gap=list(bridge_pin["gap"]),
                word_len=len(bridge_pin["word"]), om=bridge_pin["om"],
                source=bridge_pin["source"],
                state_key=_bridge_pin_state_key(bridge_pin["state"]))
    if bridge_pin is not None:
        # SPAN SPLIT AT gap_b (armed-pin path ONLY).  The
        # synthetic rest can only END on an existing span boundary; in one
        # observed case the span containing gap_b ran to 5642, so the boundary
        # overshot the certified rest and D'@5636 stayed subsumed.  Split
        # the span containing gap_b at gap_b BEFORE any span-derived
        # structure is built, making the certified boundary expressible
        # exactly.  Arrays are REBUILT, never mutated (S1 input purity);
        # env absent => bridge_pin None => untouched.  Fail-soft + typed
        # receipt; on refusal the lattice boundary stands (visible in the
        # engagement receipt's window_frames-vs-gap comparison).
        try:
            _bp_split = _bridge_pin_split_gap_span(
                int(bridge_pin["gap"][1]),
                meta=meta, layer_reads=layer_reads,
                layer_sub_frames=layer_sub_frames,
                evidence_reads=evidence_reads,
                evidence_sub_frames=evidence_sub_frames,
                evidence_provenance=evidence_provenance,
                onset_clusters_per_span=onset_clusters_per_span,
                covT_per_span=covT_per_span,
                purity_per_span=purity_per_span,
                gate_flags_per_span=gate_flags_per_span,
                margin_per_span=margin_per_span,
                move_layer=move_layer,
                certified_rest_spans=certified_rest_spans)
            bridge_pin_stats["span_split"] = _bp_split["receipt"]
            if _bp_split["applied"]:
                meta = _bp_split["meta"]
                layer_reads = _bp_split["layer_reads"]
                layer_sub_frames = _bp_split["layer_sub_frames"]
                evidence_reads = _bp_split["evidence_reads"]
                evidence_sub_frames = _bp_split["evidence_sub_frames"]
                evidence_provenance = _bp_split["evidence_provenance"]
                onset_clusters_per_span = _bp_split[
                    "onset_clusters_per_span"]
                covT_per_span = _bp_split["covT_per_span"]
                purity_per_span = _bp_split["purity_per_span"]
                gate_flags_per_span = _bp_split["gate_flags_per_span"]
                margin_per_span = _bp_split["margin_per_span"]
                move_layer = _bp_split["move_layer"]
                certified_rest_spans = _bp_split["certified_rest_spans"]
            print(f"  [scrub-bridge-pin] span-split "
                  f"{json.dumps(_bp_split['receipt'], sort_keys=True)}",
                  flush=True)
        except Exception as exc:       # fail-soft: lattice boundary stands
            bridge_pin_stats["span_split"] = dict(
                status=f"failsoft:{type(exc).__name__}: {exc}")
            print(f"  [scrub-bridge-pin] span-split FAILSOFT "
                  f"(lattice boundary kept): {type(exc).__name__}: {exc}",
                  flush=True)
    # WINDOW AUDIT (CUBED_SCRUB_WINDOW_AUDIT; _load_window_audit has the
    # schema + receipts-only contract).  Default absent/empty =>
    # byte-identical: no key, no branch, no report field (state_rank
    # surfacing pattern).  Load is fail-soft like the pin's.
    _window_audit_src = os.environ.get("CUBED_SCRUB_WINDOW_AUDIT")
    _window_audit_capability = "window-audit-v2"
    window_audit = None
    window_audit_stats = {
        "requested": bool(_window_audit_src), "armed": False,
        "capability": _window_audit_capability,
        "matched": 0, "out": None, "failsoft": None}
    if _window_audit_src:
        window_audit, _wa_reason = _load_window_audit(
            _window_audit_src, tag=tag)
        if window_audit is None:
            window_audit_stats["failsoft"] = f"load:{_wa_reason}"
            print(f"  [scrub-window-audit] FAILSOFT (no audit written): "
                  f"{_wa_reason} src={_window_audit_src}", flush=True)
        else:
            window_audit_stats["armed"] = True
            window_audit_stats["out"] = window_audit["out"]
    view = SSV.build_scrub_span_view(
        moves=moves, meta=meta, n_spans=len(layer_reads),
        n_bridge=n_bridge)
    n_sp, n_solve = view["n_sp"], view["n_solve"]
    if n_solve <= 0 or n_sp == 0:
        report["status"] = "no-solve-region"
        return moves, move_layer, move_oms, report
    # Exact read timestamps are required to place orientation changes between
    # aligned sticker observations.  A single read can safely inherit its
    # span's end frame for older direct callers; multiple reads cannot be
    # ordered honestly without the tracker's parallel layer_sub_frames stream.
    read_frames = []
    read_frame_error = None
    supplied_frames = layer_sub_frames is not None
    for si in range(n_sp):
        reads_si = list(layer_reads[si])
        if supplied_frames:
            try:
                frames_si = [int(f) for f in layer_sub_frames[si]]
            except (IndexError, TypeError, ValueError):
                read_frame_error = f"invalid read-frame stream at span {si}"
                frames_si = []
            if len(frames_si) != len(reads_si):
                read_frame_error = f"read/frame length mismatch at span {si}"
        elif len(reads_si) <= 1:
            frames_si = ([int(view["meta_f"](si)[1])] if reads_si else [])
        else:
            frames_si = []
            read_frame_error = f"multiple reads lack timestamps at span {si}"
        read_frames.append(frames_si)
    score_layer_reads = layer_reads
    score_read_frames = read_frames
    if dense_evidence:
        if len(evidence_reads) != n_sp or len(evidence_sub_frames) != n_sp:
            raise ValueError("dense scrub evidence span count mismatch")
        if (evidence_provenance is not None
                and len(evidence_provenance) != n_sp):
            raise ValueError("dense scrub evidence provenance count mismatch")
        score_layer_reads = evidence_reads
        score_read_frames = []
        provenance_rows = []
        for si in range(n_sp):
            reads_si = list(score_layer_reads[si])
            try:
                frames_si = [int(f) for f in evidence_sub_frames[si]]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid dense read-frame stream at span {si}") from exc
            if len(frames_si) != len(reads_si):
                raise ValueError(
                    f"dense read/frame length mismatch at span {si}")
            score_read_frames.append(frames_si)
            supplied = (dict(evidence_provenance[si])
                        if evidence_provenance is not None else {})
            supplied.update(
                span=int(si),
                control_count=int(len(layer_reads[si])),
                evidence_count=int(len(reads_si)))
            provenance_rows.append(supplied)
        report["dense_evidence"] = {
            "enabled": True,
            "control_counts": [int(len(reads)) for reads in layer_reads],
            "evidence_counts": [int(len(reads))
                                for reads in score_layer_reads],
            "spans": provenance_rows,
            # Reuse bridge_search's existing score-state capacity as a
            # deterministic read x state work guard.  The orientation axis is
            # cube geometry and therefore does not need another threshold.
            "scale_guard": {
                "read_state_bound": int(BS.SCORE_STATE_BOUND),
                "max_read_state_cells": 0,
                "tripped": False,
                "checks": 0,
            },
        }
    dense_prefix_active = bool(dense_prefix and dense_evidence)
    if dense_prefix_active:
        report["dense_prefix"] = {
            "enabled": True,
            "plane": "independent chronological dense-evidence prefix beam",
            "candidate_policy": (
                "control survivors union dense survivors; scores never blend; "
                "shared words keep control candidate metadata and the union of "
                "control+dense final-frontier timings, dense-only words keep "
                "dense metadata/timings; common final evidence rescores every "
                "retained timing; each plane owns its own 2sigma band and "
                "K-per-OM cut"),
            "structure_policy": (
                "same immutable typed slots, hard motion count, [L_lo,L_hi], "
                "and OM graph as control; independent dense union is "
                "suppressed for seeded continuations and every optional typed "
                "slot so every retained word has complete physical provenance"),
            "failure_policy": (
                "any dense-prefix bound or scoring failure drops only the "
                "dense plane; the finalized control generator is retained"),
            "state_bound": int(BS.SCORE_STATE_BOUND),
            "attempts": 0,
            "successes": 0,
            "fallbacks": 0,
            "control_words": 0,
            "dense_words": 0,
            "union_words": 0,
            "rescued_words": 0,
            "typed_rescue_suppressions": 0,
            "optional_slot_suppressions": 0,
            "seeded_suppressions": 0,
            "final_rollbacks": 0,
            "max_frontier": 0,
            "max_expansion_cells": 0,
            "max_score_cells": 0,
        }

    def dense_grid_guard(n_reads, n_states, *, context):
        """Record one dense read×state admission check.

        The final candidate scorer treats rejection as a transaction failure;
        an evidence-local optional visual scorer may instead fall back to its
        control rows.  Keeping one seam guarantees both paths use the same
        capacity and makes every decision visible in the dynamic sidecar row.
        """
        if not dense_evidence:
            return False, 0, int(BS.SCORE_STATE_BOUND)
        admissible, cells, bound = _dense_grid_admissible(n_reads, n_states)
        guard = report["dense_evidence"]["scale_guard"]
        guard["checks"] = int(guard["checks"]) + 1
        if int(cells) >= int(guard["max_read_state_cells"]):
            guard["max_read_state_cells"] = int(cells)
            guard["max_context"] = str(context)
        if not admissible:
            guard["tripped"] = True
            guard.setdefault("trips", []).append({
                "context": str(context),
                "read_state_cells": int(cells),
            })
        return bool(admissible), int(cells), int(bound)

    def record_dense_control_fallback(context, error=None):
        """Record one evidence-plane transaction that retried on control."""
        dense = report["dense_evidence"]
        dense["control_fallbacks"] = int(
            dense.get("control_fallbacks", 0)) + 1
        if error is None:
            return
        dense["scoring_failures"] = int(
            dense.get("scoring_failures", 0)) + 1
        dense.setdefault("failures", []).append({
            "context": str(context),
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
    events = sorted(
        IBM.require_exact_integer(frame, "scrub rotation-event frame")
        for frame in rot_event_frames)
    gate_streams, gate_reason = _stateful_gate_streams(gate_events, events)
    if intraburst_phase_slots and gate_streams is not None:
        producer_events = tuple(
            IBM.require_exact_integer(frame, "admitted authoritative event")
            for frame in intraburst_phase_receipt["producer"].get(
                "authoritative_events", ()))
        if producer_events != tuple(gate_streams["move"]):
            intraburst_phase_slots = ()
            intraburst_phase_receipt.update(
                enabled=False,
                admission_status="disabled",
                admission_reason=(
                    "producer authoritative events do not match the complete "
                    "kept final-gate event stream"),
                slot_count=0,
                optional_action_rows=0,
            )
    stateful_nbrs = (om_adjacency(orientations, normalize_names=True)
                     if om_stateful and gate_streams is not None else None)
    if om_stateful and gate_streams is None:
        report.update(om_stateful=False, om_stateful_reason=gate_reason)
        raise ValueError(f"stateful OM unavailable: {gate_reason}")
    if om_stateful and stateful_nbrs is None:
        report.update(om_stateful=False,
                      om_stateful_reason="invalid-orientation-geometry")
        raise ValueError("stateful OM unavailable: invalid orientation geometry")
    if om_stateful and read_frame_error is not None:
        report.update(om_stateful=False, om_stateful_reason=read_frame_error)
        raise ValueError(f"stateful OM unavailable: {read_frame_error}")
    om_stateful_active = bool(om_stateful)
    gate_drop_slots_active = bool(
        gate_drop_slots and om_stateful_active and gate_streams is not None
        and gate_streams["rotation_intervals"])
    gate_drop_intervals = tuple(
        dict(interval) for interval in (
            gate_streams["contested_intervals"]
            if gate_drop_slots_active else ()))
    gate_drop_all_intervals = tuple(
        dict(interval) for interval in (
            gate_streams["rotation_intervals"]
            if gate_drop_slots_active else ()))
    if gate_drop_slots_active:
        verdict_counts = dict(gate_streams["gate_verdict_counts"])
        report["gate_drop_slots"] = {
            "enabled": True,
            "producer": (
                "tier1 authoritative alignment-veto disagreement; tier2 "
                "all final drops with one transaction-wide SINGLE"),
            "final_dropped_records": int(
                verdict_counts["dropped_records"]),
            "optional_contested_records": int(
                verdict_counts["contested_records"]),
            "optional_alignment_veto_records": int(
                verdict_counts["align_regrip_only_records"]),
            "color_veto_rotation_only_records": int(
                verdict_counts["color_regrip_only_records"]),
            "rotation_only_records": int(
                verdict_counts["dropped_records"]
                - verdict_counts["contested_records"]),
            "verdict_counts": verdict_counts,
            "coalesced_rotation_gaps": int(len(
                gate_streams["rotation_intervals"])),
            "coalesced_gap_slots": int(len(gate_drop_intervals)),
            "tier2_coalesced_gap_slots": int(len(gate_drop_all_intervals)),
            "contested_gap_provenance": list(
                gate_streams["contested_gap_provenance"]),
            "action_alphabet": "SKIP + 18 SINGLE; no DOUBLE",
            "hard_count_budget": "unchanged",
            "ownership": "window_start < exact_gap_representative <= window_end",
            "coalescing": (
                "exact final-gate structural gap; any authoritative "
                "color=False/align=True record makes the gap contested"),
            "eligibility": (
                "dropped is true, color_regrip is exactly False, and "
                "align_regrip is exactly True"),
            "tier2_eligibility": (
                "after tier1 cannot adopt: every authoritative final-dropped "
                "gap remains SKIP/re-grip; at most one gap across the complete "
                "rescue transaction may instead emit one SINGLE"),
            "activation": (
                "primary search receives no optional drop slots; rerun the "
                "same envelope only after unresolved; tier1 runs first, then "
                "the globally one-SINGLE tier2; a nondecisive tier1 final "
                "frontier is unioned with tier2 before either adopts only "
                "commit/low_conf at the unchanged 2sigma gate"),
            "om_semantics": (
                "SKIP/regrip = OM stay or one existing neighbor; "
                "SINGLE/layer-move = OM fixed"),
            "evidence_policy": (
                "candidate-owned chronological control evidence; guarded "
                "dense rows only inside the exact owned gap; final dense "
                "candidate scoring reuses the physically typed trajectory "
                "instead of the relaxed whole-window OM DP"),
            "failure_policy": (
                "tier1 malformed/agreed/color-veto records stay rotation-only; "
                "tier2 exposes all drops only behind one global SINGLE cap; "
                "invalid core provenance disables stateful scrub; any tier2 "
                "bound/action-score or final-union provenance failure rejects "
                "the entire tier2 result and preserves the primary incumbent"),
            "slot_evaluations": 0,
            "max_step_expansions": 0,
            "bound_fallbacks": 0,
            "score_error_fallbacks": 0,
            "dense_final_typed_reuses": 0,
            "physical_path_rejections": 0,
            "rescue_attempts": 0,
            "rescue_adoptions": 0,
            "rescue_nondecisive": 0,
            "rescue_failures": 0,
            "rescue_wall_ms": 0,
            "tier1_attempts": 0,
            "tier1_adoptions": 0,
            "tier1_nondecisive": 0,
            "tier1_failures": 0,
            "tier1_wall_ms": 0,
            "tier2_attempts": 0,
            "tier2_adoptions": 0,
            "tier2_nondecisive": 0,
            "tier2_failures": 0,
            "tier2_wall_ms": 0,
            "tier_unions": 0,
            "tier_union_failures": 0,
            "tier_union_candidates": 0,
        }
    certified_rest_spans = {
        int(si) for si in (certified_rest_spans or ())
        if 0 <= int(si) < n_sp
    }
    if not om_stateful:
        om_stateful_reason = "not-requested"
    elif gate_streams is None:
        om_stateful_reason = gate_reason or "invalid-gate-events"
    elif stateful_nbrs is None:
        om_stateful_reason = "invalid-orientation-geometry"
    else:
        om_stateful_reason = None
    move_events = (gate_streams["move"] if om_stateful_active else events)
    rotation_events = (gate_streams["rotation"]
                       if om_stateful_active else events)
    # Stateful OM must not redefine scrub's canonical move/rest windows.
    # Dropped gaps are soft OM transitions *inside* those windows; using them
    # as hard rest boundaries changes L budgets and can delete real moves.
    rest_events = events
    timeline_events = (gate_streams["all"]
                       if om_stateful_active else events)
    report["om_stateful"] = om_stateful_active
    report["om_stateful_reason"] = om_stateful_reason
    report["om_stateful_counts"] = (
        dict(move=len(move_events), rotation=len(rotation_events),
             synthetic_move=len(gate_streams["synthetic_move"]),
             reads=sum(len(frames) for frames in score_read_frames),
             all=len(timeline_events)) if om_stateful_active else None)
    # CLUSTER FALLBACK: at a window that would otherwise finish
    # UNRESOLVED, re-search once with burst slots = event CLUSTERS
    # (_cluster_events: per-solve log-gap-valley linkage derivation,
    # no constant) and slack = the raw-vs-cluster count DISAGREEMENT
    # (channel-disagreement-derived, no constant). Rationale: the burst-
    # aligned beam trusts the event stream's count; on over/under-fired
    # streams (e.g. 100 events vs 76 moves) truth becomes UNENUMERABLE.
    # Default OFF ⇒ byte-identical.
    cluster_fallback = bool(os.environ.get("CUBED_SCRUB_CLUSTER_FALLBACK"))
    move_events_cl = (_cluster_events(move_events) if cluster_fallback
                      else list(move_events))

    # MICRO-REST CANDIDACY (CUBED_SCRUB_MICROREST_CANDIDACY; default
    # OFF => byte-identical). The raw event stream OVER-fires (e.g. 100 events
    # vs 76 moves), so a spurious re-fire INSIDE a genuine pause disqualifies
    # that pause as a window END (rests require ZERO raw events in-span). Expose
    # those hidden rests: a span also counts as a rest if it is event-free under
    # the CLUSTERED stream (_cluster_events: the log-gap-valley linkage,
    # intra-burst re-fires collapse left of the valley -- the SAME machinery the
    # cluster fallback uses, zero new constants). The has_reads gate is
    # UNTOUCHED (this exposes hidden rests, never manufactures evidence). The
    # clustered stream is computed here so the flag is independent of
    # CUBED_SCRUB_CLUSTER_FALLBACK; unused (empty) when the flag is off.
    microrest = bool(os.environ.get("CUBED_SCRUB_MICROREST_CANDIDACY"))
    events_cl_mr = _cluster_events(events) if microrest else []

    # LATE-EVIDENCE BAND (default OFF).
    # The ordinary chronological beam keeps its existing 2sigma evidence band
    # but defers the subsequent per-OM K cut so later reads may resolve an
    # uncertain prefix.  Scores, the evidence band, the configured K, and the
    # final commit gate are untouched.  BS.SCORE_STATE_BOUND remains the hard
    # work/memory stop; this is a bounded mechanism probe, not K widening.
    late_evidence_band = bool(
        os.environ.get("CUBED_SCRUB_LATE_EVIDENCE_BAND"))
    late_evidence_window_raw = os.environ.get(
        "CUBED_SCRUB_LATE_EVIDENCE_WINDOW")
    late_evidence_window = (None if late_evidence_window_raw is None else
                            int(late_evidence_window_raw))
    report["late_evidence_band"] = late_evidence_band
    report["late_evidence_window"] = late_evidence_window
    if late_evidence_band:
        print("  [scrub-late-evidence-band] ON (full existing 2sigma "
              "prefix band; per-OM K deferred; scores unchanged; "
              f"window={late_evidence_window})",
              flush=True)
    last_solve_span = min(n_sp - 1,
                          max((move_layer[i] for i in range(n_solve)),
                              default=n_sp - 1))
    # scan ceiling: rests CONFIRMING the last solve move live after its span;
    # spans carrying terminal-bridge moves are excluded because their states
    # are evaluation anchors and are never scrubbed.
    bridge_spans = [move_layer[i] for i in range(n_solve, len(moves))
                    if 0 <= move_layer[i] < n_sp]
    scan_hi = (min(bridge_spans) - 1) if bridge_spans else (n_sp - 1)
    scan_hi = max(scan_hi, last_solve_span)

    # ---- GT-TEACHER instruments (day-1 gates 2+3; EVAL-side, default OFF).
    # Both consume a movegt file as the TEACHER. Plumbed via kwarg OR env var:
    # a None kwarg falls back to the env var; absent/unloadable =>
    # the instrument never engages and the decode is BYTE-IDENTICAL.
    #   * CONTAINMENT AUDIT (--scrub-containment-audit / CUBED_SCRUB_CONTAINMENT_AUDIT):
    #     SHADOW. At every prune point it records whether the GT window
    #     trajectory survives; it NEVER touches candidate sets, scores,
    #     decisions, or emissions. Rows stream to a SEPARATE `.containment.jsonl`.
    #   * PERFECT-PERCEPTION ORACLE (--scrub-oracle-likelihood / CUBED_SCRUB_ORACLE_LL):
    #     EVAL ORACLE. It ADDS a per-candidate term (+0 on the GT trajectory,
    #     -band otherwise) at the pre-prune scoring seam; this DELIBERATELY
    #     changes the decode (the perfect-perception upper bound, never prod).
    _caud_src = (containment_audit if containment_audit is not None
                 else os.environ.get("CUBED_SCRUB_CONTAINMENT_AUDIT"))
    _orac_src = (oracle_likelihood if oracle_likelihood is not None
                 else os.environ.get("CUBED_SCRUB_ORACLE_LL"))
    caud_movegt = _load_movegt_entries(_caud_src) if _caud_src else []
    orac_movegt = _load_movegt_entries(_orac_src) if _orac_src else []
    # FRAME ATTRIBUTION FIX (a control run falsified raw-frame membership --
    # see _snap_movegt_to_events provenance): snap BLE movegt frames onto the
    # decoder's own detected event frames BEFORE window membership. ONE shared
    # mapping for the containment audit AND the oracle (same helper, same fix).
    def _gt_prefix(mg):
        """Drop non-HTM tokens (recorded) + build the GLOBAL GT prefix states
        P_0..P_n from the app-given init (decode-independent teacher states;
        P_k = init after the first k GT moves). Returns (clean, states, unmap)."""
        clean, unmap = [], []
        s = np.asarray(init_arr, np.int8).copy()
        states = [s.copy()]
        for f, mv in mg:
            try:
                mi = BS.MOVES.index(mv)
            except ValueError:
                unmap.append(mv)
                continue
            clean.append((f, mv))
            s = s[perms[mi]]
            states.append(s.copy())
        return clean, states, unmap

    _n_caud_moved = _n_orac_moved = 0
    caud_prefix = orac_prefix = None
    if caud_movegt:
        caud_movegt, caud_prefix, _cu = _gt_prefix(caud_movegt)
        report["containment_gt_unmappable"] = _cu
        snapped = _snap_movegt_to_events(caud_movegt, events)
        _n_caud_moved = sum(1 for (f0, _), (f1, _) in
                            zip(caud_movegt, snapped) if f0 != f1)
        caud_movegt = snapped
    if orac_movegt:
        orac_movegt, orac_prefix, _ou = _gt_prefix(orac_movegt)
        report["oracle_gt_unmappable"] = _ou
        snapped = _snap_movegt_to_events(orac_movegt, events)
        _n_orac_moved = sum(1 for (f0, _), (f1, _) in
                            zip(orac_movegt, snapped) if f0 != f1)
        orac_movegt = snapped
    containment_active = bool(caud_movegt)
    oracle_active = bool(orac_movegt)
    report["containment_audit"] = containment_active
    report["oracle_likelihood"] = oracle_active
    report["containment_gt_moves"] = len(caud_movegt)
    report["oracle_gt_moves"] = len(orac_movegt)
    report["containment_frames_snapped"] = _n_caud_moved
    report["oracle_frames_snapped"] = _n_orac_moved
    # SHADOW containment rows go to their OWN jsonl (never the decode sidecar).
    caud_fh = None
    if containment_active:
        caud_path = ((sidecar_path + ".containment.jsonl") if sidecar_path
                     else f"/tmp/scrub_{tag or 'run'}.containment.jsonl")
        try:
            caud_fh = open(caud_path, "w")
            report["containment_audit_path"] = caud_path
        except OSError as e:                            # IO never aborts scrub
            report["containment_audit_error"] = f"{type(e).__name__}: {e}"
            containment_active = False

    caud_transaction_buffer = None

    def caud_row(obj):
        if caud_transaction_buffer is not None:
            caud_transaction_buffer.append(copy.deepcopy(obj))
            return
        if caud_fh is None:
            return
        try:
            caud_fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
            caud_fh.flush()
        except OSError:                                 # shadow: swallow IO err
            pass

    # TRUTH PROBE (--truth-probe / CUBED_TRUTH_PROBE; roadmap #4). RECORD-ONLY
    # dev-ceiling instrument: unlike the containment audit it does NOT gate the
    # device beam off -- it watches truth's (state,om) lineage INSIDE device-
    # beam windows (rank + score margin at every band prune / K cap, killer
    # attribution, om diversity of survivors) and streams rows to its OWN
    # `.truth_probe.jsonl`. It never touches candidate sets, scores, decisions,
    # or emissions; env var unset => zero report keys => byte-identical to off.
    # The movegt is a DEV-CEILING teacher, never a production input.
    # IN-BEAM STATE RANK counters (CUBED_BEAM_STATE_RANK).
    # Always allocated (cheap); surfaced in report/summary ONLY when the flag
    # is set, so an off run's outputs are byte-identical.
    state_rank_stats = {
        "device_windows_active": 0, "window_bins_total": 0,
        "bursts_biased": 0, "rank_nats_l1": 0.0,
        "window_errors": 0, "last_error": None}
    _tprobe_src = os.environ.get("CUBED_TRUTH_PROBE")
    _tprobe_capability = "truth-probe-v2.1"
    tprobe_movegt = _load_movegt_entries(_tprobe_src) if _tprobe_src else []
    tprobe_prefix = None
    tprobe_movegt_raw = []
    if tprobe_movegt:
        tprobe_movegt, tprobe_prefix, _tpu = _gt_prefix(tprobe_movegt)
        report["truth_probe_gt_unmappable"] = _tpu
        # The RAW (pre-snap) frames are the truth-probe-v2 boundary
        # currency (`_window_truth_v2`'s direct half-open membership) -- the
        # snapped copy below stays ONLY for `_gt_prefix`'s index-aligned
        # `tprobe_prefix` array (snap-invariant: snapping rewrites frames,
        # never move order/identity) and any v1-shaped receipt fields.
        tprobe_movegt_raw = list(tprobe_movegt)
        _tp_snapped = _snap_movegt_to_events(tprobe_movegt, events)
        report["truth_probe_frames_snapped"] = sum(
            1 for (f0, _), (f1, _) in zip(tprobe_movegt, _tp_snapped)
            if f0 != f1)
        tprobe_movegt = _tp_snapped
    truth_probe_active = bool(tprobe_movegt)
    # Optional override: `--truth-boundaries` / CUBED_TRUTH_BOUNDARIES
    # (truth-boundaries-v1 schema).  Loaded whenever the probe itself is
    # requested, independent of the (rare) case that this decode's own
    # movegt has zero mappable moves; keyed by (start_frame, end_frame) so
    # each window looks itself up.  Fail-soft, receipted, never authoritative
    # unless a window's own frame span is actually present in the file.
    _tboundaries_src = os.environ.get("CUBED_TRUTH_BOUNDARIES")
    tprobe_boundaries = {}
    if _tprobe_src and _tboundaries_src:
        tprobe_boundaries, _tb_movegt_sha, _tb_reason = (
            _load_truth_boundaries_v2(_tboundaries_src))
        report["truth_boundaries_loaded"] = bool(tprobe_boundaries)
        report["truth_boundaries_movegt_sha256"] = _tb_movegt_sha
        if _tb_reason:
            report["truth_boundaries_error"] = _tb_reason
    tprobe_fh = None
    if _tprobe_src:
        report["truth_probe"] = truth_probe_active
        report["truth_probe_capability"] = _tprobe_capability
        report["truth_probe_gt_moves"] = len(tprobe_movegt)
        # Spec item 4, receipted at the run level (per-window echo at
        # each window's own "window_root" sidecar row): while armed, the
        # device beam's cross-call checkpoint/frontier cache is bypassed for
        # every window so no stage receipt is ever skipped by a resume.
        # Performance-only -- off (env unset) is the untouched `False` a
        # byte-identical run always reports.
        report["truth_probe_checkpoint_bypass"] = truth_probe_active
    if truth_probe_active:
        tprobe_path = ((sidecar_path + ".truth_probe.jsonl") if sidecar_path
                       else f"/tmp/scrub_{tag or 'run'}.truth_probe.jsonl")
        try:
            tprobe_fh = open(tprobe_path, "w")
            report["truth_probe_path"] = tprobe_path
        except OSError as e:                            # IO never aborts scrub
            report["truth_probe_error"] = f"{type(e).__name__}: {e}"
            truth_probe_active = False

    tprobe_transaction_buffer = None

    def tprobe_row(obj):
        if tprobe_transaction_buffer is not None:
            tprobe_transaction_buffer.append(copy.deepcopy(obj))
            return
        if tprobe_fh is None:
            return
        try:
            tprobe_fh.write(
                json.dumps(obj, sort_keys=True, default=str) + "\n")
            tprobe_fh.flush()
        except OSError:                                 # record-only: swallow
            pass

    # DEFECT FIX (truth-probe ⊥ containment-audit): the containment
    # audit disables the device beam for EVERY window, and the probe only
    # instruments device-beam bursts -- an audited probe run records ZERO
    # burst rows (window_cpu_path markers only).  That combination must never
    # be silent: warn unmissably on stdout, stamp the report, and write a
    # leading sidecar marker row.  Record-only in both directions -- neither
    # instrument's behavior changes.
    if truth_probe_active and containment_active:
        print("  [truth-probe] WARNING: containment audit is ACTIVE -> "
              "device beam is disabled for EVERY window; NO burst rows will "
              "be recorded in the .truth_probe.jsonl sidecar (window_cpu_path "
              "markers only). Re-run without the containment audit for burst "
              "instrumentation.", flush=True)
        report["truth_probe_burst_rows_unavailable"] = "containment-audit"
        tprobe_row(dict(
            kind="probe_burst_rows_unavailable", tag=tag,
            capability=_tprobe_capability,
            reason="containment-audit-forces-cpu-path",
            detail="containment audit disables the device beam; the probe "
                   "instruments device-beam bursts only -- zero burst rows "
                   "will follow"))

    # M1 mid-motion rows -> per-span read pool by FRAME (the documented seam:
    # rows are merged into the per-window read pool). Attach each row to the span
    # whose [f0,f1] contains its frame; rows in between-span gaps drop + count
    # (honest). `read` is the geo_read raw-read shape ([(slot, 9x3)...]) that
    # AbsSegment consumes directly (RAW LAB); passed through as-is (the row's own
    # object is a stable seg_factory cache key across orientations).
    mm_by_span = {}
    n_mm_dropped = 0
    if midmotion_rows:
        span_lo = [int(view["meta_f"](si)[0]) for si in range(n_sp)]
        span_hi = [int(view["meta_f"](si)[1]) for si in range(n_sp)]
        for row in midmotion_rows:
            si = _frame_to_span(int(row.get("frame", -1)), span_lo, span_hi)
            read_obj = row.get("read")
            # read may be an ndarray (state-vector rows) or a read-entry list —
            # bare truthiness on an ndarray raises; test emptiness by length.
            if si is None or read_obj is None or len(read_obj) == 0:
                n_mm_dropped += 1
                continue
            mm_by_span.setdefault(si, []).append(
                (int(row.get("frame")), read_obj, row.get("cellconf") or {}))
    report["midmotion_rows_used"] = sum(len(v) for v in mm_by_span.values())
    report["midmotion_rows_dropped"] = n_mm_dropped

    seg_cache = {}
    read_seg_cache = {}
    evidence_read_seg_cache = {}
    mm_seg_cache = {}
    mm_one_cache = {}
    # Populated only by the default-off device-controller experiment.  These
    # caches live for one monotone window start so extension retries reuse
    # compiled read tensors and retained frontiers without carrying unreachable
    # CUDA allocations into the next committed cursor.
    device_read_run_cache = {}
    device_beam_runtime = {}
    device_beam_checkpoint_cache = {}
    # Retained device frontiers and compiled read plans are useful only while
    # the scrub cursor is extending one fixed window start.  A prior cursor can
    # never be revisited by the monotone main loop, so keeping its CUDA tensors
    # would add memory without preserving a hypothesis.  Evict both plane
    # namespaces together before the first call at a new ``a_span``; control
    # and dense retries at that cursor then continue to share their own exact
    # checkpoint families without evicting each other.
    device_beam_window_runtime = dict(
        a_span=None, evictions=0, checkpoint_families_peak=0,
        read_run_plans_peak=0)
    # Exact CPU fallback for the independent dense-prefix plane.  The ordinary
    # production path now uses its separately namespaced resident device beam;
    # if CUDA is unavailable or rejects a foreign scorer, this cache prevents
    # the transactionally restarted CPU plane from rebuilding every extension
    # from slot zero.  Eligibility below excludes every lane with observer side
    # effects or uncertified provenance, and the cache resets when the monotone
    # window cursor advances.
    dense_beam_checkpoint_cache = {}
    dense_beam_checkpoint_runtime = dict(
        a_span=None, calls=0, resumed_calls=0,
        slots_total=0, slots_skipped=0)

    def read_seg(si, ri, oi):
        """One control-stream read at its frame/orientation (cached)."""
        key = (si, ri, oi)
        if key not in read_seg_cache:
            s = seg_factory(layer_reads[si][ri], orientations[oi])
            read_seg_cache[key] = s if getattr(s, "ok", False) else None
        return read_seg_cache[key]

    def score_read_seg(si, ri, oi):
        """Candidate-conditioned scoring read; aliases control when dense is off."""
        if not dense_evidence:
            return read_seg(si, ri, oi)
        key = (si, ri, oi)
        if key not in evidence_read_seg_cache:
            s = seg_factory(score_layer_reads[si][ri], orientations[oi])
            evidence_read_seg_cache[key] = (
                s if getattr(s, "ok", False) else None)
        return evidence_read_seg_cache[key]

    # One immutable identity token certifies the production dense-prefix source
    # without granting the ordinary control chunk program direct access to its
    # frame/scorer names (the dense-read ownership contract).
    dense_checkpoint_source = (score_read_frames, score_read_seg)

    def segs_om(si, oi):
        key = (si, oi)
        if key not in seg_cache:
            out = ([read_seg(si, ri, oi)
                    for ri in range(len(layer_reads[si]))]
                   if 0 <= si < n_sp else [])
            seg_cache[key] = [s for s in out if s is not None]
        return seg_cache[key]

    def mm_segs(si, oi):
        """Mid-motion segments for span si at orientation oi (M1 evidence). Built
        through the SAME seg_factory (palette / CC / currency) as still reads,
        then per-cell reweighted by the row's STABILITY conf via the CONF_W floor
        (_mm_apply_weight). Empty unless --midmotion-reads supplied rows."""
        if not mm_by_span:
            return []
        key = (si, oi)
        if key not in mm_seg_cache:
            mm_seg_cache[key] = [
                s for mi in range(len(mm_by_span.get(si, [])))
                if (s := mm_seg_one(si, mi, oi)) is not None]
        return mm_seg_cache[key]

    def mm_seg_one(si, mi, oi):
        key = (si, mi, oi)
        if key not in mm_one_cache:
            _frame, read_obj, cellconf = mm_by_span[si][mi]
            s = seg_factory(read_obj, orientations[oi])
            if getattr(s, "ok", False):
                _mm_apply_weight(s, cellconf)
                mm_one_cache[key] = s
            else:
                mm_one_cache[key] = None
        return mm_one_cache[key]

    def segs_score(si, oi):
        """Evidence pool for the WINDOW read-fit = still reads + mid-motion reads
        (M1). Candidacy / has_reads / coverage / argmax_om stay still-reads-
        only (segs_om) -- still-reads semantics untouched; mid-motion is EVIDENCE."""
        mm = mm_segs(si, oi)
        return (segs_om(si, oi) + mm) if mm else segs_om(si, oi)

    def marg_row(si, smat, allowed=None):
        """max over ALLOWED orientations of the mean read-fit, per candidate
        state. allowed=None => full marginalization (window 0 / continuity
        disabled); otherwise the om-continuity ball around the carried om.
        Consumes segs_score = still + mid-motion reads (M1)."""
        ois = (range(len(orientations)) if allowed is None
               else sorted(allowed))
        rows = []
        for oi in ois:
            segs = segs_score(si, oi)
            if segs:
                rows.append(np.mean([s.score_states(smat) for s in segs],
                                    axis=0))
        return np.max(np.stack(rows), axis=0) if rows else None

    def has_reads(si):
        return any(segs_om(si, oi) for oi in range(len(orientations)))

    def coverage(si):
        """Geometry-derived read coverage (# observed sticker positions,
        maxed over orientations). An ORDERING, never a gate."""
        best = 0
        for oi in range(len(orientations)):
            segs = segs_om(si, oi)
            if segs:
                best = max(best,
                           sum(len(getattr(s, "_ar", ())) for s in segs))
        return best

    def argmax_om(si, state, allowed=None):
        best, boi = None, None
        st = np.asarray(state, np.int8)[None, :]
        ois = (range(len(orientations)) if allowed is None
               else sorted(allowed))
        for oi in ois:
            segs = segs_om(si, oi)
            if not segs:
                continue
            v = float(np.mean([s.score_states(st) for s in segs], axis=0)[0])
            if best is None or v > best:
                best, boi = v, oi
        return boi

    def rank_oms(si, state, allowed=None):
        """Read-fit ranking of an exact state over candidate orientations."""
        st = np.asarray(state, np.int8)[None, :]
        ois = (range(len(orientations)) if allowed is None
               else sorted(allowed))
        ranked = []
        for oi in ois:
            segs = segs_om(si, oi)
            if not segs:
                continue
            score = float(np.mean([s.score_states(st) for s in segs], axis=0)[0])
            ranked.append((oi, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def rank_oms_many(spans, state, allowed=None):
        """Aggregate every available opening read; no single-span veto."""
        st = np.asarray(state, np.int8)[None, :]
        ois = (range(len(orientations)) if allowed is None
               else sorted(allowed))
        ranked = []
        for oi in ois:
            segs = [seg for si in spans for seg in segs_om(si, oi)]
            if segs:
                score = float(np.mean(
                    [seg.score_states(st) for seg in segs], axis=0)[0])
                ranked.append((oi, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def f1_of(si):
        return view["meta_f"](si)[1] if si >= 0 else view["meta_f"](0)[0] - 1





    def sees_end(si, b_span):
        """True when span si's whole read range lies AFTER the window's last
        motion event, i.e. its reads witness the window END state."""
        f0 = view["meta_f"](si)[0]
        return _events_in(rest_events, f0 - 1, f1_of(b_span)) == 0

    # ---- rest candidates -> zero-event GROUPS (rep = best coverage) --------
    # GEOMETRY-ONLY candidacy: every event-free span with any readable segment
    # is a candidate window end. The 2-sigma window decision remains the gate;
    # candidacy does not pre-gate it.
    rests = [si for si in range(scan_hi + 1)
             if has_reads(si)
             and _events_in(rest_events, view["meta_f"](si)[0] - 1,
                            f1_of(si)) == 0]
    if microrest:
        # MICRO-REST CANDIDACY: additionally admit spans that are
        # event-free under the CLUSTERED stream (a spurious raw re-fire inside a
        # genuine pause no longer disqualifies it). Same has_reads gate; merged
        # BEFORE the group-merge loop so all downstream logic runs unchanged.
        cluster_rests = [si for si in range(scan_hi + 1)
                         if has_reads(si)
                         and _events_in(events_cl_mr, view["meta_f"](si)[0] - 1,
                                        f1_of(si)) == 0]
        rests = sorted(set(rests) | set(cluster_rests))
    groups = []                                    # dict(rep, last)
    for si in rests:
        if groups and _events_in(rest_events, f1_of(groups[-1]["last"]),
                                 f1_of(si)) == 0:
            g = groups[-1]
            g["last"] = si
            if coverage(si) > coverage(g["rep"]):
                g["rep"] = si
        else:
            groups.append(dict(rep=si, last=si))

    # sigma for the COMMIT GATE (state margin >= Z_SIGMA x sigma) and
    # for sidecar margin_sigma: std of the per-span fit population.
    fitpop = [f for f in view["fit"] if f is not None]
    sigma = float(np.std(fitpop)) if fitpop else 0.0
    sigma = sigma if sigma > 0 else 1.0

    # OM CONTINUITY structure (None when the orientation list is not a valid
    # face-pair set -- unit fixtures -- in which case marginalization stays
    # full and the header records continuity=off).
    om_nbrs = (stateful_nbrs if om_stateful_active
               else om_adjacency(orientations))

    def committed_window(a_span, b_span):
        toks = [moves[i] for i in range(n_solve)
                if a_span < move_layer[i] <= b_span]
        return BS.reduce_word(toks)

    def committed_window_oms(a_span, b_span):
        """Original per-move OMs when reduction preserved token identity."""
        idx = [i for i in range(n_solve)
               if a_span < move_layer[i] <= b_span]
        toks = [moves[i] for i in idx]
        reduced = BS.reduce_word(toks)
        if list(reduced) != toks:
            return None
        return [move_oms[i] for i in idx]

    def _gt_split(si, movegt, prefix_states):
        """GT-sequence split index at boundary rest span `si`: how many GT
        moves precede the rest. k0 = frame-suggested split (#moves with
        event-aligned frame <= f1_of(si)); REFINED by scoring the GT prefix
        states P_{k0-1}, P_k0, P_{k0+1} against the rest span's OWN reads
        (om-marginalized marg_row -- teacher states x raw measurements, no
        decoder output) and taking the best fit, k0 preferred on ties.
        FALSIFIED-BY-CONTROL provenance (a control run, v2): the event channel alone
        cannot place a boundary move when bursts are MISSED (a window: L_est=8
        detected events for 10 true moves; F'@8103's true burst is the next
        window's first) -- but the boundary REST's reads witness which side
        the move fell on (the same physics the scrub commits on). +/-1 = the
        system's own SLACK currency, not a new knob. si < 0 (video start)
        => k0 unrefined (no reads before the first span)."""
        f_hi = f1_of(si)
        k0 = sum(1 for f, _mv in movegt if f <= f_hi)
        if si < 0:
            return k0, k0
        ks = [k for k in (k0, k0 - 1, k0 + 1)
              if 0 <= k < len(prefix_states)]
        if len(ks) <= 1:
            return k0, k0
        try:                        # teacher input must never abort the scrub
            row = marg_row(si, np.stack([prefix_states[k] for k in ks]),
                           None)
        except Exception:                              # noqa: BLE001
            row = None
        if row is None:
            return k0, k0
        # ks is ordered [k0, ...] and argmax takes the FIRST max => k0 wins
        # ties (the frame attribution stands unless the reads disagree).
        return ks[int(np.argmax(row))], k0

    def _window_truth(w_state, a_span, b_span, movegt, prefix_states):
        """GT trajectory for window (a_span, b_span]: the GT moves between the
        boundary splits k(a_span) and k(b_span) (_gt_split: event-aligned
        frames + rest-read refinement), canonical-reduced. The candidate-match
        trajectory is applied to `w_state` (the SCRUB's own window-start state
        -- candidates live there); `start_matches_gt` records whether that
        state equals the GLOBAL GT prefix state P_{k_lo} (False = the truth
        was excluded upstream by start-state drift, the drift signal -- decomposes
        containment failures into pruned-here vs drifted-before). Returns None
        when movegt is empty."""
        if not movegt:
            return None
        k_lo, k0_lo = _gt_split(a_span, movegt, prefix_states)
        k_hi, k0_hi = _gt_split(b_span, movegt, prefix_states)
        k_hi = max(k_lo, k_hi)
        physical_toks = [mv for _f, mv in movegt[k_lo:k_hi]]
        toks = BS.reduce_word(physical_toks)
        states, end_b, unmap = _truth_trajectory(w_state, toks, perms)
        return dict(word=list(toks), word_nf=list(BS.normal_form(toks)),
                    physical_tokens=list(physical_toks),
                    physical_word=[BS.MOVES.index(tok)
                                   for tok in physical_toks],
                    physical_word_len=len(physical_toks),
                    traj_bytes={s.tobytes() for s in states}, end_bytes=end_b,
                    word_len=len(toks), unmappable=unmap,
                    k_lo=k_lo, k_hi=k_hi,
                    split_delta=[k_lo - k0_lo, k_hi - k0_hi],
                    start_matches_gt=bool(np.array_equal(
                        np.asarray(w_state, np.int8),
                        prefix_states[k_lo])))

    def _emit_caud(win_ctx, a_span, b_span, prune_kind, burst_k, n_before,
                   n_after, truth, survived, end_surv, rank_before, extra=None):
        """Write ONE shadow containment row for a (window, prune-point). Keyed
        by window idx/attempt (when known) + span/frame span so a consumer can
        join it to the decode sidecar's window rows."""
        row = dict(kind="containment", tag=tag,
                   window_idx=(win_ctx[0] if win_ctx else None),
                   attempt=(win_ctx[1] if win_ctx else None),
                   span_a=a_span, span_b=b_span,
                   frames=[f1_of(a_span), f1_of(b_span)],
                   prune_kind=prune_kind, burst_k=burst_k,
                   n_before=int(n_before), n_after=int(n_after),
                   truth_survived=bool(survived),
                   truth_end_survived=(None if end_surv is None
                                       else bool(end_surv)),
                   truth_rank_before=(None if rank_before is None
                                      else int(rank_before)),
                   gt_word=truth["word"], gt_word_nf=truth["word_nf"],
                   gt_word_len=truth["word_len"],
                   gt_unmappable=truth["unmappable"],
                   gt_k_lo=truth["k_lo"], gt_k_hi=truth["k_hi"],
                   gt_split_delta=truth["split_delta"],
                   start_matches_gt=truth["start_matches_gt"],
                   check_level="end-state",           # om/notation-agnostic
                   frame_attribution=("event-align+rest-read-split"))
        if extra:
            row.update(extra)
        caud_row(row)

    def _beam_generate_stateful(state, a_span, b_span, truth=None,
                                win_ctx=None, burst_frames=None,
                                skip_cap=SLACK, insert_cap=SLACK,
                                om_ctx=None, required_word=(),
                                seed_entries=None, visual_episodes=(),
                                dropped_intervals=(),
                                phase_slots=(),
                                gate_drop_single_cap=None,
                                word_capacity=None,
                                prefix_read_frames=None,
                                prefix_read_scorer=None,
                                prefix_plane_stats=None,
                                record_only_consume_provider_mode=None,
                                consume_plane_role=None,
                                terminal_complete_band=False,
                                physical_count_interval=None,
                                canonical_count_interval=False,
                                allow_partial_left_neutral_prefix=False,
                                burst_frames_are_raw=False):
        """Chronological state+OM prefix beam.

        A prefix is keyed by ``(cube state, OM, skip use, insert use,
        incumbent-prefix lane, dropped SINGLE use)``.  Reads therefore
        accumulate under one physically reachable OM history; a rival cannot
        splice its independently best OM at every read before the band/compute
        seam.  ``gate_drop_single_cap`` is a transaction-wide limit across all
        dropped gaps, never an independent per-gap branch budget.  Ordinary
        windows use ``beam_k`` states per OM; seeded computational seams retain
        their complete likelihood band and synchronously stream oversized
        per-step expansion work.

        The winning physical move timestamps are retained for final scoring.
        This is essential for onset-slotted count splits: a SKIP leaves the
        state unchanged, while both moves of a DOUBLE precede same-frame reads.
        """
        f_start, f_end = f1_of(a_span), f1_of(b_span)
        dense_prefix_plane = prefix_plane_stats is not None
        late_evidence_band_active = bool(
            late_evidence_band
            and not dense_prefix_plane
            and (late_evidence_window is None
                 or (win_ctx is not None
                     and int(win_ctx[0]) == int(late_evidence_window))))
        plane_read_frames = (read_frames if prefix_read_frames is None
                             else prefix_read_frames)
        plane_read_scorer = (read_seg if prefix_read_scorer is None
                             else prefix_read_scorer)
        count_interval = None
        if physical_count_interval is not None:
            try:
                count_lo, count_hi = map(int, physical_count_interval)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "physical count interval must contain two integers") from exc
            if count_lo < 0 or count_hi < count_lo:
                raise ValueError("physical count interval is invalid")
            count_interval = (count_lo, count_hi)
        canonical_count_interval = bool(canonical_count_interval)
        if canonical_count_interval and count_interval is None:
            raise ValueError(
                "canonical count currency requires a terminal interval")

        def plane_max(field, value):
            if prefix_plane_stats is not None:
                prefix_plane_stats[field] = max(
                    int(prefix_plane_stats.get(field, 0)), int(value))

        def plane_memory_guard(field, value, context):
            """Bound resident dense-prefix work, not cumulative read work."""
            plane_max(field, value)
            if ((dense_prefix_plane or terminal_complete_band)
                    and int(value) > int(BS.SCORE_STATE_BOUND)):
                raise RuntimeError(
                    f"bounded prefix {context} {int(value)} > "
                    f"{int(BS.SCORE_STATE_BOUND)}")
        src = events if burst_frames is None else burst_frames
        bursts = sorted(int(f) for f in src if f_start < int(f) <= f_end)
        raw_burst_stream = bool(
            burst_frames is None or burst_frames_are_raw)
        action_slots = typed_transition_slots(
            bursts, visual_episodes, f_start, f_end,
            dropped_intervals=dropped_intervals,
            intraburst_phase_slots=phase_slots)
        visual_slots_active = any(
            slot.kind != "motion" for slot in action_slots)
        # CUBED_SCRUB_SLOT_MERGE. Scoped OFF whenever this window
        # carries typed (visual/dropped/phase) slots or a seeded right-suffix
        # search: both are orthogonal features this pass does not audit for
        # interaction, and neither is implicated in the evidenced bug
        # (plain hard-motion/HTM-word-length windows only). Synthetic cluster
        # slots are excluded because their identity is not
        # a physical action trace.  The ordinary demoted
        # incumbent ``required_word`` lane is not a structural mode: it must
        # coexist with searched merge candidates, while ``expand_burst`` keeps
        # that incumbent lane literal.  Flag absent or no eligible adjacency
        # => slot_merge_engageable is False and the merge branch is never
        # taken (byte-identical).
        slot_merge_adjacencies = []
        if ((_slot_merge_active() or canonical_count_interval)
                and raw_burst_stream
                and not visual_slots_active and not seed_entries):
            merge_read_frames = {
                int(frame)
                for stream in (read_frames, score_read_frames)
                for frames_si in stream for frame in frames_si
            }
            slot_merge_adjacencies = _slot_merge_adjacency_audit(
                bursts, merge_read_frames,
                (om_ctx or {}).get("rotation_frames", ()))
        slot_merge_eligible_pairs = {
            tuple(int(v) for v in row["slots"])
            for row in slot_merge_adjacencies if row["eligible"]
        }
        slot_merge_engageable = bool(slot_merge_eligible_pairs)
        n_om = len(orientations)
        start_ois = set(int(i) for i in (om_ctx or {}).get("start_ois", ()))
        if not start_ois:
            start_ois = set(range(n_om))

        observations = []
        for frame in (om_ctx or {}).get("rotation_frames", ()):
            frame = int(frame)
            if f_start < frame <= f_end:
                observations.append((frame, 0, "rotation", None, None))
        for si in range(a_span + 1, b_span + 1):
            # The default invocation is the finalized control plane.  The
            # independent dense-prefix invocation supplies its own immutable
            # frame/scorer pair; neither invocation can see the other's rows.
            n = max(1, len(plane_read_frames[si]))
            for ri, frame in enumerate(plane_read_frames[si]):
                frame = int(frame)
                if f_start < frame <= f_end:
                    observations.append(
                        (frame, 2, "read", (si, ri), 1.0 / n))
        observations.sort(key=lambda item: (item[0], item[1]))

        # Preserve the historical point-slot partition literally when no new
        # visual slot survives exact-overlap coalescing.  This is the S1 identity
        # seam for both flag-off and all-coalesced windows.
        interval_chunks = [[] for _ in range(len(action_slots))]
        if not visual_slots_active:
            chunks = [[] for _ in range(len(bursts) + 1)]
            for item in observations:
                frame, order = int(item[0]), int(item[1])
                k = (bisect.bisect_left(bursts, frame) if order == 0 else
                     bisect.bisect_right(bursts, frame))
                chunks[k].append(item)
        else:
            # Typed slots are already chronological.  Consume the observation
            # stream once: normal chunks lie between slots; a visual slot owns
            # every exact-frame read/rotation in its closed episode interval.
            chunks = [[] for _ in range(len(action_slots) + 1)]
            pos = 0
            for slot_i, slot in enumerate(action_slots):
                if slot.kind == "motion":
                    while pos < len(observations):
                        frame, order = observations[pos][:2]
                        if (int(frame) < int(slot.frame_hi)
                                or (int(frame) == int(slot.frame_hi)
                                    and int(order) == 0)):
                            chunks[slot_i].append(observations[pos])
                            pos += 1
                        else:
                            break
                else:
                    while (pos < len(observations)
                           and int(observations[pos][0])
                           < int(slot.frame_lo)):
                        chunks[slot_i].append(observations[pos])
                        pos += 1
                    while (pos < len(observations)
                           and int(observations[pos][0])
                           <= int(slot.frame_hi)):
                        interval_chunks[slot_i].append(observations[pos])
                        pos += 1
            chunks[-1].extend(observations[pos:])
        for chunk in chunks:
            chunk.sort(key=lambda item: (item[0], item[1]))

        # Dense rows remain absent from every ordinary prefix chunk.  For a
        # visual slot only, prepare metadata for rows whose exact timestamp is
        # inside that slot's owned closed interval.  Segment construction is
        # deferred until the shared read×state guard admits the branch grid.
        dense_interval_chunks = [[] for _ in range(len(action_slots))]
        if (not dense_prefix_plane
                and visual_slots_active and dense_evidence):
            for slot_i, slot in enumerate(action_slots):
                if slot.kind == "motion":
                    continue
                dense_interval_chunks[slot_i].extend(
                    item for item in interval_chunks[slot_i]
                    if item[2] == "rotation")
                for si in range(a_span + 1, b_span + 1):
                    n = max(1, len(score_read_frames[si]))
                    for ri, frame in enumerate(score_read_frames[si]):
                        frame = int(frame)
                        if int(slot.frame_lo) <= frame <= int(slot.frame_hi):
                            dense_interval_chunks[slot_i].append(
                                (frame, 2, "read", (si, ri), 1.0 / n))
                dense_interval_chunks[slot_i].sort(
                    key=lambda item: (item[0], item[1]))

        # TRUTH PROBE window truth (record-only, v2.1 truth-probe-
        # v2.1).  Control plane only -- the dense plane is a production
        # likelihood path, never a GT observer.  `boundary_override`
        # supersedes raw-timestamp membership only when THIS window's own
        # (start,end] frame span is present in an armed --truth-boundaries
        # file.
        tprobe_truth = None
        tprobe_progress = None
        tprobe_due_frame = None
        tprobe_beam_word = None
        tprobe_beam_state_bytes = None
        tprobe_beam_tokens = None
        tprobe_metric_merges = None
        if truth_probe_active and not dense_prefix_plane:
            try:
                _tb_override = tprobe_boundaries.get((f_start, f_end))
                tprobe_truth = _window_truth_v2(
                    state, f_start, f_end, tprobe_movegt_raw, tprobe_prefix,
                    perms, boundary_override=_tb_override)
                if tprobe_truth is not None:
                    _, _tp_raw_due = _truth_probe_v2_progress(
                        tprobe_truth["physical_frames"], bursts)
                    # v2.1: fold same-face same-burst quarter
                    # pairs into the beam's own half-turn token BEFORE
                    # matching, so the chronological matcher's expectation
                    # is expressed in the beam's OWN word alphabet, not the
                    # teacher's raw QTM alphabet (see
                    # `_truth_probe_v2_merge_bursts`).  Only
                    # `tprobe_progress`/`tprobe_due_frame` -- pure matching-
                    # currency helpers, never surfaced in a receipt -- are
                    # repointed at the normalized arrays; `tprobe_truth`
                    # itself is never mutated, so `gt_word`/`gt_word_len`
                    # below keep reading its untouched raw
                    # `physical_tokens`/`physical_word_len`.
                    tprobe_beam_tokens, tprobe_due_frame, \
                        tprobe_metric_merges = _truth_probe_v2_merge_bursts(
                            tprobe_truth["physical_tokens"], _tp_raw_due)
                    tprobe_beam_word = [
                        BS.MOVES.index(t) for t in tprobe_beam_tokens]
                    _tp_beam_start = np.frombuffer(
                        tprobe_truth["phys_state_bytes"][0], dtype=np.int8)
                    _tp_beam_states, _, _ = _truth_trajectory(
                        _tp_beam_start, tprobe_beam_tokens, perms)
                    tprobe_beam_state_bytes = [
                        s.tobytes() for s in _tp_beam_states]
                    tprobe_progress = _truth_probe_v2_beam_progress(
                        tprobe_due_frame, bursts)
            except Exception:                   # record-only: never abort
                tprobe_truth = None
                tprobe_progress = None
                tprobe_due_frame = None
                tprobe_beam_word = None
                tprobe_beam_state_bytes = None
                tprobe_beam_tokens = None
                tprobe_metric_merges = None

        def _tprobe_base(kind):
            return dict(
                kind=kind, tag=tag,
                capability=_tprobe_capability,
                window_idx=(win_ctx[0] if win_ctx else None),
                attempt=(win_ctx[1] if win_ctx else None),
                span_a=a_span, span_b=b_span,
                frames=[f_start, f_end],
                gt_word=tprobe_truth["physical_tokens"],
                gt_word_len=tprobe_truth["physical_word_len"],
                gt_word_beam=tprobe_beam_tokens,
                metric_merges=tprobe_metric_merges,
                gt_k_lo=tprobe_truth["k_lo"],
                gt_k_hi=tprobe_truth["k_hi"],
                gt_boundary_source=tprobe_truth["boundary_source"],
                start_matches_gt=tprobe_truth["start_matches_gt"],
                check_level="end-state")

        def _beam_generate_stateful_device():
            """Untyped chronological beam with its hot frontier on CUDA.

            The numerical and ordering contract is the legacy loop below:
            expansion -> stable key dedup -> chronological reads/rotations ->
            stable score band -> K per OM -> required-prefix reinjection.  Word
            and timing payloads remain parallel CUDA tensors and cross to the
            host only once, when the final retained frontier is materialized.

            The independent dense-prefix plane uses this exact controller with
            its own immutable read scorer, read-plan cache namespace, runtime
            counters, and checkpoint family.  It never shares a compiled read
            plan or retained frontier with the control plane; only the device
            implementation is common.  A seeded carry uploads its already-
            scored ``(state, OM, word)`` frontier as the device root, retains
            the seed path's per-slack-lane likelihood bands, and carries only
            right-suffix timings.  Optional typed slots remain excluded below
            because they require the common-currency CPU transaction.
            """
            import torch

            from detect import scrub_device_beam as DB
            from detect.trellis_tracker import (
                prepare_stateful_read_run_device,
                score_stateful_read_run_device,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for device scrub beam")
            seeded_device = bool(seed_entries)
            # The CPU seeded controller predates the learned action twin and
            # therefore carries only the already-admitted left evidence plus
            # ordinary right-window reads.  Keep this performance port score-
            # identical; calibrated NN evidence belongs at the separate final-
            # score seam rather than being introduced accidentally by routing.
            # CPU seeded searches do not emit device truth-probe stages.  The
            # seed frontier also lacks left-window timestamps, so treating its
            # full word as a fresh device word would manufacture false probe
            # losses.  Preserve the existing record-only behavior exactly.
            device_tprobe_truth = None if seeded_device else tprobe_truth
            _advance_device_beam_window_cache(
                device_beam_checkpoint_cache, device_read_run_cache,
                device_beam_window_runtime, a_span)
            plane_name = (str(consume_plane_role)
                          if consume_plane_role is not None else
                          ("dense" if dense_prefix_plane else "ordinary"))
            plane_source = (
                plane_name, id(plane_read_frames), id(plane_read_scorer))
            runtime = device_beam_runtime.get(plane_name)
            if runtime is None:
                device = torch.device("cuda")
                perms_t = torch.as_tensor(
                    np.asarray(perms, dtype=np.int64), device=device)
                destination_rows = [
                    tuple({oi} | set(om_nbrs[oi])) for oi in range(n_om)
                ]
                max_degree = max(map(len, destination_rows), default=0)
                destination_table = np.zeros(
                    (n_om, max_degree), dtype=np.int64)
                for oi, row in enumerate(destination_rows):
                    destination_table[oi, :len(row)] = row
                runtime = dict(
                    DB=DB,
                    torch=torch,
                    device=device,
                    perms=perms_t,
                    template=DB.build_branch_template_device(perms_t),
                    destinations=torch.as_tensor(
                        destination_table, device=device),
                    degrees=torch.as_tensor(
                        [len(row) for row in destination_rows],
                        dtype=torch.int64, device=device),
                )
                device_beam_runtime[plane_name] = runtime
            device = runtime["device"]
            if seeded_device:
                runtime["seeded_attempts"] = int(
                    runtime.get("seeded_attempts", 0)) + 1

            # IN-BEAM STATE RANK (CUBED_BEAM_STATE_RANK):
            # position the v0.15 verifier's calibrated LLR nats AT the per-om
            # k_cap selection -- the kill site -- as a RANK-ONLY,
            # SCORE-PRESERVING bias.  Band pruning and every propagated score
            # stay on original scores (the band-margin-dilution
            # hazard is impossible by construction); only the top-k membership
            # of a BINDING stratum cap can change.  Control plane + device
            # path only; the ll_onset fence and frame convention are identical
            # to the shipped final-scoring consume seam (~L10790).  Fail-soft:
            # any error leaves the channel absent for this window.
            state_rank_ctx = None
            state_rank_mod = None

            def take_payload(payload, indices):
                return {name: value[indices]
                        for name, value in payload.items()}

            def replace_score(batch, score):
                return DB.BeamBatch(
                    states=batch.states, score=score, oi=batch.oi,
                    nskip=batch.nskip, nins=batch.nins, req=batch.req,
                    source=batch.source, action=batch.action,
                    move0=batch.move0, move1=batch.move1,
                    branch=batch.branch)

            def dedup_identity(payload):
                """Plane-scoped future-complete device dedup identity.

                The V18 complete-band terminal lane retains every typed
                physical program.  All other lanes keep the historical
                state/OM identity, extended only by the pre-existing slot-
                merge identity when that mechanism is active.

                A provisional quarter must retain its direction and required-
                lane repair bit until the next slot.  Once a merge completes,
                its entire physical program remains distinct so state-rank and
                the later candidate-conditioned NN can compare ``D,D`` with
                ``D',D'`` instead of losing one before their scoring seam."""
                if terminal_complete_band:
                    return _terminal_program_payload_identity(payload, torch)
                return _slot_merge_payload_identity(payload, torch)

            def slot_merge_req_override(payload, slot_index):
                """Absolute required result for this pair's matching SINGLE."""
                if ("search_len" not in payload
                        or (int(slot_index) - 2, int(slot_index) - 1)
                        not in slot_merge_eligible_pairs):
                    return None
                override = torch.full(
                    (len(payload["search_len"]), len(BS.MOVES)), -1,
                    dtype=torch.int8, device=device)
                last = payload["merge_last"]
                rows = torch.nonzero(last >= 0, as_tuple=False).flatten()
                if len(rows):
                    override[rows, last[rows]] = payload[
                        "merge_reentry"][rows].to(torch.int8)
                return override

            def expand_payload(payload, expanded, burst_frame, slot_index,
                               source_req):
                source = expanded.source
                words_t = payload["word"][source].clone()
                timed_t = payload["timed"][source].clone()
                old_len = payload["word_len"][source]
                # Seed words belong to the already-scored left half, whose
                # timestamps are intentionally not copied across the split.
                # Their right suffix therefore owns a separate, left-aligned
                # timing length while ordinary rows retain word_len==timed_len.
                old_timed_len = payload.get("timed_len", payload["word_len"])[
                    source]
                rows = torch.arange(
                    len(expanded), dtype=torch.int64, device=device)
                single = expanded.action == DB.ACTION_SINGLE
                double = expanded.action == DB.ACTION_DOUBLE
                emits = single | double
                emit_rows = rows[emits]
                emit_pos = old_len[emits]
                words_t[emit_rows, emit_pos] = expanded.move0[emits]
                timed_emit_pos = old_timed_len[emits]
                timed_t[emit_rows, timed_emit_pos] = int(burst_frame)
                double_rows = rows[double]
                double_pos = old_len[double] + 1
                words_t[double_rows, double_pos] = expanded.move1[double]
                timed_double_pos = old_timed_len[double] + 1
                timed_t[double_rows, timed_double_pos] = int(burst_frame)
                word_len_t = (old_len + single.to(torch.int64)
                              + 2 * double.to(torch.int64))
                out = dict(word=words_t, timed=timed_t,
                           word_len=word_len_t)
                if "timed_len" in payload:
                    out["timed_len"] = (
                        old_timed_len + single.to(torch.int64)
                        + 2 * double.to(torch.int64))
                if "search_len" in payload:
                    old_search_len = payload["search_len"][source]
                    source_last = payload["merge_last"][source]
                    pair_eligible = (
                        (int(slot_index) - 2, int(slot_index) - 1)
                        in slot_merge_eligible_pairs)
                    merge = (
                        single
                        & bool(pair_eligible)
                        & (source_last == expanded.move0)
                        & (expanded.move0.to(torch.int64) % 3 != 2))
                    search_len_t = (
                        old_search_len + single.to(torch.int64)
                        + 2 * double.to(torch.int64)
                        - merge.to(torch.int64))
                    # A provisional quarter is useful only when this slot can
                    # legally pair with the immediately following slot.  Do
                    # not retain dead merge provenance across an observation
                    # barrier (or past the final slot).
                    next_pair_eligible = (
                        (int(slot_index) - 1, int(slot_index))
                        in slot_merge_eligible_pairs)
                    plain_quarter = (
                        single & ~merge & bool(next_pair_eligible)
                        & (expanded.move0.to(torch.int64) % 3 != 2))
                    merge_last_t = torch.where(
                        plain_quarter,
                        expanded.move0.to(torch.int64),
                        torch.full_like(old_search_len, -1))
                    half = (expanded.move0.to(torch.int64)
                            - expanded.move0.to(torch.int64) % 3 + 2)
                    expected_half = required_pad[old_search_len]
                    merge_reentry_t = (
                        plain_quarter
                        & source_req[source]
                        & (expected_half == half))
                    merge_mask_t = payload["merge_mask"][source]
                    merge_bit = torch.bitwise_left_shift(
                        torch.ones_like(old_len), old_len)
                    merge_mask_t = torch.where(
                        merge, merge_mask_t | merge_bit, merge_mask_t)
                    out.update(
                        search_len=search_len_t,
                        merge_last=merge_last_t,
                        merge_reentry=merge_reentry_t,
                        merge_mask=merge_mask_t,
                        merge_count=(payload["merge_count"][source]
                                     + merge.to(torch.int64)))
                if "rank_nats" in payload:
                    # In-beam state rank: each emitting row accumulates the
                    # calibrated LLR of its own emitted move at its own word
                    # position under its own om-at-emission-time.  SKIP,
                    # half-turn lanes, unbridged oms, and positions beyond the
                    # window's verifier bins contribute exact zero.
                    rank_t = payload["rank_nats"][source]
                    if state_rank_ctx is not None:
                        rank_t = rank_t + state_rank_mod.expansion_rank_deltas(
                            state_rank_ctx, torch=torch,
                            oi=expanded.oi, action=expanded.action,
                            move0=expanded.move0, move1=expanded.move1,
                            old_len=old_len,
                            action_single=DB.ACTION_SINGLE,
                            action_double=DB.ACTION_DOUBLE)
                    out["rank_nats"] = rank_t
                return out

            def clone_payload(payload):
                return {
                    name: value.clone()
                    for name, value in payload.items()
                }

            def prepare_read_plan(run):
                cache_key = (plane_source, tuple(
                    (int(item[3][0]), int(item[3][1]), float(item[4]))
                    for item in run))
                plan = device_read_run_cache.get(cache_key)
                if plan is None:
                    weighted = []
                    for _rf, _ro, _rk, read_id, weight in run:
                        si, ri = read_id
                        weighted.append((
                            float(weight),
                            tuple(plane_read_scorer(si, ri, oi)
                                  for oi in range(n_om))))
                    plan = prepare_stateful_read_run_device(
                        weighted, device=device)
                    device_read_run_cache[cache_key] = plan
                    runtime["read_run_plans_compiled"] = int(
                        runtime.get("read_run_plans_compiled", 0)) + 1
                    device_beam_window_runtime["read_run_plans_peak"] = max(
                        int(device_beam_window_runtime[
                            "read_run_plans_peak"]),
                        len(device_read_run_cache))
                return plan




            def advance_chunk_device(
                    batch, payload, k, resident_observer=None):
                chunk = chunks[k]
                pos = 0
                while pos < len(chunk) and len(batch):
                    _frame, _order, kind, _read_payload, _weight = chunk[pos]
                    if kind == "rotation":
                        rotated = DB.rotate_device(
                            batch, runtime["destinations"], runtime["degrees"],
                            extra_identity_key=dedup_identity(payload))
                        payload = take_payload(payload, rotated.source_indices)
                        batch = rotated.batch
                        if resident_observer is not None:
                            resident_observer(len(batch))
                        pos += 1
                        continue

                    end = pos
                    while end < len(chunk) and chunk[end][2] == "read":
                        end += 1
                    run = chunk[pos:end]
                    plan = prepare_read_plan(run)
                    if resident_observer is not None:
                        resident_observer(len(batch))
                    plane_memory_guard(
                        "max_score_cells", len(batch),
                        f"device read-score frontier at chunk {k}")
                    scores_t, alive_t = score_stateful_read_run_device(
                        batch.states, batch.score, batch.oi, plan)
                    batch = replace_score(batch, scores_t)
                    batch = DB.take_batch(batch, alive_t)
                    payload = take_payload(payload, alive_t)
                    pos = end
                return batch, payload


            def seeded_lane_identity(batch, payload):
                return _seeded_lane_identity_device(
                    batch,
                    skip_cap=skip_cap,
                    insert_cap=insert_cap,
                )

            def select_seeded_device(batch, payload):
                """Exact CUDA form of the seeded controller's lane bands.

                SKIP/DOUBLE counts are alternative timing alignments.  The CPU
                seeded path therefore applies the unchanged likelihood band
                independently inside each ``(nskip, nins)`` lane.
                The required incumbent is retained even when off-band and no
                K cut is applied.  All reductions and selection stay on CUDA.
                """
                return _select_seeded_complete_band_device(
                    batch,
                    torch=torch,
                    DB=DB,
                    band=band,
                    skip_cap=skip_cap,
                    insert_cap=insert_cap,
                )

            def concatenate_device_batches(parts):
                """Concatenate like-shaped beam rows without leaving CUDA."""
                if not parts:
                    raise ValueError("cannot concatenate an empty beam list")

                def joined(name):
                    values = [getattr(part, name) for part in parts]
                    if values[0] is None:
                        if any(value is not None for value in values[1:]):
                            raise ValueError(
                                "device beam ancestry fields are inconsistent")
                        return None
                    return torch.cat(values, dim=0)

                return DB.BeamBatch(**{
                    name: joined(name)
                    for name in (
                        "states", "score", "oi", "nskip", "nins", "req",
                        "source", "action", "move0", "move1", "branch")
                })

            def canonical_count_indices_device(batch, payload, k):
                """Exact step-k canonical-count survivors on the beam device."""
                if not canonical_count_interval or count_interval is None:
                    raise RuntimeError(
                        "canonical count indices require the v3 interval")
                count_lo, count_hi = count_interval
                search_len = payload.get(
                    "search_len", payload["word_len"])
                merge_pending = (
                    payload["merge_last"] >= 0
                    if "merge_last" in payload else
                    torch.zeros_like(search_len, dtype=torch.bool)
                )
                valid = _canonical_count_feasible_mask(
                    step=k, n_slots=len(bursts),
                    search_len=search_len,
                    merge_pending=merge_pending,
                    nskip=batch.nskip, nins=batch.nins,
                    skip_cap=skip_cap, insert_cap=insert_cap,
                    L_lo=count_lo, L_hi=count_hi,
                )
                return torch.nonzero(valid, as_tuple=False).flatten()

            def stream_canonical_step_device(
                    source_batch, source_payload, k, burst_frame,
                    required_next, allow_actions, branch,
                    single_score_delta, live_action_score_delta):
                """Exact two-pass CUDA schedule for one oversized v5a step.

                Both passes replay the production expansion, payload,
                chronological observation, and canonical-count filter.  Pass
                one computes one global likelihood cutoff.  Pass two retains
                that complete global band plus the established required lane;
                chunking never creates a local posterior or K cut.
                """
                nonlocal canonical_streamed_steps
                nonlocal canonical_streamed_preband_rows
                nonlocal canonical_streamed_band_rows_pre_global_dedup
                nonlocal canonical_streamed_global_dedup_rows
                nonlocal canonical_streamed_max_chunk_cells
                if not canonical_count_interval or count_interval is None:
                    raise RuntimeError(
                        "canonical CUDA streaming requires canonical count")
                if seeded_device:
                    raise RuntimeError(
                        "canonical CUDA streaming cannot consume seed rows")
                bound = int(BS.SCORE_STATE_BOUND)
                template_width = int(len(runtime["template"]))
                rotation_count = sum(
                    1 for item in chunks[k] if item[2] == "rotation")
                destination_width = int(runtime["destinations"].shape[1])
                # ``rotate_device`` materializes the complete padded
                # destination width before it filters/deduplicates.  The
                # finite OM cardinality bounds retained rows but not that
                # transient, so use d**r rather than min(n_om, d**r).
                rotation_fanout = int(destination_width) ** int(
                    rotation_count)
                scheduled_row_width = (
                    int(template_width) * int(rotation_fanout))
                if (template_width <= 0 or destination_width <= 0
                        or scheduled_row_width > bound):
                    raise RuntimeError(
                        "canonical CUDA streaming action/rotation schedule "
                        "cannot fit the resident bound: "
                        f"{scheduled_row_width} > {bound}")
                # ``expand_device`` first materializes source/branch metadata
                # for the complete template and only then applies the legal-
                # action mask.  A later rotation can then fan every retained
                # physical history across the existing OM destination table.
                # Size by the complete transient action+rotation width, not by
                # the post-dedup OM cardinality or post-mask logical branch
                # count (which can be one on a SKIP-only slot).
                chunk_n = max(1, bound // scheduled_row_width)

                def slice_delta(delta, indices, width, label):
                    if delta is None:
                        return None
                    if (not isinstance(delta, torch.Tensor)
                            or not delta.is_cuda
                            or delta.device != source_batch.states.device
                            or delta.dtype != torch.float64
                            or tuple(delta.shape)
                            != (len(source_batch), int(width))):
                        raise RuntimeError(
                            f"canonical streamed {label} is not exact CUDA")
                    return delta[indices]

                def run_part(start, stop):
                    scheduled_cells = (
                        int(stop - start) * scheduled_row_width)
                    plane_memory_guard(
                        "max_expansion_cells", scheduled_cells,
                        "canonical streamed action/rotation chunk")
                    source_indices = torch.arange(
                        start, stop, dtype=torch.int64, device=device)
                    part_batch = DB.take_batch(
                        source_batch, source_indices)
                    part_payload = take_payload(
                        source_payload, source_indices)
                    expanded = DB.expand_device(
                        part_batch, runtime["perms"],
                        skip_cap=int(skip_cap),
                        insert_cap=int(insert_cap),
                        required_next=required_next[source_indices],
                        single_req_override=slot_merge_req_override(
                            part_payload, k),
                        single_score_delta=slice_delta(
                            single_score_delta, source_indices, 18,
                            "single-score plane"),
                        live_action_score_delta=slice_delta(
                            live_action_score_delta, source_indices,
                            DB.N_LIVE_ACTIONS, "live-action plane"),
                        allow_actions=bool(allow_actions),
                        template=runtime["template"],
                    )
                    expanded_n = len(expanded)
                    expanded_payload = expand_payload(
                        part_payload, expanded, burst_frame, k,
                        part_batch.req)
                    dedup = DB.stable_dedup_device(
                        expanded,
                        extra_identity_key=dedup_identity(
                            expanded_payload),
                    )
                    part_batch = dedup.batch
                    part_payload = take_payload(
                        expanded_payload, dedup.winner_indices)
                    max_materialized_rows = max(
                        int(expanded_n), int(len(part_batch)))

                    def observe_resident_rows(rows):
                        nonlocal max_materialized_rows
                        rows = int(rows)
                        max_materialized_rows = max(
                            int(max_materialized_rows), rows)
                        plane_memory_guard(
                            "max_score_cells", rows,
                            "canonical streamed post-rotation/read frontier "
                            f"at chunk {k}")

                    part_batch, part_payload = advance_chunk_device(
                        part_batch, part_payload, k,
                        resident_observer=observe_resident_rows)
                    count_before = len(part_batch)
                    max_materialized_rows = max(
                        int(max_materialized_rows), int(count_before))
                    count_indices = canonical_count_indices_device(
                        part_batch, part_payload, k)
                    part_batch = DB.take_batch(
                        part_batch, count_indices)
                    part_payload = take_payload(
                        part_payload, count_indices)
                    return (
                        part_batch, part_payload, int(expanded_n),
                        int(scheduled_cells),
                        int(max_materialized_rows),
                        int(count_before),
                        int(count_before - len(part_batch)),
                    )

                def finish_stream_receipt(
                        retained, retained_payload, *, preband_rows,
                        count_before_rows, count_removed_rows,
                        band_rows_pre_global_dedup, max_chunk_cells,
                        max_materialized_chunk_rows, terminal_count_lo,
                        terminal_count_hi, terminal_physical_lo,
                        terminal_physical_hi):
                    nonlocal canonical_streamed_steps
                    nonlocal canonical_streamed_preband_rows
                    nonlocal canonical_streamed_band_rows_pre_global_dedup
                    nonlocal canonical_streamed_global_dedup_rows
                    nonlocal canonical_streamed_max_chunk_cells
                    canonical_streamed_steps += 1
                    canonical_streamed_preband_rows += int(preband_rows)
                    canonical_streamed_band_rows_pre_global_dedup += int(
                        band_rows_pre_global_dedup)
                    canonical_streamed_global_dedup_rows += int(len(retained))
                    canonical_streamed_max_chunk_cells = max(
                        int(canonical_streamed_max_chunk_cells),
                        int(max_chunk_cells),
                    )
                    return (
                        retained,
                        retained_payload,
                        dict(
                            preband_rows=int(preband_rows),
                            count_before_rows=int(count_before_rows),
                            count_removed_rows=int(count_removed_rows),
                            count_removal_accounting=(
                                "chunk-summed-before-global-dedup"),
                            band_rows_pre_global_dedup=int(
                                band_rows_pre_global_dedup),
                            global_dedup_rows=int(len(retained)),
                            # Compatibility aliases for the v3-side summary.
                            band_rows=int(band_rows_pre_global_dedup),
                            dedup_rows=int(len(retained)),
                            max_resident_chunk_cells=int(max_chunk_cells),
                            max_materialized_chunk_rows=int(
                                max_materialized_chunk_rows),
                            action_template_width=int(template_width),
                            rotation_observations=int(rotation_count),
                            rotation_destination_width=int(
                                destination_width),
                            rotation_fanout_bound=int(rotation_fanout),
                            scheduled_rows_per_source=int(
                                scheduled_row_width),
                            max_source_chunk_rows=int(chunk_n),
                            terminal_count_min=terminal_count_lo,
                            terminal_count_max=terminal_count_hi,
                            terminal_physical_count_min=terminal_physical_lo,
                            terminal_physical_count_max=terminal_physical_hi,
                        ),
                    )

                global_best = torch.full(
                    (), -torch.inf, dtype=torch.float64, device=device)
                preband_rows = 0
                count_before_rows = 0
                count_removed_rows = 0
                max_chunk_cells = 0
                max_materialized_chunk_rows = 0
                terminal_count_lo = None
                terminal_count_hi = None
                terminal_physical_lo = None
                terminal_physical_hi = None
                for start in range(0, len(source_batch), chunk_n):
                    (part_batch, part_payload, expanded_n, scheduled_cells,
                     materialized_rows, count_before, count_removed) = run_part(
                        start, min(start + chunk_n, len(source_batch)))
                    max_chunk_cells = max(max_chunk_cells, scheduled_cells)
                    max_materialized_chunk_rows = max(
                        max_materialized_chunk_rows, materialized_rows)
                    count_before_rows += count_before
                    count_removed_rows += count_removed
                    preband_rows += len(part_batch)
                    if not len(part_batch):
                        continue
                    global_best = torch.maximum(
                        global_best, part_batch.score.max())
                    if k == len(bursts):
                        terminal_counts = part_payload.get(
                            "search_len", part_payload["word_len"])
                        physical_counts = (
                            int(k) - part_batch.nskip + part_batch.nins)
                        count_part_lo = int(
                            terminal_counts.min().detach().cpu().item())
                        count_part_hi = int(
                            terminal_counts.max().detach().cpu().item())
                        physical_part_lo = int(
                            physical_counts.min().detach().cpu().item())
                        physical_part_hi = int(
                            physical_counts.max().detach().cpu().item())
                        terminal_count_lo = (
                            count_part_lo if terminal_count_lo is None else
                            min(terminal_count_lo, count_part_lo))
                        terminal_count_hi = (
                            count_part_hi if terminal_count_hi is None else
                            max(terminal_count_hi, count_part_hi))
                        terminal_physical_lo = (
                            physical_part_lo
                            if terminal_physical_lo is None else
                            min(terminal_physical_lo, physical_part_lo))
                        terminal_physical_hi = (
                            physical_part_hi
                            if terminal_physical_hi is None else
                            max(terminal_physical_hi, physical_part_hi))

                if preband_rows == 0:
                    empty = torch.empty(
                        0, dtype=torch.int64, device=device)
                    return finish_stream_receipt(
                        DB.take_batch(source_batch, empty),
                        take_payload(source_payload, empty),
                        preband_rows=0,
                        count_before_rows=count_before_rows,
                        count_removed_rows=count_removed_rows,
                        band_rows_pre_global_dedup=0,
                        max_chunk_cells=max_chunk_cells,
                        max_materialized_chunk_rows=(
                            max_materialized_chunk_rows),
                        terminal_count_lo=None,
                        terminal_count_hi=None,
                        terminal_physical_lo=None,
                        terminal_physical_hi=None,
                    )

                kept_batches = []
                kept_payloads = []
                band_rows = 0
                for start in range(0, len(source_batch), chunk_n):
                    (part_batch, part_payload, _expanded_n, _scheduled_cells,
                     _materialized_rows, _before, _removed) = run_part(
                            start, min(start + chunk_n,
                                       len(source_batch)))
                    if not len(part_batch):
                        continue
                    required = part_batch.req
                    if "merge_reentry" in part_payload:
                        required = required | part_payload["merge_reentry"]
                    keep = (
                        (part_batch.score >= global_best - band) | required)
                    keep_indices = torch.nonzero(
                        keep, as_tuple=False).flatten()
                    if not len(keep_indices):
                        continue
                    part_batch = DB.take_batch(
                        part_batch, keep_indices)
                    part_payload = take_payload(
                        part_payload, keep_indices)
                    band_rows += len(part_batch)
                    if band_rows > bound:
                        if os.environ.get("CUBED_V5A_DIAG") == "1":
                            try:
                                _cur = part_batch.score
                                _cur_min = float(_cur.min().item())
                                _cur_max = float(_cur.max().item())
                                _cur_mean = float(_cur.mean().item())
                                if kept_batches:
                                    _ret = torch.cat(
                                        [kb.score for kb in kept_batches])
                                    _ret_n = int(_ret.numel())
                                    _ret_min = float(_ret.min().item())
                                    _ret_max = float(_ret.max().item())
                                else:
                                    _ret_n = 0
                                    _ret_min = float("nan")
                                    _ret_max = float("nan")
                                _free, _total = torch.cuda.mem_get_info()
                                _resv = int(torch.cuda.memory_reserved())
                                print(
                                    "[v5a-diag] SITE1 band-overflow "
                                    f"band_rows={band_rows} bound={bound} "
                                    f"band={float(band):.6g} "
                                    f"global_best="
                                    f"{float(global_best.item()):.6g} "
                                    f"retained_n={_ret_n} "
                                    f"retained=[{_ret_min:.6g},"
                                    f"{_ret_max:.6g}] "
                                    f"cur=[{_cur_min:.6g},{_cur_max:.6g}] "
                                    f"cur_mean={_cur_mean:.6g} "
                                    f"cuda_free={int(_free)} "
                                    f"cuda_total={int(_total)} "
                                    f"cuda_resv={_resv}",
                                    flush=True)
                            except Exception:
                                pass
                        raise RuntimeError(
                            "canonical CUDA streamed global-band rows "
                            f"{band_rows} > {bound}")
                    kept_batches.append(part_batch)
                    kept_payloads.append(part_payload)

                if not kept_batches:
                    empty = torch.empty(
                        0, dtype=torch.int64, device=device)
                    retained = DB.take_batch(source_batch, empty)
                    retained_payload = take_payload(source_payload, empty)
                else:
                    combined = concatenate_device_batches(kept_batches)
                    combined_payload = {
                        name: torch.cat([
                            payload_part[name]
                            for payload_part in kept_payloads
                        ], dim=0)
                        for name in kept_payloads[0]
                    }
                    dedup = DB.stable_dedup_device(
                        combined,
                        extra_identity_key=dedup_identity(combined_payload),
                    )
                    retained = dedup.batch
                    retained_payload = take_payload(
                        combined_payload, dedup.winner_indices)

                return finish_stream_receipt(
                    retained,
                    retained_payload,
                    preband_rows=preband_rows,
                    count_before_rows=count_before_rows,
                    count_removed_rows=count_removed_rows,
                    band_rows_pre_global_dedup=band_rows,
                    max_chunk_cells=max_chunk_cells,
                    max_materialized_chunk_rows=max_materialized_chunk_rows,
                    terminal_count_lo=terminal_count_lo,
                    terminal_count_hi=terminal_count_hi,
                    terminal_physical_lo=terminal_physical_lo,
                    terminal_physical_hi=terminal_physical_hi,
                )

            def stream_seeded_step_device(
                    source_batch, source_payload, k, burst_frame,
                    required_next, allow_actions, branch):
                """Two-pass CUDA schedule for an oversized seeded step.

                This is the device twin of the established CPU scheduler:
                pass one finds one global best per timing lane; pass two keeps
                that lane's complete likelihood band plus the incumbent, then
                performs one stable global dedup.  Chunking is only a memory
                schedule; no independently pruned posterior is introduced.
                """
                if count_interval is not None or slot_merge_engageable:
                    raise RuntimeError(
                        "seeded CUDA streaming does not support auxiliary "
                        "count or slot-merge transactions")
                bound = int(BS.SCORE_STATE_BOUND)
                chunk_n = max(1, bound // int(branch))
                _initial_lane, lane_count = seeded_lane_identity(
                    source_batch, source_payload
                )
                lane_best = torch.full(
                    (lane_count,), -torch.inf,
                    dtype=torch.float64, device=device)
                preband_n = 0

                def run_part(start, stop):
                    indices = torch.arange(
                        start, stop, dtype=torch.int64, device=device)
                    part_batch = DB.take_batch(source_batch, indices)
                    part_payload = take_payload(source_payload, indices)
                    expanded = DB.expand_device(
                        part_batch, runtime["perms"],
                        skip_cap=int(skip_cap), insert_cap=int(insert_cap),
                        required_next=required_next[indices],
                        allow_actions=bool(allow_actions),
                        template=runtime["template"])
                    expanded_payload = expand_payload(
                        part_payload, expanded, burst_frame, k,
                        part_batch.req)
                    dedup = DB.stable_dedup_device(
                        expanded,
                        extra_identity_key=dedup_identity(expanded_payload),
                    )
                    part_batch = dedup.batch
                    part_payload = take_payload(
                        expanded_payload, dedup.winner_indices)
                    return advance_chunk_device(
                        part_batch, part_payload, k)

                for start in range(0, len(source_batch), chunk_n):
                    part_batch, part_payload = run_part(
                        start, min(start + chunk_n, len(source_batch)))
                    preband_n += len(part_batch)
                    if len(part_batch):
                        lane, _lane_count = seeded_lane_identity(
                            part_batch, part_payload
                        )
                        lane_best.scatter_reduce_(
                            0, lane, part_batch.score,
                            reduce="amax", include_self=True)

                kept_batches = []
                kept_payloads = []
                raw_kept_n = 0
                for start in range(0, len(source_batch), chunk_n):
                    part_batch, part_payload = run_part(
                        start, min(start + chunk_n, len(source_batch)))
                    if not len(part_batch):
                        continue
                    lane, _lane_count = seeded_lane_identity(
                        part_batch, part_payload
                    )
                    keep = ((part_batch.score >= lane_best[lane] - band)
                            | part_batch.req)
                    indices = torch.nonzero(
                        keep, as_tuple=False).flatten()
                    if not len(indices):
                        continue
                    part_batch = DB.take_batch(part_batch, indices)
                    part_payload = take_payload(part_payload, indices)
                    raw_kept_n += len(part_batch)
                    # Capacity remains a rejection bound, never a top-K cut.
                    # Refuse to concatenate an oversized retained posterior;
                    # the outer transaction rejects without truncation.
                    if raw_kept_n > bound:
                        raise RuntimeError(
                            "seeded CUDA streamed retained rows "
                            f"{raw_kept_n} > {bound}")
                    kept_batches.append(part_batch)
                    kept_payloads.append(part_payload)

                if not kept_batches:
                    empty = torch.empty(
                        0, dtype=torch.int64, device=device)
                    return (
                        DB.take_batch(source_batch, empty),
                        take_payload(source_payload, empty),
                        int(preband_n), int(preband_n))
                combined = concatenate_device_batches(kept_batches)
                combined_payload = {
                    name: torch.cat(
                        [payload_part[name]
                         for payload_part in kept_payloads], dim=0)
                    for name in kept_payloads[0]
                }
                dedup = DB.stable_dedup_device(
                    combined,
                    extra_identity_key=dedup_identity(combined_payload),
                )
                retained = dedup.batch
                retained_payload = take_payload(
                    combined_payload, dedup.winner_indices)
                runtime["seeded_streamed_steps"] = int(
                    runtime.get("seeded_streamed_steps", 0)) + 1
                runtime["seeded_streamed_preband"] = int(
                    runtime.get("seeded_streamed_preband", 0)) + int(
                        preband_n)
                return (
                    retained, retained_payload, int(preband_n),
                    max(0, int(preband_n) - len(retained)))

            start_oi_rows = sorted(start_ois)
            band = float(z_sigma) * sigma
            required = tuple(required_word or ())
            payload_width = int(
                STRUCT_CAP if word_capacity is None else word_capacity)
            if payload_width < int(STRUCT_CAP):
                raise ValueError(
                    "device beam payload width is below STRUCT_CAP")
            required_pad = torch.full(
                (payload_width + 2,), -1, dtype=torch.int64, device=device)
            if required:
                n_required = min(len(required), len(required_pad))
                required_pad[:n_required] = torch.as_tensor(
                    required[:n_required], dtype=torch.int64, device=device)
            required_offsets = torch.arange(
                2, dtype=torch.int64, device=device).reshape(1, 2)
            all_read_frames = [
                int(item[0]) for chunk_items in chunks for item in chunk_items
                if item[2] == "read"
            ]
            witness_bits = tuple(
                _slot_has_endpoint_evidence(frame, all_read_frames)
                for frame in bursts)
            chunk_signatures = tuple(tuple(
                (int(frame), int(order), kind,
                 None if read_id is None else tuple(map(int, read_id)),
                 None if weight is None else float(weight))
                for frame, order, kind, read_id, weight in chunk_items)
                for chunk_items in chunks)
            static_key = _device_beam_checkpoint_family_key(
                plane_source=plane_source, a_span=a_span, state=state,
                start_ois=start_oi_rows, beam_k=beam_k, band=band,
                late_evidence_band_active=late_evidence_band_active,
                om_nbrs=om_nbrs, raw_burst_stream=raw_burst_stream,
                payload_width=payload_width)
            # While the truth probe is armed, checkpoint
            # reuse is bypassed the SAME way -- a resumed call would silently
            # skip every already-cached burst's stage receipts, leaving gaps
            # a reader could mistake for "nothing to report" instead of "not
            # observed this attempt".  A fresh, throwaway family never reads
            # OR writes the cross-call `device_beam_checkpoint_cache`, so it
            # cannot contaminate a later non-probed run either.  Receipted at
            # the window-root stage row below (`checkpoint_bypassed`) and in
            # the run report/summary (`truth_probe_checkpoint_bypass`).
            # Performance-only: candidate set, scores, and decisions are
            # unchanged by this branch either way.
            checkpoint_family = (
                {}
                if (truth_probe_active or seeded_device
                    or slot_merge_engageable
                    or count_interval is not None)
                else device_beam_checkpoint_cache.setdefault(static_key, {})
            )
            device_beam_window_runtime["checkpoint_families_peak"] = max(
                int(device_beam_window_runtime["checkpoint_families_peak"]),
                len(device_beam_checkpoint_cache))

            def checkpoint_key(k):
                if k == 0:
                    return (0, chunk_signatures[0])
                max_word_len = k + min(k, int(insert_cap))
                required_signature = (
                    tuple(required[:max_word_len]),
                    min(len(required), max_word_len))
                return (
                    int(k), int(skip_cap), int(insert_cap),
                    tuple(bursts[:k]), chunk_signatures[:k + 1],
                    witness_bits[:k], required_signature)

            resume_k = -1
            checkpoint = None
            for candidate_k in range(len(bursts), -1, -1):
                checkpoint = checkpoint_family.get(
                    checkpoint_key(candidate_k))
                if checkpoint is not None:
                    resume_k = candidate_k
                    break
            if checkpoint is None:
                seed_word_collision = False
                if seeded_device:
                    seed_states = []
                    seed_scores = []
                    seed_ois = []
                    seed_required = []
                    seed_words = []
                    seen_seed_words = {}
                    for entry in seed_entries:
                        state_row = np.frombuffer(
                            entry["state"], dtype=np.int8)
                        if state_row.shape != (54,):
                            raise ValueError(
                                "seed state must contain exactly 54 int8 cells")
                        oi = int(entry["oi"])
                        if not 0 <= oi < n_om:
                            raise ValueError("seed orientation index is invalid")
                        word = tuple(int(move) for move in entry["word"])
                        if (len(word) > payload_width
                                or any(not 0 <= move < len(BS.MOVES)
                                       for move in word)):
                            raise ValueError(
                                "seed word exceeds payload or move alphabet")
                        required_seed = bool(entry.get("is_incumbent"))
                        seed_key = (entry["state"], oi, required_seed)
                        old_word = seen_seed_words.get(seed_key)
                        if old_word is not None and old_word != word:
                            seed_word_collision = True
                        seen_seed_words.setdefault(seed_key, word)
                        seed_states.append(state_row.copy())
                        seed_scores.append(float(entry["score_rel"]))
                        seed_ois.append(oi)
                        seed_required.append(required_seed)
                        seed_words.append(word)
                    if not seed_states:
                        raise ValueError("seeded device beam has no seed rows")
                    n_start = len(seed_states)
                    batch = DB.BeamBatch(
                        states=torch.as_tensor(
                            np.stack(seed_states), dtype=torch.int8,
                            device=device),
                        score=torch.as_tensor(
                            seed_scores, dtype=torch.float64, device=device),
                        oi=torch.as_tensor(
                            seed_ois, dtype=torch.int64, device=device),
                        nskip=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        nins=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        req=torch.as_tensor(
                            seed_required, dtype=torch.bool, device=device))
                    word_t = torch.full(
                        (n_start, payload_width), -1, dtype=torch.int8,
                        device=device)
                    word_len_t = torch.as_tensor(
                        [len(word) for word in seed_words],
                        dtype=torch.int64, device=device)
                    for row, word in enumerate(seed_words):
                        if word:
                            word_t[row, :len(word)] = torch.as_tensor(
                                word, dtype=torch.int8, device=device)
                    payload = dict(
                        word=word_t,
                        timed=torch.full(
                            (n_start, payload_width), -1,
                            dtype=torch.int64, device=device),
                        word_len=word_len_t,
                        timed_len=torch.zeros(
                            n_start, dtype=torch.int64, device=device))
                    seed_dedup = DB.stable_dedup_device(
                        batch,
                        extra_identity_key=dedup_identity(payload),
                    )
                    batch = seed_dedup.batch
                    payload = take_payload(
                        payload, seed_dedup.winner_indices)
                    n_start = len(batch)
                else:
                    st0_t = torch.as_tensor(
                        np.asarray(state, dtype=np.int8),
                        device=device).reshape(1, 54)
                    n_start = len(start_oi_rows)
                    batch = DB.BeamBatch(
                        states=st0_t.repeat(n_start, 1),
                        score=torch.zeros(
                            n_start, dtype=torch.float64, device=device),
                        oi=torch.as_tensor(
                            start_oi_rows, dtype=torch.int64, device=device),
                        nskip=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        nins=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        req=torch.ones(
                            n_start, dtype=torch.bool, device=device))
                    payload = dict(
                        word=torch.full(
                            (n_start, payload_width), -1, dtype=torch.int8,
                            device=device),
                        timed=torch.full(
                            (n_start, payload_width), -1, dtype=torch.int64,
                            device=device),
                        word_len=torch.zeros(
                            n_start, dtype=torch.int64, device=device))
                    if terminal_complete_band:
                        payload["timed_len"] = torch.zeros(
                            n_start, dtype=torch.int64, device=device)
                if slot_merge_engageable:
                    # The established word/timing tensors stay PHYSICAL.
                    # These small parallel fields carry only the canonical
                    # search-length state needed to spend two ordinary slots
                    # on one HTM token without hiding either action from NN
                    # scoring or from raw sequence replay.
                    payload.update(
                        search_len=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        merge_last=torch.full(
                            (n_start,), -1, dtype=torch.int64, device=device),
                        merge_reentry=torch.zeros(
                            n_start, dtype=torch.bool, device=device),
                        merge_mask=torch.zeros(
                            n_start, dtype=torch.int64, device=device),
                        merge_count=torch.zeros(
                            n_start, dtype=torch.int64, device=device))
                if state_rank_ctx is not None:
                    payload["rank_nats"] = torch.zeros(
                        n_start, dtype=torch.float64, device=device)
                batch, payload = advance_chunk_device(batch, payload, 0)
                n_prn, capped = 0, False
                max_live_pairs = len(batch)
                plane_memory_guard(
                    "max_frontier", max_live_pairs,
                    "initial device frontier")
                resume_k = 0
                checkpoint_family[checkpoint_key(0)] = dict(
                    batch=batch, payload=payload, n_prn=n_prn,
                    capped=capped, max_live_pairs=max_live_pairs,
                    seed_word_collision=bool(seed_word_collision),
                    plane_stats=(dict(prefix_plane_stats)
                                 if prefix_plane_stats is not None else {}))
            else:
                batch = checkpoint["batch"]
                payload = checkpoint["payload"]
                seed_word_collision = bool(
                    checkpoint.get("seed_word_collision", False))
                n_prn = int(checkpoint["n_prn"])
                capped = bool(checkpoint["capped"])
                max_live_pairs = int(checkpoint["max_live_pairs"])
                for field, value in checkpoint.get(
                        "plane_stats", {}).items():
                    plane_max(field, value)
            runtime["checkpoint_calls"] = int(
                runtime.get("checkpoint_calls", 0)) + 1
            runtime["checkpoint_bursts_total"] = int(
                runtime.get("checkpoint_bursts_total", 0)) + len(bursts)
            runtime["checkpoint_bursts_skipped"] = int(
                runtime.get("checkpoint_bursts_skipped", 0)) + max(0, resume_k)
            if resume_k > 0:
                runtime["checkpoint_resumed_calls"] = int(
                    runtime.get("checkpoint_resumed_calls", 0)) + 1

            # TRUTH PROBE (record-only): window root/input stage.
            # Fires exactly once per call, right after the initial (possibly
            # multi-orientation) frontier is built and its own chunk-0 read
            # score applied -- before any burst has expanded a single
            # candidate.  `resume_k` doubles as the checkpoint-bypass
            # receipt: always 0 while the probe is armed (see the
            # `checkpoint_family` allocation above), since a bypassed family
            # can never resume past burst 0.  Seeds `_v2_prev_truth_rows`,
            # the running ancestry set the generation-stage receipt below
            # cross-checks each burst's `expanded.source` against -- at the
            # root every orientation-candidate row trivially matches truth
            # (zero progress, no move made yet, so nothing has diverged).
            _v2_prev_truth_rows = []
            if device_tprobe_truth is not None:
                try:
                    _v2_root_states = batch.states.detach().cpu().numpy()
                    _v2_root_match = _truth_probe_v2_match_stage(
                        word=payload["word"].detach().cpu().numpy(),
                        timed=payload["timed"].detach().cpu().numpy(),
                        word_len=(
                            payload["word_len"].detach().cpu().numpy()),
                        states=_v2_root_states,
                        target_m=tprobe_progress[0],
                        physical_word_idx=tprobe_beam_word,
                        phys_state_bytes=tprobe_beam_state_bytes,
                        due_frame=tprobe_due_frame)
                    _v2_prev_truth_rows = list(_v2_root_match["matched_rows"])
                    tprobe_row(dict(
                        _tprobe_base("window_root"),
                        plane=plane_name, backend="cuda-resident",
                        resume_k=int(resume_k),
                        checkpoint_bypassed=bool(truth_probe_active),
                        n_root=int(len(_v2_root_states)),
                        target_m=_v2_root_match["target_m"],
                        expressible=bool(_v2_root_match["expressible"]),
                        truth_alive=bool(_v2_prev_truth_rows),
                        truth_rows=list(int(i) for i in _v2_prev_truth_rows)))
                except Exception as _tp_exc:      # record-only: never abort
                    try:
                        tprobe_row(dict(
                            _tprobe_base("window_root_error"),
                            plane=plane_name,
                            error=f"{type(_tp_exc).__name__}: {_tp_exc}"))
                    except Exception:
                        pass





            terminal_pre_band_rows = None
            terminal_band_rows = None
            terminal_would_have_k_rows = None
            complete_band_prefix_steps = 0
            complete_band_prefix_max_rows = 0
            complete_band_prefix_max_would_have_k_rows = 0
            count_feasibility_steps = 0
            count_feasibility_rows_removed = 0
            terminal_count_min = None
            terminal_count_max = None
            terminal_physical_count_min = None
            terminal_physical_count_max = None
            canonical_streamed_steps = 0
            canonical_streamed_preband_rows = 0
            canonical_streamed_band_rows_pre_global_dedup = 0
            canonical_streamed_global_dedup_rows = 0
            canonical_streamed_max_chunk_cells = 0
            canonical_streamed_step_receipts = []
            twin_canonical_count_filter_steps = 0
            twin_canonical_count_rows_removed = 0
            for k in range(resume_k + 1, len(bursts) + 1):
                burst_frame = bursts[k - 1]
                # V18 action-consume: per-burst list of enabled-provider deltas
                # for the CONTROL beam, reset each burst (empty => byte-identical).
                # Each entry is (name, delta[18]-or-[N,18]-or-None, reason, meta).
                after_evidence = not witness_bits[k - 1]
                branch = (1 if after_evidence else
                          1 + 18 + (18 * 15 if insert_cap > 0 else 0))
                expansion_work = len(batch) * branch
                streamed_seed_step = bool(
                    seeded_device and count_interval is None
                    and not slot_merge_engageable
                    and expansion_work > BS.SCORE_STATE_BOUND)
                streamed_canonical_step = bool(
                    canonical_count_interval and not seeded_device
                    and expansion_work > BS.SCORE_STATE_BOUND)
                streamed_device_step = bool(
                    streamed_seed_step or streamed_canonical_step)
                plane_max("max_expansion_work", expansion_work)
                plane_memory_guard(
                    "max_expansion_cells",
                    (min(expansion_work, int(BS.SCORE_STATE_BOUND))
                     if streamed_device_step else expansion_work),
                    "device resident expansion")
                if (expansion_work > BS.SCORE_STATE_BOUND
                        and seeded_device and not streamed_seed_step):
                    raise RuntimeError(
                        "seeded device expansion requires an unsupported "
                        "auxiliary streaming transaction")
                if (expansion_work > BS.SCORE_STATE_BOUND
                        and not streamed_device_step):
                    stats = dict(
                        kept=0, tie_size=0, overflow=True,
                        beam_limit=int(beam_k) * n_om,
                        prefix_om_mode="chronological",
                        execution_backend="cuda-resident",
                        execution_fallback=False,
                        device_checkpoint_resume_step=int(resume_k),
                        sequential_payload_width=int(payload_width),
                        max_live_pairs=int(max_live_pairs),
                        scaling_tripped=(
                            f"prefix expansions {len(batch) * branch} > "
                            f"{BS.SCORE_STATE_BOUND}"))
                    if device_tprobe_truth is not None:
                        try:                    # record-only: never abort
                            tprobe_row(dict(
                                _tprobe_base("window_overflow"),
                                plane=plane_name, burst_k=int(k),
                                burst_frame=int(burst_frame),
                                expansion_work=int(expansion_work)))
                        except Exception:
                            pass
                    return {}, {}, n_prn, True, stats


                word_pos = payload.get(
                    "search_len", payload["word_len"]
                ).reshape(-1, 1)
                required_next = required_pad[word_pos + required_offsets]
                # V13 side-head PROVIDER-2 (om-conditioned, PER-CANDIDATE-ROW):
                # append its [N,18] LLR (candidate rows' om indices = batch.oi,
                # over the burst's OWNED FRAME INTERVAL [frame_lo, frame_hi)) to
                # the SAME provider list.  Historical consume uses window_delta's
                # fail-soft host provider.  Option 2 instead uses the strict CUDA
                # provider: any unavailable/non-CUDA result abstains the complete
                # typed transaction rather than falling back to host scoring.  Its
                # LLR enters UNMODIFIED (the fused = lo18 + lo13 recipe of record;
                # no base-rate subtraction), and the recorded span + meta calib_B
                # let the offline gate account for the uniform quarter-lane bias.
                # V18 action-consume: ASSEMBLE the enabled providers -> per-
                # provider center + sigma/std scale -> SUM -> ONE clamp -> the
                # ll_onset fence.  DEFAULT emits the 289 PAIR plane
                # (live_action_score_delta) whose per-pair branch deltas
                # discriminate move-2; CUBED_NN_CONSUME_SINGLE_PLANE=1 emits the
                # legacy 18-lane single_score_delta.  Exactly ONE plane is emitted
                # (the other stays None -- no double-count, risk #1).
                # Empty/abstained/fenced => both None, so expand_device is byte-
                # identical when off (risk #9).
                consume_single_delta = None
                consume_live_delta = None
                # Exactly one consume plane (never both) -- no double-count.
                assert (consume_single_delta is None
                        or consume_live_delta is None)
                if streamed_seed_step:
                    if (consume_single_delta is not None
                            or consume_live_delta is not None):
                        raise RuntimeError(
                            "seeded CUDA streaming received learned score deltas")
                    (batch, payload, streamed_preband_n,
                     streamed_pruned_n) = stream_seeded_step_device(
                        batch, payload, k, burst_frame, required_next,
                        not after_evidence, branch)
                    n_prn += int(streamed_pruned_n)
                    expanded = None
                    expanded_payload = None
                elif streamed_canonical_step:
                    batch, payload, stream_receipt = (
                        stream_canonical_step_device(
                            batch, payload, k, burst_frame, required_next,
                            not after_evidence, branch,
                            consume_single_delta, consume_live_delta,
                        )
                    )
                    canonical_streamed_step_receipts.append(dict(
                        burst_k=int(k),
                        burst_frame=int(burst_frame),
                        plane_role=plane_name,
                        **stream_receipt,
                    ))
                    # The streamed frontier has already passed expansion,
                    # local/global dedup, chronological observations, the
                    # canonical-count mask, and the global likelihood band.
                    # Record that compound boundary explicitly; reporting it
                    # later as ordinary "dedup" or "observation" would
                    # misattribute a truth loss to a stage that has already
                    # run inside the two-pass schedule.
                    if device_tprobe_truth is not None:
                        try:
                            _stream_states = (
                                batch.states.detach().cpu().numpy())
                            _stream_match = _truth_probe_v2_match_stage(
                                word=payload["word"].detach().cpu().numpy(),
                                timed=payload["timed"].detach().cpu().numpy(),
                                word_len=(payload["word_len"]
                                          .detach().cpu().numpy()),
                                states=_stream_states,
                                target_m=tprobe_progress[k],
                                physical_word_idx=tprobe_beam_word,
                                phys_state_bytes=tprobe_beam_state_bytes,
                                due_frame=tprobe_due_frame)
                            tprobe_row(dict(
                                _tprobe_base("burst"),
                                stage="streamed_global_band",
                                plane=plane_name,
                                backend="cuda-resident",
                                burst_k=int(k),
                                burst_frame=int(burst_frame),
                                count_interval=list(count_interval),
                                n_preband_chunk_summed=int(
                                    stream_receipt["preband_rows"]),
                                n_post_global_dedup=int(
                                    stream_receipt["global_dedup_rows"]),
                                target_m=_stream_match["target_m"],
                                expressible=bool(
                                    _stream_match["expressible"]),
                                truth_alive_post_stream=bool(
                                    _stream_match["matched_rows"]),
                                truth_rows_post_stream=int(len(
                                    _stream_match["matched_rows"]))))
                        except Exception as _tp_exc:
                            try:
                                tprobe_row(dict(
                                    _tprobe_base("burst_error"),
                                    stage="streamed_global_band",
                                    plane=plane_name,
                                    burst_k=int(k),
                                    error=(f"{type(_tp_exc).__name__}: "
                                           f"{_tp_exc}")))
                            except Exception:
                                pass
                    n_prn += max(
                        0,
                        int(stream_receipt["preband_rows"])
                        - int(stream_receipt["dedup_rows"]),
                    )
                    if int(stream_receipt["count_before_rows"]) > 0:
                        count_feasibility_steps += 1
                        count_feasibility_rows_removed += int(
                            stream_receipt["count_removed_rows"])
                    if k == len(bursts):
                        terminal_count_min = stream_receipt[
                            "terminal_count_min"]
                        terminal_count_max = stream_receipt[
                            "terminal_count_max"]
                        terminal_physical_count_min = stream_receipt[
                            "terminal_physical_count_min"]
                        terminal_physical_count_max = stream_receipt[
                            "terminal_physical_count_max"]
                    expanded = None
                    expanded_payload = None
                else:
                    expanded = DB.expand_device(
                        batch, runtime["perms"], skip_cap=int(skip_cap),
                        insert_cap=int(insert_cap),
                        required_next=required_next,
                        single_req_override=slot_merge_req_override(payload, k),
                        single_score_delta=consume_single_delta,
                        live_action_score_delta=consume_live_delta,
                        allow_actions=not after_evidence,
                        template=runtime["template"])
                    expanded_payload = expand_payload(
                        payload, expanded, burst_frame, k, batch.req)
                # TRUTH PROBE (record-only): candidate-generation
                # stage.  The gap this closes: the v1 probe "begins
                # only after expansion dedup and observation scoring",
                # so a truth loss AT generation (e.g. the required
                # physical move pruned by the skip/insert caps or
                # `allow_actions`) was invisible.  Cross-checks
                # `expanded.source` against the PRIOR burst's own validated
                # truth rows -- the explicit, literal ancestry receipt the
                # frozen spec asks for, independent of (and expected to
                # always agree with) the exact word/timed/state prefix check
                # `_truth_probe_v2_match_stage` performs at every stage.
                if (device_tprobe_truth is not None
                        and not streamed_device_step):
                    try:
                        _v2_gen_states = (
                            expanded.states.detach().cpu().numpy())
                        _v2_gen_source = (
                            expanded.source.detach().cpu().numpy())
                        _v2_gen_match = _truth_probe_v2_match_stage(
                            word=expanded_payload["word"]
                                .detach().cpu().numpy(),
                            timed=expanded_payload["timed"]
                                .detach().cpu().numpy(),
                            word_len=expanded_payload["word_len"]
                                .detach().cpu().numpy(),
                            states=_v2_gen_states,
                            target_m=tprobe_progress[k],
                            physical_word_idx=tprobe_beam_word,
                            phys_state_bytes=tprobe_beam_state_bytes,
                            due_frame=tprobe_due_frame)
                        _v2_gen_rows = _v2_gen_match["matched_rows"]
                        _v2_gen_ancestry = _truth_probe_v2_ancestry_ok(
                            _v2_gen_source, _v2_gen_rows,
                            _v2_prev_truth_rows)
                        tprobe_row(dict(
                            _tprobe_base("burst"), stage="generation",
                            plane=plane_name, backend="cuda-resident",
                            burst_k=int(k), burst_frame=int(burst_frame),
                            n_pre_generation=int(len(_v2_prev_truth_rows)),
                            n_post_generation=int(len(_v2_gen_states)),
                            target_m=_v2_gen_match["target_m"],
                            expressible=bool(_v2_gen_match["expressible"]),
                            truth_alive_post_generation=bool(_v2_gen_rows),
                            truth_rows_post_generation=int(
                                len(_v2_gen_rows)),
                            ancestry_ok=(
                                all(_v2_gen_ancestry.values())
                                if _v2_gen_ancestry else None),
                            ancestry_breaks=sorted(
                                row for row, ok in _v2_gen_ancestry.items()
                                if not ok)))
                    except Exception as _tp_exc:  # record-only: never abort
                        try:
                            tprobe_row(dict(
                                _tprobe_base("burst_error"),
                                stage="generation",
                                plane=plane_name, burst_k=int(k),
                                error=(f"{type(_tp_exc).__name__}: "
                                       f"{_tp_exc}")))
                        except Exception:
                            pass
                if not streamed_device_step:
                    dedup = DB.stable_dedup_device(
                        expanded,
                        extra_identity_key=dedup_identity(
                            expanded_payload))
                    batch = dedup.batch
                    payload = take_payload(
                        expanded_payload, dedup.winner_indices)
                # TRUTH PROBE (record-only): expansion-dedup stage --
                # the other half of the generation/dedup blind spot.
                # A truth row can be merged away by `stable_dedup_device`'s
                # state/OM key if it collides with a higher-scoring rival
                # sharing the SAME (state, om); this is the first point that
                # would be visible.
                if (device_tprobe_truth is not None
                        and not streamed_device_step):
                    try:
                        _v2_dedup_states = (
                            batch.states.detach().cpu().numpy())
                        _v2_dedup_match = _truth_probe_v2_match_stage(
                            word=payload["word"].detach().cpu().numpy(),
                            timed=payload["timed"].detach().cpu().numpy(),
                            word_len=(
                                payload["word_len"].detach().cpu().numpy()),
                            states=_v2_dedup_states,
                            target_m=tprobe_progress[k],
                            physical_word_idx=tprobe_beam_word,
                            phys_state_bytes=tprobe_beam_state_bytes,
                            due_frame=tprobe_due_frame)
                        tprobe_row(dict(
                            _tprobe_base("burst"), stage="dedup",
                            plane=plane_name, backend="cuda-resident",
                            burst_k=int(k), burst_frame=int(burst_frame),
                            n_post_dedup=int(len(_v2_dedup_states)),
                            target_m=_v2_dedup_match["target_m"],
                            expressible=bool(_v2_dedup_match["expressible"]),
                            truth_alive_post_dedup=bool(
                                _v2_dedup_match["matched_rows"]),
                            truth_rows_post_dedup=int(
                                len(_v2_dedup_match["matched_rows"]))))
                    except Exception as _tp_exc:  # record-only: never abort
                        try:
                            tprobe_row(dict(
                                _tprobe_base("burst_error"), stage="dedup",
                                plane=plane_name, burst_k=int(k),
                                error=(f"{type(_tp_exc).__name__}: "
                                       f"{_tp_exc}")))
                        except Exception:
                            pass
                if not streamed_device_step:
                    batch, payload = advance_chunk_device(batch, payload, k)
                # TRUTH PROBE (record-only): observation-scoring stage
                # ("chronological reads/rotations" in the controller's own
                # docstring).  `advance_chunk_device` can itself drop rows
                # (`DB.take_batch(batch, alive_t)`), a third, distinct
                # survival point from generation and dedup.
                if (device_tprobe_truth is not None
                        and not streamed_device_step):
                    try:
                        _v2_obs_states = batch.states.detach().cpu().numpy()
                        _v2_obs_match = _truth_probe_v2_match_stage(
                            word=payload["word"].detach().cpu().numpy(),
                            timed=payload["timed"].detach().cpu().numpy(),
                            word_len=(
                                payload["word_len"].detach().cpu().numpy()),
                            states=_v2_obs_states,
                            target_m=tprobe_progress[k],
                            physical_word_idx=tprobe_beam_word,
                            phys_state_bytes=tprobe_beam_state_bytes,
                            due_frame=tprobe_due_frame)
                        tprobe_row(dict(
                            _tprobe_base("burst"), stage="observation",
                            plane=plane_name, backend="cuda-resident",
                            burst_k=int(k), burst_frame=int(burst_frame),
                            n_post_observation=int(len(_v2_obs_states)),
                            target_m=_v2_obs_match["target_m"],
                            expressible=bool(_v2_obs_match["expressible"]),
                            truth_alive_post_observation=bool(
                                _v2_obs_match["matched_rows"]),
                            truth_rows_post_observation=int(
                                len(_v2_obs_match["matched_rows"]))))
                    except Exception as _tp_exc:  # record-only: never abort
                        try:
                            tprobe_row(dict(
                                _tprobe_base("burst_error"),
                                stage="observation",
                                plane=plane_name, burst_k=int(k),
                                error=(f"{type(_tp_exc).__name__}: "
                                       f"{_tp_exc}")))
                        except Exception:
                            pass
                if (count_interval is not None and len(batch)
                        and not streamed_canonical_step):
                    count_feasibility_steps += 1
                    count_lo, count_hi = count_interval
                    if canonical_count_interval:
                        count_search_len = payload.get(
                            "search_len", payload["word_len"])
                        merge_pending = (
                            payload["merge_last"] >= 0
                            if "merge_last" in payload else
                            torch.zeros_like(
                                count_search_len, dtype=torch.bool)
                        )
                        count_valid = _canonical_count_feasible_mask(
                            step=k, n_slots=len(bursts),
                            search_len=count_search_len,
                            merge_pending=merge_pending,
                            nskip=batch.nskip, nins=batch.nins,
                            skip_cap=skip_cap, insert_cap=insert_cap,
                            L_lo=count_lo, L_hi=count_hi)
                    else:
                        count_valid = _physical_count_feasible_mask(
                            step=k, n_slots=len(bursts), nskip=batch.nskip,
                            nins=batch.nins, skip_cap=skip_cap,
                            insert_cap=insert_cap, L_lo=count_lo, L_hi=count_hi)
                    count_before = int(len(batch))
                    count_indices = torch.nonzero(
                        count_valid, as_tuple=False).flatten()
                    batch = DB.take_batch(batch, count_indices)
                    payload = take_payload(payload, count_indices)
                    count_feasibility_rows_removed += (
                        count_before - int(len(batch)))
                    if k == len(bursts) and len(batch):
                        terminal_physical_counts = (
                            int(k) - batch.nskip + batch.nins)
                        terminal_counts = (
                            payload.get(
                                "search_len", payload["word_len"])
                            if canonical_count_interval else
                            terminal_physical_counts
                        )
                        terminal_count_min = int(
                            terminal_counts.min().detach().cpu().item())
                        terminal_count_max = int(
                            terminal_counts.max().detach().cpu().item())
                        terminal_physical_count_min = int(
                            terminal_physical_counts.min().detach().cpu().item())
                        terminal_physical_count_max = int(
                            terminal_physical_counts.max().detach().cpu().item())
                    if device_tprobe_truth is not None:
                        try:
                            count_states = (
                                batch.states.detach().cpu().numpy())
                            count_match = _truth_probe_v2_match_stage(
                                word=payload["word"].detach().cpu().numpy(),
                                timed=payload["timed"].detach().cpu().numpy(),
                                word_len=(payload["word_len"]
                                          .detach().cpu().numpy()),
                                states=count_states,
                                target_m=tprobe_progress[k],
                                physical_word_idx=tprobe_beam_word,
                                phys_state_bytes=tprobe_beam_state_bytes,
                                due_frame=tprobe_due_frame)
                            tprobe_row(dict(
                                _tprobe_base("burst"),
                                stage="count_feasibility",
                                plane=plane_name,
                                backend="cuda-resident",
                                burst_k=int(k),
                                burst_frame=int(burst_frame),
                                count_interval=[count_lo, count_hi],
                                n_pre_count_filter=count_before,
                                n_post_count_filter=int(len(batch)),
                                target_m=count_match["target_m"],
                                expressible=bool(count_match["expressible"]),
                                truth_alive_post_count=bool(
                                    count_match["matched_rows"]),
                                truth_rows_post_count=int(
                                    len(count_match["matched_rows"]))))
                        except Exception as _tp_exc:
                            try:
                                tprobe_row(dict(
                                    _tprobe_base("burst_error"),
                                    stage="count_feasibility",
                                    plane=plane_name, burst_k=int(k),
                                    error=(f"{type(_tp_exc).__name__}: "
                                           f"{_tp_exc}")))
                            except Exception:
                                pass
                if not len(batch):
                    break

                n_ranked = len(batch)
                # In-beam state rank: the ordering bias is worth AT MOST one
                # decision band (clamp to +/-band = z_sigma*sigma, the same
                # bound the renderer and oracle terms use).  rank_delta is
                # ordering-only inside select_device; the propagated rows
                # keep their ORIGINAL scores.
                state_rank_delta = None
                if state_rank_ctx is not None and "rank_nats" in payload:
                    state_rank_delta = torch.clamp(
                        payload["rank_nats"], min=-band, max=band)
                    state_rank_stats["bursts_biased"] += 1
                    state_rank_stats["rank_nats_l1"] += float(
                        state_rank_delta.abs().sum().item())
                complete_band_prefix_step = bool(terminal_complete_band)
                terminal_band_step = bool(
                    complete_band_prefix_step and k == len(bursts))
                defer_prefix_k = bool(
                    late_evidence_band_active or complete_band_prefix_step)
                selection_batch = batch
                if "merge_reentry" in payload:
                    # A required canonical `D2` path is temporarily spelled
                    # physical `D` after its first slot.  Let that one-step
                    # debt participate in required-lane reinjection, but do
                    # not persist `req=True`: SKIP or a mismatching second
                    # action must still leave the incumbent lane.  The next
                    # matching SINGLE restores the real req bit through the
                    # absolute override above.
                    provisional_req = payload["merge_reentry"]
                    selection_batch = DB.BeamBatch(
                        states=batch.states, score=batch.score, oi=batch.oi,
                        nskip=batch.nskip, nins=batch.nins,
                        req=batch.req | provisional_req,
                        source=batch.source, action=batch.action,
                        move0=batch.move0, move1=batch.move1,
                        branch=batch.branch)
                selected = (
                    select_seeded_device(selection_batch, payload)
                    if seeded_device else
                    DB.select_device(
                        selection_batch, band=band,
                        beam_k=(len(batch)
                                if defer_prefix_k else int(beam_k)),
                        rank_delta=state_rank_delta))
                if complete_band_prefix_step:
                    ordinary_selected = DB.select_device(
                        selection_batch, band=band,
                        beam_k=int(beam_k), rank_delta=state_rank_delta)
                    complete_band_prefix_steps += 1
                    complete_band_prefix_max_rows = max(
                        int(complete_band_prefix_max_rows),
                        int(len(selected.kept_indices)))
                    complete_band_prefix_max_would_have_k_rows = max(
                        int(complete_band_prefix_max_would_have_k_rows),
                        int(len(ordinary_selected.selected_indices)))
                if terminal_band_step:
                    terminal_pre_band_rows = (
                        int(canonical_streamed_step_receipts[-1][
                            "preband_rows"])
                        if streamed_canonical_step else int(len(batch)))
                    terminal_band_rows = int(len(selected.kept_indices))
                    terminal_would_have_k_rows = int(
                        len(ordinary_selected.selected_indices))
                if selection_batch is not batch:
                    selected = DB.SelectionResult(
                        batch=DB.take_batch(
                            batch, selected.selected_indices),
                        selected_indices=selected.selected_indices,
                        ranked_indices=selected.ranked_indices,
                        kept_indices=selected.kept_indices)
                n_kept = len(selected.kept_indices)
                n_selected = len(selected.selected_indices)
                n_prn += n_ranked - n_kept
                cut_here = n_selected < n_kept
                if cut_here:
                    capped = True
                    n_prn += n_kept - n_selected
                # TRUTH PROBE (record-only): band(+reinjection) and
                # K(+reinjection) stages, D2H copies + JSONL write only.
                # Reads pre-selection `batch` and the selection index sets;
                # touches no tensor in place and alters no control flow.
                # `_trs` = this burst's own chronological target (progress[k]
                # against the pre-select batch, exactly what `select_device`
                # itself is choosing among); `_ers` = the WINDOW's full final
                # target, checked at every burst so an early-complete row is
                # never missed (mirrors the pre-v2 `_ers`/`truth_end_*`
                # fields' intent, now via the same exact matcher).
                if device_tprobe_truth is not None:
                    try:
                        _tps = batch.states.detach().cpu().numpy()
                        _tpc = batch.score.detach().cpu().numpy()
                        _tpo = batch.oi.detach().cpu().numpy()
                        _tpr = selected.ranked_indices.detach().cpu().numpy()
                        _tp_kept = (
                            selected.kept_indices.detach().cpu().numpy())
                        _tp_selected = (
                            selected.selected_indices.detach().cpu().numpy())
                        _tp_word = payload["word"].detach().cpu().numpy()
                        _tp_timed = payload["timed"].detach().cpu().numpy()
                        _tp_wlen = (
                            payload["word_len"].detach().cpu().numpy())
                        _kset = set(int(i) for i in _tp_kept)
                        _sset = set(int(i) for i in _tp_selected)
                        _v2_burst_match = _truth_probe_v2_match_stage(
                            word=_tp_word, timed=_tp_timed,
                            word_len=_tp_wlen, states=_tps,
                            target_m=tprobe_progress[k],
                            physical_word_idx=tprobe_beam_word,
                            phys_state_bytes=tprobe_beam_state_bytes,
                            due_frame=tprobe_due_frame)
                        _v2_end_match = _truth_probe_v2_match_stage(
                            word=_tp_word, timed=_tp_timed,
                            word_len=_tp_wlen, states=_tps,
                            target_m=len(tprobe_beam_word),
                            physical_word_idx=tprobe_beam_word,
                            phys_state_bytes=tprobe_beam_state_bytes,
                            due_frame=tprobe_due_frame)
                        _trs = set(_v2_burst_match["matched_rows"])
                        _ers = set(_v2_end_match["matched_rows"])
                        # Seed the NEXT burst's generation-stage ancestry
                        # check: which of THIS burst's K-survivors are v2
                        # truth matches, re-expressed in the post-select
                        # `selected.batch` index space (row `pos` there ==
                        # pre-select row `_tp_selected[pos]` here) -- exactly
                        # the space burst k+1's `expanded.source` indexes
                        # against.  Reset to empty on any failure below
                        # (never carry a stale prior-burst set forward).
                        _v2_prev_truth_rows = [
                            pos for pos, orig in enumerate(_tp_selected)
                            if int(orig) in _trs]
                        _rank = next(
                            (pos for pos, ix in enumerate(_tpr)
                             if int(ix) in _trs), None)
                        _end_rank = next(
                            (pos for pos, ix in enumerate(_tpr)
                             if int(ix) in _ers), None)
                        _best = (float(_tpc[_tpr[0]]) if len(_tpr) else None)
                        _tbest = (max(float(_tpc[i]) for i in _trs)
                                  if _trs else None)
                        _alive_pre = bool(_trs)
                        _alive_band = any(i in _kset for i in _trs)
                        _alive_sel = any(i in _sset for i in _trs)
                        _killer = (None if not _alive_pre else
                                   ("band" if not _alive_band else
                                    ("k_cap" if not _alive_sel else None)))
                        _surv_oms = sorted(
                            {int(_tpo[i]) for i in _sset})
                        _state_rank_probe = {}
                        if (state_rank_ctx is not None
                                and state_rank_delta is not None
                                and "rank_nats" in payload):
                            try:
                                _rank_raw = (
                                    payload["rank_nats"]
                                    .detach().cpu().numpy())
                                _rank_clamped = (
                                    state_rank_delta
                                    .detach().cpu().numpy())
                                _rank_k_eff = int(beam_k)
                                if DB.stratum_redistribution_enabled():
                                    _rank_k_eff = DB.effective_beam_k(
                                        int(beam_k),
                                        len(np.unique(_tpo[_tp_kept])))
                                _state_rank_probe = (
                                    _state_rank_truth_receipt(
                                        score=_tpc, oi=_tpo,
                                        kept_indices=_tp_kept,
                                        selected_indices=_tp_selected,
                                        truth_indices=_trs,
                                        rank_nats=_rank_raw,
                                        rank_delta=_rank_clamped,
                                        effective_k=_rank_k_eff))
                            except Exception as _srp_exc:
                                # Record-only diagnostics must never suppress
                                # the established truth-probe burst row.
                                _state_rank_probe = {
                                    "state_rank_receipt_error": (
                                        f"{type(_srp_exc).__name__}: "
                                        f"{_srp_exc}")}
                        # STATE DUMP (record-only): for a burst where truth
                        # rows survive pre-K, materialize the actual 54-cell
                        # state of truth's OM stratum's top-K kept rows +
                        # truth rows + the K-cut boundary pair, so a kill
                        # burst can be diffed cell-by-cell.  Independent of
                        # the state-rank seam (works with it on or off);
                        # reuses only already-host-resident arrays plus one
                        # small D2H pull mirroring the rank_delta pull above.
                        _state_dump_probe = {}
                        if _trs:
                            try:
                                _dump_k_eff = int(beam_k)
                                if DB.stratum_redistribution_enabled():
                                    _dump_k_eff = DB.effective_beam_k(
                                        int(beam_k),
                                        len(np.unique(_tpo[_tp_kept])))
                                _dump_rank_delta = (
                                    state_rank_delta.detach().cpu().numpy()
                                    if state_rank_delta is not None
                                    else None)
                                _state_dump_probe = (
                                    _state_dump_truth_receipt(
                                        score=_tpc, oi=_tpo, states=_tps,
                                        kept_indices=_tp_kept,
                                        truth_indices=_trs,
                                        effective_k=_dump_k_eff,
                                        rank_delta=_dump_rank_delta))
                            except Exception as _sdp_exc:
                                # Record-only diagnostics must never suppress
                                # the established truth-probe burst row.
                                _state_dump_probe = {
                                    "state_dump_receipt_error": (
                                        f"{type(_sdp_exc).__name__}: "
                                        f"{_sdp_exc}")}
                        tprobe_row(dict(
                            _tprobe_base("burst"), stage="band_k",
                            plane=plane_name,
                            backend="cuda-resident",
                            burst_k=int(k), burst_frame=int(burst_frame),
                            target_m=_v2_burst_match["target_m"],
                            expressible=bool(_v2_burst_match["expressible"]),
                            n_pre_select=int(n_ranked),
                            n_post_band=int(n_kept),
                            n_post_k=int(n_selected),
                            band=float(band), beam_k=int(beam_k),
                            best_score=_best,
                            truth_alive_pre=bool(_alive_pre),
                            truth_alive_post_band=bool(_alive_band),
                            truth_alive_post_k=bool(_alive_sel),
                            truth_killer=_killer,
                            truth_rank_pre=(None if _rank is None
                                            else int(_rank)),
                            truth_best_score=_tbest,
                            truth_margin=(
                                None if (_tbest is None or _best is None)
                                else float(_best - _tbest)),
                            truth_rows_pre=int(len(_trs)),
                            truth_oms_pre=sorted(
                                {int(_tpo[i]) for i in _trs}),
                            truth_oms_post_k=sorted(
                                {int(_tpo[i]) for i in _trs if i in _sset}),
                            truth_end_alive_pre=bool(_ers),
                            truth_end_alive_post_k=any(
                                i in _sset for i in _ers),
                            truth_end_rank_pre=(None if _end_rank is None
                                                else int(_end_rank)),
                            om_diversity_pre=len(
                                {int(v) for v in _tpo}),
                            om_diversity_post_k=len(_surv_oms),
                            survivor_oms=_surv_oms,
                            **_state_rank_probe,
                            **_state_dump_probe))
                    except Exception as _tp_exc:  # record-only: never abort
                        _v2_prev_truth_rows = []
                        try:
                            tprobe_row(dict(
                                _tprobe_base("burst_error"), stage="band_k",
                                plane=plane_name, burst_k=int(k),
                                error=(f"{type(_tp_exc).__name__}: "
                                       f"{_tp_exc}")))
                        except Exception:
                            pass
                payload = take_payload(payload, selected.selected_indices)
                batch = selected.batch
                max_live_pairs = max(max_live_pairs, len(batch))
                plane_memory_guard(
                    "max_frontier", len(batch),
                    "retained device frontier")
                checkpoint_family[checkpoint_key(k)] = dict(
                    batch=batch, payload=payload, n_prn=n_prn,
                    capped=capped, max_live_pairs=max_live_pairs,
                    plane_stats=(dict(prefix_plane_stats)
                                 if prefix_plane_stats is not None else {}))


            # TRUTH PROBE (record-only): final-frontier stage.  Per
            # the frozen spec this match requires the EXACT final state AND
            # (unreduced, physical) word AND timing -- `target_m` is the
            # window's own full `physical_word_len`, so a row only counts
            # here if it reproduces the teacher's complete unreduced move
            # sequence with every timestamp at the burst it was actually due,
            # landing on the exact replayed end state.  `expressible=False`
            # means the window's own burst stream could never carry the full
            # teacher word regardless of search quality (T2/T8-shaped) --
            # never reported as a live kill.
            if device_tprobe_truth is not None:
                try:
                    _tps = batch.states.detach().cpu().numpy()
                    _tpo = batch.oi.detach().cpu().numpy()
                    _v2_final_match = _truth_probe_v2_match_stage(
                        word=payload["word"].detach().cpu().numpy(),
                        timed=payload["timed"].detach().cpu().numpy(),
                        word_len=payload["word_len"].detach().cpu().numpy(),
                        states=_tps,
                        target_m=len(tprobe_beam_word),
                        physical_word_idx=tprobe_beam_word,
                        phys_state_bytes=tprobe_beam_state_bytes,
                        due_frame=tprobe_due_frame)
                    _v2_final_rows = _v2_final_match["matched_rows"]
                    tprobe_row(dict(
                        _tprobe_base("window_final"),
                        plane=plane_name,
                        backend="cuda-resident",
                        n_bursts=len(bursts),
                        resume_k=int(resume_k),
                        n_final=int(len(_tps)),
                        target_m=_v2_final_match["target_m"],
                        expressible=bool(_v2_final_match["expressible"]),
                        truth_alive=bool(_v2_final_rows),
                        truth_end_alive=bool(_v2_final_rows),
                        om_diversity=len({int(v) for v in _tpo}),
                        truth_oms=sorted(
                            {int(_tpo[i]) for i in _v2_final_rows})))
                except Exception:               # record-only: never abort
                    pass
            scores_h = batch.score.cpu().numpy()
            ois_h = batch.oi.cpu().numpy()
            words_h = payload["word"].cpu().numpy()
            timed_h = payload["timed"].cpu().numpy()
            lengths_h = payload["word_len"].cpu().numpy()
            timed_lengths_h = (
                payload["timed_len"].cpu().numpy()
                if "timed_len" in payload else lengths_h)
            search_lengths_h = (
                payload["search_len"].cpu().numpy()
                if "search_len" in payload else lengths_h)
            merge_masks_h = (
                payload["merge_mask"].cpu().numpy()
                if "merge_mask" in payload else np.zeros_like(lengths_h))
            words, word_scores, temporal = {}, {}, {}
            merge_programs = {}
            for i, (sc, oi) in enumerate(zip(scores_h, ois_h)):
                length = int(lengths_h[i])
                timed_length = int(timed_lengths_h[i])
                seq = tuple(int(v) for v in words_h[i, :length])
                timed = tuple(int(v) for v in timed_h[i, :timed_length])
                search_seq, _search_timed, merges = (
                    _slot_merge_canonicalize_program(
                        seq, timed, int(merge_masks_h[i]),
                        slot_merge_adjacencies)
                    if int(merge_masks_h[i]) else (seq, timed, []))
                if len(search_seq) != int(search_lengths_h[i]):
                    raise RuntimeError(
                        "device slot-merge search length/provenance mismatch")
                end_seen = (not timed or any(
                    frame >= int(timed[-1]) for frame in all_read_frames))
                if not end_seen:
                    continue
                if merges:
                    program_key = (seq, timed, int(merge_masks_h[i]))
                    old_program = merge_programs.get(program_key)
                    if (old_program is None
                            or float(sc) > old_program["beam_score"]):
                        merge_programs[program_key] = dict(
                            word=seq,
                            timing=timed,
                            search_word=search_seq,
                            beam_score=float(sc),
                            merge_mask=int(merge_masks_h[i]),
                            merges=merges)
                rec = temporal.setdefault(seq, dict(
                    om_scores=np.full(n_om, -np.inf),
                    timings={}, timing_sets={}, ambiguous_ois=set(),
                    end_pinned=True))
                _record_temporal_frontier_path(
                    rec, oi, sc, timed,
                    retain_all_timings=terminal_complete_band)
                if sc > word_scores.get(seq, -np.inf):
                    word_scores[seq] = float(sc)
                    words[seq] = None
            stats = dict(
                kept=0, tie_size=0, overflow=False,
                beam_limit=(int(BS.SCORE_STATE_BOUND)
                            if (late_evidence_band_active
                                or terminal_complete_band)
                            else int(beam_k) * n_om),
                prefix_om_mode="chronological",
                late_evidence_band=late_evidence_band_active,
                execution_backend="cuda-resident",
                execution_fallback=False,
                device_checkpoint_resume_step=int(resume_k),
                sequential_payload_width=int(payload_width),
                max_live_pairs=int(max_live_pairs),
                seed_word_collision=bool(seed_word_collision),
                canonical_count_streaming=dict(
                    policy=_CANONICAL_COUNT_STREAMING_POLICY,
                    preregistration_sha256=(
                        _CANONICAL_COUNT_STREAMING_PREREG_SHA256),
                    streamed_steps=int(canonical_streamed_steps),
                    streamed_preband_rows=int(
                        canonical_streamed_preband_rows),
                    streamed_band_rows_pre_global_dedup=int(
                        canonical_streamed_band_rows_pre_global_dedup),
                    streamed_global_dedup_rows=int(
                        canonical_streamed_global_dedup_rows),
                    max_resident_chunk_cells=int(
                        canonical_streamed_max_chunk_cells),
                    capacity_bound=int(BS.SCORE_STATE_BOUND),
                    execution_backend="cuda-resident",
                    execution_fallback=False,
                    top_k_approximation=False,
                    twin_count_filter_steps=int(
                        twin_canonical_count_filter_steps),
                    twin_count_rows_removed=int(
                        twin_canonical_count_rows_removed),
                    step_receipts=copy.deepcopy(
                        canonical_streamed_step_receipts),
                ),
                slot_merge_surviving_programs=len(merge_programs),
                slot_merges=[{
                    "word": [int(move) for move in data["word"]],
                    "timing": [int(frame) for frame in data["timing"]],
                    "search_word": [
                        int(move) for move in data["search_word"]],
                    "beam_score": float(data["beam_score"]),
                    "merge_mask": int(data["merge_mask"]),
                    "merges": data["merges"],
                } for data in merge_programs.values()])
            if consume_plane_role is not None or terminal_complete_band:
                stats.update(
                    consume_plane_role=consume_plane_role,
                    consume_provider_mode=(
                        "fused"
                        if record_only_consume_provider_mode is None else
                        record_only_consume_provider_mode),
                    terminal_complete_band=bool(terminal_complete_band),
                    complete_band_prefix_steps=int(
                        complete_band_prefix_steps),
                    complete_band_prefix_max_rows=int(
                        complete_band_prefix_max_rows),
                    complete_band_prefix_max_would_have_k_rows=int(
                        complete_band_prefix_max_would_have_k_rows),
                    terminal_pre_band_rows=terminal_pre_band_rows,
                    terminal_band_rows=terminal_band_rows,
                    terminal_would_have_k_rows=(
                        terminal_would_have_k_rows))
                stats["terminal_dedup_policy"] = (
                    _TERMINAL_DEDUP_POLICY_PHYSICAL_HISTORY
                    if terminal_complete_band
                    else _TERMINAL_DEDUP_POLICY_STATE_ONLY
                )
                if terminal_complete_band:
                    stats.update(
                        terminal_dedup_key_schema=(
                            _TERMINAL_DEDUP_KEY_SCHEMA),
                        terminal_dedup_key_width=(
                            _terminal_program_dedup_key_width(
                                payload_width)),
                    )
            if count_interval is not None:
                stats.update(
                    physical_count_interval=list(count_interval),
                    count_feasibility_policy=(
                        "canonical-search-len-reachable-terminal-interval-"
                        "before-band-k-v1"
                        if canonical_count_interval else
                        "reachable-terminal-interval-before-band-k"),
                    count_currency=(
                        "canonical-search-len-htm"
                        if canonical_count_interval else
                        "physical-action-count"),
                    count_feasibility_steps=int(count_feasibility_steps),
                    count_feasibility_rows_removed=int(
                        count_feasibility_rows_removed),
                    terminal_count_min=terminal_count_min,
                    terminal_count_max=terminal_count_max,
                    terminal_physical_count_min=(
                        terminal_physical_count_min),
                    terminal_physical_count_max=(
                        terminal_physical_count_max))
            if seeded_device:
                runtime["seeded_completed"] = int(
                    runtime.get("seeded_completed", 0)) + 1
            return words, temporal, n_prn, capped, stats

        device_beam_attempted = False
        device_beam_fallback = None
        if (os.environ.get("CUBED_GPU_SCRUB_DEVICE_BEAM", "0") == "1"
                and not containment_active
                and not visual_slots_active):
            device_beam_attempted = True
            try:
                result = _beam_generate_stateful_device()
                return result
            except Exception as exc:  # OOM/foreign scorer -> whole-attempt fallback
                device_beam_fallback = (
                    f"{type(exc).__name__}: {exc}")
                global _DEVICE_BEAM_WARNED
                if not _DEVICE_BEAM_WARNED:
                    plane_name = "dense" if dense_prefix_plane else "control"
                    print(f"  [gpu-scrub-device-beam:{plane_name}] "
                          f"fell back to legacy: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    _DEVICE_BEAM_WARNED = True
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 cleanup must not mask fallback
                    pass

        # TRUTH PROBE (record-only): this window runs on the CPU/legacy path,
        # which the probe does not instrument -- record WHY so timeline gaps
        # are attributable instead of silent.
        if tprobe_truth is not None:
            try:
                _tp_reason = (
                    f"device-fallback:{device_beam_fallback}"
                    if device_beam_fallback else
                    ("env-off" if os.environ.get(
                        "CUBED_GPU_SCRUB_DEVICE_BEAM", "0") != "1" else
                     ("seeded" if seed_entries else
                      ("containment-audit" if containment_active else
                       ("visual-slots" if visual_slots_active
                        else "unknown")))))
                tprobe_row(dict(
                    _tprobe_base("window_cpu_path"),
                    device_beam_attempted=bool(device_beam_attempted),
                    reason=_tp_reason))
            except Exception:                   # record-only: never abort
                pass

        def path_root(word, timing):
            if not visual_slots_active:
                return tuple(word), tuple(timing)
            node = _BeamBackpointer(None, word, timing)
            return node, node

        def path_word(value):
            return (value.word() if isinstance(value, _BeamBackpointer)
                    else tuple(value))

        def path_timing(value):
            return (value.timing() if isinstance(value, _BeamBackpointer)
                    else tuple(value))

        def path_word_len(value):
            return (int(value.word_len)
                    if isinstance(value, _BeamBackpointer) else len(value))

        def path_extend(word, timing, moves, frames):
            if not visual_slots_active:
                return tuple(word) + tuple(moves), tuple(timing) + tuple(frames)
            parent = (word if isinstance(word, _BeamBackpointer) else
                      _BeamBackpointer(None, word, timing))
            node = _BeamBackpointer(parent, moves, frames)
            return node, node

        def required_after(req_lane, word, additions):
            if not req_lane:
                return False
            pos = path_word_len(word)
            additions = tuple(int(move) for move in additions)
            return (pos + len(additions) <= len(required_word)
                    and additions == required_word[pos:pos + len(additions)])

        # value = (coherent score, physical word, last move, centre-motion
        # accumulator, per-move timestamps).  The visual CPU path stores word
        # and timing as one persistent backpointer; the zero-visual path keeps
        # the historical tuples literally.
        def put_best(dst, key, value):
            old = dst.get(key)
            if old is None or value[0] > old[0]:
                dst[key] = value

        def _state_matrix(group):
            if os.environ.get("CUBED_SCRUB_FAST_OM", "0") == "1":
                return np.frombuffer(bytearray(
                    b"".join(key[0] for key, _value in group)),
                    np.int8).reshape(len(group), -1)
            return np.stack([
                np.frombuffer(key[0], np.int8) for key, _value in group
            ])

        def advance_chunk_fused(entries, k):
            """Advance fixed-key read runs with one state build/GPU sync.

            Reads cannot change a beam key; only rotations can.  The legacy
            loop nevertheless regrouped orientations, rebuilt every state
            matrix, synchronized CUDA, and rebuilt the dict after each read.
            Fuse consecutive reads between rotations and perform the complete
            chronological score accumulation before returning to the host.
            """
            from detect.trellis_tracker import score_state_group_runs

            cur = entries
            chunk = chunks[k]
            pos = 0
            while pos < len(chunk):
                _frame, _order, kind, _payload, _weight = chunk[pos]
                if kind == "rotation":
                    nxt = {}
                    for (sb, oi, nskip, nins, req_lane,
                         gate_singles), value in cur.items():
                        for dst_oi in {oi} | set(om_nbrs[oi]):
                            put_best(nxt, (sb, int(dst_oi), nskip, nins,
                                           req_lane, gate_singles), value)
                    cur = nxt
                    pos += 1
                    continue

                end = pos
                while end < len(chunk) and chunk[end][2] == "read":
                    end += 1
                read_run = chunk[pos:end]
                by_oi = {}
                for key, value in cur.items():
                    by_oi.setdefault(key[1], []).append((key, value))
                pending = []
                for oi, group in by_oi.items():
                    weighted_segs = []
                    for _rf, _ro, _rk, payload, weight in read_run:
                        si, ri = payload
                        seg = plane_read_scorer(si, ri, oi)
                        if seg is None:
                            weighted_segs = []
                            break
                        weighted_segs.append((seg, float(weight)))
                    if not weighted_segs:
                        continue
                    smat = _state_matrix(group)
                    base = np.fromiter(
                        (value[0] for _key, value in group),
                        dtype=np.float64, count=len(group))
                    pending.append((group, smat, base, weighted_segs))

                finals = score_state_group_runs([
                    (smat, base, weighted_segs)
                    for _group, smat, base, weighted_segs in pending
                ])
                nxt = {}
                for (group, _smat, _base, _reads), scores in zip(
                        pending, finals):
                    for (key, value), score in zip(group, scores):
                        if not np.isfinite(score):
                            continue
                        # Read-only runs cannot create duplicate keys; preserve
                        # the existing order without another dict lookup.
                        nxt[key] = (float(score), value[1], value[2], value[3],
                                    value[4], value[5])
                cur = nxt
                if not cur:
                    break
                pos = end
            return cur

        def advance_chunk(entries, k):
            if (not dense_prefix_plane
                    and os.environ.get("CUBED_GPU_SCRUB_FUSED_READS", "0")
                    == "1"):
                return advance_chunk_fused(entries, k)
            cur = entries
            for _frame, _order, kind, payload, weight in chunks[k]:
                if kind == "rotation":
                    nxt = {}
                    for (sb, oi, nskip, nins, req_lane,
                         gate_singles), value in cur.items():
                        for dst_oi in {oi} | set(om_nbrs[oi]):
                            put_best(nxt, (sb, int(dst_oi), nskip, nins,
                                           req_lane, gate_singles), value)
                    cur = nxt
                    continue
                si, ri = payload
                plane_memory_guard(
                    "max_score_cells", len(cur),
                    f"read-score frontier at span {si} read {ri}")
                nxt = {}
                by_oi = {}
                for key, value in cur.items():
                    by_oi.setdefault(key[1], []).append((key, value))
                pending = []
                for oi, group in by_oi.items():
                    seg = plane_read_scorer(si, ri, oi)
                    if seg is None:
                        continue
                    smat = _state_matrix(group)
                    pending.append((group, seg, smat))
                if os.environ.get("CUBED_GPU_SCRUB_BATCH", "0") == "1":
                    # Runtime import avoids a module cycle: trellis_tracker
                    # imports scrub_decode only after its own initialization.
                    from detect.trellis_tracker import score_state_groups
                    adds = score_state_groups([
                        (seg, smat) for _group, seg, smat in pending])
                else:
                    adds = [seg.score_states(smat)
                            for _group, seg, smat in pending]
                for (group, _seg, _smat), add in zip(pending, adds):
                    for (key, value), delta in zip(group, add):
                        if not np.isfinite(delta):
                            continue
                        put_best(nxt, key,
                                 (value[0] + float(weight) * float(delta),
                                  value[1], value[2], value[3], value[4],
                                  value[5]))
                cur = nxt
                if not cur:
                    break
            return cur

        _visual_score_cache = {}
        visual_step_inputs = {}
        visual_step_audits = {}
        gate_drop_fallback_steps = set()

        def _visual_read_delta(payload, weight, oi, state_bytes, read_scorer,
                               evidence_source):
            """One exact-frame read score for the interval DP (memoized)."""
            si, ri = payload
            key = (str(evidence_source), int(si), int(ri), int(oi),
                   state_bytes)
            cached = _visual_score_cache.get(key)
            if cached is None and key not in _visual_score_cache:
                seg = read_scorer(si, ri, oi)
                if seg is None:
                    _visual_score_cache[key] = None
                else:
                    state_row = np.frombuffer(
                        state_bytes, np.int8)[None, :]
                    cached = float(seg.score_states(state_row)[0])
                    _visual_score_cache[key] = cached
            return (None if cached is None else
                    float(weight) * float(cached))

        def _rotate_visual_lanes(lanes):
            out = {}
            for oi, value in lanes.items():
                for dst_oi in {int(oi)} | set(om_nbrs[int(oi)]):
                    old = out.get(int(dst_oi))
                    if old is None or value[0] > old[0]:
                        out[int(dst_oi)] = value
            return out

        def _score_visual_static(source_key, source_value, items, read_scorer,
                                 evidence_source):
            """Score one visual SKIP lane through its interval observations."""
            sb, start_oi, nskip, nins, req_lane, gate_singles = source_key
            sc, seq, last_mv, timed, merges = source_value
            lanes = {int(start_oi): (float(sc), timed)}
            for _frame, _order, kind, payload, weight in items:
                if kind == "rotation":
                    lanes = _rotate_visual_lanes(lanes)
                    continue
                nxt = {}
                for oi, (lane_score, lane_timed) in lanes.items():
                    delta = _visual_read_delta(
                        payload, weight, oi, sb, read_scorer,
                        evidence_source)
                    if delta is not None:
                        nxt[oi] = (lane_score + delta, lane_timed)
                lanes = nxt
                if not lanes:
                    break
            return {
                (sb, oi, nskip, nins, req_lane, gate_singles):
                (lane_score, seq, last_mv, lane_timed, merges)
                for oi, (lane_score, lane_timed) in lanes.items()
            }

        def _score_phase_skip(source_key, source_value, slot, items,
                              read_scorer, evidence_source):
            """Score one latent stay-or-neighbor whole-cube gesture.

            Unlike a final-gate dropped interval, a certified phase has no
            separate rotation observation: the phase itself is the structural
            gesture proposal.  Reads before a single latent boundary score the
            held OM; reads after it score exactly stay or one graph neighbor.
            No path may traverse two OM edges.
            """
            if any(item[2] == "rotation" for item in items):
                raise ValueError(
                    "certified phase contains an observed rotation event")
            sb, start_oi, nskip, nins, req_lane, gate_singles = source_key
            sc, seq, last_mv, timed, merges = source_value
            destinations = sorted(
                {int(start_oi)} | set(om_nbrs[int(start_oi)]))
            pre = {int(start_oi): float(sc)}
            post = {}

            def promote(frame):
                if int(start_oi) not in pre:
                    return
                lane_score = pre[int(start_oi)]
                for dst_oi in destinations:
                    old = post.get(int(dst_oi))
                    candidate = (float(lane_score), int(frame))
                    if old is None or candidate[0] > old[0]:
                        post[int(dst_oi)] = candidate

            grouped = []
            for item in items:
                if not grouped or int(grouped[-1][0][0]) != int(item[0]):
                    grouped.append([item])
                else:
                    grouped[-1].append(item)
            for group in grouped:
                frame = int(group[0][0])
                if frame > int(slot.frame_lo):
                    promote(max(int(slot.frame_lo), frame - 1))
                promote(frame)
                reads_here = [item for item in group if item[2] == "read"]
                for _rf, _ro, _kind, payload, weight in reads_here:
                    next_pre = {}
                    for oi, lane_score in pre.items():
                        delta = _visual_read_delta(
                            payload, weight, oi, sb, read_scorer,
                            evidence_source)
                        if delta is not None:
                            next_pre[oi] = lane_score + delta
                    next_post = {}
                    for oi, (lane_score, boundary) in post.items():
                        delta = _visual_read_delta(
                            payload, weight, oi, sb, read_scorer,
                            evidence_source)
                        if delta is not None:
                            next_post[oi] = (lane_score + delta, boundary)
                    pre, post = next_pre, next_post
                    if not pre and not post:
                        break
            promote(int(slot.frame_hi))
            return {
                (sb, oi, nskip, nins, req_lane, gate_singles):
                (lane_score, seq, last_mv, timed, merges)
                for oi, (lane_score, _boundary) in post.items()
            }

        def _score_visual_move(source_key, source_value, slot, mi, items,
                               read_scorer, evidence_source):
            """Score one MOVE with a single latent pre→post boundary.

            ``pre`` and ``post`` are separate dynamic-programming phases.  The
            only phase transition is pre→post; both phases follow the same exact
            rotation graph and read timestamps, so evidence can never alternate
            back to the pre-state.  A boundary is admitted before and after a
            rotation frame when the closed episode contains both possibilities.
            """
            sb, start_oi, nskip, nins, req_lane, gate_singles = source_key
            if (gate_drop_single_cap is not None
                    and str(slot.kind) == "dropped"
                    and int(gate_singles) >= int(gate_drop_single_cap)):
                return {}
            sc, seq, _last_mv, timed, merges = source_value
            pre_state = np.frombuffer(sb, np.int8)
            post_state = pre_state[perms[int(mi)]]
            post_bytes = post_state.tobytes()
            req1 = required_after(req_lane, seq, (mi,))
            # lane value = (score, chosen move frame)
            pre = {int(start_oi): (float(sc), None)}
            post = {}

            def promote(frame):
                for oi, (lane_score, _none) in pre.items():
                    old = post.get(oi)
                    cand = (float(lane_score), int(frame))
                    if old is None or cand[0] > old[0]:
                        post[oi] = cand

            # Group observations by exact frame so rotations remain pre-action
            # for a same-frame read while an unobserved gap can place the move on
            # either side of a rotation without inventing a duration tolerance.
            grouped = []
            for item in items:
                if not grouped or int(grouped[-1][0][0]) != int(item[0]):
                    grouped.append([item])
                else:
                    grouped[-1].append(item)
            for group in grouped:
                frame = int(group[0][0])
                if frame > int(slot.frame_lo) and pre:
                    promote(max(int(slot.frame_lo), frame - 1))
                rotations = [item for item in group if item[2] == "rotation"]
                reads_here = [item for item in group if item[2] == "read"]
                if optional_action_allows_om_transition(slot.kind, mi):
                    for _rotation in rotations:
                        pre = _rotate_visual_lanes(pre)
                        post = _rotate_visual_lanes(post)
                if pre:
                    promote(frame)
                for _rf, _ro, _kind, payload, weight in reads_here:
                    next_pre, next_post = {}, {}
                    for oi, (lane_score, boundary) in pre.items():
                        delta = _visual_read_delta(
                            payload, weight, oi, sb, read_scorer,
                            evidence_source)
                        if delta is not None:
                            next_pre[oi] = (lane_score + delta, boundary)
                    for oi, (lane_score, boundary) in post.items():
                        delta = _visual_read_delta(
                            payload, weight, oi, post_bytes, read_scorer,
                            evidence_source)
                        if delta is not None:
                            next_post[oi] = (lane_score + delta, boundary)
                    pre, post = next_pre, next_post
                    if not pre and not post:
                        break
            if pre:
                promote(int(slot.frame_hi))
            out = {}
            for oi, (lane_score, boundary) in post.items():
                seq1, timed1 = path_extend(
                    seq, timed, (int(mi),), (int(boundary),))
                next_gate_singles = (
                    int(gate_singles) + 1
                    if (gate_drop_single_cap is not None
                        and str(slot.kind) == "dropped")
                    else int(gate_singles))
                out[(post_bytes, oi, nskip, nins, req1,
                     next_gate_singles)] = (
                    lane_score, seq1, int(mi), timed1, merges)
            return out

        def prepare_visual_step(source_items, k, slot,
                                force_control_reason=None):
            """Choose guarded dense-vs-control evidence for one visual slot."""
            if k in visual_step_inputs:
                return
            control_items = interval_chunks[k - 1]
            if dense_prefix_plane:
                # The independent dense-prefix generator already partitioned
                # its exact-frame stream over these immutable typed slots.
                # It never falls back inside a slot: any scorer/bound failure
                # aborts this plane and leaves the separately completed
                # control generator untouched.
                visual_step_inputs[k] = (
                    control_items, plane_read_scorer, "dense-prefix")
                visual_step_audits[k] = dict(
                    slot_index=int(k), kind=str(slot.kind),
                    interval=[int(slot.frame_lo), int(slot.frame_hi)],
                    phase_provenance=(
                        dict(parent_frame=int(slot.parent_frame),
                             period=[int(slot.period_lo), int(slot.period_hi)],
                             phase_index=int(slot.phase_index),
                             phase_count=int(slot.phase_count))
                        if slot.kind == "phase" else None),
                    evidence_source="dense-prefix", fallback_reason=None,
                    dense_guard=dict(
                        checked=True, admissible=True,
                        read_state_cells=0,
                        read_state_bound=int(BS.SCORE_STATE_BOUND)),
                    actions={action: dict(
                        action=("SKIP" if action is None
                                else BS.MOVES[action]),
                        move_index=(None if action is None else int(action)),
                        best_score=None, expanded_keys=set(),
                        band_retained=False, retained=False)
                             for action in visual_slot_actions()})
                return
            dense_items = dense_interval_chunks[k - 1]
            source = "control"
            reason = (str(force_control_reason)
                      if force_control_reason is not None else
                      ("dense-disabled" if not dense_evidence else
                       "no-owned-dense-reads"))
            guard_row = dict(checked=False, admissible=None,
                             read_state_cells=0,
                             read_state_bound=int(BS.SCORE_STATE_BOUND))
            scorer = read_seg
            dense_reads = [item for item in dense_items
                           if item[2] == "read"]
            if (force_control_reason is None
                    and dense_evidence and dense_reads):
                state_bytes = set()
                for key, _value in source_items:
                    sb = key[0]
                    state_bytes.add(sb)
                    state = np.frombuffer(sb, np.int8)
                    state_bytes.update(
                        state[perms[mi]].tobytes() for mi in range(18))
                admissible, cells, bound = dense_grid_guard(
                    len(dense_reads), len(state_bytes),
                    context=(f"visual-slot:{int(slot.frame_lo)}-"
                             f"{int(slot.frame_hi)}"))
                guard_row = dict(
                    checked=True, admissible=bool(admissible),
                    n_reads=len(dense_reads), n_states=len(state_bytes),
                    read_state_cells=int(cells),
                    read_state_bound=int(bound))
                if not admissible:
                    reason = "dense-read-state-guard"
                    guard = report["dense_evidence"]["scale_guard"]
                    guard["visual_control_fallbacks"] = int(
                        guard.get("visual_control_fallbacks", 0)) + 1
                    record_dense_control_fallback(
                        f"visual-slot:{int(slot.frame_lo)}-"
                        f"{int(slot.frame_hi)}:scale-guard")
                else:
                    prepare_error = None
                    try:
                        usable = all(
                            score_read_seg(si, ri, oi) is not None
                            for _frame, _order, _kind, (si, ri), _weight
                            in dense_reads
                            for oi in range(n_om))
                    except Exception as exc:  # noqa: BLE001 evidence boundary
                        usable = False
                        prepare_error = exc
                    if usable:
                        source = "dense-owned-episode"
                        reason = None
                        scorer = score_read_seg
                        control_rotations = [
                            item for item in control_items
                            if item[2] == "rotation"]
                        visual_step_inputs[k] = (
                            control_rotations + dense_reads, scorer, source)
                        visual_step_inputs[k][0].sort(
                            key=lambda item: (item[0], item[1]))
                    else:
                        reason = ("dense-segment-error" if prepare_error
                                  is not None else "dense-segment-unusable")
                        guard = report["dense_evidence"]["scale_guard"]
                        guard["visual_control_fallbacks"] = int(
                            guard.get("visual_control_fallbacks", 0)) + 1
                        record_dense_control_fallback(
                            f"visual-slot:{int(slot.frame_lo)}-"
                            f"{int(slot.frame_hi)}:prepare",
                            prepare_error)
            if k not in visual_step_inputs:
                visual_step_inputs[k] = (control_items, scorer, source)
            visual_step_audits[k] = dict(
                slot_index=int(k), kind=str(slot.kind),
                interval=[int(slot.frame_lo), int(slot.frame_hi)],
                phase_provenance=(
                    dict(parent_frame=int(slot.parent_frame),
                         period=[int(slot.period_lo), int(slot.period_hi)],
                         phase_index=int(slot.phase_index),
                         phase_count=int(slot.phase_count))
                    if slot.kind == "phase" else None),
                evidence_source=source, fallback_reason=reason,
                dense_guard=guard_row,
                actions={action: dict(
                    action=("SKIP" if action is None else BS.MOVES[action]),
                    move_index=(None if action is None else int(action)),
                    best_score=None, expanded_keys=set(),
                    band_retained=False, retained=False)
                         for action in visual_slot_actions()})

        def fallback_visual_step_to_control(k, reason, error=None):
            """Abort one dense visual transaction without masking control."""
            interval, _scorer, source = visual_step_inputs[k]
            if source != "dense-owned-episode":
                return False
            visual_step_inputs[k] = (interval_chunks[k - 1], read_seg,
                                     "control")
            audit = visual_step_audits[k]
            audit["evidence_source"] = "control"
            audit["fallback_reason"] = str(reason)
            if error is not None:
                audit["dense_score_error"] = (
                    f"{type(error).__name__}: {error}")
            for action_row in audit["actions"].values():
                action_row.update(
                    best_score=None, expanded_keys=set(),
                    band_retained=False, retained=False)
            guard = report["dense_evidence"]["scale_guard"]
            guard["visual_control_fallbacks"] = int(
                guard.get("visual_control_fallbacks", 0)) + 1
            context = (f"visual-slot:{audit['interval'][0]}-"
                       f"{audit['interval'][1]}:{reason}")
            record_dense_control_fallback(context, error)
            if error is not None:
                guard.setdefault("visual_score_errors", []).append({
                    "context": context,
                    "error": audit["dense_score_error"],
                })
            return True

        def expand_visual(source_items, k, slot):
            """Expand one optional visual slot: SKIP + 18 SINGLE, never DOUBLE."""
            interval, read_scorer, evidence_source = visual_step_inputs[k]
            audit = visual_step_audits[k]
            action_maps = {}
            plane_memory_guard(
                "max_score_cells", len(source_items),
                f"typed-slot score frontier {slot.frame_lo}-{slot.frame_hi}")

            def score_interval_actions():
                scored_actions = {}
                for action in visual_slot_actions():
                    action_nxt = {}
                    for source_key, source_value in source_items:
                        if action is None:
                            if slot.kind == "phase":
                                scored = _score_phase_skip(
                                    source_key, source_value, slot, interval,
                                    read_scorer, evidence_source)
                            else:
                                scored = _score_visual_static(
                                    source_key, source_value, interval,
                                    read_scorer, evidence_source)
                        else:
                            scored = _score_visual_move(
                                source_key, source_value, slot, action,
                                interval, read_scorer, evidence_source)
                        for key, value in scored.items():
                            put_best(action_nxt, key, value)
                    scored_actions[action] = action_nxt
                return scored_actions

            try:
                action_maps = score_interval_actions()
            except Exception as exc:
                if evidence_source != "dense-owned-episode":
                    raise
                fallback_visual_step_to_control(
                    k, "dense-score-error", error=exc)
                interval, read_scorer, evidence_source = visual_step_inputs[k]
                # Deliberately outside another catch: control scorer/program
                # failures retain the orchestrator's existing fallback.
                action_maps = score_interval_actions()

            nxt = {}
            for action in visual_slot_actions():
                action_nxt = action_maps[action]
                action_nxt = advance_chunk(action_nxt, k)
                action_audit = audit["actions"][action]
                action_audit["expanded_keys"].update(action_nxt)
                if action_nxt:
                    best_score = max(value[0]
                                     for value in action_nxt.values())
                    old_score = action_audit["best_score"]
                    action_audit["best_score"] = (
                        float(best_score) if old_score is None else
                        max(float(old_score), float(best_score)))
                for key, value in action_nxt.items():
                    put_best(nxt, key, value)
            return nxt

        def expand_gate_drop_control(source_items, k, slot, reason,
                                     error=None):
            """Replay one dropped slot as the legacy rotation-only SKIP.

            This is a feature-local rollback: before optional actuation, every
            final-gate-dropped gap is an OM-neighbor opportunity and emits no
            layer move. A bound or scoring failure returns to that exact
            semantics without abandoning the rest of scrub.
            """
            fallback_visual_step_to_control(k, reason, error=error)
            audit = visual_step_audits[k]
            audit["fallback_reason"] = str(reason)
            audit["gate_drop_control_fallback"] = True
            gate_report = (None if dense_prefix_plane else
                           report.get("gate_drop_slots"))
            marker = (int(k), str(reason), error is not None)
            if (gate_report is not None
                    and marker not in gate_drop_fallback_steps):
                gate_drop_fallback_steps.add(marker)
                key = ("score_error_fallbacks" if error is not None
                       else "bound_fallbacks")
                gate_report[key] = int(gate_report.get(key, 0)) + 1
                if error is not None:
                    gate_report.setdefault("score_errors", []).append({
                        "interval": [int(slot.frame_lo), int(slot.frame_hi)],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    })
            action_nxt = {}
            for source_key, source_value in source_items:
                scored = _score_visual_static(
                    source_key, source_value, interval_chunks[k - 1],
                    read_seg, "control")
                for key, value in scored.items():
                    put_best(action_nxt, key, value)
            action_nxt = advance_chunk(action_nxt, k)
            skip_audit = audit["actions"][None]
            skip_audit["expanded_keys"].update(action_nxt)
            if action_nxt:
                skip_audit["best_score"] = float(max(
                    value[0] for value in action_nxt.values()))
            return action_nxt

        band = float(z_sigma) * sigma
        # Dense candidacy is ranked by dense reads alone, but it still carries
        # the control plane's candidate structure.
        required_word = tuple(required_word or ())
        all_read_frames = [
            int(item[0])
            for chunk in tuple(chunks) + tuple(interval_chunks)
            for item in chunk if item[2] == "read"
        ]
        read_chunks = [k for k, chunk in enumerate(chunks)
                       if any(item[2] == "read" for item in chunk)]
        read_chunks.extend(
            k + 1 for k, chunk in enumerate(interval_chunks)
            if any(item[2] == "read" for item in chunk))
        last_read_chunk = max(read_chunks, default=0)

        # Exact extension checkpointing is certified only for the ordinary
        # independent dense plane.  In particular, containment/teacher lanes,
        # seeds, optional slots, and foreign scorers all run cold. Those paths
        # either have side effects or carry
        # provenance not represented by the immutable tuple frontier.
        checkpoint_eligible = bool(
            dense_prefix_plane
            and truth is None and win_ctx is None
            and not seed_entries and not visual_slots_active
            and not containment_active
            and gate_drop_single_cap is None
            and prefix_read_frames is dense_checkpoint_source[0]
            and prefix_read_scorer is dense_checkpoint_source[1]
            # CUBED_SCRUB_SLOT_MERGE: `static_key` below does
            # not carry slot-merge physical-adjacency state, so a cached
            # otherwise be wrongly resumed across incompatible merge
            # conditions if this window is ALSO merge-engageable.  Excluded
            # here exactly like visual_slots_active above -- both flags
            # default off, so this is a no-op absent either.
            and not slot_merge_engageable
            and count_interval is None)
        checkpoint_family = None
        checkpoint_key = None
        resume_k = -1
        checkpoint = None
        if checkpoint_eligible:
            if dense_beam_checkpoint_runtime["a_span"] != int(a_span):
                dense_beam_checkpoint_cache.clear()
                dense_beam_checkpoint_runtime["a_span"] = int(a_span)
            chunk_signatures = tuple(tuple(
                (int(frame), int(order), str(kind),
                 None if read_id is None else tuple(map(int, read_id)),
                 None if weight is None else float(weight))
                for frame, order, kind, read_id, weight in chunk_items)
                for chunk_items in chunks)
            witness_bits = tuple(
                _slot_has_endpoint_evidence(
                    int(slot.frame_hi), all_read_frames)
                for slot in action_slots)
            perms_arr = np.asarray(perms)
            static_key = (
                int(a_span), int(f_start),
                np.asarray(state, dtype=np.int8).tobytes(),
                tuple(sorted(map(int, start_ois))), int(n_om),
                int(beam_k), float(band), int(BS.SCORE_STATE_BOUND),
                tuple(tuple(sorted(map(int, row))) for row in om_nbrs),
                (str(perms_arr.dtype), tuple(perms_arr.shape),
                 perms_arr.tobytes()),
                tuple(BS.MOVES), tuple(BS.FACE_OF),
                id(plane_read_frames), id(plane_read_scorer),
                os.environ.get("CUBED_SCRUB_FAST_OM", "0") == "1",
                os.environ.get("CUBED_GPU_SCRUB_BATCH", "0") == "1",
                raw_burst_stream)
            checkpoint_family = dense_beam_checkpoint_cache.setdefault(
                static_key, {})

            def checkpoint_key(step):
                return _dense_beam_checkpoint_key(
                    step, skip_cap=skip_cap, insert_cap=insert_cap,
                    slots=action_slots, chunk_signatures=chunk_signatures,
                    witness_bits=witness_bits, required_word=required_word)

            resume_k, checkpoint = _longest_exact_beam_checkpoint(
                checkpoint_family, checkpoint_key, len(action_slots))
            dense_beam_checkpoint_runtime["calls"] += 1
            dense_beam_checkpoint_runtime["slots_total"] += len(action_slots)
            dense_beam_checkpoint_runtime["slots_skipped"] += max(0, resume_k)
            if resume_k > 0:
                dense_beam_checkpoint_runtime["resumed_calls"] += 1

        st0 = np.asarray(state, np.int8)
        if checkpoint is not None:
            # Ordinary dense-plane keys and values are immutable tuples.  Copy
            # the dict so a resumed attempt cannot alias the stored frontier.
            beam = dict(checkpoint["beam"])
            seed_word_collision = False
            n_prn = int(checkpoint["n_prn"])
            capped = bool(checkpoint["capped"])
            max_live_pairs = int(checkpoint["max_live_pairs"])
            for field, value in checkpoint["plane_stats"].items():
                plane_max(field, value)
        else:
            if seed_entries:
                beam = {}
                seed_word_collision = False
                for entry in seed_entries:
                    key = (entry["state"], int(entry["oi"]), 0, 0,
                           bool(entry.get("is_incumbent")), 0)
                    old = beam.get(key)
                    if (old is not None
                            and path_word(old[1]) != tuple(entry["word"])):
                        seed_word_collision = True
                    seed_word, seed_timing = path_root(entry["word"], ())
                    put_best(
                        beam, key,
                        (float(entry["score_rel"]), seed_word, None,
                         seed_timing, ()))
            else:
                seed_word_collision = False
                empty_word, empty_timing = path_root((), ())
                beam = {
                    (st0.tobytes(), oi, 0, 0, True, 0):
                    (0.0, empty_word, None, empty_timing, ())
                    for oi in sorted(start_ois)
                }
            beam = advance_chunk(beam, 0)
            n_prn, capped = 0, False
            max_live_pairs = len(beam)
            plane_memory_guard(
                "max_frontier", max_live_pairs, "initial frontier")
            resume_k = 0
            if checkpoint_family is not None:
                checkpoint_family[checkpoint_key(0)] = dict(
                    beam=dict(beam), n_prn=n_prn, capped=capped,
                    max_live_pairs=max_live_pairs,
                    plane_stats=dict(prefix_plane_stats))

        def required_prefix(kv):
            return bool(kv[0][4])

        def truth_word_fields(entries, truth_row):
            """Shadow-only exact physical-word containment.

            State-trajectory membership is intentionally notation agnostic,
            but it is too weak to identify the first lost continuation: an
            unrelated candidate can revisit any earlier GT state.  Track the
            literal GT move prefix alongside it so the audit distinguishes a
            generator/timing loss from a later endpoint-state loss.  This
            helper only observes already-ranked beam values and never changes
            their score, key, or retention.
            """
            try:
                gt_word = tuple(truth_row["physical_word"])
            except (KeyError, ValueError):
                return dict(truth_word_prefix_survived=False,
                            truth_word_prefix_rank=None,
                            truth_word_exact_survived=False,
                            truth_word_exact_rank=None)
            rows = sorted(entries, key=lambda kv: -kv[1][0])
            prefix_ranks = [
                rank for rank, kv in enumerate(rows)
                if (path_word_len(kv[1][1]) <= len(gt_word)
                    and path_word(kv[1][1]) ==
                    gt_word[:path_word_len(kv[1][1])])]
            exact_ranks = [
                rank for rank, kv in enumerate(rows)
                if path_word(kv[1][1]) == gt_word]
            prefix_lengths = [path_word_len(rows[rank][1][1])
                              for rank in prefix_ranks]
            max_prefix_len = (max(prefix_lengths)
                              if prefix_lengths else None)
            max_prefix_ranks = [
                rank for rank in prefix_ranks
                if path_word_len(rows[rank][1][1]) == max_prefix_len]
            max_prefix_rank = (min(max_prefix_ranks)
                               if max_prefix_ranks else None)
            best_score = float(rows[0][1][0]) if rows else None
            prefix_score = (float(rows[max_prefix_rank][1][0])
                            if max_prefix_rank is not None else None)
            by_length = {}
            for rank in prefix_ranks:
                length = path_word_len(rows[rank][1][1])
                slot = by_length.get(length)
                score = float(rows[rank][1][0])
                if slot is None:
                    by_length[length] = dict(
                        count=1, best_rank=rank,
                        best_delta=(best_score - score
                                    if best_score is not None else None))
                else:
                    slot["count"] += 1
            return dict(
                truth_word_prefix_survived=bool(prefix_ranks),
                truth_word_prefix_rank=(min(prefix_ranks)
                                        if prefix_ranks else None),
                truth_word_prefix_max_len=max_prefix_len,
                truth_word_prefix_max_rank=max_prefix_rank,
                truth_word_prefix_max_delta=(
                    best_score - prefix_score
                    if best_score is not None and prefix_score is not None
                    else None),
                truth_word_prefix_by_length=by_length,
                truth_word_exact_survived=bool(exact_ranks),
                truth_word_exact_rank=(min(exact_ranks)
                                       if exact_ranks else None))

        if seed_entries and beam:
            # A synthetic split is a computational factorization seam, not a
            # state decision.  Preserve its complete joint state/OM likelihood
            # band plus the incumbent-control lane.  Oversized next-step work is
            # synchronously streamed below against one global cutoff, so no
            # lossy K cut is needed here.
            ranked_seed = sorted(beam.items(), key=lambda kv: -kv[1][0])
            best_seed = ranked_seed[0][1][0]
            in_band = [kv for kv in ranked_seed
                       if kv[1][0] >= best_seed - band]
            req_seed = [kv for kv in ranked_seed if kv[0][4]]
            seeded = {kv[0]: kv for kv in in_band}
            for kv in req_seed:
                seeded.setdefault(kv[0], kv)
            n_prn += len(ranked_seed) - len(seeded)
            beam = dict(seeded.values())
            if containment_active and truth is not None:
                tset = truth["traj_bytes"]
                rank_b = next(
                    (ix for ix, kv in enumerate(ranked_seed)
                     if kv[0][0] in tset), None)
                _emit_caud(
                    win_ctx, a_span, b_span, "seed_band", 0,
                    len(ranked_seed), len(beam), truth,
                    any(key[0] in tset for key in beam),
                    truth["end_bytes"] in {key[0] for key in beam}, rank_b,
                    extra=dict(prefix_om_mode="chronological",
                               burst_frames=bursts,
                               read_frames=all_read_frames,
                               skip_cap=int(skip_cap),
                               insert_cap=int(insert_cap),
                               truth_word_before=truth_word_fields(
                                   ranked_seed, truth),
                               truth_word_after=truth_word_fields(
                                   list(beam.items()), truth)))
            print(f"  [scrub-joint] temporal seed band "
                  f"{len(ranked_seed)}->{len(beam)} "
                  f"(complete + incumbent)", flush=True)

        def expand_burst(source_items, k, burst_frame):
            """Expand one action slot for a bounded source partition.

            Slot merge deliberately does not rewrite this CPU payload: doing
            so would hide a physical quarter action from the NN/trajectory
            ABI.  The resident controller carries the dual physical/search
            representation; a device failure returns here with the historical
            physical program and zero merge receipts."""
            nxt = {}
            for (sb, oi, nskip, nins, req_lane,
                 gate_singles), (sc, seq, last_mv, timed,
                                 slot_merges) in source_items:
                s = np.frombuffer(sb, np.int8)
                if nskip < skip_cap:
                    put_best(nxt, (sb, oi, nskip + 1, nins, req_lane,
                                   gate_singles),
                             (sc, seq, None, timed, slot_merges))
                after_evidence = not _slot_has_endpoint_evidence(
                    burst_frame, all_read_frames)
                if after_evidence:
                    continue
                for mi in range(18):
                    s1 = s[perms[mi]]
                    req1 = required_after(req_lane, seq, (mi,))
                    seq1, timed1 = path_extend(
                        seq, timed, (mi,), (burst_frame,))
                    put_best(nxt, (s1.tobytes(), oi, nskip, nins, req1,
                                   gate_singles),
                             (sc, seq1, mi, timed1, slot_merges))
                    if nins < insert_cap:
                        for mj in range(18):
                            if BS.FACE_OF[mj] == BS.FACE_OF[mi]:
                                continue
                            s2 = s1[perms[mj]]
                            seq2, timed2 = path_extend(
                                seq, timed, (mi, mj),
                                (burst_frame, burst_frame))
                            req2 = required_after(
                                req_lane, seq, (mi, mj))
                            put_best(nxt,
                                     (s2.tobytes(), oi, nskip, nins + 1,
                                      req2, gate_singles),
                                     (sc, seq2, None,
                                      timed2, slot_merges))
            return advance_chunk(nxt, k)

        cpu_count_feasibility_steps = 0
        cpu_count_feasibility_rows_removed = 0
        cpu_terminal_count_min = None
        cpu_terminal_count_max = None
        for k, slot in enumerate(action_slots, 1):
            if k <= resume_k:
                continue
            is_optional = slot.kind != "motion"
            is_dropped = slot.kind == "dropped"
            is_phase = slot.kind == "phase"
            burst_frame = int(slot.frame_hi)
            gate_drop_control = False
            gate_work = gate_skip_work = None
            phase_work = phase_skip_work = None
            if is_dropped:
                gate_rotation_steps = sum(
                    item[2] == "rotation"
                    for item in interval_chunks[k - 1])
                if gate_drop_single_cap is None:
                    frontier_ois = [key[1] for key in beam]
                    gate_work, gate_skip_work = gate_drop_step_work(
                        frontier_ois, om_nbrs, gate_rotation_steps)
                    admissible, checked_work, _gate_bound = (
                        gate_drop_step_admissible(
                            frontier_ois, om_nbrs, gate_rotation_steps))
                else:
                    frontier_oi_uses = [
                        (key[1], key[5]) for key in beam]
                    gate_work, gate_skip_work = gate_drop_capped_step_work(
                        frontier_oi_uses, om_nbrs, gate_rotation_steps,
                        gate_drop_single_cap)
                    admissible, checked_work, _gate_bound = (
                        gate_drop_capped_step_admissible(
                            frontier_oi_uses, om_nbrs,
                            gate_rotation_steps, gate_drop_single_cap))
                if checked_work != gate_work:
                    raise AssertionError(
                        "gate-drop expansion accounting drift")
                gate_report = (None if dense_prefix_plane else
                               report.get("gate_drop_slots"))
                if gate_report is not None:
                    gate_report["slot_evaluations"] = int(
                        gate_report.get("slot_evaluations", 0)) + 1
                    gate_report["max_step_expansions"] = max(
                        int(gate_report.get("max_step_expansions", 0)),
                        int(gate_work))
                gate_drop_control = not admissible
                if gate_drop_control and gate_drop_single_cap is not None:
                    stats = dict(
                        kept=0, tie_size=0, overflow=True,
                        beam_limit=int(beam_k) * n_om,
                        prefix_om_mode="chronological",
                        max_live_pairs=int(max_live_pairs),
                        gate_drop_single_cap=int(gate_drop_single_cap),
                        max_gate_drop_singles=max(
                            (int(key[5]) for key in beam), default=0),
                        scaling_tripped=(
                            f"globally-capped gate-drop expansions "
                            f"{gate_work} > {BS.SCORE_STATE_BOUND}"))
                    return {}, {}, n_prn, True, stats
                if dense_prefix_plane and gate_drop_control:
                    raise RuntimeError(
                        "dense-prefix gate-drop expansion exceeds "
                        "the shared state bound")
            if is_phase:
                frontier_ois = [key[1] for key in beam]
                (admissible, phase_work, phase_skip_work,
                 _phase_bound) = phase_step_admissible(
                     frontier_ois, om_nbrs)
                checked_work, checked_skip = phase_step_work(
                    frontier_ois, om_nbrs)
                if (checked_work != phase_work
                        or checked_skip != phase_skip_work):
                    raise AssertionError(
                        "intraburst phase expansion accounting drift")
                if not dense_prefix_plane:
                    phase_report = report["intraburst_phase_slots"]
                    phase_report["slot_evaluations"] = int(
                        phase_report.get("slot_evaluations", 0)) + 1
                    phase_report["max_step_expansions"] = max(
                        int(phase_report.get("max_step_expansions", 0)),
                        int(phase_work))
                    phase_report["max_skip_expansions"] = max(
                        int(phase_report.get("max_skip_expansions", 0)),
                        int(phase_skip_work))
                if not admissible:
                    stats = dict(
                        kept=0, tie_size=0, overflow=True,
                        beam_limit=int(beam_k) * n_om,
                        prefix_om_mode="chronological",
                        max_live_pairs=int(max_live_pairs),
                        scaling_tripped=(
                            f"intraburst phase expansions {phase_work} > "
                            f"{BS.SCORE_STATE_BOUND}"))
                    return {}, {}, n_prn, True, stats
            if is_optional:
                prepare_visual_step(
                    list(beam.items()), k, slot,
                    force_control_reason=(
                        "gate-drop-expansion-bound"
                        if gate_drop_control else None))
            # A visual interval is itself an evidence-localized action
            # opportunity, so it always exposes its fixed SKIP/SINGLE alphabet.
            # Motion points retain the legacy endpoint-witness suppression.
            after_evidence = (False if is_optional else
                              not _slot_has_endpoint_evidence(
                                  burst_frame, all_read_frames))
            branch = (1 if gate_drop_control else
                      (len(visual_slot_actions()) if is_optional else
                      (1 if after_evidence else
                       1 + 18 + (18 * 15 if insert_cap > 0 else 0))))
            expansion_work = (
                int(gate_skip_work if gate_drop_control else gate_work)
                if is_dropped else
                (int(phase_work) if is_phase else len(beam) * branch))
            if is_phase:
                visual_step_audits[k].update(
                    expansion_work=int(phase_work),
                    skip_expansion_work=int(phase_skip_work),
                    shared_state_bound=int(BS.SCORE_STATE_BOUND),
                    om_transition_policy=(
                        "SKIP: latent stay-or-one-neighbor without a required "
                        "rotation observation; SINGLE: held OM"))
            plane_max("max_expansion_work", expansion_work)
            if (is_optional and not gate_drop_control
                    and visual_step_inputs[k][2] == "dense-owned-episode"
                    and expansion_work > BS.SCORE_STATE_BOUND):
                # The synchronized streaming scheduler may call a partition
                # twice and has no rollback seam.  Dense scoring is therefore
                # admitted only when the whole visual transaction fits the
                # existing expansion envelope; otherwise use the exact legacy
                # control scheduler rather than risk a mixed-source frontier.
                fallback_visual_step_to_control(
                    k, "dense-transaction-streaming-envelope")
            # A typed dropped slot is an optional rescue transaction.  If its
            # exact SKIP/OM fanout alone exceeds the shared bound, fail closed
            # to the already-completed primary search instead of partitioning
            # a feature-local rollback across batches.
            streamed = bool(
                seed_entries and not is_dropped and not is_phase
                and expansion_work > BS.SCORE_STATE_BOUND)
            plane_memory_guard(
                "max_expansion_cells",
                min(expansion_work, int(BS.SCORE_STATE_BOUND))
                if streamed else expansion_work,
                "resident expansion")
            if not streamed and expansion_work > BS.SCORE_STATE_BOUND:
                stats = dict(
                    kept=0, tie_size=0, overflow=True,
                    beam_limit=int(beam_k) * n_om,
                    prefix_om_mode="chronological",
                    max_live_pairs=int(max_live_pairs),
                    scaling_tripped=(
                        f"prefix expansions {expansion_work} > "
                        f"{BS.SCORE_STATE_BOUND}"))
                return {}, {}, n_prn, True, stats
            def expand_slot(items):
                if is_optional:
                    if gate_drop_control:
                        return expand_gate_drop_control(
                            items, k, slot, "gate-drop-expansion-bound")
                    try:
                        return expand_visual(items, k, slot)
                    except Exception as exc:
                        if (dense_prefix_plane
                                or gate_drop_single_cap is not None):
                            raise
                        if not is_dropped:
                            raise
                        return expand_gate_drop_control(
                            items, k, slot, "gate-drop-score-error",
                            error=exc)
                return expand_burst(items, k, burst_frame)

            streamed_preband_n = None
            if streamed:
                # Two-pass synchronized expansion.  Partitioning is only a
                # memory schedule: pass 1 finds ONE global best for this exact
                # temporal step; pass 2 globally deduplicates and retains
                # against that cutoff.  No batch-local posterior exists.
                source = list(beam.items())
                chunk_n = max(1, BS.SCORE_STATE_BOUND // branch)
                global_best = -np.inf
                lane_best = {}
                streamed_preband_n = 0
                for pos in range(0, len(source), chunk_n):
                    part = expand_slot(source[pos:pos + chunk_n])
                    streamed_preband_n += len(part)
                    if part:
                        global_best = max(
                            global_best,
                            max(value[0] for value in part.values()))
                        for key, value in part.items():
                            lane = (key[2], key[3])
                            lane_best[lane] = max(
                                lane_best.get(lane, -np.inf), value[0])
                if not np.isfinite(global_best):
                    beam = {}
                    break
                ranked_before = {}
                for pos in range(0, len(source), chunk_n):
                    part = expand_slot(source[pos:pos + chunk_n])
                    for key, value in part.items():
                        kv = (key, value)
                        lane = (key[2], key[3])
                        if (value[0] >= lane_best[lane] - band
                                or required_prefix(kv)):
                            put_best(ranked_before, key, value)
                    if len(ranked_before) > BS.SCORE_STATE_BOUND:
                        stats = dict(
                            kept=0, tie_size=0, overflow=True,
                            beam_limit=int(beam_k) * n_om,
                            prefix_om_mode="chronological",
                            max_live_pairs=int(max_live_pairs),
                            scaling_tripped=(
                                "streamed retained prefixes "
                                f"{len(ranked_before)} > "
                                f"{BS.SCORE_STATE_BOUND}"))
                        return {}, {}, n_prn, True, stats
                n_prn += max(
                    0, int(streamed_preband_n) - len(ranked_before))
            else:
                ranked_before = expand_slot(list(beam.items()))
            if count_interval is not None and ranked_before:
                cpu_count_feasibility_steps += 1
                count_lo, count_hi = count_interval
                count_before = len(ranked_before)
                ranked_before = {
                    key: value for key, value in ranked_before.items()
                    if bool(_physical_count_feasible_mask(
                        step=k, n_slots=len(action_slots), nskip=key[2],
                        nins=key[3], skip_cap=skip_cap,
                        insert_cap=insert_cap, L_lo=count_lo, L_hi=count_hi))
                }
                cpu_count_feasibility_rows_removed += (
                    count_before - len(ranked_before))
                if k == len(action_slots) and ranked_before:
                    terminal_counts = [
                        int(k) - int(key[2]) + int(key[3])
                        for key in ranked_before]
                    cpu_terminal_count_min = min(terminal_counts)
                    cpu_terminal_count_max = max(terminal_counts)
            if not ranked_before:
                beam = {}
                break
            ranked = sorted(ranked_before.items(), key=lambda kv: -kv[1][0])
            best = ranked[0][1][0]
            if seed_entries:
                # SKIP/DOUBLE allocations are alternative alignments of the
                # independent count/timing observations.  A premature DOUBLE
                # must not globally delete the still-plausible SINGLE lane
                # before later reads can smooth the boundary.  State and OM
                # remain jointly scored; only the derived slack-use lane owns
                # its temporary fixed-lag cutoff.
                lane_best = {}
                for kv in ranked:
                    lane = (kv[0][2], kv[0][3])
                    lane_best[lane] = max(
                        lane_best.get(lane, -np.inf), kv[1][0])
                kept = [
                    kv for kv in ranked
                    if kv[1][0] >= lane_best[(kv[0][2], kv[0][3])] - band]
            else:
                kept = [kv for kv in ranked if kv[1][0] >= best - band]
            # The demoted incumbent remains one explicitly scored hypothesis.
            # Preserve every feasible timing of its exact token prefix through
            # both evidence and compute cuts so it is never re-added later in a
            # non-comparable fallback currency.
            req = [kv for kv in ranked if required_prefix(kv)]
            kept_by_key = {kv[0]: kv for kv in kept}
            for kv in req:
                kept_by_key.setdefault(kv[0], kv)
            kept = list(kept_by_key.values())
            if is_optional:
                kept_keys = {kv[0] for kv in kept}
                for action_row in visual_step_audits[k]["actions"].values():
                    action_row["band_retained"] = bool(
                        action_row["expanded_keys"] & kept_keys)
            n_prn += len(ranked) - len(kept)
            if containment_active and truth is not None:
                tset = truth["traj_bytes"]
                rank_b = next((ix for ix, kv in enumerate(ranked)
                               if kv[0][0] in tset), None)
                end_rank_b = next((ix for ix, kv in enumerate(ranked)
                                   if kv[0][0] == truth["end_bytes"]), None)
                _emit_caud(win_ctx, a_span, b_span, "band", k, len(ranked),
                           len(kept), truth,
                           any(kv[0][0] in tset for kv in kept),
                           any(kv[0][0] == truth["end_bytes"] for kv in kept),
                           rank_b,
                           extra=dict(
                               band=float(band), best_score=float(best),
                               truth_end_rank_before=end_rank_b,
                               prefix_om_mode="chronological",
                               slot_kind=slot.kind,
                               visual_interval=(
                                   [int(slot.frame_lo), int(slot.frame_hi)]
                                   if is_optional else None),
                               burst_frame=int(burst_frame),
                               endpoint_evidence=not after_evidence,
                               last_read_chunk=int(last_read_chunk),
                               skip_cap=int(skip_cap),
                               insert_cap=int(insert_cap),
                               streamed_expansion=bool(streamed),
                               streamed_preband_n=streamed_preband_n,
                               truth_word_before=truth_word_fields(
                                   ranked, truth),
                               truth_word_after=truth_word_fields(
                                   kept, truth)))

            # Ordinary windows use fixed K state hypotheses per current OM.
            # A seeded split, or the default-off late-evidence experiment, keeps
            # the complete existing evidence band; synchronized streaming above
            # bounds its expansion.
            strata = {}
            for kv in kept:
                strata.setdefault(kv[0][1], []).append(kv)
            capped_kept = (list(kept) if seed_entries else
                           _cap_stateful_prefix_band(
                               kept, endpoint_chunk=bool(
                                   late_evidence_band_active),
                               beam_k=beam_k))
            capped_by_key = {kv[0]: kv for kv in capped_kept}
            for kv in req:
                capped_by_key.setdefault(kv[0], kv)
            capped_kept = list(capped_by_key.values())
            if is_optional:
                capped_keys = {kv[0] for kv in capped_kept}
                for action_row in visual_step_audits[k]["actions"].values():
                    action_row["retained"] = bool(
                        action_row["expanded_keys"] & capped_keys)
            cut_here = len(capped_kept) < len(kept)
            if cut_here:
                capped = True
                n_prn += len(kept) - len(capped_kept)
            if containment_active and truth is not None and cut_here:
                tset = truth["traj_bytes"]
                _emit_caud(win_ctx, a_span, b_span, "k_cap", k, len(kept),
                           len(capped_kept), truth,
                           any(kv[0][0] in tset for kv in capped_kept),
                           any(kv[0][0] == truth["end_bytes"]
                               for kv in capped_kept), None,
                           extra=dict(beam_k=int(beam_k),
                                      om_strata=len(strata),
                                      prefix_om_mode="chronological",
                                      truth_word_before=truth_word_fields(
                                          kept, truth),
                                      truth_word_after=truth_word_fields(
                                          capped_kept, truth)))
            beam = dict(capped_kept)
            max_live_pairs = max(max_live_pairs, len(beam))
            plane_memory_guard(
                "max_frontier", len(beam), "retained frontier")
            if checkpoint_family is not None:
                checkpoint_family[checkpoint_key(k)] = dict(
                    beam=dict(beam), n_prn=n_prn, capped=capped,
                    max_live_pairs=max_live_pairs,
                    plane_stats=dict(prefix_plane_stats))

        words, word_scores, temporal = {}, {}, {}
        word_slot_merges = {}
        for (_sb, oi, _nskip, _nins, _req_lane,
             _gate_singles), (sc, seq, _mv, timed,
                              slot_merges) in beam.items():
            seq = path_word(seq)
            timed = path_timing(timed)
            end_seen = (not timed or any(
                frame >= int(timed[-1]) for frame in all_read_frames))
            if not end_seen:
                continue
            rec = temporal.setdefault(seq, dict(
                om_scores=np.full(n_om, -np.inf),
                timings={}, timing_sets={}, ambiguous_ois=set(),
                end_pinned=True))
            _record_temporal_frontier_path(rec, oi, sc, timed)
            if sc > word_scores.get(seq, -np.inf):
                word_scores[seq] = sc
                words[seq] = None
                word_slot_merges[seq] = list(slot_merges)
        if containment_active and truth is not None:
            tset = truth["traj_bytes"]
            _emit_caud(win_ctx, a_span, b_span, "beam_final",
                       len(action_slots),
                       len(beam), len(beam), truth,
                       any(kb[0] in tset for kb in beam),
                       truth["end_bytes"] in {kb[0] for kb in beam}, None,
                       extra=dict(
                           prefix_om_mode="chronological",
                           truth_word_after=truth_word_fields(
                               list(beam.items()), truth)))
        visual_audit_rows = []
        for step in sorted(visual_step_audits):
            row = visual_step_audits[step]
            actions = []
            for action in visual_slot_actions():
                action_row = row["actions"][action]
                actions.append(dict(
                    action=action_row["action"],
                    move_index=action_row["move_index"],
                    om_transition_policy=(
                        ("stay-or-one-neighbor"
                         if action is None else "held")
                        if row["kind"] == "phase" else None),
                    best_score=action_row["best_score"],
                    n_expanded_keys=len(action_row["expanded_keys"]),
                    band_retained=bool(action_row["band_retained"]),
                    retained=bool(action_row["retained"])))
            visual_audit_rows.append(dict(
                slot_index=row["slot_index"], kind=row["kind"],
                interval=row["interval"],
                phase_provenance=row.get("phase_provenance"),
                expansion_work=row.get("expansion_work"),
                skip_expansion_work=row.get("skip_expansion_work"),
                shared_state_bound=row.get("shared_state_bound"),
                om_transition_policy=row.get("om_transition_policy"),
                evidence_source=row["evidence_source"],
                fallback_reason=row["fallback_reason"],
                dense_guard=row["dense_guard"], actions=actions))
        stats = dict(kept=0, tie_size=0, overflow=False,
                     beam_limit=(int(BS.SCORE_STATE_BOUND)
                                 if late_evidence_band_active
                                 else int(beam_k) * n_om),
                     prefix_om_mode="chronological",
                     late_evidence_band=late_evidence_band_active,
                     execution_backend=(
                         "cpu-fallback" if device_beam_attempted else "cpu"),
                     execution_fallback=bool(device_beam_attempted),
                     execution_fallback_error=device_beam_fallback,
                     device_checkpoint_resume_step=None,
                     sequential_payload_width=int(
                         STRUCT_CAP if word_capacity is None
                         else word_capacity),
                     max_live_pairs=int(max_live_pairs),
                     gate_drop_single_cap=(
                         None if gate_drop_single_cap is None
                         else int(gate_drop_single_cap)),
                     max_gate_drop_singles=max(
                         (int(key[5]) for key in beam), default=0),
                     seed_word_collision=bool(seed_word_collision),
                     visual_slot_audit=visual_audit_rows,
                     # CUBED_SCRUB_SLOT_MERGE: one row per
                     # SURVIVING candidate word that used >=1 slot-merge (the
                     # best-scoring lineage at that word, mirroring `words`'
                     # own tie-break); [] identically whenever the flag is
                     # off or no merge fired.  Word is the token-index list
                     # (JSON-safe; tuple dict keys are not).
                     slot_merges=[
                         {"word": [int(t) for t in seq], "merges": merges}
                         for seq, merges in word_slot_merges.items()
                         if merges])
        if count_interval is not None:
            stats.update(
                physical_count_interval=list(count_interval),
                count_feasibility_policy=(
                    "reachable-terminal-interval-before-band-k"),
                count_feasibility_steps=int(cpu_count_feasibility_steps),
                count_feasibility_rows_removed=int(
                    cpu_count_feasibility_rows_removed),
                terminal_count_min=cpu_terminal_count_min,
                terminal_count_max=cpu_terminal_count_max)
        return words, temporal, n_prn, capped, stats

    # ---------------------------------------- burst-aligned beam (round 3)
    def _beam_generate(state, a_span, b_span, allowed,
                       truth=None, truth_o=None, win_ctx=None,
                       burst_frames=None, slack=SLACK,
                       exact_read_chunks=False, om_ctx=None,
                       L_lo=None, L_hi=None, required_word=(),
                       seed_entries=None, visual_episodes=(),
                       dropped_intervals=(), phase_slots=(),
                       gate_drop_single_cap=None,
                       record_only_consume_provider_mode=None,
                       burst_frames_are_raw=False):
        """Burst-aligned prefix beam -- candidate
        GENERATOR only; the survivors are re-scored by the exact monotone-DP
        currency in search_window. One move per detected burst; the +/-SLACK
        budget is realized as explicit SKIP (spurious burst) / DOUBLE (missed
        burst) alternatives; prefixes score on the read spans BETWEEN bursts
        (om ball = `allowed`, round-2 continuity), DP-merge by
        (state, slack-use), band-prune at z_sigma*sigma (the commit-gate
        currency) and width-cap at beam_k (the tracker's K convention).
        Returns (words, n_pruned, k_capped)."""
        f_start, f_end = f1_of(a_span), f1_of(b_span)
        src = events if burst_frames is None else burst_frames
        bursts = [f for f in src if f_start < f <= f_end]
        if exact_read_chunks:
            skip_cap, insert_cap = _slot_slack_caps(
                len(bursts), L_lo, L_hi)
        else:
            skip_cap = insert_cap = int(slack)
        transient_word_hi = int(L_hi)
        seed_prefix_hi = max(
            (len(entry["word"]) for entry in (seed_entries or ())),
            default=0)
        total_word_hi = (
            seed_prefix_hi + transient_word_hi
            if seed_entries else transient_word_hi)
        word_capacity = _sequential_payload_width(
            max(int(L_hi), total_word_hi, len(required_word or ())))
        if om_ctx is not None and om_nbrs is not None:
            control_result = _beam_generate_stateful(
                state, a_span, b_span, truth=truth, win_ctx=win_ctx,
                burst_frames=burst_frames, skip_cap=skip_cap,
                insert_cap=insert_cap, om_ctx=om_ctx,
                required_word=required_word, seed_entries=seed_entries,
                visual_episodes=visual_episodes,
                dropped_intervals=dropped_intervals,
                phase_slots=phase_slots,
                gate_drop_single_cap=gate_drop_single_cap,
                word_capacity=word_capacity,
                record_only_consume_provider_mode=(
                    record_only_consume_provider_mode),
                consume_plane_role=None,
                physical_count_interval=None,
                canonical_count_interval=False,
                allow_partial_left_neutral_prefix=False,
                burst_frames_are_raw=burst_frames_are_raw)
            if not dense_prefix_active:
                return control_result

            control_words, control_temporal, n_prn, capped, control_tb = (
                control_result)
            dense_report = report["dense_prefix"]
            optional_prefix_slots = tuple(
                slot for slot in typed_transition_slots(
                    bursts, visual_episodes, f_start, f_end,
                    dropped_intervals=dropped_intervals,
                    intraburst_phase_slots=phase_slots)
                if slot.kind != "motion")
            if seed_entries or optional_prefix_slots:
                if dropped_intervals:
                    counter = None
                    status = "suppressed-typed-rescue-common-currency"
                    reason = (
                        "optional dropped slots require one shared typed "
                        "evidence and OM-action program")
                elif optional_prefix_slots:
                    counter = "optional_slot_suppressions"
                    status = "suppressed-optional-typed-common-currency"
                    reason = (
                        "optional typed slots require action/OM provenance "
                        "that frame tuples alone do not encode")
                else:
                    counter = "seeded_suppressions"
                    status = "suppressed-seeded-prefix-provenance"
                    reason = (
                        "seeded words contain a left prefix whose likelihood "
                        "and timings are outside the right-window program")
                if counter is not None:
                    dense_report[counter] = int(
                        dense_report.get(counter, 0)) + 1
                control_tb["dense_prefix"] = dict(
                    status=status, reason=reason,
                    control_words=len(control_words), dense_words=0,
                    union_words=len(control_words), rescued_words=0)
                return (control_words, control_temporal, n_prn, capped,
                        control_tb)
            dense_report["attempts"] = int(dense_report["attempts"]) + 1
            plane_stats = dict(
                max_frontier=0, max_expansion_cells=0,
                max_expansion_work=0, max_score_cells=0)
            attempt_context = (
                f"window-{win_ctx[0] if win_ctx else 'na'}:"
                f"attempt-{win_ctx[1] if win_ctx else 'na'}:"
                f"spans-{a_span + 1}-{b_span}")

            def update_dense_prefix_maxima():
                for field in ("max_frontier", "max_expansion_cells",
                              "max_score_cells"):
                    dense_report[field] = max(
                        int(dense_report.get(field, 0)),
                        int(plane_stats.get(field, 0)))
                dense_report["max_expansion_work"] = max(
                    int(dense_report.get("max_expansion_work", 0)),
                    int(plane_stats.get("max_expansion_work", 0)))

            def dense_prefix_fallback(reason, error=None):
                update_dense_prefix_maxima()
                dense_report["fallbacks"] = int(
                    dense_report["fallbacks"]) + 1
                failure = {
                    "context": attempt_context,
                    "reason": str(reason),
                    **{key: int(value) for key, value
                       in plane_stats.items()},
                }
                if error is not None:
                    failure.update(
                        error_type=type(error).__name__,
                        error_message=str(error))
                dense_report.setdefault("failures", []).append(failure)
                control_tb["dense_prefix"] = dict(
                    status="control-fallback", reason=str(reason),
                    control_words=len(control_words), dense_words=0,
                    union_words=len(control_words), rescued_words=0,
                    **{key: int(value) for key, value
                       in plane_stats.items()})
                return (control_words, control_temporal, n_prn, capped,
                        control_tb)

            try:
                dense_result = _beam_generate_stateful(
                    state, a_span, b_span,
                    # The dense plane is a production likelihood path, never
                    # a second GT-teacher/containment observer.
                    truth=None, win_ctx=None,
                    burst_frames=burst_frames, skip_cap=skip_cap,
                    insert_cap=insert_cap, om_ctx=om_ctx,
                    required_word=required_word, seed_entries=seed_entries,
                    visual_episodes=visual_episodes,
                    dropped_intervals=dropped_intervals,
                    gate_drop_single_cap=gate_drop_single_cap,
                    word_capacity=word_capacity,
                    prefix_read_frames=score_read_frames,
                    prefix_read_scorer=score_read_seg,
                    prefix_plane_stats=plane_stats,
                    record_only_consume_provider_mode=(
                        record_only_consume_provider_mode),
                    burst_frames_are_raw=burst_frames_are_raw)
            except Exception as exc:  # evidence plane; control is finalized
                return dense_prefix_fallback("scoring-or-bound-error", exc)

            dense_words, dense_temporal, _dense_pruned, _dense_capped, dense_tb = (
                dense_result)
            if dense_tb.get("scaling_tripped"):
                return dense_prefix_fallback(
                    str(dense_tb["scaling_tripped"]))
            if not dense_words:
                return dense_prefix_fallback("empty-dense-prefix-frontier")

            try:
                (union_words, union_temporal,
                 union_stats) = _union_prefix_plane_survivors(
                     control_words, control_temporal,
                     dense_words, dense_temporal)
            except Exception as exc:  # malformed dense provenance: drop plane
                return dense_prefix_fallback("temporal-union-error", exc)

            control_keys = union_stats["control_keys"]
            dense_keys = union_stats["dense_keys"]
            union_keys = union_stats["union_keys"]
            if not control_keys.issubset(union_keys):
                return dense_prefix_fallback("control-subset-invariant")
            rescued = union_stats["rescued_keys"]
            update_dense_prefix_maxima()
            dense_report["successes"] = int(dense_report["successes"]) + 1
            dense_report["control_words"] = int(
                dense_report["control_words"]) + len(control_keys)
            dense_report["dense_words"] = int(
                dense_report["dense_words"]) + len(dense_keys)
            dense_report["union_words"] = int(
                dense_report["union_words"]) + len(union_keys)
            dense_report["rescued_words"] = int(
                dense_report["rescued_words"]) + len(rescued)
            control_tb["dense_prefix"] = dict(
                status="union", control_subset=True,
                control_words=len(control_keys), dense_words=len(dense_keys),
                union_words=len(union_keys), rescued_words=len(rescued),
                shared_words=len(control_keys & dense_keys),
                shared_word_provenance="control",
                shared_word_timing_provenance="control+dense",
                dense_only_word_provenance="dense",
                timing_records=int(union_stats["timing_records"]),
                dense_k_capped=bool(_dense_capped),
                dense_beam_limit=dense_tb.get("beam_limit"),
                execution_backend=dense_tb.get("execution_backend"),
                execution_fallback=bool(
                    dense_tb.get("execution_fallback", False)),
                execution_fallback_error=dense_tb.get(
                    "execution_fallback_error"),
                device_checkpoint_resume_step=dense_tb.get(
                    "device_checkpoint_resume_step"),
                **{key: int(value) for key, value in plane_stats.items()})
            return union_words, union_temporal, n_prn, capped, control_tb
        beam_limit = int(beam_k)
        bounds = [f_start] + bursts + [f_end + 1]
        chunk_spans = [[] for _ in range(len(bursts) + 1)]
        chunk_reads = [[] for _ in range(len(bursts) + 1)]
        for si in range(a_span + 1, b_span + 1):
            if exact_read_chunks:
                n = max(1, len(read_frames[si]))
                for ri, frame in enumerate(read_frames[si]):
                    frame = int(frame)
                    if f_start < frame <= f_end:
                        # A same-frame read is POST-transition, matching the
                        # chronological scorer's rotation->move->read order.
                        k = bisect.bisect_right(bursts, frame)
                        # Preserve the old per-span-mean currency: exact frame
                        # rows from one span sum to weight 1.
                        chunk_reads[k].append((si, ri, 1.0 / n))
            else:
                f0, f1 = view["meta_f"](si)
                for k in range(len(chunk_spans)):
                    # a span belongs to chunk k iff wholly between bursts k, k+1
                    # (spans straddling a burst carry mixed-state reads: skipped)
                    if f0 > bounds[k] and f1 < bounds[k + 1]:
                        chunk_spans[k].append(si)
                        break

        def marg_read(si, ri, smat):
            ois = (range(len(orientations)) if allowed is None
                   else sorted(allowed))
            rows = []
            for oi in ois:
                seg = read_seg(si, ri, oi)
                if seg is not None:
                    rows.append(seg.score_states(smat))
            return np.max(np.stack(rows), axis=0) if rows else None

        def chunk_fit(k, smat):
            if exact_read_chunks:
                rows = [(weight, marg_read(si, ri, smat))
                        for si, ri, weight in chunk_reads[k]]
                rows = [(weight, row) for weight, row in rows
                        if row is not None]
                return (np.sum([weight * row for weight, row in rows], axis=0)
                        if rows else None)
            rows = [marg_row(si, smat, allowed) for si in chunk_spans[k]]
            rows = [r for r in rows if r is not None]
            return np.sum(rows, axis=0) if rows else None

        band = float(z_sigma) * sigma
        st0 = np.asarray(state, np.int8)
        c0 = chunk_fit(0, st0[None, :])
        beam = {(st0.tobytes(), 0, 0):
                ((float(c0[0]) if c0 is not None else 0.0), ())}
        n_prn, capped = 0, False
        tb = {}
        for k in range(1, len(bursts) + 1):
            exp_states, exp_meta = [], []
            for (sb, nskip, nins), (sc, seq) in beam.items():
                s = np.frombuffer(sb, np.int8)
                if nskip < skip_cap:           # spurious-burst alternative
                    exp_states.append(s)
                    exp_meta.append((sc, seq, nskip + 1, nins))
                for mi in range(18):
                    s1 = s[perms[mi]]
                    exp_states.append(s1)
                    exp_meta.append((sc, seq + (mi,), nskip, nins))
                    if nins < insert_cap:       # missed-burst alternative
                        for mj in range(18):
                            if BS.FACE_OF[mj] == BS.FACE_OF[mi]:
                                continue       # same-face double folds away
                            exp_states.append(s1[perms[mj]])
                            exp_meta.append((sc, seq + (mi, mj),
                                             nskip, nins + 1))
            smat = np.stack(exp_states)
            cf = chunk_fit(k, smat)
            nxt = {}
            for i, (sc, seq, nskip, nins) in enumerate(exp_meta):
                v = sc + (float(cf[i]) if cf is not None else 0.0)
                # ORACLE (eval-side; pre-prune seam): -band for a prefix state
                # OFF the GT trajectory, +0 on it. band = float(z_sigma)*sigma
                # (computed above) -- the SAME 2sigma currency this beam prunes
                # in, so an on-truth prefix leads any off-truth rival by exactly
                # one prune band at equal read evidence. Byte-identical off.
                if (oracle_active and truth_o is not None
                        and smat[i].tobytes() not in truth_o["traj_bytes"]):
                    v -= band
                key = (smat[i].tobytes(), nskip, nins)
                cur = nxt.get(key)
                if cur is None or v > cur[0]:
                    nxt[key] = (v, seq)
            ranked = sorted(nxt.items(), key=lambda kv: -kv[1][0])
            best = ranked[0][1][0]
            kept = [kv for kv in ranked if kv[1][0] >= best - band]
            n_prn += len(ranked) - len(kept)
            # CONTAINMENT (shadow): did the GT trajectory survive the band prune?
            # (state-level: any surviving prefix state on the GT trajectory.)
            if containment_active and truth is not None:
                tset = truth["traj_bytes"]
                rank_b = next((ix for ix, kv in enumerate(ranked)
                               if kv[0][0] in tset), None)
                _emit_caud(win_ctx, a_span, b_span, "band", k, len(ranked),
                           len(kept), truth,
                           any(kv[0][0] in tset for kv in kept), None, rank_b,
                           extra=dict(band=float(band), best_score=float(best)))
            pre_cap = len(kept)
            if len(kept) > beam_limit:
                capped = True
                n_prn += len(kept) - beam_limit
                kept = kept[:beam_limit]
                if containment_active and truth is not None:
                    tset = truth["traj_bytes"]
                    _emit_caud(win_ctx, a_span, b_span, "k_cap", k, pre_cap,
                               len(kept), truth,
                               any(kv[0][0] in tset for kv in kept), None, None,
                               extra=dict(beam_k=int(beam_limit)))
            beam = dict(kept)
        words = {}
        for _sc, seq in beam.values():
            words[seq] = None
        if containment_active and truth is not None:
            tset = truth["traj_bytes"]
            _emit_caud(win_ctx, a_span, b_span, "beam_final", len(bursts),
                       len(beam), len(beam), truth,
                       any(kb[0] in tset for kb in beam),
                       truth["end_bytes"] in {kb[0] for kb in beam}, None)
        tb["beam_limit"] = int(beam_limit)
        return words, {}, n_prn, capped, tb

    def _stateful_candidate_scores_once(
            seqs, cols, smat, a_span, b_span, om_ctx,
            endpoint_after_frame=None, candidate_move_frames=None,
            candidate_prefit_om=None, candidate_timing_ambiguous=None,
            _use_dense=None):
        """Score materialized words through one chronological state/OM path.

        When candidate length matches accepted move slots, events and aligned
        reads are processed in frame order and every move records its incoming
        OM.  Slack candidates without a one-to-one event attribution retain a
        conservative span-level OM DP and are marked trace-ambiguous.
        """
        candidate_move_frames = candidate_move_frames or {}
        candidate_prefit_om = candidate_prefit_om or {}
        candidate_timing_ambiguous = set(candidate_timing_ambiguous or ())
        scoring_dense = (bool(dense_evidence) if _use_dense is None else
                         bool(_use_dense))
        scoring_read_frames = (score_read_frames if scoring_dense else
                               read_frames)
        scoring_read_seg = (score_read_seg if scoring_dense else read_seg)
        endpoint_cis = [ci for ci, seq in enumerate(seqs)
                        if tuple(seq) in candidate_timing_ambiguous]
        endpoint_set = set(endpoint_cis)
        # A prefix score is reusable only in its original control currency.
        # Dense final scoring instead evaluates every retained timing tuple on
        # one shared dense grid.  A tagged dense-plane union also requests that
        # fixed-timing program if it ever reaches this routine in non-dense
        # mode; the transaction wrapper normally rolls it back to the literal
        # control records first.
        prefit_records = {
            ci: candidate_prefit_om[tuple(seq)]
            for ci, seq in enumerate(seqs)
            if ci not in endpoint_set
            and candidate_prefit_om.get(tuple(seq)) is not None
        }
        prefit_timing_cis = {
            ci for ci, record in prefit_records.items()
            if (scoring_dense or "dense" in _prefix_plane_sources(record))
        }
        prefit_cis = sorted(set(prefit_records) - prefit_timing_cis)
        direct_cis = endpoint_set | set(prefit_cis)
        path_cis = [ci for ci in range(len(seqs)) if ci not in direct_cis]
        move_frames = sorted(int(f) for f in om_ctx["move_frames"])
        timing_authoritative = bool(om_ctx.get("timing_authoritative", True))
        fallback_cis, chronological_cis = [], []
        for ci in path_cis:
            if ci in prefit_timing_cis:
                chronological_cis.append(ci)
                continue
            timed_frames = candidate_move_frames.get(tuple(seqs[ci]))
            effective_frames = (list(timed_frames)
                                if timed_frames is not None else move_frames)
            target = (chronological_cis
                      if timing_authoritative
                      and len(seqs[ci]) == len(effective_frames)
                      else fallback_cis)
            target.append(ci)
        prefit_timings = {
            ci: _prefit_history_alternatives(
                prefit_records[ci], len(seqs[ci]))
            for ci in sorted(prefit_timing_cis)
        }
        n_prefit_timing_records = sum(map(len, prefit_timings.values()))
        if n_prefit_timing_records > int(BS.SCORE_STATE_BOUND):
            reason = (
                f"final prefit timing records {n_prefit_timing_records} > "
                f"{int(BS.SCORE_STATE_BOUND)}")
            if scoring_dense and any(
                    "dense" in _prefix_plane_sources(record)
                    for record in prefit_records.values()):
                raise _DensePrefixFinalRollback(reason)
            raise RuntimeError(reason)
        score_state_indices = _stateful_score_state_indices(
            len(smat), cols, path_cis, endpoint_cis)
        state_col = {state_idx: compact_idx
                     for compact_idx, state_idx
                     in enumerate(score_state_indices)}
        score_smat = (np.asarray(smat)[score_state_indices]
                      if score_state_indices else None)

        if scoring_dense:
            requested_rows = sum(
                len(scoring_read_frames[si]) + len(mm_by_span.get(si, ()))
                for si in range(a_span + 1, b_span + 1))
            admissible, read_state_cells, bound = dense_grid_guard(
                requested_rows, len(score_state_indices),
                context="final-candidate-grid")
            if not admissible:
                guard = report["dense_evidence"]["scale_guard"]
                guard["control_fallbacks"] = int(
                    guard.get("control_fallbacks", 0)) + 1
                record_dense_control_fallback(
                    "final-candidate-grid:scale-guard")
                if any("dense" in _prefix_plane_sources(record)
                       for record in prefit_records.values()):
                    raise _DensePrefixFinalRollback(
                        "final-candidate-grid:scale-guard")
                return _stateful_candidate_scores_once(
                    seqs, cols, smat, a_span, b_span, om_ctx,
                    endpoint_after_frame=endpoint_after_frame,
                    candidate_move_frames=candidate_move_frames,
                    candidate_prefit_om=candidate_prefit_om,
                    candidate_timing_ambiguous=candidate_timing_ambiguous,
                    _use_dense=False)

        if stateful_cuda_enabled:
            stateful_cuda_stats["fallback_candidates"] += len(fallback_cis)
            stateful_cuda_stats["chronological_candidates"] += len(
                chronological_cis)
            stateful_cuda_stats["endpoint_candidates"] += len(endpoint_cis)
            stateful_cuda_stats["prefit_candidates"] += len(prefit_records)

        read_rows, row_spans, row_frames = [], [], []
        segment_rows, row_weights = [], []
        for si in range(a_span + 1, b_span + 1):
            sources = [(int(frame), "still", ri)
                       for ri, frame in enumerate(scoring_read_frames[si])]
            sources += [(int(row[0]), "midmotion", mi)
                        for mi, row in enumerate(mm_by_span.get(si, []))]
            # Preserve the old per-span mean currency: chronological rows each
            # receive 1/N of a span's evidence, so splitting at a rotation does
            # not make densely sampled spans artificially stronger.
            weight = 1.0 / len(sources) if sources else 1.0
            for frame, source, ri in sorted(sources):
                segments = []
                for oi in range(len(orientations)):
                    seg = (scoring_read_seg(si, ri, oi) if source == "still"
                           else mm_seg_one(si, ri, oi))
                    segments.append(seg)
                if any(seg is not None for seg in segments):
                    if score_smat is not None:
                        segment_rows.append(tuple(segments))
                        row_weights.append(weight)
                    row_spans.append(si)
                    row_frames.append(frame)
        if not row_spans:
            return None
        # Pure-speed fused path: keep the read/orientation/state grid on CUDA so
        # fallback stateful-OM DP can consume it without the old grid-sized D2H.
        # Any incompatibility or device failure leaves ``device_T`` unset and
        # falls through to the exact historical NumPy transaction.
        T = None
        device_T = None
        device_transaction_t0 = None
        if score_smat is not None and stateful_cuda_enabled:
            try:
                device_transaction_t0 = time.perf_counter()
                device_grid = _score_stateful_segment_grid_device(
                    segment_rows, score_smat)
                if device_grid is not None:
                    import torch
                    weights = torch.as_tensor(
                        row_weights, dtype=device_grid.dtype,
                        device=device_grid.device)
                    device_T = device_grid * weights[:, None, None]
                    stateful_cuda_stats["grid_calls"] += 1
                    stateful_cuda_stats["device_grid_bytes"] += (
                        device_T.numel() * device_T.element_size())
            except Exception:
                device_T = None
                stateful_cuda_stats["failures"] += 1
        if score_smat is not None and device_T is None:
            score_grids = _score_stateful_segment_grid(
                segment_rows, score_smat)
            read_rows = [weight * grid for weight, grid
                         in zip(row_weights, score_grids)]
            T = np.stack(read_rows)

        def host_score_grid():
            nonlocal T
            if T is None and device_T is not None:
                T = device_T.detach().cpu().numpy()
                stateful_cuda_stats["grid_d2h_calls"] += 1
                stateful_cuda_stats["grid_d2h_bytes"] += T.nbytes
            return T
        end_pinned = sees_end(row_spans[-1], b_span)
        if endpoint_after_frame is not None:
            # An onset-slotted span may contain several actions and therefore
            # fail the coarse whole-SPAN rest test even though exact-frame reads
            # after its final onset directly witness the endpoint.
            end_pinned = end_pinned or any(
                int(frame) > int(endpoint_after_frame)
                for frame in row_frames)
        end_pinned = end_pinned or bool(
            candidate_prefit_om and not scoring_dense)
        rot_frames = sorted(int(f) for f in om_ctx["rotation_frames"])
        rot_intervals = list(om_ctx.get("rotation_intervals") or ())
        ambiguous_move_positions = {
            i for i, frame in enumerate(move_frames)
            if any(int(interval["lo"]) <= frame <= int(interval["hi"])
                   for interval in rot_intervals)
        }
        start_ois = set(int(i) for i in om_ctx["start_ois"])
        n_om = len(orientations)
        fixed_timing_values = {}
        fixed_timing_transaction = None
        if prefit_timings:
            program_keys = []
            programs = []
            seen_programs = set()
            retained_final_ois_by_program = {}
            constrained_programs = False
            for ci in sorted(prefit_timings):
                for (
                    final_oi,
                    timing,
                    prefix_trace,
                    checkpoint,
                    required_supports,
                    required_covered_supports,
                ) in (
                        prefit_timings[ci]):
                    key = (
                        int(ci),
                        tuple(int(frame) for frame in timing),
                        tuple(int(oi) for oi in prefix_trace),
                        (None if checkpoint is None else
                         (int(checkpoint[0]), int(checkpoint[1]))),
                        tuple(required_supports),
                        tuple(required_covered_supports),
                    )
                    constrained_programs = bool(
                        constrained_programs
                        or key[2]
                        or key[3] is not None
                    )
                    retained_final_ois_by_program.setdefault(
                        key, set()).add(int(final_oi))
                    if key in seen_programs:
                        continue
                    seen_programs.add(key)
                    program_keys.append(key)
                    programs.append((
                        tuple(state_col[int(state_idx)]
                              for state_idx in cols[ci]),
                        key[1],
                    ))
            if constrained_programs:
                raise RuntimeError(
                    "carried endpoint history constraints are unsupported"
                )
            results, fixed_timing_transaction = (
                _score_fixed_timing_programs_transaction(
                    programs,
                    device_T if stateful_cuda_enabled else None,
                    host_score_grid,
                    row_frames,
                    rot_frames,
                    om_nbrs,
                    start_ois,
                ))
            fixed_timing_values = dict(zip(program_keys, results))
            if fixed_timing_transaction[
                    "execution_backend"] in {
                    "cuda-resident"}:
                stateful_cuda_stats["fixed_timing_device_calls"] += 1
                stateful_cuda_stats["fixed_timing_device_programs"] += len(
                    programs)
                stateful_cuda_stats["fixed_timing_device_records"] += int(
                    n_prefit_timing_records)
            elif fixed_timing_transaction["execution_fallback"]:
                stateful_cuda_stats["fixed_timing_device_fallbacks"] += 1
                stateful_cuda_stats["failures"] += 1
        sc = np.full(len(seqs), -np.inf)
        om_sc = np.full((len(seqs), n_om), -np.inf)
        traces = {}
        trace_ambiguities = {}

        def merge_trace(old, candidate):
            return _merge_trace_score(old, candidate)

        fallback = np.full(len(seqs), -np.inf)
        fallback_om = np.full((len(seqs), n_om), -np.inf)
        if fallback_cis:
            fallback_cols = [
                [state_col[int(state_idx)] for state_idx in cols[ci]]
                for ci in fallback_cis
            ]
            packed = None
            if device_T is not None:
                try:
                    fb_t, fb_om_t = _dp_scores_stateful_om_device(
                        fallback_cols, device_T, row_frames,
                        f1_of(a_span), rot_frames, om_nbrs, start_ois,
                        end_pinned)
                    import torch
                    packed = torch.cat(
                        (fb_t[:, None], fb_om_t), dim=1).detach().cpu().numpy()
                    stateful_cuda_stats["fused_dp_calls"] += 1
                    stateful_cuda_stats["fused_result_d2h_bytes"] += (
                        packed.nbytes)
                    if device_transaction_t0 is not None:
                        stateful_cuda_stats["fused_fallback_wall_ms"] += (
                            (time.perf_counter() - device_transaction_t0)
                            * 1000.0)
                except Exception:
                    packed = None
                    stateful_cuda_stats["failures"] += 1
            if packed is None:
                fb, fb_om = _dp_scores_stateful_om(
                    fallback_cols, host_score_grid(), row_frames,
                    f1_of(a_span), rot_frames, om_nbrs, start_ois, end_pinned)
            else:
                fb, fb_om = packed[:, 0], packed[:, 1:]
            fallback[fallback_cis] = fb
            fallback_om[fallback_cis] = fb_om
        endpoint_om = {}
        if endpoint_cis:
            endpoint_ix = [state_col[int(cols[ci][-1])]
                           for ci in endpoint_cis]
            endpoint_mat = _endpoint_scores_stateful_om(
                host_score_grid(), row_frames, f1_of(a_span), rot_frames, om_nbrs,
                start_ois, endpoint_ix)
            endpoint_om = {ci: endpoint_mat[j]
                           for j, ci in enumerate(endpoint_cis)}
        score_modes = []
        for ci, seq in enumerate(seqs):
            if ci in endpoint_om:
                vals = endpoint_om[ci]
                om_sc[ci] = vals
                sc[ci] = float(np.max(vals))
                score_modes.append("timed-chronological")
                for oi in np.where(np.isfinite(vals))[0]:
                    trace_ambiguities[(ci, int(oi))] = frozenset(
                        range(len(seq)))
                continue
            prefit = (candidate_prefit_om.get(tuple(seq))
                      if ci in prefit_cis else None)
            if prefit is not None:
                vals = np.asarray(prefit["om_scores"], float)
                om_sc[ci] = vals
                sc[ci] = float(np.max(vals))
                score_modes.append("timed-chronological")
                # Multiple slot allocations can realize the same physical
                # word under different final OMs.  Preserve the posterior but
                # never fabricate one exact per-move OM trace from a collapsed
                # representative timing.
                for oi in np.where(np.isfinite(vals))[0]:
                    trace_ambiguities[(ci, int(oi))] = frozenset(
                        range(len(seq)))
                continue
            if ci in prefit_timings:
                merged = {}
                for (
                    final_oi,
                    timing,
                    prefix_trace,
                    checkpoint,
                    required_supports,
                    required_covered_supports,
                ) in (
                        prefit_timings[ci]):
                    timed_values = fixed_timing_values[
                        (
                            int(ci),
                            tuple(int(frame) for frame in timing),
                            tuple(int(oi) for oi in prefix_trace),
                            (None if checkpoint is None else
                             (int(checkpoint[0]), int(checkpoint[1]))),
                            tuple(required_supports),
                            tuple(required_covered_supports),
                        )
                    ]
                    if int(final_oi) not in timed_values:
                        continue
                    merged[int(final_oi)] = _merge_timing_score(
                        merged.get(int(final_oi)),
                        timed_values[int(final_oi)], timing)
                score_modes.append("timed-chronological")
                for oi in sorted(merged):
                    score, trace, ambiguity, _timing = merged[oi]
                    om_sc[ci, oi] = float(score)
                    traces[(ci, oi)] = tuple(trace)
                    trace_ambiguities[(ci, oi)] = frozenset(ambiguity)
                if merged:
                    sc[ci] = max(value[0] for value in merged.values())
                continue
            timed_frames = candidate_move_frames.get(tuple(seq))
            if ci in fallback_cis or not timing_authoritative:
                score_modes.append("fallback")
                sc[ci] = fallback[ci]
                om_sc[ci] = fallback_om[ci]
                continue
            move_frames_ci = (list(timed_frames) if timed_frames is not None
                              else move_frames)
            if len(seq) != len(move_frames_ci):
                score_modes.append("fallback")
                sc[ci] = fallback[ci]
                om_sc[ci] = fallback_om[ci]
                continue
            score_modes.append("timed-chronological" if timed_frames is not None
                               else "chronological")
            values = {oi: (0.0, (), frozenset())
                      for oi in sorted(start_ois)}
            move_i = 0
            timeline = ([(f, 0, "rotation", None) for f in rot_frames]
                        + [(f, 1, "move", i)
                           for i, f in enumerate(move_frames_ci)]
                        + [(f, 2, "read", i)
                           for i, f in enumerate(row_frames)])
            for _frame, _order, kind, payload in sorted(timeline):
                if kind == "rotation":
                    nxt = {}
                    for src, (score, trace, ambiguity) in values.items():
                        for dst in {src} | set(om_nbrs[src]):
                            nxt[dst] = merge_trace(
                                nxt.get(dst), (score, trace, ambiguity))
                    values = nxt
                elif kind == "move":
                    values = {oi: (score, trace + (oi,), ambiguity)
                              for oi, (score, trace, ambiguity)
                              in values.items()}
                    move_i += 1
                else:
                    ri = int(payload)
                    state_pos = min(move_i, len(cols[ci]) - 1)
                    state_idx = state_col[int(cols[ci][state_pos])]
                    read_scores = host_score_grid()
                    values = {
                        oi: (score + float(
                            read_scores[ri, oi, state_idx]), trace,
                             ambiguity)
                        for oi, (score, trace, ambiguity) in values.items()
                        if np.isfinite(
                            read_scores[ri, oi, state_idx])
                    }
            if end_pinned and move_i != len(seq):
                continue
            for oi, (score, trace, ambiguity) in values.items():
                om_sc[ci, oi] = score
                traces[(ci, oi)] = trace
                trace_ambiguities[(ci, oi)] = ambiguity
            if values:
                sc[ci] = max(score for score, _trace, _ambiguity
                             in values.values())
        for ci, seq in enumerate(seqs):
            if tuple(seq) not in candidate_timing_ambiguous:
                continue
            for oi in np.where(np.isfinite(om_sc[ci]))[0]:
                trace_ambiguities[(ci, int(oi))] = frozenset(range(len(seq)))
        finite_prefit_cis = [
            ci for ci in prefit_timings if np.isfinite(om_sc[ci]).any()
        ]
        prefit_trace_complete = all(
            all((ci, int(oi)) in traces
                for oi in np.where(np.isfinite(om_sc[ci]))[0])
            for ci in finite_prefit_cis
        )
        prefit_nonfallback = all(
            ci < len(score_modes)
            and score_modes[ci] == "timed-chronological"
            for ci in prefit_timings
        )
        return dict(sc=sc, om_sc=om_sc, traces=traces,
                    trace_ambiguities=trace_ambiguities,
                    score_modes=score_modes,
                    prefit_passthrough_candidates=len(prefit_cis),
                    prefit_rescored_candidates=len(prefit_timings),
                    prefit_timing_records=int(n_prefit_timing_records),
                    prefit_finite_candidates=len(finite_prefit_cis),
                    prefit_trace_complete=bool(prefit_trace_complete),
                    prefit_nonfallback=bool(prefit_nonfallback),
                    fixed_timing_backend=(
                        None if fixed_timing_transaction is None else
                        fixed_timing_transaction["execution_backend"]),
                    fixed_timing_fallback=(
                        None if fixed_timing_transaction is None else
                        fixed_timing_transaction["execution_fallback"]),
                    fixed_timing_fallback_error=(
                        None if fixed_timing_transaction is None else
                        fixed_timing_transaction[
                            "execution_fallback_error"]),
                    fixed_timing_unique_programs=len(fixed_timing_values),
                    ambiguous_move_positions=ambiguous_move_positions,
                    row_spans=row_spans, end_pinned=end_pinned)

    def stateful_candidate_scores(
            seqs, cols, smat, a_span, b_span, om_ctx,
            endpoint_after_frame=None, candidate_move_frames=None,
            candidate_prefit_om=None, candidate_timing_ambiguous=None,
            control_candidate_metadata=None,
            _use_dense=None):
        """Run dense scoring as an optional transaction, then retry control.

        Evidence construction, CPU/device grid scoring, and the downstream OM
        DP are one unit: any ordinary ``Exception`` in that dense-only unit is
        recorded and retried exactly once from finalized control rows.  A
        control-path exception still reaches the orchestrator's stock-decode
        boundary, and ``BaseException`` is never swallowed.
        """
        scoring_dense = (bool(dense_evidence) if _use_dense is None else
                         bool(_use_dense))
        # A dropped or intra-burst typed slot already owns the physical
        # transaction: SKIP may traverse the OM graph, while SINGLE keeps OM
        # fixed.  The older dense whole-window uncertainty DP does not carry
        # that action provenance and could otherwise score an impossible
        # simultaneous layer move + re-grip.  Reuse the exact typed trajectory
        # score here; its interval rows came from the guarded dense plane, while
        # all non-owned rows remain the finalized control plane by construction.
        typed_om_slots = bool(
            om_ctx.get("gate_drop_slots")
            or om_ctx.get("intraburst_phase_slots"))
        if scoring_dense and typed_om_slots:
            gate_report = report.get("gate_drop_slots")
            if gate_report is not None and om_ctx.get("gate_drop_slots"):
                gate_report["dense_final_typed_reuses"] = int(
                    gate_report.get("dense_final_typed_reuses", 0)) + 1
            phase_report = report.get("intraburst_phase_slots")
            if (phase_report is not None
                    and om_ctx.get("intraburst_phase_slots")):
                phase_report["dense_final_typed_reuses"] = int(
                    phase_report.get("dense_final_typed_reuses", 0)) + 1
            record_dense_control_fallback(
                "final-candidate-grid:typed-optional-trajectory")
            typed_result = _stateful_candidate_scores_once(
                seqs, cols, smat, a_span, b_span, om_ctx,
                endpoint_after_frame=endpoint_after_frame,
                candidate_move_frames=candidate_move_frames,
                candidate_prefit_om=candidate_prefit_om,
                candidate_timing_ambiguous=candidate_timing_ambiguous,
                _use_dense=False)
            typed_result["gate_drop_dense_final_typed_reuse"] = bool(
                om_ctx.get("gate_drop_slots"))
            typed_result["intraburst_phase_dense_final_typed_reuse"] = bool(
                om_ctx.get("intraburst_phase_slots"))
            return typed_result
        cuda_before = (dict(stateful_cuda_stats)
                       if scoring_dense else None)
        prefit_records = candidate_prefit_om or {}

        def rollback_dense_prefix_union(exc):
            """Restore the literal control candidate/scoring transaction."""
            stateful_cuda_stats.update(cuda_before)
            dense_report = report.get("dense_prefix")
            if dense_report is not None:
                dense_report["final_rollbacks"] = int(
                    dense_report.get("final_rollbacks", 0)) + 1
            record_dense_control_fallback(
                f"final-candidate-union-rollback:spans-{a_span + 1}-{b_span}",
                exc)
            keep = []
            control_prefit = {}
            control_metadata = dict(control_candidate_metadata or {})
            control_keys = set(control_metadata)
            for ci, seq in enumerate(seqs):
                key = tuple(seq)
                record = prefit_records.get(key)
                sources = (_prefix_plane_sources(record)
                           if record is not None else frozenset())
                if key not in control_keys:
                    continue
                keep.append(ci)
                if record is None:
                    continue
                if sources == {"dense"}:
                    # This word independently belongs to the control candidate
                    # set (incumbent or exact endpoint rescue), but it has no
                    # control temporal prefit.  Score that baseline provenance
                    # normally instead of reusing the dense prefix likelihood.
                    continue
                if "control" in sources:
                    control_record = record.get("_control_prefit")
                    if control_record is None:
                        raise RuntimeError(
                            "shared/control union survivor lacks rollback "
                            "provenance") from exc
                    control_prefit[key] = control_record
                else:
                    control_prefit[key] = record
            if not keep:
                return dict(
                    active_candidate_indices=(),
                    control_candidate_metadata=control_metadata,
                    dense_prefix_final_rollback=True,
                    dense_prefix_final_rollback_empty=True,
                    dense_prefix_final_rollback_error=(
                        f"{type(exc).__name__}: {exc}"))
            kept_keys = {tuple(seqs[ci]) for ci in keep}
            ambiguous_keys = set(candidate_timing_ambiguous or ())
            kept_ambiguous = {
                tuple(seqs[ci]) for ci in keep
                if tuple(seqs[ci]) in ambiguous_keys
            }
            result = _stateful_candidate_scores_once(
                [seqs[ci] for ci in keep], [cols[ci] for ci in keep], smat,
                a_span, b_span, om_ctx,
                endpoint_after_frame=endpoint_after_frame,
                candidate_move_frames={
                    key: value for key, value
                    in (candidate_move_frames or {}).items()
                    if key in kept_keys},
                candidate_prefit_om=control_prefit,
                candidate_timing_ambiguous=kept_ambiguous,
                _use_dense=False)
            result["active_candidate_indices"] = tuple(keep)
            result["control_candidate_metadata"] = control_metadata
            result["dense_prefix_final_rollback"] = True
            result["dense_prefix_final_rollback_error"] = (
                f"{type(exc).__name__}: {exc}")
            return result

        union_active = bool(
            dense_prefix_active
            and _dense_prefix_union_requires_control_rollback(
                seqs, prefit_records, control_candidate_metadata))
        try:
            return _stateful_candidate_scores_once(
                seqs, cols, smat, a_span, b_span, om_ctx,
                endpoint_after_frame=endpoint_after_frame,
                candidate_move_frames=candidate_move_frames,
                candidate_prefit_om=candidate_prefit_om,
                candidate_timing_ambiguous=candidate_timing_ambiguous,
                _use_dense=_use_dense)
        except Exception as exc:  # noqa: BLE001 - evidence transaction boundary
            if not scoring_dense:
                raise
            if union_active:
                return rollback_dense_prefix_union(exc)
            stateful_cuda_stats.update(cuda_before)
            record_dense_control_fallback(
                f"final-candidate-scoring:spans-{a_span + 1}-{b_span}",
                exc)
            return _stateful_candidate_scores_once(
                seqs, cols, smat, a_span, b_span, om_ctx,
                endpoint_after_frame=endpoint_after_frame,
                candidate_move_frames=candidate_move_frames,
                candidate_prefit_om=candidate_prefit_om,
                candidate_timing_ambiguous=candidate_timing_ambiguous,
                _use_dense=False)

    # --------------------------------------------------------- window search
    def _search_window_once(state, a_span, b_span, L_est, allowed,
                            win_ctx=None, cluster_mode=False, om_ctx=None,
                            seed_entries=None,
                            seed_required_word=None,
                            seed_required_suffix_len=None,
                            truth_override=None, dropped_intervals=(),
                            phase_slots=(), gate_drop_single_cap=None,
                            alternate_burst_frames=None,
                            alternate_burst_stream_raw=False,
                            record_only_probe=False,
                            record_only_consume_provider_mode=None):
        """Enumerate + score one window (a_span, b_span]. `allowed` = the
        om-continuity ball for this window (None = full marginalization).
        Exact ball when L_hi <= EXACT_MAX_LHI; burst-aligned beam otherwise
        (generator only -- survivors re-scored in the exact currency).
        Returns a dict with status; on status=='ok' carries the ranked
        structures."""
        if record_only_consume_provider_mode is not None:
            if not record_only_probe:
                raise ValueError(
                    "record-only provider mode requires record_only_probe")
            if (record_only_consume_provider_mode
                    not in _ORPHAN_MOTION_PROVIDER_MODES):
                raise ValueError("record-only provider mode is invalid")
        slack_eff, burst_frames = SLACK, None
        if cluster_mode:
            # Cluster fallback: clustering supplies the alternate COUNT
            # hypothesis, while original observed events remain the only
            # physical action timestamps.  Explicit SKIPs below select which
            # raw events were duplicate/re-grip observations.  A synthetic
            # cluster representative must never place a read before/after a
            # move during generation.
            burst_frames = [f for f in move_events
                            if f1_of(a_span) < f <= f1_of(b_span)]
            n_raw = len(burst_frames)
            n_cluster = _events_in(
                move_events_cl, f1_of(a_span), f1_of(b_span))
            L_est = n_cluster
            slack_eff = max(SLACK, abs(n_raw - L_est))
        L_lo, L_hi = max(0, L_est - slack_eff), L_est + slack_eff
        if alternate_burst_frames is not None:
            # A record-only probe may substitute an alternate burst stream
            # without changing the live count interval.
            if not record_only_probe:
                raise ValueError(
                    "alternate burst frames require a scoped transaction")
            if cluster_mode:
                raise RuntimeError(
                    "alternate burst probe conflicts with another burst source")
            try:
                alternate = tuple(sorted(
                    IBM.require_exact_integer(frame, "alternate burst frame")
                    for frame in alternate_burst_frames))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"malformed alternate burst stream: {exc}")
            if (not alternate or len(alternate) != len(set(alternate))
                    or any(not (f1_of(a_span) < frame <= f1_of(b_span))
                           for frame in alternate)):
                raise ValueError("alternate burst stream is invalid for window")
            burst_frames = list(alternate)
        elif alternate_burst_stream_raw:
            raise ValueError(
                "raw alternate burst identity requires alternate frames")
        raw_window_burst_stream = bool(
            burst_frames is None or alternate_burst_stream_raw)
        slot_source = events if burst_frames is None else burst_frames
        window_slots = typed_transition_slots(
            slot_source, visual_transition_episodes,
            f1_of(a_span), f1_of(b_span),
            dropped_intervals=dropped_intervals,
            intraburst_phase_slots=phase_slots)
        window_visual_slots = tuple(
            slot for slot in window_slots if slot.kind == "visual")
        window_gate_drop_slots = tuple(
            slot for slot in window_slots if slot.kind == "dropped")
        window_phase_slots = tuple(
            slot for slot in window_slots if slot.kind == "phase")
        hard_slot_identity = hard_event_slot_identity(
            slot_source, window_slots, f1_of(a_span), f1_of(b_span))
        hard_slot_digest = hard_event_slot_identity_digest(
            hard_slot_identity)
        has_optional_slots = bool(
            window_visual_slots or window_gate_drop_slots or window_phase_slots)
        ordinary_sequential_search = bool(
            L_hi > EXACT_MAX_LHI
            and not seed_entries
            and not has_optional_slots
            and om_ctx is not None
            and om_nbrs is not None
            # Preserve the outer window-growth contract: retry extensions use
            # STRUCT_CAP as the existing segmentation boundary.  Only an
            # immediate decision-rest window that is itself over-cap (an
            # observed over-cap window) enters the bounded sequential search.  Direct callers
            # without loop context retain ordinary sequential semantics.
            and (win_ctx is None
                 or (isinstance(win_ctx, (tuple, list))
                     and len(win_ctx) > 1 and int(win_ctx[1]) == 0)))
        struct_cap_bypassed = bool(
            L_hi > STRUCT_CAP and ordinary_sequential_search)
        sequential_payload_width = (
            _sequential_payload_width(L_hi)
            if ordinary_sequential_search else None)
        # CUBED_SCRUB_SLOT_MERGE window-level receipt: whether
        # the mechanism is armed and which adjacent ordinary hard slots may
        # share one canonical search token.  Intervening observations are
        # counted in the receipt and remain physically scored; `L_est`/
        # `[L_lo,L_hi]` themselves remain untouched.
        _slot_merge_win_active = _slot_merge_active()
        _slot_merge_win_bursts = sorted(
            int(f) for f in slot_source
            if f1_of(a_span) < int(f) <= f1_of(b_span))
        _slot_merge_win_adjacencies = []
        if (_slot_merge_win_active and raw_window_burst_stream
                and not has_optional_slots and not seed_entries):
            _slot_merge_win_reads = {
                int(frame)
                for stream in (read_frames, score_read_frames)
                for frames_si in stream for frame in frames_si
            }
            _slot_merge_win_adjacencies = _slot_merge_adjacency_audit(
                _slot_merge_win_bursts, _slot_merge_win_reads,
                (om_ctx or {}).get("rotation_frames", ()))
        _slot_merge_win_eligible = [
            row for row in _slot_merge_win_adjacencies if row["eligible"]]
        out = dict(status=None, L_lo=L_lo, L_hi=L_hi, end_pinned=False,
                   slot_merge_enabled=bool(_slot_merge_win_active),
                   slot_merge_adjacencies=_slot_merge_win_adjacencies,
                   slot_merge_eligible_adjacencies=len(
                       _slot_merge_win_eligible),
                   slot_merge_surviving_programs=0,
                   slot_merges=[],
                   n_candidates=None, n_end_states=None, n_word_classes=None,
                   allowed_ois=allowed, beam_engaged=False, n_pruned=0,
                   k_capped=False, cluster_mode=cluster_mode,
                   late_evidence_band=bool(
                       late_evidence_band
                       and (late_evidence_window is None
                            or (win_ctx is not None
                                and int(win_ctx[0]) ==
                                int(late_evidence_window)))),
                   L_est_eff=L_est, slack_eff=slack_eff,
                   hard_event_slot_identity=hard_slot_identity,
                   hard_event_slot_digest=hard_slot_digest,
                   n_visual_transition_slots=len(window_visual_slots),
                   visual_transition_slots=[
                       [int(slot.frame_lo), int(slot.frame_hi)]
                       for slot in window_visual_slots],
                   n_gate_drop_slots=len(window_gate_drop_slots),
                   gate_drop_single_cap=(
                       None if gate_drop_single_cap is None
                       else int(gate_drop_single_cap)),
                   gate_drop_slots=[
                       [int(slot.frame_lo), int(slot.frame_hi)]
                       for slot in window_gate_drop_slots],
                   n_intraburst_phase_slots=len(window_phase_slots),
                   struct_cap_bypassed=struct_cap_bypassed,
                   sequential_payload_width=sequential_payload_width,
                   intraburst_phase_slots=[{
                       "parent_frame": int(slot.parent_frame),
                       "frame_lo": int(slot.frame_lo),
                       "frame_hi": int(slot.frame_hi),
                       "period_lo": int(slot.period_lo),
                       "period_hi": int(slot.period_hi),
                       "phase_index": int(slot.phase_index),
                       "phase_count": int(slot.phase_count),
                   } for slot in window_phase_slots])
        if record_only_probe:
            out["record_only_probe"] = True
            out["alternate_burst_frames"] = [
                int(frame) for frame in (burst_frames or ())]
            if record_only_consume_provider_mode is not None:
                out["record_only_consume_provider_mode"] = (
                    record_only_consume_provider_mode)
        # GT trajectories for this window (from the SCRUB's own window-start
        # state) -- shared by the shadow containment audit + the eval oracle.
        truth = ((truth_override if truth_override is not None else
                  _window_truth(state, a_span, b_span, caud_movegt,
                                caud_prefix))
                 if containment_active else None)
        truth_o = (_window_truth(state, a_span, b_span, orac_movegt,
                                 orac_prefix)
                   if oracle_active else None)
        # the DEMOTED committed hypothesis: always in-set as ONE candidate
        # (never a constraint/anchor/om source).
        comm = (list(BS.tokens_of(seed_required_word))
                if seed_required_word is not None
                else ([] if seed_entries else committed_window(a_span, b_span)))
        out["committed_word"] = list(comm)
        out["committed_nf"] = list(BS.normal_form(comm))
        out["committed_represented"] = False
        comm_budget_len = (seed_required_suffix_len
                           if seed_required_suffix_len is not None else len(comm))
        out["committed_in_budget"] = bool(
            seed_required_word is not None
            and L_lo <= comm_budget_len <= L_hi) if seed_entries else bool(
                L_lo <= len(comm) <= L_hi)
        out["committed_index"] = None
        comm_key = None
        try:
            if seed_required_word is not None:
                # The structural cap applies to the searched RIGHT suffix;
                # the seeded left prefix is already materialized and bounded.
                comm_key = tuple(seed_required_word)
            elif (not seed_entries
                  and (len(comm) <= STRUCT_CAP
                       or (struct_cap_bypassed
                           and len(comm) <= sequential_payload_width))):
                comm_key = tuple(BS.MOVES.index(t) for t in comm)
        except ValueError:
            pass                                   # unparsable token: skip

        def refresh_slot_merge_incumbent_indices(candidate_seqs):
            """Bind literal incumbent comparison to its physical merge rows.

            The canonical committed word remains the public incumbent and its
            literal candidate/index contract is untouched.  When the CUDA
            beam also proves one or more physical programs whose parallel HTM
            spelling equals that word, those rows are the score-comparable
            incumbent realization: unlike the literal fallback, they own the
            same timed chronological currency as their rivals and every NN
            saw their unreduced actions.
            """
            indices = []
            if comm_key is not None:
                physical = {
                    tuple(int(move) for move in row.get("word") or ())
                    for row in out.get("slot_merges") or ()
                    if tuple(int(move) for move in
                             (row.get("search_word") or ())) == comm_key
                }
                indices = [
                    i for i, seq in enumerate(candidate_seqs)
                    if tuple(int(move) for move in seq) in physical
                ]
            out["slot_merge_incumbent_indices"] = indices
            if indices:
                # Physical slots, rather than canonical token count, are the
                # budget currency this mechanism exists to reconcile.
                out["committed_in_budget"] = True
        if L_hi > STRUCT_CAP and not struct_cap_bypassed:
            # Exact/seeded/typed architectures retain the structural cap.
            # Only the bounded chronological state/OM beam is exempt.
            out["status"] = "over-cap"
            return out
        cands = {}
        # Literal dense-prefix-off candidate map. A post-union dense final
        # failure uses this to remove dense-only words.
        control_cands = {}
        candidate_move_frames = {}
        candidate_prefit_om = {}
        candidate_timing_ambiguous = set()
        if (L_hi <= EXACT_MAX_LHI and not seed_entries
                and not has_optional_slots):
            # EXACT canonical ball (<= 343 candidates; round-3 envelope)
            fwd = BS.expand_ball(np.asarray(state, np.int8), perms, L_hi)
            for d in range(L_lo, L_hi + 1):
                for i in range(len(fwd[d]["states"])):
                    cands[BS.seq_of(fwd, d, i)] = None
            control_cands = dict(cands)
            # CONTAINMENT (shadow): the exact ball prunes by the L budget
            # [L_lo, L_hi] -- record whether the GT end state is reachable in it.
            if containment_active and truth is not None:
                ball_states = {st.tobytes() for d in range(L_lo, L_hi + 1)
                               for st in fwd[d]["states"]}
                surv = truth["end_bytes"] in ball_states
                _emit_caud(win_ctx, a_span, b_span, "exact_ball", None,
                           sum(len(fwd[d]["states"])
                               for d in range(L_lo, L_hi + 1)),
                           len(ball_states), truth, surv, surv, None,
                           extra=dict(L_lo=L_lo, L_hi=L_hi,
                                      gt_word_in_budget=bool(
                                          L_lo <= truth["word_len"] <= L_hi)))
        else:
            out["beam_engaged"] = True
            words, temporal_paths, n_prn, capped, tb_stats = _beam_generate(
                state, a_span, b_span, allowed,
                truth=truth, truth_o=truth_o, win_ctx=win_ctx,
                burst_frames=burst_frames, slack=slack_eff,
                # Derive asymmetric SKIP/DOUBLE caps from the count interval so
                # every generated word stays inside it.
                exact_read_chunks=bool(
                    cluster_mode or alternate_burst_frames is not None),
                om_ctx=om_ctx, L_lo=L_lo, L_hi=L_hi,
                required_word=comm_key, seed_entries=seed_entries,
                visual_episodes=visual_transition_episodes,
                dropped_intervals=dropped_intervals,
                phase_slots=phase_slots,
                gate_drop_single_cap=gate_drop_single_cap,
                record_only_consume_provider_mode=(
                    record_only_consume_provider_mode),
                burst_frames_are_raw=raw_window_burst_stream)
            out["n_pruned"], out["k_capped"] = n_prn, capped
            out["beam_limit"] = tb_stats["beam_limit"]
            out["prefix_om_mode"] = tb_stats.get("prefix_om_mode", "marginal")
            out["sequential_payload_width"] = tb_stats.get(
                "sequential_payload_width", out["sequential_payload_width"])
            out["max_live_state_om_pairs"] = tb_stats.get("max_live_pairs")
            out["max_gate_drop_singles"] = int(
                tb_stats.get("max_gate_drop_singles", 0))
            out["seed_word_collision"] = bool(
                tb_stats.get("seed_word_collision", False))
            out["dense_prefix"] = tb_stats.get("dense_prefix")
            out["visual_slot_audit"] = list(
                tb_stats.get("visual_slot_audit") or ())
            out["gate_drop_slot_audit"] = [
                audit for audit in out["visual_slot_audit"]
                if audit.get("kind") == "dropped"]
            out["intraburst_phase_slot_audit"] = [
                audit for audit in out["visual_slot_audit"]
                if audit.get("kind") == "phase"]
            # CUBED_SCRUB_SLOT_MERGE: [] identically whenever
            # the flag is off, this window carried typed slots, a seeded
            # search was active, or no adjacency qualified.
            out["slot_merges"] = list(tb_stats.get("slot_merges") or ())
            out["slot_merge_surviving_programs"] = int(
                tb_stats.get("slot_merge_surviving_programs", 0))
            # A merge-engageable NN window must remain CUDA-resident.  A CPU
            # fallback explicitly reports zero surviving merge programs.  Do
            # not surface new backend fields when the flag is off: the
            # default-off row schema remains byte-identical.
            if _slot_merge_win_active:
                out["execution_backend"] = tb_stats.get("execution_backend")
                out["execution_fallback"] = bool(
                    tb_stats.get("execution_fallback", False))
            if tb_stats.get("scaling_tripped"):
                out["status"] = f"tripped({tb_stats['scaling_tripped']})"
                return out
            for w in words:
                # A chronological survivor owns a physical action path.  Keep
                # it unreduced so its move timestamps and intermediate sticker
                # states remain valid through final scoring.  The legacy beam
                # retains its historical canonical reduction.
                key = (tuple(w) if w in temporal_paths else tuple(
                    BS.MOVES.index(t)
                    for t in BS.reduce_word(BS.tokens_of(w))))
                if key not in cands:
                    cands[key] = None
                    # Cluster representatives certify structure/count, not an
                    # exact timestamp for every physical action.  In
                    # particular a DOUBLE assigns two moves to one synthetic
                    # representative; treating that as authoritative timing
                    # can manufacture a decisive read margin.  Keep the
                    # generated candidate, but let the final stateful scorer
                    # use its uncertainty-aware monotone DP currency.  Raw
                    # Raw motion events retain their exact chronological
                    # prefit.
                    if w in temporal_paths and not cluster_mode:
                        candidate_prefit_om[key] = temporal_paths[w]
                record = temporal_paths.get(w)
                sources = (_prefix_plane_sources(record)
                           if record is not None else frozenset())
                if sources != {"dense"}:
                    control_cands.setdefault(key, None)
            # Optional visual SINGLEs are deliberately outside the hard
            # motion-count interval.  A temporal survivor proves that the
            # incumbent has a valid typed-slot allocation, so it is comparable
            # without widening or reinterpreting [L_lo, L_hi].
            if (has_optional_slots and comm_key is not None
                    and comm_key in temporal_paths):
                out["committed_in_budget"] = True
        # If the committed hypothesis is already a temporal beam survivor, do
        # not erase its comparable score.  Otherwise it remains the one
        # explicit fallback hypothesis in legacy searches.  The exact-history
        # NN union is already the complete count-contained finalist set: an
        # out-of-budget incumbent owns no retained physical timing/provider
        # history and cannot be appended in a different score currency.

        if not cands:
            out["status"] = "empty"
            return out
        if window_gate_drop_slots or window_phase_slots:
            # Every scored word in a typed physical window must retain an exact
            # beam allocation.  In particular, do not let the explicit
            # incumbent fallback bypass SINGLE-vs-SKIP OM semantics when its
            # required path was structurally infeasible.
            missing_paths = [
                tuple(seq) for seq in cands
                if tuple(seq) not in candidate_prefit_om]
            if missing_paths:
                source = ("phase" if window_phase_slots
                          else "gate-drop")
                out["status"] = f"unavailable({source}-physical-path)"
                out[f"{source.replace('-', '_')}_missing_physical_paths"] = [
                    list(seq) for seq in missing_paths]
                gate_report = report.get("gate_drop_slots")
                if gate_report is not None:
                    gate_report["physical_path_rejections"] = int(
                        gate_report.get("physical_path_rejections", 0)) + 1
                return out
        seqs, cols, smat, n_states = BS.materialize_states(
            cands, perms, np.asarray(state, np.int8))
        if seqs is None:
            out["status"] = f"tripped(states {n_states})"
            return out
        # A score margin is meaningful only relative to the hypotheses that
        # were actually scored.  In particular, stateful OM evidence must not
        # turn a short, count-underestimated search into a confident commit
        # when the carried-forward continuation was too long to materialize.
        # This is a containment invariant, not a quality/configuration knob:
        # the incumbent remains demoted to one candidate whenever representable.
        out["committed_represented"] = bool(
            comm_key is not None and comm_key in set(seqs))
        if out["committed_represented"]:
            out["committed_index"] = seqs.index(comm_key)
        refresh_slot_merge_incumbent_indices(seqs)
        endpoint_after = None
        scoring_om_ctx = om_ctx
        if om_ctx is not None:
            scoring_om_ctx = dict(om_ctx)
            if window_gate_drop_slots:
                scoring_om_ctx["gate_drop_slots"] = [
                    {"lo": int(slot.frame_lo), "hi": int(slot.frame_hi)}
                    for slot in window_gate_drop_slots]
            if window_phase_slots:
                scoring_om_ctx["intraburst_phase_slots"] = [
                    {"lo": int(slot.frame_lo), "hi": int(slot.frame_hi),
                     "parent_frame": int(slot.parent_frame)}
                    for slot in window_phase_slots]
        stateful_scored = (stateful_candidate_scores(
            seqs, cols, smat, a_span, b_span, scoring_om_ctx,
            endpoint_after_frame=None,
            candidate_move_frames=candidate_move_frames,
            candidate_prefit_om=candidate_prefit_om,
            candidate_timing_ambiguous=candidate_timing_ambiguous,
            control_candidate_metadata=control_cands,
            _use_dense=None)
            if om_stateful_active and om_ctx is not None else None)
        if (stateful_scored is not None
                and stateful_scored.get("dense_prefix_final_rollback_empty")):
            out["status"] = "empty"
            out["dense_prefix_final_rollback"] = True
            out["dense_prefix_final_rollback_error"] = (
                stateful_scored.get("dense_prefix_final_rollback_error"))
            if out.get("dense_prefix") is not None:
                out["dense_prefix"] = dict(out["dense_prefix"])
                out["dense_prefix"].update(
                    status="final-control-rollback",
                    final_error=out[
                        "dense_prefix_final_rollback_error"])
            return out
        om_scores = move_om_traces = move_om_trace_ambiguities = None
        if stateful_scored is not None:
            active_candidate_indices = stateful_scored.get(
                "active_candidate_indices")
            if active_candidate_indices is not None:
                seqs = [seqs[ci] for ci in active_candidate_indices]
                cols = [cols[ci] for ci in active_candidate_indices]
                for key, metadata in stateful_scored.get(
                        "control_candidate_metadata", {}).items():
                    cands[key] = metadata
                out["committed_represented"] = bool(
                    comm_key is not None and comm_key in set(seqs))
                out["committed_index"] = (
                    seqs.index(comm_key)
                    if out["committed_represented"] else None)
                refresh_slot_merge_incumbent_indices(seqs)
                out["dense_prefix_final_rollback"] = True
                out["dense_prefix_final_rollback_error"] = (
                    stateful_scored.get(
                        "dense_prefix_final_rollback_error"))
                if out.get("dense_prefix") is not None:
                    out["dense_prefix"] = dict(out["dense_prefix"])
                    out["dense_prefix"].update(
                        status="final-control-rollback",
                        final_error=out[
                            "dense_prefix_final_rollback_error"])
            row_spans = stateful_scored["row_spans"]
            desert = False
            end_pinned = stateful_scored["end_pinned"]
            sc = stateful_scored["sc"]
            om_scores = stateful_scored["om_sc"]
            move_om_traces = stateful_scored["traces"]
            move_om_trace_ambiguities = stateful_scored["trace_ambiguities"]
            out["candidate_score_modes"] = stateful_scored["score_modes"]
            out["prefit_rescored_candidates"] = int(
                stateful_scored.get("prefit_rescored_candidates", 0))
            out["prefit_timing_records"] = int(
                stateful_scored.get("prefit_timing_records", 0))
            out["prefit_finite_candidates"] = int(
                stateful_scored.get("prefit_finite_candidates", 0))
            out["prefit_trace_complete"] = bool(
                stateful_scored.get("prefit_trace_complete", True))
            out["prefit_nonfallback"] = bool(
                stateful_scored.get("prefit_nonfallback", True))
            out["fixed_timing_backend"] = stateful_scored.get(
                "fixed_timing_backend")
            out["fixed_timing_fallback"] = stateful_scored.get(
                "fixed_timing_fallback")
            out["fixed_timing_fallback_error"] = stateful_scored.get(
                "fixed_timing_fallback_error")
            out["fixed_timing_unique_programs"] = int(
                stateful_scored.get("fixed_timing_unique_programs", 0))
        else:
            rows, row_spans = [], []
            for si in range(a_span + 1, b_span + 1):
                r = marg_row(si, smat, allowed)
                if r is not None:
                    rows.append(r)
                    row_spans.append(si)
            desert = not rows
        if desert:
            out["status"] = "no-evidence"
            return out
        if stateful_scored is None:
            end_pinned = sees_end(row_spans[-1], b_span)
            if endpoint_after is not None:
                end_pinned = end_pinned or any(
                    int(frame) > int(endpoint_after)
                    for si in row_spans for frame in read_frames[si])
            T = np.stack(rows)
            sc = _dp_scores_end(cols, T, end_pinned)
        sc_read = sc.copy()
        # Report the mid-motion rows that contributed to this window's evidence.
        out["n_midmotion_rows_used"] = sum(
            len(mm_by_span.get(si, ())) for si in range(a_span + 1, b_span + 1))
        # ORACLE (eval-side) at the DECISIVE scoring seam: +0 for a candidate
        # whose NORMAL FORM matches the GT continuation, -X otherwise. X =
        # float(z_sigma)*sigma, the SAME 2sigma band the beam prunes in (band
        # = float(z_sigma)*sigma there) and the commit gate spends (margin_sigma
        # >= z_sigma) -- the oracle is worth EXACTLY one decision band, so it
        # commits the GT word on any window whose read evidence is within the
        # commit band of ambiguous. Deliberately finite (not +/-inf): it
        # measures the band-worth of perfect perception, never overrides strong
        # contrary reads. Byte-identical when oracle_active is False.
        if oracle_active and truth_o is not None:
            _ox = float(z_sigma) * sigma
            gt_nf = tuple(truth_o["word_nf"])
            for i, seq in enumerate(seqs):
                if tuple(BS.normal_form(BS.tokens_of(seq))) != gt_nf:
                    sc[i] = sc[i] - _ox
        # CONTAINMENT (shadow): the assembled + materialized candidate set, the
        # last point before scoring/decision. Records GT end-state membership
        # and GT normal-form-word membership.
        if containment_active and truth is not None:
            end_set = {smat[ix[-1]].tobytes() for ix in cols}
            nf_set = {tuple(BS.normal_form(BS.tokens_of(s))) for s in seqs}
            _emit_caud(win_ctx, a_span, b_span, "final_candidate_set", None,
                       len(seqs), len(seqs), truth,
                       truth["end_bytes"] in end_set,
                       truth["end_bytes"] in end_set, None,
                       extra=dict(
                           gt_word_in_budget=bool(
                               L_lo <= truth["word_len"] <= L_hi),
                           gt_word_nf_in_set=bool(
                               tuple(truth["word_nf"]) in nf_set)))
        by_state = {}                    # end-state grouping (state gate)
        for i, ix in enumerate(cols):
            key = smat[ix[-1]].tobytes()
            oi_row = None
            if om_scores is not None and np.isfinite(om_scores[i]).any():
                oi_row = om_scores[i] + (
                    float(sc[i]) - float(np.max(om_scores[i])))
            g = by_state.get(key)
            if g is None:
                by_state[key] = dict(state=key, best_i=i,
                                     best_score=float(sc[i]),
                                     om_scores=(oi_row.copy()
                                                if oi_row is not None else None))
            elif sc[i] > g["best_score"]:
                old_om = g.get("om_scores")
                g.update(best_i=i, best_score=float(sc[i]))
                if oi_row is not None:
                    g["om_scores"] = (oi_row.copy() if old_om is None
                                      else np.maximum(old_om, oi_row))
            elif oi_row is not None:
                old_om = g.get("om_scores")
                g["om_scores"] = (oi_row.copy() if old_om is None
                                  else np.maximum(old_om, oi_row))
        state_groups = sorted(by_state.values(),
                              key=lambda g: -g["best_score"])
        word_classes = BS.class_ranking(seqs, sc)
        # WINDOW AUDIT (CUBED_SCRUB_WINDOW_AUDIT; schema + contract on
        # _load_window_audit).  RECEIPTS-ONLY microscope at this final
        # scoring seam (the RENDER_DEBUG precedent, window-matched and
        # oracle-extended): dumps every finalist's materialized
        # decomposition, then scores caller-supplied DEV oracle words in the
        # ABI-STAMPED currency -- same BS.materialize_states from the same
        # window root, same span scorer (stateful_candidate_scores with the
        # same om context / marg_row+_dp_scores_end with the same allowed
        # ball and end_pinned). A common finalist rank is emitted only when root/window/
        # slots/timing/OM-program/score-mode all match. The stateful oracle
        # call runs inside an alias-preserving
        # snapshot/restore of the mutable accounting (search_window's
        # restore pattern) so the run's receipts are untouched.  It never
        # alters enumeration, scores, or selection; env absent => branch
        # never runs; any failure logs a typed reason and the run proceeds.
        if window_audit is not None and not record_only_probe:
            _wa_match = False
            try:
                _wa_frames = [int(f1_of(a_span)), int(f1_of(b_span))]
                _wa_idx = (int(win_ctx[0])
                           if isinstance(win_ctx, (tuple, list)) and win_ctx
                           else None)
                _wa_attempt = (int(win_ctx[1])
                               if isinstance(win_ctx, (tuple, list))
                               and len(win_ctx) > 1 else None)
                if window_audit["window_frames"] is not None:
                    _wa_match = _wa_frames in window_audit["window_frames"]
                elif window_audit["window_idx"] is not None:
                    _wa_match = (_wa_idx is not None
                                 and _wa_idx in window_audit["window_idx"])
                else:
                    # window-audit-v2: no selector at all => match every
                    # window (widened from v1's "selector required").
                    _wa_match = True
            except Exception as exc:       # fail-soft: no audit this pass
                window_audit_stats["failsoft"] = (
                    f"match:{type(exc).__name__}: {exc}")
            if _wa_match:
                try:
                    _wa_currency = ("stateful"
                                    if stateful_scored is not None
                                    else "stateless")
                    _wa_T = T if stateful_scored is None else None
                    _wa_root_arr = np.asarray(state, np.int8)
                    _wa_root_key = _bridge_pin_state_key(_wa_root_arr)
                    _wa_root_hash = hashlib.sha256(
                        _wa_root_arr.tobytes()).hexdigest()
                    _wa_allowed = (sorted(int(v) for v in allowed)
                                   if allowed is not None else None)
                    _wa_om_ctx = scoring_om_ctx or {}
                    _wa_om_provenance = dict(
                        prefix_om_mode=out.get("prefix_om_mode"),
                        allowed_ois=_wa_allowed,
                        start_ois=sorted(int(v) for v in
                                         _wa_om_ctx.get("start_ois", ())),
                        move_frames=[int(v) for v in
                                     _wa_om_ctx.get("move_frames", ())],
                        rotation_frames=[int(v) for v in
                                         _wa_om_ctx.get(
                                             "rotation_frames", ())],
                        timing_authoritative=bool(
                            _wa_om_ctx.get("timing_authoritative", True)),
                    )

                    def _wa_action_timing(mode, *, oracle=False):
                        if _wa_currency != "stateful":
                            return f"not-consumed-{_wa_currency}"
                        if oracle:
                            # Oracle input is a token word only.  It does not
                            # carry the finalist's candidate-owned physical
                            # slot/timing record, so no common rank is valid.
                            return "oracle-word-only-no-physical-path"
                        return {
                            "timed-chronological":
                                "candidate-owned-physical-path",
                            "chronological": "window-hard-event-program",
                            "fallback": "unassigned-stateful-dp",
                        }.get(str(mode), "stateful-timing-unavailable")

                    def _wa_candidate_abi(mode, *, oracle=False):
                        if str(mode).startswith("stateless"):
                            om_program = "stateless-om-marginal-dp"
                        else:
                            om_program = "stateful-action-read-om-program"
                        return dict(
                            root_state_hash=_wa_root_hash,
                            window_frames=list(_wa_frames),
                            hard_slot_digest=hard_slot_digest,
                            action_timing_provenance=_wa_action_timing(
                                mode, oracle=oracle),
                            om_program=om_program,
                            om_provenance=dict(_wa_om_provenance),
                            score_mode=str(mode),
                        )

                    def _wa_row(i, seqs_v, cols_v, smat_v, sc_v, read_v,
                                om_v, T_v):
                        toks = list(BS.tokens_of(seqs_v[i]))
                        arow = dict(
                            word=toks, nf=list(BS.normal_form(toks)),
                            score=float(sc_v[i]),
                            score_read=float(read_v[i]),
                            end_state_key=_bridge_pin_state_key(
                                smat_v[cols_v[i][-1]]))
                        om_pick = None
                        if om_v is not None and np.isfinite(om_v[i]).any():
                            vals = np.where(
                                np.isfinite(om_v[i]), om_v[i], -np.inf)
                            om_pick = int(np.argmax(vals))
                            arow["om_scores"] = [
                                [int(oi), float(om_v[i][oi])]
                                for oi in np.where(
                                    np.isfinite(om_v[i]))[0]]
                        arow["om_pick"] = om_pick
                        if T_v is not None and len(row_spans):
                            arow["span_terms"] = [
                                [int(row_spans[j]),
                                 int(f1_of(row_spans[j])),
                                 [float(v) for v in T_v[j, cols_v[i]]]]
                                for j in range(len(row_spans))]
                        return arow

                    if _wa_currency == "stateful":
                        _wa_final_modes = list(
                            out.get("candidate_score_modes") or ())
                        if len(_wa_final_modes) != len(seqs):
                            _wa_final_modes = [
                                "stateful-score-mode-unavailable"] * len(seqs)
                    else:
                        _wa_final_modes = ["stateless-monotone-dp"] * len(seqs)
                    _wa_rows = []
                    _wa_final_abis = []
                    for i in range(len(seqs)):
                        frow = _wa_row(i, seqs, cols, smat, sc, sc_read,
                                       om_scores, _wa_T)
                        _wa_abi = _wa_candidate_abi(_wa_final_modes[i])
                        frow.update(
                            kind="finalist",
                            score_mode=_wa_final_modes[i],
                            candidate_abi=_wa_abi,
                            stage=("beam-survivor"
                                   if out.get("beam_engaged")
                                   else "exact-finalist"))
                        _wa_final_abis.append(_wa_abi)
                        _wa_rows.append(frow)
                    _wa_oracle_ranks = []
                    _wa_oracle_comparable_ranks = []
                    if window_audit["oracle_words"]:
                        _wa_cands = {
                            tuple(BS.MOVES.index(t) for t in word): None
                            for word in window_audit["oracle_words"]}
                        (_wa_seqs, _wa_cols, _wa_smat, _wa_nst) = (
                            BS.materialize_states(
                                _wa_cands, perms,
                                np.asarray(state, np.int8)))
                        if _wa_seqs is None:
                            raise RuntimeError(
                                f"oracle materialization tripped "
                                f"({_wa_nst} states)")
                        _wa_note = None
                        _wa_om_o = None
                        _wa_read_o = None
                        _wa_modes_o = None
                        if stateful_scored is not None:
                            def _wa_restore(target, source):
                                for key in tuple(target):
                                    if key not in source:
                                        del target[key]
                                for key, value in source.items():
                                    if (isinstance(value, dict)
                                            and isinstance(
                                                target.get(key), dict)):
                                        _wa_restore(target[key], value)
                                    else:
                                        target[key] = copy.deepcopy(value)
                            _wa_snap_report = copy.deepcopy(report)
                            _wa_snap_cuda = copy.deepcopy(
                                stateful_cuda_stats)
                            _wa_snap_dense = copy.deepcopy(
                                dense_beam_checkpoint_runtime)
                            try:
                                _wa_st = stateful_candidate_scores(
                                    _wa_seqs, _wa_cols, _wa_smat,
                                    a_span, b_span, scoring_om_ctx,
                                    endpoint_after_frame=endpoint_after)
                                _wa_read_o = np.asarray(
                                    _wa_st["sc"], float)
                                _wa_om_o = _wa_st["om_sc"]
                                _wa_modes_o = list(_wa_st["score_modes"])
                                _wa_note = "stateful-control-no-metadata"
                            except Exception as exc:
                                _wa_note = (f"stateless-fallback:"
                                            f"{type(exc).__name__}")
                                _wa_read_o = None
                            finally:
                                _wa_restore(report, _wa_snap_report)
                                _wa_restore(stateful_cuda_stats,
                                            _wa_snap_cuda)
                                _wa_restore(dense_beam_checkpoint_runtime,
                                            _wa_snap_dense)
                        _wa_T_o = None
                        if _wa_read_o is None:
                            _wa_rows_o = [
                                marg_row(si, _wa_smat, allowed)
                                for si in row_spans]
                            if any(r is None for r in _wa_rows_o):
                                raise RuntimeError(
                                    "oracle span rows unavailable")
                            _wa_T_o = np.stack(_wa_rows_o)
                            _wa_read_o = _dp_scores_end(
                                _wa_cols, _wa_T_o, end_pinned)
                            _wa_modes_o = [
                                ("stateless-fallback"
                                 if stateful_scored is not None else
                                 "stateless-monotone-dp")
                            ] * len(_wa_seqs)
                        _wa_sc_o = np.asarray(_wa_read_o, float)
                        if oracle_active and truth_o is not None:
                            _wa_gt_nf = tuple(truth_o["word_nf"])
                            for k, seq in enumerate(_wa_seqs):
                                if tuple(BS.normal_form(
                                        BS.tokens_of(seq))) != _wa_gt_nf:
                                    _wa_sc_o[k] = _wa_sc_o[k] - (
                                        float(z_sigma) * sigma)
                        _wa_seq_index = {
                            seq: i for i, seq in enumerate(seqs)}
                        for k, seq in enumerate(_wa_seqs):
                            _wa_mode_o = _wa_modes_o[k]
                            orow = _wa_row(
                                k, _wa_seqs, _wa_cols, _wa_smat, _wa_sc_o,
                                _wa_read_o, _wa_om_o, _wa_T_o)
                            twin = _wa_seq_index.get(seq)
                            _wa_abi_o = _wa_candidate_abi(
                                _wa_mode_o,
                                oracle=(_wa_currency == "stateful"))
                            _wa_rank = _window_audit_rank_receipt(
                                float(_wa_sc_o[k]), np.asarray(sc, float),
                                _wa_abi_o, _wa_final_abis)
                            orow.update(
                                kind="oracle",
                                enumerated=twin is not None,
                                score_mode=_wa_mode_o,
                                candidate_abi=_wa_abi_o,
                                currency_note=_wa_note)
                            orow.update(_wa_rank)
                            if twin is not None:
                                orow["enumerated_score"] = float(sc[twin])
                                orow["score_matches_enumerated"] = bool(
                                    float(_wa_sc_o[k]) == float(sc[twin]))
                            _wa_oracle_ranks.append(
                                _wa_rank["rank_among_finalists"])
                            _wa_oracle_comparable_ranks.append(
                                _wa_rank[
                                    "rank_among_comparable_finalists"])
                            _wa_rows.append(orow)
                    _wa_summary = dict(
                        kind="summary", tag=tag, window_idx=_wa_idx,
                        attempt=_wa_attempt, window_frames=_wa_frames,
                        root_state_key=_wa_root_key,
                        root_state_hash=_wa_root_hash,
                        currency=_wa_currency,
                        n_cands_enumerated=len(cands),
                        n_finalists=len(seqs),
                        n_end_states=len(state_groups),
                        n_word_classes=len(word_classes),
                        n_pruned=int(out.get("n_pruned") or 0),
                        k_capped=bool(out.get("k_capped")),
                        beam_engaged=bool(out.get("beam_engaged")),
                        L_lo=int(L_lo), L_hi=int(L_hi),
                        end_pinned=bool(end_pinned),
                        sigma=float(sigma),
                        band=float(z_sigma) * float(sigma),
                        om_allowed=(sorted(int(v) for v in allowed)
                                    if allowed is not None else None),
                        row_spans=[int(si) for si in row_spans],
                        oracle_ranks=list(_wa_oracle_ranks),
                        oracle_comparable_ranks=list(
                            _wa_oracle_comparable_ranks))
                    with open(window_audit["out"], "a") as fh:
                        for _wa_out_row in _wa_rows:
                            fh.write(json.dumps(
                                _wa_out_row, sort_keys=True,
                                default=str) + "\n")
                        fh.write(json.dumps(
                            _wa_summary, sort_keys=True,
                            default=str) + "\n")
                    window_audit_stats["matched"] += 1
                    print(f"  [scrub-window-audit] WROTE "
                          f"{window_audit['out']} "
                          f"(n_finalists={len(seqs)}, "
                          f"oracle_ranks={_wa_oracle_ranks})", flush=True)
                except Exception as exc:   # fail-soft: run untouched
                    window_audit_stats["failsoft"] = (
                        f"dump:{type(exc).__name__}: {exc}")
                    print(f"  [scrub-window-audit] FAILSOFT (run "
                          f"untouched): {type(exc).__name__}: {exc}",
                          flush=True)
        out.update(status="ok", seqs=seqs, cols=cols, smat=smat, sc=sc,
                   state_groups=state_groups, word_classes=word_classes,
                   candidate_om_scores=om_scores,
                   move_om_traces=move_om_traces,
                   move_om_trace_ambiguities=move_om_trace_ambiguities,
                   ambiguous_move_positions=(
                       stateful_scored.get("ambiguous_move_positions", set())
                       if stateful_scored is not None else set()),
                   gate_drop_dense_final_typed_reuse=bool(
                       stateful_scored
                       and stateful_scored.get(
                           "gate_drop_dense_final_typed_reuse")),
                   intraburst_phase_dense_final_typed_reuse=bool(
                       stateful_scored
                       and stateful_scored.get(
                           "intraburst_phase_dense_final_typed_reuse")),
                   end_pinned=end_pinned, evidence_spans=row_spans,
                   n_candidates=len(seqs), n_end_states=len(state_groups),
                   n_word_classes=len(word_classes))
        return out

    def search_window(state, a_span, b_span, L_est, allowed, win_ctx=None,
                      cluster_mode=False, om_ctx=None,
                      seed_entries=None, seed_required_word=None,
                      seed_required_suffix_len=None, truth_override=None,
                      dropped_intervals=(), gate_drop_single_cap=None):
        """Run certified phases as an all-or-nothing window transaction.

        The no-phase search is completed first and retained as the exact
        baseline receipt.  The phase-bearing search starts from the same
        mutable accounting state.  Any exception, non-ok result, missing slot,
        or shared-bound trip restores the baseline result and accounting; no
        mixed frontier can escape.  This deliberately leaves hard events,
        ``L_est``, and ``[L_lo,L_hi]`` under the existing search's sole control.
        """
        nonlocal caud_transaction_buffer, tprobe_transaction_buffer
        owned_phase_slots = tuple(
            slot for slot in intraburst_phase_slots
            if f1_of(a_span) < int(slot.frame_hi) <= f1_of(b_span))
        common = dict(
            win_ctx=win_ctx, cluster_mode=cluster_mode, om_ctx=om_ctx,
            seed_entries=seed_entries,
            seed_required_word=seed_required_word,
            seed_required_suffix_len=seed_required_suffix_len,
            truth_override=truth_override,
            dropped_intervals=dropped_intervals,
            gate_drop_single_cap=gate_drop_single_cap,
        )
        if not owned_phase_slots:
            baseline = _search_window_once(
                state, a_span, b_span, L_est, allowed,
                phase_slots=(), **common)
            return baseline

        def snapshot_mutable_state():
            return dict(
                report=copy.deepcopy(report),
                stateful_cuda=copy.deepcopy(stateful_cuda_stats),
                dense_checkpoint_runtime=copy.deepcopy(
                    dense_beam_checkpoint_runtime),
            )

        def restore_mutable_state(snapshot):
            def restore_dict_in_place(target, source):
                for key in tuple(target):
                    if key not in source:
                        del target[key]
                for key, value in source.items():
                    if isinstance(value, dict) and isinstance(
                            target.get(key), dict):
                        restore_dict_in_place(target[key], value)
                    else:
                        target[key] = copy.deepcopy(value)

            # Preserve aliases held by gate-rescue callers while restoring
            # their contents transactionally.
            restore_dict_in_place(report, snapshot["report"])
            stateful_cuda_stats.clear()
            stateful_cuda_stats.update(copy.deepcopy(
                snapshot["stateful_cuda"]))
            dense_beam_checkpoint_runtime.clear()
            dense_beam_checkpoint_runtime.update(copy.deepcopy(
                snapshot["dense_checkpoint_runtime"]))

        def flush_caud(rows):
            nonlocal caud_transaction_buffer
            caud_transaction_buffer = None
            for row in rows:
                caud_row(row)

        before = snapshot_mutable_state()
        caud_transaction_buffer = []
        try:
            baseline = _search_window_once(
                state, a_span, b_span, L_est, allowed,
                phase_slots=(), **common)
            baseline_rows = list(caud_transaction_buffer)
            baseline_state = snapshot_mutable_state()
        except Exception:
            caud_transaction_buffer = None
            restore_mutable_state(before)
            raise

        restore_mutable_state(before)
        caud_transaction_buffer = []
        phase_error = None
        comparisons = None
        try:
            phased = _search_window_once(
                state, a_span, b_span, L_est, allowed,
                phase_slots=owned_phase_slots, **common)
            phased_rows = list(caud_transaction_buffer)
            comparisons = phase_structure_comparisons(baseline, phased)
            materialized = int(phased.get(
                "n_intraburst_phase_slots", 0))
            phase_status = str(phased.get("status"))
            if phase_status != "ok":
                phase_error = f"phase search status {phase_status}"
            elif materialized != len(owned_phase_slots):
                phase_error = (
                    f"materialized phase slots {materialized} != certified "
                    f"{len(owned_phase_slots)}")
            elif not comparisons["all_match"]:
                mismatches = sorted(
                    field for field, row in comparisons.items()
                    if field != "all_match" and not row["match"])
                phase_error = (
                    "phase structural contract mismatch: "
                    + ", ".join(mismatches))
        except Exception as exc:  # feature-local transaction boundary
            phased = None
            phased_rows = list(caud_transaction_buffer)
            phase_error = f"{type(exc).__name__}: {exc}"
            comparisons = phase_structure_comparisons(baseline, {})

        hard_identity_match = bool(
            comparisons["hard_event_slot_digest"]["match"])
        count_interval_match = all(
            comparisons[field]["match"]
            for field in ("L_est_eff", "L_lo", "L_hi"))

        if phase_error is not None:
            restore_mutable_state(baseline_state)
            receipt = report["intraburst_phase_slots"]
            receipt["window_attempts"] = int(
                receipt.get("window_attempts", 0)) + 1
            receipt["window_rollbacks"] = int(
                receipt.get("window_rollbacks", 0)) + 1
            receipt.setdefault("rollback_reasons", []).append({
                "frames": [int(f1_of(a_span)), int(f1_of(b_span))],
                "reason": str(phase_error),
                "comparisons": copy.deepcopy(comparisons),
            })
            baseline = dict(baseline)
            baseline["intraburst_phase_transaction"] = {
                "status": "rolled-back-to-baseline",
                "reason": str(phase_error),
                "certified_slot_count": len(owned_phase_slots),
                "hard_events_unchanged": hard_identity_match,
                "count_interval_unchanged": count_interval_match,
                "comparisons": copy.deepcopy(comparisons),
            }
            flush_caud(baseline_rows)
            return baseline

        receipt = report["intraburst_phase_slots"]
        receipt["window_attempts"] = int(
            receipt.get("window_attempts", 0)) + 1
        receipt["window_adoptions"] = int(
            receipt.get("window_adoptions", 0)) + 1
        phased["intraburst_phase_transaction"] = {
            "status": "adopted",
            "certified_slot_count": len(owned_phase_slots),
            "hard_events_unchanged": hard_identity_match,
            "count_interval_unchanged": count_interval_match,
            "comparisons": copy.deepcopy(comparisons),
            "baseline_status": baseline.get("status"),
        }
        flush_caud(phased_rows)
        return phased

    # Stream each completed window so a partial sidecar remains useful if a
    # later window fails.
    sc_fh = None
    if sidecar_path:
        try:
            sc_fh = open(sidecar_path, "w")
            report["sidecar"] = sidecar_path
        except OSError as exc:
            report["sidecar_error"] = f"{type(exc).__name__}: {exc}"

    def stream_row(obj):
        nonlocal sc_fh
        if sc_fh is None:
            return
        try:
            sc_fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
            sc_fh.flush()
        except OSError as e:                       # IO never aborts the scrub
            report["sidecar_error"] = f"{type(e).__name__}: {e}"
            try:
                sc_fh.close()
            except OSError:
                pass
            sc_fh = None

    header = dict(
        kind="header", tag=tag,
        init_provenance="app-session start state (view['init'])",
        final_provenance=("--final-from-gt terminal (final_arr)"
                          if final_arr is not None else
                          "none (final_arr=None)"),
        depth_cap=DEPTH_CAP, struct_cap=STRUCT_CAP,
        struct_cap_scope=(
            "exact/MITM, seeded/typed, and extension-retry searches; ordinary "
            "immediate decision-rest state/OM beam may "
            "bypass with L_hi-sized payload and SCORE_STATE_BOUND work guard"),
        slack=SLACK,
        slack_provenance=(
            "+/-1 permits a move at either side of a window boundary"
        ),
        z_sigma=float(z_sigma), sigma_fit_population=sigma,
        commit_gate=("state_margin >= z_sigma*sigma AND margins > 0 AND "
                     "word class decisive"),
        om_continuity=bool(om_nbrs is not None),
        om_stateful=om_stateful_active,
        om_stateful_reason=om_stateful_reason,
        exact_max_lhi=EXACT_MAX_LHI, beam_k=int(beam_k),
        late_evidence_band=bool(late_evidence_band),
        late_evidence_window=late_evidence_window,
        late_evidence_band_provenance=(
            "default OFF; optional evaluation window scope; retain the "
            "unchanged 2sigma prefix band and defer only per-OM K; original "
            "scores and BS.SCORE_STATE_BOUND preserved"),
        midmotion_reads=bool(mm_by_span),
        midmotion_rows_used=report.get("midmotion_rows_used", 0),
        midmotion_rows_dropped=report.get("midmotion_rows_dropped", 0),
        mm_conf_floor=MM_CONF_FLOOR,
        microrest_candidacy=bool(microrest),
        enumeration=("exact ball <= EXACT_MAX_LHI; else burst-aligned "
                     "prefix beam (band 2sigma + K cap), generator only; "
                     "stateful path carries chronological (state,OM) pairs "
                     "with fixed K states per one of 24 OMs"),
        prefix_om_provenance=("stateful beam accumulates exact-frame reads "
                              "under one reachable OM history; OM stay/neighbor "
                              "transitions occur only at authoritative dropped "
                              "rotation clusters; per-read OM teleportation is "
                              "forbidden before both band and K pruning"),
        certified_rest_spans=sorted(certified_rest_spans),
    )
    if bridge_pin is not None:
        # Immutable armed-pin banner; live engagement counters land in
        # report/summary ``bridge_pin`` (env unset => key absent everywhere
        # => byte-identical).
        header["bridge_pin"] = dict(bridge_pin_stats["pin"])
    if dense_evidence:
        # Header fields are immutable run inputs only.  The live guard counters
        # change while windows execute and are emitted in summary/error rows.
        dense_header = dict(report["dense_evidence"])
        dense_header["scale_guard"] = {
            "read_state_bound": int(BS.SCORE_STATE_BOUND),
        }
        header["dense_evidence"] = dense_header
    if dense_prefix_active:
        dense_prefix_header = dict(report["dense_prefix"])
        for dynamic_key in (
                "attempts", "successes", "fallbacks", "failures",
                "control_words", "dense_words", "union_words",
                "rescued_words", "typed_rescue_suppressions",
                "optional_slot_suppressions", "seeded_suppressions",
                "final_rollbacks",
                "max_frontier", "max_expansion_cells",
                "max_expansion_work", "max_score_cells"):
            dense_prefix_header.pop(dynamic_key, None)
        header["dense_prefix"] = dense_prefix_header
    if visual_transition_episodes:
        header["visual_transition_slots"] = report[
            "visual_transition_slots"]
    if gate_drop_slots_active:
        gate_header = dict(report["gate_drop_slots"])
        for dynamic_key in (
                "slot_evaluations", "max_step_expansions",
                "bound_fallbacks", "score_error_fallbacks", "score_errors",
                "dense_final_typed_reuses", "physical_path_rejections",
                "rescue_attempts", "rescue_adoptions",
                "rescue_nondecisive", "rescue_failures", "rescue_wall_ms",
                "tier1_attempts", "tier1_adoptions",
                "tier1_nondecisive", "tier1_failures", "tier1_wall_ms",
                "tier2_attempts", "tier2_adoptions",
                "tier2_nondecisive", "tier2_failures", "tier2_wall_ms"):
            gate_header.pop(dynamic_key, None)
        for dynamic_key in (
                "tier_unions", "tier_union_failures",
                "tier_union_candidates", "tier_union_errors"):
            gate_header.pop(dynamic_key, None)
        header["gate_drop_slots"] = gate_header
    phase_header = copy.deepcopy(report["intraburst_phase_slots"])
    for dynamic_key in (
            "window_attempts", "window_adoptions", "window_rollbacks",
            "rollback_reasons", "dense_final_typed_reuses",
            "slot_evaluations", "max_step_expansions",
            "max_skip_expansions"):
        phase_header.pop(dynamic_key, None)
    header["intraburst_phase_slots"] = phase_header
    stream_row(header)

    # ------------------------------------------------------------ main loop
    state = np.asarray(init_arr, np.int8).copy()
    cursor = -1                        # span index of the last verified rest
    last_om = None
    carried_oi = None                  # om PRIOR (None: window 0, unknown om)
    carried_ois = None                 # stateful ambiguity band
    carried_extra = 0                  # unread om drift carried forward
    out_windows = []                   # committed dicts: tokens, land, om
    ll_onset = None                    # first state-unique F2L -> LL crossing
    # Independent verifier fence: unlike LL completion, excluding learned
    # evidence after LL does not require stateful-OM machinery to be active.
    # It is evaluated only in active verifier mode, leaving OFF untouched.
    # V18 action-consume fence (parallel to and INDEPENDENT of nn_verif's):
    # each runtime keys its own flag on the SAME F2L->LL crossing, so the two
    # channels compose when enabled together -- the consume seam reads
    # trajectory_ll_started to score THROUGH the crossing window and stay
    # exactly zero after (risk #10).
    counts = dict(committed=0, extended=0, low_conf=0, unresolved=0)
    beam_stats = dict(searches=0, n_pruned=0)
    cluster_fallback_stats = dict(attempts=0, wall_ms=0)
    w_idx = 0

    if om_stateful_active:
        first_event = min(timeline_events) if timeline_events else float("inf")
        opening = [si for si in range(scan_hi + 1)
                   if has_reads(si) and f1_of(si) < first_event]
        opening_span = (max(opening, key=coverage) if opening else None)
        ranked0 = rank_oms_many(opening, state, None) if opening else []
        if ranked0:
            top0 = ranked0[0][1]
            carried_ois = {oi for oi, score in ranked0
                           if top0 - score <= float(z_sigma) * sigma}
            carried_oi = ranked0[0][0]
            report["om_stateful_opening"] = dict(
                anchored=True, span=int(opening_span), spans=list(opening),
                band=sorted(carried_ois), top=int(carried_oi))
        else:
            carried_ois = set(range(len(orientations)))
            carried_oi = 0
            report["om_stateful_opening"] = dict(
                anchored=False, span=None, band=sorted(carried_ois), top=None)

    def emit_line(**kw):
        kw.update(tag=tag, kind="window")
        stream_row(kw)

    def base_row(attempt, b_rep, b_last, L_est, res, radius, allowed):
        n_rot = _events_in(rotation_events, f1_of(cursor), f1_of(b_last))
        transition = ("mixed" if L_est and n_rot else
                      ("move-only" if L_est else
                       ("rotation-only" if n_rot else "stay")))
        dense_prefix_row = res.get("dense_prefix")
        if ((dense_prefix_row or {}).get("status")
                == "suppressed-typed-rescue-common-currency"):
            # A typed dropped-slot transaction did not run a second prefix
            # plane: it deliberately kept one physical action/OM program.
            # The run-level suppression counter remains the audit record;
            # presenting the suppression metadata as a scored prefix result
            # would incorrectly imply that two planes reached this window.
            dense_prefix_row = None
        row = dict(idx=w_idx, attempt=attempt, span_a=cursor,
                   span_b=b_rep, span_b_last=b_last,
                   frames=[f1_of(cursor), f1_of(b_rep)], L_est=L_est,
                    L_est_eff=res.get("L_est_eff"),
                    L_lo=res.get("L_lo"), L_hi=res.get("L_hi"),
                    hard_event_slot_identity=res.get(
                        "hard_event_slot_identity"),
                    hard_event_slot_digest=res.get(
                        "hard_event_slot_digest"),
                    status=res["status"], end_pinned=res.get("end_pinned"),
                    n_candidates=res.get("n_candidates"),
                    n_end_states=res.get("n_end_states"),
                    n_word_classes=res.get("n_word_classes"),
                    beam_engaged=res.get("beam_engaged", False),
                    beam_limit=res.get("beam_limit", int(beam_k)),
                    struct_cap_bypassed=res.get(
                        "struct_cap_bypassed", False),
                    sequential_payload_width=res.get(
                        "sequential_payload_width"),
                    prefix_om_mode=res.get("prefix_om_mode"),
                    max_live_state_om_pairs=res.get(
                        "max_live_state_om_pairs"),
                    dense_prefix=dense_prefix_row,
                    dense_prefix_final_rollback=res.get(
                        "dense_prefix_final_rollback", False),
                    dense_prefix_final_rollback_error=res.get(
                        "dense_prefix_final_rollback_error"),
                    prefit_rescored_candidates=res.get(
                        "prefit_rescored_candidates", 0),
                    prefit_timing_records=res.get(
                        "prefit_timing_records", 0),
                    prefit_finite_candidates=res.get(
                        "prefit_finite_candidates", 0),
                    prefit_trace_complete=res.get(
                        "prefit_trace_complete", True),
                    prefit_nonfallback=res.get(
                        "prefit_nonfallback", True),
                    fixed_timing_backend=res.get("fixed_timing_backend"),
                    fixed_timing_fallback=res.get("fixed_timing_fallback"),
                    fixed_timing_fallback_error=res.get(
                        "fixed_timing_fallback_error"),
                    fixed_timing_unique_programs=res.get(
                        "fixed_timing_unique_programs", 0),
                    exact_endpoint_rescue_depth=res.get(
                        "exact_endpoint_rescue_depth"),
                    exact_endpoint_rescue_added=res.get(
                        "exact_endpoint_rescue_added"),
                    late_evidence_band=res.get(
                        "late_evidence_band", False),
                    n_pruned=res.get("n_pruned", 0),
                    k_capped=res.get("k_capped", False),
                    om_prior=(list(om_key_of(orientations[carried_oi]))
                              if carried_oi is not None else None),
                    om_radius=radius,
                    om_transition=transition,
                    om_prior_indices=(sorted(carried_ois)
                                      if om_stateful_active else None),
                    rotation_clusters=n_rot,
                    n_om_allowed=(len(allowed) if allowed is not None
                                  else len(orientations)),
                    n_midmotion_rows_used=res.get("n_midmotion_rows_used", 0),
                    n_visual_transition_slots=res.get(
                        "n_visual_transition_slots", 0),
                    visual_transition_slots=res.get(
                        "visual_transition_slots", []),
                    visual_slot_audit=res.get("visual_slot_audit", []),
                    slot_merge_enabled=res.get("slot_merge_enabled", False),
                    slot_merge_adjacencies=res.get(
                        "slot_merge_adjacencies", []),
                    slot_merge_eligible_adjacencies=res.get(
                        "slot_merge_eligible_adjacencies", 0),
                    slot_merge_surviving_programs=res.get(
                        "slot_merge_surviving_programs", 0),
                    slot_merges=res.get("slot_merges", []),
                    n_gate_drop_slots=res.get("n_gate_drop_slots", 0),
                    gate_drop_single_cap=res.get("gate_drop_single_cap"),
                    max_gate_drop_singles=res.get(
                        "max_gate_drop_singles", 0),
                    gate_drop_slots=res.get("gate_drop_slots", []),
                    gate_drop_slot_audit=res.get(
                        "gate_drop_slot_audit", []),
                    gate_drop_dense_final_typed_reuse=res.get(
                        "gate_drop_dense_final_typed_reuse", False),
                    gate_drop_missing_physical_paths=res.get(
                        "gate_drop_missing_physical_paths", []),
                    n_intraburst_phase_slots=res.get(
                        "n_intraburst_phase_slots", 0),
                    intraburst_phase_slots=res.get(
                        "intraburst_phase_slots", []),
                    intraburst_phase_slot_audit=res.get(
                        "intraburst_phase_slot_audit", []),
                    intraburst_phase_transaction=res.get(
                        "intraburst_phase_transaction"),
                    intraburst_phase_dense_final_typed_reuse=res.get(
                        "intraburst_phase_dense_final_typed_reuse", False),
                    committed_represented=res.get(
                        "committed_represented", False),
                    committed_in_budget=res.get("committed_in_budget", False),
                    committed_comparable=res.get(
                        "committed_comparable", False),
                   committed_word=res.get(
                       "committed_word",
                       list(committed_window(cursor, b_last))))
        if bridge_pin is not None:
            # Armed-pin runs receipt every window's search-root identity so
            # a state-carry question is adjudicable from the
            # sidecar alone: the root chain is the loop's ``state`` variable
            # -- pin["state"] verbatim after engagement -- never a replay of
            # pin["word"].  Env unset => field absent => byte-identical.
            row["root_state_key"] = _bridge_pin_state_key(state)
        if res.get("slot_merge_enabled", False):
            row.update(
                execution_backend=res.get("execution_backend"),
                execution_fallback=res.get("execution_fallback", False))
        if res.get("unowned_motion_action_transaction") is not None:
            row["unowned_motion_action_transaction"] = copy.deepcopy(
                res["unowned_motion_action_transaction"])
            row["nn_terminal_survivor_union"] = copy.deepcopy(
                res.get("nn_terminal_survivor_union"))
            row["nn_common_prefix_rescore"] = copy.deepcopy(
                res.get("nn_common_prefix_rescore"))
        return row

    def dec_fields(dec):
        fields = dict(decision=dec["decision"],
                    state_margin=(dec["state_margin"]
                                  if np.isfinite(dec["state_margin"])
                                  else None),
                    margin_sigma=(dec["margin_sigma"]
                                  if np.isfinite(dec["margin_sigma"])
                                  else None),
                    word_margin=(dec["word_margin"]
                                 if np.isfinite(dec["word_margin"])
                                 else None),
                    state_unique=dec["state_unique"],
                    decisive=dec["word_decisive"])
        return fields



    def shadow_window_decision(result, at_cap, *, boundary_span,
                               site_kind, attempt):
        return window_decision(
            result["state_groups"], result["word_classes"], at_cap,
            sigma, z_sigma=z_sigma)

    def enforce_stateful_containment(dec, res, at_cap):
        """Do not spend an OM margin against an incomplete hypothesis set."""
        if not om_stateful_active:
            return dec, None
        modes = res.get("candidate_score_modes") or ()
        comm_i = res.get("committed_index")
        winner_i = (res["state_groups"][0]["best_i"]
                    if res.get("state_groups") else None)
        # Advancing on the incumbent itself is always safe: no hypothesis is
        # being replaced, so budget/mode comparability is irrelevant.  This
        # prevents a conservative containment guard from ballooning the whole
        # solve into one fallback window when motion counts under-fire.
        merge_committed = {
            int(index) for index in
            (res.get("slot_merge_incumbent_indices") or ())
            if 0 <= int(index) < len(res.get("seqs") or ())
        }
        if ((comm_i is not None and winner_i == comm_i)
                or winner_i in merge_committed):
            res["committed_comparable"] = True
            res["incumbent_selected"] = True
            return dec, None
        comparison_i = comm_i
        if merge_committed:
            scores = np.asarray(res.get("sc"), float)
            finite = [index for index in merge_committed
                      if index < len(scores) and np.isfinite(scores[index])]
            if finite:
                comparison_i = max(finite, key=lambda index: scores[index])
        reason = None
        if not res.get("committed_represented", False):
            reason = "incumbent-absent"
        elif not res.get("committed_in_budget", False):
            reason = "incumbent-out-of-budget"
        else:
            if (comparison_i is None or winner_i is None
                    or comparison_i >= len(modes) or winner_i >= len(modes)
                    or modes[comparison_i] != modes[winner_i]):
                reason = "incumbent-score-mode-mismatch"
        res["committed_comparable"] = reason is None
        if reason is None:
            return dec, None
        guarded = dict(dec)
        guarded["decision"] = "unresolved" if at_cap else "extend"
        return guarded, reason

    while cursor < scan_hi:
        ahead = [g for g in groups if g["rep"] > cursor]
        targets = [(g["rep"], g["last"]) for g in ahead]
        if not targets or targets[-1][1] < last_solve_span:
            targets.append((scan_hi, scan_hi))                     # tail
        # nothing left to decode past the last committed rest: done.
        if (targets[0][0] >= last_solve_span
                and _events_in(move_events, f1_of(cursor), f1_of(scan_hi)) == 0
                and not committed_window(cursor, scan_hi)):
            break
        # BRIDGE PIN forced boundary (armed + not-yet-engaged path ONLY; env
        # absent => bridge_pin is None => this block is inert and the stock
        # path is byte-identical).  In one observed case:
        # the engaged window overran the certified rest ([5180,5728] vs gap
        # [5534,5555]) and the splice subsumed the two GT moves inside
        # (5555,5728], so the re-root resumed two moves stale.  Fix: while a
        # window under construction reaches/crosses gap_a, its target list
        # is CAPPED at the first rest whose end frame reaches the gap -- the
        # certified rest itself whenever it is a lattice rest -- so window
        # extension TERMINATES at the pin boundary regardless of commit-gate
        # confidence.  Rationale: the certified rest supersedes the commit
        # gate's confidence requirement at that one boundary, because the
        # pin replaces the commit decision there anyway (engagement fires at
        # the application seam); moves after gap_b stay in SUBSEQUENT
        # windows, decoded from B.  Committed-state boundary unchanged: the pin state
        # remains EXTERNAL certification, never committed-trajectory.  The
        # f1_of(cursor) < gap_b guard keeps a stale/behind pin (one that can
        # no longer satisfy the engagement predicate) from capping windows
        # forever.  ``forced_boundary`` receipt = this cap removed >= 1
        # later target for the engaged window.
        bridge_pin_forced_boundary = False
        bridge_pin_synthetic_rest = False
        if (bridge_pin is not None and not bridge_pin_stats["engaged"]
                and int(f1_of(cursor)) < int(bridge_pin["gap"][1])):
            try:
                _bpf_a = int(bridge_pin["gap"][0])
                _bpf_b = int(bridge_pin["gap"][1])
                _bpf_cap_i = None
                for _bpf_i, (_bpf_rep, _bpf_last) in enumerate(targets):
                    if int(f1_of(_bpf_last)) >= _bpf_a:
                        _bpf_cap_i = _bpf_i
                        break
                if _bpf_cap_i is not None:
                    # SYNTHETIC REST: in the same observed case the
                    # certified rest [5534,5555] was ABSENT from the scrub's rest
                    # lattice (candidacy never surfaced it through the
                    # desert), so capping at existing targets kept ending the
                    # window at the next real rest (5728) and re-subsumed the
                    # post-gap moves.  The pin gap is EXTERNALLY certified as
                    # a rest (chain_cert report; n_ok usable frames), so
                    # treating it as a rest-group boundary imports no new
                    # inference: synthesize a (rep, last) target over the
                    # lattice spans covering the gap and splice it in frame
                    # order before the overshooting rest.  Dedupe is by SPAN
                    # OVERLAP, not equality: if the existing cap target
                    # already ends at/inside the gap-b span (last <= s_b) the
                    # real rest stands and nothing is inserted.  rep = the
                    # best-covered read-bearing gap span (the group-rep
                    # convention); a read-empty gap behaves exactly like the
                    # read-empty tail target (no-evidence -> honest
                    # unresolved -> pin overrides).  If the lattice cannot
                    # express a boundary by gap_b (no span end reaches it
                    # inside the scan), record a typed reason and keep the
                    # previous cap -- never invent span semantics.
                    _bpf_sb = next(
                        (si for si in range(cursor + 1, scan_hi + 1)
                         if int(f1_of(si)) >= _bpf_b), None)
                    if _bpf_sb is None:
                        if bridge_pin_stats[
                                "synthetic_rest_unavailable"] is None:
                            bridge_pin_stats["synthetic_rest_unavailable"] = (
                                "gap-beyond-scan")
                            print("  [scrub-bridge-pin] synthetic rest "
                                  "unavailable (gap-beyond-scan); capping at "
                                  "the existing lattice rest", flush=True)
                    elif targets[_bpf_cap_i][1] > _bpf_sb:
                        _bpf_sa = next(
                            (si for si in range(cursor + 1, _bpf_sb + 1)
                             if int(f1_of(si)) >= _bpf_a), _bpf_sb)
                        _bpf_readable = [
                            si for si in range(_bpf_sa, _bpf_sb + 1)
                            if has_reads(si)]
                        _bpf_rep_syn = (
                            max(_bpf_readable, key=coverage)
                            if _bpf_readable else _bpf_sb)
                        targets.insert(_bpf_cap_i,
                                       (_bpf_rep_syn, _bpf_sb))
                        bridge_pin_synthetic_rest = True
                    if _bpf_cap_i + 1 < len(targets):
                        targets = targets[:_bpf_cap_i + 1]
                        bridge_pin_forced_boundary = True
            except Exception as exc:       # fail-soft: stock behavior kept
                bridge_pin_forced_boundary = False
                bridge_pin_synthetic_rest = False
                bridge_pin_stats["failsoft"] = (
                    f"boundary:{type(exc).__name__}: {exc}")
                print(f"  [scrub-bridge-pin] boundary FAILSOFT "
                      f"(stock scrub kept): {type(exc).__name__}: {exc}",
                      flush=True)
                bridge_pin = None
        decided = None
        best_searched = None           # largest searched (ambiguous) window
        attempt = 0
        t_att = time.monotonic()
        for t_i, (b_rep, b_last) in enumerate(targets):
            t_att = time.monotonic()
            if os.environ.get("CUBED_V5A_DIAG") == "1":
                try:
                    print(
                        "[v5a-diag] SITE4 window-start "
                        f"t={time.time():.3f} "
                        f"w_idx={w_idx} t_i={t_i} attempt={attempt} "
                        f"window=[{int(f1_of(cursor))},{int(f1_of(b_last))}]",
                        flush=True)
                except Exception:
                    pass
            L_est = _events_in(move_events, f1_of(cursor), f1_of(b_rep))
            # om-continuity ball: the om can have drifted by at most one
            # rotation gesture per motion event inside the window (plus any
            # unread drift carried from earlier unresolved windows).
            radius = (_events_in(rotation_events, f1_of(cursor), f1_of(b_last))
                      if om_stateful_active else L_est + carried_extra)
            allowed = (om_reachable(om_nbrs, carried_ois, radius)
                       if om_stateful_active else
                       om_ball(om_nbrs, carried_oi, radius))
            om_ctx = (dict(
                start_ois=set(carried_ois),
                move_frames=[f for f in move_events
                             if f1_of(cursor) < f <= f1_of(b_last)],
                rotation_frames=[f for f in rotation_events
                                 if f1_of(cursor) < f <= f1_of(b_last)],
                rotation_intervals=[
                    interval for interval in gate_streams["rotation_intervals"]
                    if f1_of(cursor) < interval["frame"] <= f1_of(b_last)])
                      if om_stateful_active else None)
            res = search_window(state, cursor, b_last, L_est, allowed,
                                win_ctx=(w_idx, attempt), om_ctx=om_ctx)
            if res.get("beam_engaged"):
                beam_stats["searches"] += 1
                beam_stats["n_pruned"] += int(res.get("n_pruned") or 0)
            # over-cap / scaling-tripped windows only get WORSE when
            # extended (deeper); treat them as the structural cap now.
            hard_cap = (res["status"] == "over-cap"
                        or str(res["status"]).startswith("tripped"))
            at_cap = (t_i == len(targets) - 1) or hard_cap
            row = base_row(attempt, b_rep, b_last, L_est, res, radius,
                           allowed)
            row["wall_ms"] = int((time.monotonic() - t_att) * 1000)
            if os.environ.get("CUBED_V5A_DIAG") == "1":
                try:
                    print(
                        "[v5a-diag] SITE4 window-end "
                        f"t={time.time():.3f} "
                        f"w_idx={w_idx} t_i={t_i} attempt={attempt} "
                        f"window=[{int(f1_of(cursor))},{int(f1_of(b_last))}] "
                        f"wall_ms={row['wall_ms']}",
                        flush=True)
                except Exception:
                    pass
            if res["status"] == "ok":
                dec = shadow_window_decision(
                    res, at_cap, boundary_span=b_last,
                    site_kind="ordinary-window", attempt=attempt)
                dec, containment_reason = enforce_stateful_containment(
                    dec, res, at_cap)
                row["committed_comparable"] = res.get(
                    "committed_comparable", False)
                row["incumbent_selected"] = res.get(
                    "incumbent_selected", False)
                sg = res["state_groups"]
                row.update(best_score=sg[0]["best_score"],
                           runner_up_score=(sg[1]["best_score"]
                                            if len(sg) > 1 else None),
                           **dec_fields(dec))
                if containment_reason is not None:
                    row["stateful_containment"] = containment_reason
                if dec["decision"] == "extend":
                    counts["extended"] += 1
                    best_searched = (b_rep, b_last, res, row)
                    emit_line(**row)
                    attempt += 1
                    continue
                else:
                    decided = (dec["decision"], b_rep, b_last, res, row)
                    break
            if not at_cap:             # no-evidence: extending may find reads
                row["decision"] = "extend"
                counts["extended"] += 1
                emit_line(**row)
                attempt += 1
                continue
            # unsearchable at the cap
            emit_line(**dict(row, decision="cap-attempt"))
            if best_searched is not None:
                # state-first honesty on the largest SEARCHED window
                pb_rep, pb_last, pres, prow = best_searched
                pdec = shadow_window_decision(
                    pres, True, boundary_span=pb_last,
                    site_kind="largest-searched-at-cap", attempt=attempt)
                pdec, containment_reason = enforce_stateful_containment(
                    pdec, pres, True)
                prow["committed_comparable"] = pres.get(
                    "committed_comparable", False)
                prow["incumbent_selected"] = pres.get(
                    "incumbent_selected", False)
                prow = dict(prow, attempt=attempt,
                            cap_reason=res["status"], **dec_fields(pdec))
                if containment_reason is not None:
                    prow["stateful_containment"] = containment_reason
                decided = (pdec["decision"], pb_rep, pb_last, pres, prow)
            else:
                decided = ("unresolved", b_rep, b_last, None,
                           dict(row, decision="unresolved",
                                continuation_src="committed-hypothesis"))
            break
        if decided is None:            # loop invariant: at_cap always decides
            raise RuntimeError("scrub window loop reached no decision")

        # TWO-TIER GATE-DROP RESCUE. The primary search never sees an optional
        # final-drop slot: every drop remains only an OM-rotation opportunity.
        # Tier 1 retries authoritative color=MOVE/alignment=REGRIP conflicts.
        # If that cannot adopt, tier 2 retries every final drop but carries one
        # transaction-wide bit: at most ONE dropped gap may emit a SINGLE; every
        # other gap remains the legacy rotation/SKIP. A successful but
        # nondecisive tier 1 is retained and unioned with tier 2 *after* both
        # have used the common final scorer; tier 2 can add hypotheses but may
        # not erase tier-1 survivors through independent beam pruning. Both
        # tiers spend the same 2sigma adoption gate, and any failure leaves the
        # primary incumbent.
        gate_tier1_result = None

        def run_gate_drop_rescue(current, source_intervals, *, tier,
                                 source_name, single_cap):
            nonlocal attempt, gate_tier1_result
            if (not gate_drop_slots_active or current[0] != "unresolved"
                    or current[4].get("split_child_unresolved")):
                return current
            rescue_rep, rescue_last = current[1], current[2]
            rescue_intervals = tuple(
                interval for interval in source_intervals
                if f1_of(cursor) < int(interval["frame"])
                <= f1_of(rescue_last)
                and not any(
                    int(interval["lo"]) <= int(frame)
                    <= int(interval["hi"])
                    for frame in move_events
                    if f1_of(cursor) < int(frame) <= f1_of(rescue_last)))
            if not rescue_intervals:
                return current

            gate_report = report["gate_drop_slots"]
            gate_report["rescue_attempts"] = int(
                gate_report["rescue_attempts"]) + 1
            tier_attempts = f"{tier}_attempts"
            gate_report[tier_attempts] = int(
                gate_report.get(tier_attempts, 0)) + 1
            rescue_attempt = attempt + 1
            attempt = rescue_attempt
            rescue_L_est = _events_in(
                move_events, f1_of(cursor), f1_of(rescue_rep))
            rescue_radius = _events_in(
                rotation_events, f1_of(cursor), f1_of(rescue_last))
            rescue_allowed = om_reachable(
                om_nbrs, carried_ois, rescue_radius)
            rescue_om_ctx = dict(
                start_ois=set(carried_ois),
                move_frames=[
                    frame for frame in move_events
                    if f1_of(cursor) < frame <= f1_of(rescue_last)],
                rotation_frames=[
                    frame for frame in rotation_events
                    if f1_of(cursor) < frame <= f1_of(rescue_last)],
                rotation_intervals=[
                    interval for interval
                    in gate_streams["rotation_intervals"]
                    if f1_of(cursor) < interval["frame"]
                    <= f1_of(rescue_last)],
            )
            rescue_t0 = time.monotonic()
            rescue_result = None
            rescue_error = None
            bound_fallbacks_before = int(
                gate_report.get("bound_fallbacks", 0))
            score_fallbacks_before = int(
                gate_report.get("score_error_fallbacks", 0))
            try:
                rescue_result = search_window(
                    state, cursor, rescue_last, rescue_L_est,
                    rescue_allowed,
                    win_ctx=(w_idx, rescue_attempt),
                    om_ctx=rescue_om_ctx,
                    dropped_intervals=rescue_intervals,
                    gate_drop_single_cap=single_cap)
            except Exception as exc:  # feature-local fail-closed boundary
                rescue_error = exc
            rescue_wall_ms = int(
                (time.monotonic() - rescue_t0) * 1000)
            gate_report["rescue_wall_ms"] = int(
                gate_report["rescue_wall_ms"]) + rescue_wall_ms
            tier_wall = f"{tier}_wall_ms"
            gate_report[tier_wall] = int(
                gate_report.get(tier_wall, 0)) + rescue_wall_ms

            status_fields = dict(
                gate_drop_rescue_tried=True,
                gate_drop_rescue_tier=tier,
                gate_drop_rescue_source=source_name,
                gate_drop_rescue_wall_ms=rescue_wall_ms,
                gate_drop_rescue_single_cap=single_cap)
            tier_status = f"gate_drop_{tier}_status"
            if rescue_error is not None:
                gate_report["rescue_failures"] = int(
                    gate_report["rescue_failures"]) + 1
                tier_failures = f"{tier}_failures"
                gate_report[tier_failures] = int(
                    gate_report.get(tier_failures, 0)) + 1
                gate_report.setdefault("rescue_errors", []).append({
                    "tier": tier,
                    "frames": [f1_of(cursor), f1_of(rescue_last)],
                    "error_type": type(rescue_error).__name__,
                    "error_message": str(rescue_error),
                })
                current[4].update(
                    status_fields,
                    **{tier_status: "error",
                       "gate_drop_rescue_status": "error"})
                return current

            # Tier 1 historically replays the rotation-only control inside the
            # beam when an optional-slot transaction exceeds its exact bound
            # or its action scorer raises.  That replay is safe as a local
            # implementation fallback, but it is not an evidential rescue and
            # must never be adopted as though optional hypotheses were scored.
            # It also blocks tier 2 for this window: a technical failure in the
            # narrower transaction is not permission to open the broader one.
            bound_fallback_delta = int(
                gate_report.get("bound_fallbacks", 0)
            ) - bound_fallbacks_before
            score_fallback_delta = int(
                gate_report.get("score_error_fallbacks", 0)
            ) - score_fallbacks_before
            if bound_fallback_delta or score_fallback_delta:
                gate_report["rescue_failures"] = int(
                    gate_report["rescue_failures"]) + 1
                tier_failures = f"{tier}_failures"
                gate_report[tier_failures] = int(
                    gate_report.get(tier_failures, 0)) + 1
                fallback_status = (
                    "score-error-control-fallback"
                    if score_fallback_delta else
                    "bound-control-fallback")
                current[4].update(
                    status_fields,
                    gate_drop_rescue_status=fallback_status,
                    gate_drop_rescue_n_slots=rescue_result.get(
                        "n_gate_drop_slots", 0),
                    gate_drop_rescue_max_singles=rescue_result.get(
                        "max_gate_drop_singles", 0),
                    gate_drop_rescue_slot_audit=rescue_result.get(
                        "gate_drop_slot_audit", []),
                    **{tier_status: fallback_status})
                return current

            if rescue_result.get("status") != "ok":
                gate_report["rescue_failures"] = int(
                    gate_report["rescue_failures"]) + 1
                tier_failures = f"{tier}_failures"
                gate_report[tier_failures] = int(
                    gate_report.get(tier_failures, 0)) + 1
                result_status = rescue_result.get("status")
                current[4].update(
                    status_fields,
                    gate_drop_rescue_status=result_status,
                    gate_drop_rescue_n_slots=rescue_result.get(
                        "n_gate_drop_slots", 0),
                    gate_drop_rescue_max_singles=rescue_result.get(
                        "max_gate_drop_singles", 0),
                    gate_drop_rescue_slot_audit=rescue_result.get(
                        "gate_drop_slot_audit", []),
                    **{tier_status: result_status})
                return current

            if rescue_result.get("beam_engaged"):
                beam_stats["searches"] += 1
                beam_stats["n_pruned"] += int(
                    rescue_result.get("n_pruned") or 0)
            if tier == "tier2" and gate_tier1_result is not None:
                try:
                    rescue_result = _union_gate_tier_results(
                        gate_tier1_result, rescue_result)
                except Exception as exc:  # complete-union transaction boundary
                    gate_report["rescue_failures"] = int(
                        gate_report["rescue_failures"]) + 1
                    gate_report["tier2_failures"] = int(
                        gate_report["tier2_failures"]) + 1
                    gate_report["tier_union_failures"] = int(
                        gate_report["tier_union_failures"]) + 1
                    error_row = {
                        "frames": [f1_of(cursor), f1_of(rescue_last)],
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    gate_report.setdefault(
                        "tier_union_errors", []).append(error_row)
                    current[4].update(
                        status_fields,
                        gate_drop_rescue_status="tier-union-error",
                        gate_drop_tier_union_error=error_row,
                        **{tier_status: "tier-union-error"})
                    return current
                union_meta = rescue_result["gate_drop_tier_union"]
                gate_report["tier_unions"] = int(
                    gate_report["tier_unions"]) + 1
                gate_report["tier_union_candidates"] = int(
                    gate_report["tier_union_candidates"]
                ) + int(union_meta["union_candidates"])
            rescue_dec = shadow_window_decision(
                rescue_result, True, boundary_span=rescue_last,
                site_kind=f"gate-drop-{tier}-rescue",
                attempt=rescue_attempt)
            rescue_dec, rescue_containment = enforce_stateful_containment(
                rescue_dec, rescue_result, True)
            if rescue_dec["decision"] not in ("commit", "low_conf"):
                gate_report["rescue_nondecisive"] = int(
                    gate_report["rescue_nondecisive"]) + 1
                tier_nondecisive = f"{tier}_nondecisive"
                gate_report[tier_nondecisive] = int(
                    gate_report.get(tier_nondecisive, 0)) + 1
                decision_status = rescue_dec["decision"]
                current[4].update(
                    status_fields,
                    gate_drop_rescue_status=decision_status,
                    gate_drop_tier_union=rescue_result.get(
                        "gate_drop_tier_union"),
                    gate_drop_rescue_n_slots=rescue_result.get(
                        "n_gate_drop_slots", 0),
                    gate_drop_rescue_max_singles=rescue_result.get(
                        "max_gate_drop_singles", 0),
                    gate_drop_rescue_slot_audit=rescue_result.get(
                        "gate_drop_slot_audit", []),
                    **{tier_status: decision_status})
                if tier == "tier1":
                    gate_tier1_result = rescue_result
                return current

            gate_report["rescue_adoptions"] = int(
                gate_report["rescue_adoptions"]) + 1
            tier_adoptions = f"{tier}_adoptions"
            gate_report[tier_adoptions] = int(
                gate_report.get(tier_adoptions, 0)) + 1
            rescue_row = base_row(
                rescue_attempt, rescue_rep, rescue_last,
                rescue_L_est, rescue_result, rescue_radius,
                rescue_allowed)
            rescue_row["committed_comparable"] = rescue_result.get(
                "committed_comparable", False)
            rescue_row["incumbent_selected"] = rescue_result.get(
                "incumbent_selected", False)
            rescue_groups = rescue_result["state_groups"]
            rescue_row.update(
                status_fields,
                gate_drop_rescue=True,
                gate_drop_rescue_status=rescue_dec["decision"],
                gate_drop_rescue_max_singles=rescue_result.get(
                    "max_gate_drop_singles", 0),
                gate_drop_tier_union=rescue_result.get(
                    "gate_drop_tier_union"),
                **{tier_status: rescue_dec["decision"]},
                best_score=rescue_groups[0]["best_score"],
                runner_up_score=(
                    rescue_groups[1]["best_score"]
                    if len(rescue_groups) > 1 else None),
                wall_ms=rescue_wall_ms,
                **dec_fields(rescue_dec))
            if tier == "tier2" and gate_tier1_result is not None:
                rescue_row["gate_drop_tier1_status"] = "unresolved"
            if rescue_containment is not None:
                rescue_row["stateful_containment"] = rescue_containment
            return (rescue_dec["decision"], rescue_rep, rescue_last,
                    rescue_result, rescue_row)

        decided = run_gate_drop_rescue(
            decided, gate_drop_intervals, tier="tier1",
            source_name="alignment-veto-only-final-drops",
            single_cap=None)
        # A genuine nondecision may be narrowed by tier 2.  Any other tier-1
        # status is either an adoption (already no longer unresolved) or a
        # technical failure, which fails closed to the original incumbent.
        if decided[4].get("gate_drop_tier1_status") in (None, "unresolved"):
            decided = run_gate_drop_rescue(
                decided, gate_drop_all_intervals, tier="tier2",
                source_name="all-final-drops-at-most-one-single",
                single_cap=1)

        # CLUSTER FALLBACK: fires ONLY where the burst-aligned search
        # already gave up (unresolved). One re-search with cluster burst
        # slots + disagreement-derived slack; adopted only on a decisive
        # (commit/low_conf) verdict at the SAME 2sigma gate — otherwise the
        # original unresolved decision (committed-hypothesis continuation)
        # stands unchanged. Flag-off ⇒ this block never runs.
        if (cluster_fallback and decided[0] == "unresolved"
                and not decided[4].get("split_child_unresolved")):
            fb_rep, fb_last = decided[1], decided[2]
            n_cl_w = _events_in(
                move_events_cl, f1_of(cursor), f1_of(fb_rep))
            radius_fb = (_events_in(rotation_events, f1_of(cursor),
                                    f1_of(fb_last))
                         if om_stateful_active else n_cl_w + carried_extra)
            allowed_fb = (om_reachable(om_nbrs, carried_ois, radius_fb)
                          if om_stateful_active else
                          om_ball(om_nbrs, carried_oi, radius_fb))
            om_ctx_fb = (dict(
                start_ois=set(carried_ois),
                # A cluster representative is a structural/count slot.  It can
                # stand for multiple raw actions and is not an exact physical
                # timestamp for final replacement scoring.
                timing_authoritative=False,
                move_frames=[f for f in move_events_cl
                             if f1_of(cursor) < f <= f1_of(fb_last)],
                rotation_frames=[f for f in rotation_events
                                 if f1_of(cursor) < f <= f1_of(fb_last)],
                rotation_intervals=[
                    interval for interval in gate_streams["rotation_intervals"]
                    if f1_of(cursor) < interval["frame"] <= f1_of(fb_last)])
                         if om_stateful_active else None)
            fb_t0 = time.monotonic()
            res_fb = search_window(state, cursor, fb_last, 0, allowed_fb,
                                   win_ctx=(w_idx, attempt + 1),
                                   cluster_mode=True, om_ctx=om_ctx_fb)
            fb_wall_ms = int((time.monotonic() - fb_t0) * 1000)
            cluster_fallback_stats["attempts"] += 1
            cluster_fallback_stats["wall_ms"] += fb_wall_ms
            if res_fb.get("beam_engaged"):
                beam_stats["searches"] += 1
                beam_stats["n_pruned"] += int(res_fb.get("n_pruned") or 0)
            if res_fb["status"] == "ok":
                dec_fb = shadow_window_decision(
                    res_fb, True, boundary_span=fb_last,
                    site_kind="cluster-fallback", attempt=attempt + 1)
                dec_fb, containment_reason = enforce_stateful_containment(
                    dec_fb, res_fb, True)
                if dec_fb["decision"] in ("commit", "low_conf"):
                    row_fb = base_row(attempt + 1, fb_rep, fb_last,
                                      res_fb["L_est_eff"], res_fb, radius_fb,
                                      allowed_fb)
                    row_fb["committed_comparable"] = res_fb.get(
                        "committed_comparable", False)
                    row_fb["incumbent_selected"] = res_fb.get(
                        "incumbent_selected", False)
                    sg_fb = res_fb["state_groups"]
                    row_fb.update(cluster_fallback=True,
                                  cluster_fallback_wall_ms=fb_wall_ms,
                                  slack_eff=res_fb["slack_eff"],
                                  best_score=sg_fb[0]["best_score"],
                                  runner_up_score=(sg_fb[1]["best_score"]
                                                   if len(sg_fb) > 1
                                                   else None),
                                  wall_ms=int((time.monotonic() - t_att)
                                              * 1000),
                                  **dec_fields(dec_fb))
                    if containment_reason is not None:
                        row_fb["stateful_containment"] = containment_reason
                    # Cluster fallback is a later, independent transaction.
                    # If it adopts after a gate-drop transaction failed back
                    # to rotation-only control, keep that earlier failure
                    # audit on the one emitted window row.  Replacing the row
                    # must not erase why the optional rescue was rejected.
                    for key, value in decided[4].items():
                        if (key.startswith("gate_drop_rescue_")
                                or key.startswith("gate_drop_tier")):
                            row_fb.setdefault(key, value)
                    decided = (dec_fb["decision"], fb_rep, fb_last, res_fb,
                               row_fb)
                else:
                    decided[4]["cluster_fallback_tried"] = True
                    decided[4]["cluster_fallback_wall_ms"] = fb_wall_ms
            else:
                decided[4]["cluster_fallback_tried"] = True
                decided[4]["cluster_fallback_wall_ms"] = fb_wall_ms

        decision, b_rep, b_last, res, row = decided
        fallback_token_oms = None
        w_events = _events_in((rotation_events if om_stateful_active else events),
                              f1_of(cursor), f1_of(b_last))
        if res is not None and decision == "unresolved":
            # An unresolved window's top candidate is, by its own margin,
            # indistinguishable from its runner-up — adopting it emits noise
            # AND contaminates every downstream window's start state (observed:
            # 4 sub-1sigma junk continuations, then a 3.9sigma
            # confident-wrong commit searched from the drifted state). The
            # searched word stays in the audit row; the continuation falls
            # back to the demoted committed hypothesis, same as the
            # unsearchable branch below.
            row["searched_top_word"] = list(BS.tokens_of(
                res["seqs"][res["state_groups"][0]["best_i"]]))
            row["continuation_src"] = "committed-hypothesis"
            res = None
        chosen_search_tokens = None
        chosen_slot_merges = []
        if res is not None:
            top = res["state_groups"][0]
            chosen_seq = tuple(int(v) for v in res["seqs"][top["best_i"]])
            tokens = list(BS.tokens_of(chosen_seq))
            chosen_search_tokens = list(tokens)
            matching_merge_programs = [
                merge_program for merge_program in
                (res.get("slot_merges") or ())
                if tuple(merge_program.get("word") or ()) == chosen_seq
            ]
            if matching_merge_programs:
                merge_program = max(
                    matching_merge_programs,
                    key=lambda program: float(
                        program.get("beam_score", float("-inf"))))
                chosen_search_tokens = list(BS.tokens_of(
                    tuple(int(v) for v in merge_program["search_word"])))
                chosen_slot_merges = copy.deepcopy(
                    merge_program.get("merges") or ())
                row["chosen_slot_merge_selection"] = (
                    "best-device-prefix-for-selected-physical-word")
            new_state = np.frombuffer(top["state"], np.int8).copy()
            if chosen_slot_merges:
                replay = state.copy()
                for token in tokens:
                    replay = replay[perms[BS.MOVES.index(token)]]
                if not np.array_equal(replay, new_state):
                    raise RuntimeError(
                        "raw slot-merge emission changed selected endpoint")
            allowed_dec = res.get("allowed_ois")
        else:
            # honest continuation on the DEMOTED committed hypothesis (the
            # only candidate when the window was unsearchable); flagged.
            tokens = list(committed_window(cursor, b_last))
            fallback_token_oms = committed_window_oms(cursor, b_last)
            new_state = state.copy()
            for t in tokens:
                new_state = new_state[perms[BS.MOVES.index(t)]]
            chosen_search_tokens = list(tokens)
            allowed_dec = (om_reachable(om_nbrs, carried_ois, w_events)
                           if om_stateful_active else
                           om_ball(om_nbrs, carried_oi,
                                   w_events + carried_extra))
        # BRIDGE PIN engagement (CUBED_SCRUB_BRIDGE_PIN; _load_bridge_pin has
        # the schema + committed-state boundary).  Predicate derived from how this loop
        # frames windows -- the committed range is (f1_of(cursor),
        # f1_of(b_last)] over the rest lattice, so the FIRST window with
        # f1_of(b_last) >= gap_a and f1_of(cursor) < gap_b is the first whose
        # range reaches (ends inside [a,b], the usual ends-at-the-certified-
        # rest case) or contains (extends past frame b) the pin gap.  One-shot;
        # evaluated on the DECIDED window so extensions resolve first.  The
        # override replaces the emitted tokens with the certified bridge word
        # and the carried committed state with B *before* om resolution and
        # *before* out_windows.append -- the ordinary splice path.  res is
        # dropped to None so om resolves with the module's own machinery on
        # the pin state (a normally committed window), never from beam
        # candidate scores that described the replaced hypothesis.
        bridge_pin_engaged_now = False
        bridge_pin_om_index = None
        if bridge_pin is not None and not bridge_pin_stats["engaged"]:
            try:
                _bp_f_lo, _bp_f_hi = int(f1_of(cursor)), int(f1_of(b_last))
                _bp_a, _bp_b = bridge_pin["gap"]
                if _bp_f_hi >= _bp_a and _bp_f_lo < _bp_b:
                    # Everything that can raise happens BEFORE any override
                    # lands (atomic engage; fail-soft leaves stock intact).
                    _bp_state = np.asarray(bridge_pin["state"],
                                           np.int8).copy()
                    _bp_word = list(bridge_pin["word"])
                    _bp_pre_key = _bridge_pin_state_key(new_state)
                    _bp_key = _bridge_pin_state_key(_bp_state)
                    _bp_replaced_word = list(tokens)
                    tokens = _bp_word
                    chosen_search_tokens = list(tokens)
                    chosen_slot_merges = []
                    new_state = _bp_state
                    res = None
                    fallback_token_oms = None
                    bridge_pin_om_index = bridge_pin["om"]
                    bridge_pin_engaged_now = True
                    bridge_pin_stats["engaged"] = 1
                    bridge_pin_stats["window_idx"] = int(w_idx)
                    bridge_pin_stats["window_frames"] = [_bp_f_lo, _bp_f_hi]
                    bridge_pin_stats["forced_boundary"] = bool(
                        bridge_pin_forced_boundary)
                    bridge_pin_stats["synthetic_rest"] = bool(
                        bridge_pin_synthetic_rest)
                    row["bridge_pin"] = dict(
                        engaged=True, gap=[int(_bp_a), int(_bp_b)],
                        window_frames=[_bp_f_lo, _bp_f_hi],
                        forced_boundary=bool(bridge_pin_forced_boundary),
                        synthetic_rest=bool(bridge_pin_synthetic_rest),
                        pre_pin_state_key=_bp_pre_key,
                        pin_state_key=_bp_key,
                        word=list(_bp_word),
                        replaced_word=_bp_replaced_word,
                        source=bridge_pin["source"],
                        replaced_decision=str(decision))
                    row["continuation_src"] = "bridge-pin"
            except Exception as exc:       # fail-soft: stock behavior kept
                bridge_pin_engaged_now = False
                bridge_pin_om_index = None
                bridge_pin_stats["failsoft"] = (
                    f"engage:{type(exc).__name__}: {exc}")
                print(f"  [scrub-bridge-pin] FAILSOFT at window {w_idx} "
                      f"(stock scrub kept): {type(exc).__name__}: {exc}",
                      flush=True)
                bridge_pin = None
        prior_oi = carried_oi
        move_oi_trace = None
        fallback_incumbent_self = False
        if res is not None and res.get("incumbent_selected"):
            modes = res.get("candidate_score_modes") or ()
            selected_i = (top.get("best_i") if top is not None else None)
            fallback_incumbent_self = bool(
                selected_i is not None and selected_i < len(modes)
                and _score_mode_contains(modes[selected_i], "fallback"))
            if fallback_incumbent_self:
                # Self-continuation is state-safe, but a fallback OM posterior
                # was not scored in the temporal currency.  Preserve the known
                # baseline OM band instead of laundering it into a new exact OM.
                fallback_token_oms = committed_window_oms(cursor, b_last)
                row["om_posterior_preserved"] = "fallback-incumbent-self"
        if om_stateful_active:
            oi_scores = None
            if (not fallback_incumbent_self and res is not None
                    and res.get("candidate_om_scores") is not None):
                oi_scores = res["candidate_om_scores"][top["best_i"]]
            if oi_scores is not None and np.isfinite(oi_scores).any():
                best_score = float(np.max(oi_scores))
                carried_ois = {int(i) for i, value in enumerate(oi_scores)
                               if np.isfinite(value)
                               and best_score - float(value)
                               <= float(z_sigma) * sigma}
                oi = min(carried_ois,
                         key=lambda i: (-float(oi_scores[i]), int(i)))
                traces_for_band = [
                    (res.get("move_om_traces") or {}).get(
                        (top["best_i"], band_oi))
                    for band_oi in sorted(carried_ois)]
                trace_ambiguities = set().union(*[
                    set((res.get("move_om_trace_ambiguities") or {}).get(
                        (top["best_i"], band_oi), ()))
                    for band_oi in sorted(carried_ois)])
                if (not res.get("ambiguous_move_positions")
                        and not trace_ambiguities
                        and traces_for_band
                        and all(trace is not None
                                and trace == traces_for_band[0]
                                for trace in traces_for_band)):
                    move_oi_trace = traces_for_band[0]
                else:
                    row["move_om_path_ambiguity_positions"] = sorted(
                        trace_ambiguities)
            elif fallback_incumbent_self:
                carried_ois = set(carried_ois)
                oi = prior_oi
            else:
                ranked = rank_oms(b_rep, new_state, allowed_dec)
                if ranked:
                    best_score = ranked[0][1]
                    carried_ois = {i for i, value in ranked
                                   if best_score - value
                                   <= float(z_sigma) * sigma}
                    oi = ranked[0][0]
                else:
                    carried_ois = (set(allowed_dec) if allowed_dec is not None
                                   else set(range(len(orientations))))
                    oi = (prior_oi if prior_oi in carried_ois
                          else min(carried_ois))
            carried_oi = oi
            carried_extra = 0
            row["om_post_indices"] = sorted(carried_ois)
        else:
            oi = argmax_om(b_rep, new_state, allowed_dec)
            if oi is None and res is not None and res.get("evidence_spans"):
                oi = argmax_om(res["evidence_spans"][-1], new_state,
                               allowed_dec)
            # Legacy OM continuity carry (flag-off path kept verbatim).
            if oi is not None:
                carried_oi = oi
                carried_extra = 0
            else:
                carried_extra += w_events
        if bridge_pin_engaged_now:
            # om carry per pin contract: an int pin om IS the carried om; a
            # null pin om keeps the module's own resolution above (om_ball /
            # OM-CONTINUITY on the pin state, computed like any committed
            # window -- no new om inference).  Then the mandatory receipt.
            try:
                if bridge_pin_om_index is not None:
                    oi = int(bridge_pin_om_index)
                    carried_oi = oi
                    carried_extra = 0
                    if om_stateful_active:
                        carried_ois = {oi}
                        row["om_post_indices"] = sorted(carried_ois)
                    _bp_om_src = "pin"
                elif oi is not None:
                    _bp_om_src = "module-resolved"
                else:
                    _bp_om_src = "module-unresolved-carry"
                # Cross-instrument om adjudication data (e.g. vs the chain
                # certifier's viterbi om_timeline): the module's om INDEX is
                # only meaningful against the module's own orientation list,
                # so the receipt carries the resolved (up, front) key plus
                # the list identity (its length).  No injection, no remap --
                # om=null semantics unchanged.
                _bp_om_key = (list(om_key_of(orientations[oi]))
                              if oi is not None else None)
                row["bridge_pin"]["om_source"] = _bp_om_src
                row["bridge_pin"]["om_index"] = (
                    int(oi) if oi is not None else None)
                row["bridge_pin"]["om_key"] = _bp_om_key
                row["bridge_pin"]["n_orientations"] = len(orientations)
                bridge_pin_stats["om_source"] = _bp_om_src
                bridge_pin_stats["om_index"] = (
                    int(oi) if oi is not None else None)
                bridge_pin_stats["om_key"] = _bp_om_key
                bridge_pin_stats["n_orientations"] = len(orientations)
                print(f"  [scrub-bridge-pin] ENGAGED window={w_idx} "
                      f"frames={bridge_pin_stats['window_frames']} "
                      f"gap={row['bridge_pin']['gap']} "
                      f"pre_pin_key={row['bridge_pin']['pre_pin_state_key']} "
                      f"pin_key={row['bridge_pin']['pin_state_key']} "
                      f"om={_bp_om_src}:{row['bridge_pin']['om_index']} "
                      f"om_key={_bp_om_key} "
                      f"n_orients={len(orientations)} "
                      f"forced_boundary="
                      f"{'true' if row['bridge_pin']['forced_boundary'] else 'false'} "
                      f"synthetic_rest="
                      f"{'true' if row['bridge_pin']['synthetic_rest'] else 'false'}",
                      flush=True)
            except Exception as exc:      # receipt-only; word/state stand
                bridge_pin_stats["failsoft"] = (
                    f"om:{type(exc).__name__}: {exc}")
                print(f"  [scrub-bridge-pin] om/receipt FAILSOFT "
                      f"(module om kept): {type(exc).__name__}: {exc}",
                      flush=True)
        om_obj = (orientations[oi] if oi is not None
                  else (last_om if last_om is not None else orientations[0]))
        last_om = om_obj
        if om_stateful_active:
            if (fallback_token_oms is not None
                    and len(fallback_token_oms) == len(tokens)):
                token_oms = list(fallback_token_oms)
                row["move_om_trace_indices"] = [
                    orientations.index(om) if om in orientations else None
                    for om in token_oms]
                row["move_om_ambiguous"] = False
                row["move_om_source"] = "committed-baseline"
            elif (move_oi_trace is not None
                  and len(move_oi_trace) == len(tokens)):
                token_oms = [orientations[i] for i in move_oi_trace]
                row["move_om_trace_indices"] = list(move_oi_trace)
                row["move_om_ambiguous"] = False
                row["move_om_source"] = "stateful-path"
            else:
                fallback_oi = prior_oi if prior_oi is not None else oi
                token_oms = [orientations[fallback_oi]] * len(tokens)
                row["move_om_trace_indices"] = None
                row["move_om_ambiguous"] = bool(tokens)
                row["move_om_source"] = "ambiguous-placeholder"
                if res is not None and res.get("ambiguous_move_positions"):
                    row["move_rotation_overlap_positions"] = sorted(
                        int(i) for i in res["ambiguous_move_positions"])
        else:
            token_oms = [om_obj] * len(tokens)
        if chosen_slot_merges:
            row["chosen_search_word"] = list(chosen_search_tokens)
            row["chosen_slot_merges"] = copy.deepcopy(
                chosen_slot_merges)
        row.update(decision=decision, chosen_word=list(tokens),
                   chosen_nf=list(BS.normal_form(tokens)),
                   agrees_with_committed=(
                       list(BS.normal_form(tokens))
                       == list(BS.normal_form(
                           committed_window(cursor, b_last)))),
                   low_conf=(decision == "low_conf"), om_index=oi,
                   wall_ms=int((time.monotonic() - t_att) * 1000))
        emit_line(**row)
        counts["committed" if decision == "commit" else
               ("low_conf" if decision == "low_conf" else
                "unresolved")] += 1
        out_windows.append(dict(
            tokens=tokens,
            land=max(0, cursor + 1),
            oms=token_oms,
            # This is the terminal evidence span for the whole window. It is a
            # decoder checkpoint, not an estimate of when any move occurred.
            checkpoint_frame=int(f1_of(b_last)),
        ))
        if (ll_onset is None and om_stateful_active
                and decision in ("commit", "low_conf")
                and row.get("state_unique")):
            cross_color = LLC.verified_ll_crossing(state, new_state)
            if cross_color is not None:
                prefix_moves = tuple(
                    token for window in out_windows
                    for token in window["tokens"]
                )
                ll_onset = dict(
                    state=new_state.copy(),
                    span=int(b_last),
                    frame=int(f1_of(b_last)),
                    prefix_windows=len(out_windows),
                    prefix_moves=prefix_moves,
                    onset_ois=frozenset(int(value)
                                        for value in carried_ois),
                    cross_color=str(cross_color),
                    decision=str(decision),
                )
                report["ll_completion"] = dict(
                    status="armed",
                    onset_span=int(b_last),
                    onset_frame=int(f1_of(b_last)),
                    cross_color=str(cross_color),
                    onset_decision=str(decision),
                    onset_om_band=sorted(ll_onset["onset_ois"]),
                )
        state = new_state
        cursor = b_last
        w_idx += 1

    # ----------------------------------------------- terminal endpoint check
    tail_moves = list(moves[n_solve:])
    tail_ok = True
    full_end = state.copy()
    for t in tail_moves:
        try:
            full_end = full_end[perms[BS.MOVES.index(t)]]
        except ValueError:
            tail_ok = False
            break
    final_ok, final_dist = None, None
    if final_arr is not None and tail_ok:
        fin = np.asarray(final_arr, np.int8)
        final_ok = bool(np.array_equal(full_end, fin))
        if not final_ok:
            try:                       # bounded ball-distance report (<=5)
                fwd, bwd = BS.build_balls(full_end, fin, perms, 3)
                _st, dcands = BS.join_budget(fwd, bwd, 3, 2)
                if dcands:
                    final_dist = min(len(BS.reduce_word(BS.tokens_of(s)))
                                     for s in dcands)
            except Exception:
                final_dist = None

    # Scrub also runs in the reverse-direction arbiter, where ``final_arr`` is
    # the scramble rather than SOLVED.  The LL tables are a solved-endpoint
    # completion only; letting them rewrite a reverse/free-endpoint run would
    # turn their canonical replay into a false verdict for that caller's target.
    ll_target = None if final_arr is None else np.asarray(final_arr)
    ll_target_is_canonical_solved = bool(
        ll_target is not None and np.array_equal(ll_target, CM.SOLVED)
    )

    # Production LL completion is a terminal replacement transaction.  It is
    # armed only at the state-unique F2L crossing recorded above, competes with
    # the incumbent suffix in the same chronological read currency, and emits
    # only after replaying the app-given init + actual prefix + candidate to the
    # exact canonical solved array.  Every missing/ambiguous input preserves
    # the ordinary scrub output.
    ll_completion_applied = False
    if ll_onset is not None:
        ll_report = dict(report.get("ll_completion") or {})
        if not ll_target_is_canonical_solved:
            ll_report.update(
                status="abstained",
                reason="terminal-target-is-not-canonical-solved",
                applied=False,
            )
        elif tail_ok and np.array_equal(full_end, CM.SOLVED):
            ll_report.update(
                status="not-needed-exact-solved",
                reason="incumbent-already-replays-to-canonical-solved",
                applied=False,
            )
        elif not tail_ok:
            ll_report.update(
                status="abstained",
                reason="incumbent-tail-is-not-canonical",
                applied=False,
            )
        else:
            try:
                onset_frame = int(ll_onset["frame"])
                final_frame = int(f1_of(n_sp - 1))
                ll_move_frames = [
                    int(frame) for frame in gate_streams["move"]
                    if onset_frame < int(frame) <= final_frame
                ]
                ll_rotation_intervals = LLC.bounded_action_intervals(
                    gate_streams["rotation_intervals"],
                    onset_frame,
                    final_frame,
                )
                ll_contested_intervals = LLC.bounded_action_intervals(
                    gate_streams["contested_intervals"],
                    onset_frame,
                    final_frame,
                )
                ll_slots = LLC.build_action_slots(
                    ll_move_frames,
                    ll_rotation_intervals,
                    ll_contested_intervals,
                )
                ll_observations = []
                for si in range(int(ll_onset["span"]) + 1, n_sp):
                    for ri, frame in enumerate(score_read_frames[si]):
                        frame = int(frame)
                        if onset_frame < frame <= final_frame:
                            ll_observations.append(LLC.LLReadObservation(
                                span=int(si), frame=frame,
                                key=("still", int(si), int(ri)),
                            ))
                    for mi, row_mm in enumerate(mm_by_span.get(si, ())):
                        frame = int(row_mm[0])
                        if onset_frame < frame <= final_frame:
                            ll_observations.append(LLC.LLReadObservation(
                                span=int(si), frame=frame,
                                key=("midmotion", int(si), int(mi)),
                            ))

                def ll_score_read(observation, oi, candidate_state):
                    source, si, ri = observation.key
                    segment = (score_read_seg(si, ri, oi)
                               if source == "still"
                               else mm_seg_one(si, ri, oi))
                    if segment is None:
                        return None
                    values = segment.score_states(
                        np.asarray(candidate_state, np.int8)[None, :])
                    if len(values) != 1:
                        return None
                    return float(values[0])

                prefix_n_raw = ll_onset["prefix_windows"]
                if (isinstance(prefix_n_raw, (bool, np.bool_))
                        or not isinstance(prefix_n_raw, (int, np.integer))
                        or not 0 <= int(prefix_n_raw) <= len(out_windows)):
                    raise ValueError("LL verified prefix window count is malformed")
                prefix_n = int(prefix_n_raw)
                incumbent_suffix = tuple(
                    token for window in out_windows[prefix_n:]
                    for token in window["tokens"]
                ) + tuple(tail_moves)
                ll_decision = LLC.select_live_ll_completion(
                    init_state=init_arr,
                    prefix_moves=ll_onset["prefix_moves"],
                    entry_state=ll_onset["state"],
                    incumbent_moves=incumbent_suffix,
                    slots=ll_slots,
                    observations=ll_observations,
                    onset_ois=ll_onset["onset_ois"],
                    orientations=orientations,
                    om_neighbors=stateful_nbrs,
                    perms=perms,
                    score_read=ll_score_read,
                    authoritative_band=float(z_sigma) * float(sigma),
                )
                retained_sources = {
                    hypothesis.source for hypothesis in ll_decision.retained
                }
                retained_emissions = {
                    hypothesis.emission_key
                    for hypothesis in ll_decision.retained
                }
                retained_cases = sorted({
                    (hypothesis.oll_case, hypothesis.pll_case)
                    for hypothesis in ll_decision.retained
                    if hypothesis.source == "table"
                }, key=lambda row_case: (
                    row_case[0] or "", row_case[1] or ""))
                ll_report.update(
                    status=ll_decision.status,
                    reason=ll_decision.reason,
                    applied=False,
                    action_slots=len(ll_slots),
                    move_slots=sum(slot.kind == "move" for slot in ll_slots),
                    rotation_slots=sum(
                        slot.kind == "rotation" for slot in ll_slots),
                    contested_slots=sum(
                        slot.kind == "either" for slot in ll_slots),
                    evidence_rows=len(ll_observations),
                    evidence_rows_input=ll_decision.evidence_rows_input,
                    evidence_rows_used=ll_decision.evidence_rows_used,
                    evidence_rows_dropped_tied=(
                        ll_decision.evidence_rows_dropped_tied),
                    evidence_rows_dropped_interval=(
                        ll_decision.evidence_rows_dropped_interval),
                    dropped_evidence_frames=list(
                        ll_decision.dropped_evidence_frames),
                    authoritative_band=float(z_sigma) * float(sigma),
                    enumerated_candidates=ll_decision.enumerated_candidates,
                    timed_candidates=ll_decision.timed_candidates,
                    scored_hypotheses=ll_decision.scored_hypotheses,
                    retained_hypotheses=len(ll_decision.retained),
                    retained_sources=sorted(retained_sources),
                    retained_emissions=len(retained_emissions),
                    retained_cases=[list(case) for case in retained_cases],
                )
                if (ll_decision.status == "selected"
                        or ll_decision.selected is not None):
                    selected, proposed_windows, proposed_word, replay = (
                        _validated_ll_selected_mutation(
                            decision=ll_decision,
                            action_slots=ll_slots,
                            orientations=orientations,
                            onset_frame=onset_frame,
                            final_frame=final_frame,
                            span_ends=[int(view["meta_f"](si)[1])
                                       for si in range(n_sp)],
                            prefix_windows=list(out_windows[:prefix_n]),
                            expected_prefix_moves=ll_onset["prefix_moves"],
                            init_arr=init_arr,
                            ll_target=ll_target,
                        )
                    )
                    # Mutation follows the exact proposed-word replay above;
                    # no partial zip or stale onset-only word can reach here.
                    out_windows = proposed_windows
                    tail_moves = []
                    tail_ok = True
                    full_end = replay
                    final_ok = bool(np.array_equal(full_end, ll_target))
                    final_dist = 0
                    ll_completion_applied = True
                    ll_report.update(
                        status="selected",
                        reason=ll_decision.reason,
                        applied=True,
                        selected_oll=selected.oll_case,
                        selected_pll=selected.pll_case,
                        selected_literal_moves=list(selected.literal_moves),
                        selected_canonical_moves=list(selected.canonical_moves),
                        selected_action_frames=list(selected.action_frames),
                        selected_move_frames=list(selected.move_frames),
                        selected_move_om_indices=list(
                            selected.move_om_indices),
                        selected_emitted_word=list(proposed_word),
                        endpoint_provenance=(
                            "exact replay of the proposed emitted scrub word "
                            "from app init to canonical solved target"),
                        provided_final_matches=(
                            None if final_arr is None else bool(
                                np.array_equal(
                                    np.asarray(final_arr, np.int8),
                                    CM.SOLVED))),
                    )
            except Exception as exc:  # feature-local: preserve scrub incumbent
                ll_report.update(
                    status="abstained",
                    reason=f"live-seam-error:{type(exc).__name__}:{exc}",
                    applied=False,
                )
        report["ll_completion"] = ll_report
        stream_row(dict(kind="ll_completion", tag=tag, **ll_report))

    # --------------------------------------------------------- emit + report
    new_moves, new_layer, new_oms = [], [], []
    for w in out_windows:
        new_moves += list(w["tokens"])
        new_layer += [w["land"]] * len(w["tokens"])
        new_oms += list(w["oms"])
    new_moves += tail_moves
    if not ll_completion_applied:
        new_layer += list(move_layer[n_solve:])
        new_oms += list(move_oms[n_solve:])

    try:
        reconstruction_checkpoints = _reconstruction_checkpoint_groups(
            out_windows,
            tail_moves,
            final_checkpoint_frame=int(f1_of(n_sp - 1)),
        )
    except (TypeError, ValueError):
        # Diagnostics are fail-closed and must never change the decode word.
        reconstruction_checkpoints = []
    reconstruction_checkpoint_move_count = sum(
        checkpoint["move_count"]
        for checkpoint in reconstruction_checkpoints
    )

    report.update(status="ok", emitted=True, windows=w_idx,
                  committed=counts["committed"],
                  extended=counts["extended"], low_conf=counts["low_conf"],
                  unresolved=counts["unresolved"],
                  final_endpoint_ok=final_ok, final_dist=final_dist,
                  ll_completion_applied=ll_completion_applied,
                  beam_searches=beam_stats["searches"],
                  n_pruned_total=beam_stats["n_pruned"],
                  n_moves_in=len(moves), n_moves_out=len(new_moves),
                  reconstruction_checkpoints=reconstruction_checkpoints,
                  reconstruction_checkpoint_count=len(
                      reconstruction_checkpoints),
                  reconstruction_checkpoint_move_count=(
                      reconstruction_checkpoint_move_count))
    if cluster_fallback:
        report.update(
            cluster_fallback_attempts=cluster_fallback_stats["attempts"],
            cluster_fallback_wall_ms=cluster_fallback_stats["wall_ms"])
    if stateful_cuda_enabled:
        report.update({
            f"stateful_cuda_{key}": value
            for key, value in stateful_cuda_stats.items()
        })
    if _bridge_pin_src:
        # Env unset => key absent => byte-identical report/summary to off.
        report["bridge_pin"] = dict(bridge_pin_stats)
    if _window_audit_src:
        # Env unset => key absent => byte-identical report/summary to off.
        report["window_audit"] = dict(window_audit_stats)

    summary = dict(kind="summary", tag=tag)
    summary.update({k: report.get(k) for k in
                    ("status", "windows", "committed", "extended",
                     "low_conf", "unresolved", "final_endpoint_ok",
                     "final_dist", "ll_completion_applied",
                     "beam_searches", "n_pruned_total",
                     "n_moves_in", "n_moves_out", "midmotion_rows_used",
                     "midmotion_rows_dropped",
                     "reconstruction_checkpoint_count",
                     "reconstruction_checkpoint_move_count")})
    if report.get("ll_completion") is not None:
        summary["ll_completion"] = report["ll_completion"]
    if cluster_fallback:
        summary.update(
            cluster_fallback_attempts=cluster_fallback_stats["attempts"],
            cluster_fallback_wall_ms=cluster_fallback_stats["wall_ms"])
    if stateful_cuda_enabled:
        summary.update({
            f"stateful_cuda_{key}": value
            for key, value in stateful_cuda_stats.items()
        })
    if _bridge_pin_src:
        summary["bridge_pin"] = report["bridge_pin"]
    if _window_audit_src:
        summary["window_audit"] = report["window_audit"]
    if dense_evidence:
        summary["dense_evidence"] = report["dense_evidence"]
    if dense_prefix_active:
        summary["dense_prefix"] = report["dense_prefix"]
    if visual_transition_episodes:
        summary["visual_transition_slots"] = report[
            "visual_transition_slots"]
    if gate_drop_slots_active:
        summary["gate_drop_slots"] = report["gate_drop_slots"]
    summary["intraburst_phase_slots"] = report[
        "intraburst_phase_slots"]
    stream_row(summary)
    if sc_fh is not None:
        try:
            sc_fh.close()
        except OSError:
            pass
    if caud_fh is not None:
        try:
            caud_fh.close()
        except OSError:
            pass
    if tprobe_fh is not None:
        try:
            tprobe_fh.close()
        except OSError:
            pass
    if os.environ.get("CUBED_PERF_PROFILE", "0") == "1":
        for plane_name, label in (
                ("ordinary", "device-beam-profile"),
                ("dense", "dense-device-beam-profile")):
            device_profile = device_beam_runtime.get(plane_name)
            if device_profile is None:
                continue
            plane_plan_count = sum(
                bool(key) and key[0][0] == plane_name
                for key in device_read_run_cache)
            print(
                f"[{label}] "
                f"calls={device_profile.get('checkpoint_calls', 0)} "
                f"resumed_calls="
                f"{device_profile.get('checkpoint_resumed_calls', 0)} "
                f"bursts_skipped="
                f"{device_profile.get('checkpoint_bursts_skipped', 0)} "
                f"bursts_total="
                f"{device_profile.get('checkpoint_bursts_total', 0)} "
                f"seeded_attempts="
                f"{device_profile.get('seeded_attempts', 0)} "
                f"seeded_completed="
                f"{device_profile.get('seeded_completed', 0)} "
                f"seeded_streamed_steps="
                f"{device_profile.get('seeded_streamed_steps', 0)} "
                f"seeded_streamed_preband="
                f"{device_profile.get('seeded_streamed_preband', 0)} "
                f"read_run_plans="
                f"{device_profile.get('read_run_plans_compiled', 0)} "
                f"live_read_run_plans={plane_plan_count}",
                flush=True)
        if device_beam_window_runtime["a_span"] is not None:
            print(
                "[device-beam-cache-profile] "
                f"cursor_evictions="
                f"{device_beam_window_runtime['evictions']} "
                f"checkpoint_families_live="
                f"{len(device_beam_checkpoint_cache)} "
                f"checkpoint_families_peak="
                f"{device_beam_window_runtime['checkpoint_families_peak']} "
                f"read_run_plans_live={len(device_read_run_cache)} "
                f"read_run_plans_peak="
                f"{device_beam_window_runtime['read_run_plans_peak']}",
                flush=True)
    if (dense_beam_checkpoint_runtime["calls"]
            and os.environ.get("CUBED_PERF_PROFILE", "0") == "1"):
        print(
            "[dense-prefix-checkpoint-profile] "
            f"calls={dense_beam_checkpoint_runtime['calls']} "
            f"resumed_calls={dense_beam_checkpoint_runtime['resumed_calls']} "
            f"slots_skipped={dense_beam_checkpoint_runtime['slots_skipped']} "
            f"slots_total={dense_beam_checkpoint_runtime['slots_total']} "
            f"families={len(dense_beam_checkpoint_cache)}",
            flush=True)
    return new_moves, new_layer, new_oms, report
