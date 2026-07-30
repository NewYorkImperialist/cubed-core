"""Evidence-gated live use of the pure OLL/PLL completion enumerator.

This module is intentionally narrower than the ordinary scrub search.  It is
called only after scrub has *verified* the first-two-layer endpoint.  It then
scores complete literal OLL/PLL traces against the exact post-onset action and
read timeline.  The incumbent suffix is scored in the same currency and stays
in the hypothesis set.  Missing timing, weak evidence, orientation ambiguity,
or any replay discrepancy is an abstention.

There are no LL-specific confidence or search knobs.  The only evidence band
accepted by :func:`select_live_ll_completion` is the already-authoritative
scrub commit band supplied by its caller; all work guards reuse
``bridge_search.SCORE_STATE_BOUND``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Sequence

import numpy as np

from analysis import cubemodel as cm
from analysis import segment as sg
from analysis.ll_completion import (
    LLCanonicalTrace,
    LLTrajectory,
    canonicalize_ll_moves,
    canonicalize_ll_trajectory,
    enumerate_ll_completions,
)
import detect.bridge_search as BS

SlotKind = Literal["move", "rotation", "either"]


@dataclass(frozen=True)
class LLActionSlot:
    """One authoritative post-onset physical-action opportunity."""

    kind: SlotKind
    frame: int
    frame_lo: int
    frame_hi: int


@dataclass(frozen=True)
class LLReadObservation:
    """One time-local read in the existing per-span score currency."""

    span: int
    frame: int
    key: object


@dataclass(frozen=True)
class LLLiveHypothesis:
    """One fully timed and orientation-resolved suffix hypothesis."""

    source: Literal["table", "incumbent"]
    literal_moves: tuple[str, ...]
    canonical_moves: tuple[str, ...]
    action_frames: tuple[int, ...]
    move_frames: tuple[int, ...]
    move_om_indices: tuple[int, ...]
    score: float
    oll_case: str | None = None
    pll_case: str | None = None

    @property
    def emission_key(self) -> tuple:
        return (
            self.canonical_moves,
            self.move_frames,
            self.move_om_indices,
        )


@dataclass(frozen=True)
class LLLiveDecision:
    """A selected exact suffix, or an explicit fail-closed abstention."""

    status: Literal["selected", "abstained"]
    reason: str
    enumerated_candidates: int
    timed_candidates: int
    scored_hypotheses: int
    retained: tuple[LLLiveHypothesis, ...]
    selected: LLLiveHypothesis | None = None
    evidence_rows_input: int = 0
    evidence_rows_used: int = 0
    evidence_rows_dropped_tied: int = 0
    evidence_rows_dropped_interval: int = 0
    dropped_evidence_frames: tuple[int, ...] = ()


class _ScaleExceeded(RuntimeError):
    pass


def _face_name(value) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "u": "up", "up": "up",
        "r": "right", "right": "right",
        "f": "front", "front": "front",
        "d": "down", "down": "down",
        "l": "left", "left": "left",
        "b": "back", "back": "back",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(f"invalid orientation face {value!r}") from exc


def _om_key(orientation) -> tuple[str, str]:
    if isinstance(orientation, dict):
        return (_face_name(orientation["up"]),
                _face_name(orientation["front"]))
    return (_face_name(orientation[0]), _face_name(orientation[1]))


def _perm_om_key(perm: np.ndarray) -> tuple[str, str]:
    def source_face(face_index: int) -> str:
        source = int(perm[face_index * 9 + 4]) // 9
        return cm.ALL_FACES[source]

    return source_face(0), source_face(2)


def _orientation_geometry(orientations) -> tuple[tuple[np.ndarray, ...], dict[bytes, int]]:
    """Map detector orientation objects onto the exact cube permutations."""
    by_key = {_perm_om_key(row["perm"]): row["perm"]
              for row in cm.ORIENTATIONS}
    if len(by_key) != 24:
        raise ValueError("cube orientation geometry is incomplete")
    perms = []
    by_perm = {}
    for oi, orientation in enumerate(orientations):
        key = _om_key(orientation)
        perm = by_key.get(key)
        if perm is None or perm.tobytes() in by_perm:
            raise ValueError("supplied orientations are not the 24 cube holds")
        perms.append(perm)
        by_perm[perm.tobytes()] = int(oi)
    if len(perms) != 24:
        raise ValueError("LL completion requires all 24 cube orientations")
    return tuple(perms), by_perm


def _validated_orientation_neighbors(
    om_neighbors: Sequence[Sequence[int]],
    orientation_perms: tuple[np.ndarray, ...],
    oi_by_perm: dict[bytes, int],
) -> tuple[frozenset[int], ...]:
    """Require the complete exact one-gesture orientation graph."""
    if len(om_neighbors) != len(orientation_perms):
        raise ValueError("orientation-neighbor row count changed")
    rotation_perms = tuple(
        cm.PERMS[base + modifier]
        for base in cm.ROTATION_BASES
        for modifier in cm.MODIFIERS
    )
    normalized = []
    for source, (row, source_perm) in enumerate(
            zip(om_neighbors, orientation_perms)):
        if isinstance(row, (str, bytes, dict)):
            raise ValueError("orientation-neighbor row is malformed")
        try:
            raw_destinations = tuple(row)
        except TypeError as exc:
            raise ValueError("orientation-neighbor row is not iterable") from exc
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer))
               for value in raw_destinations):
            raise ValueError("orientation-neighbor destination is not an integer")
        destinations = frozenset(int(value) for value in raw_destinations)
        if len(destinations) != len(raw_destinations):
            raise ValueError("orientation-neighbor row contains duplicates")
        if source in destinations or any(
                value < 0 or value >= len(orientation_perms)
                for value in destinations):
            raise ValueError("orientation-neighbor destination is out of range")
        expected = frozenset(
            oi_by_perm[cm.compose(delta, source_perm).tobytes()]
            for delta in rotation_perms
        ) - {source}
        if destinations != expected:
            raise ValueError("orientation-neighbor geometry is incomplete or changed")
        normalized.append(destinations)
    return tuple(normalized)


def _unique_f2l_color(state: Sequence[int] | np.ndarray) -> str | None:
    arr = np.asarray(state, np.int8)
    hits = [color for color in cm.COLOR_LIST
            if sg.progress(arr, lock_color=color)[0] <= 2]
    return hits[0] if len(hits) == 1 else None


def verified_ll_crossing(previous_state, current_state) -> str | None:
    """Return the unique cross color only for a genuine F2L→LL crossing.

    The caller remains responsible for proving ``current_state`` through its
    calibrated state-margin gate.  This helper only verifies the deterministic
    cube-state predicate and refuses an already-solved or previously-LL state.
    """
    previous = np.asarray(previous_state, np.int8)
    current = np.asarray(current_state, np.int8)
    if previous.shape != (54,) or current.shape != (54,):
        return None
    if cm.is_solved(current):
        return None
    color = _unique_f2l_color(current)
    if color is None or sg.progress(previous, lock_color=color)[0] <= 2:
        return None
    return color


def bounded_action_intervals(
    intervals: Sequence[dict],
    frame_start: int,
    frame_end: int,
) -> tuple[dict, ...]:
    """Return only intervals known to lie wholly inside ``(start, end]``.

    An interval touching the seam but extending outside it has no authoritative
    pre/post ordering.  Reject the complete LL replacement transaction instead
    of assigning that action by its representative frame alone.
    """
    if (isinstance(frame_start, (bool, np.bool_))
            or isinstance(frame_end, (bool, np.bool_))
            or not isinstance(frame_start, (int, np.integer))
            or not isinstance(frame_end, (int, np.integer))
            or int(frame_start) < 0
            or int(frame_end) < int(frame_start)):
        raise ValueError("invalid LL action interval boundary")
    start, end = int(frame_start), int(frame_end)
    out = []
    for raw in intervals:
        if not isinstance(raw, dict):
            raise ValueError("LL action interval is not a mapping")
        values = (raw.get("lo"), raw.get("frame"), raw.get("hi"))
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer))
               for value in values):
            raise ValueError("LL action interval has a non-integer frame")
        lo, frame, hi = (int(value) for value in values)
        if lo < 0 or not lo <= frame <= hi:
            raise ValueError("LL action interval bounds are malformed")
        if hi <= start or lo > end:
            continue
        if not start < lo <= frame <= hi <= end:
            raise ValueError("LL action interval crosses a seam boundary")
        out.append(dict(raw, lo=lo, frame=frame, hi=hi))
    return tuple(out)


def build_action_slots(
    move_frames: Sequence[int],
    rotation_intervals: Sequence[dict],
    contested_intervals: Sequence[dict] = (),
) -> tuple[LLActionSlot, ...]:
    """Build a deterministic typed timeline from the final gate stream."""
    def frame_value(value, label: str) -> int:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(f"{label} is not an integer frame")
        frame = int(value)
        if frame < 0:
            raise ValueError(f"{label} is negative")
        return frame

    moves = [frame_value(frame, "accepted move") for frame in move_frames]
    if len(moves) != len(set(moves)):
        raise ValueError("duplicate accepted move frame")
    contested_rows = [
        (
            frame_value(row["lo"], "contested interval start"),
            frame_value(row["hi"], "contested interval end"),
            frame_value(row["frame"], "contested representative"),
        )
        for row in contested_intervals
    ]
    if len(contested_rows) != len(set(contested_rows)):
        raise ValueError("duplicate contested interval")
    contested = set(contested_rows)
    if any(not lo <= frame <= hi for lo, hi, frame in contested):
        raise ValueError("contested representative lies outside its interval")
    slots = [LLActionSlot("move", frame, frame, frame) for frame in moves]
    move_set = set(moves)
    seen_intervals = set()
    for row in rotation_intervals:
        lo = frame_value(row["lo"], "rotation interval start")
        hi = frame_value(row["hi"], "rotation interval end")
        frame = frame_value(row["frame"], "rotation representative")
        if not lo <= frame <= hi:
            raise ValueError("rotation representative lies outside its interval")
        key = (lo, hi, frame)
        if key in seen_intervals:
            raise ValueError("duplicate rotation interval")
        seen_intervals.add(key)
        if frame in move_set:
            raise ValueError("move and rotation share an authoritative frame")
        kind: SlotKind = "either" if key in contested else "rotation"
        slots.append(LLActionSlot(kind, frame, lo, hi))
    if not contested <= seen_intervals:
        raise ValueError("contested interval is absent from the rotation stream")
    slots.sort(key=lambda slot: (slot.frame, slot.frame_lo, slot.frame_hi,
                                 slot.kind))
    if any(a.frame_hi >= b.frame_lo for a, b in zip(slots, slots[1:])):
        raise ValueError("action intervals overlap or have ambiguous ordering")
    return tuple(slots)


def _step_kind(token: str) -> Literal["move", "rotation"]:
    return ("rotation" if cm.token_base(token) in cm.ROTATION_BASES
            else "move")


def _align_steps(
    step_kinds: tuple[str, ...],
    slots: tuple[LLActionSlot, ...],
) -> tuple[tuple[int, ...], ...]:
    """All exact order-preserving physical-action/slot correspondences."""
    slot_kinds = tuple(slot.kind for slot in slots)

    @lru_cache(maxsize=None)
    def walk(step_i: int, slot_i: int) -> tuple[tuple[int, ...], ...]:
        if step_i == len(step_kinds):
            if any(kind == "move" for kind in slot_kinds[slot_i:]):
                return ()
            return ((),)
        if slot_i == len(slots):
            return ()

        out = []
        slot_kind = slot_kinds[slot_i]
        step_kind = step_kinds[step_i]
        if slot_kind != "move":
            out.extend(walk(step_i, slot_i + 1))
        compatible = (
            (step_kind == "move" and slot_kind in ("move", "either"))
            or (step_kind == "rotation"
                and slot_kind in ("rotation", "either"))
        )
        if compatible:
            for suffix in walk(step_i + 1, slot_i + 1):
                out.append((slot_i,) + suffix)
                if len(out) > int(BS.SCORE_STATE_BOUND):
                    raise _ScaleExceeded("LL physical timing set exceeds score bound")
        return tuple(out)

    return walk(0, 0)


def _canonical_states(entry: np.ndarray, trace: LLCanonicalTrace) -> tuple[np.ndarray, ...]:
    states = [np.asarray(entry, np.int8).copy()]
    state = states[0]
    for step in trace.steps:
        state = cm.apply_seq(state, step.canonical_moves)
        states.append(state)
    return tuple(states)


def _relative_perms(trace: LLCanonicalTrace) -> tuple[np.ndarray, ...]:
    out = [cm.IDENTITY]
    for step in trace.steps:
        out.append(cm.net_perm(step.relative_orientation))
    return tuple(out)


def _score_program(
    *,
    source: Literal["table", "incumbent"],
    trajectory: LLTrajectory | None,
    trace: LLCanonicalTrace,
    action_slot_indices: tuple[int, ...],
    slots: tuple[LLActionSlot, ...],
    entry: np.ndarray,
    observations: tuple[LLReadObservation, ...],
    observation_weights: tuple[float, ...],
    onset_ois: frozenset[int],
    orientation_perms: tuple[np.ndarray, ...],
    oi_by_perm: dict[bytes, int],
    om_neighbors: Sequence[Sequence[int]],
    score_read: Callable[[LLReadObservation, int, np.ndarray], float | None],
    score_cache: dict,
    score_pairs: set,
) -> tuple[LLLiveHypothesis, ...]:
    states = _canonical_states(entry, trace)
    relative = _relative_perms(trace)
    action_by_slot = {slot_i: step_i for step_i, slot_i
                      in enumerate(action_slot_indices)}
    action_frames = tuple(slots[index].frame for index in action_slot_indices)
    if (action_slot_indices
            and not any(obs.frame > slots[action_slot_indices[-1]].frame_hi
                        for obs in observations)):
        return ()

    timeline = ([(slot.frame, 0, "slot", index)
                 for index, slot in enumerate(slots)]
                + [(obs.frame, 1, "read", index)
                   for index, obs in enumerate(observations)])
    values = {(int(oi), ()): 0.0 for oi in sorted(onset_ois)}
    state = states[0]
    action_count = 0
    for _frame, _order, kind, payload in sorted(timeline):
        if kind == "slot":
            slot_i = int(payload)
            step_i = action_by_slot.get(slot_i)
            if step_i is None:
                # An unconsumed dropped/contested slot is an optional re-grip,
                # never an unobserved puzzle turn.
                expanded = {}
                for (oi, move_trace), score in values.items():
                    destinations = {oi} | {int(dst) for dst in om_neighbors[oi]}
                    for dst in destinations:
                        key = (dst, move_trace)
                        old = expanded.get(key)
                        if old is None or score > old:
                            expanded[key] = score
                values = expanded
            else:
                if step_i != action_count:
                    raise ValueError("LL action alignment is not chronological")
                prev_rel, next_rel = relative[step_i], relative[step_i + 1]
                delta = cm.compose(next_rel, np.argsort(prev_rel))
                next_values = {}
                n_moves = len(trace.steps[step_i].canonical_moves)
                for (oi, move_trace), score in values.items():
                    absolute = cm.compose(delta, orientation_perms[oi])
                    try:
                        next_oi = oi_by_perm[absolute.tobytes()]
                    except KeyError as exc:
                        raise ValueError("literal action leaves cube orientation group") from exc
                    next_trace = move_trace + (next_oi,) * n_moves
                    key = (next_oi, next_trace)
                    old = next_values.get(key)
                    if old is None or score > old:
                        next_values[key] = score
                values = next_values
                action_count += 1
                state = states[action_count]
        else:
            read_i = int(payload)
            observation = observations[read_i]
            state_key = state.tobytes()
            score_pair = (read_i, state_key)
            if score_pair not in score_pairs:
                score_pairs.add(score_pair)
                if len(score_pairs) > int(BS.SCORE_STATE_BOUND):
                    raise _ScaleExceeded("LL read/state grid exceeds score bound")
            next_values = {}
            for (oi, move_trace), score in values.items():
                cache_key = (read_i, oi, state_key)
                if cache_key not in score_cache:
                    score_cache[cache_key] = score_read(observation, oi, state)
                read_score = score_cache[cache_key]
                if read_score is None or not np.isfinite(read_score):
                    continue
                next_values[(oi, move_trace)] = (
                    score + observation_weights[read_i] * float(read_score)
                )
            values = next_values
        if len(values) > int(BS.SCORE_STATE_BOUND):
            raise _ScaleExceeded("LL orientation-path frontier exceeds score bound")
        if not values:
            return ()

    if action_count != len(trace.steps):
        raise ValueError("LL timing did not consume every literal action")
    move_frames = tuple(
        slots[slot_i].frame
        for step, slot_i in zip(trace.steps, action_slot_indices)
        for _move in step.canonical_moves
    )
    hypotheses = []
    for (_final_oi, move_oms), score in values.items():
        if len(move_oms) != len(trace.canonical_moves):
            raise ValueError("LL move orientation trace is incomplete")
        hypotheses.append(LLLiveHypothesis(
            source=source,
            literal_moves=trace.literal_moves,
            canonical_moves=trace.canonical_moves,
            action_frames=action_frames,
            move_frames=move_frames,
            move_om_indices=tuple(move_oms),
            score=float(score),
            oll_case=(trajectory.oll.case_id if trajectory is not None else None),
            pll_case=(trajectory.pll.case_id if trajectory is not None else None),
        ))
    return tuple(hypotheses)


def _abstain(reason: str, enumerated: int, timed: int = 0,
             scored: int = 0, retained=(), *, evidence_filter=None
             ) -> LLLiveDecision:
    evidence_filter = evidence_filter or {}
    return LLLiveDecision(
        status="abstained",
        reason=reason,
        enumerated_candidates=int(enumerated),
        timed_candidates=int(timed),
        scored_hypotheses=int(scored),
        retained=tuple(retained),
        evidence_rows_input=int(evidence_filter.get("input", 0)),
        evidence_rows_used=int(evidence_filter.get("used", 0)),
        evidence_rows_dropped_tied=int(evidence_filter.get("dropped_tied", 0)),
        evidence_rows_dropped_interval=int(
            evidence_filter.get("dropped_interval", 0)
        ),
        dropped_evidence_frames=tuple(
            int(frame) for frame in evidence_filter.get("dropped_frames", ())
        ),
    )


def select_live_ll_completion(
    *,
    init_state,
    prefix_moves: Sequence[str],
    entry_state,
    incumbent_moves: Sequence[str],
    slots: Sequence[LLActionSlot],
    observations: Sequence[LLReadObservation],
    onset_ois: Sequence[int],
    orientations,
    om_neighbors: Sequence[Sequence[int]],
    perms,
    score_read: Callable[[LLReadObservation, int, np.ndarray], float | None],
    authoritative_band: float,
) -> LLLiveDecision:
    """Select one fully evidenced LL suffix, otherwise preserve the incumbent.

    ``authoritative_band`` must be the caller's existing calibrated scrub
    commit band.  It is used only to retain indistinguishable complete
    hypotheses; this module defines no independent confidence threshold.
    """
    init = np.asarray(init_state, np.int8)
    entry = np.asarray(entry_state, np.int8)
    if init.shape != (54,) or entry.shape != (54,):
        return _abstain("invalid-state-shape", 0)
    if not np.isfinite(authoritative_band) or authoritative_band < 0:
        return _abstain("invalid-authoritative-band", 0)
    try:
        prefix = tuple(str(move) for move in prefix_moves)
        incumbent = tuple(str(move) for move in incumbent_moves)
        if any(move not in BS.MOVES for move in prefix + incumbent):
            return _abstain("noncanonical-prefix-or-incumbent", 0)
        replayed_entry = init.copy()
        for move in prefix:
            replayed_entry = replayed_entry[perms[BS.MOVES.index(move)]]
        if not np.array_equal(replayed_entry, entry):
            return _abstain("prefix-entry-replay-mismatch", 0)

        result = enumerate_ll_completions(entry)
        enumerated = len(result.candidates)
        if not result.candidates:
            return _abstain(result.abstain_reason or "no-table-candidate", 0)
        slot_rows = tuple(slots)
        input_reads = tuple(sorted(observations,
                                   key=lambda row: (row.frame, row.span)))
        if not input_reads:
            return _abstain("no-post-onset-reads", enumerated)
        if not slot_rows:
            return _abstain("no-post-onset-action-slots", enumerated)
        if any(
                not isinstance(slot, LLActionSlot)
                or slot.kind not in ("move", "rotation", "either")
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer))
                       for value in (slot.frame, slot.frame_lo, slot.frame_hi))
                or slot.frame_lo < 0
                or not slot.frame_lo <= slot.frame <= slot.frame_hi
                for slot in slot_rows):
            return _abstain("invalid-action-slots", enumerated)
        if any(a.frame_hi >= b.frame_lo for a, b in zip(slot_rows,
                                                        slot_rows[1:])):
            return _abstain("nonchronological-action-slots", enumerated)
        tied_frames = {int(slot.frame) for slot in slot_rows}
        uncertain_intervals = tuple(
            (int(slot.frame_lo), int(slot.frame_hi))
            for slot in slot_rows
            if slot.frame_lo != slot.frame_hi
        )
        reads = []
        dropped_tied = []
        dropped_interval = []
        for observation in input_reads:
            frame = int(observation.frame)
            if frame in tied_frames:
                dropped_tied.append(frame)
            elif any(lo <= frame <= hi for lo, hi in uncertain_intervals):
                dropped_interval.append(frame)
            else:
                reads.append(observation)
        reads = tuple(reads)
        evidence_filter = {
            "input": len(input_reads),
            "used": len(reads),
            "dropped_tied": len(dropped_tied),
            "dropped_interval": len(dropped_interval),
            "dropped_frames": tuple(sorted(dropped_tied + dropped_interval)),
        }
        if not reads:
            return _abstain(
                "no-unambiguous-post-onset-reads",
                enumerated,
                evidence_filter=evidence_filter,
            )

        onset = frozenset(int(oi) for oi in onset_ois)
        if not onset or any(oi < 0 or oi >= len(orientations) for oi in onset):
            return _abstain(
                "invalid-onset-orientation-band",
                enumerated,
                evidence_filter=evidence_filter,
            )
        orientation_perms, oi_by_perm = _orientation_geometry(orientations)
        try:
            normalized_neighbors = _validated_orientation_neighbors(
                om_neighbors, orientation_perms, oi_by_perm)
        except (KeyError, TypeError, ValueError):
            return _abstain(
                "invalid-orientation-neighbors",
                enumerated,
                evidence_filter=evidence_filter,
            )

        counts = {}
        for observation in reads:
            counts[observation.span] = counts.get(observation.span, 0) + 1
        weights = tuple(1.0 / counts[row.span] for row in reads)

        programs: list[tuple[str, LLTrajectory | None, LLCanonicalTrace]] = []
        for trajectory in result.candidates:
            trace = canonicalize_ll_trajectory(trajectory)
            # Production accepts the exact canonical solved state, not merely
            # orientation-invariant monochromatic faces or an inferred final.
            candidate_end = cm.apply_seq(entry, trace.canonical_moves)
            full_end = cm.apply_seq(init, prefix + trace.canonical_moves)
            detector_end = init.copy()
            for move in prefix + trace.canonical_moves:
                detector_end = detector_end[perms[BS.MOVES.index(move)]]
            if (np.array_equal(candidate_end, cm.SOLVED)
                    and np.array_equal(full_end, cm.SOLVED)
                    and np.array_equal(detector_end, cm.SOLVED)):
                programs.append(("table", trajectory, trace))
        if not programs:
            return _abstain(
                "no-canonical-replay-to-solved",
                enumerated,
                evidence_filter=evidence_filter,
            )

        incumbent_trace = canonicalize_ll_moves(incumbent)
        programs.append(("incumbent", None, incumbent_trace))
        score_cache = {}
        score_pairs = set()
        hypotheses = []
        timed = 0
        incumbent_scored = False
        for source, trajectory, trace in programs:
            step_kinds = tuple(_step_kind(step.literal_move)
                               for step in trace.steps)
            alignments = _align_steps(step_kinds, slot_rows)
            timed += len(alignments)
            if timed > int(BS.SCORE_STATE_BOUND):
                raise _ScaleExceeded("LL timed candidate set exceeds score bound")
            for alignment in alignments:
                rows = _score_program(
                    source=source,
                    trajectory=trajectory,
                    trace=trace,
                    action_slot_indices=alignment,
                    slots=slot_rows,
                    entry=entry,
                    observations=reads,
                    observation_weights=weights,
                    onset_ois=onset,
                    orientation_perms=orientation_perms,
                    oi_by_perm=oi_by_perm,
                    om_neighbors=normalized_neighbors,
                    score_read=score_read,
                    score_cache=score_cache,
                    score_pairs=score_pairs,
                )
                if rows and source == "incumbent":
                    incumbent_scored = True
                hypotheses.extend(rows)
                if len(hypotheses) > int(BS.SCORE_STATE_BOUND):
                    raise _ScaleExceeded("LL scored hypothesis set exceeds score bound")
        if not incumbent_scored:
            return _abstain("incumbent-not-comparable", enumerated, timed,
                            len(hypotheses), evidence_filter=evidence_filter)
        if not hypotheses:
            return _abstain(
                "no-scoreable-timed-hypothesis",
                enumerated,
                timed,
                evidence_filter=evidence_filter,
            )

        # Exact duplicate emissions can arise from different table notation.
        # Keep the best score per full production emission while retaining the
        # literal representative for audit.
        best_by_identity = {}
        for hypothesis in hypotheses:
            identity = (hypothesis.source, hypothesis.literal_moves,
                        hypothesis.emission_key)
            incumbent_h = best_by_identity.get(identity)
            if (incumbent_h is None
                    or (hypothesis.score, tuple(hypothesis.literal_moves))
                    > (incumbent_h.score, tuple(incumbent_h.literal_moves))):
                best_by_identity[identity] = hypothesis
        ranked = sorted(
            best_by_identity.values(),
            key=lambda row: (-row.score, row.source, row.literal_moves,
                             row.emission_key),
        )
        top = ranked[0].score
        retained = tuple(row for row in ranked
                         if top - row.score <= float(authoritative_band))
        emission_keys = {row.emission_key for row in retained}
        sources = {row.source for row in retained}
        if "incumbent" in sources:
            return _abstain("incumbent-within-evidence-band", enumerated,
                            timed, len(ranked), retained,
                            evidence_filter=evidence_filter)
        if len(emission_keys) != 1:
            return _abstain("multiple-emissions-within-evidence-band",
                            enumerated, timed, len(ranked), retained,
                            evidence_filter=evidence_filter)
        selected = min(retained, key=lambda row: (
            row.literal_moves, row.oll_case or "", row.pll_case or ""))
        if selected.source != "table":
            return _abstain("non-table-winner", enumerated, timed,
                            len(ranked), retained,
                            evidence_filter=evidence_filter)
        # Re-run the complete canonical replay at the final mutation boundary.
        final = cm.apply_seq(init, prefix + selected.canonical_moves)
        detector_final = init.copy()
        for move in prefix + selected.canonical_moves:
            detector_final = detector_final[perms[BS.MOVES.index(move)]]
        if (not np.array_equal(final, cm.SOLVED)
                or not np.array_equal(detector_final, cm.SOLVED)):
            return _abstain("selected-full-replay-not-solved", enumerated,
                            timed, len(ranked), retained,
                            evidence_filter=evidence_filter)
        return LLLiveDecision(
            status="selected",
            reason="unique-evidenced-canonical-solve",
            enumerated_candidates=enumerated,
            timed_candidates=timed,
            scored_hypotheses=len(ranked),
            retained=retained,
            selected=selected,
            evidence_rows_input=evidence_filter["input"],
            evidence_rows_used=evidence_filter["used"],
            evidence_rows_dropped_tied=evidence_filter["dropped_tied"],
            evidence_rows_dropped_interval=evidence_filter[
                "dropped_interval"
            ],
            dropped_evidence_frames=evidence_filter["dropped_frames"],
        )
    except _ScaleExceeded as exc:
        return _abstain(f"scale-guard:{exc}", locals().get("enumerated", 0),
                        locals().get("timed", 0),
                        len(locals().get("hypotheses", ())),
                        evidence_filter=locals().get("evidence_filter"))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return _abstain(f"invalid-live-evidence:{type(exc).__name__}:{exc}",
                        locals().get("enumerated", 0),
                        locals().get("timed", 0),
                        len(locals().get("hypotheses", ())),
                        evidence_filter=locals().get("evidence_filter"))
