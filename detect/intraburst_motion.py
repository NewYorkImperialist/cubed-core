"""Certify optional motion phases hidden by event-period merging.

The production/eval event detector already performs two structural steps:

1. threshold a median-smoothed, dense cube-motion trace into active runs;
2. merge active runs separated by less than the existing minimum-rest duration,
   then retain one peak frame for the whole merged period.

Step 2 intentionally loses the pre-merge run boundaries.  This module recovers
only those already-computed structural phases.  It does not classify a move,
alter the authoritative event list, or choose a cube action.  A certified phase
is merely an interval that a later typed scrub slot may interpret as
``SKIP | 18 SINGLE``.

Certification is deliberately fail-closed:

* the motion trace must be finite and consecutive;
* a merged period must contain exactly one authoritative event;
* that event must belong to exactly one active phase;
* the event-owning phase is omitted, preserving the original unsplit event;
* an over-bound output remains non-actuating, but its complete typed evidence
  is retained so a later, narrower authority rebind can reapply the same bound.

There is no solve/tag threshold or learned input here.  The temporal constants
are the historical five-sample median and five-frame minimum-rest at 120 fps,
expressed in seconds exactly as in :mod:`detect.trellis_tracker`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Mapping

import numpy as np

from detect import bridge_search as BS


_MOTION_SMOOTH_RADIUS_S = 2.0 / 120.0
_MOTION_MIN_REST_S = 5.0 / 120.0

# Reuse scrub's existing absolute structural envelope (2 * MITM depth), not a
# solve-derived proposal count.  The later typed beam owns the exact, stricter
# per-step OM-aware fanout guard against ``SCORE_STATE_BOUND``; this producer
# bound prevents an unbounded number of optional transactions before that
# frontier-dependent check is possible.
OPTIONAL_ACTION_BRANCHES = 1 + len(BS.MOVES)  # SKIP + 18 SINGLE
MAX_PHASE_SLOTS = 2 * BS.DEPTH_CAP
MAX_PHASE_ACTION_ROWS = MAX_PHASE_SLOTS * OPTIONAL_ACTION_BRANCHES
assert MAX_PHASE_ACTION_ROWS <= BS.SCORE_STATE_BOUND


@dataclass(frozen=True)
class IntraBurstPhaseSlot:
    """One certified non-parent phase inside a merged motion period."""

    parent_frame: int
    frame_lo: int
    frame_hi: int
    period_lo: int
    period_hi: int
    phase_index: int
    phase_count: int


@dataclass(frozen=True)
class UnownedMotionPeriod:
    """One dense active period with no authoritative event parent.

    This is producer evidence only.  It never enters the event list or phase
    actuator implicitly; a separately scoped record-only microscope may bind
    an exact period/peak tuple and test it as an alternate burst stream.
    """

    frame_lo: int
    frame_hi: int
    peak_frame: int
    phase_count: int


@dataclass(frozen=True)
class IntraBurstAuthorityRebind:
    """One fail-closed projection of producer authority onto a consumer."""

    provenance: str
    prior_authoritative_events: tuple[int, ...]
    authoritative_events: tuple[int, ...]
    removed_parent_events: tuple[int, ...]
    removed_slot_count: int
    frame_bounds: tuple[int, int] | None = None


@dataclass(frozen=True)
class IntraBurstPhaseAudit:
    """Auditable, immutable result of one phase-certification pass."""

    status: str
    reason: str | None
    slots: tuple[IntraBurstPhaseSlot, ...]
    authoritative_events: tuple[int, ...]
    derived_event_frames: tuple[int, ...]
    active_run_count: int
    merged_period_count: int
    certified_period_count: int
    ambiguous_periods: tuple[tuple[int, int, str], ...]
    # Dense producer periods that contain zero authoritative events.  These are
    # non-actuating receipts; keeping them typed prevents a DEV microscope from
    # inventing an interval around a bare derived peak after the fact.
    unowned_periods: tuple[UnownedMotionPeriod, ...] = ()
    # Complete typed evidence retained only while ``status == "over-bound"``.
    # Consumers receive ``slots`` only; a narrower authority rebind must filter
    # and revalidate this tuple before any row can become an actuator.
    provisional_slots: tuple[IntraBurstPhaseSlot, ...] = ()
    # This is the immutable, producer-coordinate authority.  The active
    # ``authoritative_events`` may subsequently be narrowed by the final move
    # gate or reverse tracker's read domain, but the source identity remains
    # visible in every serialized receipt.
    producer_authoritative_events: tuple[int, ...] = ()
    authority_rebinds: tuple[IntraBurstAuthorityRebind, ...] = ()
    max_phase_slots: int = MAX_PHASE_SLOTS
    optional_action_branches: int = OPTIONAL_ACTION_BRANCHES
    max_phase_action_rows: int = MAX_PHASE_ACTION_ROWS
    score_state_bound: int = BS.SCORE_STATE_BOUND


def require_exact_integer(value, field):
    """Return an integer scalar without accepting lossy/coercive lookalikes."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{field} must be an exact non-bool integer")
    return int(value)


def _exact_integer_tuple(values, field):
    return tuple(
        require_exact_integer(value, field) for value in values)


def intraburst_phase_audit_payload(audit):
    """Return one JSON-safe producer receipt without changing its semantics."""
    if isinstance(audit, IntraBurstPhaseAudit):
        payload = asdict(audit)
    elif isinstance(audit, Mapping):
        payload = dict(audit)
    elif is_dataclass(audit):
        payload = asdict(audit)
    else:
        return None

    def json_value(value):
        if is_dataclass(value):
            return json_value(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return json_value(payload)


def intraburst_phase_slot_from_value(value):
    """Normalize one typed/mapping slot using exact integer validation."""
    if isinstance(value, IntraBurstPhaseSlot):
        source = {
            field: getattr(value, field)
            for field in (
                "parent_frame", "frame_lo", "frame_hi", "period_lo",
                "period_hi", "phase_index", "phase_count")
        }
    elif isinstance(value, Mapping):
        source = value
    else:
        raise TypeError("phase slot is not a mapping or IntraBurstPhaseSlot")
    fields = (
        "parent_frame", "frame_lo", "frame_hi", "period_lo",
        "period_hi", "phase_index", "phase_count")
    return IntraBurstPhaseSlot(**{
        field: require_exact_integer(source[field], f"phase slot {field}")
        for field in fields
    })


def unowned_motion_period_from_value(value):
    """Normalize one producer-only unowned-period receipt exactly."""
    if isinstance(value, UnownedMotionPeriod):
        source = {
            field: getattr(value, field)
            for field in ("frame_lo", "frame_hi", "peak_frame", "phase_count")
        }
    elif isinstance(value, Mapping):
        source = value
    else:
        raise TypeError(
            "unowned motion period is not a mapping or UnownedMotionPeriod")
    fields = ("frame_lo", "frame_hi", "peak_frame", "phase_count")
    row = UnownedMotionPeriod(**{
        field: require_exact_integer(
            source[field], f"unowned motion period {field}")
        for field in fields
    })
    if not (row.frame_lo <= row.peak_frame <= row.frame_hi):
        raise ValueError("unowned motion peak lies outside its period")
    if row.phase_count < 1:
        raise ValueError("unowned motion phase_count must be positive")
    return row


def validate_unowned_motion_period(audit, *, frame_lo, frame_hi, peak_frame):
    """Bind one exact non-actuating producer period for a DEV microscope.

    Returns ``(period, None)`` only when the typed producer receipt contains
    exactly one matching period, its peak is also in the producer's derived
    event stream, and active authority owns no event inside it.  The typed
    period itself is the immutable producer's zero-parent receipt; unlike
    ``producer_authoritative_events`` it is remapped into consumer coordinates.
    Every failure is fail-closed as ``(None, reason)``.
    """
    payload = intraburst_phase_audit_payload(audit)
    if payload is None:
        return None, "producer-audit-unavailable"
    if str(payload.get("status")) == "abstained":
        return None, "producer-audit-abstained"
    try:
        target = UnownedMotionPeriod(
            frame_lo=require_exact_integer(frame_lo, "probe frame_lo"),
            frame_hi=require_exact_integer(frame_hi, "probe frame_hi"),
            peak_frame=require_exact_integer(peak_frame, "probe peak_frame"),
            phase_count=1,
        )
        if not (target.frame_lo <= target.peak_frame <= target.frame_hi):
            return None, "probe-peak-outside-period"
        rows = tuple(
            unowned_motion_period_from_value(row)
            for row in payload.get("unowned_periods", ()))
        matches = tuple(
            row for row in rows
            if (row.frame_lo, row.frame_hi, row.peak_frame)
            == (target.frame_lo, target.frame_hi, target.peak_frame))
        derived = _exact_integer_tuple(
            payload.get("derived_event_frames", ()), "derived event")
        active = _exact_integer_tuple(
            payload.get("authoritative_events", ()), "authoritative event")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return None, f"malformed-producer-receipt:{type(exc).__name__}:{exc}"
    if len(rows) != len(set(rows)):
        return None, "producer-unowned-periods-not-unique"
    if len(matches) != 1:
        return None, f"exact-producer-period-match-count:{len(matches)}"
    match = matches[0]
    if match.peak_frame not in derived:
        return None, "producer-peak-absent-from-derived-events"
    if any(match.frame_lo <= frame <= match.frame_hi for frame in active):
        return None, "active-authority-owns-probe-period"
    return match, None


def validate_unowned_motion_periods_for_window(
    audit,
    *,
    frame_start_exclusive,
    frame_end_inclusive,
):
    """Select every eligible producer period owned by one scrub window.

    The producer receipt is validated as one transaction before containment is
    considered: all period/frame fields must be exact integers, period rows and
    authority rows must be unique, producer periods must not overlap, every
    producer peak must occur in the derived event stream, and no active
    authority may lie inside a period advertised as unowned.  A period that
    intersects but is not fully contained by ``(start, end]`` rejects the
    selection instead of being silently clipped or assigned to a convenient
    subset.

    Structural integrity is checked for the complete producer receipt.  The
    behavior-only ``phase_count == 1`` rule is then applied to every period
    owned by this window; well-formed periods wholly outside the window do not
    affect its eligibility.  Success returns typed rows in canonical
    ``(frame_lo, frame_hi, peak_frame)`` order.  Every failure is fail-closed as
    ``((), reason)``.
    """
    payload = intraburst_phase_audit_payload(audit)
    if payload is None:
        return (), "producer-audit-unavailable"
    if str(payload.get("status")) == "abstained":
        return (), "producer-audit-abstained"
    try:
        window_start = require_exact_integer(
            frame_start_exclusive, "window frame start")
        window_end = require_exact_integer(
            frame_end_inclusive, "window frame end")
        if window_start >= window_end:
            return (), "window-frame-bounds-empty-or-reversed"
        rows = tuple(
            unowned_motion_period_from_value(row)
            for row in payload.get("unowned_periods", ()))
        derived = _exact_integer_tuple(
            payload.get("derived_event_frames", ()), "derived event")
        active = _exact_integer_tuple(
            payload.get("authoritative_events", ()), "authoritative event")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return (), f"malformed-producer-receipt:{type(exc).__name__}:{exc}"

    if len(rows) != len(set(rows)):
        return (), "producer-unowned-periods-not-unique"
    if len(derived) != len(set(derived)):
        return (), "producer-derived-events-not-unique"
    if len(active) != len(set(active)):
        return (), "producer-authoritative-events-not-unique"

    canonical = tuple(sorted(
        rows,
        key=lambda row: (row.frame_lo, row.frame_hi, row.peak_frame),
    ))
    for left, right in zip(canonical, canonical[1:]):
        if right.frame_lo <= left.frame_hi:
            return (), "producer-unowned-periods-overlap"
    derived_set = set(derived)
    for row in canonical:
        if row.peak_frame not in derived_set:
            return (), "producer-peak-absent-from-derived-events"
        if any(row.frame_lo <= frame <= row.frame_hi for frame in active):
            return (), "active-authority-owns-unowned-period"

    selected = []
    for row in canonical:
        contained = (
            window_start < row.frame_lo
            and row.frame_hi <= window_end
        )
        intersects = (
            window_start < row.frame_hi
            and row.frame_lo <= window_end
        )
        if intersects and not contained:
            return (), "producer-unowned-period-straddles-window"
        if not contained:
            continue
        if row.phase_count != 1:
            return (), "producer-unowned-period-phase-count-not-one"
        selected.append(row)
    return tuple(selected), None


def validate_intraburst_phase_transaction(slots, audit):
    """Admit only a complete producer transaction; otherwise return no slots.

    The producer audit and the separately plumbed slot tuple must agree exactly.
    This prevents a stale or partially serialized slot list from becoming an
    actuator.  All failures are represented in the returned audit payload and
    preserve the empty (baseline) lattice.
    """
    payload = intraburst_phase_audit_payload(audit)
    receipt = {
        "producer": payload,
        "enabled": False,
        "admission_status": "disabled",
        "admission_reason": None,
        "slot_count": 0,
        "optional_action_rows": 0,
        "hard_event_count_policy": "unchanged",
        "count_interval_policy": "unchanged",
    }

    def reject(reason):
        receipt["admission_reason"] = str(reason)
        return (), receipt

    if payload is None:
        return reject("producer audit unavailable")
    if str(payload.get("status")) != "ok":
        return reject(f"producer status is {payload.get('status')!r}")
    if payload.get("provisional_slots"):
        return reject("ok producer audit contains provisional slots")
    try:
        audit_slots = tuple(intraburst_phase_slot_from_value(row)
                            for row in payload.get("slots", ()))
        supplied_slots = tuple(
            intraburst_phase_slot_from_value(row) for row in (slots or ()))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return reject(f"malformed phase slot: {type(exc).__name__}: {exc}")
    if not audit_slots:
        return reject("ok producer audit contains no slots")
    if supplied_slots != audit_slots:
        return reject("producer audit and supplied slots do not match")
    if len(audit_slots) > MAX_PHASE_SLOTS:
        return reject(
            f"phase slot count {len(audit_slots)} exceeds {MAX_PHASE_SLOTS}")
    if len(audit_slots) * OPTIONAL_ACTION_BRANCHES > BS.SCORE_STATE_BOUND:
        return reject("phase action rows exceed shared score-state bound")
    try:
        receipt_max_slots = require_exact_integer(
            payload.get("max_phase_slots"), "max_phase_slots")
        receipt_branches = require_exact_integer(
            payload.get("optional_action_branches"),
            "optional_action_branches")
        receipt_action_rows = require_exact_integer(
            payload.get("max_phase_action_rows"),
            "max_phase_action_rows")
        receipt_state_bound = require_exact_integer(
            payload.get("score_state_bound"), "score_state_bound")
    except (TypeError, ValueError, OverflowError) as exc:
        return reject(f"malformed producer bound receipt: {exc}")
    if receipt_max_slots != MAX_PHASE_SLOTS:
        return reject("producer phase-slot bound receipt does not match runtime")
    if receipt_branches != OPTIONAL_ACTION_BRANCHES:
        return reject("producer action alphabet receipt does not match runtime")
    if receipt_action_rows != MAX_PHASE_ACTION_ROWS:
        return reject("producer phase-action-row receipt does not match runtime")
    if receipt_state_bound != BS.SCORE_STATE_BOUND:
        return reject("producer score-state bound receipt does not match runtime")
    try:
        authoritative_rows = tuple(
            require_exact_integer(frame, "authoritative event")
            for frame in payload.get("authoritative_events", ()))
        authoritative = set(authoritative_rows)
        producer_source = payload.get("producer_authoritative_events", ())
        producer_rows = tuple(
            require_exact_integer(frame, "producer authoritative event")
            for frame in (producer_source or authoritative_rows))
        derived_rows = tuple(
            require_exact_integer(frame, "derived event")
            for frame in payload.get("derived_event_frames", ()))
    except (TypeError, ValueError, OverflowError) as exc:
        return reject(f"malformed authoritative events: {exc}")
    if len(authoritative) != len(authoritative_rows):
        return reject("authoritative event receipt is not unique")
    if len(producer_rows) != len(set(producer_rows)):
        return reject("producer authoritative event receipt is not unique")
    if len(derived_rows) != len(set(derived_rows)):
        return reject("derived event receipt is not unique")
    try:
        for row in payload.get("authority_rebinds", ()):
            if not isinstance(row, Mapping):
                raise TypeError("authority rebind receipt is not a mapping")
            for field in (
                    "prior_authoritative_events", "authoritative_events",
                    "removed_parent_events"):
                _exact_integer_tuple(
                    row.get(field, ()), f"authority rebind {field}")
            bounds = row.get("frame_bounds")
            if bounds is not None:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise TypeError("authority rebind frame bounds are malformed")
                _exact_integer_tuple(bounds, "authority rebind frame bound")
    except (TypeError, ValueError, OverflowError) as exc:
        return reject(f"malformed authority rebind receipt: {exc}")
    ordered = sorted(audit_slots, key=lambda slot: (
        slot.frame_hi, slot.frame_lo, slot.parent_frame, slot.phase_index))
    if tuple(ordered) != audit_slots:
        return reject("phase slots are not in canonical chronological order")
    seen = set()
    previous_hi = None
    for slot in audit_slots:
        if slot in seen:
            return reject("phase slot receipt contains duplicates")
        seen.add(slot)
        if not (slot.period_lo <= slot.frame_lo <= slot.frame_hi
                <= slot.period_hi):
            return reject("phase interval is outside its merged period")
        if not (slot.period_lo <= slot.parent_frame <= slot.period_hi):
            return reject("parent event is outside its merged period")
        if slot.frame_lo <= slot.parent_frame <= slot.frame_hi:
            return reject("optional phase contains its authoritative parent")
        if slot.parent_frame not in authoritative:
            return reject("phase parent is absent from authoritative events")
        if any(slot.frame_lo <= frame <= slot.frame_hi
               for frame in authoritative):
            return reject("optional phase contains an authoritative event")
        if slot.phase_count < 2 or not (0 <= slot.phase_index < slot.phase_count):
            return reject("invalid phase index/count provenance")
        if previous_hi is not None and slot.frame_lo <= previous_hi:
            return reject("certified optional phase intervals overlap")
        previous_hi = slot.frame_hi
    receipt.update(
        enabled=True,
        admission_status="admitted",
        admission_reason=None,
        slot_count=len(audit_slots),
        optional_action_rows=len(audit_slots) * OPTIONAL_ACTION_BRANCHES,
    )
    return audit_slots, receipt


def rebind_intraburst_phase_authority(
    audit,
    authoritative_events,
    *,
    provenance,
    frame_bounds=None,
):
    """Project a producer audit onto one exact downstream event authority.

    A final move gate may remove producer events, and a reverse tracker may
    consume only the forward events inside its frame domain.  Those consumers
    must not retain optional phases whose parent they no longer recognize.
    This operation therefore narrows both the active authority and its slots as
    one transaction.  It never introduces an event absent from the immediately
    preceding authority and it never rewrites
    ``producer_authoritative_events``.
    """
    if not isinstance(audit, IntraBurstPhaseAudit):
        return _abstain("intraburst authority rebind requires a typed audit")

    try:
        producer_events = _exact_integer_tuple(
            audit.producer_authoritative_events
            or audit.authoritative_events,
            "producer authoritative event")
        prior_events = _exact_integer_tuple(
            audit.authoritative_events, "prior authoritative event")
    except (TypeError, ValueError, OverflowError) as exc:
        return replace(
            audit,
            status="abstained",
            reason=f"intraburst authority rebind failed: {exc}",
            slots=(),
            provisional_slots=(),
            unowned_periods=(),
            authoritative_events=(),
            certified_period_count=0,
        )
    target_events = ()
    normalized_bounds = None

    def failed(reason):
        record = IntraBurstAuthorityRebind(
            provenance=str(provenance),
            prior_authoritative_events=prior_events,
            authoritative_events=target_events,
            removed_parent_events=tuple(sorted(
                set(prior_events) - set(target_events))),
            removed_slot_count=len(
                audit.slots or audit.provisional_slots),
            frame_bounds=normalized_bounds,
        )
        return replace(
            audit,
            status="abstained",
            reason=f"intraburst authority rebind failed: {reason}",
            slots=(),
            provisional_slots=(),
            unowned_periods=(),
            authoritative_events=target_events,
            certified_period_count=0,
            producer_authoritative_events=producer_events,
            authority_rebinds=audit.authority_rebinds + (record,),
        )

    try:
        raw_events = _exact_integer_tuple(
            authoritative_events, "rebound authoritative event")
    except (TypeError, ValueError, OverflowError) as exc:
        return failed(f"malformed authoritative events: {exc}")
    if len(raw_events) != len(set(raw_events)):
        return failed("authoritative event frames are not unique")
    target_events = tuple(sorted(raw_events))
    if frame_bounds is not None:
        try:
            if len(frame_bounds) != 2:
                raise ValueError("frame bounds must contain two endpoints")
            lo, hi = (
                require_exact_integer(frame_bounds[0], "frame bound lo"),
                require_exact_integer(frame_bounds[1], "frame bound hi"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return failed(f"malformed frame bounds: {exc}")
        if lo > hi:
            return failed("frame bounds are reversed")
        normalized_bounds = (lo, hi)
        if any(frame < lo or frame > hi for frame in target_events):
            return failed("rebound authority lies outside frame bounds")
    if not set(target_events).issubset(prior_events):
        return failed("rebound authority introduces an unowned event")
    if audit.status == "ok" and audit.provisional_slots:
        return failed("ok source contains provisional phase slots")
    if audit.status == "over-bound":
        if audit.slots:
            return failed("over-bound source contains actuator phase slots")
        if len(audit.provisional_slots) <= MAX_PHASE_SLOTS:
            return failed("over-bound source does not exceed structural bound")

    try:
        raw_source_slots = (
            audit.provisional_slots
            if audit.status == "over-bound" else audit.slots)
        source_slots = tuple(
            intraburst_phase_slot_from_value(slot)
            for slot in raw_source_slots)
        source_unowned = tuple(
            unowned_motion_period_from_value(period)
            for period in audit.unowned_periods)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return failed(f"malformed source slot: {exc}")
    target_set = set(target_events)
    retained_slots = tuple(
        slot for slot in source_slots
        if slot.parent_frame in target_set
        and (normalized_bounds is None
             or (normalized_bounds[0] <= slot.period_lo
                 and slot.period_hi <= normalized_bounds[1]))
    )
    retained_periods = {
        (slot.parent_frame, slot.period_lo, slot.period_hi)
        for slot in retained_slots
    }
    retained_unowned = tuple(
        period for period in source_unowned
        if (normalized_bounds is None
            or (normalized_bounds[0] <= period.frame_lo
                and period.frame_hi <= normalized_bounds[1])))
    record = IntraBurstAuthorityRebind(
        provenance=str(provenance),
        prior_authoritative_events=prior_events,
        authoritative_events=target_events,
        removed_parent_events=tuple(sorted(set(prior_events) - target_set)),
        removed_slot_count=len(source_slots) - len(retained_slots),
        frame_bounds=normalized_bounds,
    )
    source_rebindable = audit.status in {"ok", "over-bound"}
    if not source_rebindable:
        rebound_status = audit.status
        rebound_reason = audit.reason
        rebound_slots = ()
        rebound_provisional = ()
        rebound_period_count = 0
    elif not retained_slots:
        rebound_status = "no-certified-slots"
        rebound_reason = "authority rebind retained no certified phase slots"
        rebound_slots = ()
        rebound_provisional = ()
        rebound_period_count = 0
    elif len(retained_slots) > MAX_PHASE_SLOTS:
        rebound_status = "over-bound"
        rebound_reason = (
            f"certified phase slots {len(retained_slots)} exceed existing "
            f"structural bound {MAX_PHASE_SLOTS} after authority rebind")
        rebound_slots = ()
        rebound_provisional = retained_slots
        rebound_period_count = len(retained_periods)
    else:
        rebound_status = "ok"
        rebound_reason = None
        rebound_slots = retained_slots
        rebound_provisional = ()
        rebound_period_count = len(retained_periods)
    rebound = replace(
        audit,
        status=rebound_status,
        reason=rebound_reason,
        slots=rebound_slots,
        provisional_slots=rebound_provisional,
        unowned_periods=retained_unowned,
        authoritative_events=target_events,
        certified_period_count=rebound_period_count,
        producer_authoritative_events=producer_events,
        authority_rebinds=audit.authority_rebinds + (record,),
    )
    if rebound.status == "ok":
        admitted, receipt = validate_intraburst_phase_transaction(
            rebound.slots, rebound)
        if not admitted:
            return replace(
                rebound,
                status="abstained",
                reason=("intraburst authority rebind failed validation: "
                        f"{receipt['admission_reason']}"),
                slots=(),
                provisional_slots=(),
                unowned_periods=(),
                certified_period_count=0,
            )
    return rebound


def remap_intraburst_phase_audit(audit, frame_mapper):
    """Remap a certified producer receipt into another frame coordinate system.

    Reverse-time maps swap interval endpoints and phase order.  The operation is
    all-or-nothing: any malformed mapper result returns an abstained audit with
    no actuator slots.
    """
    if not isinstance(audit, IntraBurstPhaseAudit):
        return _abstain("intraburst phase remap requires a typed audit")

    def mapped(frame):
        source = require_exact_integer(frame, "remap source frame")
        value = frame_mapper(source)
        return require_exact_integer(value, "remap result frame")

    def interval(lo, hi):
        left, right = mapped(lo), mapped(hi)
        return min(left, right), max(left, right), left > right

    try:
        mapped_slots = []
        mapped_provisional_slots = []
        source_slots = tuple(
            intraburst_phase_slot_from_value(slot) for slot in audit.slots)
        source_provisional_slots = tuple(
            intraburst_phase_slot_from_value(slot)
            for slot in audit.provisional_slots)
        source_unowned = tuple(
            unowned_motion_period_from_value(period)
            for period in audit.unowned_periods)
        source_authority = _exact_integer_tuple(
            audit.authoritative_events, "remap authoritative event")
        source_derived = _exact_integer_tuple(
            audit.derived_event_frames, "remap derived event")
        source_producer = _exact_integer_tuple(
            audit.producer_authoritative_events or source_authority,
            "remap producer authoritative event")
        def remap_slots(source, destination):
            for slot in source:
                frame_lo, frame_hi, reversed_time = interval(
                    slot.frame_lo, slot.frame_hi)
                period_lo, period_hi, period_reversed = interval(
                    slot.period_lo, slot.period_hi)
                if reversed_time != period_reversed:
                    raise ValueError(
                        "mapper reverses phase and period inconsistently")
                phase_index = (slot.phase_count - 1 - slot.phase_index
                               if reversed_time else slot.phase_index)
                destination.append(replace(
                    slot,
                    parent_frame=mapped(slot.parent_frame),
                    frame_lo=frame_lo,
                    frame_hi=frame_hi,
                    period_lo=period_lo,
                    period_hi=period_hi,
                    phase_index=int(phase_index),
                ))

        remap_slots(source_slots, mapped_slots)
        remap_slots(source_provisional_slots, mapped_provisional_slots)
        for rows in (mapped_slots, mapped_provisional_slots):
            rows.sort(key=lambda slot: (
                slot.frame_hi, slot.frame_lo, slot.parent_frame,
                slot.phase_index))
        mapped_ambiguous = []
        for lo, hi, reason in audit.ambiguous_periods:
            mapped_lo, mapped_hi, _reversed = interval(lo, hi)
            mapped_ambiguous.append((mapped_lo, mapped_hi, str(reason)))
        mapped_unowned = []
        for period in source_unowned:
            frame_lo, frame_hi, _reversed = interval(
                period.frame_lo, period.frame_hi)
            mapped_unowned.append(unowned_motion_period_from_value(replace(
                period,
                frame_lo=frame_lo,
                frame_hi=frame_hi,
                peak_frame=mapped(period.peak_frame),
            )))
        mapped_unowned.sort(key=lambda row: (
            row.frame_hi, row.frame_lo, row.peak_frame, row.phase_count))
        remapped = replace(
            audit,
            slots=tuple(mapped_slots),
            provisional_slots=tuple(mapped_provisional_slots),
            authoritative_events=tuple(sorted(
                mapped(frame) for frame in source_authority)),
            derived_event_frames=tuple(sorted(
                mapped(frame) for frame in source_derived)),
            ambiguous_periods=tuple(sorted(mapped_ambiguous)),
            unowned_periods=tuple(mapped_unowned),
            producer_authoritative_events=source_producer,
        )
        remapped = replace(
            remapped,
            authority_rebinds=remapped.authority_rebinds + (
                IntraBurstAuthorityRebind(
                    provenance="frame-remap",
                    prior_authoritative_events=source_authority,
                    authoritative_events=tuple(
                        remapped.authoritative_events),
                    removed_parent_events=(),
                    removed_slot_count=0,
                    frame_bounds=None,
                ),
            ),
        )
        # Reuse the actuator validator to certify the transformed transaction.
        if remapped.status == "ok":
            admitted, receipt = validate_intraburst_phase_transaction(
                remapped.slots, remapped)
            if not admitted:
                raise ValueError(receipt["admission_reason"])
        return remapped
    except Exception as exc:  # mapper is an external coordinate-system seam
        try:
            producer_events = _exact_integer_tuple(
                audit.producer_authoritative_events
                or audit.authoritative_events,
                "failed-remap producer authoritative event")
        except (TypeError, ValueError, OverflowError):
            producer_events = ()
        return replace(
            audit,
            status="abstained",
            reason=(f"intraburst phase remap failed: "
                    f"{type(exc).__name__}: {exc}"),
            slots=(),
            provisional_slots=(),
            unowned_periods=(),
            authoritative_events=(),
            certified_period_count=0,
            producer_authoritative_events=producer_events,
        )


def _abstain(reason, authoritative_events=()):
    """Build a whole-input abstention without partially trusted residue."""
    return IntraBurstPhaseAudit(
        status="abstained",
        reason=str(reason),
        slots=(),
        authoritative_events=tuple(authoritative_events),
        derived_event_frames=(),
        active_run_count=0,
        merged_period_count=0,
        certified_period_count=0,
        ambiguous_periods=(),
        producer_authoritative_events=tuple(authoritative_events),
    )


def _temporal_frames(fps):
    try:
        rate = float(fps)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rate) or rate <= 0:
        return None
    return (
        max(0, int(round(rate * _MOTION_SMOOTH_RADIUS_S))),
        max(1, int(round(rate * _MOTION_MIN_REST_S))),
    )


def certify_intraburst_phase_slots(
    motions: Mapping[int, float],
    authoritative_events,
    fps,
) -> IntraBurstPhaseAudit:
    """Recover bounded optional phases without changing event/count authority.

    ``motions`` must cover one consecutive frame interval with finite scalar
    values. ``authoritative_events`` is the exact peak list the tracker already
    consumes.  The returned slots exclude the phase containing their parent
    event, so a caller retains that parent exactly as before and adds only the
    non-parent phase intervals through the typed-slot ABI.

    Ambiguity is local to a merged period: that period emits no slot and records
    why. Malformed/non-dense input rejects the whole pass. An output-bound
    violation remains disabled but retains the complete typed set so a later
    authority rebind can filter it and reapply the unchanged bound.
    """
    if not isinstance(motions, Mapping):
        return _abstain("motion trace is not a mapping")
    try:
        raw_events = _exact_integer_tuple(
            authoritative_events, "authoritative event")
    except (TypeError, ValueError, OverflowError):
        return _abstain("authoritative events are not integer frames")
    if len(raw_events) != len(set(raw_events)):
        return _abstain("authoritative event frames are not unique")
    events = tuple(sorted(raw_events))

    temporal = _temporal_frames(fps)
    if temporal is None:
        return _abstain("fps is not finite and positive", events)
    smooth_radius, min_rest = temporal

    try:
        frames = tuple(sorted(
            require_exact_integer(frame, "motion frame")
            for frame in motions))
    except (TypeError, ValueError, OverflowError):
        return _abstain("motion frames are not integers", events)
    if not frames:
        return _abstain("motion trace is empty", events)
    if len(frames) != len(motions):
        return _abstain("motion frame keys are not uniquely integer-valued", events)
    if any(right != left + 1 for left, right in zip(frames, frames[1:])):
        return _abstain("motion trace is not consecutive", events)

    try:
        motion = np.asarray([float(motions[frame]) for frame in frames], dtype=float)
    except (TypeError, ValueError, KeyError):
        return _abstain("motion values are not finite scalars", events)
    if motion.ndim != 1 or len(motion) != len(frames) or not np.all(np.isfinite(motion)):
        return _abstain("motion values are not finite scalars", events)
    if events and (events[0] < frames[0] or events[-1] > frames[-1]):
        return _abstain("authoritative event lies outside dense motion coverage", events)

    smoothed = np.asarray(
        [np.median(motion[max(0, index - smooth_radius) : index + smooth_radius + 1]) for index in range(len(motion))]
    )
    threshold = float(np.percentile(smoothed, 50))
    active = smoothed > threshold

    runs = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(active) and active[end + 1]:
            end += 1
        runs.append((index, end))
        index = end + 1

    periods: list[list[tuple[int, int]]] = []
    for run in runs:
        start, _end = run
        if periods and frames[start] - frames[periods[-1][-1][1]] < min_rest:
            periods[-1].append(run)
        else:
            periods.append([run])

    derived_events = []
    slots = []
    ambiguous = []
    unowned_periods = []
    certified_periods = 0
    for phases in periods:
        period_start_i = phases[0][0]
        period_end_i = phases[-1][1]
        peak_i = period_start_i + int(np.argmax(smoothed[period_start_i : period_end_i + 1]))
        peak_frame = int(frames[peak_i])
        derived_events.append(peak_frame)
        period_lo = int(frames[period_start_i])
        period_hi = int(frames[period_end_i])
        parents = [frame for frame in events if period_lo <= frame <= period_hi]
        if not parents:
            unowned_periods.append(UnownedMotionPeriod(
                frame_lo=period_lo,
                frame_hi=period_hi,
                peak_frame=peak_frame,
                phase_count=int(len(phases)),
            ))
        if len(phases) <= 1:
            continue

        if len(parents) != 1:
            ambiguous.append(
                (
                    period_lo,
                    period_hi,
                    f"authoritative-parent-count:{len(parents)}",
                )
            )
            continue
        parent = int(parents[0])
        parent_phases = [
            phase_i
            for phase_i, (start_i, end_i) in enumerate(phases)
            if int(frames[start_i]) <= parent <= int(frames[end_i])
        ]
        if len(parent_phases) != 1:
            ambiguous.append(
                (
                    period_lo,
                    period_hi,
                    f"parent-phase-count:{len(parent_phases)}",
                )
            )
            continue

        parent_phase = parent_phases[0]
        certified_periods += 1
        for phase_i, (start_i, end_i) in enumerate(phases):
            if phase_i == parent_phase:
                continue
            slots.append(
                IntraBurstPhaseSlot(
                    parent_frame=parent,
                    frame_lo=int(frames[start_i]),
                    frame_hi=int(frames[end_i]),
                    period_lo=period_lo,
                    period_hi=period_hi,
                    phase_index=int(phase_i),
                    phase_count=int(len(phases)),
                )
            )

    slots.sort(
        key=lambda slot: (
            slot.frame_hi,
            slot.frame_lo,
            slot.parent_frame,
            slot.phase_index,
        )
    )
    if len(slots) > MAX_PHASE_SLOTS:
        return IntraBurstPhaseAudit(
            status="over-bound",
            reason=(f"certified phase slots {len(slots)} exceed existing structural bound {MAX_PHASE_SLOTS}"),
            slots=(),
            provisional_slots=tuple(slots),
            authoritative_events=events,
            derived_event_frames=tuple(derived_events),
            active_run_count=len(runs),
            merged_period_count=sum(len(phases) > 1 for phases in periods),
            certified_period_count=certified_periods,
            ambiguous_periods=tuple(ambiguous),
            unowned_periods=tuple(unowned_periods),
            producer_authoritative_events=events,
        )

    return IntraBurstPhaseAudit(
        status="ok" if slots else "no-certified-slots",
        reason=None,
        slots=tuple(slots),
        authoritative_events=events,
        derived_event_frames=tuple(derived_events),
        active_run_count=len(runs),
        merged_period_count=sum(len(phases) > 1 for phases in periods),
        certified_period_count=certified_periods,
        ambiguous_periods=tuple(ambiguous),
        unowned_periods=tuple(unowned_periods),
        producer_authoritative_events=events,
    )
