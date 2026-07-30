"""Device-resident ordinary-window scrub beam primitives.

This module is intentionally not imported by :mod:`detect.scrub_decode` yet.  It
isolates the hot, chronological beam mechanics so their ordering and retention
semantics can be proved before the live decoder is changed:

* source-major SKIP / SINGLE / DOUBLE expansion;
* lossless fixed-width candidate keys;
* insertion-ordered, highest-score-wins deduplication; and
* the ordinary-window band, incumbent reinjection, and K-per-OM cut.

The ``*_reference`` functions are a NumPy oracle.  The ``*_device`` functions
operate only on CUDA tensors and never copy candidate data to the host.  Torch
is imported lazily, so importing this default-inert module does not initialize
Torch or CUDA.

Candidate ancestry is one edge, not a materialized Python word.  ``source``,
``action``, ``move0``, and ``move1`` on retained rows are sufficient for the
host to append an action to the corresponding source word.  A later persistent
integration can instead keep those edges device-side across multiple steps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np


N_FACELETS = 54
N_MOVES = 18
N_FACES = 6
MOVES_PER_FACE = 3
N_LIVE_ACTIONS = 1 + N_MOVES * (1 + N_MOVES - MOVES_PER_FACE)
# Rigid-cube orientation count: the OM stratum universe of the ordinary-window
# beam (matches the callers' ``beam_limit = beam_k * n_om`` and the audit field
# ``om_strata=24``).  Structural, like ``N_MOVES`` — not a tuned knob.
N_ORIENTATIONS = 24

# Constant-total-budget stratum redistribution (default OFF => byte-identical).
# Pre-registered; default OFF keeps the legacy selection byte-identical.
_STRATUM_REDIST_ENV = "CUBED_BEAM_STRATUM_REDIST"


def stratum_redistribution_enabled() -> bool:
    """True when ``CUBED_BEAM_STRATUM_REDIST`` requests budget redistribution."""

    value = os.environ.get(_STRATUM_REDIST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def effective_beam_k(
    beam_k: int, n_surviving: int, n_om_total: int = N_ORIENTATIONS
) -> int:
    """Per-stratum cap holding the TOTAL beam budget constant.

    The legacy K-per-OM cut budgets ``n_om_total * beam_k`` rows but silently
    forfeits every dead stratum's share.  Redistribution keeps that same total:
    ``floor(n_om_total * beam_k / n_surviving)``, min-clamped at the legacy
    ``beam_k`` so behaviour can never drop below today's.  With all strata
    alive this is exactly ``beam_k`` (identity); no new tuned constant exists —
    both factors of the total already exist in the configuration.
    """

    if n_surviving <= 0:
        return int(beam_k)
    return max(int(beam_k), (int(n_om_total) * int(beam_k)) // int(n_surviving))

ACTION_SKIP = 0
ACTION_SINGLE = 1
ACTION_DOUBLE = 2

# Eighteen base-6 digits fit in 47 bits.  The unused high bits carry the
# dictionary-key metadata while all three signed-int64 words remain positive.
_COLOR_DIGITS = 18
_COLOR_RADIX = 6**_COLOR_DIGITS
# The positive payload occupies [1, 6**18].  Reserve an explicit zero guard
# between metadata slabs so their boundaries cannot share a representation.
_KEY_RADIX = _COLOR_RADIX + 1
_INT64_MAX = np.iinfo(np.int64).max
_META_MAX = (_INT64_MAX - _COLOR_RADIX) // _KEY_RADIX
_COLOR_WEIGHTS = np.asarray([6**i for i in range(_COLOR_DIGITS)], dtype=np.int64)
_DEVICE_COLOR_WEIGHTS: dict[tuple[str, int | None], Any] = {}


@dataclass(frozen=True)
class BeamBatch:
    """A structure-of-arrays beam on either NumPy or one CUDA device.

    ``source`` and the action fields describe the edge that produced each row.
    They may be ``None`` for an input beam and are always populated by
    expansion.  Core state is fixed to ``int8[N,54]`` and score to ``float64``;
    the device functions enforce the corresponding Torch dtypes without a
    device synchronization.
    """

    states: Any
    score: Any
    oi: Any
    nskip: Any
    nins: Any
    req: Any
    source: Any | None = None
    action: Any | None = None
    move0: Any | None = None
    move1: Any | None = None
    branch: Any | None = None

    def __len__(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class BranchTemplate:
    """One source's branch rows in exact legacy emission order."""

    action: Any
    move0: Any
    move1: Any
    perm: Any

    def __len__(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class DedupResult:
    """Deduplicated winners, still resident on the input backend/device."""

    batch: BeamBatch
    # Indices into the expanded input.  Ordered by each key's first occurrence.
    winner_indices: Any
    first_indices: Any


@dataclass(frozen=True)
class SelectionResult:
    """Legacy-ordered ordinary-window survivors."""

    batch: BeamBatch
    # Indices into the deduplicated input.
    selected_indices: Any
    # Stable global score order, useful for diagnostics without rebuilding it.
    ranked_indices: Any
    # Global ranked rows after band + required-lane reinjection, before K.
    kept_indices: Any | None = None


@dataclass(frozen=True)
class RotationResult:
    """Stable orientation fan-out and deduplication result.

    ``source_indices`` maps every retained row directly to the corresponding
    row of the input batch.  Parallel device-resident word/timing payloads can
    therefore gather once without materializing candidate metadata on the host.
    """

    batch: BeamBatch
    source_indices: Any
    winner_indices: Any


def _torch():
    """Import Torch only for an explicitly requested device operation."""

    return import_module("torch")


def take_batch(batch: BeamBatch, indices: Any) -> BeamBatch:
    """Gather beam rows without changing their backend or device."""

    def take(value):
        return None if value is None else value[indices]

    return BeamBatch(
        states=take(batch.states),
        score=take(batch.score),
        oi=take(batch.oi),
        nskip=take(batch.nskip),
        nins=take(batch.nins),
        req=take(batch.req),
        source=take(batch.source),
        action=take(batch.action),
        move0=take(batch.move0),
        move1=take(batch.move1),
        branch=take(batch.branch),
    )


def _validate_reference_batch(batch: BeamBatch) -> None:
    n = len(batch)
    if not isinstance(batch.states, np.ndarray):
        raise TypeError("reference states must be a NumPy array")
    if batch.states.shape != (n, N_FACELETS) or batch.states.dtype != np.int8:
        raise ValueError("states must be int8[N,54]")
    if not isinstance(batch.score, np.ndarray) or batch.score.shape != (n,):
        raise ValueError("score must be a NumPy vector")
    if batch.score.dtype != np.float64:
        raise ValueError("score must be float64")
    if not np.isfinite(batch.score).all():
        raise ValueError("score must be finite")
    for name in ("oi", "nskip", "nins", "req"):
        value = getattr(batch, name)
        if not isinstance(value, np.ndarray) or value.shape != (n,):
            raise ValueError(f"{name} must be a NumPy vector")
    for name in ("oi", "nskip", "nins"):
        if getattr(batch, name).dtype != np.int64:
            raise ValueError(f"{name} must be int64")
    if batch.req.dtype != np.bool_:
        raise ValueError("req must be bool")
    for name in ("source", "action", "move0", "move1", "branch"):
        value = getattr(batch, name)
        if value is not None and (not isinstance(value, np.ndarray) or value.shape != (n,)):
            raise ValueError(f"{name} must be None or a NumPy vector")


def _validate_device_batch(batch: BeamBatch) -> Any:
    """Shape/dtype/device checks only; deliberately performs no D2H sync."""

    torch = _torch()
    n = len(batch)
    values = (
        batch.states,
        batch.score,
        batch.oi,
        batch.nskip,
        batch.nins,
        batch.req,
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("device beam fields must be Torch tensors")
    if not all(value.is_cuda for value in values):
        raise ValueError("device beam fields must remain on CUDA")
    device = batch.states.device
    if not all(value.device == device for value in values):
        raise ValueError("device beam fields must share one CUDA device")
    if batch.states.shape != (n, N_FACELETS) or batch.states.dtype != torch.int8:
        raise ValueError("states must be torch.int8[N,54]")
    if batch.score.shape != (n,) or batch.score.dtype != torch.float64:
        raise ValueError("score must be torch.float64[N]")
    for name in ("oi", "nskip", "nins"):
        value = getattr(batch, name)
        if value.shape != (n,) or value.dtype != torch.int64:
            raise ValueError(f"{name} must be torch.int64[N]")
    if batch.req.shape != (n,) or batch.req.dtype != torch.bool:
        raise ValueError("req must be torch.bool[N]")
    ancestry_dtypes = {
        "source": torch.int64,
        "action": torch.int8,
        "move0": torch.int8,
        "move1": torch.int8,
        "branch": torch.int64,
    }
    for name, dtype in ancestry_dtypes.items():
        value = getattr(batch, name)
        if value is not None and (
            not isinstance(value, torch.Tensor)
            or not value.is_cuda
            or value.device != device
            or value.shape != (n,)
            or value.dtype != dtype
        ):
            raise ValueError(f"{name} must be None or {dtype}[N] on the beam CUDA device")
    return torch


def finite_device(batch: BeamBatch) -> BeamBatch:
    """Drop nonfinite score rows on-device before ordered rotation/dedup.

    Stable dictionary semantics assume the same finite rows the legacy read
    loop admits.  Filtering before a fan-out matters: a dead first occurrence
    could otherwise change key insertion order even when a live duplicate wins.
    """

    torch = _validate_device_batch(batch)
    return take_batch(batch, torch.isfinite(batch.score))


def rotate_reference(
    batch: BeamBatch,
    destinations: np.ndarray,
    degrees: np.ndarray,
    *,
    extra_identity_key: Any = None,
) -> RotationResult:
    """Reference source-major orientation fan-out with stable key dedup.

    ``extra_identity_key`` is optional source-row identity carried through the
    fan-out and included in deduplication.  ``None`` preserves the historical
    state/OM key exactly.
    """

    _validate_reference_batch(batch)
    destinations = np.asarray(destinations, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    if destinations.ndim != 2 or degrees.shape != (destinations.shape[0],):
        raise ValueError("invalid orientation destination table")
    if extra_identity_key is not None:
        extra_identity_key = np.asarray(extra_identity_key)
        if (
            extra_identity_key.ndim != 2
            or extra_identity_key.shape[0] != len(batch)
            or extra_identity_key.dtype != np.int64
        ):
            raise ValueError("extra_identity_key must be int64[N,M]")
    finite = np.flatnonzero(np.isfinite(batch.score)).astype(np.int64)
    clean = take_batch(batch, finite)
    clean_identity = (
        None if extra_identity_key is None else extra_identity_key[finite]
    )
    source = []
    dst = []
    for src, oi in enumerate(clean.oi):
        for slot in range(int(degrees[int(oi)])):
            source.append(src)
            dst.append(int(destinations[int(oi), slot]))
    source = np.asarray(source, dtype=np.int64)
    expanded = take_batch(clean, source)
    expanded = BeamBatch(
        states=expanded.states,
        score=expanded.score,
        oi=np.asarray(dst, dtype=np.int64),
        nskip=expanded.nskip,
        nins=expanded.nins,
        req=expanded.req,
        source=expanded.source,
        action=expanded.action,
        move0=expanded.move0,
        move1=expanded.move1,
        branch=expanded.branch,
    )
    expanded_identity = (
        None if clean_identity is None else clean_identity[source]
    )
    dedup = stable_dedup_reference(
        expanded, extra_identity_key=expanded_identity
    )
    retained_source = finite[source[dedup.winner_indices]]
    return RotationResult(
        batch=dedup.batch,
        source_indices=retained_source,
        winner_indices=dedup.winner_indices,
    )


def rotate_device(
    batch: BeamBatch,
    destinations: Any,
    degrees: Any,
    *,
    extra_identity_key: Any = None,
) -> RotationResult:
    """CUDA source-major orientation fan-out followed by stable deduplication.

    ``destinations`` must encode the literal legacy iteration order produced by
    ``tuple({oi} | set(om_nbrs[oi]))`` for each orientation.  Nonfinite rows are
    removed before fan-out so a dead first occurrence cannot alter dictionary
    insertion order.  Optional ``extra_identity_key`` remains resident, follows
    the same source fan-out, and participates in stable deduplication.
    """

    torch = _validate_device_batch(batch)
    n_om = int(destinations.shape[0])
    if (
        not isinstance(destinations, torch.Tensor)
        or not destinations.is_cuda
        or destinations.device != batch.states.device
        or destinations.dtype != torch.int64
        or destinations.ndim != 2
        or not isinstance(degrees, torch.Tensor)
        or not degrees.is_cuda
        or degrees.device != batch.states.device
        or degrees.dtype != torch.int64
        or degrees.shape != (n_om,)
    ):
        raise ValueError("orientation tables must be CUDA int64 tensors")
    if extra_identity_key is not None and (
        not isinstance(extra_identity_key, torch.Tensor)
        or not extra_identity_key.is_cuda
        or extra_identity_key.device != batch.states.device
        or extra_identity_key.dtype != torch.int64
        or extra_identity_key.ndim != 2
        or extra_identity_key.shape[0] != len(batch)
    ):
        raise ValueError(
            "extra_identity_key must be torch.int64[N,M] on beam device"
        )

    finite_indices = torch.arange(
        len(batch), dtype=torch.int64, device=batch.states.device
    )[torch.isfinite(batch.score)]
    clean = take_batch(batch, finite_indices)
    clean_identity = (
        None if extra_identity_key is None else extra_identity_key[finite_indices]
    )
    max_degree = int(destinations.shape[1])
    source = torch.arange(
        len(clean), dtype=torch.int64, device=batch.states.device
    ).repeat_interleave(max_degree)
    slot = torch.arange(
        max_degree, dtype=torch.int64, device=batch.states.device
    ).repeat(len(clean))
    source_oi = clean.oi[source]
    keep = slot < degrees[source_oi]
    source = source[keep]
    slot = slot[keep]
    source_oi = source_oi[keep]
    expanded = take_batch(clean, source)
    expanded = BeamBatch(
        states=expanded.states,
        score=expanded.score,
        oi=destinations[source_oi, slot],
        nskip=expanded.nskip,
        nins=expanded.nins,
        req=expanded.req,
        source=expanded.source,
        action=expanded.action,
        move0=expanded.move0,
        move1=expanded.move1,
        branch=expanded.branch,
    )
    expanded_identity = (
        None if clean_identity is None else clean_identity[source]
    )
    dedup = stable_dedup_device(
        expanded, extra_identity_key=expanded_identity
    )
    retained_source = finite_indices[source[dedup.winner_indices]]
    return RotationResult(
        batch=dedup.batch,
        source_indices=retained_source,
        winner_indices=dedup.winner_indices,
    )


def build_branch_template_reference(perms: np.ndarray) -> BranchTemplate:
    """Build SKIP, then each SINGLE followed by its 15 legal DOUBLES.

    ``perms[mi]`` maps ``new_state = old_state[perms[mi]]``.  A DOUBLE uses
    the composed permutation ``perms[mi][perms[mj]]``, exactly matching the
    legacy ``s1 = s[p_i]; s2 = s1[p_j]`` sequence.
    """

    perms = np.asarray(perms)
    if perms.shape != (N_MOVES, N_FACELETS):
        raise ValueError("perms must have shape [18,54]")
    if not np.issubdtype(perms.dtype, np.integer):
        raise ValueError("perms must be integer indices")

    actions = [ACTION_SKIP]
    move0 = [-1]
    move1 = [-1]
    out_perms = [np.arange(N_FACELETS, dtype=np.int64)]
    for mi in range(N_MOVES):
        p1 = np.asarray(perms[mi], dtype=np.int64)
        actions.append(ACTION_SINGLE)
        move0.append(mi)
        move1.append(-1)
        out_perms.append(p1)
        for mj in range(N_MOVES):
            if mj // MOVES_PER_FACE == mi // MOVES_PER_FACE:
                continue
            actions.append(ACTION_DOUBLE)
            move0.append(mi)
            move1.append(mj)
            out_perms.append(p1[np.asarray(perms[mj], dtype=np.int64)])
    return BranchTemplate(
        action=np.asarray(actions, dtype=np.int8),
        move0=np.asarray(move0, dtype=np.int8),
        move1=np.asarray(move1, dtype=np.int8),
        perm=np.stack(out_perms).astype(np.int64, copy=False),
    )


def build_branch_template_device(perms: Any) -> BranchTemplate:
    """CUDA equivalent of :func:`build_branch_template_reference`."""

    torch = _torch()
    if not isinstance(perms, torch.Tensor) or not perms.is_cuda:
        raise ValueError("perms must be a CUDA tensor")
    if perms.shape != (N_MOVES, N_FACELETS) or perms.dtype != torch.int64:
        raise ValueError("perms must be torch.int64[18,54]")

    actions = [ACTION_SKIP]
    move0 = [-1]
    move1 = [-1]
    out_perms = [torch.arange(N_FACELETS, dtype=torch.int64, device=perms.device)]
    for mi in range(N_MOVES):
        p1 = perms[mi]
        actions.append(ACTION_SINGLE)
        move0.append(mi)
        move1.append(-1)
        out_perms.append(p1)
        for mj in range(N_MOVES):
            if mj // MOVES_PER_FACE == mi // MOVES_PER_FACE:
                continue
            actions.append(ACTION_DOUBLE)
            move0.append(mi)
            move1.append(mj)
            out_perms.append(p1[perms[mj]])
    return BranchTemplate(
        action=torch.tensor(actions, dtype=torch.int8, device=perms.device),
        move0=torch.tensor(move0, dtype=torch.int8, device=perms.device),
        move1=torch.tensor(move1, dtype=torch.int8, device=perms.device),
        perm=torch.stack(out_perms),
    )


def _validate_required_next_reference(required_next: Any, n: int) -> np.ndarray:
    if required_next is None:
        return np.full((n, 2), -1, dtype=np.int64)
    value = np.asarray(required_next, dtype=np.int64)
    if value.shape != (n, 2):
        raise ValueError("required_next must have shape [N,2]")
    return value


def expand_reference(
    batch: BeamBatch,
    perms: np.ndarray,
    *,
    skip_cap: int,
    insert_cap: int,
    required_next: Any = None,
    single_req_override: Any = None,
    single_score_delta: Any = None,
    live_action_score_delta: Any = None,
    allow_actions: bool = True,
    template: BranchTemplate | None = None,
) -> BeamBatch:
    """Expand one ordinary action slot using exact legacy branch order.

    ``required_next[i]`` contains the next two incumbent move indices for
    source ``i`` (``-1`` means the required word has ended).  SKIP retains the
    source's ``req`` flag; an action retains it only when all emitted moves
    match those next positions.  This is the device-friendly equivalent of
    comparing each materialized Python sequence with ``required_word``.

    ``single_req_override`` is optional ``int8[N,18]`` tri-state data.  ``-1``
    preserves the legacy required-prefix result for that SINGLE branch, while
    ``0``/``1`` replace it with an absolute false/true value.  This permits a
    caller whose canonical search representation rewrites the latest physical
    action to restore the required lane before deduplication.  Literal ``None``
    executes the untouched legacy path.

    ``single_score_delta`` is optional ``float64[N,18]`` data for the legacy
    single-move center-motion addition.  ``live_action_score_delta`` is an
    independent optional ``float64[N,289]`` plane over the prefix contract's
    exact live-action order: SKIP, then for each first move its SINGLE followed
    by the 15 different-face DOUBLES in ascending second-move order.  Both
    additions apply to SINGLE rows when both planes are present.
    """

    _validate_reference_batch(batch)
    n = len(batch)
    tpl = template or build_branch_template_reference(perms)
    required_next = _validate_required_next_reference(required_next, n)
    if single_req_override is not None:
        single_req_override = np.asarray(single_req_override)
        if (
            single_req_override.shape != (n, N_MOVES)
            or single_req_override.dtype != np.int8
        ):
            raise ValueError("single_req_override must be int8[N,18]")
        if np.any((single_req_override < -1) | (single_req_override > 1)):
            raise ValueError("single_req_override values must be -1, 0, or 1")
    if single_score_delta is None:
        single_score_delta = np.zeros((n, N_MOVES), dtype=np.float64)
    else:
        single_score_delta = np.asarray(single_score_delta)
        if single_score_delta.shape != (n, N_MOVES):
            raise ValueError("single_score_delta must have shape [N,18]")
        if single_score_delta.dtype != np.float64:
            raise ValueError("single_score_delta must be float64")
    if live_action_score_delta is not None:
        live_action_score_delta = np.asarray(live_action_score_delta)
        if live_action_score_delta.shape != (n, N_LIVE_ACTIONS):
            raise ValueError("live_action_score_delta must have shape [N,289]")
        if live_action_score_delta.dtype != np.float64:
            raise ValueError("live_action_score_delta must be float64")

    t = len(tpl)
    if live_action_score_delta is not None and t != N_LIVE_ACTIONS:
        raise ValueError("live_action_score_delta requires the canonical 289-row template")
    source = np.repeat(np.arange(n, dtype=np.int64), t)
    branch = np.tile(np.arange(t, dtype=np.int64), n)
    action = np.tile(tpl.action, n)
    move0 = np.tile(tpl.move0, n)
    move1 = np.tile(tpl.move1, n)
    keep = np.where(
        action == ACTION_SKIP,
        batch.nskip[source] < int(skip_cap),
        np.where(
            action == ACTION_SINGLE,
            bool(allow_actions),
            bool(allow_actions) & (batch.nins[source] < int(insert_cap)),
        ),
    )
    source = source[keep]
    branch = branch[keep]
    action = action[keep]
    move0 = move0[keep]
    move1 = move1[keep]

    states = batch.states[source[:, None], tpl.perm[branch]]
    score = batch.score[source].copy()
    if live_action_score_delta is not None:
        score += live_action_score_delta[source, branch]
    is_skip = action == ACTION_SKIP
    is_single = action == ACTION_SINGLE
    is_double = action == ACTION_DOUBLE
    single_rows = np.nonzero(is_single)[0]
    if len(single_rows):
        score[single_rows] += single_score_delta[source[single_rows], move0[single_rows]]

    req = batch.req[source].copy()
    req[is_single] &= required_next[source[is_single], 0] == move0[is_single]
    req[is_double] &= (required_next[source[is_double], 0] == move0[is_double]) & (
        required_next[source[is_double], 1] == move1[is_double]
    )
    if single_req_override is not None and len(single_rows):
        override = single_req_override[
            source[single_rows], move0[single_rows].astype(np.int64, copy=False)
        ]
        overridden = override >= 0
        req[single_rows[overridden]] = override[overridden].astype(
            np.bool_, copy=False
        )
    return BeamBatch(
        states=states.astype(np.int8, copy=False),
        score=score,
        oi=batch.oi[source].astype(np.int64, copy=False),
        nskip=batch.nskip[source].astype(np.int64, copy=False) + is_skip,
        nins=batch.nins[source].astype(np.int64, copy=False) + is_double,
        req=req,
        source=source,
        action=action,
        move0=move0,
        move1=move1,
        branch=branch,
    )


def expand_device(
    batch: BeamBatch,
    perms: Any,
    *,
    skip_cap: int,
    insert_cap: int,
    required_next: Any = None,
    single_req_override: Any = None,
    single_score_delta: Any = None,
    live_action_score_delta: Any = None,
    allow_actions: bool = True,
    template: BranchTemplate | None = None,
) -> BeamBatch:
    """CUDA-resident equivalent of :func:`expand_reference`.

    There is no ``.item()``, ``.cpu()``, or implicit candidate synchronization
    in this function.  Build and reuse ``template`` once in a persistent caller
    to avoid reconstructing the 289 branch permutations per action slot.
    """

    torch = _validate_device_batch(batch)
    n = len(batch)
    tpl = template or build_branch_template_device(perms)
    if required_next is None:
        required_next = torch.full((n, 2), -1, dtype=torch.int64, device=batch.states.device)
    elif (
        not isinstance(required_next, torch.Tensor)
        or not required_next.is_cuda
        or required_next.device != batch.states.device
        or required_next.dtype != torch.int64
        or required_next.shape != (n, 2)
    ):
        raise ValueError("required_next must be torch.int64[N,2] on beam device")
    if single_req_override is not None and (
        not isinstance(single_req_override, torch.Tensor)
        or not single_req_override.is_cuda
        or single_req_override.device != batch.states.device
        or single_req_override.dtype != torch.int8
        or single_req_override.shape != (n, N_MOVES)
    ):
        raise ValueError(
            "single_req_override must be torch.int8[N,18] on beam device"
        )
    if single_score_delta is None:
        single_score_delta = torch.zeros((n, N_MOVES), dtype=torch.float64, device=batch.states.device)
    elif (
        not isinstance(single_score_delta, torch.Tensor)
        or not single_score_delta.is_cuda
        or single_score_delta.device != batch.states.device
        or single_score_delta.dtype != torch.float64
        or single_score_delta.shape != (n, N_MOVES)
    ):
        raise ValueError("single_score_delta must be torch.float64[N,18] on beam device")
    if live_action_score_delta is not None and (
        not isinstance(live_action_score_delta, torch.Tensor)
        or not live_action_score_delta.is_cuda
        or live_action_score_delta.device != batch.states.device
        or live_action_score_delta.dtype != torch.float64
        or live_action_score_delta.shape != (n, N_LIVE_ACTIONS)
    ):
        raise ValueError("live_action_score_delta must be torch.float64[N,289] on beam device")

    t = len(tpl)
    if live_action_score_delta is not None and t != N_LIVE_ACTIONS:
        raise ValueError("live_action_score_delta requires the canonical 289-row template")
    source = torch.arange(n, dtype=torch.int64, device=batch.states.device)
    source = source.repeat_interleave(t)
    branch = torch.arange(t, dtype=torch.int64, device=batch.states.device).repeat(n)
    action = tpl.action.repeat(n)
    move0 = tpl.move0.repeat(n)
    move1 = tpl.move1.repeat(n)
    keep = torch.where(
        action == ACTION_SKIP,
        batch.nskip[source] < int(skip_cap),
        torch.where(
            action == ACTION_SINGLE,
            torch.full_like(action, bool(allow_actions), dtype=torch.bool),
            bool(allow_actions) & (batch.nins[source] < int(insert_cap)),
        ),
    )
    source = source[keep]
    branch = branch[keep]
    action = action[keep]
    move0 = move0[keep]
    move1 = move1[keep]

    states = batch.states[source[:, None], tpl.perm[branch]]
    is_skip = action == ACTION_SKIP
    is_single = action == ACTION_SINGLE
    is_double = action == ACTION_DOUBLE
    safe_move0 = move0.to(torch.int64).clamp_min(0)
    # Preserve the reference operation order exactly.  FP64 addition is not
    # associative, and adding the SINGLE term before the Hx289 term creates a
    # one-ULP CPU/CUDA drift even when survivor indices agree.
    score = batch.score[source]
    if live_action_score_delta is not None:
        score = score + live_action_score_delta[source, branch]
    delta = single_score_delta[source, safe_move0]
    score = score + torch.where(is_single, delta, torch.zeros_like(delta))
    req = batch.req[source]
    req = torch.where(
        is_single,
        req & (required_next[source, 0] == move0),
        req,
    )
    req = torch.where(
        is_double,
        req & (required_next[source, 0] == move0) & (required_next[source, 1] == move1),
        req,
    )
    if single_req_override is not None:
        override = single_req_override[source, safe_move0]
        req = torch.where(
            is_single & (override >= 0),
            override == 1,
            req,
        )
    return BeamBatch(
        states=states,
        score=score,
        oi=batch.oi[source],
        nskip=batch.nskip[source] + is_skip.to(torch.int64),
        nins=batch.nins[source] + is_double.to(torch.int64),
        req=req,
        source=source,
        action=action,
        move0=move0,
        move1=move1,
        branch=branch,
    )


def pack_keys_reference(batch: BeamBatch) -> np.ndarray:
    """Pack ``(state, oi, nskip, nins, req)`` into three positive int64s."""

    _validate_reference_batch(batch)
    states64 = batch.states.astype(np.int64, copy=False).reshape(len(batch), 3, 18)
    if np.any((states64 < 0) | (states64 >= 6)):
        raise ValueError("state colors must be in [0,5]")
    meta0 = 2 * batch.oi.astype(np.int64, copy=False) + batch.req.astype(np.int64)
    for name, value in (
        ("oi/req", meta0),
        ("nskip", batch.nskip),
        ("nins", batch.nins),
    ):
        if np.any((value < 0) | (value > _META_MAX)):
            raise ValueError(f"{name} metadata exceeds packed-key range")
    colors = np.sum(states64 * _COLOR_WEIGHTS[None, None, :], axis=2)
    keys = colors + 1
    keys[:, 0] += meta0 * _KEY_RADIX
    keys[:, 1] += batch.nskip.astype(np.int64, copy=False) * _KEY_RADIX
    keys[:, 2] += batch.nins.astype(np.int64, copy=False) * _KEY_RADIX
    return keys


def pack_keys_device(batch: BeamBatch) -> Any:
    """CUDA-resident packed keys; assumes validated color/metadata ranges.

    Range checks are intentionally left to the CPU ingress/debug path: a
    device-side assertion would introduce the synchronization this module is
    meant to remove.  Shape, dtype, and common-device checks still run.
    """

    torch = _validate_device_batch(batch)
    cache_key = (batch.states.device.type, batch.states.device.index)
    weights = _DEVICE_COLOR_WEIGHTS.get(cache_key)
    if weights is None:
        weights = torch.tensor(_COLOR_WEIGHTS.tolist(), dtype=torch.int64, device=batch.states.device)
        _DEVICE_COLOR_WEIGHTS[cache_key] = weights
    colors = (batch.states.to(torch.int64).reshape(len(batch), 3, 18) * weights.reshape(1, 1, 18)).sum(dim=2)
    keys = colors + 1
    keys[:, 0] += (2 * batch.oi + batch.req.to(torch.int64)) * _KEY_RADIX
    keys[:, 1] += batch.nskip * _KEY_RADIX
    keys[:, 2] += batch.nins * _KEY_RADIX
    return keys


def unpack_keys_reference(keys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Inverse of :func:`pack_keys_reference`, used by parity tests/debugging."""

    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype != np.int64:
        raise ValueError("keys must be int64[N,3]")
    if np.any(keys <= 0):
        raise ValueError("packed keys must be positive")
    raw = keys - 1
    colors = raw % _KEY_RADIX
    meta = raw // _KEY_RADIX
    states = np.empty((len(keys), 3, 18), dtype=np.int8)
    for digit in range(18):
        states[:, :, digit] = (colors % 6).astype(np.int8)
        colors //= 6
    req = (meta[:, 0] % 2).astype(np.bool_)
    oi = meta[:, 0] // 2
    return states.reshape(-1, 54), oi, meta[:, 1], meta[:, 2], req


def stable_dedup_reference(
    batch: BeamBatch,
    *,
    extra_identity_key: Any = None,
) -> DedupResult:
    """Match an insertion-ordered best-by-complete-key dictionary exactly.

    ``extra_identity_key`` is the parallel shadow frontier's exact history key.
    Literal ``None`` preserves the production control beam's historical
    current-state identity.  When present, equal current states reached through
    different model-visible histories remain separate before shadow band/K.
    """

    keys = pack_keys_reference(batch)
    if extra_identity_key is not None:
        extra = np.asarray(extra_identity_key)
        if (
            extra.ndim != 2
            or extra.shape[0] != len(batch)
            or extra.dtype != np.int64
        ):
            raise ValueError("extra_identity_key must be int64[N,M]")
        keys = np.concatenate((keys, extra), axis=1)
    positions: dict[tuple[int, ...], int] = {}
    first: list[int] = []
    winners: list[int] = []
    for index, row in enumerate(keys):
        key = tuple(int(value) for value in row)
        position = positions.get(key)
        if position is None:
            positions[key] = len(first)
            first.append(index)
            winners.append(index)
        elif batch.score[index] > batch.score[winners[position]]:
            # Assignment updates the value but never moves the key.
            winners[position] = index
    winner_indices = np.asarray(winners, dtype=np.int64)
    first_indices = np.asarray(first, dtype=np.int64)
    return DedupResult(
        batch=take_batch(batch, winner_indices),
        winner_indices=winner_indices,
        first_indices=first_indices,
    )


def _lexsort_device(keys: Any, order: Any) -> Any:
    """Stable lexicographic order over an arbitrary fixed-width key matrix."""

    torch = _torch()
    for column in range(int(keys.shape[1]) - 1, -1, -1):
        local = torch.argsort(keys[order, column], stable=True)
        order = order[local]
    return order


def _group_starts_device(keys: Any, order: Any) -> Any:
    torch = _torch()
    n = int(order.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.bool, device=order.device)
    sorted_keys = keys[order]
    return torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=order.device),
            torch.any(sorted_keys[1:] != sorted_keys[:-1], dim=1),
        )
    )


def stable_dedup_device(
    batch: BeamBatch,
    *,
    extra_identity_key: Any = None,
) -> DedupResult:
    """CUDA-only stable dictionary dedup without candidate D2H.

    Stable score sorting gives each key its highest score and retains the first
    input on exact ties.  A second stable key ordering recovers first-insertion
    group order.  This requires PyTorch's ``torch.argsort(..., stable=True)``
    support (present in the project's supported Torch >=2.0).
    """

    torch = _validate_device_batch(batch)
    keys = pack_keys_device(batch)
    if extra_identity_key is not None:
        if (
            not isinstance(extra_identity_key, torch.Tensor)
            or extra_identity_key.device != keys.device
            or extra_identity_key.dtype != torch.int64
            or extra_identity_key.ndim != 2
            or extra_identity_key.shape[0] != len(batch)
        ):
            raise ValueError(
                "extra_identity_key must be torch.int64[N,M] on beam device"
            )
        keys = torch.cat((keys, extra_identity_key), dim=1)
    original = torch.arange(len(batch), dtype=torch.int64, device=keys.device)

    by_first = _lexsort_device(keys, original)
    first_indices = by_first[_group_starts_device(keys, by_first)]

    by_score = torch.argsort(batch.score, descending=True, stable=True)
    by_winner = _lexsort_device(keys, by_score)
    winner_indices = by_winner[_group_starts_device(keys, by_winner)]

    # Both arrays are currently in key order.  Restore dictionary insertion
    # order using the group's first source occurrence.
    group_order = torch.argsort(first_indices, stable=True)
    first_indices = first_indices[group_order]
    winner_indices = winner_indices[group_order]
    return DedupResult(
        batch=take_batch(batch, winner_indices),
        winner_indices=winner_indices,
        first_indices=first_indices,
    )


def _validate_rank_delta_reference(rank_delta: Any, n: int) -> Any:
    """RANK-ONLY selection bias (in-beam state lane).

    ``None`` preserves the exact legacy selection code path.  When present it
    may reorder ONLY the within-stratum top-K membership; band pruning and
    every propagated score stay on the original ``score`` channel, so
    band-margin dilution (the gtD3s regression mechanism) is impossible
    by construction, and selection is exactly the identity whenever a
    stratum's band-surviving row count does not exceed the cap.
    """

    if rank_delta is None:
        return None
    value = np.asarray(rank_delta)
    if value.shape != (n,) or value.dtype != np.float64:
        raise ValueError("rank_delta must be float64[N]")
    if not np.isfinite(value).all():
        raise ValueError("rank_delta must be finite")
    return value


def select_reference(
    batch: BeamBatch, *, band: float, beam_k: int, rank_delta: Any = None
) -> SelectionResult:
    """Apply the ordinary global band, required lane, and K-per-OM cut.

    Output order matches the legacy dictionaries:

    1. stable descending score order;
    2. in-band rows, then off-band required rows;
    3. sorted OM strata, first K rows per stratum; and
    4. required rows lost to K appended in global ranked order.

    ``rank_delta`` (optional ``float64[N]``, rank-only, score-preserving):
    within each OM stratum the top-K MEMBERSHIP is chosen by
    ``score + rank_delta`` descending (ties resolved by the legacy kept
    order), but survivors are emitted in the legacy order and carry their
    ORIGINAL scores.  A stratum whose kept count is <= K is byte-identical to
    the legacy cut regardless of ``rank_delta``.
    """

    _validate_reference_batch(batch)
    if beam_k < 0:
        raise ValueError("beam_k must be non-negative")
    rank_delta = _validate_rank_delta_reference(rank_delta, len(batch))
    ranked = np.argsort(-batch.score, kind="stable").astype(np.int64, copy=False)
    if not len(ranked):
        return SelectionResult(take_batch(batch, ranked), ranked, ranked, ranked)
    in_band = batch.score[ranked] >= batch.score[ranked[0]] - float(band)
    kept = np.concatenate((ranked[in_band], ranked[~in_band & batch.req[ranked]]))
    strata_keys = sorted(int(value) for value in np.unique(batch.oi[kept]))
    k_eff = int(beam_k)
    if stratum_redistribution_enabled():
        k_eff = effective_beam_k(beam_k, len(strata_keys))
    capped_parts = []
    for oi in strata_keys:
        stratum = kept[batch.oi[kept] == oi]
        if rank_delta is None or len(stratum) <= k_eff:
            capped_parts.append(stratum[:k_eff])
            continue
        adjusted = batch.score[stratum] + rank_delta[stratum]
        order = np.argsort(-adjusted, kind="stable")
        survivor = np.zeros(len(stratum), dtype=np.bool_)
        survivor[order[:k_eff]] = True
        capped_parts.append(stratum[survivor])
    capped = np.concatenate(capped_parts) if capped_parts else np.empty(0, dtype=np.int64)
    present = np.zeros(len(batch), dtype=np.bool_)
    present[capped] = True
    missing_required = ranked[batch.req[ranked] & ~present[ranked]]
    selected = np.concatenate((capped, missing_required))
    return SelectionResult(
        batch=take_batch(batch, selected),
        selected_indices=selected,
        ranked_indices=ranked,
        kept_indices=kept,
    )


def select_device(
    batch: BeamBatch, *, band: float, beam_k: int, rank_delta: Any = None
) -> SelectionResult:
    """CUDA-resident equivalent of :func:`select_reference`.

    ``rank_delta`` follows the reference contract exactly: rank-only,
    score-preserving, legacy emission order, identity when the cap does not
    bind.  ``None`` executes the untouched legacy code path.
    """

    torch = _validate_device_batch(batch)
    if beam_k < 0:
        raise ValueError("beam_k must be non-negative")
    if rank_delta is not None and (
        not isinstance(rank_delta, torch.Tensor)
        or not rank_delta.is_cuda
        or rank_delta.device != batch.states.device
        or rank_delta.shape != (len(batch),)
        or rank_delta.dtype != torch.float64
    ):
        raise ValueError("rank_delta must be torch.float64[N] on beam device")
    ranked = torch.argsort(batch.score, descending=True, stable=True)
    if int(ranked.shape[0]) == 0:
        return SelectionResult(take_batch(batch, ranked), ranked, ranked, ranked)
    ranked_req = batch.req[ranked]
    in_band = batch.score[ranked] >= batch.score[ranked[0]] - float(band)
    kept = torch.cat((ranked[in_band], ranked[~in_band & ranked_req]))

    by_oi_order = torch.argsort(batch.oi[kept], stable=True)
    by_oi = kept[by_oi_order]
    if int(by_oi.shape[0]) == 0:
        capped = by_oi
    else:
        starts = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=by_oi.device),
                batch.oi[by_oi[1:]] != batch.oi[by_oi[:-1]],
            )
        )
        start_positions = torch.nonzero(starts, as_tuple=False).flatten()
        ends = torch.cat(
            (
                start_positions[1:],
                start_positions.new_full((1,), len(by_oi)),
            )
        )
        lengths = ends - start_positions
        row_starts = torch.repeat_interleave(start_positions, lengths)
        within = torch.arange(len(by_oi), dtype=torch.int64, device=by_oi.device)
        k_eff = int(beam_k)
        if stratum_redistribution_enabled():
            # ``start_positions``'s length is host-known already (``nonzero``
            # synchronizes); this adds no extra D2H transfer.
            k_eff = effective_beam_k(beam_k, int(start_positions.shape[0]))
        if rank_delta is None:
            capped = by_oi[within - row_starts < k_eff]
        else:
            # Rank-only membership: order ``by_oi`` positions by
            # (oi asc, score+rank_delta desc, legacy kept position asc).  The
            # oi runs of that ordering share ``start_positions``/``lengths``
            # with the legacy ordering (same sorted oi multiset), so the
            # legacy within-run cut formula selects the top-K MEMBERSHIP per
            # stratum; scattering back to ``by_oi`` positions emits survivors
            # in the untouched legacy order with their original scores.
            adjusted = (batch.score + rank_delta)[by_oi]
            by_adjusted = torch.argsort(adjusted, descending=True, stable=True)
            by_stratum = by_adjusted[
                torch.argsort(batch.oi[by_oi][by_adjusted], stable=True)
            ]
            survivor = torch.zeros(
                len(by_oi), dtype=torch.bool, device=by_oi.device
            )
            survivor[by_stratum[within - row_starts < k_eff]] = True
            capped = by_oi[survivor]

    present = torch.zeros(len(batch), dtype=torch.bool, device=batch.states.device)
    present[capped] = True
    missing_required = ranked[ranked_req & ~present[ranked]]
    selected = torch.cat((capped, missing_required))
    return SelectionResult(
        batch=take_batch(batch, selected),
        selected_indices=selected,
        ranked_indices=ranked,
        kept_indices=kept,
    )


def beam_step_reference(
    batch: BeamBatch,
    perms: np.ndarray,
    *,
    skip_cap: int,
    insert_cap: int,
    band: float,
    beam_k: int,
    required_next: Any = None,
    single_score_delta: Any = None,
    live_action_score_delta: Any = None,
    allow_actions: bool = True,
    template: BranchTemplate | None = None,
) -> SelectionResult:
    """Convenience composition for one CPU-reference ordinary beam step."""

    expanded = expand_reference(
        batch,
        perms,
        skip_cap=skip_cap,
        insert_cap=insert_cap,
        required_next=required_next,
        single_score_delta=single_score_delta,
        live_action_score_delta=live_action_score_delta,
        allow_actions=allow_actions,
        template=template,
    )
    return select_reference(stable_dedup_reference(expanded).batch, band=band, beam_k=beam_k)


def beam_step_device(
    batch: BeamBatch,
    perms: Any,
    *,
    skip_cap: int,
    insert_cap: int,
    band: float,
    beam_k: int,
    required_next: Any = None,
    single_score_delta: Any = None,
    live_action_score_delta: Any = None,
    allow_actions: bool = True,
    template: BranchTemplate | None = None,
) -> SelectionResult:
    """One fully CUDA-resident ordinary beam step; returns survivor ancestry."""

    expanded = expand_device(
        batch,
        perms,
        skip_cap=skip_cap,
        insert_cap=insert_cap,
        required_next=required_next,
        single_score_delta=single_score_delta,
        live_action_score_delta=live_action_score_delta,
        allow_actions=allow_actions,
        template=template,
    )
    return select_device(stable_dedup_device(expanded).batch, band=band, beam_k=beam_k)
