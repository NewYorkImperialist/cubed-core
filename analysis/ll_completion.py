"""Replay-verified last-layer completion from the bundled OLL/PLL tables.

This module is deliberately a pure, deterministic enumerator.  It does not
choose a suffix, score visual evidence, mutate a decoder, or claim that a solve
ended.  Given an exact cube state whose first two layers are complete, it:

* recognizes the matching OLL and PLL cases;
* enumerates the finite AUF x whole-cube-y x matching-table-algorithm paths;
* preserves the literal rotations, wide moves, and slice moves in each path;
* maps those paths from the cross-down table frame back to the caller's frame;
* returns a path only after replaying it from the caller's state and proving
  that the resulting cube is solved.

The later perception/trellis layer can therefore score complete trajectories
against evidence.  An empty result is an abstention, never an inferred solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Literal, Sequence

import numpy as np

from analysis import cases as cs
from analysis import cubemodel as cm
from analysis import segment as sg

Phase = Literal["oll", "pll"]

_AUFS: tuple[tuple[str, ...], ...] = ((), ("U",), ("U2",), ("U'",))
_Y_ROTATIONS: tuple[tuple[str, ...], ...] = ((), ("y",), ("y2",), ("y'",))


@dataclass(frozen=True)
class LLStage:
    """One table stage, with both table-frame and caller-frame notation.

    ``frame_*`` fields use the conventional cross-down OLL/PLL table frame.
    ``physical_*`` fields are the same literal actions conjugated into the
    caller's frame.  Whole-cube rotations remain rotations; they are never
    rewritten as face turns or silently discarded.
    """

    phase: Phase
    case_id: str
    case_name: str | None
    skipped: bool
    frame_pre_auf: tuple[str, ...]
    frame_y_rotation: tuple[str, ...]
    table_algorithm: tuple[str, ...]
    frame_post_auf: tuple[str, ...]
    physical_pre_auf: tuple[str, ...]
    physical_y_rotation: tuple[str, ...]
    physical_algorithm: tuple[str, ...]
    physical_post_auf: tuple[str, ...]

    @property
    def frame_moves(self) -> tuple[str, ...]:
        return (
            self.frame_pre_auf
            + self.frame_y_rotation
            + self.table_algorithm
            + self.frame_post_auf
        )

    @property
    def physical_moves(self) -> tuple[str, ...]:
        return (
            self.physical_pre_auf
            + self.physical_y_rotation
            + self.physical_algorithm
            + self.physical_post_auf
        )


@dataclass(frozen=True)
class LLTrajectory:
    """A complete, replay-verified OLL+PLL trajectory."""

    cross_color: str
    frame_moves: tuple[str, ...]
    moves: tuple[str, ...]
    oll: LLStage
    pll: LLStage


@dataclass(frozen=True)
class LLCanonicalStep:
    """Exact canonical representation of one literal physical action.

    ``canonical_moves`` contains only outer-face turns.  The relative held
    orientation is the whole-cube orientation *after* the action.  A rotation
    therefore emits no canonical move, a wide move emits one, and a slice move
    emits two.  The combination is permutation-identical to ``literal_move``;
    this is an algebraic factorization, not notation substitution.
    """

    literal_move: str
    canonical_moves: tuple[str, ...]
    relative_orientation: tuple[str, ...]
    canonical_start: int
    canonical_end: int


@dataclass(frozen=True)
class LLCanonicalTrace:
    """Canonical face path plus held-orientation sidecar for an LL trajectory."""

    literal_moves: tuple[str, ...]
    canonical_moves: tuple[str, ...]
    steps: tuple[LLCanonicalStep, ...]
    final_relative_orientation: tuple[str, ...]


@dataclass(frozen=True)
class LLCoverageGap:
    """Why a table row or entry state could not produce a verified path."""

    phase: str
    case_id: str | None
    reason: str
    algorithm: str | None = None


def _gap_sort_key(gap: LLCoverageGap) -> tuple[str, str, str, str]:
    return (gap.phase, gap.case_id or "", gap.reason, gap.algorithm or "")


@dataclass(frozen=True)
class LLCompletionResult:
    """Finite candidate set, or an explicit abstention."""

    candidates: tuple[LLTrajectory, ...]
    cross_color: str | None
    entry_level: int | None
    gaps: tuple[LLCoverageGap, ...]
    abstain_reason: str | None

    @property
    def abstained(self) -> bool:
        return not self.candidates


@dataclass(frozen=True)
class LLTableCoverageReport:
    """Semantic integrity report for the bundled recognition tables."""

    oll_cases: int
    pll_cases: int
    oll_algorithms: int
    pll_algorithms: int
    uncovered_oll_cases: tuple[str, ...]
    uncovered_pll_cases: tuple[str, ...]
    invalid_algorithms: tuple[LLCoverageGap, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.uncovered_oll_cases
            or self.uncovered_pll_cases
            or self.invalid_algorithms
        )


@dataclass(frozen=True)
class _OLLPartial:
    state: np.ndarray
    stage: LLStage


def _coerce_state(state: Sequence[int] | np.ndarray) -> np.ndarray | None:
    try:
        raw = np.asarray(state)
    except (TypeError, ValueError, OverflowError):
        return None
    if raw.shape != (54,) or not np.issubdtype(raw.dtype, np.integer):
        return None
    if np.any((raw < 0) | (raw >= 6)):
        return None
    arr = raw.astype(np.int8, copy=True)
    counts = np.bincount(arr.astype(np.int64), minlength=6)
    if len(counts) != 6 or not np.array_equal(counts, np.full(6, 9)):
        return None
    return arr


def _normalized_algorithm(algorithm: str) -> tuple[str, ...]:
    return tuple(cm.norm_token(token) for token in cm.validate_scramble(algorithm))


def _cross_down(state: np.ndarray, cross_color: str) -> np.ndarray:
    return state[cm.orientation_with_down(state, cross_color)]


def _top_oriented(state: np.ndarray, cross_color: str) -> bool:
    faces = _cross_down(state, cross_color).reshape(6, 9)
    return bool((faces[0] == faces[0, 4]).all())


def _token_category(token: str) -> str:
    base = cm.token_base(token)
    if base in cm.FACE_BASES:
        return "face"
    if base in cm.SLICE_BASES:
        return "slice"
    if base in cm.WIDE_BASES:
        return "wide"
    if base in cm.ROTATION_BASES:
        return "rotation"
    raise ValueError(f"unsupported cube token {token!r}")


@lru_cache(maxsize=1)
def _canonical_face_words() -> dict[bytes, tuple[str, ...]]:
    """Shortest deterministic outer-face word for every depth-0..2 perm."""
    face_tokens = tuple(
        base + modifier
        for base in cm.FACE_BASES
        for modifier in cm.MODIFIERS
    )
    words = [()]
    words.extend((token,) for token in face_tokens)
    words.extend(product(face_tokens, repeat=2))
    by_perm: dict[bytes, tuple[str, ...]] = {}
    for word in words:
        candidate = tuple(word)
        key = cm.net_perm(candidate).tobytes()
        incumbent = by_perm.get(key)
        if incumbent is None or (len(candidate), candidate) < (len(incumbent), incumbent):
            by_perm[key] = candidate
    return by_perm


@lru_cache(maxsize=1)
def _orientation_indices() -> dict[bytes, int]:
    return {
        orientation["perm"].tobytes(): index
        for index, orientation in enumerate(cm.ORIENTATIONS)
    }


@lru_cache(maxsize=None)
def _factor_literal_step(
    orientation_index: int,
    literal_move: str,
) -> tuple[tuple[str, ...], int]:
    """Factor ``W literal`` into ``canonical_faces W'`` exactly."""
    token = cm.norm_token(literal_move)
    if token not in cm.PERMS:
        raise ValueError(f"unsupported cube token {literal_move!r}")
    try:
        current = cm.ORIENTATIONS[orientation_index]["perm"]
    except IndexError as exc:
        raise ValueError(f"invalid orientation index {orientation_index}") from exc

    net = cm.compose(current, cm.PERMS[token])
    hits: list[tuple[int, tuple[str, ...], int]] = []
    face_words = _canonical_face_words()
    for next_index, orientation in enumerate(cm.ORIENTATIONS):
        face_perm = cm.compose(net, np.argsort(orientation["perm"]))
        face_word = face_words.get(face_perm.tobytes())
        if face_word is not None:
            hits.append((len(face_word), face_word, next_index))
    if not hits:
        raise ValueError(
            f"literal move {literal_move!r} has no exact canonical/orientation factor"
        )

    best_length = min(hit[0] for hit in hits)
    best = [hit for hit in hits if hit[0] == best_length]
    if len(best) != 1:
        raise ValueError(
            f"literal move {literal_move!r} has an ambiguous minimal factor"
        )
    _length, face_word, next_index = best[0]
    return face_word, next_index


def canonicalize_ll_moves(moves: Sequence[str]) -> LLCanonicalTrace:
    """Factor literal LL actions into production-canonical face moves.

    The returned face path is safe for the canonical decoder state.  Rotations
    remain represented in each step's orientation sidecar, so they cannot turn
    into fake puzzle moves.  Every step is derived from exact 54-facelet
    permutation equality and the minimal factor is unique for the supported
    cube vocabulary.
    """
    orientation_index = _orientation_indices()[cm.IDENTITY.tobytes()]
    canonical_moves: list[str] = []
    steps: list[LLCanonicalStep] = []
    literal_moves = tuple(cm.norm_token(token) for token in moves)
    for literal_move in literal_moves:
        start = len(canonical_moves)
        face_moves, orientation_index = _factor_literal_step(
            orientation_index,
            literal_move,
        )
        canonical_moves.extend(face_moves)
        end = len(canonical_moves)
        steps.append(LLCanonicalStep(
            literal_move=literal_move,
            canonical_moves=face_moves,
            relative_orientation=tuple(
                cm.ORIENTATIONS[orientation_index]["word"]
            ),
            canonical_start=start,
            canonical_end=end,
        ))

    final_orientation = tuple(cm.ORIENTATIONS[orientation_index]["word"])
    # Guard the public boundary even though every cached step was constructed
    # from this identity.  A future vocabulary/orientation change must fail
    # here instead of silently changing the puzzle path.
    literal_perm = cm.net_perm(literal_moves)
    factored_perm = cm.compose(
        cm.net_perm(canonical_moves),
        cm.ORIENTATIONS[orientation_index]["perm"],
    )
    if not np.array_equal(literal_perm, factored_perm):
        raise ValueError("LL canonical factorization does not preserve permutation")
    return LLCanonicalTrace(
        literal_moves=literal_moves,
        canonical_moves=tuple(canonical_moves),
        steps=tuple(steps),
        final_relative_orientation=final_orientation,
    )


def canonicalize_ll_trajectory(trajectory: LLTrajectory) -> LLCanonicalTrace:
    """Canonicalize the literal actions of a replay-verified trajectory."""
    return canonicalize_ll_moves(trajectory.moves)


def _caller_token_map(orientation: np.ndarray) -> dict[str, str]:
    """Map table-frame tokens into the caller frame by exact conjugation."""
    inverse = np.argsort(orientation)
    reverse: dict[str, dict[bytes, str]] = {
        "face": {}, "slice": {}, "wide": {}, "rotation": {},
    }
    for token, perm in cm.PERMS.items():
        reverse[_token_category(token)][perm.tobytes()] = token

    out = {}
    for token, perm in cm.PERMS.items():
        conjugate = cm.compose(cm.compose(orientation, perm), inverse)
        mapped = reverse[_token_category(token)].get(conjugate.tobytes())
        if mapped is None:
            raise ValueError(f"cannot map {token!r} into the caller frame")
        out[token] = mapped
    return out


def _map_tokens(tokens: tuple[str, ...], token_map: dict[str, str]) -> tuple[str, ...]:
    return tuple(token_map[token] for token in tokens)


def _stage(
    phase: Phase,
    case_id: str,
    case_name: str | None,
    *,
    skipped: bool,
    pre_auf: tuple[str, ...] = (),
    y_rotation: tuple[str, ...] = (),
    algorithm: tuple[str, ...] = (),
    post_auf: tuple[str, ...] = (),
    token_map: dict[str, str],
) -> LLStage:
    return LLStage(
        phase=phase,
        case_id=case_id,
        case_name=case_name,
        skipped=skipped,
        frame_pre_auf=pre_auf,
        frame_y_rotation=y_rotation,
        table_algorithm=algorithm,
        frame_post_auf=post_auf,
        physical_pre_auf=_map_tokens(pre_auf, token_map),
        physical_y_rotation=_map_tokens(y_rotation, token_map),
        physical_algorithm=_map_tokens(algorithm, token_map),
        physical_post_auf=_map_tokens(post_auf, token_map),
    )


def _skip_stage(phase: Phase, token_map: dict[str, str]) -> LLStage:
    label = phase.upper()
    return _stage(
        phase,
        f"{label}-skip",
        f"{label} skip",
        skipped=True,
        token_map=token_map,
    )


def _case_algorithms(
    phase: Phase,
    case: dict,
    gaps: set[LLCoverageGap],
) -> tuple[tuple[str, ...], ...]:
    algorithms = []
    for raw in case.get("algs") or ():
        try:
            parsed = _normalized_algorithm(raw)
        except (KeyError, ValueError) as exc:
            gaps.add(LLCoverageGap(phase, case.get("id"), f"invalid notation: {exc}", raw))
            continue
        if parsed:
            algorithms.append(parsed)
        else:
            gaps.add(LLCoverageGap(phase, case.get("id"), "empty algorithm", raw))
    return tuple(algorithms)


def _oll_partials(
    entry: np.ndarray,
    cross_color: str,
    token_map: dict[str, str],
    gaps: set[LLCoverageGap],
) -> tuple[_OLLPartial, ...]:
    if _top_oriented(entry, cross_color):
        return (_OLLPartial(entry, _skip_stage("oll", token_map)),)

    case = cs.identify_oll(_cross_down(entry, cross_color))
    case_id = case.get("id")
    if case_id is None:
        gaps.add(LLCoverageGap("oll", None, "entry pattern is not in the OLL table"))
        return ()

    algorithms = _case_algorithms("oll", case, gaps)
    partials = []
    seen = set()
    for algorithm in algorithms:
        for pre_auf in _AUFS:
            for y_rotation in _Y_ROTATIONS:
                frame_moves = pre_auf + y_rotation + algorithm
                after = cm.apply_seq(entry, frame_moves)
                level, _ = sg.progress(after, lock_color=cross_color)
                if level > 1 or not _top_oriented(after, cross_color):
                    continue
                stage = _stage(
                    "oll",
                    case_id,
                    case.get("name"),
                    skipped=False,
                    pre_auf=pre_auf,
                    y_rotation=y_rotation,
                    algorithm=algorithm,
                    token_map=token_map,
                )
                key = (after.tobytes(), stage.frame_moves)
                if key not in seen:
                    seen.add(key)
                    partials.append(_OLLPartial(after, stage))
    if not partials:
        gaps.add(LLCoverageGap("oll", case_id, "no replay-verified OLL algorithm"))
    return tuple(partials)


def _pll_stages(
    state: np.ndarray,
    cross_color: str,
    token_map: dict[str, str],
    gaps: set[LLCoverageGap],
) -> tuple[tuple[np.ndarray, LLStage], ...]:
    # The solve ends when the entry is already solved.  Do not manufacture
    # orientation-only moves after that endpoint merely because is_solved() is
    # whole-cube-orientation invariant.
    if cm.is_solved(state):
        return ((state, _skip_stage("pll", token_map)),)

    solved = []
    # A PLL skip may still need a final AUF.  Preserve a real y regrip when it
    # appears: solvedness is orientation-independent, but the evidence is not.
    for y_rotation in _Y_ROTATIONS:
        for post_auf in _AUFS:
            frame_moves = y_rotation + post_auf
            after = cm.apply_seq(state, frame_moves)
            if cm.is_solved(after):
                solved.append((
                    after,
                    _stage(
                        "pll",
                        "PLL-skip",
                        "PLL skip",
                        skipped=True,
                        y_rotation=y_rotation,
                        post_auf=post_auf,
                        token_map=token_map,
                    ),
                ))
    if solved:
        return tuple(solved)

    case = cs.identify_pll(_cross_down(state, cross_color))
    case_id = case.get("id")
    if case_id is None:
        gaps.add(LLCoverageGap("pll", None, "post-OLL pattern is not in the PLL table"))
        return ()

    algorithms = _case_algorithms("pll", case, gaps)
    seen = set()
    for algorithm in algorithms:
        for pre_auf in _AUFS:
            for y_rotation in _Y_ROTATIONS:
                prefix = pre_auf + y_rotation + algorithm
                before_post = cm.apply_seq(state, prefix)
                for post_auf in _AUFS:
                    after = cm.apply_seq(before_post, post_auf)
                    if not cm.is_solved(after):
                        continue
                    stage = _stage(
                        "pll",
                        case_id,
                        case.get("name"),
                        skipped=False,
                        pre_auf=pre_auf,
                        y_rotation=y_rotation,
                        algorithm=algorithm,
                        post_auf=post_auf,
                        token_map=token_map,
                    )
                    key = stage.frame_moves
                    if key not in seen:
                        seen.add(key)
                        solved.append((after, stage))
    if not solved:
        gaps.add(LLCoverageGap("pll", case_id, "no replay-verified PLL algorithm"))
    return tuple(solved)


def enumerate_ll_completions(state: Sequence[int] | np.ndarray) -> LLCompletionResult:
    """Enumerate every matching-table LL suffix that exactly solves ``state``.

    There are no confidence constants, search beams, candidate caps, or tag
    settings.  The finite vocabulary is the recognized OLL row followed by the
    recognized PLL row, each under all four AUFs and four whole-cube y grips.
    The single AUF between algorithms is represented as the PLL pre-AUF (rather
    than redundantly enumerating OLL post-AUF x PLL pre-AUF equivalents).
    Every returned caller-frame trajectory has passed an independent full replay.
    """
    entry = _coerce_state(state)
    if entry is None:
        return LLCompletionResult((), None, None, (), "invalid_state")

    entry_level, cross_color = sg.progress(entry)
    if cross_color is None or entry_level > 2:
        return LLCompletionResult((), cross_color, entry_level, (), "not_f2l_complete")

    orientation = cm.orientation_with_down(entry, cross_color)
    frame_entry = entry[orientation]
    token_map = _caller_token_map(orientation)
    gaps: set[LLCoverageGap] = set()
    trajectories = []
    seen_moves = set()

    for partial in _oll_partials(frame_entry, cross_color, token_map, gaps):
        for _after, pll_stage in _pll_stages(partial.state, cross_color, token_map, gaps):
            frame_moves = partial.stage.frame_moves + pll_stage.frame_moves
            physical_moves = partial.stage.physical_moves + pll_stage.physical_moves
            # This replay is intentionally against the original caller state,
            # not the intermediate table-frame state.  It is the final guard
            # against an orientation-conjugation or metadata-wiring mistake.
            if not cm.is_solved(cm.apply_seq(entry, physical_moves)):
                gaps.add(LLCoverageGap("path", None, "caller-frame replay did not solve"))
                continue
            if physical_moves in seen_moves:
                continue
            seen_moves.add(physical_moves)
            trajectories.append(LLTrajectory(
                cross_color=cross_color,
                frame_moves=frame_moves,
                moves=physical_moves,
                oll=partial.stage,
                pll=pll_stage,
            ))

    trajectories.sort(key=lambda trajectory: (len(trajectory.moves), trajectory.moves))
    candidates = tuple(trajectories)
    return LLCompletionResult(
        candidates=candidates,
        cross_color=cross_color,
        entry_level=entry_level,
        gaps=tuple(sorted(gaps, key=_gap_sort_key)),
        abstain_reason=None if candidates else "no_replay_verified_table_path",
    )


def replay_trajectory(
    state: Sequence[int] | np.ndarray,
    trajectory: LLTrajectory,
) -> tuple[np.ndarray, ...]:
    """Return the caller-frame state trace, including entry and final state."""
    entry = _coerce_state(state)
    if entry is None:
        raise ValueError("state must be a 54-facelet cube with nine stickers per color")
    trace = [entry]
    for token in trajectory.moves:
        trace.append(cm.apply(trace[-1], token))
    return tuple(trace)


@lru_cache(maxsize=1)
def table_coverage_report() -> LLTableCoverageReport:
    """Audit syntax, phase semantics, recognition, and replay for every table alg.

    Each algorithm is inverted from SOLVED to construct its own representative
    entry state.  A variant is valid only when that state has F2L complete, is
    in the expected LL phase, recognizes as the declared case, and replays to
    solved.  Case-level gaps are reported separately from invalid variants.
    """
    invalid = []
    covered: dict[Phase, set[str]] = {"oll": set(), "pll": set()}
    algorithm_counts: dict[Phase, int] = {"oll": 0, "pll": 0}

    for phase in ("oll", "pll"):
        for case in cs._load_table(phase).values():
            case_id = case["id"]
            for raw in case.get("algs") or ():
                algorithm_counts[phase] += 1
                reasons = []
                try:
                    algorithm = _normalized_algorithm(raw)
                    before = cm.apply_seq(cm.SOLVED, cm.invert(algorithm))
                except (KeyError, ValueError) as exc:
                    invalid.append(LLCoverageGap(phase, case_id, f"invalid notation: {exc}", raw))
                    continue

                level, cross_color = sg.progress(before)
                if cross_color is None or level > 2:
                    reasons.append("inverse state is not F2L-complete")
                else:
                    frame = _cross_down(before, cross_color)
                    if phase == "oll":
                        if _top_oriented(before, cross_color):
                            reasons.append("inverse state is already OLL-complete")
                        recognized = cs.identify_oll(frame).get("id")
                    else:
                        if not _top_oriented(before, cross_color):
                            reasons.append("inverse state is not OLL-complete")
                        recognized = cs.identify_pll(frame).get("id")
                    if recognized != case_id:
                        reasons.append(f"recognizes as {recognized!r}")
                if not cm.is_solved(cm.apply_seq(before, algorithm)):
                    reasons.append("algorithm replay does not solve its inverse")

                if reasons:
                    invalid.append(LLCoverageGap(phase, case_id, "; ".join(reasons), raw))
                else:
                    covered[phase].add(case_id)

    oll_table = cs._load_table("oll")
    pll_table = cs._load_table("pll")
    oll_case_ids = {case["id"] for case in oll_table.values()}
    pll_case_ids = {case["id"] for case in pll_table.values()}
    return LLTableCoverageReport(
        oll_cases=len(oll_table),
        pll_cases=len(pll_table),
        oll_algorithms=algorithm_counts["oll"],
        pll_algorithms=algorithm_counts["pll"],
        uncovered_oll_cases=tuple(sorted(oll_case_ids - covered["oll"])),
        uncovered_pll_cases=tuple(sorted(pll_case_ids - covered["pll"])),
        invalid_algorithms=tuple(sorted(invalid, key=_gap_sort_key)),
    )
