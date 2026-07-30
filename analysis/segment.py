"""CFOP phase segmentation via a monotone progress ladder.

Design (ported concept from csTimer's recons, own implementation):
- progress(state) = min level over the 6 "which face is down" reorientations,
  so cross detection is color- and face-neutral by construction.
- Levels: 7 nothing, 6..3 = 2 + #unsolved F2L slots once a cross exists,
  2 = F2L done, 1 = OLL done, 0 = solved.
- A phase closes the first time the level drops below its best-so-far; deeper
  multi-level drops emit explicit zero-move "skipped" phases (X-cross,
  OLL/PLL skips). Transient regressions (R turns break the cross mid-pair)
  never create boundaries because only drops below the best level count.
- The cross color is locked at the first boundary; later progress is
  evaluated for that color only (a solver switching cross colors mid-solve
  is out of scope).
- AUF rule: a 1-move oll/pll phase consisting of a single U-layer turn is
  absorbed into the previous non-skipped phase and marked skipped (e.g.
  "OLL done -> U -> solved" records a PLL skip with the U attributed to OLL).
  Narrower than csTimer's any-1-move-phase merge: 1-move F2L pair
  completions are real and must not be merged away.
"""

from dataclasses import dataclass

from analysis import cubemodel as cm

# phase completed by reaching a given level
PHASE_BY_TARGET = {6: "cross", 5: "f2l1", 4: "f2l2", 3: "f2l3", 2: "f2l4", 1: "oll", 0: "pll"}
PHASE_ORDER = ["cross", "f2l1", "f2l2", "f2l3", "f2l4", "oll", "pll"]

# Canonical-frame facelet index pairs (cross face = down).
# Array layout: up 0-8, right 9-17, front 18-26, down 27-35, left 36-44, back 45-53.
# Each pair (sticker, center-of-its-face) must match.
CROSS_PAIRS = (
    (28, 31), (30, 31), (32, 31), (34, 31),  # down edges
    (25, 22), (16, 13), (43, 40), (52, 49),  # their side stickers
)
SLOT_PAIRS = {
    "FR": ((29, 31), (26, 22), (15, 13), (23, 22), (12, 13)),
    "FL": ((27, 31), (24, 22), (44, 40), (21, 22), (41, 40)),
    "BR": ((35, 31), (17, 13), (51, 49), (14, 13), (48, 49)),
    "BL": ((33, 31), (42, 40), (53, 49), (39, 40), (50, 49)),
}

# One reorientation per choice of down face (predicates are y-invariant).
DOWN_PERMS = [o["perm"] for o in cm.ORIENTATIONS if "y" not in o["word"]]
assert len(DOWN_PERMS) == 6


def _pairs_ok(r, pairs):
    return all(r[a] == r[b] for a, b in pairs)


def level_in_frame(r):
    """Progress level with the cross face fixed to down in state r."""
    if not _pairs_ok(r, CROSS_PAIRS):
        return 7
    slots = sum(_pairs_ok(r, p) for p in SLOT_PAIRS.values())
    if slots < 4:
        return 6 - slots
    if not (r[0:9] == r[4]).all():
        return 2
    if not cm.is_solved(r):
        return 1
    return 0


def progress(state, lock_color=None):
    """(best level, cross color achieving it). With lock_color, only the
    orientation whose down center is that color is evaluated."""
    best, best_color = 8, None
    for perm in DOWN_PERMS:
        r = state[perm]
        color = cm.COLOR_LIST[r[31]]
        if lock_color is not None and color != lock_color:
            continue
        lv = level_in_frame(r)
        if lv < best:
            best, best_color = lv, color
    return best, best_color


@dataclass
class PhaseSpan:
    name: str
    skipped: bool
    start: int  # step index range [start, end) into the solution
    end: int


@dataclass
class SegmentResult:
    spans: list
    solved: bool
    cross_color: str | None
    level_reached: int  # 0 when solved
    last_progress_move: int | None  # index of the move that last advanced progress
    trailing_start: int | None  # index where unattributed trailing moves begin


def segment(start_state, steps):
    spans = []
    best, color = progress(start_state)
    lock = color if best <= 6 else None

    # levels already satisfied by the scrambled state -> skipped phases at t=0
    for tgt in range(6, best - 1, -1):
        spans.append(PhaseSpan(PHASE_BY_TARGET[tgt], True, 0, 0))

    seg_start = 0
    last_progress = None
    for i, st in enumerate(steps):
        cur, ccolor = progress(st.state, lock)
        if cur < best:
            if lock is None:
                lock = ccolor
            spans.append(PhaseSpan(PHASE_BY_TARGET[best - 1], False, seg_start, i + 1))
            for tgt in range(best - 2, cur - 1, -1):
                spans.append(PhaseSpan(PHASE_BY_TARGET[tgt], True, i + 1, i + 1))
            best = cur
            seg_start = i + 1
            last_progress = i
            if best == 0:
                break

    _merge_auf(spans, steps)

    solved = best == 0
    trailing = None
    if not solved and seg_start < len(steps):
        trailing = seg_start
    return SegmentResult(
        spans=spans,
        solved=solved,
        cross_color=lock,
        level_reached=best,
        last_progress_move=last_progress,
        trailing_start=trailing,
    )


def _merge_auf(spans, steps):
    for k, sp in enumerate(spans):
        if sp.skipped or sp.name not in ("oll", "pll") or sp.end - sp.start != 1:
            continue
        if cm.token_base(steps[sp.start].token) != "U":
            continue
        for j in range(k - 1, -1, -1):
            if not spans[j].skipped:
                spans[j].end = sp.end
                sp.skipped = True
                sp.start = sp.end
                break
