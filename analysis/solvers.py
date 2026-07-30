"""Optimal cross and x-cross solvers.

Cross: the four cross edges occupy P(12,4) * 2^4 = 190,080 states; a full
BFS distance table over the 18 face moves is built once per process
(numpy-vectorized, well under a second) and reused. Optimal cross is always
<= 8 moves. Solutions are reconstructed by greedy descent on the table.

X-cross: IDA* over cross + one corner/edge pair, using the cross table as
the admissible heuristic, capped by a node budget.

All solving happens in a reoriented frame (target cross color on down);
solutions are renamed back to the caller's frame via conjugation, so the
returned moves apply directly to the input state.
"""

import numpy as np

from analysis import cubemodel as cm

FACE_TOKENS = [
    b + m for b in ("R", "U", "F", "D", "L", "B") for m in ("", "'", "2")
]

# sticker-position pairs for the 12 edge slots (canonical layout indices)
EDGE_SLOTS = [
    (7, 19),  # UF
    (5, 10),  # UR
    (1, 46),  # UB
    (3, 37),  # UL
    (28, 25),  # DF
    (32, 16),  # DR
    (34, 52),  # DB
    (30, 43),  # DL
    (23, 12),  # FR
    (21, 41),  # FL
    (48, 14),  # BR
    (50, 39),  # BL
]

# corner slots as (down/up sticker, sticker on first face, sticker on second face)
CORNER_SLOTS = {
    "FR": (29, 26, 15),  # D-FR corner: down[2], front[8], right[6]
    "FL": (27, 24, 44),
    "BR": (35, 51, 17),
    "BL": (33, 42, 53),
}
# the middle-layer edge belonging to each slot
SLOT_EDGES = {"FR": (23, 12), "FL": (21, 41), "BR": (48, 14), "BL": (50, 39)}

# down-face cross sticker positions paired with their side stickers
CROSS_EDGE_HOMES = [(28, 25), (32, 16), (34, 52), (30, 43)]  # DF DR DB DL

_inv_perms = None
_move_perms = None
_cross_dist = None
_pair_dist = {}          # slot -> {encoded 5-position state: optimal moves}
_last_nodes = 0          # IDA* nodes expanded by the most recent _ida call
_W5 = 54 ** np.arange(5, dtype=np.int64)


def _tables():
    """(move_perms, inv_perms) for the 18 face moves as (18, 54) arrays."""
    global _move_perms, _inv_perms
    if _move_perms is None:
        _move_perms = np.stack([cm.PERMS[t] for t in FACE_TOKENS])
        # new[i] = old[perm[i]]  =>  a sticker at position p lands at inv[p]
        _inv_perms = np.argsort(_move_perms, axis=1)
    return _move_perms, _inv_perms


_WEIGHTS = (54 ** np.arange(8, dtype=np.int64)).reshape(1, 8)


def _encode(positions):
    return (positions.astype(np.int64) @ _WEIGHTS.T).ravel()


def cross_distance_table():
    """dict: encoded 8-sticker-position state -> optimal move count (0..8)."""
    global _cross_dist
    if _cross_dist is not None:
        return _cross_dist
    _, inv = _tables()
    home = np.array([p for pair in CROSS_EDGE_HOMES for p in pair], dtype=np.intp)
    dist = {int(_encode(home.reshape(1, 8))[0]): 0}
    frontier = home.reshape(1, 8)
    d = 0
    while len(frontier):
        nxt = []
        for m in range(18):
            moved = inv[m][frontier]  # (N, 8)
            keys = _encode(moved)
            fresh = np.array([k not in dist for k in keys.tolist()])
            if fresh.any():
                moved, keys = moved[fresh], keys[fresh]
                # dedupe within this wave
                keys_list = keys.tolist()
                seen = set()
                keep = []
                for idx, k in enumerate(keys_list):
                    if k not in seen and k not in dist:
                        seen.add(k)
                        keep.append(idx)
                for idx in keep:
                    dist[int(keys_list[idx])] = d + 1
                nxt.append(moved[keep])
        frontier = np.concatenate(nxt) if nxt else np.empty((0, 8), dtype=np.intp)
        d += 1
    _cross_dist = dist
    return dist


def pair_distance_table(slot):
    """dict: encoded 5-sticker-position state (one corner + one edge of `slot`)
    -> optimal moves to seat that pair, ignoring everything else. The orbit is
    tiny (<=576 states), so this BFS is near-instant and cached per slot. Used
    as the admissible heuristic for the constrained per-pair insertion search
    (the cross table is useless there — it's ~0 once the cross is solved)."""
    if slot in _pair_dist:
        return _pair_dist[slot]
    _, inv = _tables()
    home = np.array(list(CORNER_SLOTS[slot]) + list(SLOT_EDGES[slot]), dtype=np.intp)
    dist = {int((home.astype(np.int64) * _W5).sum()): 0}
    frontier = home.reshape(1, 5)
    d = 0
    while len(frontier):
        nxt = []
        for m in range(18):
            moved = inv[m][frontier]
            keys = (moved.astype(np.int64) @ _W5)
            seen, keep = set(), []
            for idx, k in enumerate(keys.tolist()):
                if k not in dist and k not in seen:
                    seen.add(k)
                    keep.append(idx)
            for idx in keep:
                dist[int(keys[idx])] = d + 1
            if keep:
                nxt.append(moved[keep])
        frontier = np.concatenate(nxt) if nxt else np.empty((0, 5), dtype=np.intp)
        d += 1
    _pair_dist[slot] = dist
    return dist


def _cross_positions(state):
    """Positions of the 8 cross-edge stickers of `state` (cross color = down
    center), ordered to match CROSS_EDGE_HOMES. None if state is malformed."""
    cross_color = state[31]
    # side color each cross edge must pair with, in home order
    side_colors = [state[pair[1] // 9 * 9 + 4] for pair in CROSS_EDGE_HOMES]
    want = {sc: i for i, sc in enumerate(side_colors)}
    positions = np.full(8, -1, dtype=np.intp)
    for a, b in EDGE_SLOTS:
        ca, cb = state[a], state[b]
        if ca == cross_color and cb in want:
            i = want[cb]
            positions[2 * i], positions[2 * i + 1] = a, b
        elif cb == cross_color and ca in want:
            i = want[ca]
            positions[2 * i], positions[2 * i + 1] = b, a
    if (positions < 0).any():
        return None
    return positions


def _rename_map(orient_perm):
    """token in reoriented frame -> equivalent token in the original frame."""
    inv_o = np.argsort(orient_perm)
    out = {}
    by_bytes = {cm.PERMS[t].tobytes(): t for t in FACE_TOKENS}
    for t in FACE_TOKENS:
        conj = cm.compose(cm.compose(orient_perm, cm.PERMS[t]), inv_o)
        out[t] = by_bytes[conj.tobytes()]
    return out


def _orientation_for_color(state, color):
    return cm.orientation_with_down(state, color)


def solve_cross(state, color):
    """Optimal cross solution for `color` on `state` (any frame).

    Returns {"length": int, "solution": [tokens in the caller's frame]} or
    None if the state's edges are malformed.
    """
    dist = cross_distance_table()
    operm = _orientation_for_color(state, color)
    r = state[operm]
    pos = _cross_positions(r)
    if pos is None:
        return None
    rename = _rename_map(operm)
    _, inv = _tables()
    solution = []
    d = dist.get(int(_encode(pos.reshape(1, 8))[0]))
    if d is None:
        return None
    while d > 0:
        for m in range(18):
            cand = inv[m][pos]
            nd = dist.get(int(_encode(cand.reshape(1, 8))[0]))
            if nd == d - 1:
                pos = cand
                solution.append(rename[FACE_TOKENS[m]])
                d = nd
                break
        else:
            return None  # table inconsistency; should not happen
    return {"length": len(solution), "solution": solution}


def solve_xcross(state, color, node_budget=400_000):
    """Best x-cross (cross + one corner/edge pair, any slot) via IDA* with the
    cross table as heuristic. Returns {"length", "solution", "slot"} or None
    if no slot finishes within the node budget."""
    dist = cross_distance_table()
    operm = _orientation_for_color(state, color)
    r = state[operm]
    cross_pos = _cross_positions(r)
    if cross_pos is None:
        return None
    rename = _rename_map(operm)
    moves, inv = _tables()

    def cross_h(p):
        return dist[int(_encode(p[:8].reshape(1, 8))[0])]

    best = None
    for slot, corner_home in CORNER_SLOTS.items():
        edge_home = SLOT_EDGES[slot]
        targets = list(corner_home) + list(edge_home)
        # locate the pair's stickers: corner = the corner with cross color +
        # the slot's two side colors; edge = the slot's two side colors
        home_colors = [r[p // 9 * 9 + 4] for p in targets]
        pair_pos = _find_stickers(r, targets, home_colors)
        if pair_pos is None:
            continue
        start = np.concatenate([cross_pos, pair_pos])
        goal = np.array(
            [p for pr in CROSS_EDGE_HOMES for p in pr] + targets, dtype=np.intp
        )
        result = _ida(start, goal, inv, node_budget, cross_h)
        if result is not None and (best is None or len(result) < best[0]):
            best = (len(result), [rename[FACE_TOKENS[m]] for m in result], slot)
    if best is None:
        return None
    return {"length": best[0], "solution": best[1], "slot": best[2]}


def solve_f2l_pair(state, color, target_slot, preserved_slots=(), node_budget=400_000):
    """Optimal insertion of one F2L pair into `target_slot`, given a state where
    the cross and the `preserved_slots` are already solved, WITHOUT disturbing
    them (the cross + preserved pairs must end home — keyhole through the still-
    empty slots is allowed since they aren't tracked). IDA* with the per-slot
    pair-distance heuristic. Returns {"length", "solution", "slot"} or None.
    """
    operm = _orientation_for_color(state, color)
    r = state[operm]
    rename = _rename_map(operm)
    _, inv = _tables()

    targets = list(CORNER_SLOTS[target_slot]) + list(SLOT_EDGES[target_slot])
    home_colors = [r[p // 9 * 9 + 4] for p in targets]
    pair_pos = _find_stickers(r, targets, home_colors)
    if pair_pos is None:
        return None

    # Cross + already-solved pairs are at home in `r`; track them so the search
    # must restore them. (Start == goal for these — they begin solved.)
    preserved = [p for pr in CROSS_EDGE_HOMES for p in pr]
    for slot in preserved_slots:
        preserved += list(CORNER_SLOTS[slot]) + list(SLOT_EDGES[slot])
    preserved = np.array(preserved, dtype=np.intp)

    start = np.concatenate([pair_pos, preserved])
    goal = np.concatenate([np.array(targets, dtype=np.intp), preserved])
    table = pair_distance_table(target_slot)

    def pair_h(p):
        return table.get(int((p[:5].astype(np.int64) * _W5).sum()), 99)

    result = _ida(start, goal, inv, node_budget, pair_h, max_bound=16)
    if result is None:
        return None
    return {
        "length": len(result),
        "solution": [rename[FACE_TOKENS[m]] for m in result],
        "slot": target_slot,
    }


def _find_stickers(state, home_positions, home_colors):
    """Current positions of the stickers that belong at home_positions,
    identified by the color multiset of their piece."""
    n_corner = 3
    corner_colors = frozenset(home_colors[:n_corner])
    edge_colors = frozenset(home_colors[n_corner:])
    pos = np.full(len(home_positions), -1, dtype=np.intp)

    corner_groups = [
        (29, 26, 15), (27, 24, 44), (35, 51, 17), (33, 42, 53),
        (8, 20, 9), (6, 18, 38), (2, 45, 11), (0, 47, 36),
    ]
    for grp in corner_groups:
        if frozenset(int(state[p]) for p in grp) == corner_colors:
            for p in grp:
                idx = home_colors[:n_corner].index(int(state[p]))
                pos[idx] = p
            break
    for a, b in EDGE_SLOTS:
        if frozenset((int(state[a]), int(state[b]))) == edge_colors:
            pos[n_corner + home_colors[n_corner:].index(int(state[a]))] = a
            pos[n_corner + home_colors[n_corner:].index(int(state[b]))] = b
            break
    if (pos < 0).any():
        return None
    return pos


def _ida(start, goal, inv, node_budget, h, max_bound=12):
    """IDA* over tracked sticker positions. `h` is an admissible heuristic on a
    position array; `goal` must be reached on ALL tracked stickers (so a pair
    insertion may pass through — keyhole — but must restore every preserved
    piece). Sets the module `_last_nodes` for monitoring."""
    global _last_nodes
    goal_t = tuple(goal.tolist())
    bound = max(h(start), 0 if tuple(start.tolist()) == goal_t else 1)
    nodes = [0]

    def search(positions, g, bound, last_axis, path):
        if nodes[0] > node_budget:
            return "budget"
        nodes[0] += 1
        f = g + h(positions)
        if f > bound:
            return f
        if tuple(positions.tolist()) == goal_t:
            return "found"
        minimum = None
        for m in range(18):
            axis = m // 3
            if last_axis == axis:
                continue  # same face twice never optimal here
            cand = inv[m][positions]
            path.append(m)
            res = search(cand, g + 1, bound, axis, path)
            if res == "found" or res == "budget":
                return res
            path.pop()
            if minimum is None or res < minimum:
                minimum = res
        return minimum if minimum is not None else float("inf")

    try:
        while bound <= max_bound:
            path = []
            res = search(start, 0, bound, -1, path)
            if res == "found":
                return path
            if res == "budget" or res == float("inf"):
                return None
            bound = res
        return None
    finally:
        _last_nodes = nodes[0]
