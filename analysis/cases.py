"""OLL/PLL/F2L case recognition.

Operates on states reoriented so the locked cross color is on the down face
(any of the 4 y-variants — fingerprints are canonicalized over U turns, and
the encodings are invariant to whole-cube y rotation).

OLL fingerprint: 21 booleans — does each last-layer sticker show the up-face
color — over positions [U 0..8, F 18..20, R 9..11, B 45..47, L 36..38],
canonicalized as the min string over the 4 pre-AUF (U^k) transforms.

PLL fingerprint: for the 12 band stickers, the cyclic offset (0-3) between
the face the sticker sits on and the face whose center has the sticker's
color, canonicalized over all 16 pre-AUF x viewing-rotation transforms
(U^k y^m). Center-relative encoding makes it color-neutral; unlike the OLL
boolean pattern, U and y act DIFFERENTLY on it (U changes offsets, y does
not), so both must be enumerated. The 21 standard PLLs are exactly the
nontrivial equivalence classes under this group, so the min form is exact.

F2L fingerprint (per slot): the positions of the slot's corner+edge pair only,
located BY COLOR (corner = cross color + the slot's two side colors; edge =
those two side colors) and encoded relative to the slot, min over the 4 pre-AUF
(U^k) transforms. It reads only those 5 stickers (and the centers, to identify
them), so it is independent of the cross, the other three F2L pairs, and the
last layer — all of which may still be unsolved mid-solve. Encoding positions
(not colors) makes it color-neutral. The table is keyed per slot, so a state is
queried as identify_f2l(state, slot) for slot in FR/FL/BR/BL.
"""

import json
from functools import lru_cache
from pathlib import Path

from analysis import cubemodel as cm
from analysis import solvers as sv

DATA_DIR = Path(__file__).resolve().parent / "data"

_U_POSITIONS = list(range(9))
_BAND_POSITIONS = [18, 19, 20, 9, 10, 11, 45, 46, 47, 36, 37, 38]  # F R B L top rows
# cyclic order of the side faces around the U axis (one consistent direction)
_SIDE_FACES = [2, 1, 5, 4]  # front, right, back, left (face indices)
_FACE_OF_POSITION = {p: p // 9 for p in _BAND_POSITIONS}


def _oll_pattern(state):
    up = state[4]
    return "".join(
        "1" if state[p] == up else "0" for p in _U_POSITIONS + _BAND_POSITIONS
    )


def oll_fingerprint(state):
    best = None
    s = state
    for _ in range(4):
        pat = _oll_pattern(s)
        if best is None or pat < best:
            best = pat
        s = cm.apply(s, "U")
    return best


def _pll_pattern(state):
    # face index -> position in the side cycle
    cycle_pos = {f: i for i, f in enumerate(_SIDE_FACES)}
    # color -> cycle position of the face whose center has it
    color_home = {state[f * 9 + 4]: cycle_pos[f] for f in _SIDE_FACES}
    out = []
    for p in _BAND_POSITIONS:
        here = cycle_pos[_FACE_OF_POSITION[p]]
        home = color_home[state[p]]
        out.append(str((here - home) % 4))
    return "".join(out)


def pll_fingerprint(state):
    best = None
    s = state
    for _ in range(4):  # pre-AUF
        t = s
        for _ in range(4):  # viewing rotation
            pat = _pll_pattern(t)
            if best is None or pat < best:
                best = pat
            t = cm.apply(t, "y")
        s = cm.apply(s, "U")
    return best


@lru_cache(maxsize=None)
def _load_table(kind):
    path = DATA_DIR / f"{kind}_cases.json"
    with open(path) as f:
        data = json.load(f)
    return {c["fingerprint"]: c for c in data["cases"]}


def identify_oll(state):
    """state: reoriented cross-down, F2L done, OLL pending. Returns case dict
    or a 'nonstandard' marker (never raises on unknown patterns)."""
    case = _load_table("oll").get(oll_fingerprint(state))
    if case is None:
        return {"id": None, "name": None, "group": "nonstandard", "algs": []}
    return case


def identify_pll(state):
    case = _load_table("pll").get(pll_fingerprint(state))
    if case is None:
        return {"id": None, "name": None, "group": "nonstandard", "algs": []}
    return case


# --- F2L (per-slot, pair-only) ------------------------------------------------

F2L_SLOTS = ("FR", "FL", "BR", "BL")


def _f2l_pair_positions(state, slot):
    """Current positions of the `slot` pair's 5 stickers, located by color (the
    corner carrying the cross color + the slot's two side colors, then the edge
    carrying those two side colors). Mirrors solvers._find_stickers. None if the
    pieces can't be located. Order: corner [cross, side1, side2], edge [side1,
    side2]; side colors are read from the slot's face centers, so it adapts to
    whatever frame/colors the state is in."""
    targets = list(sv.CORNER_SLOTS[slot]) + list(sv.SLOT_EDGES[slot])
    home_colors = [int(state[p // 9 * 9 + 4]) for p in targets]
    return sv._find_stickers(state, targets, home_colors)


def _f2l_pattern(state, slot):
    pos = _f2l_pair_positions(state, slot)
    if pos is None:
        return None
    return "".join(f"{int(p):02d}" for p in pos)


def f2l_fingerprint(state, slot):
    """Pair-only fingerprint of `slot` (cross-down state), min over the 4 AUFs.
    Independent of the cross, the other pairs, and the last layer. None if the
    pair's pieces can't be located."""
    best = None
    s = state
    for _ in range(4):
        pat = _f2l_pattern(s, slot)
        if pat is not None and (best is None or pat < best):
            best = pat
        s = cm.apply(s, "U")
    return best


@lru_cache(maxsize=None)
def _load_f2l():
    """slot -> {fingerprint: case dict}."""
    with open(DATA_DIR / "f2l_cases.json") as f:
        data = json.load(f)
    table = {s: {} for s in F2L_SLOTS}
    for c in data["cases"]:
        table[c["slot"]][c["fingerprint"]] = c
    return table


def identify_f2l(state, slot):
    """Identify the F2L case of `slot`'s pair on a cross-down `state`.

    Returns a _case_payload-shaped dict {id, name, group, standard_algs,
    standard_htm, fingerprint}; id/name None and group 'nonstandard' when the
    pair is solved or in no bundled case (never raises on unknown patterns)."""
    if slot not in F2L_SLOTS:
        raise ValueError(f"unknown F2L slot {slot!r}")
    fp = f2l_fingerprint(state, slot)
    case = _load_f2l()[slot].get(fp)
    if case is None:
        return {
            "id": None, "name": None, "group": "nonstandard",
            "standard_algs": [], "standard_htm": None, "fingerprint": fp,
        }
    return {
        "id": case["id"], "name": case["name"], "group": case.get("group"),
        "standard_algs": case["algs"][:2], "standard_htm": case["htm"],
        "fingerprint": fp,
    }
