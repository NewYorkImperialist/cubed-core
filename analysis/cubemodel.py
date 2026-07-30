"""54-facelet permutation model for solve analysis.

State = (54,) int8 array: 6 faces x 9 stickers, color indices 0-5.
Layout matches detect/move_detector.py: positions 0-8 = up, 9-17 = right,
18-26 = front, 27-35 = down, 36-44 = left, 45-53 = back.

Permutations are derived from cube.py via the unique-label trick so the
move semantics have a single source of truth. detect/scramble.py is loaded
by file path because importing the detect package pulls in the CV stack
(onnxruntime/cv2), which analysis must stay free of.

Rotations (x/y/z) are permutations like any other token, so replay needs no
orientation-frame bookkeeping: a pasted reconstruction with rotations just
permutes the state, and the progress predicates are evaluated under
reorientation anyway.
"""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cube import Cube  # noqa: E402  (light: face/node only)


def _load_scramble_module():
    spec = importlib.util.spec_from_file_location(
        "analysis._scramble", ROOT / "detect" / "scramble.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_scramble = _load_scramble_module()
validate_scramble = _scramble.validate_scramble
parse_scramble = _scramble.parse_scramble

ALL_FACES = ["up", "right", "front", "down", "left", "back"]
COLOR_LIST = ["white", "yellow", "red", "orange", "blue", "green"]
COLOR_INDEX = {c: i for i, c in enumerate(COLOR_LIST)}

FACE_BASES = list("RUFDLB")
SLICE_BASES = ["M", "S", "E"]
WIDE_BASES = list("rufdlb")
ROTATION_BASES = ["x", "y", "z"]
MODIFIERS = ["", "'", "2"]


def norm_token(token):
    """Normalize wide notation: 'Rw' -> 'r', 'Uw2' -> 'u2'."""
    if len(token) >= 2 and token[1] == "w":
        return token[0].lower() + token[2:]
    return token


def token_base(token):
    token = norm_token(token)
    if token[-1] in ("'", "2"):
        return token[:-1]
    return token


def state_to_array(state):
    arr = np.empty(54, dtype=np.int8)
    for i, face in enumerate(ALL_FACES):
        for j, color in enumerate(state[face]):
            arr[i * 9 + j] = COLOR_INDEX[color]
    return arr


def array_to_state(arr):
    return {
        face: [COLOR_LIST[arr[i * 9 + j]] for j in range(9)]
        for i, face in enumerate(ALL_FACES)
    }


def _labeled_cube():
    c = Cube()
    for i, face in enumerate(ALL_FACES):
        for j in range(9):
            c.state[face][j] = f"{i}_{j}"
    return c


def _perm_from_labels(c):
    perm = np.empty(54, dtype=np.intp)
    for fi, face in enumerate(ALL_FACES):
        for j in range(9):
            src_face, src_pos = c.state[face][j].split("_")
            perm[fi * 9 + j] = int(src_face) * 9 + int(src_pos)
    return perm


def _compute_perms():
    perms = {}
    bases = FACE_BASES + SLICE_BASES + WIDE_BASES + ROTATION_BASES
    for base in bases:
        for mod in MODIFIERS:
            token = base + mod
            c = _labeled_cube()
            _scramble.apply_move(c, token)
            perms[token] = _perm_from_labels(c)
    return perms


# token -> (54,) permutation; new_state = state[perm]
PERMS = _compute_perms()

SOLVED = state_to_array(Cube().state)

IDENTITY = np.arange(54, dtype=np.intp)


def compose(first, second):
    """Permutation of applying `first` then `second`."""
    return first[second]


def _compute_orientations():
    """All 24 whole-cube orientations as perms, tagged with the color that
    the orientation brings to the down face (for cross-color reorientation)."""
    top_words = [[], ["x"], ["x", "x"], ["x'"], ["z"], ["z'"]]
    orientations = []
    seen = set()
    for top in top_words:
        for n_y in range(4):
            word = top + ["y"] * n_y
            perm = IDENTITY
            for tok in word:
                perm = compose(perm, PERMS[tok])
            key = perm.tobytes()
            if key in seen:
                continue
            seen.add(key)
            down_color = COLOR_LIST[SOLVED[perm][3 * 9 + 4]]
            orientations.append({"perm": perm, "down": down_color, "word": word})
    assert len(orientations) == 24
    return orientations


ORIENTATIONS = _compute_orientations()


def apply(state, token):
    """Apply one notation token to a (54,) state array. Returns new array."""
    return state[PERMS[norm_token(token)]]


def apply_seq(state, tokens):
    for tok in tokens:
        state = state[PERMS[norm_token(tok)]]
    return state


def invert(tokens):
    """Inverse of a token sequence."""
    out = []
    for tok in reversed(tokens):
        tok = norm_token(tok)
        if tok.endswith("'"):
            out.append(tok[:-1])
        elif tok.endswith("2"):
            out.append(tok)
        else:
            out.append(tok + "'")
    return out


def rotation_to_down(color):
    """Shortest whole-cube rotation word (list of tokens) that brings `color` to the
    down face — i.e. orients a solve cross-down / last-layer-up for viewing."""
    words = [o["word"] for o in ORIENTATIONS if o["down"] == color]
    return min(words, key=len) if words else []


def conjugation_remap(w_word):
    """{token: token} mapping each move to its conjugate under whole-cube rotation
    `w_word`, so the mapped move applied in the w-rotated frame equals the original
    in the base frame. Re-expresses a whole solve in a new orientation: conjugate
    every move and the replay stays correct while the cube sits cross-down."""
    W = IDENTITY
    for tok in w_word:
        W = compose(W, PERMS[norm_token(tok)])
    inv_w = np.argsort(W)
    by_perm = {perm.tobytes(): tok for tok, perm in PERMS.items()}
    remap = {}
    for tok, perm in PERMS.items():
        conj = compose(inv_w, compose(perm, W))
        remap[tok] = by_perm.get(conj.tobytes(), tok)
    return remap


# --- faithful-move folding (wide/slice/rotation) ------------------------------
# A burst of decoded tokens whose NET permutation equals a single token's perm is
# re-expressed as that token (wide/slice/rotation), gated by the motion channel so
# a deliberate rotation+face is not mis-folded into a wide.
_PERM2TOK = {p.tobytes(): t for t, p in PERMS.items()}   # net perm -> single token


def net_perm(tokens):
    """Net (54,) permutation of a token sequence (IDENTITY for empty)."""
    perm = IDENTITY
    for tok in tokens:
        perm = compose(perm, PERMS[norm_token(tok)])
    return perm


# Fail loud if cube.py move semantics ever drift from the wide identity r == R M'
# (the whole fold algebra is keyed on net-perm equality).
assert np.array_equal(net_perm(["R", "M'"]), PERMS["r"]), "compose/perm semantics drifted"


def same_state(tokens_a, tokens_b):
    """True iff two token sequences yield the SAME facelet permutation. The
    zero-GT correctness invariant for faithful reconstruction (folding must be
    net-perm-preserving)."""
    return np.array_equal(net_perm(tokens_a), net_perm(tokens_b))


def fold_burst(tokens, band_extent=None, moved_middle=None):
    """Re-express ONE physical burst's decoded tokens as the single faithful
    token whose perm matches their net perm, else return them unchanged.

    Motion gate (the band channel's job): a wide only when the band shows a
    2-layer slide (band_extent == 2); a slice only when the middle layer moved
    (moved_middle is not False). None skips a gate (no evidence yet)."""
    toks = [norm_token(t) for t in tokens]
    if len(toks) <= 1:
        return toks
    hit = _PERM2TOK.get(net_perm(toks).tobytes())
    if hit is None:
        return toks                                  # genuine multi-move burst
    base = token_base(hit)
    if base in WIDE_BASES and band_extent not in (None, 2):
        return toks                                  # band says not 2 layers
    if base in SLICE_BASES and moved_middle is False:
        return toks                                  # band says not the middle
    return [hit]


def is_solved(state):
    """Solved up to whole-cube orientation: every face monochromatic."""
    faces = state.reshape(6, 9)
    return bool((faces == faces[:, 4:5]).all())


def state_from_scramble(scramble_str):
    """SOLVED with a scramble string applied (canonical white-top green-front frame)."""
    return apply_seq(SOLVED, validate_scramble(scramble_str))


def orientation_with_down(state, color):
    """Perm reorienting `state` so the center of `color` is on the down face."""
    for o in ORIENTATIONS:
        if "y" in o["word"]:
            continue
        if COLOR_LIST[state[o["perm"]][3 * 9 + 4]] == color:
            return o["perm"]
    raise ValueError(f"no orientation puts {color} down")
