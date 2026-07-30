"""Last-layer algorithm progress potential for the trellis search.

The module loads the shipped OLL, PLL, COLL, and CMLL tables, keeps pure
face-turn variants, and builds a bounded state-space potential from a supplied
F2L-complete state. The value is a soft search prior, not a decoded move source.
"""

import json
import os

import numpy as np

from analysis import cubemodel as cm
from analysis import cases as cs

_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis", "data", "algsets",
)
_AUF = ("", "U", "U2", "U'")
_FACE = set("URFDLB")
_FACE_TOKENS = [b + m for b in "URFDLB" for m in ("", "'", "2")]
_FT_SET = set(_FACE_TOKENS)
_IDENT = np.arange(54, dtype=np.intp)

# Comprehensive last-layer alg sources (excludes F2L = pre-LL).
_SETS = ["OLL", "PLL", "2-Look-OLL", "2-Look-PLL", "2-Look-CMLL", "COLL", "CMLL"]


# --- notation (verbatim from scripts/ollpll_faithful_check.py) ----------------
def _normtok(t):
    t = cm.norm_token(t)                 # Rw -> r
    if t.endswith("2'"):
        return t[:-1]                    # R2' -> R2
    if t.endswith("3'"):
        return t[:-2]                    # R3' -> R
    if t.endswith("3"):
        return t[:-1] + "'"              # R3 -> R'
    return t


def _tokenize(alg):
    return [t for t in alg.replace("(", " ").replace(")", " ").split() if t]


def _pure_face(toks):
    # strict: every token must be exactly one of the 18 face turns (filters
    # rotations, wide/slice, and malformed/space-less tokens like 'RR').
    return bool(toks) and all(t in _FT_SET for t in toks)


_QAMT = {"": 1, "2": 2, "'": 3}
_QINV = {1: "", 2: "2", 3: "'"}


def _base_amt(t):
    if t and t[-1] in "'2":
        return t[:-1], _QAMT[t[-1]]
    return t, 1


def _simplify(moves):
    """Merge consecutive same-base turns; cancel (U U' -> nothing, U' U' -> U2)."""
    out = []
    for m in moves:
        b, a = _base_amt(m)
        if out and _base_amt(out[-1])[0] == b:
            na = (_base_amt(out[-1])[1] + a) % 4
            out.pop()
            if na:
                out.append(b + _QINV[na])
        else:
            out.append(m)
    return out


def _perm_of(tokens):
    p = _IDENT
    for t in tokens:
        p = p[cm.PERMS[t]]
    return p


def _yrel_tables():
    """relabel[m][face-token] = y^m . token . y^-m (pure-face -> pure-face)."""
    rev = {cm.PERMS[t].tobytes(): t for t in _FACE_TOKENS}
    tabs = []
    for m in range(4):
        Rm = _IDENT
        for _ in range(m):
            Rm = Rm[cm.PERMS["y"]]
        inv = np.argsort(Rm)
        tabs.append({t: rev[cm.compose(cm.compose(Rm, cm.PERMS[t]), inv).tobytes()]
                     for t in _FACE_TOKENS})
    return tabs


_YREL = _yrel_tables()


def _load_algs():
    """All comprehensive last-layer algs as deduped pure-face token-lists."""
    raw = []
    for s in _SETS:
        p = os.path.join(_DATA, s + ".json")
        if not os.path.exists(p):
            continue
        for _case, cd in json.load(open(p)).get("cases", {}).items():
            algs = cd.get("algs")
            if isinstance(algs, dict):
                algs = list(algs.keys())
            for a in (algs or []):
                raw.append(a)
    out, seen = [], set()
    for a in raw:
        toks = _simplify([_normtok(t) for t in _tokenize(a)])
        if not _pure_face(toks):
            continue
        key = tuple(toks)
        if key not in seen:
            seen.add(key)
            out.append(toks)
    if not out:
        import sys as _sys
        print(
            f"[ll_alg_prior] no last-layer algorithms loaded from {_DATA!r}; "
            "the last-layer prior is disabled",
            file=_sys.stderr,
            flush=True,
        )
    return out


def _build_edges():
    """Edge set for the solution search: every alg conjugated through y^m and
    prefixed by AUF U^k (deduped), plus the 3 pure-AUF edges. Returns
    (moves_lists, net_perms (N,54))."""
    moves, seen = [], set()
    for k in (1, 2, 3):                       # pure-AUF edges (inter/final alignment)
        seq = ["U" + _QINV[k]] if k != 1 else ["U"]
        seen.add(tuple(seq))
        moves.append(seq)
    for alg in _ALGS:
        for m in range(4):
            ry = [_YREL[m][t] for t in alg]
            for k in range(4):
                seq = _simplify((["U" + _QINV[k]] if k else []) + ry) if k else _simplify(ry)
                if not seq:
                    continue
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    moves.append(seq)
    perms = np.stack([_perm_of(s) for s in moves]).astype(np.intp)
    return moves, perms


_ALGS = _load_algs()
_EDGE_MOVES, _EDGE_PERMS = _build_edges()


# --- F2L-complete predicate (verbatim from the validators) --------------------
def f2l_done_color(state):
    """Cross color if `state` has the first two layers solved (any color down),
    else None."""
    for o in cm.ORIENTATIONS:
        if "y" in o["word"]:
            continue
        r = state[o["perm"]].reshape(6, 9)
        if not (r[3] == r[3, 4]).all():
            continue
        if all((r[fi, 3:9] == r[fi, 4]).all() for fi in (1, 2, 4, 5)):
            return cm.COLOR_LIST[r[3, 4]]
    return None


def _top_oriented(state):
    f = state.reshape(6, 9)
    return bool((f[0] == f[0, 4]).all())


def _recognition_chains(r0):
    """Clean OLL->PLL solution token-lists (reoriented frame) via cases.py
    recognition — a guaranteed seed unioned into the search results."""
    chains = []
    info = {"oll": None, "pll": None}

    def add_pll(s, prefix, cap=14):
        if cm.is_solved(s):
            chains.append(list(prefix))
            return
        pll = cs.identify_pll(s)
        info["pll"] = info["pll"] or pll.get("id")
        palgs = _variants_for(pll, cap)
        for b in _AUF:
            sb = cm.apply_seq(s, [b] if b else [])
            if cm.is_solved(sb):
                chains.append(list(prefix) + ([b] if b else []))
                continue
            for pa in palgs:
                s2 = cm.apply_seq(sb, pa)
                for c in _AUF:
                    if cm.is_solved(cm.apply_seq(s2, [c] if c else [])):
                        chains.append(list(prefix) + ([b] if b else []) + pa
                                      + ([c] if c else []))
                        break

    if _top_oriented(r0):
        add_pll(r0, [])
    else:
        oll = cs.identify_oll(r0)
        info["oll"] = oll.get("id")
        for a in _AUF:
            pre = [a] if a else []
            b0 = cm.apply_seq(r0, pre)
            for oa in _variants_for(oll, 14):
                s1 = cm.apply_seq(b0, oa)
                if _top_oriented(s1):
                    add_pll(s1, pre + oa)
    return chains, info


def _variants_for(case, cap):
    """Pure-face token-lists for a recognized case dict (its own algs only)."""
    out, seen = [], set()
    for a in (case.get("algs") or []):
        toks = _simplify([_normtok(t) for t in _tokenize(a)])
        if not _pure_face(toks):
            continue
        key = tuple(toks)
        if key not in seen:
            seen.add(key)
            out.append(toks)
            if len(out) >= cap:
                break
    return out


def _search_chains(r0, beam, max_depth, max_chains):
    """Bounded best-first search over alg edges from r0 toward solved. An edge is
    accepted only if it strictly reduces the count of unsolved last-layer stickers
    (monotone progress); chains reaching solved within max_depth edges are
    returned (reoriented-frame token-lists)."""
    target = r0.reshape(6, 9)[:, 4]                      # face centers (solved target)

    def hvec(states):                                    # (N,54)->(N,) Hamming to solved
        return (states.reshape(-1, 6, 9) != target[None, :, None]).sum(axis=(1, 2))

    h0 = int((r0.reshape(6, 9) != target[:, None]).sum())
    frontier = [(r0, [], h0)]
    seen = {r0.tobytes()}
    chains = []
    for _ in range(max_depth):
        cand = []
        for st, mv, hcur in frontier:
            ends = st[_EDGE_PERMS]                       # (Nedge,54)
            hs = hvec(ends)
            for vi in np.where(hs == 0)[0]:
                chains.append(mv + _EDGE_MOVES[vi])
                if len(chains) >= max_chains:
                    return chains
            ok = np.where(hs < hcur)[0]
            for vi in ok[np.argsort(hs[ok])]:
                b = ends[vi].tobytes()
                if b in seen:
                    continue
                seen.add(b)
                cand.append((int(hs[vi]), ends[vi], mv + _EDGE_MOVES[vi]))
        if not cand:
            break
        cand.sort(key=lambda x: x[0])
        frontier = [(s, m, hh) for hh, s, m in cand[:beam]]
    return chains


_PERM_TOK = {cm.PERMS[t].tobytes(): t for t in _FACE_TOKENS}


def _reoriented_to_absolute(moves, operm):
    """Convert reoriented-frame (cross-down) face tokens to ABSOLUTE-frame face
    tokens. A reorientation is a pure-face relabel, so each token t maps to the
    absolute token whose permutation equals the conjugate P = operm . PERMS[t] .
    operm^-1. Returns None if any token has no pure-face image (never, for a
    whole-cube reorient of a pure-face alg)."""
    inv = np.argsort(operm)
    out = []
    for t in moves:
        P = operm[cm.PERMS[t]][inv]          # P[i] = operm[PERMS[t][inv[i]]]
        T = _PERM_TOK.get(P.tobytes())
        if T is None:
            return None
        out.append(T)
    return out


def all_solutions(s0, cross, beam=160, max_depth=5, max_chains=4000):
    """EVERY comprehensive last-layer solution from F2L-entry `s0` as a deduped list
    of ABSOLUTE-frame face-move paths (each, applied to s0 in order, reaches SOLVED).
    Sources: clean recognition (cases.py) UNIONED with the bounded best-first search
    over the full 13955-edge DB (AUF + y-conjugated OLL/PLL/2-Look/COLL/CMLL/
    algtrainer-flat). The caller picks among these by OBSERVED-STATE fit (the reads),
    so a wrong/non-standard execution is recovered by what the cube actually did, not
    by canonicalness. Empty list if no covering chain."""
    operm = cm.orientation_with_down(s0, cross)
    r0 = s0[operm]
    if cm.is_solved(r0):
        return []
    rec_chains, _info = _recognition_chains(r0)
    raw = rec_chains + _search_chains(r0, beam, max_depth, max_chains)
    out, seen = [], set()
    for c in raw:
        c = _simplify(c)
        if not c:
            continue
        ab = _reoriented_to_absolute(c, operm)
        if ab is None:
            continue
        st = s0.copy()
        for tok in ab:
            st = cm.apply(st, tok)
        if not cm.is_solved(st):
            continue
        key = tuple(ab)
        if key not in seen:
            seen.add(key)
            out.append(ab)
    return out


def build_injection(s0, cross, score_fn=None):
    """EXPRESSION half of the LL bridge: a recognized solution as an explicit
    ABSOLUTE-frame face-move path from F2L-entry `s0` (applied to s0 it reaches
    SOLVED). The trellis injects this as a forced terminal candidate so the decode
    can EXPRESS the alg the table recognized (the soft `ll_prior` only biases a
    depth-capped ball, which cannot generate a 7-12 move flurry).

    score_fn (the OBSERVED-STATE selector): callable(abs_moves)->float; when given,
    return the comprehensive chain whose trajectory the READS best support (so a
    wrong/non-standard execution is recovered from the resulting state, not assumed
    canonical). Without it, returns the SHORTEST chain (canonical fallback). None if
    no covering chain (caller leaves injection off => byte-identical)."""
    sols = all_solutions(s0, cross)
    if not sols:
        return None
    if score_fn is None:
        return min(sols, key=len)
    return max(sols, key=score_fn)


def build_potential(s0, cross, beam=160, max_depth=5, max_chains=4000):
    """Last-layer progress potential from F2L-entry state `s0` (trellis absolute
    array, solved cross color `cross`).

    Returns (prog, info): prog = {absolute-state-bytes -> progress in (0,1]}
    (1.0==solved) over every state on a known-alg solution trajectory from s0.
    (None, info) if no covering chain (soft: caller leaves the prior off)."""
    operm = cm.orientation_with_down(s0, cross)
    inv_operm = np.argsort(operm)
    r0 = s0[operm]                                       # cross down, last layer up
    info = {"oll": None, "pll": None, "n_edges": len(_EDGE_MOVES),
            "n_chains": 0, "lmax": 0, "n_states": 0}
    if cm.is_solved(r0):
        return None, info

    rec_chains, rec_info = _recognition_chains(r0)
    info.update(rec_info)
    chains = rec_chains + _search_chains(r0, beam, max_depth, max_chains)
    chains = [_simplify(c) for c in chains if c]
    chains = [c for c in chains if c]
    if not chains:
        return None, info

    lmax = max(len(c) for c in chains)
    rem = {}                                             # abs-state-bytes -> min moves-remaining
    for sol in chains:
        st = r0
        L = len(sol)
        for k, tok in enumerate(sol):
            st = cm.apply(st, tok)
            ab = st[inv_operm].tobytes()
            r = L - (k + 1)
            old = rem.get(ab)
            if old is None or r < old:
                rem[ab] = r
    prog = {b: 1.0 - (r / lmax) for b, r in rem.items()}
    info.update(n_chains=len(chains), lmax=lmax, n_states=len(prog))
    return prog, info
