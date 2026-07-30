#!/usr/bin/env python3
"""ANCHORED MEET-IN-THE-MIDDLE bridge SEARCH CORE (the postdictive bridge).

Between two CONFIDENT flanking state anchors A (last-correct pre-region state)
and B (first-correct post-region state), exhaustively expand a canonical move
ball forward from A and backward from B, hash-join at the middle, and return the
candidate A->B words. Read-evidence scoring (monotone-DP alignment against an
ordered set of mid-region read windows) ranks them; the ranking is taken over
NORMAL-FORM EQUIVALENCE CLASSES (mandatory: the unconstrained token
ranking degrades at large budgets ONLY because same-axis commuting phantom
insertions -- e.g. `D U' D U` == `D2` -- score identically to truth; the truth
CLASS is rank 1 at every budget/locus once these are folded).

This is the `dfinal=5` WORKS pattern (exhaustive ball around a KNOWN state +
hash-join) generalized to a PAIR of mid-solve anchors -- NOT the open-set /
whole-solve / lossy-beam dead ends. It is state-anchored at both ends, local
(3-4 moves per burst), and exhaustive, so fwd n bwd is guaranteed non-empty and
tiny. The committed fatal word is EXCLUDED BY CONSTRUCTION (its end-state != B).

KNOB-ZERO: every constant here is geometry (the 18-move HTM alphabet, per-move
permutations DERIVED from the canonical `apply_move`, the same-axis commutation
group used by the normal form) or a compute/memory SCALING guard (the join /
distinct-sequence / distinct-state bounds and depth cap). None is a
decode-outcome threshold: the guards only bound work; the winner is chosen by
read evidence with a >0 margin (the project-wide "positive evidence" bar), never
by a tuned constant. The scaling guards are the recorded envelope
(N_L x4-6/move; a 10-move gap @(5,5) ~= 1e4-9e4 candidates is fine; memory
rejects past (5,5)). Grammar is deliberately NOT used here as a generator,
prune, or vote -- measurements all show a bigram filter KILLS truth at
real loci; the bridge generates from geometry and scores from reads only.

The evidence source is injected as a callback so the SAME validated core serves
both the offline oracle (pinned-om mid-burst windows) and the live decoder
(per-span reads at the committed om): see `rank_classes`.
"""
import numpy as np

from core.cube import Cube
from detect.scramble import apply_move
from detect.move_detector import state_to_array, array_to_state

# ---- 18-move HTM alphabet + geometry (derived, no knob) --------------------
FACES = "UDLRFB"
AMTS = ["", "'", "2"]
MOVES = [f + a for f in FACES for a in AMTS]        # U U' U2 D D' D2 ... B2
FACE_OF = np.array([i // 3 for i in range(18)], np.int8)
# inverse-move index: U<->U', U2<->U2, per amount slot
INV = [i - i % 3 + (1 - i % 3 if i % 3 < 2 else 2) for i in range(18)]

# same-axis commutation bookkeeping (for the normal form). U/D share axis 0,
# L/R axis 1, F/B axis 2; opposite faces on one axis COMMUTE.
AXIS = {"U": 0, "D": 0, "L": 1, "R": 1, "F": 2, "B": 2}
FACE_ORD = {"U": 0, "D": 1, "L": 0, "R": 1, "F": 0, "B": 1}
_AMT = {"": 1, "'": 3, "2": 2}
_RAMT = {1: "", 3: "'", 2: "2"}

# ---- compute/memory SCALING guards (scaling envelope; NOT decode thresholds) -
JOIN_PAIR_BOUND = int(1e7)      # abort a budget level past this many join pairs
DISTINCT_SEQ_BOUND = int(1e6)   # abort past this many distinct A->B words
SCORE_STATE_BOUND = int(3e5)    # skip scoring past this many distinct states
DEPTH_CAP = 5                   # per-side ball depth cap (memory-reject beyond)


# ---------------------------------------------------------------- mechanics
def derive_perms():
    """Per-move position permutation DERIVED from the canonical `apply_move`
    (base-6 position-encoding trick), not reimplemented mechanics.
    new_state = old_state[perm[m]]."""
    perms = np.zeros((18, 54), np.int64)
    basis = [np.array([(i // 6 ** d) % 6 for i in range(54)], np.int8)
             for d in range(3)]
    for mi, tok in enumerate(MOVES):
        digs = []
        for b in basis:
            c = Cube(state=array_to_state(b))
            apply_move(c, tok)
            digs.append(np.asarray(state_to_array(c.state), np.int64))
        src = digs[0] + 6 * digs[1] + 36 * digs[2]
        assert sorted(src.tolist()) == list(range(54)), f"perm bad {tok}"
        perms[mi] = src
    return perms


def reduce_word(tokens):
    """Canonical-reduce: merge adjacent same-FACE tokens (amounts mod 4),
    dropping identities, repeating until stable. (Raw `D D` -> `D2`.)"""
    out = []
    for t in tokens:
        f, a = t[0], _AMT[t[1:]]
        if out and out[-1][0] == f:
            pf, pa = out.pop()
            na = (pa + a) % 4
            if na:
                out.append((pf, na))
        else:
            out.append((f, a))
    return [f + _RAMT[a] for f, a in out]


def normal_form(tokens):
    """Same-AXIS commutation normal form (the mandatory dedup key).
    Fixpoint of: split into maximal same-axis runs; within a run sum each
    face's amount mod 4 and emit in canonical face order; drop empties (which
    can merge neighbouring runs -> iterate). `D U' D U` -> `D2`; `U' D U D'`
    -> `` (identity)."""
    cur = [(t[0], _AMT[t[1:]]) for t in tokens]
    while True:
        runs, out = [], []
        for f, a in cur:
            if runs and AXIS[runs[-1][0][0]] == AXIS[f]:
                runs[-1].append((f, a))
            else:
                runs.append([(f, a)])
        for run in runs:
            acc = {}
            for f, a in run:
                acc[f] = (acc.get(f, 0) + a) % 4
            for f in sorted(acc, key=lambda x: FACE_ORD[x]):
                if acc[f]:
                    out.append((f, acc[f]))
        if out == cur:
            return tuple(f + _RAMT[a] for f, a in out)
        cur = out


def tokens_of(seq):
    """Move-index tuple -> HTM token list."""
    return [MOVES[i] for i in seq]


# ---------------------------------------------------------------- balls
def expand_ball(start, perms, depth, backward=False):
    """Exhaustive canonical ball (no adjacent same-face). Returns per-level
    dicts levels[d]=dict(states=(N,54) int8, edgeface=(N,), parent=(N,),
    mv=(N,)). Forward: seq applied to `start`, edgeface = LAST move's face.
    Backward: entry (s,seq) means seq applied to s gives `start` (=B);
    expansion uses INVERSE perms, mv = FIRST token of the suffix."""
    start = np.asarray(start, np.int8)
    lv = [dict(states=start[None, :].copy(),
               edgeface=np.array([-1], np.int8),
               parent=np.array([-1], np.int64),
               mv=np.array([-1], np.int8))]
    for _ in range(depth):
        S, EF = lv[-1]["states"], lv[-1]["edgeface"]
        outs, outf, outp, outm = [], [], [], []
        for mi in range(18):
            mask = EF != FACE_OF[mi]
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            p = perms[INV[mi]] if backward else perms[mi]
            outs.append(S[idx][:, p])
            outf.append(np.full(len(idx), FACE_OF[mi], np.int8))
            outp.append(idx)
            outm.append(np.full(len(idx), mi, np.int8))
        lv.append(dict(states=np.concatenate(outs),
                       edgeface=np.concatenate(outf),
                       parent=np.concatenate(outp),
                       mv=np.concatenate(outm)))
    return lv


def seq_of(lv, d, i, backward=False):
    toks = []
    while d > 0:
        toks.append(int(lv[d]["mv"][i]))
        i = int(lv[d]["parent"][i])
        d -= 1
    return tuple(toks) if backward else tuple(reversed(toks))


def build_state_index(lv, dmax):
    """dict state_bytes -> list of (depth, idx) over levels 0..dmax."""
    ix = {}
    for d in range(dmax + 1):
        S = lv[d]["states"]
        for i in range(len(S)):
            ix.setdefault(S[i].tobytes(), []).append((d, i))
    return ix


def join_budget(fwd, bwd, da, db, truth=None):
    """Hash-join fwd depth<=da with bwd depth<=db. Returns (stats, cands)
    where cands is dict seq_tuple->None (or None if a scaling bound tripped)."""
    fix = build_state_index(fwd, da)
    pairs = 0
    cands = {}
    tripped = None
    for d in range(db + 1):
        S = bwd[d]["states"]
        for i in range(len(S)):
            hits = fix.get(S[i].tobytes())
            if not hits:
                continue
            bseq = seq_of(bwd, d, i, backward=True)
            bf = FACE_OF[bseq[0]] if bseq else -2
            for (fd, fi) in hits:
                pairs += 1
                if pairs > JOIN_PAIR_BOUND:
                    tripped = f"join pairs > {JOIN_PAIR_BOUND:.0e}"
                    break
                ff = fwd[fd]["edgeface"][fi]
                if bseq and fd and ff == bf:
                    continue                      # junction canonicality
                cands[seq_of(fwd, fd, fi) + bseq] = None
                if len(cands) > DISTINCT_SEQ_BOUND:
                    tripped = f"distinct seqs > {DISTINCT_SEQ_BOUND:.0e}"
                    break
            if tripped:
                break
        if tripped:
            break
    st = dict(da=da, db=db, join_pairs=pairs, tripped=tripped,
              n_candidates=(None if tripped else len(cands)),
              truth_contained=(None if (tripped or truth is None)
                               else (truth in cands)))
    return st, (None if tripped else cands)


def count_exact_paths(fwd, bwd, L):
    """N_L = # canonical A->B words of length exactly L, counted by splitting
    at p=L//2 (each length-L word has a unique length-p prefix). Groups states
    by (state, junction-face) to subtract same-face junction pairs."""
    p, q = L // 2, L - L // 2

    def facecounts(lv, d):
        S = lv[d]["states"]
        F = lv[d]["edgeface"] if d else np.array([-1], np.int8)
        v = np.ascontiguousarray(S).view(
            np.dtype((np.void, S.shape[1]))).ravel()
        u, inv = np.unique(v, return_inverse=True)
        cnt = np.zeros((len(u), 7), np.int64)
        np.add.at(cnt, (inv, np.where(F < 0, 6, F)), 1)
        return u, cnt
    uf, cf = facecounts(fwd, p)
    ub, cb = facecounts(bwd, q)
    common, fi, bi = np.intersect1d(uf, ub, return_indices=True)
    if len(common) == 0:
        return 0
    A, Bc = cf[fi].astype(np.int64), cb[bi].astype(np.int64)
    tot = A.sum(1) * Bc.sum(1)
    same = (A[:, :6] * Bc[:, :6]).sum(1)
    return int((tot - same).sum())


# ---------------------------------------------------------------- scoring
def materialize_states(cands, perms, A):
    """For each candidate word, the sequence of column-indices into a shared
    distinct-state matrix (state before move 0 .. after last move). Returns
    (seqs, cols, smat, n_states) or (None, None, None, n_states) if the
    distinct-state scaling bound trips."""
    A = np.asarray(A, np.int8)
    seqs = list(cands)
    st_ix = {}
    cols = []
    for seq in seqs:
        s = A
        ix = [st_ix.setdefault(s.tobytes(), len(st_ix))]
        for mi in seq:
            s = s[perms[mi]]
            ix.append(st_ix.setdefault(s.tobytes(), len(st_ix)))
        cols.append(ix)
        if len(st_ix) > SCORE_STATE_BOUND:
            return None, None, None, len(st_ix)
    smat = np.zeros((len(st_ix), 54), np.int8)
    for sb, i in st_ix.items():
        smat[i] = np.frombuffer(sb, np.int8)
    return seqs, cols, smat, len(st_ix)


def dp_scores(cols, T):
    """Monotone-alignment DP: each candidate's ordered states aligned
    (non-decreasing) to the W ordered read windows; score = best alignment.
    T is the (W, n_states) window-fit table. Vectorized per candidate length.
    Knob-free maximum-likelihood alignment."""
    N = len(cols)
    out = np.full(N, -np.inf)
    bylen = {}
    for i, ix in enumerate(cols):
        bylen.setdefault(len(ix), []).append(i)
    for _k, idxs in bylen.items():
        C = np.array([cols[i] for i in idxs])         # (n, k+1)
        F = T[:, C]                                    # (W, n, k+1)
        D = F[0]
        for w in range(1, T.shape[0]):
            D = F[w] + np.maximum.accumulate(D, axis=1)
        out[np.array(idxs)] = D.max(axis=1)
    return out


def class_ranking(seqs, sc):
    """Fold candidate scores into NORMAL-FORM equivalence classes (mandatory).
    Returns a list of dicts sorted by best member score:
    dict(nf=<tuple of tokens>, best_i=<idx into seqs>, best_score=float,
    members=<list of idx>). The class each candidate maps to via `normal_form`
    of its token word."""
    best_by_nf, members = {}, {}
    for i, seq in enumerate(seqs):
        nf = normal_form(tokens_of(seq))
        members.setdefault(nf, []).append(i)
        if nf not in best_by_nf or sc[i] > sc[best_by_nf[nf]]:
            best_by_nf[nf] = i
    ranked = sorted(best_by_nf, key=lambda nf: -sc[best_by_nf[nf]])
    return [dict(nf=nf, best_i=best_by_nf[nf],
                 best_score=float(sc[best_by_nf[nf]]),
                 members=members[nf]) for nf in ranked]


# ---------------------------------------------------------------- driver
def build_balls(A, B, perms, depth=DEPTH_CAP):
    """Forward ball from A, backward ball from B, to `depth` per side."""
    return (expand_ball(A, perms, depth),
            expand_ball(B, perms, depth, backward=True))


def rank_classes(A, B, perms, budget, window_fit_fn, fwd=None, bwd=None):
    """The step-5 bridge, end to end. Expand/join at `budget`=(da,db), score
    the joined candidates against read evidence, and return the normal-form
    class ranking.

    window_fit_fn(smat)-> T maps an (S,54) distinct-state matrix to a
    (W, S) ordered-window fit table (W = number of mid-region read windows);
    this is the ONLY evidence coupling -- the offline oracle and the live
    decoder each supply their own. Returns dict(status=..., ...):
      status='ok': n_candidates, n_classes, classes (class_ranking list with a
        `tokens` field on each), top (the winning class), class_margin
        (top.best_score - best non-top-class member score; +inf if one class),
        decisive (top-class strictly rank 1 with class_margin > 0).
      status='tripped'/'skipped'/'empty': the reason, no ranking.
    """
    da, db = budget
    if fwd is None or bwd is None:
        fwd, bwd = build_balls(A, B, perms, max(da, db))
    st, cands = join_budget(fwd, bwd, da, db)
    if st["tripped"]:
        return dict(status="tripped", reason=st["tripped"], stats=st)
    if not cands:
        return dict(status="empty", stats=st)
    seqs, cols, smat, n_states = materialize_states(cands, perms, A)
    if seqs is None:
        return dict(status="skipped",
                    reason=f"distinct states {n_states} > {SCORE_STATE_BOUND:.0e}",
                    stats=st)
    T = window_fit_fn(smat)
    if T is None or T.shape[0] == 0:
        return dict(status="empty", reason="no usable read windows", stats=st)
    sc = dp_scores(cols, T)
    classes = class_ranking(seqs, sc)
    for c in classes:
        c["tokens"] = tokens_of(seqs[c["best_i"]])
    top = classes[0]
    if len(classes) > 1:
        best_non = max(sc[i] for i in range(len(seqs))
                       if normal_form(tokens_of(seqs[i])) != top["nf"])
        margin = top["best_score"] - float(best_non)
    else:
        margin = float("inf")
    return dict(status="ok", stats=st, n_candidates=len(seqs),
                n_states=n_states, n_classes=len(classes), classes=classes,
                top=top, class_margin=margin,
                decisive=bool(margin > 0))
