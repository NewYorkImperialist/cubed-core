"""Run the research trellis from prepared reads, motion, and optional sensors.

The maintained runtime profile and its flags live in
``config/decode-runtime-v1.json``. Optional teacher inputs are evaluation
evidence; their diagnostic comparisons are not camera-only measurements.
"""
import sys, json, pickle, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

from detect.trellis_tracker import (TrellisTracker, AbsSegment,
    establish_orientation_from_reads, reconstruct_with_rotations_safe, _om_key,
    simplify_moves)
from detect.calibrator import MIN_STICKER_L, refresh_palette_from_glance
from detect.move_detector import state_to_array
from calib_util import load_centroids
from core.cube import Cube
from core.perf_trace import get_trace
from detect.scramble import apply_move
from detect.intraburst_motion import (
    certify_intraburst_phase_slots, remap_intraburst_phase_audit,
    require_exact_integer,
)

PERF_TRACE = get_trace("decode")

# Runner-visible marker for the optional scrub features implemented end to end
# by this parser, the tracker, and the scrub module.
SCRUB_EXPERIMENT_API = frozenset({
    "dense-reads-v1", "dense-prefix-v1", "gate-drop-slots-v1",
    "intraburst-phase-slots-v1",
})


def edit_similarity(p, g):
    """Diagnostic normalized edit-distance similarity to teacher moves."""
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (p[i-1] != g[j-1]))
    return 1 - dp[n][m] / max(1, len(g))


def editdist(p, g):
    """Raw Levenshtein distance for diagnostic reporting."""
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (p[i-1] != g[j-1]))
    return dp[n][m]


def argval(flag, d=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


def _reconstruction_checkpoints_valid(
        *, reach, backward, raw_moves, canonical_moves):
    """Whether forward-clock scrub checkpoints may enter the result."""
    return bool(
        reach
        and not backward
        and isinstance(raw_moves, (list, tuple))
        and raw_moves
        and simplify_moves(list(raw_moves)) == canonical_moves
    )


def resolve_start_state(sess):
    """(Cube, reason_or_None) for the init/start-state (scramble=None
    handling). scr=None (free-record `session.mode=="free"`,
    or any other session missing/null `scramble`) FIRST checks for a derived
    `start_state` (scripts/build_movegt.py's permanent derive-from-solved-end
    path, the "New captures — GT integrity" gap;
    provenance `start_state_source`, e.g. "derived_from_solved_end") and
    replays THAT — the actual backward-derived state, not a default — when
    present; else falls back to a fresh SOLVED Cube + a human-readable reason
    string (no crash, no silent default — the caller prints the reason). scr
    present => the scrambled Cube + None. Pure (no file I/O beyond what's
    already in `sess`; no stdout)."""
    scr = sess.get("scramble")
    if scr is None:
        start_state = sess.get("start_state")
        if start_state is not None:
            from detect.move_detector import array_to_state
            source = sess.get("start_state_source", "derived_from_solved_end")
            c0 = Cube(state=array_to_state(np.asarray(start_state, dtype=np.int8)))
            return c0, f"derived start state ({source})"
        free_mode = sess.get("mode") == "free"
        reason = ("free-record mode (session.mode=='free')" if free_mode else
                   "scramble field missing/null (NOT free-mode — check the "
                   "session JSON)")
        return Cube(), reason
    c0 = Cube()
    for m in scr.split():
        apply_move(c0, m)
    return c0, None


# ---------------------------------------------------------------------------
# Free-mode / derived-S0 benchmark-hygiene QC gates (Layers C+D). Opt-in
# (--gt-qc-gates), pure QC output — prints only, never alters `tt`/decode
# state. Split into "verdict" (pure math over precomputed scores, testable
# with synthetic dicts, no video data) and "scores" (real per-tag fit via the
# same AbsSegment.score_states evidence path the tracker itself uses).
# ---------------------------------------------------------------------------

def rest_fit_verdict(scores):
    """Layer C — rest-fit sweep gate math. `scores` = {label: summed fit
    score over the rest-window, ...}, must include 'S0' (the claimed k=0
    start state). Returns (winner_label, margin, ok): margin = S0's score
    minus the best OTHER candidate's score (positive => S0 wins comfortably);
    ok = S0 is the winner (ties favor S0). Raises ValueError if 'S0' absent."""
    if "S0" not in scores:
        raise ValueError("scores must include the 'S0' key")
    winner = max(scores, key=lambda k: (scores[k], k == "S0"))
    others = [v for k, v in scores.items() if k != "S0"]
    margin = scores["S0"] - (max(others) if others else float("-inf"))
    return winner, margin, winner == "S0"


def offset_gate_verdict(off_scores):
    """Layer D — movegt global-offset sweep gate math. `off_scores` =
    {offset:int -> mean fit score at that global frame-offset, ...}. Returns
    (best_offset, ok): ok = offset 0 is (tied-for-)best (no detected frame-map
    drift); ties break toward the offset closest to zero, then toward zero
    itself, for determinism. Empty input => (0, True) (nothing to warn about)."""
    if not off_scores:
        return 0, True
    best_offset = min(off_scores, key=lambda d: (-off_scores[d], abs(d), d))
    return best_offset, best_offset == 0


def teacher_forced_report(records, movegt, gl_index=None):
    """Teacher-forced per-gap agreement report lines. Pure math over
    the tracker's precomputed tf_records (testable with synthetic dicts, no
    video data — the rest_fit_verdict pattern). records =
    TrellisTracker.tf_records ({gap, f_a, f_b, gt, gt_idx, got, margin,
    ...om/state probe fields}); movegt = the sorted [(frame, move)] GT
    timeline; gl_index = reach_ll LL-onset GT move index (moves[:gl_index]
    are pre-LL) or None (boundary unavailable -> preLL falls back to the
    full counts; the caller prints why). A gap agrees iff its committed
    moves equal its teacher moves after ``simplify_moves``
    (simplify_moves); the gt bucket is the tracker's READ-ALIGNED replay
    segment, not the raw stamp window (stamp skew, see _tf_align_anchors).
    A gap with GT moves is pre-LL iff ALL of them are; an empty gap is
    pre-LL iff it ends before the first LL move's frame. tf_om=chained
    stamps the om policy (the decoder's own chained om channel). NOTE:
    margins can be legitimately negative — the decisive-om / contrastive
    override paths re-rank the commit by FIT, and the margin is top-vs-
    second CUMULATIVE at the commit. Returns one greppable agreement summary
    plus one mismatch line per missed gap (moves comma-joined, '-' = none)."""
    def ok(r):
        return simplify_moves(list(r["got"])) == simplify_moves(list(r["gt"]))

    def prell(r):
        if gl_index is None:
            return True
        if r["gt_idx"]:
            return max(r["gt_idx"]) < gl_index
        return gl_index >= len(movegt) or r["f_b"] < movegt[gl_index][0]

    pre = [r for r in records if prell(r)]
    pc, fc = sum(map(ok, pre)), sum(map(ok, records))
    lines = [
        f"teacher_forced_agreement_pre_ll={pc}/{len(pre)} "
        f"full={fc}/{len(records)} orientation_policy=chained"
    ]
    for r in records:
        if not ok(r):
            m = r["margin"]
            ms = "inf" if m == float("inf") else f"{m:.2f}"
            lines.append(
                f"teacher_forced_mismatch gap={r['gap']} "
                f"frames={r['f_a']}-{r['f_b']} "
                f"teacher={','.join(r['gt']) or '-'} "
                f"emitted={','.join(r['got']) or '-'} margin={ms}"
            )
    return lines


def span_collapse_report(log):
    """Per-tag SPAN-COLLAPSE summary. `log` =
    TrellisTracker.span_collapse_log (one dict per span evaluation, from
    TrellisTracker._span_collapse). Pure math over the precomputed log (the
    teacher_forced_report pattern). Returns one span-collapse summary with the
    total evaluations, collapse count, and collapse rate; an empty log reports
    rate 0.0."""
    total = len(log)
    collapsed = sum(1 for e in log if e["collapsed"])
    rate = collapsed / total if total else 0.0
    return [f"spanCollapseSummary total={total} collapsed={collapsed} "
            f"rate={rate:.6f}"]


def rest_fit_scores(tag, scr, raw, seg_factory, init_om, init):
    """Real rest-window fit scores (Layer C) for `rest_fit_verdict`: S0
    (the claimed k=0 start, `init`) vs each of the 18 single-move
    perturbations S0+<move>, summed over reads strictly before the first
    ON-CAMERA movegt frame (frame>=0 entries only; off-camera sentinels are
    excluded).
    Returns (scores_dict, n_frames_used); n_frames_used=0 => gate can't run
    (no rest-window reads with a valid segment).

    KNOWN LIMITATION vs the forensic: this scores all candidates at a single
    FIXED orientation (`init_om`, the run's already-resolved om), not the
    forensic's full joint (state x om) sweep — the "at minimum: beat
    single-move perturbations" bar from the task, not a full reproduction.
    A fixed-orientation pass can disagree with a joint sweep, so treat a pass
    here as necessary rather than sufficient; a warning still identifies a
    concrete inconsistency."""
    mgp = f"walkthrough/movegt_{tag}.json"
    mg_frames = ([int(m["frame"]) for m in json.load(open(mgp))]
                 if os.path.exists(mgp) else [])
    mg_frames = [f for f in mg_frames if f >= 0]
    first_ev = min(mg_frames) if mg_frames else None
    rest_reads = [r for f, (r, _m) in raw.items()
                  if first_ev is None or f < first_ev]
    cands = {"S0": init}
    for face in "UDLRFB":
        for suf in ("", "'", "2"):
            mv = face + suf
            cc = Cube()
            for m in scr.split():
                apply_move(cc, m)
            apply_move(cc, mv)
            cands[f"S0+{mv}"] = state_to_array(cc.state)
    totals = {k: 0.0 for k in cands}
    n = 0
    for r in rest_reads:
        seg = seg_factory(r, init_om)
        if not seg.ok:
            continue
        n += 1
        for k, st in cands.items():
            totals[k] += float(seg.score_states(st[None, :])[0])
    return totals, n


def movegt_offset_scores(tag, scr, raw, seg_factory, init_om, offsets=range(-15, 16)):
    """Real movegt global-offset fit scores (Layer D) for
    `offset_gate_verdict`: for each candidate global frame-shift d, the mean
    fit of each in-video movegt move's TRUE post-move state against the read
    at frame+d. Only offsets with read coverage over >=1/4 of the on-camera
    movegt entries are reported (so a hollow offset can't "win" on 2 lucky
    frames). Returns {} when the gate can't run (no scramble/movegt/coverage)."""
    mgp = f"walkthrough/movegt_{tag}.json"
    if scr is None or not os.path.exists(mgp):
        return {}
    mg = [(int(m["frame"]), m["move"]) for m in json.load(open(mgp))
          if int(m["frame"]) >= 0]
    if not mg:
        return {}
    cc = Cube()
    for m in scr.split():
        apply_move(cc, m)
    mg_states = []
    for f, mv in mg:
        apply_move(cc, mv)
        mg_states.append((f, state_to_array(cc.state)))
    off_scores = {}
    for d in offsets:
        tot, n = 0.0, 0
        for f, st in mg_states:
            tf = f + d
            if tf not in raw:
                continue
            r, _m = raw[tf]
            seg = seg_factory(r, init_om)
            if not seg.ok:
                continue
            tot += float(seg.score_states(st[None, :])[0])
            n += 1
        if n >= max(3, len(mg_states) // 4):
            off_scores[d] = tot / n
    return off_scores


def main():
    PERF_TRACE.mark("main_enter")
    tag = argval("--tag", "capture")
    cal = argval("--centroids-json", "calibration.json")
    reads_path = argval("--reads", f"/tmp/reads_{tag}.pkl")
    use_reads = "--no-reads" not in sys.argv
    use_motion = "--no-motion" not in sys.argv
    events_json = argval("--events-json", "")  # event frames from a file (e.g.
    #   the learned CNN's predicted events) — the realistic cube-less source.
    gt_events = "--gt-events" in sys.argv  # ISOLATION: feed the EXACT cube move
    #   frames (movegt) as motion events instead of the noisy ramp detector, to
    #   test the reads/trellis given PERFECT move timing (the cube provides it).

    PERF_TRACE.set_meta(tag=tag, reads=reads_path)
    with PERF_TRACE.span("decode.input_reads_load"):
        with open(reads_path, "rb") as fh:
            raw, frec = pickle.load(fh)
    # Preserve the full-precision, dense FORWARD motion channel before any
    # backward coordinate remap or later read filtering.  The strict EV file is
    # authoritative for event identity; this channel only recovers structural
    # pre-merge phases and never replaces those events.
    try:
        _intraburst_forward_motion = {
            require_exact_integer(frame, "strict raw-motion frame"): value[1]
            for frame, value in raw.items()
        }
    except (TypeError, ValueError, IndexError):
        _intraburst_forward_motion = None
    setup_start_ns = PERF_TRACE.start_ns()
    if "--autocal" in sys.argv:
        # Derive six color centroids from this capture's reads without teacher
        # labels.
        from autocal_from_reads import derive_centroids
        _c = derive_centroids(raw, verbose=True)
        _tmp = f"/tmp/_autocal_{tag}.json"
        json.dump(_c, open(_tmp, "w"))
        base = load_centroids(_tmp)
        print(f"  [autocal] self-calibrated from {len(raw)} reads")
    else:
        base = load_centroids(cal)
    _scramble_override = argval("--scramble", os.environ.get("CUBED_SCRAMBLE"))
    if _scramble_override is not None:
        sess = {"scramble": _scramble_override}
    else:
        _session_path = argval(
            "--session-json", f"walkthrough/session_{tag}.json")
        sess = json.load(open(_session_path))
    scr = sess.get("scramble")
    c0, _no_scr_reason = resolve_start_state(sess)
    # --- scramble=None handling: free-record sessions
    # (session.mode=="free", with no scramble/solve structure) and any other
    # session missing a
    # scramble field used to HARD-CRASH here (`scr.split()` on None) and in
    # reach_ll.py. scripts/build_movegt.py can derive and stamp a
    # `start_state`; resolve_start_state replays it when present. Only a session with
    # NEITHER `scramble` NOR a derived `start_state` falls back to init=SOLVED
    # (the only defensible baseline for a truly unknown start), and every
    # branch is printed so it's never a silent assumption.
    if _no_scr_reason and _no_scr_reason.startswith("derived start state"):
        print(f"  [start-state] tag={tag}: {_no_scr_reason}; using the "
              f"backward-derived start rather than a solved default.")
    elif _no_scr_reason:
        print(f"  [start-state] tag={tag}: no scramble ({_no_scr_reason}); init "
              f"defaults to SOLVED and teacher comparison is unavailable.")
    else:
        print(f"  [start-state] tag={tag}: scramble-derived start "
              f"({'DERIVED' if 'DERIVED' in (sess.get('_scramble_provenance') or '').upper() else 'recorded'})")
    init = state_to_array(c0.state)
    final = state_to_array(Cube().state)
    _teacher_moves = sess.get("moves")
    if _teacher_moves is not None and (
        not isinstance(_teacher_moves, list)
        or any(
            not isinstance(move, dict) or not isinstance(move.get("move"), str)
            for move in _teacher_moves
        )
    ):
        raise SystemExit("session moves must be a list of move records when present")
    teacher_available = _teacher_moves is not None
    session_moves = _teacher_moves or []
    gt = [m["move"] for m in session_moves]
    # --- truncated-video endpoint (--final-from-gt): many recordings were chunk-
    # truncated, so the last teacher moves can happen off-camera and the cube
    # does not end solved on screen. This sets
    # `final` to the GT cube state at the video's LAST frame and scores only the
    # in-video moves. Default OFF => byte-identical (final=solved, gt=all moves).
    if "--final-from-gt" in sys.argv and scr is None:
        print(f"  [final-from-gt] skipped: tag={tag} has no scramble "
              f"(see [start-state] above)")
    elif "--final-from-gt" in sys.argv:
        _afp = f"walkthrough/align_{tag}.json"
        _fps = json.load(open(_afp))["fps"] if os.path.exists(_afp) else 120.0
        _vdur_ms = max(raw) / _fps * 1000.0
        if not teacher_available:
            raise SystemExit("--final-from-gt requires teacher moves")
        _in = [m for m in session_moves if m.get("t_ms", 0) <= _vdur_ms]
        _cf = Cube()
        for _m in scr.split():
            apply_move(_cf, _m)
        for _mv in _in:
            apply_move(_cf, _mv["move"])
        final = state_to_array(_cf.state)
        gt = [m["move"] for m in _in]
        _off = len(session_moves) - len(_in)
        _solv = list(final) == list(state_to_array(Cube().state))
        print("  [final-from-gt] video=%df@%.1ffps=%.1fs; trimmed %d off-camera move(s); "
              "gt %d->%d; final_is_solved=%s" % (max(raw), _fps, _vdur_ms/1000.0, _off,
              len(session_moves), len(gt), _solv))
    fps = json.load(open(f"walkthrough/align_{tag}.json"))["fps"] \
        if os.path.exists(f"walkthrough/align_{tag}.json") else 120.0
    motions = {f: (r.get("motion") if isinstance(r, dict) else None)
               for f, r in frec.items()}

    # BACKWARD decode (--backward): reverse the reads in time and run the
    # UNCHANGED trellis from the SOLVED state. A re-grip that is forward-
    # unidentifiable (post-grip reads don't support the true state; bestfit=None)
    # is ENTERED from the solved side with a known-good anchor, so the backward
    # pass can recover the region the forward pass lost. The trellis then decodes
    # solved->scramble over reversed frames; the real solve = the output reversed
    # and each move inverted (post-processed after track()). This is the cheap
    # validation of the forward-backward hypothesis: no trellis change, just
    # reversed inputs + swapped endpoints.
    backward = "--backward" in sys.argv
    rev = (lambda f: f)
    if backward:
        fmin, fmax = min(raw), max(raw)
        rev = lambda f, _a=fmin, _b=fmax: _a + _b - f
        raw = {rev(f): v for f, v in raw.items()}
        frec = {rev(f): v for f, v in frec.items()}
        motions = {rev(f): v for f, v in motions.items()}
        init, final = final, init          # start at solved, end at scramble
        print(f"  [backward] reversed {len(raw)} reads; init=solved final=scramble")

    trust_meta = {}

    # LEARNED PER-CELL READ-TRUST SOFT WEIGHTING (--trust-soft <model.npz>
    # [--trust-soft-floor 0.15]). Every admitted cell is kept and weighted by
    # max(p_trust, floor) through AbsSegment._wv (exempt no-conf cells at 1.0),
    # so no read is irreversibly dropped — the continuous form of the
    # hard-cutoff. The model is a numpy-only TrustNumpy (detect/trust_numpy.py,
    # loaded from a plain .npz. Same frame metadata (motion, P(aligned), and
    # optional event context) is used by the feature builder. Default off =>
    # byte-identical (TRUST_SOFT stays None).
    if "--trust-soft" in sys.argv:
        from detect.read_trust import centroid_matrix
        from detect.trust_numpy import TrustNumpy
        _tspath = argval("--trust-soft")
        if not _tspath or _tspath.startswith("--"):
            raise SystemExit("--trust-soft requires a model .npz path")
        # weight floor for observed cells. Default 0.15 REUSES the project's
        # established minimum-contribution floor (CONF_W's np.maximum(conf,0.15)
        # and the legacy LAB L-weight 0.15) — not a fresh knob. Load-bearing on
        # junk-dominated captures: floor=0 lets an all-junk read concentrate
        # weight on the marginally-least-junk cell. The floor prevents that
        # concentration.
        _tsfloor = argval("--trust-soft-floor", 0.15, float)
        _tsmodel = TrustNumpy.load(_tspath)
        _cenm, _cnames = centroid_matrix(base)
        AbsSegment.TRUST_SOFT = {"model": _tsmodel, "cen_mat": _cenm,
                                 "cache": {}, "floor": _tsfloor,
                                 "n_features": _tsmodel.n_features}
        AbsSegment.TRUST_SOFT_STATS = {"reads_scored": 0, "cells": 0,
                                       "exempt": 0, "p_hist": [0] * 20}
        _taf = argval("--align-feat", f"/tmp/alignfeat_{tag}_new.npz")
        _talign = {}
        if os.path.exists(_taf):
            _tz = np.load(_taf, allow_pickle=True)
            _talign = dict(zip(_tz["frame"].tolist(), _tz["aligned"].tolist()))
        _tev = {}
        if _tsmodel.n_features == 17:
            from detect.read_trust import EV_CAP, event_context
            _tevsrc = events_json if (events_json
                                      and os.path.exists(events_json)) else None
            _evfr = (sorted(int(e["frame"]) for e in
                            json.load(open(_tevsrc))) if _tevsrc else None)
            _tfr = [rev(f) for f in raw]
            _es, _et = event_context(_tfr, _evfr)
            _tev = {f: (float(s), float(t)) for f, s, t in zip(raw, _es, _et)}
            if not _tevsrc:
                print(f"  [trust-soft] !! v3 model but NO events file — ev "
                      f"features fall back to the {EV_CAP:.0f} missing encoding")
        trust_meta = {id(r): (m, _talign.get(rev(f)),
                              _tev.get(f, (None, None))[0],
                              _tev.get(f, (None, None))[1])
                      for f, (r, m) in raw.items()}
        print(f"  [trust-soft] model={_tspath} md5={_tsmodel.source_md5} "
              f"floor={_tsfloor} n_features={_tsmodel.n_features} — per-cell "
              f"SOFT p_trust weighting via _wv (weight=max(p_trust,floor); "
              f"no-conf cells exempt at 1.0). numpy-only (no sklearn at "
              f"decode). Default off => byte-identical.")
        print(f"  [trust-soft] d1 centroids={cal} ({len(_cnames)} colors); "
              f"aligned source={_taf if _talign else 'ABSENT -> NaN/0.5'}")

    # per-cell confidence (geo_read sets vis["_cellconf"] = {slot:[9]}); occluded
    # / finger cells get low conf -> AbsSegment excludes/down-weights them
    # (CONF_BLANK / CONF_W). Hoisted above the om0/palette block (previously
    # built only for seg_factory, further below) so those two AbsSegment
    # consumers are also confidence-aware; otherwise low-quality opening reads
    # can contaminate orientation and palette estimation.
    # Inert when CONF_BLANK/CONF_W are both off (the default): _confs is
    # stored but never read by score_states unless CONF_W builds _wv from it.
    cc_by_id = {id(r): (frec.get(f, {}).get("vis") or {}).get("_cellconf")
                for f, (r, m) in raw.items()}

    # reads -> palette refresh + reads-resolved om0 (baseline orientation)
    fs = [r for i, (r, m) in sorted(raw.items()) if m is not None and m <= 10][:25]
    om0_reads, _ = establish_orientation_from_reads(
        fs, init, base,
        cell_confs=[cc_by_id.get(id(r)) for r in fs],
        frame_meta=[trust_meta.get(id(r)) for r in fs] if trust_meta
        else None) if fs else (None, 0)
    labeled = []
    for r in fs[:20]:
        _tfm = trust_meta.get(id(r)) if trust_meta else None
        s = AbsSegment(r, base, om0_reads,
                       cell_conf=cc_by_id.get(id(r)),
                       frame_motion=_tfm[0] if _tfm else None,
                       frame_aligned=_tfm[1] if _tfm else None,
                       frame_ev=(_tfm[2], _tfm[3])
                       if _tfm and len(_tfm) > 2 else None) \
            if om0_reads else None
        if s and s.ok:
            labeled += s.labeled_for_state(init)
    pal = refresh_palette_from_glance(base, labeled) if labeled else base

    print(f"== {tag} ==  reads:{use_reads} motion:{use_motion}")
    print(f"  om0 reads-resolved: {om0_reads}")
    init_om = om0_reads

    vis_by_id = {id(r): frec.get(f, {}).get("vis") for f, (r, m) in raw.items()}

    def _segment_metadata(r, _v=vis_by_id, _cc=cc_by_id, _tm=trust_meta):
        _fm = _tm.get(id(r)) if _tm else None
        return {
            "face_vis": _v.get(id(r)),
            "cell_conf": _cc.get(id(r)),
            "frame_motion": _fm[0] if _fm else None,
            "frame_aligned": _fm[1] if _fm else None,
            "frame_ev": ((_fm[2], _fm[3])
                         if _fm and len(_fm) > 2 else None),
        }

    # Pure-speed evidence path.  A full AbsSegment construction repeats the
    # same LAB admission, centroid distances, and trust inference for every
    # orientation; only the integer gather matrix changes.  The solve-local
    # factory builds that evidence once per read and cheaply clones the gather
    # for later orientations.  The old factory remains an explicit A/B fallback.
    _cached_segments = os.environ.get("CUBED_ABS_SEGMENT_CACHE", "0") == "1"
    _batched_trust = os.environ.get("CUBED_GPU_TRUST_BATCH", "0") == "1"
    _segment_factory_stats = None
    if _cached_segments or _batched_trust:
        from detect.abs_segment_cache import CachedAbsSegmentFactory
        _segment_factory_stats = CachedAbsSegmentFactory(
            pal, metadata=_segment_metadata)
        seg_factory = _segment_factory_stats
    else:
        segcache = {}

        def seg_factory(r, om, _pal=pal, _c=segcache):
            key = (id(r), om["up"], om["front"])
            if key not in _c:
                _c[key] = AbsSegment(r, _pal, om, **_segment_metadata(r))
            return _c[key]

    # --gt-qc-gates: benchmark-hygiene QC (Layers C+D). Opt-in,
    # prints only — never touches `tt`/decode state (default off => byte-
    # identical, matches every other diagnostic in this file).
    if "--gt-qc-gates" in sys.argv:
        _prov = sess.get("_scramble_provenance") or ""
        if scr is not None and "DERIVED" in _prov.upper():
            _rf_scores, _rf_n = rest_fit_scores(tag, scr, raw, seg_factory, init_om, init)
            if _rf_n:
                _winner, _margin, _ok = rest_fit_verdict(_rf_scores)
                if _ok:
                    print(f"  [rest-fit-gate] OK tag={tag}: S0 wins the k=0 rest-window "
                          f"fit ({_rf_n} frames), margin={_margin / _rf_n:+.3f}/frame "
                          f"over the best single-move perturbation.")
                else:
                    print(f"  [rest-fit-gate] WARNING tag={tag}: S0 does NOT win the "
                          f"k=0 rest-window fit ({_rf_n} frames) — best={_winner} beats "
                          f"S0 by {-_margin / _rf_n:+.3f}/frame. The derived "
                          f"scramble may be misregistered; inspect it before "
                          f"using frame-level teacher comparisons.")
            else:
                print(f"  [rest-fit-gate] SKIP tag={tag}: no rest-window reads before "
                      f"the first on-camera movegt event.")
        else:
            print(f"  [rest-fit-gate] SKIP tag={tag}: not a scramble-derived tag "
                  f"(no 'DERIVED' _scramble_provenance stamp) — gate only applies to "
                  f"derived-S0 sessions.")
        _off_scores = movegt_offset_scores(tag, scr, raw, seg_factory, init_om)
        if _off_scores:
            _best_d, _off_ok = offset_gate_verdict(_off_scores)
            if _off_ok:
                print(f"  [movegt-offset-gate] OK tag={tag}: offset=0 is best-fit "
                      f"({_off_scores[0]:+.3f}/frame) across {len(_off_scores)} "
                      f"tested offsets — no detected frame-map drift.")
            else:
                print(f"  [movegt-offset-gate] WARNING tag={tag}: best-fit global "
                      f"frame offset = {_best_d:+d} (score {_off_scores[_best_d]:+.3f}"
                      f"/frame vs offset=0 {_off_scores.get(0, float('nan')):+.3f}"
                      f"/frame) — the frame map may be drifted. Inspect it before "
                      f"using frame-level teacher comparisons for tag={tag}.")
        else:
            print(f"  [movegt-offset-gate] SKIP tag={tag}: no movegt/scramble, or "
                  f"insufficient read coverage to sweep frame offsets.")

    # reads channel: pass empty when ablated
    reads_in = raw if use_reads else {}

    tt = TrellisTracker(seg_factory, allow_reorient=True, initial_om=init_om,
                        gapf=5 if fps == 60 else 10, rotpen=8.0)
    tt.fps = fps
    # Keep multi-hypothesis orientation resolution enabled so the reads can
    # recover when the opening estimate is ambiguous.
    tt.init_om_hyps = 4
    tt.K_reorient = 32
    # SPAN-TRELLIS COUNT/BALL knobs (default-preserving: absent flag => unchanged).
    # Adjacent moves can collapse into one motion burst. --countpen relaxes the
    # soft move-count penalty;
    # --ball-d / --dur-hi-rate let the per-span ball REACH a multi-move state and
    # widen the no-penalty count window so the post-burst readable rest pulls the
    # extra move(s) through.
    if "--countpen" in sys.argv:
        tt.countpen = argval("--countpen", tt.countpen, float)
    if "--countpen-deep" in sys.argv:
        print("  [countpen-deep] WARNING: experimental; deeper count penalties "
              "can admit decoys and flood the beam", flush=True)
        tt.countpen_deep = argval("--countpen-deep", tt.countpen_deep, float)
    if "--ball-d" in sys.argv:
        tt.d = argval("--ball-d", tt.d, int)
    if "--dur-hi-rate" in sys.argv:
        tt.dur_hi_rate = argval("--dur-hi-rate", tt.dur_hi_rate, float)
    if "--dfinal" in sys.argv:
        tt.dfinal = argval("--dfinal", tt.dfinal, int)
        print(f"  [dfinal] terminal-bridge depth = {tt.dfinal}")
    if "--lam" in sys.argv:
        tt.lam = argval("--lam", tt.lam, float)
        print(f"  [lam] bridge length penalty = {tt.lam}")
    # OLL/PLL ALG-DICTIONARY PROGRESS PRIOR (detect/ll_alg_prior.py): at F2L-complete,
    # softly bias the beam along a recognized OLL->PLL solution through the fast
    # last-layer flurry. Default off => bit-identical.
    if "--ll-prior" in sys.argv:
        tt.ll_prior = argval("--ll-prior", 0.0, float)
        if "--ll-gate-fit" in sys.argv:
            tt.ll_gate_fit = argval("--ll-gate-fit", None, float)
        if "--ll-cap" in sys.argv:
            tt.ll_cap = argval("--ll-cap", tt.ll_cap, int)
        if "--robust-arm" in sys.argv:
            tt.robust_arm = argval("--robust-arm", 0, int)  # tolerant F2L-complete arming (moves)
            print(f"  [robust-arm] tolerant F2L-complete arming within "
                  f"{tt.robust_arm} move(s) (anti read-brittleness)")
        print(f"  [ll-prior] OLL/PLL alg-dictionary prior = {tt.ll_prior} "
              f"(gate fit>={tt.ll_gate_fit}, cap={tt.ll_cap})")
    # PER-CELL SOFT EMISSION (visibility weights): --conf-w weights each sticker's
    # score by its extraction confidence (cell_conf -> AbsSegment._wv; low-conf /
    # foreshortened cells contribute partially); --seg-weight-n weights each frame's
    # vote in a span by its visible-sticker count (a low-coverage frame can't drive a
    # confident move). Default off => bit-identical.
    if "--conf-w" in sys.argv:
        print("  [conf-w] WARNING: experimental; soft confidence weighting can "
              "destabilize low-coverage reads", flush=True)
        AbsSegment.CONF_W = True
        print("  [conf-w] per-cell confidence weighting ON")
    if "--dist-cap" in sys.argv:
        AbsSegment.DIST_CAP = argval("--dist-cap", None, float)  # robust per-cell ceiling
        print(f"  [dist-cap] robust per-cell LAB-distance ceiling = {AbsSegment.DIST_CAP} "
              f"(anti read-brittleness)")
    # HARD CELL-CONFIDENCE BLANK (evidence-path fix; distinct from --conf-w's
    # SOFT down-weighting above): --conf-blank [THR] excludes any cell whose
    # cell_conf is below THR from AbsSegment's evidence entirely — the same
    # hard treatment as MIN_STICKER_L, not a partial weight. THR is optional
    # (bare --conf-blank => 0.35). This prevents hand or occlusion artifacts
    # from contributing as sticker evidence. Default off => byte-identical
    # (AbsSegment.CONF_BLANK stays None).
    if "--conf-blank" in sys.argv:
        _cbi = sys.argv.index("--conf-blank") + 1
        _cbv = (sys.argv[_cbi] if _cbi < len(sys.argv)
                and not sys.argv[_cbi].startswith("--") else None)
        AbsSegment.CONF_BLANK = float(_cbv) if _cbv is not None else 0.35
        # prereq instrumentation: per-run blanked/kept/exempt counts
        # + a conf histogram, printed in the end-of-run [conf-blank] summary.
        AbsSegment.CONF_BLANK_STATS = {"blanked": 0, "kept": 0, "exempt": 0,
                                        "hist": [0] * 20}
        print(f"  [conf-blank] hard cell-confidence blank thr={AbsSegment.CONF_BLANK} "
              f"(cells below this conf contribute NOTHING, alongside MIN_STICKER_L)")
    if "--seg-weight-n" in sys.argv:
        tt.seg_weight_n = True
        print("  [seg-weight-n] per-frame sticker-count weighting ON")
    # CONFIDENT-READ STATE RE-ANCHOR (Mechanism A: PROMOTE) — detect/trellis_tracker.py.
    # At a confident anchor span, if the read-fit DECISIVELY prefers a CARRIED state
    # over the cum-leader, re-weight cum by the fit advantage and re-rank (no new
    # states; the cascade-containment lever). All knobs; master flag default off =>
    # bit-identical. Defaults are the STRICT (3-face) gate; relax --anchor-min-faces /
    # --anchor-min-nstk to fire on the PARTIAL reads that dominate mid-solve.
    if "--anchor-resync" in sys.argv:
        tt.anchor_resync = True
        tt.anchor_min_faces = argval("--anchor-min-faces", tt.anchor_min_faces, int)
        tt.anchor_min_nstk = argval("--anchor-min-nstk", tt.anchor_min_nstk, int)
        tt.anchor_fit_thr = argval("--anchor-fit-thr", tt.anchor_fit_thr, float)
        tt.anchor_cellconf_thr = argval("--anchor-cellconf-thr", tt.anchor_cellconf_thr, float)
        tt.anchor_min_frames = argval("--anchor-min-frames", tt.anchor_min_frames, int)
        tt.anchor_promote_gap = argval("--anchor-promote-gap", tt.anchor_promote_gap, float)
        tt.anchor_weight = argval("--anchor-weight", tt.anchor_weight, float)
        tt.log_anchor = "--log-anchor" in sys.argv
        print(f"  [anchor] PROMOTE re-anchor ON: faces>={tt.anchor_min_faces} "
              f"nstk>={tt.anchor_min_nstk} fit>={tt.anchor_fit_thr} "
              f"cellconf>={tt.anchor_cellconf_thr} frames>={tt.anchor_min_frames} "
              f"promote_gap={tt.anchor_promote_gap} weight={tt.anchor_weight}")
    # Legacy evaluation preset: om-pf-reads plus orientation rescue.
    if "--gt-mode" in sys.argv:
        tt.init_om_hyps = 4
        tt.K_reorient = 32
        tt.om_on_misfit = True
        tt.misfit_thr = argval("--misfit-thr", -12.0, float)
        tt.om_rescue_extra = 6
        tt.om_rescue_topk = 6
        tt.om_decisive_gap = 0.0
        tt.om_rescue_deep = 0
        tt.om_per_frame = True
        tt.om_per_frame_from_reads = True
        tt.om_pf_margin = argval("--om-pf-margin", 2.0, float)
        tt.om_pf_min_frames = 0
        if "--dfinal" not in sys.argv:
            tt.dfinal = 5
        print(f"  [gt-mode] dashboard gt-mode config: om-pf-reads + om-rescue topk/extra=6, "
              f"misfit_thr={tt.misfit_thr}, dfinal={tt.dfinal}")
    # IN-BURST CO-COMMIT (wide/slice decode; docs/FAITHFUL_MOVES.md Phase 2). When
    # the main switch set is empty (the gt-mode default), a span may switch om at
    # burst_used=0 to the single-rotation neighbor oms at `--widepen` (default
    # rotpen), so a wide is a 1-burst candidate. Default off => bit-identical.
    if "--co-commit" in sys.argv:
        tt.co_commit = True
        if "--widepen" in sys.argv:
            tt.widepen = argval("--widepen", None, float)
        print(f"  [co-commit] in-burst wide/slice co-commit ON "
              f"(widepen={tt.widepen if tt.widepen is not None else tt.rotpen})")
    # CLEAN-REST GATING (key-risk mitigation): a deeper ball / looser count admits
    # DECOYS at low-observability re-grips. With --clean-gate, the relaxed
    # countpen + deepened ball apply ONLY to spans whose bracketing reads give a
    # decisive state (prev-anchor span-fit >= --clean-fit-thr); re-grip / occluded
    # spans keep the strict legacy count+depth. Needs the per-span clean signal
    # wired in the tracker (tt.clean_gate). Default off => bit-identical.
    if "--clean-gate" in sys.argv:
        tt.clean_gate = True
        tt.clean_fit_thr = argval("--clean-fit-thr", -10.0, float)
        # strict (legacy) values used on NON-clean spans
        tt.clean_strict_countpen = argval("--clean-strict-countpen", 0.6, float)
        tt.clean_strict_d = argval("--clean-strict-d", 2, int)
        tt.clean_max_gap = argval("--clean-max-gap", 10**9, int)
        print(f"  [clean-gate] relaxed count/ball gated to clean rests "
              f"(fit>={tt.clean_fit_thr}, gap<={tt.clean_max_gap}); non-clean keep "
              f"countpen={tt.clean_strict_countpen} d={tt.clean_strict_d}")
    if any(f in sys.argv for f in ("--countpen", "--countpen-deep", "--ball-d",
                                   "--dur-hi-rate")):
        print(f"  [count/ball] countpen={tt.countpen} "
              f"countpen_deep={tt.countpen_deep} d={tt.d} "
              f"dur_hi_rate={tt.dur_hi_rate}")
    if "--no-om-rescue" not in sys.argv:
        # re-establish the om when the reads stop fitting (a re-grip happened):
        # re-search orientations on misfit spans, ranked by the previous state's
        # fit (centers are move-invariant) and only the top-k transitioned. This is
        # This lets the trellis recover its orientation after a re-grip. ON by
        # default; --no-om-rescue disables.
        # rotpen_misfit stays at the default (rotpen): lowering it over-switches.
        tt.om_on_misfit = True
        tt.misfit_thr = argval("--misfit-thr", -12.0, float)
        tt.om_rescue_extra = argval("--om-rescue-extra", 6, int)
        tt.om_rescue_topk = argval("--om-rescue-topk", 6, int)  # 0 = all 24
        tt.om_decisive_gap = argval("--om-decisive-gap", 0.0, float)  # 0 = off
        tt.om_rescue_deep = argval("--om-rescue-deep", 0, int)  # ball depth, 0=off
        tt.deep_count_bound = "--deep-count-bound" in sys.argv  # clamp deep ball depth to exp_hi
        if "--move-gate-under" in sys.argv:
            print("  [move-gate-under] WARNING: experimental global threshold; "
                  "it is not reads-aware", flush=True)
        tt.move_gate_under = argval("--move-gate-under", 1.0, float)  # <1 relaxes phantom-move pressure
        # CO-OPERATIVE TRELLIS<->MOTION GATE (reads-veto on phantom move insertion;
        # see TrellisTracker.coop_gate). Single-pass, reads-aware: veto a count/ball-
        # forced move at a span when the reads DECISIVELY prefer staying
        # (read_fit(stay)-read_fit(best_move) >= margin). Default off => byte-identical.
        if "--coop-gate" in sys.argv:
            tt.coop_gate = True
            tt.coop_gate_margin = argval("--coop-gate-margin", tt.coop_gate_margin, float)
            tt.coop_gate_log = "--coop-gate-log" in sys.argv
            print(f"  [coop-gate] reads-veto on phantom move insertion ON "
                  f"(veto move when read_fit(stay)-read_fit(best_move) >= "
                  f"{tt.coop_gate_margin})")
        if ("--align-transition-sample" in sys.argv
                and not all(flag in sys.argv
                            for flag in ("--move-gate", "--align-gate"))):
            raise ValueError("--align-transition-sample requires "
                             "--move-gate --align-gate")
        if "--move-gate" in sys.argv:
            tt._use_move_gate_2pass = True
            tt.move_gate_margin = argval("--move-gate-margin", 3.0, float)
            if "--move-gate-allframes" in sys.argv:
                tt.move_gate_allframes = True
                tt.move_gate_max_frames = argval("--move-gate-max-frames", 80, int)
            if "--align-gate" in sys.argv:
                _afp = argval("--align-feat", f"/tmp/alignfeat_{tag}_new.npz")
                _ad = np.load(_afp, allow_pickle=True)
                tt.align_feat = dict(zip(_ad["frame"].tolist(), _ad["aligned"].tolist()))
                tt.align_gate = True
                tt.align_min_run = argval("--align-min-run", 2, int)
                # Keep this distinct from any read-admission threshold: it
                # controls an alignment transition gate, not frame admission.
                tt.align_thr = argval("--align-thr", 0.3, float)
                tt.align_gate_mode = argval("--align-gate-mode", "or")
                tt.align_transition_sample = "--align-transition-sample" in sys.argv
                if "--align-expand" in sys.argv:
                    tt.align_expand = True
                    tt.align_search_r = argval("--align-search-r", 12, int)
                if "--align-gm-protect" in sys.argv:
                    tt.align_gm_protect = argval("--align-gm-protect", 3.0, float)
                    tt.align_gm_protect_ab_min = argval("--align-gm-protect-ab", 0, int)
                print(f"  [align-gate] {len(tt.align_feat)} frames from {_afp}; "
                      f"min_run={tt.align_min_run} thr={tt.align_thr} "
                      f"mode={tt.align_gate_mode} expand={tt.align_expand} "
                      f"gm_protect={tt.align_gm_protect} "
                      f"transition_sample={tt.align_transition_sample}")
            print(f"  [move-gate] two-pass reads move-vs-re-grip event gate ON "
                  f"(drop events with state-change margin < {tt.move_gate_margin}"
                  f"{'; ALL-FRAMES margin' if tt.move_gate_allframes else ''})")
        if "--soft-witness" in sys.argv:
            tt._use_move_gate_2pass = True
            tt.soft_witness = True
            _lo = argval("--soft-lo", 0.0, float)
            _hi = argval("--soft-hi", 4.0, float)
            tt.soft_lo_range = (_lo, _hi)
            print(f"  [soft-witness] reads-weighted soft event move-floor ON "
                  f"(w=clip((margin-{_lo})/({_hi}-{_lo}),0,1))")
        rgf = argval("--regrip-frames", None)
        if rgf:                                  # scope deep re-anchor to re-grips
            tt.regrip_frames = set(int(x) for x in json.load(open(rgf)))
            print(f"  [regrip-scope] deep re-anchor limited to "
                  f"{len(tt.regrip_frames)} re-grip frames")
    if "--gate-cov-min" in sys.argv:
        # EXPERIMENTAL coverage-admissible move-gate (lever, NOT
        # canonical): color re-grip verdicts whose gap coverage
        # (visible∩changed / changed) is below this floor ABSTAIN — the event
        # survives unless alignment vetoes it independently.
        print("  [gate-cov-min] WARNING: experimental; union coverage has no "
              "established decision floor", flush=True)
        tt.gate_cov_min = float(argval("--gate-cov-min"))
        print(f"  [gate-cov] EXPERIMENTAL coverage-admissible move-gate ON: "
              f"color verdicts with coverage < {tt.gate_cov_min} abstain",
              flush=True)
    if "--verbose" in sys.argv:                  # live per-span + re-anchor log
        tt.log_reanchor = True
        tt.span_purity_log = []                  # straddled-span diagnostic
        tt.span_collapse_log = []                 # collapse-rate diagnostic
        tt.covt_commit_log = []                   # commit-time covT
        rpm = argval("--rotpen-misfit", None)
        if rpm is not None:
            tt.rotpen_misfit = float(rpm)
        print(f"  [om-rescue] on, misfit_thr={tt.misfit_thr} "
              f"topk={tt.om_rescue_topk} rotpen_misfit={tt.rotpen_misfit}")
    if "--span-purity-split" in sys.argv:        # straddled-span SPLIT
        # BEHAVIOR (not just diagnostic): a decisively-impure settled span
        # (a real move with no event inside it — the still-gate hole) is
        # re-run as two frame-median sub-spans so the buried move is
        # recovered. Armed independently of --verbose (ball-based test).
        tt.span_purity_split = True
        print("  [span-purity] SPLIT behavior ON; decisive "
              "straddled spans re-run as two frame-median sub-spans)",
              flush=True)
    if "--strat-span-sample" in sys.argv:        # lever L1 / lever (2)
        # Optional mode value: bare flag = GLOBAL (legacy, FALSIFIED as a
        # blanket policy); `cond` = CONDITIONAL (lever
        # (2), stratify only a collapsed span with a wide-enough empty side).
        # Same "peek at the next token" idiom as --conf-blank above.
        _sssi = sys.argv.index("--strat-span-sample") + 1
        _sssv = (sys.argv[_sssi] if _sssi < len(sys.argv)
                 and not sys.argv[_sssi].startswith("--") else None)
        if _sssv == "cond":
            # CONDITIONAL variant (NOT canonical, EXPERIMENTAL): the bare
            # flag's GLOBAL resample perturbs pass-1's gate/span structure.
            # This mode instead
            # stratifies only a span whose quality-ranked sample has
            # COLLAPSED with an empty side >= the derived per-fps threshold
            # (TrellisTracker.COLLAPSE_HIDE_S).
            tt.strat_span_sample_cond = True
            print("  [strat-span-sample] CONDITIONAL mode ON; resampling fires only "
                  "on a collapsed span with emptygap >= the derived "
                  f"per-fps threshold, thr={tt._collapse_strat_threshold()} "
                  f"frames @ fps={tt.fps})", flush=True)
        elif os.environ.get("CUBED_ALLOW_FALSIFIED") != "1":
            # Bare/GLOBAL mode perturbs every span, so refuse to run it
            # silently. The escape hatch is only for deliberate measurement.
            raise SystemExit(
                "--strat-span-sample (bare/global mode) is unsupported; use "
                "'--strat-span-sample cond'. To force the falsified mode "
                "for a deliberate re-measurement, set "
                "CUBED_ALLOW_FALSIFIED=1")
        else:
            # EXPERIMENTAL main-commit sampler swap (NOT canonical): the span
            # score's `ranked[:span_subsample]` sample is quality-ranked only,
            # which can be monopolized by one temporal rest of a
            # straddled span. Swap in the TIME-STRATIFIED sample (same
            # construction as --span-purity-split's straddle test) for the MAIN
            # commit's span score.
            tt.strat_span_sample = True
            print("  [strat-span-sample] EXPERIMENTAL main-commit sampler ON; "
                  "time-stratified span sampling replaces pure quality rank",
                  flush=True)
    if "--teacher-forced" in sys.argv:           # per-move harness
        # TEACHER-FORCED per-gap comparison (DIAGNOSTIC ONLY — the named
        # reach-LL evaluation remains separate):
        # only success/promotion gate): every span transition starts from the
        # oracle (true state @ preceding anchor = the READ-ALIGNED movegt
        # replay, see TrellisTracker._tf_align_anchors; om chained through
        # the production channel — movegt has no orientation) and each gap's
        # commit is compared with its aligned teacher bucket.
        if backward:
            raise SystemExit("--teacher-forced is incompatible with "
                             "--backward (movegt frames are not remapped "
                             "into reversed time)")
        _tfp = f"walkthrough/movegt_{tag}.json"
        if not os.path.exists(_tfp):
            raise SystemExit(f"--teacher-forced needs {_tfp} "
                             f"(GT moves + frame stamps)")
        tt.teacher_forced = sorted(
            ((int(m["frame"]), m["move"]) for m in json.load(open(_tfp))),
            key=lambda t: t[0])
        print(f"  [teacher-forced] ON: {len(tt.teacher_forced)} teacher moves "
              f"from {_tfp}; per-gap oracle prefix (read-aligned true state "
              f"@ anchor, orientation_policy=chained). This is an isolated "
              f"diagnostic, not a camera-only decode result.",
              flush=True)
    # Z-NORM DERIVED GATES (default off => byte-identical). De-knob the three
    # raw-LAB-scale hand constants (misfit_thr / move_gate_margin / om_pf_margin)
    # into per-solve z-scores vs the solve's OWN anchor fit-noise (mu, sigma),
    # estimated from the confident anchor spans the anchor_resync coverage test
    # already locates (detect/trellis_tracker.py _note_anchor_fit). The k's are
    # dimensionless and default-calibrated on the dev tags so the derived
    # thresholds reproduce the configured constants there and adapt on new
    # captures. --znorm-log alone
    # prints mu/sigma/n + the would-be derived thresholds WITHOUT changing the
    # decode (measurement mode for calibrating the k's).
    if "--znorm-log" in sys.argv:
        tt.log_znorm = True
    if "--znorm-gates" in sys.argv:
        tt.znorm_gates = True
        tt.log_znorm = True
        tt.znorm_k_misfit = argval("--znorm-k-misfit", tt.znorm_k_misfit, float)
        tt.znorm_k_move = argval("--znorm-k-move", tt.znorm_k_move, float)
        tt.znorm_k_ompf = argval("--znorm-k-ompf", tt.znorm_k_ompf, float)
        tt.znorm_min_anchors = argval("--znorm-min-anchors",
                                      tt.znorm_min_anchors, int)
        print(f"  [znorm-gates] ON: k_misfit={tt.znorm_k_misfit} "
              f"k_move={tt.znorm_k_move} k_ompf={tt.znorm_k_ompf} "
              f"min_anchors={tt.znorm_min_anchors} "
              f"(misfit_thr/move_gate_margin/om_pf_margin -> mu-k*sigma / "
              f"k*sigma vs per-solve anchor fit-noise)")
    # CALIBRATED MODE (recalibration phase A; default off => byte-identical).
    # --calibrated (a) turns ON the fitted Gaussian+uniform emission model
    # (AbsSegment.EMISSION: read evidence becomes -log p(LAB|color) in NATS,
    # replacing the hand-set _LAB_W distance; junk saturates at the outlier
    # plateau by construction) and (b) swaps the TEMPORAL/STRUCTURAL constants
    # for the data-estimated values (scripts/calibrate_constants.py). Every
    # OTHER mechanism — all gates (still/gapf/rotpen/misfit_thr/
    # move_gate_margin), rescue/anchor/move-gate machinery — is untouched;
    # soft-gate conversion is phase B. Files: --emission-file /
    # $CUBED_EMISSION_FILE (default data/emission_model.json) and
    # --constants-file / $CUBED_CALIBRATED_CONSTANTS (default
    # data/calibrated_constants.json). Placed AFTER om0/palette
    # resolution (those stay on the legacy read path) and after every other
    # config block, so the values printed here are the values that decode.
    # PHASE-B GRANULARITY: the
    # phase-A pieces are separately toggleable so emission and constants can be
    # A/B'd independently (the ablation the phase-A e2e verdict could not run).
    # Exactly one mode may be picked; plain --calibrated is byte-identical to
    # the phase-A behavior (same loads, same echo).
    _c_full = "--calibrated" in sys.argv
    _c_emis = "--calibrated-emission-only" in sys.argv
    _c_cons = "--calibrated-constants-only" in sys.argv
    if (_c_full + _c_emis + _c_cons) > 1:
        raise SystemExit("pick ONE of --calibrated / --calibrated-emission-only"
                         " / --calibrated-constants-only")
    if _c_full or _c_emis or _c_cons:
        from detect.trellis_tracker import (load_emission_model,
                                            apply_calibrated_constants)
        emf = argval("--emission-file",
                     os.environ.get("CUBED_EMISSION_FILE")
                     or "data/emission_model.json")
        ccf = argval("--constants-file",
                     os.environ.get("CUBED_CALIBRATED_CONSTANTS")
                     or "data/calibrated_constants.json")
        if _c_emis or _c_full:
            AbsSegment.EMISSION = load_emission_model(emf)
        if _c_cons or _c_full:
            _applied = apply_calibrated_constants(tt, json.load(open(ccf)))
        if _c_full:
            print(f"  [calibrated] EMISSION ON from {emf} "
                  f"(colors={sorted(AbsSegment.EMISSION['mu'])}, "
                  f"pi_out={AbsSegment.EMISSION['pi_out']}); constants from {ccf}: "
                  + " ".join(f"{k}={v}" for k, v in sorted(_applied.items())))
        elif _c_emis:
            print(f"  [calibrated-emission-only] EMISSION ON from {emf} "
                  f"(colors={sorted(AbsSegment.EMISSION['mu'])}, "
                  f"pi_out={AbsSegment.EMISSION['pi_out']}); structural "
                  f"constants UNTOUCHED (stock)")
        else:
            print(f"  [calibrated-constants-only] EMISSION OFF (legacy LAB "
                  f"distance); constants from {ccf}: "
                  + " ".join(f"{k}={v}" for k, v in sorted(_applied.items())))
    # PER-CONSTANT OVERRIDES (phase-B granularity): --switchpen /
    # --move-sec-typ / --move-sec-min are NEW standalone flags; together with
    # the pre-existing --countpen/--countpen-deep/--lam/--dur-hi-rate/--ball-d
    # every calibrated constant is individually settable. Explicit flags WIN
    # over any constants file: the pre-existing flags' original blocks run
    # BEFORE the constants file and would otherwise be silently clobbered, so
    # under a constants mode they are re-asserted here. Placement (after the
    # calibrated block) is what makes single-constant A/B on top of
    # --calibrated possible. No flags => byte-identical.
    _pc_new = [("--switchpen", "switchpen", float),
               ("--move-sec-typ", "MOVE_SEC_TYP", float),
               ("--move-sec-min", "MOVE_SEC_MIN", float)]
    _pc_old = [("--countpen", "countpen", float),
               ("--countpen-deep", "countpen_deep", float),
               ("--lam", "lam", float),
               ("--dur-hi-rate", "dur_hi_rate", float),
               ("--ball-d", "d", int)]
    _pc_hit = ([t for t in _pc_new if t[0] in sys.argv]
               + [t for t in _pc_old
                  if t[0] in sys.argv and (_c_full or _c_cons)])
    if _pc_hit:
        for _f, _attr, _cast in _pc_hit:
            setattr(tt, _attr, argval(_f, getattr(tt, _attr), _cast))
        print("  [constants] effective (explicit flags beat constants files): "
              + " ".join(f"{a}={getattr(tt, a)}"
                         for _f, a, _c in (_pc_new + _pc_old)))
    # FIT-SCALE CONVERSION under EMISSION (phase-B core).
    # --emission-lab-adapter: affine dist -> a + b*dist derived from the
    # LOADED model's landmarks, mapping nats onto the LAB-equivalent scale —
    # every fit-scale gate AND the evidence<->prior balance keep their tuned
    # values (the global scale refit in the LAB currency).
    # --derived-nats-gates: the gate-only ablation arm — converts ONLY
    # misfit_thr / move_gate_margin / om_pf_margin to raw-nats equivalents,
    # leaving the evidence untransformed (isolates gate rescale from the
    # global balance, per the COUNT_BOUND diagnostic). Mutually exclusive.
    _adapter = "--emission-lab-adapter" in sys.argv
    _dgates = "--derived-nats-gates" in sys.argv
    if _adapter or _dgates:
        if AbsSegment.EMISSION is None:
            raise SystemExit("--emission-lab-adapter / --derived-nats-gates "
                             "need EMISSION on (--calibrated or "
                             "--calibrated-emission-only)")
        if _adapter and _dgates:
            raise SystemExit("--emission-lab-adapter and --derived-nats-gates "
                             "are mutually exclusive (double conversion)")
        from detect.trellis_tracker import (derive_emission_adapter,
                                            emission_fit_landmarks)
        _g = argval("--adapter-good", 7.5, float)
        _j = argval("--adapter-junk", 22.0, float)
        if _adapter:
            _ad = derive_emission_adapter(AbsSegment.EMISSION, _g, _j)
            AbsSegment.EMISSION_ADAPTER = (_ad["a"], _ad["b"])
            print(f"  [emission-lab-adapter] dist -> {_ad['a']:+.3f} + "
                  f"{_ad['b']:.3f}*dist (D_in={_ad['d_in']:.2f} "
                  f"D_out={_ad['d_out']:.2f} -> good~-{_g} junk~-{_j}); "
                  f"fit-scale gates keep their LAB-tuned values")
        else:
            _din, _dout = emission_fit_landmarks(AbsSegment.EMISSION)
            _q = argval("--misfit-q", 0.35, float)
            _b = (_j - _g) / (_dout - _din)
            _old = (tt.misfit_thr, tt.move_gate_margin, tt.om_pf_margin)
            tt.misfit_thr = -(_din + _q * (_dout - _din))
            tt.move_gate_margin = tt.move_gate_margin / _b
            tt.om_pf_margin = tt.om_pf_margin / _b
            print(f"  [derived-nats-gates] q={_q} b={_b:.3f} "
                  f"(D_in={_din:.2f} D_out={_dout:.2f}): misfit_thr "
                  f"{_old[0]}->{tt.misfit_thr:.2f}, move_gate_margin "
                  f"{_old[1]}->{tt.move_gate_margin:.3f}, om_pf_margin "
                  f"{_old[2]}->{tt.om_pf_margin:.3f}")
    if use_motion:
        if events_json:
            ev = json.load(open(events_json))
            tt.rot_events = sorted(
                require_exact_integer(e["frame"], "event-file frame")
                for e in ev)
            print(f"  movement: {len(tt.rot_events)} events from {events_json}")
        elif gt_events:
            mg = json.load(open(f"walkthrough/movegt_{tag}.json"))
            tt.rot_events = sorted(
                require_exact_integer(m["frame"], "GT event frame")
                for m in mg)
            print(f"  movement: {len(tt.rot_events)} GT (cube) move events fed "
                  f"to trellis [ISOLATION]")

    # Automatic/fail-closed phase producer: strict eval engages only when the
    # decoder consumes an authoritative EV file.  GT events and re-derived ramp
    # events are isolation/legacy sources and therefore do not inherit this
    # provenance.  No solve tag, threshold, or runtime knob participates.
    if use_motion and events_json:
        try:
            _phase_audit = certify_intraburst_phase_slots(
                _intraburst_forward_motion, tt.rot_events, fps)
        except Exception as _phase_exc:  # producer seam must preserve baseline
            _phase_audit = certify_intraburst_phase_slots(
                {}, tt.rot_events, fps)
            print(f"  [intraburst-phase] producer error "
                  f"{type(_phase_exc).__name__}: {_phase_exc}; abstained")
        tt.intraburst_phase_audit = _phase_audit
        tt.intraburst_phase_slots = tuple(_phase_audit.slots)
        print(f"  [intraburst-phase] status={_phase_audit.status} "
              f"slots={len(_phase_audit.slots)} certified_periods="
              f"{_phase_audit.certified_period_count} ambiguous="
              f"{len(_phase_audit.ambiguous_periods)}")

    # WINDOWED-SCRUB decode (--scrub-decode; see
    # TrellisTracker.scrub_decode / detect.scrub_decode). Sequential small
    # windows start from the app-given INIT state, with each window's word
    # re-derived by a bounded forward search scored with
    # om-MARGINALIZED read-fit; the committed word is demoted to ONE candidate
    # hypothesis per window (no constraint/anchor/om from committed
    # states). When ON the emitted decode is REPLACED by the scrub word; any
    # exception inside the module emits the untouched stock decode.
    if "--scrub-decode" in sys.argv:
        tt.scrub_decode = True
        tt.scrub_decode_tag = tag
        # __import__ keeps this self-contained for the CLI-block contract test,
        # which executes this block in isolation with only sys/tt/tag globals.
        tt.scrub_decode_log = __import__("os").environ.get(
            "CUBED_SCRUB_LOG_PATH", f"/tmp/scrub_{tag}.scrub.jsonl")
        print(f"  [scrub-decode] ON (REPLACES the emitted decode; committed "
              f"word demoted to one hypothesis/window; "
              f"sidecar={tt.scrub_decode_log})", flush=True)

    # STATEFUL SCRUB OM (--scrub-om-stateful). Explicit experimental extra,
    # deliberately NOT part of the canonical scrub config: run_decode.sh folds
    # passthrough flags into the +extras checksum. The tracker forwards the
    # final move-gate verdict stream so accepted moves and dropped rotation
    # proposals can have distinct chronological transitions. With scrub off the
    # attr is harmless because the post-pass is never entered.
    if "--scrub-om-stateful" in sys.argv:
        missing = [flag for flag in ("--scrub-decode", "--move-gate",
                                     "--align-gate")
                   if flag not in sys.argv]
        if missing:
            raise SystemExit("--scrub-om-stateful requires "
                             + ", ".join(missing))
        tt.scrub_om_stateful = True
        print("  [scrub-om-stateful] ON", flush=True)

    # SCRUB-ONLY DENSE READS (--scrub-dense-reads). The main trellis, move-gate
    # passes, and scrub's candidacy/prior/control plane retain their finalized
    # quality-ranked sample. Only stateful candidate-conditioned likelihood
    # receives exact-frame evidence from each final alignment-stable epoch.
    # This is not the falsified global stratified-sampling policy.
    if "--scrub-dense-reads" in sys.argv:
        missing = [flag for flag in ("--scrub-decode", "--scrub-om-stateful")
                   if flag not in sys.argv]
        if missing:
            raise SystemExit("--scrub-dense-reads requires "
                             + ", ".join(missing))
        tt.scrub_dense_reads = True
        print("  [scrub-dense-reads] ON (state-owned exact-frame scoring "
              "evidence; control/candidacy unchanged)", flush=True)

    # DENSE PREFIX CANDIDATE PLANE. Every eligible window runs a second
    # stateful generator over the same slots/counts/OM graph; its independent
    # band/K survivors are unioned with the control survivors before common
    # final scoring and the window decision.
    if "--scrub-dense-prefix" in sys.argv:
        missing = [flag for flag in (
            "--scrub-decode", "--scrub-om-stateful",
            "--scrub-dense-reads") if flag not in sys.argv]
        if missing:
            raise SystemExit("--scrub-dense-prefix requires "
                             + ", ".join(missing))
        tt.scrub_dense_prefix = True
        print("  [scrub-dense-prefix] ON (all-window chronological dense "
              "candidate union; control survivors remain a subset)",
              flush=True)

    # OPTIONAL VISUAL TRANSITION SLOTS. Sustained align-gate episodes add one
    # interval-censored MOVE-or-SKIP opportunity to the stateful scrub lattice.
    # They never widen or replace the authoritative motion-count budget.
    if "--scrub-visual-transition-slots" in sys.argv:
        missing = [flag for flag in (
            "--scrub-decode", "--scrub-om-stateful", "--move-gate",
            "--align-gate", "--align-transition-sample")
                   if flag not in sys.argv]
        if missing:
            raise SystemExit("--scrub-visual-transition-slots requires "
                             + ", ".join(missing))
        tt.scrub_visual_transition_slots = True
        print("  [scrub-visual-transition-slots] ON (optional MOVE/SKIP; "
              "hard motion budget unchanged)", flush=True)

    # FINAL-GATE-DROP SLOTS. An authoritative final-dropped gap contributes one
    # optional SINGLE-or-SKIP rescue transaction only when direct color/state-
    # change evidence says MOVE and alignment vetoes it. The primary lattice is
    # untouched. This source hierarchy adds no learned/tuned threshold and
    # never changes the hard motion-count interval.
    if "--scrub-gate-drop-slots" in sys.argv:
        missing = [flag for flag in (
            "--scrub-decode", "--scrub-om-stateful", "--move-gate")
                   if flag not in sys.argv]
        if missing:
            raise SystemExit("--scrub-gate-drop-slots requires "
                             + ", ".join(missing))
        tt.scrub_gate_drop_slots = True
        print("  [scrub-gate-drop-slots] ON (unresolved rescue; "
              "alignment-veto-only drops; "
              "SKIP + 18 SINGLE; hard motion budget unchanged)", flush=True)

    # SCRUB CONTAINMENT AUDIT (--scrub-containment-audit [movegt], day-1 gate 2;
    # detect.scrub_decode). SHADOW/EVAL-side: at every prune point of the
    # burst-aligned prefix beam (exact-ball budget, band prune, K cap) it records
    # whether the GT window trajectory survives, to a `.containment.jsonl` next to
    # the scrub sidecar. Applies NOTHING to candidates/decisions/emissions =>
    # byte-identical to off. The movegt is the TEACHER (never production input);
    # plumbed via env var so the untouched tracker call site forwards it.
    if "--scrub-containment-audit" in sys.argv:
        _ci = sys.argv.index("--scrub-containment-audit")
        _cmg = (sys.argv[_ci + 1]
                if _ci + 1 < len(sys.argv)
                and not sys.argv[_ci + 1].startswith("--")
                else f"walkthrough/movegt_{tag}.json")
        os.environ["CUBED_SCRUB_CONTAINMENT_AUDIT"] = _cmg
        print(f"  [scrub-containment-audit] SHADOW ON "
              f"(movegt={_cmg if os.path.exists(_cmg) else 'ABSENT'}; "
              f"GT-window-trajectory survival at every prune; applies nothing; "
              f"rows -> {tt.scrub_decode_log}.containment.jsonl if scrub on)",
              flush=True)

    # TRUTH PROBE (--truth-probe [movegt]; capability "truth-probe-v1";
    # detect.scrub_decode). RECORD-ONLY dev-ceiling instrument: watches the GT
    # (state,om) lineage INSIDE device-beam scrub windows (rank/margin at every
    # band prune + K cap, band-vs-k_cap killer attribution, om diversity)
    # WITHOUT disabling the device beam -- the containment audit and device-
    # beam consume are mutually exclusive; this instrument is not. Applies
    # NOTHING to candidates/scores/decisions/emissions => byte-identical to
    # off. Rows -> `<scrub sidecar>.truth_probe.jsonl`. Plumbed via env var
    # (untouched tracker call site). The movegt is a DEV-CEILING teacher,
    # never a production input.
    if "--truth-probe" in sys.argv:
        _tpi = sys.argv.index("--truth-probe")
        _tmg = (sys.argv[_tpi + 1]
                if _tpi + 1 < len(sys.argv)
                and not sys.argv[_tpi + 1].startswith("--")
                else f"walkthrough/movegt_{tag}.json")
        os.environ["CUBED_TRUTH_PROBE"] = _tmg
        print(f"  [truth-probe] RECORD-ONLY ON "
              f"(movegt={_tmg if os.path.exists(_tmg) else 'ABSENT'}; "
              f"truth rank/margin/killer inside the device beam; applies "
              f"nothing; rows -> {tt.scrub_decode_log}.truth_probe.jsonl "
              f"if scrub on)", flush=True)

    # SCRUB PERFECT-PERCEPTION ORACLE (--scrub-oracle-likelihood [movegt], day-1
    # gate 3; detect.scrub_decode). EVAL ORACLE (NOT production): adds a per-
    # candidate term (+0 on the GT trajectory, -one 2sigma band otherwise) at the
    # pre-prune scoring seam, so perfect perception worth exactly one decision
    # band recovers the GT word. This DELIBERATELY CHANGES the decode -- it is the
    # upper-bound instrument, never a shipped behaviour. Absent movegt => off =>
    # byte-identical. Plumbed via env var (untouched tracker call site).
    if "--scrub-oracle-likelihood" in sys.argv:
        _oi = sys.argv.index("--scrub-oracle-likelihood")
        _omg = (sys.argv[_oi + 1]
                if _oi + 1 < len(sys.argv)
                and not sys.argv[_oi + 1].startswith("--")
                else f"walkthrough/movegt_{tag}.json")
        os.environ["CUBED_SCRUB_ORACLE_LL"] = _omg
        print(f"  [scrub-oracle-likelihood] EVAL ORACLE ON "
              f"(movegt={_omg if os.path.exists(_omg) else 'ABSENT'}; "
              f"CHANGES the decode by design; perfect-perception upper bound; "
              f"NEVER production)", flush=True)

    # MID-MOTION READS (--midmotion-reads; see detect.midmotion_reads +
    # scripts/forensic_m1_midmotion_reads.py). Augments the read set fed to the scrub
    # scorer ONLY with per-cell static-layer reads extracted from the mid-turn frames geo_read
    # vetoes (H4: during a face turn 2/3 of the cube stays grid-aligned and readable). Rows are
    # PRECOMPUTED by the forensic harness (extraction needs the video + pose client, not carried
    # by the decode's reads pickle) and loaded here from --midmotion-rows (a pickled list of
    # midmotion_reads row dicts). This sets tracker attrs ONLY; the legacy beam's `reads_in` is
    # never touched, so default-off is byte-identical. CONSUMPTION SEAM (M2 fusion,
    # `detect.scrub_decode`): the scrub scorer merges `self.midmotion_rows` into its
    # per-window read pool when `getattr(tt, 'midmotion_reads', False)` — one line, out of M1's
    # binding scope (M1 = extraction + the GT-agreement measurement).
    if "--midmotion-reads" in sys.argv:
        tt.midmotion_reads = True
        tt.midmotion_reads_tag = tag
        tt.midmotion_reads_log = f"/tmp/midmotion_{tag}.rows.jsonl"
        _mmp = argval("--midmotion-rows", f"/tmp/midmotion_{tag}.rows.pkl")
        tt.midmotion_rows = pickle.load(open(_mmp, "rb")) if os.path.exists(_mmp) else []
        print(f"  [midmotion-reads] ON (augments the scrub read set ONLY; legacy beam "
              f"untouched; rows_src={_mmp if os.path.exists(_mmp) else 'ABSENT'}; "
              f"n_rows={len(tt.midmotion_rows)}; sidecar={tt.midmotion_reads_log})", flush=True)

    # MOTION-DIRECTION AMOUNT PRIOR (read-independent): penalize candidate moves whose
    # amount (cw/ccw/180) contradicts the motion. --dir-pred <json> loads a real model's
    # per-event {frame:[logP_cw,logP_ccw,logP_180]}; --dir-oracle builds the CEILING
    # version (each motion event inherits the nearest GT move's amount) to validate the
    # lever before training/deploying the model. --dir-prior W scales it (default 1.0).
    if "--dir-oracle" in sys.argv or "--dir-pred" in sys.argv:
        import math as _math, re as _re2
        tt.dir_prior = argval("--dir-prior", 1.0, float)
        if "--dir-pred" in sys.argv:
            _dp = json.load(open(argval("--dir-pred")))
            tt.dir_pred = {int(f): [float(x) for x in v] for f, v in _dp.items()}
            print(f"  [dir-prior] loaded {len(tt.dir_pred)} per-event amount predictions, w={tt.dir_prior}")
        else:
            _am = {"": 0, "'": 1, "2": 2}
            _gm = [(round(mm.get("t_ms", 0) * fps / 1000.0), mm["move"]) for mm in sess["moves"]]
            _pred = {}
            for _ev in (tt.rot_events or []):
                _gf, _gmv = min(_gm, key=lambda t: abs(t[0] - _ev))
                if abs(_gf - _ev) > 8:            # re-grip / spurious event: no real move -> neutral
                    continue
                _mt = _re2.match(r"^[UDLRFB]([2']?)$", _gmv)
                _a = _am.get(_mt.group(1), 0) if _mt else 0
                _lp = [_math.log(0.05)] * 3
                _lp[_a] = _math.log(0.9)
                _pred[_ev] = _lp
            tt.dir_pred = _pred
            print(f"  [dir-oracle] amount prior on {len(_pred)} real-move events (within 8f of a GT move), w={tt.dir_prior}")

    # --om-oracle: build the TRUE per-frame camera-frame om timeline from GT (the
    # state replay + the per-move event frames) and feed it as tt.om_timeline.
    # This isolates the effect of a perfect orientation timeline. For each
    # settled frame, the true orientation is the one under which the GT state
    # best fits that frame's read. --om-timeline <json> loads an external CV
    # timeline.
    if "--om-oracle" in sys.argv:
        import bisect
        c = Cube()
        for m in scr.split():
            apply_move(c, m)
        states = [state_to_array(c.state).astype(np.int8)]
        for mvd in sess["moves"]:
            apply_move(c, mvd["move"])
            states.append(state_to_array(c.state).astype(np.int8))
        mvf = sorted(tt.rot_events) if tt.rot_events else \
            sorted(
                require_exact_integer(e["frame"], "oracle event-file frame")
                for e in json.load(open(events_json)))
        tl = {}
        for f, (r, m) in raw.items():
            if m is None or m > 8:
                continue
            k = bisect.bisect_right(mvf, f)
            ts = states[min(k, len(states) - 1)]
            om, _ = establish_orientation_from_reads([r], ts, pal)
            if om:
                tl[f] = _om_key(om)
        json.dump({str(f): list(v) for f, v in tl.items()},
                  open(f"/tmp/omoracle_{tag}.json", "w"))
        tt.om_timeline = tl
        print(f"  [om-oracle] true om timeline over {len(tl)} settled frames "
              f"-> /tmp/omoracle_{tag}.json")
    omtl = argval("--om-timeline", None)
    if omtl:
        raw_tl = json.load(open(omtl))   # {frame: [up, front, right]} or om_key str
        tt.om_timeline = {int(f): (tuple(v) if isinstance(v, list) else v)
                          for f, v in raw_tl.items()}
        print(f"  [om-timeline] loaded {len(tt.om_timeline)} frames from {omtl}")

    # --om-per-frame: score each settled frame under ITS OWN timeline om instead
    # of pooling the whole span under one mode om (the re-grip fix; see
    # TrellisTracker.om_per_frame). Pairs with --om-oracle (per-frame best-fit-to-
    # truth timeline) or --om-timeline (external CV per-frame source).
    if "--om-per-frame" in sys.argv:
        tt.om_per_frame = True
        if tt.om_timeline:
            print(f"  [om-per-frame] ON: fusing {len(tt.om_timeline)} per-frame "
                  f"oms in the span scorer")
        else:
            print("  [om-per-frame] WARNING: no om_timeline set -> per-frame "
                  "scoring is a no-op (need --om-oracle or --om-timeline)")

    # --om-pf-reads: SOFT-FUSION per-frame om resolved from the READS.
    # Confidence-gated with fallback to
    # the robust per-span om-rescue -> never forces a wrong om. The pure-CV
    # end-goal path; the oracle mode only validates it.
    if "--om-pf-reads" in sys.argv:
        tt.om_per_frame = True
        tt.om_per_frame_from_reads = True
        tt.om_pf_margin = argval("--om-pf-margin", 2.0, float)
        tt.om_pf_min_frames = argval("--om-pf-min-frames", 0, int)
        print(f"  [om-pf-reads] soft-fusion CV-only per-frame om from reads "
              f"(margin>={tt.om_pf_margin}, min_frames={tt.om_pf_min_frames})")

    # --om-pf-ball D: BALL-ANCHORED per-frame orientation resolution. Stock
    # _resolve_perframe_om scores the
    # 24 oms against the STALE previous anchor state (zero-move assumption);
    # junk-cell removal can amplify fit margins past om_pf_margin, pinning
    # post-gap frames to the om that best misinterprets the
    # old state. With this flag each om is scored by its MAX fit over the anchor's
    # depth-D move ball (the states the beam can actually reach this span); the
    # margin gate + span-om fallback are unchanged. Default off =>
    # byte-identical (om_pf_ball None).
    # Value: an integer depth D, "gap", or "dgap". "gap" — GAP-DERIVED depth:
    # per span, depth = min(ceil(exp_hi), d_gap), where exp_hi is the
    # decoder's own admissible move-count ceiling for that gap (duration
    # bound, burst/event-widened) and d_gap is the span's transition-ball
    # depth (the states the beam can actually reach — no new constant; see
    # the om_pf_ball comment in detect/trellis_tracker.py). "dgap" — the
    # DERIVED FIX (2026-07-03 forensic): depth = d_gap DIRECTLY, dropping
    # gap's min(ceil(exp_hi), .) shrink that pulled the pf-ball below the
    # beam's own move reach (knob-free, >= gap depth per span). LAST occurrence
    # wins so an experimental arm can override the canonical baked
    # "--om-pf-ball 3" by appending (run_decode.sh passes "$@" AFTER $CFG);
    # canonical invocations carry the flag once => unchanged.
    if "--om-pf-ball" in sys.argv:
        _bi = len(sys.argv) - 1 - sys.argv[::-1].index("--om-pf-ball")
        _bv = sys.argv[_bi + 1]
        tt.om_pf_ball = _bv if _bv in ("gap", "dgap") else int(_bv)
        tt._pf_ball_depth_hist = {}   # banner: per-depth span counts
        tt._pf_ball_scored = [0, 0]   # banner: lazy full-pass counts
        _bdesc = ({"gap": "span-budget-derived (gap) depth",
                   "dgap": "transition-ball (dgap = d_gap) depth"}
                  .get(tt.om_pf_ball, f"depth-{tt.om_pf_ball}"))
        print(f"  [om-pf-ball] pf-om resolution vs {_bdesc} anchor "
              f"ball (margin>={tt.om_pf_margin})")

    # --align-weights <npz>: SOFT alignment fusion — weight each frame's READ in the
    # span vote by its P(aligned) (from extract_alignfeat.py: {frame, aligned}). A
    # mid-turn frame is trusted less, not dropped — unlike a hard align gate, which
    # hurt the decode. The cooperative-signals fix: alignment informs the trellis.
    if "--align-weights" in sys.argv:
        _az = np.load(argval("--align-weights"), allow_pickle=True)
        tt.align_weights = {int(f): float(a) for f, a in zip(_az["frame"], _az["aligned"])}
        tt.align_weight_floor = argval("--align-weight-floor", 0.1, float)
        print(f"  [align-weights] soft per-frame alignment trust ON "
              f"({len(tt.align_weights)} frames, floor {tt.align_weight_floor})")

    if "--consensus-reads" in sys.argv:
        tt.consensus_reads = True
        tt.consensus_centroids = pal   # palette for majority-color voting
        print("  [consensus-reads] per-span robust MAJORITY-color read (uses ALL the "
              "span's aligned frames, not the top-N vote)")

    if "--dump-om-only" in sys.argv:
        print("  [dump-om-only] om timelines dumped; exiting before decode")
        return 0.0

    if backward:
        # remap all frame-keyed trellis inputs into reversed time
        if tt.rot_events:
            tt.rot_events = sorted(rev(f) for f in tt.rot_events)
        if tt.intraburst_phase_audit is not None:
            tt.intraburst_phase_audit = remap_intraburst_phase_audit(
                tt.intraburst_phase_audit, rev)
            tt.intraburst_phase_slots = tuple(
                tt.intraburst_phase_audit.slots)
            print(f"  [intraburst-phase] backward-remap status="
                  f"{tt.intraburst_phase_audit.status} slots="
                  f"{len(tt.intraburst_phase_slots)}")
        if tt.force_breaks:
            tt.force_breaks = set(rev(f) for f in tt.force_breaks)

    if "--om-trace" in sys.argv:
        tt.om_trace = []
    use_fwbw = '--fw-bw' in sys.argv
    if use_fwbw:
        print('  [fw-bw] bidirectional Viterbi smoother ON')
    # Prime only after every AbsSegment feature/gate knob above has been wired.
    # Existing cache entries (for optional pre-track diagnostics) are retained
    # and skipped; canonical runs reach this point before their first segment.
    if _segment_factory_stats is not None and _batched_trust:
        with PERF_TRACE.span("decode.trust_batch"):
            _trust_batch_report = _segment_factory_stats.prime_trust_batch(
                (r for r, _m in raw.values()), device="auto")
        print(
            "  [trust-batch] "
            f"backend={_trust_batch_report['backend']} "
            f"reads={_trust_batch_report.get('prepared_reads', 0)}/"
            f"{_trust_batch_report.get('reads', 0)} "
            f"rows={_trust_batch_report['rows']} "
            f"wall={_trust_batch_report['wall_s']:.3f}s",
            flush=True,
        )
    PERF_TRACE.finish_child("decode.pretrack_after_reads", setup_start_ns)
    with PERF_TRACE.span("decode.track_total"):
        if getattr(tt, "_use_move_gate_2pass", False):
            mv, reach, info = tt.track_2pass(
                reads_in, motions, init, final, MIN_STICKER_L,
                fw_bw=use_fwbw)
        else:
            mv, reach, info = tt.track(
                reads_in, motions, init, final, MIN_STICKER_L,
                fw_bw=use_fwbw)
    if _segment_factory_stats is not None:
        print(
            "  [segment-cache] "
            f"full={_segment_factory_stats.full_constructions} "
            f"orientation_clones={_segment_factory_stats.orientation_clones} "
            f"hits={_segment_factory_stats.cache_hits}",
            flush=True,
        )
    if "--dump-gate-events" in sys.argv and getattr(tt, "_gate_events", None):
        _mode = tt.align_gate_mode if tt.align_gate else "color"
        _ge_out = f"/tmp/gate_events_{tag}_{_mode}.json"
        _mg = sorted(int(m["frame"]) for m in
                     json.load(open(f"walkthrough/movegt_{tag}.json")))
        import bisect as _bs
        for _e in tt._gate_events:        # tag each event TP(real move) vs FP(re-grip)
            _i = _bs.bisect_left(_mg, _e["frame"])
            _e["real_move"] = any(0 <= _j < len(_mg) and abs(_mg[_j] - _e["frame"]) <= 6
                                  for _j in (_i - 1, _i))
        json.dump({"tag": tag, "mode": _mode, "events": tt._gate_events, "gt_moves": _mg},
                  open(_ge_out, "w"))
        print(f"  [dump-gate-events] {len(tt._gate_events)} events -> {_ge_out}")
    if getattr(tt, "om_trace", None) is not None:
        pickle.dump(tt.om_trace, open(f"/tmp/omtrace_{tag}.pkl", "wb"))
        print(f"  om_trace: {len(tt.om_trace)} spans -> /tmp/omtrace_{tag}.pkl")
    if backward:
        # trellis decoded solved->scramble over reversed frames: real solve =
        # output reversed + each move inverted (m_k^-1 in reverse order -> m_k).
        _inv = {"": "'", "'": "", "2": "2"}
        mv = [m[0] + _inv[m[1:]] for m in reversed(mv)]
        print(f"  [backward] reconstructed forward solve: {len(mv)} moves")
    local, used_rot = reconstruct_with_rotations_safe(
        info.get("moves_raw", mv), info.get("oms_raw", []))
    # canonical comparison: merge same-face turns + cancel (U U == U2, R R' == ø)
    # so equivalent representations don't count as errors
    smv = simplify_moves(mv)
    sgt = simplify_moves(gt) if teacher_available else []
    PERF_TRACE.mark("moves_ready")
    PERF_TRACE.set_meta(
        reaches_final=bool(reach), detected_moves=len(mv),
        canonical_moves=len(smv), teacher_available=teacher_available)
    diagnostic_similarity = None
    if teacher_available:
        diagnostic_similarity = edit_similarity(mv, gt)
        canonical_similarity = edit_similarity(smv, sgt)
        ed = editdist(smv, sgt)
        PERF_TRACE.set_meta(
            teacher_moves=len(sgt),
            edit_similarity_canonical=float(canonical_similarity),
            edit_distance=int(ed))
        print(
            f"  -> sequence_replayed_to_target={reach} emitted_moves={len(mv)} "
            f"teacher_moves={len(gt)} "
            f"diagnostic_edit_similarity_raw={diagnostic_similarity:+.3f} "
            f"diagnostic_edit_similarity_canonical={canonical_similarity:+.3f} "
            f"(simplified {len(smv)} vs {len(sgt)})"
        )
        print(
            f"     [diagnostic] canonical_edit_distance={ed}/{len(sgt)} "
            f"canonical_edit_similarity={1 - ed / max(1, len(sgt)):.6f}"
        )
    else:
        print(
            f"  -> sequence_replayed_to_target={reach} emitted_moves={len(mv)} "
            "teacher_reference=not-provided diagnostic_edit_similarity=unavailable"
        )
    # NAMED EVALUATION: replay the raw emitted sequence (before simplification,
    # which alters intermediate states) from the recorded scramble and test
    # whether it passes through last-layer onset with the correct preceding
    # state, modulo whole-cube orientation. Edit similarity is diagnostic only.
    # The evaluation is fail-soft when no teacher move reference was provided.
    if teacher_available:
        try:
            import reach_ll
            _rli = reach_ll.locate_GL(tag, session_dir=argval("--session-dir", None))
            _rls = reach_ll.score_sequence(tag, mv, info=_rli)
            PERF_TRACE.set_meta(
                reach_ll=bool(_rls["reach_GL"]),
                reach_ll_hit_step=_rls["gl_hit_step"],
                reach_ll_phase=_rli["gl_phase"], reach_ll_index=_rli["gl_index"])
            print(
                "     evaluation=reach-ll-onset-with-correct-pre-ll-state "
                f"reached={_rls['reach_GL']} "
                f"hit_step={'-' if _rls['gl_hit_step'] is None else _rls['gl_hit_step']} "
                f"deepest_phase={_rls['deepest_phase_reached'] or '-'} "
                f"first_divergence_phase={_rls['first_divergence_phase'] or '-'} "
                f"target={_rli['gl_phase']}@{_rli['gl_index']}"
            )
        except Exception as _rle:
            print(
                "     evaluation=reach-ll-onset-with-correct-pre-ll-state "
                f"status=unavailable ({type(_rle).__name__}: {_rle})"
            )
    else:
        print(
            "     evaluation=reach-ll-onset-with-correct-pre-ll-state "
            "status=unavailable reason=teacher-reference-not-provided"
        )
    # TEACHER-FORCED verdict: pre-LL split reuses reach_ll's
    # LL-onset GT move index (the same boundary the reach-LL judge uses);
    # fail-soft like the evaluation line — locate_GL failing must not crash the
    # harness (preLL then falls back to the full counts, loudly).
    if tt.teacher_forced is not None and tt.tf_records is not None:
        _tf_gl = None
        try:
            import reach_ll as _tf_rl
            _tf_gl = int(_tf_rl.locate_GL(
                tag, session_dir=argval("--session-dir", None))["gl_index"])
        except Exception as _tfe:
            print(f"  [teacher-forced] pre-LL boundary unavailable "
                  f"({type(_tfe).__name__}: {_tfe}) — the pre-LL agreement falls "
                  f"back to the full counts")
        for _ln in teacher_forced_report(tt.tf_records, tt.teacher_forced,
                                         _tf_gl):
            print(f"     {_ln}")
        # full per-gap records (incl. the om/state probe fields) for the
        # microscope — the om_trace diagnostic-dump idiom.
        json.dump(tt.tf_records, open(f"/tmp/tfrecords_{tag}.json", "w"))
        print(f"  [teacher-forced] {len(tt.tf_records)} per-gap records -> "
              f"/tmp/tfrecords_{tag}.json")
    # SPAN-COLLAPSE per-tag summary (--verbose
    # only; MEASUREMENT ONLY, same fail-soft idiom as the teacher-forced
    # block above — a summary line must never crash the harness).
    if tt.span_collapse_log is not None:
        for _ln in span_collapse_report(tt.span_collapse_log):
            print(f"     {_ln}")
    # COLLAPSE-STRAT trigger count (lever (2); ALWAYS
    # printed when the mode is armed -- triggers are rare by construction
    # (the (p) sweep found 0-2 per tag), so this is not gated behind
    # --verbose the way the measurement-only span-collapse summary above is.
    if tt.strat_span_sample_cond:
        print(f"     collapseStratSummary triggers={tt._collapse_strat_triggers} "
              f"thr={tt._collapse_strat_threshold()} fps={tt.fps}")
    if tt.align_transition_sample:
        print(f"     alignTransitionSampleSummary "
              f"fires={tt._align_transition_fires}")
    # WINDOWED-SCRUB summary (same idiom).
    if tt.scrub_decode:
        _sr = tt._scrub_report or {}
        print(f"     scrubSummary windows={_sr.get('windows', 0)} "
              f"committed={_sr.get('committed', 0)} "
              f"extended={_sr.get('extended', 0)} "
              f"low_conf={_sr.get('low_conf', 0)} "
              f"unresolved={_sr.get('unresolved', 0)} "
              f"om_stateful={_sr.get('om_stateful')} "
              f"om_reason={_sr.get('om_stateful_reason')} "
              f"om_counts={_sr.get('om_stateful_counts')} "
              f"final_endpoint_ok={_sr.get('final_endpoint_ok')} "
              f"status={_sr.get('status')} "
              f"sidecar={tt.scrub_decode_log}")
    if "--full-seq" in sys.argv:
        print(f"     [full canonical emitted] {' '.join(smv)}")
        if teacher_available:
            print(f"     [full canonical teacher] {' '.join(sgt)}")
    print(f"     emitted: {' '.join(mv[:40])}{' ...' if len(mv) > 40 else ''}")
    print(f"     canonical emitted: {' '.join(smv[:40])}{' ...' if len(smv) > 40 else ''}")
    if teacher_available:
        print(f"     canonical teacher: {' '.join(sgt[:40])}{' ...' if len(sgt) > 40 else ''}")
    if AbsSegment.CONF_BLANK_STATS is not None:
        _st = AbsSegment.CONF_BLANK_STATS
        _seen = _st["blanked"] + _st["kept"]
        _frac = _st["blanked"] / _seen if _seen else 0.0
        _hist = " ".join(f"[{i*0.05:.2f}-{i*0.05+0.05:.2f})={c}"
                          for i, c in enumerate(_st["hist"]) if c)
        print(f"  [conf-blank] thr={AbsSegment.CONF_BLANK} "
              f"blanked={_st['blanked']} kept={_st['kept']} exempt={_st['exempt']} "
              f"blanked_frac={_frac:.3f}")
        print(f"  [conf-blank] hist(0.05-bins): {_hist if _hist else '(no observed conf values)'}")
    if AbsSegment.TRUST_SOFT_STATS is not None:
        _ss = AbsSegment.TRUST_SOFT_STATS
        _obs_cells = _ss["cells"] - _ss["exempt"]
        print(f"  [trust-soft] floor={AbsSegment.TRUST_SOFT['floor']} "
              f"reads_scored={_ss['reads_scored']} cells={_ss['cells']} "
              f"(observed-conf {_obs_cells}, exempt {_ss['exempt']}) — every "
              f"cell weighted (max(p_trust,floor)), none dropped")
        _sh = " ".join(f"{i * 0.05:.2f}:{c}" for i, c in
                       enumerate(_ss["p_hist"]) if c)
        print(f"  [trust-soft] p_trust hist(0.05-bins): "
              f"{_sh if _sh else '(no observed-conf cells scored)'}")
    if getattr(tt, "_pf_ball_depth_hist", None):
        _bh = tt._pf_ball_depth_hist
        print(f"  [om-pf-ball] span ball-depth usage: "
              + " ".join(f"d{k}={_bh[k]}" for k in sorted(_bh))
              + f" (spans_resolved={sum(_bh.values())}, mode="
              + f"{tt.om_pf_ball})")
        _bs = getattr(tt, "_pf_ball_scored", None)
        if _bs and _bs[0]:
            print(f"  [om-pf-ball] lazy eval: {_bs[1]}/{_bs[0] * 24} full "
                  f"ball passes ({_bs[1] / _bs[0]:.2f} of 24 oms/frame; "
                  f"rest eliminated by the sound per-face pattern bound)")
    # STRUCTURED RESULT (opt-in): when CUBED_RESULT_JSON names an absolute
    # path, a run that terminated normally also writes a decode-result-v1
    # document there (scripts/decode_result_emit.py owns the schema). The
    # helper is imported INSIDE this branch and nothing is read or written when
    # the variable is unset, so the default run stays byte-identical. Fail-soft
    # like every other summary block above: emission observes the decode and
    # must never change its outcome.
    if os.environ.get("CUBED_RESULT_JSON"):
        try:
            import decode_result_emit
            _workstation = None
            try:
                _workstation = decode_result_emit.load_workstation_from_env()
                if _workstation is not None:
                    _timeline_info = dict(info)
                    _raw_checkpoint_moves = _timeline_info.get("moves_bt")
                    _timeline_info["reconstruction_checkpoint_valid"] = (
                        _reconstruction_checkpoints_valid(
                            reach=reach,
                            backward=backward,
                            raw_moves=_raw_checkpoint_moves,
                            canonical_moves=smv,
                        )
                    )
                    _workstation = (
                        decode_result_emit.workstation_with_decode_timeline(
                            _workstation,
                            info=_timeline_info,
                            moves=smv if bool(reach) else [],
                            events=getattr(tt, "rot_events", ()) or (),
                        )
                    )
            except Exception as _wse:
                # Viewer diagnostics are additive. A malformed or unavailable
                # projection must not alter the reconstruction verdict.
                print(
                    "  [workstation] unavailable "
                    f"({type(_wse).__name__}: {_wse})"
                )
                _workstation = None
            _result_out = decode_result_emit.emit_from_decode(
                tag=tag, moves=smv, solved_reached=bool(reach),
                inputs=(("reads", reads_path), ("events", events_json),
                        ("centroids", cal)),
                implementation_path=os.path.abspath(__file__),
                numpy_version=np.__version__,
                workstation=_workstation)
            print(f"  [result-json] wrote {_result_out}")
        except Exception as _rje:
            print(f"  [result-json] not written "
                  f"({type(_rje).__name__}: {_rje})")
    PERF_TRACE.mark("process_complete")
    PERF_TRACE.write()
    return diagnostic_similarity


if __name__ == "__main__":
    main()
