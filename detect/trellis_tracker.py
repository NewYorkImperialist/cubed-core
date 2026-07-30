"""Span-level trellis and state-search implementation used by Decode.

Camera evidence is prepared upstream. This module groups reads into spans,
scores legal cube-state transitions, applies the configured endpoint constraint,
and canonicalizes the emitted move sequence.
"""
from __future__ import annotations

import os as _os
from itertools import product

import numpy as np

from detect.move_detector import (
    MOVE_LIST, ALL_FACES, GRID_ROTATIONS, COLOR_INDEX, COLOR_LIST,
)
from detect.calibrator import MIN_STICKER_L, CENTER_COLOR_TO_FACE
from core.perf_trace import get_trace

PERF_TRACE = get_trace("decode")

# Runner-visible marker for optional scrub features implemented end to end.
SCRUB_EXPERIMENT_API = frozenset({
    "dense-reads-v1", "dense-prefix-v1", "gate-drop-slots-v1",
    "intraburst-phase-slots-v1",
})

# Express the historical five-sample median and five-frame rest split as
# durations so a different capture rate does not silently change the detector.
_MOTION_SMOOTH_RADIUS_S = 2.0 / 120.0
_MOTION_MIN_REST_S = 5.0 / 120.0


def _motion_event_temporal_frames(fps):
    """Resolve the historical 120-fps motion envelope in capture frames."""
    try:
        rate = float(fps)
    except (TypeError, ValueError):
        rate = 30.0
    if not np.isfinite(rate) or rate <= 0:
        rate = 30.0
    radius = max(0, round(rate * _MOTION_SMOOTH_RADIUS_S))
    min_rest = max(1, round(rate * _MOTION_MIN_REST_S))
    return int(radius), int(min_rest)

FACE_OFFSET = {f: i * 9 for i, f in enumerate(ALL_FACES)}
_LAB_W = np.array([0.15, 1.0, 1.0])  # chrominance-weighted LAB distance


def motion_events_from_motions(motions, fps):
    """Rest-anchored motion-event segmentation from cube-ROI frame motion.

    This is the NO-GT rotation/move event source ported from
    scripts/gen_motion_events.py: median-smooth the per-frame scalar, threshold
    at its 50th percentile, group active runs separated by less than the
    minimum-rest duration, and emit the peak-motion frame of each period.  The
    original five-sample smoother/five-frame rest at 120 fps are now expressed
    as their equivalent capture-time durations.

    motions: {frame_idx: motion_scalar_or_None} (None == no motion measured that
    frame -> treated as 0 / rest). Production supplies a consecutive stream
    from the last authoritative cube ROI, rather than alignment-gated samples.
    Returns sorted int frame indices, the TrellisTracker.rot_events shape.
    """
    frames = sorted(motions)
    if not frames:
        return []
    fa = np.array(frames)
    motion = np.array([(motions[f] if motions[f] is not None else 0.0)
                       for f in frames], float)
    smooth_radius, min_rest = _motion_event_temporal_frames(fps)
    sm = np.array([np.median(motion[
        max(0, i - smooth_radius):i + smooth_radius + 1])
                   for i in range(len(motion))])
    THR = float(np.percentile(sm, 50))
    active = sm > THR

    runs = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    periods = runs[:1]
    for s, e in runs[1:]:
        if fa[s] - fa[periods[-1][1]] < min_rest:
            periods[-1][1] = e
        else:
            periods.append([s, e])
    events = [int(fa[s + int(np.argmax(sm[s:e + 1]))]) for s, e in periods]
    return sorted(events)


def _gen_orientations():
    """The 24 cube orientations as orient_maps {spatial_slot -> model_face}: reads
    sort faces into spatial up/front/right by image position; the orientation says
    which MODEL face sits at each slot (e.g. white-up-RED-front holds)."""
    from core.cube import Cube
    from detect.scramble import apply_move
    seen, out = set(), []
    for r1 in ["", "x", "x2", "x'", "z", "z'"]:
        for r2 in ["", "y", "y2", "y'"]:
            c = Cube()
            for m in (r1 + " " + r2).split():
                apply_move(c, m)
            om = {"up": CENTER_COLOR_TO_FACE[c.state["up"][4]],
                  "front": CENTER_COLOR_TO_FACE[c.state["front"][4]],
                  "right": CENTER_COLOR_TO_FACE[c.state["right"][4]]}
            key = (om["up"], om["front"], om["right"])
            if key not in seen:
                seen.add(key)
                out.append(om)
    return out


ORIENTATIONS = _gen_orientations()


class AbsSegment:
    """Calibrated absolute-color read scorer with FIXED face assignment via an
    orient_map (spatial slot -> model face), best over the 4 grid rotations per face.
    The rank-14 oracle method; far sharper than assignment-free scoring."""

    # PER-FACE JUNK GATE (None = off, bit-identical). A near-edge-on face (the
    # camera sees it at a grazing angle) still yields
    # "detected" stickers but their colors are far from EVERY calibrated
    # centroid; those 6-9 junk stickers poison the fit of every state
    # hypothesis uniformly while sticker counts stay healthy. FACE_GATE is a
    # RELATIVE multiplier: a face
    # is dropped when its mean min-centroid distance exceeds FACE_GATE x the
    # best sibling face's mean in the SAME read (and the absolute floor
    # FACE_GATE_MIN, so clean-read ratio noise can't fire). An ABSOLUTE bar
    # was tried first and is unusable because clip-to-clip read quality varies;
    # the normalization must be per-read.
    FACE_GATE = None
    FACE_GATE_MIN = 18.0
    # STICKER-LEVEL JUNK GATE (None = off; v3 after both face-level forms
    # failed). The right discriminator is per sticker: a junk sticker
    # (edge-on/blurred) is far from EVERY calibrated color (min-centroid
    # distance high), while an honest mismatch under a wrong hypothesis is
    # CLOSE to some color — just not the hypothesis's — so it keeps its full
    # decoy-rejecting penalty. Stickers with min-dist > STICKER_GATE are
    # excluded at construction (same path as the MIN_STICKER_L filter);
    # face-level dropping amputated real signal with the junk.
    STICKER_GATE = None
    # GEOMETRIC VISIBILITY GATE (None = off). face_vis (from the extraction
    # sidecar: per-slot quad area / largest quad in the frame) directly
    # measures foreshortening — the CV layer always knew which face was
    # edge-on; the cache just didn't keep it. Faces with vis < VIS_GATE are
    # excluded from scoring. Color-space proxies cannot reliably do this job
    # because healthy and junk sticker distances overlap.
    VIS_GATE = None
    # PROBABILISTIC EMISSION MODEL (None = legacy chrominance-weighted
    # distance). A dict {mu: {color: [3]}, var: {color: [3]}, pi_out, logp0}
    # fit from calibration-glance labeled samples (scripts/fit_emission.py).
    # dist[i, c] becomes the NEGATIVE log of a Gaussian+uniform mixture:
    # the fitted per-channel variances REPLACE the hand-set 0.15 luminance
    # weight, and junk stickers SATURATE at the outlier plateau instead of
    # poisoning every hypothesis (the derived form of the junk gates).
    EMISSION = None
    # PHASE-B AFFINE FIT-SCALE ADAPTER (None = off => byte-identical; only
    # consulted on the EMISSION path). (a, b) from derive_emission_adapter():
    # dist -> a + b*dist maps the nats-scale emission cost onto the
    # LAB-equivalent fit scale the rest of the trellis was tuned on, so every
    # fit-scale consumer (misfit_thr / move_gate_margin / om_pf_margin /
    # contrast near-ties / the evidence<->prior balance) keeps its tuned
    # currency. b>0 => order-preserving (state/om rankings identical to raw
    # nats; margins scale by exactly b).
    EMISSION_ADAPTER = None
    # PER-STICKER CONFIDENCE WEIGHTING (False = legacy equal weights).
    # With cell_conf from the extraction sidecar, each sticker's score
    # contribution is weighted by its seg confidence: low-conf foreshortened
    # detections contribute PARTIALLY instead of being hard-cut (group_conf)
    # or flooding at full weight.
    CONF_W = False
    # ROBUST DISTANCE CAP (None = off, bit-identical). A Huber-style CEILING on each
    # sticker's per-color LAB distance, applied before the mean-over-stickers fit.
    # Rationale (measured): a SYSTEMATIC read corruption (e.g. a few cells
    # mis-read white->blue, the SAME cells in every frame) gives the TRUE state a
    # LARGE distance at those cells and the WRONG look-alike state a SMALL one — so a
    # handful of corrupted cells over-penalize the truth and tip a thin-margin state
    # comparison (margins measured at ~0.2 on ~28). Capping the distance bounds any
    # single corrupted cell's sway so the honest majority decides. This is SOFT (a
    # cap, NOT the failed min-distance DROP gate, which amputated healthy-but-far
    # stickers). Lives on self.dist, so _dflat / CPU / GPU scoring inherit it
    # uniformly. Off => byte-identical.
    DIST_CAP = None
    # HARD CONFIDENCE-BLANK GATE (None = off, bit-identical). A cell whose
    # per-cell extraction confidence (cell_conf, from the occlusion-aware read
    # sidecar) is below CONF_BLANK is EXCLUDED at construction — the same hard
    # treatment as MIN_STICKER_L: the cell contributes to NO hypothesis's
    # score. This is HARD blanking, NOT the SOFT down-weighting of CONF_W
    # above (that form is intentionally left off here). Root cause:
    # (first-divergence forensic, Layer A): the pose layer always
    # emits 3 face slots regardless of visibility, so an occluded/hand-covered
    # slot's stickers are NOT dark (they pass MIN_STICKER_L) but sit far from
    # every calibrated color — poisoning every state hypothesis's fit
    # uniformly (measured -24..-46 vs misfit_thr -12, triggering constant
    # om-misfit rescue and diluting move-vs-stay margins ~3x). A cell with no
    # cell_conf entry (None) is EXEMPT (kept, default weight 1.0 downstream) —
    # the gate only fires on an actual low-confidence reading, never on its
    # absence. Semantics ported from the forensic's instrumented driver
    # (scratchpad repro/trellis_gt_forensic.py "--conf-filter", which blanked
    # cells by forcing L below MIN_STICKER_L on a copied file); reimplemented
    # here as a direct per-cell exclusion at the source, on the tracked path.
    CONF_BLANK = None
    # PER-RUN CONF_BLANK INSTRUMENTATION (None = off; prereq for
    # the learned/calibrated per-cell trust model). Only ever touched inside
    # the `self.CONF_BLANK is not None` branch below, so leaving this None
    # (the default) costs zero work and changes nothing when CONF_BLANK is
    # off. The driver (scripts/trellis_gt.py --conf-blank) initializes this
    # to a dict alongside setting CONF_BLANK, e.g.:
    #   {"blanked": 0, "kept": 0, "exempt": 0, "hist": [0] * 20}
    # "blanked"/"kept" count cells that HAD an observed cell_conf and were
    # excluded/survived the threshold; "exempt" counts cells with no conf
    # value (MIN_STICKER_L-surviving cells only — the same population the
    # CONF_BLANK gate itself ever inspects). "hist" is a 20-bin (width 0.05)
    # histogram of every OBSERVED cell_conf value, spanning both sides of
    # the threshold so future threshold/trust tuning can see the shape.
    CONF_BLANK_STATS = None
    # LEARNED PER-CELL READ-TRUST SOFT WEIGHTING (None = off, byte-identical).
    # Every admitted cell is kept and its fit contribution is weighted by
    # p_trust through the existing
    # normalized weighted-mean vector self._wv — honored uniformly by every
    # scorer (CPU/batched/GPU score_states AND the pf-om ball3 resolver's bound,
    # since they all consume _wv). This trades the hard-cutoff transfer failure
    # (a single global thr can't separate occluded-junk from
    # informative-but-low-scored across capture styles, and dropping a cell is
    # irreversible) for a continuous down-weight. Set by scripts/trellis_gt.py
    # --trust-soft to a dict:
    #   {"model": TrustNumpy (detect/trust_numpy.py — numpy-only, NO sklearn at
    #    decode), "cen_mat": (C,3) calibration
    #    centroids for the d1 feature, "cache": {} per-run p_trust memo keyed by
    #    id(read) (features are om-independent — the up-to-24-om reconstructions
    #    of one read reuse one predict_proba call),
    #    "floor": float weight floor for OBSERVED-conf cells, "n_features":
    #    15 (v1) | 17 (v3)}.
    # Per-cell weight = max(p_trust, floor) for an OBSERVED-conf cell, 1.0 for a
    # cell with no observed cell_conf (exempt from CONF_BLANK and kept at full
    # weight), then L1-normalized into _wv. Soft trust never hard-blanks, and when
    # CONF_BLANK/CONF_W would otherwise build _wv the soft vector takes
    # precedence. Off => _wv falls back to the CONF_W/None path => byte-
    # identical (the standard default-off discipline).
    TRUST_SOFT = None
    # Per-run TRUST_SOFT instrumentation (None = off): {"reads_scored",
    # "cells" (scored, incl. exempt), "exempt", "p_hist": [0]*20 (0.05-wide bins
    # over p_trust of OBSERVED cells)}. Accumulated on the cache MISS only, so a
    # read is counted once. Zero cost when None (the default).
    TRUST_SOFT_STATS = None

    def __init__(self, read, centroids, orient_map=None, face_vis=None,
                 cell_conf=None, frame_motion=None, frame_aligned=None,
                 frame_ev=None):
        # frame_motion / frame_aligned: per-frame capture-condition scalars
        # (motion magnitude, P(aligned)) consumed by the TRUST_SOFT feature
        # builder; None (every pre-existing caller) => trust falls
        # back to motion=0.0 / aligned=NaN (the trained missing encoding)
        # and nothing else reads them.
        # frame_ev: (ev_since, ev_to) event-context scalars for the v3
        # 17-feature trust contract (frames since/until the nearest DETECTED
        # motion/cnnev event — the decode's own tt.rot_events source, never
        # GT); None => the EV_CAP missing encoding. Ignored (not even read)
        # for a v1 15-feature model or when TRUST_SOFT is off.
        orient_map = orient_map or {"up": "up", "front": "front", "right": "right"}
        self.orient_map = orient_map
        labs, face_names, positions, confs = [], [], [], []
        _tobs = [] if self.TRUST_SOFT is not None else None
        self._softwv_raw = None
        # RAW (pre-floor) p_trust per admitted cell, aligned to _softwv_raw at
        # every stage incl. the FACE_GATE/STICKER_GATE trim below (
        # covT instrumentation — TrellisTracker._gate_trust_argmax
        # is the only reader). Set by _trust_soft_weights when TRUST_SOFT is
        # on; pure storage, never consulted by score_states/_wv => byte-
        # identical regardless of state.
        self._trust_p_raw = None
        for face, lab9 in read:
            c9 = (cell_conf or {}).get(face)
            if (self.VIS_GATE is not None and face_vis
                    and face_vis.get(face, 1.0) < self.VIS_GATE):
                continue
            for pos in range(9):
                l = lab9[pos]
                if l[0] < MIN_STICKER_L:
                    continue
                if _tobs is not None:
                    _tobs.append(bool(c9) and c9[pos] is not None)
                if self.CONF_BLANK is not None and self.TRUST_SOFT is None:
                    _cv = c9[pos] if c9 and c9[pos] is not None else None
                    if self.CONF_BLANK_STATS is not None:
                        _st = self.CONF_BLANK_STATS
                        if _cv is None:
                            _st["exempt"] += 1
                        else:
                            _st["hist"][max(0, min(19, int(_cv / 0.05)))] += 1
                            if _cv < self.CONF_BLANK:
                                _st["blanked"] += 1
                            else:
                                _st["kept"] += 1
                    if _cv is not None and _cv < self.CONF_BLANK:
                        continue                # hard-blanked: contributes nothing
                labs.append(l)
                face_names.append(face)
                positions.append(pos)
                confs.append(c9[pos] if c9 and c9[pos] is not None else 1.0)
        if self.TRUST_SOFT is not None and labs:
            # SOFT read-trust: keep every admitted cell, compute its p_trust
            # weight (max(p, floor); exempt no-conf cells at 1.0). Stored raw
            # here (aligned to labs); L1-normalized into self._wv below, after
            # any FACE_GATE/STICKER_GATE reduction reindexes it.
            self._softwv_raw = self._trust_soft_weights(
                read, labs, face_names, confs, _tobs,
                frame_motion, frame_aligned, frame_ev)
        self.n = len(labs)
        self.ok = self.n >= 6
        if not self.ok:
            return
        labs = np.asarray(labs, dtype=np.float64)
        dist = np.full((self.n, 6), 1e3)
        if self.EMISSION is not None:
            E = self.EMISSION
            lp_out = np.log(E["pi_out"]) + E["logp0"]
            lp_in = np.log(1.0 - E["pi_out"])
            for cname in E["mu"]:
                if cname not in COLOR_INDEX:
                    continue
                mu = np.asarray(E["mu"][cname], float)
                var = np.asarray(E["var"][cname], float)
                ll = (-0.5 * (((labs - mu) ** 2) / var).sum(1)
                      - 0.5 * np.log(var).sum() - 1.5 * np.log(2 * np.pi))
                dist[:, COLOR_INDEX[cname]] = -np.logaddexp(lp_in + ll, lp_out)
            if self.EMISSION_ADAPTER is not None:
                # phase-B affine fit-scale adapter (see class attr comment):
                # one guarded line, inherited uniformly by _dflat / CPU / GPU
                # scoring and every fit consumer. None (default) => untouched.
                _aa, _ab = self.EMISSION_ADAPTER
                dist = _aa + _ab * dist
        else:
            for cname, cen in centroids.items():
                if cname in COLOR_INDEX:
                    dist[:, COLOR_INDEX[cname]] = np.sqrt(
                        (((labs - np.asarray(cen, float)) ** 2) * _LAB_W).sum(1))
        if self.FACE_GATE is not None or self.STICKER_GATE is not None:
            mind = dist.min(axis=1)
            keep = np.ones(self.n, dtype=bool)
            if self.STICKER_GATE is not None:
                keep &= mind <= self.STICKER_GATE
            if self.FACE_GATE is not None:
                fmeans = {fn: float(np.mean(mind[[i for i, f in enumerate(face_names)
                                                  if f == fn]]))
                          for fn in set(face_names)}
                best = min(fmeans.values())
                for fn, fm in fmeans.items():
                    if fm > self.FACE_GATE * best and fm > self.FACE_GATE_MIN:
                        idx = [i for i, f in enumerate(face_names) if f == fn]
                        keep[idx] = False
            if not keep.all():
                labs = labs[keep]
                dist = dist[keep]
                face_names = [f for f, k in zip(face_names, keep) if k]
                positions = [p for p, k in zip(positions, keep) if k]
                confs = [c for c, k in zip(confs, keep) if k]
                if self._softwv_raw is not None:
                    self._softwv_raw = self._softwv_raw[keep]
                if self._trust_p_raw is not None:
                    self._trust_p_raw = self._trust_p_raw[keep]
                self.n = len(face_names)
                self.ok = self.n >= 6
                if not self.ok:
                    return
        self.labs = labs
        if self.DIST_CAP is not None:          # robust ceiling (see DIST_CAP)
            dist = np.minimum(dist, float(self.DIST_CAP))
        self.dist = dist
        # per-cell extraction confidence kept for the CONFIDENT-READ STATE
        # RE-ANCHOR (its mean over a span gates a PROMOTE). Stored independently of
        # CONF_W so it is available even when soft-weighting is off; pure storage,
        # never read by the scorer => byte-identical.
        self._confs = np.asarray(confs, dtype=np.float64) if confs else None
        sw = self._softwv_raw
        if sw is not None and float(sw.sum()) > 0:
            self._wv = sw / sw.sum()           # SOFT read-trust weighting
        elif self.CONF_W and confs:
            wv = np.maximum(np.asarray(confs, np.float64), 0.15)
            self._wv = wv / wv.sum()
        else:
            self._wv = None
        loc, present = {}, []
        for fn in face_names:
            if fn not in loc:
                loc[fn] = len(present)
                present.append(fn)
        self.faces = present
        self.combos = list(product(range(4), repeat=len(present)))
        # vectorized gather construction (same integers as the per-element loop):
        # g[c,k] = FACE_OFFSET[om(face_k)] + GRID_ROTATIONS[combo_c[loc_k]][pos_k]
        pos = np.asarray(positions)
        fidx = np.array([loc[fn] for fn in face_names])
        base = np.array([FACE_OFFSET[orient_map.get(fn, fn)] for fn in face_names])
        rots = np.asarray(GRID_ROTATIONS)
        combo_m = np.asarray(self.combos)
        gm = (base[None, :] + rots[combo_m[:, fidx], pos[None, :]]).astype(np.intp)
        self.gathers = list(gm)
        self._gm = gm                       # (n_combos, n) all gathers stacked
        self._ar = np.arange(self.n)
        self._dflat = None                  # built lazily from self.dist
        # per-cell provenance kept for the lazy pf-om bound (_face_ub_table):
        # grid position + read slot per admitted cell. Pure storage — the
        # scorer never reads these => byte-identical.
        self._pos = pos
        self._face_names = list(face_names)

    def _trust_soft_weights(self, read, labs, face_names, confs, obs,
                            frame_motion, frame_aligned, frame_ev=None):
        """Per-cell SOFT read-trust weight vector (TRUST_SOFT is set): the
        weighted-mean weight max(p_trust, floor) for an OBSERVED-conf cell, 1.0
        for a no-conf cell (EXEMPT). Feature building, caching and the
        om-independent per-read memo are shared across orientation hypotheses.
        Returns a float array aligned to `labs`."""
        tm = self.TRUST_SOFT
        obs = np.asarray(obs, bool)
        cache = tm.get("cache")
        key = id(read) if cache is not None else None
        p = cache.get(key) if key is not None else None
        if p is None or len(p) != len(labs):
            from detect.read_trust import (EV_CAP, FACE_SLOTS, build_features,
                                           legacy_min_dist)
            arr = np.asarray(labs, np.float64)
            conf_v = np.asarray(confs, np.float64)
            d1 = legacy_min_dist(arr, tm["cen_mat"])
            face_i = np.array([FACE_SLOTS.index(f) for f in face_names],
                              np.int64)
            n = len(arr)
            mot = np.full(n, 0.0 if frame_motion is None
                          else float(frame_motion))
            alg = np.full(n, np.nan if frame_aligned is None
                          else float(frame_aligned))
            evkw = {}
            if tm.get("n_features", 15) == 17:
                _es, _et = (frame_ev if frame_ev is not None
                            else (EV_CAP, EV_CAP))
                evkw = dict(
                    ev_since=np.full(n, EV_CAP if _es is None else float(_es)),
                    ev_to=np.full(n, EV_CAP if _et is None else float(_et)))
            X = build_features(conf_v, d1, arr, mot, alg, face_i,
                               np.zeros(n, np.int64), **evkw)
            p = np.asarray(tm["model"].predict_proba(X)[:, 1], np.float64)
            if key is not None:
                cache[key] = p
            st = self.TRUST_SOFT_STATS
            if st is not None:
                st["reads_scored"] += 1
                st["cells"] += int(len(p))
                st["exempt"] += int((~obs).sum())
                for pv in p[obs]:
                    st["p_hist"][max(0, min(19, int(pv / 0.05)))] += 1
        # Stash the RAW (pre-floor) p_trust aligned to `labs` at THIS point
        # (same order/length the FACE_GATE/STICKER_GATE trim below re-slices
        # _softwv_raw by) — covT instrumentation reads this, never the model.
        self._trust_p_raw = p
        return np.where(obs, np.maximum(p, float(tm.get("floor", 0.0))), 1.0)

    def score_states(self, states):
        """Best (max over grid-rotation combos) negative mean calibrated distance
        per state. Vectorized over (combos x states) in row chunks; bit-identical
        to the original per-gather loop (same contiguous-axis mean reduction,
        exact max)."""
        if not self.ok:
            return np.zeros(len(states))
        if self._dflat is None:
            self._dflat = np.ascontiguousarray(self.dist).ravel()
        if getattr(self, "_off", None) is None:
            self._off = self._ar * 6        # dist[k, c] == _dflat[6k + c]; set
            #   unconditionally (not only when _dflat was just built) so a pickled
            #   or GPU-touched seg whose _dflat is cached but _off was never set
            #   self-heals here instead of AttributeError-ing under GPU scoring.
        states = np.asarray(states)
        n_st = len(states)
        if (_GPU_SCRUB_SCORE and n_st >= _GPU_MIN_STATES
                and _torch_cuda() is not None):
            try:
                return _score_states_gpu(self, states)
            except Exception as e:      # OOM / driver hiccup -> exact CPU path
                _gpu_scrub_warn(e)
        best = np.full(n_st, -1e18)
        # chunk so the (chunk, n_combos, n) index tensor stays cache/memory-sane
        step = max(1, int(262144 // max(1, self._gm.shape[0] * self.n)) + 1)
        for s0 in range(0, n_st, step):
            cols = states[s0:s0 + step, self._gm.reshape(-1)]
            cols = cols.reshape(len(cols), self._gm.shape[0], self.n)
            dd = self._dflat[self._off + cols]
            d = (dd * self._wv).sum(2) if self._wv is not None else dd.mean(2)
            np.maximum.reduce(-d, axis=1, out=best[s0:s0 + step])
        return best

    def contrast(self, state_a, state_b):
        """Score two states ONLY on sticker positions where their predicted colors
        differ (under each state's own best gather). Look-alike states differ in few
        positions; mean-over-all scoring can drown that signal in shared-sticker
        noise.
        Returns (score_a, score_b, n_diff); higher = better."""
        if not self.ok:
            return 0.0, 0.0, 0
        def best_gather(st):
            bg, bd = None, 1e18
            for g in self.gathers:
                d = self.dist[self._ar, st[g]].mean()
                if d < bd:
                    bd, bg = d, g
            return bg
        ga, gb = best_gather(state_a), best_gather(state_b)
        ca, cb = state_a[ga], state_b[gb]
        mask = ca != cb
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0.0, 0
        da = self.dist[self._ar[mask], ca[mask]].mean()
        db = self.dist[self._ar[mask], cb[mask]].mean()
        return float(-da), float(-db), n

    def labeled_for_state(self, state_arr):
        """Relabel this read by a candidate state (best-fitting rotation):
        [(lab, color_name), ...] — feeds palette refresh / AdaptiveCentroids."""
        if not self.ok:
            return []
        best_g, best_d = None, 1e18
        for g in self.gathers:
            cols = state_arr[g]
            d = self.dist[self._ar, cols].mean()
            if d < best_d:
                best_d, best_g = d, g
        cols = state_arr[best_g]
        return [(self.labs[k], COLOR_LIST[cols[k]]) for k in range(self.n)]


def _sss_chunk(segs, states, outs, idxs, gflat, n_g, n, off32, c0, c1):
    """One (group, row-chunk) scoring unit of sum_seg_scores. Pure function of
    its slice: writes outs[i][c0:c1] for the group's segs and touches nothing
    else, so units can run in any order / concurrently with bit-identical
    results. The per-chunk numpy ops are VERBATIM the original loop body
    except cols stays int8 and the off32 add does the int32 upcast (same
    integer indices — colors are 0..5 and offsets <= 6*(n-1), both exact in
    either dtype — one fewer full pass; ddg and the reductions unchanged)."""
    cols = states[c0:c1, gflat]
    idx = off32 + cols.reshape(len(cols), n_g, n)    # int8 + int32 -> int32
    for i in idxs:
        ddg = segs[i]._dflat[idx]
        d = ((ddg * segs[i]._wv).sum(2)
             if getattr(segs[i], '_wv', None) is not None else ddg.mean(2))
        np.maximum.reduce(-d, axis=1, out=outs[i][c0:c1])


# Chunk-level worker pool for sum_seg_scores. Threads (not processes): the
# heavy passes (fancy gathers, mean/max reductions) release the GIL; processes
# would be fork-unsafe inside the threaded server. The pool is lazy and capped
# to avoid memory-bandwidth oversubscription.
_SSS_POOL = None
_SSS_WORKERS = min(8, (_os.cpu_count() or 1))


def _sss_pool():
    global _SSS_POOL
    if _SSS_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _SSS_POOL = ThreadPoolExecutor(_SSS_WORKERS,
                                       thread_name_prefix="sum_seg_scores")
    return _SSS_POOL


# GPU scoring (opt-in via CUBED_GPU_SCORING=1). The deep re-anchor's scoring is
# ~95% of decode wall and is a pure gather+reduce -> the idle GPU crushes it. Only
# LARGE candidate sets route here (CUBED_GPU_MIN_STATES, default 4000): the deep
# ball is tens of thousands of states, while shallow/prune calls stay on the exact
# CPU path (so the branch-and-bound prune's exactness guarantee is untouched, and
# the deep rescue — where prune is off anyway — is the only GPU consumer). fp64 to
# track the CPU floats closely; any GPU error falls back to CPU.
_GPU_SCORING = (_os.environ.get("CUBED_GPU_SCORING", "") == "1")
_GPU_MIN_STATES = int(_os.environ.get("CUBED_GPU_MIN_STATES", "4000"))
_GPU_SCRUB_SCORE = (_os.environ.get("CUBED_GPU_SCRUB_SCORE", "0") == "1"
                    and _GPU_SCORING)
_GPU_SCRUB_BATCH = (_os.environ.get("CUBED_GPU_SCRUB_BATCH", "0") == "1"
                    and _GPU_SCORING)
_TORCH = None
_GPU_OK = None
_GPU_WARNED = False
_GPU_SCRUB_WARNED = False

# pf-om RESOLVER GPU wiring (gated by the SAME CUBED_GPU_SCORING flag, so
# default-off is byte-identical). The resolver's per-frame ball is ~3.5k states
# (< _GPU_MIN_STATES 4000) and calls pure-numpy score_states, so it never hits
# the deep-re-anchor GPU path above. This option scores it on CUDA directly.
# Mode selector (CUBED_GPU_RESOLVER_MODE):
#   "all"  (default) — GPU-batch ALL 24 oms of a frame in ONE grouped gather call
#                      (they share dist/n/combos; only the gather matrix _gm
#                      differs), dropping the CPU lazy bound. Throughput over
#                      pruning: the shared states tensor uploads once per SPAN,
#                      the per-frame dist upload is tiny.
# "lazy" — keep the CPU per-face upper-bound prune and GPU-score
#                      only the contenders (seed the best on GPU, then batch the
#                      survivors with ub >= best - margin). A/B baseline for "all".
#   "off"           — resolver stays on CPU even with GPU scoring on (isolates the
#                      existing deep-re-anchor GPU speedup for measurement).
_GPU_RESOLVER_MODE = _os.environ.get("CUBED_GPU_RESOLVER_MODE", "all")
_GPU_RESOLVER_WARNED = False


def _gpu_resolver_warn(exc):
    global _GPU_RESOLVER_WARNED
    if not _GPU_RESOLVER_WARNED:
        print(f"  [gpu-resolver] fell back to CPU: {exc}", flush=True)
        _GPU_RESOLVER_WARNED = True


def _gpu_scrub_warn(exc):
    global _GPU_SCRUB_WARNED
    if not _GPU_SCRUB_WARNED:
        print(f"  [gpu-scrub-score] fell back to CPU: {exc}", flush=True)
        _GPU_SCRUB_WARNED = True


def _score_states_gpu(seg, states):
    """GPU mirror of AbsSegment.score_states, retaining the state axis.

    The reduction structure and fp64 currency match _sum_seg_scores_gpu.
    Candidate states upload once for this call; failures fall back to the exact
    CPU implementation at the caller.
    """
    torch = _TORCH
    dev = torch.device("cuda")
    n = seg.n
    n_g = seg._gm.shape[0]
    st = np.ascontiguousarray(states)
    N = len(st)
    states_t = torch.as_tensor(st, device=dev)
    dflat = torch.as_tensor(seg._dflat, device=dev)
    wv = (torch.as_tensor(seg._wv, device=dev).reshape(1, 1, n)
          if getattr(seg, "_wv", None) is not None else None)
    off = torch.as_tensor(seg._off, device=dev).reshape(1, 1, n)
    gm_flat = torch.as_tensor(
        seg._gm.reshape(-1).astype(np.int64), device=dev)
    best = torch.full((N,), -1e18, dtype=torch.float64, device=dev)
    chunk = max(1, int(8_000_000 // max(1, n_g * n)))
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        cols = states_t[c0:c1, gm_flat].to(torch.int64).reshape(
            c1 - c0, n_g, n)
        ddg = dflat[off + cols]
        d = (ddg * wv).sum(2) if wv is not None else ddg.mean(2)
        best[c0:c1] = (-d).amax(1)
    return best.cpu().numpy()


def _score_state_groups_gpu(groups):
    """Score compatible ``(AbsSegment, states)`` groups in one GPU batch.

    Scrub's chronological beam partitions candidates by orientation. Each
    partition is often below the per-call GPU threshold even when their union
    is large, causing 24 small CPU calls. Compatible orientations share the
    distance table, weights, and tensor shapes; stack their state matrices,
    vary only the gather matrix, and synchronize once per compatibility group.
    """
    torch = _TORCH
    dev = torch.device("cuda")
    out = [None] * len(groups)
    compatible = {}
    foreign = []
    for i, (seg, states) in enumerate(groups):
        if not (getattr(seg, "ok", False) and hasattr(seg, "_gm")):
            foreign.append(i)
            continue
        if seg._dflat is None:
            seg._dflat = np.ascontiguousarray(seg.dist).ravel()
        if getattr(seg, "_off", None) is None:
            seg._off = seg._ar * 6
        wv = getattr(seg, "_wv", None)
        key = (seg._dflat.tobytes(), b"" if wv is None else wv.tobytes(),
               seg._off.tobytes(), seg.n, seg._gm.shape[0])
        compatible.setdefault(key, []).append(i)

    for (_dfb, _wvb, _offb, n, n_g), idxs in compatible.items():
        lengths = [len(groups[i][1]) for i in idxs]
        if sum(lengths) < _GPU_MIN_STATES:
            foreign.extend(idxs)
            continue
        # Concatenate rather than pad to the largest orientation group. Beam
        # strata can be very imbalanced; padding makes the GPU score phantom
        # rows and erased the batching win in the first live probe.
        stacked = np.concatenate([
            np.asarray(groups[i][1], dtype=np.int8) for i in idxs])
        states_t = torch.as_tensor(stacked, device=dev)
        seg0 = groups[idxs[0]][0]
        dflat = torch.as_tensor(seg0._dflat, device=dev)
        wv = (torch.as_tensor(seg0._wv, device=dev).reshape(1, 1, n)
              if getattr(seg0, "_wv", None) is not None else None)
        off = torch.as_tensor(seg0._off, device=dev).reshape(1, 1, n)
        gm = np.stack([groups[i][0]._gm for i in idxs]).reshape(
            len(idxs), n_g * n).astype(np.int64)
        gm_t = torch.as_tensor(gm, device=dev)
        chunk = max(1, int(8_000_000 // max(1, n_g * n)))
        device_results = []
        state_off = 0
        for gi, length in enumerate(lengths):
            best = torch.full((length,), -1e18,
                              dtype=torch.float64, device=dev)
            for c0 in range(0, length, chunk):
                c1 = min(c0 + chunk, length)
                st = states_t[state_off + c0:state_off + c1]
                cols = st[:, gm_t[gi]].to(torch.int64).reshape(
                    c1 - c0, n_g, n)
                ddg = dflat[off + cols]
                d = (ddg * wv).sum(2) if wv is not None else ddg.mean(2)
                best[c0:c1] = (-d).amax(1)
            device_results.append(best)
            state_off += length
        result = torch.cat(device_results).cpu().numpy()
        state_off = 0
        for i, length in zip(idxs, lengths):
            out[i] = result[state_off:state_off + length]
            state_off += length

    for i in foreign:
        seg, states = groups[i]
        out[i] = seg.score_states(states)
    return out


def score_state_groups(groups):
    """Public scrub hook with exact per-segment fallback."""
    total_states = sum(len(states) for _seg, states in groups)
    if (_GPU_SCRUB_BATCH and total_states >= _GPU_MIN_STATES
            and _torch_cuda() is not None):
        try:
            return _score_state_groups_gpu(groups)
        except Exception as e:
            _gpu_scrub_warn(e)
    return [seg.score_states(states) for seg, states in groups]


def _score_state_segment_grid_device_gpu(segment_rows, states):
    """Score a read/orientation grid and retain the result on CUDA.

    The stateful scrub final scorer evaluates the same materialized state
    matrix for every orientation of every read.  Treating those cells as
    independent ``score_states`` calls repeats the state upload and forces a
    host synchronization per cell. Rows produced by ``AbsSegment``
    orientation views share their distance/weight payload and differ only in
    the gather matrix, so retain the read, orientation, and state axes on CUDA.
    The host wrapper below preserves the historical one-D2H return contract;
    fused stateful-OM DP consumes this tensor directly.
    """
    torch = _TORCH
    dev = torch.device("cuda")
    st = np.ascontiguousarray(states)
    n_states = len(st)
    states_t = torch.as_tensor(st, device=dev)
    widths = {len(row) for row in segment_rows}
    if len(widths) != 1:
        raise ValueError("segment grid rows must have one orientation width")
    width = widths.pop()
    result_t = torch.full(
        (len(segment_rows), width, n_states), -torch.inf,
        dtype=torch.float64, device=dev)

    for row_i, segments in enumerate(segment_rows):
        values_t = result_t[row_i]
        valid = [(oi, seg) for oi, seg in enumerate(segments)
                 if seg is not None]
        if not valid:
            continue

        for _oi, seg in valid:
            if not (getattr(seg, "ok", False) and hasattr(seg, "_gm")):
                raise TypeError("foreign segment in state score grid")
            if seg._dflat is None:
                seg._dflat = np.ascontiguousarray(seg.dist).ravel()
            if getattr(seg, "_off", None) is None:
                seg._off = seg._ar * 6

        seg0 = valid[0][1]
        wv0 = getattr(seg0, "_wv", None)
        compatibility = (
            seg0._dflat.tobytes(),
            b"" if wv0 is None else wv0.tobytes(),
            seg0._off.tobytes(),
            int(seg0.n),
            int(seg0._gm.shape[0]),
        )
        for _oi, seg in valid[1:]:
            wv = getattr(seg, "_wv", None)
            key = (
                seg._dflat.tobytes(),
                b"" if wv is None else wv.tobytes(),
                seg._off.tobytes(),
                int(seg.n),
                int(seg._gm.shape[0]),
            )
            if key != compatibility:
                raise ValueError("incompatible segments in state score row")

        _dfb, _wvb, _offb, n, n_g = compatibility
        valid_ois = [oi for oi, _seg in valid]
        dflat = torch.as_tensor(seg0._dflat, device=dev)
        wv = (torch.as_tensor(seg0._wv, device=dev).reshape(1, 1, 1, n)
              if wv0 is not None else None)
        off = torch.as_tensor(seg0._off, device=dev).reshape(1, 1, 1, n)
        gm = np.stack([seg._gm for _oi, seg in valid]).reshape(
            len(valid), n_g * n).astype(np.int64)
        gm_t = torch.as_tensor(gm, device=dev)
        gm_flat = gm_t.reshape(-1)
        valid_t = torch.as_tensor(valid_ois, dtype=torch.int64, device=dev)
        chunk = max(
            1, int(8_000_000 // max(1, len(valid) * n_g * n)))
        for c0 in range(0, n_states, chunk):
            c1 = min(c0 + chunk, n_states)
            cols = states_t[c0:c1, gm_flat].to(torch.int64).reshape(
                c1 - c0, len(valid), n_g, n)
            ddg = dflat[off + cols]
            d = (ddg * wv).sum(3) if wv is not None else ddg.mean(3)
            values_t[valid_t, c0:c1] = (-d).amax(2).T

    return result_t


def _score_state_segment_grid_gpu(segment_rows, states):
    """Historical host result around the resident grid implementation."""
    result = _score_state_segment_grid_device_gpu(
        segment_rows, states).cpu().numpy()
    return [result[i] for i in range(len(result))]


def score_state_segment_grid_device(segment_rows, states):
    """Return a compatible score grid on CUDA, or ``None`` fail-open.

    This is the device half of :func:`score_state_segment_grid`. It applies the
    same flags, work threshold, compatibility checks, and warning policy, but
    deliberately performs no device-to-host copy. Callers must treat ``None``
    as a request to use the exact scalar/NumPy path.
    """
    rows = [tuple(row) for row in segment_rows]
    if not rows:
        return None
    n_states = len(states)
    total_work = n_states * sum(
        sum(seg is not None for seg in row) for row in rows)
    compatible = bool(n_states) and all(
        all(seg is None or (getattr(seg, "ok", False)
                            and hasattr(seg, "_gm"))
            for seg in row)
        for row in rows)
    if (compatible and _GPU_SCRUB_BATCH
            and total_work >= _GPU_MIN_STATES
            and _torch_cuda() is not None):
        try:
            return _score_state_segment_grid_device_gpu(rows, states)
        except Exception as e:
            _gpu_scrub_warn(e)
    return None


def score_state_segment_grid(segment_rows, states):
    """Score ``[read][orientation]`` segments against one shared state matrix.

    Missing orientations remain ``-inf``.  CUDA-compatible rows use one
    resident state tensor and one final host transfer; foreign segments,
    disabled CUDA, and any device failure retain the scalar implementation.
    """
    rows = [tuple(row) for row in segment_rows]
    if not rows:
        return []
    n_states = len(states)
    total_work = n_states * sum(
        sum(seg is not None for seg in row) for row in rows)
    compatible = bool(n_states) and all(
        all(seg is None or (getattr(seg, "ok", False)
                            and hasattr(seg, "_gm"))
            for seg in row)
        for row in rows)
    if (compatible and _GPU_SCRUB_BATCH
            and total_work >= _GPU_MIN_STATES
            and _torch_cuda() is not None):
        try:
            return _score_state_segment_grid_gpu(rows, states)
        except Exception as e:
            _gpu_scrub_warn(e)

    out = []
    for row in rows:
        values = [
            (np.full(n_states, -np.inf) if seg is None
             else seg.score_states(states))
            for seg in row
        ]
        out.append(np.stack(values) if values else
                   np.empty((0, n_states), dtype=float))
    return out


def _score_state_group_runs_gpu(groups):
    """Advance fixed-key scrub groups across consecutive reads on CUDA.

    ``groups`` contains ``(states, base_scores, weighted_segments)`` tuples.
    The state/orientation keys do not change during a consecutive read run, so
    uploading and synchronizing once per read is pure orchestration overhead.
    Keep the concatenated states and accumulated fp64 scores resident for the
    whole run and return one final score vector per orientation group.

    Every read is still applied in chronological order.  The only numerical
    difference from the scalar host loop is CUDA fp64 round-off, covered by the
    same result-parity contract as the existing GPU scorers.
    """
    torch = _TORCH
    dev = torch.device("cuda")
    lengths = [len(states) for states, _base, _reads in groups]
    stacked = np.concatenate([
        np.asarray(states, dtype=np.int8) for states, _base, _reads in groups
    ])
    states_t = torch.as_tensor(stacked, device=dev)
    totals_t = torch.as_tensor(np.concatenate([
        np.asarray(base, dtype=np.float64) for _states, base, _reads in groups
    ]), device=dev)
    offsets = np.cumsum([0] + lengths).tolist()
    n_reads = len(groups[0][2])

    for ri in range(n_reads):
        compatible = {}
        for gi, (_states, _base, reads) in enumerate(groups):
            seg, weight = reads[ri]
            if seg._dflat is None:
                seg._dflat = np.ascontiguousarray(seg.dist).ravel()
            if getattr(seg, "_off", None) is None:
                seg._off = seg._ar * 6
            wv = getattr(seg, "_wv", None)
            key = (seg._dflat.tobytes(),
                   b"" if wv is None else wv.tobytes(),
                   seg._off.tobytes(), seg.n, seg._gm.shape[0],
                   float(weight))
            compatible.setdefault(key, []).append(gi)

        for (_dfb, _wvb, _offb, n, n_g, weight), idxs in compatible.items():
            seg0 = groups[idxs[0]][2][ri][0]
            dflat = torch.as_tensor(seg0._dflat, device=dev)
            wv = (torch.as_tensor(seg0._wv, device=dev).reshape(1, 1, n)
                  if getattr(seg0, "_wv", None) is not None else None)
            off = torch.as_tensor(seg0._off, device=dev).reshape(1, 1, n)
            gm = np.stack([
                groups[gi][2][ri][0]._gm for gi in idxs
            ]).reshape(len(idxs), n_g * n).astype(np.int64)
            gm_t = torch.as_tensor(gm, device=dev)
            chunk = max(1, int(8_000_000 // max(1, n_g * n)))
            for local_i, gi in enumerate(idxs):
                start, end = offsets[gi], offsets[gi + 1]
                for c0 in range(start, end, chunk):
                    c1 = min(c0 + chunk, end)
                    cols = states_t[c0:c1, gm_t[local_i]].to(
                        torch.int64).reshape(c1 - c0, n_g, n)
                    ddg = dflat[off + cols]
                    d = (ddg * wv).sum(2) if wv is not None else ddg.mean(2)
                    totals_t[c0:c1].add_((-d).amax(1), alpha=weight)

    result = totals_t.cpu().numpy()
    return [result[offsets[i]:offsets[i + 1]]
            for i in range(len(groups))]


def score_state_group_runs(groups):
    """Advance scrub groups across a fixed-key chronological read run.

    Falls back to the scalar scorer for foreign segment implementations and
    off-CUDA environments.  The public boundary deliberately accepts base
    scores so the CUDA path performs the complete chronological accumulation
    without a host round-trip between reads.
    """
    if not groups:
        return []
    n_reads = len(groups[0][2])
    compatible = bool(n_reads) and all(
        len(reads) == n_reads
        and all(getattr(seg, "ok", False) and hasattr(seg, "_gm")
                for seg, _weight in reads)
        for _states, _base, reads in groups)
    total_work = sum(len(states) * n_reads
                     for states, _base, _reads in groups)
    if (compatible and _GPU_SCRUB_BATCH
            and total_work >= _GPU_MIN_STATES
            and _torch_cuda() is not None):
        try:
            return _score_state_group_runs_gpu(groups)
        except Exception as e:
            _gpu_scrub_warn(e)

    out = []
    for states, base, reads in groups:
        scores = np.asarray(base, dtype=np.float64).copy()
        for seg, weight in reads:
            scores += float(weight) * seg.score_states(states)
        out.append(scores)
    return out


def prepare_stateful_read_run_device(weighted_segments_by_read, *, device):
    """Compile chronological scrub reads for device-resident beam scoring.

    ``weighted_segments_by_read`` contains ``(weight, segments_by_oi)`` rows.
    Each orientation segment is compiled into an exact compatibility group;
    compatible orientations share distance/weight tensors and differ only in
    their gather matrix.  The returned object is intentionally opaque and
    contains CUDA tensors only for the numerical payload.

    This is a strict device helper: foreign segment implementations raise so
    the caller can restart the complete attempt transactionally on the legacy
    path.  It never copies candidate state or score data to the host.
    """
    torch = _torch_cuda()
    if torch is None:
        raise RuntimeError("CUDA is unavailable for stateful scrub reads")
    compiled_reads = []
    n_om = None
    for weight, segments in weighted_segments_by_read:
        segments = tuple(segments)
        if n_om is None:
            n_om = len(segments)
        elif len(segments) != n_om:
            raise ValueError("stateful read orientation counts differ")
        groups = {}
        for oi, seg in enumerate(segments):
            if seg is None:
                continue
            if not (getattr(seg, "ok", False) and hasattr(seg, "_gm")):
                raise TypeError("foreign stateful read scorer")
            if seg._dflat is None:
                seg._dflat = np.ascontiguousarray(seg.dist).ravel()
            if getattr(seg, "_off", None) is None:
                seg._off = seg._ar * 6
            wv = getattr(seg, "_wv", None)
            key = (
                seg._dflat.tobytes(),
                b"" if wv is None else wv.tobytes(),
                seg._off.tobytes(),
                int(seg.n),
                int(seg._gm.shape[0]),
            )
            groups.setdefault(key, []).append((oi, seg))

        kernels = []
        for (_dfb, _wvb, _offb, n, n_g), members in groups.items():
            seg0 = members[0][1]
            oi_to_gm = np.full(int(n_om), -1, dtype=np.int64)
            for local_i, (oi, _seg) in enumerate(members):
                oi_to_gm[int(oi)] = local_i
            gm = np.stack([seg._gm for _oi, seg in members]).astype(
                np.int64, copy=False)
            kernels.append(dict(
                dflat=torch.as_tensor(seg0._dflat, device=device),
                off=torch.as_tensor(seg0._off, device=device).reshape(
                    1, 1, n),
                gm=torch.as_tensor(gm, device=device),
                oi_to_gm=torch.as_tensor(oi_to_gm, device=device),
                wv=(torch.as_tensor(seg0._wv, device=device).reshape(
                    1, 1, n)
                    if getattr(seg0, "_wv", None) is not None else None),
                n=n,
                n_g=n_g,
            ))
        compiled_reads.append((float(weight), tuple(kernels)))
    return dict(n_om=int(n_om or 0), reads=tuple(compiled_reads))


def score_stateful_read_run_device(states, scores, ois, plan):
    """Apply a compiled chronological read run without a host round-trip.

    Mirrors :meth:`AbsSegment.score_states` in fp64.  Candidates whose current
    orientation has no segment, or whose score becomes nonfinite, stay dead for
    the rest of the run.  Compaction is deliberately left to the caller at the
    read-run boundary so there is no synchronization between individual reads.
    Returns ``(updated_scores, alive_mask)`` on the original CUDA device.
    """
    torch = _torch_cuda()
    if torch is None:
        raise RuntimeError("CUDA is unavailable for stateful scrub reads")
    if not all(isinstance(value, torch.Tensor) and value.is_cuda
               for value in (states, scores, ois)):
        raise TypeError("stateful scrub beam tensors must be on CUDA")
    if (states.ndim != 2 or states.shape[1] != 54
            or states.dtype != torch.int8):
        raise ValueError("states must be torch.int8[N,54]")
    if scores.shape != (len(states),) or scores.dtype != torch.float64:
        raise ValueError("scores must be torch.float64[N]")
    if ois.shape != (len(states),) or ois.dtype != torch.int64:
        raise ValueError("ois must be torch.int64[N]")
    if not (states.device == scores.device == ois.device):
        raise ValueError("stateful scrub beam tensors must share one device")

    out = scores.clone()
    alive = torch.isfinite(out)
    n_states = len(states)
    for weight, kernels in plan["reads"]:
        delta = torch.full_like(out, -torch.inf)
        for kernel in kernels:
            local = kernel["oi_to_gm"][ois]
            member = local >= 0
            n = kernel["n"]
            n_g = kernel["n_g"]
            chunk = max(1, int(8_000_000 // max(1, n_g * n)))
            for c0 in range(0, n_states, chunk):
                c1 = min(c0 + chunk, n_states)
                local_c = local[c0:c1]
                member_c = member[c0:c1]
                gm = kernel["gm"][local_c.clamp_min(0)]
                cols = torch.gather(
                    states[c0:c1], 1, gm.reshape(c1 - c0, n_g * n)
                ).to(torch.int64).reshape(c1 - c0, n_g, n)
                dd = kernel["dflat"][kernel["off"] + cols]
                wv = kernel["wv"]
                dist = (dd * wv).sum(2) if wv is not None else dd.mean(2)
                candidate = (-dist).amax(1)
                delta[c0:c1] = torch.where(
                    member_c, candidate, delta[c0:c1])
        alive &= torch.isfinite(delta)
        out = torch.where(alive, out + float(weight) * delta, -torch.inf)
    return out, alive & torch.isfinite(out)


def _score_ball_oms_gpu(torch, segs, states_t):
    """Per-om MAX over the shared ball of each AbsSegment's best-gather score,
    on the GPU. Returns an fp64 numpy array aligned with `segs`.

    Mirrors AbsSegment.score_states (best over grid-rotation combos of the
    -(mean | _wv-weighted-sum) of dist-table lookups), then takes the max over
    ball states — but BATCHED across every seg that shares (dist, weights,
    shape). In the resolver all 24 oms of a frame share dist/_wv/n/n_combos and
    differ only in the gather matrix _gm, so they collapse to ONE group: the
    dist table uploads once and the 24 gathers stack into a single
    (G, n_combos, n) index. Same reduction structure as _sum_seg_scores_gpu,
    fp64; decisions match the CPU path to within float ULPs (the accepted
    GPU-scoring contract). states_t is a pre-uploaded (N, 54) int8 CUDA tensor
    (uploaded once per span, reused across the span's frames)."""
    dev = states_t.device
    N = int(states_t.shape[0])
    out = np.empty(len(segs), np.float64)
    groups = {}
    for i, s in enumerate(segs):
        if s._dflat is None:
            s._dflat = np.ascontiguousarray(s.dist).ravel()
            s._off = s._ar * 6      # keep the CPU-fallback invariant (score_states)
        wvk = s._wv.tobytes() if getattr(s, "_wv", None) is not None else b""
        groups.setdefault((s._dflat.tobytes(), wvk, s.n, s._gm.shape[0]),
                          []).append(i)
    for (_dfb, _wvb, n, n_g), idxs in groups.items():
        s0 = segs[idxs[0]]
        dflat = torch.as_tensor(s0._dflat, device=dev)             # (n*6,) f64
        wv = (torch.as_tensor(s0._wv, device=dev)
              if getattr(s0, "_wv", None) is not None else None)   # (n,) f64
        off = torch.arange(n, device=dev, dtype=torch.int64) * 6   # (n,)
        G = len(idxs)
        gms = np.stack([segs[i]._gm for i in idxs]).reshape(
            G, n_g * n).astype(np.int64)
        gm_flat = torch.as_tensor(gms.reshape(-1), device=dev)     # (G*n_g*n,)
        gmax = torch.full((G,), -1e18, dtype=torch.float64, device=dev)
        chunk = max(1, int(8_000_000 // max(1, G * n_g * n)))      # bound idx mem
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            cols = states_t[c0:c1][:, gm_flat].to(torch.int64).reshape(
                c1 - c0, G, n_g, n)
            ddg = dflat[off + cols]                                # (c,G,n_g,n)
            d = (ddg * wv).sum(-1) if wv is not None else ddg.mean(-1)
            m = (-d).amax(2).amax(0)                               # combos, states
            gmax = torch.maximum(gmax, m)
        gm_res = gmax.cpu().numpy()
        for j, i in enumerate(idxs):
            out[i] = gm_res[j]
    return out


def _torch_cuda():
    global _TORCH, _GPU_OK
    if _GPU_OK is None:
        try:
            import torch as _t
            _GPU_OK = bool(_t.cuda.is_available())
            _TORCH = _t if _GPU_OK else None
        except Exception:
            _GPU_OK, _TORCH = False, None
    return _TORCH


def _score_transition_batch_torch(batch, segment_blocks, *, require_cuda=False):
    """Score a ragged main-trellis transition batch without leaving its device.

    ``batch`` is the structure-of-arrays ABI from
    :mod:`detect.trellis_device_transition`: one shared ``int8[M,54]`` state
    pool plus a candidate ``int64[N]`` state index and host block offsets.
    ``segment_blocks[b]`` is the ordered list of compatible ``AbsSegment``
    scorers for candidate block ``b``.  The result is a resident
    ``float64[N]`` tensor aligned exactly with the batch's candidate fields.

    Candidate states are gathered from ``state_pool`` only in bounded device
    chunks.  In particular, this function never materializes the batch's full
    candidate-state property and performs no device-to-host transfer.  The
    small segment descriptors
    (distance tables, gather matrices, and optional weights) remain host-owned
    inputs and are uploaded once per call.

    The arithmetic mirrors ``_sum_seg_scores_gpu``: segments sharing a gather
    matrix reuse one gathered color tensor, each segment performs its own fp64
    mean/weighted-sum and max-over-grid reduction, and block evidence is the
    mean over its ordered compatible segments.  ``require_cuda=False`` exists
    solely so the same tensor implementation can serve as a CPU test oracle;
    production callers use :func:`score_transition_batch_device`.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised by import environments
        raise RuntimeError("torch is required for transition device scoring") from exc

    state_pool = getattr(batch, "state_pool", None)
    state_index = getattr(batch, "state_index", None)
    if not isinstance(state_pool, torch.Tensor) or not isinstance(
            state_index, torch.Tensor):
        raise TypeError("transition state_pool and state_index must be torch tensors")
    if require_cuda and (not state_pool.is_cuda or not state_index.is_cuda):
        raise TypeError("transition state_pool and state_index must be CUDA tensors")
    if state_pool.device != state_index.device:
        raise ValueError("transition state_pool and state_index must share one device")
    if state_pool.dtype != torch.int8 or state_pool.ndim != 2 \
            or state_pool.shape[1] != 54:
        raise ValueError("transition state_pool must be int8[M,54]")
    if state_index.dtype != torch.int64 or state_index.ndim != 1:
        raise ValueError("transition state_index must be int64[N]")

    blocks = tuple(tuple(segs) for segs in segment_blocks)
    offsets = tuple(int(value) for value in getattr(batch, "block_offsets", ()))
    n_candidates = int(state_index.shape[0])
    if len(offsets) != len(blocks) + 1:
        raise ValueError("segment_blocks must align with transition block_offsets")
    if (not offsets or offsets[0] != 0 or offsets[-1] != n_candidates
            or any(right < left for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("transition block_offsets must partition all candidates")

    evidence = torch.empty(
        n_candidates, dtype=torch.float64, device=state_pool.device)
    for block_i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        segs = blocks[block_i]
        if not segs:
            raise ValueError(
                f"transition segment block {block_i} has no compatible scorers")

        groups = {}
        for seg_i, seg in enumerate(segs):
            if not getattr(seg, "ok", False) or not hasattr(seg, "_gm"):
                raise TypeError(
                    f"transition segment block {block_i} scorer {seg_i} is foreign "
                    "or incompatible")
            n = int(getattr(seg, "n", 0))
            gm = np.asarray(seg._gm)
            if n <= 0 or gm.ndim != 2 or gm.shape[1] != n:
                raise ValueError(
                    f"transition segment block {block_i} scorer {seg_i} has an "
                    "invalid gather matrix")
            if gm.size and (int(gm.min()) < 0 or int(gm.max()) >= 54):
                raise ValueError(
                    f"transition segment block {block_i} scorer {seg_i} gathers "
                    "outside the 54-facelet state")

            if getattr(seg, "_dflat", None) is None:
                if not hasattr(seg, "dist"):
                    raise TypeError(
                        f"transition segment block {block_i} scorer {seg_i} has no "
                        "distance table")
                seg._dflat = np.ascontiguousarray(seg.dist).ravel()
            if getattr(seg, "_off", None) is None:
                if not hasattr(seg, "_ar"):
                    raise TypeError(
                        f"transition segment block {block_i} scorer {seg_i} has no "
                        "distance offsets")
                seg._off = np.asarray(seg._ar) * 6

            dflat = np.asarray(seg._dflat)
            off = np.asarray(seg._off)
            wv = getattr(seg, "_wv", None)
            if dflat.shape != (n * 6,) or off.shape != (n,):
                raise ValueError(
                    f"transition segment block {block_i} scorer {seg_i} has an "
                    "invalid distance payload")
            if wv is not None:
                wv = np.asarray(wv)
                if wv.shape != (n,):
                    raise ValueError(
                        f"transition segment block {block_i} scorer {seg_i} has "
                        "invalid cell weights")

            key = (gm.tobytes(), n, int(gm.shape[0]))
            groups.setdefault(key, []).append((gm, dflat, off, wv))

        device_groups = []
        max_gather_work = 1
        for (_gmb, n, n_g), payloads in groups.items():
            gm = payloads[0][0]
            gm_t = torch.as_tensor(
                gm.reshape(-1).astype(np.int64, copy=False),
                dtype=torch.int64, device=state_pool.device)
            segment_tensors = []
            for _gm, dflat, off, wv in payloads:
                dflat_t = torch.as_tensor(
                    dflat, dtype=torch.float64, device=state_pool.device)
                off_t = torch.as_tensor(
                    off, dtype=torch.int64, device=state_pool.device,
                ).reshape(1, 1, n)
                wv_t = (torch.as_tensor(
                    wv, dtype=torch.float64, device=state_pool.device,
                ).reshape(1, 1, n) if wv is not None else None)
                segment_tensors.append((dflat_t, off_t, wv_t))
            device_groups.append((gm_t, n, n_g, segment_tensors))
            max_gather_work = max(max_gather_work, n_g * n)

        block_size = end - start
        block_total = torch.zeros(
            block_size, dtype=torch.float64, device=state_pool.device)
        chunk = max(1, int(8_000_000 // max_gather_work))
        for c0 in range(0, block_size, chunk):
            c1 = min(c0 + chunk, block_size)
            states_t = state_pool[state_index[start + c0:start + c1]]
            for gm_t, n, n_g, segment_tensors in device_groups:
                cols = states_t[:, gm_t].to(torch.int64).reshape(
                    c1 - c0, n_g, n)
                for dflat_t, off_t, wv_t in segment_tensors:
                    ddg = dflat_t[off_t + cols]
                    d = ((ddg * wv_t).sum(2)
                         if wv_t is not None else ddg.mean(2))
                    block_total[c0:c1] += (-d).amax(1)
        evidence[start:end] = block_total / len(segs)
    return evidence


def score_transition_batch_device(batch, segment_blocks):
    """CUDA-only resident evidence scorer for a main-trellis transition batch.

    Incompatible tensors or segment implementations raise before candidate
    scoring; the future live integration can catch that exception and execute
    the exact historical NumPy path transactionally.
    """
    return _score_transition_batch_torch(
        batch, segment_blocks, require_cuda=True)


def _sum_seg_scores_gpu(segs, states):
    """GPU mirror of sum_seg_scores: per seg, max over gather-combos of the
    -(mean | _wv-weighted sum) of dist-table lookups, summed over segs (the
    caller divides by len(segs)). Same reduction structure as _sss_chunk in
    fp64. Foreign scorers (no _gm) use the CPU score_states path."""
    torch = _TORCH
    dev = torch.device("cuda")
    st = np.asarray(states)
    N = len(st)
    states_t = torch.as_tensor(st, device=dev)                 # (N,54) int8
    tot = torch.zeros(N, dtype=torch.float64, device=dev)
    groups, foreign = {}, []
    for s in segs:
        if getattr(s, "ok", False) and hasattr(s, "_gm"):
            groups.setdefault((s._gm.tobytes(), s.n), []).append(s)
        else:
            foreign.append(s)
    for (_gmb, n), gsegs in groups.items():
        gm = gsegs[0]._gm                                       # (n_g, n)
        n_g = gm.shape[0]
        gm_t = torch.as_tensor(gm.reshape(-1).astype(np.int64), device=dev)
        off = (torch.arange(n, device=dev, dtype=torch.int64) * 6).reshape(1, 1, n)
        dts, wvs = [], []
        for s in gsegs:
            if s._dflat is None:
                s._dflat = np.ascontiguousarray(s.dist).ravel()
                s._off = s._ar * 6        # mirror the CPU fallback (score_states): a
                #   GPU-touched seg later re-scored on CPU else AttributeErrors on _off
            dts.append(torch.as_tensor(s._dflat, device=dev))  # (n*6,) f64
            wvs.append(torch.as_tensor(s._wv, device=dev).reshape(1, 1, n)
                       if getattr(s, "_wv", None) is not None else None)
        chunk = max(1, int(8_000_000 // max(1, n_g * n)))       # bound idx memory
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            cols = states_t[c0:c1, gm_t].to(torch.int64).reshape(c1 - c0, n_g, n)
            idx = off + cols                                   # (c,n_g,n) into dflat
            for si in range(len(gsegs)):
                ddg = dts[si][idx]                             # (c,n_g,n)
                d = (ddg * wvs[si]).sum(2) if wvs[si] is not None else ddg.mean(2)
                tot[c0:c1] += (-d).amax(1)                     # max over combos
    res = tot.cpu().numpy()
    for s in foreign:
        res = res + s.score_states(states)
    return res


def sum_seg_scores(segs, states):
    """sum of score_states(states) over segs, BIT-IDENTICAL to the naive loop
    (per-seg values identical — chunking never alters the per-row contiguous
    mean or the exact max — and the cross-seg additions run in the same order).
    Segments sharing one gather matrix (a held span seen under one om usually
    yields 16 identical visibility patterns) share ONE gathered index tensor —
    the dominant elementwise pass on big candidate sets is then the per-seg
    table lookup only (~2-3x less memory traffic than 16 solo passes).
    Big candidate sets additionally fan the FIXED chunk grid out to a thread
    pool: every unit computes the same arrays with the same chunk boundaries
    and writes a disjoint output slice, so the result is independent of
    scheduling (bit-identical to the serial loop, any worker count)."""
    if _GPU_SCORING and len(states) >= _GPU_MIN_STATES and _torch_cuda() is not None:
        try:
            return _sum_seg_scores_gpu(segs, states)
        except Exception as e:                      # OOM / driver hiccup -> CPU
            global _GPU_WARNED
            if not _GPU_WARNED:
                print(f"  [gpu-scoring] fell back to CPU: {e}", flush=True)
                _GPU_WARNED = True
    outs = [None] * len(segs)
    groups = {}
    for i, s in enumerate(segs):
        if not getattr(s, "ok", False) or not hasattr(s, "_gm"):
            outs[i] = s.score_states(states)        # foreign scorer: solo path
            continue
        groups.setdefault((s._gm.tobytes(), s.n), []).append(i)
    if groups:
        states = np.asarray(states)
        n_st = len(states)
        units = []
        for (_, n), idxs in groups.items():
            gm = segs[idxs[0]]._gm
            n_g = gm.shape[0]
            gflat = gm.reshape(-1)
            off32 = (np.arange(n) * 6).astype(np.int32)
            for i in idxs:
                outs[i] = np.full(n_st, -1e18)
                if segs[i]._dflat is None:
                    segs[i]._dflat = np.ascontiguousarray(segs[i].dist).ravel()
                    segs[i]._off = segs[i]._ar * 6
            step = max(1, int(262144 // max(1, n_g * n)) + 1)
            for c0 in range(0, n_st, step):
                units.append((idxs, gflat, n_g, n, off32,
                              c0, min(c0 + step, n_st)))
        if len(units) >= 4 and n_st >= 4096 and _SSS_WORKERS > 1:
            futs = [_sss_pool().submit(_sss_chunk, segs, states, outs, *u)
                    for u in units]
            for f in futs:
                f.result()      # re-raises worker exceptions
        else:
            for u in units:
                _sss_chunk(segs, states, outs, *u)
    tot = np.zeros(len(states))
    for o in outs:
        tot += o
    return tot


def weighted_seg_scores(segs, states, weights):
    """Sticker-count-weighted span vote: sum(w_i * score_i) / sum(w_i).
    In the 1-face-dominant read regime a 2-face read (12-18 stickers) is the
    only orientation anchor in its span — equal-weight averaging lets the
    ambiguous 1-face majority swamp it."""
    outs = [None] * len(segs)
    for i, s in enumerate(segs):
        outs[i] = s.score_states(states)
    tot = np.zeros(len(states))
    wsum = 0.0
    for o, w in zip(outs, weights):
        tot += w * o
        wsum += w
    return tot / max(wsum, 1e-9)


# --------------------------------------------------------------------------- #
# Post-solve reconstruction: canonical-frame moves + orientation timeline ->
# human notation with cube rotations (x/y/z) and locally-renamed moves.
# The tree search stays orientation-free (canonical frame); this runs after.
# --------------------------------------------------------------------------- #
_ROTS = [r + s for r in "xyz" for s in ("", "'", "2")]


def _om_of_cube(c):
    from detect.calibrator import CENTER_COLOR_TO_FACE
    return (CENTER_COLOR_TO_FACE[c.state["up"][4]],
            CENTER_COLOR_TO_FACE[c.state["front"][4]])


def _om_key(om):
    return (om["up"], om["front"])


# omk -> its stable 0-23 index in ORIENTATIONS (the same enumeration order the
# DP/emission code already uses, e.g. `omks = [_om_key(o) for o in
# ORIENTATIONS]` + `omks.index(...)`). Used by the CUBED_OMPF_LOG line
# so the om field is a plain, parseable int (ORIENTATIONS[omk] recovers it)
# instead of the nested-tuple key.
_OMK_TO_IDX = {_om_key(o): i for i, o in enumerate(ORIENTATIONS)}

_SINGLE_ROT_NEIGHBORS = {}


def _reaches_after_scrub(legacy_reaches, report):
    """Endpoint status for the word the tracker actually emits.

    Scrub may replace the legacy word after terminal selection.  When it does
    and has an endpoint verdict, production's direction arbiter must consume
    that verdict rather than the now-stale legacy status.  A fail-soft/no-op
    scrub or an unknown endpoint preserves the historical value.
    """
    endpoint = report.get("final_endpoint_ok")
    if report.get("emitted") and endpoint is not None:
        return bool(endpoint)
    return legacy_reaches


def _move_frames_after_scrub(moves, move_layer, meta, bridge, report):
    """Bind emitted raw moves to either legacy or selected-LL frame authority.

    The ordinary path is intentionally the historical implementation verbatim:
    span-entry attribution followed by terminal-bridge attribution to the last
    span.  A selected LL suffix has stronger exact action-frame receipts, and
    the pre-scrub ``bridge`` no longer describes that suffix.  In that case the
    receipt must bind exactly to the emitted suffix and remain chronological;
    any mismatch returns no attribution instead of manufacturing timestamps.
    """
    move_frames = [int(meta[min(si, len(meta) - 1)][0]) for si in move_layer]
    report = report or {}
    if report.get("ll_completion_applied") is not True:
        for i in range(len(moves) - len(bridge), len(moves)):
            move_frames[i] = int(meta[-1][1])
        return move_frames

    completion = report.get("ll_completion")
    if (report.get("emitted") is not True
            or not isinstance(completion, dict)
            or len(move_frames) != len(moves)
            or completion.get("status") != "selected"
            or completion.get("applied") is not True):
        return []
    selected_frames = completion.get("selected_move_frames")
    selected_moves = completion.get("selected_canonical_moves")
    if (not isinstance(selected_frames, (list, tuple))
            or not isinstance(selected_moves, (list, tuple))
            or not selected_frames
            or len(selected_frames) != len(selected_moves)
            or len(selected_moves) > len(moves)
            or tuple(moves[-len(selected_moves):]) != tuple(selected_moves)
            or any(isinstance(frame, (bool, np.bool_))
                   or not isinstance(frame, (int, np.integer))
                   for frame in selected_frames)):
        return []
    selected_frames = [int(frame) for frame in selected_frames]
    if (any(frame < 0 for frame in selected_frames)
            or any(a > b for a, b in zip(selected_frames, selected_frames[1:]))
            or selected_frames[0] < int(meta[0][0])
            or selected_frames[-1] > int(meta[-1][1])):
        return []
    move_frames[-len(selected_frames):] = selected_frames
    if any(a > b for a, b in zip(move_frames, move_frames[1:])):
        return []
    return move_frames


def single_rot_neighbors(omk):
    """The distinct camera oms reachable from `omk` by exactly ONE cube rotation
    token (x/y/z + primes/doubles) — the only oms a wide/slice can produce (Rw's
    om change == x; M's == x'). Bounds the co-commit search to <=9 targets instead
    of all 24. Cached."""
    if omk in _SINGLE_ROT_NEIGHBORS:
        return _SINGLE_ROT_NEIGHBORS[omk]
    from core.cube import Cube
    from detect.scramble import apply_move
    base = None                                  # a rotation word (<=2) for omk
    for seq in [[]] + [[r] for r in _ROTS] + [[a, b] for a in _ROTS for b in _ROTS]:
        c = Cube()
        for r in seq:
            apply_move(c, r)
        if _om_of_cube(c) == omk:
            base = seq
            break
    out = []
    if base is not None:
        for r in _ROTS:
            c = Cube()
            for s in base:
                apply_move(c, s)
            apply_move(c, r)
            nk = _om_of_cube(c)
            if nk != omk and nk not in out:
                out.append(nk)
    _SINGLE_ROT_NEIGHBORS[omk] = out
    return out


def rotation_word(om_from, om_to, max_len=2):
    """Shortest x/y/z word taking the holder's frame om_from -> om_to (BFS; the
    rotation group has diameter 2 over the 9 single rotations)."""
    from core.cube import Cube
    from detect.scramble import apply_move
    if _om_key(om_from) == _om_key(om_to):
        return []

    def frame_cube(om):
        # find a rotation sequence producing this om from identity (depth <= 2)
        for seq in [[]] + [[r] for r in _ROTS] + [[a, b] for a in _ROTS for b in _ROTS]:
            c = Cube()
            for r in seq:
                apply_move(c, r)
            if _om_of_cube(c) == _om_key(om):
                return c
        raise ValueError(f"unreachable orientation {om}")

    start = frame_cube(om_from)
    target = _om_key(om_to)
    frontier = [(start, [])]
    seen = {_om_of_cube(start)}
    for _ in range(max_len):
        nxt = []
        for c, word in frontier:
            for r in _ROTS:
                c2 = Cube({f: list(v) for f, v in c.state.items()})
                apply_move(c2, r)
                k = _om_of_cube(c2)
                if k == target:
                    return word + [r]
                if k not in seen:
                    seen.add(k)
                    nxt.append((c2, word + [r]))
        frontier = nxt
    raise ValueError("no rotation word found")


def _full_om(om):
    """Extend an {up, front, right} orient_map to all six spatial slots."""
    opp = {"up": "down", "down": "up", "front": "back", "back": "front",
           "left": "right", "right": "left"}
    full = dict(om)
    for slot in ("up", "front", "right"):
        full[opp[slot]] = opp[full[slot]]
    return full


_FACE_LETTER = {"up": "U", "down": "D", "front": "F", "back": "B",
                "left": "L", "right": "R"}
_LETTER_FACE = {v: k for k, v in _FACE_LETTER.items()}


def rename_to_viewer(move, om):
    """A canonical-frame move, as the HOLDER names it under orientation om
    (om: spatial slot -> model face). Model face F sits at the spatial slot s with
    om[s] == F; the holder calls a turn of that slot by the slot's letter."""
    full = _full_om(om)
    model_face = _LETTER_FACE[move[0]]
    for slot, mf in full.items():
        if mf == model_face:
            return _FACE_LETTER[slot] + move[1:]
    raise ValueError(f"face {model_face} not in om {om}")


def reconstruct_with_rotations(moves, oms):
    """Convert canonical-frame moves + the per-move orientation timeline into the
    HUMAN notation actually performed: cube rotations inserted at orientation
    changes, subsequent moves renamed into the holder's local frame.

    moves: canonical-frame move list (the tracker's output).
    oms:   orient_map active when each move was performed (same length).
    The INITIAL orientation is treated as the solver's reference frame (no leading
    rotation; all moves renamed into the frame they were performed in).
    """
    assert len(moves) == len(oms)
    out = []
    if not moves:
        return out
    # The tracked oms are the CAMERA's frames; the SOLVER's frame starts at identity
    # (protocol: white-top-green-front toward the solver) and evolves by the SAME
    # intrinsic rotation words the camera frame does (x/y/z are intrinsic in cube.py,
    # so physical rotations transfer between frames as identical words). Renaming
    # uses the solver-relative frame; a front camera seeing the cube's back face thus
    # yields canonical names for the opening moves, as the solver performed them.
    from core.cube import Cube
    from detect.scramble import apply_move as _am
    cam_prev = oms[0]
    solver_cube = Cube()                      # identity frame
    for mv, om in zip(moves, oms):
        if _om_key(om) != _om_key(cam_prev):
            word = rotation_word(cam_prev, om)
            out.extend(word)
            for r in word:
                _am(solver_cube, r)
            cam_prev = om
        sk = _om_of_cube(solver_cube)
        solver_om = {"up": sk[0], "front": sk[1]}
        # derive the right slot of the solver frame for full renaming
        for o in ORIENTATIONS:
            if _om_key(o) == sk:
                solver_om = o
                break
        out.append(rename_to_viewer(mv, solver_om))
    return out


def faithful_reconstruct(moves, oms, evidence=None):
    """reconstruct_with_rotations + per-burst FOLDING of wide/rotation tokens.

    Each emitted move forms one burst = (rotation word inserted at an om change) +
    (the renamed face move). A burst is folded to a single faithful token only when
    `evidence[i]` is present AND its motion gate permits it (band_extent==2 for a
    wide). Folding is net-perm-preserving by construction, so the solve's facelet
    outcome is unchanged (verify with cubemodel.same_state).

    evidence: optional list aligned with `moves`; each None or
              {'band_extent':int, 'moved_middle':bool}. A burst with no evidence is
              emitted UNFOLDED (conservative — no false wides on face-only solves).
              evidence=None for ALL moves => output identical to
              reconstruct_with_rotations.

    NOTE: a single emitted move yields a 1-face-move burst, so this folds wides
    (rotation+1 face) and rotations only. SLICES (M/E/S = rotation+2 faces) need
    multi-move bursts grouped together — deferred to Phase 2's in-burst co-commit.
    """
    from analysis import cubemodel as cm
    from core.cube import Cube
    from detect.scramble import apply_move as _am
    assert len(moves) == len(oms)
    out = []
    if not moves:
        return out
    cam_prev = oms[0]
    solver_cube = Cube()
    for i, (mv, om) in enumerate(zip(moves, oms)):
        burst = []
        if _om_key(om) != _om_key(cam_prev):
            word = rotation_word(cam_prev, om)
            burst.extend(word)
            for r in word:
                _am(solver_cube, r)
            cam_prev = om
        sk = _om_of_cube(solver_cube)
        solver_om = {"up": sk[0], "front": sk[1]}
        for o in ORIENTATIONS:
            if _om_key(o) == sk:
                solver_om = o
                break
        burst.append(rename_to_viewer(mv, solver_om))
        ev = evidence[i] if (evidence and i < len(evidence)) else None
        if ev is None:
            out.extend(burst)
        else:
            out.extend(cm.fold_burst(burst, ev.get("band_extent"),
                                     ev.get("moved_middle")))
    return out


def reconstruct_with_rotations_safe(moves, oms, evidence=None):
    """Safe wrapper: human notation with rotations when the orientation timeline is
    usable; FALLBACK to the canonical white-top-green-front moves on any failure
    (missing/short om timeline, unreachable orientation, internal error).

    With `evidence` supplied, additionally folds wide/rotation bursts
    (faithful_reconstruct). evidence=None preserves the exact legacy output.

    Returns (notation_list, used_rotations: bool)."""
    try:
        if not moves or not oms or len(oms) != len(moves):
            return list(moves), False
        if evidence is None:
            rec = reconstruct_with_rotations(moves, oms)
        else:
            rec = faithful_reconstruct(moves, oms, evidence)
        return rec, any(m[0] in "xyz" for m in rec)
    except Exception:
        return list(moves), False


def establish_orientation_from_reads(reads, anchor_arr, centroids, max_reads=25,
                                     cell_confs=None, frame_meta=None):
    """Pick the orient_map (of 24) under which the given reads best match a KNOWN
    state (e.g. the scramble state during the opening hold).

    cell_confs: optional list parallel to `reads` ({face: [9]} or None per read),
    threaded into AbsSegment so CONF_BLANK/CONF_W reach om0 resolution too (not
    just the main seg_factory path). None (default, and every pre-existing
    caller) => byte-identical, matching the original no-cell_conf behavior.
    frame_meta: optional list parallel to `reads` of (motion, aligned) or
    (motion, aligned, ev_since, ev_to) tuples (or None per read) for the
    TRUST_SOFT feature builder — same threading pattern as cell_confs, same
    None => byte-identical default; the 2-tuple form stays valid (v1 models
    never read the event context)."""
    best_om, best_tot = ORIENTATIONS[0], -1e18
    for om in ORIENTATIONS:
        tot = 0.0
        for i, r in enumerate(reads[:max_reads]):
            cc = cell_confs[i] if cell_confs is not None else None
            fm = frame_meta[i] if frame_meta is not None else None
            seg = AbsSegment(r, centroids, om, cell_conf=cc,
                             frame_motion=fm[0] if fm else None,
                             frame_aligned=fm[1] if fm else None,
                             frame_ev=(fm[2], fm[3])
                             if fm and len(fm) > 2 else None)
            if seg.ok:
                tot += float(seg.score_states(anchor_arr[None, :])[0])
        if tot > best_tot:
            best_tot, best_om = tot, om
    return best_om, best_tot

_OPP = {"U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"}
_QT = {"": 1, "'": 3, "2": 2}


def _face(m: str) -> str:
    return m[0]


def _commutes(a: str, b: str) -> bool:
    return _face(a) == _face(b) or _OPP[_face(a)] == _face(b)


def _mk(face: str, q: int) -> str | None:
    q %= 4
    return None if q == 0 else face + {1: "", 2: "2", 3: "'"}[q]


def simplify_moves(moves: list[str]) -> list[str]:
    """Canonical simplification (exact algebra): merge same-face turns and cancel
    across commuting (opposite-face) moves until fixpoint."""
    mv = list(moves)
    changed = True
    while changed:
        changed = False
        for i in range(len(mv)):
            j = i + 1
            while j < len(mv):
                if _face(mv[j]) == _face(mv[i]):
                    merged = _mk(_face(mv[i]), _QT[mv[i][1:]] + _QT[mv[j][1:]])
                    mv = mv[:i] + ([merged] if merged else []) + mv[i + 1:j] + mv[j + 1:]
                    changed = True
                    break
                if not _commutes(mv[i], mv[j]):
                    break
                j += 1
            if changed:
                break
    return mv


def _build_perm_tables(max_depth=4):
    """All distinct move-path permutations up to max_depth, as one int matrix per
    depth: PERMS[d] = (N_d, 54) permutation rows + PATHS[d] = the move list per row.
    Ball expansion from any state becomes states[arr][PERMS[d]] - a single gather -
    instead of ~1e5 Python dict inserts (live-latency requirement: pipeline must
    finish within ~1-2s of solve end)."""
    ident = np.arange(54)
    seen = {ident.tobytes(): []}
    rows = [ident]
    paths = [[]]
    frontier = [(ident, [])]
    by_depth = {0: (np.stack(rows), list(paths))}
    for d in range(1, max_depth + 1):
        nxt = []
        for perm, path in frontier:
            for notation, mperm in MOVE_LIST:
                np_ = perm[mperm]
                k = np_.tobytes()
                if k not in seen:
                    e = path + [notation]
                    seen[k] = e
                    rows.append(np_)
                    paths.append(e)
                    nxt.append((np_, e))
        frontier = nxt
        by_depth[d] = (np.stack(rows), list(paths))
    return by_depth


import os as _os
# Optional on-disk permutation-table cache; recomputed in-process when the
# path is absent, so a missing file degrades gracefully.
_PT_CACHE_NPZ = "/workspace/_perm_tables_d5.npz"
_PT_CACHE_PKL = "/workspace/_perm_tables_d5.pkl"
# Move notation order must match MOVE_LIST from move_detector (R-first)
_PT_MOVES = ['R', "R'", 'R2', 'U', "U'", 'U2', 'L', "L'", 'L2',
             'F', "F'", 'F2', 'D', "D'", 'D2', 'B', "B'", 'B2']
# move SUFFIX -> amount index (cw / ccw / 180) for the motion-direction prior
_AMT_IDX = {"": 0, "'": 1, "2": 2}


def _load_perm_tables_npz(path):
    """Load _PERM_TABLES from a .npz file (faster than pickle).
    Each depth d stores:
      perms_d{d}: int8 (N, 54)  -- cumulative perms including depths 0..d
      path_idx_d{d}: int8 (N, max(d,1))  -- move indices, -1 = pad/unused
    """
    npz = np.load(path, allow_pickle=False)
    pt = {}
    for d in range(6):
        perms = npz[f'perms_d{d}']        # int8 (N, 54)
        path_idx = npz[f'path_idx_d{d}']  # int8 (N, ncols)
        paths = []
        for row in path_idx:
            p = [_PT_MOVES[int(idx)] for idx in row if idx != -1]
            paths.append(p)
        pt[d] = (perms, paths)
    return pt


def _save_perm_tables_npz(pt, path):
    """Save _PERM_TABLES to .npz atomically."""
    arrays = {'moves': np.array(_PT_MOVES)}
    for d in sorted(pt.keys()):
        perms_arr, paths_list = pt[d]
        arrays[f'perms_d{d}'] = perms_arr.astype(np.int8)
        N = len(paths_list)
        ncols = max(d, 1)
        path_idx = np.full((N, ncols), -1, dtype=np.int8)
        if d > 0:
            _m2i = {m: i for i, m in enumerate(_PT_MOVES)}
            for i, path in enumerate(paths_list):
                for j, move in enumerate(path):
                    path_idx[i, j] = _m2i[move]
        arrays[f'path_idx_d{d}'] = path_idx
    import tempfile as _tmp
    dir_ = _os.path.dirname(path)
    fd, tmppath = _tmp.mkstemp(dir=dir_, suffix='.npz.tmp')
    _os.close(fd)
    try:
        np.savez_compressed(tmppath, **arrays)
        _os.rename(tmppath, path)
    except Exception:
        try:
            _os.unlink(tmppath)
        except Exception:
            pass
        raise


if _os.path.exists(_PT_CACHE_NPZ):
    try:
        _PERM_TABLES = _load_perm_tables_npz(_PT_CACHE_NPZ)
    except Exception:
        _PERM_TABLES = _build_perm_tables(5)
        try:
            _save_perm_tables_npz(_PERM_TABLES, _PT_CACHE_NPZ)
        except Exception:
            pass
elif _os.path.exists(_PT_CACHE_PKL):
    try:
        import pickle as _pickle
        with open(_PT_CACHE_PKL, 'rb') as _f:
            _PERM_TABLES = _pickle.load(_f)
        try:
            _save_perm_tables_npz(_PERM_TABLES, _PT_CACHE_NPZ)
        except Exception:
            pass
    except Exception:
        _PERM_TABLES = _build_perm_tables(5)
        try:
            _save_perm_tables_npz(_PERM_TABLES, _PT_CACHE_NPZ)
        except Exception:
            pass
else:
    _PERM_TABLES = _build_perm_tables(5)
    try:
        _save_perm_tables_npz(_PERM_TABLES, _PT_CACHE_NPZ)
    except Exception:
        pass
_PERM_LENS = {d: np.array([len(pp) for pp in _PERM_TABLES[d][1]], dtype=np.int32)
              for d in _PERM_TABLES}

# ---- rotation-event move-identity binding ---------------------------------
# A measured band-slide event names the moved LAYER in camera coords: the
# compatible cube moves are those whose changed cells on the observed face
# form exactly that band (under the warp's 4-fold rotation ambiguity).
_BAND_POS = {"row0": (0, 1, 2), "row1": (3, 4, 5), "row2": (6, 7, 8),
             "col0": (0, 3, 6), "col1": (1, 4, 7), "col2": (2, 5, 8)}


def _move_band_sets():
    """move name -> {face: frozenset(changed grid positions 0-8)}."""
    perms, paths = _PERM_TABLES[1]
    out = {}
    for i, pp in enumerate(paths):
        if len(pp) != 1:
            continue
        perm = perms[i]
        changed = {int(j) for j in range(54) if perm[j] != j}
        out[pp[0]] = {f: frozenset(p for p in range(9)
                                   if FACE_OFFSET[f] + p in changed)
                      for f in ALL_FACES}
    return out


_MOVE_BAND = _move_band_sets()
_COMPAT_CACHE = {}


def compatible_moves(face, band, rots=(0, 1, 2, 3)):
    """Moves compatible with a band-slide event on `face` under the given
    warp rotations (default: all 4 = uncalibrated, 12/18 moves; a calibrated
    single rotation shrinks to ~3). A face's own turn changes 8 cells, never
    a 3-cell band, so it is naturally excluded."""
    rots = tuple(rots)
    key = (face, band, rots)
    if key not in _COMPAT_CACHE:
        base = _BAND_POS[band]
        cands = {frozenset(GRID_ROTATIONS[r][p] for p in base)
                 for r in rots}
        _COMPAT_CACHE[key] = frozenset(
            m for m, fmap in _MOVE_BAND.items() if fmap.get(face) in cands)
    return _COMPAT_CACHE[key]


def calibrate_warp_rotations(events, moves, move_frames, oms, window=10):
    """Per-slot warp rotation from events near CONFIDENTLY-tracked moves:
    for an event (slot, band) within `window` frames of an emitted move m
    under orient map om, vote for rotations r where the band's cells under r
    equal m's changed cells on face om[slot]. Majority with margin >= 1 wins;
    slots without a clear majority stay uncalibrated (all-4). GT-free —
    pass 1's own lineage is the teacher."""
    votes = {}
    for f, slot, band in events:
        for i, mf in enumerate(move_frames or []):
            if abs(mf - f) > window or i >= len(moves):
                continue
            om = oms[i] if i < len(oms) else None
            if om is None:
                continue
            fmap = dict(zip(("up", "front", "right"), om)) \
                if not isinstance(om, dict) else om
            F = fmap.get(slot)
            changed = _MOVE_BAND.get(moves[i], {}).get(F)
            if not changed:
                continue
            base = _BAND_POS[band]
            for r in range(4):
                if frozenset(GRID_ROTATIONS[r][p] for p in base) == changed:
                    votes.setdefault(slot, {}).setdefault(r, 0)
                    votes[slot][r] += 1
    cal = {}
    for slot, vr in votes.items():
        ranked = sorted(vr.items(), key=lambda kv: -kv[1])
        if ranked[0][1] >= 2 and (len(ranked) == 1
                                  or ranked[0][1] > ranked[1][1]):
            cal[slot] = ranked[0][0]
    return cal


# Spatial-slot mapping: a face-step taught at the `up` spatial slot constrains
# the `up` slot's warp rotation; the band-event probe (layer_rotation_probe) and
# the ritual calibrator BOTH name slots by image position (topmost -> up, second
# -> front), and bind_pen keys rot_bind_rots by that same spatial slot. So the
# ritual's slot label IS the tracker's slot key directly — no remap (the prior
# falsified self-calibration's failure was sparse INFERRED votes, never a frame
# mismatch). The model face the lineage carries at that slot is supplied at
# consume time via om.get(slot); the warp rotation r is a property of the slot's
# grid projection alone, independent of which face/move taught it.


def rot_bind_from_ritual(profile, ref_om=None):
    """EXACT per-slot warp rotation {spatial_slot -> r} from a ritual profile.

    The ground-truth analog of calibrate_warp_rotations: each prescribed
    CLOCKWISE turn of a physical face (R/U/F/L/D) was OBSERVED to slide a known
    3-cell `band` at a known spatial `slot`. A face turn slides a band on every
    ADJACENT face; the slot's warp rotation r is the grid rotation under which
    the observed band's cells map onto that move's changed-cell band on the model
    face sitting at the slot. This is calibrate_warp_rotations' inner vote, but
    driven by the prescribed turn (ground truth) rather than an inferred pass-1
    move, so the answer is COMPLETE and unambiguous for every taught slot — not a
    sparse, ambiguity-diluted pass-1 majority.

    ref_om: the slot->model-face map the cube was held in during the ritual
    ("your solving grip"). Defaults to identity {up:up, front:front,
    right:right}; the prescribed face letter X names the model face X under
    identity (the canonical reference frame the tracker's om timeline is relative
    to). The model face at the OBSERVING slot is ref_om[slot]; the move that slid
    the band is the prescribed physical face renamed into the model frame
    (rename via ref_om). r is then the rotation taking the observed band to that
    move's changed band on ref_om[slot] — a single rigid value per slot for a
    fixed grip, regardless of which step taught it.

    Returns {slot -> r}. Rotation steps (y/x) reorient the whole cube (no single
    layer band) and are not used for face-step binding.
    """
    if not profile:
        return {}
    ref_om = ref_om or {"up": "up", "front": "front", "right": "right"}
    full = _full_om(ref_om)
    # The prescribed step letter X turns the spatial slot X (R = the right slot);
    # under ref_om that slot holds model face full[X], and the canonical move that
    # turns THAT model face clockwise is _FACE_LETTER[model_face]. Under identity
    # this is the identity map (R -> move R); a rotated grip renames it.
    votes = {}                          # slot -> {r: count}
    for step, res in profile.items():
        if not isinstance(res, dict) or res.get("kind") == "rotation":
            continue                    # _meta / rotation patterns skip here
        slot = res.get("slot")
        band = res.get("band")
        moved_slot = _LETTER_FACE.get(step)          # spatial slot the letter turns
        if (slot is None or band not in _BAND_POS or slot not in full
                or moved_slot is None or moved_slot not in full):
            continue
        model_face_at_slot = full[slot]              # face the OBSERVING slot holds
        moved_model_face = full[moved_slot]          # face the turned slot holds
        canonical_move = _FACE_LETTER[moved_model_face]   # canonical move name
        # the canonical move's changed band on the observing slot's model face;
        # the band slides on adjacent faces only (own face = 8-cell, no band).
        changed = _MOVE_BAND.get(canonical_move, {}).get(model_face_at_slot)
        if not changed or len(changed) != 3:
            continue
        base = _BAND_POS[band]
        for r in range(4):
            if frozenset(GRID_ROTATIONS[r][p] for p in base) == changed:
                votes.setdefault(slot, {}).setdefault(r, 0)
                votes[slot][r] += 1
    cal = {}
    for slot, vr in votes.items():
        ranked = sorted(vr.items(), key=lambda kv: -kv[1])
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            cal[slot] = ranked[0][0]
    return cal


# --------------------------------------------------------------------------- #
# Ergonomic bigram move prior: physically-awkward ball paths ("D' R U2 R2") cost
# more than natural trigger flows ("R U R' U'"), discriminating decoys when reads
# are sparse. Weights are in the SOLVER's frame: the tracked oms are CAMERA-frame
# (the camera films the cube's BACK — see reconstruct_with_rotations), so each
# cube-frame move is renamed to its camera slot via rename_to_viewer and then
# mirrored through the vertical axis (front<->back, left<->right; the solver faces
# the camera) into the solver's hands. NON-NEGATIVE soft penalties only — bonuses
# would double-count with the length penalty lam.
# --------------------------------------------------------------------------- #
_GATE_DEBUG = bool(__import__("os").environ.get("CUBED_GATE_DEBUG"))

_ERGO_MOVE_W = {"U": 0.0, "R": 0.0, "F": 0.0, "L": 0.0, "D": 0.5, "B": 0.7}
_CAM_TO_SOLVER = {"U": "U", "D": "D", "F": "B", "B": "F", "L": "R", "R": "L"}
_ERGO_PEN: dict = {}   # (depth, om_key) -> float32 array aligned with _PERM_TABLES rows


def _ergo_table(depth: int, omk, om=None) -> np.ndarray:
    """Per-row ergonomic penalty for the depth-`depth` ball under camera om `omk`:
    sum of per-move face weights (R2/U2 etc. weigh as their face) + adjacent-bigram
    weights (same solver face +0.6, D<->B adjacency +0.4), each cube-frame move
    renamed into the solver frame first. Lazy module cache: at most
    n_depths x 24 oms tables ever built (constant memory/work, ~rows x depth ops)."""
    key = (depth, omk)
    tbl = _ERGO_PEN.get(key)
    if tbl is not None:
        return tbl
    if om is None:
        om = next((o for o in ORIENTATIONS if _om_key(o) == omk), None)
    _, ppaths = _PERM_TABLES[depth]
    pens = np.zeros(len(ppaths), dtype=np.float32)
    if om is not None:                       # unknown om -> zero penalty (soft)
        solver = {f: _CAM_TO_SOLVER[rename_to_viewer(f, om)[0]] for f in "UDFBLR"}
        for i, path in enumerate(ppaths):
            p, prev = 0.0, None
            for mv in path:
                f = solver[mv[0]]
                p += _ERGO_MOVE_W[f]
                if prev is not None:
                    if f == prev:
                        p += 0.6
                    elif (f == "D" and prev == "B") or (f == "B" and prev == "D"):
                        p += 0.4
                prev = f
            pens[i] = p
    _ERGO_PEN[key] = pens
    return pens


def ball_states(arr: np.ndarray, depth: int):
    """Vectorized ball: (states_matrix (N,54), paths list). states[i] = arr[PERMS[i]]."""
    perms, paths = _PERM_TABLES[min(depth, 4)]
    return arr[perms], paths


def ball_paths(arr: np.ndarray, depth: int) -> dict:
    """{state_bytes: (state_array, shortest_move_path)} reachable within `depth`
    moves of `arr` (depth 0 = stay included)."""
    seen = {arr.tobytes(): (arr, [])}
    frontier = [(arr, [])]
    for _ in range(depth):
        nxt = []
        for a, p in frontier:
            for notation, perm in MOVE_LIST:
                na = a[perm]
                k = na.tobytes()
                if k not in seen:
                    e = p + [notation]
                    seen[k] = (na, e)
                    nxt.append((na, e))
        frontier = nxt
    return seen


def alignment_break_episodes(align_feat, frame_lo, frame_hi, *,
                             threshold, min_run):
    """Return maximal sustained lattice-break intervals in ``[lo, hi]``.

    This is the alignment gate's existing episode primitive, exposed once so
    terminal-rest sampling and scrub's optional visual transition lattice cannot
    drift.  Missing classifier frames remain aligned (probability 1.0), the
    threshold comparison remains strict, and ``min_run`` is the gate's existing
    sustained-run requirement.
    """
    lo, hi = int(frame_lo), int(frame_hi)
    need = max(1, int(min_run))
    if not align_feat or hi < lo:
        return ()
    episodes = []
    run_lo = None
    for frame in range(lo, hi + 1):
        broken = float(align_feat.get(frame, 1.0)) < float(threshold)
        if broken and run_lo is None:
            run_lo = frame
        elif not broken and run_lo is not None:
            if frame - run_lo >= need:
                episodes.append((int(run_lo), int(frame - 1)))
            run_lo = None
    if run_lo is not None and hi - run_lo + 1 >= need:
        episodes.append((int(run_lo), int(hi)))
    return tuple(episodes)


def certified_alignment_transition_episodes(
        align_feat, meta, certified_spans, *, threshold, min_run):
    """Recover intervals only for spans that passed the ownership contract.

    ``certified_spans`` is populated exclusively when
    :meth:`_align_transition_post_sample` both proves one strictly interior
    break and installs its admitted terminal-rest sample.  Recomputing that
    span's unique interval here avoids carrying raw reads while ensuring the
    scrub lattice can never consume the many unowned global classifier dips.
    """
    eligible = []
    for span_i in sorted({int(index) for index in (certified_spans or ())}):
        if span_i < 0 or span_i >= len(meta):
            continue
        lo, hi = int(meta[span_i][0]), int(meta[span_i][1])
        runs = alignment_break_episodes(
            align_feat, lo, hi, threshold=threshold, min_run=min_run)
        runs = [(a, b) for a, b in runs if lo < a and b < hi]
        if len(runs) == 1:
            eligible.append(tuple(map(int, runs[0])))
    return tuple(eligible)


def scrub_read_stream(span, sub_frames, sub_reads, dense=False, *,
                      align_feat=None, align_thr=None, align_min_run=None,
                      minspan=1, fallback_reason=None):
    """Build scrub's optional dense *scoring* stream plus provenance.

    ``sub_frames``/``sub_reads`` are the tracker's finalized, state-owned
    control stream.  With dense mode off they are returned by identity so the
    historical scrub path receives the exact same objects.  Dense mode may
    expand only inside the final alignment-stable epoch of ``span``: a
    sustained broken-lattice run uses align-gate's existing threshold and run
    length, and every observation through its final frame is excluded.  This
    prevents a dropped real turn inside one span from mixing pre- and
    post-transition cube states in scrub's chronological likelihood.

    Callers that applied a transform whose raw-frame state ownership is no
    longer one-to-one (consensus, lattice filtering, or a folded multi-state
    split) pass ``fallback_reason``.  That fail-closed path keeps the finalized
    control stream rather than resurrecting filtered or pre-transition reads.
    No new observation, threshold, or tag-specific boundary is introduced.
    """
    provenance = {
        "enabled": bool(dense),
        "control_count": int(len(sub_reads)),
        "evidence_count": int(len(sub_reads)),
        "mode": "control",
        "cut_after_frame": None,
        "fallback_reason": None,
    }
    if not dense:
        return sub_frames, sub_reads, provenance
    if fallback_reason:
        provenance["fallback_reason"] = str(fallback_reason)
        return sub_frames, sub_reads, provenance
    if not align_feat or align_thr is None or align_min_run is None:
        provenance["fallback_reason"] = "alignment-evidence-unavailable"
        return sub_frames, sub_reads, provenance

    ordered = sorted(span, key=lambda item: int(item[0]))
    if not ordered:
        provenance["fallback_reason"] = "empty-span"
        return sub_frames, sub_reads, provenance
    lo, hi = int(ordered[0][0]), int(ordered[-1][0])
    runs = alignment_break_episodes(
        align_feat, lo, hi, threshold=align_thr, min_run=align_min_run)

    cut_after = int(runs[-1][1]) if runs else None
    eligible = ([item for item in ordered if int(item[0]) > cut_after]
                if cut_after is not None else ordered)
    if len(eligible) < int(minspan):
        provenance.update(
            cut_after_frame=cut_after,
            fallback_reason="terminal-epoch-below-minspan")
        return sub_frames, sub_reads, provenance

    frames = [int(item[0]) for item in eligible]
    reads = [item[1] for item in eligible]
    provenance.update(
        evidence_count=int(len(reads)),
        mode=("terminal-aligned-epoch" if cut_after is not None
              else "whole-aligned-span"),
        cut_after_frame=cut_after)
    return frames, reads, provenance


class TrellisTracker:
    """Span-trellis tracker over per-span segment scorers.

    seg_factory: callable(read) -> scorer with .ok and .score_states(states)->(N,)
                 (e.g. a calibrated AbsSegment bound to centroids + orientation).
    Defaults are defined by the canonical runtime profile.
    """

    def __init__(self, seg_factory, gapf=10, K=8, d=2, lam=0.5, countpen=0.6,
                 switchpen=2.0, dfinal=4, minspan=4, still=12.0,
                 minstk=6, minfaces=1, span_subsample=16,
                 allow_reorient=False, rotpen=5.0, initial_om=None):
        """seg_factory(read, om) -> scorer (.ok, .score_states). With
        allow_reorient=True the trellis carries an ORIENTATION LATENT: candidates
        are (state, om) pairs; an om switch costs `rotpen` (hysteresis — same logic
        that kills misalignment decoys) and consumes one motion burst (a rigid
        whole-cube rotation moves pixels but permutes no stickers)."""
        self.seg_factory = seg_factory
        self.gapf, self.K, self.d = gapf, K, d
        # DIAGNOSTIC probe knob: env-only override of the beam
        # width K, which is also the per-OM cap used by scrub's
        # `_cap_stateful_prefix_band` (beam_k=int(self.K), line ~7548).
        # Unset => byte-identical canonical behaviour (K stays as constructed).
        # Rides as env like CUBED_GPU_MIN_STATES: NOT a fenced flag, does not
        # enter the CFG stamp.  NOT a tuning lever — a retention probe.
        _k_env = _os.environ.get("CUBED_TRELLIS_K", "").strip()
        if _k_env:
            self.K = int(_k_env)
        self.lam, self.countpen, self.switchpen = lam, countpen, switchpen
        self.dfinal, self.minspan, self.still = dfinal, minspan, still
        self.minstk, self.minfaces = minstk, minfaces
        self.span_subsample = span_subsample
        self.allow_reorient = allow_reorient
        self.rotpen = rotpen
        self.initial_om = initial_om or {"up": "up", "front": "front", "right": "right"}
        # v2: the joint (state, om) beam needs more width than the state-only beam —
        # 24 orientation targets x ball states get pruned at K=8 before genuine
        # switches can prove themselves.
        self.K_reorient = 32
        # v3: gap-length-gated rotation penalty. A reorientation at deliberate pace is
        # pause-rotate-pause = a LONG no-read gap (~0.5s); turns leave short gaps.
        # Long gaps get rotpen_low (switch invited), short gaps the expensive base
        # rotpen (noise suppressed). rot_gap_frames = threshold (server: ~0.3s * fps).
        self.rotpen_low = 2.0
        self.rot_gap_frames = None   # None -> flat rotpen everywhere
        # IN-BURST CO-COMMIT (wide/slice decode).
        # OFF by default => bit-identical. When on, a SEPARATE homogeneous transition
        # pass offers the single-rotation neighbor oms at burst_used=0 (the rotation
        # rides the MOVE's burst: Rw = x.L is ONE motion, not two), cost `widepen`
        # (None => rotpen); its top om-SWITCHING candidates are UNIONED into the
        # layer (the om_rescue_extra pattern) so the main switch_targets stays empty
        # -> misfit-rescue still fires and _scored_vec stays homogeneous. The
        # faithful renderer folds the resulting (rotation, move) burst to a wide.
        self.co_commit = False
        self.widepen = None
        self.cocommit_extra = 6   # max co-commit switching candidates unioned per span
        # MISFIT-GATED OM ENUMERATION (fluid rotations): the long-gap gate assumes
        # pause-rotate-pause, but speedsolvers rotate WITHOUT pausing — the rotation
        # hides inside a short gap, the carried om goes stale, and every later span
        # fits poorly. When a freshly-scored span's best candidate read-fit (the
        # per-span 's' term meta records) falls below misfit_thr, that span's
        # transition is re-run ONCE with all-24 om switch targets at FULL rotpen
        # (evidence must pay for the switch) and the better layer kept. <=1 re-run
        # per span => <=2x worst-case per-span work, linear overall. OFF by default
        # (om_on_misfit=False => bit-identical behavior).
        self.om_on_misfit = False
        # Threshold on the per-span fit = mean AbsSegment score = NEGATIVE mean
        # chrominance-weighted LAB distance. Scale (measured): well-calibrated
        # correct reads land ~ -5..-10; stale-om/garbage fits run -20 and worse
        # (a true state under mediocre cal scored -25.9; the om-Viterbi uses -50
        # as its no-read floor). -15.0 splits the regimes; tune at runtime.
        self.misfit_thr = -15.0
        # MISFIT-GATED DEPTH RETRY (fluid triggers): a multi-move trigger in a
        # short gap leaves the truth OUTSIDE the shallow ball — the span's best
        # candidate fits poorly because no reachable state explains the reads
        # when several moves occur inside a short gap. When the best fit is below
        # misfit_thr, re-run the transition ONCE at depth 4 and accept the deep
        # layer only if its best fit IMPROVES by >= deep_fit_gain: a genuine
        # depth-starved gap explains the reads strictly better at depth, while a
        # merely-noisy span keeps the shallow layer. Global deepening admits
        # same-fit cheaper decoys and floods the beam.
        # OFF by default (deep_on_misfit=False => bit-identical behavior).
        self.deep_on_misfit = False
        self.deep_fit_gain = 2.0
        # OM-RESCUE UNION: the misfit om retry was all-or-nothing on the TOP
        # candidate — when a stay-decoy topped both runs the whole rescued layer
        # was discarded, which can drop a plausible orientation switch. With
        # om_rescue_extra > 0, a losing rescue
        # still contributes its top-N om-SWITCHING candidates to the layer
        # (union, layer grows to K_eff+N for that span only). 0 = off
        # (bit-identical legacy either/or).
        self.om_rescue_extra = 0
        # Prune the misfit-rescue om search: rank all 24 orientations by the
        # previous state's read-fit (centers are move-invariant so this picks the
        # cube's actual orientation) and run the expensive move-ball transition
        # for only the top-k. 0 = off (try all 24). ~3-6 suffices (a re-grip lands
        # on an adjacent orientation; the reads decisively name it).
        self.om_rescue_topk = 0
        # DEEP misfit-rescue ball (RE-ANCHOR lever, 0 = off = legacy span depth).
        # At a re-grip the cube reorients AND a move may interleave, so the TRUE
        # post-grip state can sit a few face-moves from the committed one,
        # beyond the normal d_gap=1-2 ball, so the
        # misfit-rescue searches all 24 oms but never GENERATES the true state.
        # Re-running the rescue at this depth lets it reach the true state, which
        # the reads then prefer decisively. This
        # re-anchors the lineage from the reads without needing the move-ball to
        # bridge from the (wrong) committed predecessor at shallow depth.
        self.om_rescue_deep = 0
        # COUNT-BOUND the deep re-anchor (default off => bit-identical). The deep
        # ball searches up to om_rescue_deep moves regardless of evidence, so on a
        # clean gap it can pick a 2-3 move decoy that out-fits noisy reads ->
        # phantom moves. When True, the deep ball depth is
        # clamped to exp_hi (the band/event-derived move-count upper bound for THIS
        # gap): a 3-move drift burst still gets depth 3, a 1-move clean gap gets
        # depth 1 -> the re-anchor fixes drift WITHOUT over-firing. Soft bound (a
        # depth ceiling, not a hard count), so a wrong burst count can't deadlock.
        self.deep_count_bound = False
        # BURST-RECOVER (two-sided count bound; default off => byte-identical).
        # deep_count_bound only clamps the per-span move-ball DOWN (to exp_hi, a
        # DURATION estimate) to kill over-firing. But a FAST multi-move flurry (an
        # F2L pair / OLL / PLL alg) collapses several turns into ONE motion-event
        # span: short gap => small exp_hi/dur_est => the ball depth (capped at 4)
        # is too shallow to even EXPRESS the N-move sequence (it's never among the
        # candidates) => the decode under-produces and cascades. burst_recover adds
        # the OTHER side: estimate the burst's move-count from motion CONTENT (the
        # flowrot z-vorticity peaks — read-independent, finer than the collapsed
        # event channel) and, at a high-content span, ALLOW exp_hi + the ball depth
        # to expand UP to that content count (capped at burst_max, the perm-table
        # ceiling) so the ball can GENERATE the flurry. Needs flow_rotmag wired.
        self.burst_recover = False
        self.burst_max = 5            # perm tables built to depth 5 (621k states)
        self.burst_min = 4            # only EXPAND when content >= this (a genuine
        #   multi-turn collapse). rotmag peaks have a ~2-3 noise floor per normal
        #   span, so a low trigger deepens every span and floods decoys; a high
        #   absolute count is the collapse signature.
        self.burst_thr_mult = 1.5     # rotmag peak threshold = mult * median|rotmag|
        self.burst_nms_w = 6          # NMS half-width (frames) for rotmag peaks
        self.flow_rotmag = None       # set by harness: dict {frame:int -> |z-vort|}
        self.flow_rot = None          # set by harness: dict {frame:int -> z-vort}
        self._fr_frames = None        # sorted frame array (lazily built)
        self._fr_mag = None           # rotmag aligned to _fr_frames
        self._fr_thr = None           # precomputed peak threshold
        self._burst_log = None        # set to [] to capture per-span estimates
        # Scope the DEEP re-anchor to confirmed re-grip spans (gyro/CV rotation
        # frames). None = apply at EVERY misfit span (expensive: the depth-4 ball
        # is ~46k states x oms; a full solve has many misfit spans). When set to a
        # frame set, the deep ball fires ONLY on spans overlapping a re-grip — the
        # only place the state jumps multiple moves — so cost stays bounded
        # (~#re-grips per solve, not #misfit-spans). Other misfit spans keep the
        # cheap shallow om-rescue. The frames come from the orientation channel
        # (gyro angular-speed peaks, or CV pose-change) that already locates them.
        self.regrip_frames = None
        # Verbose decode log (default off). When True, track() prints one flushed
        # line per span (index, frame range, winning om, fit, #moves) plus a
        # [REANCHOR] tag with before/after fit + accept reason whenever the deep
        # re-anchor fires — so a long run is watchable live (tail the log) instead
        # of opaque. Pure instrumentation; never alters the beam or output.
        self.log_reanchor = False
        # Decisive-reads override (transient-tilt robustness): if a misfit-rescue
        # finds a candidate whose SPAN read-fit beats the current best span-fit by
        # >= this margin, take it regardless of the accumulated rotation penalty
        # (a settled tilt the cumulative score would otherwise reject). Fires ONLY
        # on a clear disagreement, so it does not over-switch like a low rotpen.
        # 0 = off.
        self.om_decisive_gap = 0.0
        # CONFIDENT-READ STATE RE-ANCHOR (Mechanism A: PROMOTE). The STATE analog of
        # om_decisive_gap above (which re-anchors ORIENTATION on a decisive read
        # preference). The trellis stores ABSOLUTE state per span and backtracks
        # prev_key, so ONE wrong tie-break rewrites every later layer's state (the
        # 18->46 wrong-sticker cascade). When a span is a CONFIDENT ANCHOR and the
        # confident-read span-fit DECISIVELY prefers a CARRIED non-leader state over
        # the cum-leader, re-weight every carried entry's cum by the fit advantage
        # and re-sort -- re-ranking states the beam ALREADY holds (no new states,
        # backtrack chains intact; the correct lineage was alive but out-ranked).
        # Master flag defaults OFF => byte-identical. anchor_min_faces/_nstk are SOFT
        # coverage floors: PROMOTE need not RESOLVE the full state, so a PARTIAL read
        # the leader gets wrong is enough to re-rank -- relax them to fire on the
        # occluded mid-solve spans where cascades start; the score margin
        # (anchor_promote_gap), per-cell conf (anchor_cellconf_thr) and span length
        # carry the safety, not the sticker count.
        self.anchor_resync = False          # master (A); off => byte-identical
        self.anchor_min_faces = 3           # >=3 faces pin all 3 axes (soft floor)
        self.anchor_min_nstk = 20           # dense-read floor (soft)
        self.anchor_fit_thr = -8.0          # span fit inside the -5..-10 correct band
        self.anchor_cellconf_thr = 0.6      # mean per-cell _cellconf (NOT kpt_conf)
        self.anchor_min_frames = 8          # temporal floor = span length
        self.anchor_promote_gap = 3.0       # fit margin over leader to fire (decoy band 0.2-1.4)
        self.anchor_weight = 1.0            # cum re-weight = w * (fit_cand - fit_leader)
        self.log_anchor = False             # one flushed line per PROMOTE
        # Rotation cost for MISFIT-GATE rescues (None = self.rotpen, the legacy
        # full charge). The gate is double-gated by evidence — the span's best
        # fit collapsed below misfit_thr AND the switch candidate must win the
        # retry — unlike blind long-gap enumeration, so its switches can carry
        # a lower prior. Fluid rotations may occur inside short gaps, where the
        # full rotation penalty can otherwise overwhelm the read evidence.
        self.rotpen_misfit = None
        # sticker-count-weighted span votes (see weighted_seg_scores): the
        # 2-face reads are the only om anchors in the 1-face-dominant regime.
        self.seg_weight_n = False
        # SOFT ALIGNMENT FUSION (default None => bit-identical). Per-frame P(aligned)
        # from the (retrained) classifier weights each frame's READ contribution to
        # the span state-vote: a mid-turn/off-axis frame (low P(aligned)) is TRUSTED
        # LESS rather than either poisoning the vote (ungated) or being dropped (a
        # hard gate, which perturbs the brittle span structure and hurt the decode).
        # {frame: P(aligned)}; align_weight_floor keeps low-align frames contributing.
        self.align_weights = None
        self.align_weight_floor = 0.1
        # ROBUST PER-SPAN CONSENSUS (default off => bit-identical). Replace the brittle
        # top-N per-frame state vote with ONE denoised read per span: median LAB per
        # cell over ALL the span's frames (occlusion / mid-turn = minority outliers the
        # median rejects). More aligned frames -> cleaner consensus -> a DECISIVE state
        # that IMPROVES with data, killing the knife-edge that any reweight/gate tipped.
        self.consensus_reads = False
        self.consensus_centroids = None   # {color: lab} for mode-of-color consensus
        # color-step span splitting (see build_spans); 0 = off (gap-only).
        self.span_split = 0.0
        # band-residual channel (see _band_peaks/build_spans): band = {frame:
        # band excess} from a --bandrec extraction; band_split = peak
        # threshold, None/0 = off (legacy identical). This path remains
        # explicit opt-in because band selectivity can over-fire.
        self.band = {}
        self.band_split = None
        self._band_peaks_cache = None
        # change-point span splitting (see _cpd_split_span): a fused span
        # covering hidden moves has a STEP in the mean read vector; pooling
        # all reads on each side is sqrt(n)-robust where the falsified
        # consecutive-frame channels were not. cpd_split = minimum
        # chrominance-weighted step of the two side-means (same units as
        # _read_step); None/0 = off (legacy identical).
        self.cpd_split = None
        self._cpd_breaks = set()
        # misfit-gated two-pass splitting: pass 1 (vanilla) identifies spans
        # whose best fit < misfit_thr; the driver sets force_breaks to their
        # CPD step maxima and re-runs. NO move witness — staying across a
        # forced break is free, so a false break costs beam compute, not output
        # correctness.
        self.force_breaks = set()
        # two-pass span DROP: spans whose pass-1 best fit is severely misfit
        # (off-lattice regrip sections fit -22..-32 against EVERY state under
        # EVERY om — there is no state to read) contribute only noise votes;
        # dropping them converts their frames into a gap the move-count
        # machinery bridges. drop_spans = {(start_frame, end_frame), ...}.
        self.drop_spans = set()
        # measured layer-rotation events (layer_rotation_probe): frames where
        # a face band slid a coherent quarter-turn. This channel can split
        # spans and provide a move-count lower bound.
        self.rot_events = []
        # move-IDENTITY binding (see compatible_moves): rot_event_info =
        # [(frame, slot, band)]; transitions whose gap contains an event but
        # whose move sequence has NO compatible move pay rot_bind per such
        # event. Soft additive penalty — the falsification record says no
        # hard constraints. None = off.
        self.rot_bind = None
        self.rot_event_info = []
        self.rot_bind_rots = {}   # slot -> calibrated warp rotation
        self._bind_cache = {}
        # bridge force-moves: {frame: (move, ...)} — a transition whose gap
        # contains a forced frame is RESTRICTED to rows whose path equals the
        # forced sequence (oracle/bridge-search injection). Empty = off.
        self.force_gap_moves = {}
        # PER-OM BEAM QUOTA: a plausible rotation lineage can be a minority
        # orientation in a sparse, low-discrimination span, so majority-
        # orientation decoys may fill a global top-K. With om_beam_quota=q,
        # the layer additionally keeps the top-q candidates of EVERY om present
        # in the transition — the rescued om's best cells survive on rank
        # WITHIN their orientation. Layer grows by <= 24q cells (linear,
        # bounded). 0 = off (bit-identical).
        self.om_beam_quota = 0
        self.contrast_margin = 1.0   # near-tie window for contrastive arbitration
        self.countpen_deep = 2.0     # countpen multiplier on deep (trigger) gaps
        # ROBUST MOVE-GATE (default 1.0 => bit-identical). Scales ONLY the
        # under-count penalty (cp on len < expected) — the term that FORCES a
        # phantom move when a re-grip burst inflates a gap's motion-burst count.
        # <1.0 relaxes that pressure so move-commit is driven by the read-fit
        # state-change margin (move = state change) rather than the burst count.
        self.move_gate_under = 1.0
        # CLEAN-REST GATING (default off => bit-identical). When clean_gate is True
        # the relaxed countpen / deepened ball-depth apply only to spans whose
        # PRECEDING anchor span fit is decisive (>= clean_fit_thr); non-clean
        # (re-grip / occluded) spans fall back to the strict legacy countpen
        # (clean_strict_countpen) and ball depth (clean_strict_d), so a looser
        # global setting cannot admit multi-move decoys at low-observability
        # re-grips. The per-span anchor fit is the predecessor span's recorded
        # best read-fit (meta), available before this span is scored.
        self.clean_gate = False
        self.clean_fit_thr = -10.0
        self.clean_strict_countpen = 0.6
        self.clean_strict_d = 2
        self.clean_max_gap = 10**9   # relaxed count only on gaps <= this (frames);
        #   default = no restriction (gate keys on fit alone). Set small (~ a few
        #   frames) to scope the relaxation to fast-adjacent bursts vs long re-grips.
        # CO-OPERATIVE MOVE-GATE (default off => bit-identical). The move detector
        # (CNN/ramps) over-fires on re-grips/reorientations; a spurious event both
        # SPLITS the span (build_spans) and WITNESSES a move (exp_lo), so both
        # segmentation and the count floor must be corrected. track_2pass()
        # fixes both: pass 1 records, per event-gap, the reads move-vs-re-grip margin
        # (_move_gate_margin: best 1..2-move read-fit minus the stay read-fit, over the
        # candidate orientations); events whose gap margin < move_gate_margin changed
        # NO cube state => re-grips => dropped from rot_events ENTIRELY (no split, no
        # witness); pass 2 decodes the cleaned set (the reads-sourced equivalent of
        # feeding only the true moves). move_gate_log collects pass 1's per-gap margins.
        self.move_gate = False
        self.move_gate_margin = 3.0
        # COVERAGE-ADMISSIBLE GATE (None = off, byte-identical — EXPERIMENTAL,
        # lever): when set, a pass-2 COLOR re-grip verdict is
        # trusted only if the pass-1 gap's visible∩changed coverage
        # (_gate_coverage) >= this floor; below it the margin was computed on
        # cells that could not see the candidate moves' changes (vacuous), so
        # the color channel ABSTAINS and the event survives unless alignment
        # vetoes it on its own evidence. Set by trellis_gt --gate-cov-min.
        self.gate_cov_min = None
        self.move_gate_log = None
        # DENOISED MARGIN (default off => bit-identical). _move_gate_margin is
        # measured over the 16-frame span SUBSAMPLE, so a noisy sample makes the
        # move-vs-re-grip verdict brittle. Scoring it over all settled
        # frames averages the read noise out: a REAL state change clears the bar on
        # every frame, a noisy misread does not, so the verdict (and one threshold)
        # becomes solve-invariant -- the "a noisy read must not commit a move" guard.
        self.move_gate_allframes = False
        self.move_gate_max_frames = 80    # quality-ranked cap when allframes (0=all)
        # CO-OPERATIVE TRELLIS<->MOTION GATE (default off => byte-identical). A
        # SINGLE-PASS, reads-AWARE veto on phantom move insertion. The
        # count/ball/deep-re-anchor
        # machinery can FORCE a move at a span (a motion burst raises the move
        # floor exp_lo, so the count penalty cp*outside sinks the (stay)
        # candidate's provisional below a move's -- the phantom; a re-grip read as
        # a turn) even when the reads show NO state change. This veto fires AT THE
        # COMMIT POINT (where the layer leader / committed move is fixed, after all
        # rescue / retry / co-commit / anchor passes -- so it has the final per-
        # candidate read-fit; in _scored_vec `pen` is computed BEFORE the read-fit,
        # so the read-vs-stay decision must be made HERE, on x[5], not inside the
        # provisional): when the top candidate would COMMIT a move but a carried
        # (stay) candidate's read-fit DECISIVELY beats the best move candidate's
        # read-fit -- read_fit(stay) - read_fit(best_move) >= coop_gate_margin --
        # the layer keeps ONLY the (stay) states (the count floor's under-count
        # penalty refunded so the carried cum is honest), so no phantom move is
        # inserted. Reads-aware, NOT the failed global move_gate_under knob; the
        # conservative margin protects genuine (read-supported) moves. Margin is on
        # the AbsSegment read-fit scale (mean -LAB distance, ~ -25..-40; real
        # move-vs-stay separations run ~1+).
        self.coop_gate = False
        self.coop_gate_margin = 1.0
        self.coop_gate_log = False
        # ALIGNMENT MOVE-GATE (default off => bit-identical). A layer MOVE shears a
        # layer off the lattice -> a SUSTAINED unaligned stretch; a rigid re-grip
        # keeps every layer square -> the lattice never breaks. So the longest run
        # of unaligned frames in an event's gap is an ORTHOGONAL (geometry, not
        # color) move-vs-re-grip signal -- it catches the re-grips color misses
        # (occlusion changes the visible faces but the lattice stays intact), and
        # gets sharper as the alignment classifier improves. Fused with the color
        # margin per align_gate_mode:
        #   "or"   : drop if color OR alignment says re-grip (aggressive; catches
        #            occlusion re-grips color kept)
        #   "and"  : drop only if BOTH say re-grip (conservative; alignment rescues
        #            an occluded move color wrongly dropped)
        #   "align": alignment decides alone (the pure lattice model)
        # align_feat = {frame: P(aligned)} is set by the driver.
        self.align_feat = None
        self.align_gate = False
        self.align_min_run = 2       # event-window longest-unaligned-run < this = re-grip
        self.align_thr = 0.3         # P(aligned) < this = a broken-lattice frame
        self.align_gate_mode = "or"
        self.align_window = 8        # +/- frames around the gap's event(s) to scan
        #   for the lattice-break. The move's unaligned dip sits AROUND the event
        #   (the probe measured +/-8), not inside the narrow no-read gap, so scan a
        #   window centred on the event frames (which align with align_feat).
        # ADAPTIVE WINDOW (default off => fixed +/-align_window). Instead of a fixed
        # width, find the unaligned run each event sits in and expand it to its full
        # extent -- fps/speed-invariant, no hardcoded frame count. align_search_r
        # bounds the outward search for the dip (absorbs event-localisation error).
        self.align_expand = False
        self.align_search_r = 12
        # TRANSITION-AWARE TERMINAL-REST SAMPLING.  A sustained alignment break
        # inside an otherwise settled span is direct evidence that the span
        # crosses a layer transition.  The state produced by that transition
        # must be scored from reads AFTER the final break, never from a cleaner
        # pre-transition rest.  This is deliberately separate from the failed
        # global time-stratified sampler: spans with no classifier-confirmed
        # transition retain the exact historical quality-ranked sample.  It is
        # pass-2/final-only so gate measurement and the span lattice cannot be
        # reshaped.  Canonical profiles enable it alongside align_gate.
        self.align_transition_sample = False
        self._align_transition_fires = 0
        self._align_transition_spans = set()
        # gm-PROTECT (default None => off). Alignment may NOT drop an event whose
        # COLOR margin gm >= align_gm_protect (color is CONFIDENT it is a real move).
        # Color shields confident moves that alignment might otherwise veto
        # in "or" mode, while alignment still drops ambiguous
        # moves while alignment still drops the ambiguous ones.
        self.align_gm_protect = None
        # ADAPTIVE gm-protect (default 0 = protect EVERY high-gm move = the global
        # form). Set to align_min_run-1 (e.g. 1) to protect ONLY where the alignment
        # evidence is BORDERLINE (a dip was almost there, ab >= this) and NOT a
        # STRONG no-dip (ab=0 = a clean re-grip a good classifier nailed). With the
        # retrained classifier this keeps clear re-grip drops while rescuing
        # borderline move over-drops.
        self.align_gm_protect_ab_min = 0
        # SOFT WITNESS (break the cliff, default off => bit-identical). Instead of the
        # hard event move-count floor (lo = #events -> over-firing FORCES phantoms),
        # each event contributes a FRACTIONAL weight w = clip((margin-LO)/(HI-LO),0,1)
        # to lo (margin = the reads move-vs-re-grip signal). LO/HI are GLOBAL (one
        # value across solves, NOT per-solve), so a clear re-grip (margin<=LO) deflates
        # the floor while a clear move (margin>=HI) keeps ~1 -> the cliff softens
        # (over-firing no longer forces phantoms; reads arbitrate the count). The gap
        # soft floors are built by track_2pass pass 1 into _soft_lo_map[(a,b)].
        self.soft_witness = False
        self.soft_lo_range = (0.0, 4.0)
        self._soft_lo_map = {}
        self.fps = 60.0              # for duration-interval move bounds
        # v2: seed layer 0 with the top-M initial-orientation hypotheses scored
        # against the opening span; the trellis resolves which one the data
        # supports.
        self.init_om_hyps = 4
        # ergonomic bigram move prior (see _ergo_table): weight on the soft
        # solver-frame awkwardness penalty of candidate ball paths. 0.0 = OFF
        # (table lookup skipped entirely; bit-identical to prior behavior).
        self.ergopen = 0.0
        # OLL/PLL alg-dictionary PROGRESS PRIOR (detect/ll_alg_prior.py): once the
        # tracked state reaches F2L-complete (confidence-gated), a soft per-row bonus
        # ll_prior * progress(candidate_state) pulls the beam along a recognized
        # OLL->PLL solution through the fast last-layer flurry where reads tie at ~0.
        # 0.0 = OFF (helper never imported; bit-identical to prior behavior).
        self.ll_prior = 0.0
        self.ll_gate_fit = None     # arm only when top read-fit >= this (None = on F2L-complete alone)
        self.ll_cap = 14            # max alg variants per OLL/PLL case
        self._ll_prog = None        # {abs_state_bytes: progress in (0,1]} once armed
        self._ll_info = None        # arming diagnostics for the log
        # OLL/PLL ALG INJECTION (the EXPRESSION half; default off => byte-identical).
        # ll_prior is a SOFT prior over a depth-capped ball — it can only reward
        # candidates the ball already GENERATED, so a 7-12 move OLL/PLL flurry that
        # collapsed into one span is never expressed (the terminal bridge can't reach
        # solved within dfinal). With ll_inject, when the table arms + recognizes the
        # case, capture the explicit recognized alg as an absolute-frame move path and
        # SUBSTITUTE it for the drifted last-layer tail at the terminal — but ONLY when
        # the forward decode failed to reach the solved endpoint (reach=False), so a
        # solve whose last layer already decodes (reach=True) is left untouched.
        self.ll_inject = False
        self._ll_inject_at = None    # (arm_layer_idx, arm_key, abs_alg_moves) once armed
        self._ll_inject_fired = False
        # MOTION-DIRECTION AMOUNT PRIOR (read-independent robustness). A per-event
        # distribution over the move AMOUNT (cw / ccw / 180) from the motion-direction
        # model. Each span's candidate paths get a
        # bonus dir_prior * sum_k logP(amount(move_k) | event_k), so a move whose amount
        # CONTRADICTS the motion is penalized — a constraint color/lighting
        # corruption cannot touch. dir_prior=0 or dir_pred None => OFF,
        # bit-identical. dir_pred maps
        # {event_frame:int -> [logP_cw, logP_ccw, logP_180]} (events = self.rot_events).
        self.dir_prior = 0.0
        self.dir_pred = None
        # ROBUST ARMING (read-robustness): the ll-prior arms ONLY when the top beam
        # state is EXACTLY F2L-complete (f2l_done_color) — a binary cliff. A single
        # read-driven 1-move drift in F2L misses it, the LL bridge never arms, and the
        # score can collapse. robust_arm>0 also arms when the top state is within
        # robust_arm moves of an F2L-complete state (the nearest UNIQUELY-crossed
        # neighbor), turning the cliff into a ramp. 0 = OFF (bit-identical).
        self.robust_arm = 0
        # Neighbor-arming is a LATE FALLBACK, never a preemption: only allowed once
        # the solve is past this fraction of its spans, so it cannot jump ahead
        # of a later exact F2L-complete arming.
        # Exact arming is NEVER gated; the LL flurry the prior helps is at the end.
        self.robust_arm_after_frac = 0.7
        # vectorized span transition + terminal bridge (numpy dedup/argsort in
        # place of the per-row Python dict loop). Exact same
        # output as the legacy loop — same candidates, bit-identical float
        # expressions, same tie-breaking. False = legacy reference path.
        self.vec = True
        # prov-bound branch-and-bound top-k pruning inside _scored_vec (see
        # _prune_score for the exactness argument): candidates are scored in
        # descending-provisional order and scoring stops once the k_top-th
        # best EXACT total provably dominates every remaining candidate
        # (evidence <= 0 structurally => total <= prov). OUTPUT-IDENTICAL to
        # the full pass for the returned tuples, including exact k-th-boundary
        # ties; auto-disabled whenever a _vec_fit_max
        # consumer (om_on_misfit / deep_on_misfit) is configured or any
        # active scorer lacks the nonnegative-distance fast path.
        # Default off: the provisional-score bound is too loose on long-gap
        # transitions. A useful bound would need a per-orientation evidence
        # upper bound, not the provisional score alone.
        self.prune_topk = False
        self.prune_block = 4096   # scoring block size (keeps the threaded
        #                           sum_seg_scores fan-out amortized)
        # DIAGNOSTIC om-trace hook (default None = OFF, zero cost). When set to a
        # list, track() appends one dict per span recording: the winning om, the
        # best span-fit achievable under EVERY orientation at that span's reads,
        # whether the carried beam still holds om0, and the winning entry's path.
        # Pure read-only instrumentation — never alters beam contents or output.
        self.om_trace = None
        # DIAGNOSTIC span-purity hook (default None = OFF, zero
        # cost / byte-identical). When set to a list, track() appends one
        # bimodality tuple per span whose first/second temporal halves prefer
        # DIFFERENT committed candidates (a straddled un-eventful move — the
        # measured still-gate hole). Pure read-only measurement — never alters
        # beam contents or output; the per-half re-scoring touches only the
        # span's sampled reads and runs ONLY when this attr is non-None.
        self.span_purity_log = None
        # SPAN-PURITY SPLIT behavior (default False = OFF,
        # byte-identical). When True, a span the ball-based straddle test finds
        # DECISIVELY impure (a real move with no event inside a settled span —
        # the still-gate hole) is re-run as TWO sub-spans split at the frame
        # median so the buried move is recovered. Armed independently of
        # --verbose: the purity computation runs when span_purity_log is not
        # None OR span_purity_split is True. Off => the whole block is skipped.
        self.span_purity_split = False
        # STRATIFIED MAIN-SPAN SAMPLE (lever L1, default False =
        # OFF, byte-identical). The main commit's span score is built from
        # `ranked[:span_subsample]` — the top-quality (_nstk) frames of the
        # span. proved that on a straddled span this quality ranking
        # samples ALL pre-move frames (a clean 3-face rest outranks the
        # post-move 2-face tail), so the commit never sees the post-move
        # majority evidence. When True, the MAIN sample uses the same
        # TIME-STRATIFIED construction already used by span_purity_split's
        # straddle test (_strat_span_sample below: split the span's frame
        # extent at its midpoint, top-quality half-budget from EACH half) —
        # no new constants, same _nstk rule, same total span_subsample budget.
        self.strat_span_sample = False
        # CONDITIONAL variant (lever (2), default
        # False = OFF, byte-identical). Bare strat_span_sample above is a
        # GLOBAL policy and was FALSIFIED (re-sampling EVERY span in
        # both passes perturbs pass-1's gate/span structure even when no
        # straddle exists. strat_span_sample_cond swaps
        # in the identical stratified sample (_strat_span_sample) but ONLY
        # for a span whose plain quality-ranked sample has COLLAPSED (all on
        # one temporal side of the span's own extent midpoint) AND whose empty
        # side is wide
        # enough to plausibly be hiding a real move (>=
        # _collapse_strat_threshold(), a TIME-derived constant, see
        # COLLAPSE_HIDE_S near _span_collapse below -- NOT a hand-picked
        # knob). CLI: `--strat-span-sample cond` (bare flag = the global
        # behavior above, unchanged).
        self.strat_span_sample_cond = False
        # count of spans where the cond trigger fired THIS decode --
        # track_2pass calls track() twice (pass 1 + pass 2) on the SAME
        # tracker instance, so this accumulates across both passes by
        # construction; the "count triggers" summary line's source.
        self._collapse_strat_triggers = 0
        # SPAN-COLLAPSE instrumentation (pre-registered next
        # step, default None = OFF, byte-identical / zero cost).
        # named refinement for L1 is CONDITIONAL stratification (trigger only
        # when the quality-ranked sample has temporally collapsed onto one
        # side of the span), but before building a trigger the collapse rate
        # must be measured per capture. When set to a list, track() appends one
        # dict per span
        # recording whether the MAIN span sample (the same `sub`
        # strat_span_sample would replace) landed entirely on one side of the
        # span's own frame-extent midpoint, plus the softer stats (sample
        # frame min/max/count, span extent, per-side fractions) so a later
        # softer collapse definition can be considered without a re-run. Pure
        # read-only measurement -- never alters beam contents, sampling, or
        # output; the check runs only when this attr is non-None (the
        # span-purity-ball probe's "log EVERY evaluation" idiom).
        self.span_collapse_log = None
        # COVT-ON-COMMIT instrumentation (next step,
        # default None = OFF, byte-identical / zero cost). The existing covT
        # reader (_gate_trust_argmax) is called ONLY at the move-gate
        # (~L4634), which scores a re-grip VERDICT and only fires when a
        # rotation event falls inside the gap. The move the span scorer commits
        # is never trust-scored, so the SPURIOUS_EXTRA phantom-over-commit
        # class (thin-margin half-turns) is un-instrumented. When set to a
        # list, track() appends one dict per span recording covT (the raw
        # pre-floor p_trust sum, via _gate_trust_argmax, UNCHANGED) for the
        # candidate the beam actually COMMITS (take[0]) against its own
        # predecessor state and its own resolved om -- plus the runner-up
        # (take[1]) when present, which falls out of the SAME segcache-cached
        # reads at no extra scoring cost. Pure read-only measurement -- never
        # alters beam contents or output; the check runs only when this attr
        # is non-None (the span-collapse-log "log EVERY span" idiom).
        self.covt_commit_log = None
        # WINDOWED-SCRUB decode (--scrub-decode; see
        # detect.scrub_decode). Sequential windowed re-derivation from the
        # app-given init; when ON it REPLACES the emitted decode (the committed
        # word is demoted to ONE candidate hypothesis per window -- :
        # no constraint, anchor state, or om may derive from committed states).
        # Default OFF => the block is skipped => byte-identical (S1).
        self.scrub_decode = False
        self.scrub_decode_log = None       # `.scrub.jsonl` sidecar path
        self.scrub_decode_tag = ""         # run id echoed into sidecar rows
        self._scrub_evals = 0              # decodes the scrub block ran
        self._scrub_report = {}            # last scrub report (trellis_gt print)
        self._scrub_perms = None            # lazily derived move permutations
        # Automatic dense-motion phase hypotheses.  Producers populate the
        # immutable typed audit + exact slot tuple; scrub admits them only as a
        # complete, bounded transaction.  No runtime flag or threshold exists.
        self.intraburst_phase_slots = ()
        self.intraburst_phase_audit = None
        # Stateful (state, OM) scrub tracking (--scrub-om-stateful).  This is a
        # separate, default-off actuator so the existing om-marginalized scrub
        # remains unchanged unless the explicitly stamped extra flag is passed.
        self.scrub_om_stateful = False
        # Dense exact-frame evidence for scrub only (--scrub-dense-reads).
        # The tracker/gate retain their historical top span_subsample reads;
        # keeping a separate stream avoids repeating the falsified global
        # stratified-sampling experiment. Default OFF => identical inputs.
        self.scrub_dense_reads = False
        # Exact-frame prefix candidate plane (--scrub-dense-prefix). It
        # requires scrub_dense_reads + stateful OM; every eligible window runs
        # an independent dense-evidence beam and unions its survivors with the
        # control survivors before common final scoring and decision.
        self.scrub_dense_prefix = False
        # Optional sustained-alignment transition slots for scrub.  Episodes
        # reuse align_gate's existing threshold/min-run semantics and supply
        # MOVE-or-SKIP structure only; they never alter the motion-count budget.
        # Default OFF keeps the historical scrub lattice byte-identical.
        self.scrub_visual_transition_slots = False
        # An authoritative alignment-only veto (color/state-change says MOVE)
        # can be interpreted by stateful scrub's unresolved-envelope rescue as
        # one optional layer MOVE or whole-cube re-grip. This remains a
        # separate default-off actuator.
        self.scrub_gate_drop_slots = False
        # Optional mid-motion rows enter the same canonical window score.
        self.midmotion_reads = False
        self.midmotion_rows = []
        # PASS-2-ONLY latch (.7(1)): track_2pass calls track twice; the
        # repair block must run only on the emitting pass. Default True so a
        # direct single-pass track() call is treated as final.
        self._in_final_pass = True
        # track_2pass pass 1 consumes only the forward loop's move_gate_log and
        # _fitnoise_vals.  This private latch is armed only around that internal
        # call; direct track() calls retain the full terminal/backtrack path.
        self._pass1_artifacts_only = False
        # TEACHER-FORCED per-move harness (H-1, default None = OFF,
        # byte-identical). When set to the sorted GT move timeline
        # [(frame, move), ...] (movegt + frame stamps), every span
        # transition starts from an ORACLE PREFIX instead of the carried
        # beam: state = the READ-ALIGNED GT replay state at the preceding
        # anchor (_tf_align_anchors — the alignment, not raw stamps, buckets
        # teacher moves per gap); orientation is chained through the production
        # machinery because the teacher timeline carries no orientation. The
        # span itself runs
        # the UNCHANGED production transition, and the commit point records
        # one per-gap verdict into tf_records: {gap, f_a, f_b, gt, gt_idx,
        # got, margin, omk, leader_omk, state_eq}. DIAGNOSTIC ONLY: the named
        # reach-LL evaluation is reported separately on a free-running result.
        self.teacher_forced = None
        self.tf_records = None
        # om-event prior (TASK-2 lever, default None = OFF, bit-identical):
        # penalize om SWITCHES in spans whose preceding gap holds NO rotation
        # event, and (optionally) discount switches that a nearby event supports.
        # Soft additive cost on the switch_targets om_pen at enumeration time.
        self.om_event_prior = None      # extra penalty added to UN-witnessed switches
        self.om_event_window = 0        # +/- frames around span for event support
        self.om_event_relief = 0.0      # subtract from witnessed switches' rotpen
        # LATTICE-CONSISTENCY PER-READ GATE (None = OFF, bit-identical). A span's
        # vote averages its reads' fits; the motion stillness gate admits reads
        # by bbox-motion alone, so a gentle wrist-flick MID-TURN read (the layer
        # caught partway through a quarter-turn = OFF the cube-state lattice) is
        # admitted and pollutes the vote even though it matches NO valid cube
        # state under ANY orientation. lattice_gate is a per-read floor: a read
        # whose BEST achievable fit over the span's candidate ball (the states
        # reachable from the live beam at this gap's depth) under ALL carried
        # orientations is below lattice_gate is DROPPED from that span's vote
        # (it is off-lattice — no settled state explains it). Computed once per
        # span before scoring (see _lattice_keep). SAFE ONLY WITH A GOOD ANCHOR:
        # the candidate ball is centered on the beam, so a diverged beam would
        # gate the wrong reads, so pair it with a fixed anchor.
        # A span is never emptied (the gate keeps >= minspan best reads even if
        # all are sub-threshold) so it can only RE-WEIGHT a vote, never delete a
        # span. Reach-gating is the caller's job (set it only when stage-1 fails).
        self.lattice_gate = None
        # diagnostic: per-span (frame_lo, frame_hi, n_in, n_kept) of the last
        # build, populated only when lattice_gate is set (read-only).
        self._lattice_log = []
        # OM TIMELINE (default None = OFF, bit-identical). A SOURCE-AGNOSTIC
        # per-frame camera-frame orientation {frame: om_key}: the cube's om is
        # GIVEN, not guessed from reads. An external CV pipeline may provide it.
        # When set, each span is FORCED to its timeline om (only switch target,
        # zero penalty; beam restricted to it) so the move-ball searches STATE
        # only. This is THE transient-tilt / re-grip fix: at a re-grip the reads
        # go off-lattice and the trellis cannot identify the post-grip om, but a
        # external orientation channel sees the reorientation directly. The
        # misfit-rescue is naturally skipped (switch_targets is non-empty).
        self.om_timeline = None
        # PER-FRAME OM SCORING (default False = bit-identical). om_timeline above
        # FORCES one MODE om per span and scores every frame under it; but a state
        # HELD ACROSS a mid-span re-grip spans 2-3 camera orientations, so pooling
        # its frames under one orientation can corrupt the state vote. With
        # om_per_frame=True a span's
        # state-vote scores EACH frame's read under ITS OWN timeline om
        # (om_timeline[frame]), fusing the vote across the orientations the span
        # covers. The beam still carries the span's MODE om (the (state,om) key,
        # rotation penalties, reconstruction) -- only the EVIDENCE term goes
        # per-frame, so om is OBSERVED not guessed (the 24-om/misfit machinery is
        # inert under a forced timeline). sum_seg_scores already groups segs by
        # their gather matrix, so heterogeneous per-frame oms sum correctly. No-op
        # unless om_timeline is set (its per-frame source). track() is shared with
        # legacy clips; they never set this -> byte-identical.
        self.om_per_frame = False
        # SOFT-FUSION per-frame orientation source (default False). Instead of
        # an external timeline, resolve each frame's orientation from the
        # READS themselves: the occlusion-tolerant 24-orientation match of the
        # visible pattern against the beam's anchor state (exactly the op om-rescue
        # uses per-span, applied per-frame). A confidence gate (om_pf_margin = min
        # best-vs-2nd fit margin) keeps only DECISIVE per-frame oms; the rest are
        # omitted so the scorer FALLS BACK to the candidate's span om. So a
        # wrong/occluded frame never forces a wrong om (unlike hard timeline
        # forcing) -> the fusion can only help re-grips, never regress below
        # baseline. Needs om_per_frame=True; composes with om-rescue (which sets
        # the robust base om from the fallback frames). _pf_om = the current span's
        # gated {frame: omk} map, refreshed per span in track().
        self.om_per_frame_from_reads = False
        self.om_pf_margin = 2.0      # min (best - 2nd) fit margin to TRUST a frame
        self.om_pf_min_frames = 0    # only resolve on spans of >= this frame extent
        # ---- Z-NORM DERIVED GATES (default OFF => byte-identical) ------------
        # De-knob the three raw-LAB-scale hand constants (misfit_thr,
        # move_gate_margin, om_pf_margin) by expressing them as z-scores against
        # the solve's OWN anchor fit-noise (mu, sigma), estimated from the
        # confident anchor spans the anchor_resync coverage test already locates
        # (>= anchor_min_faces / anchor_min_nstk, still, mean cell_conf >=
        # anchor_cellconf_thr, span length >= anchor_min_frames, fit inside the
        # -8 correct band). The raw LAB fit (negative chrominance-weighted LAB
        # distance to centroids) is the single most capture-dependent scale in
        # the decode -- it shifts with lighting/cube/camera, so a fixed -12 / 1.0
        # / 2.0 that was tuned on one read regime blows past on a junkier capture
        # (roadmap R1/R2/R3). A z-score vs the solve's
        # own read-noise floor transfers where the raw constant does not:
        #   misfit_thr      -> mu - k_misfit * sigma   (an absolute-fit floor)
        #   move_gate_margin -> k_move  * sigma         (a fit-DIFFERENCE margin)
        #   om_pf_margin     -> k_ompf  * sigma         (a fit-DIFFERENCE margin)
        # The k's are DIMENSIONLESS and calibrated on the dev tags so the derived
        # thresholds REPRODUCE the hand constants there (mu_dev - k_misfit*sig_dev
        # ~= -12; k_move*sig_dev ~= 1.0; k_ompf*sig_dev ~= 2.0) and ADAPT on new
        # capture styles. Cold start / no clean anchor (< znorm_min_anchors) =>
        # fall back to the raw constant (never worse than today). Env-overridable
        # so it can be driven at runtime without a CLI plumbing change.
        self.znorm_gates = _os.environ.get("CUBED_ZNORM_GATES", "") not in ("", "0")
        # Defaults preserve the legacy thresholds around the reference fit
        # scale while allowing per-capture adaptation. Override them through
        # the environment or the corresponding --znorm-k-* flags.
        self.znorm_k_misfit = float(_os.environ.get("CUBED_ZNORM_K_MISFIT", "-2.625"))
        self.znorm_k_move = float(_os.environ.get("CUBED_ZNORM_K_MOVE", "0.1913"))
        self.znorm_k_ompf = float(_os.environ.get("CUBED_ZNORM_K_OMPF", "0.3826"))
        self.znorm_min_anchors = int(_os.environ.get("CUBED_ZNORM_MIN_ANCHORS", "3"))
        # Fit-noise SAMPLER coverage (a CONFIDENT-read gate, NOT the anchor_resync
        # PROMOTE gate). Looser on face-count than PROMOTE (these captures are
        # 1-face-dominant, so faces>=3 samples nothing) and it must NOT gate on
        # the absolute LAB fit band (anchor_fit_thr) -- that absolute-band
        # assumption is exactly what z-norm removes. A confident read = a STILL,
        # well-covered (>= znorm_min_faces / znorm_min_nstk), clearly-seen (mean
        # cell_conf >= znorm_cellconf_thr) span; its best read-fit -- whatever it
        # scores under THIS capture -- is the sample.
        self.znorm_min_faces = int(_os.environ.get("CUBED_ZNORM_MIN_FACES", "2"))
        self.znorm_min_nstk = int(_os.environ.get("CUBED_ZNORM_MIN_NSTK", "12"))
        self.znorm_cellconf_thr = float(
            _os.environ.get("CUBED_ZNORM_CELLCONF", "0.6"))
        self.log_znorm = bool(_os.environ.get("CUBED_ZNORM_LOG", ""))
        # instrumentation: one line per per-frame om PIN decision (all
        # three _resolve_perframe_om sites -- GPU, CPU lazy, CPU plain), so
        # k_ompf can be derived offline from labeled data the way k_move was.
        # Default off = zero output, byte-identical decode (mirrors CUBED_ZNORM_LOG).
        self.log_ompf_pf = bool(_os.environ.get("CUBED_OMPF_LOG", ""))
        self._fitnoise_vals = []     # per-solve confident-anchor span fits (running)
        self._fitnoise_seed = None   # whole-solve estimate frozen from a prior pass
        # BALL-ANCHORED pf-om resolution depth (default None = stock single-state
        # path, byte-identical). Removing low-quality cells can expose wrong
        # per-frame orientation pins:
        # _resolve_perframe_om scores every om against the STALE previous anchor
        # (zero-move assumption), and cleaning junk cells can amplify fit margins
        # past om_pf_margin, so post-gap frames pin to
        # the om that best MISinterprets the old state. When om_pf_ball = d, each
        # om is scored by its MAX fit over the anchor's depth-d move ball
        # (_PERM_TABLES[d] — the states the beam can actually reach this span), so
        # an om is only pinned when it uniquely explains the read GIVEN move
        # freedom. Margin gate + fallback semantics UNCHANGED (ambiguous frames
        # still fall back to the span om).
        # om_pf_ball="gap" (GAP-DERIVED depth, the de-knobbed form): instead of a
        # fixed integer, the depth is derived per span from the decoder's OWN
        # move budget for that gap — min(ceil(exp_hi), d_gap):
        #   * exp_hi — the admissible move-count ceiling for the gap
        #     (_expected_interval duration bound, burst/event-widened), the
        #     same ceiling the count penalty charges against. A 0-move budget
        #     gives the depth-0 ball == the anchor alone == the stock check.
        #   * d_gap — the span's TRANSITION ball depth (the decoder's own
        #     depth-form of that budget: min(cap, max(eff_d, min(dur_est,
        # exp_hi), burst_hi))). The mechanism scores oms against "the
        #     states the beam can actually reach this span" — that is EXACTLY
        #     _PERM_TABLES[d_gap]; depth beyond it scores states the beam
        # cannot reach, and is also unshippable (measured:
        #     depth-5 ball = 621,649 states = 226 s/frame for the 24-om pass).
        # No new constant: both quantities already exist at the call site.
        # NOTE this is a SEMANTIC change vs a fixed depth on short gaps (a
        # smaller ball can pin MORE frames, the margin being measured over
        # fewer reachable states), so it ships as an explicit opt-in mode,
        # not a silent replacement of the canonical 3.
        self.om_pf_ball = None
        # {depth: n_spans_resolved_at_depth} — banner instrumentation so ball
        # depth usage is visible per run (set to {} by the driver when the ball
        # is on; None = off, zero cost, prod path untouched).
        self._pf_ball_depth_hist = None
        # [frames, full_scores] — lazy-evaluation visibility: how many of the
        # 24 oms actually needed a full ball pass (driver sets to [0, 0]).
        self._pf_ball_scored = None
        self._pf_om = {}             # transient {frame: omk} for the current span
    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _nstk(read, min_l):
        return sum(1 for _, l9 in read for l in l9 if l[0] >= min_l)

    @staticmethod
    def _nfaces(read, min_l):
        return sum(1 for _, l9 in read if any(l[0] >= min_l for l in l9))

    @staticmethod
    def _read_step(r1, r2, min_l):
        """Chrominance-weighted mean LAB step between two reads over shared
        (face, pos) stickers; None if <4 shared. A true hold steps ~2-5 (sensor
        noise); a face turn between the reads steps the changed stickers by a
        full color distance (~30-60) -> mean ~10+."""
        d2 = {}
        for face, lab9 in r2:
            for p in range(9):
                if lab9[p][0] >= min_l:
                    d2[(face, p)] = lab9[p]
        diffs = []
        for face, lab9 in r1:
            for p in range(9):
                o = d2.get((face, p))
                if o is None or lab9[p][0] < min_l:
                    continue
                dl = lab9[p][0] - o[0]
                da = lab9[p][1] - o[1]
                db = lab9[p][2] - o[2]
                diffs.append((0.15 * dl * dl + da * da + db * db) ** 0.5)
        if len(diffs) < 4:
            return None
        return float(np.mean(diffs))

    def _band_peaks(self):
        """NMS peaks of the band-excess channel: frames where exactly one
        3-cell band of the dominant face moved (wrist-flick turns that stay
        under the bbox stillness gate). Own channel, own threshold —
        folding band into the motion scalar is intentionally avoided. Cached
        per track() call."""
        if self._band_peaks_cache is not None:
            return self._band_peaks_cache
        peaks = []
        if self.band_split and self.band:
            fs = sorted(self.band)
            W = 9   # a real 120fps turn spans ~19 frames; NMS half-width
            for idx, f in enumerate(fs):
                v = self.band[f]
                if v is None or v < self.band_split:
                    continue
                neigh = [self.band[g] for g in fs[max(0, idx - W):idx + W + 1]
                         if abs(g - f) <= W and self.band[g] is not None]
                if v >= max(neigh):
                    peaks.append(f)
        self._band_peaks_cache = peaks
        return peaks

    def _band_peaks_between(self, a, b):
        return sum(1 for p in self._band_peaks() if a < p <= b)

    @staticmethod
    def _span_read_matrix(span, min_l):
        """Stack a span's reads into (n_reads, n_cells, 3) over the cell keys
        present in >= 80% of reads; missing cells imputed with the column
        median. Returns (X, ok)."""
        keys_count = {}
        per_read = []
        for _, r, _m in span:
            d = {}
            for face, lab9 in r:
                for p in range(9):
                    l = lab9[p]
                    if l[0] >= min_l:
                        d[(face, p)] = np.asarray(l, np.float64)
            per_read.append(d)
            for k in d:
                keys_count[k] = keys_count.get(k, 0) + 1
        n = len(per_read)
        keys = [k for k, c in keys_count.items() if c >= 0.8 * n]
        if n < 8 or len(keys) < 6:
            return None, False
        X = np.empty((n, len(keys), 3))
        for j, k in enumerate(keys):
            col = [d[k] for d in per_read if k in d]
            med = np.median(np.asarray(col), axis=0)
            for i, d in enumerate(per_read):
                X[i, j] = d.get(k, med)
        return X, True

    _CPD_W = np.array([0.15, 1.0, 1.0])

    CPD_WIN = 12   # local window (reads) for the step statistic: drift
                   # across the span contributes ~nothing to a 12v12 local
                   # step; a real move contributes its full color distance
                   # (full-side means split drifting spans at their midpoint)

    def _cpd_split_span(self, span, min_l, depth=0):
        """Recursive binary segmentation with a LOCAL windowed step.

        v2 after the v1 falsification: (a) spans are pre-segmented at
        slot-set changes — a face appearing/dropping RENAMES slots and fakes
        a giant step (measured: ~30% of v1 breaks); slot-set boundaries are
        never emitted as breaks; (b) the step statistic is windowed (see
        CPD_WIN); (c) the accept threshold self-normalizes against the
        span's own null: max(cpd_split, 4 x median step over all t)."""
        if depth == 0:
            # pre-segment at slot-set changes (no breaks emitted for these)
            segs, cur = [], [span[0]]
            sset = lambda it: tuple(sorted(name for name, _ in it[1]))
            for it in span[1:]:
                if sset(it) != sset(cur[-1]):
                    segs.append(cur); cur = []
                cur.append(it)
            segs.append(cur)
            if len(segs) > 1:
                out = []
                for s in segs:
                    if len(s) >= self.minspan:
                        out += self._cpd_split_span(s, min_l, depth + 1)
                return out if out else [span]
            depth = 1
        if depth >= 8 or len(span) < 2 * self.minspan:
            return [span]
        X, ok = self._span_read_matrix(span, min_l)
        if not ok:
            return [span]
        n = len(span)
        W = self.CPD_WIN
        steps = np.zeros(n)
        for t in range(self.minspan, n - self.minspan + 1):
            lo, hi = max(0, t - W), min(n, t + W)
            mu_l = X[lo:t].mean(axis=0)
            mu_r = X[t:hi].mean(axis=0)
            dd = ((mu_l - mu_r) ** 2 * self._CPD_W).sum(axis=1) ** 0.5
            steps[t] = float(dd.mean())
        valid = steps[self.minspan: n - self.minspan + 1]
        if valid.size == 0:
            return [span]
        thr = max(float(self.cpd_split), 4.0 * float(np.median(valid)))
        best_t = int(np.argmax(steps))
        if steps[best_t] < thr:
            return [span]
        left, right = span[:best_t], span[best_t:]
        self._cpd_breaks.add((left[-1][0], right[0][0]))
        return (self._cpd_split_span(left, min_l, depth + 1)
                + self._cpd_split_span(right, min_l, depth + 1))

    def _cpd_breaks_between(self, a, b):
        return sum(1 for (l, r) in self._cpd_breaks if a <= l and r <= b)

    def cpd_break_frames(self, span, min_l):
        """Localize-only CPD: recursive windowed-step argmax split points for
        one span (the v2 statistic — drift-immune, self-normalized accept —
        WITHOUT slot pre-segmentation or any move witness). Returns break
        frames for force_breaks."""
        out = []

        def rec(s, depth):
            if depth >= 8 or len(s) < 2 * self.minspan:
                return
            X, ok = self._span_read_matrix(s, min_l)
            if not ok:
                return
            n = len(s)
            W = self.CPD_WIN
            steps = np.zeros(n)
            for t in range(self.minspan, n - self.minspan + 1):
                lo, hi = max(0, t - W), min(n, t + W)
                dd = ((X[lo:t].mean(axis=0) - X[t:hi].mean(axis=0)) ** 2
                      * self._CPD_W).sum(axis=1) ** 0.5
                steps[t] = float(dd.mean())
            valid = steps[self.minspan: n - self.minspan + 1]
            if valid.size == 0:
                return
            thr = 4.0 * float(np.median(valid))
            best_t = int(np.argmax(steps))
            if steps[best_t] < thr or steps[best_t] <= 0:
                return
            out.append(s[best_t][0])
            rec(s[:best_t], depth + 1)
            rec(s[best_t:], depth + 1)

        rec(span, 0)
        return out

    def build_spans(self, reads: dict, min_sticker_l: float) -> list:
        """reads: {frame: (read, motion)} -> list of spans [(frame, read), ...].

        span_split > 0 additionally splits a span where consecutive reads STEP
        in color space: gentle wrist-flick turns (D-block flicks) stay under
        the stillness gate, so gap-only segmentation fuses multi-move stretches
        into one "held" span that cannot vote a single state.
        band_split > 0 splits at band-excess NMS peaks instead (the
        band-residual channel: the same fused-span failure detected by the
        per-face band signal)."""
        conf = [(i, r, m) for i, (r, m) in sorted(reads.items())
                if self._nstk(r, min_sticker_l) >= self.minstk
                and self._nfaces(r, min_sticker_l) >= self.minfaces
                and m <= self.still]
        if not conf:
            return []
        spans, cur = [], [conf[0]]
        for item in conf[1:]:
            gap_break = item[0] - cur[-1][0] > self.gapf
            step_break = False
            if not gap_break and self.span_split > 0:
                st = self._read_step(cur[-1][1], item[1], min_sticker_l)
                step_break = st is not None and st > self.span_split
            band_break = False
            if not gap_break and not step_break and self.band_split:
                band_break = self._band_peaks_between(cur[-1][0], item[0]) > 0
            forced = (not gap_break and self.force_breaks
                      and any(cur[-1][0] < bf <= item[0]
                              for bf in self.force_breaks))
            rot_break = (not gap_break and self.rot_events
                         and any(cur[-1][0] < ef <= item[0]
                                 for ef in self.rot_events))
            if gap_break or step_break or band_break or forced or rot_break:
                spans.append(cur)
                cur = []
            cur.append(item)
        spans.append(cur)
        spans = [s for s in spans if len(s) >= self.minspan]
        if self.cpd_split:
            self._cpd_breaks = set()
            spans = [sub for s in spans
                     for sub in self._cpd_split_span(s, min_sticker_l)]
        if self.drop_spans:
            spans = [s for s in spans
                     if (s[0][0], s[-1][0]) not in self.drop_spans]
        return spans

    def _span_consensus(self, span, min_l):
        """Robust per-span READ: per (slot,cell) the MAJORITY classified color over ALL
        the span's frames, emitted as that color's centroid LAB. A settled span is one
        cube state, so its clean frames agree on each cell's color; occlusion/mid-turn
        frames are a minority the majority vote rejects. Color is CATEGORICAL, so we
        vote on the classified color (NOT median LAB, which blends red+orange into a
        non-physical between-color). More aligned frames -> a cleaner consensus -> a
        decisive state vote. Returns [(slot, lab9)] or None (needs consensus_centroids)."""
        cents = self.consensus_centroids
        if not cents:
            return None
        names = list(cents)
        cmat = np.array([np.asarray(cents[c], float) for c in names])
        from collections import defaultdict, Counter
        votes = defaultdict(Counter)
        for it in span:
            for slot, lab9 in it[1]:
                for pos in range(9):
                    lab = lab9[pos]
                    if lab[0] >= min_l:
                        d = (((np.asarray(lab, float) - cmat) ** 2) * _LAB_W).sum(1)
                        votes[(slot, pos)][names[int(d.argmin())]] += 1
        if not votes:
            return None
        cons = []
        for slot in sorted({s for s, _ in votes}):
            lab9 = []
            for pos in range(9):
                c = votes.get((slot, pos))
                lab9.append([float(x) for x in cents[c.most_common(1)[0][0]]]
                            if c else [0.0, 128.0, 128.0])   # invalid -> AbsSegment drops
            cons.append((slot, lab9))
        return cons

    def _span_timeline_om(self, span):
        """The om_key the timeline assigns this span = the MODE over the span's
        frames' timeline entries (a settled span has one consistent camera-frame
        om). Returns None if the timeline covers none of the span's frames (the
        beam then resolves the om normally for that span)."""
        if not self.om_timeline:
            return None
        oms = [self.om_timeline.get(it[0]) for it in span]
        oms = [o for o in oms if o is not None]
        return max(set(oms), key=oms.count) if oms else None

    def _frame_score_om(self, fi, group_omk, om_by_key):
        """The (om_key, om_dict) a single frame's read is scored under. With
        per-frame om scoring ON, each frame uses its OWN timeline om
        (om_timeline[fi]) so a span's state-vote FUSES across the orientations its
        frames span (the re-grip fix); otherwise the span/group om (bit-identical
        legacy). The per-frame om comes from either the reads-resolved soft-fusion
        map (_pf_om, gated) or an external om_timeline. Falls back to the group om
        when the frame is absent / low-confidence or its key is unknown, so a
        sparse/partial source never mis-scores -- this fallback IS the soft-fusion's
        no-regress-below-baseline guarantee."""
        if self.om_per_frame:
            src = self._pf_om if self.om_per_frame_from_reads else self.om_timeline
            if src:
                fomk = src.get(fi)
                if fomk is not None:
                    fomk = tuple(fomk) if isinstance(fomk, list) else fomk
                    fom = om_by_key.get(fomk)
                    if fom is not None:
                        return fomk, fom
        return group_omk, om_by_key.get(group_omk)

    @staticmethod
    def _ball_face_patterns(states):
        """{model_face: (K, 9) unique sticker patterns} across the ball's
        states — the per-face joint structure _face_ub_table bounds with.
        Six unique() passes over the ball matrix; built ONCE per span."""
        st = np.asarray(states)
        return {f: np.unique(st[:, b:b + 9], axis=0)
                for f, b in FACE_OFFSET.items()}

    @staticmethod
    def _face_ub_table(seg, pats):
        """{(read_slot, model_face): minimal summed (trust-weighted) distance
        of that slot's admitted cells against ANY ball pattern of the model
        face under ANY grid rotation}.

        This is the per-face relaxation: intra-face joint color structure is
        EXACT (patterns come from the ball itself), only cross-face coupling
        and the shared-rotation constraint are relaxed — far tighter than a
        per-cell color-set bound on real reads (measured: the per-cell form
        left 18.0/24 oms alive; junk cells saturate per-cell minima). The
        table is om-INDEPENDENT (an om only selects WHICH (slot, face)
        entries are summed), so one table serves all 24 oms of a frame."""
        if seg._dflat is None:
            seg._dflat = np.ascontiguousarray(seg.dist).ravel()
        rots = np.asarray(GRID_ROTATIONS)
        T = {}
        for fn in seg.faces:
            idx = np.flatnonzero(np.asarray(
                [f == fn for f in seg._face_names]))
            base6 = idx * 6
            posf = seg._pos[idx]
            wf = seg._wv[idx] if seg._wv is not None else None
            for mf, P in pats.items():
                C = P[:, rots[:, posf]]              # (K, 4, n_f) colors
                dd = seg._dflat[base6[None, None, :] + C]
                costs = ((dd * wf[None, None, :]).sum(-1)
                         if wf is not None else dd.sum(-1))
                T[(fn, mf)] = float(costs.min())
        return T

    @classmethod
    def _om_score_ub(cls, seg, pats, table=None):
        """SOUND upper bound on seg.score_states(ball).max() for an
        AbsSegment, from the ball's per-face pattern sets.

        For any ball state s and combo: each read slot's cells score against
        s's pattern on the mapped model face under the combo's rotation —
        that pattern IS in pats[mf] (built from the same ball), so the summed
        cost is >= T[(slot, mf)] (min over all patterns AND rotations; weights
        >= 0). Summing slots and negating gives ub >= true max, for ANY dist
        sign (legacy LAB, EMISSION, caps)."""
        if not seg.ok:
            return -1e18
        T = table if table is not None else cls._face_ub_table(seg, pats)
        tot = sum(T[(fn, seg.orient_map.get(fn, fn))] for fn in seg.faces)
        return float(-tot if seg._wv is not None else -tot / seg.n)

    # ---- Z-NORM DERIVED GATES: per-solve anchor fit-noise estimator ---------
    def _note_anchor_fit(self, span, sub_frames, sub_reads, scored,
                         motions, om_by_key, segcache, min_sticker_l):
        """Accumulate the read-fit of a CONFIDENT ANCHOR span into the per-solve
        fit-noise sample. Reuses the EXACT coverage test anchor_resync's PROMOTE
        gate uses (fit in the correct band, >= anchor_min_faces / anchor_min_nstk,
        still, mean per-cell conf >= anchor_cellconf_thr, span length): a span
        that passes is a decisive correct read, so its best span read-fit is a
        live sample of 'what a correct read scores under THIS capture's lighting/
        cube'. No new detector -- the same faces/nstk/cellconf the PROMOTE gate
        computes. Called only when a z-norm gate (or the diagnostic) is active."""
        if not scored:
            return
        fit_best = max(x[5] for x in scored)
        # NOTE: no absolute LAB-band gate here -- z-norm's whole point is to LEARN
        # the correct-read band per capture, so gating the sampler on a fixed band
        # would defeat it. Confidence comes from coverage + stillness + cell_conf.
        nfaces = max((self._nfaces(r, min_sticker_l) for r in sub_reads),
                     default=0)
        if nfaces < self.znorm_min_faces:
            return
        nstk = max((self._nstk(r, min_sticker_l) for r in sub_reads), default=0)
        if nstk < self.znorm_min_nstk:
            return
        if len(span) < self.anchor_min_frames:
            return
        mv = [motions.get(fi) for fi in sub_frames]
        mv = [m for m in mv if m is not None]
        span_motion = float(np.mean(mv)) if mv else 0.0
        if span_motion > self.still / 2.0:
            return
        win_omk = scored[0][1][1]
        cv = []
        for fi, r in zip(sub_frames, sub_reads):
            fomk, fom = self._frame_score_om(fi, win_omk, om_by_key)
            seg = segcache.get((fi, fomk)) or self.seg_factory(r, fom)
            cf = getattr(seg, "_confs", None)
            if seg.ok and cf is not None and len(cf):
                cv.append(float(cf.mean()))
        mean_cellconf = float(np.mean(cv)) if cv else 0.0
        if mean_cellconf < self.znorm_cellconf_thr:
            return
        self._fitnoise_vals.append(float(fit_best))

    def _znorm_stats(self):
        """(mu, sigma, n) of the per-solve confident-anchor fit-noise. Prefer a
        frozen whole-solve seed (set by track_2pass after pass 1) so the FINAL
        pass uses a stable estimate from frame 0; else the running within-pass
        sample (cold-started, constant fallback below znorm_min_anchors)."""
        vals = (self._fitnoise_seed if self._fitnoise_seed
                else self._fitnoise_vals)
        n = len(vals)
        if n == 0:
            return None, None, 0
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
        return mu, sd, n

    def _eff_misfit_thr(self):
        """misfit_thr as mu - k*sigma vs the anchor fit-noise (z-norm on and a
        clean per-solve estimate exists); else the raw constant."""
        if not self.znorm_gates:
            return self.misfit_thr
        mu, sd, n = self._znorm_stats()
        if n < self.znorm_min_anchors or not sd or sd <= 0.0:
            return self.misfit_thr
        return mu - self.znorm_k_misfit * sd

    def _eff_move_gate_margin(self):
        """move_gate_margin as k*sigma vs the anchor fit-noise. Preserves an
        infinite raw margin (the track_2pass pass-1 disable) so the measurement
        pass still fires on every event-gap."""
        raw = self.move_gate_margin
        if not self.znorm_gates or not np.isfinite(raw):
            return raw
        mu, sd, n = self._znorm_stats()
        if n < self.znorm_min_anchors or not sd or sd <= 0.0:
            return raw
        return self.znorm_k_move * sd

    def _eff_om_pf_margin(self):
        """om_pf_margin as k*sigma vs the anchor fit-noise; else the constant."""
        if not self.znorm_gates:
            return self.om_pf_margin
        mu, sd, n = self._znorm_stats()
        if n < self.znorm_min_anchors or not sd or sd <= 0.0:
            return self.om_pf_margin
        return self.znorm_k_ompf * sd

    def _log_ompf_pin(self, fi, scs, om_pf_margin):
        """CUBED_OMPF_LOG=1 instrumentation: one line per
        per-frame om PIN decision -- frame, best omk, the raw margin (best -
        2nd), and whether it pinned -- so k_ompf can be derived offline from
        labeled data (parallel to how k_move was derived from the pass-1
        [move-gate] margins). `scs` is the SCORED subset at this site, sorted
        desc by score (already computed/sorted by every call site for its own
        pin check -- this only formats it). For the lazy/GPU ball paths a
        single survivor means every rival was already bound-eliminated below
        margin (see _resolve_perframe_om docstring): the true margin is
        right-censored -- unknown exact value, provably > om_pf_margin --
        logged as margin=inf (still a genuine, if uninformative-for-tighter-k,
        pin). `omk` is logged as its stable 0-23 ORIENTATIONS index (_OMK_TO_IDX),
        not the raw nested-tuple key, so the line stays a flat, parseable
        int/float record. Callers gate this behind `if self.log_ompf_pf:` so
        the sort/format cost (and the print itself) is zero unless the flag
        is set.
        """
        best_omk = _OMK_TO_IDX[scs[0][1]]
        margin = (scs[0][0] - scs[1][0]) if len(scs) >= 2 else float("inf")
        pinned = margin >= om_pf_margin
        print(f"[ompf-pin] f={fi} omk={best_omk} margin={margin:.6f} "
              f"pinned={int(pinned)}", flush=True)

    def _resolve_perframe_om(self, sub_frames, sub_reads, anchor, om_by_key,
                             segcache, exp_hi=None, d_gap=None):
        """SOFT-FUSION source: per-frame om from the READS (no gyro/model). For
        each frame, the orientation under which the beam's ANCHOR state best fits
        that frame's visible read (the occlusion-tolerant 24-om match om-rescue
        uses per-span, here per-frame). Keep only oms decided by a margin >=
        om_pf_margin (best minus 2nd fit); ambiguous/occluded frames are omitted so
        the scorer falls back to the span om. Reuses/fills segcache so the segs are
        shared with scoring. Returns {frame: omk}.

        om_pf_ball=d (default None = stock): score each om by its MAX fit over the
        anchor's depth-d move ball instead of the single (stale) anchor state — see
        the om_pf_ball knob comment in __init__ for the forensic rationale.
        om_pf_ball="gap": depth derived per span as min(ceil(exp_hi), d_gap)
        — the caller's admissible move-count ceiling for THIS gap, capped at
        the span's own transition-ball depth (the states the beam can reach
        this span) and the deepest precomputed table — see the __init__
        comment. The ball matrix is built ONCE per span (this method is
        called once per span).

        LAZY LOSSLESS EVALUATION (ball modes; perf only, decisions PROVABLY
        identical): full ball scoring runs ONLY for contender oms. Each om
        first gets the sound upper bound _om_score_ub; oms are then fully
        scored in descending-bound order, and as soon as every remaining om
        has ub < best_full - om_pf_margin the rest are eliminated: their true
        score is strictly below best - margin, so they can neither be the
        argmax nor raise the runner-up above the pin threshold — the pin
        decision (WHICH om, and WHETHER the >= margin gate passes, against
        the strictly-below eliminated set) is exactly the all-scored
        decision. Measured rationale: a deep ball (d4 = 46,741 states) costs
        ~0.3-0.5 s per om per frame while decoy oms sit far outside the
        margin — the bound eliminates ~21-23 of the 24 full passes. The wide
        24-om batched-chunk variant was tried first and REJECTED by
        measurement (box canonical 1507->1804 s: the 24x-wider chunk tensor
        falls out of cache; per-om score_states chunks stay cache-resident)."""
        out = {}
        if anchor is None:
            return out
        # z-norm derived margin (== self.om_pf_margin when the gate is off);
        # computed once per span so every per-frame gate below uses one value.
        om_pf_margin = self._eff_om_pf_margin()
        if self.om_pf_ball is None:
            states = np.asarray(anchor)[None, :]
        else:
            if self.om_pf_ball == "gap":
                # gap-derived depth: ceil admits a fractional budget's last
                # move (soft witnesses make exp_hi fractional); 0 => depth-0
                # ball == the anchor alone == the stock single-state check.
                depth = int(np.ceil(exp_hi)) if exp_hi is not None else 0
                if d_gap is not None:
                    depth = min(depth, int(d_gap))
                depth = int(min(depth, max(_PERM_TABLES)))
            elif self.om_pf_ball == "dgap":
                # dgap-derived depth: the span's TRANSITION ball depth d_gap
                # DIRECTLY (the states the beam can actually reach this span,
                # the mechanism), dropping gap's min(ceil(exp_hi), .)
                # shrink. That exp_hi cap pulled the pf-ball BELOW the beam's
                # own move reach; dgap matches the beam reach exactly. Knob-free
                # (d_gap
                # is derived per span, no magic constant) and STRICTLY >= gap's
                # depth. d_gap None (should not occur when a ball mode is set)
                # => depth-0 ball == the anchor alone == the stock check.
                depth = int(d_gap) if d_gap is not None else 0
                depth = int(min(depth, max(_PERM_TABLES)))
            else:
                depth = self.om_pf_ball
            if self._pf_ball_depth_hist is not None:
                self._pf_ball_depth_hist[depth] = \
                    self._pf_ball_depth_hist.get(depth, 0) + 1
            states = np.asarray(anchor)[_PERM_TABLES[depth][0]] \
                .astype(np.int8)
            states = np.unique(states, axis=0)
        # lazy path: only for a real ball (>1 state) built from AbsSegments —
        # the ub is an AbsSegment-structure bound. Stock/degenerate balls keep
        # the plain all-scored loop (already cheap, byte-identical legacy).
        lazy = self.om_pf_ball is not None and len(states) > 1
        pats = self._ball_face_patterns(states) if lazy else None
        # GPU resolver (opt-in via CUBED_GPU_SCORING; default-off => byte-identical
        # CPU). Only for a real ball; the shared ball uploads ONCE per span here and
        # is reused across the span's frames (amortizing the CPU<->GPU transfer that
        # would otherwise dominate a per-frame 24-om score). "off" mode keeps CPU.
        gpu = None
        states_t = None
        if (_GPU_SCORING and lazy and _GPU_RESOLVER_MODE != "off"):
            gpu = _torch_cuda()
        for fi, r in zip(sub_frames, sub_reads):
            segs = []
            for o in ORIENTATIONS:
                omk = _om_key(o)
                ck = (fi, omk)
                seg = segcache.get(ck)
                if seg is None:
                    seg = self.seg_factory(r, o)
                    segcache[ck] = seg
                segs.append((omk, seg))
            if (gpu is not None
                    and all(type(s) is AbsSegment and s.ok for _, s in segs)):
                # GPU-scored pin decision (identical structure to the CPU lazy
                # block: same margin gate, same pin rule — only the 24 om MAX
                # scores are computed on the 4090 instead of pruned+CPU-scored).
                try:
                    if states_t is None:
                        states_t = gpu.as_tensor(np.asarray(states),
                                                 device=gpu.device("cuda"))
                    scs = (self._resolve_frame_gpu_lazy(gpu, segs, states,
                                                        states_t, pats,
                                                        om_pf_margin)
                           if _GPU_RESOLVER_MODE == "lazy"
                           else self._resolve_frame_gpu_all(gpu, segs, states_t))
                except Exception as e:      # OOM / driver hiccup -> CPU this span
                    _gpu_resolver_warn(e)
                    gpu = None
                    scs = None
                if scs is not None:
                    if self._pf_ball_scored is not None:
                        self._pf_ball_scored[0] += 1
                        self._pf_ball_scored[1] += len(scs)
                    scs.sort(reverse=True)
                    if self.log_ompf_pf:
                        self._log_ompf_pin(fi, scs, om_pf_margin)
                    if (len(scs) == 1
                            or scs[0][0] - scs[1][0] >= om_pf_margin):
                        out[fi] = scs[0][1]
                    continue
            if lazy and all(type(s) is AbsSegment and s.ok for _, s in segs):
                # one om-independent bound table serves all 24 oms (the segs
                # share cells/dist by construction; om only picks entries).
                # Guard the shared-structure invariant per seg — a mismatch
                # (a future om-dependent construction gate) falls back to a
                # per-seg table, never an unsound shared one.
                s0 = segs[0][1]
                table = self._face_ub_table(s0, pats)
                ubs = sorted(
                    ((self._om_score_ub(
                        s, pats,
                        table=table if (s.n == s0.n and s.faces == s0.faces
                                        and np.array_equal(s.dist, s0.dist))
                        else None), omk, s)
                     for omk, s in segs), reverse=True)
                scs, best = [], -1e18
                for ub, omk, s in ubs:
                    # 1e-9 = fp-soundness guard, NOT a knob: the table sums
                    # per-slot then divides while score_states means over all
                    # cells — association differs by O(n*eps*max|dist|)
                    # ~ 6e-12 (measured worst 1.4e-14); the guard only ever
                    # ADMITS borderline oms to full scoring, so decisions
                    # remain exactly the all-scored ones.
                    if scs and ub < best - om_pf_margin - 1e-9:
                        break   # sorted desc: this om and ALL after are
                        #         strictly below best - margin => eliminated
                    v = (float(s.score_states(states).max())
                         if s.ok else -1e18)
                    scs.append((v, omk))
                    if v > best:
                        best = v
                if self._pf_ball_scored is not None:
                    self._pf_ball_scored[0] += 1
                    self._pf_ball_scored[1] += len(scs)
                scs.sort(reverse=True)
                if self.log_ompf_pf:
                    self._log_ompf_pin(fi, scs, om_pf_margin)
                # eliminated oms are strictly below best - margin, so the
                # margin gate over the scored set equals the full gate; a
                # single survivor means EVERY rival was eliminated => pin.
                if len(scs) == 1 or scs[0][0] - scs[1][0] >= om_pf_margin:
                    out[fi] = scs[0][1]
                continue
            scs = [(float(s.score_states(states).max())
                    if s.ok else -1e18, omk) for omk, s in segs]
            scs.sort(reverse=True)
            if self.log_ompf_pf:
                self._log_ompf_pin(fi, scs, om_pf_margin)
            if len(scs) >= 2 and scs[0][0] - scs[1][0] >= om_pf_margin:
                out[fi] = scs[0][1]
        return out

    def _resolve_frame_gpu_all(self, torch, segs, states_t):
        """GPU mode "all": score EVERY om's ball MAX in one batched call and
        return [(score, omk), ...] for all 24. No CPU bound — the batched
        gather makes scoring all 24 as cheap as scoring the contenders, and
        the shared states tensor is already resident (uploaded once per span).
        The pin decision the caller applies over these 24 is exactly the
        all-scored decision (== the CPU lazy decision, to float ULPs)."""
        vals = _score_ball_oms_gpu(torch, [s for _, s in segs], states_t)
        return [(float(vals[i]), segs[i][0]) for i in range(len(segs))]

    def _resolve_frame_gpu_lazy(self, torch, segs, states, states_t, pats,
                                om_pf_margin=None):
        """GPU mode "lazy" (A/B baseline for "all"): keep CPU per-face
        upper bound to PRUNE, GPU-score only the contenders. Seed the running
        best by GPU-scoring the highest-ub om, then batch-score every remaining
        om whose ub >= best - om_pf_margin. Decision-identical to the CPU lazy
        path: any om NOT scored has ub < best_seed - margin <= best_final -
        margin, so its true score is strictly below best_final - margin — it
        can be neither the argmax nor a runner-up inside the gate. Returns the
        SCORED [(score, omk), ...] subset (as the CPU lazy path does)."""
        s0 = segs[0][1]
        table = self._face_ub_table(s0, pats)
        ubs = sorted(
            ((self._om_score_ub(
                s, pats,
                table=table if (s.n == s0.n and s.faces == s0.faces
                                and np.array_equal(s.dist, s0.dist))
                else None), omk, s)
             for omk, s in segs), reverse=True)
        # seed: GPU-score the highest-bound om (best possible score is here or
        # below every other om's ub).
        if om_pf_margin is None:
            om_pf_margin = self.om_pf_margin
        seed = _score_ball_oms_gpu(torch, [ubs[0][2]], states_t)[0]
        best = float(seed)
        scs = [(best, ubs[0][1])]
        rest = [(ub, omk, s) for ub, omk, s in ubs[1:]
                if ub >= best - om_pf_margin - 1e-9]
        if rest:
            vals = _score_ball_oms_gpu(torch, [s for _, _, s in rest], states_t)
            scs.extend((float(v), omk) for (ub, omk, s), v in zip(rest, vals))
        return scs

    def _lattice_keep(self, sub_frames, sub_reads, prev_items, d_gap,
                      switch_targets, om_by_key):
        """Per-read lattice-consistency filter for one span's voting reads.

        Returns the sublist of indices (into sub_frames/sub_reads) whose BEST
        achievable fit over this span's CANDIDATE BALL — the states reachable
        from the live beam at this gap's move depth — under ALL candidate
        orientations is >= self.lattice_gate. Reads below the floor match no
        settled cube state the beam could be in (mid-turn / off-lattice) and are
        excluded from the vote.

        Candidate ball: each beam predecessor's state expanded by the depth-d_gap
        permutation table (states[anchor][perms] — the exact gather the
        transition uses), unioned over the top beam predecessors. Candidate oms:
        the oms carried by the beam, plus all 24 when this gap enumerates switches
        (long gap), so a genuine reorientation read is judged against the layer it
        actually belongs to, never dropped for failing the stale om.

        GT-FREE and beam-relative; only safe when the anchor is trustworthy (see
        the knob comment). Never returns fewer than min(minspan, n) reads — the
        gate re-weights a vote, it does not delete a span."""
        n = len(sub_reads)
        if n == 0 or self.lattice_gate is None:
            return list(range(n))
        perms = _PERM_TABLES[d_gap][0]            # (N_d, 54) ball gather rows
        # anchor states: top beam predecessors (by cumulative score). Cap the
        # count so the ball stays bounded on deep/wide layers (the max over the
        # ball is dominated by the best few anchors.
        top_prev = sorted(prev_items, key=lambda kv: -kv[1][1])[:12]
        if not top_prev:
            return list(range(n))
        # group anchor states by their carried om so each candidate state is
        # scored under the om the beam actually holds it in (plus all 24 if this
        # gap enumerates switches).
        anchors_by_om = {}
        for (pb, pomk), val in top_prev:
            parr = val[0]
            anchors_by_om.setdefault(pomk, []).append(parr)
        cand_omks = set(anchors_by_om)
        if switch_targets:
            cand_omks |= {t[0] for t in switch_targets}
        # build the candidate ball state matrix per om (anchors expanded by the
        # ball, deduped). Cap total rows per om to keep scoring cheap.
        ball_by_om = {}
        for omk in cand_omks:
            anchors = anchors_by_om.get(omk)
            if not anchors:
                # a pure switch-target om with no beam anchor: use every beam
                # anchor (the read may belong to a layer the beam will switch
                # into; judging it against the reachable ball is the point).
                anchors = [v[0] for (_k, v) in top_prev]
            mats = []
            for a in anchors:
                mats.append(np.asarray(a, dtype=np.int8)[perms])  # (N_d, 54)
            M = np.concatenate(mats, axis=0)
            # dedup rows (the ball overlaps heavily across anchors)
            M = np.unique(M, axis=0)
            ball_by_om[omk] = M
        # per read: max fit over (candidate ball x candidate oms)
        best = np.full(n, -1e18)
        for omk, states in ball_by_om.items():
            om = om_by_key.get(omk)
            for i, r in enumerate(sub_reads):
                seg = self.seg_factory(r, om)
                if not getattr(seg, "ok", False):
                    continue
                v = float(seg.score_states(states).max())
                if v > best[i]:
                    best[i] = v
        keep = [i for i in range(n) if best[i] >= self.lattice_gate]
        # never empty a span: if the gate would drop too many, keep the
        # min(minspan, n) best-fitting reads so the span still votes.
        floor = min(self.minspan, n)
        if len(keep) < floor:
            order = sorted(range(n), key=lambda i: -best[i])
            keep = sorted(order[:floor])
        if self.lattice_gate is not None:
            # log the span's frame extent (min/max — sub_frames is ordered by
            # sticker count, not frame, so use the actual range) + kept count.
            self._lattice_log.append((min(sub_frames) if sub_frames else -1,
                                      max(sub_frames) if sub_frames else -1,
                                      n, len(keep)))
        return keep

    # human move-rate: peaked prior center + slack (flat wide intervals have no
    # teeth — measured: z1 regressed to 0.08 with [dur/0.35, dur/0.10] bounds)
    MOVE_SEC_TYP = 0.16
    MOVE_SEC_MIN = 0.06   # fastest humanly-sustained trigger pace (s/move)
    COUNT_SLACK = 1
    dur_hi = True     # duration ceiling on gap move-count (off: bursts-only lower bound)
    dur_hi_rate = 0.10  # s/move for the dur_hi ceiling: hi = round(gap/(rate*fps)).
    #   Lower => more moves admitted per gap before the count penalty charges.
    #   0.10 is the legacy value (no human turns faster than ~0.1 s/move); the
    #   physical floor is MOVE_SEC_MIN=0.06. Tunable for fast-adjacent bursts.
    deep_gap = True   # extend ball depth past self.d on long/deep gaps (off: fixed d)
    active_hi = False  # active-motion-time depth bound.
    # DEPTH ONLY, never the count-penalty window: widening [lo,hi] lets long decoy
    # paths through penalty-free and floods the beam. The intended path only
    # needs to be reachable; emission evidence still decides among candidates.

    def _content_count(self, a: int, b: int) -> int:
        """CONTENT-based move-count for a gap (a, b], from the flow z-vorticity
        magnitude (flowrot). Each face turn sweeps the stickers coherently => one
        |z-vort| RAMP with a local peak; counting NMS peaks counts turns even when
        the coarse motion-event channel COLLAPSED them into one burst. Pace/duration
        independent and READ-independent (immune to colour/lighting). Returns 0 when
        no flowrot is wired (=> burst_recover becomes a no-op for that span)."""
        if self.flow_rotmag is None:
            return 0
        if self._fr_frames is None:
            fr = sorted(self.flow_rotmag)
            self._fr_frames = np.asarray(fr, dtype=np.int64)
            self._fr_mag = np.asarray([self.flow_rotmag[f] for f in fr], dtype=np.float64)
            med = float(np.median(self._fr_mag)) if len(self._fr_mag) else 0.0
            self._fr_thr = max(0.02, self.burst_thr_mult * med)
        frames, mag, thr, W = self._fr_frames, self._fr_mag, self._fr_thr, self.burst_nms_w
        # indices whose frame is in (a, b]
        i0 = int(np.searchsorted(frames, a, side="right"))
        i1 = int(np.searchsorted(frames, b, side="right"))
        n = 0
        for k in range(i0, i1):
            v = mag[k]
            if v < thr:
                continue
            lo = max(0, k - W); hi = min(len(mag), k + W + 1)
            if v >= mag[lo:hi].max() - 1e-9:   # local maximum => one turn
                n += 1
        return n

    def _expected_interval(self, motions: dict, a: int, b: int, fps: float):
        """Admissible move-count interval for a read gap: the gap DURATION bounds
        how many moves fit ([dur/slow, dur/fast]); motion bursts give a lower bound
        (bursts under-count fluid/low-motion styles - measured: 4 known moves showed
        1-2 bursts). Penalty applies outside the interval only."""
        # ASYMMETRIC duration prior — the only pace-independent fact is the upper
        # bound (no human turns faster than ~0.1 s/move). A symmetric/peaked prior
        # needs the pace, which varies clip-to-clip (deliberate 0.3-0.5 s/move vs
        # trigger 0.12-0.2) — a single rate broke whichever regime it wasn't tuned
        # for (measured both directions). lo stays burst-anchored (weak); evidence
        # and the terminal constraint do the counting.
        bursts = self._expected_moves(motions, a, b)
        gap = max(0, b - a)
        lo = bursts
        if self.band_split:
            # band-peak witness: each peak in the gap is a move the bbox
            # motion missed (wrist flicks). Without this, a band-split
            # boundary's tiny gap implies hi=0 and the count penalty
            # suppresses the very move the split exposed.
            lo = max(lo, self._band_peaks_between(a, b))
        if self.cpd_split:
            # change-point witness: a CPD break between these spans means
            # the read content STEPPED — at least one move happened there
            # regardless of what the motion scalar saw.
            lo = max(lo, self._cpd_breaks_between(a, b))
        if self.rot_events:
            # measured-rotation witness: each event is a quarter-turn the
            # probe SAW (band slid a face-width coherently). SOFT WITNESS: use the
            # gap's reads-weighted FRACTIONAL count (over-firing deflates) when on.
            n_ev = sum(1 for ef in self.rot_events if a < ef <= b)
            if self.soft_witness:
                n_ev = self._soft_lo_map.get((a, b), float(n_ev))
            lo = max(lo, n_ev)
        hi = max(lo, round(gap / (self.dur_hi_rate * fps))) if self.dur_hi else 10**9
        return lo, hi

    def _move_gate_margin(self, prev_arr, sub_reads, d_gap, cand_omks, om_by_key,
                          return_argmax=False):
        """Reads-driven move-vs-re-grip signal for a gap (see self.move_gate).
        Returns best_move_fit - best_stay_fit: the current span's mean read-fit to
        the BEST 1..min(d,2)-move neighbour of the previous state, minus its fit to
        the previous state UNCHANGED, over the CANDIDATE orientations (the previous
        om + the reads-resolved per-frame oms _pf_om) -- a small, principled set, so a
        re-grip that REORIENTS the cube without turning a face still resolves as
        no-change (its new om is among the candidates) without scanning all 24.
        >= move_gate_margin => a real state change (a move); below => re-grip.
        Returns +inf when no candidate om scores (occluded) => treated as a move
        (the event is kept) -- the safe direction (never drop on no evidence).

        return_argmax=False (default): unchanged, returns the bare float margin.
        return_argmax=True (refinement): returns
        (margin, best_omk, best_path) -- the om and move-path of the ball state
        that WON best_move, i.e. exactly the move+orientation the margin's
        verdict is actually about, for _gate_coverage_argmax below. (None, None)
        when no candidate om ever scored (occluded -- same case as the +inf
        margin)."""
        depth = int(min(max(d_gap, 1), 2))
        states, paths = ball_states(np.asarray(prev_arr).astype(np.int8), depth)
        is_move = np.array([len(p) > 0 for p in paths])
        move_rows = np.nonzero(is_move)[0]        # masked-index -> row-in-states
        best_stay, best_move = -1e18, -1e18
        best_omk, best_path = None, None
        for omk in cand_omks:
            o = om_by_key.get(omk)
            if o is None:
                continue
            segs = [self.seg_factory(r, o) for r in sub_reads]
            segs = [s for s in segs if s.ok]
            if not segs:
                continue
            fits = sum(s.score_states(states) for s in segs) / len(segs)
            best_stay = max(best_stay, float(fits[~is_move].max()))
            move_fits = fits[is_move]
            j = int(np.argmax(move_fits))
            if float(move_fits[j]) > best_move:
                best_move = float(move_fits[j])
                best_omk, best_path = omk, paths[int(move_rows[j])]
        if best_move < -1e17 or best_stay < -1e17:
            return (1e18, None, None) if return_argmax else 1e18
        margin = best_move - best_stay
        return (margin, best_omk, best_path) if return_argmax else margin

    def _gate_coverage(self, prev_arr, gm_reads, cand_omks, om_by_key):
        """Measurement-only coverage-vacuity diagnostic.
        A move-gate margin is VACUOUS when the reads cannot SEE the stickers any
        candidate move would change: best_move vs best_stay is then decided on
        shared (unchanged) cells only. Returns (coverage_frac, n_visible) for
        THIS gap. GT-free, read-only — no scoring, no state mutation.

        coverage_frac = |changed & visible| / |changed|, both sets of state
        54-indices:
          changed  = union over the depth-1 single moves (_PERM_TABLES[1]) of
                     the positions whose COLOR differs between the anchor and
                     the moved state (anchor[perm] != anchor) — anchor-dependent,
                     exactly the cells that could discriminate a move from stay.
                     Centres are fixed points of every turn, so a gap whose reads
                     resolve only centres scores 0.0 = maximally vacuous.
          visible  = every state position an admitted read cell (surviving
                     MIN_STICKER_L) could occupy under a candidate om, UNIONED
                     over the 4 grid rotations (score_states' per-face rotation
                     is unresolved here) and over cand_omks — an UPPER bound on
                     true cell-level visibility, so a LOW coverage_frac is a
                     CONSERVATIVE (never over-eager) vacuity signal.
        n_visible = |visible| (distinct covered state positions, rotation-unioned).
        Coarser than the per-argmax-move coverage below because it unions over
        all depth-1 moves and candidate orientations."""
        anchor = np.asarray(prev_arr).astype(np.int8)
        perms1, paths1 = _PERM_TABLES[1]
        moved = anchor[perms1]                             # (N1, 54)
        is_move = np.array([len(p) > 0 for p in paths1])
        diff = (moved != anchor[None, :])
        diff &= is_move[:, None]
        changed = set(int(j) for j in np.nonzero(diff.any(axis=0))[0])
        if not changed:
            return 1.0, 0
        visible = self._gate_visible_set(gm_reads, cand_omks, om_by_key)
        return len(changed & visible) / len(changed), len(visible)

    def _gate_visible_set(self, gm_reads, cand_omks, om_by_key):
        """Shared visibility core of _gate_coverage / _gate_coverage_argmax:
        every state position an admitted read cell (surviving MIN_STICKER_L)
        could occupy under any om in cand_omks, UNIONED over the 4 grid
        rotations (score_states' per-face rotation is unresolved at gate time)
        -- an UPPER bound on true cell-level visibility. Factored out so both
        callers share one visibility rule bit-for-bit."""
        visible = set()
        for omk in cand_omks:
            om = om_by_key.get(omk)
            if om is None:
                continue
            for r in gm_reads:
                for face, lab9 in r:
                    base = FACE_OFFSET.get(om.get(face, face))
                    if base is None:
                        continue
                    for pos in range(9):
                        if lab9[pos][0] < MIN_STICKER_L:
                            continue
                        for rr in range(4):
                            visible.add(base + int(GRID_ROTATIONS[rr][pos]))
        return visible

    def _gate_coverage_argmax(self, prev_arr, gm_reads, best_omk, best_path,
                              om_by_key):
        """MEASUREMENT-ONLY (pre-registered refinement to
        _gate_coverage). coverage of the SPECIFIC move _move_gate_margin's
        argmax chose, under the SPECIFIC om that won -- narrower than the
        union above, which unions visibility over 4 grid rotations x ALL
        candidate oms and changed-cells over ALL depth-1 moves (and so cannot
        tell an occluded winning move from a visible losing one).

        changed = positions where applying best_path (the winning ball-state's
        move-path, e.g. ['R'] or ['R', 'U']) to prev_arr differs from prev_arr
        -- reuses the MOVE_LIST perm-application idiom ball_paths/the om-
        Viterbi replay use, not a reimplementation.
        visible = _gate_visible_set restricted to the single best_omk (same
        4-grid-rotation union, same MIN_STICKER_L admission rule).

        Returns (cov_frac, n_changed, n_visible). best_path None/empty (the
        margin's occluded/no-candidate case) => (-1.0, 0, 0). Read-only: no
        scoring, no state mutation."""
        if not best_path:
            return -1.0, 0, 0
        perm_by_move = {n: p for n, p in MOVE_LIST}
        anchor = np.asarray(prev_arr).astype(np.int8)
        moved = anchor
        for notation in best_path:
            moved = moved[perm_by_move[notation]]
        changed = set(int(j) for j in np.nonzero(moved != anchor)[0])
        if not changed:
            return 1.0, 0, 0
        visible = self._gate_visible_set(gm_reads, {best_omk}, om_by_key)
        return len(changed & visible) / len(changed), len(changed), len(visible)

    def _gate_trust_argmax(self, prev_arr, gm_reads, best_omk, best_path,
                           om_by_key):
        """MEASUREMENT-ONLY (covT — continuation of
        _gate_coverage_argmax). Sum of RAW p_trust -- the TrustNumpy
        probability BEFORE AbsSegment._trust_soft_weights' max(p, floor)
        weight floor (see the TRUST_SOFT class-attr comment) -- over every
        admitted read-cell of gm_reads that maps, under the SAME winning
        om/move _gate_coverage_argmax scores (best_omk, best_path), to a
        state position in the argmax move's CHANGED set, under ANY of the 4
        grid rotations -- the identical conservative rotation-union
        convention _gate_visible_set/_gate_coverage_argmax use (rotation is
        unresolved at gate time). Raw trust is used because the scoring floor
        intentionally hides distinctions below that floor.

        Builds/reuses each read's AbsSegment via self.seg_factory(r, om) --
        a segcache HIT when best_omk was already scored by the SAME gate's
        _move_gate_margin call (the intended caller), so this makes no new
        scoring pass and no new predict_proba call beyond what building/
        reusing the seg already makes. Reads the seg's per-cell raw-p array
        (_trust_p_raw, set by _trust_soft_weights, trimmed in lockstep with
        _softwv_raw by any FACE_GATE/STICKER_GATE so the two stay aligned to
        the same admitted-cell order) rather than re-deriving cell admission
        from the raw read -- correct under any FACE_GATE/STICKER_GATE/
        VIS_GATE combination, not just the default-off canonical config.

        Returns (covT, n_cells, mean): covT = the raw-p sum, n_cells = how
        many admitted cells contributed, mean = covT/n_cells (0.0 when
        n_cells==0). Sentinel (-1.0, 0, 0.0) when unavailable: self.TRUST_SOFT
        is off (no p_trust model to read), no resolved om (best_omk unknown),
        or best_path is empty/None (the margin's occluded/no-candidate case --
        same sentinel trigger as _gate_coverage_argmax). Read-only: no
        scoring, no state mutation."""
        if AbsSegment.TRUST_SOFT is None or not best_path:
            return -1.0, 0, 0.0
        om = om_by_key.get(best_omk)
        if om is None:
            return -1.0, 0, 0.0
        perm_by_move = {n: p for n, p in MOVE_LIST}
        anchor = np.asarray(prev_arr).astype(np.int8)
        moved = anchor
        for notation in best_path:
            moved = moved[perm_by_move[notation]]
        changed = set(int(j) for j in np.nonzero(moved != anchor)[0])
        if not changed:
            return 0.0, 0, 0.0
        total, n = 0.0, 0
        for r in gm_reads:
            seg = self.seg_factory(r, om)
            if not seg.ok or seg._trust_p_raw is None:
                continue
            for k in range(seg.n):
                fn = seg._face_names[k]
                base = FACE_OFFSET.get(om.get(fn, fn))
                if base is None:
                    continue
                pos = seg._pos[k]
                if any(base + int(GRID_ROTATIONS[rr][pos]) in changed
                       for rr in range(4)):
                    total += float(seg._trust_p_raw[k])
                    n += 1
        return total, n, (total / n if n else 0.0)

    def _gate_trust_perread(self, prev_arr, gm_reads, best_omk, best_path,
                            om_by_key):
        """PER-READ variant of _gate_trust_argmax (Stage-0 window-locality
        plumbing, .3). Same conservative rotation-union
        changed-cell convention, but returns a LIST PARALLEL to gm_reads: entry i
        is read i's MEAN raw p_trust over the changed cells it admits (its
        per-FRAME covT), or None where the read contributes no admitted changed
        cell (occluded => no visible witness at that frame). MEASUREMENT-ONLY and
        ADDITIVE -- _gate_trust_argmax is unchanged (this is a separate method,
        never a changed default). Returns [None]*len(gm_reads) on the SAME
        sentinel conditions (TRUST_SOFT off / no resolved om / empty path / no
        changed cells). Read-only: no scoring pass beyond the seg segcache reuse
        _gate_trust_argmax already makes, no state mutation."""
        n_reads = len(gm_reads)
        if AbsSegment.TRUST_SOFT is None or not best_path:
            return [None] * n_reads
        om = om_by_key.get(best_omk)
        if om is None:
            return [None] * n_reads
        perm_by_move = {n: p for n, p in MOVE_LIST}
        anchor = np.asarray(prev_arr).astype(np.int8)
        moved = anchor
        for notation in best_path:
            moved = moved[perm_by_move[notation]]
        changed = set(int(j) for j in np.nonzero(moved != anchor)[0])
        if not changed:
            return [None] * n_reads
        out = []
        for r in gm_reads:
            seg = self.seg_factory(r, om)
            if not seg.ok or seg._trust_p_raw is None:
                out.append(None)
                continue
            tot, cnt = 0.0, 0
            for k in range(seg.n):
                fn = seg._face_names[k]
                base = FACE_OFFSET.get(om.get(fn, fn))
                if base is None:
                    continue
                pos = seg._pos[k]
                if any(base + int(GRID_ROTATIONS[rr][pos]) in changed
                       for rr in range(4)):
                    tot += float(seg._trust_p_raw[k])
                    cnt += 1
            out.append((tot / cnt) if cnt else None)
        return out

    def _strat_span_sample(self, span):
        """TIME-STRATIFIED quality sample. The plain
        quality rank `sorted(span, key=_nstk)[:span_subsample]` clusters on
        whichever temporal rest is best-lit — on a straddled span that is the
        clean PRE-move rest, so a within-span test (or the main commit) never
        sees the post-move majority evidence. Split the span's frame EXTENT
        at its midpoint and take the top-quality half-budget (same _nstk
        rule) from EACH side; same total budget as span_subsample, no new
        constants. Shared by span_purity_split's straddle test and the
        strat_span_sample main-commit sampler — a single source of truth so
        both paths sample identically.

        Returns the sample as a list of span items sorted by frame (the same
        shape `span` itself has)."""
        ext_mid = (span[0][0] + span[-1][0]) / 2.0
        hb = max(2, self.span_subsample // 2)

        def _top(items):
            return sorted(
                items,
                key=lambda it: (-self._nstk(it[1], 40),
                                it[2] if len(it) > 2 else 0))[:hb]

        return sorted(
            _top([it for it in span if it[0] <= ext_mid])
            + _top([it for it in span if it[0] > ext_mid]),
            key=lambda it: it[0])

    def _span_collapse(self, span, sub_frames):
        """MEASUREMENT-ONLY (pre-registered next step).
        Checks whether a span's SAMPLE (`sub_frames`, e.g. the main commit's
        `ranked[:span_subsample]` or `_strat_span_sample`'s output) has
        TEMPORALLY COLLAPSED onto one side of the span's own frame extent.
        The midpoint is
        DERIVED from the span's own frame extent — the identical
        `(span[0][0] + span[-1][0]) / 2.0` `_strat_span_sample` uses above,
        same <=/> side convention — no new constant. Collapsed = every
        sampled frame on the SAME side of that midpoint (frame <= mid is the
        low side, > mid the high side; a 1-frame sample is therefore always
        collapsed — it cannot straddle a midpoint by construction).

        Returns a dict: collapsed (bool), ext_lo/ext_hi (the span's own
        frame extent), mid (float), n (sample size), f_min/f_max (sampled
        frame range), frac_lo/frac_hi (fraction of the sample on each side —
        the softer stats flagged so a later, less-strict collapse
        definition can be considered without a re-run). Read-only: no state
        mutation, no sampling change, no new candidates."""
        ext_lo, ext_hi = span[0][0], span[-1][0]
        mid = (ext_lo + ext_hi) / 2.0
        n = len(sub_frames)
        if n == 0:
            return {"collapsed": False, "ext_lo": int(ext_lo),
                    "ext_hi": int(ext_hi), "mid": float(mid), "n": 0,
                    "f_min": None, "f_max": None, "frac_lo": 0.0,
                    "frac_hi": 0.0}
        n_lo = sum(1 for f in sub_frames if f <= mid)
        n_hi = n - n_lo
        return {
            "collapsed": bool(n_lo == 0 or n_hi == 0),
            "ext_lo": int(ext_lo), "ext_hi": int(ext_hi), "mid": float(mid),
            "n": int(n), "f_min": int(min(sub_frames)),
            "f_max": int(max(sub_frames)),
            "frac_lo": n_lo / n, "frac_hi": n_hi / n,
        }

    # Collapse sampling uses a duration rather than a bare frame count so the
    # trigger scales with capture rate. The constants preserve the historical
    # 206-frame threshold at 120 fps.
    _COLLAPSE_HIDE_FRAMES_REF = 206.0
    _COLLAPSE_HIDE_FPS_REF = 120.0
    COLLAPSE_HIDE_S = _COLLAPSE_HIDE_FRAMES_REF / _COLLAPSE_HIDE_FPS_REF

    def _collapse_strat_threshold(self):
        """Derived emptygap trigger floor in FRAMES at this solve's own
        self.fps -- see COLLAPSE_HIDE_S above."""
        return round(self.COLLAPSE_HIDE_S * self.fps)

    @staticmethod
    def _collapse_emptygap(cc):
        """Frames between the collapsed sample's farthest-out frame and the
        span's own far extent edge on whichever side the sample did NOT
        land -- "how much of the span's other half went completely unseen".
        `cc` is a collapsed `_span_collapse` result; None when not collapsed
        (frac_lo/frac_hi are only ever 0.0/1.0 or 1.0/0.0 when collapsed, by
        that function's construction, so the >= tie-break below is exact)."""
        if not cc["collapsed"] or cc["n"] == 0:
            return None
        if cc["frac_lo"] >= cc["frac_hi"]:      # collapsed low -> empty side high
            return cc["ext_hi"] - cc["f_max"]
        return cc["f_min"] - cc["ext_lo"]        # collapsed high -> empty side low

    def _span_purity(self, si, sub_frames, sub_reads, cands, om_by_key,
                     segcache):
        """Measurement-only straddled-span diagnostic.
        A span that STRADDLES a move with no event inside it has BIMODAL reads —
        the pre-move state early, the post-move state late — so its single span
        vote/fit is contaminated and the beam commits on mixed evidence. Split
        the span's sampled reads at their frame median; for each half pick the
        best-fitting state KEY among `cands` (the kept beam this span already
        scored); the span is IMPURE when the two halves disagree.

        Returns (span_index, f_start, f_end, halves_agree, key1, key2, fit1,
        fit2, fit_whole) or None (<2 reads / no scorable candidate). key* are
        the full (state_bytes, om_key) beam keys. Read-only: no new states, no
        commit change; the extra scoring touches only the <=span_subsample
        sampled reads and is reached only when self.span_purity_log is set."""
        if not cands or len(sub_frames) < 2:
            return None
        order = sorted(range(len(sub_frames)), key=lambda i: sub_frames[i])
        mid = len(order) // 2
        first, second = order[:mid], order[mid:]
        if not first or not second:
            return None

        def _best(idxs):
            best_key, best_fit = None, -1e18
            for x in cands:
                omk_c = x[1][1]
                o = om_by_key.get(omk_c)
                if o is None:
                    continue
                st = np.asarray(x[2])[None, :]
                score_total, m = 0.0, 0
                for i in idxs:
                    fi, r = sub_frames[i], sub_reads[i]
                    ck = (fi, omk_c)
                    if ck not in segcache:
                        segcache[ck] = self.seg_factory(r, o)
                    s = segcache[ck]
                    if getattr(s, "ok", False):
                        score_total += float(s.score_states(st)[0])
                        m += 1
                if m and score_total / m > best_fit:
                    best_fit, best_key = score_total / m, x[1]
            return best_key, best_fit

        k1, f1 = _best(first)
        k2, f2 = _best(second)
        _kw, fw = _best(order)
        if k1 is None or k2 is None:
            return None
        return (int(si), int(sub_frames[order[0]]), int(sub_frames[order[-1]]),
                bool(k1 == k2), k1, k2, float(f1), float(f2), float(fw))

    def _span_purity_ball(self, sub_frames, sub_reads, prev_arr, d_gap,
                          cand_omks, om_by_key):
        """BEHAVIOR test for --span-purity-split (straddled span).
        Distinct from the measurement-only _span_purity above: that argmaxes
        over the beam's SURVIVORS (`take`) and so is BLIND to a straddled
        post-move state the beam never kept.
        Here the candidate space is the depth-min(max(d_gap,1),2) move-BALL
        around the PREVIOUS anchor state (the EXACT space _move_gate_margin
        uses), so the pre- and post-move states are always in scope.

        Split the span's sampled reads at their FRAME median; for each half and
        for the whole span pick the best-fitting ball state (mean AbsSegment
        read score, argmax over the ball x cand_omks). Compares STATE bytes,
        not (state, om) keys: an om-flip with an IDENTICAL state (a mid-span
        re-grip, no turn) is PURE and must never split.

        Returns (f_start, f_end, differ, decisive, s1, s2, o1, o2, fit1, fit2,
        fit_whole, path1, path2) or None (<2 usable reads per half, or no
        scorable candidate). Read-only: builds only per-(read, om) scorers.
        """
        n = len(sub_frames)
        if n < 4:                              # need >=2 usable reads per half
            return None
        order = sorted(range(n), key=lambda i: sub_frames[i])
        mid = len(order) // 2
        first, second = order[:mid], order[mid:]
        if len(first) < 2 or len(second) < 2:
            return None
        depth = int(min(max(int(d_gap), 1), 2))
        states, paths = ball_states(np.asarray(prev_arr).astype(np.int8), depth)

        def _best(idxs):
            best_i, best_omk, best_fit = None, None, -1e18
            for omk in cand_omks:
                o = om_by_key.get(omk)
                if o is None:
                    continue
                segs = [self.seg_factory(sub_reads[i], o) for i in idxs]
                segs = [s for s in segs if getattr(s, "ok", False)]
                if not segs:
                    continue
                fits = sum(s.score_states(states) for s in segs) / len(segs)
                j = int(np.argmax(fits))
                if float(fits[j]) > best_fit:
                    best_fit, best_i, best_omk = float(fits[j]), j, omk
            return best_i, best_omk, best_fit

        i1, o1, f1 = _best(first)
        i2, o2, f2 = _best(second)
        iw, _ow, fw = _best(order)
        if i1 is None or i2 is None or iw is None:
            return None
        s1, s2 = states[i1].tobytes(), states[i2].tobytes()
        differ = (s1 != s2)                    # STATE-level: om-flip alone = PURE
        # DECISIVENESS (no new hand constant — derived from the tuple's own
        # fits): each half must fit its own best ball state better than the
        # CONTAMINATED whole-span fit by MORE than the two halves' fit spread
        # |f1-f2|. The spread is the natural scale of "how differently the two
        # halves score"; requiring each half's whole-span surplus to EXCEED it
        # means the straddle signal (each half cleaner than the mixed whole)
        # dominates half-to-half noise. A one-sided/marginal bimodality (only
        # one half beats the whole, or by less than the spread) is NOT split.
        spread = abs(f1 - f2)
        decisive = bool(differ and (f1 - fw) > spread and (f2 - fw) > spread)
        return (int(sub_frames[order[0]]), int(sub_frames[order[-1]]),
                bool(differ), decisive, s1, s2, o1, o2,
                float(f1), float(f2), float(fw), list(paths[i1]), list(paths[i2]))

    def _align_break(self, a, b):
        """Longest run of consecutive UNALIGNED frames (P(aligned) < align_thr) in
        the gap (a, b] -- the SUSTAINED lattice-break a layer MOVE produces. A rigid
        re-grip keeps every layer square, so it never sustains a break. Orthogonal
        to color. Missing frames count as aligned (1.0) so a no-coverage gap never
        fakes a break."""
        if not self.align_feat:
            return 0
        episodes = alignment_break_episodes(
            self.align_feat, int(a) + 1, int(b),
            threshold=self.align_thr, min_run=1)
        return max((hi - lo + 1 for lo, hi in episodes), default=0)

    def _align_break_expand(self, events, search_r):
        """Adaptive lattice-break: for each event find the nearest unaligned frame
        within search_r, then EXPAND the contiguous unaligned run it sits in (no
        fixed width) and return the longest. fps/move-speed invariant -- supersedes
        the hardcoded +/-align_window. Missing frames count aligned (1.0)."""
        if not self.align_feat or not events:
            return 0
        thr, best = self.align_thr, 0
        for ef in events:
            center = None
            for d in range(search_r + 1):
                for f in (ef - d, ef + d):
                    if self.align_feat.get(f, 1.0) < thr:
                        center = f; break
                if center is not None:
                    break
            if center is None:
                continue
            lo = hi = center
            while self.align_feat.get(lo - 1, 1.0) < thr:
                lo -= 1
            while self.align_feat.get(hi + 1, 1.0) < thr:
                hi += 1
            best = max(best, hi - lo + 1)
        return best

    def _align_transition_post_sample(self, span, quality_sub, span_index):
        """Return terminal-rest reads after the last sustained lattice break.

        ``None`` means the span does not meet the complete safety scope and the
        caller must preserve its existing sampler byte-for-byte.  A list means
        one independently eligible transition was found; it contains at most
        ``span_subsample`` best-quality reads strictly after the final broken
        run, ordered by frame.

        The detector reuses align_gate's own probability threshold and
        sustained-run requirement.  There is no solve-specific frame, duration,
        quality, or score threshold.
        """
        if not (self.align_transition_sample and self.align_gate
                and self.align_feat and span and span_index > 0
                and self._in_final_pass
                and self._gate_events_authoritative):
            return None
        lo, hi = int(span[0][0]), int(span[-1][0])
        # Kept events already own span boundaries.  This mechanism is only for
        # the measured hole where a transition lies strictly inside one final
        # event-free span; it never resurrects or double-counts an event.
        if any(lo < int(frame) < hi for frame in (self.rot_events or [])):
            return None
        runs = alignment_break_episodes(
            self.align_feat, lo, hi, threshold=self.align_thr,
            min_run=self.align_min_run)
        # Alignment breaks are not move counts.  A single strictly interior run
        # may timestamp two settled regions; zero/multiple/edge-touching runs
        # are ambiguous and preserve the existing sampler exactly.
        runs = [(a, b) for a, b in runs if lo < a and b < hi]
        if len(runs) != 1:
            return None
        run_lo, run_hi = runs[0]
        qc = self._span_collapse(span, [int(it[0]) for it in quality_sub])
        emptygap = self._collapse_emptygap(qc)
        if (not qc["collapsed"]
                or emptygap is None
                or emptygap < self._collapse_strat_threshold()
                or not quality_sub):
            return None
        quality_frames = [int(it[0]) for it in quality_sub]
        all_pre = all(frame < run_lo for frame in quality_frames)
        all_post = all(frame > run_hi for frame in quality_frames)
        if not (all_pre or all_post):
            return None
        pre = [it for it in span if int(it[0]) < run_lo]
        post = [it for it in span if int(it[0]) > run_hi]
        # Both sides must be real admitted rests.  Without endpoint evidence we
        # abstain by leaving the canonical sampler untouched; this guard also
        # rejects short pose/occlusion glitches.
        if len(pre) < self.minspan or len(post) < self.minspan:
            return None
        if all_post:
            return sorted(quality_sub, key=lambda it: it[0])
        post.sort(key=lambda it: (-self._nstk(it[1], 40),
                                  it[2] if len(it) > 2 else 0))
        return sorted(post[:self.span_subsample], key=lambda it: it[0])

    def _active_depth_hi(self, motions: dict, a: int, b: int, fps: float) -> int:
        """Active-motion-time move-count bound, used ONLY for ball-depth budgeting.
        Fluid triggers blend into one motion plateau with no per-move peak
        structure, and per-frame
        motion halves at 120fps, so both burst counts and frame-gap duration
        under-bound the move count. The pace-independent fact that survives: each
        move needs >= MOVE_SEC_MIN of ACTIVE motion. The window extends into
        adjacent span EDGES: reads taken during the motion ramp still vote with
        the old state, so move time hides inside the spans, not just the gap."""
        thr = self.still
        lo_f, hi_f = a, b
        while lo_f - 1 in motions and (motions[lo_f - 1] or 0) > thr:
            lo_f -= 1
        while hi_f + 1 in motions and (motions[hi_f + 1] or 0) > thr:
            hi_f += 1
        act = sum(1 for j in range(lo_f, hi_f + 1)
                  if motions.get(j) is not None and motions[j] > thr)
        return int(np.ceil(act / fps / self.MOVE_SEC_MIN)) + 1 if act else 0

    def _expected_moves(self, motions: dict, a: int, b: int) -> int:
        ms = [motions[j] for j in range(a, b + 1) if j in motions and motions[j] is not None]
        if not ms:
            return 1
        thr = self.still * 1.5
        n, inb = 0, False
        for m in ms:
            if m > thr and not inb:
                n, inb = n + 1, True
            elif m <= thr:
                inb = False
        return n

    def _scored_vec(self, prev_items, perms, ppaths, lens, base_sw, cp,
                    exp_lo, exp_hi, switch_targets, om_by_key,
                    sub_frames, sub_reads, segcache, k_top, d_gap):
        """Vectorized span transition: numpy dedup + batch scoring replacing the
        legacy per-row Python dict loop (the measured z-clip hotspot: up to
        n_prev x n_targets x n_rows ~ millions of dict ops per deep/long span).

        EXACT legacy semantics (self.vec=False path), verified bit-identical:
          - provisional score uses the verbatim float expression
            pcum - (base_sw + cp*outside + om_pen), with the ergo prior (if on)
            subtracted in a second step exactly as the legacy loop does;
          - dedup keeps the max provisional per (state, om), strict-> ties keep
            the FIRST in nested (pred, target, row) order  [lexsort is stable];
          - candidate enumeration order (the legacy cand-dict insertion order)
            is reproduced so the stable score sort breaks ties identically;
          - per-om scoring is the same mean-over-segs of score_states.
        Returns the top-k_top `scored` tuples (all that downstream consumes).
        Side effect: self._vec_fit_max = max span fit over ALL candidates (not
        just the returned top-k) — the misfit gate needs the legacy
        max(x[5] for x in scored)-over-everything semantics."""
        n_rows = len(lens)
        n_prev = len(prev_items)
        pred_keys = [k for k, _ in prev_items]
        pred_omks = [k[1] for k in pred_keys]
        pcums = [v[1] for _, v in prev_items]
        # all predecessors' depth-d balls in one block: (n_prev*n_rows, 54) int8
        all_states = np.concatenate([v[0][perms].astype(np.int8) for _, v in prev_items])
        sview = all_states.view(np.dtype((np.void, all_states.shape[1]))).ravel()
        # When the LL progress prior is armed it also needs each unique state's
        # first row.  Request index+inverse from ONE np.unique sort; the old path
        # sorted this same depth-ball twice (once here, once in the LL block).
        # np.unique's combined return values are exactly the two standalone
        # results, including sorted unique-id assignment and first occurrence.
        ll_active = bool(self.ll_prior and self._ll_prog is not None)
        sid, ll_first_idx = self._dedup_state_rows(
            sview, need_first=ll_active)
        self._vec_fit_max = None

        # OLL/PLL alg-dictionary progress prior: per candidate state a bonus
        # ll_prior * progress (0 off-trajectory), folded into prov exactly like ergo.
        # OFF (ll_prior==0 or not armed) => phi_all None => zero-cost, bit-identical.
        phi_all = None
        if ll_active:
            _prog = self._ll_prog
            pv = np.empty(len(ll_first_idx), dtype=np.float64)
            for u in range(len(ll_first_idx)):
                pv[u] = _prog.get(
                    all_states[ll_first_idx[u]].tobytes(), -1.0)
            cand_prog = pv[sid]                                    # (n_prev*n_rows,)
            pred_prog = np.repeat(
                np.array([_prog.get(v[0].tobytes(), -1.0) for _, v in prev_items]),
                n_rows)
            # reward each move that strictly ADVANCES along a known-alg solution
            # (flat ll_prior, accumulates over the flurry); off-traj/stay/regress = 0.
            phi_all = self.ll_prior * (cand_prog > pred_prog).astype(np.float64)

        # MOTION-DIRECTION amount prior (see self.dir_prior): per ball-path bonus
        # dir_prior * sum_k logP(amount(move_k) | event_k). Read-INDEPENDENT, so it is
        # immune to colour/lighting read corruption. Events for this span = the
        # rot_events within the span's frame span, paired with the path's moves in order
        # (a 1-move/1-event span = the common case). OFF => dir_phi None => bit-identical.
        dir_phi = None
        if self.dir_prior and self.dir_pred is not None:
            _evs = sorted(e for e in (self.rot_events or [])
                          if sub_frames and sub_frames[0] <= e <= sub_frames[-1])
            dir_phi = np.zeros(n_rows)
            for _ri in range(n_rows):
                _s = 0.0
                for _k, _mv in enumerate(ppaths[_ri]):
                    # pair move k with event k ONLY if it exists (no clamp: a ball's
                    # extra exploratory moves have no event -> no prior). dir_pred only
                    # holds REAL-move events (re-grips absent) -> .get is None there.
                    _ev = _evs[_k] if _k < len(_evs) else None
                    _lp = self.dir_pred.get(_ev) if _ev is not None else None
                    if _lp is not None and _mv:
                        _s += _lp[_AMT_IDX.get(_mv[1:], 0)]
                dir_phi[_ri] = self.dir_prior * _s

        def pen_vec(om_pen, burst_used):
            # verbatim legacy penalty expression (bit-identical floats)
            lo = max(0, exp_lo - burst_used)
            hi = max(lo, exp_hi - burst_used)
            under = ((lo - lens) * self.move_gate_under
                     if self.move_gate_under != 1.0 else lo - lens)
            outside = np.where(lens < lo, under,
                               np.where(lens > hi, lens - hi, 0))
            return base_sw + cp * outside + om_pen

        pen_stay = pen_vec(0.0, 0)
        pen_sw = (pen_vec(switch_targets[0][1], switch_targets[0][2])
                  if switch_targets else None)

        # by_om group order = first (pred, target) appearance of each om
        om_order, seen_om = [], set()
        for pomk in pred_omks:
            for omk in [pomk] + [t[0] for t in switch_targets if t[0] != pomk]:
                if omk not in seen_om:
                    seen_om.add(omk)
                    om_order.append(omk)

        pred_omk_set = set(pred_omks)
        class_cache = {}   # oms with identical (rows, prov) share one dedup pass

        def class_winners(omk):
            """Per unique candidate state for om `omk`: the winning (pred, row)
            global row index (max provisional; ties -> earliest in legacy nested
            order) + its provisional, in legacy cand-insertion order."""
            # Ergo makes provisional score orientation-dependent.
            ck = (omk if (omk in pred_omk_set or not switch_targets
                          or self.ergopen) else None)
            if ck in class_cache:
                return class_cache[ck]
            ergo = (self.ergopen * _ergo_table(d_gap, omk, om_by_key.get(omk))
                    if self.ergopen else None)

            def prov_for(p):
                # legacy float order: (pcum - pen) first, ergo subtracted after
                pr = pcums[p] - (pen_stay if pred_omks[p] == omk else pen_sw)
                if ergo is not None:
                    pr = pr - ergo
                if phi_all is not None:        # + alg-dictionary progress bonus (soft)
                    pr = pr + phi_all[p * n_rows:(p + 1) * n_rows]
                if dir_phi is not None:        # + read-independent motion-amount prior
                    pr = pr + dir_phi
                return pr

            if switch_targets:   # every pred targets every om (stay vs switch pen)
                rows = np.arange(n_prev * n_rows)
                prov = np.concatenate([prov_for(p) for p in range(n_prev)])
                sid_c = sid
            else:                # preds only target their own om
                pidx = [p for p in range(n_prev) if pred_omks[p] == omk]
                rows = np.concatenate(
                    [np.arange(p * n_rows, (p + 1) * n_rows) for p in pidx])
                prov = np.concatenate([prov_for(p) for p in pidx])
                sid_c = sid[rows]
            order = np.lexsort((-prov, sid_c))   # stable: prov ties keep loop order
            ssort = sid_c[order]
            first = np.empty(len(order), dtype=bool)
            first[0] = True
            first[1:] = ssort[1:] != ssort[:-1]
            win_local = order[first]                          # argmax per state
            starts = np.flatnonzero(first)
            first_occ = np.minimum.reduceat(order, starts)    # first-seen position
            ins = np.argsort(first_occ, kind="stable")        # cand-insertion order
            res = (rows[win_local[ins]], prov[win_local[ins]])
            class_cache[ck] = res
            return res

        om_blocks = []     # (omk, segs, winner_rows, prov) per scorable om group
        seg_fi_by_om = {}  # omk -> frame index per KEPT seg (for align weighting)
        for omk in om_order:
            segs, fis = [], []
            for fi, r in zip(sub_frames, sub_reads):
                # per-frame om (the re-grip fix) when enabled, else the group om
                fomk, fom = self._frame_score_om(fi, omk, om_by_key)
                ck = (fi, fomk)
                if ck not in segcache:
                    segcache[ck] = self.seg_factory(r, fom)
                seg = segcache[ck]
                if seg.ok:
                    segs.append(seg); fis.append(fi)
            if not segs:
                continue
            win, prov_w = class_winners(omk)
            om_blocks.append((omk, segs, win, prov_w))
            seg_fi_by_om[omk] = fis
        if not om_blocks:
            return []
        # prov-bound branch-and-bound: engages ONLY when (a) nobody consumes
        # _vec_fit_max (om_on_misfit / deep_on_misfit are its only readers),
        # (b) the evidence is the plain sum_seg_scores mean (seg_weight_n off),
        # (c) every active scorer is structurally nonpositive (_prune_ok), and
        # (d) there is anything to prune (more candidates than k_top).
        if (self.prune_topk and not self.om_on_misfit and not self.deep_on_misfit
                and not self.seg_weight_n and self.align_weights is None
                and phi_all is None
                and sum(len(b[3]) for b in om_blocks) > k_top
                and all(self._prune_ok(s) for _, segs, _, _ in om_blocks
                        for s in segs)):
            return self._prune_score(om_blocks, all_states, pred_keys,
                                     ppaths, n_rows, k_top)
        blocks = []        # (omk, winner_rows, sc, totals) per scored om group
        states_cache = {}  # shared candidate matrix across same-class oms
        for omk, segs, win, prov_w in om_blocks:
            states = states_cache.get(id(win))
            if states is None:
                states = all_states[win]
                states_cache[id(win)] = states
            if self.seg_weight_n:
                sc = weighted_seg_scores(segs, states, [s.n for s in segs])
            elif self.align_weights is not None:
                # soft alignment fusion: trust each frame's read ~ its P(aligned)
                w = [max(self.align_weights.get(fi, 1.0), self.align_weight_floor)
                     for fi in seg_fi_by_om[omk]]
                sc = weighted_seg_scores(segs, states, w)
            else:
                sc = sum_seg_scores(segs, states)   # == sum of score_states, exact
                sc /= len(segs)
            blocks.append((omk, win, sc, prov_w + sc))
            m = float(sc.max())
            if self._vec_fit_max is None or m > self._vec_fit_max:
                self._vec_fit_max = m
        if not blocks:
            return []
        tot_cat = np.concatenate([b[3] for b in blocks])
        top = np.argsort(-tot_cat, kind="stable")[:k_top]    # legacy stable sort
        bounds = np.cumsum([0] + [len(b[3]) for b in blocks])
        out = []
        for gi in top:
            bi = int(np.searchsorted(bounds, gi, side="right")) - 1
            omk, win, sc, tot = blocks[bi]
            li = int(gi - bounds[bi])
            row = int(win[li])
            arr = all_states[row].copy()   # copy: a view would pin the big block
            out.append((tot[li], (arr.tobytes(), omk), arr,
                        pred_keys[row // n_rows], ppaths[row % n_rows],
                        float(sc[li])))
        return out

    @staticmethod
    def _dedup_state_rows(sview, *, need_first=False):
        """Unique-state ids, optionally with first-row indices, in one sort.

        ``np.unique`` defines both outputs against the same sorted unique array.
        Asking for them together is therefore output-identical to two standalone
        calls but avoids the second O(N log N) void-row sort when the LL prior
        needs both.
        """
        if need_first:
            _uniq, first_idx, inverse = np.unique(
                sview, return_index=True, return_inverse=True)
            return inverse.ravel(), first_idx
        return np.unique(sview, return_inverse=True)[1].ravel(), None

    @staticmethod
    def _prune_ok(s):
        """True iff the branch-and-bound evidence bound (per-seg score <= 0)
        is structural for this scorer: the AbsSegment fast path (_gm present,
        the same gate sum_seg_scores uses for its grouped path) whose
        per-state score is max over gathers of -(mean | _wv-weighted sum) of
        dist-table entries — nonpositive whenever dist >= 0 and the weights
        are >= 0. Both mins are checked once per scorer (dist is (n,6), _wv
        is (n,) — trivially cheap) and cached on the instance. Anything
        foreign (no _gm — e.g. an emission scorer whose log-posterior
        "distances" can go negative and score positive) refuses the bound,
        which auto-disables pruning for the whole transition."""
        ok = getattr(s, "_prune_neg_ok", None)
        if ok is None:
            wv = getattr(s, "_wv", None)
            ok = bool(hasattr(s, "_gm") and float(np.min(s.dist)) >= 0.0
                      and (wv is None or float(np.min(wv)) >= 0.0))
            try:
                s._prune_neg_ok = ok
            except Exception:      # scorer forbids attributes: re-check next time
                pass
        return ok

    def _prune_score(self, om_blocks, all_states, pred_keys, ppaths, n_rows,
                     k_top):
        """Branch-and-bound scoring of _scored_vec's om groups: candidates are
        visited in descending-provisional order (per om group, round-robin in
        prune_block-sized blocks so the threaded sum_seg_scores fan-out stays
        amortized) and scoring stops once every remaining candidate is provably
        outside the returned top-k_top.

        EXACTNESS ARGUMENT (output-identical to scoring everything):
          * total_i = prov_i + evid_i and evid_i <= 0 structurally (the mean
            over segs of per-seg scores, each = max over gathers of -(nonneg
            combination of nonneg dist entries); guarded by _prune_ok), hence
            total_i <= prov_i for EVERY candidate.
          * `cutoff`, once set, is the k_top-th largest EXACT total among the
            candidates scored so far — monotone non-decreasing as the scored
            set grows, and never exceeding K_f, the k_top-th largest total
            over ALL candidates (the k-th largest over a subset is <= the
            k-th largest over the superset).
          * a candidate is skipped ONLY while prov_i < cutoff (strict). Then
            total_i <= prov_i < cutoff <= K_f: every skipped candidate sits
            STRICTLY below the k-th best, so the set {i : total_i >= K_f} —
            which contains the legacy top-k_top — is scored exactly.
          * TIES CANNOT BE LOST: a candidate tied with the k-th best has
            total_i = K_f <= prov_i; skipping it would need prov_i < cutoff
            <= K_f — impossible. So every member of the boundary tie group is
            scored, and tie-breaking is decided exactly as legacy does.
          * EMISSION ORDER: legacy returns np.argsort(-tot, kind="stable")
            [:k_top] over candidates in om-block concatenation order, i.e.
            sorts by (-total, concat_index). We emit the scored subset sorted
            by the same key (np.lexsort) — since legacy's whole top-k_top is
            inside the scored subset and the comparison key is identical, the
            emitted tuples are element-identical. The floats agree bitwise:
            per-row evidence is independent of chunk boundaries (each row's
            gather/mean/max touches only that row — see sum_seg_scores), the
            cross-seg accumulation order is the seg-list order in both paths,
            and total = prov + sc is the same two-operand float64 add.
        NOT preserved: self._vec_fit_max is the max span fit over the SCORED
        SUBSET only (a lower bound on the legacy all-candidates value); its
        only consumers are om_on_misfit / deep_on_misfit, and _scored_vec
        refuses to call this path when either is configured. (The driver-level
        misfit_rot / misfit_split / misfit_drop knobs consume meta's top-1
        fit, which IS exact here.)"""
        nb = len(om_blocks)
        sizes = [len(b[3]) for b in om_blocks]
        bounds = np.cumsum([0] + sizes)
        # descending-prov visit order per block (stable: prov ties keep the
        # legacy candidate order — irrelevant for correctness, deterministic
        # anyway); switch-target oms sharing one class share the array
        ord_cache, orders = {}, []
        for _, _, _, prov in om_blocks:
            o = ord_cache.get(id(prov))
            if o is None:
                o = np.argsort(-prov, kind="stable")
                ord_cache[id(prov)] = o
            orders.append(o)
        pos = [0] * nb
        done = [False] * nb
        chunks = [[] for _ in range(nb)]   # per block: (sel, sc, tot) scored
        topk_pool = None                   # current top-k_top exact totals
        cutoff = None                      # k_top-th best exact total so far
        fit_max = None
        # first pass small (establish a cutoff from every om's best provs
        # cheaply), then full blocks
        B = max(256, min(self.prune_block, 4 * k_top))
        while not all(done):
            for b in range(nb):
                if done[b]:
                    continue
                omk, segs, win, prov = om_blocks[b]
                sel = orders[b][pos[b]:pos[b] + B]
                if cutoff is not None and len(sel):
                    # prov non-increasing along sel: keep prov >= cutoff, the
                    # strictly-below tail is provably outside the top set
                    keep = int(np.searchsorted(-prov[sel], -cutoff,
                                               side="right"))
                    if keep < len(sel):
                        done[b] = True
                        sel = sel[:keep]
                pos[b] += len(sel)
                if pos[b] >= sizes[b]:
                    done[b] = True
                if not len(sel):
                    continue
                sc = sum_seg_scores(segs, all_states[win[sel]])
                sc /= len(segs)
                tot = prov[sel] + sc
                chunks[b].append((sel, sc, tot))
                m = float(sc.max())
                if fit_max is None or m > fit_max:
                    fit_max = m
                pool = (tot if topk_pool is None
                        else np.concatenate([topk_pool, tot]))
                if len(pool) >= k_top:
                    cut_i = len(pool) - k_top
                    pool = np.partition(pool, cut_i)[cut_i:]
                    cutoff = float(pool[0])
                topk_pool = pool
            B = self.prune_block
        self._vec_fit_max = fit_max
        gidx = np.concatenate([bounds[b] + sel
                               for b in range(nb) for sel, _, _ in chunks[b]])
        gsc = np.concatenate([sc for c in chunks for _, sc, _ in c])
        gtot = np.concatenate([tot for c in chunks for _, _, tot in c])
        # legacy emission key: (-total, concat index) — lexsort's last key is
        # primary; identical to the stable argsort over the full concatenation
        top = np.lexsort((gidx, -gtot))[:k_top]
        out = []
        for j in top:
            gi = int(gidx[j])
            bi = int(np.searchsorted(bounds, gi, side="right")) - 1
            omk, _, win, _ = om_blocks[bi]
            row = int(win[gi - bounds[bi]])
            arr = all_states[row].copy()   # copy: a view would pin the big block
            out.append((gtot[j], (arr.tobytes(), omk), arr,
                        pred_keys[row // n_rows], ppaths[row % n_rows],
                        float(gsc[j])))
        return out

    def _nearest_f2l_complete(self, top_a, max_depth):
        """Robust-arming helper: the nearest F2L-complete state within `max_depth`
        moves of `top_a`. Returns (state, cross_color, n_bridge_moves), or
        (top_a, None, 0) if none / ambiguous. Searches the move ball shell-by-shell
        (smallest bridge first) and only trusts a depth whose F2L-complete neighbors
        all share ONE cross color — an ambiguous shell (two different crosses equally
        near) is NOT armed, keeping the relaxation conservative. Cheap: f2l_done_color
        early-outs per orientation; run once per span until armed."""
        from detect import ll_alg_prior as _llp
        for d in range(1, max_depth + 1):
            perms, paths = _PERM_TABLES[d]
            ball = top_a[perms]                       # (N,54) cumulative depth<=d
            hits = [(ball[i], _llp.f2l_done_color(ball[i]))
                    for i in range(len(ball)) if len(paths[i]) == d]
            hits = [(s, c) for s, c in hits if c is not None]
            if hits:
                if len({c for _, c in hits}) == 1:    # unambiguous cross -> trust
                    return hits[0][0], hits[0][1], d
                return top_a, None, 0                 # ambiguous shell -> don't arm
        return top_a, None, 0

    def track_2pass(self, reads, motions, init_arr, final_arr, min_sticker_l,
                    fw_bw=False, *, _pass1_artifacts_only=True):
        """Co-operative move-gate (see move_gate). Pass 1 decodes once with every
        event-gap disabled and records, per
        gap, the reads move-vs-re-grip margin; events in a gap whose margin is below
        move_gate_margin changed NO cube state, so they are RE-GRIPS and dropped from
        rot_events entirely (no span split). Pass 2 decodes the cleaned
        event set.
        Returns pass 2's (moves, reaches_final, info)."""
        thr = self.move_gate_margin
        self._gate_events_authoritative = False
        full_events = list(self.rot_events)

        def rebind_phase_authority(events, provenance):
            """Keep the optional phase actuator on this pass's exact authority."""
            if self.intraburst_phase_audit is None:
                self.intraburst_phase_slots = ()
                return
            from detect.intraburst_motion import (
                rebind_intraburst_phase_authority,
            )
            rebound = rebind_intraburst_phase_authority(
                self.intraburst_phase_audit, events,
                provenance=provenance)
            self.intraburst_phase_audit = rebound
            self.intraburst_phase_slots = tuple(rebound.slots)

        self._fitnoise_seed = None   # z-norm: fresh per-solve estimate
        # pass 1: record margins; margin=+inf => fire on every event-gap (clear the
        # witness for clean anchors) while logging the true margin for the verdict.
        self.move_gate = True
        self.move_gate_margin = float("inf")
        self.move_gate_log = []
        saved_trace, self.om_trace = self.om_trace, (
            [] if self.om_trace is not None else None)
        # TEACHER-FORCED runs pass 1 FREE: the gate margins/verdicts are part
        # of the machinery under test — measuring them from forced anchors
        # changes the kept-event set and move-count floors, so
        # only pass 2 (the decode the records come from) is forced. No-op
        # when teacher_forced is None (byte-identical).
        saved_tf, self.teacher_forced = self.teacher_forced, None
        # Pass 1 records gate artifacts only; pass 2 is the emitting pass.
        self._in_final_pass = False
        saved_artifacts_only = self._pass1_artifacts_only
        self._pass1_artifacts_only = bool(_pass1_artifacts_only)
        try:
            with PERF_TRACE.span("decode.legacy_pass1"):
                self.track(reads, motions, init_arr, final_arr, min_sticker_l,
                           fw_bw=False)
        finally:
            self.teacher_forced = saved_tf
            self._in_final_pass = True
            self._pass1_artifacts_only = saved_artifacts_only
        log = self.move_gate_log
        gate_start_ns = PERF_TRACE.start_ns()
        # Z-NORM: freeze pass 1's WHOLE-SOLVE anchor fit-noise so the move-gate
        # verdict below AND pass 2's misfit/om_pf gates use one stable estimate
        # from frame 0 (no within-pass cold start). No-op unless a gate is active.
        if self.znorm_gates or self.log_znorm:
            self._fitnoise_seed = list(self._fitnoise_vals)
        # pass 2 common setup: gate off (margin now APPLIED, not re-measured)
        self.move_gate = False
        self.move_gate_margin = thr
        self.move_gate_log = None
        self.om_trace = saved_trace
        # z-norm derived move-gate cut (== thr when the gate is off / no clean
        # anchor); measured on the frozen whole-solve estimate.
        eff_thr = self._eff_move_gate_margin()
        if self.log_znorm:
            mu, sd, n = self._znorm_stats()
            if mu is not None and sd and sd > 0.0:
                d_mis = f"{mu - self.znorm_k_misfit * sd:.3f}"
                d_mov = f"{self.znorm_k_move * sd:.3f}"
                d_omp = f"{self.znorm_k_ompf * sd:.3f}"
                _mu, _sd = f"{mu:.3f}", f"{sd:.3f}"
            else:
                d_mis = d_mov = d_omp = "n/a"
                _mu = f"{mu:.3f}" if mu is not None else "None"
                _sd = f"{sd:.3f}" if sd is not None else "None"
            print(f"  [znorm] anchor-fit mu={_mu} sigma={_sd} n={n} | derived "
                  f"misfit {self.misfit_thr}->{d_mis} "
                  f"move_gate {thr}->{d_mov} "
                  f"om_pf {self.om_pf_margin}->{d_omp} "
                  f"(gates {'ON' if self.znorm_gates else 'off'})", flush=True)
        if self.soft_witness:
            # SOFT: every gap keeps its events (they still split spans) but the
            # move-count floor becomes the reads-weighted fractional sum -> over-
            # firing deflates instead of forcing phantoms. GLOBAL LO/HI, not per-solve.
            LO, HI = self.soft_lo_range
            self._soft_lo_map = {}
            for a, b, ev_in_gap, gm, *_ in log:
                w = (min(1.0, max(0.0, (gm - LO) / (HI - LO))) if HI > LO
                     else float(gm >= HI))
                self._soft_lo_map[(a, b)] = w * len(ev_in_gap)
            if self.log_reanchor:
                print(f"  [soft-witness] {len(log)} event-gaps; soft floor sum="
                      f"{sum(self._soft_lo_map.values()):.1f} vs hard "
                      f"{len(full_events)} events (LO={LO} HI={HI})", flush=True)
            rebind_phase_authority(
                full_events, "track-2pass-soft-witness-authority")
            PERF_TRACE.finish_child("decode.gate_build", gate_start_ns)
            with PERF_TRACE.span("decode.legacy_pass2"):
                res = self.track(
                    reads, motions, init_arr, final_arr, min_sticker_l,
                    fw_bw=fw_bw)
            self._fitnoise_seed = None
            return res
        # HARD gate: drop re-grip events entirely (no split, no witness). The color
        # verdict (gm < thr) is optionally FUSED with the orthogonal alignment
        # lattice-break verdict (ab = longest unaligned run in the gap) per
        # align_gate_mode -- alignment catches the occlusion re-grips color keeps.
        regrip = set()
        self._gate_events = []   # per-event verdicts for the dashboard / diagnosis
        for entry in log:
            a, b, ev_in_gap, gm = entry[0], entry[1], entry[2], entry[3]
            ab = entry[4] if len(entry) > 4 else -1
            cov = float(entry[5]) if len(entry) > 5 else -1.0
            color_regrip = gm < eff_thr
            # COVERAGE ADMISSIBILITY (gate_cov_min, default None = off): trust
            # a color re-grip verdict only when the margin could have SEEN a
            # change — visible∩changed coverage >= the floor. Below it the
            # margin is vacuous, so color
            # abstains; alignment keeps its independent veto per
            # align_gate_mode. cov=-1.0 (not computed) never abstains.
            if (self.gate_cov_min is not None and color_regrip
                    and 0.0 <= cov < self.gate_cov_min):
                color_regrip = False
            align_regrip = bool(self.align_gate and ab >= 0 and ab < self.align_min_run)
            # gm-protect: alignment may not veto a move COLOR is confident about.
            # ADAPTIVE: only protect where the no-dip is BORDERLINE (ab >= ab_min),
            # never a strong no-dip (ab=0 = a clean re-grip), so clear drops
            # survive while borderline over-drops can still be rescued.
            protected = (self.align_gm_protect is not None
                         and gm >= self.align_gm_protect
                         and ab >= self.align_gm_protect_ab_min)
            align_drop = align_regrip and not protected
            if self.align_gate and ab >= 0:
                if self.align_gate_mode == "align":
                    is_regrip = align_drop
                elif self.align_gate_mode == "and":
                    is_regrip = color_regrip and align_regrip
                else:                                    # "or"
                    is_regrip = color_regrip or align_drop
            else:
                is_regrip = color_regrip
            if is_regrip:
                regrip.update(ev_in_gap)
            for ef in ev_in_gap:
                self._gate_events.append({
                    "frame": int(ef), "gap": [int(a), int(b)],
                    "gm": round(float(gm), 2), "ab": int(ab),
                    "cov": round(cov, 3),
                    "color_regrip": bool(color_regrip),
                    "align_regrip": align_regrip, "dropped": bool(is_regrip)})
        kept = [e for e in full_events if e not in regrip]
        # Distinguish a valid hard-gate verdict with zero events from the
        # constructor's empty, never-run placeholder.
        self._gate_events_authoritative = True
        if self.log_reanchor:
            print(f"  [move-gate 2pass] {len(log)} event-gaps; dropped "
                  f"{len(full_events) - len(kept)}/{len(full_events)} events as "
                  f"re-grips (margin<{eff_thr}); {len(kept)} kept", flush=True)
        self.rot_events = kept
        # The pre-gate producer authority is provenance, not an actuator.  The
        # pass-2 scrub consumes only kept events, so remove every optional phase
        # owned by a dropped parent before entering that pass.
        rebind_phase_authority(kept, "track-2pass-final-kept-events")
        PERF_TRACE.finish_child("decode.gate_build", gate_start_ns)
        with PERF_TRACE.span("decode.legacy_pass2"):
            res = self.track(reads, motions, init_arr, final_arr, min_sticker_l,
                             fw_bw=fw_bw)
        self._gate_events_authoritative = False
        self._fitnoise_seed = None
        return res

    # -- main --------------------------------------------------------------
    def _tf_align_anchors(self, spans, states_mat, segcache):
        """TEACHER-FORCED read-alignment of the teacher replay to the spans.
        Move timeline stamps can lead
        the frame where a move's effect becomes VISIBLE in the reads (BLE vs
        video skew + straddles + flurries spread over spans), so stamp-window
        bucketing can charge spurious misses. Instead, align by what the reads
        show: emission[si][k] = om-
        MARGINALIZED (max over the 24 ORIENTATIONS — rotation-safe) mean span
        read-fit of GT state k (the same score_states evidence everything
        else uses), then a monotone pure-emission Viterbi (the replay only moves
        FORWARD through its own replay; no penalties, no knobs) picks each
        span's settled GT index; ties break to the SMALLEST k (never advance
        the oracle without read evidence). Returns ks (len(spans)): span si
        rests at states_mat[ks[si]]; gap si's GT moves = movegt[ks[si-1]:
        ks[si]]. A GT move is charged where the reads first REVEAL it — a
        move never revealed (tail-clipped capture) is uncharged, the same
        blind spot the free decode's --final-from-gt terminal has."""
        n_states = len(states_mat)
        n_spans = len(spans)
        emis = np.zeros((n_spans, n_states))
        for si, span in enumerate(spans):
            # same quality ranking + subsample the span loop votes with
            ranked = sorted(span, key=lambda it: (-self._nstk(it[1], 40),
                                                  it[2] if len(it) > 2 else 0))
            sub = ranked[: self.span_subsample]
            best = None
            for o in ORIENTATIONS:
                omk_o = _om_key(o)
                segs = []
                for it in sub:
                    ck = (it[0], omk_o)
                    if ck not in segcache:
                        segcache[ck] = self.seg_factory(it[1], o)
                    if segcache[ck].ok:
                        segs.append(segcache[ck])
                if not segs:
                    continue
                sc = np.zeros(n_states)
                for s in segs:
                    sc += s.score_states(states_mat)
                sc /= len(segs)
                best = sc if best is None else np.maximum(best, sc)
            if best is not None:
                emis[si] = best
        dp = np.zeros((n_spans, n_states))
        bp = np.zeros((n_spans, n_states), dtype=int)
        dp[0] = emis[0]
        for si in range(1, n_spans):
            # prefix running max of dp[si-1] with FIRST-index (smallest k)
            # argmax on ties — the monotone transition in O(n_states)
            run = np.empty(n_states)
            idx = np.empty(n_states, dtype=int)
            best_v, best_i = -np.inf, 0
            for k in range(n_states):
                if dp[si - 1][k] > best_v:
                    best_v, best_i = dp[si - 1][k], k
                run[k], idx[k] = best_v, best_i
            dp[si] = emis[si] + run
            bp[si] = idx
        ks = [0] * n_spans
        ks[-1] = int(np.argmax(dp[-1]))          # first max = smallest k
        for si in range(n_spans - 1, 0, -1):
            ks[si - 1] = int(bp[si][ks[si]])
        return ks

    def track(self, reads: dict, motions: dict, init_arr: np.ndarray,
              final_arr: np.ndarray | None, min_sticker_l: float,
              fw_bw: bool = False):
        """Returns (moves, reaches_final, info) where info = {"meta": per-span top
        candidates, "oms": orient_map active for each emitted move (feeds
        reconstruct_with_rotations), "reoriented": bool, "move_frames": per
        emitted move the frame index it is attributed to ([] when alignment
        with the simplified output is unavailable)}.

        reads: {frame: (read, cube_region_motion)}; motions: {frame: motion or None}
        for ALL frames with a cube bbox (used for burst counts in read gaps).
        final_arr None = free endpoint (no terminal constraint).
        """
        self._ll_prog = None        # per-solve: (re-)armed at F2L-complete below
        self._ll_info = None
        self._ll_inject_at = None   # per-solve: captured at arming when ll_inject on
        self._ll_inject_fired = False
        self._fitnoise_vals = []    # z-norm: per-pass confident-anchor fit sample
        spans = self.build_spans(reads, min_sticker_l)
        if not spans:
            return [], False, {"meta": [], "oms": [], "reoriented": False,
                               "move_frames": []}

        om0 = self.initial_om
        omk0 = _om_key(om0)
        om_by_key = {_om_key(o): o for o in ORIENTATIONS}
        om_by_key.setdefault(omk0, om0)
        K_eff = max(self.K, self.K_reorient) if self.allow_reorient else self.K


        # layer key = (state_bytes, om_key); value = (arr, cum, prev_key, path, om_key)
        ib = init_arr.astype(np.int8).tobytes()
        layer0 = {(ib, omk0): (init_arr, 0.0, None, [], omk0)}
        if self.allow_reorient and self.init_om_hyps > 1:
            # multi-hypothesis init: score the opening span's reads against init under
            # every orientation; seed the top-M as alternatives with their fit deficit
            # as a prior penalty (the trellis resolves om0 instead of us betting on it).
            first_reads = [it[1] for it in sorted(spans[0], key=lambda it: (-self._nstk(it[1], 40), it[2] if len(it) > 2 else 0))[: self.span_subsample]]
            fits = []
            for o in ORIENTATIONS:
                segs = [self.seg_factory(r, o) for r in first_reads]
                segs = [s for s in segs if s.ok]
                if not segs:
                    continue
                if self.seg_weight_n:
                    tot = float(weighted_seg_scores(
                        segs, init_arr[None, :], [s.n for s in segs])[0])
                else:
                    tot = float(np.mean([s.score_states(init_arr[None, :])[0]
                                         for s in segs]))
                fits.append((tot, _om_key(o)))
            if fits:
                fits.sort(key=lambda x: -x[0])
                best_fit = fits[0][0]
                for fit, omk in fits[: self.init_om_hyps]:
                    prior = fit - best_fit          # <= 0: deficit vs best hypothesis
                    key = (ib, omk)
                    if key not in layer0 or prior > layer0[key][1]:
                        layer0[key] = (init_arr, prior, None, [], omk)
        layers = [layer0]
        layer_reads = []   # per appended layer: that span's subsampled reads
        layer_gaps = []    # per appended layer: the no-read gap length before it
        layer_sub_frames = []  # per appended layer: subsampled frame indices
        # Dense evidence is a separate scoring plane.  It is built only on the
        # emitting pass when explicitly requested; None preserves the exact
        # historical scrub call path and object graph.
        _dense_active = bool(self.scrub_dense_reads and self._in_final_pass)
        scrub_evidence_reads = [] if _dense_active else None
        scrub_evidence_frames = [] if _dense_active else None
        scrub_evidence_provenance = [] if _dense_active else None
        layer_span_info = []   # per appended layer: (gap_len, exp_lo, exp_hi, rotpen_eff, d_gap)
        meta = []
        prev_end = spans[0][0][0]

        segcache = {}   # (frame, om_key) -> scorer; only beam-alive oms get built

        # TEACHER-FORCED init (None => byte-identical, nothing runs).
        # Reset per track() call so under track_2pass the FINAL pass's records
        # win (pass 1 is the margin-measure pass). Builds the full GT replay
        # trajectory (states_mat[k] = state after the first k GT moves; the
        # SAME MOVE_LIST permutations the om-Viterbi replay uses — no
        # reimplemented move application) and READ-ALIGNS it to the spans
        # (_tf_align_anchors; the stamp-skew fix). The forced om is CHAINED
        # through the production machinery (start = initial_om, advance to
        # the committed leader's om at each commit) — movegt carries no
        # orientation and a per-anchor 24-way argmax measured hysteresis-
        # The orientation channel stays the decoder's own, including its
        # rescue and switch logic.
        if self.teacher_forced is not None:
            self.tf_records = []
            self._tf_omk = omk0
            _tf_perm = {n: p for n, p in MOVE_LIST}
            _tf_unk = sorted({m for _f, m in self.teacher_forced
                              if m not in _tf_perm})
            if _tf_unk:
                raise ValueError(
                    f"teacher-forced: movegt move(s) outside the decoder's "
                    f"move space (MOVE_LIST): {_tf_unk}")
            _st = init_arr.astype(np.int8).copy()
            _traj = [_st.copy()]
            for _f, _m in self.teacher_forced:
                _st = _st[_tf_perm[_m]]
                _traj.append(_st.copy())
            self._tf_states = np.stack(_traj)
            self._tf_ks = self._tf_align_anchors(spans, self._tf_states,
                                                 segcache)

        for _si, span in enumerate(spans):
            gap_len = span[0][0] - prev_end
            gap_a = prev_end   # gap start; prev_end is reassigned below
            exp_lo, exp_hi = self._expected_interval(motions, prev_end, span[0][0], self.fps)
            # BURST-RECOVER (two-sided count bound; n_burst==0 => byte-identical):
            # content-based move-count for THIS gap from the flowrot peaks. A fast
            # flurry is short-duration (small exp_hi) but many-move; when the content
            # shows more turns than the duration window admits, widen exp_hi UP to
            # n_burst (capped at burst_max) so the long flurry path is not count-
            # penalty-suppressed. The depth raise (so the ball can EXPRESS it) and the
            # two-sided deep clamps follow below, all keyed on the same n_burst.
            n_burst = self._content_count(gap_a, span[0][0]) if self.burst_recover else 0
            # gate: only a HIGH content count (>= burst_min) is a genuine collapse;
            # below it, burst_hi==0 => every downstream use is byte-identical to OFF.
            burst_hi = (min(n_burst, self.burst_max)
                        if self.burst_recover and n_burst >= self.burst_min else 0)
            if burst_hi > exp_hi:
                exp_hi = burst_hi
            if self._burst_log is not None and self.burst_recover:
                self._burst_log.append((gap_a, span[0][0], int(n_burst),
                                        int(burst_hi), int(exp_hi), int(exp_lo)))
            bind_evs = ([(f, s, bnd) for (f, s, bnd) in self.rot_event_info
                         if prev_end < f <= span[0][0]]
                        if self.rot_bind else [])
            forced_path = None
            if self.force_gap_moves:
                fm = [m for f, mv in sorted(self.force_gap_moves.items())
                      if prev_end < f <= span[0][0] for m in mv]
                if fm:
                    forced_path = tuple(fm)
            prev_end = span[-1][0]
            if self.rot_gap_frames is not None and gap_len >= self.rot_gap_frames:
                rotpen_eff = self.rotpen_low
            else:
                rotpen_eff = self.rotpen
            ranked = sorted(span, key=lambda it: (-self._nstk(it[1], 40), it[2] if len(it) > 2 else 0))
            # MAIN SPAN SAMPLE (lever L1 / lever
            # (2)). Default: pure quality rank, top span_subsample by _nstk.
            # --strat-span-sample (bare, strat_span_sample) swaps in the
            # TIME-STRATIFIED sample (_strat_span_sample, shared with the
            # span_purity_split straddle test) GLOBALLY, for EVERY span --
            # Retained only as an explicit blanket policy because re-sampling
            # every span perturbs pass-1's gate/span structure.
            # --strat-span-sample cond fires the same stratified resample but
            # ONLY on a span whose plain quality-ranked sample has COLLAPSED
            # onto one temporal side of the span's own extent with an empty side wide
            # enough to plausibly be hiding a real move
            # (_collapse_strat_threshold(), a TIME-derived constant -- see
            # COLLAPSE_HIDE_S above _span_collapse). Both flags off => `sub`
            # is exactly `ranked[:span_subsample]`, byte-identical to before.
            quality_sub = ranked[: self.span_subsample]
            _ats = (self._align_transition_post_sample(
                span, quality_sub, _si)
                    if (self.align_transition_sample and self._in_final_pass)
                    else None)
            _ats_terminal_frames = None
            if self.strat_span_sample:
                sub = self._strat_span_sample(span)
            elif _ats is not None:
                # A candidate resulting state is meaningful only against the
                # terminal rest after the transition that produced it.  Unlike
                # global stratification, this branch is exactly inert unless
                # the existing alignment classifier reports a sustained break.
                sub = _ats
                _ats_terminal_frames = {int(it[0]) for it in _ats}
            elif (self.strat_span_sample_cond
                  and np.isfinite(self.move_gate_margin)):
                # OUTPUT-PASS ONLY: track_2pass's pass-1
                # measure sets move_gate_margin=inf; firing there perturbs the
                # recorded gate margins and changes the span structure. This
                # gate keeps the pass-1 structure fixed.
                sub = quality_sub
                _qc = self._span_collapse(span, [it[0] for it in quality_sub])
                if _qc["collapsed"]:
                    _eg = self._collapse_emptygap(_qc)
                    _thr = self._collapse_strat_threshold()
                    if _eg is not None and _eg >= _thr:
                        sub = self._strat_span_sample(span)
                        self._collapse_strat_triggers += 1
                        print(f"  [collapse-strat] span{_si} f{_qc['ext_lo']}-"
                              f"{_qc['ext_hi']} TRIGGERED emptygap={_eg} "
                              f"thr={_thr}", flush=True)
            else:
                sub = quality_sub
            sub_frames = [it[0] for it in sub]
            sub_reads = [it[1] for it in sub]
            # SPAN-COLLAPSE (next step, default None = OFF,
            # byte-identical). MEASUREMENT ONLY -- verifies the
            # hypothesis that a quality-ranked sample collapses onto one
            # temporal rest only on a straddled span.
            # No sampling change: `sub`/`sub_frames` are read, never written.
            # Armed by --verbose (alongside span_purity_log); prints EVERY
            # span's evaluation (the span-purity-ball "log EVERY evaluation"
            # idiom) so a tag with zero collapses is a positive, checkable
            # result rather than a silent one.
            if self.span_collapse_log is not None:
                _sc = self._span_collapse(span, sub_frames)
                self.span_collapse_log.append(_sc)
                if self.log_reanchor:
                    print(f"  [span-collapse] span{_si} f{_sc['ext_lo']}-"
                          f"{_sc['ext_hi']} spanCollapse={_sc['collapsed']} "
                          f"mid={_sc['mid']:.1f} n={_sc['n']} "
                          f"frameMin={_sc['f_min']} frameMax={_sc['f_max']} "
                          f"sideLo={_sc['frac_lo']:.2f} "
                          f"sideHi={_sc['frac_hi']:.2f}", flush=True)
            # ROBUST CONSENSUS: collapse the span to ONE median-denoised read over ALL
            # its frames (not the top-N vote) so the state estimate uses the abundance.
            if self.consensus_reads and len(span) > 1:
                cons = self._span_consensus(span, min_sticker_l)
                if cons:
                    sub_reads = [cons]
                    sub_frames = [span[len(span) // 2][0]]   # representative frame

            prev = layers[-1]
            # TEACHER-FORCED ORACLE PREFIX (None => byte-identical).
            # Replace the carried beam with the TRUE (state, om) singleton at
            # the preceding anchor: state = the READ-ALIGNED GT replay state
            # of the PREVIOUS span (_tf_align_anchors — the stamp-skew fix;
            # span 0 anchors at the replay start); om = the CHAINED forced om
            # (tf_om=chained: initial_om, advanced to the committed leader's
            # om at every commit — the production om channel, rescue/switch
            # included). GT moves for THIS gap = the replay segment between
            # the two anchors' aligned indices — charged where the reads
            # first REVEAL them, not where the (skewed) stamps land. The span
            # then runs the UNCHANGED production transition. layers[-1] is
            # REPLACED (not just prev_items) so the backtrack chain stays
            # walkable; the forced entry inherits the replaced leader's
            # prev_key/path, making the emitted sequence the concatenation of
            # per-span decisions. cum resets to 0.0 => the commit margin
            # below is per-decision.
            tf_gap_gt = None
            if self.teacher_forced is not None:
                # span 0 ANCHORS the chain (its aligned state is where the
                # video opens; there is no transition before it to decode a
                # pre/intra-span-0 move in — zero-length gap, no motion
                # history), so its bucket is empty by construction and any
                # GT prefix already visible at span 0 is uncharged.
                _k_prev = self._tf_ks[_si - 1] if _si > 0 else self._tf_ks[0]
                _k_here = self._tf_ks[_si]
                tf_gap_gt = [(i,) + self.teacher_forced[i]
                             for i in range(_k_prev, _k_here)]
                _tf_arr = self._tf_states[_k_prev]
                tf_key = (_tf_arr.tobytes(), self._tf_omk)
                if prev:
                    _old = max(prev.values(), key=lambda v: v[1])
                    _pk_keep, _path_keep = _old[2], _old[3]
                    # PROBE (microscope forensics): the replaced free-running
                    # leader's om + whether its state equals the oracle —
                    # discriminates om-choice artifacts from alignment/
                    # bucketing artifacts per missed gap in the tf record.
                    self._tf_leader_omk = _old[4]
                    self._tf_state_eq = bool(
                        _old[0].tobytes() == _tf_arr.tobytes())
                else:
                    _pk_keep, _path_keep = None, []
                    self._tf_leader_omk = None
                    self._tf_state_eq = None
                layers[-1] = {tf_key: (_tf_arr.copy(), 0.0, _pk_keep,
                                       _path_keep, self._tf_omk)}
                prev = layers[-1]
            # CLEAN-REST GATE: choose this span's effective count-penalty and ball
            # depth. When clean_gate is off, both equal the global self.countpen /
            # self.d (bit-identical legacy). When on, the relaxed global values are
            # used ONLY if the PRECEDING anchor span is clean (its recorded top
            # read-fit >= clean_fit_thr); otherwise (re-grip / occluded predecessor,
            # the decoy-prone case) fall back to the strict legacy count + depth.
            eff_countpen, eff_d = self.countpen, self.d
            if self.clean_gate:
                prev_fit = (meta[-1][2][0][2]
                            if (meta and meta[-1][2]) else None)
                clean = (prev_fit is None) or (prev_fit >= self.clean_fit_thr)
                # FAST-BURST gate: the relaxed count is meant for two turns inside
                # one motion burst (a SHORT no-read gap). A re-grip is a LONG
                # reorientation gap with the same poor-ish fit. Require
                # gap_len<=clean_max_gap
                # so only genuine fast-adjacent bursts get the relaxed penalty.
                short = gap_len <= self.clean_max_gap
                if not (clean and short):
                    eff_countpen, eff_d = (self.clean_strict_countpen,
                                           self.clean_strict_d)
            # SCALING (full-solve budget): om-switch targets are enumerated ONLY on
            # LONG gaps (pause-rotate-pause — rotations physically cannot happen
            # mid-trigger), dropping the x24 candidate factor from almost every span.
            # Deep gaps (a fluidly-executed trigger = motion bursts > d) get extra
            # ball depth (cap 4) but expand only the TOP few predecessors; such gaps
            # are bounded by moves/trigger_len per solve.
            long_gap = (self.rot_gap_frames is not None
                        and gap_len >= self.rot_gap_frames)
            # depth from the duration interval's UPPER bound — expansion is one
            # numpy gather over precomputed permutation tables (live-latency budget),
            # so depth is near-free; deeper gaps narrow predecessors to bound rows.
            dur_est = round(gap_len / (self.MOVE_SEC_TYP * self.fps))
            if not self.deep_gap:
                d_gap = eff_d
            elif self.deep_gap == "hi":
                # fluid triggers: moves OVERLAP, so per-move duration shrinks and
                # the duration estimate under-counts exactly where depth is needed
                # (z1 diag: GT needs 4 in a 30f gap, dur_est gave 3, exp_hi=5 —
                # GT unreachable yet wins by +5.15 once aligned). Budget depth
                # from the burst/duration UPPER bound; decoy admission is paid
                # for by the count penalty, not by capping reachability.
                hi_depth = exp_hi
                if self.active_hi:
                    # depth bound only — the count-penalty window keeps charging
                    # for length, so extra depth admits GT without inviting
                    # penalty-free long decoys (see active_hi note above).
                    hi_depth = max(hi_depth, self._active_depth_hi(
                        motions, gap_a, span[0][0], self.fps))
                d_gap = int(min(self.burst_max if burst_hi else 4,
                                max(eff_d, hi_depth, burst_hi)))
            else:
                d_gap = int(min(self.burst_max if burst_hi else 4,
                                max(eff_d, min(dur_est, exp_hi), burst_hi)))
            prev_items = sorted(prev.items(), key=lambda kv: -kv[1][1])
            if d_gap >= 4:
                prev_items = prev_items[:8]
            elif d_gap == 3:
                prev_items = prev_items[:16]
            # om-event prior (TASK-2 lever, default OFF): a whole-cube reorient
            # is exactly what a rotation EVENT marks. With om_event_prior set,
            # an om switch in a span whose gap (widened by om_event_window) holds
            # NO event pays an EXTRA penalty (inertia: a grip cannot reorient
            # every span); a witnessed switch optionally gets om_event_relief off
            # its rotpen. Pure additive cost on the switch penalty -> the beam
            # still CAN switch, it just needs evidence. Bit-identical when None.
            om_pen_adj = 0.0
            if self.om_event_prior is not None and self.rot_events:
                w = self.om_event_window
                witnessed = any(gap_a - w <= ef <= span[0][0] + w
                                for ef in self.rot_events)
                if witnessed:
                    om_pen_adj = -float(self.om_event_relief)
                else:
                    om_pen_adj = float(self.om_event_prior)
            switch_targets = ([(_om_key(o), rotpen_eff + om_pen_adj, 1)
                               for o in ORIENTATIONS]
                              if (self.allow_reorient and long_gap) else [])
            # OM TIMELINE override: the cube's camera-frame om at this span is
            # GIVEN by the gyro/IMU/CV orientation channel. Offer ONLY it as a
            # switch target at zero penalty (gyro/CV-confirmed reorientations are
            # free) and restrict the kept beam to it after scoring. The move-ball
            # still searches STATE freely under that om. Non-empty switch_targets
            # also skips the misfit-rescue (the timeline IS the om answer).
            tl_omk = self._span_timeline_om(span)
            if tl_omk is not None:
                switch_targets = [(tl_omk, 0.0, 1)]
            # LATTICE-CONSISTENCY GATE (default None = OFF, no-op): drop the
            # span's off-lattice (mid-turn) voting reads before scoring. Filters
            # sub_frames/sub_reads to the reads that fit some reachable settled
            # state under some candidate om; the scoring below is otherwise
            # untouched. Requires a trustworthy anchor (see knob comment).
            if self.lattice_gate is not None and sub_reads:
                keep = self._lattice_keep(sub_frames, sub_reads, prev_items,
                                          d_gap, switch_targets, om_by_key)
                if len(keep) < len(sub_reads):
                    sub_frames = [sub_frames[i] for i in keep]
                    sub_reads = [sub_reads[i] for i in keep]
            # SOFT-FUSION per-frame om (CV-only, no gyro/timeline): resolve each
            # frame's om from the reads vs the beam's anchor state (confidence-
            # gated), so the scorer fuses the re-grip and falls back to the span om
            # where occluded/ambiguous. Refreshed per span; default-off (no-op).
            self._pf_om = {}
            if (self.om_per_frame and self.om_per_frame_from_reads and prev_items
                    and (span[-1][0] - span[0][0]) >= self.om_pf_min_frames):
                self._pf_om = self._resolve_perframe_om(
                    sub_frames, sub_reads, prev_items[0][1][0], om_by_key,
                    segcache,
                    # the span's own move budget (om_pf_ball="gap" derives the
                    # ball depth from it): exp_hi here is already burst/event-
                    # widened — the same ceiling the count penalty charges
                    # against — and d_gap is this span's transition-ball depth
                    # (the beam's actual reach). Ignored by fixed/stock modes.
                    exp_hi=exp_hi, d_gap=d_gap)
            # CO-OPERATIVE MOVE-GATE (driven by track_2pass): when the gap's event(s)
            # raised the move floor, score the reads move-vs-re-grip margin against the
            # state BEFORE this gap (prev_items[0][1][0] -- uncontaminated by the move
            # about to be committed) over the candidate oms (prev om + reads-resolved
            # _pf_om). Pass 1 records it (move_gate_log) and clears the witness for
            # clean anchors; the verdict is APPLIED in pass 2 by dropping re-grip events
            # from rot_events (no split, no witness). A real move -> margin high -> kept.
            if (self.move_gate and exp_lo > 0 and prev_items and sub_reads
                    and any(gap_a < ef <= span[0][0] for ef in self.rot_events)):
                ev_in_gap = [ef for ef in self.rot_events if gap_a < ef <= span[0][0]]
                cand_omks = {prev_items[0][0][1]} | set(self._pf_om.values())
                gm_reads = sub_reads
                if self.move_gate_allframes:
                    # denoise the verdict over EVERY settled frame of the rest
                    # (quality-ranked), not just the 16-frame vote subsample
                    rk = sorted(span, key=lambda it: (-self._nstk(it[1], 40),
                                                      it[2] if len(it) > 2 else 0))
                    cap = self.move_gate_max_frames
                    gm_reads = [it[1] for it in (rk[:cap] if cap > 0 else rk)]
                # GATE-COVERAGE: the argmax (which om, which
                # move-path) is needed only for the per-argmax coverage below, so
                # it's read off the SAME margin call when that instrumentation is
                # wanted -- never a second scoring pass (margins are identical
                # either way by construction).
                _want_cov = self.log_reanchor or self.gate_cov_min is not None
                if _want_cov:
                    gm, best_omk, best_path = self._move_gate_margin(
                        prev_items[0][1][0], gm_reads, d_gap, cand_omks,
                        om_by_key, return_argmax=True)
                else:
                    gm = self._move_gate_margin(prev_items[0][1][0], gm_reads,
                                                d_gap, cand_omks, om_by_key)
                ab = -1
                if self.align_gate and ev_in_gap:
                    if self.align_expand:
                        ab = self._align_break_expand(ev_in_gap, self.align_search_r)
                    else:
                        w = self.align_window
                        ab = self._align_break(min(ev_in_gap) - w, max(ev_in_gap) + w)
                # Record how much of the move-changed sticker set the reads can
                # actually SEE -- a margin computed where this is ~0 is vacuous.
                # cov_frac/n_vis = the UNION metric (over ALL depth-1
                # moves x ALL candidate oms). cov_argmax/n_changed_argmax = the
                # per-ARGMAX-move refinement (over the specific winning move
                # under the specific winning om). The union can be misleading
                # because it never narrows to what the margin actually chose.
                # Computed under verbose (log_reanchor, measurement) OR when the
                # coverage-admissible gate is armed (gate_cov_min, behavior --
                # pass 2 needs the extended tuple). Both off => byte-identical
                # (no call, 5-tuple).
                cov_frac, n_vis = -1.0, 0
                cov_argmax, n_changed_argmax, argmax_move_str = -1.0, 0, "-"
                # covT: the SUM of RAW p_trust (pre-floor
                # TrustNumpy probability -- see TrellisTracker._gate_trust_argmax)
                # over the argmax move's changed cells under the resolved
                # (best_omk) om -- read off the SAME argmax _gate_coverage_argmax
                # already resolved, never a second scoring pass. Sentinel
                # (-1.0, 0, 0.0) when self.TRUST_SOFT is off (no p_trust model)
                # -- same gate (_want_cov) as covA, both off => byte-identical.
                covT_sum, covT_n, covT_mean = -1.0, 0, 0.0
                if _want_cov:
                    cov_frac, n_vis = self._gate_coverage(
                        prev_items[0][1][0], gm_reads, cand_omks, om_by_key)
                    cov_argmax, n_changed_argmax, _ = self._gate_coverage_argmax(
                        prev_items[0][1][0], gm_reads, best_omk, best_path,
                        om_by_key)
                    argmax_move_str = " ".join(best_path) if best_path else "-"
                    covT_sum, covT_n, covT_mean = self._gate_trust_argmax(
                        prev_items[0][1][0], gm_reads, best_omk, best_path,
                        om_by_key)
                if self.move_gate_log is not None:
                    if _want_cov:
                        self.move_gate_log.append(
                            (gap_a, span[0][0], ev_in_gap, float(gm), int(ab),
                             float(cov_frac), int(n_vis), float(cov_argmax),
                             int(n_changed_argmax), argmax_move_str,
                             float(covT_sum), int(covT_n), float(covT_mean)))
                    else:
                        self.move_gate_log.append((gap_a, span[0][0], ev_in_gap,
                                                   float(gm), int(ab)))
                _mg = self._eff_move_gate_margin()   # inf during pass-1 measure
                if gm < _mg:
                    if self.log_reanchor:
                        print(f"  [move-gate] span{_si} f{gap_a}-{span[0][0]} "
                              f"margin={gm:+.2f}<{_mg} REGRIP cov={cov_frac:.2f}"
                              f" covA={cov_argmax:.2f} mvA={argmax_move_str}"
                              f" covT={covT_sum:.2f}/{covT_n}"
                              f" (mean={covT_mean:.2f})"
                              f" -> exp_lo {exp_lo}->0", flush=True)
                    exp_lo = 0
            cp = eff_countpen * (self.countpen_deep if exp_lo > eff_d else 1.0)
            perms, ppaths = _PERM_TABLES[d_gap]
            lens = _PERM_LENS[d_gap]
            base_sw = np.where(lens > 0, self.switchpen, 0.0) + self.lam * lens

            def run_transition(sw_targets, d_over=None, prev_over=None,
                               k_over=None):
                """One span transition + scoring pass with the given om-switch
                targets. Returns [(cum, key, arr, prev_key, path, span_fit), ...]
                sorted by cumulative score; span_fit = the per-span mean AbsSegment
                read score (the 's' term meta records). d_over/prev_over re-run
                the same span at a different ball depth / predecessor set (the
                misfit-gated depth retry); with both None this is bit-identical
                to the un-parameterized version."""
                if d_over is None:
                    t_perms, t_ppaths, t_lens, t_base = perms, ppaths, lens, base_sw
                    t_d = d_gap
                else:
                    t_d = d_over
                    t_perms, t_ppaths = _PERM_TABLES[t_d]
                    t_lens = _PERM_LENS[t_d]
                    t_base = (np.where(t_lens > 0, self.switchpen, 0.0)
                              + self.lam * t_lens)
                t_prev = prev_items if prev_over is None else prev_over
                if self.vec and not bind_evs and forced_path is None:
                    # vectorized transition (exact-equivalent; see _scored_vec).
                    # downstream only reads scored[:K_eff] / [:5] / top-2 —
                    # except the om quota, which needs per-om tails (k_top 4096).
                    # binding gaps take the non-vec path: the identity penalty
                    # is per-(row, om) which _scored_vec's row vector can't carry.
                    k_top = (k_over if k_over is not None
                             else (4096 if self.om_beam_quota else max(K_eff, 5)))
                    return self._scored_vec(
                        t_prev, t_perms, t_ppaths, t_lens, t_base, cp,
                        exp_lo, exp_hi, sw_targets, om_by_key,
                        sub_frames, sub_reads, segcache, k_top, t_d)

                def bind_pen(omk):
                    """rot_bind x (#events in this gap with NO compatible move
                    in the row's path), per row; cached per (depth, om, events)."""
                    ck = (t_d, omk, tuple(bind_evs))
                    if ck not in self._bind_cache:
                        om = om_by_key.get(omk) or {}
                        pen = np.zeros(t_nrows)
                        for (_f, slot, bnd) in bind_evs:
                            rr = self.rot_bind_rots.get(slot)
                            comp = compatible_moves(
                                om.get(slot, slot), bnd,
                                (rr,) if rr is not None else (0, 1, 2, 3))
                            pen += np.array(
                                [0.0 if any(mm in comp for mm in t_ppaths[i])
                                 else 1.0 for i in range(t_nrows)])
                        self._bind_cache[ck] = self.rot_bind * pen
                    return self._bind_cache[ck]
                t_nrows = len(t_lens)
                cand = {}
                for (pb, pomk), (parr, pcum, _, _, _) in t_prev:
                    exp_states = parr[t_perms].astype(np.int8)   # (N,54) one gather
                    sb = exp_states.tobytes()
                    row_bytes = [sb[i * 54:(i + 1) * 54] for i in range(t_nrows)]
                    targets = [(pomk, 0.0, 0)] + [t for t in sw_targets if t[0] != pomk]
                    for omk, om_pen, burst_used in targets:
                        lo = max(0, exp_lo - burst_used)
                        hi = max(lo, exp_hi - burst_used)
                        under = ((lo - t_lens) * self.move_gate_under
                                 if self.move_gate_under != 1.0 else lo - t_lens)
                        outside = np.where(t_lens < lo, under,
                                           np.where(t_lens > hi, t_lens - hi, 0))
                        prov = pcum - (t_base + cp * outside + om_pen)
                        if forced_path is not None:
                            # restrict to the forced sequence (and stay, so a
                            # mis-anchored force cannot strand the beam)
                            fmask = np.array(
                                [t_ppaths[i] == forced_path or t_lens[i] == 0
                                 for i in range(t_nrows)])
                            prov = np.where(fmask, prov, -1e15)
                        if bind_evs:
                            prov = prov - bind_pen(omk)
                        if self.ergopen:
                            prov = prov - self.ergopen * _ergo_table(
                                t_d, omk, om_by_key.get(omk))
                        for i in range(t_nrows):
                            key = (row_bytes[i], omk)
                            pv = prov[i]
                            cur = cand.get(key)
                            if cur is None or pv > cur[3]:
                                cand[key] = (exp_states[i], (pb, pomk), t_ppaths[i], pv)
                # score grouped by om (one scorer set per orientation, vectorized states)
                by_om = {}
                for key in cand:
                    by_om.setdefault(key[1], []).append(key)
                scored = []
                for omk, keys in by_om.items():
                    segs = []
                    for fi, r in zip(sub_frames, sub_reads):
                        fomk, fom = self._frame_score_om(fi, omk, om_by_key)
                        ck = (fi, fomk)
                        if ck not in segcache:
                            segcache[ck] = self.seg_factory(r, fom)
                        segs.append(segcache[ck])
                    segs = [s for s in segs if s.ok]
                    if not segs:
                        continue
                    states = np.stack([cand[k][0] for k in keys])
                    if self.seg_weight_n:
                        sc = weighted_seg_scores(segs, states,
                                                 [s.n for s in segs])
                    else:
                        sc = np.zeros(len(keys))
                        for s in segs:
                            sc += s.score_states(states)
                        sc /= len(segs)
                    for j, k in enumerate(keys):
                        a, pk, path, prov = cand[k]
                        scored.append((prov + float(sc[j]), k, a, pk, path, float(sc[j])))
                scored.sort(key=lambda x: -x[0])
                return scored

            # SINGLE-OM DEEP RE-ANCHOR (the cheap, non-polluting form): when the
            # orientation channel (om_timeline -> tl_omk) GIVES the post-grip om
            # for a re-grip span, run the MAIN transition deep under THAT ONE om.
            # A re-grip can leave the true state beyond the normal d_gap ball;
            # depth om_rescue_deep reaches farther, and because
            # only the externally named orientation is in play, the pool is not
            # flooded with 24-orientation decoys. No separate rescue is needed.
            _main_deep = (self.om_rescue_deep
                          if (tl_omk is not None and self.om_rescue_deep
                              and self.regrip_frames is not None
                              and any(gap_a - 5 <= rf <= span[-1][0] + 5
                                      for rf in self.regrip_frames))
                          else None)
            if _main_deep and self.deep_count_bound:
                # TWO-SIDED: clamp DOWN to the count bound (over-fire control) but,
                # under burst_recover, allow UP to the content count n_burst so a
                # high-content re-grip flurry can re-anchor deep enough to GENERATE
                # the true N-move sequence. n_burst==0 (off) => identical clamp.
                _main_deep = min(max(_main_deep, burst_hi), max(1, int(round(exp_hi))))
            scored = run_transition(switch_targets, d_over=_main_deep)
            # MISFIT-GATED OM ENUMERATION: fluid (un-paused) rotations hide in
            # SHORT gaps, which the long-gap gate above never enumerates — the om
            # goes stale and the span's reads fit nothing in the carried oms. If
            # even the BEST candidate's own span fit is below misfit_thr, re-run
            # this span's transition with all-24 om switch targets at FULL
            # self.rotpen (same count/switch penalties — hindsight passes carrying
            # weaker costs drift switches onto noise, measured) and keep whichever
            # layer scores better. `not switch_targets` skips spans where the
            # long-gap path already enumerated all 24; at most ONE re-run per span.
            span_fit_max = (self._vec_fit_max if self.vec
                            else (max(x[5] for x in scored) if scored else None))
            # z-norm derived misfit floor (== self.misfit_thr when the gate is
            # off); one value for both the short-gap and long-gap misfit gates.
            eff_misfit = self._eff_misfit_thr()
            om_rescued = False
            om_extra = 0
            if (self.om_on_misfit and self.allow_reorient and not switch_targets
                    and scored and span_fit_max is not None
                    and span_fit_max < eff_misfit):
                rp_resc = (self.rotpen if self.rotpen_misfit is None
                           else self.rotpen_misfit)
                # om-event prior also gates the MISFIT-RESCUE switch (same lever):
                # an unwitnessed misfit span should not invent a reorientation.
                # deep re-anchor scope: at a re-grip span the state jumps a few
                # moves, so the rescue runs a DEEP ball there to GENERATE the true
                # state (regrip_frames bounds the cost to actual re-grips).
                deep_here = self.om_rescue_deep
                if deep_here and self.deep_count_bound:
                    # two-sided (see _main_deep above): n_burst==0 => identical clamp
                    deep_here = min(max(deep_here, burst_hi), max(1, int(round(exp_hi))))
                if deep_here and self.regrip_frames is not None:
                    lo_f, hi_f = gap_a - 5, span[-1][0] + 5
                    if not any(lo_f <= rf <= hi_f for rf in self.regrip_frames):
                        deep_here = 0          # not a re-grip span: stay shallow
                rescue_oms = ORIENTATIONS
                # topk pre-rank prunes the oms (set --om-rescue-topk 0 for all 24
                # if the deep re-anchor needs the post-grip om the pbest-based
                # rank may bury).
                if self.om_rescue_topk and prev_items:
                    # cheap pre-rank: fit the best previous state under each om
                    # (centers don't move, so this names the cube's orientation),
                    # then run the move-ball transition only for the top-k.
                    pbest = prev_items[0][1][0]
                    om_rank = []
                    for o in ORIENTATIONS:
                        omk = _om_key(o)
                        ss = []
                        for fi, r in zip(sub_frames, sub_reads):
                            ck = (fi, omk)
                            if ck not in segcache:
                                segcache[ck] = self.seg_factory(r, o)
                            if segcache[ck].ok:
                                ss.append(segcache[ck])
                        if ss:
                            om_rank.append((float(np.mean(
                                [s.score_states(pbest[None, :])[0] for s in ss])), o))
                    om_rank.sort(key=lambda t: -t[0])
                    rescue_oms = [o for _, o in om_rank[:self.om_rescue_topk]]
                rescored = run_transition(
                    [(_om_key(o), rp_resc + om_pen_adj, 1) for o in rescue_oms],
                    k_over=(K_eff * 4 if self.om_rescue_extra else None),
                    d_over=(deep_here or None))
                decisive = False
                if rescored and self.om_decisive_gap and span_fit_max is not None:
                    bf = max(rescored, key=lambda x: x[5])
                    if bf[5] - span_fit_max >= self.om_decisive_gap:
                        # reads DECISIVELY prefer another om (e.g. a transient tilt
                        # has settled back): take its best-fitting candidate over
                        # the accumulated rotation penalty.
                        scored = [bf] + [x for x in rescored if x is not bf]
                        om_rescued = decisive = True
                if not decisive and rescored and rescored[0][0] > scored[0][0]:
                    scored = rescored
                    om_rescued = True
                elif not decisive and rescored and self.om_rescue_extra:
                    # losing rescue: union its top om-SWITCHING candidates into
                    # the layer (see om_rescue_extra knob comment)
                    seen = {x[1] for x in scored[:K_eff]}
                    extras = [x for x in rescored
                              if x[1][1] != x[3][1] and x[1] not in seen]
                    extras = extras[: self.om_rescue_extra]
                    if extras:
                        om_extra = len(extras)
                        scored = sorted(scored[:K_eff] + extras,
                                        key=lambda x: -x[0])
                    if __debug__ and _GATE_DEBUG:
                        print(f"GATE span@{span[0][0]} union extras={om_extra} "
                              f"fit_max={span_fit_max:.2f}", flush=True)
                if self.log_reanchor and deep_here:
                    bf_fit = (max((x[5] for x in rescored), default=None)
                              if rescored else None)
                    verdict = ("decisive" if decisive
                               else "cum" if om_rescued else "rejected")
                    bfs = f"{bf_fit:.1f}" if bf_fit is not None else "none"
                    print(f"  [REANCHOR] span{_si} f{span[0][0]}-{span[-1][0]} "
                          f"deep_om={self.om_rescue_deep} fit_before={span_fit_max:.1f}"
                          f" deep_best={bfs} cand={len(rescored)} -> {verdict}",
                          flush=True)
            # MISFIT-GATED DEPTH RETRY (see knob comment): one re-run at depth 4,
            # accepted only on a strict span-fit improvement (>= deep_fit_gain) —
            # the signature of a depth-starved fluid trigger, not of a noisy span.
            # Skipped when the om retry already rescued the span (<=1 rescue/span).
            # exp_lo >= 1 = motion evidence required: a fluid trigger IS motion, so
            # a zero-burst gap physically cannot hide one (measured: y1's 9f
            # zero-burst gap drew a 4-move overfit rescue on degraded reads,
            # 1.00 -> -0.33; the depth-4 ball's 46k states always contain a
            # better-fitting decoy for junk reads — selection bias).
            if (self.deep_on_misfit and not om_rescued and d_gap < 4
                    and exp_lo >= 1
                    and scored and span_fit_max is not None
                    and span_fit_max < eff_misfit):
                deep_scored = run_transition(switch_targets, d_over=4,
                                             prev_over=prev_items[:8])
                deep_fit = (self._vec_fit_max if self.vec
                            else (max((x[5] for x in deep_scored), default=None)
                                  if deep_scored else None))
                if (deep_scored and deep_fit is not None
                        and deep_fit > span_fit_max + self.deep_fit_gain):
                    scored = deep_scored
            if not scored:
                continue
            # IN-BURST CO-COMMIT (wide/slice): a SEPARATE homogeneous pass offering
            # the single-rotation neighbor oms at burst_used=0 (rotation rides the
            # MOVE's burst). Its top om-SWITCHING candidates are UNIONED into the
            # layer (the om_rescue_extra pattern) — it NEVER fills the main
            # switch_targets, so misfit-rescue still fires and _scored_vec stays
            # homogeneous. A wide only wins if its candidate out-scores the face-only
            # path downstream (evidence pays `widepen`); on a face-only solve the
            # unioned wide candidates lose => no regression. Skipped when a timeline
            # om is given (switch_targets non-empty). Default off => untouched.
            if (self.co_commit and self.allow_reorient and not switch_targets
                    and prev_items):
                wp = self.widepen if self.widepen is not None else self.rotpen
                cc_targets = [(nk, wp + om_pen_adj, 0)
                              for nk in single_rot_neighbors(prev_items[0][0][1])]
                if cc_targets:
                    cc_scored = run_transition(cc_targets, k_over=K_eff * 4)
                    base = scored[: K_eff + om_extra]      # current kept set
                    seen = {x[1] for x in base}
                    cc_extras = [x for x in cc_scored
                                 if x[1][1] != x[3][1] and x[1] not in seen]
                    cc_extras = cc_extras[: self.cocommit_extra]
                    if cc_extras:
                        scored = sorted(base + cc_extras, key=lambda x: -x[0])
                        om_extra += len(cc_extras)
            # OM TIMELINE: keep only candidates in the timeline om (the cube's
            # given camera-frame orientation for this span); the move-ball already
            # offered it as the zero-cost switch target. Fall back to the full
            # ranking if somehow none survive (never observed) so a span is never
            # emptied. State search under the fixed om is untouched.
            if tl_omk is not None:
                scored = [x for x in scored if x[1][1] == tl_omk] or scored
            # contrastive arbitration: if the top-2 are near-tied DIFFERENT states in
            # the same om, let the differing stickers alone decide the order
            if (self.contrast_margin > 0 and len(scored) >= 2
                    and scored[0][0] - scored[1][0] < self.contrast_margin
                    and scored[0][1][0] != scored[1][1][0]
                    and scored[0][1][1] == scored[1][1][1]):
                omk_t = scored[0][1][1]
                wins = 0
                for fi, r in zip(sub_frames, sub_reads):
                    fomk, fom = self._frame_score_om(fi, omk_t, om_by_key)
                    seg = segcache.get((fi, fomk)) or self.seg_factory(r, fom)
                    if not seg.ok:
                        continue
                    sa, sb, nd = seg.contrast(scored[0][2], scored[1][2])
                    if nd:
                        wins += 1 if sa >= sb else -1
                if wins < 0:
                    scored[0], scored[1] = scored[1], scored[0]
            # CONFIDENT-READ STATE RE-ANCHOR (Mechanism A: PROMOTE). Off by default
            # (anchor_resync) => byte-identical. The trellis stores ABSOLUTE state
            # per span and backtracks prev_key, so ONE wrong tie-break here rewrites
            # every later layer's state. When THIS span is a confident anchor AND the
            # confident-read span-fit DECISIVELY prefers a CARRIED non-leader state
            # over the cum-leader (margin >= anchor_promote_gap, above the 0.2-1.4
            # look-alike decoy band), re-weight every carried entry's cum by
            # anchor_weight*(its_fit - leader_fit) and re-sort. State analog of
            # om_decisive_gap: only re-ranks states the beam ALREADY holds (the
            # correct lineage was alive but out-ranked) -- no new states, backtrack
            # chains intact.
            if self.anchor_resync and scored:
                fit_leader = scored[0][5]
                fit_best = max(x[5] for x in scored)
                # cheap gate FIRST (rare): does a carried non-leader decisively
                # out-fit the leader, inside the correct fit band? Only then pay for
                # the coverage / stillness / per-cell-confidence test.
                if (fit_best >= self.anchor_fit_thr
                        and fit_best - fit_leader >= self.anchor_promote_gap):
                    nfaces = max((self._nfaces(r, min_sticker_l)
                                  for r in sub_reads), default=0)
                    nstk = max((self._nstk(r, min_sticker_l)
                                for r in sub_reads), default=0)
                    mv = [motions.get(fi) for fi in sub_frames]
                    mv = [m for m in mv if m is not None]
                    span_motion = float(np.mean(mv)) if mv else 0.0
                    # mean per-cell extraction conf over the span (occluded/finger/
                    # glare cells are already low-conf via the occ-mask, so this is
                    # the "how clearly do we actually SEE this" gate). cell_conf is
                    # om-invariant, so read it off the winning-om segs (cached).
                    win_omk = scored[0][1][1]
                    cv = []
                    for fi, r in zip(sub_frames, sub_reads):
                        fomk, fom = self._frame_score_om(fi, win_omk, om_by_key)
                        seg = segcache.get((fi, fomk)) or self.seg_factory(r, fom)
                        cf = getattr(seg, "_confs", None)
                        if seg.ok and cf is not None and len(cf):
                            cv.append(float(cf.mean()))
                    mean_cellconf = float(np.mean(cv)) if cv else 0.0
                    if (nfaces >= self.anchor_min_faces
                            and nstk >= self.anchor_min_nstk
                            and span_motion <= self.still / 2.0
                            and mean_cellconf >= self.anchor_cellconf_thr
                            and len(span) >= self.anchor_min_frames):
                        scored = sorted(
                            [(c + self.anchor_weight * (s - fit_leader),
                              k, a, pk, p, s)
                             for (c, k, a, pk, p, s) in scored],
                            key=lambda x: -x[0])
                        if self.log_anchor:
                            print(f"  [ANCHOR] span f{span[0][0]}-{span[-1][0]} "
                                  f"PROMOTE faces={nfaces} nstk={nstk} "
                                  f"motion={span_motion:.1f} "
                                  f"cellconf={mean_cellconf:.2f} "
                                  f"fit_leader={fit_leader:.1f} "
                                  f"fit_best={fit_best:.1f} "
                                  f"gap={fit_best - fit_leader:.1f} -> "
                                  f"top_fit={scored[0][5]:.1f}", flush=True)
            # CO-OPERATIVE TRELLIS<->MOTION GATE — reads-veto on phantom move
            # insertion (see __init__ coop_gate). THIS is the single commit point:
            # `scored` is final (every rescue/retry/co-commit/anchor pass done),
            # `take`/layers.append below fixes the layer leader = the committed
            # move. Veto only when the top candidate would COMMIT a move AND a
            # carried (stay) candidate's read-fit decisively beats the best move
            # candidate's read-fit (the count/ball forced the move over the reads).
            # Then keep ONLY the (stay) candidates (refunding the under-count
            # penalty cp*exp_lo the count floor charged them, so the carried cum is
            # honest); the move is not inserted. x[4]=path (empty => stay),
            # x[5]=span read-fit. Default off => byte-identical.
            if self.coop_gate and scored and scored[0][4]:
                stays = [x for x in scored if not x[4]]
                moves = [x for x in scored if x[4]]
                if stays and moves:
                    best_stay_fit = max(x[5] for x in stays)
                    best_move_fit = max(x[5] for x in moves)
                    if best_stay_fit - best_move_fit >= self.coop_gate_margin:
                        vetoed = " ".join(scored[0][4])     # the move being dropped
                        refund = cp * max(0, exp_lo)
                        scored = sorted(
                            [(c + refund, k, a, pk, p, s)
                             for (c, k, a, pk, p, s) in stays],
                            key=lambda x: -x[0])
                        om_extra = 0
                        if self.coop_gate_log:
                            print(f"  [coop-gate] span{_si} f{span[0][0]}-"
                                  f"{span[-1][0]} VETO move {vetoed!r}: "
                                  f"read_fit(stay)={best_stay_fit:.2f} - "
                                  f"read_fit(best_move)={best_move_fit:.2f} = "
                                  f"{best_stay_fit - best_move_fit:.2f} >= "
                                  f"{self.coop_gate_margin} -> stay", flush=True)
            # Z-NORM: harvest this span's read-fit into the per-solve anchor
            # fit-noise sample if it is a confident anchor (same coverage test as
            # anchor_resync's PROMOTE gate). `scored` is final here. Only when a
            # gate (or the diagnostic) is active => zero cost / byte-identical off.
            if (self.znorm_gates or self.log_znorm) and scored:
                self._note_anchor_fit(span, sub_frames, sub_reads, scored,
                                      motions, om_by_key, segcache, min_sticker_l)
            take = scored[: K_eff + om_extra]
            if self.om_beam_quota:
                # per-om quota (see knob comment): top-q of each om present,
                # counted over the whole ranking so majority oms (already in
                # `take`) consume their quota without adding cells.
                seen_keys = {x[1] for x in take}
                per = {}
                for x in scored:
                    omk = x[1][1]
                    n = per.get(omk, 0)
                    if n >= self.om_beam_quota:
                        continue
                    per[omk] = n + 1
                    if x[1] not in seen_keys:
                        take.append(x)
            if self.om_trace is not None:
                # READ-ONLY diagnostic: best per-span read-fit under EVERY om at
                # this span's subsampled reads (does each orientation have any
                # support, or are the reads off-lattice for all?). Uses the same
                # scorers already cached for beam oms; builds the rest fresh.
                best_state = scored[0][2]   # winning candidate's expanded state
                om_fits = {}
                for o in ORIENTATIONS:
                    omk_o = _om_key(o)
                    segs = []
                    for fi, r in zip(sub_frames, sub_reads):
                        ck = (fi, omk_o)
                        if ck not in segcache:
                            segcache[ck] = self.seg_factory(r, o)
                        s = segcache[ck]
                        if s.ok:
                            segs.append(s)
                    if not segs:
                        continue
                    # fit the WINNING state under this om (apples-to-apples: the
                    # read evidence for the same physical state seen from each om)
                    sc = float(np.mean([s.score_states(best_state[None, :])[0]
                                        for s in segs]))
                    om_fits[omk_o] = round(sc, 2)
                carried_oms = sorted({k[1] for k in
                                      ({x[1] for x in take})})
                win_omk = scored[0][1][1]
                # best achievable span-fit per om over the TOP-K candidate states
                # in that om (what the beam could read if it carried that om)
                bestfit_per_om = {}
                for x in scored[:200]:
                    o = x[1][1]
                    if o not in bestfit_per_om or x[5] > bestfit_per_om[o]:
                        bestfit_per_om[o] = round(x[5], 2)
                self.om_trace.append(dict(
                    span=(span[0][0], span[-1][0]),
                    gap_len=gap_len, long_gap=long_gap,
                    win_om=win_omk, win_path=list(scored[0][4]),
                    win_fit=round(scored[0][5], 2),
                    om0_carried=(omk0 in carried_oms),
                    om0_in_top200=(omk0 in bestfit_per_om),
                    om0_bestfit=bestfit_per_om.get(omk0),
                    win_om_bestfit=bestfit_per_om.get(win_omk),
                    n_carried_oms=len(carried_oms),
                    fit_winstate_by_om=om_fits,
                    bestfit_per_om=bestfit_per_om,
                ))
            # SPAN-PURITY. Two independent, default-off paths;
            # both off => the whole block is skipped => byte-identical.
            #  (1) MEASUREMENT (span_purity_log, --verbose): the beam-based
            #      _span_purity diagnostic (semantics unchanged — existing tests
            #      lock its 9-tuple + read-only-ness). `take` is final here.
            #  (2) SPLIT behavior (span_purity_split, --span-purity-split): the
            #      BALL-based _span_purity_ball straddle test — its candidate
            #      space is the move-ball around the PREVIOUS anchor, NOT the
            #      beam survivors (which are blind to a never-kept post-move
            #      state; the reproducer span17 is silent in the beam log). A
            #      DECISIVELY impure span is re-run as two sub-spans split at
            # the frame median (option (a) of the sketch).
            _purity_committed = False
            if self.span_purity_log is not None or self.span_purity_split:
                if self.span_purity_log is not None:
                    sp = self._span_purity(_si, sub_frames, sub_reads, take,
                                           om_by_key, segcache)
                    if sp is not None:
                        self.span_purity_log.append(sp)
                        if not sp[3] and self.log_reanchor:
                            def _t(k):
                                return (f"{k[1]}#{hash(k[0]) & 0xfff:03x}"
                                        if k else "none")
                            print(f"  [span-purity] span{_si} f{sp[1]}-{sp[2]} "
                                  f"IMPURE half1={_t(sp[4])} half2={_t(sp[5])} "
                                  f"fits={sp[6]:.1f}/{sp[7]:.1f}/{sp[8]:.1f}",
                                  flush=True)
                if (self.span_purity_split and prev_items
                        and forced_path is None and len(sub_frames) >= 4):
                    _cand_omks = ({prev_items[0][0][1]}
                                  | set(self._pf_om.values()))
                    # TIME-STRATIFIED purity sample (root cause:
                    # the quality-ranked span sample at the top of this loop
                    # clusters on the clean PRE-move rest, so a straddled
                    # span's post-move tail is never sampled and NO within-span
                    # test can see the straddle). Shared construction
                    # (_strat_span_sample, L1): split the span's
                    # frame EXTENT at its midpoint and take the top-quality
                    # half-budget from EACH side, same total budget
                    # (span_subsample), same _nstk quality rule, no new
                    # constants. Only this default-off split path sees the
                    # stratified sample (unless strat_span_sample is also on).
                    _strat = self._strat_span_sample(span)
                    sp_frames = [it[0] for it in _strat]
                    sp_reads = [it[1] for it in _strat]
                    spb = (self._span_purity_ball(
                        sp_frames, sp_reads, prev_items[0][1][0], d_gap,
                        _cand_omks, om_by_key)
                        if len(sp_frames) >= 4 else None)
                    if self.log_reanchor:
                        # Probe EVERY ball evaluation, not just decisive ones
                        # (the reproducer span17 never surfaced
                        # because pure/None results were silent). A span with
                        # NO line at all failed the enclosing eligibility gate.
                        if spb is None:
                            print(f"  [span-purity-ball] span{_si} None "
                                  f"(<2 reads/half or no scorable candidate)",
                                  flush=True)
                        else:
                            print(f"  [span-purity-ball] span{_si} "
                                  f"f{spb[0]}-{spb[1]} differ={spb[2]} "
                                  f"decisive={spb[3]} mv1={len(spb[11])} "
                                  f"mv2={len(spb[12])} fits={spb[8]:.1f}/"
                                  f"{spb[9]:.1f}/{spb[10]:.1f}", flush=True)
                    if spb is not None and spb[3]:      # DECISIVE straddle
                        # frame-median partition of the SAME stratified sample
                        # the ball test judged (pre- this re-split the
                        # quality-ranked sample, which never held the tail)
                        _ord = sorted(range(len(sp_frames)),
                                      key=lambda i: sp_frames[i])
                        _mid = len(_ord) // 2
                        fr1 = [sp_frames[i] for i in _ord[:_mid]]
                        rd1 = [sp_reads[i] for i in _ord[:_mid]]
                        fr2 = [sp_frames[i] for i in _ord[_mid:]]
                        rd2 = [sp_reads[i] for i in _ord[_mid:]]
                        unsplit_cum, unsplit_fit = scored[0][0], scored[0][5]
                        decline, folded = None, None
                        # Re-run each half through the EXACT span transition:
                        # run_transition reads the closure's sub_frames/sub_reads
                        # at CALL time, so rebind them per half (restore in
                        # finally) — invents no new scoring. Half 1 transitions
                        # from the previous beam; half 2 CHAINS from half 1's
                        # committed beam (prev_over), so the buried move lands in
                        # half 2. d_over=_main_deep matches the main transition.
                        _sf, _sr = sub_frames, sub_reads
                        try:
                            sub_frames, sub_reads = fr1, rd1
                            scored1 = run_transition(switch_targets,
                                                     d_over=_main_deep)
                            if not scored1:
                                decline = "half1-empty"
                            else:
                                inter_take = scored1[:K_eff]
                                inter_layer = {k: (a, c, pk, p, k[1]) for
                                               c, k, a, pk, p, _ in inter_take}
                                inter_items = sorted(inter_layer.items(),
                                                     key=lambda kv: -kv[1][1])
                                sub_frames, sub_reads = fr2, rd2
                                scored2 = run_transition(
                                    switch_targets, d_over=_main_deep,
                                    prev_over=inter_items)
                                if not scored2:
                                    decline = "half2-empty"
                        finally:
                            sub_frames, sub_reads = _sf, _sr
                        if decline is None:
                            # FOLD the intermediate beam into ONE layer: each
                            # half-2 candidate's prev_key names a half-1 state;
                            # compose full_path = half1_path + half2_path and
                            # re-point prev_key to the ORIGINAL predecessor (a
                            # key in layers[-1]). Keeps exactly ONE layer per
                            # span, so every downstream index (meta / move_layer
                            # / om-Viterbi / fw-bw) is structurally unchanged and
                            # the backtrack emits BOTH moves via the composed
                            # path — move_frames/oms stay consistent for notation.
                            folded = []
                            for (c2, k2, a2, pk2, p2, s2) in scored2[:K_eff]:
                                inter = inter_layer.get(pk2)
                                if inter is None:
                                    continue
                                _a1, _c1, pk1_orig, p1, _o1 = inter
                                folded.append((c2, k2, a2, pk1_orig,
                                               list(p1) + list(p2), s2))
                            if not folded:
                                decline = "fold-empty"
                            else:
                                folded.sort(key=lambda x: -x[0])
                                if not folded[0][4]:
                                    decline = "no-move"    # stay-stay: no gain
                                elif (folded[0][5] < unsplit_fit
                                      or folded[0][0] < unsplit_cum):
                                    # degrade guard (fair, both per-frame means /
                                    # same accumulator): the recovered post-move
                                    # state must read at least as cleanly in its
                                    # half as the unsplit winner did whole-span,
                                    # and its cumulative must not be worse.
                                    decline = "degraded"
                        if decline is None:
                            # COMMIT the split: replace this span's take/scored
                            # with the folded post-move beam. The committed layer
                            # state is the SECOND-half (post-move) state, so its
                            # supporting reads are the second half — keep them for
                            # the notation/om-Viterbi passes.
                            take = folded[:K_eff]
                            scored = folded
                            sub_frames, sub_reads = fr2, rd2
                            _purity_committed = True
                            if self.log_reanchor:
                                def _t2(sb, omk):
                                    return f"{omk}#{hash(sb) & 0xfff:03x}"
                                _mv = " ".join(folded[0][4]) or "(stay)"
                                print(f"  [span-purity] span{_si} SPLIT @f"
                                      f"{fr2[0]}: {_t2(spb[4], spb[6])}->"
                                      f"{_t2(spb[5], spb[7])} +[{_mv}]",
                                      flush=True)
                        elif self.log_reanchor:
                            print(f"  [span-purity] span{_si} split-declined "
                                  f"{decline}", flush=True)
            if self.log_reanchor:
                rg = (self.regrip_frames is not None
                      and any(gap_a - 5 <= rf <= span[-1][0] + 5
                              for rf in self.regrip_frames))
                top = scored[0]
                print(f"  span{_si:3d}/{len(spans)} f{span[0][0]}-{span[-1][0]} "
                      f"om={top[1][1]} fit={top[5]:.1f} mv={len(top[4])} "
                      f"{'<regrip>' if rg else ''}", flush=True)
            # COVT-ON-COMMIT (next step). MEASUREMENT
            # ONLY (default None = OFF, byte-identical / zero cost). The
            # existing covT reader (_gate_trust_argmax, called only at the
            # move-gate ~L4634) scores a re-grip VERDICT and only fires when
            # a rotation event falls inside the gap -- it never scores the
            # move the span scorer actually commits here.
            # This reuses _gate_trust_argmax UNCHANGED against the COMMITTED
            # candidate `take[0]`: its own predecessor state (looked up by
            # its prev_key `pk` in `prev` == layers[-1] -- NOT assumed to be
            # the beam leader), its own path (multi-move aware --
            # _gate_trust_argmax already accumulates the full path's perms
            # cumulatively before diffing, same convention as best_path), and
            # its own resolved om. `take` is FINAL here (every rescue/split/
            # veto/anchor pass done -- the same point the tf_records commit
            # above reads take[0][4] from) and `sub_reads` is the exact read
            # set that produced it (the span-purity-split reassignment to
            # fr2/rd2 is already applied by this point). self.seg_factory(r,
            # om) is the ONLY scoring primitive _gate_trust_argmax calls, and
            # every read in `sub_reads` was already scored under every kept
            # candidate's om by run_transition's `by_om` loop above -- so
            # this is always a segcache hit, never a new predict_proba call
            # (verified: test_track_covt_commit_no_extra_predict_proba_calls).
            # The runner-up (take[1]) falls out the same way when present --
            # no separate machinery. Read-only: take/prev/sub_reads/om_by_key
            # are read, never written.
            if self.covt_commit_log is not None and take:
                def _commit_covt(cand):
                    _pv = prev.get(cand[3])
                    if _pv is None:
                        return -1.0, 0, 0.0
                    return self._gate_trust_argmax(
                        _pv[0], sub_reads, cand[1][1], cand[4], om_by_key)
                covT_c_sum, covT_c_n, covT_c_mean = _commit_covt(take[0])
                _cc_entry = {
                    "span": _si, "f_a": int(span[0][0]), "f_b": int(span[-1][0]),
                    "mv": list(take[0][4]), "covT": float(covT_c_sum),
                    "n": int(covT_c_n), "mean": float(covT_c_mean),
                }
                if len(take) >= 2:
                    _ru_sum, _ru_n, _ru_mean = _commit_covt(take[1])
                    _cc_entry["runnerup"] = {
                        "mv": list(take[1][4]), "covT": float(_ru_sum),
                        "n": int(_ru_n), "mean": float(_ru_mean),
                    }
                self.covt_commit_log.append(_cc_entry)
                if self.log_reanchor:
                    _cc_mv = " ".join(_cc_entry["mv"]) or "(stay)"
                    _cc_line = (
                        f"  [covT-commit] span{_si} f{_cc_entry['f_a']}-"
                        f"{_cc_entry['f_b']} mv={_cc_mv} covT={covT_c_sum:.2f}"
                        f"/{covT_c_n} (mean={covT_c_mean:.2f})")
                    if "runnerup" in _cc_entry:
                        _ru = _cc_entry["runnerup"]
                        _ru_mv = " ".join(_ru["mv"]) or "(stay)"
                        _cc_line += (
                            f" runnerup=(mv={_ru_mv} covT={_ru['covT']:.2f}"
                            f"/{_ru['n']} mean={_ru['mean']:.2f})")
                    print(_cc_line, flush=True)
            layers.append({k: (a, c, pk, p, k[1])
                           for c, k, a, pk, p, _ in take})
            layer_reads.append(sub_reads)
            layer_sub_frames.append(sub_frames)
            if _dense_active:
                _dense_fallback = None
                if self.consensus_reads:
                    _dense_fallback = "consensus-synthetic-read"
                elif self.lattice_gate is not None:
                    _dense_fallback = "lattice-filtered-sample"
                elif _purity_committed:
                    _dense_fallback = "folded-terminal-sample"
                (_scrub_frames, _scrub_reads,
                 _scrub_prov) = scrub_read_stream(
                    span, sub_frames, sub_reads, dense=True,
                    align_feat=self.align_feat, align_thr=self.align_thr,
                    align_min_run=self.align_min_run, minspan=self.minspan,
                    fallback_reason=_dense_fallback)
                _scrub_prov["span"] = int(_si)
                scrub_evidence_reads.append(_scrub_reads)
                scrub_evidence_frames.append(_scrub_frames)
                scrub_evidence_provenance.append(_scrub_prov)
            if (_ats_terminal_frames
                    and sub_frames
                    and set(int(f) for f in sub_frames).issubset(
                        _ats_terminal_frames)):
                self._align_transition_fires += 1
                self._align_transition_spans.add(int(_si))
                if self.log_reanchor:
                    print(f"  [align-transition-sample] span{_si} "
                          f"f{span[0][0]}-{span[-1][0]} "
                          f"post_reads={len(sub_frames)}", flush=True)
            layer_gaps.append(gap_len)
            layer_span_info.append((gap_len, exp_lo, exp_hi, rotpen_eff, d_gap))
            meta.append((span[0][0], span[-1][0],
                         [(" ".join(p) or "(stay)", k[1], round(s, 2))
                          for c, k, a, pk, p, s in scored[:5]]))
            # TEACHER-FORCED per-gap verdict (None => skipped). take
            # is final here (every rescue/split/veto pass done), so take[0][4]
            # IS this gap's committed move sequence from the oracle prefix.
            # margin = top-vs-runner-up CUMULATIVE at the commit — the decision
            # margin the span loop already computes; the singleton prefix's
            # cum=0.0 makes it a clean per-decision number (inf = unopposed).
            if tf_gap_gt is not None:
                _tf_m = (scored[0][0] - scored[1][0]) if len(scored) >= 2 \
                    else float("inf")
                self.tf_records.append({
                    "gap": _si, "f_a": int(gap_a), "f_b": int(span[-1][0]),
                    "gt": [m for _i, _f, m in tf_gap_gt],
                    "gt_idx": [_i for _i, _f, _m in tf_gap_gt],
                    "got": list(take[0][4]), "margin": float(_tf_m),
                    # probe fields (forensics; report lines don't read them)
                    "omk": self._tf_omk,
                    "leader_omk": getattr(self, "_tf_leader_omk", None),
                    "state_eq": getattr(self, "_tf_state_eq", None)})
                # advance the CHAINED forced om to the committed leader's om
                # (the production om channel under forcing; see the init note)
                self._tf_omk = take[0][1][1]

            # Arm the OLL/PLL alg-dictionary prior ONCE: the moment the top beam
            # state first reaches F2L-complete (read-fit gated), build the last-layer
            # progress potential from it (detect/ll_alg_prior.build_potential).
            if self.ll_prior and self._ll_prog is None and take:
                _tc, _tk, top_a, _tp, _tpth, top_fit = take[0]
                if self.ll_gate_fit is None or top_fit >= self.ll_gate_fit:
                    from detect import ll_alg_prior as _llp
                    cross = _llp.f2l_done_color(top_a)
                    arm_a, arm_bridge = top_a, 0
                    # ROBUST ARMING (see self.robust_arm): exact F2L-complete is a
                    # binary cliff; if the top state isn't exactly F2L-complete but is
                    # within robust_arm moves of one, arm from that nearest neighbor so
                    # a small read-driven drift still unlocks the LL trajectory.
                    if (cross is None and self.robust_arm
                            and _si >= self.robust_arm_after_frac * len(spans)):
                        arm_a, cross, arm_bridge = self._nearest_f2l_complete(
                            top_a, int(self.robust_arm))
                    if cross is not None:
                        prog, pinfo = _llp.build_potential(arm_a, cross)
                        if prog:
                            self._ll_prog, self._ll_info = prog, pinfo
                            _rb = "" if arm_bridge == 0 else f" [robust-arm bridge={arm_bridge}]"
                            print(f"  [ll-prior] ARMED @span{_si} fit={top_fit:.1f} "
                                  f"cross={cross} OLL={pinfo['oll']} PLL={pinfo['pll']} "
                                  f"chains={pinfo['n_chains']} states={pinfo['n_states']}{_rb}",
                                  flush=True)
                            # EXPRESSION half: capture the armed F2L-entry STATE +
                            # cross. The actual alg is chosen at the terminal by
                            # OBSERVED-STATE fit over the comprehensive chains (so a
                            # wrong/non-standard execution is recovered from what the
                            # cube did, not assumed canonical). See ll_inject terminal.
                            if self.ll_inject:
                                self._ll_inject_at = (len(layers) - 1, _tk,
                                                      arm_a.copy(), cross)
                                print(f"  [ll-inject] armed F2L-entry @layer"
                                      f"{len(layers) - 1} (chain chosen at terminal "
                                      f"by read-fit over comprehensive DB)", flush=True)

        # PASS-1 ARTIFACT FAST PATH. track_2pass consumes only the gate log and
        # complete fit-noise sample produced by the forward loop above; its
        # pass-1 terminal selection, backtrack, OM refinement, and output tuple
        # were discarded.  The latch is private to track_2pass's first call, so
        # direct track() and the emitting pass 2 retain their full behavior.
        if self._pass1_artifacts_only:
            return [], False, {}



        # FW-BW SMOOTHER: bidirectional Viterbi + joint posterior combination.
        # When fw_bw=True and final_arr is known, run a backward pass from `final`
        # through the spans in reverse. Each backward step is fully vectorized:
        # all beam states are expanded in one numpy gather, scored in one
        # sum_seg_scores call, then deduplicated by max provisional cost.
        # The joint posterior gamma[L] = fwd_cum[L] + bwd_cum[L] picks the path
        # where forward evidence + backward future-cost is maximized — correcting
        # mid-solve errors the forward-only Viterbi committed.
        fw_bw_override_key = None  # set when FW-BW smoothing selects a path
        if fw_bw and final_arr is not None and len(layers) >= 2:
            final_i8_bw = final_arr.astype(np.int8)
            fb_bw = final_i8_bw.tobytes()
            inv_map_bw = {"": "'", "'": "", "2": "2"}

            N_layers = len(layers) - 1  # number of span layers (layers[0]=init)

            # --- Backward beam: {(sb, omk): (arr, bwd_cum)} ---
            # bwd_cum[L] = best cost from (state,om) at L to reach final.
            # Initialization at layer N: start at the actual final state, cost=0.
            # Seed with oms that appear in the last forward layer (for joint coverage).
            bwd_beam = [{} for _ in range(N_layers + 1)]
            seed_oms = (set(k[1] for k in layers[N_layers]) | {omk0})
            for omk_seed in seed_oms:
                bwd_beam[N_layers][(fb_bw, omk_seed)] = (final_i8_bw.copy(), 0.0)

            # --- Backward Viterbi loop ---
            # At each step from layer L+1 to L:
            #   1. Stack beam arrays into (K, 54) matrix
            #   2. Expand: (K, n_rows, 54) = arr[:, perms] — all K ball expansions at once
            #   3. Score the (K, n_rows) flattened states against span L's reads
            #   4. Compute per-state total = bwd_cum + transition_cost + read_score
            #   5. Deduplicate by max total per unique (state_bytes, om) pair
            #   6. Prune to K_bw best entries
            K_bw = max(K_eff // 2, 16)  # small beam: correctness wins over coverage here

            for L in range(N_layers - 1, -1, -1):
                if not bwd_beam[L + 1]:
                    break

                gap_len_bw, exp_lo_bw, exp_hi_bw, rotpen_eff_bw, d_gap_bw = (
                    layer_span_info[L] if L < len(layer_span_info) else
                    (1, 0, 10**9, self.rotpen, self.d))
                s_reads = layer_reads[L] if L < len(layer_reads) else []
                s_frames = layer_sub_frames[L] if L < len(layer_sub_frames) else []

                self._pf_om = {}
                perms_bw, ppaths_bw = _PERM_TABLES[d_gap_bw]
                lens_bw = _PERM_LENS[d_gap_bw]
                base_sw_bw = (np.where(lens_bw > 0, self.switchpen, 0.0)
                              + self.lam * lens_bw)
                outside_bw = np.where(lens_bw < exp_lo_bw,
                                      exp_lo_bw - lens_bw,
                                      np.where(lens_bw > exp_hi_bw,
                                               lens_bw - exp_hi_bw, 0))
                trans_cost_bw = base_sw_bw + self.countpen * outside_bw  # (n_rows_bw,)

                # Group backward beam by om (same-om backward transitions only:
                # om switching in backward is expensive and rarely helps since the
                # forward pass already handles om switching forward).
                by_om_bw = {}
                for key_nxt, (arr_nxt, c_nxt) in bwd_beam[L + 1].items():
                    by_om_bw.setdefault(key_nxt[1], []).append((arr_nxt, c_nxt))

                bwd_cand_new = {}  # (sb, omk): (arr, best_total)
                for omk_bw, entries_bw in by_om_bw.items():
                    fom_bw = om_by_key.get(omk_bw)
                    if fom_bw is None:
                        continue

                    # Stack: (K_om, 54)
                    arrs_nxt = np.stack([a for a, _ in entries_bw], axis=0).astype(np.int8)
                    cums_nxt = np.array([c for _, c in entries_bw], dtype=np.float64)

                    # Expand: (K_om, n_rows_bw, 54) -> (K_om * n_rows_bw, 54)
                    expanded = arrs_nxt[:, perms_bw].reshape(-1, 54).astype(np.int8)
                    # provisional cost for each (i, j): cums_nxt[i] - trans_cost_bw[j]
                    prov = (cums_nxt[:, None] - trans_cost_bw[None, :]).ravel()  # (K_om*n_rows_bw,)

                    # Pre-prune by provisional cost before scoring
                    n_total = len(prov)
                    prune_k = min(K_bw * 4, n_total)
                    if n_total > prune_k:
                        top_idx = np.argpartition(-prov, prune_k)[:prune_k]
                        expanded = expanded[top_idx]
                        prov = prov[top_idx]

                    # Score against span L's reads
                    if s_reads and s_frames:
                        segs_bw = []
                        for fi, r in zip(s_frames, s_reads):
                            ck = (fi, omk_bw)
                            if ck not in segcache:
                                segcache[ck] = self.seg_factory(r, fom_bw)
                            segs_bw.append(segcache[ck])
                        segs_bw = [s for s in segs_bw if s.ok]
                        if segs_bw:
                            rd_sc_bw = sum_seg_scores(segs_bw, expanded) / len(segs_bw)
                        else:
                            rd_sc_bw = np.zeros(len(expanded))
                    else:
                        rd_sc_bw = np.zeros(len(expanded))

                    totals_bw = prov + rd_sc_bw

                    # Deduplicate: for each unique state bytes, keep max total
                    for j in range(len(expanded)):
                        sb_pred = expanded[j].tobytes()
                        ckey = (sb_pred, omk_bw)
                        v = float(totals_bw[j])
                        cur = bwd_cand_new.get(ckey)
                        if cur is None or v > cur[1]:
                            bwd_cand_new[ckey] = (expanded[j].copy(), v)

                # Prune to K_bw best
                if len(bwd_cand_new) > K_bw:
                    sorted_bw = sorted(bwd_cand_new.items(), key=lambda kv: -kv[1][1])
                    bwd_cand_new = dict(sorted_bw[:K_bw])
                bwd_beam[L] = bwd_cand_new

            # --- Joint posterior: soft join with djoin-ball bridge ---
            # gamma[L] = fwd_cum[L][s_f, om_f] + bwd_cum[L][s_b, om_b] - bridge_cost
            # where s_f and s_b may differ by up to djoin moves (bridge cost penalizes).
            # Exact match (djoin=0) is tried first; then d=1,2 soft bridges if no exact.
            # This handles om mismatch and small state divergence between beams.
            best_joint_score = None
            best_joint_layer = None
            best_joint_key = None
            djoin = min(self.d, 2)  # max soft-bridge depth

            # Build bwd state byte -> (key, bwd_cum) for fast lookup at each layer
            for L in range(1, N_layers + 1):
                fwd_layer = layers[L]
                bwd_layer = bwd_beam[L]
                if not bwd_layer:
                    continue
                # Index backward beam by state bytes (ignore om for matching)
                bwd_by_sb = {}
                for bk, bentry in bwd_layer.items():
                    sb = bk[0]
                    bc = bentry[1]
                    if sb not in bwd_by_sb or bc > bwd_by_sb[sb][1]:
                        bwd_by_sb[sb] = (bk, bc)

                for key in fwd_layer:
                    fwd_c = fwd_layer[key][1]
                    sb_f = key[0]

                    # Try exact match first (cost 0)
                    bwd_hit = bwd_by_sb.get(sb_f)
                    if bwd_hit is not None:
                        bwd_c = bwd_hit[1]
                        if bwd_c > -1e17:
                            gamma = fwd_c + bwd_c
                            if best_joint_score is None or gamma > best_joint_score:
                                best_joint_score = gamma
                                best_joint_layer = L
                                best_joint_key = key

                    # Try soft bridge (djoin moves connecting fwd state to bwd state)
                    if djoin > 0:
                        arr_f = fwd_layer[key][0]
                        jperms, jpaths = _PERM_TABLES[djoin]
                        jlens = _PERM_LENS[djoin]
                        bridged_states = arr_f[jperms]  # (n_jrows, 54)
                        for ji in range(len(jpaths)):
                            sb_b = bridged_states[ji].tobytes()
                            bwd_hit = bwd_by_sb.get(sb_b)
                            if bwd_hit is None:
                                continue
                            bwd_c = bwd_hit[1]
                            if bwd_c <= -1e17:
                                continue
                            bridge_cost = self.lam * jlens[ji] + 0.5 * (jlens[ji] > 0)
                            gamma = fwd_c + bwd_c - bridge_cost
                            if best_joint_score is None or gamma > best_joint_score:
                                best_joint_score = gamma
                                best_joint_layer = L
                                best_joint_key = key

            if best_joint_key is not None:
                print(f"  [fw-bw] joint-best at layer {best_joint_layer}/{N_layers} "
                      f"key=({best_joint_key[1]}) gamma={best_joint_score:.2f}",
                      flush=True)

                # Tail bridge from best_joint_key to final
                sb_best = best_joint_key[0]
                a_best = layers[best_joint_layer][best_joint_key][0]
                if sb_best == fb_bw:
                    tail_bridge = []
                    tail_reaches = True
                else:
                    # Build final ball for bridge
                    if self.dfinal in _PERM_TABLES:
                        fperms_t, fpaths_t = _PERM_TABLES[self.dfinal]
                        final_states_t = final_i8_bw[fperms_t]
                        final_ball_t = {}
                        for ii in range(len(final_states_t)):
                            kb = final_states_t[ii].tobytes()
                            if kb not in final_ball_t:
                                final_ball_t[kb] = [m[0] + inv_map_bw[m[1:]]
                                                    for m in reversed(fpaths_t[ii])]
                        br_best = final_ball_t.get(sb_best)
                    else:
                        bp_best = ball_paths(a_best, self.dfinal)
                        br_best = bp_best.get(fb_bw, (None, None))[1]
                    tail_bridge = br_best
                    tail_reaches = br_best is not None

                if tail_reaches or best_joint_layer == N_layers:
                    fw_bw_override_key = (best_joint_key, best_joint_layer,
                                          tail_bridge if tail_bridge else [])
        # END of FW-BW block




        # terminal: state must match final (om irrelevant to solved-ness)

        best = None
        if final_arr is not None:
            final_i8 = final_arr.astype(np.int8)
            fb = final_i8.tobytes()
            # INVERT-DIRECTION precompute: ball around `final` ONCE, then per-beam O(1) hash lookup.
            # `final in ball(a, d)` <=> `a in ball(final, d)` (group inverse). For each precomputed
            # path P at depth d, compute final[P_perm] = the state reachable FROM `final` via P,
            # and store its inverse path (the bridge a -> final if a == final[P_perm]).
            inv_map = {"": "'", "'": "", "2": "2"}
            def _invert(path_list):
                return [m[0] + inv_map[m[1:]] for m in reversed(path_list)]
            if self.dfinal in _PERM_TABLES:
                fperms, fpaths = _PERM_TABLES[self.dfinal]
                # final_states[i] = state reachable from `final` via fpaths[i]
                final_states = final_i8[fperms]  # (N_d, 54) int8
                # Map state-bytes -> inverse path (the bridge from a state back to `final`).
                # First-seen wins (BFS order in _build_perm_tables guarantees shortest).
                final_ball = {}
                for i in range(len(final_states)):
                    kb = final_states[i].tobytes()
                    if kb not in final_ball:
                        final_ball[kb] = _invert(fpaths[i])
            else:
                # depth-larger fallback: expensive slow path. Keep BFS for correctness.
                final_ball = None
            for key, (a, cum, pk, path, omk) in layers[-1].items():
                if key[0] == fb:
                    t = cum
                    if best is None or t > best[0]:
                        best = (t, key, [])
                else:
                    if final_ball is not None:
                        br = final_ball.get(key[0])  # key[0] IS state.tobytes()
                    else:
                        bp = ball_paths(a, self.dfinal)
                        br = bp[fb][1] if fb in bp else None
                    if br is not None:
                        t = cum - self.lam * len(br) - 0.5
                        if best is None or t > best[0]:
                            best = (t, key, br)
        if fw_bw_override_key is not None:
            # FW-BW: use the joint-best (state, layer) instead of forward-only
            jkey, j_layer, j_bridge = fw_bw_override_key
            key = jkey
            bridge = list(j_bridge)
            reaches = (jkey[0] == final_arr.astype(np.int8).tobytes()
                       or bool(j_bridge))
            # backtrack only up to j_layer (not full N)
            moves = list(bridge)
            move_oms = [om_by_key.get(key[1], om0)] * len(bridge)
            move_layer = [j_layer - 1] * len(bridge)
            cur = key
            for L in range(j_layer, 0, -1):
                if cur not in layers[L]:
                    break
                a, cum, pk, path, omk = layers[L][cur]
                om_here = om_by_key.get(omk, om0)
                moves = list(path) + moves
                move_oms = [om_here] * len(path) + move_oms
                move_layer = [L - 1] * len(path) + move_layer
                cur = pk
        else:
            if best is None:
                key = max(layers[-1], key=lambda k: layers[-1][k][1])
                bridge, reaches = [], False
            else:
                _, key, bridge = best
                reaches = True

            # backtrack: collect moves + the om each move was performed under + the
            # layer (span) index each move was committed at (for om refinement)
            moves = list(bridge)
            move_oms = [om_by_key.get(key[1], om0)] * len(bridge)
            move_layer = [len(layers) - 2] * len(bridge)
            cur = key
            for L in range(len(layers) - 1, 0, -1):
                a, cum, pk, path, omk = layers[L][cur]
                om_here = om_by_key.get(omk, om0)
                moves = list(path) + moves
                move_oms = [om_here] * len(path) + move_oms
                move_layer = [L - 1] * len(path) + move_layer
                cur = pk

        # OLL/PLL ALG INJECTION (--ll-inject; the EXPRESSION half). The forward decode
        # UNDER-PRODUCED the last layer (reach=False: the terminal bridge could not
        # reach solved within dfinal — the fast OLL/PLL flurry exceeds the per-span
        # ball, so the recognized alg was never generated). Re-anchor at the confident
        # F2L-entry state the table armed on and SUBSTITUTE the recognized alg (which
        # reaches solved by construction) for the drifted tail. Fires ONLY on the
        # terminal failure; solves whose LL already decodes (reach=True) are untouched.
        if self.ll_inject and not reaches and self._ll_inject_at is not None:
            inj_layer, inj_key, inj_moves = self._ll_inject_at
            if 0 <= inj_layer < len(layers) and inj_key in layers[inj_layer]:
                bridge = list(inj_moves)        # the alg IS the terminal bridge now
                om_inj = om_by_key.get(inj_key[1], om0)
                moves = list(bridge)
                move_oms = [om_inj] * len(bridge)
                move_layer = [inj_layer - 1] * len(bridge)
                cur = inj_key
                for L in range(inj_layer, 0, -1):
                    a, cum, pk, path, omk = layers[L][cur]
                    om_here = om_by_key.get(omk, om0)
                    moves = list(path) + moves
                    move_oms = [om_here] * len(path) + move_oms
                    move_layer = [L - 1] * len(path) + move_layer
                    cur = pk
                reaches = True
                self._ll_inject_fired = True
                print(f"  [ll-inject] FIRED: re-anchored @layer{inj_layer}, "
                      f"substituted {len(inj_moves)}-move recognized alg -> SOLVED",
                      flush=True)

        # POST-HOC OM-TIMELINE REFINEMENT (om-Viterbi): canonical moves are final,
        # so per-span states are KNOWN; the orientation timeline is then an exact
        # Viterbi over 24 om states x spans — O(spans x 24^2), LINEAR in solve
        # length, correct for ANY number of rotations (full CFOP solves rotate
        # 2-6 times; exhaustive k-switch search would explode). Switch transitions
        # pay the same gap-gated cost as tracking, so hindsight cannot drift the
        # switch onto noise (measured: an uncosted hindsight search did).
        if self.allow_reorient and moves and layer_reads:
            perm = {n: pp for n, pp in MOVE_LIST}
            state = init_arr.copy()
            span_state = []
            mi = 0
            for si in range(len(layer_reads)):
                while mi < len(moves) and move_layer[mi] <= si:
                    state = state[perm[moves[mi]]]
                    mi += 1
                span_state.append(state.copy())
            omks = [_om_key(o) for o in ORIENTATIONS]
            n_sp = len(layer_reads)
            emis = np.zeros((n_sp, len(omks)))
            for si in range(n_sp):
                st = span_state[si][None, :]
                for oi, o in enumerate(ORIENTATIONS):
                    segs = [self.seg_factory(r, o) for r in layer_reads[si]]
                    segs = [x for x in segs if x.ok]
                    emis[si, oi] = (float(np.mean([x.score_states(st)[0] for x in segs]))
                                    if segs else -50.0)
            # DP
            dp = np.full((n_sp, len(omks)), -1e18)
            bp = np.zeros((n_sp, len(omks)), dtype=int)
            start_oi = omks.index(_om_key(move_oms[0])) if move_oms else 0
            dp[0] = emis[0] - self.rotpen          # any non-start om0 pays once
            dp[0, start_oi] = emis[0, start_oi]
            for si in range(1, n_sp):
                gap = layer_gaps[si] if si < len(layer_gaps) else 0
                sw = (self.rotpen_low if (self.rot_gap_frames is not None
                                          and gap >= self.rot_gap_frames) else self.rotpen)
                stay = dp[si - 1]
                best_prev = float(stay.max())
                for oi in range(len(omks)):
                    cand_stay = stay[oi]
                    cand_switch = best_prev - sw
                    if cand_stay >= cand_switch:
                        dp[si, oi] = cand_stay + emis[si, oi]
                        bp[si, oi] = oi
                    else:
                        dp[si, oi] = cand_switch + emis[si, oi]
                        bp[si, oi] = int(stay.argmax())
            seq = [int(dp[-1].argmax())]
            for si in range(n_sp - 1, 0, -1):
                seq.append(int(bp[si, seq[-1]]))
            seq.reverse()
            om_by = {_om_key(o): o for o in ORIENTATIONS}
            best_seq = [om_by[omks[oi]] for oi in seq]
            move_oms = [best_seq[min(move_layer[i], n_sp - 1)] for i in range(len(moves))]

        # WINDOWED-SCRUB decode (--scrub-decode). Sequential window
        # re-derivation from the app-given init state; the committed word is
        # demoted to ONE
        # candidate hypothesis per window (no constraint, anchor
        # state, or om may derive from committed states -- scoring is
        # om-MARGINALIZED over all 24 ORIENTATIONS). When it runs cleanly the
        # emitted timeline is REPLACED by the scrub word (terminal-bridge tail
        # untouched) so all downstream evaluation consumes it. FAIL-SOFT: the
        # module catches its own exceptions and returns the stock decode; this
        # wrapper is belt-and-braces (F3).
        if self.scrub_decode and self._in_final_pass and moves and layer_reads:
            try:
                import detect.scrub_decode as _scrub
                if self._scrub_perms is None:
                    self._scrub_perms = _scrub.BS.derive_perms()
                self._scrub_evals += 1
                _visual_transition_episodes = ()
                if self.scrub_visual_transition_slots and meta:
                    _visual_transition_episodes = (
                        certified_alignment_transition_episodes(
                        self.align_feat, meta,
                        self._align_transition_spans,
                        threshold=self.align_thr,
                        min_run=self.align_min_run))
                with PERF_TRACE.span("decode.scrub"):
                    (_sc_moves, _sc_layer, _sc_oms,
                     _sc_rep) = _scrub.run_scrub_decode(
                        moves=moves, move_layer=move_layer, move_oms=move_oms,
                        meta=meta, layer_reads=layer_reads,
                        layer_sub_frames=layer_sub_frames,
                        evidence_reads=scrub_evidence_reads,
                        evidence_sub_frames=scrub_evidence_frames,
                        evidence_provenance=scrub_evidence_provenance,
                        seg_factory=self.seg_factory, init_arr=init_arr,
                        final_arr=final_arr, n_bridge=len(bridge),
                        orientations=ORIENTATIONS, perms=self._scrub_perms,
                        rot_event_frames=list(self.rot_events or []),
                        beam_k=int(self.K),   # the tracker's own beam width
                        # M2 input (None => byte-identical): M1 mid-motion rows
                        # merge into the window score only.
                        midmotion_rows=(
                            self.midmotion_rows
                            if getattr(self, "midmotion_reads", False)
                            else None),
                        sidecar_path=self.scrub_decode_log,
                        tag=self.scrub_decode_tag,
                        # Stateful OM uses the final pass's gate verdict stream.
                        om_stateful=bool(self.scrub_om_stateful),
                        gate_events=(
                            list(self._gate_events or [])
                            if self._gate_events_authoritative else None),
                        # Only certified transition spans may form count splits.
                        certified_rest_spans=sorted(
                            self._align_transition_spans),
                        # Only state-owned, terminal-rest-certified episodes
                        # enter the immutable global list. Scrub assigns each
                        # once by terminal frame across retries/windows.
                        visual_transition_episodes=(
                            _visual_transition_episodes),
                        intraburst_phase_slots=(
                            self.intraburst_phase_slots),
                        intraburst_phase_audit=(
                            self.intraburst_phase_audit),
                        gate_drop_slots=bool(self.scrub_gate_drop_slots),
                        dense_prefix=bool(self.scrub_dense_prefix))
                self._scrub_report = _sc_rep
                if _sc_rep.get("emitted"):
                    moves, move_layer, move_oms = _sc_moves, _sc_layer, _sc_oms
                reaches = _reaches_after_scrub(reaches, _sc_rep)
                if self.log_reanchor:
                    print(f"  [scrub-decode] windows="
                          f"{_sc_rep.get('windows', 0)} committed="
                          f"{_sc_rep.get('committed', 0)} extended="
                          f"{_sc_rep.get('extended', 0)} low_conf="
                          f"{_sc_rep.get('low_conf', 0)} unresolved="
                          f"{_sc_rep.get('unresolved', 0)} "
                          f"final_endpoint_ok={_sc_rep.get('final_endpoint_ok')} "
                          f"status={_sc_rep.get('status')}", flush=True)
            except Exception as _sc_exc:          # F3: log-and-continue, never
                self._scrub_report = {"status": "error",
                                      "error": f"{type(_sc_exc).__name__}: "
                                               f"{_sc_exc}"}
                print(f"  [scrub-decode] ERROR "      # abort the decode
                      f"{type(_sc_exc).__name__}: {_sc_exc} -- block skipped, "
                      f"decode unaffected", flush=True)

        # ghost smoothing for notation: simplify within constant-om runs only
        sm_moves, sm_oms = moves, move_oms
        if move_oms:
            out_m, out_o, i = [], [], 0
            while i < len(moves):
                j = i
                while j < len(moves) and _om_key(move_oms[j]) == _om_key(move_oms[i]):
                    j += 1
                chunk = simplify_moves(moves[i:j])
                out_m += chunk
                out_o += [move_oms[i]] * len(chunk)
                i = j
            sm_moves, sm_oms = out_m, out_o
        # Canonical output is simplified by group algebra.
        # Human-notation reconstruction uses the RAW moves + om timeline: raw is what
        # was physically performed (adjustment ghosts included), and simplification
        # breaks the per-move orientation alignment.
        simp = simplify_moves(moves)
        reoriented = len(set(_om_key(o) for o in move_oms)) > 1 if move_oms else False
        # Legacy per-move frame attribution: a move with
        # move_layer si was committed in the read gap ENTERING span si, so span
        # si's first frame is the earliest read of the post-move state; terminal
        # bridge moves (gap-fill after the last read span) get the last span's
        # end frame. Non-decreasing by construction (move_layer is sorted and
        # spans are in frame order). This remains for internal compatibility;
        # reconstruction checkpoints are separately produced by scrub and do
        # not identify when an individual move occurred.
        move_frames = _move_frames_after_scrub(
            moves, move_layer, meta, bridge, self._scrub_report)
        scrub_report = (
            self._scrub_report
            if isinstance(self._scrub_report, dict)
            else {}
        )
        reconstruction_checkpoints = scrub_report.get(
            "reconstruction_checkpoints", [])
        if not isinstance(reconstruction_checkpoints, (list, tuple)):
            reconstruction_checkpoints = []
        return simp, reaches, {"meta": meta, "moves_raw": sm_moves, "oms_raw": sm_oms,
                               "reoriented": reoriented,
                               "oms": move_oms if len(simp) == len(moves) else [],
                               "move_frames": (move_frames
                                               if len(simp) == len(moves) else []),
                               # backtracked (pre-simplification) sequence with
                               # its per-move frames: always aligned, for
                               # consumers that need frame attribution even
                               # when simplification changed the move count
                               "moves_bt": list(moves),
                               "move_frames_bt": list(move_frames),
                               "reconstruction_checkpoints": [
                                   dict(checkpoint)
                                   for checkpoint in reconstruction_checkpoints
                                   if isinstance(checkpoint, dict)
                               ],
                               "reconstruction_checkpoint_count": (
                                   scrub_report.get(
                                       "reconstruction_checkpoint_count", 0)),
                               "reconstruction_checkpoint_move_count": (
                                   scrub_report.get(
                                       "reconstruction_checkpoint_move_count",
                                       0))}


# ============================================================================
# RECALIBRATION PHASE A - opt-in loaders
# for the CALIBRATED decode mode. PURE ADDITIONS: nothing here executes unless
# a driver explicitly calls it (scripts/trellis_gt.py --calibrated), so the
# default decode stays byte-identical.
#
# Calibrated mode = (a) AbsSegment.EMISSION on — read evidence becomes
# -log p(LAB | color) in NATS from the fitted Gaussian+uniform mixture
# (scripts/emission_dataset.py --fit; the June-11 fit_emission.py format),
# replacing the hand-set _LAB_W chrominance distance, with junk stickers
# saturating at the outlier plateau by construction — and (b) the
# temporal/structural constants swapped for the data-estimated values
# (scripts/calibrate_constants.py -> a calibrated-constants JSON artifact).
# GATES ARE OUT OF SCOPE for phase A: still/gapf/rotpen/misfit_thr/
# move_gate_margin keep their hand-set values (misfit_thr and move_gate_margin
# are thresholds ON the span-fit scale, which changes from LAB to nats under
# (a) — their recalibration is phase B's soft-gate conversion).
# ============================================================================

CALIBRATED_TRACKER_KEYS = ("lam", "switchpen", "countpen", "countpen_deep",
                           "MOVE_SEC_TYP", "MOVE_SEC_MIN", "dur_hi_rate")


def load_emission_model(path=None):
    """Load a fitted emission model for AbsSegment.EMISSION and validate its
    shape. path=None resolves $CUBED_EMISSION_FILE, then the tracked default
    data/emission_model.json. Loud on malformed input — a silently-wrong
    emission model would corrupt every span fit."""
    import json as _json
    path = (path or _os.environ.get("CUBED_EMISSION_FILE")
            or "data/emission_model.json")
    with open(path) as f:
        m = _json.load(f)
    missing = {"mu", "var", "pi_out", "logp0"} - set(m)
    if missing:
        raise ValueError(f"emission model {path}: missing keys {sorted(missing)}")
    bad = [c for c in m["mu"] if c not in COLOR_INDEX]
    if bad:
        raise ValueError(f"emission model {path}: unknown colors {bad}")
    for c in m["mu"]:
        if len(m["mu"][c]) != 3 or len(m["var"][c]) != 3 \
                or any(v <= 0 for v in m["var"][c]):
            raise ValueError(f"emission model {path}: bad mu/var for {c}")
    return m


def apply_calibrated_constants(tracker, constants):
    """Apply the phase-A calibrated constants to ONE TrellisTracker instance.
    `constants` is a calibrated-constants dict, with or without its outer
    "constants" wrapper. Only CALIBRATED_TRACKER_KEYS are
    accepted; unknown keys raise (never silently ignored). Class-level
    attributes (MOVE_SEC_TYP/MOVE_SEC_MIN/dur_hi_rate) are shadowed on the
    INSTANCE so other trackers in the process are untouched.
    Returns the applied {key: value} mapping (for the driver's config stamp)."""
    cc = constants.get("constants", constants)
    unknown = [k for k in cc if k not in CALIBRATED_TRACKER_KEYS]
    if unknown:
        raise ValueError(f"calibrated constants: unsupported keys {unknown} "
                         f"(phase A swaps only {list(CALIBRATED_TRACKER_KEYS)})")
    applied = {}
    for k in CALIBRATED_TRACKER_KEYS:
        if k in cc:
            setattr(tracker, k, float(cc[k]))
            applied[k] = float(cc[k])
    return applied


# ============================================================================
# RECALIBRATION PHASE B — the
# fit-scale conversion for EMISSION (nats) scoring. PURE ADDITIONS, driver-
# invoked only (scripts/trellis_gt.py --emission-lab-adapter /
# --derived-nats-gates); nothing here executes by default.
#
# The two computable landmarks of a fitted emission model's cost scale:
#   D_out = -(log pi_out + logp0)  — the junk saturation plateau; every span
#           fit is bounded BELOW by -D_out (logaddexp construction).
#   D_in  = E[-log p(lab|color) | inlier], pooled over colors — where a
#           CORRECT read lands. For a 3-D diagonal Gaussian E[Mahalanobis^2]=3:
#           D_in(c) ~= -log(1-pi_out) + 1.5*log(2*pi) + 0.5*sum_ch log var + 1.5
#           (the logaddexp plateau correction at good fits is <~1e-3; ignored).
# The LAB fit scale these replace was measured (in-code misfit_thr comment):
# good ~ -5..-10, garbage ~ -20 and worse. The affine adapter maps one onto
# the other so every fit-scale gate + the evidence<->prior balance keep their
# tuned semantics, with parameters DERIVED from whichever model is loaded
# (transfers across refits — no hand-retuned thresholds).
# ============================================================================


def emission_fit_landmarks(model):
    """(D_in, D_out) of a fitted emission model (see block comment above).
    D_in pools per-color expectations weighted by the fitted per-color sample
    counts model["n"] when present (equal weights otherwise). Loud on a
    malformed pi_out — a silent NaN here would corrupt every derived gate."""
    import math as _math
    pi_out = float(model["pi_out"])
    if not (0.0 < pi_out < 1.0):
        raise ValueError(f"emission model pi_out={pi_out} outside (0,1)")
    d_out = -(_math.log(pi_out) + float(model["logp0"]))
    ns = model.get("n") or {}
    d_in = tot_w = 0.0
    for c, var in model["var"].items():
        w = float(ns.get(c, 1.0))
        d_c = (-_math.log(1.0 - pi_out) + 1.5 * _math.log(2.0 * _math.pi)
               + 0.5 * sum(_math.log(float(v)) for v in var) + 1.5)
        d_in += w * d_c
        tot_w += w
    if tot_w <= 0:
        raise ValueError("emission model has no colors to pool D_in over")
    return d_in / tot_w, d_out


def derive_emission_adapter(model, good_ref=7.5, junk_ref=22.0):
    """Affine (a, b) for AbsSegment.EMISSION_ADAPTER: b = (junk_ref -
    good_ref) / (D_out - D_in) > 0, a = good_ref - b*D_in — a correct read
    costs ~good_ref (span fit ~ -good_ref, the LAB good band) and saturated
    junk ~junk_ref (the LAB garbage band), so misfit_thr=-12 /
    move_gate_margin=1.0 / om_pf_margin=2.0 and the hand-tuned
    lam/switchpen/countpen/rotpen/ll_prior balance keep their measured
    semantics under EMISSION scoring. good_ref/junk_ref defaults are the
    DOCUMENTED LAB regimes (good -5..-10 -> 7.5; garbage onset -20, measured
    true-state-under-mediocre-cal -25.9 -> 22). Raises on a degenerate model
    (plateau not separated from the inlier band: an adapter would only
    amplify noise). Returns {a, b, d_in, d_out} for the driver's stamp."""
    if junk_ref <= good_ref:
        raise ValueError(f"junk_ref {junk_ref} must exceed good_ref {good_ref}")
    d_in, d_out = emission_fit_landmarks(model)
    if d_out <= d_in + 1.0:
        raise ValueError(
            f"emission model degenerate for the adapter: D_out {d_out:.2f} <= "
            f"D_in {d_in:.2f} + 1 (outlier plateau inside the inlier band)")
    b = (junk_ref - good_ref) / (d_out - d_in)
    return {"a": good_ref - b * d_in, "b": b, "d_in": d_in, "d_out": d_out}
