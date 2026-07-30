"""Solve-local cache for AbsSegment construction and read-trust inference.

``AbsSegment`` evidence (admitted LAB cells, centroid distances, trust weights,
and per-cell offsets) is independent of cube orientation.  Only its gather
matrix changes across the 24 orientations.  The historical factories rebuilt
the complete segment for every ``(read, orientation)`` pair; production also
had no cross-call cache, so the two-pass tracker rebuilt them again.

``CachedAbsSegmentFactory`` builds one authoritative ``AbsSegment`` template
per read, shallow-clones its immutable evidence, and regenerates only the exact
integer gather matrix for later orientations.  It also caches complete
``(read, orientation)`` objects across tracker passes.

``prime_trust_batch`` is the companion solve-level trust path.  It builds all
orientation-independent TrustNumpy feature rows once and scores them as one
logical batch (resident CUDA when available, bit-exact numpy fallback).  A
deferred cache adapter makes the existing AbsSegment miss path consume each
precomputed probability on first real use.  That preserves the old per-read
statistics exactly and verifies the rebuilt feature matrix byte-for-byte
before accepting a primed result; a mismatch falls back to the original model.

The module does not mutate TrellisTracker and is inert until a caller chooses
this factory / invokes ``prime_trust_batch``.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np

from detect.calibrator import MIN_STICKER_L
from detect.move_detector import GRID_ROTATIONS
from detect.trellis_tracker import AbsSegment, FACE_OFFSET


_IDENTITY_OM = {"up": "up", "front": "front", "right": "right"}
_MISSING = object()


def _orientation(orient_map):
    # AbsSegment treats both None and an empty mapping as identity.
    return dict(orient_map or _IDENTITY_OM)


def _orientation_key(orient_map):
    # Preserve every supplied mapping entry rather than assuming right is
    # derivable from up/front.  Tests and research callers sometimes provide
    # deliberately non-group maps.
    return tuple(sorted(_orientation(orient_map).items()))


@dataclass
class _Template:
    read: object
    segment: AbsSegment
    rotation_offsets: np.ndarray | None


@dataclass
class _PrimedPrediction:
    features: np.ndarray
    p_trust: np.ndarray


class _PreparedTrustModel:
    """Duck-type wrapper that serves an armed primed prediction once.

    ``AbsSegment._trust_soft_weights`` first calls ``cache.get(id(read))`` and,
    on a miss, builds X then calls ``model.predict_proba(X)``.  The paired cache
    below arms this wrapper in thread-local storage at that exact seam.
    """

    def __init__(self, base_model, pending):
        self.base_model = base_model
        self.pending = pending
        self._local = threading.local()
        self.primed_hits = 0
        self.feature_mismatch_fallbacks = 0

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def arm(self, key):
        self._local.key = key

    def predict_proba(self, X):
        key = getattr(self._local, "key", None)
        self._local.key = None
        rec = self.pending.get(key)
        arr = np.ascontiguousarray(np.asarray(X, np.float64))
        if (rec is not None and rec.features.shape == arr.shape
                and np.array_equal(rec.features, arr, equal_nan=True)):
            self.primed_hits += 1
            p = rec.p_trust
            return np.stack([1.0 - p, p], axis=1)
        if rec is not None:
            self.feature_mismatch_fallbacks += 1
        return self.base_model.predict_proba(arr)


class _PreparedTrustCache(dict):
    """Normal live cache plus deferred, stats-preserving primed entries."""

    def __init__(self, live, pending, model):
        super().__init__(live)
        self.pending = pending
        self.model = model

    def get(self, key, default=None):
        value = dict.get(self, key, _MISSING)
        if value is not _MISSING:
            return value
        if key in self.pending:
            self.model.arm(key)
            # Force the existing AbsSegment cache-miss branch.  It rebuilds X,
            # receives the primed probability from the wrapper, updates stats,
            # then stores the live value through __setitem__ below.
            return None
        return default

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.pending.pop(key, None)

    def has_prediction(self, key):
        return dict.__contains__(self, key) or key in self.pending


class CachedAbsSegmentFactory:
    """Callable ``seg_factory(read, orientation)`` with solve-local reuse.

    ``metadata`` receives the read object and returns the keyword arguments for
    ``AbsSegment`` (``face_vis``, ``cell_conf``, ``frame_motion``,
    ``frame_aligned``, and optionally ``frame_ev``).  Metadata and centroids
    must remain fixed for the lifetime of a factory, matching one decode.
    """

    def __init__(self, centroids, metadata: Callable[[object], Mapping] | None = None,
                 segment_cls=AbsSegment):
        self.centroids = centroids
        self.metadata = metadata or (lambda _read: {})
        self.segment_cls = segment_cls
        self._metadata_cache = {}
        self._templates = {}
        self._segments = {}
        self.full_constructions = 0
        self.orientation_clones = 0
        self.cache_hits = 0

    def _kwargs(self, read):
        key = id(read)
        cached = self._metadata_cache.get(key)
        if cached is not None and cached[0] is read:
            return cached[1]
        kwargs = dict(self.metadata(read) or {})
        self._metadata_cache[key] = (read, kwargs)
        return kwargs

    @staticmethod
    def _template(segment, read):
        if not getattr(segment, "ok", False):
            return _Template(read, segment, None)
        # Current AbsSegment retains these two orientation-independent arrays.
        # If a divergent/older scorer lacks them, the factory safely falls back
        # to full construction for subsequent orientations.
        if not hasattr(segment, "_face_names") or not hasattr(segment, "_pos"):
            return _Template(read, segment, None)
        face_names = list(segment._face_names)
        loc = {face: i for i, face in enumerate(segment.faces)}
        fidx = np.asarray([loc[face] for face in face_names], np.intp)
        combos = np.asarray(segment.combos)
        pos = np.asarray(segment._pos)
        offsets = np.asarray(GRID_ROTATIONS)[combos[:, fidx], pos[None, :]]

        # These are also orientation-independent.  Populate them once so every
        # clone shares the same immutable arrays instead of lazily rebuilding
        # one copy per orientation in score_states.
        if segment._dflat is None:
            segment._dflat = np.ascontiguousarray(segment.dist).ravel()
        if getattr(segment, "_off", None) is None:
            segment._off = segment._ar * 6
        # Keep a clean, unmodified header. Callers may replace instance fields
        # such as ``_wv`` for a specialized scoring pass; no returned segment
        # may then become the source for a later orientation clone. The large
        # evidence arrays remain shared and read-only.
        return _Template(read, copy.copy(segment), np.asarray(offsets, np.intp))

    @staticmethod
    def _clone(template, orient_map):
        source = template.segment
        out = copy.copy(source)
        orient = _orientation(orient_map)
        out.orient_map = orient
        if not getattr(source, "ok", False):
            return out
        if template.rotation_offsets is None:
            return None
        base = np.asarray(
            [FACE_OFFSET[orient.get(face, face)] for face in source._face_names],
            np.intp,
        )
        gm = np.asarray(base[None, :] + template.rotation_offsets, np.intp)
        out.gathers = list(gm)
        out._gm = gm
        # copy.copy already shares these, but assignments make the invariant
        # explicit and survive a future custom __copy__ implementation.
        out._dflat = source._dflat
        out._off = source._off
        return out

    def __call__(self, read, orient_map=None):
        rkey = id(read)
        okey = _orientation_key(orient_map)
        key = (rkey, okey)
        cached = self._segments.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        template = self._templates.get(rkey)
        if template is not None and template.read is read:
            segment = self._clone(template, orient_map)
            if segment is not None:
                self.orientation_clones += 1
            else:
                segment = self.segment_cls(
                    read, self.centroids, orient_map, **self._kwargs(read))
                self.full_constructions += 1
        else:
            segment = self.segment_cls(
                read, self.centroids, orient_map, **self._kwargs(read))
            template = self._template(segment, read)
            self._templates[rkey] = template
            self.full_constructions += 1
        self._segments[key] = segment
        return segment

    def clear(self):
        self._metadata_cache.clear()
        self._templates.clear()
        self._segments.clear()

    def prime_trust_batch(self, reads: Iterable[object], *, trust_soft=None,
                          device="auto", max_rows=65536):
        """Prime orientation-independent trust probabilities for ``reads``.

        Must be called after the caller wires ``AbsSegment.TRUST_SOFT`` and its
        per-read metadata, ideally before the first segment construction.
        Existing live/primed cache entries are retained and skipped.
        """
        tm = trust_soft if trust_soft is not None else self.segment_cls.TRUST_SOFT
        if not tm:
            return {"status": "disabled", "reads": 0, "rows": 0,
                    "backend": "none", "wall_s": 0.0}

        model = tm["model"]
        if isinstance(model, _PreparedTrustModel):
            proxy = model
            base_model = model.base_model
            cache = tm["cache"]
            if not isinstance(cache, _PreparedTrustCache):
                raise TypeError("prepared trust model requires prepared cache")
            pending = proxy.pending
        else:
            base_model = model
            pending = {}
            proxy = _PreparedTrustModel(base_model, pending)
            cache = _PreparedTrustCache(tm.get("cache") or {}, pending, proxy)

        unique = []
        seen = {}
        for read in reads:
            key = id(read)
            if seen.get(key) is read:
                continue
            seen[key] = read
            if (isinstance(cache, _PreparedTrustCache)
                    and cache.has_prediction(key)):
                continue
            if not isinstance(cache, _PreparedTrustCache) and key in cache:
                continue
            unique.append(read)

        start = time.perf_counter()
        matrices, keys = [], []
        for read in unique:
            X = self._trust_features(read, self._kwargs(read), tm)
            if len(X):
                matrices.append(X)
                keys.append(id(read))
        rows = sum(len(X) for X in matrices)
        if rows:
            whole = np.concatenate(matrices, axis=0)
            if hasattr(base_model, "predict_proba_batch"):
                proba, backend = base_model.predict_proba_batch(
                    whole, device=device, max_rows=max_rows,
                    return_backend=True)
            else:
                proba = base_model.predict_proba(whole)
                backend = "model-batch"
            p = np.asarray(proba[:, 1], np.float64)
            offset = 0
            for key, X in zip(keys, matrices):
                hi = offset + len(X)
                pending[key] = _PrimedPrediction(X, p[offset:hi])
                offset = hi
        else:
            backend = "none"

        tm["model"] = proxy
        tm["cache"] = cache
        return {
            "status": "ok",
            "reads": len(unique),
            "prepared_reads": len(keys),
            "rows": rows,
            "backend": backend,
            "wall_s": time.perf_counter() - start,
        }

    def _trust_features(self, read, kwargs, tm):
        """Exact pre-FACE_GATE feature matrix from AbsSegment's soft path."""
        from detect.read_trust import (EV_CAP, FACE_SLOTS, build_features,
                                       legacy_min_dist)

        face_vis = kwargs.get("face_vis")
        cell_conf = kwargs.get("cell_conf")
        labs, face_names, confs = [], [], []
        for face, lab9 in read:
            c9 = (cell_conf or {}).get(face)
            if (self.segment_cls.VIS_GATE is not None and face_vis
                    and face_vis.get(face, 1.0) < self.segment_cls.VIS_GATE):
                continue
            for pos in range(9):
                lab = lab9[pos]
                if lab[0] < MIN_STICKER_L:
                    continue
                labs.append(lab)
                face_names.append(face)
                confs.append(c9[pos] if c9 and c9[pos] is not None else 1.0)
        if not labs:
            return np.empty((0, int(tm.get("n_features", 15))), np.float64)

        arr = np.asarray(labs, np.float64)
        conf_v = np.asarray(confs, np.float64)
        d1 = legacy_min_dist(arr, tm["cen_mat"])
        face_i = np.asarray([FACE_SLOTS.index(face) for face in face_names],
                            np.int64)
        n = len(arr)
        frame_motion = kwargs.get("frame_motion")
        frame_aligned = kwargs.get("frame_aligned")
        mot = np.full(n, 0.0 if frame_motion is None else float(frame_motion))
        alg = np.full(n, np.nan if frame_aligned is None
                      else float(frame_aligned))
        evkw = {}
        if int(tm.get("n_features", 15)) == 17:
            frame_ev = kwargs.get("frame_ev")
            since, to = frame_ev if frame_ev is not None else (EV_CAP, EV_CAP)
            evkw = {
                "ev_since": np.full(n, EV_CAP if since is None else float(since)),
                "ev_to": np.full(n, EV_CAP if to is None else float(to)),
            }
        return build_features(conf_v, d1, arr, mot, alg, face_i,
                              np.zeros(n, np.int64), **evkw)
