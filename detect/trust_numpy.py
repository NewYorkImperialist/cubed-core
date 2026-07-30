"""NumPy scorer for the packaged per-cell read-trust forest.

The release artifact stores tree nodes and calibration curves in an NPZ file.
``TrustNumpy`` loads that format, scores it with NumPy, and can optionally use
resident Torch tensors for solve-level batches.
"""

import numpy as np


def _interp_linear(T, xp, yp):
    """Evaluate a piecewise-linear calibration curve in slope form.

    ``T`` must already be clipped to ``[xp[0], xp[-1]]``.
    """
    xp = np.asarray(xp, np.float64)
    yp = np.asarray(yp, np.float64)
    slopes = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    # interp1d: idx = searchsorted(x, T, side="left").clip(1, len-1); interval
    # [idx-1, idx]. (side="left" matches scipy's _call_linear.)
    idx = np.searchsorted(xp, T, side="left")
    idx = np.clip(idx, 1, len(xp) - 1)
    lo = idx - 1
    return slopes[lo] * (T - xp[lo]) + yp[lo]


class TrustNumpy:
    """Score the packaged trust forest.

    ``predict_proba(X)`` returns ``(N, 2)`` with trust probability in column 1.
    """

    def __init__(self, folds, features, source_md5=None):
        # folds: list of dicts, one per calibration fold, each with keys
        #   baseline (float), feat/thr/left/right/leaf/val/miss (T, M) arrays,
        #   ix/iy (isotonic thresholds), xmin/xmax (isotonic clip bounds).
        self.folds = folds
        self.features = list(features)
        self.n_features = len(self.features)
        self.source_md5 = source_md5
        # Lazily populated by ``predict_proba_batch(device="cuda")``.  Keeping
        # the distilled trees resident matters much more than the transfer of
        # one solve's feature rows (~10-20 MiB): rebuilding ~400 KiB of model
        # tensors for every read would put Python and H2D setup straight back
        # in the hot loop this path removes.  Torch remains a lazy optional
        # dependency; ordinary ``predict_proba`` imports numpy only.
        self._torch_cache = {}

    @classmethod
    def load(cls, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        folds = []
        for i in range(int(z["n_folds"])):
            f = {"baseline": float(z[f"f{i}_baseline"]),
                 "xmin": float(z[f"f{i}_xmin"]),
                 "xmax": float(z[f"f{i}_xmax"])}
            for k in ("feat", "thr", "left", "right", "leaf", "val", "miss",
                      "ix", "iy"):
                f[k] = z[f"f{i}_{k}"]
            folds.append(f)
        feats = [str(s) for s in z["features"].tolist()]
        smd5 = str(z["source_md5"]) if "source_md5" in z.files else None
        return cls(folds, feats, smd5)

    # ---- pure-numpy inference ----------------------------------------------
    def _fold_raw(self, X, f):
        """Return one fold's baseline plus its ordered tree leaf values."""
        feat, thr = f["feat"], f["thr"]
        left, right, leaf = f["left"], f["right"], f["leaf"]
        miss, val = f["miss"], f["val"]
        Tn, N = feat.shape[0], X.shape[0]
        tix = np.arange(Tn)[:, None]
        ncol = np.arange(N)[None, :]
        node = np.zeros((Tn, N), np.int64)
        while True:
            cur_leaf = leaf[tix, node].astype(bool)
            active = ~cur_leaf
            if not active.any():
                break
            fi = feat[tix, node]
            xv = X[ncol, fi]                              # (T, N)
            go_left = np.where(np.isnan(xv),
                               miss[tix, node].astype(bool),
                               xv <= thr[tix, node])
            nxt = np.where(go_left, left[tix, node], right[tix, node])
            node = np.where(active, nxt, node)
        leaves = val[tix, node]                           # (T, N)
        raw = np.full(N, f["baseline"], np.float64)
        for t in range(Tn):                               # sequential order
            raw = raw + leaves[t]
        return raw

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, np.float64))
        N = X.shape[0]
        proba0 = np.zeros(N, np.float64)
        proba1 = np.zeros(N, np.float64)
        for f in self.folds:                              # CCCV fold average
            raw = self._fold_raw(X, f)
            t = np.clip(raw, f["xmin"], f["xmax"])
            pf = _interp_linear(t, f["ix"], f["iy"])
            proba1 += pf
            proba0 += 1.0 - pf
        k = float(len(self.folds))
        proba0 /= k
        proba1 /= k
        return np.stack([proba0, proba1], 1)

    def _torch_folds(self, torch, device):
        """Resident fp64 Torch form of the distilled forest for ``device``.

        The model is immutable after construction, so one tensorization per
        process/device is sufficient.  ``torch.tensor`` deliberately copies
        arrays loaded from read-only npz mappings; ``as_tensor`` emits a
        non-writable-buffer warning and leaves mutation semantics undefined.
        """
        dev = torch.device(device)
        key = str(dev)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached

        out = []
        for f in self.folds:
            ix = np.asarray(f["ix"], np.float64)
            iy = np.asarray(f["iy"], np.float64)
            slopes = (iy[1:] - iy[:-1]) / (ix[1:] - ix[:-1])

            def tensor(value, dtype):
                return torch.tensor(np.array(value, copy=True), dtype=dtype,
                                    device=dev)

            out.append({
                "baseline": float(f["baseline"]),
                "feat": tensor(f["feat"], torch.int64),
                "thr": tensor(f["thr"], torch.float64),
                "left": tensor(f["left"], torch.int64),
                "right": tensor(f["right"], torch.int64),
                "leaf": tensor(f["leaf"], torch.bool),
                "val": tensor(f["val"], torch.float64),
                "miss": tensor(f["miss"], torch.bool),
                "ix": tensor(ix, torch.float64),
                "iy": tensor(iy, torch.float64),
                "slopes": tensor(slopes, torch.float64),
                "xmin": float(f["xmin"]),
                "xmax": float(f["xmax"]),
                # Traverse to the deepest leaf represented by this fold.
                # Fixed-depth traversal avoids a GPU-host ``any().item`` sync;
                # rows already at leaves retain their node.
                "steps": self._forest_max_depth(f),
            })
        self._torch_cache[key] = out
        return out

    @staticmethod
    def _forest_max_depth(fold):
        """Return the maximum root-to-leaf edge count in a padded forest."""
        leaf = np.asarray(fold["leaf"], bool)
        left = np.asarray(fold["left"], np.int64)
        right = np.asarray(fold["right"], np.int64)
        deepest = 0
        for tree in range(leaf.shape[0]):
            pending = [(0, 0)]
            while pending:
                node, depth = pending.pop()
                deepest = max(deepest, depth)
                if not leaf[tree, node]:
                    pending.append((int(left[tree, node]), depth + 1))
                    pending.append((int(right[tree, node]), depth + 1))
        return deepest

    def _predict_proba_torch(self, X, device="cuda"):
        """Torch/CUDA mirror used by the solve-level batch path.

        Tree traversal and calibration happen on device. Tree values are
        reduced by Torch rather than by the NumPy loop.
        """
        import torch  # lazy: CPU-only/default decode still imports numpy alone

        dev = torch.device(device)
        X_np = np.ascontiguousarray(np.asarray(X, np.float64))
        n_rows = X_np.shape[0]
        if n_rows == 0:
            return np.empty((0, 2), np.float64)
        X_t = torch.as_tensor(X_np, dtype=torch.float64, device=dev)
        proba0 = torch.zeros(n_rows, dtype=torch.float64, device=dev)
        proba1 = torch.zeros(n_rows, dtype=torch.float64, device=dev)
        ncol = torch.arange(n_rows, dtype=torch.int64, device=dev)[None, :]

        for f in self._torch_folds(torch, dev):
            feat, thr = f["feat"], f["thr"]
            left, right = f["left"], f["right"]
            leaf, miss, val = f["leaf"], f["miss"], f["val"]
            n_trees = feat.shape[0]
            tix = torch.arange(n_trees, dtype=torch.int64,
                                device=dev)[:, None]
            node = torch.zeros((n_trees, n_rows), dtype=torch.int64,
                               device=dev)
            for _ in range(f["steps"]):
                active = ~leaf[tix, node]
                fi = feat[tix, node]
                xv = X_t[ncol, fi]
                go_left = torch.where(torch.isnan(xv), miss[tix, node],
                                      xv <= thr[tix, node])
                nxt = torch.where(go_left, left[tix, node], right[tix, node])
                node = torch.where(active, nxt, node)

            leaves = val[tix, node]
            raw = leaves.sum(dim=0) + f["baseline"]
            t = torch.clamp(raw, f["xmin"], f["xmax"]).contiguous()
            idx = torch.searchsorted(f["ix"], t, right=False)
            idx = torch.clamp(idx, 1, len(f["ix"]) - 1)
            lo = idx - 1
            pf = f["slopes"][lo] * (t - f["ix"][lo]) + f["iy"][lo]
            proba1 += pf
            proba0 += 1.0 - pf

        k = float(len(self.folds))
        out = torch.stack([proba0 / k, proba1 / k], dim=1)
        return out.cpu().numpy()

    def predict_proba_batch(self, X, *, device="auto", max_rows=65536,
                            return_backend=False):
        """Score one solve-level feature matrix, preferring resident CUDA.

        ``device`` values:
          - ``"auto"``: CUDA when Torch reports it available, NumPy otherwise;
          - ``"cuda"`` / ``"cuda:N"``: request Torch CUDA, with NumPy fallback;
          - ``"torch:cpu"``: exercise the Torch mirror on CPU (tests/profiling);
          - ``None`` / ``"numpy"`` / ``"cpu"``: NumPy path.

        The logical batch may be split into ``max_rows``-sized device chunks to
        cap the forest traversal's temporary ``(trees, rows)`` tensors.  Samples
        are independent, so chunking does not alter any reduction.  Any Torch
        import/device/OOM failure falls back to the NumPy scorer for the
        complete matrix. ``return_backend`` exposes the selected backend.
        """
        X = np.ascontiguousarray(np.asarray(X, np.float64))
        requested = "numpy" if device is None else str(device)
        if requested in ("numpy", "cpu"):
            out, backend = self.predict_proba(X), "numpy"
        else:
            try:
                import torch  # lazy optional dependency
                if requested == "auto":
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA unavailable")
                    torch_device = "cuda"
                elif requested == "torch:cpu":
                    torch_device = "cpu"
                else:
                    torch_device = requested
                    if (torch.device(torch_device).type == "cuda"
                            and not torch.cuda.is_available()):
                        raise RuntimeError("CUDA unavailable")
                step = max(1, int(max_rows))
                chunks = [self._predict_proba_torch(X[i:i + step], torch_device)
                          for i in range(0, len(X), step)]
                out = (np.concatenate(chunks, axis=0) if chunks
                       else np.empty((0, 2), np.float64))
                backend = f"torch:{torch.device(torch_device)}"
            except Exception:  # missing Torch / bad device / OOM -> NumPy
                try:
                    if "torch" in locals() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                out, backend = self.predict_proba(X), "numpy-fallback"
        return (out, backend) if return_backend else out

    def clear_device_cache(self, device=None):
        """Drop resident Torch model tensors (mainly for tests/process teardown)."""
        if device is None:
            self._torch_cache.clear()
        else:
            self._torch_cache.pop(str(device), None)
