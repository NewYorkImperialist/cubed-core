"""Build the per-cell features consumed by the packaged read-trust scorer.

Feature order is part of the model artifact contract. Centroid distance uses
the fixed input calibration, while alignment and event context have explicit
missing-value encodings. This module depends only on NumPy.
"""

import numpy as np

# Chrominance-weighted LAB metric used by the trust feature contract. It
# mirrors the decoder metric without importing the decoder and creating a
# cycle.
_LAB_W = np.array([0.15, 1.0, 1.0])

# Spatial slot order for the ``face`` feature index.
FACE_SLOTS = ["up", "front", "right"]

# Base feature contract. Order matters and is stored in the NPZ artifact.
FEATURE_NAMES_V1 = ["conf", "d1", "L", "a", "b", "motion", "aligned",
                    "aligned_nan", "face_mean_conf", "face_min_conf",
                    "face_mean_d1", "face_n_cells", "frame_mean_conf",
                    "frame_mean_d1", "frame_n_cells"]

# Event-aware contract: the base features plus two columns appended at the end.
#   ev_since = frames since the nearest detected event at frame <= f
#   ev_to    = frames until the nearest detected event at frame > f
# Both are clipped to [0, EV_CAP]. A missing side, or an empty event stream,
# encodes as EV_CAP.
EV_CAP = 300.0
FEATURE_NAMES = FEATURE_NAMES_V1 + ["ev_since", "ev_to"]


def event_context(frames, event_frames):
    """(ev_since, ev_to) float arrays for `frames` given sorted event frames.

    event_frames None/empty -> both features = EV_CAP everywhere."""
    frames = np.asarray(frames, np.float64)
    if event_frames is None or len(event_frames) == 0:
        full = np.full(len(frames), EV_CAP)
        return full, full.copy()
    ev = np.sort(np.asarray(event_frames, np.float64))
    # side="right": idx = #events with frame <= f, so ev[idx-1] is the nearest
    # event AT-or-before f (an event AT f counts as "since 0", never "to 0")
    # and ev[idx] is the nearest event STRICTLY after f.
    idx = np.searchsorted(ev, frames, side="right")
    since = np.where(idx > 0, frames - ev[np.maximum(idx - 1, 0)], EV_CAP)
    to = np.where(idx < len(ev),
                  ev[np.minimum(idx, len(ev) - 1)] - frames, EV_CAP)
    return np.clip(since, 0, EV_CAP), np.clip(to, 0, EV_CAP)


def group_agg(keys, values):
    """Broadcast mean, minimum, and count for each unique key row."""
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    s = np.bincount(inv, weights=values)
    mean = (s / cnt)[inv]
    mn = np.full(len(uniq), np.inf)
    np.minimum.at(mn, inv, values)
    return mean, mn[inv], cnt[inv]


def build_features(conf, d1, lab, motion, aligned, face, frame, tag_idx=None,
                   ev_since=None, ev_to=None):
    """(N, 15) [v1] or (N, 17) [v3] float feature matrix in contract order.

    All inputs are per-cell arrays of length N (`lab` is (N, 3)); `frame` and
    `tag_idx` are grouping keys for face/frame context aggregates.
    `tag_idx=None` uses one group. `ev_since` and `ev_to` must be supplied
    together; omitting both returns the base 15-column contract.
    """
    if (ev_since is None) != (ev_to is None):
        raise ValueError("ev_since/ev_to must be given together (v3) or "
                         "both omitted (v1)")
    conf = np.asarray(conf, np.float64)
    d1 = np.asarray(d1, np.float64)
    lab = np.asarray(lab, np.float64)
    motion = np.asarray(motion, np.float64)
    aligned = np.asarray(aligned, np.float64)
    face = np.asarray(face, np.int64)
    frame = np.asarray(frame, np.int64)
    if tag_idx is None:
        tag_idx = np.zeros(len(conf), np.int64)
    else:
        tag_idx = np.asarray(tag_idx, np.int64)
    key_face = np.stack([tag_idx, frame, face], 1)
    key_frame = np.stack([tag_idx, frame], 1)
    f_mc, f_minc, f_cnt = group_agg(key_face, conf)
    f_md, _, _ = group_agg(key_face, d1)
    fr_mc, _, fr_cnt = group_agg(key_frame, conf)
    fr_md, _, _ = group_agg(key_frame, d1)
    align_nan = np.isnan(aligned).astype(float)
    aligned_f = np.where(np.isnan(aligned), 0.5, aligned)
    cols = [conf, d1, lab[:, 0], lab[:, 1], lab[:, 2], motion,
            aligned_f, align_nan,
            f_mc, f_minc, f_md, f_cnt,
            fr_mc, fr_md, fr_cnt]
    if ev_since is not None:
        cols.append(np.clip(np.asarray(ev_since, np.float64), 0, EV_CAP))
        cols.append(np.clip(np.asarray(ev_to, np.float64), 0, EV_CAP))
    return np.stack(cols, 1)


def centroid_matrix(centroids):
    """Return known color centroids as a matrix, preserving input order."""
    from detect.move_detector import COLOR_INDEX
    names = [c for c in centroids if c in COLOR_INDEX]
    return np.array([centroids[c] for c in names], float), names


def legacy_min_dist(labs, cen_mat):
    """Return each cell's minimum weighted distance to the centroids."""
    labs = np.asarray(labs, np.float64)
    return np.sqrt((((labs[:, None, :] - cen_mat[None, :, :]) ** 2)
                    * _LAB_W).sum(2)).min(1)
