"""ONNX Runtime inference utilities for YOLO models.

Replaces ultralytics at inference time. Provides preprocessing,
postprocessing, and a thin model wrapper for classification and
segmentation YOLO models exported to ONNX.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from cubed_core.tracker_runtime import preload_tracker_provider_dependencies


_REQUIRE_CUDA_ENV = "CUBED_ORT_REQUIRE_CUDA"


def _cuda_required() -> bool:
    value = os.environ.get(_REQUIRE_CUDA_ENV, "").strip()
    if value not in {"", "0", "1"}:
        raise RuntimeError(f"{_REQUIRE_CUDA_ENV} must be 0 or 1")
    return value == "1"


def _providers_from_env():
    """Return an optional provider order from ``CUBED_ORT_PROVIDERS``.

    Comma-separated tokens from {trt16, trt32, cuda, cpu}; unknown or
    unavailable providers are skipped. CPU is always appended as a fallback.
    An unset value preserves the caller's default provider selection.
    TensorRT is opt-in because changing providers may change model output."""
    spec = os.environ.get("CUBED_ORT_PROVIDERS", "").strip().lower()
    if not spec:
        return None
    avail = ort.get_available_providers()
    cache = os.environ.get("CUBED_TRT_CACHE", "/workspace/trt_cache")
    trt_opts = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache,
    }
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if tok in ("trt16", "trt32") and "TensorrtExecutionProvider" in avail:
            opts = dict(trt_opts)
            if tok == "trt16":
                opts["trt_fp16_enable"] = True
            out.append(("TensorrtExecutionProvider", opts))
        elif tok == "cuda" and "CUDAExecutionProvider" in avail:
            out.append("CUDAExecutionProvider")
        elif tok == "cpu":
            out.append("CPUExecutionProvider")
    names = [p if isinstance(p, str) else p[0] for p in out]
    if "CPUExecutionProvider" not in names:
        out.append("CPUExecutionProvider")
    return out or None


class OnnxModel:
    """Thin wrapper around onnxruntime.InferenceSession."""

    def __init__(self, model_path: str, providers: list[str] | None = None):
        require_cuda = _cuda_required()
        if require_cuda:
            preload_tracker_provider_dependencies(("CUDAExecutionProvider",))
        if providers is None:
            providers = _providers_from_env()
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        provider_names = [provider if isinstance(provider, str) else provider[0]
                          for provider in providers]
        if require_cuda and "CUDAExecutionProvider" not in provider_names:
            raise RuntimeError(
                "canonical Decode requires CUDAExecutionProvider for ONNX inference")
        # onnxruntime sizes its intra-op thread pool to the visible CPU count and
        # IGNORES OMP_NUM_THREADS. On a 256-core host that means one process
        # spawns ~256 threads thrashing on tiny inferences (load >700 with a few
        # parallel extractors). ORT_INTRA_OP_THREADS caps it (0 = ORT default,
        # unchanged for normal/server use; par_extract sets it low for fan-out).
        so = ort.SessionOptions()
        t = int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))
        if t > 0:
            so.intra_op_num_threads = t
            so.inter_op_num_threads = 1
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, sess_options=so,
                                            providers=providers)
        active_providers = tuple(self.session.get_providers())
        if require_cuda and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "canonical Decode ONNX session did not activate CUDAExecutionProvider")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # The leading (batch) dim of the input: a literal int (e.g. 1) for a
        # fixed-batch export (ultralytics ONNX default), or a symbolic name /
        # None for a dynamic-batch export. Only the latter can accept N>1 rows
        # in one session.run call — callers (e.g. classify_alignment_batch)
        # use this to decide whether batching is possible at all, without
        # re-exporting the model.
        batch_dim = self.session.get_inputs()[0].shape[0]
        self.supports_batch = not isinstance(batch_dim, int) or batch_dim != 1

    def run(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: input_tensor})


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_classify(img: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """Preprocess for YOLO classification: resize short edge, center-crop, normalize.

    Args:
        img: BGR HWC uint8 image.
        imgsz: Target square size (matches export imgsz).

    Returns:
        NCHW float32 tensor ready for inference.
    """
    h, w = img.shape[:2]
    # Resize so short edge == imgsz, maintaining aspect ratio
    scale = imgsz / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center-crop to imgsz x imgsz
    y0 = (new_h - imgsz) // 2
    x0 = (new_w - imgsz) // 2
    cropped = resized[y0 : y0 + imgsz, x0 : x0 + imgsz]

    # BGR -> RGB, normalize, HWC -> CHW -> NCHW
    tensor = cropped[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]
    return tensor


def preprocess_segment(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, dict]:
    """Preprocess for YOLO segmentation: letterbox, normalize.

    Args:
        img: BGR HWC uint8 image.
        imgsz: Target size (matches export imgsz).

    Returns:
        (tensor, meta) where tensor is NCHW float32 and meta contains
        info needed to map coordinates back to the original image.
    """
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to imgsz with stride-32 alignment (gray=114)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, imgsz - new_h - pad_h,
        pad_w, imgsz - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # BGR -> RGB, normalize, HWC -> CHW -> NCHW
    tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]

    meta = {
        "scale": scale,
        "pad": (pad_w, pad_h),
        "orig_shape": (h, w),
        "imgsz": imgsz,
    }
    return tensor, meta


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def postprocess_classification(output: np.ndarray) -> np.ndarray:
    """Convert classification output to probabilities.

    Args:
        output: (1, num_classes) — may be raw logits or already softmax'd.

    Returns:
        (num_classes,) probability array.
    """
    probs = output[0]
    # YOLO-cls ONNX exports include softmax — skip if already probabilities
    if np.all(probs >= 0) and abs(probs.sum() - 1.0) < 0.01:
        return probs
    return _softmax(probs)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _prepare_quad_iou(corners: np.ndarray):
    """Prepare the OpenCV hull and area reused by polygon-IoU comparisons."""
    try:
        hull = cv2.convexHull(corners.astype(np.float32))
        return hull, cv2.contourArea(hull)
    except Exception:
        return None


def _prepared_quad_iou(a, b) -> float:
    """Polygon IoU for two values returned by :func:`_prepare_quad_iou`."""
    if a is None or b is None:
        return 0.0
    ha, area_a = a
    hb, area_b = b
    try:
        inter, _ = cv2.intersectConvexConvex(ha, hb)
        if inter <= 0:
            return 0.0
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Polygon IoU between two 4-corner quads.

    Used for NMS on cube faces: adjacent faces share an edge but their quads
    barely overlap (~0 IoU), so this keeps distinct faces while removing true
    duplicate detections (~high IoU). Axis-aligned bbox IoU cannot — adjacent
    faces' bounding boxes overlap heavily and would be wrongly suppressed.
    """
    return _prepared_quad_iou(_prepare_quad_iou(a), _prepare_quad_iou(b))


def postprocess_pose(
    det_output: np.ndarray,
    meta: dict,
    conf_thresh: float = 0.75,
    iou_thresh: float = 0.15,
    nc: int = 1,
    nk: int = 4,
    kpt_conf_thresh: float = 0.5,
    min_visible_kpts: int = 1,
    snap_corners: bool = True,
) -> list[dict]:
    """Decode YOLO-pose ONNX output into keypoint detections.

    Args:
        det_output: shape (1, 4+nc+nk*3, N) — boxes, class probs, keypoints.
        meta: dict from preprocess_segment with scale/pad/orig_shape.
        conf_thresh: Minimum confidence to keep a detection.
        iou_thresh: IoU threshold for NMS.
        nc: Number of classes.
        nk: Number of keypoints per detection.
        kpt_conf_thresh: Minimum per-keypoint confidence for a keypoint to be
            considered visible.
        min_visible_kpts: Minimum number of keypoints that must exceed
            kpt_conf_thresh for the detection to be kept.

    Returns:
        List of dicts: {"corners": ndarray(nk,2), "kpt_conf": ndarray(nk,),
                        "confidence": float, "bbox": ndarray(4,)}
    """
    preds = det_output[0].T  # (N, 4+nc+nk*3)

    boxes_xywh = preds[:, :4]
    class_probs = preds[:, 4:4 + nc]
    kpts_flat = preds[:, 4 + nc:]  # (N, nk*3)

    if nc == 1:
        max_probs = class_probs[:, 0]
    else:
        max_probs = class_probs.max(axis=1)

    keep = max_probs >= conf_thresh
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    max_probs = max_probs[keep]
    kpts_all = kpts_flat[keep].reshape(-1, nk, 3)  # (M, nk, 3) in model space

    # Drop detections with too few confident keypoints (occluded/garbage quads)
    # before NMS, so their unreliable corners don't suppress good faces.
    if kpt_conf_thresh > 0:
        n_visible = (kpts_all[:, :, 2] >= kpt_conf_thresh).sum(axis=1)
    else:
        n_visible = np.full(len(kpts_all), nk)
    vmask = n_visible >= min_visible_kpts
    if not np.any(vmask):
        return []
    boxes_xywh = boxes_xywh[vmask]
    max_probs = max_probs[vmask]
    kpts_all = kpts_all[vmask]

    # NMS on the predicted QUADS (polygon IoU), not axis-aligned bounding boxes.
    # Adjacent cube faces share an edge but their quads barely overlap, so this
    # keeps all visible faces while merging true duplicate detections. Bbox NMS
    # collapsed adjacent faces together (their boxes overlap heavily).
    corners_model = kpts_all[:, :, :2].astype(np.float32)
    prepared_quads = [_prepare_quad_iou(corners) for corners in corners_model]
    order = np.argsort(-max_probs)
    indices = []
    for i in order:
        if all(_prepared_quad_iou(prepared_quads[i], prepared_quads[j])
               < iou_thresh for j in indices):
            indices.append(int(i))

    # Unscale coordinates from model space to original image
    scale = meta["scale"]
    pad_w, pad_h = meta["pad"]
    orig_h, orig_w = meta["orig_shape"]

    results = []
    for idx in indices:
        bx, by, bw, bh = boxes_xywh[idx]
        bx = (bx - pad_w) / scale
        by = (by - pad_h) / scale
        bw = bw / scale
        bh = bh / scale

        kpts = kpts_all[idx]  # (nk, 3) = (x, y, conf)
        kpt_conf = kpts[:, 2]

        corners = np.empty((nk, 2), dtype=np.float32)
        corners[:, 0] = (kpts[:, 0] - pad_w) / scale
        corners[:, 1] = (kpts[:, 1] - pad_h) / scale

        # Clip to image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h)

        results.append({
            "corners": corners,
            "kpt_conf": kpt_conf.astype(np.float32),
            "confidence": float(max_probs[idx]),
            "bbox": np.array([bx, by, bw, bh], dtype=np.float32),
        })

    # Merge shared corners across adjacent faces so the quads tessellate
    if snap_corners and len(results) > 1:
        _snap_shared_corners(results, abs_cap=0.10 * orig_w)

    return results


def _quad_edge_area(c: np.ndarray) -> tuple[float, float]:
    """Mean edge length and polygon area (shoelace) of a 4-corner quad."""
    e = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return float(e.mean()), area


def _quad_ok(c: np.ndarray, ref_area: float) -> bool:
    """A snap is only allowed if it keeps the quad convex and not collapsed."""
    _, area = _quad_edge_area(c)
    if area < 0.5 * ref_area:
        return False
    # All 4 corners on the convex hull => still a proper convex quad
    return len(cv2.convexHull(c.astype(np.float32))) >= 4


def _snap_shared_corners(
    detections: list[dict],
    max_frac: float = 0.60,
    abs_cap: float = 160.0,
) -> list[dict]:
    """Merge corners shared between adjacent faces so the quads tessellate.

    Adjacent cube faces share an EDGE (2 corners) and 3 faces meet at one centre
    vertex. Each face is detected independently, so shared corners land at slightly
    different points and the quads don't connect. We match faces by their closest
    EDGE — comparing both endpoints jointly over all edge pairs and both
    orientations — rather than by individual corners: the corner nearer the cube
    centre is usually ~0px apart, which pulls its partner corner into the match even
    when that partner is farther apart on its own, so BOTH ends of a shared edge
    merge (not just the near one — the failure in RUR'U' f14). Matched corner
    instances are unioned, so a vertex shared by 3 faces collapses to one cluster,
    then each cluster is set to its confidence-weighted centroid. A per-corner
    degeneracy guard rejects any merge that would make a quad non-convex or
    collapse its area, so a wrong match can never distort a quad into a sliver.

    Args:
        detections: list of dicts with 'corners' (nk,2) and 'kpt_conf' (nk,).
        max_frac: max per-edge average corner distance (as a fraction of the
            smaller face's mean edge) for two edges to count as a shared edge.
        abs_cap: hard upper bound (px) on that distance, regardless of face size.
    """
    n = len(detections)
    if n < 2:
        return detections

    edges = [_quad_edge_area(d["corners"])[0] for d in detections]
    items, pts, confs, face_start = [], [], [], {}
    for fi, d in enumerate(detections):
        face_start[fi] = len(items)
        for ci in range(len(d["corners"])):
            items.append((fi, ci))
            pts.append(d["corners"][ci])
            confs.append(float(d["kpt_conf"][ci]))
    pts = np.asarray(pts, dtype=np.float64)
    confs = np.asarray(confs, dtype=np.float64)
    N = len(items)

    parent = list(range(N))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # Edge-based matching: for each face pair find the closest pair of edges and,
    # if close enough, union BOTH of that edge's corners.
    for a in range(n):
        ca = detections[a]["corners"]
        na = len(ca)
        for b in range(a + 1, n):
            cb = detections[b]["corners"]
            nb = len(cb)
            thresh = min(max_frac * min(edges[a], edges[b]), abs_cap)
            best_cost, best_pairs = None, None
            for i in range(na):
                ai, ai1 = i, (i + 1) % na
                for j in range(nb):
                    bj, bj1 = j, (j + 1) % nb
                    c1 = (np.linalg.norm(ca[ai] - cb[bj]) +
                          np.linalg.norm(ca[ai1] - cb[bj1]))
                    c2 = (np.linalg.norm(ca[ai] - cb[bj1]) +
                          np.linalg.norm(ca[ai1] - cb[bj]))
                    if c2 < c1:
                        cost, pairs = c2, ((ai, bj1), (ai1, bj))
                    else:
                        cost, pairs = c1, ((ai, bj), (ai1, bj1))
                    if best_cost is None or cost < best_cost:
                        best_cost, best_pairs = cost, pairs
            if best_pairs is not None and best_cost / 2.0 <= thresh:
                for ac, bc in best_pairs:
                    ra = find(face_start[a] + ac)
                    rb = find(face_start[b] + bc)
                    if ra != rb:
                        parent[ra] = rb

    # Confidence-weighted centroid per cluster that spans >= 2 faces
    clusters: dict[int, list[int]] = {}
    for i in range(N):
        clusters.setdefault(find(i), []).append(i)
    target: dict[tuple, np.ndarray] = {}
    for members in clusters.values():
        if len({items[m][0] for m in members}) < 2:
            continue
        w = confs[members]
        P = pts[members]
        centroid = (P * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else P.mean(0)
        for m in members:
            target[items[m]] = centroid

    # Apply per face: accept each shared-corner merge that keeps the quad valid
    for fi, d in enumerate(detections):
        cand = d["corners"].copy()
        ref_area = _quad_edge_area(cand)[1]
        changed = False
        for ci in range(len(cand)):
            if (fi, ci) not in target:
                continue
            test = cand.copy()
            test[ci] = target[(fi, ci)]
            if _quad_ok(test, ref_area):  # degeneracy guard, incremental
                cand = test
                changed = True
        if changed:
            d["corners"] = cand.astype(np.float32)

    return detections


def postprocess_segmentation(
    det_output: np.ndarray,
    proto_output: np.ndarray,
    meta: dict,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.7,
    nc: int = 1,
    mask_overlap_thresh: float | None = None,
) -> list[dict]:
    """Decode YOLO-seg ONNX output into polygon detections.

    Args:
        det_output: shape (1, 4+nc+32, N) — boxes, class probs, mask coefficients.
        proto_output: shape (1, 32, mask_h, mask_w) — mask prototypes.
        meta: dict from preprocess_segment with scale/pad/orig_shape.
        conf_thresh: Minimum confidence to keep a detection.
        iou_thresh: IoU threshold for NMS.
        nc: Number of classes.
        mask_overlap_thresh: if set, run a second NMS on the decoded mask
            polygons using intersection-over-min-area. Box NMS at iou_thresh
            misses same-sticker duplicates whose boxes differ in size (box
            IoU < iou_thresh) while the masks coincide. None = legacy off.

    Returns:
        List of dicts: {"polygon": np.ndarray(N,2), "confidence": float, "class_id": int}
    """
    # Transpose: (1, 4+nc+32, N) -> (N, 4+nc+32)
    preds = det_output[0].T

    # Split columns
    boxes_xywh = preds[:, :4]
    class_probs = preds[:, 4 : 4 + nc]
    mask_coeffs = preds[:, 4 + nc : 4 + nc + 32]

    # Max class confidence per detection
    if nc == 1:
        max_probs = class_probs[:, 0]
        class_ids = np.zeros(len(preds), dtype=np.int32)
    else:
        max_probs = class_probs.max(axis=1)
        class_ids = class_probs.argmax(axis=1)

    # Filter by confidence
    keep = max_probs >= conf_thresh
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    max_probs = max_probs[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    # Convert xywh (center) to x,y,w,h for cv2.dnn.NMSBoxes
    # cv2 expects [x_topleft, y_topleft, width, height]
    nms_boxes = np.empty_like(boxes_xywh)
    nms_boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x_tl
    nms_boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y_tl
    nms_boxes[:, 2] = boxes_xywh[:, 2]  # w
    nms_boxes[:, 3] = boxes_xywh[:, 3]  # h

    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), max_probs.tolist(), conf_thresh, iou_thresh
    )
    if len(indices) == 0:
        return []
    indices = np.asarray(indices).flatten()

    # Mask reconstruction
    protos = proto_output[0]  # (32, mask_h, mask_w)
    mask_h, mask_w = protos.shape[1], protos.shape[2]
    imgsz = meta["imgsz"]
    pad_w, pad_h = meta["pad"]
    orig_h, orig_w = meta["orig_shape"]

    # Decode every survivor's mask in ONE matmul. The per-detection gemv on a
    # strided view of the transposed output measured ~40ms PER CALL (BLAS
    # thread-sync pathology on tiny inputs, 2026-06-12 profile of a real
    # solve); one contiguous gemm for all of them is ~1ms total.
    coeffs_all = np.ascontiguousarray(mask_coeffs[indices])  # (K, 32)
    raw_all = _sigmoid(coeffs_all @ protos.reshape(32, -1)).reshape(
        -1, mask_h, mask_w)

    unpad_w = imgsz - 2 * pad_w
    unpad_h = imgsz - 2 * pad_h
    # Crop-first fast path: when both upscales are INTEGER ratios (1080p is
    # exactly 3x, 720p 2x, 4K 6x; protos->imgsz is 4x), a resize of an
    # integer-aligned crop samples the identical interpolation grid as the
    # full-image resize, so thresholding/contouring only the detection's bbox
    # neighborhood yields byte-identical polygons at a fraction of the cost
    # (the legacy chain inflated each 160x160 mask to two full-frame float
    # images + a full-frame findContours — ~75ms/detection at 1080p).
    k1x, r1x = divmod(imgsz, mask_w)
    k1y, r1y = divmod(imgsz, mask_h)
    k2x, r2x = divmod(orig_w, unpad_w) if unpad_w > 0 else (0, 1)
    k2y, r2y = divmod(orig_h, unpad_h) if unpad_h > 0 else (0, 1)
    fast = not (r1x or r1y or r2x or r2y) and min(k1x, k1y, k2x, k2y) > 0

    results = []
    for n, idx in enumerate(indices):
        # Crop mask to detection bounding box (matches Ultralytics behavior).
        # Without this, shared mask prototypes bleed across the full image.
        bx, by, bw, bh = boxes_xywh[idx]
        x1 = int(max(0, bx - bw / 2))
        y1 = int(max(0, by - bh / 2))
        x2 = int(min(imgsz, bx + bw / 2))
        y2 = int(min(imgsz, by + bh / 2))

        if fast:
            decoded = _decode_polygon_cropped(
                raw_all[n], (x1, y1, x2, y2), (pad_w, pad_h),
                (unpad_w, unpad_h), (k1x, k1y), (k2x, k2y),
                (mask_w, mask_h))
        else:
            decoded = _decode_polygon_full(
                raw_all[n], (x1, y1, x2, y2), (pad_w, pad_h),
                imgsz, (orig_w, orig_h))
        if decoded is None:
            continue

        results.append({
            "polygon": decoded,
            "confidence": float(max_probs[idx]),
            "class_id": int(class_ids[idx]),
        })

    if mask_overlap_thresh is not None and len(results) > 1:
        results = _mask_nms(results, (orig_h, orig_w), mask_overlap_thresh)

    return results


def _decode_polygon_full(raw_mask, bbox, pad, imgsz, orig_wh):
    """Legacy full-frame mask decode (reference path; non-integer ratios).

    raw_mask: (mask_h, mask_w) sigmoid mask. Returns the largest-contour
    polygon in original-image coordinates, or None.
    """
    pad_w, pad_h = pad
    orig_w, orig_h = orig_wh
    x1, y1, x2, y2 = bbox

    mask_full = cv2.resize(
        raw_mask.astype(np.float32), (imgsz, imgsz),
        interpolation=cv2.INTER_LINEAR,
    )
    cropped = np.zeros_like(mask_full)
    cropped[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]

    mask_unpadded = cropped[pad_h: imgsz - pad_h, pad_w: imgsz - pad_w]
    if mask_unpadded.size == 0:
        return None

    mask_orig = cv2.resize(
        mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    binary = (mask_orig > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    polygon = largest.reshape(-1, 2).astype(np.float32)
    return polygon if len(polygon) >= 3 else None


def _decode_polygon_cropped(raw_mask, bbox, pad, unpad_wh, k1, k2, mask_wh):
    """Bbox-cropped mask decode — byte-identical to _decode_polygon_full when
    every upscale ratio is an integer.

    Why exact: cv2.resize with integer scale k maps dst pixel X to src
    coordinate (X+0.5)/k - 0.5, so a crop starting at integer src column `a`
    resized to k*size samples the same grid as the full resize shifted by
    k*a. The only divergence is border clamping at crop edges — avoided by
    (a) only trusting dst pixels whose src neighborhoods lie strictly inside
    the crop, guaranteed by a 2px source margin around the detection bbox,
    and (b) the mask being explicitly zeroed outside the bbox, so every
    pixel that can exceed the 0.5 threshold interpolates from inside the
    margin. Contours of the identical binary region are identical up to the
    integer crop offset, which is added back.
    """
    pad_w, pad_h = pad
    unpad_w, unpad_h = unpad_wh
    k1x, k1y = k1
    k2x, k2y = k2
    mask_w, mask_h = mask_wh
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    # Stage 1: upscale only the bbox neighborhood of the proto-res mask.
    cx1 = max(0, x1 // k1x - 2)
    cy1 = max(0, y1 // k1y - 2)
    cx2 = min(mask_w, -(-x2 // k1x) + 2)
    cy2 = min(mask_h, -(-y2 // k1y) + 2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    m1 = cv2.resize(
        np.ascontiguousarray(raw_mask[cy1:cy2, cx1:cx2], dtype=np.float32),
        ((cx2 - cx1) * k1x, (cy2 - cy1) * k1y),
        interpolation=cv2.INTER_LINEAR,
    )
    ox, oy = cx1 * k1x, cy1 * k1y  # crop origin in model-input space

    # Zero outside the bbox (in place — the crop covers bbox by construction).
    rx1, rx2 = x1 - ox, x2 - ox
    ry1, ry2 = y1 - oy, y2 - oy
    m1[:ry1, :] = 0
    m1[ry2:, :] = 0
    m1[:, :rx1] = 0
    m1[:, rx2:] = 0

    # Stage 2: unpad + upscale only the (bbox ± 2px zero margin) region.
    sx1 = max(0, x1 - pad_w - 2)
    sy1 = max(0, y1 - pad_h - 2)
    sx2 = min(unpad_w, x2 - pad_w + 2)
    sy2 = min(unpad_h, y2 - pad_h + 2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None  # bbox entirely inside the letterbox padding
    sub = m1[sy1 + pad_h - oy: sy2 + pad_h - oy,
             sx1 + pad_w - ox: sx2 + pad_w - ox]
    m2 = cv2.resize(
        np.ascontiguousarray(sub),
        ((sx2 - sx1) * k2x, (sy2 - sy1) * k2y),
        interpolation=cv2.INTER_LINEAR,
    )

    binary = (m2 > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    polygon = largest.reshape(-1, 2).astype(np.float32)
    if len(polygon) < 3:
        return None
    polygon[:, 0] += sx1 * k2x
    polygon[:, 1] += sy1 * k2y
    return polygon


def _mask_nms(results: list[dict], orig_shape: tuple, overlap_thresh: float) -> list[dict]:
    """Greedy NMS on decoded mask polygons, intersection-over-min-area.

    Rasterized at 1/4 scale; bbox prefilter skips disjoint pairs.
    """
    ds = 4
    h, w = max(1, orig_shape[0] // ds), max(1, orig_shape[1] // ds)
    order = sorted(range(len(results)), key=lambda i: -results[i]["confidence"])
    masks: dict[int, tuple] = {}   # i -> (crop, (x0, y0, x1, y1))
    areas: dict[int, float] = {}
    bounds: dict[int, np.ndarray] = {}

    def raster(i):
        # Rasterize each polygon only on its own bbox-sized canvas — fillPoly
        # of integer-shifted vertices is translation-invariant, and clipping
        # at canvas borders coincides with the full-frame canvas's, so crop
        # contents (and the pair intersections below) are byte-identical to
        # the original full-frame rasters at a fraction of the cost.
        if i not in masks:
            poly = (results[i]["polygon"] / ds).astype(np.int32)
            bounds[i] = np.array([poly[:, 0].min(), poly[:, 1].min(),
                                  poly[:, 0].max(), poly[:, 1].max()])
            x0 = max(0, int(bounds[i][0]))
            y0 = max(0, int(bounds[i][1]))
            x1 = min(w, int(bounds[i][2]) + 1)
            y1 = min(h, int(bounds[i][3]) + 1)
            if x1 <= x0 or y1 <= y0:
                m, rect = np.zeros((0, 0), dtype=np.uint8), (0, 0, 0, 0)
            else:
                m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.fillPoly(m, [poly - np.array([x0, y0], dtype=np.int32)], 1)
                rect = (x0, y0, x1, y1)
            masks[i] = (m, rect)
            areas[i] = float(m.sum())

    def intersection(i, k):
        mi, (ix0, iy0, ix1, iy1) = masks[i]
        mk, (kx0, ky0, kx1, ky1) = masks[k]
        x0, y0 = max(ix0, kx0), max(iy0, ky0)
        x1, y1 = min(ix1, kx1), min(iy1, ky1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        a = mi[y0 - iy0: y1 - iy0, x0 - ix0: x1 - ix0]
        b = mk[y0 - ky0: y1 - ky0, x0 - kx0: x1 - kx0]
        return float(np.logical_and(a, b).sum())

    kept: list[int] = []
    for i in order:
        raster(i)
        if areas[i] == 0:
            continue
        dup = False
        for k in kept:
            bi, bk = bounds[i], bounds[k]
            if bi[0] > bk[2] or bk[0] > bi[2] or bi[1] > bk[3] or bk[1] > bi[3]:
                continue
            if intersection(i, k) / min(areas[i], areas[k]) > overlap_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)

    kept.sort()
    return [results[i] for i in kept]
