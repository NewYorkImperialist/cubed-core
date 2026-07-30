"""
Inference client — runs ONNX models locally.

Models:
  - Alignment classifier (YOLO-cls): aligned vs unaligned
  - Face pose model (YOLO-pose): 4 corner keypoints per face
  - Sticker segmentation model (YOLO-seg, 1-class): sticker masks
"""

import logging
import os
import threading

import numpy as np

from detect.onnx_runtime import (
    OnnxModel,
    preprocess_classify,
    preprocess_segment,
    postprocess_classification,
    postprocess_pose,
    postprocess_segmentation,
)

log = logging.getLogger(__name__)

ALIGNED_IDX = 0
CLASSIFY_IMGSZ = 224
FACE_IMGSZ = 1024
STICKER_IMGSZ = 640


class LocalInference:
    """Run ONNX models locally.

    Three models:
      - Alignment classifier (224px, 2-class)
      - Face pose model (1024px, 1-class, 4 keypoints)
      - Sticker segmentation model (640px, 1-class)
    """

    def __init__(self):
        # CUBED_ALIGNED_MODEL lets us A/B a retrained alignment classifier through
        # the full pipeline WITHOUT deploying over prod (default = prod weights).
        self.aligned_model = OnnxModel(
            os.environ.get("CUBED_ALIGNED_MODEL", "weights/cube_aligned.onnx"))

        # Face pose model
        face_path = os.environ.get(
            "CUBED_FACE_POSE_MODEL", "weights/cube_face_pose.onnx")
        if os.path.exists(face_path):
            self.face_model = OnnxModel(face_path)
            log.info("Loaded face pose model: %s", face_path)
        else:
            self.face_model = None
            log.warning("Face pose model not found: %s", face_path)

        # Sticker segmentation is not consumed by the default production pose
        # reader.  Keep only its path here and create the ONNX session on the
        # first calibration/legacy-segmentation call; eagerly reserving its GPU
        # arena penalized every pose-only solve for a model it never invoked.
        self.sticker_model = None
        self._sticker_model_lock = threading.Lock()
        self._sticker_model_path = None
        sticker_path = "weights/cube_sticker_seg.onnx"
        if os.path.exists(sticker_path):
            self._sticker_model_path = sticker_path
            self._sticker_nc = 1
            log.info("Sticker seg model available (lazy): %s", sticker_path)
        else:
            # Fall back to old 2-class model
            fallback = "weights/cube_side_sticker_seg.onnx"
            if os.path.exists(fallback):
                self._sticker_model_path = fallback
                self._sticker_nc = 2
                log.info("Legacy 2-class sticker model available (lazy): %s",
                         fallback)
            else:
                self._sticker_nc = 1
                log.warning("No sticker model found")
            # Preserve the historical fallback constructor shape: when the
            # preferred one-class model is absent, initialization ended here.
            return

        # Legacy side model (only if face pose model not available)
        if self.face_model is None:
            side_path = "weights/cube_side_seg.onnx"
            if os.path.exists(side_path):
                self.side_model = OnnxModel(side_path)
                log.info("Loaded legacy side model: %s", side_path)
            else:
                self.side_model = None
        else:
            self.side_model = None

    def _ensure_sticker_model(self):
        """Create the optional sticker session once, at its first real use."""
        if self.sticker_model is not None or self._sticker_model_path is None:
            return self.sticker_model
        with self._sticker_model_lock:
            if self.sticker_model is None:
                self.sticker_model = OnnxModel(self._sticker_model_path)
                log.info("Loaded sticker seg model: %s",
                         self._sticker_model_path)
        return self.sticker_model

    def classify_alignment(self, frame: np.ndarray) -> float:
        """Run only the alignment classifier. Returns confidence."""
        tensor = preprocess_classify(frame, CLASSIFY_IMGSZ)
        outputs = self.aligned_model.run(tensor)
        probs = postprocess_classification(outputs[0])
        return float(probs[ALIGNED_IDX])

    def classify_alignment_batch(self, frames: list[np.ndarray]) -> list[float]:
        """Alignment-classifier confidence for N frames, batched into ONE
        session.run when the exported model accepts a dynamic batch dim;
        otherwise transparently falls back to one classify_alignment() call
        per frame (identical numeric result either way — same preprocessing,
        model, and postprocessing per frame, just grouped differently).

        Called by SolveFrameTracker.feed_batch (the monolithic decode groups
        up to server.CLS_BATCH frames per call). NOTE: the currently deployed
        weights/cube_aligned.onnx is exported with a FIXED batch-1 input
        ([1, 3, 224, 224], a literal int, not a symbolic dim — see
        OnnxModel.supports_batch), so today this always takes the per-frame
        fallback and there is NO session.run reduction yet; a future
        dynamic-batch re-export flips supports_batch True and batching
        activates here with no caller changes (perf-wave2, 2026-07-01)."""
        if not frames:
            return []
        if not self.aligned_model.supports_batch:
            return [self.classify_alignment(f) for f in frames]
        tensor = np.concatenate(
            [preprocess_classify(f, CLASSIFY_IMGSZ) for f in frames], axis=0)
        outputs = self.aligned_model.run(tensor)
        return [float(postprocess_classification(outputs[0][i:i + 1])[ALIGNED_IDX])
                for i in range(len(frames))]

    def infer_faces(
        self, frame: np.ndarray, conf: float = 0.75, kpt_conf: float = 0.5,
        snap_corners: bool = True,
    ) -> list[dict]:
        """Run face pose model. Returns list of {corners, kpt_conf, confidence}."""
        if self.face_model is None:
            return []
        tensor, meta = preprocess_segment(frame, FACE_IMGSZ)
        outputs = self.face_model.run(tensor)
        return postprocess_pose(
            outputs[0], meta, conf_thresh=conf, nc=1, nk=4,
            kpt_conf_thresh=kpt_conf, snap_corners=snap_corners,
        )

    def infer_stickers(self, frame: np.ndarray, conf: float = 0.5) -> list[dict]:
        """Run sticker segmentation model. Returns list of {polygon, confidence, class_id}."""
        model = self._ensure_sticker_model()
        if model is None:
            return []
        tensor, meta = preprocess_segment(frame, STICKER_IMGSZ)
        outputs = model.run(tensor)
        dets = postprocess_segmentation(
            outputs[0], outputs[1], meta,
            conf_thresh=conf, nc=self._sticker_nc,
        )
        if self._sticker_nc == 2:
            # Legacy 2-class model: only return sticker detections (class_id=1)
            return [d for d in dets if d["class_id"] == 1]
        return dets

    def infer_segmentation(self, frame: np.ndarray, side_conf: float = 0.9) -> dict:
        """Run face + sticker models (skip alignment classifier).

        Returns {"faces": [...], "stickers": [...]} for the new pipeline,
        or {"sides": [...], "stickers": [...]} for legacy compatibility.
        """
        sticker_dets = self.infer_stickers(frame, conf=0.5)

        if self.face_model is not None:
            faces = self.infer_faces(frame, conf=side_conf)
            return {"faces": faces, "stickers": sticker_dets}

        # Legacy path: extract sides from 2-class sticker model or dedicated side model
        sides = []
        stickers = []
        if self.side_model is not None:
            tensor, meta = preprocess_segment(frame, STICKER_IMGSZ)
            outputs = self.side_model.run(tensor)
            side_dets = postprocess_segmentation(
                outputs[0], outputs[1], meta, conf_thresh=side_conf, nc=1,
            )
            sides = [{"polygon": d["polygon"], "confidence": d["confidence"]} for d in side_dets]
            stickers = [{"polygon": d["polygon"], "confidence": d["confidence"]} for d in sticker_dets]
        else:
            for det in sticker_dets:
                entry = {"polygon": det["polygon"], "confidence": det["confidence"]}
                if det.get("class_id") == 0:
                    sides.append(entry)
                else:
                    stickers.append(entry)

        return {"sides": sides, "stickers": stickers}

    def infer(self, frame: np.ndarray, side_conf: float = 0.9) -> dict:
        """Run all models on a BGR frame."""
        aligned_conf = self.classify_alignment(frame)
        result = self.infer_segmentation(frame, side_conf=side_conf)
        result["aligned_conf"] = aligned_conf
        return result


_client = None


def get_inference_client():
    """Shared LocalInference instance. Construction loads alignment + pose ONNX
    sessions (sticker segmentation is lazy) — when this returned a FRESH
    instance per call, the live ritual paid that cost per frame and ran ~8x
    slower than the models themselves (2026-06-12)."""
    global _client
    if _client is None:
        _client = LocalInference()
    return _client
