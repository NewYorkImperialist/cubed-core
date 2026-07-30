from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ._runtime import require_vision_runtime
from .geometry import GeometryError, polygon_area
from .types import (
    BBoxXYXY,
    FaceDetection,
    VisionContractError,
    coerce_face_detection,
)


class TrackerError(ValueError):
    """Raised when tracker configuration, input, or injected output is invalid."""


class VisionInference(Protocol):
    def classify_alignment(self, frame: Any) -> float: ...

    def infer_faces(self, frame: Any) -> Sequence[FaceDetection]: ...


class FaceReader(Protocol):
    def __call__(
        self,
        frame: Any,
        faces: Sequence[FaceDetection],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class FrameTrackerConfig:
    fps: float
    rotation_degrees: int = 0
    gate_enabled: bool = True
    alignment_threshold: float = 0.5
    minimum_aligned_frames: int = 3
    minimum_face_confidence: float = 0.5
    minimum_keypoint_confidence: float = 0.5
    minimum_visible_keypoints: int = 1
    minimum_quad_area: float = 1.0
    maximum_faces: int = 16

    def __post_init__(self) -> None:
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(float(self.fps))
            or not 1.0 <= float(self.fps) <= 1000.0
        ):
            raise TrackerError("fps must be a finite number between 1 and 1000")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise TrackerError("rotation_degrees must be 0, 90, 180, or 270")
        if type(self.gate_enabled) is not bool:
            raise TrackerError("gate_enabled must be a boolean")
        for field_name in (
            "alignment_threshold",
            "minimum_face_confidence",
            "minimum_keypoint_confidence",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise TrackerError(f"{field_name} must be a finite probability")
        if (
            type(self.minimum_aligned_frames) is not int
            or not 1 <= self.minimum_aligned_frames <= 100_000
        ):
            raise TrackerError("minimum_aligned_frames must be a positive integer")
        if (
            type(self.minimum_visible_keypoints) is not int
            or not 1 <= self.minimum_visible_keypoints <= 4
        ):
            raise TrackerError("minimum_visible_keypoints must be between 1 and 4")
        if (
            isinstance(self.minimum_quad_area, bool)
            or not isinstance(self.minimum_quad_area, (int, float))
            or not math.isfinite(float(self.minimum_quad_area))
            or float(self.minimum_quad_area) <= 0.0
        ):
            raise TrackerError("minimum_quad_area must be finite and positive")
        if type(self.maximum_faces) is not int or not 1 <= self.maximum_faces <= 1000:
            raise TrackerError("maximum_faces must be an integer between 1 and 1000")


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """The complete camera-only evidence receipt for one processed frame."""

    frame_index: int
    timestamp_seconds: float
    alignment_confidence: float | None
    alignment_streak: int
    gate_open: bool
    faces: tuple[FaceDetection, ...]
    motion: float | None
    event_motion: float | None
    motion_bbox: BBoxXYXY | None
    read_result: Any | None


@dataclass(frozen=True, slots=True)
class FrameTrackerSnapshot:
    records: tuple[FrameRecord, ...]
    authoritative_motion_bbox: BBoxXYXY | None

    @property
    def alignment_scores(self) -> dict[int, float]:
        return {
            record.frame_index: record.alignment_confidence
            for record in self.records
            if record.alignment_confidence is not None
        }

    @property
    def event_motions(self) -> dict[int, float | None]:
        return {
            record.frame_index: record.event_motion
            for record in self.records
            if record.motion_bbox is not None
        }

    @property
    def reads(self) -> dict[int, Any]:
        return {
            record.frame_index: record.read_result
            for record in self.records
            if record.read_result is not None
        }


def _frame(value: Any) -> Any:
    _, numpy = require_vision_runtime()
    frame = numpy.asarray(value)
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
        raise TrackerError("frame must be a nonempty HxWx3 array")
    if frame.dtype != numpy.uint8:
        raise TrackerError("frame must use uint8 BGR pixels")
    return frame


def _probability(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TrackerError(f"{field} must be a finite probability")
    return float(value)


def rotate_frame(frame: Any, rotation_degrees: int) -> Any:
    cv2, _ = require_vision_runtime()
    source = _frame(frame)
    rotation_code = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation_degrees)
    if rotation_degrees not in {0, 90, 180, 270}:
        raise TrackerError("rotation_degrees must be 0, 90, 180, or 270")
    return source if rotation_code is None else cv2.rotate(source, rotation_code)


def face_motion_bbox(
    faces: Sequence[FaceDetection],
    *,
    frame_shape: Sequence[int],
) -> BBoxXYXY | None:
    """Return a clipped, slice-ready union of admitted face corners."""

    _, numpy = require_vision_runtime()
    if not faces:
        return None
    if len(frame_shape) < 2 or min(frame_shape[:2]) <= 0:
        raise TrackerError("frame_shape must contain positive height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    points = numpy.asarray(
        [point for face in faces for point in face.corners],
        dtype=numpy.float64,
    )
    x0 = max(0, int(numpy.floor(points[:, 0].min())))
    y0 = max(0, int(numpy.floor(points[:, 1].min())))
    x1 = min(width, int(points[:, 0].max()))
    y1 = min(height, int(points[:, 1].max()))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def motion_in_bbox(
    current_gray: Any,
    previous_gray: Any | None,
    bbox: BBoxXYXY | None,
) -> float | None:
    """Return mean absolute grayscale difference inside one immutable ROI."""

    _, numpy = require_vision_runtime()
    if previous_gray is None or bbox is None:
        return None
    current = numpy.asarray(current_gray)
    previous = numpy.asarray(previous_gray)
    if current.ndim != 2 or previous.ndim != 2 or current.shape != previous.shape:
        raise TrackerError("motion frames must be same-shaped grayscale arrays")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= current.shape[1] and 0 <= y0 < y1 <= current.shape[0]):
        raise TrackerError("motion bbox lies outside the grayscale frame")
    current_roi = current[y0:y1, x0:x1].astype(numpy.float32)
    previous_roi = previous[y0:y1, x0:x1].astype(numpy.float32)
    return float(numpy.abs(current_roi - previous_roi).mean())


def admit_faces(
    values: Sequence[FaceDetection | dict[str, Any]],
    *,
    frame_shape: Sequence[int],
    config: FrameTrackerConfig,
) -> tuple[FaceDetection, ...]:
    """Validate pose output and retain only geometrically admissible faces."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TrackerError("face inference output must be a sequence")
    if len(values) > config.maximum_faces:
        raise TrackerError(f"face inference output exceeds the {config.maximum_faces}-face limit")
    if len(frame_shape) < 2 or min(frame_shape[:2]) <= 0:
        raise TrackerError("frame_shape must contain positive height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    admitted: list[FaceDetection] = []
    for index, value in enumerate(values):
        try:
            face = coerce_face_detection(value)
        except VisionContractError as exc:
            raise TrackerError(f"face inference output {index} is invalid: {exc}") from exc
        if face.confidence < config.minimum_face_confidence:
            continue
        visible = sum(
            confidence >= config.minimum_keypoint_confidence
            for confidence in face.keypoint_confidences
        )
        if visible < config.minimum_visible_keypoints:
            continue
        if any(x < 0.0 or x > width or y < 0.0 or y > height for x, y in face.corners):
            continue
        try:
            area = polygon_area(face.corners)
        except GeometryError:
            continue
        if area < config.minimum_quad_area:
            continue
        admitted.append(face)
    admitted.sort(
        key=lambda face: (
            -face.confidence,
            sum(point[1] for point in face.corners),
            sum(point[0] for point in face.corners),
        )
    )
    return tuple(admitted)


class FrameTracker:
    """Deterministic alignment-streak, pose, read, and motion state machine."""

    def __init__(
        self,
        *,
        config: FrameTrackerConfig,
        inference: VisionInference,
        reader: FaceReader | None = None,
    ):
        self.config = config
        self.inference = inference
        self.reader = reader
        self._alignment_streak = 0
        self._previous_gray: Any | None = None
        self._motion_bbox: BBoxXYXY | None = None
        self._records: list[FrameRecord] = []

    @property
    def frame_count(self) -> int:
        return len(self._records)

    def snapshot(self) -> FrameTrackerSnapshot:
        return FrameTrackerSnapshot(
            records=tuple(self._records),
            authoritative_motion_bbox=self._motion_bbox,
        )

    def feed(self, frame: Any) -> FrameRecord:
        rotated = rotate_frame(frame, self.config.rotation_degrees)
        alignment_confidence = None
        if self.config.gate_enabled:
            alignment_confidence = _probability(
                self.inference.classify_alignment(rotated),
                field="alignment confidence",
            )
        return self._process_rotated(rotated, alignment_confidence)

    def feed_batch(self, frames: Sequence[Any]) -> tuple[FrameRecord, ...]:
        if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
            raise TrackerError("frames must be a sequence")
        if not frames:
            return ()
        rotated = [rotate_frame(frame, self.config.rotation_degrees) for frame in frames]
        if self.config.gate_enabled:
            batch_method = getattr(self.inference, "classify_alignment_batch", None)
            if callable(batch_method):
                raw_scores = batch_method(rotated)
            else:
                raw_scores = [self.inference.classify_alignment(frame) for frame in rotated]
            if (
                isinstance(raw_scores, (str, bytes))
                or not isinstance(raw_scores, Sequence)
                or len(raw_scores) != len(rotated)
            ):
                raise TrackerError(
                    "alignment batch returned a different number of scores than frames"
                )
            scores = tuple(
                _probability(score, field=f"alignment confidence {index}")
                for index, score in enumerate(raw_scores)
            )
        else:
            scores = (None,) * len(rotated)
        return tuple(
            self._process_rotated(frame, score)
            for frame, score in zip(rotated, scores, strict=True)
        )

    def _process_rotated(
        self,
        frame: Any,
        alignment_confidence: float | None,
    ) -> FrameRecord:
        cv2, _ = require_vision_runtime()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.gate_enabled:
            if alignment_confidence is None:
                raise TrackerError("gated processing requires an alignment confidence")
            next_streak = (
                self._alignment_streak + 1
                if alignment_confidence >= self.config.alignment_threshold
                else 0
            )
            gate_open = next_streak >= self.config.minimum_aligned_frames
        else:
            next_streak = self._alignment_streak
            gate_open = True

        held_bbox = self._motion_bbox
        event_motion = motion_in_bbox(gray, self._previous_gray, held_bbox)
        faces: tuple[FaceDetection, ...] = ()
        motion = None
        read_result = None
        next_bbox = held_bbox
        if gate_open:
            raw_faces = self.inference.infer_faces(frame)
            faces = admit_faces(
                raw_faces,
                frame_shape=frame.shape,
                config=self.config,
            )
            if faces:
                detected_bbox = face_motion_bbox(faces, frame_shape=frame.shape)
                motion = motion_in_bbox(gray, self._previous_gray, detected_bbox)
                if detected_bbox is not None:
                    next_bbox = detected_bbox
                    event_motion = motion
                if self.reader is not None:
                    read_result = self.reader(frame, faces)

        record = FrameRecord(
            frame_index=len(self._records) + 1,
            timestamp_seconds=(len(self._records) + 1) / float(self.config.fps),
            alignment_confidence=alignment_confidence,
            alignment_streak=next_streak,
            gate_open=gate_open,
            faces=faces,
            motion=motion,
            event_motion=event_motion,
            motion_bbox=next_bbox,
            read_result=read_result,
        )
        # Commit only after all injected calls and contract checks succeed.
        self._alignment_streak = next_streak
        self._previous_gray = gray
        self._motion_bbox = next_bbox
        self._records.append(record)
        return record
