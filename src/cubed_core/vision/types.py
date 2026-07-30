from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]
BBoxXYWH = tuple[float, float, float, float]
BBoxXYXY = tuple[int, int, int, int]


class VisionContractError(ValueError):
    """Raised when an inference or reader value violates the public vision contract."""


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisionContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VisionContractError(f"{field} must be a finite number")
    return result


def _probability(value: Any, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise VisionContractError(f"{field} must be between 0 and 1")
    return result


def _point(value: Any, *, field: str) -> Point:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise VisionContractError(f"{field} must contain x and y")
    return (
        _finite_number(value[0], field=f"{field}[0]"),
        _finite_number(value[1], field=f"{field}[1]"),
    )


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One four-corner face-pose detection in source-image pixel coordinates."""

    corners: tuple[Point, Point, Point, Point]
    keypoint_confidences: tuple[float, float, float, float]
    confidence: float
    bbox_xywh: BBoxXYWH

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise VisionContractError("face corners must contain exactly four points")
        corners = tuple(
            _point(point, field=f"face.corners[{index}]")
            for index, point in enumerate(self.corners)
        )
        if len(self.keypoint_confidences) != 4:
            raise VisionContractError("face keypoint_confidences must contain exactly four values")
        keypoint_confidences = tuple(
            _probability(value, field=f"face.keypoint_confidences[{index}]")
            for index, value in enumerate(self.keypoint_confidences)
        )
        confidence = _probability(self.confidence, field="face.confidence")
        if len(self.bbox_xywh) != 4:
            raise VisionContractError("face bbox_xywh must contain x, y, width, and height")
        bbox = tuple(
            _finite_number(value, field=f"face.bbox_xywh[{index}]")
            for index, value in enumerate(self.bbox_xywh)
        )
        if bbox[2] < 0.0 or bbox[3] < 0.0:
            raise VisionContractError("face bbox width and height must be nonnegative")
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "keypoint_confidences", keypoint_confidences)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox_xywh", bbox)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FaceDetection:
        allowed = {
            "corners",
            "keypoint_confidences",
            "kpt_conf",
            "confidence",
            "conf",
            "bbox_xywh",
            "bbox",
        }
        extra = value.keys() - allowed
        if extra:
            raise VisionContractError(
                f"face detection contains unsupported field {sorted(extra)[0]}"
            )
        try:
            corners_value = value["corners"]
            keypoint_value = value.get("keypoint_confidences", value.get("kpt_conf"))
            confidence_value = value.get("confidence", value.get("conf"))
            bbox_value = value.get("bbox_xywh", value.get("bbox"))
        except (KeyError, TypeError) as exc:
            raise VisionContractError("face detection is missing corners") from exc
        if keypoint_value is None:
            raise VisionContractError("face detection is missing keypoint confidences")
        if confidence_value is None:
            raise VisionContractError("face detection is missing confidence")
        if bbox_value is None:
            points = tuple(_point(point, field="face.corners") for point in corners_value)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            bbox_value = (
                (min(xs) + max(xs)) / 2.0,
                (min(ys) + max(ys)) / 2.0,
                max(xs) - min(xs),
                max(ys) - min(ys),
            )
        return cls(
            corners=tuple(corners_value),
            keypoint_confidences=tuple(keypoint_value),
            confidence=confidence_value,
            bbox_xywh=tuple(bbox_value),
        )


def coerce_face_detection(value: FaceDetection | Mapping[str, Any]) -> FaceDetection:
    if isinstance(value, FaceDetection):
        return value
    if isinstance(value, Mapping):
        return FaceDetection.from_mapping(value)
    raise VisionContractError("face detection must be a FaceDetection or mapping")


@dataclass(frozen=True, slots=True)
class FaceCellRead:
    """Nine deterministic Lab samples and their unmasked-pixel support."""

    slot: str
    lab: tuple[tuple[float, float, float], ...]
    confidence: tuple[float, ...]
    valid_pixels: tuple[int, ...]
    total_pixels: tuple[int, ...]
    used_fallback: tuple[bool, ...]
    relative_area: float
    corners: tuple[Point, Point, Point, Point]

    def __post_init__(self) -> None:
        if self.slot not in {"up", "front", "right"}:
            raise VisionContractError("face-cell slot must be up, front, or right")
        lengths = {
            len(self.lab),
            len(self.confidence),
            len(self.valid_pixels),
            len(self.total_pixels),
            len(self.used_fallback),
        }
        if lengths != {9}:
            raise VisionContractError("face-cell reads must contain exactly nine cells")
        for index, vector in enumerate(self.lab):
            if len(vector) != 3:
                raise VisionContractError(f"face-cell lab[{index}] must contain L, a, and b")
            for channel, value in enumerate(vector):
                _finite_number(value, field=f"face-cell lab[{index}][{channel}]")
        for index, value in enumerate(self.confidence):
            _probability(value, field=f"face-cell confidence[{index}]")
        for index, (valid, total) in enumerate(
            zip(self.valid_pixels, self.total_pixels, strict=True)
        ):
            if type(valid) is not int or type(total) is not int or not 0 <= valid <= total:
                raise VisionContractError(
                    f"face-cell pixel counts at index {index} must satisfy 0 <= valid <= total"
                )
        _probability(self.relative_area, field="face-cell relative_area")


@dataclass(frozen=True, slots=True)
class GeometricReadResult:
    """Reader output for the visible cube faces in one admitted frame."""

    faces: tuple[FaceCellRead, ...]

    @property
    def slot_reads(self) -> tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...]:
        return tuple((face.slot, face.lab) for face in self.faces)
