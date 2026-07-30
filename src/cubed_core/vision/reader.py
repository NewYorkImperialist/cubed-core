from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._runtime import require_numpy
from .geometry import DEFAULT_WARP_SPEC, WarpSpec, order_quad, polygon_area, warp_quad
from .sampling import (
    DEFAULT_OCCLUSION_POLICY,
    OcclusionPolicy,
    sample_warped_face_bgr,
)
from .types import FaceCellRead, FaceDetection, GeometricReadResult


@dataclass(frozen=True, slots=True)
class GeometricReaderConfig:
    warp: WarpSpec = DEFAULT_WARP_SPEC
    window_half: int = 8
    occlusion: OcclusionPolicy = DEFAULT_OCCLUSION_POLICY
    centroids: tuple[tuple[float, float, float], ...] | None = None


def assign_face_slots(
    faces: Sequence[FaceDetection],
) -> tuple[tuple[str, FaceDetection], ...]:
    """Assign up/front/right from image position without identity inference."""

    if not faces:
        return ()
    items = [
        (
            sum(point[1] for point in face.corners) / 4.0,
            sum(point[0] for point in face.corners) / 4.0,
            face,
        )
        for face in faces
    ]
    items.sort(key=lambda item: (item[0], item[1]))
    result: list[tuple[str, FaceDetection]] = [("up", items[0][2])]
    remaining = sorted(items[1:], key=lambda item: (item[1], item[0]))
    if remaining:
        result.append(("front", remaining[0][2]))
    if len(remaining) > 1:
        result.append(("right", remaining[1][2]))
    return tuple(result)


class GeometricFaceReader:
    """Model-free face-quad warp and masked nine-cell Lab reader."""

    def __init__(self, config: GeometricReaderConfig | None = None):
        self.config = config or GeometricReaderConfig()

    def __call__(
        self,
        frame: Any,
        faces: Sequence[FaceDetection],
    ) -> GeometricReadResult:
        numpy = require_numpy()
        slotted = assign_face_slots(faces)
        if not slotted:
            return GeometricReadResult(faces=())
        areas = {slot: polygon_area(face.corners) for slot, face in slotted}
        maximum_area = max(areas.values())
        reads: list[FaceCellRead] = []
        centroids = (
            None
            if self.config.centroids is None
            else numpy.asarray(self.config.centroids, dtype=numpy.float32)
        )
        for slot, face in slotted:
            ordered = order_quad(face.corners)
            crop = warp_quad(
                frame,
                ordered,
                spec=self.config.warp,
                ordered=True,
            )
            sampled = sample_warped_face_bgr(
                crop,
                spec=self.config.warp,
                window_half=self.config.window_half,
                centroids=centroids,
                policy=self.config.occlusion,
            )
            reads.append(
                FaceCellRead(
                    slot=slot,
                    lab=tuple(
                        tuple(float(channel) for channel in vector) for vector in sampled.lab
                    ),
                    confidence=tuple(float(value) for value in sampled.confidence),
                    valid_pixels=sampled.valid_pixels,
                    total_pixels=sampled.total_pixels,
                    used_fallback=sampled.used_fallback,
                    relative_area=round(float(areas[slot] / maximum_area), 3),
                    corners=tuple(
                        tuple(float(coordinate) for coordinate in point) for point in ordered
                    ),
                )
            )
        return GeometricReadResult(faces=tuple(reads))
