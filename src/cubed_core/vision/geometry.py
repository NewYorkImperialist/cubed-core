from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._runtime import require_vision_runtime


class GeometryError(ValueError):
    """Raised when a face geometry value is invalid or degenerate."""


@dataclass(frozen=True, slots=True)
class WarpSpec:
    """Canonical square warp geometry for a three-by-three face grid."""

    output_size: int = 96
    face_size: int = 72
    grid_size: int = 3
    supersample: int = 3

    def __post_init__(self) -> None:
        if type(self.output_size) is not int or not 12 <= self.output_size <= 4096:
            raise GeometryError("output_size must be an integer between 12 and 4096")
        if type(self.face_size) is not int or not 3 <= self.face_size < self.output_size:
            raise GeometryError("face_size must be smaller than output_size")
        if type(self.grid_size) is not int or not 1 <= self.grid_size <= 16:
            raise GeometryError("grid_size must be an integer between 1 and 16")
        if self.face_size % self.grid_size:
            raise GeometryError("face_size must be divisible by grid_size")
        if (self.output_size - self.face_size) % 2:
            raise GeometryError("output_size minus face_size must be even")
        if type(self.supersample) is not int or not 1 <= self.supersample <= 8:
            raise GeometryError("supersample must be an integer between 1 and 8")

    @property
    def margin(self) -> int:
        return (self.output_size - self.face_size) // 2

    @property
    def cell_size(self) -> int:
        return self.face_size // self.grid_size


DEFAULT_WARP_SPEC = WarpSpec()


def _quad_array(quad: Any, *, field: str = "quad") -> Any:
    _, numpy = require_vision_runtime()
    try:
        points = numpy.asarray(quad, dtype=numpy.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{field} must be a numeric four-by-two array") from exc
    if points.shape != (4, 2):
        raise GeometryError(f"{field} must have shape (4, 2)")
    if not numpy.isfinite(points).all():
        raise GeometryError(f"{field} must contain only finite coordinates")
    distances = points[:, None, :] - points[None, :, :]
    distance_squared = numpy.sum(distances * distances, axis=2)
    distance_squared += numpy.eye(4, dtype=numpy.float64)
    if float(distance_squared.min()) <= 1e-12:
        raise GeometryError(f"{field} must contain four distinct points")
    return points


def _signed_area(points: Any) -> float:
    _, numpy = require_vision_runtime()
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(y, numpy.roll(x, -1)))


def _crosses(points: Any) -> Any:
    _, numpy = require_vision_runtime()
    first = numpy.roll(points, -1, axis=0) - points
    second = numpy.roll(points, -2, axis=0) - numpy.roll(points, -1, axis=0)
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _validate_ordered(points: Any, *, field: str = "quad") -> Any:
    _, numpy = require_vision_runtime()
    points = _quad_array(points, field=field)
    crosses = _crosses(points)
    if float(numpy.min(numpy.abs(crosses))) <= 1e-8:
        raise GeometryError(f"{field} must be a nondegenerate convex quadrilateral")
    if not (numpy.all(crosses > 0.0) or numpy.all(crosses < 0.0)):
        raise GeometryError(f"{field} must be a convex quadrilateral")
    if abs(_signed_area(points)) <= 1e-8:
        raise GeometryError(f"{field} must have nonzero area")
    return points


def order_quad(quad: Any) -> Any:
    """Return the established centroid-angle face-corner order.

    The cyclic start is intentionally the one produced by ``argsort(atan2)``.
    Re-anchoring an otherwise valid quad can rotate a near-diamond crop and
    therefore change the row-major cell read.
    """

    _, numpy = require_vision_runtime()
    points = numpy.asarray(_quad_array(quad), dtype=numpy.float32)
    center = points.mean(axis=0)
    angles = numpy.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[numpy.argsort(angles)]
    _validate_ordered(ordered)
    return numpy.ascontiguousarray(ordered, dtype=numpy.float32)


def polygon_area(quad: Any) -> float:
    return abs(_signed_area(order_quad(quad)))


def shrink_quad(quad: Any, fraction: float) -> Any:
    """Move each ordered corner toward the centroid by ``fraction``."""

    _, numpy = require_vision_runtime()
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 <= float(fraction) < 1.0
    ):
        raise GeometryError("shrink fraction must be a finite number in [0, 1)")
    points = order_quad(quad)
    center = points.mean(axis=0)
    return numpy.ascontiguousarray(
        points + (center - points) * float(fraction),
        dtype=numpy.float32,
    )


def destination_quad(spec: WarpSpec = DEFAULT_WARP_SPEC) -> Any:
    _, numpy = require_vision_runtime()
    start = float(spec.margin)
    end = float(spec.margin + spec.face_size)
    return numpy.asarray(
        ((start, start), (end, start), (end, end), (start, end)),
        dtype=numpy.float32,
    )


def cell_centers(spec: WarpSpec = DEFAULT_WARP_SPEC) -> Any:
    """Return row-major crop-space centers for the canonical face grid."""

    _, numpy = require_vision_runtime()
    centers = [
        (
            spec.margin + (column + 0.5) * spec.cell_size,
            spec.margin + (row + 0.5) * spec.cell_size,
        )
        for row in range(spec.grid_size)
        for column in range(spec.grid_size)
    ]
    return numpy.asarray(centers, dtype=numpy.float32)


def cell_quads(spec: WarpSpec = DEFAULT_WARP_SPEC) -> Any:
    """Return row-major crop-space quadrilaterals for every grid cell."""

    _, numpy = require_vision_runtime()
    result = []
    for row in range(spec.grid_size):
        for column in range(spec.grid_size):
            x0 = spec.margin + column * spec.cell_size
            y0 = spec.margin + row * spec.cell_size
            x1 = x0 + spec.cell_size
            y1 = y0 + spec.cell_size
            result.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return numpy.asarray(result, dtype=numpy.float32)


def warp_quad(
    image: Any,
    quad: Any,
    *,
    spec: WarpSpec = DEFAULT_WARP_SPEC,
    ordered: bool = False,
) -> Any:
    """Warp a BGR image quad into the canonical square face crop."""

    cv2, numpy = require_vision_runtime()
    image_array = numpy.asarray(image)
    if (
        image_array.ndim != 3
        or image_array.shape[2] != 3
        or image_array.shape[0] <= 0
        or image_array.shape[1] <= 0
    ):
        raise GeometryError("image must be a nonempty HxWx3 array")
    if image_array.dtype != numpy.uint8:
        raise GeometryError("image must use uint8 BGR pixels")
    source = _validate_ordered(quad) if ordered else order_quad(quad)
    if _signed_area(source) < 0.0:
        raise GeometryError("an ordered quad must use clockwise image-space winding")
    scale = spec.supersample
    target = destination_quad(spec) * scale
    matrix = cv2.getPerspectiveTransform(
        numpy.asarray(source, dtype=numpy.float32),
        numpy.asarray(target, dtype=numpy.float32),
    )
    large_size = spec.output_size * scale
    warped = cv2.warpPerspective(
        image_array,
        matrix,
        (large_size, large_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if scale == 1:
        return warped
    return cv2.resize(
        warped,
        (spec.output_size, spec.output_size),
        interpolation=cv2.INTER_AREA,
    )


def project_crop_points_to_image(
    points: Any,
    quad: Any,
    *,
    spec: WarpSpec = DEFAULT_WARP_SPEC,
) -> Any:
    """Project canonical crop-space points back into source-image coordinates."""

    cv2, numpy = require_vision_runtime()
    source_points = numpy.asarray(points, dtype=numpy.float32)
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise GeometryError("points must have shape (N, 2)")
    if not numpy.isfinite(source_points).all():
        raise GeometryError("points must contain only finite coordinates")
    matrix = cv2.getPerspectiveTransform(destination_quad(spec), order_quad(quad))
    projected = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), matrix)
    return projected.reshape(-1, 2)


def cell_centers_image(quad: Any, *, spec: WarpSpec = DEFAULT_WARP_SPEC) -> Any:
    return project_crop_points_to_image(cell_centers(spec), quad, spec=spec)


def quad_iou(first: Any, second: Any) -> float:
    """Return convex-polygon intersection over union for two valid quads."""

    cv2, numpy = require_vision_runtime()
    first_ordered = order_quad(first)
    second_ordered = order_quad(second)
    first_hull = cv2.convexHull(numpy.asarray(first_ordered, dtype=numpy.float32))
    second_hull = cv2.convexHull(numpy.asarray(second_ordered, dtype=numpy.float32))
    first_area = float(cv2.contourArea(first_hull))
    second_area = float(cv2.contourArea(second_hull))
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    if intersection <= 0.0:
        return 0.0
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0
