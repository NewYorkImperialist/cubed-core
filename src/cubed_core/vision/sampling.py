from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._runtime import require_vision_runtime
from .geometry import DEFAULT_WARP_SPEC, GeometryError, WarpSpec, cell_centers


class SamplingError(ValueError):
    """Raised when a Lab image or occlusion policy is invalid."""


@dataclass(frozen=True, slots=True)
class OcclusionPolicy:
    """Configurable OpenCV-Lab rules for excluding non-sticker pixels."""

    low_chroma_threshold: float = 52.0
    bright_low_chroma_l: float = 185.0
    maximum_centroid_distance: float = 58.0
    minimum_valid_pixels: int = 8

    def __post_init__(self) -> None:
        for field_name in (
            "low_chroma_threshold",
            "bright_low_chroma_l",
            "maximum_centroid_distance",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise SamplingError(f"{field_name} must be a finite nonnegative number")
        if type(self.minimum_valid_pixels) is not int or self.minimum_valid_pixels < 1:
            raise SamplingError("minimum_valid_pixels must be a positive integer")


DEFAULT_OCCLUSION_POLICY = OcclusionPolicy()


@dataclass(frozen=True, slots=True)
class MaskedLabSamples:
    """Nine Lab vectors plus explicit pixel-support and fallback receipts."""

    lab: Any
    confidence: Any
    valid_pixels: tuple[int, ...]
    total_pixels: tuple[int, ...]
    used_fallback: tuple[bool, ...]


def _lab_image(value: Any) -> Any:
    _, numpy = require_vision_runtime()
    try:
        lab = numpy.asarray(value, dtype=numpy.float32)
    except (TypeError, ValueError) as exc:
        raise SamplingError("Lab image must be a numeric HxWx3 array") from exc
    if lab.ndim != 3 or lab.shape[2] != 3 or min(lab.shape[:2]) <= 0:
        raise SamplingError("Lab image must be a nonempty HxWx3 array")
    if not numpy.isfinite(lab).all():
        raise SamplingError("Lab image must contain only finite values")
    return lab


def _centroids(value: Any) -> Any:
    _, numpy = require_vision_runtime()
    try:
        centroids = numpy.asarray(value, dtype=numpy.float32)
    except (TypeError, ValueError) as exc:
        raise SamplingError("centroids must be a numeric Nx3 array") from exc
    if centroids.ndim != 2 or centroids.shape[1] != 3 or centroids.shape[0] < 1:
        raise SamplingError("centroids must have shape (N, 3) with at least one row")
    if not numpy.isfinite(centroids).all():
        raise SamplingError("centroids must contain only finite values")
    return centroids


def low_chroma_mask(
    lab_image: Any,
    *,
    policy: OcclusionPolicy = DEFAULT_OCCLUSION_POLICY,
) -> Any:
    """Return pixels that are low-chroma and not bright neutral surfaces."""

    _, numpy = require_vision_runtime()
    lab = _lab_image(lab_image)
    lightness = lab[..., 0]
    chroma = numpy.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return (chroma < policy.low_chroma_threshold) & (lightness < policy.bright_low_chroma_l)


def occlusion_mask(
    lab_image: Any,
    *,
    centroids: Any | None = None,
    policy: OcclusionPolicy = DEFAULT_OCCLUSION_POLICY,
) -> Any:
    """Return a boolean mask where ``True`` means unsuitable for a color read."""

    _, numpy = require_vision_runtime()
    lab = _lab_image(lab_image)
    mask = low_chroma_mask(lab, policy=policy)
    if centroids is None:
        return mask
    reference = _centroids(centroids)
    flat = lab.reshape(-1, 3)
    nearest = numpy.linalg.norm(flat[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    far = nearest.reshape(lab.shape[:2]) > policy.maximum_centroid_distance
    return mask | far


def bgr_to_lab(image: Any) -> Any:
    cv2, numpy = require_vision_runtime()
    array = numpy.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
        raise SamplingError("BGR image must be a nonempty HxWx3 array")
    if array.dtype != numpy.uint8:
        raise SamplingError("BGR image must use uint8 pixels")
    return cv2.cvtColor(array, cv2.COLOR_BGR2LAB).astype(numpy.float32)


def sample_masked_lab_cells(
    lab_image: Any,
    *,
    spec: WarpSpec = DEFAULT_WARP_SPEC,
    window_half: int = 8,
    centroids: Any | None = None,
    policy: OcclusionPolicy = DEFAULT_OCCLUSION_POLICY,
) -> MaskedLabSamples:
    """Sample row-major cell medians after excluding occluded pixels.

    A cell with too little unmasked support falls back to the median of its
    complete sampling window. The returned receipt marks every such fallback;
    consumers can reject or down-weight it instead of mistaking it for a clean
    sample.
    """

    _, numpy = require_vision_runtime()
    lab = _lab_image(lab_image)
    if type(window_half) is not int or not 1 <= window_half <= spec.cell_size // 2:
        raise SamplingError(f"window_half must be an integer between 1 and {spec.cell_size // 2}")
    mask = occlusion_mask(lab, centroids=centroids, policy=policy)
    centers = cell_centers(spec)
    samples = numpy.empty((len(centers), 3), dtype=numpy.float64)
    confidence = numpy.empty(len(centers), dtype=numpy.float64)
    valid_counts: list[int] = []
    total_counts: list[int] = []
    fallbacks: list[bool] = []
    height, width = lab.shape[:2]
    for index, (center_x, center_y) in enumerate(centers):
        x = int(round(float(center_x)))
        y = int(round(float(center_y)))
        x0, x1 = x - window_half, x + window_half
        y0, y1 = y - window_half, y + window_half
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise SamplingError(
                "canonical cell sampling window lies outside the supplied Lab image"
            )
        window = lab[y0:y1, x0:x1].reshape(-1, 3)
        masked = mask[y0:y1, x0:x1].reshape(-1)
        total = int(window.shape[0])
        valid = window[~masked]
        valid_count = int(valid.shape[0])
        valid_counts.append(valid_count)
        total_counts.append(total)
        confidence[index] = valid_count / total if total else 0.0
        use_fallback = valid_count < policy.minimum_valid_pixels
        fallbacks.append(use_fallback)
        source = window if use_fallback else valid
        if source.size:
            samples[index] = numpy.median(source, axis=0)
        else:  # The bounds check makes this unreachable for a valid spec.
            samples[index] = numpy.asarray((0.0, 128.0, 128.0), dtype=numpy.float32)
    return MaskedLabSamples(
        lab=samples,
        confidence=confidence,
        valid_pixels=tuple(valid_counts),
        total_pixels=tuple(total_counts),
        used_fallback=tuple(fallbacks),
    )


def sample_warped_face_bgr(
    crop: Any,
    *,
    spec: WarpSpec = DEFAULT_WARP_SPEC,
    window_half: int = 8,
    centroids: Any | None = None,
    policy: OcclusionPolicy = DEFAULT_OCCLUSION_POLICY,
) -> MaskedLabSamples:
    try:
        lab = bgr_to_lab(crop)
    except GeometryError as exc:
        raise SamplingError(str(exc)) from exc
    return sample_masked_lab_cells(
        lab,
        spec=spec,
        window_half=window_half,
        centroids=centroids,
        policy=policy,
    )
