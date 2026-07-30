"""Manual color calibration from six browser-produced sticker crops.

The browser already has the imported video open, so it submits one small,
lossless crop per cube color instead of asking the server to decode the movie
again.  The server remains the calibration authority: it decodes each PNG,
runs the existing 3x3 OpenCV-Lab grid sampler and quality gates, checks
six-color separation, and emits the existing ``color-centroids-v1`` sidecar.

A solve video cannot truthfully claim the locked-camera provenance required by
the fuller iOS ``color-calibration-v1`` receipt.  This path therefore uses the
centroid contract that imported captures already accept and records the source
capture, video digest, and exact submitted-crop digest in its provenance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from . import grid_calibration
from .color_calibration import (
    CENTROIDS_COLOR_SPACE,
    COLOR_CENTROIDS_SCHEMA,
    COLOR_ORDER,
    MINIMUM_L,
    ColorCentroidsError,
    validate_color_centroids,
)
from .vision._runtime import VisionRuntimeUnavailable, require_vision_runtime

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_CROP_BYTES = 2 * 1024**2
MIN_CROP_EDGE_PIXELS = 24
MAX_CROP_EDGE_PIXELS = 512


class DesktopCalibrationError(ValueError):
    """Raised when a desktop calibration crop set is invalid."""


def _decode_crop(color: str, payload: bytes) -> Any:
    if not isinstance(payload, bytes):
        raise DesktopCalibrationError(f"{color} crop must contain PNG bytes")
    if not payload:
        raise DesktopCalibrationError(f"{color} crop is empty")
    if len(payload) > MAX_CROP_BYTES:
        raise DesktopCalibrationError(f"{color} crop exceeds the 2 MiB limit")
    if not payload.startswith(PNG_SIGNATURE):
        raise DesktopCalibrationError(f"{color} crop must be a lossless PNG")
    if len(payload) < 24 or payload[8:12] != (13).to_bytes(4, "big") or payload[12:16] != b"IHDR":
        raise DesktopCalibrationError(f"{color} crop has an invalid PNG header")
    declared_width = int.from_bytes(payload[16:20], "big")
    declared_height = int.from_bytes(payload[20:24], "big")
    if (
        declared_width < MIN_CROP_EDGE_PIXELS
        or declared_height < MIN_CROP_EDGE_PIXELS
        or declared_width > MAX_CROP_EDGE_PIXELS
        or declared_height > MAX_CROP_EDGE_PIXELS
    ):
        raise DesktopCalibrationError(
            f"{color} crop dimensions must be from {MIN_CROP_EDGE_PIXELS} "
            f"through {MAX_CROP_EDGE_PIXELS} pixels per side"
        )
    try:
        cv2, numpy = require_vision_runtime()
        encoded = numpy.frombuffer(payload, dtype=numpy.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except VisionRuntimeUnavailable as exc:
        raise DesktopCalibrationError(str(exc)) from exc
    if frame_bgr is None:
        raise DesktopCalibrationError(f"{color} crop is not a valid PNG")
    height, width = frame_bgr.shape[:2]
    if width != declared_width or height != declared_height:
        raise DesktopCalibrationError(f"{color} crop dimensions do not match its PNG header")
    return frame_bgr


def build_centroids_document_from_crops(
    crops: Mapping[str, bytes],
    *,
    capture_id: str,
    video_sha256: str,
) -> dict[str, Any]:
    """Validate six sticker crops and build ``color-centroids-v1``."""

    if set(crops) != set(COLOR_ORDER):
        raise DesktopCalibrationError(
            "crops must contain exactly white, green, red, blue, orange, and yellow"
        )

    centroids: dict[str, list[float]] = {}
    crop_digest = hashlib.sha256()
    for color in COLOR_ORDER:
        payload = crops[color]
        frame_bgr = _decode_crop(color, payload)
        try:
            medians = grid_calibration.sample_grid(frame_bgr)
            accepted, reason, _ = grid_calibration.evaluate_frame(
                medians,
                medians,
                MINIMUM_L,
            )
        except VisionRuntimeUnavailable as exc:
            raise DesktopCalibrationError(str(exc)) from exc
        if not accepted:
            if reason == "dark":
                detail = "is too dark"
            elif reason == "not_uniform":
                detail = "includes more than one color or too much glare"
            else:
                detail = "could not be sampled"
            raise DesktopCalibrationError(f"{color} crop {detail}; choose a clearer sticker area")
        try:
            centroid = grid_calibration._robust_centroid(medians)
        except VisionRuntimeUnavailable as exc:
            raise DesktopCalibrationError(str(exc)) from exc
        centroids[color] = [float(component) for component in centroid]
        crop_digest.update(color.encode("ascii"))
        crop_digest.update(b"\0")
        crop_digest.update(len(payload).to_bytes(8, "big"))
        crop_digest.update(payload)

    try:
        collision = grid_calibration.find_collision(centroids, COLOR_ORDER)
    except VisionRuntimeUnavailable as exc:
        raise DesktopCalibrationError(str(exc)) from exc
    if collision is not None:
        first, second = collision
        raise DesktopCalibrationError(
            f"{second} is too close to {first}; choose clearer sticker crops"
        )

    document = {
        "schema": COLOR_CENTROIDS_SCHEMA,
        "schema_version": 1,
        "color_space": CENTROIDS_COLOR_SPACE,
        "centroids": centroids,
        "provenance": (
            f"workspace-video-sticker-crops capture:{capture_id} "
            f"video-sha256:{video_sha256} "
            f"crops-sha256:{crop_digest.hexdigest()}"
        ),
    }
    try:
        validate_color_centroids(document)
    except ColorCentroidsError as exc:
        raise DesktopCalibrationError(str(exc)) from exc
    return document
