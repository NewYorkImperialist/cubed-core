from __future__ import annotations

import hashlib
import math
from typing import Any

from . import grid_calibration
from .vision._runtime import VisionRuntimeUnavailable

COLOR_CALIBRATION_SCHEMA = "cubed-core/color-calibration-v1"
COLOR_CENTROIDS_SCHEMA = "cubed-core/color-centroids-v1"
CENTROIDS_COLOR_SPACE = "cielab"
COLOR_ORDER = ("white", "green", "red", "blue", "orange", "yellow")
COLOR_SPACE = "opencv_bgr_lab_uint8_v1"
SERVER_GRID_SAMPLER = "server_grid_opencv_lab_v1"
FRAMES_PER_COLOR = grid_calibration.FRAMES_PER_COLOR
SAMPLES_PER_FRAME = grid_calibration.SAMPLES_PER_FRAME
SAMPLES_PER_COLOR = grid_calibration.SAMPLES_PER_COLOR
DISTANCE_WEIGHTS = grid_calibration._W
MINIMUM_L = 40.0
UNIFORMITY_MAXIMUM = grid_calibration.UNIFORMITY_MAX
STABILITY_MAXIMUM = grid_calibration.STABILITY_MAX
COLLISION_MINIMUM = grid_calibration.COLLISION_MIN


class ColorCalibrationError(ValueError):
    pass


class ColorCentroidsError(ValueError):
    """Raised for a malformed cubed-core/color-centroids-v1 document or centroid map."""


def _require_object(
    value: Any,
    *,
    field: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ColorCalibrationError(f"{field} must be an object")
    actual = set(value)
    missing = sorted(keys - actual)
    unexpected = sorted(actual - keys)
    if missing:
        raise ColorCalibrationError(f"{field} is missing: {', '.join(missing)}")
    if unexpected:
        raise ColorCalibrationError(f"{field} has unsupported fields: {', '.join(unexpected)}")
    return value


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ColorCalibrationError(f"{field} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ColorCalibrationError(f"{field} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ColorCalibrationError(f"{field} must be at most {maximum:g}")
    return number


def _exact_number(value: Any, *, field: str, expected: float) -> None:
    number = _finite_number(value, field=field)
    if number != expected:
        raise ColorCalibrationError(f"{field} must equal {expected:g}")


def _exact_integer(value: Any, *, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ColorCalibrationError(f"{field} must equal {expected}")


def _lab_vector(value: Any, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ColorCalibrationError(f"{field} must be a three-number Lab vector")
    return tuple(
        _finite_number(component, field=f"{field}[{index}]", minimum=0, maximum=255)
        for index, component in enumerate(value)
    )


def _weighted_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    try:
        return grid_calibration.wdist(first, second)
    except VisionRuntimeUnavailable as exc:
        raise ColorCalibrationError(str(exc)) from exc


def _validate_fixed_contract(value: dict[str, Any]) -> None:
    if value["schema"] != COLOR_CALIBRATION_SCHEMA:
        raise ColorCalibrationError("unsupported color calibration schema")
    _exact_integer(value["schema_version"], field="schema_version", expected=1)
    if value["color_space"] != COLOR_SPACE:
        raise ColorCalibrationError(f"color_space must equal '{COLOR_SPACE}'")

    geometry = _require_object(
        value["geometry"],
        field="geometry",
        keys={
            "target_square_fraction",
            "grid_rows",
            "grid_columns",
            "central_patch_fraction",
            "coordinate_space",
        },
    )
    _exact_number(
        geometry["target_square_fraction"],
        field="geometry.target_square_fraction",
        expected=grid_calibration.GRID_FRACTION,
    )
    _exact_integer(
        geometry["grid_rows"],
        field="geometry.grid_rows",
        expected=grid_calibration.GRID_N,
    )
    _exact_integer(
        geometry["grid_columns"],
        field="geometry.grid_columns",
        expected=grid_calibration.GRID_N,
    )
    _exact_number(
        geometry["central_patch_fraction"],
        field="geometry.central_patch_fraction",
        expected=grid_calibration.PATCH_FRACTION,
    )
    if geometry["coordinate_space"] != "rotated_video_frame_pixels":
        raise ColorCalibrationError(
            "geometry.coordinate_space must equal 'rotated_video_frame_pixels'"
        )

    thresholds = _require_object(
        value["thresholds"],
        field="thresholds",
        keys={
            "minimum_l",
            "uniformity_maximum",
            "stability_maximum",
            "collision_minimum",
            "distance_weights",
        },
    )
    _exact_number(thresholds["minimum_l"], field="thresholds.minimum_l", expected=MINIMUM_L)
    _exact_number(
        thresholds["uniformity_maximum"],
        field="thresholds.uniformity_maximum",
        expected=UNIFORMITY_MAXIMUM,
    )
    _exact_number(
        thresholds["stability_maximum"],
        field="thresholds.stability_maximum",
        expected=STABILITY_MAXIMUM,
    )
    _exact_number(
        thresholds["collision_minimum"],
        field="thresholds.collision_minimum",
        expected=COLLISION_MINIMUM,
    )
    weights = thresholds["distance_weights"]
    if not isinstance(weights, list) or len(weights) != 3:
        raise ColorCalibrationError("thresholds.distance_weights must contain three numbers")
    for index, expected in enumerate(DISTANCE_WEIGHTS):
        _exact_number(
            weights[index],
            field=f"thresholds.distance_weights[{index}]",
            expected=expected,
        )

    if value["order"] != list(COLOR_ORDER):
        raise ColorCalibrationError("order must equal white, green, red, blue, orange, yellow")
    _exact_integer(
        value["frames_per_color"],
        field="frames_per_color",
        expected=FRAMES_PER_COLOR,
    )
    _exact_integer(
        value["samples_per_frame"],
        field="samples_per_frame",
        expected=SAMPLES_PER_FRAME,
    )


def _validate_provenance(value: Any) -> None:
    provenance = _require_object(
        value,
        field="provenance",
        keys={
            "captured_unix_ms",
            "collection_started_unix_ms",
            "camera_facing",
            "mirrored",
            "frame_width",
            "frame_height",
            "camera_device_type",
            "camera_format_width",
            "camera_format_height",
            "exposure_duration_seconds",
            "iso",
            "white_balance_gains",
            "lens_position",
            "exposure_locked",
            "white_balance_locked",
            "focus_locked",
            "app",
            "sampler",
        },
    )
    captured = _finite_number(
        provenance["captured_unix_ms"],
        field="provenance.captured_unix_ms",
        minimum=0,
    )
    started = _finite_number(
        provenance["collection_started_unix_ms"],
        field="provenance.collection_started_unix_ms",
        minimum=0,
    )
    if captured < started:
        raise ColorCalibrationError(
            "provenance.captured_unix_ms must not precede collection_started_unix_ms"
        )
    if provenance["camera_facing"] not in {"front", "back", "unknown"}:
        raise ColorCalibrationError("provenance.camera_facing is invalid")
    if not isinstance(provenance["mirrored"], bool):
        raise ColorCalibrationError("provenance.mirrored must be a boolean")
    for field in (
        "frame_width",
        "frame_height",
        "camera_format_width",
        "camera_format_height",
    ):
        dimension = provenance[field]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ColorCalibrationError(f"provenance.{field} must be a positive integer")
    if (
        not isinstance(provenance["camera_device_type"], str)
        or not provenance["camera_device_type"].strip()
        or len(provenance["camera_device_type"]) > 200
    ):
        raise ColorCalibrationError("provenance.camera_device_type must be a non-empty string")
    _finite_number(
        provenance["exposure_duration_seconds"],
        field="provenance.exposure_duration_seconds",
        minimum=1e-12,
    )
    _finite_number(provenance["iso"], field="provenance.iso", minimum=1e-12)
    gains = provenance["white_balance_gains"]
    if not isinstance(gains, list) or len(gains) != 3:
        raise ColorCalibrationError("provenance.white_balance_gains must contain three numbers")
    for index, gain in enumerate(gains):
        _finite_number(
            gain,
            field=f"provenance.white_balance_gains[{index}]",
            minimum=1e-12,
            maximum=32,
        )
    _finite_number(
        provenance["lens_position"],
        field="provenance.lens_position",
        minimum=0,
        maximum=1,
    )
    for field in ("exposure_locked", "white_balance_locked", "focus_locked"):
        if provenance[field] is not True:
            raise ColorCalibrationError(f"provenance.{field} must be true")
    if provenance["app"] != "cubed-capture-ios":
        raise ColorCalibrationError("provenance.app must equal 'cubed-capture-ios'")
    if provenance["sampler"] != SERVER_GRID_SAMPLER:
        raise ColorCalibrationError(f"provenance.sampler must equal '{SERVER_GRID_SAMPLER}'")


def validate_color_calibration(value: Any) -> None:
    """Validate a cubed-core/color-calibration-v1 document.

    This is the fuller, live-camera-capture contract: fixed geometry and
    thresholds plus measured per-color samples that the reported centroids
    must be derivable from (see ``validate_color_centroids`` for the bare,
    additive imported-centroids contract).
    """

    calibration = _require_object(
        value,
        field="color calibration",
        keys={
            "schema",
            "schema_version",
            "color_space",
            "geometry",
            "thresholds",
            "order",
            "frames_per_color",
            "samples_per_frame",
            "centroids",
            "samples",
            "provenance",
        },
    )
    _validate_fixed_contract(calibration)
    _validate_provenance(calibration["provenance"])

    centroid_values = _require_object(
        calibration["centroids"],
        field="centroids",
        keys=set(COLOR_ORDER),
    )
    sample_values = _require_object(
        calibration["samples"],
        field="samples",
        keys=set(COLOR_ORDER),
    )
    centroids: dict[str, tuple[float, float, float]] = {}
    for color in COLOR_ORDER:
        centroid = _lab_vector(centroid_values[color], field=f"centroids.{color}")
        raw_samples = sample_values[color]
        if not isinstance(raw_samples, list) or len(raw_samples) != SAMPLES_PER_COLOR:
            raise ColorCalibrationError(
                f"samples.{color} must contain exactly {SAMPLES_PER_COLOR} Lab vectors"
            )
        samples = [
            _lab_vector(sample, field=f"samples.{color}[{index}]")
            for index, sample in enumerate(raw_samples)
        ]
        if any(sample[0] < MINIMUM_L for sample in samples):
            raise ColorCalibrationError(f"samples.{color} contains L below {MINIMUM_L:g}")
        for frame_index in range(FRAMES_PER_COLOR):
            start = frame_index * SAMPLES_PER_FRAME
            frame = samples[start : start + SAMPLES_PER_FRAME]
            for first_index, first in enumerate(frame):
                for second in frame[first_index + 1 :]:
                    if _weighted_distance(first, second) > UNIFORMITY_MAXIMUM:
                        raise ColorCalibrationError(
                            f"samples.{color} frame {frame_index} exceeds uniformity threshold"
                        )
            if frame_index:
                previous = samples[start - SAMPLES_PER_FRAME : start]
                stability = (
                    sum(
                        _weighted_distance(current, prior)
                        for current, prior in zip(frame, previous, strict=True)
                    )
                    / SAMPLES_PER_FRAME
                )
                if stability > STABILITY_MAXIMUM:
                    raise ColorCalibrationError(
                        f"samples.{color} frame {frame_index} exceeds stability threshold"
                    )
        try:
            expected_centroid = grid_calibration._robust_centroid(samples)
        except VisionRuntimeUnavailable as exc:
            raise ColorCalibrationError(str(exc)) from exc
        if any(
            not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
            for actual, expected in zip(centroid, expected_centroid, strict=True)
        ):
            raise ColorCalibrationError(f"centroids.{color} does not match the exported samples")
        centroids[color] = centroid

    for first_index, first_color in enumerate(COLOR_ORDER):
        for second_color in COLOR_ORDER[first_index + 1 :]:
            if (
                _weighted_distance(centroids[first_color], centroids[second_color])
                < COLLISION_MINIMUM
            ):
                raise ColorCalibrationError(
                    f"centroids.{first_color} and centroids.{second_color} are not distinct"
                )


def _centroid_component(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ColorCentroidsError(f"{field} must be a finite number")
    number = float(value)
    if number < 0 or number > 255:
        raise ColorCentroidsError(f"{field} must be between 0 and 255")
    return number


def _centroid_vector(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ColorCentroidsError(f"{field} must be a three-number Lab vector")
    return [
        _centroid_component(component, field=f"{field}[{index}]")
        for index, component in enumerate(value)
    ]


def validate_color_centroid_map(value: Any, *, field: str = "centroids") -> dict[str, list[float]]:
    """Validate a flat ``{color: [L, a, b]}`` map covering exactly the six cube colors.

    This is the shape the released decode-support asset ships in, and the shape
    ``scripts/geo_read.py`` reads through ``calib_util.load_centroids``.
    """

    if not isinstance(value, dict) or set(value) != set(COLOR_ORDER):
        raise ColorCentroidsError(
            f"{field} must contain exactly the six colors: {', '.join(sorted(COLOR_ORDER))}"
        )
    return {
        color: _centroid_vector(value[color], field=f"{field}.{color}") for color in COLOR_ORDER
    }


def validate_color_centroids(value: Any) -> dict[str, list[float]]:
    """Validate a cubed-core/color-centroids-v1 document and return its centroids.

    This is the additive, imported-capture calibration format: bare Lab
    centroids with no measured samples, geometry, or thresholds. It never
    satisfies a check that specifically requires measured samples (see
    ``validate_color_calibration`` for that fuller contract).
    """

    if not isinstance(value, dict):
        raise ColorCentroidsError("color centroids document must be an object")
    actual = set(value)
    expected = {"schema", "schema_version", "color_space", "centroids", "provenance"}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ColorCentroidsError(f"color centroids document is missing: {', '.join(missing)}")
    if unexpected:
        raise ColorCentroidsError(
            f"color centroids document has unsupported fields: {', '.join(unexpected)}"
        )
    if value["schema"] != COLOR_CENTROIDS_SCHEMA:
        raise ColorCentroidsError("unsupported color centroids schema")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != 1:
        raise ColorCentroidsError("schema_version must equal 1")
    if value["color_space"] != CENTROIDS_COLOR_SPACE:
        raise ColorCentroidsError(f"color_space must equal '{CENTROIDS_COLOR_SPACE}'")
    provenance = value["provenance"]
    if not isinstance(provenance, str) or not provenance.strip():
        raise ColorCentroidsError("provenance must be a non-empty string")
    return validate_color_centroid_map(value["centroids"])


def build_imported_centroids_document(value: Any, *, source_bytes: bytes) -> dict[str, Any]:
    """Wrap a bare flat six-color Lab map into a color-centroids-v1 document.

    ``provenance`` records the SHA-256 of the originally uploaded bytes so the
    imported origin is traceable even though it never went through the paired
    live-camera grid sampler.
    """

    centroids = validate_color_centroid_map(value)
    document = {
        "schema": COLOR_CENTROIDS_SCHEMA,
        "schema_version": 1,
        "color_space": CENTROIDS_COLOR_SPACE,
        "centroids": centroids,
        "provenance": f"imported-centroids sha256:{hashlib.sha256(source_bytes).hexdigest()}",
    }
    validate_color_centroids(document)
    return document
