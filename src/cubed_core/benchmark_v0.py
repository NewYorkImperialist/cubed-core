"""Strict, CPU-only contracts for the public Cubed Core Benchmark v0.

This module never runs a tracker or decoder. It validates an immutable public
bundle, validates separately held teacher truth, and scores already-produced
camera-only predictions. Teacher artifacts are opened only after prediction
documents have been bound to the four public input roles.

Benchmark v0 is deliberately narrow: 20--30 captures for architecture
development and deterministic scoring/comparison. A valid report is not
evidence of generalization beyond the manifest population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from cubed_core.color_calibration import (
    COLOR_CALIBRATION_SCHEMA,
    COLOR_CENTROIDS_SCHEMA,
    validate_color_calibration,
    validate_color_centroids,
)
from cubed_core.cube import FACE_ORDER, ORIENTATION_KEYS, Cube, parse_algorithm

MANIFEST_SCHEMA = "cubed-core-benchmark-v0-manifest.schema.json"
SCRAMBLE_SCHEMA = "cubed-core-benchmark-v0-scramble.schema.json"
CAMERA_METADATA_SCHEMA = "cubed-core-benchmark-v0-camera-metadata.schema.json"
RIGHTS_SCHEMA = "cubed-core-benchmark-v0-rights.schema.json"
TEACHER_SCHEMA = "cubed-core-benchmark-v0-teacher.schema.json"
PREDICTION_SCHEMA = "cubed-core-benchmark-v0-prediction.schema.json"
REPORT_SCHEMA = "cubed-core-benchmark-v0-report.schema.json"
FRAME_ANNOTATIONS_SCHEMA = "frame-annotations-v1.schema.json"
VIDEO_DERIVATION_SCHEMA = "capture-derivation-v1.schema.json"

INPUT_ROLES = ("video", "scramble", "calibration", "camera_metadata")
GROUP_KEYS = ("session_id", "solver_id", "cube_id", "camera_id", "setup_id")
SPLITS = ("train", "validation", "test")
MAX_BASELINE_VIDEO_BYTES = 100 * 1024 * 1024
MAX_HIGH_SPEED_VIDEO_BYTES = 200 * 1024 * 1024
_STATE_HASH_DOMAIN = b"cubed-core-state-v1\0"
_RIGHTS_PLACEHOLDER_TOKEN = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:todo|tbd|placeholder)(?:$|[^a-z0-9])"
)
_RIGHTS_EXAMPLE_IDENTIFIER = re.compile(r"(?i)(?:^|[^a-z0-9])example(?:$|[^a-z0-9])")
_SENSITIVE_DEVICE_TEXT_MARKERS = (
    "serial",
    "udid",
    "uuid",
    "imei",
    "bluetooth",
    "bt address",
    "mac address",
    "device name",
    "unique id",
    "uniqueid",
)
_SENSITIVE_DEVICE_TEXT_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SENSITIVE_DEVICE_TEXT_MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b")
_SENSITIVE_DEVICE_TEXT_LONG_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9:_-]{16,}\b)"
    r"(?=[A-Za-z0-9:_-]*[A-Za-z])"
    r"(?=[A-Za-z0-9:_-]*[0-9])"
    r"[A-Za-z0-9:_-]+\b"
)
_SENSITIVE_DEVICE_TEXT_LONG_NUMERIC = re.compile(r"\b[0-9]{14,20}\b")
_SENSITIVE_DEVICE_TEXT_PERSONAL_NAME = re.compile(
    r"(?:'s|’s)\s+(?:iphone|phone|camera|device|ipad)\b",
)

_FACE_NORMAL = {
    "up": (0, 1, 0),
    "right": (1, 0, 0),
    "front": (0, 0, 1),
    "down": (0, -1, 0),
    "left": (-1, 0, 0),
    "back": (0, 0, -1),
}
_FACE_RIGHT = {
    "up": (1, 0, 0),
    "right": (0, 0, -1),
    "front": (1, 0, 0),
    "down": (1, 0, 0),
    "left": (0, 0, 1),
    "back": (-1, 0, 0),
}
_FACE_DOWN = {
    "up": (0, 0, 1),
    "right": (0, -1, 0),
    "front": (0, -1, 0),
    "down": (0, 0, -1),
    "left": (0, -1, 0),
    "back": (0, -1, 0),
}
_FACE_FOR_NORMAL = {normal: face for face, normal in _FACE_NORMAL.items()}


class BenchmarkV0Error(ValueError):
    """Raised when a Benchmark v0 contract fails closed."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_json_bytes(raw: bytes, *, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkV0Error(f"{label}: invalid strict JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkV0Error(f"{label}: top-level JSON value must be an object")
    _validate_finite_json(value, label=label)
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BenchmarkV0Error(f"{label}: cannot read {path}: {exc}") from exc
    return _parse_json_bytes(raw, path=path, label=label)


def _read_json_with_sha256(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BenchmarkV0Error(f"{label}: cannot read {path}: {exc}") from exc
    return _parse_json_bytes(raw, path=path, label=label), _sha256_bytes(raw)


def _validate_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BenchmarkV0Error(f"{label}: non-finite JSON numbers are not allowed")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_json(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _validate_finite_json(child, label=label)


def _default_schema_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"


def _validate_schema(
    document: Mapping[str, Any],
    schema_name: str,
    *,
    schema_dir: Path,
    label: str,
) -> None:
    schema_path = schema_dir / schema_name
    schema = _read_json(schema_path, label=f"schema {schema_name}")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.absolute_path))
    if not errors:
        return
    first = errors[0]
    location = "$"
    for part in first.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    raise BenchmarkV0Error(f"{label}: schema violation at {location}: {first.message}")


def _read_and_validate_schema(
    path: Path,
    schema_name: str,
    *,
    schema_dir: Path,
    label: str,
) -> dict[str, Any]:
    document = _read_json(path, label=label)
    _validate_schema(document, schema_name, schema_dir=schema_dir, label=label)
    return document


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkV0Error(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_artifact_path(root: Path, relative: str, *, label: str) -> Path:
    if "\\" in relative:
        raise BenchmarkV0Error(f"{label}: artifact path must use POSIX separators")
    logical = PurePosixPath(relative)
    unsafe_part = any(part in {"", ".", ".."} for part in logical.parts)
    if logical.is_absolute() or not logical.parts or unsafe_part:
        raise BenchmarkV0Error(f"{label}: artifact path must be a normalized relative path")
    root_resolved = root.resolve()
    candidate = root.joinpath(*logical.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkV0Error(f"{label}: artifact does not exist: {relative}") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BenchmarkV0Error(f"{label}: artifact escapes its declared root: {relative}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise BenchmarkV0Error(f"{label}: artifact must be a regular non-symlink file: {relative}")
    return resolved


def _verify_artifact(
    root: Path,
    reference: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _safe_artifact_path(root, str(reference["path"]), label=label)
    size = path.stat().st_size
    if size != reference["bytes"]:
        raise BenchmarkV0Error(
            f"{label}: byte count mismatch for {reference['path']}: "
            f"manifest={reference['bytes']} actual={size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != reference["sha256"]:
        raise BenchmarkV0Error(
            f"{label}: SHA-256 mismatch for {reference['path']}: "
            f"manifest={reference['sha256']} actual={actual_sha256}"
        )
    return path


def _validate_media_types(capture: Mapping[str, Any]) -> None:
    capture_id = capture["capture_id"]
    video_type = capture["inputs"]["video"]["media_type"]
    if not video_type.startswith("video/"):
        raise BenchmarkV0Error(f"capture {capture_id}: video media_type must start with video/")
    high_speed = capture["inputs"].get("high_speed_original")
    if high_speed is not None and not high_speed["media_type"].startswith("video/"):
        raise BenchmarkV0Error(
            f"capture {capture_id}: high_speed_original media_type must start with video/"
        )
    derivation = capture["inputs"].get("video_derivation")
    if derivation is not None and derivation["media_type"] != "application/json":
        raise BenchmarkV0Error(
            f"capture {capture_id}: video_derivation media_type must be application/json"
        )
    for role in ("scramble", "calibration", "camera_metadata"):
        if capture["inputs"][role]["media_type"] != "application/json":
            raise BenchmarkV0Error(
                f"capture {capture_id}: {role} media_type must be application/json"
            )
    if capture["teacher_truth"]["media_type"] != "application/json":
        raise BenchmarkV0Error(
            f"capture {capture_id}: teacher_truth media_type must be application/json"
        )
    if capture["rights_record"]["media_type"] != "application/json":
        raise BenchmarkV0Error(
            f"capture {capture_id}: rights_record media_type must be application/json"
        )
    labels = capture.get("tracker_labels")
    if labels is not None and labels["media_type"] != "application/json":
        raise BenchmarkV0Error(
            f"capture {capture_id}: tracker_labels media_type must be application/json"
        )


def _validate_state_values(state: Any, *, label: str, canonical_centers: bool) -> list[int]:
    if not isinstance(state, list) or len(state) != 54:
        raise BenchmarkV0Error(f"{label}: cube state must contain exactly 54 integers")
    if any(type(value) is not int or not 0 <= value <= 5 for value in state):
        raise BenchmarkV0Error(f"{label}: cube state values must be integers in 0..5")
    counts = [state.count(index) for index in range(6)]
    if counts != [9] * 6:
        raise BenchmarkV0Error(f"{label}: cube state must contain nine of each color index")
    if canonical_centers:
        try:
            Cube.from_array(state)
        except (TypeError, ValueError) as exc:
            raise BenchmarkV0Error(f"{label}: trajectory state is not canonical: {exc}") from exc
    return state


def _dot(first: tuple[int, int, int], second: tuple[int, int, int]) -> int:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _add(*vectors: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sum(values) for values in zip(*vectors, strict=True))


def _scale(value: int, vector: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(value * component for component in vector)


def _rotate_vector(
    vector: tuple[int, int, int],
    orientation_key: tuple[str, str, str],
) -> tuple[int, int, int]:
    model_up, model_front, model_right = orientation_key
    return _add(
        _scale(_dot(vector, _FACE_NORMAL[model_up]), _FACE_NORMAL["up"]),
        _scale(_dot(vector, _FACE_NORMAL[model_front]), _FACE_NORMAL["front"]),
        _scale(_dot(vector, _FACE_NORMAL[model_right]), _FACE_NORMAL["right"]),
    )


def _rotation_permutation(
    orientation_key: tuple[str, str, str],
) -> tuple[int, ...]:
    if orientation_key not in ORIENTATION_KEYS:
        raise BenchmarkV0Error(f"unknown canonical orientation key: {orientation_key}")
    destination_for_source = [-1] * 54
    for face_index, face in enumerate(FACE_ORDER):
        normal = _FACE_NORMAL[face]
        right = _FACE_RIGHT[face]
        down = _FACE_DOWN[face]
        for row in range(3):
            for column in range(3):
                source = face_index * 9 + row * 3 + column
                position = _add(
                    normal,
                    _scale(column - 1, right),
                    _scale(row - 1, down),
                )
                rotated_normal = _rotate_vector(normal, orientation_key)
                rotated_position = _rotate_vector(position, orientation_key)
                destination_face = _FACE_FOR_NORMAL[rotated_normal]
                relative = _add(rotated_position, _scale(-1, rotated_normal))
                destination_column = _dot(relative, _FACE_RIGHT[destination_face]) + 1
                destination_row = _dot(relative, _FACE_DOWN[destination_face]) + 1
                if not 0 <= destination_row < 3 or not 0 <= destination_column < 3:
                    raise BenchmarkV0Error("internal whole-cube rotation produced an invalid grid")
                destination_face_index = FACE_ORDER.index(destination_face)
                destination = destination_face_index * 9 + destination_row * 3 + destination_column
                destination_for_source[source] = destination
    if sorted(destination_for_source) != list(range(54)):
        raise BenchmarkV0Error("internal whole-cube rotation is not a facelet bijection")
    return tuple(destination_for_source)


_ROTATION_PERMUTATIONS = {key: _rotation_permutation(key) for key in ORIENTATION_KEYS}


def rotate_state_to_orientation(
    state: Sequence[int],
    orientation_key: Sequence[str],
) -> list[int]:
    """Rotate one facelet state into a canonical whole-cube orientation.

    ``orientation_key`` has the decoder's stable meaning: the model faces
    occupying spatial ``up``, ``front``, and ``right``. Face grids use the
    public U/R/F/D/L/B row-major frames frozen in this module.
    """

    values = _validate_state_values(
        list(state),
        label="state",
        canonical_centers=False,
    )
    key = tuple(orientation_key)
    if len(key) != 3 or key not in _ROTATION_PERMUTATIONS:
        raise BenchmarkV0Error(f"unknown canonical orientation key: {key}")
    rotated = [-1] * 54
    for source, destination in enumerate(_ROTATION_PERMUTATIONS[key]):
        rotated[destination] = values[source]
    return rotated


def state_sha256(state: Sequence[int]) -> str:
    """Hash one 54-facelet state using the public Benchmark v0 identity.

    The digest input is ``b"cubed-core-state-v1\\0"`` followed by 54 unsigned
    bytes in canonical U/R/F/D/L/B row-major order. This function validates
    color range and counts but intentionally does not claim cubie reachability.
    """

    values = _validate_state_values(list(state), label="state", canonical_centers=False)
    return _sha256_bytes(_STATE_HASH_DOMAIN + bytes(values))


def _validate_hashed_state(
    item: Mapping[str, Any],
    *,
    label: str,
    canonical_centers: bool,
) -> list[int]:
    state = _validate_state_values(
        item["state"],
        label=label,
        canonical_centers=canonical_centers,
    )
    actual = state_sha256(state)
    if item["sha256"] != actual:
        raise BenchmarkV0Error(
            f"{label}: state identity mismatch: declared={item['sha256']} actual={actual}"
        )
    return state


def _cube_trajectory(scramble: Sequence[str], moves: Sequence[str]) -> list[list[int]]:
    try:
        parse_algorithm(scramble)
        parse_algorithm(moves)
        cube = Cube.solved().apply_algorithm(scramble)
    except (TypeError, ValueError) as exc:
        raise BenchmarkV0Error(f"invalid cube algorithm: {exc}") from exc
    states = [cube.to_array().astype(int).tolist()]
    for move in moves:
        cube.apply_move(move)
        states.append(cube.to_array().astype(int).tolist())
    return states


def _validate_scramble(
    path: Path,
    capture: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        SCRAMBLE_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} scramble",
    )
    if document["capture_id"] != capture_id:
        raise BenchmarkV0Error(f"capture {capture_id}: scramble capture_id does not match")
    try:
        parse_algorithm(document["moves"])
    except (TypeError, ValueError) as exc:
        raise BenchmarkV0Error(f"capture {capture_id}: invalid scramble: {exc}") from exc
    return document


def _validate_camera_metadata(
    path: Path,
    capture: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        CAMERA_METADATA_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} camera metadata",
    )
    expected = {
        "capture_id": capture_id,
        "session_group_id": capture["groups"]["session_id"],
        "camera_group_id": capture["groups"]["camera_id"],
        "setup_group_id": capture["groups"]["setup_id"],
    }
    for field, value in expected.items():
        if document[field] != value:
            raise BenchmarkV0Error(
                f"capture {capture_id}: camera metadata {field} does not match manifest"
            )
    calibration_binding = document["calibration_binding"]
    expected_calibration_binding = {
        "calibration_sha256": capture["inputs"]["calibration"]["sha256"],
        "setup_group_id": capture["groups"]["setup_id"],
    }
    for field, value in expected_calibration_binding.items():
        if calibration_binding[field] != value:
            raise BenchmarkV0Error(
                f"capture {capture_id}: calibration binding {field} does not match manifest"
            )
    device = document["capture_device"]
    public_text_fields = {
        "capture_device.manufacturer": device["manufacturer"],
        "capture_device.model": device["model"],
        "cube.manufacturer": document["cube"]["manufacturer"],
        "cube.model": document["cube"]["model"],
    }
    forbidden_placeholders = {"unknown", "n/a", "na", "tbd", "unspecified"}
    for field, value in public_text_fields.items():
        if not value.strip() or value.strip().lower() in forbidden_placeholders:
            raise BenchmarkV0Error(
                f"capture {capture_id}: public camera metadata {field} must be known"
            )
        _reject_sensitive_device_text(value, capture_id=capture_id, field=field)
    configured_fps = device["configured_fps"]
    mode = device["recording_mode"]
    if mode == "standard-120" and configured_fps != 120:
        raise BenchmarkV0Error(
            f"capture {capture_id}: standard-120 mode requires configured_fps 120"
        )
    if mode == "research-240" and configured_fps != 240:
        raise BenchmarkV0Error(
            f"capture {capture_id}: research-240 mode requires configured_fps 240"
        )
    measured = document["measured_fps"]
    measured_fps = measured["numerator"] / measured["denominator"]
    if not 110 <= measured_fps <= 121:
        raise BenchmarkV0Error(
            f"capture {capture_id}: benchmark video requires measured fps in 110..121"
        )
    source_measured = document["source_measured_fps"]
    high_speed = capture["inputs"].get("high_speed_original")
    derivation = capture["inputs"].get("video_derivation")
    if configured_fps == 120:
        if source_measured is not None or high_speed is not None or derivation is not None:
            raise BenchmarkV0Error(
                f"capture {capture_id}: 120 fps source cannot declare high-speed artifacts"
            )
    else:
        if source_measured is None or high_speed is None or derivation is None:
            raise BenchmarkV0Error(
                f"capture {capture_id}: 240 fps source requires source cadence, original, "
                "and derivation receipt"
            )
        source_fps = source_measured["numerator"] / source_measured["denominator"]
        if not 220 <= source_fps <= 242:
            raise BenchmarkV0Error(
                f"capture {capture_id}: high-speed source requires measured fps in 220..242"
            )
    if mode == "external-high-speed" and configured_fps != 240:
        raise BenchmarkV0Error(
            f"capture {capture_id}: external-high-speed mode requires configured_fps 240"
        )
    _validate_video_probe_receipt(document, capture)
    return document


def _reject_sensitive_device_text(value: str, *, capture_id: str, field: str) -> None:
    normalized = value.strip()
    lowered = normalized.lower()
    looks_like_path = (
        normalized.startswith(("/", "~/", "./", "../"))
        or re.match(r"^[A-Za-z]:\\", normalized) is not None
        or "file://" in lowered
        or "/users/" in lowered
        or "\\users\\" in lowered
    )
    personal_device_name = _SENSITIVE_DEVICE_TEXT_PERSONAL_NAME.search(lowered)
    if (
        "@" in normalized
        or looks_like_path
        or any(marker in lowered for marker in _SENSITIVE_DEVICE_TEXT_MARKERS)
        or _SENSITIVE_DEVICE_TEXT_UUID.search(normalized)
        or _SENSITIVE_DEVICE_TEXT_MAC.search(normalized)
        or _SENSITIVE_DEVICE_TEXT_LONG_IDENTIFIER.search(normalized)
        or _SENSITIVE_DEVICE_TEXT_LONG_NUMERIC.search(normalized)
        or personal_device_name
    ):
        raise BenchmarkV0Error(
            f"capture {capture_id}: public camera metadata {field} may contain "
            "a personal or unique device identifier"
        )


def _fps_fraction(value: Mapping[str, Any]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _validate_video_probe_receipt(
    camera_metadata: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> None:
    capture_id = capture["capture_id"]
    receipt = camera_metadata["video_probe_receipt"]
    canonical = receipt["canonical_video"]
    canonical_reference = capture["inputs"]["video"]
    expected_canonical = {
        "path": canonical_reference["path"],
        "bytes": canonical_reference["bytes"],
        "sha256": canonical_reference["sha256"],
        "measured_fps": camera_metadata["measured_fps"],
        "frame_count": camera_metadata["frame_count"],
        "width": camera_metadata["width"],
        "height": camera_metadata["height"],
        "rotation_degrees": camera_metadata["rotation_degrees"],
    }
    for field, expected in expected_canonical.items():
        if canonical[field] != expected:
            raise BenchmarkV0Error(
                f"capture {capture_id}: reviewed probe canonical_video.{field} "
                "does not match the benchmark input"
            )

    source = receipt["source_video"]
    high_speed_reference = capture["inputs"].get("high_speed_original")
    if high_speed_reference is None:
        if source is not None:
            raise BenchmarkV0Error(
                f"capture {capture_id}: reviewed probe has an undeclared source video"
            )
        return
    if source is None:
        raise BenchmarkV0Error(
            f"capture {capture_id}: high-speed original requires a reviewed source probe"
        )
    expected_source = {
        "path": high_speed_reference["path"],
        "bytes": high_speed_reference["bytes"],
        "sha256": high_speed_reference["sha256"],
        "measured_fps": camera_metadata["source_measured_fps"],
        "width": camera_metadata["width"],
        "height": camera_metadata["height"],
        "rotation_degrees": camera_metadata["rotation_degrees"],
    }
    for field, expected in expected_source.items():
        if source[field] != expected:
            raise BenchmarkV0Error(
                f"capture {capture_id}: reviewed probe source_video.{field} "
                "does not match the high-speed original"
            )


def _validate_calibration(path: Path, *, capture_id: str) -> dict[str, Any]:
    document = _read_json(path, label=f"capture {capture_id} calibration")
    try:
        if document.get("schema") == COLOR_CALIBRATION_SCHEMA:
            validate_color_calibration(document)
        elif document.get("schema") == COLOR_CENTROIDS_SCHEMA:
            validate_color_centroids(document)
        else:
            raise BenchmarkV0Error(
                f"capture {capture_id}: calibration must be "
                f"{COLOR_CALIBRATION_SCHEMA} or {COLOR_CENTROIDS_SCHEMA}"
            )
    except BenchmarkV0Error:
        raise
    except ValueError as exc:
        raise BenchmarkV0Error(f"capture {capture_id}: invalid calibration: {exc}") from exc
    return document


def _artifact_roles(capture: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = {role: capture["inputs"][role] for role in INPUT_ROLES}
    for role in ("high_speed_original", "video_derivation"):
        if capture["inputs"].get(role) is not None:
            artifacts[role] = capture["inputs"][role]
    artifacts["teacher_truth"] = capture["teacher_truth"]
    if capture.get("tracker_labels") is not None:
        artifacts["tracker_labels"] = capture["tracker_labels"]
    return artifacts


def _validate_video_derivation(
    path: Path,
    capture: Mapping[str, Any],
    camera_metadata: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        VIDEO_DERIVATION_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} video derivation",
    )
    source = capture["inputs"]["high_speed_original"]
    derivative = capture["inputs"]["video"]
    expected_source = {
        "path": source["path"],
        "bytes": source["bytes"],
        "sha256": source["sha256"],
    }
    expected_derivative = {
        "path": derivative["path"],
        "bytes": derivative["bytes"],
        "sha256": derivative["sha256"],
    }
    for field, value in expected_source.items():
        if document["source"][field] != value:
            raise BenchmarkV0Error(
                f"capture {capture_id}: derivation source.{field} does not match original"
            )
    for field, value in expected_derivative.items():
        if document["derivative"][field] != value:
            raise BenchmarkV0Error(
                f"capture {capture_id}: derivation derivative.{field} does not match video"
            )
    if document["derivative"]["recording_id"] != capture_id:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation derivative recording_id does not match"
        )

    transform = document["transform"]
    try:
        source_rate = Fraction(transform["source_frame_rate"])
        target_rate = Fraction(transform["target_frame_rate"])
        ffmpeg_rate = Fraction(document["ffmpeg"]["target_frame_rate"])
    except (ValueError, ZeroDivisionError) as exc:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation cadence is not a valid rational"
        ) from exc
    reviewed_source = camera_metadata["video_probe_receipt"]["source_video"]
    if reviewed_source is None:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation lacks a reviewed high-speed source probe"
        )
    reviewed_source_rate = _fps_fraction(reviewed_source["measured_fps"])
    reviewed_target_rate = _fps_fraction(
        camera_metadata["video_probe_receipt"]["canonical_video"]["measured_fps"]
    )
    if not 220 <= source_rate <= 242 or source_rate != reviewed_source_rate:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation source cadence must equal the reviewed "
            "220..242 fps source"
        )
    if not 110 <= target_rate <= 121 or target_rate != reviewed_target_rate:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation target cadence must equal the reviewed "
            "110..121 fps baseline"
        )
    if source_rate != 2 * target_rate or ffmpeg_rate != target_rate:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation must be an exact half-rate 240-to-120 transform"
        )
    expected_filter = (
        f"select=not(mod(n\\,2)),setpts=N*{target_rate.denominator}/({target_rate.numerator}*TB)"
    )
    if transform["filter"] != expected_filter:
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation filter is not the deterministic "
            "every-other-frame transform"
        )
    source_frame_count = reviewed_source["frame_count"]
    expected_frame_count = (source_frame_count + 1) // 2
    canonical_frame_count = camera_metadata["video_probe_receipt"]["canonical_video"]["frame_count"]
    if (
        transform["expected_output_frame_count"] != expected_frame_count
        or canonical_frame_count != expected_frame_count
    ):
        raise BenchmarkV0Error(
            f"capture {capture_id}: derivation frame count must be ceil(source/2)"
        )
    return document


def _validate_rights(
    path: Path,
    capture: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        RIGHTS_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} rights record",
    )
    if document["capture_id"] != capture_id:
        raise BenchmarkV0Error(f"capture {capture_id}: rights capture_id does not match")
    if (
        document["source_status"] != "approved-for-public-redistribution"
        or capture["source_status"] != "approved-for-public-redistribution"
        or document["public_release"] is not True
    ):
        raise BenchmarkV0Error(
            f"capture {capture_id}: private-research-only data cannot enter a public manifest"
        )
    if "draft" in document["terms_version"].casefold():
        raise BenchmarkV0Error(
            f"capture {capture_id}: rights terms_version must be final, not draft"
        )
    if int(document["acceptance_receipt_sha256"], 16) == 0:
        raise BenchmarkV0Error(
            f"capture {capture_id}: acceptance receipt SHA-256 cannot be all zeroes"
        )
    for field in ("record_id", "terms_version", "reviewer_id"):
        value = document[field]
        if (
            _RIGHTS_PLACEHOLDER_TOKEN.search(value) is not None
            or _RIGHTS_EXAMPLE_IDENTIFIER.search(value) is not None
        ):
            raise BenchmarkV0Error(
                f"capture {capture_id}: rights {field} contains a placeholder token"
            )
    for field in ("terms_sha256", "acceptance_receipt_sha256"):
        digest = document[field]
        if any(
            64 % unit_length == 0 and digest == digest[:unit_length] * (64 // unit_length)
            for unit_length in range(1, 9)
        ):
            raise BenchmarkV0Error(
                f"capture {capture_id}: rights {field} is a repeated placeholder digest"
            )

    declared_entries: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(document["artifacts"]):
        if (
            _RIGHTS_PLACEHOLDER_TOKEN.search(entry["public_license"]) is not None
            or _RIGHTS_EXAMPLE_IDENTIFIER.search(entry["public_license"]) is not None
        ):
            raise BenchmarkV0Error(
                f"capture {capture_id}: rights artifacts[{index}].public_license "
                "contains a placeholder token"
            )
        if _RIGHTS_PLACEHOLDER_TOKEN.search(entry["attribution"]) is not None:
            raise BenchmarkV0Error(
                f"capture {capture_id}: rights artifacts[{index}].attribution "
                "contains placeholder material"
            )
        role = entry["role"]
        if role in declared_entries:
            raise BenchmarkV0Error(
                f"capture {capture_id}: duplicate rights entry for artifact role {role}"
            )
        declared_entries[role] = entry

    artifacts = _artifact_roles(capture)
    if set(declared_entries) != set(artifacts):
        raise BenchmarkV0Error(
            f"capture {capture_id}: rights entries must cover exactly "
            f"{sorted(artifacts)}, got {sorted(declared_entries)}"
        )
    for role, reference in artifacts.items():
        entry = declared_entries[role]
        expected = {
            "path": reference["path"],
            "sha256": reference["sha256"],
            "public_license": reference["public_license"],
            "attribution": reference["attribution"],
            "public_release": True,
        }
        for field, value in expected.items():
            if entry[field] != value:
                raise BenchmarkV0Error(
                    f"capture {capture_id}: rights {role}.{field} does not match manifest"
                )
    return document


def _validate_tracker_labels(
    path: Path,
    capture: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        FRAME_ANNOTATIONS_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} tracker labels",
    )
    source = document["source"]
    if source["capture_id"] != capture_id:
        raise BenchmarkV0Error(f"capture {capture_id}: tracker label capture_id does not match")
    if source["sha256"] != capture["inputs"]["video"]["sha256"]:
        raise BenchmarkV0Error(
            f"capture {capture_id}: tracker labels are not bound to the benchmark video"
        )
    return document


def _validate_teacher(
    path: Path,
    capture: Mapping[str, Any],
    scramble: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> dict[str, Any]:
    capture_id = capture["capture_id"]
    document = _read_and_validate_schema(
        path,
        TEACHER_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} teacher truth",
    )
    if document["capture_id"] != capture_id:
        raise BenchmarkV0Error(f"capture {capture_id}: teacher capture_id does not match")
    if document["scramble_sha256"] != capture["inputs"]["scramble"]["sha256"]:
        raise BenchmarkV0Error(f"capture {capture_id}: teacher truth is bound to another scramble")
    if document["teacher_source"]["kind"] != "ble-smart-cube":
        raise BenchmarkV0Error(
            f"capture {capture_id}: public Benchmark v0 requires reviewed BLE smart-cube truth"
        )

    teacher_moves = [item["move"] for item in document["teacher_moves"]]
    try:
        parse_algorithm(teacher_moves)
    except (TypeError, ValueError) as exc:
        raise BenchmarkV0Error(f"capture {capture_id}: invalid teacher moves: {exc}") from exc
    timestamps = [item["timestamp_ms"] for item in document["teacher_moves"]]
    numeric_timestamps = [item for item in timestamps if item is not None]
    if numeric_timestamps and len(numeric_timestamps) != len(timestamps):
        raise BenchmarkV0Error(
            f"capture {capture_id}: teacher timestamps must be all present or all null"
        )
    if any(
        second < first
        for first, second in zip(numeric_timestamps, numeric_timestamps[1:], strict=False)
    ):
        raise BenchmarkV0Error(f"capture {capture_id}: teacher timestamps are not monotonic")
    if document["teacher_source"]["timing_status"] == "content-aligned" and not numeric_timestamps:
        raise BenchmarkV0Error(
            f"capture {capture_id}: content-aligned teacher truth requires timestamps"
        )
    if document["teacher_source"]["timing_status"] == "move-order-only" and numeric_timestamps:
        raise BenchmarkV0Error(
            f"capture {capture_id}: move-order-only truth must use null timestamps"
        )

    expected_states = _cube_trajectory(scramble["moves"], teacher_moves)
    trajectory = document["trajectory"]
    if len(trajectory) != len(expected_states):
        raise BenchmarkV0Error(
            f"capture {capture_id}: teacher trajectory length must equal move count plus one"
        )
    for step, (item, expected) in enumerate(zip(trajectory, expected_states, strict=True)):
        if item["step"] != step:
            raise BenchmarkV0Error(
                f"capture {capture_id}: teacher trajectory step {step} is misnumbered"
            )
        state = _validate_hashed_state(
            item,
            label=f"capture {capture_id} teacher trajectory step {step}",
            canonical_centers=True,
        )
        if state != expected:
            raise BenchmarkV0Error(
                f"capture {capture_id}: teacher trajectory does not replay at step {step}"
            )
    if not Cube.from_array(expected_states[-1]).is_solved():
        raise BenchmarkV0Error(
            f"capture {capture_id}: teacher trajectory must finish at the solved endpoint"
        )

    phase_names: set[str] = set()
    previous_index = -1
    for phase in document["phase_boundaries"]:
        name = phase["name"]
        index = phase["teacher_move_index"]
        if name in phase_names:
            raise BenchmarkV0Error(f"capture {capture_id}: duplicate phase name {name}")
        if index <= previous_index:
            raise BenchmarkV0Error(
                f"capture {capture_id}: phase boundaries must have increasing move indices"
            )
        if index >= len(trajectory):
            raise BenchmarkV0Error(
                f"capture {capture_id}: phase {name} exceeds the teacher trajectory"
            )
        if phase["canonical_state_sha256"] != trajectory[index]["sha256"]:
            raise BenchmarkV0Error(
                f"capture {capture_id}: phase {name} does not bind its teacher state"
            )
        phase_names.add(name)
        previous_index = index

    boundary = document["ll_boundary"]
    ll_index = boundary["teacher_move_index"]
    if ll_index >= len(trajectory):
        raise BenchmarkV0Error(f"capture {capture_id}: LL boundary exceeds teacher trajectory")
    if boundary["phase_name"] not in phase_names:
        raise BenchmarkV0Error(f"capture {capture_id}: LL boundary phase is undeclared")
    matching_phase = next(
        phase for phase in document["phase_boundaries"] if phase["name"] == boundary["phase_name"]
    )
    if matching_phase["teacher_move_index"] != ll_index:
        raise BenchmarkV0Error(
            f"capture {capture_id}: LL phase and LL boundary use different move indices"
        )

    expected_orientation_keys = {tuple(key) for key in ORIENTATION_KEYS}
    actual_orientation_keys: set[tuple[str, str, str]] = set()
    canonical_gl_state = trajectory[ll_index]["state"]
    for item in boundary["accepted_target_states"]:
        key = tuple(item["orientation_key"])
        if key in actual_orientation_keys:
            raise BenchmarkV0Error(f"capture {capture_id}: duplicate LL orientation key {key}")
        actual_orientation_keys.add(key)
        state = _validate_hashed_state(
            item,
            label=f"capture {capture_id} LL target {key}",
            canonical_centers=False,
        )
        expected_state = rotate_state_to_orientation(canonical_gl_state, key)
        if state != expected_state:
            raise BenchmarkV0Error(
                f"capture {capture_id}: LL target {key} is not the exact physical "
                "whole-cube rotation of teacher G_L"
            )
    if actual_orientation_keys != expected_orientation_keys:
        raise BenchmarkV0Error(
            f"capture {capture_id}: LL target set must enumerate the canonical 24 orientations"
        )
    return document


def _group_overlap(
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for key in GROUP_KEYS:
        value_splits: dict[str, set[str]] = defaultdict(set)
        value_captures: dict[str, list[str]] = defaultdict(list)
        for capture in manifest["captures"]:
            value = capture["groups"][key]
            value_splits[value].add(capture["split"])
            value_captures[value].append(capture["capture_id"])
        overlaps = [
            {
                "value": value,
                "splits": sorted(value_splits[value], key=SPLITS.index),
                "capture_count": len(value_captures[value]),
            }
            for value in sorted(value_splits)
            if len(value_splits[value]) > 1
        ]
        policy = manifest["split_policy"]["group_policy"][key]
        if policy["mode"] == "disjoint" and overlaps:
            values = ", ".join(item["value"] for item in overlaps[:5])
            raise BenchmarkV0Error(
                f"split leakage: {key} is declared disjoint but crosses splits: {values}"
            )
        result[key] = overlaps
    return result


def _validate_manifest_cross_fields(manifest: Mapping[str, Any]) -> None:
    captures = manifest["captures"]
    if manifest["capture_count"] != len(captures):
        raise BenchmarkV0Error("manifest capture_count does not equal captures length")
    capture_ids = [capture["capture_id"] for capture in captures]
    if len(capture_ids) != len(set(capture_ids)):
        raise BenchmarkV0Error("manifest capture_id values must be unique")
    capture_by_id = {capture["capture_id"]: capture for capture in captures}
    exception_keys: set[tuple[str, str]] = set()
    for exception in manifest["collection_license"]["exceptions"]:
        key = (exception["capture_id"], exception["role"])
        if key in exception_keys:
            raise BenchmarkV0Error(f"duplicate collection-license exception for {key[0]} {key[1]}")
        exception_keys.add(key)
        capture = capture_by_id.get(key[0])
        if capture is None:
            raise BenchmarkV0Error(
                f"collection-license exception references unknown capture {key[0]}"
            )
        role = key[1]
        if role == "teacher_truth":
            reference = capture["teacher_truth"]
        elif role == "tracker_labels":
            reference = capture.get("tracker_labels")
        elif role == "rights_record":
            reference = capture["rights_record"]
        else:
            reference = capture["inputs"].get(role)
        if reference is None:
            raise BenchmarkV0Error(
                f"collection-license exception references missing artifact {key[0]} {role}"
            )
        if exception["public_license"] != reference["public_license"]:
            raise BenchmarkV0Error(
                f"collection-license exception for {key[0]} {role} does not match "
                "the artifact public_license"
            )
    for split in SPLITS:
        if not any(capture["split"] == split for capture in captures):
            raise BenchmarkV0Error(f"manifest split {split} must contain at least one capture")

    video_paths: set[str] = set()
    video_hashes: set[str] = set()
    for capture in captures:
        _validate_media_types(capture)
        video = capture["inputs"]["video"]
        if video["bytes"] > MAX_BASELINE_VIDEO_BYTES:
            raise BenchmarkV0Error(
                f"capture {capture['capture_id']}: 120 fps benchmark video exceeds 100 MiB"
            )
        high_speed = capture["inputs"].get("high_speed_original")
        if high_speed is not None and high_speed["bytes"] > MAX_HIGH_SPEED_VIDEO_BYTES:
            raise BenchmarkV0Error(
                f"capture {capture['capture_id']}: high-speed original exceeds 200 MiB"
            )
        if video["path"] in video_paths:
            raise BenchmarkV0Error("each capture must use a distinct video path")
        if video["sha256"] in video_hashes:
            raise BenchmarkV0Error("each capture must use distinct video bytes")
        video_paths.add(video["path"])
        video_hashes.add(video["sha256"])


def _validate_group_metadata_consistency(captures: Mapping[str, Mapping[str, Any]]) -> None:
    camera_traits: dict[str, Mapping[str, Any]] = {}
    cube_traits: dict[str, Mapping[str, Any]] = {}
    for capture_id, capture_data in captures.items():
        manifest_capture = capture_data["manifest"]
        metadata = capture_data["camera_metadata"]
        camera_id = manifest_capture["groups"]["camera_id"]
        cube_id = manifest_capture["groups"]["cube_id"]
        device = metadata["capture_device"]
        cube = metadata["cube"]
        if camera_id in camera_traits and camera_traits[camera_id] != device:
            raise BenchmarkV0Error(
                f"capture {capture_id}: camera metadata drifts within camera_id {camera_id}"
            )
        if cube_id in cube_traits and cube_traits[cube_id] != cube:
            raise BenchmarkV0Error(
                f"capture {capture_id}: cube metadata drifts within cube_id {cube_id}"
            )
        camera_traits[camera_id] = device
        cube_traits[cube_id] = cube


def validate_bundle(
    manifest_path: str | os.PathLike[str],
    *,
    bundle_root: str | os.PathLike[str] | None = None,
    teacher_root: str | os.PathLike[str],
    schema_dir: str | os.PathLike[str] | None = None,
    _expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one complete public Benchmark v0 bundle.

    ``bundle_root`` contains public inputs and rights records. ``teacher_root``
    contains truth and must be a distinct directory. Private-research-only
    manifests are intentionally unsupported by this public validator.
    """

    manifest_file = Path(manifest_path).resolve()
    public_root = (
        Path(bundle_root).resolve() if bundle_root is not None else manifest_file.parent.resolve()
    )
    truth_root = Path(teacher_root).resolve()
    schema_root = Path(schema_dir).resolve() if schema_dir is not None else _default_schema_dir()
    roots_overlap = False
    try:
        truth_root.relative_to(public_root)
        roots_overlap = True
    except ValueError:
        pass
    try:
        public_root.relative_to(truth_root)
        roots_overlap = True
    except ValueError:
        pass
    if roots_overlap:
        raise BenchmarkV0Error(
            "teacher_root and bundle_root must be disjoint; neither may contain the other"
        )

    manifest, manifest_sha256 = _read_json_with_sha256(
        manifest_file,
        label="benchmark manifest",
    )
    if _expected_manifest_sha256 is not None and manifest_sha256 != _expected_manifest_sha256:
        raise BenchmarkV0Error(
            "benchmark manifest changed after predictions were bound; refusing evaluation"
        )
    _validate_schema(
        manifest,
        MANIFEST_SCHEMA,
        schema_dir=schema_root,
        label="benchmark manifest",
    )
    _validate_manifest_cross_fields(manifest)
    overlap = _group_overlap(manifest)

    captures: dict[str, dict[str, Any]] = {}
    for capture in manifest["captures"]:
        capture_id = capture["capture_id"]
        verified_paths: dict[str, Path] = {}
        for role in INPUT_ROLES:
            verified_paths[role] = _verify_artifact(
                public_root,
                capture["inputs"][role],
                label=f"capture {capture_id} {role}",
            )
        for role in ("high_speed_original", "video_derivation"):
            if capture["inputs"].get(role) is not None:
                verified_paths[role] = _verify_artifact(
                    public_root,
                    capture["inputs"][role],
                    label=f"capture {capture_id} {role}",
                )
        rights_path = _verify_artifact(
            public_root,
            capture["rights_record"],
            label=f"capture {capture_id} rights record",
        )
        teacher_path = _verify_artifact(
            truth_root,
            capture["teacher_truth"],
            label=f"capture {capture_id} teacher truth",
        )
        labels_path: Path | None = None
        if capture.get("tracker_labels") is not None:
            labels_path = _verify_artifact(
                public_root,
                capture["tracker_labels"],
                label=f"capture {capture_id} tracker labels",
            )

        scramble = _validate_scramble(
            verified_paths["scramble"],
            capture,
            schema_dir=schema_root,
        )
        camera_metadata = _validate_camera_metadata(
            verified_paths["camera_metadata"],
            capture,
            schema_dir=schema_root,
        )
        video_derivation = (
            _validate_video_derivation(
                verified_paths["video_derivation"],
                capture,
                camera_metadata,
                schema_dir=schema_root,
            )
            if "video_derivation" in verified_paths
            else None
        )
        calibration = _validate_calibration(
            verified_paths["calibration"],
            capture_id=capture_id,
        )
        rights = _validate_rights(rights_path, capture, schema_dir=schema_root)
        teacher = _validate_teacher(
            teacher_path,
            capture,
            scramble,
            schema_dir=schema_root,
        )
        tracker_labels = (
            _validate_tracker_labels(labels_path, capture, schema_dir=schema_root)
            if labels_path is not None
            else None
        )
        captures[capture_id] = {
            "manifest": capture,
            "scramble": scramble,
            "camera_metadata": camera_metadata,
            "video_derivation": video_derivation,
            "calibration": calibration,
            "rights": rights,
            "teacher": teacher,
            "tracker_labels": tracker_labels,
        }

    _validate_group_metadata_consistency(captures)

    split_counts = {
        split: sum(capture["split"] == split for capture in manifest["captures"])
        for split in SPLITS
    }
    return {
        "schema": "cubed-core/benchmark-v0-validation",
        "schema_version": 1,
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "manifest_sha256": manifest_sha256,
        "release_scope": manifest["release_scope"],
        "evidence_scope": manifest["evidence_scope"],
        "capture_count": len(captures),
        "split_counts": split_counts,
        "group_overlap": overlap,
        "public_rights_checked": True,
        "teacher_truth_checked": True,
        "manifest": manifest,
        "captures": captures,
    }


def _validate_prediction(
    document: Mapping[str, Any],
    capture: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    schema_dir: Path,
) -> None:
    capture_id = capture["capture_id"]
    _validate_schema(
        document,
        PREDICTION_SCHEMA,
        schema_dir=schema_dir,
        label=f"capture {capture_id} prediction",
    )
    if document["benchmark_id"] != manifest["benchmark_id"]:
        raise BenchmarkV0Error(f"capture {capture_id}: prediction benchmark_id does not match")
    if document["capture_id"] != capture_id:
        raise BenchmarkV0Error(f"capture {capture_id}: prediction capture_id does not match")

    bindings: dict[str, str] = {}
    for binding in document["inference"]["input_artifacts"]:
        role = binding["role"]
        if role in bindings:
            raise BenchmarkV0Error(
                f"capture {capture_id}: duplicate prediction input binding for {role}"
            )
        bindings[role] = binding["sha256"]
    expected = {role: capture["inputs"][role]["sha256"] for role in INPUT_ROLES}
    if bindings != expected:
        raise BenchmarkV0Error(
            f"capture {capture_id}: camera-only input bindings must equal the four public inputs"
        )
    if document["inference"]["teacher_access"] != "none-declared":
        raise BenchmarkV0Error(f"capture {capture_id}: prediction declares teacher access")
    if document["inference"]["teacher_artifacts_visible"] is not False:
        raise BenchmarkV0Error(f"capture {capture_id}: prediction had teacher artifacts visible")

    try:
        parse_algorithm(document["result"]["moves"])
    except (TypeError, ValueError) as exc:
        raise BenchmarkV0Error(f"capture {capture_id}: invalid predicted moves: {exc}") from exc
    component_names: set[str] = set()
    for metric in document["component_metrics"]:
        if metric["name"] in component_names:
            raise BenchmarkV0Error(
                f"capture {capture_id}: duplicate component metric {metric['name']}"
            )
        if not math.isfinite(metric["value"]):
            raise BenchmarkV0Error(
                f"capture {capture_id}: component metric {metric['name']} is not finite"
            )
        component_names.add(metric["name"])
    model_names: set[str] = set()
    for artifact in document["system"]["model_artifacts"]:
        if artifact["name"] in model_names:
            raise BenchmarkV0Error(
                f"capture {capture_id}: duplicate system model artifact {artifact['name']}"
            )
        model_names.add(artifact["name"])


def _levenshtein(first: Sequence[str], second: Sequence[str]) -> int:
    if len(first) > len(second):
        first, second = second, first
    previous = list(range(len(first) + 1))
    for second_index, second_value in enumerate(second, start=1):
        current = [second_index]
        for first_index, first_value in enumerate(first, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[first_index] + 1,
                    previous[first_index - 1] + (first_value != second_value),
                )
            )
        previous = current
    return previous[-1]


def _rate(successes: int, total: int) -> float:
    return successes / total


def _mean(values: Sequence[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _component_aggregate(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        for metric in prediction["component_metrics"]:
            by_name[metric["name"]].append(metric)

    metrics = []
    for name in sorted(by_name):
        items = by_name[name]
        units = {item["unit"] for item in items}
        if len(units) != 1:
            raise BenchmarkV0Error(f"component metric {name} uses inconsistent units")
        total_samples = sum(item["sample_count"] for item in items)
        weighted_sum = sum(item["value"] * item["sample_count"] for item in items)
        metrics.append(
            {
                "name": name,
                "unit": next(iter(units)),
                "capture_count": len(items),
                "sample_count": total_samples,
                "weighted_mean": weighted_sum / total_samples,
                "scope": "component-only-not-end-to-end",
            }
        )
    return {
        "scope": (
            "component diagnostics describe tracker/read subproblems and do not establish "
            "camera-to-state reconstruction"
        ),
        "metrics": metrics,
    }


def _evaluate_capture(
    prediction: Mapping[str, Any],
    capture_data: Mapping[str, Any],
) -> dict[str, Any]:
    capture = capture_data["manifest"]
    teacher = capture_data["teacher"]
    moves = prediction["result"]["moves"]
    teacher_moves = [item["move"] for item in teacher["teacher_moves"]]
    predicted_states = _cube_trajectory(capture_data["scramble"]["moves"], moves)
    predicted_hashes_by_step = [
        {state_sha256(rotate_state_to_orientation(state, key)) for key in ORIENTATION_KEYS}
        for state in predicted_states
    ]
    predicted_hashes = [state_sha256(state) for state in predicted_states]
    accepted_ll_hashes = {
        item["sha256"] for item in teacher["ll_boundary"]["accepted_target_states"]
    }
    hit_step = next(
        (
            step
            for step, digests in enumerate(predicted_hashes_by_step)
            if digests.intersection(accepted_ll_hashes)
        ),
        None,
    )

    phases = teacher["phase_boundaries"]
    phase_hits = [
        phase["name"] for phase in phases if phase["canonical_state_sha256"] in predicted_hashes
    ]
    first_unreached = next(
        (
            phase["name"]
            for phase in phases
            if phase["canonical_state_sha256"] not in predicted_hashes
        ),
        None,
    )
    final_cube = Cube.from_array(predicted_states[-1])
    return {
        "capture_id": capture["capture_id"],
        "status": prediction["result"]["status"],
        "reach_ll": hit_step is not None,
        "reach_ll_hit_step": hit_step,
        "solved_endpoint_completed": final_cube.is_solved(),
        "edit_distance": _levenshtein(moves, teacher_moves),
        "predicted_length": len(moves),
        "teacher_length": len(teacher_moves),
        "deepest_phase_reached": phase_hits[-1] if phase_hits else None,
        "first_unreached_phase": first_unreached,
        "runtime_ms": prediction["result"]["runtime_ms"],
        "abstention_reason": prediction["result"]["abstention_reason"],
    }


def evaluate_bundle(
    manifest_path: str | os.PathLike[str],
    prediction_dir: str | os.PathLike[str],
    *,
    bundle_root: str | os.PathLike[str] | None = None,
    teacher_root: str | os.PathLike[str],
    split: str = "test",
    schema_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one split of camera-only predictions against held truth.

    The primary aggregate is ``reach_ll_rate`` over the raw predicted move
    trajectories before any move simplification. Solved endpoint is secondary;
    string edit distance and phase/runtime fields are diagnostics.
    """

    if split not in SPLITS:
        raise BenchmarkV0Error(f"split must be one of {SPLITS}")
    schema_root = Path(schema_dir).resolve() if schema_dir is not None else _default_schema_dir()
    manifest_file = Path(manifest_path).resolve()

    # Phase 1: bind and close every prediction before any teacher artifact is
    # opened. The manifest digest is computed over the same bytes parsed here.
    prediction_manifest, prediction_manifest_sha256 = _read_json_with_sha256(
        manifest_file,
        label="benchmark manifest",
    )
    _validate_schema(
        prediction_manifest,
        MANIFEST_SCHEMA,
        schema_dir=schema_root,
        label="benchmark manifest",
    )
    _validate_manifest_cross_fields(prediction_manifest)
    _group_overlap(prediction_manifest)
    selected = [
        capture["capture_id"]
        for capture in prediction_manifest["captures"]
        if capture["split"] == split
    ]
    capture_by_id = {capture["capture_id"]: capture for capture in prediction_manifest["captures"]}
    predictions_root = Path(prediction_dir).resolve()
    if not predictions_root.is_dir():
        raise BenchmarkV0Error(f"prediction_dir is not a directory: {predictions_root}")
    expected_names = {f"{capture_id}.json" for capture_id in selected}
    actual_names = {
        item.name for item in predictions_root.iterdir() if item.is_file() or item.is_symlink()
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise BenchmarkV0Error(
            f"prediction_dir must contain exactly the selected split: "
            f"missing={missing}, extra={extra}"
        )

    predictions: list[dict[str, Any]] = []
    system: dict[str, Any] | None = None
    for capture_id in selected:
        path = _safe_artifact_path(
            predictions_root,
            f"{capture_id}.json",
            label=f"capture {capture_id} prediction",
        )
        prediction = _read_json(path, label=f"capture {capture_id} prediction")
        _validate_prediction(
            prediction,
            capture_by_id[capture_id],
            prediction_manifest,
            schema_dir=schema_root,
        )
        if system is None:
            system = dict(prediction["system"])
        elif prediction["system"] != system:
            raise BenchmarkV0Error("all predictions in one report must use one system identity")
        predictions.append(prediction)
    if system is None:
        raise BenchmarkV0Error(f"split {split} contains no captures")

    # Phase 2: only after prediction documents are closed may complete bundle
    # validation open the separately rooted teacher artifacts. The manifest
    # must remain byte-identical across the two phases.
    validated = validate_bundle(
        manifest_file,
        bundle_root=bundle_root,
        teacher_root=teacher_root,
        schema_dir=schema_root,
        _expected_manifest_sha256=prediction_manifest_sha256,
    )
    if validated["manifest_sha256"] != prediction_manifest_sha256:
        raise BenchmarkV0Error(
            "benchmark manifest changed after predictions were bound; refusing evaluation"
        )
    manifest = validated["manifest"]
    per_capture = [
        _evaluate_capture(prediction, validated["captures"][prediction["capture_id"]])
        for prediction in predictions
    ]

    reach_ll_count = sum(item["reach_ll"] for item in per_capture)
    solved_count = sum(item["solved_endpoint_completed"] for item in per_capture)
    abstention_count = sum(item["status"] == "abstained" for item in per_capture)
    total = len(per_capture)
    runtime_values = [item["runtime_ms"] for item in per_capture]
    edit_values = [item["edit_distance"] for item in per_capture]
    predicted_lengths = [item["predicted_length"] for item in per_capture]
    teacher_lengths = [item["teacher_length"] for item in per_capture]

    report = {
        "schema": "cubed-core/benchmark-v0-report",
        "schema_version": 1,
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "manifest_sha256": validated["manifest_sha256"],
        "split": split,
        "capture_count": total,
        "contract_valid": True,
        "claim_tier": "interface-boundary-only",
        "system": system,
        "teacher_isolation": {
            "prediction_lane": "camera-only",
            "prediction_input_binding_verified": True,
            "teacher_access_declaration": "none-declared",
            "teacher_root_separate_from_public_bundle": True,
            "process_environment_audited": False,
            "scope": (
                "The evaluator verifies prediction documents, exact input hashes, "
                "and the separation of public inputs from the teacher root. This "
                "establishes an interface boundary only; it does not audit the "
                "process or host that produced the predictions."
            ),
        },
        "split_integrity": {
            "unit": "capture",
            "fixed": True,
            "group_policy": manifest["split_policy"]["group_policy"],
            "observed_cross_split_groups": validated["group_overlap"],
        },
        "aggregate": {
            "primary": {
                "metric": "reach_ll_rate",
                "definition": "reach-ll-onset-with-correct-pre-ll-state",
                "successes": reach_ll_count,
                "total": total,
                "rate": _rate(reach_ll_count, total),
            },
            "secondary": {
                "metric": "solved_endpoint_completion_rate",
                "successes": solved_count,
                "total": total,
                "rate": _rate(solved_count, total),
            },
            "diagnostics": {
                "abstention_count": abstention_count,
                "abstention_rate": _rate(abstention_count, total),
                "edit_distance_mean": _mean(edit_values),
                "predicted_length_mean": _mean(predicted_lengths),
                "teacher_length_mean": _mean(teacher_lengths),
                "runtime_ms_mean": _mean(runtime_values),
                "runtime_ms_total": sum(runtime_values),
            },
        },
        "component_diagnostics": _component_aggregate(predictions),
        "captures": per_capture,
        "limitations": list(
            dict.fromkeys(
                [
                    *manifest["limitations"],
                    "Benchmark v0 is a development benchmark, not generalization evidence.",
                    (
                        "The evaluator derives the frozen 24 physical whole-cube rotations "
                        "and requires every held target state to match exactly."
                    ),
                    (
                        "The prediction schema has no teacher input and the evaluator binds "
                        "frozen artifacts; the prediction host is not audited for isolation."
                    ),
                    (
                        "Solved endpoint and edit distance are not substitutes for the primary "
                        "reach-LL state-trajectory metric."
                    ),
                ]
            )
        ),
    }
    _validate_schema(
        report,
        REPORT_SCHEMA,
        schema_dir=schema_root,
        label="benchmark report",
    )
    return report


def _validation_summary(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: validated[key]
        for key in (
            "schema",
            "schema_version",
            "benchmark_id",
            "benchmark_version",
            "manifest_sha256",
            "release_scope",
            "evidence_scope",
            "capture_count",
            "split_counts",
            "group_overlap",
            "public_rights_checked",
            "teacher_truth_checked",
        )
    }


def _json_text(value: Mapping[str, Any], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _write_new_json(path: Path, value: Mapping[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_json_text(value, pretty=pretty))
    except FileExistsError as exc:
        raise BenchmarkV0Error(f"refusing to overwrite existing report: {path}") from exc
    except OSError as exc:
        raise BenchmarkV0Error(f"cannot write report {path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cubed_core.benchmark_v0",
        description="Validate or evaluate the public Cubed Core Benchmark v0.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate public inputs, rights, and truth")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--bundle-root", type=Path)
    validate.add_argument("--teacher-root", type=Path, required=True)
    validate.add_argument("--schema-dir", type=Path)
    validate.add_argument("--pretty", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate camera-only predictions")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("prediction_dir", type=Path)
    evaluate.add_argument("--bundle-root", type=Path)
    evaluate.add_argument("--teacher-root", type=Path, required=True)
    evaluate.add_argument("--schema-dir", type=Path)
    evaluate.add_argument("--split", choices=SPLITS, default="test")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "validate":
            result = _validation_summary(
                validate_bundle(
                    args.manifest,
                    bundle_root=args.bundle_root,
                    teacher_root=args.teacher_root,
                    schema_dir=args.schema_dir,
                )
            )
            sys.stdout.write(_json_text(result, pretty=args.pretty))
            return 0

        result = evaluate_bundle(
            args.manifest,
            args.prediction_dir,
            bundle_root=args.bundle_root,
            teacher_root=args.teacher_root,
            split=args.split,
            schema_dir=args.schema_dir,
        )
        if args.output is None:
            sys.stdout.write(_json_text(result, pretty=args.pretty))
        else:
            _write_new_json(args.output, result, pretty=args.pretty)
            sys.stdout.write(f"wrote {args.output}\n")
        return 0
    except BenchmarkV0Error as exc:
        print(f"benchmark-v0: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
