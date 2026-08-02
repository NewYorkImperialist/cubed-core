"""Fail-closed construction and validation for Cubed Core public datasets.

This module is intentionally local-only. It does not authenticate to a host,
create a repository, upload bytes, or change repository visibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

BUILD_PLAN_SCHEMA = "public-dataset-build-plan-v1.schema.json"
CORPUS_MANIFEST_SCHEMA = "public-corpus-manifest-v1.schema.json"
CORPUS_RIGHTS_SCHEMA = "public-corpus-rights-v1.schema.json"
BENCHMARK_MANIFEST_SCHEMA = "cubed-core-benchmark-v0-manifest.schema.json"
BENCHMARK_SPLITS_SCHEMA = "public-dataset-splits-v1.schema.json"
BENCHMARK_TEACHER_MANIFEST_SCHEMA = "public-dataset-teacher-manifest-v1.schema.json"
METADATA_ROW_SCHEMA = "public-dataset-metadata-row-v1.schema.json"
READINESS_SCHEMA = "public-dataset-readiness-v1.schema.json"

BUILD_PLAN_KIND = "cubed-core/public-dataset-build-plan"
CORPUS_MANIFEST_KIND = "cubed-core/public-corpus-manifest"
READINESS_KIND = "cubed-core/public-dataset-readiness-report"
EVIDENCE_SCOPE = "corpus-not-benchmark-or-generalization-evidence"

ROOT_CONTRACT_PATHS = frozenset(
    {
        PurePosixPath(".gitattributes"),
        PurePosixPath("README.md"),
        PurePosixPath("LICENSES/DATASET.md"),
        PurePosixPath("dataset/manifest.json"),
        PurePosixPath("metadata.jsonl"),
        PurePosixPath("readiness-report.json"),
        PurePosixPath("SHA256SUMS"),
    }
)
HF_GITATTRIBUTES = (
    b"*.avi filter=lfs diff=lfs merge=lfs -text\n"
    b"*.mkv filter=lfs diff=lfs merge=lfs -text\n"
    b"*.mov filter=lfs diff=lfs merge=lfs -text\n"
    b"*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
    b"*.webm filter=lfs diff=lfs merge=lfs -text\n"
)
BENCHMARK_PATHS = {
    "public_manifest": PurePosixPath("public-inputs/manifest.json"),
    "splits_manifest": PurePosixPath("public-inputs/splits.json"),
    "teacher_manifest": PurePosixPath("teacher/manifest.json"),
}
VIDEO_ROLES = frozenset({"video", "high_speed_original"})
REQUIRED_CAPTURE_ROLES = frozenset({"video", "rights_record"})
BENCHMARK_INPUT_ROLES = ("video", "scramble", "calibration", "camera_metadata")
BENCHMARK_REQUIRED_ROLES = frozenset({*BENCHMARK_INPUT_ROLES, "teacher_truth", "rights_record"})
ALL_ROLES = frozenset(
    {
        "video",
        "high_speed_original",
        "video_derivation",
        "scramble",
        "calibration",
        "camera_metadata",
        "teacher_truth",
        "tracker_labels",
        "rights_record",
        "ble_session",
        "ble_session_app",
        "imu",
        "capture_manifest",
    }
)
ROLE_SCHEMA_ALLOWLIST: dict[str, frozenset[str | None]] = {
    "video": frozenset({None}),
    "high_speed_original": frozenset({None}),
    "video_derivation": frozenset({"capture-derivation-v1.schema.json"}),
    "scramble": frozenset(
        {
            "cubed-core-benchmark-v0-scramble.schema.json",
            "public-corpus-scramble-v1.schema.json",
        }
    ),
    "calibration": frozenset(
        {"color-calibration-v1.schema.json", "color-centroids-v1.schema.json"}
    ),
    "camera_metadata": frozenset({"cubed-core-benchmark-v0-camera-metadata.schema.json"}),
    "teacher_truth": frozenset(
        {
            "clip-ble-ground-truth-v1.schema.json",
            "cubed-core-benchmark-v0-teacher.schema.json",
        }
    ),
    "tracker_labels": frozenset({"frame-annotations-v1.schema.json"}),
    "rights_record": frozenset(
        {
            "public-corpus-rights-v1.schema.json",
            "cubed-core-benchmark-v0-rights.schema.json",
        }
    ),
    "ble_session": frozenset(
        {"raw-cube-session-v2.schema.json", "raw-cube-session-v3.schema.json"}
    ),
    "ble_session_app": frozenset({"raw-cube-session-v2.schema.json"}),
    "imu": frozenset({"raw-imu-v1.schema.json"}),
    "capture_manifest": frozenset({"raw-capture-manifest-v1.schema.json"}),
}
MAX_BASELINE_VIDEO_BYTES = 200 * 1024 * 1024
MAX_HIGH_SPEED_VIDEO_BYTES = 200 * 1024 * 1024
BASELINE_FPS_RANGE = (Fraction(110), Fraction(121))
HIGH_SPEED_FPS_RANGE = (Fraction(220), Fraction(242))
MIN_VIDEO_SHORT_EDGE = 1080
HASH_CHUNK_BYTES = 1024 * 1024

_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"(?i)(?:^|[\"'\s(])/(?:users|home)/[^/\s\"']+/"),
    re.compile(r"(?i)(?:^|[\"'\s(])/private/var/"),
    re.compile(r"(?i)(?:^|[\"'\s(])/(?:mnt|tmp|var/tmp|volumes|workspace)/"),
    re.compile(r"(?i)\b[a-z]:\\users\\"),
    re.compile(r"(?i)\bfile://"),
    re.compile(r"(?i)\b(?:gdrive|google-drive):"),
    re.compile(r"(?i)drive\.google\.com"),
    re.compile(r"(?i)workspace/agent-state"),
)
_PLACEHOLDER_EXACT = frozenset(
    {
        "example",
        "example only",
        "example text",
        "example value",
        "placeholder",
        "placeholder text",
        "placeholder value",
        "tbd",
        "todo",
    }
)
_PLACEHOLDER_PREFIX = re.compile(r"(?i)^(?:todo|tbd|placeholder)(?:\b|[\s:_-])")
_PLACEHOLDER_TOKEN = re.compile(r"(?i)(?:^|[^a-z0-9])(?:todo|tbd|placeholder)(?:$|[^a-z0-9])")
_EXAMPLE_IDENTIFIER = re.compile(r"(?i)(?:^|[^a-z0-9])example(?:$|[^a-z0-9])")
_REPLACEMENT_INSTRUCTION = re.compile(r"(?i)^(?:fill\s+(?:this|me)\s+in|replace\s+(?:this|me))\b")

ProbeFunction = Callable[[Path, int], Mapping[str, Any]]


class PublicDatasetError(ValueError):
    """Raised when a candidate dataset fails a public-release contract."""


class _DuplicateJSONKeyError(ValueError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def schema_directory() -> Path:
    return repository_root() / "schemas"


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _validate_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PublicDatasetError(f"{label}: non-finite JSON number is not allowed")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _validate_finite(child, label=label)


def _parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicDatasetError(f"{label}: invalid strict JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise PublicDatasetError(f"{label}: top-level JSON value must be an object")
    _validate_finite(document, label=label)
    return document


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PublicDatasetError(f"{label}: cannot read {path}: {exc}") from exc
    return _parse_json(raw, label=label)


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_path = schema_directory() / schema_name
    schema = _read_json(schema_path, label=f"schema {schema_name}")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PublicDatasetError(f"invalid repository schema {schema_name}: {exc}") from exc
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate_schema(
    document: Mapping[str, Any],
    schema_name: str,
    *,
    label: str,
) -> None:
    errors = sorted(
        _schema_validator(schema_name).iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return
    first = errors[0]
    location = "$"
    for part in first.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    raise PublicDatasetError(f"{label}: schema violation at {location}: {first.message}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise PublicDatasetError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path) -> bool:
    candidate = Path(path.anchor)
    for part in path.parts[1:]:
        candidate /= part
        if candidate.is_symlink():
            return True
        if not candidate.exists():
            break
    return False


def _require_private_build_location(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise PublicDatasetError(f"{label} must be an absolute path")
    absolute = path.absolute()
    if _has_symlink_component(absolute):
        raise PublicDatasetError(f"{label} must not traverse a symlink")
    resolved = absolute.resolve(strict=False)
    if _inside(resolved, repository_root().resolve()):
        raise PublicDatasetError(
            f"{label} must be outside the Git checkout; use a private staging directory"
        )
    return resolved


def _require_source_file(path_value: Any, *, label: str) -> Path:
    if not isinstance(path_value, str):
        raise PublicDatasetError(f"{label}.source_path must be a string")
    path = Path(path_value)
    if not path.is_absolute():
        raise PublicDatasetError(f"{label}.source_path must be absolute")
    if _has_symlink_component(path.absolute()):
        raise PublicDatasetError(f"{label}.source_path must not traverse a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PublicDatasetError(f"{label}.source_path does not exist: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise PublicDatasetError(f"{label}.source_path must be a regular non-symlink file")
    return resolved


def _verify_expected_source(
    reference: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _require_source_file(reference["source_path"], label=label)
    actual_bytes = path.stat().st_size
    if actual_bytes != reference["bytes"]:
        raise PublicDatasetError(
            f"{label}: byte count mismatch: expected={reference['bytes']} actual={actual_bytes}"
        )
    actual_hash = _sha256_file(path)
    if actual_hash != reference["sha256"]:
        raise PublicDatasetError(
            f"{label}: SHA-256 mismatch: expected={reference['sha256']} actual={actual_hash}"
        )
    return path


def _public_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PublicDatasetError(f"{label} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicDatasetError(f"{label} must be a normalized relative POSIX path")
    return path


def _safe_output_file(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    if _has_symlink_component(candidate.absolute()):
        raise PublicDatasetError(f"{label} must not traverse a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PublicDatasetError(f"{label} is missing: {relative}") from exc
    if not _inside(resolved, root.resolve()):
        raise PublicDatasetError(f"{label} escapes the dataset root: {relative}")
    if candidate.is_symlink() or not resolved.is_file():
        raise PublicDatasetError(f"{label} must be a regular non-symlink file: {relative}")
    return resolved


def _reject_private_text(raw: bytes, *, label: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDatasetError(f"{label}: expected UTF-8 public text") from exc
    for pattern in _PRIVATE_TEXT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise PublicDatasetError(f"{label}: contains a private path or Google Drive reference")


def _front_matter_scalar(value: str) -> str:
    result = value.strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in {"'", '"'}:
        result = result[1:-1]
    return result.strip()


def _is_obvious_placeholder(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip()).casefold()
    unwrapped = normalized.strip(" \t\r\n<>[]{}():;,.!?*_`'\"")
    return (
        unwrapped in _PLACEHOLDER_EXACT
        or _PLACEHOLDER_PREFIX.match(unwrapped) is not None
        or _REPLACEMENT_INSTRUCTION.match(unwrapped) is not None
    )


def _require_release_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicDatasetError(f"{label} must be non-empty reviewed public text")
    if _is_obvious_placeholder(value) or _PLACEHOLDER_TOKEN.search(value) is not None:
        raise PublicDatasetError(f"{label} contains obvious placeholder material")
    return value


def _require_release_identifier(value: Any, *, label: str) -> str:
    reviewed = _require_release_text(value, label=label)
    if (
        _PLACEHOLDER_TOKEN.search(reviewed) is not None
        or _EXAMPLE_IDENTIFIER.search(reviewed) is not None
    ):
        raise PublicDatasetError(f"{label} contains a placeholder token")
    return reviewed


def _require_nonplaceholder_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise PublicDatasetError(f"{label} must be a lowercase SHA-256")
    for unit_length in range(1, 9):
        if 64 % unit_length == 0 and value == value[:unit_length] * (64 // unit_length):
            raise PublicDatasetError(f"{label} is an obvious repeated placeholder digest")
    return value


def _markdown_section_text(lines: Sequence[str], heading: str) -> str:
    start = lines.index(heading) + 1
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("#"):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _validate_dataset_card(raw: bytes, *, collection_license: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDatasetError("dataset card must be UTF-8") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise PublicDatasetError("dataset card must start with Hugging Face YAML front matter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise PublicDatasetError(
            "dataset card Hugging Face YAML front matter is not closed"
        ) from exc
    metadata: dict[str, str | list[str]] = {}
    current_list: str | None = None
    for line_number, line in enumerate(lines[1:closing], start=2):
        key_match = re.fullmatch(r"([A-Za-z0-9_-]+):(?:\s*(.*))?", line)
        if key_match is not None:
            key = key_match.group(1)
            if key in metadata:
                raise PublicDatasetError(f"dataset card front matter has duplicate key {key!r}")
            scalar = _front_matter_scalar(key_match.group(2) or "")
            if scalar:
                metadata[key] = scalar
                current_list = None
            else:
                metadata[key] = []
                current_list = key
            continue
        item_match = re.fullmatch(r"-\s+(.+)", line)
        if item_match is not None and current_list is not None:
            values = metadata[current_list]
            assert isinstance(values, list)
            values.append(_front_matter_scalar(item_match.group(1)))
            continue
        if line.strip() and not line.lstrip().startswith("#"):
            raise PublicDatasetError(
                f"dataset card front matter line {line_number} is not in the "
                "supported scalar/list form"
            )

    license_value = metadata.get("license")
    if not isinstance(license_value, str) or not license_value:
        raise PublicDatasetError("dataset card front matter requires license")
    if license_value.lower() == "other":
        if (
            metadata.get("license_name") != collection_license
            or metadata.get("license_link") != "LICENSES/DATASET.md"
        ):
            raise PublicDatasetError(
                "dataset card license=other requires the exact collection "
                "license_name and license_link: LICENSES/DATASET.md"
            )
    elif license_value.lower() != collection_license.lower():
        raise PublicDatasetError(
            "dataset card license does not match collection_license.license_id"
        )
    pretty_name = metadata.get("pretty_name")
    if (
        not isinstance(pretty_name, str)
        or not 3 <= len(pretty_name) <= 200
        or _is_obvious_placeholder(pretty_name)
    ):
        raise PublicDatasetError("dataset card front matter requires a non-placeholder pretty_name")
    tags = metadata.get("tags")
    if not isinstance(tags, list) or not {"video", "computer-vision"} <= {
        item.lower() for item in tags
    }:
        raise PublicDatasetError(
            "dataset card front matter tags must include video and computer-vision"
        )
    task_categories = metadata.get("task_categories")
    if isinstance(task_categories, list) and (
        not task_categories or any(_is_obvious_placeholder(value) for value in task_categories)
    ):
        raise PublicDatasetError(
            "dataset card task_categories must be omitted or contain reviewed values"
        )

    body_lines = lines[closing + 1 :]
    required_headings = (
        "## Intended use",
        "## Limitations",
        "## Licenses and attribution",
        "## Privacy",
    )
    missing = [heading for heading in required_headings if heading not in body_lines]
    if not any(line.startswith("# ") for line in body_lines):
        missing.insert(0, "one top-level '# ' title")
    if missing:
        raise PublicDatasetError(
            f"dataset card is missing required public documentation headings: {missing}"
        )
    title = next(line[2:].strip() for line in body_lines if line.startswith("# "))
    _require_release_text(title, label="dataset card title")
    for heading in required_headings:
        section = _markdown_section_text(body_lines, heading)
        if (
            len(section) < 20
            or _is_obvious_placeholder(section)
            or _PLACEHOLDER_TOKEN.search(section) is not None
        ):
            raise PublicDatasetError(
                f"dataset card section {heading!r} requires substantive reviewed text"
            )


def _validate_dataset_license(
    raw: bytes,
    *,
    collection_license: str,
    collection_attribution: str,
) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDatasetError("dataset license must be UTF-8") from exc
    stripped = text.strip()
    if len(stripped) < 120:
        raise PublicDatasetError(
            "dataset license is inadequate: provide substantive license, scope, "
            "artifact-license, and attribution terms"
        )
    meaningful_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not meaningful_lines or not meaningful_lines[0].startswith("# "):
        raise PublicDatasetError("dataset license requires a top-level Markdown title")
    if any(
        _is_obvious_placeholder(line) or _PLACEHOLDER_TOKEN.search(line) is not None
        for line in meaningful_lines
    ):
        raise PublicDatasetError("dataset license contains obvious placeholder material")
    if collection_license.casefold() not in text.casefold():
        raise PublicDatasetError("dataset license must name collection_license.license_id exactly")
    if collection_attribution not in text:
        raise PublicDatasetError(
            "dataset license must include collection_license.attribution exactly"
        )
    normalized = re.sub(r"[-_\s]+", " ", text.casefold())
    if "selection and arrangement" not in normalized:
        raise PublicDatasetError("dataset license must state the selection-and-arrangement scope")
    if not (
        "per artifact" in normalized
        or "artifact license" in normalized
        or "artifact licenses" in normalized
    ):
        raise PublicDatasetError(
            "dataset license must state that per-artifact licenses are authoritative"
        )


def _fraction(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str):
        raise PublicDatasetError(f"{label} must be a rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise PublicDatasetError(f"{label} is not a valid rational: {value!r}") from exc
    if result <= 0:
        raise PublicDatasetError(f"{label} must be positive")
    return result


def probe_video(path: Path, timeout_seconds: int = 300) -> dict[str, Any]:
    """Return stable ffprobe facts for one local video."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PublicDatasetError(
            "ffprobe is required to validate public dataset videos; install FFmpeg"
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,avg_frame_rate,"
            "r_frame_rate,nb_frames,nb_read_frames:format=format_name"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicDatasetError(f"ffprobe could not inspect {path.name}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe rejected the file"
        raise PublicDatasetError(f"ffprobe could not inspect {path.name}: {detail}")
    try:
        payload = json.loads(
            result.stdout,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PublicDatasetError(f"ffprobe returned invalid JSON for {path.name}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise PublicDatasetError(f"ffprobe returned an invalid stream inventory for {path.name}")
    streams = payload["streams"]
    videos = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audio_count = sum(
        isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
    )
    if len(videos) != 1:
        raise PublicDatasetError(f"{path.name}: expected exactly one video stream")
    if len(streams) != 1 or audio_count:
        raise PublicDatasetError(
            f"{path.name}: public videos must contain no audio or auxiliary streams"
        )
    video = videos[0]
    fps_value = video.get("avg_frame_rate")
    if fps_value in {None, "0/0", "N/A"}:
        fps_value = video.get("r_frame_rate")
    fps = _fraction(fps_value, label=f"{path.name} measured frame rate")
    frame_count_value = video.get("nb_read_frames")
    if frame_count_value in {None, "N/A"}:
        frame_count_value = video.get("nb_frames")
    try:
        frame_count = int(frame_count_value)
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicDatasetError(
            f"{path.name}: ffprobe did not return dimensions and a frame count"
        ) from exc
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise PublicDatasetError(f"{path.name}: video dimensions and frame count must be positive")
    codec = video.get("codec_name")
    format_value = payload.get("format")
    container = format_value.get("format_name") if isinstance(format_value, dict) else None
    if not isinstance(codec, str) or not codec:
        raise PublicDatasetError(f"{path.name}: ffprobe did not identify the codec")
    if not isinstance(container, str) or not container:
        raise PublicDatasetError(f"{path.name}: ffprobe did not identify the container")
    return {
        "codec": codec,
        "container": container,
        "width": width,
        "height": height,
        "measured_fps": {
            "numerator": fps.numerator,
            "denominator": fps.denominator,
        },
        "frame_count": frame_count,
        "audio_stream_count": audio_count,
    }


def _normalize_probe(probe: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        fps_value = probe["measured_fps"]
        if not isinstance(fps_value, Mapping):
            raise TypeError
        fps = Fraction(int(fps_value["numerator"]), int(fps_value["denominator"]))
        normalized = {
            "codec": str(probe["codec"]),
            "container": str(probe["container"]),
            "width": int(probe["width"]),
            "height": int(probe["height"]),
            "measured_fps": {
                "numerator": fps.numerator,
                "denominator": fps.denominator,
            },
            "frame_count": int(probe["frame_count"]),
            "audio_stream_count": int(probe["audio_stream_count"]),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise PublicDatasetError(f"{label}: invalid video probe facts") from exc
    if (
        not normalized["codec"]
        or not normalized["container"]
        or normalized["width"] <= 0
        or normalized["height"] <= 0
        or normalized["frame_count"] <= 0
        or fps <= 0
        or normalized["audio_stream_count"] < 0
    ):
        raise PublicDatasetError(f"{label}: invalid video probe facts")
    return normalized


def _validate_video_facts(
    path: Path,
    role: str,
    expected_bytes: int,
    *,
    probe: ProbeFunction,
    timeout_seconds: int,
) -> dict[str, Any]:
    label = f"{role} {path.name}"
    facts = _normalize_probe(probe(path, timeout_seconds), label=label)
    fps = Fraction(
        facts["measured_fps"]["numerator"],
        facts["measured_fps"]["denominator"],
    )
    if facts["audio_stream_count"] != 0:
        raise PublicDatasetError(f"{label}: audio streams are not allowed")
    if min(facts["width"], facts["height"]) < MIN_VIDEO_SHORT_EDGE:
        raise PublicDatasetError(f"{label}: minimum encoded short edge is {MIN_VIDEO_SHORT_EDGE}px")
    if role == "video":
        lower, upper = BASELINE_FPS_RANGE
        maximum_bytes = MAX_BASELINE_VIDEO_BYTES
    elif role == "high_speed_original":
        lower, upper = HIGH_SPEED_FPS_RANGE
        maximum_bytes = MAX_HIGH_SPEED_VIDEO_BYTES
    else:
        raise PublicDatasetError(f"unsupported video role: {role}")
    if not lower <= fps <= upper:
        raise PublicDatasetError(f"{label}: measured cadence {fps} is outside {lower}..{upper} fps")
    if expected_bytes > maximum_bytes:
        raise PublicDatasetError(
            f"{label}: {expected_bytes} bytes exceeds the {maximum_bytes}-byte ceiling"
        )
    return facts


def _validate_role_contract(artifact: Mapping[str, Any], *, label: str) -> None:
    role = artifact["role"]
    if role not in ALL_ROLES:
        raise PublicDatasetError(f"{label}: unsupported role {role!r}")
    schema_file = artifact["schema_file"]
    if schema_file not in ROLE_SCHEMA_ALLOWLIST[role]:
        allowed = sorted("null" if item is None else item for item in ROLE_SCHEMA_ALLOWLIST[role])
        raise PublicDatasetError(
            f"{label}: schema_file {schema_file!r} is not allowed for {role}; "
            f"expected one of {allowed}"
        )
    media_type = artifact["media_type"]
    if role in VIDEO_ROLES:
        if not media_type.startswith("video/"):
            raise PublicDatasetError(f"{label}: video roles require a video/* media_type")
    elif media_type != "application/json":
        raise PublicDatasetError(f"{label}: JSON roles require application/json")


def _validate_path_set(paths: Sequence[PurePosixPath]) -> None:
    seen: set[PurePosixPath] = set()
    reserved_paths = ROOT_CONTRACT_PATHS | frozenset(BENCHMARK_PATHS.values())
    for path in paths:
        if path in reserved_paths:
            raise PublicDatasetError(f"artifact path is reserved: {path}")
        for reserved in reserved_paths:
            artifact_is_ancestor = (
                len(path.parts) < len(reserved.parts)
                and reserved.parts[: len(path.parts)] == path.parts
            )
            reserved_is_ancestor = (
                len(reserved.parts) < len(path.parts)
                and path.parts[: len(reserved.parts)] == reserved.parts
            )
            if artifact_is_ancestor or reserved_is_ancestor:
                raise PublicDatasetError(
                    f"artifact path conflicts with generated path {reserved}: {path}"
                )
        if path in seen:
            raise PublicDatasetError(f"duplicate public artifact path: {path}")
        seen.add(path)
    ordered = sorted(seen, key=lambda item: item.as_posix())
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if len(second.parts) <= len(first.parts):
                continue
            if second.parts[: len(first.parts)] == first.parts:
                raise PublicDatasetError(
                    f"public artifact path conflicts with a child path: {first}, {second}"
                )


def _artifact_reference(
    artifact: Mapping[str, Any],
    *,
    path: PurePosixPath,
    media: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact["artifact_id"],
        "role": artifact["role"],
        "path": path.as_posix(),
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
        "media_type": artifact["media_type"],
        "public_license": artifact["public_license"],
        "attribution": artifact["attribution"],
        "privacy_reviewed": True,
        "rights_reviewed": True,
        "schema_file": artifact["schema_file"],
        "media": media,
    }


def _verify_json_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    capture_id: str,
    label: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    _reject_private_text(raw, label=label)
    document = _parse_json(raw, label=label)
    schema_file = artifact["schema_file"]
    if not isinstance(schema_file, str):
        raise PublicDatasetError(f"{label}: JSON artifact has no schema_file")
    _validate_schema(document, schema_file, label=label)
    document_capture_id = document.get("capture_id")
    if document_capture_id is not None and document_capture_id != capture_id:
        raise PublicDatasetError(f"{label}: capture_id does not match its capture")
    return document


def _generic_rights_expected(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "artifact_id": artifact["artifact_id"],
                "path": artifact["path"],
                "bytes": artifact["bytes"],
                "sha256": artifact["sha256"],
                "public_license": artifact["public_license"],
                "attribution": artifact["attribution"],
            }
            for artifact in artifacts
            if artifact["role"] != "rights_record"
        ],
        key=lambda item: item["artifact_id"],
    )


def _benchmark_rights_expected(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "role": artifact["role"],
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "public_release": True,
                "public_license": artifact["public_license"],
                "attribution": artifact["attribution"],
            }
            for artifact in artifacts
            if artifact["role"] != "rights_record"
        ],
        key=lambda item: item["role"],
    )


def _benchmark_artifact_path(role: str, path_value: Any) -> str:
    path = _public_path(path_value, label=f"Benchmark v0 {role} path")
    expected_root = "teacher" if role == "teacher_truth" else "public-inputs"
    if not path.parts or path.parts[0] != expected_root or len(path.parts) == 1:
        raise PublicDatasetError(f"Benchmark v0 {role} must be staged below {expected_root}/")
    return PurePosixPath(*path.parts[1:]).as_posix()


def _benchmark_reference(artifact: Mapping[str, Any]) -> dict[str, Any]:
    role = artifact["role"]
    return {
        "path": _benchmark_artifact_path(role, artifact["path"]),
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
        "media_type": artifact["media_type"],
        "public_license": artifact["public_license"],
        "attribution": artifact["attribution"],
    }


def _validate_rights_binding(
    rights: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    capture_id: str,
    schema_file: str,
) -> None:
    if rights.get("capture_id") != capture_id:
        raise PublicDatasetError(f"capture {capture_id}: rights record is bound to another capture")
    terms_version = rights.get("terms_version")
    if not isinstance(terms_version, str) or "draft" in terms_version.lower():
        raise PublicDatasetError(
            f"capture {capture_id}: rights terms_version must be final, not draft"
        )
    for field in ("record_id", "terms_version", "reviewer_id"):
        if field in rights:
            _require_release_identifier(
                rights[field],
                label=f"capture {capture_id}: rights {field}",
            )
    for field in ("terms_sha256", "acceptance_receipt_sha256"):
        if field in rights:
            _require_nonplaceholder_sha256(
                rights[field],
                label=f"capture {capture_id}: rights {field}",
            )
    for index, entry in enumerate(rights["artifacts"]):
        _require_release_identifier(
            entry["public_license"],
            label=f"capture {capture_id}: rights artifacts[{index}].public_license",
        )
        _require_release_text(
            entry["attribution"],
            label=f"capture {capture_id}: rights artifacts[{index}].attribution",
        )
    if schema_file == CORPUS_RIGHTS_SCHEMA:
        actual = sorted(rights["artifacts"], key=lambda item: item["artifact_id"])
        expected = _generic_rights_expected(artifacts)
    elif schema_file == "cubed-core-benchmark-v0-rights.schema.json":
        actual = sorted(
            [
                {key: value for key, value in entry.items() if key != "rights_basis"}
                for entry in rights["artifacts"]
            ],
            key=lambda item: item["role"],
        )
        expected = _benchmark_rights_expected(
            [
                {
                    **artifact,
                    "path": _benchmark_artifact_path(
                        artifact["role"],
                        artifact["path"],
                    ),
                }
                for artifact in artifacts
            ]
        )
    else:
        raise PublicDatasetError(
            f"capture {capture_id}: unsupported public rights schema {schema_file}"
        )
    if actual != expected:
        raise PublicDatasetError(
            f"capture {capture_id}: rights entries do not exactly bind every public artifact"
        )


def _artifact_roles(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {artifact["role"]: artifact for artifact in artifacts}


def _fps_from_media(artifact: Mapping[str, Any]) -> Fraction:
    media = artifact["media"]
    return Fraction(
        media["measured_fps"]["numerator"],
        media["measured_fps"]["denominator"],
    )


def _validate_capture_cross_bindings(
    capture_id: str,
    artifacts: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    by_role = _artifact_roles(artifacts)
    video = by_role["video"]
    benchmark_paths = (
        by_role.get("rights_record", {}).get("schema_file")
        == "cubed-core-benchmark-v0-rights.schema.json"
    )

    def binding_path(artifact: Mapping[str, Any]) -> str:
        if benchmark_paths:
            return _benchmark_artifact_path(artifact["role"], artifact["path"])
        return artifact["path"]

    high_speed = by_role.get("high_speed_original")
    derivation_artifact = by_role.get("video_derivation")
    if (high_speed is None) != (derivation_artifact is None):
        raise PublicDatasetError(
            f"capture {capture_id}: high_speed_original and video_derivation "
            "must be published together"
        )
    if derivation_artifact is not None and high_speed is not None:
        derivation = documents["video_derivation"]
        source = derivation["source"]
        derivative = derivation["derivative"]
        expected_source = {
            "path": binding_path(high_speed),
            "bytes": high_speed["bytes"],
            "sha256": high_speed["sha256"],
        }
        expected_derivative = {
            "path": binding_path(video),
            "bytes": video["bytes"],
            "sha256": video["sha256"],
        }
        for field, value in expected_source.items():
            if source[field] != value:
                raise PublicDatasetError(
                    f"capture {capture_id}: derivation source.{field} "
                    "does not bind high_speed_original"
                )
        for field, value in expected_derivative.items():
            if derivative[field] != value:
                raise PublicDatasetError(
                    f"capture {capture_id}: derivation derivative.{field} does not bind video"
                )
        if _fraction(
            derivation["transform"]["source_frame_rate"],
            label=f"capture {capture_id} derivation source cadence",
        ) != _fps_from_media(high_speed):
            raise PublicDatasetError(
                f"capture {capture_id}: derivation source cadence disagrees with media"
            )
        if _fraction(
            derivation["transform"]["target_frame_rate"],
            label=f"capture {capture_id} derivation target cadence",
        ) != _fps_from_media(video):
            raise PublicDatasetError(
                f"capture {capture_id}: derivation target cadence disagrees with media"
            )
        if derivation["transform"]["expected_output_frame_count"] != video["media"]["frame_count"]:
            raise PublicDatasetError(
                f"capture {capture_id}: derivation output frame count disagrees with video"
            )

    if benchmark_paths:
        camera_metadata = documents.get("camera_metadata")
        if camera_metadata is None:
            raise PublicDatasetError(f"capture {capture_id}: Benchmark v0 requires camera metadata")
        canonical_receipt = camera_metadata["video_probe_receipt"]["canonical_video"]
        canonical_expected = {
            "path": binding_path(video),
            "bytes": video["bytes"],
            "sha256": video["sha256"],
            "measured_fps": video["media"]["measured_fps"],
            "frame_count": video["media"]["frame_count"],
            "width": video["media"]["width"],
            "height": video["media"]["height"],
        }
        for field, expected in canonical_expected.items():
            if canonical_receipt[field] != expected:
                raise PublicDatasetError(
                    f"capture {capture_id}: reviewed camera receipt "
                    f"canonical_video.{field} disagrees with the staged artifact"
                )
        for field in ("measured_fps", "frame_count", "width", "height"):
            if camera_metadata[field] != video["media"][field]:
                raise PublicDatasetError(
                    f"capture {capture_id}: camera metadata {field} disagrees "
                    "with the builder's media probe"
                )

        source_receipt = camera_metadata["video_probe_receipt"]["source_video"]
        if high_speed is None:
            if source_receipt is not None:
                raise PublicDatasetError(
                    f"capture {capture_id}: camera metadata has an undeclared "
                    "high-speed source receipt"
                )
        else:
            if source_receipt is None:
                raise PublicDatasetError(
                    f"capture {capture_id}: high-speed artifact has no reviewed source receipt"
                )
            source_expected = {
                "path": binding_path(high_speed),
                "bytes": high_speed["bytes"],
                "sha256": high_speed["sha256"],
                "measured_fps": high_speed["media"]["measured_fps"],
                "frame_count": high_speed["media"]["frame_count"],
                "width": high_speed["media"]["width"],
                "height": high_speed["media"]["height"],
            }
            for field, expected in source_expected.items():
                if source_receipt[field] != expected:
                    raise PublicDatasetError(
                        f"capture {capture_id}: reviewed camera receipt "
                        f"source_video.{field} disagrees with the staged artifact"
                    )

    if "tracker_labels" in by_role:
        source = documents["tracker_labels"]["source"]
        if source["capture_id"] != capture_id:
            raise PublicDatasetError(
                f"capture {capture_id}: tracker labels require the public capture_id"
            )
        if source["sha256"] != video["sha256"]:
            raise PublicDatasetError(
                f"capture {capture_id}: tracker labels are bound to another video"
            )
        if source["frame_count"] != video["media"]["frame_count"]:
            raise PublicDatasetError(
                f"capture {capture_id}: tracker label frame count disagrees with video"
            )
        if abs(Fraction(str(source["fps"])) - _fps_from_media(video)) > Fraction(1, 100):
            raise PublicDatasetError(
                f"capture {capture_id}: tracker label cadence disagrees with video"
            )

    teacher = by_role.get("teacher_truth")
    if teacher is not None:
        teacher_document = documents["teacher_truth"]
        if teacher["schema_file"] == "clip-ble-ground-truth-v1.schema.json":
            clip = teacher_document["clip"]
            expected_clip = {
                "filename": PurePosixPath(video["path"]).name,
                "sha256": video["sha256"],
                "bytes": video["bytes"],
                "frame_count": video["media"]["frame_count"],
            }
            for field, value in expected_clip.items():
                if clip[field] != value:
                    raise PublicDatasetError(
                        f"capture {capture_id}: clip teacher {field} does not bind the public video"
                    )
            if abs(Fraction(str(clip["fps"])) - _fps_from_media(video)) > Fraction(1, 100):
                raise PublicDatasetError(
                    f"capture {capture_id}: clip teacher cadence disagrees with video"
                )
        elif teacher["schema_file"] == "cubed-core-benchmark-v0-teacher.schema.json":
            scramble = by_role.get("scramble")
            if scramble is None or teacher_document["scramble_sha256"] != scramble["sha256"]:
                raise PublicDatasetError(
                    f"capture {capture_id}: benchmark teacher does not bind scramble"
                )


def _metadata_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capture in manifest["captures"]:
        by_role = {artifact["role"]: artifact for artifact in capture["artifacts"]}
        video = by_role["video"]
        media = video["media"]
        rows.append(
            {
                "attribution": video["attribution"],
                "capture_id": capture["capture_id"],
                "evidence_scope": EVIDENCE_SCOPE,
                "file_name": video["path"],
                "fps_denominator": media["measured_fps"]["denominator"],
                "fps_numerator": media["measured_fps"]["numerator"],
                "frame_count": media["frame_count"],
                "height": media["height"],
                "high_speed_file_name": (
                    by_role["high_speed_original"]["path"]
                    if "high_speed_original" in by_role
                    else None
                ),
                "public_license": video["public_license"],
                "split": capture["split"],
                "teacher_truth_available": "teacher_truth" in by_role,
                "teacher_truth_file_name": (
                    by_role["teacher_truth"]["path"] if "teacher_truth" in by_role else None
                ),
                "tracker_labels_available": "tracker_labels" in by_role,
                "tracker_labels_file_name": (
                    by_role["tracker_labels"]["path"] if "tracker_labels" in by_role else None
                ),
                "width": media["width"],
            }
        )
    return sorted(rows, key=lambda item: item["capture_id"])


def _validate_release_metadata(manifest: Mapping[str, Any]) -> None:
    for field in ("dataset_id", "release_id"):
        _require_release_identifier(
            manifest[field],
            label=f"corpus manifest {field}",
        )
    _require_release_text(
        manifest["description"],
        label="corpus manifest description",
    )
    for index, limitation in enumerate(manifest["limitations"]):
        _require_release_text(
            limitation,
            label=f"corpus manifest limitations[{index}]",
        )
    collection = manifest["collection_license"]
    _require_release_identifier(
        collection["license_id"],
        label="collection_license.license_id",
    )
    _require_release_text(
        collection["attribution"],
        label="collection_license.attribution",
    )
    for index, exception in enumerate(collection["exceptions"]):
        _require_release_identifier(
            exception["public_license"],
            label=f"collection_license.exceptions[{index}].public_license",
        )
        _require_release_text(
            exception["reason"],
            label=f"collection_license.exceptions[{index}].reason",
        )
    for capture in manifest["captures"]:
        capture_id = capture["capture_id"]
        for artifact in capture["artifacts"]:
            role = artifact["role"]
            _require_release_identifier(
                artifact["public_license"],
                label=f"capture {capture_id}/{role} public_license",
            )
            _require_release_text(
                artifact["attribution"],
                label=f"capture {capture_id}/{role} attribution",
            )


def _validate_collection_license(manifest: Mapping[str, Any]) -> None:
    _validate_release_metadata(manifest)
    collection = manifest["collection_license"]
    collection_license = collection["license_id"]
    artifacts: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_exceptions: set[tuple[str, str]] = set()
    for capture in manifest["captures"]:
        capture_id = capture["capture_id"]
        for artifact in capture["artifacts"]:
            key = (capture_id, artifact["role"])
            artifacts[key] = artifact
            if artifact["public_license"] != collection_license:
                expected_exceptions.add(key)

    declared_exceptions: set[tuple[str, str]] = set()
    for exception in collection["exceptions"]:
        key = (exception["capture_id"], exception["role"])
        if key in declared_exceptions:
            raise PublicDatasetError(
                f"duplicate collection-license exception for {key[0]}/{key[1]}"
            )
        declared_exceptions.add(key)
        artifact = artifacts.get(key)
        if artifact is None:
            raise PublicDatasetError(
                f"collection-license exception references unknown artifact {key[0]}/{key[1]}"
            )
        if exception["public_license"] != artifact["public_license"]:
            raise PublicDatasetError(
                f"collection-license exception disagrees for {key[0]}/{key[1]}"
            )
    if declared_exceptions != expected_exceptions:
        missing = sorted(expected_exceptions - declared_exceptions)
        extra = sorted(declared_exceptions - expected_exceptions)
        raise PublicDatasetError(
            "collection-license exceptions must cover exactly the artifacts "
            f"whose license differs: missing={missing}, extra={extra}"
        )


def _benchmark_collection_license(
    collection: Mapping[str, Any],
    capture_ids: set[str],
) -> dict[str, Any]:
    return {
        "license_id": collection["license_id"],
        "attribution": collection["attribution"],
        "scope": collection["scope"],
        "artifact_licenses_authoritative": collection["artifact_licenses_authoritative"],
        "exceptions": [
            exception
            for exception in collection["exceptions"]
            if exception["capture_id"] in capture_ids
        ],
    }


def _build_benchmark_documents(
    plan: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    collection_license: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[PurePosixPath, bytes]]:
    benchmark_id = _require_release_identifier(
        plan["benchmark_id"],
        label="benchmark_v0.benchmark_id",
    )
    for index, limitation in enumerate(plan["limitations"]):
        _require_release_text(
            limitation,
            label=f"benchmark_v0.limitations[{index}]",
        )
    for key, policy in plan["split_policy"]["group_policy"].items():
        if policy["mode"] == "disclosed-overlap":
            _require_release_text(
                policy["reason"],
                label=f"benchmark_v0.split_policy.group_policy.{key}.reason",
            )

    generic_by_id = {capture["capture_id"]: capture for capture in captures}
    member_by_id: dict[str, Mapping[str, Any]] = {}
    for member in plan["captures"]:
        capture_id = member["capture_id"]
        if capture_id in member_by_id:
            raise PublicDatasetError(f"benchmark_v0 has duplicate capture_id {capture_id}")
        if capture_id not in generic_by_id:
            raise PublicDatasetError(
                f"benchmark_v0 references capture absent from the public corpus: {capture_id}"
            )
        for key, value in member["groups"].items():
            _require_release_identifier(
                value,
                label=f"benchmark_v0 capture {capture_id} groups.{key}",
            )
        member_by_id[capture_id] = member

    selected_ids = set(member_by_id)
    benchmark_captures: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    teacher_artifacts: list[dict[str, Any]] = []
    for capture_id in sorted(selected_ids):
        capture = generic_by_id[capture_id]
        split = capture["split"]
        if split not in split_ids:
            raise PublicDatasetError(
                f"benchmark_v0 capture {capture_id} requires an assigned "
                "train, validation, or test split"
            )
        by_role = _artifact_roles(capture["artifacts"])
        missing = BENCHMARK_REQUIRED_ROLES - set(by_role)
        if missing:
            raise PublicDatasetError(
                f"benchmark_v0 capture {capture_id} is missing roles {sorted(missing)}"
            )
        expected_schemas = {
            "scramble": "cubed-core-benchmark-v0-scramble.schema.json",
            "camera_metadata": "cubed-core-benchmark-v0-camera-metadata.schema.json",
            "teacher_truth": "cubed-core-benchmark-v0-teacher.schema.json",
            "rights_record": "cubed-core-benchmark-v0-rights.schema.json",
        }
        for role, expected_schema in expected_schemas.items():
            if by_role[role]["schema_file"] != expected_schema:
                raise PublicDatasetError(
                    f"benchmark_v0 capture {capture_id}/{role} requires {expected_schema}"
                )

        inputs = {role: _benchmark_reference(by_role[role]) for role in BENCHMARK_INPUT_ROLES}
        for role in ("high_speed_original", "video_derivation"):
            if role in by_role:
                inputs[role] = _benchmark_reference(by_role[role])
        benchmark_capture = {
            "capture_id": capture_id,
            "split": split,
            "source_status": capture["source_status"],
            "groups": member_by_id[capture_id]["groups"],
            "inputs": inputs,
            "teacher_truth": _benchmark_reference(by_role["teacher_truth"]),
            "rights_record": _benchmark_reference(by_role["rights_record"]),
        }
        if "tracker_labels" in by_role:
            benchmark_capture["tracker_labels"] = _benchmark_reference(by_role["tracker_labels"])
        benchmark_captures.append(benchmark_capture)
        split_ids[split].append(capture_id)
        teacher_artifacts.append(
            {
                "capture_id": capture_id,
                **benchmark_capture["teacher_truth"],
            }
        )

    empty_splits = [split for split, capture_ids in split_ids.items() if not capture_ids]
    if empty_splits:
        raise PublicDatasetError(
            f"benchmark_v0 must include every fixed split; empty={empty_splits}"
        )

    public_manifest = {
        "schema": "cubed-core/benchmark-v0-manifest",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_version": "0",
        "release_scope": "public-benchmark",
        "evidence_scope": "development-benchmark-not-generalization-evidence",
        "collection_license": _benchmark_collection_license(
            collection_license,
            selected_ids,
        ),
        "capture_count": len(benchmark_captures),
        "split_policy": plan["split_policy"],
        "limitations": plan["limitations"],
        "captures": benchmark_captures,
    }
    _validate_schema(
        public_manifest,
        BENCHMARK_MANIFEST_SCHEMA,
        label="generated Benchmark v0 public manifest",
    )
    public_manifest_bytes = _json_bytes(public_manifest)
    public_manifest_sha256 = hashlib.sha256(public_manifest_bytes).hexdigest()

    splits_manifest = {
        "schema": "cubed-core/public-dataset-splits",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_version": "0",
        "benchmark_manifest_sha256": public_manifest_sha256,
        "unit": "capture",
        "splits": split_ids,
    }
    _validate_schema(
        splits_manifest,
        BENCHMARK_SPLITS_SCHEMA,
        label="generated Benchmark v0 splits manifest",
    )
    teacher_manifest = {
        "schema": "cubed-core/public-dataset-teacher-manifest",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_version": "0",
        "benchmark_manifest_sha256": public_manifest_sha256,
        "capture_count": len(teacher_artifacts),
        "artifacts": teacher_artifacts,
    }
    _validate_schema(
        teacher_manifest,
        BENCHMARK_TEACHER_MANIFEST_SCHEMA,
        label="generated Benchmark v0 teacher manifest",
    )

    generated = {
        BENCHMARK_PATHS["public_manifest"]: public_manifest_bytes,
        BENCHMARK_PATHS["splits_manifest"]: _json_bytes(splits_manifest),
        BENCHMARK_PATHS["teacher_manifest"]: _json_bytes(teacher_manifest),
    }
    binding = {
        key: {
            "path": BENCHMARK_PATHS[key].as_posix(),
            "bytes": len(generated[BENCHMARK_PATHS[key]]),
            "sha256": hashlib.sha256(generated[BENCHMARK_PATHS[key]]).hexdigest(),
        }
        for key in BENCHMARK_PATHS
    }
    return binding, generated


def _readiness_report(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    artifact_count = sum(len(capture["artifacts"]) for capture in manifest["captures"])
    artifact_bytes = sum(
        artifact["bytes"] for capture in manifest["captures"] for artifact in capture["artifacts"]
    )
    return {
        "schema": READINESS_KIND,
        "schema_version": 1,
        "ready": True,
        "dataset_id": manifest["dataset_id"],
        "release_id": manifest["release_id"],
        "evidence_scope": EVIDENCE_SCOPE,
        "capture_count": manifest["capture_count"],
        "artifact_count": artifact_count,
        "capture_artifact_bytes": artifact_bytes,
        "manifest_sha256": manifest_sha256,
        "benchmark_v0_included": manifest["benchmark_v0"] is not None,
        "checks": [
            {"check": "explicit-source-allowlist", "status": "pass"},
            {"check": "artifact-size-and-sha256", "status": "pass"},
            {"check": "media-cadence-resolution-and-audio", "status": "pass"},
            {"check": "provenance-privacy-and-rights", "status": "pass"},
            {"check": "schema-and-cross-artifact-bindings", "status": "pass"},
            {"check": "exact-public-file-inventory", "status": "pass"},
        ],
    }


def _checksum_bytes(root: Path, paths: Sequence[PurePosixPath]) -> bytes:
    lines = []
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        if relative == PurePosixPath("SHA256SUMS"):
            continue
        path = _safe_output_file(root, relative, label="checksummed file")
        lines.append(f"{_sha256_file(path)}  {relative.as_posix()}\n")
    return "".join(lines).encode("utf-8")


def _write_bytes(root: Path, relative: PurePosixPath, payload: bytes) -> None:
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _copy_source(root: Path, relative: PurePosixPath, source: Path) -> None:
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PublicDatasetError(f"refusing to overwrite staged artifact: {relative}")
    shutil.copyfile(source, destination)


def _validate_plan_runtime(
    plan: Mapping[str, Any],
    *,
    probe: ProbeFunction,
    timeout_seconds: int,
) -> tuple[
    dict[str, Any],
    dict[PurePosixPath, Path],
    dict[PurePosixPath, bytes],
    Path,
    Path,
]:
    capture_ids: set[str] = set()
    public_paths: list[PurePosixPath] = []
    source_by_public_path: dict[PurePosixPath, Path] = {}
    video_hashes: dict[str, tuple[str, str]] = {}
    manifest_captures: list[dict[str, Any]] = []

    card_path = _verify_expected_source(plan["dataset_card"], label="dataset_card")
    license_path = _verify_expected_source(plan["dataset_license"], label="dataset_license")
    _reject_private_text(card_path.read_bytes(), label="dataset_card")
    _reject_private_text(license_path.read_bytes(), label="dataset_license")
    _validate_dataset_card(
        card_path.read_bytes(),
        collection_license=plan["collection_license"]["license_id"],
    )
    _validate_dataset_license(
        license_path.read_bytes(),
        collection_license=plan["collection_license"]["license_id"],
        collection_attribution=plan["collection_license"]["attribution"],
    )

    for capture_index, capture in enumerate(plan["captures"]):
        capture_id = capture["capture_id"]
        if capture_id in capture_ids:
            raise PublicDatasetError(f"duplicate capture_id: {capture_id}")
        capture_ids.add(capture_id)
        roles: set[str] = set()
        artifact_ids: set[str] = set()
        manifest_artifacts: list[dict[str, Any]] = []
        json_documents: dict[str, dict[str, Any]] = {}
        for artifact_index, artifact in enumerate(capture["artifacts"]):
            label = (
                f"captures[{capture_index}].artifacts[{artifact_index}] "
                f"({capture_id}/{artifact['role']})"
            )
            role = artifact["role"]
            if role in roles:
                raise PublicDatasetError(f"capture {capture_id}: duplicate artifact role {role}")
            roles.add(role)
            artifact_id = artifact["artifact_id"]
            if artifact_id in artifact_ids:
                raise PublicDatasetError(
                    f"capture {capture_id}: duplicate artifact_id {artifact_id}"
                )
            artifact_ids.add(artifact_id)
            _validate_role_contract(artifact, label=label)
            public_path = _public_path(artifact["public_path"], label=f"{label}.public_path")
            source_path = _verify_expected_source(artifact, label=label)
            public_paths.append(public_path)
            source_by_public_path[public_path] = source_path
            media: Mapping[str, Any] | None = None
            if role in VIDEO_ROLES:
                existing = video_hashes.get(artifact["sha256"])
                if existing is not None:
                    raise PublicDatasetError(
                        f"duplicate video bytes: {capture_id}/{role} matches "
                        f"{existing[0]}/{existing[1]}"
                    )
                video_hashes[artifact["sha256"]] = (capture_id, role)
                media = _validate_video_facts(
                    source_path,
                    role,
                    artifact["bytes"],
                    probe=probe,
                    timeout_seconds=timeout_seconds,
                )
            else:
                json_documents[role] = _verify_json_artifact(
                    source_path,
                    artifact,
                    capture_id=capture_id,
                    label=label,
                )
            manifest_artifacts.append(_artifact_reference(artifact, path=public_path, media=media))
        missing_roles = REQUIRED_CAPTURE_ROLES - roles
        if missing_roles:
            raise PublicDatasetError(
                f"capture {capture_id}: missing required roles {sorted(missing_roles)}"
            )
        manifest_artifacts.sort(key=lambda item: (item["role"], item["path"]))
        rights_artifact = next(
            item for item in manifest_artifacts if item["role"] == "rights_record"
        )
        _validate_capture_cross_bindings(
            capture_id,
            manifest_artifacts,
            json_documents,
        )
        _validate_rights_binding(
            json_documents["rights_record"],
            manifest_artifacts,
            capture_id=capture_id,
            schema_file=rights_artifact["schema_file"],
        )
        manifest_captures.append(
            {
                "capture_id": capture_id,
                "source_status": capture["source_status"],
                "provenance_status": capture["provenance_status"],
                "split": capture["split"],
                "artifacts": manifest_artifacts,
            }
        )
    _validate_path_set(public_paths)
    manifest_captures.sort(key=lambda item: item["capture_id"])
    manifest: dict[str, Any] = {
        "schema": CORPUS_MANIFEST_KIND,
        "schema_version": 1,
        "dataset_id": plan["dataset_id"],
        "release_id": plan["release_id"],
        "description": plan["description"],
        "release_scope": "public-corpus",
        "evidence_scope": EVIDENCE_SCOPE,
        "collection_license": plan["collection_license"],
        "capture_count": len(manifest_captures),
        "limitations": plan["limitations"],
        "captures": manifest_captures,
        "benchmark_v0": None,
    }
    generated_files: dict[PurePosixPath, bytes] = {}
    benchmark_plan = plan.get("benchmark_v0")
    if benchmark_plan is not None:
        manifest["benchmark_v0"], generated_files = _build_benchmark_documents(
            benchmark_plan,
            manifest_captures,
            plan["collection_license"],
        )
    _validate_schema(manifest, CORPUS_MANIFEST_SCHEMA, label="generated corpus manifest")
    _validate_collection_license(manifest)
    return (
        manifest,
        source_by_public_path,
        generated_files,
        card_path,
        license_path,
    )


def build_public_dataset(
    plan_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    probe: ProbeFunction = probe_video,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Build one generic public corpus candidate atomically."""

    if timeout_seconds <= 0:
        raise PublicDatasetError("ffprobe timeout must be positive")
    plan_file = _require_private_build_location(Path(plan_path), label="build plan path")
    if not plan_file.exists() or not plan_file.is_file():
        raise PublicDatasetError(f"build plan does not exist: {plan_file}")
    plan = _read_json(plan_file, label="build plan")
    _validate_schema(plan, BUILD_PLAN_SCHEMA, label="build plan")

    output = _require_private_build_location(Path(output_dir), label="output directory")
    if output.exists() or output.is_symlink():
        raise PublicDatasetError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(output.parent.absolute()):
        raise PublicDatasetError("output directory parent must not traverse a symlink")

    manifest, sources, generated_files, card_path, license_path = _validate_plan_runtime(
        plan,
        probe=probe,
        timeout_seconds=timeout_seconds,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_bytes(
            staging,
            PurePosixPath(".gitattributes"),
            HF_GITATTRIBUTES,
        )
        _copy_source(staging, PurePosixPath("README.md"), card_path)
        _copy_source(
            staging,
            PurePosixPath("LICENSES/DATASET.md"),
            license_path,
        )
        _verify_binding(
            staging,
            {
                "path": "README.md",
                "bytes": plan["dataset_card"]["bytes"],
                "sha256": plan["dataset_card"]["sha256"],
            },
            label="staged dataset card",
        )
        _verify_binding(
            staging,
            {
                "path": "LICENSES/DATASET.md",
                "bytes": plan["dataset_license"]["bytes"],
                "sha256": plan["dataset_license"]["sha256"],
            },
            label="staged dataset license",
        )
        for relative in sorted(sources, key=lambda item: item.as_posix()):
            _copy_source(staging, relative, sources[relative])
        for relative in sorted(generated_files, key=lambda item: item.as_posix()):
            _write_bytes(staging, relative, generated_files[relative])

        manifest_bytes = _json_bytes(manifest)
        _write_bytes(
            staging,
            PurePosixPath("dataset/manifest.json"),
            manifest_bytes,
        )
        _write_bytes(
            staging,
            PurePosixPath("metadata.jsonl"),
            _jsonl_bytes(_metadata_rows(manifest)),
        )
        report = _readiness_report(
            manifest,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
        _validate_schema(report, READINESS_SCHEMA, label="generated readiness report")
        _write_bytes(
            staging,
            PurePosixPath("readiness-report.json"),
            _json_bytes(report),
        )
        inventory = sorted(
            ROOT_CONTRACT_PATHS | set(sources) | set(generated_files),
            key=lambda item: item.as_posix(),
        )
        _write_bytes(
            staging,
            PurePosixPath("SHA256SUMS"),
            _checksum_bytes(staging, inventory),
        )
        validated = validate_public_dataset(
            staging,
            probe=probe,
            timeout_seconds=timeout_seconds,
        )
        if output.exists() or output.is_symlink():
            raise PublicDatasetError(f"refusing to overwrite output directory: {output}")
        os.replace(staging, output)
        return validated
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _actual_tree(root: Path) -> tuple[set[PurePosixPath], set[PurePosixPath]]:
    files: set[PurePosixPath] = set()
    directories: set[PurePosixPath] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_symlink():
                raise PublicDatasetError(f"dataset contains a symlink directory: {relative}")
            directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_symlink() or not path.is_file():
                raise PublicDatasetError(
                    f"dataset contains a non-regular or symlink file: {relative}"
                )
            files.add(relative)
    return files, directories


def _expected_directories(paths: set[PurePosixPath]) -> set[PurePosixPath]:
    result: set[PurePosixPath] = set()
    for path in paths:
        parent = path.parent
        while parent != PurePosixPath("."):
            result.add(parent)
            parent = parent.parent
    return result


def _verify_binding(
    root: Path,
    reference: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    relative = _public_path(reference["path"], label=f"{label}.path")
    path = _safe_output_file(root, relative, label=label)
    actual_bytes = path.stat().st_size
    if actual_bytes != reference["bytes"]:
        raise PublicDatasetError(
            f"{label}: byte count mismatch for {relative}: "
            f"manifest={reference['bytes']} actual={actual_bytes}"
        )
    actual_hash = _sha256_file(path)
    if actual_hash != reference["sha256"]:
        raise PublicDatasetError(
            f"{label}: SHA-256 mismatch for {relative}: "
            f"manifest={reference['sha256']} actual={actual_hash}"
        )
    return path


def _parse_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise PublicDatasetError(f"cannot read metadata.jsonl: {exc}") from exc
    if not lines:
        raise PublicDatasetError("metadata.jsonl must contain at least one row")
    return [
        _parse_json(line, label=f"metadata.jsonl line {index}")
        for index, line in enumerate(lines, start=1)
    ]


def _validate_checksum_file(
    root: Path,
    expected_paths: set[PurePosixPath],
) -> None:
    checksum_path = _safe_output_file(root, PurePosixPath("SHA256SUMS"), label="checksum inventory")
    _reject_private_text(checksum_path.read_bytes(), label="SHA256SUMS")
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PublicDatasetError(f"cannot read SHA256SUMS: {exc}") from exc
    expected_without_self = expected_paths - {PurePosixPath("SHA256SUMS")}
    entries: dict[PurePosixPath, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None:
            raise PublicDatasetError(f"SHA256SUMS line {line_number} is malformed")
        relative = _public_path(match.group(2), label=f"SHA256SUMS line {line_number}")
        if relative == PurePosixPath("SHA256SUMS"):
            raise PublicDatasetError("SHA256SUMS must not hash itself")
        if relative in entries:
            raise PublicDatasetError(f"SHA256SUMS has duplicate path {relative}")
        entries[relative] = match.group(1)
    if list(entries) != sorted(entries, key=lambda item: item.as_posix()):
        raise PublicDatasetError("SHA256SUMS entries must be path-sorted")
    if set(entries) != expected_without_self:
        missing = sorted(
            (expected_without_self - set(entries)),
            key=lambda item: item.as_posix(),
        )
        extra = sorted(
            (set(entries) - expected_without_self),
            key=lambda item: item.as_posix(),
        )
        raise PublicDatasetError(f"SHA256SUMS inventory mismatch: missing={missing}, extra={extra}")
    for relative, expected_hash in entries.items():
        path = _safe_output_file(root, relative, label="checksummed file")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise PublicDatasetError(
                f"SHA256SUMS mismatch for {relative}: expected={expected_hash} actual={actual_hash}"
            )


def _validate_benchmark_binding(
    root: Path,
    benchmark: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    collection_license: Mapping[str, Any],
) -> set[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    documents: dict[str, dict[str, Any]] = {}
    schema_names = {
        "public_manifest": BENCHMARK_MANIFEST_SCHEMA,
        "splits_manifest": BENCHMARK_SPLITS_SCHEMA,
        "teacher_manifest": BENCHMARK_TEACHER_MANIFEST_SCHEMA,
    }
    for key, expected_path in BENCHMARK_PATHS.items():
        reference = benchmark[key]
        actual_path = _public_path(reference["path"], label=f"benchmark_v0.{key}.path")
        if actual_path != expected_path:
            raise PublicDatasetError(
                f"benchmark_v0.{key}.path must be {expected_path}, got {actual_path}"
            )
        path = _verify_binding(root, reference, label=f"benchmark_v0.{key}")
        document = _read_json(path, label=f"benchmark_v0.{key}")
        _reject_private_text(path.read_bytes(), label=f"benchmark_v0.{key}")
        _validate_schema(
            document,
            schema_names[key],
            label=f"benchmark_v0.{key}",
        )
        paths.add(actual_path)
        documents[key] = document

    public_manifest = documents["public_manifest"]
    manifest_hash = benchmark["public_manifest"]["sha256"]
    split_manifest = documents["splits_manifest"]
    teacher_manifest = documents["teacher_manifest"]
    for label, document in (
        ("splits_manifest", split_manifest),
        ("teacher_manifest", teacher_manifest),
    ):
        if document["benchmark_manifest_sha256"] != manifest_hash:
            raise PublicDatasetError(f"benchmark_v0.{label} is bound to another benchmark manifest")
    if (
        split_manifest["benchmark_id"] != public_manifest["benchmark_id"]
        or teacher_manifest["benchmark_id"] != public_manifest["benchmark_id"]
    ):
        raise PublicDatasetError("Benchmark v0 inventory benchmark_id values disagree")

    if public_manifest["captures"] != sorted(
        public_manifest["captures"],
        key=lambda item: item["capture_id"],
    ):
        raise PublicDatasetError("Benchmark v0 public manifest captures must be capture_id-sorted")
    benchmark_captures = {capture["capture_id"]: capture for capture in public_manifest["captures"]}
    generic_captures = {capture["capture_id"]: capture for capture in captures}
    if not set(benchmark_captures) <= set(generic_captures):
        raise PublicDatasetError(
            "Benchmark v0 contains captures absent from the generic corpus manifest"
        )
    selected_ids = set(benchmark_captures)
    if public_manifest["collection_license"] != _benchmark_collection_license(
        collection_license,
        selected_ids,
    ):
        raise PublicDatasetError(
            "Benchmark v0 collection license does not match the selected corpus captures"
        )

    expected_splits: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    expected_teacher: list[dict[str, Any]] = []
    for capture_id, benchmark_capture in benchmark_captures.items():
        generic_capture = generic_captures[capture_id]
        if (
            benchmark_capture["source_status"] != generic_capture["source_status"]
            or benchmark_capture["split"] != generic_capture["split"]
        ):
            raise PublicDatasetError(f"Benchmark v0 capture metadata disagrees for {capture_id}")
        generic_roles = _artifact_roles(generic_capture["artifacts"])
        missing_roles = BENCHMARK_REQUIRED_ROLES - set(generic_roles)
        if missing_roles:
            raise PublicDatasetError(
                f"Benchmark v0 generic capture {capture_id} is missing roles "
                f"{sorted(missing_roles)}"
            )
        expected_inputs = {
            role: _benchmark_reference(generic_roles[role]) for role in BENCHMARK_INPUT_ROLES
        }
        for role in ("high_speed_original", "video_derivation"):
            if role in generic_roles:
                expected_inputs[role] = _benchmark_reference(generic_roles[role])
        if benchmark_capture["inputs"] != expected_inputs:
            raise PublicDatasetError(
                f"Benchmark v0 input references disagree for capture {capture_id}"
            )
        for role, field in (
            ("teacher_truth", "teacher_truth"),
            ("rights_record", "rights_record"),
        ):
            if benchmark_capture[field] != _benchmark_reference(generic_roles[role]):
                raise PublicDatasetError(
                    f"Benchmark v0 {field} reference disagrees for capture {capture_id}"
                )
        expected_labels = (
            _benchmark_reference(generic_roles["tracker_labels"])
            if "tracker_labels" in generic_roles
            else None
        )
        if benchmark_capture.get("tracker_labels") != expected_labels:
            raise PublicDatasetError(
                f"Benchmark v0 tracker_labels reference disagrees for capture {capture_id}"
            )
        expected_splits[benchmark_capture["split"]].append(capture_id)
        expected_teacher.append(
            {
                "capture_id": capture_id,
                **benchmark_capture["teacher_truth"],
            }
        )

    if split_manifest["splits"] != expected_splits:
        raise PublicDatasetError(
            "Benchmark v0 split inventory does not exactly match the public manifest"
        )
    if teacher_manifest["artifacts"] != expected_teacher or teacher_manifest[
        "capture_count"
    ] != len(expected_teacher):
        raise PublicDatasetError(
            "Benchmark v0 teacher inventory does not exactly match the public manifest"
        )

    try:
        from cubed_core.benchmark_v0 import BenchmarkV0Error, validate_bundle
    except ImportError as exc:
        raise PublicDatasetError(f"Benchmark v0 validator is unavailable: {exc}") from exc
    try:
        validate_bundle(
            root / "public-inputs/manifest.json",
            bundle_root=root / "public-inputs",
            teacher_root=root / "teacher",
            schema_dir=schema_directory(),
        )
    except BenchmarkV0Error as exc:
        raise PublicDatasetError(f"Benchmark v0 bundle validation failed: {exc}") from exc
    return paths


def validate_public_dataset(
    dataset_root: str | os.PathLike[str],
    *,
    probe: ProbeFunction = probe_video,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Validate one complete local public corpus without making network calls."""

    if timeout_seconds <= 0:
        raise PublicDatasetError("ffprobe timeout must be positive")
    root_value = Path(dataset_root)
    if root_value.is_symlink():
        raise PublicDatasetError("dataset root must not be a symlink")
    try:
        root = root_value.resolve(strict=True)
    except OSError as exc:
        raise PublicDatasetError(f"dataset root does not exist: {root_value}") from exc
    if not root.is_dir():
        raise PublicDatasetError(f"dataset root is not a directory: {root}")

    manifest_path = _safe_output_file(
        root,
        PurePosixPath("dataset/manifest.json"),
        label="corpus manifest",
    )
    manifest_raw = manifest_path.read_bytes()
    _reject_private_text(manifest_raw, label="dataset/manifest.json")
    manifest = _parse_json(manifest_raw, label="dataset/manifest.json")
    _validate_schema(manifest, CORPUS_MANIFEST_SCHEMA, label="dataset/manifest.json")
    _validate_collection_license(manifest)
    if manifest["capture_count"] != len(manifest["captures"]):
        raise PublicDatasetError("corpus manifest capture_count does not match captures")
    if manifest["captures"] != sorted(manifest["captures"], key=lambda item: item["capture_id"]):
        raise PublicDatasetError("corpus manifest captures must be capture_id-sorted")

    capture_ids: set[str] = set()
    public_paths: list[PurePosixPath] = []
    video_hashes: dict[str, tuple[str, str]] = {}
    for capture in manifest["captures"]:
        capture_id = capture["capture_id"]
        if capture_id in capture_ids:
            raise PublicDatasetError(f"duplicate capture_id: {capture_id}")
        capture_ids.add(capture_id)
        artifacts = capture["artifacts"]
        if artifacts != sorted(artifacts, key=lambda item: (item["role"], item["path"])):
            raise PublicDatasetError(f"capture {capture_id}: artifacts must be role/path-sorted")
        roles: set[str] = set()
        artifact_ids: set[str] = set()
        json_documents: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            role = artifact["role"]
            if role in roles:
                raise PublicDatasetError(f"capture {capture_id}: duplicate artifact role {role}")
            roles.add(role)
            if artifact["artifact_id"] in artifact_ids:
                raise PublicDatasetError(
                    f"capture {capture_id}: duplicate artifact_id {artifact['artifact_id']}"
                )
            artifact_ids.add(artifact["artifact_id"])
            _validate_role_contract(artifact, label=f"capture {capture_id}/{role}")
            relative = _public_path(artifact["path"], label=f"capture {capture_id}/{role}.path")
            public_paths.append(relative)
            path = _verify_binding(
                root,
                artifact,
                label=f"capture {capture_id}/{role}",
            )
            if role in VIDEO_ROLES:
                existing = video_hashes.get(artifact["sha256"])
                if existing is not None:
                    raise PublicDatasetError(
                        f"duplicate video bytes: {capture_id}/{role} matches "
                        f"{existing[0]}/{existing[1]}"
                    )
                video_hashes[artifact["sha256"]] = (capture_id, role)
                actual_media = _validate_video_facts(
                    path,
                    role,
                    artifact["bytes"],
                    probe=probe,
                    timeout_seconds=timeout_seconds,
                )
                if actual_media != artifact["media"]:
                    raise PublicDatasetError(
                        f"capture {capture_id}/{role}: ffprobe facts do not match manifest"
                    )
            else:
                if artifact["media"] is not None:
                    raise PublicDatasetError(
                        f"capture {capture_id}/{role}: non-video media facts must be null"
                    )
                json_documents[role] = _verify_json_artifact(
                    path,
                    artifact,
                    capture_id=capture_id,
                    label=f"capture {capture_id}/{role}",
                )
        missing_roles = REQUIRED_CAPTURE_ROLES - roles
        if missing_roles:
            raise PublicDatasetError(
                f"capture {capture_id}: missing required roles {sorted(missing_roles)}"
            )
        rights_artifact = next(item for item in artifacts if item["role"] == "rights_record")
        _validate_capture_cross_bindings(
            capture_id,
            artifacts,
            json_documents,
        )
        _validate_rights_binding(
            json_documents["rights_record"],
            artifacts,
            capture_id=capture_id,
            schema_file=rights_artifact["schema_file"],
        )
    _validate_path_set(public_paths)

    expected_paths = set(ROOT_CONTRACT_PATHS) | set(public_paths)
    if manifest["benchmark_v0"] is not None:
        expected_paths |= _validate_benchmark_binding(
            root,
            manifest["benchmark_v0"],
            manifest["captures"],
            manifest["collection_license"],
        )
    actual_files, actual_directories = _actual_tree(root)
    expected_directories = _expected_directories(expected_paths)
    if actual_files != expected_paths:
        missing = sorted(expected_paths - actual_files, key=lambda item: item.as_posix())
        extra = sorted(actual_files - expected_paths, key=lambda item: item.as_posix())
        raise PublicDatasetError(
            f"dataset file inventory mismatch: missing={missing}, extra={extra}"
        )
    if actual_directories != expected_directories:
        missing = sorted(
            expected_directories - actual_directories,
            key=lambda item: item.as_posix(),
        )
        extra = sorted(
            actual_directories - expected_directories,
            key=lambda item: item.as_posix(),
        )
        raise PublicDatasetError(
            f"dataset directory inventory mismatch: missing={missing}, extra={extra}"
        )

    for relative in (
        PurePosixPath("README.md"),
        PurePosixPath("LICENSES/DATASET.md"),
    ):
        path = _safe_output_file(root, relative, label="public text")
        if path.stat().st_size == 0:
            raise PublicDatasetError(f"{relative} must not be empty")
        _reject_private_text(path.read_bytes(), label=relative.as_posix())
    _validate_dataset_card(
        (root / "README.md").read_bytes(),
        collection_license=manifest["collection_license"]["license_id"],
    )
    _validate_dataset_license(
        (root / "LICENSES/DATASET.md").read_bytes(),
        collection_license=manifest["collection_license"]["license_id"],
        collection_attribution=manifest["collection_license"]["attribution"],
    )
    attributes_path = _safe_output_file(
        root,
        PurePosixPath(".gitattributes"),
        label="Hugging Face Git attributes",
    )
    if attributes_path.read_bytes() != HF_GITATTRIBUTES:
        raise PublicDatasetError(
            ".gitattributes does not match the deterministic public video policy"
        )

    metadata_path = _safe_output_file(
        root, PurePosixPath("metadata.jsonl"), label="viewer metadata"
    )
    _reject_private_text(metadata_path.read_bytes(), label="metadata.jsonl")
    actual_rows = _parse_metadata_jsonl(metadata_path)
    expected_rows = _metadata_rows(manifest)
    for index, row in enumerate(actual_rows, start=1):
        _validate_schema(
            row,
            METADATA_ROW_SCHEMA,
            label=f"metadata.jsonl line {index}",
        )
    if actual_rows != expected_rows:
        raise PublicDatasetError("metadata.jsonl does not exactly match the corpus manifest")

    report_path = _safe_output_file(
        root, PurePosixPath("readiness-report.json"), label="readiness report"
    )
    report_raw = report_path.read_bytes()
    _reject_private_text(report_raw, label="readiness-report.json")
    actual_report = _parse_json(report_raw, label="readiness-report.json")
    _validate_schema(
        actual_report,
        READINESS_SCHEMA,
        label="readiness-report.json",
    )
    expected_report = _readiness_report(
        manifest,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    if actual_report != expected_report:
        raise PublicDatasetError(
            "readiness-report.json does not exactly match the validated dataset"
        )
    _validate_checksum_file(root, expected_paths)
    return actual_report
