from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

from send2trash import send2trash

from . import __version__
from .capture_derivative import (
    CaptureDerivativeError,
    create_240_to_120_video,
)
from .color_calibration import (
    COLOR_CALIBRATION_SCHEMA,
    COLOR_CENTROIDS_SCHEMA,
    COLOR_ORDER,
    ColorCalibrationError,
    ColorCentroidsError,
    build_imported_centroids_document,
    validate_color_calibration,
    validate_color_centroids,
)
from .media import (
    VIDEO_EXTENSIONS,
    classify_frame_rate,
    is_native_240_capture_frame_rate,
    is_standard_capture_resolution,
    probe_video,
)


class WorkspaceError(ValueError):
    pass


CALIBRATION_DISPLAY_NAME_MAX_CHARS = 200
_USE_DIRECTORY_FDS = os.name != "nt"
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_FACE_MOVE_PATTERN = re.compile(r"[UDLRFB](?:'|2)?")
_OUT_OF_RANGE_FPS_WARNING = (
    "Measured cadence is outside the recommended 110–121 fps profile. "
    "Decode is allowed, but timing and reconstruction may be less reliable."
)
_LOW_RESOLUTION_WARNING = (
    "Encoded short edge is below the recommended 1080-pixel profile. "
    "Decode is allowed, but face reads may be less reliable."
)


@dataclass(frozen=True, slots=True)
class DecodeCaptureArtifacts:
    """Server-resolved, submission-consistent inputs for one decode attempt."""

    capture_id: str
    receipt: dict[str, Any]
    receipt_json: bytes
    calibration_json: bytes
    scramble: str


def _entry_name(value: str) -> str:
    if not value or Path(value).name != value or "/" in value or "\\" in value:
        raise WorkspaceError("workspace file name is invalid")
    return value


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    """Return whether an entry is a symlink or another Windows reparse point."""

    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _path_is_reparse_point(path: Path) -> bool:
    try:
        return _is_reparse_point(path.lstat())
    except FileNotFoundError:
        return False


def _open_directory_fd(directory: Path, *, description: str) -> int | Path:
    """Open a directory safely, with a checked path fallback for Windows.

    POSIX keeps descriptor-relative operations and ``O_NOFOLLOW``. Native
    Windows cannot open directories through ``os.open`` or use ``dir_fd`` for
    these operations, so it receives an absolute path only after the directory
    has been lstat'd and rejected if it is any kind of reparse point.
    """

    if not _USE_DIRECTORY_FDS:
        directory = Path(os.path.abspath(directory))
        try:
            directory_stat = directory.lstat()
        except OSError as exc:
            raise WorkspaceError(f"{description} directory is unavailable") from exc
        if _is_reparse_point(directory_stat) or not stat.S_ISDIR(directory_stat.st_mode):
            raise WorkspaceError(f"{description} directory is unavailable")
        return directory

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(directory, flags)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise WorkspaceError(f"{description} directory is unavailable")
        return directory_fd
    except WorkspaceError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    except OSError as exc:
        raise WorkspaceError(f"{description} directory is unavailable") from exc


def _close_directory_fd(directory_fd: int | Path) -> None:
    if isinstance(directory_fd, int):
        os.close(directory_fd)


def _entry_path(directory_fd: int | Path, name: str) -> str | Path:
    return name if isinstance(directory_fd, int) else directory_fd / name


def _entry_stat(directory_fd: int | Path, name: str) -> os.stat_result | None:
    try:
        if isinstance(directory_fd, int):
            return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        return (_entry_path(directory_fd, name)).lstat()
    except FileNotFoundError:
        return None


def _open_entry(
    directory_fd: int | Path,
    name: str,
    flags: int,
    mode: int | None = None,
) -> int:
    if isinstance(directory_fd, int):
        if mode is None:
            return os.open(name, flags, dir_fd=directory_fd)
        return os.open(name, flags, mode, dir_fd=directory_fd)
    path = _entry_path(directory_fd, name)
    if mode is None:
        return os.open(path, flags)
    return os.open(path, flags, mode)


def _replace_entry(
    source_directory_fd: int | Path,
    source_name: str,
    target_directory_fd: int | Path,
    target_name: str,
) -> None:
    if isinstance(source_directory_fd, int) and isinstance(target_directory_fd, int):
        os.replace(
            source_name,
            target_name,
            src_dir_fd=source_directory_fd,
            dst_dir_fd=target_directory_fd,
        )
        return
    if isinstance(source_directory_fd, int) or isinstance(target_directory_fd, int):
        raise RuntimeError("directory operation modes may not be mixed")
    os.replace(
        _entry_path(source_directory_fd, source_name),
        _entry_path(target_directory_fd, target_name),
    )


def _mkdir_entry(directory_fd: int | Path, name: str, *, mode: int) -> None:
    if isinstance(directory_fd, int):
        os.mkdir(name, mode=mode, dir_fd=directory_fd)
        return
    os.mkdir(_entry_path(directory_fd, name), mode=mode)


def _unlink_directory_entry(directory_fd: int | Path, name: str) -> None:
    if isinstance(directory_fd, int):
        os.unlink(name, dir_fd=directory_fd)
        return
    os.unlink(_entry_path(directory_fd, name))


def _rmdir_entry(directory_fd: int | Path, name: str) -> None:
    if isinstance(directory_fd, int):
        os.rmdir(name, dir_fd=directory_fd)
        return
    os.rmdir(_entry_path(directory_fd, name))


def _list_directory(directory_fd: int | Path) -> list[str]:
    return os.listdir(directory_fd)


def _sync_directory(directory_fd: int | Path) -> None:
    if not isinstance(directory_fd, int):
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass


def _entry_exists(parent: Path, name: str, *, description: str) -> bool:
    name = _entry_name(name)
    parent_fd = _open_directory_fd(parent, description=description)
    try:
        return _entry_stat(parent_fd, name) is not None
    finally:
        _close_directory_fd(parent_fd)


def _validate_replace_target(
    directory_fd: int | Path,
    name: str,
    *,
    description: str,
    allow_symlink: bool,
) -> None:
    target_stat = _entry_stat(directory_fd, name)
    if target_stat is None:
        return
    if _is_reparse_point(target_stat):
        if not allow_symlink:
            raise WorkspaceError(f"{description} target may not be a symlink")
        return
    if not stat.S_ISREG(target_stat.st_mode):
        raise WorkspaceError(f"{description} target must be a regular file")


def _atomic_write_bytes(
    directory: Path,
    name: str,
    payload: bytes,
    *,
    description: str,
    allow_replace_symlink: bool = False,
) -> None:
    """Write one file through a unique exclusive sibling and atomic replace."""

    name = _entry_name(name)
    directory_fd = _open_directory_fd(directory, description=description)
    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        _validate_replace_target(
            directory_fd,
            name,
            description=description,
            allow_symlink=allow_replace_symlink,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        for _ in range(128):
            candidate = f".{name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_fd = _open_entry(
                    directory_fd,
                    candidate,
                    flags,
                    mode=0o600,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd is None or temporary_name is None:
            raise WorkspaceError(f"could not allocate a unique {description} temporary file")

        with os.fdopen(temporary_fd, "wb") as stream:
            temporary_fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        _validate_replace_target(
            directory_fd,
            name,
            description=description,
            allow_symlink=allow_replace_symlink,
        )
        _replace_entry(
            directory_fd,
            temporary_name,
            directory_fd,
            name,
        )
        temporary_name = None
        # Some filesystems do not support directory fsync. The file itself was
        # still flushed and fsynced before the atomic replacement.
        _sync_directory(directory_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None:
            try:
                _unlink_directory_entry(directory_fd, temporary_name)
            except FileNotFoundError:
                pass
        _close_directory_fd(directory_fd)


def _read_regular_bytes(
    directory: Path,
    name: str,
    *,
    description: str,
    maximum_bytes: int | None = None,
    missing_ok: bool = False,
) -> bytes | None:
    """Read one regular file without following a final-component symlink."""

    name = _entry_name(name)
    directory_fd = _open_directory_fd(directory, description=description)
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        expected_stat: os.stat_result | None = None
        if not isinstance(directory_fd, int):
            expected_stat = _entry_stat(directory_fd, name)
            if expected_stat is None:
                if missing_ok:
                    return None
                raise FileNotFoundError(name)
            if _is_reparse_point(expected_stat):
                raise WorkspaceError(f"{description} is unavailable")
        try:
            file_fd = _open_entry(directory_fd, name, flags)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        file_stat = os.fstat(file_fd)
        if expected_stat is not None and (
            expected_stat.st_dev,
            expected_stat.st_ino,
        ) != (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            raise WorkspaceError(f"{description} is unavailable")
        if not stat.S_ISREG(file_stat.st_mode):
            raise WorkspaceError(f"{description} must be a regular file")
        if maximum_bytes is not None and file_stat.st_size > maximum_bytes:
            raise WorkspaceError(f"{description} exceeds the safety limit")
        with os.fdopen(file_fd, "rb") as stream:
            file_fd = None
            payload = stream.read(None if maximum_bytes is None else maximum_bytes + 1)
        if maximum_bytes is not None and len(payload) > maximum_bytes:
            raise WorkspaceError(f"{description} exceeds the safety limit")
        return payload
    except OSError as exc:
        if missing_ok and isinstance(exc, FileNotFoundError):
            return None
        raise WorkspaceError(f"{description} is unavailable") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        _close_directory_fd(directory_fd)


def _unlink_entry(directory: Path, name: str, *, description: str) -> None:
    """Remove a directory entry without following it."""

    name = _entry_name(name)
    directory_fd = _open_directory_fd(directory, description=description)
    try:
        try:
            _unlink_directory_entry(directory_fd, name)
        except FileNotFoundError:
            return
        _sync_directory(directory_fd)
    finally:
        _close_directory_fd(directory_fd)


def _replace_temporary_entry(
    directory: Path,
    temporary_name: str,
    target_name: str,
    *,
    description: str,
) -> None:
    """Atomically install one exclusively created regular temporary file."""

    temporary_name = _entry_name(temporary_name)
    target_name = _entry_name(target_name)
    directory_fd = _open_directory_fd(directory, description=description)
    try:
        temporary_stat = _entry_stat(directory_fd, temporary_name)
        if (
            temporary_stat is None
            or _is_reparse_point(temporary_stat)
            or not stat.S_ISREG(temporary_stat.st_mode)
        ):
            raise WorkspaceError(f"{description} temporary file is unavailable")
        _validate_replace_target(
            directory_fd,
            target_name,
            description=description,
            allow_symlink=False,
        )
        _replace_entry(
            directory_fd,
            temporary_name,
            directory_fd,
            target_name,
        )
        _sync_directory(directory_fd)
    finally:
        _close_directory_fd(directory_fd)


def _ensure_child_directory(parent: Path, name: str, *, description: str) -> Path:
    """Create/open one child directory without following a planted symlink."""

    name = _entry_name(name)
    parent_fd = _open_directory_fd(parent, description=description)
    child_fd: int | None = None
    try:
        try:
            _mkdir_entry(parent_fd, name, mode=0o700)
        except FileExistsError:
            pass
        child_stat = _entry_stat(parent_fd, name)
        if child_stat is None or _is_reparse_point(child_stat):
            raise WorkspaceError(f"{description} directory may not be a symlink")
        if not stat.S_ISDIR(child_stat.st_mode):
            raise WorkspaceError(f"{description} directory is unavailable")
        if isinstance(parent_fd, int):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                child_fd = _open_entry(parent_fd, name, flags)
            except OSError as exc:
                raise WorkspaceError(f"{description} directory may not be a symlink") from exc
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                raise WorkspaceError(f"{description} directory is unavailable")
        return parent / name
    finally:
        if child_fd is not None:
            os.close(child_fd)
        _close_directory_fd(parent_fd)


def normalize_scramble(value: str) -> str | None:
    if not isinstance(value, str):
        raise WorkspaceError("scramble must be a string")
    if len(value) > 500:
        raise WorkspaceError("scramble must be at most 500 characters")
    tokens = value.split()
    if not tokens:
        return None
    if any(not _FACE_MOVE_PATTERN.fullmatch(token) for token in tokens):
        raise WorkspaceError("scramble must contain canonical face moves")
    return " ".join(tokens)


def _safe_original_name(value: str) -> str:
    name = Path(value or "capture.mp4").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned[:180] or "capture.mp4"


def calibration_upload_display_name(value: str | None) -> str | None:
    """Return a bounded, path-free label for an uploaded calibration file."""

    if not isinstance(value, str) or not value.strip():
        return None
    return _safe_original_name(value)


def _safe_calibration_display_name(value: str | None) -> str | None:
    """Bound one server-authored calibration label for receipt/UI display."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkspaceError("calibration display name must be a string")
    normalized = " ".join(value.split())
    return normalized[:CALIBRATION_DISPLAY_NAME_MAX_CHARS] or None


def _legacy_sampled_calibration_display_name(
    capture_dir: Path,
    receipt: dict[str, Any],
) -> str | None:
    """Recognize old desktop-sampled calibrations without mutating their receipt."""

    capture_id = receipt.get("capture_id")
    video_sha256 = receipt.get("video", {}).get("sha256")
    calibration = receipt.get("calibration")
    if (
        not isinstance(capture_id, str)
        or capture_dir.name != capture_id
        or not isinstance(video_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", video_sha256) is None
        or not isinstance(calibration, dict)
        or "display_name" in calibration
        or calibration.get("path") != "calibration.json"
    ):
        return None
    expected_sha256 = calibration.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
    ):
        return None
    try:
        payload = _read_regular_bytes(
            capture_dir,
            "calibration.json",
            description="capture calibration",
            maximum_bytes=8 * 1024**2,
            missing_ok=True,
        )
        if payload is None or hashlib.sha256(payload).hexdigest() != expected_sha256:
            return None
        document = json.loads(payload)
        validate_color_centroids(document)
    except (
        WorkspaceError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ColorCentroidsError,
    ):
        return None

    provenance = document.get("provenance")
    prefix = (
        f"workspace-video-sticker-crops capture:{capture_id} "
        f"video-sha256:{video_sha256} crops-sha256:"
    )
    if (
        not isinstance(provenance, str)
        or len(provenance) != len(prefix) + 64
        or not provenance.startswith(prefix)
        or re.fullmatch(r"[a-f0-9]{64}", provenance[len(prefix) :]) is None
    ):
        return None
    return "Sampled from this video"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_fd: int | None = None
    try:
        expected_stat = path.lstat()
        if _is_reparse_point(expected_stat):
            raise WorkspaceError("workspace file is unavailable")
        file_fd = os.open(path, flags)
        file_stat = os.fstat(file_fd)
        if (expected_stat.st_dev, expected_stat.st_ino) != (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            raise WorkspaceError("workspace file is unavailable")
        if not stat.S_ISREG(file_stat.st_mode):
            raise WorkspaceError("workspace file must be a regular file")
        with os.fdopen(file_fd, "rb") as stream:
            file_fd = None
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WorkspaceError("workspace file is unavailable") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
    return digest.hexdigest()


def _probe_frame_rate_fraction(probe: dict[str, Any]) -> Fraction | None:
    value = probe.get("fps_rational")
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", value):
        return None
    numerator_text, denominator_text = value.split("/", 1)
    if len(numerator_text) > 10 or len(denominator_text) > 10:
        return None
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if numerator > 1_000_000_000 or denominator > 1_000_000_000:
        return None
    return Fraction(numerator, denominator)


def validate_ble_session(value: Any) -> None:
    if not isinstance(value, dict):
        raise WorkspaceError("BLE teacher session must be a JSON object")
    required = {
        "schema",
        "schema_version",
        "ble_session_id",
        "capture_session_id",
        "video_recording_id",
        "started_at",
        "ended_at",
        "device",
        "clock",
        "scramble",
        "moves",
        "orientations",
        "states",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise WorkspaceError(f"BLE teacher session is missing: {', '.join(missing)}")
    if value.get("schema") != "cubed-core/ble-session" or value.get("schema_version") != 1:
        raise WorkspaceError("unsupported BLE teacher session schema")
    for field in ("ble_session_id", "capture_session_id", "started_at", "ended_at"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise WorkspaceError(f"BLE teacher {field} must be a non-empty string")
    if not isinstance(value.get("device"), dict) or not isinstance(value.get("clock"), dict):
        raise WorkspaceError("BLE teacher device and clock receipts must be objects")
    if not isinstance(value["device"].get("name"), str) or not value["device"]["name"].strip():
        raise WorkspaceError("BLE teacher device name must be a non-empty string")
    scramble = value["scramble"]
    if scramble is not None:
        try:
            normalized_scramble = normalize_scramble(scramble)
        except WorkspaceError as exc:
            raise WorkspaceError("BLE teacher scramble must contain canonical moves") from exc
        if normalized_scramble is None:
            raise WorkspaceError("BLE teacher scramble must not be blank")
    clock = value["clock"]
    if (
        clock.get("relative_timebase") != "host_performance_now"
        or clock.get("move_event_time") != "event_local_timestamp_or_host_monotonic_fallback"
        or clock.get("orientation_event_time") != "host_monotonic_receive"
        or clock.get("cross_device_alignment") != "content_affine_required"
    ):
        raise WorkspaceError("BLE teacher session has an unsupported clock contract")
    anchor = _finite_number(
        clock.get("monotonic_start_ms"),
        field="clock.monotonic_start_ms",
        nonnegative=True,
    )
    _finite_number(
        clock.get("unix_start_ms"),
        field="clock.unix_start_ms",
        nonnegative=True,
    )
    if "zero_quat" in value:
        _validate_quaternion(
            value["zero_quat"],
            field="zero_quat",
            nullable=False,
        )
    for field in ("moves", "orientations", "states"):
        if not isinstance(value.get(field), list):
            raise WorkspaceError(f"BLE teacher {field} must be an array")
    seen_sequences: set[int] = set()
    previous_move_time = -1.0
    for index, move in enumerate(value["moves"]):
        if not isinstance(move, dict):
            raise WorkspaceError(f"BLE teacher move {index} must be an object")
        required_move = {
            "sequence",
            "t_ms",
            "move",
            "event_local_timestamp_ms",
            "event_host_timestamp_ms",
            "cube_timestamp_ms",
            "serial",
            "clock_source",
        }
        if not required_move.issubset(move):
            raise WorkspaceError(f"BLE teacher move {index} is incomplete")
        if (
            not isinstance(move["sequence"], int)
            or isinstance(move["sequence"], bool)
            or move["sequence"] < 0
            or not isinstance(move["move"], str)
            or not _FACE_MOVE_PATTERN.fullmatch(move["move"])
            or not _is_valid_event_serial(move["serial"])
        ):
            raise WorkspaceError(f"BLE teacher move {index} is invalid")
        if move["sequence"] in seen_sequences:
            raise WorkspaceError("BLE teacher event sequence values must be unique")
        seen_sequences.add(move["sequence"])
        move_time = _finite_number(
            move["t_ms"],
            field=f"moves[{index}].t_ms",
            nonnegative=True,
        )
        if move_time < previous_move_time:
            raise WorkspaceError("BLE teacher move times must be monotonic")
        previous_move_time = move_time
        _validate_quaternion(
            move.get("quaternion"),
            field=f"moves[{index}].quaternion",
            nullable=True,
        )
        facelets = move.get("facelets")
        if facelets is not None and not _has_valid_facelets(facelets):
            raise WorkspaceError(f"BLE teacher move {index} facelets are invalid")
        _validate_move_clock_source(
            move,
            index=index,
            move_time=move_time,
            anchor=anchor,
            label="BLE teacher",
        )
        _finite_number(
            move["event_host_timestamp_ms"],
            field=f"moves[{index}].event_host_timestamp_ms",
            nonnegative=True,
        )
        if move["cube_timestamp_ms"] is not None:
            _finite_number(
                move["cube_timestamp_ms"],
                field=f"moves[{index}].cube_timestamp_ms",
                nonnegative=True,
            )
    previous_orientation_time = -1.0
    for index, orientation in enumerate(value["orientations"]):
        if not isinstance(orientation, dict):
            raise WorkspaceError(f"BLE teacher orientation {index} must be an object")
        sequence = orientation.get("sequence")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or sequence in seen_sequences
        ):
            raise WorkspaceError(f"BLE teacher orientation {index} sequence is invalid")
        seen_sequences.add(sequence)
        event_time = _finite_number(
            orientation.get("t_ms"),
            field=f"orientations[{index}].t_ms",
            nonnegative=True,
        )
        if event_time < previous_orientation_time:
            raise WorkspaceError("BLE teacher orientation times must be monotonic")
        previous_orientation_time = event_time
        _validate_quaternion(
            orientation.get("quaternion"),
            field=f"orientations[{index}].quaternion",
            nullable=False,
        )
        if orientation.get("clock_source") != "host_monotonic_receive":
            raise WorkspaceError(f"BLE teacher orientation {index} clock source is invalid")
        _finite_number(
            orientation.get("event_host_timestamp_ms"),
            field=f"orientations[{index}].event_host_timestamp_ms",
            nonnegative=True,
        )
    previous_state_time = -1.0
    for index, state in enumerate(value["states"]):
        if not isinstance(state, dict):
            raise WorkspaceError(f"BLE teacher state {index} must be an object")
        sequence = state.get("sequence")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or sequence in seen_sequences
        ):
            raise WorkspaceError(f"BLE teacher state {index} sequence is invalid")
        seen_sequences.add(sequence)
        event_time = _finite_number(
            state.get("t_ms"),
            field=f"states[{index}].t_ms",
            nonnegative=True,
        )
        if event_time < previous_state_time:
            raise WorkspaceError("BLE teacher state times must be monotonic")
        previous_state_time = event_time
        facelets = state.get("facelets")
        if not _has_valid_facelets(facelets):
            raise WorkspaceError(f"BLE teacher state {index} facelets are invalid")
        serial = state.get("serial")
        if not _is_valid_event_serial(serial):
            raise WorkspaceError(f"BLE teacher state {index} serial is invalid")
        _finite_number(
            state.get("event_host_timestamp_ms"),
            field=f"states[{index}].event_host_timestamp_ms",
            nonnegative=True,
        )


def _finite_number(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WorkspaceError(f"{field} must be a finite number")
    number = float(value)
    if nonnegative and number < 0:
        raise WorkspaceError(f"{field} must be nonnegative")
    return number


def normalize_camera_intrinsics(value: Any) -> dict[str, Any] | None:
    """Validate the existing native-camera `{matrix,ref_w,ref_h}` contract."""

    if value is None:
        return None
    expected = {"matrix", "ref_w", "ref_h"}
    if not isinstance(value, dict) or set(value) != expected:
        raise WorkspaceError("camera intrinsics must contain exactly matrix, ref_w, and ref_h")
    matrix = value["matrix"]
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise WorkspaceError("camera intrinsics matrix must contain exactly 3 rows")
    normalized_matrix: list[list[float]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 3:
            raise WorkspaceError(
                f"camera intrinsics matrix[{row_index}] must contain exactly 3 values"
            )
        normalized_row: list[float] = []
        for column_index, item in enumerate(row):
            number = _finite_number(
                item,
                field=f"camera intrinsics matrix[{row_index}][{column_index}]",
            )
            if abs(number) > 10_000_000:
                raise WorkspaceError("camera intrinsics matrix values are out of range")
            normalized_row.append(number)
        normalized_matrix.append(normalized_row)
    for field in ("ref_w", "ref_h"):
        dimension = value[field]
        if type(dimension) is not int or not 1 <= dimension <= 100_000:
            raise WorkspaceError(
                f"camera intrinsics {field} must be an integer between 1 and 100000"
            )
    return {
        "matrix": normalized_matrix,
        "ref_w": value["ref_w"],
        "ref_h": value["ref_h"],
    }


def parse_camera_intrinsics_form(value: str) -> dict[str, Any] | None:
    """Parse the optional multipart form field without accepting JSON extensions."""

    if not isinstance(value, str):
        raise WorkspaceError("camera intrinsics form field must be a string")
    if not value.strip():
        return None
    if len(value) > 4096:
        raise WorkspaceError("camera intrinsics form field is too large")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WorkspaceError("camera intrinsics must be valid JSON") from exc
    return normalize_camera_intrinsics(parsed)


def _validate_quaternion(value: Any, *, field: str, nullable: bool) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise WorkspaceError(f"{field} must be a four-number quaternion")
    components = [
        _finite_number(component, field=f"{field}[{index}]")
        for index, component in enumerate(value)
    ]
    norm_squared = sum(component * component for component in components)
    if not 0.90 <= norm_squared <= 1.10:
        raise WorkspaceError(f"{field} must be a unit quaternion")


def _has_valid_facelets(value: Any) -> bool:
    """Return whether ``value`` is a 54-character URFDLB facelet string."""

    return (
        isinstance(value, str)
        and len(value) == 54
        and all(value.count(symbol) == 9 for symbol in "URFDLB")
    )


def _is_valid_event_serial(value: Any) -> bool:
    """Return whether ``value`` is a valid one-byte BLE event serial."""

    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255


def _validate_move_clock_source(
    move: dict[str, Any],
    *,
    index: int,
    move_time: float,
    anchor: float,
    label: str,
) -> None:
    """Validate one move's ``clock_source``/``event_local_timestamp_ms`` pairing.

    Shared by the BLE teacher session and CubeSession v2 schemas, which apply
    the same clock contract under different error-message prefixes.
    """

    source = move.get("clock_source")
    local_timestamp = move.get("event_local_timestamp_ms")
    if source == "event_local_timestamp":
        local = _finite_number(
            local_timestamp,
            field=f"moves[{index}].event_local_timestamp_ms",
            nonnegative=True,
        )
        if abs(move_time - max(0.0, local - anchor)) > 1.0:
            raise WorkspaceError(f"{label} move {index} clock receipt disagrees")
    elif source == "host_monotonic_fallback":
        if local_timestamp is not None:
            raise WorkspaceError(f"{label} move {index} fallback clock is invalid")
    else:
        raise WorkspaceError(f"{label} move {index} clock source is invalid")


def _validate_cube_session_v2(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise WorkspaceError("teacher session must declare CubeSession schema version 2")
    required = {
        "clock",
        "recording_id",
        "capture_session_id",
        "device",
        "started_unix_ms",
        "moves",
        "orientations",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise WorkspaceError(f"CubeSession v2 is missing: {', '.join(missing)}")
    for field in ("recording_id", "capture_session_id", "device"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise WorkspaceError(f"CubeSession v2 {field} must be a non-empty string")
    _finite_number(value["started_unix_ms"], field="started_unix_ms", nonnegative=True)
    if "zero_quat" in value:
        _validate_quaternion(
            value["zero_quat"],
            field="zero_quat",
            nullable=True,
        )
    scramble = value.get("scramble")
    if scramble is not None:
        normalized_scramble = normalize_scramble(scramble)
        if normalized_scramble is None:
            raise WorkspaceError("CubeSession v2 scramble must not be blank")
    clock = value["clock"]
    if not isinstance(clock, dict):
        raise WorkspaceError("CubeSession v2 clock must be an object")
    expected_clock = {
        "schema_version": 1,
        "relative_timebase": "host_performance_now",
        "move_event_time": "event_local_timestamp_or_host_monotonic_fallback",
        "orientation_event_time": "host_monotonic_receive",
    }
    for field, expected in expected_clock.items():
        if clock.get(field) != expected:
            raise WorkspaceError(f"CubeSession v2 clock.{field} must equal {expected!r}")
    anchor = _finite_number(
        clock.get("monotonic_start_ms"),
        field="clock.monotonic_start_ms",
        nonnegative=True,
    )
    moves = value["moves"]
    orientations = value["orientations"]
    if not isinstance(moves, list) or not isinstance(orientations, list):
        raise WorkspaceError("CubeSession v2 moves and orientations must be arrays")
    previous_move_time = -1.0
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            raise WorkspaceError(f"CubeSession v2 move {index} must be an object")
        move_time = _finite_number(
            move.get("t_ms"),
            field=f"moves[{index}].t_ms",
            nonnegative=True,
        )
        if move_time < previous_move_time:
            raise WorkspaceError("CubeSession v2 move times must be monotonic")
        previous_move_time = move_time
        if not isinstance(move.get("move"), str) or not _FACE_MOVE_PATTERN.fullmatch(move["move"]):
            raise WorkspaceError(f"CubeSession v2 move {index} is not canonical")
        serial = move.get("serial")
        if not _is_valid_event_serial(serial):
            raise WorkspaceError(f"CubeSession v2 move {index} serial is invalid")
        _validate_quaternion(
            move.get("quat"),
            field=f"moves[{index}].quat",
            nullable=True,
        )
        facelets = move.get("facelets")
        if facelets is not None and not _has_valid_facelets(facelets):
            raise WorkspaceError(f"CubeSession v2 move {index} facelets are invalid")
        _validate_move_clock_source(
            move,
            index=index,
            move_time=move_time,
            anchor=anchor,
            label="CubeSession v2",
        )
        cube_timestamp = move.get("cube_timestamp_ms")
        if cube_timestamp is not None:
            _finite_number(
                cube_timestamp,
                field=f"moves[{index}].cube_timestamp_ms",
                nonnegative=True,
            )
    previous_orientation_time = -1.0
    for index, orientation in enumerate(orientations):
        if not isinstance(orientation, dict):
            raise WorkspaceError(f"CubeSession v2 orientation {index} must be an object")
        orientation_time = _finite_number(
            orientation.get("t_ms"),
            field=f"orientations[{index}].t_ms",
            nonnegative=True,
        )
        if orientation_time < previous_orientation_time:
            raise WorkspaceError("CubeSession v2 orientation times must be monotonic")
        previous_orientation_time = orientation_time
        _validate_quaternion(
            orientation.get("quat"),
            field=f"orientations[{index}].quat",
            nullable=False,
        )
        if orientation.get("clock_source") != "host_monotonic_receive":
            raise WorkspaceError(f"CubeSession v2 orientation {index} clock source is invalid")


class Workspace:
    def __init__(
        self,
        root: Path,
        *,
        max_upload_bytes: int,
    ):
        self.root = root
        self.max_upload_bytes = max_upload_bytes
        self.captures_dir = root / "captures"
        self.annotations_dir = root / "annotations"
        self.exports_dir = root / "exports"
        self._locks_guard = threading.Lock()
        self._capture_locks: dict[str, threading.Lock] = {}

    def _capture_lock(self, capture_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._capture_locks.setdefault(capture_id, threading.Lock())

    def initialize(self) -> None:
        if _path_is_reparse_point(self.root):
            raise WorkspaceError("workspace root may not be a symlink")
        if _path_is_reparse_point(self.captures_dir):
            raise WorkspaceError("workspace captures directory may not be a symlink")
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        if _path_is_reparse_point(self.annotations_dir):
            raise WorkspaceError("workspace annotations directory may not be a symlink")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        if _path_is_reparse_point(self.exports_dir):
            raise WorkspaceError("workspace exports directory may not be a symlink")
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _capture_dir(self, capture_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
            raise WorkspaceError("invalid capture id")
        capture_dir = self.captures_dir / capture_id
        captures_fd = _open_directory_fd(
            self.captures_dir,
            description="workspace captures",
        )
        try:
            capture_stat = _entry_stat(captures_fd, capture_id)
        finally:
            _close_directory_fd(captures_fd)
        if (
            capture_stat is None
            or _is_reparse_point(capture_stat)
            or not stat.S_ISDIR(capture_stat.st_mode)
        ):
            raise WorkspaceError("capture not found")
        return capture_dir

    @staticmethod
    def _read_receipt(capture_dir: Path) -> dict[str, Any]:
        try:
            payload = _read_regular_bytes(
                capture_dir,
                "capture.json",
                description="capture metadata",
                maximum_bytes=8 * 1024**2,
            )
            assert payload is not None
            value = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError("capture metadata is unavailable") from exc
        if not isinstance(value, dict):
            raise WorkspaceError("capture metadata is invalid")
        return value

    @staticmethod
    def _write_receipt(capture_dir: Path, receipt: dict[str, Any]) -> None:
        _atomic_write_bytes(
            capture_dir,
            "capture.json",
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            description="capture metadata",
        )

    @staticmethod
    def _write_checksums(capture_dir: Path, receipt: dict[str, Any]) -> None:
        entries = [
            (receipt["video"]["sha256"], Path(receipt["video"]["path"]).name),
        ]
        calibration = receipt.get("calibration")
        if isinstance(calibration, dict):
            entries.append((calibration["sha256"], calibration["path"]))
        teacher = receipt.get("teacher")
        if isinstance(teacher, dict):
            entries.append((teacher["sha256"], teacher["path"]))
        phone_imu = receipt.get("sensors", {}).get("phone_imu")
        if isinstance(phone_imu, dict):
            entries.append((phone_imu["sha256"], phone_imu["path"]))
        ble_raw = receipt.get("sensors", {}).get("ble_raw")
        if isinstance(ble_raw, dict):
            entries.append((ble_raw["sha256"], ble_raw["path"]))
        _atomic_write_bytes(
            capture_dir,
            "checksums.sha256",
            "".join(f"{digest}  {path}\n" for digest, path in entries).encode("utf-8"),
            description="capture checksums",
        )

    def _commit_existing_bundle_metadata(
        self,
        capture_dir: Path,
        receipt: dict[str, Any],
    ) -> None:
        previous_metadata = _read_regular_bytes(
            capture_dir,
            "capture.json",
            description="capture metadata",
            maximum_bytes=8 * 1024**2,
        )
        previous_checksums = _read_regular_bytes(
            capture_dir,
            "checksums.sha256",
            description="capture checksums",
            maximum_bytes=1024**2,
        )
        assert previous_metadata is not None
        assert previous_checksums is not None
        try:
            self._write_checksums(capture_dir, receipt)
            self._write_receipt(capture_dir, receipt)
        except Exception as exc:
            rollback_errors: list[Exception] = []
            for name, payload, description in (
                ("checksums.sha256", previous_checksums, "capture checksums"),
                ("capture.json", previous_metadata, "capture metadata"),
            ):
                try:
                    _atomic_write_bytes(
                        capture_dir,
                        name,
                        payload,
                        description=description,
                        allow_replace_symlink=True,
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if rollback_errors:
                raise WorkspaceError(
                    "bundle metadata update failed and rollback could not be completed"
                ) from exc
            raise

    def import_video(
        self,
        stream: BinaryIO,
        *,
        filename: str,
        source: str,
        notes: str = "",
        capture_session_id: str = "",
        scramble: str = "",
        camera_facing: str = "unknown",
        mirrored: bool = False,
        camera_intrinsics: dict[str, Any] | None = None,
        configured_fps: int | None = None,
        derivation: dict[str, Any] | None = None,
        normalized_from_sha256: str | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        original_name = _safe_original_name(filename)
        extension = Path(original_name).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise WorkspaceError(f"unsupported video extension {extension!r}; allowed: {allowed}")
        if source not in {
            "browser",
            "derived-240-to-120",
            "external-camera",
            "import",
            "native-ios",
            "qr-upload",
        }:
            raise WorkspaceError("unknown capture source")
        if source == "derived-240-to-120":
            if (
                not isinstance(derivation, dict)
                or derivation.get("schema") != "cubed-core/frame-rate-derivation"
                or derivation.get("schema_version") != 1
                or not isinstance(normalized_from_sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", normalized_from_sha256)
            ):
                raise WorkspaceError("derived capture requires a valid linkage receipt")
        elif derivation is not None or normalized_from_sha256 is not None:
            raise WorkspaceError("derivation metadata is reserved for derived captures")
        if type(mirrored) is not bool:
            raise WorkspaceError("mirrored must be a boolean")
        if source == "native-ios" and mirrored:
            raise WorkspaceError("native iOS capture evidence must be unmirrored")
        if configured_fps is not None:
            if source != "native-ios":
                raise WorkspaceError("configured_fps is reserved for native iOS captures")
            if type(configured_fps) is not int or not 60 <= configured_fps <= 240:
                raise WorkspaceError("configured_fps must be an integer from 60 through 240")
        if camera_facing not in {"front", "back", "external", "unknown"}:
            raise WorkspaceError("camera_facing must be front, back, external, or unknown")
        normalized_intrinsics = normalize_camera_intrinsics(camera_intrinsics)
        normalized_scramble = normalize_scramble(scramble)
        capture_id = secrets.token_hex(16)
        capture_dir = self.captures_dir / capture_id
        capture_dir.mkdir(mode=0o700)
        video_path = capture_dir / f"source{extension}"
        staging_fd = -1
        staging_path: Path | None = None
        digest = hashlib.sha256()
        byte_count = 0
        receipt_committed = False
        try:
            staging_fd, staging_raw = tempfile.mkstemp(
                prefix=f".source{extension}.",
                suffix=".tmp",
                dir=capture_dir,
            )
            staging_path = Path(staging_raw)
            with os.fdopen(staging_fd, "wb") as output:
                staging_fd = -1
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > self.max_upload_bytes:
                        raise WorkspaceError(
                            "video exceeds the configured "
                            f"{self.max_upload_bytes}-byte upload limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if byte_count == 0:
                raise WorkspaceError("video is empty")
            _replace_temporary_entry(
                capture_dir,
                staging_path.name,
                video_path.name,
                description="capture video",
            )
            probe = probe_video(video_path)
            actual_fps = probe.get("fps")
            fps_is_known = (
                not isinstance(actual_fps, bool)
                and isinstance(actual_fps, (int, float))
                and math.isfinite(float(actual_fps))
                and float(actual_fps) > 0
            )
            if not fps_is_known:
                fps_missing = ["video.actual_fps"]
            elif is_native_240_capture_frame_rate(actual_fps):
                fps_missing = ["video.derive-240-to-120"]
            else:
                fps_missing = []

            encoded_width = probe.get("width")
            encoded_height = probe.get("height")
            resolution_missing = (
                ["video.encoded_dimensions"]
                if (
                    type(encoded_width) is not int
                    or encoded_width <= 0
                    or type(encoded_height) is not int
                    or encoded_height <= 0
                )
                else []
            )

            if not fps_is_known:
                fps_warnings = [
                    "Frame rate is unknown; decode sealing requires measured media metadata."
                ]
            elif actual_fps < 110:
                fps_warnings = [_OUT_OF_RANGE_FPS_WARNING]
            elif is_native_240_capture_frame_rate(actual_fps):
                fps_warnings = [
                    "Measured 220–242 fps is the native-240 research regime. Preserve "
                    "the native source; Decode prepares a deterministic 120 fps derivative."
                ]
            elif actual_fps > 121:
                fps_warnings = [_OUT_OF_RANGE_FPS_WARNING]
            else:
                fps_warnings = []

            if resolution_missing:
                resolution_warnings = [
                    "Encoded dimensions are unknown; decode sealing requires measured "
                    "media metadata."
                ]
            elif not is_standard_capture_resolution(encoded_width, encoded_height):
                resolution_warnings = [_LOW_RESOLUTION_WARNING]
            else:
                resolution_warnings = []
            receipt: dict[str, Any] = {
                "schema": "cubed-core/capture-bundle",
                "schema_version": 1,
                "recording_id": capture_id,
                "capture_id": capture_id,
                "capture_session_id": capture_session_id.strip()[:200] or capture_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "state": "incomplete",
                "sealed_at": None,
                "seal_purpose": None,
                "original_filename": original_name,
                "video": {
                    "path": video_path.relative_to(self.root).as_posix(),
                    "bytes": byte_count,
                    "sha256": digest.hexdigest(),
                    "container": probe.get("container"),
                    "codec": probe.get("codec"),
                    "encoded_width": probe.get("width"),
                    "encoded_height": probe.get("height"),
                    "configured_fps": configured_fps,
                    "actual_fps": probe.get("fps"),
                    "frame_count": probe.get("frame_count"),
                    "rotation_degrees": probe.get("rotation_degrees", 0),
                    "mirrored": mirrored,
                    "time_origin": None,
                },
                "camera": {
                    "facing": camera_facing,
                    "device_model": None,
                    "camera_id": None,
                    "intrinsics": normalized_intrinsics,
                },
                "solve": {
                    "scramble": normalized_scramble,
                    "end_condition": "solved",
                    "start_frame": None,
                    "end_frame": None,
                },
                "calibration": None,
                "teacher": None,
                "sensors": {"phone_imu": None, "ble_raw": None},
                "provenance": {
                    "producer": "cubed-core",
                    "producer_version": __version__,
                    "normalized_from_sha256": normalized_from_sha256,
                },
                "probe": probe,
                "notes": notes.strip()[:2000],
                "readiness": {
                    "can_label": True,
                    "can_decode": False,
                    "missing_for_decode": [
                        "calibration",
                        *(["solve.scramble"] if normalized_scramble is None else []),
                        *fps_missing,
                        *resolution_missing,
                        "decoder extraction",
                    ],
                    "warnings": [*fps_warnings, *resolution_warnings],
                },
            }
            if derivation is not None:
                stored_derivation = deepcopy(derivation)
                derivative = stored_derivation.get("derivative")
                if not isinstance(derivative, dict):
                    raise WorkspaceError("derived capture linkage receipt is incomplete")
                if (
                    derivative.get("sha256") != digest.hexdigest()
                    or derivative.get("bytes") != byte_count
                ):
                    raise WorkspaceError("derived video changed before it was stored")
                derivative.update(
                    {
                        "recording_id": capture_id,
                        "path": video_path.relative_to(self.root).as_posix(),
                        "probe_at_import": probe,
                    }
                )
                receipt["derivation"] = stored_derivation
            self._write_checksums(capture_dir, receipt)
            self._write_receipt(capture_dir, receipt)
            receipt_committed = True
            return receipt
        except Exception:
            if staging_fd >= 0:
                os.close(staging_fd)
                staging_fd = -1
            if staging_path is not None:
                _unlink_entry(capture_dir, staging_path.name, description="capture video")
            _unlink_entry(
                capture_dir,
                "checksums.sha256",
                description="capture checksums",
            )
            if not receipt_committed:
                _unlink_entry(capture_dir, video_path.name, description="capture video")
            try:
                capture_dir.rmdir()
            except OSError:
                pass
            raise

    def register_verified_video(
        self,
        source_path: Path,
        *,
        capture_id: str,
        original_filename: str,
        capture_session_id: str,
        scramble: str | None,
        expected_bytes: int,
        expected_sha256: str,
        container: str,
        codec: str,
        encoded_width: int,
        encoded_height: int,
        fps_numerator: int,
        fps_denominator: int,
        frame_count: int,
        notes: str,
        warnings: tuple[str, ...] = (),
        _preflight: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Register an already-verified local video without copying its bytes.

        This narrow path is for revision-pinned public dataset artifacts that
        have just passed their download receipt. The source is hard-linked into
        a normal workspace capture, so every existing capture, media, sidecar,
        seal, and Decode path continues to operate on the standard bundle
        contract. Cross-filesystem copies and symlinks are deliberately not
        used. ``_preflight`` validates identity and hard-link capability without
        creating a capture; it exists for the public dataset batch registrar.
        """

        self.initialize()
        if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
            raise WorkspaceError("verified video capture id must be 32 lowercase hex characters")
        normalized_name = _safe_original_name(original_filename)
        extension = Path(normalized_name).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise WorkspaceError("verified video filename has an unsupported extension")
        session_id = capture_session_id.strip()
        if not session_id or len(session_id) > 200:
            raise WorkspaceError("verified video capture session id is invalid")
        normalized_scramble = normalize_scramble(scramble or "")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 1
        ):
            raise WorkspaceError("verified video byte count is invalid")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", expected_sha256
        ):
            raise WorkspaceError("verified video SHA-256 is invalid")
        if not isinstance(container, str) or not container.strip() or len(container) > 200:
            raise WorkspaceError("verified video container is invalid")
        if not isinstance(codec, str) or not codec.strip() or len(codec) > 100:
            raise WorkspaceError("verified video codec is invalid")
        for value, label in (
            (encoded_width, "width"),
            (encoded_height, "height"),
            (fps_numerator, "frame-rate numerator"),
            (fps_denominator, "frame-rate denominator"),
            (frame_count, "frame count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise WorkspaceError(f"verified video {label} is invalid")
        if fps_numerator > 1_000_000_000 or fps_denominator > 1_000_000_000:
            raise WorkspaceError("verified video frame rate is invalid")
        if not isinstance(notes, str):
            raise WorkspaceError("verified video notes must be a string")
        if not isinstance(warnings, tuple) or any(
            not isinstance(item, str) or not item.strip() or len(item) > 1000 for item in warnings
        ):
            raise WorkspaceError("verified video warnings are invalid")

        try:
            source_stat = source_path.lstat()
        except OSError as exc:
            raise WorkspaceError("verified video source is unavailable") from exc
        if _is_reparse_point(source_stat) or not stat.S_ISREG(source_stat.st_mode):
            raise WorkspaceError("verified video source must be a regular file")
        if source_stat.st_size != expected_bytes:
            raise WorkspaceError("verified video source byte count changed after verification")

        actual_fps = float(Fraction(fps_numerator, fps_denominator))
        capture_class, guidance = classify_frame_rate(actual_fps)
        rate_text = f"{fps_numerator}/{fps_denominator}"
        missing = [
            "calibration",
            *(["solve.scramble"] if normalized_scramble is None else []),
            "decoder extraction",
        ]
        receipt_warnings = list(warnings)
        if capture_class != "target":
            receipt_warnings.append(_OUT_OF_RANGE_FPS_WARNING)
        if not is_standard_capture_resolution(encoded_width, encoded_height):
            receipt_warnings.append(_LOW_RESOLUTION_WARNING)

        capture_dir = self.captures_dir / capture_id
        video_name = f"source{extension}"
        video_relative = (Path("captures") / capture_id / video_name).as_posix()
        video_path = capture_dir / video_name

        with self._capture_lock(capture_id):
            captures_fd = _open_directory_fd(
                self.captures_dir,
                description="workspace captures",
            )
            try:
                existing_stat = _entry_stat(captures_fd, capture_id)
            finally:
                _close_directory_fd(captures_fd)
            if existing_stat is not None:
                if _is_reparse_point(existing_stat) or not stat.S_ISDIR(existing_stat.st_mode):
                    raise WorkspaceError(
                        "verified video capture id collides with a workspace entry"
                    )
                existing = self._read_receipt(capture_dir)
                expected_existing = (
                    existing.get("schema") == "cubed-core/capture-bundle"
                    and existing.get("schema_version") == 1
                    and existing.get("recording_id") == capture_id
                    and existing.get("capture_id", capture_id) == capture_id
                    and existing.get("capture_session_id") == session_id
                    and existing.get("source") == "import"
                    and existing.get("original_filename") == normalized_name
                    and existing.get("video", {}).get("path") == video_relative
                    and existing.get("video", {}).get("bytes") == expected_bytes
                    and existing.get("video", {}).get("sha256") == expected_sha256
                    and existing.get("solve", {}).get("scramble") == normalized_scramble
                )
                if not expected_existing:
                    raise WorkspaceError(
                        "verified video capture id collides with a different workspace capture"
                    )
                try:
                    registered_video = self.capture_video_path(capture_id)
                    registered_stat = registered_video.lstat()
                except OSError as exc:
                    raise WorkspaceError("registered public dataset video is unavailable") from exc
                if (
                    _is_reparse_point(registered_stat)
                    or not stat.S_ISREG(registered_stat.st_mode)
                    or registered_stat.st_size != expected_bytes
                    or _sha256_file(registered_video) != expected_sha256
                ):
                    raise WorkspaceError(
                        "registered public dataset video does not match its verified receipt"
                    )
                return deepcopy(existing), False

            if _preflight:
                probe_name = f".verified-video-link-{secrets.token_hex(16)}"
                probe_path = self.captures_dir / probe_name
                try:
                    try:
                        os.link(source_path, probe_path, follow_symlinks=False)
                    except OSError as exc:
                        if exc.errno == errno.EXDEV:
                            raise WorkspaceError(
                                "public dataset and workspace are on different filesystems; "
                                "registration uses hard links and will not copy video bytes"
                            ) from exc
                        raise WorkspaceError(
                            "could not hard-link the verified public dataset video "
                            "into the workspace"
                        ) from exc
                    probe_stat = probe_path.lstat()
                    if (
                        not stat.S_ISREG(probe_stat.st_mode)
                        or probe_stat.st_size != expected_bytes
                        or not os.path.samefile(source_path, probe_path)
                    ):
                        raise WorkspaceError("verified public dataset hard-link probe is invalid")
                finally:
                    _unlink_entry(
                        self.captures_dir,
                        probe_name,
                        description="verified video link probe",
                    )
                return {}, True

            try:
                capture_dir.mkdir(mode=0o700)
            except OSError as exc:
                raise WorkspaceError("could not create verified workspace capture") from exc

            try:
                try:
                    os.link(source_path, video_path, follow_symlinks=False)
                except OSError as exc:
                    if exc.errno == errno.EXDEV:
                        raise WorkspaceError(
                            "public dataset and workspace are on different filesystems; "
                            "registration uses hard links and will not copy video bytes"
                        ) from exc
                    raise WorkspaceError(
                        "could not hard-link the verified public dataset video into the workspace"
                    ) from exc
                linked_stat = video_path.lstat()
                if (
                    not stat.S_ISREG(linked_stat.st_mode)
                    or linked_stat.st_size != expected_bytes
                    or not os.path.samefile(source_path, video_path)
                ):
                    raise WorkspaceError("verified public dataset hard link is invalid")

                created_at = datetime.now(timezone.utc).isoformat()
                receipt: dict[str, Any] = {
                    "schema": "cubed-core/capture-bundle",
                    "schema_version": 1,
                    "recording_id": capture_id,
                    "capture_id": capture_id,
                    "capture_session_id": session_id,
                    "created_at": created_at,
                    "source": "import",
                    "state": "incomplete",
                    "sealed_at": None,
                    "seal_purpose": None,
                    "original_filename": normalized_name,
                    "video": {
                        "path": video_relative,
                        "bytes": expected_bytes,
                        "sha256": expected_sha256,
                        "container": container,
                        "codec": codec,
                        "encoded_width": encoded_width,
                        "encoded_height": encoded_height,
                        "configured_fps": None,
                        "actual_fps": actual_fps,
                        "frame_count": frame_count,
                        "rotation_degrees": 0,
                        "mirrored": False,
                        "time_origin": None,
                    },
                    "camera": {
                        "facing": "unknown",
                        "device_model": None,
                        "camera_id": None,
                        "intrinsics": None,
                    },
                    "solve": {
                        "scramble": normalized_scramble,
                        "end_condition": "solved",
                        "start_frame": None,
                        "end_frame": None,
                    },
                    "calibration": None,
                    "teacher": None,
                    "sensors": {"phone_imu": None, "ble_raw": None},
                    "provenance": {
                        "producer": "cubed-core",
                        "producer_version": __version__,
                        "normalized_from_sha256": None,
                    },
                    "probe": {
                        "status": "ok",
                        "container": container,
                        "codec": codec,
                        "width": encoded_width,
                        "height": encoded_height,
                        "fps": actual_fps,
                        "fps_rational": rate_text,
                        "avg_frame_rate": rate_text,
                        "r_frame_rate": rate_text,
                        "fps_basis": "published-manifest",
                        "duration_seconds": round(frame_count / actual_fps, 6),
                        "frame_count": frame_count,
                        "rotation_degrees": 0,
                        "capture_class": capture_class,
                        "guidance": guidance,
                    },
                    "notes": notes.strip()[:2000],
                    "readiness": {
                        "can_label": True,
                        "can_decode": False,
                        "missing_for_decode": missing,
                        "warnings": receipt_warnings,
                    },
                }
                self._write_checksums(capture_dir, receipt)
                self._write_receipt(capture_dir, receipt)
                return receipt, True
            except Exception:
                for name, description in (
                    ("capture.json", "capture metadata"),
                    ("checksums.sha256", "capture checksums"),
                    (video_name, "capture video"),
                ):
                    _unlink_entry(capture_dir, name, description=description)
                try:
                    capture_dir.rmdir()
                except OSError:
                    pass
                raise

    def derive_240_to_120(self, capture_id: str) -> dict[str, Any]:
        """Create a separately receipted every-other-frame capture."""

        with self._capture_lock(capture_id):
            self.initialize()
            capture_dir = self._capture_dir(capture_id)
            source_receipt = self._read_receipt(capture_dir)
            if source_receipt.get("state") != "incomplete":
                raise WorkspaceError("sealed captures cannot be used to create derivatives")
            if (
                source_receipt.get("schema") != "cubed-core/capture-bundle"
                or source_receipt.get("schema_version") != 1
            ):
                raise WorkspaceError("source capture receipt is unsupported")
            if (
                source_receipt.get("recording_id") != capture_id
                or source_receipt.get("capture_id", capture_id) != capture_id
            ):
                raise WorkspaceError("source capture identity does not match its directory")

            source_probe_at_import = source_receipt.get("probe")
            source_video = source_receipt.get("video")
            if (
                not isinstance(source_probe_at_import, dict)
                or source_probe_at_import.get("status") != "ok"
                or not isinstance(source_video, dict)
            ):
                raise WorkspaceError("source capture is unprobed")
            camera = source_receipt.get("camera")
            solve = source_receipt.get("solve")
            capture_session_id = source_receipt.get("capture_session_id")
            camera_facing = camera.get("facing") if isinstance(camera, dict) else None
            mirrored = source_video.get("mirrored")
            scramble = solve.get("scramble") if isinstance(solve, dict) else None
            if (
                not isinstance(capture_session_id, str)
                or not capture_session_id.strip()
                or len(capture_session_id) > 200
                or camera_facing not in {"front", "back", "external", "unknown"}
                or not isinstance(mirrored, bool)
                or not isinstance(solve, dict)
                or (scramble is not None and not isinstance(scramble, str))
            ):
                raise WorkspaceError("source capture metadata is incomplete or invalid")
            receipted_fps = source_video.get("actual_fps")
            import_probe_fps = source_probe_at_import.get("fps")
            import_frame_rate = _probe_frame_rate_fraction(source_probe_at_import)
            if (
                isinstance(receipted_fps, bool)
                or not isinstance(receipted_fps, (int, float))
                or isinstance(import_probe_fps, bool)
                or not isinstance(import_probe_fps, (int, float))
                or not math.isclose(
                    float(receipted_fps),
                    float(import_probe_fps),
                    rel_tol=0,
                    abs_tol=0.0001,
                )
                or (
                    import_frame_rate is not None
                    and not math.isclose(
                        float(import_frame_rate),
                        float(import_probe_fps),
                        rel_tol=0,
                        abs_tol=0.0001,
                    )
                )
                or classify_frame_rate(float(receipted_fps))[0] != "research-high-speed"
                or classify_frame_rate(float(receipted_fps) / 2)[0] != "target"
            ):
                raise WorkspaceError(
                    "source capture must be in the verified 240 fps regime "
                    "(halved cadence must fall within 110-121 fps)"
                )

            source_path = self.capture_video_path(capture_id)
            expected_source_sha256 = source_video.get("sha256")
            expected_source_bytes = source_video.get("bytes")
            if (
                not isinstance(expected_source_sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", expected_source_sha256)
                or not isinstance(expected_source_bytes, int)
                or isinstance(expected_source_bytes, bool)
                or source_path.stat().st_size != expected_source_bytes
                or _sha256_file(source_path) != expected_source_sha256
            ):
                raise WorkspaceError("source video bytes do not match the capture receipt")

            source_probe = probe_video(source_path, count_frames=True)
            fresh_fps = source_probe.get("fps")
            source_frame_count = source_probe.get("frame_count")
            source_frame_rate = _probe_frame_rate_fraction(source_probe)
            target_frame_rate = source_frame_rate / 2 if source_frame_rate is not None else None
            if (
                source_probe.get("status") != "ok"
                or isinstance(fresh_fps, bool)
                or not isinstance(fresh_fps, (int, float))
                or classify_frame_rate(float(fresh_fps))[0] != "research-high-speed"
                or classify_frame_rate(float(fresh_fps) / 2)[0] != "target"
                or source_frame_rate is None
                or (import_frame_rate is not None and source_frame_rate != import_frame_rate)
                or target_frame_rate is None
                or classify_frame_rate(float(target_frame_rate))[0] != "target"
                or not isinstance(source_frame_count, int)
                or isinstance(source_frame_count, bool)
                or source_frame_count < 1
            ):
                raise WorkspaceError(
                    "fresh source probe must verify a countable 240 fps-regime video"
                )

            source_relative = source_path.relative_to(self.root.resolve()).as_posix()
            with tempfile.TemporaryDirectory(prefix="derivative-", dir=self.exports_dir) as raw:
                temporary_dir = Path(raw)
                output_path = temporary_dir / "every-other-frame-120.mp4"
                output_relative = output_path.relative_to(self.root).as_posix()
                try:
                    ffmpeg_receipt = create_240_to_120_video(
                        self.root,
                        source_relative=source_relative,
                        output_relative=output_relative,
                        target_frame_rate=(
                            f"{target_frame_rate.numerator}/{target_frame_rate.denominator}"
                        ),
                    )
                except CaptureDerivativeError as exc:
                    raise WorkspaceError(str(exc)) from exc

                if _sha256_file(source_path) != expected_source_sha256:
                    raise WorkspaceError("source video changed while the derivative was created")
                output_probe = probe_video(output_path, count_frames=True)
                output_fps = output_probe.get("fps")
                output_frame_count = output_probe.get("frame_count")
                expected_output_frames = (source_frame_count + 1) // 2
                measured_output_rate = _probe_frame_rate_fraction(output_probe)
                if (
                    output_probe.get("status") != "ok"
                    or isinstance(output_fps, bool)
                    or not isinstance(output_fps, (int, float))
                    or classify_frame_rate(float(output_fps))[0] != "target"
                    or measured_output_rate != target_frame_rate
                    or output_frame_count != expected_output_frames
                ):
                    raise WorkspaceError(
                        "derivative probe did not verify target cadence and exact "
                        "every-other-frame count"
                    )

                output_sha256 = _sha256_file(output_path)
                output_bytes = output_path.stat().st_size
                created_at = datetime.now(timezone.utc).isoformat()
                linkage = {
                    "schema": "cubed-core/frame-rate-derivation",
                    "schema_version": 1,
                    "kind": "every-other-frame-240-to-120",
                    "created_at": created_at,
                    "source": {
                        "recording_id": capture_id,
                        "path": source_relative,
                        "bytes": expected_source_bytes,
                        "sha256": expected_source_sha256,
                        "probe_at_import": source_probe_at_import,
                        "probe_before_derivation": source_probe,
                    },
                    "transform": {
                        "source_frame_index_origin": 0,
                        "source_frame_index_step": 2,
                        "frame_selection": "zero-based source frames 0,2,4,...",
                        "filter": ffmpeg_receipt["argv"][ffmpeg_receipt["argv"].index("-vf") + 1],
                        "source_frame_rate": (
                            f"{source_frame_rate.numerator}/{source_frame_rate.denominator}"
                        ),
                        "target_frame_rate": (
                            f"{target_frame_rate.numerator}/{target_frame_rate.denominator}"
                        ),
                        "timestamp_policy": ("retime-selected-frames-to-uniform-derived-cadence"),
                        "audio_policy": "discard",
                        "subtitle_policy": "discard",
                        "data_stream_policy": "discard",
                        "video_encoder": "libx264-crf18-yuv420p-single-thread",
                        "expected_output_frame_count": expected_output_frames,
                        "semantic_determinism": (
                            "The selected decoded presentation-order frame indices and "
                            "their uniform exact-rational output cadence are fixed by "
                            "this receipt."
                        ),
                        "compressed_byte_determinism": (
                            "Exact compressed bytes require the recorded FFmpeg build, "
                            "identical source bytes, argv, and execution environment; "
                            "bytes are not promised across FFmpeg builds."
                        ),
                    },
                    "ffmpeg": ffmpeg_receipt,
                    "derivative": {
                        "recording_id": "",
                        "path": "",
                        "bytes": output_bytes,
                        "sha256": output_sha256,
                        "probe": output_probe,
                    },
                }
                original_filename = source_receipt.get("original_filename")
                stem = (
                    Path(original_filename).stem if isinstance(original_filename, str) else "solve"
                )
                filename = f"{stem}.every-other-frame-120.mp4"
                with output_path.open("rb") as stream:
                    derivative_receipt = self.import_video(
                        stream,
                        filename=filename,
                        source="derived-240-to-120",
                        notes=f"Every-other-frame baseline derived from capture {capture_id}.",
                        capture_session_id=capture_session_id,
                        scramble=scramble or "",
                        camera_facing=camera_facing,
                        mirrored=mirrored,
                        derivation=linkage,
                        normalized_from_sha256=expected_source_sha256,
                    )
                return derivative_receipt

    def attach_json_sidecar(
        self,
        capture_id: str,
        stream: BinaryIO,
        *,
        kind: str,
        max_bytes: int = 64 * 1024**2,
        calibration_display_name: str | None = None,
    ) -> dict[str, Any]:
        with self._capture_lock(capture_id):
            return self._attach_json_sidecar_unlocked(
                capture_id,
                stream,
                kind=kind,
                max_bytes=max_bytes,
                calibration_display_name=calibration_display_name,
            )

    def _attach_json_sidecar_unlocked(
        self,
        capture_id: str,
        stream: BinaryIO,
        *,
        kind: str,
        max_bytes: int,
        calibration_display_name: str | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        destinations = {
            "calibration": Path("calibration.json"),
            "teacher": Path("teacher/cube_session.json"),
            "ble-raw": Path("teacher/ble_events.json"),
            "phone-imu": Path("sensors/phone_imu.json"),
        }
        destination = destinations.get(kind)
        if destination is None:
            raise WorkspaceError("sidecar kind must be calibration, teacher, ble-raw, or phone-imu")
        if calibration_display_name is not None and kind != "calibration":
            raise WorkspaceError("a calibration display name requires a calibration sidecar")
        calibration_display_name = _safe_calibration_display_name(calibration_display_name)
        capture_dir = self._capture_dir(capture_id)
        receipt = self._read_receipt(capture_dir)
        sealed = receipt.get("state") == "sealed"
        calibration_replaceable = (
            kind == "calibration" and sealed and receipt.get("seal_purpose") == "decode"
        )
        if sealed and not calibration_replaceable:
            raise WorkspaceError(
                "locked capture inputs are immutable; only calibration may be replaced "
                "when locked for decode"
            )

        payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise WorkspaceError(f"sidecar exceeds the {max_bytes}-byte limit")
        if not payload:
            raise WorkspaceError("sidecar is empty")
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError("sidecar must be valid JSON") from exc
        if not isinstance(parsed, (dict, list)):
            raise WorkspaceError("sidecar JSON must contain an object or array")
        calibration_kind: str | None = None
        if kind == "calibration":
            if isinstance(parsed, dict) and set(parsed) == set(COLOR_ORDER):
                # A bare flat six-color Lab map, e.g. the released decode-support
                # asset. Normalize it into a color-centroids-v1 document before
                # storing so the stored sidecar always declares its schema.
                try:
                    parsed = build_imported_centroids_document(parsed, source_bytes=payload)
                except ColorCentroidsError as exc:
                    raise WorkspaceError(str(exc)) from exc
                payload = json.dumps(parsed, sort_keys=True).encode("utf-8")
                calibration_kind = "color_centroids_v1"
            elif isinstance(parsed, dict) and parsed.get("schema") == COLOR_CENTROIDS_SCHEMA:
                # Second, explicitly-labeled calibration format for imported captures.
                # It never satisfies a check that specifically requires measured
                # samples; only the color-calibration-v1 branch below does.
                try:
                    validate_color_centroids(parsed)
                except ColorCentroidsError as exc:
                    raise WorkspaceError(str(exc)) from exc
                calibration_kind = "color_centroids_v1"
            else:
                # Everything else, including the phone-native contract and any
                # malformed upload, goes through the unweakened, unchanged
                # color-calibration-v1 validator exactly as before.
                try:
                    validate_color_calibration(parsed)
                except ColorCalibrationError as exc:
                    raise WorkspaceError(str(exc)) from exc
                calibration_kind = "color_calibration_v1"
                provenance = parsed["provenance"]
                receipt_facing = receipt.get("camera", {}).get("facing")
                if receipt_facing not in {None, "unknown"} and provenance["camera_facing"] not in {
                    "unknown",
                    receipt_facing,
                }:
                    raise WorkspaceError("calibration camera_facing does not match capture")
                if provenance["mirrored"] != receipt.get("video", {}).get("mirrored"):
                    raise WorkspaceError("calibration mirrored state does not match capture")
        elif kind == "ble-raw":
            validate_ble_session(parsed)
            if parsed.get("video_recording_id") != capture_id:
                raise WorkspaceError("BLE teacher video_recording_id does not match capture")
            if parsed.get("capture_session_id") != receipt.get("capture_session_id"):
                raise WorkspaceError("BLE teacher capture_session_id does not match capture")
            linked_scramble = (
                normalize_scramble(parsed["scramble"])
                if parsed.get("scramble") is not None
                else None
            )
            capture_scramble = receipt.get("solve", {}).get("scramble")
            if capture_scramble != linked_scramble:
                raise WorkspaceError("BLE teacher scramble does not match capture")
        elif kind == "teacher":
            _validate_cube_session_v2(parsed)
            if parsed.get("video_recording_id") != capture_id:
                raise WorkspaceError("CubeSession v2 video_recording_id does not match capture")
            if parsed.get("capture_session_id") != receipt.get("capture_session_id"):
                raise WorkspaceError("CubeSession v2 capture_session_id does not match capture")
            linked_scramble = (
                normalize_scramble(parsed["scramble"])
                if parsed.get("scramble") is not None
                else None
            )
            capture_scramble = receipt.get("solve", {}).get("scramble")
            if capture_scramble != linked_scramble:
                raise WorkspaceError("CubeSession v2 scramble does not match capture")
            raw_ble = receipt.get("sensors", {}).get("ble_raw")
            if isinstance(raw_ble, dict):
                if parsed["recording_id"] != raw_ble.get("ble_session_id"):
                    raise WorkspaceError("CubeSession v2 recording_id does not match raw BLE")
                if parsed["capture_session_id"] != raw_ble.get("capture_session_id"):
                    raise WorkspaceError("CubeSession v2 capture_session_id does not match raw BLE")

        digest = hashlib.sha256(payload).hexdigest()
        target = capture_dir / destination
        target_directory = (
            capture_dir
            if destination.parent == Path(".")
            else _ensure_child_directory(
                capture_dir,
                destination.parent.name,
                description="capture sidecar",
            )
        )
        previous_target = _read_regular_bytes(
            target_directory,
            target.name,
            description="capture sidecar",
            maximum_bytes=max_bytes,
            missing_ok=True,
        )
        _atomic_write_bytes(
            target_directory,
            target.name,
            payload,
            description="capture sidecar",
        )
        sidecar = {
            "path": destination.as_posix(),
            "sha256": digest,
        }
        if kind == "calibration":
            sidecar["kind"] = calibration_kind
            sidecar["schema"] = (
                COLOR_CALIBRATION_SCHEMA
                if calibration_kind == "color_calibration_v1"
                else COLOR_CENTROIDS_SCHEMA
            )
            sidecar["schema_version"] = 1
            if calibration_kind == "color_calibration_v1":
                sidecar["sample_count"] = 6 * 5 * 9
            if calibration_display_name is not None:
                sidecar["display_name"] = calibration_display_name
            receipt["calibration"] = sidecar
            missing = receipt["readiness"]["missing_for_decode"]
            receipt["readiness"]["missing_for_decode"] = [
                item for item in missing if item != "calibration"
            ]
        elif kind == "teacher":
            sidecar["kind"] = "cube_session_v2"
            sidecar["schema_version"] = 2
            sidecar["move_count"] = len(parsed["moves"])
            sidecar["orientation_count"] = len(parsed["orientations"])
            receipt["teacher"] = sidecar
        elif kind == "ble-raw":
            sidecar["kind"] = "cubed_core_ble_session_v1"
            sidecar["schema_version"] = 1
            sidecar["move_count"] = len(parsed["moves"])
            sidecar["orientation_count"] = len(parsed["orientations"])
            sidecar["ble_session_id"] = parsed["ble_session_id"]
            sidecar["capture_session_id"] = parsed["capture_session_id"]
            receipt["sensors"]["ble_raw"] = sidecar
        else:
            receipt["sensors"]["phone_imu"] = sidecar
        try:
            self._commit_existing_bundle_metadata(capture_dir, receipt)
        except Exception:
            if previous_target is None:
                _unlink_entry(
                    target_directory,
                    target.name,
                    description="capture sidecar",
                )
            else:
                _atomic_write_bytes(
                    target_directory,
                    target.name,
                    previous_target,
                    description="capture sidecar",
                    allow_replace_symlink=True,
                )
            raise
        return receipt

    def attach_reused_calibration(
        self,
        capture_id: str,
        *,
        source: str,
        source_capture_id: str | None,
        repo_root: Path,
        max_bytes: int = 64 * 1024**2,
    ) -> dict[str, Any]:
        """Attach a calibration sidecar copied from an existing source
        instead of an upload: the bundled GAN 12 release asset (the cube
        used in the gtD1s demo video), or another capture's own already-
        attached calibration.

        Both sources resolve to plain bytes that are then funneled through
        `_attach_json_sidecar_unlocked` exactly like an uploaded file, so
        every validation rule (schema, camera_facing/mirrored match, and the
        decode-lock calibration replacement boundary) applies unchanged; this
        method adds no validation logic of its own.
        """
        with self._capture_lock(capture_id):
            if source == "bundled":
                calibration_display_name = "Published shared calibration"
                asset_dir = repo_root / "workspace" / "release-assets"
                # A missing release-assets directory raises eagerly (before
                # missing_ok gets a say), same as a missing file inside an
                # existing one returns None; both mean the same thing here,
                # so both collapse into the one friendly message.
                try:
                    payload = _read_regular_bytes(
                        asset_dir,
                        "calibration_gan12.json",
                        description="bundled GAN 12 calibration asset",
                        maximum_bytes=max_bytes,
                        missing_ok=True,
                    )
                except WorkspaceError:
                    payload = None
                if payload is None:
                    raise WorkspaceError(
                        "the bundled GAN 12 calibration asset is not downloaded; "
                        "run make download-decode-support"
                    )
            elif source == "capture":
                if not source_capture_id:
                    raise WorkspaceError("source_capture_id is required when source is capture")
                source_dir = self._capture_dir(source_capture_id)
                source_receipt = self._read_receipt(source_dir)
                if not source_receipt.get("calibration"):
                    raise WorkspaceError("source capture has no calibration attached")
                source_filename = source_receipt.get("original_filename")
                calibration_display_name = (
                    f"From {_safe_original_name(source_filename)}"
                    if isinstance(source_filename, str) and source_filename.strip()
                    else "From another recording"
                )
                payload = _read_regular_bytes(
                    source_dir,
                    "calibration.json",
                    description="source capture calibration",
                    maximum_bytes=max_bytes,
                )
                assert payload is not None
            else:
                raise WorkspaceError("calibration reuse source must be bundled or capture")
            return self._attach_json_sidecar_unlocked(
                capture_id,
                io.BytesIO(payload),
                kind="calibration",
                max_bytes=max_bytes,
                calibration_display_name=calibration_display_name,
            )

    def seal_capture(self, capture_id: str, *, purpose: str) -> dict[str, Any]:
        with self._capture_lock(capture_id):
            return self._seal_capture_unlocked(capture_id, purpose=purpose)

    def _seal_capture_unlocked(self, capture_id: str, *, purpose: str) -> dict[str, Any]:
        self.initialize()
        if purpose not in {"label", "decode"}:
            raise WorkspaceError("seal purpose must be label or decode")
        capture_dir = self._capture_dir(capture_id)
        receipt = self._read_receipt(capture_dir)
        if receipt.get("state") == "sealed":
            return receipt
        if purpose == "decode":
            missing = []
            if receipt.get("calibration") is None:
                missing.append("calibration")
            if not receipt.get("solve", {}).get("scramble"):
                missing.append("solve.scramble")
            actual_fps = receipt.get("video", {}).get("actual_fps")
            if (
                isinstance(actual_fps, bool)
                or not isinstance(actual_fps, (int, float))
                or not math.isfinite(float(actual_fps))
                or float(actual_fps) <= 0
            ):
                missing.append("video.actual_fps")
            elif is_native_240_capture_frame_rate(actual_fps):
                missing.append(
                    "video.derive-240-to-120 (native 240 fps cannot be decoded directly)"
                )
            video = receipt.get("video", {})
            encoded_width = video.get("encoded_width")
            encoded_height = video.get("encoded_height")
            if (
                type(encoded_width) is not int
                or encoded_width <= 0
                or type(encoded_height) is not int
                or encoded_height <= 0
            ):
                missing.append("video.encoded_dimensions")
            if missing:
                raise WorkspaceError(
                    f"capture is not ready to seal for decode; missing: {', '.join(missing)}"
                )
        receipt["state"] = "sealed"
        receipt["sealed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["seal_purpose"] = purpose
        self._commit_existing_bundle_metadata(capture_dir, receipt)
        return receipt

    def capture_video_path(self, capture_id: str) -> Path:
        """Resolve a capture's original video without trusting receipt paths."""
        self.initialize()
        capture_dir = self._capture_dir(capture_id)
        receipt = self._read_receipt(capture_dir)
        relative_path = receipt.get("video", {}).get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise WorkspaceError("capture video is unavailable")
        candidate = self.root / relative_path
        try:
            candidate_stat = candidate.lstat()
            if _is_reparse_point(candidate_stat):
                raise WorkspaceError("capture video is unavailable")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(capture_dir.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise WorkspaceError("capture video is unavailable") from exc
        if not resolved.is_file():
            raise WorkspaceError("capture video is unavailable")
        return resolved

    def read_decode_artifacts(
        self,
        capture_id: str,
        *,
        maximum_calibration_bytes: int = 8 * 1024**2,
    ) -> DecodeCaptureArtifacts:
        """Read and re-verify one decode-locked camera capture.

        The caller receives the exact video/scramble lock plus one consistent
        calibration version, not paths supplied by an HTTP client. Video and
        calibration digests are rechecked while the capture's workspace lock is
        held. A decode job must snapshot the returned calibration bytes before
        execution because calibration may be replaced for a later attempt.
        Teacher and sensor sidecars are never opened.
        """

        if (
            type(maximum_calibration_bytes) is not int
            or maximum_calibration_bytes < 1
            or maximum_calibration_bytes > 64 * 1024**2
        ):
            raise WorkspaceError("maximum calibration size is invalid")

        with self._capture_lock(capture_id):
            self.initialize()
            capture_dir = self._capture_dir(capture_id)
            receipt_json = _read_regular_bytes(
                capture_dir,
                "capture.json",
                description="capture metadata",
                maximum_bytes=8 * 1024**2,
            )
            assert receipt_json is not None
            try:
                receipt = json.loads(receipt_json)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkspaceError("capture metadata is unavailable") from exc
            if not isinstance(receipt, dict):
                raise WorkspaceError("capture metadata is invalid")
            if (
                receipt.get("schema") != "cubed-core/capture-bundle"
                or receipt.get("schema_version") != 1
                or receipt.get("recording_id") != capture_id
                or receipt.get("capture_id", capture_id) != capture_id
            ):
                raise WorkspaceError("capture identity or schema is invalid")
            if receipt.get("state") != "sealed" or receipt.get("seal_purpose") != "decode":
                raise WorkspaceError("capture must be sealed for decode")

            solve = receipt.get("solve")
            scramble_value = solve.get("scramble") if isinstance(solve, dict) else None
            if not isinstance(scramble_value, str):
                raise WorkspaceError("capture requires a canonical scramble")
            scramble = normalize_scramble(scramble_value)
            if scramble is None:
                raise WorkspaceError("capture requires a canonical scramble")

            video = receipt.get("video")
            if not isinstance(video, dict):
                raise WorkspaceError("capture video receipt is invalid")
            expected_video_sha256 = video.get("sha256")
            expected_video_bytes = video.get("bytes")
            actual_fps = video.get("actual_fps")
            encoded_width = video.get("encoded_width")
            encoded_height = video.get("encoded_height")
            if (
                not isinstance(expected_video_sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", expected_video_sha256)
                or type(expected_video_bytes) is not int
                or expected_video_bytes < 1
                or isinstance(actual_fps, bool)
                or not isinstance(actual_fps, (int, float))
                or not math.isfinite(float(actual_fps))
                or float(actual_fps) <= 0
                or is_native_240_capture_frame_rate(actual_fps)
                or type(encoded_width) is not int
                or encoded_width <= 0
                or type(encoded_height) is not int
                or encoded_height <= 0
            ):
                raise WorkspaceError("capture video receipt is invalid for decode")
            video_path = self.capture_video_path(capture_id)
            if (
                video_path.stat().st_size != expected_video_bytes
                or _sha256_file(video_path) != expected_video_sha256
            ):
                raise WorkspaceError("capture video does not match its receipt")

            calibration = receipt.get("calibration")
            if not isinstance(calibration, dict):
                raise WorkspaceError("capture calibration receipt is missing")
            calibration_path = calibration.get("path")
            expected_calibration_sha256 = calibration.get("sha256")
            if (
                calibration_path != "calibration.json"
                or not isinstance(expected_calibration_sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", expected_calibration_sha256)
            ):
                raise WorkspaceError("capture calibration receipt is invalid")
            calibration_json = _read_regular_bytes(
                capture_dir,
                "calibration.json",
                description="capture calibration",
                maximum_bytes=maximum_calibration_bytes,
            )
            assert calibration_json is not None
            if hashlib.sha256(calibration_json).hexdigest() != expected_calibration_sha256:
                raise WorkspaceError("capture calibration does not match its receipt")

            return DecodeCaptureArtifacts(
                capture_id=capture_id,
                receipt=deepcopy(receipt),
                receipt_json=receipt_json,
                calibration_json=calibration_json,
                scramble=scramble,
            )

    def read_teacher_moves_sidecar(self, capture_id: str) -> bytes | None:
        """Read a capture's attached smart-cube teacher sidecar, if any.

        This is a diagnostic-only accessor for post-hoc review. It is
        deliberately never called from ``read_decode_artifacts``, whose
        docstring states that teacher and sensor sidecars are never opened for
        a decode: nothing here may influence what a decode runner receives.

        Returns ``None`` when the capture carries no ``teacher`` sidecar.
        """

        self.initialize()
        capture_dir = self._capture_dir(capture_id)
        receipt = self._read_receipt(capture_dir)
        teacher = receipt.get("teacher")
        if not isinstance(teacher, dict):
            return None
        relative_path = teacher.get("path")
        expected_sha256 = teacher.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
        ):
            raise WorkspaceError("capture teacher sidecar receipt is invalid")
        destination = Path(relative_path)
        if destination.is_absolute() or ".." in destination.parts:
            raise WorkspaceError("capture teacher sidecar receipt is invalid")
        directory = (
            capture_dir if destination.parent == Path(".") else capture_dir / destination.parent
        )
        payload = _read_regular_bytes(
            directory,
            destination.name,
            description="capture teacher sidecar",
            maximum_bytes=64 * 1024**2,
            missing_ok=True,
        )
        if payload is None:
            return None
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise WorkspaceError("capture teacher sidecar does not match its receipt")
        return payload

    def read_frame_annotations(self, capture_id: str, *, maximum_bytes: int) -> bytes:
        """Read mutable labels kept outside the immutable capture receipt."""

        self.initialize()
        self._capture_dir(capture_id)
        directory = self.annotations_dir / capture_id
        try:
            payload = _read_regular_bytes(
                directory,
                "frame-annotations.json",
                description="capture annotations",
                maximum_bytes=maximum_bytes,
            )
        except WorkspaceError as exc:
            if not _entry_exists(
                self.annotations_dir,
                capture_id,
                description="capture annotations",
            ):
                raise WorkspaceError("capture annotations not found") from exc
            raise
        assert payload is not None
        return payload

    def write_frame_annotations(self, capture_id: str, content: bytes) -> Path:
        """Atomically replace one derived annotation document."""

        self.initialize()
        self._capture_dir(capture_id)
        with self._capture_lock(capture_id):
            directory = _ensure_child_directory(
                self.annotations_dir,
                capture_id,
                description="capture annotations",
            )
            _atomic_write_bytes(
                directory,
                "frame-annotations.json",
                content,
                description="capture annotations",
            )
            return directory / "frame-annotations.json"

    def trash_capture(self, capture_id: str) -> dict[str, Any]:
        """Move one capture and its annotations to the operating-system Trash.

        Only workspace-owned directory entries are renamed. A published dataset
        video may be hard-linked into the capture bundle; renaming that link
        never opens, changes, or removes the source dataset entry.
        """

        self.initialize()
        if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
            raise WorkspaceError("capture not found")

        with self._capture_lock(capture_id):
            captures_fd = _open_directory_fd(
                self.captures_dir,
                description="workspace captures",
            )
            annotations_fd = _open_directory_fd(
                self.annotations_dir,
                description="workspace annotations",
            )
            root_fd = _open_directory_fd(
                self.root,
                description="workspace root",
            )
            destination_fd: int | Path | None = None
            staging_name: str | None = None
            capture_moved = False
            annotations_moved = False
            try:
                capture_stat = _entry_stat(captures_fd, capture_id)
                if capture_stat is None:
                    raise WorkspaceError("capture not found")
                if _is_reparse_point(capture_stat) or not stat.S_ISDIR(capture_stat.st_mode):
                    raise WorkspaceError("capture workspace entry is invalid")

                capture_dir = self.captures_dir / capture_id
                try:
                    capture_dir.resolve(strict=True).relative_to(
                        self.captures_dir.resolve(strict=True)
                    )
                except (OSError, ValueError) as exc:
                    raise WorkspaceError("capture workspace entry is invalid") from exc
                receipt = self._read_receipt(capture_dir)
                if (
                    receipt.get("schema") != "cubed-core/capture-bundle"
                    or receipt.get("recording_id") != capture_id
                    or receipt.get("capture_id", capture_id) != capture_id
                ):
                    raise WorkspaceError("capture metadata is invalid")

                annotations_stat = _entry_stat(annotations_fd, capture_id)
                if annotations_stat is not None and (
                    _is_reparse_point(annotations_stat)
                    or not stat.S_ISDIR(annotations_stat.st_mode)
                ):
                    raise WorkspaceError("capture annotations directory is invalid")
                if annotations_stat is not None:
                    try:
                        (self.annotations_dir / capture_id).resolve(strict=True).relative_to(
                            self.annotations_dir.resolve(strict=True)
                        )
                    except (OSError, ValueError) as exc:
                        raise WorkspaceError("capture annotations directory is invalid") from exc

                for _ in range(128):
                    candidate = f"Cubed Core capture {capture_id[:8]}-{secrets.token_hex(8)}"
                    try:
                        _mkdir_entry(root_fd, candidate, mode=0o700)
                    except FileExistsError:
                        continue
                    staging_name = candidate
                    break
                if staging_name is None:
                    raise WorkspaceError("could not allocate a unique Trash staging directory")
                try:
                    destination_fd = _open_directory_fd(
                        self.root / staging_name,
                        description="capture Trash staging",
                    )
                except OSError as exc:
                    raise WorkspaceError("capture Trash staging directory is unavailable") from exc

                try:
                    _replace_entry(
                        captures_fd,
                        capture_id,
                        destination_fd,
                        "capture",
                    )
                    capture_moved = True
                    if annotations_stat is not None:
                        _replace_entry(
                            annotations_fd,
                            capture_id,
                            destination_fd,
                            "annotations",
                        )
                        annotations_moved = True
                except OSError as exc:
                    rollback_error: OSError | None = None
                    if annotations_moved:
                        try:
                            _replace_entry(
                                destination_fd,
                                "annotations",
                                annotations_fd,
                                capture_id,
                            )
                            annotations_moved = False
                        except OSError as rollback_exc:
                            rollback_error = rollback_exc
                    if capture_moved:
                        try:
                            _replace_entry(
                                destination_fd,
                                "capture",
                                captures_fd,
                                capture_id,
                            )
                            capture_moved = False
                        except OSError as rollback_exc:
                            rollback_error = rollback_error or rollback_exc
                    if rollback_error is not None:
                        raise WorkspaceError("capture Trash rollback failed") from rollback_error
                    raise WorkspaceError("capture could not be moved to Trash") from exc

                for directory_fd in (
                    destination_fd,
                    captures_fd,
                    annotations_fd,
                    root_fd,
                ):
                    _sync_directory(directory_fd)
            finally:
                if destination_fd is not None:
                    _close_directory_fd(destination_fd)
                if staging_name is not None and not capture_moved and not annotations_moved:
                    try:
                        _rmdir_entry(root_fd, staging_name)
                    except OSError:
                        pass
                _close_directory_fd(root_fd)
                _close_directory_fd(annotations_fd)
                _close_directory_fd(captures_fd)

            assert staging_name is not None
            staging_path = self.root / staging_name
            try:
                send2trash(str(staging_path))
                if staging_path.exists():
                    raise OSError("Trash operation left the staging directory in place")
            except Exception as exc:
                rollback_error: OSError | None = None
                try:
                    os.replace(staging_path / "capture", self.captures_dir / capture_id)
                    capture_moved = False
                except OSError as rollback_exc:
                    rollback_error = rollback_exc
                if annotations_moved:
                    try:
                        os.replace(
                            staging_path / "annotations",
                            self.annotations_dir / capture_id,
                        )
                        annotations_moved = False
                    except OSError as rollback_exc:
                        rollback_error = rollback_error or rollback_exc
                try:
                    staging_path.rmdir()
                except OSError as rollback_exc:
                    if rollback_error is None and staging_path.exists():
                        rollback_error = rollback_exc
                if rollback_error is not None:
                    raise WorkspaceError("capture Trash rollback failed") from rollback_error
                raise WorkspaceError("capture could not be moved to Trash") from exc

        return {
            "schema": "cubed-core/capture-delete-v1",
            "schema_version": 1,
            "capture_id": capture_id,
            "trashed": True,
            "recoverable": True,
        }

    def create_export_directory(self) -> Path:
        """Create a private, disposable directory below the validated workspace."""

        self.initialize()
        if _path_is_reparse_point(self.exports_dir) or not self.exports_dir.is_dir():
            raise WorkspaceError("workspace exports directory is unavailable")
        return Path(tempfile.mkdtemp(prefix="export-", dir=self.exports_dir))

    def list_captures(self) -> list[dict[str, Any]]:
        self.initialize()
        captures: list[dict[str, Any]] = []
        captures_fd = _open_directory_fd(
            self.captures_dir,
            description="workspace captures",
        )
        try:
            names = _list_directory(captures_fd)
            capture_names = [
                name
                for name in names
                if re.fullmatch(r"[a-f0-9]{32}", name)
                and (entry_stat := _entry_stat(captures_fd, name)) is not None
                and not _is_reparse_point(entry_stat)
                and stat.S_ISDIR(entry_stat.st_mode)
            ]
        finally:
            _close_directory_fd(captures_fd)
        for capture_name in capture_names:
            capture_dir = self.captures_dir / capture_name
            try:
                value = self._read_receipt(capture_dir)
            except WorkspaceError:
                continue
            inferred_display_name = _legacy_sampled_calibration_display_name(
                capture_dir,
                value,
            )
            if inferred_display_name is not None:
                value["calibration"]["display_name"] = inferred_display_name
            captures.append(value)
        return sorted(captures, key=lambda row: str(row.get("created_at", "")), reverse=True)
