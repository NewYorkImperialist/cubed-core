"""Strict, post-hoc access to published smart-cube ground truth.

The public dataset ships camera videos beside BLE sessions, but the standard
Decode request contains only camera-side inputs. Dataset registration therefore
writes a hash-bound index outside capture bundles. A completed camera decode may
use this module *after* its result is frozen to build an optional diagnostic.

The safe entry point, :func:`try_build_public_ground_truth_diagnostic`, returns
``None`` for every missing, malformed, or mismatched ground-truth condition.
Ground-truth availability can never change a camera decode's lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .decode_diagnostics import DecodeDiagnosticsError, build_editdist_diagnostic

INDEX_SCHEMA = "cubed-core/public-ground-truth-index-v1"
INDEX_SCHEMA_VERSION = 1
INDEX_DIRECTORY = "public-ground-truth"
INDEX_FILENAME = "index-v1.json"

DIAGNOSTIC_SCHEMA = "cubed-core/decode-ground-truth-diagnostic-v1"
DIAGNOSTIC_SCHEMA_VERSION = 1

FRAME_GROUND_TRUTH_SCHEMA = "cubed-core/clip-ble-ground-truth-v1"
PUBLIC_SCRAMBLE_SCHEMA = "cubed-core/public-corpus-scramble"
SOLVED_FACELETS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_BOUND_JSON_BYTES = 8 * 1024 * 1024
MAX_SEQUENCE_MOVES = 400

_CAPTURE_ID = re.compile(r"[a-f0-9]{32}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_RAW_QTM_MOVE = re.compile(r"[URFDLB]'?\Z")
_LOGGER = logging.getLogger(__name__)


class PublicGroundTruthError(ValueError):
    """Published ground truth is unavailable, malformed, or mismatched."""


class PublicGroundTruthNotIndexed(PublicGroundTruthError):
    """No published reference is registered for this otherwise valid capture."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GroundTruthDocumentIdentity:
    video_recording_id: str
    video_link_status: str
    capture_session_id: str | None
    recording_id: str | None
    scramble_field: str
    raw_qtm_moves: tuple[str, ...]
    canonical_htm_moves: tuple[str, ...]
    reference_scope: str

    def index_linkage(self) -> dict[str, Any]:
        return {
            "video_recording_id": self.video_recording_id,
            "video_link_status": self.video_link_status,
            "capture_session_id": self.capture_session_id,
            "recording_id": self.recording_id,
            "scramble_field": self.scramble_field,
            "raw_qtm_count": len(self.raw_qtm_moves),
            "canonical_htm_count": len(self.canonical_htm_moves),
            "reference_scope": self.reference_scope,
        }


@dataclass(frozen=True, slots=True)
class PublicGroundTruthBinding:
    dataset_id: str
    revision: str
    bootstrap_manifest_sha256: str
    download_receipt: ArtifactIdentity
    corpus_manifest: ArtifactIdentity
    dataset_root: Path
    capture_id: str
    tag: str
    video: ArtifactIdentity
    video_frame_count: int
    scramble: ArtifactIdentity
    ble_session: ArtifactIdentity
    frame_ground_truth: ArtifactIdentity | None
    linkage: dict[str, Any]


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicGroundTruthError(f"{field} must be a JSON object")
    return value


def _array(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicGroundTruthError(f"{field} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, field: str) -> None:
    if set(value) != keys:
        raise PublicGroundTruthError(f"{field} has unsupported fields")


def _string(
    value: Any,
    *,
    field: str,
    maximum: int = 1000,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise PublicGroundTruthError(f"{field} is invalid")
    return value


def _optional_string(value: Any, *, field: str, maximum: int = 200) -> str | None:
    if value is None:
        return None
    return _string(value, field=field, maximum=maximum)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PublicGroundTruthError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicGroundTruthError(f"{field} must be a non-negative integer")
    return value


def _relative_path(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=500)
    if "\\" in text or "\x00" in text:
        raise PublicGroundTruthError(f"{field} must be a normalized relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicGroundTruthError(f"{field} must be a normalized relative path")
    return text


def _artifact(value: Any, *, field: str) -> ArtifactIdentity:
    item = _object(value, field=field)
    _exact_keys(item, {"path", "bytes", "sha256"}, field=field)
    return ArtifactIdentity(
        path=_relative_path(item.get("path"), field=f"{field}.path"),
        bytes=_positive_int(item.get("bytes"), field=f"{field}.bytes"),
        sha256=_string(
            item.get("sha256"),
            field=f"{field}.sha256",
            maximum=64,
            pattern=_SHA256,
        ),
    )


def _safe_bound_file(
    root: Path,
    relative: str,
    *,
    field: str,
) -> Path:
    relative_path = PurePosixPath(_relative_path(relative, field=field))
    candidate = root
    try:
        for part in relative_path.parts:
            candidate /= part
            item_stat = candidate.lstat()
            if stat.S_ISLNK(item_stat.st_mode):
                raise PublicGroundTruthError(f"{field} may not use symlinks")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except PublicGroundTruthError:
        raise
    except (OSError, ValueError) as exc:
        raise PublicGroundTruthError(f"{field} is outside the published dataset") from exc
    if not resolved.is_file():
        raise PublicGroundTruthError(f"{field} must be a regular file")
    return resolved


def _read_bound_bytes(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    field: str,
    maximum_bytes: int,
) -> bytes:
    if artifact.bytes > maximum_bytes:
        raise PublicGroundTruthError(f"{field} exceeds the supported size")
    path = _safe_bound_file(root, artifact.path, field=field)
    try:
        if path.stat().st_size != artifact.bytes:
            raise PublicGroundTruthError(f"{field} size does not match its index")
        payload = path.read_bytes()
    except OSError as exc:
        raise PublicGroundTruthError(f"{field} is unavailable") from exc
    if len(payload) != artifact.bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise PublicGroundTruthError(f"{field} does not match its index")
    return payload


def _read_bound_json(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    field: str,
) -> dict[str, Any]:
    payload = _read_bound_bytes(
        root,
        artifact,
        field=field,
        maximum_bytes=MAX_BOUND_JSON_BYTES,
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicGroundTruthError(f"{field} must be UTF-8 JSON") from exc
    return _object(value, field=field)


def _canonical_algorithm(value: Any, *, field: str) -> str:
    try:
        from .cube.notation import format_algorithm

        normalized = format_algorithm(_string(value, field=field, maximum=1000))
    except (TypeError, ValueError) as exc:
        raise PublicGroundTruthError(f"{field} contains an invalid move sequence") from exc
    if not normalized:
        raise PublicGroundTruthError(f"{field} may not be empty")
    return normalized


def fold_raw_qtm_to_htm(moves: Sequence[Any]) -> list[str]:
    """Collapse raw adjacent same-face quarter turns into canonical HTM.

    Raw public BLE sessions contain only clockwise or counter-clockwise quarter
    turns.  A stack fold combines adjacent same-face turns modulo four.  A
    cancellation can expose a new adjacency, which the next token then folds.
    """

    if isinstance(moves, (str, bytes)) or not isinstance(moves, Sequence):
        raise PublicGroundTruthError("raw BLE moves must be a sequence")
    if len(moves) > MAX_SEQUENCE_MOVES:
        raise PublicGroundTruthError(
            f"raw BLE moves exceed the {MAX_SEQUENCE_MOVES}-move diagnostic cap"
        )

    stack: list[list[Any]] = []
    for index, raw_token in enumerate(moves):
        token = _string(
            raw_token,
            field=f"raw BLE moves[{index}]",
            maximum=2,
            pattern=_RAW_QTM_MOVE,
        )
        face = token[0]
        amount = 3 if token.endswith("'") else 1
        if stack and stack[-1][0] == face:
            turns = (int(stack[-1][1]) + amount) % 4
            if turns == 0:
                stack.pop()
            else:
                stack[-1][1] = turns
        else:
            stack.append([face, amount])

    result: list[str] = []
    for face, raw_turns in stack:
        turns = int(raw_turns) % 4
        result.append(str(face) if turns == 1 else f"{face}2" if turns == 2 else f"{face}'")
    return result


def _public_scramble(
    capture_id: str,
    document: Mapping[str, Any],
) -> str:
    if (
        document.get("schema") != PUBLIC_SCRAMBLE_SCHEMA
        or document.get("schema_version") != 1
        or document.get("capture_id") != capture_id
    ):
        raise PublicGroundTruthError("published scramble identity is invalid")
    moves = _array(document.get("moves"), field="published scramble moves")
    if not moves or len(moves) > 200 or any(not isinstance(move, str) for move in moves):
        raise PublicGroundTruthError("published scramble moves are invalid")
    scramble_object = _object(document.get("scramble"), field="published scramble")
    notation = _canonical_algorithm(
        scramble_object.get("notation"),
        field="published scramble.notation",
    )
    if notation.split() != moves:
        raise PublicGroundTruthError("published scramble notation and moves disagree")
    nested_moves = scramble_object.get("moves")
    if nested_moves is not None and nested_moves != moves:
        raise PublicGroundTruthError("published nested scramble moves disagree")

    derivation = _object(document.get("derivation"), field="published scramble derivation")
    verification = _object(
        derivation.get("verification"),
        field="published scramble verification",
    )
    if (
        verification.get("core_replay_solved") is not True
        and verification.get("core_engine_full_log_replay_solved") is not True
    ):
        raise PublicGroundTruthError("published scramble lacks a solved replay verification")

    initial_state = document.get("initial_state")
    recording_start = (
        initial_state.get("recording_start_facelets") if isinstance(initial_state, dict) else None
    )
    if recording_start is not None and recording_start != SOLVED_FACELETS:
        raise PublicGroundTruthError("published recording start state is unsupported")
    return notation


def _ble_session_identity(
    capture_id: str,
    document: Mapping[str, Any],
    *,
    expected_scramble: str,
) -> tuple[str, str, str | None, str | None, str, tuple[str, ...]]:
    schema_version = document.get("schema_version")
    if schema_version is not None and schema_version != 2:
        raise PublicGroundTruthError("published BLE session schema version is unsupported")
    video_recording_id = _string(
        document.get("video_recording_id"),
        field="BLE video_recording_id",
        maximum=200,
    )
    if video_recording_id != capture_id:
        raise PublicGroundTruthError("BLE session is bound to another video")
    video_link_status = _string(
        document.get("video_link_status"),
        field="BLE video_link_status",
        maximum=64,
    )
    if video_link_status not in {"linked", "failed_timeout"}:
        raise PublicGroundTruthError("BLE video_link_status is unsupported")

    scramble_fields: list[tuple[str, str]] = []
    for name in ("scramble", "scramble_recorded"):
        value = document.get(name)
        if value is not None:
            scramble_fields.append((name, _canonical_algorithm(value, field=f"BLE {name}")))
    if not scramble_fields:
        raise PublicGroundTruthError("BLE session has no scramble identity")
    if any(notation != expected_scramble for _, notation in scramble_fields):
        raise PublicGroundTruthError("BLE and published scramble identities disagree")
    if len({notation for _, notation in scramble_fields}) != 1:
        raise PublicGroundTruthError("BLE scramble fields disagree")

    raw_moves = _array(document.get("moves"), field="BLE moves")
    if not raw_moves or len(raw_moves) > MAX_SEQUENCE_MOVES:
        raise PublicGroundTruthError("BLE move sequence is empty or unbounded")
    tokens: list[str] = []
    for index, raw_move in enumerate(raw_moves):
        move = _object(raw_move, field=f"BLE moves[{index}]")
        tokens.append(
            _string(
                move.get("move"),
                field=f"BLE moves[{index}].move",
                maximum=2,
                pattern=_RAW_QTM_MOVE,
            )
        )

    return (
        video_recording_id,
        video_link_status,
        _optional_string(
            document.get("capture_session_id"),
            field="BLE capture_session_id",
        ),
        _optional_string(document.get("recording_id"), field="BLE recording_id"),
        scramble_fields[0][0],
        tuple(tokens),
    )


def _validate_frame_ground_truth(
    document: Mapping[str, Any],
    *,
    tag: str,
    video_bytes: int,
    video_sha256: str,
    video_frame_count: int,
    scramble: str,
    raw_qtm_moves: Sequence[str],
    canonical_htm_moves: Sequence[str],
) -> None:
    if (
        document.get("schema") != FRAME_GROUND_TRUTH_SCHEMA
        or document.get("schema_version") != 1
        or document.get("tag") != tag
    ):
        raise PublicGroundTruthError("frame-indexed ground-truth identity is invalid")
    clip = _object(document.get("clip"), field="frame ground truth clip")
    if (
        clip.get("filename") != "video.mp4"
        or clip.get("sha256") != video_sha256
        or clip.get("bytes") != video_bytes
        or clip.get("frame_count") != video_frame_count
    ):
        raise PublicGroundTruthError("frame ground truth is bound to another video")
    frame_index = _object(
        document.get("frame_index"),
        field="frame ground truth frame_index",
    )
    if frame_index.get("basis") != "clip-local":
        raise PublicGroundTruthError("frame ground truth must use clip-local frame indices")

    solve = _object(document.get("solve"), field="frame ground truth solve")
    if (
        _canonical_algorithm(
            solve.get("scramble"),
            field="frame ground truth solve.scramble",
        )
        != scramble
        or solve.get("move_metric") != "quarter-turn"
        or solve.get("move_count") != len(raw_qtm_moves)
    ):
        raise PublicGroundTruthError("frame ground truth solve identity is invalid")

    moves = _array(document.get("moves"), field="frame ground truth moves")
    if len(moves) != len(raw_qtm_moves):
        raise PublicGroundTruthError("frame ground truth move count disagrees with BLE")
    frames: list[int] = []
    for index, (raw_move, expected_move) in enumerate(zip(moves, raw_qtm_moves, strict=True)):
        move = _object(raw_move, field=f"frame ground truth moves[{index}]")
        frame = _nonnegative_int(
            move.get("frame"),
            field=f"frame ground truth moves[{index}].frame",
        )
        if (
            move.get("index") != index
            or move.get("move") != expected_move
            or frame >= video_frame_count
        ):
            raise PublicGroundTruthError("frame ground truth moves disagree with BLE or video")
        frames.append(frame)
    if (
        frame_index.get("first_move_frame") != frames[0]
        or frame_index.get("last_move_frame") != frames[-1]
    ):
        raise PublicGroundTruthError("frame ground truth bounds disagree with its moves")

    canonical = _array(
        document.get("canonical_moves"),
        field="frame ground truth canonical_moves",
    )
    if canonical != list(canonical_htm_moves):
        raise PublicGroundTruthError("frame ground truth canonical fold disagrees with BLE")


def validate_public_ground_truth_documents(
    *,
    capture_id: str,
    tag: str,
    video_bytes: int,
    video_sha256: str,
    video_frame_count: int,
    scramble_document: Mapping[str, Any],
    ble_document: Mapping[str, Any],
    frame_ground_truth_document: Mapping[str, Any] | None,
) -> GroundTruthDocumentIdentity:
    """Cross-bind the public scramble, BLE session, and optional frame truth."""

    _string(capture_id, field="capture_id", maximum=32, pattern=_CAPTURE_ID)
    _string(tag, field="tag", maximum=64)
    _positive_int(video_bytes, field="video bytes")
    _string(video_sha256, field="video sha256", maximum=64, pattern=_SHA256)
    _positive_int(video_frame_count, field="video frame count")

    scramble = _public_scramble(capture_id, scramble_document)
    (
        video_recording_id,
        video_link_status,
        capture_session_id,
        recording_id,
        scramble_field,
        raw_qtm_moves,
    ) = _ble_session_identity(
        capture_id,
        ble_document,
        expected_scramble=scramble,
    )
    canonical_htm_moves = tuple(fold_raw_qtm_to_htm(raw_qtm_moves))

    reference_scope = "sequence-only"
    if frame_ground_truth_document is not None:
        _validate_frame_ground_truth(
            frame_ground_truth_document,
            tag=tag,
            video_bytes=video_bytes,
            video_sha256=video_sha256,
            video_frame_count=video_frame_count,
            scramble=scramble,
            raw_qtm_moves=raw_qtm_moves,
            canonical_htm_moves=canonical_htm_moves,
        )
        reference_scope = "sequence-and-clip-frame-indexed"

    return GroundTruthDocumentIdentity(
        video_recording_id=video_recording_id,
        video_link_status=video_link_status,
        capture_session_id=capture_session_id,
        recording_id=recording_id,
        scramble_field=scramble_field,
        raw_qtm_moves=raw_qtm_moves,
        canonical_htm_moves=canonical_htm_moves,
        reference_scope=reference_scope,
    )


def validate_public_ground_truth_index(value: Any) -> dict[str, Any]:
    """Validate and normalize one local public-ground-truth index."""

    document = _object(value, field="public ground-truth index")
    if (
        document.get("schema") != INDEX_SCHEMA
        or document.get("schema_version") != INDEX_SCHEMA_VERSION
        or set(document) != {"schema", "schema_version", "dataset", "captures"}
    ):
        raise PublicGroundTruthError("public ground-truth index identity is invalid")
    dataset = _object(document.get("dataset"), field="public ground-truth dataset")
    expected_dataset_keys = {
        "dataset_id",
        "revision",
        "bootstrap_manifest_sha256",
        "download_receipt",
        "corpus_manifest",
        "local_dataset_root",
    }
    _exact_keys(dataset, expected_dataset_keys, field="public ground-truth dataset")
    _string(dataset.get("dataset_id"), field="dataset_id", maximum=128)
    _string(dataset.get("revision"), field="revision", maximum=128)
    _string(
        dataset.get("bootstrap_manifest_sha256"),
        field="bootstrap_manifest_sha256",
        maximum=64,
        pattern=_SHA256,
    )
    download_receipt = _artifact(
        dataset.get("download_receipt"),
        field="download_receipt",
    )
    if download_receipt.path != "download-receipt.json":
        raise PublicGroundTruthError("download receipt path is not canonical")
    corpus_manifest = _artifact(
        dataset.get("corpus_manifest"),
        field="corpus_manifest",
    )
    if corpus_manifest.path != "dataset/manifest.json":
        raise PublicGroundTruthError("corpus manifest path is not canonical")

    root_text = _string(
        dataset.get("local_dataset_root"),
        field="local_dataset_root",
        maximum=4096,
    )
    root_path = Path(root_text)
    if not root_path.is_absolute():
        raise PublicGroundTruthError("local_dataset_root must be absolute")

    expected_capture_keys = {
        "capture_id",
        "tag",
        "video",
        "scramble",
        "ble_session",
        "frame_ground_truth",
        "linkage",
    }
    expected_linkage_keys = {
        "video_recording_id",
        "video_link_status",
        "capture_session_id",
        "recording_id",
        "scramble_field",
        "raw_qtm_count",
        "canonical_htm_count",
        "reference_scope",
    }
    captures = _array(document.get("captures"), field="public ground-truth captures")
    seen: set[str] = set()
    for index, raw_capture in enumerate(captures):
        capture = _object(raw_capture, field=f"captures[{index}]")
        _exact_keys(capture, expected_capture_keys, field=f"captures[{index}]")
        capture_id = _string(
            capture.get("capture_id"),
            field=f"captures[{index}].capture_id",
            maximum=32,
            pattern=_CAPTURE_ID,
        )
        if capture_id in seen:
            raise PublicGroundTruthError("public ground-truth capture ids are duplicated")
        seen.add(capture_id)
        _string(capture.get("tag"), field=f"captures[{index}].tag", maximum=64)

        video = _object(capture.get("video"), field=f"captures[{index}].video")
        _exact_keys(
            video,
            {"path", "bytes", "sha256", "frame_count"},
            field=f"captures[{index}].video",
        )
        video_artifact = _artifact(
            {key: video.get(key) for key in ("path", "bytes", "sha256")},
            field=f"captures[{index}].video artifact",
        )
        if video_artifact.path != f"captures/{capture_id}/video.mp4":
            raise PublicGroundTruthError("public ground-truth video path is not canonical")
        _positive_int(
            video.get("frame_count"),
            field=f"captures[{index}].video.frame_count",
        )

        scramble = _artifact(
            capture.get("scramble"),
            field=f"captures[{index}].scramble",
        )
        if scramble.path != f"captures/{capture_id}/scramble.json":
            raise PublicGroundTruthError("public ground-truth scramble path is not canonical")
        ble_session = _artifact(
            capture.get("ble_session"),
            field=f"captures[{index}].ble_session",
        )
        if ble_session.path != f"captures/{capture_id}/cube_session.json":
            raise PublicGroundTruthError("public ground-truth BLE path is not canonical")
        frame_value = capture.get("frame_ground_truth")
        if frame_value is not None:
            frame_artifact = _artifact(
                frame_value,
                field=f"captures[{index}].frame_ground_truth",
            )
            if frame_artifact.path != (f"captures/{capture_id}/clip_ble_ground_truth.json"):
                raise PublicGroundTruthError("frame-indexed ground-truth path is not canonical")

        linkage = _object(
            capture.get("linkage"),
            field=f"captures[{index}].linkage",
        )
        _exact_keys(linkage, expected_linkage_keys, field=f"captures[{index}].linkage")
        if linkage.get("video_recording_id") != capture_id:
            raise PublicGroundTruthError("public ground-truth video linkage is invalid")
        if linkage.get("video_link_status") not in {"linked", "failed_timeout"}:
            raise PublicGroundTruthError("public ground-truth video link status is invalid")
        _optional_string(
            linkage.get("capture_session_id"),
            field=f"captures[{index}].linkage.capture_session_id",
        )
        _optional_string(
            linkage.get("recording_id"),
            field=f"captures[{index}].linkage.recording_id",
        )
        if linkage.get("scramble_field") not in {"scramble", "scramble_recorded"}:
            raise PublicGroundTruthError("public ground-truth scramble field is invalid")
        _positive_int(
            linkage.get("raw_qtm_count"),
            field=f"captures[{index}].linkage.raw_qtm_count",
        )
        canonical_count = linkage.get("canonical_htm_count")
        if (
            isinstance(canonical_count, bool)
            or not isinstance(canonical_count, int)
            or canonical_count < 0
        ):
            raise PublicGroundTruthError(
                f"captures[{index}].linkage.canonical_htm_count is invalid"
            )
        expected_scope = (
            "sequence-and-clip-frame-indexed" if frame_value is not None else "sequence-only"
        )
        if linkage.get("reference_scope") != expected_scope:
            raise PublicGroundTruthError("public ground-truth reference scope is invalid")
    return document


def _index_path(workspace_root: Path) -> Path:
    return workspace_root / INDEX_DIRECTORY / INDEX_FILENAME


def write_public_ground_truth_index(workspace_root: Path, value: Any) -> Path:
    """Atomically replace the complete local index after full validation."""

    document = validate_public_ground_truth_index(value)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > MAX_INDEX_BYTES:
        raise PublicGroundTruthError("public ground-truth index exceeds the supported size")

    if workspace_root.is_symlink():
        raise PublicGroundTruthError("workspace root may not be a symlink")
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PublicGroundTruthError("workspace root is unavailable") from exc
    index_directory = workspace_root / INDEX_DIRECTORY
    if index_directory.is_symlink():
        raise PublicGroundTruthError("public ground-truth index directory may not be a symlink")
    try:
        index_directory.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise PublicGroundTruthError("public ground-truth index directory is unavailable") from exc
    if not index_directory.is_dir():
        raise PublicGroundTruthError("public ground-truth index directory is invalid")

    target = index_directory / INDEX_FILENAME
    if target.is_symlink():
        raise PublicGroundTruthError("public ground-truth index may not be a symlink")
    temporary = index_directory / f".{INDEX_FILENAME}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if target.is_symlink():
            raise PublicGroundTruthError("public ground-truth index may not be a symlink")
        os.replace(temporary, target)
        try:
            directory_fd = os.open(
                index_directory,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except PublicGroundTruthError:
        raise
    except OSError as exc:
        raise PublicGroundTruthError("could not persist the public ground-truth index") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return target


def _load_index(workspace_root: Path) -> dict[str, Any] | None:
    if workspace_root.is_symlink():
        raise PublicGroundTruthError("workspace root may not be a symlink")
    index_directory = workspace_root / INDEX_DIRECTORY
    if index_directory.is_symlink():
        raise PublicGroundTruthError("public ground-truth index directory may not be a symlink")
    path = _index_path(workspace_root)
    if path.is_symlink():
        raise PublicGroundTruthError("public ground-truth index may not be a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise PublicGroundTruthError("public ground-truth index is not a regular file")
    try:
        if path.stat().st_size > MAX_INDEX_BYTES:
            raise PublicGroundTruthError("public ground-truth index exceeds the supported size")
        payload = path.read_bytes()
        value = json.loads(payload)
    except PublicGroundTruthError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicGroundTruthError("public ground-truth index is unavailable") from exc
    return validate_public_ground_truth_index(value)


def load_public_ground_truth_binding(
    workspace_root: Path,
    capture_id: str,
) -> PublicGroundTruthBinding | None:
    """Load one indexed capture binding without opening teacher artifacts."""

    _string(capture_id, field="capture_id", maximum=32, pattern=_CAPTURE_ID)
    index = _load_index(workspace_root)
    if index is None:
        return None
    dataset = _object(index.get("dataset"), field="public ground-truth dataset")
    root = Path(str(dataset["local_dataset_root"]))
    if root.is_symlink():
        raise PublicGroundTruthError("published dataset root may not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise PublicGroundTruthError("published dataset root is unavailable") from exc
    if not root.is_dir():
        raise PublicGroundTruthError("published dataset root must be a directory")

    for raw_capture in _array(index.get("captures"), field="public ground-truth captures"):
        capture = _object(raw_capture, field="public ground-truth capture")
        if capture.get("capture_id") != capture_id:
            continue
        video = _object(capture.get("video"), field="public ground-truth video")
        frame_value = capture.get("frame_ground_truth")
        return PublicGroundTruthBinding(
            dataset_id=str(dataset["dataset_id"]),
            revision=str(dataset["revision"]),
            bootstrap_manifest_sha256=str(dataset["bootstrap_manifest_sha256"]),
            download_receipt=_artifact(
                dataset["download_receipt"],
                field="download_receipt",
            ),
            corpus_manifest=_artifact(
                dataset["corpus_manifest"],
                field="corpus_manifest",
            ),
            dataset_root=root,
            capture_id=capture_id,
            tag=str(capture["tag"]),
            video=_artifact(
                {key: video[key] for key in ("path", "bytes", "sha256")},
                field="video",
            ),
            video_frame_count=int(video["frame_count"]),
            scramble=_artifact(capture["scramble"], field="scramble"),
            ble_session=_artifact(capture["ble_session"], field="ble_session"),
            frame_ground_truth=(
                None if frame_value is None else _artifact(frame_value, field="frame_ground_truth")
            ),
            linkage=dict(_object(capture["linkage"], field="linkage")),
        )
    return None


def _verified_documents(
    binding: PublicGroundTruthBinding,
) -> tuple[GroundTruthDocumentIdentity, dict[str, Any]]:
    # Reverify the two dataset-wide authorities before opening the capture
    # artifacts.  The receipt itself is not portable identity, but its hash
    # records which verified local download produced this index.
    _read_bound_bytes(
        binding.dataset_root,
        binding.download_receipt,
        field="download receipt",
        maximum_bytes=MAX_INDEX_BYTES,
    )
    _read_bound_bytes(
        binding.dataset_root,
        binding.corpus_manifest,
        field="corpus manifest",
        maximum_bytes=MAX_BOUND_JSON_BYTES,
    )
    scramble_document = _read_bound_json(
        binding.dataset_root,
        binding.scramble,
        field="published scramble",
    )
    ble_document = _read_bound_json(
        binding.dataset_root,
        binding.ble_session,
        field="published BLE session",
    )
    frame_document = (
        None
        if binding.frame_ground_truth is None
        else _read_bound_json(
            binding.dataset_root,
            binding.frame_ground_truth,
            field="frame-indexed ground truth",
        )
    )
    identity = validate_public_ground_truth_documents(
        capture_id=binding.capture_id,
        tag=binding.tag,
        video_bytes=binding.video.bytes,
        video_sha256=binding.video.sha256,
        video_frame_count=binding.video_frame_count,
        scramble_document=scramble_document,
        ble_document=ble_document,
        frame_ground_truth_document=frame_document,
    )
    if identity.index_linkage() != binding.linkage:
        raise PublicGroundTruthError("published ground-truth linkage changed after registration")
    return identity, scramble_document


def build_public_ground_truth_diagnostic(
    workspace_root: Path,
    *,
    capture_id: str,
    decoded_moves: Sequence[Any],
    video_sha256: str,
    scramble: str,
) -> dict[str, Any]:
    """Build one verified, post-hoc sequence diagnostic.

    This strict entry point raises on unavailable or mismatched ground truth.
    Decode lifecycle code should call the safe ``try_`` wrapper below.
    """

    binding = load_public_ground_truth_binding(workspace_root, capture_id)
    if binding is None:
        raise PublicGroundTruthNotIndexed("published ground truth is not indexed for this capture")
    expected_video_sha256 = _string(
        video_sha256,
        field="decode video sha256",
        maximum=64,
        pattern=_SHA256,
    )
    if expected_video_sha256 != binding.video.sha256:
        raise PublicGroundTruthError("decode and published video identities disagree")

    identity, scramble_document = _verified_documents(binding)
    published_scramble = _public_scramble(binding.capture_id, scramble_document)
    if _canonical_algorithm(scramble, field="decode scramble") != published_scramble:
        raise PublicGroundTruthError("decode and published scramble identities disagree")

    try:
        editdist = build_editdist_diagnostic(
            decoded_moves,
            identity.canonical_htm_moves,
            reference="published-smart-cube-ble",
        )
    except DecodeDiagnosticsError as exc:
        raise PublicGroundTruthError(str(exc)) from exc

    reference: dict[str, Any] = {
        "kind": "published-smart-cube-ble",
        "scope": identity.reference_scope,
        "dataset_id": binding.dataset_id,
        "revision": binding.revision,
        "bootstrap_manifest_sha256": binding.bootstrap_manifest_sha256,
        "download_receipt_sha256": binding.download_receipt.sha256,
        "corpus_manifest_sha256": binding.corpus_manifest.sha256,
        "video_sha256": binding.video.sha256,
        "scramble_sha256": binding.scramble.sha256,
        "ble_sha256": binding.ble_session.sha256,
        "video_link_status": identity.video_link_status,
        "video_recording_id_verified": True,
    }
    result: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "capture_id": binding.capture_id,
        "reference": reference,
        "normalization": {
            "raw_metric": "quarter-turn",
            "comparison_metric": "half-turn",
            "method": "adjacent-same-face-mod-4",
        },
        "counts": {
            "decoded_htm": editdist["decoded_length"],
            "ble_raw_qtm": len(identity.raw_qtm_moves),
            "ble_canonical_htm": len(identity.canonical_htm_moves),
        },
        "comparison": {
            "distance": editdist["distance"],
            "ops": editdist["ops"],
        },
    }
    if binding.frame_ground_truth is not None:
        reference["frame_ground_truth_sha256"] = binding.frame_ground_truth.sha256
        result["frame_timing"] = {
            "available": True,
            "basis": "clip-local",
            "source_schema": FRAME_GROUND_TRUTH_SCHEMA,
            "source_sha256": binding.frame_ground_truth.sha256,
        }
    return result


def try_build_public_ground_truth_diagnostic(
    workspace_root: Path,
    *,
    capture_id: str,
    decoded_moves: Sequence[Any],
    video_sha256: str,
    scramble: str,
) -> dict[str, Any] | None:
    """Return a verified diagnostic, or ``None`` without affecting Decode."""

    try:
        return build_public_ground_truth_diagnostic(
            workspace_root,
            capture_id=capture_id,
            decoded_moves=decoded_moves,
            video_sha256=video_sha256,
            scramble=scramble,
        )
    except PublicGroundTruthNotIndexed:
        # Most OSS users will decode their own videos. No published teacher
        # binding for those captures is the normal case, not an operator warning.
        return None
    except PublicGroundTruthError:
        safe_capture = (
            capture_id
            if isinstance(capture_id, str) and _CAPTURE_ID.fullmatch(capture_id)
            else "invalid"
        )
        _LOGGER.warning(
            "published ground-truth diagnostic unavailable for capture %s",
            safe_capture,
        )
        return None
