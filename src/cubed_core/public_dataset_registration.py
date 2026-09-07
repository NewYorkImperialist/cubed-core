from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .public_ground_truth import (
    SOLVED_FACELETS,
    GroundTruthDocumentIdentity,
    PublicGroundTruthError,
    validate_public_ground_truth_documents,
    write_public_ground_truth_index,
)
from .workspace import Workspace, WorkspaceError, normalize_scramble

DOWNLOAD_RECEIPT_SCHEMA = "cubed-core/public-dataset-download-receipt-v3"
CORPUS_MANIFEST_SCHEMA = "cubed-core/public-corpus-manifest"
SCRAMBLE_SCHEMA = "cubed-core/public-corpus-scramble"
HASH_CHUNK_BYTES = 1024 * 1024


class PublicDatasetRegistrationError(ValueError):
    """Raised when verified public data cannot become workspace captures safely."""


@dataclass(frozen=True, slots=True)
class PublicDatasetRegistrationSummary:
    dataset_id: str
    revision: str
    capture_count: int
    created_count: int
    existing_count: int
    on_camera_scramble_count: int


@dataclass(frozen=True, slots=True)
class _Artifact:
    path: PurePosixPath
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _Capture:
    capture_id: str
    tag: str
    video_path: Path
    video_bytes: int
    video_sha256: str
    container: str
    codec: str
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    scramble: str | None
    on_camera_scramble: bool
    scramble_artifact: _Artifact
    ble_artifact: _Artifact
    frame_ground_truth_artifact: _Artifact | None
    ground_truth_identity: GroundTruthDocumentIdentity


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicDatasetRegistrationError(f"{field} must be a JSON object")
    return value


def _array(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicDatasetRegistrationError(f"{field} must be a JSON array")
    return value


def _string(value: Any, *, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise PublicDatasetRegistrationError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PublicDatasetRegistrationError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise PublicDatasetRegistrationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, *, field: str) -> PurePosixPath:
    text = _string(value, field=field, maximum=500)
    if "\\" in text or "\x00" in text:
        raise PublicDatasetRegistrationError(f"{field} must be a normalized relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicDatasetRegistrationError(f"{field} must be a normalized relative path")
    return path


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicDatasetRegistrationError(f"{field} is not readable UTF-8 JSON") from exc
    return _object(value, field=field)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PublicDatasetRegistrationError("public dataset artifact is unavailable") from exc
    return digest.hexdigest()


def _safe_file(root: Path, relative: PurePosixPath, *, field: str) -> Path:
    candidate = root
    try:
        for part in relative.parts:
            candidate /= part
            item_stat = candidate.lstat()
            if stat.S_ISLNK(item_stat.st_mode):
                raise PublicDatasetRegistrationError(f"{field} may not use symlinks")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except PublicDatasetRegistrationError:
        raise
    except (OSError, ValueError) as exc:
        raise PublicDatasetRegistrationError(f"{field} is outside the verified dataset") from exc
    if not resolved.is_file():
        raise PublicDatasetRegistrationError(f"{field} must be a regular file")
    return resolved


def _receipt_artifacts(
    receipt: dict[str, Any],
) -> dict[PurePosixPath, _Artifact]:
    artifacts: dict[PurePosixPath, _Artifact] = {}
    for index, raw in enumerate(_array(receipt.get("artifacts"), field="download artifacts")):
        item = _object(raw, field=f"download artifacts[{index}]")
        path = _relative_path(item.get("path"), field=f"download artifacts[{index}].path")
        if path in artifacts:
            raise PublicDatasetRegistrationError(f"duplicate download artifact path: {path}")
        artifacts[path] = _Artifact(
            path=path,
            bytes=_positive_int(
                item.get("bytes"),
                field=f"download artifacts[{index}].bytes",
            ),
            sha256=_sha256(
                item.get("sha256"),
                field=f"download artifacts[{index}].sha256",
            ),
        )
    return artifacts


def _read_bound_json(
    root: Path,
    artifacts: dict[PurePosixPath, _Artifact],
    relative: PurePosixPath,
    *,
    field: str,
) -> dict[str, Any]:
    artifact = artifacts.get(relative)
    if artifact is None:
        raise PublicDatasetRegistrationError(f"{field} is absent from the download receipt")
    path = _safe_file(root, relative, field=field)
    if path.stat().st_size != artifact.bytes or _digest(path) != artifact.sha256:
        raise PublicDatasetRegistrationError(f"{field} changed after download verification")
    return _read_json(path, field=field)


def _artifact_for_role(
    capture_id: str,
    capture: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    matching = [
        _object(item, field=f"capture {capture_id} artifact")
        for item in _array(capture.get("artifacts"), field=f"capture {capture_id}.artifacts")
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matching) != 1:
        raise PublicDatasetRegistrationError(
            f"capture {capture_id} must declare exactly one {role} artifact"
        )
    return matching[0]


def _optional_artifact_for_role(
    capture_id: str,
    capture: dict[str, Any],
    role: str,
) -> dict[str, Any] | None:
    matching = [
        _object(item, field=f"capture {capture_id} artifact")
        for item in _array(capture.get("artifacts"), field=f"capture {capture_id}.artifacts")
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matching) > 1:
        raise PublicDatasetRegistrationError(
            f"capture {capture_id} may declare at most one {role} artifact"
        )
    return matching[0] if matching else None


def _expect_capture_path(
    capture_id: str,
    actual: PurePosixPath,
    filename: str,
    *,
    label: str,
) -> None:
    if actual != PurePosixPath("captures") / capture_id / filename:
        raise PublicDatasetRegistrationError(f"capture {capture_id} {label} path is not canonical")


def _cross_check_artifact(
    reference: dict[str, Any],
    artifacts: dict[PurePosixPath, _Artifact],
    *,
    field: str,
) -> tuple[PurePosixPath, _Artifact]:
    relative = _relative_path(reference.get("path"), field=f"{field}.path")
    artifact = artifacts.get(relative)
    if artifact is None:
        raise PublicDatasetRegistrationError(f"{field} is absent from the download receipt")
    if (
        _positive_int(reference.get("bytes"), field=f"{field}.bytes") != artifact.bytes
        or _sha256(reference.get("sha256"), field=f"{field}.sha256") != artifact.sha256
    ):
        raise PublicDatasetRegistrationError(f"{field} disagrees with the download receipt")
    return relative, artifact


def _tag_map(
    report: dict[str, Any] | None,
) -> dict[str, str]:
    if report is None:
        return {}
    result: dict[str, str] = {}
    tags = _object(report.get("tags"), field="derivation report tags")
    for capture_id, tag_value in tags.items():
        tag = _string(tag_value, field=f"derivation report tag {capture_id}", maximum=64)
        result[capture_id] = tag
    for index, raw in enumerate(_array(report.get("annex"), field="derivation report annex")):
        item = _object(raw, field=f"derivation report annex[{index}]")
        capture_id = _string(
            item.get("capture_id"),
            field=f"derivation report annex[{index}].capture_id",
            maximum=32,
        )
        tag = _string(
            item.get("tag"),
            field=f"derivation report annex[{index}].tag",
            maximum=64,
        )
        existing = result.get(capture_id)
        if existing is not None and existing != tag:
            raise PublicDatasetRegistrationError(
                f"derivation report assigns multiple tags to {capture_id}"
            )
        result[capture_id] = tag
    return result


def _scramble_for_capture(
    capture_id: str,
    document: dict[str, Any],
) -> tuple[str | None, bool]:
    if (
        document.get("schema") != SCRAMBLE_SCHEMA
        or document.get("schema_version") != 1
        or document.get("capture_id") != capture_id
    ):
        raise PublicDatasetRegistrationError(f"capture {capture_id} scramble identity is invalid")
    moves = _array(document.get("moves"), field=f"capture {capture_id} scramble moves")
    if not moves or any(not isinstance(move, str) for move in moves):
        raise PublicDatasetRegistrationError(f"capture {capture_id} scramble moves are invalid")
    scramble_object = _object(
        document.get("scramble"),
        field=f"capture {capture_id} scramble",
    )
    notation = _string(
        scramble_object.get("notation"),
        field=f"capture {capture_id} scramble.notation",
        maximum=500,
    )
    try:
        normalized = normalize_scramble(notation)
    except WorkspaceError as exc:
        raise PublicDatasetRegistrationError(
            f"capture {capture_id} scramble notation is invalid"
        ) from exc
    if normalized is None or normalized.split() != moves:
        raise PublicDatasetRegistrationError(
            f"capture {capture_id} scramble notation and moves disagree"
        )
    nested_moves = scramble_object.get("moves")
    if nested_moves is not None and nested_moves != moves:
        raise PublicDatasetRegistrationError(f"capture {capture_id} nested scramble moves disagree")

    initial_state = document.get("initial_state")
    recording_start = (
        initial_state.get("recording_start_facelets") if isinstance(initial_state, dict) else None
    )
    if recording_start is None:
        return normalized, False
    if recording_start != SOLVED_FACELETS:
        raise PublicDatasetRegistrationError(
            f"capture {capture_id} declares an unsupported recording start state"
        )
    return None, True


def _capture_rows(
    root: Path,
    artifacts: dict[PurePosixPath, _Artifact],
    manifest: dict[str, Any],
    tags: dict[str, str],
) -> list[_Capture]:
    _string(manifest.get("dataset_id"), field="corpus manifest dataset_id", maximum=128)
    captures_value = _array(manifest.get("captures"), field="corpus manifest captures")
    if _positive_int(manifest.get("capture_count"), field="corpus manifest capture_count") != len(
        captures_value
    ):
        raise PublicDatasetRegistrationError("corpus manifest capture count is inconsistent")

    rows: list[_Capture] = []
    seen: set[str] = set()
    for index, raw_capture in enumerate(captures_value):
        capture = _object(raw_capture, field=f"corpus capture[{index}]")
        capture_id = _string(
            capture.get("capture_id"),
            field=f"corpus capture[{index}].capture_id",
            maximum=32,
        )
        if not re.fullmatch(r"[a-f0-9]{32}", capture_id) or capture_id in seen:
            raise PublicDatasetRegistrationError("corpus capture ids are invalid or duplicated")
        seen.add(capture_id)

        video_reference = _artifact_for_role(capture_id, capture, "video")
        scramble_reference = _artifact_for_role(capture_id, capture, "scramble")
        ble_reference = _artifact_for_role(capture_id, capture, "ble_session")
        frame_ground_truth_reference = _optional_artifact_for_role(
            capture_id,
            capture,
            "teacher_truth",
        )
        video_relative, video_artifact = _cross_check_artifact(
            video_reference,
            artifacts,
            field=f"capture {capture_id} video",
        )
        scramble_relative, scramble_artifact = _cross_check_artifact(
            scramble_reference,
            artifacts,
            field=f"capture {capture_id} scramble",
        )
        ble_relative, ble_artifact = _cross_check_artifact(
            ble_reference,
            artifacts,
            field=f"capture {capture_id} BLE session",
        )
        frame_ground_truth_relative: PurePosixPath | None = None
        frame_ground_truth_artifact: _Artifact | None = None
        if frame_ground_truth_reference is not None:
            frame_ground_truth_relative, frame_ground_truth_artifact = _cross_check_artifact(
                frame_ground_truth_reference,
                artifacts,
                field=f"capture {capture_id} frame ground truth",
            )
        _expect_capture_path(capture_id, video_relative, "video.mp4", label="video")
        _expect_capture_path(capture_id, scramble_relative, "scramble.json", label="scramble")
        _expect_capture_path(capture_id, ble_relative, "cube_session.json", label="BLE session")
        if frame_ground_truth_relative is not None:
            _expect_capture_path(
                capture_id,
                frame_ground_truth_relative,
                "clip_ble_ground_truth.json",
                label="frame ground-truth",
            )

        video_path = _safe_file(root, video_relative, field=f"capture {capture_id} video")
        if video_path.stat().st_size != video_artifact.bytes:
            raise PublicDatasetRegistrationError(
                f"capture {capture_id} video changed after download verification"
            )
        scramble_document = _read_bound_json(
            root,
            artifacts,
            scramble_relative,
            field=f"capture {capture_id} scramble",
        )
        scramble, on_camera_scramble = _scramble_for_capture(
            capture_id,
            scramble_document,
        )
        ble_document = _read_bound_json(
            root,
            artifacts,
            ble_relative,
            field=f"capture {capture_id} BLE session",
        )
        frame_ground_truth_document = (
            None
            if frame_ground_truth_relative is None
            else _read_bound_json(
                root,
                artifacts,
                frame_ground_truth_relative,
                field=f"capture {capture_id} frame ground truth",
            )
        )

        media = _object(
            video_reference.get("media"),
            field=f"capture {capture_id} video.media",
        )
        measured_fps = _object(
            media.get("measured_fps"),
            field=f"capture {capture_id} video.media.measured_fps",
        )
        numerator = _positive_int(
            measured_fps.get("numerator"),
            field=f"capture {capture_id} fps numerator",
        )
        denominator = _positive_int(
            measured_fps.get("denominator"),
            field=f"capture {capture_id} fps denominator",
        )
        if numerator > 1_000_000_000 or denominator > 1_000_000_000:
            raise PublicDatasetRegistrationError(f"capture {capture_id} frame rate is invalid")
        fps = numerator / denominator
        if not math.isfinite(fps) or fps <= 0:
            raise PublicDatasetRegistrationError(f"capture {capture_id} frame rate is invalid")

        raw_tag = tags.get(capture_id)
        tag = (
            raw_tag
            if raw_tag is not None and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", raw_tag)
            else f"published-{capture_id[:8]}"
        )
        frame_count = _positive_int(
            media.get("frame_count"),
            field=f"capture {capture_id} video frame count",
        )
        try:
            ground_truth_identity = validate_public_ground_truth_documents(
                capture_id=capture_id,
                tag=tag,
                video_bytes=video_artifact.bytes,
                video_sha256=video_artifact.sha256,
                video_frame_count=frame_count,
                scramble_document=scramble_document,
                ble_document=ble_document,
                frame_ground_truth_document=frame_ground_truth_document,
            )
        except PublicGroundTruthError as exc:
            raise PublicDatasetRegistrationError(
                f"capture {capture_id} ground truth is invalid: {exc}"
            ) from exc
        rows.append(
            _Capture(
                capture_id=capture_id,
                tag=tag,
                video_path=video_path,
                video_bytes=video_artifact.bytes,
                video_sha256=video_artifact.sha256,
                container=_string(
                    media.get("container"),
                    field=f"capture {capture_id} video container",
                    maximum=200,
                ),
                codec=_string(
                    media.get("codec"),
                    field=f"capture {capture_id} video codec",
                    maximum=100,
                ),
                width=_positive_int(
                    media.get("width"),
                    field=f"capture {capture_id} video width",
                ),
                height=_positive_int(
                    media.get("height"),
                    field=f"capture {capture_id} video height",
                ),
                fps_numerator=numerator,
                fps_denominator=denominator,
                frame_count=frame_count,
                scramble=scramble,
                on_camera_scramble=on_camera_scramble,
                scramble_artifact=scramble_artifact,
                ble_artifact=ble_artifact,
                frame_ground_truth_artifact=frame_ground_truth_artifact,
                ground_truth_identity=ground_truth_identity,
            )
        )

    unknown_tags = sorted(set(tags) - seen)
    if unknown_tags:
        raise PublicDatasetRegistrationError(
            "derivation report references captures absent from the corpus manifest"
        )
    return rows


def _index_artifact(artifact: _Artifact) -> dict[str, Any]:
    return {
        "path": artifact.path.as_posix(),
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
    }


def _ground_truth_index(
    *,
    root: Path,
    dataset_id: str,
    revision: str,
    bootstrap_manifest_sha256: str,
    receipt_artifact: _Artifact,
    corpus_manifest_artifact: _Artifact,
    rows: list[_Capture],
) -> dict[str, Any]:
    return {
        "schema": "cubed-core/public-ground-truth-index-v1",
        "schema_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "revision": revision,
            "bootstrap_manifest_sha256": bootstrap_manifest_sha256,
            "download_receipt": _index_artifact(receipt_artifact),
            "corpus_manifest": _index_artifact(corpus_manifest_artifact),
            "local_dataset_root": str(root),
        },
        "captures": [
            {
                "capture_id": row.capture_id,
                "tag": row.tag,
                "video": {
                    "path": f"captures/{row.capture_id}/video.mp4",
                    "bytes": row.video_bytes,
                    "sha256": row.video_sha256,
                    "frame_count": row.frame_count,
                },
                "scramble": _index_artifact(row.scramble_artifact),
                "ble_session": _index_artifact(row.ble_artifact),
                "frame_ground_truth": (
                    None
                    if row.frame_ground_truth_artifact is None
                    else _index_artifact(row.frame_ground_truth_artifact)
                ),
                "linkage": row.ground_truth_identity.index_linkage(),
            }
            for row in rows
        ],
    }


def _verified_video_kwargs(
    row: _Capture,
    *,
    dataset_id: str,
    revision: str,
    notes: str,
    warnings: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "capture_id": row.capture_id,
        "original_filename": f"{row.tag}.mp4",
        "capture_session_id": f"public-dataset:{dataset_id}@{revision}:{row.capture_id}",
        "scramble": row.scramble,
        "expected_bytes": row.video_bytes,
        "expected_sha256": row.video_sha256,
        "container": row.container,
        "codec": row.codec,
        "encoded_width": row.width,
        "encoded_height": row.height,
        "fps_numerator": row.fps_numerator,
        "fps_denominator": row.fps_denominator,
        "frame_count": row.frame_count,
        "notes": notes,
        "warnings": warnings,
    }


def register_downloaded_public_dataset(
    dataset_root: Path,
    workspace: Workspace,
    *,
    expected_dataset_id: str,
    expected_manifest_sha256: str,
) -> PublicDatasetRegistrationSummary:
    """Register every verified public video as an incomplete workspace capture.

    The caller must invoke this only after the downloader has verified the full
    revision. This function rechecks the small JSON identities it consumes and
    cross-binds every video to that receipt, but intentionally does not hash the
    multi-gigabyte video set a second time.
    """

    if dataset_root.is_symlink():
        raise PublicDatasetRegistrationError("public dataset root may not be a symlink")
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as exc:
        raise PublicDatasetRegistrationError("public dataset root is unavailable") from exc
    if not root.is_dir():
        raise PublicDatasetRegistrationError("public dataset root must be a directory")

    receipt_relative = PurePosixPath("download-receipt.json")
    receipt_path = _safe_file(root, receipt_relative, field="download receipt")
    receipt = _read_json(receipt_path, field="download receipt")
    receipt_artifact = _Artifact(
        path=receipt_relative,
        bytes=receipt_path.stat().st_size,
        sha256=_digest(receipt_path),
    )
    if receipt.get("schema") != DOWNLOAD_RECEIPT_SCHEMA:
        raise PublicDatasetRegistrationError("download receipt schema is unsupported")
    dataset_id = _string(
        receipt.get("dataset_id"), field="download receipt dataset_id", maximum=128
    )
    revision = _string(receipt.get("revision"), field="download receipt revision", maximum=128)
    if dataset_id != expected_dataset_id:
        raise PublicDatasetRegistrationError("download receipt dataset id changed")
    if (
        _sha256(
            receipt.get("manifest_sha256"),
            field="download receipt manifest_sha256",
        )
        != expected_manifest_sha256
    ):
        raise PublicDatasetRegistrationError("download receipt is bound to another manifest")
    artifacts = _receipt_artifacts(receipt)

    corpus_manifest_path = PurePosixPath("dataset/manifest.json")
    corpus_manifest_artifact = artifacts.get(corpus_manifest_path)
    if corpus_manifest_artifact is None:
        raise PublicDatasetRegistrationError("corpus manifest is absent from the download receipt")
    corpus_manifest = _read_bound_json(
        root,
        artifacts,
        corpus_manifest_path,
        field="corpus manifest",
    )
    if (
        corpus_manifest.get("schema") != CORPUS_MANIFEST_SCHEMA
        or corpus_manifest.get("schema_version") != 1
        or corpus_manifest.get("dataset_id") != dataset_id
    ):
        raise PublicDatasetRegistrationError("corpus manifest identity is invalid")

    report_path = PurePosixPath("benchmark/derivation-report.json")
    report = (
        _read_bound_json(
            root,
            artifacts,
            report_path,
            field="benchmark derivation report",
        )
        if report_path in artifacts
        else None
    )
    rows = _capture_rows(root, artifacts, corpus_manifest, _tag_map(report))

    workspace.initialize()
    registrations: list[tuple[_Capture, str, tuple[str, ...]]] = []
    for row in rows:
        if row.on_camera_scramble:
            warnings = (
                "This published recording starts solved and includes its scramble on camera. "
                "No frame-zero scramble was registered, so Decode remains blocked.",
            )
            notes = (
                f"Published {row.tag} capture {row.capture_id} from "
                f"{dataset_id}@{revision}. The video starts solved and records its "
                "scramble on camera; the published notation is not a frame-zero "
                "Decode input."
            )
        else:
            warnings = ()
            notes = f"Published {row.tag} capture {row.capture_id} from {dataset_id}@{revision}."
        registrations.append((row, notes, warnings))

    # Validate every collision, existing receipt, existing workspace video, and
    # new hard-link operation before creating the first capture in the batch.
    # A bad later row therefore cannot strand a partial registration.
    preflighted: list[tuple[_Capture, str, tuple[str, ...], bool]] = []
    for row, notes, warnings in registrations:
        kwargs = _verified_video_kwargs(
            row,
            dataset_id=dataset_id,
            revision=revision,
            notes=notes,
            warnings=warnings,
        )
        try:
            _, would_create = workspace.register_verified_video(
                row.video_path,
                **kwargs,
                _preflight=True,
            )
        except WorkspaceError as exc:
            raise PublicDatasetRegistrationError(
                f"could not register published capture {row.tag} ({row.capture_id}): {exc}"
            ) from exc
        preflighted.append((row, notes, warnings, would_create))

    created = 0
    existing = 0
    for row, notes, warnings, would_create in preflighted:
        if not would_create:
            existing += 1
            continue
        kwargs = _verified_video_kwargs(
            row,
            dataset_id=dataset_id,
            revision=revision,
            notes=notes,
            warnings=warnings,
        )
        try:
            _, was_created = workspace.register_verified_video(row.video_path, **kwargs)
        except WorkspaceError as exc:
            raise PublicDatasetRegistrationError(
                f"could not register published capture {row.tag} ({row.capture_id}): {exc}"
            ) from exc
        if was_created:
            created += 1
        else:
            existing += 1

    index = _ground_truth_index(
        root=root,
        dataset_id=dataset_id,
        revision=revision,
        bootstrap_manifest_sha256=expected_manifest_sha256,
        receipt_artifact=receipt_artifact,
        corpus_manifest_artifact=corpus_manifest_artifact,
        rows=rows,
    )
    try:
        write_public_ground_truth_index(workspace.root, index)
    except PublicGroundTruthError as exc:
        raise PublicDatasetRegistrationError(
            f"could not persist the public ground-truth index: {exc}"
        ) from exc

    return PublicDatasetRegistrationSummary(
        dataset_id=dataset_id,
        revision=revision,
        capture_count=len(rows),
        created_count=created,
        existing_count=existing,
        on_camera_scramble_count=sum(row.on_camera_scramble for row in rows),
    )
