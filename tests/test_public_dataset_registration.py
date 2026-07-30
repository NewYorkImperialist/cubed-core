from __future__ import annotations

import errno
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import cubed_core.workspace as workspace_module
from cubed_core.public_dataset_registration import (
    SOLVED_FACELETS,
    PublicDatasetRegistrationError,
    register_downloaded_public_dataset,
)
from cubed_core.public_ground_truth import (
    INDEX_DIRECTORY,
    INDEX_FILENAME,
    build_public_ground_truth_diagnostic,
    fold_raw_qtm_to_htm,
    try_build_public_ground_truth_diagnostic,
)
from cubed_core.workspace import Workspace

MANIFEST_SHA256 = "a" * 64
REVISION = "b" * 40
NORMAL_ID = "1" * 32
ON_CAMERA_ID = "2" * 32


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _scramble(capture_id: str, *, on_camera: bool) -> bytes:
    document: dict[str, object] = {
        "schema": "cubed-core/public-corpus-scramble",
        "schema_version": 1,
        "capture_id": capture_id,
        "moves": ["R", "U", "R'", "U'"],
        "scramble": {
            "notation": "R U R' U'",
            "moves": ["R", "U", "R'", "U'"],
        },
        "initial_state": {"facelets": "ignored-by-registration"},
        "derivation": {
            "method": "synthetic test fixture",
            "source_file": "fixture.json",
            "verification": {
                ("core_engine_full_log_replay_solved" if on_camera else "core_replay_solved"): True,
            },
        },
    }
    if on_camera:
        initial_state = document["initial_state"]
        assert isinstance(initial_state, dict)
        initial_state["recording_start_facelets"] = SOLVED_FACELETS
    return _json_bytes(document)


def _ble_session(
    capture_id: str,
    *,
    on_camera: bool,
    link_status: str,
) -> bytes:
    scramble = "R U R' U'"
    raw_moves = ["R", "R", "U", "U'", "F"]
    document: dict[str, object] = {
        "schema_version": 2,
        "video_recording_id": capture_id,
        "video_link_status": link_status,
        "device": "synthetic cube",
        "mode": "normal",
        "moves": [
            {
                "move": move,
                "facelets": "U" * 54,
                "quat": [0.0, 0.0, 0.0, 1.0],
            }
            for move in raw_moves
        ],
        "orientations": [],
        "zero_quat": None,
    }
    if on_camera:
        document["scramble_recorded"] = scramble
        document["start_state"] = [0] * 54
        document["start_state_source"] = "solved recording start"
    else:
        document["scramble"] = scramble
        document["capture_session_id"] = "capture-session"
        document["recording_id"] = "recording-session"
    if link_status == "failed_timeout":
        document["video_ingest_fail_reason"] = "synthetic timeout"
    return _json_bytes(document)


def _frame_ground_truth(
    *,
    tag: str,
    video_payload: bytes,
) -> bytes:
    raw_moves = ["R", "R", "U", "U'", "F"]
    return _json_bytes(
        {
            "schema": "cubed-core/clip-ble-ground-truth-v1",
            "schema_version": 1,
            "tag": tag,
            "clip": {
                "filename": "video.mp4",
                "sha256": _sha256(video_payload),
                "bytes": len(video_payload),
                "frame_count": 1200,
            },
            "frame_index": {
                "basis": "clip-local",
                "first_move_frame": 10,
                "last_move_frame": 50,
            },
            "solve": {
                "scramble": "R U R' U'",
                "move_count": len(raw_moves),
                "move_metric": "quarter-turn",
            },
            "moves": [
                {"index": index, "frame": (index + 1) * 10, "move": move}
                for index, move in enumerate(raw_moves)
            ],
            "canonical_moves": ["R2", "F"],
        }
    )


def _video_reference(
    capture_id: str,
    payload: bytes,
    *,
    sha256: str | None = None,
) -> dict[str, object]:
    return {
        "role": "video",
        "path": f"captures/{capture_id}/video.mp4",
        "bytes": len(payload),
        "sha256": sha256 or _sha256(payload),
        "media": {
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "codec": "h264",
            "width": 1080,
            "height": 1920,
            "measured_fps": {"numerator": 120, "denominator": 1},
            "frame_count": 1200,
            "audio_stream_count": 0,
        },
    }


def _scramble_reference(capture_id: str, payload: bytes) -> dict[str, object]:
    return {
        "role": "scramble",
        "path": f"captures/{capture_id}/scramble.json",
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "media": None,
    }


def _json_reference(
    capture_id: str,
    payload: bytes,
    *,
    filename: str,
    role: str,
) -> dict[str, object]:
    return {
        "role": role,
        "path": f"captures/{capture_id}/{filename}",
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "media": None,
    }


def _write_public_dataset(
    root: Path,
    *,
    mismatched_video_reference: bool = False,
) -> None:
    normal_video = b"normal published video"
    on_camera_video = b"on-camera published video"
    normal_scramble = _scramble(NORMAL_ID, on_camera=False)
    on_camera_scramble = _scramble(ON_CAMERA_ID, on_camera=True)
    normal_ble = _ble_session(
        NORMAL_ID,
        on_camera=False,
        link_status="failed_timeout",
    )
    on_camera_ble = _ble_session(
        ON_CAMERA_ID,
        on_camera=True,
        link_status="linked",
    )
    normal_frame_ground_truth = _frame_ground_truth(
        tag="cs10",
        video_payload=normal_video,
    )

    payloads: dict[str, bytes] = {
        f"captures/{NORMAL_ID}/video.mp4": normal_video,
        f"captures/{NORMAL_ID}/scramble.json": normal_scramble,
        f"captures/{NORMAL_ID}/cube_session.json": normal_ble,
        f"captures/{NORMAL_ID}/clip_ble_ground_truth.json": normal_frame_ground_truth,
        f"captures/{ON_CAMERA_ID}/video.mp4": on_camera_video,
        f"captures/{ON_CAMERA_ID}/scramble.json": on_camera_scramble,
        f"captures/{ON_CAMERA_ID}/cube_session.json": on_camera_ble,
    }
    captures = [
        {
            "capture_id": NORMAL_ID,
            "artifacts": [
                _scramble_reference(NORMAL_ID, normal_scramble),
                _json_reference(
                    NORMAL_ID,
                    normal_ble,
                    filename="cube_session.json",
                    role="ble_session",
                ),
                _json_reference(
                    NORMAL_ID,
                    normal_frame_ground_truth,
                    filename="clip_ble_ground_truth.json",
                    role="teacher_truth",
                ),
                _video_reference(
                    NORMAL_ID,
                    normal_video,
                    sha256="f" * 64 if mismatched_video_reference else None,
                ),
            ],
        },
        {
            "capture_id": ON_CAMERA_ID,
            "artifacts": [
                _scramble_reference(ON_CAMERA_ID, on_camera_scramble),
                _json_reference(
                    ON_CAMERA_ID,
                    on_camera_ble,
                    filename="cube_session.json",
                    role="ble_session",
                ),
                _video_reference(ON_CAMERA_ID, on_camera_video),
            ],
        },
    ]
    corpus_manifest = _json_bytes(
        {
            "schema": "cubed-core/public-corpus-manifest",
            "schema_version": 1,
            "dataset_id": "cubed-solves-v1",
            "capture_count": len(captures),
            "captures": captures,
        }
    )
    derivation_report = _json_bytes(
        {
            "tags": {NORMAL_ID: "cs10"},
            "annex": [{"capture_id": ON_CAMERA_ID, "tag": "gtD2"}],
        }
    )
    payloads["dataset/manifest.json"] = corpus_manifest
    payloads["benchmark/derivation-report.json"] = derivation_report

    for relative, payload in payloads.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    receipt = {
        "schema": "cubed-core/public-dataset-download-receipt-v3",
        "dataset_id": "cubed-solves-v1",
        "manifest_sha256": MANIFEST_SHA256,
        "revision": REVISION,
        "base_url": f"https://example.invalid/{REVISION}",
        "artifacts": [
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
            for relative, payload in payloads.items()
        ],
    }
    (root / "download-receipt.json").write_bytes(_json_bytes(receipt))


def _workspace(root: Path) -> Workspace:
    return Workspace(root, max_upload_bytes=1)


def test_registers_every_video_as_an_incomplete_hard_link(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)

    summary = register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert summary.capture_count == 2
    assert summary.created_count == 2
    assert summary.existing_count == 0
    assert summary.on_camera_scramble_count == 1

    captures = {capture["capture_id"]: capture for capture in workspace.list_captures()}
    normal = captures[NORMAL_ID]
    assert normal["original_filename"] == "cs10.mp4"
    assert normal["state"] == "incomplete"
    assert normal["seal_purpose"] is None
    assert normal["solve"]["scramble"] == "R U R' U'"
    assert normal["calibration"] is None
    assert normal["teacher"] is None
    assert normal["sensors"] == {"phone_imu": None, "ble_raw": None}
    assert normal["readiness"]["missing_for_decode"] == [
        "calibration",
        "decoder extraction",
    ]
    capture_schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/capture-bundle-v1.schema.json").read_text()
    )
    Draft202012Validator(capture_schema).validate(normal)

    on_camera = captures[ON_CAMERA_ID]
    assert on_camera["original_filename"] == "gtD2.mp4"
    assert on_camera["solve"]["scramble"] is None
    assert "solve.scramble" in on_camera["readiness"]["missing_for_decode"]
    assert any("starts solved" in warning for warning in on_camera["readiness"]["warnings"])

    assert (dataset / f"captures/{NORMAL_ID}/video.mp4").samefile(
        workspace.capture_video_path(NORMAL_ID)
    )
    assert (dataset / f"captures/{ON_CAMERA_ID}/video.mp4").samefile(
        workspace.capture_video_path(ON_CAMERA_ID)
    )


def test_registration_works_without_directory_descriptor_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workspace_module, "_USE_DIRECTORY_FDS", False)
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)

    summary = register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert summary.capture_count == 2
    assert (dataset / f"captures/{NORMAL_ID}/video.mp4").samefile(
        workspace.capture_video_path(NORMAL_ID)
    )
    assert {capture["capture_id"] for capture in workspace.list_captures()} == {
        NORMAL_ID,
        ON_CAMERA_ID,
    }


def test_registration_is_idempotent_without_resetting_capture_state(
    tmp_path: Path,
    flat_color_centroids_payload: dict[str, list[float]],
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    first = register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    first_index = (workspace.root / INDEX_DIRECTORY / INDEX_FILENAME).read_bytes()
    workspace.attach_json_sidecar(
        NORMAL_ID,
        BytesIO(json.dumps(flat_color_centroids_payload).encode()),
        kind="calibration",
    )
    workspace.seal_capture(NORMAL_ID, purpose="decode")
    second = register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert first.created_count == 2
    assert second.created_count == 0
    assert second.existing_count == 2
    rows = {row["capture_id"]: row for row in workspace.list_captures()}
    assert len(rows) == 2
    assert rows[NORMAL_ID]["state"] == "sealed"
    assert rows[NORMAL_ID]["calibration"] is not None
    assert workspace.read_decode_artifacts(NORMAL_ID).scramble == "R U R' U'"
    assert (workspace.root / INDEX_DIRECTORY / INDEX_FILENAME).read_bytes() == first_index


def test_reregistered_fresh_dataset_inodes_preserve_workspace_state(
    tmp_path: Path,
    flat_color_centroids_payload: dict[str, list[float]],
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    workspace.attach_json_sidecar(
        NORMAL_ID,
        BytesIO(json.dumps(flat_color_centroids_payload).encode()),
        kind="calibration",
    )
    workspace.seal_capture(NORMAL_ID, purpose="decode")

    for capture_id in (NORMAL_ID, ON_CAMERA_ID):
        dataset_video = dataset / f"captures/{capture_id}/video.mp4"
        replacement = dataset_video.with_name("replacement.mp4")
        replacement.write_bytes(dataset_video.read_bytes())
        replacement.replace(dataset_video)
        assert not dataset_video.samefile(workspace.capture_video_path(capture_id))

    summary = register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert summary.created_count == 0
    assert summary.existing_count == 2
    rows = {row["capture_id"]: row for row in workspace.list_captures()}
    assert rows[NORMAL_ID]["state"] == "sealed"
    assert rows[NORMAL_ID]["calibration"] is not None
    assert workspace.read_decode_artifacts(NORMAL_ID).scramble == "R U R' U'"


def test_existing_workspace_video_must_still_match_receipted_bytes(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    dataset_video = dataset / f"captures/{NORMAL_ID}/video.mp4"
    replacement = dataset_video.with_name("replacement.mp4")
    replacement.write_bytes(dataset_video.read_bytes())
    replacement.replace(dataset_video)
    workspace_video = workspace.capture_video_path(NORMAL_ID)
    workspace_video.write_bytes(b"x" * workspace_video.stat().st_size)

    with pytest.raises(
        PublicDatasetRegistrationError,
        match="does not match its verified receipt",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )


def test_later_capture_collision_is_found_before_any_new_registration(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    colliding_video = dataset / f"captures/{ON_CAMERA_ID}/video.mp4"
    collision_receipt, created = workspace.register_verified_video(
        colliding_video,
        capture_id=ON_CAMERA_ID,
        original_filename="different.mp4",
        capture_session_id="different-session",
        scramble=None,
        expected_bytes=colliding_video.stat().st_size,
        expected_sha256=_sha256(colliding_video.read_bytes()),
        container="mov,mp4,m4a,3gp,3g2,mj2",
        codec="h264",
        encoded_width=1080,
        encoded_height=1920,
        fps_numerator=120,
        fps_denominator=1,
        frame_count=1200,
        notes="Intentional collision fixture.",
    )
    assert created is True

    with pytest.raises(
        PublicDatasetRegistrationError,
        match="collides with a different workspace capture",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )

    assert not (workspace.captures_dir / NORMAL_ID).exists()
    captures = workspace.list_captures()
    assert [capture["capture_id"] for capture in captures] == [ON_CAMERA_ID]
    assert captures[0]["capture_session_id"] == collision_receipt["capture_session_id"]


def test_manifest_video_identity_must_match_download_receipt(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset, mismatched_video_reference=True)

    with pytest.raises(
        PublicDatasetRegistrationError,
        match="video disagrees with the download receipt",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )

    assert workspace.list_captures() == []


def test_cross_filesystem_link_failure_never_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)

    def fail_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(workspace_module.os, "link", fail_link)
    with pytest.raises(
        PublicDatasetRegistrationError,
        match="different filesystems",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )

    assert workspace.list_captures() == []
    assert (dataset / f"captures/{NORMAL_ID}/video.mp4").read_bytes() == (b"normal published video")


def test_later_link_failure_is_found_before_any_capture_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    original_link = workspace_module.os.link
    link_count = 0

    def fail_second_link(*args: object, **kwargs: object) -> None:
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            raise OSError(errno.EACCES, "synthetic denied link")
        original_link(*args, **kwargs)

    monkeypatch.setattr(workspace_module.os, "link", fail_second_link)
    with pytest.raises(
        PublicDatasetRegistrationError,
        match="could not hard-link",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )

    assert workspace.list_captures() == []


def test_registration_builds_a_hash_bound_posthoc_ble_diagnostic(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)

    register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    index_path = workspace.root / INDEX_DIRECTORY / INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema"] == "cubed-core/public-ground-truth-index-v1"
    assert index["dataset"]["revision"] == REVISION
    normal_binding = next(item for item in index["captures"] if item["capture_id"] == NORMAL_ID)
    assert normal_binding["linkage"] == {
        "video_recording_id": NORMAL_ID,
        "video_link_status": "failed_timeout",
        "capture_session_id": "capture-session",
        "recording_id": "recording-session",
        "scramble_field": "scramble",
        "raw_qtm_count": 5,
        "canonical_htm_count": 2,
        "reference_scope": "sequence-and-clip-frame-indexed",
    }

    diagnostic = build_public_ground_truth_diagnostic(
        workspace.root,
        capture_id=NORMAL_ID,
        decoded_moves=["R2", "F"],
        video_sha256=_sha256(b"normal published video"),
        scramble="R U R' U'",
    )
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["reference"]["video_link_status"] == "failed_timeout"
    assert diagnostic["reference"]["scope"] == "sequence-and-clip-frame-indexed"
    assert diagnostic["normalization"] == {
        "raw_metric": "quarter-turn",
        "comparison_metric": "half-turn",
        "method": "adjacent-same-face-mod-4",
    }
    assert diagnostic["counts"] == {
        "decoded_htm": 2,
        "ble_raw_qtm": 5,
        "ble_canonical_htm": 2,
    }
    assert diagnostic["comparison"]["distance"] == 0
    assert diagnostic["frame_timing"]["basis"] == "clip-local"
    assert str(tmp_path) not in json.dumps(diagnostic)

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas/decode-ground-truth-diagnostic-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(diagnostic)


def test_sequence_only_binding_never_claims_frame_timing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)
    register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    diagnostic = build_public_ground_truth_diagnostic(
        workspace.root,
        capture_id=ON_CAMERA_ID,
        decoded_moves=["R2", "F"],
        video_sha256=_sha256(b"on-camera published video"),
        scramble="R U R' U'",
    )

    assert diagnostic["reference"]["scope"] == "sequence-only"
    assert "frame_ground_truth_sha256" not in diagnostic["reference"]
    assert "frame_timing" not in diagnostic


def test_safe_diagnostic_is_unavailable_for_missing_or_mismatched_ground_truth(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="cubed_core.public_ground_truth")
    workspace_root = tmp_path / "workspace"
    assert (
        try_build_public_ground_truth_diagnostic(
            workspace_root,
            capture_id=NORMAL_ID,
            decoded_moves=[],
            video_sha256="a" * 64,
            scramble="R",
        )
        is None
    )
    assert caplog.records == []

    dataset = tmp_path / "dataset"
    workspace = _workspace(workspace_root)
    _write_public_dataset(dataset)
    register_downloaded_public_dataset(
        dataset,
        workspace,
        expected_dataset_id="cubed-solves-v1",
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    assert (
        try_build_public_ground_truth_diagnostic(
            workspace_root,
            capture_id=NORMAL_ID,
            decoded_moves=["R2", "F"],
            video_sha256="f" * 64,
            scramble="R U R' U'",
        )
        is None
    )
    assert (
        try_build_public_ground_truth_diagnostic(
            workspace_root,
            capture_id=NORMAL_ID,
            decoded_moves=["R2", "F"],
            video_sha256=_sha256(b"normal published video"),
            scramble="R",
        )
        is None
    )
    ble_path = dataset / f"captures/{NORMAL_ID}/cube_session.json"
    ble_path.write_bytes(ble_path.read_bytes().replace(b'"move": "F"', b'"move": "B"', 1))
    assert (
        try_build_public_ground_truth_diagnostic(
            workspace_root,
            capture_id=NORMAL_ID,
            decoded_moves=["R2", "F"],
            video_sha256=_sha256(b"normal published video"),
            scramble="R U R' U'",
        )
        is None
    )
    assert len(caplog.records) == 3
    assert all(
        record.message == f"published ground-truth diagnostic unavailable for capture {NORMAL_ID}"
        for record in caplog.records
    )


def test_qtm_fold_is_deterministic_and_rejects_non_qtm_tokens() -> None:
    assert fold_raw_qtm_to_htm(["R", "R", "U", "U'", "R", "R'"]) == ["R2"]
    assert fold_raw_qtm_to_htm(["R", "R", "R"]) == ["R'"]
    assert fold_raw_qtm_to_htm(["R", "R'"]) == []
    with pytest.raises(ValueError, match="raw BLE moves"):
        fold_raw_qtm_to_htm(["R2"])


def test_ble_video_identity_mismatch_fails_before_capture_registration(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    workspace = _workspace(tmp_path / "workspace")
    _write_public_dataset(dataset)

    relative = f"captures/{NORMAL_ID}/cube_session.json"
    ble_path = dataset.joinpath(*relative.split("/"))
    ble = json.loads(ble_path.read_text(encoding="utf-8"))
    ble["video_recording_id"] = ON_CAMERA_ID
    replacement = _json_bytes(ble)
    ble_path.write_bytes(replacement)

    receipt_path = dataset / "download-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_entry = next(item for item in receipt["artifacts"] if item["path"] == relative)
    receipt_entry["bytes"] = len(replacement)
    receipt_entry["sha256"] = _sha256(replacement)
    receipt_path.write_bytes(_json_bytes(receipt))

    manifest_path = dataset / "dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = next(item for item in manifest["captures"] if item["capture_id"] == NORMAL_ID)
    manifest_entry = next(item for item in capture["artifacts"] if item["role"] == "ble_session")
    manifest_entry["bytes"] = len(replacement)
    manifest_entry["sha256"] = _sha256(replacement)
    manifest_payload = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_payload)
    corpus_receipt_entry = next(
        item for item in receipt["artifacts"] if item["path"] == "dataset/manifest.json"
    )
    corpus_receipt_entry["bytes"] = len(manifest_payload)
    corpus_receipt_entry["sha256"] = _sha256(manifest_payload)
    receipt_path.write_bytes(_json_bytes(receipt))

    with pytest.raises(
        PublicDatasetRegistrationError,
        match="BLE session is bound to another video",
    ):
        register_downloaded_public_dataset(
            dataset,
            workspace,
            expected_dataset_id="cubed-solves-v1",
            expected_manifest_sha256=MANIFEST_SHA256,
        )

    assert workspace.list_captures() == []
    assert not (workspace.root / INDEX_DIRECTORY / INDEX_FILENAME).exists()
