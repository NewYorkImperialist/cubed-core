from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest

from cubed_core import workspace as workspace_module
from cubed_core.workspace import Workspace, WorkspaceError


def test_import_video_is_content_addressed_and_receipted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)

    receipt = workspace.import_video(
        BytesIO(b"video-bytes"),
        filename="../../solve.MOV",
        source="import",
        notes="test capture",
    )

    assert receipt["schema"] == "cubed-core/capture-bundle"
    assert receipt["schema_version"] == 1
    assert receipt["original_filename"] == "solve.MOV"
    assert receipt["video"]["bytes"] == 11
    assert len(receipt["video"]["sha256"]) == 64
    assert receipt["probe"]["fps"] == 120.0
    assert receipt["video"]["actual_fps"] == 120.0
    assert receipt["camera"]["facing"] == "unknown"
    assert receipt["state"] == "incomplete"
    assert (tmp_path / "workspace" / receipt["video"]["path"]).exists()
    assert (
        tmp_path / "workspace" / "captures" / receipt["capture_id"] / "checksums.sha256"
    ).exists()
    assert workspace.list_captures()[0]["capture_id"] == receipt["capture_id"]


def test_calibration_upload_display_name_is_safe_and_bounded() -> None:
    assert (
        workspace_module.calibration_upload_display_name("../../my <calibration>.json")
        == "my _calibration_.json"
    )
    assert workspace_module.calibration_upload_display_name("  ") is None
    assert len(workspace_module.calibration_upload_display_name("x" * 400) or "") == 180


def test_import_video_rejects_unknown_extension_and_size(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=5)

    with pytest.raises(WorkspaceError, match="unsupported video extension"):
        workspace.import_video(BytesIO(b"x"), filename="solve.txt", source="import")
    with pytest.raises(WorkspaceError, match="exceeds"):
        workspace.import_video(BytesIO(b"123456"), filename="solve.mp4", source="import")

    assert list((tmp_path / "workspace" / "captures").iterdir()) == []


def test_import_video_streams_to_disk_in_one_mib_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )

    class RecordingStream(BytesIO):
        requested_sizes: list[int]

        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self.requested_sizes = []

        def read(self, size: int = -1) -> bytes:
            self.requested_sizes.append(size)
            return super().read(size)

    payload = b"x" * (2 * 1024**2 + 17)
    stream = RecordingStream(payload)
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=3 * 1024**2)

    receipt = workspace.import_video(
        stream,
        filename="solve.mp4",
        source="import",
    )

    assert receipt["video"]["bytes"] == len(payload)
    assert stream.requested_sizes
    assert set(stream.requested_sizes) == {1024 * 1024}


@pytest.mark.parametrize("fps", [120.0, 219.999, 220.0, 242.0, 242.001])
def test_video_upload_limit_does_not_depend_on_measured_frame_rate(
    tmp_path,
    monkeypatch,
    fps: float,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": fps},
    )
    workspace = Workspace(
        tmp_path / "capture",
        max_upload_bytes=20,
    )
    receipt = workspace.import_video(
        BytesIO(b"x" * 15),
        filename="solve.mov",
        source="import",
    )

    assert receipt["video"]["bytes"] == 15
    assert receipt["video"]["actual_fps"] == fps


def test_import_video_rejects_unknown_camera_facing(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)

    with pytest.raises(WorkspaceError, match="camera_facing"):
        workspace.import_video(
            BytesIO(b"video"),
            filename="solve.mp4",
            source="import",
            camera_facing="sideways",
        )


def test_native_ios_import_rejects_mirrored_evidence_before_writing(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)

    with pytest.raises(WorkspaceError, match="native iOS capture evidence must be unmirrored"):
        workspace.import_video(
            BytesIO(b"video"),
            filename="solve.mov",
            source="native-ios",
            mirrored=True,
        )

    assert list((tmp_path / "workspace" / "captures").iterdir()) == []


def test_native_ios_import_persists_exact_raw_axis_camera_intrinsics(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1080,
            "height": 1920,
            "rotation_degrees": 90,
        },
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    intrinsics = {
        "matrix": [
            [1200.0, 0.0, 960.0],
            [0.0, 1180.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        "ref_w": 1920,
        "ref_h": 1080,
    }

    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mov",
        source="native-ios",
        camera_facing="front",
        mirrored=False,
        camera_intrinsics=intrinsics,
        configured_fps=120,
    )

    assert receipt["camera"]["intrinsics"] == intrinsics
    assert receipt["video"]["configured_fps"] == 120
    assert receipt["video"]["rotation_degrees"] == 90
    stored = workspace.list_captures()[0]
    assert stored["camera"]["intrinsics"] == intrinsics


@pytest.mark.parametrize(
    ("source", "configured_fps", "message"),
    [
        ("import", 120, "reserved for native iOS"),
        ("native-ios", 59, "from 60 through 240"),
        ("native-ios", 241, "from 60 through 240"),
        ("native-ios", 120.0, "from 60 through 240"),
    ],
)
def test_import_rejects_unreceiptable_configured_camera_cadence(
    tmp_path,
    source,
    configured_fps,
    message,
) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)

    with pytest.raises(WorkspaceError, match=message):
        workspace.import_video(
            BytesIO(b"video"),
            filename="solve.mov",
            source=source,
            configured_fps=configured_fps,
        )


@pytest.mark.parametrize(
    "intrinsics",
    [
        {
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "ref_w": 1920,
            "ref_h": 1080,
            "distortion": [],
        },
        {
            "matrix": [[1.0]],
            "ref_w": 1920,
            "ref_h": 1080,
        },
        {
            "matrix": [[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]],
            "ref_w": 1920,
            "ref_h": 1080,
        },
        {
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "ref_w": 0,
            "ref_h": 1080,
        },
    ],
)
def test_native_ios_import_rejects_invalid_camera_intrinsics_before_writing(
    tmp_path,
    intrinsics,
) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)

    with pytest.raises(WorkspaceError, match="camera intrinsics"):
        workspace.import_video(
            BytesIO(b"video"),
            filename="solve.mov",
            source="native-ios",
            camera_intrinsics=intrinsics,
        )

    assert list((tmp_path / "workspace" / "captures").iterdir()) == []


def test_import_video_rejects_noncanonical_scramble(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    with pytest.raises(WorkspaceError, match="canonical face moves"):
        workspace.import_video(
            BytesIO(b"video"),
            filename="solve.mp4",
            source="import",
            scramble="banana",
        )


def test_sidecars_can_be_attached_before_sealing(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )

    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )
    assert updated["calibration"]["schema_version"] == 1
    assert updated["calibration"]["kind"] == "color_calibration_v1"
    assert updated["calibration"]["sample_count"] == 270
    assert "display_name" not in updated["calibration"]
    assert "calibration" not in updated["readiness"]["missing_for_decode"]

    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"
    assert sealed["seal_purpose"] == "decode"
    with pytest.raises(WorkspaceError, match="immutable"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b"{}"),
            kind="teacher",
        )


def test_label_locked_capture_rejects_calibration_replacement(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )
    workspace.seal_capture(receipt["capture_id"], purpose="label")

    with pytest.raises(WorkspaceError, match="immutable"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(color_calibration_payload).encode()),
            kind="calibration",
        )


def test_attach_reused_calibration_from_bundled_asset(
    tmp_path,
    monkeypatch,
    flat_color_centroids_payload,
) -> None:
    """Explicit demo-only reuse uses the same validation path as an upload."""

    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    repo_root = tmp_path / "repo"
    asset_dir = repo_root / "workspace" / "release-assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "calibration_gan12.json").write_text(json.dumps(flat_color_centroids_payload))

    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )

    updated = workspace.attach_reused_calibration(
        receipt["capture_id"],
        source="bundled",
        source_capture_id=None,
        repo_root=repo_root,
    )
    assert updated["calibration"]["kind"] == "color_centroids_v1"
    assert updated["calibration"]["display_name"] == "Published shared calibration"
    assert "calibration" not in updated["readiness"]["missing_for_decode"]

    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"


def test_attach_reused_calibration_missing_bundled_asset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )

    with pytest.raises(WorkspaceError, match="make download-decode-support"):
        workspace.attach_reused_calibration(
            receipt["capture_id"],
            source="bundled",
            source_capture_id=None,
            repo_root=tmp_path / "repo-without-assets",
        )


def test_attach_reused_calibration_from_another_capture(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    source_receipt = workspace.import_video(
        BytesIO(b"source video"),
        filename="source.mp4",
        source="import",
        scramble="R U R'",
    )
    workspace.attach_json_sidecar(
        source_receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )
    target_receipt = workspace.import_video(
        BytesIO(b"target video"),
        filename="target.mp4",
        source="import",
        scramble="F R U",
    )

    updated = workspace.attach_reused_calibration(
        target_receipt["capture_id"],
        source="capture",
        source_capture_id=source_receipt["capture_id"],
        repo_root=tmp_path,
    )
    assert updated["calibration"]["kind"] == "color_calibration_v1"
    assert updated["calibration"]["display_name"] == "From source.mp4"
    assert "calibration" not in updated["readiness"]["missing_for_decode"]

    # A third capture with no calibration of its own is not a valid reuse
    # source.
    bare_receipt = workspace.import_video(
        BytesIO(b"bare video"),
        filename="bare.mp4",
        source="import",
        scramble="U D",
    )
    with pytest.raises(WorkspaceError, match="no calibration attached"):
        workspace.attach_reused_calibration(
            source_receipt["capture_id"],
            source="capture",
            source_capture_id=bare_receipt["capture_id"],
            repo_root=tmp_path,
        )


def test_attach_reused_calibration_replaces_decode_locked_calibration(
    tmp_path,
    monkeypatch,
    flat_color_centroids_payload,
) -> None:
    """Decode locking freezes video/scramble while calibration stays versionable."""

    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    repo_root = tmp_path / "repo"
    asset_dir = repo_root / "workspace" / "release-assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "calibration_gan12.json").write_text(json.dumps(flat_color_centroids_payload))

    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )
    first = workspace.attach_reused_calibration(
        receipt["capture_id"],
        source="bundled",
        source_capture_id=None,
        repo_root=repo_root,
    )
    workspace.seal_capture(receipt["capture_id"], purpose="decode")

    replacement = {
        **flat_color_centroids_payload,
        "red": [
            flat_color_centroids_payload["red"][0] + 0.5,
            *flat_color_centroids_payload["red"][1:],
        ],
    }
    (asset_dir / "calibration_gan12.json").write_text(json.dumps(replacement))
    updated = workspace.attach_reused_calibration(
        receipt["capture_id"],
        source="bundled",
        source_capture_id=None,
        repo_root=repo_root,
    )

    assert updated["state"] == "sealed"
    assert updated["seal_purpose"] == "decode"
    assert updated["video"] == first["video"]
    assert updated["solve"] == first["solve"]
    assert updated["calibration"]["sha256"] != first["calibration"]["sha256"]


def test_flat_centroid_map_is_normalized_and_seals_for_decode(
    tmp_path,
    monkeypatch,
    flat_color_centroids_payload,
) -> None:
    """The released decode-support asset (a bare flat six-color Lab map) attaches
    as-is and satisfies seal-for-decode readiness, closing the flagship
    walkthrough gap."""

    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )
    uploaded = json.dumps(flat_color_centroids_payload).encode()

    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(uploaded),
        kind="calibration",
    )

    assert updated["calibration"]["kind"] == "color_centroids_v1"
    assert updated["calibration"]["schema"] == "cubed-core/color-centroids-v1"
    assert updated["calibration"]["schema_version"] == 1
    assert "sample_count" not in updated["calibration"]
    assert "calibration" not in updated["readiness"]["missing_for_decode"]

    stored = tmp_path / "workspace" / "captures" / receipt["capture_id"] / "calibration.json"
    stored_document = json.loads(stored.read_text())
    assert stored_document["schema"] == "cubed-core/color-centroids-v1"
    assert stored_document["schema_version"] == 1
    assert stored_document["color_space"] == "cielab"
    assert stored_document["centroids"] == flat_color_centroids_payload
    assert stored_document["provenance"] == (
        f"imported-centroids sha256:{hashlib.sha256(uploaded).hexdigest()}"
    )
    # The stored bytes are the normalized envelope, not the raw upload, so the
    # receipted sha256 must match what is actually on disk.
    assert updated["calibration"]["sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()

    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"
    assert sealed["seal_purpose"] == "decode"


def test_color_centroids_envelope_attaches_unchanged(
    tmp_path,
    monkeypatch,
    color_centroids_envelope_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )

    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_centroids_envelope_payload).encode()),
        kind="calibration",
    )

    assert updated["calibration"]["kind"] == "color_centroids_v1"
    assert updated["calibration"]["schema"] == "cubed-core/color-centroids-v1"
    assert "calibration" not in updated["readiness"]["missing_for_decode"]

    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"


def test_list_captures_recognizes_legacy_sampled_calibration_without_mutating_receipt(
    tmp_path,
    monkeypatch,
    color_centroids_envelope_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U",
    )
    document = {
        **color_centroids_envelope_payload,
        "provenance": (
            f"workspace-video-sticker-crops capture:{receipt['capture_id']} "
            f"video-sha256:{receipt['video']['sha256']} "
            f"crops-sha256:{'c' * 64}"
        ),
    }
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(document).encode()),
        kind="calibration",
    )
    receipt_path = tmp_path / "workspace" / "captures" / receipt["capture_id"] / "capture.json"
    assert "display_name" not in json.loads(receipt_path.read_text())["calibration"]

    listed = workspace.list_captures()[0]

    assert listed["calibration"]["display_name"] == "Sampled from this video"
    assert "display_name" not in json.loads(receipt_path.read_text())["calibration"]


def test_list_captures_does_not_guess_name_for_unbound_legacy_calibrations(
    tmp_path,
    monkeypatch,
    color_centroids_envelope_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U",
    )
    documents = [
        color_centroids_envelope_payload,
        {
            **color_centroids_envelope_payload,
            "provenance": (
                f"workspace-video-sticker-crops capture:{'f' * 32} "
                f"video-sha256:{receipt['video']['sha256']} "
                f"crops-sha256:{'c' * 64}"
            ),
        },
        {
            **color_centroids_envelope_payload,
            "provenance": (
                f"workspace-video-sticker-crops capture:{receipt['capture_id']} "
                f"video-sha256:{'e' * 64} crops-sha256:{'c' * 64}"
            ),
        },
    ]
    for document in documents:
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(document).encode()),
            kind="calibration",
        )
        assert "display_name" not in workspace.list_captures()[0]["calibration"]


def test_list_captures_requires_legacy_sampled_calibration_sha_match(
    tmp_path,
    monkeypatch,
    color_centroids_envelope_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "width": 1920, "height": 1080},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U",
    )
    document = {
        **color_centroids_envelope_payload,
        "provenance": (
            f"workspace-video-sticker-crops capture:{receipt['capture_id']} "
            f"video-sha256:{receipt['video']['sha256']} "
            f"crops-sha256:{'c' * 64}"
        ),
    }
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(document).encode()),
        kind="calibration",
    )
    calibration_path = (
        tmp_path / "workspace" / "captures" / receipt["capture_id"] / "calibration.json"
    )
    calibration_path.write_text(json.dumps({**document, "provenance": "changed"}))

    assert "display_name" not in workspace.list_captures()[0]["calibration"]


@pytest.mark.parametrize(
    "garbage",
    [
        # Only five colors: missing "green".
        {
            "white": [235.0, 128.0, 128.0],
            "red": [140.0, 190.0, 165.0],
            "blue": [100.0, 160.0, 70.0],
            "orange": [180.0, 170.0, 200.0],
            "yellow": [220.0, 115.0, 210.0],
        },
        # Wrong vector shape.
        {
            "white": [235.0, 128.0],
            "green": [140.0, 70.0, 155.0],
            "red": [140.0, 190.0, 165.0],
            "blue": [100.0, 160.0, 70.0],
            "orange": [180.0, 170.0, 200.0],
            "yellow": [220.0, 115.0, 210.0],
        },
        # Unrecognized schema declaration.
        {"schema": "cubed-core/not-a-real-schema", "schema_version": 1},
    ],
)
def test_garbage_calibration_uploads_are_rejected(
    tmp_path,
    monkeypatch,
    garbage,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U R'",
    )

    with pytest.raises(WorkspaceError):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(garbage).encode()),
            kind="calibration",
        )
    assert workspace.list_captures()[0]["calibration"] is None


def test_preview_calibration_format_can_differ_from_recorded_movie_format(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1920,
            "height": 1080,
        },
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="native-ios",
        camera_facing="back",
        mirrored=False,
    )

    assert color_calibration_payload["provenance"]["camera_format_width"] == 1280
    assert color_calibration_payload["provenance"]["camera_format_height"] == 720
    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )

    assert updated["video"]["encoded_width"] == 1920
    assert updated["video"]["encoded_height"] == 1080
    assert updated["calibration"]["kind"] == "color_calibration_v1"


def test_concurrent_sidecar_updates_keep_both_receipt_entries(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )

    def attach(kind: str, payload: dict[str, object]) -> None:
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(payload).encode()),
            kind=kind,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(attach, "calibration", color_calibration_payload),
            executor.submit(attach, "phone-imu", {"samples": [1, 2, 3]}),
        ]
        for future in futures:
            future.result()

    updated = workspace.list_captures()[0]
    assert updated["calibration"]["kind"] == "color_calibration_v1"
    assert updated["sensors"]["phone_imu"] is not None
    checksums = (
        tmp_path / "workspace" / "captures" / receipt["capture_id"] / "checksums.sha256"
    ).read_text()
    assert "calibration.json" in checksums
    assert "sensors/phone_imu.json" in checksums


def test_sidecar_commit_failure_restores_previous_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_dir = tmp_path / "workspace" / "captures" / receipt["capture_id"]
    before_receipt = (capture_dir / "capture.json").read_bytes()
    before_checksums = (capture_dir / "checksums.sha256").read_bytes()

    def fail_write(*args, **kwargs) -> None:
        raise OSError("injected metadata failure")

    monkeypatch.setattr(workspace, "_write_receipt", fail_write)
    with pytest.raises(OSError, match="injected"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"samples": [1]}'),
            kind="phone-imu",
        )

    assert not (capture_dir / "sensors" / "phone_imu.json").exists()
    assert (capture_dir / "capture.json").read_bytes() == before_receipt
    assert (capture_dir / "checksums.sha256").read_bytes() == before_checksums


def test_seal_commit_failure_restores_receipt_and_checksums(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_dir = workspace.captures_dir / receipt["capture_id"]
    before_receipt = (capture_dir / "capture.json").read_bytes()
    before_checksums = (capture_dir / "checksums.sha256").read_bytes()

    def fail_write(*args, **kwargs) -> None:
        raise OSError("injected metadata failure")

    monkeypatch.setattr(workspace, "_write_receipt", fail_write)
    with pytest.raises(OSError, match="injected"):
        workspace.seal_capture(receipt["capture_id"], purpose="label")

    assert (capture_dir / "capture.json").read_bytes() == before_receipt
    assert (capture_dir / "checksums.sha256").read_bytes() == before_checksums
    assert workspace.list_captures()[0]["state"] == "incomplete"


def test_ble_teacher_session_preserves_clock_and_event_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="native-ios",
        capture_session_id="session-1",
    )
    session = {
        "schema": "cubed-core/ble-session",
        "schema_version": 1,
        "ble_session_id": "ble-1",
        "capture_session_id": "session-1",
        "video_recording_id": receipt["capture_id"],
        "started_at": "2026-07-23T12:00:00Z",
        "ended_at": "2026-07-23T12:00:10Z",
        "device": {
            "name": "GAN 12 ui",
            "hardware_name": "GAN 12 ui",
            "hardware_version": None,
            "software_version": None,
            "product_date": None,
            "gyro_supported": True,
            "battery_start_percent": 90,
            "battery_end_percent": 89,
        },
        "clock": {
            "schema_version": 1,
            "monotonic_start_ms": 1000,
            "unix_start_ms": 1_800_000_000_000,
            "relative_timebase": "host_performance_now",
            "move_event_time": "event_local_timestamp_or_host_monotonic_fallback",
            "orientation_event_time": "host_monotonic_receive",
            "cross_device_alignment": "content_affine_required",
        },
        "zero_quat": [1.0, 0.0, 0.0, 0.0],
        "scramble": None,
        "moves": [
            {
                "sequence": 0,
                "t_ms": 12.5,
                "move": "R",
                "facelets": None,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "event_local_timestamp_ms": 1012.5,
                "event_host_timestamp_ms": 1012.7,
                "cube_timestamp_ms": 800,
                "serial": 3,
                "clock_source": "event_local_timestamp",
            }
        ],
        "orientations": [
            {
                "sequence": 1,
                "t_ms": 11.0,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "event_host_timestamp_ms": 1011.0,
                "clock_source": "host_monotonic_receive",
            }
        ],
        "states": [],
    }

    raw_updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(session).encode()),
        kind="ble-raw",
    )

    assert raw_updated["sensors"]["ble_raw"]["kind"] == "cubed_core_ble_session_v1"
    assert raw_updated["sensors"]["ble_raw"]["move_count"] == 1
    raw_path = (
        tmp_path
        / "workspace"
        / "captures"
        / receipt["capture_id"]
        / raw_updated["sensors"]["ble_raw"]["path"]
    )
    assert json.loads(raw_path.read_text())["zero_quat"] == [1.0, 0.0, 0.0, 0.0]
    compatibility = {
        "schema_version": 2,
        "clock": {
            "schema_version": 1,
            "monotonic_start_ms": 1000,
            "relative_timebase": "host_performance_now",
            "move_event_time": "event_local_timestamp_or_host_monotonic_fallback",
            "orientation_event_time": "host_monotonic_receive",
        },
        "recording_id": "ble-1",
        "capture_session_id": "session-1",
        "device": "GAN 12 ui",
        "started_unix_ms": 1_800_000_000_000,
        "zero_quat": [1.0, 0.0, 0.0, 0.0],
        "video_recording_id": receipt["capture_id"],
        "moves": [
            {
                "t_ms": 12.5,
                "move": "R",
                "facelets": None,
                "quat": [1.0, 0.0, 0.0, 0.0],
                "event_local_timestamp_ms": 1012.5,
                "cube_timestamp_ms": 800,
                "serial": 3,
                "clock_source": "event_local_timestamp",
            }
        ],
        "orientations": [
            {
                "t_ms": 11.0,
                "quat": [1.0, 0.0, 0.0, 0.0],
                "clock_source": "host_monotonic_receive",
            }
        ],
    }
    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(compatibility).encode()),
        kind="teacher",
    )
    assert updated["teacher"]["kind"] == "cube_session_v2"
    assert updated["teacher"]["move_count"] == 1
    assert updated["teacher"]["orientation_count"] == 1
    teacher_path = (
        tmp_path / "workspace" / "captures" / receipt["capture_id"] / updated["teacher"]["path"]
    )
    assert json.loads(teacher_path.read_text())["zero_quat"] == [1.0, 0.0, 0.0, 0.0]

    invalid_raw = {**session, "zero_quat": [0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(WorkspaceError, match="zero_quat must be a unit quaternion"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(invalid_raw).encode()),
            kind="ble-raw",
        )

    invalid_compatibility = {
        **compatibility,
        "zero_quat": [0.0, 0.0, 0.0, 0.0],
    }
    with pytest.raises(WorkspaceError, match="zero_quat must be a unit quaternion"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(invalid_compatibility).encode()),
            kind="teacher",
        )


def test_ble_sidecar_cannot_rewrite_capture_linkage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="native-ios",
        capture_session_id="capture-session",
        scramble="R U",
    )
    session = {
        "schema": "cubed-core/ble-session",
        "schema_version": 1,
        "ble_session_id": "ble-1",
        "capture_session_id": "other-session",
        "video_recording_id": receipt["capture_id"],
        "started_at": "2026-07-23T12:00:00Z",
        "ended_at": "2026-07-23T12:00:10Z",
        "device": {"name": "GAN 12 ui"},
        "clock": {
            "schema_version": 1,
            "monotonic_start_ms": 1000,
            "unix_start_ms": 1_800_000_000_000,
            "relative_timebase": "host_performance_now",
            "move_event_time": "event_local_timestamp_or_host_monotonic_fallback",
            "orientation_event_time": "host_monotonic_receive",
            "cross_device_alignment": "content_affine_required",
        },
        "scramble": "R U",
        "moves": [],
        "orientations": [],
        "states": [],
    }

    with pytest.raises(WorkspaceError, match="capture_session_id does not match"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(session).encode()),
            kind="ble-raw",
        )

    session["capture_session_id"] = "capture-session"
    session["scramble"] = None
    with pytest.raises(WorkspaceError, match="scramble does not match"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(session).encode()),
            kind="ble-raw",
        )

    unchanged = workspace.list_captures()[0]
    assert unchanged["capture_session_id"] == "capture-session"
    assert unchanged["solve"]["scramble"] == "R U"
    assert unchanged["sensors"]["ble_raw"] is None


def test_ble_teacher_session_rejects_mixed_clock_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="native-ios",
    )
    invalid = {
        "schema": "cubed-core/ble-session",
        "schema_version": 1,
        "ble_session_id": "ble-1",
        "capture_session_id": "session-1",
        "video_recording_id": None,
        "started_at": "2026-07-23T12:00:00Z",
        "ended_at": "2026-07-23T12:00:10Z",
        "device": {"name": "GAN 12 ui"},
        "clock": {
            "monotonic_start_ms": 1000,
            "unix_start_ms": 1_800_000_000_000,
            "relative_timebase": "unix_epoch",
            "cross_device_alignment": "content_affine_required",
        },
        "scramble": None,
        "moves": [],
        "orientations": [],
        "states": [],
    }

    with pytest.raises(WorkspaceError, match="clock contract"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps(invalid).encode()),
            kind="ble-raw",
        )


def test_teacher_slot_rejects_unversioned_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )

    with pytest.raises(WorkspaceError, match="CubeSession schema version 2"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"events": []}'),
            kind="teacher",
        )


def test_decode_seal_requires_calibration_and_scramble(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 60.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(BytesIO(b"video"), filename="solve.mp4", source="import")

    with pytest.raises(WorkspaceError, match="calibration, solve.scramble"):
        workspace.seal_capture(receipt["capture_id"], purpose="decode")

    assert workspace.seal_capture(receipt["capture_id"], purpose="label")["state"] == "sealed"


def test_decode_seal_allows_30_fps_capture_with_warning(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {
            "status": "ok",
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "capture_class": "unsupported",
        },
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
        scramble="R U",
    )
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )

    assert any("recommended 110–121 fps" in item for item in receipt["readiness"]["warnings"])
    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"


def test_decode_seal_rejects_native_240_until_a_derivative_is_imported(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {
            "status": "ok",
            "fps": 240.0,
            "width": 1920,
            "height": 1080,
            "capture_class": "native-240",
        },
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="native.mov",
        source="import",
        scramble="R U",
    )
    assert "video.derive-240-to-120" in receipt["readiness"]["missing_for_decode"]
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )

    with pytest.raises(WorkspaceError, match="derive-240-to-120"):
        workspace.seal_capture(receipt["capture_id"], purpose="decode")


def test_decode_seal_allows_video_below_1080_short_edge_with_warning(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1280,
            "height": 720,
            "capture_class": "target",
        },
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="low-resolution.mov",
        source="import",
        scramble="R U",
    )
    assert "video.short_edge>=1080" not in receipt["readiness"]["missing_for_decode"]
    assert any("recommended 1080-pixel" in item for item in receipt["readiness"]["warnings"])
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(json.dumps(color_calibration_payload).encode()),
        kind="calibration",
    )

    sealed = workspace.seal_capture(receipt["capture_id"], purpose="decode")
    assert sealed["state"] == "sealed"


def test_workspace_rejects_symlinked_capture_root(tmp_path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "captures").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="captures directory may not be a symlink"):
        Workspace(root, max_upload_bytes=1024).initialize()


def test_workspace_path_fallback_supports_capture_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    """Exercise the native-Windows path without requiring a Windows runner."""

    monkeypatch.setattr(workspace_module, "_USE_DIRECTORY_FDS", False)
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    original_open = workspace_module.os.open

    def reject_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        assert dir_fd is None
        if workspace_module.Path(path).is_dir():
            raise PermissionError("native Windows cannot os.open a directory")
        return original_open(path, flags, mode)

    monkeypatch.setattr(workspace_module.os, "open", reject_directory_open)
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_id = receipt["capture_id"]

    assert workspace.list_captures()[0]["capture_id"] == capture_id
    assert workspace.capture_video_path(capture_id).read_bytes() == b"video"
    workspace.write_frame_annotations(capture_id, b"labels")
    assert workspace.read_frame_annotations(capture_id, maximum_bytes=1024) == b"labels"

    original_replace = workspace_module.os.replace
    failed_annotation_move = False

    def fail_path_annotation_move(source, destination) -> None:
        nonlocal failed_annotation_move
        if (
            not failed_annotation_move
            and workspace_module.Path(source).name == capture_id
            and workspace_module.Path(destination).name == "annotations"
        ):
            failed_annotation_move = True
            raise OSError("injected Windows-path annotation move failure")
        original_replace(source, destination)

    monkeypatch.setattr(workspace_module.os, "replace", fail_path_annotation_move)
    with pytest.raises(WorkspaceError, match="could not be moved to Trash"):
        workspace.trash_capture(capture_id)
    assert (workspace.captures_dir / capture_id / "capture.json").is_file()
    assert (
        workspace.annotations_dir / capture_id / "frame-annotations.json"
    ).read_bytes() == b"labels"
    monkeypatch.setattr(workspace_module.os, "replace", original_replace)

    system_trash = tmp_path / "system-trash"
    system_trash.mkdir()

    def fake_send2trash(value: str) -> None:
        source = workspace_module.Path(value)
        source.rename(system_trash / source.name)

    monkeypatch.setattr(workspace_module, "send2trash", fake_send2trash)
    assert workspace.trash_capture(capture_id)["trashed"] is True
    assert workspace.list_captures() == []


def test_workspace_path_fallback_rejects_symlinks_and_windows_reparse_points(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(workspace_module, "_USE_DIRECTORY_FDS", False)
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "captures").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="captures directory may not be a symlink"):
        Workspace(root, max_upload_bytes=1024).initialize()

    fake_reparse_stat = type(
        "FakeWindowsStat",
        (),
        {
            "st_mode": workspace_module.stat.S_IFDIR,
            "st_file_attributes": workspace_module._WINDOWS_REPARSE_POINT,
        },
    )()
    assert workspace_module._is_reparse_point(fake_reparse_stat)


def test_temporary_replace_rejects_regular_mode_windows_reparse_point(
    tmp_path,
    monkeypatch,
) -> None:
    fake_reparse_stat = type(
        "FakeWindowsRegularReparseStat",
        (),
        {
            "st_mode": workspace_module.stat.S_IFREG,
            "st_file_attributes": workspace_module._WINDOWS_REPARSE_POINT,
        },
    )()
    monkeypatch.setattr(
        workspace_module,
        "_entry_stat",
        lambda directory_fd, name: fake_reparse_stat if name == "source.tmp" else None,
    )

    with pytest.raises(WorkspaceError, match="temporary file is unavailable"):
        workspace_module._replace_temporary_entry(
            tmp_path,
            "source.tmp",
            "target.json",
            description="workspace test",
        )


def test_annotation_storage_rejects_symlinked_parent_and_ignores_legacy_temp_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_id = receipt["capture_id"]
    outside = tmp_path / "outside"
    outside.mkdir()
    annotation_parent = workspace.annotations_dir / capture_id
    annotation_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceError, match="directory is unavailable"):
        workspace.read_frame_annotations(capture_id, maximum_bytes=1024)
    with pytest.raises(WorkspaceError, match="directory may not be a symlink"):
        workspace.write_frame_annotations(capture_id, b"{}")

    annotation_parent.unlink()
    annotation_parent.mkdir()
    outside_target = tmp_path / "outside-target"
    outside_target.write_bytes(b"unchanged")
    (annotation_parent / ".frame-annotations.json.part").symlink_to(outside_target)
    workspace.write_frame_annotations(capture_id, b"changed")
    assert outside_target.read_bytes() == b"unchanged"
    assert workspace.read_frame_annotations(capture_id, maximum_bytes=1024) == b"changed"

    (annotation_parent / "frame-annotations.json").unlink()
    (annotation_parent / "frame-annotations.json").symlink_to(outside_target)
    with pytest.raises(WorkspaceError, match="unavailable"):
        workspace.read_frame_annotations(capture_id, maximum_bytes=1024)
    with pytest.raises(WorkspaceError, match="target may not be a symlink"):
        workspace.write_frame_annotations(capture_id, b"new")
    assert outside_target.read_bytes() == b"unchanged"


def test_trash_capture_moves_bundle_and_annotations_to_system_trash(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_id = receipt["capture_id"]
    source_video = tmp_path / "hf-dataset" / "source.mp4"
    source_video.parent.mkdir()
    source_video.write_bytes(b"video")
    workspace_video = workspace.capture_video_path(capture_id)
    workspace_video.unlink()
    workspace_module.os.link(source_video, workspace_video)
    workspace.write_frame_annotations(capture_id, b"labels")
    system_trash = tmp_path / "system-trash"
    system_trash.mkdir()

    def fake_send2trash(value: str) -> None:
        source = workspace_module.Path(value)
        source.rename(system_trash / source.name)

    monkeypatch.setattr(workspace_module, "send2trash", fake_send2trash)

    result = workspace.trash_capture(capture_id)

    assert result == {
        "schema": "cubed-core/capture-delete-v1",
        "schema_version": 1,
        "capture_id": capture_id,
        "trashed": True,
        "recoverable": True,
    }
    destination = next(system_trash.iterdir())
    assert not (workspace.captures_dir / capture_id).exists()
    assert not (workspace.annotations_dir / capture_id).exists()
    assert (destination / "capture" / "capture.json").is_file()
    assert (destination / "annotations" / "frame-annotations.json").read_bytes() == b"labels"
    assert source_video.read_bytes() == b"video"
    assert workspace_module.os.path.samefile(
        source_video,
        destination / "capture" / workspace_video.name,
    )
    assert workspace.list_captures() == []


def test_trash_capture_rejects_unknown_and_symlinked_entries_without_following_them(
    tmp_path,
) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    workspace.initialize()
    with pytest.raises(WorkspaceError, match="capture not found"):
        workspace.trash_capture("a" * 32)

    outside = tmp_path / "outside-capture"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"unchanged")
    (workspace.captures_dir / ("b" * 32)).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(WorkspaceError, match="workspace entry is invalid"):
        workspace.trash_capture("b" * 32)

    assert sentinel.read_bytes() == b"unchanged"
    assert not list(workspace.root.glob("Cubed Core capture *"))


def test_trash_capture_rolls_back_capture_when_annotation_move_fails(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_id = receipt["capture_id"]
    workspace.write_frame_annotations(capture_id, b"labels")
    original_replace = workspace_module.os.replace
    failed = False

    def fail_annotation_move(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ) -> None:
        nonlocal failed
        if not failed and source == capture_id and destination == "annotations":
            failed = True
            raise OSError("injected annotation move failure")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(workspace_module.os, "replace", fail_annotation_move)
    with pytest.raises(WorkspaceError, match="could not be moved to Trash"):
        workspace.trash_capture(capture_id)

    assert (workspace.captures_dir / capture_id / "capture.json").is_file()
    assert (
        workspace.annotations_dir / capture_id / "frame-annotations.json"
    ).read_bytes() == b"labels"
    assert not list(workspace.root.glob("Cubed Core capture *"))
    assert workspace.list_captures()[0]["capture_id"] == capture_id


def test_trash_capture_rolls_back_when_system_trash_rejects(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_id = receipt["capture_id"]
    workspace.write_frame_annotations(capture_id, b"labels")

    def reject_trash(value: str) -> None:
        raise OSError("injected Trash failure")

    monkeypatch.setattr(workspace_module, "send2trash", reject_trash)
    with pytest.raises(WorkspaceError, match="could not be moved to Trash"):
        workspace.trash_capture(capture_id)

    assert (workspace.captures_dir / capture_id / "capture.json").is_file()
    assert (
        workspace.annotations_dir / capture_id / "frame-annotations.json"
    ).read_bytes() == b"labels"
    assert not list(workspace.root.glob("Cubed Core capture *"))


def test_legacy_receipt_and_checksum_temp_symlinks_cannot_clobber_outside_files(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_dir = workspace.captures_dir / receipt["capture_id"]
    outside_metadata = tmp_path / "outside-metadata"
    outside_checksums = tmp_path / "outside-checksums"
    outside_metadata.write_bytes(b"metadata-sentinel")
    outside_checksums.write_bytes(b"checksums-sentinel")
    (capture_dir / ".capture.json.part").symlink_to(outside_metadata)
    (capture_dir / ".checksums.sha256.part").symlink_to(outside_checksums)

    updated = workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(b'{"samples": [1]}'),
        kind="phone-imu",
    )

    assert updated["sensors"]["phone_imu"] is not None
    assert outside_metadata.read_bytes() == b"metadata-sentinel"
    assert outside_checksums.read_bytes() == b"checksums-sentinel"
    assert not (capture_dir / "capture.json").is_symlink()
    assert not (capture_dir / "checksums.sha256").is_symlink()


def test_legacy_sidecar_temp_symlink_cannot_clobber_outside_file(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    sensors_dir = workspace.captures_dir / receipt["capture_id"] / "sensors"
    sensors_dir.mkdir()
    outside = tmp_path / "outside-sidecar"
    outside.write_bytes(b"sidecar-sentinel")
    (sensors_dir / ".phone_imu.json.part").symlink_to(outside)

    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(b'{"samples": [1]}'),
        kind="phone-imu",
    )

    assert outside.read_bytes() == b"sidecar-sentinel"
    assert json.loads((sensors_dir / "phone_imu.json").read_bytes()) == {"samples": [1]}


def test_sidecar_rollback_ignores_legacy_rollback_symlink_and_restores_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    workspace.attach_json_sidecar(
        receipt["capture_id"],
        BytesIO(b'{"samples": [1]}'),
        kind="phone-imu",
    )
    capture_dir = workspace.captures_dir / receipt["capture_id"]
    sensors_dir = capture_dir / "sensors"
    before_sidecar = (sensors_dir / "phone_imu.json").read_bytes()
    before_receipt = (capture_dir / "capture.json").read_bytes()
    before_checksums = (capture_dir / "checksums.sha256").read_bytes()
    outside = tmp_path / "outside-rollback"
    outside.write_bytes(b"rollback-sentinel")
    (sensors_dir / ".phone_imu.json.rollback").symlink_to(outside)

    def fail_write(*args, **kwargs) -> None:
        raise OSError("injected metadata failure")

    monkeypatch.setattr(workspace, "_write_receipt", fail_write)
    with pytest.raises(OSError, match="injected"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"samples": [2]}'),
            kind="phone-imu",
        )

    assert outside.read_bytes() == b"rollback-sentinel"
    assert (sensors_dir / "phone_imu.json").read_bytes() == before_sidecar
    assert (capture_dir / "capture.json").read_bytes() == before_receipt
    assert (capture_dir / "checksums.sha256").read_bytes() == before_checksums


def test_sidecar_rejects_symlinked_parent_and_target_without_writing_outside(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_dir = workspace.captures_dir / receipt["capture_id"]
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    sensors_dir = capture_dir / "sensors"
    sensors_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="directory may not be a symlink"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"samples": [1]}'),
            kind="phone-imu",
        )
    assert list(outside_dir.iterdir()) == []

    sensors_dir.unlink()
    sensors_dir.mkdir()
    outside_target = tmp_path / "outside-target-sidecar"
    outside_target.write_bytes(b"target-sentinel")
    (sensors_dir / "phone_imu.json").symlink_to(outside_target)
    with pytest.raises(WorkspaceError, match="unavailable"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"samples": [1]}'),
            kind="phone-imu",
        )
    assert outside_target.read_bytes() == b"target-sentinel"


@pytest.mark.parametrize("bundle_name", ["capture.json", "checksums.sha256"])
def test_sidecar_rejects_symlinked_bundle_file_without_clobbering_outside(
    tmp_path,
    monkeypatch,
    bundle_name,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="import",
    )
    capture_dir = workspace.captures_dir / receipt["capture_id"]
    outside = tmp_path / f"outside-{bundle_name}"
    outside.write_bytes(b"bundle-sentinel")
    (capture_dir / bundle_name).unlink()
    (capture_dir / bundle_name).symlink_to(outside)

    with pytest.raises(WorkspaceError, match="unavailable"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(b'{"samples": [1]}'),
            kind="phone-imu",
        )

    assert outside.read_bytes() == b"bundle-sentinel"
    assert not (capture_dir / "sensors" / "phone_imu.json").exists()
