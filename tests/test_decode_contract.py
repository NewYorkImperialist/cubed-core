from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cubed_core.decode_api import router
from cubed_core.decode_contract import (
    build_decode_preflight,
    load_runtime_manifest,
    posix_cksum,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDING_ID = "a" * 32


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _calibration_payload() -> dict[str, object]:
    centroids = {
        "white": [235.0, 128.0, 128.0],
        "green": [140.0, 70.0, 155.0],
        "red": [140.0, 190.0, 165.0],
        "blue": [100.0, 160.0, 70.0],
        "orange": [180.0, 170.0, 200.0],
        "yellow": [220.0, 115.0, 210.0],
    }
    return {
        "schema": "cubed-core/color-calibration-v1",
        "schema_version": 1,
        "color_space": "opencv_bgr_lab_uint8_v1",
        "geometry": {
            "target_square_fraction": 0.56,
            "grid_rows": 3,
            "grid_columns": 3,
            "central_patch_fraction": 0.40,
            "coordinate_space": "rotated_video_frame_pixels",
        },
        "thresholds": {
            "minimum_l": 40.0,
            "uniformity_maximum": 28.0,
            "stability_maximum": 10.0,
            "collision_minimum": 15.0,
            "distance_weights": [0.15, 1.0, 1.0],
        },
        "order": ["white", "green", "red", "blue", "orange", "yellow"],
        "frames_per_color": 5,
        "samples_per_frame": 9,
        "centroids": centroids,
        "samples": {
            color: [list(centroid) for _ in range(45)] for color, centroid in centroids.items()
        },
        "provenance": {
            "captured_unix_ms": 1_800_000_010_000.0,
            "collection_started_unix_ms": 1_800_000_000_000.0,
            "camera_facing": "back",
            "mirrored": False,
            "frame_width": 1920,
            "frame_height": 1080,
            "camera_device_type": "builtInWideAngleCamera",
            "camera_format_width": 1920,
            "camera_format_height": 1080,
            "exposure_duration_seconds": 0.001,
            "iso": 100.0,
            "white_balance_gains": [1.0, 1.0, 1.0],
            "lens_position": 0.5,
            "exposure_locked": True,
            "white_balance_locked": True,
            "focus_locked": True,
            "app": "cubed-capture-ios",
            "sampler": "server_grid_opencv_lab_v1",
        },
    }


def _centroids_payload() -> dict[str, object]:
    return {
        "schema": "cubed-core/color-centroids-v1",
        "schema_version": 1,
        "color_space": "cielab",
        "centroids": {
            "red": [114.5, 195.682, 170.477],
            "blue": [98.622, 132.489, 83.889],
            "green": [159.6, 75.156, 154.378],
            "white": [205.143, 129.381, 135.595],
            "orange": [143.178, 182.489, 189.244],
            "yellow": [199.333, 118.289, 202.022],
        },
        "provenance": "imported-centroids sha256:" + "a" * 64,
    }


def _capture(
    tmp_path: Path,
    *,
    fps: float = 120.0,
    width: int = 1920,
    height: int = 1080,
    sealed: bool = True,
    video_sha256: str | None = None,
    calibration_bytes: bytes | None = None,
) -> tuple[dict[str, object], Path]:
    workspace = tmp_path / "workspace"
    capture_dir = workspace / "captures" / RECORDING_ID
    capture_dir.mkdir(parents=True)
    video = b"video"
    calibration = (
        calibration_bytes
        if calibration_bytes is not None
        else json.dumps(_calibration_payload()).encode()
    )
    (capture_dir / "source.mov").write_bytes(video)
    (capture_dir / "calibration.json").write_bytes(calibration)
    receipt: dict[str, object] = {
        "schema": "cubed-core/capture-bundle",
        "schema_version": 1,
        "recording_id": RECORDING_ID,
        "capture_id": RECORDING_ID,
        "state": "sealed" if sealed else "incomplete",
        "seal_purpose": "decode" if sealed else None,
        "video": {
            "path": f"captures/{RECORDING_ID}/source.mov",
            "bytes": len(video),
            "sha256": video_sha256 or _sha256(video),
            "actual_fps": fps,
            "encoded_width": width,
            "encoded_height": height,
        },
        "solve": {"scramble": "R U R'"},
        "calibration": {
            "path": "calibration.json",
            "sha256": _sha256(calibration),
        },
        "teacher": None,
    }
    (capture_dir / "capture.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt, workspace


def _check(report: dict[str, object], check_id: str) -> dict[str, str]:
    return next(item for item in report["checks"] if item["id"] == check_id)


@pytest.fixture
def runtime_ready_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo_root = tmp_path / "runtime-ready-repo"
    manifest_path = REPO_ROOT / "config" / "decode-runtime-v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_manifest = repo_root / "config" / "decode-runtime-v1.json"
    copied_manifest.parent.mkdir(parents=True)
    copied_manifest.write_bytes(manifest_path.read_bytes())
    for requirement in manifest["runtime_requirements"]:
        relative_path = Path(requirement["path"])
        candidate = repo_root / relative_path
        if (REPO_ROOT / relative_path).is_dir():
            candidate.mkdir(parents=True)
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"test runtime artifact")
    monkeypatch.setattr(
        "cubed_core.decode_contract.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    return repo_root


def test_manifest_cfg_hashes_match_posix_cksum() -> None:
    manifest = load_runtime_manifest(REPO_ROOT)
    profiles = manifest["profiles"]
    requirements = {
        requirement["id"]: requirement["path"] for requirement in manifest["runtime_requirements"]
    }
    assert profiles["canonical_eval_reference"]["cfg_hash"] == "3894620704"
    assert profiles["local_camera_v1"]["cfg_hash"] == "2429939321"
    assert requirements["pose-model"] == (
        "workspace/release-assets/camera-tracker-v1-runtime/artifacts/face-pose.onnx"
    )
    assert requirements["alignment-model"] == (
        "workspace/release-assets/camera-tracker-v1-runtime/artifacts/alignment-classifier.onnx"
    )
    assert requirements["read-trust-model"] == ("workspace/release-assets/trust_v1_numpy.npz")
    for profile in profiles.values():
        payload = profile["cfg_hash_input"].encode()
        assert str(posix_cksum(payload)) == profile["cfg_hash"]
        system = subprocess.run(
            ["cksum"],
            input=payload,
            capture_output=True,
            check=True,
        )
        assert system.stdout.decode().split()[0] == profile["cfg_hash"]


def test_preflight_accepts_capture_receipts_but_fails_closed_on_runtime(tmp_path) -> None:
    capture, workspace = _capture(tmp_path)

    # A synthetic repo root with the real code and manifest but no downloaded
    # release assets, so the missing-model expectation holds no matter what
    # the developer's own workspace/release-assets contains.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for name in ("config", "core", "analysis", "scripts"):
        source = REPO_ROOT / name
        if source.exists():
            (repo_root / name).symlink_to(source)
    # The decoder-engine requirement points at the detect/ directory itself
    # and the check rejects symlinked leaves, so give it a real directory.
    (repo_root / "detect").mkdir()

    report = build_decode_preflight(
        capture,
        repo_root=repo_root,
        workspace_root=workspace,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["execution_allowed"] is False
    assert report["config"]["cfg_hash"] == "2429939321"
    assert _check(report, "capture.video")["status"] == "pass"
    assert _check(report, "capture.calibration")["status"] == "pass"
    teacher_check = _check(report, "teacher.containment")
    assert teacher_check["status"] == "pass"
    assert "capture metadata declares no teacher record" in teacher_check["detail"]
    assert "runtime.execution" not in report["missing"]
    assert "runtime.decoder-engine" not in report["missing"]
    assert "runtime.pose-model" in report["missing"]
    assert "runtime.alignment-model" in report["missing"]


def test_preflight_rejects_receipt_hash_mismatch(tmp_path) -> None:
    capture, workspace = _capture(tmp_path, video_sha256="0" * 64)

    report = build_decode_preflight(
        capture,
        repo_root=REPO_ROOT,
        workspace_root=workspace,
    )

    assert _check(report, "capture.video") == {
        "id": "capture.video",
        "status": "fail",
        "detail": "video SHA-256 does not match its receipt",
        "path": f"captures/{RECORDING_ID}/source.mov",
    }


def test_preflight_accepts_color_centroids_v1_calibration(tmp_path) -> None:
    """The additive imported-centroids format satisfies capture.calibration."""

    capture, workspace = _capture(
        tmp_path,
        calibration_bytes=json.dumps(_centroids_payload()).encode(),
    )

    report = build_decode_preflight(
        capture,
        repo_root=REPO_ROOT,
        workspace_root=workspace,
    )

    assert _check(report, "capture.calibration")["status"] == "pass"


def test_preflight_rejects_calibration_missing_required_centroid(tmp_path) -> None:
    broken = _centroids_payload()
    del broken["centroids"]["green"]
    capture, workspace = _capture(
        tmp_path,
        calibration_bytes=json.dumps(broken).encode(),
    )

    report = build_decode_preflight(
        capture,
        repo_root=REPO_ROOT,
        workspace_root=workspace,
    )

    calibration_check = _check(report, "capture.calibration")
    assert calibration_check["status"] == "fail"
    assert "calibration sidecar is invalid" in calibration_check["detail"]


def test_preflight_rejects_calibration_with_unrecognized_schema(tmp_path) -> None:
    capture, workspace = _capture(
        tmp_path,
        calibration_bytes=json.dumps({"schema": "cubed-core/not-a-real-schema"}).encode(),
    )

    report = build_decode_preflight(
        capture,
        repo_root=REPO_ROOT,
        workspace_root=workspace,
    )

    calibration_check = _check(report, "capture.calibration")
    assert calibration_check["status"] == "fail"
    assert "color-calibration-v1 or color-centroids-v1" in calibration_check["detail"]


def test_preflight_warns_for_60_fps_but_allows_decode(
    tmp_path,
    runtime_ready_repo,
) -> None:
    capture, workspace = _capture(tmp_path, fps=60.0)

    report = build_decode_preflight(
        capture,
        repo_root=runtime_ready_repo,
        workspace_root=workspace,
    )

    frame_rate = _check(report, "capture.frame-rate")
    assert frame_rate["status"] == "warning"
    assert report["ready"] is True
    assert "decode is allowed" in frame_rate["detail"]
    assert "110 through 121 fps" in frame_rate["detail"]


def test_preflight_requires_a_linked_120_fps_derivative_for_native_240(tmp_path) -> None:
    capture, workspace = _capture(tmp_path, fps=240.0)

    report = build_decode_preflight(
        capture,
        repo_root=REPO_ROOT,
        workspace_root=workspace,
    )

    frame_rate = _check(report, "capture.frame-rate")
    assert frame_rate["status"] == "fail"
    assert "220–242 fps" in frame_rate["detail"]
    assert "derive-240-to-120" in frame_rate["detail"]


def test_preflight_warns_for_video_below_the_recommended_resolution(
    tmp_path,
    runtime_ready_repo,
) -> None:
    capture, workspace = _capture(tmp_path, width=1280, height=720)

    report = build_decode_preflight(
        capture,
        repo_root=runtime_ready_repo,
        workspace_root=workspace,
    )

    resolution = _check(report, "capture.resolution")
    assert resolution["status"] == "warning"
    assert report["ready"] is True
    assert "recommended 1080-pixel" in resolution["detail"]


@pytest.mark.parametrize("fps", [219.999, 242.001])
def test_preflight_warns_for_nonstandard_high_frame_rates(
    tmp_path,
    runtime_ready_repo,
    fps: float,
) -> None:
    capture, workspace = _capture(tmp_path, fps=fps)

    report = build_decode_preflight(
        capture,
        repo_root=runtime_ready_repo,
        workspace_root=workspace,
    )

    frame_rate = _check(report, "capture.frame-rate")
    assert frame_rate["status"] == "warning"
    assert report["ready"] is True
    assert "110 through 121 fps" in frame_rate["detail"]


def test_decode_router_exposes_specs_and_read_only_preflight(tmp_path) -> None:
    capture, workspace_root = _capture(tmp_path)
    schema_dir = tmp_path / "repo" / "schemas"
    schema_dir.mkdir(parents=True)
    for name in (
        "decode-result-v1.schema.json",
        "decode-preflight-v1.schema.json",
    ):
        (schema_dir / name).write_bytes((REPO_ROOT / "schemas" / name).read_bytes())
    config_dir = tmp_path / "repo" / "config"
    config_dir.mkdir()
    (config_dir / "decode-runtime-v1.json").write_bytes(
        (REPO_ROOT / "config" / "decode-runtime-v1.json").read_bytes()
    )

    class StubWorkspace:
        def list_captures(self):
            return [capture]

    app = FastAPI()
    app.state.settings = SimpleNamespace(
        repo_root=tmp_path / "repo",
        workspace=workspace_root,
    )
    app.state.workspace = StubWorkspace()
    app.include_router(router)

    with TestClient(app) as client:
        result_schema = client.get("/api/specs/decode-result")
        assert result_schema.status_code == 200
        assert result_schema.json()["properties"]["profile"]["const"] == "local_camera_v1"
        preflight_schema = client.get("/api/specs/decode-preflight")
        assert preflight_schema.status_code == 200
        response = client.get(f"/api/captures/{RECORDING_ID}/decode-preflight")
        assert response.status_code == 200
        assert response.json()["status"] == "blocked"
        assert client.get(f"/api/captures/{'b' * 32}/decode-preflight").status_code == 404
