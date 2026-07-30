from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cubed_core.app import CALIBRATION_CROPS_REQUEST_MAX_BYTES, create_app
from cubed_core.color_calibration import COLOR_ORDER, validate_color_centroids
from cubed_core.desktop_calibration import (
    MAX_CROP_BYTES,
    DesktopCalibrationError,
    build_centroids_document_from_crops,
)
from cubed_core.settings import Settings

cv2 = pytest.importorskip("cv2")
numpy = pytest.importorskip("numpy")

ADMIN_TOKEN = "test-admin-token-that-is-long-enough"
ADMIN_HEADERS = {"X-Cubed-Admin-Token": ADMIN_TOKEN}

_BGR_COLORS = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "orange": (0, 128, 255),
    "yellow": (0, 255, 255),
}


def _png(
    bgr: tuple[int, int, int],
    *,
    width: int = 96,
    height: int = 96,
) -> bytes:
    image = numpy.empty((height, width, 3), dtype=numpy.uint8)
    image[:, :] = bgr
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    return payload.tobytes()


def _crops() -> dict[str, bytes]:
    return {color: _png(_BGR_COLORS[color]) for color in COLOR_ORDER}


def _client(tmp_path) -> TestClient:
    settings = Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_TOKEN,
    )
    return TestClient(create_app(settings), headers=ADMIN_HEADERS)


def _import_capture(client: TestClient, monkeypatch) -> dict[str, object]:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 1200,
            "capture_class": "target",
        },
    )
    response = client.post(
        "/api/captures/import",
        data={"scramble": "R U"},
        files={"video": ("solve.mp4", b"video", "video/mp4")},
    )
    assert response.status_code == 201
    return response.json()


def _multipart_crops(crops: dict[str, bytes]) -> dict[str, tuple[str, bytes, str]]:
    return {color: (f"{color}.png", crops[color], "image/png") for color in COLOR_ORDER}


def test_build_centroids_document_uses_existing_contract_and_binds_provenance() -> None:
    document = build_centroids_document_from_crops(
        _crops(),
        capture_id="a" * 32,
        video_sha256="b" * 64,
    )

    assert document["schema"] == "cubed-core/color-centroids-v1"
    assert document["color_space"] == "cielab"
    assert set(document["centroids"]) == set(COLOR_ORDER)
    assert "capture:" + "a" * 32 in document["provenance"]
    assert "video-sha256:" + "b" * 64 in document["provenance"]
    assert "crops-sha256:" in document["provenance"]
    validate_color_centroids(document)


def test_crop_quality_rejects_dark_nonuniform_and_colliding_samples() -> None:
    dark = _crops()
    dark["white"] = _png((0, 0, 0))
    with pytest.raises(DesktopCalibrationError, match="white crop is too dark"):
        build_centroids_document_from_crops(
            dark,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )

    nonuniform = _crops()
    image = numpy.empty((96, 96, 3), dtype=numpy.uint8)
    image[:, :48] = (0, 0, 255)
    image[:, 48:] = (0, 255, 0)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    nonuniform["white"] = payload.tobytes()
    with pytest.raises(DesktopCalibrationError, match="white crop includes more than one color"):
        build_centroids_document_from_crops(
            nonuniform,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )

    colliding = _crops()
    colliding["green"] = colliding["white"]
    with pytest.raises(DesktopCalibrationError, match="too close"):
        build_centroids_document_from_crops(
            colliding,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )


def test_crop_contract_rejects_lossy_small_and_oversized_images() -> None:
    lossy = _crops()
    image = numpy.empty((96, 96, 3), dtype=numpy.uint8)
    image[:, :] = _BGR_COLORS["white"]
    encoded, payload = cv2.imencode(".jpg", image)
    assert encoded
    lossy["white"] = payload.tobytes()
    with pytest.raises(DesktopCalibrationError, match="lossless PNG"):
        build_centroids_document_from_crops(
            lossy,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )

    small = _crops()
    small["white"] = _png(_BGR_COLORS["white"], width=23, height=24)
    with pytest.raises(DesktopCalibrationError, match="24 through 512"):
        build_centroids_document_from_crops(
            small,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )

    oversized = _crops()
    oversized["white"] = b"\x89PNG\r\n\x1a\n" + b"x" * MAX_CROP_BYTES
    with pytest.raises(DesktopCalibrationError, match="2 MiB"):
        build_centroids_document_from_crops(
            oversized,
            capture_id="a" * 32,
            video_sha256="b" * 64,
        )


def test_from_crops_api_attaches_calibration_to_imported_capture(
    tmp_path,
    monkeypatch,
) -> None:
    with _client(tmp_path) as client:
        imported = _import_capture(client, monkeypatch)
        capture_id = imported["capture_id"]
        response = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=_multipart_crops(_crops()),
        )
        listed = client.get("/api/captures").json()["captures"]

    assert response.status_code == 200
    attached = response.json()
    assert attached["calibration"]["kind"] == "color_centroids_v1"
    assert attached["calibration"]["display_name"] == "Sampled from this video"
    persisted = next(row for row in listed if row["capture_id"] == capture_id)
    assert persisted["calibration"]["display_name"] == "Sampled from this video"
    assert "calibration" not in attached["readiness"]["missing_for_decode"]
    stored_path = tmp_path / "workspace" / "captures" / capture_id / "calibration.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored["schema"] == "cubed-core/color-centroids-v1"
    assert f"capture:{capture_id}" in stored["provenance"]
    assert f"video-sha256:{attached['video']['sha256']}" in stored["provenance"]


def test_from_crops_api_fails_without_mutating_capture(
    tmp_path,
    monkeypatch,
) -> None:
    crops = _crops()
    crops["yellow"] = crops["white"]
    with _client(tmp_path) as client:
        imported = _import_capture(client, monkeypatch)
        capture_id = imported["capture_id"]
        response = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=_multipart_crops(crops),
        )
        missing = client.post(
            f"/api/captures/{'f' * 32}/calibration/from-crops",
            files=_multipart_crops(_crops()),
        )

    assert response.status_code == 400
    assert "too close" in response.json()["detail"]
    assert not (tmp_path / "workspace" / "captures" / capture_id / "calibration.json").exists()
    assert missing.status_code == 404


def test_from_crops_request_guard_rejects_oversized_total_body(
    tmp_path,
    monkeypatch,
) -> None:
    crops = _crops()
    crops["white"] = b"x" * CALIBRATION_CROPS_REQUEST_MAX_BYTES
    with _client(tmp_path) as client:
        imported = _import_capture(client, monkeypatch)
        capture_id = imported["capture_id"]
        response = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=_multipart_crops(crops),
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "request body exceeds the configured upload limit"}
    assert not (tmp_path / "workspace" / "captures" / capture_id / "calibration.json").exists()


def test_from_crops_api_requires_every_color_without_mutating_capture(
    tmp_path,
    monkeypatch,
) -> None:
    files = _multipart_crops(_crops())
    del files["yellow"]
    with _client(tmp_path) as client:
        imported = _import_capture(client, monkeypatch)
        capture_id = imported["capture_id"]
        response = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=files,
        )

    assert response.status_code == 422
    assert any(error.get("loc") == ["body", "yellow"] for error in response.json()["detail"])
    assert not (tmp_path / "workspace" / "captures" / capture_id / "calibration.json").exists()


def test_from_crops_api_replaces_calibration_after_decode_lock(
    tmp_path,
    monkeypatch,
) -> None:
    files = _multipart_crops(_crops())
    with _client(tmp_path) as client:
        imported = _import_capture(client, monkeypatch)
        capture_id = imported["capture_id"]
        attached = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=files,
        )
        assert attached.status_code == 200
        sealed = client.post(
            f"/api/captures/{capture_id}/seal",
            data={"purpose": "decode"},
        )
        assert sealed.status_code == 200
        replaced = client.post(
            f"/api/captures/{capture_id}/calibration/from-crops",
            files=_multipart_crops(_crops()),
        )

    assert replaced.status_code == 200
    assert replaced.json()["state"] == "sealed"
    assert replaced.json()["seal_purpose"] == "decode"
