from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cubed_core.app import create_app
from cubed_core.label_assist import LabelAssistUnavailable
from cubed_core.settings import Settings

ADMIN_HEADERS = {"X-Cubed-Admin-Token": "label-test-admin"}


def client_for(tmp_path: Path, **settings_overrides) -> TestClient:
    settings = Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_HEADERS["X-Cubed-Admin-Token"],
        **settings_overrides,
    )
    return TestClient(create_app(settings), headers=ADMIN_HEADERS)


def import_capture(client: TestClient) -> dict[str, object]:
    return client.post(
        "/api/captures/import",
        files={"video": ("solve.mp4", b"video", "video/mp4")},
    ).json()


def document(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "cubed-core/frame-annotations",
        "schema_version": 1,
        "created_at": "2026-07-23T12:00:00+00:00",
        "source": {
            "kind": "workspace-capture",
            "capture_id": receipt["capture_id"],
            "filename": receipt["original_filename"],
            "sha256": receipt["video"]["sha256"],
            "fps": 120,
            "frame_count": 10,
        },
        "image": {
            "width": 640,
            "height": 480,
            "coordinate_space": "display-oriented-video-pixels",
        },
        "frames": [
            {
                "frame_index": 1,
                "time_seconds": 1 / 120,
                "aligned_label": "aligned",
                "polygons": [],
                "wireframe": None,
                "faces": [
                    {
                        "id": "face-1",
                        "corners": [[10, 10], [20, 10], [20, 20], [10, 20]],
                        "visible": [True, True, True, False],
                        "vertices": [None, None, None, None],
                        "pinned": [False, False, False, False],
                        "origin": "manual",
                    }
                ],
            }
        ],
    }


def test_annotation_autosave_round_trip_and_file_backed_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda _path: {
            "status": "ok",
            "fps": 120.0,
            "frame_count": 10,
            "width": 640,
            "height": 480,
            "capture_class": "target",
        },
    )

    def fake_archive(_video, _document, output):
        output.write_bytes(b"zip-bytes")
        return output

    monkeypatch.setattr("cubed_core.app.build_yolo_pose_archive", fake_archive)
    with client_for(tmp_path) as client:
        receipt = import_capture(client)
        path = f"/api/captures/{receipt['capture_id']}/annotations"
        saved = client.put(path, json=document(receipt))
        assert saved.status_code == 200
        assert saved.json()["path"].startswith("annotations/")
        assert client.get(path).json()["frames"][0]["faces"][0]["origin"] == "manual"
        assert not (
            tmp_path / "workspace" / "captures" / receipt["capture_id"] / "frame-annotations.json"
        ).exists()

        exported = client.get(f"{path}/dataset.zip")
        assert exported.status_code == 200
        assert exported.content == b"zip-bytes"
        assert list((tmp_path / "workspace" / "exports").iterdir()) == []

        malformed = document(receipt)
        malformed["frames"][0]["faces"][0]["corners"][0] = [-1, 1]
        assert client.put(path, json=malformed).status_code == 400


def test_label_routes_are_bounded_disabled_and_busy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda _path: {"status": "ok", "fps": 120.0, "capture_class": "target"},
    )
    with client_for(tmp_path) as client:
        receipt = import_capture(client)
        predict_path = f"/api/captures/{receipt['capture_id']}/label-predict"
        alignment_path = f"/api/captures/{receipt['capture_id']}/label-alignment"
        request = {"width": 640, "height": 480, "frame_indices": [1]}
        disabled = client.post(predict_path, json=request)
        assert disabled.status_code == 503
        assert client.post(alignment_path).status_code == 503

        lock = client.app.state.label_prediction_lock
        assert lock.acquire(blocking=False)
        try:
            assert client.post(predict_path, json=request).status_code == 429
            assert client.post(alignment_path).status_code == 429
        finally:
            lock.release()

        oversized = client.post(
            "/api/label/assist/extrapolate",
            content=json.dumps({"padding": "x" * 70_000}),
            headers={"Content-Type": "application/json", **ADMIN_HEADERS},
        )
        assert oversized.status_code == 413

        oversized_predict = client.post(
            predict_path,
            content=json.dumps({"padding": "x" * 70_000}),
            headers={"Content-Type": "application/json", **ADMIN_HEADERS},
        )
        assert oversized_predict.status_code == 413


def test_label_alignment_route_uses_capture_rotation_and_frame_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda _path: {
            "status": "ok",
            "fps": 120.0,
            "frame_count": 10,
            "rotation_degrees": 90,
            "capture_class": "target",
        },
    )
    observed = {}

    def fake_scan(
        _settings,
        *,
        video_path,
        rotation_degrees,
        expected_frame_count,
    ):
        observed.update(
            {
                "video_path": video_path,
                "rotation_degrees": rotation_degrees,
                "expected_frame_count": expected_frame_count,
            }
        )
        return {
            "schema": "cubed-core/label-alignment-v1",
            "threshold": 0.5,
            "model_profile": "camera-tracker-v1",
            "width": 480,
            "height": 640,
            "frame_count": 10,
            "aligned_frames": [2, 5],
            "alignment_confs": [0.0] * 10,
        }

    monkeypatch.setattr("cubed_core.app.run_label_alignment_scan", fake_scan)
    with client_for(tmp_path) as client:
        receipt = import_capture(client)
        response = client.post(f"/api/captures/{receipt['capture_id']}/label-alignment")

    assert response.status_code == 200
    assert response.json()["aligned_frames"] == [2, 5]
    assert observed["rotation_degrees"] == 90
    assert observed["expected_frame_count"] == 10
    assert observed["video_path"].is_file()


def test_pnp_endpoint_reports_missing_optional_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.app.extrapolate_request",
        lambda _body: (_ for _ in ()).throw(LabelAssistUnavailable("install label extra")),
    )
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/label/assist/extrapolate",
            json={
                "width": 640,
                "height": 480,
                "labeled_corners": [[10, 10], [20, 10], [20, 20]],
            },
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "install label extra"


def test_pnp_endpoint_rejects_degenerate_geometry(tmp_path: Path) -> None:
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/label/assist/extrapolate",
            json={
                "width": 640,
                "height": 480,
                "labeled_corners": [[10, 10], [20, 20], [30, 30]],
            },
        )
    assert response.status_code == 400
    assert "nondegenerate" in response.json()["detail"]
