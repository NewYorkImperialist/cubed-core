from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from cubed_core.app import create_app
from cubed_core.settings import Settings

ADMIN_TOKEN = "test-admin-token-that-is-long-enough"
ADMIN_HEADERS = {"X-Cubed-Admin-Token": ADMIN_TOKEN}


def _client(tmp_path) -> TestClient:
    schemas = Path(__file__).resolve().parents[1] / "schemas"
    for name in (
        "capture-bundle-v1.schema.json",
        "capture-derivation-v1.schema.json",
        "ble-session-v1.schema.json",
        "color-calibration-v1.schema.json",
        "frame-annotations-v1.schema.json",
        "model-artifact-manifest-v1.schema.json",
    ):
        schema_source = schemas / name
        schema_target = tmp_path / "schemas" / name
        schema_target.parent.mkdir(parents=True, exist_ok=True)
        schema_target.write_bytes(schema_source.read_bytes())
    settings = Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_TOKEN,
    )
    return TestClient(create_app(settings), headers=ADMIN_HEADERS)


def test_health_and_empty_capture_library(tmp_path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/captures").json() == {"captures": [], "count": 0}
        schema = client.get("/api/specs/capture-bundle").json()
        assert schema["properties"]["schema_version"]["const"] == 1
        derivation_schema = client.get("/api/specs/capture-derivation").json()
        assert (
            derivation_schema["properties"]["schema"]["const"] == "cubed-core/frame-rate-derivation"
        )
        ble_schema = client.get("/api/specs/ble-session").json()
        assert ble_schema["properties"]["schema"]["const"] == "cubed-core/ble-session"
        calibration_schema = client.get("/api/specs/color-calibration").json()
        assert (
            calibration_schema["properties"]["schema"]["const"] == "cubed-core/color-calibration-v1"
        )
        annotations_schema = client.get("/api/specs/frame-annotations").json()
        assert annotations_schema["properties"]["schema"]["const"] == "cubed-core/frame-annotations"
        model_manifest_schema = client.get("/api/specs/model-artifact-manifest").json()
        assert (
            model_manifest_schema["properties"]["schema"]["const"]
            == "cubed-core/model-artifact-manifest-v1"
        )


def test_capabilities_publish_one_video_upload_limit(tmp_path) -> None:
    with _client(tmp_path) as client:
        limits = client.get("/api/capabilities").json()["upload_limits"]

    assert limits == {"video_bytes": 1024 * 1024}


def test_openapi_has_decode_jobs_without_a_standalone_tracker_service(tmp_path) -> None:
    with _client(tmp_path) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert any("/decode-jobs" in path for path in paths)
    assert all("/tracker-jobs" not in path for path in paths)
    assert all(not path.startswith("/api/tracker/jobs") for path in paths)
    assert "/api/specs/tracker-dump" not in paths


def test_capture_delete_cors_preflight_allows_delete(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.options(
            f"/api/captures/{'a' * 32}",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "X-Cubed-Admin-Token",
            },
        )

    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_delete_capture_moves_it_to_system_trash(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    system_trash = tmp_path / "system-trash"
    system_trash.mkdir()

    def fake_send2trash(value: str) -> None:
        source = Path(value)
        source.rename(system_trash / source.name)

    monkeypatch.setattr("cubed_core.workspace.send2trash", fake_send2trash)
    with _client(tmp_path) as client:
        capture_id = client.post(
            "/api/captures/import",
            files={"video": ("solve.mp4", b"video", "video/mp4")},
        ).json()["capture_id"]

        response = client.delete(f"/api/captures/{capture_id}")

        assert response.status_code == 200
        value = response.json()
        assert value == {
            "schema": "cubed-core/capture-delete-v1",
            "schema_version": 1,
            "capture_id": capture_id,
            "trashed": True,
            "recoverable": True,
        }
        destination = next(system_trash.iterdir())
        assert (destination / "capture" / "capture.json").is_file()
        assert not (tmp_path / "workspace" / "captures" / capture_id).exists()
        assert client.get("/api/captures").json() == {"captures": [], "count": 0}
        assert client.delete(f"/api/captures/{capture_id}").status_code == 404


@pytest.mark.parametrize("status", ["queued", "running"])
def test_delete_capture_refuses_active_jobs(
    tmp_path,
    monkeypatch,
    status,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    with _client(tmp_path) as client:
        capture_id = client.post(
            "/api/captures/import",
            files={"video": ("solve.mp4", b"video", "video/mp4")},
        ).json()["capture_id"]
        service = client.app.state.decode_jobs
        monkeypatch.setattr(
            service,
            "list_for_capture",
            lambda requested: [{"capture_id": requested, "status": status}],
        )

        response = client.delete(f"/api/captures/{capture_id}")

        assert response.status_code == 409
        assert "queued or running" in response.json()["detail"]
        assert (tmp_path / "workspace" / "captures" / capture_id).is_dir()


def test_import_request_guard_rejects_oversized_video_before_staging(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/captures/import",
            files={
                "video": (
                    "oversized.mp4",
                    b"x" * (3 * 1024 * 1024),
                    "video/mp4",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "request body exceeds the configured upload limit"
    assert list((tmp_path / "workspace" / "captures").iterdir()) == []


def test_capture_spec_exposes_strict_camera_intrinsics_contract(tmp_path) -> None:
    with _client(tmp_path) as client:
        schema = client.get("/api/specs/capture-bundle").json()

    intrinsics_schema = schema["properties"]["camera"]["properties"]["intrinsics"]
    assert "raw sample-buffer axis" in intrinsics_schema["description"]
    assert "video.rotation_degrees" in intrinsics_schema["description"]
    configured_fps_schema = schema["properties"]["video"]["properties"]["configured_fps"]
    assert configured_fps_schema["minimum"] == 60
    assert configured_fps_schema["maximum"] == 240
    assert "actual_fps remains the authoritative" in configured_fps_schema["description"]
    validator = Draft202012Validator(intrinsics_schema)
    valid = {
        "matrix": [
            [1200.0, 0.0, 960.0],
            [0.0, 1180.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        "ref_w": 1920,
        "ref_h": 1080,
    }
    assert list(validator.iter_errors(valid)) == []
    assert list(validator.iter_errors(None)) == []
    assert list(validator.iter_errors({**valid, "distortion": []}))
    assert list(validator.iter_errors({**valid, "matrix": [[1.0]]}))
    assert list(validator.iter_errors({**valid, "ref_w": 0}))


def test_security_headers_cover_success_and_guard_rejection(tmp_path) -> None:
    with _client(tmp_path) as client:
        responses = (
            client.get("/api/health"),
            client.get(
                "/api/captures",
                headers={"X-Cubed-Admin-Token": ""},
            ),
        )
        assert [response.status_code for response in responses] == [200, 403]
        for response in responses:
            assert response.headers["referrer-policy"] == "no-referrer"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["x-frame-options"] == "DENY"


def _local_session_headers(origin: str = "http://127.0.0.1:8000") -> dict[str, str]:
    return {
        "Origin": origin,
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }


def _origin_only_local_session_headers(
    origin: str = "http://127.0.0.1:8000",
) -> dict[str, str]:
    return {"Origin": origin}


def _local_session_client(
    tmp_path: Path,
    *,
    enabled: bool,
    client_host: str = "127.0.0.1",
) -> TestClient:
    settings = Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_TOKEN,
        local_browser_auth=enabled,
    )
    return TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8000",
        client=(client_host, 43123),
    )


def test_local_browser_bootstrap_reuses_internal_admin_token(tmp_path) -> None:
    with _local_session_client(tmp_path, enabled=True) as client:
        response = client.post("/api/local-session", headers=_local_session_headers())

        assert response.status_code == 200
        assert response.json() == {"admin_token": ADMIN_TOKEN}
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["vary"] == "Origin"
        assert (
            client.get(
                "/api/captures",
                headers={"X-Cubed-Admin-Token": response.json()["admin_token"]},
            ).status_code
            == 200
        )
        assert client.app.state.settings.admin_token == response.json()["admin_token"]


def test_local_browser_bootstrap_allows_absent_fetch_metadata(tmp_path) -> None:
    with _local_session_client(tmp_path, enabled=True) as client:
        response = client.post(
            "/api/local-session",
            headers=_origin_only_local_session_headers(),
        )

    assert response.status_code == 200
    assert response.json() == {"admin_token": ADMIN_TOKEN}


def test_vite_proxy_origin_can_bootstrap_when_proxy_preserves_host(tmp_path) -> None:
    settings = Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_TOKEN,
        local_browser_auth=True,
    )
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:5173",
        client=("127.0.0.1", 43123),
    ) as client:
        response = client.post(
            "/api/local-session",
            headers=_local_session_headers("http://127.0.0.1:5173"),
        )

    assert response.status_code == 200


def test_local_browser_bootstrap_is_disabled_by_default(tmp_path) -> None:
    with _local_session_client(tmp_path, enabled=False) as client:
        response = client.post("/api/local-session", headers=_local_session_headers())

    assert response.status_code == 403
    assert response.json()["detail"] == "local browser authorization unavailable"


def test_local_browser_bootstrap_rejects_nonlocal_client(tmp_path) -> None:
    with _local_session_client(
        tmp_path / "remote",
        enabled=True,
        client_host="192.0.2.8",
    ) as remote_client:
        remote_response = remote_client.post(
            "/api/local-session",
            headers=_local_session_headers(),
        )

    assert remote_response.status_code == 403


def test_local_browser_bootstrap_rejects_cross_site_browser_metadata(tmp_path) -> None:
    requests = (
        _local_session_headers("https://attacker.example"),
        {
            **_local_session_headers(),
            "Sec-Fetch-Site": "cross-site",
        },
        {
            **_local_session_headers(),
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        },
        {
            **_origin_only_local_session_headers(),
            "Sec-Fetch-Site": "cross-site",
        },
        {key: value for key, value in _local_session_headers().items() if key != "Origin"},
    )
    with _local_session_client(tmp_path, enabled=True) as client:
        responses = [client.post("/api/local-session", headers=headers) for headers in requests]

    assert [response.status_code for response in responses] == [403, 403, 403, 403, 403]


def test_pairing_routes_are_not_exposed(tmp_path) -> None:
    with _client(tmp_path) as client:
        assert client.post("/api/pairings").status_code == 404
        assert client.get("/api/pairings/ABC123").status_code == 404


def test_attach_and_seal_capture_through_api(
    tmp_path,
    monkeypatch,
    color_calibration_payload,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1920,
            "height": 1080,
            "capture_class": "target",
        },
    )
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/captures/import",
            data={"scramble": "R U"},
            files={"video": ("solve.mp4", b"video", "video/mp4")},
        ).json()
        attached = client.post(
            f"/api/captures/{imported['capture_id']}/sidecars/calibration",
            files={
                "sidecar": (
                    "../uploaded calibration.json",
                    json.dumps(color_calibration_payload).encode(),
                    "application/json",
                )
            },
        )
        assert attached.status_code == 200
        assert attached.json()["calibration"]["display_name"] == "uploaded calibration.json"
        listed = client.get("/api/captures").json()["captures"]
        persisted = next(row for row in listed if row["capture_id"] == imported["capture_id"])
        assert persisted["calibration"]["display_name"] == "uploaded calibration.json"
        sealed = client.post(
            f"/api/captures/{imported['capture_id']}/seal",
            data={"purpose": "decode"},
        )
        assert sealed.status_code == 200
        assert sealed.json()["state"] == "sealed"


def test_reused_calibration_route_wires_missing_asset_and_decode_locked_replacement(
    tmp_path,
    monkeypatch,
    flat_color_centroids_payload,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "width": 1920,
            "height": 1080,
            "capture_class": "target",
        },
    )
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/captures/import",
            data={"scramble": "R U"},
            files={"video": ("solve.mp4", b"video", "video/mp4")},
        ).json()
        reuse_path = f"/api/captures/{imported['capture_id']}/sidecars/calibration/reuse"

        # The bundled asset has not been downloaded in this repo checkout yet.
        missing = client.post(reuse_path, data={"source": "bundled"})
        assert missing.status_code == 404
        assert "make download-decode-support" in missing.json()["detail"]

        asset_dir = tmp_path / "workspace" / "release-assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "calibration_gan12.json").write_text(json.dumps(flat_color_centroids_payload))
        attached = client.post(reuse_path, data={"source": "bundled"})
        assert attached.status_code == 200
        assert attached.json()["calibration"]["kind"] == "color_centroids_v1"
        assert attached.json()["calibration"]["display_name"] == "Published shared calibration"

        sealed = client.post(
            f"/api/captures/{imported['capture_id']}/seal",
            data={"purpose": "decode"},
        )
        assert sealed.status_code == 200

        replaced = client.post(reuse_path, data={"source": "bundled"})
        assert replaced.status_code == 200
        assert replaced.json()["state"] == "sealed"
        assert replaced.json()["seal_purpose"] == "decode"
        assert replaced.json()["calibration"]["display_name"] == "Published shared calibration"


def test_built_frontend_is_served_without_shadowing_unknown_api_routes(tmp_path) -> None:
    dist = tmp_path / "apps" / "lab-web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<h1>Cubed Core Lab</h1>", encoding="utf-8")

    with _client(tmp_path) as client:
        phone = client.get("/phone?pair=ABC123")
        assert phone.status_code == 200
        assert "Cubed Core Lab" in phone.text
        assert client.get("/api/not-a-route").status_code == 404


def test_capture_video_endpoint_serves_original_with_private_headers(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {"status": "ok", "fps": 120.0, "capture_class": "target"},
    )
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/captures/import",
            files={"video": ("solve.mp4", b"video-bytes", "video/mp4")},
        ).json()
        response = client.get(f"/api/captures/{imported['capture_id']}/video")
        assert response.status_code == 200
        assert response.content == b"video-bytes"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"

        denied = client.get(
            f"/api/captures/{imported['capture_id']}/video",
            headers={"X-Cubed-Admin-Token": ""},
        )
        assert denied.status_code == 403
        ticket = client.post(f"/api/captures/{imported['capture_id']}/media-ticket").json()
        streamed = client.get(
            ticket["url"],
            headers={
                "X-Cubed-Admin-Token": "",
                "Range": "bytes=0-4",
            },
        )
        assert streamed.status_code == 206
        assert streamed.content == b"video"


def test_exact_frame_endpoint_uses_authenticated_decoder(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cubed_core.workspace.probe_video",
        lambda path: {
            "status": "ok",
            "fps": 120.0,
            "frame_count": 3,
            "capture_class": "target",
        },
    )
    monkeypatch.setattr(
        "cubed_core.app.extract_frame_jpeg",
        lambda path, frame_index: b"\xff\xd8frame\xff\xd9",
    )
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/captures/import",
            files={"video": ("solve.mp4", b"video-bytes", "video/mp4")},
        ).json()
        frame = client.get(f"/api/captures/{imported['capture_id']}/frames/2")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content == b"\xff\xd8frame\xff\xd9"
        assert client.get(f"/api/captures/{imported['capture_id']}/frames/3").status_code == 400
