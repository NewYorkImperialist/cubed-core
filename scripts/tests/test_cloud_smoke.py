from __future__ import annotations

import argparse
from io import BytesIO

import pytest

from scripts import cloud_smoke


def _health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": "0.1.0",
        "publication_status": "pre-publication",
    }


def _capabilities(
    *,
    gpu_available: bool,
) -> dict[str, object]:
    return {
        "schema": "cubed-core/capabilities-v1",
        "workspace": "/workspace",
        "commands": {
            "ffmpeg": "/usr/bin/ffmpeg",
            "ffprobe": "/usr/bin/ffprobe",
        },
        "gpu": {
            "available": gpu_available,
            "devices": [{"name": "Test GPU", "memory_mib": 24576}] if gpu_available else [],
            "reason": None if gpu_available else "nvidia-smi not found",
        },
        "tools": [
            {
                "id": "decoder",
                "status": "blocked-on-extraction",
            }
        ],
    }


def test_cpu_smoke_passes_and_discloses_decoder_blocker() -> None:
    report = cloud_smoke.evaluate(
        _health(),
        _capabilities(gpu_available=False),
        require_gpu=False,
    )

    assert report["ok"] is True
    assert report["decoder"] == {
        "status": "blocked-on-extraction",
        "available": False,
    }


def test_gpu_smoke_fails_when_gpu_is_required_but_missing() -> None:
    report = cloud_smoke.evaluate(
        _health(),
        _capabilities(gpu_available=False),
        require_gpu=True,
    )

    assert report["ok"] is False
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["gpu-required"]["ok"] is False


def test_smoke_fails_if_required_runtime_or_decoder_status_is_missing() -> None:
    capabilities = _capabilities(gpu_available=True)
    capabilities["commands"] = {"ffmpeg": None, "ffprobe": "/usr/bin/ffprobe"}
    capabilities["tools"] = []

    report = cloud_smoke.evaluate(_health(), capabilities, require_gpu=True)

    assert report["ok"] is False
    failed = {check["id"] for check in report["checks"] if not check["ok"]}
    assert failed == {"command-ffmpeg", "decoder-status-disclosed"}


@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1:8000",
        "file:///tmp/service",
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000?token=secret",
    ],
)
def test_base_url_rejects_non_http_or_embedded_credentials(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must"):
        cloud_smoke._base_url(value)


def test_fetch_sends_admin_token_as_header(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return BytesIO(b'{"status": "ok"}')

    monkeypatch.setattr(cloud_smoke, "urlopen", fake_urlopen)
    result = cloud_smoke._fetch_object(
        "http://127.0.0.1:8000",
        "/api/health",
        3,
        admin_token="secret-token",
    )

    assert result == {"status": "ok"}
    assert captured["request"].get_header("X-cubed-admin-token") == "secret-token"
    assert captured["timeout"] == 3
