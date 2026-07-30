from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn

from cubed_core import cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _report(
    *,
    ffmpeg: str | None = "/usr/bin/ffmpeg",
    ffprobe: str | None = "/usr/bin/ffprobe",
    prediction_status: str = "disabled",
) -> dict[str, Any]:
    return {
        "commands": {"ffmpeg": ffmpeg, "ffprobe": ffprobe},
        "label": {"prediction": {"status": prediction_status}},
        "tools": [{"id": "decoder", "status": "blocked-on-extraction"}],
    }


def test_tool_registry_publishes_only_the_supported_decode_surface() -> None:
    tools = json.loads((REPOSITORY_ROOT / "config" / "tooling.json").read_text(encoding="utf-8"))
    identifiers = [tool["id"] for tool in tools]
    by_id = {tool["id"]: tool for tool in tools}

    assert identifiers.count("decoder") == 1
    assert "decoder-foundations" not in identifiers
    assert by_id["tracker"]["surface"] == "Decode"
    assert "used by Decode" in by_id["tracker"]["name"]
    assert all(tool["surface"] != "Track" for tool in tools)


@pytest.mark.parametrize("prediction_status", ["disabled", "available"])
def test_doctor_keeps_optional_label_models_and_blocked_decoder_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prediction_status: str,
) -> None:
    monkeypatch.setattr(
        cli,
        "build_capabilities",
        lambda _settings: _report(
            prediction_status=prediction_status,
        ),
    )

    assert cli._doctor(object()) == 0
    assert "blocked-on-extraction" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("report", "expected_status"),
    [
        (_report(ffmpeg=None), 1),
        (_report(ffprobe=None), 1),
        (_report(prediction_status="misconfigured"), 1),
    ],
)
def test_doctor_fails_for_missing_required_media_or_broken_configured_tool(
    monkeypatch: pytest.MonkeyPatch,
    report: dict[str, Any],
    expected_status: int,
) -> None:
    monkeypatch.setattr(cli, "build_capabilities", lambda _settings: report)

    assert cli._doctor(object()) == expected_status


def test_doctor_requires_real_cuda_model_probe_when_cuda_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "build_capabilities", lambda _settings: _report())
    monkeypatch.setattr(
        cli,
        "probe_camera_model_cuda_session",
        lambda _manifest: {
            "status": "unavailable",
            "ready": False,
            "reason": "alignment model session did not activate CUDAExecutionProvider",
        },
    )
    settings = SimpleNamespace(
        tracker_onnx_providers=("CUDAExecutionProvider",),
        tracker_model_manifest=Path("/models/manifest.json"),
    )

    assert cli._doctor(settings) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["cuda_model_session"]["ready"] is False


def test_derive_command_prints_new_capture_receipt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    class FakeWorkspace:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def derive_240_to_120(self, capture_id: str) -> dict[str, object]:
            calls.append(capture_id)
            return {
                "schema": "cubed-core/capture-bundle",
                "capture_id": "b" * 32,
                "source": "derived-240-to-120",
            }

    monkeypatch.setattr(cli, "Workspace", FakeWorkspace)
    settings = SimpleNamespace(
        workspace="/workspace",
        max_upload_bytes=100,
    )

    assert cli._derive_240_to_120(settings, "a" * 32) == 0
    assert calls == ["a" * 32]
    assert json.loads(capsys.readouterr().out)["source"] == "derived-240-to-120"


def test_derive_command_reports_workspace_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeWorkspace:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def derive_240_to_120(self, capture_id: str) -> dict[str, object]:
            raise cli.WorkspaceError(f"capture {capture_id} is sealed")

    monkeypatch.setattr(cli, "Workspace", FakeWorkspace)
    settings = SimpleNamespace(
        workspace="/workspace",
        max_upload_bytes=100,
    )

    assert cli._derive_240_to_120(settings, "a" * 32) == 1
    assert "derivation failed" in capsys.readouterr().err


def test_verify_tracker_models_returns_capability_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "tracker_model_capability",
        lambda path: {
            "status": "verified",
            "ready": True,
            "reason": None,
            "profile": "camera-tracker-v1",
            "artifacts": [],
        },
    )
    settings = SimpleNamespace(tracker_model_manifest="/models/manifest.json")

    assert cli._verify_tracker_models(settings) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"

    monkeypatch.setattr(
        cli,
        "tracker_model_capability",
        lambda path: {
            "status": "invalid",
            "ready": False,
            "reason": "checksum mismatch",
            "profile": None,
            "artifacts": [],
        },
    )
    assert cli._verify_tracker_models(settings) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "checksum mismatch"


def test_loopback_serve_uses_environment_port_and_tokenless_workbench_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli.sys, "argv", ["cubed-core", "serve"])
    monkeypatch.setenv("CUBED_CORE_PORT", "8765")
    monkeypatch.setenv("CUBED_CORE_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("CUBED_CORE_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: calls.append({"app": app, **kwargs}),
    )

    assert cli.main() == 0
    assert calls == [
        {
            "app": "cubed_core.app:app",
            "host": "127.0.0.1",
            "port": 8765,
            "reload": False,
        }
    ]
    assert cli.os.environ["CUBED_CORE_LOCAL_BROWSER_AUTH"] == "1"
    output = capsys.readouterr().out
    assert "Workbench: http://127.0.0.1:8765/" in output
    assert "#admin=" not in output


def test_serve_browser_port_changes_only_the_generated_container_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "cubed-core",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--browser-port",
            "8765",
            "--allow-network",
        ],
    )
    monkeypatch.setenv("CUBED_CORE_PORT", "9999")
    monkeypatch.setenv("CUBED_CORE_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("CUBED_CORE_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: calls.append({"app": app, **kwargs}),
    )

    assert cli.main() == 0
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 8000
    assert cli.os.environ["CUBED_CORE_LOCAL_BROWSER_AUTH"] == "0"
    assert "Admin workbench: http://127.0.0.1:8765/#admin=" in capsys.readouterr().out


def test_allow_network_explicitly_disables_loopback_browser_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cubed-core", "serve", "--port", "8765", "--allow-network"],
    )
    monkeypatch.setenv("CUBED_CORE_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("CUBED_CORE_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)

    assert cli.main() == 0
    assert cli.os.environ["CUBED_CORE_LOCAL_BROWSER_AUTH"] == "0"
    assert "Admin workbench: http://127.0.0.1:8765/#admin=" in capsys.readouterr().out


def test_ipv6_loopback_workbench_url_is_bracketed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cubed-core", "serve", "--host", "::1", "--port", "8765"],
    )
    monkeypatch.setenv("CUBED_CORE_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("CUBED_CORE_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)

    assert cli.main() == 0
    assert "Workbench: http://[::1]:8765/" in capsys.readouterr().out


def test_serve_rejects_invalid_environment_port(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["cubed-core", "serve"])
    monkeypatch.setenv("CUBED_CORE_PORT", "70000")

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "must be an integer from 1 to 65535" in capsys.readouterr().err
