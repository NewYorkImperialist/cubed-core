from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cubed_core import decode_jobs as decode_jobs_module
from cubed_core import native_decode_runner
from cubed_core.app import create_app
from cubed_core.decode_jobs import (
    BUNDLED_REMOTE_DECODE_RUNNER,
    NATIVE_DECODE_MODULE,
    DecodeJob,
    DecodeJobError,
    DecodeJobService,
    _runner_provenance,
    build_decode_job_readiness,
    build_decode_readiness,
    decode_capability,
    replay_decode_result,
    validate_decode_result,
)
from cubed_core.settings import Settings
from cubed_core.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
ADMIN_TOKEN = "decode-test-admin-token-decode-test"
ADMIN_HEADERS = {"X-Cubed-Admin-Token": ADMIN_TOKEN}
CAPTURE_ID = "b" * 32
UNSEALED_CAPTURE_ID = "c" * 32
# Applying the scramble and then these moves returns a solved cube.
SCRAMBLE = "R U R' U'"
SOLVING_MOVES = ["U", "R", "U'", "R'"]
MODEL_REQUIREMENT_PATHS = (
    "workspace/release-assets/camera-tracker-v1-runtime/artifacts/face-pose.onnx",
    "workspace/release-assets/camera-tracker-v1-runtime/artifacts/alignment-classifier.onnx",
    "workspace/release-assets/trust_v1_numpy.npz",
)
PIPELINE_SCRIPTS = (
    "scripts/geo_read.py",
    "scripts/gen_motion_events.py",
    "scripts/extract_alignfeat.py",
    "scripts/check_nvdec.py",
    "scripts/run_research_decode.sh",
)


def _ground_truth_diagnostic(
    *,
    decoded_moves: list[str] | None = None,
    video_sha256: str,
) -> dict[str, Any]:
    moves = list(decoded_moves if decoded_moves is not None else SOLVING_MOVES)
    return {
        "schema": "cubed-core/decode-ground-truth-diagnostic-v1",
        "schema_version": 1,
        "diagnostic_only": True,
        "capture_id": CAPTURE_ID,
        "reference": {
            "kind": "published-smart-cube-ble",
            "scope": "sequence-only",
            "dataset_id": "cubed-core-gtd1",
            "revision": "main",
            "bootstrap_manifest_sha256": "2" * 64,
            "download_receipt_sha256": "3" * 64,
            "corpus_manifest_sha256": "4" * 64,
            "video_sha256": video_sha256,
            "scramble_sha256": "5" * 64,
            "ble_sha256": "6" * 64,
            "video_link_status": "linked",
            "video_recording_id_verified": True,
        },
        "normalization": {
            "raw_metric": "quarter-turn",
            "comparison_metric": "half-turn",
            "method": "adjacent-same-face-mod-4",
        },
        "counts": {
            "decoded_htm": len(moves),
            "ble_raw_qtm": len(moves),
            "ble_canonical_htm": len(moves),
        },
        "comparison": {
            "distance": 0,
            "ops": [
                {
                    "op": "equal",
                    "decoded": move,
                    "reference": move,
                    "index_decoded": index,
                    "index_reference": index,
                }
                for index, move in enumerate(moves)
            ],
        },
    }


def _repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    schema_dir = repo_root / "schemas"
    schema_dir.mkdir(parents=True)
    for name in (
        "decode-result-v1.schema.json",
        "decode-ground-truth-diagnostic-v1.schema.json",
        "decode-job-request-v1.schema.json",
        "decode-job-receipt-v1.schema.json",
    ):
        (schema_dir / name).write_bytes((REPO_ROOT / "schemas" / name).read_bytes())

    manifest = json.loads(
        (REPO_ROOT / "config" / "decode-runtime-v1.json").read_text(encoding="utf-8")
    )
    # Keep the fixture hermetic: the real ffmpeg/ffprobe presence is not what
    # these tests are about, and the profile CFG_HASH block stays untouched.
    manifest["required_commands"] = []
    (repo_root / "config").mkdir()
    (repo_root / "config" / "decode-runtime-v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (repo_root / "detect").mkdir()
    (repo_root / "detect" / "__init__.py").write_text("", encoding="utf-8")
    for relative in PIPELINE_SCRIPTS:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture stub\n", encoding="utf-8")
    for relative in MODEL_REQUIREMENT_PATHS:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    return repo_root


def _capture(
    workspace: Path,
    calibration_payload: dict[str, Any],
    *,
    capture_id: str = CAPTURE_ID,
    sealed: bool = True,
) -> None:
    capture_dir = workspace / "captures" / capture_id
    capture_dir.mkdir(parents=True)
    video = capture_dir / "source.mov"
    video.write_bytes(b"decode-test-video-bytes")
    calibration_json = json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n"
    (capture_dir / "calibration.json").write_text(calibration_json, encoding="utf-8")
    (capture_dir / "capture.json").write_text(
        json.dumps(
            {
                "schema": "cubed-core/capture-bundle",
                "schema_version": 1,
                "recording_id": capture_id,
                "capture_id": capture_id,
                "capture_session_id": "decode-test-session",
                "created_at": "2026-07-25T12:00:00+00:00",
                "source": "import",
                "state": "sealed" if sealed else "incomplete",
                "sealed_at": "2026-07-25T12:05:00+00:00" if sealed else None,
                "seal_purpose": "decode" if sealed else None,
                "original_filename": "source.mov",
                "video": {
                    "path": f"captures/{capture_id}/source.mov",
                    "bytes": video.stat().st_size,
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "container": "mov",
                    "codec": "h264",
                    "encoded_width": 1920,
                    "encoded_height": 1080,
                    "actual_fps": 120,
                    "frame_count": 240,
                    "rotation_degrees": 0,
                    "mirrored": False,
                    "time_origin": None,
                },
                "calibration": {
                    "path": "calibration.json",
                    "sha256": hashlib.sha256(
                        calibration_json.encode("utf-8"),
                    ).hexdigest(),
                },
                "camera": {
                    "facing": "back",
                    "device_model": "test-camera",
                    "camera_id": "camera-1",
                    "intrinsics": {
                        "matrix": [
                            [1000.0, 0.0, 960.0],
                            [0.0, 1000.0, 540.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "ref_w": 1920,
                        "ref_h": 1080,
                    },
                },
                "solve": {
                    "scramble": SCRAMBLE,
                    "end_condition": "solved",
                    "start_frame": 0,
                    "end_frame": None,
                },
                "provenance": {
                    "producer": "decode-test",
                    "producer_version": "1",
                    "normalized_from_sha256": None,
                },
                "readiness": {
                    "can_label": True,
                    "can_decode": True,
                    "missing_for_decode": [],
                    "warnings": [],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _settings(
    tmp_path: Path,
    calibration_payload: dict[str, Any],
    *,
    decode_command: tuple[str, ...] = (),
    decode_mode: str = "disabled",
    decode_runner_label: str = "",
) -> Settings:
    repo_root = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    _capture(workspace, calibration_payload)
    _capture(
        workspace,
        calibration_payload,
        capture_id=UNSEALED_CAPTURE_ID,
        sealed=False,
    )
    return Settings(
        repo_root=repo_root,
        workspace=workspace,
        max_upload_bytes=1024 * 1024,
        admin_token=ADMIN_TOKEN,
        decode_mode=decode_mode,
        decode_command=decode_command,
        decode_runner_label=decode_runner_label,
    )


def _runner(
    tmp_path: Path,
    *,
    status: str = "completed",
    moves: list[str] | None = None,
    solved_reached: bool | None = True,
    expected_mode: str = "external",
    expected_environment: dict[str, str] | None = None,
    include_workstation: bool = False,
    workstation_video_overrides: dict[str, Any] | None = None,
    workstation_encoded_overrides: dict[str, Any] | None = None,
    workstation_scramble: str | None = None,
    workstation_sequence_moves: list[str] | None = None,
    bind_request_video_input: bool = False,
) -> tuple[str, ...]:
    """A stand-in external runner emitting markers and one decode-result-v1 document."""

    environment_assertions = tuple(
        f"assert os.environ[{name!r}] == {value!r}"
        for name, value in sorted((expected_environment or {}).items())
    )
    workstation_lines = (
        (
            "workstation_video = dict(request['workstation_context']['video'])",
            "workstation_video['encoded'] = {",
            "    'fps': workstation_video['fps'],",
            "    'frame_count': workstation_video['frame_count'],",
            "    'width': workstation_video['width'],",
            "    'height': workstation_video['height'],",
            "}",
            *(
                (f"workstation_video.update({workstation_video_overrides!r})",)
                if workstation_video_overrides is not None
                else ()
            ),
            *(
                (f"workstation_video['encoded'].update({workstation_encoded_overrides!r})",)
                if workstation_encoded_overrides is not None
                else ()
            ),
            "workstation_scramble = request['inputs']['scramble']",
            *(
                (f"workstation_scramble = {workstation_scramble!r}",)
                if workstation_scramble is not None
                else ()
            ),
            "result['workstation'] = {",
            "    'schema': 'cubed-core/decode-workstation-v1',",
            "    'schema_version': 1,",
            "    'video': workstation_video,",
            "    'initialization': {'scramble': workstation_scramble},",
            "    'window': [0, request['workstation_context']['video']['frame_count'] - 1],",
            "    'warnings': request['workstation_context']['warnings'],",
            "    'frames': {'10': {",
            "        'motion': 1.25,",
            "        'aligned': 0.9,",
            "        'aligned_streak': 4,",
            "        'face_count': 2,",
            "    }},",
            "}",
            *(
                (
                    f"workstation_sequence_moves = {workstation_sequence_moves!r}",
                    "result['workstation']['sequence'] = {",
                    "    'moves': [",
                    "        {'move': move, 'frame': index}",
                    "        for index, move in enumerate(workstation_sequence_moves)",
                    "    ],",
                    "    'timing_basis': 'canonical',",
                    "}",
                )
                if workstation_sequence_moves is not None
                else ()
            ),
        )
        if include_workstation
        else ()
    )
    path = tmp_path / "decode_runner.py"
    path.write_text(
        "\n".join(
            (
                "import argparse",
                "import hashlib",
                "import json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "request = json.loads(Path(args.request).read_text(encoding='utf-8'))",
                "assert request['schema'] == 'cubed-core/decode-job-request-v1'",
                "assert request['profile'] == 'local_camera_v1'",
                "assert request['inference_policy']['evaluation'] == 'must-be-absent'",
                "assert Path(request['inputs']['video']).is_file()",
                "assert Path(request['inputs']['calibration']).is_file()",
                "import os",
                "assert os.environ.get('CUBED_CORE_ADMIN_TOKEN') is None",
                "assert os.environ.get('SERVICE_PASSWORD') is None",
                f"assert os.environ['CUBED_CORE_DECODE_MODE'] == {expected_mode!r}",
                *environment_assertions,
                "for token in ('reads', 'events', 'alignfeat', 'decode'):",
                "    print('[cubed-core:stage] ' + token, flush=True)",
                "result = {",
                "    'schema': 'cubed-core/decode-result',",
                "    'schema_version': 1,",
                "    'recording_id': request['capture_id'],",
                f"    'status': {status!r},",
                "    'profile': 'local_camera_v1',",
                "    'config': {",
                "        'name': request['config']['name'],",
                "        'cfg_hash': request['config']['cfg_hash'],",
                "        'cfg_hash_algorithm': 'posix-cksum',",
                "    },",
                "    'inputs': [",
                "        {'id': 'video', 'sha256': '0' * 64},",
                "        {'id': 'color-calibration-v1', 'sha256': '1' * 64},",
                "    ],",
                f"    'moves': {moves if moves is not None else SOLVING_MOVES!r},",
                f"    'endpoint': {{'solved_reached': {solved_reached!r}}},",
                "    'evaluation': None,",
                "    'provenance': {",
                "        'runtime_version': 'decode-test',",
                "        'finished_at': '2026-07-25T12:30:00+00:00',",
                "    },",
                "}",
                *workstation_lines,
                *(
                    (
                        "result['inputs'][0]['sha256'] = hashlib.sha256(",
                        "    Path(request['inputs']['video']).read_bytes()",
                        ").hexdigest()",
                    )
                    if bind_request_video_input
                    else ()
                ),
                "Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')",
                "",
            )
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(path))


def _install_bundled_remote_runner(
    settings: Settings,
    tmp_path: Path,
    *,
    expected_environment: dict[str, str],
) -> Path:
    """Install an executable test double at the shipped remote runner path."""

    source_path = Path(
        _runner(
            tmp_path,
            expected_mode="native",
            expected_environment=expected_environment,
            bind_request_video_input=True,
        )[1]
    )
    runner_path = settings.repo_root / BUNDLED_REMOTE_DECODE_RUNNER
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(
        f"#!{sys.executable}\n{source_path.read_text(encoding='utf-8')}",
        encoding="utf-8",
    )
    runner_path.chmod(0o755)
    return runner_path


def _write_remote_host(
    settings: Settings,
    *,
    overrides: dict[str, Any] | None = None,
) -> None:
    host: dict[str, Any] = {
        "id": "test-gpu-host",
        "label": "Test GPU host",
        "ssh_dest": "root@203.0.113.55",
        "ssh_port": 2202,
    }
    host.update(overrides or {})
    (settings.workspace / "remote-hosts.json").write_text(
        json.dumps(
            {
                "schema": "cubed-core/remote-hosts-v1",
                "schema_version": 1,
                "hosts": [host],
            }
        ),
        encoding="utf-8",
    )


def _failing_runner(tmp_path: Path, body: str) -> tuple[str, ...]:
    """A stand-in external runner that never writes a result document, for
    exercising the nonzero-exit-status path directly.
    """

    path = tmp_path / "failing_decode_runner.py"
    path.write_text(
        "\n".join(
            (
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                body,
                "",
            )
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(path))


def _wait_for_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/decode/jobs/{job_id}")
        assert response.status_code == 200
        value = response.json()
        if value["status"] in {"succeeded", "failed", "timed_out"}:
            return value
        time.sleep(0.01)
    raise AssertionError("decode job did not finish")


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise AssertionError(f"file was not persisted: {path}")


def test_decode_capability_is_disabled_and_fails_closed_without_configuration(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    capability = decode_capability(settings)
    assert capability["enabled"] is False
    assert capability["status"] == "disabled"
    assert capability["runner_kind"] == "disabled"
    assert capability["runner_label"] is None
    assert capability["reason"] == "CUBED_CORE_DECODE_MODE is disabled"
    assert capability["request_schema"] == "cubed-core/decode-job-request-v1"
    assert capability["output_schema"] == "cubed-core/decode-result-v1"
    assert capability["profile"] == "local_camera_v1"
    assert capability["evidence_scope"] == "reconstruction-evidence-with-replay-check"


def test_decode_mode_and_command_are_read_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBED_CORE_DECODE_COMMAND", f'{sys.executable} "runner with spaces.py"')
    settings = Settings.from_env(repo_root=tmp_path)
    # A command alone never enables execution.
    assert settings.decode_mode == "disabled"
    assert settings.decode_command == (sys.executable, "runner with spaces.py")

    monkeypatch.setenv("CUBED_CORE_DECODE_MODE", "external")
    monkeypatch.setenv("CUBED_CORE_DECODE_RUNNER_LABEL", "Cloud GPU over SSH")
    settings = Settings.from_env(repo_root=tmp_path)
    assert settings.decode_mode == "external"
    assert settings.decode_runner_label == "Cloud GPU over SSH"

    monkeypatch.setenv("CUBED_CORE_DECODE_COMMAND", "runner --output owned.json")
    with pytest.raises(ValueError, match="reserved"):
        Settings.from_env(repo_root=tmp_path)

    monkeypatch.delenv("CUBED_CORE_DECODE_COMMAND")
    monkeypatch.setenv("CUBED_CORE_DECODE_MODE", "on")
    with pytest.raises(ValueError, match="disabled, native, or external"):
        Settings.from_env(repo_root=tmp_path)


def test_decode_routes_are_admin_only_and_disabled_mode_returns_503(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        capability = client.get("/api/capabilities").json()["decode_jobs"]
        assert capability["enabled"] is False
        assert capability["status"] == "disabled"

        denied = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            headers={"X-Cubed-Admin-Token": ""},
        )
        assert denied.status_code == 403
        denied_status = client.get(
            f"/api/decode/jobs/{'0' * 32}",
            headers={"X-Cubed-Admin-Token": ""},
        )
        assert denied_status.status_code == 403
        denied_list = client.get(
            "/api/decode/jobs",
            headers={"X-Cubed-Admin-Token": ""},
        )
        assert denied_list.status_code == 403
        disabled = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs")
        assert disabled.status_code == 503
        assert "CUBED_CORE_DECODE_MODE" in disabled.json()["detail"]

        spec = client.get(
            "/api/specs/decode-job-request",
            headers={"X-Cubed-Admin-Token": ""},
        )
        assert spec.status_code == 200
        assert spec.json()["properties"]["schema"]["const"] == "cubed-core/decode-job-request-v1"


def test_decode_capability_rejects_an_unresolvable_external_runner(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=("cubed-core-decode-runner-that-does-not-exist",),
    )
    readiness = build_decode_readiness(settings)
    assert readiness.enabled is False
    assert readiness.status == "misconfigured"
    assert "was not found" in str(readiness.reason)
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs")
        assert response.status_code == 503
        assert "misconfigured" in response.json()["detail"]


def test_native_decode_job_without_remote_host_uses_the_local_module_runner(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )

    readiness = build_decode_job_readiness(settings, remote_host=None)
    assert readiness.enabled is True
    assert readiness.status == "available"
    assert readiness.runner_kind == "native"
    assert readiness.command == (sys.executable, "-m", NATIVE_DECODE_MODULE)
    assert readiness.executable == sys.executable

    provenance = _runner_provenance(settings, readiness)
    assert provenance["runner_kind"] == "native"
    assert provenance["identity_status"] == "verified-native-identity"
    assert (
        provenance["native_module"]["sha256"]
        == hashlib.sha256(Path(native_decode_runner.__file__).read_bytes()).hexdigest()
    )
    assert provenance["package"]["name"] == "cubed-core"


def test_native_decode_is_unavailable_on_windows(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decode_jobs_module, "_NATIVE_DECODE_SUPPORTED", False)
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )

    readiness = build_decode_job_readiness(settings, remote_host=None)
    assert readiness.enabled is False
    assert readiness.status == "unsupported-platform"
    assert readiness.runner_kind == "native"
    assert readiness.command == ()
    assert readiness.executable is None
    assert "WSL2 or Linux" in str(readiness.reason)

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        capability = client.get("/api/capabilities").json()["decode_jobs"]
        assert capability["enabled"] is False
        assert capability["status"] == "unsupported-platform"
        response = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs")
        assert response.status_code == 503
        assert "unsupported-platform" in response.json()["detail"]


def test_decode_submission_is_gated_on_the_capture_preflight(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        unsealed = client.post(f"/api/captures/{UNSEALED_CAPTURE_ID}/decode-jobs")
        assert unsealed.status_code == 422
        assert "capture.seal" in unsealed.json()["detail"]

        missing = client.post(f"/api/captures/{'d' * 32}/decode-jobs")
        assert missing.status_code == 404

        invalid = client.post("/api/captures/not-a-capture-id/decode-jobs")
        assert invalid.status_code == 400


def test_decode_job_reports_stages_then_publishes_a_replayed_result(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
        decode_runner_label="Cloud GPU over SSH",
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        capability = client.get("/api/capabilities").json()["decode_jobs"]
        assert capability["enabled"] is True
        assert capability["status"] == "configured-external"
        assert capability["runner_kind"] == "external"
        assert capability["runner_label"] == "Cloud GPU over SSH"

        created = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs")
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        assert created.headers["Location"] == f"/api/decode/jobs/{job_id}"
        assert created.json()["schema"] == "cubed-core/decode-job-status-v1"
        assert created.json()["schema_version"] == 1
        assert created.json()["status"] in {"queued", "running"}
        assert created.json()["result_available"] is False

        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]
        assert final["return_code"] == 0
        assert final["error"] is None
        assert final["stage"] == "decode"
        assert final["stages_seen"] == ["reads", "events", "alignfeat", "decode"]
        assert final["stage_progress"] is None
        assert final["log_truncated"] is False
        assert final["started_at"] is not None and final["finished_at"] is not None
        assert final["result_available"] is True
        assert final["result_url"] == f"/api/decode/jobs/{job_id}/result"
        assert final["result_status"] == "completed"
        assert final["replay_solved_reached"] is True
        assert final["replay_check"]["performed"] is True
        assert final["replay_check"]["move_count"] == len(SOLVING_MOVES)
        assert final["evidence_scope"] == "reconstruction-evidence-with-replay-check"
        assert final["output_schema"] == "cubed-core/decode-result-v1"

        result = client.get(f"/api/decode/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["Cache-Control"] == "private, no-store"
        document = result.json()
        assert document["schema"] == "cubed-core/decode-result"
        assert document["recording_id"] == CAPTURE_ID
        assert document["moves"] == SOLVING_MOVES
        assert document["evaluation"] is None
        assert hashlib.sha256(result.content).hexdigest() == final["result_sha256"]
        assert len(result.content) == final["result_bytes"]

        receipt = json.loads(
            (settings.workspace / "decode-jobs" / job_id / "decode-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["schema"] == "cubed-core/decode-job-receipt-v1"
        assert receipt["result_sha256"] == final["result_sha256"]
        assert receipt["replay_check"]["solved_reached"] is True
        assert receipt["runner_provenance"]["identity_status"] == "unverified-external-identity"
        assert [asset["id"] for asset in receipt["runtime_assets"]] == [
            "alignment-model",
            "pose-model",
            "read-trust-model",
        ]


def test_decode_job_publishes_one_receipt_bound_result_with_post_replay_ble(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    ordering: list[str] = []
    original_replay = decode_jobs_module.replay_decode_result

    def recording_replay(*args: Any, **kwargs: Any) -> Any:
        ordering.append("replay")
        return original_replay(*args, **kwargs)

    def public_diagnostic(
        workspace_root: Path,
        *,
        capture_id: str,
        decoded_moves: list[str],
        video_sha256: str,
        scramble: str,
    ) -> dict[str, Any]:
        assert workspace_root == settings.workspace
        assert capture_id == CAPTURE_ID
        assert list(decoded_moves) == SOLVING_MOVES
        assert scramble == SCRAMBLE
        assert ordering == ["replay"]
        ordering.append("diagnostic")
        return _ground_truth_diagnostic(
            decoded_moves=list(decoded_moves),
            video_sha256=video_sha256,
        )

    monkeypatch.setattr(decode_jobs_module, "replay_decode_result", recording_replay)
    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        public_diagnostic,
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]
        assert ordering == ["replay", "diagnostic"]

        job_dir = settings.workspace / "decode-jobs" / job_id
        runner_document = json.loads((job_dir / "runner-result.json").read_text(encoding="utf-8"))
        assert "ground_truth_diagnostic" not in runner_document

        response = client.get(f"/api/decode/jobs/{job_id}/result")
        assert response.status_code == 200
        public_bytes = response.content
        public_document = response.json()
        assert public_document["ground_truth_diagnostic"]["diagnostic_only"] is True
        assert public_document["ground_truth_diagnostic"]["comparison"]["distance"] == 0
        assert hashlib.sha256(public_bytes).hexdigest() == final["result_sha256"]
        assert len(public_bytes) == final["result_bytes"]
        assert (job_dir / "decode-result.json").read_bytes() == public_bytes

        receipt = json.loads((job_dir / "decode-receipt.json").read_text(encoding="utf-8"))
        assert receipt["result_sha256"] == final["result_sha256"]
        assert receipt["result_bytes"] == len(public_bytes)

    # Disk restore serves the same closed artifact. It must not rebuild or
    # consult public data after the receipt has been written.
    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        lambda *args, **kwargs: pytest.fail("restore must not rebuild BLE diagnostics"),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        restored_status = client.get(f"/api/decode/jobs/{job_id}").json()
        restored = client.get(f"/api/decode/jobs/{job_id}/result")
        assert restored_status["status"] == "succeeded"
        assert restored.content == public_bytes
        assert (
            restored.json()["ground_truth_diagnostic"] == public_document["ground_truth_diagnostic"]
        )


def test_unexpected_public_diagnostic_error_cannot_fail_a_camera_result(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )

    def broken_diagnostic(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("/private/provider/path must not escape")

    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        broken_diagnostic,
    )
    with caplog.at_level("WARNING"):
        with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
            job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
            final = _wait_for_terminal(client, job_id)
            response = client.get(f"/api/decode/jobs/{job_id}/result")

    assert final["status"] == "succeeded"
    assert response.status_code == 200
    assert "ground_truth_diagnostic" not in response.json()
    assert "public diagnostic unexpectedly unavailable" in caplog.text
    assert "/private/provider/path" not in caplog.text


def test_unexpected_enriched_validation_error_falls_back_to_camera_result(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        lambda workspace_root, *, capture_id, decoded_moves, video_sha256, scramble: (
            _ground_truth_diagnostic(
                decoded_moves=list(decoded_moves),
                video_sha256=video_sha256,
            )
        ),
    )
    original_validate = decode_jobs_module.validate_decode_result

    def reject_enriched(value: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(value, dict) and "ground_truth_diagnostic" in value:
            raise RuntimeError("unexpected optional validation failure")
        return original_validate(value, **kwargs)

    monkeypatch.setattr(decode_jobs_module, "validate_decode_result", reject_enriched)
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        response = client.get(f"/api/decode/jobs/{job_id}/result")

    assert final["status"] == "succeeded"
    assert "ground_truth_diagnostic" not in response.json()


def test_optional_ble_is_omitted_before_publish_when_it_crosses_result_limit(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        lambda workspace_root, *, capture_id, decoded_moves, video_sha256, scramble: (
            _ground_truth_diagnostic(
                decoded_moves=list(decoded_moves),
                video_sha256=video_sha256,
            )
        ),
    )
    # The raw camera document remains below this test ceiling while the
    # optional diagnostic pushes the final serialization above it.
    monkeypatch.setattr(decode_jobs_module, "DECODE_RESULT_MAX_BYTES", 1_500)

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        response = client.get(f"/api/decode/jobs/{job_id}/result")

    assert final["status"] == "succeeded"
    assert final["result_bytes"] < 1_500
    assert response.status_code == 200
    assert "ground_truth_diagnostic" not in response.json()
    assert (
        settings.workspace / "decode-jobs" / job_id / "decode-result.json"
    ).read_bytes() == response.content


def test_decode_job_accepts_an_optional_portable_workstation_result(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            include_workstation=True,
            workstation_sequence_moves=SOLVING_MOVES,
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]
        assert final["outcome"] == "completed"
        assert final["video_sha256"] == hashlib.sha256(b"decode-test-video-bytes").hexdigest()

        result = client.get(f"/api/decode/jobs/{job_id}/result").json()
        workstation = result["workstation"]
        assert workstation["schema"] == "cubed-core/decode-workstation-v1"
        assert workstation["video"]["sha256"] == final["video_sha256"]
        assert workstation["initialization"]["scramble"] == SCRAMBLE
        assert workstation["frames"]["10"]["aligned_streak"] == 4
        assert [entry["move"] for entry in workstation["sequence"]["moves"]] == (SOLVING_MOVES)

        request_value = json.loads(
            (settings.workspace / "decode-jobs" / job_id / "job-request.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(request_value["workstation_context"]) == {"video", "warnings"}
        assert "path" not in request_value["workstation_context"]["video"]

    # A restored terminal job has no live submission snapshot. Its already
    # accepted result remains available through the persisted receipt binding.
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        restored = client.get(f"/api/decode/jobs/{job_id}/result")
        assert restored.status_code == 200
        assert restored.json()["workstation"] == workstation


@pytest.mark.parametrize(
    ("video_overrides", "encoded_overrides", "scramble_override", "detail"),
    [
        ({"sha256": "d" * 64}, None, None, "video identity"),
        ({"bytes": 999}, None, None, "video identity"),
        (None, {"fps": 60.0}, None, "video identity"),
        (None, {"frame_count": 120}, None, "video identity"),
        (None, {"width": 1280}, None, "video identity"),
        (None, {"height": 720}, None, "video identity"),
        (None, None, "F", "initialization"),
    ],
    ids=[
        "sha256",
        "bytes",
        "fps",
        "frame-count",
        "width",
        "height",
        "scramble",
    ],
)
def test_decode_job_rejects_any_workstation_identity_mismatch(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    video_overrides: dict[str, Any] | None,
    encoded_overrides: dict[str, Any] | None,
    scramble_override: str | None,
    detail: str,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            include_workstation=True,
            workstation_video_overrides=video_overrides,
            workstation_encoded_overrides=encoded_overrides,
            workstation_scramble=scramble_override,
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "failed"
    assert detail in final["error"]
    assert final["failure"]["code"] == "invalid-result"
    assert final["result_available"] is False


def test_decode_job_accepts_rotated_workstation_display_geometry(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            include_workstation=True,
            workstation_video_overrides={"width": 1080, "height": 1920},
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        result = client.get(f"/api/decode/jobs/{job_id}/result").json()

    assert final["status"] == "succeeded", final["error"]
    assert result["workstation"]["video"]["width"] == 1080
    assert result["workstation"]["video"]["height"] == 1920
    assert result["workstation"]["video"]["encoded"]["width"] == 1920
    assert result["workstation"]["video"]["encoded"]["height"] == 1080


def test_decode_job_rejects_a_workstation_sequence_that_differs_from_result_moves(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            include_workstation=True,
            workstation_sequence_moves=["R", "U", "F"],
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "failed"
    assert "sequence does not match top-level decode moves" in final["error"]
    assert final["failure"]["code"] == "invalid-result"
    assert final["result_available"] is False


def test_decode_job_rejects_failed_as_a_portable_result_status(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            status="failed",
            moves=[],
            solved_reached=False,
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "failed"
    assert final["failure"]["code"] == "invalid-result"
    assert final["result_available"] is False
    assert "execution failures belong to the job lifecycle" in final["error"]


def test_decode_job_remote_host_overlays_the_runner_environment(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-job remote_host wins over the server's own CUBED_REMOTE_*
    environment, and the chosen host is recorded in the receipt's existing
    runner_provenance.runner_label field rather than a new persisted field.
    """

    path = tmp_path / "decode_runner_remote_host.py"
    path.write_text(
        "\n".join(
            (
                "import argparse",
                "import json",
                "import os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "request = json.loads(Path(args.request).read_text(encoding='utf-8'))",
                "assert os.environ['CUBED_CORE_DECODE_MODE'] == 'external'",
                "assert os.environ['CUBED_REMOTE_SSH_DEST'] == 'root@203.0.113.55'",
                "assert os.environ['CUBED_REMOTE_SSH_PORT'] == '2202'",
                "result = {",
                "    'schema': 'cubed-core/decode-result',",
                "    'schema_version': 1,",
                "    'recording_id': request['capture_id'],",
                "    'status': 'completed',",
                "    'profile': 'local_camera_v1',",
                "    'config': {",
                "        'name': request['config']['name'],",
                "        'cfg_hash': request['config']['cfg_hash'],",
                "        'cfg_hash_algorithm': 'posix-cksum',",
                "    },",
                "    'inputs': [",
                "        {'id': 'video', 'sha256': '0' * 64},",
                "        {'id': 'color-calibration-v1', 'sha256': '1' * 64},",
                "    ],",
                f"    'moves': {SOLVING_MOVES!r},",
                "    'endpoint': {'solved_reached': True},",
                "    'evaluation': None,",
                "    'provenance': {",
                "        'runtime_version': 'decode-test',",
                "        'finished_at': '2026-07-25T12:30:00+00:00',",
                "    },",
                "}",
                "Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')",
                "",
            )
        ),
        encoding="utf-8",
    )
    command = (sys.executable, str(path))

    # The server's own environment already points somewhere else; the chosen
    # host must still win, proving the overlay is applied after os.environ.
    monkeypatch.setenv("CUBED_REMOTE_SSH_DEST", "root@should-be-overridden")
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=command,
        decode_runner_label="Cloud GPU over SSH",
    )
    _write_remote_host(settings)

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        created = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "test-gpu-host"},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]

        receipt = json.loads(
            (settings.workspace / "decode-jobs" / job_id / "decode-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        provenance = receipt["runner_provenance"]
        assert provenance["runner_kind"] == "external"
        assert provenance["identity_status"] == "unverified-external-identity"
        assert (
            provenance["command_sha256"]
            == hashlib.sha256(
                json.dumps(
                    list(command),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        assert provenance["runner_label"] == (
            "Cloud GPU over SSH · remote host: Test GPU host (test-gpu-host)"
        )


def test_decode_job_rejects_an_unknown_remote_host_id(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "bogus-host"},
        )
        assert response.status_code == 400
        assert "bogus-host" in response.json()["detail"]


def test_native_decode_job_with_remote_host_uses_the_bundled_external_runner(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )
    host_overrides = {
        "root": "/srv/cubed-core",
        "decode_venv": "/srv/cubed-core/.venv-decode",
        "nvdec": "1",
    }
    _write_remote_host(settings, overrides=host_overrides)
    expected_environment = {
        "CUBED_REMOTE_SSH_DEST": "root@203.0.113.55",
        "CUBED_REMOTE_SSH_PORT": "2202",
        "CUBED_REMOTE_ROOT": host_overrides["root"],
        "CUBED_REMOTE_DECODE_VENV": host_overrides["decode_venv"],
        "CUBED_REMOTE_NVDEC": host_overrides["nvdec"],
    }
    runner_path = _install_bundled_remote_runner(
        settings,
        tmp_path,
        expected_environment=expected_environment,
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        created = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "test-gpu-host"},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]

        receipt = json.loads(
            (settings.workspace / "decode-jobs" / job_id / "decode-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        provenance = receipt["runner_provenance"]
        assert provenance["runner_kind"] == "external"
        assert provenance["identity_status"] == "unverified-external-identity"
        assert (
            provenance["command_sha256"]
            == hashlib.sha256(
                json.dumps(
                    [str(runner_path)],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        assert provenance["executable"] == {
            "bytes": runner_path.stat().st_size,
            "sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        }
        assert provenance["implementation_files"] == []
        assert provenance["native_module"] is None
        assert provenance["package"] is None
        assert provenance["runner_label"] == ("remote host: Test GPU host (test-gpu-host)")


def test_native_decode_job_rejects_an_unknown_remote_host_id(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "bogus-host"},
        )
        assert response.status_code == 400
        assert "unknown remote host id 'bogus-host'" in response.json()["detail"]
        assert client.get(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["jobs"] == []
    assert not (settings.workspace / "decode-jobs").exists()


def test_native_decode_job_rejects_invalid_remote_host_configuration(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )
    (settings.workspace / "remote-hosts.json").write_text("not json", encoding="utf-8")

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "test-gpu-host"},
        )
        assert response.status_code == 400
        assert "remote-hosts.json must be valid finite JSON" in response.json()["detail"]
        assert client.get(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["jobs"] == []
    assert not (settings.workspace / "decode-jobs").exists()


def test_native_remote_decode_requires_the_bundled_runner_to_be_executable(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="native",
    )
    _write_remote_host(settings)

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.post(
            f"/api/captures/{CAPTURE_ID}/decode-jobs",
            params={"remote_host": "test-gpu-host"},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "misconfigured" in detail
        assert str(BUNDLED_REMOTE_DECODE_RUNNER) in detail
        assert client.get(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["jobs"] == []
    assert not (settings.workspace / "decode-jobs").exists()


def test_decode_job_fails_when_the_moves_do_not_replay_from_the_scramble(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path, moves=["R", "U"]),
    )
    diagnostic_calls = 0

    def diagnostic_must_not_run(*args: Any, **kwargs: Any) -> None:
        nonlocal diagnostic_calls
        diagnostic_calls += 1

    monkeypatch.setattr(
        decode_jobs_module,
        "try_build_public_ground_truth_diagnostic",
        diagnostic_must_not_run,
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "failed"
        assert final["return_code"] == 0
        assert "do not replay from the sealed scramble" in final["error"]
        assert final["result_available"] is False
        assert final["result_url"] is None
        assert final["replay_solved_reached"] is False

        result = client.get(f"/api/decode/jobs/{job_id}/result")
        assert result.status_code == 409
        job_dir = settings.workspace / "decode-jobs" / job_id
        assert (job_dir / "runner-result.json").is_file()
        assert not (job_dir / "decode-result.json").exists()
        assert diagnostic_calls == 0


def test_decode_job_error_promotes_the_structured_runner_error(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    """A nonzero exit on its own only says "runner exited with status N"; the
    real cause the runner already printed as one structured
    cubed-core/decode-runner-error-v1 JSON line (ssh failure, an artifact
    mismatch, CUDA OOM, ...) must not be left buried in the log the caller
    would otherwise have to open separately.
    """
    error_line = json.dumps(
        {
            "schema": "cubed-core/decode-runner-error-v1",
            "schema_version": 1,
            "error": {
                "code": "backend_failed",
                "message": "remote box reported CUDA out of memory",
                "details": {},
            },
        }
    )
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_failing_runner(
            tmp_path,
            "\n".join(
                (
                    "import sys",
                    f"print({error_line!r})",
                    "print('noise printed after the structured error line')",
                    "sys.exit(1)",
                )
            ),
        ),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "failed"
    assert final["error"] == (
        "decode runner exited with status 1: remote box reported CUDA out of memory"
    )


def test_decode_job_error_falls_back_to_the_last_log_line(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    """Without a structured error line, the last non-empty log line is used
    instead of leaving the caller with only the bare exit status.
    """
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_failing_runner(
            tmp_path,
            "\n".join(
                (
                    "import sys",
                    "print('loading model weights')",
                    "print('CUDA error: out of memory')",
                    "sys.exit(1)",
                )
            ),
        ),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "failed"
    assert final["error"] == "decode runner exited with status 1: CUDA error: out of memory"


def test_failed_decode_attempt_is_log_free_and_searchable_after_restart(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    raw_runner_detail = "CUDA error in /private/provider/job-123/model.onnx"
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_failing_runner(
            tmp_path,
            "\n".join(
                (
                    "import sys",
                    f"print({raw_runner_detail!r})",
                    "sys.exit(17)",
                )
            ),
        ),
    )
    terminal_path: Path
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "failed"
        assert final["outcome"] == "failed"
        assert final["failure"] == {
            "code": "runner-exit",
            "message": "The decode runner exited with status 17.",
            "retryable": True,
        }
        # The live detail remains useful while the process exists.
        assert raw_runner_detail in final["error"]
        live_listing = client.get("/api/decode/jobs", params={"q": job_id[:12]})
        assert live_listing.status_code == 200
        live_row = live_listing.json()["jobs"][0]
        assert live_row["error"] is None
        assert live_row["failure"] == final["failure"]
        assert raw_runner_detail not in json.dumps(live_row)
        terminal_path = settings.workspace / "decode-jobs" / job_id / "run-terminal.json"
        _wait_for_path(terminal_path)

    terminal_text = terminal_path.read_text(encoding="utf-8")
    assert raw_runner_detail not in terminal_text
    assert "/private/" not in terminal_text
    terminal = json.loads(terminal_text)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(
        terminal,
        json.loads(
            (REPO_ROOT / "schemas" / "decode-run-terminal-v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        listing = client.get("/api/decode/jobs", params={"q": job_id[:12]})
        assert listing.status_code == 200
        rows = listing.json()["jobs"]
        assert [row["job_id"] for row in rows] == [job_id]
        restored = rows[0]
        assert restored["status"] == "failed"
        assert restored["outcome"] == "failed"
        assert restored["log"] == ""
        assert restored["log_truncated"] is False
        assert restored["failure"] == terminal["failure"]
        assert restored["video_sha256"] == hashlib.sha256(b"decode-test-video-bytes").hexdigest()
        assert restored["result_available"] is False


def test_cancelled_decode_attempt_survives_service_restart(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_failing_runner(
            tmp_path,
            "\n".join(("import time", "time.sleep(30)")),
        ),
    )
    workspace = _workspace_for(settings)
    service = DecodeJobService(settings, workspace)
    job_id = service.submit(CAPTURE_ID)["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with service._lock:
            process_started = job_id in service._processes
        if process_started:
            break
        time.sleep(0.01)
    else:
        service.close()
        raise AssertionError("decode subprocess did not start")

    with pytest.raises(DecodeJobError, match="active decode run"):
        service.trash(job_id)

    service.close()
    terminal_path = settings.workspace / "decode-jobs" / job_id / "run-terminal.json"
    _wait_for_path(terminal_path)

    restored = DecodeJobService(settings, workspace)
    try:
        status = restored.status(job_id)
        assert status["status"] == "cancelled"
        assert status["outcome"] == "cancelled"
        assert status["failure"] == {
            "code": "service-shutdown",
            "message": "The decode service stopped before this run completed.",
            "retryable": True,
        }
        assert status["log"] == ""
        assert status["result_available"] is False
    finally:
        restored.close()


def test_decode_job_error_is_untouched_on_a_clean_exit(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    """The promotion only applies to a nonzero exit; a clean run's error
    field stays None regardless of what the runner printed.
    """
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)

    assert final["status"] == "succeeded"
    assert final["error"] is None


def test_decode_job_reports_an_abstention_as_a_succeeded_job(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(
            tmp_path,
            status="abstained",
            moves=[],
            solved_reached=False,
        ),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]
        assert final["outcome"] == "abstained"
        assert final["result_status"] == "abstained"
        assert final["result_available"] is True
        assert final["replay_solved_reached"] is None
        assert final["replay_check"]["performed"] is False
        assert client.get(f"/api/decode/jobs/{job_id}/result").status_code == 200
        _wait_for_path(settings.workspace / "decode-jobs" / job_id / "run-terminal.json")

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        restored = client.get(f"/api/decode/jobs/{job_id}").json()
        assert restored["status"] == "succeeded"
        assert restored["outcome"] == "abstained"
        assert restored["result_status"] == "abstained"


def test_decode_result_endpoint_detects_a_tampered_artifact(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded", final["error"]
        assert client.get(f"/api/decode/jobs/{job_id}/result").status_code == 200

        artifact = settings.workspace / "decode-jobs" / job_id / "decode-result.json"
        document = json.loads(artifact.read_text(encoding="utf-8"))
        document["moves"] = ["R"]
        artifact.write_text(json.dumps(document, indent=2), encoding="utf-8")

        tampered = client.get(f"/api/decode/jobs/{job_id}/result")
        assert tampered.status_code == 409
        assert "changed after the job succeeded" in tampered.json()["detail"]


def _workspace_for(settings: Settings) -> Workspace:
    workspace = Workspace(
        settings.workspace,
        max_upload_bytes=settings.max_upload_bytes,
    )
    workspace.initialize()
    return workspace


def _write_fake_succeeded_decode_job(
    workspace: Workspace,
    *,
    job_id: str,
    capture_id: str = CAPTURE_ID,
    finished_at: str,
    result_status: str = "completed",
    solved_reached: bool | None = True,
    performed: bool = True,
) -> dict[str, Any]:
    """Write a self-consistent completed decode job directory without running anything.

    Only the two artifacts the index and the result route actually read --
    ``decode-result.json`` and ``decode-receipt.json`` -- need to exist and
    agree with each other; nothing here ever ran a decode subprocess.
    """

    job_dir = workspace.root / "decode-jobs" / job_id
    job_dir.mkdir(parents=True, mode=0o700)
    result = {"schema": "cubed-core/decode-result", "recording_id": capture_id, "moves": []}
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (job_dir / "decode-result.json").write_bytes(result_payload)
    receipt = {
        "schema": "cubed-core/decode-job-receipt-v1",
        "schema_version": 1,
        "job_id": job_id,
        "capture_id": capture_id,
        "finished_at": finished_at,
        "profile": "local_camera_v1",
        "request_sha256": "1" * 64,
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "result_bytes": len(result_payload),
        "result_status": result_status,
        "replay_check": {
            "performed": performed,
            "solved_reached": solved_reached,
            "move_count": 0,
            "detail": "fixture receipt",
        },
        "evidence_scope": "reconstruction-evidence-with-replay-check",
        "runner_provenance": {"runner_kind": "native"},
        "runtime_assets": [],
    }
    (job_dir / "decode-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"result_bytes": result_payload, "result_sha256": receipt["result_sha256"]}


def test_decode_job_index_rebuilds_from_a_fake_completed_job_dir(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    workspace = _workspace_for(settings)
    job_id = "d" * 32
    fake = _write_fake_succeeded_decode_job(
        workspace,
        job_id=job_id,
        finished_at="2026-07-01T00:00:00+00:00",
    )

    service = DecodeJobService(settings, workspace)

    status = service.status(job_id)
    assert status["status"] == "succeeded"
    assert status["capture_id"] == CAPTURE_ID
    assert status["finished_at"] == "2026-07-01T00:00:00+00:00"
    assert status["result_available"] is True
    assert status["result_status"] == "completed"
    assert status["replay_solved_reached"] is True
    assert status["result_sha256"] == fake["result_sha256"]
    assert status["result_bytes"] == len(fake["result_bytes"])
    assert status["log"] == ""
    assert status["stage"] is None

    listing = service.list_for_capture(CAPTURE_ID)
    assert [row["job_id"] for row in listing] == [job_id]

    payload = service.result(job_id)
    assert payload == fake["result_bytes"]


def test_decode_job_index_restores_an_abstained_result_status(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    workspace = _workspace_for(settings)
    job_id = "d" * 32
    _write_fake_succeeded_decode_job(
        workspace,
        job_id=job_id,
        finished_at="2026-07-01T00:00:00+00:00",
        result_status="abstained",
        solved_reached=None,
        performed=False,
    )

    service = DecodeJobService(settings, workspace)

    status = service.status(job_id)
    assert status["result_status"] == "abstained"
    assert status["replay_solved_reached"] is None
    assert status["replay_check"]["performed"] is False


def test_decode_job_index_skips_a_corrupt_receipt(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    workspace = _workspace_for(settings)
    job_id = "e" * 32
    job_dir = workspace.root / "decode-jobs" / job_id
    job_dir.mkdir(parents=True, mode=0o700)
    (job_dir / "decode-result.json").write_text("{}", encoding="utf-8")
    (job_dir / "decode-receipt.json").write_text("not valid json{", encoding="utf-8")

    with caplog.at_level("WARNING"):
        service = DecodeJobService(settings, workspace)

    with pytest.raises(DecodeJobError, match="decode job not found"):
        service.status(job_id)
    assert service.list_for_capture(CAPTURE_ID) == []
    assert any(job_id in record.message for record in caplog.records)


def test_decode_job_index_warns_about_orphaned_runs_missing_their_receipt(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hard server restart between the runner writing its result and this
    job's decode receipt being written leaves a finished run with no
    receipt. The rebuild contract still correctly drops it from the index
    (it never reached a verified success), but must not do so silently: the
    operator needs a clear, path-naming warning per orphan to notice and
    recover the result by hand, plus a count across every orphan found.
    """
    settings = _settings(tmp_path, color_calibration_payload)
    workspace = _workspace_for(settings)
    orphan_ids = ["7" * 32, "8" * 32]
    for job_id in orphan_ids:
        job_dir = workspace.root / "decode-jobs" / job_id
        job_dir.mkdir(parents=True, mode=0o700)
        (job_dir / "decode-result.json").write_text("{}", encoding="utf-8")

    with caplog.at_level("WARNING"):
        service = DecodeJobService(settings, workspace)

    assert service.list_for_capture(CAPTURE_ID) == []
    messages = [record.message for record in caplog.records]
    for job_id in orphan_ids:
        assert any(job_id in message and "no decode receipt" in message for message in messages)
    assert any("2 orphaned run" in message for message in messages)


def test_decode_capture_job_list_404s_for_an_unknown_capture(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        response = client.get(f"/api/captures/{'f' * 32}/decode-jobs")
    assert response.status_code == 404


def test_decode_capture_job_list_merges_disk_and_live_newest_first(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        first = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        assert _wait_for_terminal(client, first)["status"] == "succeeded"

    # Restart: a fresh app (and DecodeJobService) reusing the same workspace
    # must index `first` from disk, then a newly submitted job must sort ahead
    # of it and both must be reachable through the capture-scoped route.
    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        second = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        assert _wait_for_terminal(client, second)["status"] == "succeeded"

        listing = client.get(f"/api/captures/{CAPTURE_ID}/decode-jobs")
        assert listing.status_code == 200
        body = listing.json()
        job_ids = [row["job_id"] for row in body["jobs"]]
        assert job_ids == [second, first]
        for row in body["jobs"]:
            assert row["status"] == "succeeded"
            assert row["result_available"] is True
            assert row["result_status"] == "completed"

        direct = client.get(f"/api/decode/jobs/{first}")
        assert direct.status_code == 200
        assert direct.json()["status"] == "succeeded"
        assert client.get(f"/api/decode/jobs/{first}/result").status_code == 200


def test_each_decode_attempt_snapshots_its_selected_calibration(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    flat_color_centroids_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    capture_dir = settings.workspace / "captures" / CAPTURE_ID
    capture_calibration = capture_dir / "calibration.json"
    original_calibration = capture_calibration.read_bytes()
    capture_receipt = json.loads((capture_dir / "capture.json").read_text())
    (capture_dir / "checksums.sha256").write_text(
        f"{capture_receipt['video']['sha256']}  source.mov\n"
        f"{capture_receipt['calibration']['sha256']}  calibration.json\n",
        encoding="utf-8",
    )

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        first = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        assert _wait_for_terminal(client, first)["status"] == "succeeded"
        first_dir = settings.workspace / "decode-jobs" / first
        first_snapshot = first_dir / "calibration.json"
        first_request = json.loads((first_dir / "job-request.json").read_text())
        assert first_snapshot.read_bytes() == original_calibration
        assert first_request["inputs"]["calibration"] == str(first_snapshot)
        assert (
            first_request["inputs"]["calibration_sha256"]
            == hashlib.sha256(original_calibration).hexdigest()
        )

        replaced = client.post(
            f"/api/captures/{CAPTURE_ID}/sidecars/calibration",
            files={
                "sidecar": (
                    "replacement.json",
                    json.dumps(flat_color_centroids_payload).encode(),
                    "application/json",
                )
            },
        )
        assert replaced.status_code == 200, replaced.text
        replacement_calibration = capture_calibration.read_bytes()
        assert replacement_calibration != original_calibration

        second = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        assert _wait_for_terminal(client, second)["status"] == "succeeded"
        second_dir = settings.workspace / "decode-jobs" / second
        second_snapshot = second_dir / "calibration.json"
        second_request = json.loads((second_dir / "job-request.json").read_text())

    assert first_snapshot.read_bytes() == original_calibration
    assert second_snapshot.read_bytes() == replacement_calibration
    assert second_request["inputs"]["calibration"] == str(second_snapshot)
    assert (
        second_request["inputs"]["calibration_sha256"]
        == hashlib.sha256(replacement_calibration).hexdigest()
    )
    assert (
        second_request["inputs"]["calibration_sha256"]
        != first_request["inputs"]["calibration_sha256"]
    )


def test_delete_decode_run_moves_only_the_job_to_recoverable_trash(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(
        tmp_path,
        color_calibration_payload,
        decode_mode="external",
        decode_command=_runner(tmp_path),
    )
    capture_dir = settings.workspace / "captures" / CAPTURE_ID
    video_before = (capture_dir / "source.mov").read_bytes()
    calibration_before = (capture_dir / "calibration.json").read_bytes()

    with TestClient(create_app(settings), headers=ADMIN_HEADERS) as client:
        job_id = client.post(f"/api/captures/{CAPTURE_ID}/decode-jobs").json()["job_id"]
        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "succeeded"
        job_dir = settings.workspace / "decode-jobs" / job_id
        assert job_dir.is_dir()

        response = client.delete(f"/api/decode/jobs/{job_id}")
        assert response.status_code == 200
        value = response.json()
        assert value == {
            "schema": "cubed-core/decode-run-delete-v1",
            "schema_version": 1,
            "job_id": job_id,
            "capture_id": CAPTURE_ID,
            "trashed": True,
            "recoverable": True,
            "trash_id": value["trash_id"],
        }
        trash_dir = settings.workspace / "run-trash" / "decode-jobs" / value["trash_id"]
        assert not job_dir.exists()
        assert trash_dir.is_dir()
        assert (trash_dir / "decode-result.json").is_file()
        assert client.get(f"/api/decode/jobs/{job_id}").status_code == 404
        assert client.get(f"/api/decode/jobs/{job_id}/result").status_code == 404
        assert job_id not in {
            row["job_id"] for row in client.get("/api/decode/jobs").json()["jobs"]
        }

    assert (capture_dir / "source.mov").read_bytes() == video_before
    assert (capture_dir / "calibration.json").read_bytes() == calibration_before
    assert (capture_dir / "capture.json").is_file()


def _minimal_result(**overrides: Any) -> dict[str, Any]:
    manifest = json.loads(
        (REPO_ROOT / "config" / "decode-runtime-v1.json").read_text(encoding="utf-8")
    )
    profile = manifest["profiles"]["local_camera_v1"]
    value: dict[str, Any] = {
        "schema": "cubed-core/decode-result",
        "schema_version": 1,
        "recording_id": CAPTURE_ID,
        "status": "completed",
        "profile": "local_camera_v1",
        "config": {
            "name": profile["name"],
            "cfg_hash": profile["cfg_hash"],
            "cfg_hash_algorithm": "posix-cksum",
        },
        "inputs": [
            {"id": "video", "sha256": "0" * 64},
            {"id": "color-calibration-v1", "sha256": "1" * 64},
        ],
        "moves": list(SOLVING_MOVES),
        "endpoint": {"solved_reached": True},
        "evaluation": None,
        "provenance": {
            "runtime_version": "decode-test",
            "finished_at": "2026-07-25T12:30:00+00:00",
        },
    }
    value.update(overrides)
    return value


def test_validate_decode_result_rejects_an_evaluation_block() -> None:
    value = _minimal_result(
        evaluation={
            "metric": "reach-ll-onset-with-correct-pre-ll-state",
            "reach_ll": True,
            "teacher_sha256": "2" * 64,
        }
    )
    with pytest.raises(DecodeJobError, match="may not carry an evaluation block"):
        validate_decode_result(value, capture_id=CAPTURE_ID, repo_root=REPO_ROOT)


def test_validate_decode_result_reserves_and_binds_public_ble_diagnostics() -> None:
    diagnostic = _ground_truth_diagnostic(video_sha256="0" * 64)
    raw_runner_value = _minimal_result(
        ground_truth_diagnostic=diagnostic,
    )
    with pytest.raises(
        DecodeJobError,
        match="runner output may not carry ground-truth diagnostics",
    ):
        validate_decode_result(
            raw_runner_value,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )

    accepted = validate_decode_result(
        raw_runner_value,
        capture_id=CAPTURE_ID,
        repo_root=REPO_ROOT,
        allow_ground_truth_diagnostic=True,
        expected_ground_truth_video_sha256="0" * 64,
    )
    assert accepted["ground_truth_diagnostic"]["comparison"]["distance"] == 0

    drifted = json.loads(json.dumps(raw_runner_value))
    drifted["ground_truth_diagnostic"]["comparison"]["distance"] = 1
    with pytest.raises(
        DecodeJobError,
        match="does not match the closed camera result",
    ):
        validate_decode_result(
            drifted,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
            allow_ground_truth_diagnostic=True,
            expected_ground_truth_video_sha256="0" * 64,
        )


def test_validate_decode_result_pins_identity_and_config() -> None:
    assert (
        validate_decode_result(
            _minimal_result(),
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )["status"]
        == "completed"
    )

    with pytest.raises(DecodeJobError, match="identity does not match"):
        validate_decode_result(
            _minimal_result(recording_id="e" * 32),
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )

    drifted = _minimal_result()
    drifted["config"]["cfg_hash"] = "1"
    with pytest.raises(DecodeJobError, match="CFG_HASH does not match"):
        validate_decode_result(drifted, capture_id=CAPTURE_ID, repo_root=REPO_ROOT)

    with pytest.raises(DecodeJobError, match="shipped Draft 2020-12 schema"):
        validate_decode_result(
            _minimal_result(moves=["not a move"]),
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )


def test_validate_decode_result_replays_the_reconstruction_checkpoint_timeline() -> None:
    from cubed_core.cube import Cube

    replay = Cube.solved().apply_algorithm(SCRAMBLE)
    states = [replay.state]
    for move in SOLVING_MOVES:
        replay.apply_move(move)
        states.append(replay.state)
    workstation = {
        "schema": "cubed-core/decode-workstation-v1",
        "schema_version": 1,
        "video": {
            "sha256": "0" * 64,
            "bytes": 100,
            "fps": 120.0,
            "frame_count": 50,
            "width": 1080,
            "height": 1920,
            "encoded": {
                "fps": 120.0,
                "frame_count": 50,
                "width": 1080,
                "height": 1920,
            },
        },
        "initialization": {"scramble": SCRAMBLE},
        "window": [0, 49],
        "warnings": [],
        "frames": {},
        "reconstruction": {
            "states": states,
            "solved_reached": True,
            "timeline": {
                "moves": [
                    {"move": move, "frame": frame}
                    for move, frame in zip(
                        SOLVING_MOVES,
                        [10, 20, 20, 30],
                        strict=True,
                    )
                ],
                "timing_basis": "decoder-checkpoint",
            },
        },
    }
    value = _minimal_result(workstation=workstation)

    assert (
        validate_decode_result(
            value,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )["workstation"]["reconstruction"]["states"]
        == states
    )

    drifted = json.loads(json.dumps(value))
    drifted["workstation"]["reconstruction"]["states"][1] = states[0]
    with pytest.raises(
        DecodeJobError,
        match="checkpoint timeline does not match the result",
    ):
        validate_decode_result(
            drifted,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )

    unordered = json.loads(json.dumps(value))
    unordered["workstation"]["reconstruction"]["timeline"]["moves"][2]["frame"] = 5
    with pytest.raises(
        DecodeJobError,
        match="checkpoint timeline is inconsistent",
    ):
        validate_decode_result(
            unordered,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )

    starts_at_window = json.loads(json.dumps(value))
    starts_at_window["workstation"]["reconstruction"]["timeline"]["moves"][0]["frame"] = 0
    with pytest.raises(
        DecodeJobError,
        match="checkpoint timeline is inconsistent",
    ):
        validate_decode_result(
            starts_at_window,
            capture_id=CAPTURE_ID,
            repo_root=REPO_ROOT,
        )


def test_native_result_acceptance_binds_the_sealed_video_input(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
) -> None:
    settings = _settings(tmp_path, color_calibration_payload)
    workspace = _workspace_for(settings)
    service = DecodeJobService(settings, workspace)
    job_id = "8" * 32
    job_dir = settings.workspace / "decode-jobs" / job_id
    job_dir.mkdir(parents=True)
    output_path = job_dir / "decode-result.json"
    expected_video_sha256 = hashlib.sha256(b"decode-test-video-bytes").hexdigest()
    job = DecodeJob(
        job_id=job_id,
        capture_id=CAPTURE_ID,
        job_dir=job_dir,
        request_path=job_dir / "job-request.json",
        output_path=output_path,
        scramble=SCRAMBLE,
        runner_command=(),
        runner_provenance={"runner_kind": "native"},
        runtime_assets=[],
        status="running",
        created_at="2026-07-25T12:00:00+00:00",
        video_sha256=expected_video_sha256,
        require_video_input_binding=True,
    )
    result = _minimal_result()
    result["inputs"] = [
        {"id": "reads", "sha256": "1" * 64},
        {"id": "color-calibration-v1", "sha256": "2" * 64},
    ]
    runner_output_path = job.runner_output_path

    try:
        runner_output_path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(DecodeJobError, match="video input does not match"):
            service._load_result(job)

        result["inputs"].insert(
            0,
            {"id": "video", "sha256": "3" * 64},
        )
        runner_output_path.write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(DecodeJobError, match="video input does not match"):
            service._load_result(job)

        result["inputs"][0]["sha256"] = expected_video_sha256
        runner_output_path.write_text(json.dumps(result), encoding="utf-8")
        validated, _payload = service._load_result(job)
        assert validated["inputs"][0] == {
            "id": "video",
            "sha256": expected_video_sha256,
        }

        # Compatible external runners are still allowed to omit this additive
        # native binding.
        job.require_video_input_binding = False
        result["inputs"] = result["inputs"][1:]
        runner_output_path.write_text(json.dumps(result), encoding="utf-8")
        validated, _payload = service._load_result(job)
        assert all(receipt["id"] != "video" for receipt in validated["inputs"])
    finally:
        service.close()


def test_replay_decode_result_reports_both_outcomes_without_claiming_accuracy() -> None:
    solved = replay_decode_result(_minimal_result(), scramble=SCRAMBLE)
    assert solved.performed is True
    assert solved.solved_reached is True
    assert "replay from the sealed scramble to a solved cube" in solved.detail

    unsolved = replay_decode_result(_minimal_result(moves=["R"]), scramble=SCRAMBLE)
    assert unsolved.performed is True
    assert unsolved.solved_reached is False

    abstained = replay_decode_result(
        _minimal_result(status="abstained", moves=[], endpoint={"solved_reached": False}),
        scramble=SCRAMBLE,
    )
    assert abstained.performed is False
    assert abstained.solved_reached is None
    assert abstained.move_count == 0


class _RecordingRunner:
    """Stand-in for ``subprocess.run`` that records commands and fakes outputs."""

    def __init__(self, *, nvdec_ok: bool, result: dict[str, Any]) -> None:
        self.nvdec_ok = nvdec_ok
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        stdin: Any,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append({"command": list(command), "cwd": cwd, "env": dict(env)})
        if any(part.endswith("check_nvdec.py") for part in command):
            return subprocess.CompletedProcess(command, 0 if self.nvdec_ok else 1)
        if any(part.endswith("geo_read.py") for part in command):
            Path(command[command.index("--out") + 1]).write_bytes(b"reads")
        elif any(part.endswith("gen_motion_events.py") for part in command):
            Path(command[-1]).write_text("[]", encoding="utf-8")
        elif any(part.endswith("extract_alignfeat.py") for part in command):
            Path(command[command.index("--out") + 1]).write_bytes(b"alignfeat")
        elif any(part.endswith("run_research_decode.sh") for part in command):
            Path(env["CUBED_RESULT_JSON"]).write_text(
                json.dumps(self.result, indent=2),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)


def _native_request(repo_root: Path, job_dir: Path, capture_dir: Path) -> Path:
    assets = []
    for requirement_id, relative in (
        ("pose-model", MODEL_REQUIREMENT_PATHS[0]),
        ("alignment-model", MODEL_REQUIREMENT_PATHS[1]),
        ("read-trust-model", MODEL_REQUIREMENT_PATHS[2]),
    ):
        payload = (repo_root / relative).read_bytes()
        assets.append(
            {
                "id": requirement_id,
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    request_path = job_dir / "job-request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema": "cubed-core/decode-job-request-v1",
                "schema_version": 1,
                "job_id": job_dir.name,
                "capture_id": CAPTURE_ID,
                "created_at": "2026-07-25T12:00:00+00:00",
                "profile": "local_camera_v1",
                "config": {
                    "name": "cubed-core-local-camera-v1",
                    "cfg_hash": "2429939321",
                    "cfg_hash_algorithm": "posix-cksum",
                },
                "inference_policy": {
                    "mode": "camera-only",
                    "ground_truth": "unavailable",
                    "evaluation": "must-be-absent",
                },
                "inputs": {
                    "video": str(capture_dir / "source.mov"),
                    "calibration": str(capture_dir / "calibration.json"),
                    "calibration_sha256": hashlib.sha256(
                        (capture_dir / "calibration.json").read_bytes()
                    ).hexdigest(),
                    "scramble": SCRAMBLE,
                },
                "runtime_assets": assets,
                "expected_output": {
                    "schema": "cubed-core/decode-result",
                    "schema_version": 1,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return request_path


def test_native_decode_runner_sequences_the_four_pipeline_stages(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    _capture(workspace, color_calibration_payload)
    capture_dir = workspace / "captures" / CAPTURE_ID
    job_id = "f" * 32
    job_dir = workspace / "decode-jobs" / job_id
    job_dir.mkdir(parents=True)
    request_path = _native_request(repo_root, job_dir, capture_dir)
    output_path = job_dir / "decode-result.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("CUBED_CORE_REPO_ROOT", str(repo_root))
    monkeypatch.delenv("CUBED_NVDEC", raising=False)

    runner = _RecordingRunner(nvdec_ok=False, result=_minimal_result())
    native_decode_runner.run(
        request_path,
        output_path,
        scratch_root=scratch,
        runner=runner,
    )

    markers = [
        line.split("] ", 1)[1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[cubed-core:stage] ")
    ]
    assert markers == ["reads", "events", "alignfeat", "decode"]

    stages = [
        next(
            (
                script
                for script in (
                    "check_nvdec.py",
                    "geo_read.py",
                    "gen_motion_events.py",
                    "extract_alignfeat.py",
                    "run_research_decode.sh",
                )
                if any(part.endswith(script) for part in call["command"])
            ),
            None,
        )
        for call in runner.calls
    ]
    assert stages == [
        "check_nvdec.py",
        "geo_read.py",
        "gen_motion_events.py",
        "extract_alignfeat.py",
        "run_research_decode.sh",
    ]

    geo_call = runner.calls[1]
    assert "--gpu-reads" in geo_call["command"] and "--gpu-warp" in geo_call["command"]
    # The NVDEC probe failed under the default auto policy, so the host decoder
    # is used rather than failing the job.
    assert "--gpu-decode" not in geo_call["command"]
    assert geo_call["command"][geo_call["command"].index("--tag") + 1] == job_id
    assert geo_call["env"]["CUBED_FACE_POSE_MODEL"].endswith("face-pose.onnx")
    assert geo_call["env"]["CUBED_ALIGNED_MODEL"].endswith("alignment-classifier.onnx")
    assert str(repo_root) in geo_call["env"]["PYTHONPATH"]
    assert geo_call["cwd"] == str(repo_root)

    events_call = runner.calls[2]
    assert events_call["command"][-2] == job_id
    assert (scratch / f"reads_{job_id}_v2.pkl").exists() is False

    decode_call = runner.calls[4]
    assert decode_call["command"][-1] == job_id
    assert decode_call["env"]["CUBED_SCRAMBLE"] == SCRAMBLE
    assert decode_call["env"]["CUBED_RECORDING_ID"] == CAPTURE_ID
    assert decode_call["env"]["CUBED_RESEARCH_DECODE_EXECUTE"] == "1"
    assert decode_call["env"]["CUBED_RESULT_JSON"] == str(output_path)
    assert decode_call["env"]["CUBED_RESULT_VIDEO_INPUT"] == str(capture_dir / "source.mov")
    assert decode_call["env"]["CUBED_READS"] == str(scratch / f"reads_{job_id}_occaware.pkl")
    assert decode_call["env"]["CUBED_EVENTS"] == str(scratch / f"motion_events_{job_id}.json")
    assert decode_call["env"]["CUBED_CENTROIDS"] == str(job_dir / "centroids.json")
    # The trust model path is part of the hashed decode flag tokens, so the
    # runner must never override it through the environment. It stages the
    # verified asset at the canonical repo-relative path instead.
    assert "CUBED_TRUST_NPZ" not in decode_call["env"]
    canonical_trust = repo_root / "datasets" / "read_trust" / "trust_v1_numpy.npz"
    assert canonical_trust.is_file()
    assert canonical_trust.stat().st_size > 0

    # The result lands in the job directory, the intermediates are copied beside
    # it, and the per-tag scratch files are removed.
    assert json.loads(output_path.read_text(encoding="utf-8"))["recording_id"] == CAPTURE_ID
    assert (job_dir / f"reads_{job_id}_occaware.pkl").is_file()
    assert (job_dir / f"motion_events_{job_id}.json").is_file()
    assert (job_dir / f"alignfeat_{job_id}_new.npz").is_file()
    assert sorted(path.name for path in scratch.iterdir()) == []

    centroids = json.loads((job_dir / "centroids.json").read_text(encoding="utf-8"))
    assert sorted(centroids) == ["blue", "green", "orange", "red", "white", "yellow"]
    assert centroids["white"] == color_calibration_payload["centroids"]["white"]


def test_native_decode_runner_requires_nvdec_when_the_policy_demands_it(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    _capture(workspace, color_calibration_payload)
    job_id = "a" * 32
    job_dir = workspace / "decode-jobs" / job_id
    job_dir.mkdir(parents=True)
    request_path = _native_request(repo_root, job_dir, workspace / "captures" / CAPTURE_ID)
    monkeypatch.setenv("CUBED_CORE_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("CUBED_NVDEC", "require")

    with pytest.raises(native_decode_runner.NativeDecodeRunnerError) as excinfo:
        native_decode_runner.run(
            request_path,
            job_dir / "decode-result.json",
            scratch_root=tmp_path,
            runner=_RecordingRunner(nvdec_ok=False, result=_minimal_result()),
        )
    assert excinfo.value.code == "nvdec_unavailable"


def test_native_decode_runner_rejects_a_runtime_asset_that_changed(
    tmp_path: Path,
    color_calibration_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    _capture(workspace, color_calibration_payload)
    job_id = "9" * 31 + "a"
    job_dir = workspace / "decode-jobs" / job_id
    job_dir.mkdir(parents=True)
    request_path = _native_request(repo_root, job_dir, workspace / "captures" / CAPTURE_ID)
    (repo_root / MODEL_REQUIREMENT_PATHS[0]).write_bytes(b"a different pose model")
    monkeypatch.setenv("CUBED_CORE_REPO_ROOT", str(repo_root))

    with pytest.raises(native_decode_runner.NativeDecodeRunnerError) as excinfo:
        native_decode_runner.run(
            request_path,
            job_dir / "decode-result.json",
            scratch_root=tmp_path,
            runner=_RecordingRunner(nvdec_ok=False, result=_minimal_result()),
        )
    assert excinfo.value.code == "artifact_mismatch"
