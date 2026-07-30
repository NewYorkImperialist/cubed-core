from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROVISION_SCRIPT = REPOSITORY_ROOT / "scripts" / "provision_gpu_box.sh"
REMOTE_DECODE_RUNNER = REPOSITORY_ROOT / "scripts" / "remote_decode_runner.sh"
REMOTE_DOCTOR = REPOSITORY_ROOT / "scripts" / "remote_doctor.sh"
MAKEFILE = REPOSITORY_ROOT / "Makefile"


def _generated_environment_block() -> str:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")
    start_marker = 'cat >"$ENV_BLOCK_FILE" <<ENVBLOCK\n'
    start = source.index(start_marker) + len(start_marker)
    end = source.index("\nENVBLOCK", start)
    return source[start:end]


def test_gpu_box_provisioning_emits_the_per_job_decode_selector_contract() -> None:
    block = _generated_environment_block()

    assert "CUBED_REMOTE_SSH_DEST=$DEST" in block
    assert "CUBED_CORE_DECODE_MODE=native" in block
    assert "CUBED_CORE_DECODE_COMMAND=" not in block
    assert "CUBED_CORE_TRACKER_MODE" not in block
    assert "CUBED_REMOTE_DECODE_VENV=.venv-decode-gpu" in block


def test_gpu_box_shell_entrypoints_parse_as_bash() -> None:
    for script in (PROVISION_SCRIPT, REMOTE_DECODE_RUNNER, REMOTE_DOCTOR):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_gpu_box_host_preflight_requires_runtime_and_decode_support() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert 'LOCAL_MANIFEST="$LOCAL_ASSET_DIR/camera-tracker-v1-runtime/manifest.json"' in source
    assert 'LOCAL_TRUST_MODEL="$LOCAL_ASSET_DIR/trust_v1_numpy.npz"' in source
    assert 'LOCAL_CALIBRATION="$LOCAL_ASSET_DIR/calibration_gan12.json"' in source
    assert (
        'for asset_path in "$LOCAL_MANIFEST" "$LOCAL_TRUST_MODEL" "$LOCAL_CALIBRATION"; do'
        in source
    )
    assert "run 'make download-assets' first" in source


def test_gpu_box_checkout_sync_excludes_local_data_and_scratch_roots() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    for root in (
        "/.agents/",
        "/.claude/",
        "/.codex/",
        "/data/",
        "/datasets/",
        "/output/",
        "/tmp/",
        "/weights/",
        "/workspace/",
    ):
        assert f"--exclude '{root}'" in source
    assert "--include '/models/local/.gitkeep'" in source
    assert "--exclude '/models/local/*'" in source
    assert "--exclude 'data/'" not in source


def test_gpu_box_checkout_sync_mirrors_private_default_artifact_ignores() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert "--include '/.env.example'" in source
    assert "--exclude '.env*'" in source
    for pattern in (
        "*.key",
        "*.onnx",
        "*.npz",
        "*.joblib",
        "*.mp4",
        "*.zip",
        "*.tar.gz",
    ):
        assert f"--exclude '{pattern}'" in source


def test_gpu_box_receipt_allows_the_synced_checkout_as_a_safe_directory() -> None:
    source = PROVISION_SCRIPT.read_text(encoding="utf-8")

    assert 'GIT_SAFE=(-c "safe.directory=$REPO_ROOT")' in source
    assert 'git "${GIT_SAFE[@]}" rev-parse HEAD' in source
    assert 'git "${GIT_SAFE[@]}" status --porcelain' in source


def test_decode_gpu_bootstrap_and_doctor_include_onnx_model_rewriting() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    research_dependencies = pyproject["project"]["optional-dependencies"]["research-gpu"]
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert "onnx==1.17.0" in research_dependencies
    assert (
        "$(DECODE_GPU_PYTHON) -c 'import av, cv2, numpy, onnx, onnxruntime, scipy, torch'"
    ) in makefile
    assert "CUBED_CORE_TRACKER_ONNX_PROVIDERS=CUDAExecutionProvider" in makefile


def test_gpu_doctors_require_a_real_alignment_model_cuda_session() -> None:
    provision = PROVISION_SCRIPT.read_text(encoding="utf-8")
    remote_doctor = (REPOSITORY_ROOT / "scripts" / "remote_doctor.sh").read_text(encoding="utf-8")

    assert "record doctor-decode-gpu 1 1" in provision
    assert "onnxruntime.get_available_providers" not in provision
    assert "make doctor-decode-gpu >/dev/null" in remote_doctor
    assert "alignment model inference used an active CUDA session" in remote_doctor


def test_remote_decode_runner_documents_external_mode_as_compatibility() -> None:
    source = REMOTE_DECODE_RUNNER.read_text(encoding="utf-8")

    assert "CUBED_CORE_DECODE_MODE=native" in source
    assert "Jobs without remote_host continue to use" in source
    assert "External mode remains supported" in source


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_remote_doctor(tmp_path: Path, policy: str) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "ssh-commands.log"
    _write_executable(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
set -eu
remote_command=${!#}
printf '%s\\n' "$remote_command" >>"$TEST_COMMAND_LOG"
case "$remote_command" in
  true) ;;
  *"nvidia-smi --query-gpu"*) printf '%s\\n' "Test GPU, 24576 MiB" ;;
  "test -x "*) ;;
  "mkdir -p "*) ;;
  *"check_nvdec.py"*) printf '%s\\n' "NVDEC decoder unavailable"; exit 1 ;;
  *"doctor-decode-gpu"*) printf '%s\\n' "active" ;;
  *"rev-parse HEAD"*) printf '%s\\n' "$TEST_REMOTE_HEAD" ;;
  "rm -rf "*) ;;
  *) printf 'unexpected ssh command: %s\\n' "$remote_command" >&2; exit 9 ;;
esac
""",
    )
    for command in ("ffmpeg", "scp"):
        _write_executable(bin_dir / command, "#!/usr/bin/env bash\nexit 0\n")

    environment = os.environ.copy()
    environment.update(
        {
            "CUBED_REMOTE_NVDEC": policy,
            "CUBED_REMOTE_SSH_DEST": "test@example.invalid",
            "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
            "TEST_COMMAND_LOG": str(command_log),
            "TEST_REMOTE_HEAD": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
        }
    )
    result = subprocess.run(
        ["bash", str(REMOTE_DOCTOR)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, command_log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("policy", "status", "returncode", "runs_probe"),
    [
        ("off", "SKIP", 0, False),
        ("auto", "WARN", 0, True),
        ("require", "FAIL", 1, True),
    ],
)
def test_remote_doctor_applies_nvdec_policy(
    tmp_path: Path,
    policy: str,
    status: str,
    returncode: int,
    runs_probe: bool,
) -> None:
    result, command_log = _run_remote_doctor(tmp_path, policy)

    assert result.returncode == returncode, result.stdout + result.stderr
    assert f"box.nvdec {status}" in result.stdout
    assert ("check_nvdec.py" in command_log) is runs_probe
