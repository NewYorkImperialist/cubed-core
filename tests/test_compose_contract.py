from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_image_copies_the_lock_and_declared_legal_files() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES ./" in dockerfile
    )


def test_runtime_image_copies_the_current_decoder_layout() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for directory in ("detect", "core", "analysis", "scripts"):
        assert f"COPY {directory}/ ./{directory}/" in dockerfile
    assert "COPY research/" not in dockerfile
    assert "COPY docs/research/" not in dockerfile


def test_runtime_image_accepts_the_full_decode_gpu_extra_set() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "label,decode,tracker-gpu,research-gpu)" in dockerfile
    assert (
        'uv_extras="--extra label --extra decode --extra tracker-gpu --extra research-gpu"'
        in dockerfile
    )


def _compose() -> dict[str, object]:
    value = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _service(name: str) -> dict[str, object]:
    services = _compose()["services"]
    assert isinstance(services, dict)
    service = services[name]
    assert isinstance(service, dict)
    return service


def _model_mount(service: dict[str, object]) -> dict[str, object]:
    volumes = service["volumes"]
    assert isinstance(volumes, list)
    matches = [
        volume
        for volume in volumes
        if isinstance(volume, dict) and volume.get("target") == "/models"
    ]
    assert len(matches) == 1
    return matches[0]


def test_base_container_stays_lightweight_and_camera_model_is_optional() -> None:
    service = _service("lab")
    build = service["build"]
    assert isinstance(build, dict)
    args = build["args"]
    assert isinstance(args, dict)
    assert args["CUBED_CORE_PYTHON_EXTRAS"] == "label,decode"
    environment = service["environment"]
    assert isinstance(environment, dict)
    assert environment["CUBED_CORE_DECODE_COMMAND"] == "${CUBED_CORE_DECODE_COMMAND:-}"
    assert environment["CUBED_CORE_DECODE_MODE"] == "${CUBED_CORE_DECODE_MODE:-disabled}"
    assert environment["CUBED_CORE_DECODE_RUNNER_LABEL"] == ("${CUBED_CORE_DECODE_RUNNER_LABEL:-}")
    assert environment["CUBED_CORE_TRACKER_MODEL_MANIFEST"] == (
        "${CUBED_CORE_TRACKER_MODEL_MANIFEST:-}"
    )
    assert all(
        not (isinstance(volume, dict) and volume.get("target") == "/models")
        for volume in service["volumes"]
    )


def test_label_cpu_profile_installs_cpu_runtime_and_mounts_models_read_only() -> None:
    service = _service("lab-label-cpu")
    assert service["profiles"] == ["label-cpu"]
    build = service["build"]
    assert isinstance(build, dict)
    assert build["args"]["CUBED_CORE_PYTHON_EXTRAS"] == ("label,decode,tracker-cpu")
    environment = service["environment"]
    assert isinstance(environment, dict)
    assert environment["CUBED_CORE_DECODE_COMMAND"] == "${CUBED_CORE_DECODE_COMMAND:-}"
    assert environment["CUBED_CORE_DECODE_MODE"] == "${CUBED_CORE_DECODE_MODE:-disabled}"
    assert environment["CUBED_CORE_DECODE_RUNNER_LABEL"] == ("${CUBED_CORE_DECODE_RUNNER_LABEL:-}")
    assert environment["CUBED_CORE_TRACKER_MODEL_MANIFEST"] == "/models/manifest.json"
    assert environment["CUBED_CORE_TRACKER_ONNX_PROVIDERS"] == ("CPUExecutionProvider")
    mount = _model_mount(service)
    assert mount["read_only"] is True
    assert mount["source"] == "${CUBED_CORE_TRACKER_MODELS_DIR:-./models/local}"


def test_decode_gpu_profile_requires_cuda_without_cpu_fallback() -> None:
    service = _service("lab-decode-gpu")
    assert service["profiles"] == ["decode-gpu"]
    assert service["gpus"] == "all"
    build = service["build"]
    assert isinstance(build, dict)
    assert build["args"]["CUBED_CORE_PYTHON_EXTRAS"] == ("label,decode,tracker-gpu,research-gpu")
    environment = service["environment"]
    assert isinstance(environment, dict)
    assert environment["CUBED_CORE_DECODE_COMMAND"] == "${CUBED_CORE_DECODE_COMMAND:-}"
    assert environment["CUBED_CORE_DECODE_MODE"] == "native"
    assert environment["CUBED_CORE_DECODE_RUNNER_LABEL"] == ("${CUBED_CORE_DECODE_RUNNER_LABEL:-}")
    assert environment["CUBED_CORE_TRACKER_MODEL_MANIFEST"] == "/models/manifest.json"
    assert environment["CUBED_CORE_TRACKER_ONNX_PROVIDERS"] == ("CUDAExecutionProvider")
    assert _model_mount(service)["read_only"] is True


def test_services_keep_internal_port_fixed_and_print_the_published_host_port() -> None:
    expected_command = [
        "cubed-core",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--browser-port",
        "${CUBED_CORE_PORT:-8000}",
        "--allow-network",
    ]
    expected_port = "${CUBED_CORE_BIND_ADDRESS:-127.0.0.1}:${CUBED_CORE_PORT:-8000}:8000"
    for name in ("lab", "lab-label-cpu", "lab-decode-gpu"):
        service = _service(name)
        assert service["command"] == expected_command
        assert service["ports"] == [expected_port]


def test_docker_wrapper_creates_a_caller_owned_bind_mount_before_compose() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    workspace_target = makefile.split("docker-workspace:", 1)[1].split("\ndocker-cpu:", 1)[0]
    cpu_target = makefile.split("docker-cpu:", 1)[1].split("\ndocker-gpu-check:", 1)[0]

    assert "CUBED_CORE_WORKSPACE:-$(CURDIR)/workspace" in workspace_target
    assert 'mkdir -p -- "$$workspace_path"' in workspace_target
    assert "python" not in workspace_target
    assert "docker-cpu: docker-workspace" in makefile
    assert 'CUBED_CORE_UID="$${CUBED_CORE_UID:-$$(id -u)}"' in cpu_target
    assert 'CUBED_CORE_GID="$${CUBED_CORE_GID:-$$(id -g)}"' in cpu_target
