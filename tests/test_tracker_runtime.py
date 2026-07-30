from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cubed_core import tracker_runtime
from cubed_core.model_artifacts import MODEL_ARTIFACT_MANIFEST_SCHEMA
from cubed_core.settings import Settings
from cubed_core.tracker_runtime import (
    TrackerRuntimePreparationError,
    build_camera_tracker_readiness,
    preload_tracker_provider_dependencies,
    probe_camera_model_cuda_session,
    tracker_onnx_providers_from_env,
)


def _manifest(tmp_path: Path) -> Path:
    artifacts = []
    for artifact_id, role in (
        ("alignment-v1", "alignment-classifier"),
        ("pose-v1", "face-pose"),
    ):
        path = tmp_path / f"{artifact_id}.onnx"
        path.write_bytes(artifact_id.encode())
        payload = path.read_bytes()
        artifacts.append(
            {
                "id": artifact_id,
                "role": role,
                "format": "onnx",
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "redistribution": "external-only",
                "license": "NOASSERTION",
                "source": "synthetic readiness fixture",
                "model_card": None,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
                "schema_version": 1,
                "profile": "camera-tracker-v1",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _settings(
    tmp_path: Path,
    *,
    providers: tuple[str, ...] = (),
    manifest: Path | None = None,
) -> Settings:
    return Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024,
        tracker_onnx_providers=providers,
        tracker_model_manifest=manifest,
    )


class _FakeOrt:
    __version__ = "1.23.2"

    def __init__(
        self,
        providers: tuple[str, ...],
        *,
        preload_failure: Exception | None = None,
        provider_failure: Exception | None = None,
    ) -> None:
        self.providers = providers
        self.preload_failure = preload_failure
        self.provider_failure = provider_failure
        self.calls: list[tuple[str, Any]] = []

    def preload_dlls(self, *, directory: str) -> None:
        self.calls.append(("preload", directory))
        if self.preload_failure is not None:
            raise self.preload_failure

    def get_available_providers(self) -> list[str]:
        self.calls.append(("providers", None))
        if self.provider_failure is not None:
            raise self.provider_failure
        return list(self.providers)


def _runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    ort: _FakeOrt | None,
    *,
    missing: str | None = None,
) -> None:
    modules = {
        "numpy": SimpleNamespace(__version__="1.26.4"),
        "cv2": SimpleNamespace(__version__="4.10.0"),
        "onnxruntime": ort,
    }

    def load(name: str) -> Any:
        if name == missing or modules[name] is None:
            raise ImportError(f"{name} unavailable")
        return modules[name]

    monkeypatch.setattr(tracker_runtime.importlib, "import_module", load)


def test_readiness_verifies_models_and_automatic_provider_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = _FakeOrt(("CUDAExecutionProvider", "CPUExecutionProvider"))
    _runtime_modules(monkeypatch, ort)

    readiness = build_camera_tracker_readiness(_settings(tmp_path, manifest=_manifest(tmp_path)))

    assert readiness.enabled is True
    assert readiness.status == "available"
    assert readiness.runtime["status"] == "verified"
    assert readiness.runtime["provider_selection"] == "automatic"
    assert readiness.runtime["selected_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert readiness.models["status"] == "verified"
    assert readiness.models["profile"] == "camera-tracker-v1"
    assert str(tmp_path) not in json.dumps(asdict(readiness))
    assert ort.calls == [("providers", None)]


def test_unconfigured_manifest_disables_native_label_without_probing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracker_runtime.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("runtime must not be probed")),
    )
    readiness = build_camera_tracker_readiness(_settings(tmp_path))
    assert readiness.enabled is False
    assert readiness.status == "disabled"
    assert readiness.runtime["status"] == "disabled"


def test_explicit_cuda_preflight_never_falls_back_to_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = _FakeOrt(("CPUExecutionProvider",))
    _runtime_modules(monkeypatch, ort)
    readiness = build_camera_tracker_readiness(
        _settings(
            tmp_path,
            manifest=_manifest(tmp_path),
            providers=("CUDAExecutionProvider",),
        )
    )

    assert readiness.enabled is False
    assert readiness.status == "misconfigured"
    assert readiness.runtime["available_providers"] == ["CPUExecutionProvider"]
    assert readiness.runtime["selected_providers"] == []
    assert "CUDAExecutionProvider is unavailable" in readiness.reason
    assert ort.calls == [("preload", ""), ("providers", None)]


def test_cuda_preload_failure_is_safe_and_stops_provider_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = _FakeOrt(
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        preload_failure=RuntimeError("sensitive system loader detail"),
    )
    _runtime_modules(monkeypatch, ort)
    readiness = build_camera_tracker_readiness(
        _settings(
            tmp_path,
            manifest=_manifest(tmp_path),
            providers=("CUDAExecutionProvider",),
        )
    )

    assert readiness.enabled is False
    assert readiness.runtime["available_providers"] == []
    assert readiness.reason == "ONNX Runtime CUDA dependency preload failed (RuntimeError)"
    assert "sensitive" not in json.dumps(asdict(readiness))
    assert ort.calls == [("preload", "")]


def test_provider_dependency_preload_is_a_reusable_strict_cuda_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = _FakeOrt(("CUDAExecutionProvider", "CPUExecutionProvider"))
    _runtime_modules(monkeypatch, ort)

    preload_tracker_provider_dependencies(None)
    preload_tracker_provider_dependencies(("CPUExecutionProvider",))
    assert ort.calls == []

    preload_tracker_provider_dependencies(("CUDAExecutionProvider",))
    assert ort.calls == [("preload", "")]

    ort.preload_failure = OSError("private loader path")
    with pytest.raises(
        TrackerRuntimePreparationError,
        match=r"CUDA dependency preload failed \(OSError\)",
    ) as raised:
        preload_tracker_provider_dependencies(("CUDAExecutionProvider",))
    assert "private loader path" not in str(raised.value)


class _ProbeSession:
    def __init__(self, active_providers: tuple[str, ...]) -> None:
        self.active_providers = active_providers
        self.run_calls: list[tuple[object, dict[str, object]]] = []

    def get_providers(self) -> list[str]:
        return list(self.active_providers)

    @staticmethod
    def get_inputs() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                name="images",
                type="tensor(float)",
                shape=[1, 3, 224, 224],
            )
        ]

    def run(self, output_names: object, inputs: dict[str, object]) -> list[object]:
        self.run_calls.append((output_names, inputs))
        return [object()]


class _ProbeOrt(_FakeOrt):
    def __init__(self, active_providers: tuple[str, ...]) -> None:
        super().__init__(("CUDAExecutionProvider", "CPUExecutionProvider"))
        self.session = _ProbeSession(active_providers)
        self.session_requests: list[tuple[str, list[str]]] = []

    def InferenceSession(self, path: str, *, providers: list[str]) -> _ProbeSession:
        self.session_requests.append((path, providers))
        return self.session


class _ProbeNumpy:
    __version__ = "1.26.4"
    float32 = object()

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: object) -> tuple[tuple[int, ...], object]:
        return shape, dtype


def test_cuda_model_probe_runs_verified_alignment_model_with_active_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _ProbeOrt(("CUDAExecutionProvider", "CPUExecutionProvider"))
    modules = {"numpy": _ProbeNumpy(), "onnxruntime": runtime}
    preload_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        tracker_runtime,
        "_import_optional_module",
        lambda name: (modules[name], None),
    )
    monkeypatch.setattr(
        tracker_runtime,
        "preload_tracker_provider_dependencies",
        lambda providers: preload_calls.append(providers),
    )

    result = probe_camera_model_cuda_session(_manifest(tmp_path))

    assert result == {
        "status": "verified",
        "ready": True,
        "reason": None,
        "model_role": "alignment-classifier",
        "available_providers": [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        "active_providers": [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    }
    assert preload_calls == [("CUDAExecutionProvider",)]
    assert runtime.session_requests[0][1] == ["CUDAExecutionProvider"]
    assert len(runtime.session.run_calls) == 1


def test_cuda_model_probe_rejects_session_level_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _ProbeOrt(("CPUExecutionProvider",))
    modules = {"numpy": _ProbeNumpy(), "onnxruntime": runtime}
    monkeypatch.setattr(
        tracker_runtime,
        "_import_optional_module",
        lambda name: (modules[name], None),
    )
    monkeypatch.setattr(
        tracker_runtime,
        "preload_tracker_provider_dependencies",
        lambda _providers: None,
    )

    result = probe_camera_model_cuda_session(_manifest(tmp_path))

    assert result["ready"] is False
    assert result["active_providers"] == ["CPUExecutionProvider"]
    assert result["reason"] == "alignment model session did not activate CUDAExecutionProvider"
    assert runtime.session.run_calls == []


def test_provider_query_and_missing_component_fail_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = _FakeOrt(
        ("CPUExecutionProvider",),
        provider_failure=RuntimeError("/private/runtime/provider/detail"),
    )
    _runtime_modules(monkeypatch, ort)
    failed_query = build_camera_tracker_readiness(_settings(tmp_path, manifest=_manifest(tmp_path)))
    assert failed_query.reason == "ONNX Runtime provider query failed (RuntimeError)"
    assert "/private/runtime/provider/detail" not in json.dumps(asdict(failed_query))

    _runtime_modules(monkeypatch, _FakeOrt(("CPUExecutionProvider",)), missing="cv2")
    missing_component = build_camera_tracker_readiness(
        _settings(tmp_path, manifest=_manifest(tmp_path))
    )
    assert missing_component.enabled is False
    assert missing_component.reason == "missing native tracker runtime: OpenCV"


def test_provider_environment_helper_maps_blank_to_automatic() -> None:
    assert tracker_onnx_providers_from_env({}) is None
    assert tracker_onnx_providers_from_env(
        {"CUBED_CORE_TRACKER_ONNX_PROVIDERS": "CPUExecutionProvider"}
    ) == ("CPUExecutionProvider",)
    with pytest.raises(ValueError, match="duplicate provider"):
        tracker_onnx_providers_from_env(
            {"CUBED_CORE_TRACKER_ONNX_PROVIDERS": ("CPUExecutionProvider,CPUExecutionProvider")}
        )
