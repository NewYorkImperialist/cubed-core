from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    runtime_stub = ModuleType("onnxruntime")
    sys.modules["onnxruntime"] = runtime_stub
    try:
        from detect import onnx_runtime
    finally:
        del sys.modules["onnxruntime"]
else:
    from detect import onnx_runtime


class _Session:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=[1, 3, 224, 224])]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="scores")]


class _Runtime:
    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def __init__(self, *, active_providers: tuple[str, ...]) -> None:
        self.active_providers = active_providers
        self.session_providers: list[object] | None = None

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def InferenceSession(
        self,
        _path: str,
        *,
        sess_options: object,
        providers: list[object],
    ) -> _Session:
        assert sess_options is not None
        self.session_providers = providers
        return _Session(self.active_providers)


def test_canonical_decode_preloads_and_keeps_cuda_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime(active_providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    preload_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(onnx_runtime, "ort", runtime)
    monkeypatch.setattr(
        onnx_runtime,
        "preload_tracker_provider_dependencies",
        lambda providers: preload_calls.append(providers),
    )
    monkeypatch.setenv("CUBED_ORT_REQUIRE_CUDA", "1")
    monkeypatch.setenv("CUBED_ORT_PROVIDERS", "cuda")

    model = onnx_runtime.OnnxModel("/models/alignment.onnx")

    assert preload_calls == [("CUDAExecutionProvider",)]
    assert runtime.session_providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert model.session.get_providers()[0] == "CUDAExecutionProvider"


def test_canonical_decode_rejects_onnx_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime(active_providers=("CPUExecutionProvider",))
    monkeypatch.setattr(onnx_runtime, "ort", runtime)
    monkeypatch.setattr(
        onnx_runtime,
        "preload_tracker_provider_dependencies",
        lambda _providers: None,
    )
    monkeypatch.setenv("CUBED_ORT_REQUIRE_CUDA", "1")
    monkeypatch.setenv("CUBED_ORT_PROVIDERS", "cuda")

    with pytest.raises(RuntimeError, match="did not activate CUDAExecutionProvider"):
        onnx_runtime.OnnxModel("/models/alignment.onnx")


def test_noncanonical_onnx_cpu_session_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime(active_providers=("CPUExecutionProvider",))
    monkeypatch.setattr(onnx_runtime, "ort", runtime)
    monkeypatch.delenv("CUBED_ORT_REQUIRE_CUDA", raising=False)
    monkeypatch.setenv("CUBED_ORT_PROVIDERS", "cpu")

    model = onnx_runtime.OnnxModel("/models/alignment.onnx")

    assert model.session.get_providers() == ["CPUExecutionProvider"]
