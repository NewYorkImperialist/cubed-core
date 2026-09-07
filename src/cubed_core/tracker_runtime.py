from __future__ import annotations

import importlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_artifacts import (
    ModelArtifactError,
    load_and_verify_tracker_model_manifest,
    tracker_model_capability,
)
from .vision.onnx import OnnxModelError, resolve_execution_providers

TRACKER_MODEL_MANIFEST_ENV = "CUBED_CORE_TRACKER_MODEL_MANIFEST"
TRACKER_ONNX_PROVIDERS_ENV = "CUBED_CORE_TRACKER_ONNX_PROVIDERS"
_ONNX_PROVIDER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*ExecutionProvider")


class TrackerRuntimePreparationError(RuntimeError):
    """A provider dependency failure whose message is safe for UI output."""


def parse_onnx_provider_list(value: str) -> tuple[str, ...]:
    """Normalize exact comma-separated ONNX Runtime provider names."""

    if not isinstance(value, str):
        raise ValueError(f"{TRACKER_ONNX_PROVIDERS_ENV} must be a string")
    if not value.strip():
        return ()
    providers: list[str] = []
    for raw_provider in value.split(","):
        provider = raw_provider.strip()
        if not provider or not _ONNX_PROVIDER_NAME.fullmatch(provider):
            raise ValueError(
                f"{TRACKER_ONNX_PROVIDERS_ENV} must contain comma-separated exact "
                "ONNX Runtime provider names such as CUDAExecutionProvider"
            )
        if provider in providers:
            raise ValueError(f"{TRACKER_ONNX_PROVIDERS_ENV} contains duplicate provider {provider}")
        providers.append(provider)
    return tuple(providers)


def tracker_onnx_providers_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...] | None:
    """Return explicit providers, or ``None`` for automatic selection."""

    source = os.environ if environ is None else environ
    providers = parse_onnx_provider_list(source.get(TRACKER_ONNX_PROVIDERS_ENV, ""))
    return providers or None


def preload_tracker_provider_dependencies(
    requested_providers: tuple[str, ...] | None,
) -> None:
    """Preload pip-managed CUDA/cuDNN libraries for a strict CUDA request."""

    if not requested_providers or "CUDAExecutionProvider" not in requested_providers:
        return
    try:
        onnxruntime = importlib.import_module("onnxruntime")
    except Exception as exc:
        raise TrackerRuntimePreparationError(
            f"ONNX Runtime is unavailable for CUDA dependency preload ({type(exc).__name__})"
        ) from exc
    preload_dlls = getattr(onnxruntime, "preload_dlls", None)
    if not callable(preload_dlls):
        return
    try:
        preload_dlls(directory="")
    except Exception as exc:
        raise TrackerRuntimePreparationError(
            f"ONNX Runtime CUDA dependency preload failed ({type(exc).__name__})"
        ) from exc


def _component(*, available: bool, module: Any | None, error: Exception | None) -> dict[str, Any]:
    version = getattr(module, "__version__", None) if module is not None else None
    return {
        "available": available,
        "version": str(version) if version is not None else None,
        "error_type": type(error).__name__ if error is not None else None,
    }


def _import_optional_module(name: str) -> tuple[Any | None, Exception | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, exc


def probe_camera_model_cuda_session(manifest_path: Path | None) -> dict[str, Any]:
    """Run the released alignment model once through an active CUDA session."""

    result: dict[str, Any] = {
        "status": "unavailable",
        "ready": False,
        "reason": None,
        "model_role": "alignment-classifier",
        "available_providers": [],
        "active_providers": [],
    }
    if manifest_path is None:
        result["reason"] = "camera model CUDA probe requires a configured model manifest"
        return result
    try:
        manifest = load_and_verify_tracker_model_manifest(manifest_path)
    except (ModelArtifactError, OSError, TypeError, ValueError) as exc:
        result["reason"] = f"camera model manifest verification failed ({type(exc).__name__})"
        return result

    numpy_module, numpy_error = _import_optional_module("numpy")
    onnx_module, onnx_error = _import_optional_module("onnxruntime")
    if numpy_module is None or onnx_module is None:
        missing = []
        if numpy_module is None:
            missing.append(f"NumPy ({type(numpy_error).__name__})")
        if onnx_module is None:
            missing.append(f"ONNX Runtime ({type(onnx_error).__name__})")
        result["reason"] = "camera model CUDA probe is missing " + ", ".join(missing)
        return result

    try:
        preload_tracker_provider_dependencies(("CUDAExecutionProvider",))
    except TrackerRuntimePreparationError as exc:
        result["reason"] = str(exc)
        return result
    try:
        available = tuple(str(provider) for provider in onnx_module.get_available_providers())
    except Exception as exc:
        result["reason"] = f"ONNX Runtime provider query failed ({type(exc).__name__})"
        return result
    result["available_providers"] = list(available)
    if "CUDAExecutionProvider" not in available:
        result["reason"] = "CUDAExecutionProvider is unavailable"
        return result

    artifact = manifest.by_role()["alignment-classifier"]
    try:
        session = onnx_module.InferenceSession(
            str(artifact.path),
            providers=["CUDAExecutionProvider"],
        )
        active = tuple(str(provider) for provider in session.get_providers())
        result["active_providers"] = list(active)
        if "CUDAExecutionProvider" not in active:
            result["reason"] = "alignment model session did not activate CUDAExecutionProvider"
            return result
        inputs = tuple(session.get_inputs())
        if len(inputs) != 1 or getattr(inputs[0], "type", None) != "tensor(float)":
            raise ValueError("alignment model input contract is unsupported")
        shape = tuple(inputs[0].shape)
        if not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
            raise ValueError("alignment model input shape must be concrete")
        tensor = numpy_module.zeros(shape, dtype=numpy_module.float32)
        outputs = session.run(None, {inputs[0].name: tensor})
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise ValueError("alignment model produced no outputs")
    except Exception as exc:
        result["reason"] = f"alignment model CUDA inference failed ({type(exc).__name__})"
        return result

    result.update({"status": "verified", "ready": True, "reason": None})
    return result


def _inactive_runtime(
    *,
    status: str,
    reason: str,
    requested_providers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "status": status,
        "ready": False,
        "reason": reason,
        "provider_selection": "explicit" if requested_providers else "automatic",
        "requested_providers": list(requested_providers),
        "available_providers": [],
        "selected_providers": [],
        "numpy": {"available": False, "version": None, "error_type": None},
        "opencv": {"available": False, "version": None, "error_type": None},
        "onnxruntime": {"available": False, "version": None, "error_type": None},
    }


def _runtime_capability(
    requested_providers: tuple[str, ...],
) -> dict[str, Any]:
    numpy_module, numpy_error = _import_optional_module("numpy")
    opencv_module, opencv_error = _import_optional_module("cv2")
    onnx_module, onnx_error = _import_optional_module("onnxruntime")
    available_providers: tuple[str, ...] = ()
    selected_providers: tuple[str, ...] = ()
    provider_error: str | None = None

    if onnx_module is not None:
        if "CUDAExecutionProvider" in requested_providers:
            try:
                preload_tracker_provider_dependencies(requested_providers)
            except TrackerRuntimePreparationError as exc:
                onnx_error = exc
                provider_error = str(exc)
        if onnx_error is None:
            try:
                raw_providers = onnx_module.get_available_providers()
                if isinstance(raw_providers, (str, bytes)):
                    raise TypeError("provider result must be a sequence")
                available_providers = tuple(str(provider) for provider in raw_providers)
                if len(set(available_providers)) != len(available_providers):
                    raise ValueError("provider result contains duplicates")
                selected_providers = resolve_execution_providers(
                    requested_providers or None,
                    available=available_providers,
                )
            except OnnxModelError as exc:
                provider_error = str(exc)
            except Exception as exc:
                provider_error = f"ONNX Runtime provider query failed ({type(exc).__name__})"

    missing_components = [
        name
        for name, module in (
            ("NumPy", numpy_module),
            ("OpenCV", opencv_module),
            ("ONNX Runtime", onnx_module),
        )
        if module is None
    ]
    if missing_components:
        reason = "missing native tracker runtime: " + ", ".join(missing_components)
    elif onnx_error is not None:
        reason = provider_error or (
            f"ONNX Runtime failed during native tracker preflight ({type(onnx_error).__name__})"
        )
    elif provider_error is not None:
        reason = provider_error
    else:
        reason = None
    ready = (
        numpy_module is not None
        and opencv_module is not None
        and onnx_module is not None
        and onnx_error is None
        and reason is None
        and bool(selected_providers)
    )
    return {
        "status": "verified" if ready else "unavailable",
        "ready": ready,
        "reason": None if ready else reason or "native tracker runtime is unavailable",
        "provider_selection": "explicit" if requested_providers else "automatic",
        "requested_providers": list(requested_providers),
        "available_providers": list(available_providers),
        "selected_providers": list(selected_providers),
        "numpy": _component(
            available=numpy_module is not None,
            module=numpy_module,
            error=numpy_error,
        ),
        "opencv": _component(
            available=opencv_module is not None,
            module=opencv_module,
            error=opencv_error,
        ),
        "onnxruntime": _component(
            available=onnx_module is not None and onnx_error is None,
            module=onnx_module,
            error=onnx_error,
        ),
    }


@dataclass(frozen=True)
class CameraTrackerReadiness:
    enabled: bool
    status: str
    reason: str | None
    runtime: dict[str, Any]
    models: dict[str, Any]


def build_camera_tracker_readiness(settings: Any) -> CameraTrackerReadiness:
    """Report the model/runtime readiness used by native Label inference."""

    requested_providers = settings.tracker_onnx_providers
    models = tracker_model_capability(settings.tracker_model_manifest)
    if settings.tracker_model_manifest is None:
        reason = str(models["reason"])
        return CameraTrackerReadiness(
            enabled=False,
            status="disabled",
            reason=reason,
            runtime=_inactive_runtime(
                status="disabled",
                reason=reason,
                requested_providers=requested_providers,
            ),
            models=models,
        )

    runtime = _runtime_capability(requested_providers)
    if not models["ready"]:
        reason = str(models["reason"])
    elif not runtime["ready"]:
        reason = str(runtime["reason"])
    else:
        reason = None
    return CameraTrackerReadiness(
        enabled=reason is None,
        status="available" if reason is None else "misconfigured",
        reason=reason,
        runtime=runtime,
        models=models,
    )
