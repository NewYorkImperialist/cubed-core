from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .model_artifacts import TRACKER_REQUIRED_MODEL_ROLES, VerifiedModelManifest
from .vision._runtime import VisionRuntimeUnavailable
from .vision.onnx import (
    OnnxModelError,
    OnnxRuntimeUnavailable,
    OnnxVisionClient,
    PosePostprocessConfig,
)

MAX_VIDEO_FRAMES = 250_000


class CameraTrackerBackendError(ValueError):
    """A camera-model setup failure safe to expose through Label."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _backend_error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> CameraTrackerBackendError:
    return CameraTrackerBackendError(code, message, details=details)


class CameraTrackerBackend:
    """Registry-verified ONNX inference used by the Label workbench."""

    def __init__(self, *, inference: Any) -> None:
        self.inference = inference

    @classmethod
    def from_verified_manifest(
        cls,
        manifest: VerifiedModelManifest,
        *,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> CameraTrackerBackend:
        if not isinstance(manifest, VerifiedModelManifest):
            raise _backend_error(
                "backend_model_error",
                "the tracker model manifest is not verified",
            )
        roles = manifest.by_role()
        missing = sorted(set(TRACKER_REQUIRED_MODEL_ROLES) - roles.keys())
        if missing:
            raise _backend_error(
                "missing_artifact",
                "required tracker model artifacts are not configured",
                details={"required_roles": missing},
            )
        try:
            inference = OnnxVisionClient.from_verified_artifacts(
                alignment_artifact=roles["alignment-classifier"],
                face_pose_artifact=roles["face-pose"],
                providers=providers,
                intra_op_threads=intra_op_threads,
                alignment_class_index=0,
                alignment_image_size=224,
                pose_image_size=1024,
                pose_config=PosePostprocessConfig(
                    confidence_threshold=0.50,
                    keypoint_confidence_threshold=0.50,
                    minimum_visible_keypoints=1,
                    snap_shared_corners=True,
                ),
            )
        except (OnnxRuntimeUnavailable, VisionRuntimeUnavailable) as exc:
            raise _backend_error(
                "backend_runtime_unavailable",
                "the local camera tracker runtime is unavailable",
                details={"exception_type": type(exc).__name__},
            ) from exc
        except OnnxModelError as exc:
            raise _backend_error(
                "backend_model_error",
                "the verified camera tracker model could not be used",
                details={
                    "exception_type": type(exc).__name__,
                    "model_profile": manifest.profile,
                },
            ) from exc
        return cls(inference=inference)
