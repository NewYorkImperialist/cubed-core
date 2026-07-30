from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cubed_core.camera_tracker_backend import (
    CameraTrackerBackend,
    CameraTrackerBackendError,
)
from cubed_core.model_artifacts import (
    VerifiedModelArtifact,
    VerifiedModelManifest,
)
from cubed_core.vision.onnx import OnnxModelError, OnnxRuntimeUnavailable, OnnxVisionClient


def _manifest(
    tmp_path: Path,
    *,
    include_pose: bool = True,
) -> VerifiedModelManifest:
    rows = [
        ("alignment-v1", "alignment-classifier", "a"),
        *([("pose-v1", "face-pose", "b")] if include_pose else []),
    ]
    artifacts = tuple(
        VerifiedModelArtifact(
            artifact_id=artifact_id,
            role=role,
            format="onnx",
            path=(tmp_path / f"{artifact_id}.onnx").resolve(),
            bytes=1,
            sha256=character * 64,
            redistribution="external-only",
            license="NOASSERTION",
            source="synthetic test artifact",
            model_card=None,
        )
        for artifact_id, role, character in rows
    )
    return VerifiedModelManifest(
        profile="synthetic-tracker",
        manifest_path=(tmp_path / "manifest.json").resolve(),
        artifacts=artifacts,
    )


def test_verified_manifest_factory_passes_exact_models_providers_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    inference = object()
    captured: dict[str, Any] = {}

    def fake_factory(**kwargs: Any) -> object:
        captured.update(kwargs)
        return inference

    monkeypatch.setattr(
        OnnxVisionClient,
        "from_verified_artifacts",
        staticmethod(fake_factory),
    )
    providers = ("CUDAExecutionProvider",)
    backend = CameraTrackerBackend.from_verified_manifest(
        manifest,
        providers=providers,
        intra_op_threads=4,
    )
    roles = manifest.by_role()

    assert backend.inference is inference
    assert captured["alignment_artifact"] is roles["alignment-classifier"]
    assert captured["face_pose_artifact"] is roles["face-pose"]
    assert captured["providers"] is providers
    assert captured["intra_op_threads"] == 4
    assert captured["alignment_class_index"] == 0
    assert captured["alignment_image_size"] == 224
    assert captured["pose_image_size"] == 1024
    pose_config = captured["pose_config"]
    assert pose_config.confidence_threshold == 0.50
    assert pose_config.keypoint_confidence_threshold == 0.50
    assert pose_config.minimum_visible_keypoints == 1
    assert pose_config.snap_shared_corners is True


def test_verified_manifest_factory_rejects_missing_model_roles(tmp_path: Path) -> None:
    with pytest.raises(CameraTrackerBackendError) as raised:
        CameraTrackerBackend.from_verified_manifest(
            _manifest(tmp_path, include_pose=False),
        )
    assert raised.value.code == "missing_artifact"
    assert raised.value.details == {"required_roles": ["face-pose"]}


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (OnnxRuntimeUnavailable("synthetic runtime"), "backend_runtime_unavailable"),
        (OnnxModelError("synthetic model"), "backend_model_error"),
    ],
)
def test_verified_manifest_factory_maps_runtime_and_model_failures(
    failure: Exception,
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_factory(**_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(
        OnnxVisionClient,
        "from_verified_artifacts",
        staticmethod(fail_factory),
    )
    with pytest.raises(CameraTrackerBackendError) as raised:
        CameraTrackerBackend.from_verified_manifest(_manifest(tmp_path))
    assert raised.value.code == expected_code
    assert raised.value.details["exception_type"] == type(failure).__name__
