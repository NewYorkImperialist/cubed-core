from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cubed_core.model_artifacts import (
    MODEL_ARTIFACT_MANIFEST_SCHEMA,
    ModelArtifactError,
    load_and_verify_model_manifest,
    tracker_model_capability,
)


def _artifact(
    path: Path,
    *,
    artifact_id: str = "alignment-v1",
    role: str = "alignment-classifier",
) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "id": artifact_id,
        "role": role,
        "format": "onnx",
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "redistribution": "external-only",
        "license": "NOASSERTION",
        "source": "operator-supplied test fixture",
        "model_card": None,
    }


def _write_manifest(path: Path, artifacts: list[dict[str, object]]) -> Path:
    manifest = {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "schema_version": 1,
        "profile": "camera-tracker-v1",
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_model_manifest_verifies_exact_artifacts_and_required_roles(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.onnx"
    alignment.write_bytes(b"alignment")
    pose = tmp_path / "pose.onnx"
    pose.write_bytes(b"pose")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(alignment),
            _artifact(pose, artifact_id="pose-v1", role="face-pose"),
        ],
    )

    manifest = load_and_verify_model_manifest(
        manifest_path,
        required_roles=("alignment-classifier", "face-pose"),
    )

    assert manifest.profile == "camera-tracker-v1"
    assert manifest.by_role()["face-pose"].path == pose
    assert (
        manifest.by_role()["alignment-classifier"].sha256
        == hashlib.sha256(b"alignment").hexdigest()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bytes", 99, "expected 99 but found"),
        ("sha256", "0" * 64, "expected"),
    ],
)
def test_model_manifest_rejects_integrity_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    item = _artifact(model)
    item[field] = value
    manifest_path = _write_manifest(tmp_path / "manifest.json", [item])

    with pytest.raises(ModelArtifactError, match=message):
        load_and_verify_model_manifest(manifest_path)


def test_model_manifest_rejects_path_escape_and_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.onnx"
    outside.write_bytes(b"outside")
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(outside)])
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["artifacts"][0]["path"] = f"../{outside.name}"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="normalized relative POSIX path"):
        load_and_verify_model_manifest(manifest_path)

    model = tmp_path / "real.onnx"
    model.write_bytes(b"model")
    link = tmp_path / "linked.onnx"
    link.symlink_to(model)
    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(link)])
    with pytest.raises(ModelArtifactError, match="may not traverse a symlink"):
        load_and_verify_model_manifest(manifest_path)


def test_model_manifest_rejects_duplicate_or_missing_roles(tmp_path: Path) -> None:
    first = tmp_path / "first.onnx"
    first.write_bytes(b"first")
    second = tmp_path / "second.onnx"
    second.write_bytes(b"second")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(first),
            _artifact(second, artifact_id="second-v1"),
        ],
    )
    with pytest.raises(ModelArtifactError, match="role duplicates"):
        load_and_verify_model_manifest(manifest_path)

    manifest_path = _write_manifest(tmp_path / "manifest.json", [_artifact(first)])
    with pytest.raises(ModelArtifactError, match="missing required role face-pose"):
        load_and_verify_model_manifest(manifest_path, required_roles=("face-pose",))


def test_model_manifest_schema_is_packaged_and_matches_runtime_contract() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "model-artifact-manifest-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == MODEL_ARTIFACT_MANIFEST_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == 1


def test_tracker_model_capability_is_fail_closed_and_redacts_paths(tmp_path: Path) -> None:
    assert tracker_model_capability(None) == {
        "status": "not-configured",
        "ready": False,
        "reason": "CUBED_CORE_TRACKER_MODEL_MANIFEST is not configured",
        "required_roles": ["alignment-classifier", "face-pose"],
        "profile": None,
        "artifacts": [],
    }

    alignment = tmp_path / "alignment.onnx"
    alignment.write_bytes(b"alignment")
    pose = tmp_path / "pose.onnx"
    pose.write_bytes(b"pose")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(alignment),
            _artifact(pose, artifact_id="pose-v1", role="face-pose"),
        ],
    )
    capability = tracker_model_capability(manifest_path)

    assert capability["status"] == "verified"
    assert capability["ready"] is True
    assert capability["profile"] == "camera-tracker-v1"
    assert {item["role"] for item in capability["artifacts"]} == {
        "alignment-classifier",
        "face-pose",
    }
    assert str(tmp_path) not in json.dumps(capability)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile", "synthetic-camera-tracker", "profile synthetic-camera-tracker is unsupported"),
        (
            "format",
            "numpy-npz",
            "role alignment-classifier must use ONNX format",
        ),
    ],
)
def test_native_tracker_capability_rejects_unsupported_profile_or_format(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    alignment = tmp_path / "alignment.onnx"
    alignment.write_bytes(b"alignment")
    pose = tmp_path / "pose.onnx"
    pose.write_bytes(b"pose")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _artifact(alignment),
            _artifact(pose, artifact_id="pose-v1", role="face-pose"),
        ],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "profile":
        manifest["profile"] = value
    else:
        manifest["artifacts"][0]["format"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    capability = tracker_model_capability(manifest_path)

    assert capability["status"] == "invalid"
    assert capability["ready"] is False
    assert message in capability["reason"]
    assert str(tmp_path) not in json.dumps(capability)
