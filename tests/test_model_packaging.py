from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cubed_core.model_artifacts import load_and_verify_tracker_model_manifest
from cubed_core.model_packaging import (
    TRACKER_MODEL_PACKAGE_AUDIT_SCHEMA,
    TRACKER_MODEL_PACKAGE_SPEC_SCHEMA,
    ModelPackageError,
    inspect_tracker_onnx,
    load_tracker_package_spec,
    package_tracker_models,
)


def _model_card(*, extra: str = "") -> str:
    return f"""# Camera tracker model

## Release status

Reviewed release candidate with explicit artifact terms.

## Summary

Camera-only alignment and pose evidence models.

## Intended use

Local camera-evidence research and warm-start training.

## Artifact identity

Identity is bound by the adjacent manifest.

## Provenance and reproducibility

Training receipt is linked by each artifact source.

## Training data

Consent and grouping were reviewed separately.

## Limitations and failure modes

Setup-specific and unmeasured on unrelated rigs.

## Privacy, rights, and safety review

The release owner recorded the applicable review.

{extra}
"""


def _spec(
    root: Path,
    *,
    purpose: str = "publication-candidate",
    redistribution: str = "approved",
    card_extra: str = "",
) -> Path:
    alignment = root / "private-alignment-name.onnx"
    alignment.write_bytes(b"synthetic alignment onnx")
    pose = root / "private-pose-name.onnx"
    pose.write_bytes(b"synthetic pose onnx")
    card = root / "private-model-card.md"
    card.write_text(_model_card(extra=card_extra), encoding="utf-8")
    value = {
        "schema": TRACKER_MODEL_PACKAGE_SPEC_SCHEMA,
        "schema_version": 1,
        "purpose": purpose,
        "profile": "camera-tracker-v1",
        "onnx_input_trust": "operator-reviewed",
        "model_card": str(card),
        "artifacts": [
            {
                "id": "alignment-warm-start-v1",
                "role": "alignment-classifier",
                "path": str(alignment),
                "license": "AGPL-3.0-only",
                "source": "https://example.org/releases/alignment-warm-start-v1",
                "redistribution": redistribution,
            },
            {
                "id": "pose-warm-start-v1",
                "role": "face-pose",
                "path": str(pose),
                "license": "AGPL-3.0-only",
                "source": "urn:cubed-core:model-training-receipt:pose-v1",
                "redistribution": redistribution,
            },
        ],
    }
    path = root / "private-package-spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fake_inspector(path: Path, role: str) -> dict[str, object]:
    assert path.name == f"{role}.onnx"
    image_size = 224 if role == "alignment-classifier" else 1024
    output_shape = [1, 2] if role == "alignment-classifier" else [1, 17, 42]
    return {
        "input": {"type": "tensor(float)", "shape": [1, 3, image_size, image_size]},
        "first_output": {"type": "tensor(float)", "shape": output_shape},
        "output_count": 1,
        "provider": "CPUExecutionProvider",
    }


def _package_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_package_is_deterministic_runtime_valid_and_omits_paths_from_generated_text(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private-inputs"
    private.mkdir()
    spec = _spec(private)
    first = tmp_path / "first-package"
    second = tmp_path / "second-package"

    first_report = package_tracker_models(spec, first, inspector=_fake_inspector)
    second_report = package_tracker_models(spec, second, inspector=_fake_inspector)

    assert _package_files(first) == _package_files(second)
    assert first_report == second_report
    assert first_report["manifest"] == "manifest.json"
    assert str(tmp_path) not in json.dumps(first_report)
    manifest = load_and_verify_tracker_model_manifest(first / "manifest.json")
    assert tuple(manifest.by_role()) == ("alignment-classifier", "face-pose")
    assert {artifact.model_card for artifact in manifest.artifacts} == {"MODEL_CARD.md"}
    assert {path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()} == {
        "MODEL_CARD.md",
        "SHA256SUMS",
        "artifacts/alignment-classifier.onnx",
        "artifacts/face-pose.onnx",
        "audit.json",
        "manifest.json",
    }
    generated_text = "\n".join(
        (first / name).read_text(encoding="utf-8")
        for name in ("MODEL_CARD.md", "SHA256SUMS", "audit.json", "manifest.json")
    )
    assert str(private) not in generated_text
    assert "private-alignment-name" not in generated_text
    audit = json.loads((first / "audit.json").read_text(encoding="utf-8"))
    assert audit["schema"] == TRACKER_MODEL_PACKAGE_AUDIT_SCHEMA
    assert audit["legal_clearance"] == "not-evaluated"
    assert audit["quality_claim"] == "unmeasured"
    checksum_rows = (first / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksum_rows == sorted(checksum_rows, key=lambda row: row.split("  ", 1)[1])
    for row in checksum_rows:
        digest, relative = row.split("  ", 1)
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == digest


def test_publication_candidate_requires_explicit_approved_redistribution(
    tmp_path: Path,
) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(private, redistribution="external-only")

    with pytest.raises(ModelPackageError, match="requires redistribution=approved"):
        load_tracker_package_spec(spec)

    value = json.loads(spec.read_text(encoding="utf-8"))
    value["purpose"] = "local-review"
    spec.write_text(json.dumps(value), encoding="utf-8")
    package = tmp_path / "review-package"
    package_tracker_models(spec, package, inspector=_fake_inspector)
    audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
    assert audit["purpose"] == "local-review"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert {item["redistribution"] for item in manifest["artifacts"]} == {"external-only"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["artifacts"][0].pop("license"), "missing license"),
        (
            lambda value: value["artifacts"][0].update({"license": "NOASSERTION"}),
            "explicit reviewed license",
        ),
        (
            lambda value: value["artifacts"][0].update(
                {"source": "/Users/example/private/model.onnx"}
            ),
            "public locator",
        ),
        (
            lambda value: value["artifacts"][0].update(
                {"source": "https://user:secret@example.org/model"}
            ),
            "credential-free HTTPS",
        ),
    ],
)
def test_spec_rejects_missing_placeholder_or_private_metadata(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(private)
    value = json.loads(spec.read_text(encoding="utf-8"))
    mutation(value)
    spec.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ModelPackageError, match=message):
        load_tracker_package_spec(spec)


@pytest.mark.parametrize(
    "private_path",
    [
        "/home/example/cubed/weights/model.onnx",
        "/Volumes/private/models/model.onnx",
        "/mnt/private/models/model.onnx",
        "/workspace/private/models/model.onnx",
        r"D:\models\private\model.onnx",
        r"\\server\share\private\model.onnx",
    ],
)
def test_model_card_rejects_private_local_paths(
    tmp_path: Path,
    private_path: str,
) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(private, card_extra=f"Private receipt: {private_path}")
    with pytest.raises(ModelPackageError, match="private local path"):
        load_tracker_package_spec(spec)


def test_model_card_must_be_complete_without_placeholders(tmp_path: Path) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(
        private,
        card_extra="Manifest receipt: /absolute/path/to/manifest.json",
    )
    with pytest.raises(ModelPackageError, match="placeholder metadata"):
        load_tracker_package_spec(spec)

    card = private / "private-model-card.md"
    card.write_text("# Incomplete\n\nTODO\n", encoding="utf-8")
    with pytest.raises(ModelPackageError, match="placeholder metadata"):
        load_tracker_package_spec(spec)


def test_package_rejects_git_root_output_symlinks_and_existing_target(
    tmp_path: Path,
) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(private)
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "models").mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ModelPackageError, match="outside"):
        package_tracker_models(
            spec,
            repository / "models" / "release",
            inspector=_fake_inspector,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ModelPackageError, match="new path"):
        package_tracker_models(spec, existing, inspector=_fake_inspector)

    alignment = private / "private-alignment-name.onnx"
    real = private / "alignment-real.onnx"
    alignment.replace(real)
    alignment.symlink_to(real)
    with pytest.raises(ModelPackageError, match="may not be a symlink"):
        load_tracker_package_spec(spec)


def test_failed_onnx_audit_leaves_no_output_or_staging_directory(tmp_path: Path) -> None:
    private = tmp_path / "inputs"
    private.mkdir()
    spec = _spec(private)
    output = tmp_path / "failed-package"

    def fail_inspection(path: Path, role: str) -> dict[str, object]:
        raise ModelPackageError(f"{role} interface mismatch")

    with pytest.raises(ModelPackageError, match="interface mismatch"):
        package_tracker_models(spec, output, inspector=fail_inspection)

    assert not output.exists()
    assert not list(tmp_path.glob(".cubed-core-model-package-*"))


class _FakeMeta:
    def __init__(self, tensor_type: str, shape: list[object]) -> None:
        self.type = tensor_type
        self.shape = shape


def _runtime_module(input_shape: list[object], output_shape: list[object]) -> object:
    class SessionOptions:
        graph_optimization_level: object | None = None

    class InferenceSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_inputs(self) -> list[_FakeMeta]:
            return [_FakeMeta("tensor(float)", input_shape)]

        def get_outputs(self) -> list[_FakeMeta]:
            return [_FakeMeta("tensor(float)", output_shape)]

    return SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=SessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL="disabled"),
        InferenceSession=InferenceSession,
    )


def test_onnx_interface_audit_accepts_runtime_shapes_and_scrubs_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pose.onnx"
    path.write_bytes(b"synthetic")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _runtime_module(["private_batch_name", 3, 1024, 1024], [1, 17, "detections"]),
    )

    audit = inspect_tracker_onnx(path, "face-pose")

    assert audit["input"]["shape"] == ["dynamic", 3, 1024, 1024]
    assert audit["first_output"]["shape"] == [1, 17, "dynamic"]
    assert "private_batch_name" not in json.dumps(audit)


def test_onnx_interface_audit_rejects_incompatible_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "alignment.onnx"
    path.write_bytes(b"synthetic")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _runtime_module([1, 3, 640, 640], [1, 2]),
    )

    with pytest.raises(ModelPackageError, match="must accept 224"):
        inspect_tracker_onnx(path, "alignment-classifier")


def test_package_spec_schema_matches_runtime_constants() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "tracker-model-package-spec-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == TRACKER_MODEL_PACKAGE_SPEC_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["properties"]["profile"]["const"] == "camera-tracker-v1"
    assert schema["properties"]["artifacts"]["minItems"] == 2
    assert schema["properties"]["artifacts"]["maxItems"] == 2
