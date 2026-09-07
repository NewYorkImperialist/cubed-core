from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ARTIFACT_MANIFEST_SCHEMA = "cubed-core/model-artifact-manifest-v1"
MODEL_ARTIFACT_MANIFEST_MAX_BYTES = 1024 * 1024
MODEL_ARTIFACT_MAX_BYTES = 2 * 1024**3
_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,99}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_FORMATS = {"onnx", "numpy-npz", "generated-cache"}
_REDISTRIBUTION = {"approved", "external-only", "train-your-own", "withheld"}
TRACKER_REQUIRED_MODEL_ROLES = ("alignment-classifier", "face-pose")
TRACKER_SUPPORTED_MODEL_PROFILES = ("camera-tracker-v1",)


class ModelArtifactError(ValueError):
    """A fail-closed model artifact manifest validation failure."""


@dataclass(frozen=True)
class VerifiedModelArtifact:
    artifact_id: str
    role: str
    format: str
    path: Path
    bytes: int
    sha256: str
    redistribution: str
    license: str
    source: str
    model_card: str | None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.artifact_id,
            "role": self.role,
            "format": self.format,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "redistribution": self.redistribution,
            "license": self.license,
            "source": self.source,
            "model_card": self.model_card,
        }


@dataclass(frozen=True)
class VerifiedModelManifest:
    profile: str
    manifest_path: Path
    artifacts: tuple[VerifiedModelArtifact, ...]

    def by_role(self) -> dict[str, VerifiedModelArtifact]:
        return {artifact.role: artifact for artifact in self.artifacts}

    def public(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
            "schema_version": 1,
            "profile": self.profile,
            "manifest_path": str(self.manifest_path),
            "artifacts": [artifact.public() for artifact in self.artifacts],
        }


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _object(
    value: Any,
    *,
    field: str,
    required: set[str],
    allowed: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelArtifactError(f"{field} must be an object")
    missing = required - value.keys()
    if missing:
        raise ModelArtifactError(f"{field} is missing {sorted(missing)[0]}")
    extra = value.keys() - allowed
    if extra:
        raise ModelArtifactError(f"{field} contains unsupported field {sorted(extra)[0]}")
    return value


def _string(value: Any, *, field: str, minimum: int = 1, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ModelArtifactError(
            f"{field} must be a string between {minimum} and {maximum} characters"
        )
    if "\0" in value:
        raise ModelArtifactError(f"{field} may not contain a NUL byte")
    return value


def _identifier(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=100)
    if not _IDENTIFIER.fullmatch(text):
        raise ModelArtifactError(f"{field} must be a lowercase kebab-case identifier")
    return text


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ModelArtifactError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def _secure_regular_file(path: Path, *, base: Path, field: str) -> Path:
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ModelArtifactError(f"{field} must remain inside the manifest directory") from exc
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ModelArtifactError(f"{field} may not traverse a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"{field} is unavailable") from exc
    if not resolved.is_file():
        raise ModelArtifactError(f"{field} must reference a regular file")
    return resolved


def _relative_artifact_path(value: Any, *, base: Path, field: str) -> Path:
    text = _string(value, field=field, maximum=8192)
    posix = PurePosixPath(text)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in text
    ):
        raise ModelArtifactError(
            f"{field} must be a normalized relative POSIX path without parent traversal"
        )
    return _secure_regular_file(base.joinpath(*posix.parts), base=base, field=field)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ModelArtifactError("model artifact manifest may not be a symlink")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelArtifactError("model artifact manifest is unavailable") from exc
    if size <= 0:
        raise ModelArtifactError("model artifact manifest is empty")
    if size > MODEL_ARTIFACT_MANIFEST_MAX_BYTES:
        raise ModelArtifactError(
            f"model artifact manifest exceeds the {MODEL_ARTIFACT_MANIFEST_MAX_BYTES}-byte limit"
        )
    try:
        value = json.loads(path.read_bytes(), parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelArtifactError("model artifact manifest must be valid finite JSON") from exc
    return _object(
        value,
        field="model artifact manifest",
        required={"schema", "schema_version", "profile", "artifacts"},
        allowed={"schema", "schema_version", "profile", "artifacts"},
    )


def _parse_manifest_artifact(
    item: dict[str, Any],
    *,
    artifact_id: str,
    role: str,
    base: Path,
    field: str,
) -> VerifiedModelArtifact:
    format_name = _string(item["format"], field=f"{field}.format", maximum=100)
    if format_name not in _FORMATS:
        raise ModelArtifactError(f"{field}.format is unsupported")
    redistribution = _string(
        item["redistribution"],
        field=f"{field}.redistribution",
        maximum=100,
    )
    if redistribution not in _REDISTRIBUTION:
        raise ModelArtifactError(f"{field}.redistribution is unsupported")
    expected_bytes = _integer(
        item["bytes"],
        field=f"{field}.bytes",
        minimum=1,
        maximum=MODEL_ARTIFACT_MAX_BYTES,
    )
    expected_sha256 = _string(item["sha256"], field=f"{field}.sha256", maximum=64)
    if not _SHA256.fullmatch(expected_sha256):
        raise ModelArtifactError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    artifact_path = _relative_artifact_path(
        item["path"],
        base=base,
        field=f"{field}.path",
    )
    actual_bytes = artifact_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ModelArtifactError(
            f"{field}.bytes expected {expected_bytes} but found {actual_bytes}"
        )
    actual_sha256 = _sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ModelArtifactError(
            f"{field}.sha256 expected {expected_sha256} but found {actual_sha256}"
        )
    model_card = item["model_card"]
    if model_card is not None:
        model_card = _string(model_card, field=f"{field}.model_card", maximum=8192)
    return VerifiedModelArtifact(
        artifact_id=artifact_id,
        role=role,
        format=format_name,
        path=artifact_path,
        bytes=actual_bytes,
        sha256=actual_sha256,
        redistribution=redistribution,
        license=_string(item["license"], field=f"{field}.license", maximum=500),
        source=_string(item["source"], field=f"{field}.source", maximum=2000),
        model_card=model_card,
    )


def load_and_verify_model_manifest(
    manifest_path: Path,
    *,
    required_roles: tuple[str, ...] = (),
) -> VerifiedModelManifest:
    """Load a manifest and verify every artifact before an inference session opens."""

    manifest_path = manifest_path.absolute()
    value = _load_json(manifest_path)
    if value["schema"] != MODEL_ARTIFACT_MANIFEST_SCHEMA or value["schema_version"] != 1:
        raise ModelArtifactError(
            "model artifact manifest must use "
            "cubed-core/model-artifact-manifest-v1 schema version 1"
        )
    profile = _identifier(value["profile"], field="model artifact manifest.profile")
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or not 1 <= len(raw_artifacts) <= 64:
        raise ModelArtifactError(
            "model artifact manifest.artifacts must contain between 1 and 64 items"
        )

    base = manifest_path.parent.resolve(strict=True)
    artifacts: list[VerifiedModelArtifact] = []
    seen_ids: set[str] = set()
    seen_roles: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        field = f"model artifact manifest.artifacts[{index}]"
        item = _object(
            raw,
            field=field,
            required={
                "id",
                "role",
                "format",
                "path",
                "bytes",
                "sha256",
                "redistribution",
                "license",
                "source",
                "model_card",
            },
            allowed={
                "id",
                "role",
                "format",
                "path",
                "bytes",
                "sha256",
                "redistribution",
                "license",
                "source",
                "model_card",
            },
        )
        artifact_id = _identifier(item["id"], field=f"{field}.id")
        role = _identifier(item["role"], field=f"{field}.role")
        if artifact_id in seen_ids:
            raise ModelArtifactError(f"{field}.id duplicates {artifact_id}")
        if role in seen_roles:
            raise ModelArtifactError(f"{field}.role duplicates {role}")
        seen_ids.add(artifact_id)
        seen_roles.add(role)
        artifacts.append(
            _parse_manifest_artifact(
                item, artifact_id=artifact_id, role=role, base=base, field=field
            )
        )

    missing_roles = sorted(set(required_roles) - seen_roles)
    if missing_roles:
        raise ModelArtifactError(
            "model artifact manifest is missing required role " + missing_roles[0]
        )
    return VerifiedModelManifest(
        profile=profile,
        manifest_path=manifest_path.resolve(strict=True),
        artifacts=tuple(artifacts),
    )


def load_and_verify_tracker_model_manifest(
    manifest_path: Path,
) -> VerifiedModelManifest:
    """Verify the stricter semantic contract supported by the native tracker."""

    manifest = load_and_verify_model_manifest(
        manifest_path,
        required_roles=TRACKER_REQUIRED_MODEL_ROLES,
    )
    if manifest.profile not in TRACKER_SUPPORTED_MODEL_PROFILES:
        supported = ", ".join(TRACKER_SUPPORTED_MODEL_PROFILES)
        raise ModelArtifactError(
            f"native tracker model profile {manifest.profile} is unsupported; expected {supported}"
        )
    by_role = manifest.by_role()
    for role in TRACKER_REQUIRED_MODEL_ROLES:
        if by_role[role].format != "onnx":
            raise ModelArtifactError(f"native tracker model role {role} must use ONNX format")
    return manifest


def tracker_model_capability(manifest_path: Path | None) -> dict[str, Any]:
    """Report the native tracker's model readiness without raising on invalid input."""

    if manifest_path is None:
        return {
            "status": "not-configured",
            "ready": False,
            "reason": "CUBED_CORE_TRACKER_MODEL_MANIFEST is not configured",
            "required_roles": list(TRACKER_REQUIRED_MODEL_ROLES),
            "profile": None,
            "artifacts": [],
        }
    try:
        manifest = load_and_verify_tracker_model_manifest(manifest_path)
    except ModelArtifactError as exc:
        return {
            "status": "invalid",
            "ready": False,
            "reason": str(exc),
            "required_roles": list(TRACKER_REQUIRED_MODEL_ROLES),
            "profile": None,
            "artifacts": [],
        }
    return {
        "status": "verified",
        "ready": True,
        "reason": None,
        "required_roles": list(TRACKER_REQUIRED_MODEL_ROLES),
        "profile": manifest.profile,
        "artifacts": [
            {
                "id": artifact.artifact_id,
                "role": artifact.role,
                "format": artifact.format,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
                "redistribution": artifact.redistribution,
                "license": artifact.license,
            }
            for artifact in manifest.artifacts
        ],
    }
