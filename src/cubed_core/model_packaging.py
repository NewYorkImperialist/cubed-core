from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .model_artifacts import (
    MODEL_ARTIFACT_MANIFEST_SCHEMA,
    MODEL_ARTIFACT_MAX_BYTES,
    TRACKER_REQUIRED_MODEL_ROLES,
    load_and_verify_tracker_model_manifest,
)

TRACKER_MODEL_PACKAGE_SPEC_SCHEMA = "cubed-core/tracker-model-package-spec-v1"
TRACKER_MODEL_PACKAGE_AUDIT_SCHEMA = "cubed-core/tracker-model-package-audit-v1"
TRACKER_MODEL_PACKAGE_SPEC_MAX_BYTES = 1024 * 1024
TRACKER_MODEL_CARD_MAX_BYTES = 1024 * 1024
TRACKER_MODEL_PROFILE = "camera-tracker-v1"
TRACKER_MODEL_PACKAGE_PURPOSES = ("local-review", "publication-candidate")
TRACKER_MODEL_REDISTRIBUTION_VALUES = (
    "approved",
    "external-only",
    "train-your-own",
    "withheld",
)

_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,99}")
_LICENSE_EXPRESSION = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9.+-]{0,99})"
    r"(?: (?:AND|OR|WITH) [A-Za-z0-9][A-Za-z0-9.+-]{0,99})*"
)
_PLACEHOLDER = re.compile(
    r"(?i)(?:\b(?:noassertion|unknown|tbd|todo)\b|replace[- ]with|"
    r"`\[(?:name|immutable version|actual|public project contact|"
    r"exact SPDX identifier|review state|status|terms)[^]]*\]`|"
    r"/absolute/path(?:/|\b))"
)
_PRIVATE_PATH_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[\s\"'=(])/"
        r"(?:Users|home|Volumes|mnt|workspace|private|tmp)(?:/|\b)"
    ),
    re.compile(r"(?i)(?:^|[\s\"'=(])[A-Z]:\\"),
    re.compile(r"(?i)(?:^|[\s\"'=(])\\\\[^\\\s]+\\[^\\\s]+\\"),
    re.compile(r"(?i)\bfile://"),
)


class ModelPackageError(ValueError):
    """A fail-closed model package validation or construction failure."""


@dataclass(frozen=True)
class TrackerPackageArtifact:
    artifact_id: str
    role: str
    source_path: Path
    license: str
    source: str
    redistribution: str


@dataclass(frozen=True)
class TrackerPackageSpec:
    purpose: str
    profile: str
    model_card_path: Path
    artifacts: tuple[TrackerPackageArtifact, ...]


OnnxInspector = Callable[[Path, str], dict[str, Any]]


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _exact_object(
    value: Any,
    *,
    field: str,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelPackageError(f"{field} must be an object")
    missing = sorted(required - value.keys())
    if missing:
        raise ModelPackageError(f"{field} is missing {missing[0]}")
    extra = sorted(value.keys() - required)
    if extra:
        raise ModelPackageError(f"{field} contains unsupported field {extra[0]}")
    return value


def _string(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ModelPackageError(f"{field} must be a string between 1 and {maximum} characters")
    if "\0" in value:
        raise ModelPackageError(f"{field} may not contain a NUL byte")
    return value


def _identifier(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=100)
    if not _IDENTIFIER.fullmatch(text):
        raise ModelPackageError(f"{field} must be a lowercase kebab-case identifier")
    return text


def _read_json_object(path: Path, *, field: str, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise ModelPackageError(f"{field} may not be a symlink")
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelPackageError(f"{field} is unavailable") from exc
    if not path.is_file() or stat.st_size <= 0:
        raise ModelPackageError(f"{field} must be a nonempty regular file")
    if stat.st_size > maximum_bytes:
        raise ModelPackageError(f"{field} exceeds the {maximum_bytes}-byte limit")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelPackageError(f"{field} must be valid finite JSON with unique keys") from exc
    if not isinstance(value, dict):
        raise ModelPackageError(f"{field} must be an object")
    return value


def _regular_input_file(
    value: Any,
    *,
    field: str,
    base: Path,
    maximum_bytes: int,
    suffix: str | None = None,
) -> Path:
    text = _string(value, field=field, maximum=8192)
    candidate = Path(text)
    path = candidate if candidate.is_absolute() else base / candidate
    if path.is_symlink():
        raise ModelPackageError(f"{field} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise ModelPackageError(f"{field} is unavailable") from exc
    if not resolved.is_file() or stat.st_size <= 0:
        raise ModelPackageError(f"{field} must be a nonempty regular file")
    if stat.st_size > maximum_bytes:
        raise ModelPackageError(f"{field} exceeds the {maximum_bytes}-byte limit")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ModelPackageError(f"{field} must reference a {suffix} file")
    return resolved


def _contains_private_path(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRIVATE_PATH_PATTERNS)


def _explicit_license(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=500).strip()
    if _PLACEHOLDER.search(text):
        raise ModelPackageError(f"{field} must be an explicit reviewed license expression")
    if _LICENSE_EXPRESSION.fullmatch(text):
        return text
    parsed = urlsplit(text)
    if (
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    ):
        return text
    raise ModelPackageError(
        f"{field} must be an SPDX-style expression or credential-free HTTPS terms URL"
    )


def _public_source(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=2000).strip()
    if _PLACEHOLDER.search(text) or _contains_private_path(text):
        raise ModelPackageError(
            f"{field} must be an explicit public locator or path-free receipt URN"
        )
    parsed = urlsplit(text)
    if parsed.scheme == "urn":
        if len(parsed.path) < 3 or any(character.isspace() for character in text):
            raise ModelPackageError(f"{field} contains an invalid receipt URN")
        return text
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelPackageError(
            f"{field} must be a credential-free HTTPS URL without query/fragment "
            "or a path-free receipt URN"
        )
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ModelPackageError(f"{field} may not use a local hostname")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ModelPackageError(f"{field} may not use a private or local IP address")
    return text


def _load_model_card(path: Path) -> bytes:
    if path.is_symlink():
        raise ModelPackageError("model card may not be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ModelPackageError("model card is unavailable") from exc
    if not raw or len(raw) > TRACKER_MODEL_CARD_MAX_BYTES:
        raise ModelPackageError(
            f"model card must contain between 1 and {TRACKER_MODEL_CARD_MAX_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelPackageError("model card must be UTF-8 text") from exc
    if "\0" in text:
        raise ModelPackageError("model card may not contain a NUL byte")
    if _PLACEHOLDER.search(text):
        raise ModelPackageError("model card still contains placeholder metadata")
    if _contains_private_path(text):
        raise ModelPackageError("model card contains a private local path")
    required_sections = (
        "## Release status",
        "## Summary",
        "## Intended use",
        "## Artifact identity",
        "## Provenance and reproducibility",
        "## Training data",
        "## Limitations and failure modes",
        "## Privacy, rights, and safety review",
    )
    missing = next((section for section in required_sections if section not in text), None)
    if missing is not None:
        raise ModelPackageError(f"model card is missing required section {missing}")
    for section in required_sections:
        body_start = text.index(section) + len(section)
        next_section = text.find("\n## ", body_start)
        body = text[body_start : next_section if next_section >= 0 else None].strip()
        if len(body) < 20:
            raise ModelPackageError(
                f"model card section {section} must contain substantive review content"
            )
    return raw


def load_tracker_package_spec(spec_path: Path) -> TrackerPackageSpec:
    spec_path = spec_path.absolute()
    value = _exact_object(
        _read_json_object(
            spec_path,
            field="tracker model package spec",
            maximum_bytes=TRACKER_MODEL_PACKAGE_SPEC_MAX_BYTES,
        ),
        field="tracker model package spec",
        required={
            "schema",
            "schema_version",
            "purpose",
            "profile",
            "onnx_input_trust",
            "model_card",
            "artifacts",
        },
    )
    if (
        value["schema"] != TRACKER_MODEL_PACKAGE_SPEC_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ModelPackageError(
            "tracker model package spec must use "
            "cubed-core/tracker-model-package-spec-v1 schema version 1"
        )
    purpose = _string(value["purpose"], field="tracker model package spec.purpose", maximum=40)
    if purpose not in TRACKER_MODEL_PACKAGE_PURPOSES:
        raise ModelPackageError("tracker model package spec.purpose is unsupported")
    if value["onnx_input_trust"] != "operator-reviewed":
        raise ModelPackageError(
            "tracker model package spec.onnx_input_trust must be operator-reviewed; "
            "ONNX Runtime is not a sandbox"
        )
    profile = _identifier(value["profile"], field="tracker model package spec.profile")
    if profile != TRACKER_MODEL_PROFILE:
        raise ModelPackageError(
            f"tracker model package spec.profile must be {TRACKER_MODEL_PROFILE}"
        )
    base = spec_path.parent.resolve(strict=True)
    model_card_path = _regular_input_file(
        value["model_card"],
        field="tracker model package spec.model_card",
        base=base,
        maximum_bytes=TRACKER_MODEL_CARD_MAX_BYTES,
        suffix=".md",
    )
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(
        TRACKER_REQUIRED_MODEL_ROLES
    ):
        raise ModelPackageError(
            "tracker model package spec.artifacts must contain exactly "
            f"{len(TRACKER_REQUIRED_MODEL_ROLES)} items"
        )

    artifacts: list[TrackerPackageArtifact] = []
    seen_ids: set[str] = set()
    seen_roles: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        field = f"tracker model package spec.artifacts[{index}]"
        item = _exact_object(
            raw,
            field=field,
            required={"id", "role", "path", "license", "source", "redistribution"},
        )
        artifact_id = _identifier(item["id"], field=f"{field}.id")
        role = _identifier(item["role"], field=f"{field}.role")
        if artifact_id in seen_ids:
            raise ModelPackageError(f"{field}.id duplicates {artifact_id}")
        if role in seen_roles:
            raise ModelPackageError(f"{field}.role duplicates {role}")
        seen_ids.add(artifact_id)
        seen_roles.add(role)
        redistribution = _string(
            item["redistribution"],
            field=f"{field}.redistribution",
            maximum=40,
        )
        if redistribution not in TRACKER_MODEL_REDISTRIBUTION_VALUES:
            raise ModelPackageError(f"{field}.redistribution is unsupported")
        artifacts.append(
            TrackerPackageArtifact(
                artifact_id=artifact_id,
                role=role,
                source_path=_regular_input_file(
                    item["path"],
                    field=f"{field}.path",
                    base=base,
                    maximum_bytes=MODEL_ARTIFACT_MAX_BYTES,
                    suffix=".onnx",
                ),
                license=_explicit_license(item["license"], field=f"{field}.license"),
                source=_public_source(item["source"], field=f"{field}.source"),
                redistribution=redistribution,
            )
        )
    expected_roles = set(TRACKER_REQUIRED_MODEL_ROLES)
    if seen_roles != expected_roles:
        missing = sorted(expected_roles - seen_roles)
        extra = sorted(seen_roles - expected_roles)
        if missing:
            raise ModelPackageError(
                f"tracker model package spec is missing required role {missing[0]}"
            )
        raise ModelPackageError(f"tracker model package spec has unsupported role {extra[0]}")
    if purpose == "publication-candidate":
        unapproved = sorted(
            artifact.role for artifact in artifacts if artifact.redistribution != "approved"
        )
        if unapproved:
            raise ModelPackageError(
                "publication-candidate package requires redistribution=approved for role "
                + unapproved[0]
            )
    _load_model_card(model_card_path)
    ordered = tuple(
        next(artifact for artifact in artifacts if artifact.role == role)
        for role in TRACKER_REQUIRED_MODEL_ROLES
    )
    return TrackerPackageSpec(
        purpose=purpose,
        profile=profile,
        model_card_path=model_card_path,
        artifacts=ordered,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any, *, field: str) -> list[int | str | None]:
    if not isinstance(value, (list, tuple)):
        raise ModelPackageError(f"{field} must be a tensor shape")
    result: list[int | str | None] = []
    for dimension in value:
        if dimension is None:
            result.append(None)
        elif type(dimension) is int:
            if dimension == 0 or dimension < -1:
                raise ModelPackageError(f"{field} contains an invalid integer dimension")
            result.append(dimension)
        elif isinstance(dimension, str) and 1 <= len(dimension) <= 200:
            result.append("dynamic")
        else:
            raise ModelPackageError(f"{field} contains an unsupported dimension")
    return result


def _accepts_shape(
    declared: list[int | str | None],
    supplied: tuple[int, ...],
    *,
    field: str,
) -> None:
    if len(declared) != len(supplied):
        raise ModelPackageError(f"{field} must have rank {len(supplied)}")
    for index, (expected, actual) in enumerate(zip(declared, supplied, strict=True)):
        if isinstance(expected, int) and expected > 0 and expected != actual:
            raise ModelPackageError(
                f"{field} dimension {index} must accept {actual}, found {expected}"
            )


def inspect_tracker_onnx(path: Path, role: str) -> dict[str, Any]:
    """Open an ONNX artifact through the pinned runtime and audit its static interface."""

    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelPackageError(
            "ONNX audit requires the tracker-cpu or tracker-gpu optional dependency"
        ) from exc
    available = tuple(onnxruntime.get_available_providers())
    if "CPUExecutionProvider" not in available:
        raise ModelPackageError("ONNX audit requires CPUExecutionProvider")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = onnxruntime.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ModelPackageError(f"{role} artifact is not a loadable standalone ONNX model") from exc
    inputs = tuple(session.get_inputs())
    outputs = tuple(session.get_outputs())
    if len(inputs) != 1:
        raise ModelPackageError(f"{role} ONNX model must expose exactly one input")
    if not outputs:
        raise ModelPackageError(f"{role} ONNX model must expose at least one output")
    input_meta = inputs[0]
    if getattr(input_meta, "type", None) != "tensor(float)":
        raise ModelPackageError(f"{role} ONNX input must be tensor(float)")
    input_shape = _shape(getattr(input_meta, "shape", None), field=f"{role} ONNX input")
    supplied = (1, 3, 224, 224) if role == "alignment-classifier" else (1, 3, 1024, 1024)
    _accepts_shape(input_shape, supplied, field=f"{role} ONNX input")

    output_meta = outputs[0]
    output_type = getattr(output_meta, "type", None)
    if output_type != "tensor(float)":
        raise ModelPackageError(f"{role} first ONNX output must be tensor(float)")
    output_shape = _shape(
        getattr(output_meta, "shape", None),
        field=f"{role} first ONNX output",
    )
    if role == "alignment-classifier":
        if len(output_shape) not in {1, 2}:
            raise ModelPackageError(
                "alignment-classifier first ONNX output must have shape (C,) or (1, C)"
            )
        classes = output_shape[-1]
        if isinstance(classes, int) and classes > 0 and classes < 2:
            raise ModelPackageError(
                "alignment-classifier first ONNX output must expose at least two classes"
            )
        if len(output_shape) == 2:
            batch = output_shape[0]
            if isinstance(batch, int) and batch > 0 and batch != 1:
                raise ModelPackageError(
                    "alignment-classifier first ONNX output must accept batch one"
                )
    else:
        if len(output_shape) not in {2, 3}:
            raise ModelPackageError(
                "face-pose first ONNX output must have shape (17, N) or (1, 17, N)"
            )
        channels = output_shape[-2]
        if isinstance(channels, int) and channels > 0 and channels != 17:
            raise ModelPackageError("face-pose first ONNX output must expose exactly 17 channels")
        if len(output_shape) == 3:
            batch = output_shape[0]
            if isinstance(batch, int) and batch > 0 and batch != 1:
                raise ModelPackageError("face-pose first ONNX output must accept batch one")
    return {
        "input": {"type": "tensor(float)", "shape": input_shape},
        "first_output": {"type": output_type, "shape": output_shape},
        "output_count": len(outputs),
        "provider": "CPUExecutionProvider",
    }


def _assert_outside(path: Path, protected_roots: Iterable[Path]) -> None:
    candidate = path.absolute()
    for root in protected_roots:
        try:
            candidate.relative_to(root.resolve(strict=True))
        except ValueError:
            continue
        raise ModelPackageError(
            "model package output must be outside the Cubed Core repository and other "
            "protected Git roots"
        )


def _enclosing_git_root(path: Path) -> Path | None:
    current = path.resolve(strict=True)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.exists() or marker.is_symlink():
            return candidate
    return None


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o644)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _scan_package_text(path: Path) -> None:
    for name in ("MODEL_CARD.md", "manifest.json", "audit.json", "SHA256SUMS"):
        text = (path / name).read_text(encoding="utf-8")
        if _contains_private_path(text):
            raise ModelPackageError(f"generated {name} contains a private local path")


def package_tracker_models(
    spec_path: Path,
    output_dir: Path,
    *,
    protected_roots: Iterable[Path] = (),
    inspector: OnnxInspector = inspect_tracker_onnx,
) -> dict[str, Any]:
    """Build a deterministic package whose generated metadata omits source paths.

    ONNX artifacts are copied byte for byte; embedded model metadata is not
    rewritten or scrubbed.
    """

    spec = load_tracker_package_spec(spec_path)
    output_dir = output_dir.absolute()
    if output_dir.name in {"", ".", ".."}:
        raise ModelPackageError("model package output must name a new directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise ModelPackageError("model package output must be a new path")
    parent = output_dir.parent
    try:
        parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ModelPackageError("model package output parent is unavailable") from exc
    if not parent.is_dir():
        raise ModelPackageError("model package output parent must be a directory")
    output_dir = parent / output_dir.name
    _assert_outside(output_dir, protected_roots)
    if _enclosing_git_root(parent) is not None:
        raise ModelPackageError("model package output must be outside every enclosing Git worktree")
    stage = Path(tempfile.mkdtemp(prefix=".cubed-core-model-package-", dir=parent))
    try:
        card_bytes = _load_model_card(spec.model_card_path)
        _write_bytes(stage / "MODEL_CARD.md", card_bytes)
        manifest_artifacts: list[dict[str, Any]] = []
        audit_artifacts: list[dict[str, Any]] = []
        for artifact in spec.artifacts:
            source_sha = _sha256(artifact.source_path)
            destination_relative = Path("artifacts") / f"{artifact.role}.onnx"
            destination = stage / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact.source_path, destination)
            destination.chmod(0o644)
            destination_sha = _sha256(destination)
            if destination_sha != source_sha:
                raise ModelPackageError(
                    f"{artifact.role} source changed while the package was being built"
                )
            byte_count = destination.stat().st_size
            onnx_audit = inspector(destination, artifact.role)
            manifest_artifacts.append(
                {
                    "id": artifact.artifact_id,
                    "role": artifact.role,
                    "format": "onnx",
                    "path": destination_relative.as_posix(),
                    "bytes": byte_count,
                    "sha256": destination_sha,
                    "redistribution": artifact.redistribution,
                    "license": artifact.license,
                    "source": artifact.source,
                    "model_card": "MODEL_CARD.md",
                }
            )
            audit_artifacts.append(
                {
                    "id": artifact.artifact_id,
                    "role": artifact.role,
                    "path": destination_relative.as_posix(),
                    "bytes": byte_count,
                    "sha256": destination_sha,
                    "onnx_interface": onnx_audit,
                }
            )
        manifest = {
            "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
            "schema_version": 1,
            "profile": spec.profile,
            "artifacts": manifest_artifacts,
        }
        _write_bytes(stage / "manifest.json", _json_bytes(manifest))
        verified = load_and_verify_tracker_model_manifest(stage / "manifest.json")
        if tuple(verified.by_role()) != TRACKER_REQUIRED_MODEL_ROLES:
            raise ModelPackageError("generated runtime manifest role order is not canonical")
        audit = {
            "schema": TRACKER_MODEL_PACKAGE_AUDIT_SCHEMA,
            "schema_version": 1,
            "purpose": spec.purpose,
            "profile": spec.profile,
            "metadata_gate": "passed",
            "legal_clearance": "not-evaluated",
            "publication_approval": "not-evaluated",
            "quality_claim": "unmeasured",
            "onnx_input_trust": "operator-asserted-reviewed",
            "model_card": "MODEL_CARD.md",
            "runtime_manifest": "manifest.json",
            "artifacts": audit_artifacts,
        }
        _write_bytes(stage / "audit.json", _json_bytes(audit))
        checksum_names = sorted(
            [
                "MODEL_CARD.md",
                "audit.json",
                "manifest.json",
                *(f"artifacts/{role}.onnx" for role in TRACKER_REQUIRED_MODEL_ROLES),
            ]
        )
        checksum_text = "".join(f"{_sha256(stage / name)}  {name}\n" for name in checksum_names)
        _write_bytes(stage / "SHA256SUMS", checksum_text.encode("utf-8"))
        _scan_package_text(stage)
        stage.chmod(0o755)
        stage.replace(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "packaged",
        "profile": spec.profile,
        "purpose": spec.purpose,
        "manifest": "manifest.json",
        "artifacts": audit_artifacts,
        "legal_clearance": "not-evaluated",
        "publication_approval": "not-evaluated",
        "quality_claim": "unmeasured",
    }
