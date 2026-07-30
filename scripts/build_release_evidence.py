#!/usr/bin/env python3
"""Build and verify a deterministic dependency-evidence bundle for a release.

The bundle is deliberately narrower than a legal-clearance packet. It preserves
the lock-derived CycloneDX reports, dependency declarations, repository notices,
and a checksum receipt for one clean Git commit. Platform binaries, containers,
and separately published assets still require their own inventories and review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_SCHEMA = "cubed-core/release-dependency-evidence-v1"
UV_VERSION = "0.11.16"
NODE_VERSION = "22.23.1"
NPM_VERSION = "10.9.8"
CYCLONEDX_VERSION = "1.5"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOWER_HEX_RE = re.compile(r"^[0-9a-f]+$")
UUID_NAMESPACE = uuid.UUID("28159e7e-67c5-58fb-b291-53f143230d11")

PYTHON_PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("python-workbench", ("dev", "label", "decode")),
    (
        "python-label-cpu",
        ("dev", "label", "decode", "tracker-cpu"),
    ),
    (
        "python-decode-gpu",
        ("dev", "label", "decode", "tracker-gpu"),
    ),
    (
        "python-research-gpu",
        (
            "dev",
            "label",
            "decode",
            "tracker-gpu",
            "research-gpu",
        ),
    ),
    ("python-training", ("training",)),
)

COPIED_INPUTS: tuple[tuple[str, str], ...] = (
    ("LICENSE", "notices/LICENSE"),
    ("NOTICE", "notices/NOTICE"),
    ("THIRD_PARTY_NOTICES", "notices/THIRD_PARTY_NOTICES"),
    ("pyproject.toml", "inputs/pyproject.toml"),
    ("uv.lock", "inputs/uv.lock"),
    ("apps/lab-web/package.json", "inputs/apps/lab-web/package.json"),
    ("apps/lab-web/package-lock.json", "inputs/apps/lab-web/package-lock.json"),
)

RELEASE_LIMITATIONS: tuple[str, ...] = (
    "Lock-derived SBOMs are not selected-platform wheel or native-binary receipts.",
    "The frontend lock SBOM is not an inventory of emitted browser chunks.",
    "Container base images, system packages, FFmpeg/codecs, CUDA, and cuDNN are excluded.",
    "Models, training inputs, datasets, demo media, and mobile platform terms are excluded.",
    "License compatibility and release clearance require separate maintainer/counsel review.",
)

Runner = Callable[[list[str], Path, dict[str, str]], str]
BinaryRunner = Callable[[list[str], Path, dict[str, str]], bytes]


class ReleaseEvidenceError(ValueError):
    """A safe, actionable release-evidence failure."""


class _DuplicateJSONKeyError(ValueError):
    """Internal signal for an ambiguous JSON object."""


class _InvalidJSONConstantError(ValueError):
    """Internal signal for a non-standard JSON numeric constant."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _InvalidJSONConstantError(f"non-standard JSON constant {value!r}")


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKeyError, _InvalidJSONConstantError) as exc:
        raise ReleaseEvidenceError(f"{label} did not produce unambiguous JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"{label} must produce a JSON object")
    return value


def _run_text(args: list[str], cwd: Path, env: dict[str, str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ReleaseEvidenceError(f"could not run {args[0]!r}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise ReleaseEvidenceError(
            f"{args[0]!r} failed with exit code {result.returncode}: {detail[-2000:]}"
        )
    return result.stdout


def _run_bytes(args: list[str], cwd: Path, env: dict[str, str]) -> bytes:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ReleaseEvidenceError(f"could not run {args[0]!r}: {exc}") from exc
    if result.returncode != 0:
        diagnostic = result.stderr or result.stdout
        detail = diagnostic[-2000:].decode("utf-8", errors="replace").strip()
        raise ReleaseEvidenceError(
            f"{args[0]!r} failed with exit code {result.returncode}: "
            f"{detail or 'no diagnostic output'}"
        )
    return result.stdout


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(root: Path, relative_path: str, data: bytes) -> dict[str, object]:
    destination = root / PurePosixPath(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.chmod(0o644)
    return {
        "path": relative_path,
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _commit_timestamp(epoch_text: str) -> tuple[int, str]:
    try:
        epoch = int(epoch_text.strip())
    except ValueError as exc:
        raise ReleaseEvidenceError("Git commit timestamp was not an integer") from exc
    if epoch < 0:
        raise ReleaseEvidenceError("Git commit timestamp cannot be negative")
    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return epoch, timestamp


def _git_object_id(value: str, *, object_format: str, label: str) -> str:
    lengths = {"sha1": 40, "sha256": 64}
    expected_length = lengths.get(object_format)
    if expected_length is None:
        raise ReleaseEvidenceError(f"unsupported Git object format: {object_format!r}")
    value = value.strip()
    if len(value) != expected_length or not LOWER_HEX_RE.fullmatch(value):
        raise ReleaseEvidenceError(f"Git did not return a full {object_format} {label} identity")
    return value


def _validate_tool_versions(
    *,
    uv_output: str,
    node_output: str,
    npm_versions_output: str,
) -> dict[str, str]:
    uv_match = re.fullmatch(r"uv ([0-9]+\.[0-9]+\.[0-9]+)(?: .*)?", uv_output.strip())
    uv_version = uv_match.group(1) if uv_match else ""
    node_version = node_output.strip().removeprefix("v")
    npm_versions = _json_object(npm_versions_output, "npm --versions --json")
    npm_version_value = npm_versions.get("npm")
    npm_node_version_value = npm_versions.get("node")
    npm_version = npm_version_value if isinstance(npm_version_value, str) else ""
    npm_node_version = (
        npm_node_version_value.removeprefix("v") if isinstance(npm_node_version_value, str) else ""
    )
    expected = {
        "uv": UV_VERSION,
        "node": NODE_VERSION,
        "npm": NPM_VERSION,
    }
    actual = {
        "uv": uv_version,
        "node": node_version,
        "npm": npm_version,
    }
    mismatches = [
        f"{name} {required} is required; found {actual[name] or 'unknown'}"
        for name, required in expected.items()
        if actual[name] != required
    ]
    if npm_node_version != NODE_VERSION:
        mismatches.append(
            f"npm must run under Node {NODE_VERSION}; found {npm_node_version or 'unknown'}"
        )
    if mismatches:
        raise ReleaseEvidenceError("; ".join(mismatches))
    return actual


def normalize_sbom(
    document: dict[str, Any],
    *,
    profile: str,
    source_commit: str,
    commit_timestamp: str,
) -> dict[str, Any]:
    """Replace generator entropy with deterministic, source-bound metadata."""

    normalized = copy.deepcopy(document)
    if normalized.get("bomFormat") != "CycloneDX":
        raise ReleaseEvidenceError(f"{profile} output is not a CycloneDX BOM")
    if normalized.get("specVersion") != CYCLONEDX_VERSION:
        raise ReleaseEvidenceError(f"{profile} output must use CycloneDX {CYCLONEDX_VERSION}")
    components = normalized.get("components")
    dependencies = normalized.get("dependencies")
    if not isinstance(components, list) or not isinstance(dependencies, list):
        raise ReleaseEvidenceError(f"{profile} output lacks component/dependency arrays")

    deterministic_uuid = uuid.uuid5(UUID_NAMESPACE, f"{source_commit}:{profile}")
    normalized["serialNumber"] = deterministic_uuid.urn

    metadata = normalized.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ReleaseEvidenceError(f"{profile} metadata must be an object")
    metadata["timestamp"] = commit_timestamp
    properties = metadata.setdefault("properties", [])
    if not isinstance(properties, list) or any(not isinstance(item, dict) for item in properties):
        raise ReleaseEvidenceError(f"{profile} metadata.properties must be an object array")

    owned_names = {
        "io.github.kingbobjoeiv.cubed-core:profile",
        "io.github.kingbobjoeiv.cubed-core:source-commit",
    }
    properties = [item for item in properties if item.get("name") not in owned_names]
    properties.extend(
        [
            {
                "name": "io.github.kingbobjoeiv.cubed-core:profile",
                "value": profile,
            },
            {
                "name": "io.github.kingbobjoeiv.cubed-core:source-commit",
                "value": source_commit,
            },
        ]
    )
    metadata["properties"] = sorted(
        properties,
        key=lambda item: (str(item.get("name", "")), str(item.get("value", ""))),
    )
    return normalized


def _assert_portable(document: object, *, repository_root: Path, label: str) -> None:
    forbidden = {str(repository_root), str(Path.home())}

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for prefix in forbidden:
                if prefix and prefix in value:
                    raise ReleaseEvidenceError(
                        f"{label} contains a machine-local path and cannot be released"
                    )

    walk(document)


def _validate_locations(repository_root: Path, output_dir: Path) -> tuple[Path, Path]:
    repository_root = repository_root.resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    try:
        output_dir.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ReleaseEvidenceError("output directory must be outside the source checkout")
    if os.path.lexists(output_dir):
        raise ReleaseEvidenceError(f"refusing to overwrite existing output: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ReleaseEvidenceError(f"output parent does not exist: {output_dir.parent}")
    return repository_root, output_dir


def _read_release_input(repository_root: Path, relative_path: str) -> bytes:
    source = repository_root / PurePosixPath(relative_path)
    if source.is_symlink() or not source.is_file():
        raise ReleaseEvidenceError(
            f"required release input must be a regular in-tree file: {relative_path}"
        )
    return source.read_bytes()


def _read_committed_inputs(
    repository_root: Path,
    *,
    source_commit: str,
    environment: dict[str, str],
    binary_runner: BinaryRunner,
) -> dict[str, bytes]:
    committed_inputs: dict[str, bytes] = {}
    for source_path, _bundle_path in COPIED_INPUTS:
        worktree_bytes = _read_release_input(repository_root, source_path)
        committed_bytes = binary_runner(
            ["git", "cat-file", "blob", f"{source_commit}:{source_path}"],
            repository_root,
            environment,
        )
        if worktree_bytes != committed_bytes:
            raise ReleaseEvidenceError(
                f"working-tree bytes differ from the committed blob for {source_path}; "
                "disable checkout filters or line-ending conversion before release"
            )
        committed_inputs[source_path] = committed_bytes
    return committed_inputs


def _sbom_command(profile: str, extras: tuple[str, ...], uv_command: str) -> list[str]:
    command = [
        uv_command,
        "export",
        "--locked",
        "--format",
        "cyclonedx1.5",
        "--preview-features",
        "sbom-export",
    ]
    for extra in extras:
        command.extend(["--extra", extra])
    return command


def _frontend_sbom_command(npm_command: str) -> list[str]:
    return [
        npm_command,
        "--prefix",
        "apps/lab-web",
        "sbom",
        "--package-lock-only",
        "--sbom-format=cyclonedx",
    ]


def _readme_bytes(*, source_ref: str, source_commit: str) -> bytes:
    return (
        "Cubed Core release dependency evidence\n"
        "======================================\n\n"
        f"Source ref: {source_ref}\n"
        f"Source commit: {source_commit}\n\n"
        "Verify this directory with:\n\n"
        "  shasum -a 256 --check SHA256SUMS\n\n"
        "This lock-derived bundle is evidence, not a legal-clearance conclusion. "
        "See manifest.json, THIRD_PARTY_NOTICES, and the source repository's "
        "docs/LICENSING.md for scope and limitations.\n"
    ).encode()


def _normalize_bundle_modes(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleaseEvidenceError(f"generated bundle contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)


def build_release_evidence(
    repository_root: Path,
    output_dir: Path,
    *,
    source_ref: str,
    uv_command: str = "uv",
    node_command: str = "node",
    npm_command: str = "npm",
    runner: Runner | None = None,
    binary_runner: BinaryRunner | None = None,
) -> Path:
    """Build one atomic, source-bound dependency-evidence directory."""

    repository_root, output_dir = _validate_locations(repository_root, output_dir)
    runner = runner or _run_text
    binary_runner = binary_runner or _run_bytes
    if not source_ref.strip():
        raise ReleaseEvidenceError("source ref must not be empty")

    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "TZ": "UTC",
            "UV_OFFLINE": "1",
            "npm_config_audit": "false",
            "npm_config_fund": "false",
            "npm_config_offline": "true",
        }
    )

    def run(args: list[str]) -> str:
        return runner(args, repository_root, environment)

    top_level = Path(run(["git", "rev-parse", "--show-toplevel"]).strip()).resolve(strict=True)
    if top_level != repository_root:
        raise ReleaseEvidenceError(
            f"repository root mismatch: expected {repository_root}, Git reported {top_level}"
        )
    dirty = run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ReleaseEvidenceError("source checkout must be clean before evidence generation")

    object_format = run(["git", "rev-parse", "--show-object-format"]).strip()
    source_commit = _git_object_id(
        run(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{source_ref}^{{commit}}",
            ]
        ),
        object_format=object_format,
        label="commit",
    )
    head_commit = _git_object_id(
        run(["git", "rev-parse", "--verify", "HEAD"]),
        object_format=object_format,
        label="commit",
    )
    if source_commit != head_commit:
        raise ReleaseEvidenceError(
            f"source ref resolves to {source_commit}, but the clean checkout is {head_commit}"
        )
    source_tree = _git_object_id(
        run(["git", "rev-parse", "--verify", f"{source_commit}^{{tree}}"]),
        object_format=object_format,
        label="tree",
    )
    commit_epoch, commit_timestamp = _commit_timestamp(
        run(["git", "show", "-s", "--format=%ct", source_commit])
    )
    environment["SOURCE_DATE_EPOCH"] = str(commit_epoch)
    committed_inputs = _read_committed_inputs(
        repository_root,
        source_commit=source_commit,
        environment=environment,
        binary_runner=binary_runner,
    )

    tools = _validate_tool_versions(
        uv_output=run([uv_command, "--version"]),
        node_output=run([node_command, "--version"]),
        npm_versions_output=run([npm_command, "--versions", "--json"]),
    )

    raw_sboms: list[tuple[str, list[str], dict[str, Any]]] = []
    for profile, extras in PYTHON_PROFILES:
        command = _sbom_command(profile, extras, uv_command)
        raw_sboms.append((profile, command, _json_object(run(command), profile)))
    frontend_command = _frontend_sbom_command(npm_command)
    raw_sboms.append(
        (
            "frontend-lock",
            frontend_command,
            _json_object(run(frontend_command), "frontend-lock"),
        )
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        file_records: list[dict[str, object]] = []
        for source_path, bundle_path in COPIED_INPUTS:
            file_records.append(
                _write_bytes(
                    staging,
                    bundle_path,
                    committed_inputs[source_path],
                )
            )

        sbom_records: list[dict[str, object]] = []
        for profile, command, raw_document in raw_sboms:
            normalized = normalize_sbom(
                raw_document,
                profile=profile,
                source_commit=source_commit,
                commit_timestamp=commit_timestamp,
            )
            _assert_portable(normalized, repository_root=repository_root, label=profile)
            sbom_path = f"sbom/{profile}.cdx.json"
            file_records.append(_write_bytes(staging, sbom_path, _canonical_json(normalized)))
            sbom_records.append(
                {
                    "path": sbom_path,
                    "profile": profile,
                    "component_count": len(normalized["components"]),
                    "dependency_count": len(normalized["dependencies"]),
                    "command": [
                        "uv" if token == uv_command else "npm" if token == npm_command else token
                        for token in command
                    ],
                }
            )

        file_records.append(
            _write_bytes(
                staging,
                "README.txt",
                _readme_bytes(source_ref=source_ref, source_commit=source_commit),
            )
        )
        file_records.sort(key=lambda item: str(item["path"]))
        sbom_records.sort(key=lambda item: str(item["path"]))

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": 1,
            "source": {
                "ref": source_ref,
                "object_format": object_format,
                "commit": source_commit,
                "tree": source_tree,
                "commit_timestamp": commit_timestamp,
            },
            "tools": tools,
            "sboms": sbom_records,
            "files": file_records,
            "limitations": list(RELEASE_LIMITATIONS),
        }
        manifest_record = _write_bytes(staging, "manifest.json", _canonical_json(manifest))

        checksummed = [*file_records, manifest_record]
        checksum_lines = [
            f"{item['sha256']}  {item['path']}\n"
            for item in sorted(checksummed, key=lambda item: str(item["path"]))
        ]
        (staging / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
        _normalize_bundle_modes(staging)

        verify_release_evidence(staging)
        final_ref_commit = _git_object_id(
            run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{source_ref}^{{commit}}",
                ]
            ),
            object_format=object_format,
            label="commit",
        )
        final_head_commit = _git_object_id(
            run(["git", "rev-parse", "--verify", "HEAD"]),
            object_format=object_format,
            label="commit",
        )
        final_source_tree = _git_object_id(
            run(["git", "rev-parse", "--verify", f"{final_head_commit}^{{tree}}"]),
            object_format=object_format,
            label="tree",
        )
        final_dirty = run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
        if (
            final_ref_commit != source_commit
            or final_head_commit != source_commit
            or final_source_tree != source_tree
            or final_dirty
        ):
            raise ReleaseEvidenceError(
                "source ref, HEAD, tree, or worktree changed during evidence generation"
            )
        final_inputs = _read_committed_inputs(
            repository_root,
            source_commit=source_commit,
            environment=environment,
            binary_runner=binary_runner,
        )
        if final_inputs != committed_inputs:
            raise ReleaseEvidenceError("release inputs changed during evidence generation")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _safe_checksum_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ReleaseEvidenceError(f"unsafe checksum path: {raw_path!r}")
    return path


def _read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceError(f"{label} must be UTF-8") from exc


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReleaseEvidenceError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _manifest_object_id(value: object, *, object_format: str, label: str) -> str:
    lengths = {"sha1": 40, "sha256": 64}
    expected_length = lengths.get(object_format)
    if expected_length is None:
        raise ReleaseEvidenceError(
            f"manifest source.object_format is unsupported: {object_format!r}"
        )
    if (
        not isinstance(value, str)
        or len(value) != expected_length
        or not LOWER_HEX_RE.fullmatch(value)
    ):
        raise ReleaseEvidenceError(
            f"manifest source.{label} must be a full lowercase {object_format} object ID"
        )
    return value


def _canonical_utc_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ReleaseEvidenceError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ReleaseEvidenceError(f"{label} must be a canonical UTC timestamp") from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise ReleaseEvidenceError(f"{label} must be a canonical UTC timestamp")
    return value


def _validate_manifest_header(manifest: dict[str, Any]) -> tuple[str, str, str]:
    _exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "source",
            "tools",
            "sboms",
            "files",
            "limitations",
        },
        "manifest",
    )
    if manifest["schema"] != BUNDLE_SCHEMA or type(manifest["schema_version"]) is not int:
        raise ReleaseEvidenceError("manifest.json has an unsupported schema")
    if manifest["schema_version"] != 1:
        raise ReleaseEvidenceError("manifest.json has an unsupported schema")

    source = manifest["source"]
    if not isinstance(source, dict):
        raise ReleaseEvidenceError("manifest source must be an object")
    _exact_keys(
        source,
        {"ref", "object_format", "commit", "tree", "commit_timestamp"},
        "manifest source",
    )
    source_ref = source["ref"]
    if (
        not isinstance(source_ref, str)
        or not source_ref.strip()
        or any(ord(character) < 32 for character in source_ref)
    ):
        raise ReleaseEvidenceError("manifest source.ref must be a non-empty printable string")
    object_format = source["object_format"]
    if not isinstance(object_format, str):
        raise ReleaseEvidenceError("manifest source.object_format must be a string")
    source_commit = _manifest_object_id(
        source["commit"],
        object_format=object_format,
        label="commit",
    )
    _manifest_object_id(source["tree"], object_format=object_format, label="tree")
    commit_timestamp = _canonical_utc_timestamp(
        source["commit_timestamp"],
        "manifest source.commit_timestamp",
    )

    tools = manifest["tools"]
    if not isinstance(tools, dict):
        raise ReleaseEvidenceError("manifest tools must be an object")
    _exact_keys(tools, {"uv", "node", "npm"}, "manifest tools")
    expected_tools = {
        "uv": UV_VERSION,
        "node": NODE_VERSION,
        "npm": NPM_VERSION,
    }
    if tools != expected_tools:
        raise ReleaseEvidenceError(
            f"manifest tools must equal the pinned generator versions: {expected_tools}"
        )
    limitations = manifest["limitations"]
    if limitations != list(RELEASE_LIMITATIONS):
        raise ReleaseEvidenceError("manifest limitations do not match the release-evidence schema")
    return source_ref, source_commit, commit_timestamp


def _expected_sbom_profiles() -> dict[str, tuple[str, tuple[str, ...]]]:
    profiles = {
        profile: (f"sbom/{profile}.cdx.json", extras) for profile, extras in PYTHON_PROFILES
    }
    profiles["frontend-lock"] = ("sbom/frontend-lock.cdx.json", ())
    return profiles


def _expected_payload_paths() -> set[str]:
    return {
        "README.txt",
        *(bundle_path for _source_path, bundle_path in COPIED_INPUTS),
        *(path for path, _extras in _expected_sbom_profiles().values()),
    }


def _expected_sbom_command(profile: str, extras: tuple[str, ...]) -> list[str]:
    if profile == "frontend-lock":
        return _frontend_sbom_command("npm")
    return _sbom_command(profile, extras, "uv")


def _validate_normalized_sbom(
    document: dict[str, Any],
    raw_bytes: bytes,
    *,
    profile: str,
    source_commit: str,
    commit_timestamp: str,
    component_count: int,
    dependency_count: int,
) -> None:
    if raw_bytes != _canonical_json(document):
        raise ReleaseEvidenceError(f"{profile} SBOM is not canonically serialized")
    expected = normalize_sbom(
        document,
        profile=profile,
        source_commit=source_commit,
        commit_timestamp=commit_timestamp,
    )
    if document != expected:
        raise ReleaseEvidenceError(
            f"{profile} SBOM is not normalized and bound to the manifest source"
        )
    if len(document["components"]) != component_count:
        raise ReleaseEvidenceError(f"{profile} component_count does not match the SBOM")
    if len(document["dependencies"]) != dependency_count:
        raise ReleaseEvidenceError(f"{profile} dependency_count does not match the SBOM")


def verify_release_evidence(bundle_dir: Path) -> dict[str, Any]:
    """Verify bundle bytes plus the fail-closed source and SBOM receipt schema."""

    bundle_dir = bundle_dir.expanduser().resolve(strict=True)
    if not bundle_dir.is_dir():
        raise ReleaseEvidenceError(f"bundle is not a directory: {bundle_dir}")
    checksum_path = bundle_dir / "SHA256SUMS"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ReleaseEvidenceError("bundle lacks a regular SHA256SUMS file")

    checksum_text = _read_utf8(checksum_path, "SHA256SUMS")
    expected_hashes: dict[str, str] = {}
    for line_number, line in enumerate(checksum_text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ReleaseEvidenceError(f"invalid SHA256SUMS line {line_number}")
        digest, raw_path = match.groups()
        relative = _safe_checksum_path(raw_path).as_posix()
        if relative in expected_hashes:
            raise ReleaseEvidenceError(f"duplicate SHA256SUMS path: {relative}")
        expected_hashes[relative] = digest

    actual_paths: set[str] = set()
    for path in bundle_dir.rglob("*"):
        if path.is_symlink():
            raise ReleaseEvidenceError(f"bundle contains a symlink: {path.relative_to(bundle_dir)}")
        if path.is_file() and path != checksum_path:
            actual_paths.add(path.relative_to(bundle_dir).as_posix())
        elif not path.is_dir() and path != checksum_path:
            raise ReleaseEvidenceError(
                f"bundle contains a non-regular entry: {path.relative_to(bundle_dir)}"
            )
    if set(expected_hashes) != actual_paths:
        missing = sorted(actual_paths - set(expected_hashes))
        extra = sorted(set(expected_hashes) - actual_paths)
        raise ReleaseEvidenceError(
            f"SHA256SUMS inventory mismatch; unlisted={missing}, absent={extra}"
        )
    canonical_checksum_text = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(expected_hashes.items())
    )
    if checksum_text != canonical_checksum_text:
        raise ReleaseEvidenceError("SHA256SUMS is not in canonical path order")

    expected_payload_paths = _expected_payload_paths()
    expected_bundle_paths = expected_payload_paths | {"manifest.json"}
    if actual_paths != expected_bundle_paths:
        missing = sorted(expected_bundle_paths - actual_paths)
        extra = sorted(actual_paths - expected_bundle_paths)
        raise ReleaseEvidenceError(f"bundle file schema mismatch; missing={missing}, extra={extra}")

    for relative, expected_digest in sorted(expected_hashes.items()):
        actual_digest = _sha256((bundle_dir / PurePosixPath(relative)).read_bytes())
        if actual_digest != expected_digest:
            raise ReleaseEvidenceError(f"checksum mismatch: {relative}")

    manifest_path = bundle_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _json_object(_read_utf8(manifest_path, "manifest.json"), "manifest.json")
    if manifest_bytes != _canonical_json(manifest):
        raise ReleaseEvidenceError("manifest.json is not canonically serialized")
    source_ref, source_commit, commit_timestamp = _validate_manifest_header(manifest)
    file_records = manifest["files"]
    sbom_records = manifest["sboms"]
    if not isinstance(file_records, list) or not isinstance(sbom_records, list):
        raise ReleaseEvidenceError("manifest files/sboms must be arrays")

    manifest_payload_paths: set[str] = set()
    ordered_file_paths: list[str] = []
    for record in file_records:
        if not isinstance(record, dict):
            raise ReleaseEvidenceError("manifest file record must be an object")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "manifest file record")
        raw_path = record["path"]
        if not isinstance(raw_path, str):
            raise ReleaseEvidenceError("manifest file path must be a string")
        relative = _safe_checksum_path(raw_path).as_posix()
        if relative not in expected_payload_paths:
            raise ReleaseEvidenceError(f"unexpected manifest file path: {relative}")
        digest = record["sha256"]
        size_bytes = record["size_bytes"]
        if (
            relative in manifest_payload_paths
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise ReleaseEvidenceError(f"invalid manifest file record: {relative}")
        payload = (bundle_dir / PurePosixPath(relative)).read_bytes()
        if (
            _sha256(payload) != digest
            or len(payload) != size_bytes
            or expected_hashes.get(relative) != digest
        ):
            raise ReleaseEvidenceError(f"manifest file receipt mismatch: {relative}")
        manifest_payload_paths.add(relative)
        ordered_file_paths.append(relative)

    if manifest_payload_paths != expected_payload_paths:
        raise ReleaseEvidenceError("manifest file inventory does not cover the exact payload")
    if ordered_file_paths != sorted(ordered_file_paths):
        raise ReleaseEvidenceError("manifest file records are not in canonical path order")

    if (bundle_dir / "README.txt").read_bytes() != _readme_bytes(
        source_ref=source_ref,
        source_commit=source_commit,
    ):
        raise ReleaseEvidenceError("README.txt does not match the manifest source")

    expected_profiles = _expected_sbom_profiles()
    sboms_by_profile: dict[str, dict[str, Any]] = {}
    ordered_sbom_paths: list[str] = []
    for record in sbom_records:
        if not isinstance(record, dict):
            raise ReleaseEvidenceError("manifest SBOM record must be an object")
        _exact_keys(
            record,
            {
                "path",
                "profile",
                "component_count",
                "dependency_count",
                "command",
            },
            "manifest SBOM record",
        )
        profile = record["profile"]
        raw_path = record["path"]
        if not isinstance(profile, str) or profile not in expected_profiles:
            raise ReleaseEvidenceError(f"unknown manifest SBOM profile: {profile!r}")
        if profile in sboms_by_profile:
            raise ReleaseEvidenceError(f"duplicate manifest SBOM profile: {profile}")
        if not isinstance(raw_path, str):
            raise ReleaseEvidenceError(f"{profile} SBOM path must be a string")
        relative = _safe_checksum_path(raw_path).as_posix()
        expected_path, extras = expected_profiles[profile]
        if relative != expected_path:
            raise ReleaseEvidenceError(f"{profile} SBOM path must be {expected_path}")
        component_count = record["component_count"]
        dependency_count = record["dependency_count"]
        if (
            type(component_count) is not int
            or component_count < 0
            or type(dependency_count) is not int
            or dependency_count < 0
        ):
            raise ReleaseEvidenceError(f"{profile} SBOM counts must be non-negative integers")
        command = record["command"]
        if (
            not isinstance(command, list)
            or any(not isinstance(token, str) for token in command)
            or command != _expected_sbom_command(profile, extras)
        ):
            raise ReleaseEvidenceError(f"{profile} SBOM command does not match its profile")
        sboms_by_profile[profile] = record
        ordered_sbom_paths.append(relative)

    if set(sboms_by_profile) != set(expected_profiles):
        raise ReleaseEvidenceError("manifest does not name the exact expected SBOM profiles")
    if ordered_sbom_paths != sorted(ordered_sbom_paths):
        raise ReleaseEvidenceError("manifest SBOM records are not in canonical path order")

    for profile, (relative, _extras) in sorted(expected_profiles.items()):
        record = sboms_by_profile[profile]
        sbom_path = bundle_dir / PurePosixPath(relative)
        raw_bytes = sbom_path.read_bytes()
        document = _json_object(_read_utf8(sbom_path, profile), profile)
        _validate_normalized_sbom(
            document,
            raw_bytes,
            profile=profile,
            source_commit=source_commit,
            commit_timestamp=commit_timestamp,
            component_count=record["component_count"],
            dependency_count=record["dependency_count"],
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a new evidence directory")
    build.add_argument("--source-ref", required=True, help="tag or full commit checked out")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument("--uv", default="uv")
    build.add_argument("--node", default="node")
    build.add_argument("--npm", default="npm")

    verify = subparsers.add_parser("verify", help="verify an existing evidence directory")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            output = build_release_evidence(
                args.repository_root,
                args.output_dir,
                source_ref=args.source_ref,
                uv_command=args.uv,
                node_command=args.node,
                npm_command=args.npm,
            )
            print(output)
        else:
            manifest = verify_release_evidence(args.bundle_dir)
            print(
                f"verified {args.bundle_dir}: "
                f"{manifest['source']['commit']} ({len(manifest['files'])} payload files)"
            )
    except ReleaseEvidenceError as exc:
        print(f"release evidence error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
