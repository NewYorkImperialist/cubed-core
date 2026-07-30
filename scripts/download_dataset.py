#!/usr/bin/env python3
"""Download one immutable, checksummed Cubed Core public dataset revision."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cubed-core/public-dataset-download-v1"
RECEIPT_SCHEMA = "cubed-core/public-dataset-download-receipt-v3"
DEFAULT_MANIFEST = Path("config/public-dataset-v1.json")
DEFAULT_OUTPUT_DIR = Path("datasets/public/cubed-solves-v1")
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
USER_AGENT = "cubed-core-dataset-download/1"
REQUIRED_PUBLISHED_ARTIFACTS = frozenset(
    PurePosixPath(path)
    for path in (
        ".gitattributes",
        "README.md",
        "LICENSES/DATASET.md",
        "dataset/manifest.json",
        "metadata.jsonl",
        "readiness-report.json",
        "SHA256SUMS",
    )
)
BENCHMARK_V0_ARTIFACTS = frozenset(
    PurePosixPath(path)
    for path in (
        "public-inputs/manifest.json",
        "public-inputs/splits.json",
        "teacher/manifest.json",
    )
)


class DatasetDownloadError(RuntimeError):
    """Raised when a public dataset cannot be downloaded and verified safely."""


@dataclass(frozen=True)
class Artifact:
    path: PurePosixPath
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    description: str
    publication_status: str
    source_kind: str | None
    revision: str | None
    base_url_template: str | None
    artifacts: tuple[Artifact, ...]
    sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a published Cubed Core dataset revision and verify every "
            "artifact against the repository's public download manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"download manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--base-url",
        help=(
            "override the manifest's public artifact root; HTTPS is required "
            "except for loopback testing, and one {revision} placeholder is required"
        ),
    )
    parser.add_argument(
        "--revision",
        help="override the immutable dataset revision recorded in the receipt",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help=(
            "after full verification, hard-link every published video into this "
            "Cubed Core workspace as an incomplete Decode capture"
        ),
    )
    return parser


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetDownloadError(f"{context} must be a JSON object")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing:
        raise DatasetDownloadError(f"{context} is missing: {', '.join(missing)}")
    if extra:
        raise DatasetDownloadError(f"{context} has unknown fields: {', '.join(extra)}")


def _validate_revision(value: Any, context: str = "revision") -> str:
    if not isinstance(value, str) or not 7 <= len(value) <= 128:
        raise DatasetDownloadError(f"{context} must be a 7-128 character immutable identifier")
    if not value[0].isalnum() or any(
        not (character.isalnum() or character in "._-") for character in value
    ):
        raise DatasetDownloadError(f"{context} contains unsafe characters")
    return value


def _artifact_path(value: Any, index: int) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 500:
        raise DatasetDownloadError(f"artifacts[{index}].path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise DatasetDownloadError(f"artifacts[{index}].path is not a normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DatasetDownloadError(f"artifacts[{index}].path is not a normalized relative path")
    if "download-receipt.json" in path.parts:
        raise DatasetDownloadError("artifact path download-receipt.json is reserved")
    return path


def _validate_artifact_path_set(paths: set[PurePosixPath]) -> None:
    for path in sorted(paths, key=lambda item: item.as_posix()):
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            if parent in paths:
                raise DatasetDownloadError(
                    "artifact paths have a file/child conflict: "
                    f"{parent.as_posix()} and {path.as_posix()}"
                )


def load_manifest(path: Path) -> DatasetManifest:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DatasetDownloadError(f"could not read manifest {path}: {error}") from error
    try:
        document = _require_object(json.loads(raw), "manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetDownloadError(f"manifest is not valid UTF-8 JSON: {error}") from error

    required = {
        "schema",
        "dataset_id",
        "description",
        "publication_status",
        "revision",
        "source",
        "artifacts",
    }
    _require_exact_keys(document, required=required, context="manifest")
    if document["schema"] != SCHEMA:
        raise DatasetDownloadError(f"manifest schema must be {SCHEMA}")

    dataset_id = document["dataset_id"]
    if (
        not isinstance(dataset_id, str)
        or not 3 <= len(dataset_id) <= 128
        or not dataset_id[0].isalnum()
        or any(
            not (character.islower() or character.isdigit() or character in "._-")
            for character in dataset_id
        )
    ):
        raise DatasetDownloadError("dataset_id must be a lowercase, path-safe identifier")
    description = document["description"]
    if not isinstance(description, str) or not 1 <= len(description) <= 500:
        raise DatasetDownloadError("description must be a non-empty string")

    status = document["publication_status"]
    if status not in {"pending-rights-review", "published", "withdrawn"}:
        raise DatasetDownloadError("publication_status is not recognized")

    revision_value = document["revision"]
    revision = None if revision_value is None else _validate_revision(revision_value)
    source_value = document["source"]
    source_kind: str | None = None
    base_url_template: str | None = None
    if source_value is not None:
        source = _require_object(source_value, "source")
        _require_exact_keys(
            source,
            required={"kind", "base_url_template"},
            context="source",
        )
        if source["kind"] not in {"hugging-face", "direct-mirror"}:
            raise DatasetDownloadError("source.kind is not recognized")
        source_kind = source["kind"]
        if not isinstance(source["base_url_template"], str):
            raise DatasetDownloadError("source.base_url_template must be a string")
        base_url_template = source["base_url_template"]
        _validate_base_url_template(base_url_template)

    artifacts_value = document["artifacts"]
    if not isinstance(artifacts_value, list):
        raise DatasetDownloadError("artifacts must be an array")
    artifacts: list[Artifact] = []
    seen_paths: set[PurePosixPath] = set()
    for index, value in enumerate(artifacts_value):
        artifact = _require_object(value, f"artifacts[{index}]")
        _require_exact_keys(
            artifact,
            required={"path", "bytes", "sha256"},
            context=f"artifacts[{index}]",
        )
        relative_path = _artifact_path(artifact["path"], index)
        if relative_path in seen_paths:
            raise DatasetDownloadError(f"duplicate artifact path: {relative_path}")
        seen_paths.add(relative_path)
        size = artifact["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DatasetDownloadError(f"artifacts[{index}].bytes must be a positive integer")
        sha256 = artifact["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise DatasetDownloadError(f"artifacts[{index}].sha256 must be lowercase SHA-256")
        artifacts.append(Artifact(relative_path, size, sha256))

    _validate_artifact_path_set(seen_paths)

    benchmark_files = seen_paths & BENCHMARK_V0_ARTIFACTS
    if benchmark_files and benchmark_files != BENCHMARK_V0_ARTIFACTS:
        missing_benchmark_files = BENCHMARK_V0_ARTIFACTS - seen_paths
        names = ", ".join(sorted(path.as_posix() for path in missing_benchmark_files))
        raise DatasetDownloadError(
            f"manifest has a partial Benchmark v0 contract; missing: {names}"
        )

    if status == "published":
        if revision is None or base_url_template is None:
            raise DatasetDownloadError("a published manifest requires a revision and source")
        if source_kind == "hugging-face" and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise DatasetDownloadError(
                "a Hugging Face dataset revision must be its full 40-character commit hash"
            )
        missing_contract_files = REQUIRED_PUBLISHED_ARTIFACTS - seen_paths
        if missing_contract_files:
            names = ", ".join(sorted(path.as_posix() for path in missing_contract_files))
            raise DatasetDownloadError(
                f"a published manifest is missing required dataset contract files: {names}"
            )

    return DatasetManifest(
        dataset_id=dataset_id,
        description=description,
        publication_status=status,
        source_kind=source_kind,
        revision=revision,
        base_url_template=base_url_template,
        artifacts=tuple(artifacts),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _is_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_download_url(value: str, *, allow_loopback_http: bool) -> urllib.parse.SplitResult:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise DatasetDownloadError("dataset URL contains control characters")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise DatasetDownloadError(f"invalid dataset URL: {error}") from error
    if parsed.username or parsed.password:
        raise DatasetDownloadError("dataset URL must not contain credentials")
    if parsed.scheme == "https" and hostname:
        return parsed
    if parsed.scheme == "http" and allow_loopback_http and _is_loopback(hostname):
        return parsed
    raise DatasetDownloadError("dataset URL must use HTTPS or permitted loopback HTTP")


def _validate_base_url_template(value: str) -> None:
    if not value or len(value) > 2048 or value.count("{revision}") != 1:
        raise DatasetDownloadError(
            "base URL template must contain exactly one {revision} placeholder"
        )
    probe = value.replace("{revision}", "revision0")
    if "{" in probe or "}" in probe:
        raise DatasetDownloadError("base URL template has an unknown placeholder")
    parsed = _validate_download_url(probe, allow_loopback_http=True)
    if parsed.query or parsed.fragment:
        raise DatasetDownloadError("dataset base URL must not contain a query or fragment")


def _resolve_base_url(template: str, revision: str) -> str:
    _validate_base_url_template(template)
    base_url = template.replace(
        "{revision}",
        urllib.parse.quote(revision, safe=""),
    ).rstrip("/")
    parsed = _validate_download_url(base_url, allow_loopback_http=True)
    if parsed.query or parsed.fragment:
        raise DatasetDownloadError("dataset base URL must not contain a query or fragment")
    return base_url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allow_loopback_http: bool) -> None:
        super().__init__()
        self._allow_loopback_http = allow_loopback_http

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_download_url(target, allow_loopback_http=self._allow_loopback_http)
        return super().redirect_request(req, fp, code, msg, headers, target)


def _open_url(url: str):
    parsed = _validate_download_url(url, allow_loopback_http=True)
    allow_loopback_http = parsed.scheme == "http"
    opener = urllib.request.build_opener(
        _SafeRedirectHandler(allow_loopback_http=allow_loopback_http)
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return opener.open(request, timeout=60)
    except (urllib.error.URLError, OSError) as error:
        raise DatasetDownloadError(f"could not download {url}: {error}") from error


def _download_artifact(url: str, destination: Path, artifact: Artifact) -> None:
    digest = hashlib.sha256()
    received = 0
    try:
        with _open_url(url) as response, destination.open("xb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise DatasetDownloadError(
                        f"invalid Content-Length for {artifact.path}"
                    ) from error
                if declared_size != artifact.bytes:
                    raise DatasetDownloadError(
                        f"{artifact.path} expected {artifact.bytes} bytes; server declared "
                        f"{declared_size}"
                    )
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                received += len(chunk)
                if received > artifact.bytes:
                    raise DatasetDownloadError(
                        f"{artifact.path} exceeds its declared {artifact.bytes} bytes"
                    )
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if received != artifact.bytes:
        destination.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"{artifact.path} expected {artifact.bytes} bytes; received {received}"
        )
    if digest.hexdigest() != artifact.sha256:
        destination.unlink(missing_ok=True)
        raise DatasetDownloadError(f"{artifact.path} failed SHA-256 verification")


def _artifact_matches(path: Path, artifact: Artifact) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != artifact.bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest() == artifact.sha256


def _receipt(
    manifest: DatasetManifest,
    *,
    revision: str,
    base_url: str,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": manifest.sha256,
        "revision": revision,
        "base_url": base_url,
        "artifacts": [
            {
                "path": artifact.path.as_posix(),
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            }
            for artifact in manifest.artifacts
        ],
    }


def _verify_existing(output_dir: Path, expected_receipt: dict[str, Any]) -> bool:
    receipt_path = output_dir / "download-receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if existing != expected_receipt:
        return False
    expected_paths = {"download-receipt.json"}
    for artifact in expected_receipt["artifacts"]:
        path = output_dir.joinpath(*PurePosixPath(artifact["path"]).parts)
        expected_paths.add(artifact["path"])
        if not _artifact_matches(
            path,
            Artifact(
                path=PurePosixPath(artifact["path"]),
                bytes=artifact["bytes"],
                sha256=artifact["sha256"],
            ),
        ):
            return False
    actual_paths = {
        path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        return False
    return True


def download_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    base_url_override: str | None = None,
    revision_override: str | None = None,
) -> DatasetManifest:
    manifest = load_manifest(manifest_path)
    if manifest.publication_status != "published":
        raise DatasetDownloadError(
            f"{manifest.dataset_id} is not published "
            f"(manifest status: {manifest.publication_status}); no dataset bytes were requested"
        )
    revision = (
        _validate_revision(revision_override, "revision override")
        if revision_override is not None
        else manifest.revision
    )
    if (
        manifest.source_kind == "hugging-face"
        and revision is not None
        and not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise DatasetDownloadError(
            "a Hugging Face dataset revision override must be a full 40-character commit hash"
        )
    template = base_url_override or manifest.base_url_template
    if revision is None or template is None:
        raise DatasetDownloadError("published dataset source is incomplete")
    base_url = _resolve_base_url(template, revision)
    expected_receipt = _receipt(manifest, revision=revision, base_url=base_url)

    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise DatasetDownloadError(f"output path is not a real directory: {output_dir}")
        if _verify_existing(output_dir, expected_receipt):
            print(f"Dataset already verified: {output_dir}")
            return manifest
        raise DatasetDownloadError(
            f"refusing to overwrite unverified or different dataset directory: {output_dir}"
        )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.part-", dir=output_parent))
    cache = output_parent / (f".{output_dir.name}.cache-{manifest.sha256[:16]}-{revision}")
    cache.mkdir(parents=True, exist_ok=True)
    try:
        for artifact in manifest.artifacts:
            destination = stage.joinpath(*artifact.path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cached = cache.joinpath(*artifact.path.parts)
            cached.parent.mkdir(parents=True, exist_ok=True)
            if _artifact_matches(cached, artifact):
                print(f"Using verified cache for {artifact.path}")
            else:
                cached.unlink(missing_ok=True)
                partial = cached.with_name(f".{cached.name}.partial")
                partial.unlink(missing_ok=True)
                artifact_url = (
                    f"{base_url}/{urllib.parse.quote(artifact.path.as_posix(), safe='/')}"
                )
                print(f"Downloading {artifact.path} ({artifact.bytes} bytes)")
                _download_artifact(artifact_url, partial, artifact)
                os.replace(partial, cached)
            try:
                os.link(cached, destination)
            except OSError:
                shutil.copyfile(cached, destination)
        (stage / "download-receipt.json").write_text(
            json.dumps(expected_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output_dir)
        shutil.rmtree(cache, ignore_errors=True)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        f"Verified {len(manifest.artifacts)} artifacts for "
        f"{manifest.dataset_id}@{revision}: {output_dir}"
    )
    return manifest


def _register_workspace(
    manifest: DatasetManifest,
    output_dir: Path,
    workspace_dir: Path,
) -> None:
    try:
        from cubed_core.public_dataset_registration import (
            PublicDatasetRegistrationError,
            register_downloaded_public_dataset,
        )
        from cubed_core.settings import Settings
        from cubed_core.workspace import Workspace, WorkspaceError
    except ImportError as exc:
        raise DatasetDownloadError(
            "workspace registration requires the installed cubed-core package"
        ) from exc

    try:
        settings = Settings.from_env()
        workspace = Workspace(
            workspace_dir,
            max_upload_bytes=settings.max_upload_bytes,
        )
        summary = register_downloaded_public_dataset(
            output_dir,
            workspace,
            expected_dataset_id=manifest.dataset_id,
            expected_manifest_sha256=manifest.sha256,
        )
    except (PublicDatasetRegistrationError, WorkspaceError, ValueError, OSError) as exc:
        raise DatasetDownloadError(f"workspace registration failed: {exc}") from exc

    print(
        "Registered "
        f"{summary.capture_count} published captures in {workspace_dir} "
        f"({summary.created_count} new, {summary.existing_count} already present; "
        f"{summary.on_camera_scramble_count} start solved and need an explicit "
        "frame-zero starting-state contract)."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = download_dataset(
            args.manifest,
            args.output_dir,
            base_url_override=args.base_url,
            revision_override=args.revision,
        )
        if args.workspace is not None:
            _register_workspace(manifest, args.output_dir, args.workspace)
    except (DatasetDownloadError, OSError) as error:
        print(f"dataset download: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
