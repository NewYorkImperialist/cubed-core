#!/usr/bin/env python3
"""Download and verify Cubed Core's GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPOSITORY = "KingBobJoeIV/cubed-core"
DEFAULT_RELEASE_TAG = "v1.0.0"
DEFAULT_OUTPUT_DIR = Path("workspace/release-assets")
CHECKSUM_FILENAME = "SHA256SUMS"
MAX_CHECKSUM_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
USER_AGENT = "cubed-core-release-assets/1"


class AssetDownloadError(RuntimeError):
    """Raised when an asset cannot be downloaded or verified safely."""


@dataclass(frozen=True)
class Asset:
    key: str
    filename: str
    archive_root: str | None
    description: str


@dataclass(frozen=True)
class _RedirectPolicy:
    allow_loopback_http: bool

    @classmethod
    def for_initial_url(cls, url: str) -> _RedirectPolicy:
        parsed = _validate_url_transport(url, allow_loopback_http=True)
        return cls(allow_loopback_http=parsed.scheme == "http")

    def validate(self, url: str) -> None:
        _validate_url_transport(url, allow_loopback_http=self.allow_loopback_http)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy: _RedirectPolicy) -> None:
        super().__init__()
        self._policy = policy

    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get("location") or headers.get("uri")
        if location is None:
            return None
        try:
            location.encode("ascii")
        except UnicodeEncodeError as error:
            fp.close()
            raise AssetDownloadError("redirect URL must be ASCII") from error
        if any(character.isspace() or ord(character) < 32 for character in location):
            fp.close()
            raise AssetDownloadError("redirect URL contains unsafe whitespace")

        target = urllib.parse.urljoin(req.full_url, location)
        try:
            self._policy.validate(target)
            redirected = super().redirect_request(req, fp, code, msg, headers, target)
            if redirected is None:
                fp.close()
                return None

            visited = getattr(req, "redirect_dict", {})
            if visited.get(target, 0) >= self.max_repeats or len(visited) >= self.max_redirections:
                raise urllib.error.HTTPError(
                    req.full_url,
                    code,
                    self.inf_msg + msg,
                    headers,
                    fp,
                )
            redirected.redirect_dict = visited
            visited[target] = visited.get(target, 0) + 1
        except Exception:
            fp.close()
            raise

        # Closing avoids urllib's default unbounded read of redirect bodies.
        fp.close()
        return self.parent.open(redirected, timeout=req.timeout)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


ASSETS = {
    "runtime": Asset(
        key="runtime",
        filename="camera-tracker-v1-runtime.tar.gz",
        archive_root="camera-tracker-v1-runtime",
        description="ONNX tracker runtime",
    ),
    "demo": Asset(
        key="demo",
        filename="gtD1s_decoder-demo-clip_f3919-f9722.mp4",
        archive_root=None,
        description="gtD1s 120 fps demo recording",
    ),
    "demo-license": Asset(
        key="demo-license",
        filename="gtD1s_decoder-demo-clip_f3919-f9722.LICENSE.txt",
        archive_root=None,
        description="gtD1s license and attribution notice",
    ),
    "decode-support": Asset(
        key="decode-support",
        filename="trust_v1_numpy.npz",
        archive_root=None,
        description="distilled numpy read-trust model (--trust-soft)",
    ),
    "decode-support-calibration": Asset(
        key="decode-support-calibration",
        filename="calibration_gan12.json",
        archive_root=None,
        description="gtD1s six-color demo calibration (--centroids-json)",
    ),
    "decode-support-license": Asset(
        key="decode-support-license",
        filename="decode-support.LICENSE.txt",
        archive_root=None,
        description="decode-support license notice",
    ),
}
REQUESTABLE_ASSETS = ("runtime", "demo", "decode-support")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Cubed Core release assets, verify SHA-256 checksums, and "
            "safely extract selected model archives. By default this downloads "
            "the ONNX runtime, gtD1s demo recording, and decode support."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--include",
        action="append",
        choices=REQUESTABLE_ASSETS,
        dest="includes",
        help=(
            "asset group to download; repeat as needed. "
            "Defaults to runtime, demo, and decode-support "
            "(read-trust model + demo calibration + license notice)."
        ),
    )
    release = parser.add_mutually_exclusive_group()
    release.add_argument(
        "--tag",
        help=f"release tag to download instead of {DEFAULT_RELEASE_TAG}",
    )
    release.add_argument(
        "--base-url",
        help=(
            "exact release-asset base URL, mainly for mirrors and local testing; "
            "must use HTTPS or loopback HTTP"
        ),
    )
    release.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "offline directory containing SHA256SUMS and the release assets; "
            "the same checksum and archive checks apply"
        ),
    )
    return parser


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_url_transport(
    value: str,
    *,
    allow_loopback_http: bool,
) -> urllib.parse.SplitResult:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AssetDownloadError("release URL contains control characters")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise AssetDownloadError(f"invalid release URL: {error}") from error
    if parsed.username or parsed.password:
        raise AssetDownloadError("release URL must not contain credentials")
    if parsed.scheme == "https" and hostname:
        return parsed
    if parsed.scheme == "http" and allow_loopback_http and _is_loopback_host(hostname):
        return parsed
    raise AssetDownloadError("release URL must use HTTPS or permitted loopback HTTP")


def _validate_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    parsed = _validate_url_transport(base_url, allow_loopback_http=True)
    if parsed.query or parsed.fragment:
        raise AssetDownloadError("release base URL must not contain a query or fragment")
    return base_url


def _release_base_url(tag: str | None, override: str | None) -> str:
    if override:
        return _validate_base_url(override)
    if tag:
        if tag in {".", ".."} or "/" in tag or "\\" in tag:
            raise AssetDownloadError("release tag must be one path segment")
        quoted_tag = urllib.parse.quote(tag, safe="")
        return f"https://github.com/{REPOSITORY}/releases/download/{quoted_tag}"
    return f"https://github.com/{REPOSITORY}/releases/download/{DEFAULT_RELEASE_TAG}"


def _asset_url(base_url: str, filename: str) -> str:
    return f"{base_url}/{urllib.parse.quote(filename, safe='')}"


def _open_url(url: str):
    policy = _RedirectPolicy.for_initial_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": USER_AGENT,
        },
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler(policy))
    try:
        response = opener.open(request, timeout=60)
    except AssetDownloadError:
        raise
    except (OSError, urllib.error.URLError) as error:
        message = f"could not download {url}: {error}"
        if (
            isinstance(error, urllib.error.HTTPError)
            and error.code == 404
            and url.startswith(f"https://github.com/{REPOSITORY}/releases/")
        ):
            message += (
                f"\n  Release assets were not found for {REPOSITORY}."
                "\n  The workbench can still open, but live Decode needs its assets."
                "\n  Check RELEASE_TAG, or use a trusted mirror:"
                "\n    make download-assets RELEASE_ASSET_BASE_URL=<mirror-url>"
            )
        raise AssetDownloadError(message) from error

    final_url = response.geturl()
    try:
        policy.validate(final_url)
    except AssetDownloadError:
        response.close()
        raise AssetDownloadError(f"download redirected to an unsafe URL: {final_url}") from None
    return response


class _LocalAssetResponse:
    """Adapts a local file to the minimal response interface `_open_url` returns.

    This lets `_read_url` and `_download_to_path` reuse their existing size
    checks, hashing, and streaming unchanged when the source is a local
    directory (`--source-dir`) instead of an HTTP(S) release.
    """

    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self.headers = {"Content-Length": str(path.stat().st_size)}

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def __enter__(self) -> _LocalAssetResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._handle.close()

    def close(self) -> None:
        self._handle.close()


def _open_local_asset(source_dir: Path, name: str) -> _LocalAssetResponse:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise AssetDownloadError(f"unsafe local asset name: {name!r}")
    path = source_dir / name
    try:
        if path.is_symlink():
            raise AssetDownloadError(f"refusing to read symlinked local asset: {path}")
        if not path.is_file():
            raise AssetDownloadError(f"local release asset not found: {path}")
        return _LocalAssetResponse(path)
    except OSError as error:
        raise AssetDownloadError(f"could not read local asset {path}: {error}") from error


def _read_url(
    url: str,
    *,
    limit: int,
    opener: Callable[[str], object] = _open_url,
) -> bytes:
    with opener(url) as response:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise AssetDownloadError(f"invalid Content-Length for {url}") from error
            if declared_size < 0 or declared_size > limit:
                raise AssetDownloadError(f"download is larger than the {limit}-byte limit: {url}")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise AssetDownloadError(f"download is larger than the {limit}-byte limit: {url}")
    return payload


def _download_to_path(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    opener: Callable[[str], object] = _open_url,
) -> int:
    digest = hashlib.sha256()
    total = 0
    with opener(url) as response, destination.open("xb") as output:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise AssetDownloadError(f"invalid Content-Length for {url}") from error
            if declared_size < 0 or declared_size > MAX_ASSET_BYTES:
                raise AssetDownloadError(
                    f"asset is larger than the {MAX_ASSET_BYTES}-byte limit: {url}"
                )

        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ASSET_BYTES:
                raise AssetDownloadError(
                    f"asset is larger than the {MAX_ASSET_BYTES}-byte limit: {url}"
                )
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise AssetDownloadError(
            f"checksum mismatch for {destination.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return total


def _parse_checksums(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AssetDownloadError("SHA256SUMS is not valid UTF-8") from error

    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        parts = raw_line.split("  ", 1)
        if len(parts) != 2:
            raise AssetDownloadError(f"invalid SHA256SUMS line {line_number}")
        checksum, filename = parts
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise AssetDownloadError(f"invalid SHA-256 on SHA256SUMS line {line_number}")
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise AssetDownloadError(f"unsafe filename on SHA256SUMS line {line_number}")
        if filename in checksums:
            raise AssetDownloadError(f"duplicate SHA256SUMS entry: {filename}")
        checksums[filename] = checksum
    if not checksums:
        raise AssetDownloadError("SHA256SUMS is empty")
    return checksums


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise AssetDownloadError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AssetDownloadError(f"unsafe archive path: {name!r}")
    return path


def _safe_extract_tar(archive: Path, staging_dir: Path, expected_root: str) -> Path:
    extraction_dir = staging_dir / f".extract-{expected_root}"
    extraction_dir.mkdir(mode=0o700)
    seen: set[PurePosixPath] = set()
    declared_bytes = 0

    try:
        with tarfile.open(archive, mode="r:gz") as source:
            checked: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
            for member_number, member in enumerate(source, start=1):
                if member_number > MAX_ARCHIVE_MEMBERS:
                    raise AssetDownloadError(
                        f"{archive.name} has more than {MAX_ARCHIVE_MEMBERS} members"
                    )
                path = _safe_member_path(member.name)
                if path in seen:
                    raise AssetDownloadError(f"duplicate archive path: {member.name}")
                seen.add(path)
                if path.parts[0] != expected_root:
                    raise AssetDownloadError(
                        f"{archive.name} must contain only the {expected_root} root"
                    )
                if not (member.isdir() or member.isreg()):
                    raise AssetDownloadError(
                        f"{archive.name} contains unsupported entry: {member.name}"
                    )
                if member.size < 0:
                    raise AssetDownloadError(f"invalid archive member size: {member.name}")
                if member.isreg():
                    declared_bytes += member.size
                    if declared_bytes > MAX_EXTRACTED_BYTES:
                        raise AssetDownloadError(
                            f"{archive.name} expands beyond the {MAX_EXTRACTED_BYTES}-byte limit"
                        )
                checked.append((member, path))

            for member, path in checked:
                destination = extraction_dir.joinpath(*path.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True, mode=0o755)
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source_file = source.extractfile(member)
                if source_file is None:
                    raise AssetDownloadError(f"could not read archive member: {member.name}")
                copied = 0
                with source_file, destination.open("xb") as output:
                    while True:
                        chunk = source_file.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > member.size:
                            raise AssetDownloadError(
                                f"archive member exceeds its declared size: {member.name}"
                            )
                        output.write(chunk)
                if copied != member.size:
                    raise AssetDownloadError(f"archive member size mismatch: {member.name}")
                destination.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        extracted_root = extraction_dir / expected_root
        if not extracted_root.is_dir():
            raise AssetDownloadError(f"{archive.name} does not contain {expected_root}")
        return extracted_root
    except (OSError, tarfile.TarError) as error:
        raise AssetDownloadError(f"could not extract {archive.name}: {error}") from error


def _publish_file_no_clobber(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise AssetDownloadError(f"refusing to overwrite existing path: {destination}") from error
    except OSError as error:
        raise AssetDownloadError(f"could not publish {destination}: {error}") from error


def _publish_tree_no_clobber(source: Path, destination: Path) -> None:
    try:
        destination.mkdir(mode=stat.S_IMODE(source.stat().st_mode))
    except FileExistsError as error:
        raise AssetDownloadError(f"refusing to overwrite existing path: {destination}") from error
    except OSError as error:
        raise AssetDownloadError(f"could not publish {destination}: {error}") from error

    try:
        for source_path in sorted(
            source.rglob("*"),
            key=lambda path: (len(path.relative_to(source).parts), str(path)),
        ):
            relative = source_path.relative_to(source)
            destination_path = destination.joinpath(*relative.parts)
            if source_path.is_symlink():
                raise AssetDownloadError(
                    f"refusing to publish symlink from extracted archive: {relative}"
                )
            if source_path.is_dir():
                destination_path.mkdir(mode=stat.S_IMODE(source_path.stat().st_mode))
            elif source_path.is_file():
                _publish_file_no_clobber(source_path, destination_path)
            else:
                raise AssetDownloadError(
                    f"refusing to publish unsupported extracted path: {relative}"
                )
    except (AssetDownloadError, OSError) as error:
        try:
            shutil.rmtree(destination)
        except OSError as cleanup_error:
            raise AssetDownloadError(
                f"could not clean partial publication {destination}: {cleanup_error}"
            ) from error
        if isinstance(error, AssetDownloadError):
            raise
        raise AssetDownloadError(f"could not publish {destination}: {error}") from error


def _rollback_published(paths: list[Path]) -> list[Path]:
    failed: list[Path] = []
    for path in reversed(paths):
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            failed.append(path)
    return failed


def _selected_assets(includes: list[str] | None) -> list[Asset]:
    keys = includes or ["runtime", "demo", "decode-support"]
    ordered_keys: list[str] = []
    for key in keys:
        if key not in ordered_keys:
            ordered_keys.append(key)
        if key == "demo" and "demo-license" not in ordered_keys:
            ordered_keys.append("demo-license")
        if key == "decode-support" and "decode-support-calibration" not in ordered_keys:
            ordered_keys.append("decode-support-calibration")
        if key == "decode-support" and "decode-support-license" not in ordered_keys:
            ordered_keys.append("decode-support-license")
    return [ASSETS[key] for key in ordered_keys]


def download_release_assets(
    *,
    output_dir: Path,
    assets: list[Asset],
    base_url: str | None = None,
    source_dir: Path | None = None,
) -> list[Path]:
    if (base_url is None) == (source_dir is None):
        raise AssetDownloadError("exactly one of base_url or source_dir must be provided")

    if source_dir is not None:
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise AssetDownloadError(f"source directory is not a normal directory: {source_dir}")

    output_dir = output_dir.expanduser().resolve()
    checksum_path = output_dir / CHECKSUM_FILENAME
    final_paths: list[Path] = []
    for asset in assets:
        final_paths.append(output_dir / asset.filename)
        if asset.archive_root:
            final_paths.append(output_dir / asset.archive_root)

    collisions = [path for path in final_paths if path.exists() or path.is_symlink()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise AssetDownloadError(f"refusing to overwrite existing path(s): {rendered}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise AssetDownloadError(f"output directory is not a normal directory: {output_dir}")

    existing_checksum_payload: bytes | None = None
    if checksum_path.exists() or checksum_path.is_symlink():
        if checksum_path.is_symlink() or not checksum_path.is_file():
            raise AssetDownloadError(
                f"existing checksum path is not a normal file: {checksum_path}"
            )
        if checksum_path.stat().st_size > MAX_CHECKSUM_BYTES:
            raise AssetDownloadError(f"existing checksum file is too large: {checksum_path}")
        existing_checksum_payload = checksum_path.read_bytes()

    if source_dir is not None:
        checksum_payload = _read_url(
            CHECKSUM_FILENAME,
            limit=MAX_CHECKSUM_BYTES,
            opener=lambda name: _open_local_asset(source_dir, name),
        )
    else:
        checksum_payload = _read_url(
            _asset_url(base_url, CHECKSUM_FILENAME),
            limit=MAX_CHECKSUM_BYTES,
        )
    if existing_checksum_payload is not None and existing_checksum_payload != checksum_payload:
        raise AssetDownloadError(
            "existing SHA256SUMS differs from the selected release; use a new output directory"
        )
    checksums = _parse_checksums(checksum_payload)
    missing = [asset.filename for asset in assets if asset.filename not in checksums]
    if missing:
        raise AssetDownloadError("SHA256SUMS is missing requested asset(s): " + ", ".join(missing))

    staged_paths: dict[str, Path] = {}
    extracted_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=".cubed-core-assets-", dir=output_dir) as raw_staging:
        staging_dir = Path(raw_staging)
        staged_checksum: Path | None = None
        if existing_checksum_payload is None:
            staged_checksum = staging_dir / CHECKSUM_FILENAME
            staged_checksum.write_bytes(checksum_payload)
            staged_checksum.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        for asset in assets:
            staged_asset = staging_dir / asset.filename
            print(f"Downloading {asset.description}: {asset.filename}", flush=True)
            if source_dir is not None:
                _download_to_path(
                    asset.filename,
                    staged_asset,
                    checksums[asset.filename],
                    opener=lambda name: _open_local_asset(source_dir, name),
                )
            else:
                _download_to_path(
                    _asset_url(base_url, asset.filename),
                    staged_asset,
                    checksums[asset.filename],
                )
            staged_paths[asset.key] = staged_asset
            if asset.archive_root:
                extracted_paths[asset.key] = _safe_extract_tar(
                    staged_asset,
                    staging_dir,
                    asset.archive_root,
                )

        created: list[Path] = []
        published: list[Path] = [checksum_path]
        try:
            for asset in assets:
                final_asset = output_dir / asset.filename
                _publish_file_no_clobber(staged_paths[asset.key], final_asset)
                created.append(final_asset)
                published.append(final_asset)
                if asset.archive_root:
                    final_root = output_dir / asset.archive_root
                    _publish_tree_no_clobber(extracted_paths[asset.key], final_root)
                    created.append(final_root)
                    published.append(final_root)
            if staged_checksum is not None:
                _publish_file_no_clobber(staged_checksum, checksum_path)
                created.append(checksum_path)
        except AssetDownloadError as error:
            cleanup_failures = _rollback_published(created)
            if cleanup_failures:
                rendered = ", ".join(str(path) for path in cleanup_failures)
                raise AssetDownloadError(
                    f"{error}; could not clean partial publication: {rendered}"
                ) from error
            raise
        return published


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assets = _selected_assets(args.includes)
        if args.source_dir is not None:
            published = download_release_assets(
                output_dir=args.output_dir,
                assets=assets,
                source_dir=args.source_dir,
            )
        else:
            base_url = _release_base_url(args.tag, args.base_url)
            published = download_release_assets(
                output_dir=args.output_dir,
                assets=assets,
                base_url=base_url,
            )
    except AssetDownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("\nDownloaded and verified:", flush=True)
    for path in published:
        print(f"  {path}", flush=True)
    if any(asset.key == "runtime" for asset in assets):
        manifest = (
            args.output_dir.expanduser().resolve()
            / ASSETS["runtime"].archive_root
            / "manifest.json"
        )
        print("\nUse the runtime model with:", flush=True)
        print(f'  export CUBED_CORE_TRACKER_MODEL_MANIFEST="{manifest}"', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
