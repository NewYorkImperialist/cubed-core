from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import tarfile
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts import download_release_assets

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _UnreadRedirectBody:
    def __init__(self) -> None:
        self.closed = False

    def read(self, *args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("redirect bodies must not be drained")

    def close(self) -> None:
        self.closed = True


class _RedirectParent:
    def __init__(self) -> None:
        self.request: urllib.request.Request | None = None
        self.response = object()

    def open(self, request: urllib.request.Request, *, timeout: int):
        del timeout
        self.request = request
        return self.response


def test_makefile_exposes_only_public_release_download_targets() -> None:
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "\ndownload-assets:" in makefile
    assert "\ndownload-decode-support:" in makefile
    assert "download-assets-private" not in makefile
    assert "download-decode-support-private" not in makefile
    assert "gh-auth-check" not in makefile


@contextlib.contextmanager
def _serve(directory: Path):
    handler = partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive(path: Path, root: str, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        root_info = tarfile.TarInfo(f"{root}/")
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        archive.addfile(root_info)
        for relative, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))


def _write_checksums(directory: Path, filenames: list[str]) -> None:
    records = [f"{_sha256(directory / filename)}  {filename}\n" for filename in filenames]
    (directory / "SHA256SUMS").write_text("".join(records), encoding="utf-8")


def test_default_download_gets_runtime_demo_and_decode_support(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    runtime_name = download_release_assets.ASSETS["runtime"].filename
    demo_name = download_release_assets.ASSETS["demo"].filename
    demo_license_name = download_release_assets.ASSETS["demo-license"].filename
    trust_name = download_release_assets.ASSETS["decode-support"].filename
    calibration_name = download_release_assets.ASSETS["decode-support-calibration"].filename
    support_license_name = download_release_assets.ASSETS["decode-support-license"].filename
    _write_archive(
        release / runtime_name,
        "camera-tracker-v1-runtime",
        {
            "manifest.json": b'{"schema":"cubed-core/model-artifact-manifest-v1"}\n',
            "LICENSE": b"AGPL fixture\n",
        },
    )
    (release / demo_name).write_bytes(b"small video fixture")
    (release / demo_license_name).write_text(
        "CC BY-SA 4.0 fixture\n",
        encoding="utf-8",
    )
    (release / trust_name).write_bytes(b"PK\x03\x04 fake-npz-zip trust model bytes")
    (release / calibration_name).write_text('{"red":[1,2,3]}\n', encoding="utf-8")
    (release / support_license_name).write_text("AGPL fixture\n", encoding="utf-8")
    _write_checksums(
        release,
        [
            runtime_name,
            demo_name,
            demo_license_name,
            trust_name,
            calibration_name,
            support_license_name,
        ],
    )
    output = tmp_path / "output"

    with _serve(release) as base_url:
        result = download_release_assets.main(["--base-url", base_url, "--output-dir", str(output)])

    assert result == 0
    assert (output / "SHA256SUMS").is_file()
    assert (output / runtime_name).is_file()
    assert (output / demo_name).read_bytes() == b"small video fixture"
    assert (output / demo_license_name).read_text(encoding="utf-8") == ("CC BY-SA 4.0 fixture\n")
    assert (output / trust_name).is_file()
    assert (output / calibration_name).is_file()
    assert (output / support_license_name).read_text(encoding="utf-8") == "AGPL fixture\n"
    assert (
        output / "camera-tracker-v1-runtime" / "manifest.json"
    ).read_bytes() == b'{"schema":"cubed-core/model-artifact-manifest-v1"}\n'


def test_decode_support_pulls_trust_model_calibration_and_license(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    trust_name = download_release_assets.ASSETS["decode-support"].filename
    calib_name = download_release_assets.ASSETS["decode-support-calibration"].filename
    license_name = download_release_assets.ASSETS["decode-support-license"].filename
    trust_bytes = b"PK\x03\x04 fake-npz-zip trust model bytes"
    calib_bytes = b'{"red":[1,2,3],"blue":[4,5,6]}\n'
    license_bytes = b"AGPL fixture\n"
    (release / trust_name).write_bytes(trust_bytes)
    (release / calib_name).write_bytes(calib_bytes)
    (release / license_name).write_bytes(license_bytes)
    _write_checksums(release, [trust_name, calib_name, license_name])
    output = tmp_path / "output"

    with _serve(release) as base_url:
        result = download_release_assets.main(
            [
                "--base-url",
                base_url,
                "--output-dir",
                str(output),
                "--include",
                "decode-support",
            ]
        )

    assert result == 0
    assert (output / trust_name).read_bytes() == trust_bytes
    assert (output / calib_name).read_bytes() == calib_bytes
    assert (output / license_name).read_bytes() == license_bytes
    # An explicitly selected group does not pull unrelated groups.
    assert not (output / download_release_assets.ASSETS["runtime"].filename).exists()
    assert not (output / download_release_assets.ASSETS["demo"].filename).exists()


def test_default_download_includes_all_required_asset_groups() -> None:
    default_keys = {asset.key for asset in download_release_assets._selected_assets(None)}
    assert default_keys == {
        "runtime",
        "demo",
        "demo-license",
        "decode-support",
        "decode-support-calibration",
        "decode-support-license",
    }


def test_default_release_url_is_pinned_to_v100() -> None:
    assert download_release_assets._release_base_url(None, None) == (
        "https://github.com/KingBobJoeIV/cubed-core/releases/download/v1.0.0"
    )


def test_checksum_mismatch_leaves_no_published_asset(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    demo_license_name = download_release_assets.ASSETS["demo-license"].filename
    (release / demo_name).write_bytes(b"actual bytes")
    (release / demo_license_name).write_text("license\n", encoding="utf-8")
    (release / "SHA256SUMS").write_text(
        (f"{'0' * 64}  {demo_name}\n{_sha256(release / demo_license_name)}  {demo_license_name}\n"),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with _serve(release) as base_url:
        result = download_release_assets.main(
            [
                "--base-url",
                base_url,
                "--output-dir",
                str(output),
                "--include",
                "demo",
            ]
        )

    assert result == 2
    assert not (output / "SHA256SUMS").exists()
    assert not (output / demo_name).exists()
    assert list(output.iterdir()) == []


def test_existing_destination_refuses_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    (output / demo_name).write_bytes(b"existing")

    def unexpected_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("network must not be used after an overwrite collision")

    monkeypatch.setattr(download_release_assets, "_read_url", unexpected_read)
    with pytest.raises(
        download_release_assets.AssetDownloadError,
        match="refusing to overwrite",
    ):
        download_release_assets.download_release_assets(
            output_dir=output,
            assets=[download_release_assets.ASSETS["demo"]],
            base_url="https://example.invalid/release",
        )

    assert (output / demo_name).read_bytes() == b"existing"


def test_matching_checksum_file_can_be_reused_for_another_asset_group(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    demo_license_name = download_release_assets.ASSETS["demo-license"].filename
    runtime_name = download_release_assets.ASSETS["runtime"].filename
    (release / demo_name).write_bytes(b"demo")
    (release / demo_license_name).write_text("license\n", encoding="utf-8")
    _write_archive(
        release / runtime_name,
        "camera-tracker-v1-runtime",
        {"manifest.json": b'{"schema":"cubed-core/model-artifact-manifest-v1"}\n'},
    )
    _write_checksums(release, [demo_name, demo_license_name, runtime_name])
    output = tmp_path / "output"

    with _serve(release) as base_url:
        assert (
            download_release_assets.main(
                [
                    "--base-url",
                    base_url,
                    "--output-dir",
                    str(output),
                    "--include",
                    "demo",
                ]
            )
            == 0
        )
        checksum_before = (output / "SHA256SUMS").read_bytes()
        assert (
            download_release_assets.main(
                [
                    "--base-url",
                    base_url,
                    "--output-dir",
                    str(output),
                    "--include",
                    "runtime",
                ]
            )
            == 0
        )

    assert (output / "SHA256SUMS").read_bytes() == checksum_before
    assert (output / "camera-tracker-v1-runtime" / "manifest.json").is_file()


@pytest.mark.parametrize("entry_type", ["traversal", "symlink"])
def test_unsafe_archives_are_rejected_without_publication(
    tmp_path: Path,
    entry_type: str,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    runtime_name = download_release_assets.ASSETS["runtime"].filename
    archive_path = release / runtime_name
    with tarfile.open(archive_path, mode="w:gz") as archive:
        if entry_type == "traversal":
            info = tarfile.TarInfo("../escape")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"nope"))
        else:
            info = tarfile.TarInfo("camera-tracker-v1-runtime/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/escape"
            archive.addfile(info)
    _write_checksums(release, [runtime_name])
    output = tmp_path / "output"

    with _serve(release) as base_url:
        result = download_release_assets.main(
            [
                "--base-url",
                base_url,
                "--output-dir",
                str(output),
                "--include",
                "runtime",
            ]
        )

    assert result == 2
    assert not (output / runtime_name).exists()
    assert not (output / "camera-tracker-v1-runtime").exists()
    assert not (output / "SHA256SUMS").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"not-a-checksum\n", "invalid SHA256SUMS line"),
        (f"{'0' * 64}  ../asset\n".encode(), "unsafe filename"),
        (
            (f"{'0' * 64}  asset\n{'1' * 64}  asset\n").encode(),
            "duplicate SHA256SUMS entry",
        ),
    ],
)
def test_checksum_manifest_rejects_malformed_or_unsafe_rows(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(download_release_assets.AssetDownloadError, match=message):
        download_release_assets._parse_checksums(payload)


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/release",
        "file:///tmp/release",
        "https://user:secret@example.com/release",
        "https://example.com/release?token=secret",
    ],
)
def test_base_url_rejects_unsafe_transport_or_embedded_credentials(
    value: str,
) -> None:
    with pytest.raises(download_release_assets.AssetDownloadError):
        download_release_assets._validate_base_url(value)


def test_redirect_handler_validates_each_hop_without_reading_response_body() -> None:
    policy = download_release_assets._RedirectPolicy.for_initial_url("http://127.0.0.1/start")
    handler = download_release_assets._SafeRedirectHandler(policy)
    parent = _RedirectParent()
    handler.add_parent(parent)
    request = urllib.request.Request("http://127.0.0.1/start")
    request.timeout = 5
    body = _UnreadRedirectBody()

    response = handler.http_error_302(
        request,
        body,
        302,
        "Found",
        {"location": "/next"},
    )

    assert response is parent.response
    assert parent.request is not None
    assert parent.request.full_url == "http://127.0.0.1/next"
    assert body.closed


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/private",
        "ftp://example.com/asset",
    ],
)
def test_https_redirect_rejects_downgrades_and_unsafe_schemes(target: str) -> None:
    policy = download_release_assets._RedirectPolicy.for_initial_url("https://github.com/release")
    handler = download_release_assets._SafeRedirectHandler(policy)
    request = urllib.request.Request("https://github.com/release")
    request.timeout = 5
    body = _UnreadRedirectBody()

    with pytest.raises(download_release_assets.AssetDownloadError):
        handler.http_error_302(
            request,
            body,
            302,
            "Found",
            {"location": target},
        )

    assert body.closed


def test_archive_expansion_limit_is_checked_before_payload_is_read(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "oversized.tar.gz"
    info = tarfile.TarInfo("camera-tracker-v1-runtime/huge.bin")
    info.size = download_release_assets.MAX_EXTRACTED_BYTES + 1
    with gzip.open(archive_path, mode="wb") as archive:
        archive.write(info.tobuf())
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(download_release_assets.AssetDownloadError, match="expands beyond"):
        download_release_assets._safe_extract_tar(
            archive_path,
            staging,
            "camera-tracker-v1-runtime",
        )


def test_archive_member_limit_is_checked_during_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "members.tar.gz"
    _write_archive(
        archive_path,
        "camera-tracker-v1-runtime",
        {"first": b"", "second": b""},
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(download_release_assets, "MAX_ARCHIVE_MEMBERS", 1)

    with pytest.raises(download_release_assets.AssetDownloadError, match="more than 1 member"):
        download_release_assets._safe_extract_tar(
            archive_path,
            staging,
            "camera-tracker-v1-runtime",
        )


def test_path_created_during_download_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    asset = download_release_assets.ASSETS["demo"]
    payload = b"release bytes"
    manifest = f"{hashlib.sha256(payload).hexdigest()}  {asset.filename}\n".encode()
    victim = output / asset.filename

    monkeypatch.setattr(
        download_release_assets,
        "_read_url",
        lambda *args, **kwargs: manifest,
    )

    def fake_download(url: str, destination: Path, expected_sha256: str) -> int:
        del url, expected_sha256
        destination.write_bytes(payload)
        victim.write_bytes(b"created while the download was running")
        return len(payload)

    monkeypatch.setattr(download_release_assets, "_download_to_path", fake_download)

    with pytest.raises(download_release_assets.AssetDownloadError, match="refusing to overwrite"):
        download_release_assets.download_release_assets(
            output_dir=output,
            assets=[asset],
            base_url="https://example.invalid/release",
        )

    assert victim.read_bytes() == b"created while the download was running"
    assert not (output / "SHA256SUMS").exists()


@pytest.mark.parametrize("failure_name", ["second.bin", "SHA256SUMS"])
def test_publication_failure_rolls_back_every_created_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_name: str,
) -> None:
    output = tmp_path / "output"
    assets = [
        download_release_assets.Asset("first", "first.bin", None, "first"),
        download_release_assets.Asset("second", "second.bin", None, "second"),
    ]
    payloads = {"first.bin": b"first", "second.bin": b"second"}
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {filename}\n"
        for filename, payload in payloads.items()
    ).encode()
    monkeypatch.setattr(
        download_release_assets,
        "_read_url",
        lambda *args, **kwargs: manifest,
    )

    def fake_download(url: str, destination: Path, expected_sha256: str) -> int:
        del url, expected_sha256
        payload = payloads[destination.name]
        destination.write_bytes(payload)
        return len(payload)

    monkeypatch.setattr(download_release_assets, "_download_to_path", fake_download)
    original_link = download_release_assets.os.link

    def controlled_link(source: Path, destination: Path) -> None:
        if destination.name == failure_name:
            raise PermissionError("simulated publication failure")
        original_link(source, destination)

    monkeypatch.setattr(download_release_assets.os, "link", controlled_link)

    with pytest.raises(download_release_assets.AssetDownloadError, match="could not publish"):
        download_release_assets.download_release_assets(
            output_dir=output,
            assets=assets,
            base_url="https://example.invalid/release",
        )

    assert list(output.iterdir()) == []


def test_github_404_explains_missing_assets_and_public_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = download_release_assets._release_base_url(None, None) + "/SHA256SUMS"

    class _NotFoundOpener:
        def open(self, request: urllib.request.Request, *, timeout: int):
            del request, timeout
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(
        download_release_assets.urllib.request,
        "build_opener",
        lambda *handlers: _NotFoundOpener(),
    )

    with pytest.raises(download_release_assets.AssetDownloadError) as excinfo:
        download_release_assets._open_url(url)

    message = str(excinfo.value)
    assert "HTTP Error 404" in message
    assert "Release assets were not found" in message
    assert "Check RELEASE_TAG" in message
    assert "RELEASE_ASSET_BASE_URL=<mirror-url>" in message
    assert "private" not in message
    assert "gh auth" not in message


def test_source_dir_publishes_and_verifies_assets_without_network(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gh-release-download"
    source.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    demo_license_name = download_release_assets.ASSETS["demo-license"].filename
    (source / demo_name).write_bytes(b"small video fixture")
    (source / demo_license_name).write_text(
        "CC BY-SA 4.0 fixture\n",
        encoding="utf-8",
    )
    _write_checksums(source, [demo_name, demo_license_name])
    output = tmp_path / "output"

    result = download_release_assets.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--include",
            "demo",
        ]
    )

    assert result == 0
    assert (output / "SHA256SUMS").is_file()
    assert (output / demo_name).read_bytes() == b"small video fixture"
    assert (output / demo_license_name).read_text(encoding="utf-8") == ("CC BY-SA 4.0 fixture\n")


def test_source_dir_rejects_checksum_mismatch_without_publishing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gh-release-download"
    source.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    demo_license_name = download_release_assets.ASSETS["demo-license"].filename
    (source / demo_name).write_bytes(b"actual bytes")
    (source / demo_license_name).write_text("license\n", encoding="utf-8")
    (source / "SHA256SUMS").write_text(
        (f"{'0' * 64}  {demo_name}\n{_sha256(source / demo_license_name)}  {demo_license_name}\n"),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = download_release_assets.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--include",
            "demo",
        ]
    )

    assert result == 2
    assert not (output / "SHA256SUMS").exists()
    assert not (output / demo_name).exists()
    assert list(output.iterdir()) == []


def test_source_dir_existing_destination_refuses_before_reading_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gh-release-download"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    demo_name = download_release_assets.ASSETS["demo"].filename
    (output / demo_name).write_bytes(b"existing")

    with pytest.raises(
        download_release_assets.AssetDownloadError,
        match="refusing to overwrite",
    ):
        download_release_assets.download_release_assets(
            output_dir=output,
            assets=[download_release_assets.ASSETS["demo"]],
            source_dir=source,
        )

    # The collision is caught before the source directory is ever read.
    assert not source.exists() or list(source.iterdir()) == []
    assert (output / demo_name).read_bytes() == b"existing"


def test_non_github_404_keeps_the_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.invalid/release/SHA256SUMS"

    class _NotFoundOpener:
        def open(self, request: urllib.request.Request, *, timeout: int):
            del request, timeout
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(
        download_release_assets.urllib.request,
        "build_opener",
        lambda *handlers: _NotFoundOpener(),
    )

    with pytest.raises(download_release_assets.AssetDownloadError) as excinfo:
        download_release_assets._open_url(url)

    message = str(excinfo.value)
    assert "HTTP Error 404" in message
    assert "RELEASE_ASSET_BASE_URL" not in message
