from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator


def _load_download_dataset_module() -> ModuleType:
    scripts_directory = Path(__file__).resolve().parents[1]
    script_path = scripts_directory / "download_dataset.py"
    module_name = "cubed_core_download_dataset_test_target"
    sys.path.insert(0, str(scripts_directory))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load downloader test target: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_directory))


download_dataset = _load_download_dataset_module()

DIRECT_REVISION = "direct-revision-0001"
HF_REVISION = "a" * 40
REQUIRED_PAYLOADS = {
    # These exercise transport and bootstrap hashing only; the downloader's
    # trust anchor is the checked-in inventory, not corpus semantics.
    ".gitattributes": b"*.mp4 filter=lfs diff=lfs merge=lfs -text\n",
    "README.md": b"# Synthetic public dataset\n",
    "LICENSES/DATASET.md": b"Synthetic dataset license fixture\n",
    "dataset/manifest.json": b'{"schema":"cubed-core/public-corpus-manifest"}\n',
    "metadata.jsonl": b'{"capture_id":"00000000000000000000000000000000"}\n',
    "readiness-report.json": b'{"ready":true}\n',
    "SHA256SUMS": b"synthetic checksums fixture\n",
}
BENCHMARK_PAYLOADS = {
    "public-inputs/manifest.json": b'{"schema":"synthetic-benchmark-manifest"}\n',
    "public-inputs/splits.json": b'{"train":[],"validation":[],"test":[]}\n',
    "teacher/manifest.json": b'{"schema":"synthetic-teacher-manifest"}\n',
}


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payloads(**extra: bytes) -> dict[str, bytes]:
    return {**REQUIRED_PAYLOADS, **extra}


def _artifacts(
    payloads: dict[str, bytes],
    *,
    wrong_sha256_path: str | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": "0" * 64 if path == wrong_sha256_path else _sha256(payload),
        }
        for path, payload in payloads.items()
    ]


def _publish_files(root: Path, payloads: dict[str, bytes]) -> None:
    for relative, payload in payloads.items():
        destination = root.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _manifest_document(
    *,
    status: str,
    revision: str | None = None,
    base_url_template: str | None = None,
    source_kind: str = "direct-mirror",
    artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    source = (
        None
        if base_url_template is None
        else {
            "kind": source_kind,
            "base_url_template": base_url_template,
        }
    )
    return {
        "schema": download_dataset.SCHEMA,
        "dataset_id": "cubed-core-test-dataset",
        "description": "Synthetic downloader test fixture.",
        "publication_status": status,
        "revision": revision,
        "source": source,
        "artifacts": artifacts or [],
    }


def _write_manifest(path: Path, **kwargs: object) -> dict[str, object]:
    document = _manifest_document(**kwargs)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    return document


def _schema_validator() -> Draft202012Validator:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "schemas/public-dataset-download-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _download_arguments(manifest: Path, output: Path) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output),
    ]


def test_checked_in_published_manifest_matches_schema_and_runtime() -> None:
    repository = Path(__file__).resolve().parents[2]
    manifest_path = repository / "config/public-dataset-v1.json"
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))

    _schema_validator().validate(manifest_document)
    manifest = download_dataset.load_manifest(manifest_path)
    assert manifest.publication_status == "published"
    assert manifest.source_kind == "hugging-face"
    assert manifest.revision is not None
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.revision)
    assert manifest.base_url_template is not None
    artifact_paths = {artifact.path for artifact in manifest.artifacts}
    assert download_dataset.REQUIRED_PUBLISHED_ARTIFACTS <= artifact_paths
    assert not download_dataset.BENCHMARK_V0_ARTIFACTS & artifact_paths


def test_pending_manifest_fails_closed_even_with_source_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, status="pending-rights-review")
    output = tmp_path / "dataset"

    result = download_dataset.main(
        [
            *_download_arguments(manifest, output),
            "--base-url",
            "https://example.invalid/public/{revision}",
            "--revision",
            DIRECT_REVISION,
        ]
    )

    assert result == 2
    assert not output.exists()
    assert "is not published" in capsys.readouterr().err


def test_published_manifest_matches_schema_and_runtime(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    document = _write_manifest(
        manifest_path,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=_artifacts(_payloads()),
    )

    _schema_validator().validate(document)
    manifest = download_dataset.load_manifest(manifest_path)
    assert manifest.publication_status == "published"
    assert {artifact.path for artifact in manifest.artifacts} == (
        download_dataset.REQUIRED_PUBLISHED_ARTIFACTS
    )


@pytest.mark.parametrize(
    "missing_path",
    sorted(path.as_posix() for path in download_dataset.REQUIRED_PUBLISHED_ARTIFACTS),
)
def test_published_manifest_requires_every_dataset_contract_file(
    tmp_path: Path,
    missing_path: str,
) -> None:
    payloads = _payloads()
    del payloads[missing_path]
    manifest_path = tmp_path / "manifest.json"
    document = _write_manifest(
        manifest_path,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=_artifacts(payloads),
    )

    assert list(_schema_validator().iter_errors(document))
    with pytest.raises(
        download_dataset.DatasetDownloadError,
        match="missing required dataset contract files",
    ):
        download_dataset.load_manifest(manifest_path)


def test_published_manifest_allows_no_benchmark_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    document = _write_manifest(
        manifest_path,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=_artifacts(_payloads()),
    )

    _schema_validator().validate(document)
    assert download_dataset.load_manifest(manifest_path).publication_status == "published"


@pytest.mark.parametrize("included_path", sorted(BENCHMARK_PAYLOADS))
def test_published_manifest_rejects_partial_benchmark_contract(
    tmp_path: Path,
    included_path: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    document = _write_manifest(
        manifest_path,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=_artifacts(_payloads(**{included_path: BENCHMARK_PAYLOADS[included_path]})),
    )

    assert list(_schema_validator().iter_errors(document))
    with pytest.raises(
        download_dataset.DatasetDownloadError,
        match="partial Benchmark v0 contract",
    ):
        download_dataset.load_manifest(manifest_path)


def test_published_manifest_accepts_complete_benchmark_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    document = _write_manifest(
        manifest_path,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=_artifacts(_payloads(**BENCHMARK_PAYLOADS)),
    )

    _schema_validator().validate(document)
    assert download_dataset.load_manifest(manifest_path).publication_status == "published"


def test_hugging_face_requires_a_full_commit_revision(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    invalid_document = _write_manifest(
        manifest_path,
        status="published",
        revision="main-revision",
        source_kind="hugging-face",
        base_url_template="https://huggingface.co/datasets/example/data/resolve/{revision}",
        artifacts=_artifacts(_payloads()),
    )

    assert list(_schema_validator().iter_errors(invalid_document))
    with pytest.raises(
        download_dataset.DatasetDownloadError,
        match="full 40-character commit hash",
    ):
        download_dataset.load_manifest(manifest_path)

    valid_document = _write_manifest(
        manifest_path,
        status="published",
        revision=HF_REVISION,
        source_kind="hugging-face",
        base_url_template="https://huggingface.co/datasets/example/data/resolve/{revision}",
        artifacts=_artifacts(_payloads()),
    )
    _schema_validator().validate(valid_document)
    assert download_dataset.load_manifest(manifest_path).revision == HF_REVISION


def test_hugging_face_revision_override_must_be_a_full_commit_hash(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        status="published",
        revision=HF_REVISION,
        source_kind="hugging-face",
        base_url_template="https://huggingface.co/datasets/example/data/resolve/{revision}",
        artifacts=_artifacts(_payloads()),
    )

    with pytest.raises(
        download_dataset.DatasetDownloadError,
        match="revision override.*40-character commit hash",
    ):
        download_dataset.download_dataset(
            manifest_path,
            tmp_path / "dataset",
            revision_override="mutable-main",
        )


def test_downloads_transport_then_installs_atomically(
    tmp_path: Path,
) -> None:
    payloads = _payloads(**{"public-inputs/video.mp4": b"synthetic video bytes"})
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads),
        )
        result = download_dataset.main(_download_arguments(manifest_path, output))

    assert result == 0
    for path, payload in payloads.items():
        assert output.joinpath(*path.split("/")).read_bytes() == payload
    receipt = json.loads((output / "download-receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == download_dataset.RECEIPT_SCHEMA
    assert receipt["revision"] == DIRECT_REVISION
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert "semantic_validation" not in receipt
    assert [artifact["path"] for artifact in receipt["artifacts"]] == list(payloads)


def test_explicit_base_url_and_revision_override_select_a_mirror(
    tmp_path: Path,
) -> None:
    override_revision = "mirror-revision-0001"
    payloads = _payloads()
    server_root = tmp_path / "server"
    _publish_files(server_root / "mirror" / override_revision, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template="https://primary.invalid/{revision}",
            artifacts=_artifacts(payloads),
        )
        result = download_dataset.main(
            [
                *_download_arguments(manifest_path, output),
                "--base-url",
                f"{server_url}/mirror/{{revision}}",
                "--revision",
                override_revision,
            ]
        )

    assert result == 0
    receipt = json.loads((output / "download-receipt.json").read_text(encoding="utf-8"))
    assert receipt["revision"] == override_revision
    assert receipt["base_url"].endswith(f"/mirror/{override_revision}")


def test_checksum_failure_removes_staging_but_keeps_verified_cache(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"
    wrong_path = "dataset/manifest.json"

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads, wrong_sha256_path=wrong_path),
        )
        result = download_dataset.main(_download_arguments(manifest_path, output))

    assert result == 2
    assert not output.exists()
    assert list(tmp_path.glob(".dataset.part-*")) == []
    cache_directories = list(tmp_path.glob(f".dataset.cache-*-{DIRECT_REVISION}"))
    assert len(cache_directories) == 1
    assert (cache_directories[0] / "README.md").is_file()
    assert not (cache_directories[0] / wrong_path).exists()


def test_verified_cache_is_reused_and_bound_to_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    other_revision = "direct-revision-0002"
    payloads = _payloads()
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    _publish_files(server_root / other_revision, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"
    original_download = download_dataset._download_artifact

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads),
        )

        first_attempt: list[str] = []

        def interrupt_second(url: str, destination: Path, artifact) -> None:
            first_attempt.append(artifact.path.as_posix())
            if len(first_attempt) == 2:
                raise download_dataset.DatasetDownloadError("synthetic interruption")
            original_download(url, destination, artifact)

        monkeypatch.setattr(download_dataset, "_download_artifact", interrupt_second)
        assert download_dataset.main(_download_arguments(manifest_path, output)) == 2
        assert first_attempt == [".gitattributes", "README.md"]

        other_attempt: list[str] = []

        def interrupt_other_revision(url: str, destination: Path, artifact) -> None:
            other_attempt.append(artifact.path.as_posix())
            raise download_dataset.DatasetDownloadError("synthetic second interruption")

        monkeypatch.setattr(
            download_dataset,
            "_download_artifact",
            interrupt_other_revision,
        )
        assert (
            download_dataset.main(
                [
                    *_download_arguments(manifest_path, output),
                    "--revision",
                    other_revision,
                ]
            )
            == 2
        )
        assert other_attempt == [".gitattributes"]

        resumed_downloads: list[str] = []

        def record_resumed_download(url: str, destination: Path, artifact) -> None:
            resumed_downloads.append(artifact.path.as_posix())
            original_download(url, destination, artifact)

        monkeypatch.setattr(
            download_dataset,
            "_download_artifact",
            record_resumed_download,
        )
        assert download_dataset.main(_download_arguments(manifest_path, output)) == 0

    assert ".gitattributes" not in resumed_downloads
    assert resumed_downloads == list(payloads)[1:]
    assert "Using verified cache for .gitattributes" in capsys.readouterr().out
    assert list(tmp_path.glob(f".dataset.cache-*-{DIRECT_REVISION}")) == []
    assert list(tmp_path.glob(f".dataset.cache-*-{other_revision}"))


def test_verified_existing_download_is_accepted_without_refetch(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    server_root = tmp_path / "server"
    release_root = server_root / DIRECT_REVISION
    _publish_files(release_root, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads),
        )
        arguments = _download_arguments(manifest_path, output)
        assert download_dataset.main(arguments) == 0
        for path in sorted(release_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        assert download_dataset.main(arguments) == 0
    for path, payload in payloads.items():
        assert output.joinpath(*path.split("/")).read_bytes() == payload


def test_existing_download_rejects_undeclared_extra_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payloads = _payloads()
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"

    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads),
        )
        arguments = _download_arguments(manifest_path, output)
        assert download_dataset.main(arguments) == 0

    (output / "undeclared.txt").write_text("not in the manifest\n", encoding="utf-8")
    assert download_dataset.main(arguments) == 2
    assert "refusing to overwrite unverified or different dataset" in capsys.readouterr().err


def test_manifest_rejects_unsafe_or_duplicate_artifact_paths(tmp_path: Path) -> None:
    base_artifacts = _artifacts(_payloads())
    cases = [
        [
            *base_artifacts,
            {"path": "../capture.json", "bytes": 1, "sha256": "0" * 64},
        ],
        [*base_artifacts, base_artifacts[0]],
    ]
    for index, artifacts in enumerate(cases):
        manifest = tmp_path / f"manifest-{index}.json"
        _write_manifest(
            manifest,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template="https://example.invalid/{revision}",
            artifacts=artifacts,
        )

        with pytest.raises(download_dataset.DatasetDownloadError):
            download_dataset.load_manifest(manifest)


def test_manifest_rejects_file_child_conflict_with_graceful_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=[
            *_artifacts(_payloads()),
            {
                "path": "README.md/child.json",
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ],
    )
    output = tmp_path / "dataset"

    assert download_dataset.main(_download_arguments(manifest, output)) == 2
    assert not output.exists()
    assert "file/child conflict: README.md and README.md/child.json" in (capsys.readouterr().err)


def test_source_url_must_bind_the_revision(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        status="published",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/latest",
        artifacts=_artifacts(_payloads()),
    )

    with pytest.raises(download_dataset.DatasetDownloadError, match=r"\{revision\}"):
        download_dataset.load_manifest(manifest)


def test_workspace_registration_runs_after_new_and_existing_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads(**{"captures/example/video.mp4": b"verified video"})
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    output = tmp_path / "dataset"
    workspace = tmp_path / "workspace"
    manifest_path = tmp_path / "manifest.json"
    calls: list[tuple[Path, Path]] = []

    def register(
        manifest: download_dataset.DatasetManifest,
        registered_output: Path,
        registered_workspace: Path,
    ) -> None:
        assert manifest.dataset_id == "cubed-core-test-dataset"
        assert (registered_output / "download-receipt.json").is_file()
        assert (registered_output / "captures/example/video.mp4").read_bytes() == (
            b"verified video"
        )
        calls.append((registered_output, registered_workspace))

    monkeypatch.setattr(download_dataset, "_register_workspace", register)
    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads),
        )
        arguments = [
            *_download_arguments(manifest_path, output),
            "--workspace",
            str(workspace),
        ]
        assert download_dataset.main(arguments) == 0

    assert download_dataset.main(arguments) == 0
    assert calls == [(output, workspace), (output, workspace)]


def test_workspace_registration_does_not_run_after_download_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads()
    server_root = tmp_path / "server"
    _publish_files(server_root / DIRECT_REVISION, payloads)
    output = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"
    registration_called = False

    def register(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal registration_called
        registration_called = True

    monkeypatch.setattr(download_dataset, "_register_workspace", register)
    with _serve(server_root) as server_url:
        _write_manifest(
            manifest_path,
            status="published",
            revision=DIRECT_REVISION,
            base_url_template=f"{server_url}/{{revision}}",
            artifacts=_artifacts(payloads, wrong_sha256_path="dataset/manifest.json"),
        )
        result = download_dataset.main(
            [
                *_download_arguments(manifest_path, output),
                "--workspace",
                str(tmp_path / "workspace"),
            ]
        )

    assert result == 2
    assert registration_called is False


def test_workspace_errors_are_bounded_cli_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cubed_core import public_dataset_registration
    from cubed_core.workspace import WorkspaceError

    manifest = download_dataset.DatasetManifest(
        dataset_id="cubed-core-test-dataset",
        description="Synthetic downloader test fixture.",
        publication_status="published",
        source_kind="direct-mirror",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=(),
        sha256="a" * 64,
    )

    def fail_download(*args: object, **kwargs: object):
        del args, kwargs
        return manifest

    def fail_registration(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise WorkspaceError("synthetic workspace initialization failure")

    monkeypatch.setattr(download_dataset, "download_dataset", fail_download)
    monkeypatch.setattr(
        public_dataset_registration,
        "register_downloaded_public_dataset",
        fail_registration,
    )

    result = download_dataset.main(
        [
            "--output-dir",
            str(tmp_path / "dataset"),
            "--workspace",
            str(tmp_path / "workspace"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "workspace registration failed: synthetic workspace initialization failure" in (
        captured.err
    )
    assert "Traceback" not in captured.err


def test_settings_initialization_errors_are_bounded_cli_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = download_dataset.DatasetManifest(
        dataset_id="cubed-core-test-dataset",
        description="Synthetic downloader test fixture.",
        publication_status="published",
        source_kind="direct-mirror",
        revision=DIRECT_REVISION,
        base_url_template="https://example.invalid/{revision}",
        artifacts=(),
        sha256="a" * 64,
    )

    def fail_download(*args: object, **kwargs: object):
        del args, kwargs
        return manifest

    monkeypatch.setattr(download_dataset, "download_dataset", fail_download)
    monkeypatch.setenv("CUBED_CORE_MAX_UPLOAD_BYTES", "not-an-integer")

    result = download_dataset.main(
        [
            "--output-dir",
            str(tmp_path / "dataset"),
            "--workspace",
            str(tmp_path / "workspace"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "CUBED_CORE_MAX_UPLOAD_BYTES must be an integer" in captured.err
    assert "Traceback" not in captured.err


def test_download_dataset_make_target_honors_cubed_core_workspace(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    custom_workspace = tmp_path / "custom workspace"
    environment = os.environ.copy()
    environment["CUBED_CORE_WORKSPACE"] = str(custom_workspace)
    environment.pop("DECODE_WORKSPACE", None)

    result = subprocess.run(
        ["make", "-n", "download-dataset"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f'--workspace "{custom_workspace}"' in result.stdout
