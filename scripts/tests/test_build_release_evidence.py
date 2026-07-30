from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts import build_release_evidence as evidence

COMMIT = "a" * 40
TREE = "b" * 40
COMMIT_EPOCH = 1_767_225_600


def _seed_repository(repository: Path) -> None:
    for source_path, _bundle_path in evidence.COPIED_INPUTS:
        path = repository / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{source_path}\n", encoding="utf-8")


class _FakeRunner:
    def __init__(
        self,
        repository: Path,
        *,
        dirty: bool = False,
        dirty_on_final_check: bool = False,
        npm_runtime_node: str = evidence.NODE_VERSION,
        blob_overrides: dict[str, bytes] | None = None,
    ) -> None:
        self.repository = repository
        self.dirty = dirty
        self.dirty_on_final_check = dirty_on_final_check
        self.npm_runtime_node = npm_runtime_node
        self.blob_overrides = blob_overrides or {}
        self.status_checks = 0
        self.sbom_counter = 0
        self.calls: list[list[str]] = []
        self.binary_calls: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    def __call__(self, args: list[str], cwd: Path, env: dict[str, str]) -> str:
        assert cwd == self.repository.resolve()
        self.calls.append(args)
        self.environments.append(env)
        if args == ["git", "rev-parse", "--show-toplevel"]:
            return f"{self.repository.resolve()}\n"
        if args == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            self.status_checks += 1
            became_dirty = self.dirty_on_final_check and self.status_checks > 1
            return " M NOTICE\n" if self.dirty or became_dirty else ""
        if args == ["git", "rev-parse", "--show-object-format"]:
            return "sha1\n"
        if args == [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            "v0.1.0^{commit}",
        ]:
            return f"{COMMIT}\n"
        if args == ["git", "rev-parse", "--verify", "HEAD"]:
            return f"{COMMIT}\n"
        if args == ["git", "rev-parse", "--verify", f"{COMMIT}^{{tree}}"]:
            return f"{TREE}\n"
        if args == ["git", "show", "-s", "--format=%ct", COMMIT]:
            return f"{COMMIT_EPOCH}\n"
        if args == ["uv", "--version"]:
            return f"uv {evidence.UV_VERSION} (test build)\n"
        if args == ["node", "--version"]:
            return f"v{evidence.NODE_VERSION}\n"
        if args == ["npm", "--versions", "--json"]:
            return json.dumps(
                {
                    "npm": evidence.NPM_VERSION,
                    "node": self.npm_runtime_node,
                }
            )
        if args[0] == "uv" and args[1] == "export":
            return self._sbom_json("uv")
        if args[:4] == ["npm", "--prefix", "apps/lab-web", "sbom"]:
            return self._sbom_json("npm")
        raise AssertionError(f"unexpected command: {args}")

    def run_bytes(self, args: list[str], cwd: Path, env: dict[str, str]) -> bytes:
        assert cwd == self.repository.resolve()
        assert env["SOURCE_DATE_EPOCH"] == str(COMMIT_EPOCH)
        self.binary_calls.append(args)
        prefix = ["git", "cat-file", "blob"]
        if args[:3] != prefix or len(args) != 4:
            raise AssertionError(f"unexpected binary command: {args}")
        commit, separator, source_path = args[3].partition(":")
        assert separator == ":"
        assert commit == COMMIT
        return self.blob_overrides.get(source_path, (self.repository / source_path).read_bytes())

    def _sbom_json(self, tool: str) -> str:
        self.sbom_counter += 1
        return json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "serialNumber": f"urn:uuid:raw-{self.sbom_counter}",
                "version": 1,
                "metadata": {
                    "timestamp": f"2026-07-24T00:00:0{self.sbom_counter}.000Z",
                    "tools": [{"name": tool}],
                },
                "components": [{"name": f"component-{tool}", "version": "1"}],
                "dependencies": [{"ref": f"component-{tool}", "dependsOn": []}],
            }
        )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_fake_bundle(
    repository: Path,
    output: Path,
    **runner_options: Any,
) -> tuple[Path, _FakeRunner]:
    runner = _FakeRunner(repository, **runner_options)
    bundle = evidence.build_release_evidence(
        repository,
        output,
        source_ref="v0.1.0",
        runner=runner,
        binary_runner=runner.run_bytes,
    )
    return bundle, runner


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(evidence._canonical_json(value))


def _refresh_checksum_receipt(bundle: Path) -> None:
    records = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(bundle).as_posix()
            records.append(f"{evidence._sha256(path.read_bytes())}  {relative}\n")
    (bundle / "SHA256SUMS").write_text("".join(records), encoding="utf-8")


def _refresh_manifest_file_record(bundle: Path, relative: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (bundle / relative).read_bytes()
    record = next(item for item in manifest["files"] if item["path"] == relative)
    record["sha256"] = evidence._sha256(payload)
    record["size_bytes"] = len(payload)
    _write_canonical_json(manifest_path, manifest)


def test_build_is_deterministic_and_covers_all_profiles(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    runner = _FakeRunner(repository)

    first = evidence.build_release_evidence(
        repository,
        tmp_path / "evidence-one",
        source_ref="v0.1.0",
        runner=runner,
        binary_runner=runner.run_bytes,
    )
    second = evidence.build_release_evidence(
        repository,
        tmp_path / "evidence-two",
        source_ref="v0.1.0",
        runner=runner,
        binary_runner=runner.run_bytes,
    )

    assert _tree_bytes(first) == _tree_bytes(second)
    manifest = evidence.verify_release_evidence(first)
    assert manifest["source"]["commit"] == COMMIT
    assert manifest["source"]["tree"] == TREE
    assert manifest["source"]["object_format"] == "sha1"
    assert manifest["tools"] == {
        "uv": evidence.UV_VERSION,
        "node": evidence.NODE_VERSION,
        "npm": evidence.NPM_VERSION,
    }
    assert {item["profile"] for item in manifest["sboms"]} == {
        "python-workbench",
        "python-label-cpu",
        "python-decode-gpu",
        "python-research-gpu",
        "python-training",
        "frontend-lock",
    }
    assert (
        "python-decode-gpu",
        ("dev", "label", "decode", "tracker-gpu"),
    ) in evidence.PYTHON_PROFILES
    assert (
        "python-research-gpu",
        ("dev", "label", "decode", "tracker-gpu", "research-gpu"),
    ) in evidence.PYTHON_PROFILES
    assert (
        dict(evidence.PYTHON_PROFILES)["python-decode-gpu"]
        != dict(evidence.PYTHON_PROFILES)["python-research-gpu"]
    )

    sbom = json.loads((first / "sbom/frontend-lock.cdx.json").read_text(encoding="utf-8"))
    assert sbom["metadata"]["timestamp"] == "2026-01-01T00:00:00Z"
    assert sbom["serialNumber"].startswith("urn:uuid:")
    assert "raw-" not in sbom["serialNumber"]
    assert {item["name"]: item["value"] for item in sbom["metadata"]["properties"]} == {
        "io.github.kingbobjoeiv.cubed-core:profile": "frontend-lock",
        "io.github.kingbobjoeiv.cubed-core:source-commit": COMMIT,
    }
    assert all(call_env["UV_OFFLINE"] == "1" for call_env in runner.environments)
    assert all(call_env["npm_config_offline"] == "true" for call_env in runner.environments)


def test_checksum_verifier_rejects_tampering(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    bundle, _runner = _build_fake_bundle(repository, tmp_path / "evidence")
    (bundle / "notices/NOTICE").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(evidence.ReleaseEvidenceError, match="checksum mismatch"):
        evidence.verify_release_evidence(bundle)


def test_builder_refuses_dirty_source_or_existing_output(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)

    with pytest.raises(evidence.ReleaseEvidenceError, match="must be clean"):
        evidence.build_release_evidence(
            repository,
            tmp_path / "dirty-evidence",
            source_ref="v0.1.0",
            runner=_FakeRunner(repository, dirty=True),
        )

    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(evidence.ReleaseEvidenceError, match="refusing to overwrite"):
        evidence.build_release_evidence(
            repository,
            output,
            source_ref="v0.1.0",
            runner=_FakeRunner(repository),
        )


def test_builder_rechecks_source_state_before_publish(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    output = tmp_path / "changed-during-build"

    runner = _FakeRunner(repository, dirty_on_final_check=True)
    with pytest.raises(evidence.ReleaseEvidenceError, match="changed during"):
        evidence.build_release_evidence(
            repository,
            output,
            source_ref="v0.1.0",
            runner=runner,
            binary_runner=runner.run_bytes,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".changed-during-build.tmp-*"))


def test_builder_refuses_output_inside_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    with pytest.raises(evidence.ReleaseEvidenceError, match="outside"):
        evidence.build_release_evidence(
            repository,
            repository / "release-evidence",
            source_ref="v0.1.0",
            runner=_FakeRunner(repository),
        )


def test_normalizer_rejects_wrong_cyclonedx_version() -> None:
    with pytest.raises(evidence.ReleaseEvidenceError, match="CycloneDX 1.5"):
        evidence.normalize_sbom(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [],
                "dependencies": [],
            },
            profile="frontend-lock",
            source_commit=COMMIT,
            commit_timestamp="2026-01-01T00:00:00Z",
        )


def test_builder_rejects_worktree_bytes_that_differ_from_commit_blob(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    runner = _FakeRunner(
        repository,
        blob_overrides={"NOTICE": b"committed notice bytes\n"},
    )

    with pytest.raises(evidence.ReleaseEvidenceError, match="committed blob for NOTICE"):
        evidence.build_release_evidence(
            repository,
            tmp_path / "evidence",
            source_ref="v0.1.0",
            runner=runner,
            binary_runner=runner.run_bytes,
        )


def test_builder_validates_the_node_runtime_used_by_npm(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    runner = _FakeRunner(repository, npm_runtime_node="23.0.0")

    with pytest.raises(evidence.ReleaseEvidenceError, match="npm must run under Node"):
        evidence.build_release_evidence(
            repository,
            tmp_path / "evidence",
            source_ref="v0.1.0",
            runner=runner,
            binary_runner=runner.run_bytes,
        )


def test_builder_normalizes_bundle_modes_under_restrictive_umask(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    previous_umask = os.umask(0o077)
    try:
        bundle, _runner = _build_fake_bundle(repository, tmp_path / "evidence")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(bundle.stat().st_mode) == 0o755
    for path in bundle.rglob("*"):
        expected_mode = 0o755 if path.is_dir() else 0o644
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode


def _set_invalid_source_commit(manifest: dict[str, Any]) -> None:
    manifest["source"]["commit"] = "not-a-git-object"


def _set_invalid_tool_version(manifest: dict[str, Any]) -> None:
    manifest["tools"]["npm"] = "0.0.0"


def _set_invalid_sbom_profile(manifest: dict[str, Any]) -> None:
    manifest["sboms"][0]["profile"] = "unknown-profile"


def _set_invalid_sbom_count(manifest: dict[str, Any]) -> None:
    manifest["sboms"][0]["component_count"] += 1


def _set_invalid_sbom_command(manifest: dict[str, Any]) -> None:
    manifest["sboms"][0]["command"].append("--unexpected")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_invalid_source_commit, "source.commit"),
        (_set_invalid_tool_version, "pinned generator versions"),
        (_set_invalid_sbom_profile, "unknown manifest SBOM profile"),
        (_set_invalid_sbom_count, "component_count"),
        (_set_invalid_sbom_command, "command does not match"),
    ],
)
def test_verifier_rejects_resigned_semantically_invalid_manifest(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    bundle, _runner = _build_fake_bundle(repository, tmp_path / "evidence")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    _write_canonical_json(manifest_path, manifest)
    _refresh_checksum_receipt(bundle)

    with pytest.raises(evidence.ReleaseEvidenceError, match=message):
        evidence.verify_release_evidence(bundle)


def test_verifier_rejects_resigned_sbom_with_wrong_source_property(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _seed_repository(repository)
    bundle, _runner = _build_fake_bundle(repository, tmp_path / "evidence")
    relative = "sbom/frontend-lock.cdx.json"
    sbom_path = bundle / relative
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    properties = sbom["metadata"]["properties"]
    source_property = next(
        item
        for item in properties
        if item["name"] == "io.github.kingbobjoeiv.cubed-core:source-commit"
    )
    source_property["value"] = "c" * 40
    _write_canonical_json(sbom_path, sbom)
    _refresh_manifest_file_record(bundle, relative)
    _refresh_checksum_receipt(bundle)

    with pytest.raises(evidence.ReleaseEvidenceError, match="not normalized and bound"):
        evidence.verify_release_evidence(bundle)
