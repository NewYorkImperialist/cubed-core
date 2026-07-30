from __future__ import annotations

import json
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_RANGE = ">=3.10,<3.13"
NODE_RANGE = ">=22.3.0 <23"
CI_NODE_VERSION = (22, 23, 1)


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _toml_string(path: str, key: str) -> str:
    match = re.search(rf'^{re.escape(key)} = "([^"]+)"$', _text(path), flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_metadata_and_locks_match() -> None:
    package = json.loads(_text("apps/lab-web/package.json"))
    package_lock = json.loads(_text("apps/lab-web/package-lock.json"))

    assert _toml_string("pyproject.toml", "requires-python") == PYTHON_RANGE
    assert _toml_string("uv.lock", "requires-python").replace(" ", "") == PYTHON_RANGE
    assert package["engines"]["node"] == NODE_RANGE
    assert package_lock["packages"][""]["engines"]["node"] == NODE_RANGE


def test_ci_covers_the_declared_python_range_and_a_supported_node() -> None:
    workflow = _text(".github/workflows/check.yml")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert '"3.13"' not in workflow

    node_versions = re.findall(r'node-version: "(\d+)\.(\d+)\.(\d+)"', workflow)
    assert node_versions
    assert {tuple(map(int, version)) for version in node_versions} == {CI_NODE_VERSION}
    assert CI_NODE_VERSION[0] == 22
    assert CI_NODE_VERSION[1:] >= (3, 0)


def test_runtime_docs_state_the_tested_ranges() -> None:
    readme = _text("README.md")
    assert re.search(r"Python 3\.10\s+through\s+3\.12", readme)
    assert re.search(r"Node 22\.3 or newer\s+on the Node 22 line", readme)

    gpu = _text("docs/CLOUD_GPU.md")
    assert re.search(r"Python 3\.10\s+through\s+3\.12", gpu)


def test_makefile_gates_bootstrap_and_check_on_node_version() -> None:
    makefile = _text("Makefile")

    assert re.search(r"^node-version-check:", makefile, flags=re.MULTILINE)
    assert re.search(r"^bootstrap: .*node-version-check", makefile, flags=re.MULTILINE)
    assert re.search(r"^check: .*node-version-check", makefile, flags=re.MULTILINE)


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("v22.3.0", True),
        ("v22.23.1", True),
        ("v22.2.0", False),
        ("v23.0.0", False),
    ],
)
def test_node_version_gate_is_fail_closed(tmp_path: Path, version: str, supported: bool) -> None:
    fake_node = tmp_path / "node"
    fake_node.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n", encoding="utf-8")
    fake_node.chmod(fake_node.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        ["make", "--silent", "node-version-check", f"NODE={fake_node}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert (result.returncode == 0) is supported
    if not supported:
        assert f"Node {NODE_RANGE} is required; found {version}" in result.stderr
