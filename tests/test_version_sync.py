from __future__ import annotations

import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _required_match(pattern: str, text: str, *, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None, f"could not read {label}"
    return match.group(1)


def test_python_web_and_lock_metadata_use_one_project_version() -> None:
    pyproject_version = _required_match(
        r'^version = "([^"]+)"$',
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        label="pyproject version",
    )
    package = json.loads(
        (REPOSITORY_ROOT / "apps/lab-web/package.json").read_text(encoding="utf-8")
    )
    package_lock = json.loads(
        (REPOSITORY_ROOT / "apps/lab-web/package-lock.json").read_text(encoding="utf-8")
    )
    python_version = _required_match(
        r'^__version__ = "([^"]+)"$',
        (REPOSITORY_ROOT / "src/cubed_core/__init__.py").read_text(encoding="utf-8"),
        label="Python package version",
    )
    uv_version = _required_match(
        r'^\[\[package\]\]\nname = "cubed-core"\nversion = "([^"]+)"$',
        (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"),
        label="uv lock project version",
    )

    assert {
        pyproject_version,
        python_version,
        package["version"],
        package_lock["version"],
        package_lock["packages"][""]["version"],
        uv_version,
    } == {pyproject_version}
