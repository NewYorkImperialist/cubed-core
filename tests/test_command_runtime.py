from __future__ import annotations

import os
from pathlib import Path

import pytest

from cubed_core.command_runtime import resolve_command_executable


@pytest.mark.skipif(os.name != "posix", reason="executable-bit semantics are POSIX-specific")
def test_command_resolution_uses_launch_directory_for_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runner = bin_dir / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.chmod(0o700)

    assert resolve_command_executable(("./bin/runner",), cwd=tmp_path) == str(runner)

    monkeypatch.setenv("PATH", "bin")
    assert resolve_command_executable(("runner",), cwd=tmp_path) == str(runner)

    runner.chmod(0o600)
    assert resolve_command_executable(("runner",), cwd=tmp_path) is None
