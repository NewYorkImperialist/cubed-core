from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path


def resolve_command_executable(command: Sequence[str], *, cwd: Path) -> str | None:
    """Resolve argv[0] for a shell-free subprocess launched from ``cwd``."""

    if not command:
        return None
    executable = command[0]
    if os.path.dirname(executable):
        candidate = Path(executable)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        executable = str(candidate)
        return shutil.which(executable)
    search_path = os.environ.get("PATH")
    if search_path is not None:
        entries = []
        for raw_entry in search_path.split(os.pathsep):
            entry = Path(raw_entry or os.curdir)
            if not entry.is_absolute():
                entry = cwd / entry
            entries.append(str(entry))
        search_path = os.pathsep.join(entries)
    return shutil.which(executable, path=search_path)
