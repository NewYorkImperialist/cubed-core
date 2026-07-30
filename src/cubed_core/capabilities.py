from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from .decode_jobs import decode_capability
from .label_assist import runtime_capability
from .label_predictions import prediction_capability
from .settings import Settings


def _gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "devices": [], "reason": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "devices": [], "reason": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "devices": [],
            "reason": result.stderr.strip()[:300] or "nvidia-smi failed",
        }
    devices = []
    for line in result.stdout.splitlines():
        name, _, memory = line.partition(",")
        try:
            memory_mib = int(memory.strip())
        except ValueError:
            memory_mib = None
        devices.append({"name": name.strip(), "memory_mib": memory_mib})
    return {"available": bool(devices), "devices": devices}


def _tool_registry(settings: Settings) -> list[dict[str, Any]]:
    path = settings.repo_root / "config" / "tooling.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def build_capabilities(settings: Settings) -> dict[str, Any]:
    settings.workspace.mkdir(parents=True, exist_ok=True)
    return {
        "schema": "cubed-core/capabilities-v1",
        "workspace": str(settings.workspace),
        "commands": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "git": shutil.which("git"),
            "node": shutil.which("node"),
        },
        "gpu": _gpu_status(),
        "upload_limits": {
            "video_bytes": settings.max_upload_bytes,
        },
        "decode_jobs": decode_capability(settings),
        "label": {
            "pnp_assist": runtime_capability(),
            "prediction": prediction_capability(settings),
            "autosave": {
                "workspace_capture": True,
                "local_file": "browser-local-storage",
            },
            "exports": ["frame-annotations-v1", "yolo-pose-zip"],
        },
        "tools": _tool_registry(settings),
    }
