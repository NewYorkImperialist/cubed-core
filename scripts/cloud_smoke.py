#!/usr/bin/env python3
"""Read-only health and capability smoke check for a Cubed Core service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SMOKE_SCHEMA = "cubed-core/cloud-smoke-v1"


class SmokeError(RuntimeError):
    """Raised when the service cannot return a usable smoke-check response."""


def _fetch_object(
    base_url: str,
    endpoint: str,
    timeout: float,
    *,
    admin_token: str,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "cubed-core-cloud-smoke/1",
    }
    if admin_token:
        headers["X-Cubed-Admin-Token"] = admin_token
    request = Request(
        f"{base_url.rstrip('/')}{endpoint}",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise SmokeError(f"{endpoint} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeError(f"{endpoint} did not return a JSON object")
    return payload


def evaluate(
    health: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    require_gpu: bool,
) -> dict[str, Any]:
    checks: list[dict[str, object]] = []

    def add(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "ok": ok, "detail": detail})

    add(
        "health",
        health.get("status") == "ok",
        f"service status is {health.get('status')!r}",
    )
    add(
        "capabilities-schema",
        capabilities.get("schema") == "cubed-core/capabilities-v1",
        f"capabilities schema is {capabilities.get('schema')!r}",
    )

    commands = capabilities.get("commands")
    commands = commands if isinstance(commands, dict) else {}
    for command in ("ffmpeg", "ffprobe"):
        available = bool(commands.get(command))
        add(
            f"command-{command}",
            available,
            f"{command} is {'available' if available else 'missing'}",
        )

    workspace = capabilities.get("workspace")
    add(
        "workspace",
        isinstance(workspace, str) and bool(workspace),
        f"service workspace is {workspace!r}",
    )

    gpu = capabilities.get("gpu")
    gpu = gpu if isinstance(gpu, dict) else {}
    gpu_available = gpu.get("available") is True
    add(
        "gpu-required" if require_gpu else "gpu-optional",
        gpu_available or not require_gpu,
        "GPU detected" if gpu_available else f"GPU not detected: {gpu.get('reason', 'unknown')}",
    )

    tools = capabilities.get("tools")
    tools = tools if isinstance(tools, list) else []
    decoder = next(
        (tool for tool in tools if isinstance(tool, dict) and tool.get("id") == "decoder"),
        None,
    )
    decoder_status = decoder.get("status") if decoder else None
    add(
        "decoder-status-disclosed",
        isinstance(decoder_status, str) and bool(decoder_status),
        f"decoder status is {decoder_status!r}",
    )

    return {
        "schema": SMOKE_SCHEMA,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "service": {
            "version": health.get("version"),
            "publication_status": health.get("publication_status"),
        },
        "runtime": {
            "workspace": workspace,
            "commands": {
                "ffmpeg": commands.get("ffmpeg"),
                "ffprobe": commands.get("ffprobe"),
            },
            "gpu": gpu,
        },
        "decoder": {
            "status": decoder_status,
            "available": decoder_status == "available",
        },
    }


def _base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an http:// or https:// service URL")
    if parsed.username or parsed.password:
        raise argparse.ArgumentTypeError("must not contain credentials")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("must not contain a query or fragment")
    return value.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        type=_base_url,
        default="http://127.0.0.1:8000",
        help="Cubed Core service URL, normally a loopback address or local tunnel",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail unless the service can execute nvidia-smi and reports a GPU",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("CUBED_CORE_ADMIN_TOKEN", ""),
        help="Workbench admin token (defaults to CUBED_CORE_ADMIN_TOKEN)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        health = _fetch_object(
            args.base_url,
            "/api/health",
            args.timeout,
            admin_token=args.admin_token,
        )
        capabilities = _fetch_object(
            args.base_url,
            "/api/capabilities",
            args.timeout,
            admin_token=args.admin_token,
        )
    except SmokeError as exc:
        print(
            json.dumps(
                {
                    "schema": SMOKE_SCHEMA,
                    "ok": False,
                    "base_url": args.base_url,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    report = evaluate(
        health,
        capabilities,
        require_gpu=args.require_gpu,
    )
    report["base_url"] = args.base_url
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
