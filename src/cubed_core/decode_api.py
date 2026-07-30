from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .decode_contract import DecodeContractError, build_decode_preflight

router = APIRouter()


def _read_schema(repo_root: Path, filename: str) -> dict[str, Any]:
    path = repo_root / "schemas" / filename
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"{filename} is unavailable") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=503, detail=f"{filename} is invalid")
    return value


@router.get("/api/specs/decode-result")
async def decode_result_spec(request: Request) -> dict[str, Any]:
    return _read_schema(request.app.state.settings.repo_root, "decode-result-v1.schema.json")


@router.get("/api/specs/decode-preflight")
async def decode_preflight_spec(request: Request) -> dict[str, Any]:
    return _read_schema(request.app.state.settings.repo_root, "decode-preflight-v1.schema.json")


@router.get("/api/captures/{capture_id}/decode-preflight")
async def decode_preflight(capture_id: str, request: Request) -> dict[str, Any]:
    workspace = request.app.state.workspace
    capture = next(
        (
            item
            for item in workspace.list_captures()
            if item.get("recording_id") == capture_id or item.get("capture_id") == capture_id
        ),
        None,
    )
    if capture is None:
        raise HTTPException(status_code=404, detail="capture not found")
    try:
        return await run_in_threadpool(
            build_decode_preflight,
            capture,
            repo_root=request.app.state.settings.repo_root,
            workspace_root=request.app.state.settings.workspace,
        )
    except DecodeContractError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
