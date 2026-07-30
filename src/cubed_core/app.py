from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import threading
from dataclasses import replace
from ipaddress import ip_address
from typing import Annotated, Any
from urllib.parse import urlencode, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .capabilities import build_capabilities
from .decode_api import router as decode_router
from .decode_jobs import DecodeJobService
from .decode_jobs import router as decode_jobs_router
from .desktop_calibration import (
    MAX_CROP_BYTES,
    DesktopCalibrationError,
    build_centroids_document_from_crops,
)
from .frame_annotations import (
    MAX_ANNOTATION_BYTES,
    FrameAnnotationError,
    annotation_json_bytes,
    build_yolo_pose_archive,
    normalize_frame_annotations,
)
from .label_assist import LabelAssistError, LabelAssistUnavailable, extrapolate_request
from .label_predictions import (
    LabelPredictionDisabled,
    LabelPredictionError,
    run_label_alignment_scan,
    run_label_prediction,
)
from .media import MediaError, extract_frame_jpeg
from .media_access import MediaTicketRegistry
from .remote_hosts import RemoteHostsError, build_remote_hosts_response
from .settings import Settings
from .workspace import (
    Workspace,
    WorkspaceError,
    calibration_upload_display_name,
    parse_camera_intrinsics_form,
)

ADMIN_TOKEN_HEADER = "X-Cubed-Admin-Token"
MULTIPART_OVERHEAD_BYTES = 2 * 1024**2
SIDECAR_MAX_BYTES = 64 * 1024**2
LABEL_REQUEST_MAX_BYTES = 64 * 1024
CALIBRATION_CROPS_REQUEST_MAX_BYTES = 6 * MAX_CROP_BYTES + MULTIPART_OVERHEAD_BYTES


class _RequestBodyTooLarge(Exception):
    pass


def _is_loopback_host(value: str) -> bool:
    normalized = value.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _origin_identity(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname.lower().rstrip("."), port


def _local_browser_bootstrap_allowed(request: Request, settings: Settings) -> bool:
    """Admit only a same-origin browser talking directly over loopback.

    Host and client checks stop public reverse proxies and DNS rebinding from
    inheriting loopback trust. Exact Origin, supplemented by Fetch Metadata
    when the browser sends it, stops another web origin from extracting the
    token through localhost.
    """

    if not settings.local_browser_auth:
        return False
    request_host = request.url.hostname or ""
    client_host = request.client.host if request.client is not None else ""
    if not _is_loopback_host(request_host) or not _is_loopback_host(client_host):
        return False
    request_origin = _origin_identity(f"{request.url.scheme}://{request.url.netloc}")
    supplied_origin = _origin_identity(request.headers.get("Origin", ""))
    if request_origin is None or supplied_origin != request_origin:
        return False
    fetch_site = request.headers.get("Sec-Fetch-Site", "").lower()
    if fetch_site and fetch_site != "same-origin":
        return False
    fetch_mode = request.headers.get("Sec-Fetch-Mode", "").lower()
    if fetch_mode and fetch_mode != "cors":
        return False
    fetch_dest = request.headers.get("Sec-Fetch-Dest", "").lower()
    return not fetch_dest or fetch_dest == "empty"


class SecurityHeadersMiddleware:
    """Attach browser hardening headers to every HTTP response.

    A strict CSP needs either a nonce or a reviewed hash for the small inline
    token/theme bootstrap in ``index.html``.  Until that is implemented, keep
    this middleware to directives that are unambiguous for the local
    workbench and its API.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class RequestGuardMiddleware:
    """Reject unauthorized or oversized requests before multipart parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: Settings,
    ) -> None:
        self.app = app
        self.settings = settings

    @staticmethod
    def _header(scope: Scope, name: str) -> str:
        wanted = name.lower().encode("ascii")
        for key, value in scope.get("headers", []):
            if key.lower() == wanted:
                return value.decode("utf-8", errors="replace")
        return ""

    def _admin_matches(self, scope: Scope) -> bool:
        expected = self.settings.admin_token
        supplied = self._header(scope, ADMIN_TOKEN_HEADER)
        return bool(expected and supplied and secrets.compare_digest(expected, supplied))

    @staticmethod
    def _admin_route(path: str, method: str) -> bool:
        media_stream = method == "GET" and re.fullmatch(
            r"/api/captures/[a-f0-9]{32}/video",
            path,
        )
        return not media_stream and (
            path == "/api/capabilities"
            or path == "/api/remote-hosts"
            or path.startswith("/api/label/")
            or path == "/api/captures"
            or path.startswith("/api/captures/")
            or path == "/api/decode/jobs"
            or path.startswith("/api/decode/jobs/")
        )

    def _request_limit(self, path: str) -> int | None:
        if path == "/api/captures/import":
            return self.settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        if path == "/api/label/assist/extrapolate" or path.endswith("/label-predict"):
            return LABEL_REQUEST_MAX_BYTES
        if path.endswith("/calibration/from-crops"):
            return CALIBRATION_CROPS_REQUEST_MAX_BYTES
        if "/sidecars/" in path:
            return SIDECAR_MAX_BYTES + MULTIPART_OVERHEAD_BYTES
        if path.endswith("/annotations"):
            return MAX_ANNOTATION_BYTES
        return None

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status: int,
        detail: str,
    ) -> None:
        body: dict[str, Any] = {"detail": detail}
        await JSONResponse(body, status_code=status)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return
        admin_matches = self._admin_matches(scope)
        if self._admin_route(path, method) and not admin_matches:
            await self._reject(
                scope,
                receive,
                send,
                status=403,
                detail="admin token required",
            )
            return

        request_limit = self._request_limit(path)
        if request_limit is None:
            await self.app(scope, receive, send)
            return

        raw_content_length = self._header(scope, "content-length")
        if raw_content_length:
            try:
                content_length = int(raw_content_length)
            except ValueError:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status=400,
                    detail="invalid Content-Length",
                )
                return
            if content_length < 0:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status=400,
                    detail="invalid Content-Length",
                )
                return
            if content_length > request_limit:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status=413,
                    detail="request body exceeds the configured upload limit",
                )
                return

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > request_limit:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(
                scope,
                receive,
                send,
                status=413,
                detail="request body exceeds the configured upload limit",
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    if not active_settings.admin_token:
        active_settings = replace(
            active_settings,
            admin_token=secrets.token_urlsafe(32),
        )
    workspace = Workspace(
        active_settings.workspace,
        max_upload_bytes=active_settings.max_upload_bytes,
    )
    workspace.initialize()
    media_tickets = MediaTicketRegistry()
    decode_jobs = DecodeJobService(active_settings, workspace)
    label_prediction_lock = threading.Lock()
    frontend_dist = active_settings.repo_root / "apps" / "lab-web" / "dist"
    frontend_index = frontend_dist / "index.html"

    app = FastAPI(
        title="Cubed Core",
        version=__version__,
        description="Local decode workbench API.",
    )
    app.state.settings = active_settings
    app.state.workspace = workspace
    app.state.media_tickets = media_tickets
    app.state.decode_jobs = decode_jobs
    app.state.label_prediction_lock = label_prediction_lock
    app.router.add_event_handler("shutdown", decode_jobs.close)
    allowed_hosts = ["127.0.0.1", "localhost", "::1", "testserver"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Content-Type",
            ADMIN_TOKEN_HEADER,
        ],
    )
    app.add_middleware(
        RequestGuardMiddleware,
        settings=active_settings,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(decode_router)
    app.include_router(decode_jobs_router)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
        }

    @app.post("/api/local-session")
    async def local_session(request: Request) -> Response:
        if not _local_browser_bootstrap_allowed(request, active_settings):
            raise HTTPException(
                status_code=403,
                detail="local browser authorization unavailable",
            )
        return JSONResponse(
            {"admin_token": active_settings.admin_token},
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "Vary": "Origin",
            },
        )

    @app.get("/api/capabilities")
    async def capabilities() -> dict[str, object]:
        return build_capabilities(active_settings)

    @app.get("/api/remote-hosts")
    async def remote_hosts() -> dict[str, object]:
        try:
            return build_remote_hosts_response(active_settings)
        except RemoteHostsError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def read_schema(filename: str) -> dict[str, object]:
        schema_path = active_settings.repo_root / "schemas" / filename
        try:
            value = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail=f"{filename} is unavailable") from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=503, detail=f"{filename} is invalid")
        return value

    @app.get("/api/specs/capture-bundle")
    async def capture_bundle_spec() -> dict[str, object]:
        return read_schema("capture-bundle-v1.schema.json")

    @app.get("/api/specs/capture-derivation")
    async def capture_derivation_spec() -> dict[str, object]:
        return read_schema("capture-derivation-v1.schema.json")

    @app.get("/api/specs/ble-session")
    async def ble_session_spec() -> dict[str, object]:
        return read_schema("ble-session-v1.schema.json")

    @app.get("/api/specs/color-calibration")
    async def color_calibration_spec() -> dict[str, object]:
        return read_schema("color-calibration-v1.schema.json")

    @app.get("/api/specs/frame-annotations")
    async def frame_annotations_spec() -> dict[str, object]:
        return read_schema("frame-annotations-v1.schema.json")

    @app.get("/api/specs/model-artifact-manifest")
    async def model_artifact_manifest_spec() -> dict[str, object]:
        return read_schema("model-artifact-manifest-v1.schema.json")

    @app.get("/api/captures")
    async def captures() -> dict[str, object]:
        rows = workspace.list_captures()
        return {"captures": rows, "count": len(rows)}

    def capture_receipt(capture_id: str) -> dict[str, Any]:
        try:
            return next(
                row for row in workspace.list_captures() if row.get("capture_id") == capture_id
            )
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc

    def capture_has_active_decode(capture_id: str) -> bool:
        return any(
            job.get("status") in {"queued", "running"}
            for job in decode_jobs.list_for_capture(capture_id)
        )

    @app.delete("/api/captures/{capture_id}")
    async def delete_capture(capture_id: str) -> dict[str, Any]:
        capture_receipt(capture_id)
        active_decode = await run_in_threadpool(
            capture_has_active_decode,
            capture_id,
        )
        if active_decode:
            raise HTTPException(
                status_code=409,
                detail="capture has a queued or running decode job",
            )
        try:
            return await run_in_threadpool(workspace.trash_capture, capture_id)
        except WorkspaceError as exc:
            if str(exc) == "capture not found":
                raise HTTPException(status_code=404, detail="capture not found") from exc
            raise HTTPException(
                status_code=500,
                detail="capture could not be moved to Trash",
            ) from exc

    @app.post("/api/label/assist/extrapolate")
    async def label_assist_extrapolate(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await run_in_threadpool(extrapolate_request, body)
        except LabelAssistUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LabelAssistError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}/annotations")
    async def capture_annotations(capture_id: str) -> Response:
        receipt = capture_receipt(capture_id)
        try:
            raw = await run_in_threadpool(
                workspace.read_frame_annotations,
                capture_id,
                maximum_bytes=MAX_ANNOTATION_BYTES,
            )
            value = json.loads(raw)
            normalized = normalize_frame_annotations(
                value,
                expected_capture_id=capture_id,
                expected_sha256=str(receipt["video"]["sha256"]),
            )
        except WorkspaceError as exc:
            status = 404 if str(exc) == "capture annotations not found" else 500
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, FrameAnnotationError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"stored capture annotations are invalid: {exc}",
            ) from exc
        return JSONResponse(
            normalized,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.put("/api/captures/{capture_id}/annotations")
    async def save_capture_annotations(
        capture_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        receipt = capture_receipt(capture_id)
        try:
            normalized = normalize_frame_annotations(
                body,
                expected_capture_id=capture_id,
                expected_sha256=str(receipt["video"]["sha256"]),
            )
            encoded = annotation_json_bytes(normalized)
            target = await run_in_threadpool(
                workspace.write_frame_annotations,
                capture_id,
                encoded,
            )
        except FrameAnnotationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceError as exc:
            status = 404 if str(exc) == "capture not found" else 500
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return {
            "status": "saved",
            "capture_id": capture_id,
            "frames": len(normalized["frames"]),
            "bytes": len(encoded),
            "path": str(target.relative_to(active_settings.workspace)),
        }

    @app.get("/api/captures/{capture_id}/annotations/yolo-pose.zip")
    @app.get("/api/captures/{capture_id}/annotations/dataset.zip")
    async def export_capture_annotations_yolo(capture_id: str) -> Response:
        receipt = capture_receipt(capture_id)
        export_directory = None
        try:
            raw = await run_in_threadpool(
                workspace.read_frame_annotations,
                capture_id,
                maximum_bytes=MAX_ANNOTATION_BYTES,
            )
            document = normalize_frame_annotations(
                json.loads(raw),
                expected_capture_id=capture_id,
                expected_sha256=str(receipt["video"]["sha256"]),
            )
            video_path = workspace.capture_video_path(capture_id)
            export_directory = workspace.create_export_directory()
            archive = await run_in_threadpool(
                build_yolo_pose_archive,
                video_path,
                document,
                export_directory / f"{capture_id}.label-dataset.zip",
            )
        except WorkspaceError as exc:
            status = (
                404
                if str(exc)
                in {
                    "capture not found",
                    "capture video is unavailable",
                    "capture annotations not found",
                }
                else 500
            )
            if export_directory is not None:
                shutil.rmtree(export_directory, ignore_errors=True)
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, FrameAnnotationError) as exc:
            if export_directory is not None:
                shutil.rmtree(export_directory, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MediaError as exc:
            if export_directory is not None:
                shutil.rmtree(export_directory, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            if export_directory is not None:
                shutil.rmtree(export_directory, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail="label dataset export could not be written",
            ) from exc
        assert export_directory is not None
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"{capture_id}.label-dataset.zip",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
            background=BackgroundTask(
                shutil.rmtree,
                export_directory,
                ignore_errors=True,
            ),
        )

    @app.post("/api/captures/{capture_id}/label-predict")
    async def label_predict(
        capture_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        receipt = capture_receipt(capture_id)
        if not label_prediction_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="another label prediction is already running",
            )
        try:
            video_path = workspace.capture_video_path(capture_id)
            return await run_in_threadpool(
                run_label_prediction,
                active_settings,
                capture_id=capture_id,
                video_path=video_path,
                request=body,
                rotation_degrees=int(receipt.get("video", {}).get("rotation_degrees", 0)),
            )
        except LabelPredictionDisabled as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LabelPredictionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceError as exc:
            status = (
                404 if str(exc) in {"capture not found", "capture video is unavailable"} else 400
            )
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            label_prediction_lock.release()

    @app.post("/api/captures/{capture_id}/label-alignment")
    async def label_alignment(capture_id: str) -> dict[str, Any]:
        receipt = capture_receipt(capture_id)
        if not label_prediction_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="another Label model operation is already running",
            )
        try:
            video_path = workspace.capture_video_path(capture_id)
            video_receipt = receipt.get("video", {})
            return await run_in_threadpool(
                run_label_alignment_scan,
                active_settings,
                video_path=video_path,
                rotation_degrees=int(video_receipt.get("rotation_degrees", 0)),
                expected_frame_count=video_receipt.get("frame_count"),
            )
        except LabelPredictionDisabled as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LabelPredictionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceError as exc:
            status = (
                404 if str(exc) in {"capture not found", "capture video is unavailable"} else 400
            )
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            label_prediction_lock.release()

    @app.post("/api/captures/{capture_id}/media-ticket")
    async def create_media_ticket(capture_id: str) -> dict[str, object]:
        try:
            workspace.capture_video_path(capture_id)
        except WorkspaceError as exc:
            status = (
                404 if str(exc) in {"capture not found", "capture video is unavailable"} else 400
            )
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        ticket = media_tickets.create(capture_id)
        return {
            "capture_id": capture_id,
            "url": (f"/api/captures/{capture_id}/video?" + urlencode({"ticket": ticket.token})),
            "expires_at": ticket.expires_at,
        }

    @app.get("/api/captures/{capture_id}/video")
    async def capture_video(
        request: Request,
        capture_id: str,
        ticket: str = "",
    ) -> FileResponse:
        supplied_admin = request.headers.get(ADMIN_TOKEN_HEADER, "")
        admin_allowed = bool(
            supplied_admin and secrets.compare_digest(active_settings.admin_token, supplied_admin)
        )
        if not admin_allowed and media_tickets.authorize(capture_id, ticket) is None:
            raise HTTPException(status_code=403, detail="media ticket required")
        try:
            video_path = workspace.capture_video_path(capture_id)
        except WorkspaceError as exc:
            status = (
                404 if str(exc) in {"capture not found", "capture video is unavailable"} else 400
            )
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return FileResponse(
            video_path,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/api/captures/{capture_id}/frames/{frame_index}")
    async def capture_frame(capture_id: str, frame_index: int) -> Response:
        try:
            video_path = workspace.capture_video_path(capture_id)
            receipt = next(
                row for row in workspace.list_captures() if row.get("capture_id") == capture_id
            )
            frame_count = receipt.get("video", {}).get("frame_count")
            if isinstance(frame_count, int) and frame_index >= frame_count:
                raise WorkspaceError("frame is outside the video")
            jpeg = await run_in_threadpool(extract_frame_jpeg, video_path, frame_index)
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail="capture not found") from exc
        except WorkspaceError as exc:
            status = (
                404 if str(exc) in {"capture not found", "capture video is unavailable"} else 400
            )
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except MediaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def store_upload(
        video: UploadFile,
        *,
        source: str,
        notes: str,
        capture_session_id: str = "",
        scramble: str = "",
        camera_facing: str = "unknown",
        mirrored: bool = False,
        intrinsics: str = "",
        configured_fps: int | None = None,
    ) -> dict[str, object]:
        try:
            camera_intrinsics = parse_camera_intrinsics_form(intrinsics)
            return await run_in_threadpool(
                workspace.import_video,
                video.file,
                filename=video.filename or "capture.mp4",
                source=source,
                notes=notes,
                capture_session_id=capture_session_id,
                scramble=scramble,
                camera_facing=camera_facing,
                mirrored=mirrored,
                camera_intrinsics=camera_intrinsics,
                configured_fps=configured_fps,
            )
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await video.close()

    @app.post("/api/captures/import", status_code=201)
    async def import_capture(
        video: Annotated[UploadFile, File()],
        source: Annotated[str, Form()] = "import",
        notes: Annotated[str, Form()] = "",
        capture_session_id: Annotated[str, Form()] = "",
        scramble: Annotated[str, Form()] = "",
        camera_facing: Annotated[str, Form()] = "unknown",
        mirrored: Annotated[bool, Form()] = False,
        intrinsics: Annotated[str, Form()] = "",
        configured_fps: Annotated[int | None, Form()] = None,
    ) -> dict[str, object]:
        return await store_upload(
            video,
            source=source,
            notes=notes,
            capture_session_id=capture_session_id,
            scramble=scramble,
            camera_facing=camera_facing,
            mirrored=mirrored,
            intrinsics=intrinsics,
            configured_fps=configured_fps,
        )

    @app.post("/api/captures/{capture_id}/sidecars/{kind}")
    async def attach_sidecar(
        capture_id: str,
        kind: str,
        sidecar: Annotated[UploadFile, File()],
    ) -> dict[str, object]:
        try:
            return await run_in_threadpool(
                workspace.attach_json_sidecar,
                capture_id,
                sidecar.file,
                kind=kind,
                calibration_display_name=(
                    calibration_upload_display_name(sidecar.filename)
                    if kind == "calibration"
                    else None
                ),
            )
        except WorkspaceError as exc:
            status = 404 if str(exc) == "capture not found" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            await sidecar.close()

    @app.post("/api/captures/{capture_id}/sidecars/calibration/reuse")
    async def attach_reused_calibration(
        capture_id: str,
        request: Request,
        source: Annotated[str, Form()],
        source_capture_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        try:
            return await run_in_threadpool(
                workspace.attach_reused_calibration,
                capture_id,
                source=source,
                source_capture_id=source_capture_id,
                repo_root=request.app.state.settings.repo_root,
            )
        except WorkspaceError as exc:
            detail = str(exc)
            status = 404 if detail == "capture not found" or "not downloaded" in detail else 400
            raise HTTPException(status_code=status, detail=detail) from exc

    @app.post("/api/captures/{capture_id}/calibration/from-crops")
    async def attach_calibration_from_crops(
        capture_id: str,
        white: Annotated[UploadFile, File()],
        green: Annotated[UploadFile, File()],
        red: Annotated[UploadFile, File()],
        blue: Annotated[UploadFile, File()],
        orange: Annotated[UploadFile, File()],
        yellow: Annotated[UploadFile, File()],
    ) -> dict[str, object]:
        uploads = {
            "white": white,
            "green": green,
            "red": red,
            "blue": blue,
            "orange": orange,
            "yellow": yellow,
        }
        try:
            crops: dict[str, bytes] = {}
            for color, upload in uploads.items():
                payload = await upload.read(MAX_CROP_BYTES + 1)
                if len(payload) > MAX_CROP_BYTES:
                    raise DesktopCalibrationError(f"{color} crop exceeds the 2 MiB limit")
                crops[color] = payload

            try:
                receipt = next(
                    row for row in workspace.list_captures() if row.get("capture_id") == capture_id
                )
            except StopIteration as exc:
                raise WorkspaceError("capture not found") from exc
            video_sha256 = receipt.get("video", {}).get("sha256")
            if not isinstance(video_sha256, str) or not re.fullmatch(
                r"[a-f0-9]{64}",
                video_sha256,
            ):
                raise WorkspaceError("capture video receipt is invalid")
            document = await run_in_threadpool(
                build_centroids_document_from_crops,
                crops,
                capture_id=capture_id,
                video_sha256=video_sha256,
            )
            calibration_json = json.dumps(document, sort_keys=True).encode("utf-8")
            return await run_in_threadpool(
                workspace.attach_json_sidecar,
                capture_id,
                io.BytesIO(calibration_json),
                kind="calibration",
                calibration_display_name="Sampled from this video",
            )
        except DesktopCalibrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WorkspaceError as exc:
            detail = str(exc)
            status = 404 if detail == "capture not found" else 400
            raise HTTPException(status_code=status, detail=detail) from exc
        finally:
            for upload in uploads.values():
                await upload.close()

    @app.post("/api/captures/{capture_id}/seal")
    async def seal_capture(
        capture_id: str,
        purpose: Annotated[str, Form()] = "label",
    ) -> dict[str, object]:
        try:
            return workspace.seal_capture(capture_id, purpose=purpose)
        except WorkspaceError as exc:
            status = 404 if str(exc) == "capture not found" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    if frontend_index.is_file():
        assets_dir = frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="lab-web-assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def lab_web(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            candidate = (frontend_dist / path).resolve()
            try:
                candidate.relative_to(frontend_dist.resolve())
            except ValueError:
                candidate = frontend_index
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_index)

    return app


app = create_app()
