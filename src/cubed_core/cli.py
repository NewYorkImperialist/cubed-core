from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .capabilities import build_capabilities
from .model_artifacts import tracker_model_capability
from .model_packaging import ModelPackageError, package_tracker_models
from .settings import LOCAL_BROWSER_AUTH_ENV, Settings
from .tracker_runtime import probe_camera_model_cuda_session
from .workspace import Workspace, WorkspaceError


def _tcp_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 65535")
    return port


def _doctor(settings: Settings) -> int:
    report = build_capabilities(settings)
    required = report["commands"]
    prediction_valid = report["label"]["prediction"]["status"] != "misconfigured"
    cuda_model_valid = True
    providers = tuple(getattr(settings, "tracker_onnx_providers", ()))
    if "CUDAExecutionProvider" in providers:
        cuda_probe = probe_camera_model_cuda_session(
            getattr(settings, "tracker_model_manifest", None)
        )
        report["cuda_model_session"] = cuda_probe
        cuda_model_valid = bool(cuda_probe["ready"])
    print(json.dumps(report, indent=2))
    ready = required["ffmpeg"] and required["ffprobe"] and prediction_valid and cuda_model_valid
    return 0 if ready else 1


def _import_video(
    settings: Settings,
    path: Path,
    source: str,
    notes: str,
    *,
    scramble: str,
    capture_session_id: str,
    camera_facing: str,
    mirrored: bool,
) -> int:
    workspace = Workspace(
        settings.workspace,
        max_upload_bytes=settings.max_upload_bytes,
    )
    try:
        with path.open("rb") as stream:
            receipt = workspace.import_video(
                stream,
                filename=path.name,
                source=source,
                notes=notes,
                scramble=scramble,
                capture_session_id=capture_session_id,
                camera_facing=camera_facing,
                mirrored=mirrored,
            )
    except (OSError, WorkspaceError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


def _derive_240_to_120(settings: Settings, capture_id: str) -> int:
    workspace = Workspace(
        settings.workspace,
        max_upload_bytes=settings.max_upload_bytes,
    )
    try:
        receipt = workspace.derive_240_to_120(capture_id)
    except (OSError, WorkspaceError) as exc:
        print(f"derivation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


def _verify_tracker_models(settings: Settings) -> int:
    report = tracker_model_capability(settings.tracker_model_manifest)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


def _package_tracker_models(spec: Path, output_dir: Path) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        report = package_tracker_models(
            spec,
            output_dir,
            protected_roots=(repository_root,),
        )
    except ModelPackageError as exc:
        print(f"model package failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="cubed-core")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Report local CPU/GPU/media capabilities")
    subparsers.add_parser(
        "verify-tracker-models",
        help="Verify the configured tracker model manifest, bytes, and SHA-256 identities",
    )
    package_models = subparsers.add_parser(
        "package-tracker-models",
        help="Audit and package explicit camera-tracker ONNX artifacts outside Git",
    )
    package_models.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Tracker-model-package-spec-v1 JSON input",
    )
    package_models.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New package directory outside this repository",
    )
    serve = subparsers.add_parser("serve", help="Run the local API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument(
        "--port",
        type=_tcp_port,
        default=os.environ.get("CUBED_CORE_PORT") or "8000",
        help="API port (default: CUBED_CORE_PORT or 8000)",
    )
    serve.add_argument(
        "--browser-port",
        type=_tcp_port,
        help=(
            "Host port to print in the generated Admin workbench URL "
            "(defaults to --port; useful for container port publishing)"
        ),
    )
    serve.add_argument(
        "--reload", action="store_true", help="Reload when local source files change"
    )
    serve.add_argument(
        "--allow-network",
        action="store_true",
        help="Required when binding outside localhost; the API has no user accounts",
    )
    import_video = subparsers.add_parser("import-video", help="Import a high-frame-rate video")
    import_video.add_argument("path", type=Path)
    import_video.add_argument(
        "--source",
        choices=["browser", "external-camera", "import"],
        default="import",
    )
    import_video.add_argument("--notes", default="")
    import_video.add_argument("--scramble", default="")
    import_video.add_argument("--capture-session-id", default="")
    import_video.add_argument(
        "--camera-facing",
        choices=["front", "back", "external", "unknown"],
        default="unknown",
    )
    import_video.add_argument("--mirrored", action="store_true")
    derivative = subparsers.add_parser(
        "derive-240-to-120",
        help="Create a linked every-other-frame 120 fps capture from a verified 240 fps draft",
    )
    derivative.add_argument("capture_id", help="32-hex workspace capture ID")
    args = parser.parse_args()
    if args.command == "package-tracker-models":
        return _package_tracker_models(args.spec, args.output_dir)
    settings = Settings.from_env()
    if args.command == "doctor":
        return _doctor(settings)
    if args.command == "verify-tracker-models":
        return _verify_tracker_models(settings)
    if args.command == "import-video":
        return _import_video(
            settings,
            args.path,
            args.source,
            args.notes,
            scramble=args.scramble,
            capture_session_id=args.capture_session_id,
            camera_facing=args.camera_facing,
            mirrored=args.mirrored,
        )
    if args.command == "derive-240-to-120":
        return _derive_240_to_120(settings, args.capture_id)
    if args.command == "serve":
        import uvicorn

        loopback_host = args.host in {"127.0.0.1", "localhost", "::1"}
        if not loopback_host and not args.allow_network:
            parser.error("non-loopback --host requires --allow-network")
        local_browser_auth = loopback_host and not args.allow_network
        os.environ[LOCAL_BROWSER_AUTH_ENV] = "1" if local_browser_auth else "0"
        if not settings.admin_token:
            os.environ["CUBED_CORE_ADMIN_TOKEN"] = secrets.token_urlsafe(32)
        settings = Settings.from_env()
        local_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        browser_host = f"[{local_host}]" if ":" in local_host else local_host
        browser_port = args.browser_port or args.port
        if local_browser_auth:
            print(
                f"Workbench: http://{browser_host}:{browser_port}/",
                flush=True,
            )
        else:
            print(
                f"Admin workbench: "
                f"http://{browser_host}:{browser_port}/#admin={settings.admin_token}",
                flush=True,
            )
        uvicorn.run("cubed_core.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
