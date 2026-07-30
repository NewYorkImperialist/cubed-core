from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .color_calibration import (
    COLOR_CALIBRATION_SCHEMA,
    COLOR_CENTROIDS_SCHEMA,
    ColorCalibrationError,
    ColorCentroidsError,
    validate_color_calibration,
    validate_color_centroids,
)
from .media import (
    is_native_240_capture_frame_rate,
    is_standard_capture_frame_rate,
    is_standard_capture_resolution,
)

RUNTIME_MANIFEST = Path("config/decode-runtime-v1.json")
PROFILE_NAME = "local_camera_v1"
_MOVE_TOKEN = re.compile(r"[UDLRFB](?:'|2)?")
_RECORDING_ID = re.compile(r"[a-f0-9]{32}")
_SHA256 = re.compile(r"[a-f0-9]{64}")


class DecodeContractError(ValueError):
    pass


def posix_cksum(payload: bytes) -> int:
    """Return the POSIX ``cksum`` CRC, including its encoded length."""

    crc = 0
    polynomial = 0x04C11DB7

    def update(value: int, octet: int) -> int:
        value ^= octet << 24
        for _ in range(8):
            value = (
                ((value << 1) ^ polynomial) & 0xFFFFFFFF
                if value & 0x80000000
                else (value << 1) & 0xFFFFFFFF
            )
        return value

    for byte in payload:
        crc = update(crc, byte)
    length = len(payload)
    while length:
        crc = update(crc, length & 0xFF)
        length >>= 8
    return (~crc) & 0xFFFFFFFF


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DecodeContractError(f"{name} must be a JSON object")
    return value


def _profile_hash_input(profile: dict[str, Any]) -> str:
    flags = profile.get("flags")
    env_stamp = profile.get("behavior_env_stamp")
    if (
        not isinstance(flags, list)
        or not flags
        or any(not isinstance(token, str) or not token for token in flags)
        or not isinstance(env_stamp, str)
        or not env_stamp
    ):
        raise DecodeContractError("decode profile flags or behavior env stamp are invalid")
    return f"{' '.join(flags)} ENV:{env_stamp}"


def load_runtime_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root.resolve() / RUNTIME_MANIFEST
    if path.is_symlink():
        raise DecodeContractError("decode runtime manifest may not be a symlink")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecodeContractError("decode runtime manifest is unavailable") from exc
    manifest = _require_object(manifest, name="decode runtime manifest")
    if manifest.get("schema") != "cubed-core/decode-runtime" or manifest.get("schema_version") != 1:
        raise DecodeContractError("unsupported decode runtime manifest")
    profiles = _require_object(manifest.get("profiles"), name="decode profiles")
    for profile_id in ("canonical_eval_reference", PROFILE_NAME):
        profile = _require_object(profiles.get(profile_id), name=f"decode profile {profile_id}")
        generated_input = _profile_hash_input(profile)
        if profile.get("cfg_hash_input") != generated_input:
            raise DecodeContractError(f"decode profile {profile_id} hash input has drifted")
        expected_hash = profile.get("cfg_hash")
        actual_hash = str(posix_cksum(generated_input.encode("utf-8")))
        if (
            profile.get("cfg_hash_algorithm") != "posix-cksum"
            or not isinstance(expected_hash, str)
            or actual_hash != expected_hash
        ):
            raise DecodeContractError(f"decode profile {profile_id} CFG_HASH has drifted")
    requirements = manifest.get("runtime_requirements")
    if not isinstance(requirements, list) or not requirements:
        raise DecodeContractError("decode runtime requirements are unavailable")
    seen: set[str] = set()
    for requirement in requirements:
        requirement = _require_object(requirement, name="decode runtime requirement")
        requirement_id = requirement.get("id")
        if (
            not isinstance(requirement_id, str)
            or not requirement_id
            or requirement_id in seen
            or not isinstance(requirement.get("path"), str)
            or not requirement["path"]
            or Path(requirement["path"]).is_absolute()
            or ".." in Path(requirement["path"]).parts
        ):
            raise DecodeContractError("decode runtime requirement is invalid")
        seen.add(requirement_id)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(base: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DecodeContractError("receipted path escapes its workspace")
    candidate = base / relative
    if candidate.is_symlink():
        raise DecodeContractError("receipted file may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DecodeContractError("receipted file is unavailable") from exc
    if not resolved.is_file():
        raise DecodeContractError("receipted path is not a file")
    return resolved


def _check(
    checks: list[dict[str, str]],
    check_id: str,
    status: str,
    detail: str,
    *,
    path: str | None = None,
) -> None:
    item = {"id": check_id, "status": status, "detail": detail}
    if path is not None:
        item["path"] = path
    checks.append(item)


def _capture_recording_id(capture: dict[str, Any]) -> str | None:
    recording_id = capture.get("recording_id") or capture.get("capture_id")
    return recording_id if isinstance(recording_id, str) else None


def _validate_capture_files(
    capture: dict[str, Any],
    workspace_root: Path | None,
    checks: list[dict[str, str]],
) -> None:
    recording_id = _capture_recording_id(capture)
    if workspace_root is None:
        _check(
            checks,
            "capture.files",
            "fail",
            "workspace root is required to verify the receipted video and calibration bytes",
        )
        return
    if recording_id is None or not _RECORDING_ID.fullmatch(recording_id):
        _check(checks, "capture.files", "fail", "recording id is invalid")
        return
    workspace_root = workspace_root.resolve()
    capture_dir = workspace_root / "captures" / recording_id
    if capture_dir.is_symlink():
        _check(checks, "capture.files", "fail", "capture directory may not be a symlink")
        return
    try:
        capture_dir.resolve(strict=True).relative_to(workspace_root)
    except (OSError, ValueError):
        _check(checks, "capture.files", "fail", "capture directory is unavailable")
        return

    file_specs = (
        ("video", capture.get("video"), workspace_root),
        ("calibration", capture.get("calibration"), capture_dir),
    )
    for label, receipt_value, base in file_specs:
        if not isinstance(receipt_value, dict):
            _check(checks, f"capture.{label}", "fail", f"{label} receipt is missing")
            continue
        relative_path = receipt_value.get("path")
        expected_sha = receipt_value.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(expected_sha, str)
            or not _SHA256.fullmatch(expected_sha)
        ):
            _check(checks, f"capture.{label}", "fail", f"{label} receipt is invalid")
            continue
        try:
            path = _resolve_file(base, relative_path)
            actual_sha = _sha256(path)
        except (OSError, DecodeContractError) as exc:
            _check(checks, f"capture.{label}", "fail", str(exc), path=relative_path)
            continue
        if actual_sha != expected_sha:
            _check(
                checks,
                f"capture.{label}",
                "fail",
                f"{label} SHA-256 does not match its receipt",
                path=relative_path,
            )
            continue
        if label == "video":
            expected_bytes = receipt_value.get("bytes")
            if (
                not isinstance(expected_bytes, int)
                or isinstance(expected_bytes, bool)
                or expected_bytes != path.stat().st_size
            ):
                _check(
                    checks,
                    "capture.video",
                    "fail",
                    "video byte count does not match its receipt",
                    path=relative_path,
                )
                continue
        else:
            try:
                calibration = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                _check(
                    checks,
                    "capture.calibration",
                    "fail",
                    "calibration sidecar is not valid JSON",
                    path=relative_path,
                )
                continue
            calibration_schema = (
                calibration.get("schema") if isinstance(calibration, dict) else None
            )
            if not isinstance(calibration, dict) or calibration_schema not in (
                COLOR_CALIBRATION_SCHEMA,
                COLOR_CENTROIDS_SCHEMA,
            ):
                _check(
                    checks,
                    "capture.calibration",
                    "fail",
                    "calibration sidecar does not declare color-calibration-v1 or "
                    "color-centroids-v1",
                    path=relative_path,
                )
                continue
            if calibration.get("schema_version") != 1:
                _check(
                    checks,
                    "capture.calibration",
                    "fail",
                    "calibration sidecar schema_version must equal 1",
                    path=relative_path,
                )
                continue
            if calibration_schema == COLOR_CALIBRATION_SCHEMA:
                try:
                    validate_color_calibration(calibration)
                except ColorCalibrationError as exc:
                    _check(
                        checks,
                        "capture.calibration",
                        "fail",
                        f"calibration sidecar is invalid: {exc}",
                        path=relative_path,
                    )
                    continue
            else:
                try:
                    validate_color_centroids(calibration)
                except ColorCentroidsError as exc:
                    _check(
                        checks,
                        "capture.calibration",
                        "fail",
                        f"calibration sidecar is invalid: {exc}",
                        path=relative_path,
                    )
                    continue
        _check(
            checks,
            f"capture.{label}",
            "pass",
            f"{label} bytes match the SHA-256 receipt",
            path=relative_path,
        )


def build_decode_preflight(
    capture_value: Any,
    *,
    repo_root: Path,
    workspace_root: Path | None,
) -> dict[str, Any]:
    """Inspect one capture without invoking a decoder or loading model code."""

    capture = _require_object(capture_value, name="capture receipt")
    manifest = load_runtime_manifest(repo_root)
    profile = manifest["profiles"][PROFILE_NAME]
    checks: list[dict[str, str]] = []

    recording_id = _capture_recording_id(capture)
    if (
        capture.get("schema") == "cubed-core/capture-bundle"
        and capture.get("schema_version") == 1
        and isinstance(recording_id, str)
        and _RECORDING_ID.fullmatch(recording_id)
    ):
        _check(checks, "capture.contract", "pass", "capture-bundle-v1 receipt accepted")
    else:
        _check(
            checks,
            "capture.contract",
            "fail",
            "capture must be a capture-bundle-v1 receipt with a 32-hex recording id",
        )

    if capture.get("state") == "sealed" and capture.get("seal_purpose") == "decode":
        _check(checks, "capture.seal", "pass", "capture is immutable and sealed for decode")
    else:
        _check(
            checks,
            "capture.seal",
            "fail",
            "capture must be sealed with seal_purpose=decode before execution",
        )

    solve = capture.get("solve")
    solve = solve if isinstance(solve, dict) else {}
    scramble = solve.get("scramble")
    scramble_tokens = scramble.split() if isinstance(scramble, str) else []
    if scramble_tokens and all(_MOVE_TOKEN.fullmatch(token) for token in scramble_tokens):
        _check(checks, "capture.scramble", "pass", "starting scramble is canonical")
    else:
        _check(checks, "capture.scramble", "fail", "a canonical starting scramble is required")

    video = capture.get("video")
    video = video if isinstance(video, dict) else {}
    fps = video.get("actual_fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        _check(checks, "capture.frame-rate", "fail", "measured frame rate is required")
    elif is_standard_capture_frame_rate(fps):
        _check(checks, "capture.frame-rate", "pass", "120 fps operating regime accepted")
    elif is_native_240_capture_frame_rate(fps):
        _check(
            checks,
            "capture.frame-rate",
            "fail",
            "native 220–242 fps cannot run directly; preserve it and create a "
            "linked 120 fps capture with derive-240-to-120",
        )
    else:
        _check(
            checks,
            "capture.frame-rate",
            "warning",
            "measured cadence is outside the recommended 110 through 121 fps "
            "profile; decode is allowed but may be less reliable",
        )

    encoded_width = video.get("encoded_width")
    encoded_height = video.get("encoded_height")
    if (
        type(encoded_width) is not int
        or encoded_width <= 0
        or type(encoded_height) is not int
        or encoded_height <= 0
    ):
        _check(
            checks,
            "capture.resolution",
            "fail",
            "measured encoded width and height are required",
        )
    elif is_standard_capture_resolution(encoded_width, encoded_height):
        _check(
            checks,
            "capture.resolution",
            "pass",
            "encoded short edge meets the 1080-pixel decode floor",
        )
    else:
        _check(
            checks,
            "capture.resolution",
            "warning",
            "encoded short edge is below the recommended 1080-pixel profile; "
            "decode is allowed but face reads may be less reliable",
        )

    _validate_capture_files(capture, workspace_root, checks)

    teacher = capture.get("teacher")
    if isinstance(teacher, dict):
        _check(
            checks,
            "teacher.containment",
            "pass",
            "capture metadata declares an evaluation-only teacher record; "
            "the camera-only request does not consume it",
        )
    else:
        _check(
            checks,
            "teacher.containment",
            "pass",
            "capture metadata declares no teacher record; "
            "the camera-only request does not require one",
        )

    _check(
        checks,
        "config.cfg-hash",
        "pass",
        f"{profile['name']} configured at CFG_HASH {profile['cfg_hash']}",
    )
    for command in manifest.get("required_commands", []):
        executable = shutil.which(command)
        _check(
            checks,
            f"command.{command}",
            "pass" if executable else "fail",
            f"{command} is available" if executable else f"{command} is required but unavailable",
            path=executable,
        )

    repo_root = repo_root.resolve()
    for requirement in manifest["runtime_requirements"]:
        relative_path = requirement["path"]
        publication_status = requirement.get("publication_status")
        candidate = repo_root / relative_path
        if publication_status != "available":
            _check(
                checks,
                f"runtime.{requirement['id']}",
                "fail",
                (
                    f"{requirement['id']} is {publication_status}; "
                    f"required for {requirement['purpose']}"
                ),
                path=relative_path,
            )
        elif not candidate.exists() or candidate.is_symlink():
            _check(
                checks,
                f"runtime.{requirement['id']}",
                "fail",
                f"{requirement['id']} is declared available but its path is missing",
                path=relative_path,
            )
        else:
            _check(
                checks,
                f"runtime.{requirement['id']}",
                "pass",
                f"{requirement['id']} is available",
                path=relative_path,
            )

    if manifest.get("execution_enabled") is True:
        _check(checks, "runtime.execution", "pass", "local decoder execution is enabled")
    else:
        _check(
            checks,
            "runtime.execution",
            "fail",
            "execution is disabled until the audited decoder/runtime requirements are published",
        )

    missing = [item["id"] for item in checks if item["status"] == "fail"]
    warnings = [item["detail"] for item in checks if item["status"] == "warning"]
    ready = not missing and manifest.get("execution_enabled") is True
    return {
        "schema": "cubed-core/decode-preflight",
        "schema_version": 1,
        "recording_id": recording_id,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "execution_allowed": ready,
        "profile": PROFILE_NAME,
        "config": {
            "name": profile["name"],
            "cfg_hash": profile["cfg_hash"],
            "cfg_hash_algorithm": profile["cfg_hash_algorithm"],
            "derived_from_cfg_hash": profile["derived_from_cfg_hash"],
            "evidence_status": profile["evidence_status"],
        },
        "checks": checks,
        "missing": missing,
        "warnings": warnings,
    }


def load_workspace_capture(workspace_root: Path, recording_id: str) -> dict[str, Any]:
    if not _RECORDING_ID.fullmatch(recording_id):
        raise DecodeContractError("capture id must contain exactly 32 lowercase hex characters")
    root = workspace_root.resolve()
    capture_dir = root / "captures" / recording_id
    if capture_dir.is_symlink():
        raise DecodeContractError("capture directory may not be a symlink")
    metadata = _resolve_file(capture_dir, "capture.json")
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeContractError("capture metadata is unavailable") from exc
    return _require_object(value, name="capture receipt")


def _inferred_workspace(capture_json: Path) -> Path | None:
    resolved = capture_json.resolve()
    if (
        resolved.name == "capture.json"
        and resolved.parent.parent.name == "captures"
        and _RECORDING_ID.fullmatch(resolved.parent.name)
    ):
        return resolved.parent.parent.parent
    return None


def _render_text(result: dict[str, Any]) -> str:
    """Render one preflight result as a short human-readable check list."""

    config = result["config"]
    lines = [
        f"recording {result['recording_id']}: {result['status']} "
        f"(profile {config['name']}, CFG_HASH {config['cfg_hash']})",
        "checks:",
    ]
    for check in result["checks"]:
        location = f" ({check['path']})" if check.get("path") else ""
        lines.append(f"  [{check['status'].upper():7}] {check['id']}: {check['detail']}{location}")
    if result["warnings"]:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in result["warnings"])
    if result["missing"]:
        lines.append("blocked on: " + ", ".join(result["missing"]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decode-preflight",
        description="Validate a local capture and decoder runtime without starting a decode.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-id")
    source.add_argument("--capture-json", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="json (default, machine-readable) or text (one line per check)",
    )
    args = parser.parse_args(argv)
    try:
        if args.capture_id:
            if args.workspace is None:
                raise DecodeContractError("--workspace is required with --capture-id")
            capture = load_workspace_capture(args.workspace, args.capture_id)
            workspace = args.workspace
        else:
            capture_path = args.capture_json.resolve()
            capture = _require_object(
                json.loads(capture_path.read_text(encoding="utf-8")),
                name="capture receipt",
            )
            workspace = args.workspace or _inferred_workspace(capture_path)
        result = build_decode_preflight(
            capture,
            repo_root=args.repo_root,
            workspace_root=workspace,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DecodeContractError) as exc:
        print(f"decode preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.format == "text":
        print(_render_text(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
