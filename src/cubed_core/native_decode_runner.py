"""Native decode runner: orchestrate the local_camera_v1 pipeline for one job.

Invoked by ``decode_jobs`` as ``--request <job-request.json> --output
<decode-result.json>``. It sequences the four research stages that turn a sealed
capture into a reconstruction, in the same order and with the same invocation
shapes as ``scripts/run_research_reads.sh``:

1. ``reads``     scripts/geo_read.py          video plus calibration to a reads pickle
2. ``events``    scripts/gen_motion_events.py reads to camera-only motion events
3. ``alignfeat`` scripts/extract_alignfeat.py video to an alignment feature track
4. ``decode``    scripts/run_research_decode.sh reads plus events to the result document

Each stage boundary prints one flushed ``[cubed-core:stage]`` marker so the
server can derive progress from the captured log.

Scratch layout: the generators resolve several inputs by tag under the system
temporary directory, and ``gen_motion_events.py`` hardcodes ``reads_<tag>_v2.pkl``.
The runner therefore uses the 32-hex job id as the tag so two concurrent jobs can
never collide, copies the produced artifacts into the job directory, and removes
the scratch files on exit whether the run succeeded or failed.

This module only orchestrates. It never asserts anything about decode quality:
the result document it collects is reconstruction evidence, and the replay check
that qualifies it belongs to the server.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

RUNNER_ERROR_SCHEMA = "cubed-core/decode-runner-error-v1"
DECODE_REQUEST_SCHEMA = "cubed-core/decode-job-request-v1"
REQUEST_MAX_BYTES = 1024 * 1024
CALIBRATION_MAX_BYTES = 8 * 1024**2
WORKSTATION_MAX_BYTES = 32 * 1024**2
NVDEC_POLICIES = ("auto", "off", "require")
_IDENTIFIER = re.compile(r"[a-f0-9]{32}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_COLOR_ORDER = ("white", "green", "red", "blue", "orange", "yellow")


class NativeDecodeRunnerError(ValueError):
    """A deterministic failure that is safe to expose in the runner log."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_status: int,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_status = exit_status
        self.details = dict(details or {})

    def public(self) -> dict[str, Any]:
        return {
            "schema": RUNNER_ERROR_SCHEMA,
            "schema_version": 1,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


def _error(
    code: str,
    message: str,
    *,
    exit_status: int,
    details: Mapping[str, Any] | None = None,
) -> NativeDecodeRunnerError:
    return NativeDecodeRunnerError(code, message, exit_status=exit_status, details=details)


def emit_stage(token: str, *, current: int | None = None, total: int | None = None) -> None:
    """Print one flushed ``[cubed-core:stage]`` marker at a pipeline boundary.

    The server scans captured stdout for these lines, so every marker must be a
    single self-contained line and must be flushed immediately for the server to
    observe it while the job is still running.
    """

    if current is not None and total is not None:
        print(f"[cubed-core:stage] {token} {current}/{total}", flush=True)
    else:
        print(f"[cubed-core:stage] {token}", flush=True)


@dataclass(frozen=True)
class DecodeRunContext:
    """The complete camera-only input surface resolved from one request."""

    job_id: str
    capture_id: str
    job_dir: Path
    output_path: Path
    video_path: Path
    calibration_path: Path
    scramble: str
    runtime_assets: dict[str, dict[str, Any]]
    workstation_context: dict[str, Any] | None = None


def _parse_cli(argv: Sequence[str]) -> tuple[Path, Path]:
    if len(argv) != 4:
        raise _error(
            "invalid_arguments",
            "expected exactly --request <path> and --output <path>",
            exit_status=2,
        )
    values: dict[str, str] = {}
    for index in range(0, len(argv), 2):
        flag = argv[index]
        value = argv[index + 1]
        if flag not in {"--request", "--output"} or flag in values:
            raise _error(
                "invalid_arguments",
                "expected exactly --request <path> and --output <path>",
                exit_status=2,
            )
        if not value or "\0" in value:
            raise _error(
                "invalid_arguments",
                f"{flag} must name a non-empty local path",
                exit_status=2,
            )
        values[flag] = value
    request_path = Path(values["--request"])
    output_path = Path(values["--output"])
    if not request_path.is_absolute() or not output_path.is_absolute():
        raise _error("unsafe_path", "request and output paths must be absolute", exit_status=3)
    if output_path.parent != request_path.parent:
        raise _error(
            "unsafe_path",
            "the decode result must be written into the job directory",
            exit_status=3,
        )
    if request_path.parent.is_symlink() or output_path.is_symlink():
        raise _error(
            "unsafe_path",
            "the decode job directory and its result target may not be symlinks",
            exit_status=3,
        )
    return request_path, output_path


def _read_bounded_text(path: Path, *, description: str, maximum_bytes: int) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError(f"{description} is not a regular file")
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise OSError(f"{description} is empty or exceeds its {maximum_bytes}-byte limit")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _error("unreadable_input", f"{description} is unavailable", exit_status=4) from exc


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _existing_file(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error("invalid_request", f"{field} must be a path", exit_status=4)
    candidate = Path(value)
    if not candidate.is_absolute():
        raise _error("unsafe_path", f"{field} must be an absolute path", exit_status=3)
    if candidate.is_symlink() or not candidate.is_file():
        raise _error("unreadable_input", f"{field} is unavailable", exit_status=4)
    return candidate


def _identity_bound_file(
    value: Any,
    expected_sha256: Any,
    *,
    field: str,
    maximum_bytes: int,
) -> Path:
    """Resolve and hash one request input without following a planted symlink."""

    path = _existing_file(value, field=field)
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise _error(
            "invalid_request",
            f"{field} SHA-256 is invalid",
            exit_status=4,
        )
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise OSError(f"{field} is empty or exceeds its safety limit")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise _error(
            "unreadable_input",
            f"{field} is unavailable",
            exit_status=4,
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise _error(
            "artifact_mismatch",
            f"{field} changed while it was being verified",
            exit_status=5,
        )
    if digest.hexdigest() != expected_sha256:
        raise _error(
            "artifact_mismatch",
            f"{field} does not match the request digest",
            exit_status=5,
        )
    return path


def _load_workstation_context(value: Any) -> dict[str, Any] | None:
    """Validate the request's path-free portable media context, when present."""

    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"video", "warnings"}:
        raise _error(
            "invalid_request",
            "decode request workstation context is invalid",
            exit_status=4,
        )
    video = value.get("video")
    required_video = {"sha256", "bytes", "fps", "frame_count", "width", "height"}
    if not isinstance(video, dict) or set(video) != required_video:
        raise _error(
            "invalid_request",
            "decode request workstation video identity is invalid",
            exit_status=4,
        )
    if (
        not isinstance(video["sha256"], str)
        or not _SHA256.fullmatch(video["sha256"])
        or type(video["bytes"]) is not int
        or video["bytes"] < 1
        or isinstance(video["fps"], bool)
        or not isinstance(video["fps"], (int, float))
        or not math.isfinite(float(video["fps"]))
        or not 0 < video["fps"] <= 1000
        or type(video["frame_count"]) is not int
        or not 1 <= video["frame_count"] <= 10_000_000
        or type(video["width"]) is not int
        or not 1 <= video["width"] <= 100_000
        or type(video["height"]) is not int
        or not 1 <= video["height"] <= 100_000
    ):
        raise _error(
            "invalid_request",
            "decode request workstation video identity is invalid",
            exit_status=4,
        )
    warnings = value.get("warnings")
    if not isinstance(warnings, list) or len(warnings) > 100:
        raise _error(
            "invalid_request",
            "decode request workstation warnings are invalid",
            exit_status=4,
        )
    for warning in warnings:
        if (
            not isinstance(warning, dict)
            or set(warning) != {"code", "message"}
            or not isinstance(warning["code"], str)
            or not warning["code"]
            or len(warning["code"]) > 200
            or not isinstance(warning["message"], str)
            or not warning["message"]
            or len(warning["message"]) > 2000
        ):
            raise _error(
                "invalid_request",
                "decode request workstation warnings are invalid",
                exit_status=4,
            )
    return {
        "video": dict(video),
        "warnings": [dict(warning) for warning in warnings],
    }


def _load_context(request_path: Path, output_path: Path) -> DecodeRunContext:
    text = _read_bounded_text(
        request_path,
        description="decode request",
        maximum_bytes=REQUEST_MAX_BYTES,
    )
    try:
        request = json.loads(text, parse_constant=_reject_nonfinite_json)
    except ValueError as exc:
        raise _error("invalid_request", "decode request is not valid JSON", exit_status=4) from exc
    if not isinstance(request, dict):
        raise _error("invalid_request", "decode request must be an object", exit_status=4)
    if request.get("schema") != DECODE_REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise _error(
            "invalid_request",
            f"decode request must declare {DECODE_REQUEST_SCHEMA} version 1",
            exit_status=4,
        )
    job_id = request.get("job_id")
    capture_id = request.get("capture_id")
    if (
        not isinstance(job_id, str)
        or not _IDENTIFIER.fullmatch(job_id)
        or not isinstance(capture_id, str)
        or not _IDENTIFIER.fullmatch(capture_id)
    ):
        raise _error("invalid_request", "decode request identity is invalid", exit_status=4)
    if request_path.parent.name != job_id:
        raise _error(
            "unsafe_path",
            "the decode request must live in its own job directory",
            exit_status=3,
        )
    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise _error("invalid_request", "decode request inputs are invalid", exit_status=4)
    scramble = inputs.get("scramble")
    if not isinstance(scramble, str) or not scramble.strip():
        raise _error("invalid_request", "decode request scramble is invalid", exit_status=4)

    runtime_assets: dict[str, dict[str, Any]] = {}
    declared = request.get("runtime_assets")
    if not isinstance(declared, list) or not declared:
        raise _error("invalid_request", "decode request runtime assets are invalid", exit_status=4)
    for asset in declared:
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("id"), str)
            or not isinstance(asset.get("path"), str)
            or not isinstance(asset.get("sha256"), str)
            or not _SHA256.fullmatch(asset["sha256"])
            or type(asset.get("bytes")) is not int
        ):
            raise _error(
                "invalid_request",
                "decode request runtime assets are invalid",
                exit_status=4,
            )
        runtime_assets[asset["id"]] = dict(asset)

    calibration_path = _identity_bound_file(
        inputs.get("calibration"),
        inputs.get("calibration_sha256"),
        field="decode request calibration",
        maximum_bytes=CALIBRATION_MAX_BYTES,
    )
    return DecodeRunContext(
        job_id=job_id,
        capture_id=capture_id,
        job_dir=request_path.parent,
        output_path=output_path,
        video_path=_existing_file(inputs.get("video"), field="decode request video"),
        calibration_path=calibration_path,
        scramble=scramble.strip(),
        runtime_assets=runtime_assets,
        workstation_context=_load_workstation_context(request.get("workstation_context")),
    )


def _repo_root() -> Path:
    configured = os.environ.get("CUBED_CORE_REPO_ROOT", "")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not (root / "scripts" / "run_research_decode.sh").is_file():
        raise _error(
            "missing_component",
            "the research decode pipeline is not present in this checkout",
            exit_status=5,
            details={"repo_root": str(root)},
        )
    return root


def _verify_runtime_assets(context: DecodeRunContext, repo_root: Path) -> dict[str, Path]:
    """Re-hash every declared model artifact before any heavy stage begins."""

    resolved: dict[str, Path] = {}
    for asset_id, asset in sorted(context.runtime_assets.items()):
        path = (repo_root / str(asset["path"])).resolve()
        if path.is_symlink() or not path.is_file():
            raise _error(
                "missing_artifact",
                f"decode runtime asset {asset_id} is unavailable",
                exit_status=5,
                details={"id": asset_id, "path": str(asset["path"])},
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset["sha256"] or path.stat().st_size != asset["bytes"]:
            raise _error(
                "artifact_mismatch",
                f"decode runtime asset {asset_id} does not match the request digest",
                exit_status=5,
                details={"id": asset_id, "path": str(asset["path"])},
            )
        resolved[asset_id] = path
    for required in ("pose-model", "alignment-model", "read-trust-model"):
        if required not in resolved:
            raise _error(
                "missing_artifact",
                f"the decode request does not declare the required {required}",
                exit_status=5,
            )
    return resolved


CANONICAL_TRUST_RELATIVE = Path("datasets/read_trust/trust_v1_numpy.npz")


def _materialize_canonical_trust_model(repo_root: Path, verified_asset: Path) -> None:
    """Stage the verified trust model at the canonical repo-relative path.

    The decode configuration hashes its flag tokens, and the canonical
    CFG_HASH embeds the literal ``datasets/read_trust/trust_v1_numpy.npz``
    token. Overriding the path through the environment changes the stamp, so
    the runner copies the already-verified asset to the canonical location
    and lets the decode script use its default token.
    """

    destination = repo_root / CANONICAL_TRUST_RELATIVE
    wanted = hashlib.sha256(verified_asset.read_bytes()).hexdigest()
    if destination.exists():
        present = hashlib.sha256(destination.read_bytes()).hexdigest()
        if present == wanted:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".trust-stage-", delete=False
    ) as handle:
        staged = Path(handle.name)
    try:
        shutil.copyfile(verified_asset, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def _write_centroids(context: DecodeRunContext) -> Path:
    """Project the capture's color-calibration sidecar into the reads centroid file.

    ``scripts/geo_read.py`` loads ``--centroids-json`` through
    ``calib_util.load_centroids``, which expects a flat ``{color: [L, a, b]}``
    document. The workspace stores either the richer
    ``cubed-core/color-calibration-v1`` sidecar (measured, paired-camera
    calibration) or the additive ``cubed-core/color-centroids-v1`` sidecar
    (bare Lab centroids imported from a released decode-support asset). Both
    carry the same flat ``centroids`` map, so the projection is identical;
    only the schema declaration differs.
    """

    text = _read_bounded_text(
        context.calibration_path,
        description="capture calibration",
        maximum_bytes=CALIBRATION_MAX_BYTES,
    )
    try:
        calibration = json.loads(text, parse_constant=_reject_nonfinite_json)
    except ValueError as exc:
        raise _error(
            "invalid_calibration",
            "capture calibration is not valid JSON",
            exit_status=4,
        ) from exc
    if not isinstance(calibration, dict) or calibration.get("schema") not in (
        "cubed-core/color-calibration-v1",
        "cubed-core/color-centroids-v1",
    ):
        raise _error(
            "invalid_calibration",
            "capture calibration must be a cubed-core/color-calibration-v1 or "
            "cubed-core/color-centroids-v1 sidecar",
            exit_status=4,
        )
    centroids = calibration.get("centroids")
    if not isinstance(centroids, dict):
        raise _error("invalid_calibration", "capture calibration has no centroids", exit_status=4)
    projected: dict[str, list[float]] = {}
    for color in _COLOR_ORDER:
        value = centroids.get(color)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise _error(
                "invalid_calibration",
                f"capture calibration centroid for {color} is invalid",
                exit_status=4,
            )
        projected[color] = [float(item) for item in value]
    destination = context.job_dir / "centroids.json"
    destination.write_text(json.dumps(projected, indent=0, sort_keys=True), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def _nvdec_policy() -> str:
    policy = os.environ.get("CUBED_NVDEC", "auto").strip() or "auto"
    if policy not in NVDEC_POLICIES:
        raise _error(
            "invalid_configuration",
            "CUBED_NVDEC must be auto, off, or require",
            exit_status=6,
        )
    return policy


def _prepare_workstation_payload(
    context: DecodeRunContext,
    *,
    workstation_path: Path,
    alignfeat_path: Path,
) -> bool:
    """Finish the same-pass browser projection without decoding video again."""

    portable = context.workstation_context
    if portable is None:
        return False
    warning_rows = [dict(row) for row in portable["warnings"]]
    value: dict[str, Any]
    staged_video: dict[str, Any] | None = None
    try:
        text = _read_bounded_text(
            workstation_path,
            description="workstation stage",
            maximum_bytes=WORKSTATION_MAX_BYTES,
        )
        loaded = json.loads(
            text,
            parse_constant=_reject_nonfinite_json,
        )
        if not isinstance(loaded, dict) or not isinstance(loaded.get("frames"), dict):
            raise ValueError("workstation stage must contain frames")
        if len(loaded["frames"]) > 250_000:
            raise ValueError("workstation stage contains too many frames")
        value = loaded
        staged_video = dict(loaded["video"]) if isinstance(loaded.get("video"), dict) else None
        staged_warnings = value.get("warnings")
        if isinstance(staged_warnings, list):
            warning_rows.extend(
                row
                for row in staged_warnings
                if isinstance(row, dict)
                and isinstance(row.get("code"), str)
                and isinstance(row.get("message"), str)
            )
    except Exception as exc:
        value = {"frames": {}}
        warning_rows.append(
            {
                "code": "workstation.frame-evidence-unavailable",
                "message": (
                    "Per-frame viewer evidence could not be projected from this run "
                    f"({type(exc).__name__})."
                ),
            }
        )

    video = dict(portable["video"])
    # Keep container-reported source metadata alongside the decoded coordinate
    # space below. OpenCV can apply rotation and can report a decoded count or
    # rate that differs slightly from the container probe; neither is a change
    # to the immutable source bytes.
    video["encoded"] = {
        "fps": video["fps"],
        "frame_count": video["frame_count"],
        "width": video["width"],
        "height": video["height"],
    }
    # These fields describe the decoded coordinate space that produced the
    # overlay quads and timeline. OpenCV may apply container rotation while
    # decoding, so they can legitimately differ from ``video.encoded``.
    if staged_video is not None:
        if (
            isinstance(staged_video.get("fps"), (int, float))
            and not isinstance(staged_video["fps"], bool)
            and math.isfinite(float(staged_video["fps"]))
            and 0 < float(staged_video["fps"]) <= 1000
        ):
            video["fps"] = float(staged_video["fps"])
        for field, maximum in (
            ("frame_count", 10_000_000),
            ("width", 100_000),
            ("height", 100_000),
        ):
            staged_value = staged_video.get(field)
            if type(staged_value) is int and 1 <= staged_value <= maximum:
                video[field] = staged_value
    value.update(
        {
            "schema": "cubed-core/decode-workstation-v1",
            "schema_version": 1,
            "video": video,
            "initialization": {"scramble": context.scramble},
            "window": [0, video["frame_count"] - 1],
        }
    )
    frames = value["frames"]
    try:
        import numpy as np

        with np.load(alignfeat_path, allow_pickle=False) as aligned:
            frame_values = aligned["frame"]
            probabilities = aligned["aligned"]
            if len(frame_values) != len(probabilities):
                raise ValueError("alignment arrays have different lengths")
            rows = sorted(
                (int(frame), float(probability))
                for frame, probability in zip(
                    frame_values,
                    probabilities,
                    strict=True,
                )
                if 0 <= int(frame) < video["frame_count"] and math.isfinite(float(probability))
            )
        streak = 0
        previous = None
        for frame, probability in rows:
            key = str(frame)
            entry = frames.setdefault(key, {"motion": None})
            if not isinstance(entry, dict):
                raise ValueError(f"workstation frame {key} is not an object")
            entry.setdefault("motion", None)
            probability = min(1.0, max(0.0, probability))
            entry["aligned"] = round(probability, 4)
            if probability >= 0.5:
                streak = streak + 1 if previous is not None and frame == previous + 1 else 1
            else:
                streak = 0
            entry["aligned_streak"] = streak
            previous = frame
    except Exception as exc:
        warning_rows.append(
            {
                "code": "workstation.alignment-unavailable",
                "message": (
                    "Per-frame alignment could not be projected from this run "
                    f"({type(exc).__name__})."
                ),
            }
        )

    for key, entry in frames.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(entry, dict):
            raise _error(
                "invalid_workstation",
                "same-pass workstation frame evidence is invalid",
                exit_status=8,
            )
        entry.setdefault("motion", None)
    deduplicated = {
        (row["code"], row["message"]): {
            "code": row["code"][:200],
            "message": row["message"][:2000],
        }
        for row in warning_rows
        if row.get("code") and row.get("message")
    }
    value["warnings"] = list(deduplicated.values())
    temporary = workstation_path.with_name(f".{workstation_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, workstation_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


# Host shells can carry stale CUBED_* exports from other deployments of this
# pipeline, and several of them redirect stamped flag tokens or model paths.
# The runner therefore starts from a CUBED_-free environment and sets exactly
# the variables it means to set. Passthroughs are explicit user policy knobs.
# BASH_ENV and ENV are dropped too: a non-interactive shell sources them at
# startup, which re-injects host exports into pipeline scripts even when the
# runner hands the child a clean environment.
CUBED_ENV_PASSTHROUGH = frozenset({"CUBED_NVDEC"})
_SHELL_STARTUP_INJECTORS = frozenset({"BASH_ENV", "ENV"})


def _base_environment(repo_root: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _SHELL_STARTUP_INJECTORS
        and (not key.startswith("CUBED_") or key in CUBED_ENV_PASSTHROUGH)
    }
    existing = environment.get("PYTHONPATH", "")
    entries = [str(repo_root), str(repo_root / "scripts")]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    # Shell stages (run_research_decode.sh) invoke plain python3, which must
    # resolve to this runner's interpreter, not the system python. A fresh
    # compute box has no cv2 or torch outside the venv.
    interpreter_bin = os.path.dirname(sys.executable)
    path_entries = [interpreter_bin]
    if environment.get("PATH"):
        path_entries.append(environment["PATH"])
    environment["PATH"] = os.pathsep.join(path_entries)
    # The legacy alignment extractor otherwise accepts ONNX Runtime's automatic
    # CPU fallback when CUDA libraries cannot be loaded. Canonical Decode must
    # preload pip-managed CUDA/cuDNN and prove the created session kept CUDA.
    environment["CUBED_ORT_PROVIDERS"] = "cuda"
    environment["CUBED_ORT_REQUIRE_CUDA"] = "1"
    return environment


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    repo_root: Path,
    stage: str,
    runner: Any = subprocess.run,
) -> None:
    print(f"[cubed-core:decode] {stage}: {' '.join(command)}", flush=True)
    completed = runner(
        list(command),
        cwd=str(repo_root),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return_code = getattr(completed, "returncode", 1)
    if return_code != 0:
        raise _error(
            "stage_failed",
            f"the {stage} stage exited with status {return_code}",
            exit_status=7,
            details={"stage": stage, "return_code": return_code},
        )


def _nvdec_available(
    repo_root: Path,
    video: Path,
    *,
    environment: Mapping[str, str],
    runner: Any,
) -> bool:
    probe = [
        sys.executable,
        str(repo_root / "scripts" / "check_nvdec.py"),
        "--video",
        str(video),
    ]
    try:
        completed = runner(
            probe,
            cwd=str(repo_root),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return getattr(completed, "returncode", 1) == 0


def run(
    request_path: Path,
    output_path: Path,
    *,
    scratch_root: Path | None = None,
    runner: Any = subprocess.run,
) -> None:
    """Sequence the four pipeline stages for one job and collect its result."""

    context = _load_context(request_path, output_path)
    repo_root = _repo_root()
    assets = _verify_runtime_assets(context, repo_root)
    _materialize_canonical_trust_model(repo_root, Path(assets["read-trust-model"]))
    centroids = _write_centroids(context)
    environment = _base_environment(repo_root)

    scratch = Path(scratch_root) if scratch_root is not None else Path(tempfile.gettempdir())
    tag = context.job_id
    reads = scratch / f"reads_{tag}_occaware.pkl"
    reads_v2 = scratch / f"reads_{tag}_v2.pkl"
    events = scratch / f"motion_events_{tag}.json"
    alignfeat = scratch / f"alignfeat_{tag}_new.npz"
    workstation = scratch / f"decode_workstation_{tag}.json"
    scratch_files = (reads, reads_v2, events, alignfeat, workstation)

    try:
        geo_command = [
            sys.executable,
            "-u",
            str(repo_root / "scripts" / "geo_read.py"),
            "--tag",
            tag,
            "--video",
            str(context.video_path),
            "--centroids-json",
            str(centroids),
            "--gpu-reads",
            "--gpu-warp",
            "--out",
            str(reads),
        ]
        if context.workstation_context is not None:
            geo_command.extend(["--workstation-out", str(workstation)])
        policy = _nvdec_policy()
        if policy == "off":
            environment.pop("CUBED_GPU_DECODE", None)
        elif _nvdec_available(
            repo_root,
            context.video_path,
            environment=environment,
            runner=runner,
        ):
            geo_command.append("--gpu-decode")
        elif policy == "require":
            raise _error(
                "nvdec_unavailable",
                "NVDEC is required but failed its exact-video probe",
                exit_status=6,
            )
        else:
            environment.pop("CUBED_GPU_DECODE", None)

        reads_environment = dict(environment)
        reads_environment["CUBED_FACE_POSE_MODEL"] = str(assets["pose-model"])
        reads_environment["CUBED_ALIGNED_MODEL"] = str(assets["alignment-model"])

        emit_stage("reads")
        _run(
            geo_command,
            environment=reads_environment,
            repo_root=repo_root,
            stage="reads",
            runner=runner,
        )
        if not reads.is_file():
            raise _error(
                "stage_failed",
                "the reads stage did not produce a reads artifact",
                exit_status=7,
                details={"stage": "reads"},
            )

        # gen_motion_events.py resolves its input as /tmp/reads_<tag>_v2.pkl, so
        # the same read stream is staged under that name rather than re-read.
        shutil.copyfile(reads, reads_v2)
        emit_stage("events")
        _run(
            [
                sys.executable,
                "-u",
                str(repo_root / "scripts" / "gen_motion_events.py"),
                tag,
                str(events),
            ],
            environment=reads_environment,
            repo_root=repo_root,
            stage="events",
            runner=runner,
        )

        emit_stage("alignfeat")
        _run(
            [
                sys.executable,
                "-u",
                str(repo_root / "scripts" / "extract_alignfeat.py"),
                "--tag",
                tag,
                "--video",
                str(context.video_path),
                "--out",
                str(alignfeat),
            ],
            environment=reads_environment,
            repo_root=repo_root,
            stage="alignfeat",
            runner=runner,
        )

        workstation_ready = False
        if context.workstation_context is not None:
            try:
                workstation_ready = _prepare_workstation_payload(
                    context,
                    workstation_path=workstation,
                    alignfeat_path=alignfeat,
                )
            except Exception as exc:
                print(
                    "[cubed-core:decode] workstation projection unavailable "
                    f"({type(exc).__name__}: {str(exc)[:200]})",
                    flush=True,
                )

        decode_environment = dict(reads_environment)
        decode_environment.update(
            {
                "CUBED_READS": str(reads),
                "CUBED_EVENTS": str(events),
                "CUBED_CENTROIDS": str(centroids),
                "CUBED_SCRAMBLE": context.scramble,
                "CUBED_RECORDING_ID": context.capture_id,
                "CUBED_RESEARCH_DECODE_EXECUTE": "1",
                "CUBED_RESULT_JSON": str(context.output_path),
                "CUBED_RESULT_VIDEO_INPUT": str(context.video_path),
            }
        )
        if workstation_ready:
            decode_environment["CUBED_WORKSTATION_JSON"] = str(workstation)
        emit_stage("decode")
        _run(
            [str(repo_root / "scripts" / "run_research_decode.sh"), tag],
            environment=decode_environment,
            repo_root=repo_root,
            stage="decode",
            runner=runner,
        )
        if not context.output_path.is_file() or context.output_path.stat().st_size <= 0:
            raise _error(
                "missing_result",
                "the decode stage did not write a decode result document",
                exit_status=8,
            )

        # Keep the intermediate evidence beside the result so a job can be
        # inspected after the scratch directory is cleaned.
        for source in (reads, events, alignfeat):
            if source.is_file():
                shutil.copyfile(source, context.job_dir / source.name)
    finally:
        for path in scratch_files:
            try:
                path.unlink()
            except OSError:
                pass


def main(
    argv: Sequence[str] | None = None,
    *,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry point accepting only ``--request`` and ``--output``."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    error_stream = sys.stderr if stderr is None else stderr
    try:
        request_path, output_path = _parse_cli(arguments)
        run(request_path, output_path)
    except NativeDecodeRunnerError as exc:
        print(json.dumps(exc.public(), sort_keys=True), file=error_stream)
        return exc.exit_status
    except Exception as exc:
        failure = _error(
            "internal_error",
            "decode runner failed before producing a result",
            exit_status=9,
            details={
                "exception_type": type(exc).__name__,
                # Bounded rather than the full traceback: enough to tell CUDA
                # OOM, ENOSPC, and similar from each other in the job log,
                # without risking an unbounded message in the runner log.
                "exception_message": str(exc)[:200],
            },
        )
        print(json.dumps(failure.public(), sort_keys=True), file=error_stream)
        return failure.exit_status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
