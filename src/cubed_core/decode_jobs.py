"""Decode jobs: run the real local_camera_v1 pipeline for one locked capture.

This subsystem runs directly from the capture. A submission resolves a
decode-locked capture inside the server-owned workspace, snapshots that
attempt's calibration beside a private request document, and runs the selected runner as
``<runner> --request <path> --output <path>``. The runner produces one
private ``cubed-core/decode-result`` candidate; the server validates it against
the shipped schema, replays the emitted moves itself, optionally attaches a
published BLE diagnostic, and only then publishes the write-once public result
and receipt.

Evidence framing: a succeeded decode job is reconstruction evidence for one
recording plus an independent replay check on the emitted moves. Nothing here
measures accuracy and nothing here claims reach-LL. The replay check answers one
narrow question, whether the emitted move sequence takes the sealed scramble to
a solved cube, and a job whose runner claims a completed reconstruction that
fails that replay is marked failed.

A result with status ``abstained`` is a legitimate succeeded job. The pipeline
writes it when a run terminated normally without reaching the solved endpoint,
with no moves and no solved endpoint, and the status payload reports that
honestly rather than hiding it as a failure. There is nothing to replay in that
case, so ``replay_solved_reached`` stays null.

Process plumbing, the runner-environment secret filter, the bounded-artifact
readers, and the stage-marker parser live in ``job_runtime``. They are
security-relevant and stay shared rather than being copied into the decoder.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from starlette.concurrency import run_in_threadpool

from . import __version__
from .command_runtime import resolve_command_executable
from .decode_contract import (
    PROFILE_NAME,
    DecodeContractError,
    build_decode_preflight,
    load_runtime_manifest,
    load_workspace_capture,
)
from .job_runtime import (
    IDENTIFIER,
    RUNNER_SECRET_ENV_FRAGMENTS,
    RUNNER_SECRET_ENV_NAMES,
    SHA256,
    BoundedLog,
    JobRuntimeError,
    digest_regular_file,
    drain_stream,
    list_job_directories,
    load_bounded_artifact,
    parse_stage_markers,
    promote_runner_error_detail,
    reject_nonfinite,
    require_object,
    stop_process,
    write_exclusive_json,
)
from .public_ground_truth import try_build_public_ground_truth_diagnostic
from .remote_hosts import RemoteHost, RemoteHostsError, resolve_remote_host
from .settings import DECODE_COMMAND_ENV, DECODE_MODE_ENV, Settings
from .workspace import Workspace, WorkspaceError

_LOGGER = logging.getLogger(__name__)

DECODE_REQUEST_SCHEMA = "cubed-core/decode-job-request-v1"
DECODE_STATUS_SCHEMA = "cubed-core/decode-job-status-v1"
DECODE_RECEIPT_SCHEMA = "cubed-core/decode-job-receipt-v1"
DECODE_TERMINAL_SCHEMA = "cubed-core/decode-run-terminal-v1"
DECODE_RESULT_SCHEMA = "cubed-core/decode-result-v1"
DECODE_RESULT_DOCUMENT_SCHEMA = "cubed-core/decode-result"
DECODE_RESULT_SCHEMA_FILENAME = "decode-result-v1.schema.json"
DECODE_GROUND_TRUTH_SCHEMA_FILENAME = "decode-ground-truth-diagnostic-v1.schema.json"
NATIVE_DECODE_MODULE = "cubed_core.native_decode_runner"
BUNDLED_REMOTE_DECODE_RUNNER = Path("scripts/remote_decode_runner.sh")
_NATIVE_DECODE_SUPPORTED = os.name != "nt"
DECODE_EVIDENCE_SCOPE = "reconstruction-evidence-with-replay-check"
# native_decode_runner.py's RUNNER_ERROR_SCHEMA, duplicated here rather than
# imported: decode_jobs.py otherwise never imports the runner module itself
# (only its module path, as NATIVE_DECODE_MODULE, to invoke it as a
# subprocess), and this keeps that decoupling.
DECODE_RUNNER_ERROR_SCHEMA = "cubed-core/decode-runner-error-v1"

# A decode runs reads, motion events, alignment features, and the trellis search
# over a whole recording, so its ceiling is hours rather than the tracker's
# single model pass.
DECODE_JOB_TIMEOUT_SECONDS = 6 * 60 * 60
DECODE_LOG_MAX_BYTES = 64 * 1024
# The portable workstation projection can itself approach 32 MiB; leave bounded
# headroom for the result envelope and decoder timeline.
DECODE_RESULT_MAX_BYTES = 40 * 1024**2
DECODE_REQUEST_MAX_BYTES = 256 * 1024
DECODE_CALIBRATION_MAX_BYTES = 8 * 1024**2
DECODE_RECEIPT_MAX_BYTES = 256 * 1024
DECODE_TERMINAL_MAX_BYTES = 64 * 1024
# One decode at a time. The pipeline saturates a GPU and concurrent runs would
# contend for the same device rather than finish sooner.
DECODE_MAX_IN_FLIGHT = 1
DECODE_MAX_RETAINED = 100
DECODE_RUNNER_IDENTITY_FILE_MAX_BYTES = 64 * 1024**2
DECODE_TERMINAL_STATES = frozenset({"succeeded", "failed", "timed_out", "cancelled"})

router = APIRouter()


class DecodeJobError(ValueError):
    pass


def _encode_json_payload(value: dict[str, Any], *, description: str) -> bytes:
    """Serialize exactly as the exclusive writer does, without touching disk."""

    try:
        return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecodeJobError(f"{description} is not finite JSON") from exc


class DecodeJobsDisabled(DecodeJobError):
    pass


class DecodeJobsBusy(DecodeJobError):
    pass


class DecodeJobNotReady(DecodeJobError):
    """The capture or runtime does not satisfy the decode preflight contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _translate_job_runtime_error() -> Iterator[None]:
    """Re-raise a shared ``job_runtime`` I/O error as a ``DecodeJobError``.

    Every bounded-artifact read or exclusive-write in this module reports the
    same shared error type; this collapses the repeated translation into one
    place instead of copying the same ``try``/``except`` at each call site.
    """

    try:
        yield
    except JobRuntimeError as exc:
        raise DecodeJobError(str(exc)) from exc


@dataclass(frozen=True)
class DecodeReadiness:
    """The one readiness result used by capabilities and job submission."""

    runner_kind: str
    enabled: bool
    status: str
    reason: str | None
    command: tuple[str, ...]
    executable: str | None

    def public(self) -> dict[str, Any]:
        return {
            "runner_kind": self.runner_kind,
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "executable": self.executable,
        }


def build_decode_readiness(settings: Settings) -> DecodeReadiness:
    """Resolve the configured decode runner without touching a capture."""

    mode = settings.decode_mode
    if mode == "disabled":
        return DecodeReadiness(
            runner_kind="disabled",
            enabled=False,
            status="disabled",
            reason=f"{DECODE_MODE_ENV} is disabled",
            command=(),
            executable=None,
        )

    if mode == "native" and not _NATIVE_DECODE_SUPPORTED:
        return DecodeReadiness(
            runner_kind="native",
            enabled=False,
            status="unsupported-platform",
            reason=(
                "native Windows supports the CPU workbench only; "
                "run live Decode under WSL2 or Linux"
            ),
            command=(),
            executable=None,
        )

    if importlib.util.find_spec("numpy") is None:
        # The server replays the emitted moves itself before a job may succeed,
        # and that replay uses the packaged cube model.
        return DecodeReadiness(
            runner_kind=mode,
            enabled=False,
            status="missing-dependency",
            reason=(
                "the decode job replay check requires the decode dependency extra "
                "to be installed on the API host"
            ),
            command=(),
            executable=None,
        )

    if mode == "external":
        command = settings.decode_command
        executable = resolve_command_executable(command, cwd=settings.repo_root)
        if not command:
            return DecodeReadiness(
                runner_kind="external",
                enabled=False,
                status="misconfigured",
                reason=f"{DECODE_MODE_ENV}=external requires {DECODE_COMMAND_ENV}",
                command=command,
                executable=None,
            )
        if executable is None:
            return DecodeReadiness(
                runner_kind="external",
                enabled=False,
                status="misconfigured",
                reason=(
                    "the configured external decode executable was not found or is "
                    "not executable on the API host"
                ),
                command=command,
                executable=None,
            )
        return DecodeReadiness(
            runner_kind="external",
            enabled=True,
            status="configured-external",
            reason=None,
            command=command,
            executable=executable,
        )

    if mode == "native":
        command = (sys.executable, "-m", NATIVE_DECODE_MODULE)
        executable = resolve_command_executable(command, cwd=settings.repo_root)
        if settings.decode_command:
            reason: str | None = (
                f"native decode mode rejects {DECODE_COMMAND_ENV}; "
                "the current-Python module runner is selected automatically"
            )
        elif executable is None:
            reason = "the current Python executable is unavailable for the native decode runner"
        else:
            reason = None
        return DecodeReadiness(
            runner_kind="native",
            enabled=reason is None,
            status="available" if reason is None else "misconfigured",
            reason=reason,
            command=command,
            executable=executable,
        )

    reason = f"{DECODE_MODE_ENV} must be disabled, native, or external"
    return DecodeReadiness(
        runner_kind="disabled",
        enabled=False,
        status="misconfigured",
        reason=reason,
        command=(),
        executable=None,
    )


def build_decode_job_readiness(
    settings: Settings, *, remote_host: RemoteHost | None
) -> DecodeReadiness:
    """Resolve the runner selected for one job.

    Native mode defaults to the API host's module runner. An explicit
    remote host instead selects the shipped SSH bridge for that job, whose
    process identity is recorded as external because the actual decode runs
    outside the API host. External mode retains its operator-configured command
    regardless of whether a remote host contributes environment overrides.
    """

    readiness = build_decode_readiness(settings)
    if remote_host is None or readiness.runner_kind != "native" or not readiness.enabled:
        return readiness

    runner_path = settings.repo_root / BUNDLED_REMOTE_DECODE_RUNNER
    command = (str(runner_path),)
    executable = resolve_command_executable(command, cwd=settings.repo_root)
    if executable is None:
        return DecodeReadiness(
            runner_kind="external",
            enabled=False,
            status="misconfigured",
            reason=(
                f"the bundled remote decode runner {BUNDLED_REMOTE_DECODE_RUNNER} "
                "was not found or is not executable on the API host"
            ),
            command=command,
            executable=None,
        )
    return DecodeReadiness(
        runner_kind="external",
        enabled=True,
        status="configured-external",
        reason=None,
        command=command,
        executable=executable,
    )


def decode_capability(settings: Settings) -> dict[str, Any]:
    """Describe decode job execution for ``/api/capabilities``."""

    readiness = build_decode_readiness(settings)
    return {
        "enabled": readiness.enabled,
        "status": readiness.status,
        "runner_kind": readiness.runner_kind,
        "runner_label": settings.decode_runner_label or None,
        "reason": readiness.reason,
        "executable": readiness.executable,
        "profile": PROFILE_NAME,
        "execution_host": "runner-host",
        "evidence_scope": DECODE_EVIDENCE_SCOPE,
        "request_schema": DECODE_REQUEST_SCHEMA,
        "output_schema": DECODE_RESULT_SCHEMA,
    }


@lru_cache(maxsize=8)
def _result_validator(schema_directory: Path) -> Draft202012Validator:
    schemas: dict[str, dict[str, Any]] = {}
    for filename in (
        DECODE_RESULT_SCHEMA_FILENAME,
        DECODE_GROUND_TRUTH_SCHEMA_FILENAME,
    ):
        path = schema_directory / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DecodeJobError(f"shipped {filename} is unavailable") from exc
        if not isinstance(value, dict):
            raise DecodeJobError(f"shipped {filename} is not an object")
        Draft202012Validator.check_schema(value)
        schemas[filename] = value

    registry = Registry()
    for schema in schemas.values():
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise DecodeJobError("shipped decode schema lacks a canonical $id")
        registry = registry.with_resource(schema_id, Resource.from_contents(schema))
    return Draft202012Validator(
        schemas[DECODE_RESULT_SCHEMA_FILENAME],
        registry=registry,
    )


def validate_decode_result(
    value: Any,
    *,
    capture_id: str,
    repo_root: Path,
    expected_video_sha256: str | None = None,
    allow_ground_truth_diagnostic: bool = False,
    expected_ground_truth_video_sha256: str | None = None,
) -> dict[str, Any]:
    """Check one runner-produced document against the shipped decode contract.

    Beyond the schema this pins the document to the submitted capture and to the
    executable profile, and it rejects any evaluation block. A decode job result
    is reconstruction evidence, so it may not carry a teacher-scored metric.
    """

    if not isinstance(value, dict):
        raise DecodeJobError("decode result must be a JSON object")
    # Keep the lifecycle/result distinction explicit even though the schema
    # also rejects this pairing. This gives runner authors a stable, useful
    # error instead of exposing a conditional-schema implementation detail.
    if value.get("profile") == PROFILE_NAME and value.get("status") == "failed":
        raise DecodeJobError(
            "local_camera_v1 decode results must be completed or abstained; "
            "execution failures belong to the job lifecycle"
        )
    try:
        _result_validator(repo_root.resolve() / "schemas").validate(value)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {location}" if location else ""
        raise DecodeJobError(
            f"decode result violates the shipped Draft 2020-12 schema{suffix}: {exc.message}"
        ) from None
    if (
        value.get("schema") != DECODE_RESULT_DOCUMENT_SCHEMA
        or value.get("schema_version") != 1
        or value.get("recording_id") != capture_id
    ):
        raise DecodeJobError("decode result identity does not match the submitted capture")
    if value.get("profile") != PROFILE_NAME:
        raise DecodeJobError(f"decode result must declare the {PROFILE_NAME} profile")
    if value.get("status") not in {"completed", "abstained"}:
        raise DecodeJobError(
            "local_camera_v1 decode results must be completed or abstained; "
            "execution failures belong to the job lifecycle"
        )
    if value.get("evaluation") is not None:
        raise DecodeJobError(
            "decode result may not carry an evaluation block; a decode job produces "
            "reconstruction evidence, not a scored measurement"
        )
    ground_truth = value.get("ground_truth_diagnostic")
    if ground_truth is not None and not allow_ground_truth_diagnostic:
        raise DecodeJobError(
            "decode runner output may not carry ground-truth diagnostics; "
            "the server may add verified public diagnostics only after replay"
        )
    if ground_truth is not None:
        if expected_ground_truth_video_sha256 is None:
            raise DecodeJobError("decode ground-truth video binding is unavailable")
        decoded_from_ops: list[str] = []
        reference_from_ops: list[str] = []
        edit_count = 0
        operations_valid = True
        frame_binding_valid = (
            ground_truth["reference"]["scope"] == "sequence-only"
            or ground_truth["frame_timing"]["source_sha256"]
            == ground_truth["reference"]["frame_ground_truth_sha256"]
        )
        for operation in ground_truth["comparison"]["ops"]:
            decoded = operation["decoded"]
            reference = operation["reference"]
            decoded_index = operation["index_decoded"]
            reference_index = operation["index_reference"]
            kind = operation["op"]
            if decoded is not None:
                operations_valid = operations_valid and decoded_index == len(decoded_from_ops)
                decoded_from_ops.append(decoded)
            elif decoded_index is not None:
                operations_valid = False
            if reference is not None:
                operations_valid = operations_valid and reference_index == len(reference_from_ops)
                reference_from_ops.append(reference)
            elif reference_index is not None:
                operations_valid = False
            if kind == "equal":
                operations_valid = operations_valid and decoded is not None and decoded == reference
            elif kind == "substitute":
                operations_valid = (
                    operations_valid
                    and decoded is not None
                    and reference is not None
                    and decoded != reference
                )
                edit_count += 1
            elif kind == "insert":
                operations_valid = operations_valid and decoded is not None and reference is None
                edit_count += 1
            elif kind == "delete":
                operations_valid = operations_valid and decoded is None and reference is not None
                edit_count += 1
        if (
            ground_truth["capture_id"] != capture_id
            or ground_truth["reference"]["video_sha256"] != expected_ground_truth_video_sha256
            or ground_truth["counts"]["decoded_htm"] != len(value["moves"])
            or ground_truth["counts"]["ble_canonical_htm"] != len(reference_from_ops)
            or decoded_from_ops != value["moves"]
            or ground_truth["comparison"]["distance"] != edit_count
            or not operations_valid
            or not frame_binding_valid
        ):
            raise DecodeJobError(
                "decode ground-truth diagnostic does not match the closed camera result"
            )
    if expected_video_sha256 is not None:
        video_inputs = [receipt for receipt in value["inputs"] if receipt.get("id") == "video"]
        if len(video_inputs) != 1 or video_inputs[0].get("sha256") != expected_video_sha256:
            raise DecodeJobError(
                "native decode result video input does not match the sealed capture"
            )
    try:
        manifest = load_runtime_manifest(repo_root)
    except DecodeContractError as exc:
        raise DecodeJobError(str(exc)) from exc
    profile = manifest["profiles"][PROFILE_NAME]
    config = value.get("config")
    if not isinstance(config, dict) or config.get("cfg_hash") != profile["cfg_hash"]:
        raise DecodeJobError(
            "decode result CFG_HASH does not match the configured local_camera_v1 profile"
        )
    workstation = value.get("workstation")
    if isinstance(workstation, dict):
        sequence = workstation.get("sequence")
        if isinstance(sequence, dict):
            sequence_moves = [entry["move"] for entry in sequence["moves"]]
            if sequence_moves != value["moves"]:
                raise DecodeJobError(
                    "decode workstation sequence does not match top-level decode moves"
                )
        reconstruction = workstation.get("reconstruction")
        if (
            isinstance(reconstruction, dict)
            and reconstruction.get("solved_reached") != value["endpoint"]["solved_reached"]
        ):
            raise DecodeJobError(
                "decode workstation reconstruction endpoint does not match the top-level endpoint"
            )
        if isinstance(reconstruction, dict) and isinstance(reconstruction.get("timeline"), dict):
            if value["status"] != "completed":
                raise DecodeJobError(
                    "decode workstation reconstruction checkpoint timeline "
                    "requires a completed result"
                )
            timeline_moves = reconstruction["timeline"]["moves"]
            states = reconstruction["states"]
            frames = [entry["frame"] for entry in timeline_moves]
            window = workstation["window"]
            if (
                len(states) != len(timeline_moves) + 1
                or (frames and frames[0] <= window[0])
                or any(a > b for a, b in zip(frames, frames[1:], strict=False))
                or any(frame < window[0] or frame > window[1] for frame in frames)
            ):
                raise DecodeJobError(
                    "decode workstation reconstruction checkpoint timeline is inconsistent"
                )
            try:
                from .cube import Cube

                scramble = workstation["initialization"]["scramble"]
                canonical = Cube.solved().apply_algorithm(scramble).apply_algorithm(value["moves"])
                replay = Cube.solved().apply_algorithm(scramble)
                expected_states = [replay.state]
                for entry in timeline_moves:
                    replay.apply_move(entry["move"])
                    expected_states.append(replay.state)
            except (TypeError, ValueError) as exc:
                raise DecodeJobError(
                    "decode workstation reconstruction checkpoint timeline cannot be replayed"
                ) from exc
            if replay != canonical or states != expected_states:
                raise DecodeJobError(
                    "decode workstation reconstruction checkpoint timeline "
                    "does not match the result"
                )
    return value


@dataclass(frozen=True, slots=True)
class DecodeReplayCheck:
    """The server's own replay of one result's moves from the sealed scramble."""

    performed: bool
    solved_reached: bool | None
    move_count: int
    detail: str

    def public(self) -> dict[str, Any]:
        return {
            "performed": self.performed,
            "solved_reached": self.solved_reached,
            "move_count": self.move_count,
            "detail": self.detail,
        }


def _decode_outcome(*, status: str, result_status: str | None) -> str | None:
    """Map internal lifecycle state to the four user-facing run outcomes."""

    if status == "succeeded" and result_status in {"completed", "abstained"}:
        return result_status
    if status == "cancelled":
        return "cancelled"
    if status in {"failed", "timed_out"}:
        return "failed"
    if status == "succeeded" and result_status == "failed":
        return "failed"
    return None


def _failure(code: str, message: str, *, retryable: bool) -> dict[str, Any]:
    """Build the bounded failure object exposed by run history.

    Raw logs, commands, exception traces, and local paths deliberately stay out
    of this object and out of the persisted terminal record.
    """

    return {
        "code": code[:100] or "internal-error",
        "message": (message[:1000] or "decode run failed"),
        "retryable": bool(retryable),
    }


def replay_decode_result(value: dict[str, Any], *, scramble: str) -> DecodeReplayCheck:
    """Replay the emitted moves with the packaged cube model, not the runner's claim."""

    moves = value.get("moves")
    moves = list(moves) if isinstance(moves, list) else []
    if value.get("status") != "completed":
        return DecodeReplayCheck(
            performed=False,
            solved_reached=None,
            move_count=len(moves),
            detail=(
                "the run did not reach the solved endpoint, so there is no move sequence to replay"
            ),
        )
    try:
        from .cube import Cube
    except ModuleNotFoundError as exc:
        raise DecodeJobError(
            "the decode job replay check requires the decode dependency extra"
        ) from exc
    try:
        replay = Cube.solved().apply_algorithm(scramble).apply_algorithm(moves)
    except (TypeError, ValueError) as exc:
        raise DecodeJobError(
            f"decode result contains malformed scramble or move notation: {exc}"
        ) from exc
    solved = bool(replay.is_solved())
    return DecodeReplayCheck(
        performed=True,
        solved_reached=solved,
        move_count=len(moves),
        detail=(
            "the emitted moves replay from the sealed scramble to a solved cube"
            if solved
            else "the emitted moves do not replay from the sealed scramble to a solved cube"
        ),
    )


def _runtime_assets(settings: Settings) -> list[dict[str, Any]]:
    """Digest every model artifact the runtime manifest declares for this profile."""

    try:
        manifest = load_runtime_manifest(settings.repo_root)
    except DecodeContractError as exc:
        raise DecodeJobError(str(exc)) from exc
    assets: list[dict[str, Any]] = []
    repo_root = settings.repo_root.resolve()
    for requirement in manifest["runtime_requirements"]:
        if requirement.get("kind") != "model":
            continue
        relative_path = str(requirement["path"])
        identity = digest_regular_file(
            repo_root / relative_path,
            maximum_bytes=DECODE_RUNNER_IDENTITY_FILE_MAX_BYTES,
        )
        if identity is None:
            raise DecodeJobError(
                f"decode runtime asset {requirement['id']} is unavailable at {relative_path}"
            )
        assets.append({"id": str(requirement["id"]), "path": relative_path, **identity})
    if not assets:
        raise DecodeJobError("the decode runtime manifest declares no model artifacts")
    assets.sort(key=lambda asset: asset["id"])
    return assets


def _runner_provenance(
    settings: Settings,
    readiness: DecodeReadiness,
    *,
    remote_host: RemoteHost | None = None,
) -> dict[str, Any]:
    command_payload = json.dumps(
        list(readiness.command),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    executable = (
        digest_regular_file(
            Path(readiness.executable),
            maximum_bytes=DECODE_RUNNER_IDENTITY_FILE_MAX_BYTES,
        )
        if readiness.executable is not None
        else None
    )
    implementation_files: list[dict[str, Any]] = []
    for index, argument in enumerate(readiness.command[1:17], start=1):
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = settings.repo_root / candidate
        identity = digest_regular_file(
            candidate,
            maximum_bytes=DECODE_RUNNER_IDENTITY_FILE_MAX_BYTES,
        )
        if identity is not None:
            implementation_files.append({"argument_index": index, **identity})
    native_module = None
    package = None
    if readiness.runner_kind == "native":
        native_module = digest_regular_file(
            Path(__file__).with_name("native_decode_runner.py"),
            maximum_bytes=DECODE_RUNNER_IDENTITY_FILE_MAX_BYTES,
        )
        package = {"name": "cubed-core", "version": __version__}
    return {
        "runner_kind": readiness.runner_kind,
        "identity_status": (
            "verified-native-identity"
            if readiness.runner_kind == "native"
            else "unverified-external-identity"
        ),
        "command_sha256": hashlib.sha256(command_payload).hexdigest(),
        "executable": executable,
        "implementation_files": implementation_files,
        "native_module": native_module,
        "package": package,
        "runner_label": _decode_runner_label(settings, remote_host=remote_host),
    }


def _decode_runner_label(settings: Settings, *, remote_host: RemoteHost | None) -> str | None:
    """The recorded runner label, cheaply annotated with the chosen remote host.

    This is additive only: with no remote host chosen it is byte-identical to
    the label recorded before remote host selection existed. Job receipts
    don't otherwise validate this field's contents, only its presence.
    """

    label = settings.decode_runner_label or None
    if remote_host is None:
        return label
    host_note = f"remote host: {remote_host.label} ({remote_host.id})"
    return f"{label} · {host_note}" if label else host_note


def _runner_environment(
    settings: Settings, *, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Retain runtime/device variables while omitting obvious inherited secrets.

    ``overrides``, when given, is applied last so a job's chosen remote host
    (see ``remote_hosts.py``) always wins over both the server's own
    CUBED_REMOTE_* environment and the decode overrides below.
    """

    environment: dict[str, str] = {}
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name in RUNNER_SECRET_ENV_NAMES:
            continue
        if any(fragment in upper_name for fragment in RUNNER_SECRET_ENV_FRAGMENTS):
            continue
        environment[name] = value
    environment["CUBED_CORE_DECODE_MODE"] = settings.decode_mode
    environment["CUBED_CORE_REPO_ROOT"] = str(settings.repo_root)
    environment["CUBED_CORE_DECODE_RUNNER_LABEL"] = settings.decode_runner_label
    if overrides:
        environment.update(overrides)
    return environment


def _workstation_context(
    capture_receipt: dict[str, Any],
    *,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Project verified capture metadata into a portable, path-free context."""

    video = capture_receipt.get("video")
    if not isinstance(video, dict):
        raise DecodeJobError("capture video receipt is unavailable for Runs")
    projected = {
        "sha256": video.get("sha256"),
        "bytes": video.get("bytes"),
        "fps": video.get("actual_fps"),
        "frame_count": video.get("frame_count"),
        "width": video.get("encoded_width"),
        "height": video.get("encoded_height"),
    }
    if (
        not isinstance(projected["sha256"], str)
        or not SHA256.fullmatch(projected["sha256"])
        or type(projected["bytes"]) is not int
        or projected["bytes"] < 1
        or isinstance(projected["fps"], bool)
        or not isinstance(projected["fps"], (int, float))
        or not math.isfinite(float(projected["fps"]))
        or projected["fps"] <= 0
        or projected["fps"] > 1000
        or type(projected["frame_count"]) is not int
        or not 1 <= projected["frame_count"] <= 10_000_000
        or type(projected["width"]) is not int
        or not 1 <= projected["width"] <= 100_000
        or type(projected["height"]) is not int
        or not 1 <= projected["height"] <= 100_000
    ):
        raise DecodeJobError("capture video identity is unavailable for Runs")

    warnings: list[dict[str, str]] = []
    for check in preflight.get("checks", []):
        if (
            isinstance(check, dict)
            and check.get("status") == "warning"
            and isinstance(check.get("id"), str)
            and isinstance(check.get("detail"), str)
        ):
            code = check["id"][:200]
            message = check["detail"][:2000]
            if code and message:
                warnings.append({"code": code, "message": message})
    calibration = capture_receipt.get("calibration")
    if isinstance(calibration, dict) and calibration.get("schema") == (
        "cubed-core/color-centroids-v1"
    ):
        warnings.append(
            {
                "code": "calibration.transfer",
                "message": (
                    "This run uses imported or shared color centroids. They may not "
                    "match this cube, camera, or lighting."
                ),
            }
        )
    deduplicated = {(warning["code"], warning["message"]): warning for warning in warnings}
    return {"video": projected, "warnings": list(deduplicated.values())[:100]}


_REPLAY_CHECK_FIELDS = frozenset({"performed", "solved_reached", "move_count", "detail"})
_DECODE_TERMINAL_RESULT_FIELDS = frozenset({"sha256", "bytes", "status", "replay_check"})
_DECODE_TERMINAL_FAILURE_FIELDS = frozenset({"code", "message", "retryable"})
_DECODE_JOB_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "job_id",
        "capture_id",
        "finished_at",
        "profile",
        "request_sha256",
        "result_sha256",
        "result_bytes",
        "result_status",
        "replay_check",
        "evidence_scope",
        "runner_provenance",
        "runtime_assets",
    }
)


def _replay_check_fields_valid(replay: dict[str, Any]) -> bool:
    """Shared type/range checks for one already-``require_object``-shaped replay check."""

    return (
        type(replay["performed"]) is bool
        and (replay["solved_reached"] is None or type(replay["solved_reached"]) is bool)
        and type(replay["move_count"]) is int
        and replay["move_count"] >= 0
        and isinstance(replay["detail"], str)
        and bool(replay["detail"])
    )


def _replay_check_from_dict(value: dict[str, Any]) -> DecodeReplayCheck:
    """Build a ``DecodeReplayCheck`` from an already-validated replay check mapping."""

    return DecodeReplayCheck(
        performed=value["performed"],
        solved_reached=value["solved_reached"],
        move_count=value["move_count"],
        detail=value["detail"],
    )


def _parse_decode_job_receipt_for_index(
    path: Path,
    *,
    job_dir: Path,
    expected_job_id: str,
) -> dict[str, Any] | None:
    """Parse and self-validate one on-disk decode job receipt for the startup index.

    Mirrors ``_parse_tracker_success_receipt_for_index``: the receipt is the
    only source of truth for a job this process never ran, so there is nothing
    to check it against beyond its own directory and internal consistency.
    Never raises; returns ``None`` for anything missing, unreadable, or
    structurally invalid so the caller can skip one corrupt directory. Performs
    no re-hash of the (possibly large) decode result -- that happens only when
    the indexed job's result is actually fetched.
    """

    try:
        payload = load_bounded_artifact(
            path,
            base=job_dir,
            maximum_bytes=DECODE_RECEIPT_MAX_BYTES,
            description="decode job receipt",
        )
        value = json.loads(payload, parse_constant=reject_nonfinite)
        value = require_object(
            value,
            field="decode job receipt",
            required=_DECODE_JOB_RECEIPT_FIELDS,
            allowed=_DECODE_JOB_RECEIPT_FIELDS,
        )
        replay_check = require_object(
            value["replay_check"],
            field="decode job receipt.replay_check",
            required=_REPLAY_CHECK_FIELDS,
            allowed=_REPLAY_CHECK_FIELDS,
        )
    except (JobRuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        value["schema"] != DECODE_RECEIPT_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["job_id"] != expected_job_id
        or not isinstance(value["capture_id"], str)
        or not IDENTIFIER.fullmatch(value["capture_id"])
        or not isinstance(value["finished_at"], str)
        or not value["finished_at"]
        or value["profile"] != PROFILE_NAME
        or not isinstance(value["request_sha256"], str)
        or not SHA256.fullmatch(value["request_sha256"])
        or not isinstance(value["result_sha256"], str)
        or not SHA256.fullmatch(value["result_sha256"])
        or type(value["result_bytes"]) is not int
        or not 1 <= value["result_bytes"] <= DECODE_RESULT_MAX_BYTES
        or value["result_status"] not in {"completed", "abstained", "failed"}
        or value["evidence_scope"] != DECODE_EVIDENCE_SCOPE
        or not isinstance(value["runner_provenance"], dict)
        or not isinstance(value["runtime_assets"], list)
        or not _replay_check_fields_valid(replay_check)
    ):
        return None
    return value


def _parse_decode_terminal_for_index(
    path: Path,
    *,
    job_dir: Path,
    expected_job_id: str,
) -> dict[str, Any] | None:
    """Load one bounded server terminal record without trusting runner output."""

    required = {
        "schema",
        "schema_version",
        "job_id",
        "capture_id",
        "status",
        "outcome",
        "created_at",
        "started_at",
        "finished_at",
        "return_code",
        "failure",
        "video_sha256",
        "result",
    }
    try:
        payload = load_bounded_artifact(
            path,
            base=job_dir,
            maximum_bytes=DECODE_TERMINAL_MAX_BYTES,
            description="decode terminal record",
        )
        value = json.loads(payload, parse_constant=reject_nonfinite)
        value = require_object(
            value,
            field="decode terminal record",
            required=required,
            allowed=required,
        )
    except (JobRuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None

    if (
        value["schema"] != DECODE_TERMINAL_SCHEMA
        or value["schema_version"] != 1
        or value["job_id"] != expected_job_id
        or not isinstance(value["capture_id"], str)
        or not IDENTIFIER.fullmatch(value["capture_id"])
        or value["status"] not in DECODE_TERMINAL_STATES
        or value["outcome"] not in {"completed", "abstained", "failed", "cancelled"}
        or not isinstance(value["created_at"], str)
        or not value["created_at"]
        or (
            value["started_at"] is not None
            and (not isinstance(value["started_at"], str) or not value["started_at"])
        )
        or not isinstance(value["finished_at"], str)
        or not value["finished_at"]
        or (value["return_code"] is not None and type(value["return_code"]) is not int)
        or (
            value["video_sha256"] is not None
            and (
                not isinstance(value["video_sha256"], str)
                or not SHA256.fullmatch(value["video_sha256"])
            )
        )
    ):
        return None

    failure = value["failure"]
    if failure is not None:
        try:
            failure = require_object(
                failure,
                field="decode terminal record.failure",
                required=_DECODE_TERMINAL_FAILURE_FIELDS,
                allowed=_DECODE_TERMINAL_FAILURE_FIELDS,
            )
        except JobRuntimeError:
            return None
        if (
            not isinstance(failure["code"], str)
            or not failure["code"]
            or len(failure["code"]) > 100
            or not isinstance(failure["message"], str)
            or not failure["message"]
            or len(failure["message"]) > 1000
            or type(failure["retryable"]) is not bool
        ):
            return None

    result = value["result"]
    if result is not None:
        try:
            result = require_object(
                result,
                field="decode terminal record.result",
                required=_DECODE_TERMINAL_RESULT_FIELDS,
                allowed=_DECODE_TERMINAL_RESULT_FIELDS,
            )
            replay = require_object(
                result["replay_check"],
                field="decode terminal record.result.replay_check",
                required=_REPLAY_CHECK_FIELDS,
                allowed=_REPLAY_CHECK_FIELDS,
            )
        except JobRuntimeError:
            return None
        if (
            not isinstance(result["sha256"], str)
            or not SHA256.fullmatch(result["sha256"])
            or type(result["bytes"]) is not int
            or not 1 <= result["bytes"] <= DECODE_RESULT_MAX_BYTES
            or result["status"] not in {"completed", "abstained", "failed"}
            or not _replay_check_fields_valid(replay)
        ):
            return None
    if value["status"] == "succeeded" and result is None:
        return None
    if value["status"] != "succeeded" and value["outcome"] not in {"failed", "cancelled"}:
        return None
    return value


def _restored_job_paths(job_id: str, job_dir: Path) -> dict[str, Any]:
    """The ``DecodeJob`` fields shared by every job restored from disk at startup.

    A restored job is already terminal, so it carries none of the sealed
    scramble, runner command, or in-memory log a live submission would; those
    fields are the same fixed placeholders for both the terminal-record and
    the legacy-receipt restoration path below.
    """

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "request_path": job_dir / "job-request.json",
        "output_path": job_dir / "decode-result.json",
        "scramble": "",
        "runner_command": (),
        "log": "",
        "log_truncated": False,
    }


def _index_decode_jobs(jobs_root: Path, *, maximum: int) -> dict[str, DecodeJob]:
    """Rebuild terminal decode attempts from disk for a fresh service.

    New jobs carry a server-authored terminal record whether they completed,
    abstained, failed, timed out, or were cancelled. Older succeeded jobs are
    still recovered from their write-once receipts for backward compatibility.
    The result is capped to the newest ``maximum`` jobs by ``finished_at`` and
    handed back oldest-first so ``_prune``'s FIFO eviction keeps working.
    """

    candidates: list[DecodeJob] = []
    orphaned_run_count = 0
    for job_id, job_dir in list_job_directories(jobs_root):
        terminal_path = job_dir / "run-terminal.json"
        if terminal_path.is_file() and not terminal_path.is_symlink():
            terminal = _parse_decode_terminal_for_index(
                terminal_path,
                job_dir=job_dir,
                expected_job_id=job_id,
            )
            if terminal is None:
                _LOGGER.warning(
                    "decode job index: skipping %s, terminal record is invalid or unreadable",
                    job_id,
                )
                continue
            result = terminal["result"]
            replay_value = result["replay_check"] if result is not None else None
            failure = terminal["failure"]
            candidates.append(
                DecodeJob(
                    **_restored_job_paths(job_id, job_dir),
                    capture_id=terminal["capture_id"],
                    runner_provenance={},
                    runtime_assets=[],
                    status=terminal["status"],
                    created_at=terminal["created_at"],
                    started_at=terminal["started_at"],
                    finished_at=terminal["finished_at"],
                    return_code=terminal["return_code"],
                    error=failure["message"] if failure is not None else None,
                    failure=failure,
                    video_sha256=terminal["video_sha256"],
                    result_sha256=result["sha256"] if result is not None else None,
                    result_bytes=result["bytes"] if result is not None else None,
                    result_status=result["status"] if result is not None else None,
                    replay_check=(
                        _replay_check_from_dict(replay_value) if replay_value is not None else None
                    ),
                )
            )
            continue
        receipt_path = job_dir / "decode-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            output_path = job_dir / "decode-result.json"
            runner_output_path = job_dir / "runner-result.json"
            if output_path.is_file() and not output_path.is_symlink():
                # A hard server restart between the runner writing its result
                # being finalized and this job's decode receipt being written
                # leaves a server-owned result with no receipt. It is correctly
                # dropped from the index, but flag it for manual recovery.
                orphaned_run_count += 1
                _LOGGER.warning(
                    "decode job index: %s has a decode result but no decode "
                    "receipt (orphaned run, dropped from the index)",
                    output_path,
                )
            elif runner_output_path.is_file() and not runner_output_path.is_symlink():
                # Camera output alone is never promoted after restart: it has
                # not passed the server replay/enrichment/final-write boundary.
                orphaned_run_count += 1
                _LOGGER.warning(
                    "decode job index: %s has only a private runner result "
                    "(unfinished run, dropped from the index)",
                    runner_output_path,
                )
            continue
        receipt = _parse_decode_job_receipt_for_index(
            receipt_path,
            job_dir=job_dir,
            expected_job_id=job_id,
        )
        if receipt is None:
            _LOGGER.warning(
                "decode job index: skipping %s, decode job receipt is invalid or unreadable",
                job_id,
            )
            continue
        replay_check = receipt["replay_check"]
        candidates.append(
            DecodeJob(
                **_restored_job_paths(job_id, job_dir),
                capture_id=receipt["capture_id"],
                runner_provenance=receipt["runner_provenance"],
                runtime_assets=receipt["runtime_assets"],
                status="succeeded",
                # The true submission time is not retained in the receipt; the
                # finished_at binding is the only timestamp this job carries.
                created_at=receipt["finished_at"],
                started_at=None,
                finished_at=receipt["finished_at"],
                return_code=0,
                error=None,
                failure=None,
                video_sha256=None,
                result_sha256=receipt["result_sha256"],
                result_bytes=receipt["result_bytes"],
                result_status=receipt["result_status"],
                replay_check=_replay_check_from_dict(replay_check),
            )
        )
    if orphaned_run_count:
        _LOGGER.warning(
            "decode job index: %d orphaned run(s) had a decode result but no "
            "decode receipt and were dropped from the index",
            orphaned_run_count,
        )
    candidates.sort(key=lambda job: job.finished_at or "", reverse=True)
    capped = candidates[:maximum]
    capped.sort(key=lambda job: job.finished_at or "")
    return {job.job_id: job for job in capped}


@dataclass
class DecodeJob:
    job_id: str
    capture_id: str
    job_dir: Path
    request_path: Path
    output_path: Path
    scramble: str
    runner_command: tuple[str, ...]
    runner_provenance: dict[str, Any]
    runtime_assets: list[dict[str, Any]]
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log: str = ""
    log_truncated: bool = False
    error: str | None = None
    failure: dict[str, Any] | None = None
    video_sha256: str | None = None
    # Exact job-private calibration snapshot selected at submission. Unlike
    # video/scramble, the capture's current calibration may change before a
    # later attempt, so this digest must remain stable through receipt write.
    calibration_sha256: str | None = None
    # Native decode mode covers both the local module and the bundled remote
    # bridge. The latter records an external process identity because SSH runs
    # outside the API host, so provenance.runner_kind cannot decide whether the
    # canonical source-video receipt is mandatory.
    require_video_input_binding: bool = False
    # Snapshot the source-video identity at submission time. The result keeps
    # encoded metadata separate from decoded/display geometry, which may be
    # rotated by OpenCV. Restored terminal jobs leave this unset: their result
    # was already accepted and is bound by the write-once receipt.
    expected_workstation_video: dict[str, Any] | None = None
    result_sha256: str | None = None
    result_bytes: int | None = None
    result_status: str | None = None
    replay_check: DecodeReplayCheck | None = None
    # The remote host chosen for this job at submission time (see
    # remote_hosts.py), or None when no remote_host was given, in which case
    # the runner subprocess sees only the server's own environment exactly as
    # before. Kept on the job (rather than only its derived environment
    # overrides) so the mid-run identity check below can recompute the exact
    # same runner_provenance it was built with.
    remote_host: RemoteHost | None = None

    @property
    def runner_output_path(self) -> Path:
        """Private camera-only runner output, never served by the result API."""

        return self.job_dir / "runner-result.json"

    def public(self, *, include_log: bool = True) -> dict[str, Any]:
        succeeded = self.status == "succeeded"
        result_available = succeeded and self.result_sha256 is not None
        stage_info = parse_stage_markers(self.log)
        stage_progress = (
            {"current": stage_info.stage_progress[0], "total": stage_info.stage_progress[1]}
            if stage_info.stage_progress is not None
            else None
        )
        replay = self.replay_check
        return {
            "schema": DECODE_STATUS_SCHEMA,
            "schema_version": 1,
            "job_id": self.job_id,
            "capture_id": self.capture_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "log": self.log if include_log else "",
            "log_truncated": self.log_truncated if include_log else False,
            "stage": stage_info.stage,
            "stages_seen": list(stage_info.stages_seen),
            "stage_progress": stage_progress,
            # The capture-scoped/status routes may expose the bounded runner
            # detail needed by Decode while a job is active. The global Runs
            # index is a quiet history surface: it carries only the structured,
            # server-authored failure below, never raw runner or exception text.
            "error": self.error if include_log else None,
            "failure": self.failure,
            "outcome": _decode_outcome(status=self.status, result_status=self.result_status),
            "profile": PROFILE_NAME,
            "evidence_scope": DECODE_EVIDENCE_SCOPE,
            "video_sha256": self.video_sha256,
            "result_available": result_available,
            "result_url": (f"/api/decode/jobs/{self.job_id}/result" if result_available else None),
            "result_sha256": self.result_sha256 if result_available else None,
            "result_bytes": self.result_bytes if result_available else None,
            "result_status": self.result_status if result_available else None,
            "replay_solved_reached": replay.solved_reached if replay is not None else None,
            "replay_check": replay.public() if replay is not None else None,
            "request_schema": DECODE_REQUEST_SCHEMA,
            "output_schema": DECODE_RESULT_SCHEMA,
        }


class DecodeJobService:
    def __init__(self, settings: Settings, workspace: Workspace) -> None:
        self.settings = settings
        self.workspace = workspace
        self.jobs_root = workspace.root / "decode-jobs"
        self.trash_root = workspace.root / "run-trash" / "decode-jobs"
        self._jobs: dict[str, DecodeJob] = self._load_job_index()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled_jobs: set[str] = set()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(DECODE_MAX_IN_FLIGHT)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cubed-decode")
        self._closed = False

    @property
    def enabled(self) -> bool:
        return build_decode_readiness(self.settings).enabled

    def _load_job_index(self) -> dict[str, DecodeJob]:
        """Scan the workspace for prior terminal attempts so restarts don't lose them.

        Construction must never raise regardless of what a workspace holds, so
        any unexpected failure here (beyond the per-directory handling already
        inside ``_index_decode_jobs``) falls back to an empty index rather than
        blocking the service from starting.
        """

        if not self.jobs_root.is_dir():
            return {}
        try:
            return _index_decode_jobs(self.jobs_root, maximum=DECODE_MAX_RETAINED)
        except Exception:
            _LOGGER.exception("decode job index scan failed; starting with an empty index")
            return {}

    def _initialize_jobs_root(self) -> None:
        self.workspace.initialize()
        if self.jobs_root.is_symlink():
            raise DecodeJobError("decode-jobs workspace path may not be a symlink")
        self.jobs_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.jobs_root.resolve(strict=True).relative_to(
                self.workspace.root.resolve(strict=True)
            )
        except (OSError, ValueError) as exc:
            raise DecodeJobError("decode-jobs workspace path is invalid") from exc

    def _initialize_trash_root(self) -> None:
        self.workspace.initialize()
        parent = self.workspace.root / "run-trash"
        for path, label in ((parent, "run-trash"), (self.trash_root, "decode run-trash")):
            if path.is_symlink():
                raise DecodeJobError(f"{label} workspace path may not be a symlink")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                path.resolve(strict=True).relative_to(self.workspace.root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise DecodeJobError(f"{label} workspace path is invalid") from exc

    def _preflight(self, capture_id: str) -> dict[str, Any]:
        try:
            capture = load_workspace_capture(self.workspace.root, capture_id)
        except DecodeContractError as exc:
            raise DecodeJobError("capture not found") from exc
        try:
            return build_decode_preflight(
                capture,
                repo_root=self.settings.repo_root,
                workspace_root=self.workspace.root,
            )
        except DecodeContractError as exc:
            raise DecodeJobError(str(exc)) from exc

    def _prune(self) -> None:
        if len(self._jobs) < DECODE_MAX_RETAINED:
            return
        terminal = [
            job_id for job_id, job in self._jobs.items() if job.status in DECODE_TERMINAL_STATES
        ]
        for job_id in terminal[: len(self._jobs) - DECODE_MAX_RETAINED + 1]:
            self._jobs.pop(job_id, None)
        if len(self._jobs) >= DECODE_MAX_RETAINED:
            raise DecodeJobsBusy("decode job history is full")

    def submit(self, capture_id: str, *, remote_host: str | None = None) -> dict[str, Any]:
        readiness = build_decode_readiness(self.settings)
        if not readiness.enabled:
            raise DecodeJobsDisabled(f"decode runner is {readiness.status}: {readiness.reason}")
        if not IDENTIFIER.fullmatch(capture_id):
            raise DecodeJobError("invalid capture id")
        resolved_remote_host: RemoteHost | None = None
        if remote_host is not None:
            try:
                resolved_remote_host = resolve_remote_host(self.workspace.root, remote_host)
            except RemoteHostsError as exc:
                raise DecodeJobError(str(exc)) from exc
        readiness = build_decode_job_readiness(
            self.settings,
            remote_host=resolved_remote_host,
        )
        if not readiness.enabled:
            raise DecodeJobsDisabled(f"decode runner is {readiness.status}: {readiness.reason}")
        preflight = self._preflight(capture_id)
        if not preflight.get("ready"):
            missing = ", ".join(str(item) for item in preflight.get("missing", [])) or "unknown"
            raise DecodeJobNotReady(
                f"capture is not ready for decode; failing preflight checks: {missing}"
            )
        try:
            # The preflight already verified the capture bytes. This re-reads
            # them under the capture lock to obtain one consistent calibration
            # version and the normalized scramble from the locked receipt.
            artifacts = self.workspace.read_decode_artifacts(capture_id)
            video_path = self.workspace.capture_video_path(capture_id)
        except WorkspaceError as exc:
            raise DecodeJobError(str(exc)) from exc
        runtime_assets = _runtime_assets(self.settings)

        if not self._slots.acquire(blocking=False):
            raise DecodeJobsBusy("a decode job is already queued or running")
        try:
            self._initialize_jobs_root()
            job_id = os.urandom(16).hex()
            job_dir = self.jobs_root / job_id
            job_dir.mkdir(mode=0o700)
            try:
                job_dir = job_dir.resolve(strict=True)
                job_dir.relative_to(self.jobs_root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise DecodeJobError("decode job workspace path is invalid") from exc
            created_at = _utc_now()
            request_path = job_dir / "job-request.json"
            output_path = job_dir / "decode-result.json"
            calibration_path = job_dir / "calibration.json"
            calibration_sha256 = hashlib.sha256(artifacts.calibration_json).hexdigest()
            if not 0 < len(artifacts.calibration_json) <= DECODE_CALIBRATION_MAX_BYTES:
                raise DecodeJobError("capture calibration sidecar is unavailable")
            with calibration_path.open("xb") as stream:
                stream.write(artifacts.calibration_json)
                stream.flush()
                os.fsync(stream.fileno())
            calibration_path.chmod(0o600)
            try:
                manifest = load_runtime_manifest(self.settings.repo_root)
            except DecodeContractError as exc:
                raise DecodeJobError(str(exc)) from exc
            profile = manifest["profiles"][PROFILE_NAME]
            workstation_context = _workstation_context(
                artifacts.receipt,
                preflight=preflight,
            )
            request_value = {
                "schema": DECODE_REQUEST_SCHEMA,
                "schema_version": 1,
                "job_id": job_id,
                "capture_id": capture_id,
                "created_at": created_at,
                "profile": PROFILE_NAME,
                "config": {
                    "name": profile["name"],
                    "cfg_hash": profile["cfg_hash"],
                    "cfg_hash_algorithm": profile["cfg_hash_algorithm"],
                },
                "inference_policy": {
                    "mode": "camera-only",
                    "ground_truth": "unavailable",
                    "evaluation": "must-be-absent",
                },
                "inputs": {
                    "video": str(video_path),
                    "calibration": str(calibration_path),
                    "calibration_sha256": calibration_sha256,
                    "scramble": artifacts.scramble,
                },
                "runtime_assets": runtime_assets,
                "workstation_context": workstation_context,
                "expected_output": {
                    "schema": DECODE_RESULT_DOCUMENT_SCHEMA,
                    "schema_version": 1,
                },
            }
            with request_path.open("x", encoding="utf-8") as stream:
                json.dump(request_value, stream, indent=2, sort_keys=True)
                stream.write("\n")
            request_path.chmod(0o600)
            job = DecodeJob(
                job_id=job_id,
                capture_id=capture_id,
                job_dir=job_dir,
                request_path=request_path,
                output_path=output_path,
                scramble=artifacts.scramble,
                runner_command=readiness.command,
                runner_provenance=_runner_provenance(
                    self.settings, readiness, remote_host=resolved_remote_host
                ),
                runtime_assets=runtime_assets,
                status="queued",
                created_at=created_at,
                video_sha256=workstation_context["video"]["sha256"],
                calibration_sha256=calibration_sha256,
                require_video_input_binding=self.settings.decode_mode == "native",
                expected_workstation_video=dict(workstation_context["video"]),
                remote_host=resolved_remote_host,
            )
            with self._lock:
                if self._closed:
                    raise DecodeJobError("decode job service is shutting down")
                self._prune()
                self._jobs[job_id] = job
            future = self._executor.submit(self._run_job, job_id)
            future.add_done_callback(lambda completed: self._job_done(job_id, completed))
            return job.public()
        except Exception:
            self._slots.release()
            raise

    def _job_done(self, job_id: str, future: Future[None]) -> None:
        if future.cancelled():
            terminal_job: DecodeJob | None = None
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status == "queued":
                    job.status = "cancelled"
                    job.finished_at = _utc_now()
                    job.error = "decode job was cancelled during shutdown"
                    job.failure = _failure(
                        "service-shutdown",
                        job.error,
                        retryable=True,
                    )
                    terminal_job = job
            if terminal_job is not None:
                self._try_persist_terminal_record(terminal_job)
            self._slots.release()

    def _run_job(self, job_id: str) -> None:
        process: subprocess.Popen[bytes] | None = None
        log_buffer = BoundedLog(DECODE_LOG_MAX_BYTES)
        timed_out = False
        try:
            with self._lock:
                job = self._jobs[job_id]
                if self._closed:
                    job.status = "cancelled"
                    job.finished_at = _utc_now()
                    job.error = "decode job service is shutting down"
                    job.failure = _failure(
                        "service-shutdown",
                        job.error,
                        retryable=True,
                    )
                    return
                job.status = "running"
                job.started_at = _utc_now()
            argv = [
                *job.runner_command,
                "--request",
                str(job.request_path),
                "--output",
                str(job.runner_output_path),
            ]
            process = subprocess.Popen(
                argv,
                cwd=self.settings.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_runner_environment(
                    self.settings,
                    overrides=job.remote_host.environment_overrides()
                    if job.remote_host is not None
                    else None,
                ),
                shell=False,
                start_new_session=os.name == "posix",
            )
            with self._lock:
                self._processes[job_id] = process

            def _sync_log() -> None:
                current_log, current_truncated = log_buffer.result()
                with self._lock:
                    current_job = self._jobs.get(job_id)
                    if current_job is not None:
                        current_job.log = current_log
                        current_job.log_truncated = current_truncated

            assert process.stdout is not None
            reader = threading.Thread(
                target=drain_stream,
                args=(process.stdout, log_buffer, _sync_log),
                name=f"cubed-decode-log-{job_id[:8]}",
                daemon=True,
            )
            reader.start()
            try:
                return_code = process.wait(timeout=DECODE_JOB_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process(process)
                return_code = process.returncode
            reader.join(timeout=2)
            if reader.is_alive():
                process.stdout.close()
                reader.join(timeout=1)
            log, log_truncated = log_buffer.result()
            (job.job_dir / "runner.log").write_text(log, encoding="utf-8")

            result_sha256: str | None = None
            result_bytes: int | None = None
            result_status: str | None = None
            replay: DecodeReplayCheck | None = None
            with self._lock:
                cancelled = job_id in self._cancelled_jobs
            failure: dict[str, Any] | None
            if cancelled:
                status = "cancelled"
                error = "decode job was cancelled during shutdown"
                failure = _failure(
                    "service-shutdown",
                    "The decode service stopped before this run completed.",
                    retryable=True,
                )
            elif timed_out:
                status = "timed_out"
                error = f"decode runner exceeded the {DECODE_JOB_TIMEOUT_SECONDS}-second timeout"
                failure = _failure(
                    "runner-timeout",
                    "The decode runner exceeded its execution timeout.",
                    retryable=True,
                )
            elif return_code != 0:
                status = "failed"
                error = promote_runner_error_detail(
                    log,
                    schema=DECODE_RUNNER_ERROR_SCHEMA,
                    message=f"decode runner exited with status {return_code}",
                )
                failure = _failure(
                    "runner-exit",
                    f"The decode runner exited with status {return_code}.",
                    retryable=True,
                )
            else:
                result, _runner_payload = self._load_result(job)
                replay = replay_decode_result(result, scramble=job.scramble)
                result_status = str(result["status"])
                if replay.performed and replay.solved_reached is not True:
                    status = "failed"
                    error = (
                        "decode runner reported a completed reconstruction but the "
                        "emitted moves do not replay from the sealed scramble to a "
                        "solved cube"
                    )
                    failure = _failure("replay-failed", error, retryable=False)
                else:
                    final_result = self._finalize_result_document(job, result)
                    preview = _encode_json_payload(
                        final_result,
                        description="decode result",
                    )
                    if (
                        len(preview) > DECODE_RESULT_MAX_BYTES
                        and "ground_truth_diagnostic" in final_result
                    ):
                        _LOGGER.warning(
                            "omitting public diagnostic because the final result "
                            "would exceed the artifact limit for capture %s",
                            job.capture_id,
                        )
                        final_result = result
                        preview = _encode_json_payload(
                            final_result,
                            description="decode result",
                        )
                    if len(preview) > DECODE_RESULT_MAX_BYTES:
                        raise DecodeJobError(
                            f"decode result exceeds the {DECODE_RESULT_MAX_BYTES}-byte limit"
                        )
                    with _translate_job_runtime_error():
                        payload = write_exclusive_json(
                            job.output_path,
                            final_result,
                            description="decode result",
                        )
                    if payload != preview:  # pragma: no cover - shared serializer contract
                        raise DecodeJobError(
                            "decode result serialization changed during publication"
                        )
                    result_sha256 = hashlib.sha256(payload).hexdigest()
                    result_bytes = len(payload)
                    status = "succeeded"
                    error = None
                    failure = None
            finished_at = _utc_now()
            if status == "succeeded":
                assert result_sha256 is not None and result_bytes is not None
                assert replay is not None and result_status is not None
                self._persist_receipt(
                    job,
                    finished_at=finished_at,
                    result_sha256=result_sha256,
                    result_bytes=result_bytes,
                    result_status=result_status,
                    replay=replay,
                )
            with self._lock:
                job.status = status
                job.finished_at = finished_at
                job.return_code = return_code
                job.log = log
                job.log_truncated = log_truncated
                job.error = error
                job.failure = failure
                job.result_sha256 = result_sha256
                job.result_bytes = result_bytes
                job.result_status = result_status
                job.replay_check = replay
        except Exception as exc:
            log, log_truncated = log_buffer.result()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    cancelled = job_id in self._cancelled_jobs
                    job.status = "cancelled" if cancelled else "failed"
                    job.finished_at = _utc_now()
                    job.return_code = process.returncode if process is not None else None
                    job.log = log
                    job.log_truncated = log_truncated
                    job.error = str(exc)[:1000] or type(exc).__name__
                    if cancelled:
                        code, message = (
                            "service-shutdown",
                            "The decode service stopped before this run completed.",
                        )
                    elif isinstance(exc, DecodeJobError):
                        code, message = (
                            "invalid-result",
                            "The decode runner did not produce a valid result.",
                        )
                    else:
                        code, message = (
                            "internal-error",
                            "The decode service could not complete this run.",
                        )
                    job.failure = _failure(
                        code,
                        message,
                        retryable=cancelled or not isinstance(exc, DecodeJobError),
                    )
        finally:
            terminal_job: DecodeJob | None = None
            with self._lock:
                self._processes.pop(job_id, None)
                self._cancelled_jobs.discard(job_id)
                candidate = self._jobs.get(job_id)
                if candidate is not None and candidate.status in DECODE_TERMINAL_STATES:
                    terminal_job = candidate
            if terminal_job is not None:
                self._try_persist_terminal_record(terminal_job)
            self._slots.release()

    def _load_result(self, job: DecodeJob) -> tuple[dict[str, Any], bytes]:
        with _translate_job_runtime_error():
            payload = load_bounded_artifact(
                job.runner_output_path,
                base=job.job_dir,
                maximum_bytes=DECODE_RESULT_MAX_BYTES,
                description="decode runner result",
            )
        try:
            value = json.loads(payload, parse_constant=reject_nonfinite)
        except (UnicodeDecodeError, ValueError) as exc:
            raise DecodeJobError("decode result must be valid JSON") from exc
        if job.require_video_input_binding and job.video_sha256 is None:
            raise DecodeJobError("native decode video identity is unavailable")
        validated = validate_decode_result(
            value,
            capture_id=job.capture_id,
            repo_root=self.settings.repo_root,
            expected_video_sha256=(job.video_sha256 if job.require_video_input_binding else None),
        )
        workstation = validated.get("workstation")
        if workstation is not None and job.expected_workstation_video is not None:
            workstation_video = workstation.get("video")
            expected_video = job.expected_workstation_video
            expected_encoded = {
                "fps": expected_video["fps"],
                "frame_count": expected_video["frame_count"],
                "width": expected_video["width"],
                "height": expected_video["height"],
            }
            if (
                not isinstance(workstation_video, dict)
                or workstation_video.get("sha256") != expected_video["sha256"]
                or workstation_video.get("bytes") != expected_video["bytes"]
                or workstation_video.get("encoded") != expected_encoded
            ):
                raise DecodeJobError(
                    "decode workstation video identity does not match the sealed capture"
                )
            initialization = workstation.get("initialization")
            if (
                not isinstance(initialization, dict)
                or initialization.get("scramble") != job.scramble
            ):
                raise DecodeJobError(
                    "decode workstation initialization does not match the sealed scramble"
                )
        return validated, payload

    def _finalize_result_document(
        self,
        job: DecodeJob,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Optionally enrich a closed camera result, then validate it again.

        This runs only after the server replay stage. Public ground-truth
        availability is diagnostic-only: a missing or rejected enrichment is
        omitted and can never change the camera decode lifecycle.
        """

        try:
            diagnostic = try_build_public_ground_truth_diagnostic(
                self.workspace.root,
                capture_id=job.capture_id,
                decoded_moves=result["moves"],
                video_sha256=job.video_sha256 or "",
                scramble=job.scramble,
            )
        except Exception:
            # The camera result is already schema-valid and replay-closed.
            # Optional public context may never turn that success into a
            # failure, including when its implementation has an unexpected
            # bug. Do not put the exception text (which may contain paths)
            # into a public or operator-facing status.
            _LOGGER.warning(
                "public diagnostic unexpectedly unavailable for capture %s",
                job.capture_id,
            )
            diagnostic = None
        candidate = dict(result)
        if diagnostic is not None:
            candidate["ground_truth_diagnostic"] = diagnostic
        try:
            return validate_decode_result(
                candidate,
                capture_id=job.capture_id,
                repo_root=self.settings.repo_root,
                expected_video_sha256=(
                    job.video_sha256 if job.require_video_input_binding else None
                ),
                allow_ground_truth_diagnostic=diagnostic is not None,
                expected_ground_truth_video_sha256=(
                    job.video_sha256 if diagnostic is not None else None
                ),
            )
        except Exception:
            if diagnostic is None:
                raise
            _LOGGER.warning(
                "verified public diagnostic was rejected during final binding for capture %s",
                job.capture_id,
            )
            return validate_decode_result(
                result,
                capture_id=job.capture_id,
                repo_root=self.settings.repo_root,
                expected_video_sha256=(
                    job.video_sha256 if job.require_video_input_binding else None
                ),
            )

    def _persist_receipt(
        self,
        job: DecodeJob,
        *,
        finished_at: str,
        result_sha256: str,
        result_bytes: int,
        result_status: str,
        replay: DecodeReplayCheck,
    ) -> None:
        calibration_receipt = digest_regular_file(
            job.job_dir / "calibration.json",
            maximum_bytes=DECODE_CALIBRATION_MAX_BYTES,
        )
        if (
            calibration_receipt is None
            or calibration_receipt.get("sha256") != job.calibration_sha256
        ):
            raise DecodeJobError("decode calibration snapshot changed while the job was running")
        current_provenance = _runner_provenance(
            self.settings,
            build_decode_job_readiness(self.settings, remote_host=job.remote_host),
            remote_host=job.remote_host,
        )
        if current_provenance != job.runner_provenance:
            raise DecodeJobError("decode runner identity changed while the job was running")
        if _runtime_assets(self.settings) != job.runtime_assets:
            raise DecodeJobError("decode runtime model artifacts changed while the job was running")
        with _translate_job_runtime_error():
            request_payload = load_bounded_artifact(
                job.request_path,
                base=job.job_dir,
                maximum_bytes=DECODE_REQUEST_MAX_BYTES,
                description="decode request",
            )
            write_exclusive_json(
                job.job_dir / "decode-receipt.json",
                {
                    "schema": DECODE_RECEIPT_SCHEMA,
                    "schema_version": 1,
                    "job_id": job.job_id,
                    "capture_id": job.capture_id,
                    "finished_at": finished_at,
                    "profile": PROFILE_NAME,
                    "request_sha256": hashlib.sha256(request_payload).hexdigest(),
                    "result_sha256": result_sha256,
                    "result_bytes": result_bytes,
                    "result_status": result_status,
                    "replay_check": replay.public(),
                    "evidence_scope": DECODE_EVIDENCE_SCOPE,
                    "runner_provenance": job.runner_provenance,
                    "runtime_assets": job.runtime_assets,
                },
                description="decode job receipt",
            )

    def _persist_terminal_record(self, job: DecodeJob) -> None:
        """Persist the restart-safe, log-free summary of one terminal attempt."""

        if job.status not in DECODE_TERMINAL_STATES or job.finished_at is None:
            raise DecodeJobError("decode run is not terminal")
        outcome = _decode_outcome(status=job.status, result_status=job.result_status)
        if outcome is None:
            raise DecodeJobError("decode run outcome is unavailable")
        result: dict[str, Any] | None = None
        if (
            job.result_sha256 is not None
            and job.result_bytes is not None
            and job.result_status is not None
            and job.replay_check is not None
        ):
            result = {
                "sha256": job.result_sha256,
                "bytes": job.result_bytes,
                "status": job.result_status,
                "replay_check": job.replay_check.public(),
            }
        if job.status == "succeeded" and result is None:
            raise DecodeJobError("decode success binding is unavailable")
        with _translate_job_runtime_error():
            write_exclusive_json(
                job.job_dir / "run-terminal.json",
                {
                    "schema": DECODE_TERMINAL_SCHEMA,
                    "schema_version": 1,
                    "job_id": job.job_id,
                    "capture_id": job.capture_id,
                    "status": job.status,
                    "outcome": outcome,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "return_code": job.return_code,
                    "failure": job.failure,
                    "video_sha256": job.video_sha256,
                    "result": result,
                },
                description="decode terminal record",
            )

    def _try_persist_terminal_record(self, job: DecodeJob) -> None:
        terminal_path = job.job_dir / "run-terminal.json"
        if terminal_path.is_file() and not terminal_path.is_symlink():
            return
        try:
            self._persist_terminal_record(job)
        except DecodeJobError:
            _LOGGER.exception("could not persist terminal record for decode job %s", job.job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        if not IDENTIFIER.fullmatch(job_id):
            raise DecodeJobError("decode job not found")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DecodeJobError("decode job not found")
            return job.public()

    def list_for_capture(self, capture_id: str) -> list[dict[str, Any]]:
        """Newest-first status payloads for one capture, live and disk-restored alike.

        Live and disk-restored jobs share the one ``_jobs`` map, so a live job
        can never be shadowed by a stale disk entry for the same id: job ids are
        random per submission and a restart only ever adds entries the live
        process has not yet produced.
        """

        if not IDENTIFIER.fullmatch(capture_id):
            raise DecodeJobError("invalid capture id")
        with self._lock:
            matches = [job for job in self._jobs.values() if job.capture_id == capture_id]
        matches.sort(key=lambda job: job.created_at, reverse=True)
        return [job.public() for job in matches]

    def list_all(self, *, query: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List Decode attempts newest-first without exposing their raw logs."""

        if type(limit) is not int or not 1 <= limit <= DECODE_MAX_RETAINED:
            raise DecodeJobError(f"limit must be from 1 through {DECODE_MAX_RETAINED}")
        needle = (query or "").strip().casefold()
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        if needle:
            jobs = [
                job
                for job in jobs
                if needle
                in " ".join(
                    filter(
                        None,
                        (
                            job.job_id,
                            job.capture_id,
                            job.status,
                            job.result_status,
                            _decode_outcome(
                                status=job.status,
                                result_status=job.result_status,
                            ),
                        ),
                    )
                ).casefold()
            ]
        return [job.public(include_log=False) for job in jobs[:limit]]

    def trash(self, job_id: str) -> dict[str, Any]:
        """Move one terminal run directory to recoverable run trash.

        Capture bundles live under a disjoint workspace root and are never
        opened or moved by this operation.
        """

        if not IDENTIFIER.fullmatch(job_id):
            raise DecodeJobError("decode job not found")
        self._initialize_trash_root()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DecodeJobError("decode job not found")
            if job.status not in DECODE_TERMINAL_STATES:
                raise DecodeJobError("an active decode run cannot be trashed")
            if job.job_dir.is_symlink():
                raise DecodeJobError("decode run directory is invalid")
            try:
                source = job.job_dir.resolve(strict=True)
                source.relative_to(self.jobs_root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise DecodeJobError("decode run directory is invalid") from exc
            trash_id = f"{job_id}-{os.urandom(8).hex()}"
            destination = self.trash_root / trash_id
            if destination.exists() or destination.is_symlink():
                raise DecodeJobError("decode run trash destination is unavailable")
            try:
                os.replace(source, destination)
            except OSError as exc:
                raise DecodeJobError("decode run could not be moved to trash") from exc
            self._jobs.pop(job_id, None)
        return {
            "schema": "cubed-core/decode-run-delete-v1",
            "schema_version": 1,
            "job_id": job_id,
            "capture_id": job.capture_id,
            "trashed": True,
            "recoverable": True,
            "trash_id": trash_id,
        }

    def result(self, job_id: str) -> bytes:
        """Re-read and re-hash the persisted result against its write-once receipt."""

        if not IDENTIFIER.fullmatch(job_id):
            raise DecodeJobError("decode job not found")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DecodeJobError("decode job not found")
            if job.status != "succeeded":
                raise DecodeJobError("decode result is not available")
            job_dir = job.job_dir
            output_path = job.output_path
            capture_id = job.capture_id
            result_sha256 = job.result_sha256
            result_bytes = job.result_bytes
        if result_sha256 is None or result_bytes is None:
            raise DecodeJobError("decode success binding is unavailable")
        with _translate_job_runtime_error():
            receipt_payload = load_bounded_artifact(
                job_dir / "decode-receipt.json",
                base=job_dir,
                maximum_bytes=DECODE_RECEIPT_MAX_BYTES,
                description="decode job receipt",
            )
            payload = load_bounded_artifact(
                output_path,
                base=job_dir,
                maximum_bytes=DECODE_RESULT_MAX_BYTES,
                description="decode result",
            )
        try:
            receipt = json.loads(receipt_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecodeJobError("decode job receipt is unavailable") from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != DECODE_RECEIPT_SCHEMA
            or receipt.get("job_id") != job_id
            or receipt.get("capture_id") != capture_id
            or receipt.get("result_sha256") != result_sha256
            or receipt.get("result_bytes") != result_bytes
        ):
            raise DecodeJobError("decode job receipt binding is invalid")
        if len(payload) != result_bytes or hashlib.sha256(payload).hexdigest() != result_sha256:
            raise DecodeJobError("decode result changed after the job succeeded")
        return payload

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancelled_jobs.update(self._processes)
            self._cancelled_jobs.update(
                job_id for job_id, job in self._jobs.items() if job.status in {"queued", "running"}
            )
            processes = list(self._processes.values())
        for process in processes:
            stop_process(process)
        self._executor.shutdown(wait=False, cancel_futures=True)


def _service(request: Request) -> DecodeJobService:
    return request.app.state.decode_jobs


@router.get("/api/specs/decode-job-request")
async def decode_job_request_spec(request: Request) -> dict[str, Any]:
    path = request.app.state.settings.repo_root / "schemas" / "decode-job-request-v1.schema.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="decode-job-request-v1 schema is unavailable",
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=503, detail="decode-job-request-v1 schema is invalid")
    return value


@router.post("/api/captures/{capture_id}/decode-jobs", status_code=202)
async def submit_decode_job(
    capture_id: str,
    request: Request,
    response: Response,
    remote_host: str | None = Query(default=None),
) -> dict[str, Any]:
    normalized_remote_host = remote_host.strip() if remote_host and remote_host.strip() else None
    try:
        result = await run_in_threadpool(
            _service(request).submit,
            capture_id,
            remote_host=normalized_remote_host,
        )
    except DecodeJobsDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DecodeJobsBusy as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except DecodeJobNotReady as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DecodeJobError as exc:
        status_code = 404 if str(exc) == "capture not found" else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    response.headers["Location"] = f"/api/decode/jobs/{result['job_id']}"
    return result


@router.get("/api/captures/{capture_id}/decode-jobs")
async def list_capture_decode_jobs(capture_id: str, request: Request) -> dict[str, Any]:
    workspace: Workspace = request.app.state.workspace
    try:
        rows = await run_in_threadpool(workspace.list_captures)
    except WorkspaceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not any(row.get("capture_id") == capture_id for row in rows):
        raise HTTPException(status_code=404, detail="capture not found")
    try:
        jobs = _service(request).list_for_capture(capture_id)
    except DecodeJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"jobs": jobs}


@router.get("/api/decode/jobs")
async def list_decode_jobs(
    request: Request,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=DECODE_MAX_RETAINED),
) -> dict[str, Any]:
    try:
        jobs = _service(request).list_all(query=q, limit=limit)
    except DecodeJobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"jobs": jobs}


@router.get("/api/decode/jobs/{job_id}/result")
async def decode_job_result(job_id: str, request: Request) -> Response:
    try:
        payload = await run_in_threadpool(_service(request).result, job_id)
    except DecodeJobError as exc:
        status_code = 404 if str(exc) == "decode job not found" else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/api/decode/jobs/{job_id}")
async def decode_job_status(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return _service(request).status(job_id)
    except DecodeJobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/api/decode/jobs/{job_id}")
async def delete_decode_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_service(request).trash, job_id)
    except DecodeJobError as exc:
        detail = str(exc)
        status_code = 404 if detail == "decode job not found" else 409
        raise HTTPException(status_code=status_code, detail=detail) from exc
