"""Security-sensitive process and artifact helpers shared by compute jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

IDENTIFIER = re.compile(r"[a-f0-9]{32}")
SHA256 = re.compile(r"[a-f0-9]{64}")
RUNNER_SECRET_ENV_NAMES = {
    "CUBED_CORE_ADMIN_TOKEN",
    "GPG_AGENT_INFO",
    "KUBECONFIG",
    "NETRC",
    "SSH_AUTH_SOCK",
}
RUNNER_SECRET_ENV_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "PASSWD",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TICKET",
    "TOKEN",
)
_STAGE_MARKER = re.compile(
    r"^\[cubed-core:stage\][ \t]+(?P<token>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"(?:[ \t]+(?P<current>\d+)/(?P<total>\d+))?[ \t]*$",
    re.MULTILINE,
)
_RUNNER_ERROR_DETAIL_MAX_CHARS = 200


class JobRuntimeError(ValueError):
    pass


def require_object(
    value: Any,
    *,
    field: str,
    required: set[str],
    allowed: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobRuntimeError(f"{field} must be an object")
    missing = required - value.keys()
    if missing:
        raise JobRuntimeError(f"{field} is missing {sorted(missing)[0]}")
    extra = value.keys() - allowed
    if extra:
        raise JobRuntimeError(f"{field} contains unsupported field {sorted(extra)[0]}")
    return value


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def list_job_directories(jobs_root: Path) -> list[tuple[str, Path]]:
    """List immediate job-ID directories, failing closed on unsafe roots."""

    try:
        if jobs_root.is_symlink():
            return []
        entries = os.scandir(jobs_root)
    except OSError:
        return []
    found: list[tuple[str, Path]] = []
    try:
        for entry in entries:
            if not IDENTIFIER.fullmatch(entry.name):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            found.append((entry.name, jobs_root / entry.name))
    finally:
        entries.close()
    return found


def _secure_file(candidate: Path, *, base: Path, required: bool) -> Path | None:
    if not candidate.exists() and not candidate.is_symlink():
        if required:
            raise JobRuntimeError(f"required workspace file {candidate.name} is unavailable")
        return None
    relative = candidate.relative_to(base)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise JobRuntimeError(f"workspace input {relative.as_posix()} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise JobRuntimeError(f"workspace input {relative.as_posix()} is unavailable") from exc
    if not resolved.is_file():
        raise JobRuntimeError(f"workspace input {relative.as_posix()} is not a file")
    return resolved


def load_bounded_artifact(
    path: Path,
    *,
    base: Path,
    maximum_bytes: int,
    description: str,
) -> bytes:
    resolved = _secure_file(path, base=base, required=True)
    assert resolved is not None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise JobRuntimeError(f"{description} is not a regular file")
        if before.st_size <= 0:
            raise JobRuntimeError(f"{description} is empty")
        if before.st_size > maximum_bytes:
            raise JobRuntimeError(f"{description} exceeds the {maximum_bytes}-byte limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise JobRuntimeError(f"{description} is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise JobRuntimeError(f"{description} exceeds the {maximum_bytes}-byte limit")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise JobRuntimeError(f"{description} changed while it was being read")
    if len(payload) != before.st_size:
        raise JobRuntimeError(f"{description} changed while it was being read")
    return payload


def write_exclusive_json(
    path: Path,
    value: dict[str, Any],
    *,
    description: str,
) -> bytes:
    try:
        payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise JobRuntimeError(f"{description} is not finite JSON") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise JobRuntimeError(f"{description} could not be persisted") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload


def digest_regular_file(path: Path, *, maximum_bytes: int) -> dict[str, Any] | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            return None
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        return None
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True, slots=True)
class StageInfo:
    """Phase-progress fields derived from a runner's captured stdout markers."""

    stage: str | None
    stages_seen: tuple[str, ...]
    stage_progress: tuple[int, int] | None


def parse_stage_markers(log: str) -> StageInfo:
    """Parse flushed ``[cubed-core:stage] <token> [n/total]`` lines."""

    stage: str | None = None
    stages_seen: list[str] = []
    stage_progress: tuple[int, int] | None = None
    for match in _STAGE_MARKER.finditer(log):
        token = match.group("token")
        stage = token
        if token not in stages_seen:
            stages_seen.append(token)
        current = match.group("current")
        total = match.group("total")
        if current is not None and total is not None:
            stage_progress = (int(current), int(total))
    return StageInfo(
        stage=stage,
        stages_seen=tuple(stages_seen),
        stage_progress=stage_progress,
    )


def promote_runner_error_detail(log: str, *, schema: str, message: str) -> str:
    """Append a bounded structured runner error, or the last non-empty log line."""

    lines = log.splitlines()
    detail: str | None = None
    for line in reversed(lines):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("schema") != schema:
            continue
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
            detail = error["message"]
            break
    if detail is None:
        for line in reversed(lines):
            candidate = line.strip()
            if candidate:
                detail = candidate
                break
    if not detail:
        return message
    return f"{message}: {detail[:_RUNNER_ERROR_DETAIL_MAX_CHARS]}"


class BoundedLog:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.total = 0
        self.lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self.lock:
            self.total += len(chunk)
            if len(chunk) >= self.limit:
                self.data[:] = chunk[-self.limit :]
                return
            overflow = len(self.data) + len(chunk) - self.limit
            if overflow > 0:
                del self.data[:overflow]
            self.data.extend(chunk)

    def result(self) -> tuple[str, bool]:
        with self.lock:
            return self.data.decode("utf-8", errors="replace"), self.total > self.limit


def drain_stream(
    stream: BinaryIO,
    log: BoundedLog,
    on_update: Callable[[], None] | None = None,
) -> None:
    read = getattr(stream, "read1", stream.read)
    try:
        while chunk := read(8192):
            log.append(chunk)
            if on_update is not None:
                on_update()
    except (OSError, ValueError):
        return


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass
