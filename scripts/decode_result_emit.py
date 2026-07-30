"""Structured decode result emission for the research decode lane.

The research decoder (``scripts/trellis_gt.py``) is parity-pinned: its stdout,
its flags and its decode logic are frozen by a recorded baseline. This module is
the only place where a completed research decode is turned into a machine
readable document, so the decoder itself carries one small env-gated call and
nothing else.

The emitted document conforms to ``schemas/decode-result-v1.schema.json``. It is
written only when ``CUBED_RESULT_JSON`` names an absolute path, so a run without
that variable behaves exactly as it did before.

Status mapping follows the schema, which couples ``status`` to the endpoint
verdict: a run whose endpoint check reached the target state is ``completed``
and carries its moves, and a run that terminated normally without reaching the
endpoint is ``abstained`` and carries no moves. ``failed`` is never written here;
a crashed run is reported by the process exit code and leaves no document.

The config block is only partially knowable inside the decoder, which never sees
the runner's flag assembly. ``scripts/run_research_decode.sh`` exports its own
stamp before the run and overwrites the block afterwards, so the runner stays the
single authority on config stamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_ID = "cubed-core/decode-result"
SCHEMA_VERSION = 1
PROFILE = "local_camera_v1"
IMPLEMENTATION_ID = "cubed-core-research-decode-v1"

RESULT_PATH_ENV = "CUBED_RESULT_JSON"
RESULT_VIDEO_INPUT_ENV = "CUBED_RESULT_VIDEO_INPUT"
WORKSTATION_PATH_ENV = "CUBED_WORKSTATION_JSON"
RECORDING_ID_ENV = "CUBED_RECORDING_ID"
CFG_NAME_ENV = "CUBED_CFG_NAME"
CFG_HASH_ENV = "CUBED_CFG_HASH"
CFG_HASH_INPUT_ENV = "CUBED_CFG_HASH_INPUT"
EXTRAS_HASH_ENV = "CUBED_EXTRAS_HASH"

# The decoder is invoked directly in research work, without the runner that owns
# the config stamp. That case is labelled rather than guessed at.
UNSTAMPED_CONFIG_NAME = "unstamped-direct-invocation"
UNSTAMPED_CFG_HASH = "0"

_MOVE_RE = re.compile(r"^[UDLRFB](?:'|2)?$")
_RECORDING_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_CKSUM_RE = re.compile(r"^[0-9]+$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

_STATUS_COMPLETED = "completed"
_STATUS_ABSTAINED = "abstained"
WORKSTATION_MAX_BYTES = 32 * 1024**2


class DecodeResultError(ValueError):
    """A decode result could not be built or is structurally invalid."""


def _reject_nonfinite_json(value: str) -> None:
    raise DecodeResultError(f"non-finite JSON is not allowed: {value}")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the hex sha256 of a file, streamed so large artifacts are safe."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_receipts(
    entries: Iterable[tuple[str, str | os.PathLike[str] | None]],
) -> list[dict[str, str]]:
    """Hash the decode inputs that were actually used.

    ``entries`` are ``(id, path)`` pairs. An entry with no path, or a path that
    does not resolve to a file, is dropped: an optional input that a run did not
    use must not appear as a receipt. The schema requires at least two receipts,
    so a run that cannot account for two inputs is an error rather than a
    thinner document.
    """

    receipts: list[dict[str, str]] = []
    seen: set[str] = set()
    for identifier, path in entries:
        if not identifier:
            raise DecodeResultError("input receipt id must not be empty")
        if identifier in seen:
            raise DecodeResultError(f"duplicate input receipt id: {identifier}")
        if path is None or str(path) == "":
            continue
        if not os.path.isfile(path):
            continue
        seen.add(identifier)
        receipts.append({"id": identifier, "sha256": sha256_file(path)})
    if len(receipts) < 2:
        raise DecodeResultError(
            f"decode result needs at least 2 input receipts, resolved {len(receipts)}"
        )
    return receipts


def derive_recording_id(tag: str, receipts: Sequence[Mapping[str, str]]) -> str:
    """Derive a stable 32 hex recording id from the tag and the input identities.

    Used when the caller does not supply one. Two runs over the same tag and the
    same input bytes derive the same id, and a different capture derives a
    different id.
    """

    digest = hashlib.sha256()
    digest.update(b"cubed-core/decode-result/recording-id/v1\n")
    digest.update(tag.encode("utf-8"))
    digest.update(b"\n")
    for receipt in sorted(receipts, key=lambda item: item["id"]):
        digest.update(receipt["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(receipt["sha256"].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def resolve_recording_id(
    tag: str,
    receipts: Sequence[Mapping[str, str]],
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Prefer the caller supplied recording id, else derive one."""

    env = os.environ if environ is None else environ
    supplied = (env.get(RECORDING_ID_ENV) or "").strip().lower()
    if supplied:
        if not _RECORDING_ID_RE.match(supplied):
            raise DecodeResultError(
                f"{RECORDING_ID_ENV} must be 32 lowercase hex characters, got {supplied!r}"
            )
        return supplied
    return derive_recording_id(tag, receipts)


def config_from_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the config block from the stamp the runner exports, if any.

    The decoder does not assemble the canonical flag set and cannot compute the
    runner's cksum. When the stamp is absent the block is labelled unstamped
    rather than filled with a fabricated hash, and the runner overwrites the
    whole block after a successful run.
    """

    env = os.environ if environ is None else environ
    name = (env.get(CFG_NAME_ENV) or "").strip()
    cfg_hash = (env.get(CFG_HASH_ENV) or "").strip()
    if not name or not _CKSUM_RE.match(cfg_hash):
        return {
            "name": UNSTAMPED_CONFIG_NAME,
            "cfg_hash": UNSTAMPED_CFG_HASH,
            "cfg_hash_algorithm": "posix-cksum",
        }
    config: dict[str, Any] = {
        "name": name,
        "cfg_hash": cfg_hash,
        "cfg_hash_algorithm": "posix-cksum",
    }
    cfg_hash_input = env.get(CFG_HASH_INPUT_ENV) or ""
    if cfg_hash_input:
        config["cfg_hash_input"] = cfg_hash_input
    extras_hash = (env.get(EXTRAS_HASH_ENV) or "").strip()
    if extras_hash:
        if not _CKSUM_RE.match(extras_hash):
            raise DecodeResultError(
                f"{EXTRAS_HASH_ENV} must be a posix cksum value, got {extras_hash!r}"
            )
        config["extras_hash"] = extras_hash
    return config


def runtime_version() -> str:
    """Version of the cubed-core checkout this decode ran from."""

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.10+
        return "unpackaged-research-checkout"
    try:
        return version("cubed-core")
    except PackageNotFoundError:
        return "unpackaged-research-checkout"


def _numpy_version() -> str | None:
    try:
        import numpy
    except ImportError:  # pragma: no cover - the decode lane always has numpy
        return None
    return str(numpy.__version__)


def _finished_at(now: datetime | None) -> str:
    value = datetime.now(timezone.utc) if now is None else now
    if value.tzinfo is None:
        raise DecodeResultError("finished_at must be timezone aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_moves(moves: Iterable[str]) -> list[str]:
    tokens = [str(token) for token in moves]
    bad = [token for token in tokens if not _MOVE_RE.match(token)]
    if bad:
        raise DecodeResultError(f"non-canonical move tokens in decode result: {bad[:8]}")
    return tokens


def load_workstation_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Load the native runner's browser-safe same-pass projection, if armed."""

    env = os.environ if environ is None else environ
    raw_path = (env.get(WORKSTATION_PATH_ENV) or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise DecodeResultError("workstation payload path is unavailable")
    size = path.stat().st_size
    if size <= 0 or size > WORKSTATION_MAX_BYTES:
        raise DecodeResultError("workstation payload is empty or too large")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeResultError("workstation payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DecodeResultError("workstation payload must be an object")
    return value


def _plain_orientation(value: Any) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return [f"{key}:{value[key]}" for key in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def _checkpoint_reconstruction(
    workstation: Mapping[str, Any],
    tracker_info: Mapping[str, Any],
    canonical_moves: Sequence[str],
) -> dict[str, Any] | None:
    """Build a state replay from decoder-owned window checkpoints.

    A checkpoint establishes the state after one whole scrub window. It does
    not identify when any move in that window occurred. Repeating its frame for
    each raw move lets the browser apply the shared-frame group atomically.
    """

    if tracker_info.get("reconstruction_checkpoint_valid") is not True:
        return None
    raw_moves = tracker_info.get("moves_bt")
    raw_checkpoints = tracker_info.get("reconstruction_checkpoints")
    declared_checkpoint_count = tracker_info.get("reconstruction_checkpoint_count")
    declared_move_count = tracker_info.get("reconstruction_checkpoint_move_count")
    if (
        not isinstance(raw_moves, (list, tuple))
        or not raw_moves
        or len(raw_moves) > 2_000
        or not isinstance(raw_checkpoints, (list, tuple))
        or not raw_checkpoints
        or len(raw_checkpoints) > 2_000
        or type(declared_checkpoint_count) is not int
        or type(declared_move_count) is not int
        or declared_checkpoint_count != len(raw_checkpoints)
        or declared_move_count != len(raw_moves)
    ):
        return None
    try:
        timeline_moves = _canonical_moves(raw_moves)
    except DecodeResultError:
        return None

    checkpoint_groups = []
    running_move_count = 0
    for checkpoint in raw_checkpoints:
        if not isinstance(checkpoint, Mapping):
            return None
        frame = checkpoint.get("frame")
        move_count = checkpoint.get("move_count")
        if (
            type(frame) is not int
            or frame < 0
            or type(move_count) is not int
            or move_count <= 0
            or move_count > len(timeline_moves)
        ):
            return None
        running_move_count += move_count
        if running_move_count > declared_move_count or running_move_count > len(timeline_moves):
            return None
        checkpoint_groups.append((frame, move_count))
    checkpoint_frames = [frame for frame, _move_count in checkpoint_groups]
    if running_move_count != len(timeline_moves) or any(
        a >= b for a, b in zip(checkpoint_frames, checkpoint_frames[1:], strict=False)
    ):
        return None
    expanded_frames = [
        frame for frame, move_count in checkpoint_groups for _index in range(move_count)
    ]

    initialization = workstation.get("initialization")
    window = workstation.get("window")
    scramble = initialization.get("scramble") if isinstance(initialization, Mapping) else None
    if (
        not isinstance(scramble, str)
        or not isinstance(window, (list, tuple))
        or len(window) != 2
        or any(type(frame) is not int for frame in window)
        or checkpoint_frames[0] <= window[0]
        or any(frame < window[0] or frame > window[1] for frame in checkpoint_frames)
    ):
        return None
    try:
        from cubed_core.cube import Cube

        canonical = Cube.solved().apply_algorithm(scramble).apply_algorithm(canonical_moves)
        replay = Cube.solved().apply_algorithm(scramble)
        states = [replay.state]
        for move in timeline_moves:
            replay.apply_move(move)
            states.append(replay.state)
    except (ImportError, TypeError, ValueError):
        return None
    if replay != canonical or not replay.is_solved():
        return None
    return {
        "states": states,
        "solved_reached": True,
        "timeline": {
            "moves": [
                {"move": move, "frame": frame}
                for move, frame in zip(timeline_moves, expanded_frames, strict=True)
            ],
            "timing_basis": "decoder-checkpoint",
        },
    }


def workstation_with_decode_timeline(
    workstation: Mapping[str, Any],
    *,
    info: Mapping[str, Any] | None,
    moves: Sequence[str],
    events: Iterable[int] = (),
) -> dict[str, Any]:
    """Add trellis and decoder-owned playback data to a same-pass projection."""

    value = deepcopy(dict(workstation))
    # The top-level result owns the canonical sequence. A staged workstation
    # may contain stale timing data, so replace it only with checked data from
    # this decoder pass.
    value.pop("sequence", None)
    value.pop("reconstruction", None)
    event_frames = sorted({int(frame) for frame in events if int(frame) >= 0})
    if event_frames:
        value["events"] = event_frames

    tracker_info = info if isinstance(info, Mapping) else {}
    meta = tracker_info.get("meta")
    if isinstance(meta, (list, tuple)) and meta:
        spans = []
        for entry in meta:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            f0, f1, raw_top = entry
            if type(f0) is not int or type(f1) is not int or f0 < 0 or f1 < f0:
                continue
            top = []
            if isinstance(raw_top, (list, tuple)):
                for candidate in raw_top:
                    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
                        continue
                    path, orientation, score = candidate
                    try:
                        score_value = float(score)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(score_value):
                        continue
                    top.append(
                        {
                            "path": (
                                [str(token) for token in path]
                                if isinstance(path, (list, tuple))
                                else str(path)
                            ),
                            "orientation": _plain_orientation(orientation),
                            "score": score_value,
                        }
                    )
            event = next(
                (frame for frame in event_frames if f0 <= frame <= f1),
                None,
            )
            spans.append({"f0": f0, "f1": f1, "event": event, "top": top})
        if spans:
            value["trellis"] = {"spans": spans}

    canonical_moves = _canonical_moves(moves)
    reconstruction = _checkpoint_reconstruction(
        value,
        tracker_info,
        canonical_moves,
    )
    checkpoint_projection_declared = any(
        key in tracker_info
        for key in (
            "reconstruction_checkpoints",
            "reconstruction_checkpoint_count",
            "reconstruction_checkpoint_move_count",
            "reconstruction_checkpoint_valid",
        )
    )
    if reconstruction is not None:
        # Checkpoints are the stronger reconstruction projection even when the
        # canonical word happens to retain a one-to-one legacy frame list.
        value["reconstruction"] = reconstruction
    elif not checkpoint_projection_declared:
        move_frames = tracker_info.get("move_frames")
        window = value.get("window")
        if (
            canonical_moves
            and isinstance(move_frames, (list, tuple))
            and len(move_frames) == len(canonical_moves)
            and all(type(frame) is int and frame >= 0 for frame in move_frames)
            and all(a <= b for a, b in zip(move_frames, move_frames[1:], strict=False))
            and isinstance(window, (list, tuple))
            and len(window) == 2
            and all(type(frame) is int for frame in window)
            and move_frames[0] > window[0]
            and all(window[0] <= frame <= window[1] for frame in move_frames)
        ):
            value["sequence"] = {
                "moves": [
                    {"move": move, "frame": frame}
                    for move, frame in zip(canonical_moves, move_frames, strict=True)
                ],
                "timing_basis": "canonical",
            }
    return value


def build_decode_result(
    *,
    tag: str,
    moves: Iterable[str],
    solved_reached: bool | None,
    inputs: Iterable[tuple[str, str | os.PathLike[str] | None]],
    implementation_path: str | os.PathLike[str] | None = None,
    recording_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    numpy_version: str | None = None,
    workstation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a decode-result-v1 document for a research decode that terminated.

    ``solved_reached`` is the decoder's own endpoint verdict. It decides the
    status, because the schema only allows ``completed`` together with a reached
    endpoint, and it forbids moves on any other status.
    """

    receipts = input_receipts(inputs)
    resolved_id = (
        resolve_recording_id(tag, receipts, environ=environ)
        if recording_id is None
        else recording_id
    )
    if not _RECORDING_ID_RE.match(resolved_id):
        raise DecodeResultError(
            f"recording_id must be 32 lowercase hex characters: {resolved_id!r}"
        )

    reached = None if solved_reached is None else bool(solved_reached)
    if reached is True:
        status = _STATUS_COMPLETED
        emitted_moves = _canonical_moves(moves)
    else:
        status = _STATUS_ABSTAINED
        emitted_moves = []

    provenance: dict[str, Any] = {
        "runtime_version": runtime_version(),
        "implementation_id": IMPLEMENTATION_ID,
        "python_version": platform.python_version(),
        "finished_at": _finished_at(now),
    }
    if implementation_path is not None:
        provenance["implementation_sha256"] = sha256_file(implementation_path)
    resolved_numpy = _numpy_version() if numpy_version is None else str(numpy_version)
    if resolved_numpy:
        provenance["numpy_version"] = resolved_numpy

    document = {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "recording_id": resolved_id,
        "status": status,
        "profile": PROFILE,
        "config": dict(config) if config is not None else config_from_env(environ),
        "inputs": receipts,
        "moves": emitted_moves,
        "endpoint": {"solved_reached": reached},
        "evaluation": None,
        "provenance": provenance,
    }
    if workstation is not None:
        document["workstation"] = deepcopy(dict(workstation))
    validate_decode_result(document)
    return document


_REQUIRED_TOP_LEVEL = (
    "schema",
    "schema_version",
    "recording_id",
    "status",
    "profile",
    "config",
    "inputs",
    "moves",
    "endpoint",
    "evaluation",
    "provenance",
)


def validate_decode_result(document: Mapping[str, Any]) -> None:
    """Structural check of a decode-result-v1 document.

    Mirrors the parts of the schema that this lane can violate. It runs on every
    build and again after the runner overwrites the config block, so a malformed
    document is never left on disk.
    """

    if not isinstance(document, Mapping):
        raise DecodeResultError("decode result must be a JSON object")
    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in document]
    if missing:
        raise DecodeResultError(f"decode result is missing required fields: {missing}")
    unknown = [key for key in document if key not in {*_REQUIRED_TOP_LEVEL, "workstation"}]
    if unknown:
        raise DecodeResultError(f"decode result has unknown fields: {unknown}")
    if document["schema"] != SCHEMA_ID:
        raise DecodeResultError(f"schema must be {SCHEMA_ID}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise DecodeResultError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(document["recording_id"], str) or not _RECORDING_ID_RE.match(
        document["recording_id"]
    ):
        raise DecodeResultError("recording_id must be 32 lowercase hex characters")
    if document["status"] not in {_STATUS_COMPLETED, _STATUS_ABSTAINED}:
        raise DecodeResultError(f"unknown status: {document['status']!r}")
    if document["profile"] != PROFILE:
        raise DecodeResultError(f"unknown profile: {document['profile']!r}")
    if document["evaluation"] is not None:
        raise DecodeResultError("the research decode lane always writes evaluation null")

    _validate_config(document["config"])
    _validate_inputs(document["inputs"])

    moves = document["moves"]
    if not isinstance(moves, list):
        raise DecodeResultError("moves must be a list")
    _canonical_moves(moves)

    endpoint = document["endpoint"]
    if not isinstance(endpoint, Mapping) or set(endpoint) != {"solved_reached"}:
        raise DecodeResultError("endpoint must carry exactly solved_reached")
    reached = endpoint["solved_reached"]
    if reached is not None and not isinstance(reached, bool):
        raise DecodeResultError("endpoint.solved_reached must be a boolean or null")
    if document["status"] == _STATUS_COMPLETED and reached is not True:
        raise DecodeResultError("a completed decode must have solved_reached true")
    if document["status"] != _STATUS_COMPLETED and moves:
        raise DecodeResultError(f"a {document['status']} decode must carry no moves")

    _validate_provenance(document["provenance"])
    if "workstation" in document:
        _validate_workstation(
            document["workstation"],
            moves=moves,
            solved_reached=reached,
        )


def _validate_workstation(
    workstation: Any,
    *,
    moves: Sequence[str],
    solved_reached: bool | None,
) -> None:
    """Guard the identity-critical subset; the shipped JSON Schema is exhaustive."""

    if not isinstance(workstation, Mapping):
        raise DecodeResultError("workstation must be an object")
    if (
        workstation.get("schema") != "cubed-core/decode-workstation-v1"
        or workstation.get("schema_version") != 1
        or not isinstance(workstation.get("video"), Mapping)
        or not isinstance(workstation.get("initialization"), Mapping)
        or not isinstance(workstation.get("window"), list)
        or not isinstance(workstation.get("warnings"), list)
        or not isinstance(workstation.get("frames"), Mapping)
    ):
        raise DecodeResultError("workstation payload is incomplete")
    scramble = workstation["initialization"].get("scramble")
    if not isinstance(scramble, str) or not scramble:
        raise DecodeResultError("workstation initialization scramble is missing")
    sequence = workstation.get("sequence")
    if sequence is not None:
        if (
            not isinstance(sequence, Mapping)
            or sequence.get("timing_basis") != "canonical"
            or not isinstance(sequence.get("moves"), list)
        ):
            raise DecodeResultError("workstation sequence must use canonical timing")
        sequence_moves: list[str] = []
        for entry in sequence["moves"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("move"), str):
                raise DecodeResultError("workstation sequence moves are invalid")
            sequence_moves.append(entry["move"])
        if sequence_moves != list(moves):
            raise DecodeResultError("workstation sequence does not match top-level decode moves")
    reconstruction = workstation.get("reconstruction")
    if reconstruction is not None:
        if (
            not isinstance(reconstruction, Mapping)
            or type(reconstruction.get("solved_reached")) is not bool
        ):
            raise DecodeResultError("workstation reconstruction is invalid")
        if reconstruction["solved_reached"] != solved_reached:
            raise DecodeResultError(
                "workstation reconstruction endpoint does not match the top-level endpoint"
            )


def _validate_config(config: Any) -> None:
    if not isinstance(config, Mapping):
        raise DecodeResultError("config must be an object")
    allowed = {"name", "cfg_hash", "cfg_hash_algorithm", "cfg_hash_input", "extras_hash"}
    unknown = [key for key in config if key not in allowed]
    if unknown:
        raise DecodeResultError(f"config has unknown fields: {unknown}")
    for key in ("name", "cfg_hash", "cfg_hash_algorithm"):
        if key not in config:
            raise DecodeResultError(f"config is missing {key}")
    if not isinstance(config["name"], str) or not config["name"]:
        raise DecodeResultError("config.name must be a non-empty string")
    if not isinstance(config["cfg_hash"], str) or not _CKSUM_RE.match(config["cfg_hash"]):
        raise DecodeResultError("config.cfg_hash must be a posix cksum value")
    if config["cfg_hash_algorithm"] != "posix-cksum":
        raise DecodeResultError("config.cfg_hash_algorithm must be posix-cksum")
    if "cfg_hash_input" in config:
        value = config["cfg_hash_input"]
        if not isinstance(value, str) or not value:
            raise DecodeResultError("config.cfg_hash_input must be a non-empty string")
    if "extras_hash" in config:
        value = config["extras_hash"]
        if value is not None and (not isinstance(value, str) or not _CKSUM_RE.match(value)):
            raise DecodeResultError("config.extras_hash must be a posix cksum value or null")


def _validate_inputs(inputs: Any) -> None:
    if not isinstance(inputs, list) or len(inputs) < 2:
        raise DecodeResultError("inputs must be a list of at least 2 receipts")
    for receipt in inputs:
        if not isinstance(receipt, Mapping) or set(receipt) != {"id", "sha256"}:
            raise DecodeResultError("each input receipt must carry exactly id and sha256")
        if not isinstance(receipt["id"], str) or not receipt["id"]:
            raise DecodeResultError("input receipt id must be a non-empty string")
        if not isinstance(receipt["sha256"], str) or not _SHA256_RE.match(receipt["sha256"]):
            raise DecodeResultError("input receipt sha256 must be 64 lowercase hex characters")


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, Mapping):
        raise DecodeResultError("provenance must be an object")
    allowed = {
        "runtime_version",
        "implementation_id",
        "implementation_sha256",
        "python_version",
        "numpy_version",
        "finished_at",
    }
    unknown = [key for key in provenance if key not in allowed]
    if unknown:
        raise DecodeResultError(f"provenance has unknown fields: {unknown}")
    for key in ("runtime_version", "finished_at"):
        if key not in provenance:
            raise DecodeResultError(f"provenance is missing {key}")
    for key in ("runtime_version", "implementation_id", "python_version", "numpy_version"):
        if key in provenance and (not isinstance(provenance[key], str) or not provenance[key]):
            raise DecodeResultError(f"provenance.{key} must be a non-empty string")
    if "implementation_sha256" in provenance:
        value = provenance["implementation_sha256"]
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise DecodeResultError("provenance.implementation_sha256 must be a hex sha256")
    finished_at = provenance["finished_at"]
    if not isinstance(finished_at, str):
        raise DecodeResultError("provenance.finished_at must be a string")
    try:
        datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DecodeResultError(f"provenance.finished_at is not a date-time: {exc}") from exc


def write_decode_result(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    """Validate then atomically write the document to an absolute path."""

    target = Path(path)
    if not target.is_absolute():
        raise DecodeResultError(f"decode result path must be absolute: {target}")
    validate_decode_result(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise DecodeResultError("decode result is not strict JSON") from exc
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def read_decode_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a previously written document without validating it."""

    with open(path, encoding="utf-8") as stream:
        document = json.load(stream, parse_constant=_reject_nonfinite_json)
    if not isinstance(document, dict):
        raise DecodeResultError("decode result file must contain a JSON object")
    return document


def inject_config(
    path: str | os.PathLike[str],
    *,
    name: str,
    cfg_hash: str,
    cfg_hash_input: str | None = None,
    extras_hash: str | None = None,
) -> dict[str, Any]:
    """Overwrite the config block of a written document, then re-validate it.

    The runner owns the config stamp, so it replaces whatever the in-process
    build could infer. The document is rewritten only after it validates.
    """

    document = read_decode_result(path)
    config: dict[str, Any] = {
        "name": name,
        "cfg_hash": cfg_hash,
        "cfg_hash_algorithm": "posix-cksum",
    }
    if cfg_hash_input:
        config["cfg_hash_input"] = cfg_hash_input
    if extras_hash:
        config["extras_hash"] = extras_hash
    document["config"] = config
    write_decode_result(path, document)
    return document


def emit_from_decode(
    *,
    tag: str,
    moves: Iterable[str],
    solved_reached: bool | None,
    inputs: Iterable[tuple[str, str | os.PathLike[str] | None]],
    implementation_path: str | os.PathLike[str] | None = None,
    numpy_version: str | None = None,
    environ: Mapping[str, str] | None = None,
    workstation: Mapping[str, Any] | None = None,
) -> Path | None:
    """Build and write the result when ``CUBED_RESULT_JSON`` names a path.

    This is the single entry point the research decoder calls. It returns None
    when the lane is not armed, so the decoder's default behaviour is untouched.
    """

    env = os.environ if environ is None else environ
    destination = (env.get(RESULT_PATH_ENV) or "").strip()
    if not destination:
        return None
    resolved_inputs = list(inputs)
    video_input = (env.get(RESULT_VIDEO_INPUT_ENV) or "").strip()
    if video_input:
        if any(identifier == "video" for identifier, _path in resolved_inputs):
            raise DecodeResultError("decode result video input was supplied more than once")
        if not os.path.isfile(video_input):
            raise DecodeResultError("decode result video input is unavailable")
        resolved_inputs.insert(0, ("video", video_input))
    document = build_decode_result(
        tag=tag,
        moves=moves,
        solved_reached=solved_reached,
        inputs=resolved_inputs,
        implementation_path=implementation_path,
        environ=env,
        numpy_version=numpy_version,
        workstation=workstation,
    )
    return write_decode_result(destination, document)
