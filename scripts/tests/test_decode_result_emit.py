"""Tests for the research decode result emitter.

These never import ``scripts.trellis_gt``: the decoder is parity-pinned and
carries a 24k line import closure that needs the research runtime. The helper is
pure, so it is exercised directly and every document it builds is checked
against ``schemas/decode-result-v1.schema.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import decode_result_emit as emit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "decode-result-v1.schema.json"

FIXED_NOW = datetime(2026, 7, 25, 12, 30, 45, tzinfo=timezone.utc)


def _schema_validate(document: dict) -> None:
    """Validate against the real schema when jsonschema is installed."""

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=document, schema=schema)


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


@pytest.fixture
def decode_inputs(tmp_path: Path) -> list[tuple[str, Path]]:
    return [
        ("reads", _write(tmp_path / "reads_gt20_occaware.pkl", b"reads-bytes")),
        ("events", _write(tmp_path / "motion_events_gt20.json", b"{}")),
        ("centroids", _write(tmp_path / "calibration_gan12.json", b'{"white": []}')),
    ]


@pytest.fixture
def stamp_env() -> dict[str, str]:
    return {
        "CUBED_CFG_NAME": "move-gate-dgap-trustsoft-scrub-statefulom-robust",
        "CUBED_CFG_HASH": "1234567890",
        "CUBED_CFG_HASH_INPUT": "--om-pf-reads ENV:MICROREST=1",
        "CUBED_EXTRAS_HASH": "987654321",
    }


def _build(inputs, environ, **overrides):
    kwargs = dict(
        tag="gt20",
        moves=["R", "U'", "F2"],
        solved_reached=True,
        inputs=inputs,
        environ=environ,
        now=FIXED_NOW,
        numpy_version="1.26.4",
    )
    kwargs.update(overrides)
    return emit.build_decode_result(**kwargs)


def _workstation() -> dict:
    return {
        "schema": "cubed-core/decode-workstation-v1",
        "schema_version": 1,
        "video": {
            "sha256": "a" * 64,
            "bytes": 1234,
            "fps": 120.0,
            "frame_count": 40,
            "width": 1920,
            "height": 1080,
            "encoded": {
                "fps": 120.0,
                "frame_count": 40,
                "width": 1920,
                "height": 1080,
            },
        },
        "initialization": {"scramble": "R U R'"},
        "window": [0, 39],
        "warnings": [],
        "frames": {
            "10": {
                "motion": 1.25,
                "aligned": 0.9,
                "aligned_streak": 4,
                "face_count": 2,
            }
        },
    }


# --- sha256_file ------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"a" * (3 * 1024 * 1024 + 17)
    path = _write(tmp_path / "artifact.bin", payload)
    assert emit.sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --- input receipts ---------------------------------------------------------


def test_input_receipts_hash_every_used_artifact(decode_inputs) -> None:
    receipts = emit.input_receipts(decode_inputs)
    assert [receipt["id"] for receipt in receipts] == ["reads", "events", "centroids"]
    reads = next(receipt for receipt in receipts if receipt["id"] == "reads")
    assert reads["sha256"] == hashlib.sha256(b"reads-bytes").hexdigest()


def test_input_receipts_drop_unused_and_missing_paths(tmp_path: Path, decode_inputs) -> None:
    entries = [
        decode_inputs[0],
        ("events", ""),
        ("gyro", tmp_path / "not-there.json"),
        decode_inputs[2],
    ]
    receipts = emit.input_receipts(entries)
    assert [receipt["id"] for receipt in receipts] == ["reads", "centroids"]


def test_input_receipts_reject_fewer_than_two(decode_inputs) -> None:
    with pytest.raises(emit.DecodeResultError, match="at least 2 input receipts"):
        emit.input_receipts([decode_inputs[0]])


def test_input_receipts_reject_duplicate_ids(decode_inputs) -> None:
    duplicated = [decode_inputs[0], ("reads", decode_inputs[1][1]), decode_inputs[2]]
    with pytest.raises(emit.DecodeResultError, match="duplicate input receipt id"):
        emit.input_receipts(duplicated)


# --- recording id -----------------------------------------------------------


def test_recording_id_is_derived_and_stable(decode_inputs) -> None:
    receipts = emit.input_receipts(decode_inputs)
    first = emit.resolve_recording_id("gt20", receipts, environ={})
    second = emit.resolve_recording_id("gt20", receipts, environ={})
    assert first == second
    assert len(first) == 32
    assert emit.resolve_recording_id("gt21", receipts, environ={}) != first


def test_supplied_recording_id_wins(decode_inputs) -> None:
    receipts = emit.input_receipts(decode_inputs)
    supplied = "f" * 32
    resolved = emit.resolve_recording_id("gt20", receipts, environ={"CUBED_RECORDING_ID": supplied})
    assert resolved == supplied
    assert resolved != emit.derive_recording_id("gt20", receipts)


def test_malformed_recording_id_is_rejected(decode_inputs) -> None:
    receipts = emit.input_receipts(decode_inputs)
    with pytest.raises(emit.DecodeResultError, match="32 lowercase hex"):
        emit.resolve_recording_id("gt20", receipts, environ={"CUBED_RECORDING_ID": "nope"})


# --- config block -----------------------------------------------------------


def test_config_from_env_uses_the_runner_stamp(stamp_env) -> None:
    config = emit.config_from_env(stamp_env)
    assert config == {
        "name": "move-gate-dgap-trustsoft-scrub-statefulom-robust",
        "cfg_hash": "1234567890",
        "cfg_hash_algorithm": "posix-cksum",
        "cfg_hash_input": "--om-pf-reads ENV:MICROREST=1",
        "extras_hash": "987654321",
    }


def test_config_from_env_labels_a_direct_invocation() -> None:
    config = emit.config_from_env({})
    assert config["name"] == emit.UNSTAMPED_CONFIG_NAME
    assert config["cfg_hash"] == emit.UNSTAMPED_CFG_HASH
    assert "cfg_hash_input" not in config


def test_config_from_env_rejects_a_non_cksum_extras_hash(stamp_env) -> None:
    stamp_env["CUBED_EXTRAS_HASH"] = "deadbeef"
    with pytest.raises(emit.DecodeResultError, match="posix cksum"):
        emit.config_from_env(stamp_env)


# --- document construction --------------------------------------------------


def test_completed_document_matches_the_schema(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    _schema_validate(document)
    assert document["schema"] == "cubed-core/decode-result"
    assert document["schema_version"] == 1
    assert document["status"] == "completed"
    assert document["profile"] == "local_camera_v1"
    assert document["moves"] == ["R", "U'", "F2"]
    assert document["endpoint"] == {"solved_reached": True}
    assert document["evaluation"] is None
    assert document["config"]["cfg_hash"] == "1234567890"
    assert document["provenance"]["finished_at"] == "2026-07-25T12:30:45Z"
    assert document["provenance"]["numpy_version"] == "1.26.4"
    assert document["provenance"]["implementation_id"] == emit.IMPLEMENTATION_ID


def test_optional_workstation_projection_matches_the_schema(decode_inputs, stamp_env) -> None:
    document = _build(
        decode_inputs,
        stamp_env,
        workstation=_workstation(),
    )

    _schema_validate(document)
    assert document["workstation"]["video"]["sha256"] == "a" * 64
    assert document["workstation"]["frames"]["10"]["aligned_streak"] == 4


def test_decode_timeline_uses_decoder_owned_frames_and_trellis_candidates() -> None:
    workstation = _workstation()
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "meta": [
                (
                    8,
                    18,
                    [
                        ("R U", "up/front", -1.25),
                        (["F2"], {"up": "U", "front": "F"}, float("inf")),
                    ],
                )
            ],
            "move_frames": [11, 17],
        },
        moves=["R", "U"],
        events=[15],
    )

    assert value["events"] == [15]
    assert value["trellis"]["spans"] == [
        {
            "f0": 8,
            "f1": 18,
            "event": 15,
            "top": [
                {
                    "path": "R U",
                    "orientation": "up/front",
                    "score": -1.25,
                }
            ],
        }
    ]
    assert value["sequence"] == {
        "moves": [{"move": "R", "frame": 11}, {"move": "U", "frame": 17}],
        "timing_basis": "canonical",
    }
    # The input staging document remains reusable and untouched.
    assert "events" not in workstation
    assert "trellis" not in workstation


def test_decode_timeline_prefers_atomic_checkpoints_over_canonical_timing() -> None:
    workstation = _workstation()
    workstation["initialization"]["scramble"] = "U' R'"
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "move_frames": [11, 17],
            "moves_bt": ["R", "U"],
            "reconstruction_checkpoints": [
                {"frame": 17, "move_count": 2},
            ],
            "reconstruction_checkpoint_count": 1,
            "reconstruction_checkpoint_move_count": 2,
            "reconstruction_checkpoint_valid": True,
        },
        moves=["R", "U"],
    )

    assert "sequence" not in value
    assert value["reconstruction"]["timeline"] == {
        "moves": [{"move": "R", "frame": 17}, {"move": "U", "frame": 17}],
        "timing_basis": "decoder-checkpoint",
    }


def test_decode_timeline_drops_sequence_for_an_abstention() -> None:
    workstation = _workstation()
    workstation["sequence"] = {
        "moves": [{"move": "R", "frame": 10}],
        "timing_basis": "canonical",
    }

    value = emit.workstation_with_decode_timeline(
        workstation,
        info={"move_frames": [10]},
        moves=[],
    )

    assert "sequence" not in value


def test_decode_timeline_omits_raw_sequence_without_checkpoint_authority() -> None:
    workstation = _workstation()
    workstation["sequence"] = {
        "moves": [{"move": "F", "frame": 1}],
        "timing_basis": "canonical",
    }

    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "moves_bt": ["R", "U"],
            "reconstruction_checkpoints": [
                {"frame": 17, "move_count": 2},
            ],
            "reconstruction_checkpoint_count": 1,
            "reconstruction_checkpoint_move_count": 2,
        },
        moves=["R", "U"],
    )

    assert "sequence" not in value


def test_decode_timeline_does_not_fall_back_from_rejected_checkpoints() -> None:
    workstation = _workstation()
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "move_frames": [0, 0],
            "moves_bt": ["R", "U"],
            "reconstruction_checkpoints": [
                {"frame": 0, "move_count": 2},
            ],
            "reconstruction_checkpoint_count": 1,
            "reconstruction_checkpoint_move_count": 2,
            "reconstruction_checkpoint_valid": True,
        },
        moves=["R", "U"],
    )

    assert "reconstruction" not in value
    assert "sequence" not in value


def test_decode_timeline_legacy_sequence_starts_after_the_scramble_frame() -> None:
    workstation = _workstation()
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={"move_frames": [0, 10]},
        moves=["R", "U"],
    )

    assert "sequence" not in value


def test_decode_timeline_keeps_an_honest_checkpoint_reconstruction(
    decode_inputs,
    stamp_env,
) -> None:
    workstation = _workstation()
    workstation["initialization"]["scramble"] = "D' U2"
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "moves_bt": ["U", "D", "U"],
            "reconstruction_checkpoints": [
                {"frame": 10, "move_count": 1},
                {"frame": 30, "move_count": 2},
            ],
            "reconstruction_checkpoint_count": 2,
            "reconstruction_checkpoint_move_count": 3,
            "reconstruction_checkpoint_valid": True,
        },
        moves=["U2", "D"],
    )

    assert "sequence" not in value
    reconstruction = value["reconstruction"]
    assert reconstruction["solved_reached"] is True
    assert len(reconstruction["states"]) == 4
    assert reconstruction["states"][0] != reconstruction["states"][-1]
    assert reconstruction["timeline"] == {
        "moves": [
            {"move": "U", "frame": 10},
            {"move": "D", "frame": 30},
            {"move": "U", "frame": 30},
        ],
        "timing_basis": "decoder-checkpoint",
    }
    _schema_validate(
        _build(
            decode_inputs,
            stamp_env,
            moves=["U2", "D"],
            workstation=value,
        )
    )
    abstained = _build(
        decode_inputs,
        stamp_env,
        moves=[],
        solved_reached=False,
        workstation={
            **value,
            "reconstruction": {
                **reconstruction,
                "solved_reached": False,
            },
        },
    )
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        _schema_validate(abstained)


@pytest.mark.parametrize(
    ("raw_moves", "checkpoints", "valid", "checkpoint_count", "move_count"),
    [
        (
            ["U", "D", "U"],
            [{"frame": 20, "move_count": 1}, {"frame": 10, "move_count": 2}],
            True,
            2,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 10, "move_count": 2}],
            True,
            1,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 40, "move_count": 3}],
            True,
            1,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 0, "move_count": 3}],
            True,
            1,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 10, "move_count": 3}],
            False,
            1,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 10, "move_count": 3}],
            True,
            2,
            3,
        ),
        (
            ["U", "D", "U"],
            [{"frame": 10, "move_count": 10**18}],
            True,
            1,
            3,
        ),
        (
            ["R"],
            [{"frame": 10, "move_count": 1}],
            True,
            1,
            1,
        ),
    ],
)
def test_decode_timeline_omits_unusable_checkpoint_reconstruction(
    raw_moves: list[str],
    checkpoints: list[dict[str, int]],
    valid: bool,
    checkpoint_count: int,
    move_count: int,
) -> None:
    workstation = _workstation()
    workstation["initialization"]["scramble"] = "D' U2"
    value = emit.workstation_with_decode_timeline(
        workstation,
        info={
            "moves_bt": raw_moves,
            "reconstruction_checkpoints": checkpoints,
            "reconstruction_checkpoint_count": checkpoint_count,
            "reconstruction_checkpoint_move_count": move_count,
            "reconstruction_checkpoint_valid": valid,
        },
        moves=["U2", "D"],
    )

    assert "reconstruction" not in value


def test_unreached_endpoint_abstains_and_drops_moves(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env, solved_reached=False)
    _schema_validate(document)
    assert document["status"] == "abstained"
    assert document["moves"] == []
    assert document["endpoint"] == {"solved_reached": False}


def test_unknown_endpoint_abstains_with_a_null_verdict(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env, solved_reached=None)
    _schema_validate(document)
    assert document["status"] == "abstained"
    assert document["endpoint"] == {"solved_reached": None}


def test_implementation_sha256_pins_the_decoder_source(
    tmp_path: Path, decode_inputs, stamp_env
) -> None:
    decoder = _write(tmp_path / "trellis_gt.py", b"# pinned decoder\n")
    document = _build(decode_inputs, stamp_env, implementation_path=decoder)
    _schema_validate(document)
    assert (
        document["provenance"]["implementation_sha256"]
        == hashlib.sha256(b"# pinned decoder\n").hexdigest()
    )


def test_every_schema_move_token_is_accepted(decode_inputs, stamp_env) -> None:
    tokens = [f"{face}{suffix}" for face in "UDLRFB" for suffix in ("", "'", "2")]
    document = _build(decode_inputs, stamp_env, moves=tokens)
    _schema_validate(document)
    assert document["moves"] == tokens


@pytest.mark.parametrize("token", ["x", "M", "Rw", "R3", "r", "U''", ""])
def test_non_canonical_move_tokens_are_rejected(decode_inputs, stamp_env, token) -> None:
    with pytest.raises(emit.DecodeResultError, match="non-canonical move tokens"):
        _build(decode_inputs, stamp_env, moves=["R", token])


def test_naive_finished_at_is_rejected(decode_inputs, stamp_env) -> None:
    with pytest.raises(emit.DecodeResultError, match="timezone aware"):
        _build(decode_inputs, stamp_env, now=datetime(2026, 7, 25, 12, 0, 0))


# --- validation -------------------------------------------------------------


def test_validation_rejects_completed_without_a_reached_endpoint(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    document["endpoint"]["solved_reached"] = False
    with pytest.raises(emit.DecodeResultError, match="completed decode must have solved_reached"):
        emit.validate_decode_result(document)


def test_validation_rejects_moves_on_an_abstained_document(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env, solved_reached=False)
    document["moves"] = ["R"]
    with pytest.raises(emit.DecodeResultError, match="must carry no moves"):
        emit.validate_decode_result(document)


def test_validation_rejects_an_unknown_field(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    document["diagnostics"] = {}
    with pytest.raises(emit.DecodeResultError, match="unknown fields"):
        emit.validate_decode_result(document)


def test_validation_rejects_a_missing_field(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    del document["evaluation"]
    with pytest.raises(emit.DecodeResultError, match="missing required fields"):
        emit.validate_decode_result(document)


def test_validation_rejects_a_non_null_evaluation(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    document["evaluation"] = {"metric": "reach-ll-onset-with-correct-pre-ll-state"}
    with pytest.raises(emit.DecodeResultError, match="evaluation null"):
        emit.validate_decode_result(document)


def test_validation_rejects_workstation_sequence_drift(
    decode_inputs,
    stamp_env,
) -> None:
    workstation = _workstation()
    workstation["sequence"] = {
        "moves": [
            {"move": "R", "frame": 1},
            {"move": "U", "frame": 2},
            {"move": "F2", "frame": 3},
        ],
        "timing_basis": "canonical",
    }

    with pytest.raises(
        emit.DecodeResultError,
        match="sequence does not match top-level decode moves",
    ):
        _build(decode_inputs, stamp_env, workstation=workstation)


def test_validation_rejects_workstation_reconstruction_endpoint_drift(
    decode_inputs,
    stamp_env,
) -> None:
    workstation = _workstation()
    workstation["reconstruction"] = {
        "states": [],
        "solved_reached": False,
    }

    with pytest.raises(
        emit.DecodeResultError,
        match="reconstruction endpoint does not match",
    ):
        _build(decode_inputs, stamp_env, workstation=workstation)


def test_schema_allows_only_canonical_workstation_timing(
    decode_inputs,
    stamp_env,
) -> None:
    workstation = _workstation()
    workstation["sequence"] = {
        "moves": [
            {"move": "R", "frame": 1},
            {"move": "U'", "frame": 2},
            {"move": "F2", "frame": 3},
        ],
        "timing_basis": "canonical",
    }
    document = _build(decode_inputs, stamp_env, workstation=workstation)
    document["workstation"]["sequence"]["timing_basis"] = "backtracked"
    jsonschema = pytest.importorskip("jsonschema")

    with pytest.raises(jsonschema.ValidationError):
        _schema_validate(document)


def test_local_camera_schema_and_emitter_reject_failed_result_status(
    decode_inputs,
    stamp_env,
) -> None:
    document = _build(decode_inputs, stamp_env)
    document["status"] = "failed"
    document["moves"] = []
    document["endpoint"] = {"solved_reached": False}
    jsonschema = pytest.importorskip("jsonschema")

    with pytest.raises(jsonschema.ValidationError):
        _schema_validate(document)
    with pytest.raises(
        emit.DecodeResultError,
        match="unknown status",
    ):
        emit.validate_decode_result(document)


# --- writing and stamping ---------------------------------------------------


def test_write_requires_an_absolute_path(decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    with pytest.raises(emit.DecodeResultError, match="must be absolute"):
        emit.write_decode_result(Path("relative/result.json"), document)


def test_write_creates_parents_and_round_trips(tmp_path: Path, decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    target = tmp_path / "nested" / "run" / "result.json"
    written = emit.write_decode_result(target, document)
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == document
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob("nested/run/.*tmp"))


def test_write_refuses_an_invalid_document(tmp_path: Path, decode_inputs, stamp_env) -> None:
    document = _build(decode_inputs, stamp_env)
    document["profile"] = "made_up_profile"
    target = tmp_path / "result.json"
    with pytest.raises(emit.DecodeResultError, match="unknown profile"):
        emit.write_decode_result(target, document)
    assert not target.exists()


def test_inject_config_overwrites_the_in_process_block(tmp_path: Path, decode_inputs) -> None:
    document = _build(decode_inputs, {})
    target = tmp_path / "result.json"
    emit.write_decode_result(target, document)
    assert json.loads(target.read_text(encoding="utf-8"))["config"]["name"] == (
        emit.UNSTAMPED_CONFIG_NAME
    )

    stamped = emit.inject_config(
        target,
        name="move-gate-dgap-trustsoft-scrub-statefulom-robust",
        cfg_hash="4188304740",
        cfg_hash_input="--om-pf-reads ENV:MICROREST=1",
        extras_hash=None,
    )
    _schema_validate(stamped)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["config"] == {
        "name": "move-gate-dgap-trustsoft-scrub-statefulom-robust",
        "cfg_hash": "4188304740",
        "cfg_hash_algorithm": "posix-cksum",
        "cfg_hash_input": "--om-pf-reads ENV:MICROREST=1",
    }
    assert on_disk["recording_id"] == document["recording_id"]
    assert on_disk["moves"] == document["moves"]


def test_inject_config_rejects_a_non_cksum_stamp(tmp_path: Path, decode_inputs) -> None:
    document = _build(decode_inputs, {})
    target = tmp_path / "result.json"
    emit.write_decode_result(target, document)
    with pytest.raises(emit.DecodeResultError, match="posix cksum"):
        emit.inject_config(target, name="whatever", cfg_hash="not-a-cksum")


# --- the decoder entry point ------------------------------------------------


def test_emit_is_a_no_op_when_the_lane_is_not_armed(decode_inputs) -> None:
    assert (
        emit.emit_from_decode(
            tag="gt20",
            moves=["R"],
            solved_reached=True,
            inputs=decode_inputs,
            environ={},
        )
        is None
    )


def test_emit_writes_the_document_when_armed(tmp_path: Path, decode_inputs, stamp_env) -> None:
    target = tmp_path / "result.json"
    environ = dict(stamp_env, CUBED_RESULT_JSON=str(target))
    written = emit.emit_from_decode(
        tag="gt20",
        moves=["R", "U2"],
        solved_reached=True,
        inputs=decode_inputs,
        numpy_version="1.26.4",
        environ=environ,
    )
    assert written == target
    document = json.loads(target.read_text(encoding="utf-8"))
    _schema_validate(document)
    assert document["status"] == "completed"
    assert document["moves"] == ["R", "U2"]
    assert document["config"]["name"] == stamp_env["CUBED_CFG_NAME"]


def test_emit_adds_the_native_source_video_input(
    tmp_path: Path,
    decode_inputs,
    stamp_env,
) -> None:
    target = tmp_path / "result.json"
    video = _write(tmp_path / "source.mov", b"source-video")
    environ = dict(
        stamp_env,
        CUBED_RESULT_JSON=str(target),
        CUBED_RESULT_VIDEO_INPUT=str(video),
    )

    emit.emit_from_decode(
        tag="gt20",
        moves=["R"],
        solved_reached=True,
        inputs=decode_inputs,
        environ=environ,
    )

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["inputs"][0] == {
        "id": "video",
        "sha256": hashlib.sha256(b"source-video").hexdigest(),
    }
    _schema_validate(document)


def test_emit_rejects_an_unavailable_native_source_video_input(
    tmp_path: Path,
    decode_inputs,
    stamp_env,
) -> None:
    environ = dict(
        stamp_env,
        CUBED_RESULT_JSON=str(tmp_path / "result.json"),
        CUBED_RESULT_VIDEO_INPUT=str(tmp_path / "missing.mov"),
    )

    with pytest.raises(emit.DecodeResultError, match="video input is unavailable"):
        emit.emit_from_decode(
            tag="gt20",
            moves=["R"],
            solved_reached=True,
            inputs=decode_inputs,
            environ=environ,
        )


def test_emit_binds_the_real_decoder_inputs(tmp_path: Path, decode_inputs, stamp_env) -> None:
    target = tmp_path / "result.json"
    environ = dict(stamp_env, CUBED_RESULT_JSON=str(target))
    emit.emit_from_decode(
        tag="gt20",
        moves=["R"],
        solved_reached=True,
        inputs=decode_inputs,
        implementation_path=REPOSITORY_ROOT / "scripts" / "trellis_gt.py",
        environ=environ,
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    _schema_validate(document)
    expected = {
        identifier: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for identifier, path in decode_inputs
    }
    assert {receipt["id"]: receipt["sha256"] for receipt in document["inputs"]} == expected
    assert document["provenance"]["implementation_sha256"] == emit.sha256_file(
        REPOSITORY_ROOT / "scripts" / "trellis_gt.py"
    )
