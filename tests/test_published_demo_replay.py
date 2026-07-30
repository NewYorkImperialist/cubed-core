from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from cubed_core.cube import Cube
from cubed_core.decode_contract import posix_cksum

REPOSITORY = Path(__file__).resolve().parents[1]
DEMO_DIRECTORY = REPOSITORY / "apps/lab-web/public/demo/gtd1"
RESULT_PATH = DEMO_DIRECTORY / "decode-result.json"
RECEIPT_PATH = DEMO_DIRECTORY / "decode-receipt.json"
OVERLAY_PATH = DEMO_DIRECTORY / "overlay-track.json"
MANIFEST_PATH = REPOSITORY / "apps/lab-web/src/publishedDemo.ts"
SCRAMBLE = "R' L2 F L2 U R D F' U2 L B2 D2 F2 L2 B2 R2 U F2 B2 L2 D"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_demo_result_is_bound_to_its_receipt_and_current_contract() -> None:
    result = _json(RESULT_PATH)
    receipt = _json(RECEIPT_PATH)

    result_schema = _json(REPOSITORY / "schemas/decode-result-v1.schema.json")
    receipt_schema = _json(REPOSITORY / "schemas/decode-job-receipt-v1.schema.json")
    Draft202012Validator(result_schema).validate(result)
    Draft202012Validator(receipt_schema).validate(receipt)

    assert receipt["result_sha256"] == _sha256(RESULT_PATH)
    assert receipt["result_bytes"] == RESULT_PATH.stat().st_size
    assert receipt["capture_id"] == result["recording_id"]
    assert receipt["result_status"] == result["status"] == "completed"
    assert result["evaluation"] is None

    replay_check = receipt["replay_check"]
    assert isinstance(replay_check, dict)
    assert replay_check["performed"] is True
    assert replay_check["solved_reached"] is True
    assert replay_check["move_count"] == len(result["moves"]) == 77


def test_published_demo_replays_and_retains_its_recorded_decoder_identity() -> None:
    result = _json(RESULT_PATH)
    receipt = _json(RECEIPT_PATH)

    config = result["config"]
    assert isinstance(config, dict)
    assert config["cfg_hash"] == "1852738634"
    assert config["cfg_hash_algorithm"] == "posix-cksum"
    hash_input = config["cfg_hash_input"]
    assert isinstance(hash_input, str)
    assert str(posix_cksum(hash_input.encode("utf-8"))) == config["cfg_hash"]
    assert f'cfgHash: "{config["cfg_hash"]}"' in MANIFEST_PATH.read_text(encoding="utf-8")

    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    implementation_sha256 = provenance["implementation_sha256"]
    assert isinstance(implementation_sha256, str)
    assert len(implementation_sha256) == 64
    assert f'implementationSha256:\n    "{implementation_sha256}"' in (
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    moves = result["moves"]
    assert isinstance(moves, list)
    assert Cube.solved().apply_algorithm(SCRAMBLE).apply_algorithm(moves).is_solved()

    runner = receipt["runner_provenance"]
    assert isinstance(runner, dict)
    assert runner["identity_status"] == "unverified-external-identity"


def test_published_overlay_track_is_camera_only_and_matches_its_web_manifest() -> None:
    overlay = _json(OVERLAY_PATH)
    result = _json(RESULT_PATH)
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")

    assert overlay["schema"] == "cubed-core/demo-overlay-track-v1"
    assert overlay["schema_version"] == 1
    assert overlay["mode"] == "camera-only-native-vision-v1"
    assert overlay["capture_id"] == result["recording_id"]
    assert overlay["color_meaning"] == "sampled-color-not-a-classified-sticker"

    # The loader refuses the artifact unless these exact bytes arrive, so the
    # committed file and the checked-in manifest have to agree.
    assert f'sha256: "{_sha256(OVERLAY_PATH)}"' in manifest
    assert f"bytes: {OVERLAY_PATH.stat().st_size}," in manifest
    assert f'sourceSha256:\n    "{overlay["source_sha256"]}"' in manifest

    frames = overlay["frames"]
    assert isinstance(frames, dict)
    assert len(frames) == overlay["detected_frames"]

    faces = 0
    for key, frame in frames.items():
        assert 0 <= int(key) <= 5803
        entries = frame["f"]
        assert entries
        for face in entries:
            faces += 1
            assert len(face["q"]) == 4
            assert len(face["k"]) == 54
            assert len(face["v"]) == 9
            assert face["s"] in {"up", "front", "right", "down", "left", "back"}
    assert faces == overlay["detected_faces"]

    # A published overlay must never carry teacher-derived fields.
    assert not {"gt", "emitted", "anchors", "events", "spans"} & set(overlay)
