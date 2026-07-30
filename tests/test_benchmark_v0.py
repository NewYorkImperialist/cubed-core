from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import cubed_core.benchmark_v0 as benchmark_module
from cubed_core.benchmark_v0 import (
    BenchmarkV0Error,
    evaluate_bundle,
    main,
    rotate_state_to_orientation,
    state_sha256,
    validate_bundle,
)
from cubed_core.cube import FACE_ORDER, ORIENTATION_KEYS, Cube

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark_v0"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _artifact(
    path: Path,
    root: Path,
    media_type: str,
    license_id: str,
    attribution: str = "Cubed Core synthetic Benchmark v0 contract fixture",
) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "media_type": media_type,
        "public_license": license_id,
        "attribution": attribution,
    }


def _trajectory(scramble: list[str], moves: list[str]) -> list[list[int]]:
    cube = Cube.solved().apply_algorithm(scramble)
    states = [cube.to_array().astype(int).tolist()]
    for move in moves:
        cube.apply_move(move)
        states.append(cube.to_array().astype(int).tolist())
    return states


def _state_item(step: int, state: list[int]) -> dict[str, object]:
    return {"step": step, "state": state, "sha256": state_sha256(state)}


def _centroids(capture_id: str) -> dict[str, object]:
    return {
        "schema": "cubed-core/color-centroids-v1",
        "schema_version": 1,
        "color_space": "cielab",
        "centroids": {
            "white": [90, 0, 0],
            "green": [50, 80, 40],
            "red": [45, 180, 150],
            "blue": [35, 140, 70],
            "orange": [65, 160, 160],
            "yellow": [80, 100, 190],
        },
        "provenance": f"synthetic-contract-only {capture_id}",
    }


def _template(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _build_bundle(tmp_path: Path) -> dict[str, object]:
    public_root = tmp_path / "public"
    teacher_root = tmp_path / "held-teacher"
    public_root.mkdir()
    teacher_root.mkdir()
    captures: list[dict[str, object]] = []

    for index in range(20):
        capture_id = f"{index + 1:032x}"
        if index < 12:
            split = "train"
        elif index < 16:
            split = "validation"
        else:
            split = "test"
        groups = {
            "session_id": f"session-{split}",
            "solver_id": f"solver-{split}",
            "cube_id": f"cube-{split}",
            "camera_id": "camera-train-high-speed" if index == 0 else f"camera-{split}",
            "setup_id": f"setup-{split}",
        }

        video_path = public_root / "video" / f"{capture_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"synthetic-video-contract-{index}".encode())
        video = _artifact(video_path, public_root, "video/mp4", "CC-BY-4.0")
        high_speed_original = None
        video_derivation = None
        if index == 0:
            source_path = public_root / "video-source-240" / f"{capture_id}.mp4"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"synthetic-240-source-contract")
            high_speed_original = _artifact(
                source_path,
                public_root,
                "video/mp4",
                "CC-BY-4.0",
            )
            source_recording_id = f"{1001:032x}"
            derivation_document = {
                "schema": "cubed-core/frame-rate-derivation",
                "schema_version": 1,
                "kind": "every-other-frame-240-to-120",
                "created_at": "2026-01-01T00:00:00Z",
                "source": {
                    "recording_id": source_recording_id,
                    "path": high_speed_original["path"],
                    "bytes": high_speed_original["bytes"],
                    "sha256": high_speed_original["sha256"],
                    "probe_at_import": {},
                    "probe_before_derivation": {},
                },
                "transform": {
                    "source_frame_index_origin": 0,
                    "source_frame_index_step": 2,
                    "frame_selection": "zero-based source frames 0,2,4,...",
                    "filter": "select=not(mod(n\\,2)),setpts=N*1/(120*TB)",
                    "source_frame_rate": "240/1",
                    "target_frame_rate": "120/1",
                    "timestamp_policy": "retime-selected-frames-to-uniform-derived-cadence",
                    "audio_policy": "discard",
                    "subtitle_policy": "discard",
                    "data_stream_policy": "discard",
                    "video_encoder": "libx264-crf18-yuv420p-single-thread",
                    "expected_output_frame_count": 360,
                    "semantic_determinism": "frames 0,2,4 selected in order",
                    "compressed_byte_determinism": "synthetic contract only",
                },
                "ffmpeg": {
                    "executable": "ffmpeg",
                    "version_argv": ["ffmpeg", "-version"],
                    "version_output": "synthetic ffmpeg contract",
                    "version_output_sha256": hashlib.sha256(
                        b"synthetic ffmpeg contract"
                    ).hexdigest(),
                    "argv": ["ffmpeg", "-i", "synthetic"],
                    "working_directory": "workspace-root",
                    "target_frame_rate": "120/1",
                },
                "derivative": {
                    "recording_id": capture_id,
                    "path": video["path"],
                    "bytes": video["bytes"],
                    "sha256": video["sha256"],
                    "probe": {},
                    "probe_at_import": {},
                },
            }
            derivation_path = public_root / "derivation" / f"{capture_id}.json"
            _write_json(derivation_path, derivation_document)
            video_derivation = _artifact(
                derivation_path,
                public_root,
                "application/json",
                "CC0-1.0",
            )

        scramble_document = {
            "schema": "cubed-core/benchmark-v0-scramble",
            "schema_version": 1,
            "capture_id": capture_id,
            "moves": ["R", "U"],
        }
        scramble_path = public_root / "scramble" / f"{capture_id}.json"
        _write_json(scramble_path, scramble_document)
        scramble = _artifact(scramble_path, public_root, "application/json", "CC0-1.0")

        calibration_path = public_root / "calibration" / f"{capture_id}.json"
        _write_json(calibration_path, _centroids(capture_id))
        calibration = _artifact(
            calibration_path,
            public_root,
            "application/json",
            "CC-BY-4.0",
        )

        camera_document = {
            "schema": "cubed-core/benchmark-v0-camera-metadata",
            "schema_version": 1,
            "capture_id": capture_id,
            "session_group_id": groups["session_id"],
            "camera_group_id": groups["camera_id"],
            "setup_group_id": groups["setup_id"],
            "capture_device": {
                "manufacturer": "Example Camera Company",
                "model": "Synthetic Camera 1",
                "camera_position": "back",
                "lens_camera_class": "wide",
                "recording_mode": "research-240" if index == 0 else "standard-120",
                "configured_fps": 240 if index == 0 else 120,
                "metadata_sanitized": True,
            },
            "cube": {
                "puzzle_type": "3x3x3",
                "manufacturer": "Example Cube Company",
                "model": "Synthetic Smart Cube 1",
                "edge_length_mm": 56,
                "smart_cube": True,
                "teacher_relationship": "same-cube-ble-teacher",
            },
            "calibration_binding": {
                "calibration_sha256": calibration["sha256"],
                "setup_group_id": groups["setup_id"],
                "relationship": "captured-on-this-setup",
                "review_status": "reviewed",
                "reviewer_id": "reviewer-synthetic",
                "reviewed_at": "2026-01-01T00:00:00Z",
            },
            "measured_fps": {"numerator": 120, "denominator": 1},
            "source_measured_fps": ({"numerator": 240, "denominator": 1} if index == 0 else None),
            "frame_count": 360,
            "width": 1920,
            "height": 1080,
            "rotation_degrees": 0,
            "timing_source": "frame-timestamps",
            "video_probe_receipt": {
                "schema": "cubed-core/benchmark-v0-video-probe-receipt",
                "schema_version": 1,
                "producer": "cubed-core-server-probe",
                "review_status": "reviewed",
                "reviewer_id": "reviewer-synthetic",
                "reviewed_at": "2026-01-01T00:00:00Z",
                "tool": {
                    "name": "ffprobe",
                    "version": "synthetic-contract-v1",
                    "implementation_sha256": hashlib.sha256(
                        b"synthetic probe implementation"
                    ).hexdigest(),
                },
                "canonical_video": {
                    "path": video["path"],
                    "bytes": video["bytes"],
                    "sha256": video["sha256"],
                    "measured_fps": {"numerator": 120, "denominator": 1},
                    "frame_count": 360,
                    "width": 1920,
                    "height": 1080,
                    "rotation_degrees": 0,
                },
                "source_video": (
                    {
                        "path": high_speed_original["path"],
                        "bytes": high_speed_original["bytes"],
                        "sha256": high_speed_original["sha256"],
                        "measured_fps": {"numerator": 240, "denominator": 1},
                        "frame_count": 720,
                        "width": 1920,
                        "height": 1080,
                        "rotation_degrees": 0,
                    }
                    if high_speed_original is not None
                    else None
                ),
            },
        }
        camera_path = public_root / "camera" / f"{capture_id}.json"
        _write_json(camera_path, camera_document)
        camera = _artifact(camera_path, public_root, "application/json", "CC0-1.0")

        teacher_moves = ["U'", "R'"]
        states = _trajectory(scramble_document["moves"], teacher_moves)
        teacher_document = {
            "schema": "cubed-core/benchmark-v0-teacher",
            "schema_version": 1,
            "capture_id": capture_id,
            "scramble_sha256": scramble["sha256"],
            "teacher_source": {
                "kind": "ble-smart-cube",
                "review_status": "reviewed",
                "timing_status": "content-aligned",
                "cube_relation": "same-physical-cube",
                "cross_device_alignment": "content-affine-reviewed",
            },
            "teacher_moves": [
                {"move": move, "timestamp_ms": 100 * move_index}
                for move_index, move in enumerate(teacher_moves, start=1)
            ],
            "trajectory": [_state_item(step, state) for step, state in enumerate(states)],
            "phase_boundaries": [
                {
                    "name": "start",
                    "teacher_move_index": 0,
                    "canonical_state_sha256": state_sha256(states[0]),
                },
                {
                    "name": "synthetic-ll",
                    "teacher_move_index": 1,
                    "canonical_state_sha256": state_sha256(states[1]),
                },
                {
                    "name": "solved",
                    "teacher_move_index": 2,
                    "canonical_state_sha256": state_sha256(states[2]),
                },
            ],
            "ll_boundary": {
                "metric": "reach-ll-onset-with-correct-pre-ll-state",
                "teacher_move_index": 1,
                "phase_name": "synthetic-ll",
                "equivalence_basis": "cubed-core-exact-whole-cube-rotation-v1",
                "state_hash_algorithm": "sha256-cubed-core-state-v1",
                "accepted_target_states": [
                    {
                        "orientation_key": list(key),
                        "state": rotate_state_to_orientation(states[1], key),
                        "sha256": state_sha256(rotate_state_to_orientation(states[1], key)),
                    }
                    for key in ORIENTATION_KEYS
                ],
            },
        }
        teacher_path = teacher_root / "truth" / f"{capture_id}.json"
        _write_json(teacher_path, teacher_document)
        teacher = _artifact(
            teacher_path,
            teacher_root,
            "application/json",
            "CC-BY-4.0",
        )

        artifacts = {
            "video": video,
            "scramble": scramble,
            "calibration": calibration,
            "camera_metadata": camera,
            "teacher_truth": teacher,
        }
        if high_speed_original is not None:
            artifacts["high_speed_original"] = high_speed_original
        if video_derivation is not None:
            artifacts["video_derivation"] = video_derivation
        rights_document = {
            "schema": "cubed-core/benchmark-v0-rights",
            "schema_version": 1,
            "record_id": f"rights-{capture_id}",
            "capture_id": capture_id,
            "source_status": "approved-for-public-redistribution",
            "public_release": True,
            "terms_version": "synthetic-test-v1",
            "terms_sha256": hashlib.sha256(
                b"synthetic final Benchmark v0 public-release terms"
            ).hexdigest(),
            "acceptance_receipt_sha256": hashlib.sha256(
                f"synthetic acceptance {capture_id}".encode()
            ).hexdigest(),
            "approved_at": "2026-01-01T00:00:00Z",
            "reviewer_id": "reviewer-synthetic",
            "authorization": {
                "rights_holder_authorized": True,
                "recording_subjects_authorized": True,
                "public_redistribution_authorized": True,
                "ml_research_use_authorized": True,
                "adaptations_under_public_license_acknowledged": True,
                "withdrawal_policy_acknowledged": True,
                "privacy_review_approved": True,
                "audio_removed": True,
            },
            "artifacts": [
                {
                    "role": role,
                    "path": reference["path"],
                    "sha256": reference["sha256"],
                    "public_release": True,
                    "public_license": reference["public_license"],
                    "attribution": reference["attribution"],
                    "rights_basis": "maintainer-owned",
                }
                for role, reference in artifacts.items()
            ],
        }
        rights_path = public_root / "rights" / f"{capture_id}.json"
        _write_json(rights_path, rights_document)
        rights = _artifact(rights_path, public_root, "application/json", "CC0-1.0")

        captures.append(
            {
                "capture_id": capture_id,
                "split": split,
                "source_status": "approved-for-public-redistribution",
                "groups": groups,
                "inputs": {
                    "video": video,
                    "scramble": scramble,
                    "calibration": calibration,
                    "camera_metadata": camera,
                    **(
                        {"high_speed_original": high_speed_original}
                        if high_speed_original is not None
                        else {}
                    ),
                    **(
                        {"video_derivation": video_derivation}
                        if video_derivation is not None
                        else {}
                    ),
                },
                "teacher_truth": teacher,
                "rights_record": rights,
            }
        )

    manifest = {
        "schema": "cubed-core/benchmark-v0-manifest",
        "schema_version": 1,
        "benchmark_id": "cubed-core-synthetic-contract",
        "benchmark_version": "0",
        "release_scope": "public-benchmark",
        "evidence_scope": "development-benchmark-not-generalization-evidence",
        "collection_license": {
            "license_id": "CC-BY-4.0",
            "attribution": "Cubed Core synthetic Benchmark v0 contract fixture",
            "scope": "selection-and-arrangement-only",
            "artifact_licenses_authoritative": True,
            "exceptions": [],
        },
        "capture_count": len(captures),
        "split_policy": {
            "unit": "capture",
            "fixed": True,
            "splits": ["train", "validation", "test"],
            "group_policy": {
                key: {"mode": "disjoint"}
                for key in (
                    "session_id",
                    "solver_id",
                    "cube_id",
                    "camera_id",
                    "setup_id",
                )
            },
        },
        "limitations": [
            "Synthetic files exercise contracts only and are not measurement evidence."
        ],
        "captures": captures,
    }
    manifest_path = public_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "public_root": public_root,
        "teacher_root": teacher_root,
    }


def _write_predictions(bundle: dict[str, object], split: str = "test") -> Path:
    manifest = bundle["manifest"]
    prediction_dir = Path(bundle["public_root"]).parent / f"predictions-{split}"
    prediction_dir.mkdir()
    template = _template("prediction-completed.json")
    for capture in manifest["captures"]:
        if capture["split"] != split:
            continue
        prediction = copy.deepcopy(template)
        prediction["benchmark_id"] = manifest["benchmark_id"]
        prediction["capture_id"] = capture["capture_id"]
        prediction["inference"]["input_artifacts"] = [
            {"role": role, "sha256": capture["inputs"][role]["sha256"]}
            for role in ("video", "scramble", "calibration", "camera_metadata")
        ]
        prediction["result"]["moves"] = ["U'", "R'"]
        _write_json(prediction_dir / f"{capture['capture_id']}.json", prediction)
    return prediction_dir


def _rebind_json_artifact(
    bundle: dict[str, object],
    *,
    capture_index: int,
    role: str,
    path: Path,
    root: Path,
) -> None:
    manifest = bundle["manifest"]
    capture = manifest["captures"][capture_index]
    old_reference = capture["teacher_truth"] if role == "teacher_truth" else capture["inputs"][role]
    new_reference = _artifact(
        path,
        root,
        old_reference["media_type"],
        old_reference["public_license"],
    )
    if role == "teacher_truth":
        capture["teacher_truth"] = new_reference
    else:
        capture["inputs"][role] = new_reference

    rights_path = bundle["public_root"] / capture["rights_record"]["path"]
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    entry = next(item for item in rights["artifacts"] if item["role"] == role)
    entry["path"] = new_reference["path"]
    entry["sha256"] = new_reference["sha256"]
    entry["public_license"] = new_reference["public_license"]
    entry["attribution"] = new_reference["attribution"]
    _write_json(rights_path, rights)
    capture["rights_record"] = _artifact(
        rights_path,
        bundle["public_root"],
        "application/json",
        capture["rights_record"]["public_license"],
    )
    _write_json(bundle["manifest_path"], manifest)


def _rewrite_capture_rights(
    bundle: dict[str, object],
    *,
    capture_index: int,
    update: Callable[[dict[str, object]], None],
) -> None:
    manifest = bundle["manifest"]
    capture = manifest["captures"][capture_index]
    rights_path = bundle["public_root"] / capture["rights_record"]["path"]
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    update(rights)
    _write_json(rights_path, rights)
    capture["rights_record"] = _artifact(
        rights_path,
        bundle["public_root"],
        "application/json",
        capture["rights_record"]["public_license"],
    )
    _write_json(bundle["manifest_path"], manifest)


@pytest.fixture
def synthetic_bundle(tmp_path: Path) -> dict[str, object]:
    return _build_bundle(tmp_path)


def test_public_bundle_validates_all_artifacts_rights_truth_and_splits(
    synthetic_bundle: dict[str, object],
) -> None:
    result = validate_bundle(
        synthetic_bundle["manifest_path"],
        bundle_root=synthetic_bundle["public_root"],
        teacher_root=synthetic_bundle["teacher_root"],
    )

    assert result["capture_count"] == 20
    assert result["split_counts"] == {"train": 12, "validation": 4, "test": 4}
    assert result["public_rights_checked"] is True
    assert result["teacher_truth_checked"] is True
    assert all(not overlap for overlap in result["group_overlap"].values())
    accepted = result["captures"][f"{1:032x}"]["teacher"]["ll_boundary"]["accepted_target_states"]
    assert len({item["sha256"] for item in accepted}) == 24


def test_whole_cube_orientation_order_is_stable() -> None:
    assert ORIENTATION_KEYS == (
        ("up", "front", "right"),
        ("up", "right", "back"),
        ("up", "back", "left"),
        ("up", "left", "front"),
        ("front", "down", "right"),
        ("front", "right", "up"),
        ("front", "up", "left"),
        ("front", "left", "down"),
        ("down", "back", "right"),
        ("down", "right", "front"),
        ("down", "front", "left"),
        ("down", "left", "back"),
        ("back", "up", "right"),
        ("back", "right", "down"),
        ("back", "down", "left"),
        ("back", "left", "up"),
        ("left", "front", "up"),
        ("left", "up", "back"),
        ("left", "back", "down"),
        ("left", "down", "front"),
        ("right", "front", "down"),
        ("right", "down", "back"),
        ("right", "back", "up"),
        ("right", "up", "front"),
    )


def test_whole_cube_rotation_contract_is_identity_and_24_bijections() -> None:
    asymmetric = Cube.solved().apply_algorithm(["R", "U", "F", "L2"])
    state = asymmetric.to_array().astype(int).tolist()
    rotated_states = {
        tuple(rotate_state_to_orientation(state, orientation)) for orientation in ORIENTATION_KEYS
    }

    assert rotate_state_to_orientation(state, ("up", "front", "right")) == state
    assert len(rotated_states) == 24
    for model_up, model_front, model_right in ORIENTATION_KEYS:
        rotated = rotate_state_to_orientation(state, (model_up, model_front, model_right))
        assert sorted(rotated) == sorted(state)
        assert rotated[4] == state[FACE_ORDER.index(model_up) * 9 + 4]
        assert rotated[22] == state[FACE_ORDER.index(model_front) * 9 + 4]
        assert rotated[13] == state[FACE_ORDER.index(model_right) * 9 + 4]


def test_evaluator_scores_raw_trajectory_primary_and_endpoint_secondary(
    synthetic_bundle: dict[str, object],
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    report = evaluate_bundle(
        synthetic_bundle["manifest_path"],
        predictions,
        bundle_root=synthetic_bundle["public_root"],
        teacher_root=synthetic_bundle["teacher_root"],
    )

    assert report["aggregate"]["primary"] == {
        "metric": "reach_ll_rate",
        "definition": "reach-ll-onset-with-correct-pre-ll-state",
        "successes": 4,
        "total": 4,
        "rate": 1.0,
    }
    assert report["aggregate"]["secondary"]["rate"] == 1.0
    assert report["aggregate"]["diagnostics"]["edit_distance_mean"] == 0
    assert report["contract_valid"] is True
    assert report["claim_tier"] == "interface-boundary-only"
    assert "publishable" not in report
    assert report["system"]["config_sha256"] == "e" * 64
    assert report["system"]["model_artifacts"] == [
        {"name": "synthetic-tracker", "sha256": "d" * 64}
    ]
    assert report["teacher_isolation"]["process_environment_audited"] is False
    assert "interface boundary only" in report["teacher_isolation"]["scope"]
    assert any(
        "prediction host is not audited" in limitation for limitation in report["limitations"]
    )
    assert all(item["reach_ll_hit_step"] == 1 for item in report["captures"])


def test_abstention_counts_as_primary_and_secondary_failure_but_is_not_an_error(
    synthetic_bundle: dict[str, object],
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    capture_id = synthetic_bundle["manifest"]["captures"][-1]["capture_id"]
    completed_path = predictions / f"{capture_id}.json"
    abstained = _template("prediction-abstained.json")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    abstained["benchmark_id"] = completed["benchmark_id"]
    abstained["capture_id"] = completed["capture_id"]
    abstained["inference"]["input_artifacts"] = completed["inference"]["input_artifacts"]
    _write_json(completed_path, abstained)

    report = evaluate_bundle(
        synthetic_bundle["manifest_path"],
        predictions,
        bundle_root=synthetic_bundle["public_root"],
        teacher_root=synthetic_bundle["teacher_root"],
    )

    assert report["aggregate"]["primary"]["rate"] == 0.75
    assert report["aggregate"]["secondary"]["rate"] == 0.75
    assert report["aggregate"]["diagnostics"]["abstention_count"] == 1
    abstained_result = next(item for item in report["captures"] if item["capture_id"] == capture_id)
    assert abstained_result["abstention_reason"] == "synthetic-contract-abstention"
    assert abstained_result["reach_ll"] is False


def test_group_leakage_cannot_cross_a_dimension_declared_disjoint(
    synthetic_bundle: dict[str, object],
) -> None:
    manifest = synthetic_bundle["manifest"]
    train_solver = manifest["captures"][0]["groups"]["solver_id"]
    manifest["captures"][-1]["groups"]["solver_id"] = train_solver
    _write_json(synthetic_bundle["manifest_path"], manifest)

    with pytest.raises(BenchmarkV0Error, match="split leakage: solver_id"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_collection_license_exceptions_bind_an_exact_artifact_license(
    synthetic_bundle: dict[str, object],
) -> None:
    manifest = synthetic_bundle["manifest"]
    capture = manifest["captures"][0]
    manifest["collection_license"]["exceptions"] = [
        {
            "capture_id": capture["capture_id"],
            "role": "video",
            "public_license": "CC0-1.0",
            "reason": "Synthetic mismatch must fail the collection-license contract.",
        }
    ]
    _write_json(synthetic_bundle["manifest_path"], manifest)

    with pytest.raises(BenchmarkV0Error, match="does not match the artifact public_license"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_disclosed_group_overlap_is_visible_in_validation(
    synthetic_bundle: dict[str, object],
) -> None:
    manifest = synthetic_bundle["manifest"]
    shared_solver = "solver-owner"
    for capture in manifest["captures"]:
        capture["groups"]["solver_id"] = shared_solver
    manifest["split_policy"]["group_policy"]["solver_id"] = {
        "mode": "disclosed-overlap",
        "reason": "All Benchmark v0 captures use one disclosed owner-solver.",
    }
    _write_json(synthetic_bundle["manifest_path"], manifest)

    result = validate_bundle(
        synthetic_bundle["manifest_path"],
        bundle_root=synthetic_bundle["public_root"],
        teacher_root=synthetic_bundle["teacher_root"],
    )

    assert result["group_overlap"]["solver_id"] == [
        {
            "value": shared_solver,
            "splits": ["train", "validation", "test"],
            "capture_count": 20,
        }
    ]


def test_private_research_capture_fails_closed_in_public_manifest(
    synthetic_bundle: dict[str, object],
) -> None:
    manifest = synthetic_bundle["manifest"]
    manifest["captures"][0]["source_status"] = "private-research-only"
    _write_json(synthetic_bundle["manifest_path"], manifest)

    with pytest.raises(BenchmarkV0Error, match="source_status"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_artifact_tampering_fails_before_evaluation(
    synthetic_bundle: dict[str, object],
) -> None:
    video_path = synthetic_bundle["public_root"] / "video" / f"{1:032x}.mp4"
    video_path.write_bytes(video_path.read_bytes() + b"tampered")

    with pytest.raises(BenchmarkV0Error, match="byte count mismatch"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_public_gate_rejects_empty_scramble(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["scramble"]["path"]
    scramble = json.loads(path.read_text(encoding="utf-8"))
    scramble["moves"] = []
    _write_json(path, scramble)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="scramble",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="moves.*should be non-empty"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_public_gate_rejects_unsanitized_unique_device_fields(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["capture_device"]["serial_number"] = "must-not-be-public"
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="serial_number.*unexpected"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("capture_device", "model"), "Manas's iPhone"),
        (("capture_device", "manufacturer"), "owner@example.test"),
        (("cube", "model"), "/Users/owner/cube-profile"),
        (("capture_device", "model"), "Camera ABCDEF0123456789"),
        (("capture_device", "model"), "490154203237518"),
        (("capture_device", "model"), "Bluetooth device name: cube"),
    ],
)
def test_public_gate_rejects_personal_or_unique_values_in_public_text(
    synthetic_bundle: dict[str, object],
    field_path: tuple[str, str],
    value: str,
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field_path[0]][field_path[1]] = value
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="personal or unique device identifier"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_lens_camera_class_is_a_frozen_non_unique_enum(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["capture_device"]["lens_camera_class"] = "lens-serial-ABC123"
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="lens_camera_class"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_public_gate_rejects_unknown_device_or_cube_values(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["cube"]["model"] = "unknown"
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="cube.*model"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_public_gate_rejects_non_ble_teacher_truth(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["teacher_root"] / capture["teacher_truth"]["path"]
    teacher = json.loads(path.read_text(encoding="utf-8"))
    teacher["teacher_source"]["kind"] = "reviewed-manual"
    _write_json(path, teacher)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="teacher_truth",
        path=path,
        root=synthetic_bundle["teacher_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="teacher_source.*kind"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_public_gate_rejects_baseline_video_declared_over_100_mib(
    synthetic_bundle: dict[str, object],
) -> None:
    manifest = synthetic_bundle["manifest"]
    manifest["captures"][1]["inputs"]["video"]["bytes"] = 100 * 1024 * 1024 + 1
    _write_json(synthetic_bundle["manifest_path"], manifest)

    with pytest.raises(BenchmarkV0Error, match="104857600"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_high_speed_tier_is_granted_only_by_220_to_242_fps_cadence(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][0]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["source_measured_fps"] = {"numerator": 219, "denominator": 1}
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=0,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="220..242"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_high_speed_derivation_receipt_must_bind_original_and_baseline(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][0]
    path = synthetic_bundle["public_root"] / capture["inputs"]["video_derivation"]["path"]
    derivation = json.loads(path.read_text(encoding="utf-8"))
    derivation["derivative"]["sha256"] = "0" * 64
    _write_json(path, derivation)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=0,
        role="video_derivation",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="derivation derivative.sha256"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_reviewed_probe_receipt_must_bind_canonical_video_identity(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["video_probe_receipt"]["canonical_video"]["sha256"] = "0" * 64
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="reviewed probe canonical_video.sha256"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_high_speed_derivation_cannot_self_attest_a_different_cadence(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][0]
    path = synthetic_bundle["public_root"] / capture["inputs"]["video_derivation"]["path"]
    derivation = json.loads(path.read_text(encoding="utf-8"))
    derivation["transform"]["source_frame_rate"] = "230/1"
    derivation["transform"]["target_frame_rate"] = "115/1"
    derivation["transform"]["filter"] = "select=not(mod(n\\,2)),setpts=N*1/(115*TB)"
    derivation["ffmpeg"]["target_frame_rate"] = "115/1"
    _write_json(path, derivation)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=0,
        role="video_derivation",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="source cadence must equal the reviewed"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_high_speed_derivation_requires_ceil_half_frame_count(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][0]
    path = synthetic_bundle["public_root"] / capture["inputs"]["video_derivation"]["path"]
    derivation = json.loads(path.read_text(encoding="utf-8"))
    derivation["transform"]["expected_output_frame_count"] = 359
    _write_json(path, derivation)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=0,
        role="video_derivation",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match=r"ceil\(source/2\)"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_calibration_binding_must_match_artifact_and_setup(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    path = synthetic_bundle["public_root"] / capture["inputs"]["camera_metadata"]["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["calibration_binding"]["calibration_sha256"] = "a" * 64
    _write_json(path, metadata)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=1,
        role="camera_metadata",
        path=path,
        root=synthetic_bundle["public_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="calibration binding calibration_sha256"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_rights_attribution_must_match_manifest_artifact(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    rights_path = synthetic_bundle["public_root"] / capture["rights_record"]["path"]
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    video_entry = next(item for item in rights["artifacts"] if item["role"] == "video")
    video_entry["attribution"] = "Different attribution"
    _write_json(rights_path, rights)
    capture["rights_record"] = _artifact(
        rights_path,
        synthetic_bundle["public_root"],
        "application/json",
        capture["rights_record"]["public_license"],
    )
    _write_json(synthetic_bundle["manifest_path"], synthetic_bundle["manifest"])

    with pytest.raises(BenchmarkV0Error, match="rights video.attribution does not match"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_rights_require_non_placeholder_acceptance_receipt_identity(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][1]
    rights_path = synthetic_bundle["public_root"] / capture["rights_record"]["path"]
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["acceptance_receipt_sha256"] = "0" * 64
    _write_json(rights_path, rights)
    capture["rights_record"] = _artifact(
        rights_path,
        synthetic_bundle["public_root"],
        "application/json",
        capture["rights_record"]["public_license"],
    )
    _write_json(synthetic_bundle["manifest_path"], synthetic_bundle["manifest"])

    with pytest.raises(BenchmarkV0Error, match="acceptance_receipt_sha256"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("record_id", "rights-final-tbd"),
        ("terms_version", "final-tbd"),
        ("reviewer_id", "reviewer-tbd"),
        ("terms_sha256", "1" * 64),
        ("acceptance_receipt_sha256", "ab" * 32),
    ),
)
def test_rights_reject_placeholder_release_decisions(
    synthetic_bundle: dict[str, object],
    field: str,
    value: str,
) -> None:
    _rewrite_capture_rights(
        synthetic_bundle,
        capture_index=1,
        update=lambda rights: rights.__setitem__(field, value),
    )

    with pytest.raises(BenchmarkV0Error, match="placeholder"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_rights_reject_draft_terms_version(
    synthetic_bundle: dict[str, object],
) -> None:
    _rewrite_capture_rights(
        synthetic_bundle,
        capture_index=1,
        update=lambda rights: rights.__setitem__("terms_version", "final-draft"),
    )

    with pytest.raises(BenchmarkV0Error, match="terms_version"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


@pytest.mark.parametrize(
    "field",
    ("privacy_review_approved", "audio_removed"),
)
def test_rights_require_privacy_review_and_audio_removal(
    synthetic_bundle: dict[str, object],
    field: str,
) -> None:
    def revoke_gate(rights: dict[str, object]) -> None:
        rights["authorization"][field] = False

    _rewrite_capture_rights(
        synthetic_bundle,
        capture_index=1,
        update=revoke_gate,
    )

    with pytest.raises(BenchmarkV0Error, match=field):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_teacher_root_must_be_disjoint_from_public_inputs(
    synthetic_bundle: dict[str, object],
) -> None:
    with pytest.raises(BenchmarkV0Error, match="must be disjoint"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["public_root"],
        )
    with pytest.raises(BenchmarkV0Error, match="must be disjoint"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["public_root"] / "nested-teacher",
        )
    with pytest.raises(BenchmarkV0Error, match="must be disjoint"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["teacher_root"] / "nested-public",
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_evaluator_closes_all_predictions_before_opening_teacher_truth(
    synthetic_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    reads: list[tuple[str, Path]] = []
    original_read_json = benchmark_module._read_json

    def recording_read_json(path: Path, *, label: str) -> dict[str, object]:
        resolved = path.resolve()
        if resolved.parent == predictions.resolve():
            reads.append(("prediction", resolved))
        try:
            resolved.relative_to(Path(synthetic_bundle["teacher_root"]).resolve())
        except ValueError:
            pass
        else:
            reads.append(("teacher", resolved))
        return original_read_json(path, label=label)

    monkeypatch.setattr(benchmark_module, "_read_json", recording_read_json)
    evaluate_bundle(
        synthetic_bundle["manifest_path"],
        predictions,
        bundle_root=synthetic_bundle["public_root"],
        teacher_root=synthetic_bundle["teacher_root"],
    )

    prediction_positions = [index for index, (kind, _) in enumerate(reads) if kind == "prediction"]
    teacher_positions = [index for index, (kind, _) in enumerate(reads) if kind == "teacher"]
    assert len(prediction_positions) == 4
    assert len(teacher_positions) == 20
    assert max(prediction_positions) < min(teacher_positions)


def test_evaluator_rejects_manifest_mutation_between_prediction_and_teacher_phases(
    synthetic_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    original_validate_bundle = benchmark_module.validate_bundle

    def mutate_then_validate(*args: object, **kwargs: object) -> dict[str, object]:
        manifest = synthetic_bundle["manifest"]
        manifest["limitations"].append("Mutation injected between evaluation phases.")
        _write_json(synthetic_bundle["manifest_path"], manifest)
        return original_validate_bundle(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "validate_bundle", mutate_then_validate)

    with pytest.raises(BenchmarkV0Error, match="manifest changed after predictions were bound"):
        evaluate_bundle(
            synthetic_bundle["manifest_path"],
            predictions,
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_prediction_that_declares_teacher_visibility_is_rejected(
    synthetic_bundle: dict[str, object],
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    path = sorted(predictions.glob("*.json"))[0]
    prediction = json.loads(path.read_text(encoding="utf-8"))
    prediction["inference"]["teacher_artifacts_visible"] = True
    _write_json(path, prediction)

    with pytest.raises(BenchmarkV0Error, match="teacher_artifacts_visible"):
        evaluate_bundle(
            synthetic_bundle["manifest_path"],
            predictions,
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_prediction_system_identity_includes_config_models_and_run_receipt(
    synthetic_bundle: dict[str, object],
) -> None:
    predictions = _write_predictions(synthetic_bundle)
    path = sorted(predictions.glob("*.json"))[0]
    prediction = json.loads(path.read_text(encoding="utf-8"))
    prediction["system"]["config_sha256"] = "0" * 64
    _write_json(path, prediction)

    with pytest.raises(BenchmarkV0Error, match="config_sha256"):
        evaluate_bundle(
            synthetic_bundle["manifest_path"],
            predictions,
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_ll_target_identity_drift_fails_closed(
    synthetic_bundle: dict[str, object],
) -> None:
    capture = synthetic_bundle["manifest"]["captures"][0]
    truth_path = synthetic_bundle["teacher_root"] / capture["teacher_truth"]["path"]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    target = truth["ll_boundary"]["accepted_target_states"][1]
    target["state"] = truth["ll_boundary"]["accepted_target_states"][0]["state"]
    target["sha256"] = state_sha256(target["state"])
    _write_json(truth_path, truth)
    _rebind_json_artifact(
        synthetic_bundle,
        capture_index=0,
        role="teacher_truth",
        path=truth_path,
        root=synthetic_bundle["teacher_root"],
    )

    with pytest.raises(BenchmarkV0Error, match="not the exact physical whole-cube rotation"):
        validate_bundle(
            synthetic_bundle["manifest_path"],
            bundle_root=synthetic_bundle["public_root"],
            teacher_root=synthetic_bundle["teacher_root"],
        )


def test_module_cli_validate_and_evaluate_emit_machine_readable_json(
    synthetic_bundle: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    validate_exit = main(
        [
            "validate",
            str(synthetic_bundle["manifest_path"]),
            "--bundle-root",
            str(synthetic_bundle["public_root"]),
            "--teacher-root",
            str(synthetic_bundle["teacher_root"]),
        ]
    )
    validation = json.loads(capsys.readouterr().out)
    assert validate_exit == 0
    assert validation["capture_count"] == 20

    predictions = _write_predictions(synthetic_bundle)
    evaluate_exit = main(
        [
            "evaluate",
            str(synthetic_bundle["manifest_path"]),
            str(predictions),
            "--bundle-root",
            str(synthetic_bundle["public_root"]),
            "--teacher-root",
            str(synthetic_bundle["teacher_root"]),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert evaluate_exit == 0
    assert report["aggregate"]["primary"]["rate"] == 1.0
