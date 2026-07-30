from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from jsonschema import Draft202012Validator


def _load_public_dataset_module() -> ModuleType:
    scripts_directory = Path(__file__).resolve().parents[1]
    script_path = scripts_directory / "public_dataset.py"
    module_name = "cubed_core_public_dataset_test_target"
    sys.path.insert(0, str(scripts_directory))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load public dataset target: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_directory))


public_dataset = _load_public_dataset_module()


def _load_benchmark_fixture_module() -> ModuleType:
    repository = Path(__file__).resolve().parents[2]
    test_path = repository / "tests" / "test_benchmark_v0.py"
    module_name = "cubed_core_benchmark_v0_public_dataset_fixture"
    spec = importlib.util.spec_from_file_location(module_name, test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Benchmark v0 fixture helpers: {test_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark_fixture_module = _load_benchmark_fixture_module()

CAPTURE_ID = "a" * 32
SECOND_CAPTURE_ID = "b" * 32
ATTRIBUTION = "Synthetic fixture by the Cubed Core test suite."
PUBLIC_LICENSE = "CC-BY-4.0"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "source_path": str(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _rewrite_source(reference: dict[str, Any], payload: bytes) -> None:
    path = Path(reference["source_path"])
    path.write_bytes(payload)
    reference["bytes"] = len(payload)
    reference["sha256"] = _sha256(payload)


def _rewrite_rights(
    document: dict[str, Any],
    mutate: Any,
) -> None:
    rights_artifact = next(
        artifact
        for artifact in document["captures"][0]["artifacts"]
        if artifact["role"] == "rights_record"
    )
    rights = json.loads(Path(rights_artifact["source_path"]).read_text())
    mutate(rights)
    _rewrite_source(
        rights_artifact,
        (json.dumps(rights, indent=2, sort_keys=True) + "\n").encode(),
    )


def _artifact(
    *,
    artifact_id: str,
    role: str,
    public_path: str,
    source: dict[str, Any],
    media_type: str,
    schema_file: str | None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "role": role,
        **source,
        "public_path": public_path,
        "media_type": media_type,
        "public_license": PUBLIC_LICENSE,
        "attribution": ATTRIBUTION,
        "privacy_reviewed": True,
        "rights_reviewed": True,
        "schema_file": schema_file,
    }


def _rights_payload(
    capture_id: str,
    video_artifact: dict[str, Any],
) -> bytes:
    document = {
        "schema": "cubed-core/public-corpus-rights",
        "schema_version": 1,
        "record_id": f"public-{capture_id}",
        "capture_id": capture_id,
        "source_status": "approved-for-public-redistribution",
        "public_release": True,
        "terms_version": "data-contribution-terms-v1",
        "terms_sha256": _sha256(b"synthetic reviewed public-release terms"),
        "acceptance_receipt_sha256": _sha256(
            f"synthetic retained acceptance receipt {capture_id}".encode()
        ),
        "approved_at": "2026-07-27T12:00:00Z",
        "reviewer_id": "release-reviewer",
        "authorization": {
            "rights_holder_authorized": True,
            "recorded_subjects_authorized": True,
            "public_redistribution_authorized": True,
            "ml_research_authorized": True,
            "privacy_review_approved": True,
            "audio_removed": True,
        },
        "artifacts": [
            {
                "artifact_id": video_artifact["artifact_id"],
                "path": video_artifact["public_path"],
                "bytes": video_artifact["bytes"],
                "sha256": video_artifact["sha256"],
                "public_license": video_artifact["public_license"],
                "attribution": video_artifact["attribution"],
            }
        ],
    }
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _capture(
    source_root: Path,
    capture_id: str,
    *,
    shared_video: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if shared_video is None:
        video_source = _write(
            source_root / capture_id / "video.mp4",
            f"synthetic video bytes for {capture_id}\n".encode(),
        )
    else:
        video_source = dict(shared_video)
    video = _artifact(
        artifact_id=f"video-{capture_id[:8]}",
        role="video",
        public_path=f"captures/{capture_id}/video.mp4",
        source=video_source,
        media_type="video/mp4",
        schema_file=None,
    )
    rights_source = _write(
        source_root / capture_id / "rights.json",
        _rights_payload(capture_id, video),
    )
    rights = _artifact(
        artifact_id=f"rights-{capture_id[:8]}",
        role="rights_record",
        public_path=f"captures/{capture_id}/rights.json",
        source=rights_source,
        media_type="application/json",
        schema_file="public-corpus-rights-v1.schema.json",
    )
    return {
        "capture_id": capture_id,
        "source_status": "approved-for-public-redistribution",
        "provenance_status": "reviewed-source-and-authorship",
        "split": "unassigned",
        "artifacts": [video, rights],
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    source_root = tmp_path / "private-sources"
    card = _write(
        source_root / "README.md",
        (
            b"---\n"
            b"license: cc-by-4.0\n"
            b"pretty_name: Synthetic Cubed Core Corpus\n"
            b"tags:\n"
            b"- video\n"
            b"- computer-vision\n"
            b"---\n"
            b"# Synthetic public corpus\n\n"
            b"## Intended use\n\nDeterministic public-dataset contract testing only.\n\n"
            b"## Limitations\n\nSynthetic bytes are not research data.\n\n"
            b"## Licenses and attribution\n\n"
            b"See the reviewed synthetic license and attribution file.\n\n"
            b"## Privacy\n\nNo people or private data are present.\n"
        ),
    )
    license_file = _write(
        source_root / "DATASET.md",
        (
            b"# Dataset license\n\n"
            b"The selection and arrangement are licensed under CC-BY-4.0.\n\n"
            b"Per-artifact licenses in the manifest are authoritative and may differ.\n\n"
            b"Required attribution: Synthetic fixture by the Cubed Core test suite.\n"
        ),
    )
    document = {
        "schema": "cubed-core/public-dataset-build-plan",
        "schema_version": 1,
        "dataset_id": "cubed-core-synthetic",
        "release_id": "synthetic-r1",
        "description": "A synthetic public corpus used only by repository tests.",
        "collection_license": {
            "license_id": PUBLIC_LICENSE,
            "attribution": ATTRIBUTION,
            "scope": "selection-and-arrangement-only",
            "artifact_licenses_authoritative": True,
            "exceptions": [],
        },
        "limitations": ["Synthetic bytes exercise the release contract, not video decoding."],
        "dataset_card": card,
        "dataset_license": license_file,
        "captures": [_capture(source_root, CAPTURE_ID)],
    }
    plan_path = tmp_path / "build-plan.json"
    plan_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return plan_path, document


def _benchmark_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], Any]:
    benchmark_sources = tmp_path / "benchmark-sources"
    benchmark_sources.mkdir()
    bundle = benchmark_fixture_module._build_bundle(benchmark_sources)
    benchmark_manifest = bundle["manifest"]
    public_root = Path(bundle["public_root"])
    teacher_root = Path(bundle["teacher_root"])
    plan_path, document = _fixture(tmp_path / "public-plan")
    collection_attribution = benchmark_manifest["collection_license"]["attribution"]
    document["dataset_id"] = "cubed-core-synthetic-benchmark"
    document["release_id"] = "synthetic-benchmark-r1"
    document["description"] = (
        "A deterministic synthetic public corpus with Benchmark v0 inventories."
    )
    document["collection_license"]["attribution"] = collection_attribution
    document["limitations"] = [
        "Synthetic contract artifacts are not benchmark measurements or research data."
    ]
    _rewrite_source(
        document["dataset_license"],
        (
            "# Dataset license\n\n"
            "The selection and arrangement are licensed under CC-BY-4.0.\n\n"
            "Per-artifact licenses in the manifest are authoritative and may differ.\n\n"
            f"Required attribution: {collection_attribution}\n"
        ).encode(),
    )

    schema_for_role = {
        "video": None,
        "high_speed_original": None,
        "video_derivation": "capture-derivation-v1.schema.json",
        "scramble": "cubed-core-benchmark-v0-scramble.schema.json",
        "calibration": "color-centroids-v1.schema.json",
        "camera_metadata": "cubed-core-benchmark-v0-camera-metadata.schema.json",
        "teacher_truth": "cubed-core-benchmark-v0-teacher.schema.json",
        "rights_record": "cubed-core-benchmark-v0-rights.schema.json",
    }

    plan_captures: list[dict[str, Any]] = []
    benchmark_members: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    for benchmark_capture in benchmark_manifest["captures"]:
        capture_id = benchmark_capture["capture_id"]
        references = {
            **benchmark_capture["inputs"],
            "teacher_truth": benchmark_capture["teacher_truth"],
            "rights_record": benchmark_capture["rights_record"],
        }
        artifacts: list[dict[str, Any]] = []
        for role, reference in references.items():
            root = teacher_root if role == "teacher_truth" else public_root
            prefix = "teacher" if role == "teacher_truth" else "public-inputs"
            source_path = root / reference["path"]
            artifact = {
                "artifact_id": f"{role.replace('_', '-')}-{capture_id[:8]}",
                "role": role,
                "source_path": str(source_path),
                "public_path": f"{prefix}/{reference['path']}",
                "bytes": reference["bytes"],
                "sha256": reference["sha256"],
                "media_type": reference["media_type"],
                "public_license": reference["public_license"],
                "attribution": reference["attribution"],
                "privacy_reviewed": True,
                "rights_reviewed": True,
                "schema_file": schema_for_role[role],
            }
            artifacts.append(artifact)
            if reference["public_license"] != document["collection_license"]["license_id"]:
                exceptions.append(
                    {
                        "capture_id": capture_id,
                        "role": role,
                        "public_license": reference["public_license"],
                        "reason": ("This artifact uses its explicitly reviewed declared license."),
                    }
                )
        plan_captures.append(
            {
                "capture_id": capture_id,
                "source_status": benchmark_capture["source_status"],
                "provenance_status": "reviewed-source-and-authorship",
                "split": benchmark_capture["split"],
                "artifacts": artifacts,
            }
        )
        benchmark_members.append(
            {
                "capture_id": capture_id,
                "groups": benchmark_capture["groups"],
            }
        )

    document["captures"] = plan_captures
    document["collection_license"]["exceptions"] = exceptions
    document["benchmark_v0"] = {
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "split_policy": benchmark_manifest["split_policy"],
        "limitations": benchmark_manifest["limitations"],
        "captures": benchmark_members,
    }
    plan_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    def probe(path: Path, timeout_seconds: int) -> dict[str, Any]:
        del timeout_seconds
        high_speed = "video-source-240" in path.parts
        return {
            "codec": "h264",
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "width": 1920,
            "height": 1080,
            "measured_fps": {
                "numerator": 240 if high_speed else 120,
                "denominator": 1,
            },
            "frame_count": 720 if high_speed else 360,
            "audio_stream_count": 0,
        }

    return plan_path, document, probe


def _good_probe(path: Path, timeout_seconds: int) -> dict[str, Any]:
    del path, timeout_seconds
    return {
        "codec": "h264",
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "width": 1920,
        "height": 1080,
        "measured_fps": {"numerator": 120, "denominator": 1},
        "frame_count": 1200,
        "audio_stream_count": 0,
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_public_dataset_schemas_are_valid_draft_2020_12() -> None:
    repository = Path(__file__).resolve().parents[2]
    names = (
        "public-corpus-manifest-v1.schema.json",
        "public-corpus-rights-v1.schema.json",
        "public-corpus-scramble-v1.schema.json",
        "public-dataset-build-plan-v1.schema.json",
        "public-dataset-download-v1.schema.json",
        "public-dataset-metadata-row-v1.schema.json",
        "public-dataset-readiness-v1.schema.json",
        "public-dataset-splits-v1.schema.json",
        "public-dataset-teacher-manifest-v1.schema.json",
        "raw-capture-manifest-v1.schema.json",
        "raw-cube-session-v2.schema.json",
        "raw-imu-v1.schema.json",
    )
    for name in names:
        schema = json.loads((repository / "schemas" / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_artifact_schema_patterns_cover_runtime_allowlist() -> None:
    repository = Path(__file__).resolve().parents[2]
    allowed_schema_files = {
        schema_file
        for role_allowlist in public_dataset.ROLE_SCHEMA_ALLOWLIST.values()
        for schema_file in role_allowlist
        if schema_file is not None
    }

    for name in (
        "public-corpus-manifest-v1.schema.json",
        "public-dataset-build-plan-v1.schema.json",
    ):
        schema = json.loads((repository / "schemas" / name).read_text(encoding="utf-8"))
        pattern = schema["$defs"]["artifact"]["properties"]["schema_file"]["oneOf"][1]["pattern"]
        for schema_file in allowed_schema_files:
            assert re.fullmatch(pattern, schema_file), (
                f"{name} rejects runtime-allowed schema_file {schema_file}"
            )
        assert re.fullmatch(pattern, "../private-v1.schema.json") is None
        assert re.fullmatch(pattern, "private-v1.schema.json.backup") is None


def test_artifact_role_enums_match_runtime_roles() -> None:
    repository = Path(__file__).resolve().parents[2]
    for name in (
        "public-corpus-manifest-v1.schema.json",
        "public-dataset-build-plan-v1.schema.json",
    ):
        schema = json.loads((repository / "schemas" / name).read_text(encoding="utf-8"))
        enum = schema["$defs"]["artifact"]["properties"]["role"]["enum"]
        assert len(enum) == len(set(enum))
        assert set(enum) == set(public_dataset.ALL_ROLES), name
    assert set(public_dataset.ROLE_SCHEMA_ALLOWLIST) == set(public_dataset.ALL_ROLES)


def _load_schema_validator(name: str) -> Draft202012Validator:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads((repository / "schemas" / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _minimal_raw_cube_session() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "capture_session_id": "250ce08b-0000-4000-8000-000000000000",
        "recording_id": "f93528e0-0000-4000-8000-000000000000",
        "video_recording_id": "a" * 32,
        "video_link_status": "linked",
        "device": "GAN-a0000",
        "mode": "solve",
        "clock": {
            "schema_version": 1,
            "monotonic_start_ms": 1000,
            "relative_timebase": "host_performance_now",
            "move_event_time": "event_local_timestamp_or_host_monotonic_fallback",
            "orientation_event_time": "host_monotonic_receive",
        },
        "scramble": "U R' F2",
        "solve_ms": 12345,
        "started_unix_ms": 1753600000000,
        "zero_quat": [1.0, 0.0, 0.0, 0.0],
        "moves": [
            {
                "t_ms": 10,
                "move": "U",
                "facelets": "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9,
                "quat": [1.0, 0.0, 0.0, 0.0],
                "event_local_timestamp_ms": 100,
                "cube_timestamp_ms": 100,
                "serial": 7,
                "clock_source": "event_local_timestamp",
            }
        ],
        "orientations": [
            {
                "t_ms": 11,
                "quat": [1.0, 0.0, 0.0, 0.0],
                "clock_source": "host_monotonic_receive",
            }
        ],
    }


def _minimal_public_corpus_scramble() -> dict[str, Any]:
    return {
        "schema": "cubed-core/public-corpus-scramble",
        "schema_version": 1,
        "capture_id": CAPTURE_ID,
        "moves": ["U'", "B2", "L"],
        "scramble": {"notation": "U' B2 L", "moves": ["U'", "B2", "L"]},
        "initial_state": {
            "alphabet": "URFDLB",
            "facelets": "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9,
            "layout": "URFDLB",
        },
        "derivation": {
            "method": "app scramble field + full move-log replay verification",
            "source_file": "session.json",
            "engines": ["cubed-core src/cubed_core/cube/state.py Cube"],
            "move_count": 3,
            "verification": {"core_replay_solved": True},
        },
    }


def test_new_role_schemas_validate_minimal_fixtures() -> None:
    fixtures: dict[str, tuple[str, dict[str, Any]]] = {
        "ble_session": ("raw-cube-session-v2.schema.json", _minimal_raw_cube_session()),
        "ble_session_app": (
            "raw-cube-session-v2.schema.json",
            _minimal_raw_cube_session(),
        ),
        "imu": (
            "raw-imu-v1.schema.json",
            {
                "camera": {"facing": "front", "rotation": 0},
                "device_motion_available": True,
                "n": 1,
                "samples": [{"q": [1.0, 0.0, 0.0, 0.0], "t": 123}],
                "was_active": True,
            },
        ),
        "capture_manifest": (
            "raw-capture-manifest-v1.schema.json",
            {"content_sha256": "0" * 64, "payload": {"kind": "capture"}},
        ),
        "scramble": (
            "public-corpus-scramble-v1.schema.json",
            _minimal_public_corpus_scramble(),
        ),
    }
    for role, (schema_file, document) in fixtures.items():
        assert schema_file in public_dataset.ROLE_SCHEMA_ALLOWLIST[role]
        validator = _load_schema_validator(schema_file)
        errors = sorted(validator.iter_errors(document), key=str)
        assert not errors, f"{role}: {[error.message for error in errors]}"


def test_public_corpus_scramble_schema_rejects_unreshaped_sidecar() -> None:
    validator = _load_schema_validator("public-corpus-scramble-v1.schema.json")
    unreshaped = _minimal_public_corpus_scramble()
    del unreshaped["schema"]
    del unreshaped["schema_version"]
    del unreshaped["moves"]
    assert not validator.is_valid(unreshaped)
    pathlike_source = _minimal_public_corpus_scramble()
    pathlike_source["derivation"]["source_file"] = "datasets/private/session.json"
    assert not validator.is_valid(pathlike_source)


def test_build_is_deterministic_and_validates(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    first = tmp_path / "public-candidate-one"
    second = tmp_path / "public-candidate-two"

    first_report = public_dataset.build_public_dataset(plan_path, first, probe=_good_probe)
    second_report = public_dataset.build_public_dataset(plan_path, second, probe=_good_probe)

    assert first_report["ready"] is True
    assert first_report == second_report
    assert _tree_bytes(first) == _tree_bytes(second)
    assert public_dataset.validate_public_dataset(first, probe=_good_probe) == first_report
    assert (first / ".gitattributes").read_bytes() == public_dataset.HF_GITATTRIBUTES
    manifest = json.loads((first / "dataset/manifest.json").read_text())
    assert manifest["capture_count"] == 1
    assert manifest["benchmark_v0"] is None
    assert manifest["evidence_scope"] == ("corpus-not-benchmark-or-generalization-evidence")
    metadata = json.loads((first / "metadata.jsonl").read_text())
    assert metadata["file_name"] == f"captures/{CAPTURE_ID}/video.mp4"


def test_benchmark_build_is_deterministic_complete_and_validates(
    tmp_path: Path,
) -> None:
    plan_path, _, probe = _benchmark_fixture(tmp_path)
    first = tmp_path / "benchmark-candidate-one"
    second = tmp_path / "benchmark-candidate-two"

    first_report = public_dataset.build_public_dataset(
        plan_path,
        first,
        probe=probe,
    )
    second_report = public_dataset.build_public_dataset(
        plan_path,
        second,
        probe=probe,
    )

    assert first_report == second_report
    assert first_report["ready"] is True
    assert first_report["benchmark_v0_included"] is True
    assert _tree_bytes(first) == _tree_bytes(second)
    corpus = json.loads((first / "dataset/manifest.json").read_text())
    benchmark = json.loads((first / "public-inputs/manifest.json").read_text())
    splits = json.loads((first / "public-inputs/splits.json").read_text())
    teacher = json.loads((first / "teacher/manifest.json").read_text())
    assert corpus["benchmark_v0"] is not None
    assert benchmark["capture_count"] == 20
    assert benchmark["captures"] == sorted(
        benchmark["captures"],
        key=lambda item: item["capture_id"],
    )
    benchmark_sha256 = _sha256((first / "public-inputs/manifest.json").read_bytes())
    assert splits["benchmark_manifest_sha256"] == benchmark_sha256
    assert teacher["benchmark_manifest_sha256"] == benchmark_sha256
    assert teacher["capture_count"] == 20
    checksums = (first / "SHA256SUMS").read_text()
    for relative in public_dataset.BENCHMARK_PATHS.values():
        assert f"  {relative.as_posix()}\n" in checksums


def test_benchmark_builder_rejects_duplicate_member(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    members = document["benchmark_v0"]["captures"]
    members[1]["capture_id"] = members[0]["capture_id"]
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="duplicate capture_id",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_benchmark_builder_rejects_unknown_member(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    document["benchmark_v0"]["captures"][0]["capture_id"] = "c" * 32
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="absent from the public corpus",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_benchmark_builder_rejects_wrong_required_role_schema(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    scramble = next(
        artifact
        for artifact in document["captures"][0]["artifacts"]
        if artifact["role"] == "scramble"
    )
    scramble["schema_file"] = "cubed-core-benchmark-v0-camera-metadata.schema.json"
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="is not allowed for scramble",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_benchmark_builder_rejects_unassigned_member(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    selected_id = document["benchmark_v0"]["captures"][0]["capture_id"]
    capture = next(item for item in document["captures"] if item["capture_id"] == selected_id)
    capture["split"] = "unassigned"
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="requires an assigned",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_benchmark_builder_rejects_empty_fixed_split(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    for capture in document["captures"]:
        if capture["split"] == "test":
            capture["split"] = "validation"
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match=r"empty=\['test'\]",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_benchmark_builder_rejects_actual_probe_receipt_drift(
    tmp_path: Path,
) -> None:
    plan_path, _, probe = _benchmark_fixture(tmp_path)

    def drifted_probe(path: Path, timeout_seconds: int) -> dict[str, Any]:
        facts = probe(path, timeout_seconds)
        if "video-source-240" not in path.parts:
            facts["width"] += 1
        return facts

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="builder's media probe|reviewed camera receipt",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=drifted_probe,
        )


def test_benchmark_builder_rejects_wrong_public_namespace(
    tmp_path: Path,
) -> None:
    plan_path, document, probe = _benchmark_fixture(tmp_path)
    video = next(
        artifact for artifact in document["captures"][0]["artifacts"] if artifact["role"] == "video"
    )
    video["public_path"] = f"captures/{video['public_path'].split('/')[-1]}"
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="must be staged below public-inputs",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=probe,
        )


def test_builder_rejects_generated_benchmark_path_collision(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    video = document["captures"][0]["artifacts"][0]
    video["public_path"] = "public-inputs/manifest.json"

    def update_rights(rights: dict[str, Any]) -> None:
        rights["artifacts"][0]["path"] = video["public_path"]

    _rewrite_rights(document, update_rights)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="artifact path is reserved",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


@pytest.mark.parametrize(
    "public_path",
    (
        "dataset",
        "README.md/child.mp4",
    ),
)
def test_builder_rejects_generated_path_ancestor_collisions(
    tmp_path: Path,
    public_path: str,
) -> None:
    plan_path, document = _fixture(tmp_path)
    video = document["captures"][0]["artifacts"][0]
    video["public_path"] = public_path

    def update_rights(rights: dict[str, Any]) -> None:
        rights["artifacts"][0]["path"] = public_path

    _rewrite_rights(document, update_rights)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="conflicts with generated path",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_benchmark_validator_rejects_generic_reference_drift(
    tmp_path: Path,
) -> None:
    plan_path, _, probe = _benchmark_fixture(tmp_path)
    output = tmp_path / "benchmark-candidate"
    public_dataset.build_public_dataset(plan_path, output, probe=probe)
    corpus = json.loads((output / "dataset/manifest.json").read_text())
    public_path = output / "public-inputs/manifest.json"
    public_manifest = json.loads(public_path.read_text())
    public_manifest["captures"][0]["inputs"]["video"]["attribution"] += " tampered"
    public_path.write_bytes(public_dataset._json_bytes(public_manifest))
    public_hash = _sha256(public_path.read_bytes())

    for key in ("splits_manifest", "teacher_manifest"):
        path = output / corpus["benchmark_v0"][key]["path"]
        document = json.loads(path.read_text())
        document["benchmark_manifest_sha256"] = public_hash
        path.write_bytes(public_dataset._json_bytes(document))
    for key in public_dataset.BENCHMARK_PATHS:
        path = output / corpus["benchmark_v0"][key]["path"]
        corpus["benchmark_v0"][key]["bytes"] = path.stat().st_size
        corpus["benchmark_v0"][key]["sha256"] = _sha256(path.read_bytes())

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="input references disagree",
    ):
        public_dataset._validate_benchmark_binding(
            output,
            corpus["benchmark_v0"],
            corpus["captures"],
            corpus["collection_license"],
        )


def test_benchmark_validator_rejects_teacher_inventory_drift(
    tmp_path: Path,
) -> None:
    plan_path, _, probe = _benchmark_fixture(tmp_path)
    output = tmp_path / "benchmark-candidate"
    public_dataset.build_public_dataset(plan_path, output, probe=probe)
    corpus = json.loads((output / "dataset/manifest.json").read_text())
    teacher_path = output / "teacher/manifest.json"
    teacher = json.loads(teacher_path.read_text())
    teacher["artifacts"][0]["attribution"] += " tampered"
    teacher_path.write_bytes(public_dataset._json_bytes(teacher))
    binding = corpus["benchmark_v0"]["teacher_manifest"]
    binding["bytes"] = teacher_path.stat().st_size
    binding["sha256"] = _sha256(teacher_path.read_bytes())

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="teacher inventory does not exactly match",
    ):
        public_dataset._validate_benchmark_binding(
            output,
            corpus["benchmark_v0"],
            corpus["captures"],
            corpus["collection_license"],
        )


def test_builder_rejects_source_hash_mismatch_without_output(tmp_path: Path) -> None:
    plan_path, document = _fixture(tmp_path)
    document["captures"][0]["artifacts"][0]["sha256"] = "0" * 64
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    output = tmp_path / "public-candidate"

    with pytest.raises(public_dataset.PublicDatasetError, match="SHA-256 mismatch"):
        public_dataset.build_public_dataset(plan_path, output, probe=_good_probe)

    assert not output.exists()
    assert not list(tmp_path.glob(".public-candidate.staging-*"))


def test_builder_rejects_incomplete_hugging_face_dataset_card(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    card_path = Path(document["dataset_card"]["source_path"])
    replacement = b"# Missing Hugging Face metadata and release sections\n"
    card_path.write_bytes(replacement)
    document["dataset_card"]["bytes"] = len(replacement)
    document["dataset_card"]["sha256"] = _sha256(replacement)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="Hugging Face YAML front matter",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_placeholder_dataset_card_section(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    card = document["dataset_card"]
    replacement = (
        Path(card["source_path"])
        .read_bytes()
        .replace(
            b"No people or private data are present.",
            b"TODO",
        )
    )
    _rewrite_source(card, replacement)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="substantive reviewed text",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_token_or_placeholder_dataset_license(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    _rewrite_source(document["dataset_license"], b"TODO\n")
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="dataset license is inadequate",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("terms_version", "final-tbd"),
        ("reviewer_id", "reviewer-tbd"),
        ("terms_version", "example"),
        ("terms_version", "placeholder"),
    ),
)
def test_builder_rejects_placeholder_rights_decision_fields(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    plan_path, document = _fixture(tmp_path)
    _rewrite_rights(document, lambda rights: rights.__setitem__(field, value))
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="placeholder",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("terms_sha256", "1" * 64),
        ("acceptance_receipt_sha256", "ab" * 32),
    ),
)
def test_builder_rejects_repeated_placeholder_rights_hashes(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    plan_path, document = _fixture(tmp_path)
    _rewrite_rights(document, lambda rights: rights.__setitem__(field, value))
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="placeholder digest",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_placeholder_word_in_legitimate_prose_is_not_overrejected(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    document["description"] = (
        "For example, this synthetic corpus verifies deterministic release gates."
    )
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    report = public_dataset.build_public_dataset(
        plan_path,
        tmp_path / "public-candidate",
        probe=_good_probe,
    )

    assert report["ready"] is True


def test_builder_rejects_dataset_card_without_top_level_title(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    card_path = Path(document["dataset_card"]["source_path"])
    replacement = card_path.read_bytes().replace(
        b"# Synthetic public corpus\n",
        b"Synthetic public corpus\n",
    )
    card_path.write_bytes(replacement)
    document["dataset_card"]["bytes"] = len(replacement)
    document["dataset_card"]["sha256"] = _sha256(replacement)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="top-level",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"audio_stream_count": 1}, "audio streams are not allowed"),
        ({"width": 720, "height": 1280}, "minimum encoded short edge"),
        (
            {"measured_fps": {"numerator": 30, "denominator": 1}},
            "outside 110..121 fps",
        ),
    ),
)
def test_builder_rejects_unsupported_media(
    tmp_path: Path,
    change: dict[str, Any],
    message: str,
) -> None:
    plan_path, _ = _fixture(tmp_path)

    def invalid_probe(path: Path, timeout_seconds: int) -> dict[str, Any]:
        facts = _good_probe(path, timeout_seconds)
        facts.update(change)
        return facts

    with pytest.raises(public_dataset.PublicDatasetError, match=message):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=invalid_probe,
        )


def test_builder_rejects_duplicate_video_bytes(tmp_path: Path) -> None:
    plan_path, document = _fixture(tmp_path)
    first_video = document["captures"][0]["artifacts"][0]
    shared_source = {
        "source_path": first_video["source_path"],
        "bytes": first_video["bytes"],
        "sha256": first_video["sha256"],
    }
    document["captures"].append(
        _capture(
            tmp_path / "private-sources",
            SECOND_CAPTURE_ID,
            shared_video=shared_source,
        )
    )
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(public_dataset.PublicDatasetError, match="duplicate video bytes"):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_unknown_provenance(tmp_path: Path) -> None:
    plan_path, document = _fixture(tmp_path)
    document["captures"][0]["provenance_status"] = "unknown"
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(public_dataset.PublicDatasetError, match="schema violation"):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_dangling_collection_license_exception(
    tmp_path: Path,
) -> None:
    plan_path, document = _fixture(tmp_path)
    document["collection_license"]["exceptions"] = [
        {
            "capture_id": CAPTURE_ID,
            "role": "video",
            "public_license": PUBLIC_LICENSE,
            "reason": "This exception is intentionally unnecessary.",
        }
    ]
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="exceptions must cover exactly",
    ):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_unapproved_rights(tmp_path: Path) -> None:
    plan_path, document = _fixture(tmp_path)
    rights_artifact = document["captures"][0]["artifacts"][1]
    rights_path = Path(rights_artifact["source_path"])
    rights = json.loads(rights_path.read_text())
    rights["authorization"]["public_redistribution_authorized"] = False
    payload = (json.dumps(rights, indent=2, sort_keys=True) + "\n").encode()
    rights_path.write_bytes(payload)
    rights_artifact["bytes"] = len(payload)
    rights_artifact["sha256"] = _sha256(payload)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(public_dataset.PublicDatasetError, match="schema violation"):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )


def test_builder_rejects_symlink_source_and_existing_output(tmp_path: Path) -> None:
    plan_path, document = _fixture(tmp_path)
    video = document["captures"][0]["artifacts"][0]
    target = Path(video["source_path"])
    symlink = tmp_path / "video-link.mp4"
    symlink.symlink_to(target)
    video["source_path"] = str(symlink)
    plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(public_dataset.PublicDatasetError, match="symlink"):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )

    plan_path, _ = _fixture(tmp_path)
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(public_dataset.PublicDatasetError, match="refusing to overwrite"):
        public_dataset.build_public_dataset(plan_path, existing, probe=_good_probe)


def test_validator_rejects_extra_file_and_hash_corruption(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    output = tmp_path / "public-candidate"
    public_dataset.build_public_dataset(plan_path, output, probe=_good_probe)

    extra = output / "private-note.txt"
    extra.write_text("not allowlisted\n", encoding="utf-8")
    with pytest.raises(public_dataset.PublicDatasetError, match="file inventory mismatch"):
        public_dataset.validate_public_dataset(output, probe=_good_probe)
    extra.unlink()

    (output / f"captures/{CAPTURE_ID}/video.mp4").write_bytes(b"changed")
    with pytest.raises(public_dataset.PublicDatasetError, match="byte count mismatch"):
        public_dataset.validate_public_dataset(output, probe=_good_probe)


def test_validator_rejects_symlink_and_private_text(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    output = tmp_path / "public-candidate"
    public_dataset.build_public_dataset(plan_path, output, probe=_good_probe)
    (output / "linked").symlink_to(output / "README.md")

    with pytest.raises(public_dataset.PublicDatasetError, match="symlink"):
        public_dataset.validate_public_dataset(output, probe=_good_probe)

    (output / "linked").unlink()
    card = output / "README.md"
    card.write_text("# Dataset\n\nPrivate source: /Users/example/backup/video.mp4\n")
    with pytest.raises(public_dataset.PublicDatasetError, match="private path"):
        public_dataset.validate_public_dataset(output, probe=_good_probe)


def test_validator_rejects_hugging_face_attribute_drift(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    output = tmp_path / "public-candidate"
    public_dataset.build_public_dataset(plan_path, output, probe=_good_probe)
    (output / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")

    with pytest.raises(
        public_dataset.PublicDatasetError,
        match="deterministic public video policy",
    ):
        public_dataset.validate_public_dataset(output, probe=_good_probe)


def test_build_plan_must_remain_outside_git_checkout(tmp_path: Path) -> None:
    _, document = _fixture(tmp_path)
    repository = Path(__file__).resolve().parents[2]
    plan_path = repository / "build-plan-should-not-be-here.json"
    try:
        plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        with pytest.raises(public_dataset.PublicDatasetError, match="outside the Git checkout"):
            public_dataset.build_public_dataset(
                plan_path,
                tmp_path / "public-candidate",
                probe=_good_probe,
            )
    finally:
        plan_path.unlink(missing_ok=True)


def test_schema_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    original = plan_path.read_text(encoding="utf-8")
    plan_path.write_text(
        original.replace(
            '"schema": "cubed-core/public-dataset-build-plan",',
            (
                '"schema": "cubed-core/public-dataset-build-plan",'
                '"schema": "cubed-core/public-dataset-build-plan",'
            ),
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(public_dataset.PublicDatasetError, match="duplicate JSON key"):
        public_dataset.build_public_dataset(
            plan_path,
            tmp_path / "public-candidate",
            probe=_good_probe,
        )
