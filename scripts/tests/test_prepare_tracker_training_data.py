from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_tracker_training_data import (
    LABEL_EXPORT_SCHEMA,
    TRAINING_DATASET_SCHEMA,
    TrainingDataError,
    main,
    prepare_tracker_training_data,
)

JPEG = b"\xff\xd8synthetic exact frame\xff\xd9"
LABEL = b"0 0.5 0.5 0.5 0.5 0.1 0.1 2 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2\n"


def _label_export(
    path: Path,
    *,
    capture_id: str,
    prefix: int,
    unsafe_member: str | None = None,
) -> Path:
    manifest = {
        "schema": LABEL_EXPORT_SCHEMA,
        "source_capture_id": capture_id,
    }
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for offset, label_name in enumerate(("aligned", "unaligned")):
            stem = f"frame_{prefix + offset:08d}"
            archive.writestr(f"images/train/{stem}.jpg", JPEG)
            archive.writestr(f"labels/train/{stem}.txt", LABEL)
            archive.writestr(f"classification/{label_name}/{stem}.jpg", JPEG)
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")
    return path


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_prepare_keeps_capture_groups_separate_and_is_deterministic(tmp_path: Path) -> None:
    inputs = tmp_path / "private-inputs"
    inputs.mkdir()
    train = _label_export(
        inputs / "owner-train.zip",
        capture_id="a" * 32,
        prefix=10,
    )
    validation = _label_export(
        inputs / "owner-validation.zip",
        capture_id="b" * 32,
        prefix=20,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = prepare_tracker_training_data([train], [validation], first)
    second_report = prepare_tracker_training_data([train], [validation], second)

    assert first_report == second_report
    assert _files(first) == _files(second)
    assert first_report["schema"] == TRAINING_DATASET_SCHEMA
    manifest = json.loads((first / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_policy"] == "explicit-capture-group"
    assert manifest["privacy"]["source_capture_ids_included"] is False
    assert manifest["privacy"]["source_paths_included"] is False
    serialized = json.dumps(manifest)
    assert "a" * 32 not in serialized
    assert "b" * 32 not in serialized
    assert str(inputs) not in serialized
    groups = manifest["groups"]
    assert {group["split"] for group in groups} == {"train", "val"}
    assert {group["archive_sha256"] for group in groups} == {
        hashlib.sha256(train.read_bytes()).hexdigest(),
        hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    assert len(list((first / "pose" / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((first / "pose" / "images" / "val").glob("*.jpg"))) == 2
    for split in ("train", "val"):
        for label_name in ("aligned", "unaligned"):
            assert len(list((first / "alignment" / split / label_name).glob("*.jpg"))) == 1


def test_prepare_requires_explicit_train_and_validation_groups(tmp_path: Path) -> None:
    archive = _label_export(
        tmp_path / "capture.zip",
        capture_id="c" * 32,
        prefix=1,
    )
    with pytest.raises(TrainingDataError, match="at least one explicit train"):
        prepare_tracker_training_data([archive], [], tmp_path / "output")


def test_prepare_rejects_duplicate_capture_group(tmp_path: Path) -> None:
    train = _label_export(
        tmp_path / "train.zip",
        capture_id="d" * 32,
        prefix=1,
    )
    validation = _label_export(
        tmp_path / "validation.zip",
        capture_id="d" * 32,
        prefix=10,
    )
    with pytest.raises(TrainingDataError, match="same source capture group"):
        prepare_tracker_training_data([train], [validation], tmp_path / "output")


def test_prepare_rejects_path_traversal_and_bad_labels(tmp_path: Path) -> None:
    unsafe = _label_export(
        tmp_path / "unsafe.zip",
        capture_id="e" * 32,
        prefix=1,
        unsafe_member="../escape",
    )
    valid = _label_export(
        tmp_path / "valid.zip",
        capture_id="f" * 32,
        prefix=10,
    )
    with pytest.raises(TrainingDataError, match="unsafe member path"):
        prepare_tracker_training_data([unsafe], [valid], tmp_path / "unsafe-output")

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "x") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"schema": LABEL_EXPORT_SCHEMA, "source_capture_id": "1" * 32}),
        )
        archive.writestr("images/train/frame_00000001.jpg", JPEG)
        archive.writestr("labels/train/frame_00000001.txt", b"0 0.5\n")
        archive.writestr("classification/aligned/frame_00000001.jpg", JPEG)
        archive.writestr("images/train/frame_00000002.jpg", JPEG)
        archive.writestr("labels/train/frame_00000002.txt", LABEL)
        archive.writestr("classification/unaligned/frame_00000002.jpg", JPEG)
    with pytest.raises(TrainingDataError, match="box, and four keypoints"):
        prepare_tracker_training_data([bad], [valid], tmp_path / "bad-output")


def test_prepare_refuses_repository_local_output(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "generated").mkdir()
    (repository / ".git").mkdir()
    train = _label_export(
        tmp_path / "train.zip",
        capture_id="2" * 32,
        prefix=1,
    )
    validation = _label_export(
        tmp_path / "validation.zip",
        capture_id="3" * 32,
        prefix=10,
    )
    with pytest.raises(TrainingDataError, match="outside"):
        prepare_tracker_training_data(
            [train],
            [validation],
            repository / "generated" / "dataset",
        )


def test_prepare_cli_reports_relative_outputs_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train = _label_export(
        tmp_path / "train.zip",
        capture_id="4" * 32,
        prefix=1,
    )
    validation = _label_export(
        tmp_path / "validation.zip",
        capture_id="5" * 32,
        prefix=10,
    )
    output = tmp_path / "dataset"

    status = main(
        [
            "--train-export",
            str(train),
            "--validation-export",
            str(validation),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pose_data"] == "pose/dataset.yaml"
    assert str(tmp_path) not in json.dumps(payload)
