#!/usr/bin/env python3
"""Build leakage-aware tracker datasets from explicit Cubed Core Label exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

LABEL_EXPORT_SCHEMA = "cubed-core/yolo-pose-export-v1"
TRAINING_DATASET_SCHEMA = "cubed-core/tracker-training-dataset-v1"
MAX_ARCHIVES = 128
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_LABEL_BYTES = 1024**2
MAX_MANIFEST_BYTES = 1024**2
_CAPTURE_ID = re.compile(r"[a-f0-9]{32}")
_IMAGE = re.compile(r"images/(?:train|val)/(frame_[0-9]{8})\.jpg")
_LABEL = re.compile(r"labels/(?:train|val)/(frame_[0-9]{8})\.txt")
_CLASS_IMAGE = re.compile(r"classification/(aligned|unaligned)/(frame_[0-9]{8})\.jpg")


class TrainingDataError(ValueError):
    """A safe, actionable Label-export preparation failure."""


@dataclass(frozen=True)
class PreparedGroup:
    group_id: str
    archive_sha256: str
    split: str
    pose_images: int
    aligned_images: int
    unaligned_images: int

    def public(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "archive_sha256": self.archive_sha256,
            "split": self.split,
            "pose_images": self.pose_images,
            "aligned_images": self.aligned_images,
            "unaligned_images": self.unaligned_images,
        }


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive(path: Path) -> Path:
    if path.is_symlink():
        raise TrainingDataError("Label export may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as exc:
        raise TrainingDataError("Label export is unavailable") from exc
    if not resolved.is_file() or not 1 <= size <= MAX_ARCHIVE_BYTES:
        raise TrainingDataError(
            f"Label export must be a nonempty ZIP no larger than {MAX_ARCHIVE_BYTES} bytes"
        )
    return resolved


def _members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not 1 <= len(infos) <= MAX_ARCHIVE_MEMBERS:
        raise TrainingDataError(
            f"Label export must contain between 1 and {MAX_ARCHIVE_MEMBERS} members"
        )
    total_size = 0
    members: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        posix = PurePosixPath(name)
        if (
            posix.is_absolute()
            or not posix.parts
            or any(part in {"", ".", ".."} for part in posix.parts)
            or "\\" in name
        ):
            raise TrainingDataError("Label export contains an unsafe member path")
        if name in members:
            raise TrainingDataError(f"Label export repeats member {name}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise TrainingDataError("Label export may not contain symlinks")
        if info.flag_bits & 0x1:
            raise TrainingDataError("Label export may not contain encrypted members")
        if info.file_size < 0:
            raise TrainingDataError("Label export contains an invalid member size")
        total_size += info.file_size
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise TrainingDataError("Label export exceeds the bounded uncompressed-size limit")
        members[name] = info
    return members


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    field: str,
    maximum: int,
) -> bytes:
    if info.file_size > maximum:
        raise TrainingDataError(f"{field} exceeds the {maximum}-byte limit")
    with archive.open(info, "r") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) != info.file_size or len(raw) > maximum:
        raise TrainingDataError(f"{field} could not be read within its declared bound")
    return raw


def _manifest(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
) -> dict[str, Any]:
    try:
        info = members["manifest.json"]
    except KeyError as exc:
        raise TrainingDataError("Label export is missing manifest.json") from exc
    raw = _read_member(
        archive,
        info,
        field="Label export manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrainingDataError(
            "Label export manifest must be finite JSON with unique keys"
        ) from exc
    if not isinstance(value, dict) or value.get("schema") != LABEL_EXPORT_SCHEMA:
        raise TrainingDataError(f"Label export manifest.schema must be {LABEL_EXPORT_SCHEMA}")
    capture_id = value.get("source_capture_id")
    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        raise TrainingDataError(
            "Label export manifest.source_capture_id must be 32 lowercase hex characters"
        )
    return value


def _validate_yolo_label(raw: bytes, *, field: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingDataError(f"{field} must be UTF-8 text") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 17 or fields[0] != "0":
            raise TrainingDataError(
                f"{field} line {line_number} must contain class 0, box, and four keypoints"
            )
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise TrainingDataError(f"{field} line {line_number} is not numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise TrainingDataError(f"{field} line {line_number} must be finite")
        continuous = values[:4] + [
            value for index, value in enumerate(values[4:]) if index % 3 != 2
        ]
        visibility = [value for index, value in enumerate(values[4:]) if index % 3 == 2]
        if any(not 0 <= value <= 1 for value in continuous):
            raise TrainingDataError(f"{field} line {line_number} has coordinates outside [0, 1]")
        if any(value not in {0, 1, 2} for value in visibility):
            raise TrainingDataError(
                f"{field} line {line_number} has unsupported keypoint visibility"
            )


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o644)


def _prepare_archive(
    path: Path,
    *,
    split: str,
    stage: Path,
) -> PreparedGroup:
    archive_sha = _sha256(path)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TrainingDataError("Label export must be a valid ZIP") from exc
    with archive:
        members = _members(archive)
        manifest = _manifest(archive, members)
        capture_id = manifest["source_capture_id"]
        group_id = hashlib.sha256(
            f"cubed-core/tracker-training-group-v1:{capture_id}".encode()
        ).hexdigest()[:20]

        images: dict[str, zipfile.ZipInfo] = {}
        labels: dict[str, zipfile.ZipInfo] = {}
        classes: dict[tuple[str, str], zipfile.ZipInfo] = {}
        for name, info in members.items():
            if match := _IMAGE.fullmatch(name):
                if match.group(1) in images:
                    raise TrainingDataError("Label export repeats a pose frame across splits")
                images[match.group(1)] = info
            elif match := _LABEL.fullmatch(name):
                if match.group(1) in labels:
                    raise TrainingDataError("Label export repeats a label frame across splits")
                labels[match.group(1)] = info
            elif match := _CLASS_IMAGE.fullmatch(name):
                key = (match.group(1), match.group(2))
                if key in classes:
                    raise TrainingDataError("Label export repeats a classification frame")
                classes[key] = info
        if not images:
            raise TrainingDataError("Label export contains no pose images")
        if set(images) != set(labels):
            raise TrainingDataError("Label export pose images and labels do not match exactly")

        image_digests: dict[str, str] = {}
        for stem in sorted(images):
            image = _read_member(
                archive,
                images[stem],
                field=f"pose image {stem}",
                maximum=MAX_IMAGE_BYTES,
            )
            if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                raise TrainingDataError(f"pose image {stem} is not a complete JPEG")
            label = _read_member(
                archive,
                labels[stem],
                field=f"pose label {stem}",
                maximum=MAX_LABEL_BYTES,
            )
            _validate_yolo_label(label, field=f"pose label {stem}")
            destination_stem = f"{group_id}_{stem}"
            _write(stage / "pose" / "images" / split / f"{destination_stem}.jpg", image)
            _write(stage / "pose" / "labels" / split / f"{destination_stem}.txt", label)
            image_digests[stem] = hashlib.sha256(image).hexdigest()

        class_counts = {"aligned": 0, "unaligned": 0}
        seen_class_stems: set[str] = set()
        for (label_name, stem), info in sorted(classes.items()):
            if stem not in images:
                raise TrainingDataError(
                    "classification image does not have a matching exact pose image"
                )
            if stem in seen_class_stems:
                raise TrainingDataError("one frame may not have both aligned and unaligned classes")
            image = _read_member(
                archive,
                info,
                field=f"classification image {stem}",
                maximum=MAX_IMAGE_BYTES,
            )
            if hashlib.sha256(image).hexdigest() != image_digests[stem]:
                raise TrainingDataError(
                    "classification image bytes differ from the matching exact pose image"
                )
            _write(
                stage / "alignment" / split / label_name / f"{group_id}_{stem}.jpg",
                image,
            )
            seen_class_stems.add(stem)
            class_counts[label_name] += 1
        return PreparedGroup(
            group_id=group_id,
            archive_sha256=archive_sha,
            split=split,
            pose_images=len(images),
            aligned_images=class_counts["aligned"],
            unaligned_images=class_counts["unaligned"],
        )


def _outside(path: Path, protected_roots: tuple[Path, ...]) -> None:
    for root in protected_roots:
        try:
            path.relative_to(root.resolve(strict=True))
        except ValueError:
            continue
        raise TrainingDataError("training dataset output must be outside the Cubed Core repository")


def _enclosing_git_root(path: Path) -> Path | None:
    current = path.resolve(strict=True)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.exists() or marker.is_symlink():
            return candidate
    return None


def prepare_tracker_training_data(
    train_exports: list[Path],
    validation_exports: list[Path],
    output_dir: Path,
    *,
    protected_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Combine Label ZIPs while keeping every capture group in one explicit split."""

    if not train_exports or not validation_exports:
        raise TrainingDataError(
            "provide at least one explicit train export and one explicit validation export"
        )
    if len(train_exports) + len(validation_exports) > MAX_ARCHIVES:
        raise TrainingDataError(f"at most {MAX_ARCHIVES} Label exports are supported")
    inputs = [
        (_safe_archive(path), split)
        for split, paths in (("train", train_exports), ("val", validation_exports))
        for path in paths
    ]
    input_paths = [path for path, _ in inputs]
    if len(set(input_paths)) != len(input_paths):
        raise TrainingDataError("each Label export must appear exactly once")
    output_dir = output_dir.absolute()
    if output_dir.name in {"", ".", ".."}:
        raise TrainingDataError("training dataset output must name a new directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise TrainingDataError("training dataset output must be a new path")
    try:
        parent = output_dir.parent.resolve(strict=True)
    except OSError as exc:
        raise TrainingDataError("training dataset output parent is unavailable") from exc
    output_dir = parent / output_dir.name
    _outside(output_dir, protected_roots)
    if _enclosing_git_root(parent) is not None:
        raise TrainingDataError(
            "training dataset output must be outside every enclosing Git worktree"
        )

    stage = Path(tempfile.mkdtemp(prefix=".cubed-core-training-data-", dir=parent))
    try:
        groups = [
            _prepare_archive(path, split=split, stage=stage)
            for path, split in sorted(inputs, key=lambda item: (_sha256(item[0]), item[1]))
        ]
        group_ids = [group.group_id for group in groups]
        if len(set(group_ids)) != len(group_ids):
            raise TrainingDataError("Label exports repeat the same source capture group")
        for split in ("train", "val"):
            split_groups = [group for group in groups if group.split == split]
            if not split_groups:
                raise TrainingDataError(f"{split} split must contain at least one capture group")
            for label_name in ("aligned", "unaligned"):
                count = sum(getattr(group, f"{label_name}_images") for group in split_groups)
                if count == 0:
                    raise TrainingDataError(
                        f"{split} split has no {label_name} classification images"
                    )
        pose_yaml = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [4, 3]\n"
            "flip_idx: [1, 0, 3, 2]\n"
            "names:\n"
            "  0: cube-face\n"
        )
        _write(stage / "pose" / "dataset.yaml", pose_yaml.encode("utf-8"))
        manifest = {
            "schema": TRAINING_DATASET_SCHEMA,
            "schema_version": 1,
            "split_policy": "explicit-capture-group",
            "groups": [group.public() for group in sorted(groups, key=lambda item: item.group_id)],
            "summary": {
                "capture_groups": len(groups),
                "train_capture_groups": sum(group.split == "train" for group in groups),
                "validation_capture_groups": sum(group.split == "val" for group in groups),
                "pose_images": sum(group.pose_images for group in groups),
                "aligned_images": sum(group.aligned_images for group in groups),
                "unaligned_images": sum(group.unaligned_images for group in groups),
            },
            "outputs": {
                "pose": "pose/dataset.yaml",
                "alignment": "alignment/{train,val}/{aligned,unaligned}",
            },
            "privacy": {
                "source_capture_ids_included": False,
                "source_paths_included": False,
                "group_ids": "one-way SHA-256-derived pseudonyms",
            },
        }
        _write(
            stage / "dataset-manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        stage.chmod(0o755)
        stage.replace(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "prepared",
        "schema": TRAINING_DATASET_SCHEMA,
        "dataset_manifest": "dataset-manifest.json",
        "pose_data": "pose/dataset.yaml",
        "alignment_data": "alignment",
        "summary": manifest["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare capture-group-separated tracker training data from Cubed Core "
            "Label dataset ZIPs."
        )
    )
    parser.add_argument(
        "--train-export",
        action="append",
        type=Path,
        default=[],
        help="Label dataset ZIP assigned wholly to train; repeat for more capture groups",
    )
    parser.add_argument(
        "--validation-export",
        action="append",
        type=Path,
        default=[],
        help="Label dataset ZIP assigned wholly to validation; repeat for more groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        report = prepare_tracker_training_data(
            args.train_export,
            args.validation_export,
            args.output_dir,
            protected_roots=(repository_root,),
        )
    except TrainingDataError as exc:
        print(f"tracker training data failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
