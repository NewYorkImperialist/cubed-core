from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from cubed_core.frame_annotations import (
    FrameAnnotationError,
    build_yolo_pose_archive,
    normalize_frame_annotations,
)


def annotation_document(*, frame_count: int = 10) -> dict[str, object]:
    return {
        "schema": "cubed-core/frame-annotations",
        "schema_version": 1,
        "created_at": "2026-07-23T12:00:00+00:00",
        "source": {
            "kind": "workspace-capture",
            "capture_id": "a" * 32,
            "filename": "solve.mp4",
            "sha256": "b" * 64,
            "fps": 120,
            "frame_count": frame_count,
        },
        "image": {
            "width": 640,
            "height": 480,
            "coordinate_space": "display-oriented-video-pixels",
        },
        "frames": [
            {
                "frame_index": 1,
                "time_seconds": 1 / 120,
                "aligned_label": "aligned",
                "polygons": [
                    {
                        "id": "polygon-1",
                        "points": [[10, 10], [30, 10], [20, 30]],
                    }
                ],
                "wireframe": [
                    [100, 100],
                    [200, 100],
                    [200, 200],
                    [100, 200],
                    [120, 80],
                    [220, 80],
                    [220, 180],
                    [120, 180],
                ],
                "faces": [
                    {
                        "id": "face-1",
                        "corners": [[100, 100], [200, 100], [200, 200], [100, 200]],
                        "visible": [True, True, False, True],
                        "vertices": [0, 1, 2, 3],
                        "pinned": [True, False, False, False],
                        "origin": "pnp-assist",
                        "confidence": 0.9,
                        "assist": {
                            "face_name": "labeled",
                            "generated": False,
                            "intrinsics_source": "explicit-camera-matrix",
                        },
                    }
                ],
            }
        ],
    }


def test_additive_annotation_fields_are_normalized_and_bounded() -> None:
    normalized = normalize_frame_annotations(
        annotation_document(),
        expected_capture_id="a" * 32,
        expected_sha256="b" * 64,
    )
    frame = normalized["frames"][0]
    assert frame["polygons"][0]["id"] == "polygon-1"
    assert frame["wireframe"][7] == [120, 180]
    assert frame["faces"][0]["vertices"] == [0, 1, 2, 3]
    assert frame["faces"][0]["pinned"] == [True, False, False, False]
    assert frame["faces"][0]["confidence"] == 0.9
    assert frame["faces"][0]["assist"]["generated"] is False


def test_annotation_validation_rejects_unknown_and_out_of_image_fields() -> None:
    unknown = annotation_document()
    unknown["frames"][0]["faces"][0]["private"] = True
    with pytest.raises(FrameAnnotationError, match="unsupported field private"):
        normalize_frame_annotations(unknown)

    outside = annotation_document()
    outside["frames"][0]["faces"][0]["corners"][0] = [-1, 10]
    with pytest.raises(FrameAnnotationError, match="finite number"):
        normalize_frame_annotations(outside)


def test_yolo_archive_is_file_backed_deterministic_and_has_distinct_split(
    tmp_path: Path,
) -> None:
    document = annotation_document(frame_count=20)
    template = document["frames"][0]
    document["frames"] = [
        {
            **json.loads(json.dumps(template)),
            "frame_index": frame_index,
            "time_seconds": frame_index / 120,
        }
        for frame_index in range(1, 6)
    ]
    output = tmp_path / "labels.zip"
    result = build_yolo_pose_archive(
        tmp_path / "source.mp4",
        normalize_frame_annotations(document),
        output,
        frame_extractor=lambda _path, frame: f"jpeg-{frame}".encode(),
    )
    assert result == output
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert any(name.startswith("images/train/") for name in names)
        assert any(name.startswith("images/val/") for name in names)
        assert any(name.startswith("classification/aligned/") for name in names)
        assert not any(name == "images/frame_00000001.jpg" for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        coco = json.loads(archive.read("coco/annotations.json"))
        assert manifest["split"]["train_frames"]
        assert manifest["split"]["validation_frames"]
        assert "not a valid held-out research" in manifest["split"]["research_warning"]
        assert len(coco["images"]) == 5
        assert len(coco["annotations"]) == 5
        assert coco["annotations"][0]["segmentation"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_tiny_yolo_export_is_train_only_without_false_validation_claim(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tiny.zip"
    build_yolo_pose_archive(
        tmp_path / "source.mp4",
        normalize_frame_annotations(annotation_document()),
        output,
        frame_extractor=lambda _path, _frame: b"jpeg",
    )
    with zipfile.ZipFile(output) as archive:
        yaml = archive.read("dataset.yaml").decode()
        manifest = json.loads(archive.read("manifest.json"))
    assert "val:" not in yaml
    assert manifest["split"]["method"] == "train-only-fewer-than-five-frames"
