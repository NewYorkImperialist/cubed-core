from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .media import extract_frame_jpeg

FRAME_ANNOTATIONS_SCHEMA = "cubed-core/frame-annotations"
MAX_ANNOTATION_BYTES = 64 * 1024**2
MAX_YOLO_EXPORT_FRAMES = 2_048
MAX_YOLO_EXPORT_BYTES = 2 * 1024**3
_FACE_NAMES = {"labeled", "top", "right", "bottom", "left", "back"}


class FrameAnnotationError(ValueError):
    pass


def _object(
    value: Any,
    *,
    field: str,
    required: set[str],
    allowed: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrameAnnotationError(f"{field} must be an object")
    missing = required - value.keys()
    if missing:
        raise FrameAnnotationError(f"{field} is missing {sorted(missing)[0]}")
    extra = value.keys() - allowed
    if extra:
        raise FrameAnnotationError(f"{field} contains unsupported field {sorted(extra)[0]}")
    return value


def _array(
    value: Any,
    *,
    field: str,
    maximum: int,
    minimum: int = 0,
    exact: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise FrameAnnotationError(f"{field} must be an array")
    if exact is not None and len(value) != exact:
        raise FrameAnnotationError(f"{field} must contain exactly {exact} items")
    if not minimum <= len(value) <= maximum:
        raise FrameAnnotationError(f"{field} must contain between {minimum} and {maximum} items")
    return value


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FrameAnnotationError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def _number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float | int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise FrameAnnotationError(
            f"{field} must be a finite number between {minimum} and {maximum}"
        )
    return value


def _string(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise FrameAnnotationError(
            f"{field} must be a string between {minimum} and {maximum} characters"
        )
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise FrameAnnotationError(f"{field} must be a boolean")
    return value


def _point(
    value: Any,
    *,
    field: str,
    width: int,
    height: int,
) -> list[float | int]:
    coordinates = _array(value, field=field, maximum=2, exact=2)
    return [
        _number(coordinates[0], field=f"{field}[0]", minimum=0, maximum=width),
        _number(coordinates[1], field=f"{field}[1]", minimum=0, maximum=height),
    ]


def normalize_face(
    value: Any,
    *,
    field: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    face = _object(
        value,
        field=field,
        required={"id", "corners", "visible"},
        allowed={
            "id",
            "corners",
            "visible",
            "vertices",
            "pinned",
            "assist",
            "origin",
            "confidence",
        },
    )
    normalized: dict[str, Any] = {
        "id": _string(face["id"], field=f"{field}.id", minimum=1, maximum=100),
        "corners": [
            _point(point, field=f"{field}.corners[{index}]", width=width, height=height)
            for index, point in enumerate(
                _array(face["corners"], field=f"{field}.corners", maximum=4, exact=4)
            )
        ],
        "visible": [
            _boolean(visible, field=f"{field}.visible[{index}]")
            for index, visible in enumerate(
                _array(face["visible"], field=f"{field}.visible", maximum=4, exact=4)
            )
        ],
    }
    if "vertices" in face:
        vertices: list[int | None] = []
        for index, vertex in enumerate(
            _array(face["vertices"], field=f"{field}.vertices", maximum=4, exact=4)
        ):
            vertices.append(
                None
                if vertex is None
                else _integer(
                    vertex,
                    field=f"{field}.vertices[{index}]",
                    minimum=0,
                    maximum=7,
                )
            )
        normalized["vertices"] = vertices
    if "pinned" in face:
        normalized["pinned"] = [
            _boolean(pinned, field=f"{field}.pinned[{index}]")
            for index, pinned in enumerate(
                _array(face["pinned"], field=f"{field}.pinned", maximum=4, exact=4)
            )
        ]
    if "origin" in face:
        origin = _string(
            face["origin"],
            field=f"{field}.origin",
            minimum=1,
            maximum=30,
        )
        if origin not in {"manual", "pnp-assist", "model"}:
            raise FrameAnnotationError(f"{field}.origin must be manual, pnp-assist, or model")
        normalized["origin"] = origin
    if "confidence" in face:
        normalized["confidence"] = _number(
            face["confidence"],
            field=f"{field}.confidence",
            minimum=0,
            maximum=1,
        )
    if "assist" in face:
        assist = _object(
            face["assist"],
            field=f"{field}.assist",
            required={"face_name", "generated"},
            allowed={"face_name", "generated", "intrinsics_source"},
        )
        face_name = _string(
            assist["face_name"],
            field=f"{field}.assist.face_name",
            minimum=1,
            maximum=20,
        )
        if face_name not in _FACE_NAMES:
            raise FrameAnnotationError(f"{field}.assist.face_name is unsupported")
        normalized_assist: dict[str, Any] = {
            "face_name": face_name,
            "generated": _boolean(
                assist["generated"],
                field=f"{field}.assist.generated",
            ),
        }
        if "intrinsics_source" in assist:
            normalized_assist["intrinsics_source"] = _string(
                assist["intrinsics_source"],
                field=f"{field}.assist.intrinsics_source",
                minimum=1,
                maximum=100,
            )
        normalized["assist"] = normalized_assist
    return normalized


def normalize_frame_annotations(
    value: Any,
    *,
    expected_capture_id: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a bounded, JSON-safe v1 annotation document.

    The contract remains version 1; the polygon, rigid-pose, and provenance
    fields are additive and older four-corner documents remain valid.
    """

    root = _object(
        value,
        field="annotation document",
        required={"schema", "schema_version", "created_at", "source", "image", "frames"},
        allowed={"schema", "schema_version", "created_at", "source", "image", "frames"},
    )
    if root["schema"] != FRAME_ANNOTATIONS_SCHEMA or root["schema_version"] != 1:
        raise FrameAnnotationError(
            "annotation document must use cubed-core/frame-annotations schema version 1"
        )
    created_at = _string(
        root["created_at"],
        field="annotation document.created_at",
        minimum=1,
        maximum=100,
    )
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FrameAnnotationError(
            "annotation document.created_at must be an ISO-8601 date-time"
        ) from exc

    source = _object(
        root["source"],
        field="annotation document.source",
        required={"kind", "capture_id", "filename", "sha256", "fps", "frame_count"},
        allowed={"kind", "capture_id", "filename", "sha256", "fps", "frame_count"},
    )
    kind = source["kind"]
    if kind not in {"workspace-capture", "local-file"}:
        raise FrameAnnotationError(
            "annotation document.source.kind must be workspace-capture or local-file"
        )
    capture_id = source["capture_id"]
    if capture_id is not None:
        capture_id = _string(
            capture_id,
            field="annotation document.source.capture_id",
            minimum=32,
            maximum=32,
        )
        if any(character not in "0123456789abcdef" for character in capture_id):
            raise FrameAnnotationError(
                "annotation document.source.capture_id must be a lowercase capture id"
            )
    sha256 = source["sha256"]
    if sha256 is not None:
        sha256 = _string(
            sha256,
            field="annotation document.source.sha256",
            minimum=64,
            maximum=64,
        )
        if any(character not in "0123456789abcdef" for character in sha256):
            raise FrameAnnotationError(
                "annotation document.source.sha256 must be a lowercase SHA-256 digest"
            )
    if kind == "workspace-capture" and (capture_id is None or sha256 is None):
        raise FrameAnnotationError(
            "workspace-capture annotations require capture_id and source sha256"
        )
    if kind == "local-file" and capture_id is not None:
        raise FrameAnnotationError("local-file annotations must use a null capture_id")
    if expected_capture_id is not None and (
        kind != "workspace-capture" or capture_id != expected_capture_id
    ):
        raise FrameAnnotationError("annotation source does not match the capture route")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise FrameAnnotationError("annotation source SHA-256 does not match the capture")
    frame_count = source["frame_count"]
    if frame_count is not None:
        frame_count = _integer(
            frame_count,
            field="annotation document.source.frame_count",
            minimum=1,
            maximum=10_000_000,
        )
    normalized_source = {
        "kind": kind,
        "capture_id": capture_id,
        "filename": _string(
            source["filename"],
            field="annotation document.source.filename",
            minimum=1,
            maximum=500,
        ),
        "sha256": sha256,
        "fps": _number(
            source["fps"],
            field="annotation document.source.fps",
            minimum=0.000_001,
            maximum=1_000,
        ),
        "frame_count": frame_count,
    }

    image = _object(
        root["image"],
        field="annotation document.image",
        required={"width", "height", "coordinate_space"},
        allowed={"width", "height", "coordinate_space"},
    )
    if image["coordinate_space"] != "display-oriented-video-pixels":
        raise FrameAnnotationError(
            "annotation document.image.coordinate_space must be display-oriented-video-pixels"
        )
    width = _integer(
        image["width"],
        field="annotation document.image.width",
        minimum=1,
        maximum=100_000,
    )
    height = _integer(
        image["height"],
        field="annotation document.image.height",
        minimum=1,
        maximum=100_000,
    )

    frames = _array(
        root["frames"],
        field="annotation document.frames",
        maximum=200_000,
    )
    normalized_frames: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for frame_position, frame_value in enumerate(frames):
        field = f"annotation document.frames[{frame_position}]"
        frame = _object(
            frame_value,
            field=field,
            required={"frame_index", "time_seconds", "aligned_label", "faces"},
            allowed={
                "frame_index",
                "time_seconds",
                "aligned_label",
                "faces",
                "polygons",
                "wireframe",
            },
        )
        frame_index = _integer(
            frame["frame_index"],
            field=f"{field}.frame_index",
            minimum=0,
            maximum=10_000_000,
        )
        if frame_index in seen_frames:
            raise FrameAnnotationError(
                f"annotation document.frames repeats frame_index {frame_index}"
            )
        seen_frames.add(frame_index)
        if frame_count is not None and frame_index >= frame_count:
            raise FrameAnnotationError(f"{field}.frame_index is outside source.frame_count")
        aligned_label = frame["aligned_label"]
        if aligned_label not in {"aligned", "unaligned", None}:
            raise FrameAnnotationError(f"{field}.aligned_label must be aligned, unaligned, or null")
        normalized_frame: dict[str, Any] = {
            "frame_index": frame_index,
            "time_seconds": _number(
                frame["time_seconds"],
                field=f"{field}.time_seconds",
                minimum=0,
                maximum=1_000_000,
            ),
            "aligned_label": aligned_label,
            "faces": [
                normalize_face(
                    face,
                    field=f"{field}.faces[{face_index}]",
                    width=width,
                    height=height,
                )
                for face_index, face in enumerate(
                    _array(frame["faces"], field=f"{field}.faces", maximum=64)
                )
            ],
        }
        if "polygons" in frame:
            polygons: list[dict[str, Any]] = []
            for polygon_index, polygon_value in enumerate(
                _array(frame["polygons"], field=f"{field}.polygons", maximum=64)
            ):
                polygon_field = f"{field}.polygons[{polygon_index}]"
                polygon = _object(
                    polygon_value,
                    field=polygon_field,
                    required={"id", "points"},
                    allowed={"id", "points"},
                )
                polygons.append(
                    {
                        "id": _string(
                            polygon["id"],
                            field=f"{polygon_field}.id",
                            minimum=1,
                            maximum=100,
                        ),
                        "points": [
                            _point(
                                point,
                                field=f"{polygon_field}.points[{point_index}]",
                                width=width,
                                height=height,
                            )
                            for point_index, point in enumerate(
                                _array(
                                    polygon["points"],
                                    field=f"{polygon_field}.points",
                                    minimum=3,
                                    maximum=128,
                                )
                            )
                        ],
                    }
                )
            normalized_frame["polygons"] = polygons
        if "wireframe" in frame:
            normalized_frame["wireframe"] = (
                None
                if frame["wireframe"] is None
                else [
                    _point(
                        point,
                        field=f"{field}.wireframe[{point_index}]",
                        width=width,
                        height=height,
                    )
                    for point_index, point in enumerate(
                        _array(
                            frame["wireframe"],
                            field=f"{field}.wireframe",
                            maximum=8,
                            exact=8,
                        )
                    )
                ]
            )
        normalized_frames.append(normalized_frame)

    return {
        "schema": FRAME_ANNOTATIONS_SCHEMA,
        "schema_version": 1,
        "created_at": created_at,
        "source": normalized_source,
        "image": {
            "width": width,
            "height": height,
            "coordinate_space": "display-oriented-video-pixels",
        },
        "frames": sorted(normalized_frames, key=lambda frame: frame["frame_index"]),
    }


def annotation_json_bytes(value: dict[str, Any]) -> bytes:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ANNOTATION_BYTES:
        raise FrameAnnotationError(
            f"annotation document exceeds the {MAX_ANNOTATION_BYTES}-byte limit"
        )
    return encoded


def _zip_write(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content)


def _yolo_line(face: dict[str, Any], width: int, height: int) -> str:
    corners = face["corners"]
    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    fields: list[str] = [
        "0",
        f"{((left + right) / 2) / width:.9f}",
        f"{((top + bottom) / 2) / height:.9f}",
        f"{(right - left) / width:.9f}",
        f"{(bottom - top) / height:.9f}",
    ]
    for point, visible in zip(corners, face["visible"], strict=True):
        fields.extend(
            [
                f"{float(point[0]) / width:.9f}",
                f"{float(point[1]) / height:.9f}",
                "2" if visible else "1",
            ]
        )
    return " ".join(fields)


def _coco_polygon(
    polygon: dict[str, Any],
    *,
    annotation_id: int,
    image_id: int,
) -> dict[str, Any]:
    points = polygon["points"]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    area = abs(
        sum(
            xs[index] * ys[(index + 1) % len(points)] - ys[index] * xs[(index + 1) % len(points)]
            for index in range(len(points))
        )
        / 2
    )
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [[coordinate for point in points for coordinate in point]],
        "bbox": [left, top, right - left, bottom - top],
        "area": area,
        "iscrowd": 0,
        "source_polygon_id": polygon["id"],
    }


def build_yolo_pose_archive(
    video_path: Path,
    document: dict[str, Any],
    output_path: Path,
    *,
    frame_extractor: Callable[[Path, int], bytes] = extract_frame_jpeg,
) -> Path:
    """Build one bounded exact-frame dataset with YOLO, COCO, and class folders."""

    frames = [
        frame
        for frame in document["frames"]
        if frame["faces"] or frame.get("polygons") or frame["aligned_label"] is not None
    ]
    if len(frames) > MAX_YOLO_EXPORT_FRAMES:
        raise FrameAnnotationError(
            f"YOLO export is limited to {MAX_YOLO_EXPORT_FRAMES} annotated frames"
        )
    width = int(document["image"]["width"])
    height = int(document["image"]["height"])
    capture_id = document["source"]["capture_id"]
    ranked = sorted(
        frames,
        key=lambda frame: hashlib.sha256(f"{capture_id}:{frame['frame_index']}".encode()).digest(),
    )
    validation_count = max(1, round(len(ranked) * 0.2)) if len(ranked) >= 5 else 0
    validation_frames = {frame["frame_index"] for frame in ranked[:validation_count]}
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, frame in enumerate(frames, start=1):
        split = "val" if frame["frame_index"] in validation_frames else "train"
        stem = f"frame_{frame['frame_index']:08d}"
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{split}/{stem}.jpg",
                "width": width,
                "height": height,
                "frame_index": frame["frame_index"],
                "aligned_label": frame["aligned_label"],
            }
        )
        for polygon in frame.get("polygons", []):
            coco_annotations.append(
                _coco_polygon(
                    polygon,
                    annotation_id=annotation_id,
                    image_id=image_id,
                )
            )
            annotation_id += 1
    if output_path.exists() or output_path.is_symlink():
        raise FrameAnnotationError("YOLO export target must be a new regular path")
    with zipfile.ZipFile(output_path, "x", allowZip64=True) as archive:
        dataset_yaml = (
            "path: .\n"
            "train: images/train\n"
            + ("val: images/val\n" if validation_count else "")
            + "kpt_shape: [4, 3]\n"
            "flip_idx: [1, 0, 3, 2]\n"
            "names:\n"
            "  0: cube-face\n"
        )
        _zip_write(
            archive,
            "dataset.yaml",
            dataset_yaml.encode(),
        )
        manifest = {
            "schema": "cubed-core/yolo-pose-export-v1",
            "source_capture_id": capture_id,
            "source_sha256": document["source"]["sha256"],
            "coordinate_space": document["image"]["coordinate_space"],
            "keypoint_order": ["corner-1", "corner-2", "corner-3", "corner-4"],
            "visibility": {"0": "not labeled", "1": "occluded", "2": "visible"},
            "frames": [frame["frame_index"] for frame in frames],
            "outputs": {
                "yolo_pose": "labels/{train,val}/*.txt",
                "coco_polygons": "coco/annotations.json",
                "alignment_images": "classification/{aligned,unaligned}/*.jpg",
            },
            "split": {
                "method": (
                    "deterministic-sha256-80-20-within-capture"
                    if validation_count
                    else "train-only-fewer-than-five-frames"
                ),
                "train_frames": [
                    frame["frame_index"]
                    for frame in frames
                    if frame["frame_index"] not in validation_frames
                ],
                "validation_frames": [
                    frame["frame_index"]
                    for frame in frames
                    if frame["frame_index"] in validation_frames
                ],
                "research_warning": (
                    "This within-capture convenience split is not a valid held-out "
                    "research or evaluation split. Split by capture/solve identity "
                    "before reporting metrics."
                ),
            },
        }
        _zip_write(
            archive,
            "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        coco = {
            "info": {
                "description": "Cubed Core exact-frame polygon export",
                "source_capture_id": capture_id,
                "source_sha256": document["source"]["sha256"],
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "cube-face", "supercategory": "cube"}],
        }
        _zip_write(
            archive,
            "coco/annotations.json",
            (json.dumps(coco, indent=2, sort_keys=True) + "\n").encode(),
        )
        for frame in frames:
            stem = f"frame_{frame['frame_index']:08d}"
            split = "val" if frame["frame_index"] in validation_frames else "train"
            jpeg = frame_extractor(video_path, frame["frame_index"])
            labels = "\n".join(_yolo_line(face, width, height) for face in frame["faces"])
            _zip_write(archive, f"images/{split}/{stem}.jpg", jpeg)
            _zip_write(
                archive,
                f"labels/{split}/{stem}.txt",
                ((labels + "\n") if labels else "").encode(),
            )
            if frame["aligned_label"] is not None:
                _zip_write(
                    archive,
                    f"classification/{frame['aligned_label']}/{stem}.jpg",
                    jpeg,
                )
            if output_path.stat().st_size > MAX_YOLO_EXPORT_BYTES:
                raise FrameAnnotationError(
                    f"YOLO export exceeds the {MAX_YOLO_EXPORT_BYTES}-byte limit"
                )
    return output_path
