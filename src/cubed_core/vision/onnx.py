from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._runtime import VisionRuntimeUnavailable, require_vision_runtime
from .geometry import GeometryError, quad_iou
from .types import FaceDetection

_SHA256 = re.compile(r"[a-f0-9]{64}")
_AUTOMATIC_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


class OnnxRuntimeUnavailable(VisionRuntimeUnavailable):
    """Raised when ONNX Runtime is needed but not installed."""


class OnnxModelError(RuntimeError):
    """Raised when a model, provider, tensor, or output violates its contract."""


class ModelNotConfigured(OnnxModelError):
    """Raised when an inference method has no verified artifact configured."""


class VerifiedArtifactLike(Protocol):
    role: str
    format: str
    path: Path
    sha256: str


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime
    except (ImportError, OSError) as exc:
        raise OnnxRuntimeUnavailable(
            "camera model inference requires ONNX Runtime; install exactly one compatible "
            "`onnxruntime` or `onnxruntime-gpu` package"
        ) from exc
    return onnxruntime


def available_execution_providers() -> tuple[str, ...]:
    runtime = _load_onnxruntime()
    return tuple(str(provider) for provider in runtime.get_available_providers())


def resolve_execution_providers(
    requested: Sequence[str] | None,
    *,
    available: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve providers without silently dropping an explicit request."""

    actual = tuple(available) if available is not None else available_execution_providers()
    if not actual:
        raise OnnxModelError("ONNX Runtime reports no execution providers")
    if requested is None:
        selected = tuple(provider for provider in _AUTOMATIC_PROVIDER_ORDER if provider in actual)
        if not selected:
            raise OnnxModelError(
                "no supported automatic ONNX provider is available; configure a provider explicitly"
            )
        return selected
    if isinstance(requested, (str, bytes)) or not requested:
        raise OnnxModelError("requested providers must be a nonempty sequence")
    selected: list[str] = []
    for index, provider in enumerate(requested):
        if not isinstance(provider, str) or not provider:
            raise OnnxModelError(f"requested provider {index} must be a nonempty string")
        if provider in selected:
            raise OnnxModelError(f"requested provider {provider} is duplicated")
        if provider not in actual:
            raise OnnxModelError(f"requested provider {provider} is unavailable")
        selected.append(provider)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class VerifiedOnnxModel:
    """The narrow handoff from the verified model-artifact registry."""

    role: str
    path: Path
    sha256: str

    @classmethod
    def from_artifact(
        cls,
        artifact: VerifiedArtifactLike,
        *,
        required_role: str,
    ) -> VerifiedOnnxModel:
        if artifact.role != required_role:
            raise OnnxModelError(
                f"verified artifact role must be {required_role}, found {artifact.role}"
            )
        if artifact.format != "onnx":
            raise OnnxModelError(f"verified {required_role} artifact must use ONNX format")
        path = artifact.path
        if not isinstance(path, Path) or not path.is_absolute():
            raise OnnxModelError(f"verified {required_role} artifact must expose an absolute Path")
        digest = artifact.sha256
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise OnnxModelError(
                f"verified {required_role} artifact must expose a lowercase SHA-256 digest"
            )
        return cls(role=required_role, path=path, sha256=digest)

    def __post_init__(self) -> None:
        if self.role not in {"alignment-classifier", "face-pose"}:
            raise OnnxModelError("ONNX model role must be alignment-classifier or face-pose")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise OnnxModelError("verified ONNX model path must be absolute")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise OnnxModelError("verified ONNX model digest must be lowercase SHA-256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise OnnxModelError(f"verified model artifact is unavailable: {path}") from exc
    return digest.hexdigest()


class LazyOnnxModel:
    """A checksum-bound ONNX session created only by the first inference."""

    def __init__(
        self,
        source: VerifiedOnnxModel,
        *,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
    ):
        if intra_op_threads is not None and (
            type(intra_op_threads) is not int or not 1 <= intra_op_threads <= 1024
        ):
            raise OnnxModelError("intra_op_threads must be an integer between 1 and 1024")
        self.source = source
        self._requested_providers = None if providers is None else tuple(providers)
        self._intra_op_threads = intra_op_threads
        self._session: Any | None = None
        self._input: Any | None = None
        self._outputs: tuple[Any, ...] = ()
        self._providers: tuple[str, ...] = ()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if self._session is not None:
            return
        path = self.source.path
        if path.is_symlink():
            raise OnnxModelError("verified model artifact may not become a symlink")
        try:
            stat = path.stat()
        except OSError as exc:
            raise OnnxModelError(f"verified model artifact is unavailable: {path}") from exc
        if not path.is_file() or stat.st_size <= 0:
            raise OnnxModelError("verified model artifact must be a nonempty regular file")
        actual_digest = _sha256(path)
        if actual_digest != self.source.sha256:
            raise OnnxModelError("verified model artifact changed after manifest verification")

        runtime = _load_onnxruntime()
        providers = resolve_execution_providers(
            self._requested_providers,
            available=runtime.get_available_providers(),
        )
        options = runtime.SessionOptions()
        if self._intra_op_threads is not None:
            options.intra_op_num_threads = self._intra_op_threads
            options.inter_op_num_threads = 1
        try:
            session = runtime.InferenceSession(
                str(path),
                sess_options=options,
                providers=list(providers),
            )
        except Exception as exc:
            raise OnnxModelError(
                f"could not open verified {self.source.role} ONNX artifact"
            ) from exc
        inputs = tuple(session.get_inputs())
        outputs = tuple(session.get_outputs())
        if len(inputs) != 1:
            raise OnnxModelError("ONNX model must expose exactly one input")
        if not outputs:
            raise OnnxModelError("ONNX model must expose at least one output")
        if getattr(inputs[0], "type", "tensor(float)") != "tensor(float)":
            raise OnnxModelError("ONNX model input must be float32")
        self._session = session
        self._input = inputs[0]
        self._outputs = outputs
        self._providers = providers

    @property
    def providers(self) -> tuple[str, ...]:
        self._load()
        return self._providers

    @property
    def supports_batch(self) -> bool:
        self._load()
        batch_dimension = self._input.shape[0]
        return not isinstance(batch_dimension, int) or batch_dimension != 1

    def run(self, input_tensor: Any) -> tuple[Any, ...]:
        _, numpy = require_vision_runtime()
        self._load()
        tensor = numpy.asarray(input_tensor)
        if tensor.dtype != numpy.float32:
            raise OnnxModelError("ONNX input tensor must use float32")
        if tensor.ndim != 4 or any(dimension <= 0 for dimension in tensor.shape):
            raise OnnxModelError("ONNX input tensor must have nonempty NCHW rank four")
        if not numpy.isfinite(tensor).all():
            raise OnnxModelError("ONNX input tensor must contain only finite values")
        expected_shape = tuple(self._input.shape)
        if len(expected_shape) != tensor.ndim:
            raise OnnxModelError("ONNX model input rank does not match the prepared tensor")
        for index, (expected, actual) in enumerate(zip(expected_shape, tensor.shape, strict=True)):
            if isinstance(expected, int) and expected > 0 and expected != actual:
                raise OnnxModelError(
                    f"ONNX input dimension {index} expected {expected}, found {actual}"
                )
        try:
            output_values = self._session.run(
                [output.name for output in self._outputs],
                {self._input.name: numpy.ascontiguousarray(tensor)},
            )
        except Exception as exc:
            raise OnnxModelError(f"{self.source.role} ONNX inference failed") from exc
        if len(output_values) != len(self._outputs):
            raise OnnxModelError("ONNX Runtime returned an unexpected number of outputs")
        result: list[Any] = []
        for index, value in enumerate(output_values):
            array = numpy.asarray(value)
            if not numpy.issubdtype(array.dtype, numpy.number):
                raise OnnxModelError(f"ONNX output {index} must be numeric")
            if not numpy.isfinite(array).all():
                raise OnnxModelError(f"ONNX output {index} contains non-finite values")
            result.append(array)
        return tuple(result)


def _image(value: Any) -> Any:
    _, numpy = require_vision_runtime()
    image = numpy.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise OnnxModelError("model input image must be a nonempty HxWx3 array")
    if image.dtype != numpy.uint8:
        raise OnnxModelError("model input image must use uint8 BGR pixels")
    return image


def _image_size(value: Any) -> int:
    if type(value) is not int or not 32 <= value <= 4096:
        raise OnnxModelError("model image size must be an integer between 32 and 4096")
    return value


def preprocess_classification(image: Any, *, image_size: int = 224) -> Any:
    """Resize the short edge, center-crop, RGB-normalize, and return NCHW."""

    cv2, numpy = require_vision_runtime()
    frame = _image(image)
    size = _image_size(image_size)
    height, width = frame.shape[:2]
    scale = size / min(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    x0 = (resized_width - size) // 2
    y0 = (resized_height - size) // 2
    crop = resized[y0 : y0 + size, x0 : x0 + size]
    tensor = crop[:, :, ::-1].astype(numpy.float32) / 255.0
    return numpy.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    input_size: int
    scale: float
    pad_left: int
    pad_top: int

    def __post_init__(self) -> None:
        if min(self.original_height, self.original_width, self.input_size) <= 0:
            raise OnnxModelError("letterbox dimensions must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise OnnxModelError("letterbox scale must be finite and positive")
        if self.pad_left < 0 or self.pad_top < 0:
            raise OnnxModelError("letterbox padding must be nonnegative")


def preprocess_pose(
    image: Any,
    *,
    image_size: int = 1024,
) -> tuple[Any, LetterboxTransform]:
    """Letterbox one BGR frame and return its exact inverse transform."""

    cv2, numpy = require_vision_runtime()
    frame = _image(image)
    size = _image_size(image_size)
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        size - resized_height - pad_top,
        pad_left,
        size - resized_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].astype(numpy.float32) / 255.0
    return (
        numpy.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...]),
        LetterboxTransform(
            original_height=height,
            original_width=width,
            input_size=size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        ),
    )


def classification_probabilities(output: Any) -> Any:
    """Return probabilities without guessing between per-row logits and softmax."""

    _, numpy = require_vision_runtime()
    values = numpy.asarray(output, dtype=numpy.float64)
    original_rank = values.ndim
    if original_rank == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise OnnxModelError("classification output must have shape (N, C) with C >= 2")
    if not numpy.isfinite(values).all():
        raise OnnxModelError("classification output must contain only finite values")
    row_sums = values.sum(axis=1)
    already_probabilities = (
        numpy.all(values >= 0.0)
        and numpy.all(values <= 1.0)
        and numpy.all(numpy.abs(row_sums - 1.0) <= 1e-3)
    )
    if already_probabilities:
        probabilities = values
    else:
        shifted = values - values.max(axis=1, keepdims=True)
        exponentials = numpy.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    result = probabilities.astype(numpy.float32)
    return result[0] if original_rank == 1 else result


@dataclass(frozen=True, slots=True)
class PosePostprocessConfig:
    class_count: int = 1
    keypoint_count: int = 4
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.15
    keypoint_confidence_threshold: float = 0.5
    minimum_visible_keypoints: int = 1
    maximum_detections: int = 16
    snap_shared_corners: bool = True

    def __post_init__(self) -> None:
        if type(self.class_count) is not int or not 1 <= self.class_count <= 1000:
            raise OnnxModelError("class_count must be an integer between 1 and 1000")
        if self.keypoint_count != 4:
            raise OnnxModelError("the cube face-pose contract requires exactly four keypoints")
        for field_name in (
            "confidence_threshold",
            "iou_threshold",
            "keypoint_confidence_threshold",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise OnnxModelError(f"{field_name} must be a finite probability")
        if (
            type(self.minimum_visible_keypoints) is not int
            or not 1 <= self.minimum_visible_keypoints <= self.keypoint_count
        ):
            raise OnnxModelError("minimum_visible_keypoints is outside the keypoint count")
        if type(self.maximum_detections) is not int or not 1 <= self.maximum_detections <= 1000:
            raise OnnxModelError("maximum_detections must be an integer between 1 and 1000")
        if type(self.snap_shared_corners) is not bool:
            raise OnnxModelError("snap_shared_corners must be a boolean")


DEFAULT_POSE_POSTPROCESS_CONFIG = PosePostprocessConfig()


def _quad_edge_and_area(corners: Any) -> tuple[float, float]:
    _, numpy = require_vision_runtime()
    edges = numpy.linalg.norm(corners - numpy.roll(corners, -1, axis=0), axis=1)
    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * abs(float(numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(y, numpy.roll(x, -1))))
    return float(edges.mean()), area


def _snap_preserves_quad(corners: Any, reference_area: float) -> bool:
    cv2, numpy = require_vision_runtime()
    _, area = _quad_edge_and_area(corners)
    if area < 0.5 * reference_area:
        return False
    hull = cv2.convexHull(numpy.asarray(corners, dtype=numpy.float32))
    return hull is not None and hull.reshape(-1, 2).shape == (4, 2)


def snap_shared_corners(
    detections: Sequence[FaceDetection],
    *,
    maximum_edge_fraction: float = 0.60,
    absolute_cap: float = 160.0,
) -> tuple[FaceDetection, ...]:
    """Confidence-weight corners belonging to a shared adjacent-face edge.

    Edges are matched jointly, vertices shared by three faces are unioned, and
    each proposed corner move is rejected if it makes the affected quad
    non-convex or retains less than half of its original area.
    """

    _, numpy = require_vision_runtime()
    if (
        isinstance(maximum_edge_fraction, bool)
        or not isinstance(maximum_edge_fraction, (int, float))
        or not math.isfinite(float(maximum_edge_fraction))
        or float(maximum_edge_fraction) < 0.0
    ):
        raise OnnxModelError("maximum_edge_fraction must be finite and nonnegative")
    if (
        isinstance(absolute_cap, bool)
        or not isinstance(absolute_cap, (int, float))
        or not math.isfinite(float(absolute_cap))
        or float(absolute_cap) < 0.0
    ):
        raise OnnxModelError("absolute_cap must be finite and nonnegative")
    if len(detections) < 2:
        return tuple(detections)

    corners_by_face = [
        numpy.asarray(detection.corners, dtype=numpy.float64) for detection in detections
    ]
    edge_scales = [_quad_edge_and_area(corners)[0] for corners in corners_by_face]
    items: list[tuple[int, int]] = []
    points: list[Any] = []
    confidences: list[float] = []
    face_start: dict[int, int] = {}
    for face_index, detection in enumerate(detections):
        face_start[face_index] = len(items)
        for corner_index, point in enumerate(detection.corners):
            items.append((face_index, corner_index))
            points.append(point)
            confidences.append(detection.keypoint_confidences[corner_index])
    point_array = numpy.asarray(points, dtype=numpy.float64)
    confidence_array = numpy.asarray(confidences, dtype=numpy.float64)
    parents = list(range(len(items)))

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = parents[root]
        while parents[index] != root:
            parents[index], index = root, parents[index]
        return root

    for first_index, first_corners in enumerate(corners_by_face):
        for second_index in range(first_index + 1, len(corners_by_face)):
            second_corners = corners_by_face[second_index]
            threshold = min(
                float(maximum_edge_fraction)
                * min(edge_scales[first_index], edge_scales[second_index]),
                float(absolute_cap),
            )
            best_cost: float | None = None
            best_pairs: tuple[tuple[int, int], tuple[int, int]] | None = None
            for first_corner in range(4):
                first_next = (first_corner + 1) % 4
                for second_corner in range(4):
                    second_next = (second_corner + 1) % 4
                    same_direction = float(
                        numpy.linalg.norm(
                            first_corners[first_corner] - second_corners[second_corner]
                        )
                        + numpy.linalg.norm(first_corners[first_next] - second_corners[second_next])
                    )
                    opposite_direction = float(
                        numpy.linalg.norm(first_corners[first_corner] - second_corners[second_next])
                        + numpy.linalg.norm(
                            first_corners[first_next] - second_corners[second_corner]
                        )
                    )
                    if opposite_direction < same_direction:
                        cost = opposite_direction
                        pairs = (
                            (first_corner, second_next),
                            (first_next, second_corner),
                        )
                    else:
                        cost = same_direction
                        pairs = (
                            (first_corner, second_corner),
                            (first_next, second_next),
                        )
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_pairs = pairs
            if best_pairs is not None and best_cost is not None and best_cost / 2.0 <= threshold:
                for first_corner, second_corner in best_pairs:
                    first_root = find(face_start[first_index] + first_corner)
                    second_root = find(face_start[second_index] + second_corner)
                    if first_root != second_root:
                        parents[first_root] = second_root

    clusters: dict[int, list[int]] = {}
    for item_index in range(len(items)):
        clusters.setdefault(find(item_index), []).append(item_index)
    targets: dict[tuple[int, int], Any] = {}
    for members in clusters.values():
        if len({items[member][0] for member in members}) < 2:
            continue
        weights = confidence_array[members]
        member_points = point_array[members]
        target = (
            (member_points * weights[:, None]).sum(axis=0) / weights.sum()
            if weights.sum() > 0.0
            else member_points.mean(axis=0)
        )
        for member in members:
            targets[items[member]] = target

    result: list[FaceDetection] = []
    for face_index, detection in enumerate(detections):
        candidate = corners_by_face[face_index].copy()
        reference_area = _quad_edge_and_area(candidate)[1]
        changed = False
        for corner_index in range(4):
            target = targets.get((face_index, corner_index))
            if target is None:
                continue
            proposed = candidate.copy()
            proposed[corner_index] = target
            if _snap_preserves_quad(proposed, reference_area):
                candidate = proposed
                changed = True
        if not changed:
            result.append(detection)
            continue
        result.append(
            FaceDetection(
                corners=tuple((float(point[0]), float(point[1])) for point in candidate),
                keypoint_confidences=detection.keypoint_confidences,
                confidence=detection.confidence,
                bbox_xywh=detection.bbox_xywh,
            )
        )
    return tuple(result)


def postprocess_pose(
    output: Any,
    transform: LetterboxTransform,
    *,
    config: PosePostprocessConfig = DEFAULT_POSE_POSTPROCESS_CONFIG,
) -> tuple[FaceDetection, ...]:
    """Decode a probability-valued YOLO-pose tensor with polygon-IoU NMS."""

    _, numpy = require_vision_runtime()
    values = numpy.asarray(output, dtype=numpy.float32)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise OnnxModelError("pose postprocessing accepts exactly one batch row")
        values = values[0]
    expected_channels = 4 + config.class_count + config.keypoint_count * 3
    if values.ndim != 2 or values.shape[0] != expected_channels:
        raise OnnxModelError(f"pose output must have shape (1, {expected_channels}, N)")
    if not numpy.isfinite(values).all():
        raise OnnxModelError("pose output must contain only finite values")
    predictions = values.T
    class_probabilities = predictions[:, 4 : 4 + config.class_count]
    if numpy.any(class_probabilities < 0.0) or numpy.any(class_probabilities > 1.0):
        raise OnnxModelError("pose class scores must already be probabilities")
    scores = (
        class_probabilities[:, 0] if config.class_count == 1 else class_probabilities.max(axis=1)
    )
    keypoints = predictions[:, 4 + config.class_count :].reshape(-1, config.keypoint_count, 3)
    if numpy.any(keypoints[:, :, 2] < 0.0) or numpy.any(keypoints[:, :, 2] > 1.0):
        raise OnnxModelError("pose keypoint scores must already be probabilities")

    candidates: list[tuple[int, float, Any, Any]] = []
    for index in numpy.flatnonzero(scores >= config.confidence_threshold):
        visible = int(
            numpy.count_nonzero(keypoints[index, :, 2] >= config.keypoint_confidence_threshold)
        )
        if visible < config.minimum_visible_keypoints:
            continue
        corners_model = keypoints[index, :, :2].astype(numpy.float32)
        try:
            # Geometry validation rejects degenerate and self-intersecting quads.
            quad_iou(corners_model, corners_model)
        except GeometryError:
            continue
        candidates.append(
            (
                int(index),
                float(scores[index]),
                corners_model,
                keypoints[index, :, 2].astype(numpy.float32),
            )
        )

    candidates.sort(key=lambda item: (-item[1], item[0]))
    kept: list[tuple[int, float, Any, Any]] = []
    for candidate in candidates:
        if all(quad_iou(candidate[2], previous[2]) < config.iou_threshold for previous in kept):
            kept.append(candidate)
        if len(kept) >= config.maximum_detections:
            break

    result: list[FaceDetection] = []
    for index, score, corners_model, keypoint_scores in kept:
        box_x, box_y, box_width, box_height = predictions[index, :4]
        corners = corners_model.copy()
        corners[:, 0] = (corners[:, 0] - transform.pad_left) / transform.scale
        corners[:, 1] = (corners[:, 1] - transform.pad_top) / transform.scale
        corners[:, 0] = numpy.clip(corners[:, 0], 0, transform.original_width)
        corners[:, 1] = numpy.clip(corners[:, 1], 0, transform.original_height)
        bbox = (
            (float(box_x) - transform.pad_left) / transform.scale,
            (float(box_y) - transform.pad_top) / transform.scale,
            float(box_width) / transform.scale,
            float(box_height) / transform.scale,
        )
        result.append(
            FaceDetection(
                corners=tuple((float(point[0]), float(point[1])) for point in corners),
                keypoint_confidences=tuple(float(value) for value in keypoint_scores),
                confidence=score,
                bbox_xywh=bbox,
            )
        )
    detections = tuple(result)
    if config.snap_shared_corners and len(detections) > 1:
        detections = snap_shared_corners(
            detections,
            absolute_cap=0.10 * transform.original_width,
        )
    return detections


class OnnxVisionClient:
    """Alignment and face-pose inference backed only by verified artifacts."""

    def __init__(
        self,
        *,
        alignment_model: LazyOnnxModel,
        face_pose_model: LazyOnnxModel,
        alignment_class_index: int = 0,
        alignment_image_size: int = 224,
        pose_image_size: int = 1024,
        pose_config: PosePostprocessConfig = DEFAULT_POSE_POSTPROCESS_CONFIG,
    ):
        if alignment_model.source.role != "alignment-classifier":
            raise OnnxModelError("alignment_model has the wrong verified role")
        if face_pose_model.source.role != "face-pose":
            raise OnnxModelError("face_pose_model has the wrong verified role")
        if type(alignment_class_index) is not int or alignment_class_index < 0:
            raise OnnxModelError("alignment_class_index must be a nonnegative integer")
        self.alignment_model = alignment_model
        self.face_pose_model = face_pose_model
        self.alignment_class_index = alignment_class_index
        self.alignment_image_size = _image_size(alignment_image_size)
        self.pose_image_size = _image_size(pose_image_size)
        self.pose_config = pose_config

    @classmethod
    def from_verified_artifacts(
        cls,
        *,
        alignment_artifact: VerifiedArtifactLike,
        face_pose_artifact: VerifiedArtifactLike,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
        **kwargs: Any,
    ) -> OnnxVisionClient:
        """Construct from registry-verified artifacts without filename discovery."""

        # Keep package import safe, but fail at client construction rather than
        # letting a configured tracker appear runnable without its model runtime.
        _load_onnxruntime()
        alignment_source = VerifiedOnnxModel.from_artifact(
            alignment_artifact,
            required_role="alignment-classifier",
        )
        pose_source = VerifiedOnnxModel.from_artifact(
            face_pose_artifact,
            required_role="face-pose",
        )
        return cls(
            alignment_model=LazyOnnxModel(
                alignment_source,
                providers=providers,
                intra_op_threads=intra_op_threads,
            ),
            face_pose_model=LazyOnnxModel(
                pose_source,
                providers=providers,
                intra_op_threads=intra_op_threads,
            ),
            **kwargs,
        )

    def classify_alignment(self, frame: Any) -> float:
        tensor = preprocess_classification(
            frame,
            image_size=self.alignment_image_size,
        )
        output = self.alignment_model.run(tensor)
        probabilities = classification_probabilities(output[0])
        row = probabilities[0] if probabilities.ndim == 2 else probabilities
        if self.alignment_class_index >= len(row):
            raise OnnxModelError("alignment class index is outside model output")
        return float(row[self.alignment_class_index])

    def classify_alignment_batch(self, frames: Sequence[Any]) -> tuple[float, ...]:
        if not frames:
            return ()
        tensors = [
            preprocess_classification(frame, image_size=self.alignment_image_size)
            for frame in frames
        ]
        if not self.alignment_model.supports_batch:
            return tuple(self.classify_alignment(frame) for frame in frames)
        _, numpy = require_vision_runtime()
        output = self.alignment_model.run(numpy.concatenate(tensors, axis=0))
        probabilities = classification_probabilities(output[0])
        if probabilities.ndim != 2 or probabilities.shape[0] != len(frames):
            raise OnnxModelError("batched alignment output does not match its input batch")
        if self.alignment_class_index >= probabilities.shape[1]:
            raise OnnxModelError("alignment class index is outside model output")
        return tuple(float(value) for value in probabilities[:, self.alignment_class_index])

    def infer_faces(
        self,
        frame: Any,
        *,
        pose_config: PosePostprocessConfig | None = None,
    ) -> tuple[FaceDetection, ...]:
        tensor, transform = preprocess_pose(frame, image_size=self.pose_image_size)
        output = self.face_pose_model.run(tensor)
        return postprocess_pose(
            output[0],
            transform,
            config=self.pose_config if pose_config is None else pose_config,
        )
