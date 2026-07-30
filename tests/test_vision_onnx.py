from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from cubed_core.vision import onnx as vision_onnx  # noqa: E402
from cubed_core.vision.onnx import (  # noqa: E402
    LazyOnnxModel,
    LetterboxTransform,
    OnnxModelError,
    OnnxRuntimeUnavailable,
    OnnxVisionClient,
    PosePostprocessConfig,
    VerifiedOnnxModel,
    classification_probabilities,
    postprocess_pose,
    preprocess_classification,
    preprocess_pose,
    resolve_execution_providers,
    snap_shared_corners,
)
from cubed_core.vision.types import FaceDetection  # noqa: E402


class _Node:
    def __init__(self, name: str, shape: list[object], type_name: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class _FakeSession:
    def __init__(self, runtime: _FakeRuntime, providers: list[str]):
        self.runtime = runtime
        self.providers = providers

    def get_inputs(self) -> list[_Node]:
        return [_Node("images", [1, 3, 4, 4])]

    def get_outputs(self) -> list[_Node]:
        return [_Node("scores", [1, 2])]

    def run(self, output_names: list[str], inputs: dict[str, object]) -> list[object]:
        self.runtime.run_calls.append((output_names, inputs))
        return [numpy.asarray([[0.8, 0.2]], numpy.float32)]


class _FakeRuntime:
    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def __init__(self):
        self.session_calls: list[tuple[str, object, list[str]]] = []
        self.run_calls: list[tuple[list[str], dict[str, object]]] = []

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    def InferenceSession(
        self,
        path: str,
        *,
        sess_options: object,
        providers: list[str],
    ) -> _FakeSession:
        self.session_calls.append((path, sess_options, providers))
        return _FakeSession(self, providers)


def _verified_file(tmp_path: Path, *, role: str = "alignment-classifier") -> VerifiedOnnxModel:
    path = tmp_path / f"{role}.onnx"
    path.write_bytes(f"synthetic-{role}".encode())
    return VerifiedOnnxModel(
        role=role,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_import_is_safe_and_missing_onnxruntime_error_is_precise() -> None:
    if importlib.util.find_spec("onnxruntime") is not None:
        pytest.skip("provider-specific absence check only applies without onnxruntime")
    with pytest.raises(OnnxRuntimeUnavailable, match="onnxruntime.*onnxruntime-gpu"):
        vision_onnx.available_execution_providers()


def test_provider_resolution_never_silently_drops_explicit_requests() -> None:
    available = ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert resolve_execution_providers(None, available=available) == available
    assert resolve_execution_providers(
        ("CPUExecutionProvider",),
        available=available,
    ) == ("CPUExecutionProvider",)
    with pytest.raises(OnnxModelError, match="unavailable"):
        resolve_execution_providers(("CoreMLExecutionProvider",), available=available)
    with pytest.raises(OnnxModelError, match="duplicated"):
        resolve_execution_providers(
            ("CPUExecutionProvider", "CPUExecutionProvider"),
            available=available,
        )


def test_lazy_model_verifies_digest_and_opens_session_only_on_first_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(vision_onnx, "_load_onnxruntime", lambda: runtime)
    model = LazyOnnxModel(
        _verified_file(tmp_path),
        providers=("CPUExecutionProvider",),
        intra_op_threads=2,
    )
    assert model.loaded is False
    assert runtime.session_calls == []

    result = model.run(numpy.zeros((1, 3, 4, 4), numpy.float32))

    assert model.loaded is True
    assert len(runtime.session_calls) == 1
    assert runtime.session_calls[0][2] == ["CPUExecutionProvider"]
    assert runtime.session_calls[0][1].intra_op_num_threads == 2
    assert numpy.allclose(result[0], [[0.8, 0.2]])
    model.run(numpy.zeros((1, 3, 4, 4), numpy.float32))
    assert len(runtime.session_calls) == 1


def test_lazy_model_rejects_artifact_change_before_session_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(vision_onnx, "_load_onnxruntime", lambda: runtime)
    source = _verified_file(tmp_path)
    source.path.write_bytes(b"changed-after-verification")
    model = LazyOnnxModel(source)

    with pytest.raises(OnnxModelError, match="changed"):
        model.run(numpy.zeros((1, 3, 4, 4), numpy.float32))
    assert runtime.session_calls == []


def test_manifest_backed_factory_checks_roles_and_does_not_open_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(vision_onnx, "_load_onnxruntime", lambda: runtime)
    alignment = _verified_file(tmp_path, role="alignment-classifier")
    pose = _verified_file(tmp_path, role="face-pose")
    alignment_artifact = SimpleNamespace(
        role=alignment.role,
        format="onnx",
        path=alignment.path,
        sha256=alignment.sha256,
    )
    pose_artifact = SimpleNamespace(
        role=pose.role,
        format="onnx",
        path=pose.path,
        sha256=pose.sha256,
    )

    client = OnnxVisionClient.from_verified_artifacts(
        alignment_artifact=alignment_artifact,
        face_pose_artifact=pose_artifact,
        providers=("CPUExecutionProvider",),
    )

    assert client.alignment_model.loaded is False
    assert client.face_pose_model.loaded is False
    assert runtime.session_calls == []
    with pytest.raises(OnnxModelError, match="role"):
        OnnxVisionClient.from_verified_artifacts(
            alignment_artifact=pose_artifact,
            face_pose_artifact=alignment_artifact,
        )


def test_vision_client_allows_label_threshold_override_without_a_second_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alignment_model = SimpleNamespace(source=SimpleNamespace(role="alignment-classifier"))
    face_pose_model = SimpleNamespace(
        source=SimpleNamespace(role="face-pose"),
        run=lambda _tensor: (numpy.zeros((1, 17, 1), numpy.float32),),
    )
    client = OnnxVisionClient(
        alignment_model=alignment_model,
        face_pose_model=face_pose_model,
    )
    override = PosePostprocessConfig(confidence_threshold=0.75)
    observed = []
    monkeypatch.setattr(
        vision_onnx,
        "preprocess_pose",
        lambda _frame, image_size: (
            numpy.zeros((1, 3, image_size, image_size), numpy.float32),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        vision_onnx,
        "postprocess_pose",
        lambda _output, _transform, *, config: observed.append(config) or (),
    )

    client.infer_faces(object(), pose_config=override)
    client.infer_faces(object())

    assert observed == [override, client.pose_config]


def test_classification_preprocessing_is_rgb_normalized_nchw() -> None:
    frame = numpy.zeros((40, 80, 3), numpy.uint8)
    frame[:] = (10, 20, 30)
    tensor = preprocess_classification(frame, image_size=32)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == numpy.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, :, 0, 0] == pytest.approx([30 / 255, 20 / 255, 10 / 255])


def test_pose_preprocessing_records_exact_inverse_letterbox() -> None:
    frame = numpy.zeros((100, 200, 3), numpy.uint8)
    tensor, transform = preprocess_pose(frame, image_size=320)
    assert tensor.shape == (1, 3, 320, 320)
    assert transform == LetterboxTransform(
        original_height=100,
        original_width=200,
        input_size=320,
        scale=1.6,
        pad_left=0,
        pad_top=80,
    )


def test_classification_postprocess_handles_probabilities_and_logits() -> None:
    probabilities = numpy.asarray([[0.25, 0.75]], numpy.float32)
    assert numpy.array_equal(classification_probabilities(probabilities), probabilities)
    logits = classification_probabilities(numpy.asarray([0.0, 1.0], numpy.float32))
    assert logits.shape == (2,)
    assert logits.sum() == pytest.approx(1.0)
    assert logits[1] > logits[0]


def _pose_output() -> numpy.ndarray:
    output = numpy.zeros((1, 17, 4), numpy.float32)
    rows = [
        # score, bbox, corners, keypoint scores
        (
            0.95,
            (60, 70, 40, 40),
            ((40, 50), (80, 50), (80, 90), (40, 90)),
            (0.9, 0.9, 0.9, 0.9),
        ),
        (
            0.80,
            (61, 71, 40, 40),
            ((41, 51), (81, 51), (81, 91), (41, 91)),
            (0.9, 0.9, 0.9, 0.9),
        ),
        (
            0.90,
            (120, 70, 40, 40),
            ((100, 50), (140, 50), (140, 90), (100, 90)),
            (0.9, 0.9, 0.9, 0.9),
        ),
        (
            0.99,
            (180, 70, 40, 40),
            ((160, 50), (200, 50), (200, 90), (160, 90)),
            (0.9, 0.1, 0.1, 0.1),
        ),
    ]
    for column, (score, bbox, corners, keypoint_scores) in enumerate(rows):
        output[0, :4, column] = bbox
        output[0, 4, column] = score
        for keypoint, ((x, y), confidence) in enumerate(zip(corners, keypoint_scores, strict=True)):
            output[0, 5 + keypoint * 3 : 8 + keypoint * 3, column] = (
                x,
                y,
                confidence,
            )
    return output


def test_pose_postprocess_filters_visibility_and_polygon_duplicates() -> None:
    result = postprocess_pose(
        _pose_output(),
        LetterboxTransform(
            original_height=100,
            original_width=100,
            input_size=200,
            scale=2.0,
            pad_left=0,
            pad_top=0,
        ),
        config=PosePostprocessConfig(
            minimum_visible_keypoints=3,
            snap_shared_corners=False,
        ),
    )

    assert len(result) == 2
    assert [face.confidence for face in result] == pytest.approx([0.95, 0.90])
    assert result[0].corners == (
        (20.0, 25.0),
        (40.0, 25.0),
        (40.0, 45.0),
        (20.0, 45.0),
    )
    assert result[1].corners[2] == (70.0, 45.0)


def _face_detection(
    corners: tuple[tuple[float, float], ...],
    *,
    keypoint_confidences: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9),
) -> FaceDetection:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return FaceDetection(
        corners=corners,
        keypoint_confidences=keypoint_confidences,
        confidence=0.9,
        bbox_xywh=(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs),
            max(ys) - min(ys),
        ),
    )


def test_shared_corner_snap_confidence_weights_both_edge_endpoints() -> None:
    left = _face_detection(
        ((0, 0), (10, 0), (10, 10), (0, 10)),
        keypoint_confidences=(0.9, 0.9, 0.9, 0.9),
    )
    right = _face_detection(
        ((10.5, 0), (20.5, 0), (20.5, 10), (10.5, 10)),
        keypoint_confidences=(0.3, 0.9, 0.9, 0.3),
    )

    snapped_left, snapped_right = snap_shared_corners((left, right))

    expected_x = (10.0 * 0.9 + 10.5 * 0.3) / 1.2
    assert snapped_left.corners[1] == pytest.approx((expected_x, 0.0))
    assert snapped_left.corners[2] == pytest.approx((expected_x, 10.0))
    assert snapped_right.corners[0] == pytest.approx((expected_x, 0.0))
    assert snapped_right.corners[3] == pytest.approx((expected_x, 10.0))


def test_shared_corner_snap_degeneracy_guard_rejects_collapse_and_concavity() -> None:
    square = numpy.asarray(((0, 0), (10, 0), (10, 10), (0, 10)), numpy.float64)
    collapsed = numpy.asarray(((0, 0), (1, 0), (1, 10), (0, 10)), numpy.float64)
    concave = numpy.asarray(((0, 0), (10, 0), (2, 2), (0, 10)), numpy.float64)

    assert vision_onnx._snap_preserves_quad(square, 100.0) is True
    assert vision_onnx._snap_preserves_quad(collapsed, 100.0) is False
    assert vision_onnx._snap_preserves_quad(concave, 100.0) is False


def test_pose_postprocess_snaps_adjacent_faces_by_default() -> None:
    result = postprocess_pose(
        _pose_output(),
        LetterboxTransform(
            original_height=100,
            original_width=100,
            input_size=200,
            scale=2.0,
            pad_left=0,
            pad_top=0,
        ),
        config=PosePostprocessConfig(minimum_visible_keypoints=3),
    )

    assert result[0].corners[1] == result[1].corners[0]
    assert result[0].corners[2] == result[1].corners[3]


def test_pose_postprocess_rejects_ambiguous_scores_or_shape() -> None:
    transform = LetterboxTransform(100, 100, 100, 1.0, 0, 0)
    invalid = _pose_output()
    invalid[0, 4, 0] = 2.0
    with pytest.raises(OnnxModelError, match="probabilities"):
        postprocess_pose(invalid, transform)
    with pytest.raises(OnnxModelError, match="shape"):
        postprocess_pose(numpy.zeros((1, 16, 2), numpy.float32), transform)
