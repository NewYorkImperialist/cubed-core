from __future__ import annotations

from typing import Any

import pytest

numpy = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from cubed_core.vision.tracker import (  # noqa: E402
    FrameTracker,
    FrameTrackerConfig,
    TrackerError,
    admit_faces,
    face_motion_bbox,
    motion_in_bbox,
    rotate_frame,
)
from cubed_core.vision.types import FaceDetection  # noqa: E402


def _face(
    *,
    corners: tuple[tuple[float, float], ...] = (
        (4.0, 4.0),
        (20.0, 4.0),
        (20.0, 20.0),
        (4.0, 20.0),
    ),
    confidence: float = 0.9,
    keypoint_confidences: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9),
) -> FaceDetection:
    return FaceDetection(
        corners=corners,
        keypoint_confidences=keypoint_confidences,
        confidence=confidence,
        bbox_xywh=(12.0, 12.0, 16.0, 16.0),
    )


class FakeInference:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0
        self.pose_calls = 0

    @staticmethod
    def _alignment(frame: Any) -> float:
        return float(frame[0, 0, 0]) / 255.0

    def classify_alignment(self, frame: Any) -> float:
        self.single_calls += 1
        return self._alignment(frame)

    def classify_alignment_batch(self, frames: list[Any]) -> list[float]:
        self.batch_calls += 1
        return [self._alignment(frame) for frame in frames]

    def infer_faces(self, frame: Any) -> tuple[FaceDetection, ...]:
        self.pose_calls += 1
        return (_face(),)


class RecordingReader:
    def __init__(self):
        self.calls: list[tuple[int, tuple[FaceDetection, ...]]] = []

    def __call__(self, frame: Any, faces: tuple[FaceDetection, ...]) -> dict[str, int]:
        self.calls.append((int(frame[0, 0, 0]), faces))
        return {"face_count": len(faces)}


def _frame(alignment_byte: int, *, roi_value: int) -> Any:
    frame = numpy.zeros((32, 32, 3), numpy.uint8)
    frame[4:21, 4:21] = roi_value
    frame[0, 0, 0] = alignment_byte
    return frame


def _tracker(
    *,
    inference: FakeInference | None = None,
    reader: RecordingReader | None = None,
    gate_enabled: bool = True,
    rotation_degrees: int = 0,
) -> FrameTracker:
    return FrameTracker(
        config=FrameTrackerConfig(
            fps=120,
            rotation_degrees=rotation_degrees,
            gate_enabled=gate_enabled,
            alignment_threshold=0.5,
            minimum_aligned_frames=2,
            minimum_face_confidence=0.5,
        ),
        inference=inference or FakeInference(),
        reader=reader,
    )


def test_alignment_streak_gates_pose_and_reader_then_carries_motion_roi() -> None:
    inference = FakeInference()
    reader = RecordingReader()
    tracker = _tracker(inference=inference, reader=reader)

    first = tracker.feed(_frame(220, roi_value=10))
    second = tracker.feed(_frame(220, roi_value=30))
    third = tracker.feed(_frame(10, roi_value=70))

    assert first.frame_index == 1
    assert first.timestamp_seconds == pytest.approx(1 / 120)
    assert first.gate_open is False
    assert first.faces == ()
    assert second.frame_index == 2
    assert second.timestamp_seconds == pytest.approx(2 / 120)
    assert second.gate_open is True
    assert second.alignment_streak == 2
    assert second.motion_bbox == (4, 4, 20, 20)
    assert second.motion == pytest.approx(20.0)
    assert second.event_motion == second.motion
    assert second.read_result == {"face_count": 1}
    assert third.gate_open is False
    assert third.alignment_streak == 0
    assert third.faces == ()
    assert third.motion is None
    assert third.motion_bbox == second.motion_bbox
    assert third.event_motion == pytest.approx(40.0)
    assert inference.pose_calls == 1
    assert len(reader.calls) == 1
    snapshot = tracker.snapshot()
    assert snapshot.authoritative_motion_bbox == (4, 4, 20, 20)
    assert snapshot.alignment_scores == {1: 220 / 255, 2: 220 / 255, 3: 10 / 255}
    assert snapshot.reads == {2: {"face_count": 1}}


def test_event_motion_fails_closed_until_an_admitted_face_exists() -> None:
    tracker = _tracker()
    for value in (10, 20, 30):
        tracker.feed(_frame(0, roi_value=value))
    snapshot = tracker.snapshot()
    assert snapshot.authoritative_motion_bbox is None
    assert snapshot.event_motions == {}
    assert all(record.motion is None and record.event_motion is None for record in snapshot.records)


def test_feed_batch_matches_single_feed_and_uses_batch_classifier() -> None:
    frames = [
        _frame(220 if index not in {3, 4} else 20, roi_value=index * 10) for index in range(8)
    ]
    single_reader = RecordingReader()
    batch_reader = RecordingReader()
    single = _tracker(reader=single_reader)
    batch = _tracker(reader=batch_reader)
    for frame in frames:
        single.feed(frame)
    batch.feed_batch(frames[:3])
    batch.feed_batch(frames[3:])

    assert batch.snapshot() == single.snapshot()
    assert batch.inference.batch_calls == 2
    assert batch.inference.single_calls == 0


def test_batch_score_mismatch_fails_before_tracker_state_changes() -> None:
    class Truncated(FakeInference):
        def classify_alignment_batch(self, frames: list[Any]) -> list[float]:
            return [0.9] * (len(frames) - 1)

    tracker = _tracker(inference=Truncated())
    with pytest.raises(TrackerError, match="different number"):
        tracker.feed_batch([_frame(220, roi_value=1), _frame(220, roi_value=2)])
    assert tracker.frame_count == 0
    assert tracker.snapshot().authoritative_motion_bbox is None


def test_reader_failure_does_not_commit_partial_frame_state() -> None:
    class FailingReader:
        def __call__(self, frame: Any, faces: tuple[FaceDetection, ...]) -> object:
            raise RuntimeError("reader failed")

    tracker = FrameTracker(
        config=FrameTrackerConfig(
            fps=120,
            gate_enabled=False,
            minimum_face_confidence=0.5,
        ),
        inference=FakeInference(),
        reader=FailingReader(),
    )
    with pytest.raises(RuntimeError, match="reader failed"):
        tracker.feed(_frame(220, roi_value=10))
    assert tracker.frame_count == 0
    assert tracker.snapshot().authoritative_motion_bbox is None


def test_ungated_mode_skips_alignment_and_runs_pose_every_frame() -> None:
    inference = FakeInference()
    tracker = _tracker(inference=inference, gate_enabled=False)
    records = tracker.feed_batch([_frame(0, roi_value=10), _frame(0, roi_value=20)])
    assert all(record.gate_open for record in records)
    assert all(record.alignment_confidence is None for record in records)
    assert inference.single_calls == 0
    assert inference.batch_calls == 0
    assert inference.pose_calls == 2


def test_pose_admission_enforces_confidence_visibility_bounds_and_convexity() -> None:
    config = FrameTrackerConfig(
        fps=120,
        minimum_face_confidence=0.5,
        minimum_keypoint_confidence=0.5,
        minimum_visible_keypoints=3,
        minimum_quad_area=10,
    )
    valid = _face(confidence=0.9)
    low_detection = _face(confidence=0.2)
    low_keypoints = _face(keypoint_confidences=(0.9, 0.9, 0.1, 0.1))
    out_of_bounds = _face(corners=((-1, 4), (20, 4), (20, 20), (4, 20)))
    concave = _face(corners=((4, 4), (20, 4), (8, 8), (4, 20)))

    admitted = admit_faces(
        (low_keypoints, concave, valid, out_of_bounds, low_detection),
        frame_shape=(32, 32, 3),
        config=config,
    )

    assert admitted == (valid,)


def test_motion_helpers_use_slice_ready_union_and_exact_mean_difference() -> None:
    bbox = face_motion_bbox((_face(),), frame_shape=(32, 32, 3))
    assert bbox == (4, 4, 20, 20)
    previous = numpy.zeros((32, 32), numpy.uint8)
    current = previous.copy()
    current[4:21, 4:21] = 25
    assert motion_in_bbox(current, previous, bbox) == 25.0
    assert motion_in_bbox(current, None, bbox) is None


def test_rotation_uses_display_orientation_before_inference() -> None:
    frame = numpy.arange(3 * 4 * 3, dtype=numpy.uint8).reshape(3, 4, 3)
    assert numpy.array_equal(
        rotate_frame(frame, 90),
        cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
    )
    with pytest.raises(TrackerError, match="rotation"):
        rotate_frame(frame, 45)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"fps": 120, "rotation_degrees": 45},
        {"fps": 120, "minimum_aligned_frames": 0},
        {"fps": 120, "minimum_visible_keypoints": 5},
        {"fps": 120, "minimum_quad_area": 0},
    ],
)
def test_tracker_config_rejects_invalid_policy(kwargs: dict[str, object]) -> None:
    with pytest.raises(TrackerError):
        FrameTrackerConfig(**kwargs)
