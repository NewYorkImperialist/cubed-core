from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from .camera_tracker_backend import (
    MAX_VIDEO_FRAMES,
    CameraTrackerBackend,
    CameraTrackerBackendError,
)
from .command_runtime import resolve_command_executable
from .model_artifacts import ModelArtifactError, load_and_verify_tracker_model_manifest
from .settings import Settings
from .tracker_runtime import build_camera_tracker_readiness
from .vision import OnnxModelError, PosePostprocessConfig, TrackerError, rotate_frame
from .vision._runtime import VisionRuntimeUnavailable, require_vision_runtime

LABEL_PREDICTION_SCHEMA = "cubed-core/label-predictions-v1"
LABEL_ALIGNMENT_SCHEMA = "cubed-core/label-alignment-v1"
LABEL_ALIGNMENT_THRESHOLD = 0.50
LABEL_CURRENT_FRAME_FACE_THRESHOLD = 0.75
LABEL_PREDICTION_MAX_BYTES = 64 * 1024**2
LABEL_PREDICTION_TIMEOUT_SECONDS = 30 * 60
LABEL_PREDICTION_LOG_MAX_BYTES = 64 * 1024
_SECRET_ENVIRONMENT = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "CREDENTIAL",
    "AUTH",
)
_FORBIDDEN_ENVIRONMENT = {
    "CUBED_CORE_ADMIN_TOKEN",
    "GPG_AGENT_INFO",
    "KUBECONFIG",
    "NETRC",
    "SSH_AUTH_SOCK",
}
_NATIVE_CLIENTS: dict[tuple[object, ...], tuple[Any, str]] = {}
_NATIVE_CLIENTS_LOCK = threading.Lock()


class LabelPredictionError(ValueError):
    pass


class LabelPredictionDisabled(LabelPredictionError):
    pass


class _BoundedOutput:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.total = 0
        self.lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self.lock:
            self.total += len(chunk)
            if len(chunk) >= self.limit:
                self.data[:] = chunk[-self.limit :]
                return
            overflow = len(self.data) + len(chunk) - self.limit
            if overflow > 0:
                del self.data[:overflow]
            self.data.extend(chunk)

    def text(self) -> str:
        with self.lock:
            return self.data.decode("utf-8", errors="replace")


def _drain(stream: BinaryIO, output: _BoundedOutput) -> None:
    try:
        while chunk := stream.read(8192):
            output.append(chunk)
    except (OSError, ValueError):
        return


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _runner_environment() -> dict[str, str]:
    """Keep normal runtime/CUDA settings while withholding ambient credentials."""

    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _FORBIDDEN_ENVIRONMENT
        and not any(marker in key.upper() for marker in _SECRET_ENVIRONMENT)
    }


def _external_prediction_capability(settings: Settings) -> dict[str, Any]:
    if not settings.label_predict_command:
        return {
            "enabled": False,
            "status": "disabled",
            "reason": "configure a verified camera-tracker-v1 manifest",
            "executable": None,
            "execution_host": "api-host",
            "output_schema": LABEL_PREDICTION_SCHEMA,
            "backend": None,
            "model_profile": None,
            "model_aligned_navigation": False,
        }
    executable = resolve_command_executable(
        settings.label_predict_command,
        cwd=settings.repo_root,
    )
    model = settings.label_model_path
    if executable is None:
        reason = (
            "the configured label predictor executable was not found or is not executable "
            "on the API host"
        )
    elif model is None:
        reason = "CUBED_CORE_LABEL_MODEL_PATH is not configured"
    elif model.is_symlink() or not model.is_file():
        reason = "CUBED_CORE_LABEL_MODEL_PATH is not a regular local file"
    else:
        reason = ""
    return {
        "enabled": not reason,
        "status": "available" if not reason else "misconfigured",
        "reason": reason or None,
        "executable": executable,
        "execution_host": "api-host",
        "output_schema": LABEL_PREDICTION_SCHEMA,
        "backend": "external-command",
        "model_profile": None,
        "model_aligned_navigation": False,
    }


def prediction_capability(settings: Settings) -> dict[str, Any]:
    """Prefer the already-configured native tracker models for Label inference."""

    readiness = build_camera_tracker_readiness(settings)
    native = {
        "enabled": readiness.enabled,
        "status": readiness.status,
        "reason": readiness.reason,
        "executable": None,
        "execution_host": "api-host",
        "output_schema": LABEL_PREDICTION_SCHEMA,
        "backend": "camera-tracker-v1",
        "model_profile": readiness.models.get("profile"),
        "model_aligned_navigation": readiness.enabled,
    }
    if readiness.enabled:
        return native

    external = _external_prediction_capability(settings)
    if external["enabled"] or settings.label_predict_command:
        return external
    return native


def _number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise LabelPredictionError(
            f"{field} must be a finite number between {minimum} and {maximum}"
        )
    return float(value)


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise LabelPredictionError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def _point(value: Any, *, field: str, width: int, height: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise LabelPredictionError(f"{field} must contain x and y")
    return [
        _number(value[0], field=f"{field}[0]", minimum=0, maximum=width),
        _number(value[1], field=f"{field}[1]", minimum=0, maximum=height),
    ]


def _parse_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelPredictionError("prediction request must be an object")
    required = {"width", "height", "frame_indices"}
    allowed = {*required, "skip_frame_indices"}
    if not required.issubset(value) or set(value) - allowed:
        raise LabelPredictionError(
            "prediction request must contain width, height, frame_indices, "
            "and optional skip_frame_indices"
        )
    width = _integer(value["width"], field="width", minimum=1, maximum=100_000)
    height = _integer(value["height"], field="height", minimum=1, maximum=100_000)
    frame_indices = value["frame_indices"]
    if frame_indices is not None:
        if not isinstance(frame_indices, list) or not 1 <= len(frame_indices) <= 10_000:
            raise LabelPredictionError(
                "frame_indices must be null or an array with 1 to 10000 items"
            )
        parsed = [
            _integer(
                frame_index,
                field=f"frame_indices[{index}]",
                minimum=0,
                maximum=10_000_000,
            )
            for index, frame_index in enumerate(frame_indices)
        ]
        if len(set(parsed)) != len(parsed):
            raise LabelPredictionError("frame_indices must not contain duplicates")
        frame_indices = parsed
    skip_frame_indices = value.get("skip_frame_indices", [])
    if not isinstance(skip_frame_indices, list) or len(skip_frame_indices) > MAX_VIDEO_FRAMES:
        raise LabelPredictionError(
            f"skip_frame_indices must be an array with at most {MAX_VIDEO_FRAMES} items"
        )
    parsed_skip = [
        _integer(
            frame_index,
            field=f"skip_frame_indices[{index}]",
            minimum=0,
            maximum=10_000_000,
        )
        for index, frame_index in enumerate(skip_frame_indices)
    ]
    if len(set(parsed_skip)) != len(parsed_skip):
        raise LabelPredictionError("skip_frame_indices must not contain duplicates")
    if frame_indices is not None and parsed_skip:
        raise LabelPredictionError(
            "skip_frame_indices is supported only when frame_indices is null"
        )
    return {
        "width": width,
        "height": height,
        "frame_indices": frame_indices,
        "skip_frame_indices": parsed_skip,
    }


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _normalize_prediction_face(
    face_value: Any,
    *,
    field: str,
    frame_index: int,
    face_position: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    if not isinstance(face_value, dict):
        raise LabelPredictionError(f"{field} must be an object")
    allowed = {"corners", "visible", "confidence"}
    if set(face_value) - allowed or "corners" not in face_value:
        raise LabelPredictionError(f"{field} must contain corners and optional visible/confidence")
    corners = face_value["corners"]
    if not isinstance(corners, list) or len(corners) != 4:
        raise LabelPredictionError(f"{field}.corners must contain four points")
    visible = face_value.get("visible", [True, True, True, True])
    if (
        not isinstance(visible, list)
        or len(visible) != 4
        or any(type(item) is not bool for item in visible)
    ):
        raise LabelPredictionError(f"{field}.visible must contain four booleans")
    normalized_face = {
        "id": f"model-{frame_index}-{face_position}",
        "corners": [
            _point(
                point,
                field=f"{field}.corners[{corner_position}]",
                width=width,
                height=height,
            )
            for corner_position, point in enumerate(corners)
        ],
        "visible": visible,
        "origin": "model",
    }
    if "confidence" in face_value:
        normalized_face["confidence"] = _number(
            face_value["confidence"],
            field=f"{field}.confidence",
            minimum=0,
            maximum=1,
        )
    return normalized_face


def _normalize_prediction_frame(
    frame_value: Any,
    *,
    field: str,
    width: int,
    height: int,
    requested: set[int] | None,
    seen: set[int],
) -> dict[str, Any]:
    if not isinstance(frame_value, dict) or set(frame_value) != {"frame_index", "faces"}:
        raise LabelPredictionError(f"{field} must contain only frame_index and faces")
    frame_index = _integer(
        frame_value["frame_index"],
        field=f"{field}.frame_index",
        minimum=0,
        maximum=10_000_000,
    )
    if frame_index in seen:
        raise LabelPredictionError(f"predictor output repeats frame {frame_index}")
    if requested is not None and frame_index not in requested:
        raise LabelPredictionError(f"predictor returned unrequested frame {frame_index}")
    seen.add(frame_index)
    faces = frame_value["faces"]
    if not isinstance(faces, list) or len(faces) > 64:
        raise LabelPredictionError(f"{field}.faces must contain at most 64 faces")
    normalized_faces = [
        _normalize_prediction_face(
            face_value,
            field=f"{field}.faces[{face_position}]",
            frame_index=frame_index,
            face_position=face_position,
            width=width,
            height=height,
        )
        for face_position, face_value in enumerate(faces)
    ]
    return {"frame_index": frame_index, "faces": normalized_faces}


def _normalize_output(
    value: Any,
    *,
    width: int,
    height: int,
    requested_frames: list[int] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "frames"}:
        raise LabelPredictionError("predictor output must contain only schema and frames")
    if value["schema"] != LABEL_PREDICTION_SCHEMA:
        raise LabelPredictionError(f"predictor output must declare {LABEL_PREDICTION_SCHEMA}")
    frames = value["frames"]
    if not isinstance(frames, list) or len(frames) > 200_000:
        raise LabelPredictionError("predictor output frames must be a bounded array")
    requested = None if requested_frames is None else set(requested_frames)
    seen: set[int] = set()
    normalized_frames = [
        _normalize_prediction_frame(
            frame_value,
            field=f"predictor output.frames[{frame_position}]",
            width=width,
            height=height,
            requested=requested,
            seen=seen,
        )
        for frame_position, frame_value in enumerate(frames)
    ]
    return {
        "schema": LABEL_PREDICTION_SCHEMA,
        "frames": sorted(normalized_frames, key=lambda frame: frame["frame_index"]),
    }


def _run_external_label_prediction(
    settings: Settings,
    *,
    capture_id: str,
    video_path: Path,
    request: Any,
) -> dict[str, Any]:
    parsed = _parse_request(request)
    model_path = settings.label_model_path
    assert model_path is not None
    jobs_root = settings.workspace / "label-prediction-jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    if jobs_root.is_symlink():
        raise LabelPredictionError("label prediction job directory may not be a symlink")
    with tempfile.TemporaryDirectory(prefix="job-", dir=jobs_root) as temporary:
        job_dir = Path(temporary)
        request_path = job_dir / "request.json"
        output_path = job_dir / "output.json"
        runner_request = {
            "schema": "cubed-core/label-prediction-request-v1",
            "capture_id": capture_id,
            "video_path": str(video_path),
            "model_path": str(model_path),
            "width": parsed["width"],
            "height": parsed["height"],
            "frame_indices": parsed["frame_indices"],
        }
        request_path.write_text(
            json.dumps(runner_request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            *settings.label_predict_command,
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--model",
            str(model_path),
        ]
        process: subprocess.Popen[bytes] | None = None
        output = _BoundedOutput(LABEL_PREDICTION_LOG_MAX_BYTES)
        reader: threading.Thread | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=settings.repo_root,
                env=_runner_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=os.name == "posix",
            )
            assert process.stdout is not None
            reader = threading.Thread(
                target=_drain,
                args=(process.stdout, output),
                name="cubed-label-predict-log",
                daemon=True,
            )
            reader.start()
            try:
                return_code = process.wait(timeout=LABEL_PREDICTION_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                _stop_process(process)
                raise LabelPredictionError("label predictor timed out") from exc
        except OSError as exc:
            raise LabelPredictionError(f"label predictor could not start: {exc}") from exc
        finally:
            if reader is not None:
                reader.join(timeout=2)
                if reader.is_alive() and process is not None and process.stdout is not None:
                    process.stdout.close()
                    reader.join(timeout=1)
        if return_code != 0:
            detail = output.text().strip()
            raise LabelPredictionError(
                f"label predictor exited with status {return_code}: "
                f"{detail[:500] or 'no error output'}"
            )
        try:
            if output_path.is_symlink() or output_path.stat().st_size > LABEL_PREDICTION_MAX_BYTES:
                raise LabelPredictionError("label predictor output is unsafe or too large")
            value = json.loads(
                output_path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite,
            )
        except FileNotFoundError as exc:
            raise LabelPredictionError("label predictor did not write its output file") from exc
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            if isinstance(exc, LabelPredictionError):
                raise
            raise LabelPredictionError("label predictor output is not valid JSON") from exc
        normalized = _normalize_output(
            value,
            width=parsed["width"],
            height=parsed["height"],
            requested_frames=parsed["frame_indices"],
        )
        skipped = set(parsed["skip_frame_indices"])
        if skipped:
            normalized["frames"] = [
                frame for frame in normalized["frames"] if frame["frame_index"] not in skipped
            ]
        return normalized


def _native_vision_client(settings: Settings) -> tuple[Any, str]:
    manifest_path = settings.tracker_model_manifest
    if manifest_path is None:
        raise LabelPredictionDisabled("CUBED_CORE_TRACKER_MODEL_MANIFEST is not configured")
    try:
        manifest = load_and_verify_tracker_model_manifest(manifest_path)
        key = (
            str(manifest.manifest_path),
            tuple(
                (artifact.role, artifact.sha256, artifact.bytes) for artifact in manifest.artifacts
            ),
            settings.tracker_onnx_providers,
        )
        with _NATIVE_CLIENTS_LOCK:
            cached = _NATIVE_CLIENTS.get(key)
            if cached is not None:
                return cached
            backend = CameraTrackerBackend.from_verified_manifest(
                manifest,
                providers=settings.tracker_onnx_providers or None,
            )
            result = (backend.inference, manifest.profile)
            _NATIVE_CLIENTS[key] = result
            return result
    except (ModelArtifactError, CameraTrackerBackendError) as exc:
        raise LabelPredictionError(f"native Label models are unavailable: {exc}") from exc


def _validated_rotation(value: Any) -> int:
    if type(value) is not int or value not in {0, 90, 180, 270}:
        raise LabelPredictionError("capture rotation must be 0, 90, 180, or 270 degrees")
    return value


def _open_native_capture(video_path: Path) -> tuple[Any, Any]:
    if video_path.is_symlink() or not video_path.is_file():
        raise LabelPredictionError("capture video must be a regular local file")
    try:
        cv2, _ = require_vision_runtime()
        capture = cv2.VideoCapture(str(video_path))
        if not bool(capture.isOpened()):
            capture.release()
            raise LabelPredictionError("capture video could not be opened")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if (
            not isinstance(count, bool)
            and isinstance(count, (int, float))
            and math.isfinite(float(count))
            and float(count) > MAX_VIDEO_FRAMES
        ):
            capture.release()
            raise LabelPredictionError(
                f"capture video exceeds the {MAX_VIDEO_FRAMES}-frame Label limit"
            )
        return cv2, capture
    except LabelPredictionError:
        raise
    except VisionRuntimeUnavailable as exc:
        raise LabelPredictionError(f"native Label runtime is unavailable: {exc}") from exc
    except Exception as exc:
        raise LabelPredictionError(
            f"capture video could not be opened ({type(exc).__name__})"
        ) from exc


def _display_frame(
    frame: Any,
    *,
    rotation_degrees: int,
    width: int | None = None,
    height: int | None = None,
) -> Any:
    try:
        displayed = rotate_frame(frame, rotation_degrees)
    except (TrackerError, VisionRuntimeUnavailable) as exc:
        raise LabelPredictionError(f"capture frame is invalid: {exc}") from exc
    if width is not None and height is not None:
        actual_height, actual_width = displayed.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise LabelPredictionError(
                "decoded model frame dimensions do not match the Label display; "
                "check capture rotation metadata"
            )
    return displayed


def _prediction_face(face: Any) -> dict[str, Any]:
    return {
        "corners": [
            [round(float(point[0]), 1), round(float(point[1]), 1)] for point in face.corners
        ],
        "visible": [True, True, True, True],
        "confidence": round(float(face.confidence), 3),
    }


def _run_native_label_prediction(
    settings: Settings,
    *,
    video_path: Path,
    request: Any,
    rotation_degrees: int,
) -> dict[str, Any]:
    parsed = _parse_request(request)
    rotation = _validated_rotation(rotation_degrees)
    client, _profile = _native_vision_client(settings)
    cv2, capture = _open_native_capture(video_path)
    requested = parsed["frame_indices"]
    skipped = set(parsed["skip_frame_indices"])
    pose_config: PosePostprocessConfig | None = None
    if requested is not None:
        pose_config = replace(
            client.pose_config,
            confidence_threshold=LABEL_CURRENT_FRAME_FACE_THRESHOLD,
        )
    raw_frames: list[dict[str, Any]] = []
    try:
        if requested is not None:
            for frame_index in requested:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                available, raw_frame = capture.read()
                if not available:
                    raise LabelPredictionError(f"frame {frame_index} is outside the video")
                frame = _display_frame(
                    raw_frame,
                    rotation_degrees=rotation,
                    width=parsed["width"],
                    height=parsed["height"],
                )
                faces = client.infer_faces(frame, pose_config=pose_config)
                raw_frames.append(
                    {
                        "frame_index": frame_index,
                        "faces": [_prediction_face(face) for face in faces],
                    }
                )
        else:
            frame_index = 0
            while True:
                available, raw_frame = capture.read()
                if not available:
                    break
                if frame_index >= MAX_VIDEO_FRAMES:
                    raise LabelPredictionError(
                        f"capture video exceeds the {MAX_VIDEO_FRAMES}-frame Label limit"
                    )
                if frame_index in skipped:
                    frame_index += 1
                    continue
                frame = _display_frame(
                    raw_frame,
                    rotation_degrees=rotation,
                    width=parsed["width"],
                    height=parsed["height"],
                )
                faces = client.infer_faces(frame)
                if faces:
                    raw_frames.append(
                        {
                            "frame_index": frame_index,
                            "faces": [_prediction_face(face) for face in faces],
                        }
                    )
                frame_index += 1
            if frame_index == 0:
                raise LabelPredictionError("capture video contains no readable frames")
    except LabelPredictionError:
        raise
    except (OnnxModelError, TrackerError, VisionRuntimeUnavailable) as exc:
        raise LabelPredictionError(f"native Label inference failed: {exc}") from exc
    except Exception as exc:
        raise LabelPredictionError(f"native Label inference failed ({type(exc).__name__})") from exc
    finally:
        try:
            capture.release()
        except Exception:
            pass
    return _normalize_output(
        {
            "schema": LABEL_PREDICTION_SCHEMA,
            "frames": raw_frames,
        },
        width=parsed["width"],
        height=parsed["height"],
        requested_frames=requested,
    )


def run_label_alignment_scan(
    settings: Settings,
    *,
    video_path: Path,
    rotation_degrees: int,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    capability = prediction_capability(settings)
    if not capability["enabled"] or capability["backend"] != "camera-tracker-v1":
        raise LabelPredictionDisabled(
            str(
                capability["reason"]
                or "model-aligned navigation requires the native camera-tracker-v1 backend"
            )
        )
    if expected_frame_count is not None and (
        type(expected_frame_count) is not int or not 1 <= expected_frame_count <= MAX_VIDEO_FRAMES
    ):
        raise LabelPredictionError(f"capture frame count must be between 1 and {MAX_VIDEO_FRAMES}")
    rotation = _validated_rotation(rotation_degrees)
    client, profile = _native_vision_client(settings)
    _cv2, capture = _open_native_capture(video_path)
    alignment_confs: list[float] = []
    width = 0
    height = 0
    try:
        while expected_frame_count is None or len(alignment_confs) < expected_frame_count:
            available, raw_frame = capture.read()
            if not available:
                if expected_frame_count is None:
                    break
                alignment_confs.append(0.0)
                continue
            if len(alignment_confs) >= MAX_VIDEO_FRAMES:
                raise LabelPredictionError(
                    f"capture video exceeds the {MAX_VIDEO_FRAMES}-frame Label limit"
                )
            frame = _display_frame(raw_frame, rotation_degrees=rotation)
            frame_height, frame_width = frame.shape[:2]
            if not width:
                width, height = frame_width, frame_height
            elif (frame_width, frame_height) != (width, height):
                raise LabelPredictionError("capture frame dimensions changed during alignment scan")
            confidence = client.classify_alignment(frame)
            alignment_confs.append(
                round(
                    _number(
                        confidence,
                        field=f"alignment_confs[{len(alignment_confs)}]",
                        minimum=0,
                        maximum=1,
                    ),
                    3,
                )
            )
        if not alignment_confs or not width or not height:
            raise LabelPredictionError("capture video contains no readable frames")
    except LabelPredictionError:
        raise
    except (OnnxModelError, TrackerError, VisionRuntimeUnavailable) as exc:
        raise LabelPredictionError(f"native alignment scan failed: {exc}") from exc
    except Exception as exc:
        raise LabelPredictionError(f"native alignment scan failed ({type(exc).__name__})") from exc
    finally:
        try:
            capture.release()
        except Exception:
            pass
    aligned_frames = [
        frame_index
        for frame_index, confidence in enumerate(alignment_confs)
        if confidence >= LABEL_ALIGNMENT_THRESHOLD
    ]
    return {
        "schema": LABEL_ALIGNMENT_SCHEMA,
        "threshold": LABEL_ALIGNMENT_THRESHOLD,
        "model_profile": profile,
        "width": width,
        "height": height,
        "frame_count": len(alignment_confs),
        "aligned_frames": aligned_frames,
        "alignment_confs": alignment_confs,
    }


def run_label_prediction(
    settings: Settings,
    *,
    capture_id: str,
    video_path: Path,
    request: Any,
    rotation_degrees: int = 0,
) -> dict[str, Any]:
    capability = prediction_capability(settings)
    if not capability["enabled"]:
        raise LabelPredictionDisabled(str(capability["reason"]))
    if capability["backend"] == "camera-tracker-v1":
        return _run_native_label_prediction(
            settings,
            video_path=video_path,
            request=request,
            rotation_degrees=rotation_degrees,
        )
    return _run_external_label_prediction(
        settings,
        capture_id=capture_id,
        video_path=video_path,
        request=request,
    )
