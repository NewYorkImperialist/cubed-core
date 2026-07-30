"""Optional NVDEC video input for the GPU reads path.

Frames stay on the CUDA device through color conversion and resize. The
``verify_parity`` guard compares frame count and sampled pixels against the
OpenCV path before the result is used. The feature is off unless
``--gpu-decode`` or ``CUBED_GPU_DECODE=1`` is set.
"""

import os
import sys
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from core.perf_trace import get_trace

PERF_TRACE = get_trace("reads")

LUT_CACHE = os.environ.get("CUBED_SWS_LUT", "/tmp/cubed_swscale_bgr_lut.npy")
_LUT_SCHEMA = 1
_SUPPORTED_PIX_FMTS = {"yuv420p"}
_DEVICE_LUT_CACHE = {}

_DISPLAY_TRANSFORM_SCHEMA = 1
_DISPLAY_MATRIX_FP16_ONE = 1 << 16
_DISPLAY_MATRIX_FP30_ONE = 1 << 30
_DISPLAY_MATRIX_TOLERANCE = 2
_DISPLAY_TRANSFORM_NORMALIZATION = (
    "pure-cardinal-unit-transform; translation-normalized-to-zero-origin"
)
_NATIVE_PIXEL_AXIS = "native-coded-frame-pixel-xy"
_DISPLAY_PIXEL_AXIS = "post-container-display-transform-pixel-xy"
_DISPLAY_TRANSFORM_FIELDS = {
    "schema_version",
    "kind",
    "metadata_source",
    "decoded_pixel_axis",
    "display_pixel_axis",
    "native_frame_dimensions",
    "display_frame_dimensions",
    "container_rotation_degrees_counterclockwise",
    "applied_rotation_degrees_clockwise",
    "display_matrix_3x3",
    "legacy_rotate_tag_degrees_clockwise",
    "transform_normalization",
}


def _dimensions(value, *, field):
    try:
        width, height = value
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a (width, height) pair") from exc
    if isinstance(width, bool) or isinstance(height, bool):
        raise RuntimeError(f"{field} must contain positive integers")
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must contain positive integers") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{field} must contain positive integers")
    return width, height


def _cardinal_degrees(value, *, field):
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be a cardinal degree value")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a cardinal degree value") from exc
    if not np.isfinite(number):
        raise RuntimeError(f"{field} must be finite")
    normalized = number % 360.0
    cardinal = int(round(normalized / 90.0)) * 90 % 360
    distance = abs((normalized - cardinal + 180.0) % 360.0 - 180.0)
    if distance > 0.01:
        raise RuntimeError(f"{field} must normalize to 0/90/180/270")
    return cardinal


def display_frame_dimensions(native_dimensions, applied_rotation_degrees_clockwise):
    """Return the zero-origin pixel dimensions after a cardinal display rotation."""

    width, height = _dimensions(native_dimensions, field="native_frame_dimensions")
    rotation = _cardinal_degrees(
        applied_rotation_degrees_clockwise,
        field="applied_rotation_degrees_clockwise",
    )
    return (height, width) if rotation in (90, 270) else (width, height)


def _parse_display_matrix(value):
    if not isinstance(value, str):
        raise RuntimeError("ffprobe displaymatrix must be text")
    rows = []
    for line in value.splitlines():
        match = re.fullmatch(
            r"\s*[0-9A-Fa-f]+:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*",
            line,
        )
        if match is not None:
            rows.append(tuple(int(item) for item in match.groups()))
    if len(rows) != 3:
        raise RuntimeError("ffprobe displaymatrix is not a complete 3x3 matrix")
    return tuple(item for row in rows for item in row)


def _matrix_cardinal_counterclockwise(matrix):
    try:
        values = tuple(int(item) for item in matrix)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("display matrix must contain nine integers") from exc
    if len(values) != 9:
        raise RuntimeError("display matrix must contain nine integers")
    # FFmpeg's matrix is |a b u; c d v; x y w|. x/y are a harmless origin
    # translation after a pure rotation; u/v would introduce perspective.
    if (
        abs(values[2]) > _DISPLAY_MATRIX_TOLERANCE
        or abs(values[5]) > _DISPLAY_MATRIX_TOLERANCE
        or abs(values[8] - _DISPLAY_MATRIX_FP30_ONE) > _DISPLAY_MATRIX_TOLERANCE
    ):
        raise RuntimeError("display matrix contains perspective or a non-unit homogeneous scale")
    observed = (values[0], values[1], values[3], values[4])
    one = _DISPLAY_MATRIX_FP16_ONE
    patterns = {
        0: (one, 0, 0, one),
        90: (0, -one, one, 0),
        180: (-one, 0, 0, -one),
        270: (0, one, -one, 0),
    }
    matches = [
        degrees
        for degrees, expected in patterns.items()
        if all(
            abs(actual - wanted) <= _DISPLAY_MATRIX_TOLERANCE
            for actual, wanted in zip(observed, expected, strict=True)
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "display matrix is not a pure cardinal unit rotation "
            "(scale, reflection, or skew is unsupported)"
        )
    return matches[0], values


def _display_transform_contract(
    *,
    native_dimensions,
    container_rotation_degrees_counterclockwise,
    metadata_source,
    display_matrix=None,
    legacy_rotate_tag_degrees_clockwise=None,
):
    native = _dimensions(native_dimensions, field="native_frame_dimensions")
    counterclockwise = _cardinal_degrees(
        container_rotation_degrees_counterclockwise,
        field="container_rotation_degrees_counterclockwise",
    )
    clockwise = (-counterclockwise) % 360
    legacy = (
        None
        if legacy_rotate_tag_degrees_clockwise is None
        else _cardinal_degrees(
            legacy_rotate_tag_degrees_clockwise,
            field="legacy_rotate_tag_degrees_clockwise",
        )
    )
    payload = {
        "schema_version": _DISPLAY_TRANSFORM_SCHEMA,
        "kind": "cardinal-container-display-transform-v1",
        "metadata_source": metadata_source,
        "decoded_pixel_axis": _NATIVE_PIXEL_AXIS,
        "display_pixel_axis": _DISPLAY_PIXEL_AXIS,
        "native_frame_dimensions": list(native),
        "display_frame_dimensions": list(display_frame_dimensions(native, clockwise)),
        "container_rotation_degrees_counterclockwise": counterclockwise,
        "applied_rotation_degrees_clockwise": clockwise,
        "display_matrix_3x3": None if display_matrix is None else list(display_matrix),
        "legacy_rotate_tag_degrees_clockwise": legacy,
        "transform_normalization": _DISPLAY_TRANSFORM_NORMALIZATION,
    }
    return validate_display_transform_contract(payload, native_dimensions=native)


def identity_display_transform(native_dimensions):
    """Build the explicit no-container-transform contract used by mocked decoders."""

    return _display_transform_contract(
        native_dimensions=native_dimensions,
        container_rotation_degrees_counterclockwise=0,
        metadata_source="identity-no-container-display-metadata",
    )


def validate_display_transform_contract(value, *, native_dimensions=None):
    """Validate and canonicalize one fail-closed native-to-display axis contract."""

    if not isinstance(value, Mapping) or set(value) != _DISPLAY_TRANSFORM_FIELDS:
        raise RuntimeError("decoder display-transform contract fields are invalid")
    if value["schema_version"] != _DISPLAY_TRANSFORM_SCHEMA:
        raise RuntimeError("decoder display-transform schema version is unsupported")
    if value["kind"] != "cardinal-container-display-transform-v1":
        raise RuntimeError("decoder display-transform kind is unsupported")
    if (
        value["decoded_pixel_axis"] != _NATIVE_PIXEL_AXIS
        or value["display_pixel_axis"] != _DISPLAY_PIXEL_AXIS
    ):
        raise RuntimeError("decoder display-transform pixel axes are invalid")
    if value["transform_normalization"] != _DISPLAY_TRANSFORM_NORMALIZATION:
        raise RuntimeError("decoder display-transform normalization is invalid")
    native = _dimensions(value["native_frame_dimensions"], field="native_frame_dimensions")
    if native_dimensions is not None and native != _dimensions(
        native_dimensions, field="native_frame_dimensions"
    ):
        raise RuntimeError("decoder display-transform native dimensions disagree")
    counterclockwise = _cardinal_degrees(
        value["container_rotation_degrees_counterclockwise"],
        field="container_rotation_degrees_counterclockwise",
    )
    clockwise = _cardinal_degrees(
        value["applied_rotation_degrees_clockwise"],
        field="applied_rotation_degrees_clockwise",
    )
    if clockwise != (-counterclockwise) % 360:
        raise RuntimeError("decoder display-transform clockwise/counterclockwise angles disagree")
    display = _dimensions(value["display_frame_dimensions"], field="display_frame_dimensions")
    if display != display_frame_dimensions(native, clockwise):
        raise RuntimeError("decoder display-transform display dimensions disagree")
    legacy = value["legacy_rotate_tag_degrees_clockwise"]
    if legacy is not None:
        legacy = _cardinal_degrees(legacy, field="legacy_rotate_tag_degrees_clockwise")
        if legacy != clockwise:
            raise RuntimeError("legacy rotate tag conflicts with the display transform")
    matrix = value["display_matrix_3x3"]
    if matrix is not None:
        matrix_angle, matrix = _matrix_cardinal_counterclockwise(matrix)
        if matrix_angle != counterclockwise:
            raise RuntimeError("display matrix conflicts with its declared rotation")
    source = value["metadata_source"]
    allowed_sources = {
        "identity-no-container-display-metadata",
        "stream-side-data-display-matrix",
        "stream-side-data-rotation",
        "legacy-stream-tag-rotate",
    }
    if source not in allowed_sources:
        raise RuntimeError("decoder display-transform metadata source is unsupported")
    if source == "identity-no-container-display-metadata" and (
        clockwise != 0 or matrix is not None or legacy is not None
    ):
        raise RuntimeError("identity display-transform contract contains transform metadata")
    if source == "stream-side-data-display-matrix" and matrix is None:
        raise RuntimeError("display-matrix contract is missing its matrix")
    if source in {"stream-side-data-rotation", "legacy-stream-tag-rotate"} and matrix is not None:
        raise RuntimeError(
            "rotation-only display-transform contract unexpectedly contains a matrix"
        )
    if source == "legacy-stream-tag-rotate" and legacy is None:
        raise RuntimeError("legacy display-transform contract is missing its rotate tag")
    return {
        **dict(value),
        "native_frame_dimensions": list(native),
        "display_frame_dimensions": list(display),
        "container_rotation_degrees_counterclockwise": counterclockwise,
        "applied_rotation_degrees_clockwise": clockwise,
        "display_matrix_3x3": None if matrix is None else list(matrix),
        "legacy_rotate_tag_degrees_clockwise": legacy,
    }


def display_transform_from_ffprobe_payload(payload, *, native_dimensions):
    """Parse ffprobe JSON without reading pixels; reject non-cardinal transforms."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("ffprobe display metadata must be one JSON object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise RuntimeError("ffprobe must return exactly one selected video stream")
    stream = streams[0]
    native = _dimensions(native_dimensions, field="native_frame_dimensions")
    if (int(stream.get("width", -1)), int(stream.get("height", -1))) != native:
        raise RuntimeError("ffprobe display metadata dimensions disagree with the decoder")
    side_data = stream.get("side_data_list") or []
    if not isinstance(side_data, list) or any(not isinstance(item, Mapping) for item in side_data):
        raise RuntimeError("ffprobe stream side data is invalid")
    matrix_values = [item["displaymatrix"] for item in side_data if "displaymatrix" in item]
    rotation_values = [item["rotation"] for item in side_data if "rotation" in item]
    if len(matrix_values) > 1 or len(rotation_values) > 1:
        raise RuntimeError("ffprobe exposes multiple display transforms")
    tags = stream.get("tags") or {}
    if not isinstance(tags, Mapping):
        raise RuntimeError("ffprobe stream tags are invalid")
    legacy_raw = tags.get("rotate")
    legacy = (
        None if legacy_raw is None else _cardinal_degrees(legacy_raw, field="legacy rotate tag")
    )

    if matrix_values:
        matrix_angle, matrix = _matrix_cardinal_counterclockwise(
            _parse_display_matrix(matrix_values[0])
        )
        if rotation_values:
            side_angle = _cardinal_degrees(rotation_values[0], field="ffprobe side-data rotation")
            if side_angle != matrix_angle:
                raise RuntimeError("ffprobe rotation conflicts with its display matrix")
        return _display_transform_contract(
            native_dimensions=native,
            container_rotation_degrees_counterclockwise=matrix_angle,
            metadata_source="stream-side-data-display-matrix",
            display_matrix=matrix,
            legacy_rotate_tag_degrees_clockwise=legacy,
        )
    if rotation_values:
        side_angle = _cardinal_degrees(rotation_values[0], field="ffprobe side-data rotation")
        return _display_transform_contract(
            native_dimensions=native,
            container_rotation_degrees_counterclockwise=side_angle,
            metadata_source="stream-side-data-rotation",
            legacy_rotate_tag_degrees_clockwise=legacy,
        )
    if legacy is not None:
        return _display_transform_contract(
            native_dimensions=native,
            container_rotation_degrees_counterclockwise=(-legacy) % 360,
            metadata_source="legacy-stream-tag-rotate",
            legacy_rotate_tag_degrees_clockwise=legacy,
        )
    return identity_display_transform(native)


def _probe_display_transform(path, *, native_dimensions):
    def _command(include_side_data):
        entries = "stream=width,height:stream_tags=rotate"
        if include_side_data:
            entries += ":stream_side_data=rotation,displaymatrix"
        return [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            entries,
            "-of",
            "json",
            str(path),
        ]

    try:
        result = subprocess.run(
            _command(True),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        # ffprobe before version 5 rejects the stream_side_data section
        # selector outright. Rotation there is still exposed through the
        # rotate stream tag, so retry without side data.
        if result.returncode != 0 and "stream_side_data" in (result.stderr or ""):
            result = subprocess.run(
                _command(False),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("ffprobe could not inspect container display metadata") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[:300]
        raise RuntimeError(f"ffprobe could not inspect container display metadata: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid display-metadata JSON") from exc
    return display_transform_from_ffprobe_payload(payload, native_dimensions=native_dimensions)


def rotate_device_u8_clockwise(image, degrees):
    """Apply one cardinal clockwise rotation to a CUDA uint8 HW/HWC tensor."""

    if (
        not isinstance(image, torch.Tensor)
        or image.device.type != "cuda"
        or image.dtype != torch.uint8
        or image.ndim not in (2, 3)
    ):
        raise ValueError("display rotation requires a CUDA uint8 HW or HWC tensor")
    rotation = _cardinal_degrees(degrees, field="display rotation")
    turns = {0: 0, 90: -1, 180: 2, 270: 1}[rotation]
    return image if turns == 0 else torch.rot90(image, turns, dims=(0, 1)).contiguous()


def _video_contract(path):
    """Properties that can change libswscale's YUV->BGR answer.

    PyNvVideoCodec's StreamMetadata exposes dimensions/codec but not the color
    contract. PyAV reads the same container metadata without decoding frames;
    it is already required when a LUT cache miss has to be built.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        cc = stream.codec_context

        def enum_code(value):
            return None if value is None else int(value)

        contract = {
            "codec": str(cc.name),
            "width": int(cc.width),
            "height": int(cc.height),
            "pix_fmt": str(cc.pix_fmt),
            "color_range": enum_code(cc.color_range),
            "colorspace": enum_code(cc.colorspace),
            "color_primaries": enum_code(cc.color_primaries),
            "color_trc": enum_code(cc.color_trc),
        }
    contract["display_transform"] = _probe_display_transform(
        path,
        native_dimensions=(contract["width"], contract["height"]),
    )
    return contract


def _container_frame_count(path):
    """Return the container's declared frame count without decoding the video."""
    import av

    with av.open(path) as container:
        count = int(container.streams.video[0].frames or 0)
    if count <= 0:
        cap = cv2.VideoCapture(path)
        try:
            count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()
    if count <= 0:
        raise RuntimeError(f"--gpu-decode could not determine frame count for {path}")
    return count


def _lut_cache_path(base, contract):
    """Content-address the LUT cache by the complete stream color contract."""
    if not base:
        return None
    payload = {
        "schema": _LUT_SCHEMA,
        **{key: value for key, value in contract.items() if key != "display_transform"},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    p = Path(base)
    suffix = p.suffix or ".npy"
    stem = p.name[: -len(suffix)] if p.suffix else p.name
    return str(p.with_name(f"{stem}.{digest}{suffix}"))


def _validate_contract(contract):
    pix_fmt = contract.get("pix_fmt")
    if pix_fmt not in _SUPPORTED_PIX_FMTS:
        raise RuntimeError(
            "--gpu-decode only supports 8-bit planar 4:2:0 input "
            f"({_SUPPORTED_PIX_FMTS}); got pix_fmt={pix_fmt!r}. Refusing to "
            "guess the NVDEC plane layout/color conversion."
        )
    validate_display_transform_contract(
        contract.get("display_transform"),
        native_dimensions=(contract.get("width"), contract.get("height")),
    )


# ---------------------------------------------------------------------------
# The exact-swscale colour table (see module docstring)
# ---------------------------------------------------------------------------


def build_swscale_lut(cache=LUT_CACHE):
    """(2**24, 3) uint8 BGR indexed by (Y<<16)|(U<<8)|V — libswscale's OWN answer
    for every possible (Y,U,V), obtained by running swscale (via PyAV, the same
    libswscale cv2.VideoCapture links) on synthetic yuv420p frames whose chroma is
    constant down every column, so the vertical chroma filter is an identity and
    each output pixel is a pure function of the triple."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    import av  # only needed to BUILD the table

    W, H = 4096, 256  # 2048 chroma cols x 256 luma rows
    ch = W // 2
    lut = np.zeros((1 << 24, 3), np.uint8)
    ys = np.arange(256, dtype=np.uint8)[:, None].repeat(W, 1)  # Y = row index
    for f in range(65536 // ch):
        pair = np.arange(f * ch, (f + 1) * ch, dtype=np.uint32)
        us = (pair >> 8).astype(np.uint8)
        vs = (pair & 0xFF).astype(np.uint8)
        buf = np.concatenate(
            [
                ys.reshape(-1),
                np.tile(us, (H // 2, 1)).reshape(-1),
                np.tile(vs, (H // 2, 1)).reshape(-1),
            ]
        )
        fr = av.VideoFrame.from_ndarray(buf.reshape(H * 3 // 2, W), format="yuv420p")
        bgr = fr.reformat(format="bgr24").to_ndarray()
        idx = (
            (np.arange(256, dtype=np.uint32)[:, None] << 16)
            | (np.repeat(us.astype(np.uint32), 2)[None, :] << 8)
            | np.repeat(vs.astype(np.uint32), 2)[None, :]
        )
        lut[idx.reshape(-1)] = bgr.reshape(-1, 3)
    if cache:
        np.save(cache, lut)
    return lut


# ---------------------------------------------------------------------------
# cv2.resize(INTER_LINEAR) on uint8, reproduced EXACTLY (resize.cpp fixed-point)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _resize_coeffs_cached(src, dst, device_key):
    """cv2's xofs / ialpha for one axis: 11-bit fixed-point weights, computed from
    a float32 source coordinate exactly as resize.cpp's `fixedpt` branch does.

    The tensors depend only on the axis dimensions and target device. A reads
    pass repeats the same model resizes for every frame, so retain these
    immutable values instead of rebuilding and uploading them per frame.
    """
    device = torch.device(device_key)
    scale = src / dst
    fx = ((np.arange(dst, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    sx = np.floor(fx).astype(np.int64)
    fx = (fx - sx).astype(np.float32)
    fx[sx < 0] = 0
    sx[sx < 0] = 0
    fx[sx >= src - 1] = 0  # tail: single tap, weight 2048
    sx[sx >= src - 1] = src - 1
    a1 = np.rint(fx.astype(np.float64) * 2048).astype(np.int64)  # saturate_cast<short>
    a0 = np.rint((1.0 - fx).astype(np.float64) * 2048).astype(np.int64)
    t = lambda a: torch.as_tensor(a, device=device)
    return t(sx), t(np.minimum(sx + 1, src - 1)), t(a0), t(a1)


def _resize_coeffs(src, dst, device):
    """Return cached cv2 fixed-point resize coefficients for one axis."""
    return _resize_coeffs_cached(int(src), int(dst), str(torch.device(device)))


def resize_u8(img, new_w, new_h):
    """(H,W,C) uint8 cuda -> (new_h,new_w,C) uint8. Bit-identical to
    cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)."""
    H, W = img.shape[:2]
    xs, xs1, xa0, xa1 = _resize_coeffs(W, new_w, img.device)
    ys, ys1, ya0, ya1 = _resize_coeffs(H, new_h, img.device)
    src = img.int()
    # horizontal pass -> int rows:  S[sx]*a0 + S[sx+1]*a1
    h = src[:, xs, :] * xa0[None, :, None] + src[:, xs1, :] * xa1[None, :, None]
    # vertical pass: cv2's uchar specialisation
    #   dst = ((b0*(S0>>4))>>16) + ((b1*(S1>>4))>>16) + 2) >> 2
    out = (
        ((ya0[:, None, None] * (h[ys] >> 4)) >> 16)
        + ((ya1[:, None, None] * (h[ys1] >> 4)) >> 16)
        + 2
    ) >> 2
    return out.clamp(0, 255).to(torch.uint8)


def letterbox_u8_gpu(img, imgsz):
    """gpu_reads.letterbox_u8() on a device tensor: byte-identical `padded`,
    identical meta (this is preprocess_segment's cv2 half)."""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    r = resize_u8(img, new_w, new_h)
    pad_w, pad_h = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    out = torch.full((imgsz, imgsz, 3), 114, dtype=torch.uint8, device=img.device)
    out[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = r
    return out, {"scale": scale, "pad": (pad_w, pad_h), "orig_shape": (h, w), "imgsz": imgsz}


def preprocess_classify_gpu(img, imgsz):
    """detect.onnx_runtime.preprocess_classify() on a device tensor -> (1,3,S,S)
    float32. Resize short edge + centre-crop are the exact integer ops above; the
    BGR->RGB / uint8->float32 / *(1/255) tail is exact on either device."""
    h, w = img.shape[:2]
    scale = imgsz / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    r = resize_u8(img, new_w, new_h)
    y0, x0 = (new_h - imgsz) // 2, (new_w - imgsz) // 2
    c = r[y0 : y0 + imgsz, x0 : x0 + imgsz]
    return c[..., [2, 1, 0]].permute(2, 0, 1)[None].float().div_(255.0)


# ---------------------------------------------------------------------------


class GpuFrame:
    """One decoded frame, resident on the GPU. `.bgr` / `.gray` are device uint8
    tensors; `.host_bgr()` materialises the numpy frame ONLY if some consumer
    still needs cv2 (e.g. warp_crop without --gpu-warp) — and is bit-identical to
    what native-axis cv2.VideoCapture (orientation auto-apply disabled) would
    have handed back. ``presentation_timestamp_ticks``
    is raw source-container PTS metadata from NVDEC, not a pixel transfer; a
    strict consumer must bind the stream time base separately."""

    __slots__ = ("fi", "bgr", "gray", "presentation_timestamp_ticks", "_host")

    def __init__(self, fi, bgr, gray, presentation_timestamp_ticks=None):
        self.fi = fi
        self.bgr = bgr
        self.gray = gray
        self.presentation_timestamp_ticks = presentation_timestamp_ticks
        self._host = None

    def host_bgr(self):
        if self._host is None:
            with PERF_TRACE.span("reads.frame_d2h"):
                self._host = self.bgr.cpu().numpy()
            PERF_TRACE.increment("host_frame_downloads")
            PERF_TRACE.increment("host_frame_download_bytes", self._host.nbytes)
        return self._host


def _presentation_timestamp_ticks(frame):
    """Best-effort additive metadata for consumers that require a VFR axis.

    Older PyNvVideoCodec builds may expose ``getPTS()`` instead of the current
    ``timestamp`` property. Ordinary read consumers do not require timestamps;
    strict corpus producers validate that this result is present.
    """
    value = getattr(frame, "timestamp", None)
    if value is None:
        getter = getattr(frame, "getPTS", None)
        value = None if getter is None else getter()
    return None if value is None else int(value)


class GpuVideoDecoder:
    """NVDEC H.264 decode -> BGR + GRAY device tensors. Drop-in for the
    cv2.VideoCapture loop in geo_read.main(), same frames, same indices."""

    def __init__(self, path, device="cuda:0", verbose=True):
        import PyNvVideoCodec as nvc  # NVDEC binding (no onnxruntime dep)

        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-decode requires CUDA (torch.cuda unavailable)")
        self.path = path
        self.dev = torch.device(device)
        self.verbose = verbose
        gpu_id = self.dev.index
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()
        # SimpleDecoder crashes natively on some valid iPhone H.264 streams.
        # The packet API uses the same NVDEC engine without that fragile
        # high-level parser and preserves the original decode order and PTS.
        self._demux = nvc.CreateDemuxer(filename=path)
        self._dec = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=self._demux.GetNvCodecId(),
            usedevicememory=True,
        )
        self.n_frames = _container_frame_count(path)
        self.width = int(self._demux.Width())
        self.height = int(self._demux.Height())

        self.contract = _video_contract(path)
        _validate_contract(self.contract)
        if (self.contract["width"], self.contract["height"]) != (self.width, self.height):
            raise RuntimeError(
                "--gpu-decode stream metadata disagreement: PyAV reports "
                f"{self.contract['width']}x{self.contract['height']} while "
                f"PyNvVideoCodec reports {self.width}x{self.height}."
            )
        self.display_transform = validate_display_transform_contract(
            self.contract["display_transform"],
            native_dimensions=(self.width, self.height),
        )
        self.display_width, self.display_height = self.display_transform["display_frame_dimensions"]
        self.applied_rotation_degrees_clockwise = self.display_transform[
            "applied_rotation_degrees_clockwise"
        ]
        self.lut_cache = _lut_cache_path(LUT_CACHE, self.contract)
        # geo_read's startup parity guard constructs a decoder before the real
        # pass. Reuse its already-uploaded 66 MiB BGR/gray tables rather than
        # loading and transferring them a second time in the same process.
        device_key = (str(self.dev), self.lut_cache)
        cached = _DEVICE_LUT_CACHE.get(device_key)
        if cached is None:
            lut = build_swscale_lut(cache=self.lut_cache)
            cached = (
                torch.from_numpy(lut).to(self.dev),  # 50 MB
                torch.from_numpy(
                    cv2.cvtColor(lut.reshape(4096, 4096, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
                ).to(self.dev),  # 16 MB
            )
            _DEVICE_LUT_CACHE[device_key] = cached
        self.lut, self.glut = cached
        if verbose:
            print(
                f"  [gpu-decode] NVDEC {self.width}x{self.height} "
                f"native -> {self.display_width}x{self.display_height} display "
                f"(rotate={self.applied_rotation_degrees_clockwise}deg CW); "
                f"frames={self.n_frames} (cv2-exact swscale LUT; "
                f"contract={self.contract}; cache={self.lut_cache})",
                flush=True,
            )

    def __len__(self):
        return self.n_frames

    def _convert(self, fr):
        """NV12 (device) -> (BGR, GRAY) device uint8. Chroma replicated 2x2 and
        the colour math gathered from the swscale table = cv2's exact answer."""
        pl = fr.cuda()
        H, W = self.height, self.width
        y = torch.as_tensor(pl[0], device=self.dev)[..., 0][:H, :W]
        uv = torch.as_tensor(pl[1], device=self.dev)
        u = uv[..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1)[:H, :W]
        v = uv[..., 1].repeat_interleave(2, 0).repeat_interleave(2, 1)[:H, :W]
        idx = ((y.int() << 16) | (u.int() << 8) | v.int()).reshape(-1).long()
        return self.lut[idx].reshape(H, W, 3), self.glut[idx].reshape(H, W)

    def frames(self):
        """Yield GpuFrame(fi, bgr, gray) for fi = 0..n-1, in decode order — the
        same frames, in the same order, with the same absolute indices, that
        cv2.VideoCapture's read() loop produces."""
        fi = 0
        for packet in self._demux:
            if PERF_TRACE.enabled:
                with PERF_TRACE.span("reads.nvdec.decode_packet"):
                    decoded = self._dec.Decode(packet)
            else:
                decoded = self._dec.Decode(packet)
            for fr in decoded:
                if PERF_TRACE.enabled:
                    # Torch conversion kernels are asynchronous. This span is
                    # CPU submission time; frame_loop is authoritative wall.
                    with PERF_TRACE.span("reads.nvdec.convert_submit"):
                        bgr, gray = self._convert(fr)
                else:
                    bgr, gray = self._convert(fr)
                yield GpuFrame(fi, bgr, gray, _presentation_timestamp_ticks(fr))
                fi += 1


# ---------------------------------------------------------------------------
# Decode-drift guard: prove frame-count + pixel parity against cv2 on THIS video
# ---------------------------------------------------------------------------


def verify_parity(path, n=4, full_count=False, verbose=True):
    """Standing guard against the ffmpeg-pipe decode's two failure modes.

    That dead end died of (1) a silent frame-count shift (CFR padding) and (2) colour drift.
    Both are checked here, and a failure RAISES rather than writing shifted
    evidence — frame indices are load-bearing (raw[fi] / frec[fi] are keyed by
    absolute index).

      count : NVDEC's frame count vs cv2's. `full_count` decodes the whole video
              with cv2 to count it exactly (that costs a full CPU decode, so it is
              opt-in via --gpu-decode-verify); otherwise the container's count is
              cross-checked, which is free.
      pixels: the first `n` frames are decoded BOTH ways and compared byte for
              byte. Cheap (~n frames of cv2) and it catches any colour drift on
              the spot, so it always runs.
    """
    dec = GpuVideoDecoder(path, verbose=False)  # its own decoder: the caller's
    cap = cv2.VideoCapture(path)  # must stay unconsumed
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        # NVDEC yields coded/native pixels. Keep the parity reference on that same
        # axis; display rotation is a separate, receipted CUDA operation.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    ref = []
    ncv = 0
    if full_count:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if ncv < n:
                ref.append(f)
            ncv += 1
    else:
        ncv = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # container metadata
        while len(ref) < n:
            ok, f = cap.read()
            if not ok:
                break
            ref.append(f)
    cap.release()

    if ncv > 0 and dec.n_frames != ncv:
        raise RuntimeError(
            f"--gpu-decode frame-count parity failed: NVDEC {dec.n_frames} vs "
            f"cv2 {ncv} ({'decoded' if full_count else 'container'}) on {path}. "
            f"Frame indices are load-bearing. Refusing to run."
        )

    worst_mean = worst_max = 0.0
    nexact = 0
    for gf in dec.frames():
        if gf.fi >= len(ref):
            break
        if ref[gf.fi].shape != (dec.height, dec.width, 3):
            raise RuntimeError(
                "--gpu-decode cv2 parity reference is not on the native-coded "
                f"pixel axis: got {ref[gf.fi].shape}, expected "
                f"{(dec.height, dec.width, 3)}"
            )
        d = np.abs(gf.host_bgr().astype(np.int16) - ref[gf.fi].astype(np.int16))
        worst_mean = max(worst_mean, float(d.mean()))
        worst_max = max(worst_max, float(d.max()))
        nexact += int(d.max() == 0)
    if nexact != len(ref):
        raise RuntimeError(
            f"--gpu-decode pixel parity failed: only {nexact}/{len(ref)} frames "
            f"bit-identical to cv2 (mean|d|={worst_mean:.4f}, max|d|={worst_max:.0f})."
        )
    if verbose:
        print(
            f"  [gpu-decode] parity guard: frames NVDEC {dec.n_frames} == cv2 {ncv}"
            f"{' (fully decoded)' if full_count else ' (container)'}; first "
            f"{len(ref)} frames bit-identical to cv2 ({nexact}/{len(ref)}, "
            f"max|d|={worst_max:.0f})",
            flush=True,
        )
    return {
        "frames": dec.n_frames,
        "cv2_frames": ncv,
        "bit_identical": nexact,
        "n_checked": len(ref),
        "mean_abs": worst_mean,
        "max_abs": worst_max,
    }


if __name__ == "__main__":
    v = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(verify_parity(v, n))
