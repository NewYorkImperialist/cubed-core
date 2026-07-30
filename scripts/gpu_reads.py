"""Batched CUDA execution for the reads path in ``scripts/geo_read.py``.

The engine batches alignment and pose inference and keeps color sampling on the
device. OpenCV-compatible lookup tables preserve its quantized BGR-to-Lab
conversion. Batched model output and the optional ``--gpu-warp`` path are not
treated as byte-identical to single-frame OpenCV execution.

Enable with ``--gpu-reads`` or ``CUBED_GPU_READS=1``. An explicit
``--gpu-batch-size`` overrides the free-memory estimate. The caller halves the
batch and retries after a recognized CUDA allocation failure.
"""
import gc
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from detect.inference_client import ALIGNED_IDX, CLASSIFY_IMGSZ, FACE_IMGSZ
from detect.onnx_runtime import (postprocess_classification, postprocess_pose,
                                 preprocess_classify)
from cell_common import CROP, _DST, _SS, cell_centers
from occ_mask import C_SKIN, L_WHITE
from onnx_dynamic_batch import dynb_path, patch
from core.perf_trace import get_trace

PERF_TRACE = get_trace("reads")

HALF = 8            # masked_cell_labs' window half-width (must track occ_mask)
WIN = 2 * HALF      # 16 -> 256 px per cell window


class _PoseCompactUnavailable(RuntimeError):
    """The session cannot use caller-owned device output buffers."""


def _compact_pose_candidates(det, conf_thresh, nc=1, nk=4,
                             kpt_conf_thresh=0.5, min_visible_kpts=1):
    """Apply postprocess_pose's two admission gates without leaving device.

    The returned rows are ``frame_index + raw candidate``.  Flattening the
    ``(frame, candidate)`` mask before ``nonzero`` keeps the original frame and
    candidate order, including the tie order later observed by numpy's NMS.
    Class admission is deliberately applied first, then visible-keypoint
    admission, matching :func:`detect.onnx_runtime.postprocess_pose`.
    """
    if det.ndim != 3:
        raise _PoseCompactUnavailable(
            f"pose output must be rank 3, got shape={tuple(det.shape)}")
    batch, channels, n_candidates = det.shape
    expected_channels = 4 + nc + nk * 3
    if channels != expected_channels:
        raise _PoseCompactUnavailable(
            f"pose output has {channels} channels, expected {expected_channels}")

    preds = det.permute(0, 2, 1).contiguous()  # (B, N, channels)
    class_probs = preds[:, :, 4:4 + nc]
    max_probs = (class_probs[:, :, 0] if nc == 1
                 else class_probs.amax(dim=2))
    class_flat = torch.nonzero(
        (max_probs >= conf_thresh).reshape(-1), as_tuple=False).flatten()
    flat_preds = preds.reshape(batch * n_candidates, channels)
    admitted = flat_preds.index_select(0, class_flat)

    if kpt_conf_thresh > 0:
        kpts = admitted[:, 4 + nc:].reshape(-1, nk, 3)
        visible = (kpts[:, :, 2] >= kpt_conf_thresh).sum(dim=1)
    else:
        visible = torch.full(
            (admitted.shape[0],), nk, dtype=torch.int64, device=det.device)
    visible_keep = visible >= min_visible_kpts
    admitted = admitted[visible_keep]
    frame_ids = torch.div(
        class_flat, n_candidates, rounding_mode="floor")[visible_keep]

    # Pack the frame id into the same float tensor so one .cpu() below carries
    # both the candidates and their batch boundaries. Any feasible batch index
    # is far below float32's exact-integer limit (2**24).
    packed = admitted.new_empty((admitted.shape[0], channels + 1))
    packed[:, 0] = frame_ids
    packed[:, 1:] = admitted
    return packed


def _unpack_pose_candidates(packed, batch, channels):
    """Restore one postprocess_pose-shaped ndarray per input frame."""
    def empty():
        return np.empty((1, channels, 0), dtype=np.float32)
    if packed.shape != (len(packed), channels + 1):
        raise _PoseCompactUnavailable(
            f"bad compact pose shape={packed.shape}, channels={channels}")
    if len(packed) == 0:
        return [empty() for _ in range(batch)]

    frame_ids = packed[:, 0].astype(np.int64, copy=False)
    if (np.any(frame_ids < 0) or np.any(frame_ids >= batch)
            or np.any(frame_ids[1:] < frame_ids[:-1])):
        raise _PoseCompactUnavailable("compact pose frame order is invalid")
    values = packed[:, 1:]
    return [values[frame_ids == frame].T[None] if np.any(frame_ids == frame)
            else empty() for frame in range(batch)]


# ---------------------------------------------------------------------------
# Runtime batch sizing uses a conservative per-frame memory estimate, scales it
# up for larger native frames, and leaves half of currently free VRAM unused.
# An explicit --gpu-batch-size still overrides the estimate.
_REF_MIB_PER_FRAME = 25.0
_REF_PIXELS = 1080 * 1920
_MAX_AUTO_BATCH = 64


def derive_batch_size(free_bytes, h, w, safety=0.5):
    """Frames-per-flush that keeps one batch's GPU working set within
    `safety` of currently-free VRAM, scaled for native frame resolution.

    The estimate uses runtime memory and frame dimensions, not decode output.
    """
    px = max(1, h * w)
    mib_per_frame = _REF_MIB_PER_FRAME * max(1.0, px / _REF_PIXELS)
    budget_mib = (free_bytes / 2 ** 20) * safety
    batch = int(budget_mib // mib_per_frame)
    return max(1, min(_MAX_AUTO_BATCH, batch))


# ---------------------------------------------------------------------------
# OpenCV LAB lookup table
# ---------------------------------------------------------------------------

def _build_lab_lut(device):
    """(2**24, 3) uint8 LAB, indexed by (B<<16)|(G<<8)|R — cv2's own answer for
    every possible BGR pixel, so a GPU gather IS cv2.cvtColor(BGR2LAB)."""
    idx = np.arange(1 << 24, dtype=np.uint32)
    bgr = np.empty((1 << 24, 3), np.uint8)
    bgr[:, 0] = (idx >> 16) & 0xFF        # B
    bgr[:, 1] = (idx >> 8) & 0xFF         # G
    bgr[:, 2] = idx & 0xFF                # R
    lab = cv2.cvtColor(bgr.reshape(4096, 4096, 3), cv2.COLOR_BGR2LAB)
    return torch.from_numpy(lab.reshape(-1, 3).copy()).to(device)


def _cell_window_index():
    """Return flattened indices for each cell's sampling window."""
    cc = cell_centers().astype(int)                     # (9,2) as (x,y)
    off = np.arange(-HALF, HALF)
    idx = np.empty((9, WIN, WIN), np.int64)
    for i, (x, y) in enumerate(cc):
        ys = (y + off)[:, None]
        xs = (x + off)[None, :]
        idx[i] = ys * CROP + xs
    return idx.reshape(-1)


class GpuReadEngine:
    """Batched ONNX inference + GPU cell reads. One per process."""

    def __init__(self, batch=None, gpu_warp=False, device="cuda:0", verbose=True,
                 frame_hw=None):
        self._closed = False
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-reads requires CUDA (torch.cuda unavailable)")
        self.dev = torch.device(device)
        self.gpu_warp = gpu_warp
        self.verbose = verbose
        self.frame_hw = frame_hw
        # batch=None (the default, geo_read.py's --gpu-batch-size omitted) =>
        # derive it below, after the fixed session/LUT overhead is loaded, from
        # the VRAM that's ACTUALLY free at that moment. An explicit int is
        # still honored verbatim (never overridden).
        self._batch_explicit = batch is not None
        self.batch = max(1, int(batch)) if batch is not None else 1

        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()]
        if "CUDAExecutionProvider" not in prov:
            raise RuntimeError("--gpu-reads requires onnxruntime CUDAExecutionProvider")
        so = ort.SessionOptions()
        t = int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))
        if t > 0:
            so.intra_op_num_threads = t
            so.inter_op_num_threads = 1

        # Canonical align_thresh=0 runs never request alignment scores. Capture
        # the old construction-time configuration now, but defer both dynamic
        # model patching and session creation until a classify method is used.
        self._aligned_model = os.environ.get(
            "CUBED_ALIGNED_MODEL", "weights/cube_aligned.onnx")
        self._session_options = so
        self._providers = prov
        self.cls_sess = None
        self.cls_out = None
        self.pose_model_src = os.environ.get(
            "CUBED_FACE_POSE_MODEL", "weights/cube_face_pose.onnx")
        self.pose_model_loaded = self._dynb(self.pose_model_src)
        self.pose_sess = ort.InferenceSession(self.pose_model_loaded,
                                              sess_options=so, providers=prov)
        self.pose_out = [o.name for o in self.pose_sess.get_outputs()]
        self._pose_output_buffers = {}
        self._pose_compaction_enabled = True

        self.lab_lut = _build_lab_lut(self.dev)
        self.win_idx = torch.from_numpy(_cell_window_index()).to(self.dev)

        # Query free VRAM after fixed session and lookup-table allocation.
        free, total = torch.cuda.mem_get_info(self.dev)
        if not self._batch_explicit:
            h, w = frame_hw if frame_hw else (1920, 1080)   # unknown -> conservative default
            self.batch = derive_batch_size(free, h, w)
        if verbose:
            src = "explicit" if self._batch_explicit else (
                f"auto: free={free/2**20:.0f}MiB frame={frame_hw or '(unknown, assumed 1080x1920)'}")
            print(f"  [gpu-reads] batch={self.batch} ({src}) gpu_warp={gpu_warp} "
                  f"vram_free={free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)

    def empty_cache(self):
        """Release cached-but-unused CUDA blocks back to the driver. Called by
        geo_read.py's OOM guard before it retries a halved batch."""
        torch.cuda.empty_cache()

    def close(self):
        """Release sessions and CUDA storage owned by this engine.

        Cleanup is intentionally best-effort so a partially initialized engine
        can still be torn down after a construction or CUDA failure. Synchronize
        before severing references, then return allocator blocks only after
        Python and module-level cache owners have released their tensors.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True

        device = getattr(self, "dev", None)
        if device is not None:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass

        buffers = getattr(self, "_pose_output_buffers", None)
        if buffers is not None:
            try:
                buffers.clear()
            except Exception:
                pass

        for name in (
            "cls_sess",
            "cls_out",
            "pose_sess",
            "pose_out",
            "_pose_output_buffers",
            "lab_lut",
            "win_idx",
            "_session_options",
            "_providers",
        ):
            try:
                setattr(self, name, None)
            except Exception:
                pass

        try:
            _warp_target_grid_cached.cache_clear()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _dynb(src):
        """Build and return the dynamic-batch model path on first use."""
        dst = dynb_path(src)
        if not os.path.exists(dst):
            patch(src, dst)
        return dst

    def _ensure_classifier(self):
        """Create the optional alignment classifier on its first actual use."""
        if self.cls_sess is None:
            self.cls_sess = ort.InferenceSession(
                self._dynb(self._aligned_model),
                sess_options=self._session_options,
                providers=self._providers,
            )
            self.cls_out = [o.name for o in self.cls_sess.get_outputs()]
        return self.cls_sess, self.cls_out

    # -- ONNX, batched --------------------------------------------------------

    def _run_cuda(self, sess, out_names, x, *, presynced=False):
        """session.run on a torch CUDA tensor with ZERO host round-trip of the
        input (IOBinding takes the device pointer). Outputs come back on CPU —
        they are small (pose: 1.4 MB/frame) and postprocess_pose is numpy."""
        x = x.contiguous()
        if not presynced:
            torch.cuda.synchronize(self.dev)  # ORT runs on its own stream
        io = sess.io_binding()
        io.bind_input("images", "cuda", self.dev.index or 0, np.float32,
                      tuple(x.shape), x.data_ptr())
        for n in out_names:
            io.bind_output(n)                 # -> CPU
        sess.run_with_iobinding(io)
        out = io.copy_outputs_to_cpu()
        PERF_TRACE.increment(
            "ort_output_d2h_bytes", sum(int(value.nbytes) for value in out))
        PERF_TRACE.increment("ort_output_d2h_calls")
        return out

    @staticmethod
    def _pose_output_shape(sess, batch, channels):
        """Resolve the dynamic batch while requiring the known pose layout."""
        outputs = sess.get_outputs()
        if len(outputs) != 1:
            raise _PoseCompactUnavailable(
                f"pose compaction requires one output, got {len(outputs)}")
        meta = outputs[0]
        shape = getattr(meta, "shape", None)
        if shape is None or len(shape) != 3:
            raise _PoseCompactUnavailable("pose output shape metadata unavailable")
        if getattr(meta, "type", "tensor(float)") != "tensor(float)":
            raise _PoseCompactUnavailable(
                f"pose output must be float32, got {getattr(meta, 'type', None)}")
        tail = shape[1:]
        if not all(isinstance(value, (int, np.integer)) and value > 0
                   for value in tail):
            raise _PoseCompactUnavailable(
                f"pose output tail must be static, got {shape}")
        if int(tail[0]) != channels:
            raise _PoseCompactUnavailable(
                f"pose output has {tail[0]} channels, expected {channels}")
        fixed_batch = shape[0]
        if (isinstance(fixed_batch, (int, np.integer))
                and int(fixed_batch) != batch):
            raise _PoseCompactUnavailable(
                f"pose output batch is fixed at {fixed_batch}, requested {batch}")
        return batch, int(tail[0]), int(tail[1])

    def _run_pose_cuda_compact(self, x, *, conf, presynced=False, nc=1,
                               nk=4, kpt_conf_thresh=0.5,
                               min_visible_kpts=1):
        """Bind pose output on-device, admit there, then perform one compact D2H."""
        channels = 4 + nc + nk * 3
        shape = self._pose_output_shape(self.pose_sess, len(x), channels)
        if len(self.pose_out) != 1:
            raise _PoseCompactUnavailable(
                f"pose compaction requires one output name, got {self.pose_out}")

        x = x.contiguous()
        if not presynced:
            torch.cuda.synchronize(self.dev)  # ORT runs on its own stream
        buffers = getattr(self, "_pose_output_buffers", None)
        if buffers is None:
            buffers = self._pose_output_buffers = {}
        det = buffers.get(shape)
        if det is None or det.device != self.dev:
            det = torch.empty(shape, dtype=torch.float32, device=self.dev)
            buffers[shape] = det

        io = self.pose_sess.io_binding()
        device_id = self.dev.index or 0
        io.bind_input("images", "cuda", device_id, np.float32,
                      tuple(x.shape), x.data_ptr())
        io.bind_output(self.pose_out[0], "cuda", device_id, np.float32,
                       shape, det.data_ptr())
        with PERF_TRACE.span("reads.pose.ort_infer"):
            self.pose_sess.run_with_iobinding(io)

        with PERF_TRACE.span("reads.pose.compact_gpu"):
            packed = _compact_pose_candidates(
                det, conf, nc=nc, nk=nk, kpt_conf_thresh=kpt_conf_thresh,
                min_visible_kpts=min_visible_kpts)
            # Compact kernels are asynchronous. Synchronize only while tracing
            # so their work is attributed before the blocking D2H span.
            if PERF_TRACE.enabled:
                torch.cuda.synchronize(self.dev)
        raw_candidates = shape[0] * shape[2]
        raw_bytes = det.numel() * det.element_size()
        compact_candidates = packed.shape[0]
        compact_bytes = packed.numel() * packed.element_size()
        # This is the sole pose-output device-to-host transfer for the batch.
        with PERF_TRACE.span("reads.pose.compact_d2h"):
            host = packed.cpu().numpy()
        # "raw" is the full-output D2H this path replaces; "compact" is the
        # actual packed transfer (including one float frame id per survivor).
        PERF_TRACE.increment("pose_output_raw_candidates", raw_candidates)
        PERF_TRACE.increment("pose_output_compact_candidates", compact_candidates)
        PERF_TRACE.increment("pose_output_raw_d2h_bytes", raw_bytes)
        PERF_TRACE.increment("pose_output_compact_d2h_bytes", compact_bytes)
        PERF_TRACE.increment("pose_output_compact_d2h_calls")
        PERF_TRACE.increment("ort_output_d2h_bytes", compact_bytes)
        PERF_TRACE.increment("ort_output_d2h_calls")
        with PERF_TRACE.span("reads.pose.unpack_cpu"):
            return _unpack_pose_candidates(host, shape[0], shape[1])

    def _run_pose_outputs(self, x, *, conf, presynced=False):
        """Use compact CUDA output when compatible; otherwise retain old behavior."""
        if getattr(self, "_pose_compaction_enabled", True):
            try:
                return self._run_pose_cuda_compact(
                    x, conf=conf, presynced=presynced)
            except Exception as exc:
                oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
                if ((oom_type and isinstance(exc, oom_type))
                        or "out of memory" in str(exc).lower()):
                    raise
                # Nothing was published to the caller yet. Disable this path
                # for the session and transactionally re-run through the known
                # CPU-bound output binding used before this optimization.
                self._pose_compaction_enabled = False
                PERF_TRACE.increment("pose_output_compaction_fallbacks")
                if getattr(self, "verbose", False):
                    print("  [gpu-reads] pose CUDA-output compaction unavailable "
                          f"({type(exc).__name__}: {exc}); using CPU output",
                          flush=True)
        det = self._run_cuda(
            self.pose_sess, self.pose_out, x, presynced=presynced)[0]
        PERF_TRACE.increment("pose_output_raw_candidates",
                             int(det.shape[0] * det.shape[2]))
        PERF_TRACE.increment("pose_output_raw_d2h_bytes", int(det.nbytes))
        PERF_TRACE.increment("pose_output_fallback_d2h_bytes", int(det.nbytes))
        return [det[j:j + 1] for j in range(det.shape[0])]

    def _to_nchw(self, u8_hwc):
        """(B,H,W,3) uint8 BGR (host) -> (B,3,H,W) float32 RGB/255 on the GPU.

        Uploading uint8 input avoids transferring a float32 staging array.
        """
        t = torch.from_numpy(u8_hwc).to(self.dev, non_blocking=True)
        return t[..., [2, 1, 0]].permute(0, 3, 1, 2).float().div_(255.0)

    def classify_batch(self, frames):
        """Return alignment confidence for a batch of CPU frames."""
        if not frames:
            return []
        sess, out_names = self._ensure_classifier()
        with PERF_TRACE.span("reads.align.preprocess"):
            x = np.concatenate(
                [preprocess_classify(f, CLASSIFY_IMGSZ) for f in frames])
            tx = torch.from_numpy(x).to(self.dev)
            if PERF_TRACE.enabled:
                torch.cuda.synchronize(self.dev)
        with PERF_TRACE.span("reads.align.ort_d2h"):
            out = self._run_cuda(
                sess, out_names, tx,
                presynced=PERF_TRACE.enabled)[0]
        with PERF_TRACE.span("reads.align.post_cpu"):
            return [float(postprocess_classification(out[i:i + 1])[ALIGNED_IDX])
                    for i in range(len(frames))]

    def pose_batch(self, frames, conf=0.5):
        """Return pose detections for CPU frames using batched GPU inference."""
        if not frames:
            return []
        with PERF_TRACE.span("reads.pose.preprocess_cpu"):
            pads, metas = zip(*(letterbox_u8(f, FACE_IMGSZ) for f in frames))
        out = []
        for s in range(0, len(frames), self.batch):
            with PERF_TRACE.span("reads.pose.preprocess_gpu"):
                chunk = np.stack(pads[s:s + self.batch])
                x = self._to_nchw(chunk)
                if PERF_TRACE.enabled:
                    torch.cuda.synchronize(self.dev)
            with PERF_TRACE.span("reads.pose.ort_d2h"):
                dets = self._run_pose_outputs(
                    x, conf=conf, presynced=PERF_TRACE.enabled)
            with PERF_TRACE.span("reads.pose.post_cpu"):
                for j, det in enumerate(dets):
                    out.append(postprocess_pose(
                        det, metas[s + j], conf_thresh=conf,
                        nc=1, nk=4, kpt_conf_thresh=0.5,
                        snap_corners=True))
        return out

    # -- same two models, fed frames that are ALREADY on the GPU -------------
    #
    # These are the --gpu-decode entry points. NVDEC frames stay on device, so
    # preprocessing uses the corresponding device implementations.

    def classify_batch_gpu(self, gframes):
        """classify_batch() for GpuFrames — preprocessing on-device."""
        if not gframes:
            return []
        sess, out_names = self._ensure_classifier()
        from gpu_decode import preprocess_classify_gpu
        with PERF_TRACE.span("reads.align.preprocess_gpu"):
            x = torch.cat([preprocess_classify_gpu(g.bgr, CLASSIFY_IMGSZ)
                           for g in gframes])
            if PERF_TRACE.enabled:
                torch.cuda.synchronize(self.dev)
        with PERF_TRACE.span("reads.align.ort_d2h"):
            out = self._run_cuda(
                sess, out_names, x,
                presynced=PERF_TRACE.enabled)[0]
        with PERF_TRACE.span("reads.align.post_cpu"):
            return [float(postprocess_classification(out[i:i + 1])[ALIGNED_IDX])
                    for i in range(len(gframes))]

    def pose_batch_gpu(self, gframes, conf=0.5):
        """Run pose batching for GpuFrames with on-device letterboxing."""
        if not gframes:
            return []
        from gpu_decode import letterbox_u8_gpu
        with PERF_TRACE.span("reads.pose.preprocess_gpu"):
            pads, metas = zip(
                *(letterbox_u8_gpu(g.bgr, FACE_IMGSZ) for g in gframes))
        out = []
        for s in range(0, len(gframes), self.batch):
            with PERF_TRACE.span("reads.pose.preprocess_gpu"):
                chunk = torch.stack(
                    pads[s:s + self.batch])       # (B,S,S,3) uint8
                x = (chunk[..., [2, 1, 0]].permute(0, 3, 1, 2)
                     .float().div_(255.0))
                if PERF_TRACE.enabled:
                    torch.cuda.synchronize(self.dev)
            with PERF_TRACE.span("reads.pose.ort_d2h"):
                dets = self._run_pose_outputs(
                    x, conf=conf, presynced=PERF_TRACE.enabled)
            with PERF_TRACE.span("reads.pose.post_cpu"):
                for j, det in enumerate(dets):
                    out.append(postprocess_pose(
                        det, metas[s + j], conf_thresh=conf,
                        nc=1, nk=4, kpt_conf_thresh=0.5,
                        snap_corners=True))
        return out

    # -- cell reads, on the GPU ----------------------------------------------

    def cell_labs_batch(self, crops):
        """(F,9,3) float64 LAB + (F,9) float64 conf for F CROPxCROP BGR crops —
        batched over every face in the batch.

        LAB values come from the OpenCV lookup table. A cell with fewer than
        eight unmasked pixels falls back to the median of its full window.
        """
        if len(crops) == 0:
            return np.zeros((0, 9, 3)), np.zeros((0, 9))
        # Host crops are uploaded; device crops stay resident.
        if torch.is_tensor(crops[0]):
            t = torch.stack(list(crops)).contiguous()
        else:
            t = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).to(self.dev)
        F = t.shape[0]
        key = (t[..., 0].int() << 16) | (t[..., 1].int() << 8) | t[..., 2].int()
        lab = self.lab_lut[key.reshape(F, -1).long()].float()          # (F, CROP*CROP, 3)

        chroma = torch.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        skin = (chroma < C_SKIN) & (lab[..., 0] < L_WHITE)              # (F, CROP*CROP)

        win = lab[:, self.win_idx].reshape(F, 9, WIN * WIN, 3)          # (F,9,256,3)
        msk = skin[:, self.win_idx].reshape(F, 9, WIN * WIN)            # (F,9,256)

        n_skin = msk.sum(-1)                                            # (F,9)
        k = (WIN * WIN) - n_skin                                        # unmasked count
        conf = 1.0 - n_skin.double() / (WIN * WIN)                      # == 1 - m.mean()

        s_val = win.masked_fill(msk.unsqueeze(-1), float("inf")).sort(dim=2).values
        s_all = win.sort(dim=2).values
        med = torch.where((k >= 8).unsqueeze(-1),
                          _median_of_first_k(s_val, k),
                          _median_of_first_k(s_all, torch.full_like(k, WIN * WIN)))
        med_out = med.double().cpu().numpy()
        conf_out = conf.cpu().numpy()
        PERF_TRACE.increment(
            "cell_output_d2h_bytes", int(med_out.nbytes + conf_out.nbytes))
        PERF_TRACE.increment("cell_output_d2h_calls")
        return med_out, conf_out

    def warp_batch(self, frame, quads, keep_gpu=False):
        """Warp one frame's quads with Torch bilinear sampling.

        This path may differ numerically from OpenCV. It supersamples by
        ``_SS`` and box-averages to the maintained crop shape.
        """
        if not quads:
            return []
        big = CROP * _SS
        # `frame` may be a host numpy frame (cv2 path) or an on-device uint8
        # tensor (--gpu-decode) — in the latter case there is nothing to upload.
        t = frame if torch.is_tensor(frame) else torch.from_numpy(frame).to(self.dev)
        img = t.permute(2, 0, 1)[None].float()
        h, w = frame.shape[:2]
        dst = _warp_target_grid(self.dev)

        grids = []
        for q in quads:
            M = cv2.getPerspectiveTransform(_DST * _SS, np.asarray(q, np.float32))
            Mt = torch.from_numpy(M).to(self.dev)                        # crop -> image
            PERF_TRACE.increment("homography_uploads")
            PERF_TRACE.increment("homography_upload_bytes", int(M.nbytes))
            p = dst @ Mt.T
            src = p[:, :2] / p[:, 2:3]
            gx = src[:, 0] / (w - 1) * 2 - 1
            gy = src[:, 1] / (h - 1) * 2 - 1
            grids.append(torch.stack([gx, gy], -1).reshape(big, big, 2))
        grid = torch.stack(grids).float()                               # (F,big,big,2)
        s = torch.nn.functional.grid_sample(
            img.expand(len(quads), -1, -1, -1), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True)                   # (F,3,big,big)
        # INTER_AREA down by the integer factor _SS == an exact box mean
        s = torch.nn.functional.avg_pool2d(s, _SS)
        out = s.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        if keep_gpu:                    # --gpu-decode: hand cell_labs_batch the
            return [c for c in out]     # crops on-device; nothing crosses PCIe
        return [c for c in out.cpu().numpy()]


@lru_cache(maxsize=8)
def _warp_target_grid_cached(device_key):
    """Homogeneous CROP*_SS target pixels, invariant across frames/quads."""
    device = torch.device(device_key)
    big = CROP * _SS
    ys, xs = torch.meshgrid(
        torch.arange(big, device=device, dtype=torch.float64),
        torch.arange(big, device=device, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)


def _warp_target_grid(device):
    """Return the cached constant warp target grid for `device`."""
    return _warp_target_grid_cached(str(torch.device(device)))


def _median_of_first_k(s, k):
    """Median of the first ``k`` sorted rows of ``s`` with shape ``(F,9,N,3)``."""
    kk = k.clamp(min=1)
    hi = (kk // 2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    lo = ((kk - 1) // 2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    a = torch.gather(s, 2, lo).squeeze(2)
    b = torch.gather(s, 2, hi).squeeze(2)
    return torch.where((kk % 2 == 1).unsqueeze(-1), b, (a + b) / 2.0)


def letterbox_u8(img, imgsz):
    """Letterbox with OpenCV and retain uint8 pixels for GPU conversion."""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, imgsz - new_h - pad_h,
                                pad_w, imgsz - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, {"scale": scale, "pad": (pad_w, pad_h),
                    "orig_shape": (h, w), "imgsz": imgsz}
