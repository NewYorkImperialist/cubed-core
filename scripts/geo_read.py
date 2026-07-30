"""Geometric grid reader for a stickerless cube.

The reader warps each detected face quad to a known 3×3 grid, samples the nine
cell centers, and classifies them against the supplied calibration. Per-cell
confidence lets the trellis down-weight occluded or unsupported samples.

Output (pickle, trellis-compatible):  (raw, frec)
  raw  : {frame: (read, motion)}   read = [(slot, lab9[9,3]), ...]
  frec : {frame: {bbox, faces, stk, motion, vis{slot: area_ratio, _cellconf{slot:[9]}}}}

Run through ``scripts/run_research_reads.sh`` so the video, calibration, and
output paths are explicit.
"""
import sys, json, pickle, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, cv2
try:
    import torch                    # only needed by --gpu-reads / --gpu-decode
except ImportError:                 # CPU-only box: the default path never uses it
    torch = None

from detect.inference_client import get_inference_client
from calib_util import load_centroids
from cell_common import order_quad, warp_crop
from occ_mask import masked_cell_labs
from core.perf_trace import get_trace

PERF_TRACE = get_trace("reads")
_LEGACY_CLIENT = None


def _get_legacy_client():
    """Construct the per-frame CPU inference stack only when it is consumed.

    The batched GPU path owns its pose/alignment sessions and must not also
    reserve memory for the legacy aligned/pose/sticker sessions.  CPU callers
    retain the old process-wide singleton behavior after their first use.
    """
    global _LEGACY_CLIENT
    if _LEGACY_CLIENT is None:
        with PERF_TRACE.span("reads.legacy_client_init"):
            _LEGACY_CLIENT = get_inference_client()
    return _LEGACY_CLIENT


def argval(flag, d=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


def _is_cuda_oom(e):
    """Return whether an exception is a recognized CUDA/ORT allocation failure."""
    msg = str(e).lower()
    return any(s in msg for s in (
        "out of memory", "failed to allocate", "cuda_error_out_of_memory",
        "cudnn_status_alloc_failed", "cublas_status_alloc_failed", "cuda error"))


def qarea(q):
    x, y = np.asarray(q, float)[:, 0], np.asarray(q, float)[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def slot_names(faces):
    """Name visible faces by their relative image position."""

    items = [
        (
            np.asarray(face["corners"], float)[:, 1].mean(),
            np.asarray(face["corners"], float)[:, 0].mean(),
            face,
        )
        for face in faces
    ]
    items.sort(key=lambda item: item[0])
    out = [("up", items[0][2])]
    rest = sorted(items[1:], key=lambda item: item[1])
    if rest:
        out.append(("front", rest[0][2]))
    if len(rest) > 1:
        out.append(("right", rest[1][2]))
    return out


def _rounded_quad(value):
    """Browser-safe display-pixel quad without NumPy scalar leakage."""

    quad = np.asarray(value, float)
    return [[round(max(0.0, float(x)), 3), round(max(0.0, float(y)), 3)]
            for x, y in quad]


def _workstation_face_evidence(slotted_faces, labs, confs):
    """Project already-computed pose/read values into the portable viewer shape."""

    if not slotted_faces:
        return {"faces": [], "reads": []}
    areas = {name: qarea(face["corners"]) for name, face in slotted_faces}
    largest = max(areas.values()) or 1.0
    faces_out, reads_out = [], []
    for index, (name, face) in enumerate(slotted_faces):
        keypoint = np.asarray(face.get("kpt_conf", [0.0] * 4), float).reshape(-1)
        if len(keypoint) != 4:
            keypoint = np.zeros(4, dtype=float)
        face_confidence = float(face.get("confidence", 0.0))
        ordered = order_quad(np.asarray(face["corners"], np.float32))
        faces_out.append({
            "corners": _rounded_quad(ordered),
            "confidence": round(min(1.0, max(0.0, face_confidence)), 4),
            "keypoint_confidence": [
                round(min(1.0, max(0.0, float(value))), 4)
                for value in keypoint
            ],
        })
        reads_out.append({
            "slot": name,
            "lab": [
                [round(float(component), 3) for component in np.asarray(sample).reshape(-1)]
                for sample in labs[index]
            ],
            "confidence": [
                round(min(1.0, max(0.0, float(value))), 4)
                for value in confs[index]
            ],
            "relative_area": round(float(areas[name] / largest), 4),
            "corners": _rounded_quad(ordered),
        })
    return {"faces": faces_out, "reads": reads_out}


def build_workstation_stage(frec, geometry, *, fps, frame_count, width, height):
    """Build the path-free same-pass projection consumed by the native runner."""

    frames = {}
    for frame in sorted(frec):
        record = frec[frame]
        entry = {
            "motion": record.get("motion"),
            "face_count": int(record.get("faces") or 0),
        }
        evidence = geometry.get(frame)
        if evidence is not None:
            entry.update(evidence)
        frames[str(int(frame))] = entry
    resolved_count = int(frame_count) if int(frame_count or 0) > 0 else (
        max((int(frame) for frame in frec), default=-1) + 1
    )
    return {
        "schema": "cubed-core/decode-workstation-v1",
        "schema_version": 1,
        "video": {
            "fps": float(fps),
            "frame_count": resolved_count,
            "width": int(width),
            "height": int(height),
        },
        "window": [0, max(0, resolved_count - 1)],
        "warnings": [],
        "frames": frames,
    }


def write_workstation_stage(path, value):
    """Atomically write one compact internal sidecar beside scratch artifacts."""

    destination = os.path.abspath(path)
    temporary = f"{destination}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


class GeoReadStage:
    """Per-frame state for one reads extraction pass."""

    def __init__(self, cal, conf=0.5, occ=40.0,
                 align_thresh=0.0, mgate=False, move_thr=3.0, n_rest=8,
                 tag=None, verbose=True, gpu=None, gpu_batch=None, gpu_warp=False,
                 gpu_decode=False, gpu_frame_hw=None, workstation=False):
        self.tag = tag
        self.gpu_decode = gpu_decode
        self.conf = conf              # face-detect conf (geometry tolerant)
        self.occ = occ                # cell conf = max(0, 1 - dist/occ)
        self.align_thresh = align_thresh
        self.mgate = mgate
        self.move_thr = move_thr
        self.n_rest = n_rest
        self.verbose = verbose

        self.base = load_centroids(cal)
        self.cents = [np.asarray(v, float) for v in self.base.values()]
        if align_thresh > 0 and verbose:
            print(f"  [align-gate] thr={align_thresh} (frames below skip)",
                  flush=True)
        if mgate and verbose:
            print(f"  [motion-gate] move_thr={move_thr} n_rest={n_rest}", flush=True)

        self.raw, self.frec = {}, {}
        self.workstation_geometry = {} if workstation else None
        self.workstation_warning = None
        self.prev_gray = None
        self.t0 = time.perf_counter()
        self.n_align_dropped = 0
        self.rest_reads = 0
        self.n_motion_skip = 0

        # GPU mode batches model and cell work. The CPU path remains the
        # fallback when GPU mode is disabled.
        #
        # --motion-gate keeps this engine enabled.  Its cheap central-frame
        # motion values are reduced as one batch, then the existing rest_reads
        # state machine is folded in strict frame order after pose/cell work.
        # This deliberately computes and discards model results for gated
        # frames: existing gate decisions take priority over pre-filtering
        # because rest_reads depends on earlier pose hits.
        self.gpu = None
        self.buf = []
        if gpu:
            from gpu_reads import GpuReadEngine
            # gpu_batch=None (--gpu-batch-size omitted) => GpuReadEngine
            # derives it at construction time from free VRAM + this
            # video's frame size; an explicit value overrides verbatim.
            self.gpu = GpuReadEngine(batch=gpu_batch, gpu_warp=gpu_warp,
                                     verbose=verbose, frame_hw=gpu_frame_hw)

    @staticmethod
    def _bbox_motion(gray, prev_gray, bbox):
        """Mean |Δ| over the face bbox between consecutive frames, or None.

        The host path uses NumPy arrays; the GPU path sums integer differences
        on device and transfers one scalar.
        """
        g1 = gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        g0 = prev_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if g1.shape != g0.shape or 0 in tuple(g1.shape):
            return None
        if torch is not None and torch.is_tensor(g1):
            d = (g1.long() - g0.long()).abs()
            with PERF_TRACE.span("reads.motion_item_sync"):
                total = d.sum().item()
            PERF_TRACE.increment("motion_item_syncs")
            return float(total) / d.numel()
        return float(np.abs(g1.astype(int) - g0.astype(int)).mean())

    def _central_motion_batch(self, grays):
        """Return the legacy motion-gate mean for every gray frame.

        The serial gate compares each frame's central half with the immediately
        preceding frame.  Under NVDEC those grays are CUDA tensors; reducing
        them one at a time with ``.item()`` would serialize the GPU once per
        frame.  Stack equal-shaped crops, reduce on device, and transfer one
        vector per shape/device group instead.
        """
        means = [None] * len(grays)
        gpu_groups = {}
        prev = self.prev_gray
        for i, gray in enumerate(grays):
            if prev is not None and tuple(prev.shape) == tuple(gray.shape):
                h, w = gray.shape
                g1 = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
                g0 = prev[h // 4:3 * h // 4, w // 4:3 * w // 4]
                if torch is not None and torch.is_tensor(g1):
                    if not torch.is_tensor(g0):
                        raise TypeError("motion-gate gray backends changed mid-stream")
                    key = (str(g1.device), tuple(g1.shape))
                    gpu_groups.setdefault(key, []).append((i, g1, g0))
                else:
                    means[i] = float(np.abs(
                        g1.astype(np.int16) - g0.astype(np.int16)).mean())
            prev = gray

        for entries in gpu_groups.values():
            with PERF_TRACE.span("reads.motion_gate_batch"):
                current = torch.stack([entry[1] for entry in entries]).to(torch.int16)
                previous = torch.stack([entry[2] for entry in entries]).to(torch.int16)
                totals = (current - previous).abs().flatten(1).sum(
                    dim=1, dtype=torch.int64)
            with PERF_TRACE.span("reads.motion_gate_batch_d2h"):
                host_totals = totals.cpu().numpy()
            PERF_TRACE.increment("motion_gate_batch_d2h_calls")
            PERF_TRACE.increment("motion_gate_batch_d2h_items", len(entries))
            count = entries[0][1].numel()
            for (i, _g1, _g0), total in zip(entries, host_totals):
                means[i] = float(int(total)) / count
        return means

    def _emit_face_reads(self, fi, gray, prev_gray, faces, labs, confs):
        """Assemble one frame's rec/raw from its faces + already-computed cell
        reads. Pure code-motion of the tail of process_frame() so the batched
        path and the per-frame path share ONE definition of the record."""
        rec = {"bbox": None, "faces": 0, "stk": 0, "motion": None, "vis": {}}
        if faces:
            slotted_faces = list(slot_names(faces))
            pts = np.vstack([fc["corners"] for fc in faces])
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            bbox = (max(0, int(x0)), max(0, int(y0)), int(x1), int(y1))
            rec["bbox"] = list(bbox)
            read, cellconf, areas = [], {}, {}
            for j, (name, fc) in enumerate(slotted_faces):
                read.append((name, [labs[j][i] for i in range(9)]))
                cellconf[name] = [float(c) for c in confs[j]]
                areas[name] = qarea(fc["corners"])
            amax = max(areas.values()) or 1.0
            vis = {n: round(areas[n] / amax, 3) for n in areas}
            vis["_cellconf"] = cellconf
            rec["vis"] = vis
            rec["faces"] = len(read)
            rec["stk"] = sum(sum(1 for x in cc if x > 0.3) for cc in cellconf.values())
            if self.workstation_geometry is not None:
                try:
                    self.workstation_geometry[fi] = _workstation_face_evidence(
                        slotted_faces,
                        labs,
                        confs,
                    )
                except Exception as error:
                    # Viewer projection must never change the camera evidence
                    # consumed by the decoder. Disable only the optional
                    # geometry stream and let the reads pickle continue.
                    self.workstation_geometry = None
                    self.workstation_warning = (
                        f"overlay projection unavailable: {type(error).__name__}"
                    )
            if prev_gray is not None:
                m = self._bbox_motion(gray, prev_gray, bbox)
                if m is not None:
                    rec["motion"] = round(m, 1)
                    self.raw[fi] = (read, m)
                    self.rest_reads += 1
        self.frec[fi] = rec

    def _flush(self):
        """Drain self.buf through the GPU engine (with the OOM guard below)."""
        buf, self.buf = self.buf, []
        if not buf:
            return
        self._flush_with_retry(buf)

    def _flush_with_retry(self, buf):
        """Run `buf` through _flush_batch; on a CUDA/ORT allocator failure,
        halve it and retry the two halves. Also shrink ``self.gpu.batch`` so
        later flushes do not immediately hit the same allocation failure."""
        # `_flush_batch` performs all model/cell work first, then folds records
        # in frame order. A late CUDA failure in that fold can still leave a
        # prefix committed, so retry must be transactional with respect to the
        # sequential motion state.
        prev_gray = self.prev_gray
        rest_reads = self.rest_reads
        n_align_dropped = self.n_align_dropped
        n_motion_skip = self.n_motion_skip
        retry_msg = None
        try:
            self._flush_batch(buf)
        except Exception as e:
            if len(buf) <= 1 or not _is_cuda_oom(e):
                raise
            retry_msg = str(e)
            self.prev_gray = prev_gray
            self.rest_reads = rest_reads
            self.n_align_dropped = n_align_dropped
            self.n_motion_skip = n_motion_skip
            for fi, _ in buf:
                self.raw.pop(fi, None)
                self.frec.pop(fi, None)
                if self.workstation_geometry is not None:
                    self.workstation_geometry.pop(fi, None)

        # Leave the except block before releasing allocator caches: Python has
        # now cleared the exception variable/traceback that can retain failed
        # CUDA temporaries. Then persist the learned smaller batch for this and
        # all later flushes.
        if retry_msg is not None:
            new_batch = max(1, len(buf) // 2)
            print(f"  [gpu-reads] OOM at batch={len(buf)} ({retry_msg}) -- "
                  f"halving to <={new_batch} and retrying", flush=True)
            self.gpu.empty_cache()
            self.gpu.batch = min(self.gpu.batch, new_batch)
            mid = max(1, len(buf) // 2)
            self._flush_with_retry(buf[:mid])
            self._flush_with_retry(buf[mid:])

    def _flush_batch(self, buf):
        """Run one buffered batch through the GPU engine, then fold the results
        back in frame order. Motion bookkeeping remains sequential even though
        model and cell computation is batched.
        """
        frames = [f for _, f in buf]
        # --gpu-decode: `frames` are GpuFrames (scripts/gpu_decode.py) that were
        # born in NVDEC and stay on device. Gray conversion and model input use
        # their device implementations.
        gd = self.gpu_decode
        with PERF_TRACE.span("reads.batch.gray"):
            grays = ([f.gray for f in frames] if gd
                     else [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        gate_motion = (self._central_motion_batch(grays) if self.mgate
                       else [None] * len(grays))

        # 1) alignment classifier: batched, on EVERY frame (it is the cheap gate)
        if self.align_thresh > 0:
            with PERF_TRACE.span("reads.batch.alignment"):
                acs = (self.gpu.classify_batch_gpu(frames) if gd
                       else self.gpu.classify_batch(frames))
            surv = [i for i, ac in enumerate(acs) if ac >= self.align_thresh]
        else:
            acs = [None] * len(frames)
            surv = list(range(len(frames)))

        # 2) pose: batched only on frames that passed the optional reads-stage
        #    gate. A zero threshold keeps classifier-rejected frames eligible.
        faces_by_i = {}
        if surv:
            sub = [frames[i] for i in surv]
            with PERF_TRACE.span("reads.batch.pose"):
                got = (self.gpu.pose_batch_gpu(sub, conf=self.conf) if gd
                       else self.gpu.pose_batch(sub, conf=self.conf))
            faces_by_i = dict(zip(surv, got))

        # 3) warp every face in the batch, then ONE GPU call for all their cells
        crops, owner = [], []
        with PERF_TRACE.span("reads.batch.warp"):
            for i in surv:
                faces = faces_by_i[i]
                if not faces:
                    continue
                quads = [order_quad(np.asarray(fc["corners"], np.float32))
                         for _, fc in slot_names(faces)]
                if self.gpu.gpu_warp:
                    # under --gpu-decode the source frame is already on the card and
                    # the crops stay there too — zero host frame traffic in the pass
                    src = frames[i].bgr if gd else frames[i]
                    cs = self.gpu.warp_batch(src, quads, keep_gpu=gd)
                else:
                    # With GPU warp disabled, pull back only frames that carry
                    # a face and use the maintained OpenCV warp.
                    src = frames[i].host_bgr() if gd else frames[i]
                    cs = [warp_crop(src, q, ordered=True) for q in quads]
                for c in cs:
                    crops.append(c)
                    owner.append(i)
        with PERF_TRACE.span("reads.batch.cell_sample_d2h"):
            labs, confs = self.gpu.cell_labs_batch(crops)
        per_frame = {}
        for k, i in enumerate(owner):
            per_frame.setdefault(i, ([], []))
            per_frame[i][0].append(labs[k])
            per_frame[i][1].append(confs[k])

        # 4) serial fold preserves record structure and frame order.
        with PERF_TRACE.span("reads.batch.motion_record_fold"):
            for i, (fi, _) in enumerate(buf):
                if self.verbose and fi % 300 == 0 and fi:
                    r = fi / (time.perf_counter() - self.t0)
                    print(f"frame {fi} reads={len(self.raw)} {r:.1f}f/s "
                          f"align_dropped={self.n_align_dropped} "
                          f"motion_skip={self.n_motion_skip}", flush=True)
                cm = gate_motion[i]
                if self.mgate and cm is not None:
                    if cm >= self.move_thr:
                        self.rest_reads = 0
                    if cm >= self.move_thr or self.rest_reads >= self.n_rest:
                        self.n_motion_skip += 1
                        self.frec[fi] = {
                            "bbox": None, "faces": 0, "stk": 0,
                            "motion": round(cm, 1), "vis": {},
                        }
                        self.prev_gray = grays[i]
                        continue
                if i not in faces_by_i:
                    self.n_align_dropped += 1
                    self.frec[fi] = {"bbox": None, "faces": 0, "stk": 0,
                                     "motion": None, "vis": {},
                                     "align_conf": acs[i]}
                else:
                    l, c = per_frame.get(i, ([], []))
                    self._emit_face_reads(fi, grays[i], self.prev_gray,
                                          faces_by_i[i], l, c)
                self.prev_gray = grays[i]

    def process_frame(self, fi, frame):
        """Update read and frame records for one frame index."""
        if self.gpu is not None:
            self.buf.append((fi, frame))
            if len(self.buf) >= self.gpu.batch:
                self._flush()
            return
        if self.verbose and fi % 300 == 0 and fi:
            r = fi / (time.perf_counter() - self.t0)
            print(f"frame {fi} reads={len(self.raw)} {r:.1f}f/s "
                  f"align_dropped={self.n_align_dropped} motion_skip={self.n_motion_skip}",
                  flush=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # MOTION-FIRST gate: skip mid-turn (moving) + redundant-held frames before
        # paying for pose+color. Moving = rest boundary (reset); >n_rest reads in a
        # rest = redundant. The decode is rest-anchored, so a few reads/rest suffice.
        if self.mgate and self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            h, w = gray.shape
            cm = float(np.abs(
                gray[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.int16)
                - self.prev_gray[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.int16)).mean())
            if cm >= self.move_thr:
                self.rest_reads = 0                       # motion = new rest boundary
            if cm >= self.move_thr or self.rest_reads >= self.n_rest:
                self.n_motion_skip += 1
                self.frec[fi] = {"bbox": None, "faces": 0, "stk": 0,
                                  "motion": round(cm, 1), "vis": {}}
                self.prev_gray = gray
                return
        # Align-cls gate (only when --align-thresh > 0; pose-only baseline keeps
        # the legacy "all face-detected frames" behavior). Motion FIRST happens
        # implicitly via the bbox-diff motion field; alignment classifier is the
        # 2nd-pass gate per the production pipeline order.
        client = _get_legacy_client()
        if self.align_thresh > 0:
            ac = client.classify_alignment(frame)
            if ac < self.align_thresh:
                self.n_align_dropped += 1
                self.frec[fi] = {"bbox": None, "faces": 0, "stk": 0,
                                  "motion": None, "vis": {}, "align_conf": ac}
                # don't bother running pose if cls rejects → save inference time
                self.prev_gray = gray
                return
        faces = client.infer_faces(frame, conf=self.conf)
        labs, confs = [], []
        for _, fc in slot_names(faces):
            q = order_quad(np.asarray(fc["corners"], np.float32))
            crop = warp_crop(frame, q, ordered=True)
            l, c = masked_cell_labs(crop)                        # chroma skin-masked
            labs.append(l)
            confs.append(c)
        self._emit_face_reads(fi, gray, self.prev_gray, faces, labs, confs)
        self.prev_gray = gray

    def finalize(self):
        """Drain the final batch and return reads plus frame records."""

        self._flush()
        return self.raw, self.frec


def main():
    PERF_TRACE.mark("main_enter")
    tag = argval("--tag", None)
    if not tag:
        sys.exit("--tag is required")
    vid = argval("--video", None)
    if not vid:
        sys.exit("--video is required")
    cal = argval("--centroids-json", "calibration.json")
    out = argval("--out", f"/tmp/reads_{tag}_geo.pkl")
    workstation_out = argval("--workstation-out", None)
    conf = argval("--conf", 0.5, float)          # face-detect conf (geometry tolerant)
    occ = argval("--occ", 40.0, float)           # cell conf = max(0, 1 - dist/occ)
    # Alignment-classifier gate: when > 0, frames whose aligned classifier
    # confidence is below this threshold are SKIPPED (no read emitted).
    # The default is off because downstream state fitting can still consume reads
    # from frames below the alignment threshold.
    align_thresh = argval("--align-thresh", 0.0, float)
    # Motion-first gate: retain the first N frames in a rest and skip redundant
    # held frames plus frames above the motion threshold.
    mgate = "--motion-gate" in sys.argv
    move_thr = argval("--move-thr", 3.0, float)
    n_rest = argval("--n-rest", 8, int)
    PERF_TRACE.set_meta(tag=tag, video=vid)

    # Metadata-only probe for reporting and GPU batch sizing.
    PERF_TRACE.mark("video_open_start")
    with PERF_TRACE.span("reads.metadata_probe"):
        _probe = cv2.VideoCapture(vid)
        _vid_fps = _probe.get(cv2.CAP_PROP_FPS) or 120.0
        _vid_n = int(round(_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        _vid_w = _probe.get(cv2.CAP_PROP_FRAME_WIDTH)
        _vid_h = _probe.get(cv2.CAP_PROP_FRAME_HEIGHT)
        _probe.release()
    frame_hw = (int(_vid_h), int(_vid_w)) if _vid_w and _vid_h else None

    # --gpu-reads batches the two ONNX models and cell reads. --gpu-warp also
    # moves perspective warping to Torch and may differ numerically from OpenCV.
    gpu = ("--gpu-reads" in sys.argv
           or os.environ.get("CUBED_GPU_READS", "") == "1")
    # --gpu-batch-size / CUBED_GPU_BATCH omitted => GpuReadEngine derives it at
    # construction time from actually-free VRAM + this video's frame size
    # (scripts/gpu_reads.py derive_batch_size); an explicit value still
    # overrides verbatim, and a mid-run OOM halves it and retries rather than
    # crashing (GeoReadStage._flush_with_retry).
    _gb = argval("--gpu-batch-size", None) or os.environ.get("CUBED_GPU_BATCH")
    gpu_batch = int(_gb) if _gb else None
    gpu_warp = "--gpu-warp" in sys.argv

    # --gpu-decode uses NVDEC and keeps frames on device. The parity guard checks
    # representative output against OpenCV; --gpu-decode-verify also compares
    # the full decoded frame count.
    gpu_decode = ("--gpu-decode" in sys.argv
                  or os.environ.get("CUBED_GPU_DECODE", "") == "1")
    gpu_verify = "--gpu-decode-verify" in sys.argv
    if gpu_decode:
        if not gpu:
            sys.exit("--gpu-decode requires --gpu-reads (the batched engine is "
                     "what consumes the on-device frames)")

    with PERF_TRACE.span("reads.engine_init"):
        stage = GeoReadStage(
            cal, conf=conf, occ=occ, align_thresh=align_thresh,
            mgate=mgate, move_thr=move_thr, n_rest=n_rest,
            tag=tag, verbose=True, gpu=gpu, gpu_batch=gpu_batch,
            gpu_warp=gpu_warp, gpu_decode=gpu_decode,
            gpu_frame_hw=frame_hw, workstation=workstation_out is not None)
    PERF_TRACE.set_meta(
        gpu_reads=bool(stage.gpu), gpu_decode=gpu_decode, gpu_warp=gpu_warp,
        motion_gate=mgate, move_thr=move_thr, n_rest=n_rest,
        align_thresh=align_thresh,
        gpu_batch=(stage.gpu.batch if stage.gpu is not None else None),
        frame_hw=frame_hw, fps=_vid_fps, container_frames=_vid_n)

    fi = -1
    if gpu_decode:
        from gpu_decode import GpuVideoDecoder, verify_parity
        with PERF_TRACE.span("reads.nvdec_guard"):
            verify_parity(vid, full_count=gpu_verify)  # decode-drift guard, before any work
        with PERF_TRACE.span("reads.nvdec_open"):
            decoder = GpuVideoDecoder(vid)
        with PERF_TRACE.span("reads.frame_loop"):
            for gf in decoder.frames():
                fi = gf.fi
                stage.process_frame(fi, gf)
    else:
        with PERF_TRACE.span("reads.frame_loop"):
            cap = cv2.VideoCapture(vid)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fi += 1
                stage.process_frame(fi, frame)
            cap.release()

    with PERF_TRACE.span("reads.finalize"):
        raw, frec = stage.finalize()
    with PERF_TRACE.span("reads.serialize"):
        with open(out, "wb") as fh:
            pickle.dump((raw, frec), fh)
        if workstation_out is not None:
            try:
                workstation_stage = build_workstation_stage(
                    frec,
                    stage.workstation_geometry or {},
                    fps=_vid_fps,
                    frame_count=fi + 1,
                    width=int(_vid_w),
                    height=int(_vid_h),
                )
                if stage.workstation_warning:
                    workstation_stage["warnings"].append({
                        "code": "workstation.overlay-unavailable",
                        "message": stage.workstation_warning,
                    })
                write_workstation_stage(workstation_out, workstation_stage)
            except Exception as error:
                print(
                    f"  [workstation] WARNING: sidecar not written "
                    f"({type(error).__name__}: {error})",
                    flush=True,
                )
    PERF_TRACE.mark("artifact_closed")
    PERF_TRACE.set_meta(
        decoded_frames=fi + 1, read_frames=len(raw), frec_frames=len(frec),
        output=out)

    # how many cells survive occlusion gating
    vis_stk = [frec[f]["stk"] for f in frec if frec[f]["stk"]]
    print(f"\n{tag}: frames={fi + 1}  reads(any face+motion)={len(raw)}  "
          f"frames-with-faces={sum(1 for f in frec if frec[f]['faces'])}")
    if vis_stk:
        print(f"  visible cells/frame (conf>0.3): median={np.median(vis_stk):.0f} "
              f"max={max(vis_stk)}  (9 = one full face)")
    print(f"  wrote -> {out}")
    PERF_TRACE.mark("process_complete")
    PERF_TRACE.write()


if __name__ == "__main__":
    main()
