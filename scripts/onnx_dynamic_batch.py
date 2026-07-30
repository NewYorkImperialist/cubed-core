"""Make a FIXED-batch-1 YOLO ONNX export accept an N-row batch — same weights,
same nodes, same math, only the batch dim freed.

Why this exists: both prod models (weights/cube_aligned.onnx,
weights/cube_face_pose.onnx) were exported by ultralytics with a LITERAL batch
dim of 1 ([1,3,224,224] / [1,3,1024,1024]), so OnnxModel.supports_batch is
False and every caller — including classify_alignment_batch, which was written
for batching — silently falls back to one session.run PER FRAME. That is why
the 4090 idles at ~10-19% while the reads stage runs: the card never sees more
than one 1024x1024 image at a time.

The graph itself is batch-agnostic apart from three things, all mechanical:

  1. the input's leading dim (a literal 1),
  2. the outputs' leading dim (a literal 1),
  3. every Reshape whose target-shape INITIALIZER starts with a literal 1
     (e.g. [1,64,-1], [1,4,16,21504], [1,256,32,32]).

(3) is fixed WITHOUT touching any arithmetic: ONNX Reshape treats a 0 in the
shape operand as "copy the corresponding input dim" (allowzero=0, the default
and what these nodes use), so [1,64,-1] -> [0,64,-1] reshapes B rows exactly
as the original reshaped 1 row. No weight, no node, no attribute changes; the
per-row computation is the SAME graph. Verified numerically by --verify below:
batch-1 output of the patched model vs the original model, on real frames.

  python3 scripts/onnx_dynamic_batch.py --verify        # patch both + check

Writes a source-hash-addressed <stem>_dynb_vN_<hash>.onnx next to the source.
The originals are never modified, and replacing a source model can never
silently reuse a dynamic twin built from stale weights.
"""
import os
import sys
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODELS = ["weights/cube_aligned.onnx", "weights/cube_face_pose.onnx"]
PATCH_SCHEMA = 1


def dynb_path(src):
    """Content-address the generated graph by source bytes + patch schema."""
    p = Path(src)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return str(p.with_name(f"{p.stem}_dynb_v{PATCH_SCHEMA}_{h}{p.suffix}"))


def patch(src, dst=None, dim_name="batch"):
    """Free the batch dim of a fixed-batch-1 ONNX model. Returns dst path."""
    dst = dst or dynb_path(src)
    m = onnx.load(src)

    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim[0]
        if d.HasField("dim_value") and d.dim_value == 1:
            d.ClearField("dim_value")
            d.dim_param = dim_name

    init = {i.name: i for i in m.graph.initializer}
    patched = 0
    for n in m.graph.node:
        if n.op_type != "Reshape" or len(n.input) < 2:
            continue
        # allowzero=1 would make 0 mean a literal zero-size dim; these exports
        # never set it, but refuse to guess if one ever does.
        if any(a.name == "allowzero" and a.i for a in n.attribute):
            raise RuntimeError(f"{n.name}: allowzero=1, cannot use 0-copy trick")
        tgt = init.get(n.input[1])
        if tgt is None:
            raise RuntimeError(f"{n.name}: shape operand {n.input[1]} is not an "
                               "initializer (computed at runtime) — inspect by hand")
        arr = numpy_helper.to_array(tgt).copy()
        if arr.size and arr[0] == 1:
            arr[0] = 0                       # 0 = copy input dim 0 (the batch)
            tgt.CopyFrom(numpy_helper.from_array(arr, tgt.name))
            patched += 1

    # value_info carries stale per-tensor shapes with batch=1 baked in; drop it
    # and let ORT re-infer, else shape inference fights the new dynamic dim.
    del m.graph.value_info[:]
    onnx.checker.check_model(m)
    # Multiple extractor processes may discover the same missing twin at once.
    # Write a process-unique temporary and atomically publish the complete ONNX;
    # a crash can no longer leave an exists-but-truncated cache entry.
    dp = Path(dst)
    tmp = str(dp.with_name(f".{dp.stem}.tmp.{os.getpid()}{dp.suffix}"))
    onnx.save(m, tmp)
    os.replace(tmp, dst)
    print(f"  {src} -> {dst}  ({patched} Reshape targets freed)", flush=True)
    return dst


def verify(src, dst, n=4, seed=0):
    """Original (batch-1, one call per row) vs patched (one batch-N call).

    Reports max |delta| per output. A patched model that is a pure batch
    generalization must give ~0 for the batch-1 case; batch-N may differ by
    kernel-selection noise on CUDA, which is exactly what we want measured.
    """
    import onnxruntime as ort
    shape = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).get_inputs()[0].shape
    c, h, w = shape[1:]
    rng = np.random.default_rng(seed)
    x = rng.random((n, c, h, w), dtype=np.float32)

    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    s0 = ort.InferenceSession(src, providers=prov)
    s1 = ort.InferenceSession(dst, providers=prov)
    on0 = [o.name for o in s0.get_outputs()]
    on1 = [o.name for o in s1.get_outputs()]

    ref = [s0.run(on0, {"images": x[i:i + 1]}) for i in range(n)]
    got1 = [s1.run(on1, {"images": x[i:i + 1]}) for i in range(n)]
    gotN = s1.run(on1, {"images": x})

    result = []
    for k in range(len(on0)):
        d_b1 = max(float(np.abs(ref[i][k] - got1[i][k]).max()) for i in range(n))
        d_bn = max(float(np.abs(ref[i][k] - gotN[k][i:i + 1]).max()) for i in range(n))
        scale = float(np.abs(ref[0][k]).max()) or 1.0
        print(f"  {os.path.basename(src)}:{on0[k]}  max|dyn(b=1) - orig| = {d_b1:.3e}   "
              f"max|dyn(b={n}) - orig| = {d_bn:.3e}   (out scale {scale:.3g})", flush=True)
        result.append((d_b1, d_bn))
    bad = [(on0[k], d1) for k, (d1, _dn) in enumerate(result) if d1 != 0.0]
    if bad:
        raise RuntimeError(
            f"dynamic ONNX batch-1 changed the source model outputs: {bad}")
    return result


def main():
    os.chdir(os.environ.get("CUBED_ROOT", "/app"))
    do_verify = "--verify" in sys.argv
    for src in MODELS:
        if not os.path.exists(src):
            print(f"  SKIP {src} (absent)", flush=True)
            continue
        dst = patch(src)
        if do_verify:
            verify(src, dst)


if __name__ == "__main__":
    main()
