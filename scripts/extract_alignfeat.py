"""Write per-frame alignment probabilities for the explicit input video."""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, cv2
from detect.inference_client import get_inference_client


def argval(flag, d=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


def main():
    tag = argval("--tag", None)
    if not tag:
        sys.exit("--tag is required")
    vid = argval("--video", None)
    if not vid:
        sys.exit("--video is required")
    out = argval("--out", f"/tmp/alignfeat_{tag}.npz")
    client = get_inference_client()
    cap = cv2.VideoCapture(vid)
    frames, al = [], []
    fi = -1
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        al.append(float(client.classify_alignment(frame)))
        frames.append(fi)
        if fi % 500 == 0 and fi:
            print(f"frame {fi} {fi / (time.perf_counter() - t0):.1f}f/s "
                  f"aligned_mean={np.mean(al[-500:]):.2f}", flush=True)
    cap.release()
    np.savez(out, frame=np.array(frames), aligned=np.array(al, np.float32))
    print(f"{tag}: {fi + 1} frames  aligned_mean={np.mean(al):.3f}  "
          f"frac>0.5={np.mean(np.array(al) > 0.5):.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
