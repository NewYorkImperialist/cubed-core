"""Generate trellis events from the READ stream's motion alone (NO GT): the
rest-anchored segmentation. Low bbox-motion = rest; the peak-motion frame of each
active period between rests = one event. Output [{"frame": N}] for
trellis_gt --events-json. Usage: <tag> [out.json]
"""
import sys, json, pickle
import numpy as np

tag = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/motion_events_{tag}.json"
raw, _ = pickle.load(open(f"/tmp/reads_{tag}_v2.pkl", "rb"))
frames = sorted(raw)
fa = np.array(frames)
motion = np.array([raw[f][1] for f in frames], float)
sm = np.array([np.median(motion[max(0, i - 2):i + 3]) for i in range(len(motion))])
THR = float(np.percentile(sm, 50))
active = sm > THR

MIN_REST = 5
runs = []
i = 0
while i < len(active):
    if active[i]:
        j = i
        while j + 1 < len(active) and active[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1
    else:
        i += 1
periods = runs[:1]
for s, e in runs[1:]:
    if fa[s] - fa[periods[-1][1]] < MIN_REST:
        periods[-1][1] = e
    else:
        periods.append([s, e])

events = [{"frame": int(fa[s + int(np.argmax(sm[s:e + 1]))])} for s, e in periods]
json.dump(events, open(out, "w"))
print(f"{tag}: {len(events)} motion events -> {out}")
