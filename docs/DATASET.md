# Public dataset

Cubed Core's public solve corpus is hosted at
[`cubed-core/cubed-data-v1` on Hugging Face](https://huggingface.co/datasets/cubed-core/cubed-data-v1).
The checked-in download manifest pins one reviewed immutable revision:

```text
dataset id: cubed-solves-v1
Hugging Face revision: 7fae604962c590ac9c658ba6ee0350e86de9c4f5
captures: 35
```

[`config/public-dataset-v1.json`](../config/public-dataset-v1.json) records the
full revision, expected byte count, and SHA-256 of every downloaded file.

The corpus contains named completed, abstained, and failed outcomes. These are
records of specific attempts, not accuracy labels or download failures. They do
not predict what changed code, models, or configuration will produce.

## Download

From the repository root:

```bash
make download-dataset
```

Without Make, including from native Windows PowerShell:

```powershell
uv run --no-sync python scripts/download_dataset.py --workspace workspace
```

The downloader:

1. fetches the exact pinned revision
2. streams each file into a staging area
3. verifies every byte count and SHA-256
4. writes `download-receipt.json`
5. exposes the destination only after the complete revision verifies
6. registers every verified video in the local workspace with a hard link

The installed path is:

```text
datasets/public/cubed-solves-v1/
```

A failed download never publishes a partial destination. Rerun the same command
to reuse revision-bound cache entries.

Registration creates normal incomplete workspace recordings without copying
the video bytes. It also writes a local ignored index that binds each of the 35
published BLE references to the verified dataset revision, video, recording,
and scramble. Teacher and sensor files are not copied into capture inputs.
Registration does not attach calibration, lock a recording, or run Decode. The
dataset directory and workspace must be on the same filesystem because the
supported path fails instead of silently making a second copy.

You can also inspect the immutable revision directly on
[Hugging Face](https://huggingface.co/datasets/cubed-core/cubed-data-v1/tree/7fae604962c590ac9c658ba6ee0350e86de9c4f5).

## Run a capture

Download the release assets too:

```bash
make download-assets
```

Decode lists all 35 registered recordings. Thirty-one have an exact frame-zero
starting scramble and can be prepared normally:

1. select a recording by filename in Decode
2. attach **Published dataset shared calibration**
3. read its cross-cohort warning and lock the video and scramble
4. choose local or remote CUDA and run

The picker groups personal recordings first and the published dataset second.
Published recordings use natural filename order. Duplicate filenames add the
recorded timestamp. The lock is not a one-run seal. Submit any number of Decode
attempts sequentially from the same video and scramble. If a calibration
produces poor reads, replace it and submit another attempt. Every job keeps the
exact calibration bytes and SHA-256 that it started with, so changing the
attached calibration never rewrites an earlier run.

`gtD2`, `gtD3`, `gtD4`, and `gtD5` start solved and scramble on-camera. They
remain listed for inspection, but the current preflight requires an explicit
honest starting-state contract and cannot run them directly. Do not invent a
frame-zero scramble or reuse the later on-camera scramble as if it were already
applied.

The published captures use `calibration_gan12.json`. It was measured on gtD1
under one lighting condition while the corpus spans multiple recording cohorts.
Decode keeps this limitation visible. The file remains selectable for another
video after the mismatch warning, but another cube, camera, lens, exposure, or
lighting setup can be less reliable.

Add video and Label are not required to run the downloaded videos.
`cube_session` and IMU files remain in the dataset tree as optional teacher or
sensor evidence. The standard Decode request does not include them. This
interface boundary does not prove that an upstream process or host was
teacher-isolated.

After the server freezes a camera result, it may add one optional
`ground_truth_diagnostic` to that same result JSON. A completed sequence must
replay to solved first. The diagnostic folds adjacent same-face quarter turns
from the BLE reference into canonical half-turn-metric moves, then compares
move sequences. Thirty-four captures have sequence comparison only. The gtD1
demo clip also has clip-local frame timing. Runs displays this precomputed
diagnostic and performs no comparison itself. Missing, changed, or mismatched
reference data simply omits the diagnostic and never fails Decode.

## Layout

The reviewed revision has:

```text
.gitattributes
README.md
SHA256SUMS
LICENSES/
  DATASET.md
dataset/
  manifest.json
metadata.jsonl
readiness-report.json
captures/
  <capture-id>/
    video.mp4
    scramble.json
    rights_record.json
    ... optional reviewed teacher or sensor sidecars ...
benchmark/
  BENCHMARK_V0_RUN_REPORT.md
  benchmark-manifest.json
  ... generated reports, scores, and narratives ...
```

`dataset/manifest.json` is the machine-readable release authority. It binds the
dataset and release IDs, every capture, every artifact role, SHA-256, byte
count, media type, license, attribution, provenance, privacy approval, and
measured video metadata.

`metadata.jsonl` has one row per canonical video. Use its `file_name`,
`capture_id`, split, dimensions, frame count, cadence, and rights fields rather
than guessing from filenames.

For 31 captures, `scramble.json` supplies the exact state already present at
frame zero. The four solved-start recordings named above need a future explicit
starting-state sequence contract instead. Optional sidecars vary by capture:

- `cube_session.json` or `app_cube_session.json` is teacher/evaluation data
- `imu.json` is sensor evidence
- `capture_manifest.json` records source-side capture metadata
- `clip_ble_ground_truth.json` is the reviewed gtD1 clip-local teacher record

Keep teacher and sensor files outside production inference even though they are
published in the same repository.

## Cards and benchmark evidence

GitHub holds code and the bootstrap manifest. Dataset, recording, BLE,
calibration, and benchmark cards belong with the data on Hugging Face:

- [dataset card](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/README.md)
- [benchmark run report](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/benchmark/BENCHMARK_V0_RUN_REPORT.md)
- [gtD1 teacher record](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/captures/bef12b32179b4409bbb45a032f2fec7a/clip_ble_ground_truth.json)

Treat `dataset/manifest.json`, `SHA256SUMS`, and the checked-in bootstrap
manifest as the identity authorities.

Do not copy benchmark claims back into GitHub prose from memory. Cite the exact
Hugging Face report and score artifact.

## ML and CV use

Read the inventory with the standard library:

```python
import json
from pathlib import Path

root = Path("datasets/public/cubed-solves-v1")
rows = [
    json.loads(line)
    for line in (root / "metadata.jsonl").read_text().splitlines()
]

for row in rows:
    print(row["capture_id"], root / row["file_name"], row["split"])
```

Use whole-capture splits. Never put frames from one capture in both training and
evaluation. An `unassigned` split can support an experiment-local split, but it
is not a published benchmark split.

For a named closed evaluation, teacher truth must remain closed until the
prediction is finalized. The primary research metric is reach-LL: reaching
last-layer onset with the correct pre-last-layer state, up to whole-cube
orientation. Full-sequence edit distance and solved endpoint are separate
diagnostics. See [Evidence](EVIDENCE.md).

## License, privacy, and contributions

The dataset's `LICENSES/DATASET.md`, per-artifact manifest entries, and rights
records are authoritative. A collection license does not erase an artifact's
own terms. See [Licensing](LICENSING.md).

Published video can contain hands, faces, rooms, device information, and other
personal data. A checksum proves identity, not permission or anonymity. See
[Privacy](PRIVACY.md).

This GitHub repository does not accept recording or label contributions. Do not
attach captures, labels, calibration files, BLE records, or dataset exports to
issues or pull requests. See [Contributing](../CONTRIBUTING.md).
