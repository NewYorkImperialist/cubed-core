# Decode and Runs

Routes: `/decode` and `/runs`

Decode runs Cubed Core's complete camera-to-moves pipeline. Runs is the
read-only history and inspector for those attempts. The Decode result is the
sole public compute artifact. Any tracker diagnostics are embedded in it.

## Start with Demo

Open `/demo` first. Demo renders one preserved run with its video, overlays,
sampled face reads, alignment timeline, move sequence, and reconstructed cube.
It runs no inference and needs no GPU or model download.

For this preserved artifact, the job completed on this input and the sequence
replayed to solved. That is not an accuracy rate, reach-LL result, speed
benchmark, or claim about another user, cube, camera, or lighting setup. See
[Evidence](../EVIDENCE.md).

## Inputs and requirements

A live run needs:

- a readable video with finite measured timing and dimensions
- the exact starting scramble
- a six-color calibration, preferably sampled from that video's visible stickers
- the released camera models and read-trust model
- FFmpeg and FFprobe
- an NVIDIA CUDA host

The beta native Windows path is intended for recording preparation and Runs.
It rejects the bundled local and remote Decode runners. On a Windows computer,
run the workbench under WSL2 or Linux before submitting any live Decode job.

The released models were trained on the maintainer's data. The published
dataset uses a shared calibration measured on gtD1 under one lighting
condition. Decode shows that limitation when you select it. You may proceed
with it on another video after reading the mismatch warning, but another cube,
camera, lens, exposure, or lighting setup can be less reliable.

The recording picker has two groups. **Your recordings** is newest first.
**Published dataset** uses natural filename order. Both lead with the filename,
and a duplicate name adds its recorded timestamp. The calibration row uses
human labels such as **Sampled from this video**, **Published shared
calibration**, **From _filename_**, or the safe uploaded filename. It never
shows internal schema IDs, capture IDs, or hashes.

BLE, smart-cube, teacher, and IMU files are optional evaluation or sensor
evidence. The standard Decode request does not include them. Dataset
registration verifies published BLE references separately. This interface
separation does not prove that the process or host was teacher-isolated.

Download the release inputs with:

```bash
make download-assets
```

The runtime asset contains the alignment and face-pose ONNX models.
`decode-support` contains `trust_v1_numpy.npz` and the shared published-corpus
calibration plus its license notice. The pinned `v1.0.0` download also includes
the Demo clip and its notice. The downloader verifies SHA-256 values.
The exact model identities and limitations are recorded in the
[camera tracker](https://huggingface.co/cubed-core/camera-tracker-v1) and
[read-trust](https://huggingface.co/cubed-core/read-trust-v1) model cards.

## Pinned Hugging Face quickstart

This is the minimum public-data workflow. It uses the immutable dataset
revision in
[`config/public-dataset-v1.json`](../../config/public-dataset-v1.json) and
registers every verified published capture. The preserved `gtD1s` demo source,
`bef12b32179b4409bbb45a032f2fec7a`, is one of the 31 directly runnable
recordings.

Install the base workbench once:

```bash
./setup.sh --install-uv
```

Download the reviewed dataset and decode assets:

```bash
make download-dataset download-assets
make workbench
```

The dataset command verifies the pinned revision, then hard-links every
published video into the local workspace. Decode lists all 35 recordings
without copying their video bytes. Registration does not copy teacher or IMU
files, attach calibration, lock a recording, or run any compute.

Thirty-one recordings include their exact frame-zero starting scramble. To run
one:

1. Open **Decode** and select one of those 31 recordings from **Published
   dataset**.
2. Choose **Published dataset shared calibration** and lock the video and
   scramble.
3. Read the calibration warning. It is expected for this corpus.
4. In **Compute target**, choose **Local CUDA (this workbench)** or a configured
   remote GPU host.
5. Confirm that preflight is ready, then select **Run decode**.
6. When the attempt finishes, choose **Open in Runs** or **Download run JSON**.

That lock is reusable. It prevents the video and starting scramble from
drifting, but it does not limit the recording to one job. Submit as many
sequential attempts as you need. You may also replace its attached calibration
between attempts. Each job copies the selected calibration into its own
directory and hash-binds it before compute starts, so later changes cannot alter
an in-flight or historical run.

Four recordings use a different, still-visible contract:

- `gtD2`
- `gtD3`
- `gtD4`
- `gtD5`

They start solved and scramble on-camera. The current preflight needs an
explicit honest frame-zero starting state, so these four are listed but are not
directly runnable. Do not paste the later on-camera scramble into the
frame-zero field or invent another scramble to bypass the check.

The published dataset card, capture outcomes, data cards, and benchmark records
live in the
[Hugging Face dataset repository](https://huggingface.co/datasets/cubed-core/cubed-data-v1).
The checked-in manifest pins revision
`7fae604962c590ac9c658ba6ee0350e86de9c4f5`.

## Use your own video

The workbench runs the same pipeline for your video:

1. Follow [Add video](IMPORT.md). For a new solve, apply its displayed scramble,
   record with your normal camera app, confirm the video starts from that
   scramble, and upload the original. For an existing recording, enter its
   exact known scramble.
2. In Add video, scrub the recording and sample one clear sticker for
   each of the six colors. The local server creates and validates the
   calibration, attaches it to the recording, and makes it available
   automatically in Decode. Shared, reused, and uploaded calibration files
   remain secondary choices.
3. Lock the workspace recording's video and scramble for decode.
4. Open Decode and choose the compute target.

For example:

```bash
source .venv/bin/activate
cubed-core import-video /absolute/path/to/solve.mp4 \
  --source import \
  --camera-facing external \
  --scramble "R U R' U'"
```

Import prints a receipt with the new capture ID. Finish calibration and lock
the video and scramble in the workbench before Decode.

If the recording never shows a clear sticker for every color, you may
select the published shared calibration and proceed after reading the mismatch
warning. It was measured on gtD1 under one lighting condition, so another cube
or recording setup can be less reliable. Decode warns rather than treating
resolution, frame rate, or calibration provenance as proof that two setups
match.

The repository does not include a phone recorder or pairing flow. Do not create
placeholder teacher, BLE, or IMU files. They do not make a recording
Decode-ready.

## Choose local or remote CUDA

The page uses one **Compute target** selector:

- **Local CUDA (this workbench)** runs on the machine hosting the API.
- A configured remote or Vast.ai entry uses the bundled SSH bridge.

Both choices use native mode and produce the same result contract. A host
appearing in the selector means only that its configuration parsed. Preflight
reports the checks it performed. Only the terminal attempt records whether the
job completed on its input.

For local CUDA:

```bash
make bootstrap-research-gpu
source .venv-decode-gpu/bin/activate
CUBED_CORE_DECODE_MODE=native make workbench-decode-gpu
```

For a remote box, provision it and register the connection on the workbench
host. Follow [GPU setup](../CLOUD_GPU.md#remote-gpu-with-vastai-or-ssh). Keep
`CUBED_CORE_DECODE_MODE=native`. Choosing the remote entry makes the service use
`scripts/remote_decode_runner.sh` for that attempt.

Decode execution is disabled when `CUBED_CORE_DECODE_MODE` is blank. The API
reads the setting at startup, so restart the workbench after changing it.

## What the job runs

`local_camera_v1` performs the camera and search stages in one job:

1. locate and warp visible cube faces
2. sample per-cell Lab color evidence
3. derive motion and alignment evidence
4. build settled spans and legal cube-state hypotheses
5. run the trellis and endpoint search
6. write one structured result

Native workstation diagnostics are projected from that same pass. Cubed Core
does not decode or infer the video a second time for Runs.

At submission, the server reads one consistent version of the locked video,
scramble, and attached calibration. It copies the exact calibration bytes to
the job directory and includes their SHA-256 in the request. The native runner
checks that digest before heavy work begins. This same check applies after a
remote upload. Replacing calibration later affects only later submissions.

The runtime authority is
[`config/decode-runtime-v1.json`](../../config/decode-runtime-v1.json).
Its executable mirror is `scripts/run_research_decode.sh`. Model choice and
decoder flags stay in those versioned files rather than the browser. The
workbench is the supported OSS entry point. Lower-level research automation
uses that canonical runner on CUDA, never a laptop CPU.

## Result and attempt status

The portable artifact is always:

```text
schema: cubed-core/decode-result
schema_version: 1
workspace filename: decode-result.json
browser filename: cubed-core-run-<job-or-recording-id>.json
```

Read the result's `status`:

- `completed`: “The job completed on this input.” The server separately checks
  that “The sequence replayed to solved.”
- `abstained`: “The run abstained.” The pipeline terminated normally without a
  solved reconstruction.

The API independently replays a completed move sequence from the locked
scramble. A claimed completion that does not replay to solved becomes a failed
attempt.

For an exactly matched published recording, the server may then add an optional
top-level `ground_truth_diagnostic`. It remains inside the same downloadable
result JSON. The camera result is already validated before the reference is
opened. A completed sequence must replay to solved first.

Attempt lifecycle is separate from result status. The API reports
`queued`, `running`, `succeeded`, `failed`, `timed_out`, or `cancelled`, plus a
user-facing outcome when known. Every terminal attempt writes a bounded,
log-free `run-terminal.json`, so completed, abstained, failed, and cancelled
attempts survive restart. Persisted failure messages are generic and never copy
raw runner logs.

The service accepts one queued or running Decode job at a time across the
workspace, and times out a runner after six hours. Its job index and the Runs
list are bounded to the newest 100 attempts. Older job directories are not
deleted by that indexing bound.

## Runs

Runs groups attempts by exact video SHA, falling back to capture ID when a SHA
is unavailable, newest first. It provides:

- search by filename, status, or ID
- **Inspect** and **Download** when a result exists
- **Refresh** for active attempts
- **Delete run** for terminal attempts
- **Open run JSON** for a portable result from another workspace

Runs never starts, retries, or resumes compute. Return to Decode to submit
another attempt from the same locked recording. You do not need to unlock or
re-import its video first. Runs only renders any precomputed
`ground_truth_diagnostic`. It does not compare sequences itself.

Deleting a terminal run atomically moves only its job directory to:

```text
workspace/run-trash/decode-jobs/<trash-id>/
```

The operation is recoverable at the filesystem level. It never removes the
capture, video, calibration, annotations, or labels. Active attempts cannot be
deleted.

## Optional published BLE diagnostic

`make download-dataset` verifies one published BLE reference for each of the 35
recordings and binds it to the exact dataset revision, video, recording, and
scramble. It does not copy teacher data into the capture or job inputs.

After the server freezes the camera result, it folds adjacent same-face quarter
turns into canonical half-turn-metric moves and stores a diagnostic
sequence comparison in the result. A completed sequence must replay to solved
first. Thirty-four references are sequence-only. The gtD1 demo clip also
declares clip-local frame timing.

This comparison is not a Decode input or a success metric. Missing, changed, or
mismatched reference data simply omits the diagnostic. It never changes or
fails the camera result.

## Portable video pairing

The result JSON does not embed video. A native result's optional workstation
declares the source video SHA-256, byte count, FPS, frame count, dimensions, and
frame window.

Runs automatically pairs an exact workspace video when available. For a local
portable result, use **Choose run JSON** and then **Choose matching video**.
Playback and overlays are enabled only when the selected video SHA-256 matches.
A mismatch is refused and shows the expected and actual hashes. Non-video run
diagnostics remain available.

## Optional workstation diagnostics

A native result can include strict browser-safe
`cubed-core/decode-workstation-v1` data:

- video identity and frame window
- starting scramble and warnings
- per-frame motion, alignment probability, aligned streak, face count, pose
  corners, and sampled Up/Front/Right reads
- event frames
- trellis spans and top candidates
- decoder checkpoint frames for atomic reconstructed-state updates
- decoder-checkpoint reconstruction states and endpoint information

Top-level `moves` and `endpoint` remain authoritative. The browser derives cube
states from the scramble and moves when native output omits a state list. If
native output includes a reconstruction timeline, moves that share a checkpoint
frame remain one atomic saved state. Runs may evenly pace those ordered moves
between checkpoints as a display-only estimate, then snap to the saved state at
the checkpoint. Estimated frames are not per-move or physical-action
timestamps, BLE timing, or teacher timing, and they do not replace top-level
`moves`. Compatible external runners may omit the entire workstation section.
Runs renders only data actually present.

The portable JSON excludes local paths, provider details, raw logs, pickle and
NPZ intermediates, motion scratch data, raw teacher records, and credentials.

Schemas:

- [`decode-result-v1`](../../schemas/decode-result-v1.schema.json)
- [`decode-ground-truth-diagnostic-v1`](../../schemas/decode-ground-truth-diagnostic-v1.schema.json)
- [`decode-run-terminal-v1`](../../schemas/decode-run-terminal-v1.schema.json)
- [`decode-job-request-v1`](../../schemas/decode-job-request-v1.schema.json)

## Troubleshooting

### Decode has no recordings

Run `make download-dataset` to register the verified public videos, or use Add
video for a local recording. Then attach calibration and lock the selected
recording's video and scramble for decode.

### A calibration produced poor reads

Return to the selected recording's preparation panel and attach or sample a
replacement calibration. The video and starting scramble stay locked. Submit a
new Decode attempt. Runs keeps the previous result and its job-private
calibration snapshot unchanged.

### Preflight is blocked

Read the named checks. Common blockers are a missing scramble, calibration,
FFmpeg, model asset, unreadable timing, or unknown dimensions. Known readable
nonstandard rates and resolutions produce warnings rather than blockers. A
native 220–242 fps original needs its deterministic linked 120 fps derivative.
Do not create placeholder files to silence a check.

For a workspace capture:

```bash
make decode-preflight CAPTURE=<full-capture-id>
```

### No compute target is available

Local CUDA requires the workbench to run from `.venv-decode-gpu` with native
decode enabled. A remote entry requires non-interactive SSH and a provisioned
CUDA environment. Use [GPU setup](../CLOUD_GPU.md).

### The attempt failed

Runs preserves failed, timed-out, and cancelled terminal attempts even when no
result JSON exists. The displayed failure is intentionally bounded. Inspect
the local workspace job directory and runner log on a trusted machine when
deeper diagnosis is needed. Do not paste raw logs into a public issue.

### The selected video is refused

The video bytes do not match the SHA-256 declared by the run. Select the exact
original file. Do not override the check to make overlays appear.

## What this does not prove

For a completed result, use the exact statements “The job completed on this
input.” and “The sequence replayed to solved.” For an abstention, say “The run
abstained.” None of these statements establishes per-move accuracy, reach-LL,
model quality, teacher isolation, or generalization.

For the exact claims and research metric boundary, read
[Evidence](../EVIDENCE.md). For tried approaches and real dead ends, read
[Research notes](../RESEARCH_NOTES.md).
