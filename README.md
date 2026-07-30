# Cubed Core

Cubed Core reconstructs a 3x3 puzzle cube solve from ordinary 120 fps phone
video. It combines per-frame camera evidence with an exact search over legal
cube states and whole-cube orientations.

The public workflow is intentionally small:

1. **Demo** shows one preserved run with synchronized video, overlays, face
   reads, alignment timeline, moves, and reconstruction. It needs no GPU.
2. **Decode** runs the complete `local_camera_v1` pipeline on a published
   Hugging Face video or your own compatible recording. Live decode needs an
   NVIDIA CUDA GPU.
3. **Runs** keeps Decode history and opens the same artifact for read-only
   frame inspection. Runs never starts or retries compute.
4. **Add video** displays a scramble for a new recording, then accepts the
   original video from your normal camera app. It can also import an existing
   recording when you know its exact starting scramble. After upload, scrub the
   video and sample one clear sticker for each of the six colors.
5. **Label** is an optional local annotation tool for people preparing their
   own training data.

Cubed Core is research software. The released models are overfit to the
maintainer's capture setup, and the shared dataset calibration was measured on
one recording under one lighting condition. For a completed result, use the
exact statements “The job completed on this input.” and “The sequence replayed
to solved.” Neither is an accuracy or generalization claim. Read the [evidence
boundary](docs/EVIDENCE.md) before publishing a result.

**Data and models:** [public dataset](https://huggingface.co/datasets/cubed-core/cubed-data-v1)
· [camera tracker](https://huggingface.co/cubed-core/camera-tracker-v1)
· [read-trust model](https://huggingface.co/cubed-core/read-trust-v1)

## Try the workbench

On macOS, Linux, or WSL2, install Python 3.10 through 3.12, Node 22.3 or
newer on the Node 22 line, FFmpeg, Git, Make, and the pinned `uv` version:

```bash
git clone https://github.com/KingBobJoeIV/cubed-core.git
cd cubed-core
./setup.sh --install-uv
make workbench
```

> [!WARNING]
> Native Windows is experimental and may still be broken. The current fix for
> the `/api/captures` and media-ticket failure has not yet run on a real
> Windows machine or in hosted Windows CI. Use WSL2 or Linux when you need a
> dependable path. Reproducible Windows failures are high priority and will be
> fixed as quickly as possible.

The experimental native Windows path is limited to the CPU workbench. Its
intended scope is Demo, Add video, manual Label, Decode preparation, and Runs.
The Windows CI job covers the capture-storage lifecycle, core API, dataset
registration, asset downloaders, frontend tests and build, and mocked browser
flows. It does not yet exercise real Windows media import end to end. With
Python 3.12, Node 22.3 or newer on the Node 22 line, FFmpeg, Git, and
`uv 0.11.16` on `PATH`, run from PowerShell:

```powershell
git clone https://github.com/KingBobJoeIV/cubed-core.git
cd cubed-core
uv sync --locked --python 3.12 --extra dev --extra label --extra decode
npm --prefix apps/lab-web ci
npm --prefix apps/lab-web run build
uv run --no-sync cubed-core serve
```

Native mode rejects live Decode submissions on Windows, including the bundled
SSH GPU bridge. Use WSL2 or Linux for the supported live Decode path.

Open the loopback URL printed by the service. The local browser session
bootstraps access automatically. There is no admin-token setup step for normal
same-machine use.

For Docker:

```bash
git clone https://github.com/KingBobJoeIV/cubed-core.git
cd cubed-core
make docker-cpu
```

The base workbench can run Demo, Runs, Decode preparation, and Label without a
GPU. A live Decode job still needs CUDA on this machine or on a configured
remote host.

## Run the public dataset

Download the reviewed dataset revision and release assets:

```bash
make download-dataset download-assets
```

Without Make, including from native Windows PowerShell, use:

```powershell
uv run --no-sync python scripts/download_dataset.py --workspace workspace
uv run --no-sync python scripts/download_release_assets.py
```

Release downloads are pinned to `v1.0.0`. `download-assets` fetches the camera
models, Demo clip, read-trust model, shared calibration, and their notices.
The dataset command verifies the pinned revision and registers all 35 videos in
the local workspace with hard links, so Decode can list them without a second
copy. It does not attach calibration, copy teacher or IMU sidecars, lock a
recording for decode, or run compute. It separately verifies the 35 published
BLE references for optional post-decode diagnostics without exposing them to
the camera runner.

Decode groups your recordings first and the published dataset second. Your
recordings are newest first. Published recordings use natural filename order,
and duplicate names add their recorded timestamp.

Thirty-one recordings include an exact frame-zero scramble. Select one in
Decode by filename, attach the published shared calibration, lock its video and
scramble, and run the same pipeline used for your own recording. The lock is
reusable: submit as many sequential Decode attempts as you need, and replace
the calibration between attempts without changing the video or scramble. `gtD2`,
`gtD3`, `gtD4`, and `gtD5` start solved and show the scramble on-camera. They
remain visible but are not directly runnable until Decode has an explicit
honest contract for that starting-state sequence. See the
[pinned Hugging Face quickstart](docs/tutorials/DECODE.md#pinned-hugging-face-quickstart).

The **Compute target** selector always offers
**Local CUDA (this workbench)**. Any configured remote or Vast.ai GPU hosts
appear in that same selector. Both paths return the same
`cubed-core/decode-result` v1 JSON document. The native Windows workbench can
prepare a request but cannot submit to either target. Run it under WSL2 or
Linux first. See:

- [Decode and Runs](docs/tutorials/DECODE.md)
- [GPU setup, including Vast.ai](docs/CLOUD_GPU.md)
- [Public dataset](docs/DATASET.md)

## Add your own recording

Open **Add video** for the recommended path. Apply the displayed scramble to a
solved cube, record the full solve with your normal camera app, transfer the
original file to this computer, and confirm that the recording starts from that
scramble before uploading it.

If you already have a recording, use **Use an existing recording** and enter
the exact scramble that was applied before it began. Do not guess it.

After upload, use **Sample this video** for calibration. Scrub to clear frames
and click the middle of the white, green, red, blue, orange, and yellow center
stickers. The local server converts and validates those six crops, then attaches
the calibration to that recording. Decode uses the attached calibration
automatically and names its source in plain language rather than showing schema
IDs or hashes. A shared, reused, or uploaded calibration remains available as a
secondary choice.

Support for attaching BLE data to your own recordings is planned for an
upcoming release. Today, BLE is available only as a post-decode diagnostic for
matching published dataset recordings. It is never a camera Decode input.

Every video uses the same byte limit, regardless of frame rate. The default is
1 GiB and can be changed with `CUBED_CORE_MAX_UPLOAD_BYTES`. Native 220–242 fps
footage is accepted under that same cap, but standard Decode needs its linked
120 fps derivative.

## What a run contains

One Decode submission creates one workspace attempt. The video and scramble
stay locked to the recording. The selected calibration is copied into that
attempt before compute starts:

```text
locked video + scramble ─┐
                         ├─► Decode attempt ─► decode-result.json
calibration snapshot ────┘                      │
                                               ▼
                                  Runs: inspect or delete
```

Replacing the recording's attached calibration affects only later submissions.
An in-flight or older attempt keeps its exact snapshot and SHA-256 identity.
Runs does not retry or recompute an attempt.

For a recording registered from the pinned dataset revision, the same result
JSON may also contain a precomputed BLE move-sequence comparison. Runs only
displays it. The server freezes the camera result first. If its status is
completed, the sequence must replay to solved before this optional diagnostic
is added.

The browser download is named
`cubed-core-run-<job-or-recording-id>.json`. The portable JSON does not embed
the video. Runs pairs it with an exact matching workspace or local video by
SHA-256 before showing synchronized playback and overlays.

Native runs can include an optional browser-safe `workstation` projection with
per-frame motion, alignment, streak, faces, face reads, trellis spans, and
reconstruction checkpoints. Each checkpoint applies one or more reconstructed
moves as an atomic state update at a frame supported by the decoder. These
frames are for scrubbing between decoder states, not timestamps for individual
physical turns. Runs renders only the sections present. Compatible
external runners remain valid without those diagnostics.

Raw runner logs, local paths, reads pickle files, alignment NPZ files, motion
scratch data, credentials, and provider details are not portable run content.

## Current boundaries

- The tested and recommended profile is 120 fps with a short edge of at least
  1080 pixels. Known readable nonstandard rates and resolutions warn but can
  still run. A native 220–242 fps original requires a deterministic linked
  120 fps derivative for standard decode.
- Decode requires the exact starting scramble and a six-color calibration
  selection. Add video can create one by sampling the six visible center
  stickers from the imported video. The published shared calibration remains
  selectable for another video with a visible setup-mismatch warning.
- The standard Decode request has no BLE, smart-cube, or IMU input. Published
  BLE can appear only as an optional post-decode diagnostic. Missing or
  mismatched reference data never blocks Decode. This interface boundary is not
  proof of an isolated upstream environment.
- A calibration sampled from the same recording is the recommended choice.
  Shared, reused, and uploaded calibration JSON remain secondary options. A
  calibration from another cube, camera, lens, exposure, or lighting setup can
  be less reliable.
- The public dataset records named completed, abstained, and failed outcomes.
  Changed code, models, or configuration can produce a different outcome.
- The repository does not include a phone recorder or pairing flow. Record with
  a camera app you already use. Custom synchronized teacher collection is
  future work and is not required for Decode.

## Repository map

```text
apps/lab-web/          React workbench
config/                versioned runtime and download manifests
core/                  cube representation and legal move logic
detect/                camera analysis
models/                local ignored model directory and artifact pointers
schemas/               machine-readable JSON contracts
scripts/               canonical decode, GPU, and dataset tooling
src/cubed_core/        local API, workspace, jobs, and CLI
tests/                  Python and browser-contract tests
workspace/              ignored local captures, runs, labels, and assets
```

The maintained user documents are:

- [Add a video](docs/tutorials/IMPORT.md)
- [Decode and Runs](docs/tutorials/DECODE.md)
- [Label](docs/tutorials/LABEL.md)
- [Dataset](docs/DATASET.md)
- [GPU setup](docs/CLOUD_GPU.md)
- [Evidence](docs/EVIDENCE.md)
- [Privacy](docs/PRIVACY.md)
- [Licensing](docs/LICENSING.md)
- [Research notes](docs/RESEARCH_NOTES.md)
- [Camera tracker model](https://huggingface.co/cubed-core/camera-tracker-v1)
- [Read-trust model](https://huggingface.co/cubed-core/read-trust-v1)

## Contributing

Run the full local gate before a pull request:

```bash
make check
npm --prefix apps/lab-web run test:e2e
```

The browser command is required for web changes. See
[CONTRIBUTING.md](CONTRIBUTING.md) for DCO sign-off, evidence, and data-safety
rules. Report vulnerabilities through
[GitHub private vulnerability reporting](SECURITY.md).

Code is licensed under [AGPL-3.0-only](LICENSE). Models, datasets, recordings,
and other artifacts carry separate terms described in
[Licensing](docs/LICENSING.md).
