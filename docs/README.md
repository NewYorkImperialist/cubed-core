# Documentation

Start with the root [README](../README.md). It has the supported setup and
public-dataset quickstart. Native Windows is experimental and may be broken.
Use the warning in that README to choose between native Windows and WSL2.

Use one guide for the task you are doing:

| Task | Canonical guide |
| --- | --- |
| Record a new solve or import one | [Add video](tutorials/IMPORT.md) |
| Run Decode or inspect a run | [Decode and Runs](tutorials/DECODE.md) |
| Annotate video and export labels | [Label](tutorials/LABEL.md) |
| Download and inspect the public corpus | [Dataset](DATASET.md) |
| Configure local or remote CUDA | [GPU setup](CLOUD_GPU.md) |
| Train or package tracker models | [Train tracker](TRAIN_TRACKER.md) |
| Understand the decode architecture | [Decoder internals](DECODER_INTERNALS.md) |

For a new solve, Add video displays a scramble. Apply it, record with your
normal camera app, confirm the video starts from it, and upload the original
file. The secondary existing-recording path requires its exact known scramble.
Then scrub the imported video and sample one clear sticker for each of
the six colors. The local server creates and validates the calibration. Shared,
reused, and uploaded calibration files remain secondary choices. The selected
calibration stays attached to the recording and appears automatically in
Decode. Its row uses a human source label and hides internal schema IDs and
hashes. There is no bundled phone recorder or pairing step.

All frame rates use the one `CUBED_CORE_MAX_UPLOAD_BYTES` limit, which defaults
to 1 GiB. Native 220–242 fps footage still needs a linked 120 fps derivative
for Decode, but that media step does not change the upload cap.

Camera-only Decode needs the exact scramble applied before the recording and a
six-color calibration selection. Sampling one sticker of each color from that
recording is preferred. The published shared calibration may be selected for
another video after its setup-mismatch warning is shown. It is not evidence
that the setups match.

`make download-dataset` verifies and hard-links all published videos into the
workspace without duplicating their bytes. Thirty-one have frame-zero
scrambles and can be prepared in Decode. The picker leads with their filenames,
groups your own recordings before the published set, and adds the recorded
timestamp only when names collide. Your recordings are newest first. Published
recordings use natural filename order. Locking a recording preserves its video
and scramble for any number of sequential Decode attempts. You may replace its
calibration between attempts, and each attempt keeps a private snapshot.
`gtD2`, `gtD3`, `gtD4`, and `gtD5` start solved and scramble on-camera, so they
remain listed but are not directly runnable under the current starting-state
preflight.

The standard Decode request has no BLE, teacher, or IMU input. For a recording
registered from the pinned dataset revision, a BLE move-sequence comparison may
be added only after the server freezes the camera result. A completed sequence
must replay to solved first. Runs displays the precomputed diagnostic without
running compute. This request boundary does not prove the upstream environment
was teacher-isolated.

## Project boundaries

- [Evidence](EVIDENCE.md)
- [Privacy](PRIVACY.md)
- [Licensing](LICENSING.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Research notes](RESEARCH_NOTES.md)

Machine-readable JSON formats live under [`schemas/`](../schemas/). Dataset,
capture, calibration, benchmark, and model cards live beside their artifacts
on Hugging Face:

- [Public dataset](https://huggingface.co/datasets/cubed-core/cubed-data-v1)
- [gtD1 teacher record](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/captures/bef12b32179b4409bbb45a032f2fec7a/clip_ble_ground_truth.json)
- [Benchmark report](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/benchmark/BENCHMARK_V0_RUN_REPORT.md)
- [Camera tracker model](https://huggingface.co/cubed-core/camera-tracker-v1)
- [Read-trust model](https://huggingface.co/cubed-core/read-trust-v1)
