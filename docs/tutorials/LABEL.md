# Label

Route: `/label`

Label is an optional desktop annotation tool for cube footage. It is
keyboard-first, autosaves drafts, and can export portable annotation JSON or an
exact-frame training ZIP.

Interactive navigation seeks the already-open video directly on its frame
clock. It does not invoke FFmpeg once per frame. FFmpeg is used only when you
explicitly build an exact-frame dataset ZIP.

## Open a source

Label supports:

| Source | Frame navigation | Autosave | Exact-frame ZIP |
| --- | --- | --- | --- |
| Workspace recording | Shared streamed video, measured frame clock | `workspace/annotations/<capture-id>/frame-annotations.json` | Yes |
| Local video file | Browser-local video, chosen frame rate | Browser storage keyed to file identity | No |

Use a workspace recording when stable SHA-256 identity, durable autosave, or
exact-frame export matters. Use a local file for quick manual work.

To import an existing video into the workspace:

```console
uv run --no-sync cubed-core import-video /absolute/path/to/video.mp4 --source import --notes "Label source"
```

The [Add video page](IMPORT.md) imports an existing video into the workspace.
Label does not require a starting scramble, calibration, phone pairing, BLE,
IMU, or Decode-ready recording.

Manual labeling needs no GPU or model weights. The standard setup installs the
`label` dependency extra:

```bash
./setup.sh --install-uv
make workbench
```

Manual Label is inside the beta native Windows scope described in the root
README. Its storage contract and frontend are covered separately in CI, not by
a real-media Windows browser run. Model assistance below remains a POSIX
workflow and is not claimed on native Windows.

## Annotation types

Label can store:

- free-form polygons
- ordered four-corner face poses
- per-corner visibility
- rigid same-frame cube completion
- `aligned` and `unaligned` frame classes

Annotations live outside the sealed capture bundle, so editing labels never
changes the source receipt or video checksum.

## Basic workflow

1. Choose a workspace recording or local file.
2. Seek to the frame you want.
3. Press `D` for a polygon or `C` for face corners.
4. Draw and inspect the annotation.
5. Mark the frame aligned or unaligned when appropriate.
6. Move to the next frame with the arrow keys.
7. Download JSON regularly as a portable checkpoint.

Drafts autosave, but export is still the deliberate finish step.

## Four-corner and rigid-pose workflow

1. Enter **Corners** mode with `C`.
2. Click three consecutive corners of one face. The second click is the shared
   elbow between the visible edges.
3. When geometry is valid, the CPU helper fills the fourth corner, solves one
   rigid cube pose, and proposes the other camera-facing faces.
4. Drag a numbered vertex to correct it. Shared vertex IDs move together.
5. Use `1` through `4` or the visibility controls to mark occluded corners.
6. Press `X` to remove generated completion while keeping the manually drawn
   face.

Autofill is a suggestion, not ground truth. Inspect every generated corner.
When the solve is unavailable or degenerate, corner mode stays active so you
can place the fourth point manually.

The PnP helper can use:

- camera intrinsics from a workspace capture receipt
- a generic image-centered focal prior
- a custom focal length

It assumes zero lens distortion and cannot infer an unknown crop, mirror, or
sensor-mode change. A valid camera matrix does not make generated points exact.

## Polygon workflow

1. Press `D`.
2. Click at least three points.
3. Click near the first point, double-click, or press Enter to close.
4. Press `V` to select and edit vertices.
5. Delete with Delete or Backspace.

An exactly four-point polygon can be converted into editable face corners.
Polygons and pose faces can coexist on one frame.

## Navigation and shortcuts

Shortcuts are ignored while focus is inside an input, select, textarea, or
button.

| Shortcut | Action |
| --- | --- |
| Left / Right Arrow | Previous / next matching frame |
| Shift + Left / Right Arrow | Move ten matching frames |
| `D` | Draw polygon |
| `C` | Draw corners or convert a selected four-point polygon |
| `V` | Select mode |
| `R` | Copy raw geometry from the nearest earlier annotated frame |
| `P` | Predict the current frame when models are configured |
| `F` | Run or rerun rigid autofill |
| `X` | Clear generated rigid completion |
| `1` to `4` | Toggle selected-corner visibility |
| `A` / `U` | Mark aligned / unaligned |
| Enter | Close a polygon |
| Delete / Backspace | Delete selected geometry |
| Command-Z / Control-Z | Undo |
| Command-S / Control-S | Save now |
| Escape | Cancel the active draft |

Navigation filters can restrict arrows to:

- annotated frames
- human-aligned frames
- frames accepted by the alignment model

Enabled filters intersect. The scrubber still reaches the full source.

## Model assistance

Label reuses the same released `camera-tracker-v1` alignment and face-pose
models as the camera pipeline. There is no Label-only model stack.

For local CPU assistance:

```bash
make download-runtime
make bootstrap-label-cpu
source .venv-label-cpu/bin/activate

export CUBED_CORE_TRACKER_MODEL_MANIFEST="$PWD/workspace/release-assets/camera-tracker-v1-runtime/manifest.json"
export CUBED_CORE_TRACKER_ONNX_PROVIDERS=CPUExecutionProvider

cubed-core verify-tracker-models
make doctor-label-cpu
make workbench-label-cpu
```

Enabling the model-aligned filter runs a whole-video alignment scan on demand.
Opening a recording alone does not run inference. **Predict** operates on the
current frame. **Predict all** skips frames that already have annotations and
asks for confirmation before the longer run.

Only one model operation runs at a time. Predictions remain suggestions and
must be reviewed before export.

For CUDA-assisted Label and Decode on the same host, use
`make bootstrap-decode-gpu` and `make workbench-decode-gpu`. See
[GPU setup](../CLOUD_GPU.md).

## Portable annotation JSON

**Download JSON** creates:

```text
<video-stem>.frame-annotations.json
```

The v1 document stores source identity, display dimensions, frame and time
records, alignment classes, polygons, pose corners, visibility, assist origin,
model confidence, and optional rigid-cube geometry.

Schema:
[`frame-annotations-v1`](../../schemas/frame-annotations-v1.schema.json)

Reload the JSON only with the same source video. A workspace record carries the
capture ID and source SHA-256. A local-file record may carry only its filename
and no SHA-256. Browser autosave additionally keys local files by size and
modification time, but portable JSON does not include those values. Manually
verify a same-named local file before loading its annotations.

## Exact-frame dataset ZIP

For a workspace recording, **Download dataset ZIP** saves the current
annotations and builds:

```text
dataset.yaml
manifest.json
images/train/*.jpg
labels/train/*.txt
images/val/*.jpg
labels/val/*.txt
coco/annotations.json
classification/aligned/*.jpg
classification/unaligned/*.jpg
```

This explicit export is the only Label operation that uses FFmpeg frame
extraction. Ordinary seeking, scrubbing, and arrow navigation continue to use
the shared video directly.

Pose faces become YOLO keypoint labels. Free-form polygons become COCO
segmentation records. Alignment classes become classification folders.

With at least five annotated frames, the exporter creates a deterministic
SHA-256-ranked 80/20 convenience split. This stays inside one capture and is
not a valid held-out research split. Real training and evaluation must separate
capture and solve identities.

Export is capped at 2,048 annotated frames and 2 GiB. Local-file mode cannot
export exact source frames. Import the original into the workspace first.

## Quality checklist

- Confirm the source identity and exact frame.
- Keep one corner-order convention.
- Mark occluded corners without moving them to a visible location.
- Inspect every generated face and rigid wireframe.
- Treat model output and PnP as suggestions.
- Use capture-level splits for actual evaluation.
- Download and reload JSON before calling a long session complete.

## Troubleshooting

### The next frame is slow

Label should seek the shared HTML video directly. It should not show a
per-frame FFmpeg decode. Confirm you are on the current workbench build and
that the browser can seek the source codec. Variable-frame-rate video can still
make frame-time navigation approximate.

### Dataset ZIP export fails

Run `make doctor` and confirm FFmpeg can read the preserved source. ZIP export
is different from interactive navigation.

### Autosave fails

Check workspace permissions in workspace mode or browser storage in local-file
mode. Download JSON before continuing.

### Source mismatch appears

The open video and annotation record have different identities. Compare the
capture ID and SHA-256 when present. A local record may have only a filename, so
verify the exact file manually. Do not continue on a guessed source.

### Prediction is unavailable

Verify the manifest and provider:

```bash
cubed-core verify-tracker-models
make doctor-label-cpu
```

Model bytes are release assets, not Git-tracked files.

## Data boundary

This repository does not accept recording or label contributions. Keep videos,
annotations, and dataset ZIPs local unless every artifact has separate consent,
privacy review, and license clearance. See [Privacy](../PRIVACY.md),
[Licensing](../LICENSING.md), and [Contributing](../../CONTRIBUTING.md).

To run a video through the camera-to-moves pipeline, use
[Decode and Runs](DECODE.md).
