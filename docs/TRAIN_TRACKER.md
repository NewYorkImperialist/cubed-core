# Build or package tracker models

This guide gets you a `camera-tracker-v1` model directory that the native
tracker can load. You have two ways to make one:

- **Package ONNX files you already have** into the layout the runtime expects, or
- **Train your own** from Label exports using the pinned recipe here.

Cubed Core does not ship the trained models in Git. Use the
[model artifact index](../models/README.md), bring your own compatible ONNX
files, or train your own with this guide. See the
[current model card](https://huggingface.co/cubed-core/camera-tracker-v1).

A tracker directory holds two roles:

- `alignment-classifier`: class index zero is `aligned`
- `face-pose`: one face class and four ordered keypoints

These models produce camera evidence. They do not turn the camera into cube
moves. A separate decoder stage does that. Training, packaging, or running one
of these models tells you nothing about camera-to-moves accuracy.

To check the model interfaces before you invest in training, run
`make bootstrap-label-cpu` and the focused model/runtime tests:

```bash
.venv-label-cpu/bin/python -m pytest -q \
  tests/test_model_artifacts.py \
  tests/test_camera_tracker_backend.py \
  tests/test_tracker_runtime.py
```

These are contract checks, not a warm start, benchmark, or trained model. They
do not replace the workflow below.

## Package existing ONNX files

Install the CPU tracker environment. Packaging opens each ONNX file with the
pinned ONNX Runtime CPU provider, so the base workbench environment is not
enough on its own:

```bash
make bootstrap-label-cpu
source .venv-label-cpu/bin/activate
```

Write a private `MODEL_CARD.md` that records the intended use, exact artifact
identity, training provenance, license, evaluation boundary, and limitations.
The published
[camera tracker card](https://huggingface.co/cubed-core/camera-tracker-v1)
shows the expected level of detail. Then write a private spec like this:

```json
{
  "schema": "cubed-core/tracker-model-package-spec-v1",
  "schema_version": 1,
  "purpose": "local-review",
  "profile": "camera-tracker-v1",
  "onnx_input_trust": "operator-reviewed",
  "model_card": "/absolute/private-review/CAMERA_TRACKER_MODEL_CARD.md",
  "artifacts": [
    {
      "id": "my-alignment-v1",
      "role": "alignment-classifier",
      "path": "/absolute/private-models/alignment.onnx",
      "license": "AGPL-3.0-only",
      "source": "https://example.org/immutable-training-receipt",
      "redistribution": "external-only"
    },
    {
      "id": "my-face-pose-v1",
      "role": "face-pose",
      "path": "/absolute/private-models/face-pose.onnx",
      "license": "AGPL-3.0-only",
      "source": "urn:example:immutable-model-receipt:face-pose-v1",
      "redistribution": "external-only"
    }
  ]
}
```

Fill in every field. `source` must be an HTTPS link with no credentials in it,
or a receipt URN with no file path in it. The packager rejects license
placeholders like `NOASSERTION`, private paths left in public metadata,
half-finished model-card sections, and any artifact where you never made a
redistribution decision.

Use `local-review` while rights or release review is still open. A
`publication-candidate` spec also requires you to set `redistribution:
approved` on both artifacts. That only records the decision as metadata. Legal
clearance and publication approval still need real review, which the generated
`audit.json` says out loud.

Only feed in ONNX files from a source you trust. The
`onnx_input_trust: operator-reviewed` acknowledgement is required because ONNX
Runtime parses an executable model graph. There is no sandbox around it.

Pick an output path outside every Git worktree:

```bash
make package-tracker-models \
  SPEC=/absolute/private-review/package-spec.json \
  OUTPUT=/absolute/outside-git/camera-tracker-v1
```

Same thing as a direct command:

```bash
cubed-core package-tracker-models \
  --spec /absolute/private-review/package-spec.json \
  --output-dir /absolute/outside-git/camera-tracker-v1
```

The command downloads nothing and refuses to overwrite an existing output
directory. It copies your files under normalized names and always produces the
same layout:

```text
camera-tracker-v1/
├── MODEL_CARD.md
├── SHA256SUMS
├── artifacts/
│   ├── alignment-classifier.onnx
│   └── face-pose.onnx
├── audit.json
└── manifest.json
```

It checks that:

- the input files are nonempty, regular `.onnx` files (symlinks are rejected)
- the alignment input accepts `(1, 3, 224, 224)` float32 and its first output
  has at least two classes
- the pose input accepts `(1, 3, 1024, 1024)` float32 and its first output has
  17 channels
- the packaged bytes match their computed size and SHA-256
- the generated runtime manifest passes Cubed Core's strict loader
- the generated text leaks no input paths or common private home paths

It cannot check class meaning, keypoint order, training consent, license
ownership, patent clearance, redistribution authority, model quality, or
safety. Those are on you.

Verify the finished directory yourself:

```bash
cd /absolute/outside-git/camera-tracker-v1
shasum -a 256 -c SHA256SUMS
export CUBED_CORE_TRACKER_MODEL_MANIFEST="$PWD/manifest.json"
cubed-core verify-tracker-models
```

That block uses macOS `shasum`. On Linux, run `sha256sum -c SHA256SUMS`
instead.

Point `CUBED_CORE_TRACKER_MODELS_DIR` at that directory for Compose, or copy it
to a cloud host over your normal authenticated file transfer. The
[local/cloud GPU guide](CLOUD_GPU.md) has the full CPU, NVIDIA, container,
SSH-tunnel, and real-job commands.

## Prepare train-your-own data from Label exports

On `/label`, **Download dataset ZIP** builds exact JPEGs, YOLO pose labels, and
`aligned` / `unaligned` classification folders. Export several separate
captures. A generalization claim requires a named, leakage-controlled
evaluation over held-out capture groups. Merely using more than one capture is
not enough.

Assign each whole capture ZIP to either train or validation yourself. The
preparer ignores the ZIP's frame-level split and won't run unless each lane has
at least one capture:

```bash
python scripts/prepare_tracker_training_data.py \
  --train-export /absolute/private-data/capture-a.dataset.zip \
  --train-export /absolute/private-data/capture-b.dataset.zip \
  --validation-export /absolute/private-data/capture-c.dataset.zip \
  --output-dir /absolute/outside-git/tracker-training-data
```

The command:

- reads size-limited, unencrypted ZIP members without `extractall`
- rejects path traversal, symlinks, duplicate members, images and labels that
  don't match frame-for-frame, malformed YOLO rows, and contradictory alignment
  classes
- keeps every capture whole, in the split you chose
- swaps capture IDs for pseudonyms and writes no source paths or capture IDs
- emits `pose/dataset.yaml`, `alignment/{train,val}/{aligned,unaligned}`, and
  `dataset-manifest.json`
- refuses to write inside a Git worktree

This only handles the technical packaging. Consent and data licensing are
separate concerns. See [Privacy and sensitive-data handling](PRIVACY.md) and
[Dataset and media licensing](LICENSING.md#dataset-and-media).
Keep raw recordings and prepared datasets outside Git. Split by
solver/cube/camera/setup as well as by capture whenever those groups could leak
between train and validation.

## Minimal pinned Ultralytics training/export recipe

The optional training environment pins the exact train/export toolchain:
Ultralytics `8.4.76`, NumPy `1.26.4`, OpenCV `4.11.0.86`, Torch `2.2.2`,
Torchvision `0.17.2`, ONNX `1.17.0`, ONNX Runtime `1.23.2`, and ONNXSlim
`0.1.94`:

```bash
make bootstrap-training
source .venv-training/bin/activate
python -c 'import ultralytics; assert ultralytics.__version__ == "8.4.76"'
```

This pinned training environment and the base workbench require Python 3.10
through 3.12. The pinned Torch release publishes no wheels for newer Python. If
your `python3` is newer, point the bootstrap at an installed compatible
interpreter:

```bash
make bootstrap-training TRAINING_BOOTSTRAP_PYTHON=python3.12
```

This pin defines the documented train-your-own environment. It is a starting
point, not a promise of byte-for-byte reproduction. For a durable release,
record in the model card the resolved selected-platform package set from the
universal `uv.lock`, your Python/CUDA/driver identity, the commands, seeds,
base hashes, dataset-manifest hash, and produced hashes.

Ultralytics can auto-download a named base. This recipe forbids that: get the
base files yourself, review their rights, hash them, and pass absolute existing
paths. Offline mode blocks the library's online availability checks:

```bash
export YOLO_OFFLINE=true
export TRACKER_DATA=/absolute/outside-git/tracker-training-data
export TRACKER_RUNS=/absolute/outside-git/tracker-training-runs
export ALIGNMENT_BASE=/absolute/reviewed-bases/yolo11n-cls.pt
export POSE_BASE=/absolute/reviewed-bases/yolo11s-pose.pt

test -f "$ALIGNMENT_BASE"
test -f "$POSE_BASE"
mkdir -p "$TRACKER_RUNS"

yolo classify train \
  model="$ALIGNMENT_BASE" \
  data="$TRACKER_DATA/alignment" \
  project="$TRACKER_RUNS" name=alignment \
  epochs=60 imgsz=224 batch=64 seed=0 deterministic=True

yolo pose train \
  model="$POSE_BASE" \
  data="$TRACKER_DATA/pose/dataset.yaml" \
  project="$TRACKER_RUNS" name=face-pose \
  epochs=200 imgsz=1024 batch=16 seed=0 deterministic=True

yolo export \
  model="$TRACKER_RUNS/alignment/weights/best.pt" \
  format=onnx imgsz=224 batch=1 dynamic=False simplify=True device=cpu

yolo export \
  model="$TRACKER_RUNS/face-pose/weights/best.pt" \
  format=onnx imgsz=1024 batch=1 dynamic=False simplify=True device=cpu
```

Those epoch and batch values just mirror the recorded candidate budgets. They
are not tuned recommendations or proof of quality. Your hardware may need
an explicit `device=` and a smaller batch. Do not report validation numbers
from this convenience recipe as held-out model or decoder performance without a
rights-cleared evaluation plan and immutable evidence.

Finally, write an honest model card and feed the two resulting ONNX paths into
the packager above. Do not publish a whole training run directory: vendor
checkpoints and logs can hold local paths and other metadata. Only the
normalized ONNX package is meant for normal runtime distribution.
Package-generated metadata drops the input paths, but the exact ONNX bytes and
their embedded metadata are preserved. If you ship a raw checkpoint, make it a
separate, checksummed, trusted-source asset with an explicit pickle warning.

## License boundary

The training dependency and the recorded candidate lineage both use
Ultralytics. The vendor's [licensing page](https://www.ultralytics.com/license)
lays out its AGPL and Enterprise options and its stance on trained models.
Cubed Core's AGPL-3.0-only code license does not, by itself, prove that a
particular base, dataset, trained checkpoint, exported model, or use inside a
separate hosted product is cleared. Record the exact upstream terms and get
legal or vendor review where you need it.
