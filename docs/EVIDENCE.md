# Evidence

Cubed Core separates execution, reconstruction, and evaluation. State only what
the named artifact or check supports.

Use these sentences:

- “The job completed on this input.”
- “The sequence replayed to solved.”
- “The run abstained.”
- “The named closed evaluation reached its reported metric.”

Do not infer model accuracy from completion, GPU-path validation from device
visibility, generalization from Demo or one corpus, or teacher isolation from
an input schema.

## The evidence ladder

| Check | What it supports | What it does not support |
| --- | --- | --- |
| `nvidia-smi` | An NVIDIA device and driver are visible | CUDA library readiness, model execution, or quality |
| Runtime and model verification | Named packages, providers, manifests, and model bytes are available | Inference on a real video |
| Synthetic interface smoke | The covered preprocessing, model interface, routing, and output contract execute | Trained-model quality or camera performance |
| Real camera stage | The stage completed on one named video and produced schema-valid camera evidence | A reconstructed move sequence or accuracy |
| Completed Decode result plus server replay | The job completed on this input and the sequence replayed from the named scramble to solved | Per-move correctness, reach-LL, or generalization |
| Closed teacher comparison | A frozen prediction was compared with separate reviewed truth under a named metric | Performance outside that named evaluation |

Checksums support byte identity. They do not establish origin, permission,
privacy, quality, or correctness.

## Decode outcomes

One Decode submission creates one attempt. A successful runner exit is not
enough to create a result. The API validates the result contract and, for a
completed result, replays the emitted moves from the locked scramble.

### Completed

Say:

- “The job completed on this input.”
- “The sequence replayed to solved.”

Do not say that the moves matched the video or that the model was accurate.
Multiple legal sequences can reach the same endpoint. If the server replay
does not reach solved, the attempt fails.

### Abstained

Say: “The run abstained.”

This means the pipeline terminated normally without a solved reconstruction.
It is a result, not a crash.

### Failed, timed out, or cancelled

These are attempt outcomes and may have no `decode-result.json`. Runs keeps a
bounded terminal record. Raw runner output stays local because it may contain
private paths or provider details.

## What to preserve

For a workbench attempt, keep:

- `decode-result.json` when present
- the server-authored request, receipt, and `run-terminal.json`
- the capture receipt and exact video, scramble, and calibration identities
- configuration and model hashes
- the local runner log only when diagnosis requires it

Only `decode-result.json` is portable. It excludes local paths, raw logs,
pickle and NPZ intermediates, credentials, and provider details.

Deleting a run moves its job directory to recoverable workspace trash. It does
not delete the input video, calibration, annotations, or labels.

## Workstation diagnostics

A native result can include `cubed-core/decode-workstation-v1`. It binds the
video identity and can show motion, alignment, streak, faces, sampled reads,
trellis spans, and reconstruction checkpoints from the existing pass. Each
checkpoint groups one or more reconstructed moves into an atomic state update
at a decoder-supported frame. A checkpoint frame is not a timestamp for an
individual physical turn, BLE observation, or teacher event. This is an
inspection projection, not another inference or evaluation. External runners
may omit it.

Runs may evenly pace the ordered moves between two checkpoints to make playback
easier to follow. Those intermediate cube states are labeled estimated. Only
the saved state at the checkpoint frame is decoder-supported.

Runs refuses a playback video with a different SHA-256. That protects artifact
identity. It does not establish that the recording was captured honestly or
that the reconstructed moves match the video.

## The published Demo replay

Demo is a zero-compute replay of one preserved gtD1 result. Its server receipt
records an `unverified-external-identity` runner and a 77-move sequence that
replayed from the named scramble to solved.

A preserved result keeps the configuration identity recorded when it ran. Its
hash need not equal the current canonical runner hash.

A separate public smart-cube record has 91 quarter turns. The documented
half-turn normalization produces 77 tokens equal to the preserved sequence for
this capture. This is a comparison for one named input. It is not a reach-LL
measurement, accuracy rate, speed benchmark, teacher-isolation result, or
evidence about another setup.

The source artifacts and benchmark report live at the pinned
[Hugging Face dataset revision](https://huggingface.co/datasets/cubed-core/cubed-data-v1/tree/7fae604962c590ac9c658ba6ee0350e86de9c4f5).

## Teacher isolation

The standard Decode request has no teacher, raw BLE, or phone IMU input. The
portable workstation schema has no teacher-record field. Those are interface
boundaries, not proof that the process or host was isolated from teacher data.

Native and external subprocesses run as the same operating-system user and are
not sandboxed. Claim teacher isolation only when the upstream execution
environment was separately isolated and never received teacher artifacts.

For a closed evaluation:

1. freeze inputs and configuration
2. run and preserve the prediction
3. close the prediction
4. open separately held truth
5. evaluate with the named metric contract

Do not use truth to select thresholds, anchors, budgets, or retries for the
prediction being evaluated.

## Research metric

The primary metric for a named closed research evaluation is reach-LL:

> Does the raw predicted state path reach last-layer onset with the correct
> pre-last-layer state, up to whole-cube orientation?

Solved endpoint, edit distance, phase progress, runtime, and abstention are
diagnostics. They do not replace reach-LL.

Say: “The named closed evaluation reached its reported metric.” Cite the exact
report and score artifact, not memory:

[Benchmark v0 run report](https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/benchmark/BENCHMARK_V0_RUN_REPORT.md)

## Generalization

The released models and decoder were developed against the maintainer's
recordings. The public corpus contains named cohorts. It does not represent
arbitrary users, cubes, cameras, lighting, occlusion, or solve speeds.

A result on one recording says nothing about another recording without a
separate, named evaluation.

See [Decode and Runs](tutorials/DECODE.md), [Dataset](DATASET.md), and
[Research notes](RESEARCH_NOTES.md).
