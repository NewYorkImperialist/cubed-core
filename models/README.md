# Model artifacts

Trained artifacts live outside Git with their canonical cards and checksums:

- [Camera tracker v1](https://huggingface.co/cubed-core/camera-tracker-v1)
- [Read-trust v1](https://huggingface.co/cubed-core/read-trust-v1)

Keep your own weights under `models/local/`; Git ignores them. Native tracking
expects a manifest that validates against
[`model-artifact-manifest-v1.schema.json`](../schemas/model-artifact-manifest-v1.schema.json).
See [Decode](../docs/tutorials/DECODE.md) to use the released artifacts or
[Train tracker](../docs/TRAIN_TRACKER.md) to package your own.
