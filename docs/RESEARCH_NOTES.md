# Research notes

This is the repository's one research history. It records the current design,
the ideas that materially helped, and the approaches that were tried and set
aside. It is not a benchmark report and contains no remembered metrics.
These qualitative summaries describe the tested formulations. They do not
establish generalization.

Measured benchmark reports and capture narratives live with the public data on
[Hugging Face](https://huggingface.co/datasets/cubed-core/cubed-data-v1/tree/7fae604962c590ac9c658ba6ee0350e86de9c4f5/benchmark).

## Current design

The decoder is a legal-state search, not an end-to-end move classifier.
Learned models locate faces and weight camera reads. Deterministic cube
transitions, whole-cube orientation, trellis search, and endpoint replay make
the final decisions.

The standard pipeline:

1. samples visible face colors and alignment evidence per frame
2. derives motion events and settled spans
3. runs a first legal-state pass to gather margins and orientations
4. freezes gate evidence and runs a second pass
5. scrubs weak reads and rescues uncertain spans
6. uses bounded last-layer priors and endpoint search
7. emits a result only under the structured result contract

For a named closed research evaluation, the primary metric is reach-LL: the raw
state path must reach last-layer onset with the correct pre-last-layer state,
up to whole-cube orientation. Solved endpoint and edit distance are
diagnostics.

## Approaches that helped

### Legal search with learned soft evidence

The durable division of labor is:

- learned models propose pose, alignment, and trust
- color and motion enter as attributable soft evidence
- legal cube state and orientation stay explicit
- the search decides the move history

End-to-end move predictors tested in this project did not replace the search.
Learning was more useful for scoring hypotheses than for owning cube legality.

### Two-pass gating

A first pass can gather alignment-break margins, move-gate margins, and
orientation evidence. Freezing those statistics before rebuilding spans made
the second pass more stable than changing thresholds while the same search was
running.

### Settled spans and read scrubbing

Reasoning over settled evidence is more reliable than treating every frame as
equally informative. Keeping multiple chronological reads within a span, then
selecting useful evidence during transition scoring, retained information that
median or majority collapse lost.

### Soft read trust

The read-trust model weights individual cell evidence without deleting it. A
floor keeps weak observations available. This worked better than a global
confidence cutoff that could remove the only useful view of a sticker.

### Explicit orientations

Maintaining the 24 whole-cube orientations inside the search prevents regrips
from being mistaken for face turns.

### Physically motivated calibration

Color calibration and a constrained illumination model transferred better than
per-capture threshold tuning. Calibration still remains setup-specific.

## Tried and set aside

These are real dead ends under the tested formulations. A materially different
formulation can still be worth testing, but repeating the same premise needs new
evidence.

### Outcome-tuned global thresholds

Thresholds selected to improve final decode results transferred poorly. A gain
on one capture often moved the failure elsewhere or admitted more plausible
wrong states. Production thresholds should come from geometry or separately
labeled low-level measurements.

### Hard move-count constraints

Motion and audio event counts contain misses and extras. Treating a count as
exact could deadlock an otherwise recoverable path. Counts remain useful as
soft evidence only.

### Smart-cube gyro as a Decode input

Frame-aligned smart-cube orientation samples were tested for seeding cube
orientation and locating regrips. That path required a separately synchronized
sensor sidecar, so it was not a deployable camera-only input. The maintained
Decode request does not consume gyro or IMU data. This records the tested
integration boundary. It does not establish that inertial cues are useless.

### Full reconstruction from motion alone

Motion timing and direction cannot uniquely identify a long legal history.
Many move sequences share similar activity and endpoints. Motion is useful for
event support and short bridges, not as a full decoder.

### One orientation for an entire span

A settled span can cross a regrip. Forcing one orientation pooled incompatible
views. Chronological orientation alternatives worked better.

### Hard read aggregation

Collapsing a span to one median, majority color, or globally gated read removed
the diversity needed to resolve state and orientation. Soft per-read scoring
was more robust.

### Mismatched data growth

Adding recordings from a different capture distribution did not reliably
improve the tested event model. More data is not automatically better. Capture
conditions, grouping, and leakage-safe splits matter.

### In-setup verifier success

A learned verifier passed its frozen in-setup exam but did not hold the same bar
on a fresh held-out recording set. It was not integrated. Fresh-data
replication remains a promotion gate.

### Diagonal color constancy as a universal fix

Per-channel gains model some illumination changes but cannot express arbitrary
color-space rotation. Retuning the same model cannot repair every lighting
shift.

### Pose interpolation without correspondence

Interpolating sparse poses did not resolve symmetry-branch jumps. Continuous
pose needs frame-to-frame correspondence and the known discrete cube
orientation. Interpolation alone provides neither.

### Implicit clip time origins

A clip can retain timestamps from its parent recording. Treating the first
retained event as time zero shifted generated frame labels. Clip origin and
frame mapping must be explicit and verified.

### Metric-to-mechanism substitution

Activity or directional accuracy does not prove that a signal retains move
identity. A detector used both to size a search and localize events needs
separate evidence for each role.

### Oracle-pinned anchors

Ground-truth anchors can bound feasibility, but they cannot prove the deployable
anchor producer works. Oracle results must stay labeled as oracle diagnostics.

### Post-hoc bridge and repair passes

Postdictive bridges, anchor-to-anchor rewrites, and standalone window-repair
passes added a second decision path after the main search. Under the tested
formulations they did not justify that extra surface. The maintained decoder
keeps one sequential scrub path instead.

### Auxiliary per-window visual scorers

Several optional render, center-motion, slab, and predict-verify terms were
tested as additive window scores. They stayed outside the canonical
configuration and were removed rather than presenting dormant alternatives as
supported features.

### Speed changes without parity

Changing video decode, resize, batching, frame delivery, or numeric backends can
change pixels, frame counts, and downstream evidence. The higher-level NVDEC
`SimpleDecoder` path crashed on a tested public video even though lower-level
NVDEC decoding worked. The current probe isolates native decode in a child and
falls back under `auto`.

A speedup is a behavior change until structural and semantic parity are checked
on the exact input.

### Footage-derived centroids replacing calibration

Estimating color centroids from the same solve can reinforce the decoder's own
mistakes and hide distribution shift. A separately collected calibration prior
remains the honest input.

## Why 120 fps remains the tested profile

Frame-rate plumbing is explicit, but the decision statistics were measured at
120 fps. Scaling them arithmetically to 240 fps would be a guess.

The research policy is:

- treat 120 fps and a 1080-pixel short edge as the recommended, measured profile
- allow known readable nonstandard timing and dimensions with visible warnings
- preserve 240 fps originals
- create a separately hashed every-other-frame 120 fps derivative
- treat native 240 fps as a new research regime that needs new measurements

Faster solves remain a weak area because turn events overlap and settled
evidence becomes sparse.

## Open problems

- Generalization beyond the maintainer's cubes, cameras, hands, backgrounds,
  and lighting.
- Robust calibration transfer across recording cohorts.
- Fast turns and regrips with limited settled frames.
- Native 240 fps evidence measured from a dedicated corpus.
- Stronger pose continuity without importing ground truth.
- Fresh-data replication for learned trust and event models.

## Research discipline

- Run canonical decode and evaluation through the repository's required
  workflows.
- Record exact input, commit, configuration hash, model hashes, and receipt.
- Keep teacher data closed until camera-only output is frozen.
- Never quote a metric from memory.
- Say “The named closed evaluation reached its reported metric.” only when the
  checked report and score artifact support it.
- Preserve abstentions and negative results.
- Separate execution checks from quality claims.
- Treat a change that affects pixels, timing, or numeric order as semantic
  until parity is demonstrated.

See [Evidence](EVIDENCE.md) for claim language and
[Decode](tutorials/DECODE.md) for the maintained execution path.
