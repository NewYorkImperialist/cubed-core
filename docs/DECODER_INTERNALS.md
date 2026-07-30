# Decoder internals

This is a design overview of the research decode pipeline that lives in this
repository at top level under `detect/`, `core/`, `analysis/`, and `scripts/`.
It explains what each module does and how the pieces connect, so that an outside
reader can follow a capture all the way to a reconstructed move sequence.

The pipeline turns a captured solve (a video plus per-frame camera reads) into
the most-likely sequence of cube moves. It does this with a marginalized trellis
search over cube states and orientations, scored by color evidence and a learned
read-trust model. It is not an end-to-end network. The learned parts act as
visual likelihood verifiers that score hypotheses. The search itself is
deterministic and makes the decisions.

## Layout

```
detect/     vision, ONNX, geometric reads, the trellis tracker, scrub decode,
            read-trust, motion evidence, and last-layer completion
core/       pure cube runtime: cube state, faces, nodes, perf tracing
analysis/   deterministic last-layer analysis + data/ (alg sets, OLL/PLL cases)
scripts/    the trellis_gt decode runner, the video→reads bridge, calibration
            and GPU helpers, and the run_research_*.sh entry points
```

`core/` has no dependency on `detect/`. It is the cube algebra the rest of the
pipeline builds on. `detect/` and `scripts/` import `core/`. `analysis/`
provides last-layer priors consumed during decode.

## The three load-bearing modules

Three modules carry the decode:

- **`detect/trellis_tracker.py`** is the trellis tracker. Given per-frame camera
  reads and motion/orientation events, it builds absolute color-evidence
  segments (`AbsSegment`), enumerates candidate cube states and the 24 cube
  orientations, and runs a beam search (`run_transition`) whose transitions pay
  move, count, and orientation-switch costs. It is the core state-recognition
  engine: two-pass tracking (`track_2pass`), the move gate and z-normalized gate
  margins, per-frame orientation-map production, anchor-fit harvesting, and the
  long-gap orientation-switch logic all live here.

- **`detect/scrub_decode.py`** is the scrub decode layer. It refines the trellis
  output with bounded sequential windows, chronological read scoring,
  orientation continuity, endpoint completion, and conservative abstention.
  `detect/scrub_span_view.py` supplies its minimal per-span read projection.

- **`scripts/trellis_gt.py`** is the decode runner. It assembles the full flag
  surface, loads reads/events/alignment artifacts, drives the tracker and scrub
  layers, and prints the decoded sequence plus diagnostics.
  `scripts/run_research_decode.sh` wraps it with the versioned configuration so
  standard runs use one entry point rather than hand-written flags.

## How a decode flows

1. **Reads and events in.** The decoder consumes per-frame camera reads
   (CIE-Lab cell colors, confidences, visibility, pose-derived slots) plus
   motion and orientation events. These can be produced ahead of time or by the
   video→reads bridge below.

2. **Segments.** `AbsSegment` construction (`trellis_tracker.py`) collapses the
   reads into orientation-independent absolute color evidence per span, scored
   against the six calibrated cube colors with a weighted CIE-Lab distance
   (`detect/color_distance.py`) and the read-trust model
   (`detect/read_trust.py` / `trust_numpy.py`).

3. **Pass one.** The tracker runs a first trellis pass from the full event set,
   producing per-event move-gate and alignment-break margins and a per-frame
   orientation map resolved from the previous retained-layer leader. It measures
   the exact event-gap margin and harvests anchor fits after ranking.

4. **Hard gate + pass two.** The frozen pass-one samples derive effective move
   and orientation-prior gate thresholds (z-normalized). A hard gate decides
   which events survive. Spans are rebuilt from the kept events and the trellis
   runs again with a fresh session.

5. **Scrub / rescue.** `scrub_decode.py` applies misfit-gated rescue, deep
   re-anchoring, and retry over the two-pass result, tightening spans where the
   base search was uncertain.

6. **Last-layer priors.** `analysis/` supplies OLL/PLL and alg-set priors
   (`ll_alg_prior`) that bias the final layers toward known algorithms. The alg
   tables live under `analysis/data/`.

7. **Result.** The runner emits a candidate decoded move sequence. For a
   completed API result, the server replays it from the sealed scramble. The
   supported claim is “The sequence replayed to solved.” A named closed
   evaluation can separately measure reach-LL against held-back truth. A normal
   result document does not establish that metric.

## Video→reads bridge

The four decode modules operate on pre-existing reads/events/alignment
artifacts. To let a user go from their own captured video to those artifacts,
three generators in `scripts/` form the video→reads bridge, wired by
`scripts/run_research_reads.sh`:

| Module | Role |
| --- | --- |
| `scripts/geo_read.py` | video → reads (pose + warp + calibrated Lab sampling) |
| `scripts/extract_alignfeat.py` | video → per-frame alignment features `.npz` |
| `scripts/gen_motion_events.py` | reads → rest-anchored camera-motion events |

These lean on the cell-geometry, occlusion-mask, and GPU helpers in `scripts/`
(`cell_common.py`, `occ_mask.py`, `gpu_reads.py`, `gpu_decode.py`, and
`onnx_dynamic_batch.py`). The bridge stages its outputs under the names the
decode runner expects, then hands off to Decode.

`run_research_reads.sh` is a dry run by default (`CUBED_RESEARCH_DECODE_EXECUTE=1`
to invoke), and real standard runs belong on a CUDA machine. The canonical
reads stage requires `torch.cuda` and the ONNX Runtime CUDA provider and stops
instead of falling back to CPU. NVDEC may fall back to host video decoding, but
that is only a video-I/O fallback. It does not make the standard
`local_camera_v1` pipeline CPU-capable.

## Learned parts

The default decode uses the pose and alignment ONNX models (via
`detect/onnx_runtime.py`) and the read-trust NumPy model as visual verifiers.
The legal-state search remains deterministic. GPU libraries accelerate the
same reads and scoring path. They do not introduce a separate decoder.

## The default configuration

`scripts/run_research_decode.sh` assembles the standard OSS configuration: the
standard decode flag set (base + move-gate + d-gap-trust + scrub + robust) minus
`--final-from-gt`. The app supplies the starting scramble, and the runner uses
the fixed solved task target. No ground-truth data is needed at runtime. The
runner defaults to a dry run and just prints the command. Set
`CUBED_RESEARCH_DECODE_EXECUTE=1` to actually run it. Without the optional
third-party backends and rights-cleared weights installed, it stops at the
first missing dependency. The exact configuration is recorded in
`config/decode-runtime-v1.json`
(`profiles.local_camera_v1.cfg_hash`), and the runner prints the same hash, so
you can confirm that both sides name the same configuration. The hash does not
establish execution or quality. The pose and alignment ONNX weights are
separate release assets. Canonical preflight lists any missing files. The full
runner is invoked from the shell.

## API integration

The API does not contain a second decoder. `native_decode_runner.py` invokes
the same reads, motion, alignment, and decode scripts described above, then
stores their portable result. `decode_jobs.py` owns submission and job state,
while Runs only reads completed artifacts.

The packaged cube runtime handles notation, state transitions, orientation,
and server replay. The packaged vision modules support tracking, calibration,
and Label. Neither replaces the research pipeline.

## Data assets and rights

The `--ll-prior` last-layer prior loads alg tables from `analysis/data/`. The
shipped case tables under `analysis/data/algsets/*.json` are MIT-licensed
(© 2024 Spencer Chubb, `spencerchubb/cubingapp`), recorded in
`THIRD_PARTY_NOTICES`. The `oll_cases.json` / `pll_cases.json` recognition
fingerprints are original to this project (AGPL-3.0-only). The prior degrades
gracefully when an optional table is absent.

## Honest limitations

- Decode is read-quality-bound because the search can rank only states supported
  by available color evidence. Weak or occluded reads can limit its hypotheses.
  This does not show that search errors are absent or secondary on another
  input.
- The pose and alignment ONNX weights and some calibration artifacts are not
  bundled. A real decode needs them plus the optional backends installed.
- The scrub layer's per-burst refinement is the pipeline's speed bottleneck.
- For a named closed research evaluation, the primary metric is reaching
  last-layer onset in the correct pre-LL state. Full-sequence accuracy is a
  separate diagnostic.
