# Contributing to Cubed Core

Cubed Core welcomes focused code, test, documentation, and reproducible research
contributions.

This repository does not accept recording or label contributions. Do not attach
or link videos, frames, labels, calibration, BLE, IMU, dataset exports, model
binaries, credentials, or private logs in issues or pull requests.

## Before you start

1. Search existing issues.
2. Propose a small change before implementing a large feature.
3. State the exact problem, intended behavior, evidence, and limitations.
4. Use only synthetic, already public, or expressly cleared fixtures.

Good first contributions include:

- minimal, sanitized bug reproductions
- focused setup or UI fixes
- schema and validation improvements
- documentation corrections
- tests for an existing contract
- carefully documented negative research results

## Development setup

The complete development and verification workflow uses a POSIX shell and Make
on macOS, Linux, or WSL2. Native Windows is experimental and may be broken.
Its limited CPU-workbench path and current verification gap are documented in
the [README](README.md). Use WSL2 before submitting a change.

Requirements are Python 3.10 through 3.12, Node 22.3 or newer on the Node 22
line, FFmpeg, Git, Make, and the pinned `uv` version.

```bash
./setup.sh --install-uv
source .venv/bin/activate
```

Use `make dev` for API and frontend hot reload. Use `make workbench` for the
built single-port app.

## Verification

Run before every commit:

```bash
make check
```

For any web change, also run:

```bash
npm --prefix apps/lab-web run test:e2e
```

Report exact commands and results in the pull request. Do not quote metrics from
memory.

## Change expectations

- Keep changes focused and leave unrelated work alone.
- Preserve existing artifact and schema contracts unless the change explicitly
  updates them.
- Fail visibly when a dependency, model, input, or evidence requirement is
  missing.
- Add the narrowest useful automated test.
- Update the one maintained guide for the affected function.
- Keep teacher data out of camera-only inference.
- Distinguish execution, reconstruction, and evaluation claims.
- Record negative results when they rule out a real approach.

Do not mix formatting-only refactors into a functional change.

## Data and privacy

GitHub is public. Read [Privacy](docs/PRIVACY.md) before posting text,
screenshots, logs, or fixtures.

Keep these outside Git:

- `workspace/`, datasets, captures, annotations, and exports
- `.env`, `.env.hub`, tokens, SSH keys, and provider details
- downloaded models, checkpoints, and training data
- raw runner logs and private research receipts

`.gitignore` is a safety net, not permission to publish.

The public Hugging Face dataset has its own immutable manifest, rights records,
and licenses. Those approvals apply only to the exact published artifacts. See
[Dataset](docs/DATASET.md).

## Model proposals

Open a feature proposal before integrating another model. Do not attach weights
or checkpoints. State its role, interface, public provenance, licenses,
evaluation boundary, and limitations. Released artifacts need their own rights,
privacy, and redistribution review. The
[camera tracker](https://huggingface.co/cubed-core/camera-tracker-v1) and
[read-trust](https://huggingface.co/cubed-core/read-trust-v1) cards show the
expected metadata.

## Research claims

Use [Evidence](docs/EVIDENCE.md) for claim language and
[Research notes](docs/RESEARCH_NOTES.md) for prior dead ends.

Any new measurement must bind:

- exact input identities
- source commit
- configuration and `CFG_HASH`
- model identities
- command or runner
- result and receipt
- metric definition
- known limitations

Camera-only prediction must close before teacher truth opens. For an ordinary
completed result, use the exact statements “The job completed on this input.”
and “The sequence replayed to solved.” Neither statement is an accuracy or
reach-LL result.

## Developer Certificate of Origin

Every commit requires a Developer Certificate of Origin 1.1 sign-off:

```bash
git commit --signoff
```

The sign-off certifies that you wrote the contribution or have the right to
submit it under the project's license, and that the contribution and sign-off
are public records.

To amend a local commit:

```bash
git commit --amend --signoff --no-edit
```

There is no CLA. The DCO does not grant rights to recordings, datasets, or model
artifacts.

## Pull request

Explain:

- the smallest useful change and why it belongs here
- user-visible and schema impact
- exact verification commands
- evidence and privacy review
- limitations left behind

Code is [AGPL-3.0-only](LICENSE). Models and data have separate terms. See
[Licensing](docs/LICENSING.md).

Report security issues privately through [SECURITY.md](SECURITY.md).
