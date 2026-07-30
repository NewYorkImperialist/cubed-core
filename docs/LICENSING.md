# Licensing

The top-level [`LICENSE`](../LICENSE) is the authoritative code license. Cubed
Core code is licensed under **GNU AGPL v3 only** (`AGPL-3.0-only`).

This page explains the repository's artifact boundaries. It does not change any
license or provide legal advice.

This page describes the tagged source-tree distribution. That distribution does
not include compiled applications, containers, wheels, emitted browser bundles,
model or checkpoint packages, training datasets, or capture-media binaries.
Separately distributed artifacts have independent inventories, notices,
checksums, provenance, and terms.

## Code

AGPL is an OSI-approved open-source license. It permits use, modification,
redistribution, commercial use, and compliant paid hosting.

When users interact over a network with a modified covered program, AGPL's
network-source provision requires that those users receive the opportunity to
obtain the Corresponding Source of that version as the license describes.
Distribution of covered binaries and modified versions brings the license's
other source, notice, and same-license conditions.

Read the primary text:

- [GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.html)
- [Open Source Initiative license list](https://opensource.org/licenses)

A process, API, or directory boundary does not automatically decide whether a
larger combination is a covered work. That is fact-specific. Get legal advice
for a particular integration.

## Code, models, and data are separate

The code license does not automatically license:

- model weights, checkpoints, or training artifacts
- datasets, videos, frames, labels, calibration, BLE, or sensor records
- third-party source or binaries
- names, logos, and trademarks
- material the repository owner lacks authority to license

Every released artifact needs its own identity, provenance, permission, and
license record.

## Models

No model bytes are included in the source-repository release. The repository
records separate artifact cards for these model families:

- [Camera tracker v1](https://huggingface.co/cubed-core/camera-tracker-v1)
- [Read-trust v1](https://huggingface.co/cubed-core/read-trust-v1)

Those cards govern only exact model artifacts that are separately made
available. They record the artifact license, hashes, origin gaps, intended use,
and limitations. A model manifest proves that bytes match the declared package.
It does not grant rights or prove quality.

The camera-model lineage described by those cards uses Ultralytics training and
export tooling. The camera model card records the exact files and the upstream
guidance used for their current `AGPL-3.0-only` label. Use in another
proprietary or hosted product needs its own review.

PyTorch checkpoints use pickle and can execute code while loading. Verify their
release checksum and load them only in an environment you trust.

## Dataset and media

No dataset or capture-media bytes are included in the source-repository
release. The repository pins a separately managed corpus at
[`cubed-core/cubed-data-v1`](https://huggingface.co/datasets/cubed-core/cubed-data-v1).
Its pinned revision records:

- `LICENSES/DATASET.md` for the collection
- per-artifact license and attribution entries in `dataset/manifest.json`
- rights records beside captures
- exact checksums

Per-artifact terms are authoritative. A collection license covers selection and
arrangement only and cannot erase a recording's own license.

The pinned corpus manifest records per-artifact Creative Commons terms. Check
the exact manifest entry before reusing any file. Those terms do not assign a
license to separate JSON fixtures merely because they derive from the same
recording.

A Creative Commons license grants only the rights it states. It does not by
itself establish subject consent or clear privacy, publicity, personality,
trademark, patent, or other third-party rights.

Dataset, recording, BLE, calibration, and benchmark cards are maintained with
the separately managed bytes. GitHub's
[`config/public-dataset-v1.json`](../config/public-dataset-v1.json) pins the
reviewed revision and file identities.

The four committed gtD1 replay fixtures—`ble-ground-truth.json`,
`decode-receipt.json`, `decode-result.json`, and `overlay-track.json` under
`apps/lab-web/public/demo/gtd1/`—are © 2026 Manas and licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Their
filename-specific descriptions and attribution are in
[`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES).

## Contributions

Code and documentation contributions use Developer Certificate of Origin 1.1
sign-off:

```bash
git commit --signoff
```

There is no CLA. The DCO covers the contributor's authority to submit the code
or documentation. It does not grant separate rights to recordings, labels,
datasets, or models.

This repository does not accept recording or label contributions. Do not attach
those artifacts to issues or pull requests.

Model proposals must identify:

- exact bytes and hashes
- source and training-data authority
- base-model and dependency licenses
- intended artifact license
- privacy and redistribution review
- limitations and missing provenance

See [Contributing](../CONTRIBUTING.md).

## Dependencies and notices

The lockfiles, generated SBOM checks, and
[`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES) record dependency evidence.
Dependency labels are inputs to review, not automatic compatibility decisions.

Do not add GPLv2-only code to this AGPL-3.0-only distribution. Do not replace
the official AGPL text with an approximate or edited copy.

## Trademarks

The software license does not grant rights to the **Cubed** or **Cubed Core**
names, logos, trade dress, or other source identifiers.

You may use the names factually to describe compatibility, origin, or a fork.
Do not imply endorsement, official status, or affiliation. A redistributed or
hosted modified version should use its own branding prominently and state that
it is based on Cubed Core.

Do not use the project name or branding in a way likely to confuse users about
who operates a service or supports a build. Ordinary nominative references,
copyright notices, and license notices remain allowed.

Rubik's is a third-party trademark owned by Spin Master. Cubed Core is an
independent project and is not sponsored by, endorsed by, or affiliated with
Spin Master or the Rubik's brand. The project uses the generic term “3x3 puzzle
cube” for the product category. See the
[official brand-use guide](https://www.rubiks.com/brand-use-guide).

See [Privacy](PRIVACY.md) for personal-data boundaries and
[Security](../SECURITY.md) for private reporting.
