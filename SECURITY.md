# Security policy

Cubed Core is a local research workbench. It is not a hosted service and does
not offer a security bounty.

## Supported versions

| Version | Support |
| --- | --- |
| Latest release | Supported |
| `main` | Best-effort review |
| Older releases | Unsupported |

## Report privately

Use GitHub's **Report a vulnerability** action for this repository. Do not open
a public issue, discussion, or pull request containing:

- vulnerability or exploit details
- credentials or private URLs
- recordings, labels, calibration, BLE, models, or datasets
- private paths, hosts, provider details, or participant information

If private reporting is unavailable, open a normal issue asking the maintainer
to contact you. Include no vulnerability detail.

[GitHub private vulnerability reporting](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)

## Include

Use synthetic or expressly cleared inputs. Provide only:

- affected commit and component
- impact and realistic attack conditions
- minimal reproduction or proof of concept
- sanitized configuration
- whether the issue is already public
- a suggested fix when available

Reproduce artifact-handling problems with synthetic fixtures.

## Scope

In scope:

- repository source and default configuration
- packaging and first-party dependency integration
- local API and browser workbench
- workspace and artifact validation
- authentication, media tickets, and remote-runner boundaries

Not a security report by itself:

- disagreement with model or decoder quality
- ordinary setup failures or missing features
- exposing a loopback-only workbench to the public Internet
- issues present only in an unmodified third-party project
- requests to send private recordings, labels, datasets, or models

Test only systems and data you are authorized to test. Do not access another
workspace, degrade availability, retain private data, or publish an exploit
before coordinated review.

## Known advisory exception

`npm audit` reports
[`GHSA-qwww-vcr4-c8h2`](https://github.com/advisories/GHSA-qwww-vcr4-c8h2)
for the pinned `react-router` 7.18.2. The affected path uses unstable React
Server Component APIs. Cubed Core is a client-only Vite app and imports no RSC
API. Revisit this exception if the workbench adds server rendering or RSC.

## Handling

The maintainer triages reports on a best-effort basis and may open a private
GitHub advisory. No response time, remediation, bounty, credit, or embargo is
promised.

See [Privacy](docs/PRIVACY.md) and [Contributing](CONTRIBUTING.md).
