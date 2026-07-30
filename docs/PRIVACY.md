# Privacy

Cubed Core runs locally by default, but its inputs can be highly identifying.
Videos, frames, labels, BLE records, calibration material, logs, and cloud
configuration can expose people, rooms, devices, credentials, and private
research history.

`.gitignore` reduces accidental commits. It is not a publication review.

## Never post private artifacts

Do not attach or link these in a public issue, pull request, comment, CI log, or
chat:

- solve videos, audio, frames, thumbnails, labels, or dataset ZIPs
- calibration images or files from a private setup
- raw or decoded BLE, smart-cube, or phone IMU records
- faces, voices, minors, bystanders, names, locations, documents, screens, or
  reflections
- admin tokens, cookies, SSH keys, provider credentials, private
  URLs, signed links, hostnames, and instance details
- absolute paths, usernames, private repository names, or environment dumps
- model weights, checkpoints, training data, or logs without a completed rights
  and release review

A checksum proves byte identity. It does not anonymize an artifact or grant
permission to publish it.

This repository does not accept recording or label contributions. Keep those
bytes local.

## Safer recording

Reduce sensitive content before it enters a recording:

- frame tightly around the cube and hands
- use a controlled room and plain background
- keep screens, mail, photographs, and reflective objects out of view
- exclude bystanders and minors
- disable audio collection when possible
- use non-identifying recording names and notes

If a take captures unexpected sensitive material, keep it private and record a
new one. A cropped derivative does not clear the preserved original.

## Local workspace

- Keep the workbench bound to loopback.
- Treat explicit network-mode admin tokens as credentials.
- Keep `workspace/`, `datasets/`, local models, and exports out of Git.
- Use an encrypted disk and access-controlled backup for private recordings.
- Review screenshots for tabs, notifications, filenames, faces, and
  reflections.
- Do not put sensitive data in Git and rely on a later deletion. History and
  forks may retain it.

Normal same-machine browser use bootstraps local access without asking the user
to manage an admin token. This convenience does not make the service safe to
expose publicly.

The native Windows compatibility path rejects known reparse points but uses
checked paths rather than POSIX directory file descriptors. Keep its checkout
and workspace on local storage that another account cannot modify. Use WSL2 or
Linux when that local filesystem boundary is not available.

## Portable run JSON

`cubed-core/decode-result` is designed to exclude:

- video bytes
- local paths
- raw runner logs
- pickle and NPZ intermediates
- credentials and provider details
- raw teacher and sensor records

It can still reveal a scramble, moves, video SHA-256, configuration identity,
model hashes, warnings, and per-frame diagnostics. A published-dataset result
may also contain a post-decode `ground_truth_diagnostic` with a verified
reference move sequence and comparison hashes. It does not contain the raw BLE
or sensor record. Review the JSON before sharing. Share the matching video
separately only when that video has its own permission and license.

Runs verifies an exact video SHA before synchronized playback. That protects
identity, not privacy.

Deleting a terminal run moves only its job directory to recoverable workspace
trash. It does not delete the source video or labels. Manage those input files
separately.

In Add video, **Move to Trash** sends the local capture bundle and its workspace
labels to the operating system's Trash. Existing Runs remain, but they can no
longer play that video. The operating system controls retention and permanent
deletion from Trash. A registered published copy in the downloaded dataset is
not changed.

## Remote GPU

A remote Decode job copies the video and calibration to a rented or owned GPU
host for the duration of the attempt. Before using one:

- review the provider's disk, snapshot, backup, and logging policies
- use non-interactive SSH with a dedicated key
- keep the workbench and remote service on loopback
- preserve wanted output before stopping the instance
- remove or destroy the instance yourself when finished

The bundled bridge attempts to remove its staged job and capture directories
when it exits. A lost SSH connection or interrupted host can leave them behind,
so verify cleanup before stopping or releasing the instance. Provider snapshots
or logs remain outside Cubed Core's control.

## Logs and bug reports

Before sharing a failure:

1. reproduce it with a synthetic or already public fixture where possible
2. copy only the smallest relevant excerpt
3. remove tokens, URLs, hosts, IP addresses, usernames, absolute paths, hashes,
   and unrelated environment values
4. inspect the final text and screenshot as an unauthenticated stranger

Do not paste a full environment or runner log because a secret scanner found
nothing.

Apply the same standard to AI coding assistants. Unless you independently
verified a local-only boundary, treat prompts, attachments, transcripts, and
workspace indexing as disclosure to a third party.

## Public dataset

The published corpus has its own rights records, manifest, licenses, and privacy
review on
[Hugging Face](https://huggingface.co/datasets/cubed-core/cubed-data-v1).
Those approvals apply only to the exact named artifacts and immutable revision.
They do not clear a new recording, extracted frame, label, calibration, or
derivative.

## Accidental disclosure

If sensitive material is posted:

1. do not quote or mirror it
2. remove it when authorized
3. rotate exposed credentials immediately
4. report the incident through GitHub's private vulnerability channel
5. publish only a sanitized summary after containment

Removing public content may not remove caches or forks. Follow GitHub's
sensitive-data removal process when history rewriting or platform support is
required.

See [Security](../SECURITY.md), [Licensing](LICENSING.md), and
[Contributing](../CONTRIBUTING.md).
