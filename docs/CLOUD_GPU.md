# GPU setup

Live Decode requires NVIDIA CUDA. The base workbench and Demo do not.

The Decode page uses one **Compute target** selector:

- **Local CUDA (this workbench)** runs on the machine hosting the API.
- Configured remote or Vast.ai hosts use the bundled SSH bridge.

Both choices produce the same `cubed-core/decode-result` v1 artifact. Runs is
read-only and never starts compute.

## Choose a topology

| Setup | Workbench | CUDA job | Best for |
| --- | --- | --- | --- |
| Local GPU | Linux CUDA machine | Same machine | Desktop or workstation with NVIDIA GPU |
| Remote GPU | macOS, Linux, or WSL2 laptop | Rented or owned Linux CUDA host | Most contributors |
| Workbench on GPU host | Remote Linux CUDA host through SSH tunnel | Same remote host | Advanced single-host operation |

Native Windows is CPU-workbench only and cannot invoke either bundled Decode
runner. On Windows, use WSL2 or Linux for live local Decode and the SSH bridge.

The pipeline saturates one device and the service runs one Decode job at a
time. A host being listed is configuration, not proof of readiness or
availability.

## Local CUDA

Requirements:

- Linux with a supported NVIDIA driver
- Python 3.10 through 3.12
- FFmpeg and FFprobe
- the runtime and decode-support assets

Prepare the environment:

```bash
make download-assets
make bootstrap-research-gpu
source .venv-decode-gpu/bin/activate
nvidia-smi
```

Start the workbench with native Decode enabled:

```bash
CUBED_CORE_DECODE_MODE=native make workbench-decode-gpu
```

Open Decode and keep **Local CUDA (this workbench)** selected. Preflight reports
missing assets, unreadable or unknown capture metadata, and missing FFmpeg
before submission. Use the local or remote doctor for CUDA, PyTorch, and ONNX
Runtime provider checks. The job also fails visibly if those runtime
dependencies are unavailable. Known readable nonstandard rates and resolutions
remain warnings.

The reads stage requires `torch.cuda` and ONNX Runtime's
`CUDAExecutionProvider`. It stops rather than silently falling back to CPU.

## Remote GPU with Vast.ai or SSH

The repository includes one idempotent provisioner for a Linux GPU box:

```text
scripts/provision_gpu_box.sh
```

It checks non-interactive SSH, requires the runtime manifest, read-trust model,
and shared calibration locally, then syncs the checkout and release assets. It
creates the pinned GPU environment on the box, verifies CUDA and model routing,
writes `workspace/provision-receipt.json`, and prints the workbench environment
block. It never stops or destroys the provider instance.

The provisioner currently targets a Debian/Ubuntu-style image with `dpkg` and
`apt-get`, and expects the SSH user to have root privileges without `sudo`.
Both the workbench host and the initial remote image must already provide
`rsync`. It is needed before the provisioner can install on-box packages.
Vast.ai's common root-login templates fit that contract.

### 1. Prepare the workbench host

```bash
./setup.sh --install-uv
make download-assets
```

Register an SSH public key with the provider. Verify that the rented host
accepts non-interactive authentication:

```bash
ssh -p <port> -o BatchMode=yes root@<host> true
```

For Vast.ai, attach the key through Vast before provisioning. The instance must
offer an NVIDIA GPU and enough disk for the checkout, assets, environment, and
temporary video staging.

### 2. Provision the box

Run from the repository root on the workbench host:

```bash
./scripts/provision_gpu_box.sh \
  --dest root@<host> \
  --port <port> \
  --append-env
```

`--append-env` writes or refreshes the generated block in ignored `.env.hub`.
The remote checkout defaults to `/workspace/cubed-core`. Override it with
`CUBED_REMOTE_ROOT` when needed.

The fetched receipt is:

```text
workspace/provision-receipt.json
```

Inspect it before submitting data. A successful receipt records that the
provisioner's named mechanical checks passed on that host. It does not show
that a Decode job completed on a real input, and it does not establish model
quality, accuracy, or provider identity.

### 3. Start the local workbench

`make hub` loads `.env.hub` and starts the normal loopback workbench:

```bash
make hub
```

Open Decode. The environment-configured host appears beside
**Local CUDA (this workbench)**. Select it and run preflight.

You can also export the printed environment block manually, then start:

```bash
CUBED_CORE_DECODE_MODE=native make workbench
```

### 4. Verify before a long run

```bash
make remote-doctor
```

This is a non-destructive readiness check, not a zero-write probe. It creates a
tiny synthetic clip locally, stages it in a temporary remote directory to test
video decode, and attempts to remove that directory afterward. Verify cleanup
if the connection drops. A real Decode attempt remains the only end-to-end
check on the selected video.

Per attempt, `scripts/remote_decode_runner.sh`:

1. copies the video and calibration to a job-local remote directory
2. rewrites only the request paths for the remote checkout
3. re-hashes runtime assets on the remote host
4. runs the native pipeline with flushed progress markers
5. copies the result back to the workbench
6. attempts to remove the staged remote job and capture directories

Models are not uploaded per job. A digest mismatch stops the attempt. A lost
connection can prevent cleanup, so inspect the host before releasing it.

## Multiple named hosts

To show more than one remote option, create ignored
`workspace/remote-hosts.json` on the workbench host:

```json
{
  "schema": "cubed-core/remote-hosts-v1",
  "schema_version": 1,
  "hosts": [
    {
      "id": "vast-4090",
      "label": "Vast RTX 4090",
      "ssh_dest": "root@203.0.113.7",
      "ssh_port": 40122
    },
    {
      "id": "office-gpu",
      "label": "Office GPU",
      "ssh_dest": "cubed@10.0.0.5",
      "ssh_port": 22,
      "root": "/workspace/cubed-core",
      "decode_venv": ".venv-decode-gpu",
      "nvdec": "auto"
    }
  ]
}
```

Host IDs are lowercase slugs. `environment` is reserved for the single host
configured through `CUBED_REMOTE_*` environment variables.

The selector always keeps **Local CUDA (this workbench)**. It never silently
replaces local execution with the first remote entry.

## NVDEC

CUDA inference and NVIDIA video decode are separate capabilities. The default
policy is:

```bash
export CUBED_NVDEC=auto
```

`auto` probes the exact video in a child process. It uses NVDEC when compatible
and otherwise continues with host video decode while keeping GPU inference.

Other policies:

```bash
export CUBED_NVDEC=off
export CUBED_NVDEC=require
```

Use `off` to skip the probe. Use `require` only when fallback must be rejected.
For a remote host, use `CUBED_REMOTE_NVDEC` or the `nvdec` field in
`remote-hosts.json`.

Some fractional cloud GPU rentals do not expose NVDEC. That is not fatal under
`auto`. Rent a whole GPU when hardware video decode is important.

No speed figure in this repository is a performance guarantee. Video codec,
resolution, provider load, storage, and GPU model all matter.

## Workbench on the remote host

Advanced users can run the complete workbench on the CUDA host and tunnel its
loopback port:

```bash
# On the remote host
source .venv-decode-gpu/bin/activate
CUBED_CORE_DECODE_MODE=native make workbench-decode-gpu
```

```bash
# On the local computer
ssh -N -L 8000:127.0.0.1:8000 root@<host> -p <port>
```

Open `http://127.0.0.1:8000/` locally. Keep the service bound to remote
loopback. Do not expose the workbench directly to the public Internet. The
local admin token is capability-style access, not production authentication.

When the workbench itself runs on the CUDA host, choose
**Local CUDA (this workbench)** because local now means that remote machine.

## Security and data handling

- Use a dedicated SSH key and `BatchMode=yes`.
- Keep `.env`, `.env.hub`, SSH keys, provider tokens, instance addresses, and
  receipts out of Git.
- Treat the remote runner as trusted same-user code. It is not sandboxed.
- Review instance disks, snapshots, logs, and provider retention before upload.
- Stop or destroy the provider instance yourself after preserving wanted
  results. Cubed Core never performs provider deletion.
- Do not paste raw runner logs or provider details into public issues.

See [Privacy](PRIVACY.md) and [Evidence](EVIDENCE.md).

## Troubleshooting

### The host is listed but unavailable

The JSON or environment parsed, but SSH, CUDA, assets, or capacity may still be
wrong. Run `make remote-doctor`, then inspect the attempt's bounded failure.

### Provisioning cannot connect

Verify the exact user, host, SSH port, and registered public key:

```bash
ssh -p <port> -o BatchMode=yes root@<host> true
```

An existing Vast instance may need the key attached after creation.

### CUDA is visible but Decode is blocked

`nvidia-smi` alone is insufficient. Confirm the GPU environment was bootstrapped
and ONNX Runtime exposes `CUDAExecutionProvider`. Re-run the idempotent
provisioner or local bootstrap.

### The selector does not show a configured host

Restart the workbench after changing environment variables. For
`remote-hosts.json`, check its schema, lowercase unique IDs, SSH ports, and that
the file is a regular file inside the active workspace.

### NVDEC fails

Keep `auto` unless hardware decode is a hard requirement. Host video decode
fallback does not disable CUDA inference.

### A remote attempt fails after upload

Runs preserves the terminal attempt. Review the local job directory and trusted
local log. The persisted public-facing failure intentionally excludes raw
runner output.

For the run workflow and artifact semantics, see
[Decode and Runs](tutorials/DECODE.md).
