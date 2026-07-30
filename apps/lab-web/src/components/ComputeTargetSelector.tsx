import {
  ROUTE_GUIDE_LINK_REL,
  ROUTE_GUIDE_LINK_TARGET,
  routeGuideDocumentHref,
} from "../routeGuideLinks";
import type { GpuCapability, RemoteHost } from "../types";
import "./ComputeTargetSelector.css";

function localCudaSummary(gpu: GpuCapability | null): string {
  if (gpu === null) return "Checking CUDA on this workbench…";
  if (!gpu.available) {
    return gpu.reason
      ? `Local CUDA unavailable: ${gpu.reason}`
      : "Local CUDA unavailable on this workbench.";
  }
  const device = gpu.devices[0];
  if (!device) return "CUDA detected on this workbench.";
  const memory =
    typeof device.memory_mib === "number"
      ? ` · ${Math.round(device.memory_mib / 1024)} GiB`
      : "";
  const extra = gpu.devices.length > 1 ? ` · ${gpu.devices.length} GPUs` : "";
  return `${device.name}${memory}${extra}`;
}

/**
 * Selects where one decode job runs. Empty is an intentional value: it means
 * Local CUDA and omits remote_host from the request. Configured remote hosts
 * are opt-in and never replace local as an automatic fallback.
 */
export function ComputeTargetSelector({
  hosts,
  selectedId,
  onChange,
  localGpu,
  idPrefix,
  disabled = false,
  loading = false,
  error = "",
}: {
  hosts: RemoteHost[];
  selectedId: string;
  onChange: (next: string) => void;
  localGpu: GpuCapability | null;
  idPrefix: string;
  disabled?: boolean;
  loading?: boolean;
  error?: string;
}) {
  const selectedHost =
    hosts.find((host) => host.id === selectedId) ?? null;
  const noteId = `${idPrefix}-compute-target-note`;
  const localGuideId = `${idPrefix}-local-gpu-guide`;
  const remoteGuideId = `${idPrefix}-remote-gpu-guide`;
  const localUnavailable = localGpu !== null && !localGpu.available;

  return (
    <div className="compute-target-control">
      <select
        className="compute-target-select"
        id={`${idPrefix}-compute-target-select`}
        value={selectedId}
        disabled={disabled}
        aria-describedby={noteId}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="" disabled={localUnavailable}>
          Local CUDA (this workbench)
        </option>
        {hosts.map((host) => (
          <option key={host.id} value={host.id}>
            {host.label}
          </option>
        ))}
      </select>

      <small className="compute-target-note" id={noteId}>
        {selectedHost
          ? "Remote target configured · run readiness to verify it."
          : localCudaSummary(localGpu)}
      </small>

      {loading && (
        <small className="compute-target-message" role="status">
          Checking configured remote hosts…
        </small>
      )}
      {error && (
        <small className="compute-target-message compute-target-error" role="alert">
          Remote hosts could not be loaded: {error}
        </small>
      )}
      {!loading && !error && hosts.length === 0 && (
        <small className="compute-target-message">
          No remote hosts configured. Add{" "}
          <code>workspace/remote-hosts.json</code> as described in{" "}
          <a
            href={routeGuideDocumentHref("docs/CLOUD_GPU.md")}
            target={ROUTE_GUIDE_LINK_TARGET}
            rel={ROUTE_GUIDE_LINK_REL}
          >
            the GPU guide
          </a>
          .
        </small>
      )}
      {disabled && (
        <small className="compute-target-message">
          This job keeps the compute target it started with.
        </small>
      )}

      <details className="compute-target-setup">
        <summary>
          <span>
            <strong>Set up a GPU</strong>
            <small>Local CUDA or remote / Vast.ai</small>
          </span>
        </summary>

        <div className="compute-target-setup-body">
          <section aria-labelledby={localGuideId}>
            <header>
              <span>On this machine</span>
              <strong id={localGuideId}>Local CUDA</strong>
            </header>
            <ol>
              <li>
                <span>Prepare the runtime</span>
                <code>make download-assets</code>
                <code>make bootstrap-research-gpu</code>
              </li>
              <li>
                <span>Start the CUDA workbench</span>
                <code>source .venv-decode-gpu/bin/activate</code>
                <code>
                  CUBED_CORE_DECODE_MODE=native make workbench-decode-gpu
                </code>
              </li>
            </ol>
          </section>

          <section aria-labelledby={remoteGuideId}>
            <header>
              <span>Rented or owned host</span>
              <strong id={remoteGuideId}>Remote / Vast.ai</strong>
            </header>
            <ol>
              <li>
                <span>Rent a CUDA Linux host and attach your SSH public key.</span>
              </li>
              <li>
                <span>Verify non-interactive SSH</span>
                <code>
                  ssh -p &lt;port&gt; -o BatchMode=yes root@&lt;host&gt; true
                </code>
              </li>
              <li>
                <span>Provision and save the connection</span>
                <code>
                  ./scripts/provision_gpu_box.sh --dest
                  root@&lt;host&gt; --port &lt;port&gt; --append-env
                </code>
              </li>
              <li>
                <span>Start this workbench</span>
                <code>make hub</code>
              </li>
              <li>
                <span>
                  Select the configured target above, then use Refresh readiness
                  before Run decode.
                </span>
              </li>
            </ol>
          </section>

          <p className="compute-target-setup-footnote">
            These live Decode commands require Linux or WSL2. A listed target
            is configured, not proven ready. Readiness checks SSH, CUDA, and
            required assets.{" "}
            <a
              href={routeGuideDocumentHref("docs/CLOUD_GPU.md")}
              target={ROUTE_GUIDE_LINK_TARGET}
              rel={ROUTE_GUIDE_LINK_REL}
            >
              Open the full GPU guide
            </a>
            .
          </p>
        </div>
      </details>
    </div>
  );
}
