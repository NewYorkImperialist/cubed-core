import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  RequestError,
  fetchCapabilities,
  fetchCaptures,
} from "../api";
import {
  DecodePreflight,
  fetchDecodeJobResult,
  fetchDecodeJobResultBlob,
  fetchDecodeJobStatus,
  fetchDecodeJobsForCapture,
  fetchDecodePreflight,
  submitDecodeJob,
} from "../decodeApi";
import type {
  Capabilities,
  CaptureReceipt,
  DecodeJobStatus,
  DecodeResultDocument,
  RemoteHost,
  DecodeJobState,
} from "../types";
import { formatFinishedAt, isTerminalDecodeState } from "../decodeJobContract";
import {
  decodeStageChips,
  decodeStageProgressLabel,
} from "../decodeStages";
import { captureHref, captureIdFromParams } from "../captureSelection";
import { useRemoteHostSelection } from "../remoteHosts";
import {
  ROUTE_GUIDE_LINK_REL,
  ROUTE_GUIDE_LINK_TARGET,
  routeGuideDocumentHref,
} from "../routeGuideLinks";
import { ComputeTargetSelector } from "./ComputeTargetSelector";
import "../labTools.css";
import "./DecodeStage.css";

const JOB_POLL_INTERVAL_MS = 750;

function isAbortError(reason: unknown): boolean {
  return reason instanceof Error && reason.name === "AbortError";
}

function waitForJobPoll(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Job polling was cancelled.", "AbortError"));
      return;
    }
    const cancel = () => {
      window.clearTimeout(timer);
      reject(new DOMException("Job polling was cancelled.", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, JOB_POLL_INTERVAL_MS);
    signal.addEventListener("abort", cancel, { once: true });
  });
}

function jobStateLabel(status: DecodeJobState): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "succeeded":
      return "Succeeded";
    case "failed":
      return "Failed";
    case "timed_out":
      return "Timed out";
    case "cancelled":
      return "Cancelled";
  }
}

// The server 400s a submit whose remote_host id no longer exists (removed
// from workspace/remote-hosts.json since the browser cached it). Recognizing
// that specific rejection lets the caller refresh the host list and prompt a
// re-pick instead of leaving the stale id from localStorage silently
// failing on every retry.
function isRemoteHostRejection(reason: unknown): boolean {
  return (
    reason instanceof RequestError &&
    reason.status === 400 &&
    /remote host/i.test(reason.message)
  );
}

function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Primary step three of the reconstruction pipeline.
 *
 * The page above owns the capture selection and the capability report and
 * passes both down. The uncontrolled fallback keeps this component usable on
 * its own: it then reads captures itself and takes the capture from the
 * ?capture= param through captureIdFromParams.
 */
export function DecodeStage({
  captures: sharedCaptures,
  capabilities: sharedCapabilities,
  capabilitiesError: sharedCapabilitiesError,
  onCapabilitiesRetry: sharedOnCapabilitiesRetry,
  showCapabilitiesError = true,
  captureId: sharedCaptureId,
  captureRevision = "",
  remoteHosts: sharedRemoteHosts,
  remoteHostId: sharedRemoteHostId,
  onRemoteHostIdChange: sharedOnRemoteHostIdChange,
  remoteHostsError: sharedRemoteHostsError,
  remoteHostsLoading: sharedRemoteHostsLoading,
  onRemoteHostsRefresh: sharedOnRemoteHostsRefresh,
  onResultAvailableChange,
  onRunningChange,
}: {
  captures?: CaptureReceipt[];
  capabilities?: Capabilities | null;
  capabilitiesError?: string;
  onCapabilitiesRetry?: () => void;
  showCapabilitiesError?: boolean;
  captureId?: string;
  captureRevision?: string;
  remoteHosts?: RemoteHost[];
  remoteHostId?: string;
  onRemoteHostIdChange?: (next: string) => void;
  remoteHostsError?: string;
  remoteHostsLoading?: boolean;
  onRemoteHostsRefresh?: () => void;
  // Lets a parent reflect this stage's outcome without owning a second copy
  // of the decode state.
  onResultAvailableChange?: (available: boolean) => void;
  onRunningChange?: (running: boolean) => void;
}) {
  const [searchParams] = useSearchParams();
  const [ownCapabilities, setOwnCapabilities] = useState<Capabilities | null>(
    null,
  );
  const [ownCapabilitiesError, setOwnCapabilitiesError] = useState("");
  const [ownCaptureId, setOwnCaptureId] = useState("");

  const capabilities =
    sharedCapabilities === undefined ? ownCapabilities : sharedCapabilities;
  const capabilitiesError =
    sharedCapabilitiesError ?? ownCapabilitiesError;
  const captureId =
    sharedCaptureId === undefined ? ownCaptureId : sharedCaptureId;
  const ownRemoteHostSelection = useRemoteHostSelection(
    sharedRemoteHosts !== undefined,
  );
  const remoteHosts = sharedRemoteHosts ?? ownRemoteHostSelection.hosts;
  const remoteHostId =
    sharedRemoteHostId === undefined
      ? ownRemoteHostSelection.selectedId
      : sharedRemoteHostId;
  const onRemoteHostIdChange =
    sharedOnRemoteHostIdChange ?? ownRemoteHostSelection.setSelectedId;
  const remoteHostsError =
    sharedRemoteHostsError ?? ownRemoteHostSelection.error;
  const remoteHostsLoading =
    sharedRemoteHostsLoading ?? ownRemoteHostSelection.loading;
  const onRemoteHostsRefresh =
    sharedOnRemoteHostsRefresh ?? ownRemoteHostSelection.refresh;
  const onCapabilitiesRetry =
    sharedOnCapabilitiesRetry ??
    (() => {
      setOwnCapabilitiesError("");
      void fetchCapabilities()
        .then(setOwnCapabilities)
        .catch((reason) =>
          setOwnCapabilitiesError(
            reason instanceof Error
              ? reason.message
              : "The decode capability could not be read.",
          ),
        );
    });

  const [preflight, setPreflight] = useState<DecodePreflight | null>(null);
  const [preflightBusy, setPreflightBusy] = useState(false);
  const [preflightError, setPreflightError] = useState("");

  const [decodeJob, setDecodeJob] = useState<DecodeJobStatus | null>(null);
  const [decodeResult, setDecodeResult] = useState<DecodeResultDocument | null>(
    null,
  );
  const [decodeError, setDecodeError] = useState("");
  const [decodeRunning, setDecodeRunning] = useState(false);
  // Set when a submit or poll call throws (a transport failure, not a
  // job-reported failure). decodeJob is left in place; this flag just tells
  // the console its last-known status is no longer being refreshed, so the
  // stage strip does not sit there looking live next to the error banner.
  const [decodePollFailed, setDecodePollFailed] = useState(false);
  const [restoreNote, setRestoreNote] = useState("");
  const decodeGenerationRef = useRef(0);
  const decodeMonitorRef = useRef<AbortController | null>(null);
  const decodeRestoreGenerationRef = useRef(0);
  const decodeRestoreControllerRef = useRef<AbortController | null>(null);
  const preflightControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (sharedCaptures !== undefined) return;
    let active = true;
    void fetchCaptures()
      .then((payload) => {
        if (!active) return;
        setOwnCaptureId(
          (current) =>
            current || captureIdFromParams(searchParams, payload.captures),
        );
      })
      .catch(() => {
        // The preflight and capability panels report the outage instead.
      });
    return () => {
      active = false;
    };
  }, [sharedCaptures]);

  useEffect(() => {
    if (sharedCapabilities !== undefined) return;
    let active = true;
    void fetchCapabilities()
      .then((payload) => {
        if (active) {
          setOwnCapabilities(payload);
          setOwnCapabilitiesError("");
        }
      })
      .catch((reason) => {
        if (active) {
          setOwnCapabilitiesError(
            reason instanceof Error
              ? reason.message
              : "The decode capability could not be read.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [sharedCapabilities]);

  useEffect(
    () => () => {
      decodeGenerationRef.current += 1;
      decodeMonitorRef.current?.abort();
      decodeRestoreGenerationRef.current += 1;
      decodeRestoreControllerRef.current?.abort();
      preflightControllerRef.current?.abort();
    },
    [],
  );

  useEffect(() => {
    onResultAvailableChange?.(decodeResult !== null);
  }, [decodeResult, onResultAvailableChange]);

  useEffect(() => {
    onRunningChange?.(decodeRunning);
  }, [decodeRunning, onRunningChange]);

  const decodeCapability = capabilities?.decode_jobs ?? null;
  const decodeUnavailableReason = useMemo(() => {
    if (capabilities === null) return null;
    if (!decodeCapability) {
      return "The API host returned an incomplete Decode capability response.";
    }
    if (decodeCapability.enabled) return null;
    return (
      decodeCapability.reason ??
      "Configure a native or external decode runner on the API host."
    );
  }, [capabilities, decodeCapability]);

  // Capture-scoped job/result state resets only when the capture identity
  // changes. Preparation updates the same capture id, so preflight refresh is
  // deliberately separate below and keyed to the receipt revision.
  useEffect(() => {
    decodeGenerationRef.current += 1;
    decodeMonitorRef.current?.abort();
    decodeMonitorRef.current = null;
    setDecodeRunning(false);
    setDecodePollFailed(false);
    setDecodeJob(null);
    setDecodeResult(null);
    setDecodeError("");
    setRestoreNote("");
    setPreflight(null);
    setPreflightError("");
  }, [captureId]);

  const loadPreflight = useCallback(async () => {
    preflightControllerRef.current?.abort();
    if (!captureId) {
      setPreflightBusy(false);
      setPreflight(null);
      setPreflightError("");
      return;
    }
    const controller = new AbortController();
    preflightControllerRef.current = controller;
    setPreflightBusy(true);
    setPreflightError("");
    setPreflight(null);
    try {
      setPreflight(await fetchDecodePreflight(captureId, controller.signal));
    } catch (reason) {
      if (!isAbortError(reason)) {
        setPreflightError(
          reason instanceof Error
            ? reason.message
            : "The decode preflight could not be completed.",
        );
      }
    } finally {
      if (preflightControllerRef.current === controller) {
        preflightControllerRef.current = null;
        setPreflightBusy(false);
      }
    }
  }, [captureId]);

  // Refresh whenever Prepare changes the selected capture's immutable-input
  // fingerprint. This is what makes attach → seal → decode work without a
  // reload. The button in the render also refreshes host/runtime state.
  useEffect(() => {
    void loadPreflight();
    return () => preflightControllerRef.current?.abort();
  }, [captureId, captureRevision, loadPreflight]);

  const detailForRequirement = useCallback(
    (id: string) =>
      preflight?.checks.find((check) => check.id === id)?.detail ??
      "The API host reports this requirement as unmet.",
    [preflight],
  );

  const selectedRemoteHost =
    remoteHosts.find((host) => host.id === remoteHostId) ?? null;
  const computeTargetUnavailableReason = useMemo(() => {
    if (capabilities === null) return null;
    if (!remoteHostId) {
      return capabilities.gpu.available
        ? null
        : capabilities.gpu.reason ??
            "No CUDA GPU was detected on this workbench. Configure a remote GPU host or run the API on a CUDA machine.";
    }
    if (selectedRemoteHost) return null;
    if (remoteHostsLoading) {
      return "The configured remote host list is still loading.";
    }
    if (remoteHostsError) {
      return "The selected remote host could not be verified. Refresh the host list or choose Local CUDA.";
    }
    return "That remote host is no longer configured. Choose Local CUDA or another host.";
  }, [
    capabilities,
    remoteHostId,
    remoteHostsError,
    remoteHostsLoading,
    selectedRemoteHost,
  ]);

  const decodeReady =
    !capabilitiesError &&
    decodeUnavailableReason === null &&
    computeTargetUnavailableReason === null &&
    capabilities !== null &&
    Boolean(captureId) &&
    preflight?.ready === true;

  const runDecode = async () => {
    if (!captureId || !decodeReady) return;
    const generation = decodeGenerationRef.current + 1;
    decodeGenerationRef.current = generation;
    decodeMonitorRef.current?.abort();
    const controller = new AbortController();
    decodeMonitorRef.current = controller;
    setDecodeRunning(true);
    setDecodeError("");
    setDecodePollFailed(false);
    setDecodeResult(null);
    setRestoreNote("");

    try {
      let job = await submitDecodeJob(
        captureId,
        remoteHostId || undefined,
        controller.signal,
      );
      if (decodeGenerationRef.current !== generation) return;
      setDecodeJob(job);

      while (!isTerminalDecodeState(job.status)) {
        await waitForJobPoll(controller.signal);
        job = await fetchDecodeJobStatus(job.job_id, controller.signal);
        if (decodeGenerationRef.current !== generation) return;
        setDecodeJob(job);
      }

      if (job.status !== "succeeded") {
        setDecodeError(
          job.error ??
            `The decode runner finished with status ${jobStateLabel(job.status)}.`,
        );
        return;
      }
      if (!job.result_available) {
        setDecodeError(
          "The decode job finished without publishing a result document.",
        );
        return;
      }

      const document = await fetchDecodeJobResult(job.job_id, controller.signal);
      if (decodeGenerationRef.current !== generation) return;
      setDecodeResult(document);
    } catch (reason) {
      if (!isAbortError(reason) && decodeGenerationRef.current === generation) {
        setDecodePollFailed(true);
        if (isRemoteHostRejection(reason)) {
          onRemoteHostsRefresh?.();
          setDecodeError(
            `${(reason as RequestError).message} The remote host list has been refreshed. Pick a host and try again.`,
          );
        } else {
          setDecodeError(
            reason instanceof Error
              ? reason.message
              : "The decode job could not be completed.",
          );
        }
      }
    } finally {
      if (decodeGenerationRef.current === generation) {
        setDecodeRunning(false);
        if (decodeMonitorRef.current === controller) {
          decodeMonitorRef.current = null;
        }
      }
    }
  };

  // Picks a queued or running decode job back up after a page reload
  // instead of showing an idle "Run decode" button. It shares the run
  // console, log, and stage chips with a freshly submitted job.
  const resumeDecodeJob = async (initialJob: DecodeJobStatus) => {
    const generation = decodeGenerationRef.current + 1;
    decodeGenerationRef.current = generation;
    decodeMonitorRef.current?.abort();
    const controller = new AbortController();
    decodeMonitorRef.current = controller;
    setDecodeRunning(true);
    setDecodeError("");
    setDecodePollFailed(false);
    setDecodeResult(null);
    setRestoreNote(
      "Resuming the decode job that was already running for this capture.",
    );
    setDecodeJob(initialJob);

    try {
      let job = initialJob;
      while (!isTerminalDecodeState(job.status)) {
        await waitForJobPoll(controller.signal);
        job = await fetchDecodeJobStatus(job.job_id, controller.signal);
        if (decodeGenerationRef.current !== generation) return;
        setDecodeJob(job);
      }

      if (job.status !== "succeeded") {
        setRestoreNote("");
        setDecodeError(
          job.error ??
            `The decode runner finished with status ${jobStateLabel(job.status)}.`,
        );
        return;
      }
      if (!job.result_available) {
        setRestoreNote("");
        setDecodeError(
          "The decode job finished without publishing a result document.",
        );
        return;
      }

      const document = await fetchDecodeJobResult(job.job_id, controller.signal);
      if (decodeGenerationRef.current !== generation) return;
      setDecodeResult(document);
      setRestoreNote(
        `Loaded the decode result from ${formatFinishedAt(job.finished_at)}.`,
      );
    } catch (reason) {
      if (!isAbortError(reason) && decodeGenerationRef.current === generation) {
        setRestoreNote("");
        setDecodePollFailed(true);
        setDecodeError(
          reason instanceof Error
            ? reason.message
            : "The decode job could not be completed.",
        );
      }
    } finally {
      if (decodeGenerationRef.current === generation) {
        setDecodeRunning(false);
        if (decodeMonitorRef.current === controller) {
          decodeMonitorRef.current = null;
        }
      }
    }
  };

  // Loads an already-finished result instead of requiring a rerun, and
  // resumes a still-running job instead of sitting idle after a reload.
  // Guarded on no job/result already present and no decode actively running,
  // so a manual "Run decode" always wins. The generation ref plus its own
  // controller mean a quick capture switch cancels whichever restore fetch
  // was still in flight instead of cross-loading a result.
  useEffect(() => {
    decodeRestoreGenerationRef.current += 1;
    const generation = decodeRestoreGenerationRef.current;
    decodeRestoreControllerRef.current?.abort();
    if (!captureId || decodeJob || decodeResult || decodeRunning) return;
    const controller = new AbortController();
    decodeRestoreControllerRef.current = controller;

    void (async () => {
      let jobs: DecodeJobStatus[];
      try {
        const payload = await fetchDecodeJobsForCapture(captureId, controller.signal);
        jobs = payload.jobs;
      } catch (reason) {
        if (isAbortError(reason)) return;
        if (reason instanceof RequestError && reason.status === 404) return;
        setDecodeError(
          reason instanceof Error
            ? `Previous decode jobs could not be checked: ${reason.message}`
            : "Previous decode jobs could not be checked.",
        );
        return;
      }
      if (decodeRestoreGenerationRef.current !== generation) return;
      const newestJob = jobs[0];
      if (!newestJob) return;

      if (!isTerminalDecodeState(newestJob.status)) {
        void resumeDecodeJob(newestJob);
        return;
      }
      if (newestJob.status !== "succeeded" || !newestJob.result_available) return;

      try {
        const document = await fetchDecodeJobResult(newestJob.job_id, controller.signal);
        if (decodeRestoreGenerationRef.current !== generation) return;
        setDecodeJob(newestJob);
        setDecodeResult(document);
        setRestoreNote(
          `Loaded the decode result from ${formatFinishedAt(newestJob.finished_at)}.`,
        );
      } catch (reason) {
        if (isAbortError(reason)) return;
        setDecodeError(
          reason instanceof Error
            ? `The previous decode result could not be restored: ${reason.message}`
            : "The previous decode result could not be restored.",
        );
      }
    })();

    return () => controller.abort();
  }, [captureId, decodeJob, decodeResult, decodeRunning]);

  const stageChips = useMemo(
    () => decodeStageChips(decodeJob?.stages_seen, decodeJob?.stage),
    [decodeJob?.stage, decodeJob?.stages_seen],
  );
  const stageProgressLabel = decodeStageProgressLabel(
    decodeJob?.stage_progress,
  );

  const replayReached = decodeJob?.replay_solved_reached ?? null;
  const resultCompleted = decodeResult?.status === "completed";
  const downloadResult = async () => {
    if (!decodeJob) {
      setDecodeError(
        "The exact run artifact is unavailable without its Decode job receipt.",
      );
      return;
    }
    setDecodeError("");
    try {
      const artifact = await fetchDecodeJobResultBlob(decodeJob.job_id);
      triggerBlobDownload(
        artifact,
        `cubed-core-run-${decodeJob.job_id}.json`,
      );
    } catch (reason) {
      setDecodeError(
        reason instanceof Error
          ? reason.message
          : "The exact run artifact could not be downloaded.",
      );
    }
  };

  const badgeCopy =
    decodeResult
      ? "Previous result"
      : capabilitiesError
        ? "Route check failed"
      : capabilities === null
      ? "Checking route"
      : decodeUnavailableReason !== null
        ? "Runner unavailable"
        : decodeReady
          ? "Ready to run"
          : "Setup required";

  return (
    <section className="station-card decode-stage-card">
      <div className="station-card-head">
        <div>
          <p className="station-code">Step 3</p>
          <h2>Decode and replay</h2>
        </div>
        <span
          className={`capability-badge ${
            decodeReady ? "capability-ready" : "capability-migration"
          }`}
        >
          {badgeCopy}
        </span>
      </div>

      {decodeResult && (
        <div className="decode-result-panel" data-status={decodeResult.status}>
          <div className="decode-result-head">
            <div>
              <p className="station-code">Decode outcome</p>
              <h3>
                {resultCompleted
                  ? `${decodeResult.moves.length} decoded move${
                      decodeResult.moves.length === 1 ? "" : "s"
                    }`
                  : "The run abstained."}
              </h3>
              <p className="decode-result-note">
                {resultCompleted
                  ? "The job completed on this input. The sequence replayed to solved. Inspect the run alongside its recording in Runs."
                  : "The run JSON preserves that outcome."}
              </p>
            </div>
            <div className="decode-result-actions">
              <button
                className="button button-secondary"
                type="button"
                disabled={!decodeJob}
                onClick={() => void downloadResult()}
              >
                Download run JSON
              </button>
              {decodeJob && (
                <Link
                  className="button button-primary"
                  to={`/runs?run=${encodeURIComponent(decodeJob.job_id)}`}
                >
                  Open in Runs
                </Link>
              )}
            </div>
          </div>
          <dl className="decode-result-provenance">
            <div>
              <dt>Outcome</dt>
              <dd>{decodeResult.status}</dd>
            </div>
            <div>
              <dt>Profile</dt>
              <dd>{decodeResult.profile}</dd>
            </div>
            <div>
              <dt>CFG_HASH</dt>
              <dd>{decodeResult.config.cfg_hash}</dd>
            </div>
            <div>
              <dt>Server replay</dt>
              <dd>
                {replayReached === null
                  ? "N/A"
                  : replayReached
                    ? "Solved"
                    : "Not solved"}
              </dd>
            </div>
          </dl>
        </div>
      )}

      <p className="decode-stage-intro">
        {decodeResult
          ? "The completed attempt is saved in Runs. Use the controls below only to submit another attempt."
          : "Run local_camera_v1 on the locked recording and review the returned move sequence."}
      </p>

      <div className="decode-compute-target-field">
        <label className="field-label" htmlFor="decode-compute-target-select">
          <span>Compute target</span>
          <small>
            Local CUDA runs on this workbench. Configured hosts use the remote
            GPU bridge.
          </small>
        </label>
        <ComputeTargetSelector
          hosts={remoteHosts}
          selectedId={remoteHostId}
          onChange={onRemoteHostIdChange}
          localGpu={capabilities?.gpu ?? null}
          idPrefix="decode"
          disabled={decodeRunning}
          loading={remoteHostsLoading}
          error={remoteHostsError}
        />
      </div>

      {capabilitiesError ? (
        showCapabilitiesError ? (
          <div className="decode-stage-blocked" role="alert">
            <strong>The decode route could not be checked.</strong>
            <p>{capabilitiesError}</p>
            <button
              className="button button-secondary"
              type="button"
              onClick={onCapabilitiesRetry}
            >
              Retry capability check
            </button>
          </div>
        ) : null
      ) : decodeUnavailableReason !== null ? (
        <div className="decode-stage-blocked" role="status">
          <strong>Decode jobs are unavailable on this API host.</strong>
          <p>{decodeUnavailableReason}</p>
          <p>
            The decode tutorial covers both runner modes:{" "}
            <a
              href={routeGuideDocumentHref("docs/tutorials/DECODE.md")}
              target={ROUTE_GUIDE_LINK_TARGET}
              rel={ROUTE_GUIDE_LINK_REL}
            >
              docs/tutorials/DECODE.md
            </a>
            .
          </p>
        </div>
      ) : capabilities === null ? (
        <p className="decode-stage-message" role="status">
          Reading the decode route from the API host.
        </p>
      ) : computeTargetUnavailableReason !== null && decodeResult === null ? (
        <div className="decode-stage-blocked" role="status">
          <strong>Choose an available compute target.</strong>
          <p>{computeTargetUnavailableReason}</p>
        </div>
      ) : !captureId ? (
        <p className="decode-stage-message" role="status">
          Choose a capture at the top of this page to check its decode
          prerequisites.
        </p>
      ) : preflightError ? (
        <div className="tool-message tool-message-error" role="alert">
          <p>{preflightError}</p>
          <button
            className="button button-secondary"
            type="button"
            onClick={() => void loadPreflight()}
          >
            Retry readiness check
          </button>
        </div>
      ) : preflightBusy || !preflight ? (
        <p className="decode-stage-message" role="status">
          Checking this capture against the local_camera_v1 requirements.
        </p>
      ) : !preflight.ready ? (
        <div className="decode-stage-blocked">
          <p className="decode-readiness-line decode-readiness-blocked" role="status">
            <strong>Blocked:</strong>{" "}
            {preflight.missing.length > 0
              ? detailForRequirement(preflight.missing[0])
              : "This capture is not ready to decode."}
          </p>
          <details className="decode-check-disclosure">
            <summary>
              All checks ({preflight.missing.length} blocking)
            </summary>
            <div className="decode-check-disclosure-body">
              <p>
                Every item below has to pass before the API host will accept a
                decode job for it.
              </p>
              <ul className="decode-missing-checklist">
                {preflight.missing.map((item) => (
                  <li key={item}>
                    <code>{item}</code>
                    <span>{detailForRequirement(item)}</span>
                  </li>
                ))}
              </ul>
              <Link
                className="text-button"
                to={captureHref("/import", captureId)}
              >
                Open Add video to fix the prerequisites
              </Link>
            </div>
          </details>
          <button
            className="button button-secondary decode-preflight-refresh"
            type="button"
            onClick={() => void loadPreflight()}
          >
            Refresh readiness
          </button>
        </div>
      ) : (
        <div className="tracker-route-action decode-stage-run">
          <p className="decode-readiness-line decode-readiness-ready" role="status">
            <strong>Ready to decode.</strong>
          </p>
          <p className="decode-readiness-copy">
            The API host resolves the locked video and scramble plus the current
            calibration from its own workspace. This attempt saves its own
            calibration snapshot. Config {preflight.config.name} with CFG_HASH{" "}
            {preflight.config.cfg_hash} will run.
          </p>
          {preflight.warnings.length > 0 && (
            <div className="decode-preflight-warnings" role="note">
              <strong>Warnings</strong>
              <ul>
                {preflight.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="decode-readiness-actions">
            <button
              className="button button-secondary decode-preflight-refresh"
              type="button"
              onClick={() => void loadPreflight()}
            >
              Refresh readiness
            </button>
            <button
              className="button button-primary"
              type="button"
              disabled={
                !decodeReady ||
                decodeRunning ||
                Boolean(decodeJob && !isTerminalDecodeState(decodeJob.status))
              }
              onClick={() => void runDecode()}
            >
              {decodeRunning
                ? "Decode running…"
                : decodeJob && isTerminalDecodeState(decodeJob.status)
                  ? "Run decode again"
                  : "Run decode"}
            </button>
          </div>
        </div>
      )}

      {restoreNote && (
        <p className="tool-message" role="status">
          {restoreNote}
        </p>
      )}

      {decodeJob && (
        <div
          className="decode-job-console"
          aria-live="polite"
        >
          <div className="decode-job-console-heading">
            <div>
              <span>Decode job {decodeJob.job_id.slice(0, 12)}</span>
              <strong>{jobStateLabel(decodeJob.status)}</strong>
            </div>
            <dl>
              <div>
                <dt>Capture</dt>
                <dd>{decodeJob.capture_id.slice(0, 8)}</dd>
              </div>
              <div>
                <dt>Return</dt>
                <dd>{decodeJob.return_code ?? "N/A"}</dd>
              </div>
              <div>
                <dt>Result</dt>
                <dd>
                  {decodeJob.result_available ? "published" : "not available"}
                </dd>
              </div>
            </dl>
          </div>
          {!isTerminalDecodeState(decodeJob.status) && (
            <div className="decode-stage-strip">
              {decodePollFailed ? (
                <span className="decode-stage-fallback" role="alert">
                  Connection to the runner was lost while this job was still{" "}
                  {jobStateLabel(decodeJob.status).toLowerCase()}. This strip
                  stopped updating. The error below explains what happened.
                </span>
              ) : stageChips.length > 0 ? (
                <>
                  <ol className="decode-stage-chips">
                    {stageChips.map((chip) => (
                      <li
                        className={`decode-stage-chip decode-stage-${chip.state}`}
                        key={chip.token}
                        aria-current={
                          chip.state === "current" ? "step" : undefined
                        }
                      >
                        {chip.label}
                      </li>
                    ))}
                  </ol>
                  {stageProgressLabel && (
                    <span className="decode-stage-progress">
                      {stageProgressLabel}
                    </span>
                  )}
                </>
              ) : (
                <span className="decode-stage-fallback">
                  This runner reports no stage breakdown. The runner output
                  below fills in while the job runs.
                </span>
              )}
            </div>
          )}
          {decodeJob.error && (
            <p className="decode-job-error" role="alert">
              {decodeJob.error}
            </p>
          )}
          <details className="decode-job-log-disclosure">
            <summary>Runner output</summary>
            <div className="decode-job-log">
              <span>
                Shows up to the final 64 KiB.
                {decodeJob.log_truncated ? " Earlier output was omitted." : ""}
              </span>
              <pre>{decodeJob.log.slice(-65_536) || "No runner output yet."}</pre>
            </div>
          </details>
        </div>
      )}

      {decodeError && (
        <p className="tool-message tool-message-error" role="alert">
          {decodeError}
        </p>
      )}
      {decodePollFailed &&
        decodeJob &&
        !isTerminalDecodeState(decodeJob.status) && (
          <button
            className="button button-secondary"
            type="button"
            disabled={decodeRunning}
            onClick={() => void resumeDecodeJob(decodeJob)}
          >
            Resume status for this job
          </button>
        )}

    </section>
  );
}
