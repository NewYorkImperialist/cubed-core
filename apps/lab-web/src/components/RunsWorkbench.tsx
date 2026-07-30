import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { fetchCaptures } from "../api";
import {
  deleteDecodeRun,
  fetchDecodeJobResult,
  fetchDecodeJobResultBlob,
  fetchDecodeJobs,
} from "../decodeApi";
import type {
  CaptureReceipt,
  DecodeJobStatus,
  DecodeResultDocument,
} from "../types";
import { isTerminalDecodeState } from "../decodeJobContract";
import { useWorkbenchContext } from "../workbenchContext";
import { DecodeRunInspector } from "./DecodeRunInspector";

const RUN_PARAM = "run";
const SOURCE_PARAM = "source";
const PORTABLE_SOURCE = "file";

interface WorkspaceAttempt {
  capture: CaptureReceipt | null;
  job: DecodeJobStatus;
}

interface RunGroup {
  key: string;
  capture: CaptureReceipt | null;
  attempts: WorkspaceAttempt[];
}

function runTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function recordingDuration(capture: CaptureReceipt | null): string {
  const frames = capture?.video.frame_count;
  const fps =
    capture?.video.actual_fps ?? capture?.video.configured_fps ?? null;
  if (
    typeof frames !== "number" ||
    !Number.isFinite(frames) ||
    frames <= 0 ||
    typeof fps !== "number" ||
    !Number.isFinite(fps) ||
    fps <= 0
  ) {
    return "Unknown length";
  }
  const seconds = frames / fps;
  if (seconds < 60) return `${seconds.toFixed(1)}s video`;
  return `${Math.floor(seconds / 60)}m ${String(
    Math.round(seconds % 60),
  ).padStart(2, "0")}s video`;
}

function runOutcome(job: DecodeJobStatus): string {
  switch (job.outcome) {
    case "completed":
      return "Completed";
    case "abstained":
      return "Abstained";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
  }
  switch (job.status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "failed":
      return "Failed";
    case "timed_out":
      return "Timed out";
    case "cancelled":
      return "Cancelled";
    case "succeeded":
      return job.result_available ? "Result ready" : "Finished without a result";
  }
}

function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function attemptMatches(attempt: WorkspaceAttempt, query: string): boolean {
  if (!query) return true;
  const { capture, job } = attempt;
  return [
    capture?.original_filename ?? "",
    job.capture_id,
    job.video_sha256 ?? capture?.video.sha256 ?? "",
    job.job_id,
    job.status,
    job.outcome ?? "",
    job.failure?.code ?? "",
    job.failure?.message ?? job.error ?? "",
  ]
    .join(" ")
    .toLocaleLowerCase()
    .includes(query);
}

export function RunsWorkbench() {
  const { apiState } = useWorkbenchContext();
  const [searchParams, setSearchParams] = useSearchParams();
  const [captures, setCaptures] = useState<CaptureReceipt[]>([]);
  const [attempts, setAttempts] = useState<WorkspaceAttempt[]>([]);
  const [selectedResult, setSelectedResult] =
    useState<DecodeResultDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [resultLoading, setResultLoading] = useState(false);
  const [deletingJobId, setDeletingJobId] = useState("");
  const [error, setError] = useState("");
  const [metadataWarning, setMetadataWarning] = useState("");
  const [resultError, setResultError] = useState("");
  const [search, setSearch] = useState("");
  const selectedRunId = searchParams.get(RUN_PARAM) ?? "";
  const portableMode = searchParams.get(SOURCE_PARAM) === PORTABLE_SOURCE;
  const requestedCaptureId = searchParams.get("capture") ?? "";

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError("");
    setMetadataWarning("");
    try {
      const [jobsResult, capturesResult] = await Promise.allSettled([
        fetchDecodeJobs(),
        fetchCaptures(),
      ]);
      if (jobsResult.status === "rejected") throw jobsResult.reason;
      const captureList =
        capturesResult.status === "fulfilled"
          ? capturesResult.value.captures
          : [];
      setCaptures(captureList);
      if (capturesResult.status === "rejected") {
        setMetadataWarning(
          "Run history is available, but recording metadata and video pairing could not be loaded.",
        );
      }
      const capturesById = new Map(
        captureList.map((capture) => [
          capture.capture_id,
          capture,
        ]),
      );
      setAttempts(
        jobsResult.value.jobs
          .map((job) => ({
            capture: capturesById.get(job.capture_id) ?? null,
            job,
          }))
          .sort(
            (left, right) =>
              new Date(right.job.created_at).valueOf() -
              new Date(left.job.created_at).valueOf(),
          ),
      );
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Workspace decode runs could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (apiState !== "online") {
      setLoading(false);
      return;
    }
    void loadRuns();
  }, [apiState, loadRuns]);

  const groups = useMemo(() => {
    const grouped = new Map<string, WorkspaceAttempt[]>();
    for (const attempt of attempts) {
      const key =
        attempt.job.video_sha256 ||
        attempt.capture?.video.sha256 ||
        `capture:${attempt.job.capture_id}`;
      const group = grouped.get(key) ?? [];
      group.push(attempt);
      grouped.set(key, group);
    }
    return Array.from(grouped.entries())
      .map(([key, groupAttempts]): RunGroup => ({
        key,
        capture: groupAttempts[0].capture,
        attempts: groupAttempts,
      }))
      .sort(
        (left, right) =>
          new Date(right.attempts[0].job.created_at).valueOf() -
          new Date(left.attempts[0].job.created_at).valueOf(),
      );
  }, [attempts]);
  const normalizedSearch = search.trim().toLocaleLowerCase();
  const visibleGroups = useMemo(
    () =>
      groups.filter((group) =>
        group.attempts.some((attempt) =>
          attemptMatches(attempt, normalizedSearch),
        ),
      ),
    [groups, normalizedSearch],
  );
  const selectedAttempt = useMemo(
    () =>
      attempts.find(({ job }) => job.job_id === selectedRunId) ?? null,
    [attempts, selectedRunId],
  );
  const requestedCaptureHasRun = useMemo(
    () =>
      !requestedCaptureId ||
      attempts.some(
        ({ job }) => job.capture_id === requestedCaptureId,
      ),
    [attempts, requestedCaptureId],
  );

  useEffect(() => {
    if (selectedRunId || portableMode || !requestedCaptureId) return;
    const newestForCapture = attempts.find(
      ({ job }) =>
        job.capture_id === requestedCaptureId && job.result_available,
    );
    if (!newestForCapture) return;
    const next = new URLSearchParams();
    next.set(RUN_PARAM, newestForCapture.job.job_id);
    setSearchParams(next, { replace: true });
  }, [
    attempts,
    portableMode,
    requestedCaptureId,
    selectedRunId,
    setSearchParams,
  ]);

  useEffect(() => {
    setSelectedResult(null);
    setResultError("");
    setResultLoading(false);
    if (!selectedAttempt?.job.result_available) return;
    const controller = new AbortController();
    setResultLoading(true);
    void fetchDecodeJobResult(selectedAttempt.job.job_id, controller.signal)
      .then(setSelectedResult)
      .catch((reason) => {
        if (reason instanceof Error && reason.name === "AbortError") return;
        setResultError(
          reason instanceof Error
            ? reason.message
            : "The selected run JSON could not be loaded.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setResultLoading(false);
      });
    return () => controller.abort();
  }, [selectedAttempt]);

  const openRun = (runId: string) => {
    const next = new URLSearchParams();
    next.set(RUN_PARAM, runId);
    setSearchParams(next);
  };
  const openPortable = () => {
    const next = new URLSearchParams();
    next.set(SOURCE_PARAM, PORTABLE_SOURCE);
    setSearchParams(next);
  };
  const changeRun = () => setSearchParams(new URLSearchParams());

  const downloadRun = async (attempt: WorkspaceAttempt) => {
    setResultError("");
    try {
      const artifact = await fetchDecodeJobResultBlob(attempt.job.job_id);
      triggerBlobDownload(
        artifact,
        `cubed-core-run-${attempt.job.job_id}.json`,
      );
    } catch (reason) {
      setResultError(
        reason instanceof Error
          ? reason.message
          : "The run JSON could not be downloaded.",
      );
    }
  };

  const removeRun = async (attempt: WorkspaceAttempt) => {
    if (
      !window.confirm(
        `Delete run ${attempt.job.job_id.slice(
          0,
          12,
        )}? Only this run attempt will move to recoverable trash. The capture and video stay in the workspace.`,
      )
    ) {
      return;
    }
    setDeletingJobId(attempt.job.job_id);
    setError("");
    try {
      await deleteDecodeRun(attempt.job.job_id);
      if (selectedRunId === attempt.job.job_id) changeRun();
      await loadRuns();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "The run could not be moved to trash.",
      );
    } finally {
      setDeletingJobId("");
    }
  };

  const renderAttempt = (
    attempt: WorkspaceAttempt,
    label: "Latest" | "Previous",
  ) => {
    const { job } = attempt;
    const terminal = isTerminalDecodeState(job.status);
    const failure = job.failure?.message ?? job.error;
    return (
      <div className="run-attempt" key={job.job_id}>
        <div className="run-attempt-identity">
          <span>{label}</span>
          <code>{job.job_id.slice(0, 12)}</code>
        </div>
        <time dateTime={job.created_at}>{runTime(job.created_at)}</time>
        <div>
          <strong className={`run-status run-status-${job.status}`}>
            {runOutcome(job)}
          </strong>
          {failure && <small className="run-attempt-error">{failure}</small>}
        </div>
        <span title="Recording duration">
          {recordingDuration(attempt.capture)}
        </span>
        <div className="run-attempt-actions">
          {job.result_available && (
            <>
              <button
                className="button button-quiet"
                type="button"
                onClick={() => openRun(job.job_id)}
              >
                Inspect
              </button>
              <button
                className="text-button"
                type="button"
                onClick={() => void downloadRun(attempt)}
              >
                Download
              </button>
            </>
          )}
          {!terminal && (
            <button
              className="button button-quiet"
              type="button"
              onClick={() => void loadRuns()}
            >
              Refresh
            </button>
          )}
          {terminal && (
            <button
              className="text-button run-delete-action"
              type="button"
              disabled={deletingJobId === job.job_id}
              onClick={() => void removeRun(attempt)}
            >
              {deletingJobId === job.job_id ? "Deleting…" : "Delete run"}
            </button>
          )}
        </div>
      </div>
    );
  };

  if (portableMode) {
    return (
      <div className="page runs-page runs-inspector-page">
        <header className="runs-inspector-heading">
          <div>
            <p className="eyebrow">Run inspector</p>
            <h1>Open a run JSON</h1>
            <p>
              Choose the exact JSON downloaded from Decode or Runs. Pair its
              matching video only when you want synchronized playback and
              overlays.
            </p>
          </div>
          <button
            className="button button-secondary"
            type="button"
            onClick={changeRun}
          >
            Change run
          </button>
        </header>
        <DecodeRunInspector captures={captures} portable />
      </div>
    );
  }

  if (selectedAttempt) {
    return (
      <div className="page runs-page runs-inspector-page">
        <header className="runs-inspector-heading">
          <div>
            <p className="eyebrow">Run inspector</p>
            <h1>
              {selectedAttempt.capture?.original_filename ??
                `Capture ${selectedAttempt.job.capture_id.slice(0, 12)}`}
            </h1>
            <p>
              {selectedAttempt.job.job_id.slice(0, 12)} ·{" "}
              {runTime(selectedAttempt.job.created_at)}
            </p>
          </div>
          <button
            className="button button-secondary"
            type="button"
            onClick={changeRun}
          >
            Change run
          </button>
        </header>

        {resultLoading && (
          <p className="runs-empty" role="status">
            Loading run JSON…
          </p>
        )}
        {resultError && (
          <p className="tool-message tool-message-error" role="alert">
            {resultError}
          </p>
        )}
        {!resultLoading && selectedResult && (
          <DecodeRunInspector
            key={selectedAttempt.job.job_id}
            captures={captures}
            captureId={selectedAttempt.job.capture_id}
            job={selectedAttempt.job}
            result={selectedResult}
            resultName={`cubed-core-run-${selectedAttempt.job.job_id}.json`}
          />
        )}
        {!resultLoading &&
          !selectedResult &&
          !resultError &&
          !selectedAttempt.job.result_available && (
            <section className="run-no-result">
              <p className="station-code">Decode attempt</p>
              <h2>{runOutcome(selectedAttempt.job)}</h2>
              <p>
                {selectedAttempt.job.failure?.message ??
                  selectedAttempt.job.error ??
                  "This attempt has no run JSON to inspect."}
              </p>
              <p>
                Runs is read-only. Return to Decode if you want to submit
                another attempt.
              </p>
            </section>
          )}
      </div>
    );
  }

  return (
    <div className="page runs-page">
      <header className="page-header runs-header">
        <div>
          <p className="eyebrow">Decode history</p>
          <h1>Runs</h1>
          <p className="page-description">
            Review Decode attempts and their available per-frame diagnostics.
            Runs never starts compute.
          </p>
        </div>
        <button
          className="button button-secondary"
          type="button"
          onClick={openPortable}
        >
          Open run JSON
        </button>
      </header>

      {apiState === "offline" && (
        <p className="tool-message tool-message-error" role="alert">
          The local service is offline. Start the workbench to browse workspace
          runs, or open a portable run JSON.
        </p>
      )}
      {apiState === "online" && error && (
        <div className="tool-message tool-message-error" role="alert">
          <p>{error}</p>
          <button
            className="text-button"
            type="button"
            onClick={() => void loadRuns()}
          >
            Retry
          </button>
        </div>
      )}
      {apiState === "online" && metadataWarning && !error && (
        <p className="tool-message" role="status">
          {metadataWarning}
        </p>
      )}
      {resultError && !selectedRunId && (
        <p className="tool-message tool-message-error" role="alert">
          {resultError}
        </p>
      )}
      {apiState === "online" &&
        !loading &&
        !error &&
        selectedRunId &&
        !selectedAttempt && (
          <p className="tool-message tool-message-error" role="alert">
            The selected run is no longer available. Choose another workspace
            run or open a portable run JSON.
          </p>
        )}
      {apiState === "online" &&
        !loading &&
        !error &&
        requestedCaptureId &&
        !requestedCaptureHasRun && (
          <p className="tool-message" role="status">
            This capture has no Decode attempt yet. Other workspace runs remain
            available below.
          </p>
        )}

      <section className="runs-list" aria-labelledby="recent-runs-title">
        <div className="runs-list-heading">
          <div>
            <p className="station-code">Workspace</p>
            <h2 id="recent-runs-title">Recent recordings</h2>
          </div>
          <div className="runs-list-tools">
            <label>
              <span className="sr-only">Search runs</span>
              <input
                type="search"
                value={search}
                placeholder="Search filename, status, or ID"
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <button
              className="text-button"
              type="button"
              disabled={loading || apiState !== "online"}
              onClick={() => void loadRuns()}
            >
              {loading ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>

        {loading ? (
          <p className="runs-empty" role="status">
            Loading workspace runs…
          </p>
        ) : attempts.length === 0 ? (
          <div className="runs-empty">
            <strong>No Decode attempts in this workspace</strong>
            <p>
              Submit a prepared recording from Decode. Every attempt appears
              here automatically, including abstentions and failures.
            </p>
            <button className="text-button" type="button" onClick={openPortable}>
              Open a portable run JSON
            </button>
          </div>
        ) : visibleGroups.length === 0 ? (
          <p className="runs-empty" role="status">
            No runs match “{search.trim()}”.
          </p>
        ) : (
          <div className="run-groups" aria-label="Recent workspace runs">
            {visibleGroups.map((group) => {
              const [latest, ...history] = group.attempts;
              return (
                <article className="run-group" key={group.key}>
                  <header>
                    <div>
                      <h3>
                        {group.capture?.original_filename ??
                          `Capture ${latest.job.capture_id.slice(0, 12)}`}
                      </h3>
                      <code>
                        {(
                          latest.job.video_sha256 ??
                          group.capture?.video.sha256 ??
                          "video-sha-unavailable"
                        ).slice(0, 12)}{" "}
                        · {latest.job.capture_id.slice(0, 8)}
                      </code>
                    </div>
                    <span>
                      {group.attempts.length} attempt
                      {group.attempts.length === 1 ? "" : "s"}
                    </span>
                  </header>
                  {renderAttempt(latest, "Latest")}
                  {history.length > 0 && (
                    <details
                      className="run-history"
                      open={normalizedSearch ? true : undefined}
                    >
                      <summary>
                        {history.length} earlier attempt
                        {history.length === 1 ? "" : "s"}
                      </summary>
                      <div>
                        {history.map((attempt) =>
                          renderAttempt(attempt, "Previous"),
                        )}
                      </div>
                    </details>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
