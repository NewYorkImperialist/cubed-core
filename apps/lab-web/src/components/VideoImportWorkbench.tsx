import {
  ChangeEvent,
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useSearchParams } from "react-router-dom";

import { deleteCapture, fetchCaptures, importCapture } from "../api";
import {
  CAPTURE_PARAM,
  captureHref,
  isPublishedDatasetCapture,
  orderDecodeCaptures,
} from "../captureSelection";
import {
  loadOrCreateRecordingScramble,
  normalizeCanonicalScramble,
  replaceRecordingScramble,
  scrambleTokens,
} from "../importWorkflow";
import type { CaptureReceipt } from "../types";
import { useWorkbenchContext } from "../workbenchContext";
import { PrepareForDecodeCard } from "./PrepareForDecodeCard";
import "./VideoImportWorkbench.css";

const DEFAULT_VIDEO_UPLOAD_BYTES = 1024 * 1024 * 1024;
const VIDEO_ACCEPT = "video/*,.mov,.mp4,.m4v,.avi,.mkv,.webm";

type UploadPath = "record" | "existing";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "N/A";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** index;
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function recordingSummary(capture: CaptureReceipt): string {
  const width = capture.video.encoded_width ?? capture.probe.width;
  const height = capture.video.encoded_height ?? capture.probe.height;
  const fps = capture.video.actual_fps ?? capture.probe.fps;
  const dimensions =
    typeof width === "number" && typeof height === "number"
      ? `${width}×${height}`
      : "resolution unavailable";
  const frameRate =
    typeof fps === "number" && Number.isFinite(fps)
      ? `${fps.toFixed(3)} fps`
      : "frame rate unavailable";
  return `${dimensions} · ${frameRate}`;
}

function sessionStorageOrNull(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function VideoImportWorkbench() {
  const { capabilities } = useWorkbenchContext();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedCaptureId = searchParams.get(CAPTURE_PARAM) ?? "";
  const primaryFileInputRef = useRef<HTMLInputElement>(null);
  const existingFileInputRef = useRef<HTMLInputElement>(null);

  const [captures, setCaptures] = useState<CaptureReceipt[]>([]);
  const [recordingScramble, setRecordingScramble] = useState(() =>
    loadOrCreateRecordingScramble(sessionStorageOrNull()),
  );
  const [primaryFile, setPrimaryFile] = useState<File | null>(null);
  const [primaryConfirmed, setPrimaryConfirmed] = useState(false);
  const [existingFile, setExistingFile] = useState<File | null>(null);
  const [existingScramble, setExistingScramble] = useState("");
  const [activeCapture, setActiveCapture] = useState<CaptureReceipt | null>(
    null,
  );
  const [busyPath, setBusyPath] = useState<UploadPath | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [listError, setListError] = useState("");
  const [removingCaptureId, setRemovingCaptureId] = useState("");
  const [removeMessage, setRemoveMessage] = useState("");
  const [removeError, setRemoveError] = useState("");

  const videoUploadLimit =
    capabilities?.upload_limits?.video_bytes ?? DEFAULT_VIDEO_UPLOAD_BYTES;

  const loadCaptures = useCallback(async () => {
    setListError("");
    try {
      const payload = await fetchCaptures();
      setCaptures(payload.captures);
    } catch (reason) {
      setListError(
        reason instanceof Error
          ? reason.message
          : "Existing workspace recordings could not be read.",
      );
    }
  }, []);

  useEffect(() => {
    void loadCaptures();
  }, [loadCaptures]);

  useEffect(() => {
    if (!requestedCaptureId) return;
    const requested =
      captures.find(
        (capture) => capture.capture_id === requestedCaptureId,
      ) ?? null;
    setActiveCapture(requested);
  }, [captures, requestedCaptureId]);

  const selectFile = (
    path: UploadPath,
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const next = event.target.files?.[0] ?? null;
    setError("");
    setMessage("");
    if (next && next.size > videoUploadLimit) {
      event.target.value = "";
      setError(
        `This file is ${formatBytes(next.size)}. The largest supported upload is ${formatBytes(videoUploadLimit)}.`,
      );
      if (path === "record") {
        setPrimaryFile(null);
        setPrimaryConfirmed(false);
      } else {
        setExistingFile(null);
      }
      return;
    }
    if (path === "record") {
      setPrimaryFile(next);
      setPrimaryConfirmed(false);
    } else {
      setExistingFile(next);
    }
  };

  const finishImport = (receipt: CaptureReceipt) => {
    setActiveCapture(receipt);
    setCaptures((current) => [
      receipt,
      ...current.filter(
        (capture) => capture.capture_id !== receipt.capture_id,
      ),
    ]);
    setPrimaryFile(null);
    setPrimaryConfirmed(false);
    setExistingFile(null);
    setExistingScramble("");
    if (primaryFileInputRef.current) primaryFileInputRef.current.value = "";
    if (existingFileInputRef.current) existingFileInputRef.current.value = "";
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set(CAPTURE_PARAM, receipt.capture_id);
    setSearchParams(nextParams, { replace: true });
    setMessage("Video added. Choose a calibration below.");
  };

  const upload = async (
    path: UploadPath,
    file: File | null,
    scramble: string,
  ) => {
    setError("");
    setMessage("");
    if (!file) {
      setError(
        path === "record"
          ? "Choose the video you recorded."
          : "Choose an existing video.",
      );
      return;
    }
    if (path === "record" && !primaryConfirmed) {
      setError("Confirm that the recording starts from the displayed scramble.");
      return;
    }

    let normalizedScramble: string;
    try {
      normalizedScramble = normalizeCanonicalScramble(scramble);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Enter a canonical starting scramble.",
      );
      return;
    }

    setBusyPath(path);
    setMessage("Adding video to the local workspace…");
    try {
      const receipt = await importCapture(file, "", normalizedScramble, "");
      finishImport(receipt);
    } catch (reason) {
      setMessage("");
      setError(
        reason instanceof Error
          ? reason.message
          : "The video could not be added to the workspace.",
      );
    } finally {
      setBusyPath(null);
    }
  };

  const submitRecorded = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void upload("record", primaryFile, recordingScramble);
  };

  const submitExisting = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void upload("existing", existingFile, existingScramble);
  };

  const newScramble = () => {
    setRecordingScramble(
      replaceRecordingScramble(sessionStorageOrNull()),
    );
    setPrimaryFile(null);
    setPrimaryConfirmed(false);
    setMessage("");
    setError("");
    if (primaryFileInputRef.current) primaryFileInputRef.current.value = "";
  };

  const updateCapture = useCallback((next: CaptureReceipt) => {
    setActiveCapture(next);
    setCaptures((current) => [
      next,
      ...current.filter(
        (capture) => capture.capture_id !== next.capture_id,
      ),
    ]);
  }, []);

  const captureGroups = useMemo(
    () => orderDecodeCaptures(captures),
    [captures],
  );

  const removeCapture = async (capture: CaptureReceipt) => {
    const publishedNote = isPublishedDatasetCapture(capture)
      ? " The published copy is untouched and can be downloaded again."
      : "";
    const confirmed = window.confirm(
      `Remove "${capture.original_filename}" from this workspace? The video and labels move to your system Trash. Saved Runs stay, but video playback will be unavailable.${publishedNote}`,
    );
    if (!confirmed) return;

    setRemovingCaptureId(capture.capture_id);
    setRemoveMessage("");
    setRemoveError("");
    try {
      await deleteCapture(capture.capture_id);
      setCaptures((current) =>
        current.filter((entry) => entry.capture_id !== capture.capture_id),
      );
      if (
        activeCapture?.capture_id === capture.capture_id ||
        requestedCaptureId === capture.capture_id
      ) {
        setActiveCapture(null);
        const nextParams = new URLSearchParams(searchParams);
        nextParams.delete(CAPTURE_PARAM);
        setSearchParams(nextParams, { replace: true });
      }
      setRemoveMessage(
        `${capture.original_filename} moved to Trash.`,
      );
      await loadCaptures();
    } catch (reason) {
      setRemoveError(
        reason instanceof Error
          ? reason.message
          : "The workspace video could not be moved to Trash.",
      );
    } finally {
      setRemovingCaptureId("");
    }
  };

  const recordingTokens = scrambleTokens(recordingScramble);

  return (
    <div className="page import-workbench-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Workspace input</p>
          <h1>Add video</h1>
          <p className="page-description">
            Record a solve or add an existing recording for Decode. Each video
            needs its exact starting scramble and a color calibration.
          </p>
        </div>
      </header>

      <section
        className="station-card import-source-card"
        aria-labelledby="import-source-title"
      >
        <div className="station-card-head import-source-head">
          <div>
            <p className="station-code">Step 1</p>
            <h2 id="import-source-title">Record a new solve</h2>
          </div>
          <span className="import-primary-label">Recommended</span>
        </div>

        <form className="import-record-form" onSubmit={submitRecorded}>
          <div className="import-scramble-head">
            <div>
              <strong>Apply this scramble</strong>
              <small>{recordingTokens.length} canonical moves</small>
            </div>
            <button
              className="text-button"
              type="button"
              disabled={busyPath !== null}
              onClick={newScramble}
            >
              New scramble
            </button>
          </div>

          <div
            className="import-scramble-tape"
            aria-label={`Starting scramble: ${recordingScramble}`}
          >
            {recordingTokens.map((token, index) => (
              <span
                className={`import-scramble-move import-scramble-face-${token[0].toLowerCase()}`}
                key={`${index}-${token}`}
              >
                {token}
              </span>
            ))}
          </div>

          <p className="import-record-guidance">
            Apply the scramble, then record the full solve in your normal camera
            app. Transfer the original video to this computer.
          </p>

          <div className="import-record-upload">
            <label className="import-video-picker">
              <input
                ref={primaryFileInputRef}
                type="file"
                accept={VIDEO_ACCEPT}
                disabled={busyPath !== null}
                onChange={(event) => selectFile("record", event)}
              />
              <span>
                <strong>
                  {primaryFile ? primaryFile.name : "Choose the recording"}
                </strong>
                <small>
                  {primaryFile
                    ? `${formatBytes(primaryFile.size)} selected`
                    : `Original video, up to ${formatBytes(videoUploadLimit)}.`}
                </small>
              </span>
            </label>

            <label className="import-record-confirm">
              <input
                type="checkbox"
                checked={primaryConfirmed}
                disabled={busyPath !== null || !primaryFile}
                onChange={(event) =>
                  setPrimaryConfirmed(event.target.checked)
                }
              />
              <span>
                This recording starts from the displayed scramble.
              </span>
            </label>

            <button
              className="button button-primary import-submit"
              type="submit"
              disabled={busyPath !== null || !primaryFile || !primaryConfirmed}
            >
              {busyPath === "record" ? "Adding video…" : "Add to workspace"}
            </button>
          </div>
        </form>

        <details className="import-existing-panel">
          <summary>
            <span>
              <strong>Use an existing recording</strong>
              <small>For a video whose starting scramble you already know</small>
            </span>
          </summary>
          <form className="import-existing-form" onSubmit={submitExisting}>
            <label className="import-video-picker">
              <input
                ref={existingFileInputRef}
                type="file"
                accept={VIDEO_ACCEPT}
                disabled={busyPath !== null}
                onChange={(event) => selectFile("existing", event)}
              />
              <span>
                <strong>
                  {existingFile ? existingFile.name : "Choose an existing video"}
                </strong>
                <small>
                  {existingFile
                    ? `${formatBytes(existingFile.size)} selected`
                    : "Use the original recording file."}
                </small>
              </span>
            </label>

            <label
              className="field-label import-scramble-field"
              htmlFor="import-existing-scramble"
            >
              <span>Exact starting scramble</span>
              <small>Canonical face moves, separated by spaces</small>
              <input
                className="text-input"
                id="import-existing-scramble"
                value={existingScramble}
                onChange={(event) => setExistingScramble(event.target.value)}
                placeholder="R U R' U'"
                maxLength={500}
                autoComplete="off"
                spellCheck={false}
                disabled={busyPath !== null}
              />
            </label>

            <button
              className="button button-secondary"
              type="submit"
              disabled={busyPath !== null}
            >
              {busyPath === "existing"
                ? "Adding video…"
                : "Add existing video"}
            </button>
          </form>
        </details>

        {message && (
          <p className="import-message" role="status">
            {message}
          </p>
        )}
        {error && (
          <p className="import-message import-message-error" role="alert">
            {error}
          </p>
        )}
      </section>

      <section className="station-card" aria-labelledby="import-prepare-title">
        <div className="station-card-head">
          <div>
            <p className="station-code">Step 2</p>
            <h2 id="import-prepare-title">Calibrate colors</h2>
          </div>
        </div>
        {activeCapture ? (
          <>
            <div className="import-receipt">
              <div>
                <span>Workspace video</span>
                <strong>{activeCapture.original_filename}</strong>
              </div>
              <code>{recordingSummary(activeCapture)}</code>
            </div>
            <PrepareForDecodeCard
              capture={activeCapture}
              captures={captures}
              onUpdated={updateCapture}
            />
          </>
        ) : (
          <p className="step-locked-sentence">
            Add a recording above before calibrating its colors.
          </p>
        )}
      </section>

      <section
        className="station-card import-next-card"
        aria-labelledby="import-next-title"
      >
        <div className="station-card-head">
          <div>
            <p className="station-code">Step 3</p>
            <h2 id="import-next-title">Continue</h2>
          </div>
        </div>
        {activeCapture ? (
          <div className="import-next-actions">
            <Link
              className="import-next-action"
              to={captureHref("/decode", activeCapture.capture_id)}
            >
              <span>
                <strong>Decode</strong>
                <small>Check readiness and choose a CUDA runner.</small>
              </span>
              <b aria-hidden="true">→</b>
            </Link>
            <Link
              className="import-next-action"
              to={captureHref("/label", activeCapture.capture_id)}
            >
              <span>
                <strong>Label</strong>
                <small>Open the same video in the annotation tool.</small>
              </span>
              <b aria-hidden="true">→</b>
            </Link>
          </div>
        ) : (
          <p className="step-locked-sentence">
            The recording will open in Decode or Label without another copy.
          </p>
        )}
      </section>

      {listError && (
        <div className="tool-message tool-message-error" role="alert">
          <p>{listError}</p>
          <button
            className="button button-secondary"
            type="button"
            onClick={() => void loadCaptures()}
          >
            Retry workspace list
          </button>
        </div>
      )}

      <details className="import-workspace-manager">
        <summary>
          <span>
            <strong>Manage workspace videos</strong>
            <small>Move local video bundles to your system Trash</small>
          </span>
        </summary>
        <div className="import-workspace-manager-body">
          {removeMessage && (
            <p className="import-workspace-message" role="status">
              {removeMessage}
            </p>
          )}
          {removeError && (
            <p
              className="import-workspace-message import-workspace-message-error"
              role="alert"
            >
              {removeError}
            </p>
          )}
          {captures.length === 0 ? (
            <p className="import-workspace-empty">No workspace videos.</p>
          ) : (
            <>
              {captureGroups.yourRecordings.length > 0 && (
                <section
                  className="import-workspace-group"
                  aria-labelledby="import-workspace-personal"
                >
                  <h3 id="import-workspace-personal">Your recordings</h3>
                  <ul>
                    {captureGroups.yourRecordings.map((capture) => (
                      <li
                        className="import-workspace-video-row"
                        key={capture.capture_id}
                      >
                        <span>
                          <strong>{capture.original_filename}</strong>
                          <small>{recordingSummary(capture)}</small>
                        </span>
                        <button
                          className="button button-quiet"
                          type="button"
                          disabled={removingCaptureId !== ""}
                          onClick={() => void removeCapture(capture)}
                        >
                          {removingCaptureId === capture.capture_id
                            ? "Moving…"
                            : "Move to Trash"}
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {captureGroups.publishedDataset.length > 0 && (
                <section
                  className="import-workspace-group"
                  aria-labelledby="import-workspace-published"
                >
                  <h3 id="import-workspace-published">Published dataset</h3>
                  <ul>
                    {captureGroups.publishedDataset.map((capture) => (
                      <li
                        className="import-workspace-video-row"
                        key={capture.capture_id}
                      >
                        <span>
                          <strong>{capture.original_filename}</strong>
                          <small>{recordingSummary(capture)}</small>
                        </span>
                        <button
                          className="button button-quiet"
                          type="button"
                          disabled={removingCaptureId !== ""}
                            onClick={() => void removeCapture(capture)}
                          >
                            {removingCaptureId === capture.capture_id
                              ? "Moving…"
                              : "Move to Trash"}
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          )}
        </div>
      </details>
    </div>
  );
}
