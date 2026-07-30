import { useEffect, useMemo, useState } from "react";

import {
  attachCaptureSidecar,
  attachReusedCalibration,
  fetchCaptures,
  sealCapture,
} from "../api";
import type { CaptureReceipt } from "../types";
import { ColorCalibrationSampler } from "./ColorCalibrationSampler";
import "./PrepareForDecodeCard.css";

type RowBusy = "calibration" | "seal" | null;

// "bundled" (the published dataset's shared calibration), "capture:<id>"
// (reuse another capture's own calibration), or "upload" (a new file).
// Encoded as one string so the selector needs only one <select>, one value,
// and one Apply action.
const BUNDLED_SOURCE = "bundled";
const UPLOAD_SOURCE = "upload";
const CAPTURE_SOURCE_PREFIX = "capture:";
const NO_CALIBRATION_SOURCE = "";

function shortDate(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? iso
    : parsed.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function attachedCalibrationLabel(capture: CaptureReceipt): string {
  const displayName = capture.calibration?.display_name?.trim();
  return displayName && displayName.length <= 200
    ? displayName
    : "Calibration";
}

/**
 * Step 2 of the reconstruction pipeline: sample or attach color calibration,
 * then lock the exact video and scramble for decode.
 */
export function PrepareForDecodeCard({
  capture,
  captures,
  onUpdated,
}: {
  capture: CaptureReceipt;
  // The page's full capture list, read once at the top of the pipeline.
  // Reused here only to list other captures that already carry a
  // calibration; enumerating that is free, since the data is already in
  // hand, not a new fetch.
  captures: CaptureReceipt[];
  onUpdated: (next: CaptureReceipt) => void;
}) {
  const [busy, setBusy] = useState<RowBusy>(null);
  const [message, setMessage] = useState("");
  const [messageIsError, setMessageIsError] = useState(false);
  const [samplerBusy, setSamplerBusy] = useState(false);
  const [replaceSamplerOpen, setReplaceSamplerOpen] = useState(false);
  const [calibrationSource, setCalibrationSource] = useState<string>(
    NO_CALIBRATION_SOURCE,
  );
  const [calibrationFile, setCalibrationFile] = useState<File | null>(null);

  const sealed = capture.state === "sealed";
  const sealedForDecode = sealed && capture.seal_purpose === "decode";
  const sealedForAnotherPurpose = sealed && !sealedForDecode;
  const calibrationEditable = !sealed || sealedForDecode;
  const encodedWidth = capture.video.encoded_width;
  const encodedHeight = capture.video.encoded_height;
  const actualFps = capture.video.actual_fps;
  const measuredFps =
    typeof actualFps === "number" &&
    Number.isFinite(actualFps) &&
    actualFps > 0
      ? actualFps
      : null;
  const measuredWidth =
    typeof encodedWidth === "number" &&
    Number.isFinite(encodedWidth) &&
    encodedWidth > 0
      ? encodedWidth
      : null;
  const measuredHeight =
    typeof encodedHeight === "number" &&
    Number.isFinite(encodedHeight) &&
    encodedHeight > 0
      ? encodedHeight
      : null;
  const metadataKnown =
    capture.probe.status === "ok" &&
    measuredFps !== null &&
    measuredWidth !== null &&
    measuredHeight !== null;
  const native240Source =
    metadataKnown && measuredFps! >= 220 && measuredFps! <= 242;
  const standardFrameRate =
    metadataKnown && measuredFps! >= 110 && measuredFps! <= 121;
  const standardResolution =
    metadataKnown && Math.min(measuredWidth!, measuredHeight!) >= 1080;
  const nonstandardMedia =
    metadataKnown &&
    !native240Source &&
    (!standardFrameRate || !standardResolution);
  const mediaReady = metadataKnown && !native240Source;
  const canSeal =
    Boolean(capture.calibration) &&
    Boolean(capture.solve?.scramble) &&
    mediaReady;
  const otherCalibratedCaptures = useMemo(
    () =>
      captures.filter(
        (other) => other.capture_id !== capture.capture_id && other.calibration,
      ),
    [captures, capture.capture_id],
  );
  const calibrationReady =
    calibrationSource !== NO_CALIBRATION_SOURCE &&
    (calibrationSource === UPLOAD_SOURCE ? calibrationFile !== null : true);

  useEffect(() => {
    setCalibrationSource(NO_CALIBRATION_SOURCE);
    setCalibrationFile(null);
    setMessage("");
    setMessageIsError(false);
    setSamplerBusy(false);
    setReplaceSamplerOpen(false);
  }, [capture.capture_id]);

  // A stale row (attached from another tab, already sealed by a previous
  // click this row missed) would otherwise keep offering controls that
  // 400 forever. Re-reading the capture list and splicing this one row's
  // fresh receipt back in, on error as well as on success, keeps the row
  // honest instead of stuck.
  const refetchRow = async () => {
    try {
      const payload = await fetchCaptures();
      const fresh = payload.captures.find(
        (candidate) => candidate.capture_id === capture.capture_id,
      );
      if (fresh) onUpdated(fresh);
    } catch {
      // Best-effort: leave the row as it was rather than compounding one
      // failure with a second one from this background refresh.
    }
  };

  const applyCalibration = async () => {
    if (!calibrationReady) return;
    setBusy("calibration");
    setMessageIsError(false);
    setMessage("Attaching calibration…");
    try {
      const updated = calibrationSource.startsWith(CAPTURE_SOURCE_PREFIX)
        ? await attachReusedCalibration(
            capture.capture_id,
            "capture",
            calibrationSource.slice(CAPTURE_SOURCE_PREFIX.length),
          )
        : calibrationSource === UPLOAD_SOURCE
          ? await attachCaptureSidecar(
              capture.capture_id,
              "calibration",
              calibrationFile as File,
            )
          : await attachReusedCalibration(capture.capture_id, "bundled");
      onUpdated(updated);
      setCalibrationFile(null);
      setMessage("Calibration attached.");
    } catch (error) {
      setMessageIsError(true);
      setMessage(
        error instanceof Error ? error.message : "The calibration could not be attached.",
      );
      await refetchRow();
    } finally {
      setBusy(null);
    }
  };

  const seal = async () => {
    const confirmed = window.confirm(
      "Lock this video and starting scramble for Decode? You can still replace the calibration and run more attempts.",
    );
    if (!confirmed) return;
    const alreadySealed = sealed;
    setBusy("seal");
    setMessageIsError(false);
    setMessage(alreadySealed ? "Confirming the existing lock…" : "Locking for decode…");
    try {
      const updated = await sealCapture(capture.capture_id, "decode");
      onUpdated(updated);
      setMessage(
        alreadySealed
          ? "Video and scramble are already locked."
          : "Video and scramble locked for decode.",
      );
    } catch (error) {
      setMessageIsError(true);
      setMessage(
        error instanceof Error ? error.message : "The capture could not be locked.",
      );
      await refetchRow();
    } finally {
      setBusy(null);
    }
  };

  const sampler = (
    <ColorCalibrationSampler
      key={capture.capture_id}
      captureId={capture.capture_id}
      videoName={capture.original_filename}
      fps={measuredFps ?? capture.probe.fps ?? 30}
      onBusyChange={setSamplerBusy}
      onCreated={(updated) => {
        setMessageIsError(false);
        setMessage("Calibration created from this video.");
        onUpdated(updated);
      }}
    />
  );

  return (
    <div className="prepare-card">
      {sealedForAnotherPurpose && (
        <p className="prepare-message prepare-message-error" role="alert">
          This capture is locked for {capture.seal_purpose ?? "another use"}.
          That lock cannot be changed to Decode.
        </p>
      )}
      <div className={`prepare-row${capture.calibration ? " prepare-row-attached" : ""}`}>
        <div className="prepare-row-heading">
          <span className="prepare-row-letter" aria-hidden="true">
            a
          </span>
          <div>
            <strong>Attach color calibration</strong>
            <p>
              Sample six stickers from this recording. Color conversion and
              validation happen on the local server.
            </p>
          </div>
        </div>
        {capture.calibration ? (
          <div className="prepare-row-state">
            <span className="prepare-row-check">Attached</span>
            <span className="prepare-row-name">
              {attachedCalibrationLabel(capture)}
            </span>
          </div>
        ) : null}
        {calibrationEditable && !capture.calibration && sampler}
        {calibrationEditable && capture.calibration && (
          <details
            className="prepare-calibration-disclosure prepare-video-replace"
            open={replaceSamplerOpen}
            onToggle={(event) =>
              setReplaceSamplerOpen(event.currentTarget.open)
            }
          >
            <summary>
              <span>
                <strong>Replace by sampling this video</strong>
                <small>Choose six new sticker crops</small>
              </span>
            </summary>
            {replaceSamplerOpen && sampler}
          </details>
        )}
        {calibrationEditable && (
          <details className="prepare-calibration-disclosure prepare-alternate-calibration">
            <summary>
              <span>
                <strong>Use another calibration source</strong>
                <small>Shared, reused, or uploaded calibration</small>
              </span>
            </summary>
            <div className="prepare-calibration-picker">
              <label className="field-label" htmlFor="prepare-calibration-source">
                <span>{capture.calibration ? "Replace with" : "Source"}</span>
              </label>
              <select
                id="prepare-calibration-source"
                className="text-input"
                value={calibrationSource}
                disabled={busy !== null || samplerBusy}
                onChange={(event) => {
                  setCalibrationSource(event.target.value);
                  setCalibrationFile(null);
                }}
              >
                <option value={NO_CALIBRATION_SOURCE}>
                  Choose a calibration…
                </option>
                <option value={BUNDLED_SOURCE}>
                  Published dataset shared calibration
                </option>
                {otherCalibratedCaptures.map((other) => (
                  <option
                    key={other.capture_id}
                    value={`${CAPTURE_SOURCE_PREFIX}${other.capture_id}`}
                  >
                    From {other.original_filename} · {shortDate(other.created_at)}
                  </option>
                ))}
                <option value={UPLOAD_SOURCE}>Upload a file</option>
              </select>
              {calibrationSource === UPLOAD_SOURCE && (
                <label className="file-action">
                  <input
                    type="file"
                    accept=".json,application/json"
                    disabled={busy !== null || samplerBusy}
                    onChange={(event) =>
                      setCalibrationFile(event.target.files?.[0] ?? null)
                    }
                  />
                  <strong>
                    {calibrationFile ? calibrationFile.name : "Choose a file"}
                  </strong>
                </label>
              )}
              {calibrationSource === BUNDLED_SOURCE && (
                <p className="prepare-calibration-warning" role="note">
                  This shared calibration is allowed for any video. It was
                  measured for one camera and lighting cohort, so a different
                  cube, camera, lens, or lighting may reduce decode quality.
                </p>
              )}
              <button
                className="button button-secondary"
                type="button"
                disabled={
                  busy !== null || samplerBusy || !calibrationReady
                }
                onClick={() => void applyCalibration()}
              >
                {busy === "calibration"
                  ? "Attaching…"
                  : capture.calibration
                    ? "Replace"
                    : "Attach"}
              </button>
            </div>
          </details>
        )}
      </div>

      <div className={`prepare-row${sealed ? " prepare-row-attached" : ""}`}>
        <div className="prepare-row-heading">
          <span className="prepare-row-letter" aria-hidden="true">
            b
          </span>
          <div>
            <strong>Lock video + scramble</strong>
            <p>
              Decode attempts keep these inputs fixed. Calibration stays
              replaceable, and every run saves the exact calibration it used.
            </p>
          </div>
        </div>
        {sealed ? (
          <p
            className={
              sealedForDecode
                ? "prepare-row-check"
                : "prepare-row-check prepare-row-check-blocked"
            }
          >
            {sealedForDecode
              ? "Video + scramble locked."
              : "Locked for label."}
          </p>
        ) : (
          <>
            {!metadataKnown && (
              <p className="prepare-message prepare-message-error" role="alert">
                Decode needs readable frame-rate and resolution metadata.
                Re-import or repair this video before locking it.
              </p>
            )}
            {native240Source && (
              <p className="prepare-message prepare-message-error" role="alert">
                Preserve this native 220–242 fps original and create its linked
                120 fps derivative for Decode.
              </p>
            )}
            {nonstandardMedia && (
              <p className="prepare-message prepare-message-warning" role="note">
                This video measures {measuredFps!.toFixed(3)} fps at{" "}
                {measuredWidth}×{measuredHeight}. The expected setup is 110–121 fps with an
                encoded short edge of at least 1080 pixels. You can continue,
                but this warning will remain attached to readiness and the run.
              </p>
            )}
            <button
              className="button button-primary"
              type="button"
              disabled={busy !== null || samplerBusy || !canSeal}
              title={
                canSeal
                  ? ""
                  : "Attach a calibration, keep the exact starting scramble, and provide readable video metadata before sealing."
              }
              onClick={() => void seal()}
            >
              {busy === "seal" ? "Locking…" : "Lock for decode"}
            </button>
          </>
        )}
      </div>

      {message && (
        <p
          className={`prepare-message${messageIsError ? " prepare-message-error" : ""}`}
          role={messageIsError ? "alert" : "status"}
        >
          {message}
        </p>
      )}
    </div>
  );
}
