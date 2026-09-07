import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  createCaptureMediaTicket,
  fetchCapabilities,
  fetchCaptures,
  isAbortError,
} from "../api";
import {
  CAPTURE_PARAM,
  explicitCaptureIdFromParams,
  orderDecodeCaptures,
} from "../captureSelection";
import {
  ROUTE_GUIDE_LINK_REL,
  ROUTE_GUIDE_LINK_TARGET,
  routeGuideDocumentHref,
} from "../routeGuideLinks";
import { useRemoteHostSelection } from "../remoteHosts";
import type { Capabilities, CaptureReceipt } from "../types";
import { DecodeStage } from "./DecodeStage";
import { PrepareForDecodeCard } from "./PrepareForDecodeCard";

type LoadState = "loading" | "success" | "error";

function captureDateLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      });
}

function StepHeader({
  step,
  title,
  titleId,
}: {
  step: number;
  title: string;
  titleId: string;
}) {
  return (
    <div className="station-card-head">
      <div>
        <p className="station-code">Step {step}</p>
        <h2 id={titleId}>{title}</h2>
      </div>
    </div>
  );
}

/**
 * The normal OSS decode path has three jobs: choose a workspace capture,
 * prepare and lock it, then run the standard camera-only decoder. Runs stays
 * a separate inspector, and capture tracking stays out of this focused path.
 */
export function ReconstructWorkbench() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialCaptureParamsRef = useRef(new URLSearchParams(searchParams));

  const [captures, setCaptures] = useState<CaptureReceipt[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [captureId, setCaptureId] = useState("");
  const [captureLoadState, setCaptureLoadState] =
    useState<LoadState>("loading");
  const [captureError, setCaptureError] = useState("");
  const [capabilitiesError, setCapabilitiesError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [previewRetry, setPreviewRetry] = useState(0);
  const computeSelection = useRemoteHostSelection();

  const loadCaptures = useCallback(async () => {
    setCaptureLoadState("loading");
    setCaptureError("");
    try {
      const payload = await fetchCaptures();
      setCaptures(payload.captures);
      setCaptureId((current) => {
        if (
          current &&
          payload.captures.some((capture) => capture.capture_id === current)
        ) {
          return current;
        }
        return explicitCaptureIdFromParams(
          initialCaptureParamsRef.current,
          payload.captures,
        );
      });
      setCaptureLoadState("success");
    } catch (reason) {
      if (isAbortError(reason)) return;
      setCaptureError(
        reason instanceof Error
          ? reason.message
          : "The capture workspace could not be read.",
      );
      setCaptureLoadState("error");
    }
  }, []);

  const loadCapabilities = useCallback(async () => {
    setCapabilitiesError("");
    try {
      setCapabilities(await fetchCapabilities());
    } catch (reason) {
      if (isAbortError(reason)) return;
      setCapabilities(null);
      setCapabilitiesError(
        reason instanceof Error
          ? reason.message
          : "The workbench capabilities could not be read.",
      );
    }
  }, []);

  useEffect(() => {
    void loadCaptures();
  }, [loadCaptures]);

  useEffect(() => {
    void loadCapabilities();
  }, [loadCapabilities]);

  useEffect(() => {
    setPreviewUrl("");
    setPreviewError("");
    if (!captureId) return;
    let active = true;
    void createCaptureMediaTicket(captureId)
      .then((ticket) => {
        if (active) setPreviewUrl(ticket.url);
      })
      .catch((reason) => {
        if (!active) return;
        setPreviewError(
          reason instanceof Error
            ? reason.message
            : "The capture video could not be opened.",
        );
      });
    return () => {
      active = false;
    };
  }, [captureId, previewRetry]);

  const selectCaptureId = useCallback(
    (next: string) => {
      setCaptureId(next);
      if (searchParams.get(CAPTURE_PARAM) === next) return;
      const nextParams = new URLSearchParams(searchParams);
      if (next) nextParams.set(CAPTURE_PARAM, next);
      else nextParams.delete(CAPTURE_PARAM);
      setSearchParams(nextParams, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  const updateCapture = useCallback((next: CaptureReceipt) => {
    setCaptures((current) =>
      current.map((existing) =>
        existing.capture_id === next.capture_id ? next : existing,
      ),
    );
  }, []);

  const noCaptures =
    captureLoadState === "success" && captures.length === 0;
  const duplicateCaptureFilenames = useMemo(() => {
    const filenameCounts = new Map<string, number>();
    for (const capture of captures) {
      filenameCounts.set(
        capture.original_filename,
        (filenameCounts.get(capture.original_filename) ?? 0) + 1,
      );
    }
    return new Set(
      [...filenameCounts]
        .filter(([, count]) => count > 1)
        .map(([filename]) => filename),
    );
  }, [captures]);
  const captureGroups = useMemo(
    () => orderDecodeCaptures(captures),
    [captures],
  );
  const selectedCapture = useMemo(
    () => captures.find((entry) => entry.capture_id === captureId) ?? null,
    [captureId, captures],
  );
  const captureRevision = useMemo(
    () =>
      selectedCapture
        ? [
            selectedCapture.state ?? "",
            selectedCapture.seal_purpose ?? "",
            selectedCapture.solve?.scramble ?? "",
            selectedCapture.calibration?.sha256 ?? "",
            selectedCapture.video.sha256 ?? "",
            selectedCapture.video.actual_fps ?? "",
            selectedCapture.video.encoded_width ?? "",
            selectedCapture.video.encoded_height ?? "",
          ].join(":")
        : "",
    [selectedCapture],
  );

  return (
    <div className="page reconstruct-page decode-workbench-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Camera-only reconstruction</p>
          <h1>Decode</h1>
          <p className="page-description">
            Prepare one workspace capture, choose its CUDA runner, and review
            the reconstructed move sequence.
          </p>
        </div>
        <Link className="text-button" to="/demo">
          View the no-GPU demo
        </Link>
      </header>

      <aside
        className="decode-dataset-onboarding"
        aria-labelledby="decode-dataset-onboarding-title"
      >
        <p id="decode-dataset-onboarding-title">
          Download the published Hugging Face dataset and Decode support. Its
          downloaded videos appear here automatically. Choose a frame-zero-scrambled
          recording, attach calibration, and lock its video and scramble.
        </p>
        <code>
          make download-dataset download-assets
        </code>
        <a
          className="text-button"
          href={routeGuideDocumentHref("docs/DATASET.md")}
          target={ROUTE_GUIDE_LINK_TARGET}
          rel={ROUTE_GUIDE_LINK_REL}
        >
          Dataset preparation
        </a>
      </aside>

      {capabilitiesError && (
        <div className="tool-message tool-message-error" role="alert">
          <p>{capabilitiesError}</p>
          <button
            className="button button-secondary"
            type="button"
            onClick={() => void loadCapabilities()}
          >
            Retry capability check
          </button>
        </div>
      )}

      <section
        className="station-card reconstruct-capture-bar"
        aria-labelledby="reconstruct-capture-title"
      >
        <StepHeader
          step={1}
          title="Choose a recording"
          titleId="reconstruct-capture-title"
        />
        <p className="reconstruct-capture-note">
          Decode accepts prepared workspace captures. Choose the published
          dataset recording or your own compatible video after importing it.
        </p>
        <div className="reconstruct-capture-picker">
          <label className="field-label" htmlFor="reconstruct-capture">
            <span>Workspace capture</span>
            <small>The selection stays in this page&apos;s URL</small>
          </label>
          <select
            className="text-input"
            id="reconstruct-capture"
            value={captureId}
            disabled={
              captureLoadState !== "success" || captures.length === 0
            }
            onChange={(event) => selectCaptureId(event.target.value)}
          >
            <option value="">
              {captureLoadState === "loading"
                ? "Loading captures…"
                : captureLoadState === "error"
                  ? "Captures unavailable"
                  : captures.length === 0
                    ? "No captures yet"
                    : "Choose a capture…"}
            </option>
            {captureGroups.yourRecordings.length > 0 && (
              <optgroup label="Your recordings">
                {captureGroups.yourRecordings.map((capture) => (
                  <option key={capture.capture_id} value={capture.capture_id}>
                    {capture.original_filename}
                    {duplicateCaptureFilenames.has(capture.original_filename)
                      ? ` · ${captureDateLabel(capture.created_at)}`
                      : ""}
                  </option>
                ))}
              </optgroup>
            )}
            {captureGroups.publishedDataset.length > 0 && (
              <optgroup label="Published dataset">
                {captureGroups.publishedDataset.map((capture) => (
                  <option key={capture.capture_id} value={capture.capture_id}>
                    {capture.original_filename}
                    {duplicateCaptureFilenames.has(capture.original_filename)
                      ? ` · ${captureDateLabel(capture.created_at)}`
                      : ""}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          {captureLoadState === "error" ? (
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void loadCaptures()}
            >
              Retry capture list
            </button>
          ) : noCaptures ? (
            <Link className="text-button" to="/import">
              Import your own video
            </Link>
          ) : null}
        </div>

        <div className="reconstruct-capture-preview">
          {!captureId ? (
            <p className="step-empty-sentence">
              Choose a recording to preview it.
            </p>
          ) : previewError ? (
            <div className="tool-message tool-message-error" role="alert">
              <p>{previewError}</p>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => setPreviewRetry((current) => current + 1)}
              >
                Retry preview
              </button>
            </div>
          ) : previewUrl ? (
            <video
              className="reconstruct-capture-preview-video"
              src={previewUrl}
              controls
              muted
              playsInline
              preload="metadata"
              onError={() =>
                setPreviewError(
                  "The capture ticket was issued, but this browser could not open the video.",
                )
              }
            />
          ) : (
            <p className="step-empty-sentence">Loading preview…</p>
          )}
        </div>
      </section>

      {captureError && (
        <p className="tool-message tool-message-error" role="alert">
          {captureError}
        </p>
      )}

      <section
        className="station-card"
        aria-labelledby="reconstruct-prepare-title"
      >
        <StepHeader
          step={2}
          title="Prepare for decode"
          titleId="reconstruct-prepare-title"
        />
        {selectedCapture === null ? (
          <p className="step-locked-sentence">
            Choose a recording above to prepare it.
          </p>
        ) : (
          <PrepareForDecodeCard
            capture={selectedCapture}
            captures={captures}
            onUpdated={updateCapture}
          />
        )}
      </section>

      {selectedCapture === null ? (
        <section
          className="station-card"
          id="reconstruct-decode"
          aria-labelledby="reconstruct-decode-title"
        >
          <StepHeader
            step={3}
            title="Decode and replay"
            titleId="reconstruct-decode-title"
          />
          <p className="step-locked-sentence">
            Choose a recording above to check readiness and run the decoder.
          </p>
        </section>
      ) : (
        <section
          className="reconstruct-decode-anchor"
          id="reconstruct-decode"
          aria-label="Decode and replay"
        >
          <DecodeStage
            captures={captures}
            capabilities={capabilities}
            capabilitiesError={capabilitiesError}
            onCapabilitiesRetry={() => void loadCapabilities()}
            showCapabilitiesError={false}
            captureId={captureId}
            captureRevision={captureRevision}
            remoteHosts={computeSelection.hosts}
            remoteHostId={computeSelection.selectedId}
            onRemoteHostIdChange={computeSelection.setSelectedId}
            remoteHostsError={computeSelection.error}
            remoteHostsLoading={computeSelection.loading}
            onRemoteHostsRefresh={computeSelection.refresh}
          />
        </section>
      )}
    </div>
  );
}
