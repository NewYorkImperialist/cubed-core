import {
  KeyboardEvent,
  MouseEvent,
  PointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  createCaptureCalibration,
  createCaptureMediaTicket,
  type CaptureCalibrationPatches,
} from "../api";
import {
  CALIBRATION_COLORS,
  type CalibrationColor,
  type NativeCropBox,
  type NativeVideoPoint,
  clientPointToNativeVideo,
  moveNativePoint,
  nativeCropBox,
} from "../calibrationSampling";
import type { CaptureReceipt } from "../types";
import "./ColorCalibrationSampler.css";

const OUTPUT_PATCH_SIDE = 96;

const COLOR_LABELS: Record<CalibrationColor, string> = {
  white: "White",
  green: "Green",
  red: "Red",
  blue: "Blue",
  orange: "Orange",
  yellow: "Yellow",
};

interface CalibrationPatch {
  file: File;
  previewUrl: string;
  box: NativeCropBox;
  time: number;
}

type PatchMap = Partial<Record<CalibrationColor, CalibrationPatch>>;

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.00";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function pngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("The browser could not encode the sampled crop."));
    }, "image/png");
  });
}

function revokePreviews(patches: PatchMap): void {
  for (const patch of Object.values(patches)) {
    if (patch) URL.revokeObjectURL(patch.previewUrl);
  }
}

export function ColorCalibrationSampler({
  captureId,
  videoName,
  fps,
  onCreated,
  onBusyChange,
}: {
  captureId: string;
  videoName: string;
  fps: number;
  onCreated: (capture: CaptureReceipt) => void;
  onBusyChange?: (busy: boolean) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const patchRef = useRef<PatchMap>({});
  const ticketGenerationRef = useRef(0);
  const seekingRef = useRef(false);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoLoading, setVideoLoading] = useState(true);
  const [videoError, setVideoError] = useState("");
  const [videoDimensions, setVideoDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [seeking, setSeeking] = useState(false);
  const [activeColor, setActiveColor] =
    useState<CalibrationColor>("white");
  const [pointer, setPointer] = useState<NativeVideoPoint | null>(null);
  const [patches, setPatches] = useState<PatchMap>({});
  const [sampling, setSampling] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  patchRef.current = patches;

  const loadVideo = useCallback(async () => {
    const generation = ++ticketGenerationRef.current;
    setVideoLoading(true);
    setVideoError("");
    setVideoUrl("");
    setVideoDimensions({ width: 0, height: 0 });
    setDuration(0);
    setCurrentTime(0);
    setPlaying(false);
    seekingRef.current = false;
    setSeeking(false);
    setPointer(null);
    try {
      const ticket = await createCaptureMediaTicket(captureId);
      if (ticketGenerationRef.current !== generation) return;
      setVideoUrl(ticket.url);
    } catch (reason) {
      if (ticketGenerationRef.current !== generation) return;
      setVideoLoading(false);
      setVideoError(
        reason instanceof Error
          ? reason.message
          : "Video access could not be opened.",
      );
    }
  }, [captureId]);

  useEffect(() => {
    void loadVideo();
    return () => {
      ticketGenerationRef.current += 1;
    };
  }, [loadVideo]);

  useEffect(
    () => () => {
      revokePreviews(patchRef.current);
    },
    [],
  );

  useEffect(() => {
    onBusyChange?.(submitting);
    return () => onBusyChange?.(false);
  }, [onBusyChange, submitting]);

  const clearPatches = () => {
    revokePreviews(patches);
    setPatches({});
    setActiveColor("white");
    setMessage("");
    setError("");
  };

  const videoBounds = () => {
    const video = videoRef.current;
    if (!video) return null;
    const rect = video.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
  };

  const pointFromClient = (clientX: number, clientY: number) => {
    const bounds = videoBounds();
    if (!bounds) return null;
    return clientPointToNativeVideo(
      clientX,
      clientY,
      bounds,
      videoDimensions.width,
      videoDimensions.height,
    );
  };

  const capturePoint = async (point: NativeVideoPoint) => {
    if (sampling || submitting) return;
    const video = videoRef.current;
    if (seekingRef.current) {
      setError("Wait for the selected video frame to finish loading.");
      return;
    }
    if (
      !video ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      videoDimensions.width < 1 ||
      videoDimensions.height < 1
    ) {
      setError("Wait for the current video frame to finish loading.");
      return;
    }

    video.pause();
    setPlaying(false);
    setPointer(point);
    setSampling(true);
    setError("");
    setMessage(`Sampling ${COLOR_LABELS[activeColor].toLowerCase()}…`);
    try {
      const box = nativeCropBox(
        point,
        videoDimensions.width,
        videoDimensions.height,
      );
      const canvas = document.createElement("canvas");
      canvas.width = OUTPUT_PATCH_SIDE;
      canvas.height = OUTPUT_PATCH_SIDE;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("This browser could not open a 2D sampling canvas.");
      }
      context.imageSmoothingEnabled = false;
      context.drawImage(
        video,
        box.x,
        box.y,
        box.side,
        box.side,
        0,
        0,
        OUTPUT_PATCH_SIDE,
        OUTPUT_PATCH_SIDE,
      );
      const blob = await pngBlob(canvas);
      const file = new File([blob], `${activeColor}.png`, {
        type: "image/png",
      });
      const previous = patches[activeColor];
      if (previous) URL.revokeObjectURL(previous.previewUrl);
      const nextPatches: PatchMap = {
        ...patches,
        [activeColor]: {
          file,
          previewUrl: URL.createObjectURL(file),
          box,
          time: video.currentTime,
        },
      };
      setPatches(nextPatches);
      const nextTarget = CALIBRATION_COLORS.find(
        (color) => !nextPatches[color],
      );
      if (nextTarget) setActiveColor(nextTarget);
      setMessage(
        nextTarget
          ? `${COLOR_LABELS[activeColor]} sampled. Next: ${COLOR_LABELS[nextTarget].toLowerCase()}.`
          : "All six colors are sampled. Review the thumbnails, then create the calibration.",
      );
    } catch (reason) {
      setMessage("");
      setError(
        reason instanceof DOMException && reason.name === "SecurityError"
          ? "The video frame could not be read. Refresh video access and try again."
          : reason instanceof Error
            ? reason.message
            : "The sticker crop could not be sampled.",
      );
    } finally {
      setSampling(false);
    }
  };

  const sampleClientPoint = (clientX: number, clientY: number) => {
    const point = pointFromClient(clientX, clientY);
    if (!point) {
      setError("Click inside the visible video frame, not its letterbox.");
      return;
    }
    void capturePoint(point);
  };

  const handleVideoClick = (event: MouseEvent<SVGSVGElement>) => {
    sampleClientPoint(event.clientX, event.clientY);
  };

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const point = pointFromClient(event.clientX, event.clientY);
    if (point) setPointer(point);
  };

  const handleVideoKey = (event: KeyboardEvent<SVGSVGElement>) => {
    const direction =
      event.key === "ArrowLeft"
        ? "left"
        : event.key === "ArrowRight"
          ? "right"
          : event.key === "ArrowUp"
            ? "up"
            : event.key === "ArrowDown"
              ? "down"
              : null;
    if (direction) {
      event.preventDefault();
      setPointer((current) =>
        moveNativePoint(
          current,
          direction,
          videoDimensions.width,
          videoDimensions.height,
          event.shiftKey,
        ),
      );
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const point =
        pointer ??
        moveNativePoint(
          null,
          "right",
          videoDimensions.width,
          videoDimensions.height,
        );
      void capturePoint(point);
    }
  };

  const seek = (nextTime: number) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(nextTime)) return;
    video.pause();
    const maximum = Number.isFinite(video.duration)
      ? Math.max(0, video.duration - 0.000_001)
      : Math.max(0, nextTime);
    const next = Math.min(maximum, Math.max(0, nextTime));
    if (Math.abs(video.currentTime - next) < 0.000_001) {
      seekingRef.current = false;
      setSeeking(false);
      setCurrentTime(next);
      setPlaying(false);
      return;
    }
    seekingRef.current = true;
    setSeeking(true);
    try {
      video.currentTime = next;
    } catch {
      seekingRef.current = false;
      setSeeking(false);
      setError("The selected video position could not be opened.");
      return;
    }
    setCurrentTime(next);
    setPlaying(false);
  };

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch(() =>
        setVideoError("Video playback could not start. Refresh video access."),
      );
    } else {
      video.pause();
    }
  };

  const submit = async () => {
    if (CALIBRATION_COLORS.some((color) => !patches[color])) return;
    const files = {} as CaptureCalibrationPatches;
    for (const color of CALIBRATION_COLORS) {
      const patch = patches[color];
      if (!patch) return;
      files[color] = patch.file;
    }
    setSubmitting(true);
    setError("");
    setMessage("Creating and attaching calibration…");
    try {
      const updated = await createCaptureCalibration(captureId, files);
      setMessage("Calibration created and attached.");
      onCreated(updated);
    } catch (reason) {
      setMessage("");
      setError(
        reason instanceof Error
          ? reason.message
          : "The calibration could not be created.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const sampledCount = CALIBRATION_COLORS.filter(
    (color) => patches[color],
  ).length;
  const allSampled = sampledCount === CALIBRATION_COLORS.length;
  const ready =
    videoDimensions.width > 0 &&
    videoDimensions.height > 0 &&
    Boolean(videoUrl) &&
    !videoError;
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 30;
  const crosshairRadius = Math.max(
    8,
    Math.round(Math.min(videoDimensions.width, videoDimensions.height) * 0.015),
  );

  return (
    <section
      className="color-calibration-sampler"
      aria-labelledby="color-calibration-title"
    >
      <div className="color-calibration-head">
        <div>
          <span className="color-calibration-recommended">Recommended</span>
          <h3 id="color-calibration-title">Sample this video</h3>
          <p>
            Scrub to a clear frame, choose a target color, then click the
            middle of one sticker.
          </p>
        </div>
        <span className="color-calibration-progress" aria-live="polite">
          {sampledCount} / 6 sampled
        </span>
      </div>

      <div className="color-calibration-grid">
        <div className="color-calibration-video-column">
          <div className="color-calibration-stage">
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  aria-label={`Calibration source: ${videoName}`}
                  crossOrigin="anonymous"
                  muted
                  playsInline
                  preload="auto"
                  onLoadedMetadata={(event) => {
                    const video = event.currentTarget;
                    const nextDuration =
                      Number.isFinite(video.duration) && video.duration > 0
                        ? video.duration
                        : 0;
                    setVideoDimensions({
                      width: video.videoWidth,
                      height: video.videoHeight,
                    });
                    setDuration(nextDuration);
                    setCurrentTime(video.currentTime);
                    seekingRef.current = false;
                    setSeeking(false);
                    setPointer({
                      x: Math.floor(video.videoWidth / 2),
                      y: Math.floor(video.videoHeight / 2),
                    });
                    setVideoLoading(false);
                    setVideoError("");
                  }}
                  onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  onSeeking={() => {
                    seekingRef.current = true;
                    setSeeking(true);
                  }}
                  onSeeked={(event) => {
                    setCurrentTime(event.currentTarget.currentTime);
                    seekingRef.current = false;
                    setSeeking(false);
                  }}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onError={() => {
                    setVideoLoading(false);
                    seekingRef.current = false;
                    setSeeking(false);
                    setVideoError(
                      "Video access expired or this browser could not decode the source.",
                    );
                  }}
                />
                {videoDimensions.width > 0 && videoDimensions.height > 0 && (
                  <svg
                    className="color-calibration-hit-target"
                    viewBox={`0 0 ${videoDimensions.width} ${videoDimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="button"
                    tabIndex={0}
                    focusable="true"
                    aria-label={`Sample a ${COLOR_LABELS[activeColor].toLowerCase()} sticker. Move the crosshair with arrow keys and press Enter to sample.`}
                    aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown Enter Space"
                    aria-disabled={sampling || submitting || seeking}
                    onClick={handleVideoClick}
                    onPointerMove={handlePointerMove}
                    onKeyDown={handleVideoKey}
                  >
                    <title>
                      {`Sample a ${COLOR_LABELS[activeColor].toLowerCase()} sticker`}
                    </title>
                    {pointer && (
                      <g className="color-calibration-crosshair">
                        <circle
                          cx={pointer.x}
                          cy={pointer.y}
                          r={crosshairRadius}
                        />
                        <line
                          x1={pointer.x - crosshairRadius * 1.55}
                          y1={pointer.y}
                          x2={pointer.x + crosshairRadius * 1.55}
                          y2={pointer.y}
                        />
                        <line
                          x1={pointer.x}
                          y1={pointer.y - crosshairRadius * 1.55}
                          x2={pointer.x}
                          y2={pointer.y + crosshairRadius * 1.55}
                        />
                      </g>
                    )}
                  </svg>
                )}
                <span className="color-calibration-target-badge">
                  <i
                    className={`color-calibration-swatch color-calibration-swatch-${activeColor}`}
                    aria-hidden="true"
                  />
                  Click {COLOR_LABELS[activeColor].toLowerCase()}
                </span>
              </>
            ) : (
              <div className="color-calibration-video-placeholder">
                {videoLoading ? "Opening video…" : "Video unavailable"}
              </div>
            )}
          </div>

          <div className="color-calibration-transport">
            <button
              type="button"
              disabled={!ready || submitting || seeking}
              aria-label="Previous video frame"
              onClick={() => seek(currentTime - 1 / safeFps)}
            >
              −1
            </button>
            <button
              className="color-calibration-play"
              type="button"
              disabled={!ready || submitting || seeking}
              onClick={togglePlayback}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              disabled={!ready || submitting || seeking}
              aria-label="Next video frame"
              onClick={() => seek(currentTime + 1 / safeFps)}
            >
              +1
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(duration, 0.001)}
              step={0.001}
              value={Math.min(currentTime, Math.max(duration, 0.001))}
              disabled={!ready || submitting}
              aria-label="Video position"
              onChange={(event) => seek(Number(event.target.value))}
            />
            <output aria-label="Video time">
              {formatTime(currentTime)} / {formatTime(duration)}
            </output>
          </div>

          {videoError && (
            <div className="color-calibration-inline-error" role="alert">
              <span>{videoError}</span>
              <button
                className="text-button"
                type="button"
                disabled={submitting}
                onClick={() => void loadVideo()}
              >
                Refresh video access
              </button>
            </div>
          )}
        </div>

        <div className="color-calibration-targets">
          <div className="color-calibration-target-head">
            <strong>Target stickers</strong>
            <small>Select a sampled color to replace it.</small>
          </div>
          <div className="color-calibration-patch-grid">
            {CALIBRATION_COLORS.map((color) => {
              const patch = patches[color];
              const active = activeColor === color;
              return (
                <button
                  className={`color-calibration-patch${active ? " color-calibration-patch-active" : ""}${patch ? " color-calibration-patch-sampled" : ""}`}
                  type="button"
                  key={color}
                  disabled={submitting}
                  aria-pressed={active}
                  aria-label={`${COLOR_LABELS[color]}. ${patch ? "Sampled. Select to resample." : "Not sampled. Select this target."}`}
                  onClick={() => {
                    setActiveColor(color);
                    setError("");
                  }}
                >
                  <span className="color-calibration-patch-preview">
                    {patch ? (
                      <img src={patch.previewUrl} alt="" />
                    ) : (
                      <i
                        className={`color-calibration-swatch color-calibration-swatch-${color}`}
                        aria-hidden="true"
                      />
                    )}
                  </span>
                  <span>
                    <strong>{COLOR_LABELS[color]}</strong>
                    <small>
                      {patch
                        ? `${formatTime(patch.time)} · sample again`
                        : active
                          ? "Click video"
                          : "Select"}
                    </small>
                  </span>
                </button>
              );
            })}
          </div>

          <div className="color-calibration-actions">
            <button
              className="button button-primary"
              type="button"
              disabled={!allSampled || submitting}
              onClick={() => void submit()}
            >
              {submitting ? "Creating…" : "Create calibration"}
            </button>
            <button
              className="text-button"
              type="button"
              disabled={sampledCount === 0 || submitting}
              onClick={clearPatches}
            >
              Reset all
            </button>
          </div>

          {message && (
            <p className="color-calibration-message" role="status">
              {message}
            </p>
          )}
          {error && (
            <p
              className="color-calibration-message color-calibration-message-error"
              role="alert"
            >
              {error}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
