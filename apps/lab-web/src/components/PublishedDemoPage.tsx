import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createCaptureMediaTicket } from "../api";
import { cubeTrajectory } from "../lib/cubePerms";
import {
  ROUTE_GUIDE_LINK_REL,
  ROUTE_GUIDE_LINK_TARGET,
  routeGuideDocumentHref,
} from "../routeGuideLinks";
import {
  GTD1_OVERLAY_TRACK,
  GTD1_PUBLISHED_DEMO,
  GTD1_RELEASE_TAG,
  GTD1_VIDEO_TIMING,
  GTD1_VIDEO_URL,
  bleTimedMoves,
  frameIndexForTime,
  loadPublishedBleGroundTruth,
  loadPublishedDemo,
  loadPublishedOverlayTrack,
  moveStepForFrame,
  timeForFrameIndex,
  type LoadedPublishedDemo,
  type OverlayFrame,
  type PublishedBleGroundTruth,
  type PublishedOverlayTrack,
  type TimedMove,
} from "../publishedDemo";
import { CubeState } from "./CubeState";
import "../labTools.css";
import "./PublishedDemoPage.css";

type LoadState = "loading" | "loaded" | "error";

interface PublishedResource<T> {
  error: string;
  retry: () => void;
  state: LoadState;
  value: T | null;
}

interface PublishedResourceOutcome<T> {
  error: string;
  value: T | null;
}

interface PendingPublishedResource<T> {
  attempt: number;
  controller: AbortController;
  promise: Promise<PublishedResourceOutcome<T>>;
}

const RESOURCE_LOAD_TIMEOUT_MS = 15_000;
const DEMO_DATA_CARD =
  "https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/README.md";
const DEMO_EVIDENCE = "docs/EVIDENCE.md#the-published-demo-replay";

function usePublishedResource<T>(
  label: string,
  loader: (signal: AbortSignal) => Promise<T>,
): PublishedResource<T> {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<LoadState>("loading");
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState("");
  const pendingRef = useRef<PendingPublishedResource<T> | null>(null);
  const cleanupTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (cleanupTimerRef.current !== null) {
      window.clearTimeout(cleanupTimerRef.current);
      cleanupTimerRef.current = null;
    }

    let pending = pendingRef.current;
    if (!pending || pending.attempt !== attempt) {
      pending?.controller.abort();
      const controller = new AbortController();
      let timedOut = false;
      const timeout = window.setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, RESOURCE_LOAD_TIMEOUT_MS);
      const promise: Promise<PublishedResourceOutcome<T>> = loader(
        controller.signal,
      )
        .then((loaded) => ({ error: "", value: loaded }))
        .catch((reason: unknown) => ({
          error: timedOut
            ? `${label} did not load within 15 seconds.`
            : reason instanceof Error
              ? reason.message
              : `${label} could not be opened.`,
          value: null,
        }))
        .finally(() => window.clearTimeout(timeout));
      pending = { attempt, controller, promise };
      pendingRef.current = pending;
    }

    let active = true;
    setState("loading");
    setValue(null);
    setError("");
    void pending.promise.then((outcome) => {
      if (!active) return;
      if (outcome.error) {
        setState("error");
        setError(outcome.error);
        return;
      }
      setValue(outcome.value);
      setState("loaded");
    });

    return () => {
      active = false;
      // React Strict Mode immediately remounts effects in development. Delay
      // cancellation by one task so that remount can reuse the same checked
      // request, while a real unmount still aborts promptly.
      cleanupTimerRef.current = window.setTimeout(() => {
        if (pendingRef.current !== pending) return;
        pending.controller.abort();
        pendingRef.current = null;
      }, 0);
    };
  }, [attempt, label, loader]);

  const retry = useCallback(() => setAttempt((current) => current + 1), []);
  return { error, retry, state, value };
}

const INT_COLORS = ["white", "yellow", "red", "orange", "blue", "green"];
const FACE_SLICES: [string, number][] = [
  ["up", 0],
  ["right", 9],
  ["front", 18],
  ["down", 27],
  ["left", 36],
  ["back", 45],
];
const MOVE_FACES: Record<string, string> = {
  U: "up",
  R: "right",
  F: "front",
  D: "down",
  L: "left",
  B: "back",
};

// The React tree redraws far slower than the overlay canvas. 20 updates a
// second keeps the frame readout, timeline, and cube honest without
// re-rendering 77 move buttons on every animation frame.
const UI_INTERVAL_MS = 50;

const MOVE_TIMING_NOTE = "Move timing comes from the smart-cube record.";

// The transport works in recorded frame indices, so a step needs no rate.
const FRAME_STEP = 1;
const COARSE_FRAME_STEP = 10;

function toNet(state: Int8Array): Record<string, string[]> {
  const net: Record<string, string[]> = {};
  for (const [face, offset] of FACE_SLICES) {
    net[face] = Array.from(
      state.slice(offset, offset + 9),
      (value) => INT_COLORS[value] ?? "white",
    );
  }
  return net;
}

function faceForMove(move: string | null): string | undefined {
  if (!move) return undefined;
  return MOVE_FACES[move[0]?.toUpperCase() ?? ""];
}

function bilinear(
  quad: readonly (readonly [number, number])[],
  u: number,
  v: number,
): [number, number] {
  const [a, b, c, d] = quad;
  return [
    (1 - u) * (1 - v) * a[0] + u * (1 - v) * b[0] + u * v * c[0] + (1 - u) * v * d[0],
    (1 - u) * (1 - v) * a[1] + u * (1 - v) * b[1] + u * v * c[1] + (1 - u) * v * d[1],
  ];
}

// The slice of the source frame the overlay canvas shows, in the normalized
// coordinates the track records. The full frame is the identity window.
interface OverlayWindow {
  x: number;
  y: number;
  size: number;
}

const FULL_FRAME: OverlayWindow = { x: 0, y: 0, size: 1 };

// The clip is a release asset, so on a clone without it the frame is an empty
// box with a small quad floating in the middle. Zooming to the region the
// tracker actually used makes that state readable. The window is computed once
// from the whole track, so it never jitters from frame to frame, and it stays
// square, so the quads keep the shape they were recorded with.
function trackWindow(track: PublishedOverlayTrack | null): OverlayWindow {
  if (!track) return FULL_FRAME;
  let left = 1;
  let top = 1;
  let right = 0;
  let bottom = 0;
  for (const entry of Object.values(track.frames)) {
    for (const face of entry.f) {
      for (const [x, y] of face.q) {
        if (x < left) left = x;
        if (y < top) top = y;
        if (x > right) right = x;
        if (y > bottom) bottom = y;
      }
    }
  }
  if (!(right > left && bottom > top)) return FULL_FRAME;
  const pad = 0.04;
  const size = Math.min(
    1,
    Math.max(right - left, bottom - top) + pad * 2,
  );
  const clamp = (center: number) =>
    Math.min(Math.max(0, center - size / 2), 1 - size);
  return {
    x: clamp((left + right) / 2),
    y: clamp((top + bottom) / 2),
    size,
  };
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  entry: OverlayFrame | undefined,
  outline: string,
  view: OverlayWindow,
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio)) {
    canvas.width = Math.round(width * ratio);
  }
  if (canvas.height !== Math.round(height * ratio)) {
    canvas.height = Math.round(height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!entry) return;

  for (const face of entry.f) {
    if (face.q.length !== 4) continue;
    const quad = face.q.map(
      ([x, y]) =>
        [
          ((x - view.x) / view.size) * width,
          ((y - view.y) / view.size) * height,
        ] as [number, number],
    );

    if (face.k && face.k.length === 54) {
      for (let index = 0; index < 9; index += 1) {
        const row = Math.floor(index / 3);
        const column = index % 3;
        const cell = [
          bilinear(quad, (column + 0.12) / 3, (row + 0.12) / 3),
          bilinear(quad, (column + 0.88) / 3, (row + 0.12) / 3),
          bilinear(quad, (column + 0.88) / 3, (row + 0.88) / 3),
          bilinear(quad, (column + 0.12) / 3, (row + 0.88) / 3),
        ];
        const confidence = face.v?.[index] ?? 0;
        context.globalAlpha = 0.2 + 0.7 * Math.max(0, Math.min(1, confidence));
        context.fillStyle = `#${face.k.slice(index * 6, index * 6 + 6)}`;
        context.beginPath();
        context.moveTo(cell[0][0], cell[0][1]);
        for (const [x, y] of cell.slice(1)) context.lineTo(x, y);
        context.closePath();
        context.fill();
      }
    }

    context.globalAlpha = 0.35 + 0.6 * Math.max(0, Math.min(1, face.c));
    context.strokeStyle = outline;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(quad[0][0], quad[0][1]);
    for (const [x, y] of quad.slice(1)) context.lineTo(x, y);
    context.closePath();
    context.stroke();
  }
  context.globalAlpha = 1;
}

const MOVES_LANE_HEIGHT = 18;
const ALIGNMENT_LANE_HEIGHT = 34;

// The playhead sits in the track's coordinate space, which includes the label
// column, so a percentage of the clip has to be mapped past that column.
function lanePosition(fraction: number): string {
  const bounded = Math.min(1, Math.max(0, fraction));
  return `calc(var(--lane-label-width) + (100% - var(--lane-label-width)) * ${bounded})`;
}

/**
 * The decoded moves as a horizontal timeline over the clip. Each lane draws
 * its series once to a canvas, and the moving cursor is a positioned div. A
 * 120 fps clip would otherwise redraw the canvases on every frame of playback.
 *
 * Two labelled lanes rather than two stacked layers. Unlabelled marks read as
 * decoration, and a viewer cannot tell a recorded turn from a tracker signal.
 */
function MoveTimeline({
  frame,
  seekToFrame,
  step,
  timedMoves,
  track,
}: {
  frame: number;
  seekToFrame: (next: number) => void;
  step: number;
  timedMoves: readonly TimedMove[];
  track: PublishedOverlayTrack | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const movesCanvasRef = useRef<HTMLCanvasElement>(null);
  const alignmentCanvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(600);
  const last = GTD1_VIDEO_TIMING.lastFrame;

  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() =>
      setWidth(element.clientWidth || 600),
    );
    observer.observe(element);
    setWidth(element.clientWidth || 600);
    return () => observer.disconnect();
  }, []);

  const prepare = (canvas: HTMLCanvasElement | null, height: number) => {
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return null;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return context;
  };

  // One tick per decoded move.
  useEffect(() => {
    const canvas = movesCanvasRef.current;
    const context = prepare(canvas, MOVES_LANE_HEIGHT);
    if (!canvas || !context) return;
    context.fillStyle = getComputedStyle(canvas)
      .getPropertyValue("--timeline-anchor")
      .trim();
    for (const move of timedMoves) {
      const x = Math.round((move.frame / last) * (width - 1));
      context.fillRect(x, 3, 1, MOVES_LANE_HEIGHT - 6);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [last, timedMoves, width]);

  // Tracker alignment per pixel column, downsampled by column maximum. Columns
  // with no entry are frames the tracker found no cube on, so they stay empty.
  useEffect(() => {
    const canvas = alignmentCanvasRef.current;
    const context = prepare(canvas, ALIGNMENT_LANE_HEIGHT);
    if (!canvas || !context || !track) return;
    const columns = new Float32Array(Math.max(1, Math.round(width))).fill(0);
    for (const [key, entry] of Object.entries(track.frames)) {
      const column = Math.round((Number(key) / last) * (width - 1));
      if (column < 0 || column >= columns.length) continue;
      columns[column] = Math.max(columns[column], entry.a);
    }
    const styles = getComputedStyle(canvas);
    context.fillStyle = styles.getPropertyValue("--timeline-series").trim();
    for (let column = 0; column < columns.length; column += 1) {
      const value = Math.max(0, Math.min(1, columns[column]));
      if (value === 0) continue;
      const height = 2 + value * (ALIGNMENT_LANE_HEIGHT - 2);
      context.fillRect(column, ALIGNMENT_LANE_HEIGHT - height, 1, height);
    }
    // The minimal scale: one rule at P(aligned) 0.5, no axis.
    context.fillStyle = styles.getPropertyValue("--timeline-rule").trim();
    context.fillRect(0, Math.round(ALIGNMENT_LANE_HEIGHT / 2), width, 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [last, track, width]);

  const fraction = frame / last;
  const currentAlignment = track?.frames[String(frame)]?.a;
  const seekFromClientX = (clientX: number) => {
    const element = wrapRef.current;
    if (!element) return;
    const box = element.getBoundingClientRect();
    if (box.width === 0) return;
    const hit = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
    seekToFrame(Math.round(hit * last));
  };
  const onLanePointerDown = (event: { clientX: number }) =>
    seekFromClientX(event.clientX);

  return (
    <div className="demo-timeline">
      <p className="demo-timeline-readout">
        <span>move</span>
        <strong>
          {step} of {timedMoves.length}
        </strong>
        <code>{step > 0 ? timedMoves[step - 1].move : "scramble"}</code>
      </p>
      <div
        className="demo-timeline-track"
        role="img"
        aria-label={`Move and alignment timeline. Frame ${frame} of ${last}. Move ${step} of ${
          timedMoves.length
        }. Alignment probability ${
          currentAlignment == null ? "not sampled" : currentAlignment.toFixed(2)
        }.`}
      >
        <span className="demo-timeline-lane-label">moves</span>
        <div
          ref={wrapRef}
          className="demo-timeline-lane"
          style={{ height: MOVES_LANE_HEIGHT }}
          onPointerDown={onLanePointerDown}
        >
          <canvas
            ref={movesCanvasRef}
            className="demo-timeline-canvas"
            aria-hidden="true"
            style={{ height: MOVES_LANE_HEIGHT, width: "100%" }}
          />
        </div>

        <span className="demo-timeline-lane-label">
          alignment
          <small>0 to 1</small>
        </span>
        <div
          className="demo-timeline-lane"
          style={{ height: ALIGNMENT_LANE_HEIGHT }}
          onPointerDown={onLanePointerDown}
        >
          <canvas
            ref={alignmentCanvasRef}
            className="demo-timeline-canvas"
            aria-hidden="true"
            style={{ height: ALIGNMENT_LANE_HEIGHT, width: "100%" }}
          />
        </div>

        {step > 0 && (
          <div
            className="demo-timeline-anchor-current"
            style={{ left: lanePosition(timedMoves[step - 1].frame / last) }}
          />
        )}
        {/* The one high-contrast element, spanning both lanes to tie them. */}
        <div
          className="demo-timeline-cursor"
          style={{ left: lanePosition(fraction) }}
        />
      </div>
    </div>
  );
}

// The alignment confidence gate recorded by the decode artifact.
const ALIGN_THRESH = 0.5;

function ResourceNotice({
  action,
  error = false,
  message,
  onRetry,
  title,
}: {
  action?: { href: string; label: string };
  error?: boolean;
  message: string;
  onRetry?: () => void;
  title: string;
}) {
  return (
    <div
      className={`demo-resource-state${error ? " demo-resource-state-error" : ""}`}
      role={error ? "alert" : "status"}
    >
      <span aria-hidden="true">{error ? "!" : "…"}</span>
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      {(onRetry || action) && (
        <div className="demo-resource-actions">
          {onRetry && (
            <button
              aria-label={`Retry ${title.toLowerCase()}`}
              className="text-button"
              type="button"
              onClick={onRetry}
            >
              Retry
            </button>
          )}
          {action && (
            <a
              href={action.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {action.label}
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// The tracker only ever assigns these three slots on this clip, and a frame
// carries one, two, or three of them. Rendering the fixed set keeps a slot in
// the same place all the way through the solve, so a viewer can watch one face
// rather than re-reading the labels every time a detection drops.
const READ_SLOTS = ["up", "front", "right"] as const;

/**
 * The tracker's sampled color reads for the current frame, one 3x3 per slot.
 *
 * The cell background is the sampled sRGB the artifact carries, rendered as
 * recorded. The artifact envelope declares
 * `color_meaning: "sampled-color-not-a-classified-sticker"`, so nothing here
 * classifies a value, snaps it to a palette, or calls it a sticker. That is an
 * evidence boundary, not a styling choice.
 *
 * Every element has a fixed size and is always present. At 120 fps a panel
 * that grew or shrank as detections came and went would jitter the whole
 * column, so an undetected slot draws an empty grid.
 */
function SampledReads({ entry }: { entry: OverlayFrame | undefined }) {
  const misaligned = entry != null && entry.a < ALIGN_THRESH;
  const sampled = new Map(
    (misaligned ? [] : (entry?.f ?? [])).flatMap((face) =>
      face.k?.length === 54 && face.s ? [[face.s, face] as const] : [],
    ),
  );

  return (
    <div className="demo-reads">
      <h3>Sampled color reads</h3>
      <div className="demo-read-faces">
        {READ_SLOTS.map((slot) => {
          const face = sampled.get(slot);
          return (
            <div className="demo-read-face" key={slot}>
              <div className="demo-read-slot">{slot}</div>
              <div className="demo-read-grid">
                {Array.from({ length: 9 }, (_unused, cell) => {
                  if (!face) {
                    return (
                      <div
                        aria-label={`${slot} cell ${cell + 1}: not sampled`}
                        className="demo-read-cell demo-read-empty"
                        key={cell}
                        role="img"
                      />
                    );
                  }
                  const hex = `#${face.k!.slice(cell * 6, cell * 6 + 6)}`;
                  const confidence = face.v?.[cell];
                  return (
                    <div
                      aria-label={`${slot} cell ${cell + 1}: sampled ${hex}, confidence ${
                        confidence == null ? "not recorded" : confidence.toFixed(2)
                      }`}
                      className="demo-read-cell"
                      key={cell}
                      role="img"
                      title={`sampled ${hex} · confidence ${
                        confidence == null ? "not recorded" : confidence.toFixed(2)
                      }`}
                      style={{
                        background: hex,
                        opacity:
                          confidence == null
                            ? 0.35
                            : 0.45 + 0.55 * Math.min(1, confidence),
                      }}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function PublishedDemoPage() {
  const coreResource = usePublishedResource<LoadedPublishedDemo>(
    "The reconstruction result and receipt",
    loadPublishedDemo,
  );
  const trackResource = usePublishedResource<PublishedOverlayTrack>(
    "The camera overlay",
    loadPublishedOverlayTrack,
  );
  const bleResource = usePublishedResource<PublishedBleGroundTruth>(
    "The smart-cube timing record",
    loadPublishedBleGroundTruth,
  );
  const demo = coreResource.value;
  const track = trackResource.value;
  const ble = bleResource.value;
  const [videoUrl, setVideoUrl] = useState(GTD1_VIDEO_URL);
  const [videoAttempt, setVideoAttempt] = useState(0);
  const [frame, setFrame] = useState(0);
  const [manualStep, setManualStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [videoFailed, setVideoFailed] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackRef = useRef<PublishedOverlayTrack | null>(null);
  const frameRef = useRef(0);
  const clockRef = useRef(0);
  const playingRef = useRef(false);
  const videoReadyRef = useRef(false);

  playingRef.current = playing;
  videoReadyRef.current = videoReady && !videoFailed;
  trackRef.current = track;

  // With the clip present the overlay has to sit on the video pixel for pixel,
  // so the window stays the whole frame. Without it there is nothing to align
  // to and a zoom is the only way to read the tracker output.
  const trackView = useMemo(() => trackWindow(track), [track]);
  const viewRef = useRef<OverlayWindow>(FULL_FRAME);
  viewRef.current = videoFailed ? trackView : FULL_FRAME;

  // Source ladder for the clip. A workbench that already holds the capture can
  // serve the exact bytes over a media ticket, which is the only path that
  // works while the repository is private. The release asset is the fallback
  // for a public clone, and a failure of both leaves the offline state.
  //
  // This is strictly optional. It never blocks the overlay, timeline, cube, or
  // transport, and a rejected request is treated as "no local copy".
  useEffect(() => {
    let cancelled = false;
    void createCaptureMediaTicket(GTD1_OVERLAY_TRACK.captureId)
      .then((ticket) => {
        if (cancelled) return;
        // The release URL may already have failed. Give the local bytes their
        // own attempt rather than staying in the offline state.
        setVideoFailed(false);
        setVideoReady(false);
        setVideoUrl(ticket.url);
      })
      .catch(() => {
        // No local capture and no reachable service. The release asset stays.
      });
    return () => {
      cancelled = true;
    };
  }, [videoAttempt]);

  useEffect(() => {
    if (
      coreResource.state !== "loaded" ||
      videoReady ||
      videoFailed
    ) {
      return;
    }
    const timeout = window.setTimeout(() => {
      setVideoReady(false);
      setVideoFailed(true);
    }, RESOURCE_LOAD_TIMEOUT_MS);
    return () => window.clearTimeout(timeout);
  }, [
    coreResource.state,
    videoAttempt,
    videoFailed,
    videoReady,
    videoUrl,
  ]);

  const retryVideo = useCallback(() => {
    videoRef.current?.pause();
    setPlaying(false);
    setVideoReady(false);
    setVideoFailed(false);
    setVideoUrl(GTD1_VIDEO_URL);
    setVideoAttempt((current) => current + 1);
  }, []);

  const seek = useCallback((next: number) => {
    const bounded = Math.max(0, Math.min(GTD1_VIDEO_TIMING.lastFrame, next));
    frameRef.current = bounded;
    clockRef.current = timeForFrameIndex(bounded);
    setFrame(bounded);
    const video = videoRef.current;
    if (video && videoReadyRef.current) video.currentTime = clockRef.current;
    const canvas = canvasRef.current;
    if (canvas) {
      drawOverlay(
        canvas,
        trackRef.current?.frames[String(bounded)],
        getComputedStyle(canvas).getPropertyValue("--overlay-outline").trim(),
        viewRef.current,
      );
    }
  }, []);

  useEffect(() => {
    if (coreResource.state !== "loaded") return;
    const duration = timeForFrameIndex(GTD1_VIDEO_TIMING.lastFrame);
    let request = 0;
    let previous = performance.now();
    let pushed = 0;

    const tick = (now: number) => {
      const delta = (now - previous) / 1000;
      previous = now;
      const video = videoRef.current;
      if (video && videoReadyRef.current) {
        frameRef.current = frameIndexForTime(video.currentTime);
        clockRef.current = video.currentTime;
      } else if (playingRef.current) {
        clockRef.current = Math.min(duration, clockRef.current + delta);
        frameRef.current = frameIndexForTime(clockRef.current);
        if (clockRef.current >= duration) setPlaying(false);
      }
      const canvas = canvasRef.current;
      if (canvas) {
        drawOverlay(
          canvas,
          trackRef.current?.frames[String(frameRef.current)],
          getComputedStyle(canvas).getPropertyValue("--overlay-outline").trim(),
          viewRef.current,
        );
      }
      if (now - pushed >= UI_INTERVAL_MS) {
        pushed = now;
        setFrame(frameRef.current);
      }
      request = window.requestAnimationFrame(tick);
    };

    request = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(request);
  }, [coreResource.state]);

  const togglePlaying = useCallback(() => {
    const video = videoRef.current;
    if (video && videoReadyRef.current) {
      if (video.paused) void video.play().catch(() => setVideoFailed(true));
      else video.pause();
      return;
    }
    if (!playing && clockRef.current >= timeForFrameIndex(GTD1_VIDEO_TIMING.lastFrame)) {
      seek(0);
    }
    setPlaying((current) => !current);
  }, [playing, seek]);

  const moves = useMemo(() => demo?.result.moves ?? [], [demo]);
  const scrambleTokens = useMemo(
    () => GTD1_PUBLISHED_DEMO.scramble.trim().split(/\s+/).filter(Boolean),
    [],
  );
  const states = useMemo(() => {
    if (moves.length === 0) return [];
    return cubeTrajectory([...scrambleTokens, ...moves]).slice(
      scrambleTokens.length,
    );
  }, [moves, scrambleTokens]);

  // Each decoded move is pinned to the clip frame the smart cube recorded its
  // turn on. A semantic mismatch is an integrity failure for timing only: the
  // preserved result remains available as a move-by-move replay.
  const timedMoveResult = useMemo(
    () => {
      if (!ble || moves.length === 0) return { error: "", moves: [] };
      try {
        return { error: "", moves: bleTimedMoves(ble, moves) };
      } catch (reason) {
        return {
          error:
            reason instanceof Error
              ? reason.message
              : "The smart-cube timing record could not be applied.",
          moves: [],
        };
      }
    },
    [ble, moves],
  );
  const timedMoves = timedMoveResult.moves;
  const bleState: LoadState = timedMoveResult.error
    ? "error"
    : bleResource.state;
  const bleError = timedMoveResult.error || bleResource.error;
  const hasMoveTiming = bleState === "loaded" && timedMoves.length > 0;
  const step = hasMoveTiming
    ? moveStepForFrame(timedMoves, frame)
    : Math.max(0, Math.min(moves.length, manualStep));
  const currentMove = step > 0 ? moves[step - 1] : null;
  const overlayFrame = track?.frames[String(frame)];

  const selectStep = useCallback(
    (next: number) => {
      const bounded = Math.max(0, Math.min(moves.length, next));
      if (hasMoveTiming) {
        seek(bounded === 0 ? 0 : timedMoves[bounded - 1].frame);
        return;
      }
      setManualStep(bounded);
    },
    [hasMoveTiming, moves.length, seek, timedMoves],
  );

  // The custom transport is the only one on the page, so it also carries the
  // keyboard. The listener sits on the window because the clip is often
  // unavailable, and then there is no video element to focus.
  useEffect(() => {
    if (coreResource.state !== "loaded") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        target?.isContentEditable
      ) {
        return;
      }
      const current = frameRef.current;
      switch (event.key) {
        case "ArrowLeft":
          seek(current - (event.shiftKey ? COARSE_FRAME_STEP : FRAME_STEP));
          break;
        case "ArrowRight":
          seek(current + (event.shiftKey ? COARSE_FRAME_STEP : FRAME_STEP));
          break;
        // Up moves forward and down moves back, matching the transport row.
        case "ArrowUp":
          selectStep(
            hasMoveTiming ? moveStepForFrame(timedMoves, current) + 1 : step + 1,
          );
          break;
        case "ArrowDown":
          // The number of onsets strictly before this frame is the index of
          // the one to land on, so a frame inside a move rewinds to its start.
          selectStep(
            hasMoveTiming
              ? moveStepForFrame(timedMoves, current - 1)
              : step - 1,
          );
          break;
        case "Home":
          seek(0);
          if (!hasMoveTiming) setManualStep(0);
          break;
        case "End":
          seek(GTD1_VIDEO_TIMING.lastFrame);
          if (!hasMoveTiming) setManualStep(moves.length);
          break;
        case " ":
          // A focused button already treats Space as activation.
          if (tag === "BUTTON" || tag === "A") return;
          togglePlaying();
          break;
        default:
          return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    coreResource.state,
    hasMoveTiming,
    moves.length,
    seek,
    selectStep,
    step,
    timedMoves,
    togglePlaying,
  ]);

  if (coreResource.state === "error") {
    return (
      <div className="page demo-page">
        <DemoHeader />
        <ResourceNotice
          action={{
            href: DEMO_DATA_CARD,
            label: "Open the demo data card",
          }}
          error
          message={coreResource.error}
          onRetry={coreResource.retry}
          title="Reconstruction files unavailable"
        />
      </div>
    );
  }

  if (coreResource.state === "loading" || !demo) {
    return (
      <div className="page demo-page">
        <DemoHeader />
        <p className="demo-loading" role="status">
          Checking the reconstruction result and receipt in this browser…
        </p>
      </div>
    );
  }

  return (
    <div className="page demo-page">
      <DemoHeader />

      {/*
        Video and decoded output sit side by side on a wide viewport so the
        move timeline advances while the clip plays. Stacked below 1100px, a
        compact strip under the video carries the current move instead.
      */}
      <div className="demo-live">
        <section className="demo-stage" aria-labelledby="demo-stage-title">
          <div className="demo-stage-head">
            <div>
              <p className="station-code">Source clip and camera overlay</p>
              <h2 id="demo-stage-title">Watch the reconstruction</h2>
            </div>
          </div>

          {trackResource.state === "loading" && (
            <ResourceNotice
              message="The decoded result is ready while the camera overlay is checked."
              title="Loading camera overlay"
            />
          )}
          {trackResource.state === "error" && (
            <ResourceNotice
              error
              message={`${trackResource.error} The decoded cube replay and source clip remain available.`}
              onRetry={trackResource.retry}
              title="Camera overlay unavailable"
            />
          )}

          <div className="demo-viewer">
            <div className="demo-frame">
              {!videoFailed && (
                <video
                  ref={videoRef}
                  className="demo-video"
                  key={`${videoUrl}-${videoAttempt}`}
                  src={videoUrl}
                  muted
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={() => setVideoReady(true)}
                  onError={() => {
                    setVideoReady(false);
                    setVideoFailed(true);
                  }}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                />
              )}
              <canvas ref={canvasRef} className="demo-overlay" aria-hidden="true" />
              {videoFailed && (
                <p className="demo-frame-placeholder">Clip not loaded</p>
              )}
            </div>

            <div className="demo-transport">
              <div
                className="demo-transport-row"
                role="group"
                aria-label="Clip transport"
                aria-keyshortcuts="ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight ArrowUp ArrowDown Home End Space"
              >
                <button
                  className="button button-primary"
                  type="button"
                  aria-pressed={playing}
                  onClick={togglePlaying}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  className="button button-quiet"
                  type="button"
                  onClick={() => seek(0)}
                  disabled={frame === 0}
                >
                  Start
                </button>
                <button
                  className="button button-quiet"
                  type="button"
                  aria-label="Back one frame"
                  onClick={() => seek(frame - FRAME_STEP)}
                  disabled={frame === 0}
                >
                  Back
                </button>
                <button
                  className="button button-quiet"
                  type="button"
                  aria-label="Forward one frame"
                  onClick={() => seek(frame + FRAME_STEP)}
                  disabled={frame >= GTD1_VIDEO_TIMING.lastFrame}
                >
                  Forward
                </button>
                <p className="demo-frame-readout">
                  frame <strong>{frame}</strong> of {GTD1_VIDEO_TIMING.lastFrame}
                </p>
              </div>
              <input
                className="frame-scrubber"
                type="range"
                min={0}
                max={GTD1_VIDEO_TIMING.lastFrame}
                step={FRAME_STEP}
                value={frame}
                aria-label="Clip frame"
                onChange={(event) => seek(Number(event.target.value))}
              />
              {/* Discoverability without a help panel. The full set is on the
                  transport's aria-keyshortcuts. */}
              <div className="demo-context-line">
                <p className="demo-key-hint">
                  <kbd>←</kbd>
                  <kbd>→</kbd> frame · <kbd>↑</kbd>
                  <kbd>↓</kbd> move · <kbd>space</kbd> play
                </p>
              </div>
              {videoFailed && (
                <p className="demo-video-note" role="status">
                  The demo clip did not load. The replay controls and preserved
                  result remain available.{" "}
                  <button className="text-button" type="button" onClick={retryVideo}>
                    Retry clip
                  </button>{" "}
                  or{" "}
                  <a
                    href={GTD1_VIDEO_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    open the exact release clip
                  </a>
                  . To download a checksum-verified copy for import from a clone,
                  run{" "}
                  <code>
                    uv run --no-sync python scripts/download_release_assets.py
                    --include demo --tag {GTD1_RELEASE_TAG}
                  </code>
                  .
                </p>
              )}
              <p className="demo-move-strip" role="status">
                <span className="demo-move-strip-label">Decoded move</span>
                <span className="demo-move-strip-count">
                  {step} of {moves.length}
                </span>
                <span className="demo-move-strip-tokens" aria-hidden="true">
                  {[step - 1, step, step + 1].map((slot) => (
                    <span
                      className={
                        slot === step
                          ? "demo-move-strip-token demo-move-strip-token-current"
                          : "demo-move-strip-token"
                      }
                      key={slot}
                    >
                      {slot > 0 && slot <= moves.length ? moves[slot - 1] : "·"}
                    </span>
                  ))}
                </span>
              </p>
            </div>
          </div>
        </section>

        <section className="demo-moves" aria-labelledby="demo-moves-title">
          <div className="demo-stage-head">
            <p className="station-code" id="demo-moves-title">
              Decoded output
            </p>
          </div>

          <section
            className="demo-scramble"
            aria-labelledby="demo-scramble-title"
          >
            <div className="demo-scramble-head">
              <h3 id="demo-scramble-title">Starting scramble</h3>
              <span>{scrambleTokens.length} moves</span>
            </div>
            <div
              className="demo-scramble-tape"
              aria-label={`Starting scramble: ${GTD1_PUBLISHED_DEMO.scramble}`}
            >
              {scrambleTokens.map((token, index) => (
                <span
                  className={`demo-scramble-move demo-scramble-face-${token[0].toLowerCase()}`}
                  key={`${token}-${index}`}
                >
                  {token}
                </span>
              ))}
            </div>
          </section>

          <SampledReads entry={overlayFrame} />

          {hasMoveTiming ? (
            <MoveTimeline
              frame={frame}
              seekToFrame={seek}
              step={step}
              timedMoves={timedMoves}
              track={track}
            />
          ) : bleState === "loading" ? (
            <ResourceNotice
              message="The preserved moves can be stepped while frame timing is checked."
              title="Loading move timing"
            />
          ) : (
            <ResourceNotice
              error
              message={`${bleError} The controls below still step through the preserved decoded sequence without assigning clip frames.`}
              onRetry={bleResource.retry}
              title="Move timing unavailable"
            />
          )}

          <div
            className="demo-step-controls"
            role="group"
            aria-label="Decoded move controls"
          >
            <button
              className="button button-quiet"
              type="button"
              disabled={step === 0}
              onClick={() => selectStep(step - 1)}
            >
              Previous move
            </button>
            <p>
              move <strong>{step}</strong> of {moves.length}
            </p>
            <button
              className="button button-quiet"
              type="button"
              disabled={step >= moves.length}
              onClick={() => selectStep(step + 1)}
            >
              Next move
            </button>
          </div>

          <p className="demo-moves-note">
            {hasMoveTiming
              ? MOVE_TIMING_NOTE
              : "Move controls replay the preserved result without claiming clip timing."}
          </p>

          <div className="demo-moves-body">
            <div className="demo-cube">
              <CubeState
                state={toNet(states[step])}
                faceHighlight={faceForMove(currentMove)}
              />
              <p className="move-replay-caption">
                {step === 0
                  ? "Step 0: the sealed scramble state"
                  : `Step ${step} of ${moves.length}: after ${currentMove}`}
              </p>
            </div>
          </div>
        </section>
      </div>

      <div className="demo-boundary-row">
        <p className="demo-boundary-line">
          Preserved browser replay. It runs no decoder and reports no accuracy.
        </p>

        <nav className="demo-reference-links" aria-label="Demo documentation">
          <a
            href={DEMO_DATA_CARD}
            target={ROUTE_GUIDE_LINK_TARGET}
            rel={ROUTE_GUIDE_LINK_REL}
          >
            Dataset card
          </a>
          <a
            href={routeGuideDocumentHref(DEMO_EVIDENCE)}
            target={ROUTE_GUIDE_LINK_TARGET}
            rel={ROUTE_GUIDE_LINK_REL}
          >
            Evidence boundary
          </a>
        </nav>
      </div>

    </div>
  );
}

function DemoHeader() {
  return (
    <header className="page-header demo-header">
      <div>
        <h1>
          Reconstruction demo
          <span className="sr-only">, gtD1 demo artifact</span>
        </h1>
        <p className="page-description">
          Scrub a recorded solve with synchronized face reads, move timing,
          and cube reconstruction.
        </p>
        <p className="demo-artifact-id">
          Artifact <code>gtD1</code>
        </p>
      </div>
    </header>
  );
}
