import {
  type ChangeEvent,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { createCaptureMediaTicket } from "../api";
import { safeDownloadStem, triggerBlobDownload } from "../browserDownload";
import { bytesToHex } from "../hex";
import {
  canonicalDecodeSequence,
  decodeResultVideoSha256,
  decodeWorkstationAsTrackerDump,
  parseDecodeResultDocument,
} from "../decodeRunContracts";
import { fetchDecodeJobResultBlob } from "../decodeApi";
import { cubeTrajectory } from "../lib/cubePerms";
import {
  LocalContractError,
  type TrackerDump,
  type TrackerFace,
  type TrackerFaceRead,
  type TrackerFrame,
  readJsonFile,
} from "../localContracts";
import type {
  CaptureReceipt,
  DecodeGroundTruthDiagnostic,
  DecodeJobStatus,
  DecodeResultDocument,
  DecodeWorkstationSequence,
} from "../types";
import "../labTools.css";
import { CubeState } from "./CubeState";
import { TrellisBeam } from "./TrellisBeam";
import "./DecodeDiagnostics.css";

function frameEntries(dump: TrackerDump): [number, TrackerFrame][] {
  return Object.entries(dump.frames)
    .map(([frame, record]) => [Number(frame), record] as [number, TrackerFrame])
    .sort((left, right) => left[0] - right[0]);
}

function seriesPoints(
  entries: [number, TrackerFrame][],
  window: [number, number],
  field: "motion" | "aligned",
  maxValue: number,
): string {
  const width = 1_000;
  const height = 40;
  const span = Math.max(1, window[1] - window[0]);
  const stride = Math.max(1, Math.ceil(entries.length / 1_500));
  const points: string[] = [];
  for (let index = 0; index < entries.length; index += stride) {
    const [frame, record] = entries[index];
    const raw = field === "motion" ? record.motion : record.aligned;
    if (raw === null || raw === undefined) continue;
    const x = ((frame - window[0]) / span) * width;
    const y =
      height -
      5 -
      (Math.max(0, Math.min(maxValue, raw)) / maxValue) * (height - 10);
    points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return points.join(" ");
}

function RunSignalStrip({
  dump,
  currentFrame,
  onSeek,
}: {
  dump: TrackerDump;
  currentFrame: number;
  onSeek: (frame: number) => void;
}) {
  const entries = useMemo(() => frameEntries(dump), [dump]);
  const motionMax = useMemo(
    () =>
      Math.max(
        1,
        ...entries.map(([, record]) =>
          typeof record.motion === "number" ? record.motion : 0,
        ),
      ),
    [entries],
  );
  const motion = useMemo(
    () => seriesPoints(entries, dump.window, "motion", motionMax),
    [dump.window, entries, motionMax],
  );
  const aligned = useMemo(
    () => seriesPoints(entries, dump.window, "aligned", 1),
    [dump.window, entries],
  );
  const span = Math.max(1, dump.window[1] - dump.window[0]);
  const framePercent = Math.max(
    0,
    Math.min(100, ((currentFrame - dump.window[0]) / span) * 100),
  );

  const seekFromPointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width === 0) return;
    const fraction = Math.max(
      0,
      Math.min(1, (event.clientX - bounds.left) / bounds.width),
    );
    onSeek(Math.round(dump.window[0] + fraction * span));
  };

  return (
    <div className="tracker-strip">
      <div
        className="tracker-strip-chart"
        role="img"
        aria-label={`Decode signals from frame ${dump.window[0]} to ${dump.window[1]}`}
      >
        <div className="tracker-strip-labels" aria-hidden="true">
          <span>Motion</span>
          <span>Alignment</span>
        </div>
        <div className="tracker-strip-plots" onPointerDown={seekFromPointer}>
          <svg viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true">
            {motion && <polyline className="strip-motion" points={motion} />}
          </svg>
          <svg viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true">
            {aligned && <polyline className="strip-aligned" points={aligned} />}
            {dump.events.slice(0, 10_000).map((frame, index) => {
              const x = ((frame - dump.window[0]) / span) * 1_000;
              return (
                <line
                  className="strip-event"
                  key={`${frame}-${index}`}
                  x1={x}
                  x2={x}
                  y1="30"
                  y2="40"
                />
              );
            })}
          </svg>
          <span
            className="strip-cursor"
            style={{ "--tracker-cursor": `${framePercent}%` } as CSSProperties}
          />
        </div>
      </div>
      {dump.events.length > 0 && (
        <div className="tracker-strip-key">
          <span>Bottom ticks: candidate events</span>
        </div>
      )}
    </div>
  );
}

const READ_SLOTS: TrackerFaceRead["slot"][] = ["up", "front", "right"];

function sampledLabColor(
  [lightness, a, b]: TrackerFaceRead["lab"][number],
): string {
  return `lab(${((lightness / 255) * 100).toFixed(2)}% ${(a - 128).toFixed(
    2,
  )} ${(b - 128).toFixed(2)})`;
}

function quadPoint(
  corners: TrackerFaceRead["corners"],
  u: number,
  v: number,
): [number, number] {
  const [topLeft, topRight, bottomRight, bottomLeft] = corners;
  const topWeight = 1 - v;
  const bottomWeight = v;
  return [
    (1 - u) * topWeight * topLeft[0] +
      u * topWeight * topRight[0] +
      u * bottomWeight * bottomRight[0] +
      (1 - u) * bottomWeight * bottomLeft[0],
    (1 - u) * topWeight * topLeft[1] +
      u * topWeight * topRight[1] +
      u * bottomWeight * bottomRight[1] +
      (1 - u) * bottomWeight * bottomLeft[1],
  ];
}

function readCellPoints(
  corners: TrackerFaceRead["corners"],
  cellIndex: number,
): string {
  const column = cellIndex % 3;
  const row = Math.floor(cellIndex / 3);
  const left = column / 3;
  const right = (column + 1) / 3;
  const top = row / 3;
  const bottom = (row + 1) / 3;
  return [
    quadPoint(corners, left, top),
    quadPoint(corners, right, top),
    quadPoint(corners, right, bottom),
    quadPoint(corners, left, bottom),
  ]
    .map(([x, y]) => `${x},${y}`)
    .join(" ");
}

function SampledReads({ reads = [] }: { reads?: TrackerFaceRead[] }) {
  const bySlot = new Map(reads.map((read) => [read.slot, read]));
  return (
    <div className="tracker-current-reads-body">
      <div className="tracker-read-faces">
        {READ_SLOTS.map((slot) => {
          const read = bySlot.get(slot);
          const meanSupport = read
            ? read.confidence.reduce((sum, value) => sum + value, 0) /
              read.confidence.length
            : null;
          return (
            <div className="tracker-read-face" key={slot}>
              <span className="tracker-read-slot">{slot}</span>
              <div
                className={`tracker-read-grid${
                  read ? "" : " tracker-read-grid-empty"
                }`}
                role="img"
                aria-label={
                  read
                    ? `${slot} face: nine Lab samples from the video`
                    : `${slot} face: no sample on this frame`
                }
              >
                {Array.from({ length: 9 }, (_, index) => {
                  const lab = read?.lab[index];
                  return (
                    <span
                      className="tracker-read-cell"
                      key={index}
                      style={
                        lab ? { backgroundColor: sampledLabColor(lab) } : undefined
                      }
                      title={
                        lab
                          ? `Lab ${lab
                              .map((value) => value.toFixed(1))
                              .join(", ")}`
                          : undefined
                      }
                    />
                  );
                })}
              </div>
              <small>
                {meanSupport === null
                  ? "not sampled"
                  : `${meanSupport.toFixed(2)} support`}
              </small>
            </div>
          );
        })}
      </div>
      <p className="tracker-read-note">
        Camera samples only. These colors are not sticker classifications.
      </p>
    </div>
  );
}

function RunVideoStage({
  videoUrl,
  videoName,
  workspaceVideo,
  videoRef,
  frameIndex,
  overlayWidth,
  overlayHeight,
  reads,
  faces,
  nativeControls = false,
  onVideoError,
  onLoadedMetadata,
  onPlay,
  onPause,
}: {
  videoUrl: string;
  videoName: string;
  workspaceVideo: boolean;
  videoRef: RefObject<HTMLVideoElement>;
  frameIndex: number;
  overlayWidth: number;
  overlayHeight: number;
  reads: TrackerFaceRead[];
  faces: TrackerFace[];
  nativeControls?: boolean;
  onVideoError: () => void;
  onLoadedMetadata: (event: SyntheticEvent<HTMLVideoElement>) => void;
  onPlay: () => void;
  onPause: () => void;
}) {
  return (
    <div className="tracker-video-stage">
      <video
        ref={videoRef}
        src={videoUrl}
        muted
        controls={nativeControls}
        playsInline
        preload="metadata"
        onError={onVideoError}
        onLoadedMetadata={onLoadedMetadata}
        onPlay={onPlay}
        onPause={onPause}
      />
      {overlayWidth > 0 && overlayHeight > 0 && (
        <svg
          className="tracker-pose-overlay"
          viewBox={`0 0 ${overlayWidth} ${overlayHeight}`}
          preserveAspectRatio="xMidYMid meet"
          aria-label={`Pose overlay for frame ${frameIndex}`}
        >
          {reads.flatMap((read, readIndex) =>
            read.lab.map((lab, cellIndex) => (
              <polygon
                className="tracker-read-overlay-cell"
                key={`${read.slot}-${readIndex}-${cellIndex}`}
                points={readCellPoints(read.corners, cellIndex)}
                style={{
                  fill: sampledLabColor(lab),
                  fillOpacity: Math.max(
                    0.08,
                    Math.min(0.72, read.confidence[cellIndex] * 0.72),
                  ),
                }}
              />
            )),
          )}
          {faces.map((face, faceIndex) => (
            <g key={faceIndex}>
              <polygon
                className="tracker-face-outline"
                points={face.corners.map(([x, y]) => `${x},${y}`).join(" ")}
              />
              {face.corners.map(([x, y], cornerIndex) => (
                <g key={cornerIndex}>
                  <circle
                    className={`tracker-confidence tracker-confidence-${
                      face.kpt_conf[cornerIndex] >= 0.75
                        ? "high"
                        : face.kpt_conf[cornerIndex] >= 0.5
                          ? "mid"
                          : "low"
                    }`}
                    cx={x}
                    cy={y}
                    r="9"
                  />
                  <text x={x + 13} y={y - 10}>
                    {face.kpt_conf[cornerIndex].toFixed(2)}
                  </text>
                </g>
              ))}
            </g>
          ))}
        </svg>
      )}
      <span className="stage-frame-stamp">
        F{String(frameIndex).padStart(6, "0")}
      </span>
      <span className="stage-source-stamp" title={videoName}>
        {workspaceVideo ? "workspace video" : videoName}
      </span>
    </div>
  );
}

function RunTransport({
  window,
  frameIndex,
  playing,
  videoAvailable,
  onTogglePlaying,
  onSeek,
}: {
  window: [number, number];
  frameIndex: number;
  playing: boolean;
  videoAvailable: boolean;
  onTogglePlaying: () => void;
  onSeek: (frame: number) => void;
}) {
  return (
    <div className="demo-transport">
      <div
        className="demo-transport-row tracker-frame-transport"
        role="group"
        aria-label="Frame transport"
        aria-keyshortcuts="ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight Home End Space"
      >
        <button
          className="button button-primary"
          type="button"
          aria-pressed={playing}
          disabled={!videoAvailable}
          onClick={onTogglePlaying}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          className="button button-quiet"
          type="button"
          disabled={frameIndex === window[0]}
          onClick={() => onSeek(window[0])}
        >
          Start
        </button>
        <button
          className="button button-quiet"
          type="button"
          disabled={frameIndex === window[0]}
          onClick={() => onSeek(frameIndex - 1)}
        >
          Back
        </button>
        <button
          className="button button-quiet"
          type="button"
          disabled={frameIndex === window[1]}
          onClick={() => onSeek(frameIndex + 1)}
        >
          Forward
        </button>
        <p className="demo-frame-readout">
          frame <strong>{frameIndex}</strong> of {window[1]}
        </p>
      </div>
      <input
        className="frame-scrubber tracker-frame-scrubber"
        type="range"
        min={window[0]}
        max={window[1]}
        value={frameIndex}
        aria-label="Current decode frame"
        onChange={(event) => onSeek(Number(event.target.value))}
      />
    </div>
  );
}

const INT_COLORS = ["white", "yellow", "red", "orange", "blue", "green"];
const FACE_SLICES: Array<[string, number]> = [
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

function stateToNet(state: Int8Array): Record<string, string[]> {
  return Object.fromEntries(
    FACE_SLICES.map(([face, offset]) => [
      face,
      Array.from(
        state.slice(offset, offset + 9),
        (value) => INT_COLORS[value] ?? "white",
      ),
    ]),
  );
}

function estimatedMovesTowardCheckpoint({
  frame,
  previousFrame,
  checkpointFrame,
  moveCount,
}: {
  frame: number;
  previousFrame: number;
  checkpointFrame: number;
  moveCount: number;
}): number {
  if (
    moveCount <= 1 ||
    frame <= previousFrame ||
    frame >= checkpointFrame ||
    checkpointFrame <= previousFrame
  ) {
    return 0;
  }
  const progress = (frame - previousFrame) / (checkpointFrame - previousFrame);
  return Math.min(moveCount - 1, Math.floor(progress * moveCount));
}

function ReconstructionAtFrame({
  result,
  sequence,
  frame,
  startingScramble,
}: {
  result: DecodeResultDocument;
  sequence?: DecodeWorkstationSequence;
  frame: number;
  startingScramble: string;
}) {
  const scramble = useMemo(
    () =>
      startingScramble
        .trim()
        .split(/\s+/)
        .filter(Boolean),
    [startingScramble],
  );
  const trajectory = useMemo(
    () => cubeTrajectory([...scramble, ...result.moves]).slice(scramble.length),
    [result.moves, scramble],
  );
  const reconstruction = result.workstation?.reconstruction;
  const reconstructionTimeline = reconstruction?.timeline;
  const hasReconstructionTimeline = Boolean(
    reconstructionTimeline &&
      reconstruction &&
      reconstruction.states.length === reconstructionTimeline.moves.length + 1,
  );
  const reconstructionCheckpoints = useMemo(() => {
    if (!hasReconstructionTimeline || !reconstructionTimeline) return [];
    return reconstructionTimeline.moves.reduce<
      Array<{ frame: number; startStep: number; endStep: number }>
    >((checkpoints, move, index) => {
      const previous = checkpoints[checkpoints.length - 1];
      if (previous?.frame === move.frame) {
        previous.endStep = index + 1;
      } else {
        checkpoints.push({
          frame: move.frame,
          startStep: index,
          endStep: index + 1,
        });
      }
      return checkpoints;
    }, []);
  }, [hasReconstructionTimeline, reconstructionTimeline]);
  const checkpointIndex = reconstructionCheckpoints.filter(
    (checkpoint) => checkpoint.frame <= frame,
  ).length;
  const currentCheckpoint =
    checkpointIndex > 0
      ? reconstructionCheckpoints[checkpointIndex - 1]
      : undefined;
  const nextCheckpoint = reconstructionCheckpoints[checkpointIndex];
  const timelineStep = currentCheckpoint?.endStep ?? 0;
  const canonicalStep = sequence
    ? Math.min(
        result.moves.length,
        sequence.moves.filter((move) => move.frame <= frame).length,
      )
    : 0;
  const previousCheckpointFrame =
    currentCheckpoint?.frame ?? result.workstation?.window[0] ?? 0;
  const nextCheckpointMoveCount = nextCheckpoint
    ? nextCheckpoint.endStep - nextCheckpoint.startStep
    : 0;
  const estimatedMoveCount = nextCheckpoint
    ? estimatedMovesTowardCheckpoint({
        frame,
        previousFrame: previousCheckpointFrame,
        checkpointFrame: nextCheckpoint.frame,
        moveCount: nextCheckpointMoveCount,
      })
    : 0;
  const isEstimatedBetweenCheckpoints = Boolean(
    nextCheckpoint &&
      estimatedMoveCount > 0 &&
      frame > previousCheckpointFrame &&
      frame < nextCheckpoint.frame,
  );
  const step = hasReconstructionTimeline
    ? timelineStep + estimatedMoveCount
    : canonicalStep;
  const state =
    hasReconstructionTimeline && reconstruction
      ? reconstruction.states[Math.min(step, reconstruction.states.length - 1)]
      : stateToNet(trajectory[Math.min(step, trajectory.length - 1)]);
  const move =
    step > 0
      ? hasReconstructionTimeline && reconstructionTimeline
        ? estimatedMoveCount > 0
          ? reconstructionTimeline.moves[step - 1].move
          : currentCheckpoint &&
            currentCheckpoint.endStep - currentCheckpoint.startStep === 1
            ? reconstructionTimeline.moves[step - 1].move
            : null
        : result.moves[step - 1]
      : null;
  if (!state) return null;
  return (
    <div className="decode-run-reconstruction">
      <CubeState
        compact
        state={state}
        faceHighlight={move ? MOVE_FACES[move[0]?.toUpperCase() ?? ""] : undefined}
      />
      <p>
        {hasReconstructionTimeline && reconstructionTimeline
          ? isEstimatedBetweenCheckpoints && nextCheckpoint
            ? `Frame ${frame} · estimated move ${step}/${reconstructionTimeline.moves.length} · next checkpoint f${nextCheckpoint.frame}`
            : checkpointIndex === 0
            ? `Frame ${frame} · starting scramble${
                nextCheckpoint
                  ? ` · next decoder checkpoint f${nextCheckpoint.frame}`
                  : ""
              }`
            : `Frame ${frame} · decoder checkpoint ${checkpointIndex}/${
                reconstructionCheckpoints.length
              } · ${step} of ${reconstructionTimeline.moves.length} moves`
          : sequence
          ? `Frame ${frame} · after move ${step} of ${result.moves.length}${
              move ? ` (${move})` : ""
            }`
          : `Frame ${frame} · starting scramble · no decoder checkpoints`}
      </p>
    </div>
  );
}

function BleReference({
  diagnostic,
}: {
  diagnostic: DecodeGroundTruthDiagnostic;
}) {
  const frameIndexed =
    diagnostic.reference.scope === "sequence-and-clip-frame-indexed";
  return (
    <section
      className="decode-run-ble-reference"
      aria-labelledby="decode-run-ble-reference-title"
    >
      <div className="decode-run-ble-reference-head">
        <div>
          <p className="station-code">BLE reference</p>
          <h3 id="decode-run-ble-reference-title">
            Published smart-cube sequence
          </h3>
        </div>
        <p>
          {frameIndexed
            ? "Reviewed clip-local timing is available separately and is not applied to the decoded video timeline."
            : "Move order only · no video-frame timing."}
        </p>
      </div>
      <dl className="decode-run-ble-reference-metrics">
        <div>
          <dt>Decoded</dt>
          <dd>{diagnostic.counts.decoded_htm} HTM</dd>
        </div>
        <div>
          <dt>BLE</dt>
          <dd>{diagnostic.counts.ble_canonical_htm} HTM</dd>
        </div>
        <div>
          <dt>Edit distance</dt>
          <dd>{diagnostic.comparison.distance}</dd>
        </div>
        <div>
          <dt>Use</dt>
          <dd>Diagnostic only</dd>
        </div>
      </dl>
      <div
        className="decode-run-ble-comparison"
        aria-label="Decoded moves compared with the BLE reference"
      >
        <div className="decode-run-ble-comparison-labels" aria-hidden="true">
          <span>Decoded</span>
          <span>BLE</span>
        </div>
        {diagnostic.comparison.ops.map((operation, index) => (
          <div
            className="decode-run-ble-pair"
            data-operation={operation.op}
            key={`${operation.op}-${operation.index_decoded ?? "x"}-${
              operation.index_reference ?? "x"
            }-${index}`}
            title={operation.op}
          >
            <span>{operation.decoded ?? "·"}</span>
            <span>{operation.reference ?? "·"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

async function sha256File(file: File): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return bytesToHex(new Uint8Array(digest));
}

export function DecodeRunInspector({
  captures = [],
  captureId = "",
  result,
  job = null,
  resultName = "",
  portable = false,
}: {
  captures?: CaptureReceipt[];
  captureId?: string;
  result?: DecodeResultDocument | null;
  job?: DecodeJobStatus | null;
  resultName?: string;
  portable?: boolean;
}) {
  const [runResult, setRunResult] = useState<DecodeResultDocument | null>(
    result ?? null,
  );
  const [runName, setRunName] = useState(resultName);
  const [sourceResultFile, setSourceResultFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoName, setVideoName] = useState("");
  const [workspaceVideo, setWorkspaceVideo] = useState(false);
  const [videoFailed, setVideoFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [frameIndex, setFrameIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [pairingNote, setPairingNote] = useState("");
  const resultFileInputRef = useRef<HTMLInputElement>(null);
  const videoFileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoObjectUrlRef = useRef("");
  const pairingGenerationRef = useRef(0);

  const conversion = useMemo(() => {
    try {
      return {
        dump: runResult ? decodeWorkstationAsTrackerDump(runResult) : null,
        error: "",
      };
    } catch (reason) {
      return {
        dump: null,
        error:
          reason instanceof Error
            ? `The run result loaded, but its workstation diagnostics were rejected: ${reason.message}`
            : "The run result loaded, but its workstation diagnostics were rejected.",
      };
    }
  }, [runResult]);
  const dump = conversion.dump;
  const expectedVideoSha = runResult
    ? decodeResultVideoSha256(runResult)
    : null;
  const declaredVideoSha = expectedVideoSha ?? job?.video_sha256 ?? null;
  const pairedCapture = useMemo(() => {
    const selectedCaptureId = captureId || job?.capture_id || "";
    const selected = captures.find(
      (capture) => capture.capture_id === selectedCaptureId,
    );
    if (declaredVideoSha) {
      if (selected?.video.sha256 === declaredVideoSha) return selected;
      return (
        captures.find(
          (capture) => capture.video.sha256 === declaredVideoSha,
        ) ?? null
      );
    }
    return job && selected ? selected : null;
  }, [captureId, captures, declaredVideoSha, job]);
  const workstationVideo = runResult?.workstation?.video;
  const recordingFps =
    workstationVideo?.encoded.fps ??
    pairedCapture?.video.actual_fps ??
    pairedCapture?.video.configured_fps ??
    null;
  const recordingFrameCount =
    workstationVideo?.encoded.frame_count ??
    pairedCapture?.video.frame_count ??
    null;
  const recordingWidth =
    workstationVideo?.encoded.width ??
    pairedCapture?.video.encoded_width ??
    pairedCapture?.probe.width ??
    null;
  const recordingHeight =
    workstationVideo?.encoded.height ??
    pairedCapture?.video.encoded_height ??
    pairedCapture?.probe.height ??
    null;
  const startingScramble =
    runResult?.workstation?.initialization.scramble ??
    pairedCapture?.solve?.scramble ??
    "";
  const recordingDuration =
    recordingFps !== null &&
    recordingFps > 0 &&
    recordingFrameCount !== null &&
    recordingFrameCount > 0
      ? recordingFrameCount / recordingFps
      : null;
  const frameWindow = useMemo<[number, number] | null>(
    () =>
      dump?.window ??
      (recordingFrameCount !== null && recordingFrameCount > 0
        ? [0, recordingFrameCount - 1]
        : null),
    [dump, recordingFrameCount],
  );
  const fps = dump?.fps ?? recordingFps ?? 120;

  const clearVideo = useCallback(() => {
    videoRef.current?.pause();
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
      videoObjectUrlRef.current = "";
    }
    setVideoUrl("");
    setVideoName("");
    setWorkspaceVideo(false);
    setVideoFailed(false);
    setPlaying(false);
    setVideoDimensions({ width: 0, height: 0 });
  }, []);

  const installWorkspaceVideo = useCallback(
    (url: string, name: string) => {
      clearVideo();
      setVideoUrl(url);
      setVideoName(name);
      setWorkspaceVideo(true);
      setPairingNote(`Paired exact workspace video ${name}.`);
    },
    [clearVideo],
  );

  const installLocalVideo = useCallback(
    (file: File) => {
      clearVideo();
      const nextUrl = URL.createObjectURL(file);
      videoObjectUrlRef.current = nextUrl;
      setVideoUrl(nextUrl);
      setVideoName(file.name);
      setWorkspaceVideo(false);
      setPairingNote(`Paired exact local video ${file.name}.`);
    },
    [clearVideo],
  );

  useEffect(() => {
    if (result === undefined) return;
    setRunResult(result);
    setRunName(resultName);
    setSourceResultFile(null);
    setMessage("");
    setError("");
  }, [result, resultName]);

  useEffect(() => {
    pairingGenerationRef.current += 1;
    const generation = pairingGenerationRef.current;
    clearVideo();
    setPairingNote("");
    if (!runResult) return;
    if (!pairedCapture) {
      setPairingNote(
        declaredVideoSha
          ? `No workspace video matches ${declaredVideoSha.slice(
              0,
              12,
            )}…. Choose the matching video to enable playback and overlays.`
          : "This portable run does not include enough source identity to pair a video safely.",
      );
      return;
    }

    void createCaptureMediaTicket(pairedCapture.capture_id)
      .then((ticket) => {
        if (pairingGenerationRef.current !== generation) return;
        installWorkspaceVideo(ticket.url, pairedCapture.original_filename);
      })
      .catch((reason) => {
        if (pairingGenerationRef.current !== generation) return;
        setPairingNote(
          `The exact workspace video was found, but could not be opened: ${
            reason instanceof Error ? reason.message : "media ticket unavailable"
          }. You can still choose a matching local copy.`,
        );
      });
  }, [
    clearVideo,
    declaredVideoSha,
    installWorkspaceVideo,
    pairedCapture,
    runResult,
  ]);

  useEffect(
    () => () => {
      pairingGenerationRef.current += 1;
      if (videoObjectUrlRef.current) {
        URL.revokeObjectURL(videoObjectUrlRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (!frameWindow) return;
    setFrameIndex(frameWindow[0]);
    if (videoRef.current && videoRef.current.readyState >= 1) {
      videoRef.current.currentTime = (frameWindow[0] + 0.5) / fps;
    }
  }, [fps, frameWindow]);

  useEffect(() => {
    let animationFrame = 0;
    const update = () => {
      const video = videoRef.current;
      if (frameWindow && video && !video.paused) {
        setFrameIndex(
          Math.max(
            frameWindow[0],
            Math.min(
              frameWindow[1],
              Math.floor(video.currentTime * fps),
            ),
          ),
        );
        animationFrame = window.requestAnimationFrame(update);
      }
    };
    const start = () => {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = window.requestAnimationFrame(update);
    };
    const video = videoRef.current;
    video?.addEventListener("play", start);
    return () => {
      video?.removeEventListener("play", start);
      window.cancelAnimationFrame(animationFrame);
    };
  }, [fps, frameWindow, videoUrl]);

  const loadResult = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const parsed = parseDecodeResultDocument(
        await readJsonFile(file, 64 * 1024 * 1024, "Run JSON"),
      );
      setRunResult(parsed);
      setRunName(file.name);
      setSourceResultFile(file);
      setMessage(`Opened ${file.name}.`);
    } catch (reason) {
      setError(
        reason instanceof LocalContractError
          ? reason.message
          : "The run JSON could not be read.",
      );
    } finally {
      setBusy(false);
    }
  };

  const loadVideo = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !declaredVideoSha) return;
    setBusy(true);
    setError("");
    try {
      const actual = await sha256File(file);
      if (actual !== declaredVideoSha) {
        setError(
          `Video SHA mismatch. Expected ${declaredVideoSha}. Selected ${actual}. The run remains available, but this video was not paired.`,
        );
        return;
      }
      installLocalVideo(file);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? `The selected video could not be verified: ${reason.message}`
          : "The selected video could not be verified.",
      );
    } finally {
      setBusy(false);
    }
  };

  const seekFrame = useCallback(
    (requested: number) => {
      if (!frameWindow) return;
      const next = Math.max(
        frameWindow[0],
        Math.min(frameWindow[1], Math.round(requested)),
      );
      const video = videoRef.current;
      video?.pause();
      if (video && Number.isFinite(video.duration)) {
        video.currentTime = Math.min(
          Math.max(0, video.duration - 0.000_001),
          (next + 0.5) / fps,
        );
      }
      setFrameIndex(next);
    },
    [fps, frameWindow],
  );

  const togglePlaying = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch(() =>
        setError("Video playback could not start. Reopen the source video."),
      );
    } else {
      video.pause();
    }
  }, []);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const element = event.target as HTMLElement | null;
      const tag = element?.tagName ?? "";
      if (
        tag === "INPUT" ||
        tag === "SELECT" ||
        tag === "TEXTAREA" ||
        element?.isContentEditable ||
        !frameWindow
      ) {
        return;
      }
      switch (event.key) {
        case "ArrowLeft":
          seekFrame(frameIndex - (event.shiftKey ? 10 : 1));
          break;
        case "ArrowRight":
          seekFrame(frameIndex + (event.shiftKey ? 10 : 1));
          break;
        case "Home":
          seekFrame(frameWindow[0]);
          break;
        case "End":
          seekFrame(frameWindow[1]);
          break;
        case " ":
          if (tag === "BUTTON" || tag === "A") return;
          togglePlaying();
          break;
        default:
          return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [frameIndex, frameWindow, seekFrame, togglePlaying]);

  const entries = useMemo(() => (dump ? frameEntries(dump) : []), [dump]);
  const current = dump?.frames[String(frameIndex)] ?? null;
  const runSequence = runResult
    ? canonicalDecodeSequence(runResult)
    : null;
  const sequenceMismatch = Boolean(
    runResult?.workstation?.sequence && !runSequence,
  );
  const currentSequenceIndex = runSequence
    ? runSequence.moves.reduce(
        (last, entry, index) => (entry.frame <= frameIndex ? index : last),
        -1,
      )
    : -1;
  const currentSequenceEntry =
    runSequence && currentSequenceIndex >= 0
      ? runSequence.moves[currentSequenceIndex]
      : null;
  const overlayWidth =
    dump?.width ?? recordingWidth ?? videoDimensions.width;
  const overlayHeight =
    dump?.height ?? recordingHeight ?? videoDimensions.height;
  const stageAspectStyle = useMemo<CSSProperties | undefined>(() => {
    if (overlayWidth <= 0 || overlayHeight <= 0) return undefined;
    return {
      "--tracker-stage-ratio": (overlayWidth / overlayHeight).toFixed(4),
    } as CSSProperties;
  }, [overlayHeight, overlayWidth]);
  const wallTimeLabel = useMemo(() => {
    const elapsedFrom = job?.started_at ?? job?.created_at;
    const started = elapsedFrom ? Date.parse(elapsedFrom) : NaN;
    const finished = job?.finished_at ? Date.parse(job.finished_at) : NaN;
    if (
      !Number.isFinite(started) ||
      !Number.isFinite(finished) ||
      finished < started
    ) {
      return null;
    }
    const seconds = (finished - started) / 1_000;
    return seconds < 60
      ? `${seconds.toFixed(1)}s`
      : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  }, [job?.created_at, job?.finished_at, job?.started_at]);
  const downloadName =
    resultName ||
    `cubed-core-run-${job?.job_id ?? runResult?.recording_id ?? "result"}.json`;
  const downloadExactResult = async () => {
    setError("");
    try {
      const artifact = job
        ? await fetchDecodeJobResultBlob(job.job_id)
        : sourceResultFile;
      if (!artifact) {
        throw new Error(
          "The exact source bytes are unavailable for this run.",
        );
      }
      triggerBlobDownload(
        artifact,
        downloadName.endsWith(".json")
          ? downloadName
          : `${safeDownloadStem(downloadName, "run")}.json`,
      );
      setMessage("Downloaded the exact run JSON.");
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "The exact run artifact could not be downloaded.",
      );
    }
  };
  const hasReconstruction = Boolean(
    runResult?.status === "completed" &&
      runResult.moves.length > 0 &&
      (runResult.workstation?.initialization || pairedCapture?.solve),
  );
  const hasTrellis = Boolean(
    dump &&
      (dump.spans.length > 0 || (dump.bridge && dump.bridge.length > 0)),
  );
  const hasReads = entries.some(([, frame]) => Boolean(frame.reads?.length));
  const videoStage =
    videoUrl ? (
      <RunVideoStage
        videoUrl={videoUrl}
        videoName={videoName}
        workspaceVideo={workspaceVideo}
        videoRef={videoRef}
        frameIndex={frameIndex}
        overlayWidth={dump ? overlayWidth : 0}
        overlayHeight={dump ? overlayHeight : 0}
        reads={current?.reads ?? []}
        faces={current?.faces ?? []}
        nativeControls={!frameWindow}
        onVideoError={() => {
          setVideoFailed(true);
          setError(
            "The paired video became unavailable. The run diagnostics remain available.",
          );
        }}
        onLoadedMetadata={(event) => {
          setVideoDimensions({
            width: event.currentTarget.videoWidth,
            height: event.currentTarget.videoHeight,
          });
          event.currentTarget.currentTime = (frameIndex + 0.5) / fps;
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
    ) : null;

  return (
    <div className="page local-tool-page tracker-tool-page tracker-inspection-only">
      {portable && (
        <section
          className="local-source-rack tracker-source-rack tracker-source-rack-portable"
          aria-labelledby="decode-run-source-title"
        >
          <div className="source-rack-heading">
            <p className="station-code">Inputs</p>
            <h3 id="decode-run-source-title">Run and video</h3>
          </div>
          <div className="tracker-source-choice tracker-source-choice-json">
            <div>
              <strong>Run JSON</strong>
              <small>Required · downloaded from Decode or Runs.</small>
            </div>
            <input
              ref={resultFileInputRef}
              id="decode-run-file"
              className="tracker-source-file-input"
              type="file"
              tabIndex={-1}
              accept=".json,application/json"
              disabled={busy}
              onChange={loadResult}
            />
            <button
              className="button button-primary"
              type="button"
              disabled={busy}
              onClick={() => resultFileInputRef.current?.click()}
            >
              {busy ? "Checking…" : "Choose run JSON"}
            </button>
          </div>
          <div className="tracker-source-choice tracker-source-choice-video">
            <div>
              <strong>Matching video</strong>
              <small>
                {declaredVideoSha
                  ? "Optional · exact SHA required. Older results support video-only playback."
                  : "This run does not declare a source-video SHA."}
              </small>
            </div>
            <input
              ref={videoFileInputRef}
              id="decode-run-video-file"
              className="tracker-source-file-input"
              type="file"
              tabIndex={-1}
              accept="video/*,.mov,.mp4,.m4v,.webm"
              disabled={busy || !runResult || !declaredVideoSha}
              onChange={loadVideo}
            />
            <button
              className="button button-secondary"
              type="button"
              disabled={busy || !runResult || !declaredVideoSha}
              onClick={() => videoFileInputRef.current?.click()}
            >
              Choose matching video
            </button>
          </div>
        </section>
      )}

      {!portable &&
        runResult &&
        !videoUrl &&
        declaredVideoSha && (
          <div className="decode-run-video-action">
            <p>{pairingNote}</p>
            <input
              ref={videoFileInputRef}
              className="tracker-source-file-input"
              type="file"
              tabIndex={-1}
              accept="video/*,.mov,.mp4,.m4v,.webm"
              disabled={busy}
              onChange={loadVideo}
            />
            <button
              className="button button-secondary"
              type="button"
              disabled={busy}
              onClick={() => videoFileInputRef.current?.click()}
            >
              Choose matching video
            </button>
          </div>
        )}

      {conversion.error && (
        <p className="tool-message tool-message-error" role="alert">
          {conversion.error}
        </p>
      )}
      {error && (
        <p className="tool-message tool-message-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="tool-message tracker-loaded-status" role="status">
          {message}
        </p>
      )}
      {portable && runResult && pairingNote && (
        <p className="tool-message decode-run-pairing-note" role="status">
          {pairingNote}
        </p>
      )}

      {!runResult ? (
        <section className="tool-empty-state">
          <span className="empty-frame" aria-hidden="true">
            R
          </span>
          <div>
            <strong>No run JSON loaded</strong>
            <p>
              Choose the exact JSON downloaded from Decode or a workspace run.
            </p>
          </div>
        </section>
      ) : (
        <>
          <section
            className="decode-run-outcome"
            data-status={runResult.status}
            aria-label="Decode outcome"
          >
            <div>
              <p className="station-code">Decode outcome</p>
              <h3>
                {runResult.status === "completed"
                  ? `${runResult.moves.length} decoded move${
                      runResult.moves.length === 1 ? "" : "s"
                    }`
                  : "The run abstained."}
              </h3>
              <p>
                {runResult.status === "completed"
                  ? "The job completed on this input. The sequence replayed to solved. Review it against the recording."
                  : "No solved reconstruction was emitted."}
              </p>
            </div>
            <dl>
              <div>
                <dt>Profile</dt>
                <dd>{runResult.profile}</dd>
              </div>
              <div>
                <dt>CFG_HASH</dt>
                <dd>{runResult.config.cfg_hash}</dd>
              </div>
              {job && (
                <div>
                  <dt>Server replay</dt>
                  <dd>
                    {job.replay_solved_reached === null
                      ? "N/A"
                      : job.replay_solved_reached
                        ? "Solved"
                        : "Not solved"}
                  </dd>
                </div>
              )}
            </dl>
          </section>

          {(runResult.workstation || pairedCapture) && (
            <section
              className="decode-run-recording-summary"
              aria-label="Recording summary"
            >
              <div className="decode-run-recording-identity">
                <p className="station-code">Recording</p>
                <strong>
                  {pairedCapture?.original_filename ??
                    runResult.recording_id ??
                    "Portable recording"}
                </strong>
                <code>
                  {runResult.recording_id ||
                    job?.capture_id ||
                    captureId ||
                    "Capture ID unavailable"}
                </code>
              </div>
              <dl>
                <div>
                  <dt>Frames</dt>
                  <dd>
                    {recordingFrameCount ?? "N/A"}
                    {dump
                      ? ` · window ${dump.window[0]}–${dump.window[1]}`
                      : ""}
                  </dd>
                </div>
                <div>
                  <dt>Frame rate</dt>
                  <dd>
                    {recordingFps === null
                      ? "N/A"
                      : `${recordingFps.toFixed(3)} fps`}
                  </dd>
                </div>
                <div>
                  <dt>Resolution</dt>
                  <dd>
                    {recordingWidth !== null && recordingHeight !== null
                      ? `${recordingWidth}×${recordingHeight}`
                      : "N/A"}
                  </dd>
                </div>
                <div>
                  <dt>Duration</dt>
                  <dd>
                    {recordingDuration === null
                      ? "N/A"
                      : `${recordingDuration.toFixed(2)}s`}
                  </dd>
                </div>
              </dl>
              {startingScramble && (
                <p className="decode-run-starting-scramble">
                  <span>Starting scramble</span>
                  <code>{startingScramble}</code>
                </p>
              )}
            </section>
          )}

          <details className="tracker-receipt-disclosure">
            <summary>
              <span>Run JSON</span>
              <strong>{runName || downloadName}</strong>
              <small>
                {runResult.schema} · {runResult.status}
              </small>
            </summary>
            <div className="tracker-receipt-body">
              <dl className="tool-metrics tracker-run-metrics">
                <div>
                  <dt>Run time</dt>
                  <dd>{wallTimeLabel ?? "N/A"}</dd>
                </div>
                <div>
                  <dt>Video SHA</dt>
                  <dd title={declaredVideoSha ?? undefined}>
                    {declaredVideoSha?.slice(0, 12) ?? "N/A"}
                  </dd>
                </div>
                <div>
                  <dt>Artifact</dt>
                  <dd>{runName || downloadName}</dd>
                </div>
                {job?.result_sha256 && (
                  <div>
                    <dt>Artifact SHA</dt>
                    <dd title={job.result_sha256}>
                      {job.result_sha256.slice(0, 12)}
                    </dd>
                  </div>
                )}
                <div>
                  <dt>Config</dt>
                  <dd title={runResult.config.cfg_hash}>
                    {runResult.config.name} ·{" "}
                    {runResult.config.cfg_hash.slice(0, 12)}
                  </dd>
                </div>
              </dl>
              <button
                className="button button-secondary"
                type="button"
                disabled={!job && !sourceResultFile}
                onClick={() => void downloadExactResult()}
              >
                Download run JSON
              </button>
            </div>
          </details>

          {runResult.workstation?.warnings?.map((warning) => (
            <p className="tool-message decode-run-warning" role="note" key={warning.code}>
              <strong>{warning.code}</strong> · {warning.message}
            </p>
          ))}
          {sequenceMismatch && (
            <p className="tool-message decode-run-warning" role="note">
              <strong>sequence-mismatch</strong> · Stored move timing does not
              match the decoded move list, so sequence timing is omitted.
            </p>
          )}
          {dump && (
            <section
              className="tracker-workstation"
              aria-labelledby="decode-frame-evidence-title"
            >
              <div className="tracker-workstation-viewer">
                <div
                  className="tracker-workstation-video tracker-video-stage-standalone"
                  style={stageAspectStyle}
                >
                  {!videoFailed && videoStage ? (
                    videoStage
                  ) : (
                    <div className="tracker-workstation-video-empty">
                      <strong>
                        {videoFailed
                          ? "Source video unavailable"
                          : "Matching video needed"}
                      </strong>
                      <p>
                        Pair the exact video to enable playback and overlays.
                        The run diagnostics remain available.
                      </p>
                    </div>
                  )}
                </div>

                <RunTransport
                  window={dump.window}
                  frameIndex={frameIndex}
                  playing={playing}
                  videoAvailable={Boolean(videoUrl)}
                  onTogglePlaying={togglePlaying}
                  onSeek={seekFrame}
                />
              </div>

              <div className="tracker-workstation-evidence">
                <div className="tracker-frame-evidence-heading">
                  <div>
                    <p className="station-code">Per-frame diagnostics</p>
                    <h3 id="decode-frame-evidence-title">Frame signals</h3>
                  </div>
                  <span>{fps.toFixed(3)} fps</span>
                </div>

                <div
                  className={`tracker-frame-summary${
                    hasReads ? "" : " tracker-frame-summary-no-reads"
                  }`}
                >
                  {hasReads && (
                  <div className="tracker-current-reads">
                    <p className="station-code">Sampled reads</p>
                    <SampledReads reads={current?.reads} />
                  </div>
                  )}
                  <div className="tracker-current-status">
                    <dl className="tool-metrics tracker-frame-metrics">
                      <div>
                        <dt>Motion</dt>
                        <dd>{current?.motion?.toFixed(3) ?? "N/A"}</dd>
                      </div>
                      <div>
                        <dt>P(aligned)</dt>
                        <dd>{current?.aligned?.toFixed(3) ?? "N/A"}</dd>
                      </div>
                      <div>
                        <dt>Aligned streak</dt>
                        <dd>{current?.stk ?? "N/A"}</dd>
                      </div>
                      <div>
                        <dt>Faces</dt>
                        <dd>
                          {current?.nfaces ?? current?.faces?.length ?? "N/A"}
                        </dd>
                      </div>
                    </dl>
                    {runSequence && runSequence.moves.length > 0 && (
                      <p className="decode-run-frame-sequence-context">
                        <span>Decoded move</span>
                        <strong>
                          {currentSequenceEntry
                            ? `${currentSequenceIndex + 1}/${
                                runSequence.moves.length
                              } ${currentSequenceEntry.move}`
                            : `0/${runSequence.moves.length}`}
                        </strong>
                        <span>
                          {currentSequenceEntry
                            ? `at frame ${currentSequenceEntry.frame}`
                            : `next at frame ${runSequence.moves[0].frame}`}
                        </span>
                        <span>canonical timing</span>
                      </p>
                    )}
                  </div>
                </div>

                <RunSignalStrip
                  dump={dump}
                  currentFrame={frameIndex}
                  onSeek={seekFrame}
                />

                {(hasReconstruction || hasTrellis) && (
                  <div
                    className="tracker-inline-diagnostics"
                    aria-label="Run diagnostics"
                  >
                    {hasReconstruction && (
                      <section
                        className="tracker-diagnostic-section tracker-diagnostic-reconstruction"
                        aria-label="Reconstruction"
                      >
                        <h4>Reconstruction</h4>
                        <ReconstructionAtFrame
                          result={runResult}
                          sequence={runSequence ?? undefined}
                          frame={frameIndex}
                          startingScramble={startingScramble}
                        />
                      </section>
                    )}
                    {hasTrellis && (
                      <section
                        className="tracker-diagnostic-section tracker-diagnostic-trellis"
                        aria-label="Trellis and spans"
                      >
                        <h4>Trellis / spans</h4>
                        <TrellisBeam
                          dump={dump}
                          frameIdx={frameIndex}
                          seekToFrame={seekFrame}
                        />
                      </section>
                    )}
                  </div>
                )}

                {runResult.ground_truth_diagnostic && (
                  <BleReference
                    diagnostic={runResult.ground_truth_diagnostic}
                  />
                )}
              </div>
            </section>
          )}

          {!dump && frameWindow && pairedCapture && (
            <section
              className="tracker-workstation decode-run-legacy-workstation"
              aria-label="Recording playback"
            >
              <div className="tracker-workstation-viewer">
                <div
                  className="tracker-workstation-video tracker-video-stage-standalone"
                  style={stageAspectStyle}
                >
                  {!videoFailed && videoStage ? (
                    videoStage
                  ) : (
                    <div className="tracker-workstation-video-empty">
                      <strong>
                        {videoFailed
                          ? "Source video unavailable"
                          : "Opening workspace recording"}
                      </strong>
                      <p>
                        {pairingNote ||
                          "The capture video is being opened without recomputing the run."}
                      </p>
                    </div>
                  )}
                </div>
                <RunTransport
                  window={frameWindow}
                  frameIndex={frameIndex}
                  playing={playing}
                  videoAvailable={Boolean(videoUrl)}
                  onTogglePlaying={togglePlaying}
                  onSeek={seekFrame}
                />
              </div>
              <div className="tracker-workstation-evidence decode-run-legacy-evidence">
                <div>
                  <p className="station-code">Per-frame diagnostics</p>
                  <h3>Not embedded in this older result</h3>
                  <p>
                    The recording and frame clock come from the capture
                    receipt. This result has no stored overlays, reads, or
                    signal timeline. Nothing is recomputed here.
                  </p>
                </div>
              </div>
            </section>
          )}

          {!dump && !frameWindow && videoStage && (
            <section
              className="tracker-workstation decode-run-video-only-workstation"
              aria-label="Video-only playback"
            >
              <div className="tracker-workstation-viewer">
                <div
                  className="tracker-workstation-video tracker-video-stage-standalone"
                  style={stageAspectStyle}
                >
                  {videoStage}
                </div>
              </div>
              <div className="tracker-workstation-evidence decode-run-legacy-evidence">
                <div>
                  <p className="station-code">Video-only playback</p>
                  <h3>Frame diagnostics were not embedded</h3>
                  <p>
                    The exact video is paired by SHA. This portable result has
                    no stored FPS or frame window, so native video controls are
                    shown without a frame scrubber. Nothing is recomputed.
                  </p>
                </div>
              </div>
            </section>
          )}

          {!dump && runResult.ground_truth_diagnostic && (
            <BleReference
              diagnostic={runResult.ground_truth_diagnostic}
            />
          )}
        </>
      )}
    </div>
  );
}
