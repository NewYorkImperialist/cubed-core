// Explicit extension: the Node test runner strips types and resolves this
// specifier literally when a test imports this module directly.
import { simplifyMovesWithFrames } from "./moveFold.ts";
import { bytesToHex } from "./hex.ts";
import type { DecodeResultDocument } from "./types";

export interface PublishedDemoManifest {
  id: string;
  title: string;
  video: {
    filename: string;
    sha256: string;
  };
  scramble: string;
  resultUrl: string;
  receiptUrl: string;
  resultSha256: string;
  receiptSha256: string;
  jobId: string;
  profile: "local_camera_v1";
  cfgHash: string;
  implementationSha256: string;
  recordedRuntimeVersion: string;
  moveCount: number;
  claims: {
    scope: string;
    replay: string;
    exclusions: readonly string[];
    runnerIdentity: "unverified-external-identity";
  };
}

export interface PublishedDecodeReceipt {
  schema: "cubed-core/decode-job-receipt-v1";
  schema_version: 1;
  job_id: string;
  capture_id: string;
  profile: "local_camera_v1";
  evidence_scope: "reconstruction-evidence-with-replay-check";
  result_bytes: number;
  result_sha256: string;
  result_status: "completed";
  replay_check: {
    performed: true;
    solved_reached: true;
    move_count: number;
    detail: string;
  };
  runner_provenance: {
    runner_kind: "external";
    identity_status: "unverified-external-identity";
  };
}

export interface OverlayFace {
  /** Face quad in normalized frame coordinates, in sampling order. */
  q: readonly (readonly [number, number])[];
  /** Face detector confidence. */
  c: number;
  /** Nine sampled colors as one hex string of nine RGB triplets. */
  k?: string;
  /** Nine per-facelet sampling confidences. */
  v?: readonly number[];
  /** Slot the tracker assigned this face. */
  s?: string;
}

export interface OverlayFrame {
  f: readonly OverlayFace[];
  /** Alignment score for the frame. */
  a: number;
  /** Tracker stack depth for the frame. */
  t: number;
}

export interface PublishedOverlayTrack {
  schema: "cubed-core/demo-overlay-track-v1";
  schema_version: 1;
  capture_id: string;
  mode: string;
  source_sha256: string;
  fps: number;
  window: readonly [number, number];
  frame_count: number;
  detected_frames: number;
  detected_faces: number;
  gated_frames: number;
  color_encoding: string;
  color_meaning: string;
  frames: Record<string, OverlayFrame>;
}

/** One recorded quarter turn from the smart cube, on a clip-local frame. */
export interface BleGroundTruthMove {
  index: number;
  frame: number;
  move: string;
}

export interface PublishedBleGroundTruth {
  schema: "cubed-core/clip-ble-ground-truth-v1";
  schema_version: 1;
  tag: string;
  moves: readonly BleGroundTruthMove[];
  canonical_moves: readonly string[];
}

/** A decoder-emitted move with the clip frame its turn was recorded on. */
export interface TimedMove {
  move: string;
  frame: number;
}

export interface LoadedPublishedDemo {
  result: DecodeResultDocument;
  receipt: PublishedDecodeReceipt;
  resultSha256: string;
}

export const GTD1_PUBLISHED_DEMO = {
  id: "gtd1",
  title: "gtD1 published reconstruction",
  video: {
    filename: "gtD1s_decoder-demo-clip_f3919-f9722.mp4",
    sha256: "7329c9b3c46007f618b043a61ee00b34c56c74aa22b8a803270c6ea4c9013776",
  },
  scramble: "R' L2 F L2 U R D F' U2 L B2 D2 F2 L2 B2 R2 U F2 B2 L2 D",
  resultUrl: "/demo/gtd1/decode-result.json",
  receiptUrl: "/demo/gtd1/decode-receipt.json",
  resultSha256: "9cbb4b17860dd88af61eb1da68f86c9c60c0f37054c5eeca505126fbfac8702b",
  receiptSha256: "96ba873a9fa89a39cc3d5d5d4e6b63d002814f991d649d51b9c673230acef410",
  jobId: "57d2f76317557205e479b14602dcc1f4",
  profile: "local_camera_v1",
  cfgHash: "1852738634",
  implementationSha256:
    "22124598154864394d0da5cd7a74b9b97e89097ee84596525d48fb2c81026c23",
  recordedRuntimeVersion: "0.2.0",
  moveCount: 77,
  claims: {
    scope: "Published reconstruction result and endpoint replay receipt.",
    replay: "The sequence replayed to solved.",
    exclusions: [
      "Not live inference.",
      "Not per-move accuracy.",
      "Not reach-LL evidence.",
      "Not generalization evidence.",
    ],
    runnerIdentity: "unverified-external-identity",
  },
} as const satisfies PublishedDemoManifest;

/** Release tag that publishes the demo clip. */
export const GTD1_RELEASE_TAG = "v1.0.0";

const RELEASE_DOWNLOAD_BASE =
  "https://github.com/KingBobJoeIV/cubed-core/releases/download";

/** Remote clip URL. The page keeps working when this asset cannot load. */
export const GTD1_VIDEO_URL = `${RELEASE_DOWNLOAD_BASE}/${GTD1_RELEASE_TAG}/${GTD1_PUBLISHED_DEMO.video.filename}`;

/** Measured clip timing. A playback second maps to `fps` source frames. */
export const GTD1_VIDEO_TIMING = {
  fps: 696480 / 5803,
  frameCount: 5804,
  lastFrame: 5803,
  width: 1080,
  height: 1920,
} as const;

export const GTD1_OVERLAY_TRACK = {
  url: "/demo/gtd1/overlay-track.json",
  captureId: "6fd2d3257fe2b2107941ec20bbbdc2bc",
  sha256: "fdce4748c5a1adf7a13c1b0a84c902b00b4607d336d1fd04c56cace05f36ca60",
  bytes: 2225394,
  sourceSha256:
    "d03896850601914971c4d0a722060f0dee281954e508c7df1dfe0002e739d3a3",
  detectedFrames: 4102,
  detectedFaces: 10643,
} as const;

export const GTD1_BLE_GROUND_TRUTH = {
  url: "/demo/gtd1/ble-ground-truth.json",
  tag: "gtD1s",
  sha256: "8b26129ba8f24929705433f8b8579b2d56fd840b19fb48d05f38c56f8295eec3",
  bytes: 130844,
  quarterTurns: 91,
  canonicalMoves: 77,
  firstMoveFrame: 120,
  lastMoveFrame: 5563,
  card:
    "https://huggingface.co/datasets/cubed-core/cubed-data-v1/blob/7fae604962c590ac9c658ba6ee0350e86de9c4f5/README.md",
} as const;

/**
 * Pair each decoded move with the clip frame its turn was recorded on.
 *
 * A wrong fold desynchronizes the cube from the clip without erroring, so the
 * result is checked against the artifact's own canonical list.
 */
export function bleTimedMoves(
  ble: PublishedBleGroundTruth,
  decoded: readonly string[],
): TimedMove[] {
  const folded = simplifyMovesWithFrames(ble.moves);
  if (
    folded.moves.length !== ble.canonical_moves.length ||
    folded.moves.some((token, index) => token !== ble.canonical_moves[index])
  ) {
    throw new Error(
      "The BLE quarter-turn fold does not reproduce the published canonical moves.",
    );
  }
  if (folded.moves.length !== decoded.length) {
    throw new Error(
      `The BLE record folds to ${folded.moves.length} moves and the decoder emitted ${decoded.length}.`,
    );
  }
  const timed = folded.frames.map((frame, index) => ({
    move: decoded[index],
    frame,
  }));
  for (let index = 1; index < timed.length; index += 1) {
    if (timed[index].frame <= timed[index - 1].frame) {
      throw new Error("The BLE move frames are not strictly increasing.");
    }
  }
  return timed;
}

/** Index of the last move whose recorded turn is at or before `frame`. */
export function moveStepForFrame(
  timed: readonly TimedMove[],
  frame: number,
): number {
  let low = 0;
  let high = timed.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (timed[middle].frame <= frame) low = middle + 1;
    else high = middle;
  }
  return low;
}

/**
 * Frame and time round-trip through the middle of a frame's interval, so
 * seeking to a frame and reading the element's clock back lands on the same
 * frame with half a frame of slack in both directions. Seeking to the leading
 * edge instead leaves no margin below it, and a decoder that reports a hair
 * early steps the readout back one frame.
 */
export function frameIndexForTime(seconds: number): number {
  const frame = Math.floor(seconds * GTD1_VIDEO_TIMING.fps);
  return Math.max(0, Math.min(GTD1_VIDEO_TIMING.lastFrame, frame));
}

export function timeForFrameIndex(frame: number): number {
  return (frame + 0.5) / GTD1_VIDEO_TIMING.fps;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseJson(text: string, label: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`The published ${label} is not valid JSON.`);
  }
}

export function parsePublishedDecodeResult(value: unknown): DecodeResultDocument {
  if (!isRecord(value)) {
    throw new Error("The published result is not a JSON object.");
  }
  const config = value.config;
  const endpoint = value.endpoint;
  const moves = value.moves;
  if (
    value.schema !== "cubed-core/decode-result" ||
    value.schema_version !== 1 ||
    value.status !== "completed" ||
    value.profile !== GTD1_PUBLISHED_DEMO.profile ||
    !isRecord(config) ||
    config.cfg_hash !== GTD1_PUBLISHED_DEMO.cfgHash ||
    typeof config.name !== "string" ||
    !Array.isArray(moves) ||
    moves.length !== GTD1_PUBLISHED_DEMO.moveCount ||
    !moves.every((move) => typeof move === "string") ||
    !isRecord(endpoint) ||
    endpoint.solved_reached !== true ||
    !Array.isArray(value.inputs) ||
    !isRecord(value.provenance) ||
    value.provenance.implementation_sha256 !==
      GTD1_PUBLISHED_DEMO.implementationSha256 ||
    value.provenance.runtime_version !==
      GTD1_PUBLISHED_DEMO.recordedRuntimeVersion ||
    value.evaluation !== null
  ) {
    throw new Error(
      "The published result does not match the reviewed gtD1 result contract.",
    );
  }
  return value as unknown as DecodeResultDocument;
}

export function parsePublishedDecodeReceipt(value: unknown): PublishedDecodeReceipt {
  if (!isRecord(value)) {
    throw new Error("The published receipt is not a JSON object.");
  }
  const replayCheck = value.replay_check;
  const runnerProvenance = value.runner_provenance;
  if (
    value.schema !== "cubed-core/decode-job-receipt-v1" ||
    value.schema_version !== 1 ||
    value.job_id !== GTD1_PUBLISHED_DEMO.jobId ||
    value.profile !== GTD1_PUBLISHED_DEMO.profile ||
    value.evidence_scope !== "reconstruction-evidence-with-replay-check" ||
    value.result_sha256 !== GTD1_PUBLISHED_DEMO.resultSha256 ||
    value.result_status !== "completed" ||
    !isRecord(replayCheck) ||
    replayCheck.performed !== true ||
    replayCheck.solved_reached !== true ||
    replayCheck.move_count !== GTD1_PUBLISHED_DEMO.moveCount ||
    !isRecord(runnerProvenance) ||
    runnerProvenance.runner_kind !== "external" ||
    runnerProvenance.identity_status !==
      GTD1_PUBLISHED_DEMO.claims.runnerIdentity
  ) {
    throw new Error(
      "The published receipt does not match the reviewed gtD1 receipt contract.",
    );
  }
  return value as unknown as PublishedDecodeReceipt;
}

export function parsePublishedOverlayTrack(value: unknown): PublishedOverlayTrack {
  if (!isRecord(value)) {
    throw new Error("The published overlay track is not a JSON object.");
  }
  const frames = value.frames;
  if (
    value.schema !== "cubed-core/demo-overlay-track-v1" ||
    value.schema_version !== 1 ||
    value.capture_id !== GTD1_OVERLAY_TRACK.captureId ||
    value.source_sha256 !== GTD1_OVERLAY_TRACK.sourceSha256 ||
    value.mode !== "camera-only-native-vision-v1" ||
    value.frame_count !== GTD1_VIDEO_TIMING.frameCount ||
    value.detected_frames !== GTD1_OVERLAY_TRACK.detectedFrames ||
    value.detected_faces !== GTD1_OVERLAY_TRACK.detectedFaces ||
    !isRecord(frames) ||
    Object.keys(frames).length !== GTD1_OVERLAY_TRACK.detectedFrames
  ) {
    throw new Error(
      "The published overlay track does not match the reviewed gtD1 overlay contract.",
    );
  }
  return value as unknown as PublishedOverlayTrack;
}

export function parsePublishedBleGroundTruth(
  value: unknown,
): PublishedBleGroundTruth {
  if (!isRecord(value)) {
    throw new Error("The published BLE record is not a JSON object.");
  }
  const moves = value.moves;
  const canonical = value.canonical_moves;
  if (
    value.schema !== "cubed-core/clip-ble-ground-truth-v1" ||
    value.schema_version !== 1 ||
    value.tag !== GTD1_BLE_GROUND_TRUTH.tag ||
    !Array.isArray(moves) ||
    moves.length !== GTD1_BLE_GROUND_TRUTH.quarterTurns ||
    !moves.every(
      (move) =>
        isRecord(move) &&
        typeof move.move === "string" &&
        typeof move.frame === "number" &&
        Number.isInteger(move.frame),
    ) ||
    !Array.isArray(canonical) ||
    canonical.length !== GTD1_BLE_GROUND_TRUTH.canonicalMoves ||
    !canonical.every((move) => typeof move === "string")
  ) {
    throw new Error(
      "The published BLE record does not match the reviewed gtD1 ground-truth contract.",
    );
  }
  return value as unknown as PublishedBleGroundTruth;
}

export async function loadPublishedBleGroundTruth(
  signal?: AbortSignal,
): Promise<PublishedBleGroundTruth> {
  const response = await fetch(GTD1_BLE_GROUND_TRUTH.url, { signal });
  if (!response.ok) {
    throw new Error(
      "The published BLE record is missing from this web build. Reinstall or rebuild the Lab app.",
    );
  }
  const text = await response.text();
  if (
    new TextEncoder().encode(text).byteLength !== GTD1_BLE_GROUND_TRUTH.bytes
  ) {
    throw new Error("The published BLE record failed its byte-count check.");
  }
  if ((await sha256Hex(text)) !== GTD1_BLE_GROUND_TRUTH.sha256) {
    throw new Error("The published BLE record failed its SHA-256 check.");
  }
  return parsePublishedBleGroundTruth(parseJson(text, "BLE record"));
}

export async function sha256Hex(text: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("This browser cannot verify the published result hash.");
  }
  const bytes = new TextEncoder().encode(text);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return bytesToHex(new Uint8Array(digest));
}

export async function loadPublishedDemo(
  signal?: AbortSignal,
): Promise<LoadedPublishedDemo> {
  const [resultResponse, receiptResponse] = await Promise.all([
    fetch(GTD1_PUBLISHED_DEMO.resultUrl, { signal }),
    fetch(GTD1_PUBLISHED_DEMO.receiptUrl, { signal }),
  ]);
  if (!resultResponse.ok) {
    throw new Error(
      "The reconstruction result is missing from this web build. Reinstall or rebuild the Lab app.",
    );
  }
  if (!receiptResponse.ok) {
    throw new Error(
      "The reconstruction receipt is missing from this web build. Reinstall or rebuild the Lab app.",
    );
  }

  const [resultText, receiptText] = await Promise.all([
    resultResponse.text(),
    receiptResponse.text(),
  ]);
  const [resultSha256, receiptSha256] = await Promise.all([
    sha256Hex(resultText),
    sha256Hex(receiptText),
  ]);
  if (resultSha256 !== GTD1_PUBLISHED_DEMO.resultSha256) {
    throw new Error("The published result failed its SHA-256 receipt check.");
  }
  if (receiptSha256 !== GTD1_PUBLISHED_DEMO.receiptSha256) {
    throw new Error("The published receipt failed its SHA-256 manifest check.");
  }

  const result = parsePublishedDecodeResult(parseJson(resultText, "result"));
  const receipt = parsePublishedDecodeReceipt(parseJson(receiptText, "receipt"));
  if (
    receipt.result_sha256 !== resultSha256 ||
    receipt.result_bytes !== new TextEncoder().encode(resultText).byteLength
  ) {
    throw new Error("The published result does not match its write-once receipt.");
  }

  return { result, receipt, resultSha256 };
}

export async function loadPublishedOverlayTrack(
  signal?: AbortSignal,
): Promise<PublishedOverlayTrack> {
  const response = await fetch(GTD1_OVERLAY_TRACK.url, { signal });
  if (!response.ok) {
    throw new Error(
      "The published overlay track is missing from this web build. Reinstall or rebuild the Lab app.",
    );
  }
  const text = await response.text();
  if (new TextEncoder().encode(text).byteLength !== GTD1_OVERLAY_TRACK.bytes) {
    throw new Error("The published overlay track failed its byte-count check.");
  }
  if ((await sha256Hex(text)) !== GTD1_OVERLAY_TRACK.sha256) {
    throw new Error("The published overlay track failed its SHA-256 check.");
  }
  return parsePublishedOverlayTrack(parseJson(text, "overlay track"));
}
