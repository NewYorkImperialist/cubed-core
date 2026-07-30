import {
  LocalContractError,
  parseTrackerDump,
  type TrackerDump,
} from "./localContracts";
import { cubeTrajectory } from "./lib/cubePerms";
import type {
  DecodeGroundTruthDiagnostic,
  DecodeResultDocument,
  DecodeWorkstation,
  DecodeWorkstationRead,
  DecodeWorkstationSequence,
} from "./types";

const SHA256 = /^[0-9a-f]{64}$/;
const RECORDING_ID = /^[0-9a-f]{32}$/;
const POSIX_CKSUM = /^[0-9]+$/;
const CANONICAL_MOVE = /^[UDLRFB](?:'|2)?$/;
const FRAME_KEY = /^(0|[1-9][0-9]*)$/;
const RECONSTRUCTION_COLORS = [
  "white",
  "yellow",
  "red",
  "orange",
  "blue",
  "green",
] as const;
const RECONSTRUCTION_FACE_SLICES = [
  ["up", 0],
  ["right", 9],
  ["front", 18],
  ["down", 27],
  ["left", 36],
  ["back", 45],
] as const;

function objectValue(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new LocalContractError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function strictObject(
  value: unknown,
  label: string,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, unknown> {
  const parsed = objectValue(value, label);
  const allowed = new Set([...required, ...optional]);
  const unknown = Object.keys(parsed).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new LocalContractError(
      `${label} contains unsupported ${unknown.length === 1 ? "field" : "fields"}: ${unknown.join(", ")}.`,
    );
  }
  const missing = required.filter((key) => !hasOwn(parsed, key));
  if (missing.length > 0) {
    throw new LocalContractError(
      `${label} is missing required ${missing.length === 1 ? "field" : "fields"}: ${missing.join(", ")}.`,
    );
  }
  return parsed;
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new LocalContractError(`${label} must be a string.`);
  }
  return value;
}

function boundedString(
  value: unknown,
  label: string,
  minimumLength = 1,
  maximumLength = Number.POSITIVE_INFINITY,
): string {
  const parsed = stringValue(value, label);
  if (
    parsed.length < minimumLength ||
    parsed.length > maximumLength
  ) {
    const maximum =
      Number.isFinite(maximumLength) ? ` and at most ${maximumLength}` : "";
    throw new LocalContractError(
      `${label} must contain at least ${minimumLength}${maximum} characters.`,
    );
  }
  return parsed;
}

function patternString(
  value: unknown,
  label: string,
  pattern: RegExp,
  expectation: string,
): string {
  const parsed = stringValue(value, label);
  if (!pattern.test(parsed)) {
    throw new LocalContractError(`${label} must be ${expectation}.`);
  }
  return parsed;
}

interface NumberBounds {
  minimum?: number;
  exclusiveMinimum?: number;
  maximum?: number;
}

function finiteNumber(
  value: unknown,
  label: string,
  bounds: NumberBounds = {},
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {
    throw new LocalContractError(`${label} must be a finite number.`);
  }
  if (
    (bounds.minimum !== undefined && value < bounds.minimum) ||
    (bounds.exclusiveMinimum !== undefined &&
      value <= bounds.exclusiveMinimum) ||
    (bounds.maximum !== undefined && value > bounds.maximum)
  ) {
    throw new LocalContractError(`${label} is outside the allowed range.`);
  }
  return value;
}

function integer(
  value: unknown,
  label: string,
  bounds: NumberBounds = {},
): number {
  const parsed = finiteNumber(value, label, bounds);
  if (!Number.isInteger(parsed)) {
    throw new LocalContractError(`${label} must be an integer.`);
  }
  return parsed;
}

function arrayValue(
  value: unknown,
  label: string,
  minimumLength = 0,
  maximumLength = Number.POSITIVE_INFINITY,
): unknown[] {
  if (
    !Array.isArray(value) ||
    value.length < minimumLength ||
    value.length > maximumLength
  ) {
    const maximum =
      Number.isFinite(maximumLength) ? ` and at most ${maximumLength}` : "";
    throw new LocalContractError(
      `${label} must be an array with at least ${minimumLength}${maximum} items.`,
    );
  }
  return value;
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new LocalContractError(`${label} must be a boolean.`);
  }
  return value;
}

function canonicalMove(value: unknown, label: string): string {
  return patternString(
    value,
    label,
    CANONICAL_MOVE,
    "a canonical face turn such as R, U', or F2",
  );
}

function validatePoint(value: unknown, label: string): void {
  arrayValue(value, label, 2, 2).forEach((coordinate, index) => {
    finiteNumber(coordinate, `${label}[${index}]`, {
      minimum: 0,
      maximum: 100_000,
    });
  });
}

function validateQuad(value: unknown, label: string): void {
  arrayValue(value, label, 4, 4).forEach((point, index) => {
    validatePoint(point, `${label}[${index}]`);
  });
}

function validateFixedNumbers(
  value: unknown,
  label: string,
  length: number,
  bounds: NumberBounds = {},
): void {
  arrayValue(value, label, length, length).forEach((entry, index) => {
    finiteNumber(entry, `${label}[${index}]`, bounds);
  });
}

function validateFixedIntegers(
  value: unknown,
  label: string,
  length: number,
): void {
  arrayValue(value, label, length, length).forEach((entry, index) => {
    integer(entry, `${label}[${index}]`, { minimum: 0 });
  });
}

function validateFixedBooleans(
  value: unknown,
  label: string,
  length: number,
): void {
  arrayValue(value, label, length, length).forEach((entry, index) => {
    booleanValue(entry, `${label}[${index}]`);
  });
}

function validateFace(value: unknown, label: string): void {
  const face = strictObject(
    value,
    label,
    ["corners", "confidence", "keypoint_confidence"],
  );
  validateQuad(face.corners, `${label}.corners`);
  finiteNumber(face.confidence, `${label}.confidence`, {
    minimum: 0,
    maximum: 1,
  });
  validateFixedNumbers(
    face.keypoint_confidence,
    `${label}.keypoint_confidence`,
    4,
    { minimum: 0, maximum: 1 },
  );
}

function validateRead(value: unknown, label: string): void {
  const read = strictObject(
    value,
    label,
    ["slot", "lab", "confidence", "relative_area", "corners"],
    ["valid_pixels", "total_pixels", "used_fallback"],
  );
  if (read.slot !== "up" && read.slot !== "front" && read.slot !== "right") {
    throw new LocalContractError(`${label}.slot must be up, front, or right.`);
  }
  arrayValue(read.lab, `${label}.lab`, 9, 9).forEach((entry, index) => {
    validateFixedNumbers(entry, `${label}.lab[${index}]`, 3);
  });
  validateFixedNumbers(read.confidence, `${label}.confidence`, 9);
  if (hasOwn(read, "valid_pixels")) {
    validateFixedIntegers(read.valid_pixels, `${label}.valid_pixels`, 9);
  }
  if (hasOwn(read, "total_pixels")) {
    validateFixedIntegers(read.total_pixels, `${label}.total_pixels`, 9);
  }
  if (hasOwn(read, "used_fallback")) {
    validateFixedBooleans(read.used_fallback, `${label}.used_fallback`, 9);
  }
  finiteNumber(read.relative_area, `${label}.relative_area`, { minimum: 0 });
  validateQuad(read.corners, `${label}.corners`);
}

function validateFrame(value: unknown, label: string): void {
  const frame = strictObject(
    value,
    label,
    ["motion"],
    [
      "aligned",
      "aligned_streak",
      "face_count",
      "gated",
      "faces",
      "reads",
    ],
  );
  if (frame.motion !== null) {
    finiteNumber(frame.motion, `${label}.motion`);
  }
  if (hasOwn(frame, "aligned")) {
    finiteNumber(frame.aligned, `${label}.aligned`, {
      minimum: 0,
      maximum: 1,
    });
  }
  for (const key of ["aligned_streak", "face_count"] as const) {
    if (hasOwn(frame, key) && frame[key] !== null) {
      integer(frame[key], `${label}.${key}`, { minimum: 0 });
    }
  }
  if (hasOwn(frame, "gated")) {
    booleanValue(frame.gated, `${label}.gated`);
  }
  if (hasOwn(frame, "faces")) {
    arrayValue(frame.faces, `${label}.faces`, 0, 16).forEach(
      (entry, index) => validateFace(entry, `${label}.faces[${index}]`),
    );
  }
  if (hasOwn(frame, "reads")) {
    arrayValue(frame.reads, `${label}.reads`, 0, 3).forEach(
      (entry, index) => validateRead(entry, `${label}.reads[${index}]`),
    );
  }
}

function validateStringOrStringArray(
  value: unknown,
  label: string,
  stringMaximum: number,
  arrayMaximum: number,
  itemMaximum: number,
): void {
  if (typeof value === "string") {
    boundedString(value, label, 0, stringMaximum);
    return;
  }
  arrayValue(value, label, 0, arrayMaximum).forEach((entry, index) => {
    boundedString(entry, `${label}[${index}]`, 0, itemMaximum);
  });
}

function validateSpan(value: unknown, label: string): void {
  const span = strictObject(value, label, ["f0", "f1", "event", "top"]);
  integer(span.f0, `${label}.f0`, { minimum: 0 });
  integer(span.f1, `${label}.f1`, { minimum: 0 });
  if (span.event !== null) {
    integer(span.event, `${label}.event`, { minimum: 0 });
  }
  arrayValue(span.top, `${label}.top`, 0, 100).forEach((entry, index) => {
    const candidateLabel = `${label}.top[${index}]`;
    const candidate = strictObject(
      entry,
      candidateLabel,
      ["path", "orientation", "score"],
    );
    validateStringOrStringArray(
      candidate.path,
      `${candidateLabel}.path`,
      5_000,
      2_000,
      32,
    );
    validateStringOrStringArray(
      candidate.orientation,
      `${candidateLabel}.orientation`,
      1_000,
      100,
      100,
    );
    finiteNumber(candidate.score, `${candidateLabel}.score`);
  });
}

function validateFacelets(value: unknown, label: string): void {
  arrayValue(value, label, 9, 9).forEach((entry, index) => {
    boundedString(entry, `${label}[${index}]`, 1, 32);
  });
}

function validateReconstructionState(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const state = strictObject(
    value,
    label,
    ["up", "right", "front", "down", "left", "back"],
  );
  for (const face of ["up", "right", "front", "down", "left", "back"]) {
    validateFacelets(state[face], `${label}.${face}`);
  }
  return state;
}

function reconstructionStateMatches(
  actual: Record<string, unknown>,
  expected: Int8Array,
): boolean {
  return RECONSTRUCTION_FACE_SLICES.every(([face, offset]) => {
    const facelets = actual[face] as unknown[];
    return facelets.every(
      (color, index) =>
        color === RECONSTRUCTION_COLORS[expected[offset + index]],
    );
  });
}

function cubeStatesMatch(left: Int8Array, right: Int8Array): boolean {
  return left.every((value, index) => value === right[index]);
}

function validateWorkstation(
  value: unknown,
  moves: readonly string[],
  solvedReached: boolean | null,
  status: "completed" | "abstained",
): DecodeWorkstation {
  const workstation = strictObject(
    value,
    "workstation",
    [
      "schema",
      "schema_version",
      "video",
      "initialization",
      "window",
      "warnings",
      "frames",
    ],
    ["events", "trellis", "sequence", "reconstruction"],
  );
  if (workstation.schema !== "cubed-core/decode-workstation-v1") {
    throw new LocalContractError(
      "workstation.schema must be cubed-core/decode-workstation-v1.",
    );
  }
  if (workstation.schema_version !== 1) {
    throw new LocalContractError("Unsupported decode workstation version.");
  }

  const video = strictObject(
    workstation.video,
    "workstation.video",
    ["sha256", "bytes", "fps", "frame_count", "width", "height", "encoded"],
  );
  patternString(
    video.sha256,
    "workstation.video.sha256",
    SHA256,
    "a lowercase SHA-256 digest",
  );
  integer(video.bytes, "workstation.video.bytes", { minimum: 1 });
  finiteNumber(video.fps, "workstation.video.fps", {
    exclusiveMinimum: 0,
    maximum: 1_000,
  });
  integer(video.frame_count, "workstation.video.frame_count", {
    minimum: 1,
    maximum: 10_000_000,
  });
  integer(video.width, "workstation.video.width", {
    minimum: 1,
    maximum: 100_000,
  });
  integer(video.height, "workstation.video.height", {
    minimum: 1,
    maximum: 100_000,
  });
  const encoded = strictObject(
    video.encoded,
    "workstation.video.encoded",
    ["fps", "frame_count", "width", "height"],
  );
  finiteNumber(encoded.fps, "workstation.video.encoded.fps", {
    exclusiveMinimum: 0,
    maximum: 1_000,
  });
  integer(encoded.frame_count, "workstation.video.encoded.frame_count", {
    minimum: 1,
    maximum: 10_000_000,
  });
  integer(encoded.width, "workstation.video.encoded.width", {
    minimum: 1,
    maximum: 100_000,
  });
  integer(encoded.height, "workstation.video.encoded.height", {
    minimum: 1,
    maximum: 100_000,
  });

  const initialization = strictObject(
    workstation.initialization,
    "workstation.initialization",
    ["scramble"],
  );
  const initializationScramble = boundedString(
    initialization.scramble,
    "workstation.initialization.scramble",
    1,
    4_096,
  );

  const window = arrayValue(
    workstation.window,
    "workstation.window",
    2,
    2,
  ).map((entry, index) =>
      integer(entry, `workstation.window[${index}]`, {
        minimum: 0,
        maximum: 10_000_000,
      }),
  );

  arrayValue(workstation.warnings, "workstation.warnings", 0, 100).forEach(
    (entry, index) => {
      const warningLabel = `workstation.warnings[${index}]`;
      const warning = strictObject(
        entry,
        warningLabel,
        ["code", "message"],
      );
      boundedString(warning.code, `${warningLabel}.code`, 1, 200);
      boundedString(
        warning.message,
        `${warningLabel}.message`,
        1,
        2_000,
      );
    },
  );

  const frames = objectValue(workstation.frames, "workstation.frames");
  const frameEntries = Object.entries(frames);
  if (frameEntries.length > 250_000) {
    throw new LocalContractError(
      "workstation.frames may contain at most 250000 frames.",
    );
  }
  frameEntries.forEach(([key, frame]) => {
    if (!FRAME_KEY.test(key)) {
      throw new LocalContractError(
        `workstation.frames contains invalid frame key ${key}.`,
      );
    }
    validateFrame(frame, `workstation.frames.${key}`);
  });

  if (hasOwn(workstation, "events")) {
    arrayValue(workstation.events, "workstation.events", 0, 100_000).forEach(
      (entry, index) => {
        integer(entry, `workstation.events[${index}]`, {
          minimum: 0,
          maximum: 10_000_000,
        });
      },
    );
  }

  if (hasOwn(workstation, "trellis")) {
    const trellis = strictObject(
      workstation.trellis,
      "workstation.trellis",
      ["spans"],
      ["bridge"],
    );
    arrayValue(trellis.spans, "workstation.trellis.spans", 0, 100_000).forEach(
      (entry, index) => {
        validateSpan(entry, `workstation.trellis.spans[${index}]`);
      },
    );
    if (hasOwn(trellis, "bridge")) {
      arrayValue(trellis.bridge, "workstation.trellis.bridge", 0, 2_000).forEach(
        (entry, index) => {
          canonicalMove(entry, `workstation.trellis.bridge[${index}]`);
        },
      );
    }
  }

  if (hasOwn(workstation, "sequence")) {
    const sequence = strictObject(
      workstation.sequence,
      "workstation.sequence",
      ["moves", "timing_basis"],
    );
    if (sequence.timing_basis !== "canonical") {
      throw new LocalContractError(
        "workstation.sequence.timing_basis must be canonical.",
      );
    }
    const sequenceMoves = arrayValue(
      sequence.moves,
      "workstation.sequence.moves",
      0,
      2_000,
    ).map((entry, index) => {
      const moveLabel = `workstation.sequence.moves[${index}]`;
      const move = strictObject(
        entry,
        moveLabel,
        ["move", "frame"],
      );
      const token = canonicalMove(move.move, `${moveLabel}.move`);
      integer(move.frame, `${moveLabel}.frame`, {
        minimum: 0,
        maximum: 10_000_000,
      });
      return token;
    });
    if (
      sequenceMoves.length !== moves.length ||
      sequenceMoves.some((move, index) => move !== moves[index])
    ) {
      throw new LocalContractError(
        "workstation.sequence.moves must exactly match the top-level moves.",
      );
    }
  }

  if (hasOwn(workstation, "reconstruction")) {
    const reconstruction = strictObject(
      workstation.reconstruction,
      "workstation.reconstruction",
      ["states", "solved_reached"],
      ["timeline"],
    );
    const reconstructionStates = arrayValue(
      reconstruction.states,
      "workstation.reconstruction.states",
      0,
      2_001,
    ).map((entry, index) =>
      validateReconstructionState(
        entry,
        `workstation.reconstruction.states[${index}]`,
      ),
    );
    const reconstructionSolved = booleanValue(
      reconstruction.solved_reached,
      "workstation.reconstruction.solved_reached",
    );
    if (reconstructionSolved !== solvedReached) {
      throw new LocalContractError(
        "workstation.reconstruction.solved_reached must match the top-level endpoint.",
      );
    }
    if (hasOwn(reconstruction, "timeline")) {
      if (status !== "completed") {
        throw new LocalContractError(
          "workstation.reconstruction.timeline requires a completed result.",
        );
      }
      const timeline = strictObject(
        reconstruction.timeline,
        "workstation.reconstruction.timeline",
        ["moves", "timing_basis"],
      );
      if (timeline.timing_basis !== "decoder-checkpoint") {
        throw new LocalContractError(
          "workstation.reconstruction.timeline.timing_basis must be decoder-checkpoint.",
        );
      }
      let previousFrame = -1;
      const timelineMoves = arrayValue(
        timeline.moves,
        "workstation.reconstruction.timeline.moves",
        0,
        2_000,
      );
      const timelineMoveTokens = timelineMoves.map((entry, index) => {
        const moveLabel = `workstation.reconstruction.timeline.moves[${index}]`;
        const move = strictObject(entry, moveLabel, ["move", "frame"]);
        const token = canonicalMove(move.move, `${moveLabel}.move`);
        const frame = integer(move.frame, `${moveLabel}.frame`, {
          minimum: 0,
          maximum: 10_000_000,
        });
        if (index === 0 && frame <= window[0]) {
          throw new LocalContractError(
            "workstation.reconstruction.timeline must place its first checkpoint after workstation.window start.",
          );
        }
        if (frame < window[0] || frame > window[1]) {
          throw new LocalContractError(
            "workstation.reconstruction.timeline.moves must stay within workstation.window.",
          );
        }
        if (frame < previousFrame) {
          throw new LocalContractError(
            "workstation.reconstruction.timeline.moves must be ordered by frame.",
          );
        }
        previousFrame = frame;
        return token;
      });
      if (reconstructionStates.length !== timelineMoves.length + 1) {
        throw new LocalContractError(
          "workstation.reconstruction.states must contain the initial state and one state per timeline move.",
        );
      }
      const scrambleMoves = initializationScramble
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .map((move, index) =>
          canonicalMove(
            move,
            `workstation.initialization.scramble move ${index}`,
          ),
        );
      const replayStates = cubeTrajectory([
        ...scrambleMoves,
        ...timelineMoveTokens,
      ]).slice(scrambleMoves.length);
      if (
        replayStates.length !== reconstructionStates.length ||
        replayStates.some(
          (state, index) =>
            !reconstructionStateMatches(reconstructionStates[index], state),
        )
      ) {
        throw new LocalContractError(
          "workstation.reconstruction.states must exactly replay the decoder-checkpoint timeline.",
        );
      }
      const canonicalStates = cubeTrajectory([
        ...scrambleMoves,
        ...moves,
      ]);
      const canonicalState = canonicalStates[canonicalStates.length - 1];
      const timelineState = replayStates[replayStates.length - 1];
      if (
        !canonicalState ||
        !timelineState ||
        !cubeStatesMatch(timelineState, canonicalState)
      ) {
        throw new LocalContractError(
          "workstation.reconstruction.timeline must end at the top-level decoded state.",
        );
      }
    }
  }

  return workstation as unknown as DecodeWorkstation;
}

function validateGroundTruthDiagnostic(
  value: unknown,
  recordingId: string,
  decodedMoves: string[],
  declaredVideoSha256: string | null,
): DecodeGroundTruthDiagnostic {
  const diagnostic = strictObject(
    value,
    "ground_truth_diagnostic",
    [
      "schema",
      "schema_version",
      "diagnostic_only",
      "capture_id",
      "reference",
      "normalization",
      "counts",
      "comparison",
    ],
    ["frame_timing"],
  );
  if (
    diagnostic.schema !==
    "cubed-core/decode-ground-truth-diagnostic-v1"
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic must declare cubed-core/decode-ground-truth-diagnostic-v1.",
    );
  }
  if (diagnostic.schema_version !== 1) {
    throw new LocalContractError(
      "Unsupported ground-truth diagnostic version.",
    );
  }
  if (diagnostic.diagnostic_only !== true) {
    throw new LocalContractError(
      "ground_truth_diagnostic.diagnostic_only must be true.",
    );
  }
  const captureId = patternString(
    diagnostic.capture_id,
    "ground_truth_diagnostic.capture_id",
    RECORDING_ID,
    "exactly 32 lowercase hexadecimal characters",
  );
  if (captureId !== recordingId) {
    throw new LocalContractError(
      "ground_truth_diagnostic.capture_id must match recording_id.",
    );
  }

  const reference = strictObject(
    diagnostic.reference,
    "ground_truth_diagnostic.reference",
    [
      "kind",
      "scope",
      "dataset_id",
      "revision",
      "bootstrap_manifest_sha256",
      "download_receipt_sha256",
      "corpus_manifest_sha256",
      "video_sha256",
      "scramble_sha256",
      "ble_sha256",
      "video_link_status",
      "video_recording_id_verified",
    ],
    ["frame_ground_truth_sha256"],
  );
  if (reference.kind !== "published-smart-cube-ble") {
    throw new LocalContractError(
      "ground_truth_diagnostic.reference.kind must be published-smart-cube-ble.",
    );
  }
  if (
    reference.scope !== "sequence-only" &&
    reference.scope !== "sequence-and-clip-frame-indexed"
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic.reference.scope is unsupported.",
    );
  }
  boundedString(
    reference.dataset_id,
    "ground_truth_diagnostic.reference.dataset_id",
    1,
    128,
  );
  boundedString(
    reference.revision,
    "ground_truth_diagnostic.reference.revision",
    1,
    128,
  );
  for (const field of [
    "bootstrap_manifest_sha256",
    "download_receipt_sha256",
    "corpus_manifest_sha256",
    "video_sha256",
    "scramble_sha256",
    "ble_sha256",
  ] as const) {
    patternString(
      reference[field],
      `ground_truth_diagnostic.reference.${field}`,
      SHA256,
      "a lowercase SHA-256 digest",
    );
  }
  if (hasOwn(reference, "frame_ground_truth_sha256")) {
    patternString(
      reference.frame_ground_truth_sha256,
      "ground_truth_diagnostic.reference.frame_ground_truth_sha256",
      SHA256,
      "a lowercase SHA-256 digest",
    );
  }
  if (
    reference.video_link_status !== "linked" &&
    reference.video_link_status !== "failed_timeout"
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic.reference.video_link_status is unsupported.",
    );
  }
  if (reference.video_recording_id_verified !== true) {
    throw new LocalContractError(
      "ground_truth_diagnostic.reference.video_recording_id_verified must be true.",
    );
  }
  if (
    declaredVideoSha256 &&
    reference.video_sha256 !== declaredVideoSha256
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic.reference.video_sha256 must match the decoded video.",
    );
  }

  const normalization = strictObject(
    diagnostic.normalization,
    "ground_truth_diagnostic.normalization",
    ["raw_metric", "comparison_metric", "method"],
  );
  if (
    normalization.raw_metric !== "quarter-turn" ||
    normalization.comparison_metric !== "half-turn" ||
    normalization.method !== "adjacent-same-face-mod-4"
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic.normalization is unsupported.",
    );
  }

  const counts = strictObject(
    diagnostic.counts,
    "ground_truth_diagnostic.counts",
    ["decoded_htm", "ble_raw_qtm", "ble_canonical_htm"],
  );
  const decodedCount = integer(
    counts.decoded_htm,
    "ground_truth_diagnostic.counts.decoded_htm",
    { minimum: 0, maximum: 400 },
  );
  integer(
    counts.ble_raw_qtm,
    "ground_truth_diagnostic.counts.ble_raw_qtm",
    { minimum: 1, maximum: 400 },
  );
  const referenceCount = integer(
    counts.ble_canonical_htm,
    "ground_truth_diagnostic.counts.ble_canonical_htm",
    { minimum: 0, maximum: 400 },
  );
  if (decodedCount !== decodedMoves.length) {
    throw new LocalContractError(
      "ground_truth_diagnostic.counts.decoded_htm must match moves.",
    );
  }

  const comparison = strictObject(
    diagnostic.comparison,
    "ground_truth_diagnostic.comparison",
    ["distance", "ops"],
  );
  const distance = integer(
    comparison.distance,
    "ground_truth_diagnostic.comparison.distance",
    { minimum: 0, maximum: 800 },
  );
  const alignedDecoded: string[] = [];
  const alignedReference: string[] = [];
  let editCount = 0;
  arrayValue(
    comparison.ops,
    "ground_truth_diagnostic.comparison.ops",
    0,
    800,
  ).forEach((entry, index) => {
    const label = `ground_truth_diagnostic.comparison.ops[${index}]`;
    const operation = strictObject(entry, label, [
      "op",
      "decoded",
      "reference",
      "index_decoded",
      "index_reference",
    ]);
    if (
      operation.op !== "equal" &&
      operation.op !== "substitute" &&
      operation.op !== "insert" &&
      operation.op !== "delete"
    ) {
      throw new LocalContractError(`${label}.op is unsupported.`);
    }
    const decoded =
      operation.decoded === null
        ? null
        : canonicalMove(operation.decoded, `${label}.decoded`);
    const expected =
      operation.reference === null
        ? null
        : canonicalMove(operation.reference, `${label}.reference`);
    const decodedIndex =
      operation.index_decoded === null
        ? null
        : integer(operation.index_decoded, `${label}.index_decoded`, {
            minimum: 0,
            maximum: 399,
          });
    const referenceIndex =
      operation.index_reference === null
        ? null
        : integer(operation.index_reference, `${label}.index_reference`, {
            minimum: 0,
            maximum: 399,
          });
    const bothPresent =
      decoded !== null &&
      expected !== null &&
      decodedIndex !== null &&
      referenceIndex !== null;
    if (
      (operation.op === "equal" || operation.op === "substitute") &&
      !bothPresent
    ) {
      throw new LocalContractError(
        `${label} must carry both decoded and reference moves.`,
      );
    }
    if (
      operation.op === "insert" &&
      !(
        decoded !== null &&
        decodedIndex !== null &&
        expected === null &&
        referenceIndex === null
      )
    ) {
      throw new LocalContractError(
        `${label} insert must carry only a decoded move.`,
      );
    }
    if (
      operation.op === "delete" &&
      !(
        decoded === null &&
        decodedIndex === null &&
        expected !== null &&
        referenceIndex !== null
      )
    ) {
      throw new LocalContractError(
        `${label} delete must carry only a reference move.`,
      );
    }
    if (operation.op === "equal" && decoded !== expected) {
      throw new LocalContractError(
        `${label} equal must carry matching moves.`,
      );
    }
    if (operation.op === "substitute" && decoded === expected) {
      throw new LocalContractError(
        `${label} substitute must carry different moves.`,
      );
    }
    if (
      decoded !== null &&
      decodedIndex !== alignedDecoded.length
    ) {
      throw new LocalContractError(
        `${label}.index_decoded must follow the decoded sequence.`,
      );
    }
    if (
      expected !== null &&
      referenceIndex !== alignedReference.length
    ) {
      throw new LocalContractError(
        `${label}.index_reference must follow the BLE sequence.`,
      );
    }
    if (operation.op !== "equal") editCount += 1;
    if (decoded !== null) alignedDecoded.push(decoded);
    if (expected !== null) alignedReference.push(expected);
  });
  if (distance !== editCount) {
    throw new LocalContractError(
      "ground_truth_diagnostic.comparison.distance must match its edit operations.",
    );
  }
  if (
    alignedDecoded.length !== decodedMoves.length ||
    alignedDecoded.some((move, index) => move !== decodedMoves[index])
  ) {
    throw new LocalContractError(
      "ground_truth_diagnostic.comparison.ops must reconstruct moves.",
    );
  }
  if (alignedReference.length !== referenceCount) {
    throw new LocalContractError(
      "ground_truth_diagnostic.comparison.ops must match the BLE move count.",
    );
  }

  if (reference.scope === "sequence-and-clip-frame-indexed") {
    if (
      !hasOwn(reference, "frame_ground_truth_sha256") ||
      !hasOwn(diagnostic, "frame_timing")
    ) {
      throw new LocalContractError(
        "Frame-indexed BLE reference data requires frame timing.",
      );
    }
    const frameTiming = strictObject(
      diagnostic.frame_timing,
      "ground_truth_diagnostic.frame_timing",
      ["available", "basis", "source_schema", "source_sha256"],
    );
    if (
      frameTiming.available !== true ||
      frameTiming.basis !== "clip-local" ||
      frameTiming.source_schema !== "cubed-core/clip-ble-ground-truth-v1"
    ) {
      throw new LocalContractError(
        "ground_truth_diagnostic.frame_timing is unsupported.",
      );
    }
    const frameSourceSha256 = patternString(
      frameTiming.source_sha256,
      "ground_truth_diagnostic.frame_timing.source_sha256",
      SHA256,
      "a lowercase SHA-256 digest",
    );
    if (
      frameSourceSha256 !== reference.frame_ground_truth_sha256
    ) {
      throw new LocalContractError(
        "ground_truth_diagnostic.frame_timing must match its frame reference.",
      );
    }
  } else if (
    hasOwn(reference, "frame_ground_truth_sha256") ||
    hasOwn(diagnostic, "frame_timing")
  ) {
    throw new LocalContractError(
      "Sequence-only BLE reference data may not claim frame timing.",
    );
  }

  return diagnostic as unknown as DecodeGroundTruthDiagnostic;
}

export function parseDecodeResultDocument(
  value: unknown,
): DecodeResultDocument {
  const declaration = objectValue(value, "run JSON");
  if (declaration.schema !== "cubed-core/decode-result") {
    throw new LocalContractError(
      "Run JSON must declare cubed-core/decode-result.",
    );
  }
  if (declaration.schema_version !== 1) {
    throw new LocalContractError("Unsupported decode result version.");
  }
  const root = strictObject(
    value,
    "run JSON",
    [
      "schema",
      "schema_version",
      "recording_id",
      "status",
      "profile",
      "config",
      "inputs",
      "moves",
      "endpoint",
      "evaluation",
      "provenance",
    ],
    ["workstation", "ground_truth_diagnostic"],
  );
  const recordingId = patternString(
    root.recording_id,
    "recording_id",
    RECORDING_ID,
    "exactly 32 lowercase hexadecimal characters",
  );
  if (root.status !== "completed" && root.status !== "abstained") {
    throw new LocalContractError(
      "Run JSON status must be completed or abstained.",
    );
  }
  if (root.profile !== "local_camera_v1") {
    throw new LocalContractError(
      "Run JSON profile must be local_camera_v1.",
    );
  }

  const config = strictObject(
    root.config,
    "config",
    ["name", "cfg_hash", "cfg_hash_algorithm"],
    ["cfg_hash_input", "extras_hash"],
  );
  boundedString(config.name, "config.name");
  patternString(
    config.cfg_hash,
    "config.cfg_hash",
    POSIX_CKSUM,
    "a decimal POSIX cksum value",
  );
  if (config.cfg_hash_algorithm !== "posix-cksum") {
    throw new LocalContractError(
      "config.cfg_hash_algorithm must be posix-cksum.",
    );
  }
  if (hasOwn(config, "cfg_hash_input")) {
    boundedString(config.cfg_hash_input, "config.cfg_hash_input");
  }
  if (hasOwn(config, "extras_hash") && config.extras_hash !== null) {
    patternString(
      config.extras_hash,
      "config.extras_hash",
      POSIX_CKSUM,
      "a decimal POSIX cksum value or null",
    );
  }

  const inputs = arrayValue(root.inputs, "inputs", 2).map((entry, index) => {
    const inputLabel = `inputs[${index}]`;
    const input = strictObject(entry, inputLabel, ["id", "sha256"]);
    const id = boundedString(input.id, `${inputLabel}.id`);
    const sha256 = patternString(
      input.sha256,
      `${inputLabel}.sha256`,
      SHA256,
      "a lowercase SHA-256 digest",
    );
    return { id, sha256 };
  });

  const moves = arrayValue(root.moves, "moves").map((move, index) =>
    canonicalMove(move, `moves[${index}]`),
  );
  const endpoint = strictObject(root.endpoint, "endpoint", ["solved_reached"]);
  if (
    endpoint.solved_reached !== true &&
    endpoint.solved_reached !== false &&
    endpoint.solved_reached !== null
  ) {
    throw new LocalContractError(
      "endpoint.solved_reached must be true, false, or null.",
    );
  }
  if (root.status === "completed") {
    if (moves.length === 0 || endpoint.solved_reached !== true) {
      throw new LocalContractError(
        "A completed result must carry at least one canonical move and reach the solved endpoint.",
      );
    }
  } else if (
    moves.length !== 0 ||
    (endpoint.solved_reached !== false && endpoint.solved_reached !== null)
  ) {
    throw new LocalContractError(
      "An abstained result must carry no moves and must not claim a solved endpoint.",
    );
  }

  if (root.evaluation !== null) {
    throw new LocalContractError(
      "A run result may not contain an evaluation block.",
    );
  }
  const provenance = strictObject(
    root.provenance,
    "provenance",
    ["runtime_version", "finished_at"],
    [
      "implementation_id",
      "implementation_sha256",
      "python_version",
      "numpy_version",
    ],
  );
  boundedString(provenance.runtime_version, "provenance.runtime_version");
  stringValue(provenance.finished_at, "provenance.finished_at");
  for (const field of [
    "implementation_id",
    "python_version",
    "numpy_version",
  ] as const) {
    if (hasOwn(provenance, field)) {
      boundedString(provenance[field], `provenance.${field}`);
    }
  }
  if (hasOwn(provenance, "implementation_sha256")) {
    patternString(
      provenance.implementation_sha256,
      "provenance.implementation_sha256",
      SHA256,
      "a lowercase SHA-256 digest",
    );
  }

  const workstation =
    !hasOwn(root, "workstation")
      ? undefined
      : validateWorkstation(
          root.workstation,
          moves,
          endpoint.solved_reached,
          root.status,
        );
  const declaredVideoSha256 =
    workstation?.video.sha256 ??
    inputs.find((entry) =>
      /(^|[-_.])video($|[-_.])/i.test(entry.id),
    )?.sha256 ??
    null;
  const groundTruthDiagnostic =
    !hasOwn(root, "ground_truth_diagnostic")
      ? undefined
      : validateGroundTruthDiagnostic(
          root.ground_truth_diagnostic,
          recordingId,
          moves,
          declaredVideoSha256,
        );

  return {
    ...(root as unknown as DecodeResultDocument),
    ...(groundTruthDiagnostic
      ? { ground_truth_diagnostic: groundTruthDiagnostic }
      : {}),
    ...(workstation ? { workstation } : {}),
  };
}

function displayRead(read: DecodeWorkstationRead) {
  const totalPixels =
    read.total_pixels ??
    (Array.from({ length: 9 }, () => 1_000) as DecodeWorkstationRead["total_pixels"]);
  const validPixels =
    read.valid_pixels ??
    (read.confidence.map((value) =>
      Math.round(value * 1_000),
    ) as DecodeWorkstationRead["valid_pixels"]);
  const usedFallback =
    read.used_fallback ??
    (Array.from({ length: 9 }, () => false) as DecodeWorkstationRead["used_fallback"]);
  return {
    slot: read.slot,
    lab: read.lab,
    confidence: read.confidence,
    valid_pixels: validPixels,
    total_pixels: totalPixels,
    used_fallback: usedFallback,
    relative_area: read.relative_area,
    corners: read.corners,
  };
}

export function canonicalDecodeSequence(
  result: DecodeResultDocument,
): DecodeWorkstationSequence | null {
  const sequence = result.workstation?.sequence;
  if (!sequence || sequence.moves.length !== result.moves.length) return null;
  return sequence.moves.every(
    (entry, index) => entry.move === result.moves[index],
  )
    ? sequence
    : null;
}

export function decodeWorkstationAsTrackerDump(
  result: DecodeResultDocument,
): TrackerDump | null {
  const workstation = result.workstation;
  if (!workstation) return null;
  const sequence = canonicalDecodeSequence(result)?.moves ?? [];
  return parseTrackerDump({
    schema: "cubed-core/tracker-dump-v1",
    schema_version: 1,
    mode: "decode-workstation",
    fps: workstation.video.encoded.fps,
    window: workstation.window,
    width: workstation.video.width,
    height: workstation.video.height,
    gt: [],
    emitted: result.moves,
    emitted_canonical: result.moves,
    anchors: sequence.map(({ frame, move }) => [
      frame,
      move,
      frame / workstation.video.encoded.fps,
    ]),
    events: workstation.events ?? [],
    spans:
      workstation.trellis?.spans.map((span) => ({
        f0: span.f0,
        f1: span.f1,
        event: span.event,
        top: span.top.map((candidate) => ({
          path: Array.isArray(candidate.path)
            ? candidate.path.join(" ")
            : candidate.path,
          om: candidate.orientation,
          score: candidate.score,
        })),
      })) ?? [],
    bridge: workstation.trellis?.bridge,
    frames: Object.fromEntries(
      Object.entries(workstation.frames).map(([frame, value]) => [
        frame,
        {
          motion: value.motion,
          ...(value.aligned === undefined ? {} : { aligned: value.aligned }),
          ...(value.aligned_streak === undefined
            ? {}
            : { stk: value.aligned_streak }),
          ...(value.face_count === undefined
            ? {}
            : { nfaces: value.face_count }),
          ...(value.gated === undefined ? {} : { gated: value.gated }),
          ...(value.faces
            ? {
                faces: value.faces.map((face) => ({
                  corners: face.corners,
                  conf: face.confidence,
                  kpt_conf: face.keypoint_confidence,
                })),
              }
            : {}),
          ...(value.reads
            ? { reads: value.reads.map(displayRead) }
            : {}),
        },
      ]),
    ),
  });
}

export function decodeResultVideoSha256(
  result: DecodeResultDocument,
): string | null {
  if (result.workstation?.video.sha256) {
    return result.workstation.video.sha256;
  }
  return (
    result.inputs.find((entry) =>
      /(^|[-_.])video($|[-_.])/i.test(entry.id),
    )?.sha256 ?? null
  );
}
