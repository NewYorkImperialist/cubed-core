const CANONICAL_FACE_MOVE = /^[UDLRFB](?:'|2)?$/;
const FACES = ["R", "L", "U", "D", "F", "B"] as const;
const SUFFIXES = ["", "'", "2"] as const;
const RECORDING_SCRAMBLE_STORAGE_KEY =
  "cubed-core-recording-scramble-v1";
const RECORDING_SCRAMBLE_LENGTH = 20;

type RandomIndex = (upperBound: number) => number;

export interface ScrambleStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function randomIndex(upperBound: number): number {
  if (!Number.isInteger(upperBound) || upperBound < 1) {
    throw new RangeError("upperBound must be a positive integer");
  }
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    const rejectionLimit =
      Math.floor(0x1_0000_0000 / upperBound) * upperBound;
    const sample = new Uint32Array(1);
    do {
      crypto.getRandomValues(sample);
    } while (sample[0] >= rejectionLimit);
    return sample[0] % upperBound;
  }
  return Math.floor(Math.random() * upperBound);
}

function checkedIndex(chooseIndex: RandomIndex, upperBound: number): number {
  const value = chooseIndex(upperBound);
  if (!Number.isInteger(value) || value < 0 || value >= upperBound) {
    throw new RangeError(`random index must be between 0 and ${upperBound - 1}`);
  }
  return value;
}

export function scrambleTokens(value: string): string[] {
  return value.trim().split(/\s+/).filter(Boolean);
}

/**
 * Match the workspace's canonical scramble contract before a large video
 * upload begins. The server remains authoritative and repeats this check.
 */
export function normalizeCanonicalScramble(value: string): string {
  const tokens = scrambleTokens(value);
  if (tokens.length === 0) {
    throw new Error("Enter the exact starting scramble.");
  }
  if (tokens.some((token) => !CANONICAL_FACE_MOVE.test(token))) {
    throw new Error(
      "Use canonical face moves such as R, U', or F2, separated by spaces.",
    );
  }
  return tokens.join(" ");
}

/**
 * Generate a local 3x3 random-move scramble without a runtime cube library.
 * Adjacent moves never repeat the same face.
 */
export function generateRecordingScramble(
  length = RECORDING_SCRAMBLE_LENGTH,
  chooseIndex: RandomIndex = randomIndex,
): string {
  if (!Number.isInteger(length) || length < 1 || length > 100) {
    throw new RangeError("scramble length must be an integer between 1 and 100");
  }

  const tokens: string[] = [];
  let previousFace: (typeof FACES)[number] | null = null;
  for (let index = 0; index < length; index += 1) {
    const choices = FACES.filter((face) => face !== previousFace);
    const face = choices[checkedIndex(chooseIndex, choices.length)];
    const suffix = SUFFIXES[checkedIndex(chooseIndex, SUFFIXES.length)];
    tokens.push(`${face}${suffix}`);
    previousFace = face;
  }
  return tokens.join(" ");
}

function validStoredRecordingScramble(value: string | null): string | null {
  if (!value) return null;
  try {
    const normalized = normalizeCanonicalScramble(value);
    return scrambleTokens(normalized).length === RECORDING_SCRAMBLE_LENGTH
      ? normalized
      : null;
  } catch {
    return null;
  }
}

function persistScramble(storage: ScrambleStorage | null, value: string): void {
  if (!storage) return;
  try {
    storage.setItem(RECORDING_SCRAMBLE_STORAGE_KEY, value);
  } catch {
    // The component state keeps the active scramble when storage is blocked.
  }
}

export function loadOrCreateRecordingScramble(
  storage: ScrambleStorage | null,
): string {
  let stored: string | null = null;
  try {
    stored = storage?.getItem(RECORDING_SCRAMBLE_STORAGE_KEY) ?? null;
  } catch {
    // Fall through to a new in-memory scramble.
  }
  const scramble =
    validStoredRecordingScramble(stored) ?? generateRecordingScramble();
  persistScramble(storage, scramble);
  return scramble;
}

export function replaceRecordingScramble(
  storage: ScrambleStorage | null,
): string {
  const scramble = generateRecordingScramble();
  persistScramble(storage, scramble);
  return scramble;
}
