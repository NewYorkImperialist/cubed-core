import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  generateRecordingScramble,
  loadOrCreateRecordingScramble,
  normalizeCanonicalScramble,
  replaceRecordingScramble,
  scrambleTokens,
} from "../src/importWorkflow.ts";

const sourceRoot = resolve(import.meta.dirname, "../src");
const appSource = readFileSync(resolve(sourceRoot, "App.tsx"), "utf8");
const importSource = readFileSync(
  resolve(sourceRoot, "components/VideoImportWorkbench.tsx"),
  "utf8",
);
const prepareSource = readFileSync(
  resolve(sourceRoot, "components/PrepareForDecodeCard.tsx"),
  "utf8",
);
const apiSource = readFileSync(resolve(sourceRoot, "api.ts"), "utf8");

function cyclingIndex() {
  let value = 0;
  return (upperBound) => {
    const selected = value % upperBound;
    value += 1;
    return selected;
  };
}

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    values,
  };
}

test("the launch surface removes Capture and phone pairing in favor of Add video", () => {
  assert.match(appSource, /path="\/import" element=\{<VideoImportWorkbench \/>}/);
  assert.match(
    appSource,
    /path="\/phone" element=\{<Navigate to="\/import" replace \/>}/,
  );
  assert.match(
    appSource,
    /path="\/capture\/:code" element=\{<Navigate to="\/import" replace \/>}/,
  );
  assert.doesNotMatch(appSource, /PhoneCapturePage|BleCapturePanel|QRCodeSVG/);
});

test("the recommended flow generates a stable canonical recording scramble", () => {
  const scramble = generateRecordingScramble(20, cyclingIndex());
  const tokens = scrambleTokens(scramble);

  assert.equal(tokens.length, 20);
  for (let index = 1; index < tokens.length; index += 1) {
    assert.notEqual(tokens[index][0], tokens[index - 1][0]);
  }
  assert.equal(
    generateRecordingScramble(4, () => 0),
    "R L R L",
  );
  assert.throws(() => generateRecordingScramble(0), /between 1 and 100/);
});

test("the active recording scramble survives rerenders and explicit replacement", () => {
  const storage = memoryStorage();
  const first = loadOrCreateRecordingScramble(storage);
  const second = loadOrCreateRecordingScramble(storage);
  const replacement = replaceRecordingScramble(storage);

  assert.equal(second, first);
  assert.equal(scrambleTokens(first).length, 20);
  assert.equal(scrambleTokens(replacement).length, 20);
  assert.equal([...storage.values.values()][0], replacement);
});

test("Record a new solve carries its displayed scramble into the workspace", () => {
  assert.match(importSource, /loadOrCreateRecordingScramble/);
  assert.match(importSource, /className="import-scramble-tape"/);
  assert.match(importSource, /record the full solve in your normal camera\s+app/);
  assert.match(
    importSource,
    /This recording starts from the displayed scramble/,
  );
  assert.match(
    importSource,
    /upload\("record", primaryFile, recordingScramble\)/,
  );
  assert.match(
    importSource,
    /importCapture\(file, "", normalizedScramble, ""\)/,
  );
  assert.doesNotMatch(importSource, /cubing\/scramble|controlled-decode/);
});

test("New scramble cannot leave a stale recording attached", () => {
  assert.match(importSource, /replaceRecordingScramble/);
  assert.match(
    importSource,
    /setPrimaryFile\(null\);\s+setPrimaryConfirmed\(false\);/,
  );
  assert.match(
    importSource,
    /primaryFileInputRef\.current\.value = ""/,
  );
});

test("an existing recording remains a secondary exact-scramble path", () => {
  assert.match(importSource, /<details className="import-existing-panel">/);
  assert.match(importSource, /Use an existing recording/);
  assert.match(importSource, /id="import-existing-scramble"/);
  assert.match(
    importSource,
    /upload\("existing", existingFile, existingScramble\)/,
  );
  assert.equal(
    normalizeCanonicalScramble("  R   U'  F2 "),
    "R U' F2",
  );
  assert.throws(
    () => normalizeCanonicalScramble("R x"),
    /canonical face moves/,
  );
  assert.throws(
    () => normalizeCanonicalScramble(" "),
    /exact starting scramble/,
  );
});

test("the main import flow states only the three Decode inputs", () => {
  assert.doesNotMatch(
    importSource,
    /attachCaptureSidecar|sidecars\/teacher|sidecars\/ble-raw|sidecars\/phone-imu/,
  );
  assert.match(
    importSource,
    /exact starting\s+scramble and a color calibration/,
  );
  assert.doesNotMatch(importSource, /iPhone|BLE|IMU|smart cube/);
});

test("every imported video can be sampled or use shared calibration with a warning", () => {
  assert.match(prepareSource, /<ColorCalibrationSampler/);
  assert.match(prepareSource, /This shared calibration is allowed for any video/);
  assert.doesNotMatch(
    prepareSource,
    /disabled=\{[^}]*calibrationSource === BUNDLED_SOURCE/,
  );
});

test("one imported workspace item continues to Decode or Label", () => {
  assert.match(
    importSource,
    /captureHref\("\/decode", activeCapture\.capture_id\)/,
  );
  assert.match(
    importSource,
    /captureHref\("\/label", activeCapture\.capture_id\)/,
  );
  assert.match(
    importSource,
    /<PrepareForDecodeCard\s+capture=\{activeCapture\}\s+captures=\{captures\}\s+onUpdated=\{updateCapture\}/,
  );
});

test("large-file checks use one FPS-independent video limit", () => {
  assert.match(importSource, /upload_limits\?\.video_bytes/);
  assert.match(importSource, /Original video, up to/);
  assert.doesNotMatch(
    importSource,
    /standard_bytes|native_240_fps_bytes|DEFAULT_HIGH_SPEED/,
  );
  assert.match(
    importSource,
    /video\/\*,\.mov,\.mp4,\.m4v,\.avi,\.mkv,\.webm/,
  );
});

test("a successful upload clears both native pickers for same-file retries", () => {
  assert.match(
    importSource,
    /primaryFileInputRef\.current\) primaryFileInputRef\.current\.value = "";/,
  );
  assert.match(
    importSource,
    /existingFileInputRef\.current\) existingFileInputRef\.current\.value = "";/,
  );
});

test("Add video keeps system-Trash removal compact and out of task pages", () => {
  assert.match(
    importSource,
    /<details className="import-workspace-manager">[\s\S]*?Manage workspace videos/,
  );
  assert.match(importSource, /orderDecodeCaptures\(captures\)/);
  assert.match(
    importSource,
    /captureGroups\.yourRecordings\.map[\s\S]*?captureGroups\.publishedDataset\.map/,
  );
  assert.match(
    importSource,
    /Remove "\$\{capture\.original_filename\}" from this workspace\?/,
  );
  assert.match(importSource, /The video and labels move to your system Trash/);
  assert.match(importSource, /Saved Runs stay, but video playback will be unavailable/);
  assert.match(importSource, /isPublishedDatasetCapture\(capture\)/);
  assert.match(
    importSource,
    /The published copy is untouched and can be downloaded again/,
  );
  assert.match(importSource, /Move to Trash/);
  assert.match(importSource, /disabled=\{removingCaptureId !== ""\}/);
  assert.match(importSource, /setActiveCapture\(null\)/);
  assert.match(importSource, /nextParams\.delete\(CAPTURE_PARAM\)/);
  assert.match(importSource, /await loadCaptures\(\)/);
  assert.match(importSource, /role="alert"/);
  assert.match(
    importSource,
    /import-workspace-manager-body[\s\S]*?\{removeMessage &&[\s\S]*?\{removeError &&[\s\S]*?\{captures\.length === 0/,
  );
  assert.doesNotMatch(prepareSource, /deleteCapture|Remove workspace video/);
});

test("capture removal uses the authenticated workspace API route", () => {
  assert.match(
    apiSource,
    /export interface CaptureDeleteReceipt[\s\S]*?schema: "cubed-core\/capture-delete-v1"/,
  );
  assert.match(
    apiSource,
    /export function deleteCapture\([\s\S]*?`\/api\/captures\/\$\{encodeURIComponent\(captureId\)\}`[\s\S]*?method: "DELETE"/,
  );
});
