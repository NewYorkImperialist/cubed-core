import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import { simplifyMovesWithFrames } from "../src/moveFold.ts";
import {
  bleTimedMoves,
  frameIndexForTime,
  GTD1_BLE_GROUND_TRUTH,
  GTD1_OVERLAY_TRACK,
  GTD1_PUBLISHED_DEMO,
  GTD1_RELEASE_TAG,
  GTD1_VIDEO_TIMING,
  GTD1_VIDEO_URL,
  moveStepForFrame,
  parsePublishedBleGroundTruth,
  parsePublishedDecodeReceipt,
  parsePublishedDecodeResult,
  parsePublishedOverlayTrack,
  timeForFrameIndex,
} from "../src/publishedDemo.ts";

const appRoot = resolve(import.meta.dirname, "..");
const resultPath = resolve(appRoot, "public/demo/gtd1/decode-result.json");
const receiptPath = resolve(appRoot, "public/demo/gtd1/decode-receipt.json");
const overlayPath = resolve(appRoot, "public/demo/gtd1/overlay-track.json");
const resultBytes = readFileSync(resultPath);
const receiptBytes = readFileSync(receiptPath);
const overlayBytes = readFileSync(overlayPath);
const blePath = resolve(appRoot, "public/demo/gtd1/ble-ground-truth.json");
const bleBytes = readFileSync(blePath);
const result = JSON.parse(resultBytes.toString("utf8"));
const receipt = JSON.parse(receiptBytes.toString("utf8"));
const overlay = JSON.parse(overlayBytes.toString("utf8"));
const ble = JSON.parse(bleBytes.toString("utf8"));
const demoPageSource = readFileSync(
  resolve(appRoot, "src/components/PublishedDemoPage.tsx"),
  "utf8",
);
const manifestSource = readFileSync(
  resolve(appRoot, "src/publishedDemo.ts"),
  "utf8",
);
const demoPageStyles = readFileSync(
  resolve(appRoot, "src/components/PublishedDemoPage.css"),
  "utf8",
);

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

test("the published result bytes match the write-once receipt", () => {
  assert.equal(resultBytes.byteLength, receipt.result_bytes);
  assert.equal(sha256(resultBytes), receipt.result_sha256);
  assert.equal(receipt.result_sha256, GTD1_PUBLISHED_DEMO.resultSha256);
  assert.equal(sha256(receiptBytes), GTD1_PUBLISHED_DEMO.receiptSha256);
});

test("the published manifest stays bound to the reviewed demo and replay", () => {
  assert.equal(
    GTD1_PUBLISHED_DEMO.video.filename,
    "gtD1s_decoder-demo-clip_f3919-f9722.mp4",
  );
  assert.equal(
    GTD1_PUBLISHED_DEMO.video.sha256,
    "7329c9b3c46007f618b043a61ee00b34c56c74aa22b8a803270c6ea4c9013776",
  );
  assert.equal(result.recording_id, receipt.capture_id);
  assert.equal(result.profile, GTD1_PUBLISHED_DEMO.profile);
  assert.equal(result.config.cfg_hash, GTD1_PUBLISHED_DEMO.cfgHash);
  assert.equal(result.moves.length, GTD1_PUBLISHED_DEMO.moveCount);
  assert.equal(result.endpoint.solved_reached, true);
  assert.equal(receipt.replay_check.performed, true);
  assert.equal(receipt.replay_check.solved_reached, true);
  assert.equal(receipt.replay_check.move_count, result.moves.length);
  assert.doesNotThrow(() => parsePublishedDecodeResult(result));
  assert.doesNotThrow(() => parsePublishedDecodeReceipt(receipt));
});

test("the published result retains its recorded implementation identity", () => {
  assert.equal(
    result.provenance.implementation_sha256,
    GTD1_PUBLISHED_DEMO.implementationSha256,
  );
  assert.equal(
    result.provenance.runtime_version,
    GTD1_PUBLISHED_DEMO.recordedRuntimeVersion,
  );
  assert.equal(
    receipt.runner_provenance.identity_status,
    "unverified-external-identity",
  );
});

test("the published-demo claim copy excludes unsupported conclusions", () => {
  assert.equal(
    GTD1_PUBLISHED_DEMO.claims.scope,
    "Published reconstruction result and endpoint replay receipt.",
  );
  assert.equal(
    GTD1_PUBLISHED_DEMO.claims.replay,
    "The sequence replayed to solved.",
  );
  assert.deepEqual(GTD1_PUBLISHED_DEMO.claims.exclusions, [
    "Not live inference.",
    "Not per-move accuracy.",
    "Not reach-LL evidence.",
    "Not generalization evidence.",
  ]);
  assert.equal(
    GTD1_PUBLISHED_DEMO.claims.runnerIdentity,
    "unverified-external-identity",
  );
});

test("the committed overlay track matches the hash the loader checks", () => {
  assert.equal(overlayBytes.byteLength, GTD1_OVERLAY_TRACK.bytes);
  assert.equal(sha256(overlayBytes), GTD1_OVERLAY_TRACK.sha256);
  assert.equal(overlay.capture_id, GTD1_OVERLAY_TRACK.captureId);
  assert.equal(overlay.capture_id, result.recording_id);
  assert.equal(overlay.source_sha256, GTD1_OVERLAY_TRACK.sourceSha256);
  assert.equal(overlay.mode, "camera-only-native-vision-v1");
  assert.equal(overlay.frame_count, GTD1_VIDEO_TIMING.frameCount);
  assert.equal(overlay.detected_frames, GTD1_OVERLAY_TRACK.detectedFrames);
  assert.equal(overlay.detected_faces, GTD1_OVERLAY_TRACK.detectedFaces);
  assert.doesNotThrow(() => parsePublishedOverlayTrack(overlay));
});

test("every overlay frame carries normalized quads and sampled colors", () => {
  const keys = Object.keys(overlay.frames);
  assert.equal(keys.length, GTD1_OVERLAY_TRACK.detectedFrames);

  let faces = 0;
  for (const key of keys) {
    assert.match(key, /^\d+$/);
    const index = Number(key);
    assert.ok(index >= 0 && index <= GTD1_VIDEO_TIMING.lastFrame);
    const frame = overlay.frames[key];
    assert.ok(Array.isArray(frame.f) && frame.f.length > 0);
    assert.equal(typeof frame.a, "number");
    assert.equal(typeof frame.t, "number");
    for (const face of frame.f) {
      faces += 1;
      assert.equal(face.q.length, 4);
      for (const [x, y] of face.q) {
        assert.ok(x >= -0.5 && x <= 1.5, `x ${x}`);
        assert.ok(y >= -0.5 && y <= 1.5, `y ${y}`);
      }
      assert.equal(typeof face.c, "number");
      assert.match(face.k, /^[0-9a-f]{54}$/);
      assert.equal(face.v.length, 9);
      assert.ok(["up", "front", "right", "down", "left", "back"].includes(face.s));
    }
  }
  assert.equal(faces, GTD1_OVERLAY_TRACK.detectedFaces);
});

test("the overlay never claims the sampled colors are decoded stickers", () => {
  const forbidden = /classified sticker|decoded sticker|sticker color(?:s)? read/i;
  assert.doesNotMatch(demoPageSource.replace(/no color decision/g, ""), forbidden);
  assert.match(demoPageSource, /sampled-color-not-a-classified-sticker/);
  assert.match(demoPageSource, /classifies a value, snaps it to a palette/);
});

test("the per-frame reads render sampled color without classifying it", () => {
  assert.equal(overlay.color_meaning, "sampled-color-not-a-classified-sticker");

  const reads =
    /function SampledReads\(\{[\s\S]*?(?=\nexport function PublishedDemoPage)/.exec(
      demoPageSource,
    );
  assert.ok(reads, "the sampled reads panel was not found");

  // Cell colour is the recorded sRGB, opacity is the recorded per-facelet
  // confidence, and the label is the recorded slot.
  assert.match(reads[0], /face\.k!\.slice\(cell \* 6, cell \* 6 \+ 6\)/);
  assert.match(reads[0], /0\.45 \+ 0\.55 \* Math\.min\(1, confidence\)/);
  assert.match(reads[0], /background: hex/);
  assert.match(reads[0], /title=\{`sampled \$\{hex\}/);

  // A fixed slot set in a fixed order. The tracker only ever assigns these
  // three on this clip, and a slot must not move when a detection drops.
  assert.match(
    demoPageSource,
    /const READ_SLOTS = \["up", "front", "right"\] as const;/,
  );
  assert.deepEqual(
    [...new Set(
      Object.values(overlay.frames).flatMap((entry) =>
        entry.f.map((face) => face.s),
      ),
    )].sort(),
    ["front", "right", "up"],
  );
  assert.match(reads[0], /READ_SLOTS\.map\(\(slot\) =>/);

  // Nothing may resize between frames: an undetected slot draws an empty grid.
  // The redundant sampled-face count is intentionally omitted.
  assert.match(reads[0], /demo-read-empty/);
  assert.doesNotMatch(reads[0], /demo-reads-status|faces sampled/);
  assert.doesNotMatch(demoPageStyles, /\.demo-reads-status \{/);

  // No palette lookup and no classification may appear anywhere in the block.
  assert.doesNotMatch(reads[0], /CUBE_COLORS|classif|sticker|palette|snap/i);

  // Local fixed tracks keep changes to the general dashboard styles from
  // reintroducing playback reflow on this page.
  assert.match(reads[0], /className="demo-read-face"/);
  assert.match(reads[0], /className="demo-read-grid"/);
  assert.match(reads[0], /className="demo-read-cell/);
  assert.match(
    demoPageStyles,
    /\.demo-read-faces \{[^}]*grid-template-columns: repeat\(3, 70px\);/,
  );
  assert.match(
    demoPageStyles,
    /\.demo-reads \{[^}]*grid-template-rows: 20px 82px;[^}]*min-height: 109px;/,
  );

  // The dashboard's alignment gate suppresses the reads the same way.
  assert.match(reads[0], /entry\.a < ALIGN_THRESH/);
  assert.match(demoPageSource, /const ALIGN_THRESH = 0\.5;/);

  // It belongs under the scramble in the decoded-output column.
  const scramble = demoPageSource.indexOf('className="demo-scramble"');
  const panel = demoPageSource.indexOf("<SampledReads");
  const timeline = demoPageSource.indexOf("<MoveTimeline");
  assert.ok(scramble > 0 && panel > scramble, "reads follow the scramble");
  assert.ok(panel < timeline, "reads sit above the move timeline");
});

test("the timeline names both of its lanes", () => {
  // Unlabelled marks read as decoration. A viewer has to be able to tell a
  // recorded turn from a tracker signal without guessing.
  const track = /<div[\s\S]*?className="demo-timeline-track"[\s\S]*?\n      <\/div>/.exec(
    demoPageSource,
  );
  assert.ok(track, "the timeline track was not found");
  const labels = [...track[0].matchAll(/demo-timeline-lane-label">\s*([a-z ]+)/g)];
  assert.deepEqual(
    labels.map(([, text]) => text.trim()),
    ["moves", "alignment"],
  );
  // The signal is named, and it is the quantity actually drawn.
  assert.match(demoPageSource, /entry\.a\)/);
  assert.match(demoPageSource, /alignmentCanvasRef/);
  assert.match(demoPageSource, /<small>0 to 1<\/small>/);

  // The playhead spans both lanes and stays the only high-contrast mark.
  assert.match(demoPageStyles, /\.demo-timeline-cursor \{[^}]*background: var\(--yellow\)/);
  assert.match(demoPageStyles, /\.demo-timeline-cursor \{[^}]*top: 0;[^}]*bottom: 0;/);
  assert.match(demoPageSource, /style=\{\{ left: lanePosition\(fraction\) \}\}/);
  assert.match(demoPageSource, /--lane-label-width/);
});

test("playback maps clip time to a source frame with the measured rate", () => {
  assert.equal(GTD1_VIDEO_TIMING.frameCount, 5804);
  assert.equal(GTD1_VIDEO_TIMING.lastFrame, 5803);
  assert.equal(GTD1_VIDEO_TIMING.fps, 696480 / 5803);
  assert.equal(frameIndexForTime(0), 0);
  assert.equal(frameIndexForTime(-5), 0);
  assert.equal(frameIndexForTime(9999), GTD1_VIDEO_TIMING.lastFrame);
  assert.equal(frameIndexForTime(timeForFrameIndex(1234)), 1234);
});

test("the clip loads from the published release and never asks for CORS", () => {
  assert.equal(GTD1_RELEASE_TAG, "v1.0.0");
  assert.equal(
    GTD1_VIDEO_URL,
    "https://github.com/KingBobJoeIV/cubed-core/releases/download/v1.0.0/gtD1s_decoder-demo-clip_f3919-f9722.mp4",
  );
  assert.doesNotMatch(demoPageSource, /crossOrigin/);
  assert.match(demoPageSource, /preload="metadata"/);
  assert.match(demoPageSource, /playsInline/);

  // Source ladder: a workbench that holds the capture serves the exact bytes
  // over a media ticket, which is the only path that works while the
  // repository is private. The release asset is the public-clone fallback and
  // the starting value, so a failed or absent service changes nothing.
  assert.match(demoPageSource, /useState\(GTD1_VIDEO_URL\)/);
  assert.match(demoPageSource, /src=\{videoUrl\}/);
  assert.match(
    demoPageSource,
    /createCaptureMediaTicket\(GTD1_OVERLAY_TRACK\.captureId\)/,
  );
  // The ticket may never gate the render path.
  assert.match(demoPageSource, /\.catch\(\(\) => \{/);
  assert.doesNotMatch(demoPageSource, /await createCaptureMediaTicket/);
});

test("a missing clip degrades to a note and leaves the page working", () => {
  assert.match(demoPageSource, /setVideoFailed\(true\)/);
  assert.match(demoPageSource, /\{!videoFailed && \(/);
  assert.match(demoPageSource, /The demo clip did not load\./);
  assert.match(
    demoPageSource,
    /download_release_assets\.py[\s\S]*--include demo --tag \{GTD1_RELEASE_TAG\}/,
  );
  assert.match(demoPageSource, /Retry clip/);
  // The local clock keeps the overlay, timeline, and cube moving with no video.
  assert.match(demoPageSource, /clockRef\.current \+ delta/);
  assert.match(demoPageSource, /frameIndexForTime\(clockRef\.current\)/);
});

test("the public demo header describes the replay without extra framing", () => {
  const header = /function DemoHeader\(\)[\s\S]*$/.exec(demoPageSource);
  assert.ok(header, "the demo header was not found");
  assert.match(header[0], /<h1>\s*Reconstruction demo/);
  assert.match(header[0], /Artifact <code>gtD1<\/code>/);
  assert.match(
    header[0],
    /Scrub a recorded solve with synchronized face reads, move timing,\s+and cube reconstruction\./,
  );
  assert.doesNotMatch(header[0], /Preserved reconstruction/);
  assert.doesNotMatch(header[0], /className="eyebrow"/);
});

test("core and optional artifacts load, time out, and retry independently", () => {
  assert.match(
    demoPageSource,
    /usePublishedResource<LoadedPublishedDemo>\([\s\S]*?loadPublishedDemo/,
  );
  assert.match(
    demoPageSource,
    /usePublishedResource<PublishedOverlayTrack>\([\s\S]*?loadPublishedOverlayTrack/,
  );
  assert.match(
    demoPageSource,
    /usePublishedResource<PublishedBleGroundTruth>\([\s\S]*?loadPublishedBleGroundTruth/,
  );
  assert.doesNotMatch(demoPageSource, /Promise\.all\(\[\s*loadPublishedDemo/);
  assert.match(demoPageSource, /const controller = new AbortController\(\)/);
  assert.match(demoPageSource, /RESOURCE_LOAD_TIMEOUT_MS/);
  assert.match(demoPageSource, /controller\.abort\(\)/);
  assert.match(demoPageSource, /const retry = useCallback/);
  assert.match(demoPageSource, /onRetry=\{coreResource\.retry\}/);
  assert.match(demoPageSource, /onRetry=\{trackResource\.retry\}/);
  assert.match(demoPageSource, /onRetry=\{bleResource\.retry\}/);
  assert.match(demoPageSource, /onClick=\{retryVideo\}/);

  // A retry clears the prior value, so an artifact that no longer passes its
  // checks cannot remain on screen under a fresh "loaded" claim.
  assert.match(demoPageSource, /setState\("loading"\);\s*setValue\(null\)/);
  assert.match(demoPageSource, /aria-label=\{`Retry \$\{title\.toLowerCase\(\)\}`\}/);
});

test("optional failures retain a truthful core replay", () => {
  assert.match(
    demoPageSource,
    /The decoded cube replay and source clip remain available\./,
  );
  assert.match(
    demoPageSource,
    /still step through the preserved decoded sequence without assigning clip frames/,
  );
  assert.match(demoPageSource, /Next move/);
  assert.match(
    demoPageSource,
    /without claiming clip timing/,
  );
});

test("the core loader reports which required file failed", () => {
  assert.match(manifestSource, /The reconstruction result is missing/);
  assert.match(manifestSource, /The reconstruction receipt is missing/);
});

test("the demo page keeps a concise evidence boundary", () => {
  assert.match(manifestSource, /Published reconstruction result and endpoint replay receipt\./);
  assert.match(manifestSource, /Not live inference\./);
  assert.match(manifestSource, /Not per-move accuracy\./);
  assert.match(manifestSource, /Not reach-LL evidence\./);
  assert.match(manifestSource, /Not generalization evidence\./);
  assert.match(manifestSource, /unverified-external-identity/);
  assert.match(demoPageSource, /const MOVE_TIMING_NOTE =/);
  assert.match(demoPageSource, /smart-cube record/);
  assert.match(demoPageSource, /It runs no\s+decoder and reports no accuracy\./);
  assert.match(demoPageSource, />\s*Dataset card\s*</);
  assert.match(demoPageSource, />\s*Evidence boundary\s*</);
  assert.match(
    demoPageSource,
    /https:\/\/huggingface\.co\/datasets\/cubed-core\/cubed-data-v1\/blob\/7fae604962c590ac9c658ba6ee0350e86de9c4f5\/README\.md/,
  );
  assert.match(demoPageSource, /href=\{DEMO_DATA_CARD\}/);
  assert.doesNotMatch(
    demoPageSource,
    /routeGuideDocumentHref\(DEMO_DATA_CARD\)/,
  );
});

test("the public demo omits artifact-debugging and download UI", () => {
  const boundary = demoPageSource.indexOf("demo-boundary-line");
  const stage = demoPageSource.indexOf('id="demo-stage-title"');
  const moves = demoPageSource.indexOf('id="demo-moves-title"');
  assert.ok(stage < moves, "the clip must come before the move timeline");
  assert.ok(moves < boundary, "the boundary line must follow the replay");
  assert.doesNotMatch(demoPageSource, /className="demo-detail"/);
  assert.doesNotMatch(demoPageSource, /Signals, hashes, and downloads/);
  assert.doesNotMatch(demoPageSource, /Tracker signals on this frame/);
  assert.doesNotMatch(demoPageSource, /Result SHA-256/);
  assert.doesNotMatch(demoPageSource, /Open overlay JSON/);
  assert.doesNotMatch(demoPageSource, /Preserved result/);
  assert.doesNotMatch(demoPageSource, /capability-badge/);
});

test("the clip transport stays below the video and scramble stays visible", () => {
  const viewerStart = demoPageSource.indexOf('className="demo-viewer"');
  const frame = demoPageSource.indexOf('className="demo-frame"', viewerStart);
  const transport = demoPageSource.indexOf(
    'className="demo-transport"',
    viewerStart,
  );
  assert.ok(viewerStart > 0 && frame > viewerStart, "the viewer frame was not found");
  assert.ok(transport > frame, "the transport must follow the video frame");
  assert.match(
    demoPageStyles,
    /\.demo-viewer \{[^}]*grid-template-columns: minmax\(0, 1fr\);/,
  );
  assert.doesNotMatch(
    demoPageStyles,
    /@container[^{]*\{[\s\S]*?\.demo-viewer \{[^}]*grid-template-columns:/,
  );
  assert.match(
    demoPageStyles,
    /\.demo-transport \{[^}]*justify-self: center;[^}]*width: min\(100%, 360px\);/,
  );
  assert.match(
    demoPageStyles,
    /\.demo-frame \{[\s\S]*?width: min\(100%, 320px\);[\s\S]*?aspect-ratio: 1080 \/ 1920;/,
  );
  assert.match(
    demoPageStyles,
    /\.demo-transport-row \{[^}]*justify-content: center;/,
  );
  assert.match(
    demoPageStyles,
    /\.demo-transport-row \.button \{[^}]*min-height: 30px;[^}]*font-size: 11px;/,
  );
  assert.doesNotMatch(demoPageSource, /demo-overlay-key/);
  assert.doesNotMatch(demoPageSource, /faces detected · colors sampled/);

  const scramble =
    /<section\s+className="demo-scramble"[\s\S]*?<\/section>/.exec(
      demoPageSource,
    );
  assert.ok(scramble, "the starting scramble was not found");
  assert.match(scramble[0], /Starting scramble/);
  assert.match(scramble[0], /demo-scramble-tape/);
  assert.doesNotMatch(demoPageSource, /<details className="demo-scramble"/);
});

test("the clip and the decoded output share one co-visible row", () => {
  const live = demoPageSource.indexOf('className="demo-live"');
  const stage = demoPageSource.indexOf('className="demo-stage"');
  const moves = demoPageSource.indexOf('className="demo-moves"');
  const boundary = demoPageSource.indexOf("demo-boundary-line");
  assert.ok(live > 0, "both sections must share one layout wrapper");
  assert.ok(live < stage, "the wrapper must open before the clip");
  assert.ok(stage < moves, "the clip must come before the decoded output");
  assert.ok(moves < boundary, "the boundary line must follow both");

  // Two columns above the breakpoint. Below it the sections stack and the
  // compact strip under the video carries the current move instead.
  assert.match(
    demoPageStyles,
    /@media \(min-width: 1100px\) \{\s*\.demo-live \{\s*grid-template-columns:/,
  );
  assert.match(
    demoPageStyles,
    /@media \(min-width: 1100px\) \{\s*\.demo-move-strip \{\s*display: none;/,
  );
  assert.match(demoPageSource, /demo-move-strip-token-current/);
  const strip = demoPageSource.indexOf("demo-move-strip");
  assert.ok(strip > stage && strip < moves, "the strip belongs under the video");

  // Nothing may scroll the page while the clip plays.
  assert.doesNotMatch(demoPageSource, /scrollIntoView/);
});

test("the decoded moves render as a fixed-height canvas timeline", () => {
  // The series and anchors draw once; only a positioned cursor moves. A
  // 120 fps clip would otherwise redraw the canvas on every frame.
  assert.match(demoPageSource, /function MoveTimeline\(/);
  assert.match(demoPageSource, /const MOVES_LANE_HEIGHT = \d+;/);
  assert.match(demoPageSource, /const ALIGNMENT_LANE_HEIGHT = \d+;/);
  assert.match(demoPageSource, /demo-timeline-cursor/);
  assert.match(demoPageSource, /new ResizeObserver/);
  assert.match(demoPageSource, /onPointerDown=\{onLanePointerDown\}/);
  // Each lane redraws on its data or the width only, never on the frame.
  for (const ref of ["movesCanvasRef", "alignmentCanvasRef"]) {
    const draw = new RegExp(
      `const canvas = ${ref}\\.current;[\\s\\S]*?\\}, \\[([^\\]]*)\\]\\);`,
    ).exec(demoPageSource);
    assert.ok(draw, `the ${ref} draw effect was not found`);
    assert.doesNotMatch(draw[1], /\bframe\b/, `${ref} must not redraw per frame`);
  }
  // The 77-chip grid is gone.
  assert.doesNotMatch(demoPageSource, /move-replay-token/);
  assert.doesNotMatch(demoPageSource, /move-replay-sequence/);

  // The timing attribution has to travel with the timeline it qualifies.
  const timeline = demoPageSource.indexOf("<MoveTimeline");
  const note = demoPageSource.indexOf("demo-moves-note");
  assert.ok(timeline > 0 && note > timeline, "the note follows the timeline");
});

test("the canonicalising fold preserves timing while combining teacher moves", () => {
  assert.match(manifestSource, /simplifyMovesWithFrames/);

  // Grouping adjacent runs is the wrong fold: a run that nets to zero pops and
  // exposes an earlier same-face entry for the next token to merge with.
  assert.deepEqual(
    simplifyMovesWithFrames([
      { move: "U", frame: 10 },
      { move: "R", frame: 20 },
      { move: "R'", frame: 30 },
      { move: "U", frame: 40 },
    ]),
    { moves: ["U2"], frames: [40] },
  );
  // A half turn is only complete on its second quarter turn.
  assert.deepEqual(
    simplifyMovesWithFrames([
      { move: "F", frame: 5 },
      { move: "F", frame: 9 },
    ]),
    { moves: ["F2"], frames: [9] },
  );
  // A net-zero run leaves nothing behind.
  assert.deepEqual(
    simplifyMovesWithFrames([
      { move: "L", frame: 1 },
      { move: "L2", frame: 2 },
      { move: "L", frame: 3 },
    ]),
    { moves: [], frames: [] },
  );
});

test("the published BLE record times every decoded move", () => {
  assert.equal(bleBytes.byteLength, GTD1_BLE_GROUND_TRUTH.bytes);
  assert.equal(sha256(bleBytes), GTD1_BLE_GROUND_TRUTH.sha256);

  const record = parsePublishedBleGroundTruth(ble);
  assert.equal(record.moves.length, GTD1_BLE_GROUND_TRUTH.quarterTurns);

  const timed = bleTimedMoves(record, result.moves);
  assert.equal(timed.length, GTD1_PUBLISHED_DEMO.moveCount);
  assert.deepEqual(
    timed.slice(0, 6),
    [
      { move: "U", frame: 120 },
      { move: "F", frame: 178 },
      { move: "L", frame: 212 },
      { move: "B'", frame: 252 },
      { move: "L'", frame: 363 },
      { move: "B2", frame: 485 },
    ],
  );
  assert.equal(timed[0].frame, GTD1_BLE_GROUND_TRUTH.firstMoveFrame);
  assert.equal(
    timed[timed.length - 1].frame,
    GTD1_BLE_GROUND_TRUTH.lastMoveFrame,
  );
  assert.ok(
    timed.every(
      (entry, index) => index === 0 || entry.frame > timed[index - 1].frame,
    ),
    "recorded turn frames must be strictly increasing",
  );
  assert.ok(
    timed[timed.length - 1].frame < GTD1_VIDEO_TIMING.frameCount,
    "every turn lands inside the clip",
  );

  // The cube advances on recorded turns, so an even spread must not creep back.
  assert.equal(moveStepForFrame(timed, 0), 0);
  assert.equal(moveStepForFrame(timed, 119), 0);
  assert.equal(moveStepForFrame(timed, 120), 1);
  assert.equal(moveStepForFrame(timed, 177), 1);
  assert.equal(moveStepForFrame(timed, 178), 2);
  assert.equal(moveStepForFrame(timed, GTD1_VIDEO_TIMING.lastFrame), 77);
  assert.doesNotMatch(demoPageSource, /moves\.length\) \* GTD1_VIDEO_TIMING/);
});

test("a wrong fold or a short record is refused instead of desynchronizing", () => {
  const shifted = {
    ...ble,
    canonical_moves: ["Z", ...ble.canonical_moves.slice(1)],
  };
  assert.throws(
    () => bleTimedMoves(shifted, result.moves),
    /does not reproduce the published canonical moves/,
  );
  assert.throws(
    () => bleTimedMoves(parsePublishedBleGroundTruth(ble), result.moves.slice(1)),
    /folds to 77 moves and the decoder emitted 76/,
  );
  assert.throws(
    () => parsePublishedBleGroundTruth({ ...ble, moves: ble.moves.slice(1) }),
    /does not match the reviewed gtD1 ground-truth contract/,
  );
});

test("the demo shows the sealed scramble it replays from", () => {
  const block = /demo-scramble"[\s\S]*?demo-moves-note/.exec(demoPageSource);
  assert.ok(block, "the scramble must sit above the decoded move timeline");

  // Rendered from the manifest, so the page cannot drift from the sealed
  // value the trajectory is computed with.
  assert.match(block[0], /GTD1_PUBLISHED_DEMO\.scramble/);
  assert.match(block[0], /scrambleTokens\.map/);
  assert.doesNotMatch(demoPageSource, /R' L2 F L2 U R D F'/);
  assert.equal(GTD1_PUBLISHED_DEMO.scramble.trim().split(/\s+/).length, 21);

  // The heading and the chips carry it. Nothing explains a scramble in prose.
  assert.doesNotMatch(block[0], /Apply these/);
  assert.doesNotMatch(demoPageSource, /demo-scramble-note/);
});

test("contract parsing rejects drift in result identity or receipt claims", () => {
  assert.throws(
    () =>
      parsePublishedDecodeResult({
        ...result,
        config: { ...result.config, cfg_hash: "different" },
      }),
    /does not match the reviewed gtD1 result contract/,
  );
  assert.throws(
    () =>
      parsePublishedDecodeReceipt({
        ...receipt,
        runner_provenance: {
          ...receipt.runner_provenance,
          identity_status: "verified-native-identity",
        },
      }),
    /does not match the reviewed gtD1 receipt contract/,
  );
  assert.throws(
    () => parsePublishedOverlayTrack({ ...overlay, detected_faces: 1 }),
    /does not match the reviewed gtD1 overlay contract/,
  );
  assert.throws(
    () => parsePublishedOverlayTrack({ ...overlay, source_sha256: "different" }),
    /does not match the reviewed gtD1 overlay contract/,
  );
});
