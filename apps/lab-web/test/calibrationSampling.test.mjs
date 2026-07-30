import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  CALIBRATION_COLORS,
  clientPointToNativeVideo,
  containedVideoRect,
  moveNativePoint,
  nativeCropBox,
} from "../src/calibrationSampling.ts";

const sourceRoot = resolve(import.meta.dirname, "../src");
const samplerSource = readFileSync(
  resolve(sourceRoot, "components/ColorCalibrationSampler.tsx"),
  "utf8",
);
const samplerStyles = readFileSync(
  resolve(sourceRoot, "components/ColorCalibrationSampler.css"),
  "utf8",
);
const prepareSource = readFileSync(
  resolve(sourceRoot, "components/PrepareForDecodeCard.tsx"),
  "utf8",
);
const apiSource = readFileSync(resolve(sourceRoot, "api.ts"), "utf8");

test("object-fit contain maps the visible image and rejects its letterbox", () => {
  const bounds = { left: 10, top: 20, width: 1000, height: 1000 };
  const content = containedVideoRect(bounds, 1920, 1080);

  assert.ok(content);
  assert.ok(Math.abs(content.left - 10) < 1e-9);
  assert.ok(Math.abs(content.top - 238.75) < 1e-9);
  assert.ok(Math.abs(content.width - 1000) < 1e-9);
  assert.ok(Math.abs(content.height - 562.5) < 1e-9);
  assert.ok(Math.abs(content.scale - 1000 / 1920) < 1e-9);
  assert.deepEqual(
    clientPointToNativeVideo(510, 520, bounds, 1920, 1080),
    { x: 960, y: 540 },
  );
  assert.equal(
    clientPointToNativeVideo(510, 100, bounds, 1920, 1080),
    null,
  );
});

test("object-fit mapping also rejects portrait-video side bars", () => {
  const bounds = { left: 0, top: 0, width: 1000, height: 1000 };
  assert.equal(
    clientPointToNativeVideo(100, 500, bounds, 1080, 1920),
    null,
  );
  assert.deepEqual(
    clientPointToNativeVideo(500, 500, bounds, 1080, 1920),
    { x: 540, y: 960 },
  );
});

test("native crop boxes use 2.5 percent of the short edge and stay in-frame", () => {
  assert.deepEqual(nativeCropBox({ x: 960, y: 540 }, 1920, 1080), {
    x: 947,
    y: 527,
    side: 27,
  });
  assert.deepEqual(nativeCropBox({ x: 0, y: 0 }, 1920, 1080), {
    x: 0,
    y: 0,
    side: 27,
  });
  assert.equal(nativeCropBox({ x: 2000, y: 2000 }, 4000, 4000).side, 96);
  assert.equal(nativeCropBox({ x: 100, y: 100 }, 320, 240).side, 16);
});

test("keyboard crosshair movement is bounded and has a coarse modifier", () => {
  assert.deepEqual(moveNativePoint(null, "left", 1920, 1080), {
    x: 955,
    y: 540,
  });
  assert.deepEqual(
    moveNativePoint({ x: 1, y: 1 }, "left", 1920, 1080, true),
    { x: 0, y: 1 },
  );
  assert.deepEqual(
    moveNativePoint({ x: 1918, y: 1078 }, "down", 1920, 1080, true),
    { x: 1918, y: 1079 },
  );
});

test("the browser emits exactly six named lossless PNG crops", () => {
  assert.deepEqual(CALIBRATION_COLORS, [
    "white",
    "green",
    "red",
    "blue",
    "orange",
    "yellow",
  ]);
  assert.match(samplerSource, /const OUTPUT_PATCH_SIDE = 96;/);
  assert.match(samplerSource, /canvas\.toBlob\([\s\S]*"image\/png"/);
  assert.match(samplerSource, /context\.drawImage\(\s*video,/);
  assert.match(apiSource, /for \(const color of CALIBRATION_COLORS\)/);
  assert.match(apiSource, /data\.set\(color, patches\[color\], `\$\{color\}\.png`\)/);
  assert.match(
    apiSource,
    /\/calibration\/from-crops/,
  );
});

test("sampling stays on one HTML video without JS color conversion or ffmpeg", () => {
  assert.match(samplerSource, /createCaptureMediaTicket\(captureId\)/);
  assert.match(samplerSource, /<video/);
  assert.match(samplerStyles, /object-fit:\s*contain/);
  assert.match(samplerSource, /onSeeking=\{\(\) => \{\s+seekingRef\.current = true;/);
  assert.match(samplerSource, /onSeeked=\{\(event\) => \{/);
  assert.match(
    samplerSource,
    /if \(seekingRef\.current\) \{\s+setError\("Wait for the selected video frame to finish loading\."\)/,
  );
  assert.doesNotMatch(
    samplerSource,
    /ffmpeg|fetch\([^)]*frame|rgbToLab|labToRgb|color-centroid/i,
  );
});

test("video sampling is primary while alternate calibration sources are secondary", () => {
  assert.match(prepareSource, /<ColorCalibrationSampler/);
  assert.match(
    prepareSource,
    /<details className="prepare-calibration-disclosure prepare-alternate-calibration">/,
  );
  assert.match(prepareSource, /Use another calibration source/);
  assert.match(prepareSource, /Published dataset shared calibration/);
  assert.match(prepareSource, /Upload a file/);
  assert.match(prepareSource, /\{replaceSamplerOpen && sampler\}/);
});
