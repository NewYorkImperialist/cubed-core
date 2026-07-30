import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const stylesSource = readFileSync(
  resolve(import.meta.dirname, "../src/styles.css"),
  "utf8",
);
const toolsSource = readFileSync(
  resolve(import.meta.dirname, "../src/labTools.css"),
  "utf8",
);

test("capture media uses a dark-mode letterbox instead of the text color", () => {
  const stageRule = toolsSource.match(
    /\.label-stage,\s*\.tracker-video-stage\s*\{[^}]*\}/,
  )?.[0];
  const mediaRule = toolsSource.match(
    /\.label-stage video,\s*\.label-stage > img,\s*\.tracker-video-stage video\s*\{[^}]*\}/,
  )?.[0];

  assert.ok(stageRule);
  assert.ok(mediaRule);
  assert.match(
    stylesSource,
    /:root\s*\{[\s\S]*--media-letterbox:\s*#14232e;/,
  );
  assert.match(
    stylesSource,
    /:root\[data-theme="dark"\]\s*\{[\s\S]*--media-letterbox:\s*#050708;/,
  );
  assert.match(
    stageRule,
    /var\(--media-letterbox\);/,
  );
  assert.match(
    mediaRule,
    /background:\s*var\(--media-letterbox\);[\s\S]*object-fit:\s*contain;/,
  );
  assert.doesNotMatch(stageRule, /var\(--ink\)/);
});
