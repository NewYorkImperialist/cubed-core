import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const workbenchSource = readFileSync(
  resolve(
    import.meta.dirname,
    "../src/components/LocalLabelWorkbench.tsx",
  ),
  "utf8",
);
const labelStyles = readFileSync(
  resolve(
    import.meta.dirname,
    "../src/components/LocalLabelWorkbench.css",
  ),
  "utf8",
);
const globalStyles = readFileSync(
  resolve(import.meta.dirname, "../src/styles.css"),
  "utf8",
);

test("workspace Label reuses one ticketed video and seeks on the source frame clock", () => {
  const videoTag = workbenchSource.match(
    /<video\s+ref=\{videoRef\}[\s\S]*?\/>/,
  )?.[0];
  assert.ok(videoTag, "shared Label <video> element not found");
  assert.match(videoTag, /src=\{sourceUrl\}/);
  assert.match(videoTag, /preload="auto"/);
  assert.match(videoTag, /onError=\{/);
  assert.match(
    videoTag,
    /Workspace video access expired or the stream became unavailable/,
  );
  assert.match(workbenchSource, /createCaptureMediaTicket\(receipt\.capture_id\)/);
  assert.match(workbenchSource, /\(next \+ 0\.5\) \/ fps/);
  assert.doesNotMatch(workbenchSource, /fetchCaptureFrame/);
  assert.doesNotMatch(workbenchSource, /frameImageUrl|frameBusy|frameRequestRef/);
});

test("opening a capture does not start the whole-video model alignment scan", () => {
  const installStart = workbenchSource.indexOf("const installWorkspaceSource");
  const installEnd = workbenchSource.indexOf("\n  useEffect(", installStart);
  assert.ok(installStart >= 0 && installEnd > installStart);
  assert.doesNotMatch(
    workbenchSource.slice(installStart, installEnd),
    /scanCaptureLabelAlignment/,
  );

  assert.match(
    workbenchSource,
    /const scanModelAlignment = useCallback\(async \(\) => \{/,
  );
  assert.match(workbenchSource, /Model-aligned only \(scan on demand\)/);
  assert.match(workbenchSource, /void scanModelAlignment\(\)/);
});

test("an unknown frame range renders an explanatory message near the scrubber", () => {
  assert.match(workbenchSource, /const frameRangeUnknown = useMemo\(/);
  assert.match(
    workbenchSource,
    /does not report a frame count, so the scrubber/,
  );
  assert.match(workbenchSource, /\{frameRangeUnknown && \(/);
});

test("workspace loading distinguishes failure from an empty capture list and offers retry", () => {
  assert.match(workbenchSource, /Workspace captures unavailable/);
  assert.match(workbenchSource, /Workspace captures could not be loaded:/);
  assert.match(workbenchSource, /Workbench capabilities could not be loaded:/);
  assert.match(workbenchSource, /Retry workspace/);
  assert.match(
    workbenchSource,
    /setWorkspaceLoadAttempt\(\(attempt\) => attempt \+ 1\)/,
  );
});

test("Label shares Decode's deterministic recording groups and ordering", () => {
  assert.match(
    workbenchSource,
    /import \{[\s\S]*?orderDecodeCaptures,[\s\S]*?\} from "\.\.\/captureSelection";/,
  );
  assert.match(
    workbenchSource,
    /const captureGroups = useMemo\([\s\S]*?orderDecodeCaptures\(captures\)/,
  );
  assert.match(
    workbenchSource,
    /const orderedCaptures = orderDecodeCaptures\(capturePayload\.captures\)/,
  );
  assert.match(
    workbenchSource,
    /<optgroup label="Your recordings">[\s\S]*?captureGroups\.yourRecordings\.map/,
  );
  assert.match(
    workbenchSource,
    /<optgroup label="Published dataset">[\s\S]*?captureGroups\.publishedDataset\.map/,
  );
});

test("annotations have a keyboard-selectable list and bounded arrow-key movement", () => {
  assert.match(workbenchSource, /className="label-annotation-list"/);
  assert.match(workbenchSource, /aria-pressed=/);
  assert.match(workbenchSource, /if \(!event\.key\.startsWith\("Arrow"\)\) return;/);
  assert.match(workbenchSource, /const step = event\.shiftKey \? 10 : 1;/);
  assert.match(workbenchSource, /translatePolygonWithinImage\(/);
  assert.match(workbenchSource, /dimensions\.width - Math\.max\(\.\.\.xs\)/);
});

test("the Label layout keeps source and filter disclosures from moving neighboring controls", () => {
  assert.match(
    labelStyles,
    /\.label-source-panel\s*\{[\s\S]*?grid-template-areas:[\s\S]*?"heading main"[\s\S]*?"\. secondary"/,
  );
  assert.match(
    labelStyles,
    /\.label-source-panel\s*\{[\s\S]*?align-items:\s*start;/,
  );
  assert.match(
    labelStyles,
    /\.label-filter-menu \.label-filter-group\s*\{[\s\S]*?position:\s*absolute;/,
  );
  assert.match(
    labelStyles,
    /\.label-help-popover\s*\{[\s\S]*?position:\s*absolute;/,
  );
  assert.doesNotMatch(labelStyles, /\.label-filter-menu\[open\][^{]*\{[^}]*flex-basis:/);
  assert.doesNotMatch(
    labelStyles,
    /\.label-secondary-source\[open\][^{]*\{[^}]*grid-column:/,
  );
});

test("the Label work area has grouped tools and a quiet two-column receipt", () => {
  assert.match(workbenchSource, /className="label-toolset-name">Mode</);
  assert.match(workbenchSource, /className="label-toolset-name">Assist</);
  assert.match(workbenchSource, /className="label-toolset-name">Model</);
  assert.match(workbenchSource, /<summary>Source and frame rate<\/summary>/);
  assert.match(
    labelStyles,
    /\.label-tool-page \.tool-metrics\s*\{[\s\S]*?grid-template-columns:\s*repeat\(2,/,
  );
});

test("screen-reader-only labels use a global utility", () => {
  assert.match(globalStyles, /\.sr-only\s*\{/);
  assert.match(workbenchSource, /className="sr-only" htmlFor="label-capture-select"/);
});
