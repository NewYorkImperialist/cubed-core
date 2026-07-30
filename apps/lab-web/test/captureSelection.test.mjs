import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  CAPTURE_PARAM,
  captureHref,
  captureIdFromParams,
  explicitCaptureIdFromParams,
  isPublishedDatasetCapture,
  orderDecodeCaptures,
} from "../src/captureSelection.ts";

const captures = [{ capture_id: "aaa" }, { capture_id: "bbb" }, { capture_id: "ccc" }];

test("a param matching a listed capture wins", () => {
  const params = new URLSearchParams({ [CAPTURE_PARAM]: "bbb" });
  assert.equal(captureIdFromParams(params, captures), "bbb");
});

test("an unknown param falls back to the first capture", () => {
  const params = new URLSearchParams({ [CAPTURE_PARAM]: "zzz" });
  assert.equal(captureIdFromParams(params, captures), "aaa");
});

test("a missing param falls back to the first capture", () => {
  const params = new URLSearchParams();
  assert.equal(captureIdFromParams(params, captures), "aaa");
});

test("an empty capture list yields an empty string regardless of the param", () => {
  const withParam = new URLSearchParams({ [CAPTURE_PARAM]: "aaa" });
  assert.equal(captureIdFromParams(withParam, []), "");
  const withoutParam = new URLSearchParams();
  assert.equal(captureIdFromParams(withoutParam, []), "");
});

test("explicit capture selection accepts only a listed URL capture", () => {
  const matching = new URLSearchParams({ [CAPTURE_PARAM]: "bbb" });
  assert.equal(explicitCaptureIdFromParams(matching, captures), "bbb");

  const unknown = new URLSearchParams({ [CAPTURE_PARAM]: "zzz" });
  assert.equal(explicitCaptureIdFromParams(unknown, captures), "");
  assert.equal(explicitCaptureIdFromParams(new URLSearchParams(), captures), "");
});

test("captureHref appends the capture param only when an id is given", () => {
  assert.equal(captureHref("/label", "aaa"), "/label?capture=aaa");
  assert.equal(captureHref("/label", ""), "/label");
});

test("Decode groups personal and published recordings without mutating the source order", () => {
  const source = [
    {
      capture_id: "public-10",
      capture_session_id: "public-dataset:cubed-solves-v1@rev:public-10",
      created_at: "2026-07-20T00:00:00Z",
      original_filename: "cs10.mp4",
    },
    {
      capture_id: "mine-old",
      capture_session_id: "mine-old",
      created_at: "2026-07-21T00:00:00Z",
      original_filename: "solve10.mov",
    },
    {
      capture_id: "public-2",
      capture_session_id: "public-dataset:cubed-solves-v1@rev:public-2",
      created_at: "2026-07-20T00:00:00Z",
      original_filename: "cs2.mp4",
    },
    {
      capture_id: "mine-new",
      created_at: "2026-07-22T00:00:00Z",
      original_filename: "solve2.mov",
    },
  ];

  const ordered = orderDecodeCaptures(source);

  assert.deepEqual(
    ordered.yourRecordings.map((capture) => capture.capture_id),
    ["mine-new", "mine-old"],
  );
  assert.deepEqual(
    ordered.publishedDataset.map((capture) => capture.capture_id),
    ["public-2", "public-10"],
  );
  assert.deepEqual(
    source.map((capture) => capture.capture_id),
    ["public-10", "mine-old", "public-2", "mine-new"],
  );
  assert.equal(isPublishedDatasetCapture(source[0]), true);
  assert.equal(isPublishedDatasetCapture(source[1]), false);
});

test("Decode ordering has stable filename, timestamp, and ID tie-breaks", () => {
  const sameTime = "2026-07-22T00:00:00Z";
  const ordered = orderDecodeCaptures([
    {
      capture_id: "mine-a",
      created_at: sameTime,
      original_filename: "solve10.mov",
    },
    {
      capture_id: "mine-z",
      created_at: sameTime,
      original_filename: "solve2.mov",
    },
    {
      capture_id: "published-b",
      capture_session_id: "public-dataset:set@rev:published-b",
      created_at: "2026-07-21T00:00:00Z",
      original_filename: "cs2.mp4",
    },
    {
      capture_id: "published-a",
      capture_session_id: "public-dataset:set@rev:published-a",
      created_at: "2026-07-22T00:00:00Z",
      original_filename: "cs2.mp4",
    },
    {
      capture_id: "published-c",
      capture_session_id: "public-dataset:set@rev:published-c",
      created_at: "2026-07-21T00:00:00Z",
      original_filename: "cs2.mp4",
    },
  ]);

  assert.deepEqual(
    ordered.yourRecordings.map((capture) => capture.capture_id),
    ["mine-z", "mine-a"],
  );
  assert.deepEqual(
    ordered.publishedDataset.map((capture) => capture.capture_id),
    ["published-a", "published-b", "published-c"],
  );
});

const sourceRoot = resolve(import.meta.dirname, "../src");
const appSource = readFileSync(resolve(sourceRoot, "App.tsx"), "utf8");
const importSource = readFileSync(
  resolve(sourceRoot, "components/VideoImportWorkbench.tsx"),
  "utf8",
);
const labelSource = readFileSync(
  resolve(sourceRoot, "components/LocalLabelWorkbench.tsx"),
  "utf8",
);
const decodeSource = readFileSync(
  resolve(sourceRoot, "components/DecodeStage.tsx"),
  "utf8",
);

test("old phone and capture links resolve to the existing-video importer", () => {
  assert.match(
    appSource,
    /path="\/phone" element=\{<Navigate to="\/import" replace \/>}/,
  );
  assert.match(
    appSource,
    /path="\/capture" element=\{<Navigate to="\/import" replace \/>}/,
  );
});

test("Label and Decode own URL selection", () => {
  for (const source of [labelSource, decodeSource]) {
    assert.match(source, /captureIdFromParams/);
  }
});

test("the imported recording links forward with the capture carried", () => {
  assert.match(importSource, /captureHref\("\/decode", activeCapture\.capture_id\)/);
  assert.match(importSource, /captureHref\("\/label", activeCapture\.capture_id\)/);
});
