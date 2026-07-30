import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeStageChips,
  decodeStageLabel,
  decodeStageProgressLabel,
} from "../src/decodeStages.ts";

test("every Decode stage token maps to a readable label", () => {
  assert.equal(decodeStageLabel("upload"), "Uploading to runner");
  assert.equal(decodeStageLabel("reads"), "Reading frames");
  assert.equal(decodeStageLabel("events"), "Motion events");
  assert.equal(decodeStageLabel("alignfeat"), "Alignment features");
  assert.equal(decodeStageLabel("decode"), "Decode search");
  assert.equal(decodeStageLabel("download"), "Downloading result");
  assert.equal(decodeStageLabel("validate"), "Validating");
  assert.equal(decodeStageLabel("  future-stage  "), "future-stage");
});

test("chips preserve first-seen order and mark current progress", () => {
  const chips = decodeStageChips(
    ["reads", "events", "events", "", "alignfeat", "decode"],
    "alignfeat",
  );
  assert.deepEqual(
    chips.map((chip) => chip.token),
    ["reads", "events", "alignfeat", "decode"],
  );
  assert.deepEqual(
    chips.map((chip) => chip.state),
    ["done", "done", "current", "pending"],
  );
});

test("optional stage reporting never invents a chip", () => {
  assert.deepEqual(decodeStageChips(undefined, undefined), []);
  assert.deepEqual(decodeStageChips([], null), []);
  assert.deepEqual(decodeStageChips(["upload"], "download").map((chip) => chip.token), [
    "upload",
    "download",
  ]);
});

test("progress is bounded and degenerate values render nothing", () => {
  assert.equal(
    decodeStageProgressLabel({ current: 1420, total: 14_000 }),
    "1420 / 14000 frames",
  );
  assert.equal(decodeStageProgressLabel({ current: 99, total: 10 }), "10 / 10 frames");
  assert.equal(decodeStageProgressLabel({ current: -4, total: 10 }), "0 / 10 frames");
  assert.equal(decodeStageProgressLabel({ current: 3, total: 0 }), "");
  assert.equal(decodeStageProgressLabel(undefined), "");
});
