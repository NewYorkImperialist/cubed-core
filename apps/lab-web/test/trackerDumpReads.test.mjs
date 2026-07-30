import assert from "node:assert/strict";
import test from "node:test";

import {
  LocalContractError,
  parseTrackerDump,
} from "../src/localContracts.ts";

function readDump() {
  return {
    schema: "cubed-core/tracker-dump-v1",
    schema_version: 1,
    window: [0, 0],
    gt: [],
    emitted: [],
    anchors: [],
    events: [],
    spans: [],
    frames: {
      "0": {
        motion: null,
        reads: [
          {
            slot: "up",
            lab: Array.from({ length: 9 }, () => [128, 129, 130]),
            confidence: Array(9).fill(0.75),
            valid_pixels: Array(9).fill(192),
            total_pixels: Array(9).fill(256),
            used_fallback: Array(9).fill(false),
            relative_area: 1,
            corners: [
              [5, 5],
              [20, 5],
              [20, 20],
              [5, 20],
            ],
          },
        ],
      },
    },
  };
}

function richTrackerDump() {
  const dump = readDump();
  dump.gt = ["R", "R"];
  dump.emitted = ["R2"];
  dump.gt_canonical = ["R2"];
  dump.emitted_canonical = ["R2"];
  dump.anchors = [[0, "R2", 0]];
  dump.bridge = ["U"];
  dump.reach = true;
  dump.acc = 1;
  dump.frames["0"].read = [
    {
      slot: "up",
      colors: [
        "white",
        "white",
        "white",
        "red",
        null,
        "orange",
        "blue",
        "green",
        "yellow",
      ],
      conf: [1, 0.9, 0.8, 0.7, null, 0.5, 0.4, 0.3, 0.2],
      dist: [0, 1, 2, 3, 30, 5, 6, 7, 8],
    },
  ];
  dump.states = [
    Object.fromEntries(
      [
        ["up", "white"],
        ["right", "red"],
        ["front", "green"],
        ["down", "yellow"],
        ["left", "orange"],
        ["back", "blue"],
      ].map(([face, color]) => [face, Array(9).fill(color)]),
    ),
  ];
  return dump;
}

test("tracker dump normalization preserves bounded geometric reads", () => {
  const parsed = parseTrackerDump(readDump());
  const read = parsed.frames["0"].reads[0];

  assert.equal(read.slot, "up");
  assert.deepEqual(read.lab[0], [128, 129, 130]);
  assert.equal(read.lab.length, 9);
  assert.equal(read.confidence.length, 9);
  assert.equal(read.valid_pixels.length, 9);
  assert.equal(read.total_pixels.length, 9);
  assert.equal(read.used_fallback.length, 9);
});

test("tracker dump normalization rejects the retired dashboard schema", () => {
  const legacy = readDump();
  legacy.schema = "dashboard/v1";

  assert.throws(
    () => parseTrackerDump(legacy),
    (error) =>
      error instanceof LocalContractError &&
      error.message === "Tracker JSON must declare cubed-core/tracker-dump-v1.",
  );
});

test("tracker dump normalization rejects inconsistent or duplicate reads", () => {
  const inconsistent = readDump();
  inconsistent.frames["0"].reads[0].valid_pixels[0] = 257;
  assert.throws(
    () => parseTrackerDump(inconsistent),
    (error) =>
      error instanceof LocalContractError &&
      error.message.includes("must not exceed total_pixels"),
  );

  const duplicate = readDump();
  duplicate.frames["0"].reads.push(structuredClone(duplicate.frames["0"].reads[0]));
  assert.throws(
    () => parseTrackerDump(duplicate),
    (error) =>
      error instanceof LocalContractError &&
      error.message.includes("duplicate slot up"),
  );

  const empty = readDump();
  empty.frames["0"].reads = [];
  assert.throws(
    () => parseTrackerDump(empty),
    (error) =>
      error instanceof LocalContractError &&
      error.message.includes("must not be empty when present"),
  );
});

test("tracker dump normalization preserves extended trellis fields", () => {
  const parsed = parseTrackerDump(richTrackerDump());

  assert.deepEqual(parsed.gt_canonical, ["R2"]);
  assert.deepEqual(parsed.emitted_canonical, ["R2"]);
  assert.deepEqual(parsed.bridge, ["U"]);
  assert.equal(parsed.reach, true);
  assert.equal(parsed.acc, 1);
  assert.deepEqual(parsed.frames["0"].read, richTrackerDump().frames["0"].read);
  assert.deepEqual(parsed.states, richTrackerDump().states);
});

test("tracker dump normalization rejects malformed extended evidence", () => {
  const shortRead = richTrackerDump();
  shortRead.frames["0"].read[0].colors.pop();
  assert.throws(
    () => parseTrackerDump(shortRead),
    (error) =>
      error instanceof LocalContractError &&
      error.message.includes("must contain exactly nine values"),
  );

  const incompleteState = richTrackerDump();
  delete incompleteState.states[0].back;
  assert.throws(
    () => parseTrackerDump(incompleteState),
    (error) =>
      error instanceof LocalContractError &&
      error.message.includes("states[0].back"),
  );
});
