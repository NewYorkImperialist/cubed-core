import assert from "node:assert/strict";
import test from "node:test";

import {
  formatFinishedAt,
  isTerminalDecodeState,
} from "../src/decodeJobContract.ts";

test("formatFinishedAt renders a locale timestamp and falls back on absence", () => {
  const iso = "2026-07-24T18:30:00.000Z";
  assert.equal(formatFinishedAt(iso), new Date(iso).toLocaleString());
  assert.equal(formatFinishedAt(null), "a previous run");
  assert.equal(formatFinishedAt("not-a-date"), "a previous run");
});

test("only server terminal states stop Decode polling", () => {
  assert.equal(isTerminalDecodeState("queued"), false);
  assert.equal(isTerminalDecodeState("running"), false);
  assert.equal(isTerminalDecodeState("succeeded"), true);
  assert.equal(isTerminalDecodeState("failed"), true);
  assert.equal(isTerminalDecodeState("timed_out"), true);
});
