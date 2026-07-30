import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const component = readFileSync(
  resolve(import.meta.dirname, "../src/components/TrellisBeam.tsx"),
  "utf8",
);

test("a bridge-only trellis does not render an empty span ribbon", () => {
  assert.match(
    component,
    /\{spans\.length > 0 && \(\s*<div className="dash-span-ribbon">/,
  );
});
