import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import { apiStateFromProbe } from "../src/apiSession.ts";

const appSource = readFileSync(
  resolve(import.meta.dirname, "../src/App.tsx"),
  "utf8",
);

test("a healthy service with a rejected capabilities call reads as unauthorized, not offline", () => {
  assert.equal(
    apiStateFromProbe({ healthOk: true, capabilitiesStatus: 403 }),
    "unauthorized",
  );
  assert.equal(
    apiStateFromProbe({ healthOk: true, capabilitiesStatus: 401 }),
    "unauthorized",
  );
});

test("a healthy service with successful capabilities reads as online", () => {
  assert.equal(
    apiStateFromProbe({ healthOk: true, capabilitiesStatus: null }),
    "online",
  );
});

test("a healthy service with a degraded capabilities call still reads as online", () => {
  assert.equal(
    apiStateFromProbe({ healthOk: true, capabilitiesStatus: 500 }),
    "online",
  );
});

test("an unreachable health check reads as offline regardless of capabilities", () => {
  assert.equal(
    apiStateFromProbe({ healthOk: false, capabilitiesStatus: null }),
    "offline",
  );
  assert.equal(
    apiStateFromProbe({ healthOk: false, capabilitiesStatus: 403 }),
    "offline",
  );
});

test("LabShell wires the unauthorized probe state to an in-app admin token panel", () => {
  assert.match(appSource, /"unauthorized"/);
  assert.match(appSource, /AdminTokenPanel/);
  assert.match(appSource, /AbortSignal\.timeout\(8000\)/);
  assert.doesNotMatch(appSource, /127\.0\.0\.1:8000/);
});
