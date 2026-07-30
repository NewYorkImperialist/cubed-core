import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const apiSource = readFileSync(
  resolve(import.meta.dirname, "../src/api.ts"),
  "utf8",
);
const decodeApiSource = readFileSync(
  resolve(import.meta.dirname, "../src/decodeApi.ts"),
  "utf8",
);

test("API requests have bounded waits while large transfers get a deliberate longer budget", () => {
  assert.match(apiSource, /const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;/);
  assert.match(apiSource, /const LARGE_TRANSFER_TIMEOUT_MS = 5 \* 60_000;/);
  assert.match(apiSource, /async function fetchWithTimeout\(/);
  assert.match(apiSource, /const controller = new AbortController\(\);/);
  assert.match(apiSource, /window\.setTimeout\(\(\) => \{[\s\S]*?controller\.abort\(\);/);
  assert.match(
    apiSource,
    /requestJson<CaptureReceipt>\("\/api\/captures\/import",[\s\S]*?LARGE_TRANSFER_TIMEOUT_MS/,
  );
});

test("upstream cancellation remains an AbortError instead of being mislabeled offline", () => {
  assert.match(apiSource, /upstream\?\.addEventListener\("abort", abortFromUpstream/);
  assert.match(apiSource, /upstream\?\.removeEventListener\("abort", abortFromUpstream\);/);
  assert.match(
    apiSource,
    /if \(error instanceof Error && error\.name === "AbortError"\) throw error;/,
  );
  assert.match(apiSource, /did not respond within/);
});

test("decode preflight uses the shared abortable, timed request path", () => {
  assert.match(
    decodeApiSource,
    /export async function fetchDecodePreflight\(\s*recordingId: string,\s*signal\?: AbortSignal/,
  );
  assert.match(
    decodeApiSource,
    /requestJson<DecodePreflight>\([\s\S]*?encodeURIComponent\(recordingId\)[\s\S]*?\{ signal \}/,
  );
  assert.doesNotMatch(
    decodeApiSource,
    /fetch\(`\$\{apiBaseUrl\(\)\}\$\{decodeJobPaths\.preflight/,
  );
});
