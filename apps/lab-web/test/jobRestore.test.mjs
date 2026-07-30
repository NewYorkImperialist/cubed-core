import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const sourceRoot = resolve(import.meta.dirname, "../src");

function read(relative) {
  return readFileSync(resolve(sourceRoot, relative), "utf8");
}

const decodeSource = read("components/DecodeStage.tsx");
const decodeApiSource = read("decodeApi.ts");

// Slices the source between a restore effect's opening line and its
// register-the-cleanup return, so assertions stay scoped to that block
// instead of matching text anywhere else in a 1000+ line component.
function restoreEffectBlock(source, openingComment) {
  const start = source.indexOf(openingComment);
  assert.ok(start >= 0, `restore effect comment not found: ${openingComment}`);
  const end = source.indexOf("return () => controller.abort();", start);
  assert.ok(end > start, "restore effect cleanup was not found");
  return source.slice(start, end);
}

test("the decode API exposes a capture-scoped jobs list route", () => {
  assert.match(decodeApiSource, /list\(captureId: string\): string \{/);
  assert.match(
    decodeApiSource,
    /export function fetchDecodeJobsForCapture\(\s*captureId: string,\s*signal\?: AbortSignal,\s*\): Promise<DecodeJobList> \{/,
  );
});

test("the decode stage queries the capture's own decode jobs to restore a result", () => {
  const block = restoreEffectBlock(
    decodeSource,
    "// Loads an already-finished result instead of requiring a rerun",
  );
  assert.match(block, /fetchDecodeJobsForCapture\(captureId, controller\.signal\)/);
  assert.match(block, /const newestJob = jobs\[0\];/);
});

test("the decode auto-load is gated on no job or result present and nothing running", () => {
  const block = restoreEffectBlock(
    decodeSource,
    "// Loads an already-finished result instead of requiring a rerun",
  );
  assert.match(
    block,
    /if \(!captureId \|\| decodeJob \|\| decodeResult \|\| decodeRunning\) return;/,
  );
});

test("a non-terminal newest decode job resumes polling instead of loading a result", () => {
  const block = restoreEffectBlock(
    decodeSource,
    "// Loads an already-finished result instead of requiring a rerun",
  );
  assert.match(block, /if \(!isTerminalDecodeState\(newestJob\.status\)\) \{/);
  assert.match(block, /void resumeDecodeJob\(newestJob\)/);
  assert.match(decodeSource, /const resumeDecodeJob = async \(initialJob: DecodeJobStatus\) => \{/);
});

test("the loaded decode result message names the run's finished timestamp", () => {
  assert.match(
    decodeSource,
    /`Loaded the decode result from \$\{formatFinishedAt\(newestJob\.finished_at\)\}\.`/,
  );
  assert.match(
    decodeSource,
    /`Loaded the decode result from \$\{formatFinishedAt\(job\.finished_at\)\}\.`/,
  );
  assert.match(decodeSource, /\{restoreNote && \(/);
});

test("selection changes cancel an in-flight decode restore fetch through a dedicated generation guard", () => {
  assert.match(decodeSource, /const decodeRestoreGenerationRef = useRef\(0\);/);
  assert.match(decodeSource, /const decodeRestoreControllerRef = useRef<AbortController \| null>\(null\);/);

  const decodeBlock = restoreEffectBlock(
    decodeSource,
    "// Loads an already-finished result instead of requiring a rerun",
  );
  assert.match(decodeBlock, /decodeRestoreGenerationRef\.current \+= 1;/);
  assert.match(decodeBlock, /decodeRestoreControllerRef\.current\?\.abort\(\);/);
  assert.match(decodeBlock, /if \(decodeRestoreGenerationRef\.current !== generation\) return;/);
});

test("decode restore treats an absent legacy list route as expected and surfaces other failures", () => {
  const decodeBlock = restoreEffectBlock(
    decodeSource,
    "// Loads an already-finished result instead of requiring a rerun",
  );
  assert.match(
    decodeBlock,
    /if \(reason instanceof RequestError && reason\.status === 404\) return;/,
  );
  assert.match(decodeBlock, /Previous decode jobs could not be checked/);
  assert.match(decodeBlock, /setDecodeError\(/);
});
