import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const source = readFileSync(
  resolve(import.meta.dirname, "../src/components/RunsWorkbench.tsx"),
  "utf8",
).replaceAll("\r\n", "\n");
const inspectorSource = readFileSync(
  resolve(import.meta.dirname, "../src/components/DecodeRunInspector.tsx"),
  "utf8",
);
const trellisSource = readFileSync(
  resolve(import.meta.dirname, "../src/components/TrellisBeam.tsx"),
  "utf8",
);
const contractSource = readFileSync(
  resolve(import.meta.dirname, "../src/decodeRunContracts.ts"),
  "utf8",
);
const apiSource = readFileSync(
  resolve(import.meta.dirname, "../src/decodeApi.ts"),
  "utf8",
);
const trackerStyles = readFileSync(
  resolve(import.meta.dirname, "../src/labTools.css"),
  "utf8",
);
const refreshStyles = readFileSync(
  resolve(import.meta.dirname, "../src/workbenchRefresh.css"),
  "utf8",
);
const diagnosticsStyles = readFileSync(
  resolve(import.meta.dirname, "../src/components/DecodeDiagnostics.css"),
  "utf8",
);

test("Runs lists the global Decode history newest first without capture fanout", () => {
  assert.match(
    source,
    /const \[jobsResult, capturesResult\] = await Promise\.allSettled\(\[/,
  );
  assert.match(source, /fetchDecodeJobs\(\),\s*fetchCaptures\(\),/);
  assert.match(source, /jobsResult\.value\.jobs\s*\.map\(/);
  assert.match(source, /new Date\(right\.job\.created_at\)/);
  assert.match(source, /aria-label="Recent workspace runs"/);
  assert.doesNotMatch(
    source,
    /fetchDecodeJobsForCapture/,
  );
  assert.match(apiSource, /listAll\(query = "", limit = 100\)/);
  assert.match(apiSource, /`\/api\/decode\/jobs\?\$\{params\.toString\(\)\}`/);
  assert.match(apiSource, /export function fetchDecodeJobs\(/);
});

test("global jobs remain visible when capture metadata is unavailable", () => {
  assert.match(
    source,
    /if \(jobsResult\.status === "rejected"\) throw jobsResult\.reason;/,
  );
  assert.match(
    source,
    /capturesResult\.status === "fulfilled"\s*\? capturesResult\.value\.captures\s*: \[\]/,
  );
  assert.match(
    source,
    /Run history is available, but recording metadata and video pairing could not be loaded\./,
  );
  assert.match(
    source,
    /capture: capturesById\.get\(job\.capture_id\) \?\? null/,
  );
  assert.match(
    source,
    /attempt\.job\.video_sha256 \|\|\s*attempt\.capture\?\.video\.sha256 \|\|\s*`capture:\$\{attempt\.job\.capture_id\}`/,
  );
  assert.match(
    source,
    /`Capture \$\{latest\.job\.capture_id\.slice\(0, 12\)\}`/,
  );
});

test("run cards use recording duration instead of unreliable legacy job timestamps", () => {
  assert.match(source, /function recordingDuration\(capture: CaptureReceipt \| null\)/);
  assert.match(source, /const seconds = frames \/ fps;/);
  assert.match(source, /title="Recording duration"/);
  assert.doesNotMatch(source, /finished <= started|function runDuration/);
});

test("Runs groups attempts by exact video identity with quiet history", () => {
  assert.match(source, /attempt\.job\.video_sha256/);
  assert.match(source, /attempt\.capture\?\.video\.sha256/);
  assert.match(source, /const \[latest, \.\.\.history\] = group\.attempts;/);
  assert.match(source, /className="run-history"/);
  assert.match(source, /earlier attempt/);
  assert.match(source, /group\.attempts\.length/);
});

test("Runs search covers recording, attempt, outcome, and failure identity", () => {
  assert.match(source, /function attemptMatches\(/);
  assert.match(source, /capture\?\.original_filename/);
  assert.match(source, /job\.video_sha256 \?\? capture\?\.video\.sha256/);
  assert.match(source, /job\.job_id/);
  assert.match(source, /job\.outcome \?\? ""/);
  assert.match(source, /job\.failure\?\.message \?\? job\.error/);
  assert.match(source, /placeholder="Search filename, status, or ID"/);
});

test("all Decode outcomes remain in history while only results are inspectable", () => {
  assert.match(source, /case "completed":/);
  assert.match(source, /case "abstained":/);
  assert.match(source, /case "failed":/);
  assert.match(source, /case "cancelled":/);
  assert.match(source, /\{job\.result_available && \(\s*<>/);
  assert.match(source, />\s*Inspect\s*<\/button>/);
  assert.match(source, />\s*Download\s*<\/button>/);
});

test("Runs is read-only and points new work back to Decode", () => {
  assert.doesNotMatch(
    source,
    /submitDecodeJob|Run decode/,
  );
  assert.match(source, /Runs never starts compute\./);
  assert.match(source, /Runs is read-only\. Return to Decode/);
  assert.match(source, /Submit a prepared recording from Decode\./);
});

test("terminal attempts have recoverable run-only deletion", () => {
  assert.match(source, /\{terminal && \(/);
  assert.match(source, /Delete run/);
  assert.match(source, /move to recoverable trash/);
  assert.match(source, /The capture and video stay in the workspace\./);
  assert.match(source, /await deleteDecodeRun\(attempt\.job\.job_id\)/);
  assert.match(
    apiSource,
    /remove\(jobId: string\): string \{\s*return `\/api\/decode\/jobs\/\$\{encodeURIComponent\(jobId\)\}`;/,
  );
  assert.match(apiSource, /method: "DELETE"/);
});

test("selecting a result opens its exact Decode artifact", () => {
  assert.match(source, /next\.set\(RUN_PARAM, runId\)/);
  assert.match(source, /fetchDecodeJobResult\(selectedAttempt\.job\.job_id/);
  assert.match(source, /<DecodeRunInspector/);
  assert.match(
    source,
    /resultName=\{`cubed-core-run-\$\{selectedAttempt\.job\.job_id\}\.json`\}/,
  );
  assert.match(source, /Change run/);
});

test("every run download preserves the source artifact bytes", () => {
  assert.match(
    apiSource,
    /export function fetchDecodeJobResultBlob\([\s\S]*?return requestBlob\(decodeJobPaths\.result\(jobId\), \{ signal \}\);/,
  );
  assert.match(
    source,
    /const artifact = await fetchDecodeJobResultBlob\(attempt\.job\.job_id\);/,
  );
  assert.match(
    inspectorSource,
    /job\s*\? await fetchDecodeJobResultBlob\(job\.job_id\)\s*: sourceResultFile/,
  );
  assert.doesNotMatch(
    `${source}\n${inspectorSource}`,
    /triggerJsonDownload|JSON\.stringify/,
  );
  assert.match(inspectorSource, /setSourceResultFile\(file\);/);
});

test("switching from a result fetch to a no-result attempt clears loading", () => {
  const effectStart = source.indexOf(
    "useEffect(() => {\n    setSelectedResult(null);",
  );
  const earlyReturn = source.indexOf(
    "if (!selectedAttempt?.job.result_available) return;",
    effectStart,
  );
  const beginFetch = source.indexOf("setResultLoading(true);", earlyReturn);
  const effect = source.slice(effectStart, beginFetch);
  assert.ok(effectStart >= 0 && earlyReturn > effectStart);
  assert.match(effect, /setResultLoading\(false\);/);
  assert.ok(
    effect.indexOf("setResultLoading(false);") <
      effect.indexOf("if (!selectedAttempt?.job.result_available) return;"),
  );
});

test("capture links resolve to the newest available result", () => {
  assert.match(
    source,
    /const requestedCaptureId = searchParams\.get\("capture"\) \?\? "";/,
  );
  assert.match(
    source,
    /job\.capture_id === requestedCaptureId && job\.result_available/,
  );
  assert.match(source, /setSearchParams\(next, \{ replace: true \}\);/);
  assert.match(source, /This capture has no Decode attempt yet\./);
});

test("portable inspection accepts only the canonical run result and exact video", () => {
  assert.match(inspectorSource, /parseDecodeResultDocument\(/);
  assert.match(inspectorSource, /decodeResultVideoSha256\(runResult\)/);
  assert.match(inspectorSource, /crypto\.subtle\.digest\("SHA-256"/);
  assert.match(inspectorSource, /if \(actual !== declaredVideoSha\)/);
  assert.match(inspectorSource, /Video SHA mismatch\./);
  assert.match(
    inspectorSource,
    /Selected \$\{actual\}\. The run remains available, but this video was not paired\./,
  );
  assert.match(contractSource, /"cubed-core\/decode-result"/);
  assert.match(contractSource, /schema_version/);
});

test("loaded runs lead with a Demo-like synchronized workstation", () => {
  const workstation = inspectorSource.indexOf('className="tracker-workstation"');
  const viewer = inspectorSource.indexOf(
    'className="tracker-workstation-viewer"',
    workstation,
  );
  const video = inspectorSource.indexOf(
    "tracker-workstation-video tracker-video-stage-standalone",
    viewer,
  );
  const transport = inspectorSource.indexOf("<RunTransport", video);
  const evidence = inspectorSource.indexOf(
    'className="tracker-workstation-evidence"',
    transport,
  );
  assert.ok(workstation >= 0 && viewer > workstation && video > viewer);
  assert.ok(
    transport > video && evidence > transport,
    "Runs must keep the transport and scrubber under the video before per-frame evidence",
  );
  const transportComponent = inspectorSource.indexOf("function RunTransport");
  const transportRow = inspectorSource.indexOf(
    'className="demo-transport-row tracker-frame-transport"',
    transportComponent,
  );
  const scrubber = inspectorSource.indexOf(
    'className="frame-scrubber tracker-frame-scrubber"',
    transportRow,
  );
  assert.ok(
    transportComponent >= 0 &&
      transportRow > transportComponent &&
      scrubber > transportRow,
  );
  assert.match(inspectorSource, /Aligned streak/);
  assert.match(
    trackerStyles,
    /\.tracker-read-faces \{[\s\S]*?grid-template-columns: repeat\(3,/,
  );
});

test("the shared cube diagram owns its physical sticker palette", () => {
  const cubeStateRule = diagnosticsStyles.match(/\.cube-state\s*\{[\s\S]*?\n\}/);

  assert.ok(cubeStateRule);
  for (const color of ["white", "yellow", "red", "orange", "blue", "green"]) {
    assert.match(cubeStateRule[0], new RegExp(`--cube-${color}:`));
  }
});

test("Runs overlays each sampled 3x3 read beneath the face outlines", () => {
  const stageStart = inspectorSource.indexOf("function RunVideoStage");
  const readCells = inspectorSource.indexOf(
    'className="tracker-read-overlay-cell"',
    stageStart,
  );
  const faceOutline = inspectorSource.indexOf(
    'className="tracker-face-outline"',
    stageStart,
  );
  assert.match(inspectorSource, /function quadPoint\(/);
  assert.match(inspectorSource, /function readCellPoints\(/);
  assert.match(inspectorSource, /reads\.flatMap\(/);
  assert.match(
    inspectorSource,
    /points=\{readCellPoints\(read\.corners, cellIndex\)\}/,
  );
  assert.match(inspectorSource, /fill: sampledLabColor\(lab\)/);
  assert.match(
    inspectorSource,
    /Math\.min\(0\.72, read\.confidence\[cellIndex\] \* 0\.72\)/,
  );
  assert.ok(
    readCells > stageStart && faceOutline > readCells,
    "sampled cells must render before the outer face outlines",
  );
  assert.match(refreshStyles, /\.tracker-read-overlay-cell \{/);
});

test("recording facts and starting scramble stay visible above the workstation", () => {
  const summaryStart = inspectorSource.indexOf(
    'className="decode-run-recording-summary"',
  );
  const summaryEnd = inspectorSource.indexOf("</section>", summaryStart);
  const summary = inspectorSource.slice(summaryStart, summaryEnd);
  assert.ok(summaryStart >= 0 && summaryEnd > summaryStart);
  for (const label of ["Frames", "Frame rate", "Resolution", "Duration"]) {
    assert.match(summary, new RegExp(`<dt>${label}<\\/dt>`));
  }
  assert.match(summary, /pairedCapture\?\.original_filename/);
  assert.match(summary, /<span>Starting scramble<\/span>/);
  assert.match(summary, /<code>\{startingScramble\}<\/code>/);
  assert.ok(
    summaryStart < inspectorSource.indexOf('className="tracker-workstation"'),
  );
  assert.match(
    inspectorSource,
    /recordingFrameCount \/ recordingFps/,
  );
});

test("legacy workspace results fall back to their paired capture receipt", () => {
  assert.match(
    inspectorSource,
    /\{\(runResult\.workstation \|\| pairedCapture\) && \(/,
  );
  assert.match(
    inspectorSource,
    /workstationVideo\?\.encoded\.fps \?\?\s*pairedCapture\?\.video\.actual_fps \?\?\s*pairedCapture\?\.video\.configured_fps/,
  );
  assert.match(
    inspectorSource,
    /workstationVideo\?\.encoded\.frame_count \?\?\s*pairedCapture\?\.video\.frame_count/,
  );
  assert.match(
    inspectorSource,
    /runResult\?\.workstation\?\.initialization\.scramble \?\?\s*pairedCapture\?\.solve\?\.scramble/,
  );
  assert.doesNotMatch(
    inspectorSource,
    /if \(!runResult\?\.workstation\) return;/,
  );
  assert.match(
    inspectorSource,
    /\{!dump && frameWindow && pairedCapture && \(/,
  );
  assert.match(inspectorSource, /decode-run-legacy-workstation/);
  assert.match(inspectorSource, /<RunTransport\s+window=\{frameWindow\}/);
  assert.match(
    inspectorSource,
    /Not embedded in this older result/,
  );
  assert.match(
    refreshStyles,
    /\.decode-run-legacy-workstation \{[\s\S]*?grid-template-columns: minmax\(0, 960px\);[\s\S]*?justify-content: center;/,
  );
  assert.match(
    refreshStyles,
    /\.decode-run-legacy-workstation \.tracker-workstation-viewer,\s*\.decode-run-video-only-workstation \.tracker-workstation-viewer \{[\s\S]*?border-right: 0;/,
  );
  assert.match(
    inspectorSource,
    /runResult\.workstation\?\.initialization \|\| pairedCapture\?\.solve/,
  );
  assert.match(
    inspectorSource,
    /<ReconstructionAtFrame[\s\S]*?startingScramble=\{startingScramble\}/,
  );
});

test("portable older results allow SHA-bound video-only playback", () => {
  assert.match(
    inspectorSource,
    /disabled=\{busy \|\| !runResult \|\| !declaredVideoSha\}/,
  );
  assert.match(
    inspectorSource,
    /\{!dump && !frameWindow && videoStage && \(/,
  );
  assert.match(inspectorSource, /nativeControls=\{!frameWindow\}/);
  assert.match(inspectorSource, /The exact video is paired by SHA\./);
  assert.match(
    inspectorSource,
    /const videoStage =\s*videoUrl \? \(/,
  );
});

test("the quiet run receipt keeps artifact identity without duplicating recording facts", () => {
  const receiptStart = inspectorSource.indexOf(
    'className="tracker-receipt-disclosure"',
  );
  const receiptEnd = inspectorSource.indexOf(
    "</details>",
    receiptStart,
  );
  const receipt = inspectorSource.slice(receiptStart, receiptEnd);
  for (const label of [
    "Run time",
    "Video SHA",
    "Artifact",
    "Config",
  ]) {
    assert.match(receipt, new RegExp(`<dt>${label}<\\/dt>`));
  }
  assert.doesNotMatch(
    receipt,
    /<dt>Recording<\/dt>|<dt>Frame rate<\/dt>|<dt>Starting scramble<\/dt>/,
  );
  assert.match(
    inspectorSource,
    /const elapsedFrom = job\?\.started_at \?\? job\?\.created_at;/,
  );
  assert.match(inspectorSource, /job\?\.result_sha256/);
});

test("the per-frame station surfaces decoded move and timing context", () => {
  assert.match(
    inspectorSource,
    /const currentSequenceEntry =\s*runSequence && currentSequenceIndex >= 0/,
  );
  assert.match(inspectorSource, /decode-run-frame-sequence-context/);
  assert.match(inspectorSource, /<span>Decoded move<\/span>/);
  assert.match(inspectorSource, /currentSequenceEntry\.move/);
  assert.match(inspectorSource, /at frame \$\{currentSequenceEntry\.frame\}/);
  assert.match(inspectorSource, /canonical timing/);
  assert.match(inspectorSource, /canonical timing/);
});

test("the Runs frame clock matches Demo keyboard and seek behavior", () => {
  assert.match(inspectorSource, /case "ArrowLeft":/);
  assert.match(inspectorSource, /case "ArrowRight":/);
  assert.match(inspectorSource, /event\.shiftKey \? 10 : 1/);
  assert.match(inspectorSource, /case "Home":/);
  assert.match(inspectorSource, /case "End":/);
  assert.match(inspectorSource, /case " ":\s*if \(tag === "BUTTON"/);
  assert.match(
    inspectorSource,
    /\(event\.clientX - bounds\.left\) \/ bounds\.width/,
  );
  assert.match(inspectorSource, /onPointerDown=\{seekFromPointer\}/);
  assert.match(
    inspectorSource,
    /Math\.floor\(video\.currentTime \* fps\)/,
  );
  assert.match(inspectorSource, /\(next \+ 0\.5\) \/ fps/);
  assert.doesNotMatch(
    inspectorSource,
    /\(next - frameWindow\[0\] \+ 0\.5\) \/ fps/,
  );
  assert.match(
    inspectorSource,
    /currentTime = \(frameWindow\[0\] \+ 0\.5\) \/ fps/,
  );
});

test("reconstruction and trellis share the primary workstation", () => {
  assert.match(inspectorSource, /className="tracker-inline-diagnostics"/);
  assert.match(inspectorSource, /\{hasReconstruction && \(/);
  assert.match(inspectorSource, /\{hasTrellis && \(/);
  assert.match(inspectorSource, /<h4>Reconstruction<\/h4>/);
  assert.match(inspectorSource, /<h4>Trellis \/ spans<\/h4>/);
  assert.doesNotMatch(inspectorSource, /More diagnostics|tracker-details-disclosure/);
  assert.doesNotMatch(inspectorSource, /<h4>Decoded timing<\/h4>/);
  assert.doesNotMatch(inspectorSource, /Sequence alignment/);
});

test("the reconstruction estimates intermediate states and keeps checkpoints authoritative", () => {
  assert.match(inspectorSource, /result\.workstation\?\.reconstruction/);
  assert.match(inspectorSource, /reconstructionTimeline\.moves\.reduce/);
  assert.match(inspectorSource, /estimatedMovesTowardCheckpoint/);
  assert.match(inspectorSource, /Math\.floor\(progress \* moveCount\)/);
  assert.match(inspectorSource, /estimatedMoveCount > 0/);
  assert.match(inspectorSource, /estimated move .* next checkpoint/);
  assert.match(inspectorSource, /starting scramble/);
  assert.match(inspectorSource, /decoder checkpoint/);
  assert.match(inspectorSource, /: 0;/);
  assert.match(inspectorSource, /no decoder checkpoints/);
  assert.doesNotMatch(inspectorSource, /Final decoded state/);
  assert.match(contractSource, /decoder-checkpoint/);
});

test("the left video stage preserves its measured ratio within both size caps", () => {
  const stageRule = trackerStyles.match(
    /\.tracker-workstation-video \.tracker-video-stage \{[\s\S]*?\n\}/,
  );

  assert.ok(stageRule);
  assert.match(
    trackerStyles,
    /\.tracker-workstation \{[\s\S]*?--tracker-stage-cap: min\(72vh, 720px\);/,
  );
  assert.match(
    stageRule[0],
    /width: min\([\s\S]*?var\(--tracker-stage-cap\) \* var\(--tracker-stage-ratio, 0\.5625\)/,
  );
  assert.match(stageRule[0], /height: auto;/);
  assert.match(stageRule[0], /max-height: var\(--tracker-stage-cap\);/);
  assert.match(
    stageRule[0],
    /aspect-ratio: var\(--tracker-stage-ratio, 0\.5625\);/,
  );
});

test("Trellis path, orientation, and score use fixed table tracks", () => {
  assert.match(trellisSource, /<col className="dash-path-column" \/>/);
  assert.match(trellisSource, /<col className="dash-om-column" \/>/);
  assert.match(trellisSource, /<col className="dash-score-column" \/>/);
  assert.match(
    diagnosticsStyles,
    /\.dash-beam-table \{[\s\S]*?table-layout: fixed;/,
  );
  assert.match(
    diagnosticsStyles,
    /\.dash-beam-table \.dash-score-column \{[\s\S]*?width: 72px;/,
  );
});

test("sampled reads sit beside a two-by-two frame metric grid", () => {
  const reads = inspectorSource.indexOf('className="tracker-current-reads"');
  const metrics = inspectorSource.indexOf(
    'className="tracker-current-status"',
    reads,
  );

  assert.match(inspectorSource, /className=\{`tracker-frame-summary\$\{/);
  assert.ok(reads >= 0 && metrics > reads);
  assert.match(
    trackerStyles,
    /\.tracker-current-status \.tracker-frame-metrics \{[\s\S]*?grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/,
  );
});

test("persisted BLE reference stays diagnostic-only inside the workstation", () => {
  assert.match(
    contractSource,
    /\["workstation", "ground_truth_diagnostic"\]/,
  );
  assert.match(contractSource, /validateGroundTruthDiagnostic\(/);
  assert.match(
    contractSource,
    /ground_truth_diagnostic\.capture_id must match recording_id/,
  );
  assert.match(
    contractSource,
    /ground_truth_diagnostic\.comparison\.distance must match its edit operations/,
  );

  const primaryPlayback = inspectorSource.indexOf("{dump && (");
  const inlineDiagnostics = inspectorSource.indexOf(
    'className="tracker-inline-diagnostics"',
    primaryPlayback,
  );
  const bleReference = inspectorSource.indexOf(
    "{runResult.ground_truth_diagnostic && (",
    inlineDiagnostics,
  );
  assert.ok(
    primaryPlayback >= 0 &&
      inlineDiagnostics > primaryPlayback &&
      bleReference > inlineDiagnostics,
    "BLE comparison must remain in the primary workstation after reconstruction and trellis",
  );
  for (const copy of [
    "BLE reference",
    "Published smart-cube sequence",
    "Edit distance",
    "Diagnostic only",
    "Move order only · no video-frame timing.",
  ]) {
    assert.match(inspectorSource, new RegExp(copy.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
  assert.match(inspectorSource, /diagnostic\.comparison\.ops\.map/);
  assert.doesNotMatch(
    `${apiSource}\n${inspectorSource}`,
    /diagnostics\/editdist|fetchDecodeEditDistDiagnostic/,
  );
});

test("mismatched stored timing never indexes the canonical decoded moves", () => {
  assert.match(contractSource, /export function canonicalDecodeSequence\(/);
  assert.match(
    contractSource,
    /sequence\.moves\.length !== result\.moves\.length/,
  );
  assert.match(
    contractSource,
    /entry\.move === result\.moves\[index\]/,
  );
  assert.match(
    contractSource,
    /const sequence = canonicalDecodeSequence\(result\)\?\.moves \?\? \[\];/,
  );
  assert.match(inspectorSource, /const sequenceMismatch = Boolean\(/);
  assert.match(inspectorSource, /Stored move timing does not\s*match the decoded move list/);
});

test("shared selects and disclosures keep stable intrinsic alignment", () => {
  assert.match(refreshStyles, /\.lab-shell select \{[\s\S]*?font: inherit;/);
  assert.match(
    refreshStyles,
    /\.lab-shell details \{[\s\S]*?align-self: start;/,
  );
  assert.match(
    refreshStyles,
    /\.lab-shell details \{[\s\S]*?overflow-anchor: none;/,
  );
});
