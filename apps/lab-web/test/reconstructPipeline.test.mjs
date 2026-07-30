import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const sourceRoot = resolve(import.meta.dirname, "../src");

function read(relative) {
  return readFileSync(resolve(sourceRoot, relative), "utf8");
}

const pageSource = read("components/ReconstructWorkbench.tsx");
const decodeSource = read("components/DecodeStage.tsx");
const decodeCssSource = read("components/DecodeStage.css");
const prepareSource = read("components/PrepareForDecodeCard.tsx");
const decodeApiSource = read("decodeApi.ts");
const typesSource = read("types.ts");
const apiSource = read("api.ts");

function countMatches(source, pattern) {
  return (source.match(pattern) ?? []).length;
}

// The decode result block is the part of the page that speaks about the
// output. It is checked separately so its wording stays inside the evidence
// boundary this repo keeps.
function decodeResultBlock() {
  const start = decodeSource.indexOf("{decodeResult && (");
  const end = decodeSource.indexOf("\n    </section>", start);
  assert.ok(start >= 0, "the decode result panel was not found");
  assert.ok(end > start, "the decode stage must close after the result panel");
  return decodeSource.slice(start, end);
}

test("the page reads the capture list and the capabilities exactly once", () => {
  assert.equal(countMatches(pageSource, /fetchCaptures\(/g), 1);
  assert.equal(countMatches(pageSource, /fetchCapabilities\(/g), 1);
});

test("Decode starts empty while an explicit URL or current capture is restored", () => {
  assert.match(
    pageSource,
    /explicitCaptureIdFromParams\(\s*initialCaptureParamsRef\.current,\s*payload\.captures,\s*\)/,
  );
  assert.match(
    pageSource,
    /current &&\s*payload\.captures\.some\(\(capture\) => capture\.capture_id === current\)/,
  );
  assert.match(pageSource, /"Choose a capture…"/);
});

test("the pipeline page links to but does not embed the published demo", () => {
  assert.doesNotMatch(pageSource, /PublishedDemoReplay/);
  assert.doesNotMatch(pageSource, /published-demo/);
  assert.match(pageSource, /to="\/demo"/);
  assert.match(pageSource, /<h1>Decode<\/h1>/);
});

test("the page owns the only capture dropdown", () => {
  assert.equal(countMatches(pageSource, /<select/g), 1);
  assert.match(pageSource, /id="reconstruct-capture"/);
  assert.match(pageSource, /value=\{captureId\}/);
  assert.match(pageSource, /onChange=\{\(event\) => selectCaptureId\(event\.target\.value\)\}/);
});

test("the capture picker quietly groups personal and published recordings", () => {
  const labelsStart = pageSource.indexOf(
    "const duplicateCaptureFilenames = useMemo",
  );
  const labelsEnd = pageSource.indexOf(
    "const selectedCapture = useMemo",
    labelsStart,
  );
  const labelsBlock = pageSource.slice(labelsStart, labelsEnd);
  assert.ok(labelsStart >= 0 && labelsEnd > labelsStart);
  assert.match(labelsBlock, /new Map<string, number>\(\)/);
  assert.match(labelsBlock, /\.filter\(\(\[, count\]\) => count > 1\)/);

  assert.match(pageSource, /orderDecodeCaptures\(captures\)/);
  const optionsStart = pageSource.indexOf(
    '{captureGroups.yourRecordings.length > 0 && (',
  );
  const optionsEnd = pageSource.indexOf("</select>", optionsStart);
  const optionsBlock = pageSource.slice(optionsStart, optionsEnd);
  assert.ok(optionsStart >= 0 && optionsEnd > optionsStart);
  assert.match(optionsBlock, /<optgroup label="Your recordings">/);
  assert.match(optionsBlock, /<optgroup label="Published dataset">/);
  assert.match(optionsBlock, /captureGroups\.yourRecordings\.map/);
  assert.match(optionsBlock, /captureGroups\.publishedDataset\.map/);
  assert.match(optionsBlock, /value=\{capture\.capture_id\}/);
  assert.match(optionsBlock, /\{capture\.original_filename\}/);
  assert.match(
    optionsBlock,
    /duplicateCaptureFilenames\.has\(capture\.original_filename\)[\s\S]*captureDateLabel\(capture\.created_at\)/,
  );
  assert.doesNotMatch(
    optionsBlock,
    /capture_id\.slice|capture\.state|incomplete|sealed/,
  );
});

test("the primary cards are exactly choose, prepare, and decode", () => {
  const step1 = pageSource.indexOf('title="Choose a recording"');
  const step2 = pageSource.indexOf('title="Prepare for decode"');
  const step3Locked = pageSource.indexOf('title="Decode and replay"');
  assert.ok(step1 >= 0, "step 1 header not found");
  assert.ok(step2 > step1, "step 2 header not found after step 1");
  assert.ok(step3Locked > step2, "step 3 header not found after step 2");

  assert.match(pageSource, /<StepHeader\s+step=\{1\}/);
  assert.match(pageSource, /<StepHeader\s+step=\{2\}/);
  assert.match(pageSource, /<StepHeader\s+step=\{3\}/);
  assert.equal(countMatches(pageSource, /<StepHeader/g), 3);

  assert.match(decodeSource, /<p className="station-code">Step 3<\/p>/);
  assert.match(decodeSource, /<h2>Decode and replay<\/h2>/);
});

test("the page omits redundant route-guide, progress, and status summaries", () => {
  assert.doesNotMatch(pageSource, /RouteGuide/);
  assert.doesNotMatch(pageSource, /ProgressRail/);
  assert.doesNotMatch(pageSource, /reconstruct-status-badges/);
  assert.doesNotMatch(pageSource, /capability-badge/);
});

test("locked steps say what choosing a recording unlocks", () => {
  assert.match(pageSource, /Choose a recording above to prepare it\./);
  assert.match(
    pageSource,
    /Choose a recording above to check readiness and run the decoder\./,
  );
});

test("step 1 shows a video preview reusing the existing media-ticket route, not a new one", () => {
  assert.match(pageSource, /createCaptureMediaTicket\(captureId\)/);
  assert.match(pageSource, /<video\s+className="reconstruct-capture-preview-video"\s+src=\{previewUrl\}/);
});

test("published dataset onboarding uses the canonical preparation path", () => {
  const start = pageSource.indexOf('className="decode-dataset-onboarding"');
  const end = pageSource.indexOf("</aside>", start);
  const onboarding = pageSource.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(onboarding, /published Hugging Face dataset/);
  assert.match(
    onboarding,
    /make download-dataset download-assets/,
  );
  assert.equal(countMatches(onboarding, /<code>/g), 1);
  assert.doesNotMatch(onboarding, /cubed-core import-video|cube-session|IMU/);
  assert.match(onboarding, /downloaded videos appear here automatically/);
  assert.match(onboarding, /frame-zero-scrambled\s+recording/);
  assert.match(onboarding, /routeGuideDocumentHref\("docs\/DATASET\.md"\)/);
});

test("the decode card keeps its direct anchor without a tracker anchor", () => {
  assert.match(pageSource, /id="reconstruct-decode"/);
  assert.doesNotMatch(pageSource, /id="reconstruct-track"/);
});

test("capture switches clear a running standard decode instead of leaving the card stuck", () => {
  const resetStart = decodeSource.indexOf("// Capture-scoped job/result state resets");
  const resetEnd = decodeSource.indexOf("const loadPreflight", resetStart);
  const resetBlock = decodeSource.slice(resetStart, resetEnd);
  assert.match(resetBlock, /setDecodeRunning\(false\);/);
  assert.match(resetBlock, /setDecodePollFailed\(false\);/);
  assert.match(resetBlock, /setDecodeJob\(null\);/);
  assert.match(resetBlock, /setDecodeResult\(null\);/);
});

test("a capture locked for label is surfaced as an irreversible decode blocker", () => {
  assert.match(prepareSource, /sealedForAnotherPurpose/);
  assert.match(prepareSource, /That lock cannot be changed to Decode/);
  assert.match(prepareSource, /"Locked for label\."/);
});

test("an empty workspace keeps a link to the import route", () => {
  assert.match(pageSource, /const noCaptures =\s*\n\s*captureLoadState === "success" && captures\.length === 0;/);
  assert.match(pageSource, /\) : noCaptures \? \(\s*<Link className="text-button" to="\/import">/);
});

test("a disappearing capture locks Prepare and Decode instead of casting null", () => {
  assert.equal(
    countMatches(pageSource, /\{selectedCapture === null \? \(/g),
    2,
  );
  assert.match(
    pageSource,
    /\{selectedCapture === null \? \([\s\S]*?<PrepareForDecodeCard/,
  );
});

test("the page owns one compute selection and passes it only to decode", () => {
  assert.equal(countMatches(pageSource, /useRemoteHostSelection\(\)/g), 1);
  assert.match(pageSource, /const computeSelection = useRemoteHostSelection\(\);/);
  assert.match(
    pageSource,
    /<DecodeStage[\s\S]*?remoteHosts=\{computeSelection\.hosts\}[\s\S]*?remoteHostId=\{computeSelection\.selectedId\}[\s\S]*?onRemoteHostIdChange=\{computeSelection\.setSelectedId\}/,
  );
});

test("the page keeps one visible capability failure and retry", () => {
  assert.match(pageSource, /\{capabilitiesError && \(/);
  assert.match(pageSource, /Retry capability check/);
  assert.match(pageSource, /showCapabilitiesError=\{false\}/);
});

test("step 2 wraps PrepareForDecodeCard with the page's capture list and update callback", () => {
  assert.match(
    pageSource,
    /<PrepareForDecodeCard\s+capture=\{selectedCapture\}\s+captures=\{captures\}\s+onUpdated=\{updateCapture\}/,
  );
  assert.match(pageSource, /const updateCapture = useCallback\(/);
});

test("normal preparation omits teacher sidecars while the advanced API route remains", () => {
  assert.doesNotMatch(prepareSource, /Attach the smart-cube record|attachTeacher|capture\.teacher/);
  assert.match(apiSource, /`\/api\/captures\/\$\{encodeURIComponent\(captureId\)\}\/sidecars\/\$\{kind\}`/);
  assert.match(typesSource, /export type SidecarKind = "calibration" \| "teacher" \| "ble-raw" \| "phone-imu";/);
  assert.match(prepareSource, /type RowBusy = "calibration" \| "seal" \| null;/);
  assert.match(
    prepareSource,
    /<span className="prepare-row-letter" aria-hidden="true">\s*b\s*<\/span>[\s\S]*?<strong>Lock video \+ scramble<\/strong>/,
  );
});

test("the calibration row offers the shared dataset calibration with an honest portability warning", () => {
  assert.match(prepareSource, /const BUNDLED_SOURCE = "bundled";/);
  assert.match(prepareSource, /const NO_CALIBRATION_SOURCE = "";/);
  assert.match(prepareSource, /useState<string>\(\s*NO_CALIBRATION_SOURCE,\s*\)/);
  assert.match(prepareSource, /Published dataset shared calibration/);
  assert.match(prepareSource, /This shared calibration is allowed for any video\./);
  assert.match(prepareSource, /different\s+cube, camera, lens, or lighting/);
  assert.match(prepareSource, /otherCalibratedCaptures\.map\(/);
  assert.match(
    prepareSource,
    /From \{other\.original_filename\} · \{shortDate\(other\.created_at\)\}/,
  );
  assert.doesNotMatch(prepareSource, /From capture \{other\.capture_id/);
  assert.match(prepareSource, /Upload a file/);
  assert.match(
    apiSource,
    /`\/api\/captures\/\$\{encodeURIComponent\(captureId\)\}\/sidecars\/calibration\/reuse`/,
  );
  assert.match(apiSource, /export function attachReusedCalibration\(/);
});

test("an attached calibration shows its human source and hides internal details", () => {
  assert.match(
    prepareSource,
    /function attachedCalibrationLabel\(capture: CaptureReceipt\): string/,
  );
  assert.match(prepareSource, /capture\.calibration\?\.display_name\?\.trim\(\)/);
  assert.match(prepareSource, /: "Calibration";/);
  assert.match(
    prepareSource,
    /<span className="prepare-row-check">Attached<\/span>/,
  );
  assert.match(prepareSource, /className="prepare-row-name"/);
  assert.match(
    prepareSource,
    /\{attachedCalibrationLabel\(capture\)\}/,
  );
  assert.match(typesSource, /display_name\?: string;/);
  assert.doesNotMatch(prepareSource, /capture\.calibration\.(?:kind|path|sha256)/);
  assert.doesNotMatch(prepareSource, /shortHash/);
});

test("prepare warns on readable nonstandard media and blocks unreadable or native-240 inputs", () => {
  assert.match(prepareSource, /const metadataKnown =/);
  assert.match(
    prepareSource,
    /measuredFps! >= 110 && measuredFps! <= 121/,
  );
  assert.match(
    prepareSource,
    /Math\.min\(measuredWidth!, measuredHeight!\) >= 1080/,
  );
  assert.match(
    prepareSource,
    /const mediaReady = metadataKnown && !native240Source;/,
  );
  assert.match(prepareSource, /const nonstandardMedia =/);
  assert.match(prepareSource, /You can continue,/);
  assert.match(prepareSource, /Decode needs readable frame-rate and resolution metadata\./);
  assert.match(
    prepareSource,
    /Preserve this native 220–242 fps original and create its linked\s+120 fps derivative/,
  );
});

test("decode locking keeps calibration replaceable and explains per-run snapshots", () => {
  assert.match(prepareSource, /const calibrationEditable = !sealed \|\| sealedForDecode;/);
  assert.match(prepareSource, /\{calibrationEditable && capture\.calibration && \(/);
  assert.match(prepareSource, /\{calibrationEditable && \(/);
  assert.match(prepareSource, /Calibration stays\s+replaceable/);
  assert.match(prepareSource, /every run saves the exact calibration it used/);
  assert.match(prepareSource, /Lock for decode/);
  assert.match(decodeSource, /This attempt saves its own\s+calibration snapshot/);
  assert.match(decodeSource, /"Run decode again"/);
});

test("the Decode capability is part of the required service contract", () => {
  assert.match(typesSource, /decode_jobs: DecodeJobCapability;/);
  assert.match(decodeSource, /const decodeCapability = capabilities\?\.decode_jobs \?\? null;/);
  assert.match(decodeSource, /if \(!decodeCapability\) \{/);
  assert.match(decodeSource, /incomplete Decode capability response/);
  assert.match(decodeSource, /decodeCapability\.reason \?\?/);

  // No dead control: the run button lives in the branch after the blocked
  // and not-ready branches, so an unavailable route renders no button.
  const blockedStart = decodeSource.indexOf(") : decodeUnavailableReason !== null ? (");
  const runStart = decodeSource.indexOf(">Run decode<");
  const runLabel = decodeSource.indexOf('? "Decode running…"');
  assert.ok(blockedStart >= 0, "the capability-absent branch was not found");
  assert.equal(runStart, -1, "the run label must come from the busy ternary");
  assert.ok(runLabel > blockedStart, "the run control must follow the blocked branch");
});

test("a capture that is not ready shows a compact blocked line with the full checks behind a disclosure", () => {
  assert.match(decodeSource, /fetchDecodePreflight\(captureId, controller\.signal\)/);
  assert.match(decodeSource, /\[captureId, captureRevision, loadPreflight\]/);
  assert.match(decodeSource, /Retry readiness check/);
  assert.match(decodeSource, /Refresh readiness/);
  assert.match(decodeSource, /\) : !preflight\.ready \? \(/);
  assert.match(decodeSource, /decode-readiness-line decode-readiness-blocked/);
  assert.match(decodeSource, /<strong>Blocked:<\/strong>/);
  assert.match(decodeSource, /<details className="decode-check-disclosure">/);
  assert.match(decodeSource, /\{preflight\.missing\.map\(\(item\) => \(/);
  assert.match(decodeSource, /detailForRequirement\(item\)/);
  assert.match(decodeSource, /to=\{captureHref\("\/import", captureId\)\}/);
});

test("a ready capture shows a green one-line readiness verdict before the run button", () => {
  assert.match(decodeSource, /decode-readiness-line decode-readiness-ready/);
  assert.match(decodeSource, /<strong>Ready to decode\.<\/strong>/);
  assert.match(decodeSource, /preflight\.warnings\.map\(\(warning\) =>/);
});

test("the ready strip aligns status, copy, and actions without squeezing the verdict", () => {
  assert.match(decodeSource, /className="decode-readiness-copy"/);
  assert.match(decodeSource, /className="decode-readiness-actions"/);
  assert.match(
    decodeCssSource,
    /\.decode-stage-run\s*\{[\s\S]*?grid-template-columns: max-content minmax\(18rem, 1fr\) max-content;/,
  );
  assert.match(
    decodeCssSource,
    /\.decode-stage-run \.decode-readiness-line\s*\{[\s\S]*?white-space: nowrap;/,
  );
  assert.match(
    decodeCssSource,
    /@media \(max-width: 1220px\)\s*\{[\s\S]*?\.decode-readiness-actions\s*\{[\s\S]*?grid-column: 1 \/ -1;/,
  );
});

test("the run branch polls with a generation-guarded abortable loop", () => {
  assert.match(decodeApiSource, /export function submitDecodeJob\(/);
  assert.match(decodeApiSource, /export function fetchDecodeJobStatus\(/);
  assert.match(decodeApiSource, /export function fetchDecodeJobResult\(/);
  assert.match(decodeApiSource, /\/api\/captures\/\$\{encodeURIComponent\(captureId\)\}\/decode-jobs/);
  assert.match(decodeApiSource, /\/api\/decode\/jobs\/\$\{encodeURIComponent\(jobId\)\}/);

  const loopStart = decodeSource.indexOf("while (!isTerminalDecodeState(job.status))");
  assert.ok(loopStart >= 0, "the decode poll loop was not found");
  const loop = decodeSource.slice(loopStart, decodeSource.indexOf("\n      }", loopStart));
  assert.match(loop, /await waitForJobPoll\(controller\.signal\)/);
  assert.match(loop, /await fetchDecodeJobStatus\(job\.job_id, controller\.signal\)/);
  assert.match(loop, /decodeGenerationRef\.current !== generation/);
  assert.match(loop, /setDecodeJob\(job\)/);
});

test("the decode console renders Decode stages and logs without extra job provenance", () => {
  assert.match(
    decodeSource,
    /import \{\s*decodeStageChips,\s*decodeStageProgressLabel,\s*\} from "\.\.\/decodeStages";/,
  );
  assert.match(decodeSource, /decodeStageChips\(decodeJob\?\.stages_seen, decodeJob\?\.stage\)/);
  assert.match(decodeSource, /className="decode-job-console"/);
  assert.match(decodeSource, /decode-stage-chip decode-stage-\$\{chip.state\}/);
  assert.match(decodeSource, /className="decode-job-log"/);
  assert.match(decodeSource, /decodeJob\.log\.slice\(-65_536\)/);
  assert.match(
    decodeSource,
    /<details className="decode-job-log-disclosure">\s*<summary>Runner output<\/summary>/,
  );
  assert.doesNotMatch(decodeSource, /<dt>Runner<\/dt>/);
  assert.doesNotMatch(decodeSource, /Configured new-run route|configuredRouteLabel/);
  assert.match(decodeSource, /<ComputeTargetSelector/);
});

test("a failed or timed-out job reports the error beside the log", () => {
  assert.match(decodeSource, /if \(job\.status !== "succeeded"\) \{/);
  assert.match(decodeSource, /jobStateLabel\(job\.status\)/);
  assert.match(decodeSource, /\{decodeError && \(/);
  assert.match(decodeSource, /className="tool-message tool-message-error"/);
});

test("a poll failure hides stale progress instead of leaving it beside the error", () => {
  assert.match(decodeSource, /const \[decodePollFailed, setDecodePollFailed\] = useState\(false\);/);
  assert.match(decodeSource, /decodePollFailed \? \(/);
});

test("the one standard decode lane locks its compute target while running", () => {
  assert.match(decodeSource, /<ComputeTargetSelector[\s\S]*?disabled=\{decodeRunning\}/);
  assert.match(
    decodeSource,
    /Local CUDA runs on this workbench\. Configured hosts use the remote\s+GPU bridge\./,
  );
  assert.match(decodeSource, /localGpu=\{capabilities\?\.gpu \?\? null\}/);
  assert.equal(
    countMatches(decodeSource, /onClick=\{\(\) => void runDecode\(\)\}/g),
    1,
  );
  assert.doesNotMatch(decodeSource, /experimentalBusy|Run experimental pipeline/);
});

test("the result panel leads with a compact outcome and sends inspection to Runs", () => {
  const block = decodeResultBlock();
  assert.match(block, /Decode outcome/);
  assert.match(block, /decoded move/);
  assert.match(block, /The job completed on this input\./);
  assert.match(block, /The sequence replayed to solved\./);
  assert.match(block, /Inspect the run alongside its recording in Runs\./);
  assert.match(block, /Server replay/);
  assert.match(block, /replayReached/);
  assert.match(block, /Open in Runs/);
});

test("the result panel carries compact provenance and the canonical portable download", () => {
  const block = decodeResultBlock();
  assert.match(block, /decodeResult\.profile/);
  assert.match(block, /decodeResult\.config\.cfg_hash/);
  assert.doesNotMatch(block, /configuredRouteLabel|runnerLabel/);
  assert.doesNotMatch(block, /Copy moves/);
  assert.match(block, /Download run JSON/);
  assert.match(decodeSource, /await fetchDecodeJobResultBlob\(decodeJob\.job_id\)/);
  assert.match(decodeSource, /triggerBlobDownload\(/);
  assert.doesNotMatch(decodeSource, /triggerJsonDownload|JSON\.stringify/);
  assert.match(
    decodeSource,
    /`cubed-core-run-\$\{decodeJob\.job_id\}\.json`/,
  );
});

test("the result panel never claims a measure of the decoded output", () => {
  const block = decodeResultBlock();
  assert.ok(!block.includes("accuracy"), "the result panel must not say accuracy");
  assert.doesNotMatch(block, /accuracy/i);
  assert.doesNotMatch(block, /verified/i);
  assert.doesNotMatch(block, /\bcorrect/i);
});

test("an abstained result remains downloadable without inventing a replay", () => {
  const block = decodeResultBlock();
  assert.match(decodeSource, /const resultCompleted = decodeResult\?\.status === "completed";/);
  assert.match(block, /: "The run abstained\."/);
  assert.match(block, /The run JSON preserves that outcome\./);
  assert.match(block, /Download run JSON/);
  assert.doesNotMatch(block, /<MoveReplay|decode-move-sequence/);
});

test("server replay is a compact receipt field for either result outcome", () => {
  const block = decodeResultBlock();
  assert.match(block, /<dt>Server replay<\/dt>/);
  assert.match(block, /replayReached === null/);
  assert.match(block, /\? "Solved"\s*: "Not solved"/);
});

test("Decode completion omits the duplicate replay workstation and teacher diagnostics", () => {
  const block = decodeResultBlock();
  assert.doesNotMatch(
    block,
    /<MoveReplay|decode-editdist|Edit distance|smart cube record/,
  );
});

test("the new decode surfaces stay on the theme variables", () => {
  const styles = readFileSync(
    resolve(sourceRoot, "components/DecodeStage.css"),
    "utf8",
  );
  const added = styles.slice(styles.indexOf(".decode-stage-card {"));
  assert.match(added, /var\(--surface-solid\)/);
  assert.doesNotMatch(added, /#[0-9a-fA-F]{3,8}\b/);
  assert.doesNotMatch(
    styles,
    /decode-preflight-page|decode-config-card|decode-result-endpoint|decode-move-sequence/,
  );

});

test("the reconstruct page's own new styles stay on theme variables", () => {
  const styles = readFileSync(resolve(sourceRoot, "styles.css"), "utf8");
  const added = styles.slice(
    styles.indexOf(".reconstruct-capture-bar {"),
    styles.indexOf("@media (max-width: 900px)"),
  );
  assert.doesNotMatch(added, /#[0-9a-fA-F]{3,8}\b/);

  const prepareStyles = readFileSync(
    resolve(sourceRoot, "components/PrepareForDecodeCard.css"),
    "utf8",
  );
  assert.doesNotMatch(prepareStyles, /#[0-9a-fA-F]{3,8}\b/);
});
