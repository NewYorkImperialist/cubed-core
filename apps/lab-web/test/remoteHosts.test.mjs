import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const sourceRoot = resolve(import.meta.dirname, "../src");

function read(relative) {
  return readFileSync(resolve(sourceRoot, relative), "utf8");
}

const remoteHostsSource = read("remoteHosts.ts");
const selectorSource = read("components/ComputeTargetSelector.tsx");
const selectorStyles = read("components/ComputeTargetSelector.css");
const decodeSource = read("components/DecodeStage.tsx");
const pageSource = read("components/ReconstructWorkbench.tsx");
const apiSource = read("api.ts");
const decodeApiSource = read("decodeApi.ts");
const typesSource = read("types.ts");

test("fetchRemoteHosts reads the admin-gated collection route", () => {
  assert.match(apiSource, /export function fetchRemoteHosts\(/);
  assert.match(apiSource, /"\/api\/remote-hosts"/);
});

test("the compute selector leads with explicit Local CUDA on this workbench", () => {
  const localAt = selectorSource.indexOf('<option value="" disabled={localUnavailable}>');
  const hostsAt = selectorSource.indexOf("hosts.map((host)");
  assert.ok(localAt >= 0 && hostsAt > localAt);
  assert.match(selectorSource, />\s*Local CUDA \(this workbench\)\s*<\/option>/);
  assert.match(
    selectorSource,
    /hosts\.map\(\(host\) => \(\s*<option key=\{host\.id\} value=\{host\.id\}>\s*\{host\.label\}/,
  );
});

test("empty target means local and an explicit host id is submitted remotely", () => {
  assert.match(
    decodeSource,
    /submitDecodeJob\(\s*captureId,\s*remoteHostId \|\| undefined,\s*controller\.signal,\s*\)/,
  );
  assert.match(selectorSource, /Empty is an intentional value: it means\s*\* Local CUDA and omits remote_host/);
  assert.match(
    decodeApiSource,
    /\?remote_host=\$\{encodeURIComponent\(remoteHostId\)\}/,
  );
});

test("remote hosts remain opt-in and stale choices return to local", () => {
  const fallbackAt = remoteHostsSource.indexOf(
    "// Empty means Local CUDA and is the deliberate default.",
  );
  assert.ok(fallbackAt >= 0);
  const block = remoteHostsSource.slice(fallbackAt, fallbackAt + 700);
  assert.match(block, /if \(skip \|\| loading \|\| !selectedId\) return;/);
  assert.match(block, /hosts\.some\(\(host\) => host\.id === selectedId\)/);
  assert.match(block, /setSelectedId\(""\);/);
  assert.doesNotMatch(block, /hosts\[0\]\.id/);
  assert.doesNotMatch(block, /\? envDefaultId/);
});

test("an explicit compute choice persists without changing the default", () => {
  assert.match(
    remoteHostsSource,
    /export const REMOTE_HOST_STORAGE_KEY = "cubed-core-remote-host-id";/,
  );
  assert.match(remoteHostsSource, /void fetchRemoteHosts\(\)/);
  assert.match(
    remoteHostsSource,
    /window\.localStorage\.setItem\(REMOTE_HOST_STORAGE_KEY, value\)/,
  );
  assert.match(remoteHostsSource, /else window\.localStorage\.removeItem/);
});

test("Local CUDA uses capability data and remote loading does not hide it", () => {
  assert.match(selectorSource, /localGpu: GpuCapability \| null;/);
  assert.match(selectorSource, /const localUnavailable = localGpu !== null && !localGpu\.available;/);
  assert.match(selectorSource, /disabled=\{localUnavailable\}/);
  assert.doesNotMatch(selectorSource, /if \(loading\) return/);
  assert.match(selectorSource, /Checking configured remote hosts/);
  assert.match(decodeSource, /localGpu=\{capabilities\?\.gpu \?\? null\}/);
  assert.match(decodeSource, /computeTargetUnavailableReason/);
});

test("a restored result stays primary when its previous compute target is unavailable", () => {
  assert.match(
    decodeSource,
    /computeTargetUnavailableReason !== null && decodeResult === null/,
  );
  assert.match(
    decodeSource,
    /decodeReady =[\s\S]*?computeTargetUnavailableReason === null/,
  );
  assert.match(
    decodeSource,
    /disabled=\{\s*!decodeReady \|\|\s*decodeRunning/,
  );
});

test("an empty remote list keeps local visible and links to host setup", () => {
  assert.match(selectorSource, /hosts\.length === 0/);
  assert.match(selectorSource, /workspace\/remote-hosts\.json/);
  assert.match(selectorSource, /routeGuideDocumentHref\("docs\/CLOUD_GPU\.md"\)/);
});

test("the selector provides a compact local and remote GPU setup route", () => {
  assert.match(selectorSource, /<details className="compute-target-setup">/);
  assert.match(selectorSource, /Set up a GPU/);
  assert.match(selectorSource, /make download-assets/);
  assert.match(selectorSource, /make bootstrap-research-gpu/);
  assert.match(
    selectorSource,
    /CUBED_CORE_DECODE_MODE=native make workbench-decode-gpu/,
  );
  assert.match(selectorSource, /attach your SSH public key/);
  assert.match(selectorSource, /BatchMode=yes/);
  assert.match(selectorSource, /provision_gpu_box\.sh/);
  assert.match(selectorSource, /--append-env/);
  assert.match(selectorSource, /<code>make hub<\/code>/);
  assert.match(selectorSource, /Refresh readiness/);
  assert.match(selectorSource, /live Decode commands require Linux or WSL2/);
  assert.match(
    selectorSource,
    /A listed target\s+is configured, not proven ready/,
  );
  assert.match(selectorSource, /routeGuideDocumentHref\("docs\/CLOUD_GPU\.md"\)/);
  assert.match(selectorStyles, /\.compute-target-setup > summary/);
  assert.match(selectorStyles, /border-radius/);
});

test("the compute selector never exposes a configured host address", () => {
  assert.doesNotMatch(selectorSource, /selectedHost\.ssh_dest/);
  assert.doesNotMatch(selectorSource, /selectedHost\.ssh_port/);
  assert.doesNotMatch(typesSource, /interface RemoteHost \{[\s\S]*?ssh_dest/);
  assert.doesNotMatch(typesSource, /interface RemoteHost \{[\s\S]*?ssh_port/);
  assert.match(
    selectorSource,
    /Remote target configured · run readiness to verify it\./,
  );
});

test("the compute target locks for the duration of a decode job", () => {
  assert.match(selectorSource, /disabled = false,/);
  assert.match(selectorSource, /disabled=\{disabled\}/);
  assert.match(selectorSource, /keeps the compute target it started with/);
  assert.match(decodeSource, /disabled=\{decodeRunning\}/);
});

test("decode refreshes configured hosts after a rejected remote id", () => {
  assert.match(decodeSource, /isRemoteHostRejection\(reason\)/);
  assert.match(decodeSource, /onRemoteHostsRefresh\?\.\(\);/);
  assert.match(remoteHostsSource, /refresh: \(\) => void;/);
  assert.match(remoteHostsSource, /refresh: load,/);
});

test("the decode page owns one compute selection", () => {
  assert.equal(
    (pageSource.match(/useRemoteHostSelection\(\)/g) ?? []).length,
    1,
  );
  assert.match(pageSource, /const computeSelection = useRemoteHostSelection\(\);/);
  assert.match(
    pageSource,
    /remoteHosts=\{computeSelection\.hosts\}[\s\S]{0,60}remoteHostId=\{computeSelection\.selectedId\}[\s\S]{0,80}onRemoteHostIdChange=\{computeSelection\.setSelectedId\}/,
  );
});
