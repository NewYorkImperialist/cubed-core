import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { parseDecodeResultDocument } from "../../src/decodeRunContracts";
import { cubeTrajectory } from "../../src/lib/cubePerms";

const CAPTURE_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CAPTURE_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const JOB_A = "job-for-capture-a";
const JOB_A_OLD = "older-failed-job-for-capture-a";
const JOB_B = "job-for-capture-b";
const JOB_LEGACY = "legacy-result-without-workstation";

const TEST_CUBE_COLORS = ["white", "yellow", "red", "orange", "blue", "green"];
const TEST_FACE_SLICES = [
  ["up", 0],
  ["right", 9],
  ["front", 18],
  ["down", 27],
  ["left", 36],
  ["back", 45],
] as const;

function cubeStateAfter(moves: string[]) {
  const state = cubeTrajectory(moves).at(-1)!;
  return Object.fromEntries(
    TEST_FACE_SLICES.map(([face, offset]) => [
      face,
      Array.from(
        state.slice(offset, offset + 9),
        (value) => TEST_CUBE_COLORS[value],
      ),
    ]),
  );
}

function capture(id: string, filename: string) {
  return {
    schema: "cubed-core/capture-bundle",
    schema_version: 1,
    recording_id: id,
    capture_id: id,
    capture_session_id: id,
    created_at: id === CAPTURE_A ? "2026-07-26T12:00:00Z" : "2026-07-25T12:00:00Z",
    source: "import",
    state: "incomplete",
    sealed_at: null,
    seal_purpose: null,
    original_filename: filename,
    notes: "",
    video: {
      path: `captures/${id}/source.mov`,
      bytes: 1024,
      sha256: id.repeat(2),
      encoded_width: 1080,
      encoded_height: 1920,
      configured_fps: 120,
      actual_fps: 120,
      frame_count: 120,
      rotation_degrees: 0,
      mirrored: false,
    },
    camera: { facing: "external", intrinsics: null },
    probe: {
      status: "ok",
      width: 1080,
      height: 1920,
      fps: 120,
      duration_seconds: 1,
      frame_count: 120,
      capture_class: "target",
      guidance: "The intended 120 fps operating regime.",
    },
    solve: { scramble: "R U", end_condition: "solved" },
    calibration: null,
    teacher: null,
    sensors: { ble_raw: null, phone_imu: null },
    readiness: {
      can_decode: false,
      can_label: true,
      missing_for_decode: ["calibration"],
      warnings: [],
    },
  };
}

const capabilities = {
  schema: "cubed-core/capabilities-v1",
  workspace: "/tmp/cubed-e2e",
  upload_limits: { video_bytes: 1024 * 1024 * 1024 },
  gpu: { available: false, devices: [], reason: "No GPU" },
  decode_jobs: {
    enabled: false,
    status: "disabled",
    runner_kind: "disabled",
    runner_label: null,
    reason: "Decode disabled for browser smoke test",
    executable: null,
    profile: "local_camera_v1",
    execution_host: "runner-host",
    evidence_scope: "reconstruction-evidence-with-replay-check",
    request_schema: "cubed-core/decode-job-request-v1",
    output_schema: "cubed-core/decode-result-v1",
  },
  label: {
    pnp_assist: {
      enabled: false,
      status: "missing-dependency",
      reason: "Disabled for browser smoke test",
      execution_host: "api-host-cpu",
      opencv_version: null,
      numpy_version: null,
    },
    prediction: {
      enabled: false,
      status: "disabled",
      reason: "Disabled for browser smoke test",
      executable: null,
      execution_host: "api-host",
      output_schema: "cubed-core/label-predictions-v1",
      backend: null,
      model_profile: null,
      model_aligned_navigation: false,
    },
    autosave: {
      workspace_capture: true,
      local_file: "browser-local-storage",
    },
    exports: [],
  },
  tools: [],
};

const decodeResultA = {
  schema: "cubed-core/decode-result",
  schema_version: 1,
  recording_id: CAPTURE_A,
  status: "completed",
  profile: "local_camera_v1",
  config: {
    name: "canonical-e2e",
    cfg_hash: "1852738634",
    cfg_hash_algorithm: "posix-cksum",
  },
  inputs: [
    { id: "video", sha256: CAPTURE_A.repeat(2) },
    { id: "color-calibration-v1", sha256: CAPTURE_B.repeat(2) },
  ],
  moves: ["R", "U"],
  endpoint: { solved_reached: true },
  evaluation: null,
  provenance: {
    runtime_version: "e2e-runtime",
    finished_at: "2026-07-26T12:01:30Z",
  },
  workstation: {
    schema: "cubed-core/decode-workstation-v1",
    schema_version: 1,
    video: {
      sha256: CAPTURE_A.repeat(2),
      bytes: 1024,
      fps: 120,
      frame_count: 120,
      width: 1080,
      height: 1920,
      encoded: {
        fps: 120,
        frame_count: 120,
        width: 1080,
        height: 1920,
      },
    },
    initialization: { scramble: "R U" },
    window: [0, 119],
    warnings: [],
    frames: {
      "0": {
        motion: 0.12,
        aligned: 0.98,
        aligned_streak: 12,
        face_count: 1,
        gated: false,
        faces: [
          {
            corners: [
              [5, 5],
              [20, 5],
              [20, 20],
              [5, 20],
            ],
            confidence: 0.91,
            keypoint_confidence: [0.9, 0.91, 0.92, 0.93],
          },
        ],
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
      "10": {
        motion: 0.4,
        aligned: 0.99,
        aligned_streak: 22,
        face_count: 1,
      },
      "20": {
        motion: 0.3,
        aligned: 0.97,
        aligned_streak: 32,
        face_count: 1,
      },
    },
    events: [10, 20],
    trellis: {
      spans: [
        {
          f0: 8,
          f1: 12,
          event: 10,
          top: [{ path: "R", orientation: "UF", score: 1.5 }],
        },
      ],
      bridge: ["R", "U"],
    },
    sequence: {
      moves: [
        { move: "R", frame: 10 },
        { move: "U", frame: 20 },
      ],
      timing_basis: "canonical",
    },
    reconstruction: {
      states: [
        cubeStateAfter(["R", "U"]),
        cubeStateAfter(["R", "U", "R"]),
        cubeStateAfter(["R", "U", "R", "U"]),
      ],
      solved_reached: true,
      timeline: {
        moves: [
          { move: "R", frame: 10 },
          { move: "U", frame: 10 },
        ],
        timing_basis: "decoder-checkpoint",
      },
    },
  },
};
const { workstation: workstationFixture, ...legacyDecodeResultA } =
  decodeResultA;
void workstationFixture;

function decodeJob(
  jobId: string,
  captureId: string,
  status:
    | "queued"
    | "running"
    | "succeeded"
    | "failed"
    | "timed_out"
    | "cancelled",
) {
  const terminal = ["succeeded", "failed", "timed_out", "cancelled"].includes(
    status,
  );
  return {
    schema: "cubed-core/decode-job-status-v1",
    schema_version: 1,
    job_id: jobId,
    capture_id: captureId,
    status,
    created_at:
      captureId === CAPTURE_A
        ? "2026-07-26T12:01:00Z"
        : "2026-07-25T12:01:00Z",
    started_at:
      status === "queued"
        ? null
        : captureId === CAPTURE_A
          ? "2026-07-26T12:01:01Z"
          : "2026-07-25T12:01:01Z",
    finished_at:
      terminal
        ? captureId === CAPTURE_A
          ? "2026-07-26T12:01:08Z"
          : "2026-07-25T12:01:08Z"
        : null,
    return_code: status === "succeeded" ? 0 : null,
    log: status === "running" ? "decoding frames" : "decode complete",
    log_truncated: false,
    error: status === "failed" ? "Decode failed in the fixture." : null,
    result_available: status === "succeeded",
    result_url:
      status === "succeeded" ? `/api/decode/jobs/${jobId}/result` : null,
    result_sha256: status === "succeeded" ? "0".repeat(64) : null,
    result_bytes: status === "succeeded" ? 2048 : null,
    replay_solved_reached: status === "succeeded" ? true : null,
    video_sha256: captureId.repeat(2),
    outcome:
      status === "succeeded"
        ? "completed"
        : status === "failed" || status === "timed_out"
          ? "failed"
          : status === "cancelled"
            ? "cancelled"
            : null,
    failure:
      status === "failed"
        ? {
            code: "runner-failed",
            message: "Decode failed in the fixture.",
            retryable: true,
          }
        : null,
  };
}

const succeededRun = {
  ...decodeJob(JOB_A, CAPTURE_A, "succeeded"),
  // Disk-restored v1 jobs retain the receipt-bound submission time but not
  // the exact subprocess start time.
  started_at: null,
};
const failedRun = {
  ...decodeJob(JOB_A_OLD, CAPTURE_A, "failed"),
  created_at: "2026-07-26T11:01:00Z",
  started_at: "2026-07-26T11:01:01Z",
  finished_at: "2026-07-26T11:01:04Z",
};
const runningRun = decodeJob(JOB_B, CAPTURE_B, "running");
const legacyRun = {
  ...decodeJob(JOB_LEGACY, CAPTURE_A, "succeeded"),
  created_at: "2026-07-27T12:01:00Z",
  started_at: null,
  finished_at: "2026-07-27T12:01:00Z",
};
const runHistory = {
  [CAPTURE_A]: [succeededRun, failedRun],
  [CAPTURE_B]: [runningRun],
};

// The demo clip is a GitHub release asset. The suite never reaches the public
// internet, so every run exercises the page's clip-unavailable path.
async function blockReleaseClip(page: Page) {
  await page.route("https://github.com/**", (route) =>
    route.abort("connectionrefused"),
  );
}

// The page hashes a 2 MB overlay artifact before it renders anything, and an
// unwarmed dev server adds its own first-request cost.
async function demoReady(page: Page) {
  await expect(
    page.getByRole("heading", { name: "Watch the reconstruction" }),
  ).toBeVisible({ timeout: 30_000 });
}

async function mockWorkbench(
  page: Page,
  captureList = [
    capture(CAPTURE_A, "capture-a.mov"),
    capture(CAPTURE_B, "capture-b.mov"),
  ],
  importBodies?: string[],
  decodeJobs: Record<string, ReturnType<typeof decodeJob>[]> = {},
  decodeResults: Record<string, unknown> = {
    [JOB_A]: decodeResultA,
  },
) {
  const mutableDecodeJobs = Object.fromEntries(
    Object.entries(decodeJobs).map(([captureId, jobs]) => [
      captureId,
      [...jobs],
    ]),
  );
  await page.route("**/mock-video", (route) =>
    route.fulfill({ status: 204, body: "" }),
  );
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (path === "/api/health") {
      return json({ status: "ok", version: "e2e", publication_status: "source-testing" });
    }
    if (path === "/api/local-session" && route.request().method() === "POST") {
      return json({ admin_token: "e2e-local-session-token" });
    }
    if (path === "/api/capabilities") {
      return json(capabilities);
    }
    if (path === "/api/captures") {
      return json({
        captures: captureList,
        count: captureList.length,
      });
    }
    const captureDeleteMatch = /^\/api\/captures\/([^/]+)$/.exec(path);
    if (
      captureDeleteMatch &&
      route.request().method() === "DELETE"
    ) {
      const captureId = captureDeleteMatch[1];
      const index = captureList.findIndex(
        (candidate) => candidate.capture_id === captureId,
      );
      if (index < 0) return json({ detail: "capture not found" }, 404);
      captureList.splice(index, 1);
      return json({
        schema: "cubed-core/capture-delete-v1",
        schema_version: 1,
        capture_id: captureId,
        trashed: true,
        recoverable: true,
      });
    }
    if (
      path === "/api/captures/import" &&
      route.request().method() === "POST"
    ) {
      const body = route.request().postDataBuffer();
      if (body && importBodies) importBodies.push(body.toString("utf8"));
      return json(capture(CAPTURE_A, "imported.mov"));
    }
    if (path === "/api/remote-hosts") return json({ hosts: [] });
    if (path.endsWith("/media-ticket")) {
      return json({
        capture_id: path.split("/")[3],
        url: "/mock-video",
        expires_at: Date.now() / 1000 + 60,
      });
    }
    if (
      path === "/api/decode/jobs" &&
      route.request().method() === "GET"
    ) {
      const query = url.searchParams.get("q")?.toLocaleLowerCase() ?? "";
      const jobs = Object.values(mutableDecodeJobs)
        .flat()
        .filter((job) =>
          query
            ? [
                job.job_id,
                job.capture_id,
                job.status,
                job.outcome ?? "",
              ]
                .join(" ")
                .toLocaleLowerCase()
                .includes(query)
            : true,
        )
        .sort(
          (left, right) =>
            new Date(right.created_at).valueOf() -
            new Date(left.created_at).valueOf(),
        );
      return json({ jobs });
    }
    const captureJobsMatch =
      /^\/api\/captures\/([^/]+)\/decode-jobs$/.exec(path);
    if (
      captureJobsMatch &&
      route.request().method() === "GET"
    ) {
      return json({ jobs: mutableDecodeJobs[captureJobsMatch[1]] ?? [] });
    }
    const decodeJobMatch = /^\/api\/decode\/jobs\/([^/]+)$/.exec(path);
    if (decodeJobMatch && route.request().method() === "GET") {
      const job = Object.values(mutableDecodeJobs)
        .flat()
        .find((candidate) => candidate.job_id === decodeJobMatch[1]);
      return job
        ? json(job)
        : json({ detail: "Decode job not found" }, 404);
    }
    if (decodeJobMatch && route.request().method() === "DELETE") {
      const jobId = decodeJobMatch[1];
      let deletedCapture = "";
      for (const [captureId, jobs] of Object.entries(mutableDecodeJobs)) {
        const next = jobs.filter((candidate) => candidate.job_id !== jobId);
        if (next.length !== jobs.length) deletedCapture = captureId;
        mutableDecodeJobs[captureId] = next;
      }
      return deletedCapture
        ? json({
            schema: "cubed-core/decode-run-delete-v1",
            schema_version: 1,
            job_id: jobId,
            capture_id: deletedCapture,
            trashed: true,
            recoverable: true,
            trash_id: `trash-${jobId}`,
          })
        : json({ detail: "Decode job not found" }, 404);
    }
    const decodeResultMatch =
      /^\/api\/decode\/jobs\/([^/]+)\/result$/.exec(path);
    if (decodeResultMatch) {
      const result = decodeResults[decodeResultMatch[1]];
      return result
        ? json(result)
        : json({ detail: "Decode result not found" }, 404);
    }
    if (path.endsWith("/decode-preflight")) {
      return json({
        schema: "cubed-core/decode-preflight",
        schema_version: 1,
        recording_id: path.split("/")[3],
        status: "blocked",
        ready: false,
        execution_allowed: false,
        profile: "local_camera_v1",
        config: {
          name: "e2e",
          cfg_hash: "0",
          cfg_hash_algorithm: "posix-cksum",
          derived_from_cfg_hash: "0",
          evidence_status: "test",
        },
        checks: [{ id: "calibration", status: "fail", detail: "Attach calibration." }],
        missing: ["calibration"],
        warnings: [],
      });
    }
    return json({ detail: `Unmocked ${path}` }, 404);
  });
}

function multipartField(body: string, name: string): string {
  const match = new RegExp(
    `name="${name}"\\r\\n\\r\\n([^\\r\\n]*)`,
  ).exec(body);
  return match?.[1] ?? "";
}

async function mockCalibrationVideo(page: Page) {
  await page.addInitScript(() => {
    const sources = new WeakMap<HTMLMediaElement, string>();
    const times = new WeakMap<HTMLMediaElement, number>();
    const paused = new WeakMap<HTMLMediaElement, boolean>();
    Object.defineProperties(HTMLMediaElement.prototype, {
      src: {
        configurable: true,
        get() {
          return sources.get(this) ?? "";
        },
        set(value: string) {
          sources.set(this, value);
          window.setTimeout(
            () => this.dispatchEvent(new Event("loadedmetadata")),
            0,
          );
        },
      },
      currentSrc: {
        configurable: true,
        get() {
          return sources.get(this) ?? "";
        },
      },
      currentTime: {
        configurable: true,
        get() {
          return times.get(this) ?? 0;
        },
        set(value: number) {
          this.dispatchEvent(new Event("seeking"));
          times.set(this, value);
          this.dispatchEvent(new Event("timeupdate"));
          window.setTimeout(
            () => this.dispatchEvent(new Event("seeked")),
            0,
          );
        },
      },
      duration: {
        configurable: true,
        get() {
          return 4;
        },
      },
      readyState: {
        configurable: true,
        get() {
          return 4;
        },
      },
      paused: {
        configurable: true,
        get() {
          return paused.get(this) ?? true;
        },
      },
    });
    Object.defineProperties(HTMLVideoElement.prototype, {
      videoWidth: {
        configurable: true,
        get() {
          return 1920;
        },
      },
      videoHeight: {
        configurable: true,
        get() {
          return 1080;
        },
      },
    });
    Object.defineProperties(HTMLMediaElement.prototype, {
      play: {
        configurable: true,
        value() {
          paused.set(this, false);
          this.dispatchEvent(new Event("play"));
          return Promise.resolve();
        },
      },
      pause: {
        configurable: true,
        value() {
          paused.set(this, true);
          this.dispatchEvent(new Event("pause"));
        },
      },
    });
    Object.defineProperties(HTMLCanvasElement.prototype, {
      getContext: {
        configurable: true,
        value() {
          return {
            imageSmoothingEnabled: true,
            drawImage() {},
          };
        },
      },
      toBlob: {
        configurable: true,
        value(callback: BlobCallback) {
          callback(new Blob(["lossless-png"], { type: "image/png" }));
        },
      },
    });
  });
}

test("an offline new user lands on the static replay", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.route("**/api/**", (route) =>
    route.abort("connectionrefused"),
  );
  await blockReleaseClip(page);
  const demoArtifactRequests: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/demo/gtd1/")) demoArtifactRequests.push(path);
  });
  await page.goto("/");
  await expect(page).toHaveURL(/\/demo$/);
  await expect(
    page.getByRole("heading", { name: /Reconstruction demo/ }),
  ).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".lab-nav .nav-name")).toHaveText([
    "Demo",
    "Decode",
    "Runs",
    "Add video",
    "Label",
  ]);
  await expect(page.locator(".start-route")).toHaveCount(0);
  await demoReady(page);
  await expect(page.locator(".demo-overlay")).toBeVisible();
  await expect(page.locator(".demo-video-note")).toContainText(
    "The demo clip did not load.",
  );
  await expect(page.locator(".demo-video-note")).toContainText(
    "download_release_assets.py",
  );
  await expect(page.locator(".demo-boundary-line")).toContainText(
    "runs no decoder",
  );
  await expect(page.locator(".demo-moves-note")).toContainText(
    "smart-cube record",
  );
  await expect(page.locator(".demo-timeline-readout")).toContainText(
    "0 of 77",
  );

  // Both timeline lanes are named, so neither layer reads as decoration.
  await expect(page.locator(".demo-timeline-lane-label")).toHaveText([
    "moves",
    /alignment/,
  ]);

  // The three slots are always present in the same order, whatever the frame
  // holds. Frame 0 has no detection, so every cell is empty.
  await expect(page.locator(".demo-reads h3")).toHaveText(
    "Sampled color reads",
  );
  await expect(page.locator(".demo-reads .demo-read-slot")).toHaveText([
    "up",
    "front",
    "right",
  ]);
  await expect(page.locator(".demo-reads .demo-read-empty")).toHaveCount(27);
  await expect(page.locator(".demo-reads-status")).toHaveCount(0);
  await expect(page.locator(".cube-state")).toBeVisible();
  await expect(page.locator(".move-replay-caption")).toContainText("Step 0");

  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
  await expect
    .poll(async () => page.locator(".demo-frame-readout strong").textContent())
    .not.toBe("0");
  await page.getByRole("button", { name: "Pause" }).click();

  // Clicking the timeline seeks, and the cube follows the recorded turns.
  const strip = page.locator(".demo-timeline-lane").first();
  const box = (await strip.boundingBox())!;
  await strip.click({
    position: { x: box.width / 2, y: box.height / 2 },
  });
  await expect(page.locator(".move-replay-caption")).toContainText(" of 77");

  // A frame with detections fills the same three grids from the recorded
  // sampled colours. Nothing in the panel classifies a value, and no box may
  // change size as detections come and go across frames.
  const geometry = () =>
    page.evaluate(() => {
      const reads = document.querySelector(".demo-reads")!.getBoundingClientRect();
      const cube = document.querySelector(".demo-cube")!.getBoundingClientRect();
      return {
        height: Math.round(reads.height),
        cubeOffset: Math.round(cube.top - reads.top),
        slots: document.querySelectorAll(".demo-read-slot").length,
      };
    });
  const atZero = await geometry();

  await page.locator(".frame-scrubber").fill("2600");
  await expect(page.locator(".demo-reads .demo-read-empty")).toHaveCount(0);
  await expect(page.locator(".demo-reads .demo-read-cell")).toHaveCount(27);
  const cell = page.locator(".demo-reads .demo-read-cell").first();
  await expect(cell).toHaveAttribute("title", /^sampled #[0-9a-f]{6} · confidence/);
  await expect(page.locator(".demo-reads")).not.toContainText(/sticker|classified/i);
  expect(await geometry()).toEqual(atZero);

  // A partial detection keeps the same geometry too.
  await page.locator(".frame-scrubber").fill("4800");
  await expect(page.locator(".demo-reads .demo-read-empty")).toHaveCount(9);
  expect(await geometry()).toEqual(atZero);

  await expect(page.locator(".demo-detail")).toHaveCount(0);
  await expect(page.getByText("Signals, hashes, and downloads")).toHaveCount(0);

  expect(demoArtifactRequests.sort()).toEqual([
    "/demo/gtd1/ble-ground-truth.json",
    "/demo/gtd1/decode-receipt.json",
    "/demo/gtd1/decode-result.json",
    "/demo/gtd1/overlay-track.json",
  ]);
});

test("the legacy published-demo fragment still reaches the demo page", async ({
  page,
}) => {
  await mockWorkbench(page);
  await blockReleaseClip(page);
  await page.goto("/reconstruct#published-demo");
  await expect(page).toHaveURL(/\/demo$/);
  await demoReady(page);
  await expect(
    page.getByRole("heading", { name: "Reconstruct", exact: true }),
  ).toHaveCount(0);
});

test("the top bar exposes the five OSS pages", async ({
  page,
}) => {
  await mockWorkbench(page);
  const nav = page.locator(".lab-nav");
  await page.goto("/runs");
  await expect(nav.locator(".nav-name")).toHaveText([
    "Demo",
    "Decode",
    "Runs",
    "Add video",
    "Label",
  ]);
  await expect(nav.locator(".nav-badge")).toHaveCount(0);
  await expect(nav.locator(".nav-group-label")).toHaveText(["Data tools"]);
  await expect(nav.locator(".nav-item-active")).toHaveText("Runs");

  await nav.getByRole("link", { name: "Add video" }).click();
  await expect(page).toHaveURL(/\/import$/);
  await expect(
    page.getByRole("heading", { name: "Add video", exact: true }),
  ).toBeVisible();
  await expect(nav.locator(".nav-item-active")).toHaveText("Add video");

  await nav.getByRole("link", { name: "Label" }).click();
  await expect(page).toHaveURL(/\/label$/);
  await expect(
    page.getByRole("heading", { name: "Label", exact: true }),
  ).toBeVisible();
  await expect(nav.locator(".nav-item-active")).toHaveText("Label");
});

test("the brand returns workbench users to the demo", async ({
  page,
}) => {
  await mockWorkbench(page);
  await blockReleaseClip(page);
  await page.goto("/runs");
  await expect(
    page.getByRole("heading", { name: "Runs", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Cubed Core demo" }).click();
  await expect(page).toHaveURL(/\/demo$/);
  await expect(
    page.getByRole("heading", { name: /Reconstruction demo/ }),
  ).toBeVisible();
});

test("loopback workbench bootstraps its local session before protected reads", async ({
  page,
}) => {
  const requests: { path: string; token: string }[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (!path.startsWith("/api/")) return;
    requests.push({
      path,
      token: request.headers()["x-cubed-admin-token"] ?? "",
    });
  });
  await mockWorkbench(page);
  await page.goto("/runs");
  await expect(
    page.getByRole("heading", { name: "Recent recordings" }),
  ).toBeVisible();
  await expect
    .poll(() => requests.some(({ path }) => path === "/api/capabilities"))
    .toBe(true);

  const localSessionIndex = requests.findIndex(
    ({ path }) => path === "/api/local-session",
  );
  const healthIndex = requests.findIndex(({ path }) => path === "/api/health");
  expect(localSessionIndex).toBeGreaterThanOrEqual(0);
  expect(healthIndex).toBeGreaterThan(localSessionIndex);
  expect(
    requests.find(({ path }) => path === "/api/capabilities")?.token,
  ).toBe("e2e-local-session-token");
});

test("a remote #admin link authenticates and removes the token fragment", async ({
  page,
}) => {
  const protectedTokens: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path === "/api/capabilities" || path === "/api/captures") {
      protectedTokens.push(
        request.headers()["x-cubed-admin-token"] ?? "",
      );
    }
  });
  await mockWorkbench(page);
  await page.goto("/runs#admin=remote-e2e-token");
  await expect(page).toHaveURL(/\/runs$/);
  await expect(
    page.getByRole("heading", { name: "Recent recordings" }),
  ).toBeVisible();
  expect(protectedTokens).toContain("remote-e2e-token");
});

test("Add video carries the displayed recording scramble into Decode", async ({
  page,
}) => {
  test.setTimeout(90_000);
  const importBodies: string[] = [];
  await mockWorkbench(page, [], importBodies);
  await page.goto("/import");
  await expect(
    page.getByRole("heading", { name: "Add video", exact: true }),
  ).toBeVisible();

  const importSection = page.locator(".import-source-card");
  const recordingForm = importSection.locator(".import-record-form");
  const fileInput = recordingForm.locator('input[type="file"]');
  const submit = recordingForm.getByRole("button", {
    name: "Add to workspace",
  });
  const confirmation = recordingForm.getByLabel(
    "This recording starts from the displayed scramble.",
  );
  await fileInput.setInputFiles({
    name: "recorded.mov",
    mimeType: "video/quicktime",
    buffer: Buffer.from("video"),
  });
  await expect(submit).toBeDisabled();
  await confirmation.check();
  await recordingForm.getByRole("button", { name: "New scramble" }).click();
  await expect(fileInput).toHaveValue("");
  await expect(confirmation).not.toBeChecked();
  await expect(confirmation).toBeDisabled();

  const scrambleLabel = await recordingForm
    .locator(".import-scramble-tape")
    .getAttribute("aria-label");
  const displayedScramble = scrambleLabel?.replace("Starting scramble: ", "");
  expect(displayedScramble?.split(" ")).toHaveLength(20);
  await fileInput.setInputFiles({
    name: "recorded.mov",
    mimeType: "video/quicktime",
    buffer: Buffer.from("video"),
  });
  await confirmation.check();
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect.poll(() => importBodies.length).toBe(1);
  expect(multipartField(importBodies[0], "scramble")).toBe(
    displayedScramble,
  );
  expect(multipartField(importBodies[0], "capture_session_id")).toBe("");
  await expect(page.getByText("Video added. Choose a calibration below.")).toBeVisible();
  await expect(
    page.getByRole("option", { name: "Published dataset shared calibration" }),
  ).toHaveCount(1);
  const nextActions = page.locator(".import-next-actions");
  await expect(nextActions.getByRole("link", { name: /Decode/ })).toHaveAttribute(
    "href",
    `/decode?capture=${CAPTURE_A}`,
  );
  await expect(nextActions.getByRole("link", { name: /Label/ })).toHaveAttribute(
    "href",
    `/label?capture=${CAPTURE_A}`,
  );
});

test("Add video accepts an existing recording with a known scramble", async ({
  page,
}) => {
  const importBodies: string[] = [];
  await mockWorkbench(page, [], importBodies);
  await page.goto("/import");

  const existingPanel = page.locator(".import-existing-panel");
  await existingPanel.getByText("Use an existing recording").click();
  const existingForm = existingPanel.locator(".import-existing-form");
  await existingForm.locator('input[type="file"]').setInputFiles({
    name: "existing.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("video"),
  });
  await existingForm.locator("#import-existing-scramble").fill("R U' F2");
  await existingForm.getByRole("button", {
    name: "Add existing video",
  }).click();

  await expect.poll(() => importBodies.length).toBe(1);
  expect(multipartField(importBodies[0], "scramble")).toBe("R U' F2");
  expect(multipartField(importBodies[0], "capture_session_id")).toBe("");
});

test("Add video moves a selected workspace capture to system Trash", async ({
  page,
}) => {
  const captures = [
    capture(CAPTURE_B, "solve10.mov"),
    capture(CAPTURE_A, "solve2.mov"),
  ];
  await mockWorkbench(page, captures);
  await page.goto(`/import?capture=${CAPTURE_A}`);

  await expect(page.locator(".import-receipt")).toContainText("solve2.mov");
  const manager = page.locator(".import-workspace-manager");
  await manager.getByText("Manage workspace videos").click();
  await expect(manager.locator(".import-workspace-video-row strong")).toHaveText([
    "solve2.mov",
    "solve10.mov",
  ]);

  const selectedRow = manager
    .locator(".import-workspace-video-row")
    .filter({ hasText: "solve2.mov" });
  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toContain('Remove "solve2.mov"');
    expect(dialog.message()).toContain("video and labels move to your system Trash");
    expect(dialog.message()).toContain("Saved Runs stay");
    expect(dialog.message()).not.toContain("published copy");
    await dialog.accept();
  });
  await selectedRow.getByRole("button", { name: "Move to Trash" }).click();

  await expect(page).toHaveURL(/\/import$/);
  await expect(
    manager
      .locator(".import-workspace-video-row")
      .filter({ hasText: "solve2.mov" }),
  ).toHaveCount(0);
  await expect(manager.getByText("solve10.mov")).toBeVisible();
  await expect(page.locator(".import-receipt")).toHaveCount(0);
  await expect(
    manager.getByText("solve2.mov moved to Trash."),
  ).toBeVisible();
});

test("Add video keeps a capture listed when Trash rejects it", async ({
  page,
}) => {
  const captures = [capture(CAPTURE_A, "active.mov")];
  await mockWorkbench(page, captures);
  await page.route(`**/api/captures/${CAPTURE_A}`, (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "Capture has an active decode job.",
      }),
    }),
  );
  await page.goto("/import");

  const manager = page.locator(".import-workspace-manager");
  await manager.getByText("Manage workspace videos").click();
  page.once("dialog", (dialog) => dialog.accept());
  await manager.getByRole("button", { name: "Move to Trash" }).click();

  await expect(
    manager.getByRole("alert").filter({
      hasText: "Capture has an active decode job.",
    }),
  ).toBeVisible();
  await expect(
    manager.locator(
      ".import-workspace-manager-body > .import-workspace-message-error + .import-workspace-group",
    ),
  ).toHaveCount(1);
  await expect(manager.getByText("active.mov")).toBeVisible();
});

test("Add video samples and attaches six desktop calibration crops", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await mockCalibrationVideo(page);
  await mockWorkbench(page);
  const calibrationBodies: string[] = [];
  await page.route(
    "**/api/captures/*/calibration/from-crops",
    async (route) => {
      const body = route.request().postDataBuffer();
      if (body) calibrationBodies.push(body.toString("latin1"));
      const baseCapture = capture(CAPTURE_A, "capture-a.mov");
      const updated = {
        ...baseCapture,
        calibration: {
          path: `captures/${CAPTURE_A}/sidecars/color-centroids-v1.json`,
          sha256: "c".repeat(64),
          kind: "color-centroids-v1",
        },
        readiness: {
          ...baseCapture.readiness,
          can_decode: true,
          missing_for_decode: [],
        },
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(updated),
      });
    },
  );
  await page.goto(`/import?capture=${CAPTURE_A}`);

  const sampler = page.locator(".color-calibration-sampler").first();
  await expect(
    sampler.getByRole("heading", { name: "Sample this video" }),
  ).toBeVisible();
  const video = sampler.locator("video");
  await expect(video).toBeVisible();
  await video.evaluate((element) =>
    element.dispatchEvent(new Event("loadedmetadata")),
  );
  const hitTarget = sampler.locator(".color-calibration-hit-target");
  await expect(hitTarget).toBeVisible();

  await video.evaluate((element) =>
    element.dispatchEvent(new Event("seeking")),
  );
  const hitBounds = await hitTarget.boundingBox();
  expect(hitBounds).not.toBeNull();
  await hitTarget.dispatchEvent("click", {
    clientX: hitBounds!.x + 260,
    clientY: hitBounds!.y + 160,
  });
  await expect(
    sampler.locator(".color-calibration-message-error"),
  ).toContainText(
    "Wait for the selected video frame to finish loading.",
  );
  await video.evaluate((element) =>
    element.dispatchEvent(new Event("seeked")),
  );
  await expect(sampler.locator(".color-calibration-progress")).toHaveText(
    "0 / 6 sampled",
  );

  await hitTarget.focus();
  await hitTarget.press("ArrowRight");
  await hitTarget.press("Enter");
  await expect(sampler.locator(".color-calibration-progress")).toHaveText(
    "1 / 6 sampled",
  );
  await sampler.getByRole("button", { name: "Reset all" }).click();
  await expect(sampler.locator(".color-calibration-progress")).toHaveText(
    "0 / 6 sampled",
  );

  for (let count = 1; count <= 6; count += 1) {
    await hitTarget.click({ position: { x: 260, y: 160 } });
    await expect(sampler.locator(".color-calibration-progress")).toHaveText(
      `${count} / 6 sampled`,
    );
  }
  await expect(
    sampler.getByRole("button", { name: "Create calibration" }),
  ).toBeEnabled();
  await sampler.getByRole("button", { name: "Create calibration" }).click();

  await expect.poll(() => calibrationBodies.length).toBe(1);
  for (const color of ["white", "green", "red", "blue", "orange", "yellow"]) {
    expect(calibrationBodies[0]).toContain(`name="${color}"`);
    expect(calibrationBodies[0]).toContain(`filename="${color}.png"`);
    expect(calibrationBodies[0]).toContain("Content-Type: image/png");
  }
  await expect(page.getByText("Calibration created from this video.")).toBeVisible();
  await expect(page.locator(".prepare-row-state")).toContainText("Attached");
  await expect(
    page.getByText("Use another calibration source"),
  ).toBeVisible();
});

test("Runs groups Decode attempts and opens an exact read-only result", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await mockWorkbench(page, undefined, undefined, runHistory);
  await page.goto("/runs");

  await expect(
    page.getByRole("heading", { name: "Runs", exact: true }),
  ).toBeVisible();
  const groups = page.locator(".run-group");
  await expect(groups).toHaveCount(2);
  await expect(groups.nth(0)).toContainText("capture-a.mov");
  await expect(groups.nth(0)).toContainText("2 attempts");
  await expect(groups.nth(0).locator(".run-attempt").first()).toContainText(
    "Completed",
  );
  await expect(groups.nth(0).locator(".run-attempt").first()).toContainText(
    "1.0s video",
  );
  await expect(groups.nth(0).locator(".run-history")).toContainText(
    "1 earlier attempt",
  );
  await expect(groups.nth(1)).toContainText("capture-b.mov");
  await expect(groups.nth(1)).toContainText("Running");
  await expect(groups.nth(1)).toContainText("1.0s video");
  await expect(
    groups.nth(1).getByRole("button", { name: "Refresh" }),
  ).toBeVisible();

  const listDownloadStarted = page.waitForEvent("download");
  await groups
    .nth(0)
    .locator(".run-attempt")
    .first()
    .getByRole("button", { name: "Download" })
    .click();
  const listDownload = await listDownloadStarted;
  expect(listDownload.suggestedFilename()).toBe(
    `cubed-core-run-${JOB_A}.json`,
  );
  expect(await readFile((await listDownload.path())!, "utf8")).toBe(
    JSON.stringify(decodeResultA),
  );

  await page.getByPlaceholder("Search filename, status, or ID").fill("failed");
  await expect(groups).toHaveCount(1);
  await expect(groups.nth(0)).toContainText("Decode failed in the fixture.");
  await page.getByPlaceholder("Search filename, status, or ID").fill("");

  await groups
    .nth(0)
    .locator(".run-attempt")
    .first()
    .getByRole("button", { name: "Inspect" })
    .click();
  await expect(page).toHaveURL(new RegExp(`/runs\\?run=${JOB_A}$`));
  await expect(
    page.getByRole("heading", { name: "capture-a.mov" }),
  ).toBeVisible();
  await expect(
    page.locator(".tracker-receipt-disclosure"),
  ).toContainText(`cubed-core-run-${JOB_A}.json`, {
    timeout: 15_000,
  });
  await expect(page.locator(".tracker-workstation")).toBeVisible();
  await expect(
    page.locator(".tracker-workstation").getByRole("slider", {
      name: "Current decode frame",
    }),
  ).toBeVisible();
  const videoColumn = await page
    .locator(".tracker-workstation-video")
    .boundingBox();
  const transport = await page
    .locator(".tracker-frame-transport")
    .boundingBox();
  const scrubber = await page
    .locator(".tracker-workstation-viewer .frame-scrubber")
    .boundingBox();
  expect(videoColumn).not.toBeNull();
  expect(transport).not.toBeNull();
  expect(scrubber).not.toBeNull();
  expect(transport!.y).toBeGreaterThanOrEqual(
    videoColumn!.y + videoColumn!.height,
  );
  expect(scrubber!.y).toBeGreaterThanOrEqual(
    transport!.y + transport!.height,
  );
  expect(
    Math.abs(
      transport!.x +
        transport!.width / 2 -
        (videoColumn!.x + videoColumn!.width / 2),
    ),
  ).toBeLessThanOrEqual(1);
  await expect(page.getByText("Sampled reads")).toBeVisible();
  await expect(page.locator(".tracker-strip")).toHaveCount(1);
  await expect(page.locator(".tracker-frame-metrics")).toHaveCount(1);
  await expect(page.locator(".tracker-current-reads")).toHaveCount(1);
  const frameSummary = page.locator(".tracker-frame-summary");
  const readsBox = await frameSummary
    .locator(".tracker-current-reads")
    .boundingBox();
  const metricsBox = await frameSummary
    .locator(".tracker-current-status")
    .boundingBox();
  expect(readsBox).not.toBeNull();
  expect(metricsBox).not.toBeNull();
  expect(metricsBox!.x).toBeGreaterThanOrEqual(readsBox!.x + readsBox!.width);
  const metricColumns = await page
    .locator(".tracker-frame-metrics")
    .evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length);
  expect(metricColumns).toBe(2);
  const diagnostics = page.locator(".tracker-inline-diagnostics");
  await expect(
    diagnostics.getByRole("heading", { name: "Reconstruction", exact: true }),
  ).toBeVisible();
  await expect(
    diagnostics.getByRole("heading", { name: "Trellis / spans", exact: true }),
  ).toBeVisible();
  await expect(diagnostics.locator(".tracker-diagnostic-empty")).toHaveCount(0);
  await expect(page.getByText("More diagnostics", { exact: true })).toHaveCount(0);

  const recordingSummary = page.locator(".decode-run-recording-summary");
  await expect(recordingSummary).toContainText("capture-a.mov");
  await expect(recordingSummary).toContainText("window 0–119");
  await expect(recordingSummary).toContainText("120.000 fps");
  await expect(recordingSummary).toContainText("1080×1920");
  await expect(recordingSummary).toContainText("1.00s");
  await expect(recordingSummary).toContainText("Starting scramble");
  await expect(recordingSummary).toContainText("R U");

  await page.locator(".tracker-receipt-disclosure > summary").click();
  const receipt = page.locator(".tracker-receipt-body");
  await expect(receipt).toContainText("Run time");
  await expect(receipt).toContainText("Video SHA");
  await expect(receipt).toContainText("canonical-e2e");

  const frameSlider = page.getByRole("slider", {
    name: "Current decode frame",
  });
  await expect(page.locator(".decode-run-reconstruction")).toContainText(
    "Frame 0 · starting scramble · next decoder checkpoint f10",
  );
  await frameSlider.fill("1");
  await expect(page.locator(".decode-run-reconstruction")).toContainText(
    "Frame 1 · starting scramble · next decoder checkpoint f10",
  );
  await frameSlider.fill("5");
  await expect(page.locator(".decode-run-reconstruction")).toContainText(
    "Frame 5 · estimated move 1/2 · next checkpoint f10",
  );
  await frameSlider.fill("9");
  await expect(page.locator(".decode-run-reconstruction")).toContainText(
    "Frame 9 · estimated move 1/2 · next checkpoint f10",
  );
  await frameSlider.fill("10");
  await expect(page.locator(".decode-run-frame-sequence-context")).toContainText(
    "1/2 R",
  );
  await expect(page.locator(".decode-run-frame-sequence-context")).toContainText(
    "at frame 10",
  );
  await expect(page.locator(".decode-run-frame-sequence-context")).toContainText(
    "canonical timing",
  );
  await expect(page.locator(".decode-run-reconstruction")).toContainText(
    "decoder checkpoint 1/1 · 2 of 2 moves",
  );
  await page.keyboard.press("ArrowRight");
  await expect(frameSlider).toHaveValue("11");

  await page.evaluate((jobId) => {
    const url = new URL(window.location.href);
    url.search = `?run=${encodeURIComponent(jobId)}`;
    window.history.pushState({}, "", url);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }, JOB_A_OLD);
  await expect(page.locator(".run-no-result")).toContainText(
    "Decode failed in the fixture.",
  );
  await expect(page.getByText("Loading run JSON…")).toHaveCount(0);

  await page.getByRole("button", { name: "Change run" }).click();
  await expect(page).toHaveURL(/\/runs$/);
  await expect(groups).toHaveCount(2);

  await groups.nth(0).locator(".run-history > summary").click();
  page.once("dialog", (dialog) => dialog.accept());
  await groups
    .nth(0)
    .locator(".run-history")
    .getByRole("button", { name: "Delete run" })
    .click();
  await expect(groups.nth(0)).toContainText("1 attempt");
  await expect(groups.nth(0).locator(".run-history")).toHaveCount(0);
});

test("Runs can inspect a portable artifact without binding it to a capture", async ({
  page,
}) => {
  test.setTimeout(90_000);
  expect(() => parseDecodeResultDocument(decodeResultA)).not.toThrow();
  await mockWorkbench(page);
  await page.goto("/runs");
  await page.getByRole("button", { name: "Open run JSON" }).click();
  await expect(page).toHaveURL(/\/runs\?source=file$/);
  await expect(
    page.getByRole("button", { name: "Choose run JSON" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Choose matching video" }),
  ).toBeVisible();
  await expect(page.getByText("No run JSON loaded")).toBeVisible();

  const dumpInput = page.locator('input[type="file"][accept*=".json"]');
  await dumpInput.setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ schema: "not-a-decode-run" })),
  });
  await expect(page.getByRole("alert")).toContainText(
    /(Run JSON must declare|run JSON is missing required fields)/i,
  );
  await expect(page.getByText("No run JSON loaded")).toBeVisible();

  const portableArtifactBytes = `${JSON.stringify(decodeResultA, null, 3)}\n`;
  await dumpInput.setInputFiles({
      name: "cubed-core-run-portable.json",
      mimeType: "application/json",
      buffer: Buffer.from(portableArtifactBytes),
  });
  await expect(page.locator(".tracker-receipt-disclosure")).toContainText(
    "cubed-core-run-portable.json",
  );
  await expect(page.locator(".tracker-workstation")).toBeVisible();
  await expect(page.locator(".tracker-current-reads")).toContainText("up");

  await page.locator(".tracker-receipt-disclosure > summary").click();
  const portableDownloadStarted = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download run JSON" }).click();
  const portableDownload = await portableDownloadStarted;
  expect(await readFile((await portableDownload.path())!, "utf8")).toBe(
    portableArtifactBytes,
  );

  const videoInput = page.locator(
    'input[type="file"][accept*="video"]',
  );
  await videoInput.setInputFiles({
    name: "wrong.mov",
    mimeType: "video/quicktime",
    buffer: Buffer.from("wrong-video"),
  });
  await expect(page.getByRole("alert")).toContainText("Video SHA mismatch");
  await expect(page.locator(".tracker-workstation")).toBeVisible();
});

test("portable older results pair a local video with native controls", async ({
  page,
}) => {
  const videoBytes = Buffer.from("portable-video-only-fixture");
  const videoSha = createHash("sha256").update(videoBytes).digest("hex");
  const videoOnlyResult = {
    ...legacyDecodeResultA,
    recording_id: CAPTURE_B,
    inputs: decodeResultA.inputs.map((input) =>
      input.id === "video" ? { ...input, sha256: videoSha } : input,
    ),
  };
  expect(() => parseDecodeResultDocument(videoOnlyResult)).not.toThrow();

  await mockWorkbench(page, []);
  await page.goto("/runs?source=file");
  await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
    name: "portable-video-only.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(videoOnlyResult)),
  });
  await page.locator('input[type="file"][accept*="video"]').setInputFiles({
    name: "matching.mov",
    mimeType: "video/quicktime",
    buffer: videoBytes,
  });

  const videoOnlyStation = page.locator(
    ".decode-run-video-only-workstation",
  );
  await expect(videoOnlyStation).toBeVisible();
  await expect(videoOnlyStation.locator("video")).toHaveJSProperty(
    "controls",
    true,
  );
  await expect(videoOnlyStation).toContainText(
    "Frame diagnostics were not embedded",
  );
  await expect(
    videoOnlyStation.getByRole("slider", { name: "Current decode frame" }),
  ).toHaveCount(0);
});

test("legacy workspace results keep capture video transport without recompute", async ({
  page,
}) => {
  expect(() => parseDecodeResultDocument(legacyDecodeResultA)).not.toThrow();
  const requestedPaths: string[] = [];
  page.on("request", (request) =>
    requestedPaths.push(new URL(request.url()).pathname),
  );
  await mockWorkbench(
    page,
    [capture(CAPTURE_A, "capture-a.mov")],
    undefined,
    { [CAPTURE_A]: [legacyRun] },
    { [JOB_LEGACY]: legacyDecodeResultA },
  );
  await page.goto(`/runs?run=${JOB_LEGACY}`);

  await expect(page.locator(".decode-run-recording-summary")).toContainText(
    "capture-a.mov",
    { timeout: 15_000 },
  );
  await expect(page.locator(".decode-run-recording-summary")).toContainText(
    "120.000 fps",
  );
  const legacyStation = page.locator(".decode-run-legacy-workstation");
  await expect(legacyStation).toBeVisible({ timeout: 15_000 });
  await expect(
    legacyStation.getByRole("slider", { name: "Current decode frame" }),
  ).toHaveAttribute("max", "119");
  await expect(legacyStation).toContainText(
    "Not embedded in this older result",
  );
  await expect(legacyStation).toContainText("Nothing is recomputed here.");
  await expect
    .poll(() =>
      requestedPaths.includes(`/api/captures/${CAPTURE_A}/media-ticket`),
    )
    .toBe(true);
});

test("Decode is a dedicated three-stage execution page", async ({ page }) => {
  await mockWorkbench(page);
  await page.goto(`/decode?capture=${CAPTURE_A}`);

  await expect(page).toHaveURL(new RegExp(`/decode\\?capture=${CAPTURE_A}$`));
  await expect(
    page.getByRole("heading", { name: "Decode", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Workspace capture")).toHaveValue(CAPTURE_A);
  await expect(page.getByLabel("Compute target")).toHaveValue("");
  await expect(
    page.getByRole("option", { name: "Local CUDA (this workbench)" }),
  ).toBeDisabled();
  await expect(
    page.getByText("Decode jobs are unavailable on this API host."),
  ).toBeVisible();
  await expect(
    page.getByText(/published Hugging Face dataset/i),
  ).toBeVisible();
  await expect(page.locator(".decode-dataset-onboarding code")).toHaveCount(1);
  await expect(page.getByRole("button", { name: /Run tracker/i })).toHaveCount(0);
});

test("Decode starts with an explicit capture choice", async ({ page }) => {
  await mockWorkbench(page);
  await page.goto("/decode");

  await expect(page).toHaveURL(/\/decode$/);
  const picker = page.getByLabel("Workspace capture");
  await expect(picker).toHaveValue("");
  await expect(
    picker.locator('option[value=""]'),
  ).toHaveText("Choose a capture…");
  await expect(
    page.getByText("Choose a recording above to prepare it."),
  ).toBeVisible();
  await expect(
    page.getByText(
      "Choose a recording above to check readiness and run the decoder.",
    ),
  ).toBeVisible();
});

test("Decode keeps a restored result primary when Local CUDA is unavailable", async ({
  page,
}) => {
  const jobId = "restored-decode-job";
  await mockWorkbench(page);
  await page.route("**/api/capabilities", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...capabilities,
        decode_jobs: {
          ...capabilities.decode_jobs,
          enabled: true,
          status: "ready",
          runner_kind: "native",
          runner_label: "Built-in decode runner",
          reason: null,
        },
      }),
    }),
  );
  await page.route(`**/api/captures/${CAPTURE_A}/decode-jobs`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        jobs: [
          {
            schema: "cubed-core/decode-job-status-v1",
            schema_version: 1,
            job_id: jobId,
            capture_id: CAPTURE_A,
            status: "succeeded",
            created_at: "2026-07-28T12:00:00Z",
            started_at: "2026-07-28T12:00:01Z",
            finished_at: "2026-07-28T12:00:05Z",
            return_code: 0,
            log: "",
            log_truncated: false,
            error: null,
            result_available: true,
            result_url: `/api/decode/jobs/${jobId}/result`,
            result_sha256: "0".repeat(64),
            result_bytes: 512,
            replay_solved_reached: true,
          },
        ],
      }),
    }),
  );
  await page.route(`**/api/captures/${CAPTURE_A}/decode-preflight`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "cubed-core/decode-preflight",
        schema_version: 1,
        recording_id: CAPTURE_A,
        status: "ready",
        ready: true,
        execution_allowed: true,
        profile: "local_camera_v1",
        config: {
          name: "e2e",
          cfg_hash: "0",
          cfg_hash_algorithm: "posix-cksum",
          derived_from_cfg_hash: "0",
          evidence_status: "test",
        },
        checks: [],
        missing: [],
        warnings: [],
      }),
    }),
  );
  await page.route(`**/api/decode/jobs/${jobId}/result`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "cubed-core/decode-result",
        schema_version: 1,
        recording_id: CAPTURE_A,
        status: "completed",
        profile: "local_camera_v1",
        config: {
          name: "e2e",
          cfg_hash: "0",
          cfg_hash_algorithm: "posix-cksum",
        },
        inputs: [],
        moves: ["R", "U"],
        endpoint: { solved_reached: true },
        evaluation: null,
        provenance: {},
      }),
    }),
  );

  await page.goto(`/decode?capture=${CAPTURE_A}`);
  await expect(page.locator(".decode-result-panel")).toBeVisible({
    timeout: 15_000,
  });
  await expect(
    page.getByText("Choose an available compute target."),
  ).toHaveCount(0);
  await expect(page.getByLabel("Compute target")).toHaveValue("");
  await expect(page.getByRole("button", { name: "Run decode" })).toBeDisabled();
  const decodeDownloadStarted = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download run JSON" }).click();
  const decodeDownload = await decodeDownloadStarted;
  expect(decodeDownload.suggestedFilename()).toBe(
    `cubed-core-run-${jobId}.json`,
  );
  expect(
    JSON.parse(await readFile((await decodeDownload.path())!, "utf8")),
  ).toMatchObject({
    schema: "cubed-core/decode-result",
    recording_id: CAPTURE_A,
  });
});

test("/reconstruct aliases the Decode page", async ({ page }) => {
  await mockWorkbench(page);
  await page.goto(`/reconstruct?capture=${CAPTURE_A}`);
  await expect(page).toHaveURL(new RegExp(`/decode\\?capture=${CAPTURE_A}$`));
  await expect(
    page.getByRole("heading", { name: "Decode", exact: true }),
  ).toBeVisible();
});

test("narrow workbench shows the desktop-only notice", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockWorkbench(page);
  await page.goto("/runs");

  await expect(page.locator(".desktop-only-notice")).toBeVisible();
  await expect(page.locator(".desktop-only-notice")).toContainText(
    "Cubed Core is a desktop workbench.",
  );
  await expect(page.locator(".lab-shell")).toBeHidden();
});

test("the demo shows the scramble and stays coherent across desktop widths", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockWorkbench(page);
  await blockReleaseClip(page);
  await page.goto("/demo");
  await demoReady(page);

  // The sealed 21-move scramble is what makes the replay checkable by hand.
  await expect(page.locator(".demo-scramble-move")).toHaveCount(21);
  await expect(page.locator(".demo-scramble-tape")).toHaveAttribute(
    "aria-label",
    "Starting scramble: R' L2 F L2 U R D F' U2 L B2 D2 F2 L2 B2 R2 U F2 B2 L2 D",
  );
  await expect(page.locator(".demo-scramble-tape")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Starting scramble" }),
  ).toBeVisible();
  // The heading and the chips carry it. No prose explains a scramble.
  await expect(page.locator(".demo-scramble-note")).toHaveCount(0);

  // The two panels must never overlap, and nothing inside either may spill
  // past its own card. This is checked across supported desktop widths.
  for (const width of [1024, 1280, 1440, 1920, 2000]) {
    await page.setViewportSize({ width, height: 900 });
    const audit = await page.evaluate(() => {
      const stage = document.querySelector(".demo-stage")!;
      const moves = document.querySelector(".demo-moves")!;
      const sb = stage.getBoundingClientRect();
      const mb = moves.getBoundingClientRect();
      const spill: string[] = [];
      for (const panel of [stage, moves]) {
        const box = panel.getBoundingClientRect();
        const style = getComputedStyle(panel);
        const limit =
          box.right -
          parseFloat(style.paddingRight) -
          parseFloat(style.borderRightWidth);
        panel.querySelectorAll("*").forEach((node) => {
          if (node.getBoundingClientRect().right > limit + 0.5) {
            spill.push(String(node.className));
          }
        });
      }
      return {
        overlap:
          sb.right > mb.left + 0.5 &&
          mb.right > sb.left + 0.5 &&
          sb.bottom > mb.top + 0.5 &&
          mb.bottom > sb.top + 0.5,
        spill,
        pageOverflow:
          document.documentElement.scrollWidth -
          document.documentElement.clientWidth,
      };
    });
    expect(audit, `at ${width}px`).toEqual({
      overlap: false,
      spill: [],
      pageOverflow: 0,
    });
  }

  // Wide viewport: the clip and the move timeline occupy one row, so neither
  // scrolls the other off screen while the clip plays.
  await page.setViewportSize({ width: 1440, height: 900 });
  const frame = await page.locator(".demo-frame").boundingBox();
  const controls = await page.locator(".demo-transport-row").boundingBox();
  const scrubber = await page.locator(".frame-scrubber").boundingBox();
  const timeline = await page.locator(".demo-timeline-track").boundingBox();
  expect(frame).not.toBeNull();
  expect(controls).not.toBeNull();
  expect(scrubber).not.toBeNull();
  expect(timeline).not.toBeNull();
  expect(
    Math.abs(
      controls!.x +
        controls!.width / 2 -
        (frame!.x + frame!.width / 2),
    ),
  ).toBeLessThanOrEqual(1);
  expect(scrubber!.y).toBeGreaterThanOrEqual(frame!.y + frame!.height);
  expect(timeline!.x).toBeGreaterThan(frame!.x + frame!.width);
  expect(timeline!.y).toBeLessThan(frame!.y + frame!.height);
  await expect(page.locator(".demo-move-strip")).toBeHidden();

  // Clicking the strip seeks without moving the page.
  const pageScroll = await page.evaluate(() => window.scrollY);
  const strip = (await page.locator(".demo-timeline-track").boundingBox())!;
  await page.mouse.click(strip.x + strip.width * 0.9, strip.y + strip.height / 2);
  await expect(page.locator(".demo-timeline-readout")).toContainText(" of 77");
  expect(await page.evaluate(() => window.scrollY)).toBe(pageScroll);

});

test("the demo transport steps one frame and answers the keyboard", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockWorkbench(page);
  await blockReleaseClip(page);
  await page.goto("/demo");
  await demoReady(page);

  // One transport only. The native controls fought the custom arrow keys.
  await expect(page.locator("video.demo-video")).toHaveCount(0);
  expect(await page.evaluate(() => document.querySelector(".demo-video"))).toBe(
    null,
  );

  const readout = page.locator(".demo-frame-readout");
  const caption = page.locator(".move-replay-caption");
  await expect(readout).toContainText("frame 0 of 5803");

  await page.getByRole("button", { name: "Forward one frame" }).click();
  await expect(readout).toContainText("frame 1 of 5803");
  await page.getByRole("button", { name: "Back one frame" }).click();
  await expect(readout).toContainText("frame 0 of 5803");

  await page.keyboard.press("ArrowRight");
  await expect(readout).toContainText("frame 1 of 5803");
  await page.keyboard.press("Shift+ArrowRight");
  await expect(readout).toContainText("frame 11 of 5803");

  // Up steps forward through the recorded smart-cube turn frames and down
  // steps back. The first three turns land on 120, 178, and 212.
  await page.keyboard.press("ArrowUp");
  await expect(readout).toContainText("frame 120 of 5803");
  await expect(caption).toContainText("Step 1 of 77: after U");
  await page.keyboard.press("ArrowUp");
  await expect(readout).toContainText("frame 178 of 5803");
  await expect(caption).toContainText("Step 2 of 77: after F");
  await page.keyboard.press("ArrowUp");
  await expect(readout).toContainText("frame 212 of 5803");
  await page.keyboard.press("ArrowDown");
  await expect(readout).toContainText("frame 178 of 5803");

  await page.keyboard.press("End");
  await expect(readout).toContainText("frame 5803 of 5803");
  await expect(caption).toContainText("Step 77 of 77");
  await page.keyboard.press("Home");
  await expect(readout).toContainText("frame 0 of 5803");

  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();

  // Typing in a field must never drive the transport.
  await page.keyboard.press("Home");
  await expect(readout).toContainText("frame 0 of 5803");
  await page.evaluate(() => {
    const field = document.createElement("input");
    field.id = "keyboard-guard-probe";
    field.type = "text";
    document.body.append(field);
    field.focus();
  });
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowUp");
  await expect(readout).toContainText("frame 0 of 5803");
  await page.evaluate(() =>
    document.getElementById("keyboard-guard-probe")?.remove(),
  );
});

test("the desktop-only notice replaces the workbench below 1000px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/api/**", (route) =>
    route.abort("connectionrefused"),
  );
  await page.goto("/");

  await expect(page).toHaveURL(/\/demo$/);
  await expect(page.locator(".desktop-only-notice")).toBeVisible();
  await expect(page.locator(".lab-shell")).toBeHidden();
});

test("Runs has no serious automated accessibility violations", async ({
  page,
}) => {
  await mockWorkbench(page);
  await page.goto("/runs");
  await expect(
    page.getByRole("heading", { name: "Runs", exact: true }),
  ).toBeVisible();

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  const serious = results.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact ?? ""),
  );
  expect(serious).toEqual([]);
});

test("run inspector has no serious automated accessibility violations", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await mockWorkbench(page, undefined, undefined, runHistory);
  await page.goto(`/runs?run=${JOB_A}`);
  await expect(
    page.getByRole("heading", { name: "capture-a.mov" }),
  ).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".tracker-workstation")).toBeVisible({
    timeout: 15_000,
  });

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  const serious = results.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact ?? ""),
  );
  expect(serious).toEqual([]);
});

test("the demo page has no serious automated accessibility violations", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await mockWorkbench(page);
  await blockReleaseClip(page);
  await page.goto("/demo");
  await demoReady(page);
  await expect(page.locator(".demo-detail")).toHaveCount(0);

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  const serious = results.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact ?? ""),
  );
  expect(serious).toEqual([]);
});
