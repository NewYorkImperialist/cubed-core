import { defineConfig } from "@playwright/test";

// Local development uses the installed Chrome channel. CI installs
// Playwright's pinned Chromium build for reproducible GitHub runs.
const browserChannel = process.env.CI ? undefined : "chrome";

export default defineConfig({
  testDir: "./test/e2e",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  expect: {
    timeout: 15_000,
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: browserChannel,
    trace: "retain-on-failure",
  },
  webServer: {
    // Force a fresh dependency graph so a prior local Vite session cannot
    // leave worker entries pointing at an obsolete optimized-deps directory.
    command: "npm run dev -- --port 4173 --force",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
