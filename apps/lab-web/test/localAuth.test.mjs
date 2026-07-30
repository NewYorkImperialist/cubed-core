import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

const apiSource = readFileSync(
  resolve(import.meta.dirname, "../src/api.ts"),
  "utf8",
);
const viteConfigSource = readFileSync(
  resolve(import.meta.dirname, "../vite.config.ts"),
  "utf8",
);

test("API requests bootstrap one local session before applying admin auth", () => {
  assert.match(
    apiSource,
    /localSessionAttempt \?\?= bootstrapLocalSession\(\);/,
  );
  assert.match(
    apiSource,
    /if \(!conclusive && localSessionAttempt === attempt\) \{\s*localSessionAttempt = null;/,
  );
  assert.match(
    apiSource,
    /await ensureAdminToken\(\);\s*const headers = new Headers\(init\?\.headers\);\s*applyAdminToken\(headers\);/,
  );
  assert.match(
    apiSource,
    /fetchWithTimeout\(`\$\{API_BASE\}\/api\/local-session`, \{\s*method: "POST",\s*cache: "no-store",\s*credentials: "omit"/,
  );
});

test("a successful bootstrap reuses the token for protected HTTP requests", () => {
  assert.match(
    apiSource,
    /typeof payload\.admin_token === "string" && payload\.admin_token[\s\S]*?setAdminToken\(payload\.admin_token\);/,
  );
});

test("a stale loopback token refreshes once after auth rejection", () => {
  assert.match(
    apiSource,
    /async function bootstrapLocalSession\(force = false\)/,
  );
  assert.match(apiSource, /if \(adminToken && !force\) return true;/);
  assert.match(
    apiSource,
    /const staleToken = adminToken;[\s\S]*?await bootstrapLocalSession\(true\);[\s\S]*?adminToken !== staleToken/,
  );
  assert.match(
    apiSource,
    /response\.status !== 401 && response\.status !== 403/,
  );
  assert.equal(
    apiSource.match(
      /response = await retryWithFreshLocalSession\(response, path, init, timeoutMs\);/g,
    )?.length,
    2,
    "JSON and blob requests must both recover from a restarted loopback service",
  );
});

test("the Vite API proxy preserves the exact loopback browser origin", () => {
  assert.match(
    viteConfigSource,
    /"\/api": \{[\s\S]*?changeOrigin: false,[\s\S]*?ws: true,/,
  );
});
