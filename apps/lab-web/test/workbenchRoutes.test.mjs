import assert from "node:assert/strict";
import test from "node:test";

import {
  LEGACY_PUBLISHED_DEMO_HASH,
  PUBLISHED_DEMO_ROUTE,
  publishedDemoRedirectFor,
  WORKBENCH_NAV_GROUPS,
  WORKBENCH_NAV_ITEMS,
  workbenchNavIdForPath,
} from "../src/workbenchRoutes.ts";

test("navigation keeps execution separate from run inspection", () => {
  assert.equal(workbenchNavIdForPath("/"), "demo");
  assert.equal(workbenchNavIdForPath("/demo"), "demo");
  assert.equal(workbenchNavIdForPath("/decode"), "decode");
  assert.equal(workbenchNavIdForPath("/reconstruct"), "decode");
  assert.equal(workbenchNavIdForPath("/runs"), "runs");
  assert.equal(workbenchNavIdForPath("/track"), null);
  assert.equal(workbenchNavIdForPath("/process"), null);
  assert.equal(workbenchNavIdForPath("/analyze"), "runs");
  assert.equal(workbenchNavIdForPath("/analysis"), "runs");
  assert.equal(workbenchNavIdForPath("/capture"), "import");
  assert.equal(workbenchNavIdForPath("/capture/setup"), "import");
  assert.equal(workbenchNavIdForPath("/capture/record"), "import");
  assert.equal(workbenchNavIdForPath("/import"), "import");
  assert.equal(workbenchNavIdForPath("/label"), "label");
  assert.equal(workbenchNavIdForPath("/research"), null);
  assert.equal(workbenchNavIdForPath("/unknown"), null);
});

test("the flat nav list stays the groups in order", () => {
  assert.deepEqual(
    WORKBENCH_NAV_ITEMS.map((item) => item.id),
    WORKBENCH_NAV_GROUPS.flatMap((group) => group.items.map((item) => item.id)),
  );
  assert.deepEqual(
    WORKBENCH_NAV_ITEMS.slice(0, 3).map((item) => item.id),
    ["demo", "decode", "runs"],
  );
  for (const item of WORKBENCH_NAV_ITEMS) {
    assert.equal(workbenchNavIdForPath(item.to), item.id, item.to);
  }
});

test("legacy published-demo deep links still land on the dedicated demo", () => {
  assert.equal(PUBLISHED_DEMO_ROUTE, "/demo");
  assert.equal(LEGACY_PUBLISHED_DEMO_HASH, "#published-demo");
  for (const path of ["/runs", "/decode", "/reconstruct"]) {
    assert.equal(publishedDemoRedirectFor(path, "#published-demo"), "/demo");
  }
  assert.equal(publishedDemoRedirectFor("/reconstruct", ""), null);
  assert.equal(
    publishedDemoRedirectFor("/reconstruct", "#reconstruct-decode"),
    null,
  );
  assert.equal(publishedDemoRedirectFor("/capture", "#published-demo"), null);
  assert.equal(publishedDemoRedirectFor("/demo", "#published-demo"), null);
});
