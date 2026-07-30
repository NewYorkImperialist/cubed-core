import { expect, test, type Page } from "@playwright/test";

async function blockOptionalNetwork(page: Page) {
  await page.route("**/api/**", (route) =>
    route.abort("connectionrefused"),
  );
  await page.route("https://github.com/**", (route) =>
    route.abort("connectionrefused"),
  );
}

test("the root opens the demo without release lifecycle copy", async ({
  page,
}) => {
  await blockOptionalNetwork(page);
  await page.goto("/");

  await expect(page).toHaveURL(/\/demo$/);
  await expect(
    page.getByRole("heading", { name: /Reconstruction demo/ }),
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".start-route")).toHaveCount(0);

  const releaseFacingCopy = await page.locator("main").innerText();
  for (const phrase of [
    "Works now",
    "assets pending release",
    "Current state",
    "In progress",
  ]) {
    expect(releaseFacingCopy).not.toMatch(new RegExp(phrase, "i"));
  }
});

test("a required demo file can be retried without reloading the page", async ({
  page,
}) => {
  await blockOptionalNetwork(page);
  let resultAttempts = 0;
  let allowResult = false;
  await page.route("**/demo/gtd1/decode-result.json", async (route) => {
    resultAttempts += 1;
    if (!allowResult) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "temporary test failure" }),
      });
      return;
    }
    await route.continue();
  });

  await page.goto("/demo");
  await expect(
    page.getByRole("heading", { name: /Reconstruction demo/ }),
  ).toBeVisible();
  await expect(page.getByText("Reconstruction files unavailable")).toBeVisible();
  allowResult = true;
  await page
    .getByRole("button", { name: "Retry reconstruction files unavailable" })
    .click();

  await expect(
    page.getByRole("heading", { name: "Watch the reconstruction" }),
  ).toBeVisible({ timeout: 30_000 });
  expect(resultAttempts).toBeGreaterThanOrEqual(2);
});

test("overlay and timing failures retain the verified move replay", async ({
  page,
}) => {
  await blockOptionalNetwork(page);
  await page.route("**/demo/gtd1/overlay-track.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.route("**/demo/gtd1/ble-ground-truth.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );

  await page.goto("/demo");
  await expect(
    page.getByRole("heading", { name: "Watch the reconstruction" }),
  ).toBeVisible({ timeout: 30_000 });
  await expect(
    page.getByRole("alert").filter({ hasText: "Camera overlay unavailable" }),
  ).toBeVisible();
  await expect(
    page.getByRole("alert").filter({ hasText: "Move timing unavailable" }),
  ).toBeVisible();
  await expect(page.locator(".cube-state")).toBeVisible();
  await expect(page.locator(".move-replay-caption")).toContainText("Step 0");

  await page.getByRole("button", { name: "Next move" }).click();
  await expect(page.locator(".move-replay-caption")).toContainText("Step 1 of 77");
  await expect(page.locator(".demo-moves-note")).toContainText(
    "without claiming clip timing",
  );
});
