import { test, expect } from "@playwright/test";
test("actual locally packaged models and comparison, no mocked generation", async ({
  page, request,
}) => {
  const readiness = await request.get((process.env.NEXT_PUBLIC_API_ORIGIN || 'http://127.0.0.1:8000') + '/ready');
  expect(readiness.status(), 'All three verified packages are required for complete E2E acceptance').toBe(200);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link',{name:'Skip to content'})).toBeFocused();
  await page.getByRole("link", { name: /Open the playground/ }).click();
  await expect(page.getByText("Local inference connected")).toBeVisible();
  for (const name of ["Compact", "Balanced", "Quality"]) {
    await page.getByRole("button", { name: new RegExp(name) }).click();
    await page.getByLabel("New tokens").selectOption("64");
    await page.getByRole("button", { name: /Generate continuation/ }).click();
    await expect(page.getByText("64 new tokens")).toBeVisible({
      timeout: 90000,
    });
    await expect(page.locator(".generated")).toContainText("ROMEO:");
  }
  await page.screenshot({
    path: "../outputs/studio/desktop-playground.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Compare", exact: true }).click();
  await expect(page.getByText("Local inference connected")).toBeVisible();
  await page.getByLabel("New tokens").selectOption("64");
  await page.getByRole("button", { name: /Compare all three/ }).click();
  await expect(page.getByText("64 new tokens")).toHaveCount(3, {
    timeout: 90000,
  });
  await page.screenshot({
    path: "../outputs/studio/desktop-compare.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: /Research/ }).click();
  await expect(
    page.getByRole("heading", { name: /Capacity has a cost/ }),
  ).toBeVisible();
  await page.screenshot({
    path: "../outputs/studio/desktop-research.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
test("responsive mobile navigation and error handling", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("link", { name: /Open the playground/ }),
  ).toBeVisible();
  await page.getByRole("link", { name: /Open the playground/ }).click();
  await page.getByRole("button", { name: /Balanced/ }).click();
  await page.getByLabel("New tokens").selectOption("64");
  await page.getByRole("button", { name: /Generate continuation/ }).click();
  await expect(page.getByText("64 new tokens")).toBeVisible({ timeout: 90000 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../outputs/studio/mobile-playground.png",
    fullPage: true,
  });
  await page.route("**/v1/generate", (route) => route.abort());
  await page.getByRole("button", { name: /Regenerate/ }).click();
  await expect(page.locator(".output [role=alert]")).toBeVisible();
  await page.screenshot({
    path: "../outputs/studio/mobile-error.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 320, height: 740 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
