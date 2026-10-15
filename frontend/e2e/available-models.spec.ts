/** Explicit partial integration; does not certify the missing E02 package. */
import { test, expect } from "@playwright/test";
test("genuine available B01/F01 integration and honest missing E02 slot", async ({
  page,
  context,
  request,
}) => {
  const ready = await request.get((process.env.NEXT_PUBLIC_API_ORIGIN || 'http://127.0.0.1:8000') + '/ready');
  test.skip(ready.status() === 200, 'Historical partial-availability workflow; full three-model acceptance is covered by studio.spec.ts');
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link',{name:'Skip to content'})).toBeFocused();
  await expect(page.getByText("Local inference connected")).toBeVisible();
  await page.screenshot({
    path: "../outputs/studio/desktop-home.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: /Open the playground/ }).click();
  for (const name of ["Balanced", "Quality"]) {
    await page.getByRole("button", { name: new RegExp(name) }).click();
    await page.getByLabel("New tokens").selectOption("64");
    await page.getByRole("button", { name: /Generate continuation/ }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByText("64 new tokens")).toBeVisible({
      timeout: 90000,
    });
    await expect(page.locator(".generated")).toContainText("ROMEO:");
  }
  await page.screenshot({
    path: "../outputs/studio/desktop-playground.png",
    fullPage: true,
  });
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const literalOutput = await page.locator('.generated').textContent();
  await page.getByRole('button', {name:/Copy output/}).click();
  await expect(page.getByText('Copied ✓')).toBeVisible();
  // Windows native clipboard converts LF to CRLF; application text stays literal.
  const clipboard = await page.evaluate(()=>navigator.clipboard.readText());
  expect(clipboard.replace(/\r\n/g, '\n')).toBe(literalOutput);
  await page.getByRole("link", { name: "Compare", exact: true }).click();
  await expect(page.getByText("Local inference connected")).toBeVisible();
  await page.getByLabel("New tokens").selectOption("64");
  await page.getByRole("button", { name: /Compare all three/ }).click();
  await expect(page.getByText("64 new tokens")).toHaveCount(2, {
    timeout: 90000,
  });
  await expect(
    page.getByText("Verified model artifact unavailable"),
  ).toBeVisible();
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
