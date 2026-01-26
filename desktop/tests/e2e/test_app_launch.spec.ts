/**
 * E2E tests for app launch and initial state.
 *
 * Verifies the app opens without errors and displays expected initial content.
 */

import { test, expect, waitForAppLoad, getConnectionStatus } from "./fixtures";

test.describe("App Launch", () => {
  test("app opens without errors", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Verify no console errors during load
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    // App should display
    await expect(page.locator(".app")).toBeVisible();

    // Give time for any async errors
    await page.waitForTimeout(1000);

    // Filter out expected errors (like network errors when mocking)
    const unexpectedErrors = consoleErrors.filter(
      (err) => !err.includes("Failed to fetch") && !err.includes("NetworkError")
    );
    expect(unexpectedErrors).toHaveLength(0);
  });

  test("displays JARVIS logo", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Check for the J logo icon
    await expect(page.locator(".logo-icon")).toHaveText("J");

    // Check for the JARVIS text
    await expect(page.locator(".logo-text")).toHaveText("JARVIS");
  });

  test("shows sidebar navigation", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // All nav items should be present
    await expect(page.locator('.nav-item[title="Dashboard"]')).toBeVisible();
    await expect(page.locator('.nav-item[title="Messages"]')).toBeVisible();
    await expect(page.locator('.nav-item[title="Health Status"]')).toBeVisible();
    await expect(page.locator('.nav-item[title="Settings"]')).toBeVisible();
  });

  test("shows connection status indicator", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Status indicator should be visible
    await expect(page.locator(".status")).toBeVisible();

    // With mocked API, should show connected
    const status = await getConnectionStatus(page);
    expect(status).toBe("connected");

    // Status text should show Connected
    await expect(page.locator(".status-text")).toHaveText("Connected");
  });

  test("shows disconnected when API unavailable", async ({
    disconnectedPage: page,
  }) => {
    await page.goto("/");

    // Wait for the sidebar at least
    await page.waitForSelector(".sidebar", { state: "visible" });

    // Give time for connection check to complete
    await page.waitForTimeout(2000);

    // Status should show disconnected
    await expect(page.locator(".status-dot.disconnected")).toBeVisible();
    await expect(page.locator(".status-text")).toHaveText("Disconnected");
  });

  test("defaults to messages view", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Messages nav item should be active
    await expect(page.locator('.nav-item[title="Messages"]')).toHaveClass(
      /active/
    );

    // Conversation list should be visible
    await expect(page.locator(".conversation-list")).toBeVisible();
  });

  test("applies dark theme styling", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Check that CSS variables are applied (dark theme colors)
    const bgColor = await page.evaluate(() => {
      return getComputedStyle(document.body).backgroundColor;
    });

    // The background should be dark (not white/light)
    // #1c1c1e in RGB is approximately rgb(28, 28, 30)
    expect(bgColor).toMatch(/rgb\(28,\s*28,\s*30\)|rgba\(28,\s*28,\s*30/);
  });

  test("is responsive at different viewport sizes", async ({
    mockedPage: page,
  }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    // Test at different sizes
    const sizes = [
      { width: 1200, height: 800 },
      { width: 1000, height: 700 },
      { width: 800, height: 600 },
    ];

    for (const size of sizes) {
      await page.setViewportSize(size);

      // Core elements should still be visible
      await expect(page.locator(".sidebar")).toBeVisible();
      await expect(page.locator(".app")).toBeVisible();
    }
  });
});
