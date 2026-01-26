/**
 * E2E tests for health status panel.
 *
 * Verifies health panel shows system information correctly.
 */

import { test, expect, waitForAppLoad, navigateToView } from "./fixtures";
import { mockHealthResponse, mockHealthResponseDegraded } from "../mocks";

test.describe("Health Status Panel", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
    await navigateToView(page, "health");
  });

  test("shows health page header", async ({ mockedPage: page }) => {
    // Header should display "System Health"
    await expect(page.locator("h1")).toHaveText("System Health");
  });

  test("shows system status banner", async ({ mockedPage: page }) => {
    // Status banner should be visible
    const statusBanner = page.locator(".status-banner");
    await expect(statusBanner).toBeVisible();

    // Should show "healthy" status
    await expect(statusBanner).toHaveClass(/healthy/);
    await expect(statusBanner).toContainText("healthy");
  });

  test("displays memory metrics", async ({ mockedPage: page }) => {
    // Memory card should be visible
    const memoryCard = page.locator(".metric-card", { hasText: "Memory" });
    await expect(memoryCard).toBeVisible();

    // Should show available memory
    await expect(memoryCard).toContainText(
      `${mockHealthResponse.memory_available_gb.toFixed(1)} GB`
    );

    // Should show memory mode
    await expect(memoryCard).toContainText(mockHealthResponse.memory_mode);
  });

  test("displays JARVIS process metrics", async ({ mockedPage: page }) => {
    // Process card should be visible
    const processCard = page.locator(".metric-card", {
      hasText: "JARVIS Process",
    });
    await expect(processCard).toBeVisible();

    // Should show RSS memory
    await expect(processCard).toContainText(
      `${mockHealthResponse.jarvis_rss_mb.toFixed(0)} MB`
    );

    // Should show VMS memory
    await expect(processCard).toContainText("Virtual:");
  });

  test("displays AI model status", async ({ mockedPage: page }) => {
    // Model card should be visible
    const modelCard = page.locator(".metric-card", { hasText: "AI Model" });
    await expect(modelCard).toBeVisible();

    // Should show "Loaded" since model_loaded is true in mock
    await expect(modelCard).toContainText("Loaded");
    await expect(modelCard).toContainText("Ready for inference");
  });

  test("displays iMessage access status", async ({ mockedPage: page }) => {
    // iMessage card should be visible
    const imessageCard = page.locator(".metric-card", {
      hasText: "iMessage Access",
    });
    await expect(imessageCard).toBeVisible();

    // Should show "Connected" since imessage_access is true in mock
    await expect(imessageCard).toContainText("Connected");
    await expect(imessageCard).toContainText("Full Disk Access granted");
  });

  test("shows refresh button", async ({ mockedPage: page }) => {
    const refreshBtn = page.locator(".refresh-btn");
    await expect(refreshBtn).toBeVisible();
    await expect(refreshBtn).toContainText("Refresh");
  });

  test("refresh button updates health data", async ({ mockedPage: page }) => {
    // Click refresh
    await page.locator(".refresh-btn").click();

    // Button should show refreshing state
    await expect(page.locator(".refresh-btn")).toContainText("Refresh");

    // Data should still be displayed after refresh
    await expect(page.locator(".status-banner.healthy")).toBeVisible();
  });

  test("shows memory usage bar", async ({ mockedPage: page }) => {
    // Memory bar should be visible
    const memoryBar = page.locator(".metric-bar");
    await expect(memoryBar.first()).toBeVisible();

    // Fill should be present
    const fill = page.locator(".metric-fill");
    await expect(fill.first()).toBeVisible();
  });

  test("shows degraded status when system is degraded", async ({ page }) => {
    // Mock degraded health response
    await page.route("http://localhost:8742/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockHealthResponseDegraded),
      });
    });

    // Mock ping
    await page.route("http://localhost:8742/", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok", service: "jarvis-api" }),
      });
    });

    await page.goto("/");
    await page.waitForSelector(".sidebar");
    await page.locator('.nav-item[title="Health Status"]').click();

    // Should show degraded status
    await expect(page.locator(".status-banner.degraded")).toBeVisible();
    await expect(page.locator(".status-banner")).toContainText("degraded");
  });

  test("shows issues section when details present", async ({ page }) => {
    // Mock health with issues
    await page.route("http://localhost:8742/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockHealthResponseDegraded),
      });
    });

    await page.route("http://localhost:8742/", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok", service: "jarvis-api" }),
      });
    });

    await page.goto("/");
    await page.waitForSelector(".sidebar");
    await page.locator('.nav-item[title="Health Status"]').click();

    // Issues section should be visible
    await expect(page.locator(".details")).toBeVisible();
    await expect(page.locator(".details")).toContainText("Issues");
    await expect(page.locator(".details")).toContainText("memory");
  });

  test("shows error banner when health check fails", async ({
    errorPage: page,
  }) => {
    await page.goto("/");
    await page.waitForSelector(".sidebar");
    await page.locator('.nav-item[title="Health Status"]').click();

    // Error banner should appear
    await expect(page.locator(".error-banner")).toBeVisible({ timeout: 5000 });
  });

  test("navigates to health from sidebar", async ({ mockedPage: page }) => {
    // Go back to messages first
    await page.locator('.nav-item[title="Messages"]').click();

    // Then click health
    await page.locator('.nav-item[title="Health Status"]').click();

    // Health nav should be active
    await expect(page.locator('.nav-item[title="Health Status"]')).toHaveClass(
      /active/
    );

    // Health content should be visible
    await expect(page.locator("h1")).toHaveText("System Health");
  });
});
