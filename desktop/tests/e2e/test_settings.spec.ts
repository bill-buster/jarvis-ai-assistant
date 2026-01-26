/**
 * E2E tests for settings page functionality.
 *
 * Verifies settings page opens and saves changes correctly.
 */

import { test, expect, waitForAppLoad, navigateToView } from "./fixtures";
import { mockModels, mockSettingsResponse } from "../mocks";

test.describe("Settings Page", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
    await navigateToView(page, "settings");
  });

  test("shows settings page header", async ({ mockedPage: page }) => {
    await expect(page.locator("h1")).toHaveText("Settings");
  });

  test("displays model selection section", async ({ mockedPage: page }) => {
    // Model section should be visible
    const modelSection = page.locator(".section", { hasText: "Model" });
    await expect(modelSection).toBeVisible();

    // Should show model cards
    const modelCards = page.locator(".model-card");
    await expect(modelCards).toHaveCount(mockModels.length);
  });

  test("shows currently selected model", async ({ mockedPage: page }) => {
    // Find the selected model card
    const selectedModel = page.locator(".model-card.selected");
    await expect(selectedModel).toBeVisible();

    // Should show the model name
    await expect(selectedModel).toContainText(mockModels[0].name);
  });

  test("shows recommended badge on recommended model", async ({
    mockedPage: page,
  }) => {
    // Find model with recommended badge
    const recommendedBadge = page.locator(".badge.recommended");
    await expect(recommendedBadge).toBeVisible();
    await expect(recommendedBadge).toHaveText("Recommended");
  });

  test("shows loaded badge on loaded model", async ({ mockedPage: page }) => {
    // Find model with loaded badge
    const loadedBadge = page.locator(".badge.loaded");
    await expect(loadedBadge).toBeVisible();
    await expect(loadedBadge).toHaveText("Loaded");
  });

  test("shows model details (size, quality, RAM)", async ({
    mockedPage: page,
  }) => {
    // First model card
    const firstModel = page.locator(".model-card").first();

    // Should show size
    await expect(firstModel).toContainText(`${mockModels[0].size_gb} GB`);

    // Should show quality tier
    await expect(firstModel).toContainText("Basic");

    // Should show RAM requirement
    await expect(firstModel).toContainText(
      `Requires ${mockModels[0].ram_requirement_gb} GB RAM`
    );
  });

  test("displays generation settings section", async ({ mockedPage: page }) => {
    const generationSection = page.locator(".section", {
      hasText: "Generation",
    });
    await expect(generationSection).toBeVisible();
  });

  test("shows temperature slider", async ({ mockedPage: page }) => {
    const tempLabel = page.locator("label", { hasText: "Temperature" });
    await expect(tempLabel).toBeVisible();

    // Slider should be present
    const tempSlider = page.locator("#temperature");
    await expect(tempSlider).toBeVisible();

    // Current value should be displayed
    await expect(
      page.locator("label", { hasText: "Temperature" }).locator(".value")
    ).toHaveText(mockSettingsResponse.generation.temperature.toFixed(1));
  });

  test("shows max tokens sliders", async ({ mockedPage: page }) => {
    // Reply max tokens
    await expect(
      page.locator("label", { hasText: "Reply Max Tokens" })
    ).toBeVisible();
    await expect(page.locator("#maxTokensReply")).toBeVisible();

    // Summary max tokens
    await expect(
      page.locator("label", { hasText: "Summary Max Tokens" })
    ).toBeVisible();
    await expect(page.locator("#maxTokensSummary")).toBeVisible();
  });

  test("displays suggestions settings section", async ({
    mockedPage: page,
  }) => {
    const suggestionsSection = page.locator(".section", {
      hasText: "Suggestions",
    });
    await expect(suggestionsSection).toBeVisible();
  });

  test("shows auto-suggest toggle", async ({ mockedPage: page }) => {
    const toggleLabel = page.locator("label", {
      hasText: "Auto-suggest replies",
    });
    await expect(toggleLabel).toBeVisible();

    // Toggle button should be present and "on" by default
    const toggle = page.locator(".toggle-btn");
    await expect(toggle).toBeVisible();
    await expect(toggle).toHaveClass(/on/);
  });

  test("can toggle auto-suggest off and on", async ({ mockedPage: page }) => {
    const toggle = page.locator(".toggle-btn");

    // Initially on
    await expect(toggle).toHaveClass(/on/);

    // Click to turn off
    await toggle.click();
    await expect(toggle).not.toHaveClass(/on/);

    // Click to turn back on
    await toggle.click();
    await expect(toggle).toHaveClass(/on/);
  });

  test("shows conditional settings when auto-suggest is on", async ({
    mockedPage: page,
  }) => {
    // When auto-suggest is on, should show suggestion count slider
    await expect(
      page.locator("label", { hasText: "Suggestions Count" })
    ).toBeVisible();
    await expect(
      page.locator("label", { hasText: "Context Messages (Replies)" })
    ).toBeVisible();
    await expect(
      page.locator("label", { hasText: "Context Messages (Summaries)" })
    ).toBeVisible();
  });

  test("hides conditional settings when auto-suggest is off", async ({
    mockedPage: page,
  }) => {
    // Turn off auto-suggest
    await page.locator(".toggle-btn").click();

    // Conditional settings should be hidden
    await expect(
      page.locator("label", { hasText: "Suggestions Count" })
    ).not.toBeVisible();
  });

  test("displays system information section", async ({ mockedPage: page }) => {
    const systemSection = page.locator(".section", { hasText: "System" });
    await expect(systemSection).toBeVisible();
    await expect(systemSection).toContainText("System information (read-only)");
  });

  test("shows system info values", async ({ mockedPage: page }) => {
    // System RAM
    await expect(
      page.locator(".info-row", { hasText: "System RAM" })
    ).toBeVisible();
    await expect(
      page.locator(".info-row", { hasText: "System RAM" }).locator(".info-value")
    ).toContainText("GB");

    // Current Memory Usage
    await expect(
      page.locator(".info-row", { hasText: "Current Memory Usage" })
    ).toBeVisible();

    // Model Status
    await expect(
      page.locator(".info-row", { hasText: "Model Status" })
    ).toBeVisible();

    // iMessage Access
    await expect(
      page.locator(".info-row", { hasText: "iMessage Access" })
    ).toBeVisible();
  });

  test("shows save and reset buttons", async ({ mockedPage: page }) => {
    await expect(page.locator(".btn-primary")).toHaveText("Save Settings");
    await expect(page.locator(".btn-secondary", { hasText: "Reset" })).toHaveText(
      "Reset to Defaults"
    );
  });

  test("save button saves settings", async ({ mockedPage: page }) => {
    // Click save
    await page.locator(".btn-primary").click();

    // Success message should appear
    await expect(page.locator(".success-banner")).toBeVisible();
    await expect(page.locator(".success-banner")).toContainText(
      "Settings saved successfully"
    );
  });

  test("can change temperature and save", async ({ mockedPage: page }) => {
    // Get the temperature slider
    const tempSlider = page.locator("#temperature");

    // Change temperature value
    await tempSlider.fill("0.9");

    // Value display should update
    await expect(
      page.locator("label", { hasText: "Temperature" }).locator(".value")
    ).toHaveText("0.9");

    // Save changes
    await page.locator(".btn-primary").click();
    await expect(page.locator(".success-banner")).toBeVisible();
  });

  test("can select different model", async ({ mockedPage: page }) => {
    // Find a non-selected, downloaded model
    const secondModel = page.locator(".model-card").nth(1);

    // Click to select it
    await secondModel.click();

    // It should become selected
    await expect(secondModel).toHaveClass(/selected/);
  });

  test("shows download button for not-downloaded models", async ({
    mockedPage: page,
  }) => {
    // Find the model that isn't downloaded (Qwen 2.5 3B)
    const notDownloadedModel = page.locator(".model-card", {
      hasText: "Qwen 2.5 3B",
    });

    // Should have download button
    const downloadBtn = notDownloadedModel.locator("button", {
      hasText: "Download",
    });
    await expect(downloadBtn).toBeVisible();
  });

  test("reset to defaults resets values", async ({ mockedPage: page }) => {
    // Change temperature
    await page.locator("#temperature").fill("0.9");

    // Reset to defaults
    await page.locator(".btn-secondary", { hasText: "Reset" }).click();

    // Temperature should be back to default (0.7)
    await expect(
      page.locator("label", { hasText: "Temperature" }).locator(".value")
    ).toHaveText("0.7");
  });

  test("shows error when save fails", async ({ page }) => {
    // Setup mock that fails on save
    await page.route("http://localhost:8742/settings", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(mockSettingsResponse),
        });
      } else if (route.request().method() === "PUT") {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({
            error: "Save failed",
            detail: "Failed to save settings",
          }),
        });
      }
    });

    await page.route("http://localhost:8742/settings/models", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockModels),
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
    await page.locator('.nav-item[title="Settings"]').click();
    await page.waitForSelector("h1");

    // Try to save
    await page.locator(".btn-primary").click();

    // Error should appear
    await expect(page.locator(".error-banner")).toBeVisible();
  });
});
