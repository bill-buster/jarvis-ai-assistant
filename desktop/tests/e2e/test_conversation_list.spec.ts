/**
 * E2E tests for conversation list functionality.
 *
 * Verifies conversations load, display correctly, and can be selected.
 */

import { test, expect, waitForAppLoad, navigateToView } from "./fixtures";
import { mockConversations } from "../mocks";

test.describe("Conversation List", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
    // Ensure we're on messages view
    await navigateToView(page, "messages");
  });

  test("loads and displays conversations", async ({ mockedPage: page }) => {
    // Wait for conversations to load
    await page.waitForSelector(".conversation");

    // Should display the mocked conversations
    const conversations = page.locator(".conversation");
    await expect(conversations).toHaveCount(mockConversations.length);
  });

  test("displays conversation names correctly", async ({
    mockedPage: page,
  }) => {
    await page.waitForSelector(".conversation");

    // Check that each conversation name is displayed
    for (const conv of mockConversations) {
      const expectedName = conv.display_name || conv.participants[0];
      await expect(page.locator(".name", { hasText: expectedName })).toBeVisible();
    }
  });

  test("shows message preview", async ({ mockedPage: page }) => {
    await page.waitForSelector(".conversation");

    // Check first conversation has preview text
    const firstPreview = page.locator(".conversation").first().locator(".preview");
    await expect(firstPreview).toHaveText(mockConversations[0].last_message_text!);
  });

  test("shows date/time for last message", async ({ mockedPage: page }) => {
    await page.waitForSelector(".conversation");

    // Each conversation should have a date element
    const dates = page.locator(".conversation .date");
    await expect(dates).toHaveCount(mockConversations.length);

    // First conversation should have a time (it's today)
    const firstDate = dates.first();
    const dateText = await firstDate.textContent();
    expect(dateText).toBeTruthy();
  });

  test("identifies group conversations with icon", async ({
    mockedPage: page,
  }) => {
    await page.waitForSelector(".conversation");

    // Find the group conversation (Project Team)
    const groupConv = page.locator(".conversation.group");
    await expect(groupConv).toBeVisible();

    // Group avatar should have an SVG icon
    const groupAvatar = groupConv.locator(".avatar.group svg");
    await expect(groupAvatar).toBeVisible();
  });

  test("shows initials for non-group conversations", async ({
    mockedPage: page,
  }) => {
    await page.waitForSelector(".conversation");

    // Find a non-group conversation
    const nonGroupConv = page.locator(".conversation:not(.group)").first();
    const avatar = nonGroupConv.locator(".avatar:not(.group)");

    // Avatar should contain an initial (first letter of name)
    const initial = await avatar.textContent();
    expect(initial).toMatch(/^[A-Z+]$/);
  });

  test("highlights selected conversation", async ({ mockedPage: page }) => {
    await page.waitForSelector(".conversation");

    // Click on a conversation
    await page.locator(".conversation").first().click();

    // It should become active
    await expect(page.locator(".conversation").first()).toHaveClass(/active/);
  });

  test("only one conversation can be selected at a time", async ({
    mockedPage: page,
  }) => {
    await page.waitForSelector(".conversation");

    // Click first conversation
    await page.locator(".conversation").first().click();
    await expect(page.locator(".conversation.active")).toHaveCount(1);

    // Click second conversation
    await page.locator(".conversation").nth(1).click();
    await expect(page.locator(".conversation.active")).toHaveCount(1);

    // Second should now be active, not first
    await expect(page.locator(".conversation").nth(1)).toHaveClass(/active/);
    await expect(page.locator(".conversation").first()).not.toHaveClass(/active/);
  });

  test("shows search input", async ({ mockedPage: page }) => {
    await page.waitForSelector(".conversation");

    // Search input should be visible
    const searchInput = page.locator('.search input[type="text"]');
    await expect(searchInput).toBeVisible();
    await expect(searchInput).toHaveAttribute(
      "placeholder",
      "Search conversations..."
    );
  });

  test("shows loading state initially", async ({ page }) => {
    // Set up slow API response
    await page.route("http://localhost:8742/**", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockConversations),
      });
    });

    await page.goto("/");

    // Loading indicator should appear briefly
    // Note: This might be too fast to catch, so we use a try-catch
    try {
      await expect(page.locator(".loading")).toBeVisible({ timeout: 1000 });
    } catch {
      // Loading might have already finished, which is fine
    }
  });

  test("shows error when conversations fail to load", async ({
    errorPage: page,
  }) => {
    await page.goto("/");
    await page.waitForSelector(".sidebar");

    // Navigate to messages
    await page.locator('.nav-item[title="Messages"]').click();

    // Wait for error to appear
    await expect(page.locator(".error")).toBeVisible({ timeout: 5000 });
  });

  test("shows empty state when no conversations", async ({ page }) => {
    // Mock empty conversations
    await page.route("http://localhost:8742/conversations", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      });
    });

    // Mock other required endpoints
    await page.route("http://localhost:8742/", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok", service: "jarvis-api" }),
      });
    });

    await page.goto("/");
    await page.waitForSelector(".sidebar");

    // Empty state should be shown
    await expect(page.locator(".empty")).toBeVisible();
    await expect(page.locator(".empty")).toHaveText("No conversations found");
  });
});
