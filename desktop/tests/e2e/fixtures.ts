/**
 * Shared test fixtures for E2E tests.
 *
 * Extends the base Playwright test with custom fixtures that automatically
 * set up API mocking and provide helper utilities.
 */

import { test as base, expect, Page } from "@playwright/test";
import { setupApiMocks, setupApiMocksWithErrors, setupDisconnectedMocks } from "../mocks";

/**
 * Custom test fixtures
 */
type CustomFixtures = {
  /** Page with API mocks already set up */
  mockedPage: Page;
  /** Page with error mocks set up */
  errorPage: Page;
  /** Page with disconnected mocks set up */
  disconnectedPage: Page;
};

/**
 * Extended test with custom fixtures
 */
export const test = base.extend<CustomFixtures>({
  mockedPage: async ({ page }, use) => {
    await setupApiMocks(page);
    await use(page);
  },
  errorPage: async ({ page }, use) => {
    await setupApiMocksWithErrors(page);
    await use(page);
  },
  disconnectedPage: async ({ page }, use) => {
    await setupDisconnectedMocks(page);
    await use(page);
  },
});

export { expect };

/**
 * Helper to wait for the app to fully load.
 * Waits for the sidebar and initial content to be visible.
 */
export async function waitForAppLoad(page: Page): Promise<void> {
  // Wait for sidebar to be visible
  await page.waitForSelector(".sidebar", { state: "visible" });

  // Wait for the JARVIS logo
  await page.waitForSelector(".logo-text", { state: "visible" });
}

/**
 * Helper to navigate to a specific view.
 */
export async function navigateToView(
  page: Page,
  view: "dashboard" | "messages" | "health" | "settings"
): Promise<void> {
  // Click the nav button with the matching title
  const navButton = page.locator(`.nav-item[title="${capitalizeFirst(view)}"]`);

  // For "health", the title is "Health Status"
  if (view === "health") {
    await page.locator('.nav-item[title="Health Status"]').click();
  } else {
    await navButton.click();
  }

  // Wait for view to be active
  await page.waitForSelector(`.nav-item.active[title="${view === "health" ? "Health Status" : capitalizeFirst(view)}"]`);
}

function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Helper to select a conversation from the list.
 */
export async function selectConversation(
  page: Page,
  displayName: string
): Promise<void> {
  // Find the conversation by name and click it
  const conversation = page.locator(".conversation", {
    has: page.locator(".name", { hasText: displayName }),
  });
  await conversation.click();

  // Wait for it to become active
  await page.waitForSelector(".conversation.active");
}

/**
 * Helper to wait for messages to load in the message view.
 */
export async function waitForMessagesLoad(page: Page): Promise<void> {
  // Wait for either messages to appear or the "Select a conversation" placeholder
  await page.waitForSelector(".messages, .message-view .empty", {
    state: "visible",
  });
}

/**
 * Helper to get the connection status indicator.
 */
export async function getConnectionStatus(
  page: Page
): Promise<"connected" | "disconnected"> {
  const dot = page.locator(".status-dot");
  const hasConnectedClass = await dot.evaluate((el) =>
    el.classList.contains("connected")
  );
  return hasConnectedClass ? "connected" : "disconnected";
}
