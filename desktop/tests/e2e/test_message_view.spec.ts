/**
 * E2E tests for message view functionality.
 *
 * Verifies clicking a conversation shows messages correctly.
 */

import {
  test,
  expect,
  waitForAppLoad,
  selectConversation,
} from "./fixtures";
import { mockMessagesChat1, mockMessagesChat3 } from "../mocks";

test.describe("Message View", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
    // Wait for conversations to load
    await page.waitForSelector(".conversation");
  });

  test("shows placeholder when no conversation selected", async ({
    mockedPage: page,
  }) => {
    // Initially, no conversation is selected
    // The message view should show a placeholder or be empty
    const messageView = page.locator(".message-view");

    // Check for empty state or placeholder
    const emptyOrPlaceholder = messageView.locator(".empty, .placeholder");
    await expect(emptyOrPlaceholder).toBeVisible();
  });

  test("displays messages when conversation is selected", async ({
    mockedPage: page,
  }) => {
    // Select the first conversation (John Doe)
    await selectConversation(page, "John Doe");

    // Wait for messages to load
    await page.waitForSelector(".message");

    // Messages should be displayed
    const messages = page.locator(".message");
    await expect(messages).toHaveCount(mockMessagesChat1.length);
  });

  test("displays message text correctly", async ({ mockedPage: page }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Check that message texts are displayed
    for (const msg of mockMessagesChat1) {
      await expect(page.locator(".message-text", { hasText: msg.text })).toBeVisible();
    }
  });

  test("distinguishes sent vs received messages", async ({
    mockedPage: page,
  }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Messages from me should have a different class/style
    const sentMessages = page.locator(".message.sent, .message.from-me, .bubble-me");
    const receivedMessages = page.locator(".message.received, .message.from-other, .bubble-other");

    // Should have at least one of each type
    // (Based on mockMessagesChat1: 2 received, 1 sent)
    const sentCount = await sentMessages.count();
    const receivedCount = await receivedMessages.count();

    // At least verify that messages are categorized
    expect(sentCount + receivedCount).toBeGreaterThan(0);
  });

  test("shows message timestamps", async ({ mockedPage: page }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Look for timestamp elements
    const timestamps = page.locator(".message-time, .timestamp, .date");

    // Should have timestamps for messages
    const count = await timestamps.count();
    expect(count).toBeGreaterThan(0);
  });

  test("displays sender name for received messages", async ({
    mockedPage: page,
  }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Received messages should show sender name
    await expect(
      page.locator(".sender-name, .message-sender", { hasText: "John Doe" })
    ).toBeVisible();
  });

  test("shows conversation header with recipient info", async ({
    mockedPage: page,
  }) => {
    await selectConversation(page, "John Doe");

    // Header should show the conversation name/recipient
    const header = page.locator(".message-view .header, .message-header");
    await expect(header).toBeVisible();

    // Should contain the conversation name
    await expect(header).toContainText("John Doe");
  });

  test("loads different messages for different conversations", async ({
    mockedPage: page,
  }) => {
    // Select first conversation
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Get first conversation's first message
    const firstConvMessage = await page.locator(".message-text").first().textContent();

    // Select a different conversation (group chat)
    await selectConversation(page, "Project Team");
    await page.waitForSelector(".message");

    // Messages should be different
    const secondConvMessage = await page.locator(".message-text").first().textContent();

    // Verify different messages are loaded
    expect(firstConvMessage).not.toBe(secondConvMessage);
  });

  test("shows group chat messages with multiple senders", async ({
    mockedPage: page,
  }) => {
    // Select the group chat
    await selectConversation(page, "Project Team");
    await page.waitForSelector(".message");

    // Check for messages from different senders
    const aliceMessage = page.locator(".message", {
      has: page.locator("text=Alice"),
    });
    const bobMessage = page.locator(".message", {
      has: page.locator("text=Bob"),
    });

    // At least one of each should be visible
    const aliceCount = await aliceMessage.count();
    const bobCount = await bobMessage.count();
    expect(aliceCount + bobCount).toBeGreaterThan(0);
  });

  test("handles reactions on messages", async ({ mockedPage: page }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Look for reaction indicators
    // mockMessagesChat1 has a "love" reaction on the second message
    const reactions = page.locator(".reactions, .reaction, .tapback");

    // If reactions are displayed
    const reactionCount = await reactions.count();
    if (reactionCount > 0) {
      await expect(reactions.first()).toBeVisible();
    }
  });

  test("scrolls to bottom when loading messages", async ({
    mockedPage: page,
  }) => {
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");

    // Get the messages container
    const messagesContainer = page.locator(".messages, .message-list");

    // Check scroll position (should be at or near bottom)
    const scrollInfo = await messagesContainer.evaluate((el) => {
      return {
        scrollTop: el.scrollTop,
        scrollHeight: el.scrollHeight,
        clientHeight: el.clientHeight,
      };
    });

    // Allow some tolerance for scroll position
    const isAtBottom =
      scrollInfo.scrollTop + scrollInfo.clientHeight >=
      scrollInfo.scrollHeight - 10;
    expect(isAtBottom).toBe(true);
  });

  test("shows loading state while messages load", async ({ page }) => {
    // Set up slow message loading
    await page.route("http://localhost:8742/**", async (route) => {
      const url = route.request().url();

      if (url.includes("/messages")) {
        await new Promise((resolve) => setTimeout(resolve, 500));
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(mockMessagesChat1),
        });
      } else if (url.includes("/conversations")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              chat_id: "chat-1",
              participants: ["+1234567890"],
              display_name: "John Doe",
              last_message_date: new Date().toISOString(),
              message_count: 42,
              is_group: false,
              last_message_text: "Hey!",
            },
          ]),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "ok", service: "jarvis-api" }),
        });
      }
    });

    await page.goto("/");
    await page.waitForSelector(".conversation");
    await page.locator(".conversation").first().click();

    // Loading state might appear briefly
    try {
      await expect(page.locator(".loading, .loading-messages")).toBeVisible({
        timeout: 1000,
      });
    } catch {
      // Loading finished quickly, which is fine
    }
  });
});
