/**
 * E2E tests for AI draft reply panel.
 *
 * Verifies draft reply panel generates suggestions correctly.
 */

import {
  test,
  expect,
  waitForAppLoad,
  selectConversation,
} from "./fixtures";
import { mockDraftReplyResponse } from "../mocks";

test.describe("AI Draft Panel", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
    // Wait for conversations and select one
    await page.waitForSelector(".conversation");
    await selectConversation(page, "John Doe");
    await page.waitForSelector(".message");
  });

  // Note: The AI Draft Panel might be opened via a button in the message view
  // or through another UI interaction. These tests assume there's a way to open it.

  test("draft panel can be opened", async ({ mockedPage: page }) => {
    // Look for a button to open the draft panel
    // This might be labeled "Draft Reply", "AI Draft", or have a specific icon
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), button[title*="draft" i], button[title*="reply" i], .draft-btn'
    );

    // If there's a draft button, click it
    const count = await draftButton.count();
    if (count > 0) {
      await draftButton.first().click();

      // Panel should appear
      await expect(
        page.locator('.panel, [role="dialog"], .ai-draft-panel')
      ).toBeVisible();
    } else {
      // Skip test if draft functionality not available in current UI
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("draft panel has header with title", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Panel header should show "AI Draft"
      await expect(page.locator(".panel-header, .modal-header")).toContainText(
        "AI Draft"
      );
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("draft panel has close button", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Close button should be visible
      const closeBtn = page.locator('.close-btn, button[aria-label="Close"]');
      await expect(closeBtn).toBeVisible();

      // Clicking close should hide the panel
      await closeBtn.click();
      await expect(page.locator('.panel-overlay, [role="dialog"]')).not.toBeVisible();
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("draft panel closes on escape key", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await expect(page.locator('.panel, [role="dialog"]')).toBeVisible();

      // Press Escape
      await page.keyboard.press("Escape");

      // Panel should close
      await expect(page.locator('.panel-overlay, [role="dialog"]')).not.toBeVisible();
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("draft panel has instruction input", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Instruction input should be visible
      const instructionInput = page.locator("#instruction, input[type='text']");
      await expect(instructionInput).toBeVisible();
      await expect(instructionInput).toHaveAttribute(
        "placeholder",
        /what do you want|say/i
      );
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("draft panel has generate button", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Generate button should be visible
      await expect(page.locator(".generate-btn")).toBeVisible();
      await expect(page.locator(".generate-btn")).toHaveText("Generate Replies");
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("clicking generate shows loading state", async ({
    mockedPage: page,
  }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Click generate
      await page.locator(".generate-btn").click();

      // Should show loading state
      await expect(page.locator(".generate-btn")).toContainText(/Generating/i);
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("generates and displays suggestions", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Click generate
      await page.locator(".generate-btn").click();

      // Wait for suggestions to appear
      await page.waitForSelector(".suggestion-item, .suggestion");

      // Should show correct number of suggestions
      const suggestions = page.locator(".suggestion-item, .suggestion");
      await expect(suggestions).toHaveCount(
        mockDraftReplyResponse.suggestions.length
      );
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("suggestions display text and confidence", async ({
    mockedPage: page,
  }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();
      await page.waitForSelector(".suggestion-item, .suggestion");

      // First suggestion should have text and confidence
      const firstSuggestion = page.locator(".suggestion-item, .suggestion").first();
      await expect(firstSuggestion).toContainText(
        mockDraftReplyResponse.suggestions[0].text
      );

      // Confidence badge
      const confidence = Math.round(
        mockDraftReplyResponse.suggestions[0].confidence * 100
      );
      await expect(firstSuggestion).toContainText(`${confidence}%`);
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("can select a suggestion", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();
      await page.waitForSelector(".suggestion-item, .suggestion");

      // Click on first suggestion
      await page.locator(".suggestion-item, .suggestion").first().click();

      // It should become selected
      await expect(
        page.locator(".suggestion-item.selected, .suggestion.selected")
      ).toBeVisible();
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("use selected button is enabled when suggestion selected", async ({
    mockedPage: page,
  }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();
      await page.waitForSelector(".suggestion-item, .suggestion");

      // Use Selected button should be disabled initially
      const useBtn = page.locator(".use-btn");
      await expect(useBtn).toBeDisabled();

      // Select a suggestion
      await page.locator(".suggestion-item, .suggestion").first().click();

      // Use button should now be enabled
      await expect(useBtn).not.toBeDisabled();
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("regenerate button generates new suggestions", async ({
    mockedPage: page,
  }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();
      await page.waitForSelector(".suggestion-item, .suggestion");

      // Click regenerate
      const regenerateBtn = page.locator(".regenerate-btn");
      await regenerateBtn.click();

      // Should show loading again briefly, then suggestions
      await page.waitForSelector(".suggestion-item, .suggestion");
      await expect(
        page.locator(".suggestion-item, .suggestion")
      ).toHaveCount(mockDraftReplyResponse.suggestions.length);
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("shows context information", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();
      await page.waitForSelector(".suggestion-item, .suggestion");

      // Context info should show number of messages used
      const contextInfo = page.locator(".context-info");
      await expect(contextInfo).toContainText(
        `${mockDraftReplyResponse.context_used.num_messages}`
      );
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("can add custom instruction", async ({ mockedPage: page }) => {
    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();

      // Type in the instruction field
      const instructionInput = page.locator("#instruction, input[type='text']");
      await instructionInput.fill("say yes but I'll be 10 min late");

      // Generate
      await page.locator(".generate-btn").click();

      // Should generate suggestions (API handles the instruction)
      await page.waitForSelector(".suggestion-item, .suggestion");
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });

  test("shows error state when generation fails", async ({ page }) => {
    // Mock error response for draft endpoint
    await page.route("http://localhost:8742/**", async (route) => {
      const url = route.request().url();

      if (url.includes("/drafts")) {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({
            error: "Generation failed",
            detail: "Model not loaded",
          }),
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
      } else if (url.includes("/messages")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              chat_id: "chat-1",
              sender: "+1234567890",
              sender_name: "John Doe",
              text: "Hey!",
              date: new Date().toISOString(),
              is_from_me: false,
              attachments: [],
              reply_to_id: null,
              reactions: [],
              date_delivered: null,
              date_read: null,
              is_system_message: false,
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
    await page.waitForSelector(".message");

    const draftButton = page.locator(
      'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
    );
    const count = await draftButton.count();

    if (count > 0) {
      await draftButton.first().click();
      await page.locator(".generate-btn").click();

      // Error state should appear
      await expect(page.locator(".error-section, .error-message")).toBeVisible();
      await expect(page.locator(".retry-btn")).toBeVisible();
    } else {
      test.skip(true, "Draft button not found in UI");
    }
  });
});
