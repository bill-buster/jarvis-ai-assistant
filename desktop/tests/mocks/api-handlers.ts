/**
 * API route handlers for mocking the FastAPI backend in E2E tests.
 *
 * These handlers intercept HTTP requests to localhost:8742 and return
 * mock data, allowing tests to run without the actual backend.
 */

import { Page, Route } from "@playwright/test";
import {
  mockConversations,
  mockDraftReplyResponse,
  mockHealthResponse,
  mockMessagesChat1,
  mockMessagesChat3,
  mockModels,
  mockPingResponse,
  mockSettingsResponse,
} from "./api-data";

const API_BASE = "http://localhost:8742";

/**
 * Setup all API mocks for a page.
 * Call this in beforeEach to ensure all API calls are intercepted.
 */
export async function setupApiMocks(page: Page): Promise<void> {
  // Health endpoint
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockHealthResponse),
    });
  });

  // Ping endpoint
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPingResponse),
    });
  });

  // Conversations list
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockConversations),
      });
    } else {
      await route.continue();
    }
  });

  // Individual conversation
  await page.route(
    `${API_BASE}/conversations/chat-*`,
    async (route: Route) => {
      const url = route.request().url();

      // Check if it's a messages request
      if (url.includes("/messages")) {
        const chatId = url.split("/conversations/")[1].split("/")[0];
        const messages =
          chatId === "chat-1"
            ? mockMessagesChat1
            : chatId === "chat-3"
              ? mockMessagesChat3
              : [];

        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(messages),
        });
      } else {
        // It's a conversation detail request
        const chatId = url.split("/conversations/")[1].split("?")[0];
        const conversation = mockConversations.find((c) => c.chat_id === chatId);

        if (conversation) {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(conversation),
          });
        } else {
          await route.fulfill({
            status: 404,
            contentType: "application/json",
            body: JSON.stringify({ error: "Not found", detail: "Conversation not found" }),
          });
        }
      }
    }
  );

  // Settings endpoints
  await page.route(`${API_BASE}/settings`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockSettingsResponse),
      });
    } else if (route.request().method() === "PUT") {
      // Return the updated settings (just echo back for tests)
      const body = route.request().postDataJSON();
      const updated = {
        ...mockSettingsResponse,
        ...body,
        generation: { ...mockSettingsResponse.generation, ...body?.generation },
        behavior: { ...mockSettingsResponse.behavior, ...body?.behavior },
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(updated),
      });
    } else {
      await route.continue();
    }
  });

  // Models list
  await page.route(`${API_BASE}/settings/models`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockModels),
    });
  });

  // Model download
  await page.route(
    `${API_BASE}/settings/models/*/download`,
    async (route: Route) => {
      const url = route.request().url();
      const modelId = url.split("/models/")[1].split("/download")[0];

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          model_id: modelId,
          status: "completed",
          progress: 100,
          error: null,
        }),
      });
    }
  );

  // Model activate
  await page.route(
    `${API_BASE}/settings/models/*/activate`,
    async (route: Route) => {
      const url = route.request().url();
      const modelId = url.split("/models/")[1].split("/activate")[0];

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          success: true,
          model_id: modelId,
          error: null,
        }),
      });
    }
  );

  // Draft replies endpoint
  await page.route(
    `${API_BASE}/conversations/*/drafts`,
    async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockDraftReplyResponse),
      });
    }
  );
}

/**
 * Setup API mocks that simulate errors.
 * Use this to test error handling in the UI.
 */
export async function setupApiMocksWithErrors(page: Page): Promise<void> {
  // Health endpoint returns error
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Internal Server Error",
        detail: "Failed to check system health",
      }),
    });
  });

  // Ping endpoint returns error (simulates backend down)
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await route.abort("connectionrefused");
  });

  // Conversations list returns error
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Database Error",
        detail: "Failed to fetch conversations",
      }),
    });
  });
}

/**
 * Setup mocks for disconnected state.
 */
export async function setupDisconnectedMocks(page: Page): Promise<void> {
  // All requests fail with connection refused
  await page.route(`${API_BASE}/**`, async (route: Route) => {
    await route.abort("connectionrefused");
  });
}
