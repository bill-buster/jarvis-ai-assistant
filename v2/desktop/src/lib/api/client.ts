/**
 * JARVIS v2 API Client
 *
 * Simple, focused client for the v2 API endpoints.
 */

import type {
  ContactProfile,
  Conversation,
  ConversationListResponse,
  GenerateRepliesResponse,
  HealthResponse,
  Message,
  MessageListResponse,
  SendMessageResponse,
  Settings,
} from "./types";

const API_BASE = "http://localhost:8000";

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail: string | null = null
  ) {
    super(message);
    this.name = "APIError";
  }
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new APIError(
        error.detail || "Request failed",
        response.status,
        error.detail
      );
    }

    return response.json();
  }

  // Health
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  // Conversations
  async getConversations(limit: number = 50): Promise<Conversation[]> {
    const response = await this.request<ConversationListResponse>(
      `/conversations?limit=${limit}`
    );
    return response.conversations;
  }

  // Messages
  async getMessages(
    chatId: string,
    limit: number = 50,
    before?: string
  ): Promise<Message[]> {
    let url = `/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`;
    if (before) {
      url += `&before=${encodeURIComponent(before)}`;
    }
    const response = await this.request<MessageListResponse>(url);
    return response.messages;
  }

  // Reply Generation
  async generateReplies(
    chatId: string,
    numReplies: number = 3,
    signal?: AbortSignal
  ): Promise<GenerateRepliesResponse> {
    const response = await fetch(`${this.baseUrl}/generate/replies`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        num_replies: numReplies,
      }),
      signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new APIError(
        error.detail || "Request failed",
        response.status,
        error.detail
      );
    }

    return response.json();
  }

  // Send Message
  async sendMessage(chatId: string, text: string, isGroup: boolean = false): Promise<SendMessageResponse> {
    return this.request<SendMessageResponse>("/conversations/send", {
      method: "POST",
      body: JSON.stringify({
        chat_id: chatId,
        text: text,
        is_group: isGroup,
      }),
    });
  }

  // Settings
  async getSettings(): Promise<Settings> {
    return this.request<Settings>("/settings");
  }

  async updateSettings(settings: Partial<Settings>): Promise<Settings> {
    return this.request<Settings>("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  // Contact Profile
  async getContactProfile(chatId: string): Promise<ContactProfile> {
    return this.request<ContactProfile>(
      `/conversations/${encodeURIComponent(chatId)}/profile`
    );
  }

  // Index Preloading
  async preloadIndices(chatIds: string[]): Promise<{ preloading: number; already_cached: number }> {
    return this.request<{ preloading: number; already_cached: number; message: string }>(
      "/conversations/preload",
      {
        method: "POST",
        body: JSON.stringify({ chat_ids: chatIds }),
      }
    );
  }
}

export const api = new ApiClient();
