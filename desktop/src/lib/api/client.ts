/**
 * API client for JARVIS backend
 */

import type {
  ActivateResponse,
  Conversation,
  DownloadStatus,
  DraftReplyResponse,
  HealthResponse,
  Message,
  ModelInfo,
  SearchFilters,
  SettingsResponse,
  SettingsUpdateRequest,
  SmartReplySuggestionsResponse,
  SummaryResponse,
  TopicsResponse,
} from "./types";

const API_BASE = "http://localhost:8742";

/**
 * Custom API error with additional details
 */
export class APIError extends Error {
  status: number;
  detail: string | null;

  constructor(message: string, status: number, detail: string | null = null) {
    super(message);
    this.name = "APIError";
    this.status = status;
    this.detail = detail;
  }
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
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
        error: "Request failed",
        detail: response.statusText,
      }));
      throw new APIError(
        error.error || "Request failed",
        response.status,
        error.detail || null
      );
    }

    return response.json();
  }

  // Health endpoints
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  async ping(): Promise<{ status: string; service: string }> {
    return this.request<{ status: string; service: string }>("/");
  }

  // Conversation endpoints
  async getConversations(): Promise<Conversation[]> {
    return this.request<Conversation[]>("/conversations");
  }

  async getConversation(chatId: string): Promise<Conversation> {
    return this.request<Conversation>(
      `/conversations/${encodeURIComponent(chatId)}`
    );
  }

  async getMessages(
    chatId: string,
    limit: number = 50,
    before?: string
  ): Promise<Message[]> {
    let url = `/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`;
    if (before) {
      url += `&before=${encodeURIComponent(before)}`;
    }
    return this.request<Message[]>(url);
  }

  // Topics endpoints
  async getTopics(
    chatId: string,
    limit: number = 50,
    refresh: boolean = false
  ): Promise<TopicsResponse> {
    let url = `/conversations/${encodeURIComponent(chatId)}/topics?limit=${limit}`;
    if (refresh) {
      url += "&refresh=true";
    }
    return this.request<TopicsResponse>(url, { method: "POST" });
  }

  // Search endpoint
  async searchMessages(
    query: string,
    filters: SearchFilters = {},
    limit: number = 50,
    signal?: AbortSignal
  ): Promise<Message[]> {
    const params = new URLSearchParams();
    params.set("q", query);
    params.set("limit", limit.toString());

    if (filters.sender) {
      params.set("sender", filters.sender);
    }
    if (filters.after) {
      params.set("after", filters.after);
    }
    if (filters.before) {
      params.set("before", filters.before);
    }
    if (filters.has_attachments !== undefined) {
      params.set("has_attachments", filters.has_attachments.toString());
    }

    const url = `/conversations/search?${params.toString()}`;
    const response = await fetch(`${this.baseUrl}${url}`, {
      headers: { "Content-Type": "application/json" },
      signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: "Request failed",
        detail: response.statusText,
      }));
      throw new APIError(
        error.error || "Request failed",
        response.status,
        error.detail || null
      );
    }

    return response.json();
  }

  // Settings endpoints
  async getSettings(): Promise<SettingsResponse> {
    return this.request<SettingsResponse>("/settings");
  }

  async updateSettings(
    settings: SettingsUpdateRequest
  ): Promise<SettingsResponse> {
    return this.request<SettingsResponse>("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  async getModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>("/settings/models");
  }

  async downloadModel(modelId: string): Promise<DownloadStatus> {
    return this.request<DownloadStatus>(
      `/settings/models/${encodeURIComponent(modelId)}/download`,
      {
        method: "POST",
      }
    );
  }

  async activateModel(modelId: string): Promise<ActivateResponse> {
    return this.request<ActivateResponse>(
      `/settings/models/${encodeURIComponent(modelId)}/activate`,
      {
        method: "POST",
      }
    );
  }

  // Draft reply endpoints
  async getDraftReplies(
    chatId: string,
    instruction?: string,
    numSuggestions: number = 3,
    signal?: AbortSignal
  ): Promise<DraftReplyResponse> {
    const body = {
      chat_id: chatId,
      instruction: instruction || null,
      num_suggestions: numSuggestions,
      context_messages: 20,
    };

    const response = await fetch(`${this.baseUrl}/drafts/reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: "Request failed",
        detail: response.statusText,
      }));
      throw new APIError(
        error.error || "Request failed",
        response.status,
        error.detail || null
      );
    }

    return response.json();
  }

  // Smart reply suggestions endpoint
  async getSmartReplySuggestions(
    lastMessage: string,
    numSuggestions: number = 3
  ): Promise<SmartReplySuggestionsResponse> {
    return this.request<SmartReplySuggestionsResponse>("/suggestions", {
      method: "POST",
      body: JSON.stringify({
        last_message: lastMessage,
        num_suggestions: numSuggestions,
      }),
    });
  }

  // Summary endpoints
  async getSummary(
    chatId: string,
    numMessages: number = 50,
    signal?: AbortSignal
  ): Promise<SummaryResponse> {
    const body = {
      chat_id: chatId,
      num_messages: numMessages,
    };

    const response = await fetch(`${this.baseUrl}/drafts/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: "Request failed",
        detail: response.statusText,
      }));
      throw new APIError(
        error.error || "Request failed",
        response.status,
        error.detail || null
      );
    }

    const data = await response.json();
    // Add message_count for UI display
    return { ...data, message_count: numMessages };
  }
}

// Export singleton instance
export const api = new ApiClient();

// Alias for alternate naming
export const apiClient = api;
