/**
 * API client for JARVIS backend
 */

import type {
  ActivateResponse,
  Conversation,
  ConversationStats,
  CreateExperimentRequest,
  DownloadStatus,
  DraftReplyResponse,
  Experiment,
  ExperimentListResponse,
  ExperimentResults,
  HealthResponse,
  Message,
  ModelInfo,
  PDFExportRequest,
  PDFExportResponse,
  RecordOutcomeRequest,
  RecordOutcomeResponse,
  SearchFilters,
  SettingsResponse,
  SettingsUpdateRequest,
  SmartReplySuggestionsResponse,
  SummaryResponse,
  TemplateAnalyticsDashboard,
  TemplateAnalyticsSummary,
  TimeRange,
  TopicsResponse,
  UserAction,
  VariantConfig,
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

  // Statistics endpoints
  async getConversationStats(
    chatId: string,
    timeRange: TimeRange = "month",
    limit: number = 500
  ): Promise<ConversationStats> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<ConversationStats>(
      `/stats/${encodeURIComponent(chatId)}?${params.toString()}`
    );
  }

  async invalidateStatsCache(chatId: string): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      `/stats/${encodeURIComponent(chatId)}/cache`,
      { method: "DELETE" }
    );
  }

  // Template Analytics endpoints
  async getTemplateAnalytics(): Promise<TemplateAnalyticsSummary> {
    return this.request<TemplateAnalyticsSummary>("/metrics/templates");
  }

  async getTemplateAnalyticsDashboard(): Promise<TemplateAnalyticsDashboard> {
    return this.request<TemplateAnalyticsDashboard>("/metrics/templates/dashboard");
  }

  async resetTemplateAnalytics(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      "/metrics/templates/reset",
      { method: "POST" }
    );
  }

  async exportTemplateAnalytics(): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/metrics/templates/export`);
    if (!response.ok) {
      throw new APIError("Export failed", response.status, null);
    }
    return response.blob();
  }

  // PDF Export endpoints
  async exportPDF(
    chatId: string,
    options: PDFExportRequest = {},
    signal?: AbortSignal
  ): Promise<PDFExportResponse> {
    const body = {
      include_attachments: options.include_attachments ?? true,
      include_reactions: options.include_reactions ?? true,
      date_range: options.date_range || null,
      limit: options.limit ?? 1000,
    };

    const response = await fetch(
      `${this.baseUrl}/export/pdf/${encodeURIComponent(chatId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      }
    );

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

  /**
   * Downloads a PDF export directly as a blob
   */
  async downloadPDF(
    chatId: string,
    options: PDFExportRequest = {},
    signal?: AbortSignal
  ): Promise<{ blob: Blob; filename: string }> {
    const body = {
      include_attachments: options.include_attachments ?? true,
      include_reactions: options.include_reactions ?? true,
      date_range: options.date_range || null,
      limit: options.limit ?? 1000,
    };

    const response = await fetch(
      `${this.baseUrl}/export/pdf/${encodeURIComponent(chatId)}/download`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      }
    );

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

    const contentDisposition = response.headers.get("Content-Disposition");
    let filename = "conversation.pdf";
    if (contentDisposition) {
      const match = contentDisposition.match(/filename="?([^"]+)"?/);
      if (match) {
        filename = match[1];
      }
    }

    const blob = await response.blob();
    return { blob, filename };
  }

  // Experiment endpoints
  async getExperiments(activeOnly: boolean = false): Promise<ExperimentListResponse> {
    const params = activeOnly ? "?active_only=true" : "";
    return this.request<ExperimentListResponse>(`/experiments${params}`);
  }

  async getExperiment(name: string): Promise<Experiment> {
    return this.request<Experiment>(
      `/experiments/${encodeURIComponent(name)}`
    );
  }

  async getExperimentResults(name: string): Promise<ExperimentResults> {
    return this.request<ExperimentResults>(
      `/experiments/${encodeURIComponent(name)}/results`
    );
  }

  async recordExperimentOutcome(
    experimentName: string,
    request: RecordOutcomeRequest
  ): Promise<RecordOutcomeResponse> {
    return this.request<RecordOutcomeResponse>(
      `/experiments/${encodeURIComponent(experimentName)}/record`,
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    );
  }

  async createExperiment(request: CreateExperimentRequest): Promise<Experiment> {
    return this.request<Experiment>("/experiments", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async updateExperiment(
    name: string,
    enabled: boolean
  ): Promise<Experiment> {
    return this.request<Experiment>(
      `/experiments/${encodeURIComponent(name)}`,
      {
        method: "PUT",
        body: JSON.stringify({ enabled }),
      }
    );
  }

  async deleteExperiment(name: string): Promise<{ status: string; experiment_name: string }> {
    return this.request<{ status: string; experiment_name: string }>(
      `/experiments/${encodeURIComponent(name)}`,
      { method: "DELETE" }
    );
  }

  async clearExperimentOutcomes(name: string): Promise<{ status: string; experiment_name: string }> {
    return this.request<{ status: string; experiment_name: string }>(
      `/experiments/${encodeURIComponent(name)}/outcomes`,
      { method: "DELETE" }
    );
  }

  async getVariantForContact(
    experimentName: string,
    contactId: string
  ): Promise<VariantConfig | null> {
    const params = new URLSearchParams({ contact_id: contactId });
    return this.request<VariantConfig | null>(
      `/experiments/${encodeURIComponent(experimentName)}/variant?${params.toString()}`
    );
  }
}

// Export singleton instance
export const api = new ApiClient();

// Alias for alternate naming
export const apiClient = api;
