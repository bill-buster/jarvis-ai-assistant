/**
 * API client for JARVIS backend
 */

import type {
  ActivateResponse,
  AttachmentStats,
  AttachmentType,
  AttachmentWithContext,
  Calendar,
  CalendarEvent,
  Conversation,
  ConversationInsights,
  ConversationStats,
  CreateEventResponse,
  CreateExperimentRequest,
  CustomTemplate,
  CustomTemplateCreateRequest,
  CustomTemplateExportResponse,
  CustomTemplateImportRequest,
  CustomTemplateImportResponse,
  CustomTemplateListResponse,
  CustomTemplateTestRequest,
  CustomTemplateTestResponse,
  CustomTemplateUpdateRequest,
  CustomTemplateUsageStats,
  DetectedEvent,
  DigestExportRequest,
  DigestExportResponse,
  DigestGenerateRequest,
  DigestPeriod,
  DigestPreferences,
  DigestPreferencesUpdateRequest,
  DigestResponse,
  DownloadStatus,
  DraftReplyResponse,
  Experiment,
  ExperimentListResponse,
  ExperimentResults,
  FrequencyTrends,
  FeedbackAction,
  FeedbackStatsResponse,
  HealthResponse,
  ImprovementsResponse,
  ImportantContactResponse,
  MarkHandledResponse,
  Message,
  ModelInfo,
  PDFExportRequest,
  PDFExportResponse,
  PriorityInboxResponse,
  PriorityLevel,
  PriorityStats,
  QualityDashboardData,
  QualitySummary,
  RecordFeedbackResponse,
  RecordOutcomeRequest,
  RecordOutcomeResponse,
  RelationshipHealth,
  ResponsePatterns,
  SearchFilters,
  SemanticCacheStats,
  SemanticSearchFilters,
  SemanticSearchResponse,
  SentimentResult,
  SettingsResponse,
  SettingsUpdateRequest,
  SmartReplySuggestionsResponse,
  StorageSummary,
  SummaryResponse,
  TemplateAnalyticsDashboard,
  TemplateAnalyticsSummary,
  ThreadedViewResponse,
  ThreadingConfigRequest,
  ThreadResponse,
  TimeRange,
  TopicsResponse,
  UserAction,
  VariantConfig,
} from "./types";

const API_BASE = "http://localhost:8000";

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
    const response = await this.request<{ conversations: Conversation[]; total: number }>(
      "/conversations"
    );
    return response.conversations;
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
    const response = await this.request<{ messages: Message[]; chat_id: string; total: number }>(
      url
    );
    return response.messages;
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

  // Semantic search endpoint
  async semanticSearch(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
      indexLimit?: number;
      filters?: SemanticSearchFilters;
    } = {},
    signal?: AbortSignal
  ): Promise<SemanticSearchResponse> {
    const body = {
      query,
      limit: options.limit ?? 20,
      threshold: options.threshold ?? 0.3,
      index_limit: options.indexLimit ?? 1000,
      filters: options.filters || null,
    };

    const response = await fetch(`${this.baseUrl}/search/semantic`, {
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

  // Semantic search cache stats
  async getSemanticCacheStats(): Promise<SemanticCacheStats> {
    return this.request<SemanticCacheStats>("/search/semantic/cache/stats");
  }

  // Clear semantic search cache
  async clearSemanticCache(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      "/search/semantic/cache",
      { method: "DELETE" }
    );
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

  // Insights endpoints
  async getConversationInsights(
    chatId: string,
    timeRange: TimeRange = "month",
    limit: number = 500
  ): Promise<ConversationInsights> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<ConversationInsights>(
      `/insights/${encodeURIComponent(chatId)}?${params.toString()}`
    );
  }

  async getSentimentAnalysis(
    chatId: string,
    timeRange: TimeRange = "month",
    limit: number = 200
  ): Promise<SentimentResult> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<SentimentResult>(
      `/insights/${encodeURIComponent(chatId)}/sentiment?${params.toString()}`
    );
  }

  async getResponsePatterns(
    chatId: string,
    timeRange: TimeRange = "month",
    limit: number = 500
  ): Promise<ResponsePatterns> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<ResponsePatterns>(
      `/insights/${encodeURIComponent(chatId)}/response-patterns?${params.toString()}`
    );
  }

  async getFrequencyTrends(
    chatId: string,
    timeRange: TimeRange = "three_months",
    limit: number = 1000
  ): Promise<FrequencyTrends> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<FrequencyTrends>(
      `/insights/${encodeURIComponent(chatId)}/frequency?${params.toString()}`
    );
  }

  async getRelationshipHealth(
    chatId: string,
    timeRange: TimeRange = "month",
    limit: number = 500
  ): Promise<RelationshipHealth> {
    const params = new URLSearchParams({
      time_range: timeRange,
      limit: limit.toString(),
    });
    return this.request<RelationshipHealth>(
      `/insights/${encodeURIComponent(chatId)}/health?${params.toString()}`
    );
  }

  async invalidateInsightsCache(chatId: string): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      `/insights/${encodeURIComponent(chatId)}/cache`,
      { method: "DELETE" }
    );
  }

  // Priority Inbox endpoints
  async getPriorityInbox(
    limit: number = 50,
    includeHandled: boolean = false,
    minLevel?: PriorityLevel,
    signal?: AbortSignal
  ): Promise<PriorityInboxResponse> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      include_handled: includeHandled.toString(),
    });
    if (minLevel) {
      params.set("min_level", minLevel);
    }

    const response = await fetch(
      `${this.baseUrl}/priority?${params.toString()}`,
      {
        headers: { "Content-Type": "application/json" },
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

  async markMessageHandled(
    chatId: string,
    messageId: number
  ): Promise<MarkHandledResponse> {
    return this.request<MarkHandledResponse>("/priority/handled", {
      method: "POST",
      body: JSON.stringify({
        chat_id: chatId,
        message_id: messageId,
      }),
    });
  }

  async unmarkMessageHandled(
    chatId: string,
    messageId: number
  ): Promise<MarkHandledResponse> {
    return this.request<MarkHandledResponse>("/priority/handled", {
      method: "DELETE",
      body: JSON.stringify({
        chat_id: chatId,
        message_id: messageId,
      }),
    });
  }

  async markContactImportant(
    identifier: string,
    important: boolean = true
  ): Promise<ImportantContactResponse> {
    return this.request<ImportantContactResponse>(
      "/priority/contacts/important",
      {
        method: "POST",
        body: JSON.stringify({
          identifier,
          important,
        }),
      }
    );
  }

  async getPriorityStats(): Promise<PriorityStats> {
    return this.request<PriorityStats>("/priority/stats");
  }

  async clearHandledItems(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(
      "/priority/clear-handled",
      { method: "POST" }
    );
  }

  // Calendar endpoints
  async getCalendars(): Promise<Calendar[]> {
    return this.request<Calendar[]>("/calendars");
  }

  async getCalendarEvents(
    calendarId?: string,
    days: number = 30,
    limit: number = 50
  ): Promise<CalendarEvent[]> {
    const params = new URLSearchParams({
      days: days.toString(),
      limit: limit.toString(),
    });
    if (calendarId) {
      params.set("calendar_id", calendarId);
    }
    return this.request<CalendarEvent[]>(`/calendars/events?${params.toString()}`);
  }

  async searchCalendarEvents(
    query: string,
    calendarId?: string,
    limit: number = 50
  ): Promise<CalendarEvent[]> {
    const params = new URLSearchParams({
      query,
      limit: limit.toString(),
    });
    if (calendarId) {
      params.set("calendar_id", calendarId);
    }
    return this.request<CalendarEvent[]>(`/calendars/events/search?${params.toString()}`);
  }

  async detectEventsInText(
    text: string,
    messageId?: number
  ): Promise<DetectedEvent[]> {
    return this.request<DetectedEvent[]>("/calendars/detect", {
      method: "POST",
      body: JSON.stringify({
        text,
        message_id: messageId || null,
      }),
    });
  }

  async detectEventsInMessages(
    chatId: string,
    limit: number = 50
  ): Promise<DetectedEvent[]> {
    return this.request<DetectedEvent[]>("/calendars/detect/messages", {
      method: "POST",
      body: JSON.stringify({
        chat_id: chatId,
        limit,
      }),
    });
  }

  async createCalendarEvent(
    calendarId: string,
    title: string,
    start: string,
    end: string,
    options: {
      all_day?: boolean;
      location?: string;
      notes?: string;
      url?: string;
    } = {}
  ): Promise<CreateEventResponse> {
    return this.request<CreateEventResponse>("/calendars/events", {
      method: "POST",
      body: JSON.stringify({
        calendar_id: calendarId,
        title,
        start,
        end,
        ...options,
      }),
    });
  }

  async createEventFromDetected(
    calendarId: string,
    detectedEvent: DetectedEvent
  ): Promise<CreateEventResponse> {
    return this.request<CreateEventResponse>("/calendars/events/from-detected", {
      method: "POST",
      body: JSON.stringify({
        calendar_id: calendarId,
        detected_event: detectedEvent,
      }),
    });
  }

  // Custom Template endpoints
  async getCustomTemplates(
    category?: string,
    tag?: string,
    enabledOnly: boolean = false
  ): Promise<CustomTemplateListResponse> {
    const params = new URLSearchParams();
    if (category) params.set("category", category);
    if (tag) params.set("tag", tag);
    if (enabledOnly) params.set("enabled_only", "true");
    const queryString = params.toString();
    const url = queryString ? `/templates?${queryString}` : "/templates";
    return this.request<CustomTemplateListResponse>(url);
  }

  async getCustomTemplate(templateId: string): Promise<CustomTemplate> {
    return this.request<CustomTemplate>(
      `/templates/${encodeURIComponent(templateId)}`
    );
  }

  async createCustomTemplate(
    template: CustomTemplateCreateRequest
  ): Promise<CustomTemplate> {
    return this.request<CustomTemplate>("/templates", {
      method: "POST",
      body: JSON.stringify(template),
    });
  }

  async updateCustomTemplate(
    templateId: string,
    updates: CustomTemplateUpdateRequest
  ): Promise<CustomTemplate> {
    return this.request<CustomTemplate>(
      `/templates/${encodeURIComponent(templateId)}`,
      {
        method: "PUT",
        body: JSON.stringify(updates),
      }
    );
  }

  async deleteCustomTemplate(
    templateId: string
  ): Promise<{ status: string; template_id: string }> {
    return this.request<{ status: string; template_id: string }>(
      `/templates/${encodeURIComponent(templateId)}`,
      { method: "DELETE" }
    );
  }

  async testCustomTemplate(
    templateId: string,
    testData: CustomTemplateTestRequest
  ): Promise<CustomTemplateTestResponse> {
    return this.request<CustomTemplateTestResponse>(
      `/templates/${encodeURIComponent(templateId)}/test`,
      {
        method: "POST",
        body: JSON.stringify(testData),
      }
    );
  }

  async getCustomTemplateStats(
    templateId: string
  ): Promise<CustomTemplateUsageStats> {
    return this.request<CustomTemplateUsageStats>(
      `/templates/${encodeURIComponent(templateId)}/stats`
    );
  }

  async exportCustomTemplates(
    category?: string
  ): Promise<CustomTemplateExportResponse> {
    const url = category
      ? `/templates/export?category=${encodeURIComponent(category)}`
      : "/templates/export";
    return this.request<CustomTemplateExportResponse>(url);
  }

  async importCustomTemplates(
    data: CustomTemplateImportRequest
  ): Promise<CustomTemplateImportResponse> {
    return this.request<CustomTemplateImportResponse>("/templates/import", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Threading endpoints
  async getThreadedView(
    chatId: string,
    limit: number = 200,
    before?: string,
    timeGapMinutes: number = 30,
    useSemantic: boolean = true,
    refresh: boolean = false
  ): Promise<ThreadedViewResponse> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      time_gap_minutes: timeGapMinutes.toString(),
      use_semantic: useSemantic.toString(),
      refresh: refresh.toString(),
    });
    if (before) {
      params.set("before", before);
    }
    return this.request<ThreadedViewResponse>(
      `/conversations/${encodeURIComponent(chatId)}/threads?${params.toString()}`
    );
  }

  async getThreadMessages(
    chatId: string,
    threadId: string,
    limit: number = 200
  ): Promise<Message[]> {
    return this.request<Message[]>(
      `/conversations/${encodeURIComponent(chatId)}/threads/${encodeURIComponent(threadId)}?limit=${limit}`
    );
  }

  async analyzeThreads(
    chatId: string,
    config?: ThreadingConfigRequest,
    limit: number = 200
  ): Promise<ThreadResponse[]> {
    const params = new URLSearchParams({ limit: limit.toString() });
    return this.request<ThreadResponse[]>(
      `/conversations/${encodeURIComponent(chatId)}/threads/analyze?${params.toString()}`,
      {
        method: "POST",
        body: config ? JSON.stringify(config) : undefined,
      }
    );
  }

  // Attachment Manager endpoints

  /**
   * List attachments with optional filtering
   */
  async getAttachments(
    options: {
      chatId?: string;
      attachmentType?: AttachmentType;
      after?: string;
      before?: string;
      limit?: number;
    } = {}
  ): Promise<AttachmentWithContext[]> {
    const params = new URLSearchParams();
    if (options.chatId) {
      params.set("chat_id", options.chatId);
    }
    if (options.attachmentType && options.attachmentType !== "all") {
      params.set("attachment_type", options.attachmentType);
    }
    if (options.after) {
      params.set("after", options.after);
    }
    if (options.before) {
      params.set("before", options.before);
    }
    if (options.limit) {
      params.set("limit", options.limit.toString());
    }

    const queryString = params.toString();
    const url = `/attachments${queryString ? `?${queryString}` : ""}`;
    return this.request<AttachmentWithContext[]>(url);
  }

  /**
   * Get attachment statistics for a conversation
   */
  async getAttachmentStats(chatId: string): Promise<AttachmentStats> {
    return this.request<AttachmentStats>(
      `/attachments/stats/${encodeURIComponent(chatId)}`
    );
  }

  /**
   * Get storage breakdown by conversation
   */
  async getStorageSummary(limit: number = 50): Promise<StorageSummary> {
    return this.request<StorageSummary>(`/attachments/storage?limit=${limit}`);
  }

  /**
   * Get thumbnail URL for an attachment
   */
  getThumbnailUrl(filePath: string): string {
    return `${this.baseUrl}/attachments/thumbnail?file_path=${encodeURIComponent(filePath)}`;
  }

  /**
   * Get download URL for an attachment
   */
  getAttachmentUrl(filePath: string): string {
    return `${this.baseUrl}/attachments/file?file_path=${encodeURIComponent(filePath)}`;
  }

  // Digest endpoints
  async generateDigest(
    options: DigestGenerateRequest = {},
    signal?: AbortSignal
  ): Promise<DigestResponse> {
    const body = {
      period: options.period ?? "daily",
      end_date: options.end_date || null,
    };

    const response = await fetch(`${this.baseUrl}/digest/generate`, {
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

  async getDailyDigest(signal?: AbortSignal): Promise<DigestResponse> {
    return this.request<DigestResponse>("/digest/daily", { signal });
  }

  async getWeeklyDigest(signal?: AbortSignal): Promise<DigestResponse> {
    return this.request<DigestResponse>("/digest/weekly", { signal });
  }

  async exportDigest(
    options: DigestExportRequest = {},
    signal?: AbortSignal
  ): Promise<DigestExportResponse> {
    const body = {
      period: options.period ?? "daily",
      format: options.format ?? "markdown",
      end_date: options.end_date || null,
    };

    const response = await fetch(`${this.baseUrl}/digest/export`, {
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

  async getDigestPreferences(): Promise<DigestPreferences> {
    return this.request<DigestPreferences>("/digest/preferences");
  }

  async updateDigestPreferences(
    settings: DigestPreferencesUpdateRequest
  ): Promise<DigestPreferences> {
    return this.request<DigestPreferences>("/digest/preferences", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  // Feedback endpoints
  async recordFeedback(
    action: FeedbackAction,
    suggestionText: string,
    chatId: string,
    contextMessages: string[],
    editedText?: string | null,
    includeEvaluation: boolean = true,
    metadata?: Record<string, unknown>
  ): Promise<RecordFeedbackResponse> {
    return this.request<RecordFeedbackResponse>("/feedback/response", {
      method: "POST",
      body: JSON.stringify({
        action,
        suggestion_text: suggestionText,
        chat_id: chatId,
        context_messages: contextMessages,
        edited_text: editedText,
        include_evaluation: includeEvaluation,
        metadata,
      }),
    });
  }

  async getFeedbackStats(): Promise<FeedbackStatsResponse> {
    return this.request<FeedbackStatsResponse>("/feedback/stats");
  }

  async getFeedbackImprovements(limit: number = 10): Promise<ImprovementsResponse> {
    return this.request<ImprovementsResponse>(
      `/feedback/improvements?limit=${limit}`
    );
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

  // Quality Metrics endpoints
  async getQualitySummary(): Promise<QualitySummary> {
    return this.request<QualitySummary>("/quality/summary");
  }

  async getQualityDashboard(
    trendDays: number = 7,
    topContactsLimit: number = 10
  ): Promise<QualityDashboardData> {
    const params = new URLSearchParams({
      trend_days: trendDays.toString(),
      top_contacts_limit: topContactsLimit.toString(),
    });
    return this.request<QualityDashboardData>(
      `/quality/dashboard?${params.toString()}`
    );
  }

  async resetQualityMetrics(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      "/quality/reset",
      { method: "POST" }
    );
  }
}

// Export singleton instance
export const api = new ApiClient();

// Alias for alternate naming
export const apiClient = api;
