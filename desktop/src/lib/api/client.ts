/**
 * API client for communicating with the JARVIS backend.
 */

import type {
  Conversation,
  DraftReplyResponse,
  HealthStatus,
  Message,
  SendMessageRequest,
  SendMessageResponse,
  SuggestionResponse,
  SummaryResponse,
} from "./types";

const API_BASE_URL = "http://localhost:8000";

class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new APIError(
      errorData.error || `HTTP ${response.status}`,
      response.status,
      errorData.detail
    );
  }
  return response.json();
}

/**
 * JARVIS API Client
 */
export class JarvisClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Health endpoints

  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${this.baseUrl}/health`);
    return handleResponse<HealthStatus>(response);
  }

  async ping(): Promise<{ status: string; service: string }> {
    const response = await fetch(`${this.baseUrl}/`);
    return handleResponse<{ status: string; service: string }>(response);
  }

  // Conversation endpoints

  async getConversations(options?: {
    limit?: number;
    since?: string;
    before?: string;
  }): Promise<Conversation[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", options.limit.toString());
    if (options?.since) params.set("since", options.since);
    if (options?.before) params.set("before", options.before);

    const url = `${this.baseUrl}/conversations${params.toString() ? `?${params}` : ""}`;
    const response = await fetch(url);
    return handleResponse<Conversation[]>(response);
  }

  async getMessages(
    chatId: string,
    options?: {
      limit?: number;
      before?: string;
    }
  ): Promise<Message[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", options.limit.toString());
    if (options?.before) params.set("before", options.before);

    const url = `${this.baseUrl}/conversations/${encodeURIComponent(chatId)}/messages${params.toString() ? `?${params}` : ""}`;
    const response = await fetch(url);
    return handleResponse<Message[]>(response);
  }

  async searchMessages(
    query: string,
    options?: {
      limit?: number;
      sender?: string;
      after?: string;
      before?: string;
      chatId?: string;
      hasAttachments?: boolean;
    }
  ): Promise<Message[]> {
    const params = new URLSearchParams();
    params.set("q", query);
    if (options?.limit) params.set("limit", options.limit.toString());
    if (options?.sender) params.set("sender", options.sender);
    if (options?.after) params.set("after", options.after);
    if (options?.before) params.set("before", options.before);
    if (options?.chatId) params.set("chat_id", options.chatId);
    if (options?.hasAttachments !== undefined) {
      params.set("has_attachments", options.hasAttachments.toString());
    }

    const url = `${this.baseUrl}/conversations/search?${params}`;
    const response = await fetch(url);
    return handleResponse<Message[]>(response);
  }

  async sendMessage(
    chatId: string,
    request: SendMessageRequest
  ): Promise<SendMessageResponse> {
    const response = await fetch(
      `${this.baseUrl}/conversations/${encodeURIComponent(chatId)}/send`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }
    );
    return handleResponse<SendMessageResponse>(response);
  }

  async sendAttachment(
    chatId: string,
    filePath: string,
    options?: { recipient?: string; isGroup?: boolean }
  ): Promise<SendMessageResponse> {
    const response = await fetch(
      `${this.baseUrl}/conversations/${encodeURIComponent(chatId)}/send-attachment`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_path: filePath,
          recipient: options?.recipient,
          is_group: options?.isGroup,
        }),
      }
    );
    return handleResponse<SendMessageResponse>(response);
  }

  // Suggestion endpoints

  async getSuggestions(
    lastMessage: string,
    numSuggestions: number = 3
  ): Promise<SuggestionResponse> {
    const response = await fetch(`${this.baseUrl}/suggestions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        last_message: lastMessage,
        num_suggestions: numSuggestions,
      }),
    });
    return handleResponse<SuggestionResponse>(response);
  }

  // AI-powered endpoints

  /**
   * Get AI-generated draft replies for a conversation.
   *
   * @param chatId - The conversation ID
   * @param instruction - Optional instruction for what kind of reply to generate
   * @param numSuggestions - Number of suggestions to generate (default: 3)
   */
  async getDraftReplies(
    chatId: string,
    instruction?: string,
    numSuggestions: number = 3
  ): Promise<DraftReplyResponse> {
    const response = await fetch(
      `${this.baseUrl}/conversations/${encodeURIComponent(chatId)}/draft-replies`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction: instruction || null,
          num_suggestions: numSuggestions,
        }),
      }
    );
    return handleResponse<DraftReplyResponse>(response);
  }

  /**
   * Get an AI-generated summary of a conversation.
   *
   * @param chatId - The conversation ID
   * @param numMessages - Number of recent messages to summarize (default: 50)
   */
  async getSummary(
    chatId: string,
    numMessages: number = 50
  ): Promise<SummaryResponse> {
    const response = await fetch(
      `${this.baseUrl}/conversations/${encodeURIComponent(chatId)}/summary`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_messages: numMessages,
        }),
      }
    );
    return handleResponse<SummaryResponse>(response);
  }
}

// Default client instance
export const apiClient = new JarvisClient();

export { APIError };
