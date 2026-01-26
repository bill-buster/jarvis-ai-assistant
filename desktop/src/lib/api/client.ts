/**
 * API client for JARVIS backend.
 * Communicates with the FastAPI server at localhost:8742
 */

import type {
  Conversation,
  Message,
  HealthStatus,
  SummaryRequest,
  SummaryResponse,
  SendMessageRequest,
  SendMessageResponse,
  ErrorResponse,
} from "./types";

const API_BASE = "http://localhost:8742";
const DEFAULT_TIMEOUT = 30000; // 30 seconds

class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public detail?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = DEFAULT_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail: string | undefined;
    try {
      const errorData: ErrorResponse = await response.json();
      errorDetail = errorData.detail || errorData.error;
    } catch {
      errorDetail = response.statusText;
    }
    throw new ApiError(
      `API request failed: ${response.status}`,
      response.status,
      errorDetail
    );
  }
  return response.json();
}

export const api = {
  /**
   * Get list of conversations
   */
  async getConversations(limit: number = 50): Promise<Conversation[]> {
    const response = await fetchWithTimeout(
      `${API_BASE}/conversations?limit=${limit}`
    );
    return handleResponse<Conversation[]>(response);
  },

  /**
   * Get messages for a specific conversation
   */
  async getMessages(chatId: string, limit: number = 100): Promise<Message[]> {
    const response = await fetchWithTimeout(
      `${API_BASE}/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`
    );
    return handleResponse<Message[]>(response);
  },

  /**
   * Get system health status
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await fetchWithTimeout(`${API_BASE}/health`);
    return handleResponse<HealthStatus>(response);
  },

  /**
   * Generate a summary for a conversation
   */
  async getSummary(
    chatId: string,
    messageCount: number = 50
  ): Promise<SummaryResponse> {
    const request: SummaryRequest = {
      chat_id: chatId,
      message_count: messageCount,
    };

    const response = await fetchWithTimeout(
      `${API_BASE}/drafts/summarize`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      },
      DEFAULT_TIMEOUT
    );

    return handleResponse<SummaryResponse>(response);
  },

  /**
   * Send a message to a conversation
   */
  async sendMessage(
    chatId: string,
    request: SendMessageRequest
  ): Promise<SendMessageResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE}/conversations/${encodeURIComponent(chatId)}/send`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      }
    );
    return handleResponse<SendMessageResponse>(response);
  },

  /**
   * Check if API is reachable
   */
  async ping(): Promise<boolean> {
    try {
      const response = await fetchWithTimeout(`${API_BASE}/health`, {}, 5000);
      return response.ok;
    } catch {
      return false;
    }
  },
};

export { ApiError };
export default api;
