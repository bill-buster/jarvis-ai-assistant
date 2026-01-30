/**
 * JARVIS v2 API Client
 * Simplified client for the v2 backend
 */

const V2_API_BASE = "http://localhost:8000";

export interface V2HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  imessage_accessible: boolean;
}

export interface V2Conversation {
  chat_id: string;
  display_name: string | null;
  participants: string[];
  last_message_date: string | null;
  last_message_text: string | null;
  message_count: number;
  is_group: boolean;
}

export interface V2Message {
  id: number;
  text: string;
  sender: string;
  is_from_me: boolean;
  timestamp: string | null;
  chat_id: string;
}

export interface V2GeneratedReply {
  text: string;
  reply_type: string;
  confidence: number;
}

export interface V2GenerateRepliesResponse {
  replies: V2GeneratedReply[];
  chat_id: string;
  model_used: string;
  generation_time_ms: number;
  context_summary: string;
}

export interface V2ModelInfo {
  id: string;
  display_name: string;
  size_gb: number;
  quality: string;
  description: string;
}

class V2ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = V2_API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Health
  async getHealth(): Promise<V2HealthResponse> {
    return this.fetch<V2HealthResponse>("/health");
  }

  // Conversations
  async getConversations(limit: number = 50): Promise<{ conversations: V2Conversation[]; total: number }> {
    return this.fetch(`/conversations?limit=${limit}`);
  }

  async getMessages(chatId: string, limit: number = 50): Promise<{ messages: V2Message[]; chat_id: string; total: number }> {
    return this.fetch(`/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`);
  }

  // Generation
  async generateReplies(chatId: string, numReplies: number = 3): Promise<V2GenerateRepliesResponse> {
    return this.fetch<V2GenerateRepliesResponse>("/generate/replies", {
      method: "POST",
      body: JSON.stringify({ chat_id: chatId, num_replies: numReplies }),
    });
  }

  // Settings
  async getSettings(): Promise<{ model_id: string; auto_suggest: boolean; max_replies: number }> {
    return this.fetch("/settings");
  }

  async updateSettings(settings: { model_id?: string; auto_suggest?: boolean; max_replies?: number }): Promise<{ model_id: string; auto_suggest: boolean; max_replies: number }> {
    return this.fetch("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  async getModels(): Promise<{ models: V2ModelInfo[] }> {
    return this.fetch("/settings/models");
  }
}

// Export singleton instance
export const v2Api = new V2ApiClient();
