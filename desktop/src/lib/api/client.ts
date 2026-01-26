/**
 * API client for JARVIS backend
 */

import type {
  ActivateResponse,
  Conversation,
  DownloadStatus,
  HealthResponse,
  Message,
  ModelInfo,
  SettingsResponse,
  SettingsUpdateRequest,
} from "./types";

const API_BASE = "http://localhost:8742";

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
      throw new Error(error.detail || error.error || "Request failed");
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
}

// Export singleton instance
export const api = new ApiClient();
