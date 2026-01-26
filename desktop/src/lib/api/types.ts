/**
 * TypeScript types for JARVIS API responses.
 * These mirror the Pydantic schemas in api/schemas.py
 */

export interface Attachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
}

export interface Reaction {
  type: string;
  sender: string;
  sender_name: string | null;
  date: string; // ISO datetime
}

export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string; // ISO datetime
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: number | null;
  reactions: Reaction[];
  date_delivered: string | null;
  date_read: string | null;
  is_system_message: boolean;
}

export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: string | null;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: string | null;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  imessage_access: boolean;
  memory_available_gb: number;
  memory_used_gb: number;
  memory_mode: "FULL" | "LITE" | "MINIMAL";
  model_loaded: boolean;
  permissions_ok: boolean;
  details: Record<string, string> | null;
  jarvis_rss_mb: number;
  jarvis_vms_mb: number;
}

export interface ErrorResponse {
  error: string;
  detail: string;
  code: string | null;
}

export interface SendMessageRequest {
  text: string;
  recipient?: string | null;
  is_group?: boolean;
}

export interface SendMessageResponse {
  success: boolean;
  error: string | null;
}

// Summary-related types
export interface SummaryRequest {
  chat_id: string;
  message_count?: number; // Number of recent messages to summarize
}

export interface SummaryResponse {
  summary: string;
  key_points: string[];
  message_count: number;
  date_range: {
    start: string; // ISO datetime
    end: string; // ISO datetime
  };
  generation_time_ms?: number;
}
