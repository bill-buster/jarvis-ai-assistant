/**
 * API type definitions for the JARVIS desktop client.
 */

// Attachment metadata
export interface Attachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
}

// Tapback reaction
export interface Reaction {
  type: string;
  sender: string;
  sender_name: string | null;
  date: string;
}

// Message response
export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: number | null;
  reactions: Reaction[];
  date_delivered: string | null;
  date_read: string | null;
  is_system_message: boolean;
}

// Conversation summary
export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: string | null;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: string | null;
}

// Health status
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

// Quick reply suggestion
export interface Suggestion {
  text: string;
  score: number;
}

// Suggestion response
export interface SuggestionResponse {
  suggestions: Suggestion[];
}

// Send message request
export interface SendMessageRequest {
  text: string;
  recipient?: string;
  is_group?: boolean;
}

// Send attachment request
export interface SendAttachmentRequest {
  file_path: string;
  recipient?: string;
  is_group?: boolean;
}

// Send message response
export interface SendMessageResponse {
  success: boolean;
  error: string | null;
}

// Error response
export interface ErrorResponse {
  error: string;
  detail: string;
  code?: string;
}

// AI Draft reply response
export interface DraftReplyResponse {
  suggestions: Array<{
    text: string;
    confidence: number;
  }>;
  context_used: {
    num_messages: number;
    participants: string[];
    last_message: string;
  };
}

// Conversation summary response
export interface SummaryResponse {
  summary: string;
  key_points: string[];
  date_range: {
    start: string;
    end: string;
  };
}
