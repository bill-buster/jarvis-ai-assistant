/**
 * TypeScript types for JARVIS v2 API
 */

// Health
export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  imessage_accessible: boolean;
}

// Conversations
export interface Conversation {
  chat_id: string;
  display_name: string | null;
  participants: string[];
  last_message_date: string | null;
  last_message_text: string | null;
  last_message_is_from_me: boolean;
  message_count: number;
  is_group: boolean;
}

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
}

// Messages
export interface Message {
  id: number;
  text: string;
  sender: string;
  sender_name: string | null;
  is_from_me: boolean;
  timestamp: string | null;
  chat_id: string;
}

export interface MessageListResponse {
  messages: Message[];
  chat_id: string;
  total: number;
}

// Reply Generation
export interface GeneratedReply {
  text: string;
  reply_type: string;
  confidence: number;
}

export interface PastReply {
  their_message: string;
  your_reply: string;
  similarity: number;
}

export interface GenerationDebugInfo {
  style_instructions: string;
  intent_detected: string;
  past_replies_found: PastReply[];
  full_prompt: string;
}

export interface GenerateRepliesResponse {
  replies: GeneratedReply[];
  chat_id: string;
  model_used: string;
  generation_time_ms: number;
  context_summary: string;
  debug: GenerationDebugInfo | null;
}

// Send Message
export interface SendMessageResponse {
  success: boolean;
  error: string | null;
}

// Settings
export interface Settings {
  model_id: string;
  auto_suggest: boolean;
  max_replies: number;
}

// Contact Profile
export interface TopicCluster {
  name: string;
  keywords: string[];
  message_count: number;
  percentage: number;
}

export interface ContactProfile {
  chat_id: string;
  display_name: string | null;
  relationship_type: string;
  relationship_confidence: number;
  total_messages: number;
  you_sent: number;
  they_sent: number;
  avg_your_length: number;
  avg_their_length: number;
  tone: string;
  uses_emoji: boolean;
  uses_slang: boolean;
  is_playful: boolean;
  topics: TopicCluster[];
  most_active_hours: number[];
  first_message_date: string | null;
  last_message_date: string | null;
  their_common_phrases: string[];
  your_common_phrases: string[];
  summary: string;
}
