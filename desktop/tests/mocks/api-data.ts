/**
 * Mock API data for E2E tests.
 *
 * These mock responses match the types defined in src/lib/api/types.ts
 * and replicate the structure returned by the FastAPI backend.
 */

import type {
  Conversation,
  HealthResponse,
  Message,
  ModelInfo,
  SettingsResponse,
} from "../../src/lib/api/types";

/**
 * Mock health response - healthy system
 */
export const mockHealthResponse: HealthResponse = {
  status: "healthy",
  imessage_access: true,
  memory_available_gb: 6.5,
  memory_used_gb: 9.5,
  memory_mode: "FULL",
  model_loaded: true,
  permissions_ok: true,
  details: null,
  jarvis_rss_mb: 512,
  jarvis_vms_mb: 2048,
};

/**
 * Mock health response - degraded system
 */
export const mockHealthResponseDegraded: HealthResponse = {
  status: "degraded",
  imessage_access: true,
  memory_available_gb: 2.0,
  memory_used_gb: 14.0,
  memory_mode: "LITE",
  model_loaded: false,
  permissions_ok: true,
  details: { memory: "Low memory, running in LITE mode" },
  jarvis_rss_mb: 256,
  jarvis_vms_mb: 1024,
};

/**
 * Mock conversations list
 */
export const mockConversations: Conversation[] = [
  {
    chat_id: "chat-1",
    participants: ["+1234567890"],
    display_name: "John Doe",
    last_message_date: new Date().toISOString(),
    message_count: 42,
    is_group: false,
    last_message_text: "Hey, are you free for lunch tomorrow?",
  },
  {
    chat_id: "chat-2",
    participants: ["+1987654321"],
    display_name: "Jane Smith",
    last_message_date: new Date(Date.now() - 3600000).toISOString(),
    message_count: 128,
    is_group: false,
    last_message_text: "Thanks for the help!",
  },
  {
    chat_id: "chat-3",
    participants: ["+1111111111", "+2222222222", "+3333333333"],
    display_name: "Project Team",
    last_message_date: new Date(Date.now() - 86400000).toISOString(),
    message_count: 256,
    is_group: true,
    last_message_text: "Meeting moved to 3pm",
  },
  {
    chat_id: "chat-4",
    participants: ["+4444444444"],
    display_name: null,
    last_message_date: new Date(Date.now() - 172800000).toISOString(),
    message_count: 15,
    is_group: false,
    last_message_text: "See you next week!",
  },
];

/**
 * Mock messages for chat-1
 */
export const mockMessagesChat1: Message[] = [
  {
    id: 1,
    chat_id: "chat-1",
    sender: "+1234567890",
    sender_name: "John Doe",
    text: "Hey, are you free for lunch tomorrow?",
    date: new Date().toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 2,
    chat_id: "chat-1",
    sender: "me",
    sender_name: null,
    text: "Sure! What time works for you?",
    date: new Date(Date.now() - 300000).toISOString(),
    is_from_me: true,
    attachments: [],
    reply_to_id: null,
    reactions: [
      {
        type: "love",
        sender: "+1234567890",
        sender_name: "John Doe",
        date: new Date(Date.now() - 290000).toISOString(),
      },
    ],
    date_delivered: new Date(Date.now() - 299000).toISOString(),
    date_read: new Date(Date.now() - 298000).toISOString(),
    is_system_message: false,
  },
  {
    id: 3,
    chat_id: "chat-1",
    sender: "+1234567890",
    sender_name: "John Doe",
    text: "How about noon at the usual place?",
    date: new Date(Date.now() - 600000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
];

/**
 * Mock messages for chat-3 (group chat)
 */
export const mockMessagesChat3: Message[] = [
  {
    id: 10,
    chat_id: "chat-3",
    sender: "+1111111111",
    sender_name: "Alice",
    text: "Meeting moved to 3pm",
    date: new Date(Date.now() - 86400000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 11,
    chat_id: "chat-3",
    sender: "+2222222222",
    sender_name: "Bob",
    text: "Works for me!",
    date: new Date(Date.now() - 86500000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: 10,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 12,
    chat_id: "chat-3",
    sender: "me",
    sender_name: null,
    text: "I can make it too",
    date: new Date(Date.now() - 86600000).toISOString(),
    is_from_me: true,
    attachments: [],
    reply_to_id: null,
    reactions: [
      {
        type: "like",
        sender: "+1111111111",
        sender_name: "Alice",
        date: new Date(Date.now() - 86590000).toISOString(),
      },
    ],
    date_delivered: new Date(Date.now() - 86599000).toISOString(),
    date_read: new Date(Date.now() - 86598000).toISOString(),
    is_system_message: false,
  },
];

/**
 * Mock available models
 */
export const mockModels: ModelInfo[] = [
  {
    model_id: "qwen2.5-0.5b-instruct-4bit",
    name: "Qwen 2.5 0.5B Instruct (4-bit)",
    size_gb: 0.4,
    quality_tier: "basic",
    ram_requirement_gb: 4,
    is_downloaded: true,
    is_loaded: true,
    is_recommended: true,
    description: "Fast and lightweight model for quick responses",
  },
  {
    model_id: "qwen2.5-1.5b-instruct-4bit",
    name: "Qwen 2.5 1.5B Instruct (4-bit)",
    size_gb: 1.0,
    quality_tier: "good",
    ram_requirement_gb: 6,
    is_downloaded: true,
    is_loaded: false,
    is_recommended: false,
    description: "Balanced model with good quality and speed",
  },
  {
    model_id: "qwen2.5-3b-instruct-4bit",
    name: "Qwen 2.5 3B Instruct (4-bit)",
    size_gb: 2.0,
    quality_tier: "best",
    ram_requirement_gb: 8,
    is_downloaded: false,
    is_loaded: false,
    is_recommended: false,
    description: "Highest quality responses, requires more memory",
  },
];

/**
 * Mock settings response
 */
export const mockSettingsResponse: SettingsResponse = {
  model_id: "qwen2.5-0.5b-instruct-4bit",
  generation: {
    temperature: 0.7,
    max_tokens_reply: 150,
    max_tokens_summary: 500,
  },
  behavior: {
    auto_suggest_replies: true,
    suggestion_count: 3,
    context_messages_reply: 20,
    context_messages_summary: 50,
  },
  system: {
    system_ram_gb: 16,
    current_memory_usage_gb: 9.5,
    model_loaded: true,
    model_memory_usage_gb: 0.5,
    imessage_access: true,
  },
};

/**
 * Mock draft reply response
 */
export const mockDraftReplyResponse = {
  suggestions: [
    { text: "Sounds great! I'll be there.", confidence: 0.92 },
    { text: "Yes, noon works perfectly for me!", confidence: 0.87 },
    { text: "Looking forward to it. See you then!", confidence: 0.81 },
  ],
  context_used: {
    num_messages: 10,
    participants: ["John Doe"],
  },
};

/**
 * Mock API ping response
 */
export const mockPingResponse = {
  status: "ok",
  service: "jarvis-api",
};
