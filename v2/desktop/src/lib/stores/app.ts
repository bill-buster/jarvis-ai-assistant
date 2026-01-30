/**
 * JARVIS v2 App Store
 *
 * Centralized state management with WebSocket for real-time updates.
 */

import { writable, derived, get } from "svelte/store";
import type { ContactProfile, Conversation, Message, GeneratedReply, GenerationDebugInfo } from "../api/types";
import { api } from "../api/client";
import { jarvisWs, type ReplyEvent, type GenerationCompleteEvent, type NewMessageEvent } from "../api/websocket";

// Connection status
export type ConnectionStatus = "connected" | "disconnected" | "connecting";

// App state
export interface AppState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  replies: GeneratedReply[];
  generationDebug: GenerationDebugInfo | null;
  generationTimeMs: number;
  contactProfile: ContactProfile | null;
  loading: boolean;
  loadingMessages: boolean;
  loadingReplies: boolean;
  loadingProfile: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  wsConnected: boolean;
  // Track last seen date per conversation for unread indicators
  lastSeenDate: Record<string, string>;
  // Streaming state
  streamingReplies: GeneratedReply[];
  isStreaming: boolean;
}

const initialState: AppState = {
  conversations: [],
  selectedChatId: null,
  messages: [],
  replies: [],
  generationDebug: null,
  generationTimeMs: 0,
  contactProfile: null,
  loading: false,
  loadingMessages: false,
  loadingReplies: false,
  loadingProfile: false,
  error: null,
  connectionStatus: "disconnected",
  wsConnected: false,
  lastSeenDate: {},
  streamingReplies: [],
  isStreaming: false,
};

// Main store
export const appStore = writable<AppState>(initialState);

// Derived stores
export const selectedConversation = derived(appStore, ($state) =>
  $state.conversations.find((c) => c.chat_id === $state.selectedChatId) || null
);

export const connectionStatus = derived(appStore, ($state) => $state.connectionStatus);

// Get unread status for a conversation
export const unreadChats = derived(appStore, ($state) => {
  const unread = new Set<string>();
  for (const conv of $state.conversations) {
    if (conv.last_message_is_from_me) continue;

    const lastSeen = $state.lastSeenDate[conv.chat_id];
    if (!lastSeen && conv.last_message_date) {
      unread.add(conv.chat_id);
    } else if (lastSeen && conv.last_message_date && conv.last_message_date > lastSeen) {
      unread.add(conv.chat_id);
    }
  }
  return unread;
});

// WebSocket setup
let previousChatId: string | null = null;

function setupWebSocket(): void {
  jarvisWs.setHandlers({
    onConnect: () => {
      console.log("WebSocket connected");
      appStore.update((s) => ({ ...s, wsConnected: true }));
      // Re-watch current chat if any
      const state = get(appStore);
      if (state.selectedChatId) {
        jarvisWs.watchMessages(state.selectedChatId);
      }
    },

    onDisconnect: () => {
      console.log("WebSocket disconnected");
      appStore.update((s) => ({ ...s, wsConnected: false }));
    },

    onStateChange: (state) => {
      console.log("WebSocket state:", state);
    },

    onReply: (event: ReplyEvent) => {
      // Streaming reply received
      const reply: GeneratedReply = {
        text: event.text,
        reply_type: event.reply_type,
        confidence: event.confidence,
      };

      appStore.update((s) => ({
        ...s,
        streamingReplies: [...s.streamingReplies, reply],
        isStreaming: true,
      }));
    },

    onGenerationStart: () => {
      appStore.update((s) => ({
        ...s,
        streamingReplies: [],
        replies: [],
        isStreaming: true,
        loadingReplies: true,
      }));
    },

    onGenerationComplete: (event: GenerationCompleteEvent) => {
      appStore.update((s) => ({
        ...s,
        replies: s.streamingReplies,
        streamingReplies: [],
        isStreaming: false,
        loadingReplies: false,
        generationTimeMs: event.generation_time_ms,
        generationDebug: {
          style_instructions: event.style_instructions,
          intent_detected: event.intent_detected,
          past_replies_found: event.past_replies.map((pr) => ({
            their_message: pr.their_message,
            your_reply: pr.your_reply,
            similarity: pr.similarity,
          })),
          full_prompt: event.full_prompt,
        },
      }));
    },

    onGenerationError: (event) => {
      console.error("Generation error:", event.error);
      appStore.update((s) => ({
        ...s,
        isStreaming: false,
        loadingReplies: false,
        error: event.error,
      }));
    },

    onNewMessage: (event: NewMessageEvent) => {
      const state = get(appStore);
      if (event.chat_id === state.selectedChatId) {
        // Add new message to current conversation
        const newMsg: Message = {
          id: event.message.id,
          text: event.message.text,
          sender: event.message.sender,
          sender_name: null,
          is_from_me: event.message.is_from_me,
          timestamp: event.message.timestamp,
          chat_id: event.chat_id,
        };

        appStore.update((s) => ({
          ...s,
          messages: [...s.messages, newMsg],
          lastSeenDate: { ...s.lastSeenDate, [event.chat_id]: event.message.timestamp },
        }));

        // Auto-generate replies if someone else sent the message
        if (!event.message.is_from_me) {
          generateRepliesViaWs();
        }
      }

      // Refresh conversations to update last message
      fetchConversations();
    },

    onError: (error) => {
      console.error("WebSocket error:", error);
      appStore.update((s) => ({ ...s, error }));
    },
  });

  jarvisWs.connect();
}

// Actions
export async function fetchConversations(): Promise<void> {
  const state = get(appStore);
  if (state.loading) return;

  appStore.update((s) => ({ ...s, loading: true, error: null, connectionStatus: "connecting" }));

  try {
    const conversations = await api.getConversations();
    appStore.update((s) => ({
      ...s,
      conversations,
      loading: false,
      connectionStatus: "connected",
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch conversations";
    appStore.update((s) => ({
      ...s,
      loading: false,
      error: message,
      connectionStatus: "disconnected",
    }));
  }
}

export async function selectConversation(chatId: string): Promise<void> {
  // Unwatch previous chat
  if (previousChatId && jarvisWs.isConnected) {
    jarvisWs.unwatchMessages(previousChatId);
  }
  previousChatId = chatId;

  // Watch new chat for real-time updates
  if (jarvisWs.isConnected) {
    jarvisWs.watchMessages(chatId);
  }

  appStore.update((s) => ({
    ...s,
    selectedChatId: chatId,
    replies: [],
    streamingReplies: [],
    generationDebug: null,
    generationTimeMs: 0,
    contactProfile: null,
    loadingMessages: true,
    loadingProfile: true,
    loadingReplies: true,
    isStreaming: false,
    error: null,
  }));

  // Fetch messages and profile
  try {
    const [messages, profile] = await Promise.all([
      api.getMessages(chatId),
      api.getContactProfile(chatId).catch(() => null),
    ]);

    const chronological = [...messages].reverse();
    const lastMsgDate = chronological.length > 0 ? chronological[chronological.length - 1].timestamp : null;

    appStore.update((s) => ({
      ...s,
      messages: chronological,
      contactProfile: profile,
      loadingMessages: false,
      loadingProfile: false,
      lastSeenDate: lastMsgDate ? { ...s.lastSeenDate, [chatId]: lastMsgDate } : s.lastSeenDate,
    }));

    // Generate replies via WebSocket for streaming, or fall back to HTTP
    if (jarvisWs.isConnected) {
      generateRepliesViaWs();
    } else {
      generateRepliesViaHttp(chatId);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch messages";
    appStore.update((s) => ({
      ...s,
      loadingMessages: false,
      loadingProfile: false,
      loadingReplies: false,
      error: message,
    }));
  }
}

function generateRepliesViaWs(): void {
  const state = get(appStore);
  if (!state.selectedChatId) return;

  appStore.update((s) => ({
    ...s,
    loadingReplies: true,
    replies: [],
    streamingReplies: [],
    isStreaming: true,
  }));

  jarvisWs.generateReplies(state.selectedChatId);
}

async function generateRepliesViaHttp(chatId: string): Promise<void> {
  try {
    const response = await api.generateReplies(chatId);
    appStore.update((s) => ({
      ...s,
      replies: response.replies,
      generationDebug: response.debug ?? null,
      generationTimeMs: response.generation_time_ms,
      loadingReplies: false,
    }));
  } catch {
    appStore.update((s) => ({ ...s, loadingReplies: false }));
  }
}

export async function generateReplies(): Promise<void> {
  const state = get(appStore);
  if (!state.selectedChatId) return;

  if (jarvisWs.isConnected) {
    generateRepliesViaWs();
  } else {
    appStore.update((s) => ({ ...s, loadingReplies: true, replies: [], generationDebug: null, generationTimeMs: 0 }));
    await generateRepliesViaHttp(state.selectedChatId);
  }
}

export function clearSelection(): void {
  const state = get(appStore);
  if (state.selectedChatId && jarvisWs.isConnected) {
    jarvisWs.unwatchMessages(state.selectedChatId);
  }
  previousChatId = null;

  appStore.update((s) => ({
    ...s,
    selectedChatId: null,
    messages: [],
    replies: [],
    streamingReplies: [],
  }));
}

export function clearError(): void {
  appStore.update((s) => ({ ...s, error: null }));
}

export async function sendMessage(text: string): Promise<boolean> {
  const state = get(appStore);
  if (!state.selectedChatId || !text.trim()) return false;

  const chatId = state.selectedChatId;

  // Get is_group from selected conversation
  const conversation = state.conversations.find(c => c.chat_id === chatId);
  const isGroup = conversation?.is_group ?? false;

  try {
    const result = await api.sendMessage(chatId, text, isGroup);
    if (result.success) {
      // Optimistically add the sent message
      const optimisticMessage: Message = {
        id: Date.now(),
        text: text,
        sender: "Me",
        sender_name: null,
        is_from_me: true,
        timestamp: new Date().toISOString(),
        chat_id: chatId,
      };

      appStore.update((s) => ({
        ...s,
        messages: [...s.messages, optimisticMessage],
        replies: [],
        streamingReplies: [],
        generationDebug: null,
        loadingReplies: true,
      }));

      // Generate new replies
      if (jarvisWs.isConnected) {
        generateRepliesViaWs();
      } else {
        generateRepliesViaHttp(chatId);
      }

      return true;
    } else {
      appStore.update((s) => ({ ...s, error: result.error || "Failed to send message" }));
      return false;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to send message";
    appStore.update((s) => ({ ...s, error: message }));
    return false;
  }
}

// Polling for conversation list (WebSocket handles messages)
let conversationPollInterval: ReturnType<typeof setInterval> | null = null;

export function startPolling(): void {
  stopPolling();

  // Initialize WebSocket
  setupWebSocket();

  // Initial fetch
  fetchConversations();

  // Poll conversation list every 15 seconds (WebSocket handles message updates)
  conversationPollInterval = setInterval(() => {
    fetchConversations();
  }, 15000);

  // Refresh on window focus
  if (typeof window !== "undefined") {
    window.addEventListener("focus", handleWindowFocus);
  }
}

export function stopPolling(): void {
  if (conversationPollInterval) {
    clearInterval(conversationPollInterval);
    conversationPollInterval = null;
  }
  if (typeof window !== "undefined") {
    window.removeEventListener("focus", handleWindowFocus);
  }
  jarvisWs.disconnect();
}

function handleWindowFocus(): void {
  fetchConversations();
}
