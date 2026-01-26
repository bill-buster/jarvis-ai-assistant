/**
 * Conversations store for managing conversation list and selection.
 */

import { apiClient } from "../api/client";
import type { Conversation, Message } from "../api/types";

// State
let conversations = $state<Conversation[]>([]);
let selectedChatId = $state<string | null>(null);
let messages = $state<Message[]>([]);
let loadingConversations = $state(false);
let loadingMessages = $state(false);
let error = $state<string | null>(null);

/**
 * Fetch conversations from the API.
 */
export async function fetchConversations(options?: {
  limit?: number;
  since?: string;
  before?: string;
}): Promise<Conversation[]> {
  loadingConversations = true;
  error = null;

  try {
    conversations = await apiClient.getConversations(options);
    return conversations;
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to fetch conversations";
    return [];
  } finally {
    loadingConversations = false;
  }
}

/**
 * Select a conversation and load its messages.
 */
export async function selectConversation(chatId: string): Promise<void> {
  selectedChatId = chatId;
  await fetchMessages(chatId);
}

/**
 * Clear the current selection.
 */
export function clearSelection(): void {
  selectedChatId = null;
  messages = [];
}

/**
 * Fetch messages for a conversation.
 */
export async function fetchMessages(
  chatId: string,
  options?: { limit?: number; before?: string }
): Promise<Message[]> {
  loadingMessages = true;
  error = null;

  try {
    messages = await apiClient.getMessages(chatId, options);
    return messages;
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to fetch messages";
    return [];
  } finally {
    loadingMessages = false;
  }
}

/**
 * Refresh messages for the currently selected conversation.
 */
export async function refreshMessages(): Promise<Message[]> {
  if (!selectedChatId) return [];
  return fetchMessages(selectedChatId);
}

/**
 * Get current conversations list.
 */
export function getConversations(): Conversation[] {
  return conversations;
}

/**
 * Get currently selected chat ID.
 */
export function getSelectedChatId(): string | null {
  return selectedChatId;
}

/**
 * Get current messages.
 */
export function getMessages(): Message[] {
  return messages;
}

/**
 * Get the currently selected conversation.
 */
export function getSelectedConversation(): Conversation | null {
  if (!selectedChatId) return null;
  return conversations.find((c) => c.chat_id === selectedChatId) || null;
}

// Export reactive getters for Svelte components
export function getConversationsStore() {
  return {
    get conversations() {
      return conversations;
    },
    get selectedChatId() {
      return selectedChatId;
    },
    get messages() {
      return messages;
    },
    get loadingConversations() {
      return loadingConversations;
    },
    get loadingMessages() {
      return loadingMessages;
    },
    get error() {
      return error;
    },
    get selectedConversation() {
      return conversations.find((c) => c.chat_id === selectedChatId) || null;
    },
  };
}
