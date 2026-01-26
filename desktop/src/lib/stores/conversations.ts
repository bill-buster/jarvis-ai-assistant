/**
 * Conversations store for JARVIS desktop app.
 */

import { writable, derived } from "svelte/store";
import type { Conversation, Message } from "../api/types";
import { api } from "../api/client";

// Conversations list
export const conversations = writable<Conversation[]>([]);
export const conversationsLoading = writable<boolean>(false);
export const conversationsError = writable<string | null>(null);

// Selected conversation
export const selectedChatId = writable<string | null>(null);
export const messages = writable<Message[]>([]);
export const messagesLoading = writable<boolean>(false);
export const messagesError = writable<string | null>(null);

// Derived store for selected conversation details
export const selectedConversation = derived(
  [conversations, selectedChatId],
  ([$conversations, $selectedChatId]) => {
    if (!$selectedChatId) return null;
    return $conversations.find((c) => c.chat_id === $selectedChatId) || null;
  }
);

/**
 * Fetch conversations list
 */
export async function fetchConversations(limit: number = 50): Promise<void> {
  conversationsLoading.set(true);
  conversationsError.set(null);

  try {
    const data = await api.getConversations(limit);
    conversations.set(data);
  } catch (error) {
    conversationsError.set(
      error instanceof Error ? error.message : "Failed to fetch conversations"
    );
  } finally {
    conversationsLoading.set(false);
  }
}

/**
 * Select a conversation and fetch its messages
 */
export async function selectConversation(chatId: string): Promise<void> {
  selectedChatId.set(chatId);
  messagesLoading.set(true);
  messagesError.set(null);

  try {
    const data = await api.getMessages(chatId);
    messages.set(data);
  } catch (error) {
    messagesError.set(
      error instanceof Error ? error.message : "Failed to fetch messages"
    );
  } finally {
    messagesLoading.set(false);
  }
}

/**
 * Clear the current selection
 */
export function clearSelection(): void {
  selectedChatId.set(null);
  messages.set([]);
  messagesError.set(null);
}

/**
 * Refresh messages for current conversation
 */
export async function refreshMessages(): Promise<void> {
  let chatId: string | null = null;
  selectedChatId.subscribe((v) => (chatId = v))();

  if (chatId) {
    await selectConversation(chatId);
  }
}
