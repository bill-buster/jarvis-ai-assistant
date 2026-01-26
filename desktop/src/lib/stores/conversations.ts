/**
 * Conversations store for managing chat state
 */

import { writable, derived } from "svelte/store";
import type { Conversation, Message } from "../api/types";
import { api } from "../api/client";

export interface ConversationsState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  loading: boolean;
  loadingMessages: boolean;
  error: string | null;
}

const initialState: ConversationsState = {
  conversations: [],
  selectedChatId: null,
  messages: [],
  loading: false,
  loadingMessages: false,
  error: null,
};

export const conversationsStore = writable<ConversationsState>(initialState);

// Derived store for selected conversation
export const selectedConversation = derived(
  conversationsStore,
  ($state) =>
    $state.conversations.find((c) => c.chat_id === $state.selectedChatId) || null
);

export async function fetchConversations(): Promise<void> {
  conversationsStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const conversations = await api.getConversations();
    conversationsStore.update((state) => ({
      ...state,
      conversations,
      loading: false,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch conversations";
    conversationsStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function selectConversation(chatId: string): Promise<void> {
  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: chatId,
    messages: [],
    loadingMessages: true,
    error: null,
  }));

  try {
    const messages = await api.getMessages(chatId);
    // Reverse messages so oldest is at top, newest at bottom (API returns newest first)
    const chronologicalMessages = [...messages].reverse();
    conversationsStore.update((state) => ({
      ...state,
      messages: chronologicalMessages,
      loadingMessages: false,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch messages";
    conversationsStore.update((state) => ({
      ...state,
      loadingMessages: false,
      error: message,
    }));
  }
}

export function clearSelection(): void {
  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: null,
    messages: [],
  }));
}
