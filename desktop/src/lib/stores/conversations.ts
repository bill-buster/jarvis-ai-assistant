/**
 * Conversations store for managing chat state
 */

import { writable, derived, get } from "svelte/store";
import type { Conversation, Message } from "../api/types";
import { api } from "../api/client";

/** Default number of messages to fetch per page */
const PAGE_SIZE = 50;

/** Pagination state for a conversation */
export interface PaginationState {
  hasMore: boolean;
  loadingMore: boolean;
}

/** Cache entry for a conversation's messages */
interface MessageCacheEntry {
  messages: Message[];
  pagination: PaginationState;
}

export interface ConversationsState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  loading: boolean;
  loadingMessages: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
}

/** Message cache keyed by chat_id to avoid re-fetching */
const messageCache = new Map<string, MessageCacheEntry>();

const initialState: ConversationsState = {
  conversations: [],
  selectedChatId: null,
  messages: [],
  loading: false,
  loadingMessages: false,
  loadingMore: false,
  hasMore: true,
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
  // Check if we have cached messages for this conversation
  const cached = messageCache.get(chatId);
  if (cached) {
    conversationsStore.update((state) => ({
      ...state,
      selectedChatId: chatId,
      messages: cached.messages,
      hasMore: cached.pagination.hasMore,
      loadingMore: false,
      loadingMessages: false,
      error: null,
    }));
    return;
  }

  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: chatId,
    messages: [],
    loadingMessages: true,
    loadingMore: false,
    hasMore: true,
    error: null,
  }));

  try {
    const messages = await api.getMessages(chatId, PAGE_SIZE);
    // Reverse messages so oldest is at top, newest at bottom (API returns newest first)
    const chronologicalMessages = [...messages].reverse();
    // If we got fewer messages than requested, we've reached the end
    const hasMore = messages.length >= PAGE_SIZE;

    // Cache the messages
    messageCache.set(chatId, {
      messages: chronologicalMessages,
      pagination: { hasMore, loadingMore: false },
    });

    conversationsStore.update((state) => ({
      ...state,
      messages: chronologicalMessages,
      loadingMessages: false,
      hasMore,
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

/**
 * Load more (older) messages for the currently selected conversation.
 * Uses the 'before' parameter to fetch messages older than the oldest loaded message.
 * Returns true if more messages were loaded, false if at the end of history.
 */
export async function loadMoreMessages(): Promise<boolean> {
  const { selectedChatId, messages, loadingMore, hasMore } = get(conversationsStore);

  // Guard: don't fetch if no conversation selected, already loading, or no more messages
  if (!selectedChatId || loadingMore || !hasMore || messages.length === 0) {
    return false;
  }

  // Get the oldest message's date (first in array since messages are chronological)
  const oldestMessage = messages[0];
  const beforeDate = oldestMessage.date;

  conversationsStore.update((state) => ({
    ...state,
    loadingMore: true,
  }));

  try {
    const olderMessages = await api.getMessages(selectedChatId, PAGE_SIZE, beforeDate);

    // If we got fewer messages than requested, we've reached the end
    const newHasMore = olderMessages.length >= PAGE_SIZE;

    // Reverse to get chronological order and prepend to existing messages
    const chronologicalOlder = [...olderMessages].reverse();

    conversationsStore.update((state) => {
      const newMessages = [...chronologicalOlder, ...state.messages];

      // Update cache
      messageCache.set(selectedChatId, {
        messages: newMessages,
        pagination: { hasMore: newHasMore, loadingMore: false },
      });

      return {
        ...state,
        messages: newMessages,
        loadingMore: false,
        hasMore: newHasMore,
      };
    });

    return olderMessages.length > 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load more messages";
    conversationsStore.update((state) => ({
      ...state,
      loadingMore: false,
      error: message,
    }));
    return false;
  }
}

export function clearSelection(): void {
  conversationsStore.update((state) => ({
    ...state,
    selectedChatId: null,
    messages: [],
    hasMore: true,
    loadingMore: false,
  }));
}

/**
 * Invalidate the message cache for a specific conversation or all conversations.
 * Useful when messages might have changed (e.g., new message received).
 */
export function invalidateMessageCache(chatId?: string): void {
  if (chatId) {
    messageCache.delete(chatId);
  } else {
    messageCache.clear();
  }
}
