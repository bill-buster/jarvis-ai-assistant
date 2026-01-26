/**
 * Conversations store for managing chat state with real-time polling and pagination
 */

import { writable, derived, get } from "svelte/store";
import type { Conversation, Message, SearchFilters } from "../api/types";
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

export type ConnectionStatus = "connected" | "disconnected" | "connecting";

// Store for message highlighting (when navigating from search results)
export const highlightedMessageId = writable<number | null>(null);

// Store for triggering scroll to a specific message
export const scrollToMessageId = writable<number | null>(null);

export interface ConversationsState {
  conversations: Conversation[];
  selectedChatId: string | null;
  messages: Message[];
  loading: boolean;
  loadingMessages: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  conversationsWithNewMessages: Set<string>;
  lastKnownMessageDates: Map<string, string>;
  isWindowFocused: boolean;
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
  connectionStatus: "disconnected",
  conversationsWithNewMessages: new Set(),
  lastKnownMessageDates: new Map(),
  isWindowFocused: true,
};

export const conversationsStore = writable<ConversationsState>(initialState);

// Derived store for selected conversation
export const selectedConversation = derived(
  conversationsStore,
  ($state) =>
    $state.conversations.find((c) => c.chat_id === $state.selectedChatId) || null
);

// Derived store for connection status
export const connectionStatus = derived(
  conversationsStore,
  ($state) => $state.connectionStatus
);

// Derived store for checking if a conversation has new messages
export const hasNewMessages = derived(
  conversationsStore,
  ($state) => (chatId: string) => $state.conversationsWithNewMessages.has(chatId)
);

// Polling intervals (in milliseconds)
const CONVERSATION_POLL_INTERVAL = 30000; // 30 seconds
const MESSAGE_POLL_INTERVAL = 10000; // 10 seconds

// Polling interval handles
let conversationPollInterval: ReturnType<typeof setInterval> | null = null;
let messagePollInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Update connection status
 */
function setConnectionStatus(status: ConnectionStatus): void {
  conversationsStore.update((state) => ({
    ...state,
    connectionStatus: status,
  }));
}

/**
 * Mark a conversation as having new messages
 */
export function markConversationAsNew(chatId: string): void {
  conversationsStore.update((state) => {
    const newSet = new Set(state.conversationsWithNewMessages);
    newSet.add(chatId);
    return { ...state, conversationsWithNewMessages: newSet };
  });
}

/**
 * Clear new message indicator for a conversation
 */
export function clearNewMessageIndicator(chatId: string): void {
  conversationsStore.update((state) => {
    const newSet = new Set(state.conversationsWithNewMessages);
    newSet.delete(chatId);
    return { ...state, conversationsWithNewMessages: newSet };
  });
}

/**
 * Fetch conversations and detect new messages
 */
export async function fetchConversations(isPolling = false): Promise<void> {
  if (!isPolling) {
    conversationsStore.update((state) => ({ ...state, loading: true, error: null }));
  }

  setConnectionStatus("connecting");

  try {
    const conversations = await api.getConversations();

    conversationsStore.update((state) => {
      const newConversationsWithNewMessages = new Set(state.conversationsWithNewMessages);
      const newLastKnownDates = new Map(state.lastKnownMessageDates);

      // Check for new messages by comparing last_message_date
      for (const conv of conversations) {
        const lastKnown = state.lastKnownMessageDates.get(conv.chat_id);

        if (lastKnown && conv.last_message_date > lastKnown) {
          // New message detected - only mark if not currently viewing this conversation
          if (conv.chat_id !== state.selectedChatId) {
            newConversationsWithNewMessages.add(conv.chat_id);
          }
        }

        // Update last known date
        newLastKnownDates.set(conv.chat_id, conv.last_message_date);
      }

      return {
        ...state,
        conversations,
        loading: false,
        connectionStatus: "connected",
        conversationsWithNewMessages: newConversationsWithNewMessages,
        lastKnownMessageDates: newLastKnownDates,
      };
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch conversations";
    conversationsStore.update((state) => ({
      ...state,
      loading: false,
      error: isPolling ? state.error : message, // Don't overwrite error if polling
      connectionStatus: "disconnected",
    }));
  }
}

/**
 * Fetch messages for a conversation
 * Returns the new messages if any (for polling use)
 */
export async function fetchMessages(chatId: string): Promise<Message[]> {
  try {
    const messages = await api.getMessages(chatId);
    // Reverse messages so oldest is at top, newest at bottom (API returns newest first)
    return [...messages].reverse();
  } catch (error) {
    console.error("Failed to fetch messages:", error);
    return [];
  }
}

/**
 * Poll for new messages in the currently selected conversation
 * Returns newly added messages (if any) for animation purposes
 */
export async function pollMessages(): Promise<Message[]> {
  const state = get(conversationsStore);

  if (!state.selectedChatId || !state.isWindowFocused) {
    return [];
  }

  try {
    const freshMessages = await fetchMessages(state.selectedChatId);
    const currentMessages = state.messages;

    // Find messages that are new (not in current list)
    const currentIds = new Set(currentMessages.map((m) => m.id));
    const newMessages = freshMessages.filter((m) => !currentIds.has(m.id));

    if (newMessages.length > 0) {
      // Update store with fresh messages
      conversationsStore.update((s) => ({
        ...s,
        messages: freshMessages,
      }));

      // Update cache with new messages
      const cached = messageCache.get(state.selectedChatId);
      if (cached) {
        messageCache.set(state.selectedChatId, {
          ...cached,
          messages: freshMessages,
        });
      }
    }

    return newMessages;
  } catch (error) {
    console.error("Error polling messages:", error);
    return [];
  }
}

/**
 * Select a conversation and load its messages
 */
export async function selectConversation(chatId: string): Promise<void> {
  // Clear new message indicator when selecting
  clearNewMessageIndicator(chatId);

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
    // Start message polling for this conversation
    startMessagePolling();
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

    // Start message polling for this conversation
    startMessagePolling();
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

/**
 * Clear the current conversation selection
 */
export function clearSelection(): void {
  stopMessagePolling();
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

/**
 * Start polling for conversations
 */
export function startConversationPolling(): void {
  // Clear existing interval if any
  if (conversationPollInterval) {
    clearInterval(conversationPollInterval);
  }

  // Initial fetch
  fetchConversations();

  // Set up polling interval
  conversationPollInterval = setInterval(() => {
    const state = get(conversationsStore);
    if (state.isWindowFocused) {
      fetchConversations(true);
    }
  }, CONVERSATION_POLL_INTERVAL);
}

/**
 * Stop polling for conversations
 */
export function stopConversationPolling(): void {
  if (conversationPollInterval) {
    clearInterval(conversationPollInterval);
    conversationPollInterval = null;
  }
}

/**
 * Start polling for messages in the selected conversation
 */
export function startMessagePolling(): void {
  // Clear existing interval if any
  if (messagePollInterval) {
    clearInterval(messagePollInterval);
  }

  // Set up polling interval
  messagePollInterval = setInterval(() => {
    const state = get(conversationsStore);
    if (state.isWindowFocused && state.selectedChatId) {
      pollMessages();
    }
  }, MESSAGE_POLL_INTERVAL);
}

/**
 * Stop polling for messages
 */
export function stopMessagePolling(): void {
  if (messagePollInterval) {
    clearInterval(messagePollInterval);
    messagePollInterval = null;
  }
}

/**
 * Handle window focus change
 */
export function setWindowFocused(focused: boolean): void {
  conversationsStore.update((state) => ({
    ...state,
    isWindowFocused: focused,
  }));

  if (focused) {
    // Resume polling when window gains focus
    const state = get(conversationsStore);
    fetchConversations(true);
    if (state.selectedChatId) {
      pollMessages();
    }
  }
}

/**
 * Initialize polling and window focus listeners
 * Call this on app mount
 */
export function initializePolling(): () => void {
  // Start conversation polling
  startConversationPolling();

  // Handle window focus/blur
  const handleFocus = () => setWindowFocused(true);
  const handleBlur = () => setWindowFocused(false);

  window.addEventListener("focus", handleFocus);
  window.addEventListener("blur", handleBlur);

  // Cleanup function
  return () => {
    stopConversationPolling();
    stopMessagePolling();
    window.removeEventListener("focus", handleFocus);
    window.removeEventListener("blur", handleBlur);
  };
}

/**
 * Navigate to a specific message from search results
 * Selects the conversation and highlights the target message
 */
export async function navigateToMessage(chatId: string, messageId: number): Promise<void> {
  // First select the conversation
  await selectConversation(chatId);

  // Then trigger scroll and highlight
  scrollToMessageId.set(messageId);
  highlightedMessageId.set(messageId);

  // Clear highlight after 3 seconds
  setTimeout(() => {
    highlightedMessageId.set(null);
  }, 3000);
}

/**
 * Clear the scroll target after scrolling is complete
 */
export function clearScrollTarget(): void {
  scrollToMessageId.set(null);
}
