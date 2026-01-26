<script lang="ts">
  import { onMount } from "svelte";
  import { apiClient } from "../api/client";
  import type { Message, Suggestion } from "../api/types";
  import {
    getConversationsStore,
    refreshMessages,
  } from "../stores/conversations";
  import AIDraftPanel from "./AIDraftPanel.svelte";

  const store = getConversationsStore();

  // Local state
  let messageInput = $state("");
  let suggestions = $state<Suggestion[]>([]);
  let loadingSuggestions = $state(false);
  let showAIDraftPanel = $state(false);
  let messagesContainer: HTMLDivElement | undefined = $state();

  // Scroll to bottom when messages change
  $effect(() => {
    if (store.messages && messagesContainer) {
      // Wait for DOM update
      setTimeout(() => {
        if (messagesContainer) {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }, 0);
    }
  });

  // Load suggestions when last message changes
  $effect(() => {
    const lastReceivedMessage = getLastReceivedMessage();
    if (lastReceivedMessage) {
      loadSuggestions(lastReceivedMessage.text);
    } else {
      suggestions = [];
    }
  });

  function getLastReceivedMessage(): Message | null {
    // Messages are newest first, find the most recent non-self message
    const received = store.messages.find((m) => !m.is_from_me && !m.is_system_message);
    return received || null;
  }

  async function loadSuggestions(text: string) {
    if (!text.trim()) {
      suggestions = [];
      return;
    }

    loadingSuggestions = true;
    try {
      const response = await apiClient.getSuggestions(text, 3);
      // Filter out low-confidence suggestions
      suggestions = response.suggestions.filter((s) => s.score > 0.3);
    } catch {
      suggestions = [];
    } finally {
      loadingSuggestions = false;
    }
  }

  function useSuggestion(text: string) {
    messageInput = text;
  }

  async function sendMessage() {
    if (!messageInput.trim() || !store.selectedConversation) return;

    const conv = store.selectedConversation;
    const text = messageInput.trim();
    messageInput = "";

    try {
      // Determine recipient for individual chats
      const recipient = conv.is_group ? undefined : conv.participants[0];

      await apiClient.sendMessage(conv.chat_id, {
        text,
        recipient,
        is_group: conv.is_group,
      });

      // Refresh messages to show the sent message
      await refreshMessages();
    } catch (e) {
      console.error("Failed to send message:", e);
      // Restore the message on error
      messageInput = text;
    }
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function formatTime(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return "Today";
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: "long" });
    } else {
      return date.toLocaleDateString([], {
        month: "short",
        day: "numeric",
        year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
      });
    }
  }

  function shouldShowDateSeparator(index: number): boolean {
    const messages = store.messages;
    if (index === messages.length - 1) return true; // Always show for oldest (last in reversed list)

    const current = new Date(messages[index].date);
    const next = new Date(messages[index + 1].date);

    return (
      current.getDate() !== next.getDate() ||
      current.getMonth() !== next.getMonth() ||
      current.getFullYear() !== next.getFullYear()
    );
  }

  function handleAIDraftSelect(text: string) {
    messageInput = text;
    showAIDraftPanel = false;
  }
</script>

<main class="message-view">
  {#if !store.selectedChatId}
    <div class="empty-state">
      <div class="empty-icon">💬</div>
      <h2>Select a conversation</h2>
      <p>Choose a conversation from the list to view messages</p>
    </div>
  {:else}
    <header class="chat-header">
      <div class="chat-info">
        <div class="avatar" class:group={store.selectedConversation?.is_group}>
          {#if store.selectedConversation?.is_group}
            <span>👥</span>
          {:else}
            <span>
              {(store.selectedConversation?.display_name ||
                store.selectedConversation?.participants[0] ||
                "?")
                .charAt(0)
                .toUpperCase()}
            </span>
          {/if}
        </div>
        <div class="chat-details">
          <h2>
            {store.selectedConversation?.display_name ||
              store.selectedConversation?.participants.join(", ") ||
              "Unknown"}
          </h2>
          {#if store.selectedConversation?.is_group}
            <span class="participants">
              {store.selectedConversation.participants.length} participants
            </span>
          {/if}
        </div>
      </div>
      <button class="refresh-btn" onclick={() => refreshMessages()}>
        ↻
      </button>
    </header>

    <div class="messages-container" bind:this={messagesContainer}>
      {#if store.loadingMessages}
        <div class="loading">Loading messages...</div>
      {:else if store.messages.length === 0}
        <div class="no-messages">No messages yet</div>
      {:else}
        <div class="messages-list">
          {#each [...store.messages].reverse() as message, index}
            {#if shouldShowDateSeparator(store.messages.length - 1 - index)}
              <div class="date-separator">
                <span>{formatDate(message.date)}</span>
              </div>
            {/if}

            <div
              class="message"
              class:from-me={message.is_from_me}
              class:system={message.is_system_message}
            >
              {#if message.is_system_message}
                <div class="system-message">
                  {message.text}
                </div>
              {:else}
                {#if !message.is_from_me && store.selectedConversation?.is_group}
                  <span class="sender-name">
                    {message.sender_name || message.sender}
                  </span>
                {/if}

                <div class="bubble">
                  <span class="text">{message.text}</span>

                  {#if message.attachments.length > 0}
                    <div class="attachments">
                      {#each message.attachments as attachment}
                        <div class="attachment">
                          📎 {attachment.filename}
                        </div>
                      {/each}
                    </div>
                  {/if}

                  {#if message.reactions.length > 0}
                    <div class="reactions">
                      {#each message.reactions as reaction}
                        <span class="reaction" title={reaction.sender_name || reaction.sender}>
                          {reaction.type}
                        </span>
                      {/each}
                    </div>
                  {/if}
                </div>

                <span class="time">{formatTime(message.date)}</span>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <div class="input-area">
      <!-- Suggestions bar with AI button -->
      <div class="suggestions-bar">
        {#if loadingSuggestions}
          <span class="loading-suggestions">Loading suggestions...</span>
        {:else if suggestions.length > 0}
          {#each suggestions as suggestion}
            <button
              class="suggestion-chip"
              onclick={() => useSuggestion(suggestion.text)}
            >
              {suggestion.text}
            </button>
          {/each}
        {/if}

        <button
          class="ai-draft-btn"
          onclick={() => (showAIDraftPanel = true)}
          title="Draft with AI"
        >
          <span class="ai-icon">✨</span>
          <span class="ai-label">Draft with AI</span>
        </button>
      </div>

      <div class="input-row">
        <input
          type="text"
          bind:value={messageInput}
          onkeydown={handleKeyDown}
          placeholder="Type a message..."
        />
        <button
          class="send-btn"
          onclick={sendMessage}
          disabled={!messageInput.trim()}
        >
          ↑
        </button>
      </div>
    </div>
  {/if}

  {#if showAIDraftPanel && store.selectedChatId}
    <AIDraftPanel
      chatId={store.selectedChatId}
      onSelect={handleAIDraftSelect}
      onClose={() => (showAIDraftPanel = false)}
    />
  {/if}
</main>

<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    min-width: 0;
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    text-align: center;
    padding: 40px;
  }

  .empty-icon {
    font-size: 64px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state h2 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .empty-state p {
    font-size: 14px;
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .chat-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 600;
    color: white;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .chat-details h2 {
    font-size: 16px;
    font-weight: 600;
  }

  .participants {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .refresh-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 8px;
    border-radius: 6px;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .loading,
  .no-messages {
    text-align: center;
    color: var(--text-secondary);
    padding: 40px;
  }

  .messages-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .date-separator {
    display: flex;
    justify-content: center;
    margin: 16px 0;
  }

  .date-separator span {
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 12px;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
    margin: 2px 0;
  }

  .message.from-me {
    align-self: flex-end;
    align-items: flex-end;
  }

  .message.system {
    align-self: center;
    max-width: 90%;
  }

  .system-message {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-radius: 12px;
  }

  .sender-name {
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 2px;
    margin-left: 12px;
  }

  .bubble {
    padding: 10px 14px;
    border-radius: 18px;
    background: var(--bg-bubble-other);
    position: relative;
  }

  .message.from-me .bubble {
    background: var(--bg-bubble-me);
  }

  .text {
    font-size: 15px;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .attachments {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attachment {
    font-size: 13px;
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.2);
    padding: 6px 10px;
    border-radius: 8px;
  }

  .message.from-me .attachment {
    background: rgba(255, 255, 255, 0.2);
  }

  .reactions {
    position: absolute;
    bottom: -8px;
    right: -4px;
    display: flex;
    gap: 2px;
  }

  .reaction {
    font-size: 14px;
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 2px 4px;
  }

  .time {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
    margin-left: 12px;
  }

  .message.from-me .time {
    margin-right: 12px;
    margin-left: 0;
  }

  .input-area {
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
    padding: 12px 16px;
  }

  .suggestions-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
    align-items: center;
  }

  .loading-suggestions {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .suggestion-chip {
    padding: 6px 14px;
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .suggestion-chip:hover {
    background: var(--bg-active);
    border-color: var(--accent-color);
  }

  .ai-draft-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 16px;
    color: white;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    margin-left: auto;
  }

  .ai-draft-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }

  .ai-icon {
    font-size: 14px;
  }

  .input-row {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .input-row input {
    flex: 1;
    padding: 12px 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    color: var(--text-primary);
    font-size: 15px;
    outline: none;
  }

  .input-row input:focus {
    border-color: var(--accent-color);
  }

  .input-row input::placeholder {
    color: var(--text-secondary);
  }

  .send-btn {
    width: 36px;
    height: 36px;
    background: var(--accent-color);
    border: none;
    border-radius: 50%;
    color: white;
    font-size: 18px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
  }

  .send-btn:hover:not(:disabled) {
    background: #0a84e0;
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
