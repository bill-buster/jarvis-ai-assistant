<script lang="ts">
  import {
    selectedConversation,
    messages,
    messagesLoading,
    messagesError,
    refreshMessages,
    selectedChatId,
  } from "../stores/conversations";
  import SummaryModal from "./SummaryModal.svelte";
  import type { Message } from "../api/types";

  let showSummary = false;

  function formatTime(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return "Today";
    } else if (date.toDateString() === yesterday.toDateString()) {
      return "Yesterday";
    } else {
      return date.toLocaleDateString("en-US", {
        weekday: "long",
        month: "short",
        day: "numeric",
      });
    }
  }

  function shouldShowDateSeparator(
    messages: Message[],
    index: number
  ): boolean {
    if (index === 0) return true;
    const current = new Date(messages[index].date).toDateString();
    const previous = new Date(messages[index - 1].date).toDateString();
    return current !== previous;
  }

  function getDisplayName(): string {
    if ($selectedConversation?.display_name) {
      return $selectedConversation.display_name;
    }
    if ($selectedConversation?.participants.length === 1) {
      return $selectedConversation.participants[0];
    }
    return "Conversation";
  }

  function getParticipantCount(): string {
    const count = $selectedConversation?.participants.length || 0;
    if (count === 1) return "1 participant";
    return `${count} participants`;
  }

  function openSummary() {
    showSummary = true;
  }

  function closeSummary() {
    showSummary = false;
  }
</script>

<div class="message-view">
  {#if $selectedChatId}
    <!-- Header -->
    <div class="header">
      <div class="header-info">
        <h2 class="conversation-name">{getDisplayName()}</h2>
        <span class="participant-count">{getParticipantCount()}</span>
      </div>
      <div class="header-actions">
        <button
          class="action-btn summary-btn"
          on:click={openSummary}
          title="Summarize conversation"
          disabled={$messagesLoading || $messages.length < 5}
        >
          <span class="btn-icon">📋</span>
          <span class="btn-text">Summary</span>
        </button>
        <button
          class="action-btn refresh-btn"
          on:click={refreshMessages}
          title="Refresh messages"
          disabled={$messagesLoading}
        >
          <span class="btn-icon" class:spinning={$messagesLoading}>↻</span>
        </button>
      </div>
    </div>

    <!-- Messages -->
    <div class="messages-container">
      {#if $messagesLoading && $messages.length === 0}
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p>Loading messages...</p>
        </div>
      {:else if $messagesError}
        <div class="error-state">
          <p>{$messagesError}</p>
          <button on:click={refreshMessages}>Try Again</button>
        </div>
      {:else if $messages.length === 0}
        <div class="empty-state">
          <p>No messages in this conversation</p>
        </div>
      {:else}
        <div class="messages-list">
          {#each $messages as message, index (message.id)}
            {#if shouldShowDateSeparator($messages, index)}
              <div class="date-separator">
                <span>{formatDate(message.date)}</span>
              </div>
            {/if}

            <div
              class="message"
              class:from-me={message.is_from_me}
              class:system-message={message.is_system_message}
            >
              {#if message.is_system_message}
                <div class="system-message-content">
                  {message.text}
                </div>
              {:else}
                {#if !message.is_from_me && $selectedConversation?.is_group}
                  <div class="sender-name">
                    {message.sender_name || message.sender}
                  </div>
                {/if}

                <div class="bubble">
                  <p class="text">{message.text}</p>

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

                <div class="message-time">
                  {formatTime(message.date)}
                  {#if message.is_from_me}
                    {#if message.date_read}
                      <span class="status read">Read</span>
                    {:else if message.date_delivered}
                      <span class="status delivered">Delivered</span>
                    {/if}
                  {/if}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {:else}
    <!-- No conversation selected -->
    <div class="no-selection">
      <div class="no-selection-content">
        <span class="no-selection-icon">💬</span>
        <p>Select a conversation to view messages</p>
      </div>
    </div>
  {/if}

  <!-- Summary Modal -->
  {#if showSummary && $selectedChatId}
    <SummaryModal chatId={$selectedChatId} onClose={closeSummary} />
  {/if}
</div>

<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    min-width: 0;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .header-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .conversation-name {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .participant-count {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .header-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-active);
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-icon {
    font-size: 14px;
  }

  .refresh-btn {
    padding: 6px 10px;
  }

  .refresh-btn .btn-icon {
    font-size: 16px;
    display: inline-block;
  }

  .refresh-btn .btn-icon.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: var(--text-secondary);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .error-state button {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
  }

  .messages-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .date-separator {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px 0;
  }

  .date-separator span {
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 4px 12px;
    border-radius: 10px;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 75%;
    margin-bottom: 4px;
  }

  .message.from-me {
    align-self: flex-end;
    align-items: flex-end;
  }

  .message:not(.from-me) {
    align-self: flex-start;
    align-items: flex-start;
  }

  .message.system-message {
    align-self: center;
    max-width: 90%;
  }

  .system-message-content {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
    text-align: center;
    padding: 8px 0;
  }

  .sender-name {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 2px;
    margin-left: 12px;
  }

  .bubble {
    padding: 8px 12px;
    border-radius: 18px;
    word-wrap: break-word;
  }

  .message.from-me .bubble {
    background: var(--bg-bubble-me);
    color: white;
    border-bottom-right-radius: 4px;
  }

  .message:not(.from-me) .bubble {
    background: var(--bg-bubble-other);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
  }

  .text {
    margin: 0;
    font-size: 15px;
    line-height: 1.4;
    white-space: pre-wrap;
  }

  .attachments {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attachment {
    font-size: 13px;
    opacity: 0.9;
  }

  .reactions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .reaction {
    font-size: 14px;
    background: var(--bg-hover);
    padding: 2px 6px;
    border-radius: 10px;
    cursor: default;
  }

  .message-time {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
    margin-left: 4px;
    margin-right: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .status {
    font-size: 10px;
    text-transform: uppercase;
  }

  .status.delivered {
    color: var(--text-secondary);
  }

  .status.read {
    color: var(--accent-color);
  }

  .no-selection {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .no-selection-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: var(--text-secondary);
  }

  .no-selection-icon {
    font-size: 48px;
    opacity: 0.5;
  }
</style>
