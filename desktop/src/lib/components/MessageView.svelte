<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte";
  import {
    conversationsStore,
    selectedConversation,
    loadMoreMessages,
  } from "../stores/conversations";
  import AIDraftPanel from "./AIDraftPanel.svelte";
  import SummaryModal from "./SummaryModal.svelte";
  import SmartReplyChips from "./SmartReplyChips.svelte";

  // Panel visibility state
  let showDraftPanel = $state(false);
  let showSummaryModal = $state(false);
  let messageViewFocused = $state(true);

  // Compute the last received message (for smart reply chips)
  // Only show chips when the last message is NOT from the user
  function getLastReceivedMessage(): string {
    const messages = $conversationsStore.messages;
    if (messages.length === 0) return "";

    // Messages are in chronological order (oldest first), so last message is at the end
    const lastMessage = messages[messages.length - 1];

    // Only return if the last message is NOT from the current user
    if (!lastMessage.is_from_me && lastMessage.text) {
      return lastMessage.text;
    }
    return "";
  }

  // Scroll container reference
  let messagesContainer: HTMLDivElement | null = $state(null);

  // Track previous message count and scroll height for position restoration
  let previousMessageCount = $state(0);
  let previousScrollHeight = $state(0);

  // Threshold for triggering load (200px from top)
  const SCROLL_THRESHOLD = 200;

  // Handle scroll event for infinite scroll
  async function handleScroll(event: Event) {
    const container = event.target as HTMLDivElement;
    if (!container) return;

    // Check if user scrolled near the top
    if (
      container.scrollTop < SCROLL_THRESHOLD &&
      $conversationsStore.hasMore &&
      !$conversationsStore.loadingMore &&
      $conversationsStore.messages.length > 0
    ) {
      // Save current scroll position info before loading
      previousScrollHeight = container.scrollHeight;
      previousMessageCount = $conversationsStore.messages.length;

      await loadMoreMessages();
    }
  }

  // Handle explicit load button click
  async function handleLoadEarlier() {
    if (messagesContainer) {
      previousScrollHeight = messagesContainer.scrollHeight;
      previousMessageCount = $conversationsStore.messages.length;
    }
    await loadMoreMessages();
  }

  // Restore scroll position after messages are prepended
  $effect(() => {
    const currentMessageCount = $conversationsStore.messages.length;
    if (
      messagesContainer &&
      currentMessageCount > previousMessageCount &&
      previousMessageCount > 0
    ) {
      // Messages were prepended, restore scroll position
      tick().then(() => {
        if (messagesContainer) {
          const newScrollHeight = messagesContainer.scrollHeight;
          const scrollDelta = newScrollHeight - previousScrollHeight;
          messagesContainer.scrollTop += scrollDelta;
        }
      });
    }
    previousMessageCount = currentMessageCount;
  });

  // Handle keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    // Check for Cmd (Mac) or Ctrl (Windows/Linux)
    const isMod = event.metaKey || event.ctrlKey;

    if (isMod && event.key === "d") {
      event.preventDefault();
      if ($selectedConversation) {
        showDraftPanel = true;
      }
    } else if (isMod && event.key === "s") {
      event.preventDefault();
      if ($selectedConversation) {
        showSummaryModal = true;
      }
    }
  }

  // Handle draft panel selection
  function handleDraftSelect(text: string) {
    // Copy the selected draft to clipboard
    navigator.clipboard.writeText(text).catch(() => {
      console.error("Failed to copy draft to clipboard");
    });
    showDraftPanel = false;
  }

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener("keydown", handleKeydown);
  });

  function formatTime(dateStr: string): string {
    return new Date(dateStr).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
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
      return date.toLocaleDateString([], {
        weekday: "long",
        month: "long",
        day: "numeric",
      });
    }
  }

  function shouldShowDateHeader(
    messages: typeof $conversationsStore.messages,
    index: number
  ): boolean {
    if (index === 0) return true;
    const curr = new Date(messages[index].date).toDateString();
    const prev = new Date(messages[index - 1].date).toDateString();
    return curr !== prev;
  }
</script>

<div
  class="message-view"
  tabindex="-1"
  onfocus={() => messageViewFocused = true}
  onblur={() => messageViewFocused = false}
>
  {#if !$selectedConversation}
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path
          d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
        />
      </svg>
      <h3>Select a conversation</h3>
      <p>Choose a conversation from the list to view messages</p>
    </div>
  {:else}
    <div class="header">
      <div class="avatar" class:group={$selectedConversation.is_group}>
        {#if $selectedConversation.is_group}
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6z"/>
          </svg>
        {:else}
          {($selectedConversation.display_name || $selectedConversation.participants[0] || "?").charAt(0).toUpperCase()}
        {/if}
      </div>
      <div class="info">
        <h2>{$selectedConversation.display_name || $selectedConversation.participants.join(", ")}</h2>
        <p>{$selectedConversation.message_count} messages</p>
      </div>
      <div class="header-actions">
        <button
          class="action-btn"
          onclick={() => showSummaryModal = true}
          title="Summarize conversation (Cmd+S)"
          aria-label="Summarize conversation"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="21" y1="10" x2="3" y2="10"></line>
            <line x1="21" y1="6" x2="3" y2="6"></line>
            <line x1="21" y1="14" x2="3" y2="14"></line>
            <line x1="21" y1="18" x2="3" y2="18"></line>
          </svg>
        </button>
        <button
          class="action-btn primary"
          onclick={() => showDraftPanel = true}
          title="Generate AI reply (Cmd+D)"
          aria-label="Generate AI reply"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 20h9"></path>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
          </svg>
          <span>AI Draft</span>
        </button>
      </div>
    </div>

    <div
      class="messages"
      bind:this={messagesContainer}
      onscroll={handleScroll}
    >
      {#if $conversationsStore.loadingMessages}
        <div class="loading">Loading messages...</div>
      {:else if $conversationsStore.messages.length === 0}
        <div class="empty">No messages in this conversation</div>
      {:else}
        <!-- Load earlier messages section -->
        <div class="load-earlier-section">
          {#if $conversationsStore.loadingMore}
            <div class="loading-more">
              <div class="spinner"></div>
              <span>Loading earlier messages...</span>
            </div>
          {:else if $conversationsStore.hasMore}
            <button
              class="load-earlier-btn"
              onclick={handleLoadEarlier}
              aria-label="Load earlier messages"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="17 11 12 6 7 11"></polyline>
                <line x1="12" y1="6" x2="12" y2="18"></line>
              </svg>
              Load earlier messages
            </button>
          {:else}
            <div class="end-of-history">
              <span>Beginning of conversation</span>
            </div>
          {/if}
        </div>

        {#each $conversationsStore.messages as message, index (message.id)}
          {#if shouldShowDateHeader($conversationsStore.messages, index)}
            <div class="date-header">
              <span>{formatDate(message.date)}</span>
            </div>
          {/if}

          {#if message.is_system_message}
            <div class="system-message">
              {message.text}
            </div>
          {:else}
            <div class="message" class:from-me={message.is_from_me}>
              <div class="bubble" class:from-me={message.is_from_me}>
                {#if !message.is_from_me && $selectedConversation.is_group}
                  <span class="sender">{message.sender_name || message.sender}</span>
                {/if}
                <p>{message.text}</p>
                {#if message.attachments.length > 0}
                  <div class="attachments">
                    {#each message.attachments as attachment}
                      <div class="attachment">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                        </svg>
                        <span>{attachment.filename}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                <span class="time">{formatTime(message.date)}</span>
              </div>
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
          {/if}
        {/each}
      {/if}
    </div>

    <!-- Smart Reply Chips -->
    {#if getLastReceivedMessage()}
      <SmartReplyChips
        lastMessage={getLastReceivedMessage()}
        isFocused={messageViewFocused}
      />
    {/if}
  {/if}
</div>

<!-- AI Draft Panel -->
{#if showDraftPanel && $selectedConversation}
  <AIDraftPanel
    chatId={$selectedConversation.chat_id}
    onSelect={handleDraftSelect}
    onClose={() => showDraftPanel = false}
  />
{/if}

<!-- Summary Modal -->
{#if showSummaryModal && $selectedConversation}
  <SummaryModal
    chatId={$selectedConversation.chat_id}
    onClose={() => showSummaryModal = false}
  />
{/if}

<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    gap: 12px;
  }

  .empty-state svg {
    width: 64px;
    height: 64px;
    opacity: 0.5;
  }

  .empty-state h3 {
    font-size: 18px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .empty-state p {
    font-size: 14px;
  }

  .header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .header .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: white;
  }

  .header .avatar.group {
    background: var(--group-color);
  }

  .header .avatar svg {
    width: 20px;
    height: 20px;
  }

  .header .info h2 {
    font-size: 16px;
    font-weight: 600;
  }

  .header .info p {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .header-actions {
    display: flex;
    gap: 8px;
    margin-left: auto;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    font-size: 13px;
  }

  .action-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--accent-color);
  }

  .action-btn svg {
    width: 16px;
    height: 16px;
  }

  .action-btn.primary {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .action-btn.primary:hover {
    background: #0a82e0;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .date-header {
    text-align: center;
    margin: 16px 0 8px;
  }

  .date-header span {
    background: var(--bg-secondary);
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .system-message {
    text-align: center;
    font-size: 13px;
    color: var(--text-secondary);
    font-style: italic;
    padding: 8px 0;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
  }

  .message.from-me {
    align-self: flex-end;
  }

  .bubble {
    padding: 10px 14px;
    border-radius: 18px;
    background: var(--bg-bubble-other);
  }

  .bubble.from-me {
    background: var(--bg-bubble-me);
  }

  .bubble .sender {
    font-size: 12px;
    font-weight: 500;
    color: var(--accent-color);
    display: block;
    margin-bottom: 4px;
  }

  .bubble p {
    font-size: 15px;
    line-height: 1.4;
    word-wrap: break-word;
  }

  .bubble .time {
    font-size: 11px;
    color: var(--text-secondary);
    opacity: 0.7;
    display: block;
    text-align: right;
    margin-top: 4px;
  }

  .from-me .bubble .time {
    color: rgba(255, 255, 255, 0.7);
  }

  .attachments {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .attachment {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .attachment svg {
    width: 14px;
    height: 14px;
  }

  .reactions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .reaction {
    font-size: 14px;
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 10px;
    cursor: default;
  }

  .loading,
  .empty {
    text-align: center;
    color: var(--text-secondary);
    padding: 24px;
  }

  /* Load earlier messages section */
  .load-earlier-section {
    display: flex;
    justify-content: center;
    padding: 16px 0;
    min-height: 48px;
  }

  .loading-more {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 13px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .load-earlier-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .load-earlier-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--accent-color);
  }

  .load-earlier-btn svg {
    width: 14px;
    height: 14px;
  }

  .end-of-history {
    text-align: center;
    font-size: 12px;
    color: var(--text-secondary);
    opacity: 0.7;
  }

  .end-of-history span {
    background: var(--bg-secondary);
    padding: 4px 12px;
    border-radius: 12px;
  }
</style>
