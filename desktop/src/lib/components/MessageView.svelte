<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte";
  import {
    conversationsStore,
    selectedConversation,
    loadMoreMessages,
    pollMessages,
    stopMessagePolling,
    highlightedMessageId,
    scrollToMessageId,
    clearScrollTarget,
  } from "../stores/conversations";
  import AIDraftPanel from "./AIDraftPanel.svelte";
  import SummaryModal from "./SummaryModal.svelte";
  import SmartReplyChips from "./SmartReplyChips.svelte";
  import ConversationStats from "./ConversationStats.svelte";
  import PDFExportModal from "./PDFExportModal.svelte";

  // Panel visibility state
  let showDraftPanel = $state(false);
  let showSummaryModal = $state(false);
  let showStatsModal = $state(false);
  let showPDFExportModal = $state(false);
  let messageViewFocused = $state(true);

  // Compose message state
  let composeText = $state("");
  let sendingMessage = $state(false);
  let sendError = $state<string | null>(null);

  // Auto-resize textarea as user types
  function autoResizeTextarea(event: Event) {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  }

  // Handle Enter key in compose input
  function handleComposeKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }

  // Send the message
  async function handleSendMessage() {
    if (!composeText.trim() || sendingMessage || !$selectedConversation) return;

    sendingMessage = true;
    sendError = null;

    try {
      const chatId = $selectedConversation.chat_id;
      const isGroup = $selectedConversation.is_group;
      const recipient = !isGroup && $selectedConversation.participants?.length > 0
        ? $selectedConversation.participants[0]
        : undefined;

      const response = await fetch(`http://localhost:8742/conversations/${encodeURIComponent(chatId)}/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: composeText.trim(),
          recipient: recipient,
          is_group: isGroup,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        sendError = `Send failed: ${response.status} - ${errorText}`;
        return;
      }

      const result = await response.json();

      if (result.success) {
        composeText = "";
        // Reset textarea height
        const textarea = document.querySelector(".compose-input") as HTMLTextAreaElement;
        if (textarea) textarea.style.height = "auto";
        // Refresh messages in background - don't await to prevent blocking
        pollMessages().catch(err => console.error("Poll error:", err));
      } else {
        sendError = result.error || "Failed to send message";
      }
    } catch (err) {
      sendError = "Failed to send message. Check if Messages app has permissions.";
      console.error("Send error:", err);
    } finally {
      sendingMessage = false;
    }
  }

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

  // Track previous message count and scroll height for position restoration (infinite scroll)
  let previousMessageCount = $state(0);
  let previousScrollHeight = $state(0);

  // Threshold for triggering load (200px from top)
  const SCROLL_THRESHOLD = 200;

  // Scroll tracking state for new messages
  let isAtBottom = $state(true);
  let hasNewMessagesBelow = $state(false);
  let newMessageIds = $state<Set<number>>(new Set());

  // Handle scroll event for infinite scroll and bottom detection
  async function handleScroll(event: Event) {
    const container = event.target as HTMLDivElement;
    if (!container) return;

    // Check if user scrolled near the top (for loading older messages)
    if (
      container.scrollTop < SCROLL_THRESHOLD &&
      $conversationsStore.hasMore &&
      !$conversationsStore.loadingMore &&
      $conversationsStore.messages.length > 0
    ) {
      // Save current scroll position info before loading
      previousScrollHeight = container.scrollHeight;
      const prevCount = $conversationsStore.messages.length;

      await loadMoreMessages();

      // Restore scroll position after messages are prepended
      tick().then(() => {
        if (messagesContainer && $conversationsStore.messages.length > prevCount) {
          const newScrollHeight = messagesContainer.scrollHeight;
          const scrollDelta = newScrollHeight - previousScrollHeight;
          messagesContainer.scrollTop += scrollDelta;
        }
      });
    }

    // Check if user is at bottom (for new message indicators)
    const { scrollTop, scrollHeight, clientHeight } = container;
    const threshold = 50; // pixels from bottom to consider "at bottom"
    isAtBottom = scrollHeight - scrollTop - clientHeight < threshold;

    // Clear "new messages below" if user scrolls to bottom
    if (isAtBottom) {
      hasNewMessagesBelow = false;
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

  // Track message count to detect new messages (from polling)
  $effect(() => {
    const currentCount = $conversationsStore.messages.length;
    if (currentCount > previousMessageCount && previousMessageCount > 0) {
      // New messages arrived
      const newMessages = $conversationsStore.messages.slice(previousMessageCount);
      const newIds = new Set(newMessageIds);
      newMessages.forEach((m) => newIds.add(m.id));
      newMessageIds = newIds;

      if (isAtBottom) {
        // Auto-scroll to bottom if user was already at bottom
        scrollToBottom();
      } else {
        // Show "new messages below" indicator
        hasNewMessagesBelow = true;
      }

      // Clear new message highlight after animation
      setTimeout(() => {
        newMessageIds = new Set();
      }, 2000);
    }
    previousMessageCount = currentCount;
  });

  // Reset state when conversation changes
  $effect(() => {
    if ($selectedConversation) {
      previousMessageCount = 0;
      newMessageIds = new Set();
      hasNewMessagesBelow = false;
      isAtBottom = true;
      // Scroll to bottom when loading new conversation
      tick().then(() => scrollToBottom());
    }
  });

  // Scroll to bottom of messages
  async function scrollToBottom() {
    await tick();
    if (messagesContainer) {
      messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: "smooth",
      });
      hasNewMessagesBelow = false;
    }
  }

  // Handle "new messages below" button click
  function handleNewMessagesClick() {
    scrollToBottom();
  }

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
    } else if (isMod && event.key === "e") {
      event.preventDefault();
      if ($selectedConversation) {
        showPDFExportModal = true;
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

  // Handle scrolling to a message from search results
  let unsubscribeScroll: (() => void) | null = null;

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);

    // Subscribe to scroll target changes
    unsubscribeScroll = scrollToMessageId.subscribe(async (messageId) => {
      if (messageId !== null) {
        // Wait for DOM to update
        await tick();
        const element = document.querySelector(`[data-message-id="${messageId}"]`);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        clearScrollTarget();
      }
    });
  });

  onDestroy(() => {
    window.removeEventListener("keydown", handleKeydown);
    stopMessagePolling();
    unsubscribeScroll?.();
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

  function isNewMessage(messageId: number): boolean {
    return newMessageIds.has(messageId);
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
          onclick={() => showStatsModal = true}
          title="View conversation statistics"
          aria-label="View statistics"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
          </svg>
        </button>
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
            <div class="system-message" data-message-id={message.id}>
              {message.text}
            </div>
          {:else}
            <div
              class="message"
              class:from-me={message.is_from_me}
              class:new-message={isNewMessage(message.id)}
              class:highlighted={$highlightedMessageId === message.id}
              data-message-id={message.id}
            >
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

    <!-- Smart Reply Chips - HIDDEN: Using AI Draft button instead (LLM-powered) -->
    <!-- TODO: Re-enable when SmartReplyChips uses LLM instead of templates -->
    <!-- {#if getLastReceivedMessage()}
      <SmartReplyChips
        lastMessage={getLastReceivedMessage()}
        isFocused={messageViewFocused}
      />
    {/if} -->

    {#if hasNewMessagesBelow}
      <button class="new-messages-button" onclick={handleNewMessagesClick}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
        New messages below
      </button>
    {/if}

    <!-- Compose Message Input -->
    <div class="compose-area">
      <div class="compose-input-wrapper">
        <textarea
          class="compose-input"
          bind:value={composeText}
          placeholder="iMessage"
          rows="1"
          onkeydown={handleComposeKeydown}
          oninput={autoResizeTextarea}
        ></textarea>
        <button
          class="send-button"
          onclick={handleSendMessage}
          disabled={!composeText.trim() || sendingMessage}
          title="Send message (Enter)"
        >
          {#if sendingMessage}
            <div class="send-spinner"></div>
          {:else}
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          {/if}
        </button>
      </div>
      {#if sendError}
        <div class="send-error">{sendError}</div>
      {/if}
    </div>
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

<!-- Stats Modal -->
{#if showStatsModal && $selectedConversation}
  <ConversationStats
    chatId={$selectedConversation.chat_id}
    onClose={() => showStatsModal = false}
  />
{/if}

<!-- PDF Export Modal -->
{#if showPDFExportModal && $selectedConversation}
  <PDFExportModal
    chatId={$selectedConversation.chat_id}
    conversationName={$selectedConversation.display_name || $selectedConversation.participants.join(", ")}
    onClose={() => showPDFExportModal = false}
  />
{/if}

<style>
  .message-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    position: relative;
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
    animation: none;
  }

  .message.new-message {
    animation: slideIn 0.3s ease-out, highlight 2s ease-out;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes highlight {
    0% {
      background: rgba(0, 122, 255, 0.2);
    }
    100% {
      background: transparent;
    }
  }

  .message.from-me {
    align-self: flex-end;
  }

  .message.highlighted {
    animation: highlightPulse 3s ease-out;
  }

  @keyframes highlightPulse {
    0% {
      background: rgba(251, 191, 36, 0.4);
      border-radius: 12px;
    }
    50% {
      background: rgba(251, 191, 36, 0.2);
    }
    100% {
      background: transparent;
    }
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

  .new-messages-button {
    position: absolute;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    background: var(--accent-color);
    color: white;
    border: none;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
    animation: bounceIn 0.3s ease-out;
    z-index: 10;
  }

  .new-messages-button:hover {
    background: #0a82e0;
    transform: translateX(-50%) scale(1.05);
  }

  .new-messages-button svg {
    width: 16px;
    height: 16px;
  }

  @keyframes bounceIn {
    0% {
      opacity: 0;
      transform: translateX(-50%) translateY(20px);
    }
    60% {
      transform: translateX(-50%) translateY(-5px);
    }
    100% {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }

  /* Compose Area Styles */
  .compose-area {
    padding: 8px 16px 16px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .compose-input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 8px 8px 8px 16px;
  }

  .compose-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 15px;
    line-height: 1.4;
    resize: none;
    outline: none;
    max-height: 120px;
    min-height: 24px;
  }

  .compose-input::placeholder {
    color: var(--text-secondary);
  }

  .send-button {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--accent-color);
    border: none;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: opacity 0.15s ease, transform 0.15s ease;
  }

  .send-button:hover:not(:disabled) {
    transform: scale(1.05);
  }

  .send-button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .send-button svg {
    width: 18px;
    height: 18px;
  }

  .send-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .send-error {
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(255, 59, 48, 0.1);
    border: 1px solid rgba(255, 59, 48, 0.3);
    border-radius: 8px;
    color: #ff3b30;
    font-size: 13px;
  }
</style>
