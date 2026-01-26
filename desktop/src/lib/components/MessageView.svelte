<script lang="ts">
  import {
    conversationsStore,
    selectedConversation,
  } from "../stores/conversations";

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

<div class="message-view">
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
    </div>

    <div class="messages">
      {#if $conversationsStore.loadingMessages}
        <div class="loading">Loading messages...</div>
      {:else if $conversationsStore.messages.length === 0}
        <div class="empty">No messages in this conversation</div>
      {:else}
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
  {/if}
</div>

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
</style>
