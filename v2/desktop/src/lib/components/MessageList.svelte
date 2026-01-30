<script lang="ts">
  import type { Message } from "../api/types";
  import { onMount, afterUpdate } from "svelte";

  export let messages: Message[] = [];
  export let loading: boolean = false;

  let container: HTMLDivElement;

  function formatTime(dateStr: string | null): string {
    if (!dateStr) return "";
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    // Today: show relative or time
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 12) return `${diffHours}h ago`;
    if (diffDays === 0) return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return date.toLocaleDateString([], { weekday: "short" });
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function getSenderDisplay(msg: Message): string {
    if (msg.is_from_me) return "";
    return msg.sender_name || msg.sender || "Unknown";
  }

  // Auto-scroll to bottom on new messages
  afterUpdate(() => {
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
</script>

<div class="messages-container" bind:this={container}>
  {#if loading}
    <div class="loading">
      <div class="spinner"></div>
    </div>
  {:else if messages.length === 0}
    <div class="empty-state">No messages</div>
  {:else}
    {#each messages as msg (msg.id)}
      <div class="message" class:incoming={!msg.is_from_me} class:outgoing={msg.is_from_me}>
        {#if !msg.is_from_me}
          <div class="message-sender">{getSenderDisplay(msg)}</div>
        {/if}
        <div class="message-bubble">{msg.text}</div>
        <div class="message-time">{formatTime(msg.timestamp)}</div>
      </div>
    {/each}
  {/if}
</div>

<style>
  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
  }

  .loading,
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-secondary);
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .message {
    max-width: 70%;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
  }

  .message.incoming {
    align-self: flex-start;
  }

  .message.outgoing {
    align-self: flex-end;
  }

  .message-sender {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    margin-left: 12px;
  }

  .message-bubble {
    padding: 10px 14px;
    border-radius: 18px;
    line-height: 1.4;
    word-wrap: break-word;
  }

  .message.incoming .message-bubble {
    background: var(--bubble-incoming);
    border-bottom-left-radius: 4px;
  }

  .message.outgoing .message-bubble {
    background: var(--bubble-outgoing);
    border-bottom-right-radius: 4px;
  }

  .message-time {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
    padding: 0 12px;
  }

  .message.outgoing .message-time {
    text-align: right;
  }
</style>
