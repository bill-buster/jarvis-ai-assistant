<script lang="ts">
  import { onMount } from "svelte";
  import {
    conversationsStore,
    fetchConversations,
    selectConversation,
  } from "../stores/conversations";

  onMount(() => {
    fetchConversations();
  });

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function getDisplayName(conv: typeof $conversationsStore.conversations[0]): string {
    if (conv.display_name) return conv.display_name;
    if (conv.participants.length === 1) return conv.participants[0];
    return conv.participants.slice(0, 2).join(", ") +
      (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : "");
  }
</script>

<div class="conversation-list">
  <div class="header">
    <h2>Messages</h2>
  </div>

  <div class="search">
    <input type="text" placeholder="Search conversations..." />
  </div>

  {#if $conversationsStore.loading}
    <div class="loading">Loading conversations...</div>
  {:else if $conversationsStore.error}
    <div class="error">{$conversationsStore.error}</div>
  {:else if $conversationsStore.conversations.length === 0}
    <div class="empty">No conversations found</div>
  {:else}
    <div class="list">
      {#each $conversationsStore.conversations as conv (conv.chat_id)}
        <button
          class="conversation"
          class:active={$conversationsStore.selectedChatId === conv.chat_id}
          class:group={conv.is_group}
          on:click={() => selectConversation(conv.chat_id)}
        >
          <div class="avatar" class:group={conv.is_group}>
            {#if conv.is_group}
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm-6 8v-2c0-2.67 5.33-4 6-4s6 1.33 6 4v2H6zm10-8c1.93 0 3.5-1.57 3.5-3.5S17.93 5 16 5c-.54 0-1.04.13-1.5.35.63.89 1 1.98 1 3.15s-.37 2.26-1 3.15c.46.22.96.35 1.5.35z"/>
              </svg>
            {:else}
              {getDisplayName(conv).charAt(0).toUpperCase()}
            {/if}
          </div>
          <div class="info">
            <div class="name-row">
              <span class="name">{getDisplayName(conv)}</span>
              <span class="date">{formatDate(conv.last_message_date)}</span>
            </div>
            <div class="preview">
              {conv.last_message_text || "No messages"}
            </div>
          </div>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .conversation-list {
    width: 300px;
    min-width: 300px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
  }

  .header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .header h2 {
    font-size: 20px;
    font-weight: 600;
  }

  .search {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .search input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .search input::placeholder {
    color: var(--text-secondary);
  }

  .list {
    flex: 1;
    overflow-y: auto;
  }

  .conversation {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s ease;
  }

  .conversation:hover {
    background: var(--bg-hover);
  }

  .conversation.active {
    background: var(--bg-active);
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 18px;
    color: white;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .avatar svg {
    width: 24px;
    height: 24px;
  }

  .info {
    flex: 1;
    min-width: 0;
  }

  .name-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .name {
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .date {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
    margin-left: 8px;
  }

  .preview {
    font-size: 13px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .loading,
  .error,
  .empty {
    padding: 24px 16px;
    text-align: center;
    color: var(--text-secondary);
  }

  .error {
    color: var(--error-color);
  }
</style>
