<script lang="ts">
  import { onMount } from "svelte";
  import {
    fetchConversations,
    selectConversation,
    getConversationsStore,
  } from "../stores/conversations";

  const store = getConversationsStore();

  onMount(() => {
    fetchConversations({ limit: 50 });
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

  function getDisplayName(conv: { display_name: string | null; participants: string[] }): string {
    if (conv.display_name) return conv.display_name;
    if (conv.participants.length === 0) return "Unknown";
    if (conv.participants.length === 1) return conv.participants[0];
    return conv.participants.slice(0, 2).join(", ") +
      (conv.participants.length > 2 ? ` +${conv.participants.length - 2}` : "");
  }

  function truncateText(text: string | null, maxLength: number = 40): string {
    if (!text) return "";
    return text.length > maxLength ? text.slice(0, maxLength) + "..." : text;
  }
</script>

<aside class="conversation-list">
  <header class="list-header">
    <h2>Messages</h2>
    <button class="refresh-btn" onclick={() => fetchConversations({ limit: 50 })}>
      ↻
    </button>
  </header>

  {#if store.loadingConversations}
    <div class="loading">Loading conversations...</div>
  {:else if store.error}
    <div class="error">{store.error}</div>
  {:else if store.conversations.length === 0}
    <div class="empty">No conversations found</div>
  {:else}
    <ul class="conversations">
      {#each store.conversations as conv (conv.chat_id)}
        <li>
          <button
            class="conversation-item"
            class:active={store.selectedChatId === conv.chat_id}
            class:group={conv.is_group}
            onclick={() => selectConversation(conv.chat_id)}
          >
            <div class="avatar" class:group={conv.is_group}>
              {#if conv.is_group}
                <span>👥</span>
              {:else}
                <span>{getDisplayName(conv).charAt(0).toUpperCase()}</span>
              {/if}
            </div>
            <div class="conversation-info">
              <div class="conversation-header">
                <span class="name">{getDisplayName(conv)}</span>
                <span class="date">{formatDate(conv.last_message_date)}</span>
              </div>
              <div class="preview">{truncateText(conv.last_message_text)}</div>
            </div>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</aside>

<style>
  .conversation-list {
    width: 280px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }

  .list-header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .list-header h2 {
    font-size: 18px;
    font-weight: 600;
  }

  .refresh-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 18px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .loading,
  .error,
  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-secondary);
  }

  .error {
    color: var(--error-color);
  }

  .conversations {
    list-style: none;
    overflow-y: auto;
    flex: 1;
  }

  .conversation-item {
    width: 100%;
    padding: 12px 16px;
    background: none;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    text-align: left;
    transition: background-color 0.15s ease;
  }

  .conversation-item:hover {
    background: var(--bg-hover);
  }

  .conversation-item.active {
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
    font-size: 18px;
    font-weight: 600;
    color: white;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .conversation-info {
    flex: 1;
    min-width: 0;
  }

  .conversation-header {
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
</style>
