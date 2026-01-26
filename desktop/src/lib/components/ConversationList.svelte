<script lang="ts">
  import { onMount } from "svelte";
  import {
    conversations,
    conversationsLoading,
    conversationsError,
    fetchConversations,
    selectConversation,
    selectedChatId,
  } from "../stores/conversations";

  let searchQuery = "";

  $: filteredConversations = searchQuery
    ? $conversations.filter((c) => {
        const name = c.display_name || c.participants.join(", ");
        return name.toLowerCase().includes(searchQuery.toLowerCase());
      })
    : $conversations;

  function formatLastMessageDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      });
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return date.toLocaleDateString("en-US", { weekday: "short" });
    } else {
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });
    }
  }

  function getDisplayName(conv: (typeof $conversations)[0]): string {
    return conv.display_name || conv.participants.join(", ");
  }

  function truncateText(text: string | null, maxLength: number): string {
    if (!text) return "";
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + "...";
  }

  onMount(() => {
    fetchConversations();
  });
</script>

<div class="conversation-list">
  <div class="search-container">
    <input
      type="text"
      class="search-input"
      placeholder="Search conversations..."
      bind:value={searchQuery}
    />
  </div>

  <div class="list-container">
    {#if $conversationsLoading && $conversations.length === 0}
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading conversations...</p>
      </div>
    {:else if $conversationsError}
      <div class="error-state">
        <p>{$conversationsError}</p>
        <button on:click={() => fetchConversations()}>Retry</button>
      </div>
    {:else if filteredConversations.length === 0}
      <div class="empty-state">
        {#if searchQuery}
          <p>No conversations match "{searchQuery}"</p>
        {:else}
          <p>No conversations found</p>
        {/if}
      </div>
    {:else}
      {#each filteredConversations as conv (conv.chat_id)}
        <button
          class="conversation-item"
          class:selected={$selectedChatId === conv.chat_id}
          on:click={() => selectConversation(conv.chat_id)}
        >
          <div class="avatar" class:group={conv.is_group}>
            {#if conv.is_group}
              👥
            {:else}
              {getDisplayName(conv).charAt(0).toUpperCase()}
            {/if}
          </div>

          <div class="conversation-info">
            <div class="conversation-header">
              <span class="conversation-name">
                {getDisplayName(conv)}
              </span>
              <span class="conversation-time">
                {formatLastMessageDate(conv.last_message_date)}
              </span>
            </div>
            <div class="conversation-preview">
              {truncateText(conv.last_message_text, 50)}
            </div>
          </div>
        </button>
      {/each}
    {/if}
  </div>
</div>

<style>
  .conversation-list {
    width: 300px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
  }

  .search-container {
    padding: 12px;
    border-bottom: 1px solid var(--border-color);
  }

  .search-input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .search-input::placeholder {
    color: var(--text-secondary);
  }

  .search-input:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .list-container {
    flex: 1;
    overflow-y: auto;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 12px;
    color: var(--text-secondary);
    padding: 20px;
    text-align: center;
  }

  .loading-spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-state button {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }

  .conversation-item {
    display: flex;
    gap: 12px;
    padding: 12px 16px;
    background: transparent;
    border: none;
    width: 100%;
    text-align: left;
    cursor: pointer;
    transition: background-color 0.15s;
  }

  .conversation-item:hover {
    background: var(--bg-hover);
  }

  .conversation-item.selected {
    background: var(--bg-active);
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-color);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
    font-size: 20px;
  }

  .conversation-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .conversation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }

  .conversation-name {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .conversation-time {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .conversation-preview {
    font-size: 13px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
