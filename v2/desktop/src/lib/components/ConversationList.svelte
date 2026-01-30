<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type { Conversation } from "../api/types";
  import { api } from "../api/client";

  export let conversations: Conversation[] = [];
  export let selectedChatId: string | null = null;
  export let onSelect: (chatId: string) => void = () => {};
  export let unreadChats: Set<string> = new Set();

  // Track visible conversations for index preloading
  let visibleChatIds = new Set<string>();
  let preloadedChatIds = new Set<string>();
  let observer: IntersectionObserver | null = null;
  let preloadTimeout: ReturnType<typeof setTimeout> | null = null;
  let itemRefs = new Map<string, HTMLElement>();
  let listContainer: HTMLElement;

  // Debounced preload function
  function schedulePreload() {
    if (preloadTimeout) {
      clearTimeout(preloadTimeout);
    }

    preloadTimeout = setTimeout(() => {
      const toPreload = [...visibleChatIds].filter(
        (id) => !preloadedChatIds.has(id)
      );

      if (toPreload.length > 0) {
        // Mark as preloaded immediately to avoid duplicate requests
        toPreload.forEach((id) => preloadedChatIds.add(id));

        // Fire and forget - no need to await
        api.preloadIndices(toPreload).catch((err) => {
          console.warn("Index preload failed:", err);
          // Remove from preloaded set so it can retry later
          toPreload.forEach((id) => preloadedChatIds.delete(id));
        });
      }
    }, 150); // Debounce 150ms while scrolling
  }

  function setupObserver() {
    if (!listContainer) return;

    observer = new IntersectionObserver(
      (entries) => {
        let changed = false;

        for (const entry of entries) {
          const chatId = entry.target.getAttribute("data-chat-id");
          if (!chatId) continue;

          if (entry.isIntersecting) {
            if (!visibleChatIds.has(chatId)) {
              visibleChatIds.add(chatId);
              changed = true;
            }
          } else {
            visibleChatIds.delete(chatId);
          }
        }

        if (changed) {
          schedulePreload();
        }
      },
      {
        root: listContainer, // Use scrollable container as root
        rootMargin: "100px", // Preload 100px before visible
        threshold: 0,
      }
    );

    // Observe all current items
    for (const [_, element] of itemRefs) {
      observer.observe(element);
    }
  }

  function registerItem(chatId: string, element: HTMLElement) {
    itemRefs.set(chatId, element);
    if (observer) {
      observer.observe(element);
    }
  }

  function unregisterItem(chatId: string) {
    const element = itemRefs.get(chatId);
    if (element && observer) {
      observer.unobserve(element);
    }
    itemRefs.delete(chatId);
    visibleChatIds.delete(chatId);
  }

  onMount(() => {
    setupObserver();
  });

  onDestroy(() => {
    if (observer) {
      observer.disconnect();
      observer = null;
    }
    if (preloadTimeout) {
      clearTimeout(preloadTimeout);
    }
    itemRefs.clear();
    visibleChatIds.clear();
  });

  // Svelte action for observing items
  function observeItem(node: HTMLElement, chatId: string) {
    registerItem(chatId, node);

    return {
      update(newChatId: string) {
        // If chat_id changes (unlikely but handle it)
        if (newChatId !== chatId) {
          unregisterItem(chatId);
          chatId = newChatId;
          registerItem(chatId, node);
        }
      },
      destroy() {
        unregisterItem(chatId);
      },
    };
  }

  function getInitials(conv: Conversation): string {
    const name = conv.display_name || conv.participants[0] || "?";
    if (name.startsWith("+")) {
      return name.slice(-2);
    }
    const parts = name.split(" ");
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return name[0]?.toUpperCase() || "?";
  }

  function getDisplayName(conv: Conversation): string {
    return conv.display_name || conv.participants[0] || "Unknown";
  }

  function formatTime(dateStr: string | null): string {
    if (!dateStr) return "";
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function truncate(text: string | null, maxLen: number = 40): string {
    if (!text) return "";
    return text.length > maxLen ? text.slice(0, maxLen) + "..." : text;
  }
</script>

<div class="conversation-list" bind:this={listContainer}>
  {#each conversations as conv (conv.chat_id)}
    <button
      class="conversation-item"
      class:selected={conv.chat_id === selectedChatId}
      data-chat-id={conv.chat_id}
      use:observeItem={conv.chat_id}
      on:click={() => onSelect(conv.chat_id)}
    >
      <div class="avatar">
        {getInitials(conv)}
      </div>
      <div class="conversation-info">
        <div class="conversation-name">{getDisplayName(conv)}</div>
        <div class="conversation-preview">{truncate(conv.last_message_text)}</div>
      </div>
      <div class="conversation-meta">
        <div class="conversation-time">{formatTime(conv.last_message_date)}</div>
        {#if unreadChats.has(conv.chat_id)}
          <div class="unread-dot"></div>
        {/if}
      </div>
    </button>
  {/each}
</div>

<style>
  .conversation-list {
    flex: 1;
    overflow-y: auto;
  }

  .conversation-item {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    cursor: pointer;
    border: none;
    border-bottom: 1px solid var(--border-color);
    background: transparent;
    width: 100%;
    text-align: left;
    transition: background 0.15s;
  }

  .conversation-item:hover {
    background: var(--hover-color);
  }

  .conversation-item.selected {
    background: var(--bg-tertiary);
  }

  .avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 16px;
    margin-right: 12px;
    flex-shrink: 0;
    color: white;
  }

  .conversation-info {
    flex: 1;
    min-width: 0;
  }

  .conversation-name {
    font-weight: 600;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-primary);
  }

  .conversation-preview {
    color: var(--text-secondary);
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .conversation-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    margin-left: 8px;
    flex-shrink: 0;
  }

  .conversation-time {
    color: var(--text-secondary);
    font-size: 12px;
  }

  .unread-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-blue);
  }
</style>
