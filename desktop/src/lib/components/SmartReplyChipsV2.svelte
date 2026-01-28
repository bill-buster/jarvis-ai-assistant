<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { v2Api, type V2GeneratedReply } from "../api/v2";

  // Props
  let { chatId = "", isFocused = false }: { chatId: string; isFocused: boolean } = $props();

  // State
  let suggestions = $state<V2GeneratedReply[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let visible = $state(false);
  let toastMessage = $state<string | null>(null);
  let toastTimeout: ReturnType<typeof setTimeout> | null = null;
  let generationTime = $state<number | null>(null);
  let modelUsed = $state<string | null>(null);

  // Fetch suggestions when chatId changes
  let lastFetchedChatId = "";

  $effect(() => {
    if (chatId && chatId !== lastFetchedChatId) {
      lastFetchedChatId = chatId;
      fetchSuggestions(chatId);
    }
  });

  async function fetchSuggestions(id: string) {
    loading = true;
    error = null;
    visible = false;
    suggestions = [];

    try {
      const response = await v2Api.generateReplies(id, 3);
      suggestions = response.replies;
      generationTime = response.generation_time_ms;
      modelUsed = response.model_used;

      // Trigger animation after a small delay
      setTimeout(() => {
        visible = true;
      }, 50);
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to generate replies";
      suggestions = [];
    } finally {
      loading = false;
    }
  }

  // Manual refresh
  function handleRefresh() {
    if (chatId) {
      lastFetchedChatId = ""; // Force re-fetch
      fetchSuggestions(chatId);
    }
  }

  async function handleChipClick(suggestion: V2GeneratedReply) {
    try {
      await navigator.clipboard.writeText(suggestion.text);
      showToast("Copied to clipboard");
    } catch (err) {
      showToast("Failed to copy");
    }
  }

  function showToast(message: string) {
    toastMessage = message;
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
    toastTimeout = setTimeout(() => {
      toastMessage = null;
    }, 2000);
  }

  // Keyboard shortcuts handler
  function handleKeydown(event: KeyboardEvent) {
    if (!isFocused || suggestions.length === 0) return;

    // Check if user pressed 1, 2, or 3
    const keyNum = parseInt(event.key);
    if (keyNum >= 1 && keyNum <= 3 && suggestions[keyNum - 1]) {
      event.preventDefault();
      handleChipClick(suggestions[keyNum - 1]);
    }
  }

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener("keydown", handleKeydown);
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
  });
</script>

<div class="smart-reply-container">
  {#if loading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Generating replies...</span>
    </div>
  {:else if error}
    <div class="error-state">
      <span class="error-icon">!</span>
      <span class="error-text">{error}</span>
      <button class="retry-btn" onclick={handleRefresh}>Retry</button>
    </div>
  {:else if suggestions.length > 0}
    <div class="suggestions-header">
      <span class="header-label">Suggested replies</span>
      {#if generationTime}
        <span class="generation-info">{(generationTime / 1000).toFixed(1)}s</span>
      {/if}
      <button class="refresh-btn" onclick={handleRefresh} title="Generate new replies">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
        </svg>
      </button>
    </div>
    <div class="chips-container" class:visible>
      {#each suggestions as suggestion, i}
        <button
          class="chip"
          onclick={() => handleChipClick(suggestion)}
          title="Click to copy (or press {i + 1})"
        >
          <span class="chip-number">{i + 1}</span>
          <span class="chip-text">{suggestion.text}</span>
        </button>
      {/each}
    </div>
  {/if}

  {#if toastMessage}
    <div class="toast">{toastMessage}</div>
  {/if}
</div>

<style>
  .smart-reply-container {
    padding: 8px 12px;
    border-top: 1px solid var(--border-color, #e0e0e0);
    background: var(--bg-secondary, #f8f9fa);
  }

  .loading-state {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary, #666);
    font-size: 13px;
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border-color, #e0e0e0);
    border-top-color: var(--accent-color, #007aff);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-state {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--error-color, #dc3545);
    font-size: 13px;
  }

  .error-icon {
    width: 18px;
    height: 18px;
    background: var(--error-color, #dc3545);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 12px;
  }

  .error-text {
    flex: 1;
  }

  .retry-btn {
    padding: 4px 8px;
    border: 1px solid var(--border-color, #e0e0e0);
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 12px;
  }

  .retry-btn:hover {
    background: var(--bg-hover, #f0f0f0);
  }

  .suggestions-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .header-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary, #666);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .generation-info {
    font-size: 10px;
    color: var(--text-tertiary, #999);
  }

  .refresh-btn {
    margin-left: auto;
    padding: 4px;
    border: none;
    background: transparent;
    cursor: pointer;
    color: var(--text-secondary, #666);
    border-radius: 4px;
  }

  .refresh-btn:hover {
    background: var(--bg-hover, #f0f0f0);
    color: var(--accent-color, #007aff);
  }

  .refresh-btn svg {
    width: 14px;
    height: 14px;
  }

  .chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.2s ease, transform 0.2s ease;
  }

  .chips-container.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid var(--border-color, #e0e0e0);
    border-radius: 16px;
    background: white;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s ease;
    max-width: 100%;
  }

  .chip:hover {
    border-color: var(--accent-color, #007aff);
    background: var(--accent-bg, #f0f7ff);
  }

  .chip:active {
    transform: scale(0.98);
  }

  .chip-number {
    width: 16px;
    height: 16px;
    background: var(--bg-tertiary, #eee);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 600;
    color: var(--text-secondary, #666);
    flex-shrink: 0;
  }

  .chip:hover .chip-number {
    background: var(--accent-color, #007aff);
    color: white;
  }

  .chip-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .toast {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 16px;
    background: var(--toast-bg, #333);
    color: white;
    border-radius: 8px;
    font-size: 13px;
    animation: fadeIn 0.2s ease;
    z-index: 1000;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateX(-50%) translateY(8px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
</style>
