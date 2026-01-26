<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { api, APIError } from "../api/client";
  import type { Message, SearchFilters, SemanticSearchResultItem } from "../api/types";
  import { navigateToMessage, conversationsStore } from "../stores/conversations";

  interface Props {
    onClose: () => void;
  }

  let { onClose }: Props = $props();

  // Search mode: text (keyword) or semantic (AI-powered)
  type SearchMode = "text" | "semantic";
  let searchMode: SearchMode = $state("text");

  // Search state
  type SearchState = "idle" | "searching" | "results" | "no-results" | "error";
  let searchState: SearchState = $state("idle");
  let query = $state("");
  let results: Message[] = $state([]);
  let semanticResults: SemanticSearchResultItem[] = $state([]);
  let errorMessage = $state("");
  let selectedIndex = $state(-1);

  // Semantic search settings
  let semanticThreshold = $state(0.3);

  // Filter state
  let showFilters = $state(false);
  let filterSender = $state("");
  let filterStartDate = $state("");
  let filterEndDate = $state("");
  let filterHasAttachments = $state(false);

  // Debounce timer
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let abortController: AbortController | null = null;

  // Input ref for focus
  let searchInput: HTMLInputElement | null = null;

  // Group results by conversation
  interface GroupedResults {
    chat_id: string;
    conversation_name: string;
    messages: Array<Message & { similarity?: number }>;
  }

  let groupedResults: GroupedResults[] = $derived.by(() => {
    const items = searchMode === "semantic"
      ? semanticResults.map(r => ({ ...r.message, similarity: r.similarity }))
      : results;

    if (items.length === 0) return [];

    const groups = new Map<string, GroupedResults>();

    for (const msg of items) {
      if (!groups.has(msg.chat_id)) {
        // Find conversation name from store
        const conv = $conversationsStore.conversations.find(c => c.chat_id === msg.chat_id);
        groups.set(msg.chat_id, {
          chat_id: msg.chat_id,
          conversation_name: conv?.display_name || conv?.participants?.join(", ") || msg.chat_id,
          messages: []
        });
      }
      groups.get(msg.chat_id)!.messages.push(msg);
    }

    return Array.from(groups.values());
  });

  // Flat list of all messages for keyboard navigation
  let flatResults: { msg: Message & { similarity?: number }; groupIndex: number; msgIndex: number }[] = $derived.by(() => {
    const flat: { msg: Message & { similarity?: number }; groupIndex: number; msgIndex: number }[] = [];
    groupedResults.forEach((group, groupIndex) => {
      group.messages.forEach((msg, msgIndex) => {
        flat.push({ msg, groupIndex, msgIndex });
      });
    });
    return flat;
  });

  // Total results count
  let totalResultsCount = $derived(searchMode === "semantic" ? semanticResults.length : results.length);

  onMount(() => {
    searchInput?.focus();
  });

  onDestroy(() => {
    if (debounceTimer) clearTimeout(debounceTimer);
    abortController?.abort();
  });

  function handleInput() {
    // Clear previous timer
    if (debounceTimer) clearTimeout(debounceTimer);

    if (!query.trim()) {
      searchState = "idle";
      results = [];
      semanticResults = [];
      selectedIndex = -1;
      return;
    }

    // Debounce search - longer for semantic to avoid unnecessary API calls
    const delay = searchMode === "semantic" ? 500 : 300;
    debounceTimer = setTimeout(() => {
      performSearch();
    }, delay);
  }

  async function performSearch() {
    const searchQuery = query.trim();
    if (!searchQuery) return;

    // Abort any existing request
    abortController?.abort();
    abortController = new AbortController();

    searchState = "searching";
    errorMessage = "";
    selectedIndex = -1;

    try {
      if (searchMode === "semantic") {
        await performSemanticSearch(searchQuery);
      } else {
        await performTextSearch(searchQuery);
      }
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        return;
      }
      searchState = "error";
      if (e instanceof APIError) {
        errorMessage = e.detail || e.message;
      } else if (e instanceof Error) {
        errorMessage = e.message;
      } else {
        errorMessage = "An unknown error occurred";
      }
    }
  }

  async function performTextSearch(searchQuery: string) {
    const filters: SearchFilters = {};
    if (filterSender.trim()) {
      filters.sender = filterSender.trim();
    }
    if (filterStartDate) {
      filters.after = new Date(filterStartDate).toISOString();
    }
    if (filterEndDate) {
      filters.before = new Date(filterEndDate).toISOString();
    }
    if (filterHasAttachments) {
      filters.has_attachments = true;
    }

    const searchResults = await api.searchMessages(
      searchQuery,
      filters,
      100,
      abortController!.signal
    );

    results = searchResults;
    semanticResults = [];
    searchState = searchResults.length > 0 ? "results" : "no-results";
  }

  async function performSemanticSearch(searchQuery: string) {
    const filters: { sender?: string; after?: string; before?: string } = {};
    if (filterSender.trim()) {
      filters.sender = filterSender.trim();
    }
    if (filterStartDate) {
      filters.after = new Date(filterStartDate).toISOString();
    }
    if (filterEndDate) {
      filters.before = new Date(filterEndDate).toISOString();
    }

    const response = await api.semanticSearch(
      searchQuery,
      {
        limit: 50,
        threshold: semanticThreshold,
        indexLimit: 1000,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
      },
      abortController!.signal
    );

    semanticResults = response.results;
    results = [];
    searchState = response.results.length > 0 ? "results" : "no-results";
  }

  function handleKeyDown(event: KeyboardEvent) {
    switch (event.key) {
      case "Escape":
        onClose();
        break;
      case "ArrowDown":
        event.preventDefault();
        if (flatResults.length > 0) {
          selectedIndex = Math.min(selectedIndex + 1, flatResults.length - 1);
          scrollSelectedIntoView();
        }
        break;
      case "ArrowUp":
        event.preventDefault();
        if (flatResults.length > 0) {
          selectedIndex = Math.max(selectedIndex - 1, 0);
          scrollSelectedIntoView();
        }
        break;
      case "Enter":
        event.preventDefault();
        if (selectedIndex >= 0 && flatResults[selectedIndex]) {
          handleResultClick(flatResults[selectedIndex].msg);
        } else if (query.trim()) {
          performSearch();
        }
        break;
    }
  }

  function scrollSelectedIntoView() {
    const element = document.querySelector(`[data-result-index="${selectedIndex}"]`);
    element?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  async function handleResultClick(message: Message) {
    await navigateToMessage(message.chat_id, message.id);
    onClose();
  }

  function highlightMatch(text: string, searchQuery: string): string {
    if (!searchQuery.trim() || !text) return text;

    // For semantic search, don't highlight (no exact matches)
    if (searchMode === "semantic") return text;

    // Escape special regex characters in the search query
    const escaped = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escaped})`, "gi");
    return text.replace(regex, '<mark class="highlight">$1</mark>');
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: "short" });
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  }

  function getSnippet(text: string, maxLength: number = 150): string {
    if (!text) return "";
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }

  function toggleFilters() {
    showFilters = !showFilters;
  }

  function clearFilters() {
    filterSender = "";
    filterStartDate = "";
    filterEndDate = "";
    filterHasAttachments = false;
    if (query.trim()) {
      performSearch();
    }
  }

  function handleOverlayClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function toggleSearchMode() {
    searchMode = searchMode === "text" ? "semantic" : "text";
    // Clear results and re-search
    results = [];
    semanticResults = [];
    if (query.trim()) {
      performSearch();
    }
  }

  function formatSimilarity(similarity: number): string {
    return `${Math.round(similarity * 100)}%`;
  }
</script>

<svelte:window onkeydown={handleKeyDown} />

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="search-overlay" onclick={handleOverlayClick} role="presentation">
  <!-- svelte-ignore a11y_interactive_supports_focus -->
  <div class="search-modal" onclick={(e) => e.stopPropagation()} role="dialog" aria-label="Global Search">
    <div class="search-header">
      <div class="search-input-wrapper">
        {#if searchMode === "semantic"}
          <svg class="search-icon semantic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" title="Semantic Search">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v10M1 12h6m6 0h10M4.22 4.22l4.24 4.24m7.08 7.08l4.24 4.24M4.22 19.78l4.24-4.24m7.08-7.08l4.24-4.24"></path>
          </svg>
        {:else}
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
        {/if}
        <input
          bind:this={searchInput}
          type="text"
          class="search-input"
          placeholder={searchMode === "semantic" ? "Search by meaning..." : "Search messages..."}
          bind:value={query}
          oninput={handleInput}
        />
        {#if query}
          <button class="clear-btn" onclick={() => { query = ""; searchState = "idle"; results = []; semanticResults = []; }} aria-label="Clear search">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        {/if}
      </div>
      <button
        class="mode-toggle"
        class:semantic={searchMode === "semantic"}
        onclick={toggleSearchMode}
        aria-label="Toggle search mode"
        title={searchMode === "semantic" ? "Switch to keyword search" : "Switch to semantic search"}
      >
        {#if searchMode === "semantic"}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v10M1 12h6m6 0h10"></path>
          </svg>
        {:else}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        {/if}
      </button>
      <button class="filter-toggle" class:active={showFilters} onclick={toggleFilters} aria-label="Toggle filters" title="Filters">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
        </svg>
      </button>
      <button class="close-btn" onclick={onClose} aria-label="Close">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>

    {#if showFilters}
      <div class="filters-panel">
        <div class="filter-row">
          <div class="filter-group">
            <label for="filter-sender">Sender</label>
            <input
              id="filter-sender"
              type="text"
              placeholder="Phone or email"
              bind:value={filterSender}
              onchange={performSearch}
            />
          </div>
          <div class="filter-group">
            <label for="filter-start">From</label>
            <input
              id="filter-start"
              type="date"
              bind:value={filterStartDate}
              onchange={performSearch}
            />
          </div>
          <div class="filter-group">
            <label for="filter-end">To</label>
            <input
              id="filter-end"
              type="date"
              bind:value={filterEndDate}
              onchange={performSearch}
            />
          </div>
        </div>
        <div class="filter-row">
          {#if searchMode === "text"}
            <label class="checkbox-label">
              <input
                type="checkbox"
                bind:checked={filterHasAttachments}
                onchange={performSearch}
              />
              <span>Has attachments</span>
            </label>
          {:else}
            <div class="threshold-control">
              <label for="threshold">Similarity: {Math.round(semanticThreshold * 100)}%</label>
              <input
                id="threshold"
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                bind:value={semanticThreshold}
                onchange={performSearch}
              />
            </div>
          {/if}
          <button class="clear-filters-btn" onclick={clearFilters}>Clear filters</button>
        </div>
      </div>
    {/if}

    <div class="search-results">
      {#if searchState === "idle"}
        <div class="empty-state">
          {#if searchMode === "semantic"}
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M12 1v6m0 6v10M1 12h6m6 0h10M4.22 4.22l4.24 4.24m7.08 7.08l4.24 4.24M4.22 19.78l4.24-4.24m7.08-7.08l4.24-4.24"></path>
            </svg>
            <p>Semantic search finds messages by meaning</p>
            <span class="hint">Try "dinner plans" to find messages about eating out</span>
          {:else}
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.35-4.35"></path>
            </svg>
            <p>Search across all your conversations</p>
            <span class="hint">Use filters to narrow down results</span>
          {/if}
        </div>
      {:else if searchState === "searching"}
        <div class="loading-state">
          <div class="spinner"></div>
          <p>{searchMode === "semantic" ? "Computing semantic similarity..." : "Searching..."}</p>
        </div>
      {:else if searchState === "error"}
        <div class="error-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <p>{errorMessage}</p>
          <button class="retry-btn" onclick={performSearch}>Try Again</button>
        </div>
      {:else if searchState === "no-results"}
        <div class="no-results-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
            <line x1="8" y1="11" x2="14" y2="11"></line>
          </svg>
          <p>No results found for "{query}"</p>
          <div class="search-tips">
            <h4>Search tips:</h4>
            <ul>
              {#if searchMode === "semantic"}
                <li>Try describing what you're looking for differently</li>
                <li>Lower the similarity threshold in filters</li>
                <li>Use simpler, more common phrases</li>
              {:else}
                <li>Try different keywords</li>
                <li>Check for typos</li>
                <li>Use fewer or more general words</li>
              {/if}
              <li>Try removing filters</li>
            </ul>
          </div>
        </div>
      {:else if searchState === "results"}
        <div class="results-count">
          {totalResultsCount} result{totalResultsCount !== 1 ? "s" : ""} in {groupedResults.length} conversation{groupedResults.length !== 1 ? "s" : ""}
          {#if searchMode === "semantic"}
            <span class="mode-indicator">Semantic</span>
          {/if}
        </div>
        <div class="results-list">
          {#each groupedResults as group, groupIndex}
            <div class="result-group">
              <div class="group-header">
                <span class="conversation-name">{group.conversation_name}</span>
                <span class="message-count">{group.messages.length}</span>
              </div>
              {#each group.messages as message, msgIndex}
                {@const flatIndex = flatResults.findIndex(f => f.msg.id === message.id)}
                <button
                  class="result-item"
                  class:selected={selectedIndex === flatIndex}
                  data-result-index={flatIndex}
                  onclick={() => handleResultClick(message)}
                >
                  <div class="result-meta">
                    <span class="sender">{message.sender_name || message.sender}</span>
                    <div class="meta-right">
                      {#if searchMode === "semantic" && message.similarity !== undefined}
                        <span class="similarity-badge">{formatSimilarity(message.similarity)}</span>
                      {/if}
                      <span class="date">{formatDate(message.date)}</span>
                    </div>
                  </div>
                  <div class="result-text">
                    {@html highlightMatch(getSnippet(message.text), query)}
                  </div>
                  {#if message.attachments.length > 0}
                    <div class="attachment-indicator">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                      </svg>
                      <span>{message.attachments.length} attachment{message.attachments.length !== 1 ? "s" : ""}</span>
                    </div>
                  {/if}
                </button>
              {/each}
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <div class="search-footer">
      <div class="keyboard-hints">
        <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
        <span><kbd>Enter</kbd> Select</span>
        <span><kbd>Esc</kbd> Close</span>
      </div>
      <div class="mode-hint">
        {searchMode === "semantic" ? "AI-powered" : "Keyword"} search
      </div>
    </div>
  </div>
</div>

<style>
  .search-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 10vh;
    z-index: 1000;
    animation: fadeIn 0.15s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .search-modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    width: 90%;
    max-width: 640px;
    max-height: 70vh;
    display: flex;
    flex-direction: column;
    animation: slideDown 0.2s ease;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  }

  @keyframes slideDown {
    from {
      transform: translateY(-20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  .search-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .search-input-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px 12px;
    transition: border-color 0.15s ease;
  }

  .search-input-wrapper:focus-within {
    border-color: var(--accent-color);
  }

  .search-icon {
    width: 18px;
    height: 18px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .search-icon.semantic {
    color: var(--accent-color);
  }

  .search-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 15px;
    outline: none;
  }

  .search-input::placeholder {
    color: var(--text-secondary);
  }

  .clear-btn {
    background: none;
    border: none;
    padding: 4px;
    cursor: pointer;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .clear-btn:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .clear-btn svg {
    width: 14px;
    height: 14px;
  }

  .mode-toggle,
  .filter-toggle,
  .close-btn {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px;
    cursor: pointer;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
  }

  .mode-toggle:hover,
  .filter-toggle:hover,
  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--accent-color);
  }

  .mode-toggle.semantic {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .filter-toggle.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .mode-toggle svg,
  .filter-toggle svg,
  .close-btn svg {
    width: 18px;
    height: 18px;
  }

  .filters-panel {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-primary);
    animation: slideDown 0.15s ease;
  }

  .filter-row {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }

  .filter-row + .filter-row {
    margin-top: 12px;
  }

  .filter-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    min-width: 120px;
  }

  .filter-group label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .filter-group input {
    padding: 8px 10px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s ease;
  }

  .filter-group input:focus {
    border-color: var(--accent-color);
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .checkbox-label input[type="checkbox"] {
    accent-color: var(--accent-color);
    width: 16px;
    height: 16px;
  }

  .threshold-control {
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
  }

  .threshold-control label {
    font-size: 12px;
    color: var(--text-secondary);
    white-space: nowrap;
  }

  .threshold-control input[type="range"] {
    flex: 1;
    accent-color: var(--accent-color);
    max-width: 150px;
  }

  .clear-filters-btn {
    background: none;
    border: none;
    color: var(--accent-color);
    font-size: 13px;
    cursor: pointer;
    padding: 4px 8px;
    margin-left: auto;
  }

  .clear-filters-btn:hover {
    text-decoration: underline;
  }

  .search-results {
    flex: 1;
    overflow-y: auto;
    min-height: 200px;
  }

  .empty-state,
  .loading-state,
  .error-state,
  .no-results-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    color: var(--text-secondary);
    text-align: center;
  }

  .empty-state svg,
  .no-results-state svg,
  .error-state svg {
    width: 48px;
    height: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state p,
  .loading-state p,
  .error-state p,
  .no-results-state p {
    font-size: 15px;
    margin-bottom: 8px;
  }

  .hint {
    font-size: 13px;
    opacity: 0.7;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 16px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-state svg {
    color: var(--error-color);
    opacity: 1;
  }

  .retry-btn {
    padding: 8px 16px;
    background: var(--error-color);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 13px;
    cursor: pointer;
    margin-top: 8px;
    transition: opacity 0.15s ease;
  }

  .retry-btn:hover {
    opacity: 0.9;
  }

  .search-tips {
    margin-top: 16px;
    text-align: left;
    background: var(--bg-primary);
    padding: 16px;
    border-radius: 8px;
  }

  .search-tips h4 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .search-tips ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .search-tips li {
    font-size: 13px;
    color: var(--text-secondary);
    padding: 4px 0;
  }

  .search-tips li::before {
    content: "*";
    margin-right: 8px;
    color: var(--accent-color);
  }

  .results-count {
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .mode-indicator {
    background: var(--accent-color);
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
  }

  .results-list {
    padding: 8px;
  }

  .result-group {
    margin-bottom: 16px;
  }

  .group-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .conversation-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }

  .message-count {
    background: var(--bg-hover);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    margin-left: 8px;
  }

  .result-item {
    width: 100%;
    text-align: left;
    background: var(--bg-primary);
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .result-item:hover {
    background: var(--bg-hover);
    border-color: var(--border-color);
  }

  .result-item.selected {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .result-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .meta-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .sender {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .date {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .similarity-badge {
    background: var(--accent-color);
    color: white;
    padding: 2px 6px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 500;
  }

  .result-text {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.4;
    word-break: break-word;
  }

  :global(.highlight) {
    background-color: #fbbf24;
    color: #1c1c1e;
    padding: 1px 2px;
    border-radius: 2px;
  }

  .attachment-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .attachment-indicator svg {
    width: 14px;
    height: 14px;
  }

  .search-footer {
    padding: 10px 16px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-primary);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .keyboard-hints {
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .keyboard-hints span {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .mode-hint {
    font-size: 11px;
    color: var(--text-secondary);
    opacity: 0.7;
  }

  kbd {
    display: inline-block;
    padding: 2px 6px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-family: inherit;
    font-size: 11px;
  }
</style>
