<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { listen } from "@tauri-apps/api/event";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import ConversationList from "./lib/components/ConversationList.svelte";
  import MessageView from "./lib/components/MessageView.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import HealthStatus from "./lib/components/HealthStatus.svelte";
  import Settings from "./lib/components/Settings.svelte";
  import GlobalSearch from "./lib/components/GlobalSearch.svelte";
  import { checkApiConnection } from "./lib/stores/health";
  import { clearSelection } from "./lib/stores/conversations";

  let currentView: "messages" | "dashboard" | "health" | "settings" = "messages";
  let showSearch = $state(false);

  function handleKeydown(event: KeyboardEvent) {
    // Cmd+K or Cmd+F to open search (when search is not open)
    const isMod = event.metaKey || event.ctrlKey;
    if (isMod && (event.key === "k" || event.key === "f") && !showSearch) {
      event.preventDefault();
      showSearch = true;
    }
  }

  onMount(async () => {
    // Check API connection on start
    await checkApiConnection();

    // Listen for navigation events from tray menu
    const unlisten = await listen<string>("navigate", (event) => {
      if (
        event.payload === "health" ||
        event.payload === "dashboard" ||
        event.payload === "messages" ||
        event.payload === "settings"
      ) {
        currentView = event.payload;
        if (event.payload !== "messages") {
          clearSelection();
        }
      }
    });

    // Add keyboard listener for search
    window.addEventListener("keydown", handleKeydown);

    // Cleanup on unmount
    return () => {
      unlisten();
    };
  });

  onDestroy(() => {
    window.removeEventListener("keydown", handleKeydown);
  });

  function openSearch() {
    showSearch = true;
  }

  function closeSearch() {
    showSearch = false;
  }
</script>

<main class="app">
  <Sidebar bind:currentView />

  {#if currentView === "dashboard"}
    <Dashboard on:navigate={(e) => currentView = e.detail} />
  {:else if currentView === "health"}
    <HealthStatus />
  {:else if currentView === "settings"}
    <Settings />
  {:else}
    <div class="messages-container">
      <div class="search-bar">
        <button class="search-button" onclick={openSearch} title="Search messages (Cmd+K)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
          <span>Search messages...</span>
          <kbd>⌘K</kbd>
        </button>
      </div>
      <div class="messages-content">
        <ConversationList />
        <MessageView />
      </div>
    </div>
  {/if}
</main>

{#if showSearch}
  <GlobalSearch onClose={closeSearch} />
{/if}

<style>
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    overflow: hidden;
  }

  :global(:root) {
    --bg-primary: #1c1c1e;
    --bg-secondary: #2c2c2e;
    --bg-hover: #3a3a3c;
    --bg-active: #48484a;
    --bg-bubble-me: #0b93f6;
    --bg-bubble-other: #3a3a3c;
    --text-primary: #ffffff;
    --text-secondary: #8e8e93;
    --border-color: #38383a;
    --accent-color: #0b93f6;
    --group-color: #5856d6;
    --error-color: #ff5f57;
  }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
  }

  .messages-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .search-bar {
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
  }

  .search-button {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .search-button:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
    color: var(--text-primary);
  }

  .search-button svg {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .search-button span {
    flex: 1;
    text-align: left;
  }

  .search-button kbd {
    padding: 2px 6px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-family: inherit;
    font-size: 11px;
  }

  .messages-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }
</style>
