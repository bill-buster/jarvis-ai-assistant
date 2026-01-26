<script lang="ts">
  import { onMount } from "svelte";
  import { listen } from "@tauri-apps/api/event";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import ConversationList from "./lib/components/ConversationList.svelte";
  import MessageView from "./lib/components/MessageView.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import HealthStatus from "./lib/components/HealthStatus.svelte";
  import { checkApiConnection } from "./lib/stores/health";
  import { clearSelection } from "./lib/stores/conversations";

  let currentView: "messages" | "dashboard" | "health" = "messages";

  onMount(() => {
    // Check API connection on start
    checkApiConnection();

    // Listen for navigation events from tray menu
    let unlisten: (() => void) | undefined;

    listen<string>("navigate", (event) => {
      if (
        event.payload === "health" ||
        event.payload === "dashboard" ||
        event.payload === "messages"
      ) {
        currentView = event.payload;
        if (event.payload !== "messages") {
          clearSelection();
        }
      }
    }).then((fn) => {
      unlisten = fn;
    });

    // Cleanup on unmount
    return () => {
      if (unlisten) unlisten();
    };
  });
</script>

<main class="app">
  <Sidebar bind:currentView />

  {#if currentView === "dashboard"}
    <Dashboard on:navigate={(e) => currentView = e.detail as typeof currentView} />
  {:else if currentView === "health"}
    <HealthStatus />
  {:else}
    <ConversationList />
    <MessageView />
  {/if}
</main>

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
</style>
