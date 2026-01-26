<script lang="ts">
  import { healthStore } from "../stores/health";

  export let currentView: "messages" | "dashboard" | "health" | "settings" =
    "messages";

  function navigate(view: "messages" | "dashboard" | "health" | "settings") {
    currentView = view;
  }
</script>

<aside class="sidebar">
  <div class="logo">
    <span class="logo-icon">J</span>
    <span class="logo-text">JARVIS</span>
  </div>

  <nav class="nav">
    <button
      class="nav-item"
      class:active={currentView === "dashboard"}
      on:click={() => navigate("dashboard")}
      title="Dashboard"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
      <span>Dashboard</span>
    </button>

    <button
      class="nav-item"
      class:active={currentView === "messages"}
      on:click={() => navigate("messages")}
      title="Messages"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path
          d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
        />
      </svg>
      <span>Messages</span>
    </button>

    <button
      class="nav-item"
      class:active={currentView === "health"}
      on:click={() => navigate("health")}
      title="Health Status"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path
          d="M22 12h-4l-3 9L9 3l-3 9H2"
        />
      </svg>
      <span>Health</span>
    </button>

    <button
      class="nav-item"
      class:active={currentView === "settings"}
      on:click={() => navigate("settings")}
      title="Settings"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3" />
        <path
          d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
        />
      </svg>
      <span>Settings</span>
    </button>
  </nav>

  <div class="status">
    {#if $healthStore.connected}
      <span class="status-dot connected" />
      <span class="status-text">Connected</span>
    {:else}
      <span class="status-dot disconnected" />
      <span class="status-text">Disconnected</span>
    {/if}
  </div>
</aside>

<style>
  .sidebar {
    width: 200px;
    min-width: 200px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 16px 0;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 16px;
    margin-bottom: 24px;
  }

  .logo-icon {
    width: 32px;
    height: 32px;
    background: var(--accent-color);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 18px;
  }

  .logo-text {
    font-weight: 600;
    font-size: 18px;
  }

  .nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0 8px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s ease;
    text-align: left;
  }

  .nav-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .nav-item.active {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .nav-item svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    margin-top: auto;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .status-dot.connected {
    background: #34c759;
  }

  .status-dot.disconnected {
    background: var(--error-color);
  }

  .status-text {
    font-size: 12px;
    color: var(--text-secondary);
  }
</style>
