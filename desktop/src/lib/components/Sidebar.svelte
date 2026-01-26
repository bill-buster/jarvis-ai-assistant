<script lang="ts">
  import { isApiConnected } from "../stores/health";

  export let currentView: "messages" | "dashboard" | "health" = "messages";

  const navItems = [
    { id: "messages" as const, icon: "💬", label: "Messages" },
    { id: "dashboard" as const, icon: "📊", label: "Dashboard" },
    { id: "health" as const, icon: "❤️", label: "Health" },
  ];
</script>

<nav class="sidebar">
  <div class="logo">
    <span class="logo-icon">🤖</span>
    <span class="logo-text">JARVIS</span>
  </div>

  <div class="nav-items">
    {#each navItems as item}
      <button
        class="nav-item"
        class:active={currentView === item.id}
        on:click={() => (currentView = item.id)}
      >
        <span class="nav-icon">{item.icon}</span>
        <span class="nav-label">{item.label}</span>
      </button>
    {/each}
  </div>

  <div class="sidebar-footer">
    <div class="connection-status" class:connected={$isApiConnected}>
      <span class="status-dot"></span>
      <span class="status-text">
        {$isApiConnected ? "Connected" : "Disconnected"}
      </span>
    </div>
  </div>
</nav>

<style>
  .sidebar {
    width: 72px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 0;
  }

  .logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    margin-bottom: 24px;
  }

  .logo-icon {
    font-size: 28px;
  }

  .logo-text {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
  }

  .nav-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
  }

  .nav-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 10px 8px;
    background: transparent;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    width: 56px;
  }

  .nav-item:hover {
    background: var(--bg-hover);
  }

  .nav-item.active {
    background: var(--bg-active);
  }

  .nav-icon {
    font-size: 22px;
  }

  .nav-label {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .nav-item.active .nav-label {
    color: var(--text-primary);
  }

  .sidebar-footer {
    margin-top: auto;
    padding-top: 16px;
  }

  .connection-status {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--error-color);
  }

  .connection-status.connected .status-dot {
    background: #30d158;
  }

  .status-text {
    font-size: 9px;
    color: var(--text-secondary);
  }
</style>
