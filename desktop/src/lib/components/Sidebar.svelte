<script lang="ts">
  interface Props {
    currentView: "messages" | "dashboard" | "health";
  }

  let { currentView = $bindable() }: Props = $props();
</script>

<nav class="sidebar">
  <div class="logo">
    <span class="logo-icon">J</span>
    <span class="logo-text">JARVIS</span>
  </div>

  <ul class="nav-list">
    <li>
      <button
        class="nav-item"
        class:active={currentView === "messages"}
        onclick={() => (currentView = "messages")}
      >
        <span class="nav-icon">💬</span>
        <span class="nav-label">Messages</span>
      </button>
    </li>
    <li>
      <button
        class="nav-item"
        class:active={currentView === "dashboard"}
        onclick={() => (currentView = "dashboard")}
      >
        <span class="nav-icon">📊</span>
        <span class="nav-label">Dashboard</span>
      </button>
    </li>
    <li>
      <button
        class="nav-item"
        class:active={currentView === "health"}
        onclick={() => (currentView = "health")}
      >
        <span class="nav-icon">❤️</span>
        <span class="nav-label">Health</span>
      </button>
    </li>
  </ul>
</nav>

<style>
  .sidebar {
    width: 64px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 0;
    flex-shrink: 0;
  }

  .logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 24px;
  }

  .logo-icon {
    font-size: 24px;
    font-weight: bold;
    color: var(--accent-color);
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-hover);
    border-radius: 10px;
  }

  .logo-text {
    font-size: 8px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: 4px;
    letter-spacing: 1px;
  }

  .nav-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
    padding: 0 8px;
  }

  .nav-item {
    width: 100%;
    padding: 10px 0;
    background: none;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    transition: background-color 0.15s ease;
  }

  .nav-item:hover {
    background: var(--bg-hover);
  }

  .nav-item.active {
    background: var(--bg-active);
  }

  .nav-icon {
    font-size: 20px;
  }

  .nav-label {
    font-size: 9px;
    color: var(--text-secondary);
    font-weight: 500;
  }

  .nav-item.active .nav-label {
    color: var(--text-primary);
  }
</style>
