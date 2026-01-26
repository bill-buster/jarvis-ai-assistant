<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { getConversationsStore } from "../stores/conversations";
  import { getHealthStore } from "../stores/health";

  const dispatch = createEventDispatcher<{ navigate: "messages" | "health" }>();

  const conversationsStore = getConversationsStore();
  const healthStore = getHealthStore();
</script>

<main class="dashboard">
  <header class="dashboard-header">
    <h1>Dashboard</h1>
    <p class="subtitle">JARVIS AI Assistant Overview</p>
  </header>

  <div class="stats-grid">
    <button class="stat-card" onclick={() => dispatch("navigate", "messages")}>
      <div class="stat-icon">💬</div>
      <div class="stat-content">
        <div class="stat-value">{conversationsStore.conversations.length}</div>
        <div class="stat-label">Conversations</div>
      </div>
    </button>

    <button class="stat-card" onclick={() => dispatch("navigate", "health")}>
      <div class="stat-icon">
        {#if healthStore.health?.status === "healthy"}
          ✅
        {:else if healthStore.health?.status === "degraded"}
          ⚠️
        {:else}
          ❌
        {/if}
      </div>
      <div class="stat-content">
        <div class="stat-value">
          {healthStore.health?.status || "Unknown"}
        </div>
        <div class="stat-label">System Health</div>
      </div>
    </button>

    <div class="stat-card">
      <div class="stat-icon">🧠</div>
      <div class="stat-content">
        <div class="stat-value">
          {healthStore.health?.memory_mode || "N/A"}
        </div>
        <div class="stat-label">Memory Mode</div>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon">📱</div>
      <div class="stat-content">
        <div class="stat-value">
          {healthStore.health?.imessage_access ? "Connected" : "Disconnected"}
        </div>
        <div class="stat-label">iMessage</div>
      </div>
    </div>
  </div>

  <section class="quick-actions">
    <h2>Quick Actions</h2>
    <div class="action-buttons">
      <button class="action-btn" onclick={() => dispatch("navigate", "messages")}>
        <span class="action-icon">✉️</span>
        <span>View Messages</span>
      </button>
      <button class="action-btn" onclick={() => dispatch("navigate", "health")}>
        <span class="action-icon">📊</span>
        <span>System Health</span>
      </button>
    </div>
  </section>

  {#if healthStore.health}
    <section class="memory-info">
      <h2>Memory Usage</h2>
      <div class="memory-bar-container">
        <div
          class="memory-bar"
          style="width: {Math.min(
            100,
            (healthStore.health.memory_used_gb /
              (healthStore.health.memory_used_gb + healthStore.health.memory_available_gb)) *
              100
          )}%"
        ></div>
      </div>
      <div class="memory-stats">
        <span>Used: {healthStore.health.memory_used_gb.toFixed(1)} GB</span>
        <span>Available: {healthStore.health.memory_available_gb.toFixed(1)} GB</span>
      </div>
    </section>
  {/if}
</main>

<style>
  .dashboard {
    flex: 1;
    padding: 32px;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  .dashboard-header {
    margin-bottom: 32px;
  }

  .dashboard-header h1 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .subtitle {
    color: var(--text-secondary);
    font-size: 14px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }

  .stat-card:hover {
    background: var(--bg-hover);
    transform: translateY(-2px);
  }

  .stat-icon {
    font-size: 32px;
  }

  .stat-value {
    font-size: 20px;
    font-weight: 600;
    text-transform: capitalize;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  .quick-actions {
    margin-bottom: 32px;
  }

  .quick-actions h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .action-buttons {
    display: flex;
    gap: 12px;
  }

  .action-btn {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    color: var(--text-primary);
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--bg-hover);
  }

  .action-icon {
    font-size: 18px;
  }

  .memory-info h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .memory-bar-container {
    height: 8px;
    background: var(--bg-secondary);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .memory-bar {
    height: 100%;
    background: var(--accent-color);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .memory-stats {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
  }
</style>
