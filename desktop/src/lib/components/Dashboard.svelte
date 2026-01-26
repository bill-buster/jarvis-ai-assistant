<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { conversations } from "../stores/conversations";
  import { healthStatus, isApiConnected } from "../stores/health";

  const dispatch = createEventDispatcher<{ navigate: string }>();

  $: totalConversations = $conversations.length;
  $: groupChats = $conversations.filter((c) => c.is_group).length;
  $: directChats = totalConversations - groupChats;
</script>

<div class="dashboard">
  <div class="dashboard-header">
    <h1>Dashboard</h1>
  </div>

  <div class="dashboard-content">
    <div class="stats-grid">
      <button
        class="stat-card clickable"
        on:click={() => dispatch("navigate", "messages")}
      >
        <span class="stat-icon">💬</span>
        <div class="stat-info">
          <span class="stat-value">{totalConversations}</span>
          <span class="stat-label">Conversations</span>
        </div>
      </button>

      <div class="stat-card">
        <span class="stat-icon">👤</span>
        <div class="stat-info">
          <span class="stat-value">{directChats}</span>
          <span class="stat-label">Direct Chats</span>
        </div>
      </div>

      <div class="stat-card">
        <span class="stat-icon">👥</span>
        <div class="stat-info">
          <span class="stat-value">{groupChats}</span>
          <span class="stat-label">Group Chats</span>
        </div>
      </div>

      <button
        class="stat-card clickable"
        class:healthy={$healthStatus?.status === "healthy"}
        class:degraded={$healthStatus?.status === "degraded"}
        class:unhealthy={$healthStatus?.status === "unhealthy"}
        on:click={() => dispatch("navigate", "health")}
      >
        <span class="stat-icon">❤️</span>
        <div class="stat-info">
          <span class="stat-value">
            {$isApiConnected ? $healthStatus?.status || "Unknown" : "Offline"}
          </span>
          <span class="stat-label">System Health</span>
        </div>
      </button>
    </div>

    {#if $healthStatus}
      <div class="info-section">
        <h2>System Info</h2>
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">Memory Mode</span>
            <span class="info-value">{$healthStatus.memory_mode}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Available Memory</span>
            <span class="info-value">
              {$healthStatus.memory_available_gb.toFixed(1)} GB
            </span>
          </div>
          <div class="info-item">
            <span class="info-label">Model Loaded</span>
            <span class="info-value">
              {$healthStatus.model_loaded ? "Yes" : "No"}
            </span>
          </div>
          <div class="info-item">
            <span class="info-label">iMessage Access</span>
            <span class="info-value">
              {$healthStatus.imessage_access ? "Granted" : "Denied"}
            </span>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    overflow-y: auto;
  }

  .dashboard-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-color);
  }

  .dashboard-header h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .dashboard-content {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 32px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    text-align: left;
  }

  .stat-card.clickable {
    cursor: pointer;
    transition: all 0.15s;
  }

  .stat-card.clickable:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .stat-card.healthy {
    border-color: #30d158;
  }

  .stat-card.degraded {
    border-color: #ff9f0a;
  }

  .stat-card.unhealthy {
    border-color: var(--error-color);
  }

  .stat-icon {
    font-size: 32px;
  }

  .stat-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .stat-label {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .info-section h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }

  .info-item {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .info-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .info-value {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-primary);
  }
</style>
