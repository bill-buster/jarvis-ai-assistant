<script lang="ts">
  import { onMount } from "svelte";
  import { fetchHealth, getHealthStore } from "../stores/health";

  const store = getHealthStore();

  onMount(() => {
    fetchHealth();
  });

  function getStatusColor(status: string | undefined): string {
    switch (status) {
      case "healthy":
        return "#34c759";
      case "degraded":
        return "#ff9f0a";
      case "unhealthy":
        return "#ff3b30";
      default:
        return "#8e8e93";
    }
  }

  function getStatusIcon(status: string | undefined): string {
    switch (status) {
      case "healthy":
        return "✅";
      case "degraded":
        return "⚠️";
      case "unhealthy":
        return "❌";
      default:
        return "❓";
    }
  }
</script>

<main class="health-status">
  <header class="health-header">
    <h1>System Health</h1>
    <button class="refresh-btn" onclick={() => fetchHealth()}>
      {#if store.loading}
        ⏳
      {:else}
        ↻
      {/if}
    </button>
  </header>

  {#if store.loading && !store.health}
    <div class="loading">Loading health status...</div>
  {:else if store.error && !store.health}
    <div class="error">
      <span class="error-icon">❌</span>
      <span>{store.error}</span>
      <button class="retry-btn" onclick={() => fetchHealth()}>Retry</button>
    </div>
  {:else if store.health}
    <div class="status-overview">
      <div
        class="status-badge"
        style="--status-color: {getStatusColor(store.health.status)}"
      >
        <span class="status-icon">{getStatusIcon(store.health.status)}</span>
        <span class="status-text">{store.health.status}</span>
      </div>
    </div>

    <div class="health-grid">
      <div class="health-card">
        <div class="card-header">
          <span class="card-icon">📱</span>
          <h3>iMessage Access</h3>
        </div>
        <div class="card-value" class:success={store.health.imessage_access}>
          {store.health.imessage_access ? "Connected" : "Not Connected"}
        </div>
        {#if !store.health.imessage_access}
          <div class="card-hint">
            Grant Full Disk Access in System Settings
          </div>
        {/if}
      </div>

      <div class="health-card">
        <div class="card-header">
          <span class="card-icon">🧠</span>
          <h3>Memory Mode</h3>
        </div>
        <div class="card-value">{store.health.memory_mode}</div>
        <div class="card-hint">
          {store.health.memory_mode === "FULL"
            ? "Full model capabilities"
            : store.health.memory_mode === "LITE"
              ? "Reduced memory usage"
              : "Minimal operation mode"}
        </div>
      </div>

      <div class="health-card">
        <div class="card-header">
          <span class="card-icon">🤖</span>
          <h3>AI Model</h3>
        </div>
        <div class="card-value" class:success={store.health.model_loaded}>
          {store.health.model_loaded ? "Loaded" : "Not Loaded"}
        </div>
        <div class="card-hint">
          {store.health.model_loaded
            ? "Ready for generation"
            : "Will load on first use"}
        </div>
      </div>

      <div class="health-card">
        <div class="card-header">
          <span class="card-icon">📊</span>
          <h3>System Memory</h3>
        </div>
        <div class="memory-details">
          <div class="memory-bar-container">
            <div
              class="memory-bar"
              style="width: {Math.min(
                100,
                (store.health.memory_used_gb /
                  (store.health.memory_used_gb + store.health.memory_available_gb)) *
                  100
              )}%"
            ></div>
          </div>
          <div class="memory-values">
            <span>Used: {store.health.memory_used_gb.toFixed(1)} GB</span>
            <span>Available: {store.health.memory_available_gb.toFixed(1)} GB</span>
          </div>
        </div>
      </div>

      <div class="health-card">
        <div class="card-header">
          <span class="card-icon">⚡</span>
          <h3>JARVIS Process</h3>
        </div>
        <div class="process-details">
          <div class="process-stat">
            <span class="stat-label">RAM (RSS)</span>
            <span class="stat-value">{store.health.jarvis_rss_mb.toFixed(0)} MB</span>
          </div>
          <div class="process-stat">
            <span class="stat-label">Virtual</span>
            <span class="stat-value">{store.health.jarvis_vms_mb.toFixed(0)} MB</span>
          </div>
        </div>
      </div>
    </div>

    {#if store.health.details && Object.keys(store.health.details).length > 0}
      <section class="details-section">
        <h2>Details</h2>
        <ul class="details-list">
          {#each Object.entries(store.health.details) as [key, value]}
            <li>
              <span class="detail-key">{key}:</span>
              <span class="detail-value">{value}</span>
            </li>
          {/each}
        </ul>
      </section>
    {/if}
  {:else}
    <div class="empty">No health data available</div>
  {/if}
</main>

<style>
  .health-status {
    flex: 1;
    padding: 32px;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  .health-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  .health-header h1 {
    font-size: 28px;
    font-weight: 600;
  }

  .refresh-btn {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 18px;
    color: var(--text-primary);
    transition: all 0.15s ease;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
  }

  .loading,
  .empty {
    text-align: center;
    padding: 40px;
    color: var(--text-secondary);
  }

  .error {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--error-color);
    border-radius: 8px;
    color: var(--error-color);
  }

  .retry-btn {
    margin-left: auto;
    background: var(--error-color);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
  }

  .status-overview {
    display: flex;
    justify-content: center;
    margin-bottom: 32px;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 32px;
    background: var(--bg-secondary);
    border: 2px solid var(--status-color);
    border-radius: 16px;
  }

  .status-icon {
    font-size: 32px;
  }

  .status-text {
    font-size: 24px;
    font-weight: 600;
    text-transform: capitalize;
    color: var(--status-color);
  }

  .health-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }

  .health-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .card-icon {
    font-size: 20px;
  }

  .card-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .card-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .card-value.success {
    color: #34c759;
  }

  .card-hint {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 8px;
  }

  .memory-details {
    margin-top: 8px;
  }

  .memory-bar-container {
    height: 8px;
    background: var(--bg-hover);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .memory-bar {
    height: 100%;
    background: var(--accent-color);
    border-radius: 4px;
  }

  .memory-values {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .process-details {
    display: flex;
    gap: 24px;
  }

  .process-stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
  }

  .details-section h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .details-list {
    list-style: none;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
  }

  .details-list li {
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .details-list li:last-child {
    border-bottom: none;
  }

  .detail-key {
    font-weight: 500;
    margin-right: 8px;
  }

  .detail-value {
    color: var(--text-secondary);
  }
</style>
