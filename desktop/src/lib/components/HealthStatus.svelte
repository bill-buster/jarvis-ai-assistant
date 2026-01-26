<script lang="ts">
  import { onMount } from "svelte";
  import {
    healthStatus,
    healthError,
    isApiConnected,
    refreshHealth,
  } from "../stores/health";

  let refreshing = false;

  async function handleRefresh() {
    refreshing = true;
    await refreshHealth();
    refreshing = false;
  }

  function getStatusColor(status: string | undefined): string {
    switch (status) {
      case "healthy":
        return "#30d158";
      case "degraded":
        return "#ff9f0a";
      case "unhealthy":
        return "#ff5f57";
      default:
        return "#8e8e93";
    }
  }

  function getMemoryModeDescription(mode: string): string {
    switch (mode) {
      case "FULL":
        return "Full capabilities enabled";
      case "LITE":
        return "Reduced memory usage mode";
      case "MINIMAL":
        return "Minimal memory footprint";
      default:
        return mode;
    }
  }

  onMount(() => {
    refreshHealth();
  });
</script>

<div class="health-status">
  <div class="health-header">
    <h1>System Health</h1>
    <button
      class="refresh-btn"
      on:click={handleRefresh}
      disabled={refreshing}
    >
      <span class:spinning={refreshing}>↻</span>
      Refresh
    </button>
  </div>

  <div class="health-content">
    {#if !$isApiConnected}
      <div class="disconnected-state">
        <span class="disconnected-icon">⚠️</span>
        <h2>API Disconnected</h2>
        <p>Unable to connect to JARVIS API server</p>
        <button on:click={handleRefresh}>Retry Connection</button>
      </div>
    {:else if $healthError}
      <div class="error-state">
        <span class="error-icon">❌</span>
        <h2>Error</h2>
        <p>{$healthError}</p>
        <button on:click={handleRefresh}>Retry</button>
      </div>
    {:else if $healthStatus}
      <div class="status-overview">
        <div
          class="status-indicator"
          style="--status-color: {getStatusColor($healthStatus.status)}"
        >
          <span class="status-dot"></span>
          <span class="status-text">{$healthStatus.status}</span>
        </div>
      </div>

      <div class="health-grid">
        <div class="health-card">
          <h3>Memory</h3>
          <div class="card-content">
            <div class="metric">
              <span class="metric-label">Available</span>
              <span class="metric-value">
                {$healthStatus.memory_available_gb.toFixed(1)} GB
              </span>
            </div>
            <div class="metric">
              <span class="metric-label">Used</span>
              <span class="metric-value">
                {$healthStatus.memory_used_gb.toFixed(1)} GB
              </span>
            </div>
            <div class="metric">
              <span class="metric-label">Mode</span>
              <span class="metric-value mode">
                {$healthStatus.memory_mode}
              </span>
            </div>
            <p class="metric-description">
              {getMemoryModeDescription($healthStatus.memory_mode)}
            </p>
          </div>
        </div>

        <div class="health-card">
          <h3>JARVIS Process</h3>
          <div class="card-content">
            <div class="metric">
              <span class="metric-label">RSS (RAM)</span>
              <span class="metric-value">
                {$healthStatus.jarvis_rss_mb.toFixed(0)} MB
              </span>
            </div>
            <div class="metric">
              <span class="metric-label">VMS (Virtual)</span>
              <span class="metric-value">
                {$healthStatus.jarvis_vms_mb.toFixed(0)} MB
              </span>
            </div>
          </div>
        </div>

        <div class="health-card">
          <h3>Permissions</h3>
          <div class="card-content">
            <div class="permission-item">
              <span
                class="permission-status"
                class:granted={$healthStatus.imessage_access}
              >
                {$healthStatus.imessage_access ? "✓" : "✗"}
              </span>
              <span>iMessage Access</span>
            </div>
            <div class="permission-item">
              <span
                class="permission-status"
                class:granted={$healthStatus.permissions_ok}
              >
                {$healthStatus.permissions_ok ? "✓" : "✗"}
              </span>
              <span>Full Disk Access</span>
            </div>
          </div>
        </div>

        <div class="health-card">
          <h3>Model Status</h3>
          <div class="card-content">
            <div class="model-status">
              <span
                class="model-indicator"
                class:loaded={$healthStatus.model_loaded}
              ></span>
              <span>
                {$healthStatus.model_loaded ? "Model Loaded" : "Model Not Loaded"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {#if $healthStatus.details && Object.keys($healthStatus.details).length > 0}
        <div class="details-section">
          <h3>Details</h3>
          <div class="details-list">
            {#each Object.entries($healthStatus.details) as [key, value]}
              <div class="detail-item">
                <span class="detail-key">{key}</span>
                <span class="detail-value">{value}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {:else}
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading health status...</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .health-status {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    overflow-y: auto;
  }

  .health-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-color);
  }

  .health-header h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-active);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .refresh-btn span.spinning {
    animation: spin 1s linear infinite;
    display: inline-block;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .health-content {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .disconnected-state,
  .error-state,
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    gap: 12px;
    text-align: center;
  }

  .disconnected-icon,
  .error-icon {
    font-size: 48px;
  }

  .disconnected-state h2,
  .error-state h2 {
    margin: 0;
    font-size: 20px;
    color: var(--text-primary);
  }

  .disconnected-state p,
  .error-state p {
    color: var(--text-secondary);
    margin: 0;
  }

  .disconnected-state button,
  .error-state button {
    margin-top: 12px;
    padding: 8px 20px;
    background: var(--accent-color);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    cursor: pointer;
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .status-overview {
    display: flex;
    justify-content: center;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 32px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
  }

  .status-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--status-color);
  }

  .status-text {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .health-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  .health-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .health-card h3 {
    margin: 0 0 16px 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .card-content {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .metric-label {
    font-size: 14px;
    color: var(--text-secondary);
  }

  .metric-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .metric-value.mode {
    background: var(--bg-hover);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .metric-description {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
  }

  .permission-item {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .permission-status {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--error-color);
    color: white;
    font-size: 12px;
    font-weight: bold;
  }

  .permission-status.granted {
    background: #30d158;
  }

  .model-status {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .model-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--text-secondary);
  }

  .model-indicator.loaded {
    background: #30d158;
  }

  .details-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .details-section h3 {
    margin: 0 0 16px 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .details-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .detail-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .detail-item:last-child {
    border-bottom: none;
  }

  .detail-key {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .detail-value {
    font-size: 13px;
    color: var(--text-primary);
  }
</style>
