<script lang="ts">
  import { onMount } from "svelte";
  import { healthStore, fetchHealth } from "../stores/health";

  let refreshing = false;

  onMount(() => {
    fetchHealth();
  });

  async function refresh() {
    refreshing = true;
    await fetchHealth();
    refreshing = false;
  }
</script>

<div class="health-status">
  <div class="header">
    <h1>System Health</h1>
    <button class="refresh-btn" on:click={refresh} disabled={refreshing}>
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        class:spinning={refreshing}
      >
        <path d="M23 4v6h-6M1 20v-6h6" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
      </svg>
      {refreshing ? "Refreshing..." : "Refresh"}
    </button>
  </div>

  {#if $healthStore.loading && !$healthStore.data}
    <div class="loading">Loading health status...</div>
  {:else if $healthStore.error}
    <div class="error-banner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      <span>{$healthStore.error}</span>
    </div>
  {:else if $healthStore.data}
    <div class="status-banner" class:healthy={$healthStore.data.status === "healthy"} class:degraded={$healthStore.data.status === "degraded"} class:unhealthy={$healthStore.data.status === "unhealthy"}>
      <div class="status-icon">
        {#if $healthStore.data.status === "healthy"}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        {:else if $healthStore.data.status === "degraded"}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        {:else}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        {/if}
      </div>
      <div class="status-text">
        <h2>System is {$healthStore.data.status}</h2>
        <p>
          {#if $healthStore.data.status === "healthy"}
            All systems operational
          {:else if $healthStore.data.status === "degraded"}
            Some features may be limited
          {:else}
            Some services are unavailable
          {/if}
        </p>
      </div>
    </div>

    <div class="metrics">
      <div class="metric-card">
        <h3>Memory</h3>
        <div class="metric-value">
          {$healthStore.data.memory_available_gb.toFixed(1)} GB
          <span class="metric-label">available</span>
        </div>
        <div class="metric-bar">
          <div
            class="metric-fill"
            style="width: {Math.min(100, ($healthStore.data.memory_used_gb / ($healthStore.data.memory_used_gb + $healthStore.data.memory_available_gb)) * 100)}%"
          />
        </div>
        <p class="metric-detail">
          {$healthStore.data.memory_used_gb.toFixed(1)} GB used of{" "}
          {($healthStore.data.memory_used_gb + $healthStore.data.memory_available_gb).toFixed(1)} GB
        </p>
        <p class="metric-mode">Mode: {$healthStore.data.memory_mode}</p>
      </div>

      <div class="metric-card">
        <h3>JARVIS Process</h3>
        <div class="metric-value">
          {$healthStore.data.jarvis_rss_mb.toFixed(0)} MB
          <span class="metric-label">RSS</span>
        </div>
        <p class="metric-detail">
          Virtual: {$healthStore.data.jarvis_vms_mb.toFixed(0)} MB
        </p>
      </div>

      <div class="metric-card">
        <h3>AI Model</h3>
        <div class="metric-value" class:loaded={$healthStore.data.model_loaded}>
          {$healthStore.data.model_loaded ? "Loaded" : "Not Loaded"}
        </div>
        <p class="metric-detail">
          {#if $healthStore.data.model_loaded}
            Ready for inference
          {:else}
            Will load on first request
          {/if}
        </p>
      </div>

      <div class="metric-card">
        <h3>iMessage Access</h3>
        <div class="metric-value" class:connected={$healthStore.data.imessage_access}>
          {$healthStore.data.imessage_access ? "Connected" : "Not Connected"}
        </div>
        <p class="metric-detail">
          {#if $healthStore.data.imessage_access}
            Full Disk Access granted
          {:else}
            Enable in System Settings
          {/if}
        </p>
      </div>
    </div>

    {#if $healthStore.data.details && Object.keys($healthStore.data.details).length > 0}
      <div class="details">
        <h3>Issues</h3>
        <ul>
          {#each Object.entries($healthStore.data.details) as [key, value]}
            <li>
              <strong>{key}:</strong> {value}
            </li>
          {/each}
        </ul>
      </div>
    {/if}
  {/if}
</div>

<style>
  .health-status {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  h1 {
    font-size: 28px;
    font-weight: 600;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s ease;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .refresh-btn svg {
    width: 16px;
    height: 16px;
  }

  .refresh-btn svg.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .loading {
    text-align: center;
    color: var(--text-secondary);
    padding: 48px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
    border-radius: 12px;
    color: var(--error-color);
    margin-bottom: 24px;
  }

  .error-banner svg {
    width: 24px;
    height: 24px;
    flex-shrink: 0;
  }

  .status-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 24px;
  }

  .status-banner.healthy {
    background: rgba(52, 199, 89, 0.1);
    border: 1px solid #34c759;
  }

  .status-banner.degraded {
    background: rgba(255, 159, 10, 0.1);
    border: 1px solid #ff9f0a;
  }

  .status-banner.unhealthy {
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
  }

  .status-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .healthy .status-icon {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .degraded .status-icon {
    background: rgba(255, 159, 10, 0.2);
    color: #ff9f0a;
  }

  .unhealthy .status-icon {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .status-icon svg {
    width: 24px;
    height: 24px;
  }

  .status-text h2 {
    font-size: 18px;
    font-weight: 600;
    text-transform: capitalize;
    margin-bottom: 4px;
  }

  .status-text p {
    font-size: 14px;
    color: var(--text-secondary);
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .metric-card h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 12px;
  }

  .metric-value {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .metric-value.loaded,
  .metric-value.connected {
    color: #34c759;
  }

  .metric-label {
    font-size: 14px;
    font-weight: 400;
    color: var(--text-secondary);
  }

  .metric-bar {
    height: 6px;
    background: var(--bg-active);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .metric-fill {
    height: 100%;
    background: var(--accent-color);
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .metric-detail {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .metric-mode {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .details {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .details h3 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .details ul {
    list-style: none;
  }

  .details li {
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
    font-size: 14px;
  }

  .details li:last-child {
    border-bottom: none;
  }

  .details strong {
    text-transform: capitalize;
  }
</style>
