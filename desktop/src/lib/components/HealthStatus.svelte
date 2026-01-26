<script lang="ts">
  import { onMount } from "svelte";
  import { healthStore, fetchHealth } from "../stores/health";
  import {
    templateAnalyticsStore,
    fetchTemplateAnalytics,
    resetTemplateAnalytics,
    exportTemplateAnalytics,
  } from "../stores/templateAnalytics";

  let refreshing = false;
  let resettingAnalytics = false;
  let exportingAnalytics = false;

  onMount(() => {
    fetchHealth();
    fetchTemplateAnalytics();
  });

  async function refresh() {
    refreshing = true;
    await Promise.all([fetchHealth(), fetchTemplateAnalytics()]);
    refreshing = false;
  }

  async function handleResetAnalytics() {
    if (confirm("Are you sure you want to reset template analytics? This cannot be undone.")) {
      resettingAnalytics = true;
      await resetTemplateAnalytics();
      resettingAnalytics = false;
    }
  }

  async function handleExportAnalytics() {
    exportingAnalytics = true;
    await exportTemplateAnalytics();
    exportingAnalytics = false;
  }

  // Calculate pie chart percentages
  function getPieChartStyle(templatePercent: number): string {
    const modelPercent = 100 - templatePercent;
    return `conic-gradient(#34c759 0% ${templatePercent}%, #ff9f0a ${templatePercent}% 100%)`;
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

    <!-- Template Analytics Section -->
    <div class="template-analytics">
      <div class="analytics-header">
        <h2>Template Analytics</h2>
        <div class="analytics-actions">
          <button
            class="action-btn export-btn"
            on:click={handleExportAnalytics}
            disabled={exportingAnalytics}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            {exportingAnalytics ? "Exporting..." : "Export JSON"}
          </button>
          <button
            class="action-btn reset-btn"
            on:click={handleResetAnalytics}
            disabled={resettingAnalytics}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="1 4 1 10 7 10" />
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
            </svg>
            {resettingAnalytics ? "Resetting..." : "Clear Metrics"}
          </button>
        </div>
      </div>

      {#if $templateAnalyticsStore.loading && !$templateAnalyticsStore.data}
        <div class="loading">Loading template analytics...</div>
      {:else if $templateAnalyticsStore.error}
        <div class="analytics-error">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{$templateAnalyticsStore.error}</span>
        </div>
      {:else if $templateAnalyticsStore.data}
        <!-- Coverage Overview -->
        <div class="analytics-grid">
          <div class="analytics-card coverage-card">
            <h3>Template Coverage</h3>
            <div class="coverage-chart">
              <div
                class="pie-chart"
                style="background: {getPieChartStyle($templateAnalyticsStore.data.summary.hit_rate_percent)}"
              >
                <div class="pie-center">
                  <span class="coverage-percent">{$templateAnalyticsStore.data.summary.hit_rate_percent.toFixed(1)}%</span>
                  <span class="coverage-label">Coverage</span>
                </div>
              </div>
              <div class="pie-legend">
                <div class="legend-item">
                  <span class="legend-color template-color"></span>
                  <span>Template: {$templateAnalyticsStore.data.pie_chart_data.template_responses}</span>
                </div>
                <div class="legend-item">
                  <span class="legend-color model-color"></span>
                  <span>Model: {$templateAnalyticsStore.data.pie_chart_data.model_responses}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="analytics-card stats-card">
            <h3>Statistics</h3>
            <div class="stats-grid">
              <div class="stat-item">
                <span class="stat-value">{$templateAnalyticsStore.data.summary.total_queries}</span>
                <span class="stat-label">Total Queries</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{$templateAnalyticsStore.data.summary.template_hits}</span>
                <span class="stat-label">Template Hits</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{($templateAnalyticsStore.data.summary.cache_hit_rate * 100).toFixed(1)}%</span>
                <span class="stat-label">Cache Hit Rate</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{$templateAnalyticsStore.data.coverage.total_templates}</span>
                <span class="stat-label">Templates</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Top Templates Bar Chart -->
        {#if $templateAnalyticsStore.data.top_templates.length > 0}
          <div class="analytics-card top-templates">
            <h3>Top Matched Templates</h3>
            <div class="bar-chart">
              {#each $templateAnalyticsStore.data.top_templates.slice(0, 10) as template}
                {@const maxCount = $templateAnalyticsStore.data.top_templates[0]?.match_count || 1}
                <div class="bar-item">
                  <div class="bar-label" title={template.template_name}>
                    {template.template_name.replace(/_/g, " ")}
                  </div>
                  <div class="bar-container">
                    <div
                      class="bar-fill"
                      style="width: {(template.match_count / maxCount) * 100}%"
                    ></div>
                    <span class="bar-count">{template.match_count}</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Category Averages -->
        {#if $templateAnalyticsStore.data.category_averages.length > 0}
          <div class="analytics-card category-averages">
            <h3>Similarity by Category</h3>
            <div class="category-list">
              {#each $templateAnalyticsStore.data.category_averages as cat}
                <div class="category-item">
                  <span class="category-name">{cat.category}</span>
                  <div class="similarity-bar-container">
                    <div
                      class="similarity-bar"
                      style="width: {cat.average_similarity * 100}%"
                    ></div>
                  </div>
                  <span class="similarity-value">{(cat.average_similarity * 100).toFixed(0)}%</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Missed Queries -->
        {#if $templateAnalyticsStore.data.missed_queries.length > 0}
          <div class="analytics-card missed-queries">
            <h3>Recent Missed Queries</h3>
            <p class="missed-description">
              Queries below 0.7 threshold - potential template opportunities
            </p>
            <div class="missed-list">
              {#each $templateAnalyticsStore.data.missed_queries.slice(0, 10) as query}
                <div class="missed-item">
                  <div class="missed-info">
                    <span class="query-hash">#{query.query_hash}</span>
                    {#if query.best_template}
                      <span class="best-match">Best: {query.best_template.replace(/_/g, " ")}</span>
                    {/if}
                  </div>
                  <div class="similarity-badge" class:low={query.similarity < 0.5}>
                    {(query.similarity * 100).toFixed(0)}%
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/if}
    </div>
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

  /* Template Analytics Styles */
  .template-analytics {
    margin-top: 32px;
    padding-top: 24px;
    border-top: 1px solid var(--border-color);
  }

  .analytics-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .analytics-header h2 {
    font-size: 20px;
    font-weight: 600;
  }

  .analytics-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .action-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .action-btn svg {
    width: 14px;
    height: 14px;
  }

  .reset-btn:hover:not(:disabled) {
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .export-btn:hover:not(:disabled) {
    border-color: var(--accent-color);
  }

  .analytics-error {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
    border-radius: 8px;
    color: var(--error-color);
  }

  .analytics-error svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .analytics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .analytics-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .analytics-card h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 16px;
  }

  /* Coverage Pie Chart */
  .coverage-card {
    display: flex;
    flex-direction: column;
  }

  .coverage-chart {
    display: flex;
    align-items: center;
    gap: 24px;
  }

  .pie-chart {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .pie-center {
    width: 80px;
    height: 80px;
    background: var(--bg-secondary);
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .coverage-percent {
    font-size: 20px;
    font-weight: 600;
    color: #34c759;
  }

  .coverage-label {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .pie-legend {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .template-color {
    background: #34c759;
  }

  .model-color {
    background: #ff9f0a;
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Bar Chart */
  .top-templates {
    margin-bottom: 16px;
  }

  .bar-chart {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .bar-item {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .bar-label {
    width: 140px;
    font-size: 12px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    text-transform: capitalize;
  }

  .bar-container {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    height: 20px;
    background: var(--bg-active);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }

  .bar-fill {
    height: 100%;
    background: var(--accent-color);
    border-radius: 4px;
    transition: width 0.3s ease;
    min-width: 2px;
  }

  .bar-count {
    position: absolute;
    right: 8px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary);
  }

  /* Category Averages */
  .category-averages {
    margin-bottom: 16px;
  }

  .category-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .category-item {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .category-name {
    width: 80px;
    font-size: 13px;
    text-transform: capitalize;
  }

  .similarity-bar-container {
    flex: 1;
    height: 8px;
    background: var(--bg-active);
    border-radius: 4px;
    overflow: hidden;
  }

  .similarity-bar {
    height: 100%;
    background: linear-gradient(90deg, #ff9f0a, #34c759);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .similarity-value {
    width: 40px;
    font-size: 12px;
    text-align: right;
    color: var(--text-secondary);
  }

  /* Missed Queries */
  .missed-queries {
    margin-bottom: 16px;
  }

  .missed-description {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 12px;
  }

  .missed-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .missed-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: var(--bg-active);
    border-radius: 6px;
  }

  .missed-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .query-hash {
    font-size: 12px;
    font-family: monospace;
    color: var(--text-secondary);
  }

  .best-match {
    font-size: 12px;
    text-transform: capitalize;
  }

  .similarity-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(255, 159, 10, 0.2);
    color: #ff9f0a;
  }

  .similarity-badge.low {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  @media (max-width: 600px) {
    .analytics-grid {
      grid-template-columns: 1fr;
    }

    .analytics-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 12px;
    }

    .coverage-chart {
      flex-direction: column;
      align-items: center;
    }

    .bar-label {
      width: 100px;
    }
  }
</style>
