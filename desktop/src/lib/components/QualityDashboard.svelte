<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    qualityStore,
    fetchQualityDashboard,
    type QualityState,
  } from "../stores/quality";
  import type { QualityDashboardData, Recommendation } from "../api/types";

  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    fetchQualityDashboard();
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(fetchQualityDashboard, 30000);
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  function formatPercent(value: number | null | undefined): string {
    if (value === null || value === undefined) return "N/A";
    return `${value.toFixed(1)}%`;
  }

  function formatNumber(value: number | null | undefined, decimals: number = 0): string {
    if (value === null || value === undefined) return "N/A";
    return value.toFixed(decimals);
  }

  function formatLatency(ms: number | null | undefined): string {
    if (ms === null || ms === undefined) return "N/A";
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function getPriorityColor(priority: string): string {
    switch (priority) {
      case "high":
        return "var(--error-color)";
      case "medium":
        return "var(--warning-color, #ff9500)";
      case "low":
        return "#34c759";
      default:
        return "var(--text-secondary)";
    }
  }

  function getScoreColor(score: number | null | undefined): string {
    if (score === null || score === undefined) return "var(--text-secondary)";
    if (score >= 0.7) return "#34c759";
    if (score >= 0.5) return "var(--warning-color, #ff9500)";
    return "var(--error-color)";
  }

  $: data = $qualityStore.data;
  $: loading = $qualityStore.loading;
  $: error = $qualityStore.error;
</script>

<div class="quality-dashboard">
  <div class="header">
    <h1>Quality Metrics</h1>
    <button class="refresh-btn" on:click={fetchQualityDashboard} disabled={loading}>
      {#if loading}
        <span class="spinner"></span>
      {:else}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
      {/if}
      Refresh
    </button>
  </div>

  {#if error}
    <div class="error-banner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      {error}
    </div>
  {/if}

  {#if data}
    <!-- Key Metrics Cards -->
    <div class="metrics-cards">
      <div class="metric-card">
        <div class="metric-icon template">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
          </svg>
        </div>
        <div class="metric-content">
          <span class="metric-label">Template Hit Rate</span>
          <span class="metric-value">{formatPercent(data.summary.template_hit_rate_percent)}</span>
          <span class="metric-sub">{data.summary.template_responses} / {data.summary.total_responses} responses</span>
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-icon acceptance">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        </div>
        <div class="metric-content">
          <span class="metric-label">Acceptance Rate</span>
          <span class="metric-value">{formatPercent(data.summary.acceptance_rate_percent)}</span>
          <span class="metric-sub">
            {data.summary.accepted_unchanged_count + data.summary.accepted_modified_count} accepted,
            {data.summary.rejected_count} rejected
          </span>
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-icon hhem" style="color: {getScoreColor(data.summary.avg_hhem_score)}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <div class="metric-content">
          <span class="metric-label">HHEM Score</span>
          <span class="metric-value" style="color: {getScoreColor(data.summary.avg_hhem_score)}">
            {data.summary.avg_hhem_score !== null ? formatNumber(data.summary.avg_hhem_score, 3) : "N/A"}
          </span>
          <span class="metric-sub">{data.summary.hhem_score_count} model responses scored</span>
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-icon latency">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
        </div>
        <div class="metric-content">
          <span class="metric-label">Avg Latency</span>
          <span class="metric-value">
            {formatLatency(data.summary.avg_template_latency_ms)} / {formatLatency(data.summary.avg_model_latency_ms)}
          </span>
          <span class="metric-sub">Template / Model</span>
        </div>
      </div>
    </div>

    <div class="dashboard-grid">
      <!-- Recommendations Panel -->
      <div class="panel recommendations-panel">
        <h2>Recommendations</h2>
        <div class="recommendations-list">
          {#each data.recommendations as rec}
            <div class="recommendation-item" style="border-left-color: {getPriorityColor(rec.priority)}">
              <div class="recommendation-header">
                <span class="priority-badge" style="background: {getPriorityColor(rec.priority)}">
                  {rec.priority}
                </span>
                <span class="recommendation-title">{rec.title}</span>
              </div>
              <p class="recommendation-desc">{rec.description}</p>
              {#if rec.metric_value !== null && rec.target_value !== null}
                <div class="recommendation-progress">
                  <span class="progress-label">
                    Current: {formatNumber(rec.metric_value, 1)} / Target: {formatNumber(rec.target_value, 1)}
                  </span>
                  <div class="progress-bar">
                    <div
                      class="progress-fill"
                      style="width: {Math.min((rec.metric_value / rec.target_value) * 100, 100)}%;
                             background: {getPriorityColor(rec.priority)}"
                    ></div>
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Time of Day Chart -->
      <div class="panel time-panel">
        <h2>Activity by Hour</h2>
        <div class="time-chart">
          {#each data.time_of_day as hourData}
            <div class="hour-bar" title="{hourData.hour}:00 - {hourData.total_responses} responses">
              <div
                class="bar-fill"
                style="height: {hourData.total_responses > 0 ? Math.max((hourData.total_responses / Math.max(...data.time_of_day.map(h => h.total_responses))) * 100, 5) : 0}%"
              ></div>
              <span class="hour-label">{hourData.hour}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Intent Breakdown -->
      <div class="panel intent-panel">
        <h2>By Intent</h2>
        <div class="intent-list">
          {#each data.by_intent as intentData}
            <div class="intent-item">
              <div class="intent-header">
                <span class="intent-name">{intentData.intent}</span>
                <span class="intent-count">{intentData.total_responses} responses</span>
              </div>
              <div class="intent-metrics">
                <span class="intent-metric">
                  <span class="metric-icon-small template"></span>
                  {formatPercent(intentData.template_hit_rate)}
                </span>
                <span class="intent-metric">
                  <span class="metric-icon-small acceptance"></span>
                  {formatPercent(intentData.acceptance_rate)}
                </span>
                <span class="intent-metric">
                  <span class="metric-icon-small latency"></span>
                  {formatLatency(intentData.avg_latency_ms)}
                </span>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Top Contacts -->
      <div class="panel contacts-panel">
        <h2>Top Contacts</h2>
        <div class="contacts-list">
          {#if data.top_contacts.length === 0}
            <p class="empty-state">No contact data yet</p>
          {:else}
            {#each data.top_contacts as contact}
              <div class="contact-item">
                <div class="contact-avatar">
                  {contact.contact_id.charAt(0).toUpperCase()}
                </div>
                <div class="contact-info">
                  <span class="contact-id">{contact.contact_id}</span>
                  <span class="contact-stats">
                    {contact.total_responses} responses |
                    {formatPercent(contact.acceptance_rate)} accepted
                  </span>
                </div>
                <div class="contact-latency">
                  {formatLatency(contact.avg_latency_ms)}
                </div>
              </div>
            {/each}
          {/if}
        </div>
      </div>

      <!-- Conversation Type Comparison -->
      <div class="panel conv-type-panel">
        <h2>Conversation Types</h2>
        <div class="conv-type-comparison">
          <div class="conv-type-card">
            <h3>1:1 Chats</h3>
            <div class="conv-type-metrics">
              <div class="conv-metric">
                <span class="conv-metric-label">Responses</span>
                <span class="conv-metric-value">{data.by_conversation_type["1:1"]?.total_responses ?? 0}</span>
              </div>
              <div class="conv-metric">
                <span class="conv-metric-label">Template Rate</span>
                <span class="conv-metric-value">{formatPercent(data.by_conversation_type["1:1"]?.template_hit_rate_percent)}</span>
              </div>
              <div class="conv-metric">
                <span class="conv-metric-label">Acceptance</span>
                <span class="conv-metric-value">{formatPercent(data.by_conversation_type["1:1"]?.acceptance_rate_percent)}</span>
              </div>
            </div>
          </div>
          <div class="conv-type-card">
            <h3>Group Chats</h3>
            <div class="conv-type-metrics">
              <div class="conv-metric">
                <span class="conv-metric-label">Responses</span>
                <span class="conv-metric-value">{data.by_conversation_type["group"]?.total_responses ?? 0}</span>
              </div>
              <div class="conv-metric">
                <span class="conv-metric-label">Template Rate</span>
                <span class="conv-metric-value">{formatPercent(data.by_conversation_type["group"]?.template_hit_rate_percent)}</span>
              </div>
              <div class="conv-metric">
                <span class="conv-metric-label">Acceptance</span>
                <span class="conv-metric-value">{formatPercent(data.by_conversation_type["group"]?.acceptance_rate_percent)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Trends Chart (simplified) -->
      <div class="panel trends-panel">
        <h2>7-Day Trend</h2>
        {#if data.trends.length === 0}
          <p class="empty-state">No trend data available yet</p>
        {:else}
          <div class="trends-summary">
            <div class="trend-item">
              <span class="trend-label">Latest Template Rate</span>
              <span class="trend-value">
                {formatPercent(data.trends[data.trends.length - 1]?.template_hit_rate_percent)}
              </span>
            </div>
            <div class="trend-item">
              <span class="trend-label">Latest Acceptance Rate</span>
              <span class="trend-value">
                {formatPercent(data.trends[data.trends.length - 1]?.acceptance_rate_percent)}
              </span>
            </div>
            <div class="trend-item">
              <span class="trend-label">Data Points</span>
              <span class="trend-value">{data.trends.length} days</span>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {:else if !loading}
    <div class="empty-dashboard">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
      <p>No quality metrics data available</p>
      <span class="empty-hint">Quality metrics will appear as you use JARVIS to generate responses</span>
    </div>
  {/if}
</div>

<style>
  .quality-dashboard {
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
    margin: 0;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
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

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
    border-radius: 8px;
    margin-bottom: 24px;
    color: var(--error-color);
  }

  .error-banner svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .metrics-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    gap: 16px;
  }

  .metric-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .metric-icon svg {
    width: 24px;
    height: 24px;
  }

  .metric-icon.template {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .metric-icon.acceptance {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .metric-icon.hhem {
    background: rgba(88, 86, 214, 0.2);
  }

  .metric-icon.latency {
    background: rgba(255, 149, 0, 0.2);
    color: #ff9500;
  }

  .metric-content {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-label {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .metric-value {
    font-size: 24px;
    font-weight: 600;
  }

  .metric-sub {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 16px;
  }

  .panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .panel h2 {
    font-size: 16px;
    font-weight: 600;
    margin: 0 0 16px 0;
  }

  /* Recommendations */
  .recommendations-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .recommendation-item {
    padding: 12px;
    background: var(--bg-active);
    border-radius: 8px;
    border-left: 4px solid;
  }

  .recommendation-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .priority-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 4px;
    color: white;
  }

  .recommendation-title {
    font-weight: 500;
    font-size: 14px;
  }

  .recommendation-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
  }

  .recommendation-progress {
    margin-top: 8px;
  }

  .progress-label {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .progress-bar {
    height: 4px;
    background: var(--border-color);
    border-radius: 2px;
    margin-top: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  /* Time Chart */
  .time-chart {
    display: flex;
    gap: 2px;
    height: 120px;
    align-items: flex-end;
  }

  .hour-bar {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
  }

  .bar-fill {
    width: 100%;
    background: var(--accent-color);
    border-radius: 2px 2px 0 0;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  .hour-label {
    font-size: 9px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  /* Intent List */
  .intent-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .intent-item {
    padding: 12px;
    background: var(--bg-active);
    border-radius: 8px;
  }

  .intent-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .intent-name {
    font-weight: 500;
    text-transform: capitalize;
  }

  .intent-count {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .intent-metrics {
    display: flex;
    gap: 16px;
  }

  .intent-metric {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 13px;
  }

  .metric-icon-small {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .metric-icon-small.template {
    background: var(--accent-color);
  }

  .metric-icon-small.acceptance {
    background: #34c759;
  }

  .metric-icon-small.latency {
    background: #ff9500;
  }

  /* Contacts List */
  .contacts-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .contact-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
    background: var(--bg-active);
    border-radius: 8px;
  }

  .contact-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
    color: white;
  }

  .contact-info {
    flex: 1;
    min-width: 0;
  }

  .contact-id {
    display: block;
    font-weight: 500;
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .contact-stats {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .contact-latency {
    font-size: 13px;
    color: var(--text-secondary);
  }

  /* Conversation Types */
  .conv-type-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .conv-type-card {
    padding: 12px;
    background: var(--bg-active);
    border-radius: 8px;
  }

  .conv-type-card h3 {
    font-size: 14px;
    font-weight: 500;
    margin: 0 0 12px 0;
  }

  .conv-type-metrics {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .conv-metric {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }

  .conv-metric-label {
    color: var(--text-secondary);
  }

  .conv-metric-value {
    font-weight: 500;
  }

  /* Trends */
  .trends-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }

  .trend-item {
    text-align: center;
    padding: 12px;
    background: var(--bg-active);
    border-radius: 8px;
  }

  .trend-label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .trend-value {
    font-size: 18px;
    font-weight: 600;
  }

  /* Empty States */
  .empty-state {
    text-align: center;
    color: var(--text-secondary);
    padding: 24px;
    font-size: 14px;
  }

  .empty-dashboard {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    color: var(--text-secondary);
  }

  .empty-dashboard svg {
    width: 64px;
    height: 64px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-dashboard p {
    font-size: 18px;
    margin: 0 0 8px 0;
  }

  .empty-hint {
    font-size: 14px;
  }
</style>
