<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../api/client";
  import type { ConversationInsights, TimeRange } from "../api/types";

  export let chatId: string;
  export let onClose: () => void;

  let loading = true;
  let error: string | null = null;
  let insights: ConversationInsights | null = null;
  let selectedTimeRange: TimeRange = "month";

  const timeRangeOptions: { value: TimeRange; label: string }[] = [
    { value: "week", label: "Last Week" },
    { value: "month", label: "Last Month" },
    { value: "three_months", label: "Last 3 Months" },
    { value: "all_time", label: "All Time" },
  ];

  async function fetchInsights() {
    loading = true;
    error = null;
    try {
      insights = await api.getConversationInsights(chatId, selectedTimeRange, 500);
    } catch (e) {
      if (e instanceof Error) {
        error = `Failed to load insights: ${e.message}`;
      } else {
        error = "Failed to load insights. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function handleTimeRangeChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedTimeRange = target.value as TimeRange;
    fetchInsights();
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      onClose();
    }
  }

  function formatDateRange(): string {
    if (!insights?.first_message_date || !insights?.last_message_date) return "";
    const start = new Date(insights.first_message_date);
    const end = new Date(insights.last_message_date);
    const options: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
    };
    return `${start.toLocaleDateString("en-US", options)} - ${end.toLocaleDateString("en-US", options)}`;
  }

  function formatResponseTime(minutes: number | null): string {
    if (minutes === null) return "N/A";
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }

  function getSentimentColor(score: number): string {
    if (score >= 0.3) return "#34c759"; // green
    if (score <= -0.3) return "#ff3b30"; // red
    return "#ff9f0a"; // orange/neutral
  }

  function getHealthColor(label: string): string {
    switch (label) {
      case "excellent": return "#34c759";
      case "good": return "#30d158";
      case "fair": return "#ff9f0a";
      case "needs_attention": return "#ff6b35";
      case "concerning": return "#ff3b30";
      default: return "#8e8e93";
    }
  }

  function getTrendIcon(direction: string): string {
    switch (direction) {
      case "increasing": return "^";
      case "decreasing": return "v";
      default: return "-";
    }
  }

  function getTrendColor(direction: string): string {
    switch (direction) {
      case "increasing": return "#34c759";
      case "decreasing": return "#ff3b30";
      default: return "#8e8e93";
    }
  }

  function getHourLabel(hour: number): string {
    if (hour === 0) return "12am";
    if (hour === 12) return "12pm";
    return hour < 12 ? `${hour}am` : `${hour - 12}pm`;
  }

  function getMaxSentimentTrendCount(): number {
    if (!insights?.sentiment_trends) return 1;
    return Math.max(...insights.sentiment_trends.map((t) => t.message_count), 1);
  }

  onMount(() => {
    fetchInsights();
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="modal-overlay" on:click={handleBackdropClick}>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-header">
      <h2 id="modal-title">Conversation Insights</h2>
      <div class="header-controls">
        <select
          class="time-range-select"
          value={selectedTimeRange}
          on:change={handleTimeRangeChange}
          disabled={loading}
        >
          {#each timeRangeOptions as option}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
        <button class="close-btn" on:click={onClose} aria-label="Close">
          &times;
        </button>
      </div>
    </div>

    <div class="modal-content">
      {#if loading}
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p>Analyzing conversation patterns...</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" on:click={fetchInsights}>Try Again</button>
        </div>
      {:else if insights}
        <div class="insights-content">
          <!-- Date Range Info -->
          <div class="date-range-info">
            {insights.total_messages_analyzed} messages analyzed ({formatDateRange()})
          </div>

          <!-- Relationship Health Score - Main Card -->
          <div class="health-card" style="--health-color: {getHealthColor(insights.relationship_health.health_label)}">
            <div class="health-header">
              <h3>Relationship Health</h3>
              <span class="health-badge" style="background: {getHealthColor(insights.relationship_health.health_label)}">
                {insights.relationship_health.health_label.replace("_", " ")}
              </span>
            </div>
            <div class="health-score">
              <div class="score-circle" style="--progress: {insights.relationship_health.overall_score}%">
                <span class="score-value">{Math.round(insights.relationship_health.overall_score)}</span>
                <span class="score-label">/ 100</span>
              </div>
            </div>
            <div class="health-breakdown">
              <div class="health-factor">
                <span class="factor-label">Engagement</span>
                <div class="factor-bar-container">
                  <div class="factor-bar" style="width: {insights.relationship_health.engagement_score}%"></div>
                </div>
                <span class="factor-value">{Math.round(insights.relationship_health.engagement_score)}</span>
              </div>
              <div class="health-factor">
                <span class="factor-label">Sentiment</span>
                <div class="factor-bar-container">
                  <div class="factor-bar" style="width: {insights.relationship_health.sentiment_score}%"></div>
                </div>
                <span class="factor-value">{Math.round(insights.relationship_health.sentiment_score)}</span>
              </div>
              <div class="health-factor">
                <span class="factor-label">Responsiveness</span>
                <div class="factor-bar-container">
                  <div class="factor-bar" style="width: {insights.relationship_health.responsiveness_score}%"></div>
                </div>
                <span class="factor-value">{Math.round(insights.relationship_health.responsiveness_score)}</span>
              </div>
              <div class="health-factor">
                <span class="factor-label">Consistency</span>
                <div class="factor-bar-container">
                  <div class="factor-bar" style="width: {insights.relationship_health.consistency_score}%"></div>
                </div>
                <span class="factor-value">{Math.round(insights.relationship_health.consistency_score)}</span>
              </div>
            </div>
            {#if Object.keys(insights.relationship_health.factors).length > 0}
              <div class="health-factors-list">
                {#each Object.entries(insights.relationship_health.factors) as [key, value]}
                  <div class="factor-item">
                    <span class="factor-key">{key}:</span>
                    <span class="factor-desc">{value}</span>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

          <!-- Sentiment Analysis Section -->
          <div class="section">
            <h3>Sentiment Analysis</h3>
            <div class="sentiment-overview">
              <div class="sentiment-score-card" style="--sentiment-color: {getSentimentColor(insights.sentiment_overall.score)}">
                <div class="sentiment-value">{(insights.sentiment_overall.score * 100).toFixed(0)}%</div>
                <div class="sentiment-label">{insights.sentiment_overall.label}</div>
              </div>
              <div class="sentiment-breakdown">
                <div class="sentiment-stat">
                  <span class="stat-icon positive">+</span>
                  <span class="stat-value">{insights.sentiment_overall.positive_count}</span>
                  <span class="stat-label">Positive signals</span>
                </div>
                <div class="sentiment-stat">
                  <span class="stat-icon negative">-</span>
                  <span class="stat-value">{insights.sentiment_overall.negative_count}</span>
                  <span class="stat-label">Negative signals</span>
                </div>
              </div>
            </div>

            {#if insights.sentiment_trends.length > 0}
              <h4>Sentiment Over Time</h4>
              <div class="sentiment-chart">
                {#each insights.sentiment_trends as trend}
                  <div class="trend-bar-container" title="{trend.date}: {(trend.score * 100).toFixed(0)}% ({trend.message_count} messages)">
                    <div class="trend-bar-bg"></div>
                    <div
                      class="trend-bar"
                      style="height: {Math.abs(trend.score) * 50 + 50}%; background: {getSentimentColor(trend.score)}"
                    ></div>
                    <div class="trend-zero-line"></div>
                  </div>
                {/each}
              </div>
              <div class="trend-labels">
                <span>{insights.sentiment_trends[0]?.date || ""}</span>
                <span>{insights.sentiment_trends[insights.sentiment_trends.length - 1]?.date || ""}</span>
              </div>
            {/if}
          </div>

          <!-- Response Patterns Section -->
          <div class="section">
            <h3>Response Patterns</h3>
            <div class="response-cards">
              <div class="response-card">
                <div class="response-value">{formatResponseTime(insights.response_patterns.avg_response_time_minutes)}</div>
                <div class="response-label">Avg Response Time</div>
              </div>
              <div class="response-card you">
                <div class="response-value">{formatResponseTime(insights.response_patterns.my_avg_response_time_minutes)}</div>
                <div class="response-label">Your Avg Response</div>
              </div>
              <div class="response-card them">
                <div class="response-value">{formatResponseTime(insights.response_patterns.their_avg_response_time_minutes)}</div>
                <div class="response-label">Their Avg Response</div>
              </div>
            </div>

            <div class="response-details">
              <div class="response-detail">
                <span class="detail-label">Fastest:</span>
                <span class="detail-value">{formatResponseTime(insights.response_patterns.fastest_response_minutes)}</span>
              </div>
              <div class="response-detail">
                <span class="detail-label">Slowest:</span>
                <span class="detail-value">{formatResponseTime(insights.response_patterns.slowest_response_minutes)}</span>
              </div>
              <div class="response-detail">
                <span class="detail-label">Median:</span>
                <span class="detail-value">{formatResponseTime(insights.response_patterns.median_response_time_minutes)}</span>
              </div>
            </div>

            {#if Object.keys(insights.response_patterns.response_times_by_day).length > 0}
              <h4>Response Time by Day</h4>
              <div class="day-response-chart">
                {#each Object.entries(insights.response_patterns.response_times_by_day) as [day, time]}
                  <div class="day-response-item">
                    <span class="day-name">{day.slice(0, 3)}</span>
                    <span class="day-time">{formatResponseTime(time)}</span>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

          <!-- Message Frequency Section -->
          <div class="section">
            <h3>Message Frequency</h3>
            <div class="frequency-overview">
              <div class="frequency-card">
                <div class="frequency-value">{insights.frequency_trends.messages_per_day_avg.toFixed(1)}</div>
                <div class="frequency-label">Messages/Day</div>
              </div>
              <div class="frequency-card trend" style="--trend-color: {getTrendColor(insights.frequency_trends.trend_direction)}">
                <div class="frequency-value">
                  <span class="trend-icon">{getTrendIcon(insights.frequency_trends.trend_direction)}</span>
                  {Math.abs(insights.frequency_trends.trend_percentage).toFixed(0)}%
                </div>
                <div class="frequency-label">{insights.frequency_trends.trend_direction}</div>
              </div>
              <div class="frequency-card">
                <div class="frequency-value">{insights.frequency_trends.most_active_day || "N/A"}</div>
                <div class="frequency-label">Most Active Day</div>
              </div>
              <div class="frequency-card">
                <div class="frequency-value">{insights.frequency_trends.most_active_hour !== null ? getHourLabel(insights.frequency_trends.most_active_hour) : "N/A"}</div>
                <div class="frequency-label">Peak Hour</div>
              </div>
            </div>

            {#if Object.keys(insights.frequency_trends.weekly_counts).length > 0}
              <h4>Weekly Activity</h4>
              <div class="weekly-chart">
                {#each Object.entries(insights.frequency_trends.weekly_counts).slice(-12) as [week, count]}
                  {@const maxCount = Math.max(...Object.values(insights.frequency_trends.weekly_counts))}
                  <div class="week-bar-container" title="{week}: {count} messages">
                    <div class="week-bar" style="height: {(count / maxCount) * 100}%"></div>
                  </div>
                {/each}
              </div>
              <div class="chart-labels">
                <span>Older</span>
                <span>Recent</span>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button class="btn secondary" on:click={fetchInsights} disabled={loading}>
        Refresh
      </button>
      <button class="btn primary" on:click={onClose}>
        Close
      </button>
    </div>
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    width: 90%;
    max-width: 750px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 17px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .time-range-select {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    padding: 6px 10px;
    font-size: 13px;
    cursor: pointer;
  }

  .time-range-select:hover:not(:disabled) {
    border-color: var(--accent-color);
  }

  .time-range-select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: background-color 0.15s;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    gap: 16px;
    color: var(--text-secondary);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 16px;
    text-align: center;
  }

  .error-message {
    color: var(--error-color);
    font-size: 14px;
  }

  .retry-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
  }

  .retry-btn:hover {
    background: var(--bg-active);
  }

  .insights-content {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .date-range-info {
    font-size: 13px;
    color: var(--text-secondary);
    text-align: center;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
  }

  /* Health Card */
  .health-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid var(--health-color);
  }

  .health-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .health-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .health-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    color: white;
    text-transform: capitalize;
  }

  .health-score {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
  }

  .score-circle {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: conic-gradient(
      var(--health-color) var(--progress),
      var(--bg-secondary) var(--progress)
    );
    position: relative;
  }

  .score-circle::before {
    content: "";
    position: absolute;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: var(--bg-primary);
  }

  .score-value {
    position: relative;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .score-label {
    position: relative;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .health-breakdown {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 16px;
  }

  .health-factor {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .factor-label {
    width: 100px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .factor-bar-container {
    flex: 1;
    height: 8px;
    background: var(--bg-secondary);
    border-radius: 4px;
    overflow: hidden;
  }

  .factor-bar {
    height: 100%;
    background: var(--accent-color);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .factor-value {
    width: 30px;
    text-align: right;
    font-size: 12px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .health-factors-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
  }

  .factor-item {
    font-size: 12px;
  }

  .factor-key {
    color: var(--text-secondary);
    text-transform: capitalize;
  }

  .factor-desc {
    color: var(--text-primary);
    margin-left: 4px;
  }

  /* Sections */
  .section {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
  }

  .section h3 {
    margin: 0 0 16px 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .section h4 {
    margin: 16px 0 12px 0;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  /* Sentiment */
  .sentiment-overview {
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .sentiment-score-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 24px;
    background: var(--bg-secondary);
    border-radius: 8px;
    border-left: 4px solid var(--sentiment-color);
  }

  .sentiment-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--sentiment-color);
  }

  .sentiment-label {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: capitalize;
  }

  .sentiment-breakdown {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .sentiment-stat {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .stat-icon {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 14px;
  }

  .stat-icon.positive {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .stat-icon.negative {
    background: rgba(255, 59, 48, 0.2);
    color: #ff3b30;
  }

  .sentiment-stat .stat-value {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 40px;
  }

  .sentiment-stat .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Sentiment Chart */
  .sentiment-chart {
    display: flex;
    align-items: center;
    height: 80px;
    gap: 2px;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 8px;
  }

  .trend-bar-container {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  .trend-bar-bg {
    position: absolute;
    width: 100%;
    height: 100%;
    background: var(--bg-primary);
    border-radius: 2px;
  }

  .trend-bar {
    position: absolute;
    bottom: 50%;
    width: 80%;
    border-radius: 2px;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  .trend-zero-line {
    position: absolute;
    width: 100%;
    height: 1px;
    background: var(--border-color);
    top: 50%;
  }

  .trend-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-secondary);
    padding: 4px 8px 0;
  }

  /* Response Patterns */
  .response-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .response-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .response-card.you {
    border-left: 3px solid var(--accent-color);
  }

  .response-card.them {
    border-left: 3px solid #34c759;
  }

  .response-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .response-label {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .response-details {
    display: flex;
    gap: 16px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .response-detail {
    flex: 1;
    text-align: center;
  }

  .detail-label {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .detail-value {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    margin-left: 4px;
  }

  .day-response-chart {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .day-response-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .day-name {
    font-size: 12px;
    color: var(--text-secondary);
    min-width: 32px;
  }

  .day-time {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  /* Frequency */
  .frequency-overview {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .frequency-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .frequency-card.trend {
    border-left: 3px solid var(--trend-color);
  }

  .frequency-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
  }

  .trend-icon {
    font-size: 14px;
    color: var(--trend-color, var(--text-primary));
  }

  .frequency-label {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: capitalize;
  }

  .weekly-chart {
    display: flex;
    align-items: flex-end;
    height: 80px;
    gap: 3px;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 8px;
  }

  .week-bar-container {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: flex-end;
  }

  .week-bar {
    width: 100%;
    background: var(--group-color);
    border-radius: 2px 2px 0 0;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  .chart-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-secondary);
    padding: 4px 8px 0;
  }

  /* Modal Footer */
  .modal-footer {
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-color);
    justify-content: flex-end;
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.primary {
    background: var(--accent-color);
    color: white;
    border: none;
  }

  .btn.primary:hover:not(:disabled) {
    background: #0a82e0;
  }

  .btn.secondary {
    background: var(--bg-hover);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--bg-active);
  }

  /* Responsive */
  @media (max-width: 600px) {
    .response-cards {
      grid-template-columns: 1fr;
    }

    .frequency-overview {
      grid-template-columns: repeat(2, 1fr);
    }

    .sentiment-overview {
      flex-direction: column;
    }

    .response-details {
      flex-direction: column;
      gap: 8px;
    }
  }
</style>
