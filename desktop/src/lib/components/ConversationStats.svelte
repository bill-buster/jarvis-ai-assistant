<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../api/client";
  import type { ConversationStats, TimeRange } from "../api/types";

  export let chatId: string;
  export let onClose: () => void;

  let loading = true;
  let error: string | null = null;
  let stats: ConversationStats | null = null;
  let selectedTimeRange: TimeRange = "month";

  const timeRangeOptions: { value: TimeRange; label: string }[] = [
    { value: "week", label: "Last Week" },
    { value: "month", label: "Last Month" },
    { value: "three_months", label: "Last 3 Months" },
    { value: "all_time", label: "All Time" },
  ];

  async function fetchStats() {
    loading = true;
    error = null;
    try {
      stats = await api.getConversationStats(chatId, selectedTimeRange, 500);
    } catch (e) {
      if (e instanceof Error) {
        error = `Failed to load statistics: ${e.message}`;
      } else {
        error = "Failed to load statistics. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function handleTimeRangeChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedTimeRange = target.value as TimeRange;
    fetchStats();
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
    if (!stats?.first_message_date || !stats?.last_message_date) return "";
    const start = new Date(stats.first_message_date);
    const end = new Date(stats.last_message_date);
    const options: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
    };
    return `${start.toLocaleDateString("en-US", options)} - ${end.toLocaleDateString("en-US", options)}`;
  }

  function getMaxHourlyCount(): number {
    if (!stats?.hourly_activity) return 1;
    return Math.max(...stats.hourly_activity.map((h) => h.count), 1);
  }

  function getMaxDailyCount(): number {
    if (!stats?.daily_activity) return 1;
    return Math.max(...Object.values(stats.daily_activity), 1);
  }

  function formatResponseTime(minutes: number | null): string {
    if (minutes === null) return "N/A";
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }

  function getPercentage(value: number, total: number): number {
    return total > 0 ? Math.round((value / total) * 100) : 0;
  }

  // Generate hour labels (12am, 3am, 6am, etc.)
  function getHourLabel(hour: number): string {
    if (hour === 0) return "12am";
    if (hour === 12) return "12pm";
    return hour < 12 ? `${hour}am` : `${hour - 12}pm`;
  }

  onMount(() => {
    fetchStats();
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
      <h2 id="modal-title">Conversation Statistics</h2>
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
          <p>Analyzing conversation...</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" on:click={fetchStats}>Try Again</button>
        </div>
      {:else if stats}
        <div class="stats-content">
          <!-- Date Range Info -->
          <div class="date-range-info">
            {stats.total_messages} messages analyzed ({formatDateRange()})
          </div>

          <!-- Overview Cards -->
          <div class="overview-section">
            <div class="stat-card">
              <div class="stat-value">{stats.total_messages}</div>
              <div class="stat-label">Total Messages</div>
            </div>
            <div class="stat-card">
              <div class="stat-value sent">{stats.sent_count}</div>
              <div class="stat-label">Sent</div>
              <div class="stat-percent">{getPercentage(stats.sent_count, stats.total_messages)}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-value received">{stats.received_count}</div>
              <div class="stat-label">Received</div>
              <div class="stat-percent">{getPercentage(stats.received_count, stats.total_messages)}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{formatResponseTime(stats.avg_response_time_minutes)}</div>
              <div class="stat-label">Avg Response</div>
            </div>
          </div>

          <!-- Sent/Received Bar -->
          <div class="ratio-section">
            <h3>Message Balance</h3>
            <div class="ratio-bar">
              <div
                class="ratio-sent"
                style="width: {getPercentage(stats.sent_count, stats.total_messages)}%"
              >
                {#if getPercentage(stats.sent_count, stats.total_messages) > 15}
                  Sent
                {/if}
              </div>
              <div
                class="ratio-received"
                style="width: {getPercentage(stats.received_count, stats.total_messages)}%"
              >
                {#if getPercentage(stats.received_count, stats.total_messages) > 15}
                  Received
                {/if}
              </div>
            </div>
          </div>

          <!-- Hourly Activity Chart -->
          <div class="chart-section">
            <h3>Activity by Hour</h3>
            <div class="hourly-chart">
              {#each stats.hourly_activity as hour}
                <div class="hour-bar-container" title="{getHourLabel(hour.hour)}: {hour.count} messages">
                  <div
                    class="hour-bar"
                    style="height: {(hour.count / getMaxHourlyCount()) * 100}%"
                  ></div>
                </div>
              {/each}
            </div>
            <div class="hour-labels">
              <span>12am</span>
              <span>6am</span>
              <span>12pm</span>
              <span>6pm</span>
              <span>11pm</span>
            </div>
          </div>

          <!-- Daily Activity Chart -->
          <div class="chart-section">
            <h3>Activity by Day</h3>
            <div class="daily-chart">
              {#each Object.entries(stats.daily_activity) as [day, count]}
                <div class="day-row">
                  <span class="day-label">{day.slice(0, 3)}</span>
                  <div class="day-bar-container">
                    <div
                      class="day-bar"
                      style="width: {(count / getMaxDailyCount()) * 100}%"
                    ></div>
                  </div>
                  <span class="day-count">{count}</span>
                </div>
              {/each}
            </div>
          </div>

          <!-- Message Length Distribution -->
          <div class="chart-section">
            <h3>Message Length</h3>
            <div class="length-distribution">
              {#each Object.entries(stats.message_length_distribution) as [category, count]}
                <div class="length-item">
                  <div class="length-label">
                    {category === "short" ? "Short (1-20)" :
                     category === "medium" ? "Medium (21-100)" :
                     category === "long" ? "Long (101-300)" : "Very Long (300+)"}
                  </div>
                  <div class="length-bar-container">
                    <div
                      class="length-bar"
                      style="width: {getPercentage(count, stats.total_messages)}%"
                    ></div>
                  </div>
                  <span class="length-count">{count}</span>
                </div>
              {/each}
            </div>
          </div>

          <!-- Two Column Layout for Words and Emoji -->
          <div class="two-column">
            <!-- Top Words -->
            <div class="chart-section">
              <h3>Top Words</h3>
              {#if stats.top_words.length > 0}
                <div class="word-list">
                  {#each stats.top_words.slice(0, 10) as word}
                    <div class="word-item">
                      <span class="word-text">{word.word}</span>
                      <span class="word-count">{word.count}</span>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="no-data">No word data available</p>
              {/if}
            </div>

            <!-- Emoji Usage -->
            <div class="chart-section">
              <h3>Top Emojis</h3>
              {#if Object.keys(stats.emoji_usage).length > 0}
                <div class="emoji-list">
                  {#each Object.entries(stats.emoji_usage) as [emoji, count]}
                    <div class="emoji-item">
                      <span class="emoji">{emoji}</span>
                      <span class="emoji-count">{count}</span>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="no-data">No emoji data available</p>
              {/if}
            </div>
          </div>

          <!-- Attachment Breakdown -->
          {#if Object.keys(stats.attachment_breakdown).length > 0}
            <div class="chart-section">
              <h3>Attachments</h3>
              <div class="attachment-grid">
                {#each Object.entries(stats.attachment_breakdown) as [type, count]}
                  <div class="attachment-item">
                    <span class="attachment-icon">
                      {#if type === "images"}
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                          <circle cx="8.5" cy="8.5" r="1.5"/>
                          <polyline points="21 15 16 10 5 21"/>
                        </svg>
                      {:else if type === "videos"}
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <polygon points="23 7 16 12 23 17 23 7"/>
                          <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                        </svg>
                      {:else if type === "audio"}
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M9 18V5l12-2v13"/>
                          <circle cx="6" cy="18" r="3"/>
                          <circle cx="18" cy="16" r="3"/>
                        </svg>
                      {:else if type === "documents"}
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                          <polyline points="14 2 14 8 20 8"/>
                          <line x1="16" y1="13" x2="8" y2="13"/>
                          <line x1="16" y1="17" x2="8" y2="17"/>
                        </svg>
                      {:else}
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                        </svg>
                      {/if}
                    </span>
                    <span class="attachment-type">{type}</span>
                    <span class="attachment-count">{count}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button class="btn secondary" on:click={fetchStats} disabled={loading}>
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
    max-width: 700px;
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

  .stats-content {
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

  /* Overview Cards */
  .overview-section {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .stat-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-value.sent {
    color: var(--accent-color);
  }

  .stat-value.received {
    color: #34c759;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .stat-percent {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  /* Ratio Bar */
  .ratio-section h3,
  .chart-section h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
  }

  .ratio-bar {
    display: flex;
    height: 24px;
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg-primary);
  }

  .ratio-sent {
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: white;
    font-weight: 500;
  }

  .ratio-received {
    background: #34c759;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: white;
    font-weight: 500;
  }

  /* Hourly Chart */
  .hourly-chart {
    display: flex;
    align-items: flex-end;
    height: 100px;
    gap: 2px;
    background: var(--bg-primary);
    border-radius: 8px;
    padding: 12px 8px 4px;
  }

  .hour-bar-container {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: flex-end;
  }

  .hour-bar {
    width: 100%;
    background: var(--accent-color);
    border-radius: 2px 2px 0 0;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  .hour-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-secondary);
    padding: 4px 8px 0;
  }

  /* Daily Chart */
  .daily-chart {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .day-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .day-label {
    width: 36px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .day-bar-container {
    flex: 1;
    height: 20px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
  }

  .day-bar {
    height: 100%;
    background: var(--group-color);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .day-count {
    width: 40px;
    text-align: right;
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Message Length Distribution */
  .length-distribution {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .length-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .length-label {
    width: 120px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .length-bar-container {
    flex: 1;
    height: 16px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
  }

  .length-bar {
    height: 100%;
    background: #ff9f0a;
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .length-count {
    width: 40px;
    text-align: right;
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Two Column Layout */
  .two-column {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  /* Word List */
  .word-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 200px;
    overflow-y: auto;
  }

  .word-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    font-size: 13px;
  }

  .word-text {
    color: var(--text-primary);
  }

  .word-count {
    color: var(--text-secondary);
  }

  /* Emoji List */
  .emoji-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .emoji-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--bg-primary);
    border-radius: 16px;
    font-size: 13px;
  }

  .emoji {
    font-size: 18px;
  }

  .emoji-count {
    color: var(--text-secondary);
    font-size: 12px;
  }

  .no-data {
    font-size: 13px;
    color: var(--text-secondary);
    text-align: center;
    padding: 16px;
  }

  /* Attachment Grid */
  .attachment-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .attachment-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--bg-primary);
    border-radius: 8px;
  }

  .attachment-icon {
    width: 20px;
    height: 20px;
    color: var(--accent-color);
  }

  .attachment-icon svg {
    width: 100%;
    height: 100%;
  }

  .attachment-type {
    font-size: 13px;
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .attachment-count {
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 500;
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
    .overview-section {
      grid-template-columns: repeat(2, 1fr);
    }

    .two-column {
      grid-template-columns: 1fr;
    }

    .stat-value {
      font-size: 20px;
    }
  }
</style>
