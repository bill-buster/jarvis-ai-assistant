<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import {
    digestStore,
    fetchDigest,
    exportDigest,
  } from "../stores/digest";
  import type { DigestPeriod, DigestFormat } from "../api/types";

  const dispatch = createEventDispatcher<{
    navigate: { view: string; chatId?: string };
  }>();

  let selectedPeriod: DigestPeriod = "daily";
  let showExportMenu = false;

  onMount(() => {
    fetchDigest(selectedPeriod);
  });

  function handlePeriodChange(period: DigestPeriod) {
    selectedPeriod = period;
    fetchDigest(period);
  }

  async function handleExport(format: DigestFormat) {
    showExportMenu = false;
    const result = await exportDigest(selectedPeriod, format);
    if (result && result.success) {
      // Download the file
      const blob = new Blob([result.data], {
        type: format === "html" ? "text/html" : "text/markdown",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = result.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }

  function navigateToConversation(chatId: string) {
    dispatch("navigate", { view: "messages", chatId });
  }

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return "Unknown";
    const date = new Date(dateStr);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function formatHour(hour: number | null): string {
    if (hour === null) return "N/A";
    const ampm = hour >= 12 ? "PM" : "AM";
    const h = hour % 12 || 12;
    return `${h}:00 ${ampm}`;
  }

  $: digest = $digestStore.data;
  $: loading = $digestStore.loading;
  $: exporting = $digestStore.exporting;
  $: error = $digestStore.error;
</script>

<div class="digest-view">
  <header class="digest-header">
    <div class="header-left">
      <h1>Digest</h1>
      <div class="period-toggle">
        <button
          class:active={selectedPeriod === "daily"}
          on:click={() => handlePeriodChange("daily")}
        >
          Daily
        </button>
        <button
          class:active={selectedPeriod === "weekly"}
          on:click={() => handlePeriodChange("weekly")}
        >
          Weekly
        </button>
      </div>
    </div>
    <div class="header-actions">
      <div class="export-dropdown">
        <button
          class="export-btn"
          on:click={() => (showExportMenu = !showExportMenu)}
          disabled={loading || exporting || !digest}
        >
          {#if exporting}
            Exporting...
          {:else}
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export
          {/if}
        </button>
        {#if showExportMenu}
          <div class="dropdown-menu">
            <button on:click={() => handleExport("markdown")}>
              Markdown (.md)
            </button>
            <button on:click={() => handleExport("html")}>
              HTML (.html)
            </button>
          </div>
        {/if}
      </div>
      <button
        class="refresh-btn"
        on:click={() => fetchDigest(selectedPeriod)}
        disabled={loading}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6" />
          <path d="M1 20v-6h6" />
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
      </button>
    </div>
  </header>

  {#if loading}
    <div class="loading">
      <div class="spinner"></div>
      <p>Generating {selectedPeriod} digest...</p>
    </div>
  {:else if error}
    <div class="error">
      <p>{error}</p>
      <button on:click={() => fetchDigest(selectedPeriod)}>Try Again</button>
    </div>
  {:else if digest}
    <div class="digest-content">
      <!-- Activity Summary -->
      <section class="section stats-section">
        <h2>Activity Summary</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{digest.stats.total_messages}</div>
            <div class="stat-label">Total Messages</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{digest.stats.total_sent}</div>
            <div class="stat-label">Sent</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{digest.stats.total_received}</div>
            <div class="stat-label">Received</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{digest.stats.active_conversations}</div>
            <div class="stat-label">Active Chats</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{digest.stats.avg_messages_per_day.toFixed(1)}</div>
            <div class="stat-label">Avg/Day</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{formatHour(digest.stats.busiest_hour)}</div>
            <div class="stat-label">Busiest Hour</div>
          </div>
        </div>
        {#if digest.stats.most_active_conversation}
          <div class="most-active">
            Most active: <strong>{digest.stats.most_active_conversation}</strong>
            ({digest.stats.most_active_count} messages)
          </div>
        {/if}
      </section>

      <!-- Needs Attention -->
      {#if digest.needs_attention.length > 0}
        <section class="section attention-section">
          <h2>
            <span class="attention-icon">!</span>
            Needs Attention
          </h2>
          <p class="section-subtitle">Conversations with unanswered messages</p>
          <div class="attention-list">
            {#each digest.needs_attention as conv (conv.chat_id)}
              <button
                class="attention-item"
                on:click={() => navigateToConversation(conv.chat_id)}
              >
                <div class="avatar" class:group={conv.is_group}>
                  {(conv.display_name || "?").charAt(0).toUpperCase()}
                </div>
                <div class="item-content">
                  <div class="item-header">
                    <span class="item-name">{conv.display_name}</span>
                    <span class="item-badge">{conv.unanswered_count}</span>
                  </div>
                  {#if conv.last_message_preview}
                    <div class="item-preview">{conv.last_message_preview}</div>
                  {/if}
                  {#if conv.last_message_date}
                    <div class="item-date">{formatDate(conv.last_message_date)}</div>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        </section>
      {/if}

      <!-- Group Highlights -->
      {#if digest.highlights.length > 0}
        <section class="section highlights-section">
          <h2>Group Highlights</h2>
          <p class="section-subtitle">Active group conversations</p>
          <div class="highlights-list">
            {#each digest.highlights as highlight (highlight.chat_id)}
              <button
                class="highlight-item"
                on:click={() => navigateToConversation(highlight.chat_id)}
              >
                <div class="avatar group">
                  {(highlight.display_name || "G").charAt(0).toUpperCase()}
                </div>
                <div class="item-content">
                  <div class="item-header">
                    <span class="item-name">{highlight.display_name}</span>
                    <span class="message-count">{highlight.message_count} msgs</span>
                  </div>
                  <div class="active-participants">
                    Active: {highlight.active_participants.slice(0, 3).join(", ")}
                  </div>
                  {#if highlight.top_topics.length > 0}
                    <div class="topics">
                      {#each highlight.top_topics as topic}
                        <span class="topic-tag">{topic}</span>
                      {/each}
                    </div>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        </section>
      {/if}

      <!-- Action Items -->
      {#if digest.action_items.length > 0}
        <section class="section action-items-section">
          <h2>Action Items</h2>
          <p class="section-subtitle">Tasks, questions, events, and reminders</p>
          <div class="action-items-list">
            {#each digest.action_items as item (item.message_id)}
              <button
                class="action-item"
                on:click={() => navigateToConversation(item.chat_id)}
              >
                <div class="action-type" class:task={item.item_type === "task"} class:question={item.item_type === "question"} class:event={item.item_type === "event"} class:reminder={item.item_type === "reminder"}>
                  {#if item.item_type === "task"}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M9 11l3 3L22 4" />
                      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
                    </svg>
                  {:else if item.item_type === "question"}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                  {:else if item.item_type === "event"}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                      <line x1="16" y1="2" x2="16" y2="6" />
                      <line x1="8" y1="2" x2="8" y2="6" />
                      <line x1="3" y1="10" x2="21" y2="10" />
                    </svg>
                  {:else}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10" />
                      <polyline points="12 6 12 12 16 14" />
                    </svg>
                  {/if}
                </div>
                <div class="item-content">
                  <div class="action-text">{item.text}</div>
                  <div class="action-meta">
                    <span class="action-sender">{item.sender}</span>
                    <span class="action-separator">in</span>
                    <span class="action-conv">{item.conversation_name}</span>
                    <span class="action-date">{formatDate(item.date)}</span>
                  </div>
                </div>
              </button>
            {/each}
          </div>
        </section>
      {/if}

      {#if digest.needs_attention.length === 0 && digest.highlights.length === 0 && digest.action_items.length === 0}
        <div class="no-activity">
          <p>No significant activity in this period.</p>
        </div>
      {/if}
    </div>
  {:else}
    <div class="empty">
      <p>No digest data available. Click refresh to generate.</p>
    </div>
  {/if}
</div>

<style>
  .digest-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .digest-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  h1 {
    font-size: 24px;
    font-weight: 600;
    margin: 0;
  }

  .period-toggle {
    display: flex;
    background: var(--bg-active);
    border-radius: 8px;
    padding: 4px;
  }

  .period-toggle button {
    padding: 6px 16px;
    border: none;
    background: none;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .period-toggle button.active {
    background: var(--accent-color);
    color: white;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .export-dropdown {
    position: relative;
  }

  .export-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    color: var(--text-primary);
    transition: all 0.15s ease;
  }

  .export-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .export-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .export-btn svg {
    width: 16px;
    height: 16px;
  }

  .dropdown-menu {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 4px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    overflow: hidden;
    z-index: 100;
  }

  .dropdown-menu button {
    display: block;
    width: 100%;
    padding: 10px 16px;
    border: none;
    background: none;
    text-align: left;
    cursor: pointer;
    font-size: 14px;
    white-space: nowrap;
  }

  .dropdown-menu button:hover {
    background: var(--bg-hover);
  }

  .refresh-btn {
    padding: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .refresh-btn svg {
    width: 18px;
    height: 18px;
    display: block;
  }

  .digest-content {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  .section {
    margin-bottom: 32px;
  }

  .section h2 {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 4px 0;
  }

  .section-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
    margin: 0 0 16px 0;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }

  .stat-value {
    font-size: 28px;
    font-weight: 600;
    color: var(--accent-color);
  }

  .stat-label {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .most-active {
    background: var(--bg-active);
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    color: var(--text-secondary);
  }

  .most-active strong {
    color: var(--text-primary);
  }

  .attention-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--error-color);
    color: white;
    border-radius: 50%;
    font-size: 14px;
    font-weight: bold;
  }

  .attention-list,
  .highlights-list,
  .action-items-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .attention-item,
  .highlight-item,
  .action-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    cursor: pointer;
    text-align: left;
    width: 100%;
    transition: all 0.15s ease;
  }

  .attention-item:hover,
  .highlight-item:hover,
  .action-item:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: white;
    flex-shrink: 0;
  }

  .avatar.group {
    background: var(--group-color);
  }

  .item-content {
    flex: 1;
    min-width: 0;
  }

  .item-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .item-name {
    font-weight: 500;
    font-size: 15px;
  }

  .item-badge {
    background: var(--error-color);
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
  }

  .item-preview {
    font-size: 14px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
  }

  .item-date {
    font-size: 12px;
    color: var(--text-tertiary);
  }

  .message-count {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .active-participants {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .topics {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .topic-tag {
    background: var(--bg-active);
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .action-type {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .action-type svg {
    width: 18px;
    height: 18px;
  }

  .action-type.task {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .action-type.question {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .action-type.event {
    background: rgba(88, 86, 214, 0.2);
    color: var(--group-color);
  }

  .action-type.reminder {
    background: rgba(255, 159, 10, 0.2);
    color: #ff9f0a;
  }

  .action-text {
    font-size: 14px;
    margin-bottom: 4px;
    line-height: 1.4;
  }

  .action-meta {
    font-size: 12px;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .action-separator {
    color: var(--text-tertiary);
  }

  .action-conv {
    color: var(--text-primary);
  }

  .action-date {
    color: var(--text-tertiary);
  }

  .loading,
  .error,
  .empty,
  .no-activity {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: var(--text-secondary);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 16px;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error {
    color: var(--error-color);
  }

  .error button {
    margin-top: 16px;
    padding: 8px 16px;
    border: 1px solid var(--accent-color);
    background: none;
    color: var(--accent-color);
    border-radius: 8px;
    cursor: pointer;
  }

  .error button:hover {
    background: var(--accent-color);
    color: white;
  }
</style>
