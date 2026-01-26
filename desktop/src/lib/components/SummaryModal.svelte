<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../api/client";
  import type { SummaryResponse } from "../api/types";

  export let chatId: string;
  export let onClose: () => void;

  let loading = true;
  let error: string | null = null;
  let summary: SummaryResponse | null = null;
  let copySuccess = false;

  async function fetchSummary() {
    loading = true;
    error = null;
    try {
      summary = await api.getSummary(chatId, 50);
    } catch (e) {
      if (e instanceof Error) {
        // Check for specific error conditions
        if (e.message.includes("timeout") || e.message.includes("abort")) {
          error = "Request timed out. Please try again.";
        } else if (e.message.includes("Not enough messages")) {
          error = "Not enough messages to summarize (need at least 5 messages).";
        } else {
          error = `Failed to generate summary: ${e.message}`;
        }
      } else {
        error = "Failed to generate summary. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  async function copyToClipboard() {
    if (!summary) return;

    const text = formatSummaryForClipboard();
    try {
      await navigator.clipboard.writeText(text);
      copySuccess = true;
      setTimeout(() => {
        copySuccess = false;
      }, 2000);
    } catch {
      error = "Failed to copy to clipboard";
    }
  }

  function formatSummaryForClipboard(): string {
    if (!summary) return "";

    let text = `Conversation Summary\n`;
    text += `${summary.message_count} messages (${formatDateRange()})\n\n`;
    text += `Summary:\n${summary.summary}\n`;

    if (summary.key_points && summary.key_points.length > 0) {
      text += `\nKey Points:\n`;
      summary.key_points.forEach((point) => {
        text += `• ${point}\n`;
      });
    }

    return text;
  }

  function formatDateRange(): string {
    if (!summary?.date_range) return "";
    const start = new Date(summary.date_range.start);
    const end = new Date(summary.date_range.end);
    const options: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
    };
    return `${start.toLocaleDateString("en-US", options)} - ${end.toLocaleDateString("en-US", options)}`;
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

  function exportAsText() {
    if (!summary) return;

    const text = formatSummaryForClipboard();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;

    // Generate filename with current date
    const date = new Date().toISOString().split("T")[0];
    link.download = `conversation-summary-${date}.txt`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // Fetch on mount
  onMount(() => {
    fetchSummary();
    // Add keyboard listener
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
      <h2 id="modal-title">
        <span class="icon">📋</span>
        Conversation Summary
      </h2>
      <button class="close-btn" on:click={onClose} aria-label="Close">
        ×
      </button>
    </div>

    <div class="modal-content">
      {#if loading}
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p>Generating summary...</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" on:click={fetchSummary}>
            Try Again
          </button>
        </div>
      {:else if summary}
        <div class="summary-content">
          <div class="meta-info">
            Last {summary.message_count} messages ({formatDateRange()})
          </div>

          <div class="section">
            <h3>Summary:</h3>
            <p class="summary-text">{summary.summary}</p>
          </div>

          {#if summary.key_points && summary.key_points.length > 0}
            <div class="section">
              <h3>Key Points:</h3>
              <ul class="key-points">
                {#each summary.key_points as point}
                  <li>{point}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button
        class="btn secondary"
        on:click={fetchSummary}
        disabled={loading}
      >
        Regenerate
      </button>
      <button
        class="btn secondary export-btn"
        on:click={exportAsText}
        disabled={loading || !summary}
        title="Save as text file"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Export
      </button>
      <button
        class="btn primary"
        on:click={copyToClipboard}
        disabled={loading || !summary}
      >
        {copySuccess ? "Copied!" : "Copy to Clipboard"}
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
    max-width: 560px;
    max-height: 80vh;
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
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .icon {
    font-size: 18px;
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
    min-height: 200px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
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
    transition: background-color 0.15s;
  }

  .retry-btn:hover {
    background: var(--bg-active);
  }

  .summary-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .meta-info {
    font-size: 13px;
    color: var(--text-secondary);
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
  }

  .section h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
  }

  .summary-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
    margin: 0;
  }

  .key-points {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .key-points li {
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-primary);
    padding: 6px 0;
    padding-left: 20px;
    position: relative;
  }

  .key-points li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: var(--accent-color);
    font-weight: bold;
  }

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

  .export-btn {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .export-btn svg {
    width: 16px;
    height: 16px;
  }
</style>
