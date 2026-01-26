<script lang="ts">
  import { api } from "../api/client";
  import type { PDFExportRequest } from "../api/types";

  interface Props {
    chatId: string;
    onClose: () => void;
  }

  let { chatId, onClose }: Props = $props();

  // Export options
  let includeAttachments = $state(true);
  let includeReactions = $state(true);
  let useDateRange = $state(false);
  let startDate = $state("");
  let endDate = $state("");
  let messageLimit = $state(1000);

  // State
  let loading = $state(false);
  let progress = $state(0);
  let error: string | null = $state(null);
  let success = $state(false);

  async function handleExport() {
    loading = true;
    progress = 10;
    error = null;
    success = false;

    try {
      // Build request options
      const options: PDFExportRequest = {
        include_attachments: includeAttachments,
        include_reactions: includeReactions,
        limit: messageLimit,
      };

      if (useDateRange && (startDate || endDate)) {
        options.date_range = {};
        if (startDate) {
          options.date_range.start = new Date(startDate).toISOString();
        }
        if (endDate) {
          options.date_range.end = new Date(endDate).toISOString();
        }
      }

      progress = 30;

      // Fetch PDF data
      const response = await api.exportPDF(chatId, options);

      progress = 70;

      // Convert base64 to blob and download
      const binaryString = atob(response.data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "application/pdf" });

      progress = 90;

      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = response.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      progress = 100;
      success = true;

      // Close after short delay to show success
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (e) {
      if (e instanceof Error) {
        error = e.message;
      } else {
        error = "Failed to export PDF. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget && !loading) {
      onClose();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && !loading) {
      onClose();
    }
  }

  // Set up keyboard listener
  $effect(() => {
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={handleBackdropClick}>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-header">
      <h2 id="modal-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        Export to PDF
      </h2>
      <button
        class="close-btn"
        onclick={onClose}
        aria-label="Close"
        disabled={loading}
      >
        &times;
      </button>
    </div>

    <div class="modal-content">
      {#if loading}
        <div class="loading-state">
          <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%"></div>
          </div>
          <p>Generating PDF... {progress}%</p>
          <span class="loading-hint">This may take a moment for long conversations</span>
        </div>
      {:else if success}
        <div class="success-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
          <p>PDF exported successfully!</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" onclick={() => (error = null)}>
            Try Again
          </button>
        </div>
      {:else}
        <div class="options-form">
          <div class="option-group">
            <h3>Content Options</h3>

            <label class="checkbox-option">
              <input type="checkbox" bind:checked={includeAttachments} />
              <span class="checkbox-label">
                <span class="label-text">Include Attachments</span>
                <span class="label-hint">Show image thumbnails in the PDF</span>
              </span>
            </label>

            <label class="checkbox-option">
              <input type="checkbox" bind:checked={includeReactions} />
              <span class="checkbox-label">
                <span class="label-text">Include Reactions</span>
                <span class="label-hint">Show message reactions (tapbacks)</span>
              </span>
            </label>
          </div>

          <div class="option-group">
            <h3>Message Limit</h3>
            <div class="input-row">
              <input
                type="number"
                bind:value={messageLimit}
                min="1"
                max="10000"
                class="number-input"
              />
              <span class="input-hint">messages (max 10,000)</span>
            </div>
          </div>

          <div class="option-group">
            <label class="checkbox-option">
              <input type="checkbox" bind:checked={useDateRange} />
              <span class="checkbox-label">
                <span class="label-text">Filter by Date Range</span>
                <span class="label-hint">Export only messages within a date range</span>
              </span>
            </label>

            {#if useDateRange}
              <div class="date-range-inputs">
                <div class="date-input-group">
                  <label for="start-date">From:</label>
                  <input
                    type="date"
                    id="start-date"
                    bind:value={startDate}
                    class="date-input"
                  />
                </div>
                <div class="date-input-group">
                  <label for="end-date">To:</label>
                  <input
                    type="date"
                    id="end-date"
                    bind:value={endDate}
                    class="date-input"
                  />
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button
        class="btn secondary"
        onclick={onClose}
        disabled={loading}
      >
        Cancel
      </button>
      <button
        class="btn primary"
        onclick={handleExport}
        disabled={loading || success}
      >
        {#if loading}
          Exporting...
        {:else}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          Export PDF
        {/if}
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
    max-width: 480px;
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

  .modal-header h2 svg {
    width: 20px;
    height: 20px;
    color: var(--accent-color);
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

  .close-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .close-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
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

  .progress-container {
    width: 100%;
    max-width: 300px;
    height: 8px;
    background: var(--bg-hover);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: var(--accent-color);
    transition: width 0.3s ease;
  }

  .loading-hint {
    font-size: 12px;
    opacity: 0.7;
  }

  .success-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 16px;
    color: var(--success-color, #22c55e);
  }

  .success-state svg {
    width: 48px;
    height: 48px;
  }

  .success-state p {
    font-size: 16px;
    font-weight: 500;
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

  .options-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .option-group {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .option-group h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0;
  }

  .checkbox-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    cursor: pointer;
  }

  .checkbox-option input[type="checkbox"] {
    margin-top: 2px;
    width: 16px;
    height: 16px;
    accent-color: var(--accent-color);
  }

  .checkbox-label {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .label-text {
    font-size: 14px;
    color: var(--text-primary);
  }

  .label-hint {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .input-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .number-input {
    width: 100px;
    padding: 8px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 14px;
  }

  .number-input:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .input-hint {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .date-range-inputs {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    padding-left: 26px;
  }

  .date-input-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .date-input-group label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .date-input {
    padding: 8px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 14px;
  }

  .date-input:focus {
    outline: none;
    border-color: var(--accent-color);
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
    display: flex;
    align-items: center;
    gap: 6px;
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

  .btn.primary svg {
    width: 16px;
    height: 16px;
  }

  .btn.secondary {
    background: var(--bg-hover);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--bg-active);
  }
</style>
