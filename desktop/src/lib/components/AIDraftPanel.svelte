<script lang="ts">
  import { apiClient, APIError } from "../api/client";
  import type { DraftReplyResponse } from "../api/types";

  interface Props {
    chatId: string;
    onSelect: (text: string) => void;
    onClose: () => void;
  }

  let { chatId, onSelect, onClose }: Props = $props();

  // State
  type PanelState = "idle" | "loading" | "results" | "error";
  let panelState: PanelState = $state("idle");
  let instruction = $state("");
  let suggestions: DraftReplyResponse["suggestions"] = $state([]);
  let contextUsed: DraftReplyResponse["context_used"] | null = $state(null);
  let selectedIndex: number | null = $state(null);
  let errorMessage = $state("");

  // AbortController for cancelling in-flight requests
  let abortController: AbortController | null = null;

  async function generateReplies() {
    // Cancel any existing request
    abortController?.abort();
    abortController = new AbortController();

    panelState = "loading";
    errorMessage = "";
    selectedIndex = null;

    try {
      const response = await apiClient.getDraftReplies(
        chatId,
        instruction.trim() || undefined,
        3,
        abortController.signal
      );
      suggestions = response.suggestions;
      contextUsed = response.context_used;
      panelState = "results";
    } catch (e) {
      // Don't show error if request was aborted
      if (e instanceof Error && e.name === "AbortError") {
        return;
      }
      panelState = "error";
      if (e instanceof APIError) {
        errorMessage = e.detail || e.message;
      } else if (e instanceof Error) {
        errorMessage = e.message;
      } else {
        errorMessage = "An unknown error occurred";
      }
    }
  }

  function handleUseSelected() {
    if (selectedIndex !== null && suggestions[selectedIndex]) {
      onSelect(suggestions[selectedIndex].text);
    }
  }

  function handleClose() {
    // Cancel any in-flight request when closing
    abortController?.abort();
    onClose();
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      handleClose();
    }
  }
</script>

<svelte:window onkeydown={handleKeyDown} />

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="panel-overlay" onclick={handleClose} role="presentation">
  <!-- svelte-ignore a11y_interactive_supports_focus -->
  <div class="panel" onclick={(e) => e.stopPropagation()} role="dialog" aria-label="AI Draft Panel">
    <header class="panel-header">
      <div class="panel-title">
        <span class="ai-icon">✨</span>
        <h2>AI Draft</h2>
      </div>
      <button class="close-btn" onclick={handleClose} aria-label="Close">
        ×
      </button>
    </header>

    <div class="panel-content">
      {#if panelState === "idle" || panelState === "loading"}
        <div class="input-section">
          <label for="instruction">Optional: What do you want to say?</label>
          <input
            id="instruction"
            type="text"
            bind:value={instruction}
            placeholder="e.g., 'say yes but I'll be 10 min late'"
            disabled={panelState === "loading"}
          />
        </div>

        <button
          class="generate-btn"
          onclick={generateReplies}
          disabled={panelState === "loading"}
        >
          {#if panelState === "loading"}
            <span class="spinner"></span>
            Generating...
          {:else}
            Generate Replies
          {/if}
        </button>
      {/if}

      {#if panelState === "loading"}
        <div class="loading-section">
          <div class="loading-indicator">
            <span class="spinner large"></span>
            <p>Analyzing conversation and generating replies...</p>
          </div>
        </div>
      {/if}

      {#if panelState === "results"}
        <div class="results-section">
          <div class="results-header">
            <h3>Suggestions</h3>
            {#if contextUsed}
              <span class="context-info">
                Based on last {contextUsed.num_messages} messages
              </span>
            {/if}
          </div>

          <div class="suggestions-list">
            {#each suggestions as suggestion, index}
              <label class="suggestion-item" class:selected={selectedIndex === index}>
                <input
                  type="radio"
                  name="suggestion"
                  checked={selectedIndex === index}
                  onchange={() => (selectedIndex = index)}
                />
                <span class="suggestion-text">{suggestion.text}</span>
                <span class="confidence-badge">
                  {Math.round(suggestion.confidence * 100)}%
                </span>
              </label>
            {/each}
          </div>

          <div class="action-buttons">
            <button
              class="use-btn"
              onclick={handleUseSelected}
              disabled={selectedIndex === null}
            >
              Use Selected
            </button>
            <button class="regenerate-btn" onclick={generateReplies}>
              Regenerate
            </button>
          </div>
        </div>
      {/if}

      {#if panelState === "error"}
        <div class="error-section">
          <div class="error-icon">❌</div>
          <p class="error-message">{errorMessage}</p>
          <button class="retry-btn" onclick={generateReplies}>
            Try Again
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .panel-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.15s ease;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    width: 90%;
    max-width: 500px;
    max-height: 80vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: slideUp 0.2s ease;
  }

  @keyframes slideUp {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .ai-icon {
    font-size: 20px;
  }

  .panel-title h2 {
    font-size: 18px;
    font-weight: 600;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 24px;
    color: var(--text-secondary);
    cursor: pointer;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: all 0.15s ease;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .panel-content {
    padding: 20px;
    overflow-y: auto;
  }

  .input-section {
    margin-bottom: 16px;
  }

  .input-section label {
    display: block;
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .input-section input {
    width: 100%;
    padding: 12px 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    transition: border-color 0.15s ease;
  }

  .input-section input:focus {
    border-color: var(--accent-color);
  }

  .input-section input::placeholder {
    color: var(--text-secondary);
  }

  .generate-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent-color);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.15s ease;
  }

  .generate-btn:hover:not(:disabled) {
    background: #0a84e0;
  }

  .generate-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .spinner.large {
    width: 32px;
    height: 32px;
    border-width: 3px;
    border-color: rgba(255, 255, 255, 0.2);
    border-top-color: var(--accent-color);
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .loading-section {
    padding: 40px 20px;
  }

  .loading-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    color: var(--text-secondary);
  }

  .results-section {
    animation: fadeIn 0.2s ease;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .results-header h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .context-info {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .suggestions-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 20px;
  }

  .suggestion-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: var(--bg-primary);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .suggestion-item:hover {
    border-color: var(--bg-hover);
    background: var(--bg-hover);
  }

  .suggestion-item.selected {
    border-color: var(--accent-color);
    background: rgba(11, 147, 246, 0.1);
  }

  .suggestion-item input[type="radio"] {
    margin-top: 2px;
    accent-color: var(--accent-color);
    width: 18px;
    height: 18px;
    flex-shrink: 0;
  }

  .suggestion-text {
    flex: 1;
    font-size: 14px;
    line-height: 1.4;
  }

  .confidence-badge {
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 4px;
    flex-shrink: 0;
  }

  .action-buttons {
    display: flex;
    gap: 12px;
  }

  .use-btn {
    flex: 1;
    padding: 12px;
    background: var(--accent-color);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .use-btn:hover:not(:disabled) {
    background: #0a84e0;
  }

  .use-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .regenerate-btn {
    padding: 12px 20px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .regenerate-btn:hover {
    background: var(--bg-hover);
  }

  .error-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    text-align: center;
  }

  .error-icon {
    font-size: 40px;
    margin-bottom: 12px;
  }

  .error-message {
    color: var(--error-color);
    margin-bottom: 16px;
  }

  .retry-btn {
    padding: 10px 24px;
    background: var(--error-color);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .retry-btn:hover {
    opacity: 0.9;
  }
</style>
