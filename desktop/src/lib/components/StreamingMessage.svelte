<script lang="ts">
  /**
   * Streaming message display component
   *
   * Shows generation progress with token-by-token text display
   * and streaming animation.
   */
  import {
    isStreaming,
    streamingText,
    streamingError,
    websocketStore,
    cancelGeneration,
    resetStreamingState,
  } from "../stores/websocket";

  interface Props {
    onComplete?: (text: string) => void;
    onError?: (error: string) => void;
    showControls?: boolean;
  }

  let { onComplete, onError, showControls = true }: Props = $props();

  // Track previous state to detect completion
  let wasStreaming = false;

  $effect(() => {
    // Detect transition from streaming to not streaming
    if (wasStreaming && !$isStreaming) {
      if ($streamingError) {
        onError?.($streamingError);
      } else if ($streamingText) {
        onComplete?.($streamingText);
      }
    }
    wasStreaming = $isStreaming;
  });

  function handleCancel() {
    cancelGeneration();
  }

  function handleReset() {
    resetStreamingState();
  }

  // Calculate streaming stats
  $: elapsedTime =
    $websocketStore.streaming.startTime && $websocketStore.streaming.endTime
      ? (
          ($websocketStore.streaming.endTime -
            $websocketStore.streaming.startTime) /
          1000
        ).toFixed(1)
      : $websocketStore.streaming.startTime
        ? ((Date.now() - $websocketStore.streaming.startTime) / 1000).toFixed(1)
        : null;

  $: tokenCount = $websocketStore.streaming.tokens.length;
</script>

{#if $isStreaming || $streamingText || $streamingError}
  <div class="streaming-container" class:error={$streamingError}>
    <div class="streaming-header">
      <div class="streaming-status">
        {#if $isStreaming}
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <span class="status-text">Generating...</span>
        {:else if $streamingError}
          <svg
            class="error-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span class="status-text">Error</span>
        {:else}
          <svg
            class="complete-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          <span class="status-text">Complete</span>
        {/if}
      </div>

      {#if showControls}
        <div class="streaming-controls">
          {#if $isStreaming}
            <button class="cancel-btn" onclick={handleCancel}>
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              </svg>
              Cancel
            </button>
          {:else}
            <button class="reset-btn" onclick={handleReset}>
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              Dismiss
            </button>
          {/if}
        </div>
      {/if}
    </div>

    <div class="streaming-content">
      {#if $streamingError}
        <p class="error-message">{$streamingError}</p>
      {:else if $streamingText}
        <p class="generated-text">{$streamingText}</p>
        {#if $isStreaming}
          <span class="cursor"></span>
        {/if}
      {:else if $isStreaming}
        <p class="placeholder-text">Waiting for response...</p>
      {/if}
    </div>

    {#if elapsedTime || tokenCount > 0}
      <div class="streaming-stats">
        {#if tokenCount > 0}
          <span class="stat">{tokenCount} tokens</span>
        {/if}
        {#if elapsedTime}
          <span class="stat">{elapsedTime}s</span>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .streaming-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
  }

  .streaming-container.error {
    border-color: var(--error-color);
    background: rgba(255, 95, 87, 0.05);
  }

  .streaming-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .streaming-status {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-text {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  /* Typing indicator animation */
  .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 4px;
  }

  .typing-indicator span {
    width: 6px;
    height: 6px;
    background: var(--accent-color);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
  }

  .typing-indicator span:nth-child(1) {
    animation-delay: -0.32s;
  }

  .typing-indicator span:nth-child(2) {
    animation-delay: -0.16s;
  }

  @keyframes bounce {
    0%,
    80%,
    100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }

  .error-icon {
    width: 18px;
    height: 18px;
    color: var(--error-color);
  }

  .complete-icon {
    width: 18px;
    height: 18px;
    color: #34c759;
  }

  .streaming-controls {
    display: flex;
    gap: 8px;
  }

  .streaming-controls button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid var(--border-color);
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .streaming-controls button:hover {
    background: var(--bg-hover);
  }

  .cancel-btn:hover {
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .streaming-controls button svg {
    width: 14px;
    height: 14px;
  }

  .streaming-content {
    line-height: 1.6;
    font-size: 14px;
    position: relative;
  }

  .generated-text {
    white-space: pre-wrap;
    word-break: break-word;
  }

  .placeholder-text {
    color: var(--text-secondary);
    font-style: italic;
  }

  .error-message {
    color: var(--error-color);
  }

  /* Blinking cursor */
  .cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--accent-color);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
  }

  @keyframes blink {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0;
    }
  }

  .streaming-stats {
    display: flex;
    gap: 16px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
  }

  .stat {
    font-size: 12px;
    color: var(--text-secondary);
  }
</style>
