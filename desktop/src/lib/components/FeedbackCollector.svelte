<script lang="ts">
  /**
   * FeedbackCollector - Unobtrusively captures user feedback on AI suggestions.
   *
   * This component tracks when users:
   * - Send suggestions unchanged (implicit positive)
   * - Edit suggestions before sending (captures before/after)
   * - Dismiss suggestions (implicit negative)
   * - Copy suggestions (partial positive)
   *
   * Feedback is sent to the backend asynchronously without blocking the UI.
   */

  import { onMount, onDestroy } from "svelte";
  import { api } from "../api/client";
  import type { FeedbackAction, EvaluationScores } from "../api/types";

  // Props
  let {
    chatId = "",
    contextMessages = [] as string[],
    enabled = true,
  }: {
    chatId: string;
    contextMessages: string[];
    enabled?: boolean;
  } = $props();

  // State for feedback queue (for async processing)
  let feedbackQueue = $state<Array<{
    action: FeedbackAction;
    suggestionText: string;
    editedText?: string | null;
    timestamp: number;
  }>>([]);
  let processing = $state(false);
  let lastEvaluation = $state<EvaluationScores | null>(null);
  let showFeedbackToast = $state(false);
  let feedbackToastMessage = $state("");
  let toastTimeout: ReturnType<typeof setTimeout> | null = null;

  // Process feedback queue
  $effect(() => {
    if (feedbackQueue.length > 0 && !processing && enabled) {
      processNextFeedback();
    }
  });

  async function processNextFeedback() {
    if (feedbackQueue.length === 0 || processing) return;

    processing = true;
    const item = feedbackQueue[0];

    try {
      const response = await api.recordFeedback(
        item.action,
        item.suggestionText,
        chatId,
        contextMessages,
        item.editedText,
        true,
        { timestamp: item.timestamp }
      );

      if (response.evaluation) {
        lastEvaluation = response.evaluation;
      }

      // Remove processed item from queue
      feedbackQueue = feedbackQueue.slice(1);
    } catch (err) {
      console.error("Failed to record feedback:", err);
      // Remove failed item to prevent infinite retry
      feedbackQueue = feedbackQueue.slice(1);
    } finally {
      processing = false;
    }
  }

  /**
   * Record feedback when a suggestion is sent unchanged.
   * Call this when the user clicks "Send" on an unmodified suggestion.
   */
  export function recordSent(suggestionText: string) {
    if (!enabled || !suggestionText) return;

    feedbackQueue = [...feedbackQueue, {
      action: "sent" as FeedbackAction,
      suggestionText,
      timestamp: Date.now(),
    }];

    showToast("Thanks for the feedback!");
  }

  /**
   * Record feedback when a suggestion is edited before sending.
   * Call this when the user modifies and sends a suggestion.
   */
  export function recordEdited(originalText: string, editedText: string) {
    if (!enabled || !originalText || !editedText) return;

    feedbackQueue = [...feedbackQueue, {
      action: "edited" as FeedbackAction,
      suggestionText: originalText,
      editedText,
      timestamp: Date.now(),
    }];

    showToast("Feedback recorded");
  }

  /**
   * Record feedback when a suggestion is dismissed.
   * Call this when the user closes/dismisses a suggestion without using it.
   */
  export function recordDismissed(suggestionText: string) {
    if (!enabled || !suggestionText) return;

    feedbackQueue = [...feedbackQueue, {
      action: "dismissed" as FeedbackAction,
      suggestionText,
      timestamp: Date.now(),
    }];
  }

  /**
   * Record feedback when a suggestion is copied.
   * Call this when the user copies a suggestion to clipboard.
   */
  export function recordCopied(suggestionText: string) {
    if (!enabled || !suggestionText) return;

    feedbackQueue = [...feedbackQueue, {
      action: "copied" as FeedbackAction,
      suggestionText,
      timestamp: Date.now(),
    }];
  }

  /**
   * Get the last evaluation scores (if available).
   */
  export function getLastEvaluation(): EvaluationScores | null {
    return lastEvaluation;
  }

  function showToast(message: string) {
    feedbackToastMessage = message;
    showFeedbackToast = true;

    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }

    toastTimeout = setTimeout(() => {
      showFeedbackToast = false;
    }, 2000);
  }

  onDestroy(() => {
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
  });
</script>

{#if showFeedbackToast}
  <div class="feedback-toast" class:visible={showFeedbackToast}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M20 6L9 17l-5-5" />
    </svg>
    <span>{feedbackToastMessage}</span>
  </div>
{/if}

{#if lastEvaluation}
  <div class="evaluation-indicator" title="Response quality scores">
    <div class="score-bar">
      <div
        class="score-fill"
        style="width: {lastEvaluation.overall_score * 100}%"
        class:high={lastEvaluation.overall_score >= 0.7}
        class:medium={lastEvaluation.overall_score >= 0.4 && lastEvaluation.overall_score < 0.7}
        class:low={lastEvaluation.overall_score < 0.4}
      ></div>
    </div>
  </div>
{/if}

<style>
  .feedback-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 20px;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #444);
    border-radius: 8px;
    color: var(--text-primary, #fff);
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    opacity: 0;
    transform: translateY(10px);
    transition: opacity 0.2s ease, transform 0.2s ease;
    z-index: 1000;
  }

  .feedback-toast.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .feedback-toast svg {
    width: 16px;
    height: 16px;
    color: #34c759;
  }

  .evaluation-indicator {
    position: fixed;
    bottom: 80px;
    right: 24px;
    width: 100px;
    opacity: 0.7;
    transition: opacity 0.2s ease;
  }

  .evaluation-indicator:hover {
    opacity: 1;
  }

  .score-bar {
    height: 4px;
    background: var(--bg-tertiary, #333);
    border-radius: 2px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .score-fill.high {
    background: #34c759;
  }

  .score-fill.medium {
    background: #ff9500;
  }

  .score-fill.low {
    background: #ff3b30;
  }
</style>
