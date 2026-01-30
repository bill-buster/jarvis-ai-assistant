<script lang="ts">
  import type { GenerationDebugInfo } from "$lib/api/types";

  export let debug: GenerationDebugInfo | null = null;
  export let generationTimeMs: number = 0;

  let expanded = false;
</script>

{#if debug}
  <div class="debug-panel">
    <button class="debug-toggle" on:click={() => (expanded = !expanded)}>
      <span class="debug-icon">🔍</span>
      <span class="debug-label">Generation Analysis</span>
      <span class="debug-time">{generationTimeMs.toFixed(0)}ms</span>
      <span class="expand-icon">{expanded ? "▼" : "▶"}</span>
    </button>

    {#if expanded}
      <div class="debug-content">
        <!-- Intent -->
        <div class="debug-section">
          <span class="section-label">Intent:</span>
          <span class="section-value intent">{debug.intent_detected}</span>
        </div>

        <!-- Style -->
        <div class="debug-section">
          <span class="section-label">Style:</span>
          <span class="section-value style">{debug.style_instructions}</span>
        </div>

        <!-- Past Replies -->
        {#if debug.past_replies_found.length > 0}
          <div class="debug-section">
            <span class="section-label">Past replies found:</span>
            <div class="past-replies">
              {#each debug.past_replies_found as pr}
                <div class="past-reply">
                  <span class="similarity">{(pr.similarity * 100).toFixed(0)}%</span>
                  <span class="their-msg">"{pr.their_message}"</span>
                  <span class="arrow">→</span>
                  <span class="your-reply">"{pr.your_reply}"</span>
                </div>
              {/each}
            </div>
          </div>
        {:else}
          <div class="debug-section">
            <span class="section-label">Past replies:</span>
            <span class="section-value none">None found (using generic examples)</span>
          </div>
        {/if}

        <!-- Full Prompt -->
        <div class="debug-section">
          <span class="section-label">Full prompt sent to LLM:</span>
          <pre class="full-prompt">{debug.full_prompt}</pre>
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .debug-panel {
    margin: 0 16px 8px 16px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    font-size: 11px;
    overflow: hidden;
  }

  .debug-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    text-align: left;
  }

  .debug-toggle:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .debug-icon {
    font-size: 12px;
  }

  .debug-label {
    flex: 1;
    font-weight: 500;
  }

  .debug-time {
    color: var(--accent-green);
    font-family: monospace;
  }

  .expand-icon {
    font-size: 10px;
    opacity: 0.6;
  }

  .debug-content {
    padding: 8px 12px 12px 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .debug-section {
    margin-bottom: 10px;
  }

  .debug-section:last-child {
    margin-bottom: 0;
  }

  .section-label {
    display: block;
    color: var(--text-secondary);
    margin-bottom: 4px;
    font-weight: 500;
  }

  .section-value {
    color: var(--text-primary);
  }

  .section-value.intent {
    background: rgba(88, 86, 214, 0.2);
    color: #5856d6;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .section-value.style {
    font-style: italic;
    color: var(--accent-green);
  }

  .section-value.none {
    color: var(--text-secondary);
    font-style: italic;
  }

  .past-replies {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .past-reply {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    flex-wrap: wrap;
  }

  .similarity {
    background: var(--accent-blue);
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 10px;
  }

  .their-msg {
    color: var(--text-secondary);
  }

  .arrow {
    color: var(--text-secondary);
  }

  .your-reply {
    color: var(--accent-green);
    font-weight: 500;
  }

  .full-prompt {
    background: rgba(0, 0, 0, 0.3);
    padding: 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 10px;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-primary);
    max-height: 300px;
    overflow-y: auto;
    margin: 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
</style>
