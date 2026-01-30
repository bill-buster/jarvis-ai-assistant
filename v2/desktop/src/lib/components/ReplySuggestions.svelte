<script lang="ts">
  import type { GeneratedReply } from "../api/types";

  export let replies: GeneratedReply[] = [];
  export let streamingReplies: GeneratedReply[] = [];
  export let loading: boolean = false;
  export let isStreaming: boolean = false;
  export let chatId: string | null = null;
  export let onGenerate: () => void = () => {};
  export let onSend: (text: string) => Promise<boolean> = async () => false;

  // Show streaming replies while streaming, otherwise show final replies
  $: displayReplies = isStreaming ? streamingReplies : replies;

  let inputText: string = "";
  let sending: boolean = false;
  let sent: boolean = false;

  // Clear input when chat changes
  $: if (chatId) {
    inputText = "";
    sent = false;
  }

  function formatType(type: string): string {
    return type.replace(/_/g, " ");
  }

  function selectReply(reply: GeneratedReply) {
    // Populate the text input instead of auto-sending
    inputText = reply.text;
  }

  async function sendInput() {
    if (!inputText.trim() || sending) return;

    sending = true;
    const success = await onSend(inputText);
    sending = false;

    if (success) {
      inputText = "";
      sent = true;
      setTimeout(() => {
        sent = false;
      }, 2000);
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    // Cmd/Ctrl+Enter or just Enter to send
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) {
      e.preventDefault();
      sendInput();
    }
    // Escape to clear
    if (e.key === "Escape") {
      inputText = "";
    }
  }

  // Global keyboard shortcuts
  function handleGlobalKeydown(e: KeyboardEvent) {
    // Cmd+1/2/3 to select suggestion
    if (e.metaKey && ["1", "2", "3"].includes(e.key)) {
      const index = parseInt(e.key) - 1;
      if (replies[index]) {
        e.preventDefault();
        selectReply(replies[index]);
      }
    }
  }
</script>

<svelte:window on:keydown={handleGlobalKeydown} />

<div class="reply-section">
  <!-- Text Input -->
  <div class="input-container">
    <textarea
      bind:value={inputText}
      placeholder="Type a message..."
      rows="1"
      on:keydown={handleKeydown}
      disabled={sending}
    ></textarea>
    <button class="send-btn" on:click={sendInput} disabled={!inputText.trim() || sending}>
      {#if sending}
        <span class="sending-dots">...</span>
      {:else if sent}
        Sent!
      {:else}
        Send
      {/if}
    </button>
  </div>

  <!-- Suggestions -->
  <div class="suggestions-header">
    <h3>AI Suggestions</h3>
    <button class="generate-btn" on:click={onGenerate} disabled={loading}>
      {loading ? "Generating..." : "Generate"}
    </button>
  </div>

  {#if loading && displayReplies.length === 0}
    <div class="loading-replies">
      <div class="spinner"></div>
      <span>{isStreaming ? "Streaming replies..." : "Generating replies..."}</span>
    </div>
  {:else if displayReplies.length > 0}
    <div class="reply-suggestions">
      {#each displayReplies as reply, i}
        <button
          class="reply-suggestion"
          class:streaming={isStreaming && i === displayReplies.length - 1}
          on:click={() => selectReply(reply)}
          title="⌘{i + 1} to use"
        >
          <div class="reply-content">
            <div class="reply-type">{formatType(reply.reply_type)}</div>
            <div class="reply-text">{reply.text}</div>
          </div>
          <div class="reply-action">
            <span class="use-label">Use</span>
          </div>
        </button>
      {/each}
      {#if isStreaming}
        <div class="streaming-indicator">
          <div class="spinner small"></div>
          <span>More coming...</span>
        </div>
      {/if}
    </div>
    {#if !isStreaming}
      <p class="hint">Click a suggestion to use it</p>
    {/if}
  {:else}
    <div class="empty-replies">
      Click "Generate" to get AI-powered reply suggestions
    </div>
  {/if}
</div>

<style>
  .reply-section {
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .input-container {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }

  textarea {
    flex: 1;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    resize: none;
    outline: none;
    min-height: 40px;
    max-height: 100px;
  }

  textarea:focus {
    border-color: var(--accent-blue);
  }

  textarea::placeholder {
    color: var(--text-secondary);
  }

  textarea:disabled {
    opacity: 0.6;
  }

  .send-btn {
    padding: 8px 20px;
    background: var(--accent-blue);
    border: none;
    border-radius: 20px;
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
    min-width: 70px;
  }

  .send-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .sending-dots {
    letter-spacing: 2px;
  }

  .suggestions-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .suggestions-header h3 {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0;
  }

  .generate-btn {
    padding: 4px 10px;
    background: transparent;
    border: 1px solid var(--accent-blue);
    border-radius: 4px;
    color: var(--accent-blue);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }

  .generate-btn:hover:not(:disabled) {
    background: var(--accent-blue);
    color: white;
  }

  .generate-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .loading-replies {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 0;
    color: var(--text-secondary);
    font-size: 13px;
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .reply-suggestions {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .reply-suggestion {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid transparent;
    color: var(--text-primary);
    text-align: left;
    width: 100%;
  }

  .reply-suggestion:hover {
    border-color: var(--accent-blue);
    background: var(--hover-color);
  }

  .reply-content {
    flex: 1;
    min-width: 0;
  }

  .reply-type {
    font-size: 10px;
    color: var(--accent-blue);
    margin-bottom: 2px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .reply-text {
    font-size: 13px;
    line-height: 1.4;
  }

  .reply-action {
    margin-left: 12px;
    flex-shrink: 0;
  }

  .use-label {
    font-size: 12px;
    color: var(--accent-blue);
    font-weight: 600;
  }

  .hint {
    font-size: 11px;
    color: var(--text-secondary);
    margin: 8px 0 0;
    text-align: center;
  }

  .empty-replies {
    color: var(--text-secondary);
    font-size: 12px;
    padding: 8px 0;
  }

  .reply-suggestion.streaming {
    border-color: var(--accent-green);
    animation: pulse-border 1s infinite;
  }

  @keyframes pulse-border {
    0%, 100% {
      border-color: var(--accent-green);
    }
    50% {
      border-color: transparent;
    }
  }

  .streaming-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 0;
    color: var(--accent-green);
    font-size: 11px;
  }

  .spinner.small {
    width: 10px;
    height: 10px;
    border-width: 1.5px;
  }
</style>
