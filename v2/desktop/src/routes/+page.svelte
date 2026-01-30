<script lang="ts">
  import ConversationList from "$lib/components/ConversationList.svelte";
  import ContactProfilePanel from "$lib/components/ContactProfilePanel.svelte";
  import GenerationDebugPanel from "$lib/components/GenerationDebugPanel.svelte";
  import MessageList from "$lib/components/MessageList.svelte";
  import ReplySuggestions from "$lib/components/ReplySuggestions.svelte";
  import {
    appStore,
    selectedConversation,
    connectionStatus,
    unreadChats,
    selectConversation,
    generateReplies,
    sendMessage,
  } from "$lib/stores/app";

  let profileExpanded = false;

  function handleSelectConversation(chatId: string) {
    selectConversation(chatId);
  }

  function handleGenerateReplies() {
    generateReplies();
  }

  async function handleSendMessage(text: string): Promise<boolean> {
    return sendMessage(text);
  }

  function getConversationTitle(conv: typeof $selectedConversation): string {
    if (!conv) return "Select a conversation";
    // Use display_name if available
    if (conv.display_name) return conv.display_name;
    // For groups without display_name, join participant names
    if (conv.is_group && conv.participants.length > 1) {
      // Show first 3 names, then "+N more"
      const names = conv.participants.slice(0, 3);
      const more = conv.participants.length > 3 ? ` +${conv.participants.length - 3}` : "";
      return names.join(", ") + more;
    }
    return conv.participants[0] || "Unknown";
  }
</script>

<div class="app">
  <!-- Sidebar -->
  <div class="sidebar">
    <div class="header">
      <div class="status">
        <span
          class="status-dot"
          class:connected={$connectionStatus === "connected"}
          class:disconnected={$connectionStatus === "disconnected"}
          class:connecting={$connectionStatus === "connecting"}
        ></span>
        <h1>Messages</h1>
        {#if $appStore.wsConnected}
          <span class="ws-badge" title="Real-time updates active">LIVE</span>
        {/if}
      </div>
    </div>

    {#if $appStore.loading && $appStore.conversations.length === 0}
      <div class="loading">
        <div class="spinner"></div>
      </div>
    {:else}
      <ConversationList
        conversations={$appStore.conversations}
        selectedChatId={$appStore.selectedChatId}
        unreadChats={$unreadChats}
        onSelect={handleSelectConversation}
      />
    {/if}
  </div>

  <!-- Main content -->
  <div class="main">
    <div class="header">
      <h1>{getConversationTitle($selectedConversation)}</h1>
    </div>

    {#if $appStore.selectedChatId}
      <ContactProfilePanel
        profile={$appStore.contactProfile}
        loading={$appStore.loadingProfile}
        bind:expanded={profileExpanded}
      />

      <MessageList messages={$appStore.messages} loading={$appStore.loadingMessages} />

      <ReplySuggestions
        replies={$appStore.replies}
        streamingReplies={$appStore.streamingReplies}
        loading={$appStore.loadingReplies}
        isStreaming={$appStore.isStreaming}
        chatId={$appStore.selectedChatId}
        onGenerate={handleGenerateReplies}
        onSend={handleSendMessage}
      />

      <GenerationDebugPanel
        debug={$appStore.generationDebug}
        generationTimeMs={$appStore.generationTimeMs}
      />
    {:else}
      <div class="empty-state">
        <p>Select a conversation to view messages</p>
      </div>
    {/if}
  </div>
</div>

{#if $appStore.error}
  <button class="error-toast" on:click={() => appStore.update((s) => ({ ...s, error: null }))}>
    {$appStore.error}
  </button>
{/if}

<style>
  .app {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  .sidebar {
    width: 320px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .header {
    height: 56px;
    padding: 0 16px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
  }

  .header h1 {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .ws-badge {
    font-size: 9px;
    font-weight: 700;
    background: var(--accent-green);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    animation: pulse 2s infinite;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
  }

  .status-dot.connected {
    background: var(--accent-green);
  }

  .status-dot.disconnected {
    background: #ff453a;
  }

  .status-dot.connecting {
    background: #ff9f0a;
    animation: pulse 1s infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  .loading,
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: var(--text-secondary);
  }

  .spinner {
    width: 24px;
    height: 24px;
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

  .error-toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: #ff453a;
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    z-index: 1000;
  }
</style>
