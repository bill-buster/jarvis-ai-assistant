<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import { healthStore, fetchHealth } from "../stores/health";
  import { conversationsStore, fetchConversations } from "../stores/conversations";

  const dispatch = createEventDispatcher<{ navigate: string }>();

  onMount(() => {
    fetchHealth();
    fetchConversations();
  });

  $: totalMessages = $conversationsStore.conversations.reduce(
    (sum, c) => sum + c.message_count,
    0
  );
</script>

<div class="dashboard">
  <h1>Dashboard</h1>

  <div class="cards">
    <button class="card" on:click={() => dispatch("navigate", "messages")}>
      <div class="card-icon messages">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </div>
      <div class="card-content">
        <h3>Conversations</h3>
        <p class="stat">{$conversationsStore.conversations.length}</p>
        <p class="sub">{totalMessages.toLocaleString()} total messages</p>
      </div>
    </button>

    <button class="card" on:click={() => dispatch("navigate", "health")}>
      <div class="card-icon health" class:healthy={$healthStore.data?.status === "healthy"}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
      </div>
      <div class="card-content">
        <h3>System Health</h3>
        <p class="stat" class:healthy={$healthStore.data?.status === "healthy"}>
          {$healthStore.data?.status || "Unknown"}
        </p>
        <p class="sub">
          {#if $healthStore.data}
            {$healthStore.data.memory_available_gb.toFixed(1)} GB available
          {:else}
            Checking...
          {/if}
        </p>
      </div>
    </button>

    <div class="card">
      <div class="card-icon model" class:loaded={$healthStore.data?.model_loaded}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
      </div>
      <div class="card-content">
        <h3>AI Model</h3>
        <p class="stat">{$healthStore.data?.model_loaded ? "Loaded" : "Not Loaded"}</p>
        <p class="sub">{$healthStore.data?.memory_mode || "FULL"} mode</p>
      </div>
    </div>

    <div class="card">
      <div class="card-icon imessage" class:connected={$healthStore.data?.imessage_access}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="5" y="2" width="14" height="20" rx="2" ry="2" />
          <line x1="12" y1="18" x2="12.01" y2="18" />
        </svg>
      </div>
      <div class="card-content">
        <h3>iMessage</h3>
        <p class="stat">{$healthStore.data?.imessage_access ? "Connected" : "Not Connected"}</p>
        <p class="sub">
          {#if $healthStore.data?.imessage_access}
            Full Disk Access granted
          {:else}
            Grant access in System Settings
          {/if}
        </p>
      </div>
    </div>
  </div>

  <div class="recent">
    <h2>Recent Conversations</h2>
    {#if $conversationsStore.conversations.length === 0}
      <p class="empty">No conversations yet</p>
    {:else}
      <div class="recent-list">
        {#each $conversationsStore.conversations.slice(0, 5) as conv (conv.chat_id)}
          <div class="recent-item">
            <div class="recent-avatar" class:group={conv.is_group}>
              {(conv.display_name || conv.participants[0] || "?").charAt(0).toUpperCase()}
            </div>
            <div class="recent-info">
              <span class="recent-name">
                {conv.display_name || conv.participants.join(", ")}
              </span>
              <span class="recent-preview">
                {conv.last_message_text || "No messages"}
              </span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  h1 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 24px;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }

  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    gap: 16px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }

  .card:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: var(--bg-active);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .card-icon svg {
    width: 24px;
    height: 24px;
  }

  .card-icon.messages {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .card-icon.health {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .card-icon.health.healthy {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-icon.model {
    background: rgba(88, 86, 214, 0.2);
    color: var(--group-color);
  }

  .card-icon.model.loaded {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-icon.imessage {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .card-icon.imessage.connected {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-content h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .card-content .stat {
    font-size: 24px;
    font-weight: 600;
    text-transform: capitalize;
  }

  .card-content .stat.healthy {
    color: #34c759;
  }

  .card-content .sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .recent {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .recent h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .recent-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .recent-item {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .recent-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
    color: white;
  }

  .recent-avatar.group {
    background: var(--group-color);
  }

  .recent-info {
    flex: 1;
    min-width: 0;
  }

  .recent-name {
    display: block;
    font-weight: 500;
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .recent-preview {
    display: block;
    font-size: 13px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .empty {
    color: var(--text-secondary);
    text-align: center;
    padding: 24px;
  }
</style>
