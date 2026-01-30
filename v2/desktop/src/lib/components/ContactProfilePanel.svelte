<script lang="ts">
  import type { ContactProfile } from "$lib/api/types";

  export let profile: ContactProfile | null = null;
  export let loading: boolean = false;
  export let expanded: boolean = false;

  function formatRelationship(type: string): string {
    const labels: Record<string, string> = {
      close_friend: "Close Friend",
      family: "Family",
      coworker: "Work Contact",
      acquaintance: "Acquaintance",
      service: "Service/Business",
      unknown: "Unknown",
    };
    return labels[type] || type;
  }

  function formatNumber(n: number): string {
    if (n >= 1000) {
      return (n / 1000).toFixed(1) + "k";
    }
    return n.toString();
  }

  function getToneEmoji(tone: string, playful: boolean): string {
    if (playful) return "";
    if (tone === "casual") return "";
    if (tone === "formal") return "";
    return "";
  }
</script>

{#if loading}
  <div class="profile-loading">
    <div class="spinner"></div>
  </div>
{:else if profile && profile.total_messages > 0}
  <div class="profile-panel" class:expanded>
    <!-- Compact view (always shown) -->
    <button class="profile-summary" on:click={() => (expanded = !expanded)}>
      <div class="badges">
        <span class="badge relationship">{formatRelationship(profile.relationship_type)}</span>
        <span class="badge tone">{profile.tone}{profile.is_playful ? " (playful)" : ""}</span>
        {#if !profile.uses_emoji}
          <span class="badge no-emoji">no emoji</span>
        {/if}
        {#if profile.uses_slang}
          <span class="badge slang">casual</span>
        {/if}
      </div>
      <span class="message-count">{formatNumber(profile.total_messages)} msgs</span>
      <span class="expand-icon">{expanded ? "" : ""}</span>
    </button>

    <!-- Expanded view -->
    {#if expanded}
      <div class="profile-details">
        <div class="stat-row">
          <div class="stat">
            <span class="stat-value">{formatNumber(profile.you_sent)}</span>
            <span class="stat-label">you sent</span>
          </div>
          <div class="stat">
            <span class="stat-value">{formatNumber(profile.they_sent)}</span>
            <span class="stat-label">they sent</span>
          </div>
          <div class="stat">
            <span class="stat-value">{Math.round(profile.avg_your_length)}</span>
            <span class="stat-label">your avg len</span>
          </div>
        </div>

        {#if profile.topics.length > 0}
          <div class="topics">
            <span class="topics-label">Topics:</span>
            {#each profile.topics.slice(0, 3) as topic}
              <span class="topic-tag">{topic.name.toLowerCase()}</span>
            {/each}
          </div>
        {/if}

        {#if profile.their_common_phrases.length > 0}
          <div class="phrases">
            <span class="phrases-label">They say:</span>
            <span class="phrase-list">{profile.their_common_phrases.slice(0, 3).join(", ")}</span>
          </div>
        {/if}

        <p class="summary">{profile.summary}</p>
      </div>
    {/if}
  </div>
{/if}

<style>
  .profile-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
  }

  .spinner {
    width: 16px;
    height: 16px;
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

  .profile-panel {
    background: var(--bg-tertiary, rgba(0, 0, 0, 0.2));
    border-radius: 8px;
    margin: 0 16px 8px 16px;
    overflow: hidden;
  }

  .profile-summary {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    width: 100%;
    background: none;
    border: none;
    color: var(--text-primary);
    cursor: pointer;
    font-size: 12px;
  }

  .profile-summary:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .badges {
    display: flex;
    gap: 6px;
    flex: 1;
    flex-wrap: wrap;
  }

  .badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
  }

  .badge.relationship {
    background: var(--accent-blue);
    color: white;
  }

  .badge.tone {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .badge.no-emoji {
    background: rgba(255, 159, 10, 0.2);
    color: #ff9f0a;
  }

  .badge.slang {
    background: rgba(175, 82, 222, 0.2);
    color: #af52de;
  }

  .message-count {
    color: var(--text-secondary);
    font-size: 11px;
  }

  .expand-icon {
    color: var(--text-secondary);
    font-size: 10px;
  }

  .profile-details {
    padding: 0 12px 12px 12px;
    border-top: 1px solid var(--border-color);
  }

  .stat-row {
    display: flex;
    gap: 16px;
    padding: 12px 0;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .topics,
  .phrases {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .topics-label,
  .phrases-label {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .topic-tag {
    padding: 3px 10px;
    background: rgba(88, 86, 214, 0.3);
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    color: #a5a4ff;
  }

  .phrase-list {
    font-size: 11px;
    color: var(--text-primary);
    font-style: italic;
  }

  .summary {
    font-size: 11px;
    color: var(--text-secondary);
    margin: 8px 0 0 0;
    line-height: 1.4;
  }
</style>
