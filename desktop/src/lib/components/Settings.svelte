<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../api/client";
  import type {
    ModelInfo,
    SettingsResponse,
    GenerationSettings,
    BehaviorSettings,
  } from "../api/types";

  let settings: SettingsResponse | null = null;
  let models: ModelInfo[] = [];
  let loading = true;
  let saving = false;
  let error: string | null = null;
  let successMessage: string | null = null;

  // Local state for form
  let selectedModelId = "";
  let temperature = 0.7;
  let maxTokensReply = 150;
  let maxTokensSummary = 500;
  let autoSuggestReplies = true;
  let suggestionCount = 3;
  let contextMessagesReply = 20;
  let contextMessagesSummary = 50;

  // Download/activate state
  let downloadingModel: string | null = null;
  let activatingModel: string | null = null;

  onMount(async () => {
    await loadData();
  });

  async function loadData() {
    loading = true;
    error = null;
    try {
      const [settingsData, modelsData] = await Promise.all([
        api.getSettings(),
        api.getModels(),
      ]);
      settings = settingsData;
      models = modelsData;

      // Populate form with current settings
      selectedModelId = settings.model_id;
      temperature = settings.generation.temperature;
      maxTokensReply = settings.generation.max_tokens_reply;
      maxTokensSummary = settings.generation.max_tokens_summary;
      autoSuggestReplies = settings.behavior.auto_suggest_replies;
      suggestionCount = settings.behavior.suggestion_count;
      contextMessagesReply = settings.behavior.context_messages_reply;
      contextMessagesSummary = settings.behavior.context_messages_summary;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load settings";
    } finally {
      loading = false;
    }
  }

  async function saveSettings() {
    saving = true;
    error = null;
    successMessage = null;
    try {
      const generation: GenerationSettings = {
        temperature,
        max_tokens_reply: maxTokensReply,
        max_tokens_summary: maxTokensSummary,
      };
      const behavior: BehaviorSettings = {
        auto_suggest_replies: autoSuggestReplies,
        suggestion_count: suggestionCount,
        context_messages_reply: contextMessagesReply,
        context_messages_summary: contextMessagesSummary,
      };

      settings = await api.updateSettings({
        model_id: selectedModelId,
        generation,
        behavior,
      });
      successMessage = "Settings saved successfully";
      setTimeout(() => (successMessage = null), 3000);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to save settings";
    } finally {
      saving = false;
    }
  }

  async function resetToDefaults() {
    temperature = 0.7;
    maxTokensReply = 150;
    maxTokensSummary = 500;
    autoSuggestReplies = true;
    suggestionCount = 3;
    contextMessagesReply = 20;
    contextMessagesSummary = 50;
    await saveSettings();
  }

  async function downloadModel(modelId: string) {
    downloadingModel = modelId;
    error = null;
    try {
      const status = await api.downloadModel(modelId);
      if (status.status === "failed") {
        error = status.error || "Download failed";
      } else {
        // Refresh models to update download status
        models = await api.getModels();
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Download failed";
    } finally {
      downloadingModel = null;
    }
  }

  async function activateModel(modelId: string) {
    activatingModel = modelId;
    error = null;
    try {
      const response = await api.activateModel(modelId);
      if (!response.success) {
        error = response.error || "Activation failed";
      } else {
        selectedModelId = modelId;
        // Refresh data
        await loadData();
        successMessage = "Model activated successfully";
        setTimeout(() => (successMessage = null), 3000);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Activation failed";
    } finally {
      activatingModel = null;
    }
  }

  function getQualityLabel(tier: string): string {
    switch (tier) {
      case "basic":
        return "Basic";
      case "good":
        return "Good";
      case "best":
        return "Best";
      default:
        return tier;
    }
  }

  function getQualityClass(tier: string): string {
    return tier;
  }
</script>

<div class="settings">
  <h1>Settings</h1>

  {#if loading}
    <div class="loading">Loading settings...</div>
  {:else if error}
    <div class="error-banner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      <span>{error}</span>
      <button class="dismiss" on:click={() => (error = null)}>Dismiss</button>
    </div>
  {/if}

  {#if successMessage}
    <div class="success-banner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </svg>
      <span>{successMessage}</span>
    </div>
  {/if}

  {#if !loading}
    <section class="section">
      <h2>Model</h2>
      <p class="section-desc">Select an AI model for text generation</p>

      <div class="model-list">
        {#each models as model (model.model_id)}
          <label
            class="model-card"
            class:selected={selectedModelId === model.model_id}
            class:disabled={!model.is_downloaded && model.ram_requirement_gb > (settings?.system.system_ram_gb || 0)}
          >
            <input
              type="radio"
              name="model"
              value={model.model_id}
              bind:group={selectedModelId}
              disabled={!model.is_downloaded}
            />
            <div class="model-radio" />
            <div class="model-info">
              <div class="model-header">
                <span class="model-name">{model.name}</span>
                {#if model.is_recommended}
                  <span class="badge recommended">Recommended</span>
                {/if}
                {#if model.is_loaded}
                  <span class="badge loaded">Loaded</span>
                {/if}
              </div>
              <div class="model-details">
                <span class="model-size">{model.size_gb} GB</span>
                <span class="model-quality {getQualityClass(model.quality_tier)}">
                  {getQualityLabel(model.quality_tier)}
                </span>
                <span class="model-ram">Requires {model.ram_requirement_gb} GB RAM</span>
              </div>
              {#if model.description}
                <p class="model-desc">{model.description}</p>
              {/if}
              {#if model.ram_requirement_gb > (settings?.system.system_ram_gb || 0)}
                <p class="model-warning">
                  Your system has {settings?.system.system_ram_gb.toFixed(0)} GB RAM
                </p>
              {/if}
            </div>
            <div class="model-actions">
              {#if !model.is_downloaded}
                <button
                  class="btn-secondary"
                  on:click|stopPropagation={() => downloadModel(model.model_id)}
                  disabled={downloadingModel === model.model_id}
                >
                  {downloadingModel === model.model_id ? "Downloading..." : "Download"}
                </button>
              {:else if !model.is_loaded && selectedModelId === model.model_id}
                <button
                  class="btn-secondary"
                  on:click|stopPropagation={() => activateModel(model.model_id)}
                  disabled={activatingModel === model.model_id}
                >
                  {activatingModel === model.model_id ? "Activating..." : "Activate"}
                </button>
              {/if}
            </div>
          </label>
        {/each}
      </div>
    </section>

    <section class="section">
      <h2>Generation</h2>
      <p class="section-desc">Configure text generation parameters</p>

      <div class="form-group">
        <label for="temperature">
          Temperature
          <span class="value">{temperature.toFixed(1)}</span>
        </label>
        <input
          type="range"
          id="temperature"
          min="0.1"
          max="1.0"
          step="0.1"
          bind:value={temperature}
        />
        <p class="help">Lower = more focused, Higher = more creative</p>
      </div>

      <div class="form-group">
        <label for="maxTokensReply">
          Reply Max Tokens
          <span class="value">{maxTokensReply}</span>
        </label>
        <input
          type="range"
          id="maxTokensReply"
          min="50"
          max="300"
          step="10"
          bind:value={maxTokensReply}
        />
        <p class="help">Maximum length for reply suggestions (50-300)</p>
      </div>

      <div class="form-group">
        <label for="maxTokensSummary">
          Summary Max Tokens
          <span class="value">{maxTokensSummary}</span>
        </label>
        <input
          type="range"
          id="maxTokensSummary"
          min="200"
          max="1000"
          step="50"
          bind:value={maxTokensSummary}
        />
        <p class="help">Maximum length for conversation summaries (200-1000)</p>
      </div>
    </section>

    <section class="section">
      <h2>Suggestions</h2>
      <p class="section-desc">Configure reply suggestion behavior</p>

      <div class="form-group toggle">
        <label for="autoSuggest">Auto-suggest replies</label>
        <button
          class="toggle-btn"
          class:on={autoSuggestReplies}
          on:click={() => (autoSuggestReplies = !autoSuggestReplies)}
          role="switch"
          aria-checked={autoSuggestReplies}
        >
          <span class="toggle-slider" />
        </button>
      </div>

      {#if autoSuggestReplies}
        <div class="form-group">
          <label for="suggestionCount">
            Suggestions Count
            <span class="value">{suggestionCount}</span>
          </label>
          <input
            type="range"
            id="suggestionCount"
            min="1"
            max="5"
            step="1"
            bind:value={suggestionCount}
          />
          <p class="help">Number of reply suggestions to show (1-5)</p>
        </div>

        <div class="form-group">
          <label for="contextMessagesReply">
            Context Messages (Replies)
            <span class="value">{contextMessagesReply}</span>
          </label>
          <input
            type="range"
            id="contextMessagesReply"
            min="10"
            max="50"
            step="5"
            bind:value={contextMessagesReply}
          />
          <p class="help">Messages to include for reply context (10-50)</p>
        </div>

        <div class="form-group">
          <label for="contextMessagesSummary">
            Context Messages (Summaries)
            <span class="value">{contextMessagesSummary}</span>
          </label>
          <input
            type="range"
            id="contextMessagesSummary"
            min="20"
            max="100"
            step="10"
            bind:value={contextMessagesSummary}
          />
          <p class="help">Messages to include for summaries (20-100)</p>
        </div>
      {/if}
    </section>

    <section class="section">
      <h2>System</h2>
      <p class="section-desc">System information (read-only)</p>

      {#if settings}
        <div class="system-info">
          <div class="info-row">
            <span class="info-label">System RAM</span>
            <span class="info-value">{settings.system.system_ram_gb.toFixed(1)} GB</span>
          </div>
          <div class="info-row">
            <span class="info-label">Current Memory Usage</span>
            <span class="info-value">{settings.system.current_memory_usage_gb.toFixed(1)} GB</span>
          </div>
          <div class="info-row">
            <span class="info-label">Model Status</span>
            <span class="info-value" class:status-ok={settings.system.model_loaded}>
              {settings.system.model_loaded
                ? `Loaded (${settings.system.model_memory_usage_gb.toFixed(1)} GB)`
                : "Not Loaded"}
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">iMessage Access</span>
            <span class="info-value" class:status-ok={settings.system.imessage_access}>
              {settings.system.imessage_access ? "Connected" : "Not Connected"}
            </span>
          </div>
        </div>
      {/if}
    </section>

    <div class="actions">
      <button class="btn-primary" on:click={saveSettings} disabled={saving}>
        {saving ? "Saving..." : "Save Settings"}
      </button>
      <button class="btn-secondary" on:click={resetToDefaults} disabled={saving}>
        Reset to Defaults
      </button>
    </div>
  {/if}
</div>

<style>
  .settings {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    max-width: 800px;
  }

  h1 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 24px;
  }

  .loading {
    text-align: center;
    color: var(--text-secondary);
    padding: 48px;
  }

  .error-banner,
  .success-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 24px;
  }

  .error-banner {
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
    color: var(--error-color);
  }

  .success-banner {
    background: rgba(52, 199, 89, 0.1);
    border: 1px solid #34c759;
    color: #34c759;
  }

  .error-banner svg,
  .success-banner svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .error-banner .dismiss {
    margin-left: auto;
    background: transparent;
    border: none;
    color: var(--error-color);
    cursor: pointer;
    text-decoration: underline;
  }

  .section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }

  .section h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
  }

  .section-desc {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 20px;
  }

  /* Model selection */
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .model-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: var(--bg-primary);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .model-card:hover:not(.disabled) {
    border-color: var(--accent-color);
  }

  .model-card.selected {
    border-color: var(--accent-color);
    background: rgba(11, 147, 246, 0.05);
  }

  .model-card.disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .model-card input[type="radio"] {
    display: none;
  }

  .model-radio {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-color);
    border-radius: 50%;
    flex-shrink: 0;
    position: relative;
    margin-top: 2px;
  }

  .model-card.selected .model-radio {
    border-color: var(--accent-color);
  }

  .model-card.selected .model-radio::after {
    content: "";
    position: absolute;
    top: 4px;
    left: 4px;
    width: 8px;
    height: 8px;
    background: var(--accent-color);
    border-radius: 50%;
  }

  .model-info {
    flex: 1;
  }

  .model-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    flex-wrap: wrap;
  }

  .model-name {
    font-weight: 600;
    font-size: 15px;
  }

  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .badge.recommended {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .badge.loaded {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .model-details {
    display: flex;
    gap: 12px;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .model-quality.basic {
    color: var(--text-secondary);
  }

  .model-quality.good {
    color: var(--accent-color);
  }

  .model-quality.best {
    color: #34c759;
  }

  .model-desc {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .model-warning {
    font-size: 12px;
    color: #ff9f0a;
    margin-top: 6px;
  }

  .model-actions {
    flex-shrink: 0;
  }

  /* Form groups */
  .form-group {
    margin-bottom: 20px;
  }

  .form-group:last-child {
    margin-bottom: 0;
  }

  .form-group label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 8px;
  }

  .form-group .value {
    color: var(--accent-color);
    font-weight: 600;
  }

  .form-group input[type="range"] {
    width: 100%;
    height: 6px;
    background: var(--bg-active);
    border-radius: 3px;
    appearance: none;
    cursor: pointer;
  }

  .form-group input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    background: var(--accent-color);
    border-radius: 50%;
    cursor: pointer;
  }

  .form-group .help {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 6px;
  }

  /* Toggle */
  .form-group.toggle {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .form-group.toggle label {
    margin-bottom: 0;
  }

  .toggle-btn {
    width: 48px;
    height: 28px;
    background: var(--bg-active);
    border: none;
    border-radius: 14px;
    cursor: pointer;
    position: relative;
    transition: background 0.2s ease;
  }

  .toggle-btn.on {
    background: var(--accent-color);
  }

  .toggle-slider {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 24px;
    height: 24px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s ease;
  }

  .toggle-btn.on .toggle-slider {
    transform: translateX(20px);
  }

  /* System info */
  .system-info {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .info-row:last-child {
    border-bottom: none;
  }

  .info-label {
    color: var(--text-secondary);
    font-size: 14px;
  }

  .info-value {
    font-weight: 500;
    font-size: 14px;
  }

  .info-value.status-ok {
    color: #34c759;
  }

  /* Actions */
  .actions {
    display: flex;
    gap: 12px;
    padding-top: 8px;
  }

  .btn-primary,
  .btn-secondary {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-primary {
    background: var(--accent-color);
    border: none;
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: #0a84e0;
  }

  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .btn-secondary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
