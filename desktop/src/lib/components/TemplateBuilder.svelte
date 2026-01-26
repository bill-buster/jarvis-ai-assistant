<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../api/client";
  import type {
    CustomTemplate,
    CustomTemplateCreateRequest,
    CustomTemplateListResponse,
    CustomTemplateTestResult,
    CustomTemplateUsageStats,
  } from "../api/types";

  // State
  let templates: CustomTemplate[] = $state([]);
  let categories: string[] = $state([]);
  let allTags: string[] = $state([]);
  let usageStats: CustomTemplateUsageStats | null = $state(null);
  let loading = $state(true);
  let error: string | null = $state(null);

  // Filters
  let filterCategory = $state("");
  let filterTag = $state("");
  let filterEnabled = $state(false);

  // Edit mode
  let editingTemplate: CustomTemplate | null = $state(null);
  let showEditor = $state(false);

  // Form state
  let formName = $state("");
  let formTemplateText = $state("");
  let formTriggerPhrases = $state("");
  let formCategory = $state("general");
  let formTags = $state("");
  let formMinGroupSize = $state<number | null>(null);
  let formMaxGroupSize = $state<number | null>(null);
  let formEnabled = $state(true);

  // Test mode
  let showTester = $state(false);
  let testInputs = $state("");
  let testResults: CustomTemplateTestResult[] = $state([]);
  let testMatchRate = $state(0);
  let testThreshold = $state(0.7);
  let testing = $state(false);

  // Import/Export
  let showImportExport = $state(false);
  let importData = $state("");
  let importing = $state(false);
  let exporting = $state(false);

  // Tab state
  type TabType = "list" | "stats";
  let activeTab: TabType = $state("list");

  // Saving state
  let saving = $state(false);

  onMount(async () => {
    await loadTemplates();
    await loadUsageStats();
  });

  async function loadTemplates() {
    loading = true;
    error = null;
    try {
      const response: CustomTemplateListResponse = await api.getCustomTemplates(
        filterCategory || undefined,
        filterTag || undefined,
        filterEnabled
      );
      templates = response.templates;
      categories = response.categories;
      allTags = response.tags;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load templates";
    } finally {
      loading = false;
    }
  }

  async function loadUsageStats() {
    try {
      usageStats = await api.getCustomTemplateUsageStats();
    } catch (e) {
      console.error("Failed to load usage stats:", e);
    }
  }

  function openEditor(template?: CustomTemplate) {
    if (template) {
      editingTemplate = template;
      formName = template.name;
      formTemplateText = template.template_text;
      formTriggerPhrases = template.trigger_phrases.join("\n");
      formCategory = template.category;
      formTags = template.tags.join(", ");
      formMinGroupSize = template.min_group_size;
      formMaxGroupSize = template.max_group_size;
      formEnabled = template.enabled;
    } else {
      editingTemplate = null;
      formName = "";
      formTemplateText = "";
      formTriggerPhrases = "";
      formCategory = "general";
      formTags = "";
      formMinGroupSize = null;
      formMaxGroupSize = null;
      formEnabled = true;
    }
    showEditor = true;
    showTester = false;
  }

  function closeEditor() {
    showEditor = false;
    editingTemplate = null;
    testResults = [];
  }

  async function saveTemplate() {
    const triggerPhrases = formTriggerPhrases
      .split("\n")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    if (!formName.trim()) {
      error = "Template name is required";
      return;
    }
    if (!formTemplateText.trim()) {
      error = "Template response text is required";
      return;
    }
    if (triggerPhrases.length === 0) {
      error = "At least one trigger phrase is required";
      return;
    }

    const tags = formTags
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    saving = true;
    error = null;

    try {
      if (editingTemplate) {
        await api.updateCustomTemplate(editingTemplate.id, {
          name: formName.trim(),
          template_text: formTemplateText.trim(),
          trigger_phrases: triggerPhrases,
          category: formCategory,
          tags,
          min_group_size: formMinGroupSize,
          max_group_size: formMaxGroupSize,
          enabled: formEnabled,
        });
      } else {
        const request: CustomTemplateCreateRequest = {
          name: formName.trim(),
          template_text: formTemplateText.trim(),
          trigger_phrases: triggerPhrases,
          category: formCategory,
          tags,
          min_group_size: formMinGroupSize,
          max_group_size: formMaxGroupSize,
          enabled: formEnabled,
        };
        await api.createCustomTemplate(request);
      }
      closeEditor();
      await loadTemplates();
      await loadUsageStats();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to save template";
    } finally {
      saving = false;
    }
  }

  async function deleteTemplate(template: CustomTemplate) {
    if (!confirm(`Delete template "${template.name}"?`)) return;

    try {
      await api.deleteCustomTemplate(template.id);
      await loadTemplates();
      await loadUsageStats();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to delete template";
    }
  }

  async function toggleEnabled(template: CustomTemplate) {
    try {
      await api.updateCustomTemplate(template.id, {
        enabled: !template.enabled,
      });
      await loadTemplates();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to update template";
    }
  }

  function openTester() {
    showTester = true;
    testResults = [];
  }

  async function runTest() {
    const triggerPhrases = formTriggerPhrases
      .split("\n")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    const inputs = testInputs
      .split("\n")
      .map((i) => i.trim())
      .filter((i) => i.length > 0);

    if (triggerPhrases.length === 0) {
      error = "Add trigger phrases first";
      return;
    }
    if (inputs.length === 0) {
      error = "Enter test inputs";
      return;
    }

    testing = true;
    error = null;

    try {
      const response = await api.testCustomTemplate({
        trigger_phrases: triggerPhrases,
        test_inputs: inputs,
      });
      testResults = response.results;
      testMatchRate = response.match_rate;
      testThreshold = response.threshold;
    } catch (e) {
      error = e instanceof Error ? e.message : "Test failed";
    } finally {
      testing = false;
    }
  }

  async function exportTemplates() {
    exporting = true;
    error = null;

    try {
      const response = await api.exportCustomTemplates();
      const blob = new Blob([JSON.stringify(response, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `jarvis-templates-${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      error = e instanceof Error ? e.message : "Export failed";
    } finally {
      exporting = false;
    }
  }

  async function importTemplates() {
    if (!importData.trim()) {
      error = "Paste export data to import";
      return;
    }

    importing = true;
    error = null;

    try {
      const data = JSON.parse(importData);
      const response = await api.importCustomTemplates({
        data,
        overwrite: false,
      });
      alert(
        `Imported ${response.imported} templates (${response.errors} errors)`
      );
      importData = "";
      showImportExport = false;
      await loadTemplates();
      await loadUsageStats();
    } catch (e) {
      error = e instanceof Error ? e.message : "Import failed";
    } finally {
      importing = false;
    }
  }

  function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      importData = e.target?.result as string;
    };
    reader.readAsText(file);
  }
</script>

<div class="template-builder">
  <header class="header">
    <div class="header-left">
      <h1>Template Builder</h1>
      <p class="subtitle">Create and manage custom response templates</p>
    </div>
    <div class="header-actions">
      <button class="btn-secondary" onclick={() => (showImportExport = true)}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        Import/Export
      </button>
      <button class="btn-primary" onclick={() => openEditor()}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
        New Template
      </button>
    </div>
  </header>

  <!-- Tabs -->
  <div class="tabs">
    <button
      class="tab"
      class:active={activeTab === "list"}
      onclick={() => (activeTab = "list")}
    >
      Templates ({templates.length})
    </button>
    <button
      class="tab"
      class:active={activeTab === "stats"}
      onclick={() => (activeTab = "stats")}
    >
      Usage Stats
    </button>
  </div>

  {#if error}
    <div class="error-banner">
      <span>{error}</span>
      <button onclick={() => (error = null)}>Dismiss</button>
    </div>
  {/if}

  {#if activeTab === "list"}
    <!-- Filters -->
    <div class="filters">
      <div class="filter-group">
        <label>Category</label>
        <select bind:value={filterCategory} onchange={() => loadTemplates()}>
          <option value="">All Categories</option>
          {#each categories as cat}
            <option value={cat}>{cat}</option>
          {/each}
        </select>
      </div>
      <div class="filter-group">
        <label>Tag</label>
        <select bind:value={filterTag} onchange={() => loadTemplates()}>
          <option value="">All Tags</option>
          {#each allTags as tag}
            <option value={tag}>{tag}</option>
          {/each}
        </select>
      </div>
      <div class="filter-group checkbox">
        <label>
          <input
            type="checkbox"
            bind:checked={filterEnabled}
            onchange={() => loadTemplates()}
          />
          Enabled only
        </label>
      </div>
    </div>

    <!-- Template List -->
    {#if loading}
      <div class="loading">Loading templates...</div>
    {:else if templates.length === 0}
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <line x1="9" y1="15" x2="15" y2="15" />
        </svg>
        <h3>No templates yet</h3>
        <p>Create your first custom template to get started</p>
        <button class="btn-primary" onclick={() => openEditor()}>
          Create Template
        </button>
      </div>
    {:else}
      <div class="template-list">
        {#each templates as template}
          <div class="template-card" class:disabled={!template.enabled}>
            <div class="template-header">
              <div class="template-info">
                <h3>{template.name}</h3>
                <span class="category-badge">{template.category}</span>
                {#if !template.enabled}
                  <span class="disabled-badge">Disabled</span>
                {/if}
              </div>
              <div class="template-actions">
                <button
                  class="icon-btn"
                  title={template.enabled ? "Disable" : "Enable"}
                  onclick={() => toggleEnabled(template)}
                >
                  {#if template.enabled}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  {:else}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  {/if}
                </button>
                <button
                  class="icon-btn"
                  title="Edit"
                  onclick={() => openEditor(template)}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                </button>
                <button
                  class="icon-btn danger"
                  title="Delete"
                  onclick={() => deleteTemplate(template)}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            </div>
            <div class="template-body">
              <div class="response-preview">
                <span class="label">Response:</span>
                <p>{template.template_text}</p>
              </div>
              <div class="triggers-preview">
                <span class="label">Triggers ({template.trigger_phrases.length}):</span>
                <div class="trigger-chips">
                  {#each template.trigger_phrases.slice(0, 3) as phrase}
                    <span class="trigger-chip">{phrase}</span>
                  {/each}
                  {#if template.trigger_phrases.length > 3}
                    <span class="trigger-chip more">+{template.trigger_phrases.length - 3} more</span>
                  {/if}
                </div>
              </div>
              {#if template.tags.length > 0}
                <div class="tags-preview">
                  {#each template.tags as tag}
                    <span class="tag-chip">{tag}</span>
                  {/each}
                </div>
              {/if}
            </div>
            <div class="template-footer">
              <span class="usage">Used {template.usage_count} times</span>
              <span class="updated">Updated {new Date(template.updated_at).toLocaleDateString()}</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {:else if activeTab === "stats"}
    <!-- Usage Stats -->
    {#if usageStats}
      <div class="stats-grid">
        <div class="stat-card">
          <h3>Total Templates</h3>
          <span class="stat-value">{usageStats.total_templates}</span>
        </div>
        <div class="stat-card">
          <h3>Enabled</h3>
          <span class="stat-value">{usageStats.enabled_templates}</span>
        </div>
        <div class="stat-card">
          <h3>Total Usage</h3>
          <span class="stat-value">{usageStats.total_usage}</span>
        </div>
      </div>

      {#if Object.keys(usageStats.usage_by_category).length > 0}
        <div class="stats-section">
          <h3>Usage by Category</h3>
          <div class="category-stats">
            {#each Object.entries(usageStats.usage_by_category) as [cat, count]}
              <div class="category-stat">
                <span class="category-name">{cat}</span>
                <span class="category-count">{count}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if usageStats.top_templates.length > 0}
        <div class="stats-section">
          <h3>Top Templates</h3>
          <div class="top-templates">
            {#each usageStats.top_templates as t, i}
              <div class="top-template">
                <span class="rank">#{i + 1}</span>
                <span class="name">{t.name}</span>
                <span class="count">{t.usage_count} uses</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {:else}
      <div class="loading">Loading stats...</div>
    {/if}
  {/if}
</div>

<!-- Editor Modal -->
{#if showEditor}
  <div class="modal-overlay" onclick={closeEditor}>
    <div class="modal editor-modal" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h2>{editingTemplate ? "Edit Template" : "New Template"}</h2>
        <button class="close-btn" onclick={closeEditor}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div class="modal-body">
        <div class="form-row">
          <div class="form-group">
            <label for="name">Template Name *</label>
            <input
              type="text"
              id="name"
              bind:value={formName}
              placeholder="e.g., Work Acknowledgment"
            />
          </div>
          <div class="form-group">
            <label for="category">Category</label>
            <input
              type="text"
              id="category"
              bind:value={formCategory}
              placeholder="e.g., work, personal, casual"
              list="category-suggestions"
            />
            <datalist id="category-suggestions">
              {#each categories as cat}
                <option value={cat} />
              {/each}
              <option value="general" />
              <option value="work" />
              <option value="personal" />
              <option value="casual" />
            </datalist>
          </div>
        </div>

        <div class="form-group">
          <label for="response">Response Text *</label>
          <textarea
            id="response"
            bind:value={formTemplateText}
            placeholder="The text that will be suggested when this template matches"
            rows="3"
          ></textarea>
        </div>

        <div class="form-group">
          <label for="triggers">
            Trigger Phrases * <span class="hint">(one per line)</span>
          </label>
          <textarea
            id="triggers"
            bind:value={formTriggerPhrases}
            placeholder="got your update&#10;thanks for sending&#10;received the file"
            rows="4"
          ></textarea>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="tags">Tags <span class="hint">(comma-separated)</span></label>
            <input
              type="text"
              id="tags"
              bind:value={formTags}
              placeholder="professional, acknowledgment"
            />
          </div>
          <div class="form-group checkbox-group">
            <label>
              <input type="checkbox" bind:checked={formEnabled} />
              Enabled
            </label>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="minGroup">Min Group Size</label>
            <input
              type="number"
              id="minGroup"
              bind:value={formMinGroupSize}
              min="1"
              placeholder="Any"
            />
          </div>
          <div class="form-group">
            <label for="maxGroup">Max Group Size</label>
            <input
              type="number"
              id="maxGroup"
              bind:value={formMaxGroupSize}
              min="1"
              placeholder="Any"
            />
          </div>
        </div>

        <!-- Test Section -->
        <div class="test-section">
          <button class="btn-secondary" onclick={openTester}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="4 17 10 11 4 5" />
              <line x1="12" y1="19" x2="20" y2="19" />
            </svg>
            Test Template
          </button>

          {#if showTester}
            <div class="tester">
              <div class="form-group">
                <label for="test-inputs">
                  Test Inputs <span class="hint">(one per line)</span>
                </label>
                <textarea
                  id="test-inputs"
                  bind:value={testInputs}
                  placeholder="Enter sample messages to test against your triggers"
                  rows="3"
                ></textarea>
              </div>
              <button
                class="btn-primary"
                onclick={runTest}
                disabled={testing}
              >
                {testing ? "Testing..." : "Run Test"}
              </button>

              {#if testResults.length > 0}
                <div class="test-results">
                  <div class="test-summary">
                    Match rate: <strong>{(testMatchRate * 100).toFixed(0)}%</strong>
                    (threshold: {testThreshold})
                  </div>
                  {#each testResults as result}
                    <div class="test-result" class:matched={result.matched}>
                      <span class="input">"{result.input}"</span>
                      <span class="match">
                        {#if result.matched}
                          Matched: "{result.best_match}" ({(result.similarity * 100).toFixed(0)}%)
                        {:else}
                          No match (best: {(result.similarity * 100).toFixed(0)}%)
                        {/if}
                      </span>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </div>
      </div>

      <div class="modal-footer">
        <button class="btn-secondary" onclick={closeEditor}>Cancel</button>
        <button class="btn-primary" onclick={saveTemplate} disabled={saving}>
          {saving ? "Saving..." : editingTemplate ? "Update" : "Create"}
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- Import/Export Modal -->
{#if showImportExport}
  <div class="modal-overlay" onclick={() => (showImportExport = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h2>Import/Export Templates</h2>
        <button class="close-btn" onclick={() => (showImportExport = false)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div class="modal-body">
        <div class="import-export-section">
          <h3>Export</h3>
          <p>Download all templates as a JSON file to share or backup.</p>
          <button
            class="btn-primary"
            onclick={exportTemplates}
            disabled={exporting}
          >
            {exporting ? "Exporting..." : "Export All Templates"}
          </button>
        </div>

        <div class="divider"></div>

        <div class="import-export-section">
          <h3>Import</h3>
          <p>Import templates from a JSON file or paste export data.</p>
          <input
            type="file"
            accept=".json"
            onchange={handleFileUpload}
            class="file-input"
          />
          <textarea
            bind:value={importData}
            placeholder="Or paste exported JSON data here..."
            rows="6"
          ></textarea>
          <button
            class="btn-primary"
            onclick={importTemplates}
            disabled={importing || !importData.trim()}
          >
            {importing ? "Importing..." : "Import Templates"}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .template-builder {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 24px;
    overflow: auto;
    background: var(--bg-primary);
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
  }

  .header h1 {
    font-size: 24px;
    margin: 0;
  }

  .subtitle {
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .header-actions {
    display: flex;
    gap: 12px;
  }

  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0;
  }

  .tab {
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s ease;
    margin-bottom: -1px;
  }

  .tab:hover {
    color: var(--text-primary);
  }

  .tab.active {
    color: var(--accent-color);
    border-bottom-color: var(--accent-color);
  }

  .filters {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .filter-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .filter-group.checkbox {
    flex-direction: row;
    align-items: center;
  }

  .filter-group.checkbox label {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .filter-group label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .filter-group select {
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
  }

  .btn-primary, .btn-secondary {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-primary {
    background: var(--accent-color);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  .btn-secondary:hover {
    background: var(--bg-hover);
  }

  .btn-primary svg, .btn-secondary svg {
    width: 16px;
    height: 16px;
  }

  .error-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid var(--error-color);
    border-radius: 8px;
    margin-bottom: 16px;
    color: var(--error-color);
  }

  .error-banner button {
    background: transparent;
    border: none;
    color: var(--error-color);
    cursor: pointer;
    text-decoration: underline;
  }

  .loading, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-secondary);
  }

  .empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state h3 {
    margin: 0 0 8px 0;
    color: var(--text-primary);
  }

  .empty-state p {
    margin-bottom: 20px;
  }

  .template-list {
    display: grid;
    gap: 16px;
  }

  .template-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.15s ease;
  }

  .template-card:hover {
    border-color: var(--accent-color);
  }

  .template-card.disabled {
    opacity: 0.6;
  }

  .template-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .template-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .template-info h3 {
    margin: 0;
    font-size: 16px;
  }

  .category-badge {
    padding: 4px 8px;
    background: var(--accent-color);
    color: white;
    border-radius: 4px;
    font-size: 11px;
    text-transform: uppercase;
  }

  .disabled-badge {
    padding: 4px 8px;
    background: var(--bg-hover);
    color: var(--text-secondary);
    border-radius: 4px;
    font-size: 11px;
  }

  .template-actions {
    display: flex;
    gap: 8px;
  }

  .icon-btn {
    width: 32px;
    height: 32px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .icon-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .icon-btn.danger:hover {
    background: rgba(255, 95, 87, 0.1);
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .icon-btn svg {
    width: 16px;
    height: 16px;
  }

  .template-body {
    padding: 16px;
  }

  .response-preview, .triggers-preview {
    margin-bottom: 12px;
  }

  .label {
    font-size: 12px;
    color: var(--text-secondary);
    display: block;
    margin-bottom: 4px;
  }

  .response-preview p {
    margin: 0;
    font-style: italic;
  }

  .trigger-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .trigger-chip {
    padding: 4px 10px;
    background: var(--bg-hover);
    border-radius: 12px;
    font-size: 12px;
  }

  .trigger-chip.more {
    color: var(--text-secondary);
  }

  .tags-preview {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag-chip {
    padding: 2px 8px;
    background: var(--bg-active);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .template-footer {
    display: flex;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-hover);
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Stats */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }

  .stat-card h3 {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .stat-value {
    font-size: 36px;
    font-weight: bold;
    color: var(--accent-color);
  }

  .stats-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }

  .stats-section h3 {
    margin: 0 0 16px 0;
    font-size: 14px;
  }

  .category-stats, .top-templates {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .category-stat, .top-template {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .category-stat:last-child, .top-template:last-child {
    border-bottom: none;
  }

  .top-template .rank {
    width: 30px;
    color: var(--text-secondary);
  }

  .top-template .name {
    flex: 1;
  }

  .top-template .count {
    color: var(--text-secondary);
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    width: 90%;
    max-width: 600px;
    max-height: 90vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .editor-modal {
    max-width: 700px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 18px;
  }

  .close-btn {
    width: 32px;
    height: 32px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 6px;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .close-btn svg {
    width: 20px;
    height: 20px;
  }

  .modal-body {
    padding: 24px;
    overflow-y: auto;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    border-top: 1px solid var(--border-color);
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
  }

  .form-group .hint {
    font-weight: normal;
    color: var(--text-secondary);
  }

  .form-group input[type="text"],
  .form-group input[type="number"],
  .form-group textarea,
  .form-group select {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
  }

  .form-group textarea {
    resize: vertical;
  }

  .form-group input:focus,
  .form-group textarea:focus,
  .form-group select:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .checkbox-group {
    display: flex;
    align-items: center;
    padding-top: 28px;
  }

  .checkbox-group label {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0;
  }

  /* Test Section */
  .test-section {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
  }

  .tester {
    margin-top: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .test-results {
    margin-top: 16px;
  }

  .test-summary {
    margin-bottom: 12px;
    padding: 8px 12px;
    background: var(--bg-hover);
    border-radius: 6px;
  }

  .test-result {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    margin-bottom: 4px;
    border-radius: 6px;
    font-size: 13px;
  }

  .test-result.matched {
    background: rgba(52, 199, 89, 0.1);
    border: 1px solid rgba(52, 199, 89, 0.3);
  }

  .test-result:not(.matched) {
    background: rgba(255, 95, 87, 0.1);
    border: 1px solid rgba(255, 95, 87, 0.3);
  }

  .test-result .input {
    flex: 1;
    font-style: italic;
  }

  .test-result .match {
    color: var(--text-secondary);
  }

  /* Import/Export */
  .import-export-section {
    margin-bottom: 20px;
  }

  .import-export-section h3 {
    margin: 0 0 8px 0;
  }

  .import-export-section p {
    margin: 0 0 12px 0;
    color: var(--text-secondary);
    font-size: 13px;
  }

  .divider {
    height: 1px;
    background: var(--border-color);
    margin: 20px 0;
  }

  .file-input {
    display: block;
    width: 100%;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--bg-secondary);
    border: 1px dashed var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
  }

  .import-export-section textarea {
    width: 100%;
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-family: monospace;
    font-size: 12px;
    margin-bottom: 12px;
    resize: vertical;
  }
</style>
