<script lang="ts">
  import { onMount } from "svelte";
  import { api, APIError } from "../api/client";
  import type {
    Experiment,
    ExperimentResults,
    VariantResults,
  } from "../api/types";

  // State
  let experiments: Experiment[] = [];
  let selectedExperiment: Experiment | null = null;
  let results: ExperimentResults | null = null;
  let loading = true;
  let loadingResults = false;
  let error: string | null = null;

  // Fetch experiments on mount
  onMount(() => {
    fetchExperiments();
  });

  async function fetchExperiments() {
    loading = true;
    error = null;
    try {
      const response = await api.getExperiments();
      experiments = response.experiments;
      if (experiments.length > 0 && !selectedExperiment) {
        selectExperiment(experiments[0]);
      }
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      } else {
        error = "Failed to load experiments";
      }
    } finally {
      loading = false;
    }
  }

  async function selectExperiment(exp: Experiment) {
    selectedExperiment = exp;
    loadingResults = true;
    error = null;
    try {
      results = await api.getExperimentResults(exp.name);
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      } else {
        error = "Failed to load results";
      }
      results = null;
    } finally {
      loadingResults = false;
    }
  }

  async function toggleExperiment(exp: Experiment) {
    try {
      await api.updateExperiment(exp.name, !exp.enabled);
      await fetchExperiments();
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      }
    }
  }

  async function clearOutcomes() {
    if (!selectedExperiment) return;
    if (!confirm(`Clear all outcomes for "${selectedExperiment.name}"?`)) return;

    try {
      await api.clearExperimentOutcomes(selectedExperiment.name);
      await selectExperiment(selectedExperiment);
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      }
    }
  }

  function getStatusClass(variant: VariantResults): string {
    if (variant.total_impressions === 0) return "";
    if (variant.conversion_rate >= 50) return "good";
    if (variant.conversion_rate >= 30) return "moderate";
    return "low";
  }

  function formatPercent(value: number): string {
    return value.toFixed(1) + "%";
  }

  function formatPValue(value: number | null): string {
    if (value === null) return "N/A";
    if (value < 0.001) return "< 0.001";
    return value.toFixed(4);
  }
</script>

<div class="experiment-dashboard">
  <header class="dashboard-header">
    <h1>A/B Testing Dashboard</h1>
    <button class="refresh-btn" on:click={fetchExperiments} disabled={loading}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
      </svg>
      Refresh
    </button>
  </header>

  {#if error}
    <div class="error-banner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      {error}
    </div>
  {/if}

  {#if loading}
    <div class="loading">Loading experiments...</div>
  {:else if experiments.length === 0}
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
      <h2>No Experiments</h2>
      <p>Create experiments in ~/.jarvis/experiments.yaml to get started.</p>
    </div>
  {:else}
    <div class="dashboard-content">
      <!-- Experiment List -->
      <aside class="experiment-list">
        <h2>Experiments</h2>
        <ul>
          {#each experiments as exp (exp.name)}
            <li>
              <button
                class="experiment-item"
                class:selected={selectedExperiment?.name === exp.name}
                class:disabled={!exp.enabled}
                on:click={() => selectExperiment(exp)}
              >
                <span class="experiment-name">{exp.name}</span>
                <span class="experiment-status" class:active={exp.enabled}>
                  {exp.enabled ? "Active" : "Inactive"}
                </span>
              </button>
              <button
                class="toggle-btn"
                title={exp.enabled ? "Disable experiment" : "Enable experiment"}
                on:click|stopPropagation={() => toggleExperiment(exp)}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  {#if exp.enabled}
                    <path d="M18 6L6 18M6 6l12 12" />
                  {:else}
                    <path d="M5 13l4 4L19 7" />
                  {/if}
                </svg>
              </button>
            </li>
          {/each}
        </ul>
      </aside>

      <!-- Results Panel -->
      <main class="results-panel">
        {#if selectedExperiment}
          <div class="experiment-header">
            <div>
              <h2>{selectedExperiment.name}</h2>
              {#if selectedExperiment.description}
                <p class="description">{selectedExperiment.description}</p>
              {/if}
            </div>
            <div class="header-actions">
              <button class="clear-btn" on:click={clearOutcomes}>
                Clear Data
              </button>
            </div>
          </div>

          <!-- Variants -->
          <section class="variants-section">
            <h3>Variants ({selectedExperiment.variants.length})</h3>
            <div class="variants-grid">
              {#each selectedExperiment.variants as variant}
                <div class="variant-card">
                  <div class="variant-header">
                    <span class="variant-id">{variant.id}</span>
                    <span class="variant-weight">{variant.weight}%</span>
                  </div>
                  {#if Object.keys(variant.config).length > 0}
                    <div class="variant-config">
                      {#each Object.entries(variant.config) as [key, value]}
                        <div class="config-item">
                          <span class="config-key">{key}:</span>
                          <span class="config-value">{JSON.stringify(value)}</span>
                        </div>
                      {/each}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </section>

          <!-- Results -->
          {#if loadingResults}
            <div class="loading">Loading results...</div>
          {:else if results}
            <section class="results-section">
              <h3>Results</h3>

              <!-- Summary Stats -->
              <div class="stats-grid">
                <div class="stat-card">
                  <span class="stat-label">Total Outcomes</span>
                  <span class="stat-value">{results.total_outcomes}</span>
                </div>
                <div class="stat-card">
                  <span class="stat-label">Leading Variant</span>
                  <span class="stat-value winner">{results.winner || "N/A"}</span>
                </div>
                <div class="stat-card" class:significant={results.is_significant}>
                  <span class="stat-label">Statistical Significance</span>
                  <span class="stat-value">{results.is_significant ? "Yes" : "No"}</span>
                </div>
                <div class="stat-card">
                  <span class="stat-label">P-Value</span>
                  <span class="stat-value">{formatPValue(results.p_value)}</span>
                </div>
              </div>

              <!-- Conversion Rates Table -->
              <div class="results-table-container">
                <table class="results-table">
                  <thead>
                    <tr>
                      <th>Variant</th>
                      <th>Impressions</th>
                      <th>Sent Unchanged</th>
                      <th>Edited</th>
                      <th>Dismissed</th>
                      <th>Regenerated</th>
                      <th>Conversion Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each results.variants as variant}
                      <tr class:winner={results.winner === variant.variant_id}>
                        <td class="variant-id-cell">
                          {variant.variant_id}
                          {#if results.winner === variant.variant_id}
                            <span class="winner-badge">Leader</span>
                          {/if}
                        </td>
                        <td>{variant.total_impressions}</td>
                        <td class="sent-unchanged">{variant.sent_unchanged}</td>
                        <td class="sent-edited">{variant.sent_edited}</td>
                        <td class="dismissed">{variant.dismissed}</td>
                        <td class="regenerated">{variant.regenerated}</td>
                        <td class="conversion-rate {getStatusClass(variant)}">
                          {formatPercent(variant.conversion_rate)}
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>

              <!-- Conversion Chart -->
              {#if results.total_outcomes > 0}
                <div class="chart-section">
                  <h4>Conversion Rate Comparison</h4>
                  <div class="bar-chart">
                    {#each results.variants as variant}
                      <div class="bar-row">
                        <span class="bar-label">{variant.variant_id}</span>
                        <div class="bar-container">
                          <div
                            class="bar"
                            class:winner={results.winner === variant.variant_id}
                            style="width: {Math.min(variant.conversion_rate, 100)}%"
                          >
                            <span class="bar-value">{formatPercent(variant.conversion_rate)}</span>
                          </div>
                        </div>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Significance Note -->
              {#if results.total_outcomes > 0 && !results.is_significant}
                <div class="significance-note">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4M12 8h.01" />
                  </svg>
                  <p>
                    Results are not yet statistically significant.
                    Continue collecting data for more reliable conclusions.
                  </p>
                </div>
              {/if}
            </section>
          {/if}
        {:else}
          <div class="no-selection">
            <p>Select an experiment to view results</p>
          </div>
        {/if}
      </main>
    </div>
  {/if}
</div>

<style>
  .experiment-dashboard {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-primary);
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
  }

  .dashboard-header h1 {
    font-size: 24px;
    font-weight: 600;
    margin: 0;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s ease;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .refresh-btn svg {
    width: 16px;
    height: 16px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 24px;
    background: rgba(255, 95, 87, 0.1);
    color: var(--error-color);
    border-bottom: 1px solid var(--border-color);
  }

  .error-banner svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: var(--text-secondary);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 64px;
    text-align: center;
    color: var(--text-secondary);
  }

  .empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state h2 {
    margin: 0 0 8px;
    font-size: 20px;
    color: var(--text-primary);
  }

  .dashboard-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .experiment-list {
    width: 280px;
    border-right: 1px solid var(--border-color);
    overflow-y: auto;
    background: var(--bg-secondary);
  }

  .experiment-list h2 {
    padding: 16px;
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-secondary);
  }

  .experiment-list ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .experiment-list li {
    display: flex;
    align-items: center;
  }

  .experiment-item {
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border: none;
    background: transparent;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s ease;
  }

  .experiment-item:hover {
    background: var(--bg-hover);
  }

  .experiment-item.selected {
    background: var(--bg-active);
    border-left: 3px solid var(--accent-color);
  }

  .experiment-item.disabled {
    opacity: 0.6;
  }

  .experiment-name {
    font-weight: 500;
    font-size: 14px;
  }

  .experiment-status {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .experiment-status.active {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .toggle-btn {
    padding: 8px;
    background: transparent;
    border: none;
    cursor: pointer;
    opacity: 0.5;
    transition: opacity 0.15s ease;
  }

  .toggle-btn:hover {
    opacity: 1;
  }

  .toggle-btn svg {
    width: 16px;
    height: 16px;
  }

  .results-panel {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  .experiment-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
  }

  .experiment-header h2 {
    margin: 0 0 4px;
    font-size: 20px;
  }

  .description {
    margin: 0;
    color: var(--text-secondary);
    font-size: 14px;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .clear-btn {
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .clear-btn:hover {
    background: var(--bg-hover);
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .variants-section,
  .results-section {
    margin-bottom: 32px;
  }

  .variants-section h3,
  .results-section h3 {
    margin: 0 0 16px;
    font-size: 16px;
    font-weight: 600;
  }

  .variants-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }

  .variant-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
  }

  .variant-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .variant-id {
    font-weight: 600;
    font-size: 16px;
  }

  .variant-weight {
    font-size: 13px;
    padding: 2px 8px;
    background: var(--bg-active);
    border-radius: 4px;
  }

  .variant-config {
    font-size: 12px;
    font-family: monospace;
  }

  .config-item {
    display: flex;
    gap: 8px;
    padding: 4px 0;
    border-top: 1px solid var(--border-color);
  }

  .config-key {
    color: var(--text-secondary);
  }

  .config-value {
    color: var(--accent-color);
    word-break: break-all;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }

  .stat-card.significant {
    border-color: #34c759;
    background: rgba(52, 199, 89, 0.1);
  }

  .stat-label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
  }

  .stat-value.winner {
    color: var(--accent-color);
  }

  .results-table-container {
    overflow-x: auto;
    margin-bottom: 24px;
  }

  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }

  .results-table th,
  .results-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
  }

  .results-table th {
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 12px;
    text-transform: uppercase;
  }

  .results-table tr.winner {
    background: rgba(52, 199, 89, 0.05);
  }

  .variant-id-cell {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
  }

  .winner-badge {
    font-size: 10px;
    padding: 2px 6px;
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
    border-radius: 4px;
    font-weight: 600;
  }

  .sent-unchanged {
    color: #34c759;
  }

  .sent-edited {
    color: var(--accent-color);
  }

  .dismissed {
    color: var(--error-color);
  }

  .regenerated {
    color: var(--text-secondary);
  }

  .conversion-rate {
    font-weight: 600;
  }

  .conversion-rate.good {
    color: #34c759;
  }

  .conversion-rate.moderate {
    color: #ff9f0a;
  }

  .conversion-rate.low {
    color: var(--error-color);
  }

  .chart-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }

  .chart-section h4 {
    margin: 0 0 16px;
    font-size: 14px;
    font-weight: 600;
  }

  .bar-chart {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .bar-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .bar-label {
    width: 100px;
    font-size: 13px;
    font-weight: 500;
    text-align: right;
  }

  .bar-container {
    flex: 1;
    height: 32px;
    background: var(--bg-active);
    border-radius: 6px;
    overflow: hidden;
  }

  .bar {
    height: 100%;
    background: var(--accent-color);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    min-width: 60px;
    transition: width 0.3s ease;
  }

  .bar.winner {
    background: #34c759;
  }

  .bar-value {
    font-size: 12px;
    font-weight: 600;
    color: white;
  }

  .significance-note {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: rgba(255, 159, 10, 0.1);
    border: 1px solid rgba(255, 159, 10, 0.3);
    border-radius: 12px;
    color: #ff9f0a;
  }

  .significance-note svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .significance-note p {
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
  }

  .no-selection {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-secondary);
  }

  @media (max-width: 768px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .experiment-list {
      width: 200px;
    }
  }
</style>
