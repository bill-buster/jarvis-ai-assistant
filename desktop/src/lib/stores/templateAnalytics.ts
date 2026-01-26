/**
 * Template Analytics store for tracking template matching performance
 */

import { writable } from "svelte/store";
import type { TemplateAnalyticsDashboard } from "../api/types";
import { api } from "../api/client";

export interface TemplateAnalyticsState {
  loading: boolean;
  error: string | null;
  data: TemplateAnalyticsDashboard | null;
}

const initialState: TemplateAnalyticsState = {
  loading: false,
  error: null,
  data: null,
};

export const templateAnalyticsStore = writable<TemplateAnalyticsState>(initialState);

export async function fetchTemplateAnalytics(): Promise<void> {
  templateAnalyticsStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.getTemplateAnalyticsDashboard();
    templateAnalyticsStore.update((state) => ({
      ...state,
      data,
      loading: false,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch template analytics";
    templateAnalyticsStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function resetTemplateAnalytics(): Promise<boolean> {
  try {
    await api.resetTemplateAnalytics();
    // Refresh data after reset
    await fetchTemplateAnalytics();
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to reset analytics";
    templateAnalyticsStore.update((state) => ({
      ...state,
      error: message,
    }));
    return false;
  }
}

export async function exportTemplateAnalytics(): Promise<void> {
  try {
    const blob = await api.exportTemplateAnalytics();
    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `template_analytics_${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to export analytics";
    templateAnalyticsStore.update((state) => ({
      ...state,
      error: message,
    }));
  }
}
