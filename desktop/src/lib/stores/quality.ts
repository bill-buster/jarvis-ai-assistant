/**
 * Quality metrics store for tracking response generation performance
 */

import { writable } from "svelte/store";
import type { QualityDashboardData } from "../api/types";
import { api } from "../api/client";

export interface QualityState {
  loading: boolean;
  error: string | null;
  data: QualityDashboardData | null;
}

const initialState: QualityState = {
  loading: false,
  error: null,
  data: null,
};

export const qualityStore = writable<QualityState>(initialState);

export async function fetchQualityDashboard(): Promise<void> {
  qualityStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.getQualityDashboard();
    qualityStore.update((state) => ({
      ...state,
      data,
      loading: false,
    }));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch quality metrics";
    qualityStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function fetchQualitySummary(): Promise<void> {
  qualityStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const summary = await api.getQualitySummary();
    qualityStore.update((state) => ({
      ...state,
      data: state.data ? { ...state.data, summary } : null,
      loading: false,
    }));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch quality summary";
    qualityStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function resetQualityMetrics(): Promise<void> {
  try {
    await api.resetQualityMetrics();
    // Refresh the data after reset
    await fetchQualityDashboard();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to reset quality metrics";
    qualityStore.update((state) => ({
      ...state,
      error: message,
    }));
  }
}

export { type QualityState };
