/**
 * Digest store for tracking digest generation state
 */

import { writable } from "svelte/store";
import type {
  DigestResponse,
  DigestPeriod,
  DigestFormat,
  DigestExportResponse,
  DigestPreferences,
} from "../api/types";
import { api } from "../api/client";

export interface DigestState {
  loading: boolean;
  exporting: boolean;
  error: string | null;
  data: DigestResponse | null;
  preferences: DigestPreferences | null;
  lastExport: DigestExportResponse | null;
}

const initialState: DigestState = {
  loading: false,
  exporting: false,
  error: null,
  data: null,
  preferences: null,
  lastExport: null,
};

export const digestStore = writable<DigestState>(initialState);

export async function fetchDigest(
  period: DigestPeriod = "daily"
): Promise<void> {
  digestStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.generateDigest({ period });
    digestStore.update((state) => ({
      ...state,
      data,
      loading: false,
    }));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch digest";
    digestStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function fetchDailyDigest(): Promise<void> {
  digestStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.getDailyDigest();
    digestStore.update((state) => ({
      ...state,
      data,
      loading: false,
    }));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch daily digest";
    digestStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function fetchWeeklyDigest(): Promise<void> {
  digestStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.getWeeklyDigest();
    digestStore.update((state) => ({
      ...state,
      data,
      loading: false,
    }));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to fetch weekly digest";
    digestStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}

export async function exportDigest(
  period: DigestPeriod = "daily",
  format: DigestFormat = "markdown"
): Promise<DigestExportResponse | null> {
  digestStore.update((state) => ({ ...state, exporting: true, error: null }));

  try {
    const response = await api.exportDigest({ period, format });
    digestStore.update((state) => ({
      ...state,
      lastExport: response,
      exporting: false,
    }));
    return response;
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to export digest";
    digestStore.update((state) => ({
      ...state,
      exporting: false,
      error: message,
    }));
    return null;
  }
}

export async function fetchDigestPreferences(): Promise<void> {
  try {
    const preferences = await api.getDigestPreferences();
    digestStore.update((state) => ({
      ...state,
      preferences,
    }));
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Failed to fetch digest preferences";
    digestStore.update((state) => ({
      ...state,
      error: message,
    }));
  }
}

export async function updateDigestPreferences(
  updates: Partial<DigestPreferences>
): Promise<void> {
  try {
    const preferences = await api.updateDigestPreferences(updates);
    digestStore.update((state) => ({
      ...state,
      preferences,
    }));
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Failed to update digest preferences";
    digestStore.update((state) => ({
      ...state,
      error: message,
    }));
  }
}

export function clearDigestError(): void {
  digestStore.update((state) => ({ ...state, error: null }));
}

export function resetDigestStore(): void {
  digestStore.set(initialState);
}
