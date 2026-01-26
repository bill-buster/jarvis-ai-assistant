/**
 * Health store for tracking API connection and system status
 */

import { writable } from "svelte/store";
import type { HealthResponse } from "../api/types";
import { api } from "../api/client";

export interface HealthState {
  connected: boolean;
  loading: boolean;
  error: string | null;
  data: HealthResponse | null;
}

const initialState: HealthState = {
  connected: false,
  loading: false,
  error: null,
  data: null,
};

export const healthStore = writable<HealthState>(initialState);

export async function checkApiConnection(): Promise<boolean> {
  healthStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    await api.ping();
    healthStore.update((state) => ({
      ...state,
      connected: true,
      loading: false,
    }));
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Connection failed";
    healthStore.update((state) => ({
      ...state,
      connected: false,
      loading: false,
      error: message,
    }));
    return false;
  }
}

export async function fetchHealth(): Promise<void> {
  healthStore.update((state) => ({ ...state, loading: true, error: null }));

  try {
    const data = await api.getHealth();
    healthStore.update((state) => ({
      ...state,
      data,
      connected: true,
      loading: false,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch health";
    healthStore.update((state) => ({
      ...state,
      loading: false,
      error: message,
    }));
  }
}
