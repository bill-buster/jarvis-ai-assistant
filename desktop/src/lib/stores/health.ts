/**
 * Health status store for JARVIS desktop app.
 */

import { writable } from "svelte/store";
import type { HealthStatus } from "../api/types";
import { api } from "../api/client";

// Health status store
export const healthStatus = writable<HealthStatus | null>(null);
export const healthError = writable<string | null>(null);
export const isApiConnected = writable<boolean>(false);

/**
 * Check API connection and fetch health status
 */
export async function checkApiConnection(): Promise<boolean> {
  try {
    const connected = await api.ping();
    isApiConnected.set(connected);

    if (connected) {
      const status = await api.getHealth();
      healthStatus.set(status);
      healthError.set(null);
    } else {
      healthError.set("Unable to connect to JARVIS API");
    }

    return connected;
  } catch (error) {
    isApiConnected.set(false);
    healthError.set(
      error instanceof Error ? error.message : "Connection failed"
    );
    return false;
  }
}

/**
 * Refresh health status
 */
export async function refreshHealth(): Promise<void> {
  try {
    const status = await api.getHealth();
    healthStatus.set(status);
    healthError.set(null);
  } catch (error) {
    healthError.set(
      error instanceof Error ? error.message : "Failed to fetch health status"
    );
  }
}
