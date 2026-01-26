/**
 * Health store for tracking API connection and system health.
 */

import { apiClient } from "../api/client";
import type { HealthStatus } from "../api/types";

// State
let connected = $state(false);
let health = $state<HealthStatus | null>(null);
let loading = $state(false);
let error = $state<string | null>(null);

/**
 * Check API connection status.
 */
export async function checkApiConnection(): Promise<boolean> {
  loading = true;
  error = null;

  try {
    await apiClient.ping();
    connected = true;
    return true;
  } catch (e) {
    connected = false;
    error = e instanceof Error ? e.message : "Connection failed";
    return false;
  } finally {
    loading = false;
  }
}

/**
 * Fetch full health status from the API.
 */
export async function fetchHealth(): Promise<HealthStatus | null> {
  loading = true;
  error = null;

  try {
    health = await apiClient.getHealth();
    connected = true;
    return health;
  } catch (e) {
    connected = false;
    error = e instanceof Error ? e.message : "Failed to fetch health";
    return null;
  } finally {
    loading = false;
  }
}

/**
 * Get current connection status.
 */
export function isConnected(): boolean {
  return connected;
}

/**
 * Get current health status.
 */
export function getHealth(): HealthStatus | null {
  return health;
}

/**
 * Get loading state.
 */
export function isLoading(): boolean {
  return loading;
}

/**
 * Get error message if any.
 */
export function getError(): string | null {
  return error;
}

// Export reactive getters for Svelte components
export function getHealthStore() {
  return {
    get connected() {
      return connected;
    },
    get health() {
      return health;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
  };
}
