/**
 * WebSocket store for real-time connection management
 *
 * Provides Svelte store integration for WebSocket state,
 * streaming generation, and connection status.
 */

import { writable, derived, get } from "svelte/store";
import {
  jarvisWs,
  type ConnectionState,
  type TokenEvent,
  type GenerationStartEvent,
  type GenerationCompleteEvent,
  type GenerationErrorEvent,
  type ConnectedEvent,
  type GenerateRequest,
} from "../api/websocket";

/**
 * Streaming generation state
 */
export interface StreamingState {
  isStreaming: boolean;
  generationId: string | null;
  tokens: string[];
  fullText: string;
  error: string | null;
  startTime: number | null;
  endTime: number | null;
}

/**
 * WebSocket store state
 */
export interface WebSocketState {
  connectionState: ConnectionState;
  clientId: string | null;
  connectedAt: number | null;
  error: string | null;
  streaming: StreamingState;
}

const initialStreamingState: StreamingState = {
  isStreaming: false,
  generationId: null,
  tokens: [],
  fullText: "",
  error: null,
  startTime: null,
  endTime: null,
};

const initialState: WebSocketState = {
  connectionState: "disconnected",
  clientId: null,
  connectedAt: null,
  error: null,
  streaming: initialStreamingState,
};

// Create the main store
export const websocketStore = writable<WebSocketState>(initialState);

// Derived stores for specific state slices
export const connectionState = derived(
  websocketStore,
  ($state) => $state.connectionState
);

export const isConnected = derived(
  websocketStore,
  ($state) => $state.connectionState === "connected"
);

export const isStreaming = derived(
  websocketStore,
  ($state) => $state.streaming.isStreaming
);

export const streamingText = derived(
  websocketStore,
  ($state) => $state.streaming.fullText
);

export const streamingError = derived(
  websocketStore,
  ($state) => $state.streaming.error
);

/**
 * Initialize WebSocket connection and set up event handlers
 */
export function initializeWebSocket(): void {
  jarvisWs.setHandlers({
    onConnect: (event: ConnectedEvent) => {
      websocketStore.update((state) => ({
        ...state,
        connectionState: "connected",
        clientId: event.client_id,
        connectedAt: event.timestamp,
        error: null,
      }));
    },

    onDisconnect: () => {
      websocketStore.update((state) => ({
        ...state,
        clientId: null,
        connectedAt: null,
      }));
    },

    onStateChange: (connectionState: ConnectionState) => {
      websocketStore.update((state) => ({
        ...state,
        connectionState,
      }));
    },

    onToken: (event: TokenEvent) => {
      websocketStore.update((state) => {
        if (state.streaming.generationId !== event.generation_id) {
          return state;
        }

        return {
          ...state,
          streaming: {
            ...state.streaming,
            tokens: [...state.streaming.tokens, event.token],
            fullText: state.streaming.fullText + event.token,
          },
        };
      });
    },

    onGenerationStart: (event: GenerationStartEvent) => {
      websocketStore.update((state) => ({
        ...state,
        streaming: {
          isStreaming: true,
          generationId: event.generation_id,
          tokens: [],
          fullText: "",
          error: null,
          startTime: Date.now(),
          endTime: null,
        },
      }));
    },

    onGenerationComplete: (event: GenerationCompleteEvent) => {
      websocketStore.update((state) => {
        if (state.streaming.generationId !== event.generation_id) {
          return state;
        }

        return {
          ...state,
          streaming: {
            ...state.streaming,
            isStreaming: false,
            fullText: event.text,
            endTime: Date.now(),
          },
        };
      });
    },

    onGenerationError: (event: GenerationErrorEvent) => {
      websocketStore.update((state) => {
        if (
          state.streaming.generationId &&
          state.streaming.generationId !== event.generation_id
        ) {
          return state;
        }

        return {
          ...state,
          streaming: {
            ...state.streaming,
            isStreaming: false,
            error: event.error,
            endTime: Date.now(),
          },
        };
      });
    },

    onHealthUpdate: (data: unknown) => {
      // Health updates can be handled by the health store if needed
      console.log("Health update received:", data);
    },

    onError: (error: string) => {
      websocketStore.update((state) => ({
        ...state,
        error,
      }));
    },
  });

  jarvisWs.connect();
}

/**
 * Disconnect the WebSocket
 */
export function disconnectWebSocket(): void {
  jarvisWs.disconnect();
}

/**
 * Request streaming text generation
 */
export function generateStream(request: GenerateRequest): boolean {
  // Reset streaming state before starting
  websocketStore.update((state) => ({
    ...state,
    streaming: initialStreamingState,
  }));

  return jarvisWs.generateStream(request);
}

/**
 * Request non-streaming text generation
 */
export function generate(request: GenerateRequest): boolean {
  return jarvisWs.generate(request);
}

/**
 * Cancel active generation
 */
export function cancelGeneration(): void {
  jarvisWs.cancelGeneration();

  websocketStore.update((state) => ({
    ...state,
    streaming: {
      ...state.streaming,
      isStreaming: false,
      error: "Generation cancelled",
      endTime: Date.now(),
    },
  }));
}

/**
 * Subscribe to health updates
 */
export function subscribeToHealth(): boolean {
  return jarvisWs.subscribeHealth();
}

/**
 * Unsubscribe from health updates
 */
export function unsubscribeFromHealth(): boolean {
  return jarvisWs.unsubscribeHealth();
}

/**
 * Clear any error state
 */
export function clearError(): void {
  websocketStore.update((state) => ({
    ...state,
    error: null,
  }));
}

/**
 * Reset streaming state
 */
export function resetStreamingState(): void {
  websocketStore.update((state) => ({
    ...state,
    streaming: initialStreamingState,
  }));
}

/**
 * Get the current WebSocket state
 */
export function getWebSocketState(): WebSocketState {
  return get(websocketStore);
}
