/**
 * WebSocket client for real-time JARVIS communication
 *
 * Provides:
 * - Automatic reconnection with exponential backoff
 * - Streaming generation support
 * - Health status subscriptions
 * - Connection state management
 */

const WS_BASE = "ws://localhost:8742";

/**
 * WebSocket message types (server -> client)
 */
export enum ServerMessageType {
  CONNECTED = "connected",
  TOKEN = "token",
  GENERATION_START = "generation_start",
  GENERATION_COMPLETE = "generation_complete",
  GENERATION_ERROR = "generation_error",
  HEALTH_UPDATE = "health_update",
  PONG = "pong",
  ERROR = "error",
}

/**
 * WebSocket message types (client -> server)
 */
export enum ClientMessageType {
  GENERATE = "generate",
  GENERATE_STREAM = "generate_stream",
  SUBSCRIBE_HEALTH = "subscribe_health",
  UNSUBSCRIBE_HEALTH = "unsubscribe_health",
  PING = "ping",
  CANCEL = "cancel",
}

/**
 * Connection state
 */
export type ConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "reconnecting";

/**
 * WebSocket message structure
 */
export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
}

/**
 * Generation request data
 */
export interface GenerateRequest {
  prompt: string;
  context_documents?: string[];
  few_shot_examples?: Array<{ input: string; output: string }>;
  max_tokens?: number;
  temperature?: number;
  stop_sequences?: string[];
}

/**
 * Token event data
 */
export interface TokenEvent {
  generation_id: string;
  token: string;
  token_index: number;
}

/**
 * Generation start event data
 */
export interface GenerationStartEvent {
  generation_id: string;
  streaming: boolean;
}

/**
 * Generation complete event data
 */
export interface GenerationCompleteEvent {
  generation_id: string;
  text: string;
  tokens_used: number;
  generation_time_ms: number;
  model_name: string;
  used_template: boolean;
  template_name: string | null;
  finish_reason: string;
}

/**
 * Generation error event data
 */
export interface GenerationErrorEvent {
  generation_id: string;
  error: string;
}

/**
 * Connected event data
 */
export interface ConnectedEvent {
  client_id: string;
  timestamp: number;
}

/**
 * Event handler types
 */
export interface WebSocketEventHandlers {
  onConnect?: (event: ConnectedEvent) => void;
  onDisconnect?: () => void;
  onToken?: (event: TokenEvent) => void;
  onGenerationStart?: (event: GenerationStartEvent) => void;
  onGenerationComplete?: (event: GenerationCompleteEvent) => void;
  onGenerationError?: (event: GenerationErrorEvent) => void;
  onHealthUpdate?: (data: unknown) => void;
  onError?: (error: string) => void;
  onStateChange?: (state: ConnectionState) => void;
}

/**
 * WebSocket client for JARVIS real-time communication
 */
class JarvisWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WebSocketEventHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private _state: ConnectionState = "disconnected";
  private _clientId: string | null = null;
  private shouldReconnect = true;

  constructor(baseUrl: string = WS_BASE) {
    this.url = `${baseUrl}/ws`;
  }

  /**
   * Get the current connection state
   */
  get state(): ConnectionState {
    return this._state;
  }

  /**
   * Get the client ID assigned by the server
   */
  get clientId(): string | null {
    return this._clientId;
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this._state === "connected" && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Set event handlers
   */
  setHandlers(handlers: WebSocketEventHandlers): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

  /**
   * Connect to the WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.shouldReconnect = true;
    this.setState("connecting");

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.startPingInterval();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onclose = () => {
        this.handleDisconnect();
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.handlers.onError?.("WebSocket connection error");
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      this.handleDisconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;
    this.stopPingInterval();
    this.clearReconnectTimer();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this._clientId = null;
    this.setState("disconnected");
  }

  /**
   * Send a message to the server
   */
  send(type: ClientMessageType, data: unknown = {}): boolean {
    if (!this.isConnected) {
      console.warn("Cannot send message: not connected");
      return false;
    }

    try {
      const message: WebSocketMessage = { type, data };
      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error("Failed to send message:", error);
      return false;
    }
  }

  /**
   * Request text generation (non-streaming)
   */
  generate(request: GenerateRequest): boolean {
    return this.send(ClientMessageType.GENERATE, request);
  }

  /**
   * Request streaming text generation
   */
  generateStream(request: GenerateRequest): boolean {
    return this.send(ClientMessageType.GENERATE_STREAM, request);
  }

  /**
   * Cancel active generation
   */
  cancelGeneration(): boolean {
    return this.send(ClientMessageType.CANCEL);
  }

  /**
   * Subscribe to health updates
   */
  subscribeHealth(): boolean {
    return this.send(ClientMessageType.SUBSCRIBE_HEALTH);
  }

  /**
   * Unsubscribe from health updates
   */
  unsubscribeHealth(): boolean {
    return this.send(ClientMessageType.UNSUBSCRIBE_HEALTH);
  }

  /**
   * Send a ping to keep the connection alive
   */
  ping(): boolean {
    return this.send(ClientMessageType.PING);
  }

  private setState(state: ConnectionState): void {
    if (this._state !== state) {
      this._state = state;
      this.handlers.onStateChange?.(state);
    }
  }

  private handleMessage(rawData: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(rawData);

      switch (message.type) {
        case ServerMessageType.CONNECTED:
          this._clientId = (message.data as ConnectedEvent).client_id;
          this.setState("connected");
          this.handlers.onConnect?.(message.data as ConnectedEvent);
          break;

        case ServerMessageType.TOKEN:
          this.handlers.onToken?.(message.data as TokenEvent);
          break;

        case ServerMessageType.GENERATION_START:
          this.handlers.onGenerationStart?.(
            message.data as GenerationStartEvent
          );
          break;

        case ServerMessageType.GENERATION_COMPLETE:
          this.handlers.onGenerationComplete?.(
            message.data as GenerationCompleteEvent
          );
          break;

        case ServerMessageType.GENERATION_ERROR:
          this.handlers.onGenerationError?.(
            message.data as GenerationErrorEvent
          );
          break;

        case ServerMessageType.HEALTH_UPDATE:
          this.handlers.onHealthUpdate?.(message.data);
          break;

        case ServerMessageType.PONG:
          // Pong received, connection is alive
          break;

        case ServerMessageType.ERROR:
          const errorData = message.data as { error: string };
          this.handlers.onError?.(errorData.error);
          break;

        default:
          console.warn("Unknown message type:", message.type);
      }
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  }

  private handleDisconnect(): void {
    this.stopPingInterval();
    this._clientId = null;
    this.handlers.onDisconnect?.();

    if (this.shouldReconnect) {
      this.scheduleReconnect();
    } else {
      this.setState("disconnected");
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts reached");
      this.setState("disconnected");
      this.handlers.onError?.("Connection lost. Please refresh the page.");
      return;
    }

    this.setState("reconnecting");

    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts) +
        Math.random() * 1000,
      this.maxReconnectDelay
    );

    console.log(
      `Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts + 1})`
    );

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startPingInterval(): void {
    this.stopPingInterval();
    // Send ping every 30 seconds to keep connection alive
    this.pingTimer = setInterval(() => {
      this.ping();
    }, 30000);
  }

  private stopPingInterval(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
}

// Export singleton instance
export const jarvisWs = new JarvisWebSocket();

// Export class for testing or custom instances
export { JarvisWebSocket };
