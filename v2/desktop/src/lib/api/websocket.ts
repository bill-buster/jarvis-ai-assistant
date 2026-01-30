/**
 * WebSocket client for real-time JARVIS v2 communication
 *
 * Provides:
 * - Real-time message notifications
 * - Streaming reply generation
 * - Automatic reconnection
 */

const WS_BASE = "ws://localhost:8000";

export enum ServerMessageType {
  CONNECTED = "connected",
  TOKEN = "token",
  REPLY = "reply",
  GENERATION_START = "generation_start",
  GENERATION_COMPLETE = "generation_complete",
  GENERATION_ERROR = "generation_error",
  NEW_MESSAGE = "new_message",
  PONG = "pong",
  ERROR = "error",
}

export enum ClientMessageType {
  GENERATE_REPLIES = "generate_replies",
  WATCH_MESSAGES = "watch_messages",
  UNWATCH_MESSAGES = "unwatch_messages",
  PING = "ping",
  CANCEL = "cancel",
}

export type ConnectionState = "disconnected" | "connecting" | "connected" | "reconnecting";

export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
}

export interface ReplyEvent {
  generation_id: string;
  reply_index: number;
  text: string;
  reply_type: string;
  confidence: number;
}

export interface GenerationStartEvent {
  generation_id: string;
  chat_id: string;
}

export interface PastReplyWs {
  their_message: string;
  your_reply: string;
  similarity: number;
}

export interface GenerationCompleteEvent {
  generation_id: string;
  chat_id: string;
  generation_time_ms: number;
  model_used: string;
  style_instructions: string;
  intent_detected: string;
  past_replies_count: number;
  past_replies: PastReplyWs[];
  full_prompt: string;
}

export interface GenerationErrorEvent {
  generation_id: string;
  error: string;
}

export interface NewMessageEvent {
  chat_id: string;
  message: {
    id: number;
    text: string;
    sender: string;
    is_from_me: boolean;
    timestamp: string;
  };
}

export interface ConnectedEvent {
  client_id: string;
  timestamp: number;
}

export interface WebSocketEventHandlers {
  onConnect?: (event: ConnectedEvent) => void;
  onDisconnect?: () => void;
  onReply?: (event: ReplyEvent) => void;
  onGenerationStart?: (event: GenerationStartEvent) => void;
  onGenerationComplete?: (event: GenerationCompleteEvent) => void;
  onGenerationError?: (event: GenerationErrorEvent) => void;
  onNewMessage?: (event: NewMessageEvent) => void;
  onError?: (error: string) => void;
  onStateChange?: (state: ConnectionState) => void;
}

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

  get state(): ConnectionState {
    return this._state;
  }

  get clientId(): string | null {
    return this._clientId;
  }

  get isConnected(): boolean {
    return this._state === "connected" && this.ws?.readyState === WebSocket.OPEN;
  }

  setHandlers(handlers: WebSocketEventHandlers): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

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

  generateReplies(chatId: string): boolean {
    return this.send(ClientMessageType.GENERATE_REPLIES, { chat_id: chatId });
  }

  watchMessages(chatId: string): boolean {
    return this.send(ClientMessageType.WATCH_MESSAGES, { chat_id: chatId });
  }

  unwatchMessages(chatId: string): boolean {
    return this.send(ClientMessageType.UNWATCH_MESSAGES, { chat_id: chatId });
  }

  cancelGeneration(): boolean {
    return this.send(ClientMessageType.CANCEL);
  }

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

        case ServerMessageType.REPLY:
          this.handlers.onReply?.(message.data as ReplyEvent);
          break;

        case ServerMessageType.GENERATION_START:
          this.handlers.onGenerationStart?.(message.data as GenerationStartEvent);
          break;

        case ServerMessageType.GENERATION_COMPLETE:
          this.handlers.onGenerationComplete?.(message.data as GenerationCompleteEvent);
          break;

        case ServerMessageType.GENERATION_ERROR:
          this.handlers.onGenerationError?.(message.data as GenerationErrorEvent);
          break;

        case ServerMessageType.NEW_MESSAGE:
          this.handlers.onNewMessage?.(message.data as NewMessageEvent);
          break;

        case ServerMessageType.PONG:
          // Connection alive
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

    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts) + Math.random() * 1000,
      this.maxReconnectDelay
    );

    console.log(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts + 1})`);

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

export const jarvisWs = new JarvisWebSocket();
export { JarvisWebSocket };
