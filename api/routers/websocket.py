"""WebSocket router for real-time communication.

Provides WebSocket endpoints for:
- Real-time conversation updates
- Streaming model generation responses (token-by-token)
- Health/connection status updates
- Graceful reconnection handling
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from contracts.models import GenerationRequest
from models import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    GENERATE = "generate"
    GENERATE_STREAM = "generate_stream"
    SUBSCRIBE_HEALTH = "subscribe_health"
    UNSUBSCRIBE_HEALTH = "unsubscribe_health"
    PING = "ping"
    CANCEL = "cancel"

    # Server -> Client
    CONNECTED = "connected"
    TOKEN = "token"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    GENERATION_ERROR = "generation_error"
    HEALTH_UPDATE = "health_update"
    PONG = "pong"
    ERROR = "error"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""

    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    subscribed_to_health: bool = False
    active_generation_id: str | None = None


class ConnectionManager:
    """Manages WebSocket connections and broadcasts.

    Thread-safe connection tracking with support for:
    - Individual client messaging
    - Broadcast to all clients
    - Health update subscriptions
    - Active generation tracking for cancellation
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self._clients: dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None

    @property
    def active_connections(self) -> int:
        """Return count of active connections."""
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> WebSocketClient:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept.

        Returns:
            WebSocketClient instance for the new connection.
        """
        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = WebSocketClient(websocket=websocket, client_id=client_id)

        async with self._lock:
            self._clients[client_id] = client

        logger.info("WebSocket client connected: %s", client_id)
        return client

    async def disconnect(self, client_id: str) -> None:
        """Remove a client from the connection pool.

        Args:
            client_id: The ID of the client to disconnect.
        """
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info("WebSocket client disconnected: %s", client_id)

    async def send_message(
        self, client_id: str, message_type: MessageType, data: dict[str, Any] | None = None
    ) -> bool:
        """Send a message to a specific client.

        Args:
            client_id: Target client ID.
            message_type: Type of message to send.
            data: Optional message payload.

        Returns:
            True if message was sent successfully, False otherwise.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if not client:
                return False

        try:
            message = {"type": message_type.value, "data": data or {}}
            await client.websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning("Failed to send message to %s: %s", client_id, e)
            return False

    async def broadcast(
        self, message_type: MessageType, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message_type: Type of message to broadcast.
            data: Optional message payload.
        """
        async with self._lock:
            client_ids = list(self._clients.keys())

        for client_id in client_ids:
            await self.send_message(client_id, message_type, data)

    async def broadcast_health_update(self, health_data: dict[str, Any]) -> None:
        """Broadcast health update to subscribed clients.

        Args:
            health_data: Health status data to broadcast.
        """
        async with self._lock:
            subscribed_clients = [
                c.client_id for c in self._clients.values() if c.subscribed_to_health
            ]

        for client_id in subscribed_clients:
            await self.send_message(client_id, MessageType.HEALTH_UPDATE, health_data)

    def get_client(self, client_id: str) -> WebSocketClient | None:
        """Get a client by ID.

        Args:
            client_id: The client ID to look up.

        Returns:
            WebSocketClient if found, None otherwise.
        """
        return self._clients.get(client_id)

    async def set_health_subscription(self, client_id: str, subscribed: bool) -> None:
        """Update a client's health subscription status.

        Args:
            client_id: The client ID to update.
            subscribed: Whether to subscribe or unsubscribe.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.subscribed_to_health = subscribed

    async def set_active_generation(
        self, client_id: str, generation_id: str | None
    ) -> None:
        """Set the active generation ID for a client.

        Args:
            client_id: The client ID to update.
            generation_id: The generation ID, or None to clear.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.active_generation_id = generation_id

    def get_all_client_ids(self) -> list[str]:
        """Get all connected client IDs.

        Returns:
            List of client IDs.
        """
        return list(self._clients.keys())


# Global connection manager instance
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance.

    Returns:
        The singleton ConnectionManager instance.
    """
    return manager


async def _handle_generate(
    client: WebSocketClient, data: dict[str, Any], stream: bool = False
) -> None:
    """Handle a generation request.

    Args:
        client: The client making the request.
        data: Request data including prompt, context, etc.
        stream: Whether to stream tokens or return complete response.
    """
    generation_id = str(uuid.uuid4())
    await manager.set_active_generation(client.client_id, generation_id)

    try:
        # Extract request parameters
        prompt = data.get("prompt", "")
        if not prompt:
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {"error": "Prompt is required", "generation_id": generation_id},
            )
            return

        context_documents = data.get("context_documents", [])
        few_shot_examples = data.get("few_shot_examples", [])
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.7)
        stop_sequences = data.get("stop_sequences")

        # Convert few_shot_examples to list of tuples
        examples = [
            (ex.get("input", ""), ex.get("output", ""))
            for ex in few_shot_examples
            if isinstance(ex, dict)
        ]

        # Notify generation start
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_START,
            {
                "generation_id": generation_id,
                "streaming": stream,
            },
        )

        # Create generation request
        request = GenerationRequest(
            prompt=prompt,
            context_documents=context_documents,
            few_shot_examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

        # Get generator
        try:
            generator = get_generator()
        except Exception as e:
            logger.error("Failed to get generator: %s", e)
            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_ERROR,
                {
                    "error": "Model service unavailable",
                    "generation_id": generation_id,
                },
            )
            return

        if stream:
            # Use streaming generation
            await _stream_generation(client, generator, request, generation_id)
        else:
            # Use regular generation
            start_time = time.perf_counter()
            response = generator.generate(request)
            generation_time = (time.perf_counter() - start_time) * 1000

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": response.text,
                    "tokens_used": response.tokens_used,
                    "generation_time_ms": generation_time,
                    "model_name": response.model_name,
                    "used_template": response.used_template,
                    "template_name": response.template_name,
                    "finish_reason": response.finish_reason,
                },
            )

    except Exception as e:
        logger.exception("Generation failed")
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_ERROR,
            {"error": str(e), "generation_id": generation_id},
        )
    finally:
        await manager.set_active_generation(client.client_id, None)


async def _stream_generation(
    client: WebSocketClient,
    generator: Any,
    request: GenerationRequest,
    generation_id: str,
) -> None:
    """Stream generation tokens to the client.

    Args:
        client: The client to stream to.
        generator: The generator instance.
        request: The generation request.
        generation_id: Unique ID for this generation.
    """
    start_time = time.perf_counter()
    tokens_sent = 0
    full_text = ""

    try:
        # Check if generator supports streaming
        if hasattr(generator, "generate_stream"):
            async for token_data in generator.generate_stream(request):
                # Check for cancellation
                current_client = manager.get_client(client.client_id)
                if not current_client or current_client.active_generation_id != generation_id:
                    logger.info("Generation %s was cancelled", generation_id)
                    return

                token = token_data.get("token", "")
                full_text += token
                tokens_sent += 1

                await manager.send_message(
                    client.client_id,
                    MessageType.TOKEN,
                    {
                        "generation_id": generation_id,
                        "token": token,
                        "token_index": tokens_sent,
                    },
                )

            generation_time = (time.perf_counter() - start_time) * 1000

            model_name = "unknown"
            if hasattr(generator, "config"):
                model_name = generator.config.model_path

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": full_text,
                    "tokens_used": tokens_sent,
                    "generation_time_ms": generation_time,
                    "model_name": model_name,
                    "used_template": False,
                    "template_name": None,
                    "finish_reason": "stop",
                },
            )
        else:
            # Fallback: use regular generation and simulate streaming
            response = generator.generate(request)
            generation_time = (time.perf_counter() - start_time) * 1000

            # Send tokens one at a time with small delays to simulate streaming
            words = response.text.split()
            for i, word in enumerate(words):
                # Check for cancellation
                current_client = manager.get_client(client.client_id)
                if not current_client or current_client.active_generation_id != generation_id:
                    logger.info("Generation %s was cancelled", generation_id)
                    return

                token = word + (" " if i < len(words) - 1 else "")
                await manager.send_message(
                    client.client_id,
                    MessageType.TOKEN,
                    {
                        "generation_id": generation_id,
                        "token": token,
                        "token_index": i,
                    },
                )
                # Small delay between tokens for visual effect
                await asyncio.sleep(0.02)

            await manager.send_message(
                client.client_id,
                MessageType.GENERATION_COMPLETE,
                {
                    "generation_id": generation_id,
                    "text": response.text,
                    "tokens_used": response.tokens_used,
                    "generation_time_ms": generation_time,
                    "model_name": response.model_name,
                    "used_template": response.used_template,
                    "template_name": response.template_name,
                    "finish_reason": response.finish_reason,
                },
            )

    except Exception as e:
        logger.exception("Streaming generation failed")
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_ERROR,
            {"error": str(e), "generation_id": generation_id},
        )


async def _handle_message(client: WebSocketClient, message: dict[str, Any]) -> None:
    """Handle an incoming WebSocket message.

    Args:
        client: The client that sent the message.
        message: The parsed message data.
    """
    msg_type = message.get("type", "")
    data = message.get("data", {})

    if msg_type == MessageType.PING.value:
        await manager.send_message(
            client.client_id,
            MessageType.PONG,
            {"timestamp": time.time()},
        )

    elif msg_type == MessageType.GENERATE.value:
        # Run generation in background to not block message handling
        asyncio.create_task(_handle_generate(client, data, stream=False))

    elif msg_type == MessageType.GENERATE_STREAM.value:
        asyncio.create_task(_handle_generate(client, data, stream=True))

    elif msg_type == MessageType.SUBSCRIBE_HEALTH.value:
        await manager.set_health_subscription(client.client_id, True)
        await manager.send_message(
            client.client_id,
            MessageType.HEALTH_UPDATE,
            {"subscribed": True},
        )

    elif msg_type == MessageType.UNSUBSCRIBE_HEALTH.value:
        await manager.set_health_subscription(client.client_id, False)

    elif msg_type == MessageType.CANCEL.value:
        # Cancel active generation by clearing the generation ID
        await manager.set_active_generation(client.client_id, None)

    else:
        await manager.send_message(
            client.client_id,
            MessageType.ERROR,
            {"error": f"Unknown message type: {msg_type}"},
        )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time communication.

    Handles bidirectional communication with clients including:
    - Generation requests (streaming and non-streaming)
    - Health status subscriptions
    - Connection management

    Message format (JSON):
        {
            "type": "<message_type>",
            "data": { ... }
        }

    Supported message types (client -> server):
        - ping: Keep-alive ping
        - generate: Non-streaming generation
        - generate_stream: Streaming generation (token-by-token)
        - subscribe_health: Subscribe to health updates
        - unsubscribe_health: Unsubscribe from health updates
        - cancel: Cancel active generation

    Server -> client message types:
        - connected: Connection established
        - pong: Response to ping
        - generation_start: Generation started
        - token: Single token (streaming)
        - generation_complete: Generation finished
        - generation_error: Generation failed
        - health_update: Health status update
        - error: Generic error
    """
    client = await manager.connect(websocket)

    # Send connection confirmation
    await manager.send_message(
        client.client_id,
        MessageType.CONNECTED,
        {
            "client_id": client.client_id,
            "timestamp": client.connected_at,
        },
    )

    try:
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await manager.send_message(
                    client.client_id,
                    MessageType.ERROR,
                    {"error": "Invalid JSON"},
                )
                continue

            # Handle message
            await _handle_message(client, message)

    except WebSocketDisconnect:
        logger.info("WebSocket client %s disconnected", client.client_id)
    except Exception as e:
        logger.exception("WebSocket error for client %s: %s", client.client_id, e)
    finally:
        await manager.disconnect(client.client_id)


@router.get(
    "/ws/status",
    summary="Get WebSocket connection status",
    response_description="Current WebSocket server status",
    responses={
        200: {
            "description": "WebSocket server status",
            "content": {
                "application/json": {
                    "example": {
                        "active_connections": 5,
                        "health_subscribers": 2,
                        "status": "operational",
                    }
                }
            },
        }
    },
)
def get_websocket_status() -> dict[str, Any]:
    """Get the current WebSocket server status.

    Returns:
        Dictionary with connection counts and status.
    """
    client_ids = manager.get_all_client_ids()
    health_subscribers = sum(
        1 for cid in client_ids if (c := manager.get_client(cid)) and c.subscribed_to_health
    )

    return {
        "active_connections": manager.active_connections,
        "health_subscribers": health_subscribers,
        "status": "operational",
    }
