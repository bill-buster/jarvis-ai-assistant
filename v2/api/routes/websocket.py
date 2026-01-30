"""WebSocket router for real-time communication in JARVIS v2.

Provides:
- Real-time new message notifications
- Streaming reply generation
- Connection management
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    GENERATE_REPLIES = "generate_replies"
    WATCH_MESSAGES = "watch_messages"
    UNWATCH_MESSAGES = "unwatch_messages"
    PING = "ping"
    CANCEL = "cancel"

    # Server -> Client
    CONNECTED = "connected"
    TOKEN = "token"
    REPLY = "reply"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    GENERATION_ERROR = "generation_error"
    NEW_MESSAGE = "new_message"
    PONG = "pong"
    ERROR = "error"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""

    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    watching_chat_ids: set = field(default_factory=set)
    active_generation_id: str | None = None


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self._clients: dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()
        self._message_watcher_task: asyncio.Task | None = None

    @property
    def active_connections(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> WebSocketClient:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = WebSocketClient(websocket=websocket, client_id=client_id)

        async with self._lock:
            self._clients[client_id] = client

        logger.info("WebSocket client connected: %s", client_id)
        return client

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info("WebSocket client disconnected: %s", client_id)

    async def send_message(
        self, client_id: str, message_type: MessageType, data: dict[str, Any] | None = None
    ) -> bool:
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

    async def broadcast_new_message(self, chat_id: str, message_data: dict) -> None:
        """Broadcast new message to clients watching this chat."""
        async with self._lock:
            watching_clients = [
                c.client_id for c in self._clients.values()
                if chat_id in c.watching_chat_ids
            ]

        for client_id in watching_clients:
            await self.send_message(
                client_id,
                MessageType.NEW_MESSAGE,
                {"chat_id": chat_id, "message": message_data}
            )

    def get_client(self, client_id: str) -> WebSocketClient | None:
        return self._clients.get(client_id)

    async def set_watching(self, client_id: str, chat_id: str, watching: bool) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                if watching:
                    client.watching_chat_ids.add(chat_id)
                else:
                    client.watching_chat_ids.discard(chat_id)

    async def set_active_generation(self, client_id: str, generation_id: str | None) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.active_generation_id = generation_id


# Global connection manager
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    return manager


async def _handle_generate_replies(
    client: WebSocketClient, data: dict[str, Any]
) -> None:
    """Handle streaming reply generation."""
    from core.generation import ReplyGenerator
    from core.imessage import MessageReader
    from core.models import get_model_loader

    generation_id = str(uuid.uuid4())
    await manager.set_active_generation(client.client_id, generation_id)

    chat_id = data.get("chat_id")
    if not chat_id:
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_ERROR,
            {"error": "chat_id is required", "generation_id": generation_id},
        )
        return

    try:
        # Notify generation start
        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_START,
            {"generation_id": generation_id, "chat_id": chat_id},
        )

        # Get messages
        reader = MessageReader()
        try:
            messages = reader.get_messages(chat_id=chat_id, limit=30)
            messages_dict = [
                {
                    "text": m.text,
                    "sender": m.sender,
                    "is_from_me": m.is_from_me,
                    "timestamp": m.timestamp,
                }
                for m in reversed(messages)
            ]
        finally:
            reader.close()

        # Generate replies
        loader = get_model_loader()
        generator = ReplyGenerator(loader)

        start_time = time.time()
        result = generator.generate_replies(
            messages=messages_dict,
            chat_id=chat_id,
            num_replies=1,  # Generate 1 reply for speed
            user_name="Jwalin",
        )
        generation_time = (time.time() - start_time) * 1000

        # Stream each reply as it's ready
        for i, reply in enumerate(result.replies):
            # Check for cancellation
            current_client = manager.get_client(client.client_id)
            if not current_client or current_client.active_generation_id != generation_id:
                logger.info("Generation %s was cancelled", generation_id)
                return

            await manager.send_message(
                client.client_id,
                MessageType.REPLY,
                {
                    "generation_id": generation_id,
                    "reply_index": i,
                    "text": reply.text,
                    "reply_type": reply.reply_type,
                    "confidence": reply.confidence,
                },
            )

        # Send completion with full debug info
        past_replies_data = [
            {
                "their_message": their_msg,
                "your_reply": your_reply,
                "similarity": sim,
            }
            for their_msg, your_reply, sim in result.past_replies
        ]

        await manager.send_message(
            client.client_id,
            MessageType.GENERATION_COMPLETE,
            {
                "generation_id": generation_id,
                "chat_id": chat_id,
                "generation_time_ms": generation_time,
                "model_used": result.model_used,
                "style_instructions": result.style_instructions,
                "intent_detected": result.context.intent.value,
                "past_replies_count": len(result.past_replies),
                "past_replies": past_replies_data,
                "full_prompt": result.prompt_used,
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


async def _handle_message(client: WebSocketClient, message: dict[str, Any]) -> None:
    """Handle an incoming WebSocket message."""
    msg_type = message.get("type", "")
    data = message.get("data", {})

    if msg_type == MessageType.PING.value:
        await manager.send_message(
            client.client_id,
            MessageType.PONG,
            {"timestamp": time.time()},
        )

    elif msg_type == MessageType.GENERATE_REPLIES.value:
        task = asyncio.create_task(_handle_generate_replies(client, data))
        task.add_done_callback(lambda t: _log_task_exception(t, client.client_id))

    elif msg_type == MessageType.WATCH_MESSAGES.value:
        chat_id = data.get("chat_id")
        if chat_id:
            await manager.set_watching(client.client_id, chat_id, True)
            await manager.send_message(
                client.client_id,
                MessageType.PONG,
                {"watching": chat_id},
            )

    elif msg_type == MessageType.UNWATCH_MESSAGES.value:
        chat_id = data.get("chat_id")
        if chat_id:
            await manager.set_watching(client.client_id, chat_id, False)

    elif msg_type == MessageType.CANCEL.value:
        await manager.set_active_generation(client.client_id, None)

    else:
        await manager.send_message(
            client.client_id,
            MessageType.ERROR,
            {"error": f"Unknown message type: {msg_type}"},
        )


def _log_task_exception(task: asyncio.Task, client_id: str) -> None:
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error("Background task failed for client %s: %s", client_id, exc)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time communication.

    Supported message types (client -> server):
        - ping: Keep-alive ping
        - generate_replies: Generate reply suggestions (streams back)
        - watch_messages: Subscribe to new messages for a chat
        - unwatch_messages: Unsubscribe from chat messages
        - cancel: Cancel active generation

    Server -> client message types:
        - connected: Connection established
        - pong: Response to ping
        - generation_start: Generation started
        - reply: Single reply suggestion
        - generation_complete: All replies generated
        - generation_error: Generation failed
        - new_message: New message in watched chat
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

            await _handle_message(client, message)

    except WebSocketDisconnect:
        logger.info("WebSocket client %s disconnected", client.client_id)
    except Exception as e:
        logger.exception("WebSocket error for client %s: %s", client.client_id, e)
    finally:
        await manager.disconnect(client.client_id)


@router.get("/ws/status")
def get_websocket_status() -> dict[str, Any]:
    """Get WebSocket server status."""
    return {
        "active_connections": manager.active_connections,
        "status": "operational",
    }
