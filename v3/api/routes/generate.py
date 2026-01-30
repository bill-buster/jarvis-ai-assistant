"""Reply generation endpoints."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, HTTPException

from core.config import settings

router = APIRouter()

# Module-level singleton ReplyGenerator to avoid initialization overhead per request
_generator = None
_generator_lock = threading.Lock()


def _get_generator():
    """Get or create singleton ReplyGenerator."""
    global _generator
    if _generator is not None:
        return _generator

    with _generator_lock:
        if _generator is not None:
            return _generator

        from core.generation import ReplyGenerator
        from core.models import get_model_loader

        loader = get_model_loader()
        _generator = ReplyGenerator(loader)
        return _generator


def _fetch_messages(chat_id: str, limit: int | None = None):
    """Fetch messages from iMessage database (blocking I/O)."""
    from core.imessage import MessageReader

    if limit is None:
        limit = settings.api.generation_context_limit

    reader = MessageReader()
    try:
        if not reader.check_access():
            return None, "Cannot access iMessage database"
        messages = reader.get_messages(chat_id=chat_id, limit=limit)
        return messages, None
    finally:
        reader.close()


def _generate_reply(messages_dict: list, chat_id: str, user_name: str):
    """Generate reply using the model (blocking I/O)."""
    generator = _get_generator()
    return generator.generate_replies(
        messages=messages_dict,
        chat_id=chat_id,
        num_replies=1,
        user_name=user_name,
    )


@router.post("/replies")
async def generate_replies(request: dict[str, Any]) -> dict[str, Any]:
    """Generate reply suggestions for a conversation."""
    chat_id = request.get("chat_id")
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")

    # Fetch messages in thread pool to avoid blocking async event loop
    messages, error = await asyncio.to_thread(
        _fetch_messages, chat_id, settings.api.generation_context_limit
    )

    if error:
        raise HTTPException(status_code=503, detail=error)

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Convert to dict format for generator
    messages_dict = [
        {
            "text": m.text,
            "sender": m.sender,
            "sender_name": m.sender_name,
            "is_from_me": m.is_from_me,
            "timestamp": m.timestamp,
        }
        for m in reversed(messages)
    ]

    # Get user name from request or use default
    user_name = request.get("user_name", "User")

    # Generate reply in thread pool to avoid blocking
    result = await asyncio.to_thread(_generate_reply, messages_dict, chat_id, user_name)

    return {
        "replies": [
            {
                "text": r.text,
                "reply_type": r.reply_type,
                "confidence": r.confidence,
            }
            for r in result.replies
        ],
        "chat_id": chat_id,
        "model_used": result.model_used,
        "generation_time_ms": result.generation_time_ms,
        "context_summary": result.context.summary,
    }
