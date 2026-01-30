"""Reply generation endpoints."""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException

from ..schemas import (
    GeneratedReplyResponse,
    GenerateRepliesRequest,
    GenerateRepliesResponse,
    GenerationDebugInfo,
    PastReplyResponse,
)

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


def _fetch_messages(chat_id: str, limit: int = 30):
    """Fetch messages from iMessage database (blocking I/O)."""
    from core.imessage import MessageReader

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


@router.post("/replies", response_model=GenerateRepliesResponse)
async def generate_replies(request: GenerateRepliesRequest) -> GenerateRepliesResponse:
    """Generate reply suggestions for a conversation."""
    from .settings import get_user_name

    # Fetch messages in thread pool to avoid blocking async event loop
    messages, error = await asyncio.to_thread(_fetch_messages, request.chat_id, 30)

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

    # Generate reply in thread pool to avoid blocking
    result = await asyncio.to_thread(
        _generate_reply, messages_dict, request.chat_id, get_user_name()
    )

    # Build debug info
    debug_info = GenerationDebugInfo(
        style_instructions=result.style_instructions,
        intent_detected=result.context.intent.value,
        past_replies_found=[
            PastReplyResponse(
                their_message=their_msg,
                your_reply=your_reply,
                similarity=sim,
            )
            for their_msg, your_reply, sim in result.past_replies
        ],
        full_prompt=result.prompt_used,
        formatted_prompt=result.formatted_prompt,
    )

    return GenerateRepliesResponse(
        replies=[
            GeneratedReplyResponse(
                text=r.text,
                reply_type=r.reply_type,
                confidence=r.confidence,
            )
            for r in result.replies
        ],
        chat_id=request.chat_id,
        model_used=result.model_used,
        generation_time_ms=result.generation_time_ms,
        context_summary=result.context.summary,
        debug=debug_info,
    )
