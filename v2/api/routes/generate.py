"""Reply generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import (
    GeneratedReplyResponse,
    GenerateRepliesRequest,
    GenerateRepliesResponse,
)

router = APIRouter()


@router.post("/replies", response_model=GenerateRepliesResponse)
async def generate_replies(request: GenerateRepliesRequest) -> GenerateRepliesResponse:
    """Generate reply suggestions for a conversation."""
    from v2.core.generation import ReplyGenerator
    from v2.core.imessage import MessageReader
    from v2.core.models import get_model_loader

    # Get conversation messages
    reader = MessageReader()
    if not reader.check_access():
        raise HTTPException(
            status_code=503,
            detail="Cannot access iMessage database",
        )

    try:
        messages = reader.get_messages(chat_id=request.chat_id, limit=30)

        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Convert to dict format for generator
        messages_dict = [
            {
                "text": m.text,
                "sender": m.sender,
                "is_from_me": m.is_from_me,
                "timestamp": m.timestamp,
            }
            for m in reversed(messages)  # Oldest first for context
        ]

        # Generate replies
        loader = get_model_loader()
        generator = ReplyGenerator(loader)

        result = generator.generate_replies(
            messages=messages_dict,
            chat_id=request.chat_id,
            num_replies=request.num_replies,
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
        )

    finally:
        reader.close()
