"""Conversation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
)

router = APIRouter()


def _get_reader():
    """Get iMessage reader instance."""
    from v2.core.imessage import MessageReader

    reader = MessageReader()
    if not reader.check_access():
        raise HTTPException(
            status_code=503,
            detail="Cannot access iMessage database. Grant Full Disk Access permission.",
        )
    return reader


@router.get("", response_model=ConversationListResponse)
async def list_conversations(limit: int = 50) -> ConversationListResponse:
    """List recent conversations."""
    reader = _get_reader()

    try:
        conversations = reader.get_conversations(limit=limit)

        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    chat_id=c.chat_id,
                    display_name=c.display_name,
                    participants=c.participants,
                    last_message_date=c.last_message_date,
                    last_message_text=c.last_message_text,
                    message_count=c.message_count,
                    is_group=c.is_group,
                )
                for c in conversations
            ],
            total=len(conversations),
        )
    finally:
        reader.close()


@router.get("/{chat_id}/messages", response_model=MessageListResponse)
async def get_messages(chat_id: str, limit: int = 50) -> MessageListResponse:
    """Get messages for a conversation."""
    reader = _get_reader()

    try:
        messages = reader.get_messages(chat_id=chat_id, limit=limit)

        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return MessageListResponse(
            messages=[
                MessageResponse(
                    id=m.id,
                    text=m.text,
                    sender=m.sender,
                    is_from_me=m.is_from_me,
                    timestamp=m.timestamp,
                    chat_id=m.chat_id,
                )
                for m in messages
            ],
            chat_id=chat_id,
            total=len(messages),
        )
    finally:
        reader.close()
