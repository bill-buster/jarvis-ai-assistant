"""Conversation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from core.config import settings

router = APIRouter()


def _get_reader():
    """Get iMessage reader instance."""
    from core.imessage import MessageReader

    reader = MessageReader()
    if not reader.check_access():
        raise HTTPException(
            status_code=503,
            detail="Cannot access iMessage database. Grant Full Disk Access permission.",
        )
    return reader


@router.get("")
async def list_conversations(limit: int = 50) -> dict[str, Any]:
    """List recent conversations."""
    reader = _get_reader()

    try:
        conversations = reader.get_conversations(limit=limit)

        return {
            "conversations": [
                {
                    "chat_id": c.chat_id,
                    "display_name": c.display_name,
                    "participants": c.participants,
                    "last_message_date": c.last_message_date,
                    "last_message_text": c.last_message_text,
                    "last_message_is_from_me": c.last_message_is_from_me,
                    "message_count": c.message_count,
                    "is_group": c.is_group,
                }
                for c in conversations
            ],
            "total": len(conversations),
        }
    finally:
        reader.close()


@router.get("/{chat_id}/messages")
async def get_messages(
    chat_id: str,
    limit: int | None = None,
    before: str | None = None,
) -> dict[str, Any]:
    """Get messages for a conversation.

    Args:
        chat_id: Conversation ID
        limit: Maximum number of messages to return
        before: ISO timestamp - only return messages before this time (for pagination)
    """
    if limit is None:
        limit = settings.api.default_message_limit
    reader = _get_reader()

    try:
        # Parse before timestamp if provided
        before_dt = None
        if before:
            try:
                before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'before' timestamp format")

        messages = reader.get_messages(chat_id=chat_id, limit=limit, before=before_dt)

        if not messages and before is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "messages": [
                {
                    "id": m.id,
                    "text": m.text,
                    "sender": m.sender,
                    "sender_name": m.sender_name,
                    "is_from_me": m.is_from_me,
                    "timestamp": m.timestamp,
                    "chat_id": m.chat_id,
                }
                for m in messages
            ],
            "chat_id": chat_id,
            "total": len(messages),
        }
    finally:
        reader.close()
