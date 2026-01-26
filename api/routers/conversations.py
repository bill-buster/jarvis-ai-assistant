"""Conversations API endpoints.

Provides endpoints for listing conversations and retrieving messages.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_imessage_reader
from api.schemas import ConversationResponse, MessageResponse
from integrations.imessage import ChatDBReader

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=500, description="Max conversations to return"),
    since: datetime | None = Query(default=None, description="Only convos with messages after"),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[ConversationResponse]:
    """List recent conversations.

    Returns conversations sorted by last message date (newest first).
    """
    conversations = reader.get_conversations(limit=limit, since=since)
    return [ConversationResponse.model_validate(c) for c in conversations]


@router.get("/{chat_id}/messages", response_model=list[MessageResponse])
def get_messages(
    chat_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Max messages to return"),
    before: datetime | None = Query(default=None, description="Only messages before this date"),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[MessageResponse]:
    """Get messages for a conversation.

    Returns messages sorted by date (newest first).
    """
    messages = reader.get_messages(chat_id=chat_id, limit=limit, before=before)
    return [MessageResponse.model_validate(m) for m in messages]


@router.get("/search", response_model=list[MessageResponse])
def search_messages(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    sender: str | None = Query(default=None, description="Filter by sender"),
    after: datetime | None = Query(default=None, description="Messages after this date"),
    before: datetime | None = Query(default=None, description="Messages before this date"),
    chat_id: str | None = Query(default=None, description="Filter by conversation"),
    has_attachments: bool | None = Query(default=None, description="Filter by attachments"),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[MessageResponse]:
    """Search messages across all conversations."""
    messages = reader.search(
        query=q,
        limit=limit,
        sender=sender,
        after=after,
        before=before,
        chat_id=chat_id,
        has_attachments=has_attachments,
    )
    return [MessageResponse.model_validate(m) for m in messages]
