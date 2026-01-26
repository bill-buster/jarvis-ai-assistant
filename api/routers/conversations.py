"""Conversations API endpoints.

Provides endpoints for listing conversations and retrieving messages.
Uses TTL caching for frequently accessed data.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ConversationResponse,
    MessageResponse,
    SendAttachmentRequest,
    SendMessageRequest,
    SendMessageResponse,
)
from integrations.imessage import ChatDBReader, IMessageSender
from jarvis.metrics import get_conversation_cache

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=500, description="Max conversations to return"),
    since: datetime | None = Query(default=None, description="Only convos with messages after"),
    before: datetime | None = Query(default=None, description="Pagination cursor"),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[ConversationResponse]:
    """List recent conversations.

    Uses TTL cache (30s) for repeated requests with same parameters.
    Returns conversations sorted by last message date (newest first).
    Use 'before' parameter with the last conversation's last_message_date for pagination.
    """
    # Build cache key from parameters
    since_str = since.isoformat() if since else "none"
    before_str = before.isoformat() if before else "none"
    cache_key = f"conversations:{limit}:{since_str}:{before_str}"

    # Check cache
    cache = get_conversation_cache()
    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[return-value]

    conversations = reader.get_conversations(limit=limit, since=since, before=before)
    result = [ConversationResponse.model_validate(c) for c in conversations]

    cache.set(cache_key, result)
    return result


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


@router.post("/{chat_id}/send", response_model=SendMessageResponse)
def send_message(
    chat_id: str,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """Send a message to a conversation.

    Requires Automation permission for Messages app (prompted on first use).

    For individual chats: provide recipient (phone/email).
    For group chats: set is_group=True (uses chat_id to target group).

    Args:
        chat_id: The conversation ID
        request: Message text, recipient (for individual), and is_group flag
    """
    sender = IMessageSender()

    if request.is_group:
        # For group chats, send to the chat ID directly
        result = sender.send_message(
            text=request.text,
            chat_id=chat_id,
            is_group=True,
        )
    else:
        # For individual chats, send to the recipient
        if not request.recipient:
            raise HTTPException(
                status_code=400,
                detail="Recipient is required for individual chats",
            )
        result = sender.send_message(
            text=request.text,
            recipient=request.recipient,
        )

    return SendMessageResponse(success=result.success, error=result.error)


@router.post("/{chat_id}/send-attachment", response_model=SendMessageResponse)
def send_attachment(
    chat_id: str,
    request: SendAttachmentRequest,
) -> SendMessageResponse:
    """Send a file attachment to a conversation.

    Requires Automation permission for Messages app (prompted on first use).

    Args:
        chat_id: The conversation ID
        request: File path, recipient (for individual), and is_group flag
    """
    sender = IMessageSender()

    if request.is_group:
        result = sender.send_attachment(
            file_path=request.file_path,
            chat_id=chat_id,
            is_group=True,
        )
    else:
        if not request.recipient:
            raise HTTPException(
                status_code=400,
                detail="Recipient is required for individual chats",
            )
        result = sender.send_attachment(
            file_path=request.file_path,
            recipient=request.recipient,
        )

    return SendMessageResponse(success=result.success, error=result.error)
