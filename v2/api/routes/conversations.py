"""Conversation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# Configuration
MAX_PRELOAD_CHATS = 20  # Maximum number of chat indices to preload at once

from ..schemas import (
    ContactProfileResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
    PreloadIndicesRequest,
    PreloadIndicesResponse,
    SendMessageRequest,
    SendMessageResponse,
    TopicClusterResponse,
)

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
                    last_message_is_from_me=c.last_message_is_from_me,
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
async def get_messages(
    chat_id: str,
    limit: int = 50,
    before: str | None = None,
) -> MessageListResponse:
    """Get messages for a conversation.

    Args:
        chat_id: Conversation ID
        limit: Maximum number of messages to return
        before: ISO timestamp - only return messages before this time (for pagination)
    """
    from datetime import datetime

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

        return MessageListResponse(
            messages=[
                MessageResponse(
                    id=m.id,
                    text=m.text,
                    sender=m.sender,
                    sender_name=m.sender_name,
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


@router.post("/send", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest) -> SendMessageResponse:
    """Send a message to a conversation via AppleScript.

    Requires Automation permission for Messages.app.
    """
    from core.imessage.sender import send_message as do_send

    # Use is_group from request (frontend knows from conversation data)
    # Fallback to heuristic if not provided (for backwards compatibility)
    is_group = request.is_group

    result = do_send(
        text=request.text,
        chat_id=request.chat_id,
        is_group=is_group,
    )

    return SendMessageResponse(
        success=result.success,
        error=result.error,
    )


@router.get("/{chat_id}/profile", response_model=ContactProfileResponse)
async def get_contact_profile(chat_id: str) -> ContactProfileResponse:
    """Get a rich profile of a contact based on message history.

    Analyzes the conversation to determine:
    - Relationship type (close friend, family, coworker, etc.)
    - Communication tone (casual, playful, formal)
    - Common topics discussed
    - Communication patterns
    """
    from core.embeddings import get_contact_profile as build_profile

    try:
        profile = build_profile(chat_id)

        return ContactProfileResponse(
            chat_id=profile.chat_id,
            display_name=profile.display_name,
            relationship_type=profile.relationship_type,
            relationship_confidence=profile.relationship_confidence,
            total_messages=profile.total_messages,
            you_sent=profile.you_sent,
            they_sent=profile.they_sent,
            avg_your_length=profile.avg_your_length,
            avg_their_length=profile.avg_their_length,
            tone=profile.tone,
            uses_emoji=profile.uses_emoji,
            uses_slang=profile.uses_slang,
            is_playful=profile.is_playful,
            topics=[
                TopicClusterResponse(
                    name=t.name,
                    keywords=t.keywords,
                    message_count=t.message_count,
                    percentage=t.percentage,
                )
                for t in profile.topics
            ],
            most_active_hours=profile.most_active_hours,
            first_message_date=profile.first_message_date,
            last_message_date=profile.last_message_date,
            their_common_phrases=profile.their_common_phrases,
            your_common_phrases=profile.your_common_phrases,
            summary=profile.summary,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to build contact profile")


@router.post("/preload", response_model=PreloadIndicesResponse)
async def preload_indices(request: PreloadIndicesRequest) -> PreloadIndicesResponse:
    """Preload FAISS indices for conversations in background.

    Call this for visible conversations to ensure instant search when selected.
    Indices are built in background threads and cached to disk.
    """
    import threading

    from core.embeddings import get_embedding_store

    store = get_embedding_store()
    already_cached = 0
    to_preload = []

    # Check which need preloading
    for chat_id in request.chat_ids[:MAX_PRELOAD_CHATS]:
        if store.is_index_ready(chat_id, only_from_me=False):
            already_cached += 1
        else:
            to_preload.append(chat_id)

    # Preload in background threads
    def _preload_one(cid: str) -> None:
        try:
            store._get_or_build_faiss_index(cid, only_from_me=False)
        except Exception as e:
            logger.debug(f"Index preload failed for {cid}: {e}")

    for chat_id in to_preload:
        thread = threading.Thread(target=_preload_one, args=(chat_id,), daemon=True)
        thread.start()

    return PreloadIndicesResponse(
        preloading=len(to_preload),
        already_cached=already_cached,
        message=f"Preloading {len(to_preload)} indices, {already_cached} already cached",
    )
