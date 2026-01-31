"""Priority Inbox API endpoints.

Provides endpoints for the smart priority inbox feature, including:
- Getting prioritized messages sorted by importance
- Marking messages as handled
- Managing important contacts

The priority scoring uses ML-based detection of questions, action items,
time-sensitive content, and contact importance.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_imessage_reader
from jarvis.priority import (
    PriorityLevel,
    get_priority_scorer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/priority", tags=["priority"])


class PriorityMessageResponse(BaseModel):
    """A message with its priority score and metadata.

    Example:
        ```json
        {
            "message_id": 12345,
            "chat_id": "chat123456",
            "sender": "+1234567890",
            "sender_name": "John Doe",
            "text": "Can you pick up milk on your way home?",
            "date": "2024-01-15T14:30:00",
            "priority_score": 0.85,
            "priority_level": "high",
            "reasons": ["action_requested", "contains_question"],
            "needs_response": true,
            "handled": false,
            "conversation_name": "John Doe"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_id": 12345,
                "chat_id": "chat123456",
                "sender": "+1234567890",
                "sender_name": "John Doe",
                "text": "Can you pick up milk on your way home?",
                "date": "2024-01-15T14:30:00",
                "priority_score": 0.85,
                "priority_level": "high",
                "reasons": ["action_requested", "contains_question"],
                "needs_response": True,
                "handled": False,
                "conversation_name": "John Doe",
            }
        }
    )

    message_id: int = Field(
        ...,
        description="Unique message ID",
    )
    chat_id: str = Field(
        ...,
        description="Conversation ID",
    )
    sender: str = Field(
        ...,
        description="Sender phone number or email",
    )
    sender_name: str | None = Field(
        None,
        description="Resolved sender name from contacts",
    )
    text: str = Field(
        ...,
        description="Message text content",
    )
    date: str = Field(
        ...,
        description="Message timestamp in ISO format",
    )
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical priority score (0.0 to 1.0)",
    )
    priority_level: str = Field(
        ...,
        description="Priority level: critical, high, medium, or low",
    )
    reasons: list[str] = Field(
        ...,
        description="Reasons contributing to the priority score",
    )
    needs_response: bool = Field(
        ...,
        description="Whether this message likely needs a reply",
    )
    handled: bool = Field(
        ...,
        description="Whether the user has marked this as handled",
    )
    conversation_name: str | None = Field(
        None,
        description="Display name for the conversation",
    )


class PriorityInboxResponse(BaseModel):
    """Response containing prioritized messages.

    Example:
        ```json
        {
            "messages": [...],
            "total_count": 25,
            "unhandled_count": 18,
            "needs_response_count": 12,
            "critical_count": 2,
            "high_count": 5
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [],
                "total_count": 25,
                "unhandled_count": 18,
                "needs_response_count": 12,
                "critical_count": 2,
                "high_count": 5,
            }
        }
    )

    messages: list[PriorityMessageResponse] = Field(
        ...,
        description="List of prioritized messages sorted by importance",
    )
    total_count: int = Field(
        ...,
        description="Total number of messages analyzed",
    )
    unhandled_count: int = Field(
        ...,
        description="Number of unhandled messages",
    )
    needs_response_count: int = Field(
        ...,
        description="Number of messages needing a response",
    )
    critical_count: int = Field(
        ...,
        description="Number of critical priority messages",
    )
    high_count: int = Field(
        ...,
        description="Number of high priority messages",
    )


class MarkHandledRequest(BaseModel):
    """Request to mark a message as handled.

    Example:
        ```json
        {
            "chat_id": "chat123456",
            "message_id": 12345
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456",
                "message_id": 12345,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID",
    )
    message_id: int = Field(
        ...,
        description="Message ID to mark as handled",
    )


class MarkHandledResponse(BaseModel):
    """Response after marking a message as handled.

    Example:
        ```json
        {
            "success": true,
            "chat_id": "chat123456",
            "message_id": 12345,
            "handled": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "chat_id": "chat123456",
                "message_id": 12345,
                "handled": True,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the operation succeeded",
    )
    chat_id: str = Field(
        ...,
        description="Conversation ID",
    )
    message_id: int = Field(
        ...,
        description="Message ID",
    )
    handled: bool = Field(
        ...,
        description="Current handled status",
    )


class ImportantContactRequest(BaseModel):
    """Request to mark a contact as important.

    Example:
        ```json
        {
            "identifier": "+1234567890",
            "important": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "identifier": "+1234567890",
                "important": True,
            }
        }
    )

    identifier: str = Field(
        ...,
        description="Contact phone number or email",
    )
    important: bool = Field(
        True,
        description="Whether to mark as important (true) or not (false)",
    )


class ImportantContactResponse(BaseModel):
    """Response after updating contact importance.

    Example:
        ```json
        {
            "success": true,
            "identifier": "+1234567890",
            "important": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "identifier": "+1234567890",
                "important": True,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the operation succeeded",
    )
    identifier: str = Field(
        ...,
        description="Contact identifier",
    )
    important: bool = Field(
        ...,
        description="Current importance status",
    )


def _get_conversation_name(
    reader: object, chat_id: str, conversations_cache: dict[str, str]
) -> str | None:
    """Get conversation display name with caching.

    Args:
        reader: iMessage reader instance
        chat_id: Conversation ID
        conversations_cache: Cache of chat_id -> display_name

    Returns:
        Display name or None
    """
    if chat_id in conversations_cache:
        return conversations_cache[chat_id]

    try:
        conversations = reader.get_conversations(limit=100)  # type: ignore[attr-defined]
        for conv in conversations:
            conversations_cache[conv.chat_id] = conv.display_name or ", ".join(
                conv.participants[:3]
            )
        return conversations_cache.get(chat_id)
    except Exception:
        return None


@router.get(
    "",
    response_model=PriorityInboxResponse,
    response_model_exclude_unset=True,
    response_description="Prioritized list of messages that need attention",
    summary="Get priority inbox",
    responses={
        200: {
            "description": "Priority inbox retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "messages": [
                            {
                                "message_id": 12345,
                                "chat_id": "chat123",
                                "sender": "+1234567890",
                                "sender_name": "John",
                                "text": "Can you call me back?",
                                "date": "2024-01-15T14:30:00",
                                "priority_score": 0.85,
                                "priority_level": "high",
                                "reasons": ["action_requested", "contains_question"],
                                "needs_response": True,
                                "handled": False,
                                "conversation_name": "John",
                            }
                        ],
                        "total_count": 25,
                        "unhandled_count": 18,
                        "needs_response_count": 12,
                        "critical_count": 2,
                        "high_count": 5,
                    }
                }
            },
        },
        503: {
            "description": "iMessage access unavailable",
        },
    },
)
def get_priority_inbox(
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=200,
            description="Maximum number of messages to analyze",
        ),
    ] = 50,
    include_handled: Annotated[
        bool,
        Query(
            description="Whether to include already-handled messages",
        ),
    ] = False,
    min_level: Annotated[
        str | None,
        Query(
            description="Minimum priority level to include (critical, high, medium, low)",
        ),
    ] = None,
) -> PriorityInboxResponse:
    """Get prioritized messages sorted by importance.

    Returns messages from recent conversations scored and sorted by priority.
    Messages are analyzed for questions, action items, time-sensitive content,
    and sender importance.

    **Priority Levels:**
    - **critical**: Score >= 0.8 - Urgent, needs immediate attention
    - **high**: Score >= 0.6 - Important, should respond soon
    - **medium**: Score >= 0.3 - Normal priority
    - **low**: Score < 0.3 - Can wait, informational

    **Priority Reasons:**
    - `contains_question`: Message contains a question
    - `action_requested`: Message requests an action
    - `time_sensitive`: Message mentions deadlines or urgency
    - `important_contact`: Sender is marked as important
    - `frequent_contact`: Sender messages frequently
    - `awaiting_response`: You haven't responded recently
    - `multiple_messages`: Multiple unanswered messages from sender

    **Example Request:**
    ```
    GET /priority?limit=50&include_handled=false&min_level=medium
    ```

    Args:
        limit: Maximum number of messages to analyze (1-200)
        include_handled: Whether to include handled messages
        min_level: Minimum priority level filter

    Returns:
        PriorityInboxResponse with prioritized messages and counts
    """
    reader = get_imessage_reader()
    scorer = get_priority_scorer()

    if not reader.check_access():
        raise HTTPException(
            status_code=503,
            detail="iMessage database access unavailable. Grant Full Disk Access permission.",
        )

    # Get recent conversations and their messages
    try:
        conversations = reader.get_conversations(limit=20)
    except Exception as e:
        logger.exception("Failed to get conversations")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read conversations: {e}",
        ) from e

    # Build conversation name cache
    conv_cache: dict[str, str] = {}
    for conv in conversations:
        conv_cache[conv.chat_id] = conv.display_name or ", ".join(conv.participants[:3])

    # Collect recent messages from each conversation
    all_messages = []
    for conv in conversations[:15]:  # Limit conversations to analyze
        try:
            messages = reader.get_messages(conv.chat_id, limit=limit // 5)
            all_messages.extend(messages)
        except Exception:
            continue

    # Score all messages
    priority_scores = scorer.score_messages(all_messages)

    # Filter based on parameters
    filtered_scores = []
    for score in priority_scores:
        # Skip handled if not requested
        if not include_handled and score.handled:
            continue

        # Filter by minimum level
        if min_level:
            try:
                min_level_enum = PriorityLevel(min_level.lower())
                level_order = {
                    PriorityLevel.CRITICAL: 4,
                    PriorityLevel.HIGH: 3,
                    PriorityLevel.MEDIUM: 2,
                    PriorityLevel.LOW: 1,
                }
                if level_order[score.level] < level_order[min_level_enum]:
                    continue
            except ValueError:
                logger.warning("Invalid priority level filter: %s", min_level)

        filtered_scores.append(score)

    # Build message lookup
    message_lookup = {(m.chat_id, m.id): m for m in all_messages}

    # Build response
    response_messages = []
    for score in filtered_scores[:limit]:
        message = message_lookup.get((score.chat_id, score.message_id))
        if not message:
            continue

        response_messages.append(
            PriorityMessageResponse(
                message_id=score.message_id,
                chat_id=score.chat_id,
                sender=message.sender,
                sender_name=message.sender_name,
                text=message.text,
                date=message.date.isoformat(),
                priority_score=score.score,
                priority_level=score.level.value,
                reasons=[r.value for r in score.reasons],
                needs_response=score.needs_response,
                handled=score.handled,
                conversation_name=conv_cache.get(score.chat_id),
            )
        )

    # Calculate counts
    critical_count = sum(1 for s in filtered_scores if s.level == PriorityLevel.CRITICAL)
    high_count = sum(1 for s in filtered_scores if s.level == PriorityLevel.HIGH)
    needs_response_count = sum(1 for s in filtered_scores if s.needs_response)
    unhandled_count = sum(1 for s in filtered_scores if not s.handled)

    return PriorityInboxResponse(
        messages=response_messages,
        total_count=len(priority_scores),
        unhandled_count=unhandled_count,
        needs_response_count=needs_response_count,
        critical_count=critical_count,
        high_count=high_count,
    )


@router.post(
    "/handled",
    response_model=MarkHandledResponse,
    response_description="Message marked as handled",
    summary="Mark message as handled",
    responses={
        200: {
            "description": "Message marked as handled successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "chat_id": "chat123",
                        "message_id": 12345,
                        "handled": True,
                    }
                }
            },
        },
    },
)
def mark_handled(request: MarkHandledRequest) -> MarkHandledResponse:
    """Mark a message as handled.

    Use this when the user has dealt with a message (responded, noted, etc.)
    to remove it from the priority inbox.

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456",
        "message_id": 12345
    }
    ```

    Args:
        request: MarkHandledRequest with chat_id and message_id

    Returns:
        MarkHandledResponse confirming the operation
    """
    scorer = get_priority_scorer()
    scorer.mark_handled(request.chat_id, request.message_id)

    return MarkHandledResponse(
        success=True,
        chat_id=request.chat_id,
        message_id=request.message_id,
        handled=True,
    )


@router.delete(
    "/handled",
    response_model=MarkHandledResponse,
    response_description="Message unmarked as handled",
    summary="Unmark message as handled",
    responses={
        200: {
            "description": "Message unmarked as handled successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "chat_id": "chat123",
                        "message_id": 12345,
                        "handled": False,
                    }
                }
            },
        },
    },
)
def unmark_handled(request: MarkHandledRequest) -> MarkHandledResponse:
    """Unmark a message as handled.

    Use this to restore a message to the priority inbox.

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456",
        "message_id": 12345
    }
    ```

    Args:
        request: MarkHandledRequest with chat_id and message_id

    Returns:
        MarkHandledResponse confirming the operation
    """
    scorer = get_priority_scorer()
    scorer.unmark_handled(request.chat_id, request.message_id)

    return MarkHandledResponse(
        success=True,
        chat_id=request.chat_id,
        message_id=request.message_id,
        handled=False,
    )


@router.post(
    "/contacts/important",
    response_model=ImportantContactResponse,
    response_description="Contact importance updated",
    summary="Mark contact as important",
    responses={
        200: {
            "description": "Contact importance updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "identifier": "+1234567890",
                        "important": True,
                    }
                }
            },
        },
    },
)
def mark_contact_important(request: ImportantContactRequest) -> ImportantContactResponse:
    """Mark or unmark a contact as important.

    Important contacts receive higher priority scores for their messages.
    Use this to ensure messages from key people are always prioritized.

    **Example Request:**
    ```json
    {
        "identifier": "+1234567890",
        "important": true
    }
    ```

    Args:
        request: ImportantContactRequest with identifier and importance flag

    Returns:
        ImportantContactResponse confirming the operation
    """
    scorer = get_priority_scorer()
    scorer.mark_contact_important(request.identifier, request.important)

    return ImportantContactResponse(
        success=True,
        identifier=request.identifier,
        important=request.important,
    )


@router.get(
    "/stats",
    response_description="Priority inbox statistics",
    summary="Get priority inbox stats",
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "handled_count": 15,
                        "important_contacts_count": 5,
                    }
                }
            },
        },
    },
)
def get_priority_stats() -> dict[str, int]:
    """Get priority inbox statistics.

    Returns counts of handled items and important contacts.

    Returns:
        Dictionary with handled_count and important_contacts_count
    """
    scorer = get_priority_scorer()

    return {
        "handled_count": scorer.get_handled_count(),
        "important_contacts_count": len(scorer._important_contacts),
    }


@router.post(
    "/clear-handled",
    response_description="All handled items cleared",
    summary="Clear all handled items",
    responses={
        200: {
            "description": "Handled items cleared successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "All handled items cleared",
                    }
                }
            },
        },
    },
)
def clear_handled() -> dict[str, str | bool]:
    """Clear all handled items.

    Resets the handled status of all messages, restoring them to the
    priority inbox if they meet priority criteria.

    Returns:
        Success confirmation
    """
    scorer = get_priority_scorer()
    scorer.clear_handled()

    return {
        "success": True,
        "message": "All handled items cleared",
    }
