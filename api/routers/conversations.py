"""Conversations API endpoints.

Provides endpoints for listing conversations, retrieving messages,
searching across conversations, and sending messages.

All endpoints require Full Disk Access permission to read the iMessage database.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ConversationResponse,
    ErrorResponse,
    MessageResponse,
    SendAttachmentRequest,
    SendMessageRequest,
    SendMessageResponse,
)
from integrations.imessage import ChatDBReader, IMessageSender

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(
    "",
    response_model=list[ConversationResponse],
    response_model_exclude_unset=True,
    response_description="List of conversations sorted by last message date",
    summary="List recent conversations",
    responses={
        200: {
            "description": "Conversations retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "chat_id": "chat123456789",
                            "participants": ["+15551234567"],
                            "display_name": "John Doe",
                            "last_message_date": "2024-01-15T10:30:00Z",
                            "message_count": 150,
                            "is_group": False,
                            "last_message_text": "See you later!",
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def list_conversations(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of conversations to return",
        examples=[50, 100],
    ),
    since: datetime | None = Query(
        default=None,
        description="Only return conversations with messages after this date (ISO 8601 format)",
        examples=["2024-01-01T00:00:00Z"],
    ),
    before: datetime | None = Query(
        default=None,
        description="Pagination cursor - only return conversations before this date",
        examples=["2024-01-15T10:30:00Z"],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[ConversationResponse]:
    """List recent iMessage conversations.

    Returns conversations sorted by last message date (newest first).
    Supports pagination using the `before` parameter with the last conversation's
    `last_message_date` value.

    **Pagination Example:**
    1. First request: `GET /conversations?limit=50`
    2. Get `last_message_date` from the last item in the response
    3. Next request: `GET /conversations?limit=50&before=2024-01-10T08:00:00Z`

    **Filtering by Date:**
    Use the `since` parameter to only return conversations that have had
    activity after a specific date. This is useful for syncing updates.

    **Example Response:**
    ```json
    [
        {
            "chat_id": "chat123456789",
            "participants": ["+15551234567"],
            "display_name": "John Doe",
            "last_message_date": "2024-01-15T10:30:00Z",
            "message_count": 150,
            "is_group": false,
            "last_message_text": "See you later!"
        },
        {
            "chat_id": "chat987654321",
            "participants": ["+15559876543", "+15551111111"],
            "display_name": "Family Group",
            "last_message_date": "2024-01-15T09:00:00Z",
            "message_count": 500,
            "is_group": true,
            "last_message_text": "Dinner at 7?"
        }
    ]
    ```

    Args:
        limit: Maximum conversations to return (1-500, default 50)
        since: Only conversations with messages after this date
        before: Pagination cursor for older conversations

    Returns:
        List of ConversationResponse objects sorted by last_message_date descending

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    conversations = reader.get_conversations(limit=limit, since=since, before=before)
    return [ConversationResponse.model_validate(c) for c in conversations]


@router.get(
    "/{chat_id}/messages",
    response_model=list[MessageResponse],
    response_model_exclude_unset=True,
    response_description="List of messages in the conversation",
    summary="Get messages for a conversation",
    responses={
        200: {
            "description": "Messages retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 12345,
                            "chat_id": "chat123456789",
                            "sender": "+15551234567",
                            "sender_name": "John Doe",
                            "text": "Hey, are you free for lunch?",
                            "date": "2024-01-15T10:30:00Z",
                            "is_from_me": False,
                            "attachments": [],
                            "reactions": [],
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_messages(
    chat_id: str,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of messages to return",
        examples=[100, 500],
    ),
    before: datetime | None = Query(
        default=None,
        description="Only return messages before this date (for pagination)",
        examples=["2024-01-15T10:30:00Z"],
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[MessageResponse]:
    """Get messages for a specific conversation.

    Returns messages sorted by date (newest first). Includes attachments,
    reactions (tapbacks), and threading information (reply_to_id).

    **Pagination:**
    Use the `before` parameter with the `date` of the last message to
    fetch older messages.

    **Message Types:**
    - Regular messages with text content
    - Messages with attachments (images, videos, files)
    - System messages (participant joined/left, name changes)
    - Threaded replies (check `reply_to_id`)

    **Example Response:**
    ```json
    [
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "text": "Hey, are you free for lunch?",
            "date": "2024-01-15T10:30:00Z",
            "is_from_me": false,
            "attachments": [],
            "reply_to_id": null,
            "reactions": [
                {
                    "type": "love",
                    "sender": "+15559876543",
                    "sender_name": "Jane",
                    "date": "2024-01-15T10:31:00Z"
                }
            ],
            "is_system_message": false
        }
    ]
    ```

    Args:
        chat_id: The unique conversation identifier
        limit: Maximum messages to return (1-1000, default 100)
        before: Only messages before this date (for pagination)

    Returns:
        List of MessageResponse objects sorted by date descending

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    messages = reader.get_messages(chat_id=chat_id, limit=limit, before=before)
    return [MessageResponse.model_validate(m) for m in messages]


@router.get(
    "/search",
    response_model=list[MessageResponse],
    response_model_exclude_unset=True,
    response_description="List of messages matching the search query",
    summary="Search messages across conversations",
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 12345,
                            "chat_id": "chat123456789",
                            "sender": "+15551234567",
                            "sender_name": "John Doe",
                            "text": "Let's meet for dinner tomorrow",
                            "date": "2024-01-15T10:30:00Z",
                            "is_from_me": False,
                            "attachments": [],
                            "reactions": [],
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def search_messages(
    q: str = Query(
        ...,
        min_length=1,
        description="Search query - matches against message text",
        examples=["dinner", "meeting tomorrow"],
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results to return",
        examples=[50, 100],
    ),
    sender: str | None = Query(
        default=None,
        description="Filter by sender phone number or email",
        examples=["+15551234567", "john@example.com"],
    ),
    after: datetime | None = Query(
        default=None,
        description="Only messages after this date",
        examples=["2024-01-01T00:00:00Z"],
    ),
    before: datetime | None = Query(
        default=None,
        description="Only messages before this date",
        examples=["2024-01-15T23:59:59Z"],
    ),
    chat_id: str | None = Query(
        default=None,
        description="Filter to a specific conversation",
        examples=["chat123456789"],
    ),
    has_attachments: bool | None = Query(
        default=None,
        description="Filter by presence of attachments (true/false)",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[MessageResponse]:
    """Search messages across all conversations.

    Performs a text search across all message content with optional filters
    for sender, date range, conversation, and attachments.

    **Search Tips:**
    - The query is case-insensitive
    - Use specific phrases for better results
    - Combine with date filters to narrow down results

    **Filtering Examples:**
    - Find messages from a specific person: `?q=meeting&sender=+15551234567`
    - Find recent messages: `?q=dinner&after=2024-01-01T00:00:00Z`
    - Find messages with photos: `?q=photo&has_attachments=true`

    **Example Response:**
    ```json
    [
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "text": "Let's meet for dinner tomorrow at 7pm",
            "date": "2024-01-15T10:30:00Z",
            "is_from_me": false,
            "attachments": [],
            "reactions": []
        }
    ]
    ```

    Args:
        q: Search query (required, min 1 character)
        limit: Maximum results to return (1-500, default 50)
        sender: Filter by sender phone/email
        after: Only messages after this date
        before: Only messages before this date
        chat_id: Filter to specific conversation
        has_attachments: Filter by attachment presence

    Returns:
        List of MessageResponse objects matching the query

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
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


@router.post(
    "/{chat_id}/send",
    response_model=SendMessageResponse,
    response_model_exclude_unset=True,
    response_description="Result of the send operation",
    summary="Send a message to a conversation",
    responses={
        200: {
            "description": "Message sent successfully",
            "content": {"application/json": {"example": {"success": True, "error": None}}},
        },
        400: {
            "description": "Invalid request (missing recipient for individual chat)",
            "model": ErrorResponse,
        },
    },
)
def send_message(
    chat_id: str,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """Send a text message to a conversation.

    Sends an iMessage to either an individual contact or a group chat.
    Requires Automation permission for the Messages app, which will be
    prompted on first use.

    **For Individual Chats:**
    - Set `is_group` to `false` (default)
    - Provide the `recipient` phone number or email
    - The `chat_id` is used for reference only

    **For Group Chats:**
    - Set `is_group` to `true`
    - The `chat_id` is used to target the group
    - `recipient` is not required

    **Example Request (Individual):**
    ```json
    {
        "text": "Hey, are you free for lunch?",
        "recipient": "+15551234567",
        "is_group": false
    }
    ```

    **Example Request (Group):**
    ```json
    {
        "text": "Hey everyone!",
        "is_group": true
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "error": null
    }
    ```

    Args:
        chat_id: The conversation ID
        request: Message text, recipient (for individual), and is_group flag

    Returns:
        SendMessageResponse indicating success or failure

    Raises:
        HTTPException 400: Recipient required for individual chats
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


@router.post(
    "/{chat_id}/send-attachment",
    response_model=SendMessageResponse,
    response_model_exclude_unset=True,
    response_description="Result of the attachment send operation",
    summary="Send a file attachment to a conversation",
    responses={
        200: {
            "description": "Attachment sent successfully",
            "content": {"application/json": {"example": {"success": True, "error": None}}},
        },
        400: {
            "description": "Invalid request (missing recipient for individual chat)",
            "model": ErrorResponse,
        },
    },
)
def send_attachment(
    chat_id: str,
    request: SendAttachmentRequest,
) -> SendMessageResponse:
    """Send a file attachment to a conversation.

    Sends a file (image, video, document, etc.) via iMessage to either
    an individual contact or a group chat. Requires Automation permission
    for the Messages app.

    **Supported File Types:**
    - Images: jpg, png, gif, heic
    - Videos: mp4, mov
    - Documents: pdf, doc, docx
    - And more (any file the Messages app can handle)

    **For Individual Chats:**
    - Set `is_group` to `false` (default)
    - Provide the `recipient` phone number or email

    **For Group Chats:**
    - Set `is_group` to `true`
    - The `chat_id` is used to target the group

    **Example Request:**
    ```json
    {
        "file_path": "/Users/john/Documents/photo.jpg",
        "recipient": "+15551234567",
        "is_group": false
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "error": null
    }
    ```

    Args:
        chat_id: The conversation ID
        request: File path, recipient (for individual), and is_group flag

    Returns:
        SendMessageResponse indicating success or failure

    Raises:
        HTTPException 400: Recipient required for individual chats
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
