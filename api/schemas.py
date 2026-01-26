"""Pydantic schemas for API responses.

Converts dataclasses from contracts/ to Pydantic models for FastAPI serialization.
All schemas include OpenAPI metadata for automatic documentation generation.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# iMessage Data Models
# =============================================================================


class AttachmentResponse(BaseModel):
    """iMessage attachment metadata.

    Represents a file attachment (image, video, document, etc.) sent or
    received in an iMessage conversation.

    Example:
        ```json
        {
            "filename": "IMG_1234.jpg",
            "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
            "mime_type": "image/jpeg",
            "file_size": 245760
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "filename": "IMG_1234.jpg",
                "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
                "mime_type": "image/jpeg",
                "file_size": 245760,
            }
        },
    )

    filename: str = Field(
        ...,
        description="Original filename of the attachment",
        examples=["IMG_1234.jpg", "Document.pdf"],
    )
    file_path: str | None = Field(
        default=None,
        description="Absolute path to the attachment file on disk",
        examples=["~/Library/Messages/Attachments/.../IMG_1234.jpg"],
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type of the attachment",
        examples=["image/jpeg", "application/pdf", "video/mp4"],
    )
    file_size: int | None = Field(
        default=None,
        description="File size in bytes",
        examples=[245760, 1048576],
        ge=0,
    )


class ReactionResponse(BaseModel):
    """iMessage tapback reaction.

    Represents a tapback reaction (love, like, dislike, laugh, emphasis,
    question) added to a message by a participant.

    Example:
        ```json
        {
            "type": "love",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "date": "2024-01-15T10:30:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "type": "love",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "date": "2024-01-15T10:30:00Z",
            }
        },
    )

    type: str = Field(
        ...,
        description="Tapback reaction type",
        examples=["love", "like", "dislike", "laugh", "emphasis", "question"],
    )
    sender: str = Field(
        ...,
        description="Phone number or email of the person who added the reaction",
        examples=["+15551234567", "john@example.com"],
    )
    sender_name: str | None = Field(
        default=None,
        description="Display name of the sender from Contacts",
        examples=["John Doe", "Mom"],
    )
    date: datetime = Field(
        ...,
        description="Timestamp when the reaction was added",
    )


class MessageResponse(BaseModel):
    """iMessage response model.

    Represents a single message in a conversation, including metadata,
    attachments, and reactions.

    Example:
        ```json
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "text": "Hey, are you free for lunch tomorrow?",
            "date": "2024-01-15T10:30:00Z",
            "is_from_me": false,
            "attachments": [],
            "reply_to_id": null,
            "reactions": [{"type": "love", "sender": "+15559876543"}],
            "is_system_message": false
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 12345,
                "chat_id": "chat123456789",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "text": "Hey, are you free for lunch tomorrow?",
                "date": "2024-01-15T10:30:00Z",
                "is_from_me": False,
                "attachments": [],
                "reply_to_id": None,
                "reactions": [],
                "date_delivered": None,
                "date_read": None,
                "is_system_message": False,
            }
        },
    )

    id: int = Field(
        ...,
        description="Unique message identifier from the iMessage database",
        examples=[12345, 67890],
    )
    chat_id: str = Field(
        ...,
        description="Conversation identifier this message belongs to",
        examples=["chat123456789", "iMessage;-;+15551234567"],
    )
    sender: str = Field(
        ...,
        description="Phone number or email of the sender",
        examples=["+15551234567", "john@example.com"],
    )
    sender_name: str | None = Field(
        default=None,
        description="Display name of the sender from Contacts",
        examples=["John Doe", "Mom"],
    )
    text: str = Field(
        ...,
        description="Message text content",
        examples=["Hey, are you free for lunch tomorrow?", "Sounds great!"],
    )
    date: datetime = Field(
        ...,
        description="Timestamp when the message was sent/received",
    )
    is_from_me: bool = Field(
        ...,
        description="True if the message was sent by the current user",
    )
    attachments: list[AttachmentResponse] = Field(
        default_factory=list,
        description="List of file attachments in this message",
    )
    reply_to_id: int | None = Field(
        default=None,
        description="Message ID this is a reply to (threaded conversation)",
        examples=[12340, None],
    )
    reactions: list[ReactionResponse] = Field(
        default_factory=list,
        description="List of tapback reactions on this message",
    )
    date_delivered: datetime | None = Field(
        default=None,
        description="Timestamp when the message was delivered",
    )
    date_read: datetime | None = Field(
        default=None,
        description="Timestamp when the message was read",
    )
    is_system_message: bool = Field(
        default=False,
        description="True for system messages (e.g., 'John left the group')",
    )


class ConversationResponse(BaseModel):
    """Conversation summary response model.

    Provides a summary of a conversation including participants, message count,
    and preview of the last message.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "participants": ["+15551234567", "+15559876543"],
            "display_name": "Family Group",
            "last_message_date": "2024-01-15T10:30:00Z",
            "message_count": 150,
            "is_group": true,
            "last_message_text": "See you at dinner!"
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "participants": ["+15551234567", "+15559876543"],
                "display_name": "Family Group",
                "last_message_date": "2024-01-15T10:30:00Z",
                "message_count": 150,
                "is_group": True,
                "last_message_text": "See you at dinner!",
            }
        },
    )

    chat_id: str = Field(
        ...,
        description="Unique conversation identifier",
        examples=["chat123456789", "iMessage;-;+15551234567"],
    )
    participants: list[str] = Field(
        ...,
        description="List of participant phone numbers or emails",
        examples=[["+15551234567", "+15559876543"]],
    )
    display_name: str | None = Field(
        default=None,
        description="Group name or contact display name",
        examples=["Family Group", "John Doe"],
    )
    last_message_date: datetime = Field(
        ...,
        description="Timestamp of the most recent message",
    )
    message_count: int = Field(
        ...,
        description="Total number of messages in the conversation",
        examples=[150, 42],
        ge=0,
    )
    is_group: bool = Field(
        ...,
        description="True if this is a group conversation",
    )
    last_message_text: str | None = Field(
        default=None,
        description="Preview of the most recent message text",
        examples=["See you at dinner!", "Thanks!"],
    )


# =============================================================================
# Health & System Models
# =============================================================================


class ModelInfo(BaseModel):
    """Current model information.

    Provides details about the currently loaded MLX language model.

    Example:
        ```json
        {
            "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "display_name": "Qwen 0.5B (Fast)",
            "loaded": true,
            "memory_usage_mb": 450.5,
            "quality_tier": "basic"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "display_name": "Qwen 0.5B (Fast)",
                "loaded": True,
                "memory_usage_mb": 450.5,
                "quality_tier": "basic",
            }
        }
    )

    id: str | None = Field(
        default=None,
        description="Model identifier (HuggingFace path)",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    display_name: str = Field(
        ...,
        description="Human-readable model name",
        examples=["Qwen 0.5B (Fast)", "Qwen 1.5B (Balanced)"],
    )
    loaded: bool = Field(
        ...,
        description="True if the model is currently loaded in memory",
    )
    memory_usage_mb: float = Field(
        ...,
        description="Current memory usage of the model in megabytes",
        examples=[450.5, 1024.0],
        ge=0,
    )
    quality_tier: str | None = Field(
        default=None,
        description="Quality tier: 'basic', 'good', or 'best'",
        examples=["basic", "good", "best"],
    )


class HealthResponse(BaseModel):
    """System health status response.

    Comprehensive health check including memory, permissions, model state,
    and overall system status.

    Example:
        ```json
        {
            "status": "healthy",
            "imessage_access": true,
            "memory_available_gb": 12.5,
            "memory_used_gb": 3.5,
            "memory_mode": "FULL",
            "model_loaded": true,
            "permissions_ok": true,
            "jarvis_rss_mb": 256.5,
            "jarvis_vms_mb": 1024.0
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "imessage_access": True,
                "memory_available_gb": 12.5,
                "memory_used_gb": 3.5,
                "memory_mode": "FULL",
                "model_loaded": True,
                "permissions_ok": True,
                "details": None,
                "jarvis_rss_mb": 256.5,
                "jarvis_vms_mb": 1024.0,
                "model": None,
                "recommended_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "system_ram_gb": 16.0,
            }
        }
    )

    status: str = Field(
        ...,
        description="Overall health status: 'healthy', 'degraded', or 'unhealthy'",
        examples=["healthy", "degraded", "unhealthy"],
    )
    imessage_access: bool = Field(
        ...,
        description="True if Full Disk Access is granted for iMessage database",
    )
    memory_available_gb: float = Field(
        ...,
        description="Available system memory in gigabytes",
        examples=[12.5, 4.0],
        ge=0,
    )
    memory_used_gb: float = Field(
        ...,
        description="Used system memory in gigabytes",
        examples=[3.5, 8.0],
        ge=0,
    )
    memory_mode: str = Field(
        ...,
        description="Memory controller mode: 'FULL', 'LITE', or 'MINIMAL'",
        examples=["FULL", "LITE", "MINIMAL"],
    )
    model_loaded: bool = Field(
        ...,
        description="True if the MLX language model is loaded",
    )
    permissions_ok: bool = Field(
        ...,
        description="True if all required permissions are granted",
    )
    details: dict[str, str] | None = Field(
        default=None,
        description="Additional details about any issues detected",
        examples=[{"imessage": "Full Disk Access required"}],
    )
    jarvis_rss_mb: float = Field(
        default=0.0,
        description="JARVIS process Resident Set Size (actual RAM usage) in MB",
        examples=[256.5, 512.0],
        ge=0,
    )
    jarvis_vms_mb: float = Field(
        default=0.0,
        description="JARVIS process Virtual Memory Size in MB",
        examples=[1024.0, 2048.0],
        ge=0,
    )
    model: ModelInfo | None = Field(
        default=None,
        description="Information about the currently loaded model",
    )
    recommended_model: str | None = Field(
        default=None,
        description="Recommended model ID based on system RAM",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    system_ram_gb: float | None = Field(
        default=None,
        description="Total system RAM in gigabytes",
        examples=[16.0, 32.0],
        ge=0,
    )


class ErrorResponse(BaseModel):
    """Standardized error response model.

    All API errors return this format for consistent client handling.

    Attributes:
        error: The exception class name (e.g., "ValidationError", "ModelLoadError").
        code: Machine-readable error code (e.g., "VAL_INVALID_INPUT", "MDL_LOAD_FAILED").
        detail: Human-readable error message describing what went wrong.
        details: Optional additional context about the error (field names, paths, etc.).

    Example Response (400 Bad Request):
        {
            "error": "ValidationError",
            "code": "VAL_MISSING_REQUIRED",
            "detail": "Missing required field: email",
            "details": {"field": "email"}
        }

    Example Response (403 Forbidden):
        {
            "error": "iMessageAccessError",
            "code": "MSG_ACCESS_DENIED",
            "detail": "Full Disk Access is required to read iMessages",
            "details": {
                "requires_permission": true,
                "permission_instructions": [
                    "Open System Settings",
                    "Go to Privacy & Security > Full Disk Access",
                    "Add and enable your terminal application",
                    "Restart JARVIS"
                ]
            }
        }

    Example Response (503 Service Unavailable):
        {
            "error": "ModelLoadError",
            "code": "RES_MEMORY_EXHAUSTED",
            "detail": "Insufficient memory to load model: qwen-3b",
            "details": {"available_mb": 1024, "required_mb": 2048}
        }
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "code": "VAL_INVALID_INPUT",
                "detail": "Missing required field: email",
                "details": {"field": "email"},
            }
        }
    )

    error: str = Field(
        ...,
        description="Exception class name (e.g., 'ValidationError')",
        examples=["ValidationError", "ModelLoadError", "iMessageAccessError"],
    )
    code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling",
        examples=["VAL_INVALID_INPUT", "MDL_LOAD_FAILED", "MSG_ACCESS_DENIED"],
    )
    detail: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Missing required field: email", "Failed to load model"],
    )
    details: dict[str, str | int | float | bool | list[str]] | None = Field(
        default=None,
        description="Additional context about the error (optional)",
        examples=[{"field": "email"}, {"available_mb": 1024, "required_mb": 2048}],
    )


# =============================================================================
# Messaging Models
# =============================================================================


class SendMessageRequest(BaseModel):
    """Request to send an iMessage.

    Send a text message to an individual or group conversation.

    Example:
        ```json
        {
            "text": "Hey, are you free for lunch?",
            "recipient": "+15551234567",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hey, are you free for lunch?",
                "recipient": "+15551234567",
                "is_group": False,
            }
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message text content to send",
        examples=["Hey, are you free for lunch?", "Thanks!"],
    )
    recipient: str | None = Field(
        default=None,
        description="Recipient phone number or email (required for individual chats)",
        examples=["+15551234567", "john@example.com"],
    )
    is_group: bool = Field(
        default=False,
        description="Set to true for group chats (uses chat_id from path)",
    )


class SendAttachmentRequest(BaseModel):
    """Request to send a file attachment.

    Send a file (image, document, etc.) to an individual or group conversation.

    Example:
        ```json
        {
            "file_path": "/Users/john/Documents/photo.jpg",
            "recipient": "+15551234567",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/Users/john/Documents/photo.jpg",
                "recipient": "+15551234567",
                "is_group": False,
            }
        }
    )

    file_path: str = Field(
        ...,
        description="Absolute path to the file to send",
        examples=["/Users/john/Documents/photo.jpg"],
    )
    recipient: str | None = Field(
        default=None,
        description="Recipient phone number or email (required for individual chats)",
        examples=["+15551234567", "john@example.com"],
    )
    is_group: bool = Field(
        default=False,
        description="Set to true for group chats (uses chat_id from path)",
    )


class SendMessageResponse(BaseModel):
    """Response after sending a message.

    Indicates whether the send operation succeeded or failed.

    Example:
        ```json
        {
            "success": true,
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="True if the message was sent successfully",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the send failed",
        examples=[None, "Automation permission denied"],
    )


# =============================================================================
# Draft API Schemas
# =============================================================================


class DraftSuggestion(BaseModel):
    """A suggested reply for a conversation.

    AI-generated reply suggestion with confidence score.

    Example:
        ```json
        {
            "text": "Sounds great! What time works for you?",
            "confidence": 0.85
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Sounds great! What time works for you?",
                "confidence": 0.85,
            }
        }
    )

    text: str = Field(
        ...,
        description="Suggested reply text",
        examples=["Sounds great! What time works for you?", "Sure, I'll take a look!"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) indicating relevance",
        examples=[0.85, 0.7],
    )


class ContextInfo(BaseModel):
    """Information about the context used to generate a reply.

    Metadata about the conversation context used for AI generation.

    Example:
        ```json
        {
            "num_messages": 20,
            "participants": ["John Doe", "Jane Smith"],
            "last_message": "Are you free for dinner?"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "num_messages": 20,
                "participants": ["John Doe", "Jane Smith"],
                "last_message": "Are you free for dinner?",
            }
        }
    )

    num_messages: int = Field(
        ...,
        description="Number of messages used for context",
        examples=[20, 50],
        ge=0,
    )
    participants: list[str] = Field(
        ...,
        description="Names of conversation participants",
        examples=[["John Doe", "Jane Smith"]],
    )
    last_message: str | None = Field(
        default=None,
        description="The most recent message in the conversation",
        examples=["Are you free for dinner?"],
    )


class DraftReplyRequest(BaseModel):
    """Request for draft reply generation.

    Request AI-generated reply suggestions for a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "instruction": "accept enthusiastically",
            "num_suggestions": 3,
            "context_messages": 20
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "instruction": "accept enthusiastically",
                "num_suggestions": 3,
                "context_messages": 20,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to generate replies for",
        examples=["chat123456789"],
    )
    instruction: str | None = Field(
        default=None,
        description="Optional instruction for reply tone/content",
        examples=["accept enthusiastically", "politely decline", "ask for more details"],
    )
    num_suggestions: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of reply suggestions to generate (1-5)",
    )
    context_messages: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of previous messages to use for context (5-50)",
    )


class DraftReplyResponse(BaseModel):
    """Response with generated reply suggestions.

    Contains AI-generated reply suggestions with context metadata.

    Example:
        ```json
        {
            "suggestions": [
                {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                {"text": "Sure! Let me know the details.", "confidence": 0.8}
            ],
            "context_used": {
                "num_messages": 20,
                "participants": ["John"],
                "last_message": "Dinner tonight?"
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestions": [
                    {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                    {"text": "Sure! Let me know the details.", "confidence": 0.8},
                ],
                "context_used": {
                    "num_messages": 20,
                    "participants": ["John"],
                    "last_message": "Dinner tonight?",
                },
            }
        }
    )

    suggestions: list[DraftSuggestion] = Field(
        ...,
        description="List of generated reply suggestions",
    )
    context_used: ContextInfo = Field(
        ...,
        description="Information about the context used for generation",
    )


class DraftSummaryRequest(BaseModel):
    """Request for conversation summarization.

    Request an AI-generated summary of a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "num_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "num_messages": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to summarize",
        examples=["chat123456789"],
    )
    num_messages: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of messages to include in summary (10-200)",
    )


class DateRange(BaseModel):
    """Date range for a conversation summary.

    Represents the time span covered by a summary.

    Example:
        ```json
        {
            "start": "2024-01-01",
            "end": "2024-01-15"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": "2024-01-01",
                "end": "2024-01-15",
            }
        }
    )

    start: str = Field(
        ...,
        description="Start date of the summarized period (YYYY-MM-DD)",
        examples=["2024-01-01"],
    )
    end: str = Field(
        ...,
        description="End date of the summarized period (YYYY-MM-DD)",
        examples=["2024-01-15"],
    )


class DraftSummaryResponse(BaseModel):
    """Response with conversation summary.

    Contains AI-generated summary with key points and date range.

    Example:
        ```json
        {
            "summary": "Discussion about planning a weekend trip to the beach.",
            "key_points": [
                "Decided on Saturday departure",
                "Meeting at John's place at 9am",
                "Bringing snacks and sunscreen"
            ],
            "date_range": {"start": "2024-01-10", "end": "2024-01-15"}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": "Discussion about planning a weekend trip to the beach.",
                "key_points": [
                    "Decided on Saturday departure",
                    "Meeting at John's place at 9am",
                    "Bringing snacks and sunscreen",
                ],
                "date_range": {"start": "2024-01-10", "end": "2024-01-15"},
            }
        }
    )

    summary: str = Field(
        ...,
        description="Brief summary of the conversation",
        examples=["Discussion about planning a weekend trip to the beach."],
    )
    key_points: list[str] = Field(
        ...,
        description="List of key points extracted from the conversation",
    )
    date_range: DateRange = Field(
        ...,
        description="Date range of messages included in the summary",
    )


# =============================================================================
# Settings Schemas
# =============================================================================


class AvailableModelInfo(BaseModel):
    """Information about an available model.

    Details about a model that can be selected for use.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "name": "Qwen 0.5B (Fast)",
            "size_gb": 0.4,
            "quality_tier": "basic",
            "ram_requirement_gb": 4.0,
            "is_downloaded": true,
            "is_loaded": false,
            "is_recommended": false,
            "description": "Fastest responses, good for simple tasks"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "name": "Qwen 0.5B (Fast)",
                "size_gb": 0.4,
                "quality_tier": "basic",
                "ram_requirement_gb": 4.0,
                "is_downloaded": True,
                "is_loaded": False,
                "is_recommended": False,
                "description": "Fastest responses, good for simple tasks",
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Unique model identifier (HuggingFace path)",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        examples=["Qwen 0.5B (Fast)", "Qwen 1.5B (Balanced)"],
    )
    size_gb: float = Field(
        ...,
        description="Model size on disk in gigabytes",
        examples=[0.4, 1.0, 2.0],
        ge=0,
    )
    quality_tier: str = Field(
        ...,
        description="Quality tier: 'basic', 'good', or 'best'",
        examples=["basic", "good", "best"],
    )
    ram_requirement_gb: float = Field(
        ...,
        description="Minimum RAM required to run this model",
        examples=[4.0, 8.0, 16.0],
        ge=0,
    )
    is_downloaded: bool = Field(
        ...,
        description="True if the model is downloaded locally",
    )
    is_loaded: bool = Field(
        ...,
        description="True if the model is currently loaded in memory",
    )
    is_recommended: bool = Field(
        ...,
        description="True if this is the recommended model for the system",
    )
    description: str | None = Field(
        default=None,
        description="Brief description of the model's characteristics",
        examples=["Fastest responses, good for simple tasks"],
    )


class GenerationSettings(BaseModel):
    """Generation parameter settings.

    Controls how the AI model generates text.

    Example:
        ```json
        {
            "temperature": 0.7,
            "max_tokens_reply": 150,
            "max_tokens_summary": 500
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 0.7,
                "max_tokens_reply": 150,
                "max_tokens_summary": 500,
            }
        }
    )

    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Sampling temperature (0.1=focused, 1.0=creative)",
        examples=[0.7, 0.5, 0.9],
    )
    max_tokens_reply: int = Field(
        default=150,
        ge=50,
        le=300,
        description="Maximum tokens for reply generation",
        examples=[150, 200],
    )
    max_tokens_summary: int = Field(
        default=500,
        ge=200,
        le=1000,
        description="Maximum tokens for summary generation",
        examples=[500, 750],
    )


class BehaviorSettings(BaseModel):
    """Behavior preference settings.

    Controls JARVIS behavior and defaults.

    Example:
        ```json
        {
            "auto_suggest_replies": true,
            "suggestion_count": 3,
            "context_messages_reply": 20,
            "context_messages_summary": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "auto_suggest_replies": True,
                "suggestion_count": 3,
                "context_messages_reply": 20,
                "context_messages_summary": 50,
            }
        }
    )

    auto_suggest_replies: bool = Field(
        default=True,
        description="Automatically suggest replies when viewing conversations",
    )
    suggestion_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Default number of reply suggestions to generate",
    )
    context_messages_reply: int = Field(
        default=20,
        ge=10,
        le=50,
        description="Default number of messages to use for reply context",
    )
    context_messages_summary: int = Field(
        default=50,
        ge=20,
        le=100,
        description="Default number of messages to use for summaries",
    )


class SystemInfo(BaseModel):
    """Read-only system information.

    Current system state including memory and model status.

    Example:
        ```json
        {
            "system_ram_gb": 16.0,
            "current_memory_usage_gb": 8.5,
            "model_loaded": true,
            "model_memory_usage_gb": 0.5,
            "imessage_access": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "system_ram_gb": 16.0,
                "current_memory_usage_gb": 8.5,
                "model_loaded": True,
                "model_memory_usage_gb": 0.5,
                "imessage_access": True,
            }
        }
    )

    system_ram_gb: float = Field(
        ...,
        description="Total system RAM in gigabytes",
        examples=[16.0, 32.0],
        ge=0,
    )
    current_memory_usage_gb: float = Field(
        ...,
        description="Current system memory usage in gigabytes",
        examples=[8.5, 12.0],
        ge=0,
    )
    model_loaded: bool = Field(
        ...,
        description="True if an AI model is currently loaded",
    )
    model_memory_usage_gb: float = Field(
        ...,
        description="Memory used by the loaded model in gigabytes",
        examples=[0.5, 1.5],
        ge=0,
    )
    imessage_access: bool = Field(
        ...,
        description="True if iMessage database access is available",
    )


class SettingsResponse(BaseModel):
    """Complete settings response.

    Full settings state including model, generation, behavior, and system info.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "generation": {"temperature": 0.7, "max_tokens_reply": 150},
            "behavior": {"auto_suggest_replies": true, "suggestion_count": 3},
            "system": {"system_ram_gb": 16.0, "model_loaded": true}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "generation": {
                    "temperature": 0.7,
                    "max_tokens_reply": 150,
                    "max_tokens_summary": 500,
                },
                "behavior": {
                    "auto_suggest_replies": True,
                    "suggestion_count": 3,
                    "context_messages_reply": 20,
                    "context_messages_summary": 50,
                },
                "system": {
                    "system_ram_gb": 16.0,
                    "current_memory_usage_gb": 8.5,
                    "model_loaded": True,
                    "model_memory_usage_gb": 0.5,
                    "imessage_access": True,
                },
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Currently selected model ID",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    generation: GenerationSettings = Field(
        ...,
        description="Generation parameter settings",
    )
    behavior: BehaviorSettings = Field(
        ...,
        description="Behavior preference settings",
    )
    system: SystemInfo = Field(
        ...,
        description="Read-only system information",
    )


class SettingsUpdateRequest(BaseModel):
    """Request to update settings.

    Partial update - only provided fields are changed.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "generation": {"temperature": 0.8}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "generation": {"temperature": 0.8},
            }
        }
    )

    model_id: str | None = Field(
        default=None,
        description="New model ID to switch to",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    generation: GenerationSettings | None = Field(
        default=None,
        description="Updated generation settings",
    )
    behavior: BehaviorSettings | None = Field(
        default=None,
        description="Updated behavior settings",
    )


class DownloadStatus(BaseModel):
    """Model download status.

    Current status of a model download operation.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "status": "downloading",
            "progress": 45.5,
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "status": "downloading",
                "progress": 45.5,
                "error": None,
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Model ID being downloaded",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    status: str = Field(
        ...,
        description="Download status: 'downloading', 'completed', or 'failed'",
        examples=["downloading", "completed", "failed"],
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Download progress percentage (0-100)",
        examples=[45.5, 100.0],
    )
    error: str | None = Field(
        default=None,
        description="Error message if download failed",
        examples=[None, "Network error: connection timed out"],
    )


class ActivateResponse(BaseModel):
    """Response after activating a model.

    Result of switching to a different model.

    Example:
        ```json
        {
            "success": true,
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="True if the model was activated successfully",
    )
    model_id: str = Field(
        ...,
        description="The model ID that was activated",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    error: str | None = Field(
        default=None,
        description="Error message if activation failed",
        examples=[None, "Model not downloaded. Please download first."],
    )


# =============================================================================
# Export Schemas
# =============================================================================


class ExportFormatEnum(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    TXT = "txt"


class ExportDateRange(BaseModel):
    """Date range filter for exports."""

    start: datetime | None = Field(default=None, description="Start date (inclusive)")
    end: datetime | None = Field(default=None, description="End date (inclusive)")


class ExportConversationRequest(BaseModel):
    """Request to export a single conversation."""

    format: ExportFormatEnum = Field(
        default=ExportFormatEnum.JSON,
        description="Export format (json, csv, txt)",
    )
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )
    include_attachments: bool = Field(
        default=False,
        description="Include attachment info in export (CSV only)",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum messages to export",
    )


class ExportSearchRequest(BaseModel):
    """Request to export search results."""

    query: str = Field(..., min_length=1, description="Search query")
    format: ExportFormatEnum = Field(
        default=ExportFormatEnum.JSON,
        description="Export format (json, csv, txt)",
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum results to export",
    )
    sender: str | None = Field(default=None, description="Filter by sender")
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )


class ExportBackupRequest(BaseModel):
    """Request for full conversation backup."""

    conversation_limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum conversations to include",
    )
    messages_per_conversation: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum messages per conversation",
    )
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter for messages",
    )


class ExportResponse(BaseModel):
    """Response containing exported data."""

    success: bool
    format: str
    filename: str
    data: str = Field(..., description="Exported data as string")
    message_count: int
    export_type: str = Field(
        default="conversation",
        description="Type of export: conversation, search, or backup",
    )


# =============================================================================
# Statistics Schemas
# =============================================================================


class TimeRangeEnum(str, Enum):
    """Time range options for statistics calculation."""

    WEEK = "week"
    MONTH = "month"
    THREE_MONTHS = "three_months"
    ALL_TIME = "all_time"


class HourlyActivity(BaseModel):
    """Hourly message activity data point.

    Represents the number of messages sent/received during a specific hour.

    Example:
        ```json
        {
            "hour": 14,
            "count": 45
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hour": 14,
                "count": 45,
            }
        }
    )

    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0-23)",
        examples=[9, 14, 20],
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of messages during this hour",
        examples=[45, 23, 78],
    )


class WordFrequency(BaseModel):
    """Word frequency data for conversation analytics.

    Represents how often a specific word appears in conversations.

    Example:
        ```json
        {
            "word": "hello",
            "count": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "word": "hello",
                "count": 50,
            }
        }
    )

    word: str = Field(
        ...,
        description="The word",
        examples=["hello", "thanks", "meeting"],
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of occurrences",
        examples=[50, 35, 28],
    )


class ConversationStatsResponse(BaseModel):
    """Comprehensive conversation statistics response.

    Contains analytics and insights about messaging patterns in a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "time_range": "month",
            "total_messages": 500,
            "sent_count": 250,
            "received_count": 250,
            "avg_response_time_minutes": 15.5,
            "hourly_activity": [{"hour": 9, "count": 45}],
            "daily_activity": {"Monday": 80, "Tuesday": 70},
            "message_length_distribution": {
                "short": 200,
                "medium": 200,
                "long": 80,
                "very_long": 20
            },
            "top_words": [{"word": "hello", "count": 50}],
            "emoji_usage": {"heart": 25, "smile": 20},
            "attachment_breakdown": {"images": 30, "videos": 5}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "time_range": "month",
                "total_messages": 500,
                "sent_count": 250,
                "received_count": 250,
                "avg_response_time_minutes": 15.5,
                "hourly_activity": [
                    {"hour": 9, "count": 45},
                    {"hour": 10, "count": 52},
                ],
                "daily_activity": {
                    "Monday": 80,
                    "Tuesday": 70,
                    "Wednesday": 65,
                },
                "message_length_distribution": {
                    "short": 200,
                    "medium": 200,
                    "long": 80,
                    "very_long": 20,
                },
                "top_words": [
                    {"word": "hello", "count": 50},
                    {"word": "thanks", "count": 35},
                ],
                "emoji_usage": {"❤️": 25, "😊": 20},
                "attachment_breakdown": {"images": 30, "videos": 5},
                "first_message_date": "2024-01-01T10:00:00Z",
                "last_message_date": "2024-01-15T18:30:00Z",
                "participants": ["+15551234567"],
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation identifier",
        examples=["chat123456789"],
    )
    time_range: TimeRangeEnum = Field(
        ...,
        description="Time range used for statistics",
    )
    total_messages: int = Field(
        ...,
        ge=0,
        description="Total number of messages analyzed",
        examples=[500, 1000],
    )
    sent_count: int = Field(
        ...,
        ge=0,
        description="Number of messages sent by user",
        examples=[250, 480],
    )
    received_count: int = Field(
        ...,
        ge=0,
        description="Number of messages received",
        examples=[250, 520],
    )
    avg_response_time_minutes: float | None = Field(
        default=None,
        description="Average response time in minutes (within 24h window)",
        examples=[15.5, 8.2],
    )
    hourly_activity: list[HourlyActivity] = Field(
        default_factory=list,
        description="Message count by hour of day (0-23)",
    )
    daily_activity: dict[str, int] = Field(
        default_factory=dict,
        description="Message count by day of week",
        examples=[{"Monday": 80, "Tuesday": 70}],
    )
    message_length_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of message lengths (short/medium/long/very_long)",
        examples=[{"short": 200, "medium": 200, "long": 80, "very_long": 20}],
    )
    top_words: list[WordFrequency] = Field(
        default_factory=list,
        description="Most frequently used words",
    )
    emoji_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Most frequently used emojis with counts",
        examples=[{"❤️": 25, "😊": 20}],
    )
    attachment_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Attachment counts by type (images/videos/audio/documents/other)",
        examples=[{"images": 30, "videos": 5}],
    )
    first_message_date: datetime | None = Field(
        default=None,
        description="Date of earliest message in the analyzed range",
    )
    last_message_date: datetime | None = Field(
        default=None,
        description="Date of most recent message in the analyzed range",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="List of conversation participants",
        examples=[["+15551234567", "+15559876543"]],
    )


# =============================================================================
# Digest Schemas
# =============================================================================


class DigestPeriodEnum(str, Enum):
    """Time period for digest generation."""

    DAILY = "daily"
    WEEKLY = "weekly"


class DigestFormatEnum(str, Enum):
    """Supported digest export formats."""

    MARKDOWN = "markdown"
    HTML = "html"


class UnansweredConversationResponse(BaseModel):
    """A conversation with unanswered messages.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "display_name": "John Doe",
            "participants": ["+15551234567"],
            "unanswered_count": 3,
            "last_message_date": "2024-01-15T10:30:00Z",
            "last_message_preview": "Hey, are you free tomorrow?",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "display_name": "John Doe",
                "participants": ["+15551234567"],
                "unanswered_count": 3,
                "last_message_date": "2024-01-15T10:30:00Z",
                "last_message_preview": "Hey, are you free tomorrow?",
                "is_group": False,
            }
        }
    )

    chat_id: str = Field(..., description="Conversation identifier")
    display_name: str = Field(..., description="Display name for the conversation")
    participants: list[str] = Field(..., description="List of participants")
    unanswered_count: int = Field(..., ge=0, description="Number of unanswered messages")
    last_message_date: datetime | None = Field(
        default=None, description="Date of last unanswered message"
    )
    last_message_preview: str | None = Field(
        default=None, description="Preview of last unanswered message"
    )
    is_group: bool = Field(..., description="Whether this is a group conversation")


class GroupHighlightResponse(BaseModel):
    """Highlight from an active group chat.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "display_name": "Family Group",
            "participants": ["+15551234567", "+15559876543"],
            "message_count": 45,
            "active_participants": ["John", "Jane"],
            "top_topics": ["meeting", "plans"],
            "last_activity": "2024-01-15T18:30:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "display_name": "Family Group",
                "participants": ["+15551234567", "+15559876543"],
                "message_count": 45,
                "active_participants": ["John", "Jane"],
                "top_topics": ["meeting", "plans"],
                "last_activity": "2024-01-15T18:30:00Z",
            }
        }
    )

    chat_id: str = Field(..., description="Conversation identifier")
    display_name: str = Field(..., description="Group display name")
    participants: list[str] = Field(..., description="List of participants")
    message_count: int = Field(..., ge=0, description="Message count in period")
    active_participants: list[str] = Field(..., description="Most active participants")
    top_topics: list[str] = Field(..., description="Detected topics")
    last_activity: datetime | None = Field(default=None, description="Last activity timestamp")


class ActionItemResponse(BaseModel):
    """Detected action item from messages.

    Example:
        ```json
        {
            "text": "Can you send me the report?",
            "chat_id": "chat123456789",
            "conversation_name": "John Doe",
            "sender": "John Doe",
            "date": "2024-01-15T10:30:00Z",
            "message_id": 12345,
            "item_type": "task"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Can you send me the report?",
                "chat_id": "chat123456789",
                "conversation_name": "John Doe",
                "sender": "John Doe",
                "date": "2024-01-15T10:30:00Z",
                "message_id": 12345,
                "item_type": "task",
            }
        }
    )

    text: str = Field(..., description="Action item text")
    chat_id: str = Field(..., description="Conversation identifier")
    conversation_name: str = Field(..., description="Conversation display name")
    sender: str = Field(..., description="Sender of the message")
    date: datetime = Field(..., description="Message date")
    message_id: int = Field(..., description="Message identifier")
    item_type: str = Field(
        ...,
        description="Type of action item (task, question, event, reminder)",
        examples=["task", "question", "event", "reminder"],
    )


class MessageStatsResponse(BaseModel):
    """Message volume statistics.

    Example:
        ```json
        {
            "total_sent": 150,
            "total_received": 200,
            "total_messages": 350,
            "active_conversations": 12,
            "most_active_conversation": "Family Group",
            "most_active_count": 45,
            "avg_messages_per_day": 50.0,
            "busiest_hour": 14,
            "hourly_distribution": {"9": 20, "14": 35}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_sent": 150,
                "total_received": 200,
                "total_messages": 350,
                "active_conversations": 12,
                "most_active_conversation": "Family Group",
                "most_active_count": 45,
                "avg_messages_per_day": 50.0,
                "busiest_hour": 14,
                "hourly_distribution": {"9": 20, "14": 35},
            }
        }
    )

    total_sent: int = Field(..., ge=0, description="Total messages sent")
    total_received: int = Field(..., ge=0, description="Total messages received")
    total_messages: int = Field(..., ge=0, description="Total messages")
    active_conversations: int = Field(..., ge=0, description="Active conversations count")
    most_active_conversation: str | None = Field(
        default=None, description="Name of most active conversation"
    )
    most_active_count: int = Field(default=0, ge=0, description="Message count in most active")
    avg_messages_per_day: float = Field(..., ge=0, description="Average messages per day")
    busiest_hour: int | None = Field(
        default=None, ge=0, le=23, description="Hour with most messages (0-23)"
    )
    hourly_distribution: dict[str, int] = Field(
        default_factory=dict, description="Message count by hour"
    )


class DigestResponse(BaseModel):
    """Complete digest response.

    Example:
        ```json
        {
            "period": "daily",
            "generated_at": "2024-01-15T08:00:00Z",
            "start_date": "2024-01-14T08:00:00Z",
            "end_date": "2024-01-15T08:00:00Z",
            "needs_attention": [],
            "highlights": [],
            "action_items": [],
            "stats": {...}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "generated_at": "2024-01-15T08:00:00Z",
                "start_date": "2024-01-14T08:00:00Z",
                "end_date": "2024-01-15T08:00:00Z",
                "needs_attention": [],
                "highlights": [],
                "action_items": [],
                "stats": {
                    "total_sent": 50,
                    "total_received": 75,
                    "total_messages": 125,
                    "active_conversations": 5,
                    "most_active_conversation": "Family Group",
                    "most_active_count": 30,
                    "avg_messages_per_day": 125.0,
                    "busiest_hour": 14,
                    "hourly_distribution": {},
                },
            }
        }
    )

    period: DigestPeriodEnum = Field(..., description="Digest period")
    generated_at: datetime = Field(..., description="When the digest was generated")
    start_date: datetime = Field(..., description="Start of digest period")
    end_date: datetime = Field(..., description="End of digest period")
    needs_attention: list[UnansweredConversationResponse] = Field(
        default_factory=list, description="Conversations needing attention"
    )
    highlights: list[GroupHighlightResponse] = Field(
        default_factory=list, description="Group chat highlights"
    )
    action_items: list[ActionItemResponse] = Field(
        default_factory=list, description="Detected action items"
    )
    stats: MessageStatsResponse = Field(..., description="Message statistics")


class DigestGenerateRequest(BaseModel):
    """Request to generate a digest.

    Example:
        ```json
        {
            "period": "daily",
            "end_date": "2024-01-15T08:00:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "end_date": "2024-01-15T08:00:00Z",
            }
        }
    )

    period: DigestPeriodEnum = Field(
        default=DigestPeriodEnum.DAILY, description="Digest period (daily or weekly)"
    )
    end_date: datetime | None = Field(
        default=None, description="End date for digest (defaults to now)"
    )


class DigestExportRequest(BaseModel):
    """Request to export a digest.

    Example:
        ```json
        {
            "period": "daily",
            "format": "markdown",
            "end_date": "2024-01-15T08:00:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "format": "markdown",
                "end_date": "2024-01-15T08:00:00Z",
            }
        }
    )

    period: DigestPeriodEnum = Field(default=DigestPeriodEnum.DAILY, description="Digest period")
    format: DigestFormatEnum = Field(default=DigestFormatEnum.MARKDOWN, description="Export format")
    end_date: datetime | None = Field(
        default=None, description="End date for digest (defaults to now)"
    )


class DigestExportResponse(BaseModel):
    """Response containing exported digest.

    Example:
        ```json
        {
            "success": true,
            "format": "markdown",
            "filename": "jarvis_digest_daily_20240115.md",
            "data": "# JARVIS Daily Digest..."
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "format": "markdown",
                "filename": "jarvis_digest_daily_20240115.md",
                "data": "# JARVIS Daily Digest...",
            }
        }
    )

    success: bool = Field(..., description="Whether export succeeded")
    format: str = Field(..., description="Export format used")
    filename: str = Field(..., description="Suggested filename")
    data: str = Field(..., description="Exported content")


class DigestPreferencesResponse(BaseModel):
    """Digest preferences response.

    Example:
        ```json
        {
            "enabled": true,
            "schedule": "daily",
            "preferred_time": "08:00",
            "include_action_items": true,
            "include_stats": true,
            "max_conversations": 50,
            "export_format": "markdown"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "schedule": "daily",
                "preferred_time": "08:00",
                "include_action_items": True,
                "include_stats": True,
                "max_conversations": 50,
                "export_format": "markdown",
            }
        }
    )

    enabled: bool = Field(..., description="Whether digest generation is enabled")
    schedule: str = Field(..., description="Digest schedule (daily or weekly)")
    preferred_time: str = Field(..., description="Preferred generation time (HH:MM)")
    include_action_items: bool = Field(..., description="Include action items")
    include_stats: bool = Field(..., description="Include statistics")
    max_conversations: int = Field(..., description="Max conversations to analyze")
    export_format: str = Field(..., description="Default export format")


class DigestPreferencesUpdateRequest(BaseModel):
    """Request to update digest preferences.

    Example:
        ```json
        {
            "enabled": true,
            "schedule": "weekly"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "schedule": "weekly",
            }
        }
    )

    enabled: bool | None = Field(default=None, description="Enable/disable digest")
    schedule: str | None = Field(default=None, description="Schedule (daily/weekly)")
    preferred_time: str | None = Field(default=None, description="Preferred time (HH:MM)")
    include_action_items: bool | None = Field(default=None, description="Include action items")
    include_stats: bool | None = Field(default=None, description="Include statistics")
    max_conversations: int | None = Field(
        default=None, ge=10, le=200, description="Max conversations"
    )
    export_format: str | None = Field(default=None, description="Export format")
