"""Pydantic schemas for API responses.

Converts dataclasses from contracts/ to Pydantic models for FastAPI serialization.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AttachmentResponse(BaseModel):
    """iMessage attachment metadata."""

    model_config = ConfigDict(from_attributes=True)

    filename: str
    file_path: str | None
    mime_type: str | None
    file_size: int | None


class ReactionResponse(BaseModel):
    """iMessage tapback reaction."""

    model_config = ConfigDict(from_attributes=True)

    type: str
    sender: str
    sender_name: str | None
    date: datetime


class MessageResponse(BaseModel):
    """iMessage response model."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    chat_id: str
    sender: str
    sender_name: str | None
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[AttachmentResponse]
    reply_to_id: int | None
    reactions: list[ReactionResponse]
    date_delivered: datetime | None = None
    date_read: datetime | None = None
    is_system_message: bool = False  # Group events like "John left the group"


class ConversationResponse(BaseModel):
    """Conversation summary response model."""

    model_config = ConfigDict(from_attributes=True)

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool
    last_message_text: str | None = None  # Preview of the last message


class HealthResponse(BaseModel):
    """System health status response."""

    status: str  # "healthy", "degraded", "unhealthy"
    imessage_access: bool
    memory_available_gb: float
    memory_used_gb: float
    memory_mode: str  # "FULL", "LITE", "MINIMAL"
    model_loaded: bool
    permissions_ok: bool
    details: dict[str, str] | None = None
    # JARVIS process-specific memory
    jarvis_rss_mb: float = 0.0  # Resident Set Size (actual RAM used by JARVIS)
    jarvis_vms_mb: float = 0.0  # Virtual Memory Size (JARVIS allocation)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str
    code: str | None = None


class SendMessageRequest(BaseModel):
    """Request to send an iMessage."""

    text: str = Field(..., min_length=1, max_length=10000, description="Message text")
    recipient: str | None = Field(default=None, description="Recipient phone/email (individual)")
    is_group: bool = Field(default=False, description="Whether this is a group chat")


class SendAttachmentRequest(BaseModel):
    """Request to send a file attachment."""

    file_path: str = Field(..., description="Absolute path to file to send")
    recipient: str | None = Field(default=None, description="Recipient phone/email (individual)")
    is_group: bool = Field(default=False, description="Whether this is a group chat")


class SendMessageResponse(BaseModel):
    """Response after sending a message."""

    success: bool
    error: str | None = None
