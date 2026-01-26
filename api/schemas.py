"""Pydantic schemas for API responses.

Converts dataclasses from contracts/ to Pydantic models for FastAPI serialization.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


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


class ConversationResponse(BaseModel):
    """Conversation summary response model."""

    model_config = ConfigDict(from_attributes=True)

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool


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


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str
    code: str | None = None
