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


class ModelInfo(BaseModel):
    """Current model information."""

    id: str | None = None
    display_name: str
    loaded: bool
    memory_usage_mb: float
    quality_tier: str | None = None


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
    # Model information
    model: ModelInfo | None = None
    recommended_model: str | None = None
    system_ram_gb: float | None = None


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


# Draft API schemas


class DraftSuggestion(BaseModel):
    """A suggested reply for a conversation."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class ContextInfo(BaseModel):
    """Information about the context used to generate a reply."""

    num_messages: int
    participants: list[str]
    last_message: str | None


class DraftReplyRequest(BaseModel):
    """Request for draft reply generation."""

    chat_id: str
    instruction: str | None = None
    num_suggestions: int = Field(default=3, ge=1, le=5)
    context_messages: int = Field(default=20, ge=5, le=50)


class DraftReplyResponse(BaseModel):
    """Response with generated reply suggestions."""

    suggestions: list[DraftSuggestion]
    context_used: ContextInfo


class DraftSummaryRequest(BaseModel):
    """Request for conversation summarization."""

    chat_id: str
    num_messages: int = Field(default=50, ge=10, le=200)


class DateRange(BaseModel):
    """Date range for a conversation summary."""

    start: str
    end: str


class DraftSummaryResponse(BaseModel):
    """Response with conversation summary."""

    summary: str
    key_points: list[str]
    date_range: DateRange


# Settings schemas


class AvailableModelInfo(BaseModel):
    """Information about an available model."""

    model_id: str
    name: str
    size_gb: float
    quality_tier: str  # "basic", "good", "best"
    ram_requirement_gb: float
    is_downloaded: bool
    is_loaded: bool
    is_recommended: bool
    description: str | None = None


class GenerationSettings(BaseModel):
    """Generation parameter settings."""

    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens_reply: int = Field(default=150, ge=50, le=300)
    max_tokens_summary: int = Field(default=500, ge=200, le=1000)


class BehaviorSettings(BaseModel):
    """Behavior preference settings."""

    auto_suggest_replies: bool = True
    suggestion_count: int = Field(default=3, ge=1, le=5)
    context_messages_reply: int = Field(default=20, ge=10, le=50)
    context_messages_summary: int = Field(default=50, ge=20, le=100)


class SystemInfo(BaseModel):
    """Read-only system information."""

    system_ram_gb: float
    current_memory_usage_gb: float
    model_loaded: bool
    model_memory_usage_gb: float
    imessage_access: bool


class SettingsResponse(BaseModel):
    """Complete settings response."""

    model_id: str
    generation: GenerationSettings
    behavior: BehaviorSettings
    system: SystemInfo


class SettingsUpdateRequest(BaseModel):
    """Request to update settings."""

    model_id: str | None = None
    generation: GenerationSettings | None = None
    behavior: BehaviorSettings | None = None


class DownloadStatus(BaseModel):
    """Model download status."""

    model_id: str
    status: str  # "downloading", "completed", "failed"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error: str | None = None


class ActivateResponse(BaseModel):
    """Response after activating a model."""

    success: bool
    model_id: str
    error: str | None = None
