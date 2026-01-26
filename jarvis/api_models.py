"""Pydantic models for the JARVIS API.

Defines request and response schemas for all API endpoints.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# --- Enums ---


class MemoryModeEnum(str, Enum):
    """Memory operating mode."""

    FULL = "full"
    LITE = "lite"
    MINIMAL = "minimal"


class FeatureStateEnum(str, Enum):
    """Health state of a feature."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


# --- Chat Models ---


class ChatRequest(BaseModel):
    """Request for chat generation."""

    message: str = Field(..., min_length=1, description="User message to send to the AI")
    max_tokens: int = Field(default=200, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    context_documents: list[str] = Field(
        default_factory=list, description="RAG context documents to inject"
    )
    stream: bool = Field(default=False, description="Enable streaming via SSE")


class ChatResponse(BaseModel):
    """Response from chat generation."""

    text: str = Field(..., description="Generated response text")
    tokens_used: int = Field(..., description="Number of tokens used")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    model_name: str = Field(..., description="Model used for generation")
    used_template: bool = Field(..., description="Whether a template was matched")
    template_name: str | None = Field(None, description="Name of matched template if any")
    finish_reason: str = Field(..., description="Reason generation finished")


class ChatStreamEvent(BaseModel):
    """Server-Sent Event for streaming chat responses."""

    event: str = Field(..., description="Event type: 'token', 'done', or 'error'")
    data: str = Field(..., description="Event data (token text or final response JSON)")


# --- Search Models ---


class SearchRequest(BaseModel):
    """Request for message search."""

    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results to return")
    sender: str | None = Field(None, description="Filter by sender (phone/email or 'me')")
    after: datetime | None = Field(None, description="Filter for messages after this date")
    before: datetime | None = Field(None, description="Filter for messages before this date")
    chat_id: str | None = Field(None, description="Filter by conversation ID")
    has_attachments: bool | None = Field(None, description="Filter by attachment presence")


class AttachmentResponse(BaseModel):
    """Attachment metadata in API response."""

    filename: str
    file_path: str | None
    mime_type: str | None
    file_size: int | None


class ReactionResponse(BaseModel):
    """Reaction in API response."""

    type: str
    sender: str
    sender_name: str | None
    date: datetime


class MessageResponse(BaseModel):
    """Message in API response."""

    id: int
    chat_id: str
    sender: str
    sender_name: str | None
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[AttachmentResponse] = Field(default_factory=list)
    reply_to_id: int | None = None
    reactions: list[ReactionResponse] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from message search."""

    messages: list[MessageResponse]
    total: int = Field(..., description="Total number of results returned")
    query: str = Field(..., description="The search query that was executed")


# --- Conversations Models ---


class ConversationResponse(BaseModel):
    """Conversation summary in API response."""

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool


class ConversationsListResponse(BaseModel):
    """Response listing conversations."""

    conversations: list[ConversationResponse]
    total: int = Field(..., description="Total number of conversations returned")


class MessagesListResponse(BaseModel):
    """Response listing messages in a conversation."""

    messages: list[MessageResponse]
    chat_id: str
    total: int = Field(..., description="Total number of messages returned")


# --- Health Models ---


class FeatureHealthResponse(BaseModel):
    """Health status of a single feature."""

    name: str
    state: FeatureStateEnum
    details: str | None = None


class MemoryStatusResponse(BaseModel):
    """Memory status in health response."""

    available_mb: float
    used_mb: float
    current_mode: MemoryModeEnum
    pressure_level: str
    model_loaded: bool


class ModelStatusResponse(BaseModel):
    """Model status in health response."""

    loaded: bool
    memory_usage_mb: float | None = None
    model_name: str | None = None


class HealthResponse(BaseModel):
    """Full system health response."""

    status: str = Field(..., description="Overall status: 'healthy', 'degraded', or 'unhealthy'")
    memory: MemoryStatusResponse
    features: list[FeatureHealthResponse]
    model: ModelStatusResponse
    version: str


# --- Error Models ---


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: str | None = Field(None, description="Additional error details")
