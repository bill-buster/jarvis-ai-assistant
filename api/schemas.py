    calendar_id: str = Field(
        ...,
        description="Target calendar ID",
    )
    detected_event: DetectedEventResponse = Field(
        ...,
        description="Detected event to add to calendar",
    )


class CreateEventResponse(BaseModel):
    """Response after creating a calendar event.

    Example:
        ```json
        {
            "success": true,
            "event_id": "event-789",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "event_id": "event-789",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the event was created successfully",
    )
    event_id: str | None = Field(
        default=None,
        description="ID of created event (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
=======
    imported: int = Field(
        ...,
        ge=0,
        description="Number of templates successfully imported",
    )
    skipped: int = Field(
        ...,
        ge=0,
        description="Number of templates skipped",
    )
    errors: int = Field(
        ...,
        ge=0,
        description="Number of templates that failed to import",
    )
    total_templates: int = Field(
        ...,
        ge=0,
        description="Total templates after import",
>>>>>>> origin/claude/custom-template-builder-VPeOl
    )


# =============================================================================
# Threading Schemas
# =============================================================================


class ThreadResponse(BaseModel):
    """A conversation thread grouping related messages.

    Threads are auto-detected based on time gaps, topic shifts, and explicit
    reply references. They help organize long conversations into logical
    segments.

    Example:
        ```json
        {
            "thread_id": "a1b2c3d4e5f67890",
            "messages": [12345, 12346, 12347],
            "topic_label": "Dinner Plans",
            "start_time": "2024-01-15T18:00:00Z",
            "end_time": "2024-01-15T18:30:00Z",
            "participant_count": 2,
            "message_count": 3
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": "a1b2c3d4e5f67890",
                "messages": [12345, 12346, 12347],
                "topic_label": "Dinner Plans",
                "start_time": "2024-01-15T18:00:00Z",
                "end_time": "2024-01-15T18:30:00Z",
                "participant_count": 2,
                "message_count": 3,
            }
        }
    )

    thread_id: str = Field(
        ...,
        description="Unique thread identifier",
        examples=["a1b2c3d4e5f67890"],
    )
    messages: list[int] = Field(
        default_factory=list,
        description="List of message IDs in this thread",
        examples=[[12345, 12346, 12347]],
    )
    topic_label: str = Field(
        default="",
        description="Auto-generated topic label for the thread",
        examples=["Dinner Plans", "Work Discussion", "Weekend Plans"],
    )
    start_time: datetime | None = Field(
        default=None,
        description="Timestamp of first message in thread",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Timestamp of last message in thread",
    )
    participant_count: int = Field(
        default=0,
        description="Number of unique participants in thread",
        examples=[2, 3],
        ge=0,
    )
    message_count: int = Field(
        default=0,
        description="Number of messages in thread",
        examples=[3, 10, 25],
        ge=0,
    )


class ThreadedMessageResponse(MessageResponse):
    """Message with thread information attached.

    Extends MessageResponse with thread-specific fields to enable
    threaded conversation views.

    Example:
        ```json
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "text": "Are you free for dinner?",
            "date": "2024-01-15T18:00:00Z",
            "is_from_me": false,
            "thread_id": "a1b2c3d4e5f67890",
            "thread_position": 0,
            "is_thread_start": true
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
                "text": "Are you free for dinner?",
                "date": "2024-01-15T18:00:00Z",
                "is_from_me": False,
                "attachments": [],
                "reply_to_id": None,
                "reactions": [],
                "is_system_message": False,
                "thread_id": "a1b2c3d4e5f67890",
                "thread_position": 0,
                "is_thread_start": True,
            }
        },
    )

    thread_id: str = Field(
        ...,
        description="ID of the thread this message belongs to",
        examples=["a1b2c3d4e5f67890"],
    )
    thread_position: int = Field(
        default=0,
        description="Position within the thread (0-indexed)",
        examples=[0, 1, 2],
        ge=0,
    )
    is_thread_start: bool = Field(
        default=False,
        description="True if this message starts a new thread",
    )


class ThreadedViewResponse(BaseModel):
    """Response containing threaded view of a conversation.

    Provides both the list of threads and the messages with thread info,
    enabling the frontend to display a threaded conversation view.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "threads": [...],
            "messages": [...],
            "total_threads": 5,
            "total_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "threads": [
                    {
                        "thread_id": "a1b2c3d4e5f67890",
                        "messages": [12345, 12346],
                        "topic_label": "Dinner Plans",
                        "message_count": 2,
                    }
                ],
                "messages": [
                    {
                        "id": 12345,
                        "text": "Are you free for dinner?",
                        "thread_id": "a1b2c3d4e5f67890",
                        "thread_position": 0,
                        "is_thread_start": True,
                    }
                ],
                "total_threads": 5,
                "total_messages": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation identifier",
        examples=["chat123456789"],
    )
    threads: list[ThreadResponse] = Field(
        default_factory=list,
        description="List of detected threads",
    )
    messages: list[ThreadedMessageResponse] = Field(
        default_factory=list,
        description="Messages with thread information",
    )
    total_threads: int = Field(
        default=0,
        description="Total number of threads detected",
        examples=[5, 10],
        ge=0,
    )
    total_messages: int = Field(
        default=0,
        description="Total number of messages analyzed",
        examples=[50, 100],
        ge=0,
    )


class ThreadingConfigRequest(BaseModel):
    """Configuration options for thread analysis.

    Allows customizing how threads are detected.

    Example:
        ```json
        {
            "time_gap_threshold_minutes": 30,
            "semantic_similarity_threshold": 0.4,
            "use_semantic_analysis": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_gap_threshold_minutes": 30,
                "semantic_similarity_threshold": 0.4,
                "use_semantic_analysis": True,
            }
        }
    )

    time_gap_threshold_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minutes of silence to start a new thread (5-1440)",
        examples=[30, 60, 120],
    )
    semantic_similarity_threshold: float = Field(
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Minimum similarity to group messages (0.1-0.9)",
        examples=[0.3, 0.4, 0.5],
    )
    use_semantic_analysis: bool = Field(
        default=True,
        description="Whether to use ML-based topic detection",
    )


# =============================================================================
# Attachment Manager Schemas
# =============================================================================


class AttachmentTypeEnum(str, Enum):
    """Attachment type categories for filtering."""

    IMAGES = "images"
    VIDEOS = "videos"
    AUDIO = "audio"
    DOCUMENTS = "documents"
    OTHER = "other"
    ALL = "all"


class ExtendedAttachmentResponse(BaseModel):
    """Extended attachment metadata with additional media information.

    Includes dimensions for images/videos and duration for audio/video.

    Example:
        ```json
        {
            "filename": "IMG_1234.jpg",
            "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
            "mime_type": "image/jpeg",
            "file_size": 245760,
            "width": 1920,
            "height": 1080,
            "duration_seconds": null,
            "created_date": "2024-01-15T10:30:00Z",
            "is_sticker": false,
            "uti": "public.jpeg"
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
                "width": 1920,
                "height": 1080,
                "duration_seconds": None,
                "created_date": "2024-01-15T10:30:00Z",
                "is_sticker": False,
                "uti": "public.jpeg",
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
    width: int | None = Field(
        default=None,
        description="Image/video width in pixels",
        examples=[1920, 1080],
        ge=0,
    )
    height: int | None = Field(
        default=None,
        description="Image/video height in pixels",
        examples=[1080, 1920],
        ge=0,
    )
    duration_seconds: float | None = Field(
        default=None,
        description="Audio/video duration in seconds",
        examples=[30.5, 120.0],
        ge=0,
    )
    created_date: datetime | None = Field(
        default=None,
        description="When the attachment was created/received",
    )
    is_sticker: bool = Field(
        default=False,
        description="True if the attachment is a sticker",
    )
    uti: str | None = Field(
        default=None,
        description="Uniform Type Identifier (e.g., 'public.jpeg')",
        examples=["public.jpeg", "public.mpeg-4", "com.adobe.pdf"],
    )


class AttachmentWithContextResponse(BaseModel):
    """Attachment with message context information.

    Includes the attachment details along with information about
    the message and conversation it belongs to.

    Example:
        ```json
        {
            "attachment": { ... },
            "message_id": 12345,
            "message_date": "2024-01-15T10:30:00Z",
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "is_from_me": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "attachment": {
                    "filename": "IMG_1234.jpg",
                    "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
                    "mime_type": "image/jpeg",
                    "file_size": 245760,
                    "width": 1920,
                    "height": 1080,
                },
                "message_id": 12345,
                "message_date": "2024-01-15T10:30:00Z",
                "chat_id": "chat123456789",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "is_from_me": False,
            }
        }
    )

    attachment: ExtendedAttachmentResponse = Field(
        ...,
        description="The attachment metadata",
    )
    message_id: int = Field(
        ...,
        description="ID of the message containing this attachment",
        examples=[12345],
    )
    message_date: datetime = Field(
        ...,
        description="When the message was sent/received",
    )
    chat_id: str = Field(
        ...,
        description="ID of the conversation",
        examples=["chat123456789"],
    )
    sender: str = Field(
        ...,
        description="Phone number or email of the sender",
        examples=["+15551234567"],
    )
    sender_name: str | None = Field(
        default=None,
        description="Display name from contacts",
        examples=["John Doe"],
    )
    is_from_me: bool = Field(
        ...,
        description="True if the attachment was sent by the current user",
    )


class AttachmentStatsResponse(BaseModel):
    """Attachment statistics for a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "total_count": 150,
            "total_size_bytes": 524288000,
            "total_size_formatted": "500.0 MB",
            "by_type": {"images": 100, "videos": 30, "documents": 20},
            "size_by_type": {"images": 314572800, "videos": 157286400}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "total_count": 150,
                "total_size_bytes": 524288000,
                "total_size_formatted": "500.0 MB",
                "by_type": {"images": 100, "videos": 30, "documents": 20},
                "size_by_type": {"images": 314572800, "videos": 157286400},
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID",
        examples=["chat123456789"],
    )
    total_count: int = Field(
        ...,
        description="Total number of attachments",
        examples=[150],
        ge=0,
    )
    total_size_bytes: int = Field(
        ...,
        description="Total size of all attachments in bytes",
        examples=[524288000],
        ge=0,
    )
    total_size_formatted: str = Field(
        ...,
        description="Human-readable size (e.g., '500.0 MB')",
        examples=["500.0 MB", "1.2 GB"],
    )
    by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count by type (images, videos, audio, documents, other)",
        examples=[{"images": 100, "videos": 30, "documents": 20}],
    )
    size_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Size in bytes by type",
        examples=[{"images": 314572800, "videos": 157286400}],
    )


class StorageByConversationResponse(BaseModel):
    """Storage usage for a single conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "display_name": "John Doe",
            "attachment_count": 150,
            "total_size_bytes": 524288000,
            "total_size_formatted": "500.0 MB"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "display_name": "John Doe",
                "attachment_count": 150,
                "total_size_bytes": 524288000,
                "total_size_formatted": "500.0 MB",
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID",
        examples=["chat123456789"],
    )
    display_name: str | None = Field(
        default=None,
        description="Display name for the conversation",
        examples=["John Doe", "Family Group"],
    )
    attachment_count: int = Field(
        ...,
        description="Number of attachments in this conversation",
        examples=[150],
        ge=0,
    )
    total_size_bytes: int = Field(
        ...,
        description="Total size of attachments in bytes",
        examples=[524288000],
        ge=0,
    )
    total_size_formatted: str = Field(
        ...,
        description="Human-readable size",
        examples=["500.0 MB", "1.2 GB"],
    )


class StorageSummaryResponse(BaseModel):
    """Summary of attachment storage across all conversations.

    Example:
        ```json
        {
            "total_attachments": 1500,
            "total_size_bytes": 5242880000,
            "total_size_formatted": "5.0 GB",
            "by_conversation": [...]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_attachments": 1500,
                "total_size_bytes": 5242880000,
                "total_size_formatted": "5.0 GB",
                "by_conversation": [],
            }
        }
    )

    total_attachments: int = Field(
        ...,
        description="Total number of attachments across all conversations",
        examples=[1500],
        ge=0,
    )
    total_size_bytes: int = Field(
        ...,
        description="Total size of all attachments in bytes",
        examples=[5242880000],
        ge=0,
    )
    total_size_formatted: str = Field(
        ...,
        description="Human-readable total size",
        examples=["5.0 GB"],
    )
    by_conversation: list[StorageByConversationResponse] = Field(
        default_factory=list,
        description="Storage breakdown by conversation, sorted by size descending",
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
    """Highlight from an active group chat."""

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
    """Detected action item from messages."""

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
    """Message volume statistics."""

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
    """Complete digest response."""

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
    """Request to generate a digest."""

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
    """Request to export a digest."""

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
    """Response containing exported digest."""

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
    """Digest preferences response."""

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
    schedule: str = Field(..., description="Digest schedule (daily/weekly)")
    preferred_time: str = Field(..., description="Preferred time (HH:MM)")
    include_action_items: bool = Field(..., description="Include action items")
    include_stats: bool = Field(..., description="Include stats")
    max_conversations: int = Field(..., description="Max conversations to analyze")
    export_format: str = Field(..., description="Default export format")


class DigestPreferencesUpdate(BaseModel):
    """Request to update digest preferences."""

    enabled: bool | None = Field(default=None)
    schedule: str | None = Field(default=None)
    preferred_time: str | None = Field(default=None)
    include_action_items: bool | None = Field(default=None)
    include_stats: bool | None = Field(default=None)
    max_conversations: int | None = Field(default=None)
    export_format: str | None = Field(default=None)


# =============================================================================
# Relationship Schemas
# =============================================================================


class ToneProfileResponse(BaseModel):
    """Communication tone characteristics for a relationship."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "formality_score": 0.3,
                "emoji_frequency": 1.5,
                "exclamation_frequency": 0.8,
                "question_frequency": 0.2,
                "avg_message_length": 45.5,
                "uses_caps": False,
            }
        }
    )

    formality_score: float = Field(..., ge=0.0, le=1.0)
    emoji_frequency: float = Field(..., ge=0.0)
    exclamation_frequency: float = Field(..., ge=0.0)
    question_frequency: float = Field(..., ge=0.0)
    avg_message_length: float = Field(..., ge=0.0)
    uses_caps: bool = Field(...)


class TopicDistributionResponse(BaseModel):
    """Distribution of conversation topics."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topics": {
                    "scheduling": 0.35,
                    "food": 0.25,
                    "work": 0.2
                },
                "top_topics": ["scheduling", "food", "work"]
            }
        }
    )

    topics: dict[str, float] = Field(default_factory=dict)
    top_topics: list[str] = Field(default_factory=list)


class RelationshipProfileResponse(BaseModel):
    """Complete relationship profile for a contact."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contact_id": "a1b2c3d4e5f6g7h8",
                "contact_name": "John Doe",
                "tone_profile": {
                    "formality_score": 0.3,
                    "emoji_frequency": 1.5,
                    "exclamation_frequency": 0.8,
                    "question_frequency": 0.2,
                    "avg_message_length": 45.5,
                    "uses_caps": False,
                },
                "topic_distribution": {
                    "topics": {"scheduling": 0.35, "food": 0.25},
                    "top_topics": ["scheduling", "food"],
                },
                "response_patterns": {
                    "avg_response_time_minutes": 15.5,
                    "typical_response_length": "medium",
                    "greeting_style": ["hey", "hi"],
                    "signoff_style": ["thanks"],
                    "common_phrases": ["sounds good"],
                },
                "message_count": 250,
                "last_updated": "2024-01-15T10:30:00",
                "version": "1.0.0",
            }
        }
    )

    contact_id: str = Field(..., description="Hashed contact identifier")
    contact_name: str | None = Field(default=None, description="Display name")
    tone_profile: ToneProfileResponse = Field(..., description="Tone info")
    topic_distribution: TopicDistributionResponse = Field(..., description="Topic info")
    response_patterns: ResponsePatternsResponse = Field(..., description="Response info")
    message_count: int = Field(..., ge=0)
    last_updated: str = Field(...)
    version: str = Field(...)


class StyleGuideResponse(BaseModel):
    """Natural language style guide for a relationship."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contact_id": "a1b2c3d4e5f6g7h8",
                "contact_name": "John Doe",
                "style_guide": "Keep it casual, use emojis, keep messages brief.",
                "voice_guidance": {
                    "formality": "casual",
                    "use_emojis": True,
                    "emoji_level": "high",
                    "message_length": "short",
                    "use_exclamations": True,
                    "common_greetings": ["hey", "hi"],
                    "common_signoffs": ["thanks", "bye"],
                    "preferred_phrases": ["sounds good"],
                    "top_topics": ["scheduling", "food"],
                },
            }
        }
    )

    contact_id: str = Field(...)
    contact_name: str | None = Field(default=None)
    style_guide: str = Field(...)
    voice_guidance: dict[str, Any] = Field(...)


class RefreshProfileRequest(BaseModel):
    """Request to refresh a relationship profile."""

    message_limit: int = Field(default=500, ge=50, le=2000)
    force_refresh: bool = Field(default=False)


class RefreshProfileResponse(BaseModel):
    """Response after refreshing a relationship profile."""

    success: bool = Field(...)
    profile: RelationshipProfileResponse | None = Field(default=None)
    messages_analyzed: int = Field(...)
    previous_message_count: int = Field(...)
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
=======
    success: bool = Field(
        ...,
        description="Whether the refresh was successful",
    )
    profile: RelationshipProfileResponse | None = Field(
        default=None,
        description="The refreshed profile (if successful)",
    )
    messages_analyzed: int = Field(
        ...,
        ge=0,
        description="Number of messages analyzed",
        examples=[500, 250],
    )
    previous_message_count: int | None = Field(
        default=None,
        description="Previous profile's message count (if existed)",
        examples=[250, None],
    )
    error: str | None = Field(
        default=None,
        description="Error message if refresh failed",
        examples=[None, "Insufficient messages for profile"],
    )
>>>>>>> origin/claude/relationship-learning-system-A02qZ
