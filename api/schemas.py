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
