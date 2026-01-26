"""iMessage integration interfaces.

Workstream 10 implements against these contracts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class Attachment:
    """iMessage attachment metadata."""

    filename: str
    file_path: str | None  # Full path to attachment file
    mime_type: str | None
    file_size: int | None  # Size in bytes
    # Extended metadata
    width: int | None = None  # Image/video width in pixels
    height: int | None = None  # Image/video height in pixels
    duration_seconds: float | None = None  # Audio/video duration in seconds
    created_date: datetime | None = None  # When attachment was created/received
    is_sticker: bool = False  # True if attachment is a sticker
    uti: str | None = None  # Uniform Type Identifier (e.g., "public.jpeg")


@dataclass
class AttachmentSummary:
    """Summary of attachments for a conversation."""

    total_count: int
    total_size_bytes: int
    by_type: dict[str, int]  # Count by type (images, videos, audio, documents, other)
    size_by_type: dict[str, int]  # Size in bytes by type


@dataclass
class Reaction:
    """iMessage tapback reaction."""

    # Tapback types: love, like, dislike, laugh, emphasize, question
    # Also supports "removed_" prefix for removed reactions
    type: str
    sender: str  # Phone number or email
    sender_name: str | None  # Resolved from contacts if available
    date: datetime


@dataclass
class Message:
    """Normalized iMessage representation."""

    id: int
    chat_id: str
    sender: str  # Phone number or email
    sender_name: str | None  # Resolved from contacts if available
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[Attachment] = field(default_factory=list)
    reply_to_id: int | None = None
    reactions: list[Reaction] = field(default_factory=list)
    # Read receipt info (only for messages you sent)
    date_delivered: datetime | None = None
    date_read: datetime | None = None
    # System messages (group events like "John left the group")
    is_system_message: bool = False


@dataclass
class Conversation:
    """iMessage conversation summary."""

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool
    last_message_text: str | None = None  # Preview of the last message


class iMessageReader(Protocol):  # noqa: N801 - iMessage is a proper noun
    """Interface for iMessage integration (Workstream 10)."""

    def check_access(self) -> bool:
        """Check if we have permission to read chat.db."""
        ...

    def get_conversations(
        self, limit: int = 50, since: datetime | None = None
    ) -> list[Conversation]:
        """Get recent conversations."""
        ...

    def get_messages(
        self, chat_id: str, limit: int = 100, before: datetime | None = None
    ) -> list[Message]:
        """Get messages from a conversation."""
        ...

    def search(
        self,
        query: str,
        limit: int = 50,
        sender: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        chat_id: str | None = None,
        has_attachments: bool | None = None,
    ) -> list[Message]:
        """Full-text search across messages with optional filters.

        Args:
            query: Search query string
            limit: Maximum number of results
            sender: Filter by sender phone number or email
            after: Filter for messages after this datetime
            before: Filter for messages before this datetime
            chat_id: Filter by conversation ID
            has_attachments: Filter for messages with/without attachments
        """
        ...

    def get_conversation_context(
        self, chat_id: str, around_message_id: int, context_messages: int = 5
    ) -> list[Message]:
        """Get messages around a specific message for context."""
        ...
