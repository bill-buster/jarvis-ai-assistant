"""iMessage integration interfaces.

Workstream 10 implements against these contracts.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


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
    attachments: list[str]
    reply_to_id: int | None
    reactions: list[str]  # Tapback reactions


@dataclass
class Conversation:
    """iMessage conversation summary."""

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool


class iMessageReader(Protocol):
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

    def search(self, query: str, limit: int = 50) -> list[Message]:
        """Full-text search across messages."""
        ...

    def get_conversation_context(
        self, chat_id: str, around_message_id: int, context_messages: int = 5
    ) -> list[Message]:
        """Get messages around a specific message for context."""
        ...
