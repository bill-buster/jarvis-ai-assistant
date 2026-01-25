"""Gmail integration interfaces.

Workstream 9 implements against these contracts.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Email:
    """Normalized email representation."""

    id: str
    thread_id: str
    subject: str
    sender: str
    sender_name: str | None
    recipients: list[str]
    body_text: str
    body_html: str | None
    date: datetime
    labels: list[str]
    attachments: list[str]
    snippet: str  # Short preview


@dataclass
class EmailSearchResult:
    """Result of email search."""

    emails: list[Email]
    total_count: int
    next_page_token: str | None


class GmailClient(Protocol):
    """Interface for Gmail integration (Workstream 9)."""

    def authenticate(self) -> bool:
        """Authenticate with Gmail API. Returns success."""
        ...

    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        ...

    def search(
        self, query: str, max_results: int = 10, page_token: str | None = None
    ) -> EmailSearchResult:
        """Search emails using Gmail query syntax."""
        ...

    def get_email(self, email_id: str) -> Email:
        """Get full email by ID."""
        ...

    def get_recent(self, days: int = 7, max_results: int = 50) -> list[Email]:
        """Get recent emails."""
        ...

    def get_thread(self, thread_id: str) -> list[Email]:
        """Get all emails in a thread."""
        ...
