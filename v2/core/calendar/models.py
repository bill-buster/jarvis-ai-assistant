"""Calendar data models for JARVIS v2.

Dataclasses for calendar events, detected events, and related types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DetectedEvent:
    """Event detected from message text using NLP.

    Represents a potential calendar event extracted from a message,
    with parsed date/time information and confidence score.
    """

    title: str
    start: datetime
    end: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False
    confidence: float = 0.0
    source_text: str = ""  # Original text that was parsed
    message_id: int | None = None  # iMessage ID if from a message


@dataclass
class CalendarEvent:
    """Calendar event from macOS Calendar.

    Represents an event retrieved from or to be added to macOS Calendar.
    """

    id: str
    calendar_id: str
    calendar_name: str
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str | None = None
    notes: str | None = None
    url: str | None = None
    attendees: list[str] = field(default_factory=list)
    status: str = "confirmed"  # confirmed, tentative, cancelled


@dataclass
class Calendar:
    """macOS Calendar summary."""

    id: str
    name: str
    color: str | None = None
    is_editable: bool = True


@dataclass
class CreateEventResult:
    """Result of creating a calendar event."""

    success: bool
    event_id: str | None = None
    error: str | None = None


@dataclass
class CalendarConflict:
    """Detected conflict between a proposed event and existing calendar."""

    proposed_event: DetectedEvent
    conflicting_event: CalendarEvent
    overlap_minutes: int
    suggestion: str  # Human-readable suggestion


class CalendarAccessError(Exception):
    """Raised when calendar access is denied or unavailable."""

    def __init__(self, message: str, requires_permission: bool = False):
        super().__init__(message)
        self.requires_permission = requires_permission
