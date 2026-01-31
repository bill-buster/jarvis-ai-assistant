"""Exchange Model - Data structures for candidate exchanges.

A CandidateExchange represents a potential (trigger, response) pair with:
- Multi-message trigger and response spans
- Structured context window of previous messages
- Metadata for validation gating

Usage:
    from jarvis.exchange import CandidateExchange, ContextMessage

    exchange = CandidateExchange(
        trigger_span=[ContextMessage(speaker="them", timestamp=dt, text="hey")],
        response_span=[ContextMessage(speaker="me", timestamp=dt, text="hi!")],
        context_window=[...],
        chat_id="chat123",
        contact_id=1,
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ContextMessage:
    """A single message in the context window.

    Attributes:
        speaker: Who sent this message ("me" or "them").
        timestamp: When the message was sent.
        text: Cleaned/normalized text content.
        flags: Set of flags indicating message properties.
        raw_text: Original text before cleaning (for debugging).
    """

    speaker: Literal["me", "them"]
    timestamp: datetime
    text: str
    flags: set[str] = field(default_factory=set)
    raw_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "flags": list(self.flags),
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ContextMessage":
        """Create from dictionary."""
        return cls(
            speaker=data["speaker"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            text=data["text"],
            flags=set(data.get("flags", [])),
            raw_text=data.get("raw_text"),
        )


@dataclass
class CandidateExchange:
    """A candidate (trigger, response) exchange for validation.

    Represents a potential pair before it passes through the validity gates.
    The trigger_span contains one or more messages from "them" that form
    the trigger, and response_span contains one or more messages from "me"
    that form the response.

    Attributes:
        trigger_span: Their messages (1-N) that form the trigger.
        response_span: My messages (1-M) that form the response.
        context_window: Previous K messages before the trigger.
        chat_id: The conversation ID.
        contact_id: Optional contact ID from JARVIS database.
        trigger_msg_ids: Message IDs of trigger messages (for deduplication).
        response_msg_ids: Message IDs of response messages (for deduplication).
    """

    trigger_span: list[ContextMessage]
    response_span: list[ContextMessage]
    context_window: list[ContextMessage]
    chat_id: str
    contact_id: int | None = None
    trigger_msg_ids: list[int] = field(default_factory=list)
    response_msg_ids: list[int] = field(default_factory=list)

    @property
    def trigger_text(self) -> str:
        """Get cleaned, joined trigger text from all trigger messages."""
        texts = [msg.text for msg in self.trigger_span if msg.text]
        return "\n".join(texts)

    @property
    def response_text(self) -> str:
        """Get cleaned, joined response text from all response messages."""
        texts = [msg.text for msg in self.response_span if msg.text]
        return "\n".join(texts)

    @property
    def trigger_start_time(self) -> datetime:
        """Timestamp of first trigger message."""
        if not self.trigger_span:
            return datetime.min
        return self.trigger_span[0].timestamp

    @property
    def trigger_end_time(self) -> datetime:
        """Timestamp of last trigger message."""
        if not self.trigger_span:
            return datetime.min
        return self.trigger_span[-1].timestamp

    @property
    def response_start_time(self) -> datetime:
        """Timestamp of first response message."""
        if not self.response_span:
            return datetime.min
        return self.response_span[0].timestamp

    @property
    def response_end_time(self) -> datetime:
        """Timestamp of last response message."""
        if not self.response_span:
            return datetime.min
        return self.response_span[-1].timestamp

    @property
    def time_gap_minutes(self) -> float:
        """Time gap between trigger end and response start, in minutes."""
        if not self.trigger_span or not self.response_span:
            return 0.0
        delta = self.response_start_time - self.trigger_end_time
        return delta.total_seconds() / 60.0

    @property
    def primary_trigger_msg_id(self) -> int | None:
        """Primary (first) trigger message ID."""
        return self.trigger_msg_ids[0] if self.trigger_msg_ids else None

    @property
    def primary_response_msg_id(self) -> int | None:
        """Primary (first) response message ID."""
        return self.response_msg_ids[0] if self.response_msg_ids else None

    def context_to_json(self) -> list[dict[str, object]]:
        """Convert context window to JSON-serializable list."""
        return [msg.to_dict() for msg in self.context_window]

    @property
    def has_trigger_flags(self) -> set[str]:
        """Get union of all flags from trigger span."""
        flags: set[str] = set()
        for msg in self.trigger_span:
            flags.update(msg.flags)
        return flags

    @property
    def has_response_flags(self) -> set[str]:
        """Get union of all flags from response span."""
        flags: set[str] = set()
        for msg in self.response_span:
            flags.update(msg.flags)
        return flags


@dataclass
class ExchangeConfig:
    """Configuration for exchange-based extraction.

    Attributes:
        time_gap_boundary_minutes: Time gap that marks a new conversation thread.
            Messages after this gap are not auto-paired with prior messages.
        response_window_minutes: Max time window for bundling consecutive messages
            from same speaker into one response span.
        trigger_window_minutes: Max time window for bundling consecutive messages
            from same speaker into one trigger span.
        context_window_size: Number of previous messages to include as context.
        max_response_delay_hours: Hard cutoff for response time (drop if exceeded).
    """

    time_gap_boundary_minutes: float = 30.0
    response_window_minutes: float = 5.0
    trigger_window_minutes: float = 5.0
    context_window_size: int = 20
    max_response_delay_hours: float = 168.0  # 1 week
