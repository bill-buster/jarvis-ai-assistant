"""Shared fixtures for integration tests.

Provides common mocks and helpers for testing the RAG flow:
- Mock iMessage reader
- Mock generator
- Mock intent classifier
- Mock context fetcher
- Sample message data
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from contracts.imessage import Attachment, Conversation, Message, Reaction
from contracts.models import GenerationResponse
from core.health import reset_degradation_controller
from core.memory import reset_memory_controller

# --- Intent Classification Stubs ---
# These define the expected interface for jarvis/intent.py (not yet implemented)


class IntentType(Enum):
    """Types of user intents for iMessage interactions."""

    REPLY = "reply"  # User wants help replying to a message
    SUMMARIZE = "summarize"  # User wants a conversation summary
    SEARCH = "search"  # User wants to find messages
    QUICK_REPLY = "quick_reply"  # Simple acknowledgment (ok, thanks, etc.)
    UNKNOWN = "unknown"  # Couldn't determine intent


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    extracted_params: dict[str, Any] = field(default_factory=dict)


class MockIntentClassifier:
    """Mock intent classifier for testing.

    In production, this would use NLP/embeddings to classify user intent.
    """

    def classify(self, user_input: str) -> IntentResult:
        """Classify user intent from their input."""
        user_input_lower = user_input.lower()

        # Simple keyword-based classification for testing
        if any(word in user_input_lower for word in ["reply", "respond", "answer"]):
            # Extract person name if mentioned
            person = None
            for word in user_input.split():
                if word[0].isupper() and word.lower() not in ["help", "me", "i"]:
                    person = word.rstrip("'s").rstrip("'s")
                    break
            return IntentResult(
                intent=IntentType.REPLY,
                confidence=0.9,
                extracted_params={"person": person} if person else {},
            )

        if any(word in user_input_lower for word in ["summary", "summarize", "recap"]):
            return IntentResult(
                intent=IntentType.SUMMARIZE,
                confidence=0.85,
                extracted_params={},
            )

        if any(word in user_input_lower for word in ["find", "search", "look for"]):
            return IntentResult(
                intent=IntentType.SEARCH,
                confidence=0.8,
                extracted_params={},
            )

        # Quick replies - short acknowledgments
        quick_phrases = ["ok", "okay", "thanks", "thank you", "sounds good", "got it", "sure"]
        if any(phrase in user_input_lower for phrase in quick_phrases):
            return IntentResult(
                intent=IntentType.QUICK_REPLY,
                confidence=0.95,
                extracted_params={},
            )

        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.3,
            extracted_params={},
        )


# --- Context Fetching Stubs ---
# These define the expected interface for jarvis/context.py (not yet implemented)


@dataclass
class ReplyContext:
    """Context for generating a reply."""

    messages: list[Message]
    formatted_context: str
    last_received_message: Message
    conversation_summary: str | None = None


class MockContextFetcher:
    """Mock context fetcher for testing.

    In production, this retrieves and formats conversation context.
    """

    def __init__(self, reader: Any):
        self.reader = reader

    def get_reply_context(self, chat_id: str, num_messages: int = 10) -> ReplyContext:
        """Get context for generating a reply."""
        messages = self.reader.get_messages(chat_id, limit=num_messages)

        # Format messages as context string
        formatted_lines = []
        for msg in messages:
            sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
            formatted_lines.append(f"{sender}: {msg.text}")
        formatted_context = "\n".join(formatted_lines)

        # Find last message not from user
        last_received = None
        for msg in reversed(messages):
            if not msg.is_from_me:
                last_received = msg
                break

        if last_received is None and messages:
            last_received = messages[-1]

        return ReplyContext(
            messages=messages,
            formatted_context=formatted_context,
            last_received_message=last_received,
        )

    def get_summary_context(
        self, chat_id: str, num_messages: int = 50
    ) -> tuple[list[Message], str]:
        """Get context for summarizing a conversation."""
        messages = self.reader.get_messages(chat_id, limit=num_messages)
        formatted_lines = []
        for msg in messages:
            sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
            formatted_lines.append(f"[{msg.date.strftime('%Y-%m-%d %H:%M')}] {sender}: {msg.text}")
        return messages, "\n".join(formatted_lines)


# --- Prompt Building Stubs ---
# These define the expected interface for jarvis/prompts.py (not yet implemented)


def build_reply_prompt(context: str, last_message: str) -> str:
    """Build a prompt for generating a reply.

    Args:
        context: Formatted conversation history
        last_message: The message to reply to

    Returns:
        Formatted prompt for the generator
    """
    return f"""You are helping compose a friendly, natural reply to a message.

Conversation history:
{context}

The last message you need to reply to is:
"{last_message}"

Generate a brief, natural reply (1-2 sentences). Be conversational and friendly.
Reply:"""


def build_summary_prompt(context: str, num_messages: int) -> str:
    """Build a prompt for summarizing a conversation.

    Args:
        context: Formatted conversation history
        num_messages: Number of messages being summarized

    Returns:
        Formatted prompt for the generator
    """
    return f"""Summarize the following conversation ({num_messages} messages) in 2-3 sentences.
Focus on the main topics discussed and any action items.

Conversation:
{context}

Summary:"""


# --- Test Data Helpers ---


def create_mock_message(
    text: str,
    is_from_me: bool,
    sender: str = "+1234567890",
    sender_name: str | None = "John",
    msg_id: int = 1,
    chat_id: str = "chat123",
    date: datetime | None = None,
    attachments: list[Attachment] | None = None,
    reactions: list[Reaction] | None = None,
) -> Message:
    """Create a mock Message for testing."""
    return Message(
        id=msg_id,
        chat_id=chat_id,
        sender="" if is_from_me else sender,
        sender_name=None if is_from_me else sender_name,
        text=text,
        date=date or datetime(2024, 1, 15, 10, 30),
        is_from_me=is_from_me,
        attachments=attachments or [],
        reactions=reactions or [],
    )


def create_mock_conversation(
    chat_id: str = "chat123",
    participants: list[str] | None = None,
    display_name: str | None = "John",
    message_count: int = 50,
    is_group: bool = False,
) -> Conversation:
    """Create a mock Conversation for testing."""
    return Conversation(
        chat_id=chat_id,
        participants=participants or ["+1234567890"],
        display_name=display_name,
        last_message_date=datetime(2024, 1, 15, 10, 30),
        message_count=message_count,
        is_group=is_group,
    )


# --- Context Managers for Mocking ---


@contextmanager
def patch_imessage_reader():
    """Context manager to patch the iMessage reader."""
    mock_reader = MagicMock()
    mock_reader.check_access.return_value = True
    mock_reader.get_messages.return_value = []
    mock_reader.get_conversations.return_value = []
    mock_reader.search.return_value = []

    with patch("integrations.imessage.ChatDBReader") as mock_class:
        mock_class.return_value.__enter__ = MagicMock(return_value=mock_reader)
        mock_class.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_reader


@contextmanager
def patch_generator():
    """Context manager to patch the model generator."""
    mock_gen = MagicMock()
    mock_gen.is_loaded.return_value = True
    mock_gen.generate.return_value = GenerationResponse(
        text="Default mock response",
        tokens_used=10,
        generation_time_ms=100.0,
        model_name="mock-model",
        used_template=False,
        template_name=None,
        finish_reason="stop",
    )

    with patch("models.get_generator", return_value=mock_gen):
        yield mock_gen


@contextmanager
def patch_services():
    """Context manager to patch all RAG services for API testing."""
    with patch_imessage_reader() as mock_reader:
        with patch_generator() as mock_gen:
            with patch("jarvis.api.get_degradation_controller") as mock_deg:
                # Configure degradation controller to pass through
                mock_controller = MagicMock()
                mock_controller.execute.side_effect = lambda feature, func, *args: func(*args)
                mock_deg.return_value = mock_controller
                yield {
                    "reader": mock_reader,
                    "generator": mock_gen,
                    "degradation": mock_controller,
                }


# --- Pytest Fixtures ---


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_memory_controller()
    reset_degradation_controller()
    yield
    reset_memory_controller()
    reset_degradation_controller()


@pytest.fixture
def mock_messages() -> list[Message]:
    """Sample conversation messages for testing."""
    return [
        create_mock_message(
            "Hey, dinner tomorrow?",
            is_from_me=False,
            msg_id=1,
            date=datetime(2024, 1, 15, 18, 0),
        ),
        create_mock_message(
            "Sure, what time?",
            is_from_me=True,
            msg_id=2,
            date=datetime(2024, 1, 15, 18, 5),
        ),
        create_mock_message(
            "7pm at my place?",
            is_from_me=False,
            msg_id=3,
            date=datetime(2024, 1, 15, 18, 10),
        ),
    ]


@pytest.fixture
def mock_work_messages() -> list[Message]:
    """Sample work-related conversation for testing."""
    return [
        create_mock_message(
            "Did you finish the report?",
            is_from_me=False,
            sender_name="Boss",
            msg_id=1,
            date=datetime(2024, 1, 15, 9, 0),
        ),
        create_mock_message(
            "Almost done, just reviewing the numbers",
            is_from_me=True,
            msg_id=2,
            date=datetime(2024, 1, 15, 9, 30),
        ),
        create_mock_message(
            "Great, can you send it by EOD?",
            is_from_me=False,
            sender_name="Boss",
            msg_id=3,
            date=datetime(2024, 1, 15, 9, 35),
        ),
    ]


@pytest.fixture
def intent_classifier() -> MockIntentClassifier:
    """Get a mock intent classifier."""
    return MockIntentClassifier()


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient

    from jarvis.api import app

    return TestClient(app, raise_server_exceptions=False)
