"""Pytest configuration and fixtures for JARVIS v2 tests.

These fixtures provide reusable test data and mocks for testing
the generation pipeline without requiring actual model loading
or database access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# === Sample Data Fixtures ===


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    """Sample conversation messages for testing."""
    return [
        {
            "text": "Hey, are you free for dinner tonight?",
            "sender": "+1234567890",
            "sender_name": "John",
            "is_from_me": False,
            "timestamp": datetime(2024, 1, 15, 18, 30),
        },
        {
            "text": "Yeah sounds good! What time?",
            "sender": "me",
            "sender_name": None,
            "is_from_me": True,
            "timestamp": datetime(2024, 1, 15, 18, 32),
        },
        {
            "text": "How about 7pm at the Italian place?",
            "sender": "+1234567890",
            "sender_name": "John",
            "is_from_me": False,
            "timestamp": datetime(2024, 1, 15, 18, 35),
        },
    ]


@pytest.fixture
def casual_conversation() -> list[dict[str, Any]]:
    """Casual conversation with friends (lots of abbreviations, emojis)."""
    return [
        {"text": "lol that's hilarious 😂", "sender": "Friend", "is_from_me": False},
        {"text": "haha ikr", "sender": "me", "is_from_me": True},
        {"text": "wanna hang tmrw?", "sender": "Friend", "is_from_me": False},
        {"text": "ya sounds good", "sender": "me", "is_from_me": True},
        {"text": "cool see u then", "sender": "Friend", "is_from_me": False},
    ]


@pytest.fixture
def formal_conversation() -> list[dict[str, Any]]:
    """Formal/work conversation (proper grammar, no abbreviations)."""
    return [
        {
            "text": "Hello, I wanted to follow up on the project.",
            "sender": "Colleague",
            "is_from_me": False,
        },
        {
            "text": "Thank you for reaching out. The project is on track.",
            "sender": "me",
            "is_from_me": True,
        },
        {
            "text": "Excellent. Could we schedule a meeting for next week?",
            "sender": "Colleague",
            "is_from_me": False,
        },
    ]


@pytest.fixture
def emotional_conversation() -> list[dict[str, Any]]:
    """Emotionally charged conversation."""
    return [
        {"text": "I got the job!!!!", "sender": "Friend", "is_from_me": False},
        {"text": "OMG CONGRATS!!! 🎉🎉🎉", "sender": "me", "is_from_me": True},
        {"text": "I'm so happy I could cry", "sender": "Friend", "is_from_me": False},
    ]


@pytest.fixture
def group_chat_messages() -> list[dict[str, Any]]:
    """Messages from a group chat with multiple senders."""
    return [
        {"text": "Hey everyone!", "sender": "Alice", "is_from_me": False},
        {"text": "Hi!", "sender": "Bob", "is_from_me": False},
        {"text": "What's the plan for tonight?", "sender": "Charlie", "is_from_me": False},
        {"text": "Thinking dinner at 7", "sender": "Alice", "is_from_me": False},
        {"text": "Works for me!", "sender": "me", "is_from_me": True},
        {"text": "Same here", "sender": "Bob", "is_from_me": False},
        {"text": "Perfect!", "sender": "Alice", "is_from_me": False},
    ]


@pytest.fixture
def long_conversation() -> list[dict[str, Any]]:
    """Long conversation for testing truncation."""
    messages = []
    for i in range(20):
        is_from_me = i % 2 == 0
        messages.append(
            {
                "text": f"Message number {i}",
                "sender": "me" if is_from_me else "Friend",
                "is_from_me": is_from_me,
            }
        )
    return messages


@pytest.fixture
def sample_style() -> dict[str, Any]:
    """Sample user style for testing."""
    return {
        "avg_message_length": 45.0,
        "uses_emoji": True,
        "emoji_frequency": 0.3,
        "common_emojis": ["😊", "👍", "😂"],
        "punctuation_style": "casual",
        "capitalization": "lowercase",
        "common_phrases": ["sounds good", "for sure", "yeah"],
        "enthusiasm_level": "medium",
        "abbreviations": ["lol", "brb", "omw"],
    }


# === Mock Fixtures ===


@pytest.fixture
def mock_model_loader():
    """Mock MLX model loader for testing without GPU."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.current_model = "qwen-1.5b"
    mock.current_model_id = "qwen-1.5b"

    # Return a proper result object
    @dataclass
    class MockGenerationResult:
        text: str = "Sounds great! See you at 7."
        formatted_prompt: str = "[Mock prompt]"

    mock.generate.return_value = MockGenerationResult()
    return mock


@pytest.fixture
def mock_message_reader():
    """Mock iMessage reader for testing without database access."""
    mock = MagicMock()
    mock.check_access.return_value = True
    mock.get_messages.return_value = []
    mock.get_conversations.return_value = []
    return mock


@pytest.fixture
def mock_embedding_store():
    """Mock embedding store for testing without FAISS."""
    mock = MagicMock()
    mock.search_similar.return_value = []
    mock.is_index_ready.return_value = True
    mock.find_your_past_replies.return_value = []
    mock.find_similar.return_value = []
    mock.find_similar_hybrid.return_value = []
    mock.get_user_response_patterns.return_value = {}
    return mock


@pytest.fixture
def mock_template_matcher():
    """Mock template matcher that returns no matches."""
    mock = MagicMock()
    mock.match.return_value = None
    return mock


@pytest.fixture
def mock_template_matcher_with_match():
    """Mock template matcher that returns a match."""

    @dataclass
    class MockTemplateMatch:
        trigger: str = "test trigger"
        actual: str = "sounds good!"
        confidence: float = 0.85

    mock = MagicMock()
    mock.match.return_value = MockTemplateMatch()
    return mock


@pytest.fixture
def mock_contact_profile():
    """Mock contact profile."""

    @dataclass
    class MockContactProfile:
        display_name: str = "John"
        relationship_type: str = "close_friend"
        relationship_summary: str = "good friend who you joke around with"
        tone: str = "playful"
        total_messages: int = 100
        uses_emoji: bool = True
        uses_slang: bool = True
        is_playful: bool = True
        avg_your_length: float = 35.0
        topics: list = field(default_factory=list)
        your_common_phrases: list = field(default_factory=list)

    return MockContactProfile()


@pytest.fixture
def mock_global_style():
    """Mock global user style."""

    @dataclass
    class MockGlobalStyle:
        capitalization: str = "lowercase"
        punctuation_style: str = "minimal"
        uses_abbreviations: bool = True
        avg_word_count: float = 5.0
        personality_summary: str = "casual and friendly texter"
        interests: list = field(default_factory=lambda: ["music", "travel", "food"])
        common_phrases: list = field(
            default_factory=lambda: ["sounds good", "for sure", "lol"]
        )

    return MockGlobalStyle()


# === Helper Fixtures ===


@pytest.fixture
def mock_relationship_registry():
    """Mock relationship registry for testing."""

    @dataclass
    class MockRelationshipInfo:
        contact_name: str = "Alice"
        relationship: str = "best friend"
        category: str = "friend"
        is_group: bool = False
        phones: list = field(default_factory=lambda: ["+15551234567"])

    mock = MagicMock()
    mock.get_relationship.return_value = MockRelationshipInfo()
    mock.get_relationship_from_chat_id.return_value = MockRelationshipInfo()
    mock.get_similar_contacts.return_value = ["Bob", "Charlie"]
    mock.get_phones_for_contacts.return_value = {
        "Bob": ["+15552345678"],
        "Charlie": ["+15553456789"],
    }
    return mock


@pytest.fixture
def reply_generator_with_mocks(mock_model_loader):
    """Create a ReplyGenerator with all external dependencies mocked."""
    from core.generation.reply_generator import ReplyGenerator

    with patch(
        "core.generation.reply_generator._get_template_matcher", return_value=None
    ), patch(
        "core.generation.reply_generator._get_embedding_store", return_value=None
    ), patch(
        "core.generation.reply_generator._get_contact_profile", return_value=None
    ), patch(
        "core.generation.reply_generator.get_global_user_style", return_value=None
    ), patch(
        "core.generation.reply_generator._get_relationship_registry", return_value=None
    ):
        generator = ReplyGenerator(mock_model_loader)
        generator._template_matcher = None
        yield generator


@pytest.fixture
def context_analyzer():
    """Create a ContextAnalyzer instance."""
    from core.generation.context_analyzer import ContextAnalyzer

    return ContextAnalyzer()


@pytest.fixture
def style_analyzer():
    """Create a StyleAnalyzer instance."""
    from core.generation.style_analyzer import StyleAnalyzer

    return StyleAnalyzer()


# === Parametrized Test Data ===


def pytest_generate_tests(metafunc):
    """Generate parametrized test cases for common scenarios."""
    if "greeting_message" in metafunc.fixturenames:
        metafunc.parametrize(
            "greeting_message",
            [
                "hey",
                "Hey!",
                "Hello",
                "hi there",
                "what's up?",
                "How are you?",
                "yo",
                "sup",
                "good morning",
            ],
        )

    if "question_message" in metafunc.fixturenames:
        metafunc.parametrize(
            "question_message",
            [
                ("Do you want to come?", "yes_no"),
                ("Are you free?", "yes_no"),
                ("Can you help?", "yes_no"),
                ("Pizza or tacos?", "choice"),
                ("Red or blue?", "choice"),
                ("What time?", "open"),
                ("Where should we go?", "open"),
            ],
        )
