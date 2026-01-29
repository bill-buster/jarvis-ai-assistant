"""Minimal pytest configuration for JARVIS v3 tests.

Provides only essential fixtures for fast testing without model loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest


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
            "text": "How about 7pm?",
            "sender": "+1234567890",
            "sender_name": "John",
            "is_from_me": False,
            "timestamp": datetime(2024, 1, 15, 18, 35),
        },
    ]


@pytest.fixture
def mock_model_loader():
    """Mock MLX model loader for testing without GPU."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.current_model = "lfm2.5-1.2b"
    mock.current_model_id = "lfm2.5-1.2b"

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
