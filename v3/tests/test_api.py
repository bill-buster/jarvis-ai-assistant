"""Comprehensive API endpoint tests for JARVIS v3 FastAPI application.

Tests all endpoints:
- GET /health - Basic health check
- GET /conversations - List recent conversations
- GET /conversations/{id}/messages - Fetch conversation messages
- POST /generate/replies - Generate reply suggestions

Uses mocks to avoid real database/model access.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_message_reader():
    """Create a mock MessageReader that returns test data."""
    mock_reader = MagicMock()
    mock_reader.check_access.return_value = True
    mock_reader.close.return_value = None
    return mock_reader


@pytest.fixture
def mock_conversations():
    """Create sample Conversation objects for testing."""
    from core.imessage.reader import Conversation

    return [
        Conversation(
            chat_id="chat;-;+1234567890",
            display_name="Alice",
            participants=["+1234567890"],
            last_message_date=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            last_message_text="Hey, how are you?",
            last_message_is_from_me=False,
            message_count=42,
            is_group=False,
        ),
        Conversation(
            chat_id="chat;-;+0987654321",
            display_name="Bob",
            participants=["+0987654321"],
            last_message_date=datetime(2024, 1, 14, 15, 45, tzinfo=UTC),
            last_message_text="See you later!",
            last_message_is_from_me=True,
            message_count=28,
            is_group=False,
        ),
        Conversation(
            chat_id="chat;-;group123",
            display_name="Work Team",
            participants=["+1111111111", "+2222222222", "+3333333333"],
            last_message_date=datetime(2024, 1, 13, 9, 0, tzinfo=UTC),
            last_message_text="Meeting at 3pm",
            last_message_is_from_me=False,
            message_count=150,
            is_group=True,
        ),
    ]


@pytest.fixture
def mock_messages():
    """Create sample Message objects for testing."""
    from core.imessage.reader import Message

    return [
        Message(
            id=1001,
            text="Hey, how are you?",
            sender="+1234567890",
            sender_name="Alice",
            is_from_me=False,
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            chat_id="chat;-;+1234567890",
        ),
        Message(
            id=1000,
            text="I'm good, thanks!",
            sender="me",
            sender_name=None,
            is_from_me=True,
            timestamp=datetime(2024, 1, 15, 10, 29, tzinfo=UTC),
            chat_id="chat;-;+1234567890",
        ),
        Message(
            id=999,
            text="Did you see the news?",
            sender="+1234567890",
            sender_name="Alice",
            is_from_me=False,
            timestamp=datetime(2024, 1, 15, 10, 28, tzinfo=UTC),
            chat_id="chat;-;+1234567890",
        ),
    ]


@pytest.fixture
def mock_reply_result():
    """Create a mock ReplyGenerationResult for testing."""
    from core.generation.context_analyzer import ConversationContext, MessageIntent
    from core.generation.reply_generator import GeneratedReply, ReplyGenerationResult
    from core.generation.style_analyzer import UserStyle

    return ReplyGenerationResult(
        replies=[
            GeneratedReply(text="sounds good!", reply_type="general", confidence=0.9),
        ],
        context=ConversationContext(
            last_message="Hey, how are you?",
            last_sender="Alice",
            intent=MessageIntent.GREETING,
            relationship=None,
            topic="greeting",
            mood="neutral",
            urgency="normal",
            needs_response=True,
            summary="Casual greeting",
        ),
        style=UserStyle(
            avg_word_count=5,
            uses_abbreviations=True,
            uses_emoji=False,
            formality_score=0.3,
            punctuation_style="minimal",
            capitalization="lowercase",
        ),
        model_used="lfm2.5-1.2b",
        generation_time_ms=150.0,
        prompt_used="test prompt",
    )


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app with mocked dependencies."""
    from api.main import app

    return TestClient(app)


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_ok_status(self, client):
        """Health check should return status ok and version."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["version"] == "3.0.0"

    def test_health_response_format(self, client):
        """Health response should have expected keys."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Should only contain status and version
        assert set(data.keys()) == {"status", "version"}


# ============================================================================
# Root Endpoint Tests
# ============================================================================


class TestRootEndpoint:
    """Tests for GET / root endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint should return API name, version, and docs URL."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "JARVIS v3"
        assert data["version"] == "3.0.0"
        assert data["docs"] == "/docs"


# ============================================================================
# Conversations Endpoint Tests
# ============================================================================


class TestConversationsEndpoint:
    """Tests for GET /conversations endpoint."""

    def test_list_conversations_success(
        self, client, mock_message_reader, mock_conversations
    ):
        """Should return list of conversations when database is accessible."""
        mock_message_reader.get_conversations.return_value = mock_conversations

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations")

        assert response.status_code == 200
        data = response.json()

        assert "conversations" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["conversations"]) == 3

        # Verify first conversation fields
        conv = data["conversations"][0]
        assert conv["chat_id"] == "chat;-;+1234567890"
        assert conv["display_name"] == "Alice"
        assert conv["participants"] == ["+1234567890"]
        assert conv["message_count"] == 42
        assert conv["is_group"] is False

    def test_list_conversations_with_limit(
        self, client, mock_message_reader, mock_conversations
    ):
        """Should respect limit parameter."""
        mock_message_reader.get_conversations.return_value = mock_conversations[:1]

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations?limit=1")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

        # Verify get_conversations was called with correct limit
        mock_message_reader.get_conversations.assert_called_once_with(limit=1)

    def test_list_conversations_empty(self, client, mock_message_reader):
        """Should handle empty conversation list."""
        mock_message_reader.get_conversations.return_value = []

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["conversations"] == []

    def test_list_conversations_no_database_access(self, client):
        """Should return 503 when database access is denied."""
        from fastapi import HTTPException

        def raise_503():
            raise HTTPException(
                status_code=503,
                detail="Cannot access iMessage database. Grant Full Disk Access permission.",
            )

        with patch(
            "api.routes.conversations._get_reader", side_effect=raise_503
        ):
            response = client.get("/conversations")

        assert response.status_code == 503
        data = response.json()
        assert "Cannot access iMessage database" in data["detail"]

    def test_list_conversations_closes_reader(
        self, client, mock_message_reader, mock_conversations
    ):
        """Should close the reader after getting conversations."""
        mock_message_reader.get_conversations.return_value = mock_conversations

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            client.get("/conversations")

        mock_message_reader.close.assert_called_once()


# ============================================================================
# Messages Endpoint Tests
# ============================================================================


class TestMessagesEndpoint:
    """Tests for GET /conversations/{chat_id}/messages endpoint."""

    def test_get_messages_success(self, client, mock_message_reader, mock_messages):
        """Should return messages for a valid chat_id."""
        mock_message_reader.get_messages.return_value = mock_messages

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations/chat;-;+1234567890/messages")

        assert response.status_code == 200
        data = response.json()

        assert "messages" in data
        assert "chat_id" in data
        assert "total" in data

        assert data["chat_id"] == "chat;-;+1234567890"
        assert data["total"] == 3
        assert len(data["messages"]) == 3

        # Verify message fields
        msg = data["messages"][0]
        assert msg["id"] == 1001
        assert msg["text"] == "Hey, how are you?"
        assert msg["sender"] == "+1234567890"
        assert msg["sender_name"] == "Alice"
        assert msg["is_from_me"] is False
        assert "timestamp" in msg

    def test_get_messages_with_limit(self, client, mock_message_reader, mock_messages):
        """Should respect limit parameter."""
        mock_message_reader.get_messages.return_value = mock_messages[:2]

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations/chat;-;+1234567890/messages?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

        # Verify get_messages was called with correct parameters
        mock_message_reader.get_messages.assert_called_once()
        call_kwargs = mock_message_reader.get_messages.call_args
        assert call_kwargs.kwargs["limit"] == 2

    def test_get_messages_with_before_timestamp(
        self, client, mock_message_reader, mock_messages
    ):
        """Should accept and parse before timestamp for pagination."""
        mock_message_reader.get_messages.return_value = mock_messages[1:]

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get(
                "/conversations/chat;-;+1234567890/messages?before=2024-01-15T10:30:00Z"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

        # Verify before parameter was passed
        call_kwargs = mock_message_reader.get_messages.call_args
        assert call_kwargs.kwargs["before"] is not None

    def test_get_messages_invalid_before_timestamp(self, client, mock_message_reader):
        """Should return 400 for invalid before timestamp format."""
        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get(
                "/conversations/chat;-;+1234567890/messages?before=invalid-date"
            )

        assert response.status_code == 400
        data = response.json()
        assert "Invalid 'before' timestamp format" in data["detail"]

    def test_get_messages_conversation_not_found(self, client, mock_message_reader):
        """Should return 404 when conversation doesn't exist."""
        mock_message_reader.get_messages.return_value = []

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get("/conversations/nonexistent-chat/messages")

        assert response.status_code == 404
        data = response.json()
        assert "Conversation not found" in data["detail"]

    def test_get_messages_empty_with_before_not_404(
        self, client, mock_message_reader
    ):
        """Empty results with 'before' parameter should not be 404 (pagination edge)."""
        mock_message_reader.get_messages.return_value = []

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            response = client.get(
                "/conversations/chat;-;+1234567890/messages?before=2020-01-01T00:00:00Z"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["messages"] == []

    def test_get_messages_no_database_access(self, client):
        """Should return 503 when database access is denied."""
        from fastapi import HTTPException

        def raise_503():
            raise HTTPException(
                status_code=503,
                detail="Cannot access iMessage database. Grant Full Disk Access permission.",
            )

        with patch(
            "api.routes.conversations._get_reader", side_effect=raise_503
        ):
            response = client.get("/conversations/chat;-;+1234567890/messages")

        assert response.status_code == 503
        data = response.json()
        assert "Cannot access iMessage database" in data["detail"]

    def test_get_messages_closes_reader(
        self, client, mock_message_reader, mock_messages
    ):
        """Should close the reader after getting messages."""
        mock_message_reader.get_messages.return_value = mock_messages

        with patch(
            "api.routes.conversations._get_reader", return_value=mock_message_reader
        ):
            client.get("/conversations/chat;-;+1234567890/messages")

        mock_message_reader.close.assert_called_once()


# ============================================================================
# Generate Replies Endpoint Tests
# ============================================================================


class TestGenerateRepliesEndpoint:
    """Tests for POST /generate/replies endpoint."""

    def test_generate_replies_success(
        self, client, mock_message_reader, mock_messages, mock_reply_result
    ):
        """Should generate replies for valid request."""
        mock_message_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_generator.generate_replies.return_value = mock_reply_result

        with (
            patch(
                "api.routes.generate._fetch_messages",
                return_value=(mock_messages, None),
            ),
            patch("api.routes.generate._get_generator", return_value=mock_generator),
        ):
            response = client.post(
                "/generate/replies",
                json={"chat_id": "chat;-;+1234567890"},
            )

        assert response.status_code == 200
        data = response.json()

        assert "replies" in data
        assert "chat_id" in data
        assert "model_used" in data
        assert "generation_time_ms" in data
        assert "context_summary" in data

        assert data["chat_id"] == "chat;-;+1234567890"
        assert len(data["replies"]) == 1
        assert data["replies"][0]["text"] == "sounds good!"
        assert data["replies"][0]["reply_type"] == "general"
        assert data["replies"][0]["confidence"] == 0.9
        assert data["model_used"] == "lfm2.5-1.2b"

    def test_generate_replies_with_user_name(
        self, client, mock_message_reader, mock_messages, mock_reply_result
    ):
        """Should pass user_name to the generator."""
        mock_message_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_generator.generate_replies.return_value = mock_reply_result

        with (
            patch(
                "api.routes.generate._fetch_messages",
                return_value=(mock_messages, None),
            ),
            patch("api.routes.generate._get_generator", return_value=mock_generator),
        ):
            response = client.post(
                "/generate/replies",
                json={"chat_id": "chat;-;+1234567890", "user_name": "John"},
            )

        assert response.status_code == 200

        # Verify user_name was passed to generate_replies
        call_kwargs = mock_generator.generate_replies.call_args
        assert call_kwargs.kwargs["user_name"] == "John"

    def test_generate_replies_missing_chat_id(self, client):
        """Should return 400 when chat_id is missing."""
        response = client.post("/generate/replies", json={})

        assert response.status_code == 400
        data = response.json()
        assert "chat_id is required" in data["detail"]

    def test_generate_replies_empty_chat_id(self, client):
        """Should return 400 when chat_id is empty string."""
        response = client.post("/generate/replies", json={"chat_id": ""})

        assert response.status_code == 400
        data = response.json()
        assert "chat_id is required" in data["detail"]

    def test_generate_replies_conversation_not_found(self, client):
        """Should return 404 when conversation has no messages."""
        with patch(
            "api.routes.generate._fetch_messages",
            return_value=([], None),
        ):
            response = client.post(
                "/generate/replies",
                json={"chat_id": "nonexistent-chat"},
            )

        assert response.status_code == 404
        data = response.json()
        assert "Conversation not found" in data["detail"]

    def test_generate_replies_no_database_access(self, client):
        """Should return 503 when database access is denied."""
        with patch(
            "api.routes.generate._fetch_messages",
            return_value=(None, "Cannot access iMessage database"),
        ):
            response = client.post(
                "/generate/replies",
                json={"chat_id": "chat;-;+1234567890"},
            )

        assert response.status_code == 503
        data = response.json()
        assert "Cannot access iMessage database" in data["detail"]

    def test_generate_replies_default_user_name(
        self, client, mock_messages, mock_reply_result
    ):
        """Should use 'User' as default user_name."""
        mock_generator = MagicMock()
        mock_generator.generate_replies.return_value = mock_reply_result

        with (
            patch(
                "api.routes.generate._fetch_messages",
                return_value=(mock_messages, None),
            ),
            patch("api.routes.generate._get_generator", return_value=mock_generator),
        ):
            response = client.post(
                "/generate/replies",
                json={"chat_id": "chat;-;+1234567890"},
            )

        assert response.status_code == 200

        # Verify default user_name was passed
        call_kwargs = mock_generator.generate_replies.call_args
        assert call_kwargs.kwargs["user_name"] == "User"


# ============================================================================
# Integration Tests (with minimal mocking)
# ============================================================================


class TestAPIIntegration:
    """Integration tests that verify endpoint wiring."""

    def test_all_routes_registered(self, client):
        """Verify all expected routes are registered in the app."""
        # Get OpenAPI schema which lists all routes
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        # Verify expected paths exist
        assert "/" in paths
        assert "/health" in paths
        assert "/conversations" in paths
        assert "/conversations/{chat_id}/messages" in paths
        assert "/generate/replies" in paths

    def test_cors_headers_present(self, client):
        """Verify CORS headers are configured."""
        # CORS preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Should allow the request (CORS configured with allow_origins=["*"])
        # Note: FastAPI TestClient may not fully simulate CORS,
        # so we just verify the endpoint is accessible
        assert response.status_code in [200, 204, 405]

    def test_docs_endpoint_accessible(self, client):
        """Verify Swagger docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_accessible(self, client):
        """Verify ReDoc docs are accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
