"""Unit tests for the drafts API endpoint.

Tests AI-powered draft reply generation and conversation summarization
with mocked generator and iMessage reader.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_imessage_reader
from api.main import app
from api.routers.drafts import (
    _build_reply_prompt,
    _build_summary_prompt,
    _format_messages_for_context,
    _parse_summary_response,
)
from contracts.imessage import Message
from contracts.models import GenerationResponse


@pytest.fixture
def mock_messages():
    """Create mock messages for testing."""
    return [
        Message(
            id=3,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Hey, dinner at my place tomorrow?",
            date=datetime(2024, 1, 25, 18, 30),
            is_from_me=False,
        ),
        Message(
            id=2,
            chat_id="iMessage;-;+1234567890",
            sender="me",
            sender_name=None,
            text="What are you up to this weekend?",
            date=datetime(2024, 1, 25, 18, 25),
            is_from_me=True,
        ),
        Message(
            id=1,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Hey!",
            date=datetime(2024, 1, 25, 18, 20),
            is_from_me=False,
        ),
    ]


@pytest.fixture
def mock_reader():
    """Create a mock iMessage reader."""
    reader = MagicMock()
    reader.check_access.return_value = True
    return reader


@pytest.fixture
def client(mock_reader):
    """Create a test client with mocked iMessage reader."""
    # Override the dependency
    app.dependency_overrides[get_imessage_reader] = lambda: mock_reader
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    # Clean up
    app.dependency_overrides.clear()


class TestFormatMessagesForContext:
    """Tests for _format_messages_for_context helper."""

    def test_formats_messages_chronologically(self, mock_messages):
        """Messages are formatted in chronological order (oldest first)."""
        result = _format_messages_for_context(mock_messages)
        lines = result.split("\n")
        # mock_messages is newest-first, so after reversal:
        # "Hey!" should be first, "Hey, dinner..." should be last
        assert "[John Smith]: Hey!" in lines[0]
        assert "[John Smith]: Hey, dinner at my place tomorrow?" in lines[2]

    def test_uses_you_for_is_from_me(self, mock_messages):
        """Messages from user are labeled as 'You'."""
        result = _format_messages_for_context(mock_messages)
        assert "[You]: What are you up to this weekend?" in result

    def test_uses_sender_name_when_available(self, mock_messages):
        """Uses sender_name when available."""
        result = _format_messages_for_context(mock_messages)
        assert "[John Smith]:" in result

    def test_falls_back_to_sender_when_no_name(self):
        """Falls back to sender identifier when sender_name is None."""
        messages = [
            Message(
                id=1,
                chat_id="test",
                sender="+1234567890",
                sender_name=None,
                text="Hello",
                date=datetime.now(),
                is_from_me=False,
            )
        ]
        result = _format_messages_for_context(messages)
        assert "[+1234567890]: Hello" in result

    def test_empty_messages_list(self):
        """Returns empty string for empty message list."""
        result = _format_messages_for_context([])
        assert result == ""


class TestBuildReplyPrompt:
    """Tests for _build_reply_prompt helper."""

    def test_includes_last_message(self):
        """Prompt includes the last message."""
        result = _build_reply_prompt("Dinner tomorrow?", None, 1)
        assert "Last message: 'Dinner tomorrow?'" in result

    def test_includes_instruction_when_provided(self):
        """Prompt includes instruction when provided."""
        result = _build_reply_prompt("Dinner?", "say yes enthusiastically", 1)
        assert "Instruction: say yes enthusiastically" in result

    def test_shows_none_instruction_when_not_provided(self):
        """Prompt shows None instruction when not provided."""
        result = _build_reply_prompt("Hello", None, 1)
        assert "Instruction: None" in result

    def test_varies_hint_based_on_suggestion_number(self):
        """Different suggestion numbers get different variety hints."""
        result1 = _build_reply_prompt("Hi", None, 1)
        result2 = _build_reply_prompt("Hi", None, 2)
        result3 = _build_reply_prompt("Hi", None, 3)
        # All should have different hints
        assert "natural, conversational" in result1
        assert "casual" in result2
        assert "concise" in result3


class TestBuildSummaryPrompt:
    """Tests for _build_summary_prompt helper."""

    def test_includes_message_count(self):
        """Prompt includes the message count."""
        result = _build_summary_prompt("Some context", 42)
        assert "42 messages" in result

    def test_includes_context(self):
        """Prompt includes the conversation context."""
        result = _build_summary_prompt("John: Hello\nYou: Hi", 2)
        assert "John: Hello" in result
        assert "You: Hi" in result

    def test_requests_format(self):
        """Prompt requests specific output format."""
        result = _build_summary_prompt("Test", 1)
        assert "Summary:" in result
        assert "Key points:" in result


class TestParseSummaryResponse:
    """Tests for _parse_summary_response helper."""

    def test_parses_well_formatted_response(self):
        """Parses properly formatted LLM response."""
        response = """Summary: This was a discussion about dinner plans.
Key points:
- Meeting at 7pm
- John is cooking
- Italian food"""
        summary, key_points = _parse_summary_response(response)
        assert summary == "This was a discussion about dinner plans."
        assert "Meeting at 7pm" in key_points
        assert "John is cooking" in key_points
        assert "Italian food" in key_points

    def test_handles_bullet_points_with_dots(self):
        """Handles bullet points with • character."""
        response = """Summary: Quick chat
Key points:
• Point one
• Point two"""
        summary, key_points = _parse_summary_response(response)
        assert "Point one" in key_points
        assert "Point two" in key_points

    def test_fallback_for_unformatted_response(self):
        """Falls back gracefully for unformatted response."""
        response = "Just a plain text response without proper formatting."
        summary, key_points = _parse_summary_response(response)
        assert summary == response
        assert key_points == ["See summary for details"]

    def test_truncates_long_fallback_summary(self):
        """Truncates very long responses in fallback mode."""
        response = "A" * 300
        summary, key_points = _parse_summary_response(response)
        assert len(summary) == 200


class TestDraftReplyEndpoint:
    """Tests for POST /drafts/reply endpoint."""

    @patch("api.routers.drafts.get_generator")
    def test_successful_reply_generation(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Successfully generates reply suggestions."""
        # Setup mocks
        mock_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_response = GenerationResponse(
            text="Sounds great! What should I bring?",
            tokens_used=10,
            generation_time_ms=100.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )
        mock_generator.generate.return_value = mock_response
        mock_get_generator.return_value = mock_generator

        response = client.post(
            "/drafts/reply",
            json={
                "chat_id": "iMessage;-;+1234567890",
                "num_suggestions": 2,
                "context_messages": 20,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 2
        assert "context_used" in data
        assert data["context_used"]["num_messages"] == 3
        assert "John Smith" in data["context_used"]["participants"]

    @patch("api.routers.drafts.get_generator")
    def test_reply_with_instruction(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Generates reply with custom instruction."""
        mock_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_response = GenerationResponse(
            text="Yes! What can I bring?",
            tokens_used=8,
            generation_time_ms=80.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )
        mock_generator.generate.return_value = mock_response
        mock_get_generator.return_value = mock_generator

        response = client.post(
            "/drafts/reply",
            json={
                "chat_id": "iMessage;-;+1234567890",
                "num_suggestions": 1,
                "instruction": "say yes but ask what to bring",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) == 1

    def test_reply_no_messages_found(self, client, mock_reader):
        """Returns 404 when no messages found for chat."""
        mock_reader.get_messages.return_value = []

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "nonexistent-chat"},
        )

        assert response.status_code == 404
        assert "No messages found" in response.json()["detail"]

    def test_reply_fetch_error(self, client, mock_reader):
        """Returns 500 when message fetching fails."""
        mock_reader.get_messages.side_effect = RuntimeError("Database error")

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test-chat"},
        )

        assert response.status_code == 500
        assert "Failed to fetch" in response.json()["detail"]

    @patch("api.routers.drafts.get_generator")
    def test_reply_generator_unavailable(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Returns 503 when generator is unavailable."""
        mock_reader.get_messages.return_value = mock_messages
        mock_get_generator.side_effect = RuntimeError("Model not available")

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test-chat"},
        )

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    @patch("api.routers.drafts.get_generator")
    def test_reply_all_generations_fail(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Returns 500 when all generation attempts fail."""
        mock_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_generator.generate.side_effect = RuntimeError("Generation failed")
        mock_get_generator.return_value = mock_generator

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test-chat", "num_suggestions": 3},
        )

        assert response.status_code == 500
        assert "Failed to generate" in response.json()["detail"]

    def test_reply_validates_num_suggestions_range(self, client, mock_reader):
        """Validates num_suggestions is within range."""
        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test", "num_suggestions": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test", "num_suggestions": 10},
        )
        assert response.status_code == 422

    def test_reply_validates_context_messages_range(self, client, mock_reader):
        """Validates context_messages is within range."""
        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test", "context_messages": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/drafts/reply",
            json={"chat_id": "test", "context_messages": 200},
        )
        assert response.status_code == 422


class TestDraftSummarizeEndpoint:
    """Tests for POST /drafts/summarize endpoint."""

    @patch("api.routers.drafts.get_generator")
    def test_successful_summarization(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Successfully summarizes a conversation."""
        mock_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_response = GenerationResponse(
            text=(
                "Summary: Discussion about dinner plans.\n"
                "Key points:\n- Dinner tomorrow\n- At John's place"
            ),
            tokens_used=30,
            generation_time_ms=200.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )
        mock_generator.generate.return_value = mock_response
        mock_get_generator.return_value = mock_generator

        response = client.post(
            "/drafts/summarize",
            json={
                "chat_id": "iMessage;-;+1234567890",
                "num_messages": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "Discussion about dinner plans" in data["summary"]
        assert "key_points" in data
        assert len(data["key_points"]) >= 1
        assert "date_range" in data
        assert data["date_range"]["start"] == "2024-01-25"
        assert data["date_range"]["end"] == "2024-01-25"

    def test_summarize_no_messages_found(self, client, mock_reader):
        """Returns 404 when no messages found for chat."""
        mock_reader.get_messages.return_value = []

        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "nonexistent-chat"},
        )

        assert response.status_code == 404
        assert "No messages found" in response.json()["detail"]

    def test_summarize_fetch_error(self, client, mock_reader):
        """Returns 500 when message fetching fails."""
        mock_reader.get_messages.side_effect = RuntimeError("Database error")

        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "test-chat"},
        )

        assert response.status_code == 500
        assert "Failed to fetch" in response.json()["detail"]

    @patch("api.routers.drafts.get_generator")
    def test_summarize_generator_unavailable(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Returns 503 when generator is unavailable."""
        mock_reader.get_messages.return_value = mock_messages
        mock_get_generator.side_effect = RuntimeError("Model not available")

        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "test-chat"},
        )

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    @patch("api.routers.drafts.get_generator")
    def test_summarize_generation_fails(
        self, mock_get_generator, client, mock_reader, mock_messages
    ):
        """Returns 500 when summary generation fails."""
        mock_reader.get_messages.return_value = mock_messages

        mock_generator = MagicMock()
        mock_generator.generate.side_effect = RuntimeError("Generation failed")
        mock_get_generator.return_value = mock_generator

        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "test-chat"},
        )

        assert response.status_code == 500
        assert "Failed to generate" in response.json()["detail"]

    def test_summarize_validates_num_messages_range(self, client, mock_reader):
        """Validates num_messages is within range."""
        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "test", "num_messages": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/drafts/summarize",
            json={"chat_id": "test", "num_messages": 1000},
        )
        assert response.status_code == 422


class TestDraftSchemas:
    """Tests for draft-related Pydantic schemas."""

    def test_draft_reply_request_defaults(self):
        """DraftReplyRequest has correct defaults."""
        from api.schemas import DraftReplyRequest

        request = DraftReplyRequest(chat_id="test")
        assert request.num_suggestions == 3
        assert request.context_messages == 20
        assert request.instruction is None

    def test_draft_reply_request_custom_values(self):
        """DraftReplyRequest accepts custom values."""
        from api.schemas import DraftReplyRequest

        request = DraftReplyRequest(
            chat_id="test-chat",
            num_suggestions=5,
            context_messages=50,
            instruction="be formal",
        )
        assert request.chat_id == "test-chat"
        assert request.num_suggestions == 5
        assert request.context_messages == 50
        assert request.instruction == "be formal"

    def test_draft_suggestion_model(self):
        """DraftSuggestion model works correctly."""
        from api.schemas import DraftSuggestion

        suggestion = DraftSuggestion(text="Hello!", confidence=0.9)
        assert suggestion.text == "Hello!"
        assert suggestion.confidence == 0.9

    def test_draft_suggestion_validates_confidence(self):
        """DraftSuggestion validates confidence range."""
        from pydantic import ValidationError

        from api.schemas import DraftSuggestion

        with pytest.raises(ValidationError):
            DraftSuggestion(text="Hi", confidence=1.5)

        with pytest.raises(ValidationError):
            DraftSuggestion(text="Hi", confidence=-0.1)

    def test_context_info_model(self):
        """ContextInfo model works correctly."""
        from api.schemas import ContextInfo

        info = ContextInfo(
            num_messages=20,
            participants=["John", "Jane"],
            last_message="Hello",
        )
        assert info.num_messages == 20
        assert info.participants == ["John", "Jane"]
        assert info.last_message == "Hello"

    def test_draft_reply_response_model(self):
        """DraftReplyResponse model works correctly."""
        from api.schemas import ContextInfo, DraftReplyResponse, DraftSuggestion

        response = DraftReplyResponse(
            suggestions=[
                DraftSuggestion(text="Yes!", confidence=0.9),
                DraftSuggestion(text="Sure!", confidence=0.8),
            ],
            context_used=ContextInfo(
                num_messages=10,
                participants=["Alice"],
                last_message="Coming?",
            ),
        )
        assert len(response.suggestions) == 2
        assert response.context_used.num_messages == 10

    def test_draft_summary_request_defaults(self):
        """DraftSummaryRequest has correct defaults."""
        from api.schemas import DraftSummaryRequest

        request = DraftSummaryRequest(chat_id="test")
        assert request.num_messages == 50

    def test_date_range_model(self):
        """DateRange model works correctly."""
        from api.schemas import DateRange

        date_range = DateRange(start="2024-01-01", end="2024-01-31")
        assert date_range.start == "2024-01-01"
        assert date_range.end == "2024-01-31"

    def test_draft_summary_response_model(self):
        """DraftSummaryResponse model works correctly."""
        from api.schemas import DateRange, DraftSummaryResponse

        response = DraftSummaryResponse(
            summary="A discussion about plans.",
            key_points=["Meeting tomorrow", "Bring snacks"],
            date_range=DateRange(start="2024-01-20", end="2024-01-25"),
        )
        assert "discussion" in response.summary
        assert len(response.key_points) == 2
        assert response.date_range.start == "2024-01-20"


class TestRouterRegistration:
    """Tests for router registration in the app."""

    def test_drafts_router_is_registered(self, client, mock_reader):
        """Verify drafts router is registered with correct prefix."""
        # Check that the endpoint exists by making a request
        # Even if it fails auth, it should return 4xx not 404
        response = client.post("/drafts/reply", json={"chat_id": "test"})
        # Should not be 404 - endpoint exists but may fail for other reasons
        assert response.status_code != 404

    def test_drafts_summarize_endpoint_exists(self, client, mock_reader):
        """Verify summarize endpoint exists."""
        response = client.post("/drafts/summarize", json={"chat_id": "test"})
        assert response.status_code != 404
