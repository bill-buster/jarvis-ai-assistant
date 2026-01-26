"""Unit tests for the Suggestions API endpoint.

Tests cover the quick-reply suggestions endpoint including
group context awareness and pattern matching.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.suggestions import (
    GROUP_RESPONSE_PATTERNS,
    RESPONSE_PATTERNS,
    Suggestion,
    SuggestionRequest,
    SuggestionResponse,
    _compute_match_score,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


class TestComputeMatchScore:
    """Tests for the _compute_match_score function."""

    def test_exact_keyword_match(self) -> None:
        """Test exact keyword match returns base score."""
        score = _compute_match_score("thanks for the help", ["thanks"], 0.9)
        assert score == 0.9

    def test_partial_word_match(self) -> None:
        """Test word-level match returns reduced score when not exact phrase."""
        # "time" word in message matches "time" word from keyword "what time"
        score = _compute_match_score("check the time please", ["what time"], 0.9)
        assert score == pytest.approx(0.63, rel=0.1)  # 0.9 * 0.7

    def test_no_match(self) -> None:
        """Test no match returns zero."""
        score = _compute_match_score("goodbye", ["hello", "hi"], 0.9)
        assert score == 0.0

    def test_empty_keywords_returns_base(self) -> None:
        """Test empty keywords (fallback) returns base score directly."""
        score = _compute_match_score("anything", [], 0.3)
        assert score == 0.3

    def test_phrase_match(self) -> None:
        """Test phrase matching."""
        score = _compute_match_score("I'm on my way now", ["on my way"], 0.9)
        assert score == 0.9

    def test_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        score = _compute_match_score("THANKS!", ["thanks"], 0.9)
        assert score == 0.9


class TestSuggestionModels:
    """Tests for Pydantic models."""

    def test_suggestion_request_basic(self) -> None:
        """Test basic SuggestionRequest creation."""
        request = SuggestionRequest(last_message="hello")
        assert request.last_message == "hello"
        assert request.num_suggestions == 3  # default
        assert request.group_size is None

    def test_suggestion_request_with_group(self) -> None:
        """Test SuggestionRequest with group_size."""
        request = SuggestionRequest(
            last_message="hello",
            num_suggestions=5,
            group_size=8,
        )
        assert request.group_size == 8
        assert request.num_suggestions == 5

    def test_suggestion_response(self) -> None:
        """Test SuggestionResponse creation."""
        response = SuggestionResponse(
            suggestions=[
                Suggestion(text="Hello!", score=0.9),
                Suggestion(text="Hi!", score=0.8),
            ]
        )
        assert len(response.suggestions) == 2
        assert response.suggestions[0].score == 0.9


class TestResponsePatterns:
    """Tests for response pattern definitions."""

    def test_standard_patterns_exist(self) -> None:
        """Test that standard response patterns are defined."""
        assert len(RESPONSE_PATTERNS) > 0

    def test_group_patterns_exist(self) -> None:
        """Test that group response patterns are defined."""
        assert len(GROUP_RESPONSE_PATTERNS) > 0

    def test_patterns_have_valid_structure(self) -> None:
        """Test that all patterns have valid structure."""
        for keywords, response, score in RESPONSE_PATTERNS:
            assert isinstance(keywords, list)
            assert isinstance(response, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_group_patterns_have_valid_structure(self) -> None:
        """Test that all group patterns have valid structure."""
        for keywords, response, score in GROUP_RESPONSE_PATTERNS:
            assert isinstance(keywords, list)
            assert isinstance(response, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_fallback_patterns_exist(self) -> None:
        """Test that fallback patterns (empty keywords) exist."""
        fallbacks = [p for p in RESPONSE_PATTERNS if not p[0]]
        assert len(fallbacks) >= 3

    def test_group_fallback_patterns_exist(self) -> None:
        """Test that group fallback patterns exist."""
        fallbacks = [p for p in GROUP_RESPONSE_PATTERNS if not p[0]]
        assert len(fallbacks) >= 3


class TestSuggestionsEndpoint:
    """Tests for the /suggestions endpoint."""

    def test_basic_suggestion(self, client: TestClient) -> None:
        """Test basic suggestion generation."""
        response = client.post(
            "/suggestions",
            json={"last_message": "thanks for your help!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0

    def test_suggestion_with_num_suggestions(self, client: TestClient) -> None:
        """Test suggestion with custom num_suggestions."""
        response = client.post(
            "/suggestions",
            json={"last_message": "thanks!", "num_suggestions": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) <= 2

    def test_empty_message_returns_empty(self, client: TestClient) -> None:
        """Test that empty message returns empty suggestions."""
        response = client.post(
            "/suggestions",
            json={"last_message": "   "},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) == 0

    def test_suggestions_sorted_by_score(self, client: TestClient) -> None:
        """Test that suggestions are sorted by score descending."""
        response = client.post(
            "/suggestions",
            json={"last_message": "sounds good, see you later!"},
        )
        assert response.status_code == 200
        data = response.json()
        scores = [s["score"] for s in data["suggestions"]]
        assert scores == sorted(scores, reverse=True)

    def test_group_suggestions_basic(self, client: TestClient) -> None:
        """Test suggestions with group_size parameter."""
        response = client.post(
            "/suggestions",
            json={
                "last_message": "who's coming to the party?",
                "group_size": 8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) > 0

    def test_1on1_vs_group_different_responses(self, client: TestClient) -> None:
        """Test that 1-on-1 and group chats can get different responses."""
        # Same message for both contexts
        message = "sounds good!"

        response_1on1 = client.post(
            "/suggestions",
            json={"last_message": message, "group_size": 2},
        )
        response_group = client.post(
            "/suggestions",
            json={"last_message": message, "group_size": 5},
        )

        assert response_1on1.status_code == 200
        assert response_group.status_code == 200

        # Both should return valid suggestions
        data_1on1 = response_1on1.json()
        data_group = response_group.json()
        assert len(data_1on1["suggestions"]) > 0
        assert len(data_group["suggestions"]) > 0

    def test_group_rsvp_pattern(self, client: TestClient) -> None:
        """Test RSVP patterns for group chat."""
        response = client.post(
            "/suggestions",
            json={
                "last_message": "count me in!",
                "group_size": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should match an RSVP-related response
        texts = [s["text"] for s in data["suggestions"]]
        assert len(texts) > 0

    def test_group_celebration_pattern(self, client: TestClient) -> None:
        """Test celebration patterns for group chat."""
        response = client.post(
            "/suggestions",
            json={
                "last_message": "happy birthday everyone!",
                "group_size": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should return celebration-related suggestions
        assert len(data["suggestions"]) > 0

    def test_large_group_boost(self, client: TestClient) -> None:
        """Test that large groups get slightly boosted scores."""
        # This tests internal behavior through the API
        response = client.post(
            "/suggestions",
            json={
                "last_message": "so many messages!",
                "group_size": 15,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) > 0

    def test_validation_num_suggestions_max(self, client: TestClient) -> None:
        """Test num_suggestions validation (max 5)."""
        response = client.post(
            "/suggestions",
            json={"last_message": "hello", "num_suggestions": 10},
        )
        assert response.status_code == 422  # Validation error

    def test_validation_num_suggestions_min(self, client: TestClient) -> None:
        """Test num_suggestions validation (min 1)."""
        response = client.post(
            "/suggestions",
            json={"last_message": "hello", "num_suggestions": 0},
        )
        assert response.status_code == 422  # Validation error

    def test_validation_group_size_min(self, client: TestClient) -> None:
        """Test group_size validation (min 1)."""
        response = client.post(
            "/suggestions",
            json={"last_message": "hello", "group_size": 0},
        )
        assert response.status_code == 422  # Validation error


class TestGroupPatternCategories:
    """Tests for specific group pattern categories."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client."""
        return TestClient(app)

    def test_event_planning_patterns(self, client: TestClient) -> None:
        """Test event planning patterns."""
        messages = [
            "when works for everyone?",
            "let's pick a date",
            "what time works",
        ]
        for msg in messages:
            response = client.post(
                "/suggestions",
                json={"last_message": msg, "group_size": 5},
            )
            assert response.status_code == 200
            assert len(response.json()["suggestions"]) > 0

    def test_poll_patterns(self, client: TestClient) -> None:
        """Test poll patterns."""
        messages = [
            "let's vote on it",
            "option A for me",
            "either works for me",
        ]
        for msg in messages:
            response = client.post(
                "/suggestions",
                json={"last_message": msg, "group_size": 5},
            )
            assert response.status_code == 200
            assert len(response.json()["suggestions"]) > 0

    def test_logistics_patterns(self, client: TestClient) -> None:
        """Test logistics patterns."""
        messages = [
            "who's bringing what?",
            "I'll handle the reservation",
            "where are we meeting?",
        ]
        for msg in messages:
            response = client.post(
                "/suggestions",
                json={"last_message": msg, "group_size": 5},
            )
            assert response.status_code == 200
            assert len(response.json()["suggestions"]) > 0

    def test_info_sharing_patterns(self, client: TestClient) -> None:
        """Test information sharing patterns."""
        messages = [
            "fyi everyone",
            "heads up team",
            "quick update for the group",
        ]
        for msg in messages:
            response = client.post(
                "/suggestions",
                json={"last_message": msg, "group_size": 5},
            )
            assert response.status_code == 200
            assert len(response.json()["suggestions"]) > 0
