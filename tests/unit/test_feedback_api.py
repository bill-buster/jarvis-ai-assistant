"""Tests for the feedback API endpoints.

Tests feedback recording, statistics, and improvement suggestions.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from jarvis.evaluation import EvaluationResult, FeedbackStore, ResponseEvaluator, reset_evaluation


@pytest.fixture(autouse=True)
def reset_before_each():
    """Reset evaluation state before each test."""
    reset_evaluation()
    yield
    reset_evaluation()


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_feedback_store():
    """Create a temporary feedback store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FeedbackStore(feedback_dir=Path(tmpdir))
        with patch("api.routers.feedback.get_feedback_store", return_value=store):
            yield store


@pytest.fixture
def mock_evaluator():
    """Mock the response evaluator to avoid heavy imports."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    evaluator = ResponseEvaluator()
    evaluator._sentence_model = mock_model

    with patch("api.routers.feedback.get_response_evaluator", return_value=evaluator):
        with patch.object(evaluator, "_get_sentence_model", return_value=mock_model):
            yield evaluator


class TestRecordFeedback:
    """Tests for POST /feedback/response endpoint."""

    def test_record_feedback_sent_returns_200(self, client, mock_feedback_store):
        """Recording sent feedback returns 200 OK."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Hello there!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        assert response.status_code == 200

    def test_record_feedback_returns_suggestion_id(self, client, mock_feedback_store):
        """Recording feedback returns suggestion ID."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Hello there!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        data = response.json()

        assert data["success"] is True
        assert "suggestion_id" in data
        assert len(data["suggestion_id"]) == 16

    def test_record_feedback_edited_requires_edited_text(self, client, mock_feedback_store):
        """Recording edited feedback requires edited_text."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "edited",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
                # Missing edited_text
            },
        )
        assert response.status_code == 400

    def test_record_feedback_edited_with_text_succeeds(self, client, mock_feedback_store):
        """Recording edited feedback with text succeeds."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "edited",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
                "edited_text": "Hi there!",
            },
        )
        assert response.status_code == 200

    def test_record_feedback_invalid_action_returns_400(self, client, mock_feedback_store):
        """Recording with invalid action returns 400."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "invalid_action",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        assert response.status_code == 400

    def test_record_feedback_dismissed(self, client, mock_feedback_store):
        """Recording dismissed feedback succeeds."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "dismissed",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        assert response.status_code == 200

    def test_record_feedback_copied(self, client, mock_feedback_store):
        """Recording copied feedback succeeds."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "copied",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        assert response.status_code == 200

    def test_record_feedback_with_evaluation(self, client, mock_feedback_store):
        """Recording feedback computes evaluation scores."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "That sounds great!",
                "chat_id": "chat123",
                "context_messages": ["Hey, want to grab lunch?", "Are you free?"],
                "include_evaluation": True,
            },
        )
        data = response.json()

        assert data["success"] is True
        # Evaluation may be present if context is provided
        if data["evaluation"]:
            assert "tone_score" in data["evaluation"]
            assert "relevance_score" in data["evaluation"]
            assert "naturalness_score" in data["evaluation"]
            assert "length_score" in data["evaluation"]
            assert "overall_score" in data["evaluation"]

    def test_record_feedback_without_evaluation(self, client, mock_feedback_store):
        """Recording feedback without evaluation skips scores."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": [],
                "include_evaluation": False,
            },
        )
        data = response.json()

        assert data["success"] is True
        assert data["evaluation"] is None

    def test_record_feedback_with_metadata(self, client, mock_feedback_store):
        """Recording feedback with metadata succeeds."""
        response = client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
                "metadata": {"source": "test", "version": 1},
            },
        )
        assert response.status_code == 200


class TestGetFeedbackStats:
    """Tests for GET /feedback/stats endpoint."""

    def test_get_stats_returns_200(self, client, mock_feedback_store):
        """Stats endpoint returns 200 OK."""
        response = client.get("/feedback/stats")
        assert response.status_code == 200

    def test_get_stats_returns_json(self, client, mock_feedback_store):
        """Stats endpoint returns JSON."""
        response = client.get("/feedback/stats")
        assert "application/json" in response.headers["content-type"]

    def test_get_stats_empty_store(self, client, mock_feedback_store):
        """Stats on empty store returns zeros."""
        response = client.get("/feedback/stats")
        data = response.json()

        assert data["total_feedback"] == 0
        assert data["sent_unchanged"] == 0
        assert data["edited"] == 0
        assert data["dismissed"] == 0
        assert data["acceptance_rate"] == 0.0

    def test_get_stats_after_feedback(self, client, mock_feedback_store):
        """Stats reflect recorded feedback."""
        # Record some feedback
        client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        client.post(
            "/feedback/response",
            json={
                "action": "edited",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
                "edited_text": "Hi there!",
            },
        )

        response = client.get("/feedback/stats")
        data = response.json()

        assert data["total_feedback"] == 2
        assert data["sent_unchanged"] == 1
        assert data["edited"] == 1

    def test_get_stats_contains_acceptance_rate(self, client, mock_feedback_store):
        """Stats contains acceptance rate."""
        response = client.get("/feedback/stats")
        data = response.json()

        assert "acceptance_rate" in data
        assert "edit_rate" in data


class TestGetImprovements:
    """Tests for GET /feedback/improvements endpoint."""

    def test_get_improvements_returns_200(self, client, mock_feedback_store):
        """Improvements endpoint returns 200 OK."""
        response = client.get("/feedback/improvements")
        assert response.status_code == 200

    def test_get_improvements_returns_json(self, client, mock_feedback_store):
        """Improvements endpoint returns JSON."""
        response = client.get("/feedback/improvements")
        assert "application/json" in response.headers["content-type"]

    def test_get_improvements_empty_store(self, client, mock_feedback_store):
        """Improvements on empty store returns empty list."""
        response = client.get("/feedback/improvements")
        data = response.json()

        assert data["improvements"] == []
        assert data["based_on_entries"] == 0

    def test_get_improvements_with_limit(self, client, mock_feedback_store):
        """Improvements respects limit parameter."""
        response = client.get("/feedback/improvements?limit=5")
        data = response.json()

        assert len(data["improvements"]) <= 5

    def test_get_improvements_structure(self, client, mock_feedback_store):
        """Improvements have correct structure."""
        # Add some edited feedback
        for _ in range(5):
            client.post(
                "/feedback/response",
                json={
                    "action": "edited",
                    "suggestion_text": "This is a very long message that goes on and on.",
                    "chat_id": "chat123",
                    "context_messages": ["Hi"],
                    "edited_text": "Short.",
                },
            )

        response = client.get("/feedback/improvements")
        data = response.json()

        if data["improvements"]:
            improvement = data["improvements"][0]
            assert "type" in improvement
            assert "suggestion" in improvement
            assert "detail" in improvement
            assert "confidence" in improvement


class TestGetRecentFeedback:
    """Tests for GET /feedback/recent endpoint."""

    def test_get_recent_returns_200(self, client, mock_feedback_store):
        """Recent endpoint returns 200 OK."""
        response = client.get("/feedback/recent")
        assert response.status_code == 200

    def test_get_recent_returns_json(self, client, mock_feedback_store):
        """Recent endpoint returns JSON."""
        response = client.get("/feedback/recent")
        assert "application/json" in response.headers["content-type"]

    def test_get_recent_empty_store(self, client, mock_feedback_store):
        """Recent on empty store returns empty list."""
        response = client.get("/feedback/recent")
        data = response.json()

        assert data["entries"] == []
        assert data["total_count"] == 0

    def test_get_recent_with_limit(self, client, mock_feedback_store):
        """Recent respects limit parameter."""
        # Add some feedback
        for i in range(10):
            client.post(
                "/feedback/response",
                json={
                    "action": "sent",
                    "suggestion_text": f"Message {i}",
                    "chat_id": "chat123",
                    "context_messages": ["Hi"],
                },
            )

        response = client.get("/feedback/recent?limit=5")
        data = response.json()

        assert len(data["entries"]) == 5

    def test_get_recent_order(self, client, mock_feedback_store):
        """Recent returns entries in reverse chronological order."""
        # Add feedback
        client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "First",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )
        client.post(
            "/feedback/response",
            json={
                "action": "sent",
                "suggestion_text": "Second",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
            },
        )

        response = client.get("/feedback/recent")
        data = response.json()

        # Most recent should be first
        assert data["entries"][0]["suggestion_text"] == "Second"

    def test_get_recent_entry_structure(self, client, mock_feedback_store):
        """Recent entries have correct structure."""
        client.post(
            "/feedback/response",
            json={
                "action": "edited",
                "suggestion_text": "Hello!",
                "chat_id": "chat123",
                "context_messages": ["Hi"],
                "edited_text": "Hi there!",
            },
        )

        response = client.get("/feedback/recent")
        data = response.json()

        entry = data["entries"][0]
        assert "timestamp" in entry
        assert "action" in entry
        assert "suggestion_id" in entry
        assert "suggestion_text" in entry
        assert "chat_id" in entry
        assert entry["action"] == "edited"
        assert entry["edited_text"] == "Hi there!"


class TestEvaluateResponse:
    """Tests for POST /feedback/evaluate endpoint."""

    def test_evaluate_returns_200(self, client, mock_evaluator):
        """Evaluate endpoint returns 200 OK."""
        response = client.post(
            "/feedback/evaluate",
            json={
                "response": "Sounds great!",
                "context_messages": ["Hey, want to grab lunch?"],
            },
        )
        assert response.status_code == 200

    def test_evaluate_returns_scores(self, client, mock_evaluator):
        """Evaluate endpoint returns all scores."""
        response = client.post(
            "/feedback/evaluate",
            json={
                "response": "Yes, I'd love to! What time?",
                "context_messages": ["Hey, want to grab lunch?", "Are you free today?"],
            },
        )
        data = response.json()

        assert "tone_score" in data
        assert "relevance_score" in data
        assert "naturalness_score" in data
        assert "length_score" in data
        assert "overall_score" in data

    def test_evaluate_scores_in_range(self, client, mock_evaluator):
        """Evaluate scores are in valid range."""
        response = client.post(
            "/feedback/evaluate",
            json={
                "response": "Sounds good!",
                "context_messages": ["Shall we meet tomorrow?"],
            },
        )
        data = response.json()

        assert 0 <= data["tone_score"] <= 1
        assert 0 <= data["relevance_score"] <= 1
        assert 0 <= data["naturalness_score"] <= 1
        assert 0 <= data["length_score"] <= 1
        assert 0 <= data["overall_score"] <= 1

    def test_evaluate_with_user_messages(self, client, mock_evaluator):
        """Evaluate uses user messages for length scoring."""
        response = client.post(
            "/feedback/evaluate",
            json={
                "response": "Ok!",
                "context_messages": ["Are you coming?"],
                "user_messages": ["Yes", "Sure", "Ok"],
            },
        )
        data = response.json()

        # Short response should score well with short user messages
        assert data["length_score"] > 0.5

    def test_evaluate_requires_context(self, client, mock_evaluator):
        """Evaluate requires at least one context message."""
        response = client.post(
            "/feedback/evaluate",
            json={
                "response": "Hello!",
                "context_messages": [],
            },
        )
        # Should fail validation (min_length=1)
        assert response.status_code == 422
