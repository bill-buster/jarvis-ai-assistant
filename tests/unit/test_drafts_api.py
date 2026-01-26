"""Unit tests for the drafts API endpoint.

Tests the API routes for draft generation with timeout and fallback handling.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestDraftReplyEndpoint:
    """Tests for POST /drafts/reply endpoint."""

    def test_returns_suggestions(self):
        """Returns suggestions successfully."""
        with patch("api.routers.drafts.generate_reply_suggestions") as mock_gen:
            mock_gen.return_value = [
                ("Sounds good!", 0.9),
                ("Got it!", 0.8),
            ]

            response = client.post(
                "/drafts/reply",
                json={
                    "chat_id": "test-chat",
                    "last_message": "Are you free tomorrow?",
                    "num_suggestions": 3,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 2
        assert data["suggestions"][0]["text"] == "Sounds good!"
        assert data["suggestions"][0]["confidence"] == 0.9

    def test_handles_fallback(self):
        """Returns fallback suggestions when generation fails."""
        with patch("api.routers.drafts.generate_reply_suggestions") as mock_gen:
            # Simulate fallback (all confidence 0.5)
            mock_gen.return_value = [
                ("Sounds good!", 0.5),
                ("Got it, thanks!", 0.5),
            ]

            response = client.post(
                "/drafts/reply",
                json={
                    "chat_id": "test-chat",
                    "last_message": "Hello",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True

    def test_handles_timeout(self):
        """Returns fallback on timeout."""

        with patch("api.routers.drafts._generate_replies_sync") as mock_gen:
            # Simulate timeout by having the function take too long
            def slow_function(*args, **kwargs):
                import time

                time.sleep(0.1)
                return [("Result", 0.9)]

            mock_gen.side_effect = slow_function

            # Patch the timeout to be very short
            with patch("api.routers.drafts.GENERATION_TIMEOUT_SECONDS", 0.01):
                response = client.post(
                    "/drafts/reply",
                    json={
                        "chat_id": "test-chat",
                        "last_message": "Test",
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True
        assert data["error"] is not None

    def test_handles_model_load_error(self):
        """Returns fallback on model load failure."""
        from jarvis.fallbacks import ModelLoadError

        with patch("api.routers.drafts._generate_replies_sync") as mock_gen:
            mock_gen.side_effect = ModelLoadError("Cannot load model")

            response = client.post(
                "/drafts/reply",
                json={
                    "chat_id": "test-chat",
                    "last_message": "Test",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True
        assert data["error"] is not None
        assert "model" in data["error"].lower()

    def test_handles_unexpected_error(self):
        """Returns fallback on unexpected errors."""
        with patch("api.routers.drafts._generate_replies_sync") as mock_gen:
            mock_gen.side_effect = RuntimeError("Unexpected error")

            response = client.post(
                "/drafts/reply",
                json={
                    "chat_id": "test-chat",
                    "last_message": "Test",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True
        assert "error" in data["error"].lower() or "unexpected" in data["error"].lower()

    def test_validates_request(self):
        """Validates required fields in request."""
        # Missing chat_id
        response = client.post(
            "/drafts/reply",
            json={
                "last_message": "Test",
            },
        )
        assert response.status_code == 422

        # Missing last_message
        response = client.post(
            "/drafts/reply",
            json={
                "chat_id": "test-chat",
            },
        )
        assert response.status_code == 422

        # Empty last_message
        response = client.post(
            "/drafts/reply",
            json={
                "chat_id": "test-chat",
                "last_message": "",
            },
        )
        assert response.status_code == 422


class TestSummaryEndpoint:
    """Tests for POST /drafts/summary endpoint."""

    def test_returns_summary(self):
        """Returns summary successfully."""
        with patch("api.routers.drafts.generate_summary") as mock_gen:
            mock_gen.return_value = ("This is a summary", False)

            response = client.post(
                "/drafts/summary",
                json={
                    "chat_id": "test-chat-id",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "This is a summary"
        assert data["used_fallback"] is False

    def test_handles_fallback(self):
        """Returns fallback summary when generation fails."""
        with patch("api.routers.drafts.generate_summary") as mock_gen:
            mock_gen.return_value = ("Unable to generate summary", True)

            response = client.post(
                "/drafts/summary",
                json={
                    "chat_id": "test-chat",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True

    def test_handles_timeout(self):
        """Returns fallback on timeout."""
        with patch("api.routers.drafts._generate_summary_sync") as mock_gen:

            def slow_function(*args, **kwargs):
                import time

                time.sleep(0.1)
                return ("Summary", False)

            mock_gen.side_effect = slow_function

            with patch("api.routers.drafts.GENERATION_TIMEOUT_SECONDS", 0.01):
                response = client.post(
                    "/drafts/summary",
                    json={
                        "chat_id": "test-chat",
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fallback"] is True
        assert data["error"] is not None

    def test_validates_request(self):
        """Validates required fields in request."""
        # Missing chat_id
        response = client.post(
            "/drafts/summary",
            json={},
        )
        assert response.status_code == 422


class TestStatusEndpoint:
    """Tests for GET /drafts/status endpoint."""

    def test_returns_status(self):
        """Returns generation system status."""
        with patch("api.routers.drafts.get_generation_status") as mock_status:
            mock_status.return_value = {
                "model_loaded": True,
                "can_generate": True,
                "reason": None,
                "memory_mode": "full",
            }

            response = client.get("/drafts/status")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["can_generate"] is True
        assert data["memory_mode"] == "full"

    def test_returns_reason_when_cannot_generate(self):
        """Includes reason when generation is disabled."""
        with patch("api.routers.drafts.get_generation_status") as mock_status:
            mock_status.return_value = {
                "model_loaded": False,
                "can_generate": False,
                "reason": "Memory pressure critical",
                "memory_mode": "minimal",
            }

            response = client.get("/drafts/status")

        assert response.status_code == 200
        data = response.json()
        assert data["can_generate"] is False
        assert data["reason"] == "Memory pressure critical"


class TestDraftsResponseSchema:
    """Tests for response schema validation."""

    def test_draft_reply_response_has_all_fields(self):
        """DraftReplyResponse includes all expected fields."""
        with patch("api.routers.drafts.generate_reply_suggestions") as mock_gen:
            mock_gen.return_value = [("Test", 0.9)]

            response = client.post(
                "/drafts/reply",
                json={
                    "chat_id": "test",
                    "last_message": "Hello",
                },
            )

        data = response.json()
        assert "suggestions" in data
        assert "context_used" in data
        assert "error" in data
        assert "used_fallback" in data

    def test_summary_response_has_all_fields(self):
        """SummaryResponse includes all expected fields."""
        with patch("api.routers.drafts.generate_summary") as mock_gen:
            mock_gen.return_value = ("Summary text", False)

            response = client.post(
                "/drafts/summary",
                json={
                    "chat_id": "test",
                },
            )

        data = response.json()
        assert "summary" in data
        assert "participant" in data
        assert "message_count" in data
        assert "error" in data
        assert "used_fallback" in data

    def test_status_response_has_all_fields(self):
        """GenerationStatus includes all expected fields."""
        with patch("api.routers.drafts.get_generation_status") as mock_status:
            mock_status.return_value = {
                "model_loaded": True,
                "can_generate": True,
                "reason": None,
                "memory_mode": "full",
            }

            response = client.get("/drafts/status")

        data = response.json()
        assert "model_loaded" in data
        assert "can_generate" in data
        assert "reason" in data
        assert "memory_mode" in data
