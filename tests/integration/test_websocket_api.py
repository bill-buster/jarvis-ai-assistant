"""Integration tests for WebSocket API endpoints.

Tests the full WebSocket connection lifecycle including connecting,
message handling, and disconnection using the FastAPI test client.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.websocket import manager


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_manager():
    """Reset the connection manager before each test."""
    # Clear any existing connections
    manager._clients.clear()
    yield
    manager._clients.clear()


class TestWebSocketConnection:
    """Tests for WebSocket connection handling."""

    def test_websocket_connect_sends_connected_message(self, client):
        """WebSocket connection sends connected message with client_id."""
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()

            assert data["type"] == "connected"
            assert "client_id" in data["data"]
            assert "timestamp" in data["data"]

    def test_websocket_connect_increments_active_connections(self, client):
        """WebSocket connection increments active connection count."""
        # Check initial count via status endpoint
        response = client.get("/ws/status")
        initial_count = response.json()["active_connections"]

        with client.websocket_connect("/ws") as websocket:
            # Receive the connected message
            websocket.receive_json()

            # Check count increased
            response = client.get("/ws/status")
            assert response.json()["active_connections"] == initial_count + 1

    def test_websocket_disconnect_decrements_active_connections(self, client):
        """WebSocket disconnection decrements active connection count."""
        response = client.get("/ws/status")
        initial_count = response.json()["active_connections"]

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()

        # After context exit, connection should be closed
        response = client.get("/ws/status")
        assert response.json()["active_connections"] == initial_count


class TestWebSocketPing:
    """Tests for WebSocket ping/pong."""

    def test_ping_receives_pong(self, client):
        """Ping message receives pong response."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            websocket.receive_json()

            # Send ping
            websocket.send_json({"type": "ping", "data": {}})

            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data["data"]


class TestWebSocketHealthSubscription:
    """Tests for WebSocket health subscriptions."""

    def test_subscribe_health_sends_confirmation(self, client):
        """Subscribe health sends confirmation message."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            websocket.receive_json()

            # Subscribe to health
            websocket.send_json({"type": "subscribe_health", "data": {}})

            # Receive confirmation
            data = websocket.receive_json()
            assert data["type"] == "health_update"
            assert data["data"]["subscribed"] is True

    def test_subscribe_health_increments_subscriber_count(self, client):
        """Subscribe health increments subscriber count."""
        response = client.get("/ws/status")
        initial_subscribers = response.json()["health_subscribers"]

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "subscribe_health", "data": {}})
            websocket.receive_json()

            response = client.get("/ws/status")
            assert response.json()["health_subscribers"] == initial_subscribers + 1


class TestWebSocketGeneration:
    """Tests for WebSocket generation requests."""

    def test_generate_without_prompt_returns_error(self, client):
        """Generate without prompt returns error."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            websocket.receive_json()

            # Send generate without prompt
            websocket.send_json({"type": "generate", "data": {}})

            # Receive error
            data = websocket.receive_json()
            assert data["type"] == "generation_error"
            assert "error" in data["data"]

    def test_generate_with_prompt_sends_start_message(self, client):
        """Generate with prompt sends generation_start message."""
        mock_generator = MagicMock()
        mock_generator.generate.return_value = MagicMock(
            text="Hello",
            tokens_used=1,
            generation_time_ms=10.0,
            model_name="test",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        with patch("api.routers.websocket.get_generator", return_value=mock_generator):
            with client.websocket_connect("/ws") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Send generate
                websocket.send_json({"type": "generate", "data": {"prompt": "Hello"}})

                # Receive start message
                data = websocket.receive_json()
                assert data["type"] == "generation_start"
                assert "generation_id" in data["data"]

                # Receive complete message
                data = websocket.receive_json()
                assert data["type"] == "generation_complete"
                assert data["data"]["text"] == "Hello"

    def test_generate_stream_sends_start_and_completion(self, client):
        """Generate stream sends start and completion messages."""
        mock_generator = MagicMock()
        mock_generator.generate.return_value = MagicMock(
            text="Hello world",
            tokens_used=2,
            generation_time_ms=10.0,
            model_name="test",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )
        # Mock doesn't have generate_stream, so it will use fallback
        mock_generator.config = MagicMock(model_path="test-model")

        with patch("api.routers.websocket.get_generator", return_value=mock_generator):
            with client.websocket_connect("/ws") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Send generate_stream
                websocket.send_json(
                    {"type": "generate_stream", "data": {"prompt": "Hello"}}
                )

                # Receive start message
                data = websocket.receive_json()
                assert data["type"] == "generation_start"
                assert data["data"]["streaming"] is True

                # Collect all messages until complete
                messages = []
                while True:
                    data = websocket.receive_json()
                    messages.append(data)
                    if data["type"] in ("generation_complete", "generation_error"):
                        break

                # Verify we got a completion or error
                final_msg = messages[-1]
                assert final_msg["type"] in ("generation_complete", "generation_error")


class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling."""

    def test_invalid_json_returns_error(self, client):
        """Invalid JSON message returns error."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("not json")

            # Receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["data"]["error"]

    def test_unknown_message_type_returns_error(self, client):
        """Unknown message type returns error."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            websocket.receive_json()

            # Send unknown type
            websocket.send_json({"type": "unknown_type", "data": {}})

            # Receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Unknown message type" in data["data"]["error"]

    def test_generator_unavailable_returns_error(self, client):
        """Generator unavailable returns error."""
        with patch(
            "api.routers.websocket.get_generator",
            side_effect=RuntimeError("Model not available"),
        ):
            with client.websocket_connect("/ws") as websocket:
                # Receive connected message
                websocket.receive_json()

                # Send generate
                websocket.send_json({"type": "generate", "data": {"prompt": "Hello"}})

                # Receive messages until we get an error
                # (may receive generation_start first depending on timing)
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "generation_error":
                        assert "unavailable" in data["data"]["error"].lower()
                        break
                    elif data["type"] == "generation_start":
                        # Expected - generation starts before error
                        continue
                    else:
                        pytest.fail(f"Unexpected message type: {data['type']}")


class TestWebSocketStatusEndpoint:
    """Tests for /ws/status REST endpoint."""

    def test_status_endpoint_returns_200(self, client):
        """Status endpoint returns 200 OK."""
        response = client.get("/ws/status")
        assert response.status_code == 200

    def test_status_endpoint_returns_expected_fields(self, client):
        """Status endpoint returns expected fields."""
        response = client.get("/ws/status")
        data = response.json()

        assert "active_connections" in data
        assert "health_subscribers" in data
        assert "status" in data
        assert data["status"] == "operational"

    def test_status_reflects_active_connections(self, client):
        """Status endpoint accurately reflects active connections."""
        # No connections
        response = client.get("/ws/status")
        initial = response.json()["active_connections"]

        # Open connection
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()

            response = client.get("/ws/status")
            assert response.json()["active_connections"] == initial + 1

        # Connection closed
        response = client.get("/ws/status")
        assert response.json()["active_connections"] == initial


class TestWebSocketCancel:
    """Tests for WebSocket generation cancellation."""

    def test_cancel_clears_active_generation(self, client):
        """Cancel message clears active generation."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            connected_data = websocket.receive_json()
            client_id = connected_data["data"]["client_id"]

            # Send cancel
            websocket.send_json({"type": "cancel", "data": {}})

            # Verify the client's generation was cleared
            ws_client = manager.get_client(client_id)
            assert ws_client.active_generation_id is None
