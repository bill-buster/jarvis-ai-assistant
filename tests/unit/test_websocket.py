"""Unit tests for WebSocket router and connection manager.

Tests WebSocket connection handling, message routing, streaming generation,
and health subscriptions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.routers.websocket import (
    ConnectionManager,
    MessageType,
    WebSocketClient,
    get_connection_manager,
    manager,
)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def connection_manager():
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_connect_accepts_websocket(self, connection_manager, mock_websocket):
        """Connect accepts websocket and returns client."""
        client = run_async(connection_manager.connect(mock_websocket))

        mock_websocket.accept.assert_called_once()
        assert client is not None
        assert client.client_id is not None
        assert client.websocket is mock_websocket

    def test_connect_increments_connection_count(
        self, connection_manager, mock_websocket
    ):
        """Connection count increases when client connects."""
        assert connection_manager.active_connections == 0

        run_async(connection_manager.connect(mock_websocket))

        assert connection_manager.active_connections == 1

    def test_disconnect_removes_client(self, connection_manager, mock_websocket):
        """Disconnect removes client from manager."""
        client = run_async(connection_manager.connect(mock_websocket))
        assert connection_manager.active_connections == 1

        run_async(connection_manager.disconnect(client.client_id))

        assert connection_manager.active_connections == 0

    def test_disconnect_unknown_client_no_error(self, connection_manager):
        """Disconnecting unknown client doesn't raise error."""
        run_async(connection_manager.disconnect("unknown-id"))
        # Should not raise

    def test_send_message_to_connected_client(
        self, connection_manager, mock_websocket
    ):
        """Send message successfully to connected client."""
        client = run_async(connection_manager.connect(mock_websocket))

        result = run_async(
            connection_manager.send_message(
                client.client_id, MessageType.PONG, {"timestamp": 123}
            )
        )

        assert result is True
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "pong"
        assert call_args["data"]["timestamp"] == 123

    def test_send_message_to_unknown_client_returns_false(
        self, connection_manager
    ):
        """Send message to unknown client returns False."""
        result = run_async(
            connection_manager.send_message("unknown-id", MessageType.PONG, {})
        )

        assert result is False

    def test_broadcast_sends_to_all_clients(self, connection_manager):
        """Broadcast sends message to all connected clients."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        run_async(connection_manager.connect(ws1))
        run_async(connection_manager.connect(ws2))

        run_async(
            connection_manager.broadcast(MessageType.HEALTH_UPDATE, {"status": "ok"})
        )

        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()

    def test_health_subscription(self, connection_manager, mock_websocket):
        """Health subscription can be enabled and disabled."""
        client = run_async(connection_manager.connect(mock_websocket))

        assert client.subscribed_to_health is False

        run_async(connection_manager.set_health_subscription(client.client_id, True))
        updated_client = connection_manager.get_client(client.client_id)
        assert updated_client.subscribed_to_health is True

        run_async(connection_manager.set_health_subscription(client.client_id, False))
        updated_client = connection_manager.get_client(client.client_id)
        assert updated_client.subscribed_to_health is False

    def test_broadcast_health_update_only_to_subscribers(
        self, connection_manager
    ):
        """Health updates only sent to subscribed clients."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        client1 = run_async(connection_manager.connect(ws1))
        run_async(connection_manager.connect(ws2))

        # Subscribe only client1
        run_async(connection_manager.set_health_subscription(client1.client_id, True))

        run_async(connection_manager.broadcast_health_update({"status": "healthy"}))

        ws1.send_json.assert_called_once()
        ws2.send_json.assert_not_called()

    def test_set_active_generation(self, connection_manager, mock_websocket):
        """Active generation ID can be set and cleared."""
        client = run_async(connection_manager.connect(mock_websocket))

        run_async(
            connection_manager.set_active_generation(client.client_id, "gen-123")
        )
        updated_client = connection_manager.get_client(client.client_id)
        assert updated_client.active_generation_id == "gen-123"

        run_async(connection_manager.set_active_generation(client.client_id, None))
        updated_client = connection_manager.get_client(client.client_id)
        assert updated_client.active_generation_id is None

    def test_get_all_client_ids(self, connection_manager):
        """Get all client IDs returns list of connected clients."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()

        client1 = run_async(connection_manager.connect(ws1))
        client2 = run_async(connection_manager.connect(ws2))

        client_ids = connection_manager.get_all_client_ids()

        assert len(client_ids) == 2
        assert client1.client_id in client_ids
        assert client2.client_id in client_ids


class TestMessageTypes:
    """Tests for MessageType enum."""

    def test_client_message_types(self):
        """Client message types have expected values."""
        assert MessageType.GENERATE.value == "generate"
        assert MessageType.GENERATE_STREAM.value == "generate_stream"
        assert MessageType.SUBSCRIBE_HEALTH.value == "subscribe_health"
        assert MessageType.UNSUBSCRIBE_HEALTH.value == "unsubscribe_health"
        assert MessageType.PING.value == "ping"
        assert MessageType.CANCEL.value == "cancel"

    def test_server_message_types(self):
        """Server message types have expected values."""
        assert MessageType.CONNECTED.value == "connected"
        assert MessageType.TOKEN.value == "token"
        assert MessageType.GENERATION_START.value == "generation_start"
        assert MessageType.GENERATION_COMPLETE.value == "generation_complete"
        assert MessageType.GENERATION_ERROR.value == "generation_error"
        assert MessageType.HEALTH_UPDATE.value == "health_update"
        assert MessageType.PONG.value == "pong"
        assert MessageType.ERROR.value == "error"


class TestWebSocketClient:
    """Tests for WebSocketClient dataclass."""

    def test_client_initialization(self, mock_websocket):
        """WebSocketClient initializes with correct defaults."""
        client = WebSocketClient(websocket=mock_websocket, client_id="test-123")

        assert client.websocket is mock_websocket
        assert client.client_id == "test-123"
        assert client.subscribed_to_health is False
        assert client.active_generation_id is None
        assert client.connected_at is not None


class TestGetConnectionManager:
    """Tests for get_connection_manager function."""

    def test_returns_singleton_manager(self):
        """get_connection_manager returns global manager instance."""
        result = get_connection_manager()
        assert result is manager


class TestWebSocketStatusEndpoint:
    """Tests for /ws/status endpoint."""

    def test_status_returns_connection_info(self):
        """Status endpoint returns connection information."""
        from fastapi.testclient import TestClient

        from api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/ws/status")

        assert response.status_code == 200
        data = response.json()
        assert "active_connections" in data
        assert "health_subscribers" in data
        assert data["status"] == "operational"
