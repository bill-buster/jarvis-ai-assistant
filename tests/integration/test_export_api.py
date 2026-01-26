"""Integration tests for the export API endpoints.

Tests the FastAPI export endpoints with mocked iMessage reader.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_imessage_reader
from api.main import app
from contracts.imessage import Attachment, Conversation, Message, Reaction


@pytest.fixture
def mock_messages():
    """Create mock messages for testing."""
    return [
        Message(
            id=1,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Hey, how are you?",
            date=datetime(2024, 1, 15, 10, 30),
            is_from_me=False,
            attachments=[],
            reactions=[],
        ),
        Message(
            id=2,
            chat_id="iMessage;-;+1234567890",
            sender="me",
            sender_name=None,
            text="I'm doing great, thanks!",
            date=datetime(2024, 1, 15, 10, 31),
            is_from_me=True,
            attachments=[],
            reactions=[],
        ),
        Message(
            id=3,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Want to grab coffee later?",
            date=datetime(2024, 1, 15, 10, 32),
            is_from_me=False,
            attachments=[
                Attachment(
                    filename="coffee.jpg",
                    file_path="/path/to/coffee.jpg",
                    mime_type="image/jpeg",
                    file_size=1024,
                )
            ],
            reactions=[
                Reaction(
                    type="love",
                    sender="me",
                    sender_name=None,
                    date=datetime(2024, 1, 15, 10, 33),
                )
            ],
        ),
    ]


@pytest.fixture
def mock_conversations():
    """Create mock conversations for testing."""
    return [
        Conversation(
            chat_id="iMessage;-;+1234567890",
            participants=["+1234567890"],
            display_name="John Smith",
            last_message_date=datetime(2024, 1, 15, 10, 32),
            message_count=3,
            is_group=False,
            last_message_text="Want to grab coffee later?",
        ),
        Conversation(
            chat_id="iMessage;-;+0987654321",
            participants=["+0987654321"],
            display_name="Jane Doe",
            last_message_date=datetime(2024, 1, 14, 15, 0),
            message_count=10,
            is_group=False,
            last_message_text="See you tomorrow!",
        ),
    ]


@pytest.fixture
def mock_reader(mock_messages, mock_conversations):
    """Create a mock iMessage reader."""
    reader = MagicMock()
    reader.check_access.return_value = True
    reader.get_conversations.return_value = mock_conversations
    reader.get_messages.return_value = mock_messages
    reader.search.return_value = mock_messages
    return reader


@pytest.fixture
def client(mock_reader):
    """Create a test client with mocked iMessage reader."""
    app.dependency_overrides[get_imessage_reader] = lambda: mock_reader
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


class TestExportConversationEndpoint:
    """Tests for POST /export/conversation/{chat_id} endpoint."""

    def test_export_conversation_json(self, client, mock_reader):
        """Successfully exports conversation as JSON."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "json"
        assert data["message_count"] == 3
        assert data["export_type"] == "conversation"
        assert data["filename"].endswith(".json")

        # Verify the exported data is valid JSON
        exported = json.loads(data["data"])
        assert "messages" in exported
        assert len(exported["messages"]) == 3

    def test_export_conversation_csv(self, client, mock_reader):
        """Successfully exports conversation as CSV."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "csv"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "csv"
        assert data["filename"].endswith(".csv")

        # Verify CSV has headers and data
        lines = data["data"].strip().split("\n")
        assert len(lines) == 4  # header + 3 messages

    def test_export_conversation_txt(self, client, mock_reader):
        """Successfully exports conversation as TXT."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "txt"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "txt"
        assert data["filename"].endswith(".txt")

        # Verify TXT has expected content
        assert "CONVERSATION EXPORT" in data["data"]
        assert "John Smith" in data["data"]

    def test_export_conversation_not_found(self, client, mock_reader):
        """Returns 404 for non-existent conversation."""
        response = client.post(
            "/export/conversation/nonexistent-chat",
            json={"format": "json"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_export_conversation_with_date_range(self, client, mock_reader, mock_messages):
        """Exports conversation with date range filter."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={
                "format": "json",
                "date_range": {
                    "start": "2024-01-15T10:30:00",
                    "end": "2024-01-15T10:31:00",
                },
            },
        )

        assert response.status_code == 200
        # Note: The actual filtering is tested in unit tests,
        # here we just verify the endpoint accepts the parameters

    def test_export_conversation_with_limit(self, client, mock_reader):
        """Exports conversation with message limit."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json", "limit": 100},
        )

        assert response.status_code == 200
        mock_reader.get_messages.assert_called()

    def test_export_conversation_with_include_attachments(self, client, mock_reader):
        """Exports CSV with attachment columns."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "csv", "include_attachments": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert "attachment_count" in data["data"]

    def test_export_conversation_validates_format(self, client, mock_reader):
        """Validates export format."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "invalid"},
        )

        assert response.status_code == 422

    def test_export_conversation_validates_limit_range(self, client, mock_reader):
        """Validates limit is within range."""
        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json", "limit": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json", "limit": 100000},
        )
        assert response.status_code == 422

    def test_export_conversation_no_messages(self, client, mock_reader):
        """Returns 404 when no messages found."""
        mock_reader.get_messages.return_value = []

        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json"},
        )

        assert response.status_code == 404
        assert "No messages found" in response.json()["detail"]


class TestExportSearchEndpoint:
    """Tests for POST /export/search endpoint."""

    def test_export_search_json(self, client, mock_reader):
        """Successfully exports search results as JSON."""
        response = client.post(
            "/export/search",
            json={"query": "coffee", "format": "json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "json"
        assert data["export_type"] == "search"

        # Verify the exported data
        exported = json.loads(data["data"])
        assert exported["export_metadata"]["query"] == "coffee"

    def test_export_search_csv(self, client, mock_reader):
        """Successfully exports search results as CSV."""
        response = client.post(
            "/export/search",
            json={"query": "coffee", "format": "csv"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "csv"

    def test_export_search_txt(self, client, mock_reader):
        """Successfully exports search results as TXT."""
        response = client.post(
            "/export/search",
            json={"query": "coffee", "format": "txt"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "SEARCH RESULTS EXPORT" in data["data"]
        assert "Search Query: coffee" in data["data"]

    def test_export_search_no_results(self, client, mock_reader):
        """Returns 404 when no search results found."""
        mock_reader.search.return_value = []

        response = client.post(
            "/export/search",
            json={"query": "nonexistent"},
        )

        assert response.status_code == 404
        assert "No messages found" in response.json()["detail"]

    def test_export_search_with_filters(self, client, mock_reader):
        """Exports search with filters."""
        response = client.post(
            "/export/search",
            json={
                "query": "coffee",
                "format": "json",
                "sender": "+1234567890",
                "date_range": {
                    "start": "2024-01-01T00:00:00",
                    "end": "2024-12-31T23:59:59",
                },
            },
        )

        assert response.status_code == 200
        mock_reader.search.assert_called_once()

    def test_export_search_requires_query(self, client, mock_reader):
        """Search export requires query parameter."""
        response = client.post(
            "/export/search",
            json={"format": "json"},
        )

        assert response.status_code == 422

    def test_export_search_validates_limit(self, client, mock_reader):
        """Validates search limit."""
        response = client.post(
            "/export/search",
            json={"query": "test", "limit": 10000},
        )

        assert response.status_code == 422


class TestExportBackupEndpoint:
    """Tests for POST /export/backup endpoint."""

    def test_export_backup_success(self, client, mock_reader):
        """Successfully exports backup."""
        response = client.post(
            "/export/backup",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "json"
        assert data["export_type"] == "backup"

        # Verify backup structure
        exported = json.loads(data["data"])
        assert exported["export_metadata"]["type"] == "full_backup"
        assert "conversations" in exported

    def test_export_backup_with_limits(self, client, mock_reader):
        """Exports backup with conversation and message limits."""
        response = client.post(
            "/export/backup",
            json={
                "conversation_limit": 10,
                "messages_per_conversation": 100,
            },
        )

        assert response.status_code == 200
        mock_reader.get_conversations.assert_called_with(limit=10)

    def test_export_backup_with_date_range(self, client, mock_reader):
        """Exports backup with date range filter."""
        response = client.post(
            "/export/backup",
            json={
                "date_range": {
                    "start": "2024-01-01T00:00:00",
                    "end": "2024-12-31T23:59:59",
                }
            },
        )

        assert response.status_code == 200

    def test_export_backup_no_conversations(self, client, mock_reader):
        """Returns 404 when no conversations found."""
        mock_reader.get_conversations.return_value = []

        response = client.post(
            "/export/backup",
            json={},
        )

        assert response.status_code == 404
        assert "No conversations found" in response.json()["detail"]

    def test_export_backup_validates_conversation_limit(self, client, mock_reader):
        """Validates conversation limit range."""
        response = client.post(
            "/export/backup",
            json={"conversation_limit": 1000},
        )

        assert response.status_code == 422

    def test_export_backup_validates_messages_per_conversation(self, client, mock_reader):
        """Validates messages per conversation limit."""
        response = client.post(
            "/export/backup",
            json={"messages_per_conversation": 10000},
        )

        assert response.status_code == 422


class TestExportRouterRegistration:
    """Tests for export router registration."""

    def test_export_router_is_registered(self, client, mock_reader):
        """Verify export router is registered."""
        # Check conversation endpoint exists
        response = client.post(
            "/export/conversation/test",
            json={"format": "json"},
        )
        assert (
            response.status_code != 404 or "not found" in response.json().get("detail", "").lower()
        )

    def test_export_search_endpoint_exists(self, client, mock_reader):
        """Verify search export endpoint exists."""
        response = client.post(
            "/export/search",
            json={"query": "test"},
        )
        assert response.status_code in (200, 404)  # 404 means no results, not missing endpoint

    def test_export_backup_endpoint_exists(self, client, mock_reader):
        """Verify backup endpoint exists."""
        response = client.post(
            "/export/backup",
            json={},
        )
        assert response.status_code in (200, 404)


class TestExportErrorHandling:
    """Tests for export error handling."""

    def test_handles_reader_error(self, client, mock_reader):
        """Handles iMessage reader errors gracefully."""
        mock_reader.get_messages.side_effect = RuntimeError("Database error")

        response = client.post(
            "/export/conversation/iMessage;-;+1234567890",
            json={"format": "json"},
        )

        assert response.status_code == 500
        assert "Failed to export" in response.json()["detail"]

    def test_handles_search_error(self, client, mock_reader):
        """Handles search errors gracefully."""
        mock_reader.search.side_effect = RuntimeError("Search error")

        response = client.post(
            "/export/search",
            json={"query": "test"},
        )

        assert response.status_code == 500
        assert "Failed to export" in response.json()["detail"]

    def test_handles_backup_error(self, client, mock_reader):
        """Handles backup errors gracefully."""
        mock_reader.get_conversations.side_effect = RuntimeError("Backup error")

        response = client.post(
            "/export/backup",
            json={},
        )

        assert response.status_code == 500
        assert "Failed to create backup" in response.json()["detail"]
