"""Unit tests for the export module.

Tests export functionality for conversations in various formats (JSON, CSV, TXT).
"""

import csv
import io
import json
from datetime import datetime

import pytest

from contracts.imessage import Attachment, Conversation, Message, Reaction
from jarvis.export import (
    ExportFormat,
    export_backup,
    export_messages,
    export_messages_csv,
    export_messages_json,
    export_messages_txt,
    export_search_results,
    get_export_filename,
)


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
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
def sample_conversation() -> Conversation:
    """Create a sample conversation for testing."""
    return Conversation(
        chat_id="iMessage;-;+1234567890",
        participants=["+1234567890"],
        display_name="John Smith",
        last_message_date=datetime(2024, 1, 15, 10, 32),
        message_count=3,
        is_group=False,
        last_message_text="Want to grab coffee later?",
    )


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_json_format(self):
        """JSON format value is correct."""
        assert ExportFormat.JSON.value == "json"

    def test_csv_format(self):
        """CSV format value is correct."""
        assert ExportFormat.CSV.value == "csv"

    def test_txt_format(self):
        """TXT format value is correct."""
        assert ExportFormat.TXT.value == "txt"


class TestExportMessagesJson:
    """Tests for JSON export functionality."""

    def test_exports_messages_to_json(self, sample_messages):
        """Exports messages to valid JSON."""
        result = export_messages_json(sample_messages)
        data = json.loads(result)

        assert "messages" in data
        assert len(data["messages"]) == 3

    def test_includes_export_metadata(self, sample_messages):
        """JSON export includes metadata by default."""
        result = export_messages_json(sample_messages)
        data = json.loads(result)

        assert "export_metadata" in data
        assert data["export_metadata"]["format"] == "json"
        assert data["export_metadata"]["message_count"] == 3
        assert "exported_at" in data["export_metadata"]

    def test_excludes_metadata_when_disabled(self, sample_messages):
        """JSON export excludes metadata when disabled."""
        result = export_messages_json(sample_messages, include_metadata=False)
        data = json.loads(result)

        assert "export_metadata" not in data
        assert "messages" in data

    def test_includes_conversation_metadata(self, sample_messages, sample_conversation):
        """JSON export includes conversation metadata when provided."""
        result = export_messages_json(sample_messages, conversation=sample_conversation)
        data = json.loads(result)

        assert "conversation" in data
        assert data["conversation"]["chat_id"] == "iMessage;-;+1234567890"
        assert data["conversation"]["display_name"] == "John Smith"

    def test_message_fields_are_serialized(self, sample_messages):
        """All message fields are properly serialized."""
        result = export_messages_json(sample_messages)
        data = json.loads(result)

        msg = data["messages"][0]
        assert msg["id"] == 1
        assert msg["sender"] == "+1234567890"
        assert msg["sender_name"] == "John Smith"
        assert msg["text"] == "Hey, how are you?"
        assert msg["is_from_me"] is False
        assert "date" in msg

    def test_attachments_are_serialized(self, sample_messages):
        """Attachments are properly serialized."""
        result = export_messages_json(sample_messages)
        data = json.loads(result)

        # Third message has an attachment
        msg = data["messages"][2]
        assert len(msg["attachments"]) == 1
        assert msg["attachments"][0]["filename"] == "coffee.jpg"
        assert msg["attachments"][0]["mime_type"] == "image/jpeg"

    def test_reactions_are_serialized(self, sample_messages):
        """Reactions are properly serialized."""
        result = export_messages_json(sample_messages)
        data = json.loads(result)

        # Third message has a reaction
        msg = data["messages"][2]
        assert len(msg["reactions"]) == 1
        assert msg["reactions"][0]["type"] == "love"

    def test_empty_messages_list(self):
        """Handles empty message list."""
        result = export_messages_json([])
        data = json.loads(result)

        assert data["messages"] == []
        assert data["export_metadata"]["message_count"] == 0


class TestExportMessagesCsv:
    """Tests for CSV export functionality."""

    def test_exports_messages_to_csv(self, sample_messages):
        """Exports messages to valid CSV."""
        result = export_messages_csv(sample_messages)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 3

    def test_csv_has_correct_columns(self, sample_messages):
        """CSV has expected column headers."""
        result = export_messages_csv(sample_messages)

        reader = csv.DictReader(io.StringIO(result))
        headers = reader.fieldnames

        assert "id" in headers
        assert "chat_id" in headers
        assert "sender" in headers
        assert "sender_name" in headers
        assert "text" in headers
        assert "date" in headers
        assert "is_from_me" in headers

    def test_csv_includes_attachment_columns_when_enabled(self, sample_messages):
        """CSV includes attachment columns when enabled."""
        result = export_messages_csv(sample_messages, include_attachments=True)

        reader = csv.DictReader(io.StringIO(result))
        headers = reader.fieldnames

        assert "attachment_count" in headers
        assert "attachment_filenames" in headers

    def test_csv_excludes_attachment_columns_by_default(self, sample_messages):
        """CSV excludes attachment columns by default."""
        result = export_messages_csv(sample_messages)

        reader = csv.DictReader(io.StringIO(result))
        headers = reader.fieldnames

        assert "attachment_count" not in headers

    def test_csv_escapes_newlines(self, sample_messages):
        """CSV escapes newlines in text."""
        messages = [
            Message(
                id=1,
                chat_id="test",
                sender="test",
                sender_name=None,
                text="Line 1\nLine 2",
                date=datetime.now(),
                is_from_me=False,
            )
        ]
        result = export_messages_csv(messages)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        assert "\\n" in row["text"]

    def test_csv_reaction_count(self, sample_messages):
        """CSV includes reaction count."""
        result = export_messages_csv(sample_messages)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        # Third message has one reaction
        assert rows[2]["reaction_count"] == "1"

    def test_empty_messages_list(self):
        """Handles empty message list."""
        result = export_messages_csv([])

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 0


class TestExportMessagesTxt:
    """Tests for TXT export functionality."""

    def test_exports_messages_to_txt(self, sample_messages):
        """Exports messages to text format."""
        result = export_messages_txt(sample_messages)

        assert "John Smith" in result
        assert "Hey, how are you?" in result

    def test_includes_header_by_default(self, sample_messages):
        """TXT export includes header by default."""
        result = export_messages_txt(sample_messages)

        assert "CONVERSATION EXPORT" in result
        assert "Exported Messages:" in result

    def test_excludes_header_when_disabled(self, sample_messages):
        """TXT export excludes header when disabled."""
        result = export_messages_txt(sample_messages, include_metadata=False)

        assert "CONVERSATION EXPORT" not in result

    def test_includes_conversation_info(self, sample_messages, sample_conversation):
        """TXT export includes conversation info when provided."""
        result = export_messages_txt(sample_messages, conversation=sample_conversation)

        assert "Conversation: John Smith" in result
        assert "Type: Individual" in result

    def test_uses_me_for_sent_messages(self, sample_messages):
        """TXT export uses 'Me' for sent messages."""
        result = export_messages_txt(sample_messages)

        assert "Me:" in result

    def test_shows_attachments(self, sample_messages):
        """TXT export shows attachment info."""
        result = export_messages_txt(sample_messages)

        assert "coffee.jpg" in result

    def test_shows_reactions(self, sample_messages):
        """TXT export shows reaction info."""
        result = export_messages_txt(sample_messages)

        assert "love" in result

    def test_date_range_in_header(self, sample_messages):
        """TXT export shows date range in header."""
        result = export_messages_txt(sample_messages)

        assert "Date Range:" in result
        assert "2024-01-15" in result

    def test_empty_messages_list(self):
        """Handles empty message list."""
        result = export_messages_txt([])

        assert "Exported Messages: 0" in result


class TestExportMessages:
    """Tests for the main export_messages function."""

    def test_exports_json_format(self, sample_messages):
        """export_messages handles JSON format."""
        result = export_messages(sample_messages, ExportFormat.JSON)
        data = json.loads(result)

        assert "messages" in data

    def test_exports_csv_format(self, sample_messages):
        """export_messages handles CSV format."""
        result = export_messages(sample_messages, ExportFormat.CSV)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 3

    def test_exports_txt_format(self, sample_messages):
        """export_messages handles TXT format."""
        result = export_messages(sample_messages, ExportFormat.TXT)

        assert "CONVERSATION EXPORT" in result

    def test_passes_conversation_to_json(self, sample_messages, sample_conversation):
        """export_messages passes conversation to JSON export."""
        result = export_messages(
            sample_messages, ExportFormat.JSON, conversation=sample_conversation
        )
        data = json.loads(result)

        assert "conversation" in data

    def test_passes_include_attachments_to_csv(self, sample_messages):
        """export_messages passes include_attachments to CSV export."""
        result = export_messages(sample_messages, ExportFormat.CSV, include_attachments=True)

        reader = csv.DictReader(io.StringIO(result))
        headers = reader.fieldnames

        assert "attachment_count" in headers


class TestExportSearchResults:
    """Tests for search results export functionality."""

    def test_exports_search_results_json(self, sample_messages):
        """Exports search results to JSON."""
        result = export_search_results(sample_messages, "coffee", ExportFormat.JSON)
        data = json.loads(result)

        assert data["export_metadata"]["type"] == "search_results"
        assert data["export_metadata"]["query"] == "coffee"
        assert len(data["messages"]) == 3

    def test_exports_search_results_csv(self, sample_messages):
        """Exports search results to CSV."""
        result = export_search_results(sample_messages, "coffee", ExportFormat.CSV)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 3

    def test_exports_search_results_txt(self, sample_messages):
        """Exports search results to TXT."""
        result = export_search_results(sample_messages, "coffee", ExportFormat.TXT)

        assert "SEARCH RESULTS EXPORT" in result
        assert "Search Query: coffee" in result
        assert "Results Found: 3" in result


class TestExportBackup:
    """Tests for full backup export functionality."""

    def test_exports_backup_json(self, sample_messages, sample_conversation):
        """Exports backup to JSON."""
        conversations = [(sample_conversation, sample_messages)]
        result = export_backup(conversations)
        data = json.loads(result)

        assert data["export_metadata"]["type"] == "full_backup"
        assert data["export_metadata"]["conversation_count"] == 1
        assert data["export_metadata"]["total_message_count"] == 3

    def test_backup_includes_conversation_metadata(self, sample_messages, sample_conversation):
        """Backup includes conversation metadata."""
        conversations = [(sample_conversation, sample_messages)]
        result = export_backup(conversations)
        data = json.loads(result)

        conv = data["conversations"][0]
        assert "metadata" in conv
        assert conv["metadata"]["chat_id"] == "iMessage;-;+1234567890"

    def test_backup_includes_messages(self, sample_messages, sample_conversation):
        """Backup includes messages for each conversation."""
        conversations = [(sample_conversation, sample_messages)]
        result = export_backup(conversations)
        data = json.loads(result)

        conv = data["conversations"][0]
        assert "messages" in conv
        assert len(conv["messages"]) == 3

    def test_backup_rejects_non_json_format(self, sample_messages, sample_conversation):
        """Backup raises error for non-JSON formats."""
        conversations = [(sample_conversation, sample_messages)]

        with pytest.raises(ValueError, match="only supports JSON"):
            export_backup(conversations, format=ExportFormat.CSV)

    def test_backup_multiple_conversations(self, sample_messages, sample_conversation):
        """Backup handles multiple conversations."""
        conv2 = Conversation(
            chat_id="iMessage;-;+0987654321",
            participants=["+0987654321"],
            display_name="Jane Doe",
            last_message_date=datetime(2024, 1, 16, 12, 0),
            message_count=1,
            is_group=False,
        )
        msg2 = Message(
            id=4,
            chat_id="iMessage;-;+0987654321",
            sender="+0987654321",
            sender_name="Jane Doe",
            text="Hello there!",
            date=datetime(2024, 1, 16, 12, 0),
            is_from_me=False,
        )

        conversations = [
            (sample_conversation, sample_messages),
            (conv2, [msg2]),
        ]
        result = export_backup(conversations)
        data = json.loads(result)

        assert data["export_metadata"]["conversation_count"] == 2
        assert data["export_metadata"]["total_message_count"] == 4
        assert len(data["conversations"]) == 2


class TestGetExportFilename:
    """Tests for filename generation."""

    def test_generates_json_filename(self):
        """Generates JSON filename."""
        filename = get_export_filename(ExportFormat.JSON)

        assert filename.endswith(".json")
        assert filename.startswith("export_")

    def test_generates_csv_filename(self):
        """Generates CSV filename."""
        filename = get_export_filename(ExportFormat.CSV)

        assert filename.endswith(".csv")

    def test_generates_txt_filename(self):
        """Generates TXT filename."""
        filename = get_export_filename(ExportFormat.TXT)

        assert filename.endswith(".txt")

    def test_uses_custom_prefix(self):
        """Uses custom prefix."""
        filename = get_export_filename(ExportFormat.JSON, prefix="backup")

        assert filename.startswith("backup_")

    def test_includes_chat_id(self):
        """Includes sanitized chat ID."""
        filename = get_export_filename(ExportFormat.JSON, chat_id="iMessage;-;+1234567890")

        # Special characters should be replaced
        assert "iMessage" in filename
        assert ";" not in filename

    def test_truncates_long_chat_id(self):
        """Truncates very long chat IDs."""
        long_id = "a" * 100
        filename = get_export_filename(ExportFormat.JSON, chat_id=long_id)

        # Should be truncated
        assert len(filename) < 150

    def test_includes_timestamp(self):
        """Includes timestamp in filename."""
        filename = get_export_filename(ExportFormat.JSON)

        # Should have timestamp format YYYYMMDD_HHMMSS
        import re

        assert re.search(r"\d{8}_\d{6}", filename)


class TestExportSchemas:
    """Tests for export-related Pydantic schemas."""

    def test_export_format_enum(self):
        """ExportFormatEnum has correct values."""
        from api.schemas import ExportFormatEnum

        assert ExportFormatEnum.JSON.value == "json"
        assert ExportFormatEnum.CSV.value == "csv"
        assert ExportFormatEnum.TXT.value == "txt"

    def test_export_date_range_model(self):
        """ExportDateRange model works correctly."""
        from api.schemas import ExportDateRange

        date_range = ExportDateRange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        assert date_range.start == datetime(2024, 1, 1)
        assert date_range.end == datetime(2024, 12, 31)

    def test_export_date_range_optional(self):
        """ExportDateRange fields are optional."""
        from api.schemas import ExportDateRange

        date_range = ExportDateRange()
        assert date_range.start is None
        assert date_range.end is None

    def test_export_conversation_request_defaults(self):
        """ExportConversationRequest has correct defaults."""
        from api.schemas import ExportConversationRequest, ExportFormatEnum

        request = ExportConversationRequest()
        assert request.format == ExportFormatEnum.JSON
        assert request.include_attachments is False
        assert request.limit == 1000

    def test_export_search_request_validation(self):
        """ExportSearchRequest validates required fields."""
        from api.schemas import ExportSearchRequest

        request = ExportSearchRequest(query="test")
        assert request.query == "test"
        assert request.limit == 500

    def test_export_backup_request_defaults(self):
        """ExportBackupRequest has correct defaults."""
        from api.schemas import ExportBackupRequest

        request = ExportBackupRequest()
        assert request.conversation_limit == 50
        assert request.messages_per_conversation == 500

    def test_export_response_model(self):
        """ExportResponse model works correctly."""
        from api.schemas import ExportResponse

        response = ExportResponse(
            success=True,
            format="json",
            filename="export_20240115.json",
            data='{"messages": []}',
            message_count=0,
            export_type="conversation",
        )
        assert response.success is True
        assert response.format == "json"
        assert response.message_count == 0
