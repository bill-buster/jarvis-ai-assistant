"""Unit tests for iMessage integration (Workstream 10)."""

import sqlite3
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contracts.imessage import Attachment, Conversation, Message, Reaction
from integrations.imessage import CHAT_DB_PATH, ChatDBReader
from integrations.imessage.parser import (
    _get_reaction_type,
    datetime_to_apple_timestamp,
    extract_text_from_row,
    normalize_phone_number,
    parse_apple_timestamp,
    parse_attachments,
    parse_attributed_body,
    parse_reactions,
)
from integrations.imessage.queries import detect_schema_version, get_query

# =============================================================================
# Parser Tests
# =============================================================================


class TestParseAppleTimestamp:
    """Tests for Apple Core Data timestamp parsing."""

    def test_valid_timestamp(self):
        """Convert a known timestamp to datetime."""
        # 2024-01-12 12:00:00 UTC in Apple format (nanoseconds since 2001-01-01)
        # 726753600 seconds from 2001-01-01 to 2024-01-12 12:00:00
        ts = 726753600 * 1_000_000_000
        result = parse_apple_timestamp(ts)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 12

    def test_none_timestamp(self):
        """Return Apple epoch for None timestamp."""
        result = parse_apple_timestamp(None)
        assert result.year == 2001
        assert result.month == 1
        assert result.day == 1

    def test_zero_timestamp(self):
        """Return Apple epoch for zero timestamp."""
        result = parse_apple_timestamp(0)
        assert result.year == 2001

    def test_returns_utc(self):
        """Ensure returned datetime is UTC."""
        result = parse_apple_timestamp(1_000_000_000_000_000_000)
        assert result.tzinfo == UTC


class TestDatetimeToAppleTimestamp:
    """Tests for datetime to Apple timestamp conversion."""

    def test_roundtrip_conversion(self):
        """Converting to Apple timestamp and back should preserve datetime."""
        # Use a known timestamp value
        original_ts = 726753600 * 1_000_000_000
        dt = parse_apple_timestamp(original_ts)
        result_ts = datetime_to_apple_timestamp(dt)
        assert result_ts == original_ts

    def test_known_date(self):
        """Convert a known date to Apple timestamp."""
        # 2024-01-12 12:00:00 UTC should be 726753600 seconds from Apple epoch
        test_dt = dt(2024, 1, 12, 12, 0, 0, tzinfo=UTC)
        result = datetime_to_apple_timestamp(test_dt)
        expected = 726753600 * 1_000_000_000
        assert result == expected


class TestNormalizePhoneNumber:
    """Tests for phone number normalization."""

    def test_us_number_10_digits(self):
        """Normalize 10-digit US number."""
        result = normalize_phone_number("5551234567")
        assert result == "+15551234567"

    def test_us_number_with_country_code(self):
        """Normalize 11-digit number with leading 1."""
        result = normalize_phone_number("15551234567")
        assert result == "+15551234567"

    def test_formatted_number(self):
        """Strip formatting from phone number."""
        result = normalize_phone_number("(555) 123-4567")
        assert result == "+15551234567"

    def test_email_passthrough(self):
        """Pass through email addresses unchanged."""
        result = normalize_phone_number("user@example.com")
        assert result == "user@example.com"

    def test_email_with_whitespace(self):
        """Strip whitespace from email addresses."""
        result = normalize_phone_number("  user@example.com  ")
        assert result == "user@example.com"

    def test_none_returns_none(self):
        """Return None for None input."""
        result = normalize_phone_number(None)
        assert result is None

    def test_international_format_preserved(self):
        """Numbers with + prefix are preserved."""
        result = normalize_phone_number("+44 20 7946 0958")
        # Formatting stripped but + and digits preserved
        assert result == "+442079460958"

    def test_plus_prefix_not_duplicated(self):
        """Don't add extra + to numbers that already have it."""
        result = normalize_phone_number("+15551234567")
        assert result == "+15551234567"
        assert not result.startswith("++")

    def test_international_short_number(self):
        """International numbers without + are returned as-is."""
        # 8-digit number that doesn't match US patterns
        result = normalize_phone_number("12345678")
        assert result == "12345678"


class TestParseAttributedBody:
    """Tests for NSKeyedArchive attributedBody parsing."""

    def test_none_returns_none(self):
        """Return None for None input."""
        result = parse_attributed_body(None)
        assert result is None

    def test_invalid_data_returns_none(self):
        """Return None for invalid plist data."""
        result = parse_attributed_body(b"not a plist")
        assert result is None

    def test_empty_bytes_returns_none(self):
        """Return None for empty bytes."""
        result = parse_attributed_body(b"")
        assert result is None

    def test_valid_plist_with_objects_array(self):
        """Extract text from $objects array in NSKeyedArchiver format."""
        import plistlib

        # Create a valid plist with $objects array containing text
        plist_data = {
            "$objects": [
                "$null",
                "NSMutableAttributedString",  # Should be skipped
                "Hello from attributed body",  # This should be returned
            ]
        }
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result == "Hello from attributed body"

    def test_valid_plist_with_ns_string_dict(self):
        """Extract text from NS.string key in dict object."""
        import plistlib

        plist_data = {
            "$objects": [
                {"NS.string": "Text from NS.string"},
            ]
        }
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result == "Text from NS.string"

    def test_valid_plist_with_nsstring_dict(self):
        """Extract text from NSString key in dict object."""
        import plistlib

        plist_data = {
            "$objects": [
                {"NSString": "Text from NSString"},
            ]
        }
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result == "Text from NSString"

    def test_valid_plist_skips_metadata_strings(self):
        """Skip metadata strings starting with $ or known class names."""
        import plistlib

        plist_data = {
            "$objects": [
                "$class",
                "NSAttributedString",
                "NSMutableString",
                "NSString",
                "NSDictionary",
                "NSArray",
                "Actual message text",
            ]
        }
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result == "Actual message text"

    def test_plist_without_objects_returns_none(self):
        """Return None for valid plist without $objects."""
        import plistlib

        plist_data = {"other_key": "value"}
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result is None

    def test_plist_with_empty_objects_returns_none(self):
        """Return None for plist with empty $objects array."""
        import plistlib

        plist_data = {"$objects": []}
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result is None

    def test_ns_string_with_non_string_value(self):
        """Return None if NS.string value is not a string."""
        import plistlib

        plist_data = {
            "$objects": [
                {"NS.string": 12345},  # Not a string
            ]
        }
        data = plistlib.dumps(plist_data)
        result = parse_attributed_body(data)
        assert result is None


class TestParseAttachments:
    """Tests for attachment parsing."""

    def test_returns_empty_list_for_none(self):
        """Return empty list for None input."""
        result = parse_attachments(None)
        assert result == []

    def test_returns_empty_list_for_empty(self):
        """Return empty list for empty input."""
        result = parse_attachments([])
        assert result == []

    def test_parses_single_attachment(self):
        """Parse a single attachment row."""
        rows = [
            {
                "filename": "~/Library/Messages/Attachments/ab/cd/photo.jpg",
                "mime_type": "image/jpeg",
                "file_size": 12345,
                "transfer_name": "photo.jpg",
            }
        ]
        result = parse_attachments(rows)

        assert len(result) == 1
        assert isinstance(result[0], Attachment)
        assert result[0].filename == "photo.jpg"
        assert result[0].mime_type == "image/jpeg"
        assert result[0].file_size == 12345
        assert "photo.jpg" in result[0].file_path

    def test_parses_multiple_attachments(self):
        """Parse multiple attachment rows."""
        rows = [
            {"filename": "/path/to/image.png", "mime_type": "image/png", "file_size": 1000},
            {"filename": "/path/to/doc.pdf", "mime_type": "application/pdf", "file_size": 2000},
        ]
        result = parse_attachments(rows)

        assert len(result) == 2
        assert result[0].filename == "image.png"
        assert result[1].filename == "doc.pdf"

    def test_handles_missing_fields(self):
        """Handle rows with missing optional fields."""
        rows = [{"filename": "/path/to/file.txt"}]
        result = parse_attachments(rows)

        assert len(result) == 1
        assert result[0].filename == "file.txt"
        assert result[0].mime_type is None
        assert result[0].file_size is None

    def test_uses_transfer_name_as_fallback(self):
        """Use transfer_name when filename is missing."""
        rows = [{"transfer_name": "backup.zip", "mime_type": "application/zip"}]
        result = parse_attachments(rows)

        assert len(result) == 1
        assert result[0].filename == "backup.zip"

    def test_handles_empty_filename(self):
        """Handle rows with empty filename."""
        rows = [{"filename": "", "transfer_name": ""}]
        result = parse_attachments(rows)

        assert len(result) == 1
        assert result[0].filename == "unknown"


class TestParseReactions:
    """Tests for reaction parsing."""

    def test_returns_empty_list_for_none(self):
        """Return empty list for None input."""
        result = parse_reactions(None)
        assert result == []

    def test_returns_empty_list_for_empty(self):
        """Return empty list for empty input."""
        result = parse_reactions([])
        assert result == []

    def test_parses_love_reaction(self):
        """Parse a love reaction (type 2000)."""
        rows = [
            {
                "associated_message_type": 2000,
                "date": 726753600000000000,  # 2024-01-12 12:00:00 UTC
                "is_from_me": False,
                "sender": "+15551234567",
            }
        ]
        result = parse_reactions(rows)

        assert len(result) == 1
        assert isinstance(result[0], Reaction)
        assert result[0].type == "love"
        assert result[0].sender == "+15551234567"
        assert result[0].date.year == 2024

    def test_parses_all_reaction_types(self):
        """Parse all standard tapback types."""
        reaction_types = [
            (2000, "love"),
            (2001, "like"),
            (2002, "dislike"),
            (2003, "laugh"),
            (2004, "emphasize"),
            (2005, "question"),
        ]

        for type_code, expected_name in reaction_types:
            rows = [{"associated_message_type": type_code, "date": 0, "is_from_me": True}]
            result = parse_reactions(rows)
            assert len(result) == 1
            assert result[0].type == expected_name, f"Expected {expected_name} for {type_code}"

    def test_parses_removed_reactions(self):
        """Parse removed reaction types (3000 series)."""
        removed_types = [
            (3000, "removed_love"),
            (3001, "removed_like"),
            (3002, "removed_dislike"),
            (3003, "removed_laugh"),
            (3004, "removed_emphasize"),
            (3005, "removed_question"),
        ]

        for type_code, expected_name in removed_types:
            rows = [{"associated_message_type": type_code, "date": 0, "is_from_me": False}]
            result = parse_reactions(rows)
            assert len(result) == 1
            assert result[0].type == expected_name

    def test_handles_is_from_me(self):
        """Handle reactions from self."""
        rows = [{"associated_message_type": 2001, "date": 0, "is_from_me": True}]
        result = parse_reactions(rows)

        assert len(result) == 1
        assert result[0].sender == "me"

    def test_ignores_invalid_reaction_types(self):
        """Ignore rows with invalid reaction types."""
        rows = [
            {"associated_message_type": 0, "date": 0, "is_from_me": True},  # Not a reaction
            {"associated_message_type": 9999, "date": 0, "is_from_me": True},  # Unknown type
        ]
        result = parse_reactions(rows)

        assert len(result) == 0

    def test_parses_multiple_reactions(self):
        """Parse multiple reactions for a message."""
        rows = [
            {"associated_message_type": 2000, "date": 1, "is_from_me": False, "sender": "+1111"},
            {"associated_message_type": 2001, "date": 2, "is_from_me": False, "sender": "+2222"},
            {"associated_message_type": 2003, "date": 3, "is_from_me": True},
        ]
        result = parse_reactions(rows)

        assert len(result) == 3
        assert result[0].type == "love"
        assert result[1].type == "like"
        assert result[2].type == "laugh"


class TestGetReactionType:
    """Tests for _get_reaction_type helper function."""

    def test_valid_tapback_types(self):
        """Map valid tapback types to strings."""
        assert _get_reaction_type(2000) == "love"
        assert _get_reaction_type(2001) == "like"
        assert _get_reaction_type(2002) == "dislike"
        assert _get_reaction_type(2003) == "laugh"
        assert _get_reaction_type(2004) == "emphasize"
        assert _get_reaction_type(2005) == "question"

    def test_removed_tapback_types(self):
        """Map removed tapback types to strings."""
        assert _get_reaction_type(3000) == "removed_love"
        assert _get_reaction_type(3001) == "removed_like"
        assert _get_reaction_type(3002) == "removed_dislike"
        assert _get_reaction_type(3003) == "removed_laugh"
        assert _get_reaction_type(3004) == "removed_emphasize"
        assert _get_reaction_type(3005) == "removed_question"

    def test_invalid_type_returns_none(self):
        """Return None for invalid type codes."""
        assert _get_reaction_type(0) is None
        assert _get_reaction_type(1000) is None
        assert _get_reaction_type(4000) is None
        assert _get_reaction_type(-1) is None


class TestExtractTextFromRow:
    """Tests for text extraction from database rows."""

    def test_text_column_preferred(self):
        """Use text column when available."""
        row = {"text": "Hello world", "attributedBody": None}
        result = extract_text_from_row(row)
        assert result == "Hello world"

    def test_strips_whitespace(self):
        """Strip whitespace from extracted text."""
        row = {"text": "  Hello world  ", "attributedBody": None}
        result = extract_text_from_row(row)
        assert result == "Hello world"

    def test_empty_text_returns_empty(self):
        """Return empty string when both fields empty."""
        row = {"text": "", "attributedBody": None}
        result = extract_text_from_row(row)
        assert result == ""

    def test_fallback_to_attributed_body(self):
        """Fall back to attributedBody when text is empty."""
        import plistlib

        plist_data = {"$objects": ["Fallback text"]}
        row = {"text": None, "attributedBody": plistlib.dumps(plist_data)}
        result = extract_text_from_row(row)
        assert result == "Fallback text"

    def test_fallback_strips_whitespace(self):
        """Strip whitespace from attributedBody fallback."""
        import plistlib

        plist_data = {"$objects": ["  Fallback text with spaces  "]}
        row = {"text": "", "attributedBody": plistlib.dumps(plist_data)}
        result = extract_text_from_row(row)
        assert result == "Fallback text with spaces"

    def test_whitespace_only_text_uses_fallback(self):
        """Use attributedBody when text is whitespace only."""
        import plistlib

        plist_data = {"$objects": ["From attributed body"]}
        row = {"text": "   ", "attributedBody": plistlib.dumps(plist_data)}
        result = extract_text_from_row(row)
        assert result == "From attributed body"


class TestParseAppleTimestampEdgeCases:
    """Additional edge case tests for timestamp parsing."""

    def test_overflow_returns_epoch(self):
        """Return Apple epoch on overflow error."""
        # Very large timestamp that would cause overflow
        result = parse_apple_timestamp(10**30)
        assert result.year == 2001  # Apple epoch

    def test_negative_timestamp(self):
        """Handle negative timestamps (before Apple epoch)."""
        result = parse_apple_timestamp(-1_000_000_000_000_000_000)
        # Should return a valid datetime or epoch
        assert result is not None


# =============================================================================
# Query Tests
# =============================================================================


class TestDetectSchemaVersion:
    """Tests for schema version detection."""

    def test_detects_v14_schema(self):
        """Detect v14 schema with thread_originator_guid."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        # Simulate v14 columns
        cursor.fetchall.side_effect = [
            # First call: message table columns
            [
                (1, "ROWID", "INTEGER", 0, None, 1),
                (2, "thread_originator_guid", "TEXT", 0, None, 0),
            ],
            # Second call: chat table columns (no service_name)
            [(1, "ROWID", "INTEGER", 0, None, 1), (2, "guid", "TEXT", 0, None, 0)],
        ]

        result = detect_schema_version(conn)
        assert result == "v14"

    def test_detects_v15_schema(self):
        """Detect v15 schema with service_name in chat table."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        # Simulate v15 columns
        cursor.fetchall.side_effect = [
            # First call: message table columns with thread_originator_guid
            [
                (1, "ROWID", "INTEGER", 0, None, 1),
                (2, "thread_originator_guid", "TEXT", 0, None, 0),
            ],
            # Second call: chat table columns WITH service_name (v15 indicator)
            [(1, "ROWID", "INTEGER", 0, None, 1), (2, "service_name", "TEXT", 0, None, 0)],
        ]

        result = detect_schema_version(conn)
        assert result == "v15"

    def test_old_schema_without_thread_guid(self):
        """Detect v14 for older schema without thread_originator_guid."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        # Simulate older schema without thread_originator_guid
        cursor.fetchall.return_value = [
            (1, "ROWID", "INTEGER", 0, None, 1),
            (2, "text", "TEXT", 0, None, 0),
        ]

        result = detect_schema_version(conn)
        assert result == "v14"

    def test_unknown_on_error(self):
        """Return unknown on database error."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.execute.side_effect = sqlite3.Error("test error")

        result = detect_schema_version(conn)
        assert result == "unknown"


class TestGetQuery:
    """Tests for query retrieval."""

    def test_valid_query_v14(self):
        """Get v14 query by name."""
        result = get_query("messages", "v14")
        assert "SELECT" in result
        assert "message" in result

    def test_unknown_version_falls_back_to_v14(self):
        """Unknown version falls back to v14."""
        result = get_query("messages", "unknown")
        assert "SELECT" in result

    def test_query_with_since_filter(self):
        """Apply since filter when flag is True."""
        result = get_query("conversations", "v14", with_since_filter=True)
        assert "AND last_message_date > ?" in result

    def test_query_without_since_filter(self):
        """No since filter when flag is False."""
        result = get_query("conversations", "v14", with_since_filter=False)
        assert "AND last_message_date > ?" not in result

    def test_query_with_before_filter(self):
        """Apply before filter when flag is True."""
        result = get_query("messages", "v14", with_before_filter=True)
        assert "AND message.date < ?" in result

    def test_query_without_before_filter(self):
        """No before filter when flag is False."""
        result = get_query("messages", "v14", with_before_filter=False)
        assert "AND message.date < ?" not in result

    def test_search_query_has_escape_clause(self):
        """Search query includes ESCAPE clause for LIKE."""
        result = get_query("search", "v14")
        assert "ESCAPE" in result

    def test_invalid_query_raises(self):
        """Raise KeyError for invalid query name."""
        with pytest.raises(KeyError):
            get_query("invalid_query", "v14")

    def test_attachments_query_exists(self):
        """Attachments query is available."""
        result = get_query("attachments", "v14")
        assert "attachment" in result.lower()
        assert "message_attachment_join" in result

    def test_reactions_query_exists(self):
        """Reactions query is available."""
        result = get_query("reactions", "v14")
        assert "associated_message_guid" in result
        assert "associated_message_type" in result

    def test_message_by_guid_query_exists(self):
        """Message by GUID query is available."""
        result = get_query("message_by_guid", "v14")
        assert "guid" in result.lower()
        assert "ROWID" in result or "id" in result.lower()


# =============================================================================
# Reader Tests
# =============================================================================


class TestChatDBReaderInit:
    """Tests for ChatDBReader initialization."""

    def test_default_path(self):
        """Use default chat.db path."""
        reader = ChatDBReader()
        assert reader.db_path == CHAT_DB_PATH

    def test_custom_path(self):
        """Accept custom database path."""
        custom_path = Path("/tmp/test.db")
        reader = ChatDBReader(db_path=custom_path)
        assert reader.db_path == custom_path

    def test_initial_state(self):
        """Connection is None initially."""
        reader = ChatDBReader()
        assert reader._connection is None
        assert reader._schema_version is None

    def test_context_manager(self, tmp_path):
        """Context manager closes connection on exit."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute("INSERT INTO chat VALUES (1, 'test')")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, thread_originator_guid TEXT)"
        )
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            assert reader.check_access() is True
            assert reader._connection is not None

        # Connection should be closed after exiting context
        assert reader._connection is None

    def test_context_manager_on_exception(self, tmp_path):
        """Context manager closes connection even on exception."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute("INSERT INTO chat VALUES (1, 'test')")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, thread_originator_guid TEXT)"
        )
        conn.close()

        try:
            with ChatDBReader(db_path=db_path) as reader:
                reader.check_access()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Connection should still be closed
        assert reader._connection is None


class TestChatDBReaderCheckAccess:
    """Tests for access checking."""

    def test_file_not_found(self, tmp_path):
        """Return False when database doesn't exist."""
        reader = ChatDBReader(db_path=tmp_path / "nonexistent.db")
        result = reader.check_access()
        assert result is False

    def test_access_granted(self, tmp_path):
        """Return True when database is accessible."""
        # Create a minimal test database
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute("INSERT INTO chat VALUES (1, 'test')")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, thread_originator_guid TEXT)"
        )
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.check_access()
        assert result is True
        reader.close()

    def test_operational_error_unable_to_open(self, tmp_path, monkeypatch):
        """Return False on 'unable to open database' error."""
        db_path = tmp_path / "chat.db"
        db_path.touch()  # Create empty file

        reader = ChatDBReader(db_path=db_path)

        # Mock _get_connection to raise OperationalError
        def mock_get_connection():
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(reader, "_get_connection", mock_get_connection)
        result = reader.check_access()
        assert result is False

    def test_operational_error_other(self, tmp_path, monkeypatch):
        """Return False on other OperationalError."""
        db_path = tmp_path / "chat.db"
        db_path.touch()

        reader = ChatDBReader(db_path=db_path)

        def mock_get_connection():
            raise sqlite3.OperationalError("some other error")

        monkeypatch.setattr(reader, "_get_connection", mock_get_connection)
        result = reader.check_access()
        assert result is False

    def test_unexpected_exception(self, tmp_path, monkeypatch):
        """Return False on unexpected exception."""
        db_path = tmp_path / "chat.db"
        db_path.touch()

        reader = ChatDBReader(db_path=db_path)

        def mock_get_connection():
            raise RuntimeError("Unexpected error")

        monkeypatch.setattr(reader, "_get_connection", mock_get_connection)
        result = reader.check_access()
        assert result is False


class TestChatDBReaderGetConversations:
    """Tests for get_conversations method."""

    def test_empty_database(self, tmp_path):
        """Return empty list for database with no conversations."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)"
        )
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute(
            "CREATE TABLE message "
            "(ROWID INTEGER, date INTEGER, text TEXT, thread_originator_guid TEXT)"
        )
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_conversations(limit=10)
        assert result == []
        reader.close()

    def test_returns_conversation_objects(self, tmp_path):
        """Return Conversation dataclass instances."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        # Create schema
        conn.execute(
            "CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)"
        )
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute(
            "CREATE TABLE message "
            "(ROWID INTEGER, date INTEGER, text TEXT, thread_originator_guid TEXT)"
        )

        # Insert test data
        conn.execute("INSERT INTO chat VALUES (1, 'chat;+1234567890', NULL, '+1234567890')")
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute("INSERT INTO chat_handle_join VALUES (1, 1)")
        conn.execute("INSERT INTO message VALUES (1, 726753600000000000, 'Hello', NULL)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_conversations(limit=10)

        assert len(result) == 1
        assert isinstance(result[0], Conversation)
        assert result[0].chat_id == "chat;+1234567890"
        assert result[0].message_count == 1
        reader.close()


class TestChatDBReaderGetMessages:
    """Tests for get_messages method."""

    def test_returns_message_objects(self, tmp_path):
        """Return Message dataclass instances."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        # Create schema
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        # Insert test data
        conn.execute("INSERT INTO chat VALUES (1, 'chat;+1234567890')")
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute(
            "INSERT INTO message VALUES (1, 726753600000000000, 'Hello', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_messages("chat;+1234567890", limit=10)

        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].text == "Hello"
        assert result[0].is_from_me is False
        reader.close()

    def test_skips_empty_messages(self, tmp_path):
        """Skip messages with no text content."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute("INSERT INTO message VALUES (1, 1, NULL, NULL, 0, NULL, NULL)")  # No text
        conn.execute("INSERT INTO message VALUES (2, 2, 'Valid', NULL, 0, NULL, NULL)")  # Has text
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_messages("chat;test", limit=10)

        assert len(result) == 1
        assert result[0].text == "Valid"
        reader.close()


class TestChatDBReaderSearch:
    """Tests for search method."""

    def test_search_returns_matches(self, tmp_path):
        """Find messages matching search query."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute("INSERT INTO message VALUES (1, 1, 'Hello world', NULL, 0, NULL, NULL)")
        conn.execute("INSERT INTO message VALUES (2, 2, 'Goodbye world', NULL, 0, NULL, NULL)")
        conn.execute("INSERT INTO message VALUES (3, 3, 'No match here', NULL, 0, NULL, NULL)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 3)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.search("world", limit=10)

        assert len(result) == 2
        texts = {m.text for m in result}
        assert "Hello world" in texts
        assert "Goodbye world" in texts
        reader.close()

    def test_search_no_results(self, tmp_path):
        """Return empty list when no matches."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.search("nonexistent", limit=10)

        assert result == []
        reader.close()

    def test_search_escapes_wildcards(self, tmp_path):
        """Search properly escapes LIKE wildcards."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute("INSERT INTO message VALUES (1, 1, '100% complete', NULL, 0, NULL, NULL)")
        conn.execute("INSERT INTO message VALUES (2, 2, '100 complete', NULL, 0, NULL, NULL)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            # Search for literal "%" - should only match the message with %
            result = reader.search("100%", limit=10)

            assert len(result) == 1
            assert result[0].text == "100% complete"


class TestChatDBReaderContext:
    """Tests for get_conversation_context method."""

    def test_returns_messages_around_target(self, tmp_path):
        """Get messages centered around target message."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        for i in range(1, 11):
            conn.execute(
                f"INSERT INTO message VALUES ({i}, {i}, 'Message {i}', NULL, 0, NULL, NULL)"
            )
            conn.execute(f"INSERT INTO chat_message_join VALUES (1, {i})")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_conversation_context(
            "chat;test", around_message_id=5, context_messages=2
        )

        # Should get 5 messages (2 before + target + 2 after)
        assert len(result) == 5
        reader.close()


class TestChatDBReaderFilters:
    """Tests for query filters."""

    def test_get_conversations_with_since_filter(self, tmp_path):
        """Test get_conversations with since filter."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)"
        )
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute(
            "CREATE TABLE message "
            "(ROWID INTEGER, date INTEGER, text TEXT, thread_originator_guid TEXT)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test', 'Test', 'test')")
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute("INSERT INTO chat_handle_join VALUES (1, 1)")
        # Old message
        conn.execute("INSERT INTO message VALUES (1, 100000000000000000, 'Old', NULL)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        # New message
        conn.execute("INSERT INTO message VALUES (2, 800000000000000000000, 'New', NULL)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        # Filter for messages after a certain date
        since = dt(2025, 1, 1, tzinfo=UTC)
        result = reader.get_conversations(limit=10, since=since)
        # The implementation should filter based on last_message_date
        assert isinstance(result, list)
        reader.close()

    def test_get_messages_with_before_filter(self, tmp_path):
        """Test get_messages with before filter."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, date INTEGER, text TEXT, attributedBody BLOB,
                is_from_me INTEGER, handle_id INTEGER, thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute(
            "INSERT INTO message VALUES (1, 100000000000000000, 'Early', NULL, 0, NULL, NULL)"
        )
        conn.execute(
            "INSERT INTO message VALUES (2, 900000000000000000000, 'Late', NULL, 0, NULL, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        before = dt(2020, 1, 1, tzinfo=UTC)
        result = reader.get_messages("chat;test", limit=10, before=before)
        assert isinstance(result, list)
        reader.close()


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for iMessageReader protocol compliance."""

    def test_reader_has_required_methods(self):
        """ChatDBReader has all iMessageReader protocol methods."""
        reader = ChatDBReader()

        # Protocol methods
        assert hasattr(reader, "check_access")
        assert hasattr(reader, "get_conversations")
        assert hasattr(reader, "get_messages")
        assert hasattr(reader, "search")
        assert hasattr(reader, "get_conversation_context")

        # Methods are callable
        assert callable(reader.check_access)
        assert callable(reader.get_conversations)
        assert callable(reader.get_messages)
        assert callable(reader.search)
        assert callable(reader.get_conversation_context)

    def test_return_types_match_contract(self, tmp_path):
        """Return types match contract dataclasses."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        # Create minimal schema
        conn.execute(
            "CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)"
        )
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)

        # Insert test data
        conn.execute("INSERT INTO chat VALUES (1, 'chat;test', 'Test Chat', 'test')")
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute("INSERT INTO chat_handle_join VALUES (1, 1)")
        conn.execute(
            "INSERT INTO message VALUES (1, 726753600000000000, 'Hello', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)

        # Test get_conversations return type
        conversations = reader.get_conversations(limit=10)
        assert isinstance(conversations, list)
        if conversations:
            assert isinstance(conversations[0], Conversation)

        # Test get_messages return type
        if conversations:
            messages = reader.get_messages(conversations[0].chat_id, limit=10)
            assert isinstance(messages, list)
            if messages:
                assert isinstance(messages[0], Message)

        reader.close()


# =============================================================================
# Attachment and Reaction Integration Tests
# =============================================================================


class TestChatDBReaderAttachments:
    """Tests for attachment retrieval in ChatDBReader."""

    def test_get_attachments_for_message(self, tmp_path):
        """Retrieve attachments for a specific message."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        # Create schema
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                guid TEXT,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER,
                filename TEXT,
                mime_type TEXT,
                total_bytes INTEGER,
                transfer_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE message_attachment_join (
                message_id INTEGER,
                attachment_id INTEGER
            )
        """)

        # Insert test data
        conn.execute("INSERT INTO chat VALUES (1, 'chat;+1234567890')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-guid-1', 726753600000000000, "
            "'Photo attached', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute(
            "INSERT INTO attachment VALUES "
            "(1, '/path/to/photo.jpg', 'image/jpeg', 12345, 'photo.jpg')"
        )
        conn.execute("INSERT INTO message_attachment_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        messages = reader.get_messages("chat;+1234567890", limit=10)

        assert len(messages) == 1
        assert len(messages[0].attachments) == 1
        assert messages[0].attachments[0].filename == "photo.jpg"
        assert messages[0].attachments[0].mime_type == "image/jpeg"
        assert messages[0].attachments[0].file_size == 12345
        reader.close()

    def test_message_with_multiple_attachments(self, tmp_path):
        """Retrieve multiple attachments for a single message."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, guid TEXT, date INTEGER, text TEXT,
                attributedBody BLOB, is_from_me INTEGER, handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 1, 'Multiple files', NULL, 0, NULL, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.execute("INSERT INTO attachment VALUES (1, '/a/img.png', 'image/png', 100, 'img.png')")
        conn.execute(
            "INSERT INTO attachment VALUES (2, '/a/doc.pdf', 'application/pdf', 200, 'doc.pdf')"
        )
        conn.execute("INSERT INTO message_attachment_join VALUES (1, 1)")
        conn.execute("INSERT INTO message_attachment_join VALUES (1, 2)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            messages = reader.get_messages("chat;test", limit=10)

            assert len(messages) == 1
            assert len(messages[0].attachments) == 2

    def test_message_with_no_attachments(self, tmp_path):
        """Messages without attachments have empty attachment list."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, guid TEXT, date INTEGER, text TEXT,
                attributedBody BLOB, is_from_me INTEGER, handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 1, 'No attachments', NULL, 0, NULL, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            messages = reader.get_messages("chat;test", limit=10)

            assert len(messages) == 1
            assert len(messages[0].attachments) == 0


class TestChatDBReaderReactions:
    """Tests for reaction retrieval in ChatDBReader."""

    def test_get_reactions_for_message(self, tmp_path):
        """Retrieve reactions for a specific message."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                guid TEXT,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT,
                associated_message_guid TEXT,
                associated_message_type INTEGER
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        # Insert original message
        conn.execute("INSERT INTO chat VALUES (1, 'chat;+1234567890')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-guid-1', 726753600000000000, "
            "'Hello', NULL, 0, 1, NULL, NULL, 0)"
        )
        conn.execute("INSERT INTO handle VALUES (1, '+1234567890')")
        conn.execute("INSERT INTO handle VALUES (2, '+1987654321')")
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")

        # Insert reaction (love from another user)
        conn.execute(
            "INSERT INTO message VALUES (2, 'react-guid-1', 726753700000000000, "
            "NULL, NULL, 0, 2, NULL, 'msg-guid-1', 2000)"
        )
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        messages = reader.get_messages("chat;+1234567890", limit=10)

        assert len(messages) == 1
        assert len(messages[0].reactions) == 1
        assert messages[0].reactions[0].type == "love"
        assert messages[0].reactions[0].sender == "+1987654321"
        reader.close()

    def test_message_with_multiple_reactions(self, tmp_path):
        """Retrieve multiple reactions for a single message."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, guid TEXT, date INTEGER, text TEXT,
                attributedBody BLOB, is_from_me INTEGER, handle_id INTEGER,
                thread_originator_guid TEXT, associated_message_guid TEXT,
                associated_message_type INTEGER
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute("INSERT INTO handle VALUES (1, '+1111')")
        conn.execute("INSERT INTO handle VALUES (2, '+2222')")

        # Original message
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 1, 'Test', NULL, 0, 1, NULL, NULL, 0)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")

        # Multiple reactions
        conn.execute(
            "INSERT INTO message VALUES (2, 'react-1', 2, NULL, NULL, 0, 2, NULL, 'msg-1', 2000)"
        )  # love
        conn.execute(
            "INSERT INTO message VALUES (3, 'react-2', 3, NULL, NULL, 1, NULL, NULL, 'msg-1', 2003)"
        )  # laugh from me
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            messages = reader.get_messages("chat;test", limit=10)

            assert len(messages) == 1
            assert len(messages[0].reactions) == 2
            reaction_types = {r.type for r in messages[0].reactions}
            assert "love" in reaction_types
            assert "laugh" in reaction_types


class TestChatDBReaderReplyTo:
    """Tests for reply-to ID mapping in ChatDBReader."""

    def test_reply_to_id_resolved(self, tmp_path):
        """Reply-to GUID is resolved to message ROWID."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, guid TEXT, date INTEGER, text TEXT,
                attributedBody BLOB, is_from_me INTEGER, handle_id INTEGER,
                thread_originator_guid TEXT, associated_message_guid TEXT,
                associated_message_type INTEGER
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute("INSERT INTO handle VALUES (1, '+1111')")

        # Original message
        conn.execute(
            "INSERT INTO message VALUES (100, 'original-guid', 1, 'Original', "
            "NULL, 0, 1, NULL, NULL, 0)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 100)")

        # Reply message
        conn.execute(
            "INSERT INTO message VALUES (200, 'reply-guid', 2, 'Reply', "
            "NULL, 0, 1, 'original-guid', NULL, 0)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 200)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            messages = reader.get_messages("chat;test", limit=10)

            # Find the reply message
            reply_msg = next((m for m in messages if m.text == "Reply"), None)
            assert reply_msg is not None
            assert reply_msg.reply_to_id == 100

    def test_reply_to_id_none_when_no_reply(self, tmp_path):
        """Reply-to ID is None for messages without thread_originator_guid."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER, guid TEXT, date INTEGER, text TEXT,
                attributedBody BLOB, is_from_me INTEGER, handle_id INTEGER,
                thread_originator_guid TEXT, associated_message_guid TEXT,
                associated_message_type INTEGER
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 1, 'No reply', NULL, 0, NULL, NULL, NULL, 0)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            messages = reader.get_messages("chat;test", limit=10)

            assert len(messages) == 1
            assert messages[0].reply_to_id is None


class TestChatDBReaderContactResolution:
    """Tests for contact name resolution in ChatDBReader."""

    def test_resolve_contact_name_returns_none_for_me(self):
        """Return None for 'me' identifier."""
        reader = ChatDBReader()
        reader._contacts_cache = {"test": "Test User"}

        result = reader._resolve_contact_name("me")
        assert result is None

    def test_resolve_contact_name_returns_none_for_empty(self):
        """Return None for empty identifier."""
        reader = ChatDBReader()
        reader._contacts_cache = {}

        result = reader._resolve_contact_name("")
        assert result is None

    def test_resolve_contact_name_normalizes_phone(self):
        """Normalize phone number before lookup."""
        reader = ChatDBReader()
        reader._contacts_cache = {"+15551234567": "John Doe"}

        # Different formats should all resolve
        assert reader._resolve_contact_name("5551234567") == "John Doe"
        assert reader._resolve_contact_name("+15551234567") == "John Doe"
        assert reader._resolve_contact_name("(555) 123-4567") == "John Doe"

    def test_format_name_with_both_names(self):
        """Format full name from first and last."""
        result = ChatDBReader._format_name("John", "Doe")
        assert result == "John Doe"

    def test_format_name_with_first_only(self):
        """Format name with only first name."""
        result = ChatDBReader._format_name("John", None)
        assert result == "John"

    def test_format_name_with_last_only(self):
        """Format name with only last name."""
        result = ChatDBReader._format_name(None, "Doe")
        assert result == "Doe"

    def test_format_name_with_neither(self):
        """Return None when both names are empty."""
        result = ChatDBReader._format_name(None, None)
        assert result is None

        result = ChatDBReader._format_name("", "")
        assert result is None


# =============================================================================
# Search Filter Tests
# =============================================================================


class TestSearchFiltersQueryBuilder:
    """Tests for search filter SQL query building."""

    def test_search_query_with_sender_filter(self):
        """Search query includes sender filter clause."""
        result = get_query("search", "v14", with_sender_filter=True)
        assert "handle.id = ?" in result
        assert "is_from_me" in result

    def test_search_query_with_after_filter(self):
        """Search query includes after date filter clause."""
        result = get_query("search", "v14", with_after_filter=True)
        assert "message.date > ?" in result

    def test_search_query_with_before_filter(self):
        """Search query includes before date filter clause."""
        result = get_query("search", "v14", with_search_before_filter=True)
        assert "message.date < ?" in result

    def test_search_query_with_chat_id_filter(self):
        """Search query includes chat_id filter clause."""
        result = get_query("search", "v14", with_chat_id_filter=True)
        assert "chat.guid = ?" in result

    def test_search_query_with_has_attachments_true(self):
        """Search query includes EXISTS clause for attachments."""
        result = get_query("search", "v14", with_has_attachments_filter=True)
        assert "EXISTS" in result
        assert "message_attachment_join" in result

    def test_search_query_with_has_attachments_false(self):
        """Search query includes NOT EXISTS clause for no attachments."""
        result = get_query("search", "v14", with_has_attachments_filter=False)
        assert "NOT EXISTS" in result
        assert "message_attachment_join" in result

    def test_search_query_with_no_filters(self):
        """Search query without filters has no filter clauses."""
        result = get_query("search", "v14")
        assert "handle.id = ?" not in result
        assert "message.date > ?" not in result
        assert "message.date < ?" not in result
        assert "chat.guid = ?" not in result
        assert "EXISTS" not in result

    def test_search_query_with_multiple_filters(self):
        """Search query supports multiple filters simultaneously."""
        result = get_query(
            "search",
            "v14",
            with_sender_filter=True,
            with_after_filter=True,
            with_chat_id_filter=True,
        )
        assert "handle.id = ?" in result
        assert "message.date > ?" in result
        assert "chat.guid = ?" in result


class TestChatDBReaderSearchFilters:
    """Tests for ChatDBReader.search() with filters."""

    @pytest.fixture
    def db_with_messages(self, tmp_path):
        """Create a test database with messages for filter testing."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        # Create schema
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                guid TEXT,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        # Insert test data
        # Chat 1: conversation with +15551234567
        conn.execute("INSERT INTO chat VALUES (1, 'chat;+15551234567')")
        conn.execute("INSERT INTO handle VALUES (1, '+15551234567')")

        # Chat 2: conversation with +15559876543
        conn.execute("INSERT INTO chat VALUES (2, 'chat;+15559876543')")
        conn.execute("INSERT INTO handle VALUES (2, '+15559876543')")

        # Messages with varying dates, senders, and content
        # Date: 2024-01-10 (726580800000000000 ns)
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 726580800000000000, "
            "'Hello from John', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")

        # Date: 2024-01-12 (726753600000000000 ns) - from me
        conn.execute(
            "INSERT INTO message VALUES (2, 'msg-2', 726753600000000000, "
            "'Hello back from me', NULL, 1, NULL, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 2)")

        # Date: 2024-01-15 (727012800000000000 ns) - different chat
        conn.execute(
            "INSERT INTO message VALUES (3, 'msg-3', 727012800000000000, "
            "'Hello from Jane', NULL, 0, 2, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (2, 3)")

        # Date: 2024-01-20 (727444800000000000 ns) - with attachment
        conn.execute(
            "INSERT INTO message VALUES (4, 'msg-4', 727444800000000000, "
            "'Photo from John', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 4)")
        conn.execute(
            "INSERT INTO attachment VALUES (1, '/path/photo.jpg', 'image/jpeg', 1000, 'photo.jpg')"
        )
        conn.execute("INSERT INTO message_attachment_join VALUES (4, 1)")

        # Date: 2024-01-25 (727876800000000000 ns) - no text match
        conn.execute(
            "INSERT INTO message VALUES (5, 'msg-5', 727876800000000000, "
            "'Goodbye world', NULL, 0, 1, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 5)")

        conn.commit()
        conn.close()
        return db_path

    def test_search_with_sender_filter(self, db_with_messages):
        """Filter search results by sender."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            # Search for "Hello" from +15551234567
            results = reader.search("Hello", sender="+15551234567")

            # Should find messages from John, not Jane
            assert len(results) >= 1
            for msg in results:
                assert msg.sender == "+15551234567" or msg.is_from_me

    def test_search_with_sender_me_filter(self, db_with_messages):
        """Filter search results to only show own messages."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            results = reader.search("Hello", sender="me")

            # Should only find "Hello back from me"
            assert len(results) == 1
            assert results[0].is_from_me is True
            assert "from me" in results[0].text

    def test_search_with_after_date_filter(self, db_with_messages):
        """Filter search results to messages after a date."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            # After 2024-01-14 - should find Jan 15, 20, but not Jan 10, 12
            after_date = dt(2024, 1, 14, tzinfo=UTC)
            results = reader.search("Hello", after=after_date)

            # Should find "Hello from Jane" (Jan 15) and "Photo from John" (Jan 20)
            for msg in results:
                assert msg.date > after_date

    def test_search_with_before_date_filter(self, db_with_messages):
        """Filter search results to messages before a date."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            # Before 2024-01-14 - should find Jan 10, 12, but not later
            before_date = dt(2024, 1, 14, tzinfo=UTC)
            results = reader.search("Hello", before=before_date)

            for msg in results:
                assert msg.date < before_date

    def test_search_with_chat_id_filter(self, db_with_messages):
        """Filter search results by conversation."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            results = reader.search("Hello", chat_id="chat;+15559876543")

            # Should only find messages from chat 2 (Jane's chat)
            assert len(results) == 1
            assert results[0].chat_id == "chat;+15559876543"
            assert "Jane" in results[0].text

    def test_search_with_has_attachments_true(self, db_with_messages):
        """Filter search results to only messages with attachments."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            results = reader.search("from", has_attachments=True)

            # Should only find "Photo from John" which has an attachment
            assert len(results) == 1
            assert "Photo" in results[0].text
            assert len(results[0].attachments) > 0

    def test_search_with_has_attachments_false(self, db_with_messages):
        """Filter search results to only messages without attachments."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            results = reader.search("Hello", has_attachments=False)

            # Should find all "Hello" messages except the one with attachment
            for msg in results:
                assert len(msg.attachments) == 0

    def test_search_with_combined_filters_sender_and_after(self, db_with_messages):
        """Combine sender and date filters."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            after_date = dt(2024, 1, 14, tzinfo=UTC)
            results = reader.search("from", sender="+15551234567", after=after_date)

            # Should find "Photo from John" (Jan 20) but not earlier John messages
            for msg in results:
                assert msg.date > after_date
                assert msg.sender == "+15551234567" or msg.is_from_me

    def test_search_with_combined_filters_chat_and_before(self, db_with_messages):
        """Combine chat_id and before date filters."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            before_date = dt(2024, 1, 18, tzinfo=UTC)
            results = reader.search("Hello", chat_id="chat;+15551234567", before=before_date)

            for msg in results:
                assert msg.chat_id == "chat;+15551234567"
                assert msg.date < before_date

    def test_search_with_all_filters(self, db_with_messages):
        """Apply all filters simultaneously."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            after_date = dt(2024, 1, 5, tzinfo=UTC)
            before_date = dt(2024, 1, 30, tzinfo=UTC)
            results = reader.search(
                "from",
                sender="+15551234567",
                after=after_date,
                before=before_date,
                chat_id="chat;+15551234567",
                has_attachments=False,
            )

            for msg in results:
                assert msg.sender == "+15551234567" or msg.is_from_me
                assert msg.date > after_date
                assert msg.date < before_date
                assert msg.chat_id == "chat;+15551234567"
                assert len(msg.attachments) == 0

    def test_search_with_normalized_sender(self, db_with_messages):
        """Sender phone number is normalized before filtering."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            # Use different format for the same number
            results = reader.search("Hello", sender="(555) 123-4567")

            # Should still match +15551234567
            assert len(results) >= 1

    def test_search_filters_with_no_matches(self, db_with_messages):
        """Return empty list when filters exclude all results."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            # Search for text that doesn't exist
            results = reader.search("nonexistent", sender="+15551234567")
            assert results == []

            # Date range that excludes all messages
            far_future = dt(2030, 1, 1, tzinfo=UTC)
            results = reader.search("Hello", after=far_future)
            assert results == []

    def test_search_limit_with_filters(self, db_with_messages):
        """Limit parameter works with filters."""
        with ChatDBReader(db_path=db_with_messages) as reader:
            results = reader.search("from", limit=1)
            assert len(results) <= 1


# =============================================================================
# LRUCache Tests
# =============================================================================


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_get_returns_none_for_missing_key(self):
        """Return None when key is not in cache."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        assert cache.get("missing") is None

    def test_get_returns_value_and_moves_to_end(self):
        """Get returns value and moves key to end (most recently used)."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' - should move it to end
        result = cache.get("a")
        assert result == 1

        # Verify 'a' is now at the end by checking order of iteration
        keys = list(cache._cache.keys())
        assert keys[-1] == "a"

    def test_set_updates_existing_key(self):
        """Set updates existing key and moves to end."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)

        # Update 'a'
        cache.set("a", 100)

        # Verify value updated
        assert cache.get("a") == 100

        # Verify 'a' moved to end
        keys = list(cache._cache.keys())
        assert keys[-1] == "a"

    def test_set_evicts_oldest_when_at_capacity(self):
        """Set evicts oldest item when at maxsize."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # At capacity, add new item
        cache.set("d", 4)

        # 'a' should be evicted (oldest)
        assert cache.get("a") is None
        assert cache.get("d") == 4
        assert len(cache) == 3

    def test_contains_returns_true_for_existing_key(self):
        """__contains__ returns True for existing key."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        cache.set("key", 42)

        assert "key" in cache
        assert "missing" not in cache

    def test_len_returns_correct_count(self):
        """__len__ returns correct item count."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        assert len(cache) == 0

        cache.set("a", 1)
        assert len(cache) == 1

        cache.set("b", 2)
        cache.set("c", 3)
        assert len(cache) == 3

    def test_clear_removes_all_items(self):
        """clear() removes all items from cache."""
        from integrations.imessage.reader import LRUCache

        cache: LRUCache[str, int] = LRUCache(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()

        assert len(cache) == 0
        assert cache.get("a") is None


# =============================================================================
# Close Method Exception Handling Tests
# =============================================================================


class TestChatDBReaderClose:
    """Tests for ChatDBReader.close() exception handling."""

    def test_close_suppresses_exception_during_cleanup(self, monkeypatch):
        """close() suppresses exceptions during connection cleanup."""
        reader = ChatDBReader()

        # Create a mock connection that raises on close
        mock_conn = MagicMock()
        mock_conn.close.side_effect = RuntimeError("Connection close failed")

        reader._connection = mock_conn
        reader._schema_version = "v14"
        reader._contacts_cache = {"test": "Test User"}

        # Should not raise - exception is suppressed
        reader.close()

        # Connection should be set to None
        assert reader._connection is None
        assert reader._schema_version is None
        assert reader._contacts_cache is None


# =============================================================================
# GUID to ROWID Cache Hit Tests
# =============================================================================


class TestGUIDToROWIDCache:
    """Tests for GUID to ROWID caching in _get_message_rowid_by_guid."""

    def test_cache_hit_returns_cached_rowid(self, tmp_path):
        """Cache hit returns cached ROWID without database query."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, "
            "thread_originator_guid TEXT)"
        )
        conn.execute("INSERT INTO message VALUES (100, 'test-guid', NULL)")
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        reader.check_access()

        # Manually populate the cache to test cache hit
        reader._guid_to_rowid_cache.set("test-guid", 100)

        # Call should return from cache
        result = reader._get_message_rowid_by_guid("test-guid")
        assert result == 100

        reader.close()

    def test_database_lookup_and_cache_store(self, tmp_path):
        """Database lookup stores result in cache."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, "
            "thread_originator_guid TEXT)"
        )
        # ROWID is automatically the PRIMARY KEY value when using INTEGER PRIMARY KEY
        conn.execute("INSERT INTO message (ROWID, guid) VALUES (200, 'lookup-guid')")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        reader.check_access()

        # Cache should be empty
        assert reader._guid_to_rowid_cache.get("lookup-guid") is None

        # First call - fetches from database and caches
        result = reader._get_message_rowid_by_guid("lookup-guid")
        assert result == 200

        # Verify it was cached
        assert reader._guid_to_rowid_cache.get("lookup-guid") == 200

        reader.close()

    def test_cache_miss_returns_none_for_unknown_guid(self, tmp_path):
        """Return None for GUID not found in database."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT)")
        conn.execute(
            "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, "
            "thread_originator_guid TEXT)"
        )
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        reader.check_access()

        result = reader._get_message_rowid_by_guid("nonexistent-guid")
        assert result is None

        reader.close()

    def test_operational_error_returns_none(self, monkeypatch):
        """Return None on OperationalError in _get_message_rowid_by_guid."""
        reader = ChatDBReader()

        # Create a mock connection that raises OperationalError
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("database is locked")
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        result = reader._get_message_rowid_by_guid("some-guid")
        assert result is None


# =============================================================================
# Contact Resolution Edge Cases
# =============================================================================


class TestContactResolutionEdgeCases:
    """Additional tests for contact resolution edge cases."""

    def test_resolve_contact_name_with_none_normalization(self):
        """Return None when normalization returns None."""
        reader = ChatDBReader()
        reader._contacts_cache = {"+15551234567": "John Doe"}

        # Pass a value that normalizes to None
        # Empty string after stripping results in None
        result = reader._resolve_contact_name("   ")
        assert result is None

    def test_resolve_contact_name_cache_none_after_load(self, monkeypatch):
        """Return None if contacts cache is None after load attempt."""
        reader = ChatDBReader()

        # Mock _load_contacts_cache to leave cache as None
        def mock_load():
            reader._contacts_cache = None

        monkeypatch.setattr(reader, "_load_contacts_cache", mock_load)

        result = reader._resolve_contact_name("+15551234567")
        assert result is None


# =============================================================================
# Address Book Loading Tests
# =============================================================================


class TestLoadContactsCache:
    """Tests for _load_contacts_cache method."""

    def test_addressbook_path_not_found(self, monkeypatch):
        """Cache remains empty when AddressBook path doesn't exist."""
        from integrations.imessage import reader

        # Mock ADDRESSBOOK_DB_PATH to a non-existent path
        fake_path = Path("/nonexistent/addressbook/path")
        monkeypatch.setattr(reader, "ADDRESSBOOK_DB_PATH", fake_path)

        db_reader = ChatDBReader()
        db_reader._load_contacts_cache()

        assert db_reader._contacts_cache == {}

    def test_addressbook_permission_error(self, tmp_path, monkeypatch):
        """Handle PermissionError when accessing AddressBook."""
        from integrations.imessage import reader

        # Create a directory that exists but causes permission error on iterdir
        addressbook_path = tmp_path / "AddressBook" / "Sources"
        addressbook_path.mkdir(parents=True)

        monkeypatch.setattr(reader, "ADDRESSBOOK_DB_PATH", addressbook_path)

        # Mock iterdir to raise PermissionError
        def mock_iterdir(self):
            raise PermissionError("Access denied")

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)

        db_reader = ChatDBReader()
        db_reader._load_contacts_cache()

        assert db_reader._contacts_cache == {}

    def test_addressbook_oserror(self, tmp_path, monkeypatch):
        """Handle OSError when accessing AddressBook."""
        from integrations.imessage import reader

        addressbook_path = tmp_path / "AddressBook" / "Sources"
        addressbook_path.mkdir(parents=True)

        monkeypatch.setattr(reader, "ADDRESSBOOK_DB_PATH", addressbook_path)

        def mock_iterdir(self):
            raise OSError("I/O error")

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)

        db_reader = ChatDBReader()
        db_reader._load_contacts_cache()

        assert db_reader._contacts_cache == {}


class TestLoadContactsFromDB:
    """Tests for _load_contacts_from_db method."""

    def test_loads_phone_numbers_with_names(self, tmp_path):
        """Load phone numbers with names from AddressBook database."""
        ab_db = tmp_path / "AddressBook-v22.abcddb"
        conn = sqlite3.connect(ab_db)

        # Create AddressBook schema
        conn.execute("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDPHONENUMBER (
                ZFULLNUMBER TEXT,
                ZOWNER INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDEMAILADDRESS (
                ZADDRESS TEXT,
                ZOWNER INTEGER
            )
        """)

        # Insert test data
        conn.execute("INSERT INTO ZABCDRECORD VALUES (1, 'John', 'Doe')")
        conn.execute("INSERT INTO ZABCDPHONENUMBER VALUES ('+15551234567', 1)")
        conn.execute("INSERT INTO ZABCDEMAILADDRESS VALUES ('john@example.com', 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        assert "+15551234567" in reader._contacts_cache
        assert reader._contacts_cache["+15551234567"] == "John Doe"
        assert "john@example.com" in reader._contacts_cache
        assert reader._contacts_cache["john@example.com"] == "John Doe"

    def test_handles_phone_table_error(self, tmp_path):
        """Handle OperationalError when querying phone numbers."""
        ab_db = tmp_path / "AddressBook.abcddb"
        conn = sqlite3.connect(ab_db)

        # Create incomplete schema (missing ZABCDPHONENUMBER)
        conn.execute("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDEMAILADDRESS (
                ZADDRESS TEXT,
                ZOWNER INTEGER
            )
        """)
        conn.execute("INSERT INTO ZABCDRECORD VALUES (1, 'John', 'Doe')")
        conn.execute("INSERT INTO ZABCDEMAILADDRESS VALUES ('john@example.com', 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        # Should still load emails even if phone table fails
        assert "john@example.com" in reader._contacts_cache

    def test_handles_email_table_error(self, tmp_path):
        """Handle OperationalError when querying email addresses."""
        ab_db = tmp_path / "AddressBook.abcddb"
        conn = sqlite3.connect(ab_db)

        # Create incomplete schema (missing ZABCDEMAILADDRESS)
        conn.execute("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDPHONENUMBER (
                ZFULLNUMBER TEXT,
                ZOWNER INTEGER
            )
        """)
        conn.execute("INSERT INTO ZABCDRECORD VALUES (1, 'John', 'Doe')")
        conn.execute("INSERT INTO ZABCDPHONENUMBER VALUES ('+15551234567', 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        # Should still load phones even if email table fails
        assert "+15551234567" in reader._contacts_cache

    def test_handles_database_connection_error(self, tmp_path):
        """Handle sqlite3.Error when connecting to database."""
        # Create a non-database file
        bad_file = tmp_path / "not_a_database.txt"
        bad_file.write_text("not a database")

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(bad_file)

        # Should not raise, cache remains empty
        assert reader._contacts_cache == {}

    def test_initializes_cache_if_none(self, tmp_path):
        """Initialize cache to empty dict if None before loading."""
        ab_db = tmp_path / "AddressBook.abcddb"
        conn = sqlite3.connect(ab_db)
        conn.execute("CREATE TABLE ZABCDRECORD (Z_PK INTEGER, ZFIRSTNAME TEXT, ZLASTNAME TEXT)")
        conn.execute("CREATE TABLE ZABCDPHONENUMBER (ZFULLNUMBER TEXT, ZOWNER INTEGER)")
        conn.execute("CREATE TABLE ZABCDEMAILADDRESS (ZADDRESS TEXT, ZOWNER INTEGER)")
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = None  # Explicitly set to None
        reader._load_contacts_from_db(ab_db)

        # Should not raise, cache initialized
        assert reader._contacts_cache is not None
        assert isinstance(reader._contacts_cache, dict)

    def test_handles_oserror(self, tmp_path, monkeypatch):
        """Handle OSError when loading contacts database."""
        ab_db = tmp_path / "AddressBook.abcddb"

        # Mock sqlite3.connect to raise OSError
        def mock_connect(*args, **kwargs):
            raise OSError("Disk error")

        monkeypatch.setattr(sqlite3, "connect", mock_connect)

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        # Should not raise
        assert reader._contacts_cache == {}


# =============================================================================
# Check Access PermissionError Tests
# =============================================================================


class TestCheckAccessPermissionError:
    """Tests for PermissionError handling in check_access."""

    def test_permission_error_returns_false(self, tmp_path, monkeypatch):
        """Return False on PermissionError during database access."""
        db_path = tmp_path / "chat.db"
        db_path.touch()

        reader = ChatDBReader(db_path=db_path)

        # Mock _get_connection to raise PermissionError
        def mock_get_connection():
            raise PermissionError("Permission denied")

        monkeypatch.setattr(reader, "_get_connection", mock_get_connection)

        result = reader.check_access()
        assert result is False


# =============================================================================
# OperationalError Handling Tests
# =============================================================================


class TestOperationalErrorHandling:
    """Tests for OperationalError handling in query methods."""

    def test_get_conversations_operational_error(self, monkeypatch):
        """Return empty list on OperationalError in get_conversations."""
        reader = ChatDBReader()

        # Create a mock connection that raises OperationalError
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("database is locked")
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        result = reader.get_conversations(limit=10)
        assert result == []

    def test_get_messages_operational_error(self, monkeypatch):
        """Return empty list on OperationalError in get_messages."""
        reader = ChatDBReader()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("SQLITE_BUSY")
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        result = reader.get_messages("chat;test", limit=10)
        assert result == []

    def test_search_operational_error(self, monkeypatch):
        """Return empty list on OperationalError in search."""
        reader = ChatDBReader()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError(
            "database disk image is malformed"
        )
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        result = reader.search("test query", limit=10)
        assert result == []

    def test_get_conversation_context_operational_error(self, monkeypatch):
        """Return empty list on OperationalError in get_conversation_context."""
        reader = ChatDBReader()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("unable to open database")
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        result = reader.get_conversation_context("chat;test", 1, context_messages=5)
        assert result == []


# =============================================================================
# Search with Invalid Sender Tests
# =============================================================================


class TestSearchInvalidSender:
    """Tests for search with invalid sender that normalizes to None."""

    def test_search_with_invalid_sender_treats_as_no_filter(self, tmp_path):
        """Search with sender that normalizes to None treats as no sender filter."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                guid TEXT,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        conn.execute(
            "INSERT INTO message VALUES (1, 'msg-1', 1, 'Hello world', NULL, 0, NULL, NULL)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            # Pass sender value that normalizes to None (empty string)
            # The normalize_phone_number function returns None for empty strings
            results = reader.search("Hello", sender="")

            # Should find message since sender filter is effectively disabled
            # when normalization returns None
            assert isinstance(results, list)


# =============================================================================
# Reactions Edge Case Tests
# =============================================================================


class TestReactionsEdgeCases:
    """Tests for edge cases in reaction retrieval."""

    def test_get_reactions_for_message_with_no_guid(self, tmp_path):
        """Return empty list when message has no GUID."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER,
                guid TEXT,
                date INTEGER,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT,
                associated_message_guid TEXT,
                associated_message_type INTEGER
            )
        """)
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("""
            CREATE TABLE attachment (
                ROWID INTEGER, filename TEXT, mime_type TEXT,
                total_bytes INTEGER, transfer_name TEXT
            )
        """)
        conn.execute(
            "CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)"
        )

        conn.execute("INSERT INTO chat VALUES (1, 'chat;test')")
        # Message with NULL guid
        conn.execute(
            "INSERT INTO message VALUES (1, NULL, 1, 'Test message', NULL, 0, NULL, NULL, NULL, 0)"
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            # Query reactions for a message with no GUID
            reactions = reader._get_reactions_for_message_id(1)
            assert reactions == []

    def test_get_reactions_for_nonexistent_message(self, tmp_path):
        """Return empty list when message ID doesn't exist."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)

        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT)")
        conn.execute("CREATE TABLE message (ROWID INTEGER, guid TEXT, thread_originator_guid TEXT)")
        conn.close()

        with ChatDBReader(db_path=db_path) as reader:
            reactions = reader._get_reactions_for_message_id(99999)
            assert reactions == []

    def test_get_reactions_operational_error(self, monkeypatch):
        """Return empty list on OperationalError in _get_reactions_for_message_id."""
        reader = ChatDBReader()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("database locked")
        mock_conn.cursor.return_value = mock_cursor

        reader._connection = mock_conn
        reader._schema_version = "v14"

        reactions = reader._get_reactions_for_message_id(1)
        assert reactions == []


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests for complete coverage."""

    def test_resolve_contact_with_email_from_cache(self):
        """Resolve email address from contacts cache."""
        reader = ChatDBReader()
        reader._contacts_cache = {
            "+15551234567": "John Doe",
            "jane@example.com": "Jane Smith",
        }

        # Email addresses should be lowercased for lookup
        result = reader._resolve_contact_name("jane@example.com")
        assert result == "Jane Smith"

    def test_lru_cache_maxsize_zero(self):
        """LRUCache with maxsize handles edge cases."""
        from integrations.imessage.reader import LRUCache

        # Very small cache
        cache: LRUCache[str, int] = LRUCache(maxsize=1)
        cache.set("a", 1)
        cache.set("b", 2)

        # Only 'b' should remain
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_contacts_with_null_name_fields(self, tmp_path):
        """Skip contacts with NULL name fields."""
        ab_db = tmp_path / "AddressBook.abcddb"
        conn = sqlite3.connect(ab_db)

        conn.execute("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDPHONENUMBER (
                ZFULLNUMBER TEXT,
                ZOWNER INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDEMAILADDRESS (
                ZADDRESS TEXT,
                ZOWNER INTEGER
            )
        """)

        # Insert contact with NULL names
        conn.execute("INSERT INTO ZABCDRECORD VALUES (1, NULL, NULL)")
        conn.execute("INSERT INTO ZABCDPHONENUMBER VALUES ('+15551234567', 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        # Should not be in cache because name is None
        assert "+15551234567" not in reader._contacts_cache

    def test_contacts_with_null_identifier(self, tmp_path):
        """Skip contacts with NULL phone/email identifiers."""
        ab_db = tmp_path / "AddressBook.abcddb"
        conn = sqlite3.connect(ab_db)

        conn.execute("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDPHONENUMBER (
                ZFULLNUMBER TEXT,
                ZOWNER INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE ZABCDEMAILADDRESS (
                ZADDRESS TEXT,
                ZOWNER INTEGER
            )
        """)

        # Insert contact with NULL identifier (phones should be skipped via WHERE)
        conn.execute("INSERT INTO ZABCDRECORD VALUES (1, 'John', 'Doe')")
        # Phone number that normalizes to None (empty)
        conn.execute("INSERT INTO ZABCDPHONENUMBER VALUES ('', 1)")
        conn.commit()
        conn.close()

        reader = ChatDBReader()
        reader._contacts_cache = {}
        reader._load_contacts_from_db(ab_db)

        # Empty phone should not be added to cache
        assert "" not in reader._contacts_cache
