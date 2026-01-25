"""Unit tests for iMessage integration (Workstream 10)."""

import sqlite3
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contracts.imessage import Conversation, Message
from integrations.imessage import CHAT_DB_PATH, ChatDBReader
from integrations.imessage.parser import (
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
        from datetime import datetime

        # 2024-01-12 12:00:00 UTC should be 726753600 seconds from Apple epoch
        dt = datetime(2024, 1, 12, 12, 0, 0, tzinfo=UTC)
        result = datetime_to_apple_timestamp(dt)
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

    def test_none_returns_empty(self):
        """Return empty string for None input."""
        result = normalize_phone_number(None)
        assert result == ""

    def test_international_format_preserved(self):
        """Numbers with + are handled."""
        result = normalize_phone_number("+44 20 7946 0958")
        # Formatting stripped but otherwise preserved
        assert result == "+442079460958"


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


class TestParseAttachments:
    """Tests for attachment parsing (v1 stub)."""

    def test_returns_empty_list(self):
        """V1 always returns empty list."""
        result = parse_attachments(None)
        assert result == []

    def test_with_data_returns_empty(self):
        """V1 ignores data and returns empty list."""
        result = parse_attachments("some data")
        assert result == []


class TestParseReactions:
    """Tests for reaction parsing (v1 stub)."""

    def test_returns_empty_list(self):
        """V1 always returns empty list."""
        result = parse_reactions(None)
        assert result == []

    def test_with_data_returns_empty(self):
        """V1 ignores data and returns empty list."""
        result = parse_reactions([{"type": "like"}])
        assert result == []


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
            [(1, "ROWID", "INTEGER", 0, None, 1), (2, "thread_originator_guid", "TEXT", 0, None, 0)],
            # Second call: chat table columns (no service_name)
            [(1, "ROWID", "INTEGER", 0, None, 1), (2, "guid", "TEXT", 0, None, 0)],
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

    def test_query_with_filters(self):
        """Apply filter parameters to query."""
        result = get_query(
            "conversations",
            "v14",
            since_filter="AND last_message_date > ?",
        )
        assert "AND last_message_date > ?" in result

    def test_invalid_query_raises(self):
        """Raise KeyError for invalid query name."""
        with pytest.raises(KeyError):
            get_query("invalid_query", "v14")


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
        conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, thread_originator_guid TEXT)")
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.check_access()
        assert result is True
        reader.close()


class TestChatDBReaderGetConversations:
    """Tests for get_conversations method."""

    def test_empty_database(self, tmp_path):
        """Return empty list for database with no conversations."""
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)")
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("CREATE TABLE message (ROWID INTEGER, date INTEGER, text TEXT, thread_originator_guid TEXT)")
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
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)")
        conn.execute("CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER, id TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.execute("CREATE TABLE message (ROWID INTEGER, date INTEGER, text TEXT, thread_originator_guid TEXT)")

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
        conn.execute("INSERT INTO message VALUES (1, 726753600000000000, 'Hello', NULL, 0, 1, NULL)")
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
            conn.execute(f"INSERT INTO message VALUES ({i}, {i}, 'Message {i}', NULL, 0, NULL, NULL)")
            conn.execute(f"INSERT INTO chat_message_join VALUES (1, {i})")
        conn.commit()
        conn.close()

        reader = ChatDBReader(db_path=db_path)
        result = reader.get_conversation_context("chat;test", around_message_id=5, context_messages=2)

        # Should get 5 messages (2 before + target + 2 after)
        assert len(result) == 5
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
        conn.execute("CREATE TABLE chat (ROWID INTEGER, guid TEXT, display_name TEXT, chat_identifier TEXT)")
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
        conn.execute("INSERT INTO message VALUES (1, 726753600000000000, 'Hello', NULL, 0, 1, NULL)")
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
