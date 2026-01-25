"""Unit tests for Workstream 7: Schema Detector.

Tests chat.db schema detection and version-specific query retrieval.
"""

import sqlite3
import threading
from pathlib import Path

import pytest

from contracts.health import SchemaInfo
from core.health.schema import (
    KNOWN_SCHEMAS,
    REQUIRED_CORE_TABLES,
    ChatDBSchemaDetector,
    get_schema_detector,
    reset_schema_detector,
)


def create_test_db(path: Path, tables: list[str], columns: dict[str, list[str]] | None = None):
    """Create a test SQLite database with specified tables and columns.

    Args:
        path: Path to create the database
        tables: List of table names to create
        columns: Optional dict mapping table names to column lists
    """
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    default_columns = {
        "message": [
            "ROWID INTEGER PRIMARY KEY",
            "text TEXT",
            "date INTEGER",
            "handle_id INTEGER",
            "is_from_me INTEGER",
            "thread_originator_guid TEXT",
            "attributedBody BLOB",
        ],
        "chat": [
            "ROWID INTEGER PRIMARY KEY",
            "guid TEXT",
            "display_name TEXT",
            "chat_identifier TEXT",
        ],
        "handle": [
            "ROWID INTEGER PRIMARY KEY",
            "id TEXT",
        ],
        "chat_message_join": [
            "chat_id INTEGER",
            "message_id INTEGER",
        ],
        "chat_handle_join": [
            "chat_id INTEGER",
            "handle_id INTEGER",
        ],
    }

    # Merge custom columns
    if columns:
        for table, cols in columns.items():
            if table in default_columns:
                default_columns[table] = cols
            else:
                default_columns[table] = cols

    for table in tables:
        cols = default_columns.get(table, ["id INTEGER PRIMARY KEY"])
        col_defs = ", ".join(cols)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")

    conn.commit()
    conn.close()


class TestSchemaInfoDataclass:
    """Tests for SchemaInfo dataclass."""

    def test_schema_info_creation(self):
        """Test creating a SchemaInfo."""
        info = SchemaInfo(
            version="v14",
            tables=["message", "chat", "handle"],
            compatible=True,
            migration_needed=False,
            known_schema=True,
        )
        assert info.version == "v14"
        assert len(info.tables) == 3
        assert info.compatible is True
        assert info.migration_needed is False
        assert info.known_schema is True

    def test_schema_info_unknown_version(self):
        """Test SchemaInfo for unknown schema."""
        info = SchemaInfo(
            version="unknown",
            tables=[],
            compatible=False,
            migration_needed=False,
            known_schema=False,
        )
        assert info.version == "unknown"
        assert info.compatible is False
        assert info.known_schema is False


class TestKnownSchemas:
    """Tests for known schema definitions."""

    def test_v14_schema_defined(self):
        """Test v14 schema is defined."""
        assert "v14" in KNOWN_SCHEMAS
        assert "required_tables" in KNOWN_SCHEMAS["v14"]

    def test_v15_schema_defined(self):
        """Test v15 schema is defined."""
        assert "v15" in KNOWN_SCHEMAS
        assert "indicator_columns" in KNOWN_SCHEMAS["v15"]

    def test_required_core_tables(self):
        """Test required core tables are defined."""
        assert "message" in REQUIRED_CORE_TABLES
        assert "chat" in REQUIRED_CORE_TABLES
        assert "handle" in REQUIRED_CORE_TABLES


class TestChatDBSchemaDetectorInitialization:
    """Tests for ChatDBSchemaDetector initialization."""

    def test_initialization(self):
        """Test detector initializes correctly."""
        detector = ChatDBSchemaDetector()
        assert detector._cache == {}

    def test_cache_starts_empty(self):
        """Test cache is empty on initialization."""
        detector = ChatDBSchemaDetector()
        assert len(detector._cache) == 0


class TestSchemaDetectionV14:
    """Tests for v14 schema detection."""

    def test_detect_v14_schema(self, tmp_path):
        """Test detection of v14 schema."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.compatible is True
        assert info.known_schema is True
        assert "message" in info.tables
        assert "chat" in info.tables

    def test_detect_v14_with_thread_originator_guid(self, tmp_path):
        """Test v14 detection with thread_originator_guid column."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
            columns={
                "message": [
                    "ROWID INTEGER PRIMARY KEY",
                    "text TEXT",
                    "thread_originator_guid TEXT",
                ],
            },
        )

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.compatible is True


class TestSchemaDetectionV15:
    """Tests for v15 schema detection."""

    def test_detect_v15_schema(self, tmp_path):
        """Test detection of v15 schema with service_name column."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
            columns={
                "chat": [
                    "ROWID INTEGER PRIMARY KEY",
                    "guid TEXT",
                    "display_name TEXT",
                    "service_name TEXT",  # v15 indicator
                ],
            },
        )

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v15"
        assert info.compatible is True
        assert info.known_schema is True


class TestSchemaDetectionUnknown:
    """Tests for unknown schema detection."""

    def test_detect_unknown_missing_tables(self, tmp_path):
        """Test detection returns unknown for missing required tables."""
        db_path = tmp_path / "chat.db"
        # Missing required tables
        create_test_db(db_path, ["some_other_table"])

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False
        assert info.known_schema is False

    def test_detect_nonexistent_file(self, tmp_path):
        """Test detection for non-existent file."""
        db_path = tmp_path / "nonexistent.db"

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False
        assert info.tables == []


class TestSchemaDetectionCaching:
    """Tests for schema detection caching."""

    def test_cached_result_returned(self, tmp_path):
        """Test that cached results are returned."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        detector = ChatDBSchemaDetector()

        # First call
        info1 = detector.detect(str(db_path))
        # Second call should return cached
        info2 = detector.detect(str(db_path))

        assert info1 is info2

    def test_clear_cache(self, tmp_path):
        """Test clear_cache method."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        detector = ChatDBSchemaDetector()

        # Populate cache
        detector.detect(str(db_path))
        assert len(detector._cache) == 1

        # Clear cache
        detector.clear_cache()
        assert len(detector._cache) == 0


class TestGetQuery:
    """Tests for get_query method."""

    def test_get_conversations_query_v14(self):
        """Test getting conversations query for v14."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("conversations", "v14")

        assert "SELECT" in query
        assert "chat" in query

    def test_get_messages_query_v14(self):
        """Test getting messages query for v14."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("messages", "v14")

        assert "SELECT" in query
        assert "message" in query

    def test_get_search_query_v14(self):
        """Test getting search query for v14."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("search", "v14")

        assert "SELECT" in query
        assert "LIKE" in query

    def test_get_context_query_v14(self):
        """Test getting context query for v14."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("context", "v14")

        assert "SELECT" in query
        assert "ABS" in query

    def test_get_query_v15(self):
        """Test getting query for v15 schema."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("messages", "v15")

        assert "SELECT" in query
        assert "message" in query

    def test_get_query_unknown_version_falls_back_to_v14(self):
        """Test unknown version falls back to v14 queries."""
        detector = ChatDBSchemaDetector()
        query_unknown = detector.get_query("conversations", "unknown")
        query_v14 = detector.get_query("conversations", "v14")

        # Should get same query since unknown falls back to v14
        assert query_unknown == query_v14

    def test_get_query_invalid_name_raises(self):
        """Test invalid query name raises KeyError."""
        detector = ChatDBSchemaDetector()

        with pytest.raises(KeyError):
            detector.get_query("invalid_query_name", "v14")


class TestGetSupportedQueries:
    """Tests for get_supported_queries method."""

    def test_supported_queries_v14(self):
        """Test getting supported queries for v14."""
        detector = ChatDBSchemaDetector()
        queries = detector.get_supported_queries("v14")

        assert "conversations" in queries
        assert "messages" in queries
        assert "search" in queries
        assert "context" in queries

    def test_supported_queries_v15(self):
        """Test getting supported queries for v15."""
        detector = ChatDBSchemaDetector()
        queries = detector.get_supported_queries("v15")

        assert "conversations" in queries
        assert "messages" in queries

    def test_supported_queries_unknown_falls_back(self):
        """Test unknown version falls back for supported queries."""
        detector = ChatDBSchemaDetector()
        queries = detector.get_supported_queries("unknown")

        # Should get v14 queries
        assert "conversations" in queries


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_schema_detector_returns_singleton(self):
        """Test get_schema_detector returns same instance."""
        reset_schema_detector()

        d1 = get_schema_detector()
        d2 = get_schema_detector()

        assert d1 is d2

        reset_schema_detector()

    def test_reset_creates_new_instance(self):
        """Test reset_schema_detector creates new instance."""
        reset_schema_detector()

        d1 = get_schema_detector()
        reset_schema_detector()
        d2 = get_schema_detector()

        assert d1 is not d2

        reset_schema_detector()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_detection(self, tmp_path):
        """Test concurrent detection is thread-safe."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        detector = ChatDBSchemaDetector()
        results = []
        lock = threading.Lock()

        def detect_schema():
            for _ in range(10):
                info = detector.detect(str(db_path))
                with lock:
                    results.append(info)

        threads = [threading.Thread(target=detect_schema) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        # All results should be the same cached object
        assert all(r.version == results[0].version for r in results)

    def test_concurrent_query_retrieval(self):
        """Test concurrent query retrieval is thread-safe."""
        detector = ChatDBSchemaDetector()
        results = []
        lock = threading.Lock()

        def get_queries():
            for _ in range(10):
                query = detector.get_query("conversations", "v14")
                with lock:
                    results.append(query)

        threads = [threading.Thread(target=get_queries) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        assert all(r == results[0] for r in results)


class TestProtocolCompliance:
    """Verify ChatDBSchemaDetector implements SchemaDetector protocol."""

    def test_has_detect(self):
        """Detector has detect method."""
        detector = ChatDBSchemaDetector()
        assert hasattr(detector, "detect")
        assert callable(detector.detect)

    def test_has_get_query(self):
        """Detector has get_query method."""
        detector = ChatDBSchemaDetector()
        assert hasattr(detector, "get_query")
        assert callable(detector.get_query)

    def test_detect_returns_schema_info(self, tmp_path):
        """detect returns SchemaInfo."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        detector = ChatDBSchemaDetector()
        result = detector.detect(str(db_path))

        assert isinstance(result, SchemaInfo)

    def test_get_query_returns_string(self):
        """get_query returns string."""
        detector = ChatDBSchemaDetector()
        result = detector.get_query("conversations", "v14")

        assert isinstance(result, str)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_corrupted_database(self, tmp_path):
        """Test handling of corrupted database."""
        db_path = tmp_path / "corrupted.db"
        db_path.write_bytes(b"not a valid sqlite database")

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False

    def test_empty_database(self, tmp_path):
        """Test handling of empty database (no tables)."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False

    def test_partial_schema(self, tmp_path):
        """Test handling of database with partial schema."""
        db_path = tmp_path / "partial.db"
        # Only create message table, missing chat and handle
        create_test_db(db_path, ["message"])

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False
        assert "message" in info.tables

    def test_extra_tables_dont_affect_detection(self, tmp_path):
        """Test extra tables don't affect version detection."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            [
                "message",
                "chat",
                "handle",
                "chat_message_join",
                "chat_handle_join",
                "custom_table",
                "another_table",
            ],
        )

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.compatible is True
        assert "custom_table" in info.tables


class TestIntegrationWithQueries:
    """Integration tests with the queries module."""

    def test_detected_schema_queries_work(self, tmp_path):
        """Test that detected schema version returns working queries."""
        db_path = tmp_path / "chat.db"
        create_test_db(
            db_path,
            ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        )

        # Add some test data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat (ROWID, guid) VALUES (1, 'test-guid')")
        conn.commit()
        conn.close()

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        # Get query for detected version
        query = detector.get_query("conversations", info.version)

        # Query should be valid SQL (basic check)
        assert "SELECT" in query
        assert "FROM" in query
