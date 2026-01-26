"""iMessage chat.db schema detection implementation.

Implements the SchemaDetector protocol from contracts/health.py.
Detects schema versions and provides version-appropriate SQL queries.

Workstream 7 implementation.
"""

import logging
import sqlite3
import threading
from pathlib import Path

from contracts.health import SchemaInfo
from integrations.imessage.queries import QUERIES
from integrations.imessage.queries import detect_schema_version as imessage_detect_version
from integrations.imessage.queries import get_query as imessage_get_query

logger = logging.getLogger(__name__)

# Known schema versions and their characteristics
KNOWN_SCHEMAS = {
    "v14": {
        "description": "macOS 14 (Sonoma) and earlier",
        "required_tables": ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        "indicator_columns": {},  # Base schema
    },
    "v15": {
        "description": "macOS 15 (Sequoia) and later",
        "required_tables": ["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
        "indicator_columns": {
            "chat": ["service_name"],  # New in macOS 15
        },
    },
}

# Tables that must exist in any valid chat.db
REQUIRED_CORE_TABLES = ["message", "chat", "handle"]


class ChatDBSchemaDetector:
    """Detects iMessage chat.db schema version.

    Implements the SchemaDetector protocol from contracts/health.py.

    The detector examines table structure and column presence to determine
    which macOS version created the database, enabling version-specific
    query selection.
    """

    def __init__(self) -> None:
        """Initialize the schema detector."""
        self._lock = threading.Lock()
        self._cache: dict[str, SchemaInfo] = {}
        logger.info("ChatDBSchemaDetector initialized")

    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility.

        Opens the database in read-only mode and examines table structure
        to determine the schema version.

        Args:
            db_path: Path to the chat.db file

        Returns:
            SchemaInfo with version, tables, and compatibility information
        """
        with self._lock:
            # Check cache
            if db_path in self._cache:
                return self._cache[db_path]

        # Validate path exists
        path = Path(db_path)
        if not path.exists():
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )

        try:
            # Open in read-only mode
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row

            try:
                # Get all tables
                tables = self._get_tables(conn)

                # Check for required tables
                missing_tables = [t for t in REQUIRED_CORE_TABLES if t not in tables]
                if missing_tables:
                    logger.warning(
                        "Database missing required tables: %s",
                        missing_tables,
                    )
                    return SchemaInfo(
                        version="unknown",
                        tables=tables,
                        compatible=False,
                        migration_needed=False,
                        known_schema=False,
                    )

                # Detect version
                version = self._detect_version(conn)
                known_schema = version in KNOWN_SCHEMAS

                schema_info = SchemaInfo(
                    version=version,
                    tables=tables,
                    compatible=known_schema,
                    migration_needed=False,  # We support both v14 and v15
                    known_schema=known_schema,
                )

                # Cache the result
                with self._lock:
                    self._cache[db_path] = schema_info

                logger.info(
                    "Detected schema version %s for %s (compatible=%s)",
                    version,
                    db_path,
                    known_schema,
                )

                return schema_info

            finally:
                conn.close()

        except sqlite3.OperationalError as e:
            logger.warning("Cannot open database %s: %s", db_path, e)
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )
        except sqlite3.DatabaseError as e:
            logger.warning("Corrupted or invalid database %s: %s", db_path, e)
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )
        except PermissionError:
            logger.warning("Permission denied accessing %s", db_path)
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )

    def _get_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of all tables in the database.

        Args:
            conn: SQLite connection

        Returns:
            List of table names
        """
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def _detect_version(self, conn: sqlite3.Connection) -> str:
        """Detect schema version based on table structure.

        Delegates to the canonical detect_schema_version function in
        integrations/imessage/queries.py to maintain a single source of truth
        for schema detection logic.

        Args:
            conn: SQLite connection

        Returns:
            Schema version string (e.g., "v14", "v15", "unknown")
        """
        # Delegate to the canonical implementation in queries.py
        # This ensures consistent schema detection across the codebase
        return imessage_detect_version(conn)

    def get_query(self, query_name: str, schema_version: str) -> str:
        """Get appropriate SQL query for the detected schema.

        Delegates to the queries module in integrations/imessage for
        actual query retrieval.

        Args:
            query_name: Name of the query (conversations, messages, search, context)
            schema_version: Schema version (v14, v15)

        Returns:
            SQL query string appropriate for the schema version

        Raises:
            KeyError: If query_name is not found
        """
        # Use the queries module for actual query retrieval
        # This centralizes query management in one place
        try:
            return imessage_get_query(query_name, schema_version)
        except KeyError:
            logger.error(
                "Query '%s' not found for schema version '%s'",
                query_name,
                schema_version,
            )
            raise

    def get_supported_queries(self, schema_version: str) -> list[str]:
        """Get list of supported queries for a schema version.

        Args:
            schema_version: Schema version (v14, v15)

        Returns:
            List of query names supported for the given schema
        """
        version = schema_version if schema_version in QUERIES else "v14"
        return list(QUERIES[version].keys())

    def clear_cache(self) -> None:
        """Clear the schema detection cache.

        Useful when database might have been modified.
        """
        with self._lock:
            self._cache.clear()
            logger.debug("Schema detection cache cleared")


# Module-level singleton
_detector: ChatDBSchemaDetector | None = None
_detector_lock = threading.Lock()


def get_schema_detector() -> ChatDBSchemaDetector:
    """Get the singleton schema detector instance.

    Returns:
        The shared ChatDBSchemaDetector instance
    """
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = ChatDBSchemaDetector()
    return _detector


def reset_schema_detector() -> None:
    """Reset the singleton schema detector.

    Useful for testing or reinitializing the system.
    """
    global _detector
    with _detector_lock:
        _detector = None
        logger.info("Schema detector singleton reset")
