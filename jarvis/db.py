"""JARVIS Database Management - SQLite database for contacts, pairs, and clusters.

Manages ~/.jarvis/jarvis.db which stores:
- Contacts with relationship labels and handle mappings
- Extracted (trigger, response) pairs from iMessage history
- Intent clusters mined from response patterns (optional)
- FAISS vector index metadata and versioning

Usage:
    jarvis db init                     # Create database
    jarvis db add-contact --name "Sarah" --relationship "sister"
    jarvis db list-contacts            # View contacts
    jarvis db extract                  # Extract pairs from chat.db
    jarvis db build-index              # Build FAISS index
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Register custom timestamp converter that handles timezone-aware timestamps
def _convert_timestamp(val: bytes) -> datetime:
    """Convert timestamp bytes to datetime, handling timezone info."""
    datepart, timepart = val.split(b" ")
    year, month, day = (int(x) for x in datepart.split(b"-"))

    # Handle timezone offset (e.g., "14:30:00.123456+00:00")
    if b"+" in timepart:
        timepart, tz_offset = timepart.rsplit(b"+", 1)
    elif timepart.count(b"-") == 1:
        timepart, tz_offset = timepart.rsplit(b"-", 1)
    # Handle microseconds
    if b"." in timepart:
        timepart, microseconds = timepart.split(b".")
        microseconds = int(microseconds)
    else:
        microseconds = 0

    hours, minutes, seconds = (int(x) for x in timepart.split(b":"))

    return datetime(year, month, day, hours, minutes, seconds, microseconds)


# Register the custom converter
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)

# Default database path
JARVIS_DB_PATH = Path.home() / ".jarvis" / "jarvis.db"
INDEXES_DIR = Path.home() / ".jarvis" / "indexes"


@dataclass
class Contact:
    """Contact with relationship metadata."""

    id: int | None
    chat_id: str | None
    display_name: str
    phone_or_email: str | None
    relationship: str | None
    style_notes: str | None
    handles_json: str | None = None  # JSON array of handles ["phone", "email"]
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def handles(self) -> list[str]:
        """Get list of handles from JSON."""
        if self.handles_json:
            try:
                return json.loads(self.handles_json)
            except json.JSONDecodeError:
                return []
        return []


@dataclass
class Pair:
    """A (trigger, response) pair extracted from message history."""

    id: int | None
    contact_id: int | None
    trigger_text: str
    response_text: str
    trigger_timestamp: datetime
    response_timestamp: datetime
    chat_id: str
    # Message IDs for debugging/deduplication
    trigger_msg_id: int | None = None
    response_msg_id: int | None = None  # First response msg ID if multi-message
    trigger_msg_ids_json: str | None = None  # JSON array for multi-message triggers
    response_msg_ids_json: str | None = None  # JSON array for multi-message responses
    # Conversation context (messages leading up to the trigger)
    context_text: str | None = None  # Previous messages before trigger for LLM context
    # Quality and filtering
    quality_score: float = 1.0
    flags_json: str | None = None  # JSON: {"attachment_only":true, "short":true}
    is_group: bool = False  # True if from group chat
    is_holdout: bool = False  # True if reserved for evaluation (not in training index)
    # Validity gate results (v6+)
    gate_a_passed: bool | None = None  # Rule gate result
    gate_b_score: float | None = None  # Embedding similarity score
    gate_c_verdict: str | None = None  # NLI verdict (accept/reject/uncertain)
    validity_status: str | None = None  # Final: valid/invalid/uncertain
    # Freshness and usage tracking
    usage_count: int = 0
    last_used_at: datetime | None = None
    last_verified_at: datetime | None = None
    source_timestamp: datetime | None = None  # For decay/freshness

    @property
    def flags(self) -> dict[str, Any]:
        """Get flags dict from JSON."""
        if self.flags_json:
            try:
                return json.loads(self.flags_json)
            except json.JSONDecodeError:
                return {}
        return {}

    @property
    def trigger_msg_ids(self) -> list[int]:
        """Get list of trigger message IDs."""
        if self.trigger_msg_ids_json:
            try:
                return json.loads(self.trigger_msg_ids_json)
            except json.JSONDecodeError:
                return []
        return [self.trigger_msg_id] if self.trigger_msg_id else []

    @property
    def response_msg_ids(self) -> list[int]:
        """Get list of response message IDs."""
        if self.response_msg_ids_json:
            try:
                return json.loads(self.response_msg_ids_json)
            except json.JSONDecodeError:
                return []
        return [self.response_msg_id] if self.response_msg_id else []


@dataclass
class PairArtifact:
    """Heavy artifacts for a pair (stored separately to keep pairs table lean).

    Stored in pair_artifacts table to avoid bloating the main pairs table.
    """

    pair_id: int
    context_json: str | None = None  # Structured context window (JSON list of messages)
    gate_a_reason: str | None = None  # Why Gate A rejected (if rejected)
    gate_c_scores_json: str | None = None  # Raw NLI scores (JSON dict)
    raw_trigger_text: str | None = None  # Original text before normalization
    raw_response_text: str | None = None  # Original text before normalization

    @property
    def context_messages(self) -> list[dict]:
        """Get context messages from JSON."""
        if self.context_json:
            try:
                return json.loads(self.context_json)
            except json.JSONDecodeError:
                return []
        return []

    @property
    def gate_c_scores(self) -> dict[str, float]:
        """Get Gate C scores from JSON."""
        if self.gate_c_scores_json:
            try:
                return json.loads(self.gate_c_scores_json)
            except json.JSONDecodeError:
                return {}
        return {}


@dataclass
class ContactStyleTargets:
    """Style targets for a contact (computed from their pairs)."""

    contact_id: int
    median_reply_length: int = 10  # Median word count
    punctuation_rate: float = 0.5  # Fraction with ending punctuation
    emoji_rate: float = 0.1  # Fraction containing emojis
    greeting_rate: float = 0.2  # Fraction starting with greeting
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contact_id": self.contact_id,
            "median_reply_length": self.median_reply_length,
            "punctuation_rate": self.punctuation_rate,
            "emoji_rate": self.emoji_rate,
            "greeting_rate": self.greeting_rate,
        }


@dataclass
class Cluster:
    """An intent cluster grouping similar responses."""

    id: int | None
    name: str
    description: str | None
    example_triggers: list[str] = field(default_factory=list)
    example_responses: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass
class PairEmbedding:
    """Links a pair to its FAISS vector position."""

    pair_id: int  # PRIMARY KEY - stable reference
    faiss_id: int  # Position in FAISS index (can change on rebuild)
    cluster_id: int | None
    index_version: str | None = None  # Which index version this belongs to


@dataclass
class IndexVersion:
    """Metadata for a FAISS index version."""

    id: int | None
    version_id: str  # e.g., "20240115-143022"
    model_name: str
    embedding_dim: int
    num_vectors: int
    index_path: str
    is_active: bool
    created_at: datetime | None = None


# Schema SQL - Version 6 with validity gates and split tables
SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Contacts with relationship labels and multiple handles
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY,
    chat_id TEXT UNIQUE,              -- primary iMessage chat_id
    display_name TEXT NOT NULL,
    phone_or_email TEXT,              -- primary contact method
    handles_json TEXT,                -- JSON array: ["+15551234567", "email@x.com"]
    relationship TEXT,                -- 'sister', 'coworker', 'friend', 'boss'
    style_notes TEXT,                 -- 'casual, uses emojis'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Style targets for contacts (computed from their pairs)
CREATE TABLE IF NOT EXISTS contact_style_targets (
    contact_id INTEGER PRIMARY KEY REFERENCES contacts(id),
    median_reply_length INTEGER DEFAULT 10,   -- median word count
    punctuation_rate REAL DEFAULT 0.5,        -- fraction with ending punctuation
    emoji_rate REAL DEFAULT 0.1,              -- fraction containing emojis
    greeting_rate REAL DEFAULT 0.2,           -- fraction starting with greeting
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted message pairs from history (lean table)
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY,
    contact_id INTEGER REFERENCES contacts(id),
    trigger_text TEXT NOT NULL,       -- what they said (may be multi-message joined)
    response_text TEXT NOT NULL,      -- what you said (may be multi-message joined)
    trigger_timestamp TIMESTAMP,      -- timestamp of first trigger message
    response_timestamp TIMESTAMP,     -- timestamp of first response message
    chat_id TEXT,                     -- source conversation
    -- Message IDs for debugging and deduplication
    trigger_msg_id INTEGER,           -- primary trigger message ID
    response_msg_id INTEGER,          -- primary response message ID
    trigger_msg_ids_json TEXT,        -- JSON array for multi-message triggers
    response_msg_ids_json TEXT,       -- JSON array for multi-message responses
    -- Conversation context (legacy, use pair_artifacts for v6+)
    context_text TEXT,                -- previous messages before trigger (for LLM context)
    -- Quality and filtering
    quality_score REAL DEFAULT 1.0,   -- 0.0-1.0, lower = worse
    flags_json TEXT,                  -- JSON: {"attachment_only":true, "short":true}
    is_group BOOLEAN DEFAULT FALSE,   -- True if from group chat (for filtering)
    is_holdout BOOLEAN DEFAULT FALSE, -- True if reserved for evaluation (not in training)
    -- Validity gate results (v6+)
    gate_a_passed BOOLEAN,            -- Rule gate result
    gate_b_score REAL,                -- Embedding similarity score
    gate_c_verdict TEXT,              -- NLI verdict (accept/reject/uncertain)
    validity_status TEXT,             -- Final: valid/invalid/uncertain
    -- Freshness and usage tracking
    usage_count INTEGER DEFAULT 0,    -- times this pair was used for generation
    last_used_at TIMESTAMP,           -- last time pair was used
    last_verified_at TIMESTAMP,       -- for re-indexing workflows
    source_timestamp TIMESTAMP,       -- original message timestamp (for decay)
    -- Uniqueness: use primary message IDs
    UNIQUE(trigger_msg_id, response_msg_id)
);

-- Heavy artifacts for pairs (split table to keep pairs lean)
CREATE TABLE IF NOT EXISTS pair_artifacts (
    pair_id INTEGER PRIMARY KEY REFERENCES pairs(id),
    context_json TEXT,                -- Structured context window (JSON list of messages)
    gate_a_reason TEXT,               -- Why Gate A rejected (if rejected)
    gate_c_scores_json TEXT,          -- Raw NLI scores (JSON dict)
    raw_trigger_text TEXT,            -- Original text before normalization
    raw_response_text TEXT            -- Original text before normalization
);

-- Clustered intent groups (optional, for later analytics)
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,        -- 'INVITATION', 'GREETING', 'SCHEDULE'
    description TEXT,
    example_triggers TEXT,            -- JSON array
    example_responses TEXT,           -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Links pairs to FAISS vectors (keyed by pair_id for stability)
CREATE TABLE IF NOT EXISTS pair_embeddings (
    pair_id INTEGER PRIMARY KEY REFERENCES pairs(id),
    faiss_id INTEGER UNIQUE,          -- position in FAISS index (can change)
    cluster_id INTEGER REFERENCES clusters(id),
    index_version TEXT                -- which index version this belongs to
);

-- FAISS index versions for safe rebuilds
CREATE TABLE IF NOT EXISTS index_versions (
    id INTEGER PRIMARY KEY,
    version_id TEXT UNIQUE NOT NULL,  -- e.g., "20240115-143022"
    model_name TEXT NOT NULL,         -- e.g., "BAAI/bge-small-en-v1.5"
    embedding_dim INTEGER NOT NULL,   -- e.g., 384
    num_vectors INTEGER NOT NULL,
    index_path TEXT NOT NULL,         -- relative path to index file
    is_active BOOLEAN DEFAULT FALSE,
    normalized BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_pairs_contact ON pairs(contact_id);
CREATE INDEX IF NOT EXISTS idx_pairs_chat ON pairs(chat_id);
CREATE INDEX IF NOT EXISTS idx_pairs_quality ON pairs(quality_score);
CREATE INDEX IF NOT EXISTS idx_pairs_validity ON pairs(validity_status);
CREATE INDEX IF NOT EXISTS idx_contacts_chat ON contacts(chat_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_index ON pair_embeddings(index_version);
CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON pair_embeddings(faiss_id);
"""

CURRENT_SCHEMA_VERSION = 6  # Added validity gates and split tables


class JarvisDB:
    """Manager for the JARVIS SQLite database.

    Thread-safe connection management with context manager support.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to database file. Uses default if None.
        """
        self.db_path = db_path or JARVIS_DB_PATH
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with automatic cleanup.

        Yields:
            SQLite connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> bool:
        """Initialize database schema.

        Creates all tables if they don't exist.

        Returns:
            True if schema was created/updated, False if already current.
        """
        with self.connection() as conn:
            # Check current schema version
            try:
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                row = cursor.fetchone()
                current_version = row["version"] if row else 0
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                current_version = 0

            if current_version >= CURRENT_SCHEMA_VERSION:
                logger.debug("Schema already at version %d", current_version)
                return False

            # Apply migrations for existing databases
            if current_version == 2:
                # Migration v2 -> v3: Add context_text column to pairs
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN context_text TEXT")
                    logger.info("Added context_text column to pairs table")
                except sqlite3.OperationalError:
                    # Column already exists (e.g., during development)
                    pass

            if current_version <= 3:
                # Migration v3 -> v4: Add is_group column to pairs
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN is_group BOOLEAN DEFAULT FALSE")
                    logger.info("Added is_group column to pairs table")
                except sqlite3.OperationalError:
                    # Column already exists (e.g., during development)
                    pass

            if current_version <= 4:
                # Migration v4 -> v5: Add is_holdout column for train/test split
                try:
                    conn.execute("ALTER TABLE pairs ADD COLUMN is_holdout BOOLEAN DEFAULT FALSE")
                    logger.info("Added is_holdout column to pairs table")
                except sqlite3.OperationalError:
                    # Column already exists (e.g., during development)
                    pass

            if current_version <= 5:
                # Migration v5 -> v6: Add validity gate columns and split tables
                # Add gate columns to pairs table
                gate_columns = [
                    ("gate_a_passed", "BOOLEAN"),
                    ("gate_b_score", "REAL"),
                    ("gate_c_verdict", "TEXT"),
                    ("validity_status", "TEXT"),
                ]
                for col_name, col_type in gate_columns:
                    try:
                        conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")
                        logger.info("Added %s column to pairs table", col_name)
                    except sqlite3.OperationalError:
                        pass

                # Create new tables (handled by SCHEMA_SQL, but ensure they exist)
                # pair_artifacts and contact_style_targets are created by executescript

            # Apply schema
            conn.executescript(SCHEMA_SQL)

            # Update version
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )

            logger.info(
                "Schema updated from version %d to %d",
                current_version,
                CURRENT_SCHEMA_VERSION,
            )
            return True

    def exists(self) -> bool:
        """Check if the database file exists."""
        return self.db_path.exists()

    # -------------------------------------------------------------------------
    # Contact Operations
    # -------------------------------------------------------------------------

    def add_contact(
        self,
        display_name: str,
        chat_id: str | None = None,
        phone_or_email: str | None = None,
        relationship: str | None = None,
        style_notes: str | None = None,
        handles: list[str] | None = None,
    ) -> Contact:
        """Add or update a contact.

        If a contact with the same chat_id exists, it will be updated.

        Args:
            display_name: Contact's display name.
            chat_id: Optional chat ID from iMessage.
            phone_or_email: Phone number or email.
            relationship: Relationship type (e.g., 'sister', 'boss').
            style_notes: Communication style notes.
            handles: List of all handles (phones/emails) for this person.

        Returns:
            The created or updated Contact.
        """
        with self.connection() as conn:
            now = datetime.now()
            handles_json = json.dumps(handles) if handles else None

            if chat_id:
                # Check for existing contact
                cursor = conn.execute("SELECT id FROM contacts WHERE chat_id = ?", (chat_id,))
                existing = cursor.fetchone()

                if existing:
                    # Update existing
                    conn.execute(
                        """
                        UPDATE contacts
                        SET display_name = ?, phone_or_email = ?, relationship = ?,
                            style_notes = ?, handles_json = ?, updated_at = ?
                        WHERE chat_id = ?
                        """,
                        (
                            display_name,
                            phone_or_email,
                            relationship,
                            style_notes,
                            handles_json,
                            now,
                            chat_id,
                        ),
                    )
                    return Contact(
                        id=existing["id"],
                        chat_id=chat_id,
                        display_name=display_name,
                        phone_or_email=phone_or_email,
                        relationship=relationship,
                        style_notes=style_notes,
                        handles_json=handles_json,
                        updated_at=now,
                    )

            # Insert new contact
            cursor = conn.execute(
                """
                INSERT INTO contacts
                (chat_id, display_name, phone_or_email, relationship, style_notes, handles_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, display_name, phone_or_email, relationship, style_notes, handles_json),
            )

            return Contact(
                id=cursor.lastrowid,
                chat_id=chat_id,
                display_name=display_name,
                phone_or_email=phone_or_email,
                relationship=relationship,
                style_notes=style_notes,
                handles_json=handles_json,
                created_at=now,
                updated_at=now,
            )

    def get_contact(self, contact_id: int) -> Contact | None:
        """Get a contact by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_contact(row)
            return None

    def get_contact_by_chat_id(self, chat_id: str) -> Contact | None:
        """Get a contact by their chat ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM contacts WHERE chat_id = ?", (chat_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_contact(row)
            return None

    def get_contact_by_name(self, name: str) -> Contact | None:
        """Get a contact by display name (case-insensitive partial match)."""
        with self.connection() as conn:
            # Try exact match first
            cursor = conn.execute(
                "SELECT * FROM contacts WHERE LOWER(display_name) = LOWER(?)", (name,)
            )
            row = cursor.fetchone()

            if not row:
                # Try partial match
                cursor = conn.execute(
                    "SELECT * FROM contacts WHERE LOWER(display_name) LIKE LOWER(?)",
                    (f"%{name}%",),
                )
                row = cursor.fetchone()

            if row:
                return self._row_to_contact(row)
            return None

    def list_contacts(self, limit: int = 100) -> list[Contact]:
        """List all contacts."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM contacts ORDER BY display_name LIMIT ?", (limit,))
            return [self._row_to_contact(row) for row in cursor]

    def delete_contact(self, contact_id: int) -> bool:
        """Delete a contact and their associated pairs."""
        with self.connection() as conn:
            # Delete embeddings for this contact's pairs
            conn.execute(
                """DELETE FROM pair_embeddings WHERE pair_id IN
                   (SELECT id FROM pairs WHERE contact_id = ?)""",
                (contact_id,),
            )
            # Delete pairs
            conn.execute("DELETE FROM pairs WHERE contact_id = ?", (contact_id,))
            cursor = conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
            return cursor.rowcount > 0

    def _row_to_contact(self, row: sqlite3.Row) -> Contact:
        """Convert a database row to a Contact object."""
        return Contact(
            id=row["id"],
            chat_id=row["chat_id"],
            display_name=row["display_name"],
            phone_or_email=row["phone_or_email"],
            relationship=row["relationship"],
            style_notes=row["style_notes"],
            handles_json=row["handles_json"] if "handles_json" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -------------------------------------------------------------------------
    # Pair Operations
    # -------------------------------------------------------------------------

    def add_pair(
        self,
        trigger_text: str,
        response_text: str,
        trigger_timestamp: datetime,
        response_timestamp: datetime,
        chat_id: str,
        contact_id: int | None = None,
        trigger_msg_id: int | None = None,
        response_msg_id: int | None = None,
        trigger_msg_ids: list[int] | None = None,
        response_msg_ids: list[int] | None = None,
        context_text: str | None = None,
        quality_score: float = 1.0,
        flags: dict[str, Any] | None = None,
    ) -> Pair | None:
        """Add a (trigger, response) pair.

        Ignores duplicates based on (trigger_msg_id, response_msg_id).

        Args:
            context_text: Previous messages before trigger for LLM context.

        Returns:
            The created Pair, or None if duplicate.
        """
        with self.connection() as conn:
            trigger_msg_ids_json = json.dumps(trigger_msg_ids) if trigger_msg_ids else None
            response_msg_ids_json = json.dumps(response_msg_ids) if response_msg_ids else None
            flags_json = json.dumps(flags) if flags else None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO pairs
                    (contact_id, trigger_text, response_text, trigger_timestamp,
                     response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                     trigger_msg_ids_json, response_msg_ids_json, context_text,
                     quality_score, flags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contact_id,
                        trigger_text,
                        response_text,
                        trigger_timestamp,
                        response_timestamp,
                        chat_id,
                        trigger_msg_id,
                        response_msg_id,
                        trigger_msg_ids_json,
                        response_msg_ids_json,
                        context_text,
                        quality_score,
                        flags_json,
                    ),
                )
                return Pair(
                    id=cursor.lastrowid,
                    contact_id=contact_id,
                    trigger_text=trigger_text,
                    response_text=response_text,
                    trigger_timestamp=trigger_timestamp,
                    response_timestamp=response_timestamp,
                    chat_id=chat_id,
                    trigger_msg_id=trigger_msg_id,
                    response_msg_id=response_msg_id,
                    trigger_msg_ids_json=trigger_msg_ids_json,
                    response_msg_ids_json=response_msg_ids_json,
                    context_text=context_text,
                    quality_score=quality_score,
                    flags_json=flags_json,
                )
            except sqlite3.IntegrityError:
                # Duplicate pair
                return None

    def add_pairs_bulk(self, pairs: list[dict[str, Any]]) -> int:
        """Add multiple pairs in a single transaction.

        Args:
            pairs: List of pair dictionaries.

        Returns:
            Number of pairs successfully added.
        """
        added = 0
        with self.connection() as conn:
            for pair in pairs:
                trigger_msg_ids_json = (
                    json.dumps(pair.get("trigger_msg_ids")) if pair.get("trigger_msg_ids") else None
                )
                response_msg_ids_json = (
                    json.dumps(pair.get("response_msg_ids"))
                    if pair.get("response_msg_ids")
                    else None
                )
                flags_json = json.dumps(pair.get("flags")) if pair.get("flags") else None

                try:
                    conn.execute(
                        """
                        INSERT INTO pairs
                        (contact_id, trigger_text, response_text, trigger_timestamp,
                         response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                         trigger_msg_ids_json, response_msg_ids_json, context_text,
                         quality_score, flags_json, is_group)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pair.get("contact_id"),
                            pair["trigger_text"],
                            pair["response_text"],
                            pair["trigger_timestamp"],
                            pair["response_timestamp"],
                            pair["chat_id"],
                            pair.get("trigger_msg_id"),
                            pair.get("response_msg_id"),
                            trigger_msg_ids_json,
                            response_msg_ids_json,
                            pair.get("context_text"),
                            pair.get("quality_score", 1.0),
                            flags_json,
                            pair.get("is_group", False),
                        ),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    # Skip duplicates
                    continue
        return added

    def get_pairs(
        self,
        contact_id: int | None = None,
        chat_id: str | None = None,
        min_quality: float = 0.0,
        limit: int = 10000,
    ) -> list[Pair]:
        """Get pairs with optional filtering."""
        with self.connection() as conn:
            conditions = ["quality_score >= ?"]
            params: list[Any] = [min_quality]

            if contact_id is not None:
                conditions.append("contact_id = ?")
                params.append(contact_id)
            if chat_id is not None:
                conditions.append("chat_id = ?")
                params.append(chat_id)

            where_clause = " AND ".join(conditions)
            params.append(limit)

            cursor = conn.execute(
                f"SELECT * FROM pairs WHERE {where_clause} ORDER BY trigger_timestamp DESC LIMIT ?",
                params,
            )
            return [self._row_to_pair(row) for row in cursor]

    def get_all_pairs(self, min_quality: float = 0.0) -> list[Pair]:
        """Get all pairs in the database."""
        return self.get_pairs(min_quality=min_quality, limit=1000000)

    def count_pairs(self, min_quality: float = 0.0) -> int:
        """Count total pairs in database."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= ?",
                (min_quality,),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def update_pair_quality(
        self, pair_id: int, quality_score: float, flags: dict[str, Any] | None = None
    ) -> bool:
        """Update a pair's quality score and flags."""
        with self.connection() as conn:
            if flags is not None:
                cursor = conn.execute(
                    "UPDATE pairs SET quality_score = ?, flags_json = ? WHERE id = ?",
                    (quality_score, json.dumps(flags), pair_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE pairs SET quality_score = ? WHERE id = ?",
                    (quality_score, pair_id),
                )
            return cursor.rowcount > 0

    def clear_pairs(self) -> int:
        """Delete all pairs from the database."""
        with self.connection() as conn:
            conn.execute("DELETE FROM pair_embeddings")
            cursor = conn.execute("DELETE FROM pairs")
            return cursor.rowcount

    def get_pairs_by_trigger_pattern(
        self,
        contact_id: int,
        pattern_type: str = "acknowledgment",
        limit: int = 10,
    ) -> list[Pair]:
        """Get pairs matching a trigger pattern type.

        Used to analyze user's typical response patterns to certain message types.
        For example, checking if user typically provides substantive info after
        acknowledgments like "ok" or "sure".

        Args:
            contact_id: Contact to query.
            pattern_type: Pattern to match. Currently supported:
                - "acknowledgment": Short ack triggers like "ok", "sure", "yes"
            limit: Max pairs to return.

        Returns:
            List of Pair objects matching the pattern, ordered by recency.
        """
        if pattern_type == "acknowledgment":
            ack_triggers = (
                "ok",
                "okay",
                "sure",
                "yes",
                "yeah",
                "yep",
                "yup",
                "got it",
                "k",
                "kk",
                "alright",
                "sounds good",
            )
            placeholders = ",".join("?" * len(ack_triggers))
            query = f"""
                SELECT * FROM pairs
                WHERE contact_id = ?
                AND LOWER(TRIM(trigger_text)) IN ({placeholders})
                AND quality_score >= 0.5
                ORDER BY trigger_timestamp DESC
                LIMIT ?
            """
            with self.connection() as conn:
                cursor = conn.execute(query, (contact_id, *ack_triggers, limit))
                return [self._row_to_pair(row) for row in cursor]

        return []

    def _row_to_pair(self, row: sqlite3.Row) -> Pair:
        """Convert a database row to a Pair object."""
        keys = row.keys()
        return Pair(
            id=row["id"],
            contact_id=row["contact_id"],
            trigger_text=row["trigger_text"],
            response_text=row["response_text"],
            trigger_timestamp=row["trigger_timestamp"],
            response_timestamp=row["response_timestamp"],
            chat_id=row["chat_id"],
            trigger_msg_id=row["trigger_msg_id"] if "trigger_msg_id" in keys else None,
            response_msg_id=row["response_msg_id"] if "response_msg_id" in keys else None,
            trigger_msg_ids_json=row["trigger_msg_ids_json"]
            if "trigger_msg_ids_json" in keys
            else None,
            response_msg_ids_json=row["response_msg_ids_json"]
            if "response_msg_ids_json" in keys
            else None,
            context_text=row["context_text"] if "context_text" in keys else None,
            quality_score=row["quality_score"] if "quality_score" in keys else 1.0,
            flags_json=row["flags_json"] if "flags_json" in keys else None,
            is_group=bool(row["is_group"]) if "is_group" in keys else False,
            is_holdout=bool(row["is_holdout"]) if "is_holdout" in keys else False,
            gate_a_passed=(
                bool(row["gate_a_passed"])
                if "gate_a_passed" in keys and row["gate_a_passed"] is not None
                else None
            ),
            gate_b_score=(
                float(row["gate_b_score"])
                if "gate_b_score" in keys and row["gate_b_score"] is not None
                else None
            ),
            gate_c_verdict=row["gate_c_verdict"] if "gate_c_verdict" in keys else None,
            validity_status=row["validity_status"] if "validity_status" in keys else None,
        )

    # -------------------------------------------------------------------------
    # Train/Test Split Operations
    # -------------------------------------------------------------------------

    def split_train_test(
        self,
        holdout_ratio: float = 0.2,
        min_pairs_per_contact: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Split pairs into training and holdout sets by contact.

        All pairs for a contact go to the same set to test generalization
        to new conversation styles, not just new messages from known contacts.

        Args:
            holdout_ratio: Fraction of contacts to hold out (default 0.2 = 20%).
            min_pairs_per_contact: Minimum pairs a contact must have to be
                considered for holdout (default 5).
            seed: Random seed for reproducibility.

        Returns:
            Statistics about the split.
        """
        import random

        if seed is not None:
            random.seed(seed)

        with self.connection() as conn:
            # Get contacts with their pair counts
            cursor = conn.execute(
                """
                SELECT contact_id, COUNT(*) as pair_count
                FROM pairs
                WHERE contact_id IS NOT NULL
                GROUP BY contact_id
                HAVING pair_count >= ?
                """,
                (min_pairs_per_contact,),
            )
            eligible_contacts = [(row["contact_id"], row["pair_count"]) for row in cursor]

            if not eligible_contacts:
                return {
                    "success": False,
                    "error": f"No contacts with >= {min_pairs_per_contact} pairs",
                    "contacts_total": 0,
                    "contacts_holdout": 0,
                }

            # Shuffle and select holdout contacts
            random.shuffle(eligible_contacts)
            num_holdout = max(1, int(len(eligible_contacts) * holdout_ratio))
            holdout_contacts = [c[0] for c in eligible_contacts[:num_holdout]]
            training_contacts = [c[0] for c in eligible_contacts[num_holdout:]]

            # Reset all pairs to training first
            conn.execute("UPDATE pairs SET is_holdout = FALSE")

            # Mark holdout contact pairs
            if holdout_contacts:
                placeholders = ",".join("?" * len(holdout_contacts))
                conn.execute(
                    f"UPDATE pairs SET is_holdout = TRUE WHERE contact_id IN ({placeholders})",
                    holdout_contacts,
                )

            # Get final counts
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE")
            training_pairs = cursor.fetchone()["cnt"]

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE")
            holdout_pairs = cursor.fetchone()["cnt"]

            return {
                "success": True,
                "contacts_total": len(eligible_contacts),
                "contacts_holdout": len(holdout_contacts),
                "contacts_training": len(training_contacts),
                "pairs_training": training_pairs,
                "pairs_holdout": holdout_pairs,
                "holdout_ratio_actual": holdout_pairs / (training_pairs + holdout_pairs)
                if (training_pairs + holdout_pairs) > 0
                else 0,
                "holdout_contact_ids": holdout_contacts,
            }

    def get_training_pairs(self, min_quality: float = 0.0, limit: int = 100000) -> list[Pair]:
        """Get pairs designated for training (is_holdout=False)."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pairs
                WHERE is_holdout = FALSE AND quality_score >= ?
                ORDER BY trigger_timestamp DESC
                LIMIT ?
                """,
                (min_quality, limit),
            )
            return [self._row_to_pair(row) for row in cursor]

    def get_holdout_pairs(self, min_quality: float = 0.0, limit: int = 100000) -> list[Pair]:
        """Get pairs designated for evaluation (is_holdout=True)."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pairs
                WHERE is_holdout = TRUE AND quality_score >= ?
                ORDER BY trigger_timestamp DESC
                LIMIT ?
                """,
                (min_quality, limit),
            )
            return [self._row_to_pair(row) for row in cursor]

    def get_split_stats(self) -> dict[str, Any]:
        """Get statistics about the current train/test split."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Training pairs
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = FALSE")
            stats["training_pairs"] = cursor.fetchone()["cnt"]

            # Holdout pairs
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE is_holdout = TRUE")
            stats["holdout_pairs"] = cursor.fetchone()["cnt"]

            # Training contacts
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT contact_id) as cnt
                FROM pairs WHERE is_holdout = FALSE AND contact_id IS NOT NULL
                """
            )
            stats["training_contacts"] = cursor.fetchone()["cnt"]

            # Holdout contacts
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT contact_id) as cnt
                FROM pairs WHERE is_holdout = TRUE AND contact_id IS NOT NULL
                """
            )
            stats["holdout_contacts"] = cursor.fetchone()["cnt"]

            total = stats["training_pairs"] + stats["holdout_pairs"]
            stats["holdout_ratio"] = stats["holdout_pairs"] / total if total > 0 else 0

            return stats

    # -------------------------------------------------------------------------
    # Cluster Operations (Optional - for later analytics)
    # -------------------------------------------------------------------------

    def add_cluster(
        self,
        name: str,
        description: str | None = None,
        example_triggers: list[str] | None = None,
        example_responses: list[str] | None = None,
    ) -> Cluster:
        """Add or update a cluster."""
        with self.connection() as conn:
            triggers_json = json.dumps(example_triggers or [])
            responses_json = json.dumps(example_responses or [])

            conn.execute(
                """
                INSERT INTO clusters (name, description, example_triggers, example_responses)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    example_triggers = excluded.example_triggers,
                    example_responses = excluded.example_responses
                """,
                (name, description, triggers_json, responses_json),
            )

            cursor = conn.execute("SELECT * FROM clusters WHERE name = ?", (name,))
            row = cursor.fetchone()

            return Cluster(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                example_triggers=json.loads(row["example_triggers"])
                if row["example_triggers"]
                else [],
                example_responses=json.loads(row["example_responses"])
                if row["example_responses"]
                else [],
                created_at=row["created_at"],
            )

    def get_cluster(self, cluster_id: int) -> Cluster | None:
        """Get a cluster by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_cluster(row)
            return None

    def get_cluster_by_name(self, name: str) -> Cluster | None:
        """Get a cluster by name."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_cluster(row)
            return None

    def list_clusters(self) -> list[Cluster]:
        """List all clusters."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters ORDER BY name")
            return [self._row_to_cluster(row) for row in cursor]

    def update_cluster_label(
        self, cluster_id: int, name: str, description: str | None = None
    ) -> bool:
        """Update a cluster's name and description."""
        with self.connection() as conn:
            if description is not None:
                cursor = conn.execute(
                    "UPDATE clusters SET name = ?, description = ? WHERE id = ?",
                    (name, description, cluster_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE clusters SET name = ? WHERE id = ?",
                    (name, cluster_id),
                )
            return cursor.rowcount > 0

    def clear_clusters(self) -> int:
        """Delete all clusters."""
        with self.connection() as conn:
            conn.execute("UPDATE pair_embeddings SET cluster_id = NULL")
            cursor = conn.execute("DELETE FROM clusters")
            return cursor.rowcount

    def _row_to_cluster(self, row: sqlite3.Row) -> Cluster:
        """Convert a database row to a Cluster object."""
        return Cluster(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            example_triggers=json.loads(row["example_triggers"]) if row["example_triggers"] else [],
            example_responses=json.loads(row["example_responses"])
            if row["example_responses"]
            else [],
            created_at=row["created_at"],
        )

    # -------------------------------------------------------------------------
    # Embedding Operations (keyed by pair_id for stability)
    # -------------------------------------------------------------------------

    def add_embedding(
        self,
        pair_id: int,
        faiss_id: int,
        cluster_id: int | None = None,
        index_version: str | None = None,
    ) -> PairEmbedding:
        """Add or update a FAISS embedding reference."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_embeddings
                (pair_id, faiss_id, cluster_id, index_version)
                VALUES (?, ?, ?, ?)
                """,
                (pair_id, faiss_id, cluster_id, index_version),
            )
            return PairEmbedding(
                pair_id=pair_id,
                faiss_id=faiss_id,
                cluster_id=cluster_id,
                index_version=index_version,
            )

    def add_embeddings_bulk(self, embeddings: list[dict[str, Any]]) -> int:
        """Add multiple embeddings in a single transaction."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO pair_embeddings
                (pair_id, faiss_id, cluster_id, index_version)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (e["pair_id"], e["faiss_id"], e.get("cluster_id"), e.get("index_version"))
                    for e in embeddings
                ],
            )
            return len(embeddings)

    def get_embedding_by_pair(self, pair_id: int) -> PairEmbedding | None:
        """Get embedding by pair ID (stable key)."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM pair_embeddings WHERE pair_id = ?", (pair_id,))
            row = cursor.fetchone()
            if row:
                return PairEmbedding(
                    pair_id=row["pair_id"],
                    faiss_id=row["faiss_id"],
                    cluster_id=row["cluster_id"],
                    index_version=row["index_version"] if "index_version" in row.keys() else None,
                )
            return None

    def get_pair_by_faiss_id(self, faiss_id: int, index_version: str | None = None) -> Pair | None:
        """Get the pair associated with a FAISS ID."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    """
                    SELECT p.* FROM pairs p
                    JOIN pair_embeddings e ON p.id = e.pair_id
                    WHERE e.faiss_id = ? AND e.index_version = ?
                    """,
                    (faiss_id, index_version),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT p.* FROM pairs p
                    JOIN pair_embeddings e ON p.id = e.pair_id
                    WHERE e.faiss_id = ?
                    """,
                    (faiss_id,),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_pair(row)
            return None

    def clear_embeddings(self, index_version: str | None = None) -> int:
        """Delete embeddings, optionally for a specific index version."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    "DELETE FROM pair_embeddings WHERE index_version = ?",
                    (index_version,),
                )
            else:
                cursor = conn.execute("DELETE FROM pair_embeddings")
            return cursor.rowcount

    def count_embeddings(self, index_version: str | None = None) -> int:
        """Count embeddings."""
        with self.connection() as conn:
            if index_version:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pair_embeddings WHERE index_version = ?",
                    (index_version,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM pair_embeddings")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    # -------------------------------------------------------------------------
    # Index Version Operations
    # -------------------------------------------------------------------------

    def add_index_version(
        self,
        version_id: str,
        model_name: str,
        embedding_dim: int,
        num_vectors: int,
        index_path: str,
        is_active: bool = False,
    ) -> IndexVersion:
        """Add a new index version."""
        with self.connection() as conn:
            # If setting as active, deactivate others
            if is_active:
                conn.execute("UPDATE index_versions SET is_active = FALSE")

            cursor = conn.execute(
                """
                INSERT INTO index_versions
                (version_id, model_name, embedding_dim, num_vectors, index_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, model_name, embedding_dim, num_vectors, index_path, is_active),
            )

            return IndexVersion(
                id=cursor.lastrowid,
                version_id=version_id,
                model_name=model_name,
                embedding_dim=embedding_dim,
                num_vectors=num_vectors,
                index_path=index_path,
                is_active=is_active,
            )

    def get_active_index(self) -> IndexVersion | None:
        """Get the currently active index version."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM index_versions WHERE is_active = TRUE LIMIT 1")
            row = cursor.fetchone()
            if row:
                return IndexVersion(
                    id=row["id"],
                    version_id=row["version_id"],
                    model_name=row["model_name"],
                    embedding_dim=row["embedding_dim"],
                    num_vectors=row["num_vectors"],
                    index_path=row["index_path"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                )
            return None

    def set_active_index(self, version_id: str) -> bool:
        """Set the active index version."""
        with self.connection() as conn:
            conn.execute("UPDATE index_versions SET is_active = FALSE")
            cursor = conn.execute(
                "UPDATE index_versions SET is_active = TRUE WHERE version_id = ?",
                (version_id,),
            )
            return cursor.rowcount > 0

    def list_index_versions(self) -> list[IndexVersion]:
        """List all index versions."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM index_versions ORDER BY created_at DESC")
            return [
                IndexVersion(
                    id=row["id"],
                    version_id=row["version_id"],
                    model_name=row["model_name"],
                    embedding_dim=row["embedding_dim"],
                    num_vectors=row["num_vectors"],
                    index_path=row["index_path"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                )
                for row in cursor
            ]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Contact count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM contacts")
            stats["contacts"] = cursor.fetchone()["cnt"]

            # Pair count (with quality breakdown)
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs")
            stats["pairs"] = cursor.fetchone()["cnt"]

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE quality_score >= 0.5")
            stats["pairs_quality_gte_50"] = cursor.fetchone()["cnt"]

            # Cluster count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM clusters")
            stats["clusters"] = cursor.fetchone()["cnt"]

            # Embedding count
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pair_embeddings")
            stats["embeddings"] = cursor.fetchone()["cnt"]

            # Active index
            active_index = self.get_active_index()
            stats["active_index"] = active_index.version_id if active_index else None

            # Pairs per contact
            cursor = conn.execute(
                """
                SELECT c.display_name, COUNT(p.id) as pair_count
                FROM contacts c
                LEFT JOIN pairs p ON c.id = p.contact_id
                GROUP BY c.id
                ORDER BY pair_count DESC
                LIMIT 10
                """
            )
            stats["pairs_per_contact"] = [
                {"name": row["display_name"], "count": row["pair_count"]} for row in cursor
            ]

            return stats

    # -------------------------------------------------------------------------
    # Pair Artifact Operations (v6+)
    # -------------------------------------------------------------------------

    def add_artifact(
        self,
        pair_id: int,
        context_json: str | None = None,
        gate_a_reason: str | None = None,
        gate_c_scores_json: str | None = None,
        raw_trigger_text: str | None = None,
        raw_response_text: str | None = None,
    ) -> PairArtifact:
        """Add or update artifacts for a pair.

        Args:
            pair_id: ID of the pair these artifacts belong to.
            context_json: Structured context window (JSON list).
            gate_a_reason: Why Gate A rejected (if rejected).
            gate_c_scores_json: Raw NLI scores (JSON dict).
            raw_trigger_text: Original text before normalization.
            raw_response_text: Original text before normalization.

        Returns:
            The created or updated PairArtifact.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_artifacts
                (pair_id, context_json, gate_a_reason, gate_c_scores_json,
                 raw_trigger_text, raw_response_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_id,
                    context_json,
                    gate_a_reason,
                    gate_c_scores_json,
                    raw_trigger_text,
                    raw_response_text,
                ),
            )
            return PairArtifact(
                pair_id=pair_id,
                context_json=context_json,
                gate_a_reason=gate_a_reason,
                gate_c_scores_json=gate_c_scores_json,
                raw_trigger_text=raw_trigger_text,
                raw_response_text=raw_response_text,
            )

    def get_artifact(self, pair_id: int) -> PairArtifact | None:
        """Get artifacts for a pair."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM pair_artifacts WHERE pair_id = ?", (pair_id,))
            row = cursor.fetchone()
            if row:
                return PairArtifact(
                    pair_id=row["pair_id"],
                    context_json=row["context_json"],
                    gate_a_reason=row["gate_a_reason"],
                    gate_c_scores_json=row["gate_c_scores_json"],
                    raw_trigger_text=row["raw_trigger_text"],
                    raw_response_text=row["raw_response_text"],
                )
            return None

    def clear_artifacts(self) -> int:
        """Delete all artifacts."""
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM pair_artifacts")
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Contact Style Targets Operations (v6+)
    # -------------------------------------------------------------------------

    def set_style_targets(
        self,
        contact_id: int,
        median_reply_length: int = 10,
        punctuation_rate: float = 0.5,
        emoji_rate: float = 0.1,
        greeting_rate: float = 0.2,
    ) -> ContactStyleTargets:
        """Set style targets for a contact.

        Args:
            contact_id: Contact ID.
            median_reply_length: Median word count.
            punctuation_rate: Fraction with ending punctuation.
            emoji_rate: Fraction containing emojis.
            greeting_rate: Fraction starting with greeting.

        Returns:
            The created or updated ContactStyleTargets.
        """
        with self.connection() as conn:
            now = datetime.now()
            conn.execute(
                """
                INSERT OR REPLACE INTO contact_style_targets
                (contact_id, median_reply_length, punctuation_rate,
                 emoji_rate, greeting_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (contact_id, median_reply_length, punctuation_rate, emoji_rate, greeting_rate, now),
            )
            return ContactStyleTargets(
                contact_id=contact_id,
                median_reply_length=median_reply_length,
                punctuation_rate=punctuation_rate,
                emoji_rate=emoji_rate,
                greeting_rate=greeting_rate,
                updated_at=now,
            )

    def get_style_targets(self, contact_id: int) -> ContactStyleTargets | None:
        """Get style targets for a contact."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM contact_style_targets WHERE contact_id = ?",
                (contact_id,),
            )
            row = cursor.fetchone()
            if row:
                return ContactStyleTargets(
                    contact_id=row["contact_id"],
                    median_reply_length=row["median_reply_length"],
                    punctuation_rate=row["punctuation_rate"],
                    emoji_rate=row["emoji_rate"],
                    greeting_rate=row["greeting_rate"],
                    updated_at=row["updated_at"],
                )
            return None

    # -------------------------------------------------------------------------
    # Validated Pair Operations (v6+ extraction pipeline)
    # -------------------------------------------------------------------------

    def add_validated_pair(
        self,
        trigger_text: str,
        response_text: str,
        trigger_timestamp: datetime,
        response_timestamp: datetime,
        chat_id: str,
        contact_id: int | None = None,
        trigger_msg_id: int | None = None,
        response_msg_id: int | None = None,
        trigger_msg_ids: list[int] | None = None,
        response_msg_ids: list[int] | None = None,
        quality_score: float = 1.0,
        flags: dict[str, Any] | None = None,
        is_group: bool = False,
        # Gate results
        gate_a_passed: bool = True,
        gate_b_score: float | None = None,
        gate_c_verdict: str | None = None,
        validity_status: str = "valid",
        # Artifacts (stored separately)
        context_json: str | None = None,
        gate_a_reason: str | None = None,
        gate_c_scores_json: str | None = None,
        raw_trigger_text: str | None = None,
        raw_response_text: str | None = None,
    ) -> Pair | None:
        """Add a validated pair with gate results and artifacts.

        This is the v6+ version of add_pair that includes validity gate
        results and stores heavy artifacts in a separate table.

        Returns:
            The created Pair, or None if duplicate.
        """
        with self.connection() as conn:
            trigger_msg_ids_json = json.dumps(trigger_msg_ids) if trigger_msg_ids else None
            response_msg_ids_json = json.dumps(response_msg_ids) if response_msg_ids else None
            flags_json = json.dumps(flags) if flags else None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO pairs
                    (contact_id, trigger_text, response_text, trigger_timestamp,
                     response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                     trigger_msg_ids_json, response_msg_ids_json,
                     quality_score, flags_json, is_group,
                     gate_a_passed, gate_b_score, gate_c_verdict, validity_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contact_id,
                        trigger_text,
                        response_text,
                        trigger_timestamp,
                        response_timestamp,
                        chat_id,
                        trigger_msg_id,
                        response_msg_id,
                        trigger_msg_ids_json,
                        response_msg_ids_json,
                        quality_score,
                        flags_json,
                        is_group,
                        gate_a_passed,
                        gate_b_score,
                        gate_c_verdict,
                        validity_status,
                    ),
                )
                pair_id = cursor.lastrowid

                # Store artifacts in separate table if provided
                if context_json or gate_a_reason or gate_c_scores_json or raw_trigger_text:
                    conn.execute(
                        """
                        INSERT INTO pair_artifacts
                        (pair_id, context_json, gate_a_reason, gate_c_scores_json,
                         raw_trigger_text, raw_response_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pair_id,
                            context_json,
                            gate_a_reason,
                            gate_c_scores_json,
                            raw_trigger_text,
                            raw_response_text,
                        ),
                    )

                return Pair(
                    id=pair_id,
                    contact_id=contact_id,
                    trigger_text=trigger_text,
                    response_text=response_text,
                    trigger_timestamp=trigger_timestamp,
                    response_timestamp=response_timestamp,
                    chat_id=chat_id,
                    trigger_msg_id=trigger_msg_id,
                    response_msg_id=response_msg_id,
                    trigger_msg_ids_json=trigger_msg_ids_json,
                    response_msg_ids_json=response_msg_ids_json,
                    quality_score=quality_score,
                    flags_json=flags_json,
                    is_group=is_group,
                    gate_a_passed=gate_a_passed,
                    gate_b_score=gate_b_score,
                    gate_c_verdict=gate_c_verdict,
                    validity_status=validity_status,
                )
            except sqlite3.IntegrityError:
                # Duplicate pair
                return None

    def get_gate_stats(self) -> dict[str, Any]:
        """Get statistics about validity gate results."""
        with self.connection() as conn:
            stats: dict[str, Any] = {}

            # Total pairs with gate data
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pairs WHERE validity_status IS NOT NULL"
            )
            stats["total_gated"] = cursor.fetchone()["cnt"]

            # By validity status
            for status in ["valid", "invalid", "uncertain"]:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pairs WHERE validity_status = ?",
                    (status,),
                )
                stats[f"status_{status}"] = cursor.fetchone()["cnt"]

            # Gate A rejections
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM pairs WHERE gate_a_passed = FALSE")
            stats["gate_a_rejected"] = cursor.fetchone()["cnt"]

            # Gate A rejection reasons (from artifacts)
            cursor = conn.execute(
                """
                SELECT gate_a_reason, COUNT(*) as cnt
                FROM pair_artifacts
                WHERE gate_a_reason IS NOT NULL
                GROUP BY gate_a_reason
                ORDER BY cnt DESC
                """
            )
            stats["gate_a_reasons"] = {row["gate_a_reason"]: row["cnt"] for row in cursor}

            # Gate B score distribution
            cursor = conn.execute(
                """
                SELECT
                    CASE
                        WHEN gate_b_score >= 0.62 THEN 'accept'
                        WHEN gate_b_score >= 0.48 THEN 'borderline'
                        ELSE 'reject'
                    END as band,
                    COUNT(*) as cnt
                FROM pairs
                WHERE gate_b_score IS NOT NULL
                GROUP BY band
                """
            )
            stats["gate_b_bands"] = {row["band"]: row["cnt"] for row in cursor}

            # Gate C verdicts
            cursor = conn.execute(
                """
                SELECT gate_c_verdict, COUNT(*) as cnt
                FROM pairs
                WHERE gate_c_verdict IS NOT NULL
                GROUP BY gate_c_verdict
                """
            )
            stats["gate_c_verdicts"] = {row["gate_c_verdict"]: row["cnt"] for row in cursor}

            return stats

    def get_valid_pairs(self, min_quality: float = 0.0, limit: int = 100000) -> list[Pair]:
        """Get pairs with validity_status='valid'."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pairs
                WHERE validity_status = 'valid' AND quality_score >= ?
                ORDER BY trigger_timestamp DESC
                LIMIT ?
                """,
                (min_quality, limit),
            )
            return [self._row_to_pair(row) for row in cursor]


# Singleton instance
_db: JarvisDB | None = None


def get_db(db_path: Path | None = None) -> JarvisDB:
    """Get or create the singleton database instance."""
    global _db
    if _db is None:
        _db = JarvisDB(db_path)
    return _db


def reset_db() -> None:
    """Reset the singleton database instance."""
    global _db
    _db = None
