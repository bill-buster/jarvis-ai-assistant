"""Unit tests for JARVIS Database Management (JarvisDB).

Tests cover database initialization, contact CRUD operations, pair operations,
cluster operations, train/test split, embedding operations, and edge cases.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.db import (
    CURRENT_SCHEMA_VERSION,
    Cluster,
    Contact,
    ContactStyleTargets,
    IndexVersion,
    JarvisDB,
    Pair,
    PairArtifact,
    PairEmbedding,
    get_db,
    reset_db,
)


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that JarvisDB creates parent directory if it doesn't exist."""
        db_path = tmp_path / "nested" / "dir" / "jarvis.db"
        db = JarvisDB(db_path)

        assert db.db_path == db_path
        assert db_path.parent.exists()

    def test_init_with_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test JarvisDB uses default path when none provided."""
        # Use a real tmp_path to avoid filesystem issues
        default_path = tmp_path / "default" / "jarvis.db"
        monkeypatch.setattr("jarvis.db.JARVIS_DB_PATH", default_path)
        db = JarvisDB()  # No path argument - should use JARVIS_DB_PATH
        assert db.db_path == default_path

    def test_exists_returns_false_when_no_file(self, tmp_path: Path) -> None:
        """Test exists() returns False when database file doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        db = JarvisDB(db_path)

        assert db.exists() is False

    def test_exists_returns_true_after_init_schema(self, tmp_path: Path) -> None:
        """Test exists() returns True after schema initialization."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        assert db.exists() is True

    def test_init_schema_creates_tables(self, tmp_path: Path) -> None:
        """Test that init_schema creates all required tables."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        with db.connection() as conn:
            # Check all expected tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row["name"] for row in cursor}

            expected_tables = {
                "schema_version",
                "contacts",
                "contact_style_targets",
                "pairs",
                "pair_artifacts",
                "clusters",
                "pair_embeddings",
                "index_versions",
            }
            assert expected_tables.issubset(tables)

    def test_init_schema_sets_version(self, tmp_path: Path) -> None:
        """Test that init_schema sets the correct schema version."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        with db.connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            assert row["version"] == CURRENT_SCHEMA_VERSION

    def test_init_schema_idempotent(self, tmp_path: Path) -> None:
        """Test that calling init_schema multiple times is safe."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)

        # First call should return True (created)
        result1 = db.init_schema()
        assert result1 is True

        # Second call should return False (already current)
        result2 = db.init_schema()
        assert result2 is False

    def test_connection_enables_foreign_keys(self, tmp_path: Path) -> None:
        """Test that connections have foreign keys enabled."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        with db.connection() as conn:
            cursor = conn.execute("PRAGMA foreign_keys")
            row = cursor.fetchone()
            assert row[0] == 1  # Foreign keys enabled


class TestContactOperations:
    """Tests for contact CRUD operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_add_contact_basic(self, db: JarvisDB) -> None:
        """Test adding a basic contact."""
        contact = db.add_contact(
            display_name="John Doe",
            phone_or_email="+15551234567",
            relationship="friend",
        )

        assert contact.id is not None
        assert contact.display_name == "John Doe"
        assert contact.phone_or_email == "+15551234567"
        assert contact.relationship == "friend"

    def test_add_contact_with_all_fields(self, db: JarvisDB) -> None:
        """Test adding a contact with all fields populated."""
        contact = db.add_contact(
            display_name="Jane Smith",
            chat_id="chat123",
            phone_or_email="jane@example.com",
            relationship="sister",
            style_notes="casual, uses emojis",
            handles=["+15551234567", "jane@example.com"],
        )

        assert contact.display_name == "Jane Smith"
        assert contact.chat_id == "chat123"
        assert contact.phone_or_email == "jane@example.com"
        assert contact.relationship == "sister"
        assert contact.style_notes == "casual, uses emojis"
        assert contact.handles == ["+15551234567", "jane@example.com"]

    def test_add_contact_updates_existing_by_chat_id(self, db: JarvisDB) -> None:
        """Test that adding a contact with existing chat_id updates it."""
        # Add initial contact
        contact1 = db.add_contact(
            display_name="Original Name",
            chat_id="chat123",
            relationship="friend",
        )

        # Add with same chat_id - should update
        contact2 = db.add_contact(
            display_name="Updated Name",
            chat_id="chat123",
            relationship="best friend",
        )

        assert contact2.id == contact1.id
        assert contact2.display_name == "Updated Name"
        assert contact2.relationship == "best friend"

    def test_get_contact_by_id(self, db: JarvisDB) -> None:
        """Test retrieving a contact by ID."""
        created = db.add_contact(display_name="Test User")
        retrieved = db.get_contact(created.id)  # type: ignore[arg-type]

        assert retrieved is not None
        assert retrieved.display_name == "Test User"
        assert retrieved.id == created.id

    def test_get_contact_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_contact returns None for non-existent ID."""
        result = db.get_contact(99999)
        assert result is None

    def test_get_contact_by_chat_id(self, db: JarvisDB) -> None:
        """Test retrieving a contact by chat ID."""
        db.add_contact(display_name="Chat User", chat_id="unique_chat_id")
        retrieved = db.get_contact_by_chat_id("unique_chat_id")

        assert retrieved is not None
        assert retrieved.display_name == "Chat User"
        assert retrieved.chat_id == "unique_chat_id"

    def test_get_contact_by_chat_id_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_contact_by_chat_id returns None for non-existent chat_id."""
        result = db.get_contact_by_chat_id("nonexistent_chat_id")
        assert result is None

    def test_get_contact_by_name_exact_match(self, db: JarvisDB) -> None:
        """Test retrieving a contact by exact name match."""
        db.add_contact(display_name="Alice Johnson")
        retrieved = db.get_contact_by_name("Alice Johnson")

        assert retrieved is not None
        assert retrieved.display_name == "Alice Johnson"

    def test_get_contact_by_name_case_insensitive(self, db: JarvisDB) -> None:
        """Test that name search is case-insensitive."""
        db.add_contact(display_name="Bob Smith")
        retrieved = db.get_contact_by_name("bob smith")

        assert retrieved is not None
        assert retrieved.display_name == "Bob Smith"

    def test_get_contact_by_name_partial_match(self, db: JarvisDB) -> None:
        """Test that name search finds partial matches."""
        db.add_contact(display_name="Christopher Robin")
        retrieved = db.get_contact_by_name("Chris")

        assert retrieved is not None
        assert retrieved.display_name == "Christopher Robin"

    def test_get_contact_by_name_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_contact_by_name returns None for non-existent name."""
        result = db.get_contact_by_name("Nobody Here")
        assert result is None

    def test_list_contacts(self, db: JarvisDB) -> None:
        """Test listing all contacts."""
        db.add_contact(display_name="Alice")
        db.add_contact(display_name="Bob")
        db.add_contact(display_name="Charlie")

        contacts = db.list_contacts()

        assert len(contacts) == 3
        names = {c.display_name for c in contacts}
        assert names == {"Alice", "Bob", "Charlie"}

    def test_list_contacts_respects_limit(self, db: JarvisDB) -> None:
        """Test that list_contacts respects the limit parameter."""
        for i in range(10):
            db.add_contact(display_name=f"User{i:02d}")

        contacts = db.list_contacts(limit=5)
        assert len(contacts) == 5

    def test_list_contacts_ordered_by_name(self, db: JarvisDB) -> None:
        """Test that contacts are ordered by display name."""
        db.add_contact(display_name="Zebra")
        db.add_contact(display_name="Apple")
        db.add_contact(display_name="Mango")

        contacts = db.list_contacts()
        names = [c.display_name for c in contacts]

        assert names == ["Apple", "Mango", "Zebra"]

    def test_delete_contact(self, db: JarvisDB) -> None:
        """Test deleting a contact."""
        contact = db.add_contact(display_name="To Delete")
        result = db.delete_contact(contact.id)  # type: ignore[arg-type]

        assert result is True
        assert db.get_contact(contact.id) is None  # type: ignore[arg-type]

    def test_delete_contact_returns_false_for_missing(self, db: JarvisDB) -> None:
        """Test that delete_contact returns False for non-existent ID."""
        result = db.delete_contact(99999)
        assert result is False

    def test_delete_contact_cascades_to_pairs(self, db: JarvisDB) -> None:
        """Test that deleting a contact also deletes their pairs."""
        contact = db.add_contact(display_name="Has Pairs", chat_id="chat_with_pairs")

        # Add a pair for this contact
        now = datetime.now()
        db.add_pair(
            trigger_text="Hello",
            response_text="Hi there",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat_with_pairs",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )

        # Verify pair exists
        pairs = db.get_pairs(contact_id=contact.id)
        assert len(pairs) == 1

        # Delete contact
        db.delete_contact(contact.id)  # type: ignore[arg-type]

        # Verify pairs are also deleted
        pairs = db.get_pairs(contact_id=contact.id)
        assert len(pairs) == 0


class TestContactDataclass:
    """Tests for Contact dataclass."""

    def test_handles_property_parses_json(self) -> None:
        """Test that handles property correctly parses JSON."""
        contact = Contact(
            id=1,
            chat_id="chat1",
            display_name="Test",
            phone_or_email=None,
            relationship=None,
            style_notes=None,
            handles_json='["+15551234567", "test@example.com"]',
        )

        assert contact.handles == ["+15551234567", "test@example.com"]

    def test_handles_property_returns_empty_for_none(self) -> None:
        """Test that handles property returns empty list when JSON is None."""
        contact = Contact(
            id=1,
            chat_id="chat1",
            display_name="Test",
            phone_or_email=None,
            relationship=None,
            style_notes=None,
            handles_json=None,
        )

        assert contact.handles == []

    def test_handles_property_returns_empty_for_invalid_json(self) -> None:
        """Test that handles property returns empty list for invalid JSON."""
        contact = Contact(
            id=1,
            chat_id="chat1",
            display_name="Test",
            phone_or_email=None,
            relationship=None,
            style_notes=None,
            handles_json="not valid json",
        )

        assert contact.handles == []


class TestPairOperations:
    """Tests for pair CRUD operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    @pytest.fixture
    def contact(self, db: JarvisDB) -> Contact:
        """Create a contact for pair tests."""
        return db.add_contact(display_name="Test Contact", chat_id="chat123")

    def test_add_pair_basic(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding a basic pair."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="How are you?",
            response_text="I'm doing well, thanks!",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=101,
        )

        assert pair is not None
        assert pair.id is not None
        assert pair.trigger_text == "How are you?"
        assert pair.response_text == "I'm doing well, thanks!"
        assert pair.contact_id == contact.id

    def test_add_pair_with_context(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding a pair with conversation context."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="Yes, let's do that",
            response_text="Great, I'll set it up",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=101,
            context_text="Previous: Should we meet tomorrow?",
        )

        assert pair is not None
        assert pair.context_text == "Previous: Should we meet tomorrow?"

    def test_add_pair_with_quality_and_flags(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding a pair with quality score and flags."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="ok",
            response_text="cool",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=101,
            quality_score=0.3,
            flags={"short": True, "low_content": True},
        )

        assert pair is not None
        assert pair.quality_score == 0.3
        assert pair.flags == {"short": True, "low_content": True}

    def test_add_pair_with_multi_message_ids(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding a pair with multiple message IDs."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="Long trigger spanning multiple messages",
            response_text="Long response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=103,
            trigger_msg_ids=[100, 101],
            response_msg_ids=[103, 104, 105],
        )

        assert pair is not None
        assert pair.trigger_msg_ids == [100, 101]
        assert pair.response_msg_ids == [103, 104, 105]

    def test_add_pair_duplicate_returns_none(self, db: JarvisDB, contact: Contact) -> None:
        """Test that adding a duplicate pair returns None."""
        now = datetime.now()
        pair1 = db.add_pair(
            trigger_text="Hello",
            response_text="Hi",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=101,
        )
        assert pair1 is not None

        # Try to add with same message IDs
        pair2 = db.add_pair(
            trigger_text="Hello again",
            response_text="Hi again",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=100,
            response_msg_id=101,
        )
        assert pair2 is None

    def test_add_pairs_bulk(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding multiple pairs in bulk."""
        now = datetime.now()
        pairs_data = [
            {
                "contact_id": contact.id,
                "trigger_text": f"Trigger {i}",
                "response_text": f"Response {i}",
                "trigger_timestamp": now,
                "response_timestamp": now,
                "chat_id": "chat123",
                "trigger_msg_id": i * 2,
                "response_msg_id": i * 2 + 1,
            }
            for i in range(5)
        ]

        added = db.add_pairs_bulk(pairs_data)

        assert added == 5
        assert db.count_pairs() == 5

    def test_add_pairs_bulk_skips_duplicates(self, db: JarvisDB, contact: Contact) -> None:
        """Test that bulk add skips duplicate pairs."""
        now = datetime.now()

        # Add one pair first
        db.add_pair(
            trigger_text="Existing",
            response_text="Already here",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=0,
            response_msg_id=1,
        )

        # Try to add bulk including the duplicate
        pairs_data = [
            {
                "contact_id": contact.id,
                "trigger_text": f"Trigger {i}",
                "response_text": f"Response {i}",
                "trigger_timestamp": now,
                "response_timestamp": now,
                "chat_id": "chat123",
                "trigger_msg_id": i * 2,
                "response_msg_id": i * 2 + 1,
            }
            for i in range(3)  # Includes msg_ids 0,1 which already exist
        ]

        added = db.add_pairs_bulk(pairs_data)

        assert added == 2  # Only 2 new pairs added
        assert db.count_pairs() == 3  # 1 existing + 2 new

    def test_get_pairs_all(self, db: JarvisDB, contact: Contact) -> None:
        """Test retrieving all pairs."""
        now = datetime.now()
        for i in range(3):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat123",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )

        pairs = db.get_pairs()
        assert len(pairs) == 3

    def test_get_pairs_by_contact_id(self, db: JarvisDB, contact: Contact) -> None:
        """Test filtering pairs by contact ID."""
        other_contact = db.add_contact(display_name="Other", chat_id="other_chat")
        now = datetime.now()

        # Add pairs for both contacts
        db.add_pair(
            trigger_text="For main contact",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )
        db.add_pair(
            trigger_text="For other contact",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="other_chat",
            contact_id=other_contact.id,
            trigger_msg_id=3,
            response_msg_id=4,
        )

        pairs = db.get_pairs(contact_id=contact.id)
        assert len(pairs) == 1
        assert pairs[0].trigger_text == "For main contact"

    def test_get_pairs_by_min_quality(self, db: JarvisDB, contact: Contact) -> None:
        """Test filtering pairs by minimum quality score."""
        now = datetime.now()

        db.add_pair(
            trigger_text="Low quality",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            quality_score=0.3,
        )
        db.add_pair(
            trigger_text="High quality",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=3,
            response_msg_id=4,
            quality_score=0.9,
        )

        pairs = db.get_pairs(min_quality=0.5)
        assert len(pairs) == 1
        assert pairs[0].trigger_text == "High quality"

    def test_get_all_pairs(self, db: JarvisDB, contact: Contact) -> None:
        """Test get_all_pairs convenience method."""
        now = datetime.now()
        for i in range(3):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat123",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )

        pairs = db.get_all_pairs()
        assert len(pairs) == 3

    def test_count_pairs(self, db: JarvisDB, contact: Contact) -> None:
        """Test counting pairs in database."""
        assert db.count_pairs() == 0

        now = datetime.now()
        for i in range(5):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat123",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )

        assert db.count_pairs() == 5

    def test_update_pair_quality(self, db: JarvisDB, contact: Contact) -> None:
        """Test updating a pair's quality score."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="Test",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            quality_score=1.0,
        )

        result = db.update_pair_quality(pair.id, 0.5, flags={"recalculated": True})  # type: ignore[arg-type]
        assert result is True

        # Verify update
        pairs = db.get_pairs()
        assert pairs[0].quality_score == 0.5
        assert pairs[0].flags == {"recalculated": True}

    def test_update_pair_quality_returns_false_for_missing(self, db: JarvisDB) -> None:
        """Test that update_pair_quality returns False for non-existent ID."""
        result = db.update_pair_quality(99999, 0.5)
        assert result is False

    def test_clear_pairs(self, db: JarvisDB, contact: Contact) -> None:
        """Test clearing all pairs."""
        now = datetime.now()
        for i in range(3):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat123",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )

        assert db.count_pairs() == 3

        deleted = db.clear_pairs()
        assert deleted == 3
        assert db.count_pairs() == 0

    def test_get_pairs_by_trigger_pattern(self, db: JarvisDB, contact: Contact) -> None:
        """Test retrieving pairs by trigger pattern."""
        now = datetime.now()

        # Add acknowledgment pairs
        for trigger in ["ok", "sure", "yes"]:
            db.add_pair(
                trigger_text=trigger,
                response_text="Following up on that...",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat123",
                contact_id=contact.id,
                trigger_msg_id=hash(trigger) % 10000,
                response_msg_id=hash(trigger) % 10000 + 1,
            )

        # Add non-acknowledgment pair
        db.add_pair(
            trigger_text="What time is the meeting?",
            response_text="3pm",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat123",
            contact_id=contact.id,
            trigger_msg_id=9999,
            response_msg_id=9998,
        )

        pairs = db.get_pairs_by_trigger_pattern(contact.id, pattern_type="acknowledgment")  # type: ignore[arg-type]
        assert len(pairs) == 3

    def test_get_pairs_by_trigger_pattern_unknown_type(
        self, db: JarvisDB, contact: Contact
    ) -> None:
        """Test that unknown pattern type returns empty list."""
        pairs = db.get_pairs_by_trigger_pattern(contact.id, pattern_type="unknown")  # type: ignore[arg-type]
        assert pairs == []


class TestPairDataclass:
    """Tests for Pair dataclass."""

    def test_flags_property_parses_json(self) -> None:
        """Test that flags property correctly parses JSON."""
        pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="test",
            response_text="test",
            trigger_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
            chat_id="chat1",
            flags_json='{"short": true, "attachment": false}',
        )

        assert pair.flags == {"short": True, "attachment": False}

    def test_flags_property_returns_empty_for_none(self) -> None:
        """Test that flags property returns empty dict when JSON is None."""
        pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="test",
            response_text="test",
            trigger_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
            chat_id="chat1",
            flags_json=None,
        )

        assert pair.flags == {}

    def test_trigger_msg_ids_property(self) -> None:
        """Test trigger_msg_ids property."""
        pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="test",
            response_text="test",
            trigger_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
            chat_id="chat1",
            trigger_msg_ids_json="[1, 2, 3]",
        )

        assert pair.trigger_msg_ids == [1, 2, 3]

    def test_trigger_msg_ids_falls_back_to_single_id(self) -> None:
        """Test that trigger_msg_ids falls back to trigger_msg_id."""
        pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="test",
            response_text="test",
            trigger_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
            chat_id="chat1",
            trigger_msg_id=42,
            trigger_msg_ids_json=None,
        )

        assert pair.trigger_msg_ids == [42]

    def test_response_msg_ids_property(self) -> None:
        """Test response_msg_ids property."""
        pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="test",
            response_text="test",
            trigger_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
            chat_id="chat1",
            response_msg_ids_json="[10, 11]",
        )

        assert pair.response_msg_ids == [10, 11]


class TestClusterOperations:
    """Tests for cluster CRUD operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_add_cluster_basic(self, db: JarvisDB) -> None:
        """Test adding a basic cluster."""
        cluster = db.add_cluster(
            name="GREETING",
            description="Greetings and salutations",
        )

        assert cluster.id is not None
        assert cluster.name == "GREETING"
        assert cluster.description == "Greetings and salutations"

    def test_add_cluster_with_examples(self, db: JarvisDB) -> None:
        """Test adding a cluster with example triggers and responses."""
        cluster = db.add_cluster(
            name="INVITATION",
            description="Event invitations",
            example_triggers=["Want to come to dinner?", "Are you free Saturday?"],
            example_responses=["Sure, I'd love to!", "What time?"],
        )

        assert cluster.example_triggers == ["Want to come to dinner?", "Are you free Saturday?"]
        assert cluster.example_responses == ["Sure, I'd love to!", "What time?"]

    def test_add_cluster_updates_existing(self, db: JarvisDB) -> None:
        """Test that adding a cluster with existing name updates it."""
        cluster1 = db.add_cluster(
            name="SCHEDULE",
            description="Original description",
        )

        cluster2 = db.add_cluster(
            name="SCHEDULE",
            description="Updated description",
        )

        assert cluster2.id == cluster1.id
        assert cluster2.description == "Updated description"

    def test_get_cluster_by_id(self, db: JarvisDB) -> None:
        """Test retrieving a cluster by ID."""
        created = db.add_cluster(name="TEST_CLUSTER")
        retrieved = db.get_cluster(created.id)  # type: ignore[arg-type]

        assert retrieved is not None
        assert retrieved.name == "TEST_CLUSTER"

    def test_get_cluster_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_cluster returns None for non-existent ID."""
        result = db.get_cluster(99999)
        assert result is None

    def test_get_cluster_by_name(self, db: JarvisDB) -> None:
        """Test retrieving a cluster by name."""
        db.add_cluster(name="UNIQUE_CLUSTER", description="Test")
        retrieved = db.get_cluster_by_name("UNIQUE_CLUSTER")

        assert retrieved is not None
        assert retrieved.description == "Test"

    def test_get_cluster_by_name_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_cluster_by_name returns None for non-existent name."""
        result = db.get_cluster_by_name("NONEXISTENT")
        assert result is None

    def test_list_clusters(self, db: JarvisDB) -> None:
        """Test listing all clusters."""
        db.add_cluster(name="ALPHA")
        db.add_cluster(name="BETA")
        db.add_cluster(name="GAMMA")

        clusters = db.list_clusters()

        assert len(clusters) == 3
        names = {c.name for c in clusters}
        assert names == {"ALPHA", "BETA", "GAMMA"}

    def test_update_cluster_label(self, db: JarvisDB) -> None:
        """Test updating a cluster's name and description."""
        cluster = db.add_cluster(name="OLD_NAME", description="Old description")
        result = db.update_cluster_label(
            cluster.id,
            name="NEW_NAME",
            description="New description",  # type: ignore[arg-type]
        )

        assert result is True

        retrieved = db.get_cluster(cluster.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.name == "NEW_NAME"
        assert retrieved.description == "New description"

    def test_update_cluster_label_returns_false_for_missing(self, db: JarvisDB) -> None:
        """Test that update_cluster_label returns False for non-existent ID."""
        result = db.update_cluster_label(99999, name="WHATEVER")
        assert result is False

    def test_clear_clusters(self, db: JarvisDB) -> None:
        """Test clearing all clusters."""
        db.add_cluster(name="CLUSTER1")
        db.add_cluster(name="CLUSTER2")
        db.add_cluster(name="CLUSTER3")

        deleted = db.clear_clusters()

        assert deleted == 3
        assert len(db.list_clusters()) == 0


class TestTrainTestSplit:
    """Tests for train/test split operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def _add_pairs_for_contact(
        self, db: JarvisDB, contact_id: int, num_pairs: int, start_msg_id: int = 0
    ) -> None:
        """Helper to add pairs for a contact."""
        now = datetime.now()
        for i in range(num_pairs):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id=f"chat_{contact_id}",
                contact_id=contact_id,
                trigger_msg_id=start_msg_id + i * 2,
                response_msg_id=start_msg_id + i * 2 + 1,
            )

    def test_split_train_test_basic(self, db: JarvisDB) -> None:
        """Test basic train/test split."""
        # Create contacts with enough pairs
        contact1 = db.add_contact(display_name="Contact1")
        contact2 = db.add_contact(display_name="Contact2")
        contact3 = db.add_contact(display_name="Contact3")
        contact4 = db.add_contact(display_name="Contact4")
        contact5 = db.add_contact(display_name="Contact5")

        # Add 10 pairs per contact (meets min_pairs_per_contact)
        for i, contact in enumerate([contact1, contact2, contact3, contact4, contact5]):
            self._add_pairs_for_contact(db, contact.id, 10, start_msg_id=i * 100)  # type: ignore[arg-type]

        result = db.split_train_test(holdout_ratio=0.2, min_pairs_per_contact=5, seed=42)

        assert result["success"] is True
        assert result["contacts_total"] == 5
        assert result["contacts_holdout"] == 1  # 20% of 5 = 1
        assert result["contacts_training"] == 4
        assert result["pairs_holdout"] == 10  # All pairs from 1 contact
        assert result["pairs_training"] == 40  # All pairs from 4 contacts

    def test_split_train_test_respects_min_pairs(self, db: JarvisDB) -> None:
        """Test that split respects minimum pairs per contact."""
        contact1 = db.add_contact(display_name="Contact1")
        contact2 = db.add_contact(display_name="Contact2")

        # Contact1 has 10 pairs, Contact2 has only 3
        self._add_pairs_for_contact(db, contact1.id, 10, start_msg_id=0)  # type: ignore[arg-type]
        self._add_pairs_for_contact(db, contact2.id, 3, start_msg_id=100)  # type: ignore[arg-type]

        result = db.split_train_test(min_pairs_per_contact=5, seed=42)

        # Only contact1 should be eligible (contact2 has < 5 pairs)
        assert result["success"] is True
        assert result["contacts_total"] == 1

    def test_split_train_test_no_eligible_contacts(self, db: JarvisDB) -> None:
        """Test split when no contacts have enough pairs."""
        contact = db.add_contact(display_name="Contact1")
        self._add_pairs_for_contact(db, contact.id, 2)  # type: ignore[arg-type]

        result = db.split_train_test(min_pairs_per_contact=10)

        assert result["success"] is False
        assert "No contacts with >= 10 pairs" in result["error"]

    def test_split_train_test_reproducible_with_seed(self, db: JarvisDB) -> None:
        """Test that split is reproducible with same seed."""
        # Create contacts
        for i in range(10):
            contact = db.add_contact(display_name=f"Contact{i}")
            self._add_pairs_for_contact(db, contact.id, 10, start_msg_id=i * 100)  # type: ignore[arg-type]

        result1 = db.split_train_test(holdout_ratio=0.3, seed=123)
        holdout1 = set(result1["holdout_contact_ids"])

        # Reset and redo
        with db.connection() as conn:
            conn.execute("UPDATE pairs SET is_holdout = FALSE")

        result2 = db.split_train_test(holdout_ratio=0.3, seed=123)
        holdout2 = set(result2["holdout_contact_ids"])

        assert holdout1 == holdout2

    def test_get_training_pairs(self, db: JarvisDB) -> None:
        """Test retrieving training pairs."""
        contact1 = db.add_contact(display_name="Training Contact")
        contact2 = db.add_contact(display_name="Holdout Contact")

        self._add_pairs_for_contact(db, contact1.id, 10, start_msg_id=0)  # type: ignore[arg-type]
        self._add_pairs_for_contact(db, contact2.id, 10, start_msg_id=100)  # type: ignore[arg-type]

        # Force contact2 to holdout
        with db.connection() as conn:
            conn.execute("UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?", (contact2.id,))

        training_pairs = db.get_training_pairs()
        assert len(training_pairs) == 10
        for pair in training_pairs:
            assert pair.contact_id == contact1.id

    def test_get_holdout_pairs(self, db: JarvisDB) -> None:
        """Test retrieving holdout pairs."""
        contact1 = db.add_contact(display_name="Training Contact")
        contact2 = db.add_contact(display_name="Holdout Contact")

        self._add_pairs_for_contact(db, contact1.id, 10, start_msg_id=0)  # type: ignore[arg-type]
        self._add_pairs_for_contact(db, contact2.id, 10, start_msg_id=100)  # type: ignore[arg-type]

        # Force contact2 to holdout
        with db.connection() as conn:
            conn.execute("UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?", (contact2.id,))

        holdout_pairs = db.get_holdout_pairs()
        assert len(holdout_pairs) == 10
        for pair in holdout_pairs:
            assert pair.contact_id == contact2.id

    def test_get_training_pairs_respects_min_quality(self, db: JarvisDB) -> None:
        """Test that get_training_pairs respects min_quality filter."""
        contact = db.add_contact(display_name="Contact")
        now = datetime.now()

        # Add pairs with different quality scores
        db.add_pair(
            trigger_text="Low quality",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            quality_score=0.2,
        )
        db.add_pair(
            trigger_text="High quality",
            response_text="Response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=3,
            response_msg_id=4,
            quality_score=0.9,
        )

        pairs = db.get_training_pairs(min_quality=0.5)
        assert len(pairs) == 1
        assert pairs[0].trigger_text == "High quality"

    def test_get_split_stats(self, db: JarvisDB) -> None:
        """Test getting split statistics."""
        contact1 = db.add_contact(display_name="Contact1")
        contact2 = db.add_contact(display_name="Contact2")

        self._add_pairs_for_contact(db, contact1.id, 15, start_msg_id=0)  # type: ignore[arg-type]
        self._add_pairs_for_contact(db, contact2.id, 10, start_msg_id=100)  # type: ignore[arg-type]

        # Mark contact2 as holdout
        with db.connection() as conn:
            conn.execute("UPDATE pairs SET is_holdout = TRUE WHERE contact_id = ?", (contact2.id,))

        stats = db.get_split_stats()

        assert stats["training_pairs"] == 15
        assert stats["holdout_pairs"] == 10
        assert stats["training_contacts"] == 1
        assert stats["holdout_contacts"] == 1
        assert abs(stats["holdout_ratio"] - 0.4) < 0.01  # 10/25 = 0.4

    def test_get_split_stats_empty_database(self, db: JarvisDB) -> None:
        """Test split stats on empty database."""
        stats = db.get_split_stats()

        assert stats["training_pairs"] == 0
        assert stats["holdout_pairs"] == 0
        assert stats["training_contacts"] == 0
        assert stats["holdout_contacts"] == 0
        assert stats["holdout_ratio"] == 0


class TestEmbeddingOperations:
    """Tests for embedding and FAISS index operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    @pytest.fixture
    def pair(self, db: JarvisDB) -> Pair:
        """Create a pair for embedding tests."""
        contact = db.add_contact(display_name="Test", chat_id="chat1")
        now = datetime.now()
        return db.add_pair(
            trigger_text="Test trigger",
            response_text="Test response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )  # type: ignore[return-value]

    def test_add_embedding(self, db: JarvisDB, pair: Pair) -> None:
        """Test adding an embedding reference."""
        embedding = db.add_embedding(
            pair_id=pair.id,  # type: ignore[arg-type]
            faiss_id=0,
            cluster_id=None,
            index_version="20240115-143022",
        )

        assert embedding.pair_id == pair.id
        assert embedding.faiss_id == 0
        assert embedding.index_version == "20240115-143022"

    def test_add_embeddings_bulk(self, db: JarvisDB) -> None:
        """Test adding multiple embeddings in bulk."""
        contact = db.add_contact(display_name="Test", chat_id="chat1")
        now = datetime.now()

        # Create multiple pairs
        pairs = []
        for i in range(5):
            pair = db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat1",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )
            pairs.append(pair)

        embeddings_data = [
            {"pair_id": p.id, "faiss_id": i, "index_version": "v1"} for i, p in enumerate(pairs)
        ]

        added = db.add_embeddings_bulk(embeddings_data)

        assert added == 5
        assert db.count_embeddings() == 5

    def test_get_embedding_by_pair(self, db: JarvisDB, pair: Pair) -> None:
        """Test retrieving embedding by pair ID."""
        db.add_embedding(pair_id=pair.id, faiss_id=42, index_version="v1")  # type: ignore[arg-type]

        embedding = db.get_embedding_by_pair(pair.id)  # type: ignore[arg-type]

        assert embedding is not None
        assert embedding.faiss_id == 42
        assert embedding.index_version == "v1"

    def test_get_embedding_by_pair_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_embedding_by_pair returns None for non-existent pair."""
        result = db.get_embedding_by_pair(99999)
        assert result is None

    def test_get_pair_by_faiss_id(self, db: JarvisDB, pair: Pair) -> None:
        """Test retrieving pair by FAISS ID."""
        db.add_embedding(pair_id=pair.id, faiss_id=42, index_version="v1")  # type: ignore[arg-type]

        retrieved = db.get_pair_by_faiss_id(42)

        assert retrieved is not None
        assert retrieved.id == pair.id
        assert retrieved.trigger_text == "Test trigger"

    def test_get_pair_by_faiss_id_with_version(self, db: JarvisDB, pair: Pair) -> None:
        """Test retrieving pair by FAISS ID with version filter."""
        db.add_embedding(pair_id=pair.id, faiss_id=42, index_version="v1")  # type: ignore[arg-type]

        # Should find with correct version
        found = db.get_pair_by_faiss_id(42, index_version="v1")
        assert found is not None

        # Should not find with wrong version
        not_found = db.get_pair_by_faiss_id(42, index_version="v2")
        assert not_found is None

    def test_clear_embeddings(self, db: JarvisDB, pair: Pair) -> None:
        """Test clearing all embeddings."""
        db.add_embedding(pair_id=pair.id, faiss_id=0, index_version="v1")  # type: ignore[arg-type]

        deleted = db.clear_embeddings()

        assert deleted == 1
        assert db.count_embeddings() == 0

    def test_clear_embeddings_by_version(self, db: JarvisDB) -> None:
        """Test clearing embeddings by index version."""
        contact = db.add_contact(display_name="Test", chat_id="chat1")
        now = datetime.now()

        # Create pairs and embeddings for different versions
        for i, version in enumerate(["v1", "v1", "v2"]):
            pair = db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat1",
                contact_id=contact.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
            )
            db.add_embedding(pair_id=pair.id, faiss_id=i, index_version=version)  # type: ignore[arg-type]

        # Clear only v1
        deleted = db.clear_embeddings(index_version="v1")

        assert deleted == 2
        assert db.count_embeddings() == 1
        assert db.count_embeddings(index_version="v2") == 1

    def test_count_embeddings(self, db: JarvisDB, pair: Pair) -> None:
        """Test counting embeddings."""
        assert db.count_embeddings() == 0

        db.add_embedding(pair_id=pair.id, faiss_id=0, index_version="v1")  # type: ignore[arg-type]

        assert db.count_embeddings() == 1


class TestIndexVersionOperations:
    """Tests for index version management."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_add_index_version(self, db: JarvisDB) -> None:
        """Test adding an index version."""
        index = db.add_index_version(
            version_id="20240115-143022",
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
            num_vectors=1000,
            index_path="indexes/20240115-143022.faiss",
            is_active=False,
        )

        assert index.id is not None
        assert index.version_id == "20240115-143022"
        assert index.model_name == "BAAI/bge-small-en-v1.5"
        assert index.embedding_dim == 384
        assert index.num_vectors == 1000
        assert index.is_active is False

    def test_add_index_version_sets_active(self, db: JarvisDB) -> None:
        """Test that setting is_active=True deactivates others."""
        # Add first index as active
        db.add_index_version(
            version_id="v1",
            model_name="model",
            embedding_dim=384,
            num_vectors=100,
            index_path="v1.faiss",
            is_active=True,
        )

        # Add second index as active
        db.add_index_version(
            version_id="v2",
            model_name="model",
            embedding_dim=384,
            num_vectors=200,
            index_path="v2.faiss",
            is_active=True,
        )

        # Only v2 should be active
        active = db.get_active_index()
        assert active is not None
        assert active.version_id == "v2"

    def test_get_active_index(self, db: JarvisDB) -> None:
        """Test retrieving the active index."""
        db.add_index_version(
            version_id="active_index",
            model_name="model",
            embedding_dim=384,
            num_vectors=500,
            index_path="active.faiss",
            is_active=True,
        )

        active = db.get_active_index()

        assert active is not None
        assert active.version_id == "active_index"
        assert active.is_active  # SQLite returns 1 for True

    def test_get_active_index_returns_none_when_none_active(self, db: JarvisDB) -> None:
        """Test that get_active_index returns None when no active index."""
        db.add_index_version(
            version_id="inactive",
            model_name="model",
            embedding_dim=384,
            num_vectors=100,
            index_path="inactive.faiss",
            is_active=False,
        )

        active = db.get_active_index()
        assert active is None

    def test_set_active_index(self, db: JarvisDB) -> None:
        """Test setting the active index."""
        db.add_index_version(
            version_id="v1",
            model_name="model",
            embedding_dim=384,
            num_vectors=100,
            index_path="v1.faiss",
            is_active=True,
        )
        db.add_index_version(
            version_id="v2",
            model_name="model",
            embedding_dim=384,
            num_vectors=200,
            index_path="v2.faiss",
            is_active=False,
        )

        result = db.set_active_index("v2")

        assert result is True
        active = db.get_active_index()
        assert active is not None
        assert active.version_id == "v2"

    def test_set_active_index_returns_false_for_missing(self, db: JarvisDB) -> None:
        """Test that set_active_index returns False for non-existent version."""
        result = db.set_active_index("nonexistent")
        assert result is False

    def test_list_index_versions(self, db: JarvisDB) -> None:
        """Test listing all index versions."""
        db.add_index_version(
            version_id="v1",
            model_name="model",
            embedding_dim=384,
            num_vectors=100,
            index_path="v1.faiss",
        )
        db.add_index_version(
            version_id="v2",
            model_name="model",
            embedding_dim=384,
            num_vectors=200,
            index_path="v2.faiss",
        )

        versions = db.list_index_versions()

        assert len(versions) == 2
        version_ids = {v.version_id for v in versions}
        assert version_ids == {"v1", "v2"}


class TestStatistics:
    """Tests for database statistics."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_get_stats_empty_database(self, db: JarvisDB) -> None:
        """Test statistics on empty database."""
        stats = db.get_stats()

        assert stats["contacts"] == 0
        assert stats["pairs"] == 0
        assert stats["pairs_quality_gte_50"] == 0
        assert stats["clusters"] == 0
        assert stats["embeddings"] == 0
        assert stats["active_index"] is None

    def test_get_stats_with_data(self, db: JarvisDB) -> None:
        """Test statistics with populated database."""
        # Add contacts
        contact1 = db.add_contact(display_name="Contact1", chat_id="chat1")
        contact2 = db.add_contact(display_name="Contact2", chat_id="chat2")

        # Add pairs with varying quality
        now = datetime.now()
        for i in range(5):
            db.add_pair(
                trigger_text=f"Trigger {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat1",
                contact_id=contact1.id,
                trigger_msg_id=i * 2,
                response_msg_id=i * 2 + 1,
                quality_score=0.3 if i < 2 else 0.8,
            )

        # Add clusters
        cluster = db.add_cluster(name="TEST_CLUSTER")

        # Add embeddings
        pairs = db.get_pairs()
        for i, pair in enumerate(pairs[:3]):
            db.add_embedding(pair_id=pair.id, faiss_id=i)  # type: ignore[arg-type]

        # Add index
        db.add_index_version(
            version_id="test_v1",
            model_name="model",
            embedding_dim=384,
            num_vectors=3,
            index_path="test.faiss",
            is_active=True,
        )

        stats = db.get_stats()

        assert stats["contacts"] == 2
        assert stats["pairs"] == 5
        assert stats["pairs_quality_gte_50"] == 3  # 3 pairs with quality >= 0.5
        assert stats["clusters"] == 1
        assert stats["embeddings"] == 3
        assert stats["active_index"] == "test_v1"
        assert len(stats["pairs_per_contact"]) > 0


class TestPairArtifacts:
    """Tests for pair artifact operations (v6+)."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    @pytest.fixture
    def pair(self, db: JarvisDB) -> Pair:
        """Create a pair for artifact tests."""
        contact = db.add_contact(display_name="Test", chat_id="chat1")
        now = datetime.now()
        return db.add_pair(
            trigger_text="Test trigger",
            response_text="Test response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )  # type: ignore[return-value]

    def test_add_artifact(self, db: JarvisDB, pair: Pair) -> None:
        """Test adding pair artifacts."""
        artifact = db.add_artifact(
            pair_id=pair.id,  # type: ignore[arg-type]
            context_json='[{"role": "user", "text": "Previous message"}]',
            gate_a_reason=None,
            gate_c_scores_json='{"entailment": 0.8, "contradiction": 0.1}',
            raw_trigger_text="Test trigger!!!",
            raw_response_text="Test response...",
        )

        assert artifact.pair_id == pair.id
        assert artifact.context_messages == [{"role": "user", "text": "Previous message"}]
        assert artifact.gate_c_scores == {"entailment": 0.8, "contradiction": 0.1}

    def test_get_artifact(self, db: JarvisDB, pair: Pair) -> None:
        """Test retrieving pair artifacts."""
        db.add_artifact(
            pair_id=pair.id,  # type: ignore[arg-type]
            gate_a_reason="Too short",
            raw_trigger_text="ok",
        )

        artifact = db.get_artifact(pair.id)  # type: ignore[arg-type]

        assert artifact is not None
        assert artifact.gate_a_reason == "Too short"
        assert artifact.raw_trigger_text == "ok"

    def test_get_artifact_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_artifact returns None for non-existent pair."""
        result = db.get_artifact(99999)
        assert result is None

    def test_clear_artifacts(self, db: JarvisDB, pair: Pair) -> None:
        """Test clearing all artifacts."""
        db.add_artifact(pair_id=pair.id, raw_trigger_text="test")  # type: ignore[arg-type]

        deleted = db.clear_artifacts()

        assert deleted == 1
        assert db.get_artifact(pair.id) is None  # type: ignore[arg-type]


class TestContactStyleTargets:
    """Tests for contact style target operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    @pytest.fixture
    def contact(self, db: JarvisDB) -> Contact:
        """Create a contact for style target tests."""
        return db.add_contact(display_name="Test Contact", chat_id="chat1")

    def test_set_style_targets(self, db: JarvisDB, contact: Contact) -> None:
        """Test setting style targets for a contact."""
        targets = db.set_style_targets(
            contact_id=contact.id,  # type: ignore[arg-type]
            median_reply_length=15,
            punctuation_rate=0.8,
            emoji_rate=0.3,
            greeting_rate=0.1,
        )

        assert targets.contact_id == contact.id
        assert targets.median_reply_length == 15
        assert targets.punctuation_rate == 0.8
        assert targets.emoji_rate == 0.3
        assert targets.greeting_rate == 0.1

    def test_get_style_targets(self, db: JarvisDB, contact: Contact) -> None:
        """Test retrieving style targets."""
        db.set_style_targets(
            contact_id=contact.id,  # type: ignore[arg-type]
            median_reply_length=20,
            emoji_rate=0.5,
        )

        targets = db.get_style_targets(contact.id)  # type: ignore[arg-type]

        assert targets is not None
        assert targets.median_reply_length == 20
        assert targets.emoji_rate == 0.5

    def test_get_style_targets_returns_none_for_missing(self, db: JarvisDB) -> None:
        """Test that get_style_targets returns None for non-existent contact."""
        result = db.get_style_targets(99999)
        assert result is None

    def test_set_style_targets_updates_existing(self, db: JarvisDB, contact: Contact) -> None:
        """Test that setting style targets updates existing record."""
        db.set_style_targets(contact_id=contact.id, median_reply_length=10)  # type: ignore[arg-type]
        db.set_style_targets(contact_id=contact.id, median_reply_length=25)  # type: ignore[arg-type]

        targets = db.get_style_targets(contact.id)  # type: ignore[arg-type]
        assert targets is not None
        assert targets.median_reply_length == 25


class TestValidatedPairs:
    """Tests for validated pair operations (v6+)."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    @pytest.fixture
    def contact(self, db: JarvisDB) -> Contact:
        """Create a contact for validated pair tests."""
        return db.add_contact(display_name="Test Contact", chat_id="chat1")

    def test_add_validated_pair(self, db: JarvisDB, contact: Contact) -> None:
        """Test adding a validated pair with gate results."""
        now = datetime.now()
        pair = db.add_validated_pair(
            trigger_text="How's the project going?",
            response_text="Making good progress, should be done by Friday",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            quality_score=0.85,
            gate_a_passed=True,
            gate_b_score=0.72,
            gate_c_verdict="accept",
            validity_status="valid",
        )

        assert pair is not None
        assert pair.gate_a_passed is True
        assert pair.gate_b_score == 0.72
        assert pair.gate_c_verdict == "accept"
        assert pair.validity_status == "valid"

    def test_add_validated_pair_with_artifacts(self, db: JarvisDB, contact: Contact) -> None:
        """Test that validated pairs store artifacts separately."""
        now = datetime.now()
        pair = db.add_validated_pair(
            trigger_text="Yes",
            response_text="Great",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            gate_a_passed=False,
            gate_a_reason="Too short",
            context_json='[{"text": "Want to meet?"}]',
            raw_trigger_text="Yes!",
        )

        assert pair is not None

        # Verify artifacts were stored
        artifact = db.get_artifact(pair.id)  # type: ignore[arg-type]
        assert artifact is not None
        assert artifact.gate_a_reason == "Too short"
        assert artifact.raw_trigger_text == "Yes!"

    def test_get_valid_pairs(self, db: JarvisDB, contact: Contact) -> None:
        """Test retrieving only valid pairs."""
        now = datetime.now()

        # Add valid pair
        db.add_validated_pair(
            trigger_text="Valid trigger",
            response_text="Valid response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            validity_status="valid",
        )

        # Add invalid pair
        db.add_validated_pair(
            trigger_text="Invalid trigger",
            response_text="Invalid response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=3,
            response_msg_id=4,
            validity_status="invalid",
        )

        valid_pairs = db.get_valid_pairs()
        assert len(valid_pairs) == 1
        assert valid_pairs[0].trigger_text == "Valid trigger"

    def test_get_gate_stats(self, db: JarvisDB, contact: Contact) -> None:
        """Test retrieving gate statistics."""
        now = datetime.now()

        # Add pairs with various gate results
        db.add_validated_pair(
            trigger_text="T1",
            response_text="R1",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
            gate_a_passed=True,
            gate_b_score=0.75,
            gate_c_verdict="accept",
            validity_status="valid",
        )
        db.add_validated_pair(
            trigger_text="T2",
            response_text="R2",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=3,
            response_msg_id=4,
            gate_a_passed=False,
            validity_status="invalid",
        )

        stats = db.get_gate_stats()

        assert stats["total_gated"] == 2
        assert stats["status_valid"] == 1
        assert stats["status_invalid"] == 1
        assert stats["gate_a_rejected"] == 1


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_db()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_db()

    def test_get_db_returns_same_instance(self) -> None:
        """Test that get_db returns the same instance."""
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2

    def test_reset_db_clears_singleton(self) -> None:
        """Test that reset_db clears the singleton."""
        db1 = get_db()
        reset_db()
        db2 = get_db()
        assert db1 is not db2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> JarvisDB:
        """Create a fresh database for each test."""
        db_path = tmp_path / "jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()
        return db

    def test_empty_database_operations(self, db: JarvisDB) -> None:
        """Test operations on empty database don't crash."""
        assert db.list_contacts() == []
        assert db.get_pairs() == []
        assert db.list_clusters() == []
        assert db.get_training_pairs() == []
        assert db.get_holdout_pairs() == []

    def test_add_contact_without_optional_fields(self, db: JarvisDB) -> None:
        """Test adding contact with only required field."""
        contact = db.add_contact(display_name="Minimal Contact")

        assert contact.id is not None
        assert contact.display_name == "Minimal Contact"
        assert contact.chat_id is None
        assert contact.relationship is None

    def test_add_pair_without_contact(self, db: JarvisDB) -> None:
        """Test adding pair without associated contact."""
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="Orphan trigger",
            response_text="Orphan response",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="unknown_chat",
            contact_id=None,
            trigger_msg_id=1,
            response_msg_id=2,
        )

        assert pair is not None
        assert pair.contact_id is None

    def test_unicode_in_text_fields(self, db: JarvisDB) -> None:
        """Test handling of unicode characters in text fields."""
        contact = db.add_contact(
            display_name="Test",
            style_notes="Casual, likes to use emojis",
        )
        now = datetime.now()
        pair = db.add_pair(
            trigger_text="How about dinner?",
            response_text="Sounds great!",
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )

        assert pair is not None
        retrieved_pairs = db.get_pairs()
        assert len(retrieved_pairs) == 1
        # Note: Emoji may not render in all environments

    def test_very_long_text(self, db: JarvisDB) -> None:
        """Test handling of very long text content."""
        contact = db.add_contact(display_name="Test")
        now = datetime.now()
        long_text = "A" * 10000

        pair = db.add_pair(
            trigger_text=long_text,
            response_text=long_text,
            trigger_timestamp=now,
            response_timestamp=now,
            chat_id="chat1",
            contact_id=contact.id,
            trigger_msg_id=1,
            response_msg_id=2,
        )

        assert pair is not None
        retrieved = db.get_pairs()
        assert len(retrieved[0].trigger_text) == 10000

    def test_special_characters_in_names(self, db: JarvisDB) -> None:
        """Test handling of special characters in names."""
        special_names = [
            "O'Brien",
            "Jean-Pierre",
            "Smith, Jr.",
            "Test (Nickname)",
            "User <test>",
        ]

        for name in special_names:
            contact = db.add_contact(display_name=name)
            assert contact.display_name == name

    def test_connection_rollback_on_error(self, db: JarvisDB) -> None:
        """Test that connection rolls back on error."""
        # Add a contact first
        contact = db.add_contact(display_name="Test")

        # Try to add an invalid entry that should fail
        # (violating unique constraint on chat_id)
        db.add_contact(display_name="First", chat_id="unique_id")

        # This should not affect the database state on failure
        initial_count = len(db.list_contacts())

        try:
            with db.connection() as conn:
                # This will succeed
                conn.execute(
                    "INSERT INTO contacts (display_name, chat_id) VALUES (?, ?)",
                    ("Second", "another_id"),
                )
                # This will fail (duplicate chat_id)
                conn.execute(
                    "INSERT INTO contacts (display_name, chat_id) VALUES (?, ?)",
                    ("Third", "unique_id"),
                )
        except Exception:
            pass

        # Count should be same as before (transaction rolled back)
        assert len(db.list_contacts()) == initial_count
