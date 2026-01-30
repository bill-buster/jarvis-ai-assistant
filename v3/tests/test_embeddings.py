"""RAG/Embedding integration tests for JARVIS v3.

Tests EmbeddingStore operations with mocked embedding model.
Uses temporary SQLite database for isolation.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings.store import (
    EmbeddingStore,
    SimilarMessage,
    get_embedding_store,
    reset_embedding_store,
)


# Test constants
EMBEDDING_DIM = 384  # Same as all-MiniLM-L6-v2


def create_deterministic_embedding(text: str) -> np.ndarray:
    """Create a deterministic embedding from text.

    Uses a simple hash-based approach to generate consistent vectors.
    Similar texts will have similar vectors (based on word overlap).
    """
    # Base embedding from text hash
    np.random.seed(hash(text) % (2**31))
    embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)

    # Add word-based component for similarity
    words = set(text.lower().split())
    for i, word in enumerate(sorted(words)):
        idx = hash(word) % EMBEDDING_DIM
        embedding[idx] += 0.5

    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model that returns deterministic vectors."""
    mock_model = MagicMock()
    mock_model.model_id = "mock-model"
    mock_model.dimension = EMBEDDING_DIM
    mock_model.is_loaded = True

    def embed_single(text: str) -> np.ndarray:
        return create_deterministic_embedding(text)

    def embed_batch(texts: list[str]) -> list[np.ndarray]:
        return [create_deterministic_embedding(t) for t in texts]

    mock_model.embed = embed_single
    mock_model.embed_batch = embed_batch

    return mock_model


@pytest.fixture
def embedding_store(tmp_path, mock_embedding_model):
    """Create an EmbeddingStore with a temporary database."""
    db_path = tmp_path / "test_embeddings.db"

    with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
        store = EmbeddingStore(db_path)
        yield store

    # Cleanup
    reset_embedding_store()


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    now = datetime.now()
    base_ts = int(now.timestamp())

    return [
        {
            "id": 1,
            "text": "Hey, are you free for dinner tonight?",
            "chat_id": "chat-alice",
            "sender": "+15551234567",
            "sender_name": "Alice",
            "timestamp": base_ts - 3600,
            "is_from_me": False,
        },
        {
            "id": 2,
            "text": "Yeah, sounds good! What time?",
            "chat_id": "chat-alice",
            "sender": "me",
            "sender_name": None,
            "timestamp": base_ts - 3500,
            "is_from_me": True,
        },
        {
            "id": 3,
            "text": "How about 7pm at the Italian place?",
            "chat_id": "chat-alice",
            "sender": "+15551234567",
            "sender_name": "Alice",
            "timestamp": base_ts - 3400,
            "is_from_me": False,
        },
        {
            "id": 4,
            "text": "Perfect, see you there!",
            "chat_id": "chat-alice",
            "sender": "me",
            "sender_name": None,
            "timestamp": base_ts - 3300,
            "is_from_me": True,
        },
        {
            "id": 5,
            "text": "Want to grab coffee tomorrow?",
            "chat_id": "chat-bob",
            "sender": "+15559876543",
            "sender_name": "Bob",
            "timestamp": base_ts - 2000,
            "is_from_me": False,
        },
        {
            "id": 6,
            "text": "Sure, morning or afternoon?",
            "chat_id": "chat-bob",
            "sender": "me",
            "sender_name": None,
            "timestamp": base_ts - 1900,
            "is_from_me": True,
        },
    ]


class TestEmbeddingStoreInitialization:
    """Test EmbeddingStore initialization."""

    def test_create_store_creates_database(self, tmp_path):
        """Test that creating a store creates the database file."""
        db_path = tmp_path / "new_embeddings.db"
        assert not db_path.exists()

        store = EmbeddingStore(db_path)

        assert db_path.exists()
        assert store.db_path == db_path

    def test_create_store_creates_parent_directories(self, tmp_path):
        """Test that creating a store creates parent directories."""
        db_path = tmp_path / "nested" / "path" / "embeddings.db"
        assert not db_path.parent.exists()

        store = EmbeddingStore(db_path)

        assert db_path.parent.exists()
        assert db_path.exists()

    def test_store_creates_required_tables(self, embedding_store):
        """Test that the store creates the required database tables."""
        with embedding_store._get_connection() as conn:
            # Check main table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
            )
            assert cursor.fetchone() is not None

            # Check FTS table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            )
            assert cursor.fetchone() is not None


class TestIndexMessages:
    """Test message indexing operations."""

    def test_index_messages_basic(self, embedding_store, sample_messages, mock_embedding_model):
        """Test basic message indexing."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            stats = embedding_store.index_messages(sample_messages)

        assert stats["indexed"] == 6
        assert stats["skipped"] == 0
        assert stats["duplicates"] == 0

    def test_index_messages_skips_empty_text(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that messages with empty text are skipped."""
        messages_with_empty = sample_messages + [
            {
                "id": 100,
                "text": "",  # Empty
                "chat_id": "chat-test",
                "sender": "test",
                "timestamp": 1000,
                "is_from_me": False,
            },
            {
                "id": 101,
                "text": "ab",  # Too short (min_text_length=3)
                "chat_id": "chat-test",
                "sender": "test",
                "timestamp": 1001,
                "is_from_me": False,
            },
        ]

        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            stats = embedding_store.index_messages(messages_with_empty)

        assert stats["indexed"] == 6
        assert stats["skipped"] == 2

    def test_index_messages_handles_duplicates(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that duplicate messages are not re-indexed."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            # Index once
            stats1 = embedding_store.index_messages(sample_messages)
            assert stats1["indexed"] == 6

            # Index again - should detect duplicates
            stats2 = embedding_store.index_messages(sample_messages)
            assert stats2["indexed"] == 0
            assert stats2["duplicates"] == 6

    def test_index_messages_with_progress_callback(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that progress callback is called during indexing."""
        progress_calls = []

        def progress_callback(indexed: int, total: int):
            progress_calls.append((indexed, total))

        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages, progress_callback=progress_callback)

        assert len(progress_calls) > 0
        # Last call should have indexed all messages
        assert progress_calls[-1][0] == 6


class TestFindSimilar:
    """Test similarity search operations."""

    def test_find_similar_basic(self, embedding_store, sample_messages, mock_embedding_model):
        """Test basic similarity search."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Search for similar messages
            results = embedding_store.find_similar(
                query="dinner plans tonight",
                chat_id="chat-alice",
                limit=5,
                min_similarity=0.0,  # Low threshold to get results
            )

            assert len(results) > 0
            assert all(isinstance(r, SimilarMessage) for r in results)
            assert all(r.chat_id == "chat-alice" for r in results)

    def test_find_similar_respects_limit(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that find_similar respects the limit parameter."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            results = embedding_store.find_similar(
                query="dinner",
                chat_id="chat-alice",
                limit=2,
                min_similarity=0.0,
            )

        assert len(results) <= 2

    def test_find_similar_filters_by_chat_id(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that find_similar filters by chat_id."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Search in specific chat
            alice_results = embedding_store.find_similar(
                query="meeting",
                chat_id="chat-alice",
                min_similarity=0.0,
            )

            bob_results = embedding_store.find_similar(
                query="meeting",
                chat_id="chat-bob",
                min_similarity=0.0,
            )

        # All results should be from the specified chat
        assert all(r.chat_id == "chat-alice" for r in alice_results)
        assert all(r.chat_id == "chat-bob" for r in bob_results)

    def test_find_similar_only_from_me(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test filtering by only_from_me parameter."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Get only my messages
            my_results = embedding_store.find_similar(
                query="sounds good",
                chat_id="chat-alice",
                only_from_me=True,
                min_similarity=0.0,
            )

            # Get only their messages
            their_results = embedding_store.find_similar(
                query="sounds good",
                chat_id="chat-alice",
                only_from_me=False,
                min_similarity=0.0,
            )

        assert all(r.is_from_me for r in my_results)
        assert all(not r.is_from_me for r in their_results)

    def test_find_similar_respects_min_similarity(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that find_similar respects min_similarity threshold."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Very high threshold should return few/no results
            high_threshold_results = embedding_store.find_similar(
                query="completely unrelated query xyz abc",
                chat_id="chat-alice",
                min_similarity=0.99,
            )

            # Low threshold should return more results
            low_threshold_results = embedding_store.find_similar(
                query="completely unrelated query xyz abc",
                chat_id="chat-alice",
                min_similarity=0.0,
            )

        assert len(high_threshold_results) <= len(low_threshold_results)


class TestEmptyStore:
    """Test edge cases with empty store."""

    def test_find_similar_empty_store(self, embedding_store, mock_embedding_model):
        """Test find_similar on an empty store."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            results = embedding_store.find_similar(
                query="test query",
                chat_id="nonexistent-chat",
            )

        assert results == []

    def test_get_stats_empty_store(self, embedding_store):
        """Test get_stats on an empty store."""
        stats = embedding_store.get_stats()

        assert stats["total_messages"] == 0
        assert stats["unique_conversations"] == 0

    def test_get_style_profile_empty_store(self, embedding_store):
        """Test get_style_profile on an empty store."""
        profile = embedding_store.get_style_profile("nonexistent-chat")

        assert profile.contact_id == "nonexistent-chat"
        assert profile.total_messages == 0
        assert profile.sent_count == 0
        assert profile.received_count == 0


class TestCrossConversationSearch:
    """Test cross-conversation search functionality."""

    def test_find_your_past_replies_cross_conversation(
        self, tmp_path, sample_messages, mock_embedding_model
    ):
        """Test cross-conversation reply search."""
        db_path = tmp_path / "cross_conv_test.db"

        # Patch at the module level to catch all imports
        with patch.dict("sys.modules", {}):
            with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
                store = EmbeddingStore(db_path)
                store.index_messages(sample_messages)

                # Mock the embedding model for the search query
                # Since it imports get_embedding_model inside the method,
                # we need to ensure the mock returns a model that can embed
                mock_embedding_model.embed = lambda text: create_deterministic_embedding(text)

                # Monkeypatch the model module's function
                import core.embeddings.model as model_module
                original_get = model_module.get_embedding_model
                model_module.get_embedding_model = lambda *args, **kwargs: mock_embedding_model

                try:
                    results = store.find_your_past_replies_cross_conversation(
                        incoming_message="Want to get dinner?",
                        limit=5,
                        min_similarity=0.0,
                    )

                    # Results should be (their_text, your_reply, score, chat_id) tuples
                    for result in results:
                        assert len(result) == 4
                        their_text, your_reply, score, chat_id = result
                        assert isinstance(their_text, str)
                        assert isinstance(your_reply, str)
                        assert isinstance(score, float)
                        assert isinstance(chat_id, str)
                finally:
                    model_module.get_embedding_model = original_get

    def test_find_your_past_replies_cross_conversation_with_filter(
        self, tmp_path, sample_messages, mock_embedding_model
    ):
        """Test cross-conversation search with chat_id filter."""
        db_path = tmp_path / "cross_conv_filter_test.db"

        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            store = EmbeddingStore(db_path)
            store.index_messages(sample_messages)

            # Monkeypatch the model module
            import core.embeddings.model as model_module
            original_get = model_module.get_embedding_model
            model_module.get_embedding_model = lambda *args, **kwargs: mock_embedding_model

            try:
                results = store.find_your_past_replies_cross_conversation(
                    incoming_message="Want to get dinner?",
                    target_chat_ids=["chat-alice"],
                    limit=5,
                    min_similarity=0.0,
                )

                # All results should be from the filtered chat
                for result in results:
                    assert result[3] == "chat-alice"
            finally:
                model_module.get_embedding_model = original_get


class TestStyleProfile:
    """Test style profile extraction."""

    def test_get_style_profile_basic(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test basic style profile extraction."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            profile = embedding_store.get_style_profile("chat-alice")

        assert profile.contact_id == "chat-alice"
        assert profile.total_messages == 4
        assert profile.sent_count == 2
        assert profile.received_count == 2

    def test_get_style_profile_analyzes_patterns(
        self, embedding_store, mock_embedding_model
    ):
        """Test that style profile analyzes message patterns."""
        # Create messages with specific patterns
        now = datetime.now()
        base_ts = int(now.timestamp())

        messages = [
            {
                "id": i,
                "text": text,
                "chat_id": "chat-patterns",
                "sender": "me" if i % 2 == 0 else "+1234",
                "sender_name": None if i % 2 == 0 else "Test",
                "timestamp": base_ts - (100 - i) * 60,
                "is_from_me": i % 2 == 0,
            }
            for i, text in enumerate([
                "Hey there!",
                "hi how are you",
                "I'm good thanks!",
                "want to hang out?",
                "sounds fun",
                "cool see you later",
            ])
        ]

        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(messages)
            profile = embedding_store.get_style_profile("chat-patterns")

        assert profile.total_messages == 6
        assert profile.sent_count == 3
        assert profile.received_count == 3


class TestUserResponsePatterns:
    """Test user response pattern extraction."""

    def test_get_user_response_patterns(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test extraction of user response patterns."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            patterns = embedding_store.get_user_response_patterns(chat_id="chat-alice")

        # Returns dict mapping intent -> list of replies
        assert isinstance(patterns, dict)

    def test_get_user_response_patterns_empty_chat(self, embedding_store):
        """Test response patterns for empty/nonexistent chat."""
        patterns = embedding_store.get_user_response_patterns(chat_id="nonexistent")
        assert patterns == {}


class TestClearAndStats:
    """Test clear and statistics operations."""

    def test_clear_removes_all_embeddings(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that clear() removes all embeddings."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Verify messages were indexed
            stats_before = embedding_store.get_stats()
            assert stats_before["total_messages"] == 6

            # Clear
            embedding_store.clear()

            # Verify cleared
            stats_after = embedding_store.get_stats()
            assert stats_after["total_messages"] == 0

    def test_get_stats_returns_correct_counts(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test that get_stats returns accurate counts."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            stats = embedding_store.get_stats()

        assert stats["total_messages"] == 6
        assert stats["unique_conversations"] == 2  # chat-alice and chat-bob
        assert "db_path" in stats


class TestBM25Search:
    """Test BM25 full-text search."""

    def test_search_bm25_basic(self, embedding_store, sample_messages, mock_embedding_model):
        """Test basic BM25 search."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            results = embedding_store.search_bm25(
                query="dinner",
                limit=10,
            )

        # Results are (message_id, bm25_score) tuples
        assert isinstance(results, list)
        for result in results:
            assert len(result) == 2
            msg_id, score = result
            assert isinstance(msg_id, int)
            assert isinstance(score, float)

    def test_search_bm25_filters_by_chat(
        self, embedding_store, sample_messages, mock_embedding_model
    ):
        """Test BM25 search with chat_id filter."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            # Search in specific chat
            results = embedding_store.search_bm25(
                query="coffee",
                chat_id="chat-bob",
                limit=10,
            )

        # Should find the coffee message in Bob's chat
        assert len(results) > 0


class TestHybridSearch:
    """Test hybrid vector + BM25 search."""

    def test_find_similar_hybrid(self, embedding_store, sample_messages, mock_embedding_model):
        """Test hybrid search combining vector and BM25."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            results = embedding_store.find_similar_hybrid(
                query="dinner tonight",
                limit=5,
                min_similarity=0.0,
            )

        assert len(results) > 0
        assert all(isinstance(r, SimilarMessage) for r in results)


class TestChatEmbeddings:
    """Test retrieval of chat embeddings."""

    def test_get_chat_embeddings(self, embedding_store, sample_messages, mock_embedding_model):
        """Test getting all embeddings for a chat."""
        with patch("core.embeddings.store.get_embedding_model", return_value=mock_embedding_model):
            embedding_store.index_messages(sample_messages)

            embeddings, messages = embedding_store.get_chat_embeddings("chat-alice")

        assert len(embeddings) == 4
        assert len(messages) == 4
        assert embeddings.shape == (4, EMBEDDING_DIM)
        assert all(isinstance(m, SimilarMessage) for m in messages)

    def test_get_chat_embeddings_empty_chat(self, embedding_store):
        """Test getting embeddings for empty/nonexistent chat."""
        embeddings, messages = embedding_store.get_chat_embeddings("nonexistent")

        assert len(embeddings) == 0
        assert len(messages) == 0


class TestTimestampHandling:
    """Test proper handling of different timestamp formats."""

    def test_index_messages_with_datetime_timestamp(self, tmp_path):
        """Test indexing messages with datetime objects as timestamps.

        This test verifies that datetime objects are properly converted
        to Unix timestamps during indexing and stored correctly in the database.
        """
        db_path = tmp_path / "datetime_test.db"
        now = datetime.now()

        messages = [
            {
                "id": 9999,  # Unique ID to avoid conflicts
                "text": "Test message with datetime timestamp verification",
                "chat_id": "chat-datetime-test",
                "sender": "test",
                "sender_name": "Test",
                "timestamp": now,  # datetime object instead of int
                "is_from_me": False,
            }
        ]

        # Create mock embedding model locally
        mock_model = MagicMock()
        mock_model.embed = lambda text: create_deterministic_embedding(text)
        mock_model.embed_batch = lambda texts: [create_deterministic_embedding(t) for t in texts]

        with patch("core.embeddings.store.get_embedding_model", return_value=mock_model):
            store = EmbeddingStore(db_path)
            stats = store.index_messages(messages)

            assert stats["indexed"] == 1

            # Verify the message was stored with correct timestamp by querying DB directly
            with store._get_connection() as conn:
                rows = conn.execute(
                    "SELECT timestamp, text_preview FROM message_embeddings WHERE chat_id = ?",
                    ("chat-datetime-test",)
                ).fetchall()
                assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

                # Verify timestamp is stored correctly as integer Unix timestamp
                stored_ts = rows[0]["timestamp"]
                assert abs(stored_ts - int(now.timestamp())) < 2, "Timestamp should match"

                # Verify text was stored
                assert "datetime timestamp" in rows[0]["text_preview"]

            # Test that timestamp is properly converted back to datetime when retrieved
            # via get_chat_embeddings (which doesn't need embedding model)
            embeddings, messages_out = store.get_chat_embeddings("chat-datetime-test")
            assert len(messages_out) == 1
            assert isinstance(messages_out[0].timestamp, datetime)
            assert abs(messages_out[0].timestamp.timestamp() - now.timestamp()) < 2
