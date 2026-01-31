"""Unit tests for embedding-based conversation RAG.

Tests cover the embedding storage, semantic search, relationship profiling,
and RAG prompt building functionality.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from contracts.imessage import Message
from jarvis.embeddings import (
    ConversationContext,
    EmbeddingError,
    EmbeddingStore,
    EmbeddingStoreError,
    RelationshipProfile,
    SimilarMessage,
    find_similar_messages,
    find_similar_situations,
    get_embedding_store,
    get_relationship_profile,
    reset_embedding_store,
)
from tests.conftest import requires_sentence_transformers

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return tmp_path / "test_embeddings.db"


@pytest.fixture
def mock_embedding():
    """Create a mock normalized embedding vector."""
    embedding = np.random.randn(384).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        id=1,
        chat_id="chat123",
        sender="+15551234567",
        sender_name="John Doe",
        text="Hey, want to grab dinner tonight?",
        date=datetime(2024, 1, 15, 18, 30),
        is_from_me=False,
    )


@pytest.fixture
def sample_messages():
    """Create a list of sample messages for testing."""
    return [
        Message(
            id=1,
            chat_id="chat123",
            sender="+15551234567",
            sender_name="John",
            text="Hey, want to grab dinner tonight?",
            date=datetime(2024, 1, 15, 18, 30),
            is_from_me=False,
        ),
        Message(
            id=2,
            chat_id="chat123",
            sender="me",
            sender_name="Me",
            text="Sure, sounds great! What time?",
            date=datetime(2024, 1, 15, 18, 31),
            is_from_me=True,
        ),
        Message(
            id=3,
            chat_id="chat123",
            sender="+15551234567",
            sender_name="John",
            text="How about 7pm at the Italian place?",
            date=datetime(2024, 1, 15, 18, 32),
            is_from_me=False,
        ),
        Message(
            id=4,
            chat_id="chat123",
            sender="me",
            sender_name="Me",
            text="Perfect, see you there!",
            date=datetime(2024, 1, 15, 18, 33),
            is_from_me=True,
        ),
    ]


# =============================================================================
# Data Class Tests
# =============================================================================


class TestSimilarMessage:
    """Tests for SimilarMessage data class."""

    def test_creation(self):
        """Test creating a SimilarMessage instance."""
        msg = SimilarMessage(
            message_id=1,
            chat_id="chat123",
            text="Hello",
            sender="+15551234567",
            sender_name="John",
            timestamp=datetime(2024, 1, 15, 10, 0),
            is_from_me=False,
            similarity=0.85,
        )

        assert msg.message_id == 1
        assert msg.chat_id == "chat123"
        assert msg.text == "Hello"
        assert msg.similarity == 0.85

    def test_without_optional_fields(self):
        """Test creating SimilarMessage without optional fields."""
        msg = SimilarMessage(
            message_id=1,
            chat_id="chat123",
            text="Hello",
            sender=None,
            sender_name=None,
            timestamp=datetime(2024, 1, 15, 10, 0),
            is_from_me=True,
            similarity=0.5,
        )

        assert msg.sender is None
        assert msg.sender_name is None


class TestConversationContext:
    """Tests for ConversationContext data class."""

    def test_creation(self):
        """Test creating a ConversationContext instance."""
        messages = [
            SimilarMessage(
                message_id=1,
                chat_id="chat123",
                text="Test",
                sender=None,
                sender_name=None,
                timestamp=datetime.now(),
                is_from_me=False,
                similarity=0.8,
            )
        ]

        ctx = ConversationContext(
            messages=messages,
            topic="planning",
            avg_similarity=0.8,
        )

        assert len(ctx.messages) == 1
        assert ctx.topic == "planning"
        assert ctx.avg_similarity == 0.8


class TestRelationshipProfile:
    """Tests for RelationshipProfile data class."""

    def test_creation(self):
        """Test creating a RelationshipProfile instance."""
        profile = RelationshipProfile(
            contact_id="chat123",
            display_name="John Doe",
            total_messages=100,
            sent_count=50,
            received_count=50,
            common_topics=["dinner", "work"],
            typical_tone="casual",
            avg_message_length=45.5,
        )

        assert profile.contact_id == "chat123"
        assert profile.display_name == "John Doe"
        assert profile.total_messages == 100
        assert profile.typical_tone == "casual"

    def test_default_values(self):
        """Test RelationshipProfile default values."""
        profile = RelationshipProfile(contact_id="chat123")

        assert profile.display_name is None
        assert profile.total_messages == 0
        assert profile.sent_count == 0
        assert profile.received_count == 0
        assert profile.common_topics == []
        assert profile.typical_tone == "casual"
        assert profile.avg_message_length == 0.0
        assert profile.response_patterns == {}
        assert profile.last_interaction is None


# =============================================================================
# EmbeddingStore Tests (with mocked embeddings)
# =============================================================================


class TestEmbeddingStoreSchema:
    """Tests for EmbeddingStore database schema."""

    def test_creates_database_file(self, temp_db_path):
        """Test that store creates the database file."""
        EmbeddingStore(temp_db_path)
        assert temp_db_path.exists()

    def test_creates_parent_directory(self, tmp_path):
        """Test that store creates parent directories."""
        nested_path = tmp_path / "subdir" / "nested" / "test.db"
        EmbeddingStore(nested_path)
        assert nested_path.parent.exists()

    def test_initializes_tables(self, temp_db_path):
        """Test that store initializes required tables."""
        EmbeddingStore(temp_db_path)

        import sqlite3

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "message_embeddings" in tables
        assert "relationship_profiles" in tables
        assert "index_stats" in tables


class TestEmbeddingStoreIndexing:
    """Tests for message indexing (with mocked embedding computation)."""

    @patch("jarvis.embeddings._compute_embedding")
    def test_index_message_basic(self, mock_embed, temp_db_path, sample_message, mock_embedding):
        """Test indexing a single message."""
        mock_embed.return_value = mock_embedding

        store = EmbeddingStore(temp_db_path)
        result = store.index_message(sample_message)

        assert result is True
        mock_embed.assert_called_once_with(sample_message.text)

    @patch("jarvis.embeddings._compute_embedding")
    def test_index_message_skips_short_text(self, mock_embed, temp_db_path):
        """Test that short messages are skipped."""
        short_msg = Message(
            id=1,
            chat_id="chat123",
            sender="+15551234567",
            sender_name="John",
            text="ok",
            date=datetime.now(),
            is_from_me=False,
        )

        store = EmbeddingStore(temp_db_path)
        result = store.index_message(short_msg)

        assert result is False
        mock_embed.assert_not_called()

    @patch("jarvis.embeddings._compute_embedding")
    def test_index_message_skips_empty_text(self, mock_embed, temp_db_path):
        """Test that empty messages are skipped."""
        empty_msg = Message(
            id=1,
            chat_id="chat123",
            sender="+15551234567",
            sender_name="John",
            text="",
            date=datetime.now(),
            is_from_me=False,
        )

        store = EmbeddingStore(temp_db_path)
        result = store.index_message(empty_msg)

        assert result is False
        mock_embed.assert_not_called()

    @patch("jarvis.embeddings._compute_embedding")
    def test_index_message_skips_duplicates(
        self, mock_embed, temp_db_path, sample_message, mock_embedding
    ):
        """Test that duplicate messages are skipped."""
        mock_embed.return_value = mock_embedding

        store = EmbeddingStore(temp_db_path)
        result1 = store.index_message(sample_message)
        result2 = store.index_message(sample_message)

        assert result1 is True
        assert result2 is False
        # Should only compute embedding once
        assert mock_embed.call_count == 1

    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_index_messages_batch(
        self, mock_embed_batch, temp_db_path, sample_messages, mock_embedding
    ):
        """Test batch indexing of messages."""
        # Return embeddings for each message
        mock_embed_batch.return_value = np.array([mock_embedding for _ in sample_messages])

        store = EmbeddingStore(temp_db_path)
        stats = store.index_messages(sample_messages)

        assert stats["indexed"] == len(sample_messages)
        assert stats["skipped"] == 0
        assert stats["duplicates"] == 0

    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_index_messages_counts_skipped(self, mock_embed_batch, temp_db_path, mock_embedding):
        """Test that index_messages counts skipped messages."""
        messages = [
            Message(
                id=1,
                chat_id="chat123",
                sender="s",
                sender_name="S",
                text="ok",  # Too short
                date=datetime.now(),
                is_from_me=False,
            ),
            Message(
                id=2,
                chat_id="chat123",
                sender="s",
                sender_name="S",
                text="This is a valid message",
                date=datetime.now(),
                is_from_me=True,
            ),
        ]

        mock_embed_batch.return_value = np.array([mock_embedding])

        store = EmbeddingStore(temp_db_path)
        stats = store.index_messages(messages)

        assert stats["skipped"] == 1
        assert stats["indexed"] == 1


class TestEmbeddingStoreSimilarSearch:
    """Tests for similarity search (with mocked embedding computation)."""

    @patch("jarvis.embeddings._compute_embedding")
    def test_find_similar_empty_store(self, mock_embed, temp_db_path, mock_embedding):
        """Test search on empty store returns empty list."""
        mock_embed.return_value = mock_embedding

        store = EmbeddingStore(temp_db_path)
        results = store.find_similar("test query")

        assert results == []

    @patch("jarvis.embeddings._compute_embedding")
    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_find_similar_returns_matches(
        self, mock_embed_batch, mock_embed, temp_db_path, sample_messages
    ):
        """Test that find_similar returns matching messages."""
        # Create embeddings that are similar
        base_embedding = np.random.randn(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # Batch embeddings for indexing
        mock_embed_batch.return_value = np.array([base_embedding for _ in sample_messages])
        # Query embedding (same as indexed)
        mock_embed.return_value = base_embedding

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)
        results = store.find_similar("dinner tonight")

        assert len(results) > 0
        assert all(isinstance(r, SimilarMessage) for r in results)
        # Results should be sorted by similarity
        assert all(
            results[i].similarity >= results[i + 1].similarity for i in range(len(results) - 1)
        )

    @patch("jarvis.embeddings._compute_embedding")
    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_find_similar_respects_min_similarity(
        self, mock_embed_batch, mock_embed, temp_db_path, sample_messages
    ):
        """Test that min_similarity filters results."""
        # Create very different embeddings
        mock_embed_batch.return_value = np.array(
            [np.random.randn(384).astype(np.float32) for _ in sample_messages]
        )
        # Query with orthogonal embedding
        query_embedding = np.zeros(384, dtype=np.float32)
        query_embedding[0] = 1.0
        mock_embed.return_value = query_embedding

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)

        # With high min_similarity, may return fewer results
        results = store.find_similar("unrelated query", min_similarity=0.9)

        # All returned results should meet the threshold
        assert all(r.similarity >= 0.9 for r in results)

    @patch("jarvis.embeddings._compute_embedding")
    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_find_similar_filters_by_chat_id(
        self, mock_embed_batch, mock_embed, temp_db_path, mock_embedding
    ):
        """Test that chat_id filter works."""
        messages = [
            Message(
                id=1,
                chat_id="chat1",
                sender="s",
                sender_name="S",
                text="Message in chat 1",
                date=datetime.now(),
                is_from_me=False,
            ),
            Message(
                id=2,
                chat_id="chat2",
                sender="s",
                sender_name="S",
                text="Message in chat 2",
                date=datetime.now(),
                is_from_me=False,
            ),
        ]

        mock_embed_batch.return_value = np.array([mock_embedding, mock_embedding])
        mock_embed.return_value = mock_embedding

        store = EmbeddingStore(temp_db_path)
        store.index_messages(messages)
        results = store.find_similar("test", chat_id="chat1")

        assert all(r.chat_id == "chat1" for r in results)

    @patch("jarvis.embeddings._compute_embedding")
    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_find_similar_respects_limit(
        self, mock_embed_batch, mock_embed, temp_db_path, mock_embedding
    ):
        """Test that limit parameter works."""
        messages = [
            Message(
                id=i,
                chat_id="chat123",
                sender="s",
                sender_name="S",
                text=f"Message number {i}",
                date=datetime.now(),
                is_from_me=False,
            )
            for i in range(10)
        ]

        mock_embed_batch.return_value = np.array([mock_embedding for _ in messages])
        mock_embed.return_value = mock_embedding

        store = EmbeddingStore(temp_db_path)
        store.index_messages(messages)
        results = store.find_similar("test", limit=3)

        assert len(results) <= 3


class TestEmbeddingStoreRelationshipProfile:
    """Tests for relationship profile computation."""

    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_get_relationship_profile_basic(
        self, mock_embed_batch, temp_db_path, sample_messages, mock_embedding
    ):
        """Test basic relationship profile computation."""
        mock_embed_batch.return_value = np.array([mock_embedding for _ in sample_messages])

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)
        profile = store.get_relationship_profile("chat123")

        assert profile.contact_id == "chat123"
        assert profile.total_messages == len(sample_messages)
        assert profile.sent_count + profile.received_count == profile.total_messages

    def test_get_relationship_profile_empty(self, temp_db_path):
        """Test relationship profile for non-existent contact."""
        store = EmbeddingStore(temp_db_path)
        profile = store.get_relationship_profile("nonexistent")

        assert profile.contact_id == "nonexistent"
        assert profile.total_messages == 0

    @patch("jarvis.embeddings._compute_embeddings_batch")
    @patch("jarvis.prompts.detect_tone")
    def test_get_relationship_profile_detects_tone(
        self, mock_detect_tone, mock_embed_batch, temp_db_path, sample_messages, mock_embedding
    ):
        """Test that relationship profile detects tone."""
        mock_embed_batch.return_value = np.array([mock_embedding for _ in sample_messages])
        mock_detect_tone.return_value = "casual"

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)
        profile = store.get_relationship_profile("chat123")

        assert profile.typical_tone == "casual"


class TestEmbeddingStoreStats:
    """Tests for embedding store statistics."""

    def test_get_stats_empty_store(self, temp_db_path):
        """Test stats on empty store."""
        store = EmbeddingStore(temp_db_path)
        stats = store.get_stats()

        assert stats["total_embeddings"] == 0
        assert stats["unique_chats"] == 0
        assert stats["oldest_message"] is None
        assert stats["newest_message"] is None

    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_get_stats_with_data(
        self, mock_embed_batch, temp_db_path, sample_messages, mock_embedding
    ):
        """Test stats with indexed data."""
        mock_embed_batch.return_value = np.array([mock_embedding for _ in sample_messages])

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)
        stats = store.get_stats()

        assert stats["total_embeddings"] == len(sample_messages)
        assert stats["unique_chats"] == 1
        assert stats["oldest_message"] is not None
        assert stats["newest_message"] is not None
        assert stats["db_path"] == str(temp_db_path)
        assert stats["db_size_bytes"] > 0

    @patch("jarvis.embeddings._compute_embeddings_batch")
    def test_clear(self, mock_embed_batch, temp_db_path, sample_messages, mock_embedding):
        """Test clearing the store."""
        mock_embed_batch.return_value = np.array([mock_embedding for _ in sample_messages])

        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)
        store.clear()
        stats = store.get_stats()

        assert stats["total_embeddings"] == 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestEmbeddingStoreSingleton:
    """Tests for singleton access functions."""

    def test_get_embedding_store_returns_store(self, tmp_path):
        """Test that get_embedding_store returns an EmbeddingStore."""
        reset_embedding_store()
        db_path = tmp_path / "singleton_test.db"
        store = get_embedding_store(db_path)

        assert isinstance(store, EmbeddingStore)

    def test_get_embedding_store_custom_path(self, tmp_path):
        """Test get_embedding_store with custom path."""
        reset_embedding_store()
        custom_path = tmp_path / "custom.db"
        get_embedding_store(custom_path)

        assert custom_path.exists()

    def test_reset_embedding_store(self, tmp_path):
        """Test that reset_embedding_store allows new instance."""
        reset_embedding_store()
        db_path = tmp_path / "reset_test.db"

        _store1 = get_embedding_store(db_path)
        reset_embedding_store()
        store2 = get_embedding_store(db_path)

        # After reset, should be different instance
        # (but same path still works)
        assert store2 is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("jarvis.embeddings.get_embedding_store")
    def test_find_similar_messages_delegates(self, mock_get_store, mock_embedding):
        """Test find_similar_messages delegates to store."""
        mock_store = MagicMock()
        mock_store.find_similar.return_value = []
        mock_get_store.return_value = mock_store

        find_similar_messages("test query", contact_id="chat123", limit=5)

        mock_store.find_similar.assert_called_once_with(
            query="test query",
            chat_id="chat123",
            limit=5,
            min_similarity=0.3,
        )

    @patch("jarvis.embeddings.get_embedding_store")
    def test_find_similar_situations_delegates(self, mock_get_store):
        """Test find_similar_situations delegates to store."""
        mock_store = MagicMock()
        mock_store.find_similar_situations.return_value = []
        mock_get_store.return_value = mock_store

        find_similar_situations("current context", contact_id="chat123", limit=3)

        mock_store.find_similar_situations.assert_called_once()

    @patch("jarvis.embeddings.get_embedding_store")
    def test_get_relationship_profile_delegates(self, mock_get_store):
        """Test get_relationship_profile delegates to store."""
        mock_store = MagicMock()
        mock_store.get_relationship_profile.return_value = RelationshipProfile(contact_id="chat123")
        mock_get_store.return_value = mock_store

        result = get_relationship_profile("chat123")

        mock_store.get_relationship_profile.assert_called_once_with("chat123")
        assert result.contact_id == "chat123"


# =============================================================================
# RAG Prompt Builder Tests
# =============================================================================


class TestRAGPromptBuilder:
    """Tests for RAG-enhanced prompt building."""

    def test_build_rag_reply_prompt_basic(self):
        """Test basic RAG prompt building."""
        from jarvis.prompts import build_rag_reply_prompt

        prompt = build_rag_reply_prompt(
            context="[10:00] John: Hey, want dinner?",
            last_message="Hey, want dinner?",
            contact_name="John",
        )

        assert "John" in prompt
        assert "Hey, want dinner?" in prompt
        assert "### Your reply" in prompt  # Template ends with "### Your reply (keep it brief):"

    def test_build_rag_reply_prompt_with_exchanges(self):
        """Test RAG prompt with similar exchanges."""
        from jarvis.prompts import build_rag_reply_prompt

        exchanges = [
            ("Want to grab lunch?", "Sure, what time?"),
            ("Dinner tonight?", "Sounds great!"),
        ]

        prompt = build_rag_reply_prompt(
            context="Test context",
            last_message="Want to get food?",
            contact_name="John",
            similar_exchanges=exchanges,
        )

        assert "Similar Past Exchanges" in prompt
        assert "grab lunch" in prompt
        assert "Dinner tonight" in prompt

    def test_build_rag_reply_prompt_with_profile(self):
        """Test RAG prompt with relationship profile."""
        from jarvis.prompts import build_rag_reply_prompt

        profile = {
            "tone": "professional",
            "avg_message_length": 150.0,
            "response_patterns": {"avg_response_time_seconds": 100},
        }

        prompt = build_rag_reply_prompt(
            context="Test context",
            last_message="Meeting tomorrow?",
            contact_name="Boss",
            relationship_profile=profile,
        )

        assert "professional" in prompt.lower()

    def test_build_rag_reply_prompt_with_instruction(self):
        """Test RAG prompt with custom instruction."""
        from jarvis.prompts import build_rag_reply_prompt

        prompt = build_rag_reply_prompt(
            context="Test context",
            last_message="Want to meet?",
            contact_name="John",
            instruction="Be brief and direct",
        )

        assert "Be brief and direct" in prompt


# =============================================================================
# Integration Tests (require sentence_transformers)
# =============================================================================


@requires_sentence_transformers
class TestEmbeddingStoreIntegration:
    """Integration tests that use actual embedding computation."""

    def test_index_and_search_real_embeddings(self, temp_db_path, sample_messages):
        """Test full indexing and search with real embeddings."""
        store = EmbeddingStore(temp_db_path)
        stats = store.index_messages(sample_messages)

        assert stats["indexed"] == len(sample_messages)

        # Search for semantically similar content
        results = store.find_similar("What about food tonight?", min_similarity=0.3)

        # Should find dinner-related messages
        assert len(results) > 0
        # Top result should be related to dinner
        assert any("dinner" in r.text.lower() for r in results[:3])

    def test_search_filters_correctly(self, temp_db_path):
        """Test that semantic search filters work correctly."""
        messages = [
            Message(
                id=1,
                chat_id="work",
                sender="boss",
                sender_name="Boss",
                text="Please review the quarterly report by EOD",
                date=datetime.now(),
                is_from_me=False,
            ),
            Message(
                id=2,
                chat_id="friends",
                sender="friend",
                sender_name="Friend",
                text="Let's go to the movies this weekend",
                date=datetime.now(),
                is_from_me=False,
            ),
        ]

        store = EmbeddingStore(temp_db_path)
        store.index_messages(messages)

        # Search only in work chat
        results = store.find_similar("document review", chat_id="work")
        assert all(r.chat_id == "work" for r in results)

    def test_relationship_profile_real_data(self, temp_db_path, sample_messages):
        """Test relationship profile with real data."""
        store = EmbeddingStore(temp_db_path)
        store.index_messages(sample_messages)

        profile = store.get_relationship_profile("chat123")

        assert profile.total_messages == len(sample_messages)
        assert profile.typical_tone in ("casual", "professional", "mixed")
        assert profile.avg_message_length > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_embedding_error_creation(self):
        """Test EmbeddingError creation."""
        error = EmbeddingError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_embedding_store_error_creation(self):
        """Test EmbeddingStoreError creation."""
        error = EmbeddingStoreError("Store error")

        assert str(error) == "Store error"
        assert isinstance(error, EmbeddingError)
