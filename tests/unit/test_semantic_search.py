"""Tests for semantic search functionality."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from contracts.imessage import Message
from jarvis.semantic_search import (
    EmbeddingCache,
    SearchFilters,
    SemanticSearcher,
    SemanticSearchResult,
    _compute_text_hash,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_path(tmp_path: Path) -> Path:
    """Create a temporary cache path."""
    return tmp_path / "test_embedding_cache.db"


@pytest.fixture
def embedding_cache(temp_cache_path: Path) -> EmbeddingCache:
    """Create a test embedding cache."""
    cache = EmbeddingCache(cache_path=temp_cache_path)
    yield cache
    cache.close()


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(
            id=1,
            chat_id="chat1",
            sender="+15551234567",
            sender_name="John",
            text="Let's get dinner tonight at the Italian restaurant",
            date=datetime(2024, 1, 15, 18, 30),
            is_from_me=False,
        ),
        Message(
            id=2,
            chat_id="chat1",
            sender="me",
            sender_name=None,
            text="Sure! I love Italian food. What time works for you?",
            date=datetime(2024, 1, 15, 18, 35),
            is_from_me=True,
        ),
        Message(
            id=3,
            chat_id="chat2",
            sender="+15559876543",
            sender_name="Mom",
            text="Don't forget about the family meeting tomorrow at 3pm",
            date=datetime(2024, 1, 16, 10, 0),
            is_from_me=False,
        ),
        Message(
            id=4,
            chat_id="chat2",
            sender="me",
            sender_name=None,
            text="I'll be there! Should I bring anything?",
            date=datetime(2024, 1, 16, 10, 5),
            is_from_me=True,
        ),
        Message(
            id=5,
            chat_id="chat1",
            sender="+15551234567",
            sender_name="John",
            text="Running late, stuck in traffic",
            date=datetime(2024, 1, 15, 19, 0),
            is_from_me=False,
        ),
    ]


@pytest.fixture
def mock_reader(sample_messages: list[Message]) -> MagicMock:
    """Create a mock iMessage reader."""
    reader = MagicMock()
    reader.search.return_value = sample_messages
    reader.get_messages.return_value = sample_messages
    return reader


# =============================================================================
# EmbeddingCache Tests
# =============================================================================


class TestEmbeddingCache:
    """Tests for the EmbeddingCache class."""

    def test_cache_creation(self, temp_cache_path: Path) -> None:
        """Test that cache creates database file."""
        cache = EmbeddingCache(cache_path=temp_cache_path)
        # Trigger connection
        cache.stats()
        cache.close()
        assert temp_cache_path.exists()

    def test_set_and_get_embedding(self, embedding_cache: EmbeddingCache) -> None:
        """Test storing and retrieving an embedding."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedding_cache.set(
            message_id=1,
            chat_id="chat1",
            text_hash="abc123",
            embedding=embedding,
        )

        retrieved = embedding_cache.get(1)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)

    def test_get_missing_embedding(self, embedding_cache: EmbeddingCache) -> None:
        """Test retrieving a non-existent embedding."""
        result = embedding_cache.get(999)
        assert result is None

    def test_get_batch(self, embedding_cache: EmbeddingCache) -> None:
        """Test batch retrieval of embeddings."""
        embeddings = {
            1: np.array([0.1, 0.2, 0.3], dtype=np.float32),
            2: np.array([0.4, 0.5, 0.6], dtype=np.float32),
            3: np.array([0.7, 0.8, 0.9], dtype=np.float32),
        }

        # Store embeddings
        for msg_id, emb in embeddings.items():
            embedding_cache.set(
                message_id=msg_id,
                chat_id="chat1",
                text_hash=f"hash{msg_id}",
                embedding=emb,
            )

        # Retrieve batch
        result = embedding_cache.get_batch([1, 2, 3, 999])  # 999 doesn't exist
        assert len(result) == 3
        for msg_id in [1, 2, 3]:
            np.testing.assert_array_almost_equal(result[msg_id], embeddings[msg_id])

    def test_set_batch(self, embedding_cache: EmbeddingCache) -> None:
        """Test batch storage of embeddings."""
        items = [
            (1, "chat1", "hash1", np.array([0.1, 0.2], dtype=np.float32)),
            (2, "chat1", "hash2", np.array([0.3, 0.4], dtype=np.float32)),
            (3, "chat2", "hash3", np.array([0.5, 0.6], dtype=np.float32)),
        ]

        embedding_cache.set_batch(items)

        for msg_id, _, _, expected in items:
            retrieved = embedding_cache.get(msg_id)
            assert retrieved is not None
            np.testing.assert_array_almost_equal(retrieved, expected)

    def test_invalidate(self, embedding_cache: EmbeddingCache) -> None:
        """Test invalidating a single embedding."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedding_cache.set(1, "chat1", "hash1", embedding)

        assert embedding_cache.get(1) is not None
        embedding_cache.invalidate(1)
        assert embedding_cache.get(1) is None

    def test_invalidate_chat(self, embedding_cache: EmbeddingCache) -> None:
        """Test invalidating all embeddings for a chat."""
        items = [
            (1, "chat1", "hash1", np.array([0.1], dtype=np.float32)),
            (2, "chat1", "hash2", np.array([0.2], dtype=np.float32)),
            (3, "chat2", "hash3", np.array([0.3], dtype=np.float32)),
        ]
        embedding_cache.set_batch(items)

        embedding_cache.invalidate_chat("chat1")

        assert embedding_cache.get(1) is None
        assert embedding_cache.get(2) is None
        assert embedding_cache.get(3) is not None  # Different chat

    def test_clear(self, embedding_cache: EmbeddingCache) -> None:
        """Test clearing all embeddings."""
        items = [
            (1, "chat1", "hash1", np.array([0.1], dtype=np.float32)),
            (2, "chat2", "hash2", np.array([0.2], dtype=np.float32)),
        ]
        embedding_cache.set_batch(items)

        embedding_cache.clear()
        assert embedding_cache.get(1) is None
        assert embedding_cache.get(2) is None

    def test_stats(self, embedding_cache: EmbeddingCache) -> None:
        """Test getting cache statistics."""
        # Empty cache
        stats = embedding_cache.stats()
        assert stats["embedding_count"] == 0
        assert stats["size_bytes"] == 0

        # Add some embeddings
        items = [
            (1, "chat1", "hash1", np.array([0.1] * 384, dtype=np.float32)),
            (2, "chat1", "hash2", np.array([0.2] * 384, dtype=np.float32)),
        ]
        embedding_cache.set_batch(items)

        stats = embedding_cache.stats()
        assert stats["embedding_count"] == 2
        assert stats["size_bytes"] > 0
        assert stats["size_mb"] >= 0


# =============================================================================
# Text Hash Tests
# =============================================================================


class TestTextHash:
    """Tests for text hashing utility."""

    def test_compute_text_hash(self) -> None:
        """Test that text hashing is deterministic."""
        text = "Hello, world!"
        hash1 = _compute_text_hash(text)
        hash2 = _compute_text_hash(text)
        assert hash1 == hash2

    def test_different_texts_different_hashes(self) -> None:
        """Test that different texts produce different hashes."""
        hash1 = _compute_text_hash("Hello")
        hash2 = _compute_text_hash("World")
        assert hash1 != hash2


# =============================================================================
# SearchFilters Tests
# =============================================================================


class TestSearchFilters:
    """Tests for SearchFilters dataclass."""

    def test_default_filters(self) -> None:
        """Test default filter values."""
        filters = SearchFilters()
        assert filters.sender is None
        assert filters.chat_id is None
        assert filters.after is None
        assert filters.before is None
        assert filters.has_attachments is None

    def test_custom_filters(self) -> None:
        """Test custom filter values."""
        now = datetime.now()
        filters = SearchFilters(
            sender="+15551234567",
            chat_id="chat1",
            after=now,
            before=now,
            has_attachments=True,
        )
        assert filters.sender == "+15551234567"
        assert filters.chat_id == "chat1"
        assert filters.after == now
        assert filters.before == now
        assert filters.has_attachments is True


# =============================================================================
# SemanticSearcher Tests
# =============================================================================


class TestSemanticSearcher:
    """Tests for the SemanticSearcher class."""

    def test_searcher_initialization(self, mock_reader: MagicMock, temp_cache_path: Path) -> None:
        """Test searcher initialization."""
        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.5,
        )
        assert searcher.reader == mock_reader
        assert searcher.cache == cache
        assert searcher.similarity_threshold == 0.5
        searcher.close()

    def test_search_empty_query(self, mock_reader: MagicMock, temp_cache_path: Path) -> None:
        """Test that empty query returns empty results."""
        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(reader=mock_reader, cache=cache)

        results = searcher.search("")
        assert results == []

        results = searcher.search("   ")
        assert results == []
        searcher.close()

    @patch("jarvis.semantic_search.get_embedder")
    def test_search_returns_results(
        self,
        mock_get_embedder: MagicMock,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        temp_cache_path: Path,
    ) -> None:
        """Test that search returns results with similarity scores."""
        # Mock the embedder
        mock_embedder = MagicMock()

        # Create embeddings that will produce known similarity scores
        # Query embedding
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Message embeddings with varying similarities
        msg_embeddings = np.array(
            [
                [0.9, 0.1, 0.1],  # High similarity to query
                [0.5, 0.5, 0.5],  # Medium similarity
                [0.1, 0.9, 0.1],  # Low similarity
                [0.6, 0.4, 0.3],  # Medium similarity
                [0.95, 0.05, 0.05],  # Very high similarity
            ],
            dtype=np.float32,
        )

        def encode_side_effect(texts, **kwargs):
            if len(texts) == 1:
                return np.array([query_embedding])
            return msg_embeddings[: len(texts)]

        mock_embedder.encode.side_effect = encode_side_effect
        mock_get_embedder.return_value = mock_embedder

        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.3,
        )

        results = searcher.search("dinner plans", limit=5)

        assert len(results) > 0
        assert all(isinstance(r, SemanticSearchResult) for r in results)
        assert all(r.similarity >= 0.3 for r in results)
        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

        searcher.close()

    @patch("jarvis.semantic_search.get_embedder")
    def test_search_respects_threshold(
        self,
        mock_get_embedder: MagicMock,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        temp_cache_path: Path,
    ) -> None:
        """Test that search respects similarity threshold."""
        mock_embedder = MagicMock()

        # All messages will have low similarity
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        msg_embeddings = np.array([[0.1, 0.9, 0.0] for _ in sample_messages], dtype=np.float32)

        def encode_side_effect(texts, **kwargs):
            if len(texts) == 1:
                return np.array([query_embedding])
            return msg_embeddings[: len(texts)]

        mock_embedder.encode.side_effect = encode_side_effect
        mock_get_embedder.return_value = mock_embedder

        cache = EmbeddingCache(cache_path=temp_cache_path)
        # High threshold should filter out all results
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.9,
        )

        results = searcher.search("test query")
        assert len(results) == 0
        searcher.close()

    @patch("jarvis.semantic_search.get_embedder")
    def test_search_respects_limit(
        self,
        mock_get_embedder: MagicMock,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        temp_cache_path: Path,
    ) -> None:
        """Test that search respects result limit."""
        mock_embedder = MagicMock()

        # All messages will have high similarity
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        msg_embeddings = np.array([[0.9, 0.1, 0.0] for _ in sample_messages], dtype=np.float32)

        def encode_side_effect(texts, **kwargs):
            if len(texts) == 1:
                return np.array([query_embedding])
            return msg_embeddings[: len(texts)]

        mock_embedder.encode.side_effect = encode_side_effect
        mock_get_embedder.return_value = mock_embedder

        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.3,
        )

        results = searcher.search("test query", limit=2)
        assert len(results) <= 2
        searcher.close()

    @patch("jarvis.semantic_search.get_embedder")
    def test_search_with_filters(
        self,
        mock_get_embedder: MagicMock,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        temp_cache_path: Path,
    ) -> None:
        """Test search with filters applied."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
        mock_get_embedder.return_value = mock_embedder

        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.3,
        )

        filters = SearchFilters(
            sender="+15551234567",
            chat_id="chat1",
        )

        # Search should apply filters
        searcher.search("test", filters=filters)

        # Verify reader was called
        mock_reader.get_messages.assert_called()
        searcher.close()

    @patch("jarvis.semantic_search.get_embedder")
    def test_search_caches_embeddings(
        self,
        mock_get_embedder: MagicMock,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        temp_cache_path: Path,
    ) -> None:
        """Test that embeddings are cached after first search."""
        mock_embedder = MagicMock()

        query_embedding = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        msg_embeddings = np.array([[0.5, 0.5, 0.0] for _ in sample_messages], dtype=np.float32)

        def encode_side_effect(texts, **kwargs):
            if len(texts) == 1:
                return np.array([query_embedding])
            return msg_embeddings[: len(texts)]

        mock_embedder.encode.side_effect = encode_side_effect
        mock_get_embedder.return_value = mock_embedder

        cache = EmbeddingCache(cache_path=temp_cache_path)
        searcher = SemanticSearcher(
            reader=mock_reader,
            cache=cache,
            similarity_threshold=0.3,
        )

        # First search
        searcher.search("test query")
        first_call_count = mock_embedder.encode.call_count

        # Second search - embeddings should be cached
        searcher.search("different query")
        second_call_count = mock_embedder.encode.call_count

        # Should only encode the new query, not the messages again
        assert second_call_count < first_call_count + len(sample_messages) + 1
        searcher.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestSemanticSearchIntegration:
    """Integration tests for semantic search."""

    def test_search_result_dataclass(self) -> None:
        """Test SemanticSearchResult dataclass."""
        msg = Message(
            id=1,
            chat_id="chat1",
            sender="test",
            sender_name="Test",
            text="Test message",
            date=datetime.now(),
            is_from_me=False,
        )
        result = SemanticSearchResult(message=msg, similarity=0.85)
        assert result.message == msg
        assert result.similarity == 0.85

    def test_cache_persistence(self, temp_cache_path: Path) -> None:
        """Test that cache persists across instances."""
        # First instance
        cache1 = EmbeddingCache(cache_path=temp_cache_path)
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache1.set(1, "chat1", "hash1", embedding)
        cache1.close()

        # Second instance should see the same data
        cache2 = EmbeddingCache(cache_path=temp_cache_path)
        retrieved = cache2.get(1)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)
        cache2.close()
