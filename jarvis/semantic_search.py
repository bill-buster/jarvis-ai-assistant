"""Semantic search for iMessage conversations.

Uses sentence embeddings (bge-small-en-v1.5 via unified adapter) to find
messages by meaning rather than exact text matching. Embeddings are cached
in SQLite for efficient repeated searches.

Example:
    from jarvis.semantic_search import SemanticSearcher

    with ChatDBReader() as reader:
        searcher = SemanticSearcher(reader)
        results = searcher.search("dinner plans", limit=10)
        for result in results:
            print(f"{result.similarity:.2f}: {result.message.text}")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from contracts.imessage import Message
    from integrations.imessage import ChatDBReader

from jarvis.embedding_adapter import get_embedder

# Embedding dimension for bge-small-en-v1.5
EMBEDDING_DIM = 384

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".jarvis" / "embedding_cache.db"


@dataclass
class SemanticSearchResult:
    """Result from semantic search with similarity score."""

    message: Message
    similarity: float  # Cosine similarity score (0.0 to 1.0)


@dataclass
class SearchFilters:
    """Filters for semantic search."""

    sender: str | None = None
    chat_id: str | None = None
    after: datetime | None = None
    before: datetime | None = None
    has_attachments: bool | None = None


# Cache schema version - bump when changing embedding model to invalidate cache
CACHE_SCHEMA_VERSION = 2


class EmbeddingCache:
    """SQLite-backed cache for message embeddings.

    Stores embeddings keyed by message ID. Supports lazy-loading and
    batch operations for efficiency.

    Thread-safety:
        Uses internal locking for concurrent access. Safe for multi-threaded use.
    """

    SCHEMA_VERSION = CACHE_SCHEMA_VERSION

    def __init__(self, cache_path: Path | None = None) -> None:
        """Initialize the embedding cache.

        Args:
            cache_path: Path to SQLite database file. Defaults to ~/.jarvis/embedding_cache.db
        """
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.cache_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._conn
        if conn is None:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)

        # Check schema version
        cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        current_version = row["version"] if row else 0

        if current_version < self.SCHEMA_VERSION:
            # Create or migrate schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    message_id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_chat_id
                ON embeddings(chat_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash
                ON embeddings(text_hash)
            """)

            # Update schema version
            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )
            conn.commit()

    def get(self, message_id: int) -> np.ndarray | None:
        """Get embedding for a message.

        Args:
            message_id: The message ROWID

        Returns:
            Embedding as numpy array, or None if not cached
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE message_id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row["embedding"], dtype=np.float32)
            return None

    def get_batch(self, message_ids: list[int]) -> dict[int, np.ndarray]:
        """Get embeddings for multiple messages.

        Args:
            message_ids: List of message ROWIDs

        Returns:
            Dictionary mapping message_id to embedding
        """
        if not message_ids:
            return {}

        with self._lock:
            conn = self._get_connection()
            placeholders = ",".join("?" * len(message_ids))
            query = (
                f"SELECT message_id, embedding FROM embeddings WHERE message_id IN ({placeholders})"
            )
            cursor = conn.execute(query, message_ids)
            return {
                row["message_id"]: np.frombuffer(row["embedding"], dtype=np.float32)
                for row in cursor.fetchall()
            }

    def set(
        self,
        message_id: int,
        chat_id: str,
        text_hash: str,
        embedding: np.ndarray,
    ) -> None:
        """Store embedding for a message.

        Args:
            message_id: The message ROWID
            chat_id: The conversation ID
            text_hash: Hash of the message text (for invalidation)
            embedding: The embedding vector
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (message_id, chat_id, text_hash, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    text_hash,
                    embedding.astype(np.float32).tobytes(),
                    time.time(),
                ),
            )
            conn.commit()

    def set_batch(
        self,
        items: list[tuple[int, str, str, np.ndarray]],
    ) -> None:
        """Store embeddings for multiple messages.

        Args:
            items: List of (message_id, chat_id, text_hash, embedding) tuples
        """
        if not items:
            return

        with self._lock:
            conn = self._get_connection()
            current_time = time.time()
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (message_id, chat_id, text_hash, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        msg_id,
                        chat_id,
                        text_hash,
                        emb.astype(np.float32).tobytes(),
                        current_time,
                    )
                    for msg_id, chat_id, text_hash, emb in items
                ],
            )
            conn.commit()

    def invalidate(self, message_id: int) -> None:
        """Remove embedding for a message.

        Args:
            message_id: The message ROWID
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "DELETE FROM embeddings WHERE message_id = ?",
                (message_id,),
            )
            conn.commit()

    def invalidate_chat(self, chat_id: str) -> None:
        """Remove all embeddings for a conversation.

        Args:
            chat_id: The conversation ID
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "DELETE FROM embeddings WHERE chat_id = ?",
                (chat_id,),
            )
            conn.commit()

    def clear(self) -> None:
        """Remove all cached embeddings."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM embeddings")
            conn.commit()

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with count and size information
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) as count FROM embeddings")
            row = cursor.fetchone()
            count: int = row["count"] if row else 0

            # Get approximate size in bytes
            cursor = conn.execute("SELECT SUM(LENGTH(embedding)) as size FROM embeddings")
            row = cursor.fetchone()
            size: int = row["size"] if row and row["size"] else 0

            return {
                "embedding_count": count,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
            }

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _compute_text_hash(text: str) -> str:
    """Compute hash of message text for cache invalidation."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


class SemanticSearcher:
    """Semantic search over iMessage conversations.

    Uses sentence embeddings to find messages by semantic similarity.
    Embeddings are cached in SQLite for efficient repeated searches.

    Example:
        with ChatDBReader() as reader:
            searcher = SemanticSearcher(reader)

            # Basic search
            results = searcher.search("dinner plans", limit=10)

            # With filters
            from datetime import datetime, timedelta
            results = searcher.search(
                "meeting tomorrow",
                filters=SearchFilters(
                    after=datetime.now() - timedelta(days=7),
                    chat_id="chat123",
                ),
                limit=20,
            )
    """

    def __init__(
        self,
        reader: ChatDBReader,
        cache: EmbeddingCache | None = None,
        similarity_threshold: float = 0.3,
    ) -> None:
        """Initialize the semantic searcher.

        Args:
            reader: iMessage database reader
            cache: Embedding cache (creates default if not provided)
            similarity_threshold: Minimum similarity score for results (0.0-1.0)
        """
        self.reader = reader
        self.cache = cache or EmbeddingCache()
        self.similarity_threshold = similarity_threshold

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (shape: [len(texts), EMBEDDING_DIM])
        """
        embedder = get_embedder()
        return embedder.encode(texts, normalize=True)

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding.

        Args:
            text: Text string

        Returns:
            Embedding vector (shape: [EMBEDDING_DIM])
        """
        embedder = get_embedder()
        return embedder.encode([text], normalize=True)[0]

    def _get_messages_to_index(
        self,
        filters: SearchFilters | None = None,
        limit: int = 1000,
    ) -> list[Message]:
        """Get messages that match the filters for indexing.

        Args:
            filters: Search filters
            limit: Maximum messages to retrieve

        Returns:
            List of messages to index
        """
        filters = filters or SearchFilters()

        if filters.chat_id:
            # Get messages from specific conversation
            messages = self.reader.get_messages(
                chat_id=filters.chat_id,
                limit=limit,
                before=filters.before,
            )
        else:
            # Search across all conversations - need to use search with wildcard
            # or get recent messages from all conversations
            messages = self.reader.search(
                query="%",  # Match all
                limit=limit,
                sender=filters.sender,
                after=filters.after,
                before=filters.before,
                has_attachments=filters.has_attachments,
            )

        # Apply additional filters
        filtered = []
        for msg in messages:
            if not msg.text or len(msg.text.strip()) < 2:
                continue  # Skip empty/trivial messages

            if filters.sender and filters.sender.lower() not in (
                msg.sender.lower(),
                (msg.sender_name or "").lower(),
            ):
                if filters.sender != "me" or not msg.is_from_me:
                    continue

            if filters.after and msg.date < filters.after:
                continue

            if filters.before and msg.date > filters.before:
                continue

            if filters.has_attachments is not None:
                has_att = len(msg.attachments) > 0
                if has_att != filters.has_attachments:
                    continue

            filtered.append(msg)

        return filtered

    def _ensure_embeddings(self, messages: list[Message]) -> dict[int, np.ndarray]:
        """Ensure all messages have embeddings, computing missing ones.

        Args:
            messages: List of messages to embed

        Returns:
            Dictionary mapping message_id to embedding
        """
        message_ids = [msg.id for msg in messages]
        cached = self.cache.get_batch(message_ids)

        # Find messages without cached embeddings
        missing = []
        for msg in messages:
            if msg.id not in cached:
                text_hash = _compute_text_hash(msg.text)
                missing.append((msg, text_hash))

        if missing:
            logger.debug("Computing embeddings for %d messages", len(missing))
            texts = [msg.text for msg, _ in missing]
            embeddings = self._encode_texts(texts)

            # Store in cache
            items = [
                (msg.id, msg.chat_id, text_hash, emb)
                for (msg, text_hash), emb in zip(missing, embeddings)
            ]
            self.cache.set_batch(items)

            # Add to result
            for (msg, _), emb in zip(missing, embeddings):
                cached[msg.id] = emb

        return cached

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
        index_limit: int = 1000,
    ) -> list[SemanticSearchResult]:
        """Search messages by semantic similarity.

        Args:
            query: Search query (natural language)
            filters: Optional filters for sender, date, etc.
            limit: Maximum number of results to return
            index_limit: Maximum messages to index/search through

        Returns:
            List of SemanticSearchResult sorted by similarity (highest first)
        """
        if not query or not query.strip():
            return []

        # Get messages to search through
        messages = self._get_messages_to_index(filters, limit=index_limit)
        if not messages:
            return []

        # Ensure all messages have embeddings
        embeddings = self._ensure_embeddings(messages)

        # Encode query
        query_embedding = self._encode_single(query)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Compute similarities
        results: list[SemanticSearchResult] = []
        for msg in messages:
            if msg.id not in embeddings:
                continue

            msg_embedding = embeddings[msg.id]
            msg_norm = np.linalg.norm(msg_embedding)
            if msg_norm == 0:
                continue

            # Cosine similarity
            similarity = float(np.dot(query_embedding, msg_embedding) / (query_norm * msg_norm))

            if similarity >= self.similarity_threshold:
                results.append(SemanticSearchResult(message=msg, similarity=similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda r: r.similarity, reverse=True)

        return results[:limit]

    def search_similar_to_message(
        self,
        message: Message,
        filters: SearchFilters | None = None,
        limit: int = 10,
        index_limit: int = 500,
    ) -> list[SemanticSearchResult]:
        """Find messages similar to a given message.

        Args:
            message: The message to find similar messages for
            filters: Optional filters
            limit: Maximum results to return
            index_limit: Maximum messages to search through

        Returns:
            List of similar messages (excluding the input message)
        """
        if not message.text:
            return []

        results = self.search(
            query=message.text,
            filters=filters,
            limit=limit + 1,  # +1 to account for possibly matching itself
            index_limit=index_limit,
        )

        # Filter out the original message
        return [r for r in results if r.message.id != message.id][:limit]

    def close(self) -> None:
        """Close resources."""
        self.cache.close()


# Singleton instance management
_searcher_instance: SemanticSearcher | None = None
_searcher_lock = threading.Lock()


def get_semantic_searcher(reader: ChatDBReader) -> SemanticSearcher:
    """Get or create a singleton SemanticSearcher instance.

    Args:
        reader: iMessage database reader

    Returns:
        SemanticSearcher instance
    """
    global _searcher_instance
    if _searcher_instance is None:
        with _searcher_lock:
            if _searcher_instance is None:
                _searcher_instance = SemanticSearcher(reader)
    return _searcher_instance


def reset_semantic_searcher() -> None:
    """Reset the singleton SemanticSearcher instance."""
    global _searcher_instance
    with _searcher_lock:
        if _searcher_instance is not None:
            _searcher_instance.close()
            _searcher_instance = None
