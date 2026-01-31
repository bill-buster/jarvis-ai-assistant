"""Embedding Store - SQLite-backed message embedding storage and search.

This module provides STORAGE and SEARCH for message embeddings. It does NOT
compute embeddings directly - it delegates to jarvis/embedding_adapter.py.

Key Features:
- SQLite-backed persistent storage of message embeddings
- Semantic search across messages with optional contact filtering
- Relationship profiling based on conversation patterns
- Similar situation detection for context-aware responses
- Privacy-preserving local storage (no cloud transmission)

Architecture (3-layer embedding stack):
    1. jarvis/embeddings.py       - Embedding STORAGE (this file)
           Stores embeddings in SQLite, provides search APIs
    2. jarvis/embedding_adapter.py - UNIFIED INTERFACE (use for computing)
           MLX-first with CPU fallback
    3. models/embeddings.py       - MLX SERVICE CLIENT (low-level)
           Direct HTTP client to MLX microservice

NOTE: If you need to compute embeddings, import from jarvis/embedding_adapter.py:
    from jarvis.embedding_adapter import get_embedder

Usage:
    from jarvis.embeddings import (
        get_embedding_store,
        find_similar_messages,
        get_relationship_profile,
    )

    # Index messages (computes embeddings internally via embedding_adapter)
    store = get_embedding_store()
    store.index_messages(messages)

    # Semantic search
    results = find_similar_messages("dinner plans this weekend", contact_id="chat123")

    # Get relationship profile
    profile = get_relationship_profile("chat123")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts.imessage import Message
from jarvis.embedding_adapter import get_embedder
from jarvis.errors import ErrorCode, JarvisError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB_PATH = Path.home() / ".jarvis" / "embeddings.db"
# Use bge-small-en-v1.5 via unified adapter for consistency
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Dimension for bge-small-en-v1.5
BATCH_SIZE = 100  # Batch size for embedding computation
MIN_TEXT_LENGTH = 3  # Minimum text length to embed
# Schema version - bump when changing embedding model to invalidate cache
SCHEMA_VERSION = 2


# =============================================================================
# Exceptions
# =============================================================================


class EmbeddingError(JarvisError):
    """Raised when embedding operations fail."""

    default_message = "Embedding operation failed"
    default_code = ErrorCode.UNKNOWN


class EmbeddingStoreError(EmbeddingError):
    """Raised when embedding storage operations fail."""

    default_message = "Embedding storage operation failed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SimilarMessage:
    """A message with its similarity score."""

    message_id: int
    chat_id: str
    text: str
    sender: str | None
    sender_name: str | None
    timestamp: datetime
    is_from_me: bool
    similarity: float


@dataclass
class ConversationContext:
    """Context from a similar past conversation."""

    messages: list[SimilarMessage]
    topic: str
    avg_similarity: float


@dataclass
class RelationshipProfile:
    """Aggregated profile of communication patterns with a contact."""

    contact_id: str
    display_name: str | None = None
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    common_topics: list[str] = field(default_factory=list)
    typical_tone: str = "casual"  # casual, professional, mixed
    avg_message_length: float = 0.0
    response_patterns: dict[str, Any] = field(default_factory=dict)
    last_interaction: datetime | None = None


# =============================================================================
# Embedding Functions (using unified adapter)
# =============================================================================


def _compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Normalized embedding vector

    Raises:
        EmbeddingError: If embedding computation fails
    """
    try:
        embedder = get_embedder()
        embedding = embedder.encode([text], normalize=True)[0]
        return embedding
    except Exception as e:
        logger.exception("Failed to compute embedding")
        raise EmbeddingError(
            f"Failed to compute embedding: {e}",
            cause=e,
        ) from e


def _compute_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a batch of texts.

    Args:
        texts: List of texts to embed

    Returns:
        Array of normalized embedding vectors

    Raises:
        EmbeddingError: If embedding computation fails
    """
    if not texts:
        return np.array([])

    try:
        embedder = get_embedder()
        embeddings = embedder.encode(texts, normalize=True)
        return embeddings
    except Exception as e:
        logger.exception("Failed to compute batch embeddings")
        raise EmbeddingError(
            f"Failed to compute batch embeddings: {e}",
            cause=e,
        ) from e


# =============================================================================
# Embedding Store
# =============================================================================


class EmbeddingStore:
    """SQLite-backed storage for message embeddings.

    Stores embeddings with metadata for efficient retrieval.
    Uses numpy serialization for embedding vectors.

    Thread Safety:
        Each thread should use its own store instance, or use
        get_embedding_store() for a shared singleton.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the embedding store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.jarvis/embeddings.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_db_exists()
        self._init_schema()

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory and file exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        Returns:
            SQLite connection with row factory
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Main embeddings table
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    text_hash TEXT NOT NULL,
                    sender TEXT,
                    sender_name TEXT,
                    timestamp INTEGER NOT NULL,
                    is_from_me INTEGER NOT NULL,
                    text_preview TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                );

                -- Indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_embeddings_chat_id
                    ON message_embeddings(chat_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp
                    ON message_embeddings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_embeddings_sender
                    ON message_embeddings(sender);
                CREATE INDEX IF NOT EXISTS idx_embeddings_is_from_me
                    ON message_embeddings(is_from_me);
                CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash
                    ON message_embeddings(text_hash);

                -- Relationship profiles cache
                CREATE TABLE IF NOT EXISTS relationship_profiles (
                    contact_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    profile_data TEXT NOT NULL,
                    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
                );

                -- Index stats for monitoring
                CREATE TABLE IF NOT EXISTS index_stats (
                    stat_key TEXT PRIMARY KEY,
                    stat_value TEXT,
                    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
                );
            """)
            conn.commit()

    def index_message(self, message: Message) -> bool:
        """Index a single message.

        Args:
            message: Message to index

        Returns:
            True if indexed, False if skipped (too short or duplicate)
        """
        # Skip very short messages
        if not message.text or len(message.text.strip()) < MIN_TEXT_LENGTH:
            return False

        text_hash = hashlib.md5(message.text.encode(), usedforsecurity=False).hexdigest()

        with self._get_connection() as conn:
            # Check if already indexed
            existing = conn.execute(
                "SELECT 1 FROM message_embeddings WHERE message_id = ?",
                (message.id,),
            ).fetchone()

            if existing:
                return False

            # Compute embedding
            embedding = _compute_embedding(message.text)
            embedding_blob = embedding.tobytes()

            # Store
            conn.execute(
                """
                INSERT INTO message_embeddings
                (message_id, chat_id, embedding, text_hash, sender, sender_name,
                 timestamp, is_from_me, text_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.chat_id,
                    embedding_blob,
                    text_hash,
                    message.sender,
                    message.sender_name,
                    int(message.date.timestamp()),
                    1 if message.is_from_me else 0,
                    message.text[:200],
                ),
            )
            conn.commit()
            return True

    def index_messages(
        self, messages: list[Message], progress_callback: Any | None = None
    ) -> dict[str, int]:
        """Index multiple messages with batch processing.

        Args:
            messages: Messages to index
            progress_callback: Optional callback(indexed, total) for progress

        Returns:
            Dict with counts: indexed, skipped, duplicates
        """
        stats = {"indexed": 0, "skipped": 0, "duplicates": 0}

        # Filter out too-short messages
        valid_messages = [m for m in messages if m.text and len(m.text.strip()) >= MIN_TEXT_LENGTH]
        stats["skipped"] = len(messages) - len(valid_messages)

        if not valid_messages:
            return stats

        with self._get_connection() as conn:
            # Get existing message IDs
            message_ids = [m.id for m in valid_messages]
            placeholders = ",".join("?" * len(message_ids))
            query = (
                f"SELECT message_id FROM message_embeddings WHERE message_id IN ({placeholders})"
            )
            existing = set(row[0] for row in conn.execute(query, message_ids).fetchall())

            # Filter to only new messages
            new_messages = [m for m in valid_messages if m.id not in existing]
            stats["duplicates"] = len(valid_messages) - len(new_messages)

            if not new_messages:
                return stats

            # Batch compute embeddings
            for i in range(0, len(new_messages), BATCH_SIZE):
                batch = new_messages[i : i + BATCH_SIZE]
                texts = [m.text for m in batch]
                embeddings = _compute_embeddings_batch(texts)

                # Store batch
                for msg, embedding in zip(batch, embeddings):
                    text_hash = hashlib.md5(msg.text.encode(), usedforsecurity=False).hexdigest()
                    embedding_blob = embedding.tobytes()

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO message_embeddings
                        (message_id, chat_id, embedding, text_hash, sender, sender_name,
                         timestamp, is_from_me, text_preview)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            msg.id,
                            msg.chat_id,
                            embedding_blob,
                            text_hash,
                            msg.sender,
                            msg.sender_name,
                            int(msg.date.timestamp()),
                            1 if msg.is_from_me else 0,
                            msg.text[:200],
                        ),
                    )
                    stats["indexed"] += 1

                conn.commit()

                if progress_callback:
                    progress_callback(stats["indexed"], len(new_messages))

        return stats

    def find_similar(
        self,
        query: str,
        chat_id: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: list[int] | None = None,
    ) -> list[SimilarMessage]:
        """Find messages similar to a query.

        Args:
            query: Query text to find similar messages for
            chat_id: Optional chat ID to filter by
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude from results

        Returns:
            List of similar messages sorted by similarity
        """
        query_embedding = _compute_embedding(query)

        with self._get_connection() as conn:
            # Build query
            sql = """
                SELECT message_id, chat_id, embedding, text_preview,
                       sender, sender_name, timestamp, is_from_me
                FROM message_embeddings
            """
            params: list[Any] = []

            conditions = []
            if chat_id:
                conditions.append("chat_id = ?")
                params.append(chat_id)

            if exclude_message_ids:
                placeholders = ",".join("?" * len(exclude_message_ids))
                conditions.append(f"message_id NOT IN ({placeholders})")
                params.extend(exclude_message_ids)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Compute similarities
        results: list[SimilarMessage] = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(query_embedding, embedding))

            if similarity >= min_similarity:
                results.append(
                    SimilarMessage(
                        message_id=row["message_id"],
                        chat_id=row["chat_id"],
                        text=row["text_preview"] or "",
                        sender=row["sender"],
                        sender_name=row["sender_name"],
                        timestamp=datetime.fromtimestamp(row["timestamp"]),
                        is_from_me=bool(row["is_from_me"]),
                        similarity=similarity,
                    )
                )

        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    def find_similar_situations(
        self,
        context: str,
        chat_id: str | None = None,
        limit: int = 5,
        context_window: int = 3,
    ) -> list[ConversationContext]:
        """Find past conversations with similar patterns.

        Looks for message sequences that are semantically similar to
        the provided context, useful for finding relevant examples.

        Args:
            context: Current conversation context
            chat_id: Optional chat ID to filter by
            limit: Maximum number of situations to return
            context_window: Number of surrounding messages to include

        Returns:
            List of similar conversation contexts
        """
        # Find similar messages first
        similar = self.find_similar(
            query=context,
            chat_id=chat_id,
            limit=limit * 2,  # Get more to filter
            min_similarity=0.4,
        )

        if not similar:
            return []

        # Group by conversation and get surrounding context
        situations: list[ConversationContext] = []
        seen_message_ids: set[int] = set()

        with self._get_connection() as conn:
            for msg in similar:
                if msg.message_id in seen_message_ids:
                    continue

                # Get surrounding messages
                rows = conn.execute(
                    """
                    SELECT message_id, chat_id, text_preview, sender, sender_name,
                           timestamp, is_from_me
                    FROM message_embeddings
                    WHERE chat_id = ?
                      AND timestamp BETWEEN ? - 3600 AND ? + 3600
                    ORDER BY timestamp
                    LIMIT ?
                    """,
                    (
                        msg.chat_id,
                        int(msg.timestamp.timestamp()),
                        int(msg.timestamp.timestamp()),
                        context_window * 2 + 1,
                    ),
                ).fetchall()

                context_messages = []
                for row in rows:
                    seen_message_ids.add(row["message_id"])
                    is_match = row["message_id"] == msg.message_id
                    sim = msg.similarity if is_match else 0.0
                    context_messages.append(
                        SimilarMessage(
                            message_id=row["message_id"],
                            chat_id=row["chat_id"],
                            text=row["text_preview"] or "",
                            sender=row["sender"],
                            sender_name=row["sender_name"],
                            timestamp=datetime.fromtimestamp(row["timestamp"]),
                            is_from_me=bool(row["is_from_me"]),
                            similarity=sim,
                        )
                    )

                if context_messages:
                    # Detect topic from messages
                    topic = self._detect_topic(context_messages)
                    avg_sim = sum(m.similarity for m in context_messages) / len(context_messages)

                    situations.append(
                        ConversationContext(
                            messages=context_messages,
                            topic=topic,
                            avg_similarity=avg_sim,
                        )
                    )

                if len(situations) >= limit:
                    break

        return situations

    def _detect_topic(self, messages: list[SimilarMessage]) -> str:
        """Detect the topic of a message sequence.

        Uses simple keyword matching for topic detection.

        Args:
            messages: List of messages to analyze

        Returns:
            Detected topic string
        """
        text = " ".join(m.text.lower() for m in messages)

        # Topic keywords (simple heuristic)
        topic_keywords = {
            "planning": ["plan", "schedule", "meeting", "when", "time", "tomorrow", "today"],
            "logistics": ["where", "address", "location", "pick up", "drop off", "arrive"],
            "catching_up": ["how are you", "what's up", "been", "doing", "news"],
            "emotional_support": ["feel", "sorry", "worried", "happy", "sad", "love", "miss"],
            "questions": ["?", "what", "why", "how", "which", "who"],
            "confirmations": ["yes", "ok", "sure", "got it", "sounds good", "perfect"],
        }

        topic_scores: dict[str, int] = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            return max(topic_scores, key=lambda t: topic_scores[t])
        return "general"

    def get_relationship_profile(self, contact_id: str) -> RelationshipProfile:
        """Get or compute a relationship profile for a contact.

        Args:
            contact_id: Chat ID to get profile for

        Returns:
            RelationshipProfile with aggregated statistics
        """
        with self._get_connection() as conn:
            # Get basic stats
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received,
                    AVG(LENGTH(text_preview)) as avg_length,
                    MAX(timestamp) as last_interaction,
                    MIN(sender_name) as display_name
                FROM message_embeddings
                WHERE chat_id = ?
                """,
                (contact_id,),
            ).fetchone()

            if not row or row["total"] == 0:
                return RelationshipProfile(contact_id=contact_id)

            # Get all messages for tone detection
            messages = conn.execute(
                """
                SELECT text_preview, is_from_me
                FROM message_embeddings
                WHERE chat_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
                """,
                (contact_id,),
            ).fetchall()

            # Detect tone from messages
            tone = self._detect_tone([r["text_preview"] for r in messages if r["text_preview"]])

            # Extract common topics
            texts_for_topics = [r["text_preview"] for r in messages if r["text_preview"]]
            topics = self._extract_topics(texts_for_topics)

            # Analyze response patterns
            response_patterns = self._analyze_response_patterns(contact_id, conn)

            return RelationshipProfile(
                contact_id=contact_id,
                display_name=row["display_name"],
                total_messages=row["total"],
                sent_count=row["sent"],
                received_count=row["received"],
                common_topics=topics[:5],
                typical_tone=tone,
                avg_message_length=row["avg_length"] or 0.0,
                response_patterns=response_patterns,
                last_interaction=datetime.fromtimestamp(row["last_interaction"])
                if row["last_interaction"]
                else None,
            )

    def _detect_tone(self, texts: list[str]) -> str:
        """Detect overall tone from messages.

        Args:
            texts: List of message texts

        Returns:
            Tone string: casual, professional, or mixed
        """
        from jarvis.prompts import detect_tone as prompts_detect_tone

        tone = prompts_detect_tone(texts)
        return tone

    def _extract_topics(self, texts: list[str]) -> list[str]:
        """Extract common topics from messages.

        Args:
            texts: List of message texts

        Returns:
            List of topic strings
        """
        # Simple word frequency approach
        words: list[str] = []
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "it",
            "to",
            "and",
            "or",
            "of",
            "in",
            "on",
            "for",
            "with",
            "at",
            "by",
            "from",
            "this",
            "that",
            "i",
            "you",
            "we",
            "they",
            "he",
            "she",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "just",
            "so",
            "but",
            "if",
            "then",
            "than",
            "too",
            "very",
            "can",
            "all",
            "there",
            "here",
            "when",
            "what",
            "which",
            "who",
            "whom",
            "where",
            "why",
            "how",
            "no",
            "not",
            "yes",
            "ok",
            "okay",
        }

        for text in texts:
            if text:
                text_words = text.lower().split()
                words.extend(w for w in text_words if len(w) > 2 and w not in stop_words)

        # Get most common words as topics
        counter = Counter(words)
        return [word for word, _ in counter.most_common(10)]

    def _analyze_response_patterns(
        self, contact_id: str, conn: sqlite3.Connection
    ) -> dict[str, Any]:
        """Analyze response patterns for a contact.

        Args:
            contact_id: Chat ID to analyze
            conn: Database connection

        Returns:
            Dict with pattern analysis
        """
        # Get message timestamps for response time analysis
        rows = conn.execute(
            """
            SELECT timestamp, is_from_me
            FROM message_embeddings
            WHERE chat_id = ?
            ORDER BY timestamp
            LIMIT 500
            """,
            (contact_id,),
        ).fetchall()

        if len(rows) < 2:
            return {}

        # Calculate response times
        response_times: list[int] = []
        prev_row = rows[0]
        for row in rows[1:]:
            # Response is when sender changes
            if row["is_from_me"] != prev_row["is_from_me"]:
                diff = row["timestamp"] - prev_row["timestamp"]
                # Only count responses within 24 hours
                if 0 < diff < 86400:
                    response_times.append(diff)
            prev_row = row

        patterns: dict[str, Any] = {}
        if response_times:
            patterns["avg_response_time_seconds"] = sum(response_times) / len(response_times)
            patterns["quick_responses"] = sum(1 for t in response_times if t < 300)  # < 5 min
            patterns["slow_responses"] = sum(1 for t in response_times if t > 3600)  # > 1 hour

        return patterns

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dict with index stats
        """
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
            chats = conn.execute(
                "SELECT COUNT(DISTINCT chat_id) FROM message_embeddings"
            ).fetchone()[0]
            oldest = conn.execute("SELECT MIN(timestamp) FROM message_embeddings").fetchone()[0]
            newest = conn.execute("SELECT MAX(timestamp) FROM message_embeddings").fetchone()[0]

            return {
                "total_embeddings": total,
                "unique_chats": chats,
                "oldest_message": datetime.fromtimestamp(oldest).isoformat() if oldest else None,
                "newest_message": datetime.fromtimestamp(newest).isoformat() if newest else None,
                "db_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }

    def clear(self) -> None:
        """Clear all stored embeddings."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM message_embeddings")
            conn.execute("DELETE FROM relationship_profiles")
            conn.execute("DELETE FROM index_stats")
            conn.commit()
        logger.info("Cleared all embeddings from store")


# =============================================================================
# Singleton Access
# =============================================================================

_store: EmbeddingStore | None = None
_store_lock = threading.Lock()


def get_embedding_store(db_path: Path | str | None = None) -> EmbeddingStore:
    """Get or create the singleton embedding store.

    Args:
        db_path: Optional custom database path

    Returns:
        The shared EmbeddingStore instance
    """
    global _store

    # Fast path
    if _store is not None and db_path is None:
        return _store

    with _store_lock:
        # Double-check
        if _store is not None and db_path is None:
            return _store

        if db_path:
            # Custom path requested, create new instance
            return EmbeddingStore(db_path)

        # Create default singleton
        _store = EmbeddingStore()
        return _store


def reset_embedding_store() -> None:
    """Reset the singleton embedding store.

    Useful for testing or when the database needs to be reloaded.
    """
    global _store
    with _store_lock:
        _store = None


# =============================================================================
# Convenience Functions
# =============================================================================


def find_similar_messages(
    query: str,
    contact_id: str | None = None,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[SimilarMessage]:
    """Find messages semantically similar to a query.

    Convenience function that uses the singleton store.

    Args:
        query: Query text to find similar messages for
        contact_id: Optional chat ID to filter by
        limit: Maximum number of results
        min_similarity: Minimum similarity threshold (0-1)

    Returns:
        List of similar messages sorted by similarity
    """
    store = get_embedding_store()
    return store.find_similar(
        query=query,
        chat_id=contact_id,
        limit=limit,
        min_similarity=min_similarity,
    )


def find_similar_situations(
    context: str,
    contact_id: str | None = None,
    limit: int = 5,
) -> list[ConversationContext]:
    """Find past conversations with similar patterns.

    Convenience function that uses the singleton store.

    Args:
        context: Current conversation context
        contact_id: Optional chat ID to filter by
        limit: Maximum number of situations to return

    Returns:
        List of similar conversation contexts
    """
    store = get_embedding_store()
    return store.find_similar_situations(
        context=context,
        chat_id=contact_id,
        limit=limit,
    )


def get_relationship_profile(contact_id: str) -> RelationshipProfile:
    """Get the relationship profile for a contact.

    Convenience function that uses the singleton store.

    Args:
        contact_id: Chat ID to get profile for

    Returns:
        RelationshipProfile with aggregated statistics
    """
    store = get_embedding_store()
    return store.get_relationship_profile(contact_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "EmbeddingError",
    "EmbeddingStoreError",
    # Data classes
    "SimilarMessage",
    "ConversationContext",
    "RelationshipProfile",
    # Store
    "EmbeddingStore",
    "get_embedding_store",
    "reset_embedding_store",
    # Convenience functions
    "find_similar_messages",
    "find_similar_situations",
    "get_relationship_profile",
]
