"""Embedding store for JARVIS v2.

SQLite-backed storage for message embeddings with FAISS-indexed similarity search.
Adapted from v1's jarvis/embeddings.py.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .model import get_embedding_model, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Try to import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not installed - using brute force search. Install with: pip install faiss-cpu")

# Configuration
DEFAULT_DB_PATH = Path.home() / ".jarvis" / "embeddings.db"
BATCH_SIZE = 100
MIN_TEXT_LENGTH = 3


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
    your_reply: str | None  # What YOU replied in this situation
    topic: str
    avg_similarity: float


@dataclass
class StyleProfile:
    """Your texting style with a contact."""

    contact_id: str
    display_name: str | None = None
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    common_words: list[str] = field(default_factory=list)
    avg_message_length: float = 0.0
    uses_lowercase: bool = True
    uses_emojis: bool = False
    typical_greetings: list[str] = field(default_factory=list)
    response_examples: list[tuple[str, str]] = field(default_factory=list)  # (their_msg, your_reply)


class EmbeddingStore:
    """SQLite-backed storage for message embeddings with FAISS indexing."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        # FAISS index cache per chat_id
        self._faiss_indices: dict[str, tuple[Any, list[dict]]] = {}  # chat_id -> (index, metadata)
        self._faiss_lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
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

                CREATE INDEX IF NOT EXISTS idx_chat_id ON message_embeddings(chat_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON message_embeddings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_is_from_me ON message_embeddings(is_from_me);
                CREATE INDEX IF NOT EXISTS idx_text_hash ON message_embeddings(text_hash);
            """)
            conn.commit()

    def index_messages(
        self,
        messages: list[dict],
        progress_callback: callable | None = None,
    ) -> dict[str, int]:
        """Index messages for similarity search.

        Args:
            messages: List of message dicts with keys:
                      id, text, chat_id, sender, sender_name, timestamp, is_from_me
            progress_callback: Optional callback(indexed, total)

        Returns:
            Stats dict with indexed, skipped, duplicates counts
        """
        stats = {"indexed": 0, "skipped": 0, "duplicates": 0}

        # Filter valid messages
        valid = [m for m in messages if m.get("text") and len(m["text"].strip()) >= MIN_TEXT_LENGTH]
        stats["skipped"] = len(messages) - len(valid)

        if not valid:
            return stats

        model = get_embedding_model()

        with self._get_connection() as conn:
            # Check for existing
            msg_ids = [m["id"] for m in valid]
            placeholders = ",".join("?" * len(msg_ids))
            existing = set(
                row[0]
                for row in conn.execute(
                    f"SELECT message_id FROM message_embeddings WHERE message_id IN ({placeholders})",
                    msg_ids,
                ).fetchall()
            )

            new_messages = [m for m in valid if m["id"] not in existing]
            stats["duplicates"] = len(valid) - len(new_messages)

            if not new_messages:
                return stats

            # Batch embed
            for i in range(0, len(new_messages), BATCH_SIZE):
                batch = new_messages[i : i + BATCH_SIZE]
                texts = [m["text"] for m in batch]
                embeddings = model.embed_batch(texts)

                for msg, embedding in zip(batch, embeddings):
                    text_hash = hashlib.md5(msg["text"].encode(), usedforsecurity=False).hexdigest()
                    timestamp = msg["timestamp"]
                    if isinstance(timestamp, datetime):
                        timestamp = int(timestamp.timestamp())

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO message_embeddings
                        (message_id, chat_id, embedding, text_hash, sender, sender_name,
                         timestamp, is_from_me, text_preview)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            msg["id"],
                            msg["chat_id"],
                            embedding.tobytes(),
                            text_hash,
                            msg.get("sender"),
                            msg.get("sender_name"),
                            timestamp,
                            1 if msg.get("is_from_me") else 0,
                            msg["text"][:200],
                        ),
                    )
                    stats["indexed"] += 1

                conn.commit()

                if progress_callback:
                    progress_callback(stats["indexed"], len(new_messages))

        return stats

    def _get_or_build_faiss_index(
        self, chat_id: str, only_from_me: bool | None = None
    ) -> tuple[Any, list[dict]] | None:
        """Get or build FAISS index for a chat.

        Returns (faiss_index, metadata_list) or None if FAISS unavailable.
        """
        if not FAISS_AVAILABLE:
            return None

        # Cache key includes the filter
        cache_key = f"{chat_id}:{only_from_me}"

        with self._faiss_lock:
            if cache_key in self._faiss_indices:
                return self._faiss_indices[cache_key]

        # Build index
        start = time.time()
        with self._get_connection() as conn:
            sql = """
                SELECT message_id, chat_id, embedding, text_preview,
                       sender, sender_name, timestamp, is_from_me
                FROM message_embeddings
                WHERE chat_id = ?
            """
            params: list[Any] = [chat_id]

            if only_from_me is not None:
                sql += " AND is_from_me = ?"
                params.append(1 if only_from_me else 0)

            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return None

        # Build numpy array of embeddings
        embeddings = np.array(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows],
            dtype=np.float32
        )

        # Normalize for cosine similarity (FAISS IndexFlatIP does inner product)
        faiss.normalize_L2(embeddings)

        # Create index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)

        # Store metadata for lookup
        metadata = [
            {
                "message_id": row["message_id"],
                "chat_id": row["chat_id"],
                "text": row["text_preview"] or "",
                "sender": row["sender"],
                "sender_name": row["sender_name"],
                "timestamp": row["timestamp"],
                "is_from_me": bool(row["is_from_me"]),
            }
            for row in rows
        ]

        elapsed = (time.time() - start) * 1000
        logger.info(f"Built FAISS index for {chat_id} with {len(rows)} vectors in {elapsed:.0f}ms")

        with self._faiss_lock:
            self._faiss_indices[cache_key] = (index, metadata)

        return index, metadata

    def find_similar(
        self,
        query: str,
        chat_id: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        only_from_me: bool | None = None,
        max_messages: int = 500,  # Only used for brute-force fallback
    ) -> list[SimilarMessage]:
        """Find messages similar to query.

        Uses FAISS index for O(log n) search if available, falls back to brute force.

        Args:
            query: Text to find similar messages for
            chat_id: Optional filter by conversation
            limit: Max results
            min_similarity: Minimum similarity (0-1)
            only_from_me: If True, only your messages. If False, only others. None = all.
            max_messages: Max messages for brute-force fallback

        Returns:
            List of similar messages sorted by similarity
        """
        model = get_embedding_model()
        query_embedding = model.embed(query).astype(np.float32)

        # Try FAISS first (fast path)
        if chat_id and FAISS_AVAILABLE:
            search_start = time.time()
            faiss_result = self._get_or_build_faiss_index(chat_id, only_from_me)
            if faiss_result:
                index, metadata = faiss_result

                # Normalize query for cosine similarity
                query_norm = query_embedding.reshape(1, -1).copy()
                faiss.normalize_L2(query_norm)

                # Search - get more than needed to filter by min_similarity
                k = min(limit * 2, len(metadata))
                similarities, indices = index.search(query_norm, k)

                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or sim < min_similarity:
                        continue
                    meta = metadata[idx]
                    results.append(
                        SimilarMessage(
                            message_id=meta["message_id"],
                            chat_id=meta["chat_id"],
                            text=meta["text"],
                            sender=meta["sender"],
                            sender_name=meta["sender_name"],
                            timestamp=datetime.fromtimestamp(meta["timestamp"]),
                            is_from_me=meta["is_from_me"],
                            similarity=float(sim),
                        )
                    )
                    if len(results) >= limit:
                        break

                search_time = (time.time() - search_start) * 1000
                logger.info(f"FAISS search: {len(results)} results in {search_time:.1f}ms (index size: {len(metadata)})")
                return results

        # Brute force fallback (FAISS not available or no chat_id filter)
        logger.info(f"Using brute-force search (FAISS={'available' if FAISS_AVAILABLE else 'not installed'}, chat_id={chat_id})")
        brute_start = time.time()
        with self._get_connection() as conn:
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

            if only_from_me is not None:
                conditions.append("is_from_me = ?")
                params.append(1 if only_from_me else 0)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(max_messages)

            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Vectorized similarity calculation
        embeddings = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = embeddings_norm @ query_norm

        results = []
        for i, sim in enumerate(similarities):
            if sim >= min_similarity:
                row = rows[i]
                results.append(
                    SimilarMessage(
                        message_id=row["message_id"],
                        chat_id=row["chat_id"],
                        text=row["text_preview"] or "",
                        sender=row["sender"],
                        sender_name=row["sender_name"],
                        timestamp=datetime.fromtimestamp(row["timestamp"]),
                        is_from_me=bool(row["is_from_me"]),
                        similarity=float(sim),
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)
        brute_time = (time.time() - brute_start) * 1000
        logger.info(f"Brute-force search: {len(results[:limit])} results in {brute_time:.1f}ms (searched {len(rows)} messages)")
        return results[:limit]

    def is_index_ready(self, chat_id: str, only_from_me: bool | None = None) -> bool:
        """Check if FAISS index is already built and cached for this chat."""
        if not FAISS_AVAILABLE:
            return False
        cache_key = f"{chat_id}:{only_from_me}"
        return cache_key in self._faiss_indices

    def preload_index(self, chat_id: str) -> None:
        """Pre-build FAISS index for a chat in the background.

        Call this when a conversation is selected to avoid delay on first search.
        """
        import threading

        def _build():
            # Build index for "not from me" (needed for past_replies)
            self._get_or_build_faiss_index(chat_id, only_from_me=False)

        if not self.is_index_ready(chat_id, only_from_me=False):
            thread = threading.Thread(target=_build, daemon=True)
            thread.start()
            logger.info(f"Started background FAISS index build for {chat_id}")

    def find_your_past_replies(
        self,
        incoming_message: str,
        chat_id: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.7,
        skip_if_slow: bool = True,
    ) -> list[tuple[str, str, float]]:
        """Find YOUR past replies to similar incoming messages.

        This is the key function for learning your style!

        Args:
            incoming_message: The message you received
            chat_id: Optional filter by conversation
            limit: Max results
            min_similarity: Minimum similarity threshold
            skip_if_slow: If True, skip lookup if FAISS index isn't cached (avoids 5+ second delay)

        Returns:
            List of (their_message, your_reply, similarity) tuples
        """
        # Skip if index isn't ready and we don't want to wait for it to build
        if skip_if_slow and chat_id and not self.is_index_ready(chat_id, only_from_me=False):
            logger.info(f"Skipping past_replies - FAISS index not cached for {chat_id}")
            return []

        # Find similar messages that were NOT from you
        similar_incoming = self.find_similar(
            query=incoming_message,
            chat_id=chat_id,
            limit=limit * 3,  # Get more to find replies
            min_similarity=min_similarity,
            only_from_me=False,
        )

        results = []

        with self._get_connection() as conn:
            for msg in similar_incoming:
                # Find YOUR next reply after this message
                row = conn.execute(
                    """
                    SELECT text_preview
                    FROM message_embeddings
                    WHERE chat_id = ?
                      AND is_from_me = 1
                      AND timestamp > ?
                      AND timestamp < ? + 3600
                    ORDER BY timestamp
                    LIMIT 1
                    """,
                    (msg.chat_id, int(msg.timestamp.timestamp()), int(msg.timestamp.timestamp())),
                ).fetchone()

                if row and row["text_preview"]:
                    results.append((msg.text, row["text_preview"], msg.similarity))

                if len(results) >= limit:
                    break

        return results

    def get_style_profile(self, chat_id: str) -> StyleProfile:
        """Get your texting style profile for a conversation.

        Args:
            chat_id: Conversation to analyze

        Returns:
            StyleProfile with your patterns
        """
        with self._get_connection() as conn:
            # Basic stats
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received,
                    AVG(CASE WHEN is_from_me = 1 THEN LENGTH(text_preview) END) as avg_length,
                    MIN(sender_name) as display_name
                FROM message_embeddings
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()

            if not row or row["total"] == 0:
                return StyleProfile(contact_id=chat_id)

            # Get YOUR messages for style analysis
            your_messages = conn.execute(
                """
                SELECT text_preview
                FROM message_embeddings
                WHERE chat_id = ? AND is_from_me = 1
                ORDER BY timestamp DESC
                LIMIT 100
                """,
                (chat_id,),
            ).fetchall()

            texts = [r["text_preview"] for r in your_messages if r["text_preview"]]

            # Analyze style
            uses_lowercase = self._detect_lowercase(texts)
            uses_emojis = self._detect_emojis(texts)
            common_words = self._extract_common_words(texts)
            greetings = self._extract_greetings(texts)

            # Get example reply pairs
            examples = self._get_reply_examples(chat_id, conn)

            return StyleProfile(
                contact_id=chat_id,
                display_name=row["display_name"],
                total_messages=row["total"],
                sent_count=row["sent"],
                received_count=row["received"],
                avg_message_length=row["avg_length"] or 0.0,
                uses_lowercase=uses_lowercase,
                uses_emojis=uses_emojis,
                common_words=common_words[:10],
                typical_greetings=greetings[:5],
                response_examples=examples[:5],
            )

    def _detect_lowercase(self, texts: list[str]) -> bool:
        """Detect if user typically uses lowercase."""
        if not texts:
            return True
        lowercase_count = sum(1 for t in texts if t == t.lower())
        return lowercase_count > len(texts) * 0.6

    def _detect_emojis(self, texts: list[str]) -> bool:
        """Detect if user typically uses emojis."""
        import re
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE,
        )
        emoji_count = sum(1 for t in texts if emoji_pattern.search(t))
        return emoji_count > len(texts) * 0.2

    def _extract_common_words(self, texts: list[str]) -> list[str]:
        """Extract commonly used words."""
        stop_words = {
            "the", "a", "an", "is", "it", "to", "and", "or", "of", "in", "on",
            "for", "with", "at", "by", "from", "this", "that", "i", "you", "we",
            "my", "your", "am", "are", "was", "were", "be", "have", "has", "do",
            "does", "did", "will", "would", "could", "should", "just", "so", "but",
        }
        words = []
        for text in texts:
            if text:
                words.extend(
                    w.lower() for w in text.split()
                    if len(w) > 2 and w.lower() not in stop_words
                )
        return [word for word, _ in Counter(words).most_common(20)]

    def _extract_greetings(self, texts: list[str]) -> list[str]:
        """Extract typical greeting patterns."""
        greetings = []
        greeting_starters = ["hey", "hi", "hello", "yo", "sup", "what's up"]
        for text in texts:
            if text:
                lower = text.lower()
                if any(lower.startswith(g) for g in greeting_starters):
                    # Get first few words as greeting pattern
                    greetings.append(" ".join(text.split()[:3]))
        return list(set(greetings))[:5]

    def _get_reply_examples(
        self, chat_id: str, conn: sqlite3.Connection
    ) -> list[tuple[str, str]]:
        """Get example (their_message, your_reply) pairs."""
        rows = conn.execute(
            """
            SELECT m1.text_preview as their_msg, m2.text_preview as your_reply
            FROM message_embeddings m1
            JOIN message_embeddings m2 ON m2.chat_id = m1.chat_id
            WHERE m1.chat_id = ?
              AND m1.is_from_me = 0
              AND m2.is_from_me = 1
              AND m2.timestamp > m1.timestamp
              AND m2.timestamp < m1.timestamp + 300
            ORDER BY m1.timestamp DESC
            LIMIT 20
            """,
            (chat_id,),
        ).fetchall()

        return [(r["their_msg"], r["your_reply"]) for r in rows if r["their_msg"] and r["your_reply"]]

    def get_chat_embeddings(
        self, chat_id: str
    ) -> tuple[np.ndarray, list[SimilarMessage]]:
        """Get all embeddings and messages for a chat.

        Args:
            chat_id: Conversation to get embeddings for

        Returns:
            Tuple of (embeddings array, list of messages)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, chat_id, embedding, text_preview,
                       sender, sender_name, timestamp, is_from_me
                FROM message_embeddings
                WHERE chat_id = ?
                ORDER BY timestamp
                """,
                (chat_id,),
            ).fetchall()

        if not rows:
            return np.array([]), []

        embeddings = []
        messages = []

        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(embedding)

            messages.append(
                SimilarMessage(
                    message_id=row["message_id"],
                    chat_id=row["chat_id"],
                    text=row["text_preview"] or "",
                    sender=row["sender"],
                    sender_name=row["sender_name"],
                    timestamp=datetime.fromtimestamp(row["timestamp"]),
                    is_from_me=bool(row["is_from_me"]),
                    similarity=0.0,
                )
            )

        return np.array(embeddings), messages

    def get_user_response_patterns(
        self,
        chat_id: str | None = None,
        min_replies: int = 3,
    ) -> dict[str, list[str]]:
        """Extract user's common response patterns grouped by intent.

        Analyzes your reply pairs to find consistent response patterns,
        which can be used as personalized templates.

        Args:
            chat_id: Optional filter by conversation (None = all conversations)
            min_replies: Minimum replies per pattern to include

        Returns:
            Dict mapping intent -> list of your actual replies
            e.g., {"affirmative": ["yeah for sure", "sounds good!", "down"]}
        """
        # Intent keywords for simple classification
        intent_patterns = {
            "affirmative": {
                "yes", "yeah", "yep", "yea", "ya", "sure", "ok", "okay", "k",
                "definitely", "absolutely", "for sure", "down", "sounds good",
            },
            "negative": {
                "no", "nah", "nope", "cant", "can't", "cannot", "sorry",
                "not", "won't", "wont", "don't", "dont",
            },
            "greeting": {
                "hey", "hi", "hello", "yo", "sup", "what's up", "whats up",
            },
            "thanks": {
                "thanks", "thank", "thx", "ty", "appreciate",
            },
            "acknowledgment": {
                "got it", "gotcha", "cool", "nice", "alright", "right",
            },
        }

        with self._get_connection() as conn:
            # Get your reply pairs (their message -> your reply within 5 min)
            sql = """
                SELECT m1.text_preview as their_msg, m2.text_preview as your_reply
                FROM message_embeddings m1
                JOIN message_embeddings m2 ON m2.chat_id = m1.chat_id
                WHERE m1.is_from_me = 0
                  AND m2.is_from_me = 1
                  AND m2.timestamp > m1.timestamp
                  AND m2.timestamp < m1.timestamp + 300
            """
            params: list = []

            if chat_id:
                sql += " AND m1.chat_id = ?"
                params.append(chat_id)

            sql += " ORDER BY m1.timestamp DESC LIMIT 500"
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return {}

        # Group replies by intent
        patterns: dict[str, list[str]] = {k: [] for k in intent_patterns}

        for row in rows:
            your_reply = row["your_reply"]
            if not your_reply or len(your_reply) < 2:
                continue

            reply_lower = your_reply.lower().strip()

            # Classify by first matching intent
            for intent, keywords in intent_patterns.items():
                for keyword in keywords:
                    if reply_lower.startswith(keyword) or keyword in reply_lower.split()[:3]:
                        # Store original casing
                        if your_reply not in patterns[intent]:
                            patterns[intent].append(your_reply)
                        break
                else:
                    continue
                break

        # Filter to only intents with enough replies
        return {
            intent: replies[:10]  # Top 10 per intent
            for intent, replies in patterns.items()
            if len(replies) >= min_replies
        }

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
            chats = conn.execute("SELECT COUNT(DISTINCT chat_id) FROM message_embeddings").fetchone()[0]

            return {
                "total_messages": total,
                "unique_conversations": chats,
                "db_path": str(self.db_path),
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            }

    def clear(self) -> None:
        """Clear all embeddings."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM message_embeddings")
            conn.commit()


# Singleton
_store: EmbeddingStore | None = None
_store_lock = threading.Lock()


def get_embedding_store(db_path: Path | str | None = None) -> EmbeddingStore:
    """Get singleton embedding store."""
    global _store

    if _store is None or db_path:
        with _store_lock:
            if _store is None or db_path:
                _store = EmbeddingStore(db_path)

    return _store
