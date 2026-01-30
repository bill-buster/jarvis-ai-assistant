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
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from core.utils import STOP_WORDS, MessageDict

from .model import EMBEDDING_DIM, get_embedding_model

logger = logging.getLogger(__name__)

# Try to import FAISS for fast similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info(
        "FAISS not installed - using brute force search. Install with: pip install faiss-cpu"
    )


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
    response_examples: list[tuple[str, str]] = field(
        default_factory=list
    )  # (their_msg, your_reply)


class EmbeddingStore:
    """SQLite-backed storage for message embeddings with FAISS indexing."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else settings.embeddings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        # FAISS index cache per chat_id with LRU eviction (OrderedDict for O(1) operations)
        self._faiss_indices: OrderedDict[str, tuple[Any, list[dict]]] = OrderedDict()
        self._faiss_lock = threading.Lock()

    def _faiss_cache_get(self, cache_key: str) -> tuple[Any, list[dict]] | None:
        """Get from FAISS cache and update LRU order (O(1) with OrderedDict)."""
        with self._faiss_lock:
            if cache_key in self._faiss_indices:
                # Move to end (most recently used) - O(1) operation
                self._faiss_indices.move_to_end(cache_key)
                return self._faiss_indices[cache_key]
        return None

    def _faiss_cache_set(self, cache_key: str, value: tuple[Any, list[dict]]) -> None:
        """Set in FAISS cache with LRU eviction (O(1) with OrderedDict)."""
        with self._faiss_lock:
            # If key exists, update and move to end
            if cache_key in self._faiss_indices:
                self._faiss_indices[cache_key] = value
                self._faiss_indices.move_to_end(cache_key)
                return

            # Evict oldest if at capacity
            while len(self._faiss_indices) >= settings.embeddings.max_faiss_cache_size:
                oldest_key, _ = self._faiss_indices.popitem(last=False)
                logger.debug(f"Evicted FAISS index for {oldest_key} (LRU)")

            # Add new entry at end (most recently used)
            self._faiss_indices[cache_key] = value

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrent read/write performance
            conn.execute("PRAGMA journal_mode=WAL")

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

                -- Composite index for find_your_past_replies queries
                CREATE INDEX IF NOT EXISTS idx_chat_reply_lookup
                ON message_embeddings(chat_id, is_from_me, timestamp);

                -- FTS5 full-text search for hybrid BM25+vector search
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id,
                    chat_id,
                    text_preview,
                    content='message_embeddings',
                    content_rowid='message_id'
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON message_embeddings BEGIN
                    INSERT INTO messages_fts(rowid, message_id, chat_id, text_preview)
                    VALUES (new.message_id, new.message_id, new.chat_id, new.text_preview);
                END;

                CREATE TRIGGER IF NOT EXISTS messages_ad
                    AFTER DELETE ON message_embeddings BEGIN
                    INSERT INTO messages_fts(
                        messages_fts, rowid, message_id, chat_id, text_preview)
                    VALUES (
                        'delete', old.message_id, old.message_id,
                        old.chat_id, old.text_preview);
                END;

                CREATE TRIGGER IF NOT EXISTS messages_au
                    AFTER UPDATE ON message_embeddings BEGIN
                    INSERT INTO messages_fts(
                        messages_fts, rowid, message_id, chat_id, text_preview)
                    VALUES (
                        'delete', old.message_id, old.message_id,
                        old.chat_id, old.text_preview);
                    INSERT INTO messages_fts(rowid, message_id, chat_id, text_preview)
                    VALUES (
                        new.message_id, new.message_id, new.chat_id, new.text_preview);
                END;
            """)
            conn.commit()

        # Rebuild FTS index if needed (one-time migration)
        self._ensure_fts_populated()

    def _ensure_fts_populated(self) -> None:
        """Ensure FTS index is populated (one-time migration for existing data)."""
        with self._get_connection() as conn:
            # Check if FTS has data
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            main_count = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]

            if fts_count == 0 and main_count > 0:
                logger.info(f"Populating FTS index with {main_count} messages...")
                conn.execute("""
                    INSERT INTO messages_fts(rowid, message_id, chat_id, text_preview)
                    SELECT message_id, message_id, chat_id, text_preview
                    FROM message_embeddings
                """)
                conn.commit()
                logger.info("FTS index populated")

    def search_bm25(
        self,
        query: str,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[int, float]]:
        """Search using BM25 full-text search.

        Args:
            query: Search query
            chat_id: Optional filter by conversation
            limit: Max results

        Returns:
            List of (message_id, bm25_score) tuples
        """
        with self._get_connection() as conn:
            # Escape query for FTS5
            safe_query = query.replace('"', '""')

            if chat_id:
                rows = conn.execute(
                    """
                    SELECT message_id, bm25(messages_fts) as score
                    FROM messages_fts
                    WHERE messages_fts MATCH ? AND chat_id = ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (f'"{safe_query}"', chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT message_id, bm25(messages_fts) as score
                    FROM messages_fts
                    WHERE messages_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (f'"{safe_query}"', limit),
                ).fetchall()

            return [(row[0], row[1]) for row in rows]

    def find_similar_hybrid(
        self,
        query: str,
        chat_id: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[SimilarMessage]:
        """Hybrid search combining vector similarity and BM25.

        Uses Reciprocal Rank Fusion (RRF) to combine results.

        Args:
            query: Text to search for
            chat_id: Optional filter by conversation
            limit: Max results
            min_similarity: Minimum similarity threshold
            vector_weight: Weight for vector results in RRF
            bm25_weight: Weight for BM25 results in RRF

        Returns:
            List of similar messages sorted by fused score
        """
        # Get vector results
        vector_results = self.find_similar(
            query=query,
            chat_id=chat_id,
            limit=limit * 2,  # Get more for fusion
            min_similarity=min_similarity,
        )

        # Get BM25 results
        bm25_results = self.search_bm25(
            query=query,
            chat_id=chat_id,
            limit=limit * 2,
        )

        # RRF fusion
        k = 60  # RRF constant
        scores: dict[int, dict] = {}

        # Add vector results
        for rank, msg in enumerate(vector_results):
            rrf_score = vector_weight / (k + rank + 1)
            scores[msg.message_id] = {
                "message": msg,
                "score": rrf_score,
                "best_rank": rank,
            }

        # Add BM25 results
        for rank, (msg_id, _bm25_score) in enumerate(bm25_results):
            rrf_score = bm25_weight / (k + rank + 1)
            if msg_id in scores:
                scores[msg_id]["score"] += rrf_score
                scores[msg_id]["best_rank"] = min(scores[msg_id]["best_rank"], rank)
            else:
                # Need to look up message details
                with self._get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT message_id, chat_id, text_preview, sender,
                               sender_name, timestamp, is_from_me
                        FROM message_embeddings
                        WHERE message_id = ?
                        """,
                        (msg_id,),
                    ).fetchone()

                if row:
                    msg = SimilarMessage(
                        message_id=row["message_id"],
                        chat_id=row["chat_id"],
                        text=row["text_preview"] or "",
                        sender=row["sender"],
                        sender_name=row["sender_name"],
                        timestamp=datetime.fromtimestamp(row["timestamp"]),
                        is_from_me=bool(row["is_from_me"]),
                        similarity=0.0,  # Will be set from fused score
                    )
                    scores[msg_id] = {
                        "message": msg,
                        "score": rrf_score,
                        "best_rank": rank,
                    }

        # Top-rank bonus (from QMD research)
        for msg_id, data in scores.items():
            if data["best_rank"] == 0:
                data["score"] += 0.05
            elif data["best_rank"] <= 2:
                data["score"] += 0.02

        # Sort by fused score
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        # Update similarity scores and return
        results = []
        for item in sorted_results[:limit]:
            msg = item["message"]
            msg.similarity = item["score"]
            results.append(msg)

        return results

    def index_messages(
        self,
        messages: list[MessageDict],
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
        valid = [
            m
            for m in messages
            if m.get("text") and len(m["text"].strip()) >= settings.embeddings.min_text_length
        ]
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
                    "SELECT message_id FROM message_embeddings "
                    f"WHERE message_id IN ({placeholders})",
                    msg_ids,
                ).fetchall()
            )

            new_messages = [m for m in valid if m["id"] not in existing]
            stats["duplicates"] = len(valid) - len(new_messages)

            if not new_messages:
                return stats

            # Batch embed
            for i in range(0, len(new_messages), settings.embeddings.batch_size):
                batch = new_messages[i : i + settings.embeddings.batch_size]
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
                            msg["text"],  # Store full text, not truncated
                        ),
                    )
                    stats["indexed"] += 1

                conn.commit()

                if progress_callback:
                    progress_callback(stats["indexed"], len(new_messages))

        return stats

    def _get_index_cache_path(self, chat_id: str) -> Path:
        """Get path for cached FAISS index."""
        settings.embeddings.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize chat_id for filename
        safe_id = chat_id.replace("/", "_").replace(":", "_")
        return settings.embeddings.faiss_cache_dir / f"{safe_id}.faiss"

    def _get_metadata_cache_path(self, chat_id: str) -> Path:
        """Get path for cached metadata."""
        index_path = self._get_index_cache_path(chat_id)
        return index_path.with_suffix(".meta.json")

    def _get_or_build_faiss_index(self, chat_id: str) -> tuple[Any, list[dict]] | None:
        """Get or build FAISS index for a chat.

        Always builds full index (all messages), filtering is done in memory.
        Uses HNSW for O(log n) search and persists to disk.

        Returns (faiss_index, metadata_list) or None if FAISS unavailable.
        """
        if not FAISS_AVAILABLE:
            return None

        # Cache key is just chat_id (no filter - build full index once)
        cache_key = chat_id

        # Check memory cache first (with LRU tracking)
        cached = self._faiss_cache_get(cache_key)
        if cached is not None:
            return cached

        # Check disk cache
        index_path = self._get_index_cache_path(chat_id)
        meta_path = self._get_metadata_cache_path(chat_id)

        if index_path.exists() and meta_path.exists():
            try:
                start = time.time()
                index = faiss.read_index(str(index_path))

                import json

                with open(meta_path) as f:
                    metadata = json.load(f)

                # Verify count matches
                with self._get_connection() as conn:
                    current_count = conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings WHERE chat_id = ?", [chat_id]
                    ).fetchone()[0]

                if current_count == len(metadata):
                    elapsed = (time.time() - start) * 1000
                    logger.info(
                        f"Loaded FAISS index from cache for {chat_id} "
                        f"({len(metadata)} vectors) in {elapsed:.0f}ms"
                    )

                    self._faiss_cache_set(cache_key, (index, metadata))
                    return index, metadata
                else:
                    logger.info(
                        f"Index stale for {chat_id} "
                        f"(cached={len(metadata)}, current={current_count}), rebuilding"
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding")

        # Build index for ALL messages (no is_from_me filter)
        start = time.time()
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, chat_id, embedding, text_preview,
                       sender, sender_name, timestamp, is_from_me
                FROM message_embeddings
                WHERE chat_id = ?
                """,
                [chat_id],
            ).fetchall()

        if not rows:
            return None

        # Build numpy array of embeddings
        embeddings = np.array(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows], dtype=np.float32
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create HNSW index for O(log n) search
        # For small chats (<1000), use Flat (faster to build)
        # For large chats, use HNSW (faster to search)
        if len(rows) < 1000:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            index = faiss.IndexHNSWFlat(EMBEDDING_DIM, settings.embeddings.hnsw_m)
            index.hnsw.efConstruction = settings.embeddings.hnsw_ef_construction
            index.hnsw.efSearch = settings.embeddings.hnsw_ef_search

        index.add(embeddings)

        # Store metadata for lookup (including is_from_me for filtering)
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

        # Persist to disk
        try:
            import json

            faiss.write_index(index, str(index_path))
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            logger.debug(f"Cached FAISS index to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to cache index: {e}")

        elapsed = (time.time() - start) * 1000
        index_type = "HNSW" if len(rows) >= 1000 else "Flat"
        logger.info(
            f"Built FAISS {index_type} index for {chat_id} "
            f"with {len(rows)} vectors in {elapsed:.0f}ms"
        )

        self._faiss_cache_set(cache_key, (index, metadata))

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
            faiss_result = self._get_or_build_faiss_index(chat_id)
            if faiss_result:
                index, metadata = faiss_result

                # Normalize query for cosine similarity
                search_start = time.time()
                query_norm = query_embedding.reshape(1, -1).copy()
                faiss.normalize_L2(query_norm)

                # Search - get more than needed to filter by min_similarity and only_from_me
                # If filtering by only_from_me, search more to compensate for filtering
                search_k = limit * 4 if only_from_me is not None else limit * 2
                k = min(search_k, len(metadata))
                similarities, indices = index.search(query_norm, k)

                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or sim < min_similarity:
                        continue
                    meta = metadata[idx]

                    # Filter by is_from_me in memory
                    if only_from_me is not None and meta["is_from_me"] != only_from_me:
                        continue

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
                logger.info(
                    f"FAISS search: {len(results)} results in {search_time:.1f}ms "
                    f"(index: {len(metadata)} vectors)"
                )
                return results

        # Brute force fallback (FAISS not available or no chat_id filter)
        faiss_status = "available" if FAISS_AVAILABLE else "not installed"
        logger.info(f"Using brute-force search (FAISS={faiss_status}, chat_id={chat_id})")
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
        logger.info(
            f"Brute-force search: {len(results[:limit])} results in {brute_time:.1f}ms "
            f"(searched {len(rows)} messages)"
        )
        return results[:limit]

    def is_index_ready(self, chat_id: str, only_from_me: bool = False) -> bool:
        """Check if FAISS index is already built and cached for this chat.

        Args:
            chat_id: Conversation ID
            only_from_me: Ignored for now (we use a single index per chat)
        """
        if not FAISS_AVAILABLE:
            return False
        return chat_id in self._faiss_indices

    def preload_index(self, chat_id: str) -> None:
        """Pre-build FAISS index for a chat in the background.

        Call this when a conversation is selected to avoid delay on first search.
        """
        import threading

        def _build():
            # Build full index (all messages)
            self._get_or_build_faiss_index(chat_id)

        if not self.is_index_ready(chat_id):
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
        # Time-weighting parameters
        use_time_weighting: bool | None = None,
        recency_weight: float | None = None,
        time_window_boost: float | None = None,
        day_type_boost: float | None = None,
        max_age_days: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Find YOUR past replies to similar incoming messages.

        This is the key function for learning your style!

        Args:
            incoming_message: The message you received
            chat_id: Optional filter by conversation
            limit: Max results
            min_similarity: Minimum similarity threshold
            skip_if_slow: If True, skip lookup if FAISS index isn't cached (avoids 5+ second delay)
            use_time_weighting: If True, apply recency and time-of-day adjustments
            recency_weight: Weight for recency vs similarity (0-1)
            time_window_boost: Bonus for same time-of-day window
            day_type_boost: Bonus for same day type (weekday/weekend)
            max_age_days: Max age for recency calculation

        Returns:
            List of (their_message, your_reply, score) tuples
        """
        # Load defaults from settings if not provided
        if use_time_weighting is None:
            use_time_weighting = settings.embeddings.use_time_weighting
        if recency_weight is None:
            recency_weight = settings.embeddings.recency_weight
        if time_window_boost is None:
            time_window_boost = settings.embeddings.time_window_boost
        if day_type_boost is None:
            day_type_boost = settings.embeddings.day_type_boost
        if max_age_days is None:
            max_age_days = settings.embeddings.max_age_days

        # Skip if index isn't ready and we don't want to wait for it to build
        if skip_if_slow:
            if chat_id is None:
                # Global search uses brute-force which is slow - skip
                logger.info("Skipping past_replies - global search would use slow brute-force")
                return []
            if not self.is_index_ready(chat_id):
                logger.info(f"Skipping past_replies - FAISS index not cached for {chat_id}")
                return []

        # Find similar messages that were NOT from you
        similar_incoming = self.find_similar(
            query=incoming_message,
            chat_id=chat_id,
            limit=limit * 3,  # Get more to find replies
            min_similarity=min_similarity,
            only_from_me=False,
            max_messages=200,  # Limit brute-force fallback
        )

        now = datetime.now()
        current_hour = now.hour
        current_dow = now.weekday()  # 0=Monday, 6=Sunday
        is_weekend = current_dow >= 5

        def compute_time_weighted_score(semantic_sim: float, msg_timestamp: datetime) -> float:
            """Apply time-based adjustments to similarity score."""
            if not use_time_weighting:
                return semantic_sim

            # Recency factor (0-1, newer = higher)
            age_days = (now - msg_timestamp).days
            recency_factor = max(0, 1.0 - (age_days / max_age_days))

            # Time-of-day boost (within 3-hour window)
            msg_hour = msg_timestamp.hour
            hour_diff = min(abs(current_hour - msg_hour), 24 - abs(current_hour - msg_hour))
            time_boost = time_window_boost if hour_diff <= 3 else 0

            # Weekend/weekday boost
            msg_dow = msg_timestamp.weekday()
            msg_is_weekend = msg_dow >= 5
            day_boost = day_type_boost if is_weekend == msg_is_weekend else 0

            # Combined score
            return (
                semantic_sim * (1 - recency_weight)
                + recency_factor * recency_weight
                + time_boost
                + day_boost
            )

        if not similar_incoming:
            return []

        # Batch lookup: group messages by chat_id
        lookups: dict[str, list[tuple[int, str, float, datetime]]] = {}
        for msg in similar_incoming:
            if msg.chat_id not in lookups:
                lookups[msg.chat_id] = []
            lookups[msg.chat_id].append(
                (
                    int(msg.timestamp.timestamp()),
                    msg.text,
                    msg.similarity,
                    msg.timestamp,
                )
            )

        results = []

        with self._get_connection() as conn:
            for chat_id_key, items in lookups.items():
                min_ts = min(ts for ts, _, _, _ in items)

                # Get ALL messages in this chat (both sides) to find consecutive pairs
                rows = conn.execute(
                    """
                    SELECT timestamp, text_preview, is_from_me
                    FROM message_embeddings
                    WHERE chat_id = ? AND timestamp >= ?
                    ORDER BY timestamp
                    """,
                    (chat_id_key, min_ts),
                ).fetchall()

                # Build reply pairs: find your reply after their message(s)
                # Your reply must come within 2 hours of their last message
                # and there can't be big time gaps (>30 min) between their messages
                MAX_REPLY_GAP = 7200  # 2 hours max between their last msg and your reply
                MAX_THEIR_GAP = 1800  # 30 minutes max between their consecutive messages

                reply_pairs = {}  # timestamp -> (context, your_reply)

                i = 0
                while i < len(rows):
                    # Find a sequence of their messages (with no big time gaps)
                    their_messages = []
                    while i < len(rows) and not rows[i]["is_from_me"]:
                        if rows[i]["text_preview"]:
                            curr_ts = rows[i]["timestamp"]
                            # Check for time gap from previous message
                            if their_messages and (curr_ts - their_messages[-1][0]) > MAX_THEIR_GAP:
                                # Big gap - start fresh sequence
                                their_messages = []
                            their_messages.append((curr_ts, rows[i]["text_preview"]))
                        i += 1

                    # Check if followed by your reply within time limit
                    if their_messages and i < len(rows) and rows[i]["is_from_me"]:
                        your_reply = rows[i]["text_preview"]
                        your_ts = rows[i]["timestamp"]
                        last_their_ts = their_messages[-1][0]

                        # Only pair if your reply is within 30 min of their last message
                        if your_reply and (your_ts - last_their_ts) <= MAX_REPLY_GAP:
                            # Combine all their messages as context
                            combined_context = " | ".join(text for _, text in their_messages)
                            # Index under each of their message timestamps
                            for ts, text in their_messages:
                                reply_pairs[ts] = (combined_context, your_reply)
                    i += 1

                # Match our similar messages to these pairs
                for ts, their_text, sim, msg_timestamp in items:
                    if ts in reply_pairs:
                        context, your_reply = reply_pairs[ts]
                        score = compute_time_weighted_score(sim, msg_timestamp)
                        results.append((context, your_reply, score))

        # Re-sort by time-weighted score and limit
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

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
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map
            "\U0001f1e0-\U0001f1ff"  # flags
            "]+",
            flags=re.UNICODE,
        )
        emoji_count = sum(1 for t in texts if emoji_pattern.search(t))
        return emoji_count > len(texts) * 0.2

    def _extract_common_words(self, texts: list[str]) -> list[str]:
        """Extract commonly used words."""
        words = []
        for text in texts:
            if text:
                words.extend(
                    w.lower() for w in text.split() if len(w) > 2 and w.lower() not in STOP_WORDS
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

    def _get_reply_examples(self, chat_id: str, conn: sqlite3.Connection) -> list[tuple[str, str]]:
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

        return [
            (r["their_msg"], r["your_reply"]) for r in rows if r["their_msg"] and r["your_reply"]
        ]

    def get_chat_embeddings(self, chat_id: str) -> tuple[np.ndarray, list[SimilarMessage]]:
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
                "yes",
                "yeah",
                "yep",
                "yea",
                "ya",
                "sure",
                "ok",
                "okay",
                "k",
                "definitely",
                "absolutely",
                "for sure",
                "down",
                "sounds good",
            },
            "negative": {
                "no",
                "nah",
                "nope",
                "cant",
                "can't",
                "cannot",
                "sorry",
                "not",
                "won't",
                "wont",
                "don't",
                "dont",
            },
            "greeting": {
                "hey",
                "hi",
                "hello",
                "yo",
                "sup",
                "what's up",
                "whats up",
            },
            "thanks": {
                "thanks",
                "thank",
                "thx",
                "ty",
                "appreciate",
            },
            "acknowledgment": {
                "got it",
                "gotcha",
                "cool",
                "nice",
                "alright",
                "right",
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

    def _get_phone_to_chatids_cache_path(self) -> Path:
        """Get path for cached phone-to-chat_ids mapping."""
        settings.embeddings.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        return settings.embeddings.faiss_cache_dir / "phone_to_chatids.json"

    def _get_or_build_phone_to_chatids(self) -> dict[str, list[str]]:
        """Get or build mapping of phone numbers to actual chat_ids.

        Queries the database once to find all chat_ids where each phone
        appears as a sender. Cached to disk for fast startup.

        Returns:
            Dict mapping normalized phone -> list of chat_ids
        """
        cache_key = "__phone_to_chatids__"

        # Check memory cache
        cached = self._faiss_cache_get(cache_key)
        if cached is not None:
            return cached[0]  # Returns (mapping, None) tuple

        # Check disk cache
        cache_path = self._get_phone_to_chatids_cache_path()
        if cache_path.exists():
            try:
                import json

                with open(cache_path) as f:
                    mapping = json.load(f)

                # Verify it's not stale (check message count)
                with self._get_connection() as conn:
                    current_count = conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings"
                    ).fetchone()[0]

                cached_count = mapping.get("__message_count__", 0)
                if current_count == cached_count:
                    del mapping["__message_count__"]
                    logger.info(
                        f"Loaded phone-to-chatids mapping from cache ({len(mapping)} phones)"
                    )
                    self._faiss_cache_set(cache_key, (mapping, None))
                    return mapping
                else:
                    logger.info("Phone-to-chatids cache stale, rebuilding")
            except Exception as e:
                logger.warning(f"Failed to load phone-to-chatids cache: {e}")

        # Build mapping from database
        logger.info("Building phone-to-chatids mapping...")
        start = time.time()

        mapping: dict[str, list[str]] = {}

        with self._get_connection() as conn:
            # Get all unique (sender, chat_id) pairs where sender looks like a phone
            rows = conn.execute(
                """
                SELECT DISTINCT sender, chat_id
                FROM message_embeddings
                WHERE sender LIKE '+%'
                  AND is_from_me = 0
                """
            ).fetchall()

            for row in rows:
                phone = row["sender"]
                chat_id = row["chat_id"]

                # Normalize phone (keep only + and digits)
                import re

                normalized = "+" + re.sub(r"\D", "", phone[1:]) if phone.startswith("+") else phone

                if normalized not in mapping:
                    mapping[normalized] = []
                if chat_id not in mapping[normalized]:
                    mapping[normalized].append(chat_id)

            # Store message count for staleness check
            message_count = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]

        # Persist to disk
        try:
            import json

            cache_data = dict(mapping)
            cache_data["__message_count__"] = message_count
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Cached phone-to-chatids mapping to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache phone-to-chatids: {e}")

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"Built phone-to-chatids mapping: {len(mapping)} phones, "
            f"{sum(len(v) for v in mapping.values())} chat_ids in {elapsed:.0f}ms"
        )

        self._faiss_cache_set(cache_key, (mapping, None))
        return mapping

    def resolve_phones_to_chatids(self, phones: list[str]) -> list[str]:
        """Resolve phone numbers to actual chat_ids from the database.

        Args:
            phones: List of phone numbers (e.g., ["+15551234567", "+15559876543"])

        Returns:
            List of actual chat_ids found in the database
        """
        mapping = self._get_or_build_phone_to_chatids()

        chat_ids = set()
        for phone in phones:
            # Normalize phone
            import re

            if phone.startswith("+"):
                normalized = "+" + re.sub(r"\D", "", phone[1:])
            else:
                normalized = phone

            # Look up in mapping
            if normalized in mapping:
                chat_ids.update(mapping[normalized])

        return list(chat_ids)

    def _get_global_index_cache_path(self) -> Path:
        """Get path for cached global FAISS index."""
        settings.embeddings.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        return settings.embeddings.faiss_cache_dir / "global_index.faiss"

    def _get_global_metadata_cache_path(self) -> Path:
        """Get path for cached global metadata."""
        return self._get_global_index_cache_path().with_suffix(".meta.json")

    def _get_or_build_global_faiss_index(self) -> tuple[Any, list[dict]] | None:
        """Get or build global FAISS index for ALL conversations.

        This index contains all messages from all conversations, enabling
        cross-conversation search for relationship-aware RAG.

        Uses HNSW for O(log n) search on ~337K vectors.

        Returns:
            (faiss_index, metadata_list) or None if FAISS unavailable
        """
        if not FAISS_AVAILABLE:
            return None

        cache_key = "__global__"

        # Check memory cache first
        cached = self._faiss_cache_get(cache_key)
        if cached is not None:
            return cached

        # Check disk cache
        index_path = self._get_global_index_cache_path()
        meta_path = self._get_global_metadata_cache_path()

        if index_path.exists() and meta_path.exists():
            try:
                start = time.time()
                index = faiss.read_index(str(index_path))

                import json

                with open(meta_path) as f:
                    metadata = json.load(f)

                # Verify count matches
                with self._get_connection() as conn:
                    current_count = conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings"
                    ).fetchone()[0]

                if current_count == len(metadata):
                    elapsed = (time.time() - start) * 1000
                    logger.info(
                        f"Loaded global FAISS index from cache "
                        f"({len(metadata)} vectors) in {elapsed:.0f}ms"
                    )

                    self._faiss_cache_set(cache_key, (index, metadata))
                    return index, metadata
                else:
                    logger.info(
                        f"Global index stale (cached={len(metadata)}, "
                        f"current={current_count}), rebuilding"
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached global index: {e}, rebuilding")

        # Build index for ALL messages
        start = time.time()
        logger.info("Building global FAISS index (this may take a minute)...")

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, chat_id, embedding, text_preview,
                       sender, sender_name, timestamp, is_from_me
                FROM message_embeddings
                """
            ).fetchall()

        if not rows:
            logger.warning("No messages found for global index")
            return None

        # Build numpy array of embeddings
        embeddings = np.array(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows], dtype=np.float32
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Always use HNSW for global index (many vectors)
        index = faiss.IndexHNSWFlat(EMBEDDING_DIM, settings.embeddings.hnsw_m)
        index.hnsw.efConstruction = settings.embeddings.hnsw_ef_construction
        index.hnsw.efSearch = settings.embeddings.hnsw_ef_search
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

        # Persist to disk
        try:
            import json

            faiss.write_index(index, str(index_path))
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            logger.debug(f"Cached global FAISS index to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to cache global index: {e}")

        elapsed = (time.time() - start) * 1000
        logger.info(f"Built global FAISS HNSW index with {len(rows)} vectors in {elapsed:.0f}ms")

        self._faiss_cache_set(cache_key, (index, metadata))
        return index, metadata

    def _get_reply_pairs_index_path(self) -> Path:
        """Get path for cached reply-pairs FAISS index."""
        settings.embeddings.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        return settings.embeddings.faiss_cache_dir / "reply_pairs_index.faiss"

    def _get_reply_pairs_metadata_path(self) -> Path:
        """Get path for cached reply-pairs metadata."""
        return self._get_reply_pairs_index_path().with_suffix(".meta.json")

    def _get_reply_pairs_state_path(self) -> Path:
        """Get path for reply-pairs index state (for incremental updates)."""
        return self._get_reply_pairs_index_path().with_suffix(".state.json")

    def _is_reply_pairs_index_stale(self) -> bool:
        """Check if reply-pairs index needs rebuilding.

        Returns True if:
        - Index doesn't exist
        - New messages have been added since last build
        """
        state_path = self._get_reply_pairs_state_path()
        if not state_path.exists():
            return True

        try:
            import json
            with open(state_path) as f:
                state = json.load(f)

            last_indexed_id = state.get("last_message_id", 0)

            # Check if new messages exist
            with self._get_connection() as conn:
                max_id = conn.execute(
                    "SELECT MAX(message_id) FROM message_embeddings"
                ).fetchone()[0] or 0

            return max_id > last_indexed_id
        except Exception:
            return True

    def _get_or_build_reply_pairs_index(
        self, force_rebuild: bool = False
    ) -> tuple[Any, list[dict]] | None:
        """Get or build FAISS index of (incoming_message -> your_reply) pairs.

        This is the FAST index for cross-conversation search:
        - Only indexes incoming messages that YOU replied to
        - Stores your reply directly in metadata (no DB lookup needed)
        - Deduplicates by (their_text, your_reply) to reduce size
        - Tracks last_message_id for incremental rebuilds

        Args:
            force_rebuild: If True, rebuild even if cached index exists

        Returns:
            (faiss_index, metadata_list) or None if FAISS unavailable
        """
        if not FAISS_AVAILABLE:
            return None

        cache_key = "__reply_pairs__"

        # Check memory cache (unless forcing rebuild)
        if not force_rebuild:
            cached = self._faiss_cache_get(cache_key)
            if cached is not None:
                # Check if stale
                if not self._is_reply_pairs_index_stale():
                    return cached

        # Check disk cache
        index_path = self._get_reply_pairs_index_path()
        meta_path = self._get_reply_pairs_metadata_path()

        if not force_rebuild and index_path.exists() and meta_path.exists():
            # Check staleness before loading
            if not self._is_reply_pairs_index_stale():
                try:
                    start = time.time()
                    index = faiss.read_index(str(index_path))

                    import json
                    with open(meta_path) as f:
                        metadata = json.load(f)

                    elapsed = (time.time() - start) * 1000
                    logger.info(
                        f"Loaded reply-pairs index from cache "
                        f"({len(metadata)} pairs) in {elapsed:.0f}ms"
                    )

                    self._faiss_cache_set(cache_key, (index, metadata))
                    return index, metadata

                except Exception as e:
                    logger.warning(f"Failed to load reply-pairs index: {e}, rebuilding")

        # Build index of reply pairs
        start = time.time()
        logger.info("Building reply-pairs index...")

        # Query: find all (their_message, your_reply) pairs
        # - Your reply must be within 5 minutes after their message
        # - Deduplicate by grouping on (their_text, your_reply)
        # - Keep most recent occurrence of each unique pair
        with self._get_connection() as conn:
            # Get max message_id for state tracking
            max_id = conn.execute(
                "SELECT MAX(message_id) FROM message_embeddings"
            ).fetchone()[0] or 0

            # Query with smart filtering to reduce index size:
            # 1. Skip reactions ("Loved an image", "Laughed at")
            # 2. Require substantive replies (>=5 chars, has letters)
            # 3. Skip common filler replies (lol, haha, yeah, etc.)
            # 4. Limit to 20 examples per unique reply (diversity vs size tradeoff)
            # 5. Keep most recent examples (recency matters)
            #
            # This reduces ~560K pairs to ~50-100K quality pairs
            rows = conn.execute(
                """
                WITH filtered_pairs AS (
                    SELECT
                        m1.message_id as their_msg_id,
                        m1.chat_id,
                        m1.embedding as their_embedding,
                        m1.text_preview as their_text,
                        m1.sender,
                        m1.sender_name,
                        m1.timestamp as their_ts,
                        m2.text_preview as your_reply,
                        ROW_NUMBER() OVER (
                            PARTITION BY LOWER(TRIM(m2.text_preview))
                            ORDER BY m1.timestamp DESC
                        ) as reply_rank
                    FROM message_embeddings m1
                    JOIN message_embeddings m2 ON m2.chat_id = m1.chat_id
                    WHERE m1.is_from_me = 0
                      AND m2.is_from_me = 1
                      AND m2.timestamp > m1.timestamp
                      AND m2.timestamp < m1.timestamp + 300
                      AND m1.text_preview IS NOT NULL
                      AND m2.text_preview IS NOT NULL
                      -- Their message should be meaningful
                      AND LENGTH(m1.text_preview) >= 5
                      -- Your reply should be substantive (>=5 chars)
                      AND LENGTH(TRIM(m2.text_preview)) >= 5
                      -- Skip reactions (tapbacks)
                      AND LOWER(m2.text_preview) NOT LIKE 'loved %'
                      AND LOWER(m2.text_preview) NOT LIKE 'liked %'
                      AND LOWER(m2.text_preview) NOT LIKE 'laughed at%'
                      AND LOWER(m2.text_preview) NOT LIKE 'emphasized%'
                      AND LOWER(m2.text_preview) NOT LIKE 'questioned%'
                      AND LOWER(m2.text_preview) NOT LIKE 'disliked%'
                      -- Skip common low-info filler (keep in per-chat, not global)
                      AND LOWER(TRIM(m2.text_preview)) NOT IN (
                          'lol', 'haha', 'hahaha', 'hahahaha', 'lmao', 'lmfao',
                          'yeah', 'yea', 'yes', 'yep', 'yup', 'ya',
                          'no', 'nah', 'nope',
                          'ok', 'okay', 'k', 'kk',
                          'true', 'same', 'facts', 'fr', 'frfr',
                          'nice', 'cool', 'damn', 'bruh', 'bro',
                          'what', 'huh', 'hmm', 'idk', 'omg', 'wow'
                      )
                )
                SELECT their_msg_id, chat_id, their_embedding, their_text,
                       sender, sender_name, their_ts, your_reply
                FROM filtered_pairs
                WHERE reply_rank <= 20  -- Max 20 examples per unique reply
                ORDER BY their_ts DESC
                """
            ).fetchall()

        if not rows:
            logger.warning("No reply pairs found for index")
            return None

        logger.info(f"Found {len(rows)} unique reply pairs (deduplicated)")

        # Build numpy array of embeddings (only their messages)
        embeddings = np.array(
            [np.frombuffer(row["their_embedding"], dtype=np.float32) for row in rows],
            dtype=np.float32,
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Use HNSW for fast search (or Flat for small indices)
        if len(rows) < 1000:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            index = faiss.IndexHNSWFlat(EMBEDDING_DIM, settings.embeddings.hnsw_m)
            index.hnsw.efConstruction = settings.embeddings.hnsw_ef_construction
            index.hnsw.efSearch = settings.embeddings.hnsw_ef_search

        index.add(embeddings)

        # Store metadata WITH your reply (no DB lookup needed during search!)
        metadata = [
            {
                "their_text": row["their_text"],
                "your_reply": row["your_reply"],
                "chat_id": row["chat_id"],
                "sender": row["sender"],
                "sender_name": row["sender_name"],
                "timestamp": row["their_ts"],
            }
            for row in rows
        ]

        # Persist to disk
        try:
            import json

            faiss.write_index(index, str(index_path))
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            # Save state for incremental updates
            state_path = self._get_reply_pairs_state_path()
            with open(state_path, "w") as f:
                json.dump({"last_message_id": max_id, "pair_count": len(metadata)}, f)

            logger.debug(f"Cached reply-pairs index to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to cache reply-pairs index: {e}")

        elapsed = (time.time() - start) * 1000
        logger.info(f"Built reply-pairs HNSW index with {len(rows)} pairs in {elapsed:.0f}ms")

        self._faiss_cache_set(cache_key, (index, metadata))
        return index, metadata

    def find_your_past_replies_cross_conversation(
        self,
        incoming_message: str,
        target_chat_ids: list[str] | None = None,
        limit: int = 5,
        min_similarity: float = 0.55,
    ) -> list[tuple[str, str, float, str]]:
        """Find YOUR past replies across multiple conversations.

        FAST VERSION: Uses pre-computed reply-pairs index.
        No DB queries during search - all data in FAISS metadata.

        Args:
            incoming_message: The message you received
            target_chat_ids: List of chat_ids to search in (e.g., all "friend" chat_ids).
                           If None, searches ALL conversations.
            limit: Max results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (their_message, your_reply, score, chat_id) tuples
        """
        # Use reply-pairs index (FAST - no DB queries)
        faiss_result = self._get_or_build_reply_pairs_index()
        if not faiss_result:
            logger.info("Reply-pairs index not available")
            return []

        index, metadata = faiss_result

        # Embed query
        from .model import get_embedding_model

        model = get_embedding_model()
        query_embedding = model.embed(incoming_message).astype(np.float32)

        # Normalize query for cosine similarity
        search_start = time.time()
        query_norm = query_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query_norm)

        # Build chat_id filter set if provided
        target_set = set(target_chat_ids) if target_chat_ids else None

        # Search - get more than needed for filtering
        k = min(limit * 10, len(metadata))
        similarities, indices = index.search(query_norm, k)

        # Collect results (no DB query needed - reply is in metadata!)
        results = []
        seen_replies = set()  # Deduplicate by reply text

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or sim < min_similarity:
                continue

            meta = metadata[idx]

            # Filter by chat_id if specified
            if target_set and meta["chat_id"] not in target_set:
                continue

            # Deduplicate by reply text
            reply_key = meta["your_reply"].lower().strip()
            if reply_key in seen_replies:
                continue
            seen_replies.add(reply_key)

            results.append(
                (
                    meta["their_text"],
                    meta["your_reply"],
                    float(sim),
                    meta["chat_id"],
                )
            )

            if len(results) >= limit:
                break

        search_time = (time.time() - search_start) * 1000
        logger.info(
            f"Reply-pairs search: {len(results)} results in {search_time:.1f}ms "
            f"(index: {len(metadata)} pairs)"
        )

        return results

    def is_global_index_ready(self) -> bool:
        """Check if global FAISS index is already built and cached.

        Returns:
            True if global index is in memory or on disk
        """
        if not FAISS_AVAILABLE:
            return False

        # Check memory cache
        if "__global__" in self._faiss_indices:
            return True

        # Check disk cache
        index_path = self._get_global_index_cache_path()
        meta_path = self._get_global_metadata_cache_path()
        return index_path.exists() and meta_path.exists()

    def is_reply_pairs_index_ready(self) -> bool:
        """Check if reply-pairs FAISS index is already built and cached.

        This index is used for fast cross-conversation search.

        Returns:
            True if reply-pairs index is in memory or on disk
        """
        if not FAISS_AVAILABLE:
            return False

        # Check memory cache
        if "__reply_pairs__" in self._faiss_indices:
            return True

        # Check disk cache
        index_path = self._get_reply_pairs_index_path()
        meta_path = self._get_reply_pairs_metadata_path()
        return index_path.exists() and meta_path.exists()

    # TODO: Remove if unused - not currently called anywhere
    def preload_global_index(self) -> None:
        """Pre-build global FAISS index in the background.

        Call this during startup to avoid delay on first cross-conversation search.
        """
        import threading

        def _build():
            self._get_or_build_global_faiss_index()

        if not self.is_global_index_ready():
            thread = threading.Thread(target=_build, daemon=True)
            thread.start()
            logger.info("Started background global FAISS index build")

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
            chats = conn.execute(
                "SELECT COUNT(DISTINCT chat_id) FROM message_embeddings"
            ).fetchone()[0]

            return {
                "total_messages": total,
                "unique_conversations": chats,
                "db_path": str(self.db_path),
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024)
                if self.db_path.exists()
                else 0,
                "global_index_ready": self.is_global_index_ready(),
            }

    def clear(self) -> None:
        """Clear all embeddings."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM message_embeddings")
            conn.commit()


# Singleton using lru_cache for thread safety
_store_override: EmbeddingStore | None = None


@lru_cache(maxsize=1)
def _create_default_store() -> EmbeddingStore:
    """Create the default embedding store (thread-safe via lru_cache)."""
    return EmbeddingStore()


def get_embedding_store(db_path: Path | str | None = None) -> EmbeddingStore:
    """Get singleton embedding store.

    Args:
        db_path: Optional path override. If provided, resets the singleton to use this path.

    Returns:
        EmbeddingStore instance
    """
    global _store_override

    if db_path is not None:
        # Override requested - create new store with specific path
        _store_override = EmbeddingStore(db_path)
        return _store_override

    # Return override if set, otherwise use cached default
    if _store_override is not None:
        return _store_override

    return _create_default_store()


def reset_embedding_store() -> None:
    """Reset the embedding store singleton."""
    global _store_override
    _store_override = None
    _create_default_store.cache_clear()
