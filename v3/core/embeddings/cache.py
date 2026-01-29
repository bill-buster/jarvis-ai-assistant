"""Embedding cache using SQLite for JARVIS v2.

Content-based caching using SHA-256 hashing for deduplication.
Based on QMD research patterns.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".jarvis" / "cache"
DEFAULT_CACHE_DB = "embeddings.db"

# Singleton
_embedding_cache: EmbeddingCache | None = None
_cache_lock = threading.Lock()


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    hits: int
    misses: int
    hit_rate: float


def content_hash(text: str) -> str:
    """Generate content hash for text.

    Uses SHA-256, returns first 16 characters for compact storage.

    Args:
        text: Text to hash

    Returns:
        16-character hex hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """SQLite-backed embedding cache with content-based deduplication."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory for cache database
            embedding_model: Embedding model to use (lazy-loaded if not provided)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / DEFAULT_CACHE_DB
        self._embedding_model = embedding_model

        # Stats tracking
        self._hits = 0
        self._misses = 0

        # Thread-local connections
        self._local = threading.local()

        # Initialize database
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                text_preview TEXT,
                model_id TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_created_at
            ON embedding_cache(created_at);
        """
        )
        conn.commit()
        logger.debug(f"Embedding cache initialized at {self.db_path}")

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get embedding model (lazy-loaded)."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def get(self, text: str) -> np.ndarray | None:
        """Get cached embedding if exists.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None if not found
        """
        hash_key = content_hash(text)
        conn = self._get_conn()

        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
            (hash_key,),
        ).fetchone()

        if row:
            self._hits += 1
            return np.frombuffer(row["embedding"], dtype=np.float32)

        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Args:
            text: Original text
            embedding: Embedding vector
        """
        hash_key = content_hash(text)
        text_preview = text[:100] if len(text) > 100 else text
        model_id = self.embedding_model.model_id

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache
            (content_hash, embedding, text_preview, model_id)
            VALUES (?, ?, ?, ?)
            """,
            (hash_key, embedding.tobytes(), text_preview, model_id),
        )
        conn.commit()

    def get_or_compute(self, text: str) -> np.ndarray:
        """Get embedding from cache or compute and cache.

        This is the main entry point for getting embeddings.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        cached = self.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self.embedding_model.embed(text)

        # Store in cache
        self.put(text, embedding)

        return embedding

    def get_or_compute_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts, using cache where possible.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results = []
        to_compute = []
        to_compute_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)

        # Batch compute missing embeddings
        if to_compute:
            new_embeddings = self.embedding_model.embed_batch(to_compute)

            # Store in cache and add to results
            for text, embedding, idx in zip(to_compute, new_embeddings, to_compute_indices):
                self.put(text, embedding)
                results.append((idx, embedding))

        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss counts and rate
        """
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as count FROM embedding_cache").fetchone()
        total = row["count"]

        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=total,
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
        )

    def clear(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of entries deleted
        """
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        conn.execute("DELETE FROM embedding_cache")
        conn.commit()

        self._hits = 0
        self._misses = 0

        logger.info(f"Cleared {count} cached embeddings")
        return count

    def prune(self, max_age_days: int = 30) -> int:
        """Remove old cache entries.

        Args:
            max_age_days: Maximum age of entries to keep

        Returns:
            Number of entries deleted
        """
        conn = self._get_conn()
        cutoff = f"-{max_age_days} days"

        result = conn.execute(
            """
            DELETE FROM embedding_cache
            WHERE created_at < strftime('%s', 'now', ?)
            """,
            (cutoff,),
        )
        conn.commit()

        count = result.rowcount
        logger.info(f"Pruned {count} old cache entries (older than {max_age_days} days)")
        return count

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


def get_embedding_cache(cache_dir: Path | str | None = None) -> EmbeddingCache:
    """Get singleton embedding cache.

    Args:
        cache_dir: Directory for cache (uses default if not specified)

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache

    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache(cache_dir)

    return _embedding_cache


def reset_embedding_cache() -> None:
    """Reset the embedding cache singleton."""
    global _embedding_cache
    with _cache_lock:
        if _embedding_cache is not None:
            _embedding_cache.close()
            _embedding_cache = None
