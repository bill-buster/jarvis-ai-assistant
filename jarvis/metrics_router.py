"""Routing metrics storage for reply routing decisions.

Provides buffered writes to SQLite to reduce lock contention and overhead
from per-request database connections.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_METRICS_DB_PATH = Path.home() / ".jarvis" / "metrics.db"

# Buffering configuration
DEFAULT_BUFFER_SIZE = 50  # Flush after this many metrics
DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0  # Flush at least this often


@dataclass
class RoutingMetrics:
    timestamp: float
    query_hash: str
    latency_ms: dict[str, float]
    embedding_computations: int
    faiss_candidates: int
    routing_decision: str
    similarity_score: float
    cache_hit: bool
    model_loaded: bool


def hash_query(text: str) -> str:
    """Hash a query for metrics logging."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return digest


class RoutingMetricsStore:
    """SQLite-backed storage for routing metrics with buffered writes.

    Buffers metrics in memory and flushes to SQLite in batches to reduce
    lock contention and connection overhead. Flushes occur when:
    - Buffer reaches `buffer_size` items
    - `flush_interval_seconds` has elapsed since last flush
    - `flush()` is called explicitly
    - The store is closed or the process exits

    Args:
        db_path: Path to SQLite database. Defaults to ~/.jarvis/metrics.db.
        buffer_size: Number of metrics to buffer before flushing.
        flush_interval_seconds: Maximum time between flushes.
        enable_background_flush: Whether to start a background flush thread.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        enable_background_flush: bool = True,
    ) -> None:
        self._db_path = db_path or DEFAULT_METRICS_DB_PATH
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._lock = threading.Lock()
        self._buffer: list[RoutingMetrics] = []
        self._last_flush_time = time.monotonic()
        self._initialized = False
        self._closed = False

        # Background flush thread
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if enable_background_flush:
            self._start_background_flush()

    def _initialize(self) -> None:
        """Initialize database schema (idempotent)."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    query_hash TEXT NOT NULL,
                    routing_decision TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    model_loaded INTEGER NOT NULL,
                    embedding_computations INTEGER NOT NULL,
                    faiss_candidates INTEGER NOT NULL,
                    latency_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_metrics_timestamp
                ON routing_metrics(timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_metrics_decision
                ON routing_metrics(routing_decision)
                """
            )
        self._initialized = True

    def _start_background_flush(self) -> None:
        """Start background thread for periodic flushing."""
        if self._flush_thread is not None:
            return

        def flush_loop() -> None:
            while not self._stop_event.wait(timeout=self._flush_interval):
                try:
                    self._flush_if_needed()
                except Exception as e:
                    logger.warning(f"Background flush failed: {e}")

        self._flush_thread = threading.Thread(
            target=flush_loop,
            name="metrics-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def _flush_if_needed(self) -> None:
        """Flush buffer if interval has elapsed and buffer is non-empty."""
        with self._lock:
            if not self._buffer:
                return
            elapsed = time.monotonic() - self._last_flush_time
            if elapsed >= self._flush_interval:
                self._flush_buffer_locked()

    def _flush_buffer_locked(self) -> None:
        """Flush buffer to database. Must be called with lock held."""
        if not self._buffer:
            return

        self._initialize()
        metrics_to_write = self._buffer[:]
        self._buffer.clear()
        self._last_flush_time = time.monotonic()

        # Write outside the lock to minimize contention
        # (we've already cleared the buffer, so new records can be added)
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO routing_metrics (
                        timestamp,
                        query_hash,
                        routing_decision,
                        similarity_score,
                        cache_hit,
                        model_loaded,
                        embedding_computations,
                        faiss_candidates,
                        latency_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            m.timestamp,
                            m.query_hash,
                            m.routing_decision,
                            m.similarity_score,
                            1 if m.cache_hit else 0,
                            1 if m.model_loaded else 0,
                            m.embedding_computations,
                            m.faiss_candidates,
                            json.dumps(m.latency_ms, separators=(",", ":")),
                        )
                        for m in metrics_to_write
                    ],
                )
            logger.debug(f"Flushed {len(metrics_to_write)} routing metrics to database")
        except Exception as e:
            # On failure, put metrics back in buffer for retry
            logger.warning(f"Failed to flush metrics: {e}")
            with self._lock:
                self._buffer = metrics_to_write + self._buffer

    def record(self, metrics: RoutingMetrics) -> None:
        """Record a routing metric. Buffers until flush threshold is reached."""
        if self._closed:
            return

        with self._lock:
            self._buffer.append(metrics)
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer_locked()

    def flush(self) -> None:
        """Force flush all buffered metrics to database."""
        with self._lock:
            self._flush_buffer_locked()

    def close(self) -> None:
        """Stop background thread and flush remaining metrics."""
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None

        # Final flush
        self.flush()

    def pending_count(self) -> int:
        """Return number of metrics waiting to be flushed."""
        with self._lock:
            return len(self._buffer)


_store: RoutingMetricsStore | None = None
_store_lock = threading.Lock()
_atexit_registered = False


def _cleanup_store() -> None:
    """Cleanup handler for process exit."""
    global _store
    if _store is not None:
        try:
            _store.close()
        except Exception as e:
            logger.debug(f"Error closing metrics store on exit: {e}")


def get_routing_metrics_store() -> RoutingMetricsStore:
    """Get the singleton routing metrics store.

    The store uses buffered writes for better performance. Metrics are
    flushed automatically when the buffer fills or periodically by a
    background thread.
    """
    global _store, _atexit_registered
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RoutingMetricsStore()
                if not _atexit_registered:
                    atexit.register(_cleanup_store)
                    _atexit_registered = True
    return _store


def reset_routing_metrics_store() -> None:
    """Reset the routing metrics store singleton.

    Closes the existing store (flushing any pending metrics) before resetting.
    """
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


def flush_routing_metrics() -> None:
    """Force flush any pending routing metrics to the database.

    Useful for ensuring metrics are persisted before reading them back
    or before process exit in testing scenarios.
    """
    global _store
    if _store is not None:
        _store.flush()


def load_routing_metrics(
    db_path: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load routing metrics rows for analysis scripts."""
    path = db_path or DEFAULT_METRICS_DB_PATH
    if not path.exists():
        return []

    query = "SELECT * FROM routing_metrics ORDER BY timestamp DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_FLUSH_INTERVAL_SECONDS",
    "DEFAULT_METRICS_DB_PATH",
    "RoutingMetrics",
    "RoutingMetricsStore",
    "flush_routing_metrics",
    "get_routing_metrics_store",
    "hash_query",
    "load_routing_metrics",
    "reset_routing_metrics_store",
]
