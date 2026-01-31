"""Routing metrics storage for reply routing decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_METRICS_DB_PATH = Path.home() / ".jarvis" / "metrics.db"


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
    """SQLite-backed storage for routing metrics."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_METRICS_DB_PATH
        self._lock = threading.Lock()
        self._initialized = False

    def _initialize(self) -> None:
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

    def record(self, metrics: RoutingMetrics) -> None:
        with self._lock:
            self._initialize()
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
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
                    (
                        metrics.timestamp,
                        metrics.query_hash,
                        metrics.routing_decision,
                        metrics.similarity_score,
                        1 if metrics.cache_hit else 0,
                        1 if metrics.model_loaded else 0,
                        metrics.embedding_computations,
                        metrics.faiss_candidates,
                        json.dumps(metrics.latency_ms, separators=(",", ":")),
                    ),
                )


_store: RoutingMetricsStore | None = None
_store_lock = threading.Lock()


def get_routing_metrics_store() -> RoutingMetricsStore:
    """Get the singleton routing metrics store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RoutingMetricsStore()
    return _store


def reset_routing_metrics_store() -> None:
    """Reset the routing metrics store singleton."""
    global _store
    with _store_lock:
        _store = None


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
    "DEFAULT_METRICS_DB_PATH",
    "RoutingMetrics",
    "RoutingMetricsStore",
    "get_routing_metrics_store",
    "reset_routing_metrics_store",
    "hash_query",
    "load_routing_metrics",
]
