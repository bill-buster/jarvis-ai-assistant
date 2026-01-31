"""Tests for routing metrics storage with buffered writes."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from jarvis.metrics_router import (
    RoutingMetrics,
    RoutingMetricsStore,
    flush_routing_metrics,
    get_routing_metrics_store,
    hash_query,
    load_routing_metrics,
    reset_routing_metrics_store,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_metrics.db"


@pytest.fixture
def store(temp_db: Path) -> RoutingMetricsStore:
    """Create a test store with buffering disabled for deterministic tests."""
    s = RoutingMetricsStore(
        db_path=temp_db,
        buffer_size=10,
        flush_interval_seconds=60.0,  # Long interval - we'll flush manually
        enable_background_flush=False,
    )
    yield s
    s.close()


def make_metric(
    decision: str = "template",
    score: float = 0.95,
    timestamp: float | None = None,
) -> RoutingMetrics:
    """Create a test routing metric."""
    return RoutingMetrics(
        timestamp=time.time() if timestamp is None else timestamp,
        query_hash=hash_query("test query"),
        latency_ms={"total": 10.5, "embedding": 5.0},
        embedding_computations=1,
        faiss_candidates=5,
        routing_decision=decision,
        similarity_score=score,
        cache_hit=True,
        model_loaded=False,
    )


class TestHashQuery:
    """Tests for query hashing."""

    def test_hash_returns_hex_string(self):
        result = hash_query("hello world")
        assert isinstance(result, str)
        assert len(result) == 16  # 8 bytes = 16 hex chars

    def test_same_input_same_hash(self):
        assert hash_query("test") == hash_query("test")

    def test_different_input_different_hash(self):
        assert hash_query("hello") != hash_query("world")


class TestRoutingMetricsStore:
    """Tests for buffered routing metrics store."""

    def test_record_buffers_metrics(self, store: RoutingMetricsStore):
        """Metrics are buffered, not immediately written."""
        store.record(make_metric())
        assert store.pending_count() == 1

    def test_flush_writes_to_database(self, store: RoutingMetricsStore, temp_db: Path):
        """Flush writes buffered metrics to SQLite."""
        store.record(make_metric())
        store.record(make_metric())
        assert store.pending_count() == 2

        store.flush()

        assert store.pending_count() == 0
        # Verify in database
        with sqlite3.connect(temp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM routing_metrics").fetchone()[0]
        assert count == 2

    def test_auto_flush_on_buffer_full(self, temp_db: Path):
        """Buffer flushes automatically when full."""
        store = RoutingMetricsStore(
            db_path=temp_db,
            buffer_size=3,
            enable_background_flush=False,
        )
        try:
            # Add 2 - should stay buffered
            store.record(make_metric())
            store.record(make_metric())
            assert store.pending_count() == 2

            # Add 3rd - should trigger flush
            store.record(make_metric())
            assert store.pending_count() == 0

            with sqlite3.connect(temp_db) as conn:
                count = conn.execute("SELECT COUNT(*) FROM routing_metrics").fetchone()[0]
            assert count == 3
        finally:
            store.close()

    def test_close_flushes_remaining(self, temp_db: Path):
        """Close flushes any remaining buffered metrics."""
        store = RoutingMetricsStore(
            db_path=temp_db,
            buffer_size=100,  # Won't auto-flush
            enable_background_flush=False,
        )
        store.record(make_metric())
        store.record(make_metric())
        assert store.pending_count() == 2

        store.close()

        with sqlite3.connect(temp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM routing_metrics").fetchone()[0]
        assert count == 2

    def test_record_after_close_ignored(self, store: RoutingMetricsStore):
        """Records after close are silently ignored."""
        store.close()
        store.record(make_metric())  # Should not raise
        assert store.pending_count() == 0

    def test_database_schema_created(self, store: RoutingMetricsStore, temp_db: Path):
        """Database schema is created on first flush."""
        store.record(make_metric())
        store.flush()

        with sqlite3.connect(temp_db) as conn:
            # Check table exists
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            assert "routing_metrics" in table_names

            # Check indexes exist
            indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            index_names = [i[0] for i in indexes]
            assert "idx_routing_metrics_timestamp" in index_names
            assert "idx_routing_metrics_decision" in index_names

    def test_metrics_data_preserved(self, store: RoutingMetricsStore, temp_db: Path):
        """All metric fields are correctly stored."""
        metric = RoutingMetrics(
            timestamp=1234567890.123,
            query_hash="abc123",
            latency_ms={"total": 100.5, "embedding": 50.2},
            embedding_computations=3,
            faiss_candidates=10,
            routing_decision="generate",
            similarity_score=0.75,
            cache_hit=False,
            model_loaded=True,
        )
        store.record(metric)
        store.flush()

        with sqlite3.connect(temp_db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM routing_metrics").fetchone()

        assert row["timestamp"] == 1234567890.123
        assert row["query_hash"] == "abc123"
        assert row["routing_decision"] == "generate"
        assert row["similarity_score"] == 0.75
        assert row["cache_hit"] == 0  # False -> 0
        assert row["model_loaded"] == 1  # True -> 1
        assert row["embedding_computations"] == 3
        assert row["faiss_candidates"] == 10
        assert "100.5" in row["latency_json"]

    def test_thread_safety(self, temp_db: Path):
        """Store handles concurrent writes from multiple threads."""
        store = RoutingMetricsStore(
            db_path=temp_db,
            buffer_size=100,
            enable_background_flush=False,
        )
        num_threads = 10
        records_per_thread = 20
        errors: list[Exception] = []

        def writer():
            try:
                for _ in range(records_per_thread):
                    store.record(make_metric())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        store.close()

        assert len(errors) == 0
        with sqlite3.connect(temp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM routing_metrics").fetchone()[0]
        assert count == num_threads * records_per_thread


class TestBackgroundFlush:
    """Tests for background flush thread."""

    def test_background_flush_triggers_on_interval(self, temp_db: Path):
        """Background thread flushes after interval."""
        store = RoutingMetricsStore(
            db_path=temp_db,
            buffer_size=1000,  # High so we don't trigger size-based flush
            flush_interval_seconds=0.1,  # 100ms
            enable_background_flush=True,
        )
        try:
            store.record(make_metric())
            assert store.pending_count() == 1

            # Wait for background flush
            time.sleep(0.3)

            assert store.pending_count() == 0
            with sqlite3.connect(temp_db) as conn:
                count = conn.execute("SELECT COUNT(*) FROM routing_metrics").fetchone()[0]
            assert count == 1
        finally:
            store.close()


class TestSingletonManagement:
    """Tests for singleton store management."""

    def test_get_routing_metrics_store_returns_singleton(self):
        """get_routing_metrics_store returns the same instance."""
        reset_routing_metrics_store()
        try:
            store1 = get_routing_metrics_store()
            store2 = get_routing_metrics_store()
            assert store1 is store2
        finally:
            reset_routing_metrics_store()

    def test_reset_closes_and_clears_store(self):
        """reset_routing_metrics_store closes and clears the singleton."""
        reset_routing_metrics_store()
        store1 = get_routing_metrics_store()
        reset_routing_metrics_store()
        store2 = get_routing_metrics_store()
        assert store1 is not store2

    def test_flush_routing_metrics_flushes_singleton(self, tmp_path: Path, monkeypatch):
        """flush_routing_metrics flushes the singleton store."""
        reset_routing_metrics_store()
        # Use temp db to avoid polluting user's metrics
        monkeypatch.setattr(
            "jarvis.metrics_router.DEFAULT_METRICS_DB_PATH",
            tmp_path / "metrics.db",
        )
        try:
            store = get_routing_metrics_store()
            store.record(make_metric())
            assert store.pending_count() >= 1

            flush_routing_metrics()

            assert store.pending_count() == 0
        finally:
            reset_routing_metrics_store()


class TestLoadRoutingMetrics:
    """Tests for load_routing_metrics function."""

    def test_load_empty_database(self, temp_db: Path):
        """Returns empty list for non-existent database."""
        result = load_routing_metrics(db_path=temp_db)
        assert result == []

    def test_load_returns_metrics(self, temp_db: Path):
        """Returns stored metrics as dicts."""
        store = RoutingMetricsStore(db_path=temp_db, enable_background_flush=False)
        store.record(make_metric(decision="template", score=0.95))
        store.record(make_metric(decision="generate", score=0.75))
        store.close()

        result = load_routing_metrics(db_path=temp_db)

        assert len(result) == 2
        decisions = {r["routing_decision"] for r in result}
        assert decisions == {"template", "generate"}

    def test_load_with_limit(self, temp_db: Path):
        """Respects limit parameter."""
        store = RoutingMetricsStore(db_path=temp_db, enable_background_flush=False)
        for i in range(10):
            store.record(make_metric(timestamp=float(i)))
        store.close()

        result = load_routing_metrics(db_path=temp_db, limit=3)

        assert len(result) == 3
        # Should be ordered by timestamp DESC
        assert result[0]["timestamp"] == 9.0
