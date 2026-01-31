"""Metrics validation and audit layer for routing metrics accuracy verification.

This module provides comprehensive validation to ensure routing metrics
are accurate, complete, and trustworthy.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from jarvis.metrics_router import DEFAULT_METRICS_DB_PATH, RoutingMetrics

DEFAULT_AUDIT_LOG_PATH = Path.home() / ".jarvis" / "metrics_audit.log"


class MetricsAuditLogger:
    """Audit logger that writes every request to a log file for cross-referencing with DB."""

    def __init__(self, log_path: Path | None = None, sampling_rate: float = 1.0) -> None:
        self._log_path = log_path or DEFAULT_AUDIT_LOG_PATH
        self._sampling_rate = sampling_rate
        self._logger = logging.getLogger("jarvis.metrics_audit")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self._logger.addHandler(handler)

    def log_request(self, metrics: RoutingMetrics) -> bool:
        """Log a routing request to audit trail."""
        if random.random() > self._sampling_rate:
            return False

        entry = {
            "timestamp": metrics.timestamp,
            "query_hash": metrics.query_hash,
            "routing_decision": metrics.routing_decision,
            "similarity_score": metrics.similarity_score,
            "latency_ms": metrics.latency_ms,
            "metadata": {
                "cache_hit": metrics.cache_hit,
                "model_loaded": metrics.model_loaded,
                "embedding_computations": metrics.embedding_computations,
                "faiss_candidates": metrics.faiss_candidates,
            },
        }
        self._logger.info(json.dumps(entry))
        return True

    def cross_reference(
        self, db_path: Path | None = None, time_window: float = 60.0
    ) -> dict[str, Any]:
        """Cross-reference audit log with metrics database to find missing records."""
        db_path = db_path or DEFAULT_METRICS_DB_PATH

        audit_entries = self._load_audit_entries()
        if not audit_entries:
            return {"missing_count": 0, "total": 0, "match_rate": 1.0}

        db_entries = self._load_db_entries(db_path)

        missing = []
        matched = 0

        for audit in audit_entries:
            found = False
            for db_entry in db_entries:
                time_diff = abs(audit["timestamp"] - db_entry["timestamp"])
                if time_diff <= time_window and audit["query_hash"] == db_entry["query_hash"]:
                    found = True
                    matched += 1
                    break
            if not found:
                missing.append(audit)

        total = len(audit_entries)
        return {
            "missing_count": len(missing),
            "total": total,
            "match_rate": matched / total if total > 0 else 1.0,
            "missing_details": missing[:5],
        }

    def _load_audit_entries(self) -> list[dict]:
        """Load audit entries from log file."""
        if not self._log_path.exists():
            return []

        entries = []
        with open(self._log_path) as f:
            for line in f:
                try:
                    json_start = line.find(" - ") + 3
                    if json_start > 2:
                        entry = json.loads(line[json_start:])
                        entries.append(entry)
                except (json.JSONDecodeError, IndexError):
                    continue
        return entries

    def _load_db_entries(self, db_path: Path) -> list[dict]:
        """Load entries from metrics database."""
        if not db_path.exists():
            return []

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, query_hash, routing_decision FROM routing_metrics"
            ).fetchall()
            return [dict(row) for row in rows]


class MetricsSamplingValidator:
    """Samples 1% of requests for manual spot-checking."""

    def __init__(self, sampling_rate: float = 0.01) -> None:
        self._sampling_rate = sampling_rate

    def should_sample(self) -> bool:
        return random.random() < self._sampling_rate

    def validate_sample(
        self,
        query: str,
        decision: str,
        similarity: float,
        latency_ms: dict,
        computations: int,
        cache_hit: bool,
        model_loaded: bool,
    ) -> dict[str, Any]:
        """Print sampled request details for manual verification."""
        record = {
            "timestamp": time.time(),
            "query_preview": query[:100] if len(query) > 100 else query,
            "decision": decision,
            "similarity": similarity,
            "latency_ms": latency_ms,
            "computations": computations,
            "cache_hit": cache_hit,
            "model_loaded": model_loaded,
        }

        print("\n" + "=" * 60)
        print("METRICS VALIDATION SAMPLE")
        print("=" * 60)
        print(f"Query: {record['query_preview']}")
        print(f"Decision: {decision} (similarity: {similarity:.3f})")
        print(f"Latency: {latency_ms}")
        print(f"Embeddings computed: {computations}")
        print(f"Cache hit: {cache_hit}, Model loaded: {model_loaded}")
        print("=" * 60 + "\n")

        return record


# Singleton instances
_audit_logger: MetricsAuditLogger | None = None
_sampling_validator: MetricsSamplingValidator | None = None


def get_audit_logger() -> MetricsAuditLogger:
    """Get singleton audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = MetricsAuditLogger()
    return _audit_logger


def get_sampling_validator() -> MetricsSamplingValidator:
    """Get singleton sampling validator instance."""
    global _sampling_validator
    if _sampling_validator is None:
        _sampling_validator = MetricsSamplingValidator()
    return _sampling_validator
