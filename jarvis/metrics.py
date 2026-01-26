"""Metrics collection for JARVIS performance monitoring.

Provides:
- Memory sampling utility for tracking memory usage over time
- Request counter for tracking API endpoint usage
- Latency histogram for tracking request/operation latencies

Thread-safe implementations suitable for concurrent access.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil

# Constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3

# Default histogram buckets for latency (in seconds)
DEFAULT_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)


@dataclass
class MemorySample:
    """A single memory usage sample."""

    timestamp: datetime
    rss_mb: float  # Resident Set Size (actual RAM usage)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of system memory
    available_gb: float  # System available memory


@dataclass
class LatencyStats:
    """Statistics for a latency histogram bucket."""

    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = 0.0

    @property
    def mean(self) -> float:
        """Calculate mean latency."""
        return self.total / self.count if self.count > 0 else 0.0


@dataclass
class HistogramData:
    """Data for a latency histogram."""

    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total_count: int = 0
    total_sum: float = 0.0
    min_value: float = float("inf")
    max_value: float = 0.0

    def __post_init__(self) -> None:
        """Initialize bucket counts."""
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        """Record a value in the histogram."""
        self.total_count += 1
        self.total_sum += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        # Increment all buckets where value <= bucket boundary
        for i, boundary in enumerate(self.buckets):
            if value <= boundary:
                self.counts[i] += 1

    def percentile(self, p: float) -> float:
        """Estimate percentile from histogram buckets.

        Args:
            p: Percentile (0-100)

        Returns:
            Estimated value at the given percentile
        """
        if self.total_count == 0:
            return 0.0

        threshold = (p / 100.0) * self.total_count

        for i, count in enumerate(self.counts):
            if count >= threshold:
                # Linear interpolation within bucket
                lower = 0.0 if i == 0 else self.buckets[i - 1]
                upper = self.buckets[i]
                if upper == float("inf"):
                    return self.max_value
                prev_count = 0 if i == 0 else self.counts[i - 1]
                bucket_count = count - prev_count
                if bucket_count == 0:
                    return lower
                fraction = (threshold - prev_count) / bucket_count
                return lower + fraction * (upper - lower)

        return self.max_value

    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.total_sum / self.total_count if self.total_count > 0 else 0.0


class MemorySampler:
    """Samples memory usage at configurable intervals.

    Thread-safe implementation using a background thread.
    """

    def __init__(self, max_samples: int = 1000, interval_seconds: float = 1.0) -> None:
        """Initialize the memory sampler.

        Args:
            max_samples: Maximum number of samples to retain
            interval_seconds: Sampling interval in seconds
        """
        self._samples: list[MemorySample] = []
        self._max_samples = max_samples
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

    def start(self) -> None:
        """Start background memory sampling."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop background memory sampling."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while True:
            with self._lock:
                if not self._running:
                    break

            self._take_sample()
            time.sleep(self._interval)

    def _take_sample(self) -> None:
        """Take a memory sample."""
        try:
            mem_info = self._process.memory_info()
            system_mem = psutil.virtual_memory()

            sample = MemorySample(
                timestamp=datetime.now(UTC),
                rss_mb=mem_info.rss / BYTES_PER_MB,
                vms_mb=mem_info.vms / BYTES_PER_MB,
                percent=mem_info.rss / system_mem.total * 100,
                available_gb=system_mem.available / BYTES_PER_GB,
            )

            with self._lock:
                self._samples.append(sample)
                # Trim to max samples
                if len(self._samples) > self._max_samples:
                    self._samples = self._samples[-self._max_samples :]

        except Exception:
            pass  # Silently handle sampling errors

    def sample_now(self) -> MemorySample:
        """Take an immediate memory sample and return it.

        Returns:
            Current memory sample
        """
        self._take_sample()
        with self._lock:
            return self._samples[-1] if self._samples else self._get_current_sample()

    def _get_current_sample(self) -> MemorySample:
        """Get current memory state without storing."""
        mem_info = self._process.memory_info()
        system_mem = psutil.virtual_memory()
        return MemorySample(
            timestamp=datetime.now(UTC),
            rss_mb=mem_info.rss / BYTES_PER_MB,
            vms_mb=mem_info.vms / BYTES_PER_MB,
            percent=mem_info.rss / system_mem.total * 100,
            available_gb=system_mem.available / BYTES_PER_GB,
        )

    def get_samples(self, since: datetime | None = None) -> list[MemorySample]:
        """Get collected memory samples.

        Args:
            since: Only return samples after this time

        Returns:
            List of memory samples
        """
        with self._lock:
            if since is None:
                return list(self._samples)
            return [s for s in self._samples if s.timestamp > since]

    def get_latest(self) -> MemorySample | None:
        """Get the most recent sample.

        Returns:
            Latest sample or None if no samples
        """
        with self._lock:
            return self._samples[-1] if self._samples else None

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics summary.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            if not self._samples:
                return {
                    "sample_count": 0,
                    "current_rss_mb": 0.0,
                    "peak_rss_mb": 0.0,
                    "avg_rss_mb": 0.0,
                }

            current = self._samples[-1]
            rss_values = [s.rss_mb for s in self._samples]

            return {
                "sample_count": len(self._samples),
                "current_rss_mb": round(current.rss_mb, 2),
                "peak_rss_mb": round(max(rss_values), 2),
                "avg_rss_mb": round(sum(rss_values) / len(rss_values), 2),
                "min_rss_mb": round(min(rss_values), 2),
                "available_gb": round(current.available_gb, 2),
                "memory_percent": round(current.percent, 2),
            }

    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


class RequestCounter:
    """Thread-safe counter for API requests.

    Tracks request counts by endpoint and method.
    """

    def __init__(self) -> None:
        """Initialize the request counter."""
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total = 0
        self._lock = threading.Lock()
        self._start_time = datetime.now(UTC)

    def increment(self, endpoint: str, method: str = "GET") -> None:
        """Increment the counter for an endpoint.

        Args:
            endpoint: The API endpoint path
            method: HTTP method
        """
        with self._lock:
            self._counts[endpoint][method] += 1
            self._total += 1

    def get_count(self, endpoint: str | None = None, method: str | None = None) -> int:
        """Get request count.

        Args:
            endpoint: Filter by endpoint (None for all)
            method: Filter by method (None for all)

        Returns:
            Request count matching filters
        """
        with self._lock:
            if endpoint is None:
                return self._total

            if endpoint not in self._counts:
                return 0

            if method is None:
                return sum(self._counts[endpoint].values())

            return self._counts[endpoint].get(method, 0)

    def get_all(self) -> dict[str, dict[str, int]]:
        """Get all request counts.

        Returns:
            Dictionary mapping endpoints to method counts
        """
        with self._lock:
            return {k: dict(v) for k, v in self._counts.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get request statistics.

        Returns:
            Dictionary with request statistics
        """
        with self._lock:
            elapsed = (datetime.now(UTC) - self._start_time).total_seconds()
            rate = self._total / elapsed if elapsed > 0 else 0.0

            return {
                "total_requests": self._total,
                "endpoints": len(self._counts),
                "requests_per_second": round(rate, 4),
                "uptime_seconds": round(elapsed, 2),
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._counts.clear()
            self._total = 0
            self._start_time = datetime.now(UTC)


class LatencyHistogram:
    """Thread-safe latency histogram for tracking operation timing.

    Uses configurable buckets for Prometheus-compatible histogram output.
    """

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS) -> None:
        """Initialize the latency histogram.

        Args:
            buckets: Bucket boundaries in seconds (must include +Inf)
        """
        self._buckets = buckets
        self._histograms: dict[str, HistogramData] = {}
        self._lock = threading.Lock()

    def observe(self, operation: str, duration: float) -> None:
        """Record a latency observation.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        with self._lock:
            if operation not in self._histograms:
                self._histograms[operation] = HistogramData(buckets=self._buckets)
            self._histograms[operation].observe(duration)

    def time(self, operation: str) -> _TimerContext:
        """Context manager for timing operations.

        Args:
            operation: Name of the operation

        Returns:
            Timer context manager

        Example:
            with histogram.time("database_query"):
                result = db.query(...)
        """
        return _TimerContext(self, operation)

    def get_percentiles(self, operation: str) -> dict[str, float]:
        """Get latency percentiles for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Dictionary with p50, p90, p95, p99 percentiles
        """
        with self._lock:
            if operation not in self._histograms:
                return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

            h = self._histograms[operation]
            return {
                "p50": round(h.percentile(50), 6),
                "p90": round(h.percentile(90), 6),
                "p95": round(h.percentile(95), 6),
                "p99": round(h.percentile(99), 6),
            }

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get latency statistics.

        Args:
            operation: Specific operation (None for all)

        Returns:
            Dictionary with latency statistics
        """
        with self._lock:
            if operation:
                if operation not in self._histograms:
                    return {"count": 0}
                h = self._histograms[operation]
                return {
                    "count": h.total_count,
                    "mean_ms": round(h.mean * 1000, 3),  # Convert to ms
                    "min_ms": round(h.min_value * 1000, 3) if h.min_value != float("inf") else 0,
                    "max_ms": round(h.max_value * 1000, 3),
                    "p50_ms": round(h.percentile(50) * 1000, 3),
                    "p90_ms": round(h.percentile(90) * 1000, 3),
                    "p95_ms": round(h.percentile(95) * 1000, 3),
                    "p99_ms": round(h.percentile(99) * 1000, 3),
                }

            # Return stats for all operations
            result = {}
            for op_name, h in self._histograms.items():
                result[op_name] = {
                    "count": h.total_count,
                    "mean_ms": round(h.mean * 1000, 3),
                    "p50_ms": round(h.percentile(50) * 1000, 3),
                    "p99_ms": round(h.percentile(99) * 1000, 3),
                }
            return result

    def get_histogram_data(self, operation: str) -> dict[str, Any] | None:
        """Get raw histogram data for Prometheus export.

        Args:
            operation: Name of the operation

        Returns:
            Histogram data with buckets, or None if not found
        """
        with self._lock:
            if operation not in self._histograms:
                return None

            h = self._histograms[operation]
            return {
                "buckets": list(self._buckets),
                "counts": list(h.counts),
                "total_count": h.total_count,
                "total_sum": h.total_sum,
            }

    def reset(self, operation: str | None = None) -> None:
        """Reset histogram data.

        Args:
            operation: Specific operation to reset (None for all)
        """
        with self._lock:
            if operation:
                if operation in self._histograms:
                    self._histograms[operation] = HistogramData(buckets=self._buckets)
            else:
                self._histograms.clear()


class _TimerContext:
    """Context manager for timing operations."""

    def __init__(self, histogram: LatencyHistogram, operation: str) -> None:
        self._histogram = histogram
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> _TimerContext:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        duration = time.perf_counter() - self._start
        self._histogram.observe(self._operation, duration)


# Global metrics instances (singleton pattern)
_memory_sampler: MemorySampler | None = None
_request_counter: RequestCounter | None = None
_latency_histogram: LatencyHistogram | None = None
_metrics_lock = threading.Lock()


def get_memory_sampler() -> MemorySampler:
    """Get the global memory sampler instance.

    Returns:
        Shared MemorySampler instance
    """
    global _memory_sampler
    if _memory_sampler is None:
        with _metrics_lock:
            if _memory_sampler is None:
                _memory_sampler = MemorySampler()
    return _memory_sampler


def get_request_counter() -> RequestCounter:
    """Get the global request counter instance.

    Returns:
        Shared RequestCounter instance
    """
    global _request_counter
    if _request_counter is None:
        with _metrics_lock:
            if _request_counter is None:
                _request_counter = RequestCounter()
    return _request_counter


def get_latency_histogram() -> LatencyHistogram:
    """Get the global latency histogram instance.

    Returns:
        Shared LatencyHistogram instance
    """
    global _latency_histogram
    if _latency_histogram is None:
        with _metrics_lock:
            if _latency_histogram is None:
                _latency_histogram = LatencyHistogram()
    return _latency_histogram


def reset_metrics() -> None:
    """Reset all global metrics instances."""
    global _memory_sampler, _request_counter, _latency_histogram
    with _metrics_lock:
        if _memory_sampler:
            _memory_sampler.stop()
            _memory_sampler = None
        _request_counter = None
        _latency_histogram = None


def force_gc() -> dict[str, Any]:
    """Force garbage collection and return memory delta.

    Returns:
        Dictionary with before/after memory stats
    """
    sampler = get_memory_sampler()
    before = sampler._get_current_sample()

    collected = gc.collect()

    after = sampler._get_current_sample()

    return {
        "collected_objects": collected,
        "rss_before_mb": round(before.rss_mb, 2),
        "rss_after_mb": round(after.rss_mb, 2),
        "rss_freed_mb": round(before.rss_mb - after.rss_mb, 2),
    }


class TTLCache:
    """Thread-safe cache with time-to-live expiration.

    Useful for caching expensive computations that should refresh periodically.
    """

    def __init__(self, ttl_seconds: float = 30.0, maxsize: int = 100) -> None:
        """Initialize the TTL cache.

        Args:
            ttl_seconds: Time-to-live for cached items in seconds
            maxsize: Maximum number of items to cache
        """
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Tuple of (found, value). found is False if key doesn't exist or is expired.
        """
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return True, value
                # Expired - remove it
                del self._cache[key]
            self._misses += 1
            return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self._maxsize:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]

            self._cache[key] = (time.time(), value)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate (None for all)
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# Pre-configured caches for common use cases
_conversation_cache: TTLCache | None = None
_health_cache: TTLCache | None = None
_model_info_cache: TTLCache | None = None


def get_conversation_cache() -> TTLCache:
    """Get the conversation list cache (TTL: 30s).

    Returns:
        TTLCache for conversation data
    """
    global _conversation_cache
    if _conversation_cache is None:
        with _metrics_lock:
            if _conversation_cache is None:
                _conversation_cache = TTLCache(ttl_seconds=30.0, maxsize=50)
    return _conversation_cache


def get_health_cache() -> TTLCache:
    """Get the health status cache (TTL: 5s).

    Returns:
        TTLCache for health data
    """
    global _health_cache
    if _health_cache is None:
        with _metrics_lock:
            if _health_cache is None:
                _health_cache = TTLCache(ttl_seconds=5.0, maxsize=10)
    return _health_cache


def get_model_info_cache() -> TTLCache:
    """Get the model info cache (TTL: 60s).

    Returns:
        TTLCache for model info data
    """
    global _model_info_cache
    if _model_info_cache is None:
        with _metrics_lock:
            if _model_info_cache is None:
                _model_info_cache = TTLCache(ttl_seconds=60.0, maxsize=10)
    return _model_info_cache
