"""Tests for the metrics module.

Tests memory sampling, request counting, latency histograms, and TTL caching.
"""

import time

from jarvis.metrics import (
    LatencyHistogram,
    MemorySampler,
    RequestCounter,
    TTLCache,
    get_conversation_cache,
    get_health_cache,
    get_latency_histogram,
    get_memory_sampler,
    get_model_info_cache,
    get_request_counter,
    reset_metrics,
)


class TestRequestCounter:
    """Tests for RequestCounter."""

    def test_increment_increases_count(self):
        """Increment increases endpoint count."""
        counter = RequestCounter()
        counter.increment("/test", "GET")

        assert counter.get_count("/test", "GET") == 1
        assert counter.get_count() == 1

    def test_increment_multiple_endpoints(self):
        """Tracks multiple endpoints separately."""
        counter = RequestCounter()
        counter.increment("/api/v1", "GET")
        counter.increment("/api/v1", "GET")
        counter.increment("/api/v2", "POST")

        assert counter.get_count("/api/v1", "GET") == 2
        assert counter.get_count("/api/v2", "POST") == 1
        assert counter.get_count() == 3

    def test_get_count_missing_endpoint_returns_zero(self):
        """Get count for missing endpoint returns 0."""
        counter = RequestCounter()

        assert counter.get_count("/nonexistent") == 0
        assert counter.get_count("/nonexistent", "GET") == 0

    def test_get_all_returns_dict(self):
        """Get all returns dictionary of counts."""
        counter = RequestCounter()
        counter.increment("/a", "GET")
        counter.increment("/b", "POST")

        result = counter.get_all()

        assert result == {"/a": {"GET": 1}, "/b": {"POST": 1}}

    def test_get_stats_includes_rate(self):
        """Get stats includes request rate."""
        counter = RequestCounter()
        counter.increment("/test", "GET")

        stats = counter.get_stats()

        assert "total_requests" in stats
        assert "endpoints" in stats
        assert "requests_per_second" in stats
        assert "uptime_seconds" in stats
        assert stats["total_requests"] == 1

    def test_reset_clears_counts(self):
        """Reset clears all counts."""
        counter = RequestCounter()
        counter.increment("/test", "GET")
        counter.reset()

        assert counter.get_count() == 0
        assert counter.get_all() == {}


class TestLatencyHistogram:
    """Tests for LatencyHistogram."""

    def test_observe_records_value(self):
        """Observe records latency value."""
        histogram = LatencyHistogram()
        histogram.observe("test_op", 0.1)

        stats = histogram.get_stats("test_op")
        assert stats["count"] == 1

    def test_observe_multiple_values(self):
        """Tracks multiple observations."""
        histogram = LatencyHistogram()
        histogram.observe("test_op", 0.1)
        histogram.observe("test_op", 0.2)
        histogram.observe("test_op", 0.3)

        stats = histogram.get_stats("test_op")
        assert stats["count"] == 3

    def test_percentiles_calculated(self):
        """Percentiles are calculated from histogram."""
        histogram = LatencyHistogram()
        # Add many observations for better percentile estimation
        for i in range(100):
            histogram.observe("test_op", i / 1000.0)  # 0-99ms

        percentiles = histogram.get_percentiles("test_op")
        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

    def test_timer_context_records_duration(self):
        """Timer context manager records duration."""
        histogram = LatencyHistogram()

        with histogram.time("test_op"):
            time.sleep(0.01)  # 10ms

        stats = histogram.get_stats("test_op")
        assert stats["count"] == 1
        assert stats["mean_ms"] > 0

    def test_get_histogram_data_returns_buckets(self):
        """Get histogram data returns bucket information."""
        histogram = LatencyHistogram()
        histogram.observe("test_op", 0.1)

        data = histogram.get_histogram_data("test_op")

        assert data is not None
        assert "buckets" in data
        assert "counts" in data
        assert "total_count" in data
        assert "total_sum" in data

    def test_get_histogram_data_missing_returns_none(self):
        """Get histogram data for missing operation returns None."""
        histogram = LatencyHistogram()
        assert histogram.get_histogram_data("nonexistent") is None

    def test_reset_clears_operation(self):
        """Reset clears specific operation."""
        histogram = LatencyHistogram()
        histogram.observe("op1", 0.1)
        histogram.observe("op2", 0.2)
        histogram.reset("op1")

        assert histogram.get_stats("op1")["count"] == 0
        assert histogram.get_stats("op2")["count"] == 1

    def test_reset_all_clears_everything(self):
        """Reset without operation clears all."""
        histogram = LatencyHistogram()
        histogram.observe("op1", 0.1)
        histogram.observe("op2", 0.2)
        histogram.reset()

        assert histogram.get_stats() == {}


class TestMemorySampler:
    """Tests for MemorySampler."""

    def test_sample_now_returns_sample(self):
        """Sample now returns current memory sample."""
        sampler = MemorySampler()
        sample = sampler.sample_now()

        assert sample is not None
        assert sample.rss_mb > 0
        assert sample.vms_mb > 0
        assert sample.timestamp is not None

    def test_get_samples_returns_list(self):
        """Get samples returns list of samples."""
        sampler = MemorySampler()
        sampler.sample_now()
        sampler.sample_now()

        samples = sampler.get_samples()
        assert len(samples) == 2

    def test_get_latest_returns_most_recent(self):
        """Get latest returns most recent sample."""
        sampler = MemorySampler()
        sampler.sample_now()
        time.sleep(0.01)
        sampler.sample_now()

        latest = sampler.get_latest()
        samples = sampler.get_samples()

        assert latest == samples[-1]

    def test_get_stats_returns_statistics(self):
        """Get stats returns memory statistics."""
        sampler = MemorySampler()
        sampler.sample_now()
        sampler.sample_now()

        stats = sampler.get_stats()

        assert "sample_count" in stats
        assert "current_rss_mb" in stats
        assert "peak_rss_mb" in stats
        assert "avg_rss_mb" in stats

    def test_clear_removes_samples(self):
        """Clear removes all samples."""
        sampler = MemorySampler()
        sampler.sample_now()
        sampler.clear()

        assert len(sampler.get_samples()) == 0

    def test_max_samples_limit(self):
        """Sampler respects max samples limit."""
        sampler = MemorySampler(max_samples=5)

        for _ in range(10):
            sampler.sample_now()

        assert len(sampler.get_samples()) == 5

    def test_start_stop_background_sampling(self):
        """Background sampling can be started and stopped."""
        sampler = MemorySampler(interval_seconds=0.1)
        sampler.start()
        time.sleep(0.25)
        sampler.stop()

        # Should have collected some samples
        assert len(sampler.get_samples()) >= 1


class TestTTLCache:
    """Tests for TTLCache."""

    def test_set_and_get(self):
        """Set and get work correctly."""
        cache = TTLCache(ttl_seconds=10.0)
        cache.set("key", "value")

        found, value = cache.get("key")
        assert found is True
        assert value == "value"

    def test_get_missing_key(self):
        """Get missing key returns not found."""
        cache = TTLCache()
        found, value = cache.get("nonexistent")

        assert found is False
        assert value is None

    def test_ttl_expiration(self):
        """Cache entries expire after TTL."""
        cache = TTLCache(ttl_seconds=0.1)
        cache.set("key", "value")

        # Value should exist immediately
        found, _ = cache.get("key")
        assert found is True

        # Wait for expiration
        time.sleep(0.15)

        found, _ = cache.get("key")
        assert found is False

    def test_maxsize_eviction(self):
        """Cache evicts oldest entries when maxsize reached."""
        cache = TTLCache(ttl_seconds=60.0, maxsize=3)
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.set("key3", "v3")
        cache.set("key4", "v4")

        # key1 should be evicted
        found, _ = cache.get("key1")
        assert found is False

        # Others should exist
        found, _ = cache.get("key4")
        assert found is True

    def test_invalidate_specific_key(self):
        """Invalidate removes specific key."""
        cache = TTLCache()
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.invalidate("key1")

        found, _ = cache.get("key1")
        assert found is False

        found, _ = cache.get("key2")
        assert found is True

    def test_invalidate_all(self):
        """Invalidate without key clears all."""
        cache = TTLCache()
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.invalidate()

        stats = cache.stats()
        assert stats["size"] == 0

    def test_stats_tracking(self):
        """Stats tracks hits and misses."""
        cache = TTLCache(ttl_seconds=60.0)
        cache.set("key", "value")

        cache.get("key")  # Hit
        cache.get("key")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestGlobalSingletons:
    """Tests for global singleton getters."""

    def test_get_memory_sampler_returns_same_instance(self):
        """Get memory sampler returns same instance."""
        reset_metrics()

        s1 = get_memory_sampler()
        s2 = get_memory_sampler()

        assert s1 is s2

    def test_get_request_counter_returns_same_instance(self):
        """Get request counter returns same instance."""
        reset_metrics()

        c1 = get_request_counter()
        c2 = get_request_counter()

        assert c1 is c2

    def test_get_latency_histogram_returns_same_instance(self):
        """Get latency histogram returns same instance."""
        reset_metrics()

        h1 = get_latency_histogram()
        h2 = get_latency_histogram()

        assert h1 is h2

    def test_reset_metrics_clears_singletons(self):
        """Reset metrics creates new instances."""
        s1 = get_memory_sampler()
        reset_metrics()
        s2 = get_memory_sampler()

        assert s1 is not s2

    def test_get_conversation_cache_has_correct_ttl(self):
        """Conversation cache has 30s TTL."""
        cache = get_conversation_cache()
        stats = cache.stats()

        assert stats["ttl_seconds"] == 30.0

    def test_get_health_cache_has_correct_ttl(self):
        """Health cache has 5s TTL."""
        cache = get_health_cache()
        stats = cache.stats()

        assert stats["ttl_seconds"] == 5.0

    def test_get_model_info_cache_has_correct_ttl(self):
        """Model info cache has 60s TTL."""
        cache = get_model_info_cache()
        stats = cache.stats()

        assert stats["ttl_seconds"] == 60.0


class TestHistogramDataPercentiles:
    """Tests for histogram percentile calculation."""

    def test_empty_histogram_returns_zero_percentiles(self):
        """Empty histogram returns zero for percentiles."""
        histogram = LatencyHistogram()
        percentiles = histogram.get_percentiles("empty")

        assert percentiles["p50"] == 0.0
        assert percentiles["p99"] == 0.0

    def test_single_value_histogram(self):
        """Histogram with single value computes percentiles."""
        histogram = LatencyHistogram()
        histogram.observe("test", 0.5)

        percentiles = histogram.get_percentiles("test")

        # All percentiles should be related to the single value
        assert percentiles["p50"] >= 0
        assert percentiles["p99"] >= 0

    def test_mean_calculation(self):
        """Mean is calculated correctly."""
        histogram = LatencyHistogram()
        histogram.observe("test", 0.1)
        histogram.observe("test", 0.2)
        histogram.observe("test", 0.3)

        stats = histogram.get_stats("test")

        # Mean should be (0.1 + 0.2 + 0.3) / 3 = 0.2 seconds = 200ms
        assert abs(stats["mean_ms"] - 200.0) < 1  # Allow small floating point error
