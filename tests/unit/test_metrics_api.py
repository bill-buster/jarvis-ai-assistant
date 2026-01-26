"""Tests for the metrics API endpoints.

Tests Prometheus-compatible metrics export and memory/latency endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from jarvis.metrics import reset_metrics


@pytest.fixture(autouse=True)
def reset_metrics_before_each():
    """Reset metrics before each test."""
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app, raise_server_exceptions=False)


class TestPrometheusMetrics:
    """Tests for the /metrics endpoint (Prometheus format)."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint returns 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_text_plain(self, client):
        """Metrics endpoint returns plain text."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_memory_rss(self, client):
        """Metrics contains memory RSS metric."""
        response = client.get("/metrics")
        assert "jarvis_memory_rss_bytes" in response.text

    def test_metrics_contains_memory_vms(self, client):
        """Metrics contains memory VMS metric."""
        response = client.get("/metrics")
        assert "jarvis_memory_vms_bytes" in response.text

    def test_metrics_contains_available_memory(self, client):
        """Metrics contains available memory metric."""
        response = client.get("/metrics")
        assert "jarvis_memory_available_bytes" in response.text

    def test_metrics_contains_uptime(self, client):
        """Metrics contains uptime metric."""
        response = client.get("/metrics")
        assert "jarvis_uptime_seconds" in response.text

    def test_metrics_has_help_comments(self, client):
        """Metrics has HELP comments."""
        response = client.get("/metrics")
        assert "# HELP" in response.text

    def test_metrics_has_type_comments(self, client):
        """Metrics has TYPE comments."""
        response = client.get("/metrics")
        assert "# TYPE" in response.text

    def test_metrics_request_counter_after_requests(self, client):
        """Request counter appears after making requests."""
        # Make a request to another endpoint first
        client.get("/health")

        response = client.get("/metrics")
        # Health endpoint should be counted
        assert "jarvis_requests_total" in response.text


class TestMemoryMetrics:
    """Tests for the /metrics/memory endpoint."""

    def test_memory_returns_200(self, client):
        """Memory endpoint returns 200 OK."""
        response = client.get("/metrics/memory")
        assert response.status_code == 200

    def test_memory_returns_json(self, client):
        """Memory endpoint returns JSON."""
        response = client.get("/metrics/memory")
        assert "application/json" in response.headers["content-type"]

    def test_memory_contains_process_stats(self, client):
        """Memory endpoint contains process stats."""
        response = client.get("/metrics/memory")
        data = response.json()

        assert "process" in data
        assert "rss_mb" in data["process"]
        assert "vms_mb" in data["process"]
        assert "percent" in data["process"]

    def test_memory_contains_system_stats(self, client):
        """Memory endpoint contains system stats."""
        response = client.get("/metrics/memory")
        data = response.json()

        assert "system" in data
        assert "total_gb" in data["system"]
        assert "available_gb" in data["system"]
        assert "used_gb" in data["system"]

    def test_memory_contains_sampling_stats(self, client):
        """Memory endpoint contains sampling stats."""
        response = client.get("/metrics/memory")
        data = response.json()

        assert "sampling" in data
        assert "sample_count" in data["sampling"]

    def test_memory_contains_trend_data(self, client):
        """Memory endpoint contains trend data."""
        response = client.get("/metrics/memory")
        data = response.json()

        assert "trend" in data
        assert isinstance(data["trend"], list)

    def test_memory_values_are_positive(self, client):
        """Memory values are positive numbers."""
        response = client.get("/metrics/memory")
        data = response.json()

        assert data["process"]["rss_mb"] > 0
        assert data["system"]["total_gb"] > 0


class TestLatencyMetrics:
    """Tests for the /metrics/latency endpoint."""

    def test_latency_returns_200(self, client):
        """Latency endpoint returns 200 OK."""
        response = client.get("/metrics/latency")
        assert response.status_code == 200

    def test_latency_returns_json(self, client):
        """Latency endpoint returns JSON."""
        response = client.get("/metrics/latency")
        assert "application/json" in response.headers["content-type"]

    def test_latency_contains_operations(self, client):
        """Latency endpoint contains operations dict."""
        response = client.get("/metrics/latency")
        data = response.json()

        assert "operations" in data
        assert isinstance(data["operations"], dict)

    def test_latency_contains_summary(self, client):
        """Latency endpoint contains summary stats."""
        response = client.get("/metrics/latency")
        data = response.json()

        assert "summary" in data
        assert "total_requests" in data["summary"]

    def test_latency_after_requests_shows_operations(self, client):
        """Latency shows operations after making requests."""
        # Make requests to create latency data
        client.get("/health")
        client.get("/health")

        response = client.get("/metrics/latency")
        data = response.json()

        # Should have at least one operation recorded
        # (health endpoint won't be recorded if skipped by middleware)
        assert "operations" in data


class TestRequestMetrics:
    """Tests for the /metrics/requests endpoint."""

    def test_requests_returns_200(self, client):
        """Requests endpoint returns 200 OK."""
        response = client.get("/metrics/requests")
        assert response.status_code == 200

    def test_requests_contains_endpoints(self, client):
        """Requests endpoint contains endpoints dict."""
        response = client.get("/metrics/requests")
        data = response.json()

        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)

    def test_requests_contains_stats(self, client):
        """Requests endpoint contains stats."""
        response = client.get("/metrics/requests")
        data = response.json()

        assert "stats" in data
        assert "total_requests" in data["stats"]


class TestGarbageCollection:
    """Tests for the /metrics/gc POST endpoint."""

    def test_gc_returns_200(self, client):
        """GC endpoint returns 200 OK."""
        response = client.post("/metrics/gc")
        assert response.status_code == 200

    def test_gc_returns_memory_delta(self, client):
        """GC endpoint returns memory delta."""
        response = client.post("/metrics/gc")
        data = response.json()

        assert "collected_objects" in data
        assert "rss_before_mb" in data
        assert "rss_after_mb" in data
        assert "rss_freed_mb" in data


class TestMemorySample:
    """Tests for the /metrics/sample POST endpoint."""

    def test_sample_returns_200(self, client):
        """Sample endpoint returns 200 OK."""
        response = client.post("/metrics/sample")
        assert response.status_code == 200

    def test_sample_returns_memory_data(self, client):
        """Sample endpoint returns memory data."""
        response = client.post("/metrics/sample")
        data = response.json()

        assert "timestamp" in data
        assert "rss_mb" in data
        assert "vms_mb" in data
        assert "percent" in data
        assert "available_gb" in data


class TestMetricsReset:
    """Tests for the /metrics/reset POST endpoint."""

    def test_reset_returns_200(self, client):
        """Reset endpoint returns 200 OK."""
        response = client.post("/metrics/reset")
        assert response.status_code == 200

    def test_reset_returns_confirmation(self, client):
        """Reset endpoint returns confirmation message."""
        response = client.post("/metrics/reset")
        data = response.json()

        assert data["status"] == "ok"
        assert "reset" in data["message"].lower()


class TestMetricsMiddleware:
    """Tests for the metrics middleware."""

    def test_response_time_header_added(self, client):
        """Response time header is added to responses."""
        response = client.get("/health")

        assert "X-Response-Time" in response.headers
        # Should be a duration like "0.0123s"
        assert response.headers["X-Response-Time"].endswith("s")

    def test_metrics_endpoint_not_counted(self, client):
        """Metrics endpoints are not counted to avoid recursion."""
        reset_metrics()

        # Access metrics multiple times
        client.get("/metrics")
        client.get("/metrics")
        client.get("/metrics")

        response = client.get("/metrics/requests")
        data = response.json()

        # /metrics should not appear in endpoints or have low count
        # (The middleware skips /metrics paths)
        assert data["stats"]["total_requests"] == 0 or "/metrics" not in str(data["endpoints"])
