"""Unit tests for async API endpoints and rate limiting.

Tests the async conversion, rate limiting, and timeout handling of the JARVIS API.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from core.health import reset_degradation_controller
from core.memory import reset_memory_controller
from jarvis.config import RateLimitConfig, reset_config


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_memory_controller()
    reset_degradation_controller()
    reset_config()
    yield
    reset_memory_controller()
    reset_degradation_controller()
    reset_config()


@pytest.fixture
def app():
    """Create a fresh test app instance."""
    from api.main import app as main_app

    return main_app


@pytest.fixture
def client(app):
    """Create a test client for the API."""
    return TestClient(app, raise_server_exceptions=False)


class TestRateLimitConfig:
    """Tests for rate limit configuration."""

    def test_rate_limit_config_defaults(self):
        """RateLimitConfig has correct defaults."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.generation_timeout_seconds == 30.0
        assert config.read_timeout_seconds == 10.0

    def test_rate_limit_config_validation(self):
        """RateLimitConfig validates input ranges."""
        from pydantic import ValidationError as PydanticValidationError

        # requests_per_minute must be >= 1
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(requests_per_minute=0)

        # requests_per_minute must be <= 1000
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(requests_per_minute=1001)

        # generation_timeout_seconds must be >= 1.0
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(generation_timeout_seconds=0.5)

        # read_timeout_seconds must be >= 1.0
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(read_timeout_seconds=0.5)


class TestRateLimitModule:
    """Tests for the rate limit module."""

    def test_get_remote_address_extracts_ip(self):
        """get_remote_address extracts client IP."""
        from api.ratelimit import get_remote_address

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"user-agent": "TestClient"}

        # For non-localhost, should return IP directly
        result = get_remote_address(mock_request)
        assert result == "192.168.1.100"

    def test_get_remote_address_localhost_includes_user_agent_hash(self):
        """get_remote_address includes user agent hash for localhost."""
        from api.ratelimit import get_remote_address

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "TestClient"}

        result = get_remote_address(mock_request)
        assert result.startswith("127.0.0.1:")
        # Should include a hash of the user agent
        assert ":" in result

    def test_rate_limit_constants(self):
        """Rate limit constants are defined correctly."""
        from api.ratelimit import (
            RATE_LIMIT_GENERATION,
            RATE_LIMIT_READ,
            RATE_LIMIT_WRITE,
            TIMEOUT_GENERATION,
            TIMEOUT_READ,
        )

        assert RATE_LIMIT_GENERATION == "10/minute"
        assert RATE_LIMIT_READ == "60/minute"
        assert RATE_LIMIT_WRITE == "30/minute"
        assert TIMEOUT_GENERATION == 30.0
        assert TIMEOUT_READ == 10.0

    def test_rate_limit_exceeded_handler(self):
        """rate_limit_exceeded_handler returns proper 429 response."""
        from api.ratelimit import rate_limit_exceeded_handler

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "TestClient"}

        # Create a mock exception with detail attribute
        mock_exc = MagicMock()
        mock_exc.detail = "Rate limit exceeded: 10 per 1 minute"

        response = rate_limit_exceeded_handler(mock_request, mock_exc)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"


class TestAsyncHealthEndpoint:
    """Tests for async health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_valid_structure(self, client):
        """Health endpoint returns proper structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "memory_available_gb" in data
        assert "memory_used_gb" in data
        assert "model_loaded" in data

    def test_root_returns_200(self, client):
        """Root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "jarvis-api"


class TestAsyncConversationsEndpoint:
    """Tests for async conversations endpoint."""

    def test_list_conversations_async(self, app, client):
        """list_conversations works with async."""
        from api.dependencies import get_imessage_reader

        mock_reader_instance = MagicMock()
        mock_reader_instance.get_conversations.return_value = []

        app.dependency_overrides[get_imessage_reader] = lambda: mock_reader_instance
        try:
            response = client.get("/conversations")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        finally:
            app.dependency_overrides.clear()

    def test_get_messages_async(self, app, client):
        """get_messages works with async."""
        from api.dependencies import get_imessage_reader

        mock_reader_instance = MagicMock()
        mock_reader_instance.get_messages.return_value = []

        app.dependency_overrides[get_imessage_reader] = lambda: mock_reader_instance
        try:
            response = client.get("/conversations/chat123/messages")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        finally:
            app.dependency_overrides.clear()

    def test_search_messages_async(self, app, client):
        """search_messages works with async."""
        from api.dependencies import get_imessage_reader

        mock_reader_instance = MagicMock()
        mock_reader_instance.search.return_value = []

        app.dependency_overrides[get_imessage_reader] = lambda: mock_reader_instance
        try:
            response = client.get("/conversations/search?q=test")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        finally:
            app.dependency_overrides.clear()


class TestAsyncDraftsEndpoint:
    """Tests for async drafts endpoint."""

    def test_generate_draft_reply_async(self, app, client):
        """generate_draft_reply works with async."""
        from api.dependencies import get_imessage_reader

        mock_reader_instance = MagicMock()
        mock_message = MagicMock()
        mock_message.text = "Hello"
        mock_message.is_from_me = False
        mock_message.sender_name = "John"
        mock_message.sender = "+1234567890"
        mock_reader_instance.get_messages.return_value = [mock_message]

        app.dependency_overrides[get_imessage_reader] = lambda: mock_reader_instance

        # Also mock get_generator using patch since it's not a dependency
        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hi! How can I help?"
        mock_generator.generate.return_value = mock_response

        try:
            with patch("api.routers.drafts.get_warm_generator", return_value=mock_generator):
                response = client.post(
                    "/drafts/reply",
                    json={"chat_id": "chat123", "num_suggestions": 1},
                )
                # Should succeed or timeout (depending on thread pool)
                assert response.status_code in (200, 408)
        finally:
            app.dependency_overrides.clear()

    def test_summarize_conversation_async(self, app, client):
        """summarize_conversation works with async."""
        from api.dependencies import get_imessage_reader

        mock_reader_instance = MagicMock()
        mock_message = MagicMock()
        mock_message.text = "Hello"
        mock_message.date = datetime(2024, 1, 15, 10, 30)
        mock_message.is_from_me = False
        mock_message.sender_name = "John"
        mock_message.sender = "+1234567890"
        mock_reader_instance.get_messages.return_value = [mock_message]

        app.dependency_overrides[get_imessage_reader] = lambda: mock_reader_instance

        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Summary: Test summary\nKey points:\n- Point 1"
        mock_generator.generate.return_value = mock_response

        try:
            with patch("api.routers.drafts.get_warm_generator", return_value=mock_generator):
                response = client.post(
                    "/drafts/summarize",
                    json={"chat_id": "chat123", "num_messages": 10},
                )
                # Should succeed or timeout
                assert response.status_code in (200, 408, 404)
        finally:
            app.dependency_overrides.clear()


class TestAsyncSuggestionsEndpoint:
    """Tests for async suggestions endpoint."""

    def test_get_suggestions_async(self, client):
        """get_suggestions works with async."""
        response = client.post(
            "/suggestions",
            json={"last_message": "Thanks for your help!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0

    def test_suggestions_returns_relevant_responses(self, client):
        """Suggestions match expected patterns."""
        response = client.post(
            "/suggestions",
            json={"last_message": "Thanks!"},
        )

        assert response.status_code == 200
        data = response.json()
        suggestions = data["suggestions"]

        # Should include "You're welcome!" for "Thanks!"
        texts = [s["text"] for s in suggestions]
        assert "You're welcome!" in texts


class TestAsyncSettingsEndpoint:
    """Tests for async settings endpoint."""

    def test_get_settings_async(self, client):
        """get_settings works with async."""
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data

    def test_list_models_async(self, client):
        """list_models works with async."""
        response = client.get("/settings/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestAsyncMetricsEndpoint:
    """Tests for async metrics endpoint."""

    def test_get_prometheus_metrics_async(self, client):
        """get_prometheus_metrics works with async."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "jarvis_memory_rss_bytes" in response.text

    def test_get_memory_metrics_async(self, client):
        """get_memory_metrics works with async."""
        response = client.get("/metrics/memory")
        assert response.status_code == 200
        data = response.json()
        assert "process" in data
        assert "system" in data

    def test_get_latency_metrics_async(self, client):
        """get_latency_metrics works with async."""
        response = client.get("/metrics/latency")
        assert response.status_code == 200
        data = response.json()
        assert "operations" in data


class TestTimeoutHandling:
    """Tests for timeout handling in async endpoints."""

    @pytest.mark.anyio
    async def test_timeout_error_handler(self):
        """timeout_error_handler returns proper 408 response."""
        from api.errors import timeout_error_handler

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"

        # Use await for async function
        response = await timeout_error_handler(mock_request, TimeoutError("Timed out"))

        assert response.status_code == 408
        data = response.body.decode()
        assert "REQUEST_TIMEOUT" in data
        assert "Retry-After" in response.headers


class TestErrorHandlerRegistration:
    """Tests for error handler registration."""

    def test_rate_limit_handler_registered(self, app):
        """Rate limit exception handler is registered."""
        from slowapi.errors import RateLimitExceeded

        # Check that the app has the handler registered
        handlers = app.exception_handlers
        assert RateLimitExceeded in handlers

    def test_timeout_handler_registered(self, app):
        """Timeout exception handler is registered."""
        handlers = app.exception_handlers
        assert TimeoutError in handlers


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_limiter_in_app_state(self, app):
        """Limiter is configured in app state."""
        assert hasattr(app.state, "limiter")
        assert app.state.limiter is not None

    def test_cors_configured(self, app):
        """CORS middleware is configured."""
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestConfigMigration:
    """Tests for config migration with rate_limit section."""

    def test_config_version_4_adds_rate_limit(self):
        """Config migration to v6 adds rate_limit section."""
        from jarvis.config import CONFIG_VERSION, _migrate_config

        data = {"config_version": 3, "model_path": "test"}
        migrated = _migrate_config(data)

        assert "rate_limit" in migrated
        assert migrated["config_version"] == CONFIG_VERSION

    def test_rate_limit_in_jarvis_config(self):
        """JarvisConfig includes rate_limit field."""
        from jarvis.config import JarvisConfig

        config = JarvisConfig()
        assert hasattr(config, "rate_limit")
        assert config.rate_limit.enabled is True
        assert config.rate_limit.requests_per_minute == 60
