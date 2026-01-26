"""Unit tests for the health API endpoint.

Tests system health status including memory, model, and permission state
with mocked external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.health import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    _check_imessage_access,
    _check_model_loaded,
    _get_memory_mode,
    _get_model_info,
    _get_process_memory,
    _get_recommended_model,
)
from jarvis.metrics import get_health_cache, get_model_info_cache


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear TTL caches before each test to ensure test isolation."""
    get_health_cache().invalidate()
    get_model_info_cache().invalidate()
    yield
    # Also clear after test for good measure
    get_health_cache().invalidate()
    get_model_info_cache().invalidate()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestConstants:
    """Tests for module constants."""

    def test_bytes_per_mb(self):
        """BYTES_PER_MB is correct."""
        assert BYTES_PER_MB == 1024 * 1024

    def test_bytes_per_gb(self):
        """BYTES_PER_GB is correct."""
        assert BYTES_PER_GB == 1024**3


class TestGetProcessMemory:
    """Tests for _get_process_memory helper."""

    @patch("api.routers.health.psutil.Process")
    @patch("api.routers.health.os.getpid")
    def test_returns_memory_in_mb(self, mock_getpid, mock_process_class):
        """Returns RSS and VMS in megabytes."""
        mock_getpid.return_value = 12345
        mock_process = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 100 * BYTES_PER_MB  # 100 MB
        mock_mem_info.vms = 200 * BYTES_PER_MB  # 200 MB
        mock_process.memory_info.return_value = mock_mem_info
        mock_process_class.return_value = mock_process

        rss, vms = _get_process_memory()

        assert rss == 100.0
        assert vms == 200.0

    @patch("api.routers.health.psutil.Process")
    @patch("api.routers.health.os.getpid")
    def test_returns_zero_on_exception(self, mock_getpid, mock_process_class):
        """Returns (0.0, 0.0) when process info fails."""
        mock_getpid.return_value = 12345
        mock_process_class.side_effect = RuntimeError("Process not found")

        rss, vms = _get_process_memory()

        assert rss == 0.0
        assert vms == 0.0


class TestCheckImessageAccess:
    """Tests for _check_imessage_access helper."""

    @patch("integrations.imessage.ChatDBReader")
    def test_returns_true_when_accessible(self, mock_reader_class):
        """Returns True when iMessage database is accessible."""
        mock_reader = MagicMock()
        mock_reader.check_access.return_value = True
        mock_reader_class.return_value = mock_reader

        result = _check_imessage_access()

        assert result is True
        mock_reader.close.assert_called_once()

    @patch("integrations.imessage.ChatDBReader")
    def test_returns_false_when_not_accessible(self, mock_reader_class):
        """Returns False when iMessage database is not accessible."""
        mock_reader = MagicMock()
        mock_reader.check_access.return_value = False
        mock_reader_class.return_value = mock_reader

        result = _check_imessage_access()

        assert result is False

    @patch("integrations.imessage.ChatDBReader")
    def test_returns_false_on_exception(self, mock_reader_class):
        """Returns False when checking access throws exception."""
        mock_reader_class.side_effect = RuntimeError("Database error")

        result = _check_imessage_access()

        assert result is False


class TestGetMemoryMode:
    """Tests for _get_memory_mode helper."""

    def test_full_mode_when_high_memory(self):
        """Returns FULL when available memory >= 4GB."""
        assert _get_memory_mode(4.0) == "FULL"
        assert _get_memory_mode(5.5) == "FULL"
        assert _get_memory_mode(16.0) == "FULL"

    def test_lite_mode_when_medium_memory(self):
        """Returns LITE when available memory is 2-4GB."""
        assert _get_memory_mode(2.0) == "LITE"
        assert _get_memory_mode(3.5) == "LITE"
        assert _get_memory_mode(3.9) == "LITE"

    def test_minimal_mode_when_low_memory(self):
        """Returns MINIMAL when available memory < 2GB."""
        assert _get_memory_mode(1.9) == "MINIMAL"
        assert _get_memory_mode(1.0) == "MINIMAL"
        assert _get_memory_mode(0.5) == "MINIMAL"


class TestCheckModelLoaded:
    """Tests for _check_model_loaded helper."""

    @patch("models.get_generator")
    def test_returns_true_when_model_loaded(self, mock_get_generator):
        """Returns True when model is loaded."""
        mock_generator = MagicMock()
        mock_generator._model = "loaded_model"
        mock_get_generator.return_value = mock_generator

        result = _check_model_loaded()

        assert result is True

    @patch("models.get_generator")
    def test_returns_false_when_model_not_loaded(self, mock_get_generator):
        """Returns False when model is not loaded."""
        mock_generator = MagicMock()
        mock_generator._model = None
        mock_get_generator.return_value = mock_generator

        result = _check_model_loaded()

        assert result is False

    @patch("models.get_generator")
    def test_returns_false_on_exception(self, mock_get_generator):
        """Returns False when getting generator throws exception."""
        mock_get_generator.side_effect = RuntimeError("Model unavailable")

        result = _check_model_loaded()

        assert result is False


class TestGetModelInfo:
    """Tests for _get_model_info helper."""

    @patch("models.get_generator")
    def test_returns_model_info_when_available(self, mock_get_generator):
        """Returns ModelInfo when model info is available."""
        mock_loader = MagicMock()
        mock_loader.get_current_model_info.return_value = {
            "id": "test-model-id",
            "display_name": "Test Model",
            "loaded": True,
            "memory_usage_mb": 512.0,
            "quality_tier": "good",
        }
        mock_generator = MagicMock()
        mock_generator._loader = mock_loader
        mock_get_generator.return_value = mock_generator

        result = _get_model_info()

        assert result is not None
        assert result.id == "test-model-id"
        assert result.display_name == "Test Model"
        assert result.loaded is True
        assert result.memory_usage_mb == 512.0
        assert result.quality_tier == "good"

    @patch("models.get_generator")
    def test_returns_model_info_with_defaults(self, mock_get_generator):
        """Returns ModelInfo with defaults for missing fields."""
        mock_loader = MagicMock()
        mock_loader.get_current_model_info.return_value = {
            "id": "test-model",
        }
        mock_generator = MagicMock()
        mock_generator._loader = mock_loader
        mock_get_generator.return_value = mock_generator

        result = _get_model_info()

        assert result is not None
        assert result.id == "test-model"
        assert result.display_name == "Unknown"
        assert result.loaded is False
        assert result.memory_usage_mb == 0.0
        assert result.quality_tier is None

    @patch("models.get_generator")
    def test_returns_none_on_exception(self, mock_get_generator):
        """Returns None when getting model info throws exception."""
        mock_get_generator.side_effect = RuntimeError("Model error")

        result = _get_model_info()

        assert result is None


class TestGetRecommendedModel:
    """Tests for _get_recommended_model helper."""

    @patch("models.get_recommended_model")
    def test_returns_model_id(self, mock_get_recommended):
        """Returns recommended model ID."""
        mock_spec = MagicMock()
        mock_spec.id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        mock_get_recommended.return_value = mock_spec

        result = _get_recommended_model(16.0)

        assert result == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        mock_get_recommended.assert_called_once_with(16.0)

    @patch("models.get_recommended_model")
    def test_returns_none_on_exception(self, mock_get_recommended):
        """Returns None when getting recommendation throws exception."""
        mock_get_recommended.side_effect = RuntimeError("Registry error")

        result = _get_recommended_model(8.0)

        assert result is None


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_healthy_status_with_all_systems_ok(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Returns healthy status when all systems are operational."""
        mock_vmem.return_value = MagicMock(
            available=8 * BYTES_PER_GB,
            used=8 * BYTES_PER_GB,
            total=16 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (256.0, 512.0)
        mock_imessage.return_value = True
        mock_model_loaded.return_value = True
        mock_model_info.return_value = None
        mock_recommended.return_value = "test-model"

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["imessage_access"] is True
        assert data["memory_available_gb"] == 8.0
        assert data["memory_used_gb"] == 8.0
        assert data["memory_mode"] == "FULL"
        assert data["model_loaded"] is True
        assert data["permissions_ok"] is True
        assert data["details"] is None
        assert data["jarvis_rss_mb"] == 256.0
        assert data["jarvis_vms_mb"] == 512.0
        assert data["system_ram_gb"] == 16.0
        assert data["recommended_model"] == "test-model"

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_degraded_status_with_low_memory(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Returns degraded status when memory is low but iMessage works."""
        mock_vmem.return_value = MagicMock(
            available=1 * BYTES_PER_GB,  # Low memory
            used=7 * BYTES_PER_GB,
            total=8 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (128.0, 256.0)
        mock_imessage.return_value = True
        mock_model_loaded.return_value = False
        mock_model_info.return_value = None
        mock_recommended.return_value = None

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["imessage_access"] is True
        assert data["memory_available_gb"] == 1.0
        assert data["memory_mode"] == "MINIMAL"
        assert data["model_loaded"] is False
        assert data["details"]["memory"] == "Low memory: 1.0GB available"

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_unhealthy_status_when_imessage_denied(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Returns unhealthy status when iMessage access is denied."""
        mock_vmem.return_value = MagicMock(
            available=8 * BYTES_PER_GB,
            used=8 * BYTES_PER_GB,
            total=16 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (256.0, 512.0)
        mock_imessage.return_value = False  # Denied
        mock_model_loaded.return_value = True
        mock_model_info.return_value = None
        mock_recommended.return_value = "test-model"

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["imessage_access"] is False
        assert data["permissions_ok"] is False
        assert data["details"]["imessage"] == "Full Disk Access required"

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_unhealthy_takes_precedence_over_degraded(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Unhealthy status takes precedence when both iMessage denied and low memory."""
        mock_vmem.return_value = MagicMock(
            available=1 * BYTES_PER_GB,  # Low memory
            used=7 * BYTES_PER_GB,
            total=8 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (128.0, 256.0)
        mock_imessage.return_value = False  # Denied
        mock_model_loaded.return_value = False
        mock_model_info.return_value = None
        mock_recommended.return_value = None

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # Unhealthy takes precedence
        assert data["status"] == "unhealthy"
        # Both issues reported
        assert "imessage" in data["details"]
        assert "memory" in data["details"]

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_lite_memory_mode(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Returns LITE memory mode when 2-4GB available."""
        mock_vmem.return_value = MagicMock(
            available=3 * BYTES_PER_GB,  # Medium memory
            used=5 * BYTES_PER_GB,
            total=8 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (128.0, 256.0)
        mock_imessage.return_value = True
        mock_model_loaded.return_value = True
        mock_model_info.return_value = None
        mock_recommended.return_value = None

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["memory_mode"] == "LITE"

    @patch("api.routers.health._get_recommended_model")
    @patch("api.routers.health._get_model_info")
    @patch("api.routers.health._check_model_loaded")
    @patch("api.routers.health._check_imessage_access")
    @patch("api.routers.health._get_process_memory")
    @patch("api.routers.health.psutil.virtual_memory")
    def test_includes_model_info_when_available(
        self,
        mock_vmem,
        mock_process_mem,
        mock_imessage,
        mock_model_loaded,
        mock_model_info,
        mock_recommended,
        client,
    ):
        """Includes model info in response when available."""
        from api.schemas import ModelInfo

        mock_vmem.return_value = MagicMock(
            available=8 * BYTES_PER_GB,
            used=8 * BYTES_PER_GB,
            total=16 * BYTES_PER_GB,
        )
        mock_process_mem.return_value = (256.0, 512.0)
        mock_imessage.return_value = True
        mock_model_loaded.return_value = True
        mock_model_info.return_value = ModelInfo(
            id="test-model",
            display_name="Test Model",
            loaded=True,
            memory_usage_mb=1024.0,
            quality_tier="good",
        )
        mock_recommended.return_value = "test-model"

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] is not None
        assert data["model"]["id"] == "test-model"
        assert data["model"]["display_name"] == "Test Model"
        assert data["model"]["loaded"] is True
        assert data["model"]["memory_usage_mb"] == 1024.0


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_returns_ok_status(self, client):
        """Returns simple ok status and service name."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "jarvis-api"


class TestHealthSchemas:
    """Tests for health-related Pydantic schemas."""

    def test_health_response_schema(self):
        """HealthResponse schema works correctly."""
        from api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            imessage_access=True,
            memory_available_gb=8.0,
            memory_used_gb=8.0,
            memory_mode="FULL",
            model_loaded=True,
            permissions_ok=True,
        )
        assert response.status == "healthy"
        assert response.memory_mode == "FULL"
        assert response.details is None

    def test_health_response_with_details(self):
        """HealthResponse schema handles details."""
        from api.schemas import HealthResponse

        response = HealthResponse(
            status="unhealthy",
            imessage_access=False,
            memory_available_gb=1.0,
            memory_used_gb=7.0,
            memory_mode="MINIMAL",
            model_loaded=False,
            permissions_ok=False,
            details={"imessage": "Access denied"},
        )
        assert response.details == {"imessage": "Access denied"}

    def test_model_info_schema(self):
        """ModelInfo schema works correctly."""
        from api.schemas import ModelInfo

        info = ModelInfo(
            id="test-model",
            display_name="Test Model",
            loaded=True,
            memory_usage_mb=512.0,
            quality_tier="good",
        )
        assert info.id == "test-model"
        assert info.quality_tier == "good"

    def test_model_info_optional_fields(self):
        """ModelInfo schema handles optional fields."""
        from api.schemas import ModelInfo

        info = ModelInfo(
            display_name="Test",
            loaded=False,
            memory_usage_mb=0.0,
        )
        assert info.id is None
        assert info.quality_tier is None
