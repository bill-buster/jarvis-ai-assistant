"""Unit tests for the settings API endpoint.

Tests configuration management including model selection, generation parameters,
and behavior preferences with mocked external dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.settings import (
    AVAILABLE_MODELS,
    _check_imessage_access,
    _check_model_downloaded,
    _check_model_loaded,
    _get_default_settings,
    _get_recommended_model,
    _get_system_info,
    _load_settings,
    _save_settings,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def temp_settings_file():
    """Create a temporary settings file location."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "settings.json"


class TestAvailableModels:
    """Tests for AVAILABLE_MODELS constant."""

    def test_has_required_fields(self):
        """Each model has all required fields."""
        required_fields = {
            "model_id",
            "name",
            "size_gb",
            "quality_tier",
            "ram_requirement_gb",
            "description",
        }
        for model in AVAILABLE_MODELS:
            assert required_fields.issubset(model.keys())

    def test_has_at_least_one_model(self):
        """At least one model is available."""
        assert len(AVAILABLE_MODELS) >= 1

    def test_quality_tiers_are_valid(self):
        """Quality tiers are one of expected values."""
        valid_tiers = {"basic", "good", "best"}
        for model in AVAILABLE_MODELS:
            assert model["quality_tier"] in valid_tiers


class TestGetDefaultSettings:
    """Tests for _get_default_settings helper."""

    def test_returns_generation_settings(self):
        """Returns generation settings with expected keys."""
        defaults = _get_default_settings()
        assert "generation" in defaults
        gen = defaults["generation"]
        assert "temperature" in gen
        assert "max_tokens_reply" in gen
        assert "max_tokens_summary" in gen

    def test_returns_behavior_settings(self):
        """Returns behavior settings with expected keys."""
        defaults = _get_default_settings()
        assert "behavior" in defaults
        behav = defaults["behavior"]
        assert "auto_suggest_replies" in behav
        assert "suggestion_count" in behav
        assert "context_messages_reply" in behav
        assert "context_messages_summary" in behav

    def test_default_temperature(self):
        """Default temperature is 0.7."""
        defaults = _get_default_settings()
        assert defaults["generation"]["temperature"] == 0.7


class TestLoadSettings:
    """Tests for _load_settings helper."""

    @patch("api.routers.settings.SETTINGS_PATH")
    def test_returns_defaults_when_file_missing(self, mock_path):
        """Returns default settings when file doesn't exist."""
        mock_path.exists.return_value = False

        result = _load_settings()

        assert result == _get_default_settings()

    @patch("api.routers.settings.SETTINGS_PATH")
    def test_loads_settings_from_file(self, mock_path, temp_settings_file):
        """Loads settings from file when it exists."""
        settings = {"generation": {"temperature": 0.5}, "behavior": {"auto_suggest_replies": False}}
        temp_settings_file.parent.mkdir(parents=True, exist_ok=True)
        temp_settings_file.write_text(json.dumps(settings))

        with patch("api.routers.settings.SETTINGS_PATH", temp_settings_file):
            result = _load_settings()

        assert result["generation"]["temperature"] == 0.5
        assert result["behavior"]["auto_suggest_replies"] is False

    @patch("api.routers.settings.SETTINGS_PATH")
    def test_returns_defaults_on_json_decode_error(self, mock_path, temp_settings_file):
        """Returns defaults when file contains invalid JSON."""
        temp_settings_file.parent.mkdir(parents=True, exist_ok=True)
        temp_settings_file.write_text("not valid json {{{")

        with patch("api.routers.settings.SETTINGS_PATH", temp_settings_file):
            result = _load_settings()

        assert result == _get_default_settings()

    @patch("api.routers.settings.SETTINGS_PATH")
    def test_returns_defaults_on_os_error(self, mock_path):
        """Returns defaults when file read fails."""
        mock_path.exists.return_value = True
        mock_path.open.side_effect = OSError("Read error")

        result = _load_settings()

        assert result == _get_default_settings()


class TestSaveSettings:
    """Tests for _save_settings helper."""

    def test_saves_settings_to_file(self, temp_settings_file):
        """Saves settings to file successfully."""
        settings = {"generation": {"temperature": 0.8}}

        with patch("api.routers.settings.SETTINGS_PATH", temp_settings_file):
            result = _save_settings(settings)

        assert result is True
        saved_data = json.loads(temp_settings_file.read_text())
        assert saved_data["generation"]["temperature"] == 0.8

    def test_creates_parent_directories(self, temp_settings_file):
        """Creates parent directories if they don't exist."""
        nested_path = temp_settings_file.parent / "nested" / "dir" / "settings.json"
        settings = {"test": "value"}

        with patch("api.routers.settings.SETTINGS_PATH", nested_path):
            result = _save_settings(settings)

        assert result is True
        assert nested_path.exists()

    @patch("api.routers.settings.SETTINGS_PATH")
    def test_returns_false_on_error(self, mock_path):
        """Returns False when save fails."""
        mock_path.parent.mkdir.side_effect = OSError("Permission denied")

        result = _save_settings({"test": "value"})

        assert result is False


class TestCheckModelDownloaded:
    """Tests for _check_model_downloaded helper."""

    def test_returns_true_when_model_in_cache(self, temp_settings_file):
        """Returns True when model exists in cache."""
        # Create a mock cache directory structure
        cache_dir = temp_settings_file.parent / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--mlx-community--test-model"
        model_dir.mkdir(parents=True)

        with patch.object(Path, "home", return_value=temp_settings_file.parent):
            result = _check_model_downloaded("mlx-community/test-model")

        assert result is True

    def test_returns_false_when_model_not_in_cache(self, temp_settings_file):
        """Returns False when model doesn't exist in cache."""
        # Create cache dir without the model
        cache_dir = temp_settings_file.parent / ".cache" / "huggingface" / "hub"
        cache_dir.mkdir(parents=True)

        with patch.object(Path, "home", return_value=temp_settings_file.parent):
            result = _check_model_downloaded("nonexistent/model")

        assert result is False

    def test_checks_alternate_cache_location(self, temp_settings_file):
        """Checks alternate huggingface cache location."""
        # Create alternate cache directory
        alt_cache = temp_settings_file.parent / ".huggingface" / "hub"
        model_dir = alt_cache / "models--mlx-community--alt-model"
        model_dir.mkdir(parents=True)

        with patch.object(Path, "home", return_value=temp_settings_file.parent):
            result = _check_model_downloaded("mlx-community/alt-model")

        assert result is True


class TestCheckModelLoaded:
    """Tests for _check_model_loaded helper."""

    @patch("api.routers.settings.get_config")
    @patch("models.get_generator")
    def test_returns_true_when_model_loaded_and_matches(self, mock_get_gen, mock_get_config):
        """Returns True when the specific model is loaded."""
        mock_generator = MagicMock()
        mock_generator._model = "loaded_model"
        mock_get_gen.return_value = mock_generator

        mock_config = MagicMock()
        mock_config.model_path = "test/model"
        mock_get_config.return_value = mock_config

        result = _check_model_loaded("test/model")

        assert result is True

    @patch("api.routers.settings.get_config")
    @patch("models.get_generator")
    def test_returns_false_when_different_model_loaded(self, mock_get_gen, mock_get_config):
        """Returns False when a different model is loaded."""
        mock_generator = MagicMock()
        mock_generator._model = "loaded_model"
        mock_get_gen.return_value = mock_generator

        mock_config = MagicMock()
        mock_config.model_path = "other/model"
        mock_get_config.return_value = mock_config

        result = _check_model_loaded("test/model")

        assert result is False

    @patch("models.get_generator")
    def test_returns_false_when_no_model_loaded(self, mock_get_gen):
        """Returns False when no model is loaded."""
        mock_generator = MagicMock()
        mock_generator._model = None
        mock_get_gen.return_value = mock_generator

        result = _check_model_loaded("test/model")

        assert result is False

    @patch("models.get_generator")
    def test_returns_false_on_exception(self, mock_get_gen):
        """Returns False when getting generator fails."""
        mock_get_gen.side_effect = RuntimeError("Model error")

        result = _check_model_loaded("test/model")

        assert result is False


class TestCheckImessageAccess:
    """Tests for _check_imessage_access helper."""

    @patch("api.routers.settings.ChatDBReader")
    def test_returns_true_when_accessible(self, mock_reader_class):
        """Returns True when iMessage is accessible."""
        mock_reader = MagicMock()
        mock_reader.check_access.return_value = True
        mock_reader_class.return_value = mock_reader

        result = _check_imessage_access()

        assert result is True
        mock_reader.close.assert_called_once()

    @patch("api.routers.settings.ChatDBReader")
    def test_returns_false_when_not_accessible(self, mock_reader_class):
        """Returns False when iMessage is not accessible."""
        mock_reader = MagicMock()
        mock_reader.check_access.return_value = False
        mock_reader_class.return_value = mock_reader

        result = _check_imessage_access()

        assert result is False

    @patch("api.routers.settings.ChatDBReader")
    def test_returns_false_on_exception(self, mock_reader_class):
        """Returns False when checking access throws exception."""
        mock_reader_class.side_effect = RuntimeError("DB error")

        result = _check_imessage_access()

        assert result is False


class TestGetSystemInfo:
    """Tests for _get_system_info helper."""

    @patch("api.routers.settings._check_imessage_access")
    @patch("models.get_generator")
    @patch("api.routers.settings.psutil.virtual_memory")
    def test_returns_system_info_with_model_loaded(self, mock_vmem, mock_get_gen, mock_imessage):
        """Returns SystemInfo with model loaded information."""
        mock_vmem.return_value = MagicMock(
            total=16 * (1024**3),
            used=8 * (1024**3),
        )
        mock_generator = MagicMock()
        mock_generator._model = "loaded"
        mock_generator.config.estimated_memory_mb = 2048
        mock_get_gen.return_value = mock_generator
        mock_imessage.return_value = True

        result = _get_system_info()

        assert result.system_ram_gb == 16.0
        assert result.current_memory_usage_gb == 8.0
        assert result.model_loaded is True
        assert result.model_memory_usage_gb == 2.0
        assert result.imessage_access is True

    @patch("api.routers.settings._check_imessage_access")
    @patch("models.get_generator")
    @patch("api.routers.settings.psutil.virtual_memory")
    def test_returns_system_info_without_model(self, mock_vmem, mock_get_gen, mock_imessage):
        """Returns SystemInfo when no model is loaded."""
        mock_vmem.return_value = MagicMock(
            total=8 * (1024**3),
            used=4 * (1024**3),
        )
        mock_generator = MagicMock()
        mock_generator._model = None
        mock_get_gen.return_value = mock_generator
        mock_imessage.return_value = False

        result = _get_system_info()

        assert result.system_ram_gb == 8.0
        assert result.model_loaded is False
        assert result.model_memory_usage_gb == 0.0
        assert result.imessage_access is False

    @patch("api.routers.settings._check_imessage_access")
    @patch("models.get_generator")
    @patch("api.routers.settings.psutil.virtual_memory")
    def test_handles_generator_exception(self, mock_vmem, mock_get_gen, mock_imessage):
        """Handles exception when getting generator."""
        mock_vmem.return_value = MagicMock(
            total=8 * (1024**3),
            used=4 * (1024**3),
        )
        mock_get_gen.side_effect = RuntimeError("Model error")
        mock_imessage.return_value = True

        result = _get_system_info()

        assert result.model_loaded is False
        assert result.model_memory_usage_gb == 0.0


class TestGetRecommendedModel:
    """Tests for _get_recommended_model helper."""

    @patch("api.routers.settings.psutil.virtual_memory")
    def test_recommends_smallest_for_low_ram(self, mock_vmem):
        """Recommends smallest model for low RAM systems."""
        mock_vmem.return_value = MagicMock(total=4 * (1024**3))

        result = _get_recommended_model()

        # Should recommend 0.5B model
        assert "0.5B" in result or result == AVAILABLE_MODELS[0]["model_id"]

    @patch("api.routers.settings.psutil.virtual_memory")
    def test_recommends_larger_for_high_ram(self, mock_vmem):
        """Recommends larger model for high RAM systems."""
        mock_vmem.return_value = MagicMock(total=32 * (1024**3))

        result = _get_recommended_model()

        # Should recommend the largest model that fits
        assert result in [m["model_id"] for m in AVAILABLE_MODELS]


class TestGetSettingsEndpoint:
    """Tests for GET /settings endpoint."""

    @patch("api.routers.settings._get_system_info")
    @patch("api.routers.settings._load_settings")
    @patch("api.routers.settings.get_config")
    def test_returns_current_settings(self, mock_config, mock_load, mock_sys_info, client):
        """Returns current settings successfully."""
        from api.schemas import SystemInfo

        mock_cfg = MagicMock()
        mock_cfg.model_path = "test/model"
        mock_config.return_value = mock_cfg

        mock_load.return_value = _get_default_settings()

        mock_sys_info.return_value = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=True,
            model_memory_usage_gb=1.0,
            imessage_access=True,
        )

        response = client.get("/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test/model"
        assert "generation" in data
        assert "behavior" in data
        assert "system" in data
        assert data["system"]["system_ram_gb"] == 16.0


class TestUpdateSettingsEndpoint:
    """Tests for PUT /settings endpoint."""

    @patch("api.routers.settings._get_system_info")
    @patch("api.routers.settings._save_settings")
    @patch("api.routers.settings._load_settings")
    @patch("api.routers.settings.save_config")
    @patch("api.routers.settings.get_config")
    def test_updates_model_id(
        self, mock_get_config, mock_save_config, mock_load, mock_save, mock_sys_info, client
    ):
        """Updates model_id when provided."""
        from api.schemas import SystemInfo

        mock_cfg = MagicMock()
        mock_cfg.model_path = "old/model"
        mock_get_config.return_value = mock_cfg
        mock_load.return_value = _get_default_settings()
        mock_save.return_value = True
        mock_sys_info.return_value = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=True,
            model_memory_usage_gb=1.0,
            imessage_access=True,
        )

        response = client.put(
            "/settings",
            json={"model_id": AVAILABLE_MODELS[0]["model_id"]},
        )

        assert response.status_code == 200
        mock_save_config.assert_called_once()

    @patch("api.routers.settings._get_system_info")
    @patch("api.routers.settings._save_settings")
    @patch("api.routers.settings._load_settings")
    @patch("api.routers.settings.get_config")
    def test_updates_generation_settings(
        self, mock_config, mock_load, mock_save, mock_sys_info, client
    ):
        """Updates generation settings when provided."""
        from api.schemas import SystemInfo

        mock_cfg = MagicMock()
        mock_cfg.model_path = "test/model"
        mock_config.return_value = mock_cfg
        mock_load.return_value = _get_default_settings()
        mock_save.return_value = True
        mock_sys_info.return_value = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=False,
            model_memory_usage_gb=0.0,
            imessage_access=True,
        )

        response = client.put(
            "/settings",
            json={
                "generation": {
                    "temperature": 0.5,
                    "max_tokens_reply": 200,
                    "max_tokens_summary": 600,
                }
            },
        )

        assert response.status_code == 200
        # Verify save was called with updated settings
        mock_save.assert_called_once()
        saved_settings = mock_save.call_args[0][0]
        assert saved_settings["generation"]["temperature"] == 0.5

    @patch("api.routers.settings._get_system_info")
    @patch("api.routers.settings._save_settings")
    @patch("api.routers.settings._load_settings")
    @patch("api.routers.settings.get_config")
    def test_updates_behavior_settings(
        self, mock_config, mock_load, mock_save, mock_sys_info, client
    ):
        """Updates behavior settings when provided."""
        from api.schemas import SystemInfo

        mock_cfg = MagicMock()
        mock_cfg.model_path = "test/model"
        mock_config.return_value = mock_cfg
        mock_load.return_value = _get_default_settings()
        mock_save.return_value = True
        mock_sys_info.return_value = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=False,
            model_memory_usage_gb=0.0,
            imessage_access=True,
        )

        response = client.put(
            "/settings",
            json={
                "behavior": {
                    "auto_suggest_replies": False,
                    "suggestion_count": 5,
                    "context_messages_reply": 30,
                    "context_messages_summary": 75,
                }
            },
        )

        assert response.status_code == 200
        mock_save.assert_called_once()
        saved_settings = mock_save.call_args[0][0]
        assert saved_settings["behavior"]["auto_suggest_replies"] is False

    @patch("api.routers.settings.get_config")
    def test_rejects_unknown_model_id(self, mock_config, client):
        """Returns 400 for unknown model_id."""
        mock_cfg = MagicMock()
        mock_cfg.model_path = "test/model"
        mock_config.return_value = mock_cfg

        response = client.put(
            "/settings",
            json={"model_id": "nonexistent/model"},
        )

        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]

    @patch("api.routers.settings._get_system_info")
    @patch("api.routers.settings._save_settings")
    @patch("api.routers.settings._load_settings")
    @patch("api.routers.settings.get_config")
    def test_partial_update(self, mock_config, mock_load, mock_save, mock_sys_info, client):
        """Supports partial updates - only updates provided fields."""
        from api.schemas import SystemInfo

        mock_cfg = MagicMock()
        mock_cfg.model_path = "test/model"
        mock_config.return_value = mock_cfg

        original_settings = _get_default_settings()
        mock_load.return_value = original_settings.copy()
        mock_save.return_value = True
        mock_sys_info.return_value = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=False,
            model_memory_usage_gb=0.0,
            imessage_access=True,
        )

        # Only update generation, not behavior
        response = client.put(
            "/settings",
            json={
                "generation": {
                    "temperature": 0.9,
                    "max_tokens_reply": 100,
                    "max_tokens_summary": 400,
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Generation should be updated
        assert data["generation"]["temperature"] == 0.9
        # Behavior should remain default
        assert data["behavior"]["auto_suggest_replies"] is True


class TestListModelsEndpoint:
    """Tests for GET /settings/models endpoint."""

    @patch("api.routers.settings._check_model_loaded")
    @patch("api.routers.settings._check_model_downloaded")
    @patch("api.routers.settings._get_recommended_model")
    def test_returns_all_models(self, mock_recommended, mock_downloaded, mock_loaded, client):
        """Returns list of all available models."""
        mock_recommended.return_value = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = False
        mock_loaded.return_value = False

        response = client.get("/settings/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(AVAILABLE_MODELS)

    @patch("api.routers.settings._check_model_loaded")
    @patch("api.routers.settings._check_model_downloaded")
    @patch("api.routers.settings._get_recommended_model")
    def test_includes_download_status(self, mock_recommended, mock_downloaded, mock_loaded, client):
        """Includes download status for each model."""
        mock_recommended.return_value = AVAILABLE_MODELS[0]["model_id"]
        # First model downloaded, others not
        mock_downloaded.side_effect = lambda m: m == AVAILABLE_MODELS[0]["model_id"]
        mock_loaded.return_value = False

        response = client.get("/settings/models")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["is_downloaded"] is True
        for model in data[1:]:
            assert model["is_downloaded"] is False

    @patch("api.routers.settings._check_model_loaded")
    @patch("api.routers.settings._check_model_downloaded")
    @patch("api.routers.settings._get_recommended_model")
    def test_includes_loaded_status(self, mock_recommended, mock_downloaded, mock_loaded, client):
        """Includes loaded status for each model."""
        mock_recommended.return_value = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = True
        # First model loaded
        mock_loaded.side_effect = lambda m: m == AVAILABLE_MODELS[0]["model_id"]

        response = client.get("/settings/models")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["is_loaded"] is True
        for model in data[1:]:
            assert model["is_loaded"] is False

    @patch("api.routers.settings._check_model_loaded")
    @patch("api.routers.settings._check_model_downloaded")
    @patch("api.routers.settings._get_recommended_model")
    def test_marks_recommended_model(self, mock_recommended, mock_downloaded, mock_loaded, client):
        """Marks the recommended model."""
        recommended_id = AVAILABLE_MODELS[1]["model_id"]
        mock_recommended.return_value = recommended_id
        mock_downloaded.return_value = False
        mock_loaded.return_value = False

        response = client.get("/settings/models")

        assert response.status_code == 200
        data = response.json()
        for model in data:
            if model["model_id"] == recommended_id:
                assert model["is_recommended"] is True
            else:
                assert model["is_recommended"] is False


class TestDownloadModelEndpoint:
    """Tests for POST /settings/models/{model_id}/download endpoint."""

    @patch("api.routers.settings._check_model_downloaded")
    def test_returns_completed_if_already_downloaded(self, mock_downloaded, client):
        """Returns completed status if model already downloaded."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = True

        response = client.post(f"/settings/models/{model_id}/download")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100.0

    def test_returns_404_for_unknown_model(self, client):
        """Returns 404 for unknown model ID."""
        response = client.post("/settings/models/unknown/model/download")

        assert response.status_code == 404
        assert "Unknown model" in response.json()["detail"]

    @patch("huggingface_hub.snapshot_download")
    @patch("api.routers.settings._check_model_downloaded")
    def test_downloads_model_successfully(self, mock_downloaded, mock_snapshot, client):
        """Downloads model successfully."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = False
        mock_snapshot.return_value = "/path/to/model"

        response = client.post(f"/settings/models/{model_id}/download")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100.0
        mock_snapshot.assert_called_once_with(model_id)

    @patch("huggingface_hub.snapshot_download")
    @patch("api.routers.settings._check_model_downloaded")
    def test_returns_failed_on_download_error(self, mock_downloaded, mock_snapshot, client):
        """Returns failed status on download error."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = False
        mock_snapshot.side_effect = RuntimeError("Network error")

        response = client.post(f"/settings/models/{model_id}/download")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["progress"] == 0.0
        assert "Network error" in data["error"]


class TestActivateModelEndpoint:
    """Tests for POST /settings/models/{model_id}/activate endpoint."""

    def test_returns_404_for_unknown_model(self, client):
        """Returns 404 for unknown model ID."""
        response = client.post("/settings/models/unknown/model/activate")

        assert response.status_code == 404
        assert "Unknown model" in response.json()["detail"]

    @patch("api.routers.settings._check_model_downloaded")
    def test_fails_if_not_downloaded(self, mock_downloaded, client):
        """Returns error if model not downloaded."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = False

        response = client.post(f"/settings/models/{model_id}/activate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not downloaded" in data["error"]

    @patch("models.reset_generator")
    @patch("api.routers.settings.save_config")
    @patch("api.routers.settings.get_config")
    @patch("api.routers.settings._check_model_downloaded")
    def test_activates_downloaded_model(
        self, mock_downloaded, mock_get_config, mock_save_config, mock_reset, client
    ):
        """Activates model successfully when downloaded."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = True

        mock_cfg = MagicMock()
        mock_get_config.return_value = mock_cfg

        response = client.post(f"/settings/models/{model_id}/activate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == model_id
        assert mock_cfg.model_path == model_id
        mock_save_config.assert_called_once()
        mock_reset.assert_called_once()

    @patch("models.reset_generator")
    @patch("api.routers.settings.save_config")
    @patch("api.routers.settings.get_config")
    @patch("api.routers.settings._check_model_downloaded")
    def test_handles_activation_error(
        self, mock_downloaded, mock_get_config, mock_save_config, mock_reset, client
    ):
        """Returns error on activation failure."""
        model_id = AVAILABLE_MODELS[0]["model_id"]
        mock_downloaded.return_value = True
        mock_get_config.side_effect = RuntimeError("Config error")

        response = client.post(f"/settings/models/{model_id}/activate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Config error" in data["error"]


class TestSettingsSchemas:
    """Tests for settings-related Pydantic schemas."""

    def test_generation_settings_defaults(self):
        """GenerationSettings has correct defaults."""
        from api.schemas import GenerationSettings

        settings = GenerationSettings()
        assert settings.temperature == 0.7
        assert settings.max_tokens_reply == 150
        assert settings.max_tokens_summary == 500

    def test_generation_settings_validation(self):
        """GenerationSettings validates ranges."""
        from pydantic import ValidationError

        from api.schemas import GenerationSettings

        with pytest.raises(ValidationError):
            GenerationSettings(temperature=1.5)  # Too high

        with pytest.raises(ValidationError):
            GenerationSettings(max_tokens_reply=10)  # Too low

    def test_behavior_settings_defaults(self):
        """BehaviorSettings has correct defaults."""
        from api.schemas import BehaviorSettings

        settings = BehaviorSettings()
        assert settings.auto_suggest_replies is True
        assert settings.suggestion_count == 3
        assert settings.context_messages_reply == 20
        assert settings.context_messages_summary == 50

    def test_behavior_settings_validation(self):
        """BehaviorSettings validates ranges."""
        from pydantic import ValidationError

        from api.schemas import BehaviorSettings

        with pytest.raises(ValidationError):
            BehaviorSettings(suggestion_count=10)  # Too high

        with pytest.raises(ValidationError):
            BehaviorSettings(context_messages_reply=5)  # Too low

    def test_system_info_schema(self):
        """SystemInfo schema works correctly."""
        from api.schemas import SystemInfo

        info = SystemInfo(
            system_ram_gb=16.0,
            current_memory_usage_gb=8.0,
            model_loaded=True,
            model_memory_usage_gb=2.0,
            imessage_access=True,
        )
        assert info.system_ram_gb == 16.0
        assert info.model_loaded is True

    def test_available_model_info_schema(self):
        """AvailableModelInfo schema works correctly."""
        from api.schemas import AvailableModelInfo

        model = AvailableModelInfo(
            model_id="test/model",
            name="Test Model",
            size_gb=1.0,
            quality_tier="good",
            ram_requirement_gb=8.0,
            is_downloaded=True,
            is_loaded=False,
            is_recommended=True,
            description="A test model",
        )
        assert model.model_id == "test/model"
        assert model.is_recommended is True

    def test_download_status_schema(self):
        """DownloadStatus schema works correctly."""
        from api.schemas import DownloadStatus

        status = DownloadStatus(
            model_id="test/model",
            status="completed",
            progress=100.0,
        )
        assert status.status == "completed"
        assert status.error is None

    def test_download_status_with_error(self):
        """DownloadStatus schema handles errors."""
        from api.schemas import DownloadStatus

        status = DownloadStatus(
            model_id="test/model",
            status="failed",
            progress=0.0,
            error="Network error",
        )
        assert status.status == "failed"
        assert status.error == "Network error"

    def test_activate_response_schema(self):
        """ActivateResponse schema works correctly."""
        from api.schemas import ActivateResponse

        response = ActivateResponse(
            success=True,
            model_id="test/model",
        )
        assert response.success is True
        assert response.error is None

    def test_settings_update_request_all_optional(self):
        """SettingsUpdateRequest has all optional fields."""
        from api.schemas import SettingsUpdateRequest

        request = SettingsUpdateRequest()
        assert request.model_id is None
        assert request.generation is None
        assert request.behavior is None
