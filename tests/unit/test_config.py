"""Unit tests for JARVIS Configuration System.

Tests cover loading configuration from file, handling missing/invalid files,
validation of threshold ranges, and singleton behavior.
"""

import json
from pathlib import Path

import pytest

from jarvis.config import (
    CONFIG_PATH,
    JarvisConfig,
    MemoryThresholds,
    get_config,
    load_config,
    reset_config,
)


class TestMemoryThresholds:
    """Tests for MemoryThresholds model."""

    def test_default_values(self):
        """Test default memory threshold values."""
        thresholds = MemoryThresholds()
        assert thresholds.full_mode_mb == 8000
        assert thresholds.lite_mode_mb == 4000

    def test_custom_values(self):
        """Test custom memory threshold values."""
        thresholds = MemoryThresholds(full_mode_mb=16000, lite_mode_mb=8000)
        assert thresholds.full_mode_mb == 16000
        assert thresholds.lite_mode_mb == 8000


class TestJarvisConfig:
    """Tests for JarvisConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JarvisConfig()
        assert config.model_path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert config.template_similarity_threshold == 0.7
        assert config.memory_thresholds.full_mode_mb == 8000
        assert config.memory_thresholds.lite_mode_mb == 4000
        assert config.imessage_default_limit == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JarvisConfig(
            model_path="custom/model",
            template_similarity_threshold=0.8,
            memory_thresholds=MemoryThresholds(full_mode_mb=12000, lite_mode_mb=6000),
            imessage_default_limit=100,
        )
        assert config.model_path == "custom/model"
        assert config.template_similarity_threshold == 0.8
        assert config.memory_thresholds.full_mode_mb == 12000
        assert config.imessage_default_limit == 100

    def test_threshold_at_boundaries(self):
        """Test threshold at valid boundary values."""
        config_min = JarvisConfig(template_similarity_threshold=0.0)
        assert config_min.template_similarity_threshold == 0.0

        config_max = JarvisConfig(template_similarity_threshold=1.0)
        assert config_max.template_similarity_threshold == 1.0

    def test_threshold_out_of_range_raises(self):
        """Test that threshold outside 0-1 range raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JarvisConfig(template_similarity_threshold=-0.1)

        with pytest.raises(ValidationError):
            JarvisConfig(template_similarity_threshold=1.1)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_defaults_when_missing(self, tmp_path):
        """Test that load_config returns defaults when file doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent" / "config.json"
        config = load_config(nonexistent_path)

        assert config.model_path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert config.template_similarity_threshold == 0.7
        assert config.memory_thresholds.full_mode_mb == 8000
        assert config.imessage_default_limit == 50

    def test_load_config_reads_from_file(self, tmp_path):
        """Test that load_config reads configuration from file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "model_path": "custom/model-path",
            "template_similarity_threshold": 0.85,
            "memory_thresholds": {
                "full_mode_mb": 10000,
                "lite_mode_mb": 5000,
            },
            "imessage_default_limit": 75,
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)

        assert config.model_path == "custom/model-path"
        assert config.template_similarity_threshold == 0.85
        assert config.memory_thresholds.full_mode_mb == 10000
        assert config.memory_thresholds.lite_mode_mb == 5000
        assert config.imessage_default_limit == 75

    def test_load_config_handles_invalid_json(self, tmp_path):
        """Test that load_config returns defaults for invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json content")

        config = load_config(config_file)

        # Should return defaults
        assert config.model_path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert config.template_similarity_threshold == 0.7

    def test_load_config_validates_threshold_range(self, tmp_path):
        """Test that load_config returns defaults when threshold is out of range."""
        config_file = tmp_path / "config.json"

        # Test threshold too high
        config_data = {"template_similarity_threshold": 1.5}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)
        # Should return defaults due to validation error
        assert config.template_similarity_threshold == 0.7

        # Test threshold too low
        config_data = {"template_similarity_threshold": -0.5}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)
        # Should return defaults due to validation error
        assert config.template_similarity_threshold == 0.7

    def test_load_config_partial_file(self, tmp_path):
        """Test that load_config merges partial config with defaults."""
        config_file = tmp_path / "config.json"
        config_data = {
            "model_path": "partial/model",
            # Other fields omitted - should use defaults
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)

        assert config.model_path == "partial/model"
        assert config.template_similarity_threshold == 0.7  # default
        assert config.memory_thresholds.full_mode_mb == 8000  # default
        assert config.imessage_default_limit == 50  # default

    def test_load_config_empty_file(self, tmp_path):
        """Test that load_config handles empty JSON object."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        config = load_config(config_file)

        # Should use all defaults
        assert config.model_path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert config.template_similarity_threshold == 0.7


class TestGetConfig:
    """Tests for get_config singleton function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_config()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_config()

    def test_get_config_returns_singleton(self, tmp_path, monkeypatch):
        """Test that get_config returns the same instance on multiple calls."""
        # Point to a nonexistent path to get defaults
        monkeypatch.setattr("jarvis.config.CONFIG_PATH", tmp_path / "config.json")

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_loads_from_default_path(self, tmp_path, monkeypatch):
        """Test that get_config loads from CONFIG_PATH."""
        config_file = tmp_path / "config.json"
        config_data = {"model_path": "singleton/model"}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        monkeypatch.setattr("jarvis.config.CONFIG_PATH", config_file)

        config = get_config()

        assert config.model_path == "singleton/model"


class TestResetConfig:
    """Tests for reset_config function."""

    def test_reset_config_clears_singleton(self, tmp_path, monkeypatch):
        """Test that reset_config clears the singleton instance."""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr("jarvis.config.CONFIG_PATH", config_file)

        # Create first config
        config_data = {"model_path": "first/model"}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config1 = get_config()
        assert config1.model_path == "first/model"

        # Reset and create new config
        reset_config()

        # Update config file
        config_data = {"model_path": "second/model"}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config2 = get_config()

        # Should be a new instance with new values
        assert config1 is not config2
        assert config2.model_path == "second/model"

    def test_reset_config_allows_reinitialization(self, tmp_path, monkeypatch):
        """Test that reset_config allows complete reinitialization."""
        monkeypatch.setattr("jarvis.config.CONFIG_PATH", tmp_path / "config.json")

        # Initialize with defaults
        config1 = get_config()

        # Reset
        reset_config()

        # Create config file
        config_file = tmp_path / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with config_file.open("w") as f:
            json.dump({"imessage_default_limit": 200}, f)

        # Get new config
        config2 = get_config()

        assert config2.imessage_default_limit == 200
        assert config1 is not config2


class TestConfigPath:
    """Tests for CONFIG_PATH constant."""

    def test_config_path_is_in_home_directory(self):
        """Test that CONFIG_PATH points to ~/.jarvis/config.json."""
        expected = Path.home() / ".jarvis" / "config.json"
        assert CONFIG_PATH == expected
