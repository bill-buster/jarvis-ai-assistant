"""Unit tests for JARVIS Configuration System.

Tests cover loading configuration from file, handling missing/invalid files,
validation of threshold ranges, singleton behavior, UI/Search/Chat preferences,
config migration, and save functionality.
"""

import json
from pathlib import Path

import pytest

from jarvis.config import (
    CONFIG_PATH,
    CONFIG_VERSION,
    ChatConfig,
    JarvisConfig,
    MemoryThresholds,
    SearchConfig,
    UIConfig,
    get_config,
    load_config,
    reset_config,
    save_config,
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


class TestUIConfig:
    """Tests for UIConfig model."""

    def test_default_values(self):
        """Test default UI configuration values."""
        ui = UIConfig()
        assert ui.theme == "system"
        assert ui.font_size == 14
        assert ui.show_timestamps is True
        assert ui.compact_mode is False

    def test_custom_values(self):
        """Test custom UI configuration values."""
        ui = UIConfig(
            theme="dark",
            font_size=16,
            show_timestamps=False,
            compact_mode=True,
        )
        assert ui.theme == "dark"
        assert ui.font_size == 16
        assert ui.show_timestamps is False
        assert ui.compact_mode is True

    def test_theme_literal_values(self):
        """Test that theme only accepts valid literal values."""
        from pydantic import ValidationError

        UIConfig(theme="light")  # valid
        UIConfig(theme="dark")  # valid
        UIConfig(theme="system")  # valid

        with pytest.raises(ValidationError):
            UIConfig(theme="invalid")

    def test_font_size_boundaries(self):
        """Test font_size validation at boundaries."""
        from pydantic import ValidationError

        UIConfig(font_size=12)  # min valid
        UIConfig(font_size=24)  # max valid

        with pytest.raises(ValidationError):
            UIConfig(font_size=11)  # too small

        with pytest.raises(ValidationError):
            UIConfig(font_size=25)  # too large


class TestSearchConfig:
    """Tests for SearchConfig model."""

    def test_default_values(self):
        """Test default search configuration values."""
        search = SearchConfig()
        assert search.default_limit == 50
        assert search.default_date_range_days is None

    def test_custom_values(self):
        """Test custom search configuration values."""
        search = SearchConfig(
            default_limit=100,
            default_date_range_days=30,
        )
        assert search.default_limit == 100
        assert search.default_date_range_days == 30

    def test_default_limit_boundaries(self):
        """Test default_limit validation at boundaries."""
        from pydantic import ValidationError

        SearchConfig(default_limit=1)  # min valid
        SearchConfig(default_limit=1000)  # max valid

        with pytest.raises(ValidationError):
            SearchConfig(default_limit=0)  # too small

        with pytest.raises(ValidationError):
            SearchConfig(default_limit=1001)  # too large

    def test_date_range_days_optional(self):
        """Test that date_range_days can be None or positive."""
        from pydantic import ValidationError

        SearchConfig(default_date_range_days=None)  # valid
        SearchConfig(default_date_range_days=1)  # min valid
        SearchConfig(default_date_range_days=365)  # valid

        with pytest.raises(ValidationError):
            SearchConfig(default_date_range_days=0)  # too small


class TestChatConfig:
    """Tests for ChatConfig model."""

    def test_default_values(self):
        """Test default chat configuration values."""
        chat = ChatConfig()
        assert chat.stream_responses is True
        assert chat.show_typing_indicator is True

    def test_custom_values(self):
        """Test custom chat configuration values."""
        chat = ChatConfig(
            stream_responses=False,
            show_typing_indicator=False,
        )
        assert chat.stream_responses is False
        assert chat.show_typing_indicator is False


class TestJarvisConfigExtended:
    """Tests for extended JarvisConfig with UI/Search/Chat sections."""

    def test_default_nested_configs(self):
        """Test that nested configs have correct defaults."""
        config = JarvisConfig()
        assert config.config_version == CONFIG_VERSION
        assert config.ui.theme == "system"
        assert config.ui.font_size == 14
        assert config.search.default_limit == 50
        assert config.chat.stream_responses is True

    def test_custom_nested_configs(self):
        """Test custom nested configuration values."""
        config = JarvisConfig(
            ui=UIConfig(theme="dark", font_size=18),
            search=SearchConfig(default_limit=200, default_date_range_days=7),
            chat=ChatConfig(stream_responses=False),
        )
        assert config.ui.theme == "dark"
        assert config.ui.font_size == 18
        assert config.search.default_limit == 200
        assert config.search.default_date_range_days == 7
        assert config.chat.stream_responses is False

    def test_load_config_with_nested_sections(self, tmp_path):
        """Test loading config with all nested sections."""
        config_file = tmp_path / "config.json"
        config_data = {
            "config_version": CONFIG_VERSION,
            "model_path": "test/model",
            "ui": {
                "theme": "dark",
                "font_size": 16,
                "show_timestamps": False,
                "compact_mode": True,
            },
            "search": {
                "default_limit": 100,
                "default_date_range_days": 14,
            },
            "chat": {
                "stream_responses": False,
                "show_typing_indicator": False,
            },
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)

        assert config.model_path == "test/model"
        assert config.ui.theme == "dark"
        assert config.ui.font_size == 16
        assert config.ui.show_timestamps is False
        assert config.ui.compact_mode is True
        assert config.search.default_limit == 100
        assert config.search.default_date_range_days == 14
        assert config.chat.stream_responses is False
        assert config.chat.show_typing_indicator is False


class TestConfigMigration:
    """Tests for config migration from older versions."""

    def test_migrate_v1_to_v2(self, tmp_path):
        """Test migration from v1 (no version) to v2."""
        config_file = tmp_path / "config.json"
        # V1 config: no version field, no ui/search/chat sections
        v1_config = {
            "model_path": "old/model",
            "template_similarity_threshold": 0.8,
            "imessage_default_limit": 75,
        }
        with config_file.open("w") as f:
            json.dump(v1_config, f)

        config = load_config(config_file)

        # Old values preserved
        assert config.model_path == "old/model"
        assert config.template_similarity_threshold == 0.8
        assert config.imessage_default_limit == 75
        # Migrated: imessage_default_limit -> search.default_limit
        assert config.search.default_limit == 75
        # Migrated: template_similarity_threshold -> routing.template_threshold (v8)
        assert config.routing.template_threshold == 0.8
        # New defaults added
        assert config.config_version == CONFIG_VERSION
        assert config.ui.theme == "system"
        assert config.chat.stream_responses is True

    def test_migrate_preserves_partial_sections(self, tmp_path):
        """Test that migration preserves partially filled sections."""
        config_file = tmp_path / "config.json"
        config_data = {
            "model_path": "partial/model",
            "ui": {"theme": "dark"},  # partial - only theme set
            # search and chat missing
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_config(config_file)

        # Preserved value
        assert config.ui.theme == "dark"
        # Defaults for missing fields
        assert config.ui.font_size == 14
        assert config.ui.show_timestamps is True
        # Defaults for missing sections
        assert config.search.default_limit == 50
        assert config.chat.stream_responses is True

    def test_already_v2_not_modified(self, tmp_path):
        """Test that v2 configs are not modified during load."""
        config_file = tmp_path / "config.json"
        v2_config = {
            "config_version": CONFIG_VERSION,
            "model_path": "v2/model",
            "ui": {"theme": "light"},
            "search": {"default_limit": 200},
            "chat": {"stream_responses": False},
        }
        with config_file.open("w") as f:
            json.dump(v2_config, f)

        config = load_config(config_file)

        assert config.config_version == CONFIG_VERSION
        assert config.model_path == "v2/model"
        assert config.ui.theme == "light"
        assert config.search.default_limit == 200
        assert config.chat.stream_responses is False

    def test_migrate_v7_to_v8_template_threshold(self, tmp_path):
        """Test migration of template_similarity_threshold to routing.template_threshold."""
        config_file = tmp_path / "config.json"
        # V7 config with custom template_similarity_threshold
        v7_config = {
            "config_version": 7,
            "model_path": "test/model",
            "template_similarity_threshold": 0.85,  # Non-default value
            "routing": {},  # Empty routing section
        }
        with config_file.open("w") as f:
            json.dump(v7_config, f)

        config = load_config(config_file)

        # Legacy field preserved
        assert config.template_similarity_threshold == 0.85
        # Migrated to routing.template_threshold
        assert config.routing.template_threshold == 0.85
        assert config.config_version == CONFIG_VERSION

    def test_migrate_v7_to_v8_default_threshold_not_migrated(self, tmp_path):
        """Test that default template_similarity_threshold (0.7) is not migrated."""
        config_file = tmp_path / "config.json"
        # V7 config with default template_similarity_threshold
        v7_config = {
            "config_version": 7,
            "model_path": "test/model",
            "template_similarity_threshold": 0.7,  # Default value
            "routing": {},
        }
        with config_file.open("w") as f:
            json.dump(v7_config, f)

        config = load_config(config_file)

        # Default not migrated - routing uses its own default (0.90)
        assert config.template_similarity_threshold == 0.7
        assert config.routing.template_threshold == 0.90
        assert config.config_version == CONFIG_VERSION

    def test_migrate_v7_to_v8_existing_routing_threshold_not_overwritten(self, tmp_path):
        """Test that existing routing.template_threshold is not overwritten by migration."""
        config_file = tmp_path / "config.json"
        # V7 config with both legacy and routing thresholds set
        v7_config = {
            "config_version": 7,
            "model_path": "test/model",
            "template_similarity_threshold": 0.85,
            "routing": {
                "template_threshold": 0.95,  # Already set
            },
        }
        with config_file.open("w") as f:
            json.dump(v7_config, f)

        config = load_config(config_file)

        # Routing threshold preserved, not overwritten by legacy value
        assert config.template_similarity_threshold == 0.85
        assert config.routing.template_threshold == 0.95
        assert config.config_version == CONFIG_VERSION


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config_creates_file(self, tmp_path):
        """Test that save_config creates config file."""
        config_file = tmp_path / "new_config.json"
        config = JarvisConfig(model_path="saved/model")

        result = save_config(config, config_file)

        assert result is True
        assert config_file.exists()

        with config_file.open() as f:
            saved_data = json.load(f)
        assert saved_data["model_path"] == "saved/model"

    def test_save_config_creates_parent_dirs(self, tmp_path):
        """Test that save_config creates parent directories."""
        config_file = tmp_path / "nested" / "dirs" / "config.json"
        config = JarvisConfig()

        result = save_config(config, config_file)

        assert result is True
        assert config_file.exists()

    def test_save_config_overwrites_existing(self, tmp_path):
        """Test that save_config overwrites existing file."""
        config_file = tmp_path / "config.json"

        # Write initial config
        initial = JarvisConfig(model_path="initial/model")
        save_config(initial, config_file)

        # Overwrite with new config
        updated = JarvisConfig(model_path="updated/model")
        save_config(updated, config_file)

        with config_file.open() as f:
            saved_data = json.load(f)
        assert saved_data["model_path"] == "updated/model"

    def test_save_config_includes_all_sections(self, tmp_path):
        """Test that save_config includes all nested sections."""
        config_file = tmp_path / "config.json"
        config = JarvisConfig(
            ui=UIConfig(theme="dark"),
            search=SearchConfig(default_limit=100),
            chat=ChatConfig(stream_responses=False),
        )

        save_config(config, config_file)

        with config_file.open() as f:
            saved_data = json.load(f)

        assert "config_version" in saved_data
        assert saved_data["ui"]["theme"] == "dark"
        assert saved_data["search"]["default_limit"] == 100
        assert saved_data["chat"]["stream_responses"] is False

    def test_save_config_roundtrip(self, tmp_path):
        """Test save then load produces identical config."""
        config_file = tmp_path / "config.json"
        original = JarvisConfig(
            model_path="roundtrip/model",
            template_similarity_threshold=0.85,
            ui=UIConfig(theme="dark", font_size=16, compact_mode=True),
            search=SearchConfig(default_limit=150, default_date_range_days=30),
            chat=ChatConfig(stream_responses=False, show_typing_indicator=False),
        )

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.model_path == original.model_path
        assert loaded.template_similarity_threshold == original.template_similarity_threshold
        assert loaded.ui.theme == original.ui.theme
        assert loaded.ui.font_size == original.ui.font_size
        assert loaded.ui.compact_mode == original.ui.compact_mode
        assert loaded.search.default_limit == original.search.default_limit
        assert loaded.search.default_date_range_days == original.search.default_date_range_days
        assert loaded.chat.stream_responses == original.chat.stream_responses
        assert loaded.chat.show_typing_indicator == original.chat.show_typing_indicator
