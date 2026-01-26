"""JARVIS Configuration System.

Loads and validates configuration from ~/.jarvis/config.json.
Uses Pydantic for schema validation with sensible defaults.

Supports migration from older config versions while preserving existing values.

Usage:
    from jarvis.config import get_config, save_config

    config = get_config()
    print(config.model_path)
    print(config.ui.theme)

    # Modify and save
    config.ui.theme = "dark"
    save_config(config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"

# Current config schema version for migration tracking
CONFIG_VERSION = 2


class MemoryThresholds(BaseModel):
    """Memory thresholds for mode selection."""

    full_mode_mb: int = 8000
    lite_mode_mb: int = 4000


class UIConfig(BaseModel):
    """UI preferences for the Tauri frontend.

    Attributes:
        theme: Color theme preference ("light", "dark", or "system").
        font_size: Font size in pixels (12-24).
        show_timestamps: Whether to show message timestamps.
        compact_mode: Use compact layout with less padding.
    """

    theme: Literal["light", "dark", "system"] = "system"
    font_size: int = Field(default=14, ge=12, le=24)
    show_timestamps: bool = True
    compact_mode: bool = False


class SearchConfig(BaseModel):
    """Search preferences.

    Attributes:
        default_limit: Default number of search results to return.
        default_date_range_days: Default date range for searches (None = no limit).
    """

    default_limit: int = Field(default=50, ge=1, le=1000)
    default_date_range_days: int | None = Field(default=None, ge=1)


class ChatConfig(BaseModel):
    """Chat preferences.

    Attributes:
        stream_responses: Stream responses as they're generated.
        show_typing_indicator: Show typing indicator while generating.
    """

    stream_responses: bool = True
    show_typing_indicator: bool = True


class JarvisConfig(BaseModel):
    """JARVIS configuration schema.

    Attributes:
        config_version: Schema version for migration tracking.
        model_path: HuggingFace model path for MLX inference.
        template_similarity_threshold: Minimum similarity score for template matching (0-1).
        memory_thresholds: Memory thresholds for mode selection.
        imessage_default_limit: Default limit for iMessage search (deprecated).
        ui: UI preferences for the Tauri frontend.
        search: Search preferences.
        chat: Chat preferences.
    """

    config_version: int = CONFIG_VERSION
    model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    template_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_thresholds: MemoryThresholds = Field(default_factory=MemoryThresholds)
    imessage_default_limit: int = 50
    ui: UIConfig = Field(default_factory=UIConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)


# Module-level singleton
_config: JarvisConfig | None = None


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate config data from older versions to current schema.

    Preserves existing values while adding new defaults for missing fields.
    Handles migration from v1 (no version field) to v2 (with ui/search/chat sections).

    Args:
        data: Raw config data loaded from file.

    Returns:
        Migrated config data compatible with current schema.
    """
    version = data.get("config_version", 1)

    if version < 2:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        # Migrate imessage_default_limit to search.default_limit if not already set
        if "search" not in data:
            data["search"] = {}
        if "default_limit" not in data["search"] and "imessage_default_limit" in data:
            data["search"]["default_limit"] = data["imessage_default_limit"]

        # Add default sections if missing
        if "ui" not in data:
            data["ui"] = {}
        if "chat" not in data:
            data["chat"] = {}

        # Update version
        data["config_version"] = CONFIG_VERSION

    return data


def load_config(config_path: Path | None = None) -> JarvisConfig:
    """Load configuration from file, return defaults if missing/invalid.

    Automatically migrates older config versions while preserving existing values.

    Args:
        config_path: Optional path to config file. Defaults to ~/.jarvis/config.json.

    Returns:
        JarvisConfig instance with loaded or default values.
    """
    path = config_path or CONFIG_PATH

    if not path.exists():
        logger.debug(f"Config file not found at {path}, using defaults")
        return JarvisConfig()

    try:
        with path.open() as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in config file {path}: {e}, using defaults")
        return JarvisConfig()
    except OSError as e:
        logger.warning(f"Cannot read config file {path}: {e}, using defaults")
        return JarvisConfig()

    # Migrate from older versions
    data = _migrate_config(data)

    try:
        return JarvisConfig.model_validate(data)
    except ValidationError as e:
        logger.warning(f"Config validation failed: {e}, using defaults")
        return JarvisConfig()


def save_config(config: JarvisConfig, config_path: Path | None = None) -> bool:
    """Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to config file. Defaults to ~/.jarvis/config.json.

    Returns:
        True if saved successfully, False otherwise.
    """
    path = config_path or CONFIG_PATH

    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write config with pretty formatting
        with path.open("w") as f:
            json.dump(config.model_dump(), f, indent=2)

        logger.debug(f"Configuration saved to {path}")
        return True

    except OSError as e:
        logger.error(f"Failed to save config to {path}: {e}")
        return False


def get_config() -> JarvisConfig:
    """Get singleton configuration instance.

    Returns:
        Shared JarvisConfig instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset singleton configuration for testing."""
    global _config
    _config = None
