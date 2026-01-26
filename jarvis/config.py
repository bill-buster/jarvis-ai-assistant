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
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"

# Current config schema version for migration tracking
CONFIG_VERSION = 4


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


class ModelSettings(BaseModel):
    """Model configuration for text generation.

    Attributes:
        model_id: Model identifier from the registry (e.g., "qwen-1.5b").
        auto_select: Automatically select the best model based on available RAM.
        max_tokens_reply: Maximum tokens for reply generation.
        max_tokens_summary: Maximum tokens for summarization.
        temperature: Sampling temperature for generation (0.0-2.0).
    """

    model_id: str = "qwen-1.5b"
    auto_select: bool = True
    max_tokens_reply: int = Field(default=150, ge=1, le=2048)
    max_tokens_summary: int = Field(default=500, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class TaskQueueConfig(BaseModel):
    """Task queue configuration for background operations.

    Attributes:
        max_completed_tasks: Maximum completed tasks to keep in memory.
        worker_poll_interval: Seconds between queue polls when idle.
        max_retries: Default maximum retry attempts for failed tasks.
        auto_start_worker: Automatically start the worker when queue is used.
    """

    max_completed_tasks: int = Field(default=100, ge=10, le=1000)
    worker_poll_interval: float = Field(default=1.0, ge=0.1, le=10.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    auto_start_worker: bool = True


class JarvisConfig(BaseModel):
    """JARVIS configuration schema.

    Attributes:
        config_version: Schema version for migration tracking.
        model_path: HuggingFace model path for MLX inference (deprecated, use model.model_id).
        template_similarity_threshold: Minimum similarity score for template matching (0-1).
        memory_thresholds: Memory thresholds for mode selection.
        imessage_default_limit: Default limit for iMessage search (deprecated).
        ui: UI preferences for the Tauri frontend.
        search: Search preferences.
        chat: Chat preferences.
        model: Model configuration for text generation.
        task_queue: Task queue configuration for background operations.
    """

    config_version: int = CONFIG_VERSION
    model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    template_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_thresholds: MemoryThresholds = Field(default_factory=MemoryThresholds)
    imessage_default_limit: int = 50
    ui: UIConfig = Field(default_factory=UIConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    model: ModelSettings = Field(default_factory=ModelSettings)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)


# Module-level singleton with thread safety
_config: JarvisConfig | None = None
_config_lock = threading.Lock()


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate config data from older versions to current schema.

    Preserves existing values while adding new defaults for missing fields.
    Handles migration from:
    - v1 (no version field) to v2 (with ui/search/chat sections)
    - v2 to v3 (with model section)
    - v3 to v4 (with task_queue section)

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

        version = 2

    if version < 3:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        # Add model section if missing
        if "model" not in data:
            data["model"] = {}

        # Migrate model_path to model.model_id if possible
        if "model_path" in data and "model_id" not in data["model"]:
            # Map known paths to model IDs
            path_to_id = {
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit": "qwen-0.5b",
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "qwen-1.5b",
                "mlx-community/Qwen2.5-3B-Instruct-4bit": "qwen-3b",
            }
            model_path = data["model_path"]
            if model_path in path_to_id:
                data["model"]["model_id"] = path_to_id[model_path]

        version = 3

    if version < 4:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        # Add task_queue section if missing
        if "task_queue" not in data:
            data["task_queue"] = {}

        version = 4

    # Update version
    data["config_version"] = CONFIG_VERSION

    return data


def load_config(config_path: Path | None = None) -> JarvisConfig:
    """Load configuration from file, return defaults if missing/invalid.

    Automatically migrates older config versions while preserving existing values.
    If migration occurs, the updated config is saved back to disk.

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

    # Track original version for migration detection
    original_version = data.get("config_version", 1)

    # Migrate from older versions
    data = _migrate_config(data)

    try:
        config = JarvisConfig.model_validate(data)

        # Persist migrated config so migration doesn't run on every startup
        if original_version < CONFIG_VERSION:
            logger.info(f"Persisting migrated config (v{original_version} -> v{CONFIG_VERSION})")
            save_config(config, path)

        return config
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

    Uses double-check locking for thread safety.

    Returns:
        Shared JarvisConfig instance.
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


def reset_config() -> None:
    """Reset singleton configuration for testing."""
    global _config
    with _config_lock:
        _config = None
