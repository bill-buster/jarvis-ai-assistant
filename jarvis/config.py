"""JARVIS Configuration System.

Loads and validates configuration from ~/.jarvis/config.json.
Uses Pydantic for schema validation with sensible defaults.

Usage:
    from jarvis.config import get_config

    config = get_config()
    print(config.model_path)
    print(config.template_similarity_threshold)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"


class MemoryThresholds(BaseModel):
    """Memory thresholds for mode selection."""

    full_mode_mb: int = 8000
    lite_mode_mb: int = 4000


class JarvisConfig(BaseModel):
    """JARVIS configuration schema.

    Attributes:
        model_path: HuggingFace model path for MLX inference.
        template_similarity_threshold: Minimum similarity score for template matching (0.0-1.0).
        memory_thresholds: Memory thresholds for mode selection.
        imessage_default_limit: Default limit for iMessage search results.
    """

    model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    template_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_thresholds: MemoryThresholds = Field(default_factory=MemoryThresholds)
    imessage_default_limit: int = 50


# Module-level singleton
_config: JarvisConfig | None = None


def load_config(config_path: Path | None = None) -> JarvisConfig:
    """Load configuration from file, return defaults if missing/invalid.

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

    try:
        return JarvisConfig.model_validate(data)
    except ValidationError as e:
        logger.warning(f"Config validation failed: {e}, using defaults")
        return JarvisConfig()


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
