"""Settings endpoints with persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas import SettingsResponse, SettingsUpdateRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Config file path
CONFIG_DIR = Path.home() / ".jarvis"
CONFIG_FILE = CONFIG_DIR / "api_settings.json"

# Default settings
DEFAULT_SETTINGS = {
    "model_id": "qwen-1.5b",
    "auto_suggest": True,
    "max_replies": 3,
    "user_name": "User",
}


def _load_settings() -> dict:
    """Load settings from disk, falling back to defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
                # Merge with defaults to handle new fields
                return {**DEFAULT_SETTINGS, **saved}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load settings: {e}")
    return DEFAULT_SETTINGS.copy()


def _save_settings(settings: dict) -> None:
    """Save settings to disk."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to save settings: {e}")


# Load settings on module import
_settings = _load_settings()


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Get current settings."""
    return SettingsResponse(**_settings)


@router.put("", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings."""
    if request.model_id is not None:
        # Validate model exists
        from core.models import MODELS

        if request.model_id not in MODELS:
            raise HTTPException(
                status_code=400,
                detail="Unknown model ID",
            )
        _settings["model_id"] = request.model_id

        # Switch model
        from core.models import get_model_loader

        loader = get_model_loader()
        loader.switch_model(request.model_id)

    if request.auto_suggest is not None:
        _settings["auto_suggest"] = request.auto_suggest

    if request.max_replies is not None:
        _settings["max_replies"] = request.max_replies

    if request.user_name is not None:
        _settings["user_name"] = request.user_name

    # Persist to disk
    _save_settings(_settings)

    return SettingsResponse(**_settings)


@router.get("/models")
async def list_models():
    """List available models."""
    from core.models import MODELS

    return {
        "models": [
            {
                "id": spec.id,
                "display_name": spec.display_name,
                "size_gb": spec.size_gb,
                "quality": spec.quality,
                "description": spec.description,
            }
            for spec in MODELS.values()
        ]
    }


def get_user_name() -> str:
    """Get the configured user name."""
    return _settings.get("user_name", DEFAULT_SETTINGS["user_name"])
