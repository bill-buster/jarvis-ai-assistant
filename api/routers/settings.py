"""Settings API endpoints.

Provides configuration management for JARVIS including model selection,
generation parameters, and behavior preferences.
"""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import psutil
from fastapi import APIRouter, HTTPException

from api.schemas import (
    ActivateResponse,
    BehaviorSettings,
    DownloadStatus,
    GenerationSettings,
    ModelInfo,
    SettingsResponse,
    SettingsUpdateRequest,
    SystemInfo,
)
from integrations.imessage import ChatDBReader
from jarvis.config import get_config, save_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


class ModelRegistry(TypedDict):
    """Type definition for model registry entries."""

    model_id: str
    name: str
    size_gb: float
    quality_tier: str
    ram_requirement_gb: float
    description: str


# Model registry - defines available models with their characteristics
AVAILABLE_MODELS: list[ModelRegistry] = [
    {
        "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "name": "Qwen 0.5B (Fast)",
        "size_gb": 0.4,
        "quality_tier": "basic",
        "ram_requirement_gb": 4,
        "description": "Fastest responses, good for simple tasks",
    },
    {
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "name": "Qwen 1.5B (Balanced)",
        "size_gb": 1.0,
        "quality_tier": "good",
        "ram_requirement_gb": 8,
        "description": "Balanced speed and quality",
    },
    {
        "model_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "name": "Qwen 3B (Quality)",
        "size_gb": 2.0,
        "quality_tier": "best",
        "ram_requirement_gb": 16,
        "description": "Best quality, requires more RAM",
    },
]

# Settings file path for generation/behavior settings
SETTINGS_PATH = Path.home() / ".jarvis" / "settings.json"


def _get_default_settings() -> dict[str, Any]:
    """Get default generation and behavior settings."""
    return {
        "generation": {
            "temperature": 0.7,
            "max_tokens_reply": 150,
            "max_tokens_summary": 500,
        },
        "behavior": {
            "auto_suggest_replies": True,
            "suggestion_count": 3,
            "context_messages_reply": 20,
            "context_messages_summary": 50,
        },
    }


def _load_settings() -> dict[str, Any]:
    """Load settings from file, returning defaults if not found."""
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open() as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return _get_default_settings()


def _save_settings(settings: dict[str, Any]) -> bool:
    """Save settings to file."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SETTINGS_PATH.open("w") as f:
            json.dump(settings, f, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to save settings: {e}")
        return False


def _check_model_downloaded(model_id: str) -> bool:
    """Check if a model has been downloaded locally."""
    # Check in HuggingFace cache directories
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".huggingface" / "hub",
    ]

    # Convert model_id to cache format (e.g., mlx-community/Qwen2.5-0.5B-Instruct-4bit)
    model_cache_name = "models--" + model_id.replace("/", "--")

    for cache_dir in cache_dirs:
        model_path = cache_dir / model_cache_name
        if model_path.exists():
            return True

    return False


def _check_model_loaded(model_id: str) -> bool:
    """Check if a specific model is currently loaded."""
    try:
        from models import get_generator

        generator = get_generator()
        if generator._model is None:  # type: ignore[attr-defined]
            return False
        # Check if loaded model matches the requested model_id
        config = get_config()
        return config.model_path == model_id
    except Exception:
        return False


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible."""
    try:
        reader = ChatDBReader()
        result = reader.check_access()
        reader.close()
        return result
    except Exception:
        return False


def _get_system_info() -> SystemInfo:
    """Get current system information."""
    memory = psutil.virtual_memory()
    system_ram_gb = memory.total / (1024**3)
    current_usage_gb = memory.used / (1024**3)

    # Check model status
    model_loaded = False
    model_memory_gb = 0.0
    try:
        from models import get_generator

        generator = get_generator()
        if generator._model is not None:  # type: ignore[attr-defined]
            model_loaded = True
            model_memory_gb = generator.config.estimated_memory_mb / 1024
    except Exception:
        pass

    return SystemInfo(
        system_ram_gb=round(system_ram_gb, 1),
        current_memory_usage_gb=round(current_usage_gb, 1),
        model_loaded=model_loaded,
        model_memory_usage_gb=round(model_memory_gb, 1),
        imessage_access=_check_imessage_access(),
    )


def _get_recommended_model() -> str:
    """Get recommended model based on system RAM."""
    memory = psutil.virtual_memory()
    system_ram_gb = memory.total / (1024**3)

    # Return the largest model that fits in available RAM
    recommended: str = AVAILABLE_MODELS[0]["model_id"]
    for model in AVAILABLE_MODELS:
        if system_ram_gb >= model["ram_requirement_gb"]:
            recommended = model["model_id"]

    return recommended


@router.get("", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    """Get current settings including model, generation, behavior, and system info."""
    config = get_config()
    settings = _load_settings()

    return SettingsResponse(
        model_id=config.model_path,
        generation=GenerationSettings(**settings.get("generation", {})),
        behavior=BehaviorSettings(**settings.get("behavior", {})),
        system=_get_system_info(),
    )


@router.put("", response_model=SettingsResponse)
def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings.

    Updates are partial - only provided fields are changed.
    """
    config = get_config()
    settings = _load_settings()

    # Update model if provided
    if request.model_id is not None:
        # Validate model_id is in our registry
        valid_ids = [m["model_id"] for m in AVAILABLE_MODELS]
        if request.model_id not in valid_ids:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_id}")
        config.model_path = request.model_id
        save_config(config)

    # Update generation settings if provided
    if request.generation is not None:
        settings["generation"] = request.generation.model_dump()

    # Update behavior settings if provided
    if request.behavior is not None:
        settings["behavior"] = request.behavior.model_dump()

    # Save updated settings
    _save_settings(settings)

    return SettingsResponse(
        model_id=config.model_path,
        generation=GenerationSettings(**settings.get("generation", {})),
        behavior=BehaviorSettings(**settings.get("behavior", {})),
        system=_get_system_info(),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """List available models with their status."""
    recommended_model = _get_recommended_model()

    models = []
    for model in AVAILABLE_MODELS:
        model_id = model["model_id"]
        models.append(
            ModelInfo(
                model_id=model_id,
                name=model["name"],
                size_gb=model["size_gb"],
                quality_tier=model["quality_tier"],
                ram_requirement_gb=model["ram_requirement_gb"],
                is_downloaded=_check_model_downloaded(model_id),
                is_loaded=_check_model_loaded(model_id),
                is_recommended=(model_id == recommended_model),
                description=model["description"],
            )
        )

    return models


@router.post("/models/{model_id:path}/download", response_model=DownloadStatus)
def download_model(model_id: str) -> DownloadStatus:
    """Start downloading a model.

    Note: In a production system, this would start an async download task.
    For now, we use huggingface_hub to download synchronously.
    """
    # Validate model_id
    valid_ids = [m["model_id"] for m in AVAILABLE_MODELS]
    if model_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    # Check if already downloaded
    if _check_model_downloaded(model_id):
        return DownloadStatus(
            model_id=model_id,
            status="completed",
            progress=100.0,
        )

    try:
        # Use huggingface_hub to download
        from huggingface_hub import snapshot_download

        snapshot_download(model_id)

        return DownloadStatus(
            model_id=model_id,
            status="completed",
            progress=100.0,
        )
    except Exception as e:
        logger.exception(f"Failed to download model {model_id}")
        return DownloadStatus(
            model_id=model_id,
            status="failed",
            progress=0.0,
            error=str(e),
        )


@router.post("/models/{model_id:path}/activate", response_model=ActivateResponse)
def activate_model(model_id: str) -> ActivateResponse:
    """Switch to a different model.

    This updates the config and reloads the model.
    """
    # Validate model_id
    valid_ids = [m["model_id"] for m in AVAILABLE_MODELS]
    if model_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    # Check if model is downloaded
    if not _check_model_downloaded(model_id):
        return ActivateResponse(
            success=False,
            model_id=model_id,
            error="Model not downloaded. Please download first.",
        )

    try:
        # Update config
        config = get_config()
        config.model_path = model_id
        save_config(config)

        # Reset generator to force reload with new model
        from models import reset_generator

        reset_generator()

        return ActivateResponse(
            success=True,
            model_id=model_id,
        )
    except Exception as e:
        logger.exception(f"Failed to activate model {model_id}")
        return ActivateResponse(
            success=False,
            model_id=model_id,
            error=str(e),
        )
