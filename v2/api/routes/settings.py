"""Settings endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import SettingsResponse, SettingsUpdateRequest

router = APIRouter()

# Simple in-memory settings for now
# TODO: Persist to ~/.jarvis/config.json
_settings = {
    "model_id": "qwen-1.5b",
    "auto_suggest": True,
    "max_replies": 3,
}


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Get current settings."""
    return SettingsResponse(**_settings)


@router.put("", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings."""
    if request.model_id is not None:
        # Validate model exists
        from v2.core.models import MODELS

        if request.model_id not in MODELS:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model_id}. Available: {list(MODELS.keys())}",
            )
        _settings["model_id"] = request.model_id

        # Switch model
        from v2.core.models import get_model_loader

        loader = get_model_loader()
        loader.switch_model(request.model_id)

    if request.auto_suggest is not None:
        _settings["auto_suggest"] = request.auto_suggest

    if request.max_replies is not None:
        _settings["max_replies"] = request.max_replies

    return SettingsResponse(**_settings)


@router.get("/models")
async def list_models():
    """List available models."""
    from v2.core.models import MODELS

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
