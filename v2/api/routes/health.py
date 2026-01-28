"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and system health."""
    from v2.core.imessage import MessageReader
    from v2.core.models import get_model_loader

    # Check iMessage access
    imessage_ok = False
    try:
        reader = MessageReader()
        imessage_ok = reader.check_access()
    except Exception:
        pass

    # Check model status
    model_loaded = False
    try:
        loader = get_model_loader()
        model_loaded = loader.is_loaded
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="2.0.0",
        model_loaded=model_loaded,
        imessage_accessible=imessage_ok,
    )
