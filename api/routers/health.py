"""Health check API endpoints.

Provides system health status including memory, model, and permission state.
"""

import psutil
from fastapi import APIRouter

from api.schemas import HealthResponse
from integrations.imessage import ChatDBReader

router = APIRouter(tags=["health"])


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible."""
    try:
        reader = ChatDBReader()
        result = reader.check_access()
        reader.close()
        return result
    except Exception:
        return False


def _get_memory_mode(available_gb: float) -> str:
    """Determine memory mode based on available memory."""
    if available_gb >= 4.0:
        return "FULL"
    elif available_gb >= 2.0:
        return "LITE"
    else:
        return "MINIMAL"


def _check_model_loaded() -> bool:
    """Check if the MLX model is currently loaded."""
    try:
        from models import get_generator

        generator = get_generator()
        # Check if internal model is loaded (accessing private attribute for status check)
        return generator._model is not None  # type: ignore[attr-defined]
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Get comprehensive system health status.

    Returns information about:
    - iMessage database access
    - System memory usage
    - Memory controller mode
    - Model loading state
    - Overall system health
    """
    # Memory stats
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)

    # Check various components
    imessage_access = _check_imessage_access()
    memory_mode = _get_memory_mode(available_gb)
    model_loaded = _check_model_loaded()

    # Determine overall status
    details: dict[str, str] = {}

    if not imessage_access:
        details["imessage"] = "Full Disk Access required"

    if available_gb < 2.0:
        details["memory"] = f"Low memory: {available_gb:.1f}GB available"

    # Determine overall health status
    if not imessage_access:
        status = "unhealthy"
    elif available_gb < 2.0:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        imessage_access=imessage_access,
        memory_available_gb=round(available_gb, 2),
        memory_used_gb=round(used_gb, 2),
        memory_mode=memory_mode,
        model_loaded=model_loaded,
        permissions_ok=imessage_access,
        details=details if details else None,
    )


@router.get("/")
def root() -> dict[str, str]:
    """Root endpoint - simple health ping."""
    return {"status": "ok", "service": "jarvis-api"}
