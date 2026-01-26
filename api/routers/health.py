"""Health check API endpoints.

Provides system health status including memory, model, and permission state.
"""

import os

import psutil
from fastapi import APIRouter

from api.schemas import HealthResponse, ModelInfo
from integrations.imessage import ChatDBReader

router = APIRouter(tags=["health"])

# Constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


def _get_process_memory() -> tuple[float, float]:
    """Get JARVIS process memory usage.

    Returns:
        Tuple of (rss_mb, vms_mb) - actual RAM usage and virtual memory allocation
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / BYTES_PER_MB
        vms_mb = mem_info.vms / BYTES_PER_MB
        return rss_mb, vms_mb
    except Exception:
        return 0.0, 0.0


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


def _get_model_info() -> ModelInfo | None:
    """Get information about the current model.

    Returns:
        ModelInfo with current model details, or None if unavailable.
    """
    try:
        from models import get_generator

        generator = get_generator()
        loader = generator._loader
        info = loader.get_current_model_info()

        return ModelInfo(
            id=info.get("id"),
            display_name=info.get("display_name", "Unknown"),
            loaded=info.get("loaded", False),
            memory_usage_mb=info.get("memory_usage_mb", 0.0),
            quality_tier=info.get("quality_tier"),
        )
    except Exception:
        return None


def _get_recommended_model(total_ram_gb: float) -> str | None:
    """Get the recommended model for the system's total RAM.

    Args:
        total_ram_gb: Total system RAM in GB.

    Returns:
        Model ID string, or None if unavailable.
    """
    try:
        from models import get_recommended_model

        spec = get_recommended_model(total_ram_gb)
        return spec.id
    except Exception:
        return None


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Get comprehensive system health status.

    Returns information about:
    - iMessage database access
    - System memory usage (total system)
    - JARVIS process memory usage (what this app is using)
    - Memory controller mode
    - Model loading state and details
    - Recommended model for this system
    - Overall system health
    """
    # System memory stats
    memory = psutil.virtual_memory()
    available_gb = memory.available / BYTES_PER_GB
    used_gb = memory.used / BYTES_PER_GB
    total_gb = memory.total / BYTES_PER_GB

    # JARVIS process memory
    jarvis_rss_mb, jarvis_vms_mb = _get_process_memory()

    # Check various components
    imessage_access = _check_imessage_access()
    memory_mode = _get_memory_mode(available_gb)
    model_loaded = _check_model_loaded()

    # Get model information
    model_info = _get_model_info()
    recommended_model = _get_recommended_model(total_gb)

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
        jarvis_rss_mb=round(jarvis_rss_mb, 1),
        jarvis_vms_mb=round(jarvis_vms_mb, 1),
        model=model_info,
        recommended_model=recommended_model,
        system_ram_gb=round(total_gb, 2),
    )


@router.get("/")
def root() -> dict[str, str]:
    """Root endpoint - simple health ping."""
    return {"status": "ok", "service": "jarvis-api"}
