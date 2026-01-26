"""Health check API endpoints.

Provides system health status including memory, model, and permission state.
These endpoints are used by the frontend to monitor system status and
display appropriate warnings or errors to the user.
"""

import os

import psutil
from fastapi import APIRouter

from api.schemas import HealthResponse, ModelInfo

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
        from integrations.imessage import ChatDBReader

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


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_unset=True,
    response_description="System health status including memory, permissions, and model state",
    summary="Get system health status",
    responses={
        200: {
            "description": "Health check successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "imessage_access": True,
                        "memory_available_gb": 12.5,
                        "memory_used_gb": 3.5,
                        "memory_mode": "FULL",
                        "model_loaded": True,
                        "permissions_ok": True,
                        "jarvis_rss_mb": 256.5,
                        "jarvis_vms_mb": 1024.0,
                    }
                }
            },
        }
    },
)
def get_health() -> HealthResponse:
    """Get comprehensive system health status.

    Returns detailed information about the current state of the JARVIS system,
    including memory usage, permission status, model state, and overall health.

    **Health Status Values:**
    - `healthy`: All systems operational, iMessage access granted, sufficient memory
    - `degraded`: System running but with reduced capability (low memory)
    - `unhealthy`: Critical issue preventing normal operation (no iMessage access)

    **Memory Modes:**
    - `FULL`: >= 4GB available - all features enabled
    - `LITE`: 2-4GB available - reduced context window
    - `MINIMAL`: < 2GB available - basic functionality only

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "imessage_access": true,
        "memory_available_gb": 12.5,
        "memory_used_gb": 3.5,
        "memory_mode": "FULL",
        "model_loaded": true,
        "permissions_ok": true,
        "jarvis_rss_mb": 256.5,
        "jarvis_vms_mb": 1024.0,
        "model": {
            "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "display_name": "Qwen 0.5B (Fast)",
            "loaded": true,
            "memory_usage_mb": 450.5,
            "quality_tier": "basic"
        },
        "recommended_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "system_ram_gb": 16.0
    }
    ```

    Returns:
        HealthResponse: Comprehensive system health information
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


@router.get(
    "/",
    response_model_exclude_unset=True,
    response_description="Simple health ping response",
    summary="Root endpoint - health ping",
    responses={
        200: {
            "description": "Service is running",
            "content": {"application/json": {"example": {"status": "ok", "service": "jarvis-api"}}},
        }
    },
)
def root() -> dict[str, str]:
    """Root endpoint - simple health ping.

    A lightweight endpoint to verify the API server is running.
    Use `/health` for comprehensive system status.

    **Example Response:**
    ```json
    {
        "status": "ok",
        "service": "jarvis-api"
    }
    ```

    Returns:
        dict: Simple status object with service name
    """
    return {"status": "ok", "service": "jarvis-api"}
