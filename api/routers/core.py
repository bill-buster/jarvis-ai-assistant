"""Core API endpoints.

Provides system health monitoring, metrics collection, and WebSocket communication.
This module consolidates health, metrics, and websocket routers into a single domain.
"""

# Re-export all endpoints from the original routers
from api.routers.health import router as health_router
from api.routers.metrics import router as metrics_router
from api.routers.websocket import router as websocket_router

# Export the routers for use in main.py
router = None  # Placeholder - individual routers are used directly

__all__ = [
    "health_router",
    "metrics_router",
    "websocket_router",
]
