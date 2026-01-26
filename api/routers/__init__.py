"""API routers for JARVIS endpoints."""

from .conversations import router as conversations_router
from .health import router as health_router
from .settings import router as settings_router
from .suggestions import router as suggestions_router

__all__ = ["conversations_router", "health_router", "settings_router", "suggestions_router"]
