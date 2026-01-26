"""API routers for JARVIS endpoints."""

from .conversations import router as conversations_router
from .drafts import router as drafts_router
from .export import router as export_router
from .health import router as health_router
from .settings import router as settings_router
from .suggestions import router as suggestions_router

__all__ = [
    "conversations_router",
    "drafts_router",
    "export_router",
    "health_router",
    "settings_router",
    "suggestions_router",
]
