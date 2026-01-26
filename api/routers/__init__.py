"""API routers for JARVIS endpoints."""

from .conversations import router as conversations_router
from .drafts import router as drafts_router
from .export import router as export_router
from .health import router as health_router
from .metrics import router as metrics_router
from .settings import router as settings_router
from .suggestions import router as suggestions_router
from .topics import router as topics_router

__all__ = [
    "conversations_router",
    "drafts_router",
    "export_router",
    "health_router",
    "metrics_router",
    "settings_router",
    "suggestions_router",
    "topics_router",
]
