"""API routers for JARVIS endpoints."""

from .batch import router as batch_router
from .contacts import router as contacts_router
from .conversations import router as conversations_router
from .drafts import router as drafts_router
from .export import router as export_router
from .health import router as health_router
from .metrics import router as metrics_router
from .pdf_export import router as pdf_export_router
from .settings import router as settings_router
from .stats import router as stats_router
from .suggestions import router as suggestions_router
from .tasks import router as tasks_router
from .template_analytics import router as template_analytics_router
from .topics import router as topics_router

__all__ = [
    "batch_router",
    "contacts_router",
    "conversations_router",
    "drafts_router",
    "export_router",
    "health_router",
    "metrics_router",
    "pdf_export_router",
    "settings_router",
    "stats_router",
    "suggestions_router",
    "tasks_router",
    "template_analytics_router",
    "topics_router",
]
