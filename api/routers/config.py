"""Configuration domain API router.

Consolidates settings, experiments, and feedback endpoints into a single domain.
This module provides access to system configuration, experimental features,
and user feedback collection.
"""

# Re-export all endpoints from the original routers
from api.routers.experiments import router as experiments_router
from api.routers.feedback import router as feedback_router
from api.routers.settings import router as settings_router

__all__ = [
    "experiments_router",
    "feedback_router",
    "settings_router",
]
