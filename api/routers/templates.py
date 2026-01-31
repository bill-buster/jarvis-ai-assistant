"""Templates API endpoints.

Provides custom template management and analytics for template matching.
This module consolidates custom_templates and template_analytics routers.
"""

# Re-export all endpoints from the original routers
from api.routers.custom_templates import router as custom_templates_router
from api.routers.template_analytics import router as template_analytics_router

# Export the routers for use in main.py
router = None  # Placeholder - individual routers are used directly

__all__ = [
    "custom_templates_router",
    "template_analytics_router",
]
