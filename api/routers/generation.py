"""Generation API endpoints.

Provides AI-powered content generation including draft replies and smart suggestions.
This module consolidates drafts and suggestions routers.
"""

# Re-export all endpoints from the original routers
from api.routers.drafts import router as drafts_router
from api.routers.suggestions import router as suggestions_router

# Export the routers for use in main.py
router = None  # Placeholder - individual routers are used directly

__all__ = [
    "drafts_router",
    "suggestions_router",
]
