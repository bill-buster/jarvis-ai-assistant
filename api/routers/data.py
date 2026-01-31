"""Data domain API router.

Consolidates contacts, embeddings, and topics endpoints into a single domain.
This module provides access to user data management, vector embeddings, and
topic clustering functionality.
"""

# Re-export all endpoints from the original routers
from api.routers.contacts import router as contacts_router
from api.routers.embeddings import router as embeddings_router
from api.routers.topics import router as topics_router

__all__ = [
    "contacts_router",
    "embeddings_router",
    "topics_router",
]
