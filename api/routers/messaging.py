"""Messaging API endpoints.

Provides conversation management, message threading, attachments, and search functionality.
This module consolidates conversations, threads, attachments, and search routers.
"""

# Re-export all endpoints from the original routers
from api.routers.attachments import router as attachments_router
from api.routers.conversations import router as conversations_router
from api.routers.search import router as search_router
from api.routers.threads import router as threads_router

# Export the routers for use in main.py
router = None  # Placeholder - individual routers are used directly

__all__ = [
    "conversations_router",
    "threads_router",
    "attachments_router",
    "search_router",
]
