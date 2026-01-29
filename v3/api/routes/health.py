"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API health."""
    return {
        "status": "ok",
        "version": "3.0.0",
    }
