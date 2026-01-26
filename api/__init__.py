"""FastAPI backend for JARVIS desktop app.

Provides REST endpoints for iMessage conversations, health status, and more.

Usage:
    # Development server
    uvicorn api.main:app --reload --port 8742

    # Or via Makefile
    make api-dev
"""

from .main import app

__all__ = ["app"]
