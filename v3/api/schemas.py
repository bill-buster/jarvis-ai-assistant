"""Minimal Pydantic schemas for JARVIS v3 API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "3.0.0"


class GenerateRepliesRequest(BaseModel):
    """Request to generate reply suggestions."""

    chat_id: str
    user_name: str = "User"
