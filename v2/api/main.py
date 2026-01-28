"""JARVIS v2 FastAPI Application.

Simplified API for Tauri desktop app integration.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import conversations, generate, health, settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting JARVIS v2 API")
    yield
    logger.info("Shutting down JARVIS v2 API")


app = FastAPI(
    title="JARVIS v2",
    description="Local-first AI assistant for iMessage",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for Tauri app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://localhost",
        "http://localhost:1420",  # Vite dev server
        "http://127.0.0.1:1420",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(conversations.router, prefix="/conversations", tags=["Conversations"])
app.include_router(generate.router, prefix="/generate", tags=["Generation"])
app.include_router(settings.router, prefix="/settings", tags=["Settings"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "JARVIS v2",
        "version": "2.0.0",
        "docs": "/docs",
    }
