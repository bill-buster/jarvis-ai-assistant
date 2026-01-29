"""Minimal FastAPI for JARVIS v3.

Essential endpoints only:
- GET /health
- GET /conversations
- GET /conversations/{id}/messages
- POST /generate/replies
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("JARVIS v3 starting up...")
    yield
    logger.info("JARVIS v3 shutting down...")


# Create app
app = FastAPI(
    title="JARVIS v3",
    description="Minimal iMessage reply generation API",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routes
from .routes import conversations, generate, health

app.include_router(health.router, tags=["health"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "JARVIS v3",
        "version": "3.0.0",
        "docs": "/docs",
    }
