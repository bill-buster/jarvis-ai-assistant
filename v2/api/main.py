"""JARVIS v2 FastAPI Application.

Simplified API for Tauri desktop app integration.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure v2 directory is in path for imports to work
_v2_dir = Path(__file__).parent.parent
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import conversations, generate, health, search, settings, websocket

# Configure logging to show timing info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


import asyncio

_models_ready = False
_preload_task: asyncio.Task | None = None


def _preload_models_sync() -> None:
    """Preload models (sync, runs in executor)."""
    global _models_ready

    # Preload embedding model first (needed for similarity search)
    try:
        from core.embeddings.model import get_embedding_model

        logger.info("Preloading embedding model...")
        embedding_model = get_embedding_model()
        embedding_model.preload()
        logger.info("Embedding model preloaded")
    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}")

    # Preload LLM
    try:
        from core.models.loader import get_model_loader

        logger.info("Preloading LLM...")
        loader = get_model_loader()
        loader.preload()
        logger.info("LLM preloaded")
    except Exception as e:
        logger.warning(f"Failed to preload LLM (will load on first request): {e}")

    _models_ready = True
    logger.info("All models ready")


async def _preload_models_async() -> None:
    """Preload models in background."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _preload_models_sync)


def is_models_ready() -> bool:
    """Check if models are preloaded."""
    return _models_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _preload_task
    logger.info("Starting JARVIS v2 API")
    # Start model preloading as background task so WebSocket can connect immediately
    _preload_task = asyncio.create_task(_preload_models_async())
    logger.info("Model preloading started in background")
    yield
    logger.info("Shutting down JARVIS v2 API")


app = FastAPI(
    title="JARVIS v2",
    description="Local-first AI assistant for iMessage",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for Tauri app - permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(conversations.router, prefix="/conversations", tags=["Conversations"])
app.include_router(generate.router, prefix="/generate", tags=["Generation"])
app.include_router(settings.router, prefix="/settings", tags=["Settings"])
app.include_router(websocket.router, tags=["WebSocket"])
app.include_router(search.router, tags=["Search"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "JARVIS v2",
        "version": "2.0.0",
        "docs": "/docs",
    }
