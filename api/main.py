"""FastAPI application for JARVIS desktop frontend.

Provides REST API for the Tauri desktop app to access iMessage data,
system health, and other JARVIS functionality.

Usage:
    uvicorn api.main:app --reload --port 8742
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    conversations_router,
    drafts_router,
    health_router,
    metrics_router,
    settings_router,
    suggestions_router,
)
from jarvis.metrics import get_latency_histogram, get_request_counter

# Create FastAPI app
app = FastAPI(
    title="JARVIS API",
    description="Backend API for JARVIS desktop assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Tauri and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",  # Tauri production
        "http://localhost:5173",  # Vite dev server
        "http://localhost:1420",  # Tauri dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:1420",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware to track request timing and counts."""
    start_time = time.perf_counter()

    # Process the request
    response = await call_next(request)

    # Record metrics
    duration = time.perf_counter() - start_time
    endpoint = request.url.path
    method = request.method

    # Skip metrics endpoints to avoid infinite recursion in monitoring
    if not endpoint.startswith("/metrics"):
        counter = get_request_counter()
        histogram = get_latency_histogram()

        counter.increment(endpoint, method)
        histogram.observe(f"{method} {endpoint}", duration)

    # Add timing header
    response.headers["X-Response-Time"] = f"{duration:.4f}s"

    return response


# Include routers
app.include_router(health_router)
app.include_router(conversations_router)
app.include_router(drafts_router)
app.include_router(suggestions_router)
app.include_router(settings_router)
app.include_router(metrics_router)
