"""FastAPI application for JARVIS desktop frontend.

Provides REST API for the Tauri desktop app to access iMessage data,
system health, and other JARVIS functionality.

Usage:
    uvicorn api.main:app --reload --port 8742
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import conversations_router, health_router, settings_router, suggestions_router

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

# Include routers
app.include_router(health_router)
app.include_router(conversations_router)
app.include_router(suggestions_router)
app.include_router(settings_router)
