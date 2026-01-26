"""FastAPI application for JARVIS desktop frontend.

Provides REST API for the Tauri desktop app to access iMessage data,
generate AI-powered replies, manage settings, and monitor system health.

Usage:
    uvicorn api.main:app --reload --port 8742

Documentation:
    - Swagger UI: http://localhost:8742/docs
    - ReDoc: http://localhost:8742/redoc
    - OpenAPI JSON: http://localhost:8742/openapi.json
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from api.routers import (
    conversations_router,
    drafts_router,
    health_router,
    settings_router,
    suggestions_router,
)

# API metadata for OpenAPI documentation
API_TITLE = "JARVIS API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# JARVIS - Intelligent iMessage Assistant API

JARVIS is a local-first AI assistant for macOS that provides intelligent iMessage
management using MLX-based language models. All processing runs entirely on Apple
Silicon with no cloud data transmission.

## Features

- **📱 iMessage Access**: Read conversations, messages, attachments, and reactions
- **🤖 AI-Powered Replies**: Generate contextual reply suggestions using local LLM
- **📝 Conversation Summaries**: Get AI-generated summaries of long conversations
- **⚡ Quick Suggestions**: Fast pattern-based reply suggestions (no model required)
- **⚙️ Settings Management**: Configure models, generation parameters, and behavior
- **🏥 Health Monitoring**: System health status, memory usage, and permissions

## Authentication

This API is designed for local use by the JARVIS desktop application.
No authentication is required as the API only binds to localhost.

## Rate Limiting

No rate limiting is applied. The API is designed for single-user local access.

## Error Handling

All errors return a JSON response with the following structure:
```json
{
    "error": "Brief error message",
    "detail": "Detailed explanation",
    "code": "MACHINE_READABLE_CODE"
}
```

## Privacy

All data processing happens locally on your Mac. No conversation data,
messages, or personal information is ever sent to external servers.

## Getting Started

1. Ensure Full Disk Access is granted for iMessage database access
2. Check system health: `GET /health`
3. List conversations: `GET /conversations`
4. Generate reply suggestions: `POST /drafts/reply`
"""

API_TAGS_METADATA = [
    {
        "name": "health",
        "description": "System health monitoring and status checks. "
        "Use these endpoints to verify the API is running and check system resources.",
    },
    {
        "name": "conversations",
        "description": "iMessage conversation and message management. "
        "List conversations, retrieve messages, search, and send messages.",
    },
    {
        "name": "drafts",
        "description": "AI-powered draft generation using the local MLX language model. "
        "Generate reply suggestions and conversation summaries.",
    },
    {
        "name": "suggestions",
        "description": "Fast pattern-based reply suggestions. "
        "Lightweight alternative to AI drafts for common response scenarios.",
    },
    {
        "name": "settings",
        "description": "Application configuration management. "
        "Manage model selection, generation parameters, and behavior preferences.",
    },
]

API_CONTACT = {
    "name": "JARVIS Support",
    "url": "https://github.com/jarvis-ai/jarvis",
}

API_LICENSE = {
    "name": "MIT License",
    "url": "https://opensource.org/licenses/MIT",
}


def custom_openapi() -> dict:
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=API_TAGS_METADATA,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = API_CONTACT
    openapi_schema["info"]["license"] = API_LICENSE

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8742",
            "description": "Local development server",
        },
        {
            "url": "http://127.0.0.1:8742",
            "description": "Local development server (alternative)",
        },
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full documentation and user guide",
        "url": "https://github.com/jarvis-ai/jarvis/docs",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=API_TAGS_METADATA,
    contact=API_CONTACT,
    license_info=API_LICENSE,
)

# Use custom OpenAPI schema
app.openapi = custom_openapi  # type: ignore[method-assign]

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
app.include_router(drafts_router)
app.include_router(suggestions_router)
app.include_router(settings_router)
