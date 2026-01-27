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

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from slowapi.errors import RateLimitExceeded

from api.errors import register_exception_handlers
from api.ratelimit import limiter, rate_limit_exceeded_handler
from api.routers import (
    attachments_router,
    batch_router,
    calendar_router,
    contacts_router,
    conversations_router,
    custom_templates_router,
    digest_router,
    drafts_router,
    embeddings_router,
    experiments_router,
    export_router,
    feedback_router,
    health_router,
    insights_router,
    metrics_router,
    pdf_export_router,
    priority_router,
    relationships_router,
    search_router,
    settings_router,
    stats_router,
    suggestions_router,
    tasks_router,
    template_analytics_router,
    threads_router,
    topics_router,
    websocket_router,
)
from jarvis.metrics import get_latency_histogram, get_request_counter

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

Rate limiting is applied to protect system resources:
- **Read endpoints** (GET, search): 60 requests per minute
- **Write endpoints** (POST, PUT): 30 requests per minute
- **Generation endpoints** (AI-powered): 10 requests per minute

Exceeding these limits returns HTTP 429 with a `Retry-After` header.

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
        "name": "search",
        "description": "Semantic search using AI-powered embeddings. "
        "Find messages by meaning rather than exact keyword matching.",
    },
    {
        "name": "digest",
        "description": "Daily and weekly digest generation. "
        "Get summaries of unanswered messages, group highlights, action items, and statistics. "
        "Export digests in Markdown or HTML format.",
    },
    {
        "name": "drafts",
        "description": "AI-powered draft generation using the local MLX language model. "
        "Generate reply suggestions and conversation summaries.",
    },
    {
        "name": "embeddings",
        "description": "Semantic search and relationship profiling using message embeddings. "
        "Index conversations, search by meaning, and analyze communication patterns.",
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
    {
        "name": "custom-templates",
        "description": "User-defined template management. "
        "Create, edit, test, and organize custom response templates. "
        "Supports import/export for sharing template packs.",
    },
    {
        "name": "topics",
        "description": "Automatic topic detection for conversations. "
        "Analyzes message content to identify common themes like scheduling and questions.",
    },
    {
        "name": "stats",
        "description": "Conversation statistics and analytics. "
        "Get insights on messaging patterns, word frequency, activity by hour/day, "
        "response times, emoji usage, and attachment breakdowns.",
    },
    {
        "name": "insights",
        "description": "Advanced conversation insights and relationship analytics. "
        "Provides sentiment analysis over time, response time patterns, message frequency "
        "trends, and relationship health scores.",
    },
    {
        "name": "template-analytics",
        "description": "Template matching analytics and optimization. "
        "Monitor template hit rates, view top templates, and identify optimization opportunities.",
    },
    {
        "name": "websocket",
        "description": "Real-time WebSocket communication. "
        "Stream model generation responses, receive health updates, and manage live connections.",
    },
    {
        "name": "tasks",
        "description": "Background task management. "
        "Create, monitor, and manage background tasks for batch operations.",
    },
    {
        "name": "batch",
        "description": "Batch operations for bulk processing. "
        "Export, summarize, and generate replies for multiple conversations at once.",
    },
    {
        "name": "priority",
        "description": "Smart priority inbox for message importance scoring. "
        "Get messages sorted by urgency and detect questions and action items.",
    },
    {
        "name": "calendars",
        "description": "Calendar integration. "
        "Detect events in messages and create calendar entries via macOS Calendar.",
    },
    {
        "name": "threads",
        "description": "Conversation threading for organizing messages into logical groups. "
        "Detect threads based on time gaps, topic shifts, and explicit reply references.",
    },
    {
        "name": "attachments",
        "description": "iMessage attachment management. "
        "List, download, and analyze file attachments from conversations.",
    },
    {
        "name": "feedback",
        "description": "User feedback tracking and response evaluation. "
        "Record feedback on suggestions, track acceptance rates, and get improvement insights.",
    },
    {
        "name": "experiments",
        "description": "A/B testing infrastructure for prompt experimentation. "
        "Create experiments, assign variants to contacts, record outcomes, and analyze results.",
    },
    {
        "name": "relationships",
        "description": "Relationship learning and communication profiling. "
        "Build and manage profiles that capture communication patterns with each contact "
        "for personalized reply generation.",
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


def custom_openapi() -> dict[str, Any]:
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

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Use custom OpenAPI schema
app.openapi = custom_openapi

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
app.include_router(attachments_router)
app.include_router(calendar_router)
app.include_router(contacts_router)
app.include_router(conversations_router)
app.include_router(custom_templates_router)
app.include_router(digest_router)
app.include_router(drafts_router)
app.include_router(embeddings_router)
app.include_router(export_router)
app.include_router(pdf_export_router)
app.include_router(search_router)
app.include_router(suggestions_router)
app.include_router(settings_router)
app.include_router(stats_router)
app.include_router(insights_router)
app.include_router(metrics_router)
app.include_router(template_analytics_router)
app.include_router(threads_router)
app.include_router(topics_router)
app.include_router(websocket_router)
app.include_router(tasks_router)
app.include_router(batch_router)
app.include_router(priority_router)
app.include_router(feedback_router)
app.include_router(experiments_router)
app.include_router(relationships_router)

# Register JARVIS exception handlers for standardized error responses
register_exception_handlers(app)
