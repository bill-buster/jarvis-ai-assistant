"""FastAPI server for the JARVIS AI assistant.

Provides REST API endpoints for chat, search, health, and message management.
Designed to be called by a Tauri frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from contracts.health import FeatureState
from contracts.imessage import Attachment, Conversation, Message, Reaction
from contracts.models import GenerationRequest
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis import __version__
from jarvis.api_models import (
    AttachmentResponse,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationsListResponse,
    ErrorResponse,
    FeatureHealthResponse,
    FeatureStateEnum,
    HealthResponse,
    MemoryModeEnum,
    MemoryStatusResponse,
    MessageResponse,
    MessagesListResponse,
    ModelStatusResponse,
    ReactionResponse,
    SearchResponse,
)
from jarvis.system import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    _check_imessage_access,
    initialize_system,
)

logger = logging.getLogger(__name__)


def _cleanup() -> None:
    """Clean up system resources."""
    # Stop model warmer first (before resetting generator)
    try:
        from jarvis.model_warmer import get_model_warmer

        warmer = get_model_warmer()
        warmer.stop()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Error stopping model warmer during cleanup: %s", e)

    try:
        reset_memory_controller()
        reset_degradation_controller()
    except Exception as e:
        logger.debug("Error resetting controllers during cleanup: %s", e)

    try:
        from models import reset_generator

        reset_generator()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Error resetting generator during cleanup: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Startup
    success, warnings = initialize_system()
    if not success:
        logger.error("Failed to initialize JARVIS system")
    for warning in warnings:
        logger.warning(warning)

    # Start model warmer for smart model loading
    try:
        from jarvis.model_warmer import get_model_warmer

        warmer = get_model_warmer()
        warmer.start()
        logger.info("Model warmer started")
    except ImportError:
        logger.debug("Model warmer not available")
    except Exception as e:
        logger.warning("Failed to start model warmer: %s", e)

    yield

    # Shutdown
    _cleanup()


# Create FastAPI application
app = FastAPI(
    title="JARVIS API",
    description="Local-first AI assistant API for Tauri frontend",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware for Tauri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError) -> JSONResponse:
    """Handle permission errors with consistent ErrorResponse format."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "permission_denied",
            "message": str(exc),
            "details": "Grant Full Disk Access in System Settings.",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors with consistent ErrorResponse format."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": str(exc),
            "details": None,
        },
    )


# --- Helper Functions ---


def _message_to_response(msg: Message) -> MessageResponse:
    """Convert a Message dataclass to API response model."""
    return MessageResponse(
        id=msg.id,
        chat_id=msg.chat_id,
        sender=msg.sender,
        sender_name=msg.sender_name,
        text=msg.text,
        date=msg.date,
        is_from_me=msg.is_from_me,
        attachments=[_attachment_to_response(a) for a in msg.attachments],
        reply_to_id=msg.reply_to_id,
        reactions=[_reaction_to_response(r) for r in msg.reactions],
    )


def _attachment_to_response(att: Attachment) -> AttachmentResponse:
    """Convert an Attachment dataclass to API response model."""
    return AttachmentResponse(
        filename=att.filename,
        file_path=att.file_path,
        mime_type=att.mime_type,
        file_size=att.file_size,
    )


def _reaction_to_response(reaction: Reaction) -> ReactionResponse:
    """Convert a Reaction dataclass to API response model."""
    return ReactionResponse(
        type=reaction.type,
        sender=reaction.sender,
        sender_name=reaction.sender_name,
        date=reaction.date,
    )


def _conversation_to_response(conv: Conversation) -> ConversationResponse:
    """Convert a Conversation dataclass to API response model."""
    return ConversationResponse(
        chat_id=conv.chat_id,
        participants=conv.participants,
        display_name=conv.display_name,
        last_message_date=conv.last_message_date,
        message_count=conv.message_count,
        is_group=conv.is_group,
    )


def _feature_state_to_enum(state: FeatureState) -> FeatureStateEnum:
    """Convert FeatureState to API enum."""
    mapping = {
        FeatureState.HEALTHY: FeatureStateEnum.HEALTHY,
        FeatureState.DEGRADED: FeatureStateEnum.DEGRADED,
        FeatureState.FAILED: FeatureStateEnum.FAILED,
    }
    return mapping.get(state, FeatureStateEnum.FAILED)


# --- API Endpoints ---


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Generation error"},
    },
    summary="Send a chat message",
    description="Send a message to the AI assistant and receive a response.",
)
async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """Generate a chat response.

    Supports both regular and streaming responses via SSE.
    """
    if request.stream:
        return StreamingResponse(
            _stream_chat_response(request),
            media_type="text/event-stream",
        )

    try:
        from jarvis.model_warmer import get_warm_generator

        generator = get_warm_generator()
        deg_controller = get_degradation_controller()

        def generate_response(prompt: str) -> tuple[str, dict[str, Any]]:
            gen_request = GenerationRequest(
                prompt=prompt,
                context_documents=request.context_documents,
                few_shot_examples=[],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            response = generator.generate(gen_request)
            return response.text, {
                "tokens_used": response.tokens_used,
                "generation_time_ms": response.generation_time_ms,
                "model_name": response.model_name,
                "used_template": response.used_template,
                "template_name": response.template_name,
                "finish_reason": response.finish_reason,
            }

        result = deg_controller.execute(
            FEATURE_CHAT,
            generate_response,
            request.message,
        )

        # Handle case where degradation controller returns a simple string (fallback)
        # or a tuple (success). Safely unpack both cases.
        if isinstance(result, tuple) and len(result) == 2:
            text, metadata = result
        elif isinstance(result, str):
            text, metadata = result, {}
        else:
            # Fallback for unexpected return types
            text, metadata = str(result) if result else "", {}

        # If metadata is empty (fallback mode), use default values
        if not metadata:
            return ChatResponse(
                text=text,
                tokens_used=0,
                generation_time_ms=0.0,
                model_name="fallback",
                used_template=True,
                template_name=None,
                finish_reason="fallback",
            )

        return ChatResponse(
            text=text,
            **metadata,
        )

    except ImportError as e:
        logger.error("Model system not available: %s", e)
        raise HTTPException(status_code=500, detail="Model system not available") from e
    except Exception as e:
        logger.exception("Chat generation error")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate streaming chat response via SSE.

    Yields Server-Sent Events with token data.
    """
    try:
        from jarvis.model_warmer import get_warm_generator

        generator = get_warm_generator()

        gen_request = GenerationRequest(
            prompt=request.message,
            context_documents=request.context_documents,
            few_shot_examples=[],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Generate response (non-streaming for now, but structure supports streaming)
        response = generator.generate(gen_request)

        # Simulate streaming by yielding tokens
        words = response.text.split()
        for word in words:
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)  # Small delay between tokens

        # Send completion event
        completion_data = {
            "text": response.text,
            "tokens_used": response.tokens_used,
            "generation_time_ms": response.generation_time_ms,
            "model_name": response.model_name,
            "used_template": response.used_template,
            "template_name": response.template_name,
            "finish_reason": response.finish_reason,
        }
        yield f"event: done\ndata: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        logger.exception("Streaming chat error")
        error_data = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.get(
    "/search",
    response_model=SearchResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Permission denied"},
        500: {"model": ErrorResponse, "description": "Search error"},
    },
    summary="Search iMessages",
    description="Search iMessage conversations with optional filters.",
)
async def search_messages(
    query: str = Query(..., min_length=1, description="Search query string"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum results"),
    sender: str | None = Query(None, description="Filter by sender"),
    after: datetime | None = Query(None, description="Messages after this date"),
    before: datetime | None = Query(None, description="Messages before this date"),
    chat_id: str | None = Query(None, description="Filter by conversation ID"),
    has_attachments: bool | None = Query(None, description="Filter by attachments"),
) -> SearchResponse:
    """Search iMessages with filters."""
    deg_controller = get_degradation_controller()

    def do_search(search_query: str) -> list[Message]:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            if not reader.check_access():
                raise PermissionError("Cannot access iMessage database")
            return reader.search(
                query=search_query,
                limit=limit,
                sender=sender,
                after=after,
                before=before,
                chat_id=chat_id,
                has_attachments=has_attachments,
            )

    try:
        messages = deg_controller.execute(FEATURE_IMESSAGE, do_search, query)
        return SearchResponse(
            messages=[_message_to_response(m) for m in messages],
            total=len(messages),
            query=query,
        )

    except PermissionError as e:
        logger.warning("Permission error in search: %s", e)
        raise HTTPException(
            status_code=403,
            detail="Cannot access iMessage database. Grant Full Disk Access.",
        ) from e
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/conversations",
    response_model=ConversationsListResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Permission denied"},
        500: {"model": ErrorResponse, "description": "Error fetching conversations"},
    },
    summary="List conversations",
    description="Get a list of recent iMessage conversations.",
)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum conversations"),
    since: datetime | None = Query(None, description="Conversations since this date"),
) -> ConversationsListResponse:
    """List recent conversations."""
    deg_controller = get_degradation_controller()

    def get_conversations(_: str) -> list[Conversation]:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            if not reader.check_access():
                raise PermissionError("Cannot access iMessage database")
            return reader.get_conversations(limit=limit, since=since)

    try:
        conversations = deg_controller.execute(FEATURE_IMESSAGE, get_conversations, "")
        return ConversationsListResponse(
            conversations=[_conversation_to_response(c) for c in conversations],
            total=len(conversations),
        )

    except PermissionError as e:
        logger.warning("Permission error fetching conversations: %s", e)
        raise HTTPException(
            status_code=403,
            detail="Cannot access iMessage database. Grant Full Disk Access.",
        ) from e
    except Exception as e:
        logger.exception("Error fetching conversations")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/messages/{conversation_id:path}",
    response_model=MessagesListResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Permission denied"},
        404: {"model": ErrorResponse, "description": "Conversation not found"},
        500: {"model": ErrorResponse, "description": "Error fetching messages"},
    },
    summary="Get messages in conversation",
    description="Get messages from a specific conversation.",
)
async def get_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum messages"),
    before: datetime | None = Query(None, description="Messages before this date"),
) -> MessagesListResponse:
    """Get messages from a conversation."""
    deg_controller = get_degradation_controller()

    def fetch_messages(_: str) -> list[Message]:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            if not reader.check_access():
                raise PermissionError("Cannot access iMessage database")
            return reader.get_messages(
                chat_id=conversation_id,
                limit=limit,
                before=before,
            )

    try:
        messages = deg_controller.execute(FEATURE_IMESSAGE, fetch_messages, "")
        return MessagesListResponse(
            messages=[_message_to_response(m) for m in messages],
            chat_id=conversation_id,
            total=len(messages),
        )

    except PermissionError as e:
        logger.warning("Permission error fetching messages: %s", e)
        raise HTTPException(
            status_code=403,
            detail="Cannot access iMessage database. Grant Full Disk Access.",
        ) from e
    except Exception as e:
        logger.exception("Error fetching messages")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health status",
    description="Get the health status of all JARVIS system components.",
)
async def health() -> HealthResponse:
    """Get system health status."""
    # Memory status
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()

    # Map memory mode
    mode_mapping = {
        "full": MemoryModeEnum.FULL,
        "lite": MemoryModeEnum.LITE,
        "minimal": MemoryModeEnum.MINIMAL,
    }
    memory_mode = mode_mapping.get(state.current_mode.value.lower(), MemoryModeEnum.MINIMAL)

    memory_status = MemoryStatusResponse(
        available_mb=state.available_mb,
        used_mb=state.used_mb,
        current_mode=memory_mode,
        pressure_level=state.pressure_level,
        model_loaded=state.model_loaded,
    )

    # Feature health
    deg_controller = get_degradation_controller()
    feature_health = deg_controller.get_health()

    features = []
    for feature_name, feature_state in feature_health.items():
        details = None
        if feature_name == FEATURE_IMESSAGE:
            details = "Full Disk Access required" if not _check_imessage_access() else "OK"
        elif feature_name == FEATURE_CHAT:
            details = "OK"

        features.append(
            FeatureHealthResponse(
                name=feature_name,
                state=_feature_state_to_enum(feature_state),
                details=details,
            )
        )

    # Model status
    model_loaded = False
    model_memory_mb = None
    model_name = None

    try:
        from models import get_generator

        generator = get_generator()
        model_loaded = generator.is_loaded()
        if model_loaded:
            model_memory_mb = generator.get_memory_usage_mb()
            model_name = generator.config.model_path
    except Exception as e:
        logger.debug("Model status unavailable: %s", e)

    model_status = ModelStatusResponse(
        loaded=model_loaded,
        memory_usage_mb=model_memory_mb,
        model_name=model_name,
    )

    # Determine overall status
    has_failed = any(f.state == FeatureStateEnum.FAILED for f in features)
    has_degraded = any(f.state == FeatureStateEnum.DEGRADED for f in features)

    if has_failed:
        overall_status = "unhealthy"
    elif has_degraded:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        memory=memory_status,
        features=features,
        model=model_status,
        version=__version__,
    )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app.

    Useful for testing and custom configurations.
    """
    return app
