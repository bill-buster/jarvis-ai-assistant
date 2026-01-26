"""Draft reply generation API endpoint.

Provides AI-powered reply suggestions with timeout handling and fallback.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter

from api.schemas import (
    ContextInfo,
    DraftReplyRequest,
    DraftReplyResponse,
    GenerationStatus,
    Suggestion,
    SummaryRequest,
    SummaryResponse,
)
from jarvis.fallbacks import (
    FailureReason,
    ModelLoadError,
    get_fallback_reply_suggestions,
    get_fallback_response,
    get_fallback_summary,
)
from jarvis.generation import (
    generate_reply_suggestions,
    generate_summary,
    get_generation_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drafts", tags=["drafts"])

# Configuration
GENERATION_TIMEOUT_SECONDS = 30

# Thread pool for running blocking generation code
executor = ThreadPoolExecutor(max_workers=2)


def _generate_replies_sync(request: DraftReplyRequest) -> list[tuple[str, float]]:
    """Synchronous helper for generating replies.

    Args:
        request: The draft reply request

    Returns:
        List of (suggestion_text, confidence) tuples
    """
    return generate_reply_suggestions(
        last_message=request.last_message,
        context_messages=None,  # Context handling can be added later
        num_suggestions=request.num_suggestions,
    )


def _generate_summary_sync(
    messages: list[str],
    participant: str,
) -> tuple[str, bool]:
    """Synchronous helper for generating summaries.

    Args:
        messages: List of messages to summarize
        participant: Conversation participant name

    Returns:
        Tuple of (summary_text, used_fallback)
    """
    return generate_summary(messages, participant)


@router.post("/reply", response_model=DraftReplyResponse)
async def draft_reply(request: DraftReplyRequest) -> DraftReplyResponse:
    """Generate AI-powered reply suggestions.

    Uses the AI model to generate contextual reply suggestions.
    Falls back to generic suggestions on timeout or error.
    """
    try:
        # Run generation with timeout
        loop = asyncio.get_event_loop()
        suggestions_with_confidence = await asyncio.wait_for(
            loop.run_in_executor(executor, _generate_replies_sync, request),
            timeout=GENERATION_TIMEOUT_SECONDS,
        )

        suggestions = [
            Suggestion(text=text, confidence=confidence)
            for text, confidence in suggestions_with_confidence
        ]

        # Check if we used fallback (all confidence == 0.5)
        used_fallback = all(s.confidence == 0.5 for s in suggestions)

        return DraftReplyResponse(
            suggestions=suggestions,
            context_used=ContextInfo(messages_used=0, tokens_used=0, truncated=False),
            error=None,
            used_fallback=used_fallback,
        )

    except TimeoutError:
        logger.warning("Generation timed out after %ds", GENERATION_TIMEOUT_SECONDS)
        fallback = get_fallback_response(FailureReason.GENERATION_TIMEOUT)
        return DraftReplyResponse(
            suggestions=[
                Suggestion(text=t, confidence=0.5) for t in get_fallback_reply_suggestions()
            ],
            error=fallback.text,
            used_fallback=True,
        )

    except ModelLoadError as e:
        logger.warning("Model load failed: %s", str(e))
        fallback = get_fallback_response(FailureReason.MODEL_LOAD_FAILED)
        return DraftReplyResponse(
            suggestions=[
                Suggestion(text=t, confidence=0.5) for t in get_fallback_reply_suggestions()
            ],
            error=fallback.text,
            used_fallback=True,
        )

    except Exception as e:
        logger.exception("Unexpected error in draft generation")
        fallback = get_fallback_response(FailureReason.GENERATION_ERROR)
        return DraftReplyResponse(
            suggestions=[
                Suggestion(text=t, confidence=0.5) for t in get_fallback_reply_suggestions()
            ],
            error=f"{fallback.text}: {str(e)}",
            used_fallback=True,
        )


@router.post("/summary", response_model=SummaryResponse)
async def generate_conversation_summary(request: SummaryRequest) -> SummaryResponse:
    """Generate AI-powered conversation summary.

    Uses the AI model to summarize a conversation.
    Falls back to a generic message on timeout or error.
    """
    # For now, we don't have access to messages, so use a placeholder
    # In a real implementation, this would fetch messages from iMessage
    participant = request.chat_id  # Use chat_id as participant for now
    messages: list[str] = []  # Would be populated from iMessage reader

    try:
        # Run generation with timeout
        loop = asyncio.get_event_loop()
        summary_text, used_fallback = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _generate_summary_sync,
                messages,
                participant,
            ),
            timeout=GENERATION_TIMEOUT_SECONDS,
        )

        return SummaryResponse(
            summary=summary_text,
            participant=participant,
            message_count=len(messages),
            error=None,
            used_fallback=used_fallback,
        )

    except TimeoutError:
        logger.warning("Summary generation timed out after %ds", GENERATION_TIMEOUT_SECONDS)
        fallback = get_fallback_response(FailureReason.GENERATION_TIMEOUT)
        return SummaryResponse(
            summary=get_fallback_summary(participant),
            participant=participant,
            message_count=0,
            error=fallback.text,
            used_fallback=True,
        )

    except Exception as e:
        logger.exception("Unexpected error in summary generation")
        fallback = get_fallback_response(FailureReason.GENERATION_ERROR)
        return SummaryResponse(
            summary=get_fallback_summary(participant),
            participant=participant,
            message_count=0,
            error=f"{fallback.text}: {str(e)}",
            used_fallback=True,
        )


@router.get("/status", response_model=GenerationStatus)
async def get_draft_status() -> GenerationStatus:
    """Get the current status of the draft generation system.

    Returns information about model availability and system health.
    """
    status = get_generation_status()
    return GenerationStatus(
        model_loaded=bool(status["model_loaded"]),
        can_generate=bool(status["can_generate"]),
        reason=str(status["reason"]) if status["reason"] else None,
        memory_mode=str(status["memory_mode"]),
    )
