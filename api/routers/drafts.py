"""AI-powered draft reply generation API endpoints.

Provides endpoints for generating draft replies and conversation summaries
using the MLX language model with conversation context via RAG (Retrieval
Augmented Generation).

These endpoints use the local MLX model to generate contextually appropriate
responses without sending any data to external services.

NOTE: All prompts and examples are imported from jarvis/prompts.py,
which is the single source of truth for all prompts in the JARVIS system.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_imessage_reader
from api.ratelimit import (
    RATE_LIMIT_GENERATION,
    TIMEOUT_GENERATION,
    limiter,
)
from api.schemas import (
    ContextInfo,
    DateRange,
    DraftReplyRequest,
    DraftReplyResponse,
    DraftSuggestion,
    DraftSummaryRequest,
    DraftSummaryResponse,
    ErrorResponse,
)
from contracts.imessage import Message
from contracts.models import GenerationRequest
from integrations.imessage import ChatDBReader
from jarvis.prompts import API_REPLY_EXAMPLES, API_SUMMARY_EXAMPLES
from models import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drafts", tags=["drafts"])

# Use centralized prompts from jarvis/prompts.py
REPLY_EXAMPLES = API_REPLY_EXAMPLES
SUMMARY_EXAMPLES = API_SUMMARY_EXAMPLES


def _format_messages_for_context(messages: list[Message]) -> str:
    """Format messages as context string for RAG.

    Args:
        messages: List of messages (newest first from reader)

    Returns:
        Formatted context string with messages in chronological order
    """
    # Reverse to chronological order (oldest first)
    chronological = list(reversed(messages))

    lines = []
    for msg in chronological:
        sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
        lines.append(f"[{sender}]: {msg.text}")

    return "\n".join(lines)


def _build_reply_prompt(
    last_message: str,
    instruction: str | None,
    suggestion_num: int,
) -> str:
    """Build a prompt for generating a reply suggestion.

    Args:
        last_message: The last message in the conversation
        instruction: Optional user instruction for reply tone/content
        suggestion_num: Which suggestion number (1, 2, 3...) for variety

    Returns:
        Formatted prompt string
    """
    base_prompt = f"Last message: '{last_message}'\nInstruction: {instruction or 'None'}"

    # Add variety hints for different suggestions
    variety_hints = [
        "\nGenerate a natural, conversational reply.",
        "\nGenerate a slightly more casual reply variant.",
        "\nGenerate a concise reply variant.",
    ]

    hint_idx = (suggestion_num - 1) % len(variety_hints)
    return base_prompt + variety_hints[hint_idx]


def _build_summary_prompt(context: str, num_messages: int) -> str:
    """Build a prompt for conversation summarization.

    Args:
        context: Formatted conversation context
        num_messages: Number of messages being summarized

    Returns:
        Formatted prompt string
    """
    return (
        f"Summarize this conversation of {num_messages} messages. "
        "Provide a brief summary and extract 2-4 key points.\n\n"
        f"Conversation:\n{context}\n\n"
        "Provide your response in this format:\n"
        "Summary: [1-2 sentence summary]\n"
        "Key points:\n- [point 1]\n- [point 2]"
    )


def _parse_summary_response(response_text: str) -> tuple[str, list[str]]:
    """Parse the LLM summary response into summary and key points.

    Args:
        response_text: Raw LLM response

    Returns:
        Tuple of (summary, key_points)
    """
    lines = response_text.strip().split("\n")
    summary = ""
    key_points = []

    in_key_points = False
    for line in lines:
        line = line.strip()
        if line.lower().startswith("summary:"):
            summary = line[8:].strip()
        elif line.lower() == "key points:" or line.lower().startswith("key points"):
            in_key_points = True
        elif in_key_points and line.startswith("-"):
            key_points.append(line[1:].strip())
        elif in_key_points and line.startswith("•"):
            key_points.append(line[1:].strip())

    # Fallback if parsing fails
    if not summary:
        summary = response_text[:200] if len(response_text) > 200 else response_text
    if not key_points:
        key_points = ["See summary for details"]

    return summary, key_points


def _generate_single_suggestion(
    generator: object,
    prompt: str,
    context_text: str,
    temperature: float,
) -> str | None:
    """Generate a single reply suggestion synchronously.

    This runs in a thread pool to avoid blocking the event loop.

    Args:
        generator: The MLX generator instance.
        prompt: The prompt to generate from.
        context_text: Conversation context.
        temperature: Sampling temperature.

    Returns:
        Generated text or None if generation failed.
    """
    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[context_text],
        few_shot_examples=REPLY_EXAMPLES,
        max_tokens=200,
        temperature=temperature,
    )
    try:
        response = generator.generate(gen_request)  # type: ignore[attr-defined]
        return response.text.strip()
    except Exception as e:
        logger.warning("Generation failed: %s", e)
        return None


@router.post(
    "/reply",
    response_model=DraftReplyResponse,
    response_model_exclude_unset=True,
    response_description="AI-generated reply suggestions with context metadata",
    summary="Generate draft reply suggestions",
    responses={
        200: {
            "description": "Reply suggestions generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "suggestions": [
                            {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                            {"text": "Sure! Let me know the details.", "confidence": 0.8},
                            {"text": "Sounds good to me!", "confidence": 0.7},
                        ],
                        "context_used": {
                            "num_messages": 20,
                            "participants": ["John Doe"],
                            "last_message": "Are you free for dinner tonight?",
                        },
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Conversation not found or no messages",
            "model": ErrorResponse,
        },
        408: {
            "description": "Request timed out",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
        500: {
            "description": "Failed to generate suggestions",
            "model": ErrorResponse,
        },
        503: {
            "description": "Model service unavailable",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def generate_draft_reply(
    draft_request: DraftReplyRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftReplyResponse:
    """Generate AI-powered reply suggestions for a conversation.

    Uses the local MLX language model with conversation context to generate
    contextually appropriate reply suggestions. All processing is done locally
    without sending data to external services.

    **How It Works:**
    1. Retrieves recent messages from the conversation for context
    2. Formats the conversation as input for the language model
    3. Generates multiple reply suggestions with varying tones
    4. Returns suggestions ranked by confidence

    **Rate Limiting:**
    This endpoint is rate limited to 10 requests per minute to prevent
    resource exhaustion from CPU-intensive model generation.

    **Customizing Replies:**
    Use the `instruction` parameter to guide the tone or content:
    - "accept enthusiastically" - positive, excited response
    - "politely decline" - courteous refusal
    - "ask for more details" - clarifying questions
    - "be brief" - short, concise reply
    - "be formal" - professional tone

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456789",
        "instruction": "accept enthusiastically",
        "num_suggestions": 3,
        "context_messages": 20
    }
    ```

    **Example Response:**
    ```json
    {
        "suggestions": [
            {"text": "Yes, I'd love to! What time works for you?", "confidence": 0.9},
            {"text": "Absolutely! Count me in!", "confidence": 0.8},
            {"text": "Sure thing! Looking forward to it!", "confidence": 0.7}
        ],
        "context_used": {
            "num_messages": 20,
            "participants": ["John Doe"],
            "last_message": "Are you free for dinner tonight?"
        }
    }
    ```

    Args:
        draft_request: DraftReplyRequest with chat_id, optional instruction,
                 num_suggestions (1-5), and context_messages (5-50)
        request: FastAPI request object (for rate limiting)

    Returns:
        DraftReplyResponse with list of suggestions and context metadata

    Raises:
        HTTPException 403: Full Disk Access not granted
        HTTPException 404: No messages found for the conversation
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Failed to generate suggestions
        HTTPException 503: Model service unavailable
    """
    # Fetch conversation messages for context (I/O bound - run in threadpool)
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=draft_request.chat_id,
            limit=draft_request.context_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", draft_request.chat_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation context: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {draft_request.chat_id}",
        )

    # Build context info
    last_message = messages[0].text if messages else None
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_text = _format_messages_for_context(messages)

    # Get the generator
    try:
        generator = await run_in_threadpool(get_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate multiple suggestions with timeout
    suggestions: list[DraftSuggestion] = []

    try:
        async with asyncio.timeout(TIMEOUT_GENERATION):
            for i in range(draft_request.num_suggestions):
                prompt = _build_reply_prompt(
                    last_message=last_message or "",
                    instruction=draft_request.instruction,
                    suggestion_num=i + 1,
                )

                # Run generation in threadpool (CPU-bound)
                text = await run_in_threadpool(
                    _generate_single_suggestion,
                    generator,
                    prompt,
                    context_text,
                    0.7 + (i * 0.1),  # Vary temperature for diversity
                )

                if text:
                    confidence = max(0.5, 0.9 - (i * 0.1))
                    suggestions.append(
                        DraftSuggestion(
                            text=text,
                            confidence=confidence,
                        )
                    )
    except TimeoutError:
        logger.warning("Generation timed out after %s seconds", TIMEOUT_GENERATION)
        if not suggestions:
            raise HTTPException(
                status_code=408,
                detail=f"Request timed out after {TIMEOUT_GENERATION} seconds",
            ) from None

    if not suggestions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate any reply suggestions",
        )

    return DraftReplyResponse(
        suggestions=suggestions,
        context_used=ContextInfo(
            num_messages=len(messages),
            participants=participants,
            last_message=last_message,
        ),
    )


def _generate_summary(generator: object, prompt: str) -> str:
    """Generate a summary synchronously.

    This runs in a thread pool to avoid blocking the event loop.

    Args:
        generator: The MLX generator instance.
        prompt: The prompt to generate from.

    Returns:
        Generated summary text.
    """
    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[],  # Context already in prompt
        few_shot_examples=SUMMARY_EXAMPLES,
        max_tokens=500,
        temperature=0.5,  # Lower temperature for more focused summary
    )
    response = generator.generate(gen_request)  # type: ignore[attr-defined]
    return response.text


@router.post(
    "/summarize",
    response_model=DraftSummaryResponse,
    response_model_exclude_unset=True,
    response_description="AI-generated conversation summary with key points",
    summary="Summarize a conversation",
    responses={
        200: {
            "description": "Summary generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "summary": "Discussion about planning a weekend trip to the beach.",
                        "key_points": [
                            "Decided on Saturday departure",
                            "Meeting at John's place at 9am",
                            "Everyone bringing snacks",
                        ],
                        "date_range": {"start": "2024-01-10", "end": "2024-01-15"},
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "Conversation not found or no messages",
            "model": ErrorResponse,
        },
        408: {
            "description": "Request timed out",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
        500: {
            "description": "Failed to generate summary",
            "model": ErrorResponse,
        },
        503: {
            "description": "Model service unavailable",
            "model": ErrorResponse,
        },
    },
)
@limiter.limit(RATE_LIMIT_GENERATION)
async def summarize_conversation(
    summary_request: DraftSummaryRequest,
    request: Request,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftSummaryResponse:
    """Summarize a conversation using AI.

    Analyzes the specified number of messages from a conversation and generates
    a concise summary with key points. Uses the local MLX model for processing.

    **Rate Limiting:**
    This endpoint is rate limited to 10 requests per minute to prevent
    resource exhaustion from CPU-intensive model generation.

    **What You Get:**
    - A 1-2 sentence summary of the conversation
    - 2-4 key points extracted from the discussion
    - The date range of messages included in the summary

    **Use Cases:**
    - Catch up on a conversation you've missed
    - Get a quick overview of a long group chat
    - Find important decisions or action items

    **Example Request:**
    ```json
    {
        "chat_id": "chat123456789",
        "num_messages": 50
    }
    ```

    **Example Response:**
    ```json
    {
        "summary": "Discussion about planning a weekend trip to the beach.",
        "key_points": [
            "Decided on Saturday departure at 9am",
            "Meeting at John's place",
            "Everyone bringing snacks and sunscreen",
            "Return planned for Sunday evening"
        ],
        "date_range": {
            "start": "2024-01-10",
            "end": "2024-01-15"
        }
    }
    ```

    Args:
        summary_request: DraftSummaryRequest with chat_id and num_messages (10-200)
        request: FastAPI request object (for rate limiting)

    Returns:
        DraftSummaryResponse with summary, key_points, and date_range

    Raises:
        HTTPException 403: Full Disk Access not granted
        HTTPException 404: No messages found for the conversation
        HTTPException 408: Request timed out
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Failed to generate summary
        HTTPException 503: Model service unavailable
    """
    # Fetch conversation messages (I/O bound - run in threadpool)
    try:
        messages = await run_in_threadpool(
            reader.get_messages,
            chat_id=summary_request.chat_id,
            limit=summary_request.num_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", summary_request.chat_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {summary_request.chat_id}",
        )

    # Build context
    context_text = _format_messages_for_context(messages)

    # Determine date range (messages are newest-first)
    newest_date = messages[0].date
    oldest_date = messages[-1].date

    # Get the generator
    try:
        generator = await run_in_threadpool(get_generator)
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate summary with timeout
    prompt = _build_summary_prompt(context_text, len(messages))

    try:
        async with asyncio.timeout(TIMEOUT_GENERATION):
            response_text = await run_in_threadpool(
                _generate_summary,
                generator,
                prompt,
            )
            summary, key_points = _parse_summary_response(response_text)
    except TimeoutError:
        logger.warning("Summary generation timed out after %s seconds", TIMEOUT_GENERATION)
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {TIMEOUT_GENERATION} seconds",
        ) from None
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate conversation summary",
        ) from e

    return DraftSummaryResponse(
        summary=summary,
        key_points=key_points,
        date_range=DateRange(
            start=oldest_date.strftime("%Y-%m-%d"),
            end=newest_date.strftime("%Y-%m-%d"),
        ),
    )
