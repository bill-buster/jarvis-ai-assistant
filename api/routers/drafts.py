"""AI-powered draft reply generation API endpoints.

Provides endpoints for generating draft replies using the LLM with
conversation context via RAG.

NOTE: All prompts and examples are imported from jarvis/prompts.py,
which is the single source of truth for all prompts in the JARVIS system.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_imessage_reader
from api.schemas import (
    ContextInfo,
    DateRange,
    DraftReplyRequest,
    DraftReplyResponse,
    DraftSuggestion,
    DraftSummaryRequest,
    DraftSummaryResponse,
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


@router.post("/reply", response_model=DraftReplyResponse)
def generate_draft_reply(
    request: DraftReplyRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftReplyResponse:
    """Generate AI-powered reply suggestions for a conversation.

    Uses the MLX generator with conversation context to produce
    contextually appropriate reply suggestions.
    """
    # Fetch conversation messages for context
    try:
        messages = reader.get_messages(
            chat_id=request.chat_id,
            limit=request.context_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", request.chat_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation context: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {request.chat_id}",
        )

    # Build context info
    last_message = messages[0].text if messages else None
    participants = list({m.sender_name or m.sender for m in messages if not m.is_from_me})
    context_text = _format_messages_for_context(messages)

    # Get the generator
    try:
        generator = get_generator()
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate multiple suggestions
    suggestions: list[DraftSuggestion] = []

    for i in range(request.num_suggestions):
        prompt = _build_reply_prompt(
            last_message=last_message or "",
            instruction=request.instruction,
            suggestion_num=i + 1,
        )

        gen_request = GenerationRequest(
            prompt=prompt,
            context_documents=[context_text],
            few_shot_examples=REPLY_EXAMPLES,
            max_tokens=200,
            temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
        )

        try:
            response = generator.generate(gen_request)
            # Confidence decreases slightly for each suggestion
            confidence = max(0.5, 0.9 - (i * 0.1))
            suggestions.append(
                DraftSuggestion(
                    text=response.text.strip(),
                    confidence=confidence,
                )
            )
        except Exception as e:
            logger.warning("Failed to generate suggestion %d: %s", i + 1, e)
            # Continue trying other suggestions
            continue

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


@router.post("/summarize", response_model=DraftSummaryResponse)
def summarize_conversation(
    request: DraftSummaryRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DraftSummaryResponse:
    """Summarize a conversation using AI.

    Analyzes the specified number of messages and provides a summary
    with key points and date range.
    """
    # Fetch conversation messages
    try:
        messages = reader.get_messages(
            chat_id=request.chat_id,
            limit=request.num_messages,
        )
    except Exception as e:
        logger.error("Failed to fetch messages for chat %s: %s", request.chat_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversation: {e}",
        ) from e

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for chat_id: {request.chat_id}",
        )

    # Build context
    context_text = _format_messages_for_context(messages)

    # Determine date range (messages are newest-first)
    newest_date = messages[0].date
    oldest_date = messages[-1].date

    # Get the generator
    try:
        generator = get_generator()
    except Exception as e:
        logger.error("Failed to get generator: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable",
        ) from e

    # Generate summary
    prompt = _build_summary_prompt(context_text, len(messages))

    gen_request = GenerationRequest(
        prompt=prompt,
        context_documents=[],  # Context already in prompt
        few_shot_examples=SUMMARY_EXAMPLES,
        max_tokens=500,
        temperature=0.5,  # Lower temperature for more focused summary
    )

    try:
        response = generator.generate(gen_request)
        summary, key_points = _parse_summary_response(response.text)
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
