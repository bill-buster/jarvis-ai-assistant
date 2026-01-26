"""Health-aware generation utilities.

Provides generation functions that check system health before attempting
to use the LLM, with automatic fallback on failure.
"""

import logging

from contracts.memory import MemoryMode
from contracts.models import GenerationRequest, GenerationResponse
from core.memory import get_memory_controller
from jarvis.fallbacks import (
    FailureReason,
    ModelLoadError,
    get_fallback_reply_suggestions,
    get_fallback_response,
)
from models import get_generator

logger = logging.getLogger(__name__)


def can_use_llm() -> tuple[bool, str]:
    """Check if LLM can be used based on system state.

    Checks memory pressure and operating mode to determine if it's
    safe to attempt model generation.

    Returns:
        Tuple of (can_use, reason). If can_use is False, reason explains why.
    """
    try:
        mem = get_memory_controller()
        state = mem.get_state()

        if state.current_mode == MemoryMode.MINIMAL:
            return False, "System memory too low for AI generation"

        if state.pressure_level == "critical":
            return False, "Memory pressure critical"

        if state.pressure_level == "red":
            return False, "Memory pressure too high"

        return True, ""

    except Exception as e:
        logger.warning("Failed to check memory state: %s", str(e))
        # If we can't check memory, err on the side of caution
        return True, ""


def get_generation_status() -> dict[str, object]:
    """Get the current status of the generation system.

    Returns:
        Dictionary with model_loaded, can_generate, reason, and memory_mode
    """
    try:
        generator = get_generator()
        model_loaded = generator.is_loaded()
    except Exception:
        model_loaded = False

    can_generate, reason = can_use_llm()

    try:
        mem = get_memory_controller()
        memory_mode = mem.get_mode().value
    except Exception:
        memory_mode = "unknown"

    return {
        "model_loaded": model_loaded,
        "can_generate": can_generate,
        "reason": reason if not can_generate else None,
        "memory_mode": memory_mode,
    }


def generate_with_fallback(request: GenerationRequest) -> GenerationResponse:
    """Generate with automatic fallback on failure.

    First checks if the system is healthy enough to use the LLM.
    If not, or if generation fails, returns a fallback response.

    Args:
        request: Generation request with prompt, context, and examples

    Returns:
        GenerationResponse, possibly with fallback content
    """
    # Check if we can use the LLM
    can_generate, reason = can_use_llm()

    if not can_generate:
        logger.warning("Cannot use LLM: %s", reason)
        fallback = get_fallback_response(FailureReason.MEMORY_PRESSURE)
        return GenerationResponse(
            text=get_fallback_reply_suggestions()[0],
            tokens_used=0,
            generation_time_ms=0.0,
            model_name="fallback",
            used_template=True,
            template_name="fallback",
            finish_reason="fallback",
            error=reason,
        )

    try:
        generator = get_generator()

        # Try to load if not loaded
        if not generator.is_loaded():
            if not generator.load():
                logger.error("Failed to load model")
                fallback = get_fallback_response(FailureReason.MODEL_LOAD_FAILED)
                raise ModelLoadError(fallback.text, reason="Memory or disk issue")

        return generator.generate(request)

    except ModelLoadError as e:
        logger.exception("Model load failed")
        return GenerationResponse(
            text=get_fallback_reply_suggestions()[0],
            tokens_used=0,
            generation_time_ms=0.0,
            model_name="fallback",
            used_template=True,
            template_name="fallback",
            finish_reason="error",
            error=str(e),
        )

    except Exception as e:
        logger.exception("Generation failed")
        return GenerationResponse(
            text=get_fallback_reply_suggestions()[0],
            tokens_used=0,
            generation_time_ms=0.0,
            model_name="fallback",
            used_template=True,
            template_name="fallback",
            finish_reason="error",
            error=str(e),
        )


def generate_reply_suggestions(
    last_message: str,
    context_messages: list[str] | None = None,
    num_suggestions: int = 3,
) -> list[tuple[str, float]]:
    """Generate reply suggestions for a message.

    Args:
        last_message: The message to respond to
        context_messages: Optional list of previous messages for context
        num_suggestions: Number of suggestions to generate

    Returns:
        List of (suggestion_text, confidence) tuples
    """
    # Check if we can use the LLM
    can_generate, reason = can_use_llm()

    if not can_generate:
        logger.info("Using fallback suggestions: %s", reason)
        fallbacks = get_fallback_reply_suggestions()
        return [(text, 0.5) for text in fallbacks[:num_suggestions]]

    try:
        # Build prompt for reply generation
        context_docs = context_messages or []
        prompt = f"Generate a natural reply to: {last_message}"

        request = GenerationRequest(
            prompt=prompt,
            context_documents=context_docs,
            few_shot_examples=[],
            max_tokens=50,
            temperature=0.7,
        )

        response = generate_with_fallback(request)

        if response.finish_reason in ("error", "fallback"):
            # Use fallback suggestions
            fallbacks = get_fallback_reply_suggestions()
            return [(text, 0.5) for text in fallbacks[:num_suggestions]]

        # For now, return the generated text as a single suggestion
        # A more sophisticated implementation would generate multiple suggestions
        return [(response.text, 0.9)]

    except Exception as e:
        logger.warning("Failed to generate suggestions: %s", str(e))
        fallbacks = get_fallback_reply_suggestions()
        return [(text, 0.5) for text in fallbacks[:num_suggestions]]


def generate_summary(
    messages: list[str],
    participant: str,
) -> tuple[str, bool]:
    """Generate a conversation summary.

    Args:
        messages: List of messages to summarize
        participant: Name of the conversation participant

    Returns:
        Tuple of (summary_text, used_fallback)
    """
    from jarvis.fallbacks import get_fallback_summary

    # Check if we can use the LLM
    can_generate, reason = can_use_llm()

    if not can_generate:
        logger.info("Using fallback summary: %s", reason)
        return get_fallback_summary(participant), True

    if not messages:
        return f"No messages found in conversation with {participant}.", True

    try:
        # Build prompt for summarization
        context = "\n".join(messages[-20:])  # Last 20 messages
        prompt = f"Summarize this conversation with {participant}:\n{context}"

        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=150,
            temperature=0.3,  # Lower temperature for summaries
        )

        response = generate_with_fallback(request)

        if response.finish_reason in ("error", "fallback"):
            return get_fallback_summary(participant), True

        return response.text, False

    except Exception as e:
        logger.warning("Failed to generate summary: %s", str(e))
        return get_fallback_summary(participant), True
