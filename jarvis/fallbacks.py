"""Fallback responses when AI features fail.

Provides graceful degradation responses when the AI pipeline encounters
errors such as model load failures, generation timeouts, or context issues.
"""

from dataclasses import dataclass
from enum import Enum


class FailureReason(Enum):
    """Reasons why AI features might fail."""

    MODEL_LOAD_FAILED = "model_load_failed"
    GENERATION_TIMEOUT = "generation_timeout"
    GENERATION_ERROR = "generation_error"
    NO_CONTEXT_AVAILABLE = "no_context_available"
    IMESSAGE_ACCESS_DENIED = "imessage_access_denied"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class FallbackResponse:
    """A fallback response with user guidance."""

    text: str
    reason: FailureReason
    suggestion: str  # What user can do


# Pre-defined fallback responses for each failure reason
FALLBACK_RESPONSES: dict[FailureReason, FallbackResponse] = {
    FailureReason.MODEL_LOAD_FAILED: FallbackResponse(
        text="I couldn't load the AI model right now.",
        reason=FailureReason.MODEL_LOAD_FAILED,
        suggestion="Try closing other apps to free memory, or restart JARVIS.",
    ),
    FailureReason.GENERATION_TIMEOUT: FallbackResponse(
        text="Response generation took too long.",
        reason=FailureReason.GENERATION_TIMEOUT,
        suggestion="Try a simpler request or check system resources.",
    ),
    FailureReason.GENERATION_ERROR: FallbackResponse(
        text="An error occurred while generating a response.",
        reason=FailureReason.GENERATION_ERROR,
        suggestion="Try again, or restart JARVIS if the problem persists.",
    ),
    FailureReason.NO_CONTEXT_AVAILABLE: FallbackResponse(
        text="I don't have enough context to generate a helpful response.",
        reason=FailureReason.NO_CONTEXT_AVAILABLE,
        suggestion="Try providing more details or select a conversation first.",
    ),
    FailureReason.IMESSAGE_ACCESS_DENIED: FallbackResponse(
        text="I can't access your iMessage conversations.",
        reason=FailureReason.IMESSAGE_ACCESS_DENIED,
        suggestion="Grant Full Disk Access in System Settings > Privacy & Security.",
    ),
    FailureReason.MEMORY_PRESSURE: FallbackResponse(
        text="System memory is too low for AI generation.",
        reason=FailureReason.MEMORY_PRESSURE,
        suggestion="Close some applications to free up memory.",
    ),
    FailureReason.NETWORK_ERROR: FallbackResponse(
        text="A network error occurred while downloading the model.",
        reason=FailureReason.NETWORK_ERROR,
        suggestion="Check your internet connection and try again.",
    ),
    FailureReason.UNKNOWN: FallbackResponse(
        text="Something went wrong.",
        reason=FailureReason.UNKNOWN,
        suggestion="Try again, or restart JARVIS if the problem persists.",
    ),
}


def get_fallback_response(reason: FailureReason) -> FallbackResponse:
    """Get a fallback response for a given failure reason.

    Args:
        reason: The reason for the failure

    Returns:
        FallbackResponse with helpful text and suggestions
    """
    return FALLBACK_RESPONSES.get(
        reason,
        FALLBACK_RESPONSES[FailureReason.UNKNOWN],
    )


def get_fallback_reply_suggestions() -> list[str]:
    """Return generic reply suggestions when AI fails.

    These are safe, context-free responses that work in most situations.

    Returns:
        List of fallback reply suggestions
    """
    return [
        "Sounds good!",
        "Got it, thanks!",
        "Let me get back to you on that.",
        "Thanks for letting me know!",
    ]


def get_fallback_summary(participant: str) -> str:
    """Return fallback when summary generation fails.

    Args:
        participant: Name or identifier of the conversation participant

    Returns:
        Fallback summary message
    """
    return f"Unable to generate summary for conversation with {participant}. Try again later."


def get_fallback_draft(context: str | None = None) -> str:
    """Return fallback when draft generation fails.

    Args:
        context: Optional context about what the draft was for

    Returns:
        Fallback draft message
    """
    if context:
        return f"Unable to draft a response for: {context}. Please try again."
    return "Unable to generate a draft response. Please try again."


class ModelLoadError(Exception):
    """Raised when the AI model fails to load."""

    def __init__(self, message: str = "Failed to load AI model", reason: str | None = None):
        """Initialize ModelLoadError.

        Args:
            message: Error message
            reason: Optional detailed reason for failure
        """
        self.message = message
        self.reason = reason
        super().__init__(self.message)


class GenerationTimeoutError(Exception):
    """Raised when generation exceeds the timeout."""

    def __init__(
        self,
        message: str = "Generation timed out",
        timeout_seconds: float | None = None,
    ):
        """Initialize GenerationTimeoutError.

        Args:
            message: Error message
            timeout_seconds: The timeout value that was exceeded
        """
        self.message = message
        self.timeout_seconds = timeout_seconds
        super().__init__(self.message)


class GenerationError(Exception):
    """Raised when generation fails for any other reason."""

    def __init__(self, message: str = "Generation failed", cause: Exception | None = None):
        """Initialize GenerationError.

        Args:
            message: Error message
            cause: The underlying exception that caused the failure
        """
        self.message = message
        self.cause = cause
        super().__init__(self.message)
