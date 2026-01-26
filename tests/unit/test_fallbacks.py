"""Unit tests for fallback responses and error handling.

Tests the fallback system for graceful degradation when AI features fail.
"""

import pytest

from jarvis.fallbacks import (
    FailureReason,
    FallbackResponse,
    GenerationError,
    GenerationTimeoutError,
    ModelLoadError,
    get_fallback_draft,
    get_fallback_reply_suggestions,
    get_fallback_response,
    get_fallback_summary,
)


class TestFailureReason:
    """Tests for FailureReason enum."""

    def test_all_failure_reasons_have_values(self):
        """All failure reasons have string values."""
        for reason in FailureReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_failure_reason_values_are_unique(self):
        """All failure reason values are unique."""
        values = [reason.value for reason in FailureReason]
        assert len(values) == len(set(values))


class TestFallbackResponse:
    """Tests for FallbackResponse dataclass."""

    def test_create_fallback_response(self):
        """Can create a FallbackResponse."""
        response = FallbackResponse(
            text="Test message",
            reason=FailureReason.UNKNOWN,
            suggestion="Try again",
        )
        assert response.text == "Test message"
        assert response.reason == FailureReason.UNKNOWN
        assert response.suggestion == "Try again"


class TestGetFallbackResponse:
    """Tests for get_fallback_response function."""

    def test_returns_response_for_known_reason(self):
        """Returns correct response for known failure reasons."""
        for reason in FailureReason:
            response = get_fallback_response(reason)
            assert isinstance(response, FallbackResponse)
            assert response.reason == reason
            assert len(response.text) > 0
            assert len(response.suggestion) > 0

    def test_model_load_failed_response(self):
        """MODEL_LOAD_FAILED has appropriate message."""
        response = get_fallback_response(FailureReason.MODEL_LOAD_FAILED)
        assert "model" in response.text.lower()
        assert "memory" in response.suggestion.lower() or "restart" in response.suggestion.lower()

    def test_generation_timeout_response(self):
        """GENERATION_TIMEOUT has appropriate message."""
        response = get_fallback_response(FailureReason.GENERATION_TIMEOUT)
        assert "long" in response.text.lower() or "timeout" in response.text.lower()

    def test_imessage_access_denied_response(self):
        """IMESSAGE_ACCESS_DENIED has appropriate message."""
        response = get_fallback_response(FailureReason.IMESSAGE_ACCESS_DENIED)
        assert "imessage" in response.text.lower() or "access" in response.text.lower()
        assert (
            "full disk" in response.suggestion.lower() or "privacy" in response.suggestion.lower()
        )


class TestGetFallbackReplySuggestions:
    """Tests for get_fallback_reply_suggestions function."""

    def test_returns_non_empty_list(self):
        """Returns a non-empty list of suggestions."""
        suggestions = get_fallback_reply_suggestions()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_all_suggestions_are_strings(self):
        """All suggestions are non-empty strings."""
        suggestions = get_fallback_reply_suggestions()
        for s in suggestions:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_suggestions_are_appropriate(self):
        """Suggestions are generic and appropriate for most contexts."""
        suggestions = get_fallback_reply_suggestions()
        # Should have polite, neutral responses
        texts = " ".join(suggestions).lower()
        assert any(word in texts for word in ["thanks", "good", "got it", "know"])


class TestGetFallbackSummary:
    """Tests for get_fallback_summary function."""

    def test_includes_participant_name(self):
        """Fallback summary includes participant name."""
        summary = get_fallback_summary("John")
        assert "John" in summary

    def test_indicates_failure(self):
        """Fallback summary indicates it couldn't generate."""
        summary = get_fallback_summary("Jane")
        assert "unable" in summary.lower() or "try" in summary.lower()


class TestGetFallbackDraft:
    """Tests for get_fallback_draft function."""

    def test_without_context(self):
        """Returns generic message without context."""
        draft = get_fallback_draft()
        assert "unable" in draft.lower() or "draft" in draft.lower()

    def test_with_context(self):
        """Includes context in message."""
        draft = get_fallback_draft(context="meeting invitation")
        assert "meeting invitation" in draft.lower()


class TestModelLoadError:
    """Tests for ModelLoadError exception."""

    def test_default_message(self):
        """Has sensible default message."""
        error = ModelLoadError()
        assert "model" in str(error).lower()

    def test_custom_message(self):
        """Can provide custom message."""
        error = ModelLoadError("Custom error message")
        assert error.message == "Custom error message"

    def test_with_reason(self):
        """Can include reason."""
        error = ModelLoadError("Failed", reason="Out of memory")
        assert error.reason == "Out of memory"

    def test_is_exception(self):
        """Is a proper exception that can be raised."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("Test error")


class TestGenerationTimeoutError:
    """Tests for GenerationTimeoutError exception."""

    def test_default_message(self):
        """Has sensible default message."""
        error = GenerationTimeoutError()
        assert "timed out" in str(error).lower()

    def test_with_timeout_value(self):
        """Can include timeout value."""
        error = GenerationTimeoutError("Timed out", timeout_seconds=30.0)
        assert error.timeout_seconds == 30.0

    def test_is_exception(self):
        """Is a proper exception that can be raised."""
        with pytest.raises(GenerationTimeoutError):
            raise GenerationTimeoutError("Test timeout")


class TestGenerationError:
    """Tests for GenerationError exception."""

    def test_default_message(self):
        """Has sensible default message."""
        error = GenerationError()
        assert "generation" in str(error).lower() or "failed" in str(error).lower()

    def test_with_cause(self):
        """Can include cause exception."""
        cause = ValueError("Underlying error")
        error = GenerationError("Generation failed", cause=cause)
        assert error.cause is cause

    def test_is_exception(self):
        """Is a proper exception that can be raised."""
        with pytest.raises(GenerationError):
            raise GenerationError("Test error")
