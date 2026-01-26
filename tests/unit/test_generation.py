"""Unit tests for health-aware generation.

Tests the generation utilities that check system health before using the LLM.
"""

from unittest.mock import patch

from contracts.memory import MemoryMode, MemoryState
from contracts.models import GenerationRequest, GenerationResponse
from jarvis.fallbacks import get_fallback_reply_suggestions
from jarvis.generation import (
    can_use_llm,
    generate_reply_suggestions,
    generate_summary,
    generate_with_fallback,
    get_generation_status,
)


class TestCanUseLLM:
    """Tests for can_use_llm function."""

    def test_returns_true_in_full_mode(self):
        """Returns True when memory mode is FULL."""
        mock_state = MemoryState(
            available_mb=10000,
            used_mb=6000,
            model_loaded=False,
            current_mode=MemoryMode.FULL,
            pressure_level="green",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            can_use, reason = can_use_llm()

        assert can_use is True
        assert reason == ""

    def test_returns_true_in_lite_mode(self):
        """Returns True when memory mode is LITE."""
        mock_state = MemoryState(
            available_mb=6000,
            used_mb=10000,
            model_loaded=False,
            current_mode=MemoryMode.LITE,
            pressure_level="yellow",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            can_use, reason = can_use_llm()

        assert can_use is True
        assert reason == ""

    def test_returns_false_in_minimal_mode(self):
        """Returns False when memory mode is MINIMAL."""
        mock_state = MemoryState(
            available_mb=2000,
            used_mb=14000,
            model_loaded=False,
            current_mode=MemoryMode.MINIMAL,
            pressure_level="red",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            can_use, reason = can_use_llm()

        assert can_use is False
        assert "memory" in reason.lower()

    def test_returns_false_on_critical_pressure(self):
        """Returns False when memory pressure is critical."""
        mock_state = MemoryState(
            available_mb=5000,
            used_mb=11000,
            model_loaded=True,
            current_mode=MemoryMode.LITE,
            pressure_level="critical",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            can_use, reason = can_use_llm()

        assert can_use is False
        assert "critical" in reason.lower()

    def test_returns_false_on_red_pressure(self):
        """Returns False when memory pressure is red."""
        mock_state = MemoryState(
            available_mb=4000,
            used_mb=12000,
            model_loaded=True,
            current_mode=MemoryMode.LITE,
            pressure_level="red",
        )

        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.return_value.get_state.return_value = mock_state
            can_use, reason = can_use_llm()

        assert can_use is False
        assert "pressure" in reason.lower()

    def test_returns_true_on_exception(self):
        """Returns True when memory check fails (err on side of trying)."""
        with patch("jarvis.generation.get_memory_controller") as mock_controller:
            mock_controller.side_effect = RuntimeError("Failed to get controller")
            can_use, reason = can_use_llm()

        assert can_use is True
        assert reason == ""


class TestGetGenerationStatus:
    """Tests for get_generation_status function."""

    def test_returns_all_fields(self):
        """Returns dictionary with all expected fields."""
        mock_state = MemoryState(
            available_mb=8000,
            used_mb=8000,
            model_loaded=True,
            current_mode=MemoryMode.FULL,
            pressure_level="green",
        )

        with (
            patch("jarvis.generation.get_generator") as mock_gen,
            patch("jarvis.generation.get_memory_controller") as mock_mem,
        ):
            mock_gen.return_value.is_loaded.return_value = True
            mock_mem.return_value.get_state.return_value = mock_state
            mock_mem.return_value.get_mode.return_value = MemoryMode.FULL

            status = get_generation_status()

        assert "model_loaded" in status
        assert "can_generate" in status
        assert "reason" in status
        assert "memory_mode" in status

    def test_handles_generator_exception(self):
        """Returns model_loaded=False on generator exception."""
        with (
            patch("jarvis.generation.get_generator") as mock_gen,
            patch("jarvis.generation.get_memory_controller") as mock_mem,
        ):
            mock_gen.side_effect = RuntimeError("No generator")
            mock_mem.return_value.get_mode.return_value = MemoryMode.LITE

            status = get_generation_status()

        assert status["model_loaded"] is False


class TestGenerateWithFallback:
    """Tests for generate_with_fallback function."""

    def test_uses_fallback_when_cannot_use_llm(self):
        """Returns fallback when LLM cannot be used."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Memory too low")
            response = generate_with_fallback(request)

        assert response.finish_reason == "fallback"
        assert response.error == "Memory too low"
        assert response.model_name == "fallback"

    def test_calls_generator_when_can_use_llm(self):
        """Calls generator when system is healthy."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        mock_response = GenerationResponse(
            text="Generated text",
            tokens_used=10,
            generation_time_ms=100.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.get_generator") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value.is_loaded.return_value = True
            mock_gen.return_value.generate.return_value = mock_response

            response = generate_with_fallback(request)

        assert response.text == "Generated text"
        assert response.finish_reason == "stop"

    def test_returns_fallback_on_generation_error(self):
        """Returns fallback when generation fails."""
        request = GenerationRequest(
            prompt="Test prompt",
            context_documents=[],
            few_shot_examples=[],
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.get_generator") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value.is_loaded.return_value = True
            mock_gen.return_value.generate.side_effect = RuntimeError("Generation failed")

            response = generate_with_fallback(request)

        assert response.finish_reason == "error"
        assert "Generation failed" in response.error


class TestGenerateReplySuggestions:
    """Tests for generate_reply_suggestions function."""

    def test_returns_fallback_when_cannot_use_llm(self):
        """Returns fallback suggestions when LLM unavailable."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Memory low")
            suggestions = generate_reply_suggestions("Hello", num_suggestions=3)

        assert len(suggestions) > 0
        # All fallback suggestions have confidence 0.5
        assert all(conf == 0.5 for _, conf in suggestions)

    def test_returns_specified_number_of_suggestions(self):
        """Returns requested number of suggestions."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Test")
            suggestions = generate_reply_suggestions("Hello", num_suggestions=2)

        assert len(suggestions) <= 2

    def test_handles_exception_gracefully(self):
        """Returns fallback on any exception."""
        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.generate_with_fallback") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.side_effect = RuntimeError("Unexpected error")

            suggestions = generate_reply_suggestions("Hello")

        assert len(suggestions) > 0
        # Should be fallback suggestions
        fallbacks = get_fallback_reply_suggestions()
        assert suggestions[0][0] in fallbacks


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_returns_fallback_when_cannot_use_llm(self):
        """Returns fallback summary when LLM unavailable."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (False, "Memory low")
            summary, used_fallback = generate_summary(["msg1", "msg2"], "John")

        assert used_fallback is True
        assert "John" in summary

    def test_returns_fallback_for_empty_messages(self):
        """Returns fallback for empty message list."""
        with patch("jarvis.generation.can_use_llm") as mock_check:
            mock_check.return_value = (True, "")
            summary, used_fallback = generate_summary([], "Jane")

        assert used_fallback is True
        assert "Jane" in summary

    def test_generates_summary_when_healthy(self):
        """Generates summary when system is healthy."""
        mock_response = GenerationResponse(
            text="This is a summary of the conversation.",
            tokens_used=20,
            generation_time_ms=150.0,
            model_name="test-model",
            used_template=False,
            template_name=None,
            finish_reason="stop",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.generate_with_fallback") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value = mock_response

            summary, used_fallback = generate_summary(["Hello", "Hi there"], "Bob")

        assert used_fallback is False
        assert summary == "This is a summary of the conversation."

    def test_returns_fallback_on_error_response(self):
        """Returns fallback when generation returns error."""
        mock_response = GenerationResponse(
            text="Fallback text",
            tokens_used=0,
            generation_time_ms=0,
            model_name="fallback",
            used_template=True,
            template_name="fallback",
            finish_reason="error",
            error="Generation failed",
        )

        with (
            patch("jarvis.generation.can_use_llm") as mock_check,
            patch("jarvis.generation.generate_with_fallback") as mock_gen,
        ):
            mock_check.return_value = (True, "")
            mock_gen.return_value = mock_response

            summary, used_fallback = generate_summary(["msg"], "Alice")

        assert used_fallback is True
