"""Tests for the evaluation module.

Tests tone analysis, response evaluation, and feedback tracking.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.evaluation import (
    EvaluationResult,
    FeedbackAction,
    FeedbackEntry,
    FeedbackStore,
    ResponseEvaluator,
    ToneAnalysis,
    ToneAnalyzer,
    get_feedback_store,
    get_response_evaluator,
    reset_evaluation,
)


@pytest.fixture
def mock_sentence_model():
    """Mock the sentence model to avoid heavy imports."""
    mock_model = MagicMock()
    # Return mock embeddings (simple vectors)
    import numpy as np
    mock_model.encode.return_value = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    with patch("jarvis.evaluation.ResponseEvaluator._get_sentence_model", return_value=mock_model):
        yield mock_model


class TestToneAnalyzer:
    """Tests for ToneAnalyzer."""

    def test_analyze_empty_text(self):
        """Analyze empty text returns neutral values."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("")

        assert result.formality_score == 0.5
        assert result.emoji_density == 0.0

    def test_analyze_casual_text(self):
        """Analyze casual text detects low formality."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("hey! lol thats so cool!! 😊")

        assert result.formality_score < 0.5
        assert result.emoji_density > 0
        assert result.abbreviation_count > 0

    def test_analyze_formal_text(self):
        """Analyze formal text detects high formality."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze(
            "I would appreciate it if you could please review the attached document. "
            "Furthermore, I believe this matter requires your immediate attention."
        )

        assert result.formality_score > 0.5
        assert result.abbreviation_count == 0

    def test_analyze_emoji_density(self):
        """Analyze correctly calculates emoji density."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("😊😊😊😊😊")

        assert result.emoji_density > 0

    def test_analyze_exclamation_rate(self):
        """Analyze correctly calculates exclamation rate."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("Hello! How are you! Great!")

        assert result.exclamation_rate > 0

    def test_analyze_question_rate(self):
        """Analyze correctly calculates question rate."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("How are you? What time? Where?")

        assert result.question_rate > 0

    def test_analyze_average_sentence_length(self):
        """Analyze correctly calculates average sentence length."""
        analyzer = ToneAnalyzer()
        result = analyzer.analyze("One two three. Four five six seven.")

        assert result.avg_sentence_length > 0

    def test_compute_tone_similarity_identical(self):
        """Identical tones have similarity 1.0."""
        analyzer = ToneAnalyzer()
        tone = ToneAnalysis(
            formality_score=0.5,
            emoji_density=1.0,
            exclamation_rate=0.5,
            question_rate=0.2,
            avg_sentence_length=10.0,
            abbreviation_count=2,
        )

        similarity = analyzer.compute_tone_similarity(tone, tone)
        assert similarity == 1.0

    def test_compute_tone_similarity_different(self):
        """Very different tones have low similarity."""
        analyzer = ToneAnalyzer()
        formal = ToneAnalysis(
            formality_score=0.9,
            emoji_density=0.0,
            exclamation_rate=0.0,
            question_rate=0.1,
            avg_sentence_length=25.0,
            abbreviation_count=0,
        )
        casual = ToneAnalysis(
            formality_score=0.1,
            emoji_density=5.0,
            exclamation_rate=2.0,
            question_rate=0.0,
            avg_sentence_length=5.0,
            abbreviation_count=5,
        )

        similarity = analyzer.compute_tone_similarity(formal, casual)
        assert similarity < 0.5


class TestResponseEvaluator:
    """Tests for ResponseEvaluator."""

    def test_evaluate_returns_result(self, mock_sentence_model):
        """Evaluate returns EvaluationResult."""
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(
            response="Sounds great!",
            context_messages=["Hey, want to grab lunch?"],
        )

        assert isinstance(result, EvaluationResult)
        assert 0 <= result.tone_score <= 1
        assert 0 <= result.relevance_score <= 1
        assert 0 <= result.naturalness_score <= 1
        assert 0 <= result.length_score <= 1
        assert 0 <= result.overall_score <= 1

    def test_evaluate_empty_context(self, mock_sentence_model):
        """Evaluate with empty context returns neutral scores."""
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(
            response="Hello there",
            context_messages=[],
        )

        assert result.tone_score == 0.5
        assert result.relevance_score == 0.5

    def test_evaluate_with_user_messages(self, mock_sentence_model):
        """Evaluate uses user messages for length scoring."""
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(
            response="Yes!",
            context_messages=["Are you coming?"],
            user_messages=["Ok", "Sure", "Yes"],
        )

        # Short response should match short user messages
        assert result.length_score > 0.5

    def test_naturalness_detects_repetition(self, mock_sentence_model):
        """Naturalness score penalizes repetition."""
        evaluator = ResponseEvaluator()
        repetitive = evaluator.evaluate(
            response="the the the the the dog",
            context_messages=["Hello"],
        )
        normal = evaluator.evaluate(
            response="I love my dog",
            context_messages=["Hello"],
        )

        assert repetitive.naturalness_score < normal.naturalness_score

    def test_naturalness_detects_robotic_phrases(self, mock_sentence_model):
        """Naturalness score penalizes robotic phrases."""
        evaluator = ResponseEvaluator()
        robotic = evaluator.evaluate(
            response="As an AI, I cannot help you with that.",
            context_messages=["Hello"],
        )
        natural = evaluator.evaluate(
            response="Sorry, I can't help with that.",
            context_messages=["Hello"],
        )

        assert robotic.naturalness_score < natural.naturalness_score

    def test_naturalness_rewards_contractions(self, mock_sentence_model):
        """Naturalness score rewards contractions."""
        evaluator = ResponseEvaluator()
        with_contractions = evaluator.evaluate(
            response="I'm going to the store, I'll be back soon.",
            context_messages=["Where are you?"],
        )
        without_contractions = evaluator.evaluate(
            response="I am going to the store. I will be back soon.",
            context_messages=["Where are you?"],
        )

        assert with_contractions.naturalness_score >= without_contractions.naturalness_score

    def test_evaluate_details_include_metadata(self, mock_sentence_model):
        """Evaluation details include metadata."""
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(
            response="Hello",
            context_messages=["Hi", "How are you?"],
            user_messages=["Good"],
        )

        assert "context_message_count" in result.details
        assert result.details["context_message_count"] == 2
        assert result.details["user_message_count"] == 1


class TestFeedbackStore:
    """Tests for FeedbackStore."""

    def test_record_feedback_creates_entry(self):
        """Recording feedback creates an entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            entry = store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Hello there!",
                chat_id="chat123",
                context_messages=["Hi"],
            )

            assert isinstance(entry, FeedbackEntry)
            assert entry.action == FeedbackAction.SENT
            assert entry.suggestion_text == "Hello there!"
            assert entry.chat_id == "chat123"

    def test_record_feedback_generates_ids(self):
        """Recording feedback generates IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            entry = store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Hello there!",
                chat_id="chat123",
                context_messages=["Hi"],
            )

            assert len(entry.suggestion_id) == 16
            assert len(entry.context_hash) == 16

    def test_record_feedback_edited_stores_text(self):
        """Recording edited feedback stores both texts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            entry = store.record_feedback(
                action=FeedbackAction.EDITED,
                suggestion_text="Hello!",
                chat_id="chat123",
                context_messages=["Hi"],
                edited_text="Hi there!",
            )

            assert entry.suggestion_text == "Hello!"
            assert entry.edited_text == "Hi there!"

    def test_record_feedback_with_evaluation(self):
        """Recording feedback with evaluation stores scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            evaluation = EvaluationResult(
                tone_score=0.8,
                relevance_score=0.7,
                naturalness_score=0.9,
                length_score=0.75,
                overall_score=0.79,
            )
            entry = store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Hello!",
                chat_id="chat123",
                context_messages=["Hi"],
                evaluation=evaluation,
            )

            assert entry.evaluation is not None
            assert entry.evaluation.tone_score == 0.8

    def test_get_stats_empty(self):
        """Get stats on empty store returns zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            stats = store.get_stats()

            assert stats["total_feedback"] == 0
            assert stats["acceptance_rate"] == 0.0

    def test_get_stats_with_feedback(self):
        """Get stats calculates correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))

            # Add some feedback
            for _ in range(3):
                store.record_feedback(
                    action=FeedbackAction.SENT,
                    suggestion_text="Hello",
                    chat_id="chat123",
                    context_messages=["Hi"],
                )

            for _ in range(2):
                store.record_feedback(
                    action=FeedbackAction.EDITED,
                    suggestion_text="Hello",
                    chat_id="chat123",
                    context_messages=["Hi"],
                    edited_text="Hi there",
                )

            store.record_feedback(
                action=FeedbackAction.DISMISSED,
                suggestion_text="Hello",
                chat_id="chat123",
                context_messages=["Hi"],
            )

            stats = store.get_stats()

            assert stats["total_feedback"] == 6
            assert stats["sent_unchanged"] == 3
            assert stats["edited"] == 2
            assert stats["dismissed"] == 1
            # acceptance_rate = 3 / 6 = 0.5
            assert stats["acceptance_rate"] == 0.5

    def test_get_improvements_empty(self):
        """Get improvements on empty store returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            improvements = store.get_improvements()

            assert improvements == []

    def test_get_improvements_with_edits(self):
        """Get improvements analyzes edit patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))

            # Add edited feedback with consistent shortening
            for i in range(10):
                store.record_feedback(
                    action=FeedbackAction.EDITED,
                    suggestion_text="This is a very long response that goes on and on.",
                    chat_id="chat123",
                    context_messages=["Hi"],
                    edited_text="Short response.",
                )

            improvements = store.get_improvements()

            # Should suggest shorter responses
            assert len(improvements) > 0
            length_suggestions = [i for i in improvements if i["type"] == "length"]
            assert len(length_suggestions) > 0

    def test_get_recent_entries(self):
        """Get recent entries returns in reverse chronological order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))

            store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="First",
                chat_id="chat123",
                context_messages=["Hi"],
            )
            store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Second",
                chat_id="chat123",
                context_messages=["Hi"],
            )
            store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Third",
                chat_id="chat123",
                context_messages=["Hi"],
            )

            entries = store.get_recent_entries(limit=2)

            assert len(entries) == 2
            assert entries[0].suggestion_text == "Third"
            assert entries[1].suggestion_text == "Second"

    def test_persistence(self):
        """Feedback is persisted to disk and reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store and add feedback
            store1 = FeedbackStore(feedback_dir=Path(tmpdir))
            store1.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Persisted message",
                chat_id="chat123",
                context_messages=["Hi"],
            )

            # Create new store instance
            store2 = FeedbackStore(feedback_dir=Path(tmpdir))
            entries = store2.get_recent_entries()

            assert len(entries) == 1
            assert entries[0].suggestion_text == "Persisted message"

    def test_clear(self):
        """Clear removes all feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(feedback_dir=Path(tmpdir))
            store.record_feedback(
                action=FeedbackAction.SENT,
                suggestion_text="Hello",
                chat_id="chat123",
                context_messages=["Hi"],
            )
            store.clear()

            stats = store.get_stats()
            assert stats["total_feedback"] == 0


class TestGlobalSingletons:
    """Tests for global singleton getters."""

    def test_get_response_evaluator_returns_same_instance(self):
        """Get response evaluator returns same instance."""
        reset_evaluation()

        e1 = get_response_evaluator()
        e2 = get_response_evaluator()

        assert e1 is e2

    def test_get_feedback_store_returns_same_instance(self):
        """Get feedback store returns same instance."""
        reset_evaluation()

        s1 = get_feedback_store()
        s2 = get_feedback_store()

        assert s1 is s2

    def test_reset_evaluation_clears_singletons(self):
        """Reset evaluation creates new instances."""
        e1 = get_response_evaluator()
        reset_evaluation()
        e2 = get_response_evaluator()

        assert e1 is not e2


class TestFeedbackAction:
    """Tests for FeedbackAction enum."""

    def test_feedback_action_values(self):
        """FeedbackAction has expected values."""
        assert FeedbackAction.SENT.value == "sent"
        assert FeedbackAction.EDITED.value == "edited"
        assert FeedbackAction.DISMISSED.value == "dismissed"
        assert FeedbackAction.COPIED.value == "copied"

    def test_feedback_action_from_string(self):
        """FeedbackAction can be created from string."""
        assert FeedbackAction("sent") == FeedbackAction.SENT
        assert FeedbackAction("edited") == FeedbackAction.EDITED
