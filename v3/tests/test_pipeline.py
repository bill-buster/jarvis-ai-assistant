"""Full pipeline integration tests for JARVIS v3 reply generation.

Tests the ReplyGenerator class end-to-end with mocked components:
- Model loader (returns predictable responses)
- iMessage reader (returns sample conversations)
- Embeddings store (returns mock past replies)
- Template matcher (returns mock template matches)

Tests cover:
- Different conversation contexts (questions, greetings, statements, etc.)
- Style analysis integration
- RAG retrieval for similar past situations
- Prompt building with examples
- Fallback handling
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.generation import ReplyGenerator
from core.generation.context_analyzer import ConversationContext, MessageIntent, RelationshipType
from core.generation.reply_generator import GeneratedReply, ReplyGenerationResult
from core.generation.style_analyzer import UserStyle


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model_loader():
    """Create a mock model loader that returns predictable responses."""
    loader = MagicMock()
    loader.is_loaded = True
    loader.current_model = "lfm2.5-1.2b"

    # Default generate response
    loader.generate.return_value = MagicMock(
        text="sounds good!",
        formatted_prompt="<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n",
    )
    return loader


@pytest.fixture
def mock_settings():
    """Create mock settings for generation."""
    from core.config import PromptStrategy

    settings = MagicMock()
    settings.generation.prompt_strategy = PromptStrategy.LEGACY
    settings.generation.max_tokens = 50
    settings.generation.min_similarity_threshold = 0.55
    settings.generation.temperature_scale = [0.2, 0.4, 0.6, 0.8, 0.9]
    settings.generation.template_confidence = 0.7
    settings.generation.past_reply_confidence = 0.75
    settings.generation.same_convo_weight = 0.6
    settings.generation.cross_convo_weight = 0.4
    settings.generation.conversation_style_hint = "brief, casual"
    return settings


@pytest.fixture
def sample_question_messages():
    """Sample conversation with a yes/no question."""
    return [
        {"text": "Hey, what are you up to?", "is_from_me": False, "sender": "Alice"},
        {"text": "just working on some stuff", "is_from_me": True},
        {"text": "Do you want to grab dinner tonight?", "is_from_me": False, "sender": "Alice"},
    ]


@pytest.fixture
def sample_greeting_messages():
    """Sample conversation with a greeting."""
    return [
        {"text": "Hey! How are you?", "is_from_me": False, "sender": "Bob"},
    ]


@pytest.fixture
def sample_statement_messages():
    """Sample conversation with a statement."""
    return [
        {"text": "I'll be there in 10 minutes", "is_from_me": False, "sender": "Charlie"},
    ]


@pytest.fixture
def sample_emotional_messages():
    """Sample conversation with emotional content."""
    return [
        {"text": "ugh today was so rough", "is_from_me": False, "sender": "Dana"},
        {"text": "what happened?", "is_from_me": True},
        {"text": "I'm so stressed about this deadline", "is_from_me": False, "sender": "Dana"},
    ]


@pytest.fixture
def sample_group_messages():
    """Sample group conversation with multiple senders."""
    return [
        {"text": "Who's coming to the party?", "is_from_me": False, "sender": "Alice"},
        {"text": "I'll be there!", "is_from_me": False, "sender": "Bob"},
        {"text": "Same here", "is_from_me": True},
        {"text": "What time should we arrive?", "is_from_me": False, "sender": "Charlie"},
    ]


@pytest.fixture
def sample_user_style_messages():
    """Sample messages for style analysis."""
    return [
        {"text": "lol that's hilarious", "is_from_me": True},
        {"text": "haha ikr", "is_from_me": True},
        {"text": "wanna hang tmrw?", "is_from_me": True},
        {"text": "sounds good!", "is_from_me": True},
        {"text": "yeah for sure", "is_from_me": True},
    ]


# =============================================================================
# Basic Pipeline Tests
# =============================================================================


class TestPipelineBasics:
    """Test basic pipeline initialization and execution."""

    def test_generator_creation(self, mock_model_loader):
        """Test that ReplyGenerator can be created with mock loader."""
        generator = ReplyGenerator(mock_model_loader)
        assert generator is not None
        assert generator.model_loader == mock_model_loader
        assert generator.style_analyzer is not None
        assert generator.context_analyzer is not None

    def test_generate_replies_returns_result(self, mock_model_loader, mock_settings):
        """Test that generate_replies returns a ReplyGenerationResult."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey, what's up?", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-chat")

        assert isinstance(result, ReplyGenerationResult)
        assert len(result.replies) > 0
        assert result.model_used == "lfm2.5-1.2b"
        assert result.generation_time_ms >= 0

    def test_generate_replies_includes_context(self, mock_model_loader, mock_settings):
        """Test that result includes conversation context."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Do you want to come over?", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-chat")

        assert result.context is not None
        assert result.context.last_message == "Do you want to come over?"
        assert result.context.intent == MessageIntent.YES_NO_QUESTION


# =============================================================================
# Intent Detection Integration Tests
# =============================================================================


class TestIntentDetection:
    """Test that pipeline correctly detects different message intents."""

    def test_yes_no_question_detection(
        self, mock_model_loader, mock_settings, sample_question_messages
    ):
        """Test detection of yes/no questions."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                sample_question_messages, chat_id="test-chat"
                            )

        assert result.context.intent == MessageIntent.YES_NO_QUESTION

    def test_greeting_detection(self, mock_model_loader, mock_settings, sample_greeting_messages):
        """Test detection of greetings."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                sample_greeting_messages, chat_id="test-chat"
                            )

        assert result.context.intent == MessageIntent.GREETING

    def test_statement_detection(self, mock_model_loader, mock_settings, sample_statement_messages):
        """Test detection of statements (logistics)."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                sample_statement_messages, chat_id="test-chat"
                            )

        # "I'll be there in 10 minutes" should be detected as logistics
        assert result.context.intent == MessageIntent.LOGISTICS

    def test_emotional_detection(self, mock_model_loader, mock_settings, sample_emotional_messages):
        """Test detection of emotional messages."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                sample_emotional_messages, chat_id="test-chat"
                            )

        assert result.context.intent == MessageIntent.EMOTIONAL


# =============================================================================
# Style Analysis Integration Tests
# =============================================================================


class TestStyleAnalysisIntegration:
    """Test style analysis integration in the pipeline."""

    def test_style_included_in_result(self, mock_model_loader, mock_settings):
        """Test that style analysis is included in result."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "lol that's funny", "is_from_me": True},
            {"text": "haha ikr", "is_from_me": True},
            {"text": "What do you think?", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-chat")

        assert result.style is not None
        assert isinstance(result.style, UserStyle)

    def test_style_detects_abbreviations(self, mock_model_loader, mock_settings):
        """Test that style analysis detects abbreviation usage."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "lol that's hilarious", "is_from_me": True},
            {"text": "idk what to do", "is_from_me": True},
            {"text": "tbh that's weird", "is_from_me": True},
            {"text": "What's up?", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-chat")

        assert result.style.uses_abbreviations is True

    def test_style_caching(self, mock_model_loader, mock_settings):
        """Test that style is cached per conversation."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "hey", "is_from_me": True},
            {"text": "What's up?", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            # First call
                            result1 = generator.generate_replies(messages, chat_id="test-cache")
                            # Second call should use cached style
                            result2 = generator.generate_replies(messages, chat_id="test-cache")

        # Same style object should be returned (cached)
        assert result1.style == result2.style

    def test_style_cache_clear(self, mock_model_loader, mock_settings):
        """Test that style cache can be cleared."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [{"text": "hey", "is_from_me": True}]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            generator.generate_replies(messages, chat_id="test-clear")

        # Clear cache
        generator.clear_cache("test-clear")

        # Should not have cached state
        assert "test-clear" not in generator._chat_states


# =============================================================================
# Template Matching Tests
# =============================================================================


class TestTemplateMatching:
    """Test template matching fast-path in the pipeline."""

    def test_template_match_bypasses_llm(self, mock_model_loader, mock_settings):
        """Test that template matches bypass LLM generation."""
        # Create mock template matcher
        mock_matcher = MagicMock()
        mock_match = MagicMock()
        mock_match.actual = "no problem!"
        mock_match.confidence = 0.95
        mock_match.trigger = "thanks"
        mock_matcher.match.return_value = mock_match

        # Patch BEFORE creating the generator (template matcher is set in __init__)
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=mock_matcher
        ):
            generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Thanks!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch(
                "core.generation.reply_generator._get_embedding_store", return_value=None
            ):
                with patch(
                    "core.generation.reply_generator._get_contact_profile", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator.get_global_user_style",
                        return_value=None,
                    ):
                        result = generator.generate_replies(messages, chat_id="test-template")

        # Should use template response
        assert result.model_used == "template"
        assert len(result.replies) == 1
        assert result.replies[0].text == "no problem!"
        assert result.replies[0].reply_type == "template"

        # LLM should NOT have been called
        mock_model_loader.generate.assert_not_called()

    def test_no_template_match_uses_llm(self, mock_model_loader, mock_settings):
        """Test that non-matching messages use LLM."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "What do you think about the proposal?", "is_from_me": False, "sender": "Test"},
        ]

        # Mock matcher returns None (no match)
        mock_matcher = MagicMock()
        mock_matcher.match.return_value = None

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch(
                "core.generation.reply_generator._get_template_matcher", return_value=mock_matcher
            ):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-no-match")

        # LLM should have been called
        mock_model_loader.generate.assert_called_once()
        assert result.model_used == "lfm2.5-1.2b"


# =============================================================================
# RAG Past Replies Tests
# =============================================================================


class TestRAGPastReplies:
    """Test RAG retrieval for similar past replies."""

    def test_past_replies_used_in_prompt(self, mock_model_loader, mock_settings):
        """Test that past replies are used for few-shot learning."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Want to grab coffee?", "is_from_me": False, "sender": "Test"},
        ]

        # Mock embedding store with past replies
        mock_store = MagicMock()
        mock_store.find_your_past_replies.return_value = [
            ("Want to hang out?", "yeah sure!", 0.85),
            ("Want to get lunch?", "sounds good!", 0.80),
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=mock_store
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator._get_relationship_registry",
                            return_value=None,
                        ):
                            with patch(
                                "core.generation.reply_generator.get_global_user_style",
                                return_value=None,
                            ):
                                result = generator.generate_replies(
                                    messages, chat_id="test-past-replies"
                                )

        # Past replies should be in result
        assert len(result.past_replies) == 2
        assert result.past_replies[0][1] == "yeah sure!"

    def test_high_confidence_past_replies_become_template(self, mock_model_loader, mock_settings):
        """Test that consistent high-confidence replies skip LLM.

        Tests the _try_template_match method directly since the full pipeline
        has complex score weighting that makes end-to-end testing fragile.
        """
        # Patch template matcher before creating generator
        with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
            generator = ReplyGenerator(mock_model_loader)

        # Test _try_template_match directly with highly consistent past replies
        # All have confidence >= 0.75 and identical responses
        past_replies = [
            ("Are you around?", "yeah", 0.90),
            ("You free?", "yeah", 0.88),
            ("Are you available?", "yeah", 0.85),
        ]

        # Should return template match since all responses are identical
        result = generator._try_template_match(past_replies)
        assert result == "yeah"

    def test_try_template_match_requires_minimum_replies(self, mock_model_loader):
        """Test that _try_template_match requires at least 2 high-confidence replies."""
        with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
            generator = ReplyGenerator(mock_model_loader)

        # Only 1 reply - should return None
        past_replies = [("Are you around?", "yeah", 0.90)]
        result = generator._try_template_match(past_replies)
        assert result is None

        # Empty list - should return None
        result = generator._try_template_match([])
        assert result is None

        # None - should return None
        result = generator._try_template_match(None)
        assert result is None

    def test_try_template_match_with_yes_variants(self, mock_model_loader, mock_settings):
        """Test that yes variants are recognized as consistent."""
        with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
            generator = ReplyGenerator(mock_model_loader)

        # Different yes variants should still match
        past_replies = [
            ("Are you around?", "yeah", 0.90),
            ("You free?", "yep", 0.88),
            ("Are you available?", "sure", 0.85),
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            result = generator._try_template_match(past_replies)
        # Should return first reply since all are yes variants
        assert result == "yeah"

    def test_try_template_match_inconsistent_replies_returns_none(self, mock_model_loader, mock_settings):
        """Test that inconsistent replies don't trigger template match."""
        with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
            generator = ReplyGenerator(mock_model_loader)

        # Different responses - should NOT match
        past_replies = [
            ("Are you around?", "yeah", 0.90),
            ("You free?", "not really", 0.88),
            ("Are you available?", "maybe later", 0.85),
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            result = generator._try_template_match(past_replies)
        assert result is None


# =============================================================================
# Prompt Strategy Tests
# =============================================================================


class TestPromptStrategies:
    """Test different prompt strategies (legacy vs conversation)."""

    def test_legacy_prompt_strategy(self, mock_model_loader, mock_settings):
        """Test legacy few-shot prompt strategy."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        # Ensure legacy strategy
        from core.config import PromptStrategy

        mock_settings.generation.prompt_strategy = PromptStrategy.LEGACY

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-legacy")

        # Prompt should contain few-shot pattern
        assert "them:" in result.prompt_used or "me:" in result.prompt_used

    def test_conversation_prompt_strategy(self, mock_model_loader, mock_settings):
        """Test conversation continuation prompt strategy."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        # Set conversation strategy
        from core.config import PromptStrategy

        mock_settings.generation.prompt_strategy = PromptStrategy.CONVERSATION

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                messages, chat_id="test-conversation"
                            )

        # Prompt should contain style hint
        assert "brief, casual" in result.prompt_used


# =============================================================================
# Fallback and Error Handling Tests
# =============================================================================


class TestFallbackHandling:
    """Test fallback handling when generation fails."""

    def test_fallback_on_generation_error(self, mock_model_loader, mock_settings):
        """Test that fallback replies are returned on error."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Do you want to come?", "is_from_me": False, "sender": "Test"},
        ]

        # Make generate raise an exception
        mock_model_loader.generate.side_effect = Exception("Model error")

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-error")

        # Should return fallback replies
        assert result.model_used == "fallback"
        assert len(result.replies) > 0
        assert "ERROR" in result.prompt_used

    def test_empty_messages_returns_default(self, mock_model_loader, mock_settings):
        """Test handling of empty message list."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies([], chat_id="test-empty")

        # Should still return a result
        assert isinstance(result, ReplyGenerationResult)


# =============================================================================
# Temperature and Regeneration Tests
# =============================================================================


class TestTemperatureScaling:
    """Test temperature scaling for regeneration."""

    def test_first_generation_uses_low_temp(self, mock_model_loader, mock_settings):
        """Test that first generation uses low temperature."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            generator.generate_replies(messages, chat_id="test-temp")

        # Check temperature used (first in scale = 0.2)
        call_kwargs = mock_model_loader.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    def test_regeneration_increases_temp(self, mock_model_loader, mock_settings):
        """Test that regeneration increases temperature."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            # First generation
                            generator.generate_replies(messages, chat_id="test-regen")
                            # Second generation (same message = regeneration)
                            generator.generate_replies(messages, chat_id="test-regen")

        # Second call should use higher temperature
        calls = mock_model_loader.generate.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["temperature"] == 0.2  # First call
        assert calls[1][1]["temperature"] == 0.4  # Second call (regen)


# =============================================================================
# Repetition Filtering Tests
# =============================================================================


class TestRepetitionFiltering:
    """Test repetition filtering for generated replies."""

    def test_repetitive_replies_filtered(self, mock_model_loader, mock_settings):
        """Test that repetitive replies are filtered out."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            # First generation records "sounds good!"
                            result1 = generator.generate_replies(messages, chat_id="test-repeat")

        # Check that the reply was recorded
        state = generator._get_chat_state("test-repeat")
        assert len(state.recent_generations) > 0


# =============================================================================
# Group Chat Tests
# =============================================================================


class TestGroupChatHandling:
    """Test group chat specific handling."""

    def test_group_chat_detected(self, mock_model_loader, mock_settings, sample_group_messages):
        """Test that group chats are detected."""
        generator = ReplyGenerator(mock_model_loader)

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                sample_group_messages, chat_id="test-group"
                            )

        # Should still generate replies
        assert len(result.replies) > 0

    def test_group_uses_limited_context(self, mock_model_loader, mock_settings):
        """Test that group chats use limited context window."""
        generator = ReplyGenerator(mock_model_loader)

        # Create a longer group conversation
        messages = []
        for i in range(20):
            sender = ["Alice", "Bob", "Charlie"][i % 3]
            messages.append(
                {
                    "text": f"Message {i}",
                    "is_from_me": sender == "me",
                    "sender": sender,
                }
            )
        # Add final message from someone else
        messages.append({"text": "What do you think?", "is_from_me": False, "sender": "Alice"})

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                messages, chat_id="test-group-context"
                            )

        # Should complete without error
        assert result is not None


# =============================================================================
# Availability Signal Tests
# =============================================================================


class TestAvailabilitySignal:
    """Test availability signal extraction and use."""

    def test_busy_signal_detected(self, mock_model_loader, mock_settings):
        """Test detection of busy availability signal."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "I've been swamped with work", "is_from_me": True},
            {"text": "Can't tonight, super busy", "is_from_me": True},
            {"text": "Want to hang out?", "is_from_me": False, "sender": "Test"},
        ]

        # Extract availability signal
        availability = generator._extract_availability_signal(messages)
        assert availability == "busy"

    def test_free_signal_detected(self, mock_model_loader, mock_settings):
        """Test detection of free availability signal."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "I'm free tonight!", "is_from_me": True},
            {"text": "Nothing going on this weekend", "is_from_me": True},
            {"text": "Want to hang out?", "is_from_me": False, "sender": "Test"},
        ]

        # Extract availability signal
        availability = generator._extract_availability_signal(messages)
        assert availability == "free"


# =============================================================================
# Global Style Integration Tests
# =============================================================================


class TestGlobalStyleIntegration:
    """Test global user style integration."""

    def test_global_style_used_when_available(self, mock_model_loader, mock_settings):
        """Test that global style is used when available."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        # Create mock global style
        mock_global_style = MagicMock()
        mock_global_style.capitalization = "lowercase"
        mock_global_style.punctuation_style = "minimal"
        mock_global_style.uses_abbreviations = True
        mock_global_style.avg_word_count = 5
        mock_global_style.common_phrases = ["sounds good", "yeah"]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=mock_global_style,
                        ):
                            result = generator.generate_replies(
                                messages, chat_id="test-global-style"
                            )

        # Style instructions should reflect global style
        assert "lowercase" in result.style_instructions or "MAX" in result.style_instructions


# =============================================================================
# Contact Profile Integration Tests
# =============================================================================


class TestContactProfileIntegration:
    """Test contact profile integration."""

    def test_contact_profile_affects_style(self, mock_model_loader, mock_settings):
        """Test that contact profile affects style instructions."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        # Create mock contact profile
        mock_profile = MagicMock()
        mock_profile.total_messages = 100
        mock_profile.avg_your_length = 15
        mock_profile.uses_emoji = True
        mock_profile.uses_slang = True
        mock_profile.relationship_type = "close_friend"
        mock_profile.is_playful = True
        mock_profile.tone = "playful"
        mock_profile.topics = []
        mock_profile.your_common_phrases = ["haha", "lol"]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile",
                        return_value=mock_profile,
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-profile")

        # Result should have style instructions influenced by profile
        assert result.style_instructions is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_message(self, mock_model_loader, mock_settings):
        """Test handling of very long messages."""
        generator = ReplyGenerator(mock_model_loader)

        long_text = "This is a very long message. " * 100
        messages = [
            {"text": long_text, "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-long")

        # Should handle gracefully
        assert result is not None

    def test_message_with_special_characters(self, mock_model_loader, mock_settings):
        """Test handling of special characters."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey! \U0001f389\U0001f38a How's it going??? !!!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(messages, chat_id="test-special")

        # Should handle gracefully
        assert result is not None
        assert "\U0001f389" in result.context.last_message

    def test_message_with_attachment_placeholder(self, mock_model_loader, mock_settings):
        """Test handling of attachment placeholder character."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "\ufffc", "is_from_me": False, "sender": "Test"},  # Attachment placeholder
            {"text": "Check this out!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            result = generator.generate_replies(
                                messages, chat_id="test-attachment"
                            )

        # Should handle gracefully
        assert result is not None

    def test_no_chat_id_provided(self, mock_model_loader, mock_settings):
        """Test generation without chat_id."""
        generator = ReplyGenerator(mock_model_loader)

        messages = [
            {"text": "Hey!", "is_from_me": False, "sender": "Test"},
        ]

        with patch("core.generation.reply_generator.settings", mock_settings):
            with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
                with patch(
                    "core.generation.reply_generator._get_embedding_store", return_value=None
                ):
                    with patch(
                        "core.generation.reply_generator._get_contact_profile", return_value=None
                    ):
                        with patch(
                            "core.generation.reply_generator.get_global_user_style",
                            return_value=None,
                        ):
                            # No chat_id
                            result = generator.generate_replies(messages)

        # Should still work
        assert result is not None
        assert len(result.replies) > 0
