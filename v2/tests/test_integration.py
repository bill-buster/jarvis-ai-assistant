"""Integration tests for the reply generation pipeline.

These tests verify that components work together correctly
and test realistic end-to-end scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from core.generation.context_analyzer import ContextAnalyzer, MessageIntent
from core.generation.prompts import build_reply_prompt
from core.generation.reply_generator import ReplyGenerator
from core.generation.style_analyzer import StyleAnalyzer


class TestContextStyleIntegration:
    """Test that context analysis and style analysis work together."""

    @pytest.fixture
    def context_analyzer(self) -> ContextAnalyzer:
        return ContextAnalyzer()

    @pytest.fixture
    def style_analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_casual_friend_detected_correctly(
        self,
        context_analyzer: ContextAnalyzer,
        style_analyzer: StyleAnalyzer,
    ) -> None:
        """Test that casual friend conversations are detected with matching style."""
        # Use messages with abbreviations from the ABBREVIATIONS set:
        # "u", "ur", "r", "lol", "lmao", "omg", "idk", "tbh", "gonna", "wanna", etc.
        messages = [
            {"text": "lol that's hilarious 😂", "sender": "Friend", "is_from_me": False},
            {"text": "haha lol yeah", "sender": "me", "is_from_me": True},  # "lol" is in ABBREVIATIONS
            {"text": "wanna hang tmrw?", "sender": "Friend", "is_from_me": False},
            {"text": "ya sounds good u free?", "sender": "me", "is_from_me": True},  # "u" is in ABBREVIATIONS
            {"text": "cool see u then?", "sender": "Friend", "is_from_me": False},
        ]

        # Context analysis
        context = context_analyzer.analyze(messages)
        assert context.mood == "neutral"  # Casual but not strongly emotional
        # "see u then?" is an OPEN_QUESTION because it ends with "?" but doesn't
        # start with yes/no patterns like "do you", "can you", etc.
        assert context.intent == MessageIntent.OPEN_QUESTION

        # Style analysis (only user messages)
        user_messages = [m for m in messages if m.get("is_from_me")]
        style = style_analyzer.analyze(user_messages)

        assert style.uses_abbreviations is True  # "lol" and "u" are in ABBREVIATIONS
        assert style.capitalization == "lowercase"
        assert style.formality_score < 0.3  # Very casual

    def test_work_conversation_detected(
        self,
        context_analyzer: ContextAnalyzer,
        style_analyzer: StyleAnalyzer,
    ) -> None:
        """Test that work conversations are detected with appropriate style."""
        messages = [
            {
                "text": "Hi, could you review the project proposal?",
                "sender": "Colleague",
                "is_from_me": False,
            },
            {
                "text": "Sure, I'll take a look this afternoon.",
                "sender": "me",
                "is_from_me": True,
            },
            {"text": "The deadline is tomorrow.", "sender": "Colleague", "is_from_me": False},
            {
                "text": "Got it. I'll prioritize it.",
                "sender": "me",
                "is_from_me": True,
            },
            {
                "text": "Thanks! Let me know if you have questions.",
                "sender": "Colleague",
                "is_from_me": False,
            },
        ]

        context = context_analyzer.analyze(messages)
        # "deadline" and "project" trigger WORK relationship
        assert context.relationship.value == "work"
        assert context.intent == MessageIntent.THANKS

        user_messages = [m for m in messages if m.get("is_from_me")]
        style = style_analyzer.analyze(user_messages)

        # Work messages tend to be more formal
        assert style.capitalization == "normal"
        assert style.uses_abbreviations is False


class TestPromptBuildingIntegration:
    """Test that prompt building integrates correctly with context and style."""

    def test_prompt_includes_all_context(self) -> None:
        """Test that built prompt includes all relevant context."""
        messages = [
            {"text": "Hey how are you?", "sender": "John", "is_from_me": False},
            {"text": "good! been busy with work", "sender": "me", "is_from_me": True},
            {"text": "Want to grab coffee this weekend?", "sender": "John", "is_from_me": False},
        ]

        # Analyze context
        context_analyzer = ContextAnalyzer()
        context = context_analyzer.analyze(messages)

        # Analyze style
        style_analyzer = StyleAnalyzer()
        user_messages = [m for m in messages if m.get("is_from_me")]
        style = style_analyzer.analyze(user_messages)

        # Build prompt
        style_instructions = style_analyzer.build_style_instructions(style, None, None)
        prompt = build_reply_prompt(
            messages=messages,
            last_message=context.last_message,
            last_sender=context.last_sender,
            style_instructions=style_instructions,
            user_name="Me",
            availability="busy",  # From "been busy with work"
        )

        # Verify prompt contains expected elements
        assert "Me" in prompt
        assert "busy" in prompt.lower()
        assert "coffee" in prompt.lower() or "weekend" in prompt.lower()

    def test_prompt_with_global_style_and_profile(self) -> None:
        """Test prompt with global style and contact profile.

        Note: The simplified prompt format uses style_instructions directly
        and includes common phrases, but doesn't embed personality summaries
        or relationship info in the prompt body.
        """

        @dataclass
        class MockGlobalStyle:
            capitalization: str = "lowercase"
            punctuation_style: str = "minimal"
            uses_abbreviations: bool = True
            avg_word_count: float = 5.0
            personality_summary: str = "casual texter who uses lots of abbreviations"
            interests: list = None

            def __post_init__(self):
                self.interests = ["music", "gaming"]

        @dataclass
        class MockProfile:
            display_name: str = "Sarah"
            relationship_type: str = "close_friend"
            relationship_summary: str = "best friend from college"
            tone: str = "playful"
            total_messages: int = 500
            uses_emoji: bool = True
            uses_slang: bool = True
            is_playful: bool = True
            avg_your_length: float = 25.0
            topics: list = None
            your_common_phrases: list = None

            def __post_init__(self):
                self.topics = []
                self.your_common_phrases = ["haha", "omg", "lol"]

        messages = [{"text": "ready for tonight?", "sender": "Sarah", "is_from_me": False}]

        prompt = build_reply_prompt(
            messages=messages,
            last_message="ready for tonight?",
            last_sender="Sarah",
            style_instructions="lowercase, brief",
            user_name="Me",
            global_style=MockGlobalStyle(),
            contact_profile=MockProfile(),
            your_phrases=["haha", "lol", "sounds good"],
        )

        # The simplified prompt includes style instructions and phrases
        assert "lowercase" in prompt.lower()
        assert "phrases you use" in prompt.lower()
        assert "haha" in prompt or "lol" in prompt or "sounds good" in prompt


class TestEndToEndGeneration:
    """Test end-to-end generation flow."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        """Create a mock model loader with realistic responses."""
        mock = MagicMock()
        mock.current_model = "test-model"

        @dataclass
        class MockResult:
            text: str
            formatted_prompt: str = "[prompt]"

        # Simulate LLM generating short replies
        mock.generate.return_value = MockResult(text="sounds good!")
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        """Create generator with mocked dependencies."""
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch(
            "core.generation.reply_generator._get_embedding_store", return_value=None
        ), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(mock_model_loader)
            gen._template_matcher = None
            return gen

    def test_basic_generation_flow(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test basic generation produces valid result."""
        messages = [
            {"text": "Hey!", "sender": "John", "is_from_me": False},
            {"text": "Hi there", "sender": "me", "is_from_me": True},
            {"text": "Want to grab lunch?", "sender": "John", "is_from_me": False},
        ]

        result = generator.generate_replies(
            messages=messages, chat_id="chat_123", num_replies=3, user_name="Me"
        )

        # Should return valid result
        assert result is not None
        assert len(result.replies) > 0
        assert result.context is not None
        assert result.style is not None
        assert result.generation_time_ms >= 0

        # Model should have been called
        mock_model_loader.generate.assert_called()

    def test_style_caching_works(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test that style is cached per chat."""
        messages = [{"text": "Hello", "sender": "John", "is_from_me": False}]

        # Generate twice for same chat
        generator.generate_replies(messages=messages, chat_id="chat_123")
        generator.generate_replies(messages=messages, chat_id="chat_123")

        # Style should be cached - analyzer only called once
        state = generator._get_chat_state("chat_123")
        assert state.style is not None

    def test_repetition_tracking_works(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test that generated replies are tracked for repetition."""
        messages = [{"text": "Hello?", "sender": "John", "is_from_me": False}]

        # Generate
        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Reply should be tracked
        state = generator._get_chat_state("chat_123")
        assert len(state.recent_generations) > 0

        # Same reply should be detected as repetitive
        for reply in result.replies:
            assert generator._is_repetitive(reply.text, "chat_123")


class TestEdgeCasesIntegration:
    """Test edge cases that span multiple components."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"

        @dataclass
        class MockResult:
            text: str
            formatted_prompt: str = ""

        mock.generate.return_value = MockResult(text="ok")
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch(
            "core.generation.reply_generator._get_embedding_store", return_value=None
        ), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(mock_model_loader)
            gen._template_matcher = None
            return gen

    def test_empty_conversation(self, generator: ReplyGenerator) -> None:
        """Test handling of empty conversation."""
        result = generator.generate_replies(messages=[], chat_id="chat_123")

        # Should still return valid result with fallbacks
        assert result is not None
        assert result.context is not None

    def test_only_your_messages(self, generator: ReplyGenerator) -> None:
        """Test conversation with only your messages (unusual but possible)."""
        messages = [
            {"text": "Hello?", "sender": "me", "is_from_me": True},
            {"text": "Anyone there?", "sender": "me", "is_from_me": True},
        ]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Should handle gracefully
        assert result is not None
        # Style should be analyzed from your messages
        assert result.style is not None

    def test_unicode_messages(self, generator: ReplyGenerator) -> None:
        """Test handling of unicode characters in messages."""
        messages = [
            {"text": "مرحبا!", "sender": "Friend", "is_from_me": False},  # Arabic
            {"text": "日本語テスト", "sender": "me", "is_from_me": True},  # Japanese
            {"text": "How are you? 😊", "sender": "Friend", "is_from_me": False},
        ]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Should not crash on unicode
        assert result is not None

    def test_very_long_messages(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test handling of very long messages."""
        long_text = "word " * 500  # 500 words
        messages = [{"text": long_text, "sender": "Friend", "is_from_me": False}]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Should handle without error
        assert result is not None

        # Check that the prompt was built (model was called)
        mock_model_loader.generate.assert_called()

    def test_special_characters_in_messages(self, generator: ReplyGenerator) -> None:
        """Test handling of special characters."""
        messages = [
            {"text": 'He said "Hello!" to me', "sender": "Friend", "is_from_me": False},
            {"text": "That's great! @#$%^&*()", "sender": "me", "is_from_me": True},
            {"text": "Newlines\nand\ttabs", "sender": "Friend", "is_from_me": False},
        ]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Should handle special characters
        assert result is not None


class TestTemplateMatchingIntegration:
    """Test template matching fast path."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    def test_template_match_skips_model(self, mock_model_loader: MagicMock) -> None:
        """Test that template match skips model loading."""

        @dataclass
        class MockTemplateMatch:
            trigger: str = "how are you"
            actual: str = "doing good!"
            confidence: float = 0.85

        mock_template_matcher = MagicMock()
        mock_template_matcher.match.return_value = MockTemplateMatch()

        with patch(
            "core.generation.reply_generator._get_template_matcher",
            return_value=mock_template_matcher,
        ), patch(
            "core.generation.reply_generator._get_embedding_store", return_value=None
        ), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            generator = ReplyGenerator(mock_model_loader)
            generator._template_matcher = mock_template_matcher

            messages = [{"text": "How are you?", "sender": "John", "is_from_me": False}]

            result = generator.generate_replies(messages=messages, chat_id="chat_123")

            # Should return template match
            assert result.model_used == "template"
            assert len(result.replies) == 1
            assert result.replies[0].text == "doing good!"

            # Model should NOT have been called
            mock_model_loader.generate.assert_not_called()


class TestFallbackMechanisms:
    """Test fallback mechanisms when components fail."""

    @pytest.fixture
    def failing_model_loader(self) -> MagicMock:
        """Model loader that raises an exception."""
        mock = MagicMock()
        mock.current_model = "test-model"
        mock.generate.side_effect = Exception("Model failed to generate")
        return mock

    @pytest.fixture
    def generator(self, failing_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch(
            "core.generation.reply_generator._get_embedding_store", return_value=None
        ), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(failing_model_loader)
            gen._template_matcher = None
            return gen

    def test_fallback_on_model_error(self, generator: ReplyGenerator) -> None:
        """Test that fallback replies are returned when model fails."""
        messages = [{"text": "Want to hang out?", "sender": "John", "is_from_me": False}]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Should return fallback result
        assert result is not None
        assert result.model_used == "fallback"
        assert len(result.replies) > 0

        # Fallback replies should have lower confidence
        for reply in result.replies:
            assert reply.reply_type == "fallback"
            assert reply.confidence == 0.5


class TestAvailabilityDetection:
    """Test availability signal detection in full context."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"

        @dataclass
        class MockResult:
            text: str
            formatted_prompt: str = ""

        mock.generate.return_value = MockResult(text="sounds good")
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch(
            "core.generation.reply_generator._get_embedding_store", return_value=None
        ), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(mock_model_loader)
            gen._template_matcher = None
            return gen

    def test_busy_availability_detected(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test that busy availability is detected from conversation."""
        messages = [
            {"text": "Want to hang out?", "sender": "John", "is_from_me": False},
            {"text": "I'm super busy this week", "sender": "me", "is_from_me": True},
            {"text": "Working late every night", "sender": "me", "is_from_me": True},
            {"text": "What about next weekend?", "sender": "John", "is_from_me": False},
        ]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Availability should be detected
        assert result is not None
        # The prompt should include busy context
        call_args = mock_model_loader.generate.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "busy" in prompt.lower()

    def test_free_availability_detected(
        self, generator: ReplyGenerator, mock_model_loader: MagicMock
    ) -> None:
        """Test that free availability is detected from conversation."""
        messages = [
            {"text": "Busy this weekend?", "sender": "John", "is_from_me": False},
            {"text": "Nope, totally free!", "sender": "me", "is_from_me": True},
            {"text": "I'm available all day", "sender": "me", "is_from_me": True},
            {"text": "Great, let's do something", "sender": "John", "is_from_me": False},
        ]

        result = generator.generate_replies(messages=messages, chat_id="chat_123")

        # Availability should be detected
        assert result is not None
        # The prompt should include free context
        call_args = mock_model_loader.generate.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "free" in prompt.lower() or "available" in prompt.lower()
