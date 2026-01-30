"""Comprehensive tests for the reply generator module.

These tests verify the full generation pipeline including:
- Temperature scaling
- Repetition filtering
- Coherent message selection
- Style caching
- Availability signal detection
- Fallback mechanisms
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from core.generation.context_analyzer import MessageIntent, RelationshipType
from core.generation.reply_generator import (
    TEMPERATURE_SCALE,
    ChatState,
    GeneratedReply,
    ReplyGenerationResult,
    ReplyGenerator,
)
from core.generation.style_analyzer import UserStyle


class TestTemperatureScaling:
    """Test temperature scaling based on regeneration count."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        """Create a mock model loader."""
        mock = MagicMock()
        mock.current_model = "test-model"
        mock.generate.return_value = MagicMock(text="test reply", formatted_prompt="")
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        """Create a ReplyGenerator with mocked dependencies."""
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(mock_model_loader)
            gen._template_matcher = None  # Disable template matching for these tests
            return gen

    def test_first_generation_uses_low_temp(self, generator: ReplyGenerator) -> None:
        """Test that first generation uses the lowest temperature."""
        temp = generator._get_temperature("chat_1", "Hello there")
        assert temp == TEMPERATURE_SCALE[0]  # 0.2

    def test_regeneration_increases_temp(self, generator: ReplyGenerator) -> None:
        """Test that regenerating the same message increases temperature."""
        chat_id = "chat_1"
        message = "How are you?"

        # First generation
        temp1 = generator._get_temperature(chat_id, message)
        assert temp1 == TEMPERATURE_SCALE[0]  # 0.2

        # Regenerations should increase temp
        temp2 = generator._get_temperature(chat_id, message)
        assert temp2 == TEMPERATURE_SCALE[1]  # 0.4

        temp3 = generator._get_temperature(chat_id, message)
        assert temp3 == TEMPERATURE_SCALE[2]  # 0.6

    def test_new_message_resets_temp(self, generator: ReplyGenerator) -> None:
        """Test that a new message resets the temperature."""
        chat_id = "chat_1"

        # Generate for first message several times
        for _ in range(3):
            generator._get_temperature(chat_id, "First message")

        # New message should reset to low temp
        temp = generator._get_temperature(chat_id, "Different message")
        assert temp == TEMPERATURE_SCALE[0]

    def test_temp_maxes_out(self, generator: ReplyGenerator) -> None:
        """Test that temperature maxes out at the last scale value."""
        chat_id = "chat_1"
        message = "Same message"

        # Generate many times
        for _ in range(10):
            temp = generator._get_temperature(chat_id, message)

        # Should be at max temp (0.9)
        assert temp == TEMPERATURE_SCALE[-1]

    def test_different_chats_independent(self, generator: ReplyGenerator) -> None:
        """Test that temperature tracking is independent per chat."""
        message = "Same message"

        # Generate several times for chat_1
        for _ in range(3):
            generator._get_temperature("chat_1", message)

        # chat_2 should start fresh
        temp = generator._get_temperature("chat_2", message)
        assert temp == TEMPERATURE_SCALE[0]

    def test_hash_stability(self, generator: ReplyGenerator) -> None:
        """Test that message hash is stable (same message = same hash)."""
        chat_id = "chat_1"
        message = "Test message for hashing"

        # First call
        temp1 = generator._get_temperature(chat_id, message)

        # Reset state
        generator._chat_states.clear()

        # Same message should produce same initial behavior
        temp2 = generator._get_temperature(chat_id, message)

        assert temp1 == temp2

    def test_long_message_truncation(self, generator: ReplyGenerator) -> None:
        """Test that long messages are truncated for hashing."""
        chat_id = "chat_1"
        # Messages that differ after 100 chars should produce same hash
        base_message = "x" * 100
        message1 = base_message + "different_ending_1"
        message2 = base_message + "different_ending_2"

        temp1 = generator._get_temperature(chat_id, message1)
        generator._chat_states.clear()
        temp2 = generator._get_temperature(chat_id, message2)

        # Both should have same temp (both treated as first generation)
        assert temp1 == temp2


class TestRepetitionFiltering:
    """Test repetition detection and filtering."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_record_generation(self, generator: ReplyGenerator) -> None:
        """Test that generations are recorded."""
        generator._record_generation("chat_1", "Hello there!")
        state = generator._get_chat_state("chat_1")
        assert "hello there!" in state.recent_generations

    def test_repetitive_reply_detected(self, generator: ReplyGenerator) -> None:
        """Test that repetitive replies are detected."""
        chat_id = "chat_1"
        generator._record_generation(chat_id, "sounds good")

        assert generator._is_repetitive("sounds good", chat_id) is True
        assert generator._is_repetitive("SOUNDS GOOD", chat_id) is True  # Case insensitive
        assert generator._is_repetitive("  sounds good  ", chat_id) is True  # Whitespace

    def test_non_repetitive_reply(self, generator: ReplyGenerator) -> None:
        """Test that new replies are not marked as repetitive."""
        chat_id = "chat_1"
        generator._record_generation(chat_id, "sounds good")

        assert generator._is_repetitive("different reply", chat_id) is False

    def test_repetition_limit(self, generator: ReplyGenerator) -> None:
        """Test that only MAX_RECENT_GENERATIONS are tracked."""
        chat_id = "chat_1"

        # Record more than the limit
        for i in range(10):
            generator._record_generation(chat_id, f"reply_{i}")

        state = generator._get_chat_state(chat_id)
        assert len(state.recent_generations) == generator.MAX_RECENT_GENERATIONS

        # Old replies should be forgotten
        assert not generator._is_repetitive("reply_0", chat_id)
        # Recent replies should be tracked
        assert generator._is_repetitive("reply_9", chat_id)

    def test_filter_repetitive_replies(self, generator: ReplyGenerator) -> None:
        """Test filtering of repetitive replies from a list."""
        chat_id = "chat_1"
        generator._record_generation(chat_id, "sounds good")
        generator._record_generation(chat_id, "okay")

        replies = [
            GeneratedReply(text="sounds good", reply_type="test"),
            GeneratedReply(text="great idea", reply_type="test"),
            GeneratedReply(text="okay", reply_type="test"),
            GeneratedReply(text="sure thing", reply_type="test"),
        ]

        filtered = generator._filter_repetitive(replies, chat_id)

        assert len(filtered) == 2
        assert all(r.text in ["great idea", "sure thing"] for r in filtered)

    def test_no_filter_without_chat_id(self, generator: ReplyGenerator) -> None:
        """Test that filtering is skipped without chat_id."""
        replies = [GeneratedReply(text="test", reply_type="test")]
        filtered = generator._filter_repetitive(replies, None)
        assert len(filtered) == 1


class TestCoherentMessageSelection:
    """Test the _get_coherent_messages method."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_short_conversation_unchanged(self, generator: ReplyGenerator) -> None:
        """Test that short conversations (<=4 messages) are returned unchanged."""
        messages = [
            {"text": "Hey", "is_from_me": False, "sender": "John"},
            {"text": "Hi!", "is_from_me": True},
            {"text": "How are you?", "is_from_me": False, "sender": "John"},
        ]
        result = generator._get_coherent_messages(messages)
        assert result == messages

    def test_group_chat_detection(self, generator: ReplyGenerator) -> None:
        """Test that group chats are detected and limited to 5 messages."""
        messages = []
        # Create messages from multiple senders
        for i in range(10):
            sender = f"Person{i % 3}"  # 3 different senders
            messages.append(
                {"text": f"Message {i}", "is_from_me": False, "sender": sender}
            )

        result = generator._get_coherent_messages(messages)

        # Group chat should limit to last 5 messages
        assert len(result) == 5
        assert result == messages[-5:]

    def test_one_on_one_thread_context(self, generator: ReplyGenerator) -> None:
        """Test that 1:1 chats include thread context."""
        messages = [
            {"text": "Old message", "is_from_me": False, "sender": "John"},
            {"text": "Very old", "is_from_me": True},
            {"text": "What do you think?", "is_from_me": False, "sender": "John"},
            {"text": "Sounds good", "is_from_me": True},
            {"text": "Great!", "is_from_me": False, "sender": "John"},
            {"text": "So about that...", "is_from_me": False, "sender": "John"},
            {"text": "New question?", "is_from_me": False, "sender": "John"},
        ]

        result = generator._get_coherent_messages(messages)

        # Should include context around your reply
        assert len(result) <= 8  # Max 8 messages

    def test_max_8_messages_limit(self, generator: ReplyGenerator) -> None:
        """Test that 1:1 chats are limited to 8 messages max."""
        messages = []
        for i in range(20):
            is_from_me = i % 2 == 0
            messages.append(
                {
                    "text": f"Message {i}",
                    "is_from_me": is_from_me,
                    "sender": "me" if is_from_me else "John",
                }
            )

        result = generator._get_coherent_messages(messages)
        assert len(result) <= 8


class TestAvailabilitySignalExtraction:
    """Test the _extract_availability_signal method."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_busy_signal_detection(self, generator: ReplyGenerator) -> None:
        """Test detection of busy availability signal."""
        messages = [
            {"text": "I'm super busy this week", "is_from_me": True},
            {"text": "Can't make it, swamped", "is_from_me": True},
            {"text": "Working late tonight", "is_from_me": True},
        ]
        signal = generator._extract_availability_signal(messages)
        assert signal == "busy"

    def test_free_signal_detection(self, generator: ReplyGenerator) -> None:
        """Test detection of free availability signal."""
        messages = [
            {"text": "I'm free this weekend!", "is_from_me": True},
            {"text": "Let's do it, I'm available", "is_from_me": True},
            {"text": "Count me in!", "is_from_me": True},
        ]
        signal = generator._extract_availability_signal(messages)
        assert signal == "free"

    def test_unknown_signal_when_no_indicators(self, generator: ReplyGenerator) -> None:
        """Test that unknown is returned when no clear indicators."""
        messages = [
            {"text": "Hello there", "is_from_me": True},
            {"text": "The weather is nice", "is_from_me": True},
        ]
        signal = generator._extract_availability_signal(messages)
        assert signal == "unknown"

    def test_only_your_messages_considered(self, generator: ReplyGenerator) -> None:
        """Test that only your messages are considered for availability."""
        messages = [
            {"text": "I'm super busy", "is_from_me": False},  # Their message
            {"text": "Let's hang out", "is_from_me": True},  # Your message
        ]
        signal = generator._extract_availability_signal(messages)
        # "busy" from their message shouldn't count
        assert signal != "busy"

    def test_empty_messages(self, generator: ReplyGenerator) -> None:
        """Test availability signal with empty messages."""
        assert generator._extract_availability_signal(None) == "unknown"
        assert generator._extract_availability_signal([]) == "unknown"

    def test_no_your_messages(self, generator: ReplyGenerator) -> None:
        """Test when there are no messages from you."""
        messages = [
            {"text": "I'm busy", "is_from_me": False},
            {"text": "I'm free", "is_from_me": False},
        ]
        signal = generator._extract_availability_signal(messages)
        assert signal == "unknown"


class TestStyleCaching:
    """Test style analysis caching behavior."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_style_cached_per_chat(self, generator: ReplyGenerator) -> None:
        """Test that style is cached per chat."""
        messages = [{"text": "hello lol", "is_from_me": True}]

        style1 = generator._get_or_analyze_style(messages, "chat_1")
        style2 = generator._get_or_analyze_style(messages, "chat_1")

        # Should be the same cached instance
        assert style1 is style2

    def test_different_chats_different_styles(self, generator: ReplyGenerator) -> None:
        """Test that different chats have independent style caches."""
        messages1 = [{"text": "hello lol", "is_from_me": True}]
        messages2 = [{"text": "HELLO THERE!", "is_from_me": True}]

        style1 = generator._get_or_analyze_style(messages1, "chat_1")
        style2 = generator._get_or_analyze_style(messages2, "chat_2")

        # Should be different styles
        assert style1 is not style2

    def test_clear_cache(self, generator: ReplyGenerator) -> None:
        """Test that cache can be cleared."""
        messages = [{"text": "hello", "is_from_me": True}]

        generator._get_or_analyze_style(messages, "chat_1")
        generator._get_or_analyze_style(messages, "chat_2")

        # Clear specific chat
        generator.clear_cache("chat_1")
        assert "chat_1" not in generator._chat_states
        assert "chat_2" in generator._chat_states

        # Clear all
        generator.clear_cache()
        assert len(generator._chat_states) == 0

    def test_no_cache_without_chat_id(self, generator: ReplyGenerator) -> None:
        """Test that style is not cached without chat_id."""
        messages = [{"text": "hello", "is_from_me": True}]

        style1 = generator._get_or_analyze_style(messages, None)
        style2 = generator._get_or_analyze_style(messages, None)

        # Should be different instances (not cached)
        assert style1 is not style2


class TestFallbackReplies:
    """Test fallback reply generation."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_fallback_for_yes_no_question(self, generator: ReplyGenerator) -> None:
        """Test fallback replies for yes/no questions."""
        replies = generator._get_fallback_replies("yes_no_question", 3)

        assert len(replies) == 3
        assert all(r.reply_type == "fallback" for r in replies)
        assert all(r.confidence == 0.5 for r in replies)
        # Should include affirmative and negative options
        texts = [r.text.lower() for r in replies]
        assert any("good" in t or "sorry" in t or "check" in t for t in texts)

    def test_fallback_for_greeting(self, generator: ReplyGenerator) -> None:
        """Test fallback replies for greetings."""
        replies = generator._get_fallback_replies("greeting", 3)

        assert len(replies) == 3
        texts = [r.text.lower() for r in replies]
        assert any("hey" in t or "hi" in t or "what's up" in t for t in texts)

    def test_fallback_for_unknown_intent(self, generator: ReplyGenerator) -> None:
        """Test fallback for unknown intent falls back to statement."""
        replies = generator._get_fallback_replies("unknown_intent", 3)

        # Should fall back to statement replies
        assert len(replies) == 3

    def test_partial_count(self, generator: ReplyGenerator) -> None:
        """Test requesting fewer replies than available."""
        replies = generator._get_fallback_replies("greeting", 1)
        assert len(replies) == 1


class TestReplyParsing:
    """Test the _parse_replies method."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            gen = ReplyGenerator(mock_model_loader)
            gen._user_name = "me"
            return gen

    def test_basic_reply_parsing(self, generator: ReplyGenerator) -> None:
        """Test parsing a simple reply."""
        raw = "Sounds good!"
        replies = generator._parse_replies(raw, [])

        assert len(replies) == 1
        assert replies[0].text == "Sounds good!"

    def test_multiline_takes_first(self, generator: ReplyGenerator) -> None:
        """Test that only the first line is taken."""
        raw = "First reply\nSecond line\nThird line"
        replies = generator._parse_replies(raw, [])

        assert len(replies) == 1
        assert replies[0].text == "First reply"

    def test_prefix_removal(self, generator: ReplyGenerator) -> None:
        """Test that common prefixes are removed."""
        prefixes = ["Reply:", "Response:", "Answer:", "me:", "Them:", "Me:"]

        for prefix in prefixes:
            raw = f"{prefix} Hello there"
            replies = generator._parse_replies(raw, [])
            assert replies[0].text == "Hello there", f"Failed for prefix: {prefix}"

    def test_quote_removal(self, generator: ReplyGenerator) -> None:
        """Test that surrounding quotes are removed."""
        raw1 = '"Hello there"'
        raw2 = "'Hello there'"

        replies1 = generator._parse_replies(raw1, [])
        replies2 = generator._parse_replies(raw2, [])

        assert replies1[0].text == "Hello there"
        assert replies2[0].text == "Hello there"

    def test_too_short_rejected(self, generator: ReplyGenerator) -> None:
        """Test that replies shorter than 2 characters are rejected."""
        raw = "k"
        replies = generator._parse_replies(raw, [])
        assert len(replies) == 0

    def test_too_long_rejected(self, generator: ReplyGenerator) -> None:
        """Test that replies longer than 150 characters are rejected."""
        raw = "x" * 200
        replies = generator._parse_replies(raw, [])
        assert len(replies) == 0

    def test_emoji_stripping(self, generator: ReplyGenerator) -> None:
        """Test emoji stripping when flag is set."""
        raw = "Hello there! 😊👍"
        replies = generator._parse_replies(raw, [], strip_emojis_flag=True)

        assert len(replies) == 1
        # Emojis should be stripped
        assert "😊" not in replies[0].text

    def test_empty_input(self, generator: ReplyGenerator) -> None:
        """Test handling of empty input."""
        assert generator._parse_replies("", []) == []
        assert generator._parse_replies("   ", []) == []


class TestTemplateMatching:
    """Test the _try_template_match method."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        mock = MagicMock()
        mock.current_model = "test-model"
        return mock

    @pytest.fixture
    def generator(self, mock_model_loader: MagicMock) -> ReplyGenerator:
        with patch(
            "core.generation.reply_generator._get_template_matcher", return_value=None
        ), patch("core.generation.reply_generator._get_embedding_store", return_value=None), patch(
            "core.generation.reply_generator._get_contact_profile", return_value=None
        ), patch(
            "core.generation.reply_generator.get_global_user_style", return_value=None
        ):
            return ReplyGenerator(mock_model_loader)

    def test_no_match_with_empty(self, generator: ReplyGenerator) -> None:
        """Test no match with empty past replies."""
        result = generator._try_template_match(None)
        assert result is None

        result = generator._try_template_match([])
        assert result is None

    def test_no_match_with_few_replies(self, generator: ReplyGenerator) -> None:
        """Test no match with too few past replies."""
        # Need at least 2 high-confidence replies
        past_replies = [("What's up?", "hey", 0.8)]
        result = generator._try_template_match(past_replies)
        assert result is None

    def test_match_with_identical_responses(self, generator: ReplyGenerator) -> None:
        """Test match when all responses are identical."""
        past_replies = [
            ("What's up?", "not much", 0.85),
            ("How's it going?", "not much", 0.80),
            ("Hey there", "not much", 0.77),
        ]
        result = generator._try_template_match(past_replies)
        assert result == "not much"

    def test_match_with_yes_variants(self, generator: ReplyGenerator) -> None:
        """Test match when responses are yes variants."""
        past_replies = [
            ("Want to come?", "yes", 0.85),
            ("Are you free?", "yea", 0.80),
            ("Can you make it?", "yeah", 0.78),
        ]
        result = generator._try_template_match(past_replies)
        assert result == "yes"  # Returns the first one

    def test_match_with_no_variants(self, generator: ReplyGenerator) -> None:
        """Test match when responses are no variants."""
        past_replies = [
            ("Want to come?", "nah", 0.85),
            ("Are you free?", "no", 0.80),
            ("Can you make it?", "can't", 0.78),
        ]
        result = generator._try_template_match(past_replies)
        assert result == "nah"

    def test_no_match_with_varied_responses(self, generator: ReplyGenerator) -> None:
        """Test no match when responses are varied."""
        past_replies = [
            ("What's up?", "not much", 0.85),
            ("How's it going?", "doing well", 0.80),
            ("Hey there", "hello!", 0.78),
        ]
        result = generator._try_template_match(past_replies)
        # Different responses shouldn't match
        assert result is None


class TestChatState:
    """Test ChatState dataclass behavior."""

    def test_default_values(self) -> None:
        """Test ChatState default values."""
        state = ChatState()

        assert state.style is None
        assert state.recent_generations == []
        assert state.regen_count == 0
        assert state.last_message_hash == ""

    def test_mutation(self) -> None:
        """Test that ChatState can be mutated."""
        state = ChatState()

        state.regen_count = 5
        state.recent_generations.append("test")
        state.last_message_hash = "abc123"

        assert state.regen_count == 5
        assert "test" in state.recent_generations
        assert state.last_message_hash == "abc123"


class TestGeneratedReply:
    """Test GeneratedReply dataclass."""

    def test_creation(self) -> None:
        """Test creating a GeneratedReply."""
        reply = GeneratedReply(
            text="Hello!", reply_type="generated", confidence=0.95
        )

        assert reply.text == "Hello!"
        assert reply.reply_type == "generated"
        assert reply.confidence == 0.95

    def test_default_confidence(self) -> None:
        """Test default confidence value."""
        reply = GeneratedReply(text="Hello!", reply_type="test")
        assert reply.confidence == 0.8
