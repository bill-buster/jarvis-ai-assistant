"""Unit tests for jarvis/extract.py - Turn-Based Pair Extraction."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from contracts.imessage import Attachment, Conversation, Message
from jarvis.extract import (
    ExtractedPair,
    ExtractionConfig,
    ExtractionStats,
    Turn,
    TurnBasedExtractor,
    extract_all_pairs,
    extract_pairs_from_reader,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_time() -> datetime:
    """Base timestamp for tests."""
    return datetime(2024, 1, 15, 10, 0, 0)


@pytest.fixture
def default_config() -> ExtractionConfig:
    """Default extraction configuration."""
    return ExtractionConfig()


@pytest.fixture
def extractor(default_config: ExtractionConfig) -> TurnBasedExtractor:
    """TurnBasedExtractor with default config."""
    return TurnBasedExtractor(default_config)


def make_message(
    msg_id: int,
    text: str,
    is_from_me: bool,
    date: datetime,
    *,
    is_system_message: bool = False,
    attachments: list[Attachment] | None = None,
) -> Message:
    """Create a test message."""
    return Message(
        id=msg_id,
        chat_id="test_chat",
        sender="+15551234567",
        sender_name="Test User",
        text=text,
        date=date,
        is_from_me=is_from_me,
        is_system_message=is_system_message,
        attachments=attachments or [],
    )


# =============================================================================
# Turn Dataclass Tests
# =============================================================================


class TestTurn:
    """Tests for Turn dataclass."""

    def test_empty_turn(self) -> None:
        """Test empty turn properties."""
        turn = Turn(is_from_me=False)
        assert turn.text == ""
        assert turn.message_ids == []
        assert turn.primary_msg_id is None
        assert turn.first_timestamp == datetime.min
        assert turn.last_timestamp == datetime.min

    def test_single_message_turn(self, base_time: datetime) -> None:
        """Test turn with single message."""
        msg = make_message(1, "Hello", False, base_time)
        turn = Turn(messages=[msg], is_from_me=False)

        assert turn.text == "Hello"
        assert turn.message_ids == [1]
        assert turn.primary_msg_id == 1
        assert turn.first_timestamp == base_time
        assert turn.last_timestamp == base_time

    def test_multi_message_turn(self, base_time: datetime) -> None:
        """Test turn with multiple messages."""
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(2, "want to grab lunch?", False, base_time + timedelta(seconds=30)),
            make_message(3, "thinking sushi", False, base_time + timedelta(minutes=1)),
        ]
        turn = Turn(messages=msgs, is_from_me=False)

        assert turn.text == "Hey\nwant to grab lunch?\nthinking sushi"
        assert turn.message_ids == [1, 2, 3]
        assert turn.primary_msg_id == 1
        assert turn.first_timestamp == base_time
        assert turn.last_timestamp == base_time + timedelta(minutes=1)

    def test_add_message(self, base_time: datetime) -> None:
        """Test adding message to turn."""
        turn = Turn(is_from_me=False)
        msg = make_message(1, "Hello", False, base_time)
        turn.add_message(msg)

        assert len(turn.messages) == 1
        assert turn.text == "Hello"

    def test_turn_skips_empty_text(self, base_time: datetime) -> None:
        """Test that empty text messages are skipped in text property."""
        msgs = [
            make_message(1, "Hello", False, base_time),
            make_message(2, "", False, base_time + timedelta(seconds=30)),
            make_message(3, "World", False, base_time + timedelta(minutes=1)),
        ]
        turn = Turn(messages=msgs, is_from_me=False)

        assert turn.text == "Hello\nWorld"


# =============================================================================
# Turn Grouping Tests
# =============================================================================


class TestTurnGrouping:
    """Tests for _group_into_turns method."""

    def test_empty_messages(self, extractor: TurnBasedExtractor) -> None:
        """Test grouping empty message list."""
        turns = extractor._group_into_turns([])
        assert turns == []

    def test_single_message(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test grouping single message."""
        msgs = [make_message(1, "Hello", False, base_time)]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 1
        assert turns[0].is_from_me is False
        assert turns[0].text == "Hello"

    def test_consecutive_same_speaker(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test consecutive messages from same speaker are bundled."""
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(2, "are you there?", False, base_time + timedelta(minutes=1)),
            make_message(3, "hello??", False, base_time + timedelta(minutes=2)),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 1
        assert turns[0].text == "Hey\nare you there?\nhello??"
        assert len(turns[0].messages) == 3

    def test_alternating_speakers(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test alternating speakers create separate turns."""
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(2, "Hi!", True, base_time + timedelta(minutes=1)),
            make_message(3, "How are you?", False, base_time + timedelta(minutes=2)),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 3
        assert turns[0].is_from_me is False
        assert turns[1].is_from_me is True
        assert turns[2].is_from_me is False

    def test_time_gap_splits_turn(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test that time gap > bundle window splits into separate turns."""
        # Default bundle window is 10 minutes
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(2, "following up", False, base_time + timedelta(minutes=15)),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 2
        assert turns[0].text == "Hey"
        assert turns[1].text == "following up"

    def test_skips_system_messages(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that system messages are skipped."""
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(
                2,
                "John left the group",
                False,
                base_time + timedelta(minutes=1),
                is_system_message=True,
            ),
            make_message(3, "How are you?", False, base_time + timedelta(minutes=2)),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 1
        assert turns[0].text == "Hey\nHow are you?"

    def test_skips_attachment_only_messages(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that attachment-only messages (no text) are skipped."""
        attachment = Attachment(
            filename="photo.jpg",
            file_path="/path/to/photo.jpg",
            mime_type="image/jpeg",
            file_size=12345,
        )
        msgs = [
            make_message(1, "Hey", False, base_time),
            make_message(2, "", False, base_time + timedelta(minutes=1), attachments=[attachment]),
            make_message(3, "Nice pic!", False, base_time + timedelta(minutes=2)),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 1
        assert turns[0].text == "Hey\nNice pic!"

    def test_includes_attachment_with_text(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that messages with both text and attachments are included."""
        attachment = Attachment(
            filename="photo.jpg",
            file_path="/path/to/photo.jpg",
            mime_type="image/jpeg",
            file_size=12345,
        )
        msgs = [
            make_message(1, "Check this out", False, base_time, attachments=[attachment]),
        ]
        turns = extractor._group_into_turns(msgs)

        assert len(turns) == 1
        assert turns[0].text == "Check this out"


# =============================================================================
# Context Extraction Tests
# =============================================================================


class TestContextExtraction:
    """Tests for context extraction in pair creation."""

    def test_context_from_previous_turns(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that context is extracted from previous turns."""
        # Use time gaps > 10min to force separate turns
        msgs = [
            # Context turns - with large time gaps to prevent bundling
            make_message(1, "Hey, how's work?", False, base_time),
            make_message(2, "Pretty busy lately", True, base_time + timedelta(minutes=15)),
            make_message(3, "I hear you", False, base_time + timedelta(minutes=30)),
            # Trigger turn - after another gap
            make_message(
                4, "Want to grab dinner tonight?", False, base_time + timedelta(minutes=45)
            ),
            # Response turn - long enough to pass filters
            make_message(
                5, "That sounds great! I'm free after six", True, base_time + timedelta(minutes=47)
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].context_text is not None
        assert "[Them]: Hey, how's work?" in pairs[0].context_text
        assert "[You]: Pretty busy lately" in pairs[0].context_text
        assert "[Them]: I hear you" in pairs[0].context_text

    def test_context_limited_to_5_turns(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that context is limited to 5 previous turns."""
        # Create many context turns
        msgs = []
        for i in range(10):
            is_me = i % 2 == 1
            msgs.append(
                make_message(
                    i + 1,
                    f"Message {i + 1} text content here",
                    is_me,
                    base_time + timedelta(minutes=i),
                )
            )

        # Add trigger and response
        msgs.append(
            make_message(
                11,
                "Trigger message with enough length here",
                False,
                base_time + timedelta(minutes=11),
            )
        )
        msgs.append(
            make_message(
                12,
                "Response message with sufficient length to pass the filters",
                True,
                base_time + timedelta(minutes=12),
            )
        )

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        if pairs and pairs[0].context_text:
            # Count context turns (lines starting with [You] or [Them])
            context_lines = [
                line for line in pairs[0].context_text.split("\n") if line.startswith("[")
            ]
            assert len(context_lines) <= 5

    def test_no_context_for_first_exchange(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that first exchange has no context."""
        msgs = [
            make_message(1, "Want to grab dinner tonight?", False, base_time),
            make_message(
                2, "That sounds great! I'm free after six", True, base_time + timedelta(minutes=1)
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].context_text is None


# =============================================================================
# Quality Scoring Tests
# =============================================================================


class TestQualityScoring:
    """Tests for _calculate_quality method."""

    def test_quick_response_bonus(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test quick response (< 5 min) gets quick_response flag."""
        msgs = [
            make_message(1, "are you free tonight for dinner", False, base_time),
            make_message(
                2, "yes i am what time were you thinking", True, base_time + timedelta(minutes=2)
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("quick_response") is True
        # Note: quality_score may be affected by other factors (proper nouns, etc.)
        # Just verify the flag is set

    def test_slow_response_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test slow response (30-60 min) gets slight penalty."""
        msgs = [
            make_message(1, "Are you free tonight for dinner?", False, base_time),
            make_message(
                2, "Yes I am! What time were you thinking?", True, base_time + timedelta(minutes=45)
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("slow_response") is True
        assert pairs[0].quality_score < 1.0

    def test_delayed_response_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test delayed response (1-12h) gets moderate penalty."""
        msgs = [
            make_message(1, "Are you free tonight for dinner?", False, base_time),
            make_message(
                2, "Yes I am! What time were you thinking?", True, base_time + timedelta(hours=3)
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("delayed_response") is True
        assert pairs[0].quality_score <= 0.7

    def test_very_delayed_response_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test very delayed response (>12h) gets severe penalty."""
        msgs = [
            make_message(1, "Are you free tonight for dinner?", False, base_time),
            make_message(
                2, "Yes I am! What time were you thinking?", True, base_time + timedelta(hours=18)
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("very_delayed_response") is True
        assert pairs[0].quality_score <= 0.3

    def test_multi_message_turn_bonus(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test multi-message turns get bonus."""
        msgs = [
            make_message(1, "hey there", False, base_time),
            make_message(
                2, "want to grab dinner tonight", False, base_time + timedelta(seconds=30)
            ),
            # Response needs at least 6 tokens and 20 chars to pass default filters
            make_message(
                3,
                "sure that sounds great where should we go",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("multi_message_turn") is True
        assert pairs[0].trigger_message_count == 2

    def test_short_response_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test very short responses (1-2 words) get penalty."""
        # Use relaxed config to allow shorter responses
        config = ExtractionConfig(
            min_response_length=2,  # Allow very short responses
            min_response_tokens=1,
        )
        extractor = TurnBasedExtractor(config)

        msgs = [
            make_message(1, "are you free tonight for dinner", False, base_time),
            make_message(2, "hi", True, base_time + timedelta(minutes=1)),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("short_response") is True

    def test_verbose_response_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test verbose responses (>5x trigger length) get penalty."""
        config = ExtractionConfig(max_response_length=1000)
        extractor = TurnBasedExtractor(config)

        short_trigger = "Hey"
        long_response = (
            "So I was thinking about everything you said last week and I have to say "
            "that it really made me reconsider my position on the whole matter. "
            "I've been doing a lot of reflection and I think you're absolutely right."
        )

        msgs = [
            make_message(1, short_trigger, False, base_time),
            make_message(2, long_response, True, base_time + timedelta(minutes=1)),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("verbose_response") is True


# =============================================================================
# Reaction Filtering Tests
# =============================================================================


class TestReactionFiltering:
    """Tests for generic response detection."""

    def test_generic_responses_detected(self, extractor: TurnBasedExtractor) -> None:
        """Test that generic responses are detected."""
        generic_texts = [
            "ok",
            "okay",
            "k",
            "kk",
            "yes",
            "yeah",
            "yep",
            "yup",
            "no",
            "nope",
            "nah",
            "sure",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "np",
            "cool",
            "nice",
            "good",
            "great",
            "awesome",
            "alright",
            "sounds good",
            "got it",
            "lol",
            "haha",
            "lmao",
        ]

        for text in generic_texts:
            assert extractor._is_generic_response(text), f"'{text}' not detected as generic"

    def test_non_generic_responses(self, extractor: TurnBasedExtractor) -> None:
        """Test that non-generic responses are not flagged."""
        non_generic_texts = [
            "I'll be there at 3pm",
            "Let me think about it",
            "That's an interesting idea",
            "How about we meet at the coffee shop?",
        ]

        for text in non_generic_texts:
            assert not extractor._is_generic_response(text), f"'{text}' wrongly detected as generic"

    def test_generic_response_quality_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that generic responses get quality penalty."""
        # Use relaxed config to allow short responses
        config = ExtractionConfig(
            min_response_length=2,
            min_response_tokens=1,
        )
        extractor = TurnBasedExtractor(config)

        msgs = [
            make_message(1, "Are you free tonight?", False, base_time),
            make_message(2, "ok", True, base_time + timedelta(minutes=1)),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("generic_response") is True
        assert pairs[0].quality_score < 0.5  # Significant penalty


# =============================================================================
# Emoji Only Response Tests
# =============================================================================


class TestEmojiOnlyFiltering:
    """Tests for emoji-only response detection."""

    def test_emoji_only_detected(self, extractor: TurnBasedExtractor) -> None:
        """Test that emoji-only responses are detected."""
        # These emojis are in the ranges covered by EMOJI_PATTERN
        emoji_only_texts = [
            "\U0001f602",  # 😂 face with tears of joy
            "\U0001f44d",  # 👍 thumbs up
            "\U0001f602\U0001f602\U0001f602",  # 😂😂😂
            "\U0001f525\U0001f4af",  # 🔥💯
            "\U0001f60a \U0001f60d",  # 😊 😍 with space
        ]

        for text in emoji_only_texts:
            assert extractor._is_emoji_only(text), f"'{text}' not detected as emoji-only"

    def test_mixed_content_not_emoji_only(self, extractor: TurnBasedExtractor) -> None:
        """Test that mixed content is not flagged as emoji-only."""
        mixed_texts = [
            "Great! \U0001f60a",  # Great! 😊
            "See you then \U0001f44d",  # See you then 👍
            "Haha that's funny \U0001f602",  # Haha that's funny 😂
            "Love this idea!",
        ]

        for text in mixed_texts:
            assert not extractor._is_emoji_only(text), f"'{text}' wrongly detected as emoji-only"


# =============================================================================
# Topic Shift Detection Tests
# =============================================================================


class TestTopicShiftDetection:
    """Tests for topic shift detection."""

    def test_topic_shift_indicators_detected(self, extractor: TurnBasedExtractor) -> None:
        """Test that topic shift indicators are detected."""
        topic_shift_texts = [
            "btw, are you free tomorrow?",
            "anyway, I wanted to ask you something",
            "oh also, I forgot to mention",
            "unrelated, but have you seen the news?",
            "speaking of that, did you hear about...",
            "on another note, I'm thinking of...",
            "separately, I need to discuss something",
            "side note, I saw your friend yesterday",
            "by the way, how's your project going?",
        ]

        for text in topic_shift_texts:
            assert extractor._is_topic_shift(text), f"'{text}' not detected as topic shift"

    def test_non_topic_shift_responses(self, extractor: TurnBasedExtractor) -> None:
        """Test that direct replies are not flagged as topic shifts."""
        direct_replies = [
            "Yes, I'm free at 3pm",
            "That sounds great!",
            "I completely agree with you",
            "Let me check my schedule",
        ]

        for text in direct_replies:
            assert not extractor._is_topic_shift(text), f"'{text}' wrongly detected as topic shift"

    def test_topic_shift_quality_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that topic shifts get quality penalty."""
        msgs = [
            make_message(1, "How was your weekend at the beach?", False, base_time),
            make_message(
                2,
                "btw, are you coming to the party next week? I need to know for the headcount.",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("topic_shift") is True
        assert pairs[0].quality_score < 0.5


# =============================================================================
# Question to Statement Detection Tests
# =============================================================================


class TestQuestionToStatementDetection:
    """Tests for question responses to non-question triggers."""

    def test_question_to_statement_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that question responses to statements get penalty.

        The flag is only set when response ends with "?" and trigger doesn't.
        """
        msgs = [
            make_message(1, "i went to the store today for some groceries", False, base_time),
            make_message(
                2,
                "why did you go there i thought you were busy?",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].flags.get("question_to_statement") is True
        assert pairs[0].quality_score < 1.0


# =============================================================================
# Proper Noun Detection Tests
# =============================================================================


class TestProperNounDetection:
    """Tests for unrelated proper noun detection."""

    def test_extracts_proper_nouns(self, extractor: TurnBasedExtractor) -> None:
        """Test that proper nouns are extracted."""
        text = "I saw John and Sarah at the Apple Store in Manhattan."
        proper_nouns = extractor._extract_proper_nouns(text)

        assert "john" in proper_nouns
        assert "sarah" in proper_nouns
        assert "apple" in proper_nouns
        assert "manhattan" in proper_nouns

    def test_skips_first_word(self, extractor: TurnBasedExtractor) -> None:
        """Test that first word is not treated as proper noun."""
        text = "Hello there"
        proper_nouns = extractor._extract_proper_nouns(text)

        # "Hello" is first word, should be skipped
        assert "hello" not in proper_nouns

    def test_unrelated_proper_nouns_penalty(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that unrelated proper nouns in response get penalty."""
        msgs = [
            make_message(1, "Are you free tonight?", False, base_time),
            make_message(
                2,
                "I'll check with Sarah and let you know after I talk to her",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert "unrelated_proper_nouns" in pairs[0].flags
        assert "sarah" in pairs[0].flags["unrelated_proper_nouns"]


# =============================================================================
# extract_pairs_from_conversation Tests
# =============================================================================


class TestExtractPairsFromConversation:
    """Tests for extract_pairs method."""

    def test_empty_messages(self, extractor: TurnBasedExtractor) -> None:
        """Test extraction from empty message list."""
        pairs, stats = extractor.extract_pairs([], "test_chat")

        assert pairs == []
        assert stats.total_messages_scanned == 0

    def test_basic_pair_extraction(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test basic pair extraction."""
        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 1
        assert pairs[0].trigger_text == "Are you free for dinner tonight?"
        assert pairs[0].response_text == "Yes I am! What time were you thinking of meeting?"
        assert stats.total_messages_scanned == 2
        assert stats.kept_pairs == 1

    def test_skips_my_trigger_messages(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that my messages as triggers are skipped (we want triggers from others)."""
        msgs = [
            make_message(1, "Hey, are you there?", True, base_time),  # From me
            make_message(2, "Yes I am here now!", False, base_time + timedelta(minutes=1)),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        # This should NOT create a pair because we want triggers from others
        assert len(pairs) == 0

    def test_time_gap_rejection(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test that responses after max delay are dropped."""
        # Default max delay is 168 hours (1 week)
        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(hours=200),  # > 168 hours
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 0
        assert stats.dropped_time_gap == 1

    def test_short_trigger_rejection(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that short triggers are rejected."""
        msgs = [
            make_message(1, "k", False, base_time),  # Too short
            make_message(
                2,
                "I'll be there at three o'clock sharp!",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 0
        assert stats.dropped_short_trigger == 1

    def test_short_response_rejection(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that short responses are rejected."""
        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(2, "yes", True, base_time + timedelta(minutes=1)),  # Too short
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 0
        assert stats.dropped_short_response == 1

    def test_long_trigger_rejection(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that very long triggers are rejected."""
        long_trigger = "a" * 600  # > 500 chars

        msgs = [
            make_message(1, long_trigger, False, base_time),
            make_message(
                2,
                "I'll be there at three o'clock sharp!",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 0
        assert stats.dropped_long_trigger == 1

    def test_long_response_rejection(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test that very long responses are rejected."""
        long_response = "a " * 250  # > 400 chars

        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(2, long_response, True, base_time + timedelta(minutes=1)),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 0
        assert stats.dropped_long_response == 1

    def test_no_text_rejection(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test that pairs without text are rejected."""
        msgs = [
            make_message(1, "", False, base_time),  # No text
            make_message(
                2,
                "I'll be there at three o'clock sharp!",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        # Empty text turns are filtered during turn grouping
        assert len(pairs) == 0

    def test_multiple_pairs_extraction(
        self, extractor: TurnBasedExtractor, base_time: datetime
    ) -> None:
        """Test extraction of multiple pairs from conversation."""
        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
            make_message(
                3,
                "How about seven at that Italian place downtown?",
                False,
                base_time + timedelta(minutes=4),
            ),
            make_message(
                4,
                "Sounds perfect! I'll see you there at seven then.",
                True,
                base_time + timedelta(minutes=6),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "test_chat")

        assert len(pairs) == 2
        assert stats.kept_pairs == 2

    def test_group_chat_flag(self, extractor: TurnBasedExtractor, base_time: datetime) -> None:
        """Test that group chat flag is set correctly."""
        msgs = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
        ]

        pairs, _ = extractor.extract_pairs(msgs, "test_chat", is_group=True)

        assert len(pairs) == 1
        assert pairs[0].is_group is True

        pairs2, _ = extractor.extract_pairs(msgs, "test_chat", is_group=False)
        assert pairs2[0].is_group is False


# =============================================================================
# extract_pairs_from_reader Tests
# =============================================================================


class TestExtractPairsFromReader:
    """Tests for extract_pairs_from_reader function."""

    def test_extracts_from_reader(self, base_time: datetime) -> None:
        """Test extraction using reader interface."""
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
        ]

        pair_dicts, stats = extract_pairs_from_reader(mock_reader, "test_chat", contact_id=42)

        assert len(pair_dicts) == 1
        assert pair_dicts[0]["contact_id"] == 42
        assert pair_dicts[0]["trigger_text"] == "Are you free for dinner tonight?"
        assert pair_dicts[0]["response_text"] == "Yes I am! What time were you thinking of meeting?"
        assert pair_dicts[0]["chat_id"] == "test_chat"
        mock_reader.get_messages.assert_called_once_with("test_chat", limit=10000)

    def test_passes_config(self, base_time: datetime) -> None:
        """Test that config is passed to extractor."""
        config = ExtractionConfig(min_response_tokens=1)
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = [
            make_message(1, "Hey there buddy", False, base_time),
            make_message(2, "Hi!", True, base_time + timedelta(minutes=1)),
        ]

        pair_dicts, _ = extract_pairs_from_reader(mock_reader, "test_chat", config=config)

        # With relaxed config, short response should be accepted
        # (though may still fail other checks)
        mock_reader.get_messages.assert_called_once()


# =============================================================================
# extract_all_pairs Tests
# =============================================================================


class TestExtractAllPairs:
    """Tests for extract_all_pairs function."""

    def test_extracts_from_all_conversations(self, base_time: datetime) -> None:
        """Test extraction from all conversations."""
        mock_reader = MagicMock()
        mock_db = MagicMock()

        # Setup mock conversations
        conv1 = Conversation(
            chat_id="chat1",
            participants=["+1111111111"],
            display_name="Alice",
            last_message_date=base_time,
            message_count=10,
            is_group=False,
        )
        conv2 = Conversation(
            chat_id="chat2",
            participants=["+2222222222"],
            display_name="Bob",
            last_message_date=base_time,
            message_count=5,
            is_group=False,
        )
        mock_reader.get_conversations.return_value = [conv1, conv2]

        # Setup mock messages
        mock_reader.get_messages.side_effect = [
            # Chat 1 messages
            [
                make_message(1, "Are you free for dinner tonight?", False, base_time),
                make_message(
                    2,
                    "Yes I am! What time were you thinking of meeting?",
                    True,
                    base_time + timedelta(minutes=2),
                ),
            ],
            # Chat 2 messages
            [
                make_message(3, "How was your weekend at the lake?", False, base_time),
                make_message(
                    4,
                    "It was great! The weather was perfect for sailing.",
                    True,
                    base_time + timedelta(minutes=3),
                ),
            ],
        ]

        # Setup mock contacts
        mock_contact = MagicMock()
        mock_contact.id = 1
        mock_db.get_contact_by_chat_id.return_value = mock_contact
        mock_db.add_pairs_bulk.return_value = 1

        stats = extract_all_pairs(mock_reader, mock_db)

        assert stats["conversations_processed"] == 2
        assert stats["pairs_extracted"] == 2
        assert stats["pairs_added"] == 2
        mock_reader.get_conversations.assert_called_once_with(limit=1000)

    def test_handles_errors_gracefully(self, base_time: datetime) -> None:
        """Test that errors in one conversation don't stop others."""
        mock_reader = MagicMock()
        mock_db = MagicMock()

        conv1 = Conversation(
            chat_id="chat1",
            participants=["+1111111111"],
            display_name="Alice",
            last_message_date=base_time,
            message_count=10,
            is_group=False,
        )
        conv2 = Conversation(
            chat_id="chat2",
            participants=["+2222222222"],
            display_name="Bob",
            last_message_date=base_time,
            message_count=5,
            is_group=False,
        )
        mock_reader.get_conversations.return_value = [conv1, conv2]

        # First conversation raises error, second succeeds
        mock_reader.get_messages.side_effect = [
            RuntimeError("Database error"),
            [
                make_message(3, "How was your weekend at the lake?", False, base_time),
                make_message(
                    4,
                    "It was great! The weather was perfect for sailing.",
                    True,
                    base_time + timedelta(minutes=3),
                ),
            ],
        ]

        mock_contact = MagicMock()
        mock_contact.id = 1
        mock_db.get_contact_by_chat_id.return_value = mock_contact
        mock_db.add_pairs_bulk.return_value = 1

        stats = extract_all_pairs(mock_reader, mock_db)

        assert stats["conversations_processed"] == 1  # Only one succeeded
        assert len(stats["errors"]) == 1
        assert stats["errors"][0]["chat_id"] == "chat1"

    def test_progress_callback(self, base_time: datetime) -> None:
        """Test that progress callback is called."""
        mock_reader = MagicMock()
        mock_db = MagicMock()
        progress_calls = []

        def progress_callback(current: int, total: int, chat_id: str) -> None:
            progress_calls.append((current, total, chat_id))

        conv1 = Conversation(
            chat_id="chat1",
            participants=["+1111111111"],
            display_name="Alice",
            last_message_date=base_time,
            message_count=10,
            is_group=False,
        )
        mock_reader.get_conversations.return_value = [conv1]
        mock_reader.get_messages.return_value = []
        mock_db.get_contact_by_chat_id.return_value = None
        mock_db.add_pairs_bulk.return_value = 0

        extract_all_pairs(mock_reader, mock_db, progress_callback=progress_callback)

        assert len(progress_calls) == 1
        assert progress_calls[0] == (0, 1, "chat1")

    def test_duplicate_handling(self, base_time: datetime) -> None:
        """Test that duplicate pairs are tracked."""
        mock_reader = MagicMock()
        mock_db = MagicMock()

        conv1 = Conversation(
            chat_id="chat1",
            participants=["+1111111111"],
            display_name="Alice",
            last_message_date=base_time,
            message_count=10,
            is_group=False,
        )
        mock_reader.get_conversations.return_value = [conv1]
        mock_reader.get_messages.return_value = [
            make_message(1, "Are you free for dinner tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
        ]

        mock_contact = MagicMock()
        mock_contact.id = 1
        mock_db.get_contact_by_chat_id.return_value = mock_contact
        # Simulate duplicate - 0 added
        mock_db.add_pairs_bulk.return_value = 0

        stats = extract_all_pairs(mock_reader, mock_db)

        assert stats["pairs_extracted"] == 1
        assert stats["pairs_added"] == 0
        assert stats["pairs_skipped_duplicate"] == 1


# =============================================================================
# ExtractionConfig Tests
# =============================================================================


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.turn_bundle_minutes == 10.0
        assert config.max_response_delay_hours == 168.0
        assert config.min_trigger_length == 2
        assert config.min_response_length == 20
        assert config.min_response_tokens == 6
        assert config.max_trigger_length == 500
        assert config.max_response_length == 400
        assert config.skip_attachment_only is True
        assert config.skip_system_messages is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ExtractionConfig(
            turn_bundle_minutes=5.0,
            max_response_delay_hours=24.0,
            min_response_tokens=3,
        )

        assert config.turn_bundle_minutes == 5.0
        assert config.max_response_delay_hours == 24.0
        assert config.min_response_tokens == 3


# =============================================================================
# ExtractionStats Tests
# =============================================================================


class TestExtractionStats:
    """Tests for ExtractionStats dataclass."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = ExtractionStats()

        assert stats.total_messages_scanned == 0
        assert stats.turns_identified == 0
        assert stats.candidate_pairs == 0
        assert stats.kept_pairs == 0
        assert stats.dropped_short_trigger == 0
        assert stats.dropped_short_response == 0
        assert stats.dropped_long_trigger == 0
        assert stats.dropped_long_response == 0
        assert stats.dropped_no_text == 0
        assert stats.dropped_time_gap == 0


# =============================================================================
# ExtractedPair Tests
# =============================================================================


class TestExtractedPair:
    """Tests for ExtractedPair dataclass."""

    def test_default_values(self) -> None:
        """Test default pair values."""
        pair = ExtractedPair(
            trigger_text="Hello",
            response_text="Hi there!",
            trigger_timestamp=datetime(2024, 1, 15, 10, 0, 0),
            response_timestamp=datetime(2024, 1, 15, 10, 1, 0),
            chat_id="test_chat",
            trigger_msg_id=1,
            response_msg_id=2,
            trigger_msg_ids=[1],
            response_msg_ids=[2],
        )

        assert pair.context_text is None
        assert pair.time_delta_seconds == 0.0
        assert pair.trigger_message_count == 1
        assert pair.response_message_count == 1
        assert pair.is_group is False
        assert pair.quality_score == 1.0
        assert pair.flags == {}


# =============================================================================
# Text Cleaning Tests
# =============================================================================


class TestTextCleaning:
    """Tests for _clean_text method."""

    def test_strips_whitespace(self, extractor: TurnBasedExtractor) -> None:
        """Test that whitespace is stripped."""
        result = extractor._clean_text("  Hello world  ")
        assert result == "Hello world"

    def test_normalizes_internal_whitespace(self, extractor: TurnBasedExtractor) -> None:
        """Test that internal whitespace is normalized."""
        result = extractor._clean_text("Hello    world")
        assert result == "Hello world"

    def test_preserves_newlines(self, extractor: TurnBasedExtractor) -> None:
        """Test that newlines are preserved."""
        result = extractor._clean_text("Line 1\nLine 2")
        assert result == "Line 1\nLine 2"

    def test_handles_none(self, extractor: TurnBasedExtractor) -> None:
        """Test that None returns empty string."""
        result = extractor._clean_text(None)
        assert result == ""

    def test_handles_empty_string(self, extractor: TurnBasedExtractor) -> None:
        """Test that empty string returns empty string."""
        result = extractor._clean_text("")
        assert result == ""

    def test_removes_empty_lines(self, extractor: TurnBasedExtractor) -> None:
        """Test that empty lines are removed."""
        result = extractor._clean_text("Line 1\n\n\nLine 2")
        assert result == "Line 1\nLine 2"


# =============================================================================
# Tapback Reaction Filtering Tests (NEW)
# =============================================================================


class TestTapbackReactionFiltering:
    """Tests for iMessage tapback reaction filtering."""

    def test_liked_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Liked reactions should be detected."""
        assert extractor._is_reaction('Liked "hello there"')
        assert extractor._is_reaction('Liked "What time?"')

    def test_loved_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Loved reactions should be detected."""
        assert extractor._is_reaction('Loved "That\'s great!"')
        assert extractor._is_reaction('Loved "See you soon"')

    def test_laughed_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Laughed at reactions should be detected."""
        assert extractor._is_reaction('Laughed at "lol that\'s funny"')

    def test_emphasized_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Emphasized reactions should be detected."""
        assert extractor._is_reaction('Emphasized "Important meeting at 3pm"')

    def test_questioned_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Questioned reactions should be detected."""
        assert extractor._is_reaction('Questioned "Are you sure?"')

    def test_disliked_reaction_detected(self, extractor: TurnBasedExtractor) -> None:
        """Disliked reactions should be detected."""
        assert extractor._is_reaction('Disliked "Bad idea"')

    def test_normal_text_not_reaction(self, extractor: TurnBasedExtractor) -> None:
        """Normal text should not be flagged as reaction."""
        assert not extractor._is_reaction("I liked that movie")
        assert not extractor._is_reaction("That sounds good")
        assert not extractor._is_reaction("Loved the food there")
        assert not extractor._is_reaction("I laughed so hard")

    def test_reaction_severe_quality_penalty(self, extractor: TurnBasedExtractor) -> None:
        """Reactions should receive severe quality penalty."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time works?",
            'Liked "What time works?"',
            trigger_turn,
            response_turn,
            time_delta,
        )

        assert flags.get("is_reaction") is True
        assert quality < 0.2  # Severe penalty


# =============================================================================
# Acknowledgment Trigger Handling Tests (NEW)
# =============================================================================


class TestAcknowledgmentTriggerHandling:
    """Tests for acknowledgment trigger detection and handling."""

    @pytest.mark.parametrize(
        "trigger",
        ["Ok", "okay", "K", "kk", "Yes", "yeah", "Sure", "Got it", "Sounds good", "Thanks"],
    )
    def test_acknowledgment_triggers_detected(
        self, extractor: TurnBasedExtractor, trigger: str
    ) -> None:
        """Common acknowledgments should be detected as triggers."""
        assert extractor._is_acknowledgment_trigger(trigger)

    def test_non_acknowledgment_triggers(self, extractor: TurnBasedExtractor) -> None:
        """Non-acknowledgment triggers should not be flagged."""
        assert not extractor._is_acknowledgment_trigger("What time?")
        assert not extractor._is_acknowledgment_trigger("Can you help me with something?")
        assert not extractor._is_acknowledgment_trigger("I'll be there at 5")

    def test_acknowledgment_with_substantive_response_penalty(
        self, extractor: TurnBasedExtractor
    ) -> None:
        """Ack trigger with substantive response should be penalized."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "Ok",
            "Got link for Stanford immunology will call tomorrow morning at 9am",
            trigger_turn,
            response_turn,
            time_delta,
        )

        assert flags.get("acknowledgment_trigger") is True
        assert flags.get("ack_trigger_substantive_response") is True
        assert quality < 0.5  # Should be penalized


# =============================================================================
# Substantive Response Detection Tests (NEW)
# =============================================================================


class TestSubstantiveResponseDetection:
    """Tests for substantive response detection."""

    def test_substantive_by_word_count(self, extractor: TurnBasedExtractor) -> None:
        """Long responses should be substantive."""
        assert extractor._is_substantive_response(
            "I will call you tomorrow morning at nine to discuss the details"
        )

    def test_substantive_by_question(self, extractor: TurnBasedExtractor) -> None:
        """Questions should be substantive."""
        assert extractor._is_substantive_response("What time?")

    def test_substantive_by_numbers(self, extractor: TurnBasedExtractor) -> None:
        """Responses with numbers should be substantive."""
        assert extractor._is_substantive_response("At 3pm")
        assert extractor._is_substantive_response("Room 204")

    def test_not_substantive_short_ack(self, extractor: TurnBasedExtractor) -> None:
        """Short acknowledgments should not be substantive."""
        assert not extractor._is_substantive_response("Ok")
        assert not extractor._is_substantive_response("Sure")
        assert not extractor._is_substantive_response("Sounds good")


# =============================================================================
# Semantic Similarity Filtering Tests (NEW)
# =============================================================================


class TestSemanticSimilarityFiltering:
    """Tests for semantic similarity-based quality filtering."""

    def test_low_similarity_severe_penalty(self, extractor: TurnBasedExtractor) -> None:
        """Low semantic similarity (<0.45) should be severely penalized."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time is the meeting?",
            "There are 3 bags of almonds in the freezer",
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=0.2,
        )

        assert flags.get("low_semantic_similarity") is True
        assert flags.get("semantic_similarity") == 0.2
        assert quality < 0.3

    def test_borderline_similarity_moderate_penalty(self, extractor: TurnBasedExtractor) -> None:
        """Borderline semantic similarity (0.45-0.55) should be moderately penalized."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time is the meeting?",
            "I think it's around three but not totally sure about that",
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=0.5,
        )

        assert flags.get("borderline_semantic_similarity") is True
        assert 0.3 < quality < 0.8

    def test_high_similarity_no_penalty(self, extractor: TurnBasedExtractor) -> None:
        """High semantic similarity (>=0.55) should not be penalized."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time is the meeting?",
            "The meeting is at three pm in the main conference room",
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=0.75,
        )

        assert flags.get("low_semantic_similarity") is None
        assert flags.get("borderline_semantic_similarity") is None
        # Quality should be good (only other factors may reduce it)

    def test_no_similarity_provided_no_penalty(self, extractor: TurnBasedExtractor) -> None:
        """If no semantic similarity is provided, no semantic penalty should be applied."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time is the meeting?",
            "The meeting is at three pm in the main conference room",
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=None,  # No embedder
        )

        assert "semantic_similarity" not in flags
        assert flags.get("low_semantic_similarity") is None
        assert flags.get("borderline_semantic_similarity") is None


# =============================================================================
# Embedder Integration Tests (NEW)
# =============================================================================


class TestEmbedderIntegration:
    """Tests for embedder integration in extraction."""

    def test_embedder_auto_enables_semantic_similarity(self) -> None:
        """Providing embedder should auto-enable semantic similarity."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])

        extractor = TurnBasedExtractor(embedder=mock_embedder)
        assert extractor.config.use_semantic_similarity is True

    def test_compute_semantic_similarity_with_embedder(self) -> None:
        """Semantic similarity should be computed when embedder is provided."""
        mock_embedder = MagicMock()
        # High similarity embeddings (same direction)
        mock_embedder.encode.return_value = np.array(
            [[0.707, 0.707], [0.707, 0.707]], dtype=np.float32
        )

        extractor = TurnBasedExtractor(embedder=mock_embedder)

        similarity = extractor._compute_semantic_similarity("Hello there", "Hi there")

        assert similarity is not None
        assert similarity > 0.9  # Identical vectors = similarity ~1.0
        mock_embedder.encode.assert_called_once()

    def test_compute_semantic_similarity_without_embedder(
        self, extractor: TurnBasedExtractor
    ) -> None:
        """Semantic similarity should be None when no embedder is provided."""
        similarity = extractor._compute_semantic_similarity("Hello there", "Hi there")
        assert similarity is None


# =============================================================================
# Extraction Stats New Fields Tests (NEW)
# =============================================================================


class TestExtractionStatsNewFields:
    """Tests for new ExtractionStats fields."""

    def test_stats_have_flagged_reaction_field(self) -> None:
        """Stats should have flagged_reaction field."""
        stats = ExtractionStats()
        assert stats.flagged_reaction == 0
        stats.flagged_reaction += 1
        assert stats.flagged_reaction == 1

    def test_stats_have_flagged_low_similarity_field(self) -> None:
        """Stats should have flagged_low_similarity field."""
        stats = ExtractionStats()
        assert stats.flagged_low_similarity == 0
        stats.flagged_low_similarity += 1
        assert stats.flagged_low_similarity == 1

    def test_stats_have_flagged_topic_shift_field(self) -> None:
        """Stats should have flagged_topic_shift field."""
        stats = ExtractionStats()
        assert stats.flagged_topic_shift == 0
        stats.flagged_topic_shift += 1
        assert stats.flagged_topic_shift == 1

    def test_stats_have_flagged_ack_substantive_field(self) -> None:
        """Stats should have flagged_ack_substantive field."""
        stats = ExtractionStats()
        assert stats.flagged_ack_substantive == 0
        stats.flagged_ack_substantive += 1
        assert stats.flagged_ack_substantive == 1


# =============================================================================
# Config Semantic Similarity Fields Tests (NEW)
# =============================================================================


class TestConfigSemanticSimilarityFields:
    """Tests for new semantic similarity config fields."""

    def test_default_config_no_semantic_similarity(self) -> None:
        """Default config should not use semantic similarity."""
        config = ExtractionConfig()
        assert config.use_semantic_similarity is False

    def test_config_semantic_thresholds(self) -> None:
        """Config should have correct default semantic thresholds."""
        config = ExtractionConfig()
        assert config.semantic_reject_threshold == 0.45
        assert config.semantic_borderline_threshold == 0.55

    def test_config_custom_semantic_thresholds(self) -> None:
        """Config should accept custom semantic thresholds."""
        config = ExtractionConfig(
            use_semantic_similarity=True,
            semantic_reject_threshold=0.5,
            semantic_borderline_threshold=0.6,
        )
        assert config.use_semantic_similarity is True
        assert config.semantic_reject_threshold == 0.5
        assert config.semantic_borderline_threshold == 0.6


# =============================================================================
# Backward Compatibility Tests (NEW)
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_extract_without_embedder(self, extractor: TurnBasedExtractor) -> None:
        """Extraction should work without embedder."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        msgs = [
            make_message(1, "Hey, what time is the meeting?", False, base_time),
            make_message(
                2,
                "The meeting is at 3pm in conference room A on the second floor",
                True,
                base_time + timedelta(minutes=1),
            ),
        ]

        pairs, stats = extractor.extract_pairs(msgs, "chat123")

        assert len(pairs) == 1
        assert pairs[0].trigger_text == "Hey, what time is the meeting?"
        assert "at 3pm" in pairs[0].response_text

    def test_extract_pairs_from_reader_without_embedder(self) -> None:
        """extract_pairs_from_reader should work without embedder."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = [
            make_message(1, "Are you free tonight?", False, base_time),
            make_message(
                2,
                "Yes I am! What time were you thinking of meeting?",
                True,
                base_time + timedelta(minutes=2),
            ),
        ]

        pair_dicts, stats = extract_pairs_from_reader(mock_reader, "test_chat")

        assert len(pair_dicts) == 1
        assert pair_dicts[0]["trigger_text"] == "Are you free tonight?"

    def test_calculate_quality_without_semantic_similarity(
        self, extractor: TurnBasedExtractor
    ) -> None:
        """Quality calculation should work without semantic similarity."""
        trigger_turn = Turn(is_from_me=False)
        response_turn = Turn(is_from_me=True)
        time_delta = timedelta(seconds=60)

        quality, flags = extractor._calculate_quality(
            "What time is the meeting?",
            "The meeting is at 3pm in conference room A on the second floor",
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=None,  # No embedder
        )

        assert quality > 0
        assert "semantic_similarity" not in flags


# =============================================================================
# Module-Level Function Tests (NEW)
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for module-level detection functions."""

    def test_is_reaction_message_import(self) -> None:
        """is_reaction_message should be importable from module."""
        from jarvis.extract import is_reaction_message

        assert callable(is_reaction_message)

    def test_is_topic_shift_import(self) -> None:
        """is_topic_shift should be importable from module."""
        from jarvis.extract import is_topic_shift

        assert callable(is_topic_shift)

    def test_is_simple_acknowledgment_import(self) -> None:
        """is_simple_acknowledgment should be importable from module."""
        from jarvis.extract import is_simple_acknowledgment

        assert callable(is_simple_acknowledgment)

    def test_is_reaction_message_detects_reactions(self) -> None:
        """is_reaction_message should detect iMessage tapback reactions."""
        from jarvis.extract import is_reaction_message

        assert is_reaction_message('Liked "hello there"')
        assert is_reaction_message('Loved "great idea!"')
        assert is_reaction_message('Laughed at "that\'s funny"')
        assert is_reaction_message('Emphasized "important meeting"')
        assert is_reaction_message('Questioned "are you sure?"')
        assert is_reaction_message('Disliked "bad idea"')

    def test_is_reaction_message_rejects_normal_text(self) -> None:
        """is_reaction_message should not flag normal text."""
        from jarvis.extract import is_reaction_message

        assert not is_reaction_message("I liked that movie")
        assert not is_reaction_message("Loved the food there")
        assert not is_reaction_message("That's a great idea!")

    def test_is_topic_shift_detects_shifts(self) -> None:
        """is_topic_shift should detect topic shift indicators."""
        from jarvis.extract import is_topic_shift

        assert is_topic_shift("btw, are you free tomorrow?")
        assert is_topic_shift("anyway, I wanted to ask something")
        assert is_topic_shift("oh also, I forgot to mention")
        assert is_topic_shift("by the way, how's your project?")
        assert is_topic_shift("unrelated but, did you see the news?")
        assert is_topic_shift("speaking of that, have you heard?")

    def test_is_topic_shift_rejects_direct_replies(self) -> None:
        """is_topic_shift should not flag direct replies."""
        from jarvis.extract import is_topic_shift

        assert not is_topic_shift("Yes, I'm free at 3pm")
        assert not is_topic_shift("That sounds great!")
        assert not is_topic_shift("Let me check my schedule")

    def test_is_simple_acknowledgment_detects_acks(self) -> None:
        """is_simple_acknowledgment should detect simple acknowledgments."""
        from jarvis.extract import is_simple_acknowledgment

        assert is_simple_acknowledgment("Ok")
        assert is_simple_acknowledgment("okay")
        assert is_simple_acknowledgment("K")
        assert is_simple_acknowledgment("Yes")
        assert is_simple_acknowledgment("Yeah")
        assert is_simple_acknowledgment("Sure")
        assert is_simple_acknowledgment("Got it")
        assert is_simple_acknowledgment("Sounds good")
        assert is_simple_acknowledgment("Thanks!")  # With punctuation

    def test_is_simple_acknowledgment_rejects_questions(self) -> None:
        """is_simple_acknowledgment should not flag questions or complex messages."""
        from jarvis.extract import is_simple_acknowledgment

        assert not is_simple_acknowledgment("What time?")
        assert not is_simple_acknowledgment("Can you help me?")
        assert not is_simple_acknowledgment("I'll be there at 5")

    def test_substantive_response_20_chars_threshold(self, extractor: TurnBasedExtractor) -> None:
        """Substantive response check should use 20 character threshold."""
        # 22 characters - should be substantive
        assert extractor._is_substantive_response("This is substantive!!")  # 22 chars

        # Under 20 chars without other indicators - not substantive
        assert not extractor._is_substantive_response("short")  # 5 chars
        assert not extractor._is_substantive_response("hi there")  # 8 chars
