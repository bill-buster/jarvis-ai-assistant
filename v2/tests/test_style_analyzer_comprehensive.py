"""Comprehensive tests for the style analyzer module.

These tests verify actual functionality including edge cases,
boundary conditions, and integration points.
"""

from __future__ import annotations

import pytest

from core.generation.style_analyzer import (
    ABBREVIATIONS,
    EMOJI_PATTERN,
    StyleAnalyzer,
    UserStyle,
)


class TestUserStyleDefaults:
    """Test UserStyle dataclass default values."""

    def test_default_values(self) -> None:
        """Test that UserStyle has sensible defaults."""
        style = UserStyle()

        assert style.avg_word_count == 8.0
        assert style.avg_char_count == 40.0
        assert style.uses_emoji is False
        assert style.emoji_frequency == 0.0
        assert style.capitalization == "normal"
        assert style.uses_abbreviations is False
        assert style.punctuation_style == "normal"
        assert style.common_phrases == []
        assert style.enthusiasm_level == "medium"
        assert style.formality_score == 0.5
        assert style.humor_style == "none"
        assert style.response_tendency == "balanced"


class TestEmojiDetection:
    """Test emoji detection patterns."""

    @pytest.mark.parametrize(
        "text,has_emoji",
        [
            # Standard emojis
            ("Hello! 😊", True),
            ("Great job! 👍", True),
            ("That's funny 😂", True),
            # Multiple emojis
            ("Party! 🎉🎊🥳", True),
            # No emojis
            ("Hello world", False),
            ("Just text :)", False),  # ASCII faces are not emojis
            (":D", False),
            # Edge cases
            ("", False),
            ("   ", False),
        ],
    )
    def test_emoji_pattern_matching(self, text: str, has_emoji: bool) -> None:
        """Test that emoji pattern correctly identifies emojis."""
        result = bool(EMOJI_PATTERN.search(text))
        assert result == has_emoji


class TestAbbreviations:
    """Test abbreviation detection."""

    def test_common_abbreviations_in_set(self) -> None:
        """Test that common abbreviations are in the ABBREVIATIONS set."""
        expected = [
            "u",
            "ur",
            "r",
            "lol",
            "lmao",
            "omg",
            "idk",
            "tbh",
            "ngl",
            "rn",
            "bc",
            "w",
            "b4",
            "2",
            "4",
            "thx",
            "ty",
            "np",
            "pls",
            "plz",
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
        ]
        for abbrev in expected:
            assert abbrev in ABBREVIATIONS


class TestStyleAnalyzerCore:
    """Test core StyleAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_empty_messages_returns_default(self, analyzer: StyleAnalyzer) -> None:
        """Test that empty messages return default UserStyle."""
        style = analyzer.analyze([])
        assert style == UserStyle()

    def test_messages_without_text_returns_default(self, analyzer: StyleAnalyzer) -> None:
        """Test that messages without text return default UserStyle."""
        messages = [{"sender": "me"}, {"text": "", "sender": "me"}, {"text": None, "sender": "me"}]
        style = analyzer.analyze(messages)
        assert style == UserStyle()

    def test_word_count_calculation(self, analyzer: StyleAnalyzer) -> None:
        """Test average word count calculation."""
        messages = [
            {"text": "Hello there"},  # 2 words
            {"text": "This is a longer message with more words"},  # 8 words
            {"text": "Short one"},  # 2 words
        ]
        style = analyzer.analyze(messages)
        # Average: (2 + 8 + 2) / 3 = 4
        assert style.avg_word_count == 4.0

    def test_char_count_calculation(self, analyzer: StyleAnalyzer) -> None:
        """Test average character count calculation."""
        messages = [
            {"text": "Hello"},  # 5 chars
            {"text": "World"},  # 5 chars
        ]
        style = analyzer.analyze(messages)
        assert style.avg_char_count == 5.0


class TestCapitalizationDetection:
    """Test capitalization style detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_lowercase_detection(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of lowercase style (>70% lowercase)."""
        messages = [
            {"text": "hey how are you"},
            {"text": "doing good thanks"},
            {"text": "sounds good"},
            {"text": "cool lets do it"},
            {"text": "ok"},
            {"text": "yep"},
            {"text": "nice"},
            {"text": "later"},
            {"text": "sure thing"},
            {"text": "One with caps"},  # Only 1/10 has caps
        ]
        style = analyzer.analyze(messages)
        assert style.capitalization == "lowercase"

    def test_all_caps_detection(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of all caps style (>30% all caps)."""
        messages = [
            {"text": "OMG THIS IS GREAT"},
            {"text": "SO EXCITED"},
            {"text": "YESSS"},
            {"text": "normal message"},
            {"text": "another normal one"},
        ]
        style = analyzer.analyze(messages)
        # 3/5 = 60% are all caps, > 30% threshold
        assert style.capitalization == "all_caps"

    def test_normal_capitalization(self, analyzer: StyleAnalyzer) -> None:
        """Test normal capitalization style."""
        messages = [
            {"text": "Hello there!"},
            {"text": "How are you?"},
            {"text": "I'm doing great."},
            {"text": "See you later!"},
        ]
        style = analyzer.analyze(messages)
        assert style.capitalization == "normal"

    def test_short_caps_not_counted(self, analyzer: StyleAnalyzer) -> None:
        """Test that short all-caps words (<=2 chars) don't count as all_caps style."""
        messages = [
            {"text": "OK"},  # Short, should not count
            {"text": "Hi there"},
            {"text": "NO"},  # Short, should not count
        ]
        style = analyzer.analyze(messages)
        # Short messages like "OK" and "NO" (len <= 2) shouldn't count
        assert style.capitalization == "normal"


class TestEmojiStyleDetection:
    """Test emoji usage detection in style analysis."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_uses_emoji_true(self, analyzer: StyleAnalyzer) -> None:
        """Test that emoji usage is detected."""
        messages = [
            {"text": "Hello! 😊"},
            {"text": "Thanks! 👍"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_emoji is True
        assert style.emoji_frequency == 1.0  # 100% have emojis

    def test_uses_emoji_false(self, analyzer: StyleAnalyzer) -> None:
        """Test that lack of emojis is detected."""
        messages = [
            {"text": "Hello!"},
            {"text": "Thanks!"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_emoji is False
        assert style.emoji_frequency == 0.0

    def test_partial_emoji_usage(self, analyzer: StyleAnalyzer) -> None:
        """Test partial emoji usage calculation."""
        messages = [
            {"text": "Hello! 😊"},
            {"text": "No emoji here"},
            {"text": "Another one 👍"},
            {"text": "Plain text"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_emoji is True
        assert style.emoji_frequency == 0.5  # 2/4 = 50%


class TestAbbreviationDetection:
    """Test abbreviation usage detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_abbreviations_detected(self, analyzer: StyleAnalyzer) -> None:
        """Test that common abbreviations are detected."""
        messages = [
            {"text": "lol that's funny"},
            {"text": "u coming?"},
            {"text": "gonna be late"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_abbreviations is True

    def test_no_abbreviations(self, analyzer: StyleAnalyzer) -> None:
        """Test when no abbreviations are used."""
        messages = [
            {"text": "Hello there"},
            {"text": "How are you doing?"},
            {"text": "That is very nice"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_abbreviations is False


class TestPunctuationStyleDetection:
    """Test punctuation style detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_expressive_punctuation(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of expressive punctuation (lots of !)."""
        messages = [
            {"text": "That's amazing!!"},
            {"text": "So excited!"},
            {"text": "Yes!! Love it!"},
            {"text": "Awesome!!!"},
        ]
        style = analyzer.analyze(messages)
        # 8 exclamation marks / 4 messages = 2 per message > 1.5 threshold
        assert style.punctuation_style == "expressive"

    def test_minimal_punctuation(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of minimal punctuation."""
        messages = [
            {"text": "hey"},
            {"text": "ok sounds good"},
            {"text": "cool"},
            {"text": "later"},
        ]
        style = analyzer.analyze(messages)
        # 0 punctuation marks / 4 messages < 0.5 threshold
        assert style.punctuation_style == "minimal"

    def test_normal_punctuation(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of normal punctuation."""
        messages = [
            {"text": "Hello."},
            {"text": "How are you?"},
            {"text": "That's good!"},
            {"text": "See you later."},
        ]
        style = analyzer.analyze(messages)
        assert style.punctuation_style == "normal"


class TestEnthusiasmDetection:
    """Test enthusiasm level detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_high_enthusiasm(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of high enthusiasm."""
        messages = [
            {"text": "OMG THAT'S AMAZING!!"},
            {"text": "SO EXCITED! 🎉"},
            {"text": "YESSS!!"},
            {"text": "This is great! 😊"},
        ]
        style = analyzer.analyze(messages)
        assert style.enthusiasm_level == "high"

    def test_low_enthusiasm(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of low enthusiasm."""
        messages = [
            {"text": "ok"},
            {"text": "sure"},
            {"text": "fine"},
            {"text": "whatever"},
            {"text": "i guess"},
        ]
        style = analyzer.analyze(messages)
        assert style.enthusiasm_level == "low"

    def test_medium_enthusiasm(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of medium enthusiasm.

        Enthusiasm score = (exclaim_messages/total)*0.4 + (caps_messages/total)*0.3 + (emoji_messages/total)*0.3
        Medium is between 0.15 and 0.4
        """
        messages = [
            {"text": "Sounds good!"},  # Has !
            {"text": "Cool!"},  # Has !
            {"text": "Nice 😊"},  # Has emoji
            {"text": "Thanks"},  # Neutral
        ]
        # Score = (2/4)*0.4 + (0/4)*0.3 + (1/4)*0.3 = 0.2 + 0 + 0.075 = 0.275 (medium)
        style = analyzer.analyze(messages)
        assert style.enthusiasm_level == "medium"


class TestFormalityAnalysis:
    """Test formality score analysis."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_very_casual_formality(self, analyzer: StyleAnalyzer) -> None:
        """Test very casual (low formality) detection."""
        messages = [
            {"text": "lol bro that's hilarious"},
            {"text": "gonna wanna hang later"},
            {"text": "idk tbh ngl"},
            {"text": "bruh wtf"},
        ]
        style = analyzer.analyze(messages)
        assert style.formality_score < 0.3

    def test_formal_style(self, analyzer: StyleAnalyzer) -> None:
        """Test formal style detection."""
        messages = [
            {"text": "Thank you for your help."},
            {"text": "I appreciate your time."},
            {"text": "Please let me know if you need anything."},
            {"text": "Certainly, I will look into that."},
        ]
        style = analyzer.analyze(messages)
        assert style.formality_score > 0.3

    def test_mixed_formality(self, analyzer: StyleAnalyzer) -> None:
        """Test that mixed formality results in middle score."""
        messages = [
            {"text": "Thank you!"},  # Formal
            {"text": "lol yeah for sure"},  # Casual
        ]
        style = analyzer.analyze(messages)
        assert 0.2 < style.formality_score < 0.8


class TestHumorStyleDetection:
    """Test humor style detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_no_humor(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of no humor indicators."""
        messages = [
            {"text": "The meeting is at 3pm."},
            {"text": "Okay, I'll be there."},
            {"text": "Thanks for letting me know."},
        ]
        style = analyzer.analyze(messages)
        assert style.humor_style == "none"

    def test_playful_humor(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of playful humor."""
        messages = [
            {"text": "haha that's funny"},
            {"text": "just kidding!"},
            {"text": "hehe yeah"},
        ]
        style = analyzer.analyze(messages)
        assert style.humor_style in ["playful", "dry"]

    def test_expressive_humor(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of expressive humor."""
        messages = [
            {"text": "LMAO 😂"},
            {"text": "lmaooo I'm dying"},
            {"text": "omg 🤣 jk"},
            {"text": "that's hilarious lmao"},
        ]
        style = analyzer.analyze(messages)
        assert style.humor_style == "expressive"


class TestResponseTendency:
    """Test response tendency detection."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_brief_tendency(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of brief response tendency (<5 words avg)."""
        messages = [
            {"text": "ok"},
            {"text": "sure"},
            {"text": "sounds good"},
            {"text": "yep"},
        ]
        style = analyzer.analyze(messages)
        assert style.response_tendency == "brief"

    def test_detailed_tendency(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of detailed response tendency (>15 words avg)."""
        messages = [
            {
                "text": "I think that's a really great idea and we should definitely "
                "explore it further to see what possibilities arise"
            },
            {
                "text": "The project is going well and I've made significant progress "
                "on several key aspects that were previously blocking us"
            },
        ]
        style = analyzer.analyze(messages)
        assert style.response_tendency == "detailed"

    def test_balanced_tendency(self, analyzer: StyleAnalyzer) -> None:
        """Test detection of balanced response tendency (5-15 words avg)."""
        messages = [
            {"text": "That sounds like a good plan to me"},
            {"text": "I'll check with the team and get back"},
            {"text": "Definitely, let's schedule it for next week"},
        ]
        style = analyzer.analyze(messages)
        assert style.response_tendency == "balanced"


class TestCommonPhraseExtraction:
    """Test common phrase extraction."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_phrase_extraction(self, analyzer: StyleAnalyzer) -> None:
        """Test that common phrases are extracted."""
        messages = [
            {"text": "sounds good, let me know"},
            {"text": "sounds good!"},
            {"text": "sounds good to me"},
            {"text": "let me check"},
            {"text": "let me see"},
        ]
        style = analyzer.analyze(messages)
        # "sounds good" appears in multiple messages
        assert "sounds good" in style.common_phrases

    def test_phrase_limit(self, analyzer: StyleAnalyzer) -> None:
        """Test that common phrases are limited to 5."""
        messages = [
            {"text": "sounds good"},
            {"text": "let me check"},
            {"text": "want to do something"},
            {"text": "do you want to"},
            {"text": "are you free"},
            {"text": "can you help"},
            {"text": "lol that's funny"},
        ]
        style = analyzer.analyze(messages)
        assert len(style.common_phrases) <= 5


class TestPromptInstructions:
    """Test conversion of style to prompt instructions."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_brief_style_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test instructions for brief style."""
        style = UserStyle(avg_word_count=4.0)
        instructions = analyzer.to_prompt_instructions(style)
        assert "very short" in instructions.lower() or "under 6" in instructions.lower()

    def test_lowercase_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test instructions for lowercase style."""
        style = UserStyle(capitalization="lowercase")
        instructions = analyzer.to_prompt_instructions(style)
        assert "lowercase" in instructions.lower()

    def test_emoji_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test emoji-related instructions."""
        # Frequent emoji use
        style_emoji = UserStyle(uses_emoji=True, emoji_frequency=0.5)
        instructions = analyzer.to_prompt_instructions(style_emoji)
        assert "emoji" in instructions.lower()

        # No emoji use
        style_no_emoji = UserStyle(uses_emoji=False, emoji_frequency=0.0)
        instructions = analyzer.to_prompt_instructions(style_no_emoji)
        assert "don't use emojis" in instructions.lower()

    def test_abbreviation_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test abbreviation instructions."""
        style = UserStyle(uses_abbreviations=True)
        instructions = analyzer.to_prompt_instructions(style)
        assert "abbreviation" in instructions.lower()

    def test_expressive_punctuation_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test expressive punctuation instructions."""
        style = UserStyle(punctuation_style="expressive")
        instructions = analyzer.to_prompt_instructions(style)
        assert "exclamation" in instructions.lower()

    def test_minimal_punctuation_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test minimal punctuation instructions."""
        style = UserStyle(punctuation_style="minimal")
        instructions = analyzer.to_prompt_instructions(style)
        assert "minimal" in instructions.lower()


class TestBuildStyleInstructions:
    """Test the build_style_instructions method that combines multiple sources."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_default_when_no_data(self, analyzer: StyleAnalyzer) -> None:
        """Test that default is returned when no useful data."""
        style = UserStyle()
        instructions = analyzer.build_style_instructions(style, None, None)
        # Should fall back to detailed instructions
        assert len(instructions) > 0

    def test_global_style_priority(self, analyzer: StyleAnalyzer) -> None:
        """Test that global style takes priority."""

        # Create a mock global style
        class MockGlobalStyle:
            capitalization = "lowercase"
            punctuation_style = "minimal"
            uses_abbreviations = True
            avg_word_count = 5.0

        style = UserStyle()
        instructions = analyzer.build_style_instructions(style, None, MockGlobalStyle())

        assert "lowercase" in instructions
        # The instruction format may vary - check for punctuation-related content
        assert "periods" in instructions.lower() or "punctuation" in instructions.lower()
        assert "abbreviations" in instructions

    def test_profile_relationship_adjustment(self, analyzer: StyleAnalyzer) -> None:
        """Test that profile relationship type affects instructions."""

        class MockProfile:
            relationship_type = "coworker"
            total_messages = 50

        style = UserStyle()
        instructions = analyzer.build_style_instructions(style, MockProfile(), None)
        assert "professional" in instructions.lower()

    def test_profile_emoji_instruction(self, analyzer: StyleAnalyzer) -> None:
        """Test that profile emoji setting is included."""

        class MockProfile:
            relationship_type = "unknown"
            total_messages = 20
            uses_emoji = True
            avg_your_length = 30
            tone = ""
            uses_slang = False
            is_playful = False

        style = UserStyle()
        instructions = analyzer.build_style_instructions(style, MockProfile(), None)
        assert "emojis okay" in instructions.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        return StyleAnalyzer()

    def test_single_message(self, analyzer: StyleAnalyzer) -> None:
        """Test analysis of a single message."""
        messages = [{"text": "Hello there!"}]
        style = analyzer.analyze(messages)
        assert style.avg_word_count == 2.0
        assert style.avg_char_count == 12.0

    def test_very_long_message(self, analyzer: StyleAnalyzer) -> None:
        """Test analysis of very long messages."""
        long_text = " ".join(["word"] * 100)
        messages = [{"text": long_text}]
        style = analyzer.analyze(messages)
        assert style.avg_word_count == 100.0
        assert style.response_tendency == "detailed"

    def test_unicode_in_messages(self, analyzer: StyleAnalyzer) -> None:
        """Test that unicode characters are handled correctly."""
        messages = [
            {"text": "Café résumé"},
            {"text": "日本語テキスト"},
            {"text": "مرحبا"},
        ]
        # Should not raise an exception
        style = analyzer.analyze(messages)
        assert style is not None

    def test_only_emojis(self, analyzer: StyleAnalyzer) -> None:
        """Test messages that are only emojis."""
        messages = [
            {"text": "😊"},
            {"text": "👍👍"},
            {"text": "🎉🎊🥳"},
        ]
        style = analyzer.analyze(messages)
        assert style.uses_emoji is True
        assert style.emoji_frequency == 1.0

    def test_mixed_empty_and_valid(self, analyzer: StyleAnalyzer) -> None:
        """Test that empty messages are filtered out."""
        messages = [
            {"text": "Hello"},
            {"text": ""},
            {"text": None},
            {"text": "World"},
        ]
        style = analyzer.analyze(messages)
        # Only "Hello" and "World" should be counted
        assert style.avg_word_count == 1.0
        assert style.avg_char_count == 5.0
