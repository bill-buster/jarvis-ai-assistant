"""Tests for the style analyzer module."""

from __future__ import annotations

import pytest

from core.generation.style_analyzer import StyleAnalyzer, UserStyle


class TestStyleAnalyzer:
    """Tests for StyleAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> StyleAnalyzer:
        """Create a StyleAnalyzer instance."""
        return StyleAnalyzer()

    @pytest.fixture
    def casual_messages(self) -> list[dict]:
        """Sample casual messages for style analysis."""
        return [
            {"text": "hey whats up", "is_from_me": True},
            {"text": "lol thats hilarious 😂", "is_from_me": True},
            {"text": "yeah for sure", "is_from_me": True},
            {"text": "omw brb", "is_from_me": True},
            {"text": "sounds good 👍", "is_from_me": True},
        ]

    @pytest.fixture
    def formal_messages(self) -> list[dict]:
        """Sample formal messages for style analysis."""
        return [
            {"text": "Good morning. I hope this message finds you well.", "is_from_me": True},
            {"text": "Thank you for your prompt response.", "is_from_me": True},
            {"text": "I would be happy to assist with that matter.", "is_from_me": True},
            {"text": "Please let me know if you have any questions.", "is_from_me": True},
            {"text": "Best regards.", "is_from_me": True},
        ]

    def test_analyze_casual_style(
        self, analyzer: StyleAnalyzer, casual_messages: list[dict]
    ) -> None:
        """Test analysis of casual messaging style."""
        style = analyzer.analyze(casual_messages)

        assert isinstance(style, UserStyle)
        assert style.uses_emoji is True
        assert style.capitalization in ("lowercase", "normal")
        assert style.uses_abbreviations is True

    def test_analyze_formal_style(
        self, analyzer: StyleAnalyzer, formal_messages: list[dict]
    ) -> None:
        """Test analysis of formal messaging style."""
        style = analyzer.analyze(formal_messages)

        assert isinstance(style, UserStyle)
        assert style.uses_emoji is False
        assert style.avg_char_count > 30
        assert style.punctuation_style in ("normal", "expressive")

    def test_analyze_empty_messages(self, analyzer: StyleAnalyzer) -> None:
        """Test analysis with no messages returns default style."""
        style = analyzer.analyze([])

        assert isinstance(style, UserStyle)
        assert style.avg_word_count == 8.0  # Default value

    def test_analyze_processes_all_messages(self, analyzer: StyleAnalyzer) -> None:
        """Test that analysis processes all messages passed to it.

        Note: Filtering by is_from_me is done by the caller (reply_generator),
        not by the analyzer itself. The analyzer processes all messages given.
        """
        messages = [
            {"text": "im good thanks", "is_from_me": True},
            {"text": "doing well", "is_from_me": True},
        ]
        style = analyzer.analyze(messages)

        # 3 + 2 = 5 words / 2 messages = 2.5 avg
        assert style.avg_word_count == 2.5

    def test_detect_emoji_usage(self, analyzer: StyleAnalyzer) -> None:
        """Test emoji detection in messages."""
        emoji_messages = [
            {"text": "Love it! 😊", "is_from_me": True},
            {"text": "Haha 😂😂", "is_from_me": True},
        ]
        style = analyzer.analyze(emoji_messages)

        assert style.uses_emoji is True
        assert style.emoji_frequency > 0

    def test_detect_no_emoji_usage(self, analyzer: StyleAnalyzer) -> None:
        """Test when no emojis are used."""
        plain_messages = [
            {"text": "Sounds good", "is_from_me": True},
            {"text": "I agree", "is_from_me": True},
        ]
        style = analyzer.analyze(plain_messages)

        assert style.uses_emoji is False
        assert style.emoji_frequency == 0

    def test_detect_abbreviations(self, analyzer: StyleAnalyzer) -> None:
        """Test abbreviation detection."""
        abbrev_messages = [
            {"text": "lol thats funny", "is_from_me": True},
            {"text": "omw brb", "is_from_me": True},
            {"text": "idk tbh", "is_from_me": True},
        ]
        style = analyzer.analyze(abbrev_messages)

        assert style.uses_abbreviations is True

    def test_build_style_instructions(self, analyzer: StyleAnalyzer) -> None:
        """Test building style instructions from analyzed style."""
        messages = [
            {"text": "hey! sounds great 😊", "is_from_me": True},
            {"text": "yeah for sure", "is_from_me": True},
            {"text": "lol nice", "is_from_me": True},
        ]
        style = analyzer.analyze(messages)
        instructions = analyzer.build_style_instructions(style)

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_personality_dimensions(self, analyzer: StyleAnalyzer) -> None:
        """Test that personality dimensions are extracted."""
        messages = [
            {"text": "haha thats hilarious lol", "is_from_me": True},
            {"text": "lmao 😂", "is_from_me": True},
            {"text": "jk", "is_from_me": True},
        ]
        style = analyzer.analyze(messages)

        # Should detect humor style
        assert style.humor_style in ("playful", "expressive", "dry", "none")
        # Should detect low formality
        assert style.formality_score < 0.5
        # Should detect brief response tendency
        assert style.response_tendency == "brief"

    def test_formality_detection(self, analyzer: StyleAnalyzer) -> None:
        """Test formality score detection."""
        # Formal messages
        formal = [
            {"text": "Thank you for your help.", "is_from_me": True},
            {"text": "Please let me know.", "is_from_me": True},
        ]
        formal_style = analyzer.analyze(formal)

        # Casual messages
        casual = [
            {"text": "lol thx dude", "is_from_me": True},
            {"text": "ya gonna do it", "is_from_me": True},
        ]
        casual_style = analyzer.analyze(casual)

        assert formal_style.formality_score > casual_style.formality_score


class TestUserStyle:
    """Tests for UserStyle dataclass."""

    def test_user_style_creation(self) -> None:
        """Test creating a UserStyle instance."""
        style = UserStyle(
            avg_word_count=10.5,
            avg_char_count=50.0,
            uses_emoji=True,
            emoji_frequency=0.4,
            punctuation_style="expressive",
            capitalization="lowercase",
            common_phrases=["sounds good"],
            enthusiasm_level="high",
            uses_abbreviations=True,
            formality_score=0.3,
            humor_style="playful",
            response_tendency="brief",
        )

        assert style.avg_word_count == 10.5
        assert style.uses_emoji is True
        assert style.enthusiasm_level == "high"
        assert style.formality_score == 0.3
        assert style.humor_style == "playful"

    def test_user_style_defaults(self) -> None:
        """Test default values for UserStyle."""
        style = UserStyle()

        assert style.avg_word_count == 8.0
        assert style.uses_emoji is False
        assert style.formality_score == 0.5
        assert style.humor_style == "none"
        assert style.response_tendency == "balanced"
