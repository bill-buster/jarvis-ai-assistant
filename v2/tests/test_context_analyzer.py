"""Tests for the context analyzer module."""

from __future__ import annotations

import pytest

from core.generation.context_analyzer import (
    ContextAnalyzer,
    ConversationContext,
    MessageIntent,
    RelationshipType,
)


class TestContextAnalyzer:
    """Tests for ContextAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ContextAnalyzer:
        """Create a ContextAnalyzer instance."""
        return ContextAnalyzer()

    @pytest.fixture
    def sample_messages(self) -> list[dict]:
        """Sample messages for testing."""
        return [
            {
                "text": "Hey, what's up?",
                "sender": "+1234567890",
                "sender_name": "John",
                "is_from_me": False,
            },
            {"text": "not much, just working", "sender": "me", "is_from_me": True},
            {
                "text": "Want to grab dinner later?",
                "sender": "+1234567890",
                "sender_name": "John",
                "is_from_me": False,
            },
        ]

    def test_detect_greeting_intent(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of greeting messages."""
        greetings = ["Hey!", "Hello there", "Hi, how are you?", "What's up?"]
        for text in greetings:
            intent = analyzer._detect_intent(text)
            assert intent == MessageIntent.GREETING, f"Expected GREETING for '{text}'"

    def test_detect_question_intent(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of question messages."""
        open_questions = [
            "What time works for you?",
            "Where should we meet?",
        ]
        for text in open_questions:
            intent = analyzer._detect_intent(text)
            assert intent == MessageIntent.OPEN_QUESTION, f"Expected OPEN_QUESTION for '{text}'"

        yes_no_questions = [
            "Can you help me with this?",
            "Do you want to come?",
        ]
        for text in yes_no_questions:
            intent = analyzer._detect_intent(text)
            assert intent == MessageIntent.YES_NO_QUESTION, f"Expected YES_NO_QUESTION for '{text}'"

    def test_detect_invitation_intent(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of invitation messages (as YES_NO_QUESTION)."""
        invitations = [
            "Want to grab dinner?",
            "Are you free for coffee?",
        ]
        for text in invitations:
            intent = analyzer._detect_intent(text)
            # Invitations are detected as YES_NO_QUESTION since they require yes/no answer
            assert intent == MessageIntent.YES_NO_QUESTION, f"Expected YES_NO_QUESTION for '{text}'"

    def test_detect_thanks_intent(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of thank you messages."""
        thanks = ["Thanks!", "Thank you so much", "Appreciate it"]
        for text in thanks:
            intent = analyzer._detect_intent(text)
            assert intent == MessageIntent.THANKS, f"Expected THANKS for '{text}'"

    def test_detect_positive_mood(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of positive mood."""
        positive_messages = [
            {"text": "This is amazing!", "is_from_me": True},
            {"text": "I love this idea", "is_from_me": True},
            {"text": "That's wonderful!", "is_from_me": True},
        ]
        mood = analyzer._detect_mood(positive_messages)
        assert mood == "positive"

    def test_detect_negative_mood(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of negative mood."""
        negative_messages = [
            {"text": "This is terrible", "is_from_me": True},
            {"text": "I hate waiting", "is_from_me": True},
            {"text": "That's awful", "is_from_me": True},
            {"text": "ugh so frustrated", "is_from_me": True},
        ]
        mood = analyzer._detect_mood(negative_messages)
        assert mood == "negative"

    def test_detect_neutral_mood(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of neutral mood."""
        neutral_messages = [
            {"text": "The meeting is at 3pm", "is_from_me": True},
            {"text": "I'll be there", "is_from_me": True},
            {"text": "Okay, got it", "is_from_me": True},
        ]
        mood = analyzer._detect_mood(neutral_messages)
        assert mood == "neutral"

    def test_detect_high_urgency(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of high urgency messages."""
        urgent_texts = [
            "ASAP please!",
            "This is urgent!!",
            "Need this immediately",
            "Emergency!",
        ]
        for text in urgent_texts:
            urgency = analyzer._detect_urgency(text)
            assert urgency == "high", f"Expected high urgency for '{text}'"

    def test_detect_normal_urgency(self, analyzer: ContextAnalyzer) -> None:
        """Test detection of normal urgency messages."""
        normal_texts = [
            "When you get a chance",
            "No rush on this",
            "Whenever works for you",
        ]
        for text in normal_texts:
            urgency = analyzer._detect_urgency(text)
            assert urgency == "normal", f"Expected normal urgency for '{text}'"

    def test_analyze_returns_conversation_context(
        self, analyzer: ContextAnalyzer, sample_messages: list[dict]
    ) -> None:
        """Test that analyze() returns a valid ConversationContext."""
        context = analyzer.analyze(sample_messages)

        assert isinstance(context, ConversationContext)
        assert context.last_message == sample_messages[-1]["text"]
        assert isinstance(context.intent, MessageIntent)
        assert isinstance(context.relationship, RelationshipType)
        assert context.mood in ("positive", "neutral", "negative")
        assert context.urgency in ("high", "normal", "low")
        assert isinstance(context.needs_response, bool)

    def test_analyze_empty_messages(self, analyzer: ContextAnalyzer) -> None:
        """Test analyze() with empty message list."""
        context = analyzer.analyze([])

        assert context.last_message == ""
        assert context.intent == MessageIntent.STATEMENT  # Default intent


class TestConversationContext:
    """Tests for ConversationContext dataclass."""

    def test_conversation_context_creation(self) -> None:
        """Test creating a ConversationContext instance."""
        context = ConversationContext(
            last_message="Hello!",
            last_sender="John",
            intent=MessageIntent.GREETING,
            relationship=RelationshipType.CLOSE_FRIEND,
            topic="general",
            mood="positive",
            urgency="normal",
            needs_response=True,
            summary="A friendly greeting",
        )

        assert context.last_message == "Hello!"
        assert context.last_sender == "John"
        assert context.intent == MessageIntent.GREETING
        assert context.relationship == RelationshipType.CLOSE_FRIEND
        assert context.mood == "positive"
        assert context.needs_response is True
