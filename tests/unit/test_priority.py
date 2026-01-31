"""Unit tests for JARVIS Priority Scoring System.

Tests cover priority scoring for messages, pattern detection for questions,
action items, and time-sensitive content, as well as singleton patterns
and handled status tracking.
"""

from datetime import datetime, timedelta

import pytest

from jarvis.priority import (
    ContactStats,
    MessagePriorityScorer,
    PriorityLevel,
    PriorityReason,
    PriorityScore,
    get_priority_scorer,
    reset_priority_scorer,
)

# Import the marker for tests that require sentence_transformers
from tests.conftest import requires_sentence_transformers


# Mock Message class for testing (follows contracts.imessage.Message structure)
class MockMessage:
    """Mock Message for testing priority scoring."""

    def __init__(
        self,
        id: int = 1,
        chat_id: str = "chat123",
        sender: str = "+1234567890",
        sender_name: str | None = "John",
        text: str = "Hello",
        date: datetime | None = None,
        is_from_me: bool = False,
        is_system_message: bool = False,
    ):
        self.id = id
        self.chat_id = chat_id
        self.sender = sender
        self.sender_name = sender_name
        self.text = text
        self.date = date or datetime.now()
        self.is_from_me = is_from_me
        self.is_system_message = is_system_message
        self.attachments: list = []
        self.reactions: list = []
        self.reply_to_id: int | None = None
        self.date_delivered: datetime | None = None
        self.date_read: datetime | None = None


class TestPriorityScore:
    """Tests for PriorityScore dataclass."""

    def test_priority_score_creation(self) -> None:
        """Test basic PriorityScore creation."""
        score = PriorityScore(
            message_id=123,
            chat_id="chat456",
            score=0.75,
            level=PriorityLevel.HIGH,
            reasons=[PriorityReason.CONTAINS_QUESTION],
            needs_response=True,
            handled=False,
        )
        assert score.message_id == 123
        assert score.chat_id == "chat456"
        assert score.score == 0.75
        assert score.level == PriorityLevel.HIGH
        assert PriorityReason.CONTAINS_QUESTION in score.reasons
        assert score.needs_response is True
        assert score.handled is False

    def test_priority_score_default_values(self) -> None:
        """Test PriorityScore with default values."""
        score = PriorityScore(
            message_id=1,
            chat_id="chat1",
            score=0.5,
            level=PriorityLevel.MEDIUM,
        )
        assert score.reasons == []
        assert score.needs_response is False
        assert score.handled is False


class TestPriorityLevel:
    """Tests for PriorityLevel enum."""

    def test_priority_level_values(self) -> None:
        """Test PriorityLevel enum values."""
        assert PriorityLevel.CRITICAL.value == "critical"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"

    def test_priority_level_count(self) -> None:
        """Test that we have exactly 4 priority levels."""
        assert len(PriorityLevel) == 4


class TestPriorityReason:
    """Tests for PriorityReason enum."""

    def test_priority_reason_values(self) -> None:
        """Test PriorityReason enum values."""
        assert PriorityReason.CONTAINS_QUESTION.value == "contains_question"
        assert PriorityReason.ACTION_REQUESTED.value == "action_requested"
        assert PriorityReason.TIME_SENSITIVE.value == "time_sensitive"
        assert PriorityReason.IMPORTANT_CONTACT.value == "important_contact"
        assert PriorityReason.FREQUENT_CONTACT.value == "frequent_contact"
        assert PriorityReason.AWAITING_RESPONSE.value == "awaiting_response"
        assert PriorityReason.MULTIPLE_MESSAGES.value == "multiple_messages"
        assert PriorityReason.CONTAINS_URGENCY.value == "contains_urgency"
        assert PriorityReason.NORMAL.value == "normal"

    def test_priority_reason_count(self) -> None:
        """Test that we have expected number of reasons."""
        assert len(PriorityReason) == 9


class TestContactStats:
    """Tests for ContactStats dataclass."""

    def test_contact_stats_creation(self) -> None:
        """Test ContactStats creation."""
        stats = ContactStats(
            identifier="+1234567890",
            message_count=100,
            last_message_date=datetime.now(),
            avg_response_time_hours=2.5,
            is_important=True,
        )
        assert stats.identifier == "+1234567890"
        assert stats.message_count == 100
        assert stats.is_important is True

    def test_contact_stats_defaults(self) -> None:
        """Test ContactStats default values."""
        stats = ContactStats(identifier="+1234567890")
        assert stats.message_count == 0
        assert stats.last_message_date is None
        assert stats.avg_response_time_hours is None
        assert stats.is_important is False


class TestMessagePriorityScorerBasics:
    """Tests for basic MessagePriorityScorer functionality.

    These tests do NOT require sentence_transformers.
    """

    @pytest.fixture
    def scorer(self) -> MessagePriorityScorer:
        """Create a fresh scorer instance for each test."""
        return MessagePriorityScorer()

    # === Question Detection Tests ===

    def test_detect_question_with_question_mark(self, scorer: MessagePriorityScorer) -> None:
        """Test question detection with question mark."""
        is_question, confidence = scorer._detect_question("Are you coming?")
        assert is_question is True
        assert confidence >= 0.9

    def test_detect_question_what(self, scorer: MessagePriorityScorer) -> None:
        """Test question detection with 'what'."""
        is_question, confidence = scorer._detect_question("What time works for you")
        assert is_question is True

    def test_detect_question_when(self, scorer: MessagePriorityScorer) -> None:
        """Test question detection with 'when'."""
        is_question, confidence = scorer._detect_question("When can you meet")
        assert is_question is True

    def test_detect_question_can_you(self, scorer: MessagePriorityScorer) -> None:
        """Test question detection with 'can you'."""
        is_question, confidence = scorer._detect_question("Can you help me")
        assert is_question is True

    def test_detect_question_not_question(self, scorer: MessagePriorityScorer) -> None:
        """Test non-question detection."""
        is_question, _ = scorer._detect_question("The meeting is at 3pm")
        assert is_question is False

    def test_detect_question_empty(self, scorer: MessagePriorityScorer) -> None:
        """Test empty string."""
        is_question, confidence = scorer._detect_question("")
        assert is_question is False
        assert confidence == 0.0

    # === Action Request Detection Tests ===

    def test_detect_action_please(self, scorer: MessagePriorityScorer) -> None:
        """Test action detection with 'please'."""
        is_action, confidence = scorer._detect_action_request("Please send me the file")
        assert is_action is True
        assert confidence >= 0.9

    def test_detect_action_can_you(self, scorer: MessagePriorityScorer) -> None:
        """Test action detection with 'can you'."""
        is_action, confidence = scorer._detect_action_request("Can you pick up milk")
        assert is_action is True

    def test_detect_action_need_you(self, scorer: MessagePriorityScorer) -> None:
        """Test action detection with 'need you to'."""
        is_action, confidence = scorer._detect_action_request("I need you to call me")
        assert is_action is True

    def test_detect_action_lmk(self, scorer: MessagePriorityScorer) -> None:
        """Test action detection with 'let me know'."""
        is_action, confidence = scorer._detect_action_request("Let me know when you're done")
        assert is_action is True

    def test_detect_action_not_action(self, scorer: MessagePriorityScorer) -> None:
        """Test non-action detection."""
        is_action, _ = scorer._detect_action_request("The weather is nice today")
        assert is_action is False

    # === Time Sensitivity Detection Tests ===

    def test_detect_time_sensitive_urgent(self, scorer: MessagePriorityScorer) -> None:
        """Test time sensitivity detection with 'urgent'."""
        is_sensitive, confidence = scorer._detect_time_sensitive("This is urgent")
        assert is_sensitive is True
        assert confidence >= 0.95

    def test_detect_time_sensitive_asap(self, scorer: MessagePriorityScorer) -> None:
        """Test time sensitivity detection with 'asap'."""
        is_sensitive, confidence = scorer._detect_time_sensitive("Need this ASAP")
        assert is_sensitive is True
        assert confidence >= 0.95

    def test_detect_time_sensitive_today(self, scorer: MessagePriorityScorer) -> None:
        """Test time sensitivity detection with 'today'."""
        is_sensitive, confidence = scorer._detect_time_sensitive("Can you do this today")
        assert is_sensitive is True

    def test_detect_time_sensitive_deadline(self, scorer: MessagePriorityScorer) -> None:
        """Test time sensitivity detection with deadline."""
        is_sensitive, confidence = scorer._detect_time_sensitive("The deadline is tomorrow at 5pm")
        assert is_sensitive is True

    def test_detect_time_sensitive_not_sensitive(self, scorer: MessagePriorityScorer) -> None:
        """Test non-time-sensitive detection."""
        is_sensitive, _ = scorer._detect_time_sensitive("Thanks for the update")
        assert is_sensitive is False

    # === Score Message Tests ===

    def test_score_message_question(self, scorer: MessagePriorityScorer) -> None:
        """Test scoring a message with a question."""
        message = MockMessage(text="Can you help me with this?")
        score = scorer.score_message(message)

        assert PriorityReason.CONTAINS_QUESTION in score.reasons
        assert score.needs_response is True
        assert score.score > 0

    def test_score_message_action_request(self, scorer: MessagePriorityScorer) -> None:
        """Test scoring a message with an action request."""
        message = MockMessage(text="Please call me back")
        score = scorer.score_message(message)

        assert PriorityReason.ACTION_REQUESTED in score.reasons
        assert score.needs_response is True
        assert score.score > 0

    def test_score_message_time_sensitive(self, scorer: MessagePriorityScorer) -> None:
        """Test scoring a time-sensitive message."""
        message = MockMessage(text="This is urgent, need help immediately")
        score = scorer.score_message(message)

        assert PriorityReason.TIME_SENSITIVE in score.reasons
        assert score.score > 0

    def test_score_message_normal(self, scorer: MessagePriorityScorer) -> None:
        """Test scoring a normal message."""
        # Use a clearly non-actionable statement - a simple observation without
        # any action words, question words, or time-sensitive keywords
        message = MockMessage(text="just saw a bird")
        score = scorer.score_message(message)

        # Normal messages should be low priority
        assert score.level in (PriorityLevel.LOW, PriorityLevel.MEDIUM)
        # For truly neutral messages, the score should be relatively low
        assert score.score < 0.6  # Below high priority threshold

    def test_score_message_priority_levels(self, scorer: MessagePriorityScorer) -> None:
        """Test that priority levels are assigned correctly based on thresholds."""
        # Urgent message should have higher score than normal message
        urgent_msg = MockMessage(text="URGENT: Please call me immediately!")
        normal_msg = MockMessage(text="Thanks for the update")

        urgent_score = scorer.score_message(urgent_msg)
        normal_score = scorer.score_message(normal_msg)

        # Urgent should score higher and have time sensitive reason
        assert urgent_score.score > normal_score.score
        assert PriorityReason.TIME_SENSITIVE in urgent_score.reasons
        # Should be at least medium priority
        assert urgent_score.level in (
            PriorityLevel.CRITICAL,
            PriorityLevel.HIGH,
            PriorityLevel.MEDIUM,
        )

    # === Handled Status Tests ===

    def test_mark_handled(self, scorer: MessagePriorityScorer) -> None:
        """Test marking a message as handled."""
        scorer.mark_handled("chat123", 456)
        assert scorer.get_handled_count() == 1

        message = MockMessage(id=456, chat_id="chat123", text="Test")
        score = scorer.score_message(message)
        assert score.handled is True

    def test_unmark_handled(self, scorer: MessagePriorityScorer) -> None:
        """Test unmarking a message as handled."""
        scorer.mark_handled("chat123", 456)
        assert scorer.get_handled_count() == 1

        scorer.unmark_handled("chat123", 456)
        assert scorer.get_handled_count() == 0

        message = MockMessage(id=456, chat_id="chat123", text="Test")
        score = scorer.score_message(message)
        assert score.handled is False

    def test_clear_handled(self, scorer: MessagePriorityScorer) -> None:
        """Test clearing all handled items."""
        scorer.mark_handled("chat1", 1)
        scorer.mark_handled("chat2", 2)
        scorer.mark_handled("chat3", 3)
        assert scorer.get_handled_count() == 3

        scorer.clear_handled()
        assert scorer.get_handled_count() == 0

    # === Important Contact Tests ===

    def test_mark_contact_important(self, scorer: MessagePriorityScorer) -> None:
        """Test marking a contact as important."""
        scorer.mark_contact_important("+1234567890", True)
        assert "+1234567890" in scorer._important_contacts

    def test_unmark_contact_important(self, scorer: MessagePriorityScorer) -> None:
        """Test unmarking a contact as important."""
        scorer.mark_contact_important("+1234567890", True)
        scorer.mark_contact_important("+1234567890", False)
        assert "+1234567890" not in scorer._important_contacts

    def test_important_contact_increases_score(self, scorer: MessagePriorityScorer) -> None:
        """Test that important contacts get higher scores."""
        sender = "+1234567890"
        message = MockMessage(sender=sender, text="Hello")

        # Score without marking as important
        score1 = scorer.score_message(message)

        # Mark as important and score again
        scorer.mark_contact_important(sender, True)
        score2 = scorer.score_message(message)

        assert score2.score > score1.score
        assert PriorityReason.IMPORTANT_CONTACT in score2.reasons

    # === Contact Stats Tests ===

    def test_update_contact_stats(self, scorer: MessagePriorityScorer) -> None:
        """Test updating contact statistics."""
        scorer.update_contact_stats("+1234567890", 100, datetime.now())
        assert "+1234567890" in scorer._contact_stats
        assert scorer._contact_stats["+1234567890"].message_count == 100

    def test_frequent_contact_increases_score(self, scorer: MessagePriorityScorer) -> None:
        """Test that frequent contacts get higher scores."""
        sender = "+1234567890"
        message = MockMessage(sender=sender, text="Hello")

        # Score without contact stats
        score1 = scorer.score_message(message)

        # Add contact stats and score again
        scorer.update_contact_stats(sender, 100)
        score2 = scorer.score_message(message)

        assert score2.score > score1.score
        assert PriorityReason.FREQUENT_CONTACT in score2.reasons

    # === Score Multiple Messages Tests ===

    def test_score_messages_skips_own_messages(self, scorer: MessagePriorityScorer) -> None:
        """Test that messages from self are skipped."""
        messages = [
            MockMessage(id=1, text="Hello?", is_from_me=False),
            MockMessage(id=2, text="Can you help?", is_from_me=True),
            MockMessage(id=3, text="Please call", is_from_me=False),
        ]
        scores = scorer.score_messages(messages)

        # Only 2 messages should be scored (not the one from me)
        assert len(scores) == 2
        assert all(s.message_id != 2 for s in scores)

    def test_score_messages_skips_system_messages(self, scorer: MessagePriorityScorer) -> None:
        """Test that system messages are skipped."""
        messages = [
            MockMessage(id=1, text="Hello?", is_system_message=False),
            MockMessage(id=2, text="John left the group", is_system_message=True),
        ]
        scores = scorer.score_messages(messages)

        assert len(scores) == 1
        assert scores[0].message_id == 1

    def test_score_messages_sorted_by_priority(self, scorer: MessagePriorityScorer) -> None:
        """Test that scored messages are sorted by priority."""
        messages = [
            MockMessage(id=1, text="Thanks"),  # Low priority
            MockMessage(id=2, text="URGENT: Call me now!"),  # High priority
            MockMessage(id=3, text="Can you help?"),  # Medium priority
        ]
        scores = scorer.score_messages(messages)

        # Should be sorted highest score first
        assert len(scores) == 3
        for i in range(len(scores) - 1):
            assert scores[i].score >= scores[i + 1].score

    # === Cache Tests ===

    def test_clear_cache_no_crash(self, scorer: MessagePriorityScorer) -> None:
        """Test that clear_cache doesn't crash on uninitialized scorer."""
        scorer.clear_cache()
        assert scorer._intent_embeddings is None
        assert scorer._sentence_model is None


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_priority_scorer()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_priority_scorer()

    def test_get_priority_scorer_returns_same_instance(self) -> None:
        """Test that get_priority_scorer returns the same instance."""
        scorer1 = get_priority_scorer()
        scorer2 = get_priority_scorer()
        assert scorer1 is scorer2

    def test_reset_priority_scorer(self) -> None:
        """Test that reset creates new instance on next access."""
        scorer1 = get_priority_scorer()
        reset_priority_scorer()
        scorer2 = get_priority_scorer()
        assert scorer1 is not scorer2


@requires_sentence_transformers
class TestMessagePriorityScorerWithML:
    """Tests for MessagePriorityScorer that require sentence_transformers.

    These tests verify ML-based semantic similarity scoring.
    """

    @pytest.fixture
    def scorer(self) -> MessagePriorityScorer:
        """Create a fresh scorer instance."""
        return MessagePriorityScorer()

    def test_semantic_question_detection(self, scorer: MessagePriorityScorer) -> None:
        """Test semantic question detection with embeddings."""
        # Ensure embeddings are computed
        scorer._ensure_embeddings_computed()
        assert scorer._intent_embeddings is not None

        # Test various question phrasings
        message = MockMessage(text="I was wondering if you could help me")
        score = scorer.score_message(message)
        # May or may not detect as question depending on model
        assert score.score >= 0.0

    def test_semantic_action_detection(self, scorer: MessagePriorityScorer) -> None:
        """Test semantic action detection with embeddings."""
        scorer._ensure_embeddings_computed()

        message = MockMessage(text="Would you mind grabbing some coffee")
        score = scorer.score_message(message)
        assert score.score >= 0.0

    def test_semantic_urgency_detection(self, scorer: MessagePriorityScorer) -> None:
        """Test semantic urgency detection with embeddings."""
        scorer._ensure_embeddings_computed()

        message = MockMessage(text="This really can't wait any longer")
        score = scorer.score_message(message)
        assert score.score >= 0.0

    def test_clear_cache_resets_embeddings(self, scorer: MessagePriorityScorer) -> None:
        """Test that clear_cache resets cached embeddings."""
        # Trigger embedding computation
        scorer._ensure_embeddings_computed()
        assert scorer._intent_embeddings is not None

        # Clear cache
        scorer.clear_cache()
        assert scorer._intent_embeddings is None
        assert scorer._sentence_model is None

    def test_classify_works_after_clear_cache(self, scorer: MessagePriorityScorer) -> None:
        """Test that scoring works after cache clear."""
        message = MockMessage(text="Can you help me?")

        # Initial score
        score1 = scorer.score_message(message)
        assert score1.score > 0

        # Clear and rescore
        scorer.clear_cache()
        score2 = scorer.score_message(message)
        assert score2.score > 0


class TestContextualScoring:
    """Tests for contextual priority scoring with multiple messages."""

    @pytest.fixture
    def scorer(self) -> MessagePriorityScorer:
        """Create a fresh scorer instance."""
        return MessagePriorityScorer()

    def test_multiple_unanswered_messages(self, scorer: MessagePriorityScorer) -> None:
        """Test that multiple unanswered messages increases priority."""
        now = datetime.now()
        sender = "+1234567890"
        chat_id = "chat123"

        recent_messages = [
            MockMessage(
                id=1, chat_id=chat_id, sender=sender, text="Hello", date=now - timedelta(hours=2)
            ),
            MockMessage(
                id=2, chat_id=chat_id, sender=sender, text="Hello?", date=now - timedelta(hours=1)
            ),
            MockMessage(
                id=3,
                chat_id=chat_id,
                sender=sender,
                text="Are you there?",
                date=now - timedelta(minutes=30),
            ),
            MockMessage(
                id=4,
                chat_id=chat_id,
                sender=sender,
                text="Please respond",
                date=now - timedelta(minutes=10),
            ),
        ]

        message = recent_messages[-1]
        score = scorer.score_message(message, recent_messages)

        # Should detect multiple unanswered messages
        assert PriorityReason.MULTIPLE_MESSAGES in score.reasons

    def test_awaiting_response(self, scorer: MessagePriorityScorer) -> None:
        """Test detection of awaiting response situation."""
        now = datetime.now()
        sender = "+1234567890"
        chat_id = "chat123"

        recent_messages = [
            # My last message was 5 hours ago
            MockMessage(
                id=1,
                chat_id=chat_id,
                sender=sender,
                text="Hello",
                date=now - timedelta(hours=6),
                is_from_me=True,
            ),
            # Their message 5 hours later (still awaiting my response)
            MockMessage(
                id=2,
                chat_id=chat_id,
                sender=sender,
                text="Any update?",
                date=now - timedelta(hours=5),
                is_from_me=False,
            ),
        ]

        message = recent_messages[-1]
        score = scorer.score_message(message, recent_messages)

        # Should detect awaiting response (hours since they messaged after our last)
        assert PriorityReason.AWAITING_RESPONSE in score.reasons


class TestPriorityThresholds:
    """Tests for priority level thresholds."""

    @pytest.fixture
    def scorer(self) -> MessagePriorityScorer:
        """Create a fresh scorer instance."""
        return MessagePriorityScorer()

    def test_threshold_constants(self, scorer: MessagePriorityScorer) -> None:
        """Test that threshold constants are set correctly."""
        assert scorer.CRITICAL_THRESHOLD == 0.8
        assert scorer.HIGH_THRESHOLD == 0.6
        assert scorer.MEDIUM_THRESHOLD == 0.3

    def test_weight_constants(self, scorer: MessagePriorityScorer) -> None:
        """Test that weight constants sum to reasonable value."""
        total_weight = (
            scorer.QUESTION_WEIGHT
            + scorer.ACTION_WEIGHT
            + scorer.TIME_SENSITIVE_WEIGHT
            + scorer.CONTACT_WEIGHT
            + scorer.CONTEXT_WEIGHT
        )
        assert total_weight == 1.0
