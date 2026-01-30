"""Tests for intent classification system."""

import pytest

from core.intent import (
    IntentClassifier,
    IntentType,
    MessageIntent,
    classify_incoming_message,
    get_intent_classifier,
)


class TestMessageIntentClassification:
    """Test incoming message intent detection."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_yes_no_question_with_question_mark(self, classifier):
        """Should detect yes/no questions with '?'."""
        result = classifier.classify_message("Are you coming tonight?")
        assert result.intent == MessageIntent.YES_NO_QUESTION
        assert result.needs_response is True

    def test_yes_no_question_with_u_variant(self, classifier):
        """Should detect yes/no questions with 'u' instead of 'you'."""
        result = classifier.classify_message("Were u in vanshs league last year?")
        assert result.intent == MessageIntent.YES_NO_QUESTION
        assert result.needs_response is True

    def test_open_question(self, classifier):
        """Should detect open questions."""
        result = classifier.classify_message("How's it looking on your end?")
        assert result.intent == MessageIntent.OPEN_QUESTION
        assert result.needs_response is True

    def test_choice_question(self, classifier):
        """Should detect choice questions."""
        result = classifier.classify_message("Italian or Mexican?")
        assert result.intent == MessageIntent.CHOICE_QUESTION
        assert result.needs_response is True

    def test_statement(self, classifier):
        """Should detect statements."""
        result = classifier.classify_message("I'll be there at 5")
        # "I'll be there at 5" could be logistics or statement - both are reasonable
        assert result.intent in (MessageIntent.STATEMENT, MessageIntent.LOGISTICS)

    def test_greeting(self, classifier):
        """Should detect greetings."""
        result = classifier.classify_message("Hey! How are you?")
        assert result.intent in (MessageIntent.GREETING, MessageIntent.OPEN_QUESTION)

    def test_thanks(self, classifier):
        """Should detect thanks."""
        result = classifier.classify_message("Thanks so much!")
        assert result.intent == MessageIntent.THANKS
        assert result.needs_response is False

    def test_farewell(self, classifier):
        """Should detect farewells."""
        result = classifier.classify_message("Talk to you later!")
        assert result.intent == MessageIntent.FAREWELL
        assert result.needs_response is False

    def test_emotional(self, classifier):
        """Should detect emotional messages."""
        result = classifier.classify_message("I'm so stressed about this deadline")
        assert result.intent == MessageIntent.EMOTIONAL

    def test_request(self, classifier):
        """Should detect requests."""
        result = classifier.classify_message("Please work with cooper and make sure we are erring on the side of caution")
        assert result.intent == MessageIntent.REQUEST

    def test_empty_message(self, classifier):
        """Should handle empty messages."""
        result = classifier.classify_message("")
        assert result.confidence == 0.0


class TestQueryIntentClassification:
    """Test user query intent detection."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_reply_intent(self, classifier):
        """Should detect reply requests."""
        result = classifier.classify_query("help me reply to this message")
        assert result.intent == IntentType.REPLY

    def test_summarize_intent(self, classifier):
        """Should detect summarize requests."""
        result = classifier.classify_query("summarize my conversation with John")
        assert result.intent == IntentType.SUMMARIZE

    def test_search_intent(self, classifier):
        """Should detect search requests."""
        result = classifier.classify_query("find messages about dinner plans")
        assert result.intent == IntentType.SEARCH

    def test_general_intent(self, classifier):
        """Should fall back to general for unclear queries."""
        result = classifier.classify_query("random gibberish xyz")
        assert result.intent == IntentType.GENERAL


class TestParameterExtraction:
    """Test parameter extraction from queries."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_extract_person_name(self, classifier):
        """Should extract person names."""
        result = classifier.classify_query("summarize my chat with John")
        assert result.extracted_params.get("person_name") == "John"

    def test_extract_time_range(self, classifier):
        """Should extract time ranges."""
        result = classifier.classify_query("find messages from yesterday")
        assert result.extracted_params.get("time_range") == "yesterday"

    def test_extract_family_names(self, classifier):
        """Should extract family terms as names."""
        result = classifier.classify_query("help me reply to mom")
        assert result.extracted_params.get("person_name") == "Mom"


class TestContextAwareness:
    """Test context-awareness detection."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_information_seeking_needs_context(self, classifier):
        """Should detect that information-seeking questions need context."""
        result = classifier.classify_message("What was the address again?")
        assert result.needs_context is True
        assert result.is_specific_question is True

    def test_when_did_needs_context(self, classifier):
        """Should detect 'when did' questions need context."""
        result = classifier.classify_message("When did we plan to meet?")
        assert result.needs_context is True

    def test_remind_me_needs_context(self, classifier):
        """Should detect 'remind me' needs context."""
        result = classifier.classify_message("Can you remind me what we agreed on?")
        assert result.needs_context is True

    def test_simple_question_no_context(self, classifier):
        """Simple questions shouldn't require special context."""
        result = classifier.classify_message("How are you?")
        # Greeting/simple questions don't need special context lookup
        assert result.needs_context is False

    def test_statement_no_context(self, classifier):
        """Statements don't need context."""
        result = classifier.classify_message("I'll be there at 5")
        assert result.needs_context is False


class TestSingletonPattern:
    """Test singleton pattern works correctly."""

    def test_get_intent_classifier_returns_same_instance(self):
        """Should return the same instance on multiple calls."""
        classifier1 = get_intent_classifier()
        classifier2 = get_intent_classifier()
        assert classifier1 is classifier2

    def test_classify_incoming_message_convenience(self):
        """Should work via convenience function."""
        result = classify_incoming_message("Are you free tonight?")
        assert result.intent == MessageIntent.YES_NO_QUESTION
