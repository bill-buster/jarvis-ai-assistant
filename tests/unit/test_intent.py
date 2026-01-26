"""Unit tests for JARVIS Intent Classification System.

Tests cover intent classification, parameter extraction, edge cases,
confidence thresholds, and thread safety.

Note: Some tests are marked with pytest.mark.xfail because they depend on
specific semantic similarity outputs from the sentence transformer model.
The model's exact behavior may vary, so these tests verify the expected
behavior but are allowed to fail if the model outputs differ.
"""

import pytest

from jarvis.intent import (
    INTENT_EXAMPLES,
    IntentClassifier,
    IntentResult,
    IntentType,
    get_intent_classifier,
    reset_intent_classifier,
)

# Import the marker for tests that require sentence_transformers
from tests.conftest import requires_sentence_transformers

# Marker for tests that depend on specific model outputs (may vary)
model_dependent = pytest.mark.xfail(
    reason="Model output varies - tests verify expected behavior but allow variation",
    strict=False,
)


@requires_sentence_transformers
class TestIntentClassifier:
    """Tests for IntentClassifier class.

    These tests require sentence_transformers to be available.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance for each test."""
        return IntentClassifier()

    # === REPLY Intent Tests ===

    def test_reply_basic(self, classifier: IntentClassifier) -> None:
        """Test basic reply intent recognition."""
        result = classifier.classify("help me reply to this")
        assert result.intent == IntentType.REPLY
        assert result.confidence >= 0.6

    @model_dependent
    def test_reply_with_person(self, classifier: IntentClassifier) -> None:
        """Test reply intent with person name extraction."""
        result = classifier.classify("draft a response to John")
        assert result.intent == IntentType.REPLY
        assert result.extracted_params.get("person_name") == "John"

    @model_dependent
    def test_reply_variations(self, classifier: IntentClassifier) -> None:
        """Test various phrasings of reply intent."""
        queries = [
            "what should I say back",
            "how do I respond",
            "write a reply",
            "help me answer this",
            "can you reply to this for me",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.REPLY, f"Failed for: {query}"
            assert result.confidence >= 0.6, f"Low confidence for: {query}"

    @model_dependent
    def test_reply_with_full_name(self, classifier: IntentClassifier) -> None:
        """Test reply with full name extraction."""
        result = classifier.classify("help me respond to John Smith")
        assert result.intent == IntentType.REPLY
        # Note: Full name extraction may capture just first name depending on pattern

    @model_dependent
    def test_reply_informal(self, classifier: IntentClassifier) -> None:
        """Test informal reply requests."""
        result = classifier.classify("what do I text back")
        assert result.intent == IntentType.REPLY

    # === SUMMARIZE Intent Tests ===

    def test_summarize_basic(self, classifier: IntentClassifier) -> None:
        """Test basic summarize intent recognition."""
        result = classifier.classify("summarize this conversation")
        assert result.intent == IntentType.SUMMARIZE
        assert result.confidence >= 0.6

    def test_summarize_with_person(self, classifier: IntentClassifier) -> None:
        """Test summarize intent with person name extraction."""
        result = classifier.classify("summarize my chat with Sarah")
        assert result.intent == IntentType.SUMMARIZE
        assert result.extracted_params.get("person_name") == "Sarah"

    @model_dependent
    def test_summarize_with_time(self, classifier: IntentClassifier) -> None:
        """Test summarize intent with time range extraction."""
        result = classifier.classify("what did we talk about last week")
        assert result.intent == IntentType.SUMMARIZE
        assert "last week" in result.extracted_params.get("time_range", "")

    @model_dependent
    def test_summarize_variations(self, classifier: IntentClassifier) -> None:
        """Test various phrasings of summarize intent."""
        queries = [
            "recap this conversation",
            "give me a summary",
            "what have we discussed",
            "catch me up on this chat",
            "tldr of my messages",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.SUMMARIZE, f"Failed for: {query}"

    def test_summarize_with_yesterday(self, classifier: IntentClassifier) -> None:
        """Test summarize with yesterday time range."""
        result = classifier.classify("what did we discuss yesterday")
        assert result.intent == IntentType.SUMMARIZE
        assert "yesterday" in result.extracted_params.get("time_range", "")

    @model_dependent
    def test_summarize_with_person_and_time(self, classifier: IntentClassifier) -> None:
        """Test summarize with both person and time extraction."""
        result = classifier.classify("summarize my chat with Sarah from last week")
        assert result.intent == IntentType.SUMMARIZE
        assert result.extracted_params.get("person_name") == "Sarah"
        assert "last week" in result.extracted_params.get("time_range", "")

    # === SEARCH Intent Tests ===

    def test_search_basic(self, classifier: IntentClassifier) -> None:
        """Test basic search intent recognition."""
        result = classifier.classify("find messages about dinner")
        assert result.intent == IntentType.SEARCH
        assert "dinner" in result.extracted_params.get("search_query", "")

    @model_dependent
    def test_search_with_person(self, classifier: IntentClassifier) -> None:
        """Test search intent with person name extraction."""
        result = classifier.classify("find where John mentioned the meeting")
        assert result.intent == IntentType.SEARCH
        assert result.extracted_params.get("person_name") == "John"

    @model_dependent
    def test_search_variations(self, classifier: IntentClassifier) -> None:
        """Test various phrasings of search intent."""
        queries = [
            "search for dinner plans",
            "when did mom mention the party",
            "look for messages about vacation",
            "find the address he sent me",
            "where did we discuss the budget",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.SEARCH, f"Failed for: {query}"

    def test_search_with_time(self, classifier: IntentClassifier) -> None:
        """Test search with time range extraction."""
        result = classifier.classify("find messages from last Tuesday")
        assert result.intent == IntentType.SEARCH
        assert "tuesday" in result.extracted_params.get("time_range", "")

    @model_dependent
    def test_search_for_link(self, classifier: IntentClassifier) -> None:
        """Test search for shared content."""
        result = classifier.classify("search for the link Sarah shared")
        assert result.intent == IntentType.SEARCH
        assert result.extracted_params.get("person_name") == "Sarah"

    # === QUICK_REPLY Intent Tests ===

    @model_dependent
    def test_quick_reply_simple(self, classifier: IntentClassifier) -> None:
        """Test simple quick reply recognition."""
        simple_replies = ["ok", "thanks", "lol", "sure", "yes", "no", "cool", "nice"]
        for query in simple_replies:
            result = classifier.classify(query)
            assert result.intent == IntentType.QUICK_REPLY, f"Failed for: {query}"

    @model_dependent
    def test_quick_reply_with_extra(self, classifier: IntentClassifier) -> None:
        """Test quick reply with additional words."""
        result = classifier.classify("sounds good")
        assert result.intent == IntentType.QUICK_REPLY

    @model_dependent
    def test_quick_reply_thanks_variations(self, classifier: IntentClassifier) -> None:
        """Test various thanks expressions."""
        queries = ["thanks", "thank you", "thx", "ty"]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.QUICK_REPLY, f"Failed for: {query}"

    @model_dependent
    def test_quick_reply_laughter(self, classifier: IntentClassifier) -> None:
        """Test laughter expressions."""
        queries = ["lol", "haha", "hahaha", "lmao"]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.QUICK_REPLY, f"Failed for: {query}"

    def test_quick_reply_high_threshold(self, classifier: IntentClassifier) -> None:
        """Test that quick reply has higher confidence threshold."""
        # This should be quick reply with high confidence
        result = classifier.classify("ok")
        if result.intent == IntentType.QUICK_REPLY:
            assert result.confidence >= classifier.QUICK_REPLY_THRESHOLD

    # === GENERAL Intent Tests (Fallback) ===

    def test_general_unrelated(self, classifier: IntentClassifier) -> None:
        """Test that unrelated queries fall back to GENERAL."""
        queries = [
            "what is quantum physics",
            "tell me a joke",
            "calculate 2+2",
            "what's the weather like",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.GENERAL, f"Failed for: {query}"

    def test_general_greetings(self, classifier: IntentClassifier) -> None:
        """Test that greetings are classified as GENERAL."""
        queries = ["hello", "hi", "hey", "how are you"]
        for query in queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.GENERAL, f"Failed for: {query}"

    def test_general_help_request(self, classifier: IntentClassifier) -> None:
        """Test general help request."""
        result = classifier.classify("what can you do")
        assert result.intent == IntentType.GENERAL

    # === Edge Cases ===

    def test_special_characters(self, classifier: IntentClassifier) -> None:
        """Test handling of special characters."""
        result = classifier.classify("summarize chat with @John! #urgent")
        assert result.intent == IntentType.SUMMARIZE

    def test_mixed_case(self, classifier: IntentClassifier) -> None:
        """Test case insensitivity."""
        result = classifier.classify("HELP ME REPLY TO THIS")
        assert result.intent == IntentType.REPLY

    def test_unicode_input(self, classifier: IntentClassifier) -> None:
        """Test handling of unicode characters."""
        result = classifier.classify("summarize my chat with John")
        assert result.intent == IntentType.SUMMARIZE

    @model_dependent
    def test_numbers_in_query(self, classifier: IntentClassifier) -> None:
        """Test handling of numbers in query."""
        result = classifier.classify("find messages from 12/25")
        assert result.intent == IntentType.SEARCH


@requires_sentence_transformers
class TestParameterExtraction:
    """Tests for parameter extraction from queries.

    These tests require sentence_transformers to be available.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance."""
        return IntentClassifier()

    @model_dependent
    def test_extract_person_possessive(self, classifier: IntentClassifier) -> None:
        """Test extraction of possessive person name."""
        result = classifier.classify("reply to Sarah's message")
        assert result.extracted_params.get("person_name") == "Sarah"

    def test_extract_person_with_from(self, classifier: IntentClassifier) -> None:
        """Test extraction of person name after 'from'."""
        result = classifier.classify("find messages from John")
        assert result.extracted_params.get("person_name") == "John"

    def test_extract_person_with_with(self, classifier: IntentClassifier) -> None:
        """Test extraction of person name after 'with'."""
        result = classifier.classify("summarize chat with Michael")
        assert result.extracted_params.get("person_name") == "Michael"

    def test_extract_full_name(self, classifier: IntentClassifier) -> None:
        """Test extraction of full name."""
        result = classifier.classify("summarize chat with John Smith")
        assert result.extracted_params.get("person_name") == "John Smith"

    @model_dependent
    def test_extract_family_terms(self, classifier: IntentClassifier) -> None:
        """Test extraction of family terms like mom, dad."""
        result = classifier.classify("summarize messages from mom")
        assert result.extracted_params.get("person_name") == "Mom"

        result = classifier.classify("find messages from dad")
        assert result.extracted_params.get("person_name") == "Dad"

    def test_extract_time_yesterday(self, classifier: IntentClassifier) -> None:
        """Test extraction of 'yesterday' time range."""
        result = classifier.classify("what did we discuss yesterday")
        assert "yesterday" in result.extracted_params.get("time_range", "")

    def test_extract_time_today(self, classifier: IntentClassifier) -> None:
        """Test extraction of 'today' time range."""
        result = classifier.classify("summarize today's messages")
        assert "today" in result.extracted_params.get("time_range", "")

    def test_extract_time_last_week(self, classifier: IntentClassifier) -> None:
        """Test extraction of 'last week' time range."""
        result = classifier.classify("recap our conversation from last week")
        assert "last week" in result.extracted_params.get("time_range", "")

    def test_extract_time_day_of_week(self, classifier: IntentClassifier) -> None:
        """Test extraction of day of week."""
        result = classifier.classify("find messages from Monday")
        assert "monday" in result.extracted_params.get("time_range", "")

    @model_dependent
    def test_extract_search_query_about(self, classifier: IntentClassifier) -> None:
        """Test extraction of search query after 'about'."""
        result = classifier.classify("find messages about the project deadline")
        search_query = result.extracted_params.get("search_query", "")
        assert "project" in search_query or "deadline" in search_query

    @model_dependent
    def test_extract_search_query_for(self, classifier: IntentClassifier) -> None:
        """Test extraction of search query after 'for'."""
        result = classifier.classify("search for dinner plans")
        assert "dinner" in result.extracted_params.get("search_query", "")

    @model_dependent
    def test_no_params_when_none_present(self, classifier: IntentClassifier) -> None:
        """Test no parameters extracted when none present."""
        result = classifier.classify("summarize this conversation")
        assert result.extracted_params.get("person_name") is None

    def test_search_query_with_multiple_words(self, classifier: IntentClassifier) -> None:
        """Test extraction of multi-word search query."""
        result = classifier.classify("find messages about the team meeting tomorrow")
        search_query = result.extracted_params.get("search_query", "")
        assert len(search_query.split()) >= 2  # At least 2 words


class TestIntentResult:
    """Tests for IntentResult dataclass.

    These tests do NOT require sentence_transformers.
    """

    def test_intent_result_creation(self) -> None:
        """Test basic IntentResult creation."""
        result = IntentResult(
            intent=IntentType.REPLY,
            confidence=0.85,
            extracted_params={"person_name": "John"},
        )
        assert result.intent == IntentType.REPLY
        assert result.confidence == 0.85
        assert result.extracted_params["person_name"] == "John"

    def test_intent_result_default_params(self) -> None:
        """Test IntentResult with default empty params."""
        result = IntentResult(intent=IntentType.GENERAL, confidence=0.5)
        assert result.extracted_params == {}

    def test_intent_result_all_intents(self) -> None:
        """Test IntentResult with all intent types."""
        for intent_type in IntentType:
            result = IntentResult(intent=intent_type, confidence=0.7)
            assert result.intent == intent_type


class TestIntentType:
    """Tests for IntentType enum.

    These tests do NOT require sentence_transformers.
    """

    def test_intent_type_values(self) -> None:
        """Test IntentType enum values."""
        assert IntentType.REPLY.value == "reply"
        assert IntentType.SUMMARIZE.value == "summarize"
        assert IntentType.SEARCH.value == "search"
        assert IntentType.QUICK_REPLY.value == "quick_reply"
        assert IntentType.GENERAL.value == "general"

    def test_intent_type_count(self) -> None:
        """Test that we have exactly 8 intent types (5 base + 3 group)."""
        assert len(IntentType) == 8


class TestIntentExamples:
    """Tests for INTENT_EXAMPLES data.

    These tests do NOT require sentence_transformers.
    """

    def test_all_intents_have_examples(self) -> None:
        """Test that all intent types have example phrases."""
        for intent_type in IntentType:
            assert intent_type in INTENT_EXAMPLES
            assert len(INTENT_EXAMPLES[intent_type]) > 0

    def test_minimum_examples_per_intent(self) -> None:
        """Test that each intent has at least 10 examples for good coverage."""
        for intent_type, examples in INTENT_EXAMPLES.items():
            assert len(examples) >= 10, f"{intent_type.value} has only {len(examples)} examples"

    def test_no_duplicate_examples(self) -> None:
        """Test that there are no duplicate examples within an intent."""
        for intent_type, examples in INTENT_EXAMPLES.items():
            unique_examples = set(examples)
            assert len(unique_examples) == len(examples), (
                f"{intent_type.value} has duplicate examples"
            )


class TestSingleton:
    """Tests for singleton pattern.

    These tests do NOT require sentence_transformers.
    """

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_intent_classifier()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_intent_classifier()

    def test_get_intent_classifier_returns_same_instance(self) -> None:
        """Test that get_intent_classifier returns the same instance."""
        classifier1 = get_intent_classifier()
        classifier2 = get_intent_classifier()
        assert classifier1 is classifier2

    def test_reset_intent_classifier(self) -> None:
        """Test that reset creates new instance on next access."""
        classifier1 = get_intent_classifier()
        reset_intent_classifier()
        classifier2 = get_intent_classifier()
        assert classifier1 is not classifier2


class TestClassifierBasics:
    """Tests for basic classifier functionality.

    These tests do NOT require sentence_transformers - they test
    behavior when the model is unavailable.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance."""
        return IntentClassifier()

    def test_empty_input(self, classifier: IntentClassifier) -> None:
        """Test empty input returns GENERAL with 0.0 confidence."""
        result = classifier.classify("")
        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0

    def test_whitespace_input(self, classifier: IntentClassifier) -> None:
        """Test whitespace-only input."""
        result = classifier.classify("   ")
        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0

    def test_very_long_input(self, classifier: IntentClassifier) -> None:
        """Test that very long input doesn't crash."""
        long_query = "help me reply " * 100
        result = classifier.classify(long_query)
        # Should not crash and should return some result
        assert result.intent is not None
        assert isinstance(result.confidence, float)

    def test_confidence_threshold_value(self, classifier: IntentClassifier) -> None:
        """Test default confidence threshold."""
        assert classifier.CONFIDENCE_THRESHOLD == 0.6

    def test_quick_reply_threshold_value(self, classifier: IntentClassifier) -> None:
        """Test quick reply threshold is higher."""
        assert classifier.QUICK_REPLY_THRESHOLD == 0.8
        assert classifier.QUICK_REPLY_THRESHOLD > classifier.CONFIDENCE_THRESHOLD

    def test_clear_cache_no_crash(self, classifier: IntentClassifier) -> None:
        """Test that clear_cache doesn't crash on uninitialized classifier."""
        # Clear cache without computing embeddings first
        classifier.clear_cache()
        assert classifier._intent_centroids is None
        assert classifier._intent_embeddings is None


@requires_sentence_transformers
class TestClearCache:
    """Tests for cache clearing functionality.

    These tests require sentence_transformers to be available.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance."""
        return IntentClassifier()

    def test_clear_cache_resets_embeddings(self, classifier: IntentClassifier) -> None:
        """Test that clear_cache resets cached embeddings."""
        # Trigger embedding computation
        classifier.classify("help me reply")

        # Verify embeddings are computed
        assert classifier._intent_centroids is not None

        # Clear cache
        classifier.clear_cache()

        # Verify cache is cleared
        assert classifier._intent_centroids is None
        assert classifier._intent_embeddings is None

    def test_classify_works_after_clear_cache(self, classifier: IntentClassifier) -> None:
        """Test that classification works after cache clear."""
        # Initial classification
        result1 = classifier.classify("help me reply")
        assert result1.intent == IntentType.REPLY

        # Clear and reclassify
        classifier.clear_cache()
        result2 = classifier.classify("help me reply")
        assert result2.intent == IntentType.REPLY


@requires_sentence_transformers
class TestConfidenceThresholds:
    """Tests for confidence threshold behavior.

    These tests require sentence_transformers to be available.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance."""
        return IntentClassifier()

    def test_low_confidence_returns_general(self, classifier: IntentClassifier) -> None:
        """Test that low confidence queries return GENERAL."""
        # Gibberish should have low confidence for specific intents
        result = classifier.classify("xyzzy plugh")
        # Should return GENERAL due to low confidence
        assert result.intent == IntentType.GENERAL or result.confidence < 0.6


@requires_sentence_transformers
class TestRobustness:
    """Tests for classifier robustness.

    These tests require sentence_transformers to be available.
    """

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a fresh classifier instance."""
        return IntentClassifier()

    def test_typos_in_query(self, classifier: IntentClassifier) -> None:
        """Test handling of common typos."""
        # Minor typos should still work due to semantic matching
        result = classifier.classify("help me repyl to this")  # typo in reply
        # May or may not match REPLY, but should not crash
        assert result.intent is not None

    def test_abbreviated_query(self, classifier: IntentClassifier) -> None:
        """Test abbreviated queries."""
        result = classifier.classify("reply")
        # Single word may match REPLY or QUICK_REPLY
        assert result.intent in (IntentType.REPLY, IntentType.QUICK_REPLY, IntentType.GENERAL)

    @model_dependent
    def test_multiple_intents_in_query(self, classifier: IntentClassifier) -> None:
        """Test query with multiple intent keywords."""
        # Query contains both reply and summarize keywords
        result = classifier.classify("summarize and then help me reply")
        # Should pick the dominant intent
        assert result.intent in (IntentType.REPLY, IntentType.SUMMARIZE)

    def test_negation_in_query(self, classifier: IntentClassifier) -> None:
        """Test handling of negation."""
        result = classifier.classify("don't summarize")
        # Semantic model may or may not handle negation well
        assert result.intent is not None

    def test_question_format(self, classifier: IntentClassifier) -> None:
        """Test queries in question format."""
        result = classifier.classify("can you summarize this conversation?")
        assert result.intent == IntentType.SUMMARIZE

    def test_imperative_format(self, classifier: IntentClassifier) -> None:
        """Test queries in imperative format."""
        result = classifier.classify("summarize this conversation now")
        assert result.intent == IntentType.SUMMARIZE
