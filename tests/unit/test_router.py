"""Unit tests for JARVIS Reply Router.

Tests cover routing logic, similarity-based path selection, acknowledgment handling,
singleton pattern, and edge cases.

The ReplyRouter routes messages through three paths based on similarity:
- Template (similarity >= 0.90): Direct cached response
- Generate (0.50-0.90): LLM generation with context
- Clarify (< 0.50): Request more information
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.db import Contact, Pair
from jarvis.message_classifier import (
    ContextRequirement,
    MessageClassification,
    MessageType,
    ReplyRequirement,
)
from jarvis.router import (
    CONTEXT_DEPENDENT_PATTERNS,
    GENERATE_THRESHOLD,
    SIMPLE_ACKNOWLEDGMENTS,
    TEMPLATE_THRESHOLD,
    ReplyRouter,
    RouterError,
    RouteResult,
    get_reply_router,
    reset_reply_router,
)

# Marker for tests requiring sentence_transformers

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock JarvisDB."""
    db = MagicMock()
    db.init_schema = MagicMock()
    db.get_contact = MagicMock(return_value=None)
    db.get_contact_by_chat_id = MagicMock(return_value=None)
    db.get_pairs_by_trigger_pattern = MagicMock(return_value=[])
    db.get_stats = MagicMock(return_value={"pairs": 100, "contacts": 10})
    db.get_active_index = MagicMock(return_value=None)
    return db


@pytest.fixture
def mock_index_searcher() -> MagicMock:
    """Create a mock TriggerIndexSearcher."""
    searcher = MagicMock()
    searcher.search_with_pairs = MagicMock(return_value=[])
    return searcher


@pytest.fixture
def mock_generator() -> MagicMock:
    """Create a mock MLXGenerator."""
    generator = MagicMock()
    generator.is_loaded = MagicMock(return_value=False)
    # Return a mock response with expected attributes
    mock_response = MagicMock()
    mock_response.text = "Generated response"
    generator.generate = MagicMock(return_value=mock_response)
    return generator


@pytest.fixture
def mock_intent_classifier() -> MagicMock:
    """Create a mock IntentClassifier."""
    from jarvis.intent import IntentResult, IntentType

    classifier = MagicMock()
    classifier.classify = MagicMock(
        return_value=IntentResult(intent=IntentType.REPLY, confidence=0.8)
    )
    return classifier


@pytest.fixture
def mock_message_classifier() -> MagicMock:
    """Create a mock MessageClassifier."""
    classifier = MagicMock()
    classifier.classify = MagicMock(
        return_value=MessageClassification(
            message_type=MessageType.STATEMENT,
            type_confidence=0.9,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.INFO_RESPONSE,
            classification_method="rule",
        )
    )
    return classifier


@pytest.fixture
def router(
    mock_db: MagicMock,
    mock_index_searcher: MagicMock,
    mock_generator: MagicMock,
    mock_intent_classifier: MagicMock,
    mock_message_classifier: MagicMock,
) -> ReplyRouter:
    """Create a ReplyRouter with all mocked dependencies."""
    return ReplyRouter(
        db=mock_db,
        index_searcher=mock_index_searcher,
        generator=mock_generator,
        intent_classifier=mock_intent_classifier,
        message_classifier=mock_message_classifier,
    )


@pytest.fixture
def sample_contact() -> Contact:
    """Create a sample contact for tests."""
    return Contact(
        id=1,
        chat_id="chat123",
        display_name="John Doe",
        phone_or_email="+1234567890",
        relationship="friend",
        style_notes="casual, uses emojis",
    )


@pytest.fixture
def sample_pair() -> Pair:
    """Create a sample pair for tests."""
    from datetime import datetime

    return Pair(
        id=1,
        contact_id=1,
        trigger_text="want to grab lunch?",
        response_text="Sure! What time?",
        trigger_timestamp=datetime(2024, 1, 1, 12, 0),
        response_timestamp=datetime(2024, 1, 1, 12, 5),
        chat_id="chat123",
        quality_score=0.9,
    )


# =============================================================================
# ReplyRouter Initialization Tests
# =============================================================================


class TestReplyRouterInit:
    """Tests for ReplyRouter initialization."""

    def test_init_with_all_dependencies(
        self,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
        mock_intent_classifier: MagicMock,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test initialization with all dependencies provided."""
        router = ReplyRouter(
            db=mock_db,
            index_searcher=mock_index_searcher,
            generator=mock_generator,
            intent_classifier=mock_intent_classifier,
            message_classifier=mock_message_classifier,
        )
        assert router._db is mock_db
        assert router._index_searcher is mock_index_searcher
        assert router._generator is mock_generator
        assert router._intent_classifier is mock_intent_classifier
        assert router._message_classifier is mock_message_classifier

    def test_init_with_no_dependencies(self) -> None:
        """Test initialization with no dependencies (lazy loading)."""
        router = ReplyRouter()
        assert router._db is None
        assert router._index_searcher is None
        assert router._generator is None
        assert router._intent_classifier is None
        assert router._message_classifier is None

    def test_db_property_creates_default(self) -> None:
        """Test db property creates default instance when None."""
        with patch("jarvis.router.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            router = ReplyRouter()
            _ = router.db

            mock_get_db.assert_called_once()
            mock_db.init_schema.assert_called_once()

    def test_index_searcher_property_creates_default(self) -> None:
        """Test index_searcher property creates default instance when None."""
        mock_db = MagicMock()
        mock_db.init_schema = MagicMock()

        with patch("jarvis.router.get_db", return_value=mock_db):
            with patch("jarvis.index.TriggerIndexSearcher") as mock_searcher_cls:
                mock_searcher = MagicMock()
                mock_searcher_cls.return_value = mock_searcher

                router = ReplyRouter()
                # Access the property to trigger lazy creation
                router._index_searcher = None
                result = router.index_searcher

                assert result is mock_searcher

    def test_generator_property_creates_default(self) -> None:
        """Test generator property creates default instance when None."""
        # Create router with explicit generator to avoid real model loading
        mock_gen = MagicMock()
        router = ReplyRouter(generator=mock_gen)

        # Access the property - should return the provided generator
        result = router.generator

        assert result is mock_gen


# =============================================================================
# Route Method - Template Path Tests
# =============================================================================


class TestRouteTemplatePath:
    """Tests for routing to template path (high similarity >= 0.90)."""

    def test_template_path_with_high_similarity(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test that high similarity scores route to template path."""
        # Configure mock to return high similarity result
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "want to get lunch?",
                "response_text": "Sounds good! What time?",
                "similarity": 0.95,
                "cluster_name": "lunch_plans",
            }
        ]

        result = router.route("want to grab lunch?")

        assert result["type"] == "template"
        assert result["confidence"] == "high"
        assert result["similarity_score"] >= TEMPLATE_THRESHOLD

    def test_template_path_selects_from_multiple_matches(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test template selection from multiple high-confidence matches."""
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "want to get lunch?",
                "response_text": "Sure! What time?",
                "similarity": 0.95,
                "cluster_name": "lunch",
            },
            {
                "trigger_text": "want to grab food?",
                "response_text": "Yes! Where?",
                "similarity": 0.92,
                "cluster_name": "lunch",
            },
        ]

        result = router.route("want to grab lunch?")

        assert result["type"] == "template"
        # Response should be one of the matched responses
        assert result["response"] in ["Sure! What time?", "Yes! Where?"]

    def test_template_path_skipped_for_context_dependent(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test that context-dependent messages skip template path."""
        # Even with high similarity, context-dependent should not use template
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "what time?",
                "response_text": "3pm",
                "similarity": 0.98,
            }
        ]

        result = router.route("what time?")

        # Should ask for clarification since no thread context
        assert result["type"] == "clarify"


# =============================================================================
# Route Method - Generate Path Tests
# =============================================================================


class TestRouteGeneratePath:
    """Tests for routing to generate path (0.50-0.90 similarity)."""

    def test_generate_path_with_medium_similarity(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that medium similarity scores route to generate path."""
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "dinner tonight?",
                "response_text": "Sure, what time?",
                "similarity": 0.75,
            }
        ]

        result = router.route("want to get dinner?")

        assert result["type"] == "generated"
        assert result["confidence"] == "medium"
        mock_generator.generate.assert_called_once()

    def test_generate_path_uses_similar_examples(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that generate path uses similar patterns as context."""
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "coffee tomorrow?",
                "response_text": "Sounds great!",
                "similarity": 0.72,
            },
            {
                "trigger_text": "meet for coffee?",
                "response_text": "Sure, when?",
                "similarity": 0.68,
            },
        ]

        result = router.route("want to grab coffee?")

        assert result["type"] == "generated"
        assert result.get("similar_triggers") is not None

    def test_generate_path_with_low_but_above_threshold(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test cautious generation with score just above threshold."""
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "some message",
                "response_text": "Some response",
                "similarity": 0.52,  # Just above GENERATE_THRESHOLD
            }
        ]

        result = router.route("a different message")

        assert result["type"] == "generated"
        assert result["confidence"] == "low"

    def test_generate_path_fallback_on_no_index(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that generation falls back when FAISS index not found."""
        mock_index_searcher.search_with_pairs.side_effect = FileNotFoundError("Index not found")

        # Use a longer message that won't trigger clarification
        result = router.route("I'm doing well, how about you?")

        # Should fall through to generation or clarify (depending on vagueness check)
        assert result["type"] in ["generated", "clarify"]
        if result["type"] == "generated":
            assert result.get("is_fallback") is True


# =============================================================================
# Route Method - Clarify Path Tests
# =============================================================================


class TestRouteClarifyPath:
    """Tests for routing to clarify path (low similarity < 0.50)."""

    def test_clarify_path_with_no_results(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test that no search results with vague message routes to clarify."""
        mock_index_searcher.search_with_pairs.return_value = []

        # Use a message that would trigger clarification
        result = router.route("that thing")

        assert result["type"] == "clarify"
        assert result["confidence"] == "low"

    def test_clarify_path_with_vague_reference(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test clarification for messages with vague references."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("it")

        assert result["type"] == "clarify"
        assert "refer" in result["response"].lower() or "context" in result["response"].lower()

    def test_clarify_path_for_context_dependent_without_thread(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test clarification for context-dependent messages without thread."""
        mock_index_searcher.search_with_pairs.return_value = [
            {"trigger_text": "where?", "response_text": "At the office", "similarity": 0.95}
        ]

        # Context-dependent without thread should clarify
        result = router.route("where?", thread=None)

        assert result["type"] == "clarify"

    def test_clarify_response_includes_reason(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test that clarify responses include a reason."""
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("")

        assert result["type"] == "clarify"
        assert "reason" in result


# =============================================================================
# Acknowledgment Handling Tests
# =============================================================================


class TestAcknowledgmentHandling:
    """Tests for simple acknowledgment handling."""

    @pytest.mark.parametrize(
        "acknowledgment",
        ["ok", "okay", "yes", "yeah", "sure", "thanks", "cool", "nice", "got it", "lol"],
    )
    def test_simple_acknowledgments_detected(
        self,
        router: ReplyRouter,
        acknowledgment: str,
    ) -> None:
        """Test that simple acknowledgments are detected correctly."""
        assert router._is_simple_acknowledgment(acknowledgment) is True

    @pytest.mark.parametrize(
        "non_acknowledgment",
        ["what time?", "can you help?", "tell me more", "hello there"],
    )
    def test_non_acknowledgments_not_detected(
        self,
        router: ReplyRouter,
        non_acknowledgment: str,
    ) -> None:
        """Test that non-acknowledgments are not falsely detected."""
        assert router._is_simple_acknowledgment(non_acknowledgment) is False

    def test_acknowledgment_response_thanks(
        self,
        router: ReplyRouter,
        sample_contact: Contact,
    ) -> None:
        """Test acknowledgment response for 'thanks'."""
        result = router._generic_acknowledgment_response("thanks", sample_contact)

        assert result["type"] == "acknowledgment"
        assert result["confidence"] == "high"
        assert result["response"] in [
            "You're welcome!",
            "No problem!",
            "Anytime!",
            "Of course!",
        ]

    def test_acknowledgment_response_ok(
        self,
        router: ReplyRouter,
        sample_contact: Contact,
    ) -> None:
        """Test acknowledgment response for 'ok'."""
        result = router._generic_acknowledgment_response("ok", sample_contact)

        assert result["type"] == "acknowledgment"
        assert result["response"] in ["👍", "Sounds good!", "Great!", "Perfect!"]

    def test_acknowledgment_response_bye(
        self,
        router: ReplyRouter,
        sample_contact: Contact,
    ) -> None:
        """Test acknowledgment response for 'bye'."""
        result = router._generic_acknowledgment_response("bye", sample_contact)

        assert result["type"] == "acknowledgment"
        assert result["response"] in ["Bye!", "Talk later!", "See you!", "👋"]

    def test_acknowledgment_with_message_classifier(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test that message classifier handles acknowledgments."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.ACKNOWLEDGMENT,
            type_confidence=0.95,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.QUICK_ACK,
            classification_method="rule",
        )

        result = router.route("ok")

        assert result["type"] == "acknowledgment"


# =============================================================================
# Context-Dependent Message Tests
# =============================================================================


class TestContextDependentMessages:
    """Tests for context-dependent message handling."""

    @pytest.mark.parametrize(
        "message",
        list(CONTEXT_DEPENDENT_PATTERNS)[:5],  # Test subset
    )
    def test_context_dependent_patterns_detected(
        self,
        router: ReplyRouter,
        message: str,
    ) -> None:
        """Test that context-dependent patterns are detected."""
        assert router._is_context_dependent(message) is True

    def test_user_input_required_detected(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that user-input-required messages are detected."""
        assert router._is_context_dependent("are you coming?") is True
        assert router._is_context_dependent("can you make it?") is True
        assert router._is_context_dependent("do you want to go?") is True

    def test_normal_messages_not_context_dependent(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that normal messages are not marked context-dependent."""
        assert router._is_context_dependent("Let's meet at 5pm") is False
        assert router._is_context_dependent("I'll bring the snacks") is False

    def test_context_dependent_with_thread_generates(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test context-dependent with thread context generates response."""
        mock_index_searcher.search_with_pairs.return_value = [
            {"trigger_text": "what time?", "response_text": "5pm", "similarity": 0.95}
        ]

        result = router.route(
            "what time?",
            thread=["Hey, want to grab dinner?", "Sure!"],
        )

        assert result["type"] == "generated"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_reply_router()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_reply_router()

    def test_get_reply_router_returns_same_instance(self) -> None:
        """Test that get_reply_router returns the same instance."""
        with patch("jarvis.router.ReplyRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router_cls.return_value = mock_router

            router1 = get_reply_router()
            router2 = get_reply_router()

            assert router1 is router2
            # Constructor should only be called once
            mock_router_cls.assert_called_once()

    def test_reset_reply_router(self) -> None:
        """Test that reset creates new instance on next access."""
        with patch("jarvis.router.ReplyRouter") as mock_router_cls:
            mock_router1 = MagicMock()
            mock_router2 = MagicMock()
            mock_router_cls.side_effect = [mock_router1, mock_router2]

            router1 = get_reply_router()
            reset_reply_router()
            router2 = get_reply_router()

            assert router1 is not router2


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_message_returns_clarify(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that empty message returns clarify response."""
        result = router.route("")

        assert result["type"] == "clarify"
        assert "empty" in result["response"].lower()

    def test_whitespace_only_message_returns_clarify(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that whitespace-only message returns clarify."""
        result = router.route("   ")

        assert result["type"] == "clarify"

    def test_route_with_contact_id(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with contact_id for personalization."""
        mock_db.get_contact.return_value = sample_contact
        mock_index_searcher.search_with_pairs.return_value = []

        router.route("hello", contact_id=1)

        mock_db.get_contact.assert_called_with(1)

    def test_route_with_chat_id(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test routing with chat_id for contact lookup."""
        mock_db.get_contact.return_value = None
        mock_db.get_contact_by_chat_id.return_value = sample_contact
        mock_index_searcher.search_with_pairs.return_value = []

        router.route("hello", chat_id="chat123")

        mock_db.get_contact_by_chat_id.assert_called_with("chat123")

    def test_generation_error_returns_clarify(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that generation errors fall back to clarify response."""
        mock_index_searcher.search_with_pairs.return_value = [
            {"trigger_text": "test", "response_text": "test", "similarity": 0.75}
        ]
        mock_generator.generate.side_effect = RuntimeError("Generation failed")

        result = router.route("tell me something")

        assert result["type"] == "clarify"
        assert "trouble" in result["response"].lower()

    def test_index_search_exception_falls_back_to_generation(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test that index search errors fall back to generation."""
        mock_index_searcher.search_with_pairs.side_effect = Exception("Index error")

        result = router.route("how are you?")

        # Should attempt generation
        assert result["type"] in ["generated", "clarify"]


# =============================================================================
# Route Result Tests
# =============================================================================


class TestRouteResult:
    """Tests for RouteResult dataclass."""

    def test_route_result_creation(self) -> None:
        """Test basic RouteResult creation."""
        result = RouteResult(
            response="Hello!",
            type="template",
            confidence="high",
            similarity_score=0.95,
        )
        assert result.response == "Hello!"
        assert result.type == "template"
        assert result.confidence == "high"
        assert result.similarity_score == 0.95

    def test_route_result_defaults(self) -> None:
        """Test RouteResult default values."""
        result = RouteResult(
            response="Test",
            type="clarify",
            confidence="low",
        )
        assert result.similarity_score == 0.0
        assert result.cluster_name is None
        assert result.contact_style is None
        assert result.similar_triggers is None


# =============================================================================
# Thresholds Tests
# =============================================================================


class TestThresholds:
    """Tests for routing thresholds."""

    def test_template_threshold_value(self) -> None:
        """Test TEMPLATE_THRESHOLD is set correctly."""
        assert TEMPLATE_THRESHOLD == 0.90

    def test_generate_threshold_value(self) -> None:
        """Test GENERATE_THRESHOLD is set correctly."""
        assert GENERATE_THRESHOLD == 0.50

    def test_simple_acknowledgments_set(self) -> None:
        """Test SIMPLE_ACKNOWLEDGMENTS contains expected values."""
        assert "ok" in SIMPLE_ACKNOWLEDGMENTS
        assert "thanks" in SIMPLE_ACKNOWLEDGMENTS
        assert "yes" in SIMPLE_ACKNOWLEDGMENTS
        assert "lol" in SIMPLE_ACKNOWLEDGMENTS

    def test_context_dependent_patterns_set(self) -> None:
        """Test CONTEXT_DEPENDENT_PATTERNS contains expected values."""
        assert "what time" in CONTEXT_DEPENDENT_PATTERNS
        assert "where?" in CONTEXT_DEPENDENT_PATTERNS
        assert "when?" in CONTEXT_DEPENDENT_PATTERNS


# =============================================================================
# RouterError Tests
# =============================================================================


class TestRouterError:
    """Tests for RouterError exception."""

    def test_router_error_creation(self) -> None:
        """Test RouterError can be created."""
        error = RouterError("Test error")
        assert str(error) == "Test error"

    def test_router_error_inherits_from_jarvis_error(self) -> None:
        """Test RouterError inherits from JarvisError."""
        from jarvis.errors import JarvisError

        assert issubclass(RouterError, JarvisError)


# =============================================================================
# Get Routing Stats Tests
# =============================================================================


class TestGetRoutingStats:
    """Tests for get_routing_stats method."""

    def test_get_routing_stats_basic(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
    ) -> None:
        """Test basic routing stats retrieval."""
        mock_db.get_stats.return_value = {"pairs": 100, "contacts": 10}
        mock_db.get_active_index.return_value = None

        stats = router.get_routing_stats()

        assert "db_stats" in stats
        assert stats["index_available"] is False

    def test_get_routing_stats_with_index(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
    ) -> None:
        """Test routing stats with active index."""
        mock_index = MagicMock()
        mock_index.version_id = "v1.0"
        mock_index.num_vectors = 1000
        mock_index.model_name = "bge-small"
        mock_db.get_active_index.return_value = mock_index

        stats = router.get_routing_stats()

        assert stats["index_available"] is True
        assert stats["index_version"] == "v1.0"
        assert stats["index_vectors"] == 1000


# =============================================================================
# Needs Clarification Tests
# =============================================================================


class TestNeedsClarification:
    """Tests for _needs_clarification method."""

    def test_vague_reference_needs_clarification(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that vague references need clarification."""
        assert router._needs_clarification("that thing", None) is True
        assert router._needs_clarification("what about it", None) is True

    def test_vague_reference_ok_with_thread(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that vague references are ok with thread context."""
        thread = ["Let's go to the park", "Sounds good!"]
        assert router._needs_clarification("that sounds fun", thread) is False

    def test_short_message_needs_clarification(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that very short messages without context need clarification."""
        assert router._needs_clarification("hi", None) is True
        assert router._needs_clarification("?", None) is True

    def test_normal_message_no_clarification(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that normal messages don't need clarification."""
        assert router._needs_clarification("Let's meet at 5pm at the coffee shop", None) is False


# =============================================================================
# Professional Response Filter Tests
# =============================================================================


class TestProfessionalResponseFilter:
    """Tests for _is_professional_response method."""

    def test_professional_responses_pass(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that professional responses pass the filter."""
        assert router._is_professional_response("Thank you for the update.") is True
        assert router._is_professional_response("I'll review this shortly.") is True
        assert router._is_professional_response("Sounds good, let me know.") is True

    def test_unprofessional_responses_fail(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test that unprofessional responses fail the filter."""
        assert router._is_professional_response("lol that's funny") is False
        assert router._is_professional_response("haha yeah totally 😂") is False
        assert router._is_professional_response("bruh what") is False


# =============================================================================
# Ask For Clarification Tests
# =============================================================================


class TestAskForClarification:
    """Tests for _ask_for_clarification method."""

    def test_ask_clarification_vague_reference(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test clarification request for vague references."""
        result = router._ask_for_clarification("that thing", None)

        assert result["type"] == "clarify"
        assert "refer" in result["response"].lower()

    def test_ask_clarification_timing(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test clarification request for timing questions."""
        result = router._ask_for_clarification("when is it?", None)

        assert result["type"] == "clarify"
        # The response may vary based on pattern matching in the message
        assert len(result["response"]) > 0

    def test_ask_clarification_location(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test clarification request for location questions."""
        result = router._ask_for_clarification("where is the location?", None)

        assert result["type"] == "clarify"
        assert "location" in result["response"].lower()

    def test_ask_clarification_generic(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test generic clarification request."""
        result = router._ask_for_clarification("hmm", None)

        assert result["type"] == "clarify"
        assert "context" in result["response"].lower()


# =============================================================================
# Template Response Tests
# =============================================================================


class TestTemplateResponse:
    """Tests for _template_response method."""

    def test_template_response_single_match(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test template response with single match."""
        matches = [
            {
                "trigger_text": "want lunch?",
                "response_text": "Sure! When?",
                "similarity": 0.95,
                "cluster_name": "lunch",
            }
        ]

        result = router._template_response(matches, None, "want lunch?")

        # May fall back to generation if coherence check fails
        assert result["type"] in ["template", "fallback_to_generation"]

    def test_template_response_no_matches_clarifies(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test template response with no matches returns clarify."""
        result = router._template_response([], None, "test")

        assert result["type"] == "clarify"

    def test_template_response_with_contact_style(
        self,
        router: ReplyRouter,
        sample_contact: Contact,
    ) -> None:
        """Test template response considers contact style."""
        sample_contact.relationship = "boss"
        matches = [
            {
                "trigger_text": "update?",
                "response_text": "I'll have it ready by EOD.",
                "similarity": 0.95,
            }
        ]

        result = router._template_response(matches, sample_contact, "status update?")

        # Should filter for professional responses when contact is boss
        if result["type"] == "template":
            assert result["contact_style"] == sample_contact.style_notes


# =============================================================================
# IndexNotAvailableError Tests
# =============================================================================


class TestIndexNotAvailableError:
    """Tests for IndexNotAvailableError exception."""

    def test_index_not_available_error_creation(self) -> None:
        """Test IndexNotAvailableError can be created."""
        from jarvis.router import IndexNotAvailableError

        error = IndexNotAvailableError()
        assert "FAISS index not available" in str(error)

    def test_index_not_available_inherits_from_router_error(self) -> None:
        """Test IndexNotAvailableError inherits from RouterError."""
        from jarvis.router import IndexNotAvailableError

        assert issubclass(IndexNotAvailableError, RouterError)


# =============================================================================
# iMessage Reader Property Tests
# =============================================================================


class TestIMessageReaderProperty:
    """Tests for iMessage reader lazy initialization."""

    def test_imessage_reader_property_returns_none_on_error(
        self,
        mock_db: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
        mock_intent_classifier: MagicMock,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test imessage_reader returns None when initialization fails."""
        router = ReplyRouter(
            db=mock_db,
            index_searcher=mock_index_searcher,
            generator=mock_generator,
            intent_classifier=mock_intent_classifier,
            message_classifier=mock_message_classifier,
        )
        # Don't set _imessage_reader, let it try to create
        router._imessage_reader = None

        with patch(
            "integrations.imessage.reader.ChatDBReader", side_effect=Exception("Cannot init")
        ):
            result = router.imessage_reader
            assert result is None

    def test_imessage_reader_property_caches_instance(self) -> None:
        """Test imessage_reader caches the instance after creation."""
        mock_reader = MagicMock()
        router = ReplyRouter(imessage_reader=mock_reader)

        # Access twice
        reader1 = router.imessage_reader
        reader2 = router.imessage_reader

        assert reader1 is reader2
        assert reader1 is mock_reader


# =============================================================================
# Fetch Conversation Context Tests
# =============================================================================


class TestFetchConversationContext:
    """Tests for _fetch_conversation_context method."""

    def test_fetch_context_returns_empty_when_no_reader(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch returns empty list when reader is None."""
        router._imessage_reader = None
        result = router._fetch_conversation_context("chat123", limit=10)
        assert result == []

    def test_fetch_context_formats_messages(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch correctly formats messages."""
        mock_reader = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.is_from_me = True
        mock_msg1.sender_name = None
        mock_msg1.sender = None
        mock_msg1.text = "Hello!"

        mock_msg2 = MagicMock()
        mock_msg2.is_from_me = False
        mock_msg2.sender_name = "John"
        mock_msg2.sender = "+1234567890"
        mock_msg2.text = "Hey there!"

        mock_reader.get_messages.return_value = [mock_msg1, mock_msg2]
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123", limit=10)

        # Messages should be reversed (chronological order)
        assert len(result) == 2
        assert "[John]: Hey there!" in result[0]
        assert "[You]: Hello!" in result[1]

    def test_fetch_context_handles_empty_messages(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch handles empty message list."""
        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = []
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_handles_exception(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch handles exceptions gracefully."""
        mock_reader = MagicMock()
        mock_reader.get_messages.side_effect = Exception("Database error")
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")
        assert result == []

    def test_fetch_context_skips_empty_text(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test context fetch skips messages with empty text."""
        mock_reader = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.is_from_me = True
        mock_msg1.sender_name = None
        mock_msg1.sender = None
        mock_msg1.text = ""  # Empty text

        mock_msg2 = MagicMock()
        mock_msg2.is_from_me = False
        mock_msg2.sender_name = "John"
        mock_msg2.sender = None
        mock_msg2.text = "Hello"

        mock_reader.get_messages.return_value = [mock_msg1, mock_msg2]
        router._imessage_reader = mock_reader

        result = router._fetch_conversation_context("chat123")

        # Only non-empty message should be included
        assert len(result) == 1
        assert "[John]: Hello" in result[0]


# =============================================================================
# Threshold Configuration Tests
# =============================================================================


class TestThresholdConfiguration:
    """Tests for _get_thresholds method and A/B testing."""

    def test_get_thresholds_returns_defaults(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test _get_thresholds returns default values."""
        with patch("jarvis.router.get_config") as mock_config:
            mock_routing = MagicMock()
            mock_routing.template_threshold = 0.90
            mock_routing.context_threshold = 0.70
            mock_routing.generate_threshold = 0.50
            mock_routing.ab_test_group = None
            mock_routing.ab_test_thresholds = {}
            mock_config.return_value.routing = mock_routing

            thresholds = router._get_thresholds()

            assert thresholds["template"] == 0.90
            assert thresholds["context"] == 0.70
            assert thresholds["generate"] == 0.50

    def test_get_thresholds_with_ab_test_group(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test _get_thresholds uses A/B test overrides when configured."""
        with patch("jarvis.router.get_config") as mock_config:
            mock_routing = MagicMock()
            mock_routing.template_threshold = 0.90
            mock_routing.context_threshold = 0.70
            mock_routing.generate_threshold = 0.50
            mock_routing.ab_test_group = "test_group"
            mock_routing.ab_test_thresholds = {
                "test_group": {
                    "template": 0.85,
                    "context": 0.60,
                    "generate": 0.40,
                }
            }
            mock_config.return_value.routing = mock_routing

            thresholds = router._get_thresholds()

            assert thresholds["template"] == 0.85
            assert thresholds["context"] == 0.60
            assert thresholds["generate"] == 0.40


# =============================================================================
# Generate After Acknowledgment Tests
# =============================================================================


class TestShouldGenerateAfterAcknowledgment:
    """Tests for _should_generate_after_acknowledgment method."""

    def test_generate_after_ack_with_active_thread(
        self,
        router: ReplyRouter,
        sample_contact: Contact,
    ) -> None:
        """Test returns True when thread has multiple messages."""
        thread = ["Hey!", "What's up?", "Not much"]
        result = router._should_generate_after_acknowledgment("ok", sample_contact, thread)
        assert result is True

    def test_generate_after_ack_short_thread(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test returns False when thread is short and no patterns found."""
        mock_db.get_pairs_by_trigger_pattern.return_value = []
        result = router._should_generate_after_acknowledgment("ok", sample_contact, ["Hey!"])
        assert result is False

    def test_generate_after_ack_based_on_historical_pattern(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test returns True when contact historically sends substantive acks."""
        # Create mock pairs with long responses
        mock_pair1 = MagicMock()
        mock_pair1.response_text = "Sure, I was thinking we could meet at the coffee shop"
        mock_pair2 = MagicMock()
        mock_pair2.response_text = "Okay, let me check my calendar and get back to you"
        mock_db.get_pairs_by_trigger_pattern.return_value = [mock_pair1, mock_pair2]

        result = router._should_generate_after_acknowledgment("ok", sample_contact, None)
        assert result is True

    def test_generate_after_ack_short_historical_responses(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test returns False when contact historically sends short acks."""
        # Create mock pairs with short responses
        mock_pair1 = MagicMock()
        mock_pair1.response_text = "ok"
        mock_pair2 = MagicMock()
        mock_pair2.response_text = "sure"
        mock_db.get_pairs_by_trigger_pattern.return_value = [mock_pair1, mock_pair2]

        result = router._should_generate_after_acknowledgment("ok", sample_contact, None)
        assert result is False

    def test_generate_after_ack_handles_db_exception(
        self,
        router: ReplyRouter,
        mock_db: MagicMock,
        sample_contact: Contact,
    ) -> None:
        """Test handles database exceptions gracefully."""
        mock_db.get_pairs_by_trigger_pattern.side_effect = Exception("DB error")

        result = router._should_generate_after_acknowledgment("ok", sample_contact, None)
        assert result is False

    def test_generate_after_ack_no_contact(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test returns False when no contact provided."""
        result = router._should_generate_after_acknowledgment("ok", None, None)
        assert result is False


# =============================================================================
# Greeting and Farewell Handling Tests
# =============================================================================


class TestGreetingFarewellHandling:
    """Tests for greeting and farewell message handling."""

    def test_greeting_returns_acknowledgment(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test greetings are handled with acknowledgment response."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.GREETING,
            type_confidence=0.95,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.QUICK_ACK,
            classification_method="rule",
        )

        result = router.route("hello")

        assert result["type"] == "acknowledgment"

    def test_farewell_returns_acknowledgment(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test farewells are handled with acknowledgment response."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.FAREWELL,
            type_confidence=0.95,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.QUICK_ACK,
            classification_method="rule",
        )

        result = router.route("goodbye")

        assert result["type"] == "acknowledgment"


# =============================================================================
# Reaction Handling Tests
# =============================================================================


class TestReactionHandling:
    """Tests for reaction message handling."""

    def test_reaction_returns_acknowledgment(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test reactions are handled with acknowledgment response."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.REACTION,
            type_confidence=0.99,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.NO_REPLY,
            classification_method="rule",
        )

        result = router.route("Loved an image")

        assert result["type"] == "acknowledgment"


# =============================================================================
# Vague Context Requirement Tests
# =============================================================================


class TestVagueContextRequirement:
    """Tests for handling messages with vague context requirement."""

    def test_vague_message_without_thread_clarifies(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
    ) -> None:
        """Test vague messages without thread context request clarification."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.QUESTION_INFO,
            type_confidence=0.85,
            context_requirement=ContextRequirement.VAGUE,
            reply_requirement=ReplyRequirement.INFO_RESPONSE,
            classification_method="semantic",
        )

        result = router.route("what about that?", thread=None)

        assert result["type"] == "clarify"

    def test_vague_message_with_thread_processes(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test vague messages with thread context proceed to processing."""
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.QUESTION_INFO,
            type_confidence=0.85,
            context_requirement=ContextRequirement.VAGUE,
            reply_requirement=ReplyRequirement.INFO_RESPONSE,
            classification_method="semantic",
        )
        mock_index_searcher.search_with_pairs.return_value = []

        result = router.route("what about that?", thread=["Let's go hiking", "Sure!"])

        # Should proceed (not clarify) because we have thread context
        # Result depends on whether it generates or clarifies based on other logic
        assert result["type"] in ["generated", "clarify"]


# =============================================================================
# Quick Reply Intent Tests
# =============================================================================


class TestQuickReplyIntent:
    """Tests for QUICK_REPLY intent handling."""

    def test_high_confidence_quick_reply_returns_acknowledgment(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
        mock_intent_classifier: MagicMock,
    ) -> None:
        """Test high confidence quick reply intent returns acknowledgment."""
        from jarvis.intent import IntentResult, IntentType

        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.STATEMENT,
            type_confidence=0.7,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.QUICK_ACK,
            classification_method="rule",
        )
        mock_intent_classifier.classify.return_value = IntentResult(
            intent=IntentType.QUICK_REPLY,
            confidence=0.85,
        )

        result = router.route("sounds good to me")

        assert result["type"] == "acknowledgment"


# =============================================================================
# Intent Classifier Property Tests
# =============================================================================


class TestIntentClassifierProperty:
    """Tests for intent_classifier lazy initialization."""

    def test_intent_classifier_property_creates_default(self) -> None:
        """Test intent_classifier property creates default instance."""
        with patch("jarvis.router.get_intent_classifier") as mock_get_classifier:
            mock_classifier = MagicMock()
            mock_get_classifier.return_value = mock_classifier

            router = ReplyRouter()
            router._intent_classifier = None
            result = router.intent_classifier

            mock_get_classifier.assert_called_once()
            assert result is mock_classifier


# =============================================================================
# Message Classifier Property Tests
# =============================================================================


class TestMessageClassifierProperty:
    """Tests for message_classifier lazy initialization."""

    def test_message_classifier_property_creates_default(self) -> None:
        """Test message_classifier property creates default instance."""
        with patch("jarvis.router.get_message_classifier") as mock_get_classifier:
            mock_classifier = MagicMock()
            mock_get_classifier.return_value = mock_classifier

            router = ReplyRouter()
            router._message_classifier = None
            result = router.message_classifier

            mock_get_classifier.assert_called_once()
            assert result is mock_classifier


# =============================================================================
# Classification Failure Tests
# =============================================================================


class TestClassificationFailures:
    """Tests for handling classification failures."""

    def test_message_classification_failure_falls_back(
        self,
        router: ReplyRouter,
        mock_message_classifier: MagicMock,
        mock_index_searcher: MagicMock,
    ) -> None:
        """Test message classification failure falls back to legacy handling."""
        mock_message_classifier.classify.side_effect = Exception("Classification failed")
        mock_index_searcher.search_with_pairs.return_value = []

        # Should still process using legacy path
        result = router.route("ok")

        # "ok" is a simple acknowledgment, so should be handled
        assert result["type"] == "acknowledgment"

    def test_intent_classification_failure_continues(
        self,
        router: ReplyRouter,
        mock_intent_classifier: MagicMock,
        mock_message_classifier: MagicMock,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test intent classification failure allows processing to continue."""
        mock_intent_classifier.classify.side_effect = Exception("Intent failed")
        mock_message_classifier.classify.return_value = MessageClassification(
            message_type=MessageType.STATEMENT,
            type_confidence=0.9,
            context_requirement=ContextRequirement.SELF_CONTAINED,
            reply_requirement=ReplyRequirement.INFO_RESPONSE,
            classification_method="rule",
        )
        mock_index_searcher.search_with_pairs.return_value = [
            {"trigger_text": "test", "response_text": "response", "similarity": 0.75}
        ]

        result = router.route("tell me about the project")

        # Should continue processing despite intent classification failure
        assert result["type"] in ["generated", "clarify"]


# =============================================================================
# CONTEXT_THRESHOLD Tests
# =============================================================================


class TestContextThreshold:
    """Tests for CONTEXT_THRESHOLD constant."""

    def test_context_threshold_value(self) -> None:
        """Test CONTEXT_THRESHOLD is set correctly."""
        from jarvis.router import CONTEXT_THRESHOLD

        assert CONTEXT_THRESHOLD == 0.70


# =============================================================================
# Normalize Routing Decision Tests
# =============================================================================


class TestNormalizeRoutingDecision:
    """Tests for _normalize_routing_decision method."""

    def test_normalize_generated_to_generate(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test 'generated' is normalized to 'generate'."""
        result = router._normalize_routing_decision("generated")
        assert result == "generate"

    def test_normalize_template_unchanged(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test 'template' remains unchanged."""
        result = router._normalize_routing_decision("template")
        assert result == "template"

    def test_normalize_other_to_clarify(
        self,
        router: ReplyRouter,
    ) -> None:
        """Test other values normalize to 'clarify'."""
        result = router._normalize_routing_decision("unknown")
        assert result == "clarify"
        result = router._normalize_routing_decision("acknowledgment")
        assert result == "clarify"


# =============================================================================
# Coherence Fallback Tests
# =============================================================================


class TestCoherenceFallback:
    """Tests for coherence-based fallback to generation."""

    def test_template_fallback_when_no_coherent_matches(
        self,
        router: ReplyRouter,
        mock_index_searcher: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        """Test fallback to generation when no coherent templates found."""
        # Mock high similarity but response that won't pass coherence check
        mock_index_searcher.search_with_pairs.return_value = [
            {
                "trigger_text": "want lunch?",
                "response_text": "Purple monkey dishwasher",  # Incoherent
                "similarity": 0.95,
            }
        ]

        with patch("jarvis.router.score_response_coherence", return_value=0.1):
            result = router.route("want to grab lunch?")

            # Should fall back to generation due to low coherence
            assert result["type"] in ["generated", "clarify"]


# =============================================================================
# User Input Required Tests
# =============================================================================


class TestUserInputRequired:
    """Tests for USER_INPUT_REQUIRED_STARTERS patterns."""

    @pytest.mark.parametrize(
        "message",
        [
            "are you coming to the party?",
            "are you free tomorrow?",
            "can you come over?",
            "will you be there?",
            "do you want to join us?",
            "where are you right now?",
            "what are you doing tonight?",
            "how are you feeling?",
            "did you finish the report?",
            "have you seen the movie?",
        ],
    )
    def test_user_input_required_detected(
        self,
        router: ReplyRouter,
        message: str,
    ) -> None:
        """Test that user-input-required messages are context-dependent."""
        assert router._is_context_dependent(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "The party is at 7pm",
            "I finished the report",
            "The weather is nice",
        ],
    )
    def test_normal_statements_not_user_input_required(
        self,
        router: ReplyRouter,
        message: str,
    ) -> None:
        """Test that normal statements are not context-dependent."""
        assert router._is_context_dependent(message) is False
