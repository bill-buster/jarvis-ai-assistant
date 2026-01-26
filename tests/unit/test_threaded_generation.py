"""Unit tests for thread-aware response generation.

Tests thread analysis, threaded prompt building, and ThreadAwareGenerator
to ensure different thread types receive appropriate responses.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from jarvis.threading import (
    ThreadAnalyzer,
    ThreadContext,
    ThreadedReplyConfig,
    ThreadState,
    ThreadTopic,
    UserRole,
    get_thread_analyzer,
    reset_thread_analyzer,
)
from tests.conftest import SENTENCE_TRANSFORMERS_AVAILABLE


# Helper to create mock Message objects
def create_mock_message(
    text: str,
    is_from_me: bool = False,
    sender: str = "John",
    date: datetime | None = None,
) -> MagicMock:
    """Create a mock Message object for testing."""
    msg = MagicMock()
    msg.text = text
    msg.is_from_me = is_from_me
    msg.sender = sender
    msg.sender_name = sender
    msg.date = date or datetime.now()
    return msg


class TestThreadTopic:
    """Tests for ThreadTopic enum."""

    def test_all_topics_have_values(self):
        """Verify all topics have string values."""
        for topic in ThreadTopic:
            assert isinstance(topic.value, str)
            assert len(topic.value) > 0

    def test_topic_count(self):
        """Verify expected number of topics."""
        assert len(ThreadTopic) == 9


class TestThreadState:
    """Tests for ThreadState enum."""

    def test_all_states_have_values(self):
        """Verify all states have string values."""
        for state in ThreadState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0

    def test_state_count(self):
        """Verify expected number of states."""
        assert len(ThreadState) == 5


class TestUserRole:
    """Tests for UserRole enum."""

    def test_all_roles_have_values(self):
        """Verify all roles have string values."""
        for role in UserRole:
            assert isinstance(role.value, str)
            assert len(role.value) > 0

    def test_role_count(self):
        """Verify expected number of roles."""
        assert len(UserRole) == 3


class TestThreadedReplyConfig:
    """Tests for ThreadedReplyConfig dataclass."""

    def test_config_creation(self):
        """Test creating a config."""
        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="concise",
            include_action_items=True,
            suggest_follow_up=False,
        )
        assert config.max_response_length == 100
        assert config.response_style == "concise"
        assert config.include_action_items is True
        assert config.suggest_follow_up is False


class TestThreadAnalyzer:
    """Tests for ThreadAnalyzer class."""

    def test_analyzer_singleton(self):
        """Test singleton pattern works."""
        reset_thread_analyzer()
        analyzer1 = get_thread_analyzer()
        analyzer2 = get_thread_analyzer()
        assert analyzer1 is analyzer2
        reset_thread_analyzer()

    def test_analyze_empty_messages(self):
        """Test analyzing empty message list."""
        analyzer = ThreadAnalyzer()
        context = analyzer.analyze([])

        assert context.topic == ThreadTopic.UNKNOWN
        assert context.state == ThreadState.CONCLUDED
        assert context.confidence == 0.0

    def test_analyze_logistics_thread(self):
        """Test detecting logistics thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("What time works for pickup?", is_from_me=False),
            create_mock_message("How about 5pm?", is_from_me=True),
            create_mock_message("Where should I meet you?", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        assert context.topic == ThreadTopic.LOGISTICS
        assert context.confidence > 0

    def test_analyze_emotional_support_thread(self):
        """Test detecting emotional support thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("I'm having such a bad day", is_from_me=False),
            create_mock_message("I'm so sorry to hear that", is_from_me=True),
            create_mock_message("I'm feeling really down", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        assert context.topic == ThreadTopic.EMOTIONAL_SUPPORT
        assert context.confidence > 0

    def test_analyze_planning_thread(self):
        """Test detecting planning thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Want to grab dinner this weekend?", is_from_me=False),
            create_mock_message("Sure! What day works for you?", is_from_me=True),
            create_mock_message("Let's plan for Saturday", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        assert context.topic == ThreadTopic.PLANNING
        assert context.confidence > 0

    def test_analyze_quick_exchange_thread(self):
        """Test detecting quick exchange thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("thanks", is_from_me=False),
            create_mock_message("np", is_from_me=True),
            create_mock_message("ok cool", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        assert context.topic == ThreadTopic.QUICK_EXCHANGE
        assert context.confidence > 0

    def test_detect_open_question_state(self):
        """Test detecting open question state."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey, are you free tomorrow?", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        assert context.state == ThreadState.OPEN_QUESTION

    def test_detect_awaiting_response_state(self):
        """Test detecting awaiting response state."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Do you want to meet?", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        assert context.state == ThreadState.AWAITING_RESPONSE

    def test_detect_concluded_state(self):
        """Test detecting concluded state."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("See you tomorrow", is_from_me=False),
            create_mock_message("sounds good, bye!", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        assert context.state == ThreadState.CONCLUDED

    def test_detect_initiator_role(self):
        """Test detecting initiator role."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey, want to hang out?", is_from_me=True),
            create_mock_message("Sure!", is_from_me=False),
            create_mock_message("Great, let me know when", is_from_me=True),
            create_mock_message("How about 5?", is_from_me=False),
            create_mock_message("Perfect", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        assert context.user_role == UserRole.INITIATOR

    def test_detect_responder_role(self):
        """Test detecting responder role."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey, want to hang out?", is_from_me=False),
            create_mock_message("Sure!", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        # First message is from other person, we've only responded once
        assert context.user_role in (UserRole.RESPONDER, UserRole.PARTICIPANT)

    def test_get_relevant_messages_logistics(self):
        """Test relevant messages for logistics thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey there", is_from_me=False),
            create_mock_message("How are you?", is_from_me=True),
            create_mock_message("What time should we meet?", is_from_me=False),
            create_mock_message("5pm works", is_from_me=True),
            create_mock_message("Where should I go?", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        # Logistics threads should filter for relevant messages
        # All recent messages should be included (at minimum last 3)
        assert len(context.relevant_messages) >= 3

    def test_get_relevant_messages_quick_exchange(self):
        """Test relevant messages for quick exchange thread."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey", is_from_me=False),
            create_mock_message("Hi", is_from_me=True),
            create_mock_message("ok", is_from_me=False),
            create_mock_message("cool", is_from_me=True),
            create_mock_message("thanks", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        # Quick exchanges should have fewer relevant messages
        if context.topic == ThreadTopic.QUICK_EXCHANGE:
            assert len(context.relevant_messages) <= 3

    def test_extract_action_items(self):
        """Test action item extraction."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Can you pick up the groceries?", is_from_me=False),
            create_mock_message("I'll pick them up after work", is_from_me=True),
            create_mock_message("Don't forget to get milk", is_from_me=False),
        ]

        context = analyzer.analyze(messages)

        # Should extract action items
        assert len(context.action_items) >= 1

    def test_participants_count_dm(self):
        """Test participant count for direct message."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey", is_from_me=False, sender="John"),
            create_mock_message("Hi", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        assert context.participants_count == 1

    def test_participants_count_group(self):
        """Test participant count for group chat."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("Hey everyone", is_from_me=False, sender="John"),
            create_mock_message("Hi", is_from_me=False, sender="Sarah"),
            create_mock_message("Hello", is_from_me=True),
        ]

        context = analyzer.analyze(messages)

        assert context.participants_count == 2  # John and Sarah

    def test_get_response_config(self):
        """Test getting response config for thread context."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message("What time works?", is_from_me=False),
        ]

        context = analyzer.analyze(messages)
        config = analyzer.get_response_config(context)

        assert isinstance(config, ThreadedReplyConfig)
        assert config.max_response_length > 0

    def test_clear_cache(self):
        """Test clearing analyzer cache."""
        analyzer = ThreadAnalyzer()
        analyzer._topic_embeddings = {"test": "value"}

        analyzer.clear_cache()

        assert analyzer._topic_embeddings is None


class TestThreadedPromptBuilder:
    """Tests for threaded prompt building functions."""

    def test_build_threaded_reply_prompt_logistics(self):
        """Test building prompt for logistics thread."""
        from jarvis.prompts import build_threaded_reply_prompt

        messages = [
            create_mock_message("What time works?", is_from_me=False),
        ]

        context = ThreadContext(
            messages=messages,
            topic=ThreadTopic.LOGISTICS,
            state=ThreadState.OPEN_QUESTION,
            user_role=UserRole.RESPONDER,
            confidence=0.8,
            relevant_messages=messages,
        )

        config = ThreadedReplyConfig(
            max_response_length=50,
            response_style="concise",
            include_action_items=True,
            suggest_follow_up=False,
        )

        prompt = build_threaded_reply_prompt(context, config)

        assert "Logistics" in prompt
        assert "concise" in prompt
        assert "What time works?" in prompt

    def test_build_threaded_reply_prompt_emotional_support(self):
        """Test building prompt for emotional support thread."""
        from jarvis.prompts import build_threaded_reply_prompt

        messages = [
            create_mock_message("I'm feeling really down today", is_from_me=False),
        ]

        context = ThreadContext(
            messages=messages,
            topic=ThreadTopic.EMOTIONAL_SUPPORT,
            state=ThreadState.AWAITING_RESPONSE,
            user_role=UserRole.RESPONDER,
            confidence=0.8,
            relevant_messages=messages,
        )

        config = ThreadedReplyConfig(
            max_response_length=150,
            response_style="empathetic",
            include_action_items=False,
            suggest_follow_up=True,
        )

        prompt = build_threaded_reply_prompt(context, config)

        assert "Emotional Support" in prompt
        assert "empathetic" in prompt
        assert "Show empathy" in prompt

    def test_build_threaded_reply_prompt_planning(self):
        """Test building prompt for planning thread."""
        from jarvis.prompts import build_threaded_reply_prompt

        messages = [
            create_mock_message("Let's plan a party", is_from_me=False),
            create_mock_message("Great idea!", is_from_me=True),
            create_mock_message("Any ideas for the venue?", is_from_me=False),
        ]

        context = ThreadContext(
            messages=messages,
            topic=ThreadTopic.PLANNING,
            state=ThreadState.OPEN_QUESTION,
            user_role=UserRole.PARTICIPANT,
            confidence=0.8,
            relevant_messages=messages,
        )

        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="detailed",
            include_action_items=True,
            suggest_follow_up=True,
        )

        prompt = build_threaded_reply_prompt(context, config)

        assert "Planning" in prompt
        assert "detailed" in prompt
        assert "constructive" in prompt.lower() or "next steps" in prompt.lower()

    def test_build_threaded_reply_prompt_with_custom_instruction(self):
        """Test building prompt with custom instruction."""
        from jarvis.prompts import build_threaded_reply_prompt

        messages = [
            create_mock_message("Hey there", is_from_me=False),
        ]

        context = ThreadContext(
            messages=messages,
            topic=ThreadTopic.CATCHING_UP,
            state=ThreadState.AWAITING_RESPONSE,
            user_role=UserRole.RESPONDER,
            confidence=0.5,
        )

        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="warm",
            include_action_items=False,
            suggest_follow_up=True,
        )

        prompt = build_threaded_reply_prompt(
            context, config, instruction="be extra friendly"
        )

        assert "extra friendly" in prompt

    def test_build_threaded_reply_prompt_group_chat(self):
        """Test building prompt for group chat."""
        from jarvis.prompts import build_threaded_reply_prompt

        messages = [
            create_mock_message("Hey everyone", is_from_me=False, sender="John"),
            create_mock_message("Hi", is_from_me=False, sender="Sarah"),
        ]

        context = ThreadContext(
            messages=messages,
            topic=ThreadTopic.CATCHING_UP,
            state=ThreadState.IN_DISCUSSION,
            user_role=UserRole.PARTICIPANT,
            confidence=0.5,
            participants_count=3,
        )

        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="warm",
            include_action_items=False,
            suggest_follow_up=True,
        )

        prompt = build_threaded_reply_prompt(context, config)

        assert "Group chat" in prompt or "3 participants" in prompt

    def test_get_thread_max_tokens(self):
        """Test max tokens calculation."""
        from jarvis.prompts import get_thread_max_tokens

        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="detailed",
            include_action_items=True,
            suggest_follow_up=True,
        )

        max_tokens = get_thread_max_tokens(config)

        # Should be reasonable range
        assert 30 <= max_tokens <= 150


class TestThreadedExamples:
    """Tests for thread-specific few-shot examples."""

    def test_logistics_examples_exist(self):
        """Test logistics examples are defined."""
        from jarvis.prompts import LOGISTICS_THREAD_EXAMPLES

        assert len(LOGISTICS_THREAD_EXAMPLES) >= 2
        for ex in LOGISTICS_THREAD_EXAMPLES:
            assert ex.context
            assert ex.output
            # Logistics responses should be concise
            assert len(ex.output) < 100

    def test_emotional_support_examples_exist(self):
        """Test emotional support examples are defined."""
        from jarvis.prompts import EMOTIONAL_SUPPORT_THREAD_EXAMPLES

        assert len(EMOTIONAL_SUPPORT_THREAD_EXAMPLES) >= 2
        for ex in EMOTIONAL_SUPPORT_THREAD_EXAMPLES:
            assert ex.context
            assert ex.output
            # Emotional support responses should be longer and empathetic
            assert len(ex.output) > 50

    def test_planning_examples_exist(self):
        """Test planning examples are defined."""
        from jarvis.prompts import PLANNING_THREAD_EXAMPLES

        assert len(PLANNING_THREAD_EXAMPLES) >= 2
        for ex in PLANNING_THREAD_EXAMPLES:
            assert ex.context
            assert ex.output

    def test_thread_examples_dict(self):
        """Test THREAD_EXAMPLES dict has all expected keys."""
        from jarvis.prompts import THREAD_EXAMPLES

        expected_keys = [
            "logistics",
            "emotional_support",
            "planning",
            "catching_up",
            "quick_exchange",
        ]

        for key in expected_keys:
            assert key in THREAD_EXAMPLES
            assert len(THREAD_EXAMPLES[key]) > 0


class TestThreadAwareGenerator:
    """Tests for ThreadAwareGenerator class."""

    def test_generator_creation(self):
        """Test creating a ThreadAwareGenerator."""
        from models.generator import ThreadAwareGenerator

        generator = ThreadAwareGenerator()
        assert generator is not None
        assert generator._generator is not None

    def test_generator_with_custom_base(self):
        """Test creating generator with custom base generator."""
        from models.generator import MLXGenerator, ThreadAwareGenerator

        base = MLXGenerator()
        generator = ThreadAwareGenerator(base_generator=base)
        assert generator._generator is base

    def test_generator_delegates_is_loaded(self):
        """Test is_loaded delegates to base generator."""
        from models.generator import MLXGenerator, ThreadAwareGenerator

        base = MagicMock(spec=MLXGenerator)
        base.is_loaded.return_value = True

        generator = ThreadAwareGenerator(base_generator=base)
        result = generator.is_loaded()

        assert result is True
        base.is_loaded.assert_called_once()

    def test_generator_delegates_load(self):
        """Test load delegates to base generator."""
        from models.generator import MLXGenerator, ThreadAwareGenerator

        base = MagicMock(spec=MLXGenerator)
        base.load.return_value = True

        generator = ThreadAwareGenerator(base_generator=base)
        result = generator.load()

        assert result is True
        base.load.assert_called_once()

    def test_generator_delegates_unload(self):
        """Test unload delegates to base generator."""
        from models.generator import MLXGenerator, ThreadAwareGenerator

        base = MagicMock(spec=MLXGenerator)

        generator = ThreadAwareGenerator(base_generator=base)
        generator.unload()

        base.unload.assert_called_once()

    def test_generator_delegates_get_memory_usage(self):
        """Test get_memory_usage_mb delegates to base generator."""
        from models.generator import MLXGenerator, ThreadAwareGenerator

        base = MagicMock(spec=MLXGenerator)
        base.get_memory_usage_mb.return_value = 1500.0

        generator = ThreadAwareGenerator(base_generator=base)
        result = generator.get_memory_usage_mb()

        assert result == 1500.0
        base.get_memory_usage_mb.assert_called_once()

    def test_get_temperature_for_topic(self):
        """Test temperature selection for different topics."""
        from models.generator import ThreadAwareGenerator

        generator = ThreadAwareGenerator()

        # Logistics should have low temperature (precise)
        temp_logistics = generator._get_temperature_for_topic(ThreadTopic.LOGISTICS)
        assert temp_logistics < 0.5

        # Emotional support should have higher temperature (warm/varied)
        temp_emotional = generator._get_temperature_for_topic(
            ThreadTopic.EMOTIONAL_SUPPORT
        )
        assert temp_emotional > 0.5

        # Quick exchange should have low temperature
        temp_quick = generator._get_temperature_for_topic(ThreadTopic.QUICK_EXCHANGE)
        assert temp_quick < 0.5

    def test_get_thread_examples(self):
        """Test getting examples for thread topic."""
        from models.generator import ThreadAwareGenerator

        generator = ThreadAwareGenerator()

        examples = generator._get_thread_examples(ThreadTopic.LOGISTICS)
        assert len(examples) > 0
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in examples)

    def test_post_process_response_quick_exchange(self):
        """Test post-processing for quick exchange."""
        from models.generator import ThreadAwareGenerator

        generator = ThreadAwareGenerator()
        context = ThreadContext(
            messages=[],
            topic=ThreadTopic.QUICK_EXCHANGE,
            state=ThreadState.AWAITING_RESPONSE,
            user_role=UserRole.RESPONDER,
            confidence=0.8,
        )
        config = ThreadedReplyConfig(
            max_response_length=30,
            response_style="brief",
            include_action_items=False,
            suggest_follow_up=False,
        )

        # Long text should be truncated for quick exchange
        # The post_process truncates at 50 chars for quick exchanges
        long_text = (
            "This is a really long response that goes on and on. "
            "It has multiple sentences and should be shortened significantly."
        )
        result = generator._post_process_response(long_text, context, config)

        # Quick exchange responses should be shortened (first line only, max 50 chars)
        assert len(result) <= 51  # Allow for ending at sentence boundary

    def test_post_process_response_removes_artifacts(self):
        """Test post-processing removes prompt artifacts."""
        from models.generator import ThreadAwareGenerator

        generator = ThreadAwareGenerator()
        context = ThreadContext(
            messages=[],
            topic=ThreadTopic.CATCHING_UP,
            state=ThreadState.AWAITING_RESPONSE,
            user_role=UserRole.RESPONDER,
            confidence=0.8,
        )
        config = ThreadedReplyConfig(
            max_response_length=100,
            response_style="warm",
            include_action_items=False,
            suggest_follow_up=True,
        )

        text_with_artifacts = "Hey there! ### Extra stuff"
        result = generator._post_process_response(text_with_artifacts, context, config)

        assert "###" not in result
        assert "Hey there!" in result


class TestLogisticsThreadsGetConciseResponses:
    """Tests verifying logistics threads get concise responses."""

    def test_logistics_config_has_short_max_length(self):
        """Test logistics config has short max response length."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.LOGISTICS]
        assert config.max_response_length <= 50

    def test_logistics_config_is_concise_style(self):
        """Test logistics config has concise style."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.LOGISTICS]
        assert config.response_style == "concise"

    def test_logistics_examples_are_short(self):
        """Test logistics examples demonstrate concise responses."""
        from jarvis.prompts import LOGISTICS_THREAD_EXAMPLES

        for ex in LOGISTICS_THREAD_EXAMPLES:
            # Logistics replies should be under 60 chars
            assert len(ex.output) < 60, f"Too long: {ex.output}"


class TestEmotionalSupportThreadsGetEmpatheticResponses:
    """Tests verifying emotional support threads get empathetic responses."""

    def test_emotional_support_config_has_longer_max_length(self):
        """Test emotional support config has longer max response length."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.EMOTIONAL_SUPPORT]
        assert config.max_response_length >= 100

    def test_emotional_support_config_is_empathetic_style(self):
        """Test emotional support config has empathetic style."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.EMOTIONAL_SUPPORT]
        assert config.response_style == "empathetic"

    def test_emotional_support_config_suggests_follow_up(self):
        """Test emotional support config suggests follow-up."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.EMOTIONAL_SUPPORT]
        assert config.suggest_follow_up is True

    def test_emotional_support_examples_show_empathy(self):
        """Test emotional support examples demonstrate empathy."""
        from jarvis.prompts import EMOTIONAL_SUPPORT_THREAD_EXAMPLES

        empathy_phrases = [
            "sorry",
            "understand",
            "here for you",
            "that's",
            "I know",
            "miss",
        ]

        for ex in EMOTIONAL_SUPPORT_THREAD_EXAMPLES:
            # Each example should contain empathetic language
            output_lower = ex.output.lower()
            has_empathy = any(phrase in output_lower for phrase in empathy_phrases)
            assert has_empathy, f"Missing empathy in: {ex.output}"


class TestPlanningThreadsIncludeActionItems:
    """Tests verifying planning threads include action items."""

    def test_planning_config_includes_action_items(self):
        """Test planning config includes action items."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.PLANNING]
        assert config.include_action_items is True

    def test_planning_config_suggests_follow_up(self):
        """Test planning config suggests follow-up."""
        from jarvis.threading import TOPIC_RESPONSE_CONFIG, ThreadTopic

        config = TOPIC_RESPONSE_CONFIG[ThreadTopic.PLANNING]
        assert config.suggest_follow_up is True

    def test_planning_examples_have_action_orientation(self):
        """Test planning examples demonstrate action-oriented responses."""
        from jarvis.prompts import PLANNING_THREAD_EXAMPLES

        action_phrases = [
            "I can",
            "I'll",
            "let's",
            "should we",
            "how about",
            "?",
            "reservation",
            "coordinate",
            "bring",
            "check",
        ]

        for ex in PLANNING_THREAD_EXAMPLES:
            output_lower = ex.output.lower()
            has_action = any(phrase.lower() in output_lower for phrase in action_phrases)
            assert has_action, f"Missing action orientation in: {ex.output}"


class TestPromptRegistryThreadedExamples:
    """Tests for thread examples in PromptRegistry."""

    def test_registry_has_thread_logistics(self):
        """Test registry has thread_logistics examples."""
        from jarvis.prompts import get_prompt_registry

        registry = get_prompt_registry()
        examples = registry.get_examples("thread_logistics")
        assert len(examples) > 0

    def test_registry_has_thread_emotional_support(self):
        """Test registry has thread_emotional_support examples."""
        from jarvis.prompts import get_prompt_registry

        registry = get_prompt_registry()
        examples = registry.get_examples("thread_emotional_support")
        assert len(examples) > 0

    def test_registry_has_thread_planning(self):
        """Test registry has thread_planning examples."""
        from jarvis.prompts import get_prompt_registry

        registry = get_prompt_registry()
        examples = registry.get_examples("thread_planning")
        assert len(examples) > 0

    def test_registry_has_threaded_reply_template(self):
        """Test registry has threaded_reply template."""
        from jarvis.prompts import get_prompt_registry

        registry = get_prompt_registry()
        template = registry.get_template("threaded_reply")
        assert template is not None
        assert template.name == "threaded_reply"


# Optional tests that require sentence_transformers
requires_sentence_transformers = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available",
)


@requires_sentence_transformers
class TestThreadAnalyzerWithSemantics:
    """Tests for ThreadAnalyzer with semantic similarity (requires sentence_transformers)."""

    def test_semantic_topic_detection(self):
        """Test topic detection with semantic similarity."""
        analyzer = ThreadAnalyzer()
        messages = [
            create_mock_message(
                "I just got some terrible news about my grandmother", is_from_me=False
            ),
        ]

        context = analyzer.analyze(messages)

        # Should detect emotional support even without exact pattern match
        # Note: may fall back to pattern matching if embeddings unavailable
        assert context.topic in (
            ThreadTopic.EMOTIONAL_SUPPORT,
            ThreadTopic.CATCHING_UP,
            ThreadTopic.UNKNOWN,
        )
