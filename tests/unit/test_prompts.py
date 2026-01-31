"""Unit tests for iMessage reply prompt templates and builders.

Tests cover prompt building, tone detection, and token limit checking
for the jarvis/prompts.py module.
"""

import pytest

from jarvis.prompts import (
    API_REPLY_EXAMPLES,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES,
    API_SUMMARY_EXAMPLES_METADATA,
    CASUAL_INDICATORS,
    CASUAL_REPLY_EXAMPLES,
    MAX_CONTEXT_CHARS,
    MAX_PROMPT_TOKENS,
    PROFESSIONAL_INDICATORS,
    PROFESSIONAL_REPLY_EXAMPLES,
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    REPLY_EXAMPLES,
    REPLY_TEMPLATE,
    SEARCH_ANSWER_EXAMPLES,
    SEARCH_ANSWER_TEMPLATE,
    SUMMARIZATION_EXAMPLES,
    SUMMARY_EXAMPLES,
    SUMMARY_TEMPLATE,
    TEXT_ABBREVIATIONS,
    FewShotExample,
    PromptMetadata,
    PromptRegistry,
    PromptTemplate,
    UserStyleAnalysis,
    analyze_user_style,
    build_reply_prompt,
    build_search_answer_prompt,
    build_style_instructions,
    build_summary_prompt,
    detect_tone,
    estimate_tokens,
    get_prompt_registry,
    is_within_token_limit,
    reset_prompt_registry,
)


class TestFewShotExamples:
    """Tests for few-shot example definitions."""

    def test_casual_examples_exist(self):
        """Verify casual reply examples are defined."""
        assert len(CASUAL_REPLY_EXAMPLES) >= 5
        assert all(isinstance(ex, FewShotExample) for ex in CASUAL_REPLY_EXAMPLES)

    def test_professional_examples_exist(self):
        """Verify professional reply examples are defined."""
        assert len(PROFESSIONAL_REPLY_EXAMPLES) >= 5
        assert all(isinstance(ex, FewShotExample) for ex in PROFESSIONAL_REPLY_EXAMPLES)

    def test_summarization_examples_exist(self):
        """Verify summarization examples are defined."""
        assert len(SUMMARIZATION_EXAMPLES) >= 3
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in SUMMARIZATION_EXAMPLES)

    def test_search_answer_examples_exist(self):
        """Verify search answer examples are defined."""
        assert len(SEARCH_ANSWER_EXAMPLES) >= 3
        assert all(isinstance(ex, tuple) and len(ex) == 3 for ex in SEARCH_ANSWER_EXAMPLES)

    def test_casual_examples_have_casual_tone(self):
        """Verify casual examples are marked with casual tone."""
        for ex in CASUAL_REPLY_EXAMPLES:
            assert ex.tone == "casual"

    def test_professional_examples_have_professional_tone(self):
        """Verify professional examples are marked with professional tone."""
        for ex in PROFESSIONAL_REPLY_EXAMPLES:
            assert ex.tone == "professional"

    def test_examples_have_context_and_output(self):
        """Verify all examples have non-empty context and output."""
        all_examples = CASUAL_REPLY_EXAMPLES + PROFESSIONAL_REPLY_EXAMPLES
        for ex in all_examples:
            assert ex.context, "Example context should not be empty"
            assert ex.output, "Example output should not be empty"


class TestPromptTemplates:
    """Tests for prompt template definitions."""

    def test_reply_template_defined(self):
        """Verify reply template is properly defined."""
        assert REPLY_TEMPLATE.name == "reply_generation"
        assert "{context}" in REPLY_TEMPLATE.template
        assert "{last_message}" in REPLY_TEMPLATE.template
        assert "{tone}" in REPLY_TEMPLATE.template
        assert "{examples}" in REPLY_TEMPLATE.template

    def test_summary_template_defined(self):
        """Verify summary template is properly defined."""
        assert SUMMARY_TEMPLATE.name == "conversation_summary"
        assert "{context}" in SUMMARY_TEMPLATE.template
        assert "{examples}" in SUMMARY_TEMPLATE.template

    def test_search_answer_template_defined(self):
        """Verify search answer template is properly defined."""
        assert SEARCH_ANSWER_TEMPLATE.name == "search_answer"
        assert "{context}" in SEARCH_ANSWER_TEMPLATE.template
        assert "{question}" in SEARCH_ANSWER_TEMPLATE.template
        assert "{examples}" in SEARCH_ANSWER_TEMPLATE.template

    def test_templates_have_system_messages(self):
        """Verify all templates have system messages."""
        assert REPLY_TEMPLATE.system_message
        assert SUMMARY_TEMPLATE.system_message
        assert SEARCH_ANSWER_TEMPLATE.system_message

    def test_templates_have_max_output_tokens(self):
        """Verify all templates have max output tokens defined."""
        assert REPLY_TEMPLATE.max_output_tokens > 0
        assert SUMMARY_TEMPLATE.max_output_tokens > 0
        assert SEARCH_ANSWER_TEMPLATE.max_output_tokens > 0


class TestToneDetection:
    """Tests for tone detection functionality."""

    def test_detect_tone_empty_messages(self):
        """Test tone detection with empty message list."""
        result = detect_tone([])
        assert result == "casual"

    def test_detect_tone_casual_slang(self):
        """Test detection of casual tone from slang."""
        messages = ["hey whats up", "gonna grab lunch", "lol thats funny"]
        result = detect_tone(messages)
        assert result == "casual"

    def test_detect_tone_casual_emoji(self):
        """Test detection of casual tone from emoji."""
        messages = ["sounds good! 😊", "see you later! 👋"]
        result = detect_tone(messages)
        assert result == "casual"

    def test_detect_tone_casual_abbreviations(self):
        """Test detection of casual tone from common abbreviations."""
        messages = ["brb", "ttyl", "omw", "thx"]
        result = detect_tone(messages)
        assert result == "casual"

    def test_detect_tone_professional_language(self):
        """Test detection of professional tone from formal language."""
        messages = [
            "Please review the attached proposal",
            "I confirm our meeting is scheduled for Thursday",
            "Regarding the quarterly report",
        ]
        result = detect_tone(messages)
        assert result == "professional"

    def test_detect_tone_professional_greetings(self):
        """Test detection of professional tone from formal greetings."""
        messages = ["Dear Mr. Smith", "Good morning, team"]
        result = detect_tone(messages)
        assert result == "professional"

    def test_detect_tone_mixed(self):
        """Test detection of mixed tone."""
        messages = [
            "Hey! Can you send the quarterly report?",
            "Sure thing, I'll confirm the deadline",
        ]
        result = detect_tone(messages)
        # This may return casual, professional, or mixed depending on balance
        assert result in ("casual", "professional", "mixed")

    def test_detect_tone_repeated_characters(self):
        """Test that repeated characters increase casual score."""
        messages = ["hahahaha", "nooooo way", "yessss"]
        result = detect_tone(messages)
        assert result == "casual"

    def test_detect_tone_exclamation_marks(self):
        """Test that multiple exclamation marks increase casual score."""
        messages = ["That's awesome!!! So excited!!!"]
        result = detect_tone(messages)
        assert result == "casual"

    def test_detect_tone_neutral_messages(self):
        """Test tone detection with neutral messages."""
        messages = ["ok", "yes", "no"]
        result = detect_tone(messages)
        # Should default to casual when no strong indicators
        assert result in ("casual", "mixed")

    def test_detect_tone_case_insensitive(self):
        """Test that tone detection is case insensitive."""
        messages = ["LOL", "GONNA", "BTW"]
        result = detect_tone(messages)
        assert result == "casual"


class TestBuildReplyPrompt:
    """Tests for reply prompt building."""

    def test_build_reply_prompt_basic(self):
        """Test basic reply prompt building."""
        context = "[10:00] John: Hey, want to get lunch?"
        last_message = "Hey, want to get lunch?"

        result = build_reply_prompt(context, last_message)

        assert "### Conversation Context:" in result
        assert context in result
        assert "### Last message to reply to:" in result
        assert last_message in result
        assert "### Your reply:" in result

    def test_build_reply_prompt_casual_tone(self):
        """Test reply prompt with casual tone."""
        result = build_reply_prompt(context="Hey there", last_message="Hey there", tone="casual")

        assert "casual/friendly" in result

    def test_build_reply_prompt_professional_tone(self):
        """Test reply prompt with professional tone."""
        result = build_reply_prompt(
            context="Meeting request", last_message="Meeting request", tone="professional"
        )

        assert "professional/formal" in result

    def test_build_reply_prompt_with_instruction(self):
        """Test reply prompt with custom instruction."""
        result = build_reply_prompt(
            context="Test context",
            last_message="Test message",
            instruction="Be brief and direct",
        )

        assert "Be brief and direct" in result

    def test_build_reply_prompt_includes_examples(self):
        """Test that reply prompt includes few-shot examples."""
        result = build_reply_prompt(context="Test", last_message="Test", tone="casual")

        assert "### Examples:" in result
        assert "Context:" in result
        assert "Reply:" in result

    def test_build_reply_prompt_mixed_tone_uses_casual(self):
        """Test that mixed tone uses casual examples."""
        result = build_reply_prompt(context="Test", last_message="Test", tone="mixed")

        # Mixed tone should use casual examples
        assert "casual/friendly" in result


class TestBuildSummaryPrompt:
    """Tests for summary prompt building."""

    def test_build_summary_prompt_basic(self):
        """Test basic summary prompt building."""
        context = """[Mon] John: Meeting at 3pm
[Mon] You: Sounds good"""

        result = build_summary_prompt(context)

        assert "### Conversation:" in result
        assert context in result
        assert "### Summary:" in result
        assert "### Instructions:" in result

    def test_build_summary_prompt_with_focus(self):
        """Test summary prompt with focus area."""
        result = build_summary_prompt(context="Test conversation", focus="action items")

        assert "action items" in result
        assert "Focus especially on" in result

    def test_build_summary_prompt_includes_examples(self):
        """Test that summary prompt includes few-shot examples."""
        result = build_summary_prompt(context="Test conversation")

        assert "Conversation:" in result
        assert "Summary:" in result

    def test_build_summary_prompt_without_focus(self):
        """Test summary prompt without focus doesn't have focus instruction."""
        result = build_summary_prompt(context="Test conversation", focus=None)

        assert "Focus especially on" not in result


class TestBuildSearchAnswerPrompt:
    """Tests for search answer prompt building."""

    def test_build_search_answer_prompt_basic(self):
        """Test basic search answer prompt building."""
        context = "[Mon] John: Let's meet at the coffee shop"
        question = "Where are we meeting?"

        result = build_search_answer_prompt(context, question)

        assert "### Messages:" in result
        assert context in result
        assert "### Question:" in result
        assert question in result
        assert "### Answer:" in result

    def test_build_search_answer_prompt_includes_examples(self):
        """Test that search answer prompt includes few-shot examples."""
        result = build_search_answer_prompt(context="Test messages", question="Test question?")

        assert "Messages:" in result
        assert "Question:" in result
        assert "Answer:" in result

    def test_build_search_answer_prompt_has_instructions(self):
        """Test that search answer prompt includes instructions."""
        result = build_search_answer_prompt(context="Test messages", question="Test question?")

        assert "Answer the question based only on the messages" in result


class TestContextTruncation:
    """Tests for context truncation when exceeding limits."""

    def test_short_context_not_truncated(self):
        """Test that short context is not truncated."""
        short_context = "This is a short context"
        result = build_reply_prompt(short_context, "last message")

        assert "[Earlier messages truncated]" not in result
        assert short_context in result

    def test_long_context_is_truncated(self):
        """Test that long context is truncated."""
        # Create a context longer than MAX_CONTEXT_CHARS
        long_context = "Message line\n" * 500

        result = build_reply_prompt(long_context, "last message")

        assert "[Earlier messages truncated]" in result
        # Verify prompt is still within reasonable bounds
        assert len(result) < MAX_CONTEXT_CHARS * 2

    def test_truncation_preserves_recent_messages(self):
        """Test that truncation keeps recent messages."""
        # Create messages with identifiable markers
        messages = []
        for i in range(100):
            messages.append(f"[Message {i}] Content here that takes up space")
        long_context = "\n".join(messages)

        result = build_reply_prompt(long_context, "last message")

        # Recent messages (higher numbers) should be present
        assert "Message 99" in result
        # Old messages (lower numbers) may be truncated
        # The exact cutoff depends on MAX_CONTEXT_CHARS


class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        result = estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        # "Hello world" is 11 chars, ~2-3 tokens
        result = estimate_tokens("Hello world")
        assert result >= 2
        assert result <= 5

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        long_text = "This is a longer text " * 100
        result = estimate_tokens(long_text)

        # Should be proportional to length
        assert result > 100

    def test_is_within_token_limit_short_prompt(self):
        """Test token limit check for short prompt."""
        short_prompt = "This is a short prompt"
        assert is_within_token_limit(short_prompt) is True

    def test_is_within_token_limit_long_prompt(self):
        """Test token limit check for very long prompt."""
        # Create a prompt that exceeds the limit
        long_prompt = "word " * (MAX_PROMPT_TOKENS * 2)
        assert is_within_token_limit(long_prompt) is False

    def test_is_within_token_limit_custom_limit(self):
        """Test token limit check with custom limit."""
        prompt = "A" * 100  # ~25 tokens
        assert is_within_token_limit(prompt, limit=50) is True
        assert is_within_token_limit(prompt, limit=10) is False


class TestIndicatorSets:
    """Tests for casual and professional indicator sets."""

    def test_casual_indicators_not_empty(self):
        """Verify casual indicators set is populated."""
        assert len(CASUAL_INDICATORS) > 0

    def test_professional_indicators_not_empty(self):
        """Verify professional indicators set is populated."""
        assert len(PROFESSIONAL_INDICATORS) > 0

    def test_common_casual_terms_present(self):
        """Verify common casual terms are in the set."""
        common_casual = ["lol", "haha", "brb", "omg", "thx", "ty"]
        for term in common_casual:
            assert term in CASUAL_INDICATORS, f"Expected '{term}' in CASUAL_INDICATORS"

    def test_common_professional_terms_present(self):
        """Verify common professional terms are in the set."""
        common_professional = ["regarding", "please", "confirm", "meeting", "deadline"]
        for term in common_professional:
            assert term in PROFESSIONAL_INDICATORS, f"Expected '{term}' in PROFESSIONAL_INDICATORS"


class TestPromptTemplateDataclass:
    """Tests for PromptTemplate dataclass."""

    def test_prompt_template_creation(self):
        """Test creating a PromptTemplate instance."""
        template = PromptTemplate(
            name="test",
            system_message="Test system message",
            template="Test template with {placeholder}",
            max_output_tokens=50,
        )

        assert template.name == "test"
        assert template.system_message == "Test system message"
        assert "{placeholder}" in template.template
        assert template.max_output_tokens == 50

    def test_prompt_template_default_max_tokens(self):
        """Test PromptTemplate default max_output_tokens."""
        template = PromptTemplate(
            name="test",
            system_message="Test",
            template="Test",
        )

        assert template.max_output_tokens == 100  # Default value


class TestFewShotExampleDataclass:
    """Tests for FewShotExample dataclass."""

    def test_few_shot_example_creation(self):
        """Test creating a FewShotExample instance."""
        example = FewShotExample(
            context="Test context",
            output="Test output",
            tone="casual",
        )

        assert example.context == "Test context"
        assert example.output == "Test output"
        assert example.tone == "casual"

    def test_few_shot_example_default_tone(self):
        """Test FewShotExample default tone is casual."""
        example = FewShotExample(context="Test", output="Output")

        assert example.tone == "casual"


class TestPromptQuality:
    """Tests for prompt quality and completeness."""

    def test_reply_prompt_has_clear_structure(self):
        """Test that reply prompt has clear, parseable structure."""
        result = build_reply_prompt(
            context="Test context",
            last_message="Test message",
        )

        # Should have distinct sections
        sections = [
            "### Conversation Context:",
            "### Instructions:",
            "### Examples:",
            "### Last message to reply to:",
            "### Your reply:",
        ]
        for section in sections:
            assert section in result, f"Missing section: {section}"

    def test_summary_prompt_has_clear_structure(self):
        """Test that summary prompt has clear, parseable structure."""
        result = build_summary_prompt(context="Test context")

        sections = ["### Conversation:", "### Instructions:", "### Summary:"]
        for section in sections:
            assert section in result, f"Missing section: {section}"

    def test_search_prompt_has_clear_structure(self):
        """Test that search prompt has clear, parseable structure."""
        result = build_search_answer_prompt(
            context="Test messages",
            question="Test question?",
        )

        sections = ["### Messages:", "### Question:", "### Instructions:", "### Answer:"]
        for section in sections:
            assert section in result, f"Missing section: {section}"

    def test_prompts_end_with_generation_marker(self):
        """Test that prompts end with a marker for model to continue."""
        reply = build_reply_prompt(context="Test", last_message="Test")
        summary = build_summary_prompt(context="Test")
        search = build_search_answer_prompt(context="Test", question="Test?")

        assert reply.endswith("### Your reply:")
        assert summary.endswith("### Summary:")
        assert search.endswith("### Answer:")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_context(self):
        """Test handling of empty context."""
        result = build_reply_prompt(context="", last_message="Hello")

        # Should still produce a valid prompt
        assert "### Your reply:" in result

    def test_empty_last_message(self):
        """Test handling of empty last message."""
        result = build_reply_prompt(context="Context", last_message="")

        # Should still produce a valid prompt
        assert "### Your reply:" in result

    def test_special_characters_in_context(self):
        """Test handling of special characters in context."""
        context = "Test with special chars: {} [] () <> & | \\ \" '"
        result = build_reply_prompt(context=context, last_message="Test")

        # Should include the special characters without breaking
        assert "{}" in result or "\\{\\}" in result  # May be escaped

    def test_unicode_in_messages(self):
        """Test handling of unicode characters including emoji."""
        context = "Hey! 👋 How are you? 你好"
        result = build_reply_prompt(context=context, last_message=context)

        # Should include unicode without breaking
        assert "👋" in result
        assert "你好" in result

    def test_very_long_single_message(self):
        """Test handling of a very long single message."""
        long_message = "A" * 10000
        result = build_reply_prompt(context=long_message, last_message=long_message)

        # Should truncate and still produce valid prompt
        assert "### Your reply:" in result

    def test_multiline_messages(self):
        """Test handling of multiline messages."""
        context = """Line 1
Line 2
Line 3

Line after blank"""
        result = build_reply_prompt(context=context, last_message="Last")

        # Should preserve multiline structure
        assert "Line 1" in result


class TestPromptVersioning:
    """Tests for prompt versioning and metadata."""

    def test_prompt_version_defined(self):
        """Verify PROMPT_VERSION is defined and valid."""
        assert PROMPT_VERSION
        # Should be semver format
        parts = PROMPT_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_prompt_last_updated_defined(self):
        """Verify PROMPT_LAST_UPDATED is defined and valid."""
        assert PROMPT_LAST_UPDATED
        # Should be ISO date format YYYY-MM-DD
        parts = PROMPT_LAST_UPDATED.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # Year
        assert len(parts[1]) == 2  # Month
        assert len(parts[2]) == 2  # Day


class TestPromptMetadata:
    """Tests for PromptMetadata dataclass."""

    def test_prompt_metadata_creation(self):
        """Test creating a PromptMetadata instance."""
        metadata = PromptMetadata(
            name="test",
            version="1.0.0",
            last_updated="2026-01-01",
            description="Test description",
        )

        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.last_updated == "2026-01-01"
        assert metadata.description == "Test description"

    def test_prompt_metadata_defaults(self):
        """Test PromptMetadata uses default version and date."""
        metadata = PromptMetadata(name="test")

        assert metadata.name == "test"
        assert metadata.version == PROMPT_VERSION
        assert metadata.last_updated == PROMPT_LAST_UPDATED
        assert metadata.description == ""


class TestAPIExamples:
    """Tests for API-style prompt examples."""

    def test_api_reply_examples_exist(self):
        """Verify API reply examples are defined."""
        assert len(API_REPLY_EXAMPLES) >= 3
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in API_REPLY_EXAMPLES)

    def test_api_reply_examples_have_instruction_format(self):
        """Verify API reply examples use instruction format."""
        for context, _output in API_REPLY_EXAMPLES:
            assert "Instruction:" in context or "Last message:" in context

    def test_api_summary_examples_exist(self):
        """Verify API summary examples are defined."""
        assert len(API_SUMMARY_EXAMPLES) >= 1
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in API_SUMMARY_EXAMPLES)

    def test_api_reply_examples_metadata_defined(self):
        """Verify API reply examples have metadata."""
        assert API_REPLY_EXAMPLES_METADATA.name == "api_reply_examples"
        assert API_REPLY_EXAMPLES_METADATA.version == PROMPT_VERSION

    def test_api_summary_examples_metadata_defined(self):
        """Verify API summary examples have metadata."""
        assert API_SUMMARY_EXAMPLES_METADATA.name == "api_summary_examples"
        assert API_SUMMARY_EXAMPLES_METADATA.version == PROMPT_VERSION


class TestCompatibilityExports:
    """Tests for compatibility exports (REPLY_EXAMPLES, SUMMARY_EXAMPLES)."""

    def test_reply_examples_are_tuples(self):
        """Verify REPLY_EXAMPLES are tuples for GenerationRequest compatibility."""
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in REPLY_EXAMPLES)

    def test_reply_examples_match_casual_examples(self):
        """Verify REPLY_EXAMPLES are derived from CASUAL_REPLY_EXAMPLES."""
        assert len(REPLY_EXAMPLES) == len(CASUAL_REPLY_EXAMPLES)
        for (ctx, out), example in zip(REPLY_EXAMPLES, CASUAL_REPLY_EXAMPLES):
            assert ctx == example.context
            assert out == example.output

    def test_summary_examples_are_tuples(self):
        """Verify SUMMARY_EXAMPLES are tuples."""
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in SUMMARY_EXAMPLES)

    def test_summary_examples_alias_summarization_examples(self):
        """Verify SUMMARY_EXAMPLES is an alias for SUMMARIZATION_EXAMPLES."""
        assert SUMMARY_EXAMPLES is SUMMARIZATION_EXAMPLES


class TestPromptRegistry:
    """Tests for PromptRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh PromptRegistry for each test."""
        reset_prompt_registry()
        return PromptRegistry()

    def test_registry_initialization(self, registry):
        """Test that registry initializes with all prompts."""
        examples = registry.list_examples()
        templates = registry.list_templates()

        assert "casual_reply" in examples
        assert "professional_reply" in examples
        assert "api_reply" in examples
        assert "api_summary" in examples
        assert "summarization" in examples
        assert "search_answer" in examples

        assert "reply_generation" in templates
        assert "conversation_summary" in templates
        assert "search_answer" in templates

    def test_get_examples(self, registry):
        """Test retrieving examples by name."""
        casual = registry.get_examples("casual_reply")
        assert len(casual) == len(CASUAL_REPLY_EXAMPLES)
        assert all(isinstance(ex, tuple) for ex in casual)

    def test_get_examples_unknown_raises(self, registry):
        """Test that unknown example set raises KeyError."""
        with pytest.raises(KeyError, match="Unknown example set"):
            registry.get_examples("nonexistent")

    def test_get_template(self, registry):
        """Test retrieving template by name."""
        template = registry.get_template("reply_generation")
        assert template.name == "reply_generation"
        assert template is REPLY_TEMPLATE

    def test_get_template_unknown_raises(self, registry):
        """Test that unknown template raises KeyError."""
        with pytest.raises(KeyError, match="Unknown template"):
            registry.get_template("nonexistent")

    def test_get_metadata(self, registry):
        """Test retrieving metadata by name."""
        metadata = registry.get_metadata("casual_reply")
        assert metadata.name == "casual_reply"
        assert metadata.version == PROMPT_VERSION

    def test_get_metadata_unknown_raises(self, registry):
        """Test that unknown metadata raises KeyError."""
        with pytest.raises(KeyError, match="Unknown prompt"):
            registry.get_metadata("nonexistent")

    def test_register_examples(self, registry):
        """Test registering new example sets."""
        custom_examples = [("input1", "output1"), ("input2", "output2")]
        registry.register_examples("custom", custom_examples)

        retrieved = registry.get_examples("custom")
        assert retrieved == custom_examples
        assert "custom" in registry.list_examples()

    def test_register_examples_with_metadata(self, registry):
        """Test registering examples with custom metadata."""
        custom_examples = [("input", "output")]
        custom_metadata = PromptMetadata(
            name="custom",
            version="2.0.0",
            description="Custom examples",
        )
        registry.register_examples("custom", custom_examples, custom_metadata)

        metadata = registry.get_metadata("custom")
        assert metadata.version == "2.0.0"
        assert metadata.description == "Custom examples"

    def test_register_template(self, registry):
        """Test registering new templates."""
        custom_template = PromptTemplate(
            name="custom_template",
            system_message="Custom system message",
            template="Custom template {placeholder}",
        )
        registry.register_template(custom_template)

        retrieved = registry.get_template("custom_template")
        assert retrieved is custom_template
        assert "custom_template" in registry.list_templates()

    def test_registry_version_property(self, registry):
        """Test registry version property."""
        assert registry.version == PROMPT_VERSION

    def test_registry_last_updated_property(self, registry):
        """Test registry last_updated property."""
        assert registry.last_updated == PROMPT_LAST_UPDATED


class TestPromptRegistrySingleton:
    """Tests for PromptRegistry singleton functions."""

    def test_get_prompt_registry_returns_instance(self):
        """Test that get_prompt_registry returns a PromptRegistry."""
        reset_prompt_registry()
        registry = get_prompt_registry()
        assert isinstance(registry, PromptRegistry)

    def test_get_prompt_registry_returns_same_instance(self):
        """Test that get_prompt_registry returns the same instance."""
        reset_prompt_registry()
        registry1 = get_prompt_registry()
        registry2 = get_prompt_registry()
        assert registry1 is registry2

    def test_reset_prompt_registry(self):
        """Test that reset_prompt_registry creates a new instance."""
        registry1 = get_prompt_registry()
        reset_prompt_registry()
        registry2 = get_prompt_registry()
        assert registry1 is not registry2


class TestUserStyleAnalysis:
    """Tests for UserStyleAnalysis dataclass."""

    def test_user_style_analysis_creation(self):
        """Test creating a UserStyleAnalysis instance."""
        style = UserStyleAnalysis(
            avg_length=25.0,
            min_length=5,
            max_length=50,
            formality="very_casual",
            uses_lowercase=True,
            uses_abbreviations=True,
            uses_minimal_punctuation=True,
            common_abbreviations=["u", "ur", "lol"],
            emoji_frequency=0.5,
            exclamation_frequency=0.1,
        )

        assert style.avg_length == 25.0
        assert style.formality == "very_casual"
        assert style.uses_lowercase is True
        assert style.uses_abbreviations is True
        assert "lol" in style.common_abbreviations

    def test_user_style_analysis_defaults(self):
        """Test UserStyleAnalysis default values."""
        style = UserStyleAnalysis()

        assert style.avg_length == 50.0
        assert style.formality == "casual"
        assert style.uses_lowercase is False
        assert style.uses_abbreviations is False
        assert style.common_abbreviations == []


class TestAnalyzeUserStyle:
    """Tests for analyze_user_style function."""

    def test_analyze_empty_messages(self):
        """Test analysis with empty message list."""
        result = analyze_user_style([])
        assert result.avg_length == 50.0  # Default
        assert result.formality == "casual"  # Default

    def test_analyze_very_casual_messages(self):
        """Test analysis of very casual messages."""
        messages = ["yeah", "lol ok", "u coming?", "k thx", "gonna be late"]
        result = analyze_user_style(messages)

        assert result.formality == "very_casual"
        assert result.uses_abbreviations is True
        assert result.avg_length < 20

    def test_analyze_casual_messages(self):
        """Test analysis of casual messages."""
        messages = [
            "sounds good to me",
            "let me know when you get here",
            "see you later",
        ]
        result = analyze_user_style(messages)

        assert result.formality in ("casual", "very_casual")

    def test_analyze_lowercase_detection(self):
        """Test detection of lowercase usage."""
        messages = ["hey whats up", "yeah im down", "cool see u later", "ok sounds good"]
        result = analyze_user_style(messages)

        assert result.uses_lowercase is True

    def test_analyze_mixed_case_messages(self):
        """Test that mixed case messages don't trigger lowercase flag."""
        messages = ["Hey! What's up?", "I'll be there soon.", "Sounds good!"]
        result = analyze_user_style(messages)

        assert result.uses_lowercase is False

    def test_analyze_abbreviation_detection(self):
        """Test detection of abbreviation usage."""
        messages = ["u there?", "gonna grab lunch", "thx for letting me know", "idk maybe"]
        result = analyze_user_style(messages)

        assert result.uses_abbreviations is True
        assert len(result.common_abbreviations) >= 2

    def test_analyze_message_length(self):
        """Test message length calculation."""
        short_messages = ["ok", "k", "yes", "no", "yep"]
        result = analyze_user_style(short_messages)

        assert result.avg_length < 10
        assert result.min_length <= 2
        assert result.max_length >= 2

    def test_analyze_minimal_punctuation(self):
        """Test detection of minimal punctuation usage."""
        messages = ["yeah sounds good", "ok cool", "see you there", "thanks"]
        result = analyze_user_style(messages)

        assert result.uses_minimal_punctuation is True

    def test_analyze_emoji_frequency(self):
        """Test emoji frequency calculation."""
        messages = ["sounds good! 😊", "see you soon! 👋", "awesome 🎉"]
        result = analyze_user_style(messages)

        assert result.emoji_frequency > 0.5

    def test_analyze_no_emojis(self):
        """Test emoji frequency with no emojis."""
        messages = ["sounds good", "ok", "yes"]
        result = analyze_user_style(messages)

        assert result.emoji_frequency == 0.0

    def test_analyze_exclamation_frequency(self):
        """Test exclamation frequency calculation."""
        messages = ["Awesome!", "Can't wait!", "So excited!!"]
        result = analyze_user_style(messages)

        assert result.exclamation_frequency > 1.0


class TestBuildStyleInstructions:
    """Tests for build_style_instructions function."""

    def test_build_instructions_very_casual(self):
        """Test instructions for very casual style."""
        style = UserStyleAnalysis(
            avg_length=15.0,
            formality="very_casual",
            uses_lowercase=True,
            uses_abbreviations=True,
            common_abbreviations=["u", "ur", "lol"],
            uses_minimal_punctuation=True,
            emoji_frequency=0.0,
            exclamation_frequency=0.1,
        )
        result = build_style_instructions(style)

        # Check for length guidance (brief, short, etc.)
        assert "brief" in result.lower() or "short" in result.lower()
        assert "lowercase" in result
        assert "casual" in result.lower() or "greetings" in result.lower()
        assert "abbreviations" in result

    def test_build_instructions_longer_messages(self):
        """Test instructions for longer message style."""
        style = UserStyleAnalysis(
            avg_length=100.0,
            formality="casual",
        )
        result = build_style_instructions(style)

        assert "1-2 sentences" in result

    def test_build_instructions_no_emojis(self):
        """Test instructions include emoji guidance when frequency is low."""
        style = UserStyleAnalysis(
            emoji_frequency=0.0,
        )
        result = build_style_instructions(style)

        assert "emoji" in result.lower()

    def test_build_instructions_avoid_exclamations(self):
        """Test instructions when exclamation frequency is low."""
        style = UserStyleAnalysis(
            exclamation_frequency=0.1,
        )
        result = build_style_instructions(style)

        assert "exclamation" in result.lower()


class TestBuildReplyPromptWithStyle:
    """Tests for build_reply_prompt with user_messages parameter."""

    def test_build_reply_prompt_with_user_messages(self):
        """Test that user_messages parameter adds style instructions."""
        context = "[10:00] John: Hey, want to grab lunch?"
        last_message = "Hey, want to grab lunch?"
        user_messages = ["yeah", "k sounds good", "lol ok"]

        result = build_reply_prompt(
            context=context,
            last_message=last_message,
            user_messages=user_messages,
        )

        # Should include style-matching instructions
        assert "### Your reply:" in result
        # Should have style guidance since user_messages provided
        assert "short" in result.lower() or "brief" in result.lower()

    def test_build_reply_prompt_without_user_messages(self):
        """Test that prompt works without user_messages."""
        context = "[10:00] John: Hey, want to grab lunch?"
        last_message = "Hey, want to grab lunch?"

        result = build_reply_prompt(
            context=context,
            last_message=last_message,
        )

        # Should still produce valid prompt
        assert "### Your reply:" in result
        assert "### Conversation Context:" in result

    def test_build_reply_prompt_style_includes_length_guidance(self):
        """Test that style analysis includes length guidance."""
        context = "Test context"
        last_message = "Test message"
        user_messages = ["ok", "k", "yes", "yep", "sure"]

        result = build_reply_prompt(
            context=context,
            last_message=last_message,
            user_messages=user_messages,
        )

        # Should have length guidance for short messages
        assert "short" in result.lower() or "brief" in result.lower()


class TestTextAbbreviations:
    """Tests for TEXT_ABBREVIATIONS constant."""

    def test_text_abbreviations_not_empty(self):
        """Verify TEXT_ABBREVIATIONS set is populated."""
        assert len(TEXT_ABBREVIATIONS) > 0

    def test_common_abbreviations_present(self):
        """Verify common text abbreviations are in the set."""
        common = ["u", "ur", "lol", "omg", "brb", "idk", "thx", "ty", "pls"]
        for abbrev in common:
            assert abbrev in TEXT_ABBREVIATIONS, f"Expected '{abbrev}' in TEXT_ABBREVIATIONS"
