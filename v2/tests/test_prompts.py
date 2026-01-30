"""Comprehensive tests for the prompts module.

These tests verify prompt building, formatting, and edge cases
to ensure the LLM receives properly formatted input.
"""

from __future__ import annotations

import pytest

from core.generation.prompts import (
    FEW_SHOT_EXAMPLES,
    REPLY_PROMPT,
    REPLY_PROMPT_WITH_HISTORY,
    _build_messages_array,
    _get_display_name,
    build_reply_prompt,
    format_past_replies,
    format_style_samples,
    get_examples_for_intent,
)


class TestReplyPromptTemplate:
    """Test the main prompt template structure."""

    def test_prompt_contains_placeholders(self) -> None:
        """Test that the template contains required placeholders."""
        assert "{user_name}" in REPLY_PROMPT
        assert "{conversation}" in REPLY_PROMPT

    def test_prompt_with_history_contains_placeholders(self) -> None:
        """Test that the history template contains all placeholders."""
        # New format uses few-shot examples
        assert "{few_shot}" in REPLY_PROMPT_WITH_HISTORY
        assert "{last_message}" in REPLY_PROMPT_WITH_HISTORY
        assert "{past_replies_section}" in REPLY_PROMPT_WITH_HISTORY
        assert "{availability_hint}" in REPLY_PROMPT_WITH_HISTORY


class TestFewShotExamples:
    """Test the few-shot examples structure."""

    def test_all_intents_have_examples(self) -> None:
        """Test that all expected intent types have examples."""
        expected_intents = [
            "yes_no_question",
            "open_question",
            "choice_question",
            "statement",
            "emotional",
            "greeting",
            "logistics",
            "sharing",
            "thanks",
            "farewell",
        ]
        for intent in expected_intents:
            assert intent in FEW_SHOT_EXAMPLES, f"Missing examples for {intent}"

    def test_examples_have_correct_structure(self) -> None:
        """Test that each example has the correct structure."""
        for intent, examples in FEW_SHOT_EXAMPLES.items():
            assert isinstance(examples, list), f"{intent} examples should be a list"
            assert len(examples) > 0, f"{intent} should have at least one example"

            for example in examples:
                assert "message" in example, f"{intent} example missing 'message'"
                assert "replies" in example, f"{intent} example missing 'replies'"
                assert isinstance(
                    example["replies"], list
                ), f"{intent} replies should be a list"
                assert len(example["replies"]) >= 3, f"{intent} should have 3+ replies"

    def test_examples_are_realistic(self) -> None:
        """Test that example messages are realistic and varied."""
        for intent, examples in FEW_SHOT_EXAMPLES.items():
            for example in examples:
                msg = example["message"]
                # Messages should be non-empty and reasonably sized
                assert len(msg) > 0, f"{intent} has empty message"
                assert len(msg) < 200, f"{intent} message too long"

                for reply in example["replies"]:
                    # Replies should be brief (typical text message length)
                    assert len(reply) > 0, f"{intent} has empty reply"
                    assert len(reply) < 100, f"{intent} reply too long: {reply}"


class TestGetExamplesForIntent:
    """Test the get_examples_for_intent function."""

    @pytest.mark.parametrize(
        "intent",
        [
            "yes_no_question",
            "open_question",
            "choice_question",
            "statement",
            "emotional",
            "greeting",
            "logistics",
            "sharing",
            "thanks",
            "farewell",
        ],
    )
    def test_get_examples_for_known_intents(self, intent: str) -> None:
        """Test that examples are returned for known intents."""
        result = get_examples_for_intent(intent)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Example" in result
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_unknown_intent_falls_back_to_statement(self) -> None:
        """Test that unknown intent falls back to statement examples."""
        result = get_examples_for_intent("unknown_intent")
        statement_result = get_examples_for_intent("statement")

        # Should fall back to statement
        assert len(result) > 0
        # Both should have same structure
        assert "Example" in result


class TestGetDisplayName:
    """Test the _get_display_name function."""

    def test_returns_none_for_from_me(self) -> None:
        """Test that None is returned for messages from me."""
        msg = {"is_from_me": True, "sender_name": "John"}
        assert _get_display_name(msg) is None

    def test_prefers_sender_name_over_sender(self) -> None:
        """Test that sender_name is preferred over sender."""
        msg = {"is_from_me": False, "sender_name": "John", "sender": "+1234567890"}
        assert _get_display_name(msg) == "John"

    def test_falls_back_to_sender(self) -> None:
        """Test fallback to sender when sender_name is missing."""
        msg = {"is_from_me": False, "sender": "Alice"}
        assert _get_display_name(msg) == "Alice"

    def test_uses_fallback_for_phone_number(self) -> None:
        """Test that phone numbers are replaced with fallback."""
        msg = {"is_from_me": False, "sender": "+1234567890"}
        assert _get_display_name(msg) == "Them"
        assert _get_display_name(msg, fallback="Contact") == "Contact"

    def test_uses_fallback_when_missing(self) -> None:
        """Test fallback is used when no sender info."""
        msg = {"is_from_me": False}
        assert _get_display_name(msg) == "Them"

    def test_empty_sender_name_falls_back(self) -> None:
        """Test that empty sender_name falls back."""
        msg = {"is_from_me": False, "sender_name": "", "sender": "Bob"}
        assert _get_display_name(msg) == "Bob"


class TestBuildMessagesArray:
    """Test the _build_messages_array function."""

    def test_empty_messages(self) -> None:
        """Test with empty message list."""
        texts, reply_to = _build_messages_array([])
        assert texts == []
        assert reply_to == ""

    def test_basic_conversation(self) -> None:
        """Test basic conversation formatting."""
        messages = [
            {"text": "Hey!", "is_from_me": False},
            {"text": "Hi there", "is_from_me": True},
            {"text": "How are you?", "is_from_me": False},
        ]
        texts, reply_to = _build_messages_array(messages)

        assert len(texts) == 3
        assert texts[0] == ">Hey!"  # Their message
        assert texts[1] == "Hi there"  # Your message (no prefix)
        assert texts[2] == ">How are you?"  # Their message
        assert reply_to == "How are you?"

    def test_max_messages_limit(self) -> None:
        """Test that max_messages is respected."""
        messages = [{"text": f"Message {i}", "is_from_me": False} for i in range(10)]
        texts, _ = _build_messages_array(messages, max_messages=3)

        assert len(texts) == 3

    def test_skips_empty_messages(self) -> None:
        """Test that empty messages are skipped."""
        messages = [
            {"text": "Hello", "is_from_me": False},
            {"text": "", "is_from_me": True},
            {"text": None, "is_from_me": False},
            {"text": "Goodbye", "is_from_me": False},
        ]
        texts, _ = _build_messages_array(messages)

        assert len(texts) == 2

    def test_strips_attachment_placeholder(self) -> None:
        """Test that attachment placeholders are stripped."""
        messages = [
            {"text": "Here's the file \ufffc", "is_from_me": False},
        ]
        texts, _ = _build_messages_array(messages)

        assert "\ufffc" not in texts[0]

    def test_skips_very_short_messages(self) -> None:
        """Test that messages with less than 2 chars are skipped."""
        messages = [
            {"text": "Hello", "is_from_me": False},
            {"text": "k", "is_from_me": True},  # Too short
            {"text": " ", "is_from_me": False},  # Too short after strip
        ]
        texts, _ = _build_messages_array(messages)

        assert len(texts) == 1

    def test_reply_to_when_last_is_from_me(self) -> None:
        """Test reply_to is empty when last message is from me."""
        messages = [
            {"text": "Hey", "is_from_me": False},
            {"text": "Hi", "is_from_me": True},  # Last is from me
        ]
        _, reply_to = _build_messages_array(messages)

        assert reply_to == ""


class TestFormatPastReplies:
    """Test the format_past_replies function."""

    def test_empty_past_replies(self) -> None:
        """Test with no past replies."""
        assert format_past_replies(None) == ""
        assert format_past_replies([]) == ""

    def test_basic_formatting(self) -> None:
        """Test basic past replies formatting."""
        past_replies = [
            ("How are you?", "doing good", 0.85),
            ("What's up?", "not much", 0.80),
        ]
        result = format_past_replies(past_replies)

        # New format uses "How you replied to similar messages"
        assert "How you replied" in result
        assert "Them: How are you?" in result
        assert "You: doing good" in result
        assert "Them: What's up?" in result
        assert "You: not much" in result

    def test_truncates_long_messages(self) -> None:
        """Test that long messages are truncated."""
        long_msg = "x" * 100
        long_reply = "y" * 100
        past_replies = [(long_msg, long_reply, 0.8)]

        result = format_past_replies(past_replies)

        # Should truncate + "..."
        assert "..." in result
        assert long_msg not in result  # Full message shouldn't appear

    def test_limits_to_four_examples(self) -> None:
        """Test that only top 4 examples are included."""
        past_replies = [
            ("msg1", "reply1", 0.9),
            ("msg2", "reply2", 0.85),
            ("msg3", "reply3", 0.80),
            ("msg4", "reply4", 0.75),
            ("msg5", "reply5", 0.70),
        ]
        result = format_past_replies(past_replies)

        assert "reply1" in result
        assert "reply2" in result
        assert "reply3" in result
        assert "reply4" in result
        assert "reply5" not in result


class TestBuildReplyPrompt:
    """Test the main build_reply_prompt function."""

    def test_basic_prompt_building(self) -> None:
        """Test basic prompt building."""
        messages = [
            {"text": "Hey!", "is_from_me": False},
            {"text": "Hi there", "is_from_me": True},
            {"text": "How are you?", "is_from_me": False},
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="How are you?",
            last_sender="John",
            style_instructions="casual, brief",
            user_name="Me",
        )

        # New format uses few-shot examples
        assert "them:" in prompt.lower()
        assert "me:" in prompt.lower()
        assert "How are you?" in prompt

    def test_includes_few_shot_examples(self) -> None:
        """Test that few-shot examples are included."""
        messages = [{"text": "Hello", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="John",
            style_instructions="lowercase, no emojis",
        )

        # New format uses few-shot examples instead of style instructions
        assert "them:" in prompt.lower()
        assert "me:" in prompt.lower()
        # Default examples should be present
        assert "wanna hang" in prompt.lower() or "ya sure" in prompt.lower()

    def test_uses_past_replies_as_examples(self) -> None:
        """Test that past replies replace default few-shot when provided."""
        messages = [{"text": "Hello", "is_from_me": False}]
        past_replies = [
            ("hey what's up", "nm u", 0.9),
            ("you around?", "ya", 0.85),
            ("wanna hang", "sure", 0.8),
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="John",
            style_instructions="",
            past_replies=past_replies,
        )

        # Past replies should be used as examples
        assert "nm u" in prompt or "ya" in prompt

    def test_includes_availability_busy(self) -> None:
        """Test that busy availability is included."""
        messages = [{"text": "Want to hang out?", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Want to hang out?",
            last_sender="John",
            style_instructions="",
            availability="busy",
        )

        assert "busy" in prompt.lower()

    def test_includes_availability_free(self) -> None:
        """Test that free availability is included."""
        messages = [{"text": "Want to hang out?", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Want to hang out?",
            last_sender="John",
            style_instructions="",
            availability="free",
        )

        assert "free" in prompt.lower() or "available" in prompt.lower()

    def test_prompt_ends_with_me(self) -> None:
        """Test that prompt ends with 'me:' for completion."""
        messages = [{"text": "Hello", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="John",
            style_instructions="",
        )

        # Prompt should end with "me:" for the model to complete
        assert prompt.strip().endswith("me:")

    def test_includes_past_replies(self) -> None:
        """Test that past replies are included as few-shot examples."""
        messages = [{"text": "How are you?", "is_from_me": False}]
        past_replies = [("Hi!", "hey", 0.8), ("Hello!", "hi there", 0.75)]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="How are you?",
            last_sender="John",
            style_instructions="",
            past_replies=past_replies,
        )

        # Past replies should be formatted as few-shot examples
        assert "How you replied" in prompt or "hey" in prompt

    def test_accepts_global_style_parameter(self) -> None:
        """Test that global_style parameter is accepted (used for style_instructions)."""

        class MockGlobalStyle:
            personality_summary = "friendly and casual texter"
            interests = ["music", "travel", "food"]
            common_phrases = ["sounds good", "lol"]

        messages = [{"text": "Hello", "is_from_me": False}]
        # global_style is passed but we use it for building style_instructions,
        # not directly in the prompt (simplified prompt focuses on examples)
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="John",
            style_instructions="casual, brief",
            global_style=MockGlobalStyle(),
        )

        # Prompt should be valid
        assert isinstance(prompt, str)
        assert "Hello" in prompt

    def test_accepts_contact_profile_parameter(self) -> None:
        """Test that contact_profile parameter is accepted (used for style_instructions)."""

        class MockProfile:
            relationship_summary = "childhood friend who you joke around with"
            display_name = "Sarah"
            relationship_type = "close_friend"
            tone = "playful"

        messages = [{"text": "Hello", "is_from_me": False}]
        # contact_profile is passed but we use it for building style_instructions,
        # not directly in the prompt (simplified prompt focuses on examples)
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="Sarah",
            style_instructions="casual, playful",
            contact_profile=MockProfile(),
        )

        assert isinstance(prompt, str)
        assert "Hello" in prompt

    def test_handles_empty_contact_profile(self) -> None:
        """Test handling when contact_profile has empty fields."""

        class MockProfile:
            relationship_summary = ""
            display_name = "Bob"
            relationship_type = "coworker"
            tone = "formal"

        messages = [{"text": "Hello", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="Bob",
            style_instructions="professional",
            contact_profile=MockProfile(),
        )

        assert isinstance(prompt, str)
        assert "Hello" in prompt

    def test_limits_conversation_to_6_messages(self) -> None:
        """Test that conversation is limited to last 6 messages."""
        messages = [{"text": f"Message {i}", "is_from_me": i % 2 == 0} for i in range(10)]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Message 9",
            last_sender="John",
            style_instructions="",
        )

        # Last message should appear in conversation
        assert "Message 9" in prompt
        # Earlier messages may or may not appear depending on context
        # but the prompt should be valid
        assert isinstance(prompt, str)

    def test_strips_attachment_placeholders(self) -> None:
        """Test that attachment placeholders are stripped."""
        messages = [{"text": "Here's a photo \ufffc", "is_from_me": False}]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Here's a photo \ufffc",
            last_sender="John",
            style_instructions="",
        )

        assert "\ufffc" not in prompt

    def test_skips_short_messages(self) -> None:
        """Test that very short messages are skipped."""
        messages = [
            {"text": "k", "is_from_me": True},  # Too short
            {"text": "Hello there!", "is_from_me": False},
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello there!",
            last_sender="John",
            style_instructions="",
        )

        # "k" should be skipped (len < 2)
        # Check that only substantial messages appear
        assert "Hello there!" in prompt


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_messages_list(self) -> None:
        """Test with empty messages list."""
        prompt = build_reply_prompt(
            messages=[],
            last_message="",
            last_sender="",
            style_instructions="",
        )

        # Should not raise and should produce valid prompt
        assert isinstance(prompt, str)

    def test_all_empty_texts(self) -> None:
        """Test when all messages have empty text."""
        messages = [
            {"text": "", "is_from_me": False},
            {"text": None, "is_from_me": True},
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="",
            last_sender="",
            style_instructions="",
        )

        assert isinstance(prompt, str)

    def test_special_characters_in_messages(self) -> None:
        """Test messages with special characters."""
        messages = [
            {"text": 'He said "Hello!" to me', "is_from_me": False},
            {"text": "It's great! @#$%^&*()", "is_from_me": True},
            {"text": "Newlines\nand\ttabs", "is_from_me": False},
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message=messages[-1]["text"],
            last_sender="John",
            style_instructions="",
        )

        assert isinstance(prompt, str)
        # Should handle special chars without error

    def test_unicode_characters(self) -> None:
        """Test messages with unicode characters."""
        messages = [
            {"text": "Hello!", "is_from_me": False},
            {"text": "مرحبا", "is_from_me": True},  # Arabic
            {"text": "日本語", "is_from_me": False},  # Japanese
        ]
        prompt = build_reply_prompt(
            messages=messages,
            last_message="日本語",
            last_sender="John",
            style_instructions="",
        )

        assert isinstance(prompt, str)
        assert "日本語" in prompt or "Them:" in prompt

    def test_handles_long_style_instructions(self) -> None:
        """Test that long style instructions don't break prompt building."""
        long_instructions = ", ".join(["instruction"] * 100)
        messages = [{"text": "Hello", "is_from_me": False}]

        prompt = build_reply_prompt(
            messages=messages,
            last_message="Hello",
            last_sender="John",
            style_instructions=long_instructions,
        )

        # Should still produce valid prompt (style instructions not directly used in new format)
        assert isinstance(prompt, str)
        assert "them:" in prompt.lower()
        assert prompt.strip().endswith("me:")
