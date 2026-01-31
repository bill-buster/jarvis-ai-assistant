"""Simple Reply Generator - Minimal approach to text reply generation.

The simplest possible approach:
1. Show the LLM the conversation (user's messages already demonstrate their style)
2. Tell it to respond as the user
3. If uncertain about something, ask instead of guessing

No FAISS, no intent classification, no complex routing.
Just conversation context + simple prompt.

Usage:
    from jarvis.simple_reply import generate_reply, SimpleReplyGenerator

    # Quick usage
    result = generate_reply(
        conversation=[
            ("them", "you down for lunch?"),
            ("me", "maybe, depends on timing"),
            ("them", "how about 2pm?"),
        ]
    )
    print(result["response"])  # or result["question"] if it needs info

    # With generator instance
    gen = SimpleReplyGenerator()
    result = gen.reply(conversation, last_message="how about 2pm?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimpleReplyResult:
    """Result from simple reply generation."""

    response: str | None  # The reply, if confident
    question: str | None  # Question to ask user, if needs info
    confidence: str  # "high", "medium", "low"
    raw_output: str  # Raw LLM output for debugging


# The simplest possible prompt - v9: show style examples then ask
SIMPLE_PROMPT = """Look at how "Me" texts in this conversation:

{conversation}

Now write what "Me" would say next. Match their exact style - same length, \
same slang, no greetings like "Hey!" or encouraging phrases like "Let's go!". \
Just a normal reply."""


def format_conversation(
    messages: list[tuple[str, str]],
    my_name: str = "Me",
    their_name: str = "Them",
) -> str:
    """Format conversation for the prompt.

    Args:
        messages: List of (speaker, text) tuples. speaker is "me" or "them".
        my_name: Display name for user's messages.
        their_name: Display name for other person's messages.

    Returns:
        Formatted conversation string.
    """
    lines = []
    for speaker, text in messages:
        name = my_name if speaker.lower() == "me" else their_name
        # Handle multi-line messages
        text_lines = text.strip().split("\n")
        for i, line in enumerate(text_lines):
            if i == 0:
                lines.append(f"[{name}]: {line}")
            else:
                lines.append(f"       {line}")  # Indent continuation
    return "\n".join(lines)


def parse_response(raw: str) -> tuple[str | None, str | None, str]:
    """Parse LLM output to determine if it's a reply or a question.

    Returns:
        Tuple of (response, question, confidence).
        - If response is set, it's a reply to send.
        - If question is set, we need more info from the user.
    """
    text = raw.strip()

    # Check for explicit [NEED INFO] marker
    if "[NEED INFO:" in text.upper():
        # Extract what info is needed
        import re

        match = re.search(r"\[NEED INFO:\s*(.+?)\]", text, re.IGNORECASE)
        if match:
            return None, match.group(1).strip(), "low"
        return None, text, "low"

    # Clean up any trailing explanation the model might add
    # Sometimes models add "..." or explanations after the response
    lines = text.split("\n")
    if lines:
        text = lines[0].strip()  # Just take the first line

    # Remove [Me]: or Me: prefix if model added it
    import re

    text = re.sub(r"^\[?Me\]?:\s*", "", text, flags=re.IGNORECASE)

    # Remove quotes if the model wrapped the response
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Remove trailing punctuation artifacts
    text = text.rstrip("]").strip()

    # Confidence based on length
    word_count = len(text.split())

    if word_count < 1:
        return None, "What should I say?", "low"
    elif word_count > 30:
        confidence = "medium"  # Might be too long
    else:
        confidence = "high"

    return text, None, confidence


class SimpleReplyGenerator:
    """Minimal reply generator using just conversation context."""

    def __init__(self, model_id: str | None = None):
        """Initialize generator.

        Args:
            model_id: Optional model ID. Uses default if not specified.
        """
        self._generator = None
        self._model_id = model_id

    @property
    def generator(self):
        """Lazy-load the MLX generator."""
        if self._generator is None:
            from models import get_generator

            self._generator = get_generator(skip_templates=True)
        return self._generator

    def reply(
        self,
        conversation: list[tuple[str, str]],
        last_message: str | None = None,
        contact_name: str | None = None,
    ) -> SimpleReplyResult:
        """Generate a reply to a conversation.

        Args:
            conversation: List of (speaker, text) tuples.
                         speaker should be "me" or "them".
            last_message: Optional explicit last message (if not in conversation).
            contact_name: Optional name for the other person.

        Returns:
            SimpleReplyResult with response or question.
        """
        from contracts.models import GenerationRequest

        # Add last_message to conversation if provided separately
        if last_message and (not conversation or conversation[-1][1] != last_message):
            conversation = list(conversation) + [("them", last_message)]

        # Format conversation
        their_name = contact_name or "Them"
        conv_text = format_conversation(conversation, their_name=their_name)

        # Build prompt
        prompt = SIMPLE_PROMPT.format(conversation=conv_text)

        # Generate
        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],  # Not using RAG
            few_shot_examples=[],  # Style is in the conversation itself
            max_tokens=25,  # Short but enough to answer
            temperature=0.6,
            top_p=0.9,
        )

        response = self.generator.generate(request)
        raw_output = response.text.strip()

        # Parse response
        reply_text, question, confidence = parse_response(raw_output)

        return SimpleReplyResult(
            response=reply_text,
            question=question,
            confidence=confidence,
            raw_output=raw_output,
        )


# Module-level singleton
_simple_generator: SimpleReplyGenerator | None = None


def get_simple_generator() -> SimpleReplyGenerator:
    """Get singleton SimpleReplyGenerator instance."""
    global _simple_generator
    if _simple_generator is None:
        _simple_generator = SimpleReplyGenerator()
    return _simple_generator


def generate_reply(
    conversation: list[tuple[str, str]],
    last_message: str | None = None,
    contact_name: str | None = None,
) -> dict[str, Any]:
    """Quick function to generate a reply.

    Args:
        conversation: List of (speaker, text) tuples.
        last_message: Optional explicit last message.
        contact_name: Optional name for the other person.

    Returns:
        Dict with "response" or "question", plus "confidence" and "raw".
    """
    gen = get_simple_generator()
    result = gen.reply(conversation, last_message, contact_name)

    return {
        "response": result.response,
        "question": result.question,
        "confidence": result.confidence,
        "raw": result.raw_output,
    }


# =============================================================================
# Convenience: Load from iMessage directly
# =============================================================================


def generate_reply_for_chat(
    chat_id: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Generate a reply for a chat by loading recent messages from iMessage.

    Args:
        chat_id: The iMessage chat ID.
        limit: Number of recent messages to include.

    Returns:
        Dict with "response" or "question", plus metadata.
    """
    from integrations.imessage.reader import ChatDBReader

    reader = ChatDBReader()
    messages = reader.get_messages(chat_id, limit=limit)

    if not messages:
        return {
            "response": None,
            "question": "I couldn't find any messages in this conversation.",
            "confidence": "low",
            "raw": "",
        }

    # Convert to conversation format
    conversation = []
    for msg in reversed(messages):  # Oldest first
        speaker = "me" if msg.is_from_me else "them"
        if msg.text:
            conversation.append((speaker, msg.text))

    return generate_reply(conversation)
