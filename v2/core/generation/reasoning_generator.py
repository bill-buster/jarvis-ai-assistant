"""Reasoning-first reply generator.

Instead of blindly generating a response, the model first reasons about:
1. What is being asked/expected?
2. Do I have enough context to respond well?
3. If not, what do I need to know?

This prevents hallucination and lets smaller models perform better by
knowing when to ask for help.
"""

from dataclasses import dataclass
from enum import Enum


class ResponseType(Enum):
    """What the model decided to do."""
    REPLY = "reply"           # Model is confident, generated a response
    NEED_INFO = "need_info"   # Model needs more information from user
    NEED_INTENT = "need_intent"  # Model needs to know user's intent (accept/decline/etc)
    UNCERTAIN = "uncertain"   # Model isn't sure, offering options


@dataclass
class ReasonedResponse:
    """Response from the reasoning generator."""
    response_type: ResponseType

    # If REPLY: the actual response
    reply: str | None = None

    # If NEED_INFO or NEED_INTENT: what to ask the user
    question_for_user: str | None = None

    # Options to present (for NEED_INTENT or UNCERTAIN)
    options: list[str] | None = None

    # The model's reasoning (for debugging/transparency)
    reasoning: str | None = None

    # Confidence score (0-1)
    confidence: float = 0.0


# Reasoning prompt - asks model to think before responding
REASONING_PROMPT = """You are helping me reply to a text conversation. Before responding, think through:

1. SITUATION: What's happening in this conversation? What are they asking/saying?
2. NEEDED INFO: What would I need to know to give a good reply?
3. DECISION: Can you respond confidently, or do you need to ask me something?

Rules:
- If you're confident you can give a good casual reply, respond with REPLY: <your response>
- If you need to know my intent (like accept/decline an invite), respond with ASK_INTENT: <question for me>
- If you need specific info you don't have, respond with ASK_INFO: <what you need to know>
- Keep replies casual, short, lowercase, match texting style

Conversation:
{conversation}
me:

Think step by step, then respond:"""


# Simpler prompt for faster inference
SIMPLE_REASONING_PROMPT = """Reply to this text conversation as me.
- If you can reply confidently: just reply
- If you need to know something first: ask me

{conversation}
me:

[If you need info, start with "?" then your question. Otherwise just reply]
"""


# Even simpler - structured output
STRUCTURED_PROMPT = """Analyze this conversation and help me reply.

{conversation}

Output ONE of these:
REPLY: <casual short reply in my style>
ASK: <question you need answered first>

Output:"""


def build_reasoning_prompt(conversation: str, style: str = "structured") -> str:
    """Build the reasoning prompt."""
    if style == "full":
        return REASONING_PROMPT.format(conversation=conversation)
    elif style == "simple":
        return SIMPLE_REASONING_PROMPT.format(conversation=conversation)
    else:  # structured
        return STRUCTURED_PROMPT.format(conversation=conversation)


def parse_reasoning_response(raw_output: str) -> ReasonedResponse:
    """Parse the model's reasoning output into structured response."""
    output = raw_output.strip()
    output_lower = output.lower()

    # Check for ASK patterns
    if output_lower.startswith("ask:") or output_lower.startswith("ask_info:") or output_lower.startswith("ask_intent:"):
        # Model is asking for information
        question = output.split(":", 1)[1].strip() if ":" in output else output

        # Determine if it's intent vs info based on keywords
        intent_keywords = ["accept", "decline", "want to", "should i", "yes or no", "going to"]
        is_intent = any(kw in question.lower() for kw in intent_keywords)

        return ReasonedResponse(
            response_type=ResponseType.NEED_INTENT if is_intent else ResponseType.NEED_INFO,
            question_for_user=question,
            reasoning=raw_output,
            confidence=0.3,
        )

    # Check for REPLY patterns
    if output_lower.startswith("reply:"):
        reply = output.split(":", 1)[1].strip() if ":" in output else output
        return ReasonedResponse(
            response_type=ResponseType.REPLY,
            reply=clean_reply(reply),
            reasoning=raw_output,
            confidence=0.8,
        )

    # Check for ? prefix (simple prompt format)
    if output.startswith("?"):
        question = output[1:].strip()
        return ReasonedResponse(
            response_type=ResponseType.NEED_INFO,
            question_for_user=question,
            reasoning=raw_output,
            confidence=0.5,
        )

    # No prefix - assume it's a direct reply
    return ReasonedResponse(
        response_type=ResponseType.REPLY,
        reply=clean_reply(output),
        reasoning=raw_output,
        confidence=0.6,
    )


def clean_reply(text: str) -> str:
    """Clean up the reply text."""
    if not text:
        return ""

    # Remove quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Remove common prefixes
    for prefix in ["me:", "Me:", "Response:", "Reply:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Take first line only
    text = text.split("\n")[0].strip()

    return text


class ReasoningGenerator:
    """Generator that reasons before responding."""

    def __init__(self, model_loader=None):
        self._loader = model_loader

    def _get_loader(self):
        """Lazy load the model."""
        if self._loader is None:
            from core.models.loader import ModelLoader
            self._loader = ModelLoader()
            self._loader.preload()
        return self._loader

    def generate(
        self,
        conversation: str,
        prompt_style: str = "structured",
        max_tokens: int = 60,
        temperature: float = 0.3,
    ) -> ReasonedResponse:
        """Generate a reasoned response.

        Args:
            conversation: The conversation context (them:/me: format)
            prompt_style: "structured", "simple", or "full"
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            ReasonedResponse with either a reply or a question for the user
        """
        loader = self._get_loader()

        prompt = build_reasoning_prompt(conversation, style=prompt_style)

        result = loader.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "them:", "Them:", "<|im_end|>", "<|eot_id|>"],
        )

        return parse_reasoning_response(result.text)

    def generate_with_info(
        self,
        conversation: str,
        user_info: str,
        prompt_style: str = "structured",
    ) -> ReasonedResponse:
        """Generate after user provided additional info.

        Args:
            conversation: The conversation context
            user_info: The info/intent the user provided

        Returns:
            ReasonedResponse (should be a REPLY now)
        """
        # Add the user's info as context
        enhanced_prompt = f"""Context: {user_info}

{conversation}

Now reply (short, casual, in my texting style):"""

        loader = self._get_loader()
        result = loader.generate(
            prompt=enhanced_prompt,
            max_tokens=40,
            temperature=0.3,
            stop=["\n", "them:", "<|im_end|>", "<|eot_id|>"],
        )

        return ReasonedResponse(
            response_type=ResponseType.REPLY,
            reply=clean_reply(result.text),
            confidence=0.85,
        )


# Singleton
_reasoning_generator: ReasoningGenerator | None = None


def get_reasoning_generator() -> ReasoningGenerator:
    """Get the singleton reasoning generator."""
    global _reasoning_generator
    if _reasoning_generator is None:
        _reasoning_generator = ReasoningGenerator()
    return _reasoning_generator


# Quick test
if __name__ == "__main__":
    # Test parsing
    test_outputs = [
        "REPLY: yeah sounds good",
        "ASK: do you want to accept or decline?",
        "ASK_INFO: what time works for you?",
        "?are you free right now",
        "lol ok bet",  # No prefix, should be treated as reply
    ]

    print("Testing response parsing:")
    print("=" * 50)

    for output in test_outputs:
        result = parse_reasoning_response(output)
        print(f"Input: '{output}'")
        print(f"  Type: {result.response_type.value}")
        print(f"  Reply: {result.reply}")
        print(f"  Question: {result.question_for_user}")
        print()
