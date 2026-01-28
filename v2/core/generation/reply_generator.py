"""Reply generator for JARVIS v2.

Orchestrates style analysis, context analysis, and LLM generation
to produce contextual reply suggestions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .context_analyzer import ContextAnalyzer, ConversationContext
from .prompts import build_reply_prompt
from .style_analyzer import StyleAnalyzer, UserStyle

logger = logging.getLogger(__name__)


@dataclass
class GeneratedReply:
    """A single generated reply option."""

    text: str
    reply_type: str
    confidence: float = 0.8


@dataclass
class ReplyGenerationResult:
    """Result of reply generation."""

    replies: list[GeneratedReply]
    context: ConversationContext
    style: UserStyle
    model_used: str
    generation_time_ms: float
    prompt_used: str = ""  # For debugging


class ReplyGenerator:
    """Generates contextual reply suggestions."""

    def __init__(self, model_loader):
        """Initialize generator.

        Args:
            model_loader: ModelLoader instance for LLM generation
        """
        self.model_loader = model_loader
        self.style_analyzer = StyleAnalyzer()
        self.context_analyzer = ContextAnalyzer()

        # Cache styles per conversation
        self._style_cache: dict[str, UserStyle] = {}

    def generate_replies(
        self,
        messages: list[dict],
        chat_id: str | None = None,
        num_replies: int = 3,
    ) -> ReplyGenerationResult:
        """Generate reply suggestions for a conversation.

        Args:
            messages: Recent messages from conversation
                     [{"text": "...", "sender": "...", "is_from_me": bool}, ...]
            chat_id: Optional conversation ID for style caching
            num_replies: Number of replies to generate

        Returns:
            ReplyGenerationResult with suggestions and metadata
        """
        start_time = time.time()

        # 1. Analyze user's texting style
        style = self._get_or_analyze_style(messages, chat_id)

        # 2. Analyze conversation context
        context = self.context_analyzer.analyze(messages)

        # 3. Get reply strategy
        strategy = self.context_analyzer.get_reply_strategy(context)

        # 4. Build prompt
        style_instructions = self.style_analyzer.to_prompt_instructions(style)

        prompt = build_reply_prompt(
            messages=messages,
            last_message=context.last_message,
            last_sender=context.last_sender,
            style_instructions=style_instructions,
            reply_types=strategy.reply_types,
            tone=strategy.tone,
            max_length=strategy.max_length,
            intent_value=context.intent.value,
        )

        # 5. Generate with LLM
        try:
            result = self.model_loader.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.8,
                stop=["\n\n", "4.", "##"],
            )
            raw_output = result.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return fallback replies
            return self._fallback_result(context, style, str(e))

        # 6. Parse replies
        replies = self._parse_replies(raw_output, strategy.reply_types)

        # Ensure we have enough replies
        if len(replies) < num_replies:
            replies.extend(self._get_fallback_replies(context.intent.value, num_replies - len(replies)))

        generation_time = (time.time() - start_time) * 1000

        return ReplyGenerationResult(
            replies=replies[:num_replies],
            context=context,
            style=style,
            model_used=self.model_loader.current_model,
            generation_time_ms=generation_time,
            prompt_used=prompt,
        )

    def _get_or_analyze_style(
        self, messages: list[dict], chat_id: str | None
    ) -> UserStyle:
        """Get cached style or analyze from messages."""
        if chat_id and chat_id in self._style_cache:
            return self._style_cache[chat_id]

        # Filter to user's messages only
        user_messages = [m for m in messages if m.get("is_from_me")]
        style = self.style_analyzer.analyze(user_messages)

        if chat_id:
            self._style_cache[chat_id] = style

        return style

    def _parse_replies(
        self, raw_output: str, reply_types: list[str]
    ) -> list[GeneratedReply]:
        """Parse LLM output into structured replies."""
        replies = []

        # Clean up output - stop at repetition or double newline
        output = raw_output.strip()
        if "\n\n" in output:
            output = output.split("\n\n")[0]

        lines = output.split("\n")

        for i, line in enumerate(lines):
            text = line.strip()

            # Skip empty lines
            if not text:
                continue

            # Remove common prefixes (including emoji bullets)
            import re
            text = re.sub(r'^[\d]+[.\)]\s*', '', text)  # Remove "1." or "1)"
            text = re.sub(r'^[-*•]\s*', '', text)  # Remove bullet points
            text = text.strip()

            # Skip if still empty or too short
            if len(text) < 2:
                continue

            # Skip meta-text
            skip_patterns = ["here are", "option", "reply", "response", "example", "____"]
            if any(p in text.lower() for p in skip_patterns):
                continue

            # Remove surrounding quotes
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]

            # Skip if too long (probably not a reply)
            if len(text) > 150:
                continue

            reply_type = reply_types[i] if i < len(reply_types) else "general"

            replies.append(GeneratedReply(
                text=text,
                reply_type=reply_type,
                confidence=0.8 - (i * 0.1),  # Slightly lower confidence for later replies
            ))

            if len(replies) >= 3:
                break

        return replies

    def _get_fallback_replies(self, intent_value: str, count: int) -> list[GeneratedReply]:
        """Get fallback replies when generation fails or is incomplete."""
        fallbacks = {
            "yes_no_question": ["sounds good!", "can't right now, sorry", "let me check"],
            "open_question": ["not sure yet", "good question", "let me think about it"],
            "statement": ["got it", "cool", "nice"],
            "emotional": ["that's understandable", "i hear you", "hope things get better"],
            "greeting": ["hey!", "hi there", "what's up"],
            "logistics": ["sounds good", "got it", "on my way"],
            "thanks": ["no problem!", "anytime", "you're welcome"],
            "farewell": ["bye!", "talk soon", "later"],
        }

        replies_text = fallbacks.get(intent_value, fallbacks["statement"])
        return [
            GeneratedReply(text=text, reply_type="fallback", confidence=0.5)
            for text in replies_text[:count]
        ]

    def _fallback_result(
        self,
        context: ConversationContext,
        style: UserStyle,
        error: str,
    ) -> ReplyGenerationResult:
        """Create fallback result when generation fails."""
        replies = self._get_fallback_replies(context.intent.value, 3)

        return ReplyGenerationResult(
            replies=replies,
            context=context,
            style=style,
            model_used="fallback",
            generation_time_ms=0,
            prompt_used=f"ERROR: {error}",
        )

    def clear_style_cache(self, chat_id: str | None = None) -> None:
        """Clear cached style analysis.

        Args:
            chat_id: Specific chat to clear, or None for all
        """
        if chat_id:
            self._style_cache.pop(chat_id, None)
        else:
            self._style_cache.clear()
