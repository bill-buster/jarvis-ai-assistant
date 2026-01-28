"""Reply generator for JARVIS v2.

Orchestrates style analysis, context analysis, and LLM generation
to produce contextual reply suggestions.

Now with style learning from your past messages!
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .context_analyzer import ContextAnalyzer, ConversationContext
from .prompts import build_reply_prompt
from .style_analyzer import StyleAnalyzer, UserStyle

logger = logging.getLogger(__name__)


def _get_embedding_store():
    """Lazy import to avoid circular dependencies."""
    try:
        from core.embeddings import get_embedding_store
        return get_embedding_store()
    except Exception as e:
        logger.debug(f"Embedding store not available: {e}")
        return None


def _get_contact_profile(chat_id: str, include_topics: bool = True):
    """Lazy import contact profiler."""
    try:
        from core.embeddings import get_contact_profile
        return get_contact_profile(chat_id, include_topics=include_topics)
    except Exception as e:
        logger.debug(f"Contact profile not available: {e}")
        return None


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
    prompt_used: str = ""  # For debugging (our template)
    formatted_prompt: str = ""  # The actual ChatML prompt sent to model
    style_instructions: str = ""  # What style was used
    past_replies: list = field(default_factory=list)  # Past replies found


class ReplyGenerator:
    """Generates contextual reply suggestions."""

    # Max recent generations to track per chat
    MAX_RECENT_GENERATIONS = 5

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

        # Track recent generations per chat to avoid repetition
        self._recent_generations: dict[str, list[str]] = {}

    def _record_generation(self, chat_id: str, reply: str) -> None:
        """Record a generated reply for repetition tracking.

        Args:
            chat_id: Conversation ID
            reply: Generated reply text
        """
        recent = self._recent_generations.setdefault(chat_id, [])
        recent.append(reply.lower().strip())
        # Keep only last N generations
        if len(recent) > self.MAX_RECENT_GENERATIONS:
            recent.pop(0)

    def _is_repetitive(self, reply: str, chat_id: str | None) -> bool:
        """Check if a reply is repetitive (recently used).

        Args:
            reply: Reply text to check
            chat_id: Conversation ID

        Returns:
            True if reply was recently used in this conversation
        """
        if not chat_id:
            return False
        recent = self._recent_generations.get(chat_id, [])
        reply_lower = reply.lower().strip()
        return reply_lower in recent

    def _filter_repetitive(
        self, replies: list[GeneratedReply], chat_id: str | None
    ) -> list[GeneratedReply]:
        """Filter out repetitive replies.

        Args:
            replies: List of generated replies
            chat_id: Conversation ID

        Returns:
            Filtered list with non-repetitive replies
        """
        if not chat_id:
            return replies
        return [r for r in replies if not self._is_repetitive(r.text, chat_id)]

    def generate_replies(
        self,
        messages: list[dict],
        chat_id: str | None = None,
        num_replies: int = 3,
        user_name: str = "me",
    ) -> ReplyGenerationResult:
        """Generate reply suggestions for a conversation.

        Args:
            messages: Recent messages from conversation
                     [{"text": "...", "sender": "...", "is_from_me": bool}, ...]
            chat_id: Optional conversation ID for style caching
            num_replies: Number of replies to generate
            user_name: User's name for personalized context

        Returns:
            ReplyGenerationResult with suggestions and metadata
        """
        self._user_name = user_name
        start_time = time.time()
        timings: dict[str, float] = {}

        # 0. Filter to coherent segment (detect topic breaks)
        t0 = time.time()
        coherent_messages = self._get_coherent_messages(messages)
        timings["coherence_filter"] = (time.time() - t0) * 1000

        # 1. Analyze user's texting style (use all messages for style)
        t0 = time.time()
        style = self._get_or_analyze_style(messages, chat_id)
        timings["style_analysis"] = (time.time() - t0) * 1000

        # 2. Analyze conversation context (use coherent segment only)
        t0 = time.time()
        context = self.context_analyzer.analyze(coherent_messages)
        timings["context_analysis"] = (time.time() - t0) * 1000

        # 3. Get reply strategy
        t0 = time.time()
        strategy = self.context_analyzer.get_reply_strategy(context)
        timings["strategy"] = (time.time() - t0) * 1000

        # 4. Find YOUR past replies to similar messages (style learning!)
        # Only search if:
        # - Message is substantive (> 10 chars) - short messages have poor semantic similarity
        # - FAISS index is preloaded (to avoid 5+ second delay)
        t0 = time.time()
        past_replies = []
        if len(context.last_message) > 10 and chat_id:
            past_replies = self._find_past_replies(context.last_message, chat_id)
        timings["past_replies_lookup"] = (time.time() - t0) * 1000

        # 4b. Template matching - if past replies are very similar and consistent, skip LLM
        template_reply = self._try_template_match(past_replies)
        if template_reply:
            logger.info(f"Template match found - skipping LLM")
            generation_time = (time.time() - start_time) * 1000
            return ReplyGenerationResult(
                replies=[GeneratedReply(text=template_reply, reply_type="template", confidence=0.95)],
                context=context,
                style=style,
                model_used="template",
                generation_time_ms=generation_time,
                prompt_used="[Template match - LLM skipped]",
                style_instructions=self._build_style_instructions(style, None),
                past_replies=past_replies or [],
            )

        # 5. Get contact profile for better style info (from ALL your messages, not just recent)
        # Skip topic extraction (LLM call) - only need style info for generation
        # Profile is cached for fast retrieval after first computation
        t0 = time.time()
        profile = None
        if chat_id:
            profile = _get_contact_profile(chat_id, include_topics=False)
        timings["contact_profile"] = (time.time() - t0) * 1000

        # 6. Build style instructions - prefer contact profile (more comprehensive)
        style_instructions = self._build_style_instructions(style, profile)

        # Get recent topics for context continuity (from cached profile or skip)
        recent_topics = None
        if profile and profile.topics:
            recent_topics = [t.name for t in profile.topics[:3]]

        prompt = build_reply_prompt(
            messages=coherent_messages,  # Use coherent segment, not all messages
            last_message=context.last_message,
            last_sender=context.last_sender,
            style_instructions=style_instructions,
            reply_types=strategy.reply_types,
            tone=strategy.tone,
            max_length=strategy.max_length,
            intent_value=context.intent.value,
            past_replies=past_replies,
            user_name=self._user_name,
            recent_topics=recent_topics,
        )

        # 7. Generate with LLM
        t0 = time.time()
        formatted_prompt = ""  # Will store the actual ChatML prompt for debugging
        try:
            result = self.model_loader.generate(
                prompt=prompt,
                max_tokens=30,  # Only need 1 short reply
                temperature=0.2,  # LFM2.5 recommends 0.1, use 0.2 for slight variety
                stop=["\n", "2.", "2)", "##", "Note:", "---"],
            )
            raw_output = result.text
            formatted_prompt = result.formatted_prompt  # The actual ChatML prompt
            timings["llm_generation"] = (time.time() - t0) * 1000
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return fallback replies
            return self._fallback_result(context, style, str(e))

        # 8. Parse replies - strip emojis if profile says no emojis
        t0 = time.time()
        strip_emojis = profile and not profile.uses_emoji
        replies = self._parse_replies(raw_output, strategy.reply_types, strip_emojis=strip_emojis)
        timings["parse_replies"] = (time.time() - t0) * 1000

        # 9. Filter out repetitive replies (recently used in this conversation)
        replies = self._filter_repetitive(replies, chat_id)

        # Ensure we have enough replies
        if len(replies) < num_replies:
            fallbacks = self._get_fallback_replies(
                context.intent.value, num_replies - len(replies), chat_id
            )
            # Filter fallbacks too
            fallbacks = self._filter_repetitive(fallbacks, chat_id)
            replies.extend(fallbacks)

        # Record generations for future repetition checking
        if chat_id:
            for reply in replies[:num_replies]:
                self._record_generation(chat_id, reply.text)

        generation_time = (time.time() - start_time) * 1000

        # Log timing breakdown
        logger.info(
            f"Generation completed in {generation_time:.0f}ms - "
            f"coherence:{timings.get('coherence_filter', 0):.0f}ms, "
            f"style:{timings.get('style_analysis', 0):.0f}ms, "
            f"context:{timings.get('context_analysis', 0):.0f}ms, "
            f"profile:{timings.get('contact_profile', 0):.0f}ms, "
            f"LLM:{timings.get('llm_generation', 0):.0f}ms, "
            f"parse:{timings.get('parse_replies', 0):.0f}ms"
        )

        return ReplyGenerationResult(
            replies=replies[:num_replies],
            context=context,
            style=style,
            model_used=self.model_loader.current_model,
            generation_time_ms=generation_time,
            prompt_used=prompt,
            formatted_prompt=formatted_prompt,
            style_instructions=style_instructions,
            past_replies=past_replies or [],
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
        self, raw_output: str, reply_types: list[str], strip_emojis: bool = False
    ) -> list[GeneratedReply]:
        """Parse LLM output into structured replies."""
        import re

        replies = []
        output = raw_output.strip()

        # Take first line only (the reply)
        text = output.split("\n")[0].strip()

        # Remove common prefixes/artifacts
        for prefix in ["Reply:", "Response:", "Answer:", f"{self._user_name}:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove quotes if wrapped
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        # Strip emojis if style says no emojis (model often ignores this instruction)
        if strip_emojis:
            # Remove emoji characters
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map
                "\U0001F1E0-\U0001F1FF"  # flags
                "\U00002702-\U000027B0"  # dingbats
                "\U0001F900-\U0001F9FF"  # supplemental symbols
                "\U0001FA00-\U0001FA6F"  # chess symbols
                "\U0001FA70-\U0001FAFF"  # symbols extended
                "\U00002600-\U000026FF"  # misc symbols
                "]+",
                flags=re.UNICODE
            )
            text = emoji_pattern.sub("", text)

        # Clean up
        text = text.strip()

        if text and len(text) >= 2 and len(text) <= 150:
            replies.append(GeneratedReply(
                text=text,
                reply_type=reply_types[0] if reply_types else "general",
                confidence=0.9,
            ))

        return replies

    def _get_fallback_replies(
        self, intent_value: str, count: int, chat_id: str | None = None
    ) -> list[GeneratedReply]:
        """Get fallback replies when generation fails or is incomplete.

        Tries personal templates first (from user's actual message history),
        then falls back to generic templates.
        """
        # Try personal templates first
        if chat_id:
            personal = self._get_personal_templates(intent_value, chat_id, count)
            if personal:
                return personal

        # Fall back to generic templates
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

    def _get_personal_templates(
        self, intent_value: str, chat_id: str, count: int
    ) -> list[GeneratedReply]:
        """Get personalized fallback templates from user's message history.

        Args:
            intent_value: Detected intent type
            chat_id: Conversation ID
            count: Number of replies needed

        Returns:
            List of personalized replies, or empty if none found
        """
        store = _get_embedding_store()
        if not store:
            return []

        # Map intent to response pattern category
        intent_to_pattern = {
            "yes_no_question": "affirmative",  # Default to affirmative
            "greeting": "greeting",
            "thanks": "thanks",
            "statement": "acknowledgment",
            "logistics": "acknowledgment",
        }

        pattern_key = intent_to_pattern.get(intent_value)
        if not pattern_key:
            return []

        try:
            patterns = store.get_user_response_patterns(chat_id)
            replies = patterns.get(pattern_key, [])

            if not replies:
                return []

            return [
                GeneratedReply(text=text, reply_type="personal_template", confidence=0.7)
                for text in replies[:count]
            ]
        except Exception as e:
            logger.debug(f"Failed to get personal templates: {e}")
            return []

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
        """Clear cached style analysis and recent generations.

        Args:
            chat_id: Specific chat to clear, or None for all
        """
        if chat_id:
            self._style_cache.pop(chat_id, None)
            self._recent_generations.pop(chat_id, None)
        else:
            self._style_cache.clear()
            self._recent_generations.clear()

    def _get_coherent_messages(self, messages: list[dict]) -> list[dict]:
        """Get relevant context for reply generation.

        Strategy:
        1. Find the last substantive message from them (not just a reaction)
        2. Include context from that point forward
        3. Also include what prompted your last reply (the thread context)
        """
        if len(messages) <= 4:
            return messages

        # Detect if group chat
        senders = set()
        for msg in messages[-20:]:
            if not msg.get("is_from_me"):
                sender = msg.get("sender") or msg.get("sender_name") or "unknown"
                senders.add(sender)

        is_group_chat = len(senders) > 1

        if is_group_chat:
            logger.debug(f"Group chat ({len(senders)} senders), using last 5 messages")
            return messages[-5:] if len(messages) > 5 else messages

        # 1:1 chat - find the conversation thread
        # Look for your last reply and include what you were replying to
        result = []
        found_your_reply = False
        include_from_idx = max(0, len(messages) - 6)  # Default: last 6

        # Walk backwards to find context
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]

            if msg.get("is_from_me") and not found_your_reply:
                # Found your last reply - include what came before it
                found_your_reply = True
                # Look for the message you were replying to
                for j in range(i - 1, max(0, i - 4), -1):
                    if not messages[j].get("is_from_me"):
                        include_from_idx = j
                        break

            if found_your_reply and i <= include_from_idx:
                break

        # Include from the context point to end
        result = messages[include_from_idx:]

        # Limit to 8 messages max
        if len(result) > 8:
            result = result[-8:]

        logger.debug(f"1:1 chat, using {len(result)} messages (thread context)")
        return result

    def _build_style_instructions(self, style: UserStyle, profile) -> str:
        """Build style instructions combining style analysis and contact profile.

        The contact profile is more comprehensive (analyzes ALL messages),
        while style analysis looks at recent messages only.
        """
        instructions = []

        # Use contact profile data if available (more comprehensive)
        if profile and profile.total_messages > 10:
            # Length from profile
            avg_len = profile.avg_your_length
            if avg_len < 20:
                instructions.append("very short replies (under 5 words)")
            elif avg_len < 40:
                instructions.append("brief replies (under 10 words)")
            else:
                instructions.append("medium length replies okay")

            # Emoji usage from profile - only mention if allowed (avoid negative priming)
            if profile.uses_emoji:
                instructions.append("emojis okay")

            # Tone from profile
            if profile.tone == "casual":
                instructions.append("casual tone")
            elif profile.tone == "formal":
                instructions.append("more formal tone")

            # Slang from profile
            if profile.uses_slang:
                instructions.append("abbreviations okay (u, ur, sm, lol)")

        else:
            # Fall back to style analyzer output
            base_instructions = self.style_analyzer.to_prompt_instructions(style)
            if base_instructions:
                return base_instructions

            # Default if nothing available
            instructions.append("casual and brief")

        # Add capitalization from style analysis (more granular)
        if style.capitalization == "lowercase":
            instructions.append("lowercase only")
        elif style.punctuation_style == "minimal":
            instructions.append("minimal punctuation")

        return ", ".join(instructions)

    def _find_past_replies(
        self,
        incoming_message: str,
        chat_id: str | None,
    ) -> list[tuple[str, str, float]]:
        """Find YOUR past replies to similar incoming messages.

        Args:
            incoming_message: The message to find similar responses for
            chat_id: Optional conversation filter

        Returns:
            List of (their_message, your_reply, similarity) tuples
        """
        store = _get_embedding_store()
        if not store:
            return []

        try:
            # Get more examples for better context (was 3, now 5)
            return store.find_your_past_replies(
                incoming_message=incoming_message,
                chat_id=chat_id,
                limit=5,
                min_similarity=0.60,  # Slightly lower threshold to get more examples
            )
        except Exception as e:
            logger.debug(f"Failed to find past replies: {e}")
            return []

    def _try_template_match(
        self,
        past_replies: list[tuple[str, str, float]] | None,
    ) -> str | None:
        """Check if past replies are consistent enough to skip LLM.

        If user consistently responds the same way to similar messages,
        just return that response directly.

        Args:
            past_replies: List of (their_message, your_reply, similarity) tuples

        Returns:
            Template response if confident, None otherwise
        """
        if not past_replies or len(past_replies) < 2:
            return None

        # Check if top replies are very similar (>75% similarity)
        high_confidence = [r for r in past_replies if r[2] >= 0.75]
        if len(high_confidence) < 2:
            return None

        # Check if your responses are consistent
        responses = [r[1].lower().strip() for r in high_confidence]

        # If all responses are identical or very short and similar
        if len(set(responses)) == 1:
            # All identical - high confidence
            return high_confidence[0][1]

        # Check if responses are semantically similar (e.g., "yes", "yea", "yeah")
        yes_variants = {"yes", "yea", "yeah", "yep", "ya", "ye", "sure", "ok", "okay", "k"}
        no_variants = {"no", "nah", "nope", "cant", "can't", "cannot"}

        response_set = set(responses)

        if response_set.issubset(yes_variants):
            return high_confidence[0][1]  # Return the actual one they used
        if response_set.issubset(no_variants):
            return high_confidence[0][1]

        return None

    def _format_style_examples(
        self,
        past_replies: list[tuple[str, str, float]],
    ) -> str:
        """Format past replies as style examples for the prompt.

        Args:
            past_replies: List of (their_message, your_reply, similarity)

        Returns:
            Formatted string for prompt
        """
        if not past_replies:
            return ""

        examples = []
        for their_msg, your_reply, similarity in past_replies:
            examples.append(f'- When they said "{their_msg}", you replied "{your_reply}"')

        return "\n".join(examples)
