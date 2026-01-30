"""Reply generator for JARVIS v2.

Orchestrates style analysis, context analysis, and LLM generation
to produce contextual reply suggestions.

Now with style learning from your past messages!
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from core.config import PromptStrategy, settings

from .context_analyzer import ContextAnalyzer, ConversationContext, RelationshipType
from .global_styler import get_global_user_style
from .prompts import (
    build_conversation_prompt,
    build_rag_reply_prompt,
    build_reply_prompt,
    build_threaded_reply_prompt,
)
from .style_analyzer import StyleAnalyzer, UserStyle

logger = logging.getLogger(__name__)


def _get_template_matcher():
    """Lazy import to avoid circular dependencies."""
    try:
        from core.templates import get_template_matcher

        return get_template_matcher(preload=True)
    except ImportError:
        logger.debug("Template matcher not installed")
        return None
    except Exception as e:
        logger.warning(f"Template matcher failed unexpectedly: {e}")
        return None


def _get_embedding_store():
    """Lazy import to avoid circular dependencies."""
    try:
        from core.embeddings import get_embedding_store

        return get_embedding_store()
    except ImportError:
        logger.debug("Embedding store not installed")
        return None
    except Exception as e:
        logger.warning(f"Embedding store failed unexpectedly: {e}")
        return None


def _get_contact_profile(chat_id: str, include_topics: bool = True):
    """Lazy import contact profiler."""
    try:
        from core.embeddings import get_contact_profile

        return get_contact_profile(chat_id, include_topics=include_topics)
    except ImportError:
        logger.debug("Contact profiler not installed")
        return None
    except Exception as e:
        logger.warning(f"Contact profile failed unexpectedly: {e}")
        return None


def _get_relationship_registry():
    """Lazy import relationship registry."""
    try:
        from core.embeddings.relationship_registry import get_relationship_registry

        return get_relationship_registry()
    except ImportError:
        logger.debug("Relationship registry not installed")
        return None
    except Exception as e:
        logger.warning(f"Relationship registry failed unexpectedly: {e}")
        return None


@dataclass
class ChatState:
    """Per-conversation state for reply generation."""

    style: UserStyle | None = None
    recent_generations: list[str] = field(default_factory=list)
    regen_count: int = 0
    last_message_hash: str = ""


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

    def __init__(self, model_loader, preload_embeddings: bool = True):
        """Initialize generator.

        Args:
            model_loader: ModelLoader instance for LLM generation
            preload_embeddings: If True, preload embedding model for fast RAG
        """
        self.model_loader = model_loader
        self.style_analyzer = StyleAnalyzer()
        self.context_analyzer = ContextAnalyzer()

        # Load response templates for instant matching
        self._template_matcher = _get_template_matcher()

        # Per-conversation state (style cache, recent generations, regen tracking)
        self._chat_states: dict[str, ChatState] = {}

        # Preload embedding model for fast RAG queries
        # Without this, first RAG query takes ~10s to load the model
        if preload_embeddings:
            self._preload_embeddings()

    def _preload_embeddings(self) -> None:
        """Preload embedding model and FAISS index for fast RAG queries."""
        # 1. Preload embedding model (~10s cold, instant if cached)
        try:
            from core.embeddings.model import get_embedding_model
            model = get_embedding_model()
            if not model.is_loaded:
                logger.info("Preloading embedding model...")
                model.preload()
        except Exception as e:
            logger.warning(f"Failed to preload embedding model: {e}")

        # 2. Preload reply-pairs FAISS index (~1s from disk)
        try:
            store = _get_embedding_store()
            if store and store.is_reply_pairs_index_ready():
                logger.info("Preloading reply-pairs FAISS index...")
                store._get_or_build_reply_pairs_index()
        except Exception as e:
            logger.warning(f"Failed to preload FAISS index: {e}")

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """Get or create chat state for a conversation."""
        if chat_id not in self._chat_states:
            self._chat_states[chat_id] = ChatState()
        return self._chat_states[chat_id]

    def _get_temperature(self, chat_id: str, last_message: str) -> float:
        """Get temperature based on regeneration count.

        First generation uses low temp (0.2) for consistency.
        Subsequent regenerations increase temp for variety.

        Args:
            chat_id: Conversation ID
            last_message: The message being replied to

        Returns:
            Temperature value (0.2 to 0.9)
        """
        import hashlib

        state = self._get_chat_state(chat_id)

        # Use stable hash (MD5) of truncated message as identifier
        # This avoids Python's hash() which can vary between runs and has collisions
        msg_key = hashlib.md5(last_message[:100].encode(), usedforsecurity=False).hexdigest()[:16]

        # Check if this is a new message or regeneration
        if state.last_message_hash != msg_key:
            # New message - reset regen count
            state.last_message_hash = msg_key
            state.regen_count = 0
        else:
            # Same message - increment regen count
            state.regen_count += 1

        # Temperature scaling: increases with each regeneration
        temp = settings.generation.temperature_scale[
            min(state.regen_count, len(settings.generation.temperature_scale) - 1)
        ]

        if state.regen_count > 0:
            logger.info(f"Regeneration #{state.regen_count}, using temperature {temp}")

        return temp

    def _record_generation(self, chat_id: str, reply: str) -> None:
        """Record a generated reply for repetition tracking.

        Args:
            chat_id: Conversation ID
            reply: Generated reply text
        """
        state = self._get_chat_state(chat_id)
        state.recent_generations.append(reply.lower().strip())
        # Keep only last N generations
        if len(state.recent_generations) > self.MAX_RECENT_GENERATIONS:
            state.recent_generations.pop(0)

    def _is_repetitive(self, reply: str, chat_id: str | None) -> bool:
        """Check if a reply is repetitive (recently used).

        Args:
            reply: Reply text to check
            chat_id: Conversation ID

        Returns:
            True if reply was recently used in this conversation
        """
        if not chat_id or chat_id not in self._chat_states:
            return False
        reply_lower = reply.lower().strip()
        return reply_lower in self._chat_states[chat_id].recent_generations

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
        contact_name: str | None = None,
    ) -> ReplyGenerationResult:
        """Generate reply suggestions for a conversation.

        Args:
            messages: Recent messages from conversation
                     [{"text": "...", "sender": "...", "is_from_me": bool}, ...]
            chat_id: Optional conversation ID for style caching
            num_replies: Number of replies to generate
            user_name: User's name for personalized context
            contact_name: Optional contact name for relationship lookup
                         (used when chat_id doesn't contain phone number)

        Returns:
            ReplyGenerationResult with suggestions and metadata
        """
        self._contact_name = contact_name
        self._user_name = user_name
        start_time = time.time()
        timings: dict[str, float] = {}

        # Debug flag - set to True to see step-by-step progress
        DEBUG = True

        # 0. Filter to coherent segment (detect topic breaks)
        if DEBUG:
            print("         [step 0] coherence filter...", end=" ", flush=True)
        t0 = time.time()
        coherent_messages = self._get_coherent_messages(messages)
        timings["coherence_filter"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['coherence_filter']:.0f}ms)", flush=True)

        # 0.5 FAST PATH: Check response templates FIRST (before any other processing)
        # This skips all style analysis, context analysis, and LLM for common responses
        t0 = time.time()
        if coherent_messages and self._template_matcher:
            last_msg = coherent_messages[-1]
            if not last_msg.get("is_from_me") and last_msg.get("text"):
                template_match = self._template_matcher.match(last_msg["text"])
                if template_match:
                    timings["template_match"] = (time.time() - t0) * 1000
                    generation_time = (time.time() - start_time) * 1000
                    msg_preview = last_msg["text"][:30]
                    logger.info(
                        f"Template match! '{msg_preview}...' -> "
                        f"'{template_match.actual}' "
                        f"(conf={template_match.confidence:.2f}) in {generation_time:.0f}ms"
                    )

                    # Quick style analysis just for the result object
                    style = self._get_or_analyze_style(messages, chat_id)
                    context = ConversationContext(
                        last_message=last_msg["text"],
                        last_sender=last_msg.get("sender_name") or last_msg.get("sender") or "them",
                        intent=self.context_analyzer._detect_intent(last_msg["text"]),
                        relationship=RelationshipType.CASUAL_FRIEND,
                        topic="",
                        mood="neutral",
                        urgency="normal",
                        needs_response=True,
                        summary="",
                    )

                    return ReplyGenerationResult(
                        replies=[
                            GeneratedReply(
                                text=template_match.actual.strip(),
                                reply_type="template",
                                confidence=template_match.confidence,
                            )
                        ],
                        context=context,
                        style=style,
                        model_used="template",
                        generation_time_ms=generation_time,
                        prompt_used=f"[Template match: {template_match.trigger}]",
                        style_instructions="[Template - no style needed]",
                        past_replies=[],
                    )
        timings["template_check"] = (time.time() - t0) * 1000

        # 1. Analyze user's texting style (use all messages for style)
        if DEBUG:
            print("         [step 1] style analysis...", end=" ", flush=True)
        t0 = time.time()
        style = self._get_or_analyze_style(messages, chat_id)
        timings["style_analysis"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['style_analysis']:.0f}ms)", flush=True)

        # 2. Analyze conversation context (use coherent segment only)
        if DEBUG:
            print("         [step 2] context analysis...", end=" ", flush=True)
        t0 = time.time()
        context = self.context_analyzer.analyze(coherent_messages)
        timings["context_analysis"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['context_analysis']:.0f}ms)", flush=True)

        # 3. Find YOUR past replies to similar messages (few-shot learning)
        # Search within THIS conversation (uses FAISS index = fast)
        # TODO: Build global FAISS index to search ALL conversations
        if DEBUG:
            print("         [step 3] past replies lookup (RAG)...", end=" ", flush=True)
        t0 = time.time()
        past_replies = []
        availability = self._extract_availability_signal(coherent_messages)

        # Get the incoming message to find similar past situations
        if context.last_message and not context.last_message.startswith("Loved") and chat_id:
            past_replies = self._find_past_replies(
                incoming_message=context.last_message,
                chat_id=chat_id,  # Search this conversation (has FAISS index)
                recent_messages=coherent_messages,
                min_similarity=settings.generation.min_similarity_threshold,
            )
            if past_replies:
                logger.info(f"Found {len(past_replies)} past replies for few-shot learning")
        timings["past_replies_lookup"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['past_replies_lookup']:.0f}ms, found {len(past_replies)})", flush=True)

        # 4b. Template matching - if past replies are very similar and consistent, skip LLM
        template_reply = self._try_template_match(past_replies)
        if template_reply:
            logger.info("Template match found - skipping LLM")
            generation_time = (time.time() - start_time) * 1000
            template_generated = GeneratedReply(
                text=template_reply, reply_type="template", confidence=0.95
            )
            return ReplyGenerationResult(
                replies=[template_generated],
                context=context,
                style=style,
                model_used="template",
                generation_time_ms=generation_time,
                prompt_used="[Template match - LLM skipped]",
                style_instructions=self.style_analyzer.build_style_instructions(style, None),
                past_replies=past_replies or [],
            )

        # 5. Get contact profile for better style info (from ALL your messages, not just recent)
        # Skip topic extraction (LLM call) - only need style info for generation
        # Profile is cached for fast retrieval after first computation
        if DEBUG:
            print("         [step 5] contact profile...", end=" ", flush=True)
        t0 = time.time()
        profile = None
        if chat_id:
            profile = _get_contact_profile(chat_id, include_topics=False)
        timings["contact_profile"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['contact_profile']:.0f}ms)", flush=True)

        # 5.5 Get global user style (all messages, all chats)
        # This gives us your overall texting personality + LLM-generated summary
        if DEBUG:
            print("         [step 5.5] global style...", end=" ", flush=True)
        t0 = time.time()
        global_style = get_global_user_style(use_cache=True)
        timings["global_style"] = (time.time() - t0) * 1000
        if DEBUG:
            print(f"done ({timings['global_style']:.0f}ms)", flush=True)

        # 6. Build style instructions - use global style + contact profile
        style_instructions = self.style_analyzer.build_style_instructions(
            style,
            profile,
            global_style=global_style,
        )

        # Get recent topics for context continuity (from cached profile or skip)
        recent_topics = None
        if profile and profile.topics:
            recent_topics = [t.name for t in profile.topics[:3]]

        # Get YOUR common phrases for personalization
        # Prefer global phrases (more data = better patterns)
        your_phrases = None
        if global_style and global_style.common_phrases:
            your_phrases = global_style.common_phrases[:5]
        elif profile and profile.your_common_phrases:
            your_phrases = profile.your_common_phrases[:3]

        # 6.5 Context refresh - DISABLED
        # Was pulling old historical messages that confused the prompt
        # TODO: Revisit if we want to re-enable with better filtering
        t0 = time.time()
        timings["context_refresh"] = (time.time() - t0) * 1000

        # Use coherent messages directly (no historical injection)
        messages_for_prompt = coherent_messages.copy()

        # Build prompt based on available context
        # Priority: RAG prompt (if good past_replies) > Threaded (if group) > Conversation > Legacy

        # Detect if this is a group chat
        is_group_chat = len(set(
            m.get("sender") for m in coherent_messages if not m.get("is_from_me")
        )) > 1

        # Use RAG prompt if we have good past replies (more personalized)
        if past_replies and len(past_replies) >= 2:
            # Get average message length from profile
            avg_length = 50.0
            if profile and hasattr(profile, "avg_your_length") and profile.avg_your_length:
                avg_length = profile.avg_your_length

            # Get intent to guide response type
            message_intent = context.intent.value if context.intent else None

            prompt = build_rag_reply_prompt(
                messages=messages_for_prompt,
                last_message=context.last_message,
                contact_name=self._contact_name or "them",
                similar_exchanges=past_replies,
                relationship_type=profile.relationship_type if profile else None,
                avg_message_length=avg_length,
                message_intent=message_intent,
            )
            logger.debug("Using RAG prompt strategy (found %d past replies, intent=%s)", len(past_replies), message_intent)

        elif is_group_chat:
            # Use threaded prompt for group chats
            participants = list(set(
                m.get("sender_name") or m.get("sender") or "Someone"
                for m in coherent_messages if not m.get("is_from_me")
            ))
            prompt = build_threaded_reply_prompt(
                messages=messages_for_prompt,
                last_message=context.last_message,
                thread_topic=context.topic or "General",
                participants=participants,
                is_group=True,
            )
            logger.debug("Using THREADED prompt strategy (group chat)")

        elif settings.generation.prompt_strategy == PromptStrategy.CONVERSATION:
            # Simple conversation continuation prompt
            prompt = build_conversation_prompt(
                messages=messages_for_prompt,
                style_hint=settings.generation.conversation_style_hint,
                max_messages=10,
            )
            logger.debug("Using CONVERSATION prompt strategy")

        else:
            # LEGACY: Few-shot examples with "them: X\nme:" completion
            prompt = build_reply_prompt(
                messages=messages_for_prompt,
                last_message=context.last_message,
                last_sender=context.last_sender,
                style_instructions=style_instructions,
                past_replies=past_replies,
                user_name=self._user_name,
                recent_topics=recent_topics,
                availability=availability if availability != "unknown" else None,
                your_phrases=your_phrases,
                global_style=global_style,
                contact_profile=profile,
            )
            logger.debug("Using LEGACY prompt strategy")

        # 7. Generate with LLM
        if DEBUG:
            print("         [step 7] LLM generation...", end=" ", flush=True)
        t0 = time.time()
        formatted_prompt = ""  # Will store the actual ChatML prompt for debugging
        try:
            # Get temperature based on regen count (0.2 first time, scales up on regenerate)
            temperature = self._get_temperature(chat_id, context.last_message) if chat_id else 0.2

            result = self.model_loader.generate(
                prompt=prompt,
                max_tokens=settings.generation.max_tokens,
                temperature=temperature,
                stop=["\n", "2.", "2)", "##", "Note:", "---"],
                use_chat_template=False,  # Raw completion works better for few-shot
            )
            raw_output = result.text
            formatted_prompt = result.formatted_prompt  # The actual ChatML prompt
            timings["llm_generation"] = (time.time() - t0) * 1000
            if DEBUG:
                print(f"done ({timings['llm_generation']:.0f}ms)", flush=True)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return fallback replies
            return self._fallback_result(context, style, str(e))

        # 8. Parse replies - strip emojis if profile says no emojis
        t0 = time.time()
        should_strip_emojis = profile and not profile.uses_emoji
        replies = self._parse_replies(raw_output, [], strip_emojis_flag=should_strip_emojis)
        timings["parse_replies"] = (time.time() - t0) * 1000

        # 9. Filter out repetitive replies (recently used in this conversation)
        replies = self._filter_repetitive(replies, chat_id)

        # 9.5 Use RAG past_replies as additional suggestions (BEFORE generic fallbacks)
        # This is the key improvement: your actual past replies are better than "got it", "cool"
        if len(replies) < num_replies and past_replies:
            rag_suggestions = self._get_rag_suggestions(
                past_replies, num_replies - len(replies)
            )
            rag_suggestions = self._filter_repetitive(rag_suggestions, chat_id)
            # Don't duplicate the LLM-generated reply
            existing_texts = {r.text.lower().strip() for r in replies}
            rag_suggestions = [r for r in rag_suggestions if r.text.lower().strip() not in existing_texts]
            replies.extend(rag_suggestions)

        # 9.7 Add clarification responses if context is needed but not found
        if len(replies) < num_replies and self._should_add_clarification(past_replies, context):
            clarifications = self._get_clarification_responses(
                context.intent.value,
                needs_context=True,
                is_specific_question=True,
            )
            # Only add one clarification option
            if clarifications:
                existing_texts = {r.text.lower().strip() for r in replies}
                for c in clarifications[:1]:
                    if c.text.lower().strip() not in existing_texts:
                        replies.append(c)
                        break
            logger.debug("Added clarification response (low-confidence context)")

        # 10. Fall back to generic templates only if still not enough
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
            f"template:{timings.get('template_check', 0):.0f}ms, "
            f"coherence:{timings.get('coherence_filter', 0):.0f}ms, "
            f"style:{timings.get('style_analysis', 0):.0f}ms, "
            f"context:{timings.get('context_analysis', 0):.0f}ms, "
            f"past_replies:{timings.get('past_replies_lookup', 0):.0f}ms, "
            f"profile:{timings.get('contact_profile', 0):.0f}ms, "
            f"global_style:{timings.get('global_style', 0):.0f}ms, "
            f"refresh:{timings.get('context_refresh', 0):.0f}ms, "
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

    def _get_or_analyze_style(self, messages: list[dict], chat_id: str | None) -> UserStyle:
        """Get cached style or analyze from messages."""
        if chat_id:
            state = self._get_chat_state(chat_id)
            if state.style is not None:
                return state.style

        # Filter to user's messages only
        user_messages = [m for m in messages if m.get("is_from_me")]
        style = self.style_analyzer.analyze(user_messages)

        if chat_id:
            self._get_chat_state(chat_id).style = style

        return style

    def _parse_replies(
        self, raw_output: str, reply_types: list[str], strip_emojis_flag: bool = False
    ) -> list[GeneratedReply]:
        """Parse LLM output into structured replies."""
        from core.utils.emoji import strip_emojis

        replies = []
        output = raw_output.strip()

        # Take first line only (the reply)
        text = output.split("\n")[0].strip()

        # Remove common prefixes/artifacts
        for prefix in ["Reply:", "Response:", "Answer:", f"{self._user_name}:", "Them:", "Me:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()

        # Remove quotes if wrapped
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1]

        # Strip emojis if style says no emojis (model often ignores this instruction)
        if strip_emojis_flag:
            text = strip_emojis(text)

        # Clean up
        text = text.strip()

        if text and len(text) >= 2 and len(text) <= 150:
            replies.append(
                GeneratedReply(
                    text=text,
                    reply_type=reply_types[0] if reply_types else "general",
                    confidence=0.9,
                )
            )

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
                GeneratedReply(
                    text=text,
                    reply_type="personal_template",
                    confidence=settings.generation.template_confidence,
                )
                for text in replies[:count]
            ]
        except Exception as e:
            logger.debug(f"Failed to get personal templates: {e}")
            return []

    def _get_clarification_responses(
        self,
        intent_value: str,
        needs_context: bool,
        is_specific_question: bool,
    ) -> list[GeneratedReply]:
        """Get clarification responses when context is needed but not found.

        Args:
            intent_value: Detected intent type
            needs_context: Whether the message needs specific context
            is_specific_question: Whether asking about specific facts

        Returns:
            List of clarification responses
        """
        responses = []

        if is_specific_question or needs_context:
            # Generic clarification for specific questions
            clarifications = {
                "information_seeking": [
                    "hmm not sure, when was that?",
                    "can you give me more context?",
                    "remind me what we were talking about?",
                ],
                "open_question": [
                    "what do you mean?",
                    "can you be more specific?",
                    "not sure I follow",
                ],
                "yes_no_question": [
                    "hmm not sure",
                    "I'd have to check",
                    "remind me?",
                ],
            }

            options = clarifications.get(intent_value, clarifications["open_question"])
            for text in options[:2]:  # Max 2 clarification options
                responses.append(
                    GeneratedReply(
                        text=text,
                        reply_type="clarification",
                        confidence=0.6,
                    )
                )

        return responses

    def _should_add_clarification(
        self,
        past_replies: list[tuple[str, str, float]],
        context: ConversationContext,
        min_similarity: float = 0.35,
    ) -> bool:
        """Check if we should add a clarification response.

        Returns True if:
        - Message needs context (asking about specific info)
        - RAG didn't find good matches (low similarity)

        Args:
            past_replies: RAG results with similarity scores
            context: Conversation context with intent info
            min_similarity: Threshold for "good" RAG match

        Returns:
            True if clarification might be helpful
        """
        # Check if context detected a specific question needing context
        intent_result = None
        try:
            from core.intent import classify_incoming_message
            intent_result = classify_incoming_message(context.last_message)
        except Exception:
            pass

        if intent_result and (intent_result.needs_context or intent_result.is_specific_question):
            # Check if RAG found good matches
            if not past_replies:
                return True
            max_similarity = max(score for _, _, score in past_replies)
            if max_similarity < min_similarity:
                return True

        return False

    def _get_rag_suggestions(
        self,
        past_replies: list[tuple[str, str, float]],
        count: int,
    ) -> list[GeneratedReply]:
        """Convert RAG past_replies into reply suggestions.

        This uses your ACTUAL past replies as suggestions, which are much
        better than generic fallbacks like "got it" or "cool".

        Args:
            past_replies: List of (their_message, your_reply, similarity) tuples
            count: Number of suggestions needed

        Returns:
            List of GeneratedReply objects from your past messages
        """
        if not past_replies:
            return []

        suggestions = []
        seen_texts = set()

        for their_msg, your_reply, similarity in past_replies:
            # Skip if we've seen this exact reply
            reply_key = your_reply.lower().strip()
            if reply_key in seen_texts:
                continue
            seen_texts.add(reply_key)

            # Clean up the reply
            reply_text = your_reply.strip()

            # Skip very short or very long replies
            if len(reply_text) < 2 or len(reply_text) > 100:
                continue

            suggestions.append(
                GeneratedReply(
                    text=reply_text,
                    reply_type="rag_suggestion",
                    confidence=min(0.85, similarity + 0.1),  # Boost slightly
                )
            )

            if len(suggestions) >= count:
                break

        logger.debug(f"Created {len(suggestions)} RAG suggestions from past replies")
        return suggestions

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

    def clear_cache(self, chat_id: str | None = None) -> None:
        """Clear cached state for a conversation.

        Args:
            chat_id: Specific chat to clear, or None for all
        """
        if chat_id:
            self._chat_states.pop(chat_id, None)
        else:
            self._chat_states.clear()

    # Alias for backwards compatibility
    clear_style_cache = clear_cache

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

    def _extract_availability_signal(
        self,
        recent_messages: list[dict] | None,
    ) -> str:
        """Extract availability signal from recent messages.

        Looks at YOUR recent messages to detect if you've indicated
        being busy or free.

        Args:
            recent_messages: Recent messages in the conversation

        Returns:
            "busy", "free", or "unknown"
        """
        if not recent_messages:
            return "unknown"

        # Look at your last 5 messages
        your_recent = [
            m.get("text", "").lower()
            for m in recent_messages
            if m.get("is_from_me") and m.get("text")
        ][-5:]

        if not your_recent:
            return "unknown"

        combined = " ".join(your_recent)

        # Busy signals
        busy_patterns = [
            "busy",
            "can't",
            "cant",
            "exhausted",
            "swamped",
            "packed",
            "tired",
            "slammed",
            "hectic",
            "crazy week",
            "no time",
            "working late",
            "have to work",
            "not free",
            "won't be able",
        ]
        busy_count = sum(1 for p in busy_patterns if p in combined)

        # Free signals
        free_patterns = [
            "free",
            "down",
            "available",
            "let's do",
            "lets do",
            "sounds good",
            "i'm in",
            "im in",
            "count me in",
            "nothing going on",
            "not busy",
            "have time",
        ]
        free_count = sum(1 for p in free_patterns if p in combined)

        if busy_count > free_count and busy_count >= 1:
            return "busy"
        elif free_count > busy_count and free_count >= 1:
            return "free"

        return "unknown"

    def _find_past_replies(
        self,
        incoming_message: str,
        chat_id: str | None,
        recent_messages: list[dict] | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[str, str, float]]:
        """Find YOUR past replies to similar incoming messages.

        Now relationship-aware: searches same conversation AND
        cross-conversation from contacts with similar relationships.

        Args:
            incoming_message: The message to find similar responses for
            chat_id: Optional conversation filter
            recent_messages: Recent messages for context/availability detection
            min_similarity: Minimum similarity threshold

        Returns:
            List of (their_message, your_reply, score) tuples
        """
        store = _get_embedding_store()
        if not store:
            return []

        if min_similarity is None:
            min_similarity = settings.generation.min_similarity_threshold

        try:
            # 1. Same-conversation search (high priority)
            same_convo_results = store.find_your_past_replies(
                incoming_message=incoming_message,
                chat_id=chat_id,
                limit=5,
                min_similarity=min_similarity,
                use_time_weighting=True,
            )

            # 2. Cross-conversation search (relationship-aware)
            cross_convo_results = self._find_cross_conversation_replies(
                incoming_message=incoming_message,
                current_chat_id=chat_id,
                limit=5,
                min_similarity=min_similarity,
            )

            # 3. Merge results (prioritize same-conversation)
            results = self._merge_past_replies(
                same_convo_results,
                cross_convo_results,
                same_convo_boost=0.05,  # Boost for same-convo
            )

            # 4. Apply availability-based filtering
            availability = self._extract_availability_signal(recent_messages)
            if availability != "unknown" and results:
                results = self._filter_by_availability(results, availability)

            return results[:5]

        except Exception as e:
            logger.debug(f"Failed to find past replies: {e}")
            return []

    def _find_cross_conversation_replies(
        self,
        incoming_message: str,
        current_chat_id: str | None,
        limit: int = 5,
        min_similarity: float | None = None,
        skip_if_slow: bool = True,
    ) -> list[tuple[str, str, float]]:
        """Find past replies from contacts with similar relationships.

        Uses the RelationshipRegistry to find contacts in the same category
        (friend/family/work/other) and searches their conversations.

        Args:
            incoming_message: The message to find similar responses for
            current_chat_id: Current conversation's chat_id
            limit: Max results
            min_similarity: Minimum similarity threshold
            skip_if_slow: If True, skip if reply-pairs index isn't cached

        Returns:
            List of (their_message, your_reply, score) tuples
        """
        store = _get_embedding_store()
        registry = _get_relationship_registry()

        if not store or not registry:
            return []

        # Skip if index isn't ready (avoids slow on-demand build)
        if skip_if_slow and not store.is_reply_pairs_index_ready():
            logger.info("Skipping cross-conversation search - reply-pairs index not ready")
            return []

        if min_similarity is None:
            min_similarity = settings.generation.min_similarity_threshold

        try:
            # Get relationship info - try chat_id first, then contact_name fallback
            relationship_info = None
            if current_chat_id:
                relationship_info = registry.get_relationship_from_chat_id(current_chat_id)

            # Fallback to contact_name if chat_id lookup failed
            if not relationship_info and hasattr(self, "_contact_name") and self._contact_name:
                relationship_info = registry.get_relationship(self._contact_name)
                if relationship_info:
                    logger.debug(f"Using contact_name fallback for {self._contact_name}")

            if not relationship_info:
                contact = getattr(self, "_contact_name", None)
                logger.debug(
                    f"No relationship info found for chat_id={current_chat_id}, "
                    f"contact_name={contact}"
                )
                return []

            # Get all contacts in same category
            similar_contacts = registry.get_similar_contacts(current_chat_id)
            if not similar_contacts:
                logger.debug(f"No similar contacts found for category {relationship_info.category}")
                return []

            # Get phone numbers for similar contacts
            phones_by_contact = registry.get_phones_for_contacts(similar_contacts)

            # Collect all phones
            all_phones = []
            for phones in phones_by_contact.values():
                all_phones.extend(phones)

            # Resolve to ACTUAL chat_ids from database (cached)
            target_chat_ids = store.resolve_phones_to_chatids(all_phones)

            if not target_chat_ids:
                logger.debug("No target chat_ids resolved for cross-conversation search")
                return []

            logger.debug(
                f"Cross-conversation search: category={relationship_info.category}, "
                f"searching {len(similar_contacts)} contacts, "
                f"{len(target_chat_ids)} potential chat_ids"
            )

            # Search across these conversations
            cross_results = store.find_your_past_replies_cross_conversation(
                incoming_message=incoming_message,
                target_chat_ids=target_chat_ids,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Convert to same format (drop chat_id from tuple)
            return [
                (their_msg, your_reply, score) for their_msg, your_reply, score, _ in cross_results
            ]

        except Exception as e:
            logger.debug(f"Cross-conversation search failed: {e}")
            return []

    def _merge_past_replies(
        self,
        same_convo: list[tuple[str, str, float]],
        cross_convo: list[tuple[str, str, float]],
        same_convo_boost: float = 0.05,
    ) -> list[tuple[str, str, float]]:
        """Merge same-conversation and cross-conversation results.

        Prioritizes same-conversation results with a small score boost.
        Deduplicates by reply text.

        Args:
            same_convo: Results from same conversation
            cross_convo: Results from similar-relationship conversations
            same_convo_boost: Score boost for same-conversation results

        Returns:
            Merged and deduplicated list of (their_msg, your_reply, score)
        """
        same_weight = settings.generation.same_convo_weight
        cross_weight = settings.generation.cross_convo_weight

        # Boost same-conversation scores
        boosted_same = [
            (their_msg, your_reply, (score + same_convo_boost) * same_weight)
            for their_msg, your_reply, score in same_convo
        ]

        # Apply weight to cross-conversation scores
        weighted_cross = [
            (their_msg, your_reply, score * cross_weight)
            for their_msg, your_reply, score in cross_convo
        ]

        # Combine and sort
        combined = boosted_same + weighted_cross
        combined.sort(key=lambda x: x[2], reverse=True)

        # Deduplicate by reply text (case-insensitive)
        seen_replies = set()
        deduplicated = []
        for their_msg, your_reply, score in combined:
            reply_key = your_reply.lower().strip()
            if reply_key not in seen_replies:
                seen_replies.add(reply_key)
                deduplicated.append((their_msg, your_reply, score))

        return deduplicated

    def _filter_by_availability(
        self,
        results: list[tuple[str, str, float]],
        availability: str,
    ) -> list[tuple[str, str, float]]:
        """Filter/adjust results based on availability signal.

        Args:
            results: List of (their_msg, your_reply, score) tuples
            availability: "busy", "free", or "unknown"

        Returns:
            Adjusted results list
        """
        # Keywords that suggest accept vs decline responses
        accept_keywords = {"yes", "yeah", "yea", "sure", "ok", "okay", "down", "in", "sounds good"}
        decline_keywords = {"no", "nah", "can't", "cant", "sorry", "busy", "won't", "wont"}

        def response_type(reply: str) -> str:
            reply_lower = reply.lower().strip()
            words = set(reply_lower.split())
            if words & accept_keywords or reply_lower.startswith(tuple(accept_keywords)):
                return "accept"
            if words & decline_keywords or reply_lower.startswith(tuple(decline_keywords)):
                return "decline"
            return "neutral"

        adjusted = []
        for their_msg, your_reply, score in results:
            rtype = response_type(your_reply)
            if availability == "busy" and rtype == "decline":
                score += 0.1  # Boost decline responses when busy
            elif availability == "busy" and rtype == "accept":
                score -= 0.05  # Slight penalty for accept when busy
            elif availability == "free" and rtype == "accept":
                score += 0.1  # Boost accept responses when free
            adjusted.append((their_msg, your_reply, score))

        # Re-sort by adjusted scores
        adjusted.sort(key=lambda x: x[2], reverse=True)

        logger.debug(f"Availability signal: {availability}, adjusted {len(adjusted)} past replies")

        return adjusted

    # TODO: Remove if unused - context refresh disabled, was pulling old historical messages
    def _refresh_context_for_topic(
        self,
        messages: list[dict],
        chat_id: str | None,
    ) -> list[dict]:
        """Refresh context based on current conversation topic.

        For long conversations, the original context may drift.
        This re-queries embeddings based on the CURRENT topic.

        Inspired by PreToolUse hook pattern - mid-stream context injection.

        Args:
            messages: Recent messages in conversation
            chat_id: Conversation ID for filtering

        Returns:
            List of relevant historical messages
        """
        if not messages or len(messages) < 3:
            return []

        store = _get_embedding_store()
        if not store:
            return []

        # Extract current topic from last 3 messages
        recent_text = " ".join([m.get("text", "") for m in messages[-3:] if m.get("text")])

        if len(recent_text) < 20:
            return []

        try:
            # Query for relevant historical context (hybrid search)
            if hasattr(store, "find_similar_hybrid"):
                similar = store.find_similar_hybrid(
                    query=recent_text,
                    chat_id=chat_id,
                    limit=3,
                    min_similarity=0.4,
                )
            else:
                similar = store.find_similar(
                    query=recent_text,
                    chat_id=chat_id,
                    limit=3,
                    min_similarity=0.4,
                )

            if not similar:
                return []

            # Convert to message format
            refreshed_context = []
            for msg in similar:
                refreshed_context.append(
                    {
                        "text": msg.text,
                        "sender": msg.sender_name or msg.sender,
                        "is_from_me": msg.is_from_me,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                        "_source": "context_refresh",  # Mark as refreshed context
                    }
                )

            logger.debug(
                f"Refreshed context: {len(refreshed_context)} relevant messages "
                f"for topic '{recent_text[:50]}...'"
            )
            return refreshed_context

        except Exception as e:
            logger.debug(f"Context refresh failed: {e}")
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
        high_confidence = [
            r for r in past_replies if r[2] >= settings.generation.past_reply_confidence
        ]
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

    # TODO: Remove if unused - not currently called anywhere
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
