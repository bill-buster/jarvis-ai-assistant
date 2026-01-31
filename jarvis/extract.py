"""Turn-Based Pair Extraction - Extract (trigger, response) turn pairs from iMessage.

Key insight: Real conversations have "turns" - consecutive messages from the same person.
Instead of pairing single messages, we bundle consecutive messages into turns.

Example:
    Them: "hey"
    Them: "want to grab lunch?"
    Them: "thinking sushi"
    You: "sounds good!"
    You: "omw"

    Becomes: trigger="hey\nwant to grab lunch?\nthinking sushi"
             response="sounds good!\nomw"

Usage:
    jarvis db extract                  # Extract from all conversations
    jarvis db extract --min-length 3   # Filter short messages
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts.imessage import Message

if TYPE_CHECKING:
    from jarvis.embedding_adapter import UnifiedEmbedder
    from jarvis.exchange import CandidateExchange

logger = logging.getLogger(__name__)


# =============================================================================
# Module-Level Detection Functions (imported from text_normalizer)
# =============================================================================

# Import centralized detection functions from text_normalizer
from jarvis.text_normalizer import (
    is_acknowledgment_only as is_simple_acknowledgment,
)
from jarvis.text_normalizer import (
    is_reaction as is_reaction_message,
)
from jarvis.text_normalizer import (
    starts_new_topic as is_topic_shift,
)


@dataclass
class ExtractionConfig:
    """Configuration for turn-based pair extraction."""

    # Time window for bundling messages into a turn (same speaker)
    turn_bundle_minutes: float = 10.0

    # Max delay to consider as a response (keep high, use quality penalty instead)
    # Responses after this are dropped entirely (some people never reply)
    max_response_delay_hours: float = 168.0  # 1 week hard cutoff

    # Delay thresholds for quality penalties (hours)
    # Responses >12h are severely down-ranked (different cognitive state)
    delay_severe_penalty_hours: float = 12.0
    delay_moderate_penalty_hours: float = 1.0

    # Minimum message/turn length (characters) to include
    min_trigger_length: int = 2
    min_response_length: int = 20  # ~6-8 tokens

    # Minimum response length in tokens (crude but effective for v1)
    # Catches "ok", "lol", "k" etc.
    min_response_tokens: int = 6

    # Maximum turn length (characters) - very long = not templatable
    max_trigger_length: int = 500
    max_response_length: int = 400

    # Skip attachment-only messages when bundling
    skip_attachment_only: bool = True

    # Skip system messages (group events, etc.)
    skip_system_messages: bool = True

    # Semantic similarity filtering (requires embedder)
    # Set to True to compute trigger-response similarity for quality scoring
    use_semantic_similarity: bool = False

    # Threshold below which pairs are considered clearly unrelated (0.45 = harsh)
    semantic_reject_threshold: float = 0.45

    # Threshold for borderline pairs (between reject and this = lower quality)
    semantic_borderline_threshold: float = 0.55


@dataclass
class Turn:
    """A turn in a conversation - one or more consecutive messages from same speaker."""

    messages: list[Message] = field(default_factory=list)
    is_from_me: bool = False

    @property
    def text(self) -> str:
        """Get combined text of all messages in turn, joined with newlines."""
        texts = [m.text for m in self.messages if m.text]
        return "\n".join(texts)

    @property
    def first_timestamp(self) -> datetime:
        """Timestamp of first message in turn."""
        return self.messages[0].date if self.messages else datetime.min

    @property
    def last_timestamp(self) -> datetime:
        """Timestamp of last message in turn."""
        return self.messages[-1].date if self.messages else datetime.min

    @property
    def message_ids(self) -> list[int]:
        """Get all message IDs in this turn."""
        return [m.id for m in self.messages]

    @property
    def primary_msg_id(self) -> int | None:
        """Get primary (first) message ID."""
        return self.messages[0].id if self.messages else None

    def add_message(self, msg: Message) -> None:
        """Add a message to this turn."""
        self.messages.append(msg)


@dataclass
class ExtractedPair:
    """A (trigger_turn, response_turn) pair extracted from conversation."""

    trigger_text: str
    response_text: str
    trigger_timestamp: datetime  # First trigger message timestamp
    response_timestamp: datetime  # First response message timestamp
    chat_id: str
    # Message IDs for debugging/deduplication
    trigger_msg_id: int  # Primary trigger message ID
    response_msg_id: int  # Primary response message ID
    trigger_msg_ids: list[int]  # All trigger message IDs
    response_msg_ids: list[int]  # All response message IDs
    # Conversation context (messages before the trigger)
    context_text: str | None = None  # Previous messages for LLM context
    # Metadata
    time_delta_seconds: float = 0.0
    trigger_message_count: int = 1
    response_message_count: int = 1
    is_group: bool = False  # True if from group chat (for filtering at query time)
    # Quality signals
    quality_score: float = 1.0
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionStats:
    """Statistics from extraction run."""

    total_messages_scanned: int = 0
    turns_identified: int = 0
    candidate_pairs: int = 0
    kept_pairs: int = 0
    dropped_short_trigger: int = 0
    dropped_short_response: int = 0
    dropped_long_trigger: int = 0
    dropped_long_response: int = 0
    dropped_no_text: int = 0
    dropped_time_gap: int = 0
    # Quality breakdown (pairs are still kept but marked with flags)
    flagged_reaction: int = 0
    flagged_low_similarity: int = 0
    flagged_topic_shift: int = 0
    flagged_ack_substantive: int = 0


class TurnBasedExtractor:
    """Extracts turn-based (trigger, response) pairs from message history.

    Groups consecutive messages from the same speaker into "turns",
    then pairs incoming turns with outgoing response turns.

    Args:
        config: Extraction configuration.
        embedder: Optional embedder for semantic similarity filtering.
            If provided and config.use_semantic_similarity is True,
            computes trigger-response similarity for quality scoring.
    """

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        embedder: UnifiedEmbedder | None = None,
    ) -> None:
        """Initialize extractor with configuration and optional embedder."""
        self.config = config or ExtractionConfig()
        self._embedder = embedder

        # Auto-enable semantic similarity if embedder is provided
        if embedder is not None and not self.config.use_semantic_similarity:
            # Create a new config with semantic similarity enabled
            # (don't mutate the passed config)
            self.config = ExtractionConfig(
                turn_bundle_minutes=self.config.turn_bundle_minutes,
                max_response_delay_hours=self.config.max_response_delay_hours,
                delay_severe_penalty_hours=self.config.delay_severe_penalty_hours,
                delay_moderate_penalty_hours=self.config.delay_moderate_penalty_hours,
                min_trigger_length=self.config.min_trigger_length,
                min_response_length=self.config.min_response_length,
                min_response_tokens=self.config.min_response_tokens,
                max_trigger_length=self.config.max_trigger_length,
                max_response_length=self.config.max_response_length,
                skip_attachment_only=self.config.skip_attachment_only,
                skip_system_messages=self.config.skip_system_messages,
                use_semantic_similarity=True,
                semantic_reject_threshold=self.config.semantic_reject_threshold,
                semantic_borderline_threshold=self.config.semantic_borderline_threshold,
            )

    def _compute_semantic_similarity(self, trigger_text: str, response_text: str) -> float | None:
        """Compute semantic similarity between trigger and response.

        Returns:
            Cosine similarity (0.0-1.0) if embedder is available, None otherwise.
        """
        if self._embedder is None or not self.config.use_semantic_similarity:
            return None

        try:
            embeddings = self._embedder.encode([trigger_text, response_text], normalize=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            return similarity
        except Exception as e:
            logger.warning("Failed to compute semantic similarity: %s", e)
            return None

    def extract_pairs(
        self,
        messages: list[Message],
        chat_id: str,
        is_group: bool = False,
    ) -> tuple[list[ExtractedPair], ExtractionStats]:
        """Extract turn-based pairs from a conversation's messages.

        Args:
            messages: List of messages, sorted by date ascending (oldest first).
            chat_id: The conversation ID.
            is_group: Whether this is a group chat (from conversation metadata).

        Returns:
            Tuple of (list of extracted pairs, extraction statistics).
        """
        stats = ExtractionStats()

        if not messages:
            return [], stats

        # Sort messages by date ascending
        sorted_messages = sorted(messages, key=lambda m: m.date)
        stats.total_messages_scanned = len(sorted_messages)

        # Step 1: Group messages into turns
        turns = self._group_into_turns(sorted_messages)
        stats.turns_identified = len(turns)

        # Step 2: Pair incoming turns with outgoing response turns
        pairs: list[ExtractedPair] = []
        max_delay = timedelta(hours=self.config.max_response_delay_hours)

        i = 0
        while i < len(turns):
            turn = turns[i]

            # Skip if this is my turn (we want triggers FROM others)
            if turn.is_from_me:
                i += 1
                continue

            # Look for my response turn
            if i + 1 < len(turns) and turns[i + 1].is_from_me:
                response_turn = turns[i + 1]

                # Check time gap
                time_delta = response_turn.first_timestamp - turn.last_timestamp
                if time_delta > max_delay:
                    stats.dropped_time_gap += 1
                    i += 1
                    continue

                stats.candidate_pairs += 1

                # Gather context from previous turns (up to 5 turns before trigger)
                context_turns = turns[max(0, i - 5) : i]

                # Validate and create pair
                pair = self._create_pair(
                    turn, response_turn, chat_id, time_delta, stats, context_turns, is_group
                )
                if pair:
                    pairs.append(pair)
                    stats.kept_pairs += 1

                # Skip past the response turn
                i += 2
            else:
                i += 1

        return pairs, stats

    def _group_into_turns(self, messages: list[Message]) -> list[Turn]:
        """Group consecutive messages from same speaker into turns.

        Messages are bundled if:
        - Same speaker (is_from_me matches)
        - Within turn_bundle_minutes of previous message in turn
        """
        if not messages:
            return []

        turns: list[Turn] = []
        bundle_window = timedelta(minutes=self.config.turn_bundle_minutes)

        current_turn: Turn | None = None

        for msg in messages:
            # Skip system messages
            if self.config.skip_system_messages and msg.is_system_message:
                continue

            # Skip attachment-only messages
            if self.config.skip_attachment_only and not msg.text and msg.attachments:
                continue

            # Check if this message continues the current turn
            if current_turn is not None:
                # Same speaker?
                if msg.is_from_me == current_turn.is_from_me:
                    # Within time window?
                    time_since_last = msg.date - current_turn.last_timestamp
                    if time_since_last <= bundle_window:
                        current_turn.add_message(msg)
                        continue

                # Different speaker or too much time - save current turn
                if current_turn.text:  # Only save if has text
                    turns.append(current_turn)

            # Start new turn
            current_turn = Turn(is_from_me=msg.is_from_me)
            current_turn.add_message(msg)

        # Don't forget the last turn
        if current_turn is not None and current_turn.text:
            turns.append(current_turn)

        return turns

    def _create_pair(
        self,
        trigger_turn: Turn,
        response_turn: Turn,
        chat_id: str,
        time_delta: timedelta,
        stats: ExtractionStats,
        context_turns: list[Turn] | None = None,
        is_group: bool = False,
    ) -> ExtractedPair | None:
        """Create a pair from trigger and response turns, with validation.

        Args:
            context_turns: Previous turns before the trigger (for LLM context).
            is_group: Whether this is from a group chat.

        Returns None if validation fails.
        """
        trigger_text = self._clean_text(trigger_turn.text)
        response_text = self._clean_text(response_turn.text)

        # Build context text from previous turns
        context_text = None
        if context_turns:
            context_lines = []
            for turn in context_turns:
                turn_text = self._clean_text(turn.text)
                if turn_text:
                    speaker = "You" if turn.is_from_me else "Them"
                    context_lines.append(f"[{speaker}]: {turn_text}")
            if context_lines:
                context_text = "\n".join(context_lines)

        # Validate trigger
        if not trigger_text:
            stats.dropped_no_text += 1
            return None
        if len(trigger_text) < self.config.min_trigger_length:
            stats.dropped_short_trigger += 1
            return None
        if len(trigger_text) > self.config.max_trigger_length:
            stats.dropped_long_trigger += 1
            return None

        # Validate response
        if not response_text:
            stats.dropped_no_text += 1
            return None

        # Token-based length check (catches "ok", "lol", "k", etc.)
        response_tokens = len(response_text.split())
        if response_tokens < self.config.min_response_tokens:
            stats.dropped_short_response += 1
            return None

        # Character-based length check as backup
        if len(response_text) < self.config.min_response_length:
            stats.dropped_short_response += 1
            return None
        if len(response_text) > self.config.max_response_length:
            stats.dropped_long_response += 1
            return None

        # Compute semantic similarity if embedder is available
        semantic_similarity = self._compute_semantic_similarity(trigger_text, response_text)

        # Calculate quality score (with semantic similarity if available)
        quality_score, flags = self._calculate_quality(
            trigger_text,
            response_text,
            trigger_turn,
            response_turn,
            time_delta,
            semantic_similarity=semantic_similarity,
        )

        # Track quality flag stats (pairs are still kept but flagged)
        if flags.get("is_reaction"):
            stats.flagged_reaction += 1
        if flags.get("low_semantic_similarity"):
            stats.flagged_low_similarity += 1
        if flags.get("topic_shift"):
            stats.flagged_topic_shift += 1
        if flags.get("ack_trigger_substantive_response"):
            stats.flagged_ack_substantive += 1

        return ExtractedPair(
            trigger_text=trigger_text,
            response_text=response_text,
            trigger_timestamp=trigger_turn.first_timestamp,
            response_timestamp=response_turn.first_timestamp,
            chat_id=chat_id,
            trigger_msg_id=trigger_turn.primary_msg_id or 0,
            response_msg_id=response_turn.primary_msg_id or 0,
            trigger_msg_ids=trigger_turn.message_ids,
            response_msg_ids=response_turn.message_ids,
            context_text=context_text,
            time_delta_seconds=time_delta.total_seconds(),
            trigger_message_count=len(trigger_turn.messages),
            response_message_count=len(response_turn.messages),
            is_group=is_group,
            quality_score=quality_score,
            flags=flags,
        )

    def _clean_text(self, text: str | None) -> str:
        """Clean and normalize message text."""
        if not text:
            return ""

        # Strip whitespace
        cleaned = text.strip()

        # Normalize internal whitespace (but preserve newlines for multi-message turns)
        lines = cleaned.split("\n")
        cleaned_lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in lines]
        cleaned = "\n".join(line for line in cleaned_lines if line)

        return cleaned

    # Generic responses that are too context-dependent for templates
    GENERIC_RESPONSES = frozenset(
        {
            "ok",
            "okay",
            "k",
            "kk",
            "yes",
            "yeah",
            "yep",
            "yup",
            "no",
            "nope",
            "nah",
            "sure",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "np",
            "cool",
            "nice",
            "good",
            "great",
            "awesome",
            "alright",
            "sounds good",
            "got it",
            "lol",
            "haha",
            "hahaha",
            "lmao",
            "omw",
            "on my way",
            "be there soon",
            "see you",
            "bye",
            "later",
            "ttyl",
            "👍",
            "👌",
            "🙏",
            "😂",
            "😊",
            "❤️",
            "🔥",
            "💯",
        }
    )

    def _is_generic_response(self, text: str) -> bool:
        """Check if response is a generic/context-dependent phrase."""
        normalized = text.lower().strip()
        return normalized in self.GENERIC_RESPONSES

    def _is_emoji_only(self, text: str) -> bool:
        """Check if response contains only emojis."""
        from jarvis.text_normalizer import is_emoji_only

        return is_emoji_only(text)

    def _is_topic_shift(self, response_text: str) -> bool:
        """Check if response appears to be a topic shift, not a direct reply.

        Topic shifts start with indicators like "btw", "anyway", etc. that signal
        the responder is changing the subject rather than replying to the trigger.
        """
        return is_topic_shift(response_text)

    def _is_reaction(self, text: str) -> bool:
        """Check if text is an iMessage tapback reaction.

        Tapbacks appear as "Liked \"...\", "Loved \"...\", etc.
        These are not real responses and should be filtered out.
        """
        return is_reaction_message(text)

    def _is_acknowledgment_trigger(self, text: str) -> bool:
        """Check if trigger is just an acknowledgment.

        Acknowledgments like "Ok", "Yes", "Sure" often precede NEW topics,
        not responses to the acknowledgment itself.
        """
        return is_simple_acknowledgment(text)

    def _is_substantive_response(self, text: str) -> bool:
        """Check if response is substantive (not just a brief ack or emoji).

        A substantive response typically has:
        - More than 20 characters, OR
        - More than 5 words, OR
        - Contains a question, OR
        - Contains specific information (numbers, names, etc.)
        """
        # Check character count first (user requirement: >20 chars)
        if len(text.strip()) > 20:
            return True
        words = text.split()
        if len(words) > 5:
            return True
        if "?" in text:
            return True
        # Check for numbers or time patterns
        if re.search(r"\d", text):
            return True
        return False

    def _extract_proper_nouns(self, text: str) -> set[str]:
        """Extract potential proper nouns (capitalized words not at start of sentence)."""
        words = text.split()
        proper_nouns = set()
        for i, word in enumerate(words):
            # Skip first word of each sentence (often capitalized anyway)
            if i == 0:
                continue
            # Check if word is capitalized and not all caps
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                proper_nouns.add(clean_word.lower())
        return proper_nouns

    def _calculate_quality(
        self,
        trigger_text: str,
        response_text: str,
        trigger_turn: Turn,
        response_turn: Turn,
        time_delta: timedelta,
        semantic_similarity: float | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate quality score and flags for a pair.

        Quality factors:
        - Semantic similarity between trigger and response (if provided)
        - Response is not a tapback reaction
        - Response doesn't start a new topic
        - Trigger is not just an acknowledgment with substantive response
        - Response length relative to trigger (not too short, not way too long)
        - Response time (faster = more natural)
        - Multi-message turns (indicates conversational flow)
        - Generic response detection (context-dependent phrases)
        - Question response to non-question (context-dependent)
        - Personal reference detection (proper nouns not in trigger)
        - Emoji-only responses (not templatable)

        Args:
            semantic_similarity: Pre-computed similarity between trigger and response.
                If None, semantic checks are skipped (for backward compatibility).

        Returns:
            Tuple of (quality_score 0.0-1.0, flags dict).
        """
        quality = 1.0
        flags: dict[str, Any] = {}

        trigger_words = len(trigger_text.split())
        response_words = len(response_text.split())

        # === CRITICAL QUALITY FILTERS (new) ===

        # 1. Reaction filter - tapback reactions are NOT real responses
        if self._is_reaction(response_text):
            flags["is_reaction"] = True
            quality *= 0.1  # Severe penalty - almost always filter out

        # 2. Semantic similarity filter (if embedder was used)
        if semantic_similarity is not None:
            flags["semantic_similarity"] = round(semantic_similarity, 3)
            if semantic_similarity < 0.45:
                # Clearly unrelated - likely a topic shift or unrelated message
                flags["low_semantic_similarity"] = True
                quality *= 0.2
            elif semantic_similarity < 0.55:
                # Borderline - may be loosely related but risky
                flags["borderline_semantic_similarity"] = True
                quality *= 0.5

        # 3. Topic shift detection - response introduces a new subject
        if self._is_topic_shift(response_text):
            flags["topic_shift"] = True
            quality *= 0.3

        # 4. Acknowledgment trigger with substantive response
        #    If trigger is "Ok", "Yes" etc. and response is substantive,
        #    the response is likely a NEW topic, not a reply to the ack
        if self._is_acknowledgment_trigger(trigger_text):
            flags["acknowledgment_trigger"] = True
            if self._is_substantive_response(response_text):
                flags["ack_trigger_substantive_response"] = True
                # The response is probably new info, not a reply to the ack
                # Penalty increased from 0.3 to 0.15 for stricter filtering
                quality *= 0.15

        # 4b. Semantic coherence check for acknowledgment triggers
        #     Even if response looks substantive, check if it's actually related
        if self._is_acknowledgment_trigger(trigger_text) and semantic_similarity is not None:
            # Acknowledgment triggers need higher similarity to be valid pairs
            # because they often precede unrelated new topics
            if semantic_similarity < 0.6:
                flags["ack_low_coherence"] = True
                quality *= 0.2

        # === EXISTING QUALITY RULES ===

        # 5. Generic response penalty - these are too context-dependent
        if self._is_generic_response(response_text):
            flags["generic_response"] = True
            quality *= 0.3

        # 6. Question response to non-question penalty
        if response_text.strip().endswith("?") and not trigger_text.strip().endswith("?"):
            flags["question_to_statement"] = True
            quality *= 0.5

        # 7. Personal reference penalty - response mentions names not in trigger
        trigger_nouns = self._extract_proper_nouns(trigger_text)
        response_nouns = self._extract_proper_nouns(response_text)
        unrelated_nouns = response_nouns - trigger_nouns
        if unrelated_nouns:
            flags["unrelated_proper_nouns"] = list(unrelated_nouns)
            quality *= 0.6

        # 8. Emoji-only response penalty
        if self._is_emoji_only(response_text):
            flags["emoji_only"] = True
            quality *= 0.4

        # 9. Very short + generic combination (extra penalty)
        if response_words < 3 and self._is_generic_response(response_text):
            flags["short_generic"] = True
            quality *= 0.2  # Stacks with generic penalty

        # === LENGTH AND TIMING RULES ===

        # Flag very short responses
        if response_words <= 2:
            flags["short_response"] = True
            quality *= 0.8

        # Flag very long responses relative to trigger
        if response_words > trigger_words * 5 and response_words > 20:
            flags["verbose_response"] = True
            quality *= 0.7

        # Bonus for multi-message turns (natural conversation)
        if len(trigger_turn.messages) > 1 or len(response_turn.messages) > 1:
            flags["multi_message_turn"] = True
            quality = min(1.0, quality * 1.1)

        # Tiered delay penalties (expert recommendation)
        hours = time_delta.total_seconds() / 3600
        if hours > self.config.delay_severe_penalty_hours:
            # >12h: different cognitive state, severely down-rank
            flags["very_delayed_response"] = True
            flags["delay_hours"] = round(hours, 1)
            quality *= 0.3
        elif hours > self.config.delay_moderate_penalty_hours:
            # >1h: moderate penalty
            flags["delayed_response"] = True
            flags["delay_hours"] = round(hours, 1)
            quality *= 0.7
        elif time_delta.total_seconds() > 1800:
            # >30 min: slight penalty
            flags["slow_response"] = True
            quality *= 0.9

        # Bonus for quick responses (< 5 min)
        if time_delta.total_seconds() < 300:
            flags["quick_response"] = True
            quality = min(1.0, quality * 1.05)

        return round(quality, 2), flags


def extract_pairs_from_reader(
    chat_db_reader: Any,
    chat_id: str,
    contact_id: int | None = None,
    config: ExtractionConfig | None = None,
    is_group: bool = False,
    embedder: UnifiedEmbedder | None = None,
) -> tuple[list[dict[str, Any]], ExtractionStats]:
    """Extract turn-based pairs from a specific conversation.

    Args:
        chat_db_reader: An open ChatDBReader instance.
        chat_id: The conversation to extract from.
        contact_id: Optional contact ID to associate pairs with.
        config: Extraction configuration.
        is_group: Whether this is a group chat (from conversation metadata).
        embedder: Optional embedder for semantic similarity filtering.

    Returns:
        Tuple of (list of pair dicts ready for DB, extraction stats).
    """
    extractor = TurnBasedExtractor(config, embedder=embedder)

    # Get all messages from the conversation
    messages = chat_db_reader.get_messages(chat_id, limit=10000)

    # Extract pairs
    pairs, stats = extractor.extract_pairs(messages, chat_id, is_group=is_group)

    # Convert to dict format for database
    pair_dicts = [
        {
            "contact_id": contact_id,
            "trigger_text": pair.trigger_text,
            "response_text": pair.response_text,
            "trigger_timestamp": pair.trigger_timestamp,
            "response_timestamp": pair.response_timestamp,
            "chat_id": pair.chat_id,
            "trigger_msg_id": pair.trigger_msg_id,
            "response_msg_id": pair.response_msg_id,
            "trigger_msg_ids": pair.trigger_msg_ids,
            "response_msg_ids": pair.response_msg_ids,
            "context_text": pair.context_text,  # Conversation context for LLM
            "is_group": pair.is_group,
            "quality_score": pair.quality_score,
            "flags": pair.flags,
            "source_timestamp": pair.trigger_timestamp,  # For freshness/decay
        }
        for pair in pairs
    ]

    return pair_dicts, stats


def extract_all_pairs(
    chat_db_reader: Any,
    jarvis_db: Any,
    config: ExtractionConfig | None = None,
    progress_callback: Any | None = None,
    embedder: UnifiedEmbedder | None = None,
) -> dict[str, Any]:
    """Extract turn-based pairs from all conversations with contacts.

    Args:
        chat_db_reader: An open ChatDBReader instance.
        jarvis_db: JarvisDB instance for storing pairs and looking up contacts.
        config: Extraction configuration.
        progress_callback: Optional callback(current, total, chat_id) for progress.
        embedder: Optional embedder for semantic similarity filtering.

    Returns:
        Dictionary with extraction statistics.
    """
    extractor = TurnBasedExtractor(config, embedder=embedder)

    aggregate_stats = {
        "conversations_processed": 0,
        "total_messages_scanned": 0,
        "turns_identified": 0,
        "candidate_pairs": 0,
        "pairs_extracted": 0,
        "pairs_added": 0,
        "pairs_skipped_duplicate": 0,
        "dropped_by_reason": {
            "short_trigger": 0,
            "short_response": 0,
            "long_trigger": 0,
            "long_response": 0,
            "no_text": 0,
            "time_gap": 0,
        },
        "flagged_by_reason": {
            "reaction": 0,
            "low_similarity": 0,
            "topic_shift": 0,
            "ack_substantive": 0,
        },
        "errors": [],
    }

    # Get all conversations
    conversations = chat_db_reader.get_conversations(limit=1000)
    total = len(conversations)

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            # Look up contact
            contact = jarvis_db.get_contact_by_chat_id(conv.chat_id)
            contact_id = contact.id if contact else None

            # Get is_group from conversation metadata (not string pattern matching)
            is_group = getattr(conv, "is_group", False)

            # Get messages
            messages = chat_db_reader.get_messages(conv.chat_id, limit=10000)

            # Extract pairs
            pairs, stats = extractor.extract_pairs(messages, conv.chat_id, is_group=is_group)

            # Update aggregate stats
            aggregate_stats["total_messages_scanned"] += stats.total_messages_scanned
            aggregate_stats["turns_identified"] += stats.turns_identified
            aggregate_stats["candidate_pairs"] += stats.candidate_pairs
            aggregate_stats["dropped_by_reason"]["short_trigger"] += stats.dropped_short_trigger
            aggregate_stats["dropped_by_reason"]["short_response"] += stats.dropped_short_response
            aggregate_stats["dropped_by_reason"]["long_trigger"] += stats.dropped_long_trigger
            aggregate_stats["dropped_by_reason"]["long_response"] += stats.dropped_long_response
            aggregate_stats["dropped_by_reason"]["no_text"] += stats.dropped_no_text
            aggregate_stats["dropped_by_reason"]["time_gap"] += stats.dropped_time_gap
            # Track quality flags
            aggregate_stats["flagged_by_reason"]["reaction"] += stats.flagged_reaction
            aggregate_stats["flagged_by_reason"]["low_similarity"] += stats.flagged_low_similarity
            aggregate_stats["flagged_by_reason"]["topic_shift"] += stats.flagged_topic_shift
            aggregate_stats["flagged_by_reason"]["ack_substantive"] += stats.flagged_ack_substantive

            # Convert and add to database
            pair_dicts = [
                {
                    "contact_id": contact_id,
                    "trigger_text": pair.trigger_text,
                    "response_text": pair.response_text,
                    "trigger_timestamp": pair.trigger_timestamp,
                    "response_timestamp": pair.response_timestamp,
                    "chat_id": pair.chat_id,
                    "trigger_msg_id": pair.trigger_msg_id,
                    "response_msg_id": pair.response_msg_id,
                    "trigger_msg_ids": pair.trigger_msg_ids,
                    "response_msg_ids": pair.response_msg_ids,
                    "context_text": pair.context_text,  # Conversation context for LLM
                    "is_group": pair.is_group,
                    "quality_score": pair.quality_score,
                    "flags": pair.flags,
                    "source_timestamp": pair.trigger_timestamp,
                }
                for pair in pairs
            ]

            added = jarvis_db.add_pairs_bulk(pair_dicts)

            aggregate_stats["conversations_processed"] += 1
            aggregate_stats["pairs_extracted"] += len(pairs)
            aggregate_stats["pairs_added"] += added
            aggregate_stats["pairs_skipped_duplicate"] += len(pairs) - added

        except Exception as e:
            logger.warning("Error extracting from %s: %s", conv.chat_id, e)
            aggregate_stats["errors"].append({"chat_id": conv.chat_id, "error": str(e)})

    return aggregate_stats


# =============================================================================
# V2 Exchange-Based Extraction (with validity gates)
# =============================================================================


@dataclass
class ExchangeBuilderConfig:
    """Configuration for exchange-based extraction (v2 pipeline).

    Attributes:
        time_gap_boundary_minutes: Time gap that marks a new conversation thread.
        trigger_max_messages: Max messages to include in trigger span.
        trigger_max_duration_minutes: Max duration for trigger span.
        response_max_messages: Max messages to include in response span.
        response_max_duration_minutes: Max duration for response span.
        context_window_size: Number of previous messages for context.
        max_response_delay_hours: Hard cutoff for response time.
        min_trigger_length: Min chars for trigger.
        min_response_length: Min chars for response.
        max_trigger_length: Max chars for trigger.
        max_response_length: Max chars for response.
    """

    time_gap_boundary_minutes: float = 30.0
    trigger_max_messages: int = 5
    trigger_max_duration_minutes: float = 3.0
    response_max_messages: int = 5
    response_max_duration_minutes: float = 3.0
    context_window_size: int = 20
    max_response_delay_hours: float = 24.0  # 24h hard cutoff (stricter than v1)
    min_trigger_length: int = 2
    min_response_length: int = 15
    max_trigger_length: int = 500
    max_response_length: int = 400


@dataclass
class ExchangeExtractionStats:
    """Statistics from exchange-based extraction."""

    total_messages_scanned: int = 0
    exchanges_built: int = 0
    gate_a_passed: int = 0
    gate_a_rejected: int = 0
    gate_b_accepted: int = 0
    gate_b_borderline: int = 0
    gate_b_rejected: int = 0
    gate_c_accepted: int = 0
    gate_c_rejected: int = 0
    gate_c_uncertain: int = 0
    final_valid: int = 0
    final_invalid: int = 0
    final_uncertain: int = 0
    gate_a_rejection_reasons: dict[str, int] = field(default_factory=dict)


class ExchangeBuilder:
    """Build candidate exchanges with proper boundaries.

    Implements time-gap boundaries, speaker-run caps, and response window caps
    to avoid monster spans that confuse downstream scorers.
    """

    def __init__(self, config: ExchangeBuilderConfig | None = None) -> None:
        """Initialize exchange builder."""
        self.config = config or ExchangeBuilderConfig()

    def build_candidates(
        self,
        messages: list[Message],
        chat_id: str,
        contact_id: int | None = None,
    ) -> list[CandidateExchange]:
        """Build candidate exchanges from messages.

        Args:
            messages: List of messages, sorted by date ascending.
            chat_id: The conversation ID.
            contact_id: Optional contact ID.

        Returns:
            List of CandidateExchange objects.
        """
        from jarvis.exchange import CandidateExchange, ContextMessage
        from jarvis.text_normalizer import get_attachment_token, is_reaction, normalize_text

        if not messages:
            return []

        # Sort by date ascending
        sorted_messages = sorted(messages, key=lambda m: m.date)

        # Convert to ContextMessage format with normalization
        context_msgs: list[ContextMessage] = []
        for msg in sorted_messages:
            # Skip system messages
            if msg.is_system_message:
                continue

            # Determine flags
            flags: set[str] = set()
            raw_text = msg.text or ""

            if is_reaction(raw_text):
                flags.add("reaction")
                normalized = ""
            else:
                normalized = normalize_text(raw_text)

            # Handle attachment-only
            if not normalized and msg.attachments:
                att_type = msg.attachments[0].mime_type if msg.attachments else None
                normalized = get_attachment_token(att_type)
                flags.add("attachment")

            # Check for emoji-only after normalization
            from jarvis.text_normalizer import is_emoji_only

            if normalized and is_emoji_only(normalized):
                flags.add("emoji_only")

            context_msgs.append(
                ContextMessage(
                    speaker="me" if msg.is_from_me else "them",
                    timestamp=msg.date,
                    text=normalized,
                    flags=flags,
                    raw_text=raw_text if raw_text != normalized else None,
                )
            )

        # Build exchanges by finding (them -> me) transitions
        candidates: list[CandidateExchange] = []
        i = 0
        n = len(context_msgs)

        while i < n:
            msg = context_msgs[i]

            # Skip if this is from me (we want triggers from them)
            if msg.speaker == "me":
                i += 1
                continue

            # Skip reactions as triggers
            if "reaction" in msg.flags:
                i += 1
                continue

            # Build trigger span (with caps)
            trigger_span: list[ContextMessage] = [msg]
            trigger_msg_ids: list[int] = [sorted_messages[i].id]
            trigger_start = msg.timestamp
            j = i + 1

            # Continue adding to trigger span if:
            # - Same speaker (them)
            # - Within max duration
            # - Under max message count
            while j < n and len(trigger_span) < self.config.trigger_max_messages:
                next_msg = context_msgs[j]

                # Must be same speaker
                if next_msg.speaker != "them":
                    break

                # Skip reactions
                if "reaction" in next_msg.flags:
                    j += 1
                    continue

                # Check duration from trigger start
                duration_mins = (next_msg.timestamp - trigger_start).total_seconds() / 60
                if duration_mins > self.config.trigger_max_duration_minutes:
                    break

                trigger_span.append(next_msg)
                trigger_msg_ids.append(sorted_messages[j].id)
                j += 1

            # Now look for response (my messages)
            if j >= n or context_msgs[j].speaker != "me":
                # No response found
                i = j if j < n else i + 1
                continue

            # Check time gap (must respond within max delay)
            time_gap = context_msgs[j].timestamp - trigger_span[-1].timestamp
            gap_hours = time_gap.total_seconds() / 3600
            if gap_hours > self.config.max_response_delay_hours:
                i = j
                continue

            # Check for conversation break (time gap > boundary)
            gap_mins = time_gap.total_seconds() / 60
            if gap_mins > self.config.time_gap_boundary_minutes:
                # Too long - this response might be to something else
                i = j
                continue

            # Build response span (with caps)
            response_span: list[ContextMessage] = [context_msgs[j]]
            response_msg_ids: list[int] = [sorted_messages[j].id]
            response_start = context_msgs[j].timestamp
            k = j + 1

            # Continue adding to response span if:
            # - Same speaker (me)
            # - Within max duration
            # - Under max message count
            while k < n and len(response_span) < self.config.response_max_messages:
                next_msg = context_msgs[k]

                # Must be same speaker
                if next_msg.speaker != "me":
                    break

                # Skip reactions
                if "reaction" in next_msg.flags:
                    k += 1
                    continue

                # Check duration from response start
                duration_mins = (next_msg.timestamp - response_start).total_seconds() / 60
                if duration_mins > self.config.response_max_duration_minutes:
                    break

                response_span.append(next_msg)
                response_msg_ids.append(sorted_messages[k].id)
                k += 1

            # Build context window (previous messages before trigger)
            context_start = max(0, i - self.config.context_window_size)
            context_window = context_msgs[context_start:i]

            # Create candidate exchange
            exchange = CandidateExchange(
                trigger_span=trigger_span,
                response_span=response_span,
                context_window=context_window,
                chat_id=chat_id,
                contact_id=contact_id,
                trigger_msg_ids=trigger_msg_ids,
                response_msg_ids=response_msg_ids,
            )
            candidates.append(exchange)

            # Move past the response span
            i = k

        return candidates


def extract_all_pairs_v2(
    chat_db_reader: Any,
    jarvis_db: Any,
    config: ExchangeBuilderConfig | None = None,
    embedder: Any | None = None,
    nli_model: Any | None = None,
    progress_callback: Any | None = None,
    skip_nli: bool = False,
) -> dict[str, Any]:
    """Extract pairs using v2 exchange-based pipeline with validity gates.

    Args:
        chat_db_reader: An open ChatDBReader instance.
        jarvis_db: JarvisDB instance.
        config: Extraction configuration.
        embedder: MLXEmbedder for Gate B (optional).
        nli_model: NLI model for Gate C (optional).
        progress_callback: Optional callback(current, total, chat_id).
        skip_nli: If True, skip Gate C even if nli_model provided.

    Returns:
        Dictionary with extraction statistics.
    """
    import json as json_module

    from jarvis.validity_gate import GateConfig, ValidityGate

    builder = ExchangeBuilder(config)
    gate_config = GateConfig()
    gate = ValidityGate(
        embedder=embedder,
        nli_model=None if skip_nli else nli_model,
        config=gate_config,
    )

    stats = ExchangeExtractionStats()
    aggregate_stats = {
        "conversations_processed": 0,
        "total_messages_scanned": 0,
        "exchanges_built": 0,
        "pairs_added": 0,
        "pairs_skipped_duplicate": 0,
        "gate_a_rejected": 0,
        "gate_b_rejected": 0,
        "gate_c_rejected": 0,
        "final_valid": 0,
        "final_invalid": 0,
        "final_uncertain": 0,
        "gate_a_reasons": {},
        "errors": [],
    }

    # Get all conversations
    conversations = chat_db_reader.get_conversations(limit=1000)
    total = len(conversations)

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            # Look up contact
            contact = jarvis_db.get_contact_by_chat_id(conv.chat_id)
            contact_id = contact.id if contact else None

            is_group = getattr(conv, "is_group", False)

            # Get messages
            messages = chat_db_reader.get_messages(conv.chat_id, limit=10000)
            stats.total_messages_scanned += len(messages)
            aggregate_stats["total_messages_scanned"] += len(messages)

            # Build candidate exchanges
            exchanges = builder.build_candidates(messages, conv.chat_id, contact_id)
            stats.exchanges_built += len(exchanges)
            aggregate_stats["exchanges_built"] += len(exchanges)

            # Validate and store each exchange
            for exchange in exchanges:
                # Run validity gates
                result = gate.validate(exchange)

                # Track gate stats
                if not result.gate_a_passed:
                    stats.gate_a_rejected += 1
                    aggregate_stats["gate_a_rejected"] += 1
                    reason = result.gate_a_reason or "unknown"
                    stats.gate_a_rejection_reasons[reason] = (
                        stats.gate_a_rejection_reasons.get(reason, 0) + 1
                    )
                    aggregate_stats["gate_a_reasons"][reason] = (
                        aggregate_stats["gate_a_reasons"].get(reason, 0) + 1
                    )
                else:
                    stats.gate_a_passed += 1

                if result.gate_b_band == "accept":
                    stats.gate_b_accepted += 1
                elif result.gate_b_band == "borderline":
                    stats.gate_b_borderline += 1
                elif result.gate_b_band == "reject":
                    stats.gate_b_rejected += 1
                    aggregate_stats["gate_b_rejected"] += 1

                if result.gate_c_verdict == "accept":
                    stats.gate_c_accepted += 1
                elif result.gate_c_verdict == "reject":
                    stats.gate_c_rejected += 1
                    aggregate_stats["gate_c_rejected"] += 1
                elif result.gate_c_verdict == "uncertain":
                    stats.gate_c_uncertain += 1

                if result.final_status == "valid":
                    stats.final_valid += 1
                    aggregate_stats["final_valid"] += 1
                elif result.final_status == "invalid":
                    stats.final_invalid += 1
                    aggregate_stats["final_invalid"] += 1
                else:
                    stats.final_uncertain += 1
                    aggregate_stats["final_uncertain"] += 1

                # Store the pair with gate results
                context_json = json_module.dumps(exchange.context_to_json())
                gate_c_scores_json = (
                    json_module.dumps(result.gate_c_scores) if result.gate_c_scores else None
                )

                pair = jarvis_db.add_validated_pair(
                    trigger_text=exchange.trigger_text,
                    response_text=exchange.response_text,
                    trigger_timestamp=exchange.trigger_start_time,
                    response_timestamp=exchange.response_start_time,
                    chat_id=exchange.chat_id,
                    contact_id=exchange.contact_id,
                    trigger_msg_id=exchange.primary_trigger_msg_id,
                    response_msg_id=exchange.primary_response_msg_id,
                    trigger_msg_ids=exchange.trigger_msg_ids,
                    response_msg_ids=exchange.response_msg_ids,
                    is_group=is_group,
                    gate_a_passed=result.gate_a_passed,
                    gate_b_score=result.gate_b_score,
                    gate_c_verdict=result.gate_c_verdict,
                    validity_status=result.final_status,
                    context_json=context_json,
                    gate_a_reason=result.gate_a_reason,
                    gate_c_scores_json=gate_c_scores_json,
                )

                if pair:
                    aggregate_stats["pairs_added"] += 1
                else:
                    aggregate_stats["pairs_skipped_duplicate"] += 1

            aggregate_stats["conversations_processed"] += 1

        except Exception as e:
            logger.warning("Error extracting from %s: %s", conv.chat_id, e)
            aggregate_stats["errors"].append({"chat_id": conv.chat_id, "error": str(e)})

    return aggregate_stats
