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

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from contracts.imessage import Message

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for turn-based pair extraction."""

    # Time window for bundling messages into a turn (same speaker)
    turn_bundle_minutes: float = 10.0

    # Time window for considering a turn as a response to previous turn
    max_response_delay_hours: float = 1.0

    # Minimum message/turn length (characters) to include
    min_trigger_length: int = 2
    min_response_length: int = 2

    # Maximum turn length (characters) - very long = not templatable
    max_trigger_length: int = 500
    max_response_length: int = 400

    # Skip attachment-only messages when bundling
    skip_attachment_only: bool = True

    # Skip system messages (group events, etc.)
    skip_system_messages: bool = True


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


class TurnBasedExtractor:
    """Extracts turn-based (trigger, response) pairs from message history.

    Groups consecutive messages from the same speaker into "turns",
    then pairs incoming turns with outgoing response turns.
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """Initialize extractor with configuration."""
        self.config = config or ExtractionConfig()

    def extract_pairs(
        self,
        messages: list[Message],
        chat_id: str,
    ) -> tuple[list[ExtractedPair], ExtractionStats]:
        """Extract turn-based pairs from a conversation's messages.

        Args:
            messages: List of messages, sorted by date ascending (oldest first).
            chat_id: The conversation ID.

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
                    turn, response_turn, chat_id, time_delta, stats, context_turns
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
    ) -> ExtractedPair | None:
        """Create a pair from trigger and response turns, with validation.

        Args:
            context_turns: Previous turns before the trigger (for LLM context).

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
        if len(response_text) < self.config.min_response_length:
            stats.dropped_short_response += 1
            return None
        if len(response_text) > self.config.max_response_length:
            stats.dropped_long_response += 1
            return None

        # Calculate quality score
        quality_score, flags = self._calculate_quality(
            trigger_text, response_text, trigger_turn, response_turn, time_delta
        )

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
    GENERIC_RESPONSES = frozenset({
        "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "no", "nope", "nah",
        "sure", "thanks", "thank you", "thx", "ty", "np", "cool", "nice", "good",
        "great", "awesome", "alright", "sounds good", "got it", "lol", "haha",
        "hahaha", "lmao", "omw", "on my way", "be there soon", "see you", "bye",
        "later", "ttyl", "👍", "👌", "🙏", "😂", "😊", "❤️", "🔥", "💯",
    })

    # Emoji pattern for detecting emoji-only responses
    EMOJI_PATTERN = re.compile(
        r'^[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F'
        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\s]+$'
    )

    def _is_generic_response(self, text: str) -> bool:
        """Check if response is a generic/context-dependent phrase."""
        normalized = text.lower().strip()
        return normalized in self.GENERIC_RESPONSES

    def _is_emoji_only(self, text: str) -> bool:
        """Check if response contains only emojis."""
        return bool(self.EMOJI_PATTERN.match(text.strip()))

    def _extract_proper_nouns(self, text: str) -> set[str]:
        """Extract potential proper nouns (capitalized words not at start of sentence)."""
        words = text.split()
        proper_nouns = set()
        for i, word in enumerate(words):
            # Skip first word of each sentence (often capitalized anyway)
            if i == 0:
                continue
            # Check if word is capitalized and not all caps
            clean_word = re.sub(r'[^\w]', '', word)
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
    ) -> tuple[float, dict[str, Any]]:
        """Calculate quality score and flags for a pair.

        Quality factors:
        - Response length relative to trigger (not too short, not way too long)
        - Response time (faster = more natural)
        - Multi-message turns (indicates conversational flow)
        - Generic response detection (context-dependent phrases)
        - Question response to non-question (context-dependent)
        - Personal reference detection (proper nouns not in trigger)
        - Emoji-only responses (not templatable)

        Returns:
            Tuple of (quality_score 0.0-1.0, flags dict).
        """
        quality = 1.0
        flags: dict[str, Any] = {}

        trigger_words = len(trigger_text.split())
        response_words = len(response_text.split())

        # === NEW QUALITY RULES ===

        # 1. Generic response penalty - these are too context-dependent
        if self._is_generic_response(response_text):
            flags["generic_response"] = True
            quality *= 0.3

        # 2. Question response to non-question penalty
        if response_text.strip().endswith("?") and not trigger_text.strip().endswith("?"):
            flags["question_to_statement"] = True
            quality *= 0.5

        # 3. Personal reference penalty - response mentions names not in trigger
        trigger_nouns = self._extract_proper_nouns(trigger_text)
        response_nouns = self._extract_proper_nouns(response_text)
        unrelated_nouns = response_nouns - trigger_nouns
        if unrelated_nouns:
            flags["unrelated_proper_nouns"] = list(unrelated_nouns)
            quality *= 0.6

        # 4. Emoji-only response penalty
        if self._is_emoji_only(response_text):
            flags["emoji_only"] = True
            quality *= 0.4

        # 5. Very short + generic combination (extra penalty)
        if response_words < 3 and self._is_generic_response(response_text):
            flags["short_generic"] = True
            quality *= 0.2  # Stacks with generic penalty

        # === EXISTING QUALITY RULES ===

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

        # Slight penalty for slow responses (> 30 min)
        if time_delta.total_seconds() > 1800:
            flags["delayed_response"] = True
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
) -> tuple[list[dict[str, Any]], ExtractionStats]:
    """Extract turn-based pairs from a specific conversation.

    Args:
        chat_db_reader: An open ChatDBReader instance.
        chat_id: The conversation to extract from.
        contact_id: Optional contact ID to associate pairs with.
        config: Extraction configuration.

    Returns:
        Tuple of (list of pair dicts ready for DB, extraction stats).
    """
    extractor = TurnBasedExtractor(config)

    # Get all messages from the conversation
    messages = chat_db_reader.get_messages(chat_id, limit=10000)

    # Extract pairs
    pairs, stats = extractor.extract_pairs(messages, chat_id)

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
) -> dict[str, Any]:
    """Extract turn-based pairs from all conversations with contacts.

    Args:
        chat_db_reader: An open ChatDBReader instance.
        jarvis_db: JarvisDB instance for storing pairs and looking up contacts.
        config: Extraction configuration.
        progress_callback: Optional callback(current, total, chat_id) for progress.

    Returns:
        Dictionary with extraction statistics.
    """
    extractor = TurnBasedExtractor(config)

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

            # Get messages
            messages = chat_db_reader.get_messages(conv.chat_id, limit=10000)

            # Extract pairs
            pairs, stats = extractor.extract_pairs(messages, conv.chat_id)

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
