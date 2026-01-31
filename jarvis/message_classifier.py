"""Message Classifier - Hybrid rule-based + embeddings classification.

Classifies incoming messages by type, context requirement, and reply requirement
to enable smarter routing decisions.

Uses a two-phase approach:
1. Rule-based classification (fast, high precision) for clear patterns
2. Embedding-based classification (semantic) for ambiguous cases

Usage:
    from jarvis.message_classifier import get_message_classifier, classify_message

    # Quick classification
    result = classify_message("Can you come to dinner?")
    print(f"Type: {result.message_type.value}")
    print(f"Reply needed: {result.reply_requirement.value}")

    # Or use the classifier directly
    classifier = get_message_classifier()
    result = classifier.classify("ok sounds good")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class MessageType(Enum):
    """Types of messages based on linguistic structure."""

    QUESTION_YESNO = "question_yesno"  # "Can you come?", "Are you free?"
    QUESTION_INFO = "question_info"  # "What time?", "Where is it?"
    QUESTION_OPEN = "question_open"  # "What do you think?", "How was it?"
    REQUEST_ACTION = "request_action"  # "Please send me...", "Can you get..."
    STATEMENT = "statement"  # "I'm on my way", "The meeting is at 5"
    ACKNOWLEDGMENT = "acknowledgment"  # "ok", "got it", "sure"
    REACTION = "reaction"  # "haha", "lol", "nice"
    GREETING = "greeting"  # "hi", "hello", "hey"
    FAREWELL = "farewell"  # "bye", "ttyl", "later"


class ContextRequirement(Enum):
    """How much context is needed to understand the message."""

    SELF_CONTAINED = "self_contained"  # Makes sense alone
    NEEDS_THREAD = "needs_thread"  # Needs recent conversation context
    NEEDS_SHARED = "needs_shared"  # Needs shared knowledge/history
    VAGUE = "vague"  # Needs clarification


class ReplyRequirement(Enum):
    """What type of reply is expected."""

    NO_REPLY = "no_reply"  # No response needed
    QUICK_ACK = "quick_ack"  # Simple acknowledgment
    YES_NO = "yes_no"  # Yes/no answer expected
    INFO_RESPONSE = "info_response"  # Information expected
    ACTION_COMMIT = "action_commit"  # Commitment to action expected
    CLARIFY = "clarify"  # Need to ask for clarification


class InfoType(Enum):
    """Type of information being requested (for QUESTION_INFO)."""

    TIME = "time"  # when
    LOCATION = "location"  # where
    PERSON = "person"  # who
    REASON = "reason"  # why
    METHOD = "method"  # how
    QUANTITY = "quantity"  # how many
    PREFERENCE = "preference"  # which
    CONFIRMATION = "confirmation"  # confirming something
    GENERAL = "general"  # other


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MessageClassification:
    """Result of classifying a message."""

    message_type: MessageType
    type_confidence: float  # 0.0 to 1.0
    context_requirement: ContextRequirement
    reply_requirement: ReplyRequirement
    info_type: InfoType | None = None  # Only for QUESTION_INFO
    matched_rule: str | None = None  # Which rule matched (if rule-based)
    classification_method: str = "rule"  # 'rule' or 'embedding'


# =============================================================================
# Rule Patterns
# =============================================================================

# Compiled regex patterns for rule-based classification
# High confidence patterns that don't need embedding fallback

# Import centralized acknowledgment detection from text_normalizer
# This ensures exact matching (not substring matching) across all modules
from jarvis.text_normalizer import is_acknowledgment_only


# Legacy wrapper for regex-based checking - now uses centralized exact matching
def _match_acknowledgment(text: str) -> bool:
    """Check if text is an acknowledgment using centralized exact matching."""
    return is_acknowledgment_only(text)


# Keep REACTION_PATTERNS for emotional reactions (lol, haha, wow, etc.)
# These are different from acknowledgments and need context-aware responses

REACTION_PATTERNS = re.compile(
    r"^(?:lol|lmao|rofl|haha+|hehe+|omg|wow|nice!*|"
    r"dope|sick|fire|lit|dead|dying|"
    r"[\U0001F600-\U0001F64F]+|[\U0001F923\U0001F602\U0001F605\U0001F606]+)$",  # Emojis
    re.IGNORECASE,
)

GREETING_PATTERNS = re.compile(
    r"^(?:hi|hey|hello|yo|sup|what'?s up|howdy|hiya|morning|"
    r"good morning|good afternoon|good evening|"
    r"what up|wassup|wsg)\b",
    re.IGNORECASE,
)

FAREWELL_PATTERNS = re.compile(
    r"^(?:bye|goodbye|gn|good night|ttyl|later|cya|see ya|"
    r"talk later|peace|take care|nite|night)\b",
    re.IGNORECASE,
)

# Info question patterns - what/when/where/who/which/how
INFO_QUESTION_PATTERNS = re.compile(
    r"^(?:what|when|where|who|which|how)\s+",
    re.IGNORECASE,
)

# Yes/no question patterns - can/could/will/would/should/do/does/is/are
YESNO_QUESTION_PATTERNS = re.compile(
    r"^(?:can|could|will|would|should|do|does|is|are|have|has|"
    r"did|was|were|shall|may|might)\s+(?:you|we|i|he|she|they|it)\b",
    re.IGNORECASE,
)

# Request patterns
REQUEST_PATTERNS = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|"
    r"do you mind\s+|i need you to\s+|send me\s+|get me\s+)",
    re.IGNORECASE,
)

# Vague reference patterns that need context
VAGUE_REFERENCE_PATTERNS = re.compile(
    r"\b(?:that|it|this|those|these|the thing|what you said|"
    r"what we discussed|the other|that one)\b",
    re.IGNORECASE,
)

# Time patterns for info type detection
TIME_PATTERNS = re.compile(
    r"\b(?:when|what time|how long|how soon|what day|"
    r"until when|by when|since when)\b",
    re.IGNORECASE,
)

# Location patterns
LOCATION_PATTERNS = re.compile(
    r"\b(?:where|what place|which location|what address)\b",
    re.IGNORECASE,
)

# Person patterns
PERSON_PATTERNS = re.compile(
    r"\b(?:who|whom|whose)\b",
    re.IGNORECASE,
)

# Reason patterns
REASON_PATTERNS = re.compile(
    r"\b(?:why|how come|what for|for what reason)\b",
    re.IGNORECASE,
)

# Method patterns
METHOD_PATTERNS = re.compile(
    r"^how\s+(?:do|can|should|would|did)\b",
    re.IGNORECASE,
)

# Quantity patterns
QUANTITY_PATTERNS = re.compile(
    r"\b(?:how many|how much|how often)\b",
    re.IGNORECASE,
)


# =============================================================================
# Embedding Centroids (Example texts for each MessageType)
# =============================================================================

MESSAGE_TYPE_EXAMPLES: dict[MessageType, list[str]] = {
    MessageType.QUESTION_YESNO: [
        "Can you come to dinner?",
        "Are you free tonight?",
        "Will you be there?",
        "Do you want to join?",
        "Is that okay with you?",
        "Can we meet tomorrow?",
        "Would you like to come?",
        "Did you get my message?",
        "Have you seen it?",
        "Should we do this?",
        "Is this still happening?",
        "Do you need anything?",
        "Can I bring something?",
        "Are we still on?",
        "Should I wait?",
    ],
    MessageType.QUESTION_INFO: [
        "What time is the meeting?",
        "Where should we meet?",
        "When are you arriving?",
        "Who else is coming?",
        "How do I get there?",
        "What's the address?",
        "When does it start?",
        "How much does it cost?",
        "What should I bring?",
        "Which restaurant?",
        "How long will it take?",
        "What happened?",
        "Where did you go?",
        "Who told you?",
        "How did it go?",
    ],
    MessageType.QUESTION_OPEN: [
        "What do you think?",
        "How was your day?",
        "What's on your mind?",
        "How do you feel about it?",
        "What are your thoughts?",
        "How have you been?",
        "What's new with you?",
        "How was the trip?",
        "What did you think of the movie?",
        "How's everything going?",
        "What would you do?",
        "How should we handle this?",
        "What's your take on it?",
        "What do you want to do?",
        "How does that sound?",
    ],
    MessageType.REQUEST_ACTION: [
        "Please send me the file",
        "Can you pick me up?",
        "Could you let me know?",
        "Would you mind checking?",
        "Can you get some groceries?",
        "Please call me back",
        "Can you forward this?",
        "Would you send me the link?",
        "Can you remind me later?",
        "Please let them know",
        "Could you help me with this?",
        "Can you take care of it?",
        "Please bring the documents",
        "Can you book the table?",
        "Would you grab me a coffee?",
    ],
    MessageType.STATEMENT: [
        "I'm on my way",
        "The meeting is at 5",
        "I'll be there soon",
        "I finished the report",
        "We're running late",
        "I got the tickets",
        "Everything is ready",
        "I saw your message",
        "I'm working on it",
        "The package arrived",
        "I talked to them",
        "I made a reservation",
        "I found the place",
        "I just left",
        "I'm almost done",
    ],
    MessageType.ACKNOWLEDGMENT: [
        "ok",
        "sounds good",
        "got it",
        "sure thing",
        "no problem",
        "alright",
        "perfect",
        "works for me",
        "understood",
        "copy that",
        "noted",
        "will do",
        "makes sense",
        "fair enough",
        "all good",
    ],
    MessageType.REACTION: [
        "haha that's funny",
        "lol",
        "lmao",
        "omg",
        "wow",
        "nice!",
        "awesome!",
        "that's hilarious",
        "love it",
        "so true",
        "same",
        "mood",
        "relatable",
        "for real",
        "I know right",
    ],
    MessageType.GREETING: [
        "Hey there!",
        "Hi, how are you?",
        "Hello!",
        "Good morning",
        "What's up?",
        "Hey!",
        "Hi!",
        "Yo",
        "Howdy",
        "Good afternoon",
        "Hey, long time no see",
        "Hi, hope you're doing well",
        "Hello friend",
        "Hey stranger",
        "Morning!",
    ],
    MessageType.FAREWELL: [
        "Bye!",
        "Talk to you later",
        "See you soon",
        "Good night",
        "Take care",
        "Later!",
        "Catch you later",
        "Peace out",
        "Have a good one",
        "See ya",
        "Goodbye",
        "Night!",
        "Until next time",
        "So long",
        "Cya!",
    ],
}


# =============================================================================
# Message Classifier
# =============================================================================


class MessageClassifier:
    """Hybrid rule-based + embedding message classifier.

    Uses fast rule-based classification for clear patterns, falling back
    to embedding similarity for ambiguous cases.

    Thread Safety:
        This class is thread-safe for concurrent classify() calls.
    """

    # Confidence thresholds
    RULE_CONFIDENCE = 0.95  # High confidence for rule matches
    EMBEDDING_THRESHOLD = 0.65  # Minimum embedding similarity to use
    RULE_FALLBACK_THRESHOLD = 0.7  # If rule confidence below this, try embeddings

    def __init__(self) -> None:
        """Initialize the classifier."""
        self._embedder = None
        self._centroids: dict[MessageType, NDArray[np.float32]] | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def _get_embedder(self) -> Any:
        """Get the embedder for semantic classification."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _ensure_centroids_computed(self) -> None:
        """Compute and cache centroids for each message type."""
        if self._centroids is not None:
            return

        with self._lock:
            if self._centroids is not None:
                return

            if self._init_attempted:
                return  # Already tried and failed

            self._init_attempted = True

            try:
                embedder = self._get_embedder()

                centroids: dict[MessageType, NDArray[np.float32]] = {}
                for msg_type, examples in MESSAGE_TYPE_EXAMPLES.items():
                    # Compute embeddings for all examples
                    embeddings = embedder.encode(examples, normalize=True)
                    # Compute centroid (mean embedding)
                    centroid = np.mean(embeddings, axis=0)
                    # Normalize the centroid
                    centroid = centroid / np.linalg.norm(centroid)
                    centroids[msg_type] = centroid.astype(np.float32)

                self._centroids = centroids
                logger.info("Computed centroids for %d message types", len(centroids))

            except Exception as e:
                logger.warning("Failed to compute message type centroids: %s", e)
                self._centroids = None

    def _classify_by_rules(self, text: str) -> tuple[MessageType | None, float, str | None]:
        """Classify message using rule-based patterns.

        Args:
            text: The message text to classify.

        Returns:
            Tuple of (MessageType or None, confidence, matched_rule_name)
        """
        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        # Check acknowledgment using centralized exact matching
        if _match_acknowledgment(text_stripped):
            return MessageType.ACKNOWLEDGMENT, self.RULE_CONFIDENCE, "acknowledgment"

        # Check reaction
        if REACTION_PATTERNS.match(text_stripped):
            return MessageType.REACTION, self.RULE_CONFIDENCE, "reaction"

        # Check greeting
        if GREETING_PATTERNS.match(text_lower):
            return MessageType.GREETING, self.RULE_CONFIDENCE, "greeting"

        # Check farewell
        if FAREWELL_PATTERNS.match(text_lower):
            return MessageType.FAREWELL, self.RULE_CONFIDENCE, "farewell"

        # Check request (before questions, as "can you..." could be both)
        if REQUEST_PATTERNS.match(text_lower):
            return MessageType.REQUEST_ACTION, self.RULE_CONFIDENCE, "request"

        # Check info questions
        if INFO_QUESTION_PATTERNS.match(text_lower):
            return MessageType.QUESTION_INFO, self.RULE_CONFIDENCE, "info_question"

        # Check yes/no questions
        if YESNO_QUESTION_PATTERNS.match(text_lower):
            return MessageType.QUESTION_YESNO, self.RULE_CONFIDENCE, "yesno_question"

        # Check for question mark at end (heuristic for questions)
        if text_stripped.endswith("?"):
            # Could be any question type - lower confidence
            return MessageType.QUESTION_OPEN, 0.6, "question_mark"

        # No clear rule match
        return None, 0.0, None

    def _classify_by_embedding(
        self,
        text: str,
        embedder: Any | None = None,
    ) -> tuple[MessageType | None, float]:
        """Classify message using embedding similarity to centroids.

        Args:
            text: The message text to classify.

        Returns:
            Tuple of (MessageType or None, confidence)
        """
        try:
            self._ensure_centroids_computed()

            if self._centroids is None:
                return None, 0.0

            query_embedder = embedder or self._get_embedder()
            query_embedding = query_embedder.encode([text], normalize=True)[0]

            # Find most similar centroid
            best_type = None
            best_similarity = -1.0

            for msg_type, centroid in self._centroids.items():
                similarity = float(np.dot(query_embedding, centroid))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_type = msg_type

            if best_similarity >= self.EMBEDDING_THRESHOLD:
                return best_type, best_similarity

            return None, best_similarity

        except Exception as e:
            logger.warning("Embedding classification failed: %s", e)
            return None, 0.0

    def _classify_context(self, text: str) -> ContextRequirement:
        """Determine how much context is needed to understand the message.

        Args:
            text: The message text.

        Returns:
            ContextRequirement enum value.
        """
        text_lower = text.lower()

        # Check for vague references
        if VAGUE_REFERENCE_PATTERNS.search(text_lower):
            return ContextRequirement.NEEDS_THREAD

        # Very short messages often need context
        word_count = len(text.split())
        if word_count <= 2:
            # But acknowledgments and reactions are self-contained
            if _match_acknowledgment(text.strip()):
                return ContextRequirement.SELF_CONTAINED
            if REACTION_PATTERNS.match(text.strip()):
                return ContextRequirement.SELF_CONTAINED
            return ContextRequirement.NEEDS_THREAD

        # Pronouns without clear referent often need context
        pronoun_patterns = re.compile(r"\b(?:he|she|they|them|him|her)\b", re.IGNORECASE)
        if pronoun_patterns.search(text_lower) and word_count < 8:
            return ContextRequirement.NEEDS_THREAD

        return ContextRequirement.SELF_CONTAINED

    def _infer_reply_requirement(
        self, msg_type: MessageType, ctx_req: ContextRequirement
    ) -> ReplyRequirement:
        """Infer what kind of reply is expected based on message type.

        Args:
            msg_type: The classified message type.
            ctx_req: The context requirement.

        Returns:
            ReplyRequirement enum value.
        """
        # If context is vague, we need clarification
        if ctx_req == ContextRequirement.VAGUE:
            return ReplyRequirement.CLARIFY

        # Map message types to reply requirements
        type_to_reply = {
            MessageType.QUESTION_YESNO: ReplyRequirement.YES_NO,
            MessageType.QUESTION_INFO: ReplyRequirement.INFO_RESPONSE,
            MessageType.QUESTION_OPEN: ReplyRequirement.INFO_RESPONSE,
            MessageType.REQUEST_ACTION: ReplyRequirement.ACTION_COMMIT,
            MessageType.STATEMENT: ReplyRequirement.QUICK_ACK,
            MessageType.ACKNOWLEDGMENT: ReplyRequirement.NO_REPLY,
            MessageType.REACTION: ReplyRequirement.NO_REPLY,
            MessageType.GREETING: ReplyRequirement.QUICK_ACK,
            MessageType.FAREWELL: ReplyRequirement.QUICK_ACK,
        }

        return type_to_reply.get(msg_type, ReplyRequirement.QUICK_ACK)

    def _infer_info_type(self, text: str, msg_type: MessageType) -> InfoType | None:
        """Infer the type of information being requested.

        Args:
            text: The message text.
            msg_type: The classified message type.

        Returns:
            InfoType for QUESTION_INFO, None otherwise.
        """
        if msg_type != MessageType.QUESTION_INFO:
            return None

        text_lower = text.lower()

        if TIME_PATTERNS.search(text_lower):
            return InfoType.TIME
        if LOCATION_PATTERNS.search(text_lower):
            return InfoType.LOCATION
        if PERSON_PATTERNS.search(text_lower):
            return InfoType.PERSON
        if REASON_PATTERNS.search(text_lower):
            return InfoType.REASON
        if METHOD_PATTERNS.search(text_lower):
            return InfoType.METHOD
        if QUANTITY_PATTERNS.search(text_lower):
            return InfoType.QUANTITY

        return InfoType.GENERAL

    def classify(self, text: str, embedder: Any | None = None) -> MessageClassification:
        """Classify a message into type, context requirement, and reply requirement.

        Uses a two-phase approach:
        1. Try rule-based classification first (fast, high precision)
        2. Fall back to embedding-based if rules don't match or are low confidence

        Args:
            text: The message text to classify.
            embedder: Optional embedder override (for per-request caching)

        Returns:
            MessageClassification with all classification results.
        """
        if not text or not text.strip():
            return MessageClassification(
                message_type=MessageType.STATEMENT,
                type_confidence=0.0,
                context_requirement=ContextRequirement.VAGUE,
                reply_requirement=ReplyRequirement.CLARIFY,
                classification_method="empty",
            )

        # Phase 1: Try rule-based classification
        msg_type, rule_conf, matched_rule = self._classify_by_rules(text)

        # Phase 2: If low confidence or no match, try embeddings
        classification_method = "rule"
        if msg_type is None or rule_conf < self.RULE_FALLBACK_THRESHOLD:
            emb_type, emb_conf = self._classify_by_embedding(text, embedder=embedder)
            if emb_type is not None and emb_conf > rule_conf:
                msg_type = emb_type
                rule_conf = emb_conf
                matched_rule = None
                classification_method = "embedding"

        # Default to STATEMENT if nothing matched
        if msg_type is None:
            msg_type = MessageType.STATEMENT
            rule_conf = 0.3
            classification_method = "default"

        # Classify context requirement
        ctx_req = self._classify_context(text)

        # Infer reply requirement
        reply_req = self._infer_reply_requirement(msg_type, ctx_req)

        # Infer info type for questions
        info_type = self._infer_info_type(text, msg_type)

        return MessageClassification(
            message_type=msg_type,
            type_confidence=rule_conf,
            context_requirement=ctx_req,
            reply_requirement=reply_req,
            info_type=info_type,
            matched_rule=matched_rule,
            classification_method=classification_method,
        )


# =============================================================================
# Singleton Access
# =============================================================================

_classifier: MessageClassifier | None = None
_classifier_lock = threading.Lock()


def get_message_classifier() -> MessageClassifier:
    """Get or create the singleton MessageClassifier instance.

    Returns:
        The shared MessageClassifier instance.
    """
    global _classifier

    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = MessageClassifier()

    return _classifier


def reset_message_classifier() -> None:
    """Reset the singleton MessageClassifier.

    Clears the singleton. A new instance will be created on
    the next get_message_classifier() call.
    """
    global _classifier

    with _classifier_lock:
        _classifier = None


def classify_message(text: str) -> MessageClassification:
    """Convenience function to classify a message.

    Args:
        text: The message text to classify.

    Returns:
        MessageClassification with all classification results.

    Example:
        >>> result = classify_message("Can you come to dinner?")
        >>> print(result.message_type)
        MessageType.QUESTION_YESNO
    """
    return get_message_classifier().classify(text)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MessageType",
    "ContextRequirement",
    "ReplyRequirement",
    "InfoType",
    # Data classes
    "MessageClassification",
    # Class
    "MessageClassifier",
    # Singleton functions
    "get_message_classifier",
    "reset_message_classifier",
    "classify_message",
]
