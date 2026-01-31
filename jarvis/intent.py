"""Intent Classification System for JARVIS.

Classifies user queries into intents using semantic similarity with
sentence embeddings. Supports REPLY, SUMMARIZE, SEARCH, QUICK_REPLY,
and GENERAL intents.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents supported by JARVIS."""

    REPLY = "reply"
    SUMMARIZE = "summarize"
    SEARCH = "search"
    QUICK_REPLY = "quick_reply"
    GENERAL = "general"
    # Group chat specific intents
    GROUP_COORDINATION = "group_coordination"  # Event planning, logistics, polls
    GROUP_RSVP = "group_rsvp"  # RSVPs and attendance responses
    GROUP_CELEBRATION = "group_celebration"  # Birthdays, congratulations, holidays


@dataclass
class IntentResult:
    """Result of intent classification.

    Attributes:
        intent: The classified intent type
        confidence: Confidence score from 0.0 to 1.0
        extracted_params: Extracted parameters like person_name, time_range, search_query
    """

    intent: IntentType
    confidence: float
    extracted_params: dict[str, str] = field(default_factory=dict)


# Intent training examples - at least 15 per intent for good semantic coverage
# Based on research: diverse phrasing helps the model generalize better
INTENT_EXAMPLES: dict[IntentType, list[str]] = {
    IntentType.REPLY: [
        # Direct help requests
        "help me reply to this",
        "help me reply to this message",
        "can you help me respond",
        "I need help replying",
        "help me write a response",
        # What to say variations
        "what should I say back",
        "what should I respond with",
        "what would be a good response",
        "how should I answer this",
        "what can I say to this",
        "what's a good reply",
        "what do I say back",
        # Draft/write variations
        "draft a response",
        "draft a reply",
        "draft something to say",
        "write a reply for me",
        "write a response",
        "compose a reply",
        "create a response",
        # Respond variations
        "how should I respond",
        "how do I respond to this",
        "I don't know how to respond",
        "help me answer this message",
        "help me answer them",
        # Reply with person context
        "respond to John's message",
        "reply to Sarah's text",
        "help me reply to mom",
        "draft something to say back to him",
        "what should I tell her",
        "how do I respond to my boss",
        # Informal variations
        "can you reply to this for me",
        "reply to this",
        "respond to this",
        "answer this for me",
        "what do I text back",
    ],
    IntentType.SUMMARIZE: [
        # Direct summarize requests
        "summarize my chat with John",
        "summarize my conversation",
        "summarize this conversation",
        "summarize our messages",
        "summarize the chat",
        "summarize messages from mom",
        "give me a summary",
        "give me a summary of the chat",
        # Recap variations
        "recap this conversation",
        "recap what we talked about",
        "recap my texts with Sarah",
        "give me a recap",
        "can you recap our chat",
        # What did we discuss variations
        "what did Sarah and I talk about",
        "what have we discussed",
        "what did we talk about",
        "what did I miss",
        "what's been said",
        "what happened in this chat",
        # TLDR and catch up variations
        "what's the tldr of this chat",
        "tldr of my messages",
        "give me the tldr",
        "catch me up on this conversation",
        "catch me up",
        "fill me in on the conversation",
        "bring me up to speed",
        # Time-based summary requests
        "what did we talk about last week",
        "summarize yesterday's messages",
        "what did we discuss today",
        "recap our conversation from yesterday",
        "summary of this week's messages",
        "what have we been texting about lately",
        # Informal variations
        "what's going on in this chat",
        "sum up the convo",
        "brief me on this chat",
    ],
    IntentType.SEARCH: [
        # Find messages variations
        "find messages about the project",
        "find messages about dinner",
        "find messages mentioning vacation",
        "find where we talked about the meeting",
        "find the message where he mentioned",
        "find texts about work",
        # Search variations
        "search for dinner plans",
        "search for the link Sarah shared",
        "search messages for that address",
        "search my texts for the time",
        "search for meeting details",
        # When did variations
        "when did mom mention the party",
        "when did John say he'd be free",
        "when did she send that",
        "when was the last time we talked about this",
        "when did he mention the price",
        # Look for variations
        "look for messages about vacation",
        "look for that link he sent",
        "look up what she said about the date",
        "look for when we planned to meet",
        # Find specific content
        "find the address he sent me",
        "find the phone number in our chat",
        "find that restaurant recommendation",
        "find the link to the document",
        "find when we agreed to meet",
        # Where did variations
        "where did we discuss the budget",
        "where did he mention the party",
        "where was the meeting time decided",
        # Time-based search
        "find messages from last Tuesday",
        "search texts from yesterday",
        "look for messages from this week",
        "find what was sent on Monday",
        # Person-specific search
        "find where John mentioned the meeting",
        "search Sarah's messages for the link",
        "look for mom's message about dinner",
    ],
    IntentType.QUICK_REPLY: [
        # Simple acknowledgments
        "ok",
        "okay",
        "k",
        "kk",
        "got it",
        "sounds good",
        "sounds great",
        "perfect",
        "great",
        "cool",
        "alright",
        # Affirmatives
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "definitely",
        "absolutely",
        # Negatives
        "no",
        "nope",
        "nah",
        # Thanks variations
        "thanks",
        "thank you",
        "thx",
        "ty",
        # No problem variations
        "np",
        "no problem",
        "no worries",
        "all good",
        # Laughter/reactions
        "lol",
        "haha",
        "hahaha",
        "lmao",
        "nice",
        "awesome",
        # Short phrases
        "on my way",
        "omw",
        "be there soon",
        "see you",
        "bye",
        "later",
        "ttyl",
    ],
    IntentType.GENERAL: [
        # Weather and time
        "what's the weather",
        "what's the weather like",
        "what time is it",
        "what day is it",
        # Small talk
        "how are you",
        "hello",
        "hi",
        "hey",
        "tell me a joke",
        "tell me something interesting",
        # Random questions
        "what is the meaning of life",
        "what's the capital of France",
        "who is the president",
        "how do I cook pasta",
        "what's 2 plus 2",
        "calculate this for me",
        # Help and info
        "what can you do",
        "help",
        "what are you",
        "who are you",
    ],
    IntentType.GROUP_COORDINATION: [
        # Event planning
        "when works for everyone",
        "what time works for the group",
        "when is everyone free",
        "let's figure out a time",
        "when can everyone make it",
        "what day works for all",
        "let's pick a date",
        "scheduling for the group",
        # Logistics
        "who's bringing what",
        "what should I bring",
        "I'll handle the reservation",
        "I can make the reservation",
        "where are we meeting",
        "where should we go",
        "who's driving",
        "let's carpool",
        "anyone need a ride",
        "let's split the bill",
        "how should we split it",
        # Polls
        "let's do a poll",
        "let's vote on it",
        "what does everyone think",
        "can we get everyone's input",
        "I vote for option A",
        "either works for me",
        "no preference",
        "both options work",
        # Group updates
        "update for everyone",
        "fyi for the group",
        "heads up everyone",
        "reminder for everyone",
        "sharing with the group",
    ],
    IntentType.GROUP_RSVP: [
        # Yes responses
        "count me in",
        "I'm in",
        "I'll be there",
        "yes I'm coming",
        "definitely coming",
        "sign me up",
        "add me to the list",
        "I'm down",
        # Plus one
        "I'll be there +1",
        "count me plus one",
        "I'm bringing someone",
        "can I bring a friend",
        "plus one for me",
        # No responses
        "can't make it",
        "I won't be able to come",
        "count me out",
        "I have to skip this one",
        "sorry I can't make it",
        "unfortunately I can't come",
        # Maybe responses
        "I might be able to come",
        "I'll try to make it",
        "tentative yes",
        "put me down as a maybe",
        "I'll let you know",
        "not sure yet",
        # Attendance queries
        "who's coming",
        "how many people so far",
        "what's the headcount",
        "who's confirmed",
        "who all is going",
    ],
    IntentType.GROUP_CELEBRATION: [
        # Birthdays
        "happy birthday",
        "happy bday",
        "hbd",
        "hope you have a great birthday",
        "wishing you a happy birthday",
        "birthday wishes",
        # Congratulations
        "congrats everyone",
        "congratulations to all",
        "way to go team",
        "we did it",
        "great job everyone",
        "congrats",
        "congratulations",
        "so proud of you",
        "well done",
        "amazing job",
        "you did it",
        # Holidays
        "happy holidays",
        "happy new year",
        "merry christmas",
        "happy thanksgiving",
        "enjoy the holidays",
        # Group appreciation
        "thanks everyone",
        "thank you all",
        "appreciate everyone",
        "grateful for this group",
        "you all are the best",
        "love this group",
    ],
}


# Regex patterns for parameter extraction
PERSON_NAME_PATTERNS = [
    # "with/from/to Name" pattern
    r"(?:with|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    # "Name's message/chat" pattern
    r"([A-Z][a-z]+)(?:'s)?\s+(?:message|chat|conversation|text|texts)",
    # "reply to Name" pattern
    r"(?:reply|respond|answer)\s+(?:to\s+)?([A-Z][a-z]+)",
    # "tell Name" or "message Name" pattern
    r"(?:tell|message|text)\s+([A-Z][a-z]+)",
    # Common names mentioned (mom, dad, etc.)
    r"\b(mom|dad|mother|father|sis|bro|brother|sister)\b",
]

TIME_RANGE_PATTERNS = [
    # Relative time
    r"\b(today|yesterday|tomorrow)\b",
    r"\b(last\s+week|this\s+week|last\s+month|this\s+month)\b",
    r"\b(last\s+\d+\s+(?:days?|weeks?|months?))\b",
    # Days of week
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    # Date patterns
    r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
    r"\b(\d{1,2}-\d{1,2}(?:-\d{2,4})?)\b",
]


class IntentClassifier:
    """Classifies user queries into intents using semantic similarity.

    Uses sentence embeddings to compute semantic similarity between user queries
    and intent examples. Implements lazy loading and thread-safe initialization.

    Attributes:
        CONFIDENCE_THRESHOLD: Minimum confidence to return a specific intent (0.6)
        QUICK_REPLY_THRESHOLD: Higher threshold for quick reply intent (0.8)
    """

    CONFIDENCE_THRESHOLD = 0.6
    QUICK_REPLY_THRESHOLD = 0.8

    def __init__(self) -> None:
        """Initialize the intent classifier with lazy-loaded embeddings."""
        self._intent_embeddings: dict[IntentType, np.ndarray] | None = None
        self._intent_centroids: dict[IntentType, np.ndarray] | None = None
        self._lock = threading.Lock()

    def _get_embedder(self) -> Any:
        """Get the embedder for intent classification.

        Uses the unified embedding adapter which tries MLX first with
        SentenceTransformer fallback.

        Returns:
            The UnifiedEmbedder instance

        Raises:
            RuntimeError: If no embedding backend is available
        """
        from jarvis.embedding_adapter import get_embedder

        try:
            embedder = get_embedder()
            if not embedder.is_available():
                raise RuntimeError("No embedding backend available")
            return embedder
        except Exception as e:
            logger.warning("Failed to get embedder for intent classification: %s", e)
            raise

    def _ensure_embeddings_computed(self) -> None:
        """Compute and cache embeddings for all intent examples.

        Uses double-check locking for thread-safe lazy initialization.
        Computes both individual embeddings and centroids (mean embeddings)
        for each intent class.
        """
        # Fast path: embeddings already computed
        if self._intent_centroids is not None:
            return

        # Slow path: acquire lock and double-check
        with self._lock:
            # Double-check after acquiring lock
            if self._intent_centroids is not None:
                return

            embedder = self._get_embedder()

            # Compute embeddings for each intent
            intent_embeddings: dict[IntentType, np.ndarray] = {}
            intent_centroids: dict[IntentType, np.ndarray] = {}

            for intent_type, examples in INTENT_EXAMPLES.items():
                # Compute embeddings in batch for efficiency (normalized)
                embeddings = embedder.encode(examples, normalize=True)
                intent_embeddings[intent_type] = embeddings

                # Compute centroid (mean embedding) for this intent
                # Centroids are more robust than individual example matching
                centroid = np.mean(embeddings, axis=0)
                # Normalize the centroid for cosine similarity
                centroid = centroid / np.linalg.norm(centroid)
                intent_centroids[intent_type] = centroid

            # Assign atomically
            self._intent_embeddings = intent_embeddings
            self._intent_centroids = intent_centroids

            total_examples = sum(len(ex) for ex in INTENT_EXAMPLES.values())
            logger.info(
                "Computed intent embeddings for %d intents (%d examples)",
                len(INTENT_EXAMPLES),
                total_examples,
            )

    def classify(self, query: str, embedder: Any | None = None) -> IntentResult:
        """Classify a user query into an intent.

        Uses a hybrid approach:
        1. First checks against centroids for each intent
        2. Falls back to GENERAL if confidence is below threshold

        Args:
            query: The user query to classify
            embedder: Optional embedder override (for per-request caching)

        Returns:
            IntentResult with intent type, confidence, and extracted parameters
        """
        # Handle empty or whitespace-only input
        if not query or not query.strip():
            return IntentResult(
                intent=IntentType.GENERAL,
                confidence=0.0,
                extracted_params={},
            )

        # Truncate very long input to prevent memory issues
        max_length = 1000
        if len(query) > max_length:
            query = query[:max_length]

        try:
            self._ensure_embeddings_computed()
        except Exception:
            # If model fails to load, return GENERAL with 0.0 confidence
            logger.warning("Intent classification unavailable, returning GENERAL")
            return IntentResult(
                intent=IntentType.GENERAL,
                confidence=0.0,
                extracted_params={},
            )

        # Type guard
        if self._intent_centroids is None:
            return IntentResult(
                intent=IntentType.GENERAL,
                confidence=0.0,
                extracted_params={},
            )

        try:
            query_embedder = embedder or self._get_embedder()
            query_embedding = query_embedder.encode([query], normalize=True)[0]

            # Embedding is already normalized, but compute norm for safety
            query_norm = query_embedding / np.linalg.norm(query_embedding)

            # Compute similarity to each intent centroid
            similarities: dict[IntentType, float] = {}
            for intent_type, centroid in self._intent_centroids.items():
                # Cosine similarity with normalized vectors is just dot product
                similarity = float(np.dot(query_norm, centroid))
                similarities[intent_type] = similarity

            # Find best matching intent
            best_intent = max(similarities, key=lambda k: similarities[k])
            best_confidence = similarities[best_intent]

            # Apply thresholds
            if best_intent == IntentType.QUICK_REPLY:
                # Higher threshold for quick replies to avoid false positives
                if best_confidence < self.QUICK_REPLY_THRESHOLD:
                    # Check if any other intent meets the standard threshold
                    for intent, conf in sorted(
                        similarities.items(), key=lambda x: x[1], reverse=True
                    ):
                        if intent != IntentType.QUICK_REPLY:
                            if conf >= self.CONFIDENCE_THRESHOLD:
                                best_intent = intent
                                best_confidence = conf
                                break
                    else:
                        best_intent = IntentType.GENERAL
                        best_confidence = similarities[IntentType.GENERAL]
            elif best_confidence < self.CONFIDENCE_THRESHOLD:
                # Below threshold, return GENERAL
                best_intent = IntentType.GENERAL
                best_confidence = similarities[IntentType.GENERAL]

            # Extract parameters based on intent
            extracted_params = self._extract_params(query, best_intent)

            logger.debug(
                "Classified query as %s (confidence: %.3f)",
                best_intent.value,
                best_confidence,
            )

            return IntentResult(
                intent=best_intent,
                confidence=best_confidence,
                extracted_params=extracted_params,
            )

        except Exception:
            logger.exception("Error during intent classification")
            return IntentResult(
                intent=IntentType.GENERAL,
                confidence=0.0,
                extracted_params={},
            )

    def _extract_params(self, query: str, intent: IntentType) -> dict[str, str]:
        """Extract relevant parameters from the query based on intent.

        Args:
            query: The user query
            intent: The classified intent type

        Returns:
            Dictionary of extracted parameters
        """
        params: dict[str, str] = {}

        # Extract person name for relevant intents
        if intent in (IntentType.REPLY, IntentType.SUMMARIZE, IntentType.SEARCH):
            person_name = self._extract_person_name(query)
            if person_name:
                params["person_name"] = person_name

        # Extract time range for relevant intents
        if intent in (IntentType.SUMMARIZE, IntentType.SEARCH, IntentType.GROUP_COORDINATION):
            time_range = self._extract_time_range(query)
            if time_range:
                params["time_range"] = time_range

        # Extract search query for SEARCH intent
        if intent == IntentType.SEARCH:
            search_query = self._extract_search_query(query)
            if search_query:
                params["search_query"] = search_query

        # Extract group-specific parameters
        if intent in (IntentType.GROUP_COORDINATION, IntentType.GROUP_RSVP):
            rsvp_response = self._extract_rsvp_response(query)
            if rsvp_response:
                params["rsvp_response"] = rsvp_response

            poll_choice = self._extract_poll_choice(query)
            if poll_choice:
                params["poll_choice"] = poll_choice

        return params

    def _extract_rsvp_response(self, query: str) -> str | None:
        """Extract RSVP response type from query.

        Args:
            query: The user query

        Returns:
            'yes', 'no', 'maybe', or None if not detected
        """
        query_lower = query.lower()

        # Yes patterns
        yes_patterns = [
            r"\b(count me in|i'm in|i'll be there|yes|definitely|sign me up|i'm down)\b",
            r"\b(coming|attending|going)\b",
        ]
        for pattern in yes_patterns:
            if re.search(pattern, query_lower):
                return "yes"

        # No patterns
        no_patterns = [
            r"\b(can't make it|count me out|skip|won't be able|can't come|cannot)\b",
            r"\b(not coming|not going|not attending)\b",
        ]
        for pattern in no_patterns:
            if re.search(pattern, query_lower):
                return "no"

        # Maybe patterns
        maybe_patterns = [
            r"\b(maybe|might|tentative|not sure|let you know|try to make it)\b",
        ]
        for pattern in maybe_patterns:
            if re.search(pattern, query_lower):
                return "maybe"

        return None

    def _extract_poll_choice(self, query: str) -> str | None:
        """Extract poll choice from query.

        Args:
            query: The user query

        Returns:
            The choice (e.g., 'A', 'B', '1', '2') or None if not detected
        """
        query_lower = query.lower()

        # Option letter patterns
        letter_match = re.search(r"\boption\s+([a-z])\b", query_lower)
        if letter_match:
            return letter_match.group(1).upper()

        # Direct letter vote patterns
        vote_pattern = r"\b(?:vote(?:s)?\s+(?:for\s+)?|i\s+(?:vote|prefer|choose)\s+)([a-z])\b"
        vote_match = re.search(vote_pattern, query_lower)
        if vote_match:
            return vote_match.group(1).upper()

        # Number patterns
        number_match = re.search(r"\boption\s+(\d+)\b", query_lower)
        if number_match:
            return number_match.group(1)

        # "Either" or "both" indicates no specific choice
        if re.search(r"\b(either|both|any|no preference)\b", query_lower):
            return "any"

        return None

    def _extract_person_name(self, query: str) -> str | None:
        """Extract person name from query using regex patterns.

        Args:
            query: The user query

        Returns:
            Extracted person name or None
        """
        for pattern in PERSON_NAME_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Capitalize properly for common family terms
                if name.lower() in ("mom", "dad", "mother", "father", "sis", "bro"):
                    return name.capitalize()
                return name
        return None

    def _extract_time_range(self, query: str) -> str | None:
        """Extract time range from query using regex patterns.

        Args:
            query: The user query

        Returns:
            Extracted time range or None
        """
        for pattern in TIME_RANGE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_search_query(self, query: str) -> str | None:
        """Extract the search topic from a SEARCH intent query.

        Extracts what the user is looking for by finding content after
        keywords like 'about', 'for', 'mentioning', etc.

        Args:
            query: The user query

        Returns:
            Extracted search query or None
        """
        # Patterns for extracting search topics
        search_patterns = [
            r"(?:about|for|mentioning|regarding)\s+(?:the\s+)?(.+?)(?:\s+from|\s+in|\s+with|$)",
            r"find\s+(?:the\s+)?(.+?)(?:\s+(?:he|she|they)\s+sent|\s+in|\s+from|$)",
            r"search\s+(?:for\s+)?(?:the\s+)?(.+?)(?:\s+from|\s+in|$)",
            r"look\s+(?:for\s+)?(?:the\s+)?(.+?)(?:\s+from|\s+in|$)",
            r"where\s+(?:did\s+)?(?:we\s+)?(?:discuss|talk\s+about|mention)\s+(?:the\s+)?(.+?)$",
        ]

        for pattern in search_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                # Clean up the result
                # Remove trailing punctuation
                result = re.sub(r"[?.!]+$", "", result)
                # Remove common filler words at the end
                result = re.sub(r"\s+(please|can you|could you)$", "", result, flags=re.IGNORECASE)
                if result:
                    return result

        return None

    def clear_cache(self) -> None:
        """Clear cached embeddings.

        Call this to force recomputation of embeddings, for example
        after the sentence model has been unloaded and reloaded.
        """
        with self._lock:
            self._intent_embeddings = None
            self._intent_centroids = None
            logger.debug("Intent classifier cache cleared")


# Module-level singleton instance
_classifier: IntentClassifier | None = None
_classifier_lock = threading.Lock()


def get_intent_classifier() -> IntentClassifier:
    """Get the singleton IntentClassifier instance.

    Returns:
        The shared IntentClassifier instance
    """
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = IntentClassifier()
    return _classifier


def reset_intent_classifier() -> None:
    """Reset the singleton IntentClassifier instance.

    Call this to create a fresh classifier on next access.
    """
    global _classifier
    with _classifier_lock:
        _classifier = None
    logger.debug("Intent classifier singleton reset")
