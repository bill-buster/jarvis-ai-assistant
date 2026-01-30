"""Intent Classification System for JARVIS v3.

Classifies user queries AND incoming messages into intents using semantic
similarity with sentence embeddings. Supports conversation intent detection
for better reply generation.

Ported from root/jarvis/intent.py with enhancements for message classification.
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
    """Types of intents for messages and queries."""

    # User query intents (what the user wants JARVIS to do)
    REPLY = "reply"
    SUMMARIZE = "summarize"
    SEARCH = "search"
    QUICK_REPLY = "quick_reply"
    GENERAL = "general"

    # Group chat intents
    GROUP_COORDINATION = "group_coordination"
    GROUP_RSVP = "group_rsvp"
    GROUP_CELEBRATION = "group_celebration"


class MessageIntent(Enum):
    """Types of incoming message intents (what the message is asking for)."""

    # Questions requiring answers
    YES_NO_QUESTION = "yes_no_question"
    OPEN_QUESTION = "open_question"
    CHOICE_QUESTION = "choice_question"

    # Statements and reactions
    STATEMENT = "statement"
    EMOTIONAL = "emotional"
    GREETING = "greeting"
    THANKS = "thanks"
    FAREWELL = "farewell"

    # Action-oriented
    REQUEST = "request"
    LOGISTICS = "logistics"
    SHARING = "sharing"

    # Information-seeking (needs search/context, not just a reply)
    INFORMATION_SEEKING = "information_seeking"


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType | MessageIntent
    confidence: float
    extracted_params: dict[str, str] = field(default_factory=dict)
    needs_response: bool = True
    needs_context: bool = False  # True if message requires specific info to answer
    is_specific_question: bool = False  # True if asking about specific facts/details


# Intent training examples for USER QUERIES
QUERY_INTENT_EXAMPLES: dict[IntentType, list[str]] = {
    IntentType.REPLY: [
        "help me reply to this",
        "help me reply to this message",
        "can you help me respond",
        "I need help replying",
        "what should I say back",
        "what should I respond with",
        "draft a response",
        "draft a reply",
        "write a reply for me",
        "how should I respond",
        "respond to John's message",
        "reply to Sarah's text",
        "help me reply to mom",
        "what do I text back",
    ],
    IntentType.SUMMARIZE: [
        "summarize my chat with John",
        "summarize my conversation",
        "summarize this conversation",
        "give me a summary",
        "recap this conversation",
        "what did we talk about",
        "what's the tldr",
        "catch me up on this conversation",
        "what did we discuss today",
    ],
    IntentType.SEARCH: [
        "find messages about the project",
        "search for dinner plans",
        "when did mom mention the party",
        "look for messages about vacation",
        "find the address he sent me",
        "where did we discuss the budget",
        "find messages from last Tuesday",
    ],
    IntentType.QUICK_REPLY: [
        "ok", "okay", "k", "kk", "got it", "sounds good",
        "yes", "yeah", "yep", "sure", "definitely",
        "no", "nope", "nah",
        "thanks", "thank you", "thx",
        "lol", "haha", "nice",
        "on my way", "omw", "see you", "bye",
    ],
    IntentType.GENERAL: [
        "what's the weather",
        "how are you",
        "hello", "hi", "hey",
        "what can you do",
        "help",
    ],
}

# Intent training examples for INCOMING MESSAGES
MESSAGE_INTENT_EXAMPLES: dict[MessageIntent, list[str]] = {
    MessageIntent.YES_NO_QUESTION: [
        "want to hang out?",
        "are you coming?",
        "did you get my message?",
        "can you help me?",
        "do you have time?",
        "is that okay?",
        "would you like to come?",
        "are you free tonight?",
        "can we meet tomorrow?",
        "did you finish?",
        "is everything okay?",
        "do you want to grab dinner?",
        "can you pick me up?",
        "were you there?",
        "have you seen this?",
    ],
    MessageIntent.OPEN_QUESTION: [
        "how are you?",
        "what time works for you?",
        "where should we meet?",
        "what do you think?",
        "how was your day?",
        "what's new?",
        "how did it go?",
        "what happened?",
        "what are you up to?",
        "when are you free?",
        "why did that happen?",
        "how's work going?",
        "what's the plan?",
        "how's it looking on your end?",
        "any progress on that?",
    ],
    MessageIntent.CHOICE_QUESTION: [
        "Italian or Mexican?",
        "Friday or Saturday?",
        "morning or afternoon?",
        "coffee or tea?",
        "your place or mine?",
        "now or later?",
        "option A or B?",
    ],
    MessageIntent.STATEMENT: [
        "I'll be there at 5",
        "the meeting got moved",
        "just finished work",
        "on my way now",
        "I got the tickets",
        "everything is ready",
        "that sounds miserable",
        "it's so expensive now",
        "yes i imagine",
    ],
    MessageIntent.EMOTIONAL: [
        "I'm so stressed",
        "I got the job!!!",
        "today was rough",
        "I'm so happy",
        "that's amazing!",
        "I can't believe it",
        "this is frustrating",
        "I'm really excited",
        "feeling down today",
        "congrats!",
        "so proud of you",
    ],
    MessageIntent.GREETING: [
        "hey!",
        "hi there",
        "what's up?",
        "hello!",
        "good morning",
        "how've you been?",
        "long time no talk",
        "hey stranger",
    ],
    MessageIntent.THANKS: [
        "thanks so much!",
        "thank you!",
        "appreciate it",
        "you're the best",
        "thanks for your help",
        "I really appreciate that",
    ],
    MessageIntent.FAREWELL: [
        "talk to you later!",
        "bye!",
        "gotta run",
        "see you soon",
        "have a good one",
        "take care",
        "ttyl",
    ],
    MessageIntent.REQUEST: [
        "can you send me that?",
        "please let me know",
        "could you check on that?",
        "send me the details",
        "let me know when you're free",
        "please work with cooper and make sure",
        "can you handle this?",
    ],
    MessageIntent.LOGISTICS: [
        "I'm running late",
        "just parked",
        "I'm here",
        "where are you?",
        "I'm at your place",
        "leaving now",
        "be there in 10",
    ],
    MessageIntent.SHARING: [
        "check out this restaurant",
        "look at this meme",
        "I got you something",
        "here's the link",
        "sharing this with you",
        "thought you'd like this",
        "👀👀👀",
    ],
    MessageIntent.INFORMATION_SEEKING: [
        # Questions that require specific knowledge/search
        "where did we talk about that?",
        "when did you mention the meeting?",
        "what was the address again?",
        "do you remember when we discussed?",
        "what did you say about the project?",
        "when did we plan to meet?",
        "what was the name of that place?",
        "can you remind me what we agreed on?",
        "what time did we say?",
        "where did I put that?",
        "what was the link you sent?",
        "when is the deadline again?",
        "what was his phone number?",
    ],
}


# Patterns that indicate specific information is being requested
SPECIFIC_INFO_PATTERNS = [
    "what was", "what is", "what's the",
    "when did", "when is", "when was",
    "where did", "where is", "where was",
    "who is", "who was", "who did",
    "how much", "how many",
    "do you remember", "can you remind",
    "what did you say", "what did we",
    "the address", "the time", "the date", "the name", "the number",
    "again", "remind me",
]


class IntentClassifier:
    """Classifies queries and messages into intents using semantic similarity."""

    CONFIDENCE_THRESHOLD = 0.5
    QUICK_REPLY_THRESHOLD = 0.8

    def __init__(self) -> None:
        self._query_centroids: dict[IntentType, np.ndarray] | None = None
        self._message_centroids: dict[MessageIntent, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._model = None

    def _get_sentence_model(self) -> Any:
        """Get the sentence transformer model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._model
        except ImportError:
            logger.warning("sentence-transformers not installed")
            raise

    def _ensure_embeddings_computed(self) -> None:
        """Compute and cache embeddings for all intent examples."""
        if self._query_centroids is not None and self._message_centroids is not None:
            return

        with self._lock:
            if self._query_centroids is not None and self._message_centroids is not None:
                return

            model = self._get_sentence_model()

            # Compute query intent centroids
            query_centroids: dict[IntentType, np.ndarray] = {}
            for intent_type, examples in QUERY_INTENT_EXAMPLES.items():
                embeddings = model.encode(examples, convert_to_numpy=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                query_centroids[intent_type] = centroid

            # Compute message intent centroids
            message_centroids: dict[MessageIntent, np.ndarray] = {}
            for intent_type, examples in MESSAGE_INTENT_EXAMPLES.items():
                embeddings = model.encode(examples, convert_to_numpy=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                message_centroids[intent_type] = centroid

            self._query_centroids = query_centroids
            self._message_centroids = message_centroids

            logger.info(
                "Computed intent embeddings: %d query intents, %d message intents",
                len(query_centroids),
                len(message_centroids),
            )

    def classify_query(self, query: str) -> IntentResult:
        """Classify a user query (what they want JARVIS to do)."""
        if not query or not query.strip():
            return IntentResult(intent=IntentType.GENERAL, confidence=0.0)

        try:
            self._ensure_embeddings_computed()
        except Exception:
            return IntentResult(intent=IntentType.GENERAL, confidence=0.0)

        if self._query_centroids is None:
            return IntentResult(intent=IntentType.GENERAL, confidence=0.0)

        model = self._get_sentence_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        similarities = {}
        for intent_type, centroid in self._query_centroids.items():
            similarity = float(np.dot(query_norm, centroid))
            similarities[intent_type] = similarity

        best_intent = max(similarities, key=lambda k: similarities[k])
        best_confidence = similarities[best_intent]

        if best_confidence < self.CONFIDENCE_THRESHOLD:
            best_intent = IntentType.GENERAL
            best_confidence = similarities[IntentType.GENERAL]

        extracted_params = self._extract_params(query)

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            extracted_params=extracted_params,
        )

    def classify_message(self, message: str) -> IntentResult:
        """Classify an incoming message to determine appropriate response type."""
        if not message or not message.strip():
            return IntentResult(
                intent=MessageIntent.STATEMENT,
                confidence=0.0,
                needs_response=False,
            )

        # Quick heuristics for obvious patterns
        message_stripped = message.strip()

        # Question mark = question (most reliable signal)
        if message_stripped.endswith("?"):
            message_lower = message_stripped.lower()

            # Check if this is asking for specific info (needs context)
            needs_context = any(p in message_lower for p in SPECIFIC_INFO_PATTERNS)
            is_specific = any(p in message_lower for p in [
                "what was", "when did", "where did", "who is",
                "remember", "remind", "again",
            ])

            # Check if it's a choice question
            if " or " in message_lower:
                return IntentResult(
                    intent=MessageIntent.CHOICE_QUESTION,
                    confidence=0.9,
                    needs_response=True,
                    needs_context=needs_context,
                    is_specific_question=is_specific,
                )
            # Check for yes/no patterns
            yes_no_starters = ("do ", "does ", "did ", "is ", "are ", "was ", "were ",
                               "can ", "could ", "would ", "will ", "have ", "has ",
                               "should ", "shall ")
            if message_lower.startswith(yes_no_starters):
                return IntentResult(
                    intent=MessageIntent.YES_NO_QUESTION,
                    confidence=0.9,
                    needs_response=True,
                    needs_context=needs_context,
                    is_specific_question=is_specific,
                )
            # Default to open question
            return IntentResult(
                intent=MessageIntent.OPEN_QUESTION,
                confidence=0.85,
                needs_response=True,
                needs_context=needs_context,
                is_specific_question=is_specific,
            )

        try:
            self._ensure_embeddings_computed()
        except Exception:
            return IntentResult(
                intent=MessageIntent.STATEMENT,
                confidence=0.0,
                needs_response=True,
            )

        if self._message_centroids is None:
            return IntentResult(
                intent=MessageIntent.STATEMENT,
                confidence=0.0,
                needs_response=True,
            )

        model = self._get_sentence_model()
        msg_embedding = model.encode([message], convert_to_numpy=True)[0]
        msg_norm = msg_embedding / np.linalg.norm(msg_embedding)

        similarities = {}
        for intent_type, centroid in self._message_centroids.items():
            similarity = float(np.dot(msg_norm, centroid))
            similarities[intent_type] = similarity

        best_intent = max(similarities, key=lambda k: similarities[k])
        best_confidence = similarities[best_intent]

        # Determine if response needed
        needs_response = best_intent not in (
            MessageIntent.FAREWELL,
            MessageIntent.THANKS,
        )

        # Check if this message is seeking specific information
        message_lower = message.lower()
        needs_context = False
        is_specific_question = False

        # Check for specific info patterns
        if any(pattern in message_lower for pattern in SPECIFIC_INFO_PATTERNS):
            is_specific_question = True
            # If it's asking about past conversations/details, it needs context
            if any(p in message_lower for p in [
                "did we", "did you", "what did", "when did", "where did",
                "remember", "remind", "again", "what was", "the address",
                "the time", "the date", "the link", "the number",
            ]):
                needs_context = True

        # INFORMATION_SEEKING intent always needs context
        if best_intent == MessageIntent.INFORMATION_SEEKING:
            needs_context = True
            is_specific_question = True

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            needs_response=needs_response,
            needs_context=needs_context,
            is_specific_question=is_specific_question,
        )

    def _extract_params(self, query: str) -> dict[str, str]:
        """Extract parameters like person name, time range from query."""
        params: dict[str, str] = {}

        # Person name patterns
        person_patterns = [
            r"(?:with|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"([A-Z][a-z]+)(?:'s)?\s+(?:message|chat|conversation|text)",
            r"(?:reply|respond)\s+(?:to\s+)?([A-Z][a-z]+)",
            r"\b(mom|dad|mother|father|sis|bro)\b",
        ]
        for pattern in person_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["person_name"] = match.group(1).capitalize()
                break

        # Time range patterns
        time_patterns = [
            r"\b(today|yesterday|tomorrow)\b",
            r"\b(last\s+week|this\s+week|last\s+month)\b",
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["time_range"] = match.group(1).lower()
                break

        return params

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        with self._lock:
            self._query_centroids = None
            self._message_centroids = None


# Module-level singleton
_classifier: IntentClassifier | None = None
_classifier_lock = threading.Lock()


def get_intent_classifier() -> IntentClassifier:
    """Get the singleton IntentClassifier instance."""
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = IntentClassifier()
    return _classifier


def classify_incoming_message(message: str) -> IntentResult:
    """Convenience function to classify an incoming message."""
    return get_intent_classifier().classify_message(message)
