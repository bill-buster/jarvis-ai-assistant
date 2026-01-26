"""Thread-aware conversation analysis for iMessage.

Provides thread detection, topic classification, state tracking,
and user role identification for improved reply generation.
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
    from contracts.imessage import Message

logger = logging.getLogger(__name__)


class ThreadTopic(Enum):
    """Types of conversation thread topics."""

    PLANNING = "planning"  # Making plans, scheduling
    LOGISTICS = "logistics"  # Coordinating details, times, locations
    CATCHING_UP = "catching_up"  # General conversation, catching up
    EMOTIONAL_SUPPORT = "emotional_support"  # Support, sympathy, encouragement
    QUICK_EXCHANGE = "quick_exchange"  # Brief back-and-forth, acknowledgments
    INFORMATION = "information"  # Sharing/requesting information
    DECISION_MAKING = "decision_making"  # Making choices together
    CELEBRATION = "celebration"  # Congratulations, good news
    UNKNOWN = "unknown"


class ThreadState(Enum):
    """Current state of a conversation thread."""

    OPEN_QUESTION = "open_question"  # Waiting for an answer
    AWAITING_RESPONSE = "awaiting_response"  # General expectation of reply
    IN_DISCUSSION = "in_discussion"  # Active back-and-forth
    CONCLUDED = "concluded"  # Conversation naturally ended
    STALE = "stale"  # No recent activity


class UserRole(Enum):
    """User's role in the conversation thread."""

    INITIATOR = "initiator"  # Started the thread/topic
    RESPONDER = "responder"  # Responding to other's initiation
    PARTICIPANT = "participant"  # Equal participant in ongoing exchange


# Topic classification patterns with example phrases
TOPIC_PATTERNS: dict[ThreadTopic, list[str]] = {
    ThreadTopic.PLANNING: [
        "let's plan",
        "when should we",
        "what day works",
        "are you free",
        "want to meet",
        "let's get together",
        "should we do",
        "thinking about",
        "how about we",
        "planning to",
        "want to grab",
        "let's do",
        "wanna hang",
        "wanna go",
        "want to come",
        "are you coming",
        "can you make it",
        "will you be there",
        "planning",
        "schedule",
    ],
    ThreadTopic.LOGISTICS: [
        "what time",
        "where should",
        "what address",
        "how do I get",
        "meet at",
        "pick you up",
        "be there by",
        "running late",
        "on my way",
        "almost there",
        "just left",
        "eta",
        "where are you",
        "directions",
        "parking",
        "which entrance",
        "I'm here",
        "just arrived",
        "waiting",
    ],
    ThreadTopic.EMOTIONAL_SUPPORT: [
        "I'm sorry",
        "that's tough",
        "I understand",
        "here for you",
        "feel better",
        "thinking of you",
        "so sorry to hear",
        "hang in there",
        "I'm here",
        "let me know if you need",
        "sending love",
        "you've got this",
        "proud of you",
        "it'll be okay",
        "don't worry",
        "feeling down",
        "bad day",
        "stressed",
        "anxious",
        "worried about",
        "upset",
        "sad",
        "frustrated",
    ],
    ThreadTopic.CATCHING_UP: [
        "how are you",
        "what's new",
        "long time",
        "miss you",
        "how have you been",
        "what's up",
        "how's it going",
        "catch up",
        "tell me about",
        "what have you been up to",
        "how's life",
        "how was your",
        "what did you do",
        "anything new",
        "how's everything",
        "good to hear from",
    ],
    ThreadTopic.QUICK_EXCHANGE: [
        "ok",
        "okay",
        "sounds good",
        "got it",
        "thanks",
        "thank you",
        "cool",
        "great",
        "perfect",
        "nice",
        "lol",
        "haha",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "np",
        "kk",
        "ttyl",
        "bye",
    ],
    ThreadTopic.INFORMATION: [
        "do you know",
        "can you tell me",
        "what is",
        "where is",
        "who is",
        "how do",
        "what's the",
        "I need",
        "looking for",
        "have you heard",
        "did you see",
        "fyi",
        "just so you know",
        "heads up",
        "reminder",
        "don't forget",
        "btw",
        "wanted to let you know",
    ],
    ThreadTopic.DECISION_MAKING: [
        "what do you think",
        "should I",
        "which one",
        "your opinion",
        "help me decide",
        "can't decide",
        "what would you",
        "do you prefer",
        "better option",
        "pros and cons",
        "advice",
        "recommend",
        "suggest",
        "your thoughts",
        "vote",
        "choose",
    ],
    ThreadTopic.CELEBRATION: [
        "congratulations",
        "congrats",
        "so happy for",
        "amazing news",
        "great news",
        "celebrate",
        "proud of you",
        "you did it",
        "well done",
        "awesome",
        "exciting",
        "happy birthday",
        "cheers",
        "woohoo",
        "yay",
    ],
}

# Patterns indicating questions or expectation of response
QUESTION_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"^(what|where|when|who|why|how|which|can|could|would|will|do|does|did|is|are|was|were)\b",
    r"\b(let me know|lmk|thoughts\??|wdyt|wyt)\b",
]

# Patterns indicating thread conclusion
CONCLUSION_PATTERNS = [
    r"\b(sounds good|perfect|great|thanks|bye|later|ttyl|see you|talk soon|cya)\b",
    r"\b(got it|understood|will do|on it)\b",
    r"^(ok|okay|k|kk)$",
]


@dataclass
class ThreadContext:
    """Analyzed context for a conversation thread.

    Attributes:
        messages: The messages in the thread
        topic: Detected thread topic
        state: Current thread state
        user_role: User's role in the thread
        confidence: Confidence score for topic classification (0.0-1.0)
        relevant_messages: Subset of messages most relevant to current context
        action_items: Any detected action items or commitments
        participants_count: Number of participants (1 = DM, >1 = group)
    """

    messages: list[Message]
    topic: ThreadTopic
    state: ThreadState
    user_role: UserRole
    confidence: float
    relevant_messages: list[Message] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    participants_count: int = 1


@dataclass
class ThreadedReplyConfig:
    """Configuration for thread-aware reply generation.

    Attributes:
        max_response_length: Suggested max response length based on thread type
        response_style: Suggested response style (concise/detailed/empathetic)
        include_action_items: Whether to include action items in response
        suggest_follow_up: Whether to suggest follow-up questions
    """

    max_response_length: int
    response_style: str
    include_action_items: bool
    suggest_follow_up: bool


# Response configuration based on thread topic
TOPIC_RESPONSE_CONFIG: dict[ThreadTopic, ThreadedReplyConfig] = {
    ThreadTopic.LOGISTICS: ThreadedReplyConfig(
        max_response_length=50,
        response_style="concise",
        include_action_items=True,
        suggest_follow_up=False,
    ),
    ThreadTopic.QUICK_EXCHANGE: ThreadedReplyConfig(
        max_response_length=30,
        response_style="brief",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.EMOTIONAL_SUPPORT: ThreadedReplyConfig(
        max_response_length=150,
        response_style="empathetic",
        include_action_items=False,
        suggest_follow_up=True,
    ),
    ThreadTopic.PLANNING: ThreadedReplyConfig(
        max_response_length=100,
        response_style="detailed",
        include_action_items=True,
        suggest_follow_up=True,
    ),
    ThreadTopic.CATCHING_UP: ThreadedReplyConfig(
        max_response_length=100,
        response_style="warm",
        include_action_items=False,
        suggest_follow_up=True,
    ),
    ThreadTopic.INFORMATION: ThreadedReplyConfig(
        max_response_length=100,
        response_style="clear",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.DECISION_MAKING: ThreadedReplyConfig(
        max_response_length=120,
        response_style="thoughtful",
        include_action_items=True,
        suggest_follow_up=True,
    ),
    ThreadTopic.CELEBRATION: ThreadedReplyConfig(
        max_response_length=80,
        response_style="enthusiastic",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.UNKNOWN: ThreadedReplyConfig(
        max_response_length=80,
        response_style="natural",
        include_action_items=False,
        suggest_follow_up=False,
    ),
}


class ThreadAnalyzer:
    """Analyzes conversation threads for context-aware reply generation.

    Uses pattern matching and optional semantic similarity for topic
    classification, state detection, and user role identification.

    Thread-safe with lazy initialization of embeddings.
    """

    TOPIC_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self) -> None:
        """Initialize the thread analyzer."""
        self._topic_embeddings: dict[ThreadTopic, np.ndarray] | None = None
        self._lock = threading.Lock()

    def _get_sentence_model(self) -> Any:
        """Get the sentence transformer model from templates module.

        Returns:
            The loaded SentenceTransformer model, or None if unavailable
        """
        try:
            from models.templates import _get_sentence_model

            return _get_sentence_model()
        except (ImportError, Exception) as e:
            logger.debug("Sentence model not available: %s", e)
            return None

    def _ensure_embeddings_computed(self) -> bool:
        """Compute and cache embeddings for topic patterns.

        Returns:
            True if embeddings are available, False otherwise
        """
        if self._topic_embeddings is not None:
            return True

        with self._lock:
            if self._topic_embeddings is not None:
                return True

            model = self._get_sentence_model()
            if model is None:
                return False

            topic_embeddings: dict[ThreadTopic, np.ndarray] = {}

            for topic, patterns in TOPIC_PATTERNS.items():
                embeddings = model.encode(patterns, convert_to_numpy=True)
                # Compute centroid
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                topic_embeddings[topic] = centroid

            self._topic_embeddings = topic_embeddings
            logger.debug("Computed topic embeddings for %d topics", len(TOPIC_PATTERNS))
            return True

    def analyze(self, messages: list[Message]) -> ThreadContext:
        """Analyze a conversation thread.

        Args:
            messages: List of messages in chronological order

        Returns:
            ThreadContext with detected topic, state, and user role
        """
        if not messages:
            return ThreadContext(
                messages=[],
                topic=ThreadTopic.UNKNOWN,
                state=ThreadState.CONCLUDED,
                user_role=UserRole.PARTICIPANT,
                confidence=0.0,
            )

        # Detect topic
        topic, confidence = self._detect_topic(messages)

        # Detect state
        state = self._detect_state(messages)

        # Detect user role
        user_role = self._detect_user_role(messages)

        # Get relevant messages for context
        relevant_messages = self._get_relevant_messages(messages, topic)

        # Extract action items
        action_items = self._extract_action_items(messages)

        # Count participants
        participants = set()
        for msg in messages:
            if not msg.is_from_me:
                participants.add(msg.sender)
        participants_count = max(1, len(participants))

        return ThreadContext(
            messages=messages,
            topic=topic,
            state=state,
            user_role=user_role,
            confidence=confidence,
            relevant_messages=relevant_messages,
            action_items=action_items,
            participants_count=participants_count,
        )

    def _detect_topic(self, messages: list[Message]) -> tuple[ThreadTopic, float]:
        """Detect the thread topic using pattern matching and semantic similarity.

        Args:
            messages: List of messages

        Returns:
            Tuple of (detected topic, confidence score)
        """
        # Combine recent messages for analysis
        recent_messages = messages[-10:]  # Last 10 messages
        combined_text = " ".join(
            msg.text.lower() for msg in recent_messages if msg.text
        )

        # First try pattern matching
        pattern_scores: dict[ThreadTopic, int] = {topic: 0 for topic in ThreadTopic}

        for topic, patterns in TOPIC_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in combined_text:
                    pattern_scores[topic] += 1

        # Find best pattern match
        best_pattern_topic = max(pattern_scores, key=lambda k: pattern_scores[k])
        pattern_score = pattern_scores[best_pattern_topic]

        if pattern_score >= 2:
            # Strong pattern match
            confidence = min(0.9, 0.5 + pattern_score * 0.1)
            return best_pattern_topic, confidence

        # Try semantic similarity if embeddings available
        if self._ensure_embeddings_computed() and self._topic_embeddings:
            try:
                model = self._get_sentence_model()
                if model:
                    query_embedding = model.encode([combined_text], convert_to_numpy=True)[0]
                    query_norm = query_embedding / np.linalg.norm(query_embedding)

                    similarities: dict[ThreadTopic, float] = {}
                    for topic, centroid in self._topic_embeddings.items():
                        similarity = float(np.dot(query_norm, centroid))
                        similarities[topic] = similarity

                    best_topic = max(similarities, key=lambda k: similarities[k])
                    best_confidence = similarities[best_topic]

                    if best_confidence >= self.TOPIC_CONFIDENCE_THRESHOLD:
                        return best_topic, best_confidence
            except Exception as e:
                logger.debug("Semantic topic detection failed: %s", e)

        # Fall back to pattern match even with low confidence
        if pattern_score >= 1:
            return best_pattern_topic, 0.3 + pattern_score * 0.1

        return ThreadTopic.UNKNOWN, 0.0

    def _detect_state(self, messages: list[Message]) -> ThreadState:
        """Detect the current state of the thread.

        Args:
            messages: List of messages

        Returns:
            Current thread state
        """
        if not messages:
            return ThreadState.CONCLUDED

        last_msg = messages[-1]
        last_text = last_msg.text.lower() if last_msg.text else ""

        # Check if last message is a question
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, last_text, re.IGNORECASE):
                if last_msg.is_from_me:
                    return ThreadState.AWAITING_RESPONSE
                else:
                    return ThreadState.OPEN_QUESTION

        # Check for conclusion patterns
        for pattern in CONCLUSION_PATTERNS:
            if re.search(pattern, last_text, re.IGNORECASE):
                return ThreadState.CONCLUDED

        # Check message timing for staleness (would need timestamps)
        # For now, use heuristic based on content

        # If last message is from other person, likely awaiting our response
        if not last_msg.is_from_me:
            return ThreadState.AWAITING_RESPONSE

        # Active discussion
        return ThreadState.IN_DISCUSSION

    def _detect_user_role(self, messages: list[Message]) -> UserRole:
        """Detect the user's role in the thread.

        Args:
            messages: List of messages

        Returns:
            User's role in the conversation
        """
        if not messages:
            return UserRole.PARTICIPANT

        # Look at first few messages to determine who initiated
        recent_start = messages[:3]
        if not recent_start:
            return UserRole.PARTICIPANT

        # Count who spoke first in this topic stretch
        first_msg = messages[0]
        if first_msg.is_from_me:
            return UserRole.INITIATOR

        # If other person started and we've responded multiple times
        my_messages = sum(1 for msg in messages if msg.is_from_me)
        total = len(messages)

        if my_messages == 0:
            return UserRole.RESPONDER
        elif my_messages / total > 0.6:
            return UserRole.INITIATOR
        elif my_messages / total < 0.3:
            return UserRole.RESPONDER
        else:
            return UserRole.PARTICIPANT

    def _get_relevant_messages(
        self, messages: list[Message], topic: ThreadTopic
    ) -> list[Message]:
        """Get the most relevant messages for context.

        Filters messages to include only those relevant to the current
        topic/context, limiting context size for focused prompts.

        Args:
            messages: All messages in the thread
            topic: Detected thread topic

        Returns:
            Subset of relevant messages
        """
        if not messages:
            return []

        # For quick exchanges, keep only last few
        if topic == ThreadTopic.QUICK_EXCHANGE:
            return messages[-3:]

        # For logistics, keep messages with times/locations
        if topic == ThreadTopic.LOGISTICS:
            relevant = []
            for msg in messages[-10:]:
                text = msg.text.lower() if msg.text else ""
                if any(
                    word in text
                    for word in ["time", "where", "address", "meet", "late", "here"]
                ):
                    relevant.append(msg)
            # Always include last few
            return list(dict.fromkeys(relevant + messages[-3:]))

        # For emotional support, keep more context
        if topic == ThreadTopic.EMOTIONAL_SUPPORT:
            return messages[-8:]

        # For planning/decision making, keep more context
        if topic in (ThreadTopic.PLANNING, ThreadTopic.DECISION_MAKING):
            return messages[-7:]

        # Default: keep last 5 messages
        return messages[-5:]

    def _extract_action_items(self, messages: list[Message]) -> list[str]:
        """Extract action items from messages.

        Args:
            messages: List of messages

        Returns:
            List of detected action items
        """
        action_items = []
        action_patterns = [
            r"(?:I'll|i'll|I will|i will)\s+(.+?)(?:\.|$)",
            r"(?:can you|could you)\s+(.+?)(?:\?|$)",
            r"(?:don't forget to|remember to)\s+(.+?)(?:\.|$)",
            r"(?:please|pls)\s+(.+?)(?:\.|$)",
            r"(?:need to|have to)\s+(.+?)(?:\.|$)",
        ]

        for msg in messages[-10:]:
            if not msg.text:
                continue
            text = msg.text

            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.strip()
                    if len(item) > 5 and len(item) < 100:
                        action_items.append(item)

        # Deduplicate while preserving order
        seen = set()
        unique_items = []
        for item in action_items:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                unique_items.append(item)

        return unique_items[:5]  # Limit to 5 action items

    def get_response_config(self, context: ThreadContext) -> ThreadedReplyConfig:
        """Get recommended response configuration for a thread context.

        Args:
            context: Analyzed thread context

        Returns:
            Configuration for generating appropriate response
        """
        return TOPIC_RESPONSE_CONFIG.get(
            context.topic,
            TOPIC_RESPONSE_CONFIG[ThreadTopic.UNKNOWN],
        )

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        with self._lock:
            self._topic_embeddings = None
            logger.debug("Thread analyzer cache cleared")


# Module-level singleton instance
_analyzer: ThreadAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_thread_analyzer() -> ThreadAnalyzer:
    """Get the singleton ThreadAnalyzer instance.

    Returns:
        The shared ThreadAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = ThreadAnalyzer()
    return _analyzer


def reset_thread_analyzer() -> None:
    """Reset the singleton ThreadAnalyzer instance.

    Call this to create a fresh analyzer on next access.
    """
    global _analyzer
    with _analyzer_lock:
        _analyzer = None
    logger.debug("Thread analyzer singleton reset")
