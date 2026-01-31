"""Relationship learning system for personalized communication.

Analyzes message history to build communication profiles for each contact,
enabling personalized reply generation that matches your usual communication
style with that person.

This module extracts:
- Communication patterns: response time, message length, formality level
- Style patterns: emoji usage, punctuation style, greeting/sign-off habits
- Topic distribution: common discussion topics with each contact

Profile storage: ~/.jarvis/relationships/{contact_hash}.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

# =============================================================================
# Constants
# =============================================================================

RELATIONSHIPS_DIR = Path.home() / ".jarvis" / "relationships"
MIN_MESSAGES_FOR_PROFILE = 20  # Minimum messages needed for reliable analysis
PROFILE_VERSION = "1.0.0"

# Embedding topic analysis
MIN_MESSAGES_FOR_EMBEDDING_TOPICS = 30
MAX_EMBEDDING_MESSAGES = 200
EMBEDDING_TOPIC_WEIGHT = 0.7
KEYWORD_TOPIC_WEIGHT = 0.3

logger = logging.getLogger(__name__)

# Emoji pattern for detection
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Common greetings for detection
GREETING_PATTERNS = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "yo",
        "sup",
        "what's up",
        "whats up",
        "hola",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "afternoon",
        "evening",
        "howdy",
        "hiya",
        "heya",
    }
)

# Common sign-offs for detection
SIGNOFF_PATTERNS = frozenset(
    {
        "bye",
        "goodbye",
        "later",
        "cya",
        "see ya",
        "see you",
        "ttyl",
        "talk later",
        "take care",
        "night",
        "goodnight",
        "gn",
        "peace",
        "cheers",
        "thanks",
        "thank you",
        "thx",
        "ty",
    }
)

# Topic keywords for classification
TOPIC_KEYWORDS: dict[str, set[str]] = {
    "scheduling": {
        "meeting",
        "schedule",
        "appointment",
        "calendar",
        "time",
        "date",
        "tomorrow",
        "today",
        "tonight",
        "weekend",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "week",
        "available",
        "free",
        "busy",
        "when",
        "plan",
        "plans",
    },
    "food": {
        "lunch",
        "dinner",
        "breakfast",
        "food",
        "eat",
        "eating",
        "hungry",
        "restaurant",
        "coffee",
        "drink",
        "drinks",
        "meal",
        "cooking",
        "recipe",
        "pizza",
        "sushi",
        "order",
        "takeout",
        "delivery",
    },
    "work": {
        "work",
        "job",
        "office",
        "boss",
        "meeting",
        "project",
        "deadline",
        "client",
        "presentation",
        "report",
        "email",
        "call",
        "conference",
        "budget",
        "team",
        "manager",
        "colleague",
        "business",
    },
    "social": {
        "party",
        "event",
        "hangout",
        "hang",
        "fun",
        "movie",
        "game",
        "games",
        "concert",
        "show",
        "festival",
        "birthday",
        "celebration",
        "invite",
        "invited",
        "join",
        "coming",
        "attend",
    },
    "family": {
        "mom",
        "dad",
        "mother",
        "father",
        "sister",
        "brother",
        "family",
        "parents",
        "kids",
        "children",
        "son",
        "daughter",
        "aunt",
        "uncle",
        "grandma",
        "grandpa",
        "cousin",
        "relatives",
        "home",
    },
    "health": {
        "doctor",
        "appointment",
        "sick",
        "feeling",
        "health",
        "hospital",
        "medicine",
        "medication",
        "symptoms",
        "better",
        "worse",
        "pain",
        "exercise",
        "gym",
        "workout",
        "fitness",
    },
    "travel": {
        "trip",
        "travel",
        "vacation",
        "flight",
        "airport",
        "hotel",
        "booking",
        "destination",
        "pack",
        "packing",
        "visit",
        "visiting",
        "road trip",
        "abroad",
        "beach",
        "mountain",
    },
    "general_chat": {
        "how",
        "doing",
        "what",
        "lol",
        "haha",
        "nice",
        "cool",
        "awesome",
        "great",
        "good",
        "bad",
        "okay",
        "sure",
        "yes",
        "no",
        "maybe",
    },
}

# Common stopwords for phrase extraction
STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ToneProfile:
    """Communication tone characteristics.

    Attributes:
        formality_score: 0.0 (very casual) to 1.0 (very formal)
        emoji_frequency: Average emojis per message
        exclamation_frequency: Average exclamation marks per message
        question_frequency: Average question marks per message
        avg_message_length: Average characters per message
        uses_caps: Whether the person uses ALL CAPS occasionally
    """

    formality_score: float = 0.5
    emoji_frequency: float = 0.0
    exclamation_frequency: float = 0.0
    question_frequency: float = 0.0
    avg_message_length: float = 50.0
    uses_caps: bool = False


@dataclass
class ResponsePatterns:
    """Response time and behavior patterns.

    Attributes:
        avg_response_time_minutes: Average time to respond (within 24h window)
        typical_response_length: Typical message length category
        greeting_style: Common greeting phrases used
        signoff_style: Common sign-off phrases used
        common_phrases: Frequently used phrases
    """

    avg_response_time_minutes: float | None = None
    typical_response_length: Literal["short", "medium", "long"] = "medium"
    greeting_style: list[str] = field(default_factory=list)
    signoff_style: list[str] = field(default_factory=list)
    common_phrases: list[str] = field(default_factory=list)


@dataclass
class TopicDistribution:
    """Distribution of conversation topics.

    Attributes:
        topics: Dictionary mapping topic name to frequency (0.0-1.0)
        top_topics: List of top 3 most discussed topics
    """

    topics: dict[str, float] = field(default_factory=dict)
    top_topics: list[str] = field(default_factory=list)


@dataclass
class RelationshipProfile:
    """Complete relationship profile for a contact.

    Attributes:
        contact_id: Unique identifier for the contact (hashed)
        contact_name: Display name if available
        tone_profile: Communication tone characteristics
        topic_distribution: Topics typically discussed
        response_patterns: Response time and style patterns
        message_count: Total messages analyzed
        last_updated: ISO timestamp of last update
        version: Profile format version
    """

    contact_id: str
    contact_name: str | None = None
    tone_profile: ToneProfile = field(default_factory=ToneProfile)
    topic_distribution: TopicDistribution = field(default_factory=TopicDistribution)
    response_patterns: ResponsePatterns = field(default_factory=ResponsePatterns)
    message_count: int = 0
    last_updated: str = ""
    version: str = PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "tone_profile": asdict(self.tone_profile),
            "topic_distribution": asdict(self.topic_distribution),
            "response_patterns": asdict(self.response_patterns),
            "message_count": self.message_count,
            "last_updated": self.last_updated,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipProfile:
        """Create profile from dictionary."""
        return cls(
            contact_id=data["contact_id"],
            contact_name=data.get("contact_name"),
            tone_profile=ToneProfile(**data.get("tone_profile", {})),
            topic_distribution=TopicDistribution(**data.get("topic_distribution", {})),
            response_patterns=ResponsePatterns(**data.get("response_patterns", {})),
            message_count=data.get("message_count", 0),
            last_updated=data.get("last_updated", ""),
            version=data.get("version", PROFILE_VERSION),
        )


# =============================================================================
# Profile Analysis Functions
# =============================================================================


def _hash_contact_id(contact_id: str) -> str:
    """Create a stable hash for contact ID storage.

    Args:
        contact_id: Phone number, email, or chat_id

    Returns:
        SHA-256 hash prefix (first 16 chars) for filename safety
    """
    return hashlib.sha256(contact_id.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    """Normalize text for token-level analysis."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text)


def _contains_all_caps_word(text: str) -> bool:
    """Check if text contains any all-caps word (length >= 3)."""
    for word in text.split():
        if len(word) >= 3 and word.isupper():
            return True
    return False


def _score_topic_keywords(text: str, keywords: set[str]) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    tokens = set(_tokenize(normalized))

    score = 0
    for keyword in keywords:
        if " " in keyword:
            if f" {keyword} " in f" {normalized} ":
                score += 2
        elif keyword in tokens:
            score += 1

    return score


def _sample_messages_for_embeddings(messages: list[Any], max_messages: int) -> list[Any]:
    if len(messages) <= max_messages:
        return messages
    step = max(1, len(messages) // max_messages)
    return messages[::step][:max_messages]


def _label_cluster_topic(sample_messages: list[str]) -> str | None:
    if not sample_messages:
        return None

    combined = " ".join(sample_messages)
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == "general_chat":
            continue
        score = _score_topic_keywords(combined, keywords)
        if score > 0:
            scores[topic] = score

    if scores:
        best_score = max(scores.values())
        best_topics = [topic for topic, score in scores.items() if score == best_score]
        return best_topics[0]

    general_score = _score_topic_keywords(combined, TOPIC_KEYWORDS["general_chat"])
    if general_score >= 2:
        return "general_chat"

    return None


def _analyze_formality(messages: list[Any]) -> float:
    """Analyze the formality level of messages.

    Returns a score from 0.0 (very casual) to 1.0 (very formal).
    """
    if not messages:
        return 0.5

    casual_indicators = {
        "lol",
        "haha",
        "hehe",
        "lmao",
        "omg",
        "btw",
        "brb",
        "ttyl",
        "idk",
        "ikr",
        "nvm",
        "tbh",
        "imo",
        "np",
        "k",
        "kk",
        "yeah",
        "yep",
        "nope",
        "gonna",
        "wanna",
        "gotta",
        "cuz",
        "bc",
        "u",
        "ur",
        "r",
        "thx",
        "ty",
        "pls",
        "plz",
        "yo",
        "dude",
        "bro",
        "sis",
        "fam",
    }

    formal_indicators = {
        "regards",
        "sincerely",
        "please",
        "kindly",
        "thank you",
        "appreciate",
        "regarding",
        "attached",
        "discussed",
        "confirmed",
        "scheduled",
        "deadline",
        "meeting",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
    }

    casual_score = 0.0
    formal_score = 0.0

    for msg in messages:
        if not msg.text:
            continue
        text = msg.text.strip()
        if not text:
            continue

        normalized = _normalize_text(text)
        words = set(_tokenize(normalized))

        casual_score += len(words & casual_indicators)
        formal_score += len(words & formal_indicators)

        # Emoji presence indicates casual
        emoji_matches = EMOJI_PATTERN.findall(text)
        if emoji_matches:
            casual_score += len(emoji_matches) * 1.5

        # Multiple exclamation marks indicate casual
        if text.count("!") > 1:
            casual_score += 1.0

        # Very short messages tend to be casual
        if len(text) <= 10:
            casual_score += 0.5
        elif len(text) >= 80:
            formal_score += 0.5

        # Sentence structure hints formality
        first_alpha = next((c for c in text if c.isalpha()), "")
        if first_alpha and first_alpha.isupper():
            formal_score += 0.5
        if text.endswith((".", "?", "!")):
            formal_score += 0.3

        # ALL CAPS usage tends to be informal
        if _contains_all_caps_word(text):
            casual_score += 0.5

    total = formal_score + casual_score
    if total == 0:
        return 0.5

    # Smooth to avoid extremes with tiny samples
    return (formal_score + 1.0) / (total + 2.0)


def _analyze_emoji_usage(messages: list[Any]) -> float:
    """Calculate average emoji usage per message."""
    if not messages:
        return 0.0

    total_emojis = 0
    for msg in messages:
        if msg.text:
            emojis = EMOJI_PATTERN.findall(msg.text)
            total_emojis += len(emojis)

    return total_emojis / len(messages)


def _analyze_punctuation(messages: list[Any]) -> tuple[float, float]:
    """Analyze exclamation and question mark usage.

    Returns:
        Tuple of (exclamation_freq, question_freq) per message
    """
    if not messages:
        return 0.0, 0.0

    exclamations = 0
    questions = 0

    for msg in messages:
        if msg.text:
            exclamations += msg.text.count("!")
            questions += msg.text.count("?")

    return exclamations / len(messages), questions / len(messages)


def _analyze_message_length(
    messages: list[Any],
) -> tuple[float, Literal["short", "medium", "long"]]:
    """Analyze typical message length.

    Returns:
        Tuple of (average_length, category)
    """
    if not messages:
        return 50.0, "medium"

    lengths = [len(msg.text) for msg in messages if msg.text]
    if not lengths:
        return 50.0, "medium"

    avg_length = sum(lengths) / len(lengths)

    if avg_length <= 30:
        category: Literal["short", "medium", "long"] = "short"
    elif avg_length <= 100:
        category = "medium"
    else:
        category = "long"

    return avg_length, category


def _analyze_caps_usage(messages: list[Any]) -> bool:
    """Check if the person occasionally uses ALL CAPS."""
    caps_count = 0
    for msg in messages:
        if msg.text and len(msg.text) >= 3:
            # Check for words in all caps (excluding single letters)
            words = msg.text.split()
            caps_words = [w for w in words if len(w) >= 3 and w.isupper()]
            if caps_words:
                caps_count += 1

    # Consider it a pattern if used in >5% of messages
    return caps_count / max(len(messages), 1) > 0.05


def _analyze_response_time(messages: list[Any]) -> float | None:
    """Calculate average response time in minutes.

    Only considers responses within 24 hours.
    """
    if len(messages) < 2:
        return None

    response_times: list[float] = []
    prev_msg = None

    # Sort by date to ensure chronological order
    sorted_messages = sorted(messages, key=lambda m: m.date)

    for msg in sorted_messages:
        if prev_msg is not None:
            # Only count if sender changed (actual response)
            if msg.is_from_me != prev_msg.is_from_me:
                time_diff = (msg.date - prev_msg.date).total_seconds()
                # Only count responses within 24 hours
                if 0 < time_diff < 86400:
                    response_times.append(time_diff / 60.0)
        prev_msg = msg

    if not response_times:
        return None

    return sum(response_times) / len(response_times)


def _extract_greetings(messages: list[Any]) -> list[str]:
    """Extract common greeting phrases from messages."""
    greetings_found: Counter[str] = Counter()

    for msg in messages:
        if not msg.text:
            continue

        # Check first few words of message
        normalized = _normalize_text(msg.text)
        first_words = " ".join(normalized.split()[:4])

        for pattern in GREETING_PATTERNS:
            if first_words.startswith(pattern):
                greetings_found[pattern] += 1
                break

    # Return top 3 greetings
    return [g for g, _ in greetings_found.most_common(3)]


def _extract_signoffs(messages: list[Any]) -> list[str]:
    """Extract common sign-off phrases from messages."""
    signoffs_found: Counter[str] = Counter()

    for msg in messages:
        if not msg.text:
            continue

        # Check last few words of message
        normalized = _normalize_text(msg.text)
        last_words = " ".join(normalized.split()[-4:])

        for pattern in SIGNOFF_PATTERNS:
            if last_words.endswith(pattern) or pattern in last_words:
                signoffs_found[pattern] += 1
                break

    # Return top 3 sign-offs
    return [s for s, _ in signoffs_found.most_common(3)]


def _extract_common_phrases(messages: list[Any], min_count: int = 3) -> list[str]:
    """Extract commonly used phrases (2-4 words)."""
    phrase_counter: Counter[str] = Counter()

    for msg in messages:
        if not msg.text or len(msg.text) < 5:
            continue

        normalized = _normalize_text(msg.text)
        words = _tokenize(normalized)

        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if len(phrase) >= 5:  # Skip very short phrases
                if not all(w in STOPWORDS for w in phrase.split()):
                    phrase_counter[phrase] += 1

        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            if not all(w in STOPWORDS for w in phrase.split()):
                phrase_counter[phrase] += 1

    # Filter and return top phrases
    filtered = [(p, c) for p, c in phrase_counter.items() if c >= min_count]
    return [p for p, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:5]]


def _build_keyword_topic_distribution(messages: list[Any]) -> TopicDistribution:
    topic_counts: Counter[str] = Counter()

    for msg in messages:
        if not msg.text:
            continue

        normalized = _normalize_text(msg.text)
        if not normalized:
            continue

        scores: dict[str, int] = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            if topic == "general_chat":
                continue
            score = _score_topic_keywords(normalized, keywords)
            if score > 0:
                scores[topic] = score

        # Fallback to general chat only if no other topic matches
        if not scores:
            general_score = _score_topic_keywords(normalized, TOPIC_KEYWORDS["general_chat"])
            if general_score >= 2:
                scores["general_chat"] = general_score

        if not scores:
            continue

        max_score = max(scores.values())
        best_topics = [topic for topic, score in scores.items() if score == max_score]
        for topic in best_topics:
            topic_counts[topic] += 1

    topics: dict[str, float] = {}
    total = sum(topic_counts.values())
    if total > 0:
        for topic, count in topic_counts.items():
            topics[topic] = round(count / total, 3)

    top_topics = [t for t, _ in topic_counts.most_common(3)]

    return TopicDistribution(topics=topics, top_topics=top_topics)


def _build_embedding_topic_distribution(
    messages: list[Any],
    embedder: Any,
) -> TopicDistribution | None:
    valid_messages = [m for m in messages if m.text and len(m.text.strip()) > 2]
    if len(valid_messages) < MIN_MESSAGES_FOR_EMBEDDING_TOPICS:
        return None

    sampled = _sample_messages_for_embeddings(valid_messages, MAX_EMBEDDING_MESSAGES)

    try:
        from jarvis.embedding_profile import build_embedding_profile
    except Exception as e:
        logger.debug("Embedding profile not available: %s", e)
        return None

    try:
        profile = build_embedding_profile(
            contact_id="topic_probe",
            messages=sampled,
            embedder=embedder,
        )
    except Exception as e:
        logger.debug("Embedding topic analysis failed: %s", e)
        return None

    if not profile.topic_clusters:
        return None

    topic_counts: Counter[str] = Counter()
    for cluster in profile.topic_clusters:
        label = _label_cluster_topic(cluster.sample_messages)
        if label is None:
            continue
        topic_counts[label] += cluster.message_count

    if not topic_counts:
        return None

    total = sum(topic_counts.values())
    topics = {topic: round(count / total, 3) for topic, count in topic_counts.items()}
    top_topics = [t for t, _ in topic_counts.most_common(3)]

    return TopicDistribution(topics=topics, top_topics=top_topics)


def _analyze_topics(
    messages: list[Any],
    embedder: Any | None = None,
    use_embeddings: bool = False,
) -> TopicDistribution:
    """Analyze topic distribution in messages."""
    keyword_distribution = _build_keyword_topic_distribution(messages)

    if not use_embeddings or embedder is None:
        return keyword_distribution

    embedding_distribution = _build_embedding_topic_distribution(messages, embedder)
    if embedding_distribution is None:
        return keyword_distribution

    merged_scores: dict[str, float] = {}
    for topic, score in keyword_distribution.topics.items():
        merged_scores[topic] = merged_scores.get(topic, 0.0) + score * KEYWORD_TOPIC_WEIGHT
    for topic, score in embedding_distribution.topics.items():
        merged_scores[topic] = merged_scores.get(topic, 0.0) + score * EMBEDDING_TOPIC_WEIGHT

    total = sum(merged_scores.values())
    if total == 0:
        return keyword_distribution

    topics = {topic: round(score / total, 3) for topic, score in merged_scores.items()}
    top_topics = [t for t, _ in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]]

    return TopicDistribution(topics=topics, top_topics=top_topics)


def build_relationship_profile(
    contact_id: str,
    messages: list[Any],
    contact_name: str | None = None,
    embedder: Any | None = None,
    use_embeddings: bool = False,
) -> RelationshipProfile:
    """Build a relationship profile from message history.

    Args:
        contact_id: Unique identifier for the contact
        messages: List of Message objects from conversation history
        contact_name: Optional display name for the contact

    Returns:
        RelationshipProfile with analyzed communication patterns
    """
    hashed_id = _hash_contact_id(contact_id)

    if len(messages) < MIN_MESSAGES_FOR_PROFILE:
        # Return minimal profile with defaults
        return RelationshipProfile(
            contact_id=hashed_id,
            contact_name=contact_name,
            message_count=len(messages),
            last_updated=datetime.now().isoformat(),
        )

    # Filter to "from me" messages for analyzing MY style with this person
    my_messages = [m for m in messages if m.is_from_me]

    # Analyze tone profile (from my messages to them)
    formality = _analyze_formality(my_messages)
    emoji_freq = _analyze_emoji_usage(my_messages)
    excl_freq, quest_freq = _analyze_punctuation(my_messages)
    avg_length, length_category = _analyze_message_length(my_messages)
    uses_caps = _analyze_caps_usage(my_messages)

    tone_profile = ToneProfile(
        formality_score=round(formality, 3),
        emoji_frequency=round(emoji_freq, 3),
        exclamation_frequency=round(excl_freq, 3),
        question_frequency=round(quest_freq, 3),
        avg_message_length=round(avg_length, 1),
        uses_caps=uses_caps,
    )

    # Analyze response patterns
    response_time = _analyze_response_time(messages)
    greetings = _extract_greetings(my_messages)
    signoffs = _extract_signoffs(my_messages)
    common_phrases = _extract_common_phrases(my_messages)

    response_patterns = ResponsePatterns(
        avg_response_time_minutes=round(response_time, 1) if response_time else None,
        typical_response_length=length_category,
        greeting_style=greetings,
        signoff_style=signoffs,
        common_phrases=common_phrases,
    )

    # Analyze topics (from all messages)
    embedder_to_use = embedder
    if use_embeddings and embedder_to_use is None:
        try:
            from jarvis.embedding_adapter import get_embedder

            candidate = get_embedder()
            if candidate.is_available():
                embedder_to_use = candidate
        except Exception as e:
            logger.debug("Embedding backend unavailable: %s", e)

    topic_distribution = _analyze_topics(
        messages,
        embedder=embedder_to_use,
        use_embeddings=use_embeddings,
    )

    return RelationshipProfile(
        contact_id=hashed_id,
        contact_name=contact_name,
        tone_profile=tone_profile,
        topic_distribution=topic_distribution,
        response_patterns=response_patterns,
        message_count=len(messages),
        last_updated=datetime.now().isoformat(),
    )


# =============================================================================
# Profile Storage Functions
# =============================================================================


def _ensure_relationships_dir() -> Path:
    """Ensure the relationships directory exists."""
    RELATIONSHIPS_DIR.mkdir(parents=True, exist_ok=True)
    return RELATIONSHIPS_DIR


def _get_profile_path(contact_id: str) -> Path:
    """Get the file path for a contact's profile."""
    hashed_id = _hash_contact_id(contact_id)
    return _ensure_relationships_dir() / f"{hashed_id}.json"


def save_profile(profile: RelationshipProfile) -> bool:
    """Save a relationship profile to disk.

    Args:
        profile: The RelationshipProfile to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Use the already-hashed contact_id from the profile
        profile_path = _ensure_relationships_dir() / f"{profile.contact_id}.json"
        with profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def load_profile(contact_id: str) -> RelationshipProfile | None:
    """Load a relationship profile from disk.

    Args:
        contact_id: The contact identifier (will be hashed)

    Returns:
        RelationshipProfile if found, None otherwise
    """
    profile_path = _get_profile_path(contact_id)

    if not profile_path.exists():
        return None

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return RelationshipProfile.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def delete_profile(contact_id: str) -> bool:
    """Delete a relationship profile from disk.

    Args:
        contact_id: The contact identifier

    Returns:
        True if deleted successfully, False otherwise
    """
    profile_path = _get_profile_path(contact_id)

    try:
        if profile_path.exists():
            profile_path.unlink()
        return True
    except OSError:
        return False


def list_profiles() -> list[str]:
    """List all saved profile IDs.

    Returns:
        List of hashed contact IDs with saved profiles
    """
    try:
        _ensure_relationships_dir()
        return [p.stem for p in RELATIONSHIPS_DIR.glob("*.json")]
    except OSError:
        return []


def profile_needs_refresh(profile: RelationshipProfile, max_age_hours: int = 24) -> bool:
    """Check if a profile needs to be refreshed.

    Args:
        profile: The profile to check
        max_age_hours: Maximum age in hours before refresh is recommended

    Returns:
        True if the profile should be refreshed
    """
    if not profile.last_updated:
        return True

    try:
        last_updated = datetime.fromisoformat(profile.last_updated)
        age = datetime.now() - last_updated
        return age > timedelta(hours=max_age_hours)
    except ValueError:
        return True


# =============================================================================
# Style Guide Generation
# =============================================================================


def generate_style_guide(profile: RelationshipProfile) -> str:
    """Generate a natural language style guide from a profile.

    This guide can be used in prompts to personalize reply generation.

    Args:
        profile: The RelationshipProfile to describe

    Returns:
        A natural language description of communication style
    """
    if profile.message_count < MIN_MESSAGES_FOR_PROFILE:
        return (
            f"Limited message history ({profile.message_count} messages). "
            "Using default casual tone."
        )

    parts: list[str] = []

    # Formality guidance
    tone = profile.tone_profile
    if tone.formality_score < 0.3:
        parts.append("Keep it very casual and relaxed")
    elif tone.formality_score < 0.5:
        parts.append("Use a casual, friendly tone")
    elif tone.formality_score < 0.7:
        parts.append("Use a balanced, conversational tone")
    else:
        parts.append("Use a more professional, polished tone")

    # Emoji guidance
    if tone.emoji_frequency > 1.5:
        parts.append("feel free to use emojis liberally")
    elif tone.emoji_frequency > 0.5:
        parts.append("include some emojis where appropriate")
    elif tone.emoji_frequency > 0.1:
        parts.append("use emojis sparingly")
    else:
        parts.append("avoid emojis")

    # Message length guidance
    patterns = profile.response_patterns
    if patterns.typical_response_length == "short":
        parts.append("keep messages brief (1-2 sentences)")
    elif patterns.typical_response_length == "long":
        parts.append("you can write longer, detailed messages")

    # Punctuation guidance
    if tone.exclamation_frequency > 0.5:
        parts.append("use exclamation marks to show enthusiasm")

    # Greeting/signoff guidance
    if patterns.greeting_style:
        greetings = ", ".join(f'"{g}"' for g in patterns.greeting_style[:2])
        parts.append(f"common greetings include {greetings}")

    if patterns.signoff_style:
        signoffs = ", ".join(f'"{s}"' for s in patterns.signoff_style[:2])
        parts.append(f"typical sign-offs include {signoffs}")

    # Topic guidance
    topics = profile.topic_distribution
    if topics.top_topics:
        topic_list = ", ".join(topics.top_topics[:2])
        parts.append(f"common topics: {topic_list}")

    # Build the guide
    if len(parts) <= 2:
        return ". ".join(parts) + "."

    # Format nicely with commas and period
    guide = parts[0]
    for i, part in enumerate(parts[1:], 1):
        if i == len(parts) - 1:
            guide += f", and {part}"
        else:
            guide += f", {part}"
    guide += "."

    return guide.capitalize()


def get_voice_guidance(profile: RelationshipProfile) -> dict[str, Any]:
    """Generate structured voice guidance for prompt building.

    Args:
        profile: The RelationshipProfile to use

    Returns:
        Dictionary with structured guidance parameters
    """
    tone = profile.tone_profile
    patterns = profile.response_patterns
    topics = profile.topic_distribution

    return {
        "formality": (
            "formal"
            if tone.formality_score >= 0.7
            else "casual"
            if tone.formality_score < 0.4
            else "balanced"
        ),
        "use_emojis": tone.emoji_frequency > 0.3,
        "emoji_level": (
            "high"
            if tone.emoji_frequency > 1.0
            else "moderate"
            if tone.emoji_frequency > 0.3
            else "low"
        ),
        "message_length": patterns.typical_response_length,
        "use_exclamations": tone.exclamation_frequency > 0.3,
        "common_greetings": patterns.greeting_style,
        "common_signoffs": patterns.signoff_style,
        "preferred_phrases": patterns.common_phrases,
        "top_topics": topics.top_topics,
        "style_guide": generate_style_guide(profile),
    }


# =============================================================================
# Prompt Personalization Helpers
# =============================================================================


def select_matching_examples(
    profile: RelationshipProfile,
    casual_examples: list[tuple[str, str]],
    professional_examples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Select few-shot examples that match the relationship style.

    Args:
        profile: The relationship profile
        casual_examples: List of casual few-shot examples
        professional_examples: List of professional few-shot examples

    Returns:
        List of examples matching the relationship's tone
    """
    formality = profile.tone_profile.formality_score

    if formality < 0.4:
        return casual_examples[:3]
    elif formality >= 0.7:
        return professional_examples[:3]
    else:
        # Mixed - take some from each
        return casual_examples[:2] + professional_examples[:1]


def enhance_prompt_with_profile(
    base_prompt: str,
    profile: RelationshipProfile | None,
) -> str:
    """Enhance a prompt with relationship-specific guidance.

    Args:
        base_prompt: The base prompt to enhance
        profile: Optional relationship profile for personalization

    Returns:
        Enhanced prompt with style guidance
    """
    if profile is None or profile.message_count < MIN_MESSAGES_FOR_PROFILE:
        return base_prompt

    style_guide = generate_style_guide(profile)

    # Insert style guidance into the prompt
    guidance_section = f"""
### Communication Style:
Based on your typical communication with this contact: {style_guide}
"""

    # Find a good insertion point (before "Your reply:" or at the end)
    if "### Your reply:" in base_prompt:
        return base_prompt.replace("### Your reply:", f"{guidance_section}\n### Your reply:")
    elif "### Last message" in base_prompt:
        return base_prompt.replace("### Last message", f"{guidance_section}\n### Last message")
    else:
        return base_prompt + "\n" + guidance_section
