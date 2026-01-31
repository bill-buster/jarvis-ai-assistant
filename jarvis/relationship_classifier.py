"""Relationship classifier using embeddings and message patterns.

Automatically classifies relationships between contacts based on
message patterns and semantic analysis. Works on ALL messages with
a contact (not just extracted pairs).

Usage:
    from jarvis.relationship_classifier import RelationshipClassifier

    classifier = RelationshipClassifier()

    # Classify a contact
    result = classifier.classify_contact("chat1234")
    print(result.relationship)  # e.g., "close friend", "family", "coworker"

    # Classify from messages directly
    result = classifier.classify_messages(messages)

    # Auto-classify all contacts
    results = classifier.classify_all_contacts()
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Relationship Categories
# =============================================================================

RELATIONSHIP_CATEGORIES = {
    "family": {
        "label": "family",
        "keywords": [
            "mom",
            "dad",
            "mother",
            "father",
            "sister",
            "brother",
            "grandma",
            "grandpa",
            "aunt",
            "uncle",
            "cousin",
            "family",
            "love you",
            "miss you",
            "thanksgiving",
            "christmas",
            "birthday",
            "come home",
            "dinner at home",
            "parents",
            "son",
            "daughter",
            "kid",
            "kids",
            "baby",
            "hubby",
            "wife",
            "husband",
        ],
        "anti_keywords": [  # Keywords that suggest NOT this category
            "meeting",
            "deadline",
            "project",
            "sprint",
            "standup",
        ],
        "patterns": {
            "time_any_hour": True,  # Family texts at any hour
            "high_emoji_rate": True,  # Lots of hearts, etc.
            "medium_frequency": True,  # Regular but not constant
        },
    },
    "close friend": {
        "label": "close friend",
        "keywords": [
            "dude",
            "bro",
            "lol",
            "lmao",
            "haha",
            "omg",
            "hangout",
            "drinks",
            "party",
            "weekend",
            "chill",
            "wanna",
            "gonna",
            "miss you",
            "bestie",
            "squad",
            "crew",
            "yo",
            "bruh",
            "bar",
            "club",
            "concert",
            "game",
            "watch the game",
        ],
        "anti_keywords": [
            "deadline",
            "sprint",
            "standup",
            "sync",
            "meeting",
        ],
        "patterns": {
            "casual_language": True,
            "late_night_messages": True,
            "high_emoji_rate": True,
        },
    },
    "romantic partner": {
        "label": "romantic partner",
        "keywords": [
            "love you",
            "miss you so much",
            "baby",
            "babe",
            "honey",
            "sweetheart",
            "date night",
            "can't wait to see you",
            "thinking of you",
            "good morning",
            "good night",
            "sleep well",
            "dream of me",
            "❤️",
            "😘",
            "💕",
            "💗",
            "💞",
            "xoxo",
            "kisses",
        ],
        "anti_keywords": [],
        "patterns": {
            "very_high_frequency": True,
            "time_any_hour": True,
            "high_emoji_rate": True,
            "long_messages": True,
        },
    },
    "coworker": {
        "label": "coworker",
        "keywords": [
            "meeting",
            "deadline",
            "project",
            "office",
            "work",
            "presentation",
            "client",
            "email",
            "slack",
            "zoom",
            "standup",
            "sprint",
            "manager",
            "boss",
            "team",
            "monday",
            "friday",
            "eod",
            "eow",
            "fyi",
            "asap",
            "sync",
            "call",
            "review",
            "pr",
            "merge",
            "deploy",
        ],
        "anti_keywords": [
            "love you",
            "miss you",
            "party",
            "drinks",
            "hangout",
        ],
        "patterns": {
            "work_hours_only": True,
            "formal_language": True,
            "low_emoji_rate": True,
        },
    },
    "acquaintance": {
        "label": "acquaintance",
        "keywords": [
            "hey",
            "hi",
            "hello",
            "how are you",
            "long time",
            "nice to meet",
            "good seeing you",
            "let's catch up",
        ],
        "anti_keywords": [],
        "patterns": {
            "low_frequency": True,
            "short_messages": True,
            "formal_language": True,
        },
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClassificationResult:
    """Result of relationship classification."""

    chat_id: str
    display_name: str
    relationship: str
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class MessageFeatures:
    """Extracted features from message history."""

    avg_message_length: float
    message_frequency_per_day: float
    emoji_rate: float
    avg_hour_of_day: float
    hour_variance: float  # High variance = texts at any hour
    formal_language_score: float
    keyword_matches: dict[str, int] = field(default_factory=dict)
    anti_keyword_matches: dict[str, int] = field(default_factory=dict)
    total_messages: int = 0
    date_range_days: int = 0
    my_messages_pct: float = 0.5  # % of messages I sent
    avg_response_time_hours: float = 0.0


@dataclass
class ChatMessage:
    """Lightweight message for classification."""

    text: str
    is_from_me: bool
    date: datetime | None
    chat_id: str


# =============================================================================
# Classifier
# =============================================================================


class RelationshipClassifier:
    """Classifies relationships based on message patterns.

    Uses a combination of:
    - Keyword matching
    - Time patterns (when messages are sent)
    - Message style (formal vs casual)
    - Response patterns

    Does NOT use LLM - fast and deterministic.
    """

    def __init__(
        self,
        chat_db_path: Path | None = None,
        min_messages: int = 10,
    ):
        """Initialize the classifier.

        Args:
            chat_db_path: Path to iMessage chat.db. Uses default if None.
            min_messages: Minimum messages needed for classification.
        """
        self.chat_db_path = chat_db_path or Path.home() / "Library/Messages/chat.db"
        self.min_messages = min_messages
        self._category_embeddings: dict[str, np.ndarray] | None = None

    def _get_messages_for_chat(
        self,
        chat_id: str,
        limit: int = 500,
    ) -> list[ChatMessage]:
        """Get messages for a specific chat from chat.db.

        Args:
            chat_id: The chat identifier.
            limit: Maximum messages to retrieve.

        Returns:
            List of ChatMessage objects.
        """
        if not self.chat_db_path.exists():
            logger.warning(f"Chat database not found: {self.chat_db_path}")
            return []

        messages = []
        try:
            conn = sqlite3.connect(
                f"file:{self.chat_db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row

            # Query messages for this chat
            query = """
                SELECT
                    m.text,
                    m.is_from_me,
                    m.date as date_int,
                    c.chat_identifier
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE c.chat_identifier = ?
                    AND m.text IS NOT NULL
                    AND m.text != ''
                ORDER BY m.date DESC
                LIMIT ?
            """

            cursor = conn.execute(query, (chat_id, limit))

            for row in cursor:
                # Convert macOS timestamp to datetime
                date_int = row["date_int"]
                if date_int:
                    # macOS uses nanoseconds since 2001-01-01
                    timestamp = date_int / 1_000_000_000 + 978307200
                    date = datetime.fromtimestamp(timestamp)
                else:
                    date = None

                messages.append(
                    ChatMessage(
                        text=row["text"] or "",
                        is_from_me=bool(row["is_from_me"]),
                        date=date,
                        chat_id=row["chat_identifier"],
                    )
                )

            conn.close()

        except Exception as e:
            logger.error(f"Failed to get messages for chat {chat_id}: {e}")

        return messages

    def _get_non_group_chats(self) -> list[tuple[str, str]]:
        """Get all non-group chats (1:1 conversations).

        Returns:
            List of (chat_id, display_name) tuples.
        """
        if not self.chat_db_path.exists():
            return []

        chats = []
        try:
            conn = sqlite3.connect(
                f"file:{self.chat_db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row

            # Get 1:1 chats: phone numbers/emails that aren't group chats
            # Group chats have identifiers starting with "chat" and multiple handles
            query = """
                SELECT
                    c.chat_identifier,
                    c.display_name,
                    (SELECT COUNT(*) FROM chat_handle_join WHERE chat_id = c.ROWID) as handle_count,
                    (SELECT COUNT(*) FROM chat_message_join WHERE chat_id = c.ROWID) as msg_count
                FROM chat c
                WHERE handle_count = 1
                ORDER BY msg_count DESC
            """

            cursor = conn.execute(query)

            for row in cursor:
                chat_id = row["chat_identifier"]
                display_name = row["display_name"] or chat_id
                chats.append((chat_id, display_name))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to get chats: {e}")

        return chats

    def _extract_features(self, messages: list[ChatMessage]) -> MessageFeatures:
        """Extract features from message history.

        Args:
            messages: List of messages from a conversation.

        Returns:
            MessageFeatures with computed statistics.
        """
        if not messages:
            return MessageFeatures(
                avg_message_length=0,
                message_frequency_per_day=0,
                emoji_rate=0,
                avg_hour_of_day=12,
                hour_variance=0,
                formal_language_score=0.5,
            )

        # Calculate basic stats
        lengths = []
        hours = []
        emoji_count = 0
        total_chars = 0
        formal_indicators = 0
        informal_indicators = 0
        my_messages = 0
        keyword_matches: dict[str, int] = {cat: 0 for cat in RELATIONSHIP_CATEGORIES}
        anti_keyword_matches: dict[str, int] = {cat: 0 for cat in RELATIONSHIP_CATEGORIES}

        for msg in messages:
            text = msg.text or ""
            lengths.append(len(text))
            total_chars += len(text)

            if msg.is_from_me:
                my_messages += 1

            # Extract hour if date available
            if msg.date:
                hours.append(msg.date.hour)

            # Count emojis (simple heuristic)
            emoji_chars = sum(1 for c in text if ord(c) > 0x1F300)
            emoji_count += emoji_chars

            # Formal vs informal indicators
            text_lower = text.lower()
            if any(w in text_lower for w in ["please", "thank you", "regards", "sincerely"]):
                formal_indicators += 1
            informal_words = ["lol", "lmao", "haha", "omg", "u ", " ur ", "gonna", "wanna"]
            if any(w in text_lower for w in informal_words):
                informal_indicators += 1

            # Keyword matching
            for cat, info in RELATIONSHIP_CATEGORIES.items():
                for keyword in info["keywords"]:
                    if keyword.lower() in text_lower:
                        keyword_matches[cat] += 1
                for keyword in info.get("anti_keywords", []):
                    if keyword.lower() in text_lower:
                        anti_keyword_matches[cat] += 1

        # Calculate date range
        dates = [m.date for m in messages if m.date]
        if len(dates) >= 2:
            date_range = (max(dates) - min(dates)).days
        else:
            date_range = 1

        avg_length = sum(lengths) / len(lengths) if lengths else 0
        freq_per_day = len(messages) / max(date_range, 1)
        emoji_rate = emoji_count / max(total_chars, 1)
        avg_hour = sum(hours) / len(hours) if hours else 12
        hour_var = float(np.var(hours)) if len(hours) > 1 else 0

        # Formal language score (0-1, higher = more formal)
        total_indicators = formal_indicators + informal_indicators
        formal_score = formal_indicators / max(total_indicators, 1) if total_indicators else 0.5

        my_pct = my_messages / len(messages) if messages else 0.5

        return MessageFeatures(
            avg_message_length=avg_length,
            message_frequency_per_day=freq_per_day,
            emoji_rate=emoji_rate,
            avg_hour_of_day=avg_hour,
            hour_variance=hour_var,
            formal_language_score=formal_score,
            keyword_matches=keyword_matches,
            anti_keyword_matches=anti_keyword_matches,
            total_messages=len(messages),
            date_range_days=date_range,
            my_messages_pct=my_pct,
        )

    def _compute_scores(self, features: MessageFeatures) -> dict[str, float]:
        """Compute scores for each relationship category.

        Args:
            features: Extracted message features.

        Returns:
            Dictionary of category -> score.
        """
        scores: dict[str, float] = {}

        for cat, info in RELATIONSHIP_CATEGORIES.items():
            score = 0.0

            # Keyword match score (normalized)
            keyword_score = min(features.keyword_matches.get(cat, 0) / 10, 1.0)
            score += keyword_score * 0.4  # 40% weight

            # Anti-keyword penalty
            anti_score = features.anti_keyword_matches.get(cat, 0)
            if anti_score > 0:
                score -= min(anti_score / 5, 0.3)  # Max 30% penalty

            # Pattern matching
            patterns = info.get("patterns", {})

            # Time patterns
            if patterns.get("work_hours_only"):
                in_work_hours = 9 <= features.avg_hour_of_day <= 17
                low_variance = features.hour_variance < 10
                work_hour_score = 1.0 if (in_work_hours and low_variance) else 0.0
                score += work_hour_score * 0.2
            elif patterns.get("time_any_hour"):
                any_hour_score = min(features.hour_variance / 30, 1.0)
                score += any_hour_score * 0.15
            elif patterns.get("late_night_messages"):
                is_late_night = features.avg_hour_of_day >= 20 or features.avg_hour_of_day <= 2
                late_score = 1.0 if is_late_night else 0.0
                score += late_score * 0.15

            # Emoji rate
            if patterns.get("high_emoji_rate"):
                emoji_score = min(features.emoji_rate * 50, 1.0)
                score += emoji_score * 0.15
            elif patterns.get("low_emoji_rate"):
                emoji_score = 1.0 - min(features.emoji_rate * 50, 1.0)
                score += emoji_score * 0.15

            # Formality
            if patterns.get("formal_language"):
                score += features.formal_language_score * 0.15
            elif patterns.get("casual_language"):
                score += (1.0 - features.formal_language_score) * 0.15

            # Frequency
            if patterns.get("very_high_frequency"):
                freq_score = min(features.message_frequency_per_day / 5, 1.0)
                score += freq_score * 0.1
            elif patterns.get("low_frequency"):
                freq_score = 1.0 if features.message_frequency_per_day < 0.5 else 0.0
                score += freq_score * 0.1

            scores[cat] = max(score, 0.0)

        return scores

    def classify_messages(
        self,
        messages: list[ChatMessage],
        chat_id: str = "",
        display_name: str = "",
    ) -> ClassificationResult:
        """Classify relationship from a list of messages.

        Args:
            messages: List of ChatMessage objects.
            chat_id: Optional chat identifier.
            display_name: Optional display name.

        Returns:
            ClassificationResult with relationship type and confidence.
        """
        if len(messages) < self.min_messages:
            return ClassificationResult(
                chat_id=chat_id,
                display_name=display_name,
                relationship="unknown",
                confidence=0.0,
                sample_size=len(messages),
            )

        features = self._extract_features(messages)
        scores = self._compute_scores(features)

        # Find best match
        best_category = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_category]

        # Normalize confidence (0-1)
        confidence = min(best_score / 0.6, 1.0)

        return ClassificationResult(
            chat_id=chat_id,
            display_name=display_name,
            relationship=best_category,
            confidence=round(confidence, 2),
            signals=scores,
            sample_size=len(messages),
            features={
                "avg_length": features.avg_message_length,
                "freq_per_day": features.message_frequency_per_day,
                "emoji_rate": features.emoji_rate,
                "avg_hour": features.avg_hour_of_day,
                "formality": features.formal_language_score,
            },
        )

    def classify_contact(self, chat_id: str, display_name: str = "") -> ClassificationResult:
        """Classify a specific contact by chat_id.

        Args:
            chat_id: The chat identifier (phone/email).
            display_name: Optional display name.

        Returns:
            ClassificationResult with relationship type and confidence.
        """
        messages = self._get_messages_for_chat(chat_id)
        return self.classify_messages(messages, chat_id, display_name)

    def classify_all_contacts(
        self,
        limit: int = 100,
        min_confidence: float = 0.3,
    ) -> list[ClassificationResult]:
        """Classify all 1:1 contacts (not groups).

        Args:
            limit: Maximum number of contacts to classify.
            min_confidence: Minimum confidence to include.

        Returns:
            List of ClassificationResult objects.
        """
        chats = self._get_non_group_chats()[:limit]
        results = []

        for chat_id, display_name in chats:
            result = self.classify_contact(chat_id, display_name)
            if result.confidence >= min_confidence:
                results.append(result)

        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def evaluate(
        self,
        ground_truth: dict[str, str],
    ) -> dict[str, float]:
        """Evaluate classifier against ground truth labels.

        Args:
            ground_truth: Dict of chat_id -> correct relationship.

        Returns:
            Evaluation metrics (accuracy, per-category precision/recall).
        """
        correct = 0
        total = 0
        per_category: dict[str, dict[str, int]] = {
            cat: {"tp": 0, "fp": 0, "fn": 0} for cat in RELATIONSHIP_CATEGORIES
        }

        for chat_id, true_label in ground_truth.items():
            result = self.classify_contact(chat_id)
            predicted = result.relationship

            if predicted == true_label:
                correct += 1
                per_category[true_label]["tp"] += 1
            else:
                if predicted in per_category:
                    per_category[predicted]["fp"] += 1
                if true_label in per_category:
                    per_category[true_label]["fn"] += 1

            total += 1

        accuracy = correct / total if total > 0 else 0

        # Calculate per-category metrics
        metrics = {"accuracy": accuracy}
        for cat, counts in per_category.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"{cat}_precision"] = precision
            metrics[f"{cat}_recall"] = recall

        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================


def suggest_relationship(chat_id: str) -> str:
    """Get relationship suggestion for a contact.

    Args:
        chat_id: The chat identifier.

    Returns:
        Relationship label string.
    """
    classifier = RelationshipClassifier()
    result = classifier.classify_contact(chat_id)
    if result.confidence < 0.3:
        return "unknown"
    return result.relationship


def classify_top_contacts(n: int = 20) -> None:
    """Classify and print top N contacts.

    Args:
        n: Number of contacts to classify.
    """
    classifier = RelationshipClassifier()
    results = classifier.classify_all_contacts(limit=n)

    print(f"\n{'Contact':<25} {'Relationship':<18} {'Confidence':<12} {'Messages':<10}")
    print("-" * 70)

    for r in results:
        name = r.display_name[:24] if r.display_name else r.chat_id[:24]
        print(f"{name:<25} {r.relationship:<18} {r.confidence:<12.2f} {r.sample_size:<10}")


if __name__ == "__main__":
    classify_top_contacts(20)
