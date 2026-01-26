"""Response Evaluation Framework for JARVIS.

Provides automated metrics for evaluating AI-generated responses:
- Tone consistency score (matches conversation's historical tone)
- Relevance score (semantic similarity to recent context)
- Naturalness score (perplexity-based, sounds like real texts)
- Length appropriateness (compared to user's typical message length)

Also provides human feedback tracking:
- Track when users send suggested responses unchanged (implicit positive)
- Track when users edit suggestions (capture before/after for learning)
- Track when users dismiss suggestions (implicit negative)
- Store feedback in ~/.jarvis/feedback.jsonl

Thread-safe implementations suitable for concurrent access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
FEEDBACK_FILE_NAME = "feedback.jsonl"
MAX_FEEDBACK_ENTRIES = 10000  # Maximum entries to keep in memory
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Same model used in intent.py


class FeedbackAction(str, Enum):
    """Types of user feedback actions on suggestions."""

    SENT = "sent"  # User sent the suggestion unchanged
    EDITED = "edited"  # User edited before sending
    DISMISSED = "dismissed"  # User dismissed the suggestion
    COPIED = "copied"  # User copied but didn't send yet


@dataclass
class ToneAnalysis:
    """Results of tone analysis for a text or conversation."""

    formality_score: float  # 0.0 (very casual) to 1.0 (very formal)
    emoji_density: float  # Emoji count per 100 chars
    exclamation_rate: float  # Exclamation marks per sentence
    question_rate: float  # Questions per sentence
    avg_sentence_length: float  # Average words per sentence
    abbreviation_count: int  # Count of common abbreviations


@dataclass
class EvaluationResult:
    """Complete evaluation result for a response.

    Attributes:
        tone_score: How well the response matches conversation tone (0-1)
        relevance_score: Semantic similarity to recent context (0-1)
        naturalness_score: How natural the response sounds (0-1)
        length_score: How appropriate the length is (0-1)
        overall_score: Weighted average of all scores (0-1)
        details: Additional analysis details
    """

    tone_score: float
    relevance_score: float
    naturalness_score: float
    length_score: float
    overall_score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackEntry:
    """A single feedback entry for a suggestion.

    Attributes:
        timestamp: When the feedback was recorded
        action: The feedback action type
        suggestion_id: Hash of the suggestion text
        suggestion_text: The original suggestion
        edited_text: The edited text (if action is EDITED)
        chat_id: Conversation ID for context
        context_hash: Hash of recent context (for grouping)
        evaluation: Optional evaluation scores for the suggestion
        metadata: Additional context data
    """

    timestamp: datetime
    action: FeedbackAction
    suggestion_id: str
    suggestion_text: str
    edited_text: str | None
    chat_id: str
    context_hash: str
    evaluation: EvaluationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToneAnalyzer:
    """Analyzes the tone of text and conversations.

    Evaluates formality, emoji usage, punctuation patterns,
    and other indicators of communication style.
    """

    # Common casual abbreviations
    CASUAL_ABBREVIATIONS = {
        "lol",
        "lmao",
        "omg",
        "btw",
        "idk",
        "tbh",
        "imo",
        "ngl",
        "rn",
        "ty",
        "thx",
        "np",
        "pls",
        "plz",
        "u",
        "ur",
        "r",
        "k",
        "ok",
        "ya",
        "yea",
        "yeah",
        "nah",
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sorta",
        "cuz",
        "bc",
        "tho",
        "nvm",
        "brb",
        "ttyl",
        "ily",
        "omw",
    }

    # Formal indicators
    FORMAL_PHRASES = {
        "would you",
        "could you",
        "please",
        "thank you",
        "appreciate",
        "regards",
        "sincerely",
        "unfortunately",
        "furthermore",
        "however",
        "therefore",
        "additionally",
        "accordingly",
        "consequently",
    }

    def __init__(self) -> None:
        """Initialize the tone analyzer."""
        # Emoji regex pattern
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def analyze(self, text: str) -> ToneAnalysis:
        """Analyze the tone of a text.

        Args:
            text: The text to analyze

        Returns:
            ToneAnalysis with detailed metrics
        """
        if not text or not text.strip():
            return ToneAnalysis(
                formality_score=0.5,
                emoji_density=0.0,
                exclamation_rate=0.0,
                question_rate=0.0,
                avg_sentence_length=0.0,
                abbreviation_count=0,
            )

        text_lower = text.lower()
        words = text_lower.split()
        char_count = len(text)

        # Count emojis
        emojis = self._emoji_pattern.findall(text)
        emoji_count = sum(len(e) for e in emojis)
        emoji_density = (emoji_count / char_count * 100) if char_count > 0 else 0.0

        # Count sentences and punctuation
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))

        exclamation_count = text.count("!")
        question_count = text.count("?")
        exclamation_rate = exclamation_count / sentence_count
        question_rate = question_count / sentence_count

        # Calculate average sentence length
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_sentence_length = (
            statistics.mean(words_per_sentence) if words_per_sentence else 0.0
        )

        # Count casual abbreviations
        abbreviation_count = sum(
            1 for word in words if word.strip(".,!?") in self.CASUAL_ABBREVIATIONS
        )

        # Count formal phrases
        formal_count = sum(1 for phrase in self.FORMAL_PHRASES if phrase in text_lower)

        # Calculate formality score
        casual_indicators = abbreviation_count + emoji_count + exclamation_count
        formal_indicators = formal_count + (1 if avg_sentence_length > 15 else 0)

        total_indicators = casual_indicators + formal_indicators + 1  # +1 to avoid div/0
        formality_score = formal_indicators / total_indicators

        # Adjust based on sentence length (longer = more formal)
        if avg_sentence_length > 20:
            formality_score = min(1.0, formality_score + 0.2)
        elif avg_sentence_length < 5:
            formality_score = max(0.0, formality_score - 0.2)

        return ToneAnalysis(
            formality_score=round(formality_score, 3),
            emoji_density=round(emoji_density, 3),
            exclamation_rate=round(exclamation_rate, 3),
            question_rate=round(question_rate, 3),
            avg_sentence_length=round(avg_sentence_length, 2),
            abbreviation_count=abbreviation_count,
        )

    def compute_tone_similarity(
        self, tone1: ToneAnalysis, tone2: ToneAnalysis
    ) -> float:
        """Compute similarity between two tone analyses.

        Args:
            tone1: First tone analysis
            tone2: Second tone analysis

        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Weight the different components
        formality_diff = abs(tone1.formality_score - tone2.formality_score)
        emoji_diff = abs(tone1.emoji_density - tone2.emoji_density) / 10  # Normalize
        exclamation_diff = abs(tone1.exclamation_rate - tone2.exclamation_rate)
        length_diff = (
            abs(tone1.avg_sentence_length - tone2.avg_sentence_length) / 30
        )  # Normalize

        # Weighted average of differences
        total_diff = (
            0.4 * formality_diff
            + 0.2 * emoji_diff
            + 0.2 * exclamation_diff
            + 0.2 * length_diff
        )

        return max(0.0, min(1.0, 1.0 - total_diff))


class ResponseEvaluator:
    """Evaluates AI-generated responses against multiple quality metrics.

    Thread-safe implementation using lazy-loaded models.
    """

    # Weight for overall score calculation
    TONE_WEIGHT = 0.25
    RELEVANCE_WEIGHT = 0.35
    NATURALNESS_WEIGHT = 0.25
    LENGTH_WEIGHT = 0.15

    def __init__(self) -> None:
        """Initialize the response evaluator."""
        self._tone_analyzer = ToneAnalyzer()
        self._sentence_model: Any = None
        self._lock = threading.Lock()

    def _get_sentence_model(self) -> Any:
        """Get the sentence transformer model (lazy loaded).

        Returns:
            The loaded SentenceTransformer model
        """
        if self._sentence_model is None:
            with self._lock:
                if self._sentence_model is None:
                    try:
                        from models.templates import _get_sentence_model

                        self._sentence_model = _get_sentence_model()
                    except ImportError:
                        logger.warning(
                            "Sentence model not available, relevance scoring disabled"
                        )
        return self._sentence_model

    def evaluate(
        self,
        response: str,
        context_messages: list[str],
        user_messages: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a response against all quality metrics.

        Args:
            response: The AI-generated response to evaluate
            context_messages: Recent messages from the conversation
            user_messages: User's own messages (for length comparison)

        Returns:
            EvaluationResult with all scores
        """
        # Compute individual scores
        tone_score = self._compute_tone_score(response, context_messages)
        relevance_score = self._compute_relevance_score(response, context_messages)
        naturalness_score = self._compute_naturalness_score(response)
        length_score = self._compute_length_score(response, user_messages or [])

        # Compute weighted overall score
        overall_score = (
            self.TONE_WEIGHT * tone_score
            + self.RELEVANCE_WEIGHT * relevance_score
            + self.NATURALNESS_WEIGHT * naturalness_score
            + self.LENGTH_WEIGHT * length_score
        )

        return EvaluationResult(
            tone_score=round(tone_score, 3),
            relevance_score=round(relevance_score, 3),
            naturalness_score=round(naturalness_score, 3),
            length_score=round(length_score, 3),
            overall_score=round(overall_score, 3),
            details={
                "context_message_count": len(context_messages),
                "user_message_count": len(user_messages) if user_messages else 0,
                "response_length": len(response),
            },
        )

    def _compute_tone_score(
        self, response: str, context_messages: list[str]
    ) -> float:
        """Compute tone consistency score.

        Args:
            response: The response to evaluate
            context_messages: Recent conversation messages

        Returns:
            Tone consistency score (0-1)
        """
        if not context_messages:
            return 0.5  # Neutral score without context

        # Analyze response tone
        response_tone = self._tone_analyzer.analyze(response)

        # Analyze context tone (aggregate recent messages)
        context_text = " ".join(context_messages[-10:])  # Last 10 messages
        context_tone = self._tone_analyzer.analyze(context_text)

        return self._tone_analyzer.compute_tone_similarity(response_tone, context_tone)

    def _compute_relevance_score(
        self, response: str, context_messages: list[str]
    ) -> float:
        """Compute semantic relevance score.

        Args:
            response: The response to evaluate
            context_messages: Recent conversation messages

        Returns:
            Relevance score (0-1)
        """
        if not context_messages:
            return 0.5

        model = self._get_sentence_model()
        if model is None:
            return 0.5  # Default if model unavailable

        try:
            # Combine recent context (last 5 messages for relevance)
            recent_context = " ".join(context_messages[-5:])

            # Compute embeddings
            embeddings = model.encode(
                [response, recent_context], convert_to_numpy=True
            )

            # Cosine similarity
            response_emb = embeddings[0]
            context_emb = embeddings[1]

            similarity = float(
                np.dot(response_emb, context_emb)
                / (np.linalg.norm(response_emb) * np.linalg.norm(context_emb))
            )

            # Map similarity to 0-1 range (typically ranges from 0.3-0.9)
            return max(0.0, min(1.0, (similarity - 0.2) / 0.6))

        except Exception:
            logger.exception("Error computing relevance score")
            return 0.5

    def _compute_naturalness_score(self, response: str) -> float:
        """Compute naturalness score based on text patterns.

        Uses heuristics to detect unnatural patterns:
        - Repetitive phrases
        - Unusual punctuation
        - Overly formal or robotic language

        Args:
            response: The response to evaluate

        Returns:
            Naturalness score (0-1)
        """
        if not response or not response.strip():
            return 0.0

        score = 1.0  # Start with perfect score, deduct for issues

        words = response.lower().split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Check for repetitive words (deduct for high repetition)
        unique_words = set(words)
        repetition_ratio = len(unique_words) / word_count
        if repetition_ratio < 0.5:
            score -= 0.2

        # Check for consecutive repeated words (e.g., "the the")
        consecutive_repeats = sum(
            1 for i in range(1, len(words)) if words[i] == words[i - 1]
        )
        if consecutive_repeats > 0:
            score -= min(0.3, consecutive_repeats * 0.1)

        # Check for robotic phrases
        robotic_phrases = [
            "as an ai",
            "i cannot",
            "i don't have",
            "i apologize",
            "certainly!",
            "absolutely!",
            "i'd be happy to",
            "let me",
        ]
        response_lower = response.lower()
        for phrase in robotic_phrases:
            if phrase in response_lower:
                score -= 0.15

        # Check for unusual punctuation density
        punct_count = sum(1 for c in response if c in ".,!?;:")
        punct_ratio = punct_count / len(response)
        if punct_ratio > 0.15:  # More than 15% punctuation is unusual
            score -= 0.1

        # Bonus for contractions (more natural)
        contractions = ["'m", "'re", "'ll", "'ve", "'d", "n't", "'s"]
        has_contractions = any(c in response_lower for c in contractions)
        if has_contractions:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _compute_length_score(
        self, response: str, user_messages: list[str]
    ) -> float:
        """Compute length appropriateness score.

        Compares response length to user's typical message length.

        Args:
            response: The response to evaluate
            user_messages: User's own messages for comparison

        Returns:
            Length appropriateness score (0-1)
        """
        response_length = len(response)

        if not user_messages:
            # Default scoring based on reasonable length range
            if 10 <= response_length <= 200:
                return 1.0
            elif response_length < 10:
                return 0.5
            elif response_length <= 500:
                return 0.8
            else:
                return 0.5

        # Calculate user's average message length
        user_lengths = [len(m) for m in user_messages if m.strip()]
        if not user_lengths:
            return 0.7

        avg_user_length = statistics.mean(user_lengths)
        std_dev = statistics.stdev(user_lengths) if len(user_lengths) > 1 else 50

        # Score based on how close response is to user's typical length
        # Allow some variance (responses can be slightly longer)
        diff = abs(response_length - avg_user_length * 1.2)  # Allow 20% longer
        normalized_diff = diff / max(std_dev, 30)

        # Convert to score using exponential decay
        score = math.exp(-0.5 * normalized_diff)

        return max(0.0, min(1.0, score))


class FeedbackStore:
    """Stores and manages user feedback on suggestions.

    Thread-safe implementation with file-based persistence.
    Feedback is stored in JSONL format at ~/.jarvis/feedback.jsonl
    """

    def __init__(self, feedback_dir: Path | None = None) -> None:
        """Initialize the feedback store.

        Args:
            feedback_dir: Directory for feedback file (default: ~/.jarvis)
        """
        if feedback_dir is None:
            feedback_dir = Path.home() / ".jarvis"

        self._feedback_dir = feedback_dir
        self._feedback_file = feedback_dir / FEEDBACK_FILE_NAME
        self._lock = threading.Lock()
        self._entries: list[FeedbackEntry] = []
        self._loaded = False

        # Statistics cache
        self._stats_cache: dict[str, Any] | None = None
        self._stats_cache_time: float = 0

    def _ensure_dir(self) -> None:
        """Ensure the feedback directory exists."""
        self._feedback_dir.mkdir(parents=True, exist_ok=True)

    def _load_if_needed(self) -> None:
        """Load feedback from file if not already loaded."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            self._ensure_dir()

            if self._feedback_file.exists():
                try:
                    with open(self._feedback_file, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    entry = self._entry_from_dict(data)
                                    if entry:
                                        self._entries.append(entry)
                                except json.JSONDecodeError:
                                    continue

                    # Trim to max size if needed
                    if len(self._entries) > MAX_FEEDBACK_ENTRIES:
                        self._entries = self._entries[-MAX_FEEDBACK_ENTRIES:]

                    logger.info("Loaded %d feedback entries", len(self._entries))

                except Exception:
                    logger.exception("Error loading feedback file")

            self._loaded = True

    def _entry_from_dict(self, data: dict[str, Any]) -> FeedbackEntry | None:
        """Convert dictionary to FeedbackEntry.

        Args:
            data: Dictionary from JSON

        Returns:
            FeedbackEntry or None if invalid
        """
        try:
            evaluation = None
            if data.get("evaluation"):
                eval_data = data["evaluation"]
                evaluation = EvaluationResult(
                    tone_score=eval_data.get("tone_score", 0.5),
                    relevance_score=eval_data.get("relevance_score", 0.5),
                    naturalness_score=eval_data.get("naturalness_score", 0.5),
                    length_score=eval_data.get("length_score", 0.5),
                    overall_score=eval_data.get("overall_score", 0.5),
                    details=eval_data.get("details", {}),
                )

            return FeedbackEntry(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                action=FeedbackAction(data["action"]),
                suggestion_id=data["suggestion_id"],
                suggestion_text=data["suggestion_text"],
                edited_text=data.get("edited_text"),
                chat_id=data["chat_id"],
                context_hash=data["context_hash"],
                evaluation=evaluation,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError):
            return None

    def _entry_to_dict(self, entry: FeedbackEntry) -> dict[str, Any]:
        """Convert FeedbackEntry to dictionary.

        Args:
            entry: The feedback entry

        Returns:
            Dictionary for JSON serialization
        """
        result: dict[str, Any] = {
            "timestamp": entry.timestamp.isoformat(),
            "action": entry.action.value,
            "suggestion_id": entry.suggestion_id,
            "suggestion_text": entry.suggestion_text,
            "edited_text": entry.edited_text,
            "chat_id": entry.chat_id,
            "context_hash": entry.context_hash,
            "metadata": entry.metadata,
        }

        if entry.evaluation:
            result["evaluation"] = {
                "tone_score": entry.evaluation.tone_score,
                "relevance_score": entry.evaluation.relevance_score,
                "naturalness_score": entry.evaluation.naturalness_score,
                "length_score": entry.evaluation.length_score,
                "overall_score": entry.evaluation.overall_score,
                "details": entry.evaluation.details,
            }

        return result

    def record_feedback(
        self,
        action: FeedbackAction,
        suggestion_text: str,
        chat_id: str,
        context_messages: list[str],
        edited_text: str | None = None,
        evaluation: EvaluationResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record user feedback on a suggestion.

        Args:
            action: The feedback action type
            suggestion_text: The original suggestion
            chat_id: Conversation ID
            context_messages: Recent context messages
            edited_text: Edited text if action is EDITED
            evaluation: Optional evaluation scores
            metadata: Additional context data

        Returns:
            The created FeedbackEntry
        """
        self._load_if_needed()

        # Generate IDs
        suggestion_id = hashlib.sha256(suggestion_text.encode()).hexdigest()[:16]
        context_hash = hashlib.sha256(
            " ".join(context_messages[-5:]).encode()
        ).hexdigest()[:16]

        entry = FeedbackEntry(
            timestamp=datetime.now(UTC),
            action=action,
            suggestion_id=suggestion_id,
            suggestion_text=suggestion_text,
            edited_text=edited_text,
            chat_id=chat_id,
            context_hash=context_hash,
            evaluation=evaluation,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)

            # Persist to file
            try:
                self._ensure_dir()
                with open(self._feedback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._entry_to_dict(entry)) + "\n")
            except Exception:
                logger.exception("Error writing feedback entry")

            # Trim if needed
            if len(self._entries) > MAX_FEEDBACK_ENTRIES:
                self._entries = self._entries[-MAX_FEEDBACK_ENTRIES:]

            # Invalidate stats cache
            self._stats_cache = None

        logger.debug(
            "Recorded feedback: action=%s, suggestion_id=%s",
            action.value,
            suggestion_id,
        )

        return entry

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate feedback statistics.

        Returns:
            Dictionary with feedback metrics
        """
        self._load_if_needed()

        # Check cache (5 second TTL)
        now = time.time()
        if self._stats_cache and now - self._stats_cache_time < 5:
            return self._stats_cache

        with self._lock:
            total = len(self._entries)
            if total == 0:
                return {
                    "total_feedback": 0,
                    "sent_unchanged": 0,
                    "edited": 0,
                    "dismissed": 0,
                    "copied": 0,
                    "acceptance_rate": 0.0,
                    "edit_rate": 0.0,
                    "avg_evaluation_scores": None,
                }

            # Count by action
            action_counts = {action: 0 for action in FeedbackAction}
            eval_scores: list[EvaluationResult] = []

            for entry in self._entries:
                action_counts[entry.action] += 1
                if entry.evaluation:
                    eval_scores.append(entry.evaluation)

            sent = action_counts[FeedbackAction.SENT]
            edited = action_counts[FeedbackAction.EDITED]
            dismissed = action_counts[FeedbackAction.DISMISSED]
            copied = action_counts[FeedbackAction.COPIED]

            # Calculate rates
            total_actioned = sent + edited + dismissed
            acceptance_rate = sent / total_actioned if total_actioned > 0 else 0.0
            edit_rate = edited / total_actioned if total_actioned > 0 else 0.0

            # Average evaluation scores
            avg_scores = None
            if eval_scores:
                avg_scores = {
                    "tone_score": statistics.mean(e.tone_score for e in eval_scores),
                    "relevance_score": statistics.mean(
                        e.relevance_score for e in eval_scores
                    ),
                    "naturalness_score": statistics.mean(
                        e.naturalness_score for e in eval_scores
                    ),
                    "length_score": statistics.mean(
                        e.length_score for e in eval_scores
                    ),
                    "overall_score": statistics.mean(
                        e.overall_score for e in eval_scores
                    ),
                }

            stats = {
                "total_feedback": total,
                "sent_unchanged": sent,
                "edited": edited,
                "dismissed": dismissed,
                "copied": copied,
                "acceptance_rate": round(acceptance_rate, 3),
                "edit_rate": round(edit_rate, 3),
                "avg_evaluation_scores": avg_scores,
            }

            self._stats_cache = stats
            self._stats_cache_time = now

            return stats

    def get_improvements(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get suggested improvements based on feedback patterns.

        Analyzes edited suggestions to identify patterns for improvement.

        Args:
            limit: Maximum number of suggestions to return

        Returns:
            List of improvement suggestions
        """
        self._load_if_needed()

        improvements: list[dict[str, Any]] = []

        with self._lock:
            # Get edited entries with both original and edited text
            edited_entries = [
                e
                for e in self._entries
                if e.action == FeedbackAction.EDITED and e.edited_text
            ]

            if not edited_entries:
                return []

            # Analyze patterns in edits
            length_changes: list[float] = []
            tone_changes: list[tuple[float, float]] = []
            common_removals: dict[str, int] = {}
            common_additions: dict[str, int] = {}

            analyzer = ToneAnalyzer()

            for entry in edited_entries[-100:]:  # Analyze last 100 edits
                original = entry.suggestion_text
                edited = entry.edited_text or ""

                # Length change analysis
                length_ratio = len(edited) / len(original) if original else 1.0
                length_changes.append(length_ratio)

                # Tone change analysis
                original_tone = analyzer.analyze(original)
                edited_tone = analyzer.analyze(edited)
                tone_changes.append(
                    (original_tone.formality_score, edited_tone.formality_score)
                )

                # Word-level changes
                original_words = set(original.lower().split())
                edited_words = set(edited.lower().split())

                for word in original_words - edited_words:
                    common_removals[word] = common_removals.get(word, 0) + 1

                for word in edited_words - original_words:
                    common_additions[word] = common_additions.get(word, 0) + 1

            # Generate improvement suggestions
            if length_changes:
                avg_length_ratio = statistics.mean(length_changes)
                if avg_length_ratio < 0.7:
                    improvements.append(
                        {
                            "type": "length",
                            "suggestion": "Generate shorter responses",
                            "detail": f"Users typically shorten suggestions by {(1-avg_length_ratio)*100:.0f}%",
                            "confidence": min(0.9, len(length_changes) / 20),
                        }
                    )
                elif avg_length_ratio > 1.3:
                    improvements.append(
                        {
                            "type": "length",
                            "suggestion": "Generate longer, more detailed responses",
                            "detail": f"Users typically expand suggestions by {(avg_length_ratio-1)*100:.0f}%",
                            "confidence": min(0.9, len(length_changes) / 20),
                        }
                    )

            if tone_changes:
                orig_tones = [t[0] for t in tone_changes]
                edit_tones = [t[1] for t in tone_changes]
                avg_orig = statistics.mean(orig_tones)
                avg_edit = statistics.mean(edit_tones)

                if avg_edit < avg_orig - 0.15:
                    improvements.append(
                        {
                            "type": "tone",
                            "suggestion": "Use more casual language",
                            "detail": "Users often make responses less formal",
                            "confidence": min(0.9, len(tone_changes) / 20),
                        }
                    )
                elif avg_edit > avg_orig + 0.15:
                    improvements.append(
                        {
                            "type": "tone",
                            "suggestion": "Use more formal language",
                            "detail": "Users often make responses more formal",
                            "confidence": min(0.9, len(tone_changes) / 20),
                        }
                    )

            # Common word changes
            top_removals = sorted(
                common_removals.items(), key=lambda x: x[1], reverse=True
            )[:5]
            if top_removals and top_removals[0][1] >= 3:
                improvements.append(
                    {
                        "type": "vocabulary",
                        "suggestion": "Avoid certain words/phrases",
                        "detail": f"Frequently removed: {', '.join(w for w, _ in top_removals)}",
                        "confidence": min(0.8, top_removals[0][1] / 10),
                    }
                )

            top_additions = sorted(
                common_additions.items(), key=lambda x: x[1], reverse=True
            )[:5]
            if top_additions and top_additions[0][1] >= 3:
                improvements.append(
                    {
                        "type": "vocabulary",
                        "suggestion": "Include certain words/phrases more often",
                        "detail": f"Frequently added: {', '.join(w for w, _ in top_additions)}",
                        "confidence": min(0.8, top_additions[0][1] / 10),
                    }
                )

        return improvements[:limit]

    def get_recent_entries(self, limit: int = 50) -> list[FeedbackEntry]:
        """Get recent feedback entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent FeedbackEntry objects
        """
        self._load_if_needed()

        with self._lock:
            return list(reversed(self._entries[-limit:]))

    def clear(self) -> None:
        """Clear all feedback data (for testing)."""
        with self._lock:
            self._entries.clear()
            self._stats_cache = None
            if self._feedback_file.exists():
                self._feedback_file.unlink()
            self._loaded = False


# Module-level singleton instances
_evaluator: ResponseEvaluator | None = None
_feedback_store: FeedbackStore | None = None
_lock = threading.Lock()


def get_response_evaluator() -> ResponseEvaluator:
    """Get the singleton ResponseEvaluator instance.

    Returns:
        Shared ResponseEvaluator instance
    """
    global _evaluator
    if _evaluator is None:
        with _lock:
            if _evaluator is None:
                _evaluator = ResponseEvaluator()
    return _evaluator


def get_feedback_store() -> FeedbackStore:
    """Get the singleton FeedbackStore instance.

    Returns:
        Shared FeedbackStore instance
    """
    global _feedback_store
    if _feedback_store is None:
        with _lock:
            if _feedback_store is None:
                _feedback_store = FeedbackStore()
    return _feedback_store


def reset_evaluation() -> None:
    """Reset all evaluation singletons (for testing)."""
    global _evaluator, _feedback_store
    with _lock:
        _evaluator = None
        _feedback_store = None


# Export public symbols
__all__ = [
    # Enums
    "FeedbackAction",
    # Data classes
    "ToneAnalysis",
    "EvaluationResult",
    "FeedbackEntry",
    # Classes
    "ToneAnalyzer",
    "ResponseEvaluator",
    "FeedbackStore",
    # Singleton accessors
    "get_response_evaluator",
    "get_feedback_store",
    "reset_evaluation",
]
