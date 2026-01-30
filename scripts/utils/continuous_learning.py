"""
Continuous Learning Utilities

Handles:
1. Adaptive template weighting with recency bias
2. Concept drift detection
3. Pattern deprecation
4. Incremental updates
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def calculate_recency_boost(
    pattern_dates: list[int], current_time_ns: int, recency_window_days: int = 90
) -> float:
    """Calculate boost for recent usage.

    Args:
        pattern_dates: List of timestamps when pattern was used
        current_time_ns: Current time in nanoseconds
        recency_window_days: Days to consider "recent"

    Returns:
        Recency boost multiplier (1.0 = no boost, 2.0 = 2× boost)
    """
    recency_window_ns = recency_window_days * 24 * 3600 * 1_000_000_000

    recent_count = sum(1 for d in pattern_dates if (current_time_ns - d) <= recency_window_ns)

    # Boost by 2× for each recent use
    return 1.0 + (recent_count * 2.0)


def calculate_stability_discount(pattern_age_days: int, decay_rate: float = 0.1) -> float:
    """Calculate discount for old patterns.

    Older patterns get discounted to prefer current style.

    Args:
        pattern_age_days: Age of pattern in days
        decay_rate: How fast to decay (0.1 = 10% per year)

    Returns:
        Stability discount factor (0-1)
    """
    age_years = pattern_age_days / 365.0
    discount = 1.0 / (1.0 + age_years * decay_rate)

    return discount


def calculate_adaptive_weight(
    pattern: dict, current_time_ns: int, recency_window_days: int = 90, decay_rate: float = 0.1
) -> float:
    """Calculate adaptive weight for pattern.

    Combines:
    - Base score (frequency × consistency × recency)
    - Recent usage boost
    - Historical stability discount

    Args:
        pattern: Pattern dict
        current_time_ns: Current time in nanoseconds
        recency_window_days: Days to consider "recent"
        decay_rate: Historical decay rate

    Returns:
        Adaptive weight score
    """
    base_score = pattern.get("combined_score", 1.0)

    # Recency boost
    pattern_dates = pattern.get("all_dates", [])
    recency_boost = calculate_recency_boost(pattern_dates, current_time_ns, recency_window_days)

    # Stability discount
    age_days = pattern.get("age_days", 0)
    stability_discount = calculate_stability_discount(age_days, decay_rate)

    # Combined adaptive weight
    adaptive_weight = (base_score + recency_boost) * stability_discount

    return adaptive_weight


def detect_concept_drift(
    historical_patterns: list[dict], recent_messages: list[dict], drift_threshold: float = 0.3
) -> dict[str, Any]:
    """Detect if communication style has drifted.

    Compares historical patterns with recent message characteristics.

    Args:
        historical_patterns: Patterns mined from all-time data
        recent_messages: Recent messages (last 3-6 months)
        drift_threshold: Threshold for drift detection (0-1)

    Returns:
        Drift analysis results
    """
    # Extract characteristics from historical patterns
    historical_formality = Counter()
    historical_length = []

    for pattern in historical_patterns[:100]:  # Top 100 patterns
        formality = pattern.get("formality", "neutral")
        response = pattern.get("representative_response", "")

        historical_formality[formality] += 1
        historical_length.append(len(response))

    # Extract characteristics from recent messages
    recent_formality = Counter()
    recent_length = []

    for msg in recent_messages:
        if msg.get("is_from_me"):
            text = msg.get("text", "")
            from scripts.utils.context_analysis import detect_formality

            formality = detect_formality(text)
            recent_formality[formality] += 1
            recent_length.append(len(text))

    # Calculate drift metrics
    import numpy as np

    # Formality drift
    hist_formal_ratio = historical_formality.get("formal", 0) / max(
        1, sum(historical_formality.values())
    )
    recent_formal_ratio = recent_formality.get("formal", 0) / max(1, sum(recent_formality.values()))
    formality_drift = abs(hist_formal_ratio - recent_formal_ratio)

    # Length drift
    hist_avg_length = np.mean(historical_length) if historical_length else 0
    recent_avg_length = np.mean(recent_length) if recent_length else 0
    length_drift = abs(hist_avg_length - recent_avg_length) / max(1, hist_avg_length)

    # Overall drift
    overall_drift = (formality_drift + length_drift) / 2.0

    drift_detected = overall_drift > drift_threshold

    return {
        "drift_detected": drift_detected,
        "overall_drift": overall_drift,
        "formality_drift": formality_drift,
        "length_drift": length_drift,
        "historical_formal_ratio": hist_formal_ratio,
        "recent_formal_ratio": recent_formal_ratio,
        "historical_avg_length": hist_avg_length,
        "recent_avg_length": recent_avg_length,
        "recommendation": "Retrain templates" if drift_detected else "Templates still valid",
    }


def deprecate_outdated_patterns(
    patterns: list[dict], current_time_ns: int, max_age_days: int = 730, min_recent_usage: int = 2
) -> list[dict]:
    """Deprecate patterns that haven't been used recently.

    Args:
        patterns: List of patterns
        current_time_ns: Current time in nanoseconds
        max_age_days: Maximum age without recent usage
        min_recent_usage: Minimum recent uses to stay active

    Returns:
        Filtered patterns
    """
    filtered = []
    deprecated_count = 0

    recency_window_ns = 180 * 24 * 3600 * 1_000_000_000  # 6 months

    for pattern in patterns:
        age_days = pattern.get("age_days", 0)
        all_dates = pattern.get("all_dates", [])

        # Count recent uses
        recent_count = sum(1 for d in all_dates if (current_time_ns - d) <= recency_window_ns)

        # Deprecate if old and not recently used
        if age_days > max_age_days and recent_count < min_recent_usage:
            deprecated_count += 1
            logger.debug(
                "Deprecated outdated pattern: '%s' → '%s' (age=%dd, recent_uses=%d)",
                pattern.get("representative_incoming", "")[:40],
                pattern.get("representative_response", "")[:40],
                age_days,
                recent_count,
            )
            pattern["deprecated"] = True
            pattern["deprecation_reason"] = f"Not used in last 6 months (age={age_days}d)"
        else:
            filtered.append(pattern)

    logger.info(
        "Deprecated %d outdated patterns (%.1f%%)",
        deprecated_count,
        100 * deprecated_count / max(1, len(patterns)),
    )

    return filtered


class IncrementalTemplateIndex:
    """Incremental template index for continuous updates.

    Maintains state between runs to only process new messages.
    """

    def __init__(self, state_file: Path | None = None):
        """Initialize index.

        Args:
            state_file: Path to state file for persistence
        """
        self.state_file = state_file or Path("results/.template_index_state.json")
        self.last_processed_rowid = 0
        self.patterns = []
        self.load_state()

    def load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.last_processed_rowid = state.get("last_processed_rowid", 0)
                    self.patterns = state.get("patterns", [])
                logger.info("Loaded incremental state: last_rowid=%d", self.last_processed_rowid)
            except Exception as e:
                logger.warning("Failed to load state: %s", e)

    def save_state(self):
        """Save state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(
                    {
                        "last_processed_rowid": self.last_processed_rowid,
                        "patterns": self.patterns,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved incremental state: last_rowid=%d", self.last_processed_rowid)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def get_new_messages_query(self) -> str:
        """Get SQL query for new messages since last update.

        Returns:
            SQL query string
        """
        return f"""
            SELECT
                m.ROWID,
                m.text,
                m.is_from_me,
                m.date,
                m.handle_id,
                cmj.chat_id
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            WHERE m.ROWID > {self.last_processed_rowid}
              AND m.text IS NOT NULL
              AND m.text != ''
            ORDER BY m.ROWID
        """

    def update_with_new_messages(self, new_messages: list[dict]):
        """Update patterns with new messages.

        Args:
            new_messages: List of new message dicts
        """
        if not new_messages:
            logger.info("No new messages to process")
            return

        # Update last processed ID
        self.last_processed_rowid = max(msg["rowid"] for msg in new_messages)

        # TODO: Remine patterns with new messages
        # This is a simplified version - full implementation would:
        # 1. Extract new response pairs
        # 2. Update pattern counts
        # 3. Recalculate scores
        # 4. Merge with existing patterns

        logger.info(
            "Processed %d new messages, last_rowid=%d", len(new_messages), self.last_processed_rowid
        )

        self.save_state()
