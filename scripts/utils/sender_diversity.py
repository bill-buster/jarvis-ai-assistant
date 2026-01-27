"""
Sender Diversity Analysis

Filters templates to ensure they generalize across multiple senders,
not just work for one specific person.
"""

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def calculate_sender_diversity(patterns: list[dict]) -> list[dict]:
    """Add sender diversity metrics to patterns.

    Args:
        patterns: List of pattern dicts

    Returns:
        Patterns with added sender_diversity_score
    """
    for pattern in patterns:
        num_senders = pattern.get("num_senders", 1)
        frequency = pattern.get("total_frequency", pattern.get("frequency", 1))

        # Diversity score: what fraction of instances are from unique senders?
        # Perfect diversity: every instance is from a different sender (score=1.0)
        # No diversity: all instances from one sender (score=0.0)
        diversity_score = min(1.0, num_senders / max(1, frequency))

        pattern["sender_diversity_score"] = diversity_score

    return patterns


def filter_by_sender_diversity(
    patterns: list[dict],
    min_senders: int = 3,
    min_diversity_score: float = 0.3
) -> list[dict]:
    """Filter patterns by sender diversity.

    Keeps only patterns that appear with multiple senders,
    indicating they generalize across relationships.

    Args:
        patterns: List of pattern dicts
        min_senders: Minimum number of unique senders
        min_diversity_score: Minimum diversity score (0-1)

    Returns:
        Filtered patterns
    """
    filtered = []
    removed_count = 0

    for pattern in patterns:
        num_senders = pattern.get("num_senders", 1)
        diversity_score = pattern.get("sender_diversity_score", 0.0)

        if num_senders >= min_senders and diversity_score >= min_diversity_score:
            filtered.append(pattern)
        else:
            removed_count += 1
            logger.debug(
                "Removed low-diversity pattern: '%s' → '%s' (senders=%d, diversity=%.2f)",
                pattern.get("representative_incoming", "")[:40],
                pattern.get("representative_response", "")[:40],
                num_senders,
                diversity_score
            )

    logger.info(
        "Sender diversity filter: kept %d patterns, removed %d (%.1f%% kept)",
        len(filtered),
        removed_count,
        100 * len(filtered) / max(1, len(patterns))
    )

    return filtered


def analyze_sender_distribution(
    response_groups: list[dict]
) -> dict[str, Any]:
    """Analyze distribution of senders.

    Args:
        response_groups: List of response group dicts

    Returns:
        Statistics about sender distribution
    """
    pattern_senders = defaultdict(set)
    sender_message_counts = defaultdict(int)

    for group in response_groups:
        pattern = (
            group.get("incoming", "").lower().strip(),
            group.get("response", "").lower().strip()
        )
        sender_id = group.get("sender_id")

        if sender_id:
            pattern_senders[pattern].add(sender_id)
            sender_message_counts[sender_id] += 1

    # Calculate diversity stats
    diversities = []
    for pattern, senders in pattern_senders.items():
        diversity = len(senders)
        diversities.append(diversity)

    import numpy as np

    return {
        "total_patterns": len(pattern_senders),
        "total_senders": len(sender_message_counts),
        "avg_senders_per_pattern": np.mean(diversities) if diversities else 0,
        "median_senders_per_pattern": np.median(diversities) if diversities else 0,
        "patterns_single_sender": sum(1 for d in diversities if d == 1),
        "patterns_multi_sender": sum(1 for d in diversities if d >= 3),
        "most_prolific_senders": sorted(
            sender_message_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


def detect_overfitting_to_sender(
    pattern: dict,
    sender_threshold: float = 0.7
) -> bool:
    """Detect if pattern is overfitted to a specific sender.

    Args:
        pattern: Pattern dict with sender info
        sender_threshold: If one sender accounts for >this fraction, flag as overfit

    Returns:
        True if overfitted, False otherwise
    """
    if "sender_distribution" not in pattern:
        return False

    sender_dist = pattern["sender_distribution"]
    total = sum(sender_dist.values())

    if total == 0:
        return False

    # Check if any single sender dominates
    max_sender_count = max(sender_dist.values())
    max_sender_ratio = max_sender_count / total

    return max_sender_ratio > sender_threshold


def add_sender_distribution(
    patterns: list[dict],
    response_groups: list[dict]
) -> list[dict]:
    """Add detailed sender distribution to patterns.

    Args:
        patterns: List of cluster patterns
        response_groups: Original response groups with sender info

    Returns:
        Patterns with added sender_distribution field
    """
    # Build mapping from pattern to sender counts
    pattern_sender_counts = defaultdict(lambda: defaultdict(int))

    for group in response_groups:
        pattern_key = (
            group.get("incoming", "").lower().strip(),
            group.get("response", "").lower().strip()
        )
        sender_id = group.get("sender_id")

        if sender_id:
            pattern_sender_counts[pattern_key][sender_id] += 1

    # Add to patterns
    for pattern in patterns:
        pattern_key = (
            pattern.get("representative_incoming", "").lower().strip(),
            pattern.get("representative_response", "").lower().strip()
        )

        if pattern_key in pattern_sender_counts:
            pattern["sender_distribution"] = dict(pattern_sender_counts[pattern_key])

            # Also add anonymized version for privacy
            sender_counts = sorted(pattern_sender_counts[pattern_key].values(), reverse=True)
            pattern["sender_counts_anonymous"] = sender_counts[:5]  # Top 5 only

    return patterns
