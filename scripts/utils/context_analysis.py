"""
Context Analysis Utilities

Provides functions for:
1. Detecting message formality (formal/casual/neutral)
2. Detecting sender relationship type
3. Stratifying messages by context
4. Group size categorization
5. Day-of-week and time-of-day analysis
"""

from collections import Counter, defaultdict
from typing import Any


def detect_formality(text: str) -> str:
    """Detect if message is formal, casual, or neutral.

    Args:
        text: Message text

    Returns:
        "formal", "casual", or "neutral"
    """
    text_lower = text.lower()

    # Formal indicators
    formal_indicators = [
        "please",
        "thank you",
        "sincerely",
        "regards",
        "kindly",
        "meeting",
        "schedule",
        "appointment",
        "deadline",
        "appreciate",
        "professional",
        "business",
        "corporate",
        "respectfully",
    ]

    # Casual indicators
    casual_indicators = [
        "lol",
        "haha",
        "lmao",
        "omg",
        "wtf",
        "tbh",
        "ngl",
        "bruh",
        "yeah",
        "yep",
        "nah",
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sup",
        "hey",
        "yo",
        "dude",
        "bro",
        "man",
        "cool",
        "chill",
    ]

    formal_count = sum(1 for ind in formal_indicators if ind in text_lower)
    casual_count = sum(1 for ind in casual_indicators if ind in text_lower)

    if formal_count > casual_count and formal_count > 0:
        return "formal"
    elif casual_count > formal_count and casual_count > 0:
        return "casual"
    else:
        return "neutral"


def get_group_size_category(participant_count: int) -> str:
    """Categorize group by size.

    Args:
        participant_count: Number of participants

    Returns:
        "direct", "small_group", "medium_group", or "large_group"
    """
    if participant_count <= 2:
        return "direct"
    elif participant_count <= 5:
        return "small_group"
    elif participant_count <= 10:
        return "medium_group"
    else:
        return "large_group"


def get_time_category(hour: int) -> str:
    """Categorize time of day.

    Args:
        hour: Hour (0-23)

    Returns:
        "night", "morning", "afternoon", "evening"
    """
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def get_day_category(weekday: int) -> str:
    """Categorize day of week.

    Args:
        weekday: Day (0=Monday, 6=Sunday)

    Returns:
        "weekday" or "weekend"
    """
    return "weekend" if weekday >= 5 else "weekday"


def create_context_key(
    formality: str, group_category: str, time_category: str, day_category: str
) -> str:
    """Create a unique context key for stratification.

    Args:
        formality: "formal", "casual", or "neutral"
        group_category: "direct", "small_group", etc.
        time_category: "morning", "afternoon", etc.
        day_category: "weekday" or "weekend"

    Returns:
        Context key string like "casual_direct_evening_weekday"
    """
    return f"{formality}_{group_category}_{time_category}_{day_category}"


def stratify_by_context(
    response_groups: list[dict], min_samples_per_strata: int = 10
) -> dict[str, list[dict]]:
    """Stratify response groups by context.

    Groups messages with similar context together for separate clustering.

    Args:
        response_groups: List of response group dicts
        min_samples_per_strata: Minimum samples to keep a stratum

    Returns:
        Dict mapping context keys to lists of response groups
    """
    strata = defaultdict(list)

    for group in response_groups:
        # Extract context info
        formality = group.get("formality", "neutral")
        group_category = group.get("group_category", "direct")
        time_category = group.get("time_category", "evening")
        day_category = group.get("day_category", "weekday")

        # Create context key
        context_key = create_context_key(formality, group_category, time_category, day_category)

        strata[context_key].append(group)

    # Filter out strata with too few samples
    filtered_strata = {
        key: groups for key, groups in strata.items() if len(groups) >= min_samples_per_strata
    }

    return filtered_strata


def calculate_adaptive_conversation_gap(messages: list[dict], percentile: float = 75.0) -> float:
    """Calculate adaptive conversation gap threshold.

    Uses historical gaps to determine when a new conversation starts.

    Args:
        messages: List of message dicts with date_ns
        percentile: Percentile to use as threshold (default 75th)

    Returns:
        Gap threshold in hours
    """
    if len(messages) < 2:
        return 24.0  # Default to 24 hours

    gaps = []
    for i in range(1, len(messages)):
        gap_hours = (messages[i]["date_ns"] - messages[i - 1]["date_ns"]) / (1e9 * 3600)
        if 0 < gap_hours < 168:  # Filter out gaps > 1 week (outliers)
            gaps.append(gap_hours)

    if not gaps:
        return 24.0

    import numpy as np

    threshold = np.percentile(gaps, percentile)

    # Clamp to reasonable range (1-72 hours)
    return max(1.0, min(72.0, threshold))


def detect_sender_relationship(
    sender_id: str, all_messages: list[dict], formality_counts: dict[str, Counter]
) -> str:
    """Detect relationship type with a sender.

    Args:
        sender_id: Sender handle ID
        all_messages: All messages from this sender
        formality_counts: Dict mapping sender_id to formality Counter

    Returns:
        "professional", "friend", "family", "acquaintance"
    """
    if sender_id not in formality_counts:
        return "acquaintance"

    counts = formality_counts[sender_id]
    total = sum(counts.values())

    if total == 0:
        return "acquaintance"

    formal_ratio = counts.get("formal", 0) / total
    casual_ratio = counts.get("casual", 0) / total

    # Professional: mostly formal
    if formal_ratio > 0.6:
        return "professional"

    # Friend: mostly casual
    if casual_ratio > 0.6:
        return "friend"

    # Family: mix but high message volume
    if total > 100:
        return "family"

    return "acquaintance"


def analyze_context_distribution(response_groups: list[dict]) -> dict[str, Any]:
    """Analyze context distribution in response groups.

    Args:
        response_groups: List of response group dicts

    Returns:
        Distribution statistics
    """
    formality_counts = Counter()
    group_counts = Counter()
    time_counts = Counter()
    day_counts = Counter()

    for group in response_groups:
        formality_counts[group.get("formality", "neutral")] += 1
        group_counts[group.get("group_category", "direct")] += 1
        time_counts[group.get("time_category", "evening")] += 1
        day_counts[group.get("day_category", "weekday")] += 1

    return {
        "total_groups": len(response_groups),
        "formality": dict(formality_counts),
        "group_size": dict(group_counts),
        "time_of_day": dict(time_counts),
        "day_of_week": dict(day_counts),
    }


def segment_conversation(
    messages: list[dict], gap_threshold_hours: float = 24.0
) -> list[list[dict]]:
    """Split chat into conversations by time gaps.

    Args:
        messages: List of message dicts with date_ns
        gap_threshold_hours: Hours of inactivity that define a new conversation

    Returns:
        List of conversation segments
    """
    if not messages:
        return []

    conversations = []
    current_conv = [messages[0]]

    for i in range(1, len(messages)):
        time_gap_hours = (messages[i]["date_ns"] - messages[i - 1]["date_ns"]) / (1e9 * 3600)

        if time_gap_hours > gap_threshold_hours:
            conversations.append(current_conv)
            current_conv = [messages[i]]
        else:
            current_conv.append(messages[i])

    conversations.append(current_conv)
    return conversations
