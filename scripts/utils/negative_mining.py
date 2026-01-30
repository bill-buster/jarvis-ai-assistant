"""
Negative Pattern Mining

Identifies patterns that should NOT be used as templates:
1. Responses followed by apologies/clarifications
2. Responses that got negative reactions
3. Responses in sensitive contexts
"""

import logging

logger = logging.getLogger(__name__)

# Apology/clarification markers
APOLOGY_MARKERS = [
    "sorry",
    "my bad",
    "oops",
    "didn't mean",
    "meant to say",
    "correction",
    "wait no",
    "actually no",
    "*",  # Correction marker
]

# Negative reaction markers
NEGATIVE_REACTIONS = [
    "what?",
    "huh?",
    "confused",
    "don't understand",
    "makes no sense",
    "??",
    "???",
]


def mine_negative_patterns(messages: list[dict], lookback_window: int = 3) -> list[tuple[str, str]]:
    """Mine patterns where response was followed by apology/clarification.

    Args:
        messages: List of all messages in chronological order
        lookback_window: How many messages to look back for incoming

    Returns:
        List of (incoming, response) tuples to avoid
    """
    negative_patterns = []

    for i, msg in enumerate(messages):
        if not msg.get("is_from_me"):
            continue

        text = msg.get("text", "").lower()

        # Check if this message contains apology/clarification markers
        is_apology = any(marker in text for marker in APOLOGY_MARKERS)

        if is_apology:
            # Look back for the problematic response
            for j in range(i - 1, max(0, i - lookback_window), -1):
                prev_msg = messages[j]

                if prev_msg.get("is_from_me"):
                    # Found your previous message - this might be the bad one
                    # Look further back for the incoming
                    for k in range(j - 1, max(0, j - lookback_window), -1):
                        incoming_msg = messages[k]

                        if not incoming_msg.get("is_from_me"):
                            # Found the incoming message
                            negative_patterns.append(
                                (incoming_msg.get("text", ""), prev_msg.get("text", ""))
                            )
                            break
                    break

    logger.info(
        "Mined %d negative patterns (responses followed by apology)", len(negative_patterns)
    )
    return negative_patterns


def check_negative_reaction(incoming: str, response: str, subsequent_messages: list[str]) -> bool:
    """Check if response got negative reaction.

    Args:
        incoming: Incoming message
        response: Your response
        subsequent_messages: Following messages from other person

    Returns:
        True if negative reaction detected
    """
    for msg in subsequent_messages[:2]:  # Check next 2 messages
        msg_lower = msg.lower()

        for neg_marker in NEGATIVE_REACTIONS:
            if neg_marker in msg_lower:
                logger.debug(
                    "Negative reaction detected for: '%s' → '%s' (reaction: '%s')",
                    incoming[:40],
                    response[:40],
                    msg[:40],
                )
                return True

    return False


def filter_negative_patterns(
    patterns: list[dict], negative_patterns: list[tuple[str, str]], threshold: float = 0.8
) -> list[dict]:
    """Filter out patterns that match negative examples.

    Args:
        patterns: List of mined patterns
        negative_patterns: List of (incoming, response) tuples to avoid
        threshold: Similarity threshold for matching (0-1)

    Returns:
        Filtered patterns
    """
    if not negative_patterns:
        return patterns

    filtered = []
    removed_count = 0

    # Create set of negative patterns (lowercased and stripped)
    negative_set = set(
        (inc.lower().strip(), resp.lower().strip()) for inc, resp in negative_patterns
    )

    for pattern in patterns:
        incoming = pattern.get("representative_incoming", "").lower().strip()
        response = pattern.get("representative_response", "").lower().strip()

        pattern_tuple = (incoming, response)

        if pattern_tuple in negative_set:
            removed_count += 1
            logger.debug("Removed negative pattern: '%s' → '%s'", incoming[:40], response[:40])
        else:
            filtered.append(pattern)

    logger.info(
        "Negative pattern filter: kept %d patterns, removed %d", len(filtered), removed_count
    )

    return filtered


def detect_sensitive_context(text: str) -> bool:
    """Detect if message is in sensitive context.

    Sensitive contexts where templates should not be used:
    - Personal problems
    - Bad news
    - Emotional distress

    Args:
        text: Message text

    Returns:
        True if sensitive context detected
    """
    text_lower = text.lower()

    sensitive_markers = [
        "died",
        "passed away",
        "hospital",
        "sick",
        "cancer",
        "broke up",
        "divorce",
        "fired",
        "laid off",
        "depressed",
        "anxious",
        "crying",
        "hurt",
        "emergency",
        "accident",
        "police",
    ]

    return any(marker in text_lower for marker in sensitive_markers)


def add_negative_flags(patterns: list[dict]) -> list[dict]:
    """Add flags for potentially problematic patterns.

    Args:
        patterns: List of patterns

    Returns:
        Patterns with added negative flags
    """
    for pattern in patterns:
        incoming = pattern.get("representative_incoming", "")
        response = pattern.get("representative_response", "")

        # Check for sensitive context
        pattern["is_sensitive_context"] = detect_sensitive_context(incoming)

        # Check if response is too short for important message
        if detect_sensitive_context(incoming) and len(response.strip()) < 10:
            pattern["warning_too_brief_for_sensitive"] = True

        # Check if response contains apology markers (might be problematic)
        if any(marker in response.lower() for marker in APOLOGY_MARKERS):
            pattern["contains_apology"] = True

    return patterns
