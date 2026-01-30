"""Shared emoji utilities.

Consolidates emoji detection/removal logic used across the codebase.
"""

import re

# Comprehensive emoji pattern covering all common Unicode emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """Remove all emojis from text.

    Args:
        text: Input text potentially containing emojis

    Returns:
        Text with emojis removed
    """
    return EMOJI_PATTERN.sub("", text).strip()


def has_emoji(text: str) -> bool:
    """Check if text contains any emojis.

    Args:
        text: Input text to check

    Returns:
        True if text contains emojis
    """
    return bool(EMOJI_PATTERN.search(text))
