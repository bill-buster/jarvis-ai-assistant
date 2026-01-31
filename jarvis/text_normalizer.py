"""Text Normalizer - Canonical text cleaning for the extraction pipeline.

Single source of truth for all text cleaning used throughout the pipeline.
All text should pass through normalize_text() before being stored or compared.

Usage:
    from jarvis.text_normalizer import normalize_text, extract_text_features

    cleaned = normalize_text("Liked "hey there"")  # Returns empty string (reaction)
    features = extract_text_features("btw, are you free?")  # {"starts_new_topic": True, ...}
"""

import re
from dataclasses import dataclass
from typing import Any

# Reaction patterns - these are tapbacks in iMessage
REACTION_PATTERNS = [
    r'^Liked\s+".*"$',
    r'^Loved\s+".*"$',
    r'^Disliked\s+".*"$',
    r'^Laughed at\s+".*"$',
    r'^Emphasized\s+".*"$',
    r'^Questioned\s+".*"$',
    r'^Removed a like from\s+".*"$',
    r'^Removed a heart from\s+".*"$',
    r'^Removed a dislike from\s+".*"$',
    r'^Removed a laugh from\s+".*"$',
    r'^Removed an exclamation from\s+".*"$',
    r'^Removed a question mark from\s+".*"$',
]
REACTION_REGEX = re.compile("|".join(REACTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# Acknowledgment phrases that are too generic to be useful alone
# NOTE: Emotional reactions (cool, nice, good, great, awesome, lol, haha) are NOT acknowledgments
# They express emotion and need context-aware LLM generation
ACKNOWLEDGMENT_PHRASES = frozenset(
    {
        # True acknowledgments - confirmations and agreements
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "yeah",
        "yep",
        "yup",
        "yea",
        "ya",
        "no",
        "nope",
        "nah",
        "na",
        "sure",
        # Gratitude expressions
        "thanks",
        "thank you",
        "thx",
        "ty",
        "np",
        "yw",
        # Understanding confirmations
        "alright",
        "aight",
        "sounds good",
        "got it",
        "gotcha",
        "heard",
        "bet",
        "word",
        # Status updates (not emotional reactions)
        "omw",
        "otw",
        "on my way",
        "be there soon",
        # NOTE: Removed emotional reactions (cool, nice, good, great, awesome, lol, lmao, haha, hehe)
        # These should trigger LLM generation, not canned acknowledgment responses
        "see you",
        "bye",
        "later",
        "ttyl",
        "cya",
        "gn",
        "gm",
        "idk",
        "idc",
        "tbh",
        "nvm",
        "ight",
    }
)

# Topic-shift markers that indicate a new conversation thread
TOPIC_SHIFT_MARKERS = frozenset(
    {
        "btw",
        "by the way",
        "anyway",
        "anyways",
        "also",
        "oh also",
        "speaking of",
        "on another note",
        "random but",
        "unrelated",
        "unrelated but",
        "separately",
        "side note",
        "quick question",
        "actually",
        "oh wait",
    }
)

# Emoji pattern for detecting emoji-only content
EMOJI_PATTERN = re.compile(
    r"^[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\s\u200d]+$"
)

# Common auto-signatures to strip
AUTO_SIGNATURE_PATTERNS = [
    r"\n--\s*\n.*$",  # Standard email signature separator
    r"\nSent from my iPhone.*$",
    r"\nSent from my iPad.*$",
    r"\nGet Outlook for iOS.*$",
    r"\nSent via.*$",
]
AUTO_SIGNATURE_REGEX = re.compile("|".join(AUTO_SIGNATURE_PATTERNS), re.IGNORECASE | re.DOTALL)

# Repeated emoji pattern (3+ of same emoji in a row)
REPEATED_EMOJI_PATTERN = re.compile(
    r"([\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])\1{2,}"
)

# Question words for detecting questions
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose"}

# Temporal reference patterns
TEMPORAL_PATTERNS = [
    r"\b(today|tomorrow|yesterday|tonight|later|soon)\b",
    r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(this|next|last)\s+(week|month|weekend)\b",
    r"\bin\s+(\d+)\s+(minutes?|hours?|days?)\b",
    r"\bat\s+(\d{1,2})\s*(am|pm|o'?clock)?\b",
]
TEMPORAL_REGEX = re.compile("|".join(TEMPORAL_PATTERNS), re.IGNORECASE)


def normalize_text(
    text: str,
    collapse_emojis: bool = True,
    strip_signatures: bool = True,
) -> str:
    """Canonical text cleaning used everywhere in the pipeline.

    This is the single source of truth for text normalization. All text
    should pass through this function before being stored or compared.

    Args:
        text: Raw text to normalize.
        collapse_emojis: If True, collapse repeated emojis (3+ same emoji -> 2).
        strip_signatures: If True, remove common auto-signatures.

    Returns:
        Normalized text. Returns empty string if text is a reaction or None.
    """
    if not text:
        return ""

    # 1. Check if this is a reaction (tapback) - return empty if so
    if REACTION_REGEX.match(text.strip()):
        return ""

    cleaned = text

    # 2. Strip auto-signatures
    if strip_signatures:
        cleaned = AUTO_SIGNATURE_REGEX.sub("", cleaned)

    # 3. Normalize whitespace (collapse multiple spaces, but preserve newlines)
    lines = cleaned.split("\n")
    normalized_lines = []
    for line in lines:
        # Collapse multiple spaces/tabs to single space
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines)

    # 4. Collapse repeated emojis (3+ same emoji -> 2)
    if collapse_emojis:
        cleaned = REPEATED_EMOJI_PATTERN.sub(r"\1\1", cleaned)

    # 5. Final strip
    cleaned = cleaned.strip()

    return cleaned


def is_reaction(text: str) -> bool:
    """Check if text is an iMessage reaction/tapback."""
    if not text:
        return False
    return bool(REACTION_REGEX.match(text.strip()))


def is_acknowledgment_only(text: str) -> bool:
    """Check if text is only a generic acknowledgment phrase.

    These are context-dependent and not useful as standalone responses.
    """
    if not text:
        return False
    normalized = text.lower().strip()
    # Also check for just punctuation added
    stripped = re.sub(r"[.!?,;]+$", "", normalized)
    return stripped in ACKNOWLEDGMENT_PHRASES or normalized in ACKNOWLEDGMENT_PHRASES


def is_emoji_only(text: str) -> bool:
    """Check if text contains only emojis and whitespace."""
    if not text:
        return False
    return bool(EMOJI_PATTERN.match(text.strip()))


def starts_new_topic(text: str) -> bool:
    """Check if text starts with a topic-shift marker."""
    if not text:
        return False
    normalized = text.lower().strip()
    for marker in TOPIC_SHIFT_MARKERS:
        if normalized.startswith(marker):
            return True
    return False


def is_question(text: str) -> bool:
    """Check if text is a question.

    Uses both punctuation and question word detection.
    """
    if not text:
        return False
    stripped = text.strip()
    # Ends with question mark
    if stripped.endswith("?"):
        return True
    # Starts with question word
    first_word = stripped.split()[0].lower() if stripped.split() else ""
    return first_word in QUESTION_WORDS


def extract_temporal_refs(text: str) -> list[str]:
    """Extract temporal references from text."""
    if not text:
        return []
    matches = TEMPORAL_REGEX.findall(text)
    # Flatten matches (some patterns have groups)
    refs = []
    for match in matches:
        if isinstance(match, tuple):
            refs.extend([m for m in match if m])
        elif match:
            refs.append(match)
    return refs


def get_attachment_token(attachment_type: str | None) -> str:
    """Get a token representation for an attachment type.

    Args:
        attachment_type: MIME type or simple type like "image", "video", etc.

    Returns:
        Token like "<ATTACHMENT:image>" or "<ATTACHMENT:file>".
    """
    if not attachment_type:
        return "<ATTACHMENT:file>"

    attachment_type = attachment_type.lower()
    if "image" in attachment_type or attachment_type in ("jpg", "jpeg", "png", "gif", "heic"):
        return "<ATTACHMENT:image>"
    elif "video" in attachment_type or attachment_type in ("mov", "mp4", "m4v"):
        return "<ATTACHMENT:video>"
    elif "audio" in attachment_type or attachment_type in ("mp3", "m4a", "aac", "wav"):
        return "<ATTACHMENT:audio>"
    elif "pdf" in attachment_type:
        return "<ATTACHMENT:pdf>"
    else:
        return "<ATTACHMENT:file>"


@dataclass
class TextFeatures:
    """Features extracted from text for gating decisions."""

    is_reaction: bool = False
    is_emoji_only: bool = False
    is_acknowledgment: bool = False
    has_attachment_token: bool = False
    word_count: int = 0
    char_count: int = 0
    starts_new_topic: bool = False
    is_question: bool = False
    temporal_refs: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_reaction": self.is_reaction,
            "is_emoji_only": self.is_emoji_only,
            "is_acknowledgment": self.is_acknowledgment,
            "has_attachment_token": self.has_attachment_token,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "starts_new_topic": self.starts_new_topic,
            "is_question": self.is_question,
            "temporal_refs": self.temporal_refs or [],
        }


def extract_text_features(text: str) -> TextFeatures:
    """Extract features from text for gating decisions.

    Args:
        text: Text to analyze (should be normalized first).

    Returns:
        TextFeatures dataclass with extracted features.
    """
    if not text:
        return TextFeatures()

    return TextFeatures(
        is_reaction=is_reaction(text),
        is_emoji_only=is_emoji_only(text),
        is_acknowledgment=is_acknowledgment_only(text),
        has_attachment_token="<ATTACHMENT:" in text,
        word_count=len(text.split()),
        char_count=len(text),
        starts_new_topic=starts_new_topic(text),
        is_question=is_question(text),
        temporal_refs=extract_temporal_refs(text),
    )


def trigger_expects_content(text: str) -> bool:
    """Check if a trigger expects a content-rich response.

    Triggers that are questions, requests, or proposals expect more than
    just an acknowledgment in response.

    Args:
        text: Trigger text to analyze.

    Returns:
        True if trigger expects a substantive response.
    """
    if not text:
        return False

    normalized = text.lower().strip()

    # Questions expect content
    if is_question(text):
        return True

    # Request patterns
    request_patterns = [
        r"\bcan you\b",
        r"\bcould you\b",
        r"\bwould you\b",
        r"\bwill you\b",
        r"\bdo you want\b",
        r"\bwanna\b",
        r"\blet me know\b",
        r"\btell me\b",
        r"\bwhat do you think\b",
        r"\bthoughts\?\s*$",
    ]
    for pattern in request_patterns:
        if re.search(pattern, normalized):
            return True

    # Proposal patterns (expecting yes/no + elaboration)
    proposal_patterns = [
        r"\bhow about\b",
        r"\bwhat about\b",
        r"\bshould we\b",
        r"\bshall we\b",
        r"\blet's\b",
        r"\bwe could\b",
    ]
    for pattern in proposal_patterns:
        if re.search(pattern, normalized):
            return True

    return False
