"""User texting style analyzer for JARVIS v2.

Analyzes the user's messaging patterns to generate style-matched replies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class UserStyle:
    """Detected texting style patterns."""

    avg_word_count: float = 8.0
    avg_char_count: float = 40.0
    uses_emoji: bool = False
    emoji_frequency: float = 0.0
    capitalization: str = "normal"  # "lowercase" | "normal" | "all_caps"
    uses_abbreviations: bool = False
    punctuation_style: str = "normal"  # "minimal" | "normal" | "expressive"
    common_phrases: list[str] = field(default_factory=list)
    enthusiasm_level: str = "medium"  # "high" | "medium" | "low"


# Common texting abbreviations
ABBREVIATIONS = {
    "u", "ur", "r", "lol", "lmao", "omg", "idk", "tbh", "ngl", "rn",
    "bc", "w", "b4", "2", "4", "thx", "ty", "np", "pls", "plz",
    "gonna", "wanna", "gotta", "kinda", "sorta",
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
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


class StyleAnalyzer:
    """Analyzes user's texting style from their sent messages."""

    def analyze(self, messages: list[dict]) -> UserStyle:
        """Analyze user's texting style.

        Args:
            messages: List of user's sent messages
                     [{"text": "...", "timestamp": ...}, ...]

        Returns:
            UserStyle with detected patterns
        """
        texts = [m.get("text", "") for m in messages if m.get("text")]

        if not texts:
            return UserStyle()

        # Word and character counts
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]

        avg_words = sum(word_counts) / len(word_counts)
        avg_chars = sum(char_counts) / len(char_counts)

        # Emoji usage
        emoji_messages = [t for t in texts if EMOJI_PATTERN.search(t)]
        emoji_freq = len(emoji_messages) / len(texts)

        # Capitalization
        lowercase_count = sum(1 for t in texts if t == t.lower())
        caps_count = sum(1 for t in texts if t == t.upper() and len(t) > 2)

        if lowercase_count / len(texts) > 0.7:
            cap_style = "lowercase"
        elif caps_count / len(texts) > 0.3:
            cap_style = "all_caps"
        else:
            cap_style = "normal"

        # Abbreviations
        uses_abbrevs = any(
            any(word.lower() in ABBREVIATIONS for word in t.split())
            for t in texts
        )

        # Punctuation style
        punct_style = self._detect_punctuation_style(texts)

        # Common phrases
        common = self._extract_common_phrases(texts)

        # Enthusiasm level
        enthusiasm = self._detect_enthusiasm(texts)

        return UserStyle(
            avg_word_count=avg_words,
            avg_char_count=avg_chars,
            uses_emoji=len(emoji_messages) > 0,
            emoji_frequency=emoji_freq,
            capitalization=cap_style,
            uses_abbreviations=uses_abbrevs,
            punctuation_style=punct_style,
            common_phrases=common[:5],
            enthusiasm_level=enthusiasm,
        )

    def _detect_punctuation_style(self, texts: list[str]) -> str:
        """Detect punctuation usage patterns."""
        exclaim_count = sum(t.count("!") for t in texts)
        question_count = sum(t.count("?") for t in texts)
        period_count = sum(t.count(".") for t in texts)
        total_messages = len(texts)

        # Multiple exclamation marks or frequent use
        if exclaim_count / total_messages > 1.5:
            return "expressive"
        # Few periods or ending punctuation
        elif (period_count + exclaim_count + question_count) / total_messages < 0.5:
            return "minimal"
        return "normal"

    def _extract_common_phrases(self, texts: list[str]) -> list[str]:
        """Extract commonly used phrases."""
        # Count phrase occurrences
        phrase_counts: dict[str, int] = {}

        common_starters = [
            "sounds good", "let me", "i'll", "gonna", "want to",
            "can you", "do you", "are you", "that's", "haha",
            "lol", "omg", "oh nice", "oh cool", "no worries",
        ]

        for text in texts:
            text_lower = text.lower()
            for phrase in common_starters:
                if phrase in text_lower:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Sort by frequency
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases]

    def _detect_enthusiasm(self, texts: list[str]) -> str:
        """Detect overall enthusiasm level."""
        total = len(texts)
        if total == 0:
            return "medium"

        # Count enthusiasm indicators
        exclaim_messages = sum(1 for t in texts if "!" in t)
        caps_messages = sum(1 for t in texts if any(w.isupper() and len(w) > 2 for w in t.split()))
        emoji_messages = sum(1 for t in texts if EMOJI_PATTERN.search(t))

        enthusiasm_score = (
            (exclaim_messages / total) * 0.4 +
            (caps_messages / total) * 0.3 +
            (emoji_messages / total) * 0.3
        )

        if enthusiasm_score > 0.4:
            return "high"
        elif enthusiasm_score < 0.15:
            return "low"
        return "medium"

    def to_prompt_instructions(self, style: UserStyle) -> str:
        """Convert style analysis to prompt instructions.

        Args:
            style: Analyzed user style

        Returns:
            String instructions for the LLM
        """
        instructions = []

        # Length guidance
        if style.avg_word_count < 6:
            instructions.append("Keep replies very short (under 6 words)")
        elif style.avg_word_count < 12:
            instructions.append("Keep replies brief (under 12 words)")
        else:
            instructions.append("Medium length replies okay (under 20 words)")

        # Capitalization
        if style.capitalization == "lowercase":
            instructions.append("Use lowercase (no capitals)")
        elif style.capitalization == "all_caps":
            instructions.append("Can use caps for emphasis")

        # Emoji
        if style.uses_emoji and style.emoji_frequency > 0.3:
            instructions.append("Use emojis occasionally")
        elif style.uses_emoji and style.emoji_frequency > 0.1:
            instructions.append("Can use 1 emoji if appropriate")
        else:
            instructions.append("Don't use emojis")

        # Abbreviations
        if style.uses_abbreviations:
            instructions.append("Casual abbreviations okay (u, ur, lol, etc.)")

        # Punctuation
        if style.punctuation_style == "expressive":
            instructions.append("Can use multiple exclamation marks!")
        elif style.punctuation_style == "minimal":
            instructions.append("Minimal punctuation, no periods needed")

        # Enthusiasm
        if style.enthusiasm_level == "high":
            instructions.append("Be enthusiastic and energetic")
        elif style.enthusiasm_level == "low":
            instructions.append("Keep tone calm and understated")

        return "\n".join(f"- {i}" for i in instructions)
