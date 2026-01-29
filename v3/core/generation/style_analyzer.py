"""User texting style analyzer for JARVIS v2.

Analyzes the user's messaging patterns to generate style-matched replies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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

    # Personality dimensions
    formality_score: float = 0.5  # 0=very casual, 1=formal
    humor_style: str = "none"  # "none" | "dry" | "playful" | "expressive"
    response_tendency: str = "balanced"  # "brief" | "balanced" | "detailed"


# Common texting abbreviations
ABBREVIATIONS = {
    "u",
    "ur",
    "r",
    "lol",
    "lmao",
    "omg",
    "idk",
    "tbh",
    "ngl",
    "rn",
    "bc",
    "w",
    "b4",
    "2",
    "4",
    "thx",
    "ty",
    "np",
    "pls",
    "plz",
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


class StyleAnalyzer:
    """Analyzes user's texting style from their sent messages."""

    def analyze(self, messages: list[dict[str, Any]]) -> UserStyle:
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
        uses_abbrevs = any(any(word.lower() in ABBREVIATIONS for word in t.split()) for t in texts)

        # Punctuation style
        punct_style = self._detect_punctuation_style(texts)

        # Common phrases
        common = self._extract_common_phrases(texts)

        # Enthusiasm level
        enthusiasm = self._detect_enthusiasm(texts)

        # Personality dimensions
        formality = self._analyze_formality(texts)
        humor = self._detect_humor_style(texts)
        response_tendency = self._detect_response_tendency(texts)

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
            formality_score=formality,
            humor_style=humor,
            response_tendency=response_tendency,
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
            "sounds good",
            "let me",
            "i'll",
            "gonna",
            "want to",
            "can you",
            "do you",
            "are you",
            "that's",
            "haha",
            "lol",
            "omg",
            "oh nice",
            "oh cool",
            "no worries",
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
            (exclaim_messages / total) * 0.4
            + (caps_messages / total) * 0.3
            + (emoji_messages / total) * 0.3
        )

        if enthusiasm_score > 0.4:
            return "high"
        elif enthusiasm_score < 0.15:
            return "low"
        return "medium"

    def _analyze_formality(self, texts: list[str]) -> float:
        """Analyze formality level of messages.

        Returns:
            Score from 0 (very casual) to 1 (formal)
        """
        if not texts:
            return 0.5

        casual_count = 0
        formal_count = 0

        # Casual indicators
        casual_words = {
            "lol",
            "lmao",
            "haha",
            "hehe",
            "omg",
            "wtf",
            "idk",
            "tbh",
            "ngl",
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
            "ya",
            "yea",
            "yeah",
            "nah",
            "bruh",
            "bro",
            "dude",
            "yo",
            "sup",
            "k",
            "u",
            "ur",
            "r",
        }

        # Formal indicators
        formal_words = {
            "please",
            "thank",
            "appreciate",
            "regards",
            "sincerely",
            "hello",
            "greetings",
            "certainly",
            "absolutely",
            "however",
            "therefore",
            "regarding",
            "concerning",
            "additionally",
            "furthermore",
        }

        for text in texts:
            text_lower = text.lower()
            words = set(text_lower.split())

            casual_hits = len(words & casual_words)
            formal_hits = len(words & formal_words)

            # Also check for proper capitalization and punctuation
            has_proper_caps = text[0].isupper() if text else False
            has_period = text.endswith(".")

            if casual_hits > 0:
                casual_count += casual_hits
            if formal_hits > 0 or (has_proper_caps and has_period):
                formal_count += 1

        # Calculate score
        total_indicators = casual_count + formal_count
        if total_indicators == 0:
            return 0.5

        formality = formal_count / total_indicators
        return round(formality, 2)

    def _detect_humor_style(self, texts: list[str]) -> str:
        """Detect humor style from message patterns.

        Returns:
            "none" | "dry" | "playful" | "expressive"
        """
        if not texts:
            return "none"

        total = len(texts)

        # Count humor indicators
        lol_count = sum(1 for t in texts if "lol" in t.lower())
        haha_count = sum(1 for t in texts if "haha" in t.lower() or "hehe" in t.lower())
        lmao_count = sum(1 for t in texts if "lmao" in t.lower() or "😂" in t or "🤣" in t)

        # Playful language (teasing, sarcasm markers)
        playful_count = sum(
            1
            for t in texts
            if any(p in t.lower() for p in ["jk", "kidding", "lmao", ";)", "😏", "🙃"])
        )

        total_humor = lol_count + haha_count + lmao_count + playful_count
        humor_rate = total_humor / total if total > 0 else 0

        if humor_rate < 0.05:
            return "none"
        elif lmao_count > haha_count and playful_count > total * 0.1:
            return "expressive"
        elif playful_count > total * 0.05:
            return "playful"
        elif haha_count > lol_count:
            return "playful"
        else:
            return "dry"

    def _detect_response_tendency(self, texts: list[str]) -> str:
        """Detect if user tends toward brief, balanced, or detailed responses.

        Returns:
            "brief" | "balanced" | "detailed"
        """
        if not texts:
            return "balanced"

        word_counts = [len(t.split()) for t in texts]
        avg_words = sum(word_counts) / len(word_counts)

        if avg_words < 5:
            return "brief"
        elif avg_words > 15:
            return "detailed"
        return "balanced"

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

    def build_style_instructions(
        self,
        style: UserStyle,
        profile: Any = None,
        global_style: Any = None,
    ) -> str:
        """Build style instructions combining style analysis, contact profile, and global style.

        Priority order:
        1. Global style - your overall texting personality (most data)
        2. Contact profile - relationship-specific adjustments
        3. Style analysis - recent messages only (fallback)

        Args:
            style: Analyzed user style (from recent messages)
            profile: Optional ContactProfile (from all messages with this contact)
            global_style: Optional GlobalUserStyle (from ALL your messages)

        Returns:
            Comma-separated style instructions for the prompt
        """
        instructions = []

        # 1. Global style baseline (your overall texting personality)
        # Use EXPLICIT constraints - the model ignores vague instructions
        if global_style:
            if global_style.capitalization == "lowercase":
                instructions.append("lowercase only")
            if global_style.punctuation_style == "minimal":
                instructions.append("NO periods or exclamation marks")
            if global_style.uses_abbreviations:
                instructions.append("abbreviations okay (u, ur, idk, lol)")
            # Explicit word count limits
            if global_style.avg_word_count < 6:
                instructions.append("MAX 6 words")
            elif global_style.avg_word_count < 10:
                instructions.append("MAX 10 words")

        # 2. Relationship-based formality adjustment (from profile)
        if profile:
            rel_type = getattr(profile, "relationship_type", "unknown")
            if rel_type == "coworker":
                instructions.append("professional but friendly")
            elif rel_type == "close_friend":
                instructions.append("casual and relaxed")
            elif rel_type == "family":
                instructions.append("warm and familiar")
            # acquaintance, service, unknown - no special instruction

        # 3. Use contact profile data if available (more comprehensive)
        if profile and getattr(profile, "total_messages", 0) > 10:
            # Length from profile (only if not set by global style)
            if not global_style:
                avg_len = getattr(profile, "avg_your_length", 40)
                if avg_len < 20:
                    instructions.append("very short replies (under 5 words)")
                elif avg_len < 40:
                    instructions.append("brief replies (under 10 words)")
                else:
                    instructions.append("medium length replies okay")

            # Emoji usage from profile - only mention if allowed (avoid negative priming)
            if getattr(profile, "uses_emoji", False):
                instructions.append("emojis okay")

            # Tone from profile (skip if relationship already set tone)
            tone = getattr(profile, "tone", "")
            if tone == "playful" and getattr(profile, "is_playful", False):
                instructions.append("playful/teasing okay")
            elif tone == "formal" and not any("professional" in i for i in instructions):
                instructions.append("more formal tone")

            # Slang from profile
            if getattr(profile, "uses_slang", False):
                if not any("abbreviations" in i for i in instructions):
                    instructions.append("abbreviations okay (u, ur, sm, lol)")

        elif not global_style:
            # Fall back to detailed style-only instructions
            return self.to_prompt_instructions(style)

        # Add capitalization from style analysis (more granular) - only if not from global
        if not global_style:
            if style.capitalization == "lowercase":
                instructions.append("lowercase only")
            elif style.punctuation_style == "minimal":
                instructions.append("minimal punctuation")

        # Add humor style if detected
        if style.humor_style == "playful":
            if not any("playful" in i for i in instructions):
                instructions.append("playful tone okay")
        elif style.humor_style == "expressive":
            instructions.append("expressive/enthusiastic")

        return ", ".join(instructions) if instructions else "casual and brief"
