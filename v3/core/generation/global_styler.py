"""Global user style analyzer for JARVIS v2.

Analyzes ALL your messages across ALL conversations to build a comprehensive
understanding of your texting personality. This is different from per-contact
profiles which focus on relationship-specific patterns.

Key insight: Your texting style is mostly consistent across contacts (with minor
adjustments for formality). This module extracts your "voice" from aggregate data.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from core.embeddings.store import get_embedding_store

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".jarvis" / "global_style_cache.db"

# Minimum messages needed for meaningful analysis
MIN_MESSAGES_FOR_ANALYSIS = 50

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

# Common greeting starters
GREETING_STARTERS = {"hey", "hi", "hello", "yo", "sup", "what's up", "whats up"}

# Affirmative/negative patterns for short messages
AFFIRMATIVE_PATTERNS = {
    "yes", "yeah", "yep", "ya", "yea", "sure", "ok", "okay",
    "k", "sounds good", "for sure", "down", "bet", "cool", "def", "definitely",
}
NEGATIVE_PATTERNS = {
    "no", "nah", "nope", "can't", "cant", "sorry", "not really",
    "maybe later", "probably not", "won't", "wont",
}

# Stop words for phrase extraction
STOP_PHRASE_WORDS = {
    "i", "you", "we", "they", "he", "she", "it", "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "at", "by", "up", "out",
    "if", "so", "or", "as", "but", "and", "can", "do", "did",
    "my", "your", "our", "his", "her", "its", "their",
    "this", "that", "what", "when", "where", "who", "how", "why",
    "not", "just", "get", "got", "have", "has", "had", "will",
    "would", "should", "could", "been", "being", "with", "from",
}


@dataclass
class GlobalUserStyle:
    """Your overall texting style across ALL conversations."""

    # Phrase patterns (from ALL your messages)
    common_phrases: list[str] = field(default_factory=list)  # ["sounds good", "for sure"]
    greeting_phrases: list[str] = field(default_factory=list)  # ["hey", "yo"]
    signoff_phrases: list[str] = field(default_factory=list)  # ["later", "ttyl"]
    affirmative_phrases: list[str] = field(default_factory=list)  # ["yeah", "for sure"]
    negative_phrases: list[str] = field(default_factory=list)  # ["nah", "can't"]

    # Quantitative patterns
    avg_message_length: float = 0.0  # Average character count
    avg_word_count: float = 0.0  # Average words per message
    emoji_frequency: float = 0.0  # % of messages with emoji
    punctuation_style: str = "normal"  # "minimal" | "normal" | "expressive"
    capitalization: str = "normal"  # "lowercase" | "normal" | "mixed"

    # Qualitative
    uses_abbreviations: bool = False  # lol, brb, idk, etc.
    enthusiasm_level: str = "medium"  # "high" | "medium" | "low"

    # Metadata
    total_messages_analyzed: int = 0
    computed_at: datetime | None = None

    # LLM-generated personality description
    personality_summary: str = ""
    # Example: "You're a casual texter who rarely capitalizes and almost never
    # uses periods. You love abbreviations like 'lol' and 'sm'. You tend to be
    # brief but enthusiastic, often using '!' when excited."

    # Extracted facts about you
    facts: list[str] = field(default_factory=list)  # ["Works in tech", "Has a dog"]
    interests: list[str] = field(default_factory=list)  # ["hiking", "coffee"]
    mentioned_people: list[str] = field(default_factory=list)  # ["mom", "Sarah"]
    mentioned_places: list[str] = field(default_factory=list)  # ["SF", "Blue Bottle"]


class GlobalStyleCache:
    """Persistent SQLite cache for global user style."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_CACHE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_style_cache (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    style_json TEXT NOT NULL,
                    computed_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL
                )
            """)
            conn.commit()

    def get(
        self,
        current_message_count: int,
        max_age_hours: int = 168,  # 1 week
    ) -> GlobalUserStyle | None:
        """Get cached style if valid.

        Invalidates if:
        - Age > max_age_hours
        - message_count changed significantly (>5% growth)
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT style_json, computed_at, message_count FROM global_style_cache WHERE id = 1"
            ).fetchone()

        if not row:
            return None

        # Check staleness by age
        age_hours = (time.time() - row["computed_at"]) / 3600
        if age_hours > max_age_hours:
            logger.debug(f"Global style cache stale (age: {age_hours:.1f}h)")
            return None

        # Check if message count grew significantly (>5%)
        cached_count = row["message_count"]
        if cached_count > 0:
            growth = (current_message_count - cached_count) / cached_count
            if growth > 0.05:
                logger.debug(
                    f"Global style cache invalid (count growth: {growth:.1%})"
                )
                return None

        # Deserialize
        try:
            data = json.loads(row["style_json"])
            if data.get("computed_at"):
                data["computed_at"] = datetime.fromtimestamp(data["computed_at"])
            return GlobalUserStyle(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to deserialize global style cache: {e}")
            return None

    def set(self, style: GlobalUserStyle, message_count: int) -> None:
        """Store style in cache."""
        data = asdict(style)
        if data.get("computed_at"):
            data["computed_at"] = data["computed_at"].timestamp()

        style_json = json.dumps(data)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO global_style_cache
                (id, style_json, computed_at, message_count)
                VALUES (1, ?, ?, ?)""",
                (style_json, int(time.time()), message_count),
            )
            conn.commit()
        logger.debug(f"Cached global style ({message_count} messages)")

    def invalidate(self) -> None:
        """Clear the cache."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM global_style_cache")
            conn.commit()


@lru_cache(maxsize=1)
def get_global_style_cache() -> GlobalStyleCache:
    """Get singleton global style cache."""
    return GlobalStyleCache()


class GlobalUserStyler:
    """Analyzes ALL your messages to build a global style profile."""

    # Common abbreviations to detect
    ABBREVIATIONS = {
        "u", "ur", "r", "lol", "lmao", "omg", "idk", "tbh", "ngl", "rn",
        "bc", "w", "b4", "2", "4", "thx", "ty", "np", "pls", "plz",
        "gonna", "wanna", "gotta", "kinda", "sorta", "sm", "imo", "imho",
        "brb", "btw", "gtg", "hbu", "wbu", "ily", "jk", "ikr", "nvm",
    }

    def __init__(self):
        self.store = get_embedding_store()

    def _get_total_message_count(self) -> int:
        """Get total count of user's messages."""
        with self.store._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM message_embeddings WHERE is_from_me = 1"
            ).fetchone()
        return row["count"] if row else 0

    def _get_all_user_messages(self) -> list[str]:
        """Query ALL messages where is_from_me=1."""
        with self.store._get_connection() as conn:
            rows = conn.execute(
                "SELECT text_preview FROM message_embeddings WHERE is_from_me = 1"
            ).fetchall()
        return [row["text_preview"] for row in rows if row["text_preview"]]

    def build_global_style(self, skip_llm: bool = False) -> GlobalUserStyle:
        """Build style from ALL your sent messages.

        Args:
            skip_llm: If True, skip LLM calls for personality/facts (faster for testing)

        Returns:
            GlobalUserStyle with detected patterns
        """
        start_time = time.time()

        # 1. Query all your messages
        all_texts = self._get_all_user_messages()
        logger.info(f"Building global style from {len(all_texts)} messages")

        if len(all_texts) < MIN_MESSAGES_FOR_ANALYSIS:
            logger.warning(
                f"Not enough messages for global style "
                f"({len(all_texts)} < {MIN_MESSAGES_FOR_ANALYSIS})"
            )
            return GlobalUserStyle(total_messages_analyzed=len(all_texts))

        # 2. Extract phrase patterns
        common = self._extract_common_phrases(all_texts, min_count=5, top_n=10)
        greetings = self._extract_greeting_phrases(all_texts)
        signoffs = self._extract_signoff_phrases(all_texts)
        affirmatives = self._extract_affirmative_phrases(all_texts)
        negatives = self._extract_negative_phrases(all_texts)

        # 3. Compute quantitative metrics
        metrics = self._compute_metrics(all_texts)

        # 4. Generate LLM personality summary (optional)
        personality_summary = ""
        facts_data = {"facts": [], "interests": [], "people": [], "places": []}

        if not skip_llm:
            try:
                personality_summary = self._generate_personality_summary(all_texts, {
                    "avg_length": metrics["avg_length"],
                    "emoji_pct": metrics["emoji_frequency"] * 100,
                    "capitalization": metrics["capitalization"],
                    "punctuation": metrics["punctuation_style"],
                    "common_phrases": common,
                })
            except Exception as e:
                logger.warning(f"Failed to generate personality summary: {e}")

            # 5. Extract facts and interests about the user
            try:
                facts_data = self._extract_facts_and_interests(all_texts)
            except Exception as e:
                logger.warning(f"Failed to extract facts: {e}")

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Global style built in {elapsed:.0f}ms")

        return GlobalUserStyle(
            common_phrases=common,
            greeting_phrases=greetings,
            signoff_phrases=signoffs,
            affirmative_phrases=affirmatives,
            negative_phrases=negatives,
            avg_message_length=metrics["avg_length"],
            avg_word_count=metrics["avg_words"],
            emoji_frequency=metrics["emoji_frequency"],
            punctuation_style=metrics["punctuation_style"],
            capitalization=metrics["capitalization"],
            uses_abbreviations=metrics["uses_abbreviations"],
            enthusiasm_level=metrics["enthusiasm"],
            total_messages_analyzed=len(all_texts),
            computed_at=datetime.now(),
            personality_summary=personality_summary,
            facts=facts_data["facts"],
            interests=facts_data["interests"],
            mentioned_people=facts_data["people"],
            mentioned_places=facts_data["places"],
        )

    def _compute_metrics(self, texts: list[str]) -> dict:
        """Compute quantitative metrics from messages."""
        if not texts:
            return {
                "avg_length": 0,
                "avg_words": 0,
                "emoji_frequency": 0,
                "punctuation_style": "normal",
                "capitalization": "normal",
                "uses_abbreviations": False,
                "enthusiasm": "medium",
            }

        # Character and word counts
        char_counts = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        avg_length = sum(char_counts) / len(char_counts)
        avg_words = sum(word_counts) / len(word_counts)

        # Emoji usage
        emoji_count = sum(1 for t in texts if EMOJI_PATTERN.search(t))
        emoji_frequency = emoji_count / len(texts)

        # Capitalization style
        lowercase_count = sum(1 for t in texts if t == t.lower())
        caps_count = sum(1 for t in texts if t == t.upper() and len(t) > 2)

        if lowercase_count / len(texts) > 0.7:
            capitalization = "lowercase"
        elif caps_count / len(texts) > 0.2:
            capitalization = "mixed"
        else:
            capitalization = "normal"

        # Punctuation style
        exclaim_count = sum(t.count("!") for t in texts)
        period_count = sum(t.count(".") for t in texts)
        question_count = sum(t.count("?") for t in texts)

        punct_per_msg = (exclaim_count + period_count + question_count) / len(texts)
        exclaim_rate = exclaim_count / len(texts)

        if exclaim_rate > 1.0:
            punctuation_style = "expressive"
        elif punct_per_msg < 0.3:
            punctuation_style = "minimal"
        else:
            punctuation_style = "normal"

        # Abbreviations
        abbrev_count = 0
        for text in texts:
            words = text.lower().split()
            abbrev_count += sum(1 for w in words if w in self.ABBREVIATIONS)
        uses_abbreviations = abbrev_count > len(texts) * 0.1

        # Enthusiasm level
        enthusiasm_indicators = emoji_count + exclaim_count
        enthusiasm_rate = enthusiasm_indicators / len(texts)

        if enthusiasm_rate > 0.5:
            enthusiasm = "high"
        elif enthusiasm_rate < 0.15:
            enthusiasm = "low"
        else:
            enthusiasm = "medium"

        return {
            "avg_length": avg_length,
            "avg_words": avg_words,
            "emoji_frequency": emoji_frequency,
            "punctuation_style": punctuation_style,
            "capitalization": capitalization,
            "uses_abbreviations": uses_abbreviations,
            "enthusiasm": enthusiasm,
        }

    def _extract_common_phrases(
        self, texts: list[str], min_count: int = 5, top_n: int = 10
    ) -> list[str]:
        """Extract commonly used phrases (2-3 word n-grams)."""
        phrase_counter: Counter[str] = Counter()

        for text in texts:
            if not text or len(text) < 5:
                continue
            text_lower = text.lower()
            words = re.findall(r"\b\w+\b", text_lower)

            # Extract 2-word phrases
            for i in range(len(words) - 1):
                if words[i] in STOP_PHRASE_WORDS:
                    continue
                if words[i + 1] in STOP_PHRASE_WORDS:
                    continue
                phrase = f"{words[i]} {words[i + 1]}"
                if len(phrase) >= 5:
                    phrase_counter[phrase] += 1

            # Extract 3-word phrases
            for i in range(len(words) - 2):
                if words[i] in STOP_PHRASE_WORDS:
                    continue
                if words[i + 2] in STOP_PHRASE_WORDS:
                    continue
                phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                phrase_counter[phrase] += 1

        # Filter by min_count and return top N
        filtered = [(p, c) for p, c in phrase_counter.items() if c >= min_count]
        return [p for p, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]

    def _extract_greeting_phrases(self, texts: list[str]) -> list[str]:
        """Find how you typically start messages (greetings)."""
        greeting_counter: Counter[str] = Counter()

        for text in texts:
            if not text:
                continue
            first_word = text.lower().split()[0] if text.split() else ""
            for g in GREETING_STARTERS:
                if text.lower().startswith(g):
                    # Store the actual word (preserving potential emoji/variation)
                    greeting_counter[first_word] += 1
                    break

        return [g for g, _ in greeting_counter.most_common(5)]

    def _extract_signoff_phrases(self, texts: list[str]) -> list[str]:
        """Find how you typically end short messages (signoffs)."""
        signoff_patterns = {"later", "bye", "ttyl", "cya", "peace", "night", "gn"}
        signoff_counter: Counter[str] = Counter()

        for text in texts:
            if not text or len(text.split()) > 5:  # Only short messages
                continue
            text_lower = text.lower().strip()
            for pattern in signoff_patterns:
                if text_lower.endswith(pattern) or text_lower == pattern:
                    signoff_counter[text_lower] += 1
                    break

        return [s for s, _ in signoff_counter.most_common(5)]

    def _extract_affirmative_phrases(self, texts: list[str]) -> list[str]:
        """Find how you say yes."""
        counter: Counter[str] = Counter()

        for text in texts:
            if not text or len(text.split()) > 5:  # Short messages only
                continue
            text_lower = text.lower().strip()
            for pattern in AFFIRMATIVE_PATTERNS:
                if text_lower.startswith(pattern) or text_lower == pattern:
                    counter[text_lower] += 1
                    break

        return [p for p, _ in counter.most_common(5)]

    def _extract_negative_phrases(self, texts: list[str]) -> list[str]:
        """Find how you say no."""
        counter: Counter[str] = Counter()

        for text in texts:
            if not text or len(text.split()) > 5:  # Short messages only
                continue
            text_lower = text.lower().strip()
            for pattern in NEGATIVE_PATTERNS:
                if text_lower.startswith(pattern) or text_lower == pattern:
                    counter[text_lower] += 1
                    break

        return [p for p, _ in counter.most_common(5)]

    def _select_diverse_samples(self, texts: list[str], n: int = 50) -> list[str]:
        """Select diverse message samples for LLM analysis.

        Strategy: Mix of lengths, avoid duplicates, filter extremes.
        """
        # Filter out very short (<3 chars) and very long (>200 chars)
        valid = [t for t in texts if 3 <= len(t) <= 200]

        if len(valid) <= n:
            return valid

        # Bucket by length to get diversity
        short = [t for t in valid if len(t) < 20]
        medium = [t for t in valid if 20 <= len(t) < 80]
        long = [t for t in valid if len(t) >= 80]

        # Sample from each bucket
        samples = []
        samples.extend(random.sample(short, min(20, len(short))))
        samples.extend(random.sample(medium, min(20, len(medium))))
        samples.extend(random.sample(long, min(10, len(long))))

        return samples[:n]

    def _format_samples(self, samples: list[str]) -> str:
        """Format message samples for LLM prompt."""
        return "\n".join(f"- {s}" for s in samples)

    def _generate_personality_summary(self, texts: list[str], metrics: dict) -> str:
        """Use LLM to analyze messages and describe texting personality."""
        from core.models.loader import get_model_loader

        # 1. Sample messages (diverse selection)
        samples = self._select_diverse_samples(texts, n=50)

        # 2. Build analysis prompt
        prompt = f"""Analyze these text messages and describe their texting personality.

Messages (50 samples):
{self._format_samples(samples)}

Metrics:
- Average message length: {metrics['avg_length']:.0f} characters
- Uses emoji in {metrics['emoji_pct']:.0f}% of messages
- {metrics['capitalization']} capitalization
- {metrics['punctuation']} punctuation style
- Common phrases: {', '.join(metrics['common_phrases'][:5])}

Write a 2-3 sentence description of how this person texts. Include:
- Their communication style (casual/formal, brief/detailed)
- Any distinctive patterns (abbreviations, emoji use, punctuation quirks)
- Their tone (enthusiastic, laid-back, playful, serious)

Description:"""

        # 3. Generate with LLM
        loader = get_model_loader()
        result = loader.generate(prompt, max_tokens=150, temperature=0.3)

        return result.text.strip()

    def _extract_facts_and_interests(self, texts: list[str]) -> dict:
        """Use LLM to extract facts, interests, and entities from messages."""
        from core.models.loader import get_model_loader

        # Sample longer, more informative messages
        informative = [t for t in texts if 30 < len(t) < 300]
        if len(informative) < 20:
            return {"facts": [], "interests": [], "people": [], "places": []}

        samples = random.sample(informative, min(100, len(informative)))

        prompt = f"""Analyze these text messages and extract information about the sender.

Messages (samples):
{self._format_samples(samples)}

Extract and list:

FACTS (things that are definitely true about this person):
-

INTERESTS (topics/activities they seem interested in):
-

PEOPLE MENTIONED (names or relationships like "mom", "boss"):
-

PLACES MENTIONED (locations they frequent):
-

Be conservative - only include things clearly supported by the messages."""

        loader = get_model_loader()
        result = loader.generate(prompt, max_tokens=300, temperature=0.2)

        # Parse the structured output
        return self._parse_facts_output(result.text)

    def _parse_facts_output(self, text: str) -> dict:
        """Parse LLM output into structured facts."""
        sections: dict[str, list[str]] = {
            "facts": [],
            "interests": [],
            "people": [],
            "places": [],
        }

        current_section: str | None = None
        for line in text.strip().split("\n"):
            line = line.strip()
            line_upper = line.upper()

            if "FACTS" in line_upper:
                current_section = "facts"
            elif "INTERESTS" in line_upper:
                current_section = "interests"
            elif "PEOPLE" in line_upper:
                current_section = "people"
            elif "PLACES" in line_upper:
                current_section = "places"
            elif line.startswith("-") and current_section:
                item = line[1:].strip()
                if item and len(item) > 1:
                    sections[current_section].append(item)

        return sections


def get_global_user_style(use_cache: bool = True, skip_llm: bool = False) -> GlobalUserStyle | None:
    """Get global user style, building if needed.

    Returns None if not enough messages to analyze (<50).

    Args:
        use_cache: If True, use persistent cache
        skip_llm: If True, skip LLM calls (faster for testing)

    Returns:
        GlobalUserStyle or None if insufficient data
    """
    styler = GlobalUserStyler()

    # Get current message count
    total_count = styler._get_total_message_count()
    if total_count < MIN_MESSAGES_FOR_ANALYSIS:
        logger.debug(
            f"Not enough messages for global style ({total_count} < {MIN_MESSAGES_FOR_ANALYSIS})"
        )
        return None

    # Try cache first
    if use_cache:
        cache = get_global_style_cache()
        cached = cache.get(total_count)
        if cached:
            logger.debug("Using cached global style")
            return cached

    # Build fresh
    style = styler.build_global_style(skip_llm=skip_llm)

    # Cache result
    if use_cache and style.total_messages_analyzed >= MIN_MESSAGES_FOR_ANALYSIS:
        cache = get_global_style_cache()
        cache.set(style, total_count)

    return style
