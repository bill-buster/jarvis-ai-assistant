"""Contact profiler for JARVIS v2.

Builds rich profiles of contacts based on message history.
Answers: Who is this person? What do we talk about? How do we communicate?
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .store import get_embedding_store, SimilarMessage

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".jarvis" / "profile_cache.db"


class ProfileCache:
    """Persistent SQLite cache for contact profiles.

    Caches computed profiles to avoid re-analyzing conversations on every reply.
    Invalidates when message_count changes or profile is older than max_age_hours.
    """

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
                CREATE TABLE IF NOT EXISTS profile_cache (
                    chat_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    computed_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL
                )
            """)
            conn.commit()

    def get(
        self,
        chat_id: str,
        current_message_count: int,
        max_age_hours: int = 24,
    ) -> "ContactProfile | None":
        """Get cached profile if still valid.

        Args:
            chat_id: Conversation ID
            current_message_count: Current message count for this chat
            max_age_hours: Max age before invalidation

        Returns:
            Cached ContactProfile or None if cache miss/stale
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT profile_json, computed_at, message_count FROM profile_cache WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()

        if not row:
            return None

        # Check staleness
        age_hours = (time.time() - row["computed_at"]) / 3600
        if age_hours > max_age_hours:
            logger.debug(f"Profile cache stale for {chat_id} (age: {age_hours:.1f}h)")
            return None

        # Check message count changed
        if row["message_count"] != current_message_count:
            logger.debug(
                f"Profile cache invalid for {chat_id} (count: {row['message_count']} -> {current_message_count})"
            )
            return None

        # Deserialize profile
        try:
            data = json.loads(row["profile_json"])
            profile = _profile_from_dict(data)
            logger.debug(f"Profile cache hit for {chat_id}")
            return profile
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to deserialize cached profile: {e}")
            return None

    def set(self, chat_id: str, profile: "ContactProfile", message_count: int) -> None:
        """Store profile in cache.

        Args:
            chat_id: Conversation ID
            profile: Profile to cache
            message_count: Current message count (for invalidation)
        """
        # Serialize profile (convert datetimes to timestamps)
        data = _profile_to_dict(profile)
        profile_json = json.dumps(data)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profile_cache (chat_id, profile_json, computed_at, message_count)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, profile_json, int(time.time()), message_count),
            )
            conn.commit()
        logger.debug(f"Cached profile for {chat_id} ({message_count} messages)")

    def invalidate(self, chat_id: str) -> None:
        """Remove a profile from cache."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM profile_cache WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def clear(self) -> None:
        """Clear all cached profiles."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM profile_cache")
            conn.commit()


def _profile_to_dict(profile: "ContactProfile") -> dict:
    """Convert ContactProfile to JSON-serializable dict."""
    data = asdict(profile)
    # Convert datetime fields to timestamps
    if data.get("last_message_date"):
        data["last_message_date"] = data["last_message_date"].timestamp()
    if data.get("first_message_date"):
        data["first_message_date"] = data["first_message_date"].timestamp()
    return data


def _profile_from_dict(data: dict) -> "ContactProfile":
    """Reconstruct ContactProfile from dict."""
    # Convert timestamps back to datetimes
    if data.get("last_message_date"):
        data["last_message_date"] = datetime.fromtimestamp(data["last_message_date"])
    if data.get("first_message_date"):
        data["first_message_date"] = datetime.fromtimestamp(data["first_message_date"])
    # Reconstruct topic clusters
    if data.get("topics"):
        data["topics"] = [TopicCluster(**t) for t in data["topics"]]
    return ContactProfile(**data)


# Singleton cache instance
_profile_cache: ProfileCache | None = None


def get_profile_cache() -> ProfileCache:
    """Get singleton profile cache."""
    global _profile_cache
    if _profile_cache is None:
        _profile_cache = ProfileCache()
    return _profile_cache


@dataclass
class TopicCluster:
    """A cluster of related messages representing a topic."""

    name: str  # Inferred topic name
    keywords: list[str]  # Key words in this cluster
    sample_messages: list[str]  # Example messages
    message_count: int
    percentage: float  # % of conversation about this topic


@dataclass
class ContactProfile:
    """Rich profile of a contact based on message history."""

    chat_id: str
    display_name: str | None

    # Relationship signals
    relationship_type: str  # "close_friend", "family", "coworker", "acquaintance", "service"
    relationship_confidence: float

    # Communication patterns
    total_messages: int
    you_sent: int
    they_sent: int
    avg_your_length: float
    avg_their_length: float

    # Tone analysis
    tone: str  # "casual", "playful", "formal", "professional", "mixed"
    uses_emoji: bool
    uses_slang: bool
    is_playful: bool  # Teasing, jokes, etc.

    # Topics
    topics: list[TopicCluster] = field(default_factory=list)

    # Time patterns
    most_active_hours: list[int] = field(default_factory=list)  # Hours of day
    avg_response_time_mins: float | None = None
    last_message_date: datetime | None = None
    first_message_date: datetime | None = None

    # Key phrases
    their_common_phrases: list[str] = field(default_factory=list)
    your_common_phrases: list[str] = field(default_factory=list)

    # Summary
    summary: str = ""  # Human-readable summary


class ContactProfiler:
    """Builds profiles of contacts from message history."""

    # Relationship indicators
    CLOSE_INDICATORS = {
        "love", "miss", "babe", "baby", "honey", "sweetie", "❤️", "😘", "🥰",
        "miss you", "love you", "cant wait", "can't wait",
    }

    FAMILY_INDICATORS = {
        "mom", "dad", "brother", "sister", "grandma", "grandpa", "aunt", "uncle",
        "son", "daughter", "cousin", "family",
    }

    WORK_INDICATORS = {
        "meeting", "deadline", "project", "client", "boss", "office", "work",
        "schedule", "call", "email", "report", "presentation", "team",
    }

    PLAYFUL_INDICATORS = {
        "lol", "lmao", "haha", "😂", "🤣", "jk", "kidding", "meany", "dumb",
        "stupid", "idiot", "nerd", "loser", "weirdo",  # Affectionate insults
    }

    SLANG_WORDS = {
        "gonna", "wanna", "gotta", "kinda", "sorta", "ya", "yea", "yeah", "nah",
        "ur", "u", "r", "ty", "thx", "pls", "plz", "idk", "idc", "tbh", "ngl",
        "fr", "rn", "omg", "omfg", "bruh", "bro", "dude", "yo", "sup", "k", "ok",
    }

    def __init__(self):
        self.store = get_embedding_store()

    def build_profile(
        self,
        chat_id: str,
        display_name: str | None = None,
        include_topics: bool = True,
    ) -> ContactProfile:
        """Build a comprehensive profile for a contact.

        Args:
            chat_id: Conversation ID
            display_name: Optional display name override
            include_topics: If True, extract topics via LLM (slower). Set False for style-only.

        Returns:
            ContactProfile with analysis results
        """
        # Get all messages for this conversation from embedding store
        messages = self._get_conversation_messages(chat_id)

        if not messages:
            return ContactProfile(
                chat_id=chat_id,
                display_name=display_name,
                relationship_type="unknown",
                relationship_confidence=0.0,
                total_messages=0,
                you_sent=0,
                they_sent=0,
                avg_your_length=0.0,
                avg_their_length=0.0,
                tone="unknown",
                uses_emoji=False,
                uses_slang=False,
                is_playful=False,
            )

        # Separate your messages and theirs
        your_messages = [m for m in messages if m.is_from_me]
        their_messages = [m for m in messages if not m.is_from_me]

        # Get display name from messages if not provided
        if not display_name and their_messages:
            display_name = their_messages[0].sender_name or their_messages[0].sender

        # Analyze patterns
        relationship_type, rel_confidence = self._infer_relationship(messages)
        tone = self._analyze_tone(messages)
        active_hours = self._get_active_hours(messages)

        your_texts = [m.text for m in your_messages if m.text]
        their_texts = [m.text for m in their_messages if m.text]

        # Extract recent conversation topics (single LLM call, ~1s latency)
        # Skip for style-only profile to speed up reply generation
        topics = self._extract_recent_topics(messages) if include_topics else []

        profile = ContactProfile(
            chat_id=chat_id,
            display_name=display_name,
            relationship_type=relationship_type,
            relationship_confidence=rel_confidence,
            total_messages=len(messages),
            you_sent=len(your_messages),
            they_sent=len(their_messages),
            avg_your_length=np.mean([len(t) for t in your_texts]) if your_texts else 0.0,
            avg_their_length=np.mean([len(t) for t in their_texts]) if their_texts else 0.0,
            tone=tone,
            uses_emoji=self._check_emoji_usage(messages),
            uses_slang=self._check_slang_usage(messages),
            is_playful=self._check_playfulness(messages),
            topics=topics,
            most_active_hours=active_hours,
            last_message_date=max(m.timestamp for m in messages) if messages else None,
            first_message_date=min(m.timestamp for m in messages) if messages else None,
            their_common_phrases=[],  # TODO: extract from clusters
            your_common_phrases=[],   # TODO: extract from clusters
        )

        # Generate summary
        profile.summary = self._generate_summary(profile)

        return profile

    def _get_conversation_messages(self, chat_id: str) -> list[SimilarMessage]:
        """Get all messages for a conversation from the store."""
        with self.store._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, chat_id, text_preview, sender, sender_name,
                       timestamp, is_from_me
                FROM message_embeddings
                WHERE chat_id = ?
                ORDER BY timestamp
                """,
                (chat_id,),
            ).fetchall()

        return [
            SimilarMessage(
                message_id=row["message_id"],
                chat_id=row["chat_id"],
                text=row["text_preview"] or "",
                sender=row["sender"],
                sender_name=row["sender_name"],
                timestamp=datetime.fromtimestamp(row["timestamp"]),
                is_from_me=bool(row["is_from_me"]),
                similarity=0.0,
            )
            for row in rows
        ]

    def _infer_relationship(
        self, messages: list[SimilarMessage]
    ) -> tuple[str, float]:
        """Infer the type of relationship from message content."""
        all_text = " ".join(m.text.lower() for m in messages if m.text)

        scores = {
            "close_friend": 0.0,
            "family": 0.0,
            "coworker": 0.0,
            "acquaintance": 0.0,
            "service": 0.0,
        }

        # Check for relationship indicators
        for word in self.CLOSE_INDICATORS:
            if word in all_text:
                scores["close_friend"] += 2

        for word in self.FAMILY_INDICATORS:
            if word in all_text:
                scores["family"] += 3

        for word in self.WORK_INDICATORS:
            if word in all_text:
                scores["coworker"] += 1.5

        # Message frequency and length patterns
        total = len(messages)
        if total < 20:
            scores["acquaintance"] += 2
            scores["service"] += 1
        elif total > 500:
            scores["close_friend"] += 3
            scores["family"] += 2

        # Playfulness suggests close relationship
        if self._check_playfulness(messages):
            scores["close_friend"] += 2

        # Find highest score
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        total_score = sum(scores.values()) or 1
        confidence = min(max_score / total_score, 0.95)

        # Default to acquaintance if no strong signals
        if max_score < 2:
            return "acquaintance", 0.3

        return max_type, confidence

    def _analyze_tone(self, messages: list[SimilarMessage]) -> str:
        """Analyze the overall tone of the conversation."""
        all_text = " ".join(m.text.lower() for m in messages if m.text)

        # Count indicators
        casual_count = sum(1 for word in self.SLANG_WORDS if word in all_text.split())
        playful_count = sum(1 for word in self.PLAYFUL_INDICATORS if word in all_text)
        formal_count = len(re.findall(r'\b(please|thank you|regards|sincerely)\b', all_text))

        # Calculate average message length
        avg_length = np.mean([len(m.text) for m in messages if m.text])

        if playful_count > 5:
            return "playful"
        elif casual_count > 10 and avg_length < 50:
            return "casual"
        elif formal_count > 3 or avg_length > 100:
            return "formal"
        else:
            return "casual"  # Default

    # Comprehensive stop words to filter out
    STOP_WORDS = {
        # Common words
        "that", "this", "with", "have", "will", "your", "from", "they",
        "been", "were", "being", "their", "would", "could", "should",
        "about", "which", "there", "what", "when", "make", "like",
        "just", "know", "take", "come", "think", "good", "some",
        "than", "then", "very", "after", "before", "going", "here",
        "also", "want", "need", "said", "says", "okay", "really",
        "thing", "things", "stuff", "right", "doing", "getting",
        "want", "wanted", "wants", "liked", "like", "likes",
        "gonna", "gotta", "wanna", "cant", "dont", "didnt", "doesnt",
        "thats", "youre", "theyre", "were", "its", "heres", "theres",
        "much", "many", "more", "most", "other", "another", "same",
        "into", "over", "under", "through", "back", "down", "still",
        "even", "well", "only", "because", "though", "actually",
        "probably", "maybe", "yeah", "yea", "yes", "sure", "thanks",
        "thank", "sorry", "please", "okay", "alright", "sounds",
        "something", "anything", "everything", "nothing", "someone",
        "anyone", "everyone", "people", "time", "today", "tomorrow",
        "tonight", "morning", "night", "week", "month", "year",
        # iMessage reaction artifacts - these show up as "Loved an image" etc.
        "loved", "liked", "emphasized", "laughed", "questioned", "disliked",
        "image", "message", "attachment",
        # Common verbs/fillers
        "think", "thought", "getting", "coming", "going", "looking",
        "making", "doing", "having", "being", "saying", "trying",
        "waiting", "working", "sending", "checking", "letting",
    }

    def _extract_topics_via_clustering(self, chat_id: str) -> list[TopicCluster]:
        """Extract topics by clustering message embeddings.

        Uses k-means on pre-computed embeddings to find semantic clusters,
        then labels each cluster by finding representative messages.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("sklearn not installed, skipping topic clustering")
            return []

        # Get embeddings from store
        embeddings, messages = self.store.get_chat_embeddings(chat_id)

        if len(messages) < 20:
            return []  # Not enough messages to cluster

        # Filter out reaction messages and very short messages
        valid_indices = []
        for i, msg in enumerate(messages):
            text = msg.text.lower() if msg.text else ""
            if len(text) < 10:
                continue
            if any(r in text for r in ["loved an", "liked an", "emphasized", "laughed at", "questioned"]):
                continue
            valid_indices.append(i)

        if len(valid_indices) < 20:
            return []

        # Filter embeddings and messages
        filtered_embeddings = embeddings[valid_indices]
        filtered_messages = [messages[i] for i in valid_indices]

        # Determine number of clusters (aim for ~50-100 messages per cluster)
        n_clusters = max(3, min(8, len(filtered_messages) // 75))

        # Cluster
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(filtered_embeddings)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

        # For each cluster, find representative messages and extract topic
        topics = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if len(cluster_indices) < 5:
                continue

            cluster_msgs = [filtered_messages[i] for i in cluster_indices]
            cluster_embeddings = filtered_embeddings[cluster_indices]

            # Find message closest to centroid (most representative)
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)
            representative_msg = cluster_msgs[closest_idx]

            # Extract topic label from representative message
            topic_name = self._extract_topic_label(cluster_msgs)
            if not topic_name:
                continue

            topics.append(TopicCluster(
                name=topic_name,
                keywords=[],  # Could extract keywords from cluster
                sample_messages=[representative_msg.text],
                message_count=len(cluster_msgs),
                percentage=round(len(cluster_msgs) / len(filtered_messages) * 100, 1),
            ))

        # Sort by size and return top 5
        topics.sort(key=lambda t: t.message_count, reverse=True)
        return topics[:5]

    def _extract_recent_topics(self, messages: list[SimilarMessage]) -> list[TopicCluster]:
        """Extract topics from recent messages using a single LLM call.

        Much faster than clustering + multiple LLM calls.
        """
        if len(messages) < 10:
            return []

        # Get recent messages (last 50, excluding very short ones)
        recent = sorted(messages, key=lambda m: m.timestamp, reverse=True)
        sample_texts = []
        for m in recent:
            if m.text and len(m.text) > 10:
                # Skip reaction messages
                text = m.text.lower()
                if any(r in text for r in ["loved an", "liked an", "emphasized", "laughed at"]):
                    continue
                sample_texts.append(m.text[:150])
            if len(sample_texts) >= 30:
                break

        if len(sample_texts) < 5:
            return []

        try:
            from core.models.loader import get_model_loader

            loader = get_model_loader()
            sample_str = "\n".join(f"- {t}" for t in sample_texts[:20])

            prompt = f"""List 3 main topics from these recent messages. Give only topic names (2-4 words each), one per line.

Messages:
{sample_str}

Topics (one per line, no numbers or bullets):"""

            result = loader.generate(prompt, max_tokens=50, temperature=0.1)
            lines = result.text.strip().split("\n")

            topics = []
            for line in lines[:3]:
                name = line.strip().strip("-").strip("•").strip("1234567890.)")
                name = name.strip().strip('"').strip("'")
                if name and len(name) > 2 and len(name) < 40:
                    topics.append(TopicCluster(
                        name=name.title(),
                        keywords=[],
                        sample_messages=[],
                        message_count=0,
                        percentage=0.0,
                    ))

            return topics

        except Exception as e:
            logger.warning(f"Failed to extract topics: {e}")
            return []

    def _extract_topic_label(self, messages: list[SimilarMessage]) -> str | None:
        """Extract a meaningful topic label from a cluster of messages using LLM.

        Summarizes the cluster content into a short, descriptive topic label.
        """
        # Get sample messages (most recent, not just closest to centroid)
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp, reverse=True)
        sample_texts = []
        for m in sorted_msgs[:15]:  # Use up to 15 recent messages
            if m.text and len(m.text) > 5:
                sample_texts.append(m.text[:200])  # Truncate long messages

        if len(sample_texts) < 3:
            return None

        # Use LLM to generate topic label
        try:
            from core.models.loader import get_model_loader

            loader = get_model_loader()
            sample_str = "\n".join(f"- {t}" for t in sample_texts[:10])

            prompt = f"""Given these messages from a conversation, provide a 2-4 word topic label describing what they're about.

Messages:
{sample_str}

Topic label (2-4 words only, no quotes, no explanation):"""

            result = loader.generate(prompt, max_tokens=20, temperature=0.1)
            topic = result.text.strip().strip('"').strip("'")

            # Validate - should be short and not contain weird chars
            if topic and len(topic) < 50 and not any(c in topic for c in ['[', ']', '{', '}']):
                return topic.title()
            return None
        except Exception as e:
            logger.warning(f"Failed to generate topic label: {e}")
            return self._fallback_topic_label(messages)

    def _fallback_topic_label(self, messages: list[SimilarMessage]) -> str | None:
        """Fallback topic extraction using word frequency if LLM fails."""
        all_text = " ".join(m.text.lower() for m in messages if m.text)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        filtered = [w for w in words if w not in self.STOP_WORDS]

        if not filtered:
            return None

        word_counts = Counter(filtered)
        top_word, count = word_counts.most_common(1)[0]

        if count < len(messages) * 0.1:
            return None

        return top_word.title()

    def _get_active_hours(self, messages: list[SimilarMessage]) -> list[int]:
        """Get the most active hours for this conversation."""
        hour_counts = Counter(m.timestamp.hour for m in messages)
        # Return top 3 most active hours
        return [hour for hour, _ in hour_counts.most_common(3)]

    def _check_emoji_usage(self, messages: list[SimilarMessage]) -> bool:
        """Check if emojis are commonly used."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )
        emoji_count = sum(1 for m in messages if m.text and emoji_pattern.search(m.text))
        return emoji_count > len(messages) * 0.1

    def _check_slang_usage(self, messages: list[SimilarMessage]) -> bool:
        """Check if slang/casual language is commonly used."""
        slang_count = 0
        for m in messages:
            if m.text:
                words = m.text.lower().split()
                slang_count += sum(1 for w in words if w in self.SLANG_WORDS)
        return slang_count > len(messages) * 0.15

    def _check_playfulness(self, messages: list[SimilarMessage]) -> bool:
        """Check if the conversation has playful/teasing tone."""
        playful_count = 0
        for m in messages:
            if m.text:
                text_lower = m.text.lower()
                playful_count += sum(1 for p in self.PLAYFUL_INDICATORS if p in text_lower)
        return playful_count > len(messages) * 0.05

    # Common phrase patterns to filter out
    STOP_PHRASE_WORDS = {
        "i", "you", "we", "they", "he", "she", "it", "a", "an", "the",
        "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "at", "by", "up", "out",
        "if", "so", "or", "as", "but", "and", "can", "do", "did",
        "my", "your", "our", "his", "her", "its", "their",
    }

    def _extract_phrases(self, texts: list[str], top_n: int = 5) -> list[str]:
        """Extract commonly used meaningful phrases."""
        # Look for 2-3 word phrases
        phrases = []
        for text in texts:
            if not text:
                continue
            words = text.lower().split()

            # Bigrams and trigrams
            for i in range(len(words) - 1):
                phrase_words = words[i:i+2]

                # Skip if first or last word is a stop word
                if phrase_words[0] in self.STOP_PHRASE_WORDS:
                    continue
                if phrase_words[-1] in self.STOP_PHRASE_WORDS:
                    continue

                phrase = " ".join(phrase_words)
                if len(phrase) > 8:  # Skip short phrases
                    phrases.append(phrase)

        # Get most common meaningful phrases
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(top_n) if count >= 3]

    def _generate_summary(self, profile: ContactProfile) -> str:
        """Generate a human-readable summary of the contact."""
        parts = []

        # Relationship
        rel_desc = {
            "close_friend": "Close friend",
            "family": "Family member",
            "coworker": "Work contact",
            "acquaintance": "Acquaintance",
            "service": "Service/Business",
            "unknown": "Contact",
        }
        parts.append(f"{rel_desc.get(profile.relationship_type, 'Contact')}")

        # Tone
        if profile.is_playful:
            parts.append("playful/teasing")
        elif profile.tone == "casual":
            parts.append("casual")
        elif profile.tone == "formal":
            parts.append("formal")

        # Volume
        if profile.total_messages > 1000:
            parts.append(f"{profile.total_messages:,} messages")
        elif profile.total_messages > 100:
            parts.append(f"{profile.total_messages} messages")

        # Topics from clustering
        if profile.topics:
            topic_names = [t.name.lower() for t in profile.topics[:3]]
            parts.append(f"topics: {', '.join(topic_names)}")

        return ", ".join(parts) + "."

    def get_relationship_trajectory(self, chat_id: str) -> dict | None:
        """Track how communication style has changed over time.

        Compares style from early messages vs recent messages to detect
        relationship evolution (e.g., becoming more casual over time).

        Args:
            chat_id: Conversation ID

        Returns:
            Dict with trajectory signals, or None if not enough history
        """
        messages = self._get_conversation_messages(chat_id)

        # Need at least 100 messages for meaningful comparison
        if len(messages) < 100:
            return None

        # Sort by timestamp
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
        early = sorted_msgs[:50]
        recent = sorted_msgs[-50:]

        # Analyze tone changes
        early_tone = self._analyze_tone(early)
        recent_tone = self._analyze_tone(recent)

        # Check emoji usage change
        early_emoji = self._check_emoji_usage(early)
        recent_emoji = self._check_emoji_usage(recent)

        # Check message length change
        early_your_msgs = [m for m in early if m.is_from_me and m.text]
        recent_your_msgs = [m for m in recent if m.is_from_me and m.text]

        early_avg_len = (
            sum(len(m.text) for m in early_your_msgs) / len(early_your_msgs)
            if early_your_msgs else 0
        )
        recent_avg_len = (
            sum(len(m.text) for m in recent_your_msgs) / len(recent_your_msgs)
            if recent_your_msgs else 0
        )

        return {
            "formality_change": early_tone != recent_tone,
            "early_tone": early_tone,
            "recent_tone": recent_tone,
            "emoji_increase": recent_emoji and not early_emoji,
            "emoji_decrease": early_emoji and not recent_emoji,
            "conversation_deepening": recent_avg_len > early_avg_len * 1.3,
            "conversation_shortening": recent_avg_len < early_avg_len * 0.7,
            "early_avg_length": round(early_avg_len, 1),
            "recent_avg_length": round(recent_avg_len, 1),
        }


# Convenience function
def get_contact_profile(
    chat_id: str,
    display_name: str | None = None,
    include_topics: bool = True,
    use_cache: bool = True,
) -> ContactProfile:
    """Get a contact profile for a conversation.

    Uses persistent cache to avoid re-computing profiles on every request.
    Cache invalidates when message count changes or profile is > 24 hours old.

    Args:
        chat_id: Conversation ID
        display_name: Optional display name
        include_topics: If False, skip LLM topic extraction (faster for style-only)
        use_cache: If True, use persistent cache (default True)
    """
    profiler = ContactProfiler()

    # Get current message count for cache validation
    store = get_embedding_store()
    with store._get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM message_embeddings WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        current_count = row["count"] if row else 0

    # Try cache first (only for style-only profiles to avoid stale topics)
    if use_cache and not include_topics:
        cache = get_profile_cache()
        cached = cache.get(chat_id, current_count)
        if cached is not None:
            return cached

    # Build fresh profile
    profile = profiler.build_profile(chat_id, display_name, include_topics=include_topics)

    # Cache the result (only style-only profiles - topics change frequently)
    if use_cache and not include_topics and profile.total_messages > 0:
        cache = get_profile_cache()
        cache.set(chat_id, profile, current_count)

    return profile
