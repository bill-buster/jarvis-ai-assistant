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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.utils import STOP_WORDS

from .store import SimilarMessage, get_embedding_store

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
            # Check if old schema exists (without include_topics)
            cursor = conn.execute("PRAGMA table_info(profile_cache)")
            columns = [row[1] for row in cursor.fetchall()]

            if columns and "include_topics" not in columns:
                # Old schema - drop and recreate
                logger.info("Migrating profile_cache to new schema")
                conn.execute("DROP TABLE IF EXISTS profile_cache")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS profile_cache (
                    chat_id TEXT NOT NULL,
                    include_topics INTEGER NOT NULL DEFAULT 0,
                    profile_json TEXT NOT NULL,
                    computed_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    PRIMARY KEY (chat_id, include_topics)
                )
            """)
            conn.commit()

    def get(
        self,
        chat_id: str,
        current_message_count: int,
        include_topics: bool = False,
        max_age_hours: int | None = None,
    ) -> ContactProfile | None:
        """Get cached profile if still valid.

        Args:
            chat_id: Conversation ID
            current_message_count: Current message count for this chat
            include_topics: Whether to get full profile (with topics) or style-only
            max_age_hours: Max age before invalidation (default: 24h for style, 6h for topics)

        Returns:
            Cached ContactProfile or None if cache miss/stale
        """
        # Default TTL: shorter for full profiles since topics may change
        if max_age_hours is None:
            max_age_hours = 6 if include_topics else 24

        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT profile_json, computed_at, message_count
                FROM profile_cache WHERE chat_id = ? AND include_topics = ?""",
                (chat_id, 1 if include_topics else 0),
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
            old_count = row["message_count"]
            logger.debug(
                f"Profile cache invalid for {chat_id} "
                f"(count: {old_count} -> {current_message_count})"
            )
            return None

        # Deserialize profile
        try:
            data = json.loads(row["profile_json"])
            profile = _profile_from_dict(data)
            logger.debug(f"Profile cache hit for {chat_id} (topics={include_topics})")
            return profile
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to deserialize cached profile: {e}")
            return None

    def set(
        self,
        chat_id: str,
        profile: ContactProfile,
        message_count: int,
        include_topics: bool = False,
    ) -> None:
        """Store profile in cache.

        Args:
            chat_id: Conversation ID
            profile: Profile to cache
            message_count: Current message count (for invalidation)
            include_topics: Whether this is a full profile (with topics) or style-only
        """
        # Serialize profile (convert datetimes to timestamps)
        data = _profile_to_dict(profile)
        profile_json = json.dumps(data)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO profile_cache
                (chat_id, include_topics, profile_json, computed_at, message_count)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    chat_id, 1 if include_topics else 0, profile_json,
                    int(time.time()), message_count
                ),
            )
            conn.commit()
        logger.debug(
            f"Cached profile for {chat_id} ({message_count} messages, topics={include_topics})"
        )

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


def _profile_to_dict(profile: ContactProfile) -> dict:
    """Convert ContactProfile to JSON-serializable dict."""
    data = asdict(profile)
    # Convert datetime fields to timestamps
    if data.get("last_message_date"):
        data["last_message_date"] = data["last_message_date"].timestamp()
    if data.get("first_message_date"):
        data["first_message_date"] = data["first_message_date"].timestamp()
    return data


def _profile_from_dict(data: dict) -> ContactProfile:
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


# Singleton cache instance using lru_cache for thread safety
@lru_cache(maxsize=1)
def get_profile_cache() -> ProfileCache:
    """Get singleton profile cache (thread-safe via lru_cache)."""
    return ProfileCache()


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
    summary: str = ""  # Human-readable summary (auto-generated from profile data)

    # LLM-generated relationship description
    relationship_summary: str = ""  # Natural language description of this relationship
    # Example: "You and Sarah have a playful, close friendship. You often tease each other
    # and share memes. Your conversations are casual and brief, with lots of 'lol' and emojis.
    # You frequently make plans to hang out and discuss work stress."


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
        # Get basic stats via SQL (avoids loading all messages for counts/averages)
        stats = self._get_basic_stats(chat_id)

        if not stats:
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

        # Get display name from stats if not provided
        if not display_name:
            display_name = stats["display_name"]

        # Now load messages for text analysis (pattern detection, phrases, topics)
        messages = self._get_conversation_messages(chat_id)

        # Analyze all patterns in a single pass
        analysis = self._analyze_all_patterns(messages)
        relationship_type = analysis["relationship_type"]
        rel_confidence = analysis["relationship_confidence"]
        tone = analysis["tone"]
        active_hours = analysis["active_hours"]

        # Separate texts for phrase extraction
        your_texts = [m.text for m in messages if m.is_from_me and m.text]
        their_texts = [m.text for m in messages if not m.is_from_me and m.text]

        # Extract recent conversation topics (single LLM call, ~1s latency)
        # Skip for style-only profile to speed up reply generation
        topics = self._extract_recent_topics(messages) if include_topics else []

        profile = ContactProfile(
            chat_id=chat_id,
            display_name=display_name,
            relationship_type=relationship_type,
            relationship_confidence=rel_confidence,
            total_messages=stats["total"],
            you_sent=stats["you_sent"],
            they_sent=stats["they_sent"],
            avg_your_length=stats["avg_your_length"],
            avg_their_length=stats["avg_their_length"],
            tone=tone,
            uses_emoji=analysis["uses_emoji"],
            uses_slang=analysis["uses_slang"],
            is_playful=analysis["is_playful"],
            topics=topics,
            most_active_hours=active_hours,
            last_message_date=(
                datetime.fromtimestamp(stats["last_timestamp"])
                if stats["last_timestamp"] else None
            ),
            first_message_date=(
                datetime.fromtimestamp(stats["first_timestamp"])
                if stats["first_timestamp"] else None
            ),
            their_common_phrases=self._extract_common_phrases(their_texts),
            your_common_phrases=self._extract_common_phrases(your_texts)
        )

        # Generate summary (auto-generated from profile data)
        profile.summary = self._generate_summary(profile)

        # Generate LLM relationship summary (only if include_topics is True - same cost tier)
        if include_topics and stats["total"] >= 20:
            try:
                profile.relationship_summary = self._generate_relationship_summary(
                    messages, profile, display_name
                )
            except Exception as e:
                logger.warning(f"Failed to generate relationship summary: {e}")

        return profile

    def _get_basic_stats(self, chat_id: str) -> dict | None:
        """Get basic profile stats via SQL aggregations (no message loading).

        Returns dict with: total, you_sent, they_sent, avg_your_length, avg_their_length,
        first_timestamp, last_timestamp, display_name
        """
        with self.store._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as you_sent,
                    SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as they_sent,
                    AVG(CASE WHEN is_from_me = 1 THEN LENGTH(text_preview) END) as avg_your_length,
                    AVG(CASE WHEN is_from_me = 0 THEN LENGTH(text_preview) END) as avg_their_length,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    MIN(CASE WHEN is_from_me = 0
                        THEN COALESCE(sender_name, sender) END) as display_name
                FROM message_embeddings
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()

        if not row or row["total"] == 0:
            return None

        return {
            "total": row["total"],
            "you_sent": row["you_sent"] or 0,
            "they_sent": row["they_sent"] or 0,
            "avg_your_length": row["avg_your_length"] or 0.0,
            "avg_their_length": row["avg_their_length"] or 0.0,
            "first_timestamp": row["first_timestamp"],
            "last_timestamp": row["last_timestamp"],
            "display_name": row["display_name"],
        }

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

    def _analyze_all_patterns(self, messages: list[SimilarMessage]) -> dict:
        """Analyze relationship, tone, and activity patterns in a single pass.

        Combines _infer_relationship(), _analyze_tone(), and _get_active_hours()
        to avoid iterating over messages 3+ times.

        Returns:
            Dict with keys: relationship_type, relationship_confidence, tone,
            active_hours, uses_emoji, uses_slang, is_playful
        """
        # Counters for single-pass aggregation
        relationship_scores = {
            "close_friend": 0.0,
            "family": 0.0,
            "coworker": 0.0,
            "acquaintance": 0.0,
            "service": 0.0,
        }
        casual_count = 0
        playful_count = 0
        formal_count = 0
        emoji_count = 0
        slang_count = 0
        total_length = 0
        text_count = 0
        hour_counts: Counter = Counter()

        # Compile patterns once
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )
        formal_pattern = re.compile(r'\b(please|thank you|regards|sincerely)\b')

        # Single pass over all messages
        for m in messages:
            # Track active hours
            hour_counts[m.timestamp.hour] += 1

            if not m.text:
                continue

            text = m.text
            text_lower = text.lower()
            text_count += 1
            total_length += len(text)

            # Check emoji
            if emoji_pattern.search(text):
                emoji_count += 1

            # Check relationship indicators (in combined text check)
            for word in self.CLOSE_INDICATORS:
                if word in text_lower:
                    relationship_scores["close_friend"] += 2

            for word in self.FAMILY_INDICATORS:
                if word in text_lower:
                    relationship_scores["family"] += 3

            for word in self.WORK_INDICATORS:
                if word in text_lower:
                    relationship_scores["coworker"] += 1.5

            # Check playful indicators
            for p in self.PLAYFUL_INDICATORS:
                if p in text_lower:
                    playful_count += 1

            # Check slang (need word boundaries)
            words = text_lower.split()
            for w in words:
                if w in self.SLANG_WORDS:
                    slang_count += 1
                    casual_count += 1

            # Check formal language
            formal_count += len(formal_pattern.findall(text_lower))

        total = len(messages)

        # Compute relationship
        if total < 20:
            relationship_scores["acquaintance"] += 2
            relationship_scores["service"] += 1
        elif total > 500:
            relationship_scores["close_friend"] += 3
            relationship_scores["family"] += 2

        # Playfulness suggests close relationship
        is_playful = playful_count > total * 0.05
        if is_playful:
            relationship_scores["close_friend"] += 2

        max_type = max(relationship_scores, key=relationship_scores.get)
        max_score = relationship_scores[max_type]
        total_score = sum(relationship_scores.values()) or 1
        rel_confidence = min(max_score / total_score, 0.95)

        if max_score < 2:
            relationship_type = "acquaintance"
            rel_confidence = 0.3
        else:
            relationship_type = max_type

        # Compute tone
        avg_length = total_length / text_count if text_count > 0 else 0

        if playful_count > 5:
            tone = "playful"
        elif casual_count > 10 and avg_length < 50:
            tone = "casual"
        elif formal_count > 3 or avg_length > 100:
            tone = "formal"
        else:
            tone = "casual"

        # Get top 3 active hours
        active_hours = [hour for hour, _ in hour_counts.most_common(3)]

        return {
            "relationship_type": relationship_type,
            "relationship_confidence": rel_confidence,
            "tone": tone,
            "active_hours": active_hours,
            "uses_emoji": emoji_count > total * 0.1,
            "uses_slang": slang_count > total * 0.15,
            "is_playful": is_playful,
        }

    def _infer_relationship(
        self, messages: list[SimilarMessage]
    ) -> tuple[str, float]:
        """Infer the type of relationship from message content.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
        all_text = " ".join(m.text.lower() for m in messages if m.text)

        scores = {
            "close_friend": 0.0,
            "family": 0.0,
            "coworker": 0.0,
            "acquaintance": 0.0,
            "service": 0.0,
        }

        for word in self.CLOSE_INDICATORS:
            if word in all_text:
                scores["close_friend"] += 2

        for word in self.FAMILY_INDICATORS:
            if word in all_text:
                scores["family"] += 3

        for word in self.WORK_INDICATORS:
            if word in all_text:
                scores["coworker"] += 1.5

        total = len(messages)
        if total < 20:
            scores["acquaintance"] += 2
            scores["service"] += 1
        elif total > 500:
            scores["close_friend"] += 3
            scores["family"] += 2

        if self._check_playfulness(messages):
            scores["close_friend"] += 2

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        total_score = sum(scores.values()) or 1
        confidence = min(max_score / total_score, 0.95)

        if max_score < 2:
            return "acquaintance", 0.3

        return max_type, confidence

    def _analyze_tone(self, messages: list[SimilarMessage]) -> str:
        """Analyze the overall tone of the conversation.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
        all_text = " ".join(m.text.lower() for m in messages if m.text)

        casual_count = sum(1 for word in self.SLANG_WORDS if word in all_text.split())
        playful_count = sum(1 for word in self.PLAYFUL_INDICATORS if word in all_text)
        formal_count = len(re.findall(r'\b(please|thank you|regards|sincerely)\b', all_text))

        avg_length = np.mean([len(m.text) for m in messages if m.text])

        if playful_count > 5:
            return "playful"
        elif casual_count > 10 and avg_length < 50:
            return "casual"
        elif formal_count > 3 or avg_length > 100:
            return "formal"
        else:
            return "casual"

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
            reaction_prefixes = ["loved an", "liked an", "emphasized", "laughed at", "questioned"]
            if any(r in text for r in reaction_prefixes):
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
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
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

            prompt = (
                "List 3 main topics from these recent messages. "
                "Give only topic names (2-4 words each), one per line.\n\n"
                f"Messages:\n{sample_str}\n\n"
                "Topics (one per line, no numbers or bullets):"
            )

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

            prompt = (
                "Given these messages from a conversation, "
                "provide a 2-4 word topic label describing what they're about.\n\n"
                f"Messages:\n{sample_str}\n\n"
                "Topic label (2-4 words only, no quotes, no explanation):"
            )

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
        filtered = [w for w in words if w not in STOP_WORDS]

        if not filtered:
            return None

        word_counts = Counter(filtered)
        top_word, count = word_counts.most_common(1)[0]

        if count < len(messages) * 0.1:
            return None

        return top_word.title()

    def _get_active_hours(self, messages: list[SimilarMessage]) -> list[int]:
        """Get the most active hours for this conversation.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
        hour_counts = Counter(m.timestamp.hour for m in messages)
        return [hour for hour, _ in hour_counts.most_common(3)]

    def _check_emoji_usage(self, messages: list[SimilarMessage]) -> bool:
        """Check if emojis are commonly used.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
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
        """Check if slang/casual language is commonly used.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
        slang_count = 0
        for m in messages:
            if m.text:
                words = m.text.lower().split()
                slang_count += sum(1 for w in words if w in self.SLANG_WORDS)
        return slang_count > len(messages) * 0.15

    def _check_playfulness(self, messages: list[SimilarMessage]) -> bool:
        """Check if the conversation has playful/teasing tone.

        Note: For bulk analysis, prefer _analyze_all_patterns() which does single pass.
        """
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

    def _extract_common_phrases(
        self, texts: list[str], min_count: int = 3, top_n: int = 5
    ) -> list[str]:
        """Extract commonly used phrases (2-3 words) from messages.

        Ported from v1 relationships.py:634-658.

        Args:
            texts: List of message texts to analyze
            min_count: Minimum occurrences for a phrase to be included
            top_n: Maximum number of phrases to return

        Returns:
            List of commonly used phrases, sorted by frequency
        """
        phrase_counter: Counter[str] = Counter()

        for text in texts:
            if not text or len(text) < 5:
                continue
            text_lower = text.lower()
            # Match words including contractions (don't, that's, y'all)
            words = re.findall(r"\b[\w']+\b", text_lower)

            # Extract 2-word phrases
            for i in range(len(words) - 1):
                # Skip if starts/ends with stop words
                if words[i] in self.STOP_PHRASE_WORDS:
                    continue
                if words[i + 1] in self.STOP_PHRASE_WORDS:
                    continue
                phrase = f"{words[i]} {words[i + 1]}"
                if len(phrase) >= 5:
                    phrase_counter[phrase] += 1

            # Extract 3-word phrases
            for i in range(len(words) - 2):
                if words[i] in self.STOP_PHRASE_WORDS:
                    continue
                if words[i + 2] in self.STOP_PHRASE_WORDS:
                    continue
                phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                phrase_counter[phrase] += 1

        # Filter by min_count and return top N
        filtered = [(p, c) for p, c in phrase_counter.items() if c >= min_count]
        return [p for p, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]

    def _extract_phrases(self, texts: list[str], top_n: int = 5) -> list[str]:
        """Extract commonly used meaningful phrases (legacy wrapper)."""
        return self._extract_common_phrases(texts, min_count=3, top_n=top_n)

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

    def _generate_relationship_summary(
        self,
        messages: list[SimilarMessage],
        profile: ContactProfile,
        display_name: str | None,
    ) -> str:
        """Generate an LLM-based natural language description of this relationship.

        This provides rich context about who this person is and how you interact,
        which can be used in prompts for more personalized replies.

        Args:
            messages: All messages in this conversation
            profile: The computed profile with metrics
            display_name: Contact's name

        Returns:
            Natural language description of the relationship
        """
        if len(messages) < 20:
            return ""

        # Sample diverse messages for context
        # Get recent messages (last 30) and some older ones for perspective
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp, reverse=True)
        recent = sorted_msgs[:30]
        older = sorted_msgs[-20:] if len(sorted_msgs) > 50 else []

        # Sample messages for the prompt
        sample_msgs = []
        for m in recent[:20]:
            if m.text and len(m.text) > 5:
                prefix = "You: " if m.is_from_me else "Them: "
                sample_msgs.append(f"{prefix}{m.text[:100]}")
        for m in older[:10]:
            if m.text and len(m.text) > 5:
                prefix = "You: " if m.is_from_me else "Them: "
                sample_msgs.append(f"{prefix}{m.text[:100]}")

        if len(sample_msgs) < 10:
            return ""

        # Build context from profile data
        name = display_name or "this person"
        rel_type = profile.relationship_type
        tone = profile.tone
        topics_str = (
            ", ".join(t.name.lower() for t in profile.topics[:3])
            if profile.topics else "various topics"
        )

        # Compute interaction patterns
        you_ratio = profile.you_sent / max(profile.total_messages, 1)
        if 0.4 <= you_ratio <= 0.6:
            balance = "balanced"
        elif you_ratio > 0.6:
            balance = "you talk more"
        else:
            balance = "they talk more"

        try:
            from core.models.loader import get_model_loader

            loader = get_model_loader()
            sample_str = "\n".join(sample_msgs[:25])

            prompt = f"""Analyze this conversation and describe the relationship \
in 2-3 sentences.

Contact: {name}
Relationship type: {rel_type}
Tone: {tone}
Total messages: {profile.total_messages}
Message balance: {balance}
Common topics: {topics_str}
{'Playful/teasing dynamic' if profile.is_playful else ''}
{'Uses emoji frequently' if profile.uses_emoji else ''}

Sample messages:
{sample_str}

Write a natural 2-3 sentence description of this relationship. Include:
- How close/formal the relationship seems
- What you typically talk about
- Any notable communication patterns (playful, brief, detailed, etc.)

Description:"""

            result = loader.generate(prompt, max_tokens=150, temperature=0.3)
            summary = result.text.strip()

            # Basic validation - should be reasonable length and not repeat the prompt
            if summary and 20 < len(summary) < 500 and "Description:" not in summary:
                return summary

            return ""

        except Exception as e:
            logger.warning(f"Failed to generate relationship summary: {e}")
            return ""

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
    Cache invalidates when message count changes or profile age exceeds TTL.
    - Style-only profiles (include_topics=False): 24h TTL
    - Full profiles (include_topics=True): 6h TTL

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

    # Try cache first (both style-only and full profiles are cached)
    if use_cache:
        cache = get_profile_cache()
        cached = cache.get(chat_id, current_count, include_topics=include_topics)
        if cached is not None:
            return cached

    # Build fresh profile
    profile = profiler.build_profile(chat_id, display_name, include_topics=include_topics)

    # Cache the result
    if use_cache and profile.total_messages > 0:
        cache = get_profile_cache()
        cache.set(chat_id, profile, current_count, include_topics=include_topics)

    return profile
