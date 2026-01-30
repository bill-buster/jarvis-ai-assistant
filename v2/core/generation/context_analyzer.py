"""Conversation context analyzer for JARVIS v2.

Analyzes conversations to understand context for reply generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageIntent(Enum):
    """Detected intent of a message."""

    YES_NO_QUESTION = "yes_no_question"
    OPEN_QUESTION = "open_question"
    CHOICE_QUESTION = "choice_question"
    STATEMENT = "statement"
    EMOTIONAL = "emotional"
    GREETING = "greeting"
    LOGISTICS = "logistics"
    SHARING = "sharing"
    THANKS = "thanks"
    FAREWELL = "farewell"


class RelationshipType(Enum):
    """Inferred relationship type."""

    CLOSE_FRIEND = "close_friend"
    CASUAL_FRIEND = "casual_friend"
    FAMILY = "family"
    WORK = "work"
    ROMANTIC = "romantic"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    """Analyzed conversation context."""

    last_message: str
    last_sender: str
    intent: MessageIntent
    relationship: RelationshipType
    topic: str
    mood: str  # "positive" | "neutral" | "negative"
    urgency: str  # "high" | "normal" | "low"
    needs_response: bool
    summary: str


class ContextAnalyzer:
    """Analyzes conversation context for reply generation."""

    def analyze(self, messages: list[dict]) -> ConversationContext:
        """Analyze conversation context.

        Args:
            messages: Recent messages [{"text": "...", "sender": "...", "is_from_me": bool}, ...]

        Returns:
            ConversationContext with analysis
        """
        if not messages:
            return self._default_context()

        last_msg = messages[-1]
        last_text = last_msg.get("text", "")
        last_sender = "me" if last_msg.get("is_from_me") else last_msg.get("sender", "them")

        # Detect intent
        intent = self._detect_intent(last_text)

        # Detect relationship
        relationship = self._detect_relationship(messages)

        # Detect topic
        topic = self._detect_topic(messages[-10:])

        # Detect mood
        mood = self._detect_mood(messages[-5:])

        # Detect urgency
        urgency = self._detect_urgency(last_text)

        # Check if response is needed
        needs_response = (
            not last_msg.get("is_from_me") and
            intent in [
                MessageIntent.YES_NO_QUESTION,
                MessageIntent.OPEN_QUESTION,
                MessageIntent.CHOICE_QUESTION,
                MessageIntent.GREETING,
            ]
        )

        # Generate summary
        summary = self._summarize_thread(messages[-10:])

        return ConversationContext(
            last_message=last_text,
            last_sender=last_sender,
            intent=intent,
            relationship=relationship,
            topic=topic,
            mood=mood,
            urgency=urgency,
            needs_response=needs_response,
            summary=summary,
        )

    def _detect_intent(self, text: str) -> MessageIntent:
        """Detect intent of a message."""
        text_lower = text.lower().strip()

        # Greeting patterns - check FIRST (before questions)
        # Many greetings end with "?" like "what's up?" or "how are you?"
        greetings = [
            "hey", "hi", "hello", "what's up", "whats up", "sup", "how are",
            "how's it", "hows it", "good morning", "good afternoon", "good evening", "yo",
        ]
        if any(text_lower.startswith(g) for g in greetings):
            return MessageIntent.GREETING

        # Question detection
        if text.rstrip().endswith("?"):
            # Yes/No patterns
            yes_no_starters = [
                "do you", "are you", "can you", "will you", "would you",
                "want to", "wanna", "could you", "should we", "shall we",
                "is it", "are we", "did you", "have you", "has ", "was ",
                "were you", "r u", "u wanna", "u want",
            ]
            if any(text_lower.startswith(s) for s in yes_no_starters):
                return MessageIntent.YES_NO_QUESTION

            # Choice patterns
            if " or " in text_lower:
                return MessageIntent.CHOICE_QUESTION

            return MessageIntent.OPEN_QUESTION

        # Farewell patterns
        farewells = [
            "bye", "goodbye", "see you", "see ya", "talk later", "gotta go",
            "ttyl", "later", "night", "good night", "take care",
        ]
        if any(f in text_lower for f in farewells):
            return MessageIntent.FAREWELL

        # Thanks patterns
        thanks = ["thank", "thx", "ty", "tysm", "appreciate"]
        if any(t in text_lower for t in thanks):
            return MessageIntent.THANKS

        # Sharing/giving patterns (check BEFORE emotional to avoid "ugh" in "brought")
        sharing_patterns = [
            "http", "check out", "look at", "brought you", "got you",
            "for you", "made you", "found this", "sending you", "here's",
            "got this for", "picked up", "brought the", "i brought",
        ]
        if any(p in text_lower for p in sharing_patterns):
            return MessageIntent.SHARING

        # Logistics patterns
        logistics_words = [
            "running late", "on my way", "omw", "be there", "arrived",
            "leaving now", "eta", "here", "parking", "waiting",
        ]
        if any(w in text_lower for w in logistics_words):
            return MessageIntent.LOGISTICS

        # Emotional patterns (avoid matching "ugh" inside words like "brought")
        emotional_phrases = [
            "stressed", "so sad", "so happy", "excited", "worried", "anxious",
            "i love", "i hate", "omg", "so tired", "exhausted",
            "frustrated", "annoyed", "thrilled", "devastated", "rough day",
            "was rough", "feeling down", "feeling good",
        ]
        # Check if starts with "ugh" or has " ugh " as word
        if text_lower.startswith("ugh") or " ugh " in text_lower or " ugh" in text_lower[-4:]:
            return MessageIntent.EMOTIONAL
        if any(w in text_lower for w in emotional_phrases):
            return MessageIntent.EMOTIONAL

        return MessageIntent.STATEMENT

    def _detect_relationship(self, messages: list[dict]) -> RelationshipType:
        """Infer relationship type from conversation patterns."""
        texts = [m.get("text", "").lower() for m in messages if m.get("text")]

        if not texts:
            return RelationshipType.UNKNOWN

        combined = " ".join(texts)

        # Romantic indicators
        romantic_words = ["love you", "miss you", "babe", "baby", "honey", "sweetie", "❤️", "😘"]
        if any(w in combined for w in romantic_words):
            return RelationshipType.ROMANTIC

        # Family indicators
        family_words = ["mom", "dad", "sis", "bro", "family dinner", "grandma", "grandpa"]
        if any(w in combined for w in family_words):
            return RelationshipType.FAMILY

        # Work indicators
        work_words = ["meeting", "deadline", "project", "client", "boss", "office", "work"]
        if any(w in combined for w in work_words):
            return RelationshipType.WORK

        # Close friend indicators (frequent, casual)
        casual_indicators = ["lol", "lmao", "haha", "omg", "dude", "bro"]
        casual_count = sum(1 for w in casual_indicators if w in combined)
        if casual_count >= 3:
            return RelationshipType.CLOSE_FRIEND

        return RelationshipType.CASUAL_FRIEND

    def _detect_topic(self, messages: list[dict]) -> str:
        """Detect conversation topic."""
        texts = [m.get("text", "").lower() for m in messages if m.get("text")]
        combined = " ".join(texts)

        topics = {
            "food/dining": ["dinner", "lunch", "eat", "food", "restaurant", "hungry"],
            "plans": ["tonight", "tomorrow", "weekend", "later", "meet", "hang"],
            "work": ["work", "meeting", "project", "deadline", "boss", "office"],
            "travel": ["trip", "flight", "hotel", "vacation", "travel"],
            "entertainment": ["movie", "show", "game", "concert", "watch"],
            "catching up": ["how are", "what's up", "been up to", "how's"],
        }

        for topic, keywords in topics.items():
            if any(k in combined for k in keywords):
                return topic

        return "general"

    def _detect_mood(self, messages: list[dict]) -> str:
        """Detect conversation mood."""
        texts = [m.get("text", "").lower() for m in messages if m.get("text")]

        positive_count = 0
        negative_count = 0

        positive_words = [
            "great", "awesome", "amazing", "love", "happy", "excited",
            "yes", "yeah", "yay", "perfect", "wonderful", "!", "😊", "😄", "🎉",
        ]
        negative_words = [
            "bad", "terrible", "hate", "sad", "angry", "frustrated",
            "ugh", "no", "can't", "won't", "😢", "😠", "disappointed",
        ]

        for text in texts:
            for word in positive_words:
                if word in text:
                    positive_count += 1
            for word in negative_words:
                if word in text:
                    negative_count += 1

        if positive_count > negative_count + 2:
            return "positive"
        elif negative_count > positive_count + 2:
            return "negative"
        return "neutral"

    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level."""
        text_lower = text.lower()

        high_urgency = ["asap", "urgent", "emergency", "now", "immediately", "hurry"]
        if any(u in text_lower for u in high_urgency):
            return "high"

        # Multiple question/exclamation marks
        if text.count("?") > 1 or text.count("!") > 2:
            return "high"

        return "normal"

    def _summarize_thread(self, messages: list[dict]) -> str:
        """Generate brief thread summary."""
        if not messages:
            return "No recent messages"

        # Get participants
        senders = set()
        for m in messages:
            if m.get("is_from_me"):
                senders.add("you")
            else:
                senders.add(m.get("sender", "them"))

        # Get recent topic indicators
        recent_texts = [m.get("text", "")[:50] for m in messages[-3:] if m.get("text")]

        if len(recent_texts) == 0:
            return "Empty conversation"

        participants = " and ".join(senders)
        return f"Conversation between {participants}. Recent: {'; '.join(recent_texts)}"

    def _default_context(self) -> ConversationContext:
        """Return default context for empty conversations."""
        return ConversationContext(
            last_message="",
            last_sender="unknown",
            intent=MessageIntent.STATEMENT,
            relationship=RelationshipType.UNKNOWN,
            topic="general",
            mood="neutral",
            urgency="normal",
            needs_response=False,
            summary="No messages",
        )
