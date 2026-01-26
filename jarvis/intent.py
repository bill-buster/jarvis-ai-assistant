"""Intent classification for JARVIS user queries.

Classifies user input to route to appropriate handlers
(reply, summarize, search, quick reply, or general).
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """Types of user intent for routing."""

    QUICK_REPLY = "quick_reply"  # Simple responses handled by templates
    REPLY = "reply"  # Generate reply to a conversation
    SUMMARIZE = "summarize"  # Summarize a conversation
    SEARCH = "search"  # Search for specific messages
    GENERAL = "general"  # General questions/commands


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    extracted_params: dict[str, str | None] = field(default_factory=dict)


class IntentClassifier:
    """Classifies user input to determine the intended action.

    Uses pattern matching for high-confidence intent detection.
    Falls back to GENERAL for ambiguous queries.
    """

    # Patterns for reply intent
    REPLY_PATTERNS = [
        r"(?:help\s+me\s+)?(?:reply|respond)\s+(?:to\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:what\s+should\s+I\s+)?(?:say|write|text)\s+(?:to\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:draft|write|compose)\s+(?:a\s+)?(?:reply|response|message)\s+(?:to\s+|for\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:I\s+need\s+to\s+)?(?:reply|respond|text|message)\s+(?:back\s+)?(?:to\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:how\s+(?:should|do)\s+I\s+)?respond\s+to\s+(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:help|generate)\s+(?:me\s+)?(?:a\s+)?reply\s+(?:to\s+)?(?P<person_name>\w+(?:\s+\w+)?)?",
    ]

    # Patterns for summarize intent
    SUMMARIZE_PATTERNS = [
        r"(?:summarize|recap|summary\s+of)\s+(?:my\s+)?(?:chat|conversation|messages?)\s+(?:with\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:what\s+(?:did|have)\s+(?:I|we)\s+(?:talk|chat|discuss)(?:ed)?\s+about\s+with)\s+(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:give\s+me\s+a\s+)?(?:summary|recap|overview)\s+(?:of\s+)?(?:my\s+)?(?:chat|conversation|messages?)\s+(?:with\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"(?:catch\s+me\s+up\s+on)\s+(?:my\s+)?(?:chat|conversation|messages?)\s+(?:with\s+)?(?P<person_name>\w+(?:\s+\w+)?)",
        r"summarize\s+(?:the\s+)?(?:last\s+)?(?P<time_range>\d+\s+(?:day|week|hour|message)s?)\s+(?:with\s+)?(?P<person_name>\w+(?:\s+\w+)?)?",
        r"(?:summarize|recap)\s+(?P<person_name>\w+(?:\s+\w+)?)",
    ]

    # Patterns for search intent
    SEARCH_PATTERNS = [
        r"(?:find|search|look\s+for)\s+(?:messages?\s+)?(?:about|mentioning|with|containing)\s+['\"]?(?P<search_query>.+?)['\"]?$",
        r"(?:find|search|show)\s+(?:me\s+)?messages?\s+(?:from\s+)?(?P<person_name>\w+(?:\s+\w+)?)\s+(?:about\s+)?(?P<search_query>.+)?",
        r"(?:when\s+did)\s+(?P<person_name>\w+(?:\s+\w+)?)\s+(?:say|mention|text|message)\s+(?:about\s+)?(?P<search_query>.+)",
        r"(?:search\s+(?:for\s+)?)['\"]?(?P<search_query>.+?)['\"]?$",
    ]

    # Quick reply patterns (very short, template-suitable)
    QUICK_REPLY_KEYWORDS = {
        "ok",
        "okay",
        "k",
        "kk",
        "sure",
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "thanks",
        "thx",
        "ty",
        "lol",
        "haha",
        "bye",
        "hi",
        "hey",
        "gn",
        "gm",
    }

    def __init__(self) -> None:
        """Initialize the intent classifier."""
        # Compile patterns for efficiency
        self._reply_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.REPLY_PATTERNS
        ]
        self._summarize_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SUMMARIZE_PATTERNS
        ]
        self._search_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SEARCH_PATTERNS
        ]

    def classify(self, user_input: str) -> IntentResult:
        """Classify user input to determine intent.

        Args:
            user_input: The user's input text.

        Returns:
            IntentResult with intent type, confidence, and extracted parameters.
        """
        text = user_input.strip()
        text_lower = text.lower()

        # Check for quick reply (very short messages)
        if self._is_quick_reply(text_lower):
            return IntentResult(
                intent=IntentType.QUICK_REPLY,
                confidence=0.95,
                extracted_params={},
            )

        # Check for reply intent
        for pattern in self._reply_patterns:
            match = pattern.search(text)
            if match:
                params = match.groupdict()
                # Clean up person name
                person_name = params.get("person_name")
                if person_name:
                    person_name = person_name.strip()
                    # Don't match common words as names
                    if person_name.lower() in {"the", "a", "my", "this", "that"}:
                        person_name = None
                return IntentResult(
                    intent=IntentType.REPLY,
                    confidence=0.9,
                    extracted_params={"person_name": person_name},
                )

        # Check for summarize intent
        for pattern in self._summarize_patterns:
            match = pattern.search(text)
            if match:
                params = match.groupdict()
                person_name = params.get("person_name")
                if person_name:
                    person_name = person_name.strip()
                    if person_name.lower() in {"the", "a", "my", "this", "that"}:
                        person_name = None
                time_range = params.get("time_range")
                return IntentResult(
                    intent=IntentType.SUMMARIZE,
                    confidence=0.9,
                    extracted_params={
                        "person_name": person_name,
                        "time_range": time_range,
                    },
                )

        # Check for search intent
        for pattern in self._search_patterns:
            match = pattern.search(text)
            if match:
                params = match.groupdict()
                search_query = params.get("search_query")
                if search_query:
                    search_query = search_query.strip().strip("'\"")
                person_name = params.get("person_name")
                if person_name:
                    person_name = person_name.strip()
                return IntentResult(
                    intent=IntentType.SEARCH,
                    confidence=0.85,
                    extracted_params={
                        "search_query": search_query,
                        "person_name": person_name,
                    },
                )

        # Default to general intent
        return IntentResult(
            intent=IntentType.GENERAL,
            confidence=0.5,
            extracted_params={},
        )

    def _is_quick_reply(self, text: str) -> bool:
        """Check if text is a quick reply suitable for templates.

        Args:
            text: Lowercase input text.

        Returns:
            True if this is a quick reply pattern.
        """
        # Very short messages
        if len(text) <= 3 and text.isalpha():
            return True

        # Known quick reply keywords
        if text in self.QUICK_REPLY_KEYWORDS:
            return True

        # Single emoji or emoji-only messages
        # Simple check: if it's very short and not ASCII
        if len(text) <= 4 and not text.isascii():
            return True

        return False
