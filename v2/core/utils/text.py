"""Text processing utilities for JARVIS v2."""

from __future__ import annotations

from typing import TypedDict

# Comprehensive stop words for text analysis
# Used by embedding store and contact profiler for filtering common words
STOP_WORDS: frozenset[str] = frozenset({
    # Basic articles and prepositions
    "the", "a", "an", "is", "it", "to", "and", "or", "of", "in", "on",
    "for", "with", "at", "by", "from", "this", "that", "i", "you", "we",
    "my", "your", "am", "are", "was", "were", "be", "have", "has", "do",
    "does", "did", "will", "would", "could", "should", "just", "so", "but",
    # Extended common words
    "they", "been", "being", "their", "about", "which", "there", "what",
    "when", "make", "like", "know", "take", "come", "think", "good", "some",
    "than", "then", "very", "after", "before", "going", "here", "also",
    "want", "need", "said", "says", "okay", "really", "thing", "things",
    "stuff", "right", "doing", "getting", "wanted", "wants", "liked", "likes",
    # Contractions and casual speech
    "gonna", "gotta", "wanna", "cant", "dont", "didnt", "doesnt",
    "thats", "youre", "theyre", "its", "heres", "theres",
    # Quantifiers
    "much", "many", "more", "most", "other", "another", "same",
    # Positional
    "into", "over", "under", "through", "back", "down", "still",
    "even", "well", "only", "because", "though", "actually",
    "probably", "maybe",
    # Conversational
    "yeah", "yea", "yes", "sure", "thanks", "thank", "sorry", "please",
    "alright", "sounds", "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "people",
    # Time words
    "time", "today", "tomorrow", "tonight", "morning", "night", "week",
    "month", "year",
    # iMessage reaction artifacts
    "loved", "liked", "emphasized", "laughed", "questioned", "disliked",
    "image", "message", "attachment",
    # Common verbs/fillers
    "thought", "coming", "looking", "making", "having", "saying", "trying",
    "waiting", "working", "sending", "checking", "letting",
})


class MessageDict(TypedDict, total=False):
    """Type definition for message dictionaries used in indexing.

    This provides type hints for the dict format expected by
    EmbeddingStore.index_messages() and related functions.
    """

    id: int  # Message ROWID
    text: str  # Message text content
    chat_id: str  # Conversation identifier
    sender: str  # Sender phone/email
    sender_name: str | None  # Resolved contact name
    timestamp: float  # Unix timestamp
    is_from_me: bool  # Whether user sent this message
