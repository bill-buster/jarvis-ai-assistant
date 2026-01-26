"""Conversation statistics API endpoints.

Provides endpoints for computing and retrieving conversation analytics including
message counts, response times, activity patterns, word frequency, and more.

Uses TTL caching for computed statistics with cache invalidation based on
conversation activity.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ConversationStatsResponse,
    ErrorResponse,
    HourlyActivity,
    TimeRangeEnum,
    WordFrequency,
)
from integrations.imessage import ChatDBReader
from jarvis.metrics import TTLCache

router = APIRouter(prefix="/stats", tags=["stats"])

# Cache for computed statistics - TTL varies based on time range
_stats_cache: TTLCache | None = None


def get_stats_cache() -> TTLCache:
    """Get the stats cache with dynamic TTL based on activity."""
    global _stats_cache
    if _stats_cache is None:
        # 5 minute TTL, max 100 cached entries
        _stats_cache = TTLCache(ttl_seconds=300.0, maxsize=100)
    return _stats_cache


# Common words to exclude from word frequency analysis
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "ok",
        "okay",
        "yeah",
        "yes",
        "no",
        "like",
        "just",
        "get",
        "got",
        "go",
        "going",
        "know",
        "think",
        "see",
        "come",
        "make",
        "well",
        "back",
        "good",
        "really",
        "want",
        "time",
        "right",
        "thing",
        "way",
        "even",
        "new",
        "one",
        "two",
    }
)

# Emoji pattern for detection
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


def _get_time_range_start(time_range: TimeRangeEnum) -> datetime | None:
    """Get the start datetime for a given time range."""
    now = datetime.now()
    if time_range == TimeRangeEnum.WEEK:
        return now - timedelta(days=7)
    elif time_range == TimeRangeEnum.MONTH:
        return now - timedelta(days=30)
    elif time_range == TimeRangeEnum.THREE_MONTHS:
        return now - timedelta(days=90)
    elif time_range == TimeRangeEnum.ALL_TIME:
        return None
    return None


def _compute_response_time(messages: list[Any]) -> float | None:
    """Compute average response time in minutes between messages.

    Only considers responses within 24 hours.
    """
    if len(messages) < 2:
        return None

    response_times: list[float] = []
    prev_msg = None

    for msg in messages:
        if prev_msg is not None:
            # Only count if sender changed (actual response)
            if msg.is_from_me != prev_msg.is_from_me:
                time_diff = (msg.date - prev_msg.date).total_seconds()
                # Only count responses within 24 hours
                if 0 < time_diff < 86400:  # 24 hours in seconds
                    response_times.append(time_diff / 60.0)  # Convert to minutes
        prev_msg = msg

    if not response_times:
        return None

    return sum(response_times) / len(response_times)


def _compute_hourly_activity(messages: list[Any]) -> list[HourlyActivity]:
    """Compute message counts by hour of day."""
    hour_counts: Counter[int] = Counter(msg.date.hour for msg in messages)
    return [
        HourlyActivity(hour=hour, count=hour_counts.get(hour, 0)) for hour in range(24)
    ]


def _compute_daily_activity(messages: list[Any]) -> dict[str, int]:
    """Compute message counts by day of week."""
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_counts: Counter[int] = Counter(msg.date.weekday() for msg in messages)
    return {day_names[i]: day_counts.get(i, 0) for i in range(7)}


def _compute_message_lengths(messages: list[Any]) -> dict[str, int]:
    """Compute message length distribution."""
    length_buckets = {"short": 0, "medium": 0, "long": 0, "very_long": 0}

    for msg in messages:
        length = len(msg.text) if msg.text else 0
        if length <= 20:
            length_buckets["short"] += 1
        elif length <= 100:
            length_buckets["medium"] += 1
        elif length <= 300:
            length_buckets["long"] += 1
        else:
            length_buckets["very_long"] += 1

    return length_buckets


def _extract_words(text: str) -> list[str]:
    """Extract words from text, filtering stop words and short words."""
    if not text:
        return []
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove special characters, keep letters and numbers
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _compute_word_frequency(messages: list[Any], top_n: int = 20) -> list[WordFrequency]:
    """Compute most frequently used words."""
    word_counts: Counter[str] = Counter()

    for msg in messages:
        if msg.text:
            words = _extract_words(msg.text)
            word_counts.update(words)

    return [
        WordFrequency(word=word, count=count)
        for word, count in word_counts.most_common(top_n)
    ]


def _compute_emoji_stats(messages: list[Any]) -> dict[str, int]:
    """Compute emoji usage statistics."""
    emoji_counts: Counter[str] = Counter()

    for msg in messages:
        if msg.text:
            emojis = EMOJI_PATTERN.findall(msg.text)
            for emoji_group in emojis:
                # Split combined emojis
                for char in emoji_group:
                    if EMOJI_PATTERN.match(char):
                        emoji_counts[char] += 1

    # Return top 10 emojis
    return dict(emoji_counts.most_common(10))


def _compute_attachment_stats(messages: list[Any]) -> dict[str, int]:
    """Compute attachment type breakdown."""
    type_counts: Counter[str] = Counter()

    for msg in messages:
        for attachment in msg.attachments:
            if attachment.mime_type:
                # Categorize by mime type
                mime = attachment.mime_type.lower()
                if mime.startswith("image/"):
                    type_counts["images"] += 1
                elif mime.startswith("video/"):
                    type_counts["videos"] += 1
                elif mime.startswith("audio/"):
                    type_counts["audio"] += 1
                elif mime in ("application/pdf", "application/msword"):
                    type_counts["documents"] += 1
                else:
                    type_counts["other"] += 1
            else:
                type_counts["unknown"] += 1

    return dict(type_counts)


@router.get(
    "/{chat_id}",
    response_model=ConversationStatsResponse,
    response_model_exclude_unset=True,
    response_description="Conversation statistics and analytics",
    summary="Get conversation statistics",
    responses={
        200: {
            "description": "Statistics computed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "chat_id": "chat123456789",
                        "time_range": "month",
                        "total_messages": 500,
                        "sent_count": 250,
                        "received_count": 250,
                        "avg_response_time_minutes": 15.5,
                        "hourly_activity": [{"hour": 9, "count": 45}],
                        "daily_activity": {"Monday": 80, "Tuesday": 70},
                        "message_length_distribution": {
                            "short": 200,
                            "medium": 200,
                            "long": 80,
                            "very_long": 20,
                        },
                        "top_words": [{"word": "hello", "count": 50}],
                        "emoji_usage": {"❤️": 25, "😊": 20},
                        "attachment_breakdown": {"images": 30, "videos": 5},
                        "first_message_date": "2024-01-01T10:00:00Z",
                        "last_message_date": "2024-01-15T18:30:00Z",
                        "participants": ["+15551234567"],
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_conversation_stats(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for statistics calculation",
    ),
    limit: int = Query(
        default=500,
        ge=50,
        le=5000,
        description="Maximum number of messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ConversationStatsResponse:
    """Get comprehensive statistics for a conversation.

    Analyzes messages to compute:
    - Message counts (sent vs received)
    - Average response time
    - Hourly activity patterns (histogram data)
    - Daily activity by day of week
    - Message length distribution
    - Most frequently used words
    - Emoji usage statistics
    - Attachment type breakdown

    Statistics are cached for 5 minutes to improve performance on
    repeated requests.

    **Time Ranges:**
    - `week`: Last 7 days
    - `month`: Last 30 days (default)
    - `three_months`: Last 90 days
    - `all_time`: All messages up to limit

    **Example Response:**
    ```json
    {
        "chat_id": "chat123456789",
        "time_range": "month",
        "total_messages": 500,
        "sent_count": 250,
        "received_count": 250,
        "avg_response_time_minutes": 15.5,
        "hourly_activity": [
            {"hour": 0, "count": 5},
            {"hour": 1, "count": 2},
            ...
        ],
        "daily_activity": {
            "Monday": 80,
            "Tuesday": 70,
            ...
        },
        "message_length_distribution": {
            "short": 200,
            "medium": 200,
            "long": 80,
            "very_long": 20
        },
        "top_words": [
            {"word": "hello", "count": 50},
            {"word": "thanks", "count": 35}
        ],
        "emoji_usage": {
            "❤️": 25,
            "😊": 20,
            "😂": 15
        },
        "attachment_breakdown": {
            "images": 30,
            "videos": 5,
            "documents": 2
        }
    }
    ```

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis (week/month/three_months/all_time)
        limit: Maximum messages to analyze (50-5000, default 500)

    Returns:
        ConversationStatsResponse with computed statistics

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    # Build cache key
    cache_key = f"stats:{chat_id}:{time_range.value}:{limit}"
    cache = get_stats_cache()

    # Check cache
    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[return-value]

    # Get time range start for filtering
    time_range_start = _get_time_range_start(time_range)

    # Fetch messages - we fetch more than limit to account for time filtering
    # The reader returns messages in reverse chronological order (newest first)
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    # Filter by time range if specified
    if time_range_start is not None:
        messages = [m for m in messages if m.date >= time_range_start]

    # Sort messages chronologically for response time calculation
    messages_sorted = sorted(messages, key=lambda m: m.date)

    # Compute statistics
    total = len(messages_sorted)
    sent_count = sum(1 for m in messages_sorted if m.is_from_me)
    received_count = total - sent_count

    # Compute detailed statistics
    avg_response_time = _compute_response_time(messages_sorted)
    hourly_activity = _compute_hourly_activity(messages_sorted)
    daily_activity = _compute_daily_activity(messages_sorted)
    message_lengths = _compute_message_lengths(messages_sorted)
    top_words = _compute_word_frequency(messages_sorted)
    emoji_usage = _compute_emoji_stats(messages_sorted)
    attachment_breakdown = _compute_attachment_stats(messages_sorted)

    # Get date range
    first_date = messages_sorted[0].date if messages_sorted else None
    last_date = messages_sorted[-1].date if messages_sorted else None

    # Get participants from conversation
    conversations = reader.get_conversations(limit=100)
    participants = []
    for conv in conversations:
        if conv.chat_id == chat_id:
            participants = conv.participants
            break

    result = ConversationStatsResponse(
        chat_id=chat_id,
        time_range=time_range,
        total_messages=total,
        sent_count=sent_count,
        received_count=received_count,
        avg_response_time_minutes=round(avg_response_time, 1)
        if avg_response_time
        else None,
        hourly_activity=hourly_activity,
        daily_activity=daily_activity,
        message_length_distribution=message_lengths,
        top_words=top_words,
        emoji_usage=emoji_usage,
        attachment_breakdown=attachment_breakdown,
        first_message_date=first_date,
        last_message_date=last_date,
        participants=participants,
    )

    # Cache the result
    cache.set(cache_key, result)

    return result


@router.delete(
    "/{chat_id}/cache",
    response_description="Cache invalidation confirmation",
    summary="Invalidate stats cache for a conversation",
)
def invalidate_stats_cache(
    chat_id: str,
) -> dict[str, str]:
    """Invalidate cached statistics for a conversation.

    Use this endpoint to force a refresh of statistics on the next request.
    Useful when you know the conversation has been updated recently.

    Args:
        chat_id: The conversation identifier

    Returns:
        Confirmation message
    """
    cache = get_stats_cache()

    # Invalidate all cache entries for this chat_id
    # Since we can't iterate TTLCache, we'll just invalidate known patterns
    for time_range in TimeRangeEnum:
        for limit in [50, 100, 200, 500, 1000, 2000, 5000]:
            cache_key = f"stats:{chat_id}:{time_range.value}:{limit}"
            cache.invalidate(cache_key)

    return {"status": "ok", "message": f"Cache invalidated for {chat_id}"}
