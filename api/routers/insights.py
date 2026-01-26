"""Conversation insights API endpoints.

Provides endpoints for sentiment analysis, response time patterns,
message frequency trends, and relationship health scoring.

Uses TTL caching for computed insights with cache invalidation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ConversationInsightsResponse,
    ErrorResponse,
    FrequencyTrendsResponse,
    RelationshipHealthResponse,
    ResponsePatternsResponse,
    SentimentResponse,
    SentimentTrendResponse,
    TimeRangeEnum,
)
from integrations.imessage import ChatDBReader
from jarvis.insights import (
    analyze_frequency_trends,
    analyze_response_patterns,
    analyze_sentiment,
    calculate_relationship_health,
    generate_conversation_insights,
)
from jarvis.metrics import TTLCache

router = APIRouter(prefix="/insights", tags=["insights"])

# Cache for computed insights - 10 minute TTL
_insights_cache: TTLCache | None = None


def get_insights_cache() -> TTLCache:
    """Get the insights cache singleton."""
    global _insights_cache
    if _insights_cache is None:
        _insights_cache = TTLCache(ttl_seconds=600.0, maxsize=50)
    return _insights_cache


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


@router.get(
    "/{chat_id}",
    response_model=ConversationInsightsResponse,
    response_model_exclude_unset=True,
    response_description="Complete conversation insights including sentiment, patterns, and health",
    summary="Get conversation insights",
    responses={
        200: {
            "description": "Insights computed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "chat_id": "chat123456789",
                        "contact_name": "John Doe",
                        "time_range": "month",
                        "sentiment_overall": {
                            "score": 0.45,
                            "label": "positive",
                            "positive_count": 120,
                            "negative_count": 30,
                        },
                        "sentiment_trends": [
                            {"date": "2024-W01", "score": 0.3, "message_count": 45},
                        ],
                        "response_patterns": {
                            "avg_response_time_minutes": 15.5,
                            "median_response_time_minutes": 8.0,
                        },
                        "frequency_trends": {
                            "trend_direction": "stable",
                            "messages_per_day_avg": 12.5,
                        },
                        "relationship_health": {
                            "overall_score": 75.5,
                            "health_label": "good",
                        },
                        "total_messages_analyzed": 500,
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
def get_conversation_insights(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for analysis",
    ),
    limit: int = Query(
        default=500,
        ge=50,
        le=5000,
        description="Maximum number of messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ConversationInsightsResponse:
    """Get comprehensive insights for a conversation.

    Analyzes messages to compute:
    - Overall sentiment score and trend over time
    - Response time patterns (yours vs theirs)
    - Message frequency trends (daily/weekly/monthly)
    - Relationship health score with contributing factors

    Insights are cached for 10 minutes to improve performance.

    **Time Ranges:**
    - `week`: Last 7 days
    - `month`: Last 30 days (default)
    - `three_months`: Last 90 days
    - `all_time`: All messages up to limit

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis
        limit: Maximum messages to analyze (50-5000, default 500)

    Returns:
        ConversationInsightsResponse with computed insights
    """
    # Build cache key
    cache_key = f"insights:{chat_id}:{time_range.value}:{limit}"
    cache = get_insights_cache()

    # Check cache
    found, cached = cache.get(cache_key)
    if found and isinstance(cached, ConversationInsightsResponse):
        return cached

    # Get time range start
    time_range_start = _get_time_range_start(time_range)

    # Fetch messages
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    # Filter by time range
    if time_range_start is not None:
        messages = [m for m in messages if m.date >= time_range_start]

    # Get contact name from conversations
    contact_name = None
    conversations = reader.get_conversations(limit=100)
    for conv in conversations:
        if conv.chat_id == chat_id:
            contact_name = conv.display_name
            break

    # Generate insights
    insights = generate_conversation_insights(
        chat_id=chat_id,
        messages=messages,
        contact_name=contact_name,
        time_range=time_range.value,
    )

    # Convert to response model
    result = ConversationInsightsResponse(
        chat_id=insights.chat_id,
        contact_name=insights.contact_name,
        time_range=time_range,
        sentiment_overall=SentimentResponse(
            score=insights.sentiment_overall.score,
            label=insights.sentiment_overall.label,
            positive_count=insights.sentiment_overall.positive_count,
            negative_count=insights.sentiment_overall.negative_count,
            neutral_count=insights.sentiment_overall.neutral_count,
        ),
        sentiment_trends=[
            SentimentTrendResponse(
                date=t.date,
                score=t.score,
                message_count=t.message_count,
            )
            for t in insights.sentiment_trends
        ],
        response_patterns=ResponsePatternsResponse(
            avg_response_time_minutes=insights.response_patterns.avg_response_time_minutes,
            median_response_time_minutes=insights.response_patterns.median_response_time_minutes,
            fastest_response_minutes=insights.response_patterns.fastest_response_minutes,
            slowest_response_minutes=insights.response_patterns.slowest_response_minutes,
            response_times_by_hour=insights.response_patterns.response_times_by_hour,
            response_times_by_day=insights.response_patterns.response_times_by_day,
            my_avg_response_time_minutes=insights.response_patterns.my_avg_response_time_minutes,
            their_avg_response_time_minutes=insights.response_patterns.their_avg_response_time_minutes,
        ),
        frequency_trends=FrequencyTrendsResponse(
            daily_counts=insights.frequency_trends.daily_counts,
            weekly_counts=insights.frequency_trends.weekly_counts,
            monthly_counts=insights.frequency_trends.monthly_counts,
            trend_direction=insights.frequency_trends.trend_direction,
            trend_percentage=insights.frequency_trends.trend_percentage,
            most_active_day=insights.frequency_trends.most_active_day,
            most_active_hour=insights.frequency_trends.most_active_hour,
            messages_per_day_avg=insights.frequency_trends.messages_per_day_avg,
        ),
        relationship_health=RelationshipHealthResponse(
            overall_score=insights.relationship_health.overall_score,
            engagement_score=insights.relationship_health.engagement_score,
            sentiment_score=insights.relationship_health.sentiment_score,
            responsiveness_score=insights.relationship_health.responsiveness_score,
            consistency_score=insights.relationship_health.consistency_score,
            health_label=insights.relationship_health.health_label,
            factors=insights.relationship_health.factors,
        ),
        total_messages_analyzed=insights.total_messages_analyzed,
        first_message_date=datetime.fromisoformat(insights.first_message_date)
        if insights.first_message_date
        else None,
        last_message_date=datetime.fromisoformat(insights.last_message_date)
        if insights.last_message_date
        else None,
    )

    # Cache the result
    cache.set(cache_key, result)

    return result


@router.get(
    "/{chat_id}/sentiment",
    response_model=SentimentResponse,
    summary="Get sentiment analysis for a conversation",
    responses={
        200: {
            "description": "Sentiment analysis computed successfully",
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_sentiment_analysis(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for analysis",
    ),
    limit: int = Query(
        default=200,
        ge=20,
        le=1000,
        description="Maximum messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> SentimentResponse:
    """Get sentiment analysis for a conversation.

    Analyzes message text to determine overall sentiment polarity
    using a lexicon-based approach with emoji support.

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis
        limit: Maximum messages to analyze

    Returns:
        SentimentResponse with sentiment score and breakdown
    """
    time_range_start = _get_time_range_start(time_range)
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if time_range_start:
        messages = [m for m in messages if m.date >= time_range_start]

    all_text = " ".join(m.text for m in messages if m.text)
    sentiment = analyze_sentiment(all_text)

    return SentimentResponse(
        score=sentiment.score,
        label=sentiment.label,
        positive_count=sentiment.positive_count,
        negative_count=sentiment.negative_count,
        neutral_count=sentiment.neutral_count,
    )


@router.get(
    "/{chat_id}/response-patterns",
    response_model=ResponsePatternsResponse,
    summary="Get response time patterns",
    responses={
        200: {
            "description": "Response patterns computed successfully",
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_response_patterns(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for analysis",
    ),
    limit: int = Query(
        default=500,
        ge=50,
        le=2000,
        description="Maximum messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ResponsePatternsResponse:
    """Get response time pattern analysis.

    Analyzes when you typically respond to this contact and vice versa,
    broken down by hour of day and day of week.

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis
        limit: Maximum messages to analyze

    Returns:
        ResponsePatternsResponse with detailed timing analysis
    """
    time_range_start = _get_time_range_start(time_range)
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if time_range_start:
        messages = [m for m in messages if m.date >= time_range_start]

    # Sort chronologically for response time calculation
    messages = sorted(messages, key=lambda m: m.date)
    patterns = analyze_response_patterns(messages)

    return ResponsePatternsResponse(
        avg_response_time_minutes=patterns.avg_response_time_minutes,
        median_response_time_minutes=patterns.median_response_time_minutes,
        fastest_response_minutes=patterns.fastest_response_minutes,
        slowest_response_minutes=patterns.slowest_response_minutes,
        response_times_by_hour=patterns.response_times_by_hour,
        response_times_by_day=patterns.response_times_by_day,
        my_avg_response_time_minutes=patterns.my_avg_response_time_minutes,
        their_avg_response_time_minutes=patterns.their_avg_response_time_minutes,
    )


@router.get(
    "/{chat_id}/frequency",
    response_model=FrequencyTrendsResponse,
    summary="Get message frequency trends",
    responses={
        200: {
            "description": "Frequency trends computed successfully",
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_frequency_trends(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.THREE_MONTHS,
        description="Time range for analysis",
    ),
    limit: int = Query(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> FrequencyTrendsResponse:
    """Get message frequency trends over time.

    Provides daily, weekly, and monthly message counts along with
    trend analysis (increasing, decreasing, or stable).

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis
        limit: Maximum messages to analyze

    Returns:
        FrequencyTrendsResponse with trend data
    """
    time_range_start = _get_time_range_start(time_range)
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if time_range_start:
        messages = [m for m in messages if m.date >= time_range_start]

    trends = analyze_frequency_trends(messages)

    return FrequencyTrendsResponse(
        daily_counts=trends.daily_counts,
        weekly_counts=trends.weekly_counts,
        monthly_counts=trends.monthly_counts,
        trend_direction=trends.trend_direction,
        trend_percentage=trends.trend_percentage,
        most_active_day=trends.most_active_day,
        most_active_hour=trends.most_active_hour,
        messages_per_day_avg=trends.messages_per_day_avg,
    )


@router.get(
    "/{chat_id}/health",
    response_model=RelationshipHealthResponse,
    summary="Get relationship health score",
    responses={
        200: {
            "description": "Health score computed successfully",
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_relationship_health(
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for analysis",
    ),
    limit: int = Query(
        default=500,
        ge=50,
        le=2000,
        description="Maximum messages to analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> RelationshipHealthResponse:
    """Get relationship health score based on conversation patterns.

    Calculates a composite health score based on:
    - Engagement (message balance and frequency)
    - Sentiment (overall tone of conversation)
    - Responsiveness (how quickly both parties respond)
    - Consistency (regularity of communication)

    Args:
        chat_id: The conversation identifier
        time_range: Time range for analysis
        limit: Maximum messages to analyze

    Returns:
        RelationshipHealthResponse with detailed breakdown
    """
    time_range_start = _get_time_range_start(time_range)
    messages = reader.get_messages(chat_id=chat_id, limit=limit)

    if time_range_start:
        messages = [m for m in messages if m.date >= time_range_start]

    # Sort chronologically
    messages = sorted(messages, key=lambda m: m.date)

    # Calculate all components
    all_text = " ".join(m.text for m in messages if m.text)
    sentiment = analyze_sentiment(all_text)
    response_patterns = analyze_response_patterns(messages)
    frequency_trends = analyze_frequency_trends(messages)

    health = calculate_relationship_health(
        messages, sentiment, response_patterns, frequency_trends
    )

    return RelationshipHealthResponse(
        overall_score=health.overall_score,
        engagement_score=health.engagement_score,
        sentiment_score=health.sentiment_score,
        responsiveness_score=health.responsiveness_score,
        consistency_score=health.consistency_score,
        health_label=health.health_label,
        factors=health.factors,
    )


@router.delete(
    "/{chat_id}/cache",
    response_description="Cache invalidation confirmation",
    summary="Invalidate insights cache for a conversation",
)
def invalidate_insights_cache(
    chat_id: str,
) -> dict[str, str]:
    """Invalidate cached insights for a conversation.

    Use this endpoint to force a refresh of insights on the next request.

    Args:
        chat_id: The conversation identifier

    Returns:
        Confirmation message
    """
    cache = get_insights_cache()

    # Invalidate all cache entries for this chat_id
    for time_range in TimeRangeEnum:
        for limit in [50, 100, 200, 500, 1000, 2000, 5000]:
            cache_key = f"insights:{chat_id}:{time_range.value}:{limit}"
            cache.invalidate(cache_key)

    return {"status": "ok", "message": f"Insights cache invalidated for {chat_id}"}
