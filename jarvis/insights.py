"""Conversation insights analytics module.

Provides sentiment analysis, response time patterns, message frequency trends,
and relationship health scoring for iMessage conversations.

Uses a lightweight lexicon-based approach for sentiment analysis to maintain
performance on local systems without requiring heavy ML models.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.imessage import Message


# Sentiment lexicon for lightweight analysis
# Positive words with weights (1.0 = strong positive)
POSITIVE_WORDS: dict[str, float] = {
    # Strong positive
    "love": 1.0,
    "amazing": 1.0,
    "awesome": 1.0,
    "wonderful": 1.0,
    "fantastic": 1.0,
    "excellent": 1.0,
    "perfect": 1.0,
    "beautiful": 1.0,
    "incredible": 1.0,
    "brilliant": 0.9,
    # Medium positive
    "great": 0.8,
    "good": 0.6,
    "nice": 0.6,
    "happy": 0.8,
    "glad": 0.7,
    "excited": 0.8,
    "thrilled": 0.9,
    "pleased": 0.7,
    "delighted": 0.9,
    "grateful": 0.8,
    "thankful": 0.8,
    "appreciate": 0.7,
    "thanks": 0.5,
    "thank": 0.5,
    # Mild positive
    "like": 0.4,
    "enjoy": 0.6,
    "fun": 0.6,
    "cool": 0.5,
    "sweet": 0.6,
    "cute": 0.5,
    "interesting": 0.4,
    "helpful": 0.6,
    "kind": 0.6,
    "friendly": 0.6,
    "welcome": 0.5,
    "sure": 0.3,
    "yes": 0.3,
    "okay": 0.2,
    "ok": 0.2,
    "fine": 0.2,
    "agreed": 0.4,
    "definitely": 0.5,
    "absolutely": 0.6,
    "exactly": 0.4,
    "congrats": 0.8,
    "congratulations": 0.8,
    "proud": 0.7,
    "miss": 0.5,
}

# Negative words with weights (-1.0 = strong negative)
NEGATIVE_WORDS: dict[str, float] = {
    # Strong negative
    "hate": -1.0,
    "terrible": -1.0,
    "horrible": -1.0,
    "awful": -1.0,
    "disgusting": -1.0,
    "worst": -1.0,
    # Medium negative
    "bad": -0.7,
    "sad": -0.7,
    "angry": -0.8,
    "upset": -0.7,
    "disappointed": -0.7,
    "frustrated": -0.7,
    "annoyed": -0.6,
    "annoying": -0.6,
    "boring": -0.5,
    "tired": -0.4,
    "sick": -0.5,
    "worried": -0.6,
    "anxious": -0.6,
    "stressed": -0.6,
    "sorry": -0.3,
    "unfortunately": -0.5,
    "problem": -0.4,
    "issue": -0.3,
    "wrong": -0.5,
    "mistake": -0.5,
    "fail": -0.6,
    "failed": -0.6,
    # Mild negative
    "no": -0.2,
    "not": -0.2,
    "never": -0.4,
    "cant": -0.3,
    "cannot": -0.3,
    "wont": -0.3,
    "dont": -0.2,
    "busy": -0.3,
    "late": -0.3,
    "cancel": -0.4,
    "cancelled": -0.4,
}

# Positive emojis
POSITIVE_EMOJIS = frozenset(
    [
        "😀",
        "😁",
        "😂",
        "🤣",
        "😃",
        "😄",
        "😅",
        "😆",
        "😊",
        "😇",
        "🙂",
        "😉",
        "😌",
        "😍",
        "🥰",
        "😘",
        "😗",
        "😙",
        "😚",
        "🥲",
        "😋",
        "😛",
        "😜",
        "🤪",
        "😝",
        "🤗",
        "🤩",
        "🥳",
        "❤️",
        "🧡",
        "💛",
        "💚",
        "💙",
        "💜",
        "🖤",
        "🤍",
        "🤎",
        "💖",
        "💗",
        "💓",
        "💕",
        "💞",
        "💘",
        "💝",
        "👍",
        "👏",
        "🙌",
        "🎉",
        "🎊",
        "✨",
        "⭐",
        "🌟",
        "💯",
        "🔥",
        "👌",
        "✅",
        "💪",
    ]
)

# Negative emojis
NEGATIVE_EMOJIS = frozenset(
    [
        "😢",
        "😭",
        "😤",
        "😠",
        "😡",
        "🤬",
        "😈",
        "👿",
        "💀",
        "☠️",
        "😰",
        "😥",
        "😓",
        "😩",
        "😫",
        "🥺",
        "😖",
        "😣",
        "😞",
        "😔",
        "😟",
        "😕",
        "🙁",
        "☹️",
        "😬",
        "😵",
        "🤢",
        "🤮",
        "💔",
        "👎",
        "😱",
        "😨",
        "😰",
        "🥵",
        "🥶",
        "😳",
        "🤯",
        "😭",
    ]
)

# Emoji pattern for extraction
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001faff"  # Symbols Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class SentimentScore:
    """Sentiment score for a message or time period."""

    score: float  # -1.0 to 1.0 (negative to positive)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    @property
    def label(self) -> str:
        """Human-readable sentiment label."""
        if self.score >= 0.3:
            return "positive"
        elif self.score <= -0.3:
            return "negative"
        return "neutral"


@dataclass
class SentimentTrend:
    """Sentiment analysis over time for a contact."""

    date: str  # ISO format date (YYYY-MM-DD)
    score: float  # Average sentiment for the period
    message_count: int


@dataclass
class ResponseTimePattern:
    """Response time patterns for a contact."""

    avg_response_time_minutes: float | None
    median_response_time_minutes: float | None
    fastest_response_minutes: float | None
    slowest_response_minutes: float | None
    response_times_by_hour: dict[int, float]  # hour -> avg response time
    response_times_by_day: dict[str, float]  # day name -> avg response time
    my_avg_response_time_minutes: float | None
    their_avg_response_time_minutes: float | None


@dataclass
class MessageFrequencyTrends:
    """Message frequency trends over time."""

    daily_counts: dict[str, int]  # ISO date -> count
    weekly_counts: dict[str, int]  # ISO week (YYYY-WNN) -> count
    monthly_counts: dict[str, int]  # YYYY-MM -> count
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_percentage: float  # percentage change over period
    most_active_day: str | None  # day of week
    most_active_hour: int | None  # hour (0-23)
    messages_per_day_avg: float


@dataclass
class RelationshipHealthScore:
    """Relationship health metrics based on conversation patterns."""

    overall_score: float  # 0-100
    engagement_score: float  # Based on message frequency and balance
    sentiment_score: float  # Based on overall sentiment
    responsiveness_score: float  # Based on response times
    consistency_score: float  # Based on regular communication
    factors: dict[str, str]  # Contributing factors with descriptions

    @property
    def health_label(self) -> str:
        """Human-readable health label."""
        if self.overall_score >= 80:
            return "excellent"
        elif self.overall_score >= 60:
            return "good"
        elif self.overall_score >= 40:
            return "fair"
        elif self.overall_score >= 20:
            return "needs_attention"
        return "concerning"


@dataclass
class ConversationInsights:
    """Complete insights for a conversation."""

    chat_id: str
    contact_name: str | None
    time_range: str
    sentiment_overall: SentimentScore
    sentiment_trends: list[SentimentTrend]
    response_patterns: ResponseTimePattern
    frequency_trends: MessageFrequencyTrends
    relationship_health: RelationshipHealthScore
    total_messages_analyzed: int
    first_message_date: str | None
    last_message_date: str | None


def analyze_sentiment(text: str | None) -> SentimentScore:
    """Analyze sentiment of a single message using lexicon-based approach.

    Args:
        text: Message text to analyze

    Returns:
        SentimentScore with score between -1.0 and 1.0
    """
    if not text:
        return SentimentScore(score=0.0, neutral_count=1)

    text_lower = text.lower()

    # Extract words (simple tokenization)
    words = re.findall(r"\b[a-zA-Z\']+\b", text_lower)

    # Calculate word-based sentiment
    positive_sum = 0.0
    negative_sum = 0.0
    positive_count = 0
    negative_count = 0

    for word in words:
        # Remove apostrophes for matching
        clean_word = word.replace("'", "")
        if clean_word in POSITIVE_WORDS:
            positive_sum += POSITIVE_WORDS[clean_word]
            positive_count += 1
        elif clean_word in NEGATIVE_WORDS:
            negative_sum += abs(NEGATIVE_WORDS[clean_word])
            negative_count += 1

    # Extract and analyze emojis
    emojis = EMOJI_PATTERN.findall(text)
    for emoji_group in emojis:
        for char in emoji_group:
            if char in POSITIVE_EMOJIS:
                positive_sum += 0.5
                positive_count += 1
            elif char in NEGATIVE_EMOJIS:
                negative_sum += 0.5
                negative_count += 1

    # Calculate overall score
    total_signals = positive_count + negative_count
    if total_signals == 0:
        return SentimentScore(score=0.0, neutral_count=1)

    # Normalize: score ranges from -1 to 1
    score = (positive_sum - negative_sum) / (positive_sum + negative_sum + 1)
    score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

    return SentimentScore(
        score=round(score, 3),
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=0 if (positive_count + negative_count) > 0 else 1,
    )


def analyze_sentiment_trends(
    messages: list[Message],
    granularity: str = "day",
) -> list[SentimentTrend]:
    """Analyze sentiment trends over time.

    Args:
        messages: List of messages sorted by date
        granularity: "day", "week", or "month"

    Returns:
        List of SentimentTrend objects
    """
    if not messages:
        return []

    # Group messages by time period
    period_sentiments: dict[str, list[float]] = defaultdict(list)

    for msg in messages:
        if not msg.text:
            continue

        # Get period key based on granularity
        date = msg.date
        if granularity == "day":
            period_key = date.strftime("%Y-%m-%d")
        elif granularity == "week":
            period_key = date.strftime("%Y-W%W")
        else:  # month
            period_key = date.strftime("%Y-%m")

        sentiment = analyze_sentiment(msg.text)
        period_sentiments[period_key].append(sentiment.score)

    # Calculate average sentiment per period
    trends = []
    for period, scores in sorted(period_sentiments.items()):
        avg_score = sum(scores) / len(scores) if scores else 0.0
        trends.append(
            SentimentTrend(
                date=period,
                score=round(avg_score, 3),
                message_count=len(scores),
            )
        )

    return trends


def analyze_response_patterns(messages: list[Message]) -> ResponseTimePattern:
    """Analyze response time patterns between participants.

    Args:
        messages: List of messages sorted by date (oldest first)

    Returns:
        ResponseTimePattern with detailed timing analysis
    """
    if len(messages) < 2:
        return ResponseTimePattern(
            avg_response_time_minutes=None,
            median_response_time_minutes=None,
            fastest_response_minutes=None,
            slowest_response_minutes=None,
            response_times_by_hour={},
            response_times_by_day={},
            my_avg_response_time_minutes=None,
            their_avg_response_time_minutes=None,
        )

    # Track response times
    all_response_times: list[float] = []
    my_response_times: list[float] = []
    their_response_times: list[float] = []
    response_by_hour: dict[int, list[float]] = defaultdict(list)
    response_by_day: dict[str, list[float]] = defaultdict(list)

    prev_msg = None
    for msg in messages:
        if prev_msg is not None and msg.is_from_me != prev_msg.is_from_me:
            # This is a response
            time_diff = (msg.date - prev_msg.date).total_seconds()

            # Only count responses within 24 hours
            if 0 < time_diff < 86400:
                response_minutes = time_diff / 60.0
                all_response_times.append(response_minutes)

                # Track by sender
                if msg.is_from_me:
                    my_response_times.append(response_minutes)
                else:
                    their_response_times.append(response_minutes)

                # Track by hour and day of the previous message
                hour = prev_msg.date.hour
                day_name = prev_msg.date.strftime("%A")
                response_by_hour[hour].append(response_minutes)
                response_by_day[day_name].append(response_minutes)

        prev_msg = msg

    # Calculate statistics
    def safe_avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 1) if values else None

    def safe_median(values: list[float]) -> float | None:
        if not values:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2, 1)
        return round(sorted_vals[mid], 1)

    return ResponseTimePattern(
        avg_response_time_minutes=safe_avg(all_response_times),
        median_response_time_minutes=safe_median(all_response_times),
        fastest_response_minutes=round(min(all_response_times), 1) if all_response_times else None,
        slowest_response_minutes=round(max(all_response_times), 1) if all_response_times else None,
        response_times_by_hour={h: round(sum(t) / len(t), 1) for h, t in response_by_hour.items()},
        response_times_by_day={d: round(sum(t) / len(t), 1) for d, t in response_by_day.items()},
        my_avg_response_time_minutes=safe_avg(my_response_times),
        their_avg_response_time_minutes=safe_avg(their_response_times),
    )


def analyze_frequency_trends(messages: list[Message]) -> MessageFrequencyTrends:
    """Analyze message frequency trends over time.

    Args:
        messages: List of messages

    Returns:
        MessageFrequencyTrends with daily/weekly/monthly breakdowns
    """
    if not messages:
        return MessageFrequencyTrends(
            daily_counts={},
            weekly_counts={},
            monthly_counts={},
            trend_direction="stable",
            trend_percentage=0.0,
            most_active_day=None,
            most_active_hour=None,
            messages_per_day_avg=0.0,
        )

    # Count by different time periods
    daily_counts: Counter[str] = Counter()
    weekly_counts: Counter[str] = Counter()
    monthly_counts: Counter[str] = Counter()
    day_of_week_counts: Counter[str] = Counter()
    hour_counts: Counter[int] = Counter()

    for msg in messages:
        date = msg.date
        daily_counts[date.strftime("%Y-%m-%d")] += 1
        weekly_counts[date.strftime("%Y-W%W")] += 1
        monthly_counts[date.strftime("%Y-%m")] += 1
        day_of_week_counts[date.strftime("%A")] += 1
        hour_counts[date.hour] += 1

    # Calculate trend direction (compare first half to second half)
    sorted_daily = sorted(daily_counts.items())
    if len(sorted_daily) > 1:
        mid = len(sorted_daily) // 2
        first_half_avg = sum(c for _, c in sorted_daily[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(c for _, c in sorted_daily[mid:]) / (len(sorted_daily) - mid)

        if first_half_avg == 0:
            trend_pct = 100.0 if second_half_avg > 0 else 0.0
        else:
            trend_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100

        if trend_pct > 10:
            trend_direction = "increasing"
        elif trend_pct < -10:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "stable"
        trend_pct = 0.0

    # Find most active times
    most_active_day = day_of_week_counts.most_common(1)[0][0] if day_of_week_counts else None
    most_active_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

    # Calculate messages per day average
    if daily_counts:
        total_days = len(daily_counts)
        total_messages = sum(daily_counts.values())
        messages_per_day = total_messages / total_days
    else:
        messages_per_day = 0.0

    return MessageFrequencyTrends(
        daily_counts=dict(sorted_daily),
        weekly_counts=dict(sorted(weekly_counts.items())),
        monthly_counts=dict(sorted(monthly_counts.items())),
        trend_direction=trend_direction,
        trend_percentage=round(trend_pct, 1),
        most_active_day=most_active_day,
        most_active_hour=most_active_hour,
        messages_per_day_avg=round(messages_per_day, 2),
    )


def calculate_relationship_health(
    messages: list[Message],
    sentiment_score: SentimentScore,
    response_patterns: ResponseTimePattern,
    frequency_trends: MessageFrequencyTrends,
) -> RelationshipHealthScore:
    """Calculate relationship health score based on multiple factors.

    Args:
        messages: List of messages
        sentiment_score: Overall sentiment analysis
        response_patterns: Response time analysis
        frequency_trends: Message frequency analysis

    Returns:
        RelationshipHealthScore with detailed breakdown
    """
    factors: dict[str, str] = {}
    scores: dict[str, float] = {}

    # 1. Engagement score (0-100) - based on message balance and frequency
    if messages:
        sent_count = sum(1 for m in messages if m.is_from_me)
        received_count = len(messages) - sent_count
        total = len(messages)

        # Balance: closer to 50/50 is better
        if total > 0:
            balance_ratio = min(sent_count, received_count) / max(sent_count, received_count, 1)
            balance_score = balance_ratio * 100

            # Frequency bonus
            freq_score = min(frequency_trends.messages_per_day_avg * 10, 100)

            engagement = (balance_score * 0.6 + freq_score * 0.4)
            scores["engagement"] = engagement

            if balance_ratio >= 0.7:
                factors["engagement"] = "Balanced conversation with good message exchange"
            elif balance_ratio >= 0.4:
                factors["engagement"] = "Moderate conversation balance"
            else:
                factors["engagement"] = "Unbalanced conversation - one person messages more"
        else:
            scores["engagement"] = 0
            factors["engagement"] = "No messages to analyze"
    else:
        scores["engagement"] = 0
        factors["engagement"] = "No messages to analyze"

    # 2. Sentiment score (0-100) - convert from -1 to 1 range
    sentiment_normalized = (sentiment_score.score + 1) / 2 * 100
    scores["sentiment"] = sentiment_normalized

    if sentiment_score.score >= 0.3:
        factors["sentiment"] = "Predominantly positive communication"
    elif sentiment_score.score >= 0:
        factors["sentiment"] = "Neutral to mildly positive tone"
    elif sentiment_score.score >= -0.3:
        factors["sentiment"] = "Neutral to mildly negative tone"
    else:
        factors["sentiment"] = "Frequently negative communication"

    # 3. Responsiveness score (0-100) - based on response times
    if response_patterns.avg_response_time_minutes is not None:
        avg_response = response_patterns.avg_response_time_minutes

        # Score decreases as response time increases
        # < 5 min = 100, < 30 min = 80, < 2 hours = 60, < 8 hours = 40, else lower
        if avg_response <= 5:
            responsiveness = 100
            factors["responsiveness"] = "Very quick responses"
        elif avg_response <= 30:
            responsiveness = 80
            factors["responsiveness"] = "Good response time"
        elif avg_response <= 120:
            responsiveness = 60
            factors["responsiveness"] = "Moderate response time"
        elif avg_response <= 480:
            responsiveness = 40
            factors["responsiveness"] = "Slow response time"
        else:
            responsiveness = 20
            factors["responsiveness"] = "Very slow responses"

        scores["responsiveness"] = responsiveness
    else:
        scores["responsiveness"] = 50  # Neutral if unknown
        factors["responsiveness"] = "Insufficient data for response time analysis"

    # 4. Consistency score (0-100) - based on regular communication
    if frequency_trends.daily_counts:
        # Check for gaps in communication
        dates = sorted(frequency_trends.daily_counts.keys())
        if len(dates) > 1:
            first_date = datetime.strptime(dates[0], "%Y-%m-%d")
            last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
            total_days = (last_date - first_date).days + 1
            active_days = len(dates)

            # Higher ratio = more consistent
            consistency_ratio = active_days / total_days if total_days > 0 else 0
            consistency = consistency_ratio * 100

            if consistency_ratio >= 0.7:
                factors["consistency"] = "Very consistent communication"
            elif consistency_ratio >= 0.4:
                factors["consistency"] = "Moderately consistent communication"
            elif consistency_ratio >= 0.2:
                factors["consistency"] = "Sporadic communication"
            else:
                factors["consistency"] = "Infrequent communication with long gaps"

            scores["consistency"] = consistency
        else:
            scores["consistency"] = 50
            factors["consistency"] = "Not enough data to determine consistency"
    else:
        scores["consistency"] = 0
        factors["consistency"] = "No communication data"

    # Calculate overall score (weighted average)
    weights = {
        "engagement": 0.30,
        "sentiment": 0.25,
        "responsiveness": 0.25,
        "consistency": 0.20,
    }

    overall = sum(scores[k] * weights[k] for k in weights)

    return RelationshipHealthScore(
        overall_score=round(overall, 1),
        engagement_score=round(scores.get("engagement", 0), 1),
        sentiment_score=round(scores.get("sentiment", 0), 1),
        responsiveness_score=round(scores.get("responsiveness", 0), 1),
        consistency_score=round(scores.get("consistency", 0), 1),
        factors=factors,
    )


def generate_conversation_insights(
    chat_id: str,
    messages: list[Message],
    contact_name: str | None = None,
    time_range: str = "month",
) -> ConversationInsights:
    """Generate complete insights for a conversation.

    Args:
        chat_id: Conversation identifier
        messages: List of messages (should be sorted chronologically)
        contact_name: Display name for the contact
        time_range: Time range label (for display)

    Returns:
        ConversationInsights with all analytics
    """
    # Sort messages chronologically
    sorted_messages = sorted(messages, key=lambda m: m.date)

    # Overall sentiment
    all_text = " ".join(m.text for m in sorted_messages if m.text)
    sentiment_overall = analyze_sentiment(all_text)

    # Sentiment trends (weekly for better visualization)
    sentiment_trends = analyze_sentiment_trends(sorted_messages, granularity="week")

    # Response patterns
    response_patterns = analyze_response_patterns(sorted_messages)

    # Frequency trends
    frequency_trends = analyze_frequency_trends(sorted_messages)

    # Relationship health
    relationship_health = calculate_relationship_health(
        sorted_messages,
        sentiment_overall,
        response_patterns,
        frequency_trends,
    )

    # Date range
    first_date = sorted_messages[0].date.isoformat() if sorted_messages else None
    last_date = sorted_messages[-1].date.isoformat() if sorted_messages else None

    return ConversationInsights(
        chat_id=chat_id,
        contact_name=contact_name,
        time_range=time_range,
        sentiment_overall=sentiment_overall,
        sentiment_trends=sentiment_trends,
        response_patterns=response_patterns,
        frequency_trends=frequency_trends,
        relationship_health=relationship_health,
        total_messages_analyzed=len(sorted_messages),
        first_message_date=first_date,
        last_message_date=last_date,
    )
