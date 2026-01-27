"""Tests for the conversation insights module.

Tests sentiment analysis, response time patterns, message frequency trends,
and relationship health score calculations.
"""

from datetime import datetime, timedelta

from contracts.imessage import Message
from jarvis.insights import (
    SentimentScore,
    analyze_frequency_trends,
    analyze_response_patterns,
    analyze_sentiment,
    analyze_sentiment_trends,
    calculate_relationship_health,
    generate_conversation_insights,
)


def create_message(
    text: str,
    is_from_me: bool = False,
    date: datetime | None = None,
    msg_id: int = 1,
) -> Message:
    """Helper to create a Message object for testing."""
    if date is None:
        date = datetime.now()
    return Message(
        id=msg_id,
        chat_id="test_chat",
        sender="+15551234567" if not is_from_me else "me",
        sender_name="Test User" if not is_from_me else None,
        text=text,
        date=date,
        is_from_me=is_from_me,
        attachments=[],
        reply_to_id=None,
        reactions=[],
    )


class TestAnalyzeSentiment:
    """Tests for analyze_sentiment function."""

    def test_empty_text_returns_neutral(self):
        """Empty text returns neutral sentiment."""
        result = analyze_sentiment("")
        assert result.score == 0.0
        assert result.label == "neutral"
        assert result.neutral_count == 1

    def test_none_text_returns_neutral(self):
        """None text returns neutral sentiment."""
        result = analyze_sentiment(None)
        assert result.score == 0.0
        assert result.label == "neutral"

    def test_positive_text_detected(self):
        """Positive words are detected and scored."""
        result = analyze_sentiment("I love this! It's amazing and wonderful.")
        assert result.score > 0
        assert result.label == "positive"
        assert result.positive_count >= 3  # love, amazing, wonderful

    def test_negative_text_detected(self):
        """Negative words are detected and scored."""
        result = analyze_sentiment("I hate this. It's terrible and awful.")
        assert result.score < 0
        assert result.label == "negative"
        assert result.negative_count >= 3  # hate, terrible, awful

    def test_neutral_text_detected(self):
        """Neutral text returns neutral label."""
        result = analyze_sentiment("The meeting is scheduled for tomorrow at 3pm.")
        assert -0.3 <= result.score <= 0.3
        assert result.label == "neutral"

    def test_mixed_sentiment(self):
        """Mixed positive and negative words are handled."""
        result = analyze_sentiment("I love the food but hate the service.")
        # Should be somewhere in the middle
        assert -0.5 <= result.score <= 0.5
        assert result.positive_count >= 1  # love
        assert result.negative_count >= 1  # hate

    def test_positive_emoji_detected(self):
        """Positive emojis contribute to positive sentiment."""
        result = analyze_sentiment("That's great! ❤️ 😊 👍")
        assert result.score > 0
        assert result.positive_count >= 2  # emojis

    def test_negative_emoji_detected(self):
        """Negative emojis contribute to negative sentiment."""
        result = analyze_sentiment("This is so sad 😢 😭 💔")
        assert result.score < 0
        assert result.negative_count >= 2  # emojis

    def test_case_insensitive(self):
        """Sentiment analysis is case insensitive."""
        result1 = analyze_sentiment("I LOVE this!")
        result2 = analyze_sentiment("i love this!")
        assert result1.positive_count == result2.positive_count

    def test_score_clamped_to_range(self):
        """Score is always between -1 and 1."""
        result = analyze_sentiment("love love love love love amazing wonderful great")
        assert -1.0 <= result.score <= 1.0

        result = analyze_sentiment("hate hate hate hate terrible awful horrible")
        assert -1.0 <= result.score <= 1.0


class TestAnalyzeSentimentTrends:
    """Tests for analyze_sentiment_trends function."""

    def test_empty_messages_returns_empty(self):
        """Empty message list returns empty trends."""
        result = analyze_sentiment_trends([])
        assert result == []

    def test_single_day_trend(self):
        """Single day creates single trend entry."""
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("I love this!", date=now),
            create_message("This is great!", date=now + timedelta(hours=1)),
        ]
        result = analyze_sentiment_trends(messages, granularity="day")
        assert len(result) == 1
        assert result[0].message_count == 2

    def test_multiple_days_create_multiple_trends(self):
        """Multiple days create multiple trend entries."""
        now = datetime.now()
        messages = [
            create_message("Good day!", date=now),
            create_message("Great day!", date=now + timedelta(days=1)),
            create_message("Wonderful!", date=now + timedelta(days=2)),
        ]
        result = analyze_sentiment_trends(messages, granularity="day")
        assert len(result) == 3

    def test_weekly_granularity(self):
        """Weekly granularity groups by week."""
        now = datetime.now()
        messages = [
            create_message("Message 1", date=now),
            create_message("Message 2", date=now + timedelta(days=1)),
        ]
        result = analyze_sentiment_trends(messages, granularity="week")
        # Both messages should be in the same week
        assert len(result) == 1
        assert result[0].message_count == 2

    def test_monthly_granularity(self):
        """Monthly granularity groups by month."""
        now = datetime.now()
        messages = [
            create_message("Message 1", date=now),
            create_message("Message 2", date=now + timedelta(days=5)),
        ]
        result = analyze_sentiment_trends(messages, granularity="month")
        assert len(result) == 1

    def test_messages_without_text_skipped(self):
        """Messages without text are skipped."""
        now = datetime.now()
        messages = [
            create_message("Hello!", date=now),
            create_message("", date=now + timedelta(hours=1)),  # Empty
        ]
        result = analyze_sentiment_trends(messages, granularity="day")
        assert len(result) == 1
        assert result[0].message_count == 1

    def test_sentiment_averaged_per_period(self):
        """Sentiment scores are averaged per period."""
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("I love this!", date=now),  # Positive
            create_message("I hate this!", date=now + timedelta(hours=1)),  # Negative
        ]
        result = analyze_sentiment_trends(messages, granularity="day")
        assert len(result) == 1
        # Average of positive and negative should be closer to 0
        assert -0.5 <= result[0].score <= 0.5


class TestAnalyzeResponsePatterns:
    """Tests for analyze_response_patterns function."""

    def test_empty_messages_returns_none_values(self):
        """Empty message list returns None for all values."""
        result = analyze_response_patterns([])
        assert result.avg_response_time_minutes is None
        assert result.median_response_time_minutes is None

    def test_single_message_returns_none_values(self):
        """Single message returns None for response times."""
        result = analyze_response_patterns([create_message("Hello")])
        assert result.avg_response_time_minutes is None

    def test_calculates_response_time(self):
        """Response time is calculated between sender switches."""
        now = datetime.now()
        messages = [
            create_message("Hello", is_from_me=False, date=now),
            create_message("Hi there", is_from_me=True, date=now + timedelta(minutes=5)),
        ]
        result = analyze_response_patterns(messages)
        assert result.avg_response_time_minutes == 5.0

    def test_tracks_my_vs_their_response(self):
        """Tracks response times separately for me and them."""
        now = datetime.now()
        messages = [
            create_message("Hello", is_from_me=False, date=now),
            create_message("Hi", is_from_me=True, date=now + timedelta(minutes=5)),
            create_message("How are you?", is_from_me=False, date=now + timedelta(minutes=15)),
        ]
        result = analyze_response_patterns(messages)
        assert result.my_avg_response_time_minutes == 5.0
        assert result.their_avg_response_time_minutes == 10.0

    def test_ignores_responses_over_24h(self):
        """Responses over 24 hours are ignored."""
        now = datetime.now()
        messages = [
            create_message("Hello", is_from_me=False, date=now),
            create_message("Hi", is_from_me=True, date=now + timedelta(hours=25)),
        ]
        result = analyze_response_patterns(messages)
        assert result.avg_response_time_minutes is None

    def test_ignores_consecutive_same_sender(self):
        """Consecutive messages from same sender don't count as responses."""
        now = datetime.now()
        messages = [
            create_message("Hello", is_from_me=False, date=now),
            create_message("Are you there?", is_from_me=False, date=now + timedelta(minutes=1)),
            create_message("Yes", is_from_me=True, date=now + timedelta(minutes=10)),
        ]
        result = analyze_response_patterns(messages)
        # Response time is from the last message before sender switch
        # So it's 9 minutes (from +1min to +10min)
        assert result.avg_response_time_minutes == 9.0

    def test_median_calculation(self):
        """Median is calculated correctly."""
        now = datetime.now()
        messages = [
            create_message("1", is_from_me=False, date=now),
            create_message("2", is_from_me=True, date=now + timedelta(minutes=5)),
            create_message("3", is_from_me=False, date=now + timedelta(minutes=10)),
            create_message("4", is_from_me=True, date=now + timedelta(minutes=15)),
            create_message("5", is_from_me=False, date=now + timedelta(minutes=30)),
        ]
        result = analyze_response_patterns(messages)
        # Response times: 5, 5, 15 minutes
        assert result.median_response_time_minutes == 5.0

    def test_fastest_slowest_response(self):
        """Fastest and slowest response times are tracked."""
        now = datetime.now()
        messages = [
            create_message("1", is_from_me=False, date=now),
            create_message("2", is_from_me=True, date=now + timedelta(minutes=2)),
            create_message("3", is_from_me=False, date=now + timedelta(minutes=10)),
            create_message("4", is_from_me=True, date=now + timedelta(minutes=30)),
        ]
        result = analyze_response_patterns(messages)
        assert result.fastest_response_minutes == 2.0
        assert result.slowest_response_minutes == 20.0


class TestAnalyzeFrequencyTrends:
    """Tests for analyze_frequency_trends function."""

    def test_empty_messages_returns_defaults(self):
        """Empty message list returns default values."""
        result = analyze_frequency_trends([])
        assert result.daily_counts == {}
        assert result.trend_direction == "stable"
        assert result.messages_per_day_avg == 0.0

    def test_daily_counts_tracked(self):
        """Daily message counts are tracked."""
        now = datetime.now()
        messages = [
            create_message("1", date=now),
            create_message("2", date=now + timedelta(hours=1)),
            create_message("3", date=now + timedelta(days=1)),
        ]
        result = analyze_frequency_trends(messages)
        assert len(result.daily_counts) == 2

    def test_weekly_counts_tracked(self):
        """Weekly message counts are tracked."""
        now = datetime.now()
        messages = [
            create_message("1", date=now),
            create_message("2", date=now + timedelta(days=1)),
        ]
        result = analyze_frequency_trends(messages)
        assert len(result.weekly_counts) >= 1

    def test_monthly_counts_tracked(self):
        """Monthly message counts are tracked."""
        now = datetime.now()
        messages = [
            create_message("1", date=now),
            create_message("2", date=now + timedelta(days=1)),
        ]
        result = analyze_frequency_trends(messages)
        assert len(result.monthly_counts) >= 1

    def test_increasing_trend_detected(self):
        """Increasing message frequency is detected."""
        now = datetime.now()
        messages = []
        # First half: 1 message per day
        for i in range(5):
            messages.append(create_message(f"msg{i}", date=now + timedelta(days=i)))
        # Second half: 3 messages per day
        for i in range(5, 10):
            for j in range(3):
                messages.append(
                    create_message(f"msg{i}_{j}", date=now + timedelta(days=i, hours=j))
                )
        result = analyze_frequency_trends(messages)
        assert result.trend_direction == "increasing"
        assert result.trend_percentage > 10

    def test_decreasing_trend_detected(self):
        """Decreasing message frequency is detected."""
        now = datetime.now()
        messages = []
        # First half: 3 messages per day
        for i in range(5):
            for j in range(3):
                messages.append(
                    create_message(f"msg{i}_{j}", date=now + timedelta(days=i, hours=j))
                )
        # Second half: 1 message per day
        for i in range(5, 10):
            messages.append(create_message(f"msg{i}", date=now + timedelta(days=i)))
        result = analyze_frequency_trends(messages)
        assert result.trend_direction == "decreasing"
        assert result.trend_percentage < -10

    def test_stable_trend_detected(self):
        """Stable message frequency is detected."""
        now = datetime.now()
        messages = []
        # Constant 2 messages per day
        for i in range(10):
            for j in range(2):
                messages.append(
                    create_message(f"msg{i}_{j}", date=now + timedelta(days=i, hours=j))
                )
        result = analyze_frequency_trends(messages)
        assert result.trend_direction == "stable"
        assert -10 <= result.trend_percentage <= 10

    def test_most_active_day_detected(self):
        """Most active day of week is detected."""
        now = datetime.now()
        # Find the next Saturday and reset to midnight
        days_until_saturday = (5 - now.weekday()) % 7
        saturday = (now + timedelta(days=days_until_saturday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        messages = []
        # Add 5 messages on Saturday (spread across different hours)
        for i in range(5):
            messages.append(create_message(f"sat{i}", date=saturday + timedelta(hours=i)))
        # Add 1 message on other days
        for i in range(1, 6):
            messages.append(create_message(f"other{i}", date=saturday + timedelta(days=i)))
        result = analyze_frequency_trends(messages)
        assert result.most_active_day == "Saturday"

    def test_most_active_hour_detected(self):
        """Most active hour is detected."""
        now = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
        messages = []
        # Add 5 messages at 2pm
        for i in range(5):
            messages.append(create_message(f"msg{i}", date=now + timedelta(days=i)))
        # Add 1 message at other hours
        for h in [9, 10, 11]:
            messages.append(create_message(f"h{h}", date=now.replace(hour=h)))
        result = analyze_frequency_trends(messages)
        assert result.most_active_hour == 14

    def test_messages_per_day_average(self):
        """Messages per day average is calculated."""
        now = datetime.now()
        messages = [
            create_message("1", date=now),
            create_message("2", date=now + timedelta(hours=1)),
            create_message("3", date=now + timedelta(days=1)),
        ]
        result = analyze_frequency_trends(messages)
        # 3 messages over 2 days = 1.5
        assert result.messages_per_day_avg == 1.5


class TestCalculateRelationshipHealth:
    """Tests for calculate_relationship_health function."""

    def test_empty_messages_returns_low_score(self):
        """Empty message list returns low scores."""
        sentiment = SentimentScore(score=0.0)
        response = analyze_response_patterns([])
        frequency = analyze_frequency_trends([])

        result = calculate_relationship_health([], sentiment, response, frequency)
        assert result.overall_score < 50
        assert result.engagement_score == 0

    def test_balanced_conversation_high_engagement(self):
        """Balanced conversation gets high engagement score."""
        now = datetime.now()
        messages = []
        for i in range(10):
            msg_date = now + timedelta(hours=i * 2)
            messages.append(create_message(f"them{i}", is_from_me=False, date=msg_date))
            reply_date = now + timedelta(hours=i * 2 + 1)
            messages.append(create_message(f"me{i}", is_from_me=True, date=reply_date))

        sentiment = SentimentScore(score=0.5, positive_count=10)
        response = analyze_response_patterns(messages)
        frequency = analyze_frequency_trends(messages)

        result = calculate_relationship_health(messages, sentiment, response, frequency)
        assert result.engagement_score >= 60

    def test_positive_sentiment_high_sentiment_score(self):
        """Positive sentiment gives high sentiment score."""
        sentiment = SentimentScore(score=0.8, positive_count=20)
        response = analyze_response_patterns([])
        frequency = analyze_frequency_trends([])

        result = calculate_relationship_health([], sentiment, response, frequency)
        # Sentiment score: (0.8 + 1) / 2 * 100 = 90
        assert result.sentiment_score >= 80

    def test_negative_sentiment_low_sentiment_score(self):
        """Negative sentiment gives low sentiment score."""
        sentiment = SentimentScore(score=-0.8, negative_count=20)
        response = analyze_response_patterns([])
        frequency = analyze_frequency_trends([])

        result = calculate_relationship_health([], sentiment, response, frequency)
        # Sentiment score: (-0.8 + 1) / 2 * 100 = 10
        assert result.sentiment_score <= 20

    def test_fast_response_high_responsiveness(self):
        """Fast response times give high responsiveness score."""
        now = datetime.now()
        messages = [
            create_message("Hi", is_from_me=False, date=now),
            create_message("Hello", is_from_me=True, date=now + timedelta(minutes=3)),
        ]
        sentiment = SentimentScore(score=0.0)
        response = analyze_response_patterns(messages)
        frequency = analyze_frequency_trends(messages)

        result = calculate_relationship_health(messages, sentiment, response, frequency)
        assert result.responsiveness_score >= 80

    def test_slow_response_low_responsiveness(self):
        """Slow response times give low responsiveness score."""
        now = datetime.now()
        messages = [
            create_message("Hi", is_from_me=False, date=now),
            create_message("Hello", is_from_me=True, date=now + timedelta(hours=10)),
        ]
        sentiment = SentimentScore(score=0.0)
        response = analyze_response_patterns(messages)
        frequency = analyze_frequency_trends(messages)

        result = calculate_relationship_health(messages, sentiment, response, frequency)
        assert result.responsiveness_score <= 40

    def test_health_label_excellent(self):
        """High overall score gets 'excellent' label."""
        now = datetime.now()
        messages = []
        for i in range(30):
            msg_date = now + timedelta(days=i)
            messages.append(create_message(f"them{i}", is_from_me=False, date=msg_date))
            reply_date = now + timedelta(days=i, minutes=5)
            messages.append(create_message(f"me{i}", is_from_me=True, date=reply_date))

        sentiment = SentimentScore(score=0.8, positive_count=50)
        response = analyze_response_patterns(messages)
        frequency = analyze_frequency_trends(messages)

        result = calculate_relationship_health(messages, sentiment, response, frequency)
        assert result.health_label in ["excellent", "good"]

    def test_factors_populated(self):
        """Health factors are populated with descriptions."""
        now = datetime.now()
        messages = [
            create_message("Hi", is_from_me=False, date=now),
            create_message("Hello", is_from_me=True, date=now + timedelta(minutes=5)),
        ]
        sentiment = SentimentScore(score=0.5)
        response = analyze_response_patterns(messages)
        frequency = analyze_frequency_trends(messages)

        result = calculate_relationship_health(messages, sentiment, response, frequency)
        assert "engagement" in result.factors
        assert "sentiment" in result.factors
        assert "responsiveness" in result.factors
        assert "consistency" in result.factors


class TestGenerateConversationInsights:
    """Tests for generate_conversation_insights function."""

    def test_generates_complete_insights(self):
        """Generates complete insights object."""
        now = datetime.now()
        messages = [
            create_message("Hello!", is_from_me=False, date=now),
            create_message("Hi there!", is_from_me=True, date=now + timedelta(minutes=5)),
            create_message("How are you?", is_from_me=False, date=now + timedelta(minutes=10)),
        ]

        result = generate_conversation_insights(
            chat_id="test_chat",
            messages=messages,
            contact_name="Test User",
            time_range="month",
        )

        assert result.chat_id == "test_chat"
        assert result.contact_name == "Test User"
        assert result.time_range == "month"
        assert result.sentiment_overall is not None
        assert result.sentiment_trends is not None
        assert result.response_patterns is not None
        assert result.frequency_trends is not None
        assert result.relationship_health is not None
        assert result.total_messages_analyzed == 3

    def test_sorts_messages_chronologically(self):
        """Messages are sorted chronologically before analysis."""
        now = datetime.now()
        # Provide messages out of order
        messages = [
            create_message("Third", date=now + timedelta(hours=2)),
            create_message("First", date=now),
            create_message("Second", date=now + timedelta(hours=1)),
        ]

        result = generate_conversation_insights(
            chat_id="test_chat",
            messages=messages,
        )

        assert result.first_message_date is not None
        assert result.last_message_date is not None

    def test_date_range_captured(self):
        """First and last message dates are captured."""
        now = datetime.now()
        messages = [
            create_message("First", date=now),
            create_message("Last", date=now + timedelta(days=7)),
        ]

        result = generate_conversation_insights(
            chat_id="test_chat",
            messages=messages,
        )

        assert result.first_message_date is not None
        assert result.last_message_date is not None
        first_dt = datetime.fromisoformat(result.first_message_date)
        last_dt = datetime.fromisoformat(result.last_message_date)
        assert last_dt > first_dt

    def test_empty_messages_handled(self):
        """Empty message list is handled gracefully."""
        result = generate_conversation_insights(
            chat_id="test_chat",
            messages=[],
        )

        assert result.total_messages_analyzed == 0
        assert result.first_message_date is None
        assert result.last_message_date is None


class TestSentimentScore:
    """Tests for SentimentScore dataclass."""

    def test_positive_label(self):
        """Score >= 0.3 returns 'positive' label."""
        score = SentimentScore(score=0.5)
        assert score.label == "positive"

    def test_negative_label(self):
        """Score <= -0.3 returns 'negative' label."""
        score = SentimentScore(score=-0.5)
        assert score.label == "negative"

    def test_neutral_label(self):
        """Score between -0.3 and 0.3 returns 'neutral' label."""
        score = SentimentScore(score=0.1)
        assert score.label == "neutral"

        score = SentimentScore(score=-0.1)
        assert score.label == "neutral"

        score = SentimentScore(score=0.0)
        assert score.label == "neutral"

    def test_boundary_positive(self):
        """Score exactly 0.3 returns 'positive'."""
        score = SentimentScore(score=0.3)
        assert score.label == "positive"

    def test_boundary_negative(self):
        """Score exactly -0.3 returns 'negative'."""
        score = SentimentScore(score=-0.3)
        assert score.label == "negative"
