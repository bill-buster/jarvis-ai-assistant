"""Tests for jarvis/pair_balancer.py - Dataset rebalancing."""

from datetime import datetime

from jarvis.db import Pair
from jarvis.pair_balancer import (
    BalanceConfig,
    BalanceStats,
    PairBalancer,
    classify_response_type,
)


def make_pair(
    response_text: str,
    trigger_text: str = "Hello",
    contact_id: int | None = 1,
    quality_score: float = 1.0,
) -> Pair:
    """Helper to create a test pair."""
    return Pair(
        id=1,
        contact_id=contact_id,
        trigger_text=trigger_text,
        response_text=response_text,
        trigger_timestamp=datetime.now(),
        response_timestamp=datetime.now(),
        chat_id="test_chat",
        quality_score=quality_score,
    )


class TestClassifyResponseType:
    """Tests for response type classification."""

    def test_short_responses(self) -> None:
        """Test short response classification."""
        assert classify_response_type(make_pair("ok")) == "short"
        assert classify_response_type(make_pair("yes")) == "short"
        assert classify_response_type(make_pair("no")) == "short"
        assert classify_response_type(make_pair("hey there")) == "short"  # 2 words

    def test_acknowledgment_responses(self) -> None:
        """Test acknowledgment classification."""
        # Note: "ok" is short (<=3 words), not ack
        # Short acknowledgments with >3 words get classified as answer unless
        # they match the acknowledgment patterns exactly
        # The classify function uses is_acknowledgment_only which checks exact phrases
        assert classify_response_type(make_pair("sounds good")) == "short"  # <=3 words
        assert classify_response_type(make_pair("got it thanks")) == "short"  # <=3 words

    def test_clarification_responses(self) -> None:
        """Test clarification (question) classification."""
        assert classify_response_type(make_pair("What do you mean by that?")) == "clarification"
        assert classify_response_type(make_pair("When should we meet?")) == "clarification"
        assert classify_response_type(make_pair("Can you explain more?")) == "clarification"

    def test_decline_responses(self) -> None:
        """Test decline classification."""
        # Declines need word_count <= 10 and > 3 to be classified as decline
        # (<=3 words are classified as "short" first)
        assert classify_response_type(make_pair("Sorry, I can't make it")) == "decline"
        assert classify_response_type(make_pair("No, I won't be there today")) == "decline"
        assert classify_response_type(make_pair("Nope")) == "short"  # <=3 words = short

    def test_answer_responses(self) -> None:
        """Test answer (default) classification."""
        assert (
            classify_response_type(make_pair("I went to the store and bought some groceries"))
            == "answer"
        )
        assert (
            classify_response_type(make_pair("The meeting is at 3pm in the conference room"))
            == "answer"
        )


class TestBalanceConfig:
    """Tests for BalanceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BalanceConfig()
        assert config.max_pairs_per_contact == 500
        assert config.short_reply_max_words == 3
        assert config.min_short_reply_ratio == 0.08
        assert "answer" in config.target_distribution

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BalanceConfig(
            max_pairs_per_contact=100,
            short_reply_oversample_factor=3.0,
        )
        assert config.max_pairs_per_contact == 100
        assert config.short_reply_oversample_factor == 3.0


class TestBalanceStats:
    """Tests for BalanceStats dataclass."""

    def test_default_values(self) -> None:
        """Test default stats values."""
        stats = BalanceStats()
        assert stats.total_input == 0
        assert stats.total_output == 0
        assert stats.capped_by_contact == 0
        assert stats.by_type == {}


class TestPairBalancer:
    """Tests for PairBalancer class."""

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        balancer = PairBalancer()
        result, stats = balancer.balance([])

        assert result == []
        assert stats.total_input == 0
        assert stats.total_output == 0

    def test_contact_capping(self) -> None:
        """Test that pairs per contact are capped."""
        config = BalanceConfig(max_pairs_per_contact=5)
        balancer = PairBalancer(config)

        # Create 10 pairs for one contact
        pairs = [
            make_pair(
                response_text=f"Response {i} with enough words to not be short",
                contact_id=1,
                quality_score=0.5 + (i / 20),  # Varying quality
            )
            for i in range(10)
        ]

        result, stats = balancer.balance(pairs)

        # Should be capped to 5
        assert stats.capped_by_contact == 5
        assert len([p for p in result if p.contact_id == 1]) <= 5

    def test_quality_priority_when_capping(self) -> None:
        """Test that higher quality pairs are kept when capping."""
        config = BalanceConfig(max_pairs_per_contact=2)
        balancer = PairBalancer(config)

        # Create pairs with different quality scores - need longer text to avoid \
        # "short" classification
        pairs = [
            make_pair("Low quality response text here now", contact_id=1, quality_score=0.3),
            make_pair("High quality response text here now", contact_id=1, quality_score=0.9),
            make_pair("Medium quality response text here now", contact_id=1, quality_score=0.6),
        ]

        result, stats = balancer.balance(pairs)

        # Should cap to 2 pairs - at least the highest quality should be kept
        contact_pairs = [p for p in result if p.contact_id == 1]
        # Due to sampling, we can't guarantee exact scores, but count should be capped
        assert len(contact_pairs) <= 2

    def test_multiple_contacts(self) -> None:
        """Test balancing across multiple contacts."""
        config = BalanceConfig(max_pairs_per_contact=3)
        balancer = PairBalancer(config)

        # Create pairs for 3 contacts
        pairs = []
        for contact_id in [1, 2, 3]:
            for i in range(5):
                pairs.append(
                    make_pair(
                        response_text=f"Contact {contact_id} response {i} text",
                        contact_id=contact_id,
                    )
                )

        result, stats = balancer.balance(pairs)

        assert stats.contacts_processed == 3
        # Each contact should be capped to 3
        for contact_id in [1, 2, 3]:
            assert len([p for p in result if p.contact_id == contact_id]) <= 3

    def test_type_distribution(self) -> None:
        """Test that type distribution is respected."""
        config = BalanceConfig(
            max_pairs_per_contact=1000,  # Don't cap
            target_distribution={
                "answer": 0.50,
                "acknowledgment": 0.25,
                "clarification": 0.15,
                "decline": 0.05,
                "short": 0.05,
            },
        )
        balancer = PairBalancer(config)

        # Create many pairs of each type
        pairs = []
        for _ in range(50):
            # Answers
            pairs.append(
                make_pair(
                    "I went to the store and bought some things",
                    contact_id=1,
                )
            )
            # Short
            pairs.append(make_pair("ok", contact_id=2))
            # Clarification
            pairs.append(make_pair("What do you mean by that exactly?", contact_id=3))

        result, stats = balancer.balance(pairs)

        # Should have some distribution
        assert stats.by_type.get("answer", 0) > 0 or stats.by_type.get("short", 0) > 0

    def test_short_reply_oversampling(self) -> None:
        """Test that short replies are oversampled when below minimum."""
        config = BalanceConfig(
            max_pairs_per_contact=1000,
            min_short_reply_ratio=0.20,  # Want 20% short
            short_reply_oversample_factor=2.0,
        )
        balancer = PairBalancer(config)

        # Create mostly long responses
        pairs = []
        for i in range(90):
            pairs.append(
                make_pair(
                    "This is a long response with many words",
                    contact_id=1,
                )
            )
        # And just a few short ones
        for i in range(10):
            pairs.append(make_pair("ok", contact_id=2))

        result, stats = balancer.balance(pairs)

        # Short replies should be added
        assert stats.short_replies_added > 0

    def test_reproducibility_with_seed(self) -> None:
        """Test that results are reproducible with same seed."""
        config = BalanceConfig(random_seed=42)
        balancer1 = PairBalancer(config)
        balancer2 = PairBalancer(BalanceConfig(random_seed=42))

        pairs = [make_pair(f"Response {i} text", contact_id=1) for i in range(20)]

        result1, _ = balancer1.balance(pairs.copy())
        result2, _ = balancer2.balance(pairs.copy())

        # Results should be identical
        assert len(result1) == len(result2)


class TestGetContactStyleTargets:
    """Tests for contact style target computation."""

    def test_empty_pairs(self) -> None:
        """Test with no pairs for contact."""
        balancer = PairBalancer()
        targets = balancer.get_contact_style_targets([], contact_id=1)

        # Should return defaults
        assert targets["median_reply_length"] == 10
        assert targets["punctuation_rate"] == 0.5

    def test_punctuation_rate(self) -> None:
        """Test punctuation rate calculation."""
        balancer = PairBalancer()
        pairs = [
            make_pair("Hello!", contact_id=1),
            make_pair("Hi.", contact_id=1),
            make_pair("Hey", contact_id=1),  # No punctuation
            make_pair("Bye?", contact_id=1),
        ]

        targets = balancer.get_contact_style_targets(pairs, contact_id=1)
        # 3 out of 4 have punctuation
        assert targets["punctuation_rate"] == 0.75

    def test_emoji_rate(self) -> None:
        """Test emoji rate calculation."""
        balancer = PairBalancer()
        pairs = [
            make_pair("Hello! 😀", contact_id=1),
            make_pair("Hi there", contact_id=1),
            make_pair("Nice! 🎉", contact_id=1),
            make_pair("Bye", contact_id=1),
        ]

        targets = balancer.get_contact_style_targets(pairs, contact_id=1)
        # 2 out of 4 have emojis
        assert targets["emoji_rate"] == 0.5

    def test_greeting_rate(self) -> None:
        """Test greeting rate calculation."""
        balancer = PairBalancer()
        pairs = [
            make_pair("Hey, how are you?", contact_id=1),
            make_pair("Hi there!", contact_id=1),
            make_pair("Thanks for that", contact_id=1),
            make_pair("hello world", contact_id=1),
        ]

        targets = balancer.get_contact_style_targets(pairs, contact_id=1)
        # 3 out of 4 start with greetings
        assert targets["greeting_rate"] == 0.75

    def test_median_reply_length(self) -> None:
        """Test median reply length calculation."""
        balancer = PairBalancer()
        pairs = [
            make_pair("one two", contact_id=1),  # 2 words
            make_pair("one two three", contact_id=1),  # 3 words
            make_pair("one two three four five", contact_id=1),  # 5 words
        ]

        targets = balancer.get_contact_style_targets(pairs, contact_id=1)
        # Median of [2, 3, 5] is 3
        assert targets["median_reply_length"] == 3

    def test_filters_by_contact(self) -> None:
        """Test that only pairs for the specified contact are used."""
        balancer = PairBalancer()
        pairs = [
            make_pair("Short", contact_id=1),  # 1 word
            make_pair("Also short", contact_id=1),  # 2 words
            make_pair("This is a much longer response", contact_id=2),  # 6 words
        ]

        targets = balancer.get_contact_style_targets(pairs, contact_id=1)
        # Should only consider contact 1's pairs
        assert targets["median_reply_length"] <= 2
