"""Pair Balancer - Rebalancing the training set after gating.

Balances the dataset to:
1. Cap per-contact max pairs (prevent one chatty thread dominating)
2. Enforce target distribution across response types
3. Explicitly include short reply examples (to fix length ratio)

Usage:
    from jarvis.pair_balancer import PairBalancer, BalanceConfig

    balancer = PairBalancer(config)
    balanced_pairs = balancer.balance(pairs)
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from jarvis.db import Pair
from jarvis.text_normalizer import (
    is_acknowledgment_only,
    is_question,
)

logger = logging.getLogger(__name__)


@dataclass
class BalanceConfig:
    """Configuration for dataset balancing.

    Attributes:
        max_pairs_per_contact: Maximum pairs from any single contact.
        target_distribution: Target percentage for each response type.
        short_reply_oversample_factor: How much to oversample short replies.
        short_reply_max_words: Max words to count as "short reply".
        min_short_reply_ratio: Minimum ratio of short replies in final set.
        random_seed: Seed for reproducibility.
    """

    max_pairs_per_contact: int = 500
    target_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "answer": 0.45,  # Substantive answers
            "acknowledgment": 0.25,  # Acks, confirmations
            "clarification": 0.12,  # Questions back, clarifying
            "decline": 0.08,  # Nos, can't, sorry
            "short": 0.10,  # Very short replies (explicitly kept)
        }
    )
    short_reply_oversample_factor: float = 2.0
    short_reply_max_words: int = 3
    min_short_reply_ratio: float = 0.08
    random_seed: int | None = None


@dataclass
class BalanceStats:
    """Statistics from balancing run."""

    total_input: int = 0
    total_output: int = 0
    capped_by_contact: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    short_replies_added: int = 0
    contacts_processed: int = 0


def classify_response_type(pair: Pair) -> str:
    """Classify a pair's response into a type category.

    Categories:
    - "short": Very short replies (<= 3 words)
    - "acknowledgment": Ack phrases (ok, sure, thanks, etc.)
    - "clarification": Questions/clarifying responses
    - "decline": Negative responses (no, can't, sorry)
    - "answer": Everything else (substantive answers)
    """
    response = pair.response_text
    words = response.split()
    word_count = len(words)

    # Short replies (very short, kept explicitly)
    if word_count <= 3:
        return "short"

    # Acknowledgments
    if is_acknowledgment_only(response):
        return "acknowledgment"

    # Clarifications (response is a question)
    if is_question(response):
        return "clarification"

    # Declines
    decline_patterns = {
        "no",
        "nope",
        "nah",
        "can't",
        "cannot",
        "won't",
        "wouldn't",
        "sorry",
        "unfortunately",
        "not able",
        "not gonna",
        "not going to",
        "don't think so",
        "probably not",
        "maybe not",
    }
    response_lower = response.lower()
    for pattern in decline_patterns:
        if pattern in response_lower:
            # Check it's not negated negation
            if word_count <= 10:  # Short declines
                return "decline"

    # Default: substantive answer
    return "answer"


class PairBalancer:
    """Balance the training set after gating."""

    def __init__(self, config: BalanceConfig | None = None) -> None:
        """Initialize balancer.

        Args:
            config: Balance configuration.
        """
        self.config = config or BalanceConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def balance(self, pairs: list[Pair]) -> tuple[list[Pair], BalanceStats]:
        """Balance the dataset.

        Steps:
        1. Group by contact and cap per-contact
        2. Classify each pair by response type
        3. Sample according to target distribution
        4. Oversample short replies to ensure min ratio

        Args:
            pairs: Input pairs (should be already validated).

        Returns:
            Tuple of (balanced pairs, stats).
        """
        stats = BalanceStats(total_input=len(pairs))

        if not pairs:
            return [], stats

        # Step 1: Group by contact and cap
        by_contact: dict[int | None, list[Pair]] = defaultdict(list)
        for pair in pairs:
            by_contact[pair.contact_id].append(pair)

        capped_pairs: list[Pair] = []
        for contact_id, contact_pairs in by_contact.items():
            stats.contacts_processed += 1
            if len(contact_pairs) > self.config.max_pairs_per_contact:
                # Sample to cap (prefer higher quality)
                sorted_pairs = sorted(contact_pairs, key=lambda p: p.quality_score, reverse=True)
                capped = sorted_pairs[: self.config.max_pairs_per_contact]
                stats.capped_by_contact += len(contact_pairs) - len(capped)
                capped_pairs.extend(capped)
            else:
                capped_pairs.extend(contact_pairs)

        # Step 2: Classify by response type
        by_type: dict[str, list[Pair]] = defaultdict(list)
        for pair in capped_pairs:
            response_type = classify_response_type(pair)
            by_type[response_type].append(pair)

        # Step 3: Sample according to target distribution
        # First, determine target counts
        total_target = len(capped_pairs)
        target_counts: dict[str, int] = {}
        for rtype, ratio in self.config.target_distribution.items():
            target_counts[rtype] = int(total_target * ratio)

        # Sample from each bucket
        sampled: list[Pair] = []
        for rtype, target in target_counts.items():
            available = by_type.get(rtype, [])
            if len(available) <= target:
                sampled.extend(available)
            else:
                # Prefer higher quality
                sorted_avail = sorted(available, key=lambda p: p.quality_score, reverse=True)
                sampled.extend(sorted_avail[:target])
            stats.by_type[rtype] = min(len(available), target)

        # Step 4: Ensure minimum short reply ratio by oversampling
        short_count = sum(1 for p in sampled if classify_response_type(p) == "short")
        short_ratio = short_count / len(sampled) if sampled else 0

        if short_ratio < self.config.min_short_reply_ratio:
            # Need to add more short replies
            needed = int(len(sampled) * self.config.min_short_reply_ratio) - short_count
            available_short = by_type.get("short", [])

            # Oversample with replacement if needed
            if available_short and needed > 0:
                to_add = []
                for _ in range(needed):
                    to_add.append(random.choice(available_short))
                sampled.extend(to_add)
                stats.short_replies_added = len(to_add)

        # Shuffle final result
        random.shuffle(sampled)

        stats.total_output = len(sampled)
        return sampled, stats

    def get_contact_style_targets(self, pairs: list[Pair], contact_id: int) -> dict[str, Any]:
        """Compute style targets for a specific contact.

        Computes from their pairs:
        - median_reply_length: Median word count
        - punctuation_rate: Fraction with ending punctuation
        - emoji_rate: Fraction containing emojis
        - greeting_rate: Fraction starting with greeting

        Args:
            pairs: All pairs.
            contact_id: Contact to compute for.

        Returns:
            Dictionary of style targets.
        """
        contact_pairs = [p for p in pairs if p.contact_id == contact_id]

        if not contact_pairs:
            return {
                "median_reply_length": 10,
                "punctuation_rate": 0.5,
                "emoji_rate": 0.1,
                "greeting_rate": 0.2,
            }

        # Reply lengths
        lengths = [len(p.response_text.split()) for p in contact_pairs]
        lengths.sort()
        median_length = lengths[len(lengths) // 2]

        # Punctuation rate
        punct_count = sum(1 for p in contact_pairs if p.response_text.rstrip()[-1:] in ".!?")
        punctuation_rate = punct_count / len(contact_pairs)

        # Emoji rate (simple check)
        import re

        emoji_pattern = re.compile(
            r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF"
            r"\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]"
        )
        emoji_count = sum(1 for p in contact_pairs if emoji_pattern.search(p.response_text))
        emoji_rate = emoji_count / len(contact_pairs)

        # Greeting rate
        greetings = {"hi", "hey", "hello", "yo", "sup", "hiya", "heya"}
        greeting_count = sum(
            1
            for p in contact_pairs
            if p.response_text.lower().split()[0:1]
            and p.response_text.lower().split()[0].rstrip(",!") in greetings
        )
        greeting_rate = greeting_count / len(contact_pairs)

        return {
            "median_reply_length": median_length,
            "punctuation_rate": round(punctuation_rate, 3),
            "emoji_rate": round(emoji_rate, 3),
            "greeting_rate": round(greeting_rate, 3),
        }
