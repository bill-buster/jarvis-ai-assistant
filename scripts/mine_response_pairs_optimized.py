#!/usr/bin/env python3
"""
Optimized Response Pair Mining with Temporal Analysis

Key features:
- Multi-message response grouping (handles 10 consecutive short messages)
- Consistency scoring (patterns appearing across years = more reliable)
- Recency weighting (recent messages weighted higher)
- Parameter sweep (tests multiple clustering parameters)
- Proper filtering (system messages, reactions removed)
- Single-word responses kept (slang is important!)

Ranking formula:
  score = frequency × consistency_score × recency_weight
"""

import json
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# iMessage epoch: 2001-01-01 00:00:00 UTC
IMESSAGE_EPOCH = datetime(2001, 1, 1)

# System messages to filter out
SYSTEM_MESSAGE_PATTERNS = [
    "Loved ",
    "Laughed at ",
    "Emphasized ",
    "Questioned ",
    "Liked ",
    "Disliked ",
    "Loved an image",
    "Laughed at an image",
    "￼",  # Attachment placeholder
]


def imessage_timestamp_to_datetime(timestamp_ns: int) -> datetime:
    """Convert iMessage nanosecond timestamp to datetime."""
    return IMESSAGE_EPOCH + timedelta(microseconds=timestamp_ns / 1000)


def is_system_message(text: str) -> bool:
    """Check if message is a system message/reaction."""
    if not text or text.strip() == "":
        return True

    for pattern in SYSTEM_MESSAGE_PATTERNS:
        if pattern in text:
            return True

    return False


def get_response_groups(
    db_path: Path,
    max_time_gap_seconds: int = 300,  # 5 min between incoming and response
    max_burst_gap_seconds: int = 30,   # 30 sec between your consecutive messages
) -> list[dict]:
    """Get (incoming message → your multi-message response) pairs.

    Handles cases where you send multiple short messages in a row.

    Example:
        Friend: "wanna grab lunch?"
        You:    "yea"         (11:30:00)
        You:    "sounds good" (11:30:02)
        You:    "what time"   (11:30:05)

        → Returns: {"incoming": "wanna grab lunch?",
                    "response": "yea sounds good what time",
                    "dates": [timestamp1, timestamp2, timestamp3]}
    """

    logger.info("Extracting message pairs with multi-message grouping...")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all messages in chronological order
    query = """
        SELECT
            m.ROWID,
            m.text,
            m.is_from_me,
            m.date,
            cmj.chat_id
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE m.text IS NOT NULL
          AND m.text != ''
          AND length(m.text) > 0
        ORDER BY cmj.chat_id, m.date
    """

    cursor.execute(query)

    # Group messages by chat and build response pairs
    current_chat = None
    chat_messages = []
    response_groups = []

    for row in cursor.fetchall():
        rowid, text, is_from_me, date_ns, chat_id = row

        # New chat - process accumulated messages
        if chat_id != current_chat:
            if chat_messages:
                pairs = extract_pairs_from_chat(
                    chat_messages,
                    max_time_gap_seconds,
                    max_burst_gap_seconds
                )
                response_groups.extend(pairs)

            current_chat = chat_id
            chat_messages = []

        # Skip system messages
        if is_system_message(text):
            continue

        chat_messages.append({
            "rowid": rowid,
            "text": text,
            "is_from_me": is_from_me,
            "date_ns": date_ns,
        })

    # Process last chat
    if chat_messages:
        pairs = extract_pairs_from_chat(
            chat_messages,
            max_time_gap_seconds,
            max_burst_gap_seconds
        )
        response_groups.extend(pairs)

    conn.close()

    logger.info("Extracted %d response groups", len(response_groups))
    return response_groups


def extract_pairs_from_chat(
    messages: list[dict],
    max_time_gap_seconds: int,
    max_burst_gap_seconds: int,
) -> list[dict]:
    """Extract (incoming → response) pairs from a single chat."""

    pairs = []
    time_gap_ns = max_time_gap_seconds * 1_000_000_000
    burst_gap_ns = max_burst_gap_seconds * 1_000_000_000

    i = 0
    while i < len(messages) - 1:
        msg = messages[i]

        # Skip if this is your message (we want incoming → response)
        if msg["is_from_me"] == 1:
            i += 1
            continue

        # This is an incoming message - look for your response(s)
        incoming_text = msg["text"]
        incoming_date = msg["date_ns"]

        # Collect your consecutive response messages
        response_texts = []
        response_dates = []

        j = i + 1
        while j < len(messages):
            next_msg = messages[j]

            # Not your message - stop
            if next_msg["is_from_me"] == 0:
                break

            # Too much time passed since incoming - stop
            if next_msg["date_ns"] - incoming_date > time_gap_ns:
                break

            # Check if this is part of a burst (consecutive messages from you)
            if response_dates and next_msg["date_ns"] - response_dates[-1] > burst_gap_ns:
                # Too much gap between your messages - stop burst
                break

            response_texts.append(next_msg["text"])
            response_dates.append(next_msg["date_ns"])
            j += 1

        # If we found a response, add the pair
        if response_texts:
            # Combine multi-message responses with space
            combined_response = " ".join(response_texts)

            pairs.append({
                "incoming": incoming_text,
                "response": combined_response,
                "response_dates": response_dates,
            })

        i = j if j > i + 1 else i + 1

    return pairs


def calculate_temporal_scores(response_groups: list[dict]) -> list[dict]:
    """Calculate consistency and recency scores for each response pattern.

    Consistency score: How evenly distributed is the pattern across time periods?
    - High: appears consistently over years (reliable pattern)
    - Low: spike in one period, absent in others (temporary pattern like "mwah")

    Recency weight: Exponential decay based on age
    - Recent messages weighted higher
    - Old messages weighted lower but not zero
    """

    logger.info("Calculating temporal scores...")

    # Group pairs by pattern
    pattern_occurrences = defaultdict(list)

    for group in response_groups:
        pattern = (group["incoming"].lower().strip(), group["response"].lower().strip())
        # Use the first response date as the pattern date
        date_ns = group["response_dates"][0]
        pattern_occurrences[pattern].append(date_ns)

    # Calculate scores
    scored_patterns = []
    current_time_ns = int((datetime.now() - IMESSAGE_EPOCH).total_seconds() * 1_000_000_000)

    for (incoming, response), dates in pattern_occurrences.items():
        frequency = len(dates)

        # Skip very rare patterns
        if frequency < 2:
            continue

        # Consistency score: measure variance across years
        years = [imessage_timestamp_to_datetime(d).year for d in dates]
        year_counts = Counter(years)

        if len(year_counts) == 1:
            # All in one year - lower consistency
            consistency_score = 0.5
        else:
            # Calculate coefficient of variation (lower = more consistent)
            mean_per_year = np.mean(list(year_counts.values()))
            std_per_year = np.std(list(year_counts.values()))
            cv = std_per_year / mean_per_year if mean_per_year > 0 else 1.0

            # Invert: lower CV = higher consistency
            consistency_score = 1.0 / (1.0 + cv)

        # Recency weight: exponential decay with longer half-life
        # Use the MOST RECENT occurrence (not average)
        most_recent_date_ns = max(dates)
        age_ns = current_time_ns - most_recent_date_ns
        age_days = age_ns / (1_000_000_000 * 86400)

        # Decay constant: half-life of 730 days (2 years)
        # This means patterns from 2 years ago still have 50% weight
        decay_constant = 730
        recency_weight = np.exp(-age_days / decay_constant)

        # Consistency bonus: patterns spanning many years get a boost
        year_span = max(years) - min(years) + 1
        consistency_bonus = 1.0 + (len(year_counts) / max(3, year_span))

        # Combined score
        score = frequency * consistency_score * recency_weight * consistency_bonus

        scored_patterns.append({
            "incoming": incoming,
            "response": response,
            "frequency": frequency,
            "consistency_score": consistency_score,
            "recency_weight": recency_weight,
            "consistency_bonus": consistency_bonus,
            "combined_score": score,
            "years_active": sorted(year_counts.keys()),
            "most_recent": max(dates),
            "age_days": int(age_days),
        })

    # Sort by combined score
    scored_patterns.sort(key=lambda x: x["combined_score"], reverse=True)

    logger.info("Scored %d unique patterns", len(scored_patterns))
    return scored_patterns


def analyze_patterns(scored_patterns: list[dict]):
    """Analyze and display pattern insights."""

    print("\n" + "="*80)
    print("RESPONSE PATTERN ANALYSIS (Optimized)")
    print("="*80)

    print(f"\nTotal unique patterns: {len(scored_patterns)}")

    # Top patterns by combined score
    print("\n--- Top 30 Patterns (by score: freq × consistency × recency × year_bonus) ---\n")
    for i, p in enumerate(scored_patterns[:30], 1):
        print(f"{i:2}. Score: {p['combined_score']:>6.1f} | "
              f"Freq: {p['frequency']:>4} | "
              f"Age: {p['age_days']:>4}d | "
              f"Recency: {p['recency_weight']:.2f}")
        print(f"    Consistency: {p['consistency_score']:.2f} | "
              f"Year Bonus: {p['consistency_bonus']:.2f} | "
              f"Years: {p['years_active']}")
        print(f"    \"{p['incoming'][:50]}\" → \"{p['response'][:50]}\"")
        print()

    # Show high-consistency patterns (reliable over time)
    print("\n--- Most Consistent Patterns (appear across multiple years) ---\n")
    consistent = sorted(scored_patterns, key=lambda x: x['consistency_score'], reverse=True)
    for i, p in enumerate(consistent[:10], 1):
        if len(p['years_active']) < 2:
            continue
        print(f"{i:2}. Consistency: {p['consistency_score']:.2f} | "
              f"Years: {len(p['years_active'])} | "
              f"Freq: {p['frequency']}")
        print(f"    Active: {p['years_active']}")
        print(f"    \"{p['incoming'][:50]}\" → \"{p['response'][:50]}\"")
        print()

    # Show most recent patterns
    print("\n--- Most Recent Patterns (current style) ---\n")
    recent = sorted(scored_patterns, key=lambda x: x['recency_weight'], reverse=True)
    for i, p in enumerate(recent[:10], 1):
        most_recent_dt = imessage_timestamp_to_datetime(p['most_recent'])
        print(f"{i:2}. Recency: {p['recency_weight']:.2f} | "
              f"Last used: {most_recent_dt.strftime('%Y-%m-%d')} | "
              f"Freq: {p['frequency']}")
        print(f"    \"{p['incoming'][:50]}\" → \"{p['response'][:50]}\"")
        print()


def main():
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    output_file = Path("results/response_pairs_optimized.json")

    # Extract response groups
    response_groups = get_response_groups(db_path)

    # Calculate temporal scores
    scored_patterns = calculate_temporal_scores(response_groups)

    # Analyze
    analyze_patterns(scored_patterns)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "total_patterns": len(scored_patterns),
            "patterns": scored_patterns
        }, f, indent=2)

    logger.info("\n✓ Optimized response pairs saved to: %s", output_file)


if __name__ == "__main__":
    main()
