#!/usr/bin/env python3
"""
Mine Response Pairs from iMessage

Learns (incoming message → your response) patterns from actual conversations.
This is what we actually need for realistic reply generation.

Example output:
  "thanks!" → "np!"
  "wyd tonight" → "nothing much wbu"
  "omw" → "sounds good"
"""

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mine_response_pairs(
    db_path: Path,
    max_pairs: int = 100000,
    max_time_gap_seconds: int = 300,  # 5 minutes
) -> list[dict]:
    """Mine (incoming → response) pairs from conversations.

    Args:
        db_path: Path to iMessage chat.db
        max_pairs: Maximum number of pairs to extract
        max_time_gap_seconds: Max time between incoming msg and your reply

    Returns:
        List of {incoming, response, frequency} dicts
    """

    logger.info("Mining response pairs from: %s", db_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Query to get adjacent message pairs where you replied
    # Use chat_message_join to connect messages to chats
    query = """
        SELECT
            m1.text as incoming,
            m2.text as response,
            m2.date - m1.date as time_gap
        FROM message m1
        JOIN chat_message_join cmj1 ON m1.ROWID = cmj1.message_id
        JOIN chat_message_join cmj2 ON cmj1.chat_id = cmj2.chat_id
        JOIN message m2 ON m2.ROWID = cmj2.message_id
        WHERE m1.is_from_me = 0          -- They sent
          AND m2.is_from_me = 1          -- You replied
          AND m2.date > m1.date          -- Your message came after
          AND m2.date - m1.date < ?      -- Within time window (in nanoseconds)
          AND m2.ROWID > m1.ROWID        -- Ensure temporal ordering
          AND m1.text IS NOT NULL
          AND m1.text != ''
          AND m2.text IS NOT NULL
          AND m2.text != ''
          AND length(m1.text) > 0
          AND length(m2.text) > 0
        ORDER BY m1.date DESC
        LIMIT ?
    """

    # Convert seconds to nanoseconds (iMessage uses nanoseconds since 2001)
    time_gap_ns = max_time_gap_seconds * 1_000_000_000

    logger.info("Extracting message pairs (max gap: %ds)...", max_time_gap_seconds)
    cursor.execute(query, (time_gap_ns, max_pairs))

    pairs = []
    for row in cursor.fetchall():
        incoming, response, time_gap_ns = row
        pairs.append({
            "incoming": incoming,
            "response": response,
            "time_gap_seconds": time_gap_ns / 1_000_000_000
        })

    conn.close()

    logger.info("Extracted %d response pairs", len(pairs))

    # Count frequencies
    pair_counter = Counter()
    for pair in pairs:
        # Normalize for counting (lowercase, strip)
        key = (pair["incoming"].lower().strip(), pair["response"].lower().strip())
        pair_counter[key] += 1

    # Build templates
    templates = []
    for (incoming, response), frequency in pair_counter.most_common():
        if frequency < 3:  # Minimum frequency threshold
            continue

        templates.append({
            "incoming": incoming,
            "response": response,
            "frequency": frequency
        })

    logger.info("Found %d unique response patterns (freq >= 3)", len(templates))

    return templates


def analyze_patterns(templates: list[dict]):
    """Analyze and categorize response patterns."""

    # Group by incoming pattern
    incoming_groups = defaultdict(list)
    for t in templates:
        incoming_groups[t["incoming"]].append(t)

    print("\n" + "="*80)
    print("RESPONSE PATTERN ANALYSIS")
    print("="*80)

    print(f"\nTotal unique response patterns: {len(templates)}")
    print(f"Total unique incoming patterns: {len(incoming_groups)}")

    # Show most common incoming messages with YOUR different responses
    print("\n--- Most Common Incoming Messages (with your variations) ---\n")

    sorted_incoming = sorted(
        incoming_groups.items(),
        key=lambda x: sum(r["frequency"] for r in x[1]),
        reverse=True
    )

    for incoming, responses in sorted_incoming[:20]:
        total_freq = sum(r["frequency"] for r in responses)
        print(f"\nIncoming ({total_freq} times): \"{incoming[:60]}\"")
        print(f"  Your responses:")
        for resp in sorted(responses, key=lambda x: x["frequency"], reverse=True)[:5]:
            print(f"    [{resp['frequency']:>4}×] \"{resp['response'][:60]}\"")

    # Show most common (incoming → response) pairs
    print("\n--- Top 30 Response Pairs ---\n")
    for i, t in enumerate(templates[:30], 1):
        print(f"{i:2}. [{t['frequency']:>4}×] \"{t['incoming'][:40]}\" → \"{t['response'][:40]}\"")


def main():
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    output_file = Path("results/response_pairs.json")

    # Mine pairs
    templates = mine_response_pairs(db_path, max_pairs=100000)

    # Analyze
    analyze_patterns(templates)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "total_patterns": len(templates),
            "patterns": templates
        }, f, indent=2)

    logger.info("\n✓ Response pairs saved to: %s", output_file)


if __name__ == "__main__":
    main()
