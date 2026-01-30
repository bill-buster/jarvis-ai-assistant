#!/usr/bin/env python3
"""Create a clean, filtered test set for baseline evaluation.

This creates a test set that:
1. Filters out impossible cases (specific dates, addresses, proper nouns not in context)
2. Stratifies by important dimensions (group vs 1:1, response length)
3. Uses raw conversation without pre-baked style tags

Usage:
    python scripts/create_clean_test_set.py              # Create 500 samples
    python scripts/create_clean_test_set.py --limit 1000 # Create 1000 samples
    python scripts/create_clean_test_set.py --stats      # Show stats only
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_FILE = Path("results/test_set/clean_test_data.jsonl")


def is_achievable_response(gold: str, context: str) -> tuple[bool, str]:
    """Check if a gold response is achievable without specific knowledge.

    Returns: (is_achievable, reason)
    """
    gold_lower = gold.lower().strip()
    context_lower = context.lower()

    # Too long = probably specific
    if len(gold) > 80:
        return False, "too_long"

    # Contains specific times
    if re.search(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', gold_lower):
        return False, "specific_time"

    # Contains specific dates
    months = "january|february|march|april|may|june|july|august|september|october|november|december"
    if re.search(rf'\b({months})\s+\d+', gold_lower):
        return False, "specific_date"
    if re.search(r'\b\d{1,2}(st|nd|rd|th)\b', gold_lower):
        return False, "specific_ordinal"

    # Contains addresses
    if re.search(r'\b\d+\s+\w+\s+(street|st|ave|avenue|road|rd|blvd|drive|dr)\b', gold_lower):
        return False, "address"

    # Contains large numbers not in context
    numbers = re.findall(r'\b\d{2,}\b', gold)  # 2+ digit numbers
    for n in numbers:
        if n not in context:
            return False, f"specific_number_{n}"

    # Contains proper nouns not in context (skip first word - might be sentence start)
    words = gold.split()
    if len(words) > 1:
        proper_nouns = [w for w in words[1:] if w[0].isupper() and len(w) > 2]
        for noun in proper_nouns:
            if noun.lower() not in context_lower:
                return False, f"proper_noun_{noun}"

    # Reactions and short responses - always achievable
    if len(gold) < 20:
        return True, "short"

    return True, "generic"


def is_group_chat(contact: str) -> bool:
    """Detect group chat from contact name."""
    return "," in contact or "+" in contact


def create_clean_test_set(limit: int = 500):
    """Create clean test set from iMessage conversations."""
    try:
        from core.imessage.reader import MessageReader
    except ImportError as e:
        print(f"Import error: {e}")
        return

    reader = MessageReader()

    print("Loading conversations...")
    conversations = reader.get_conversations(limit=1000)
    print(f"Found {len(conversations)} conversations")

    # Spam filters
    spam_keywords = [
        "reward points", "expire", "your order", "tracking",
        "verification code", "click here", "unsubscribe",
        "utm_source", "law firm", "legal representation",
        "auto]", "doordash", "uber eats", "grubhub",
    ]

    # Collect candidates with metadata
    candidates = []

    print("Processing conversations...")
    for conv_idx, conv in enumerate(conversations):
        if (conv_idx + 1) % 100 == 0:
            print(f"  [{conv_idx + 1}/{len(conversations)}] {len(candidates)} candidates")

        try:
            messages = reader.get_messages(conv.chat_id, limit=300)
            if not messages or len(messages) < 3:
                continue

            messages = list(reversed(messages))  # chronological
            contact = conv.display_name or ""
            participants = conv.participants or []
            participant_str = participants[0] if participants else ""

            # Skip short codes (spam)
            if participant_str.isdigit() and 5 <= len(participant_str) <= 6:
                continue

            # Skip spam conversations
            recent_text = " ".join((m.text or "").lower() for m in messages[-30:])
            if sum(1 for kw in spam_keywords if kw in recent_text) >= 2:
                continue

            is_group = is_group_chat(contact)

            # Find (their_msg, your_reply) pairs
            for i in range(1, len(messages)):
                msg = messages[i]

                if not msg.is_from_me:
                    continue

                my_text = (msg.text or "").strip()

                # Skip empty, too short, too long
                if len(my_text) < 2 or len(my_text) > 100:
                    continue

                # Skip reactions
                if any(p in my_text.lower() for p in ["loved", "liked", "emphasized", "laughed at", "questioned"]):
                    continue

                # Previous message must be from them
                prev_msg = messages[i - 1]
                if prev_msg.is_from_me:
                    continue

                their_text = (prev_msg.text or "").strip()
                if len(their_text) < 2:
                    continue

                # Get conversation context (last 10 messages before reply)
                context_start = max(0, i - 10)
                context_msgs = messages[context_start:i]

                # Build raw conversation (no style tags)
                conv_lines = []
                for m in context_msgs:
                    text = (m.text or "").strip()
                    if text:
                        prefix = "me:" if m.is_from_me else "them:"
                        conv_lines.append(f"{prefix} {text}")

                if len(conv_lines) < 2:
                    continue

                conversation_text = "\n".join(conv_lines)

                # Check if achievable
                achievable, reason = is_achievable_response(my_text, conversation_text)
                if not achievable:
                    continue

                # Categorize by response length
                if len(my_text) < 15:
                    length_bucket = "short"
                elif len(my_text) < 40:
                    length_bucket = "medium"
                else:
                    length_bucket = "long"

                candidates.append({
                    "contact": contact or participant_str[:30],
                    "is_group": is_group,
                    "length_bucket": length_bucket,
                    "gold_response": my_text,
                    "last_message": their_text[:150],
                    "conversation": conversation_text,
                    "achievable_reason": reason,
                })

        except Exception as e:
            continue

    print(f"\nCollected {len(candidates)} achievable candidates")

    # Stratified sampling
    # Target: 20% groups, balanced length buckets
    groups = [c for c in candidates if c["is_group"]]
    individuals = [c for c in candidates if not c["is_group"]]

    print(f"Groups: {len(groups)}, Individuals: {len(individuals)}")

    # Calculate target counts
    n_groups = min(len(groups), int(limit * 0.20))  # 20% groups
    n_individuals = limit - n_groups

    # Sample groups
    random.shuffle(groups)
    selected_groups = groups[:n_groups]

    # Sample individuals with length stratification
    random.shuffle(individuals)
    short = [c for c in individuals if c["length_bucket"] == "short"]
    medium = [c for c in individuals if c["length_bucket"] == "medium"]
    long = [c for c in individuals if c["length_bucket"] == "long"]

    # Target: 30% short, 50% medium, 20% long
    n_short = min(len(short), int(n_individuals * 0.30))
    n_medium = min(len(medium), int(n_individuals * 0.50))
    n_long = min(len(long), n_individuals - n_short - n_medium)

    selected_individuals = short[:n_short] + medium[:n_medium] + long[:n_long]

    # Combine and shuffle
    samples = selected_groups + selected_individuals
    random.shuffle(samples)

    # Add IDs
    for i, s in enumerate(samples):
        s["id"] = i + 1
        s["created"] = datetime.now().isoformat()

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n✓ Created clean test set: {len(samples)} samples")
    print(f"  File: {OUTPUT_FILE}")

    show_stats(samples)


def show_stats(samples: list[dict] | None = None):
    """Show test set stats."""
    if samples is None:
        if not OUTPUT_FILE.exists():
            print(f"No test set found: {OUTPUT_FILE}")
            return
        samples = []
        with open(OUTPUT_FILE) as f:
            for line in f:
                samples.append(json.loads(line))

    print(f"\n{'='*60}")
    print("CLEAN TEST SET STATS")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")

    # By chat type
    groups = sum(1 for s in samples if s.get("is_group"))
    print(f"\nBy chat type:")
    print(f"  Groups:      {groups:4d} ({groups/len(samples)*100:5.1f}%)")
    print(f"  1:1 chats:   {len(samples)-groups:4d} ({(len(samples)-groups)/len(samples)*100:5.1f}%)")

    # By length bucket
    by_length = Counter(s.get("length_bucket", "unknown") for s in samples)
    print(f"\nBy response length:")
    for bucket in ["short", "medium", "long"]:
        count = by_length.get(bucket, 0)
        print(f"  {bucket:12}: {count:4d} ({count/len(samples)*100:5.1f}%)")

    # Actual length stats
    lengths = [len(s["gold_response"]) for s in samples]
    print(f"\nResponse length stats:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")

    # Unique contacts
    contacts = set(s.get("contact", "") for s in samples)
    print(f"\nUnique contacts: {len(contacts)}")

    # Show some examples
    print(f"\nRandom examples:")
    random.shuffle(samples)
    for s in samples[:3]:
        print(f"\n  [{s['contact']}] ({'group' if s.get('is_group') else '1:1'}, {s.get('length_bucket')})")
        print(f"  them: {s['last_message'][:50]}...")
        print(f"  gold: \"{s['gold_response']}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Number of samples")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        create_clean_test_set(limit=args.limit)


if __name__ == "__main__":
    main()
