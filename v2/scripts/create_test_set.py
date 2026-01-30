#!/usr/bin/env python3
"""Create comprehensive test set from ALL iMessage replies.

Uses LLM to analyze each conversation for:
- Relationship type
- Formality level
- Tone
- Your texting style in that convo
- What kind of response is needed

Usage:
    python scripts/create_test_set.py                # Create full test set
    python scripts/create_test_set.py --limit 100   # Limit samples
    python scripts/create_test_set.py --stats       # Show stats
    python scripts/create_test_set.py --examples 5  # Show examples
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")


def create_test_set(limit: int | None = None, use_llm: bool = True):
    """Create test set from ALL iMessage replies with LLM analysis."""
    try:
        from core.imessage.reader import MessageReader
    except ImportError as e:
        print(f"Import error: {e}")
        return

    # Load LLM analyzer if requested
    analyzer = None
    if use_llm:
        try:
            from core.generation.llm_analyzer import LLMAnalyzer
            print("Loading LLM analyzer...")
            analyzer = LLMAnalyzer()
        except Exception as e:
            print(f"Could not load LLM analyzer: {e}")
            print("Falling back to no analysis")

    reader = MessageReader()

    print("Loading ALL conversations...")
    conversations = reader.get_conversations(limit=1000)
    print(f"Found {len(conversations)} conversations")

    spam_keywords = [
        "reward points", "expire", "your order", "tracking",
        "verification code", "click here", "unsubscribe",
        "utm_source", "law firm", "legal representation",
        "auto]", "doordash", "uber eats", "grubhub",
    ]

    samples = []
    seen_responses = set()
    skipped_spam = 0
    skipped_short = 0
    skipped_dupe = 0

    print(f"\nProcessing conversations...")

    for conv_idx, conv in enumerate(conversations):
        if limit and len(samples) >= limit:
            break

        if (conv_idx + 1) % 50 == 0:
            print(f"  [{conv_idx + 1}/{len(conversations)}] {len(samples)} samples collected")

        try:
            messages = reader.get_messages(conv.chat_id, limit=200)
            if not messages or len(messages) < 3:
                continue

            messages = list(reversed(messages))  # chronological
            contact = conv.display_name or ""
            participants = conv.participants or []
            participant_str = participants[0] if participants else ""

            # Skip short codes (spam)
            if participant_str.isdigit() and 5 <= len(participant_str) <= 6:
                skipped_spam += 1
                continue

            # Skip spam conversations
            recent_text = " ".join((m.text or "").lower() for m in messages[-20:])
            if sum(1 for kw in spam_keywords if kw in recent_text) >= 2:
                skipped_spam += 1
                continue

            # Find YOUR replies to their messages
            for i in range(1, len(messages)):
                if limit and len(samples) >= limit:
                    break

                msg = messages[i]

                # Must be from me
                if not msg.is_from_me:
                    continue

                my_text = (msg.text or "").strip()

                # Skip empty, too short, too long
                if len(my_text) < 2 or len(my_text) > 200:
                    skipped_short += 1
                    continue

                # Skip reactions
                if any(p in my_text.lower() for p in ["loved", "liked", "emphasized", "laughed at"]):
                    continue

                # Previous message must be from them
                prev_msg = messages[i - 1]
                if prev_msg.is_from_me:
                    continue

                their_text = (prev_msg.text or "").strip()
                if len(their_text) < 2:
                    continue

                # Dedupe by response (avoid same reply patterns)
                response_key = my_text[:30].lower()
                if response_key in seen_responses:
                    skipped_dupe += 1
                    continue
                seen_responses.add(response_key)

                # Get conversation context
                context_start = max(0, i - 15)
                context_msgs = messages[context_start:i]

                msg_dicts = [
                    {"text": (m.text or "").strip(), "is_from_me": m.is_from_me}
                    for m in context_msgs if (m.text or "").strip()
                ]

                if len(msg_dicts) < 2:
                    continue

                # Build prompt for models
                lines = []
                for m in msg_dicts[-15:]:
                    prefix = "me:" if m.get("is_from_me") else "them:"
                    lines.append(f"{prefix} {m['text']}")

                conversation_text = "\n".join(lines)

                # LLM analysis
                analysis = None
                if analyzer:
                    try:
                        analysis = analyzer.analyze(msg_dicts)
                    except Exception:
                        pass

                sample = {
                    "id": len(samples) + 1,
                    "contact": contact or participant_str[:20],
                    "last_message": their_text[:150],
                    "gold_response": my_text,
                    "conversation": conversation_text,
                    "created": datetime.now().isoformat(),
                }

                # Add LLM analysis if available
                if analysis:
                    sample["relationship"] = analysis.relationship
                    sample["formality"] = analysis.formality
                    sample["their_tone"] = analysis.their_tone
                    sample["my_style"] = analysis.my_style
                    sample["topics"] = analysis.topics
                    sample["response_type"] = analysis.response_type

                samples.append(sample)

        except Exception as e:
            continue

    # Cleanup analyzer
    if analyzer:
        analyzer.unload()
        gc.collect()

    # Shuffle
    random.shuffle(samples)
    for i, s in enumerate(samples):
        s["id"] = i + 1

    # Save
    TEST_SET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_SET_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n✓ Created test set: {len(samples)} samples")
    print(f"  Skipped: {skipped_spam} spam, {skipped_short} too short/long, {skipped_dupe} duplicates")
    print(f"  File: {TEST_SET_FILE}")

    show_stats()


def show_stats():
    """Show test set stats."""
    if not TEST_SET_FILE.exists():
        print("No test set yet.")
        return

    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))

    print("\n" + "=" * 60)
    print("TEST SET STATS")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")

    # Response length
    lengths = [len(s["gold_response"]) for s in samples]
    print(f"Avg response length: {sum(lengths)/len(lengths):.0f} chars")

    # By relationship (if LLM analyzed)
    if samples and "relationship" in samples[0]:
        by_rel = {}
        for s in samples:
            rel = s.get("relationship", "unknown")
            by_rel[rel] = by_rel.get(rel, 0) + 1

        print("\nBy relationship:")
        for k, v in sorted(by_rel.items(), key=lambda x: -x[1]):
            pct = v / len(samples) * 100
            print(f"  {k}: {v} ({pct:.0f}%)")

    # By formality
    if samples and "formality" in samples[0]:
        by_form = {}
        for s in samples:
            form = s.get("formality", "unknown")
            by_form[form] = by_form.get(form, 0) + 1

        print("\nBy formality:")
        for k, v in sorted(by_form.items(), key=lambda x: -x[1]):
            pct = v / len(samples) * 100
            print(f"  {k}: {v} ({pct:.0f}%)")

    # By response type
    if samples and "response_type" in samples[0]:
        by_type = {}
        for s in samples:
            rtype = s.get("response_type", "unknown")
            by_type[rtype] = by_type.get(rtype, 0) + 1

        print("\nBy response type:")
        for k, v in sorted(by_type.items(), key=lambda x: -x[1])[:10]:
            pct = v / len(samples) * 100
            print(f"  {k}: {v} ({pct:.0f}%)")

    # Unique contacts
    contacts = set(s.get("contact", "") for s in samples)
    print(f"\nUnique contacts: {len(contacts)}")

    print(f"\nNext: python scripts/run_models_on_test_set.py")


def show_examples(n: int = 5):
    """Show example test cases with LLM analysis."""
    if not TEST_SET_FILE.exists():
        print("No test set yet.")
        return

    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))

    random.shuffle(samples)

    print("\n" + "=" * 60)
    print(f"EXAMPLE TEST CASES ({n})")
    print("=" * 60)

    for s in samples[:n]:
        print(f"\n[{s['contact']}]")
        if "relationship" in s:
            print(f"  LLM Analysis: {s.get('relationship')} | {s.get('formality')} | tone: {s.get('their_tone')}")
            print(f"  Your style: {s.get('my_style', 'n/a')}")
            print(f"  Response type: {s.get('response_type', 'n/a')}")
        print("-" * 40)

        # Show last few lines of conversation
        conv_lines = s["conversation"].split("\n")
        for line in conv_lines[-6:]:
            print(f"  {line}")

        print(f"\n  → YOUR REPLY: \"{s['gold_response']}\"")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--examples", type=int, metavar="N", help="Show N examples")
    parser.add_argument("--limit", type=int, default=None, help="Max samples (default: all)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.examples:
        show_examples(args.examples)
    else:
        create_test_set(limit=args.limit, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
