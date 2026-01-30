"""Generate and evaluate reply suggestions.

This script:
1. Loads real conversations from your iMessage database
2. Generates reply suggestions for each
3. Saves them for human evaluation
4. Shows metrics on quality

Usage:
    python scripts/evaluate_replies.py --samples 30
    python scripts/evaluate_replies.py --chat-id <chat_id> --interactive
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add v3 to path (must be before core imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings


def json_serializer(obj):
    """Handle non-serializable objects like datetime."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def is_real_contact(display_name: str | None, participants: list[str] | None) -> bool:
    """Check if this is a real contact (not spam/business/email).

    Args:
        display_name: The conversation display name
        participants: List of participant identifiers

    Returns:
        True if this looks like a real human contact with a proper name
    """
    if not display_name:
        return False

    name = display_name.strip()

    # Skip if display name looks like a phone number
    if name.startswith("+") or name.startswith("1"):
        return False
    if name.isdigit() or (len(name) <= 6 and name.replace("-", "").isdigit()):
        return False

    # Skip email addresses (spam often comes from random emails)
    if "@" in name:
        return False

    # Skip common spam/business patterns
    spam_patterns = [
        "verify", "code", "alert", "notification", "noreply", "donotreply",
        "verizon", "t-mobile", "tmobile", "sprint",  # carriers (not "att" - too common in names)
        "amazon", "uber", "lyft", "doordash", "grubhub",  # services
        "chase", "wells fargo", "citi",  # banks (not just "bank")
        "pharmacy", "cvs", "walgreens",  # pharmacy
        "clinic", "hospital",  # medical (not "doctor" or "health" - could be names)
    ]
    name_lower = name.lower()
    if any(p in name_lower for p in spam_patterns):
        return False

    # Must have at least 2 characters that are letters
    letter_count = sum(1 for c in name if c.isalpha())
    if letter_count < 2:
        return False

    # Must look like a name (has a space for first/last, or is a short first name)
    # Single word names are OK if they're 3+ letters (like "Mom", "Dad", "Rutu")
    has_space = " " in name
    is_short_name = len(name) >= 3 and len(name) <= 15 and name.isalpha()
    if not has_space and not is_short_name:
        # Check if it's a reasonable single-word name (letters with maybe some caps)
        alpha_chars = sum(1 for c in name if c.isalpha())
        if alpha_chars < 3:
            return False

    return True


def load_test_samples(
    n_samples: int | None = None,
    contacts_only: bool = False,
    one_to_one_only: bool = False,
) -> list[dict]:
    """Load test samples from iMessage database.

    Args:
        n_samples: Number of conversation samples to load
        contacts_only: If True, only include chats with real contact names
        one_to_one_only: If True, exclude group chats

    Returns:
        List of test cases with messages and context
    """
    from core.imessage import MessageReader

    if n_samples is None:
        n_samples = settings.api.default_conversation_limit

    filter_desc = []
    if contacts_only:
        filter_desc.append("contacts only")
    if one_to_one_only:
        filter_desc.append("1:1 only")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

    print(f"📱 Loading {n_samples} conversations from iMessage{filter_str}...")

    reader = MessageReader()
    try:
        if not reader.check_access():
            print("❌ Cannot access iMessage database")
            print("   Grant Full Disk Access in System Settings > Privacy & Security")
            return []

        # Get many more conversations to filter from (need buffer for filtering)
        conversations = reader.get_conversations(limit=n_samples * 10)

        samples = []
        skipped_no_name = 0
        skipped_group = 0

        for conv in conversations:
            if len(samples) >= n_samples:
                break

            # Filter: contacts only
            if contacts_only and not is_real_contact(conv.display_name, conv.participants):
                skipped_no_name += 1
                continue

            # Filter: 1:1 only (not group chats)
            if one_to_one_only:
                # Group chats have multiple participants or "+" in chat_id
                if conv.is_group or len(conv.participants) > 1:
                    skipped_group += 1
                    continue

            # Get last N messages from conversation
            messages = reader.get_messages(
                chat_id=conv.chat_id, limit=settings.api.generation_context_limit
            )

            if len(messages) < 2:
                continue

            # Only include if last message is from someone else (needs reply)
            if not messages[-1].is_from_me:
                samples.append(
                    {
                        "chat_id": conv.chat_id,
                        "display_name": conv.display_name,
                        "participants": conv.participants,
                        "messages": [
                            {
                                "text": m.text,
                                "sender": m.sender,
                                "is_from_me": m.is_from_me,
                                "timestamp": m.timestamp,
                            }
                            for m in reversed(messages)  # Chronological order
                        ],
                    }
                )

        print(f"✅ Loaded {len(samples)} test samples")
        if skipped_no_name:
            print(f"   (skipped {skipped_no_name} without contact names)")
        if skipped_group:
            print(f"   (skipped {skipped_group} group chats)")
        return samples

    finally:
        reader.close()


def generate_replies_for_sample(sample: dict, generator: Any, use_rag: bool = True) -> dict:
    """Generate replies for a single test sample.

    Args:
        sample: Test sample with messages
        generator: ReplyGenerator instance
        use_rag: Whether to use RAG retrieval

    Returns:
        Sample with generated replies added
    """
    messages = sample["messages"]
    chat_id = sample["chat_id"]

    # Get last incoming message for context
    last_incoming = None
    for m in reversed(messages):
        if not m["is_from_me"]:
            last_incoming = m["text"]
            break

    start_time = time.time()

    try:
        # Debug output
        print(f"\n      [DEBUG] Messages: {len(messages)}, chat_id: {chat_id[:30] if chat_id else 'None'}...")
        print(f"      [DEBUG] Last message: {last_incoming[:50] if last_incoming else 'None'}...")
        print(f"      [DEBUG] Calling generate_replies...", end=" ", flush=True)

        result = generator.generate_replies(
            messages=messages,
            chat_id=chat_id if use_rag else None,  # Disable RAG if requested
            num_replies=3,
            contact_name=sample.get("display_name"),
        )

        print(f"done!")
        generation_time = (time.time() - start_time) * 1000

        return {
            **sample,
            "generated_replies": [
                {"text": r.text, "reply_type": r.reply_type, "confidence": r.confidence}
                for r in result.replies
            ],
            "generation_time_ms": generation_time,
            "model_used": result.model_used,
            "context_summary": result.context.summary if hasattr(result, "context") else "",
            "past_replies_found": len(result.past_replies)
            if hasattr(result, "past_replies")
            else 0,
            "last_incoming": last_incoming,
        }

    except Exception as e:
        return {**sample, "error": str(e), "generated_replies": [], "generation_time_ms": 0}


def evaluate_samples(samples: list[dict]) -> None:
    """Interactive evaluation of generated replies.

    Asks user to rate each generated reply 1-5.
    """
    print("\n" + "=" * 60)
    print("HUMAN EVALUATION")
    print("=" * 60)
    print("\nRate each reply 1-5:")
    print("  1 = Terrible (would never send)")
    print("  2 = Bad (major issues)")
    print("  3 = Okay (minor issues)")
    print("  4 = Good (would probably send)")
    print("  5 = Perfect (exactly what I'd say)")
    print("\nPress 'q' to quit, 's' to skip\n")

    ratings = []

    for i, sample in enumerate(samples, 1):
        if "error" in sample:
            continue

        print(f"\n{'=' * 60}")
        print(f"Sample {i}/{len(samples)}: {sample['display_name']}")
        print(f"{'=' * 60}")

        # Show conversation context
        print("\n📱 Conversation:")
        for m in sample["messages"][-5:]:  # Last 5 messages
            sender = "You" if m["is_from_me"] else m["sender"]
            print(f"   {sender}: {m['text']}")

        # Show generated replies
        print(f"\n🤖 Generated Replies ({sample['generation_time_ms']:.0f}ms):")
        for j, reply in enumerate(sample["generated_replies"], 1):
            print(f'   {j}. "{reply["text"]}" ({reply["reply_type"]}, {reply["confidence"]:.2f})')

        # Get rating
        while True:
            try:
                user_input = input("\nRate best reply (1-5, q=quit, s=skip): ").strip().lower()

                if user_input == "q":
                    break
                elif user_input == "s":
                    ratings.append(None)
                    break
                elif user_input in ["1", "2", "3", "4", "5"]:
                    ratings.append(int(user_input))
                    break
                else:
                    print("   Please enter 1-5, q, or s")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break

        if user_input == "q":
            break

    # Calculate metrics
    valid_ratings = [r for r in ratings if r is not None]
    if valid_ratings:
        avg_rating = sum(valid_ratings) / len(valid_ratings)
        distribution = {i: valid_ratings.count(i) for i in range(1, 6)}

        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"\nSamples rated: {len(valid_ratings)}/{len(samples)}")
        print(f"Average rating: {avg_rating:.2f}/5.0")
        print("\nDistribution:")
        for score, count in distribution.items():
            bar = "█" * count
            print(f"  {score}: {bar} ({count})")

        # Save results
        results = {
            "total_samples": len(samples),
            "rated_samples": len(valid_ratings),
            "average_rating": avg_rating,
            "distribution": distribution,
            "samples": samples[: len(ratings)],
        }

        output_path = Path("results/evaluation_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=json_serializer)

        print(f"\n✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reply generation")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to evaluate")
    parser.add_argument("--chat-id", type=str, help="Evaluate specific chat only")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG (baseline comparison)")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive human evaluation"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/evaluation_samples.json", help="Output file"
    )
    parser.add_argument(
        "--contacts-only", "-c", action="store_true",
        help="Only include chats with real contact names (not phone numbers)"
    )
    parser.add_argument(
        "--one-to-one", "-1", action="store_true",
        help="Only include 1:1 chats (exclude group chats)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JARVIS v3 - Reply Generation Evaluation")
    print("=" * 60)

    # Load samples
    if args.chat_id:
        # Single chat mode
        from core.imessage import MessageReader

        reader = MessageReader()
        try:
            messages = reader.get_messages(chat_id=args.chat_id, limit=10)
            samples = [
                {
                    "chat_id": args.chat_id,
                    "display_name": "Test Chat",
                    "messages": [
                        {
                            "text": m.text,
                            "sender": m.sender,
                            "is_from_me": m.is_from_me,
                            "timestamp": m.timestamp,
                        }
                        for m in reversed(messages)
                    ],
                }
            ]
        finally:
            reader.close()
    else:
        samples = load_test_samples(
            args.samples,
            contacts_only=args.contacts_only,
            one_to_one_only=args.one_to_one,
        )

    if not samples:
        print("❌ No samples to evaluate")
        return

    # Initialize generator
    print("\n🤖 Initializing reply generator...")
    from core.generation.reply_generator import ReplyGenerator
    from core.models.loader import ModelLoader

    print("   Loading LFM2.5-1.2B model...")
    loader = ModelLoader()
    generator = ReplyGenerator(loader)
    print("   ✅ Generator ready\n")

    # Generate replies
    print(f"📝 Generating replies for {len(samples)} samples...")
    if args.no_rag:
        print("   (RAG disabled - baseline mode)")

    evaluated_samples = []
    for i, sample in enumerate(samples, 1):
        print(f"   Processing {i}/{len(samples)}: {sample['display_name'][:30]}...", end=" ")

        result = generate_replies_for_sample(sample, generator, use_rag=not args.no_rag)
        evaluated_samples.append(result)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ {result['generation_time_ms']:.0f}ms")

    # Save samples
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evaluated_samples, f, indent=2, default=json_serializer)

    print(f"\n✅ Generated samples saved to {output_path}")

    # Show statistics
    successful = [s for s in evaluated_samples if "error" not in s]
    if successful:
        avg_time = sum(s["generation_time_ms"] for s in successful) / len(successful)
        total_past = sum(s["past_replies_found"] for s in successful)

        print("\n📊 Statistics:")
        print(f"   Successful generations: {len(successful)}/{len(evaluated_samples)}")
        print(f"   Average generation time: {avg_time:.0f}ms")
        print(f"   Total past replies found: {total_past}")
        print(f"   Avg past replies per sample: {total_past / len(successful):.1f}")

    # Interactive evaluation
    if args.interactive:
        evaluate_samples(evaluated_samples)
    else:
        print("\n💡 Run with --interactive to rate replies manually")
        print("   python scripts/evaluate_replies.py --interactive")


if __name__ == "__main__":
    main()
