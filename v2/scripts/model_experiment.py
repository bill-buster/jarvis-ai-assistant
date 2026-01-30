#!/usr/bin/env python3
"""30-minute model comparison experiment.

Runs all 3 models through many real iMessage conversations,
tracking quality scores, win rates, and response patterns.

Usage:
    python scripts/model_experiment.py --duration 30
    python scripts/model_experiment.py --duration 5 --quick  # Quick test
"""

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExperimentResult:
    """Result from a single conversation test."""

    sample_id: int
    contact: str
    relationship: str
    intent: str
    mood: str
    topic: str
    last_message: str
    prompt: str  # Full prompt shown to models

    # Model responses
    responses: dict  # model_id -> {text, score, time_ms}

    # Winner
    winner_model: str
    winner_text: str
    winner_score: float

    timestamp: str


def load_models():
    """Load all 3 models."""
    from core.generation.multi_generator import MultiModelGenerator

    models = [
        ('qwen3-0.6b', 'fast'),
        ('lfm2.5-1.2b', 'balanced'),
        ('lfm2-2.6b-exp', 'best'),
    ]

    return MultiModelGenerator(models=models, preload=True)


def get_conversation_samples(limit: int = 500):
    """Get many conversation samples for testing."""
    try:
        from core.imessage.reader import MessageReader
        from core.generation.context_analyzer import ContextAnalyzer
    except ImportError:
        print("Error: Required modules not available")
        return []

    reader = MessageReader()
    context_analyzer = ContextAnalyzer()

    conversations = reader.get_conversations(limit=200)
    samples = []
    seen = set()

    spam_keywords = [
        "reward points", "expire", "your order", "tracking",
        "legal representation", "law firm", "utm_source",
        "click here", "unsubscribe", "verification code",
    ]

    for conv in conversations:
        if len(samples) >= limit:
            break

        try:
            messages = reader.get_messages(conv.chat_id, limit=50)
            if not messages or len(messages) < 5:
                continue

            messages = list(reversed(messages))

            # Skip spam
            contact = conv.display_name or ""
            participants = conv.participants or []
            participant_str = participants[0] if participants else ""

            if participant_str.isdigit() and 5 <= len(participant_str) <= 6:
                continue

            all_text = " ".join((m.text or "").lower() for m in messages[-10:])
            if sum(1 for kw in spam_keywords if kw in all_text) >= 2:
                continue

            # Find messages from them that need replies
            for i in range(len(messages) - 1, max(0, len(messages) - 20), -1):
                msg = messages[i]
                if msg.is_from_me:
                    continue

                text = (msg.text or "").strip()
                if len(text) < 3 or len(text) > 150:
                    continue

                # Skip reactions
                if any(p in text.lower() for p in ["loved", "liked", "emphasized", "laughed at"]):
                    continue

                # Dedupe
                key = text[:30].lower()
                if key in seen:
                    continue
                seen.add(key)

                # Get context (messages before this one)
                context_msgs = messages[max(0, i-15):i+1]
                msg_dicts = [
                    {"text": (m.text or "").strip(), "is_from_me": m.is_from_me}
                    for m in context_msgs if (m.text or "").strip()
                ]

                if not msg_dicts:
                    continue

                # Analyze context
                context = context_analyzer.analyze(msg_dicts)

                samples.append({
                    "contact": contact or participant_str[:15],
                    "messages": msg_dicts,
                    "context": context,
                    "last_message": text,
                })

                if len(samples) >= limit:
                    break

        except Exception:
            continue

    # Shuffle for variety
    random.shuffle(samples)
    return samples


def build_prompt(sample: dict) -> str:
    """Build prompt from sample."""
    messages = sample["messages"]
    context = sample["context"]

    # Style hints based on context
    style_parts = ["brief", "casual"]

    rel = context.relationship.value
    if rel == "close_friend":
        style_parts.append("friendly")
    elif rel == "work":
        style_parts.append("professional")
    elif rel == "romantic":
        style_parts.append("warm")

    if context.mood == "positive":
        style_parts.append("upbeat")
    elif context.mood == "negative":
        style_parts.append("supportive")

    style_hint = ", ".join(style_parts)

    # Format conversation
    lines = []
    for msg in messages[-12:]:
        text = msg.get("text", "")
        if not text:
            continue
        prefix = "me:" if msg.get("is_from_me") else "them:"
        lines.append(f"{prefix} {text}")

    conversation = "\n".join(lines)
    return f"[{style_hint}]\n\n{conversation}\nme:"


def run_experiment(duration_minutes: int = 30, output_dir: str = "results/experiment"):
    """Run the full experiment."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"experiment_{timestamp}.jsonl"
    summary_file = output_path / f"summary_{timestamp}.json"

    print("=" * 70)
    print(f"MODEL COMPARISON EXPERIMENT ({duration_minutes} minutes)")
    print("=" * 70)
    print(f"Results: {results_file}")
    print()

    # Load models
    print("Loading models...")
    generator = load_models()

    # Get samples
    print("\nLoading conversation samples...")
    samples = get_conversation_samples(limit=1000)
    print(f"Found {len(samples)} samples")

    if not samples:
        print("No samples found!")
        return

    # Stats tracking
    stats = {
        "total_tests": 0,
        "wins_by_model": defaultdict(int),
        "wins_by_relationship": defaultdict(lambda: defaultdict(int)),
        "wins_by_intent": defaultdict(lambda: defaultdict(int)),
        "avg_score_by_model": defaultdict(list),
        "avg_time_by_model": defaultdict(list),
        "errors": 0,
    }

    # Run experiment
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    sample_idx = 0

    print(f"\nStarting experiment at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Will run until {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print("-" * 70)

    try:
        with open(results_file, "w") as f:
            while time.time() < end_time:
                # Get next sample (cycle through)
                sample = samples[sample_idx % len(samples)]
                sample_idx += 1

                context = sample["context"]
                prompt = build_prompt(sample)

                try:
                    # Generate from all models
                    result = generator.generate(prompt, max_tokens=40, temperature=0.4)

                    # Collect responses
                    responses = {}
                    for reply in result.replies:
                        responses[reply.model_id] = {
                            "text": reply.text,
                            "score": reply.quality_score,
                            "time_ms": reply.generation_time_ms,
                        }
                        stats["avg_score_by_model"][reply.model_id].append(reply.quality_score)
                        stats["avg_time_by_model"][reply.model_id].append(reply.generation_time_ms)

                    # Find winner
                    winner = result.best_quality
                    if winner:
                        stats["wins_by_model"][winner.model_id] += 1
                        stats["wins_by_relationship"][context.relationship.value][winner.model_id] += 1
                        stats["wins_by_intent"][context.intent.value][winner.model_id] += 1

                    stats["total_tests"] += 1

                    # Save result
                    exp_result = ExperimentResult(
                        sample_id=sample_idx,
                        contact=sample["contact"][:10],
                        relationship=context.relationship.value,
                        intent=context.intent.value,
                        mood=context.mood,
                        topic=context.topic,
                        last_message=sample["last_message"][:50],
                        prompt=prompt,  # Save full prompt for human eval
                        responses=responses,
                        winner_model=winner.model_id if winner else "",
                        winner_text=winner.text if winner else "",
                        winner_score=winner.quality_score if winner else 0,
                        timestamp=datetime.now().isoformat(),
                    )
                    f.write(json.dumps(asdict(exp_result)) + "\n")
                    f.flush()

                    # Progress update every 20 tests
                    if stats["total_tests"] % 20 == 0:
                        elapsed = time.time() - start_time
                        remaining = end_time - time.time()
                        rate = stats["total_tests"] / elapsed * 60

                        # Current standings
                        total = stats["total_tests"]
                        standings = []
                        for model_id in ["lfm2.5-1.2b", "lfm2-2.6b-exp", "qwen3-0.6b"]:
                            wins = stats["wins_by_model"][model_id]
                            pct = wins / total * 100 if total > 0 else 0
                            standings.append(f"{model_id}: {pct:.0f}%")

                        print(f"[{stats['total_tests']:4d} tests | {elapsed/60:.1f}m elapsed | {remaining/60:.1f}m left | {rate:.0f}/min]")
                        print(f"    Win rates: {' | '.join(standings)}")

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] % 10 == 0:
                        print(f"  (Errors: {stats['errors']})")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")

    # Final stats
    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Duration: {elapsed/60:.1f} minutes")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Errors: {stats['errors']}")
    print(f"Rate: {stats['total_tests'] / elapsed * 60:.1f} tests/minute")

    # Win rates
    print("\n" + "-" * 40)
    print("OVERALL WIN RATES")
    print("-" * 40)
    total = stats["total_tests"]
    for model_id in ["lfm2.5-1.2b", "lfm2-2.6b-exp", "qwen3-0.6b"]:
        wins = stats["wins_by_model"][model_id]
        pct = wins / total * 100 if total > 0 else 0
        avg_score = sum(stats["avg_score_by_model"][model_id]) / len(stats["avg_score_by_model"][model_id]) if stats["avg_score_by_model"][model_id] else 0
        avg_time = sum(stats["avg_time_by_model"][model_id]) / len(stats["avg_time_by_model"][model_id]) if stats["avg_time_by_model"][model_id] else 0
        print(f"  {model_id:20} {wins:4d} wins ({pct:5.1f}%) | avg score: {avg_score:.2f} | avg time: {avg_time:.0f}ms")

    # Win rates by relationship
    print("\n" + "-" * 40)
    print("WIN RATES BY RELATIONSHIP")
    print("-" * 40)
    for rel_type, model_wins in sorted(stats["wins_by_relationship"].items()):
        total_rel = sum(model_wins.values())
        if total_rel < 5:
            continue
        print(f"\n  {rel_type}:")
        for model_id in ["lfm2.5-1.2b", "lfm2-2.6b-exp", "qwen3-0.6b"]:
            wins = model_wins[model_id]
            pct = wins / total_rel * 100 if total_rel > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {model_id:20} [{bar}] {pct:5.1f}%")

    # Win rates by intent
    print("\n" + "-" * 40)
    print("WIN RATES BY INTENT")
    print("-" * 40)
    for intent_type, model_wins in sorted(stats["wins_by_intent"].items()):
        total_intent = sum(model_wins.values())
        if total_intent < 5:
            continue
        print(f"\n  {intent_type}:")
        for model_id in ["lfm2.5-1.2b", "lfm2-2.6b-exp", "qwen3-0.6b"]:
            wins = model_wins[model_id]
            pct = wins / total_intent * 100 if total_intent > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {model_id:20} [{bar}] {pct:5.1f}%")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "duration_minutes": elapsed / 60,
        "total_tests": stats["total_tests"],
        "errors": stats["errors"],
        "tests_per_minute": stats["total_tests"] / elapsed * 60,
        "wins_by_model": dict(stats["wins_by_model"]),
        "win_rates": {
            model_id: stats["wins_by_model"][model_id] / total * 100 if total > 0 else 0
            for model_id in stats["wins_by_model"]
        },
        "avg_scores": {
            model_id: sum(scores) / len(scores) if scores else 0
            for model_id, scores in stats["avg_score_by_model"].items()
        },
        "avg_times": {
            model_id: sum(times) / len(times) if times else 0
            for model_id, times in stats["avg_time_by_model"].items()
        },
        "wins_by_relationship": {k: dict(v) for k, v in stats["wins_by_relationship"].items()},
        "wins_by_intent": {k: dict(v) for k, v in stats["wins_by_intent"].items()},
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    # Cleanup
    generator.unload()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Model comparison experiment")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in minutes (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (1 minute)",
    )
    args = parser.parse_args()

    duration = 1 if args.quick else args.duration
    run_experiment(duration_minutes=duration, output_dir=args.output)


if __name__ == "__main__":
    main()
