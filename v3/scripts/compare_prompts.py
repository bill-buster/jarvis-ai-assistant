#!/usr/bin/env python3
"""Compare LEGACY vs CONVERSATION prompt strategies.

Usage:
    # Compare both strategies on test messages
    uv run python scripts/compare_prompts.py

    # Test with specific strategy
    JARVIS__GENERATION__PROMPT_STRATEGY=conversation uv run python scripts/compare_prompts.py

    # Run N iterations for statistical comparison
    uv run python scripts/compare_prompts.py --iterations 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PromptComparison:
    """Results from comparing both prompt strategies."""

    message: str
    legacy_replies: list[str]
    legacy_prompt: str
    legacy_time_ms: float
    conversation_replies: list[str]
    conversation_prompt: str
    conversation_time_ms: float


TEST_SCENARIOS = [
    # Simple questions
    {
        "messages": [
            {"text": "Hey want to grab dinner tonight?", "is_from_me": False, "sender": "Alice"}
        ]
    },
    {"messages": [{"text": "Are you coming to the party?", "is_from_me": False, "sender": "Bob"}]},
    {"messages": [{"text": "What time works for you?", "is_from_me": False, "sender": "Carol"}]},
    # Statements
    {"messages": [{"text": "Just got home", "is_from_me": False, "sender": "Dave"}]},
    {"messages": [{"text": "The meeting got moved to 3pm", "is_from_me": False, "sender": "Eve"}]},
    # Emotional
    {"messages": [{"text": "I got the job!!!", "is_from_me": False, "sender": "Frank"}]},
    {"messages": [{"text": "Today was rough", "is_from_me": False, "sender": "Grace"}]},
    # Multi-turn conversations
    {
        "messages": [
            {"text": "Hey!", "is_from_me": False, "sender": "Hank"},
            {"text": "Hi!", "is_from_me": True},
            {"text": "You free this weekend?", "is_from_me": False, "sender": "Hank"},
        ]
    },
    {
        "messages": [
            {"text": "Did you see the game last night?", "is_from_me": False, "sender": "Ivy"},
            {"text": "No I missed it", "is_from_me": True},
            {"text": "It was crazy! OT win", "is_from_me": False, "sender": "Ivy"},
        ]
    },
]


def run_comparison(model_loader, iterations: int = 1) -> list[PromptComparison]:
    """Run comparison between both prompt strategies."""
    from core.config import PromptStrategy, settings
    from core.generation import ReplyGenerator

    results = []

    for scenario in TEST_SCENARIOS:
        messages = scenario["messages"]
        last_msg = messages[-1]["text"]

        legacy_times = []
        legacy_replies_all = []
        legacy_prompt = ""

        conversation_times = []
        conversation_replies_all = []
        conversation_prompt = ""

        for _ in range(iterations):
            # Test LEGACY strategy
            settings.generation.prompt_strategy = PromptStrategy.LEGACY
            generator = ReplyGenerator(model_loader)
            start = time.time()
            result = generator.generate_replies(messages, chat_id=f"test-{hash(last_msg)}")
            legacy_times.append((time.time() - start) * 1000)
            legacy_replies_all.extend([r.text for r in result.replies])
            legacy_prompt = result.prompt_used

            # Test CONVERSATION strategy
            settings.generation.prompt_strategy = PromptStrategy.CONVERSATION
            generator = ReplyGenerator(model_loader)
            start = time.time()
            result = generator.generate_replies(messages, chat_id=f"test-conv-{hash(last_msg)}")
            conversation_times.append((time.time() - start) * 1000)
            conversation_replies_all.extend([r.text for r in result.replies])
            conversation_prompt = result.prompt_used

        # Deduplicate replies
        legacy_replies = list(dict.fromkeys(legacy_replies_all))
        conversation_replies = list(dict.fromkeys(conversation_replies_all))

        results.append(
            PromptComparison(
                message=last_msg,
                legacy_replies=legacy_replies[:3],
                legacy_prompt=legacy_prompt,
                legacy_time_ms=sum(legacy_times) / len(legacy_times),
                conversation_replies=conversation_replies[:3],
                conversation_prompt=conversation_prompt,
                conversation_time_ms=sum(conversation_times) / len(conversation_times),
            )
        )

    return results


def print_results(results: list[PromptComparison], show_prompts: bool = False) -> None:
    """Pretty-print comparison results."""
    print("\n" + "=" * 70)
    print("PROMPT STRATEGY COMPARISON: LEGACY vs CONVERSATION")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"Message: \"{r.message}\"")
        print()
        print(f"LEGACY ({r.legacy_time_ms:.0f}ms):")
        for reply in r.legacy_replies:
            print(f"  -> {reply}")
        print()
        print(f"CONVERSATION ({r.conversation_time_ms:.0f}ms):")
        for reply in r.conversation_replies:
            print(f"  -> {reply}")

        if show_prompts:
            print()
            print("LEGACY PROMPT:")
            print("-" * 40)
            print(r.legacy_prompt[:500] + "..." if len(r.legacy_prompt) > 500 else r.legacy_prompt)
            print()
            print("CONVERSATION PROMPT:")
            print("-" * 40)
            print(
                r.conversation_prompt[:500] + "..."
                if len(r.conversation_prompt) > 500
                else r.conversation_prompt
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_legacy = sum(r.legacy_time_ms for r in results) / len(results)
    avg_conv = sum(r.conversation_time_ms for r in results) / len(results)
    print(f"Average time - LEGACY: {avg_legacy:.0f}ms, CONVERSATION: {avg_conv:.0f}ms")
    print()
    print("To switch strategies, set the environment variable:")
    print("  JARVIS__GENERATION__PROMPT_STRATEGY=conversation")
    print("Or modify v3/core/config.py directly.")


def main():
    parser = argparse.ArgumentParser(description="Compare prompt strategies")
    parser.add_argument(
        "--iterations", "-n", type=int, default=1, help="Iterations per scenario"
    )
    parser.add_argument("--show-prompts", "-p", action="store_true", help="Show full prompts")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument("--mock", action="store_true", help="Use mock model (no real generation)")
    args = parser.parse_args()

    if args.mock:
        # Use mock for testing without loading model
        from unittest.mock import MagicMock

        model_loader = MagicMock()
        model_loader.is_loaded = True
        model_loader.current_model = "mock-model"
        model_loader.generate.return_value = MagicMock(
            text="sounds good!",
            formatted_prompt="<mock>",
        )
        print("Using MOCK model loader (no real generation)")
    else:
        # Load real model
        print("Loading model...")
        from core.models.loader import ModelLoader

        model_loader = ModelLoader()
        print(f"Loaded: {model_loader.current_model}")

    print(f"Running {args.iterations} iteration(s) per scenario...")
    results = run_comparison(model_loader, iterations=args.iterations)

    print_results(results, show_prompts=args.show_prompts)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
