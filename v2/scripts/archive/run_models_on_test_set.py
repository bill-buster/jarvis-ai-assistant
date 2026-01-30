#!/usr/bin/env python3
"""Run all models on the test set and compare to your actual replies.

Usage:
    python scripts/run_models_on_test_set.py
    python scripts/run_models_on_test_set.py --compare  # Show comparisons
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
RESULTS_FILE = Path("results/test_set/model_results.jsonl")


def run_models():
    """Run all models on test set."""
    if not TEST_SET_FILE.exists():
        print("No test set. Run: python scripts/create_test_set.py")
        return

    # Load test set
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Test set: {len(samples)} samples (with your actual replies)")

    # Load models
    print("\nLoading models...")
    from core.generation.multi_generator import MultiModelGenerator

    models = [
        ('qwen3-0.6b', 'fast'),
        ('lfm2.5-1.2b', 'balanced'),
        ('lfm2-2.6b-exp', 'best'),
    ]

    generator = MultiModelGenerator(models=models, preload=True)

    # Run
    print(f"\nGenerating responses...")
    print("-" * 60)

    results = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        result = generator.generate(
            sample["prompt"],
            max_tokens=40,
            temperature=0.4,
            stop=["\n", "them:", "<|im_end|>", "<|eot_id|>", "<end_of_turn>"],
        )

        entry = {
            "id": sample["id"],
            "contact": sample["contact"],
            "relationship": sample["relationship"],
            "intent": sample["intent"],
            "prompt": sample["prompt"],
            "gold_response": sample["gold_response"],
            "model_responses": {
                r.model_id: {
                    "text": r.text,
                    "time_ms": r.generation_time_ms,
                }
                for r in result.replies
            }
        }
        results.append(entry)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining = (len(samples) - i - 1) / rate * 60
            print(f"  [{i+1}/{len(samples)}] {rate:.0f}/min, ~{remaining:.0f}s left")

    # Save
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start_time
    print(f"\n✓ Done in {elapsed:.0f}s ({len(samples)/elapsed*60:.0f}/min)")
    print(f"  Results: {RESULTS_FILE}")

    # Cleanup
    generator.unload()

    # Show quick comparison
    show_comparison(limit=10)


def show_comparison(limit: int = 20):
    """Show side-by-side comparison."""
    if not RESULTS_FILE.exists():
        print("No results. Run: python scripts/run_models_on_test_set.py")
        return

    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    print("\n" + "=" * 70)
    print("COMPARISON: Your replies vs Model outputs")
    print("=" * 70)

    for r in results[:limit]:
        print(f"\n[{r['contact']}] ({r['relationship']}) - {r['intent']}")

        # Show just the last few lines of prompt
        prompt_lines = r["prompt"].strip().split("\n")
        print(f"  ...{prompt_lines[-2] if len(prompt_lines) > 1 else ''}")
        print(f"  {prompt_lines[-1]}")

        print(f"\n  YOU:          \"{r['gold_response']}\"")
        for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
            resp = r["model_responses"].get(model_id, {})
            text = resp.get("text", "?")
            ms = resp.get("time_ms", 0)
            print(f"  {model_id:14} \"{text}\" ({ms:.0f}ms)")

        print("-" * 70)

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Avg times
    print("\nAvg generation time:")
    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        times = [r["model_responses"][model_id]["time_ms"] for r in results if model_id in r["model_responses"]]
        avg = sum(times) / len(times) if times else 0
        print(f"  {model_id:20} {avg:.0f}ms")

    # Avg response lengths
    print("\nAvg response length:")
    gold_lens = [len(r["gold_response"]) for r in results]
    print(f"  {'YOU':20} {sum(gold_lens)/len(gold_lens):.0f} chars")
    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        lens = [len(r["model_responses"][model_id]["text"]) for r in results if model_id in r["model_responses"]]
        avg = sum(lens) / len(lens) if lens else 0
        print(f"  {model_id:20} {avg:.0f} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Show comparison only")
    parser.add_argument("--limit", type=int, default=20, help="Samples to show")
    args = parser.parse_args()

    if args.compare:
        show_comparison(args.limit)
    else:
        run_models()


if __name__ == "__main__":
    main()
