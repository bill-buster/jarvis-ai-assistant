#!/usr/bin/env python3
"""Human evaluation of model responses.

Presents randomized A/B/C comparisons and records your preferences.

Usage:
    python scripts/human_eval.py
    python scripts/human_eval.py --samples 50
    python scripts/human_eval.py --results  # Show results so far
"""

import argparse
import json
import random
import sys
from pathlib import Path

RESULTS_FILE = Path("results/experiment/human_eval_results.json")


def load_experiment_data():
    """Load data from the experiment."""
    experiment_files = sorted(Path("results/experiment").glob("experiment_*.jsonl"))
    if not experiment_files:
        print("No experiment data found. Run model_experiment.py first.")
        sys.exit(1)

    latest = experiment_files[-1]
    print(f"Loading from: {latest.name}")

    samples = []
    with open(latest) as f:
        for line in f:
            data = json.loads(line)
            # Skip if any model failed
            if len(data.get("responses", {})) == 3:
                samples.append(data)

    return samples


def load_results():
    """Load existing results."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"rankings": [], "skipped": 0}


def save_results(results):
    """Save results."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def show_results():
    """Display current results."""
    results = load_results()

    if not results["rankings"]:
        print("No rankings yet. Run: python scripts/human_eval.py")
        return

    print("\n" + "=" * 60)
    print("HUMAN EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total rankings: {len(results['rankings'])}")
    print(f"Skipped: {results['skipped']}")

    # Count wins by model
    wins = {"qwen3-0.6b": 0, "lfm2.5-1.2b": 0, "lfm2-2.6b-exp": 0}
    avg_ranks = {"qwen3-0.6b": [], "lfm2.5-1.2b": [], "lfm2-2.6b-exp": []}

    for r in results["rankings"]:
        ranking = r["ranking"]  # List of model_ids from best to worst
        for i, model_id in enumerate(ranking):
            if i == 0:
                wins[model_id] += 1
            avg_ranks[model_id].append(i + 1)  # 1-indexed rank

    total = len(results["rankings"])

    print("\n" + "-" * 40)
    print("WIN RATES (1st place)")
    print("-" * 40)
    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        pct = wins[model_id] / total * 100 if total > 0 else 0
        avg = sum(avg_ranks[model_id]) / len(avg_ranks[model_id]) if avg_ranks[model_id] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {model_id:20} [{bar}] {pct:5.1f}% (avg rank: {avg:.2f})")

    print()


def run_eval(num_samples: int = 30):
    """Run human evaluation."""
    samples = load_experiment_data()
    results = load_results()

    # Track which samples we've already rated
    rated_ids = {r["sample_id"] for r in results["rankings"]}

    # Filter to unrated samples
    unrated = [s for s in samples if s["sample_id"] not in rated_ids]
    random.shuffle(unrated)

    if not unrated:
        print("All samples have been rated!")
        show_results()
        return

    print("\n" + "=" * 60)
    print("HUMAN EVALUATION")
    print("=" * 60)
    print(f"Samples to rate: {min(num_samples, len(unrated))}")
    print(f"Already rated: {len(rated_ids)}")
    print()
    print("Instructions:")
    print("  - You'll see the conversation context + 3 response options (A/B/C)")
    print("  - Models are randomized so you're judging blind")
    print("  - Rank them from best to worst (e.g., 'ABC' or 'BCA')")
    print("  - Type 's' to skip, 'q' to quit")
    print("-" * 60)

    models = ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]

    for i, sample in enumerate(unrated[:num_samples]):
        print(f"\n[{i+1}/{min(num_samples, len(unrated))}]")
        print(f"Contact: {sample.get('contact', '?')} ({sample.get('relationship', '?')})")
        print(f"Intent: {sample.get('intent', '?')} | Mood: {sample.get('mood', '?')}")

        # Show full prompt if available, otherwise just last message
        if sample.get("prompt"):
            print(f"\n--- PROMPT (what models saw) ---")
            print(sample["prompt"])
            print("--- END PROMPT ---")
        else:
            print(f"\n⚠️  Full context not saved. Last message only:")
            print(f"\"{sample['last_message']}\"")
        print()

        # Shuffle model order for blind comparison
        responses = sample["responses"]
        shuffled = list(responses.items())
        random.shuffle(shuffled)

        labels = ["A", "B", "C"]
        label_to_model = {}

        for label, (model_id, resp) in zip(labels, shuffled):
            label_to_model[label] = model_id
            print(f"  {label}: \"{resp['text']}\"")

        print()

        while True:
            choice = input("Rank best→worst (e.g., ABC) or s/q: ").strip().upper()

            if choice == "Q":
                save_results(results)
                print("\nSaved. Exiting.")
                show_results()
                return

            if choice == "S":
                results["skipped"] += 1
                break

            if len(choice) == 3 and set(choice) == {"A", "B", "C"}:
                # Convert labels to model IDs
                ranking = [label_to_model[label] for label in choice]
                results["rankings"].append({
                    "sample_id": sample["sample_id"],
                    "last_message": sample["last_message"],
                    "ranking": ranking,
                    "responses": {label_to_model[l]: responses[label_to_model[l]]["text"]
                                  for l in labels}
                })
                save_results(results)
                break

            print("  Invalid. Enter 3 letters (ABC/ACB/BAC/etc), 's' to skip, 'q' to quit")

    print("\nDone!")
    show_results()


def main():
    parser = argparse.ArgumentParser(description="Human evaluation of model responses")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to rate")
    parser.add_argument("--results", action="store_true", help="Show results only")
    args = parser.parse_args()

    if args.results:
        show_results()
    else:
        run_eval(args.samples)


if __name__ == "__main__":
    main()
