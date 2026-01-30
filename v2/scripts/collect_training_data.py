#!/usr/bin/env python3
"""Collect training data: write your response, then compare to models.

Workflow:
1. See conversation context
2. Write YOUR response (unbiased, without seeing models)
3. See what the models generated
4. Rank the models

Usage:
    python scripts/collect_training_data.py
    python scripts/collect_training_data.py --results
    python scripts/collect_training_data.py --export
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("results/training_data/responses.jsonl")


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
            if data.get("prompt") and len(data.get("responses", {})) == 3:
                samples.append(data)

    return samples


def load_existing_ids():
    """Load IDs we've already processed."""
    if not DATA_FILE.exists():
        return set()
    ids = set()
    with open(DATA_FILE) as f:
        for line in f:
            data = json.loads(line)
            ids.add(data["sample_id"])
    return ids


def save_entry(entry: dict):
    """Append entry to data file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def show_results():
    """Show collection stats and compare responses."""
    if not DATA_FILE.exists():
        print("No data collected yet.")
        return

    entries = []
    with open(DATA_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print("\n" + "=" * 60)
    print("TRAINING DATA RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(entries)}")

    # Win rates
    wins = {"qwen3-0.6b": 0, "lfm2.5-1.2b": 0, "lfm2-2.6b-exp": 0}
    for e in entries:
        if e.get("ranking"):
            wins[e["ranking"][0]] += 1

    total = len([e for e in entries if e.get("ranking")])

    print("\n" + "-" * 40)
    print("YOUR RANKINGS (which model was best)")
    print("-" * 40)
    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        pct = wins[model_id] / total * 100 if total else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {model_id:20} [{bar}] {pct:5.1f}% ({wins[model_id]}/{total})")

    # Gold response stats
    with_gold = [e for e in entries if e.get("gold_response")]
    print(f"\nGold responses collected: {len(with_gold)}")

    if with_gold:
        avg_len = sum(len(e["gold_response"]) for e in with_gold) / len(with_gold)
        print(f"Avg length: {avg_len:.0f} chars")

        # Compare gold to model responses
        print("\n" + "-" * 40)
        print("YOUR RESPONSE vs MODEL RESPONSES")
        print("-" * 40)

        # Show a few examples
        for e in with_gold[:5]:
            print(f"\nPrompt ending: ...{e['prompt'][-80:]}")
            print(f"  YOU:    \"{e['gold_response']}\"")
            for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
                text = e["model_responses"].get(model_id, "?")
                rank = e["ranking"].index(model_id) + 1 if e.get("ranking") else "?"
                print(f"  {model_id:15} (#{rank}): \"{text}\"")


def run_collection():
    """Run data collection."""
    samples = load_experiment_data()
    existing_ids = load_existing_ids()

    unprocessed = [s for s in samples if s["sample_id"] not in existing_ids]
    random.shuffle(unprocessed)

    if not unprocessed:
        if not samples:
            print("\n⚠️  No samples with prompts found!")
            print("Run: python scripts/model_experiment.py --duration 5")
            return
        print("All samples processed!")
        show_results()
        return

    print("\n" + "=" * 60)
    print("TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"Available: {len(unprocessed)} | Already done: {len(existing_ids)}")
    print()
    print("Workflow:")
    print("  1. See conversation → Write YOUR response")
    print("  2. See model responses → Rank them")
    print()
    print("Commands: 's' skip | 'q' quit")
    print("-" * 60)

    collected = 0

    for i, sample in enumerate(unprocessed):
        print(f"\n{'='*60}")
        print(f"[{i+1}] {sample.get('contact', '?')} ({sample.get('relationship', '?')}) - {sample.get('intent', '?')}")
        print("="*60)

        # Show prompt
        print("\n" + sample["prompt"])
        print()

        # Step 1: Get user's response FIRST (unbiased)
        print("What would YOU reply? (enter twice to submit, 's' skip, 'q' quit)")

        lines = []
        while True:
            line = input("> " if not lines else "  ")

            if line.lower() == "q":
                print(f"\nSaved {collected} entries.")
                show_results()
                return

            if line.lower() == "s":
                lines = None
                break

            if line == "" and lines:
                break

            if line:
                lines.append(line)

        if lines is None:
            continue

        gold_response = " ".join(lines).strip()

        if not gold_response:
            print("  (skipped - no response)")
            continue

        print(f"\n  Your response: \"{gold_response}\"")

        # Step 2: Show model responses
        print("\n" + "-" * 40)
        print("MODEL RESPONSES:")
        print("-" * 40)

        responses = sample["responses"]
        shuffled = list(responses.items())
        random.shuffle(shuffled)

        labels = ["A", "B", "C"]
        label_to_model = {}

        for label, (model_id, resp) in zip(labels, shuffled):
            label_to_model[label] = model_id
            print(f"  {label}: \"{resp['text']}\"")

        # Step 3: Rank models
        print()
        while True:
            choice = input("Rank best→worst (ABC/ACB/etc) or 's' skip: ").strip().upper()

            if choice == "S":
                ranking = None
                break

            if len(choice) == 3 and set(choice) == {"A", "B", "C"}:
                ranking = [label_to_model[label] for label in choice]
                break

            print("  Enter 3 letters or 's'")

        # Save
        entry = {
            "sample_id": sample["sample_id"],
            "contact": sample.get("contact"),
            "relationship": sample.get("relationship"),
            "intent": sample.get("intent"),
            "mood": sample.get("mood"),
            "prompt": sample["prompt"],
            "gold_response": gold_response,
            "model_responses": {mid: r["text"] for mid, r in responses.items()},
            "ranking": ranking,
            "timestamp": datetime.now().isoformat(),
        }
        save_entry(entry)
        collected += 1
        print(f"  ✓ Saved ({len(existing_ids) + collected} total)")

    print(f"\nDone! Collected {collected} entries.")
    show_results()


def export_data():
    """Export for fine-tuning/few-shot."""
    if not DATA_FILE.exists():
        print("No data to export.")
        return

    entries = []
    with open(DATA_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    gold_entries = [e for e in entries if e.get("gold_response")]

    # Export as prompt/completion pairs
    export_file = DATA_FILE.parent / "training_pairs.jsonl"
    with open(export_file, "w") as f:
        for e in gold_entries:
            f.write(json.dumps({
                "prompt": e["prompt"],
                "completion": e["gold_response"],
                "relationship": e.get("relationship"),
                "intent": e.get("intent"),
            }) + "\n")

    print(f"Exported {len(gold_entries)} pairs to: {export_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true", help="Show results")
    parser.add_argument("--export", action="store_true", help="Export training data")
    args = parser.parse_args()

    if args.results:
        show_results()
    elif args.export:
        export_data()
    else:
        run_collection()


if __name__ == "__main__":
    main()
