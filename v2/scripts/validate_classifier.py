#!/usr/bin/env python3
"""Validate the intent classifier against human labels.

If our classifier is only 60% accurate, then our "28% intent match"
metric is unreliable - we might actually be doing better (or worse).

This script:
1. Shows samples and asks for human labels
2. Compares classifier predictions to human labels
3. Reports accuracy, confusion matrix

Usage:
    python scripts/validate_classifier.py --samples 30
    python scripts/validate_classifier.py --auto  # Use heuristics instead of human
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")


class IntentClassifier:
    """Same classifier we use in experiments."""

    RESPONSE_ANCHORS = {
        "accept": [
            "yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely",
            "yep", "let's do it", "i'm down", "for sure", "bet", "count me in",
        ],
        "decline": [
            "no", "nah", "can't make it", "sorry can't", "busy", "not going",
            "pass", "maybe later", "i can't", "won't be able to",
        ],
        "question": [
            "what time", "when is it", "where at", "why", "how", "what's the plan",
            "which one", "who's coming", "what happened", "where", "when",
        ],
        "reaction": [
            "lol", "haha", "nice", "wow", "crazy", "damn", "omg", "that's funny",
            "hilarious", "no way", "wild", "bruh", "fr", "congrats", "ayy",
        ],
        "info": [
            "i'll be there at", "at 5pm", "tomorrow morning", "the address is",
            "it's at", "i'm at", "heading there now", "running late", "omw",
        ],
        "acknowledge": [
            "ok", "got it", "understood", "cool", "bet", "alright", "noted",
            "aight", "k", "np", "sounds good", "perfect", "works for me",
        ],
    }

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchor_embeddings = {}
        for intent, phrases in self.RESPONSE_ANCHORS.items():
            self.anchor_embeddings[intent] = self.model.encode(
                phrases, normalize_embeddings=True
            )

    def classify(self, message: str) -> tuple[str, float, dict]:
        """Returns (intent, confidence, all_scores)."""
        if not message.strip():
            return "unknown", 0.0, {}

        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]

        scores = {}
        for intent, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            scores[intent] = max_sim

        best_intent = max(scores, key=scores.get)
        return best_intent, scores[best_intent], scores


def load_test_set(limit: int) -> list[dict]:
    samples = []
    with open(CLEAN_TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def auto_label(response: str) -> str:
    """Heuristic labeling for quick validation."""
    r = response.lower().strip()

    # Question indicators
    if "?" in response or any(w in r for w in ["what", "when", "where", "why", "how", "who"]):
        return "question"

    # Strong accept
    if any(w in r for w in ["yeah", "yes", "sure", "down", "i'm in", "definitely", "yep", "bet"]):
        if "can't" not in r and "no" not in r:
            return "accept"

    # Strong decline
    if any(w in r for w in ["no", "nah", "can't", "sorry", "busy", "pass"]):
        return "decline"

    # Reaction indicators
    if any(w in r for w in ["lol", "haha", "damn", "omg", "wild", "crazy", "fr", "bruh", "nice", "wow"]):
        return "reaction"

    # Info indicators (giving specific details)
    if any(w in r for w in ["omw", "on my way", "at ", "i'll be", "running late", "heading"]):
        return "info"

    # Acknowledge
    if any(w in r for w in ["ok", "got it", "cool", "alright", "np", "k", "aight"]):
        return "acknowledge"

    return "unclear"


def run_validation(n_samples: int, auto_mode: bool):
    print(f"\n{'='*70}")
    print("INTENT CLASSIFIER VALIDATION")
    print(f"{'='*70}\n")

    print("Loading classifier...")
    classifier = IntentClassifier()

    print("Loading test set...")
    samples = load_test_set(n_samples)

    results = []

    print(f"\n{'='*70}")
    if auto_mode:
        print("Using automatic heuristic labels (approximate)")
    else:
        print("Manual labeling mode")
        print("For each response, enter the intent:")
        print("  a=accept, d=decline, q=question, r=reaction, i=info, k=acknowledge, s=skip")
    print(f"{'='*70}\n")

    intent_map = {"a": "accept", "d": "decline", "q": "question",
                  "r": "reaction", "i": "info", "k": "acknowledge"}

    for idx, sample in enumerate(samples):
        gold = sample["gold_response"]
        context = sample.get("conversation", "")[-150:]

        # Get classifier prediction
        pred_intent, confidence, all_scores = classifier.classify(gold)

        if auto_mode:
            human_label = auto_label(gold)
            if human_label == "unclear":
                continue  # Skip unclear cases for auto mode
        else:
            # Show context and response
            print(f"\n[{idx+1}/{n_samples}]")
            print(f"  Context: ...{context}")
            print(f"  Response: \"{gold}\"")
            print(f"  Classifier says: {pred_intent} ({confidence:.2f})")
            print(f"  Scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(all_scores.items(), key=lambda x: -x[1])[:3])}")

            while True:
                choice = input("  Your label (a/d/q/r/i/k/s): ").strip().lower()
                if choice == 's':
                    human_label = None
                    break
                elif choice in intent_map:
                    human_label = intent_map[choice]
                    break
                else:
                    print("  Invalid choice, try again")

            if human_label is None:
                continue

        results.append({
            "response": gold,
            "classifier": pred_intent,
            "confidence": confidence,
            "human": human_label,
            "correct": pred_intent == human_label,
        })

    # Calculate metrics
    n = len(results)
    if n == 0:
        print("No results to evaluate!")
        return

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n

    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {correct}/{n} ({accuracy*100:.1f}%)")

    # Confusion matrix
    print(f"\n--- CONFUSION MATRIX ---")
    intents = ["accept", "decline", "question", "reaction", "info", "acknowledge"]

    # Count predictions vs human labels
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["human"]][r["classifier"]] += 1

    # Print header
    print(f"{'Human↓ / Pred→':15}", end="")
    for intent in intents:
        print(f"{intent[:6]:>8}", end="")
    print(f"{'Total':>8}")

    # Print rows
    for human_intent in intents:
        row_total = sum(confusion[human_intent].values())
        if row_total == 0:
            continue
        print(f"{human_intent:15}", end="")
        for pred_intent in intents:
            count = confusion[human_intent][pred_intent]
            if count > 0:
                print(f"{count:>8}", end="")
            else:
                print(f"{'·':>8}", end="")
        print(f"{row_total:>8}")

    # Per-intent accuracy
    print(f"\n--- PER-INTENT ACCURACY ---")
    for intent in intents:
        intent_samples = [r for r in results if r["human"] == intent]
        if intent_samples:
            intent_correct = sum(1 for r in intent_samples if r["correct"])
            print(f"  {intent:12}: {intent_correct}/{len(intent_samples)} ({intent_correct/len(intent_samples)*100:.0f}%)")

    # Show misclassified examples
    print(f"\n--- MISCLASSIFIED EXAMPLES ---")
    misclassified = [r for r in results if not r["correct"]][:10]
    for r in misclassified:
        print(f"  \"{r['response'][:40]}\"")
        print(f"    Classifier: {r['classifier']} | Human: {r['human']}")

    # Save results
    save_path = Path("results/experiments/classifier_validation.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "n_samples": n,
            "accuracy": accuracy,
            "auto_mode": auto_mode,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {save_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--auto", action="store_true",
                        help="Use heuristic labels instead of manual")
    args = parser.parse_args()

    run_validation(args.samples, args.auto)


if __name__ == "__main__":
    main()
