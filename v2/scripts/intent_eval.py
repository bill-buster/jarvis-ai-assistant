#!/usr/bin/env python3
"""Intent-based evaluation for reply generation.

Instead of semantic similarity (which fails for "yeah" vs "sure"),
we compare the INTENT of gold vs generated responses.

Usage:
    python scripts/intent_eval.py --samples 50 --verbose     # Quick test
    python scripts/intent_eval.py --samples 200              # Full eval
    python scripts/intent_eval.py --model qwen2.5-1.5b       # Different model
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test set paths
CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
ORIGINAL_TEST_SET = Path("results/test_set/test_data.jsonl")
TEST_SET_FILE = CLEAN_TEST_SET if CLEAN_TEST_SET.exists() else ORIGINAL_TEST_SET
RESULTS_DIR = Path("results/intent_eval")

# Available models
MODELS = {
    "lfm2.5-1.2b": "lfm2.5-1.2b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "llama-3.2-3b": "llama-3.2-3b",
}

# Intent categories with anchor phrases
INTENT_ANCHORS = {
    "accept": ["yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely", "yep", "let's do it", "i'm down", "for sure"],
    "decline": ["no", "nah", "can't make it", "sorry can't", "busy", "not going", "pass", "maybe later", "i can't", "won't be able to", "not gonna work"],
    "question": ["what time", "when is it", "where at", "why", "how", "what's the plan", "which one", "who's coming"],
    "reaction": ["lol", "haha", "nice", "wow", "crazy", "damn", "omg", "that's funny", "hilarious", "no way", "wild"],
    "info": ["i'll be there at", "at 5pm", "tomorrow morning", "the address is", "it's at", "i'm at", "heading there now", "running late"],
    "acknowledge": ["ok got it", "thanks", "understood", "cool", "bet", "alright", "sounds good", "noted"],
}


@dataclass
class IntentResult:
    """Result for a single evaluation sample."""
    contact: str
    conversation: str
    last_message: str
    gold: str
    generated: str
    prompt_sent: str
    model: str
    # Intent analysis
    gold_intent: str
    gold_confidence: float
    gen_intent: str
    gen_confidence: float
    intent_match: bool
    # Additional metrics
    is_question_match: bool  # Both ask questions or both don't
    length_bucket: str
    is_group: bool


class IntentClassifier:
    """Classify message intent using embedding similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._compute_anchors()

    def _compute_anchors(self):
        """Pre-compute anchor embeddings for each intent."""
        self.anchor_embeddings = {}
        for intent, phrases in INTENT_ANCHORS.items():
            # Store all individual embeddings (not averaged)
            embs = self.model.encode(phrases, normalize_embeddings=True)
            self.anchor_embeddings[intent] = embs

    def classify(self, message: str) -> tuple[str, float]:
        """Classify message intent using MAX similarity to any anchor."""
        if not message.strip():
            return "unknown", 0.0

        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]

        best_intent = "unknown"
        best_score = -1

        for intent, anchor_embs in self.anchor_embeddings.items():
            # Take MAX similarity to any anchor phrase in this category
            similarities = np.dot(anchor_embs, msg_emb)
            max_sim = float(np.max(similarities))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        return best_intent, best_score

    def intents_match(self, gold: str, generated: str) -> tuple[bool, dict]:
        """Check if gold and generated have matching intents."""
        gold_intent, gold_conf = self.classify(gold)
        gen_intent, gen_conf = self.classify(generated)

        # Direct match
        match = gold_intent == gen_intent

        # Also check question alignment (both have ? or neither)
        gold_is_q = "?" in gold
        gen_is_q = "?" in generated
        question_match = gold_is_q == gen_is_q

        return match, {
            "gold_intent": gold_intent,
            "gold_confidence": gold_conf,
            "gen_intent": gen_intent,
            "gen_confidence": gen_conf,
            "question_match": question_match,
        }


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def build_reply_prompt(conversation: str, last_msg: str) -> str:
    """Build prompt with few-shot examples of casual texting style."""
    return f"""[Reply in casual texting style. Keep it short, 2-8 words max.]

them: wanna hang tonight?
me: yeah down

them: that was hilarious
me: lol ikr

them: when you coming?
me: like 10 min

them: did you see that??
me: no what happened

{conversation}
me:"""


def clean_response(text: str) -> str:
    """Clean model response."""
    if not text:
        return ""

    # Remove common prefixes
    text = text.strip()
    for prefix in ["REPLY:", "Reply:", "me:", "Me:", "Response:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Take first line only
    text = text.split("\n")[0].strip()

    # Remove quotes
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1]

    return text.strip()


def run_intent_eval(
    model_name: str = "lfm2.5-1.2b",
    limit: int = 50,
    verbose: bool = False,
) -> list[IntentResult]:
    """Run intent-based evaluation."""
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"INTENT-BASED EVALUATION")
    print(f"  Model: {model_name}")
    print(f"  Samples: {limit}")
    print(f"{'='*60}")

    # Load test set
    print("\nLoading test set...")
    samples = load_test_set(limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Load intent classifier
    print("Loading intent classifier...")
    classifier = IntentClassifier()

    # Load LLM
    print(f"Loading LLM ({model_name})...")
    loader = ModelLoader(model_name)
    loader.preload()

    # Run evaluation
    results = []
    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>", "Them:", "Me:"]

    print(f"\nEvaluating...")
    start_time = time.time()

    intent_matches = 0
    question_matches = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] Intent match: {intent_matches}/{i+1} ({intent_matches/(i+1)*100:.0f}%)")

        contact = sample.get("contact", "Unknown")
        gold = sample["gold_response"]
        conversation = sample.get("conversation", "")
        last_msg = sample.get("last_message", "")

        # Build prompt
        prompt = build_reply_prompt(conversation, last_msg)

        # Generate
        result = loader.generate(
            prompt=prompt,
            max_tokens=40,
            temperature=0.3,
            stop=stop_seqs,
        )
        generated = clean_response(result.text)

        # Classify intents
        match, intent_info = classifier.intents_match(gold, generated)

        if match:
            intent_matches += 1
        if intent_info["question_match"]:
            question_matches += 1

        # Determine length bucket
        gold_len = len(gold)
        if gold_len < 15:
            length_bucket = "short"
        elif gold_len < 40:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        eval_result = IntentResult(
            contact=contact,
            conversation=conversation[:200],
            last_message=last_msg,
            gold=gold,
            generated=generated,
            prompt_sent=prompt,
            model=model_name,
            gold_intent=intent_info["gold_intent"],
            gold_confidence=intent_info["gold_confidence"],
            gen_intent=intent_info["gen_intent"],
            gen_confidence=intent_info["gen_confidence"],
            intent_match=match,
            is_question_match=intent_info["question_match"],
            length_bucket=length_bucket,
            is_group=sample.get("is_group", False),
        )
        results.append(eval_result)

        if verbose:
            marker = "✓" if match else "✗"
            print(f"    {marker} [{intent_info['gold_intent']}] \"{gold[:30]}\" → [{intent_info['gen_intent']}] \"{generated[:30]}\"")

    elapsed = time.time() - start_time

    # Print results
    print_results(results, model_name, elapsed)

    # Save results
    save_results(results, model_name, limit)

    return results


def print_results(results: list[IntentResult], model: str, elapsed: float):
    """Print evaluation results."""
    n = len(results)

    intent_matches = sum(1 for r in results if r.intent_match)
    question_matches = sum(1 for r in results if r.is_question_match)

    # By intent category
    by_gold_intent = {}
    for r in results:
        intent = r.gold_intent
        if intent not in by_gold_intent:
            by_gold_intent[intent] = {"total": 0, "match": 0}
        by_gold_intent[intent]["total"] += 1
        if r.intent_match:
            by_gold_intent[intent]["match"] += 1

    # By length bucket
    by_length = {}
    for bucket in ["short", "medium", "long"]:
        bucket_results = [r for r in results if r.length_bucket == bucket]
        if bucket_results:
            by_length[bucket] = {
                "total": len(bucket_results),
                "match": sum(1 for r in bucket_results if r.intent_match),
            }

    print(f"\n{'='*60}")
    print(f"RESULTS: {model}")
    print(f"{'='*60}")
    print(f"Samples: {n} | Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    print(f"\n--- OVERALL ---")
    print(f"  Intent match:    {intent_matches:3d}/{n} ({intent_matches/n*100:5.1f}%)")
    print(f"  Question match:  {question_matches:3d}/{n} ({question_matches/n*100:5.1f}%)")

    print(f"\n--- BY GOLD INTENT ---")
    for intent in sorted(by_gold_intent.keys()):
        data = by_gold_intent[intent]
        rate = data["match"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"  {intent:12}: {data['match']:2d}/{data['total']:2d} ({rate:4.0f}%)")

    print(f"\n--- BY LENGTH ---")
    for bucket in ["short", "medium", "long"]:
        if bucket in by_length:
            data = by_length[bucket]
            rate = data["match"] / data["total"] * 100
            print(f"  {bucket:8}: {data['match']:2d}/{data['total']:2d} ({rate:4.0f}%)")

    # Show some examples
    print(f"\n--- SAMPLE OUTPUTS ---")
    matches = [r for r in results if r.intent_match][:3]
    mismatches = [r for r in results if not r.intent_match][:3]

    print("  Matches:")
    for r in matches:
        print(f"    [{r.gold_intent}] \"{r.gold[:35]}\" → \"{r.generated[:35]}\"")

    print("  Mismatches:")
    for r in mismatches:
        print(f"    [{r.gold_intent}→{r.gen_intent}] \"{r.gold[:30]}\" → \"{r.generated[:30]}\"")


def save_results(results: list[IntentResult], model: str, limit: int):
    """Save results to JSON."""
    n = len(results)
    intent_matches = sum(1 for r in results if r.intent_match)

    results_file = RESULTS_DIR / f"intent_{model}_{limit}.json"

    with open(results_file, "w") as f:
        json.dump({
            "model": model,
            "n_samples": n,
            "metrics": {
                "intent_match_rate": intent_matches / n,
                "question_match_rate": sum(1 for r in results if r.is_question_match) / n,
            },
            "samples": [asdict(r) for r in results],
        }, f, indent=2)

    print(f"\nSaved: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Intent-based evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--model", type=str, default="lfm2.5-1.2b",
                        choices=list(MODELS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_intent_eval(
        model_name=args.model,
        limit=args.samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
