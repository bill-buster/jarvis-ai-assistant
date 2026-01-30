#!/usr/bin/env python3
"""Experiment 1: Structured ASK/REPLY generation.

Uses the two-step approach that worked in capability tests:
1. Classify what intent is needed
2. Generate with that intent constraint

Usage:
    python scripts/exp1_structured_generation.py --samples 50
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
RESULTS_DIR = Path("results/experiments")

# Intent classification prompt - simple, single letter output
INTENT_CLASSIFY_PROMPT = """Classify this message. Output ONLY the letter.

A = respond to invitation (yes/no)
B = ask a question back
C = react (lol, wow, nice)
D = give info (time, place)
E = acknowledge (ok, got it)

them: {last_message}
Answer:"""

# Generation prompts for each intent type
INTENT_PROMPTS = {
    "A": """Reply to accept or decline. Casual texting style, 2-6 words.

them: wanna hang? → yeah down
them: can you help tomorrow? → sure what time
them: coming to the party? → nah can't make it

them: {last_message}
me:""",

    "B": """Ask a follow-up question. Casual texting style, 2-6 words.

them: let's meet up → when works for u
them: i got news → what happened
them: party this weekend → where at

them: {last_message}
me:""",

    "C": """React to their message. Casual texting style, 1-4 words.

them: i just got promoted → ayy congrats
them: you won't believe what happened → lol what
them: that was insane → fr tho

them: {last_message}
me:""",

    "D": """Provide information. Casual texting style, 2-8 words.

them: where are you → omw like 5 min
them: what's the address → 123 main st
them: when's the thing → starts at 7

them: {last_message}
me:""",

    "E": """Acknowledge their message. Casual texting style, 1-3 words.

them: i'll be there at 5 → bet
them: thanks for helping → np
them: just letting you know → got it

them: {last_message}
me:""",
}

INTENT_NAMES = {
    "A": "accept/decline",
    "B": "question",
    "C": "reaction",
    "D": "info",
    "E": "acknowledge",
}

# Map our intents to evaluation intents
INTENT_MAP = {
    "A": ["accept", "decline"],
    "B": ["question"],
    "C": ["reaction"],
    "D": ["info"],
    "E": ["acknowledge"],
}


@dataclass
class StructuredResult:
    sample_id: int
    gold: str
    gold_intent: str
    classified_intent: str
    classified_letter: str
    generated: str
    gen_intent: str
    intent_match: bool
    classification_correct: bool


def load_test_set(limit: int) -> list[dict]:
    samples = []
    with open(CLEAN_TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


class IntentClassifier:
    """Same as intent_eval.py"""
    INTENT_ANCHORS = {
        "accept": ["yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely", "yep"],
        "decline": ["no", "nah", "can't make it", "sorry can't", "busy", "not going", "pass"],
        "question": ["what time", "when is it", "where at", "why", "how", "what's the plan"],
        "reaction": ["lol", "haha", "nice", "wow", "crazy", "damn", "omg", "hilarious"],
        "info": ["i'll be there at", "at 5pm", "tomorrow morning", "it's at", "heading there"],
        "acknowledge": ["ok got it", "thanks", "understood", "cool", "bet", "alright", "aight"],
    }

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchor_embeddings = {}
        for intent, phrases in self.INTENT_ANCHORS.items():
            self.anchor_embeddings[intent] = self.model.encode(phrases, normalize_embeddings=True)

    def classify(self, message: str) -> str:
        if not message.strip():
            return "unknown"
        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]
        best_intent, best_score = "unknown", -1
        for intent, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent
        return best_intent


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for prefix in ["me:", "Me:", "REPLY:", "Reply:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.split("\n")[0].strip()
    return text


def run_experiment(model_name: str, n_samples: int, verbose: bool):
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: Structured ASK/REPLY Generation")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples}")
    print(f"{'='*70}")

    # Load components
    print("\nLoading test set...")
    samples = load_test_set(n_samples)

    print("Loading intent classifier...")
    classifier = IntentClassifier()

    print(f"Loading LLM ({model_name})...")
    loader = ModelLoader(model_name)
    loader.preload()

    # Run experiment
    results = []
    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>"]

    print(f"\nRunning structured generation...")
    start_time = time.time()

    intent_matches = 0
    classification_correct = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}] Intent match: {intent_matches}/{i+1} ({intent_matches/(i+1)*100:.0f}%)")

        last_msg = sample.get("last_message", "")
        gold = sample["gold_response"]
        gold_intent = classifier.classify(gold)

        # Step 1: Classify what intent is needed
        classify_prompt = INTENT_CLASSIFY_PROMPT.format(last_message=last_msg)
        classify_result = loader.generate(
            prompt=classify_prompt,
            max_tokens=2,
            temperature=0.1,
            stop=["\n", " ", ")", "."],
        )
        letter = classify_result.text.strip().upper()[:1]
        if letter not in "ABCDE":
            letter = "C"  # Default to reaction

        # Check if classification matches gold intent
        expected_intents = INTENT_MAP.get(letter, [])
        class_correct = gold_intent in expected_intents
        if class_correct:
            classification_correct += 1

        # Step 2: Generate with intent-specific prompt
        gen_prompt = INTENT_PROMPTS[letter].format(last_message=last_msg)
        gen_result = loader.generate(
            prompt=gen_prompt,
            max_tokens=20,
            temperature=0.3,
            stop=stop_seqs,
        )
        generated = clean_response(gen_result.text)
        gen_intent = classifier.classify(generated)

        # Check intent match
        match = (gold_intent == gen_intent)
        if match:
            intent_matches += 1

        result = StructuredResult(
            sample_id=i,
            gold=gold,
            gold_intent=gold_intent,
            classified_intent=INTENT_NAMES.get(letter, "unknown"),
            classified_letter=letter,
            generated=generated,
            gen_intent=gen_intent,
            intent_match=match,
            classification_correct=class_correct,
        )
        results.append(result)

        if verbose:
            marker = "✓" if match else "✗"
            print(f"    {marker} [{letter}→{gold_intent}] \"{gold[:25]}\" → \"{generated[:25]}\"")

    elapsed = time.time() - start_time

    # Print results
    n = len(results)
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    print(f"\n--- METRICS ---")
    print(f"  Intent Match:         {intent_matches}/{n} ({intent_matches/n*100:.1f}%)")
    print(f"  Classification Acc:   {classification_correct}/{n} ({classification_correct/n*100:.1f}%)")

    # By classified intent
    print(f"\n--- BY CLASSIFIED INTENT ---")
    for letter in "ABCDE":
        letter_results = [r for r in results if r.classified_letter == letter]
        if letter_results:
            matches = sum(1 for r in letter_results if r.intent_match)
            print(f"  {letter} ({INTENT_NAMES[letter]:12}): {matches}/{len(letter_results)} ({matches/len(letter_results)*100:.0f}%)")

    # Sample outputs
    print(f"\n--- SAMPLE MATCHES ---")
    for r in [r for r in results if r.intent_match][:5]:
        print(f"  [{r.classified_letter}] \"{r.gold[:30]}\" → \"{r.generated[:30]}\"")

    print(f"\n--- SAMPLE MISMATCHES ---")
    for r in [r for r in results if not r.intent_match][:5]:
        print(f"  [{r.classified_letter}:{r.gold_intent}→{r.gen_intent}] \"{r.gold[:25]}\" → \"{r.generated[:25]}\"")

    # Save
    save_path = RESULTS_DIR / f"exp1_structured_{model_name}_{n_samples}.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": model_name,
            "n_samples": n,
            "elapsed": elapsed,
            "intent_match_rate": intent_matches / n,
            "classification_accuracy": classification_correct / n,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nSaved: {save_path}")

    return intent_matches / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="lfm2.5-1.2b")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_experiment(args.model, args.samples, args.verbose)


if __name__ == "__main__":
    main()
