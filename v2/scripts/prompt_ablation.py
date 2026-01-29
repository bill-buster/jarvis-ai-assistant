#!/usr/bin/env python3
"""A/B test different prompt strategies for reply generation.

Tests multiple prompts on the same samples, evaluates blindly,
then reveals which prompt performed best.

Usage:
    python scripts/prompt_ablation.py --samples 50
    python scripts/prompt_ablation.py --samples 100 --verbose
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
RESULTS_DIR = Path("results/prompt_ablation")

# =============================================================================
# PROMPT STRATEGIES TO TEST
# =============================================================================

PROMPTS = {
    # Roleplay framing - tell it to BE the person
    "A_roleplay_me": lambda conv, last: f"""You are roleplaying as me in a text conversation. Reply exactly how I would - casual, brief, lowercase. DO NOT be an assistant. DO NOT explain. Just text back like a real person.

{conv}
me:""",

    # Heavy few-shot (8 examples)
    "B_heavy_fewshot": lambda conv, last: f"""them: wanna hang?
me: yeah down

them: lol nice
me: ikr

them: you coming?
me: yea omw

them: that's wild
me: fr tho

them: thanks!
me: np

them: what time
me: like 7

them: sorry can't make it
me: all good

them: did you see that
me: no what happened

{conv}
me:""",

    # Anti-assistant prompt
    "C_anti_assistant": lambda conv, last: f"""IMPORTANT: You are NOT an assistant. Do NOT say "I'm sorry" or "It sounds like" or "I understand".

You are a college student texting your friend. Reply in 2-8 words, casual, lowercase.

{conv}
me:""",

    # Completion framing (no instruction, just pattern)
    "D_pure_completion": lambda conv, last: f"""them: yo you up?
me: yeah whats up

them: nm just bored
me: same lol

them: wanna play games
me: down

{conv}
me:""",

    # Explicit persona
    "E_persona": lambda conv, last: f"""[You are Jwalin, a 22 year old. You text like: short messages, lowercase, use "lol" "fr" "bet" "nah". Never formal. Never apologize. Never explain.]

{conv}
me:""",
}

# =============================================================================
# INTENT CLASSIFIER (same as intent_eval.py)
# =============================================================================

INTENT_ANCHORS = {
    "accept": ["yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely", "yep", "let's do it", "i'm down", "for sure", "bet"],
    "decline": ["no", "nah", "can't make it", "sorry can't", "busy", "not going", "pass", "maybe later", "i can't", "won't be able to"],
    "question": ["what time", "when is it", "where at", "why", "how", "what's the plan", "which one", "who's coming", "what happened"],
    "reaction": ["lol", "haha", "nice", "wow", "crazy", "damn", "omg", "that's funny", "hilarious", "no way", "wild", "bruh"],
    "info": ["i'll be there at", "at 5pm", "tomorrow morning", "the address is", "it's at", "i'm at", "heading there now", "running late"],
    "acknowledge": ["ok got it", "thanks", "understood", "cool", "bet", "alright", "noted", "aight", "k"],
}


class IntentClassifier:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchor_embeddings = {}
        for intent, phrases in INTENT_ANCHORS.items():
            self.anchor_embeddings[intent] = self.model.encode(phrases, normalize_embeddings=True)

    def classify(self, message: str) -> tuple[str, float]:
        if not message.strip():
            return "unknown", 0.0
        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]
        best_intent, best_score = "unknown", -1
        for intent, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent
        return best_intent, best_score


def load_test_set(limit: int) -> list[dict]:
    samples = []
    with open(CLEAN_TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for prefix in ["me:", "Me:", "REPLY:", "Reply:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.split("\n")[0].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1]
    return text.strip()


@dataclass
class PromptResult:
    prompt_name: str
    sample_id: int
    gold: str
    generated: str
    gold_intent: str
    gen_intent: str
    intent_match: bool
    gen_length: int


def run_ablation(
    model_name: str = "lfm2.5-1.2b",
    n_samples: int = 50,
    verbose: bool = False,
):
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PROMPT ABLATION TEST")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"{'='*70}")

    # Load components
    print("\nLoading test set...")
    samples = load_test_set(n_samples)

    print("Loading intent classifier...")
    classifier = IntentClassifier()

    print(f"Loading LLM ({model_name})...")
    loader = ModelLoader(model_name)
    loader.preload()

    # Run all prompts on all samples
    all_results = {name: [] for name in PROMPTS}
    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>"]

    print(f"\nRunning {len(PROMPTS)} prompts × {n_samples} samples...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}")

        conv = sample.get("conversation", "")
        last_msg = sample.get("last_message", "")
        gold = sample["gold_response"]
        gold_intent, _ = classifier.classify(gold)

        # Test each prompt
        for prompt_name, prompt_fn in PROMPTS.items():
            prompt = prompt_fn(conv, last_msg)

            result = loader.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.3,
                stop=stop_seqs,
            )
            generated = clean_response(result.text)
            gen_intent, _ = classifier.classify(generated)

            pr = PromptResult(
                prompt_name=prompt_name,
                sample_id=i,
                gold=gold,
                generated=generated,
                gold_intent=gold_intent,
                gen_intent=gen_intent,
                intent_match=(gold_intent == gen_intent),
                gen_length=len(generated),
            )
            all_results[prompt_name].append(pr)

            if verbose and i < 3:  # Show first 3 samples
                marker = "✓" if pr.intent_match else "✗"
                print(f"    [{prompt_name}] {marker} \"{generated[:40]}\"")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # ==========================================================================
    # BLIND EVALUATION - compute metrics without knowing which prompt is which
    # ==========================================================================

    print(f"\n{'='*70}")
    print("RESULTS (sorted by intent match rate)")
    print(f"{'='*70}\n")

    # Compute metrics for each prompt
    metrics = {}
    for name, results in all_results.items():
        n = len(results)
        intent_matches = sum(1 for r in results if r.intent_match)
        avg_length = np.mean([r.gen_length for r in results])

        # Length appropriateness (gold is usually 5-30 chars)
        good_length = sum(1 for r in results if 3 <= r.gen_length <= 50)

        metrics[name] = {
            "intent_match_rate": intent_matches / n,
            "intent_matches": intent_matches,
            "avg_length": avg_length,
            "good_length_rate": good_length / n,
            "results": results,
        }

    # Sort by intent match rate
    sorted_prompts = sorted(metrics.items(), key=lambda x: x[1]["intent_match_rate"], reverse=True)

    print(f"{'Prompt':<25} {'Intent Match':<15} {'Avg Len':<10} {'Good Len%':<10}")
    print("-" * 60)

    for name, m in sorted_prompts:
        intent_pct = f"{m['intent_matches']}/{n_samples} ({m['intent_match_rate']*100:.0f}%)"
        print(f"{name:<25} {intent_pct:<15} {m['avg_length']:<10.1f} {m['good_length_rate']*100:<10.0f}%")

    # Show sample outputs from best prompt
    print(f"\n{'='*70}")
    print(f"SAMPLE OUTPUTS FROM BEST PROMPT: {sorted_prompts[0][0]}")
    print(f"{'='*70}\n")

    best_results = sorted_prompts[0][1]["results"]

    print("Matches:")
    matches = [r for r in best_results if r.intent_match][:5]
    for r in matches:
        print(f"  [{r.gold_intent}] \"{r.gold[:35]}\" → \"{r.generated[:35]}\"")

    print("\nMismatches:")
    mismatches = [r for r in best_results if not r.intent_match][:5]
    for r in mismatches:
        print(f"  [{r.gold_intent}→{r.gen_intent}] \"{r.gold[:30]}\" → \"{r.generated[:30]}\"")

    # Show what each prompt actually looks like
    print(f"\n{'='*70}")
    print("PROMPT TEMPLATES")
    print(f"{'='*70}\n")

    sample_conv = "them: you coming tonight?\nthem: we're starting at 8"
    for name, prompt_fn in PROMPTS.items():
        print(f"--- {name} ---")
        print(prompt_fn(sample_conv, "we're starting at 8")[:200])
        print()

    # Save results
    save_path = RESULTS_DIR / f"ablation_{model_name}_{n_samples}.json"
    save_data = {
        "model": model_name,
        "n_samples": n_samples,
        "elapsed_seconds": elapsed,
        "results": {
            name: {
                "intent_match_rate": m["intent_match_rate"],
                "avg_length": m["avg_length"],
                "good_length_rate": m["good_length_rate"],
            }
            for name, m in metrics.items()
        },
        "ranking": [name for name, _ in sorted_prompts],
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="lfm2.5-1.2b")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_ablation(model_name=args.model, n_samples=args.samples, verbose=args.verbose)


if __name__ == "__main__":
    main()
