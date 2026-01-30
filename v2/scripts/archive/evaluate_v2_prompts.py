#!/usr/bin/env python3
"""Evaluate V1 vs V2 smart prompts with lfm2-2.6b-exp.

Compares:
- V1 (original smart_prompter)
- V2 (improved retriever + style tuning)
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
RESULTS_DIR = Path("results/evaluation")


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def clean_response(text: str) -> str:
    """Clean model response."""
    if not text:
        return ""
    text = re.sub(r'^me:\s*', '', text, flags=re.IGNORECASE)
    text = text.strip()
    if '\n' in text:
        text = text.split('\n')[0].strip()
    return text


def compute_style_score(response: str, gold: str) -> float:
    """Compute style similarity score."""
    if not response or not gold:
        return 0.0

    score = 0.0

    # Punctuation match
    resp_has_punct = response.rstrip()[-1:] in ".!?" if response else False
    gold_has_punct = gold.rstrip()[-1:] in ".!?" if gold else False
    if resp_has_punct == gold_has_punct:
        score += 0.3

    # Length similarity
    len_ratio = len(response) / max(len(gold), 1)
    if 0.5 <= len_ratio <= 2.0:
        score += 0.3
    elif 0.3 <= len_ratio <= 3.0:
        score += 0.15

    # Abbreviation usage
    abbrevs = [r"\bu\b", r"\bur\b", r"\brn\b", r"\btmrw\b", r"\bidk\b", r"\byea\b"]
    gold_has_abbrev = any(re.search(a, gold.lower()) for a in abbrevs)
    resp_has_abbrev = any(re.search(a, response.lower()) for a in abbrevs)
    if gold_has_abbrev == resp_has_abbrev:
        score += 0.2

    # Casual markers
    gold_has_lol = "lol" in gold.lower() or "haha" in gold.lower()
    resp_has_lol = "lol" in response.lower() or "haha" in response.lower()
    if gold_has_lol == resp_has_lol:
        score += 0.2

    return score


def evaluate(limit: int = 50):
    """Run V1 vs V2 evaluation."""
    from sentence_transformers import SentenceTransformer
    from core.models.loader import ModelLoader
    from core.generation.smart_prompter import build_smart_prompt
    from core.generation.smart_prompter_v2 import build_smart_prompt_v2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    samples = load_test_set(limit=limit)
    print(f"Loaded {len(samples)} samples")

    print("\nLoading similarity model...")
    sim_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("\nLoading LFM2-2.6B-Exp...")
    loader = ModelLoader("lfm2-2.6b-exp")
    loader.preload()

    results = []
    v1_scores = []
    v2_scores = []

    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>"]

    print(f"\nEvaluating {len(samples)} samples...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        contact = sample.get("contact", "Unknown")
        gold = sample["gold_response"]
        prompt_text = sample.get("prompt", "")

        # Parse conversation
        lines = prompt_text.split("\n")
        conv_lines = [l for l in lines if l.startswith("me:") or l.startswith("them:")]

        conv_msgs = []
        for line in conv_lines:
            line = line.strip()
            if line.startswith("me:"):
                conv_msgs.append({"text": line[3:].strip(), "is_from_me": True})
            elif line.startswith("them:"):
                conv_msgs.append({"text": line[5:].strip(), "is_from_me": False})

        if not conv_msgs:
            continue

        # Build prompts
        v1_result = build_smart_prompt(contact, conv_msgs, n_examples=2)
        v2_result = build_smart_prompt_v2(contact, conv_msgs, n_examples=3, min_similarity=0.5)

        # Generate
        v1_gen = loader.generate(v1_result.prompt, max_tokens=50, temperature=0.3, stop=stop_seqs)
        v2_gen = loader.generate(v2_result.prompt, max_tokens=50, temperature=0.3, stop=stop_seqs)

        v1_resp = clean_response(v1_gen.text)
        v2_resp = clean_response(v2_gen.text)

        # Skip empty
        if not v1_resp and not v2_resp:
            continue

        # Compute metrics
        all_texts = [gold, v1_resp or "empty", v2_resp or "empty"]
        embeddings = sim_model.encode(all_texts, normalize_embeddings=True)

        v1_sim = float(np.dot(embeddings[0], embeddings[1])) if v1_resp else 0
        v2_sim = float(np.dot(embeddings[0], embeddings[2])) if v2_resp else 0

        v1_style = compute_style_score(v1_resp, gold)
        v2_style = compute_style_score(v2_resp, gold)

        # Combined score
        v1_score = v1_sim * 0.6 + v1_style * 0.4
        v2_score = v2_sim * 0.6 + v2_style * 0.4

        v1_scores.append(v1_score)
        v2_scores.append(v2_score)

        results.append({
            "contact": contact,
            "gold": gold,
            "v1_resp": v1_resp,
            "v2_resp": v2_resp,
            "v1_sim": v1_sim,
            "v2_sim": v2_sim,
            "v1_style": v1_style,
            "v2_style": v2_style,
            "message_type": v2_result.message_type,
            "v1_examples": len(v1_result.few_shot_examples),
            "v2_examples": len(v2_result.few_shot_examples),
        })

    elapsed = time.time() - start_time

    # Calculate stats
    v1_scores = np.array(v1_scores)
    v2_scores = np.array(v2_scores)

    v2_wins = int(np.sum(v2_scores > v1_scores + 0.02))
    v1_wins = int(np.sum(v1_scores > v2_scores + 0.02))
    ties = len(results) - v2_wins - v1_wins

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: V1 (original) vs V2 (improved)")
    print("=" * 70)

    print(f"\nSamples: {len(results)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nV2 wins: {v2_wins} ({v2_wins/len(results)*100:.1f}%)")
    print(f"V1 wins: {v1_wins} ({v1_wins/len(results)*100:.1f}%)")
    print(f"Ties: {ties} ({ties/len(results)*100:.1f}%)")

    print(f"\nAverage semantic similarity:")
    print(f"  V1: {np.mean([r['v1_sim'] for r in results]):.3f}")
    print(f"  V2: {np.mean([r['v2_sim'] for r in results]):.3f}")

    print(f"\nAverage style score:")
    print(f"  V1: {np.mean([r['v1_style'] for r in results]):.3f}")
    print(f"  V2: {np.mean([r['v2_style'] for r in results]):.3f}")

    # Print examples by message type
    print("\n" + "=" * 70)
    print("EXAMPLES BY MESSAGE TYPE")
    print("=" * 70)

    for msg_type in ["question", "statement", "reaction", "greeting"]:
        type_results = [r for r in results if r["message_type"] == msg_type]
        if not type_results:
            continue

        print(f"\n--- {msg_type.upper()} ({len(type_results)} samples) ---")
        v2_type_wins = sum(1 for r in type_results if r["v2_sim"] > r["v1_sim"] + 0.02)
        print(f"V2 win rate: {v2_type_wins/len(type_results)*100:.1f}%")

        # Show 2 examples
        for r in type_results[:2]:
            print(f"\nGold: \"{r['gold']}\"")
            print(f"  V1: \"{r['v1_resp']}\" (sim={r['v1_sim']:.2f})")
            print(f"  V2: \"{r['v2_resp']}\" (sim={r['v2_sim']:.2f})")

    # Save results
    results_file = RESULTS_DIR / f"v1_vs_v2_eval_{limit}.json"
    with open(results_file, "w") as f:
        json.dump({
            "n_samples": len(results),
            "v2_wins": v2_wins,
            "v1_wins": v1_wins,
            "ties": ties,
            "v2_win_rate": v2_wins / len(results),
            "avg_v1_sim": float(np.mean([r["v1_sim"] for r in results])),
            "avg_v2_sim": float(np.mean([r["v2_sim"] for r in results])),
            "samples": results[:30],
        }, f, indent=2)
    print(f"\n\nSaved: {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    evaluate(limit=args.limit)
