#!/usr/bin/env python3
"""Evaluate models with smart prompts (style + few-shot) vs baseline.

Compares:
1. Baseline prompts (generic)
2. Smart prompts (style-aware + few-shot)

Metrics:
- Semantic similarity to your actual reply
- Length similarity
- Style match (punctuation, abbreviations, etc.)

Usage:
    python scripts/evaluate_smart_prompts.py
    python scripts/evaluate_smart_prompts.py --limit 50
    python scripts/evaluate_smart_prompts.py --model qwen3-0.6b
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
RESULTS_DIR = Path("results/evaluation")


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    sample_id: int
    contact: str
    gold_response: str

    # Baseline
    baseline_response: str
    baseline_semantic_sim: float
    baseline_length_diff: float
    baseline_style_score: float

    # Smart prompt
    smart_response: str
    smart_semantic_sim: float
    smart_length_diff: float
    smart_style_score: float


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def compute_style_score(response: str, gold: str) -> float:
    """Compute style similarity score.

    Checks:
    - Punctuation match (both have/don't have ending punct)
    - Length similarity
    - Abbreviation usage match
    """
    score = 0.0

    # Punctuation match
    resp_has_punct = response.rstrip()[-1:] in ".!?" if response else False
    gold_has_punct = gold.rstrip()[-1:] in ".!?" if gold else False
    if resp_has_punct == gold_has_punct:
        score += 0.3

    # Length similarity (within 50% is good)
    if gold:
        len_ratio = len(response) / max(len(gold), 1)
        if 0.5 <= len_ratio <= 2.0:
            score += 0.3
        elif 0.3 <= len_ratio <= 3.0:
            score += 0.15

    # Abbreviation usage
    abbrevs = ["u", "ur", "rn", "tmrw", "idk", "nvm", "yea", "ok"]
    gold_has_abbrev = any(re.search(rf'\b{a}\b', gold.lower()) for a in abbrevs)
    resp_has_abbrev = any(re.search(rf'\b{a}\b', response.lower()) for a in abbrevs)
    if gold_has_abbrev == resp_has_abbrev:
        score += 0.2

    # Casual markers (lol, haha)
    gold_has_lol = "lol" in gold.lower() or "haha" in gold.lower()
    resp_has_lol = "lol" in response.lower() or "haha" in response.lower()
    if gold_has_lol == resp_has_lol:
        score += 0.2

    return score


def run_evaluation(model_id: str = "qwen3-0.6b", limit: int = 100):
    """Run full evaluation."""
    from core.models.loader import ModelLoader
    from core.generation.smart_prompter import build_smart_prompt
    from sentence_transformers import SentenceTransformer

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set
    print(f"Loading test set (limit={limit})...")
    samples = load_test_set(limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Load model
    print(f"\nLoading model: {model_id}")
    loader = ModelLoader(model_id)
    loader.preload()

    # Load embedding model for semantic similarity
    print("Loading embedding model for similarity...")
    sim_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    results = []
    baseline_wins = 0
    smart_wins = 0
    ties = 0

    print(f"\nEvaluating {len(samples)} samples...")
    print("-" * 60)

    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{len(samples)}] {rate:.0f}/min")

        contact = sample.get("contact", "Unknown")
        gold = sample["gold_response"]
        prompt_text = sample.get("prompt", "")

        # Parse conversation from prompt (remove style hint line)
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

        # Get conversation text (without style hint)
        conversation_text = "\n".join(conv_lines)

        # Baseline prompt (generic)
        baseline_prompt = f"[brief, casual]\n\n{conversation_text}\nme:"

        # Smart prompt (style + few-shot)
        smart_result = build_smart_prompt(contact, conv_msgs, n_examples=2)
        smart_prompt = smart_result.prompt

        # Generate responses
        stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>"]

        baseline_gen = loader.generate(baseline_prompt, max_tokens=40, temperature=0.3, stop=stop_seqs)
        smart_gen = loader.generate(smart_prompt, max_tokens=40, temperature=0.3, stop=stop_seqs)

        baseline_resp = baseline_gen.text.strip()
        smart_resp = smart_gen.text.strip()

        # Compute metrics
        # Semantic similarity
        embeddings = sim_model.encode([gold, baseline_resp, smart_resp], normalize_embeddings=True)
        baseline_sim = float(np.dot(embeddings[0], embeddings[1]))
        smart_sim = float(np.dot(embeddings[0], embeddings[2]))

        # Length difference (normalized)
        baseline_len_diff = abs(len(baseline_resp) - len(gold)) / max(len(gold), 1)
        smart_len_diff = abs(len(smart_resp) - len(gold)) / max(len(gold), 1)

        # Style score
        baseline_style = compute_style_score(baseline_resp, gold)
        smart_style = compute_style_score(smart_resp, gold)

        # Overall score (weighted)
        baseline_score = baseline_sim * 0.5 + (1 - baseline_len_diff) * 0.2 + baseline_style * 0.3
        smart_score = smart_sim * 0.5 + (1 - smart_len_diff) * 0.2 + smart_style * 0.3

        if smart_score > baseline_score + 0.02:
            smart_wins += 1
        elif baseline_score > smart_score + 0.02:
            baseline_wins += 1
        else:
            ties += 1

        results.append(EvalResult(
            sample_id=sample["id"],
            contact=contact,
            gold_response=gold,
            baseline_response=baseline_resp,
            baseline_semantic_sim=baseline_sim,
            baseline_length_diff=baseline_len_diff,
            baseline_style_score=baseline_style,
            smart_response=smart_resp,
            smart_semantic_sim=smart_sim,
            smart_length_diff=smart_len_diff,
            smart_style_score=smart_style,
        ))

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {len(results)}")
    print(f"Time: {elapsed:.0f}s ({len(results)/elapsed*60:.0f}/min)")

    print("\n" + "-" * 40)
    print("WIN RATES")
    print("-" * 40)
    total = len(results)
    print(f"  Smart prompts better: {smart_wins} ({smart_wins/total*100:.1f}%)")
    print(f"  Baseline better:      {baseline_wins} ({baseline_wins/total*100:.1f}%)")
    print(f"  Ties:                 {ties} ({ties/total*100:.1f}%)")

    print("\n" + "-" * 40)
    print("AVERAGE METRICS")
    print("-" * 40)

    avg_baseline_sim = np.mean([r.baseline_semantic_sim for r in results])
    avg_smart_sim = np.mean([r.smart_semantic_sim for r in results])
    print(f"Semantic similarity:")
    print(f"  Baseline: {avg_baseline_sim:.3f}")
    print(f"  Smart:    {avg_smart_sim:.3f} ({'+' if avg_smart_sim > avg_baseline_sim else ''}{(avg_smart_sim - avg_baseline_sim)*100:.1f}%)")

    avg_baseline_style = np.mean([r.baseline_style_score for r in results])
    avg_smart_style = np.mean([r.smart_style_score for r in results])
    print(f"\nStyle match:")
    print(f"  Baseline: {avg_baseline_style:.3f}")
    print(f"  Smart:    {avg_smart_style:.3f} ({'+' if avg_smart_style > avg_baseline_style else ''}{(avg_smart_style - avg_baseline_style)*100:.1f}%)")

    avg_baseline_len = np.mean([r.baseline_length_diff for r in results])
    avg_smart_len = np.mean([r.smart_length_diff for r in results])
    print(f"\nLength diff (lower is better):")
    print(f"  Baseline: {avg_baseline_len:.3f}")
    print(f"  Smart:    {avg_smart_len:.3f}")

    # Show some examples
    print("\n" + "-" * 40)
    print("EXAMPLE COMPARISONS")
    print("-" * 40)

    for r in results[:10]:
        print(f"\n[{r.contact}]")
        print(f"  Gold:     \"{r.gold_response}\"")
        print(f"  Baseline: \"{r.baseline_response}\" (sim={r.baseline_semantic_sim:.2f})")
        print(f"  Smart:    \"{r.smart_response}\" (sim={r.smart_semantic_sim:.2f})")

    # Save results
    results_data = {
        "model": model_id,
        "n_samples": len(results),
        "smart_wins": smart_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "avg_baseline_semantic_sim": avg_baseline_sim,
        "avg_smart_semantic_sim": avg_smart_sim,
        "avg_baseline_style": avg_baseline_style,
        "avg_smart_style": avg_smart_style,
        "samples": [
            {
                "id": r.sample_id,
                "contact": r.contact,
                "gold": r.gold_response,
                "baseline": r.baseline_response,
                "smart": r.smart_response,
                "baseline_sim": r.baseline_semantic_sim,
                "smart_sim": r.smart_semantic_sim,
            }
            for r in results
        ]
    }

    results_file = RESULTS_DIR / f"eval_{model_id}_{len(results)}.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved: {results_file}")

    # Cleanup
    loader.unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b", help="Model to evaluate")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples")
    args = parser.parse_args()

    run_evaluation(model_id=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
