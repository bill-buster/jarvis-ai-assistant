#!/usr/bin/env python3
"""Evaluate all 3 models with model-specific tuned prompts.

Compares:
- Baseline (generic prompt)
- Smart (original smart_prompter)
- Tuned (model-specific prompts from model_prompts.py)

Usage:
    python scripts/evaluate_tuned_prompts.py
    python scripts/evaluate_tuned_prompts.py --limit 50
    python scripts/evaluate_tuned_prompts.py --model lfm2-2.6b-exp
"""

import argparse
import gc
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

MODELS = [
    ("qwen3-0.6b", "fast"),
    ("lfm2.5-1.2b", "balanced"),
    ("lfm2-2.6b-exp", "best"),
]


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def clean_response(text: str, model_id: str) -> str:
    """Clean model response based on model-specific patterns."""
    if not text:
        return ""

    # Remove <think>...</think> blocks (qwen3)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)

    # Remove /no_think prefix
    text = re.sub(r'^/no_think\s*', '', text)

    # Remove "me:" prefix that some models add
    text = re.sub(r'^me:\s*', '', text, flags=re.IGNORECASE)

    # Remove meta-commentary patterns (lfm2.5)
    meta_patterns = [
        r"^Sure!?\s*Here'?s?\s*(a\s*)?(casual\s*)?(version|reply).*?:\s*",
        r"^Here'?s?\s*(a\s*)?(quick|casual|short)\s*(version|reply).*?:\s*",
        r"^Got it!?\s*Here'?s?\s*.*?:\s*",
        r"^Alright,?\s*here'?s?\s*.*?:\s*",
    ]
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean whitespace
    text = text.strip()

    # Take first line if multiple (avoid rambling)
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


def evaluate_model(model_id: str, samples: list[dict], sim_model) -> dict:
    """Evaluate a single model with baseline, smart, and tuned prompts."""
    from core.models.loader import ModelLoader
    from core.generation.smart_prompter import build_smart_prompt
    from core.generation.model_prompts import build_model_prompt

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id}")
    print("=" * 60)

    loader = ModelLoader(model_id)
    loader.preload()

    results = []
    baseline_scores = []
    smart_scores = []
    tuned_scores = []

    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<think>"]

    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 25 == 0:
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

        conversation_text = "\n".join(conv_lines)

        # 1. Baseline prompt (generic)
        baseline_prompt = f"[brief, casual]\n\n{conversation_text}\nme:"

        # 2. Smart prompt (original)
        smart_result = build_smart_prompt(contact, conv_msgs, n_examples=2)
        smart_prompt = smart_result.prompt

        # 3. Tuned prompt (model-specific)
        tuned_result = build_model_prompt(model_id, contact, conv_msgs, n_examples=3)
        tuned_prompt = tuned_result.prompt

        # Add /no_think for qwen3 on non-tuned prompts
        if "qwen3" in model_id.lower():
            baseline_prompt = "/no_think\n" + baseline_prompt
            smart_prompt = "/no_think\n" + smart_prompt
            # tuned_prompt already has /no_think

        # Generate all three
        baseline_gen = loader.generate(baseline_prompt, max_tokens=50, temperature=0.3, stop=stop_seqs)
        smart_gen = loader.generate(smart_prompt, max_tokens=50, temperature=0.3, stop=stop_seqs)
        tuned_gen = loader.generate(tuned_prompt, max_tokens=50, temperature=0.3, stop=stop_seqs)

        baseline_resp = clean_response(baseline_gen.text, model_id)
        smart_resp = clean_response(smart_gen.text, model_id)
        tuned_resp = clean_response(tuned_gen.text, model_id)

        # Skip if all empty
        if not baseline_resp and not smart_resp and not tuned_resp:
            continue

        # Compute metrics
        all_texts = [
            gold,
            baseline_resp or "empty",
            smart_resp or "empty",
            tuned_resp or "empty"
        ]
        embeddings = sim_model.encode(all_texts, normalize_embeddings=True)

        baseline_sim = float(np.dot(embeddings[0], embeddings[1])) if baseline_resp else 0
        smart_sim = float(np.dot(embeddings[0], embeddings[2])) if smart_resp else 0
        tuned_sim = float(np.dot(embeddings[0], embeddings[3])) if tuned_resp else 0

        baseline_style = compute_style_score(baseline_resp, gold)
        smart_style = compute_style_score(smart_resp, gold)
        tuned_style = compute_style_score(tuned_resp, gold)

        # Combined score (60% semantic, 40% style)
        baseline_score = baseline_sim * 0.6 + baseline_style * 0.4
        smart_score = smart_sim * 0.6 + smart_style * 0.4
        tuned_score = tuned_sim * 0.6 + tuned_style * 0.4

        baseline_scores.append(baseline_score)
        smart_scores.append(smart_score)
        tuned_scores.append(tuned_score)

        results.append({
            "contact": contact,
            "gold": gold,
            "baseline_resp": baseline_resp,
            "smart_resp": smart_resp,
            "tuned_resp": tuned_resp,
            "baseline_sim": baseline_sim,
            "smart_sim": smart_sim,
            "tuned_sim": tuned_sim,
            "baseline_style": baseline_style,
            "smart_style": smart_style,
            "tuned_style": tuned_style,
        })

    elapsed = time.time() - start_time

    # Cleanup
    loader.unload()
    gc.collect()

    if not results:
        return {
            "model_id": model_id,
            "n_samples": 0,
            "error": "No valid samples processed"
        }

    # Calculate stats
    baseline_scores = np.array(baseline_scores)
    smart_scores = np.array(smart_scores)
    tuned_scores = np.array(tuned_scores)

    # Win comparisons (with 0.02 margin for ties)
    tuned_vs_baseline_wins = int(np.sum(tuned_scores > baseline_scores + 0.02))
    tuned_vs_smart_wins = int(np.sum(tuned_scores > smart_scores + 0.02))
    smart_vs_baseline_wins = int(np.sum(smart_scores > baseline_scores + 0.02))

    return {
        "model_id": model_id,
        "n_samples": len(results),
        "time_seconds": elapsed,
        # Win rates
        "tuned_vs_baseline_wins": tuned_vs_baseline_wins,
        "tuned_vs_smart_wins": tuned_vs_smart_wins,
        "smart_vs_baseline_wins": smart_vs_baseline_wins,
        "tuned_win_rate_vs_baseline": tuned_vs_baseline_wins / len(results),
        "tuned_win_rate_vs_smart": tuned_vs_smart_wins / len(results),
        # Averages
        "avg_baseline_sim": float(np.mean([r["baseline_sim"] for r in results])),
        "avg_smart_sim": float(np.mean([r["smart_sim"] for r in results])),
        "avg_tuned_sim": float(np.mean([r["tuned_sim"] for r in results])),
        "avg_baseline_style": float(np.mean([r["baseline_style"] for r in results])),
        "avg_smart_style": float(np.mean([r["smart_style"] for r in results])),
        "avg_tuned_style": float(np.mean([r["tuned_style"] for r in results])),
        # Sample results
        "samples": results[:20],
    }


def run_evaluation(limit: int = 100, model_filter: str | None = None):
    """Run evaluation on models."""
    from sentence_transformers import SentenceTransformer

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    samples = load_test_set(limit=limit)
    print(f"Loaded {len(samples)} samples")

    print("\nLoading similarity model...")
    sim_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    all_results = {}

    models_to_eval = MODELS
    if model_filter:
        models_to_eval = [(m, t) for m, t in MODELS if model_filter in m]

    for model_id, tier in models_to_eval:
        try:
            result = evaluate_model(model_id, samples, sim_model)
            all_results[model_id] = result
        except Exception as e:
            print(f"Error evaluating {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: Baseline vs Smart vs Tuned")
    print("=" * 80)

    print("\n{:<18} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}".format(
        "Model", "Tuned>Base%", "Tuned>Smart%", "Smart>Base%", "Tuned Sim", "Smart Sim", "Base Sim"
    ))
    print("-" * 80)

    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        if model_id not in all_results:
            continue
        r = all_results[model_id]
        if r.get("n_samples", 0) == 0:
            print(f"{model_id:<18} {'N/A (no samples)':>60}")
            continue
        total = r["n_samples"]
        print("{:<18} {:>11.1f}% {:>11.1f}% {:>11.1f}% {:>10.3f} {:>10.3f} {:>10.3f}".format(
            model_id,
            r["tuned_win_rate_vs_baseline"] * 100,
            r["tuned_win_rate_vs_smart"] * 100,
            r["smart_vs_baseline_wins"] / total * 100,
            r["avg_tuned_sim"],
            r["avg_smart_sim"],
            r["avg_baseline_sim"],
        ))

    # Print example comparisons
    print("\n" + "=" * 80)
    print("EXAMPLE RESPONSES (first 5)")
    print("=" * 80)

    for model_id in ["qwen3-0.6b", "lfm2.5-1.2b", "lfm2-2.6b-exp"]:
        if model_id not in all_results:
            continue
        if all_results[model_id].get("n_samples", 0) == 0:
            print(f"\n--- {model_id} ---\nNo samples processed")
            continue

        print(f"\n--- {model_id} ---")
        for s in all_results[model_id]["samples"][:5]:
            print(f"\nGold: \"{s['gold']}\"")
            print(f"  Baseline: \"{s['baseline_resp']}\" (sim={s['baseline_sim']:.2f})")
            print(f"  Smart:    \"{s['smart_resp']}\" (sim={s['smart_sim']:.2f})")
            print(f"  Tuned:    \"{s['tuned_resp']}\" (sim={s['tuned_sim']:.2f})")

    # Save results
    results_file = RESULTS_DIR / f"tuned_eval_{limit}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved: {results_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of samples")
    parser.add_argument("--model", type=str, help="Filter to specific model")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, model_filter=args.model)


if __name__ == "__main__":
    main()
