#!/usr/bin/env python3
"""Grid search across models and prompt configs.

Finds the best (model, config) combination efficiently.

Usage:
    python scripts/grid_search_baseline.py                    # 100 samples, all combos
    python scripts/grid_search_baseline.py --samples 200      # More samples
    python scripts/grid_search_baseline.py --validate         # Validate top 3 on 500
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("results/baseline")

# Models to test - prioritized for casual text generation
MODELS = [
    "lfm2-2.6b-exp",   # RL-tuned, best on benchmarks
    "lfm2.5-1.2b",     # Fast, natural conversation
    "llama-3.2-1b",    # Good at few-shot examples
    "llama-3.2-3b",    # Better quality Llama
    "smollm2-1.7b",    # HF's best small model
    "smollm3-3b",      # Newest, beats Llama-3.2-3B
    "qwen2.5-1.5b",    # "Better for casual text" than Qwen3
]
CONFIGS = ["raw", "simple", "casual", "detailed"]


def run_eval(model: str, config: str, samples: int) -> dict | None:
    """Run single evaluation and return metrics."""
    print(f"\n{'='*50}")
    print(f"Testing: {model} + {config}")
    print(f"{'='*50}")

    cmd = [
        "uv", "run", "python", "scripts/baseline_eval.py",
        "--model", model,
        "--config", config,
        "--samples", str(samples),
    ]

    result = subprocess.run(cmd, capture_output=False)

    # Load results
    results_file = RESULTS_DIR / f"baseline_{config}_{model}_{samples}.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            return {
                "model": model,
                "config": config,
                "exact_match": data["metrics"]["exact_match_rate"],
                "semantic_sim": data["metrics"]["avg_semantic_sim"],
                "style_score": data["metrics"]["avg_style_score"],
                "first_word": data["metrics"]["first_word_match_rate"],
            }
    return None


def grid_search(samples: int = 100):
    """Run grid search across all models and configs."""
    results = []

    for model in MODELS:
        for config in CONFIGS:
            metrics = run_eval(model, config, samples)
            if metrics:
                results.append(metrics)

    # Sort by semantic similarity (primary metric)
    results.sort(key=lambda x: x["semantic_sim"], reverse=True)

    # Print leaderboard
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS (sorted by semantic similarity)")
    print("="*70)
    print(f"\n{'Rank':<5} {'Model':<15} {'Config':<10} {'Exact%':>8} {'Sim':>8} {'Style':>8} {'1stWord%':>10}")
    print("-" * 70)

    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['model']:<15} {r['config']:<10} {r['exact_match']*100:>7.1f}% {r['semantic_sim']:>8.3f} {r['style_score']:>8.3f} {r['first_word']*100:>9.1f}%")

    # Save leaderboard
    leaderboard_file = RESULTS_DIR / f"grid_search_{samples}.json"
    with open(leaderboard_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {leaderboard_file}")

    # Print recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    if results:
        best = results[0]
        print(f"\nBest combination: {best['model']} + {best['config']}")
        print(f"  Semantic sim: {best['semantic_sim']:.3f}")
        print(f"  Exact match:  {best['exact_match']*100:.1f}%")

        print(f"\nTo validate on more samples, run:")
        print(f"  uv run python scripts/baseline_eval.py --model {best['model']} --config {best['config']} --samples 500")

    return results


def validate_top_n(n: int = 3, samples: int = 500):
    """Validate top N combinations on more samples."""
    # Load grid search results
    grid_files = sorted(RESULTS_DIR.glob("grid_search_*.json"))
    if not grid_files:
        print("No grid search results found. Run grid search first.")
        return

    with open(grid_files[-1]) as f:
        grid_results = json.load(f)

    print(f"\nValidating top {n} combinations on {samples} samples...")

    validation_results = []
    for r in grid_results[:n]:
        metrics = run_eval(r["model"], r["config"], samples)
        if metrics:
            validation_results.append(metrics)

    # Print validation results
    print("\n" + "="*70)
    print(f"VALIDATION RESULTS ({samples} samples)")
    print("="*70)
    print(f"\n{'Model':<15} {'Config':<10} {'Exact%':>8} {'Sim':>8} {'Style':>8}")
    print("-" * 50)

    for r in sorted(validation_results, key=lambda x: x["semantic_sim"], reverse=True):
        print(f"{r['model']:<15} {r['config']:<10} {r['exact_match']*100:>7.1f}% {r['semantic_sim']:>8.3f} {r['style_score']:>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Samples per combo")
    parser.add_argument("--validate", action="store_true", help="Validate top 3 on 500 samples")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate:
        validate_top_n(n=3, samples=500)
    else:
        grid_search(samples=args.samples)


if __name__ == "__main__":
    main()
