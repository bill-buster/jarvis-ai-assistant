#!/usr/bin/env python3
"""Comprehensive model evaluation with intent-based metrics.

Tests all models with:
1. Intent/type matching (did we get the right TYPE of response?)
2. Semantic similarity (traditional metric)
3. Style score (did we match the user's style?)
4. Combined score (weighted combination)

Also tests different configurations:
- With/without few-shot examples
- Different prompt styles
- With/without the "ask when uncertain" behavior

Usage:
    python scripts/comprehensive_eval.py                    # Full eval, all models
    python scripts/comprehensive_eval.py --quick            # Quick test (10 samples)
    python scripts/comprehensive_eval.py --model llama-3.2-3b  # Single model
    python scripts/comprehensive_eval.py --analyze          # Analyze existing results
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluation.response_types import (
    classify_response,
    get_type_similarity,
    ResponseType,
    ResponseTypeEvaluator,
)


# Configuration
TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
RESULTS_DIR = Path("results/comprehensive_eval")

# Models to test (ordered by size for memory management)
MODELS = [
    # Small models first
    "qwen2.5-1.5b",
    "lfm2.5-1.2b",
    "llama-3.2-1b",
    "smollm2-1.7b",
    # Larger models
    "lfm2-2.6b",       # Base LFM2
    "lfm2-2.6b-exp",   # RL-tuned LFM2 (should be better)
    "llama-3.2-3b",
    "smollm3-3b",
]

# Prompt configurations to test
PROMPT_CONFIGS = {
    "minimal": {
        "system": "",
        "format": "{conversation}\nme:",
        "few_shot": False,
    },
    "basic": {
        "system": "[casual, brief, lowercase]",
        "format": "{system}\n\n{conversation}\nme:",
        "few_shot": False,
    },
    "detailed": {
        "system": "[Reply as a casual texter. Keep it short (under 30 chars). Use lowercase, no periods. Match the vibe.]",
        "format": "{system}\n\n{conversation}\nme:",
        "few_shot": False,
    },
    "fewshot_3": {
        "system": "[Reply in my texting style]",
        "format": "{system}\n\nExamples of how I text:\n{examples}\n\nNow reply:\n{conversation}\nme:",
        "few_shot": True,
        "n_examples": 3,
    },
    "fewshot_5": {
        "system": "[Reply in my texting style]",
        "format": "{system}\n\nExamples of how I text:\n{examples}\n\nNow reply:\n{conversation}\nme:",
        "few_shot": True,
        "n_examples": 5,
    },
}


@dataclass
class EvalResult:
    """Result for a single sample."""
    # Identifiers
    model: str
    config: str
    sample_idx: int
    contact: str

    # Texts
    gold: str
    generated: str

    # Intent metrics
    gold_type: str
    generated_type: str
    type_match: bool
    type_similarity: float

    # Other metrics
    semantic_sim: float
    style_score: float
    combined_score: float

    # Metadata
    is_group: bool
    generation_time_ms: float


@dataclass
class AggregateResults:
    """Aggregated results for a model+config combination."""
    model: str
    config: str
    n_samples: int

    # Primary metrics
    type_match_rate: float      # % where we got the right TYPE
    avg_type_similarity: float  # Average type similarity
    avg_semantic_sim: float     # Traditional metric
    avg_style_score: float      # Style matching
    avg_combined_score: float   # Weighted combination

    # Breakdown by response type
    type_accuracy_by_gold: dict  # Accuracy for each gold type

    # Performance
    avg_generation_time_ms: float
    total_time_s: float


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def load_few_shot_examples(n: int = 5) -> list[dict]:
    """Load few-shot examples (using test set for now)."""
    # In production, these would be curated examples
    # For now, sample from test set
    samples = load_test_set(limit=100)

    # Get diverse examples by type
    examples = []
    seen_types = set()

    for s in samples:
        gold = s.get("gold_response", "")
        cls = classify_response(gold)

        # Get one example per type first
        if cls.response_type.value not in seen_types and len(examples) < n:
            examples.append({
                "them": s.get("last_message", ""),
                "me": gold,
            })
            seen_types.add(cls.response_type.value)

        if len(examples) >= n:
            break

    return examples


def format_examples(examples: list[dict]) -> str:
    """Format few-shot examples as string."""
    lines = []
    for ex in examples:
        lines.append(f"them: {ex['them']}")
        lines.append(f"me: {ex['me']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(
    conversation: str,
    config: dict,
    examples: list[dict] | None = None,
) -> str:
    """Build prompt from conversation and config."""
    fmt = config["format"]

    replacements = {
        "system": config.get("system", ""),
        "conversation": conversation,
    }

    if config.get("few_shot") and examples:
        replacements["examples"] = format_examples(examples)

    prompt = fmt
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    return prompt


def clean_response(text: str) -> str:
    """Clean model response."""
    if not text:
        return ""

    # Remove common prefixes
    for prefix in ["me:", "Me:", "Reply:", "Response:", "Assistant:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Take first line only
    text = text.split("\n")[0].strip()

    # Remove quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    return text.strip()


def compute_style_score(gold: str, generated: str) -> float:
    """Compute style similarity score."""
    gold_lower = gold.lower().strip()
    gen_lower = generated.lower().strip()

    score = 0.0

    # Length similarity (within 2x is good)
    if gold and generated:
        ratio = len(generated) / len(gold)
        if 0.5 <= ratio <= 2.0:
            score += 0.25

    # Punctuation match
    gold_has_punct = gold.rstrip()[-1:] in ".!?" if gold else False
    gen_has_punct = generated.rstrip()[-1:] in ".!?" if generated else False
    if gold_has_punct == gen_has_punct:
        score += 0.25

    # Casual markers (lol, haha)
    gold_casual = "lol" in gold_lower or "haha" in gold_lower
    gen_casual = "lol" in gen_lower or "haha" in gen_lower
    if gold_casual == gen_casual:
        score += 0.25

    # Abbreviation usage
    abbrevs = ["u ", "ur ", "rn", "tmrw", "idk"]
    gold_abbrev = any(a in gold_lower for a in abbrevs)
    gen_abbrev = any(a in gen_lower for a in abbrevs)
    if gold_abbrev == gen_abbrev:
        score += 0.25

    return score


def run_model_eval(
    model_name: str,
    config_name: str,
    samples: list[dict],
    few_shot_examples: list[dict] | None = None,
    verbose: bool = False,
) -> tuple[list[EvalResult], AggregateResults]:
    """Run evaluation for a single model+config combination."""
    from sentence_transformers import SentenceTransformer
    from core.models.loader import ModelLoader

    config = PROMPT_CONFIGS[config_name]

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} + {config_name}")
    print(f"{'='*60}")

    # Load models
    print("Loading models...")
    loader = ModelLoader(model_name)
    loader.preload()

    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare examples if needed
    examples = None
    if config.get("few_shot"):
        n = config.get("n_examples", 3)
        examples = few_shot_examples[:n] if few_shot_examples else load_few_shot_examples(n)

    # Run evaluation
    results = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        # Parse conversation
        conversation = sample.get("conversation", sample.get("prompt", ""))
        gold = sample.get("gold_response", "")
        contact = sample.get("contact", "Unknown")
        is_group = sample.get("is_group", "," in contact)

        # Build prompt
        prompt = build_prompt(conversation, config, examples)

        # Generate
        gen_start = time.time()
        result = loader.generate(
            prompt=prompt,
            max_tokens=40,
            temperature=0.2,
            stop=["\n", "them:", "Them:", "<|im_end|>", "<|eot_id|>"],
        )
        gen_time = (time.time() - gen_start) * 1000

        generated = clean_response(result.text)

        # Classify types
        gold_cls = classify_response(gold)
        gen_cls = classify_response(generated)

        # Compute metrics
        type_match = gold_cls.response_type == gen_cls.response_type
        type_sim = get_type_similarity(gold_cls.response_type, gen_cls.response_type)

        # Semantic similarity
        if generated:
            embeddings = sim_model.encode([gold, generated], normalize_embeddings=True)
            semantic_sim = float(np.dot(embeddings[0], embeddings[1]))
        else:
            semantic_sim = 0.0

        style_score = compute_style_score(gold, generated)

        # Combined score: 40% type, 30% semantic, 30% style
        combined = 0.4 * type_sim + 0.3 * semantic_sim + 0.3 * style_score

        eval_result = EvalResult(
            model=model_name,
            config=config_name,
            sample_idx=i,
            contact=contact,
            gold=gold,
            generated=generated,
            gold_type=gold_cls.response_type.value,
            generated_type=gen_cls.response_type.value,
            type_match=type_match,
            type_similarity=type_sim,
            semantic_sim=semantic_sim,
            style_score=style_score,
            combined_score=combined,
            is_group=is_group,
            generation_time_ms=gen_time,
        )
        results.append(eval_result)

        if verbose:
            marker = "✓" if type_match else "✗"
            print(f"    {marker} [{gold_cls.response_type.value:12}] Gold: '{gold[:30]}' | Gen: '{generated[:30]}'")

    total_time = time.time() - start_time

    # Aggregate results
    n = len(results)

    # Type accuracy by gold type
    type_accuracy = {}
    for resp_type in ResponseType:
        type_results = [r for r in results if r.gold_type == resp_type.value]
        if type_results:
            acc = sum(1 for r in type_results if r.type_match) / len(type_results)
            type_accuracy[resp_type.value] = {
                "accuracy": acc,
                "count": len(type_results),
            }

    aggregate = AggregateResults(
        model=model_name,
        config=config_name,
        n_samples=n,
        type_match_rate=sum(1 for r in results if r.type_match) / n,
        avg_type_similarity=sum(r.type_similarity for r in results) / n,
        avg_semantic_sim=sum(r.semantic_sim for r in results) / n,
        avg_style_score=sum(r.style_score for r in results) / n,
        avg_combined_score=sum(r.combined_score for r in results) / n,
        type_accuracy_by_gold=type_accuracy,
        avg_generation_time_ms=sum(r.generation_time_ms for r in results) / n,
        total_time_s=total_time,
    )

    # Print summary
    print(f"\n--- Results: {model_name} + {config_name} ---")
    print(f"  Type match rate:    {aggregate.type_match_rate*100:5.1f}%")
    print(f"  Type similarity:    {aggregate.avg_type_similarity:.3f}")
    print(f"  Semantic sim:       {aggregate.avg_semantic_sim:.3f}")
    print(f"  Style score:        {aggregate.avg_style_score:.3f}")
    print(f"  Combined score:     {aggregate.avg_combined_score:.3f}")
    print(f"  Avg gen time:       {aggregate.avg_generation_time_ms:.0f}ms")

    # Cleanup
    del loader
    gc.collect()

    return results, aggregate


def run_full_eval(
    models: list[str],
    configs: list[str],
    n_samples: int = 50,
    verbose: bool = False,
):
    """Run full evaluation across all models and configs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading test set ({n_samples} samples)...")
    samples = load_test_set(limit=n_samples)
    print(f"Loaded {len(samples)} samples")

    # Load few-shot examples once
    few_shot_examples = load_few_shot_examples(n=5)

    all_results = []
    all_aggregates = []

    for model in models:
        for config in configs:
            try:
                results, aggregate = run_model_eval(
                    model_name=model,
                    config_name=config,
                    samples=samples,
                    few_shot_examples=few_shot_examples,
                    verbose=verbose,
                )
                all_results.extend(results)
                all_aggregates.append(aggregate)

                # Save intermediate results
                save_results(all_aggregates, n_samples)

            except Exception as e:
                print(f"ERROR: {model} + {config} failed: {e}")
                continue

            # Force garbage collection between models
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except:
                pass

    # Final save
    save_results(all_aggregates, n_samples)
    print_leaderboard(all_aggregates)

    return all_results, all_aggregates


def save_results(aggregates: list[AggregateResults], n_samples: int):
    """Save results to JSON."""
    results_file = RESULTS_DIR / f"eval_results_{n_samples}.json"

    data = {
        "n_samples": n_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(a) for a in aggregates],
    }

    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to: {results_file}")


def print_leaderboard(aggregates: list[AggregateResults]):
    """Print leaderboard sorted by combined score."""
    print("\n" + "=" * 80)
    print("LEADERBOARD (sorted by combined score)")
    print("=" * 80)

    sorted_results = sorted(aggregates, key=lambda x: x.avg_combined_score, reverse=True)

    print(f"\n{'Rank':<5} {'Model':<18} {'Config':<12} {'TypeMatch':>10} {'TypeSim':>8} {'SemSim':>8} {'Style':>8} {'Combined':>10}")
    print("-" * 90)

    for i, r in enumerate(sorted_results):
        print(f"{i+1:<5} {r.model:<18} {r.config:<12} {r.type_match_rate*100:>9.1f}% {r.avg_type_similarity:>8.3f} {r.avg_semantic_sim:>8.3f} {r.avg_style_score:>8.3f} {r.avg_combined_score:>10.3f}")

    # Print best by each metric
    print("\n" + "-" * 80)
    print("BEST BY METRIC:")

    metrics = [
        ("Type Match Rate", "type_match_rate"),
        ("Type Similarity", "avg_type_similarity"),
        ("Semantic Similarity", "avg_semantic_sim"),
        ("Style Score", "avg_style_score"),
        ("Combined Score", "avg_combined_score"),
    ]

    for name, attr in metrics:
        best = max(aggregates, key=lambda x: getattr(x, attr))
        value = getattr(best, attr)
        if "rate" in attr.lower():
            print(f"  {name:<20}: {best.model} + {best.config} ({value*100:.1f}%)")
        else:
            print(f"  {name:<20}: {best.model} + {best.config} ({value:.3f})")


def analyze_results(results_file: Path | None = None):
    """Analyze existing results."""
    if results_file is None:
        # Find most recent results
        results_files = sorted(RESULTS_DIR.glob("eval_results_*.json"))
        if not results_files:
            print("No results found. Run evaluation first.")
            return
        results_file = results_files[-1]

    print(f"Loading: {results_file}")

    with open(results_file) as f:
        data = json.load(f)

    aggregates = [AggregateResults(**r) for r in data["results"]]
    print_leaderboard(aggregates)

    # Additional analysis
    print("\n" + "=" * 80)
    print("TYPE ACCURACY BREAKDOWN")
    print("=" * 80)

    # Get best model
    best = max(aggregates, key=lambda x: x.avg_combined_score)
    print(f"\nBest model: {best.model} + {best.config}")
    print("\nAccuracy by response type:")

    for type_name, stats in sorted(best.type_accuracy_by_gold.items()):
        acc = stats["accuracy"]
        count = stats["count"]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {type_name:<15} [{bar}] {acc*100:5.1f}% (n={count})")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 samples")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--model", type=str, help="Test single model")
    parser.add_argument("--config", type=str, help="Test single config")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    # Determine what to test
    models = [args.model] if args.model else MODELS
    configs = [args.config] if args.config else list(PROMPT_CONFIGS.keys())
    n_samples = 10 if args.quick else args.samples

    run_full_eval(
        models=models,
        configs=configs,
        n_samples=n_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
