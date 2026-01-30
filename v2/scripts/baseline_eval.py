#!/usr/bin/env python3
"""Baseline evaluation - measure what the raw model can do.

This establishes the TRUE baseline before any fancy retrieval/prompting.
Tests multiple configurations to find what actually helps.

Usage:
    python scripts/baseline_eval.py --samples 5 --verbose          # Quick test
    python scripts/baseline_eval.py --samples 500 --config raw     # Full baseline
    python scripts/baseline_eval.py --samples 500 --config all     # Test all prompts
    python scripts/baseline_eval.py --model qwen3-0.6b             # Different model
    python scripts/baseline_eval.py --embedding bge                 # Different embedding
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Use clean test set (filtered for achievable responses) if available
CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
ORIGINAL_TEST_SET = Path("results/test_set/test_data.jsonl")
TEST_SET_FILE = CLEAN_TEST_SET if CLEAN_TEST_SET.exists() else ORIGINAL_TEST_SET
RESULTS_DIR = Path("results/baseline")

# Available embedding models
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",      # 80MB, fast, good for short text
    "bge": "BAAI/bge-base-en-v1.5",    # 440MB, higher quality
    "bge-small": "BAAI/bge-small-en-v1.5",  # 130MB, balanced
}

# Available LLMs (from model registry)
# Key = CLI arg, Value = model registry ID
LLMS = {
    "lfm2-2.6b-exp": "lfm2-2.6b-exp",
    "lfm2.5-1.2b": "lfm2.5-1.2b",
    "llama-3.2-1b": "llama-3.2-1b",
    "llama-3.2-3b": "llama-3.2-3b",
    "smollm2-1.7b": "smollm2-1.7b",
    "smollm3-3b": "smollm3-3b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "qwen3-0.6b": "qwen3-0.6b",
    "gemma3-1b": "gemma3-1b",
    "gemma3-4b": "gemma3-4b",
    "phi-3.5-mini": "phi-3.5-mini",
}


@dataclass
class EvalSample:
    """Result for a single sample - saved to JSON."""
    contact: str
    gold: str
    generated: str
    prompt_sent: str  # The actual prompt sent to LLM
    config: str
    model: str
    # Metrics
    exact_match: bool
    semantic_sim: float
    length_ratio: float
    first_word_match: bool
    style_score: float
    # Metadata
    is_group: bool
    gold_length: int
    gen_length: int
    length_bucket: str


def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test set samples."""
    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def is_group_chat(sample: dict) -> bool:
    """Detect if this is a group chat."""
    # New format has is_group field
    if "is_group" in sample:
        return sample["is_group"]
    # Old format - check contact name
    contact = sample.get("contact", "")
    return "," in contact or "+" in contact


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


def normalize_word(word: str) -> str:
    """Normalize common variants for comparison."""
    word = word.lower().strip()
    # Yes variants
    if word in {"yes", "yea", "yeah", "yep", "ya", "ye", "yup"}:
        return "yes"
    # No variants
    if word in {"no", "nah", "nope", "na"}:
        return "no"
    # Ok variants
    if word in {"ok", "okay", "k", "kk"}:
        return "ok"
    # Lol variants
    if word in {"lol", "lmao", "haha", "hahaha", "ha"}:
        return "lol"
    return word


def compute_metrics(gold: str, generated: str, sim_model) -> dict:
    """Compute all evaluation metrics."""
    gold_clean = gold.lower().strip()
    gen_clean = generated.lower().strip()

    # Exact match (case-insensitive)
    exact_match = gold_clean == gen_clean

    # Semantic similarity
    if generated and sim_model:
        embeddings = sim_model.encode([gold, generated], normalize_embeddings=True)
        semantic_sim = float(np.dot(embeddings[0], embeddings[1]))
    else:
        semantic_sim = 1.0 if exact_match else 0.0

    # Length ratio
    if gold:
        length_ratio = len(generated) / len(gold) if generated else 0.0
    else:
        length_ratio = 1.0 if not generated else 0.0

    # First word match (with normalization)
    gold_words = gold_clean.split()
    gen_words = gen_clean.split()
    gold_first = normalize_word(gold_words[0]) if gold_words else ""
    gen_first = normalize_word(gen_words[0]) if gen_words else ""
    first_word_match = gold_first == gen_first

    # Style score (simplified)
    style_score = 0.0
    # Punctuation match
    gold_has_punct = gold.rstrip()[-1:] in ".!?" if gold else False
    gen_has_punct = generated.rstrip()[-1:] in ".!?" if generated else False
    if gold_has_punct == gen_has_punct:
        style_score += 0.25
    # Length bucket match
    if 0.5 <= length_ratio <= 2.0:
        style_score += 0.25
    # Casual marker match (lol, haha)
    gold_casual = "lol" in gold_clean or "haha" in gold_clean
    gen_casual = "lol" in gen_clean or "haha" in gen_clean
    if gold_casual == gen_casual:
        style_score += 0.25
    # Abbreviation match
    abbrevs = ["u", "ur", "rn", "tmrw", "idk"]
    gold_abbrev = any(f" {a} " in f" {gold_clean} " for a in abbrevs)
    gen_abbrev = any(f" {a} " in f" {gen_clean} " for a in abbrevs)
    if gold_abbrev == gen_abbrev:
        style_score += 0.25

    return {
        "exact_match": exact_match,
        "semantic_sim": semantic_sim,
        "length_ratio": length_ratio,
        "first_word_match": first_word_match,
        "style_score": style_score,
    }


# ============================================================================
# PROMPT CONFIGURATIONS TO TEST
# ============================================================================

def build_prompt_raw(conversation: str, last_msg: str) -> str:
    """Config 0: Raw - just the conversation, no instructions."""
    return f"{conversation}\nme:"


def build_prompt_simple(conversation: str, last_msg: str) -> str:
    """Config 1: Simple instruction."""
    return f"Reply to this conversation:\n\n{conversation}\nme:"


def build_prompt_casual(conversation: str, last_msg: str) -> str:
    """Config 2: Casual style instruction."""
    return f"[casual, brief, lowercase]\n\n{conversation}\nme:"


def build_prompt_detailed(conversation: str, last_msg: str) -> str:
    """Config 3: Detailed style instruction."""
    return (
        "[Reply as a casual texter. Keep it short (under 30 chars). "
        "Use lowercase, no periods. Match the vibe.]\n\n"
        f"{conversation}\nme:"
    )


PROMPT_CONFIGS = {
    "raw": build_prompt_raw,
    "simple": build_prompt_simple,
    "casual": build_prompt_casual,
    "detailed": build_prompt_detailed,
}


def parse_conversation(sample: dict) -> tuple[str, str]:
    """Parse conversation from test set format."""
    text = sample.get("conversation", sample.get("prompt", ""))

    lines = text.strip().split("\n")
    conv_lines = []
    last_their_msg = ""

    for line in lines:
        line = line.strip()
        if line.startswith("[") and "]" in line:
            continue
        if line.startswith("me:") or line.startswith("them:"):
            conv_lines.append(line)
            if line.startswith("them:"):
                last_their_msg = line.replace("them:", "").strip()

    conv_lines = conv_lines[-10:]
    return "\n".join(conv_lines), last_their_msg


def run_baseline_eval(
    config_name: str = "raw",
    model_name: str = "lfm2-2.6b",
    embedding_name: str = "minilm",
    limit: int = 50,
    verbose: bool = False,
):
    """Run baseline evaluation with specified configuration."""
    from sentence_transformers import SentenceTransformer
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION")
    print(f"  Config: {config_name}")
    print(f"  Model: {model_name}")
    print(f"  Embedding: {embedding_name}")
    print(f"  Samples: {limit}")
    print(f"{'='*60}")

    # Load test set
    print("\nLoading test set...")
    samples = load_test_set(limit=limit)
    print(f"Loaded {len(samples)} samples from {TEST_SET_FILE.name}")

    # Load embedding model
    print(f"\nLoading embedding model ({embedding_name})...")
    embedding_model_name = EMBEDDING_MODELS.get(embedding_name, embedding_name)
    sim_model = SentenceTransformer(embedding_model_name)

    # Load LLM
    llm_model_id = LLMS.get(model_name, model_name)
    print(f"\nLoading LLM ({llm_model_id})...")
    loader = ModelLoader(llm_model_id)
    loader.preload()

    # Get prompt builder
    build_prompt = PROMPT_CONFIGS[config_name]

    # Run evaluation
    results = []
    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>", "Them:", "Me:"]

    print(f"\nEvaluating...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0 or verbose:
            print(f"  [{i+1}/{len(samples)}]")

        contact = sample.get("contact", "Unknown")
        gold = sample["gold_response"]

        # Parse conversation
        conversation, last_msg = parse_conversation(sample)
        if not conversation:
            continue

        # Build prompt
        prompt = build_prompt(conversation, last_msg)

        # Generate
        result = loader.generate(
            prompt=prompt,
            max_tokens=40,
            temperature=0.2,
            stop=stop_seqs,
        )
        generated = clean_response(result.text)

        # Compute metrics
        metrics = compute_metrics(gold, generated, sim_model)

        # Determine length bucket
        gold_len = len(gold)
        if gold_len < 15:
            length_bucket = "short"
        elif gold_len < 40:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        eval_sample = EvalSample(
            contact=contact,
            gold=gold,
            generated=generated,
            prompt_sent=prompt,
            config=config_name,
            model=model_name,
            is_group=is_group_chat(sample),
            gold_length=len(gold),
            gen_length=len(generated),
            length_bucket=length_bucket,
            **metrics,
        )
        results.append(eval_sample)

        if verbose:
            marker = "✓" if metrics["exact_match"] else ("~" if metrics["semantic_sim"] > 0.7 else "✗")
            print(f"    {marker} Gold: \"{gold[:40]}\" | Gen: \"{generated[:40]}\" | Sim: {metrics['semantic_sim']:.2f}")

    elapsed = time.time() - start_time

    if not results:
        print("No valid results!")
        return None

    # Compute aggregates
    print_results(results, config_name, model_name, embedding_name, elapsed)

    # Save results
    save_results(results, config_name, model_name, embedding_name, limit)

    return results


def print_results(results: list[EvalSample], config: str, model: str, embedding: str, elapsed: float):
    """Print evaluation results."""
    n = len(results)

    # Overall metrics
    exact_matches = sum(1 for r in results if r.exact_match)
    avg_sim = np.mean([r.semantic_sim for r in results])
    avg_style = np.mean([r.style_score for r in results])
    first_word_matches = sum(1 for r in results if r.first_word_match)
    good_length = sum(1 for r in results if 0.5 <= r.length_ratio <= 2.0)

    # By length bucket
    by_length = {}
    for bucket in ["short", "medium", "long"]:
        bucket_results = [r for r in results if r.length_bucket == bucket]
        if bucket_results:
            by_length[bucket] = {
                "n": len(bucket_results),
                "exact": sum(1 for r in bucket_results if r.exact_match),
                "avg_sim": np.mean([r.semantic_sim for r in bucket_results]),
            }

    # By chat type
    group_results = [r for r in results if r.is_group]
    individual_results = [r for r in results if not r.is_group]

    # Print
    print(f"\n{'='*60}")
    print(f"RESULTS: {config} | {model} | {embedding}")
    print(f"{'='*60}")
    print(f"Samples: {n} | Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    print(f"\n--- OVERALL ---")
    print(f"  Exact matches:     {exact_matches:3d}/{n} ({exact_matches/n*100:5.1f}%)")
    print(f"  First word match:  {first_word_matches:3d}/{n} ({first_word_matches/n*100:5.1f}%)")
    print(f"  Good length ratio: {good_length:3d}/{n} ({good_length/n*100:5.1f}%)")
    print(f"  Avg semantic sim:  {avg_sim:.3f}")
    print(f"  Avg style score:   {avg_style:.3f}")

    print(f"\n--- BY RESPONSE LENGTH ---")
    for bucket in ["short", "medium", "long"]:
        if bucket in by_length:
            b = by_length[bucket]
            print(f"  {bucket:8}: {b['n']:3d} samples | exact: {b['exact']:2d} ({b['exact']/b['n']*100:4.1f}%) | sim: {b['avg_sim']:.3f}")

    print(f"\n--- BY CHAT TYPE ---")
    if individual_results:
        avg = np.mean([r.semantic_sim for r in individual_results])
        exact = sum(1 for r in individual_results if r.exact_match)
        print(f"  1:1 chats: {len(individual_results):3d} samples | exact: {exact:2d} ({exact/len(individual_results)*100:4.1f}%) | sim: {avg:.3f}")
    if group_results:
        avg = np.mean([r.semantic_sim for r in group_results])
        exact = sum(1 for r in group_results if r.exact_match)
        print(f"  Groups:    {len(group_results):3d} samples | exact: {exact:2d} ({exact/len(group_results)*100:4.1f}%) | sim: {avg:.3f}")


def save_results(results: list[EvalSample], config: str, model: str, embedding: str, limit: int):
    """Save results to JSON."""
    n = len(results)
    results_file = RESULTS_DIR / f"baseline_{config}_{model}_{limit}.json"

    with open(results_file, "w") as f:
        json.dump({
            "config": config,
            "model": model,
            "embedding": embedding,
            "n_samples": n,
            "metrics": {
                "exact_match_rate": sum(1 for r in results if r.exact_match) / n,
                "first_word_match_rate": sum(1 for r in results if r.first_word_match) / n,
                "avg_semantic_sim": float(np.mean([r.semantic_sim for r in results])),
                "avg_style_score": float(np.mean([r.style_score for r in results])),
            },
            "samples": [asdict(r) for r in results],
        }, f, indent=2)

    print(f"\nSaved: {results_file}")


def run_all_configs(model_name: str, embedding_name: str, limit: int):
    """Run all prompt configurations and compare."""
    print("\n" + "="*70)
    print("RUNNING ALL PROMPT CONFIGURATIONS")
    print("="*70)

    all_results = {}

    for config_name in PROMPT_CONFIGS:
        results = run_baseline_eval(
            config_name=config_name,
            model_name=model_name,
            embedding_name=embedding_name,
            limit=limit,
        )
        if results:
            n = len(results)
            all_results[config_name] = {
                "exact_match_rate": sum(1 for r in results if r.exact_match) / n,
                "avg_sim": float(np.mean([r.semantic_sim for r in results])),
                "avg_style": float(np.mean([r.style_score for r in results])),
                "first_word_rate": sum(1 for r in results if r.first_word_match) / n,
            }

    # Print comparison
    print("\n" + "="*70)
    print(f"COMPARISON SUMMARY ({model_name})")
    print("="*70)
    print(f"\n{'Config':<12} {'Exact%':>8} {'Sim':>8} {'Style':>8} {'1stWord%':>10}")
    print("-" * 50)

    for config, metrics in sorted(all_results.items(), key=lambda x: x[1]["avg_sim"], reverse=True):
        print(f"{config:<12} {metrics['exact_match_rate']*100:>7.1f}% {metrics['avg_sim']:>8.3f} {metrics['avg_style']:>8.3f} {metrics['first_word_rate']*100:>9.1f}%")

    # Save comparison
    comparison_file = RESULTS_DIR / f"comparison_{model_name}_{limit}.json"
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comparison: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--config", type=str, default="raw",
                        choices=list(PROMPT_CONFIGS.keys()) + ["all"],
                        help="Prompt configuration to test")
    parser.add_argument("--model", type=str, default="lfm2-2.6b-exp",
                        choices=list(LLMS.keys()),
                        help="LLM to use")
    parser.add_argument("--embedding", type=str, default="minilm",
                        choices=list(EMBEDDING_MODELS.keys()),
                        help="Embedding model for similarity")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each sample")
    args = parser.parse_args()

    if args.config == "all":
        run_all_configs(args.model, args.embedding, args.samples)
    else:
        run_baseline_eval(
            config_name=args.config,
            model_name=args.model,
            embedding_name=args.embedding,
            limit=args.samples,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
