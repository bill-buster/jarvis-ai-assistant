"""CLI for running HHEM hallucination evaluation benchmark.

Usage: python -m benchmarks.hallucination.run --output results/hhem.json

Workstream 2: HHEM Hallucination Benchmark
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.hallucination.datasets import (
    generate_grounded_pairs,
    generate_hallucinated_pairs,
    generate_mixed_dataset,
    get_dataset_metadata,
)
from benchmarks.hallucination.hhem import get_evaluator


def main() -> int:
    """Run HHEM hallucination evaluation benchmark.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Run HHEM hallucination evaluation benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        help="Name of the model being evaluated (for metadata)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["grounded", "hallucinated", "mixed"],
        default="mixed",
        help="Which dataset to evaluate: grounded (should score high), "
        "hallucinated (should score low), or mixed (both)",
    )
    parser.add_argument(
        "--templates",
        type=str,
        nargs="*",
        default=[],
        help="Filter by prompt templates (e.g., basic rag few_shot)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress and detailed output",
    )
    args = parser.parse_args()

    if args.verbose:
        print("Initializing HHEM evaluator...")
        print(f"Model name: {args.model_name}")
        print(f"Dataset: {args.dataset}")

    # Load appropriate dataset
    if args.dataset == "grounded":
        dataset = generate_grounded_pairs()
        if args.verbose:
            print(f"Loaded {len(dataset)} grounded pairs (should score high)")
    elif args.dataset == "hallucinated":
        dataset = generate_hallucinated_pairs()
        if args.verbose:
            print(f"Loaded {len(dataset)} hallucinated pairs (should score low)")
    else:  # mixed
        dataset = generate_mixed_dataset()
        if args.verbose:
            print(f"Loaded {len(dataset)} mixed pairs")

    if args.verbose:
        metadata = get_dataset_metadata()
        print(f"Dataset metadata: {metadata}")

    # Get evaluator
    evaluator = get_evaluator()

    if args.verbose:
        print("Running HHEM evaluation...")

    # Run benchmark
    result = evaluator.run_benchmark(
        model_name=args.model_name,
        dataset=dataset,
        prompt_templates=args.templates or [],
    )

    # Prepare output data (convert dataclass to dict, excluding full results for brevity)
    output_data = {
        "model_name": result.model_name,
        "num_samples": result.num_samples,
        "mean_score": round(result.mean_score, 4),
        "median_score": round(result.median_score, 4),
        "std_score": round(result.std_score, 4),
        "pass_rate_at_05": round(result.pass_rate_at_05, 4),
        "pass_rate_at_07": round(result.pass_rate_at_07, 4),
        "timestamp": result.timestamp,
        "dataset_type": args.dataset,
        "template_filter": args.templates or "all",
    }

    # Optionally include detailed results
    if args.verbose:
        output_data["detailed_results"] = [
            {
                "prompt_template": r.prompt_template,
                "hhem_score": round(r.hhem_score, 4),
                "source_preview": r.source_text[:100] + "..."
                if len(r.source_text) > 100
                else r.source_text,
                "summary_preview": r.generated_summary[:100] + "..."
                if len(r.generated_summary) > 100
                else r.generated_summary,
            }
            for r in result.results
        ]

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print("\n" + "=" * 50)
    print("HHEM BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {result.model_name}")
    print(f"Samples: {result.num_samples}")
    print(f"Mean Score: {result.mean_score:.4f}")
    print(f"Median Score: {result.median_score:.4f}")
    print(f"Std Dev: {result.std_score:.4f}")
    print(f"Pass Rate @0.5: {result.pass_rate_at_05:.1%}")
    print(f"Pass Rate @0.7: {result.pass_rate_at_07:.1%}")
    print("=" * 50)

    # Gate check (G3: mean >= 0.5)
    if result.mean_score >= 0.5:
        print("Gate G3: PASS (mean HHEM score >= 0.5)")
    elif result.mean_score >= 0.4:
        print("Gate G3: CONDITIONAL (mean HHEM score 0.4-0.5)")
    else:
        print("Gate G3: FAIL (mean HHEM score < 0.4)")

    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
