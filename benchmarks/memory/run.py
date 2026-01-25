"""CLI for running memory profiling benchmark.

Usage: python -m benchmarks.memory.run --output results/memory.json

Workstream 1: Memory Profiler
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from benchmarks.memory.models import (
    CONTEXT_LENGTHS,
    get_default_model,
    get_models_for_profiling,
)
from benchmarks.memory.profiler import MLXMemoryProfiler

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    """Run memory profiling benchmark.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Profile MLX model memory usage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model path to profile (default: all configured models)",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
        help="Context lengths to test",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: profile only default model at 512 context",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Include memory measurement during generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and debug output",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine models to profile
    if args.model:
        from benchmarks.memory.models import ModelSpec

        models = [
            ModelSpec(
                path=args.model,
                name=args.model.split("/")[-1],
                estimated_memory_mb=0,
                description="User-specified model",
            )
        ]
    elif args.quick:
        models = [get_default_model()]
        args.context_lengths = [512]
    else:
        models = get_models_for_profiling()

    # Initialize profiler
    profiler = MLXMemoryProfiler()
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    total_profiles = len(models) * len(args.context_lengths)
    print(f"Profiling {len(models)} model(s) at {len(args.context_lengths)} context length(s)")
    print(f"Total profiles to run: {total_profiles}")
    print("-" * 60)

    profile_count = 0
    for model in models:
        print(f"\nModel: {model.name}")
        print(f"  Path: {model.path}")

        for ctx_len in args.context_lengths:
            profile_count += 1
            print(f"\n  [{profile_count}/{total_profiles}] Context length: {ctx_len}")

            try:
                if args.with_generation:
                    profile = profiler.profile_with_generation(
                        model_path=model.path,
                        context_length=ctx_len,
                    )
                else:
                    profile = profiler.profile_model(
                        model_path=model.path,
                        context_length=ctx_len,
                    )

                profile_dict = asdict(profile)
                results.append(profile_dict)

                print(f"    RSS:   {profile.rss_mb:>8.1f} MB")
                print(f"    Metal: {profile.metal_mb:>8.1f} MB")
                print(f"    Load:  {profile.load_time_seconds:>8.2f} s")

            except FileNotFoundError:
                error = {
                    "model": model.path,
                    "context_length": ctx_len,
                    "error": "Model not found",
                }
                errors.append(error)
                print(f"    ERROR: Model not found - {model.path}")

            except Exception as e:
                error = {
                    "model": model.path,
                    "context_length": ctx_len,
                    "error": str(e),
                }
                errors.append(error)
                print(f"    ERROR: {e}")
                logger.exception("Error profiling %s", model.path)

    # Prepare output data
    summary: dict[str, object] = {
        "total_profiles": len(results),
        "total_errors": len(errors),
        "models_tested": len(models),
        "context_lengths_tested": args.context_lengths,
    }

    # Add aggregate stats if we have results
    if results:
        rss_values: list[float] = [r["rss_mb"] for r in results]  # type: ignore[misc]
        metal_values: list[float] = [r["metal_mb"] for r in results]  # type: ignore[misc]
        load_times: list[float] = [r["load_time_seconds"] for r in results]  # type: ignore[misc]

        summary["max_rss_mb"] = max(rss_values)
        summary["avg_rss_mb"] = sum(rss_values) / len(rss_values)
        summary["max_metal_mb"] = max(metal_values)
        summary["avg_metal_mb"] = sum(metal_values) / len(metal_values)
        summary["avg_load_time_seconds"] = sum(load_times) / len(load_times)

    output_data: dict[str, object] = {
        "profiles": results,
        "errors": errors,
        "summary": summary,
    }

    # Save results
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Profiles completed: {len(results)}")
    print(f"Errors: {len(errors)}")

    if results:
        print(f"\nMax RSS Memory:   {summary['max_rss_mb']:.1f} MB")
        print(f"Max Metal Memory: {summary['max_metal_mb']:.1f} MB")
        print(f"Avg Load Time:    {summary['avg_load_time_seconds']:.2f} s")

    print(f"\nResults saved to: {args.output}")

    # Return error code if any profiles failed
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
