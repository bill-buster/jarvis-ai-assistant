"""CLI for running template coverage analysis.

Usage: python -m benchmarks.coverage.run --output results.json

Workstream 3: Template Coverage Analyzer
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.coverage.analyzer import TemplateCoverageAnalyzer
from benchmarks.coverage.datasets import generate_scenarios


def main() -> int:
    """Run template coverage analysis.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Run template coverage analysis benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress and detailed output",
    )
    args = parser.parse_args()

    if args.verbose:
        print("Initializing analyzer...")

    # Run analysis
    analyzer = TemplateCoverageAnalyzer()
    scenarios = generate_scenarios()

    if args.verbose:
        print(f"Analyzing {len(scenarios)} scenarios...")

    result = analyzer.analyze_dataset(scenarios)

    # Prepare output data
    output_data = {
        "total_queries": result.total_queries,
        "coverage_at_50": result.coverage_at_50,
        "coverage_at_70": result.coverage_at_70,
        "coverage_at_90": result.coverage_at_90,
        "unmatched_examples": result.unmatched_examples[:10],  # Sample
        "template_usage": result.template_usage,
        "timestamp": result.timestamp,
    }

    # Save results
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print(f"Coverage@0.5: {result.coverage_at_50:.1%}")
    print(f"Coverage@0.7: {result.coverage_at_70:.1%}")
    print(f"Coverage@0.9: {result.coverage_at_90:.1%}")
    print(f"Results saved to: {args.output}")

    if args.verbose:
        print(f"\nUnmatched examples (sample):")
        for example in result.unmatched_examples[:5]:
            print(f"  - {example[:60]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
