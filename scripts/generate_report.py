#!/usr/bin/env python3
"""Generate BENCHMARKS.md from benchmark results.

Usage:
    python scripts/generate_report.py --results-dir results/20240101_120000

Reads JSON results from memory, hallucination, coverage, and latency benchmarks
and produces a formatted Markdown report with tables and gate status.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json_safe(path: Path) -> dict[str, Any] | None:
    """Load JSON file, returning None if not found or invalid."""
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except json.JSONDecodeError:
        return None


def format_gate_status(
    value: float, thresholds: tuple[float, float], higher_is_better: bool = True
) -> str:
    """Format a value with gate status indicator."""
    pass_threshold, conditional_threshold = thresholds
    if higher_is_better:
        if value >= pass_threshold:
            return "PASS"
        elif value >= conditional_threshold:
            return "CONDITIONAL"
        else:
            return "FAIL"
    else:
        if value < pass_threshold:
            return "PASS"
        elif value < conditional_threshold:
            return "CONDITIONAL"
        else:
            return "FAIL"


def generate_report(results_dir: Path) -> str:
    """Generate Markdown report from benchmark results."""
    lines: list[str] = []

    # Header
    lines.append("# JARVIS Benchmark Results")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results Directory**: `{results_dir}`")
    lines.append("")

    # Gate Summary
    lines.append("## Gate Summary")
    lines.append("")
    lines.append("| Gate | Metric | Value | Threshold | Status |")
    lines.append("|------|--------|-------|-----------|--------|")

    gate_statuses: list[str] = []

    # G1: Coverage
    coverage = load_json_safe(results_dir / "coverage.json")
    if coverage and coverage.get("skipped"):
        gate_statuses.append("SKIP")
        reason = coverage.get("reason", "skipped")
        lines.append(f"| G1 | Template Coverage @0.7 | {reason} | >=60% | SKIP |")
    elif coverage and "coverage_at_70" in coverage:
        cov_70 = coverage["coverage_at_70"]
        status = format_gate_status(cov_70, (0.60, 0.40), higher_is_better=True)
        gate_statuses.append(status)
        lines.append(f"| G1 | Template Coverage @0.7 | {cov_70:.1%} | >=60% | {status} |")
    else:
        gate_statuses.append("SKIP")
        lines.append("| G1 | Template Coverage @0.7 | N/A | >=60% | SKIP |")

    # G2: Memory
    memory = load_json_safe(results_dir / "memory.json")
    if memory and memory.get("skipped"):
        gate_statuses.append("SKIP")
        reason = memory.get("reason", "skipped")
        lines.append(f"| G2 | Model Stack Memory | {reason} | <5.5GB | SKIP |")
    elif memory and "profiles" in memory:
        total_mb = sum(p.get("rss_mb", 0) for p in memory["profiles"])
        status = format_gate_status(total_mb, (5500, 6500), higher_is_better=False)
        gate_statuses.append(status)
        lines.append(f"| G2 | Model Stack Memory | {total_mb:.0f}MB | <5.5GB | {status} |")
    else:
        gate_statuses.append("SKIP")
        lines.append("| G2 | Model Stack Memory | N/A | <5.5GB | SKIP |")

    # G3: HHEM
    hhem = load_json_safe(results_dir / "hhem.json")
    if hhem and hhem.get("skipped"):
        gate_statuses.append("SKIP")
        reason = hhem.get("reason", "skipped")
        lines.append(f"| G3 | Mean HHEM Score | {reason} | >=0.5 | SKIP |")
    elif hhem and "mean_score" in hhem:
        mean_score = hhem["mean_score"]
        status = format_gate_status(mean_score, (0.5, 0.4), higher_is_better=True)
        gate_statuses.append(status)
        lines.append(f"| G3 | Mean HHEM Score | {mean_score:.3f} | >=0.5 | {status} |")
    else:
        gate_statuses.append("SKIP")
        lines.append("| G3 | Mean HHEM Score | N/A | >=0.5 | SKIP |")

    # G4 & G5: Latency
    latency = load_json_safe(results_dir / "latency.json")
    if latency and latency.get("skipped"):
        reason = latency.get("reason", "skipped")
        gate_statuses.extend(["SKIP", "SKIP"])
        lines.append(f"| G4 | Warm-Start Latency (p95) | {reason} | <3s | SKIP |")
        lines.append(f"| G5 | Cold-Start Latency (p95) | {reason} | <15s | SKIP |")
    elif latency and "results" in latency:
        warm_results = [r for r in latency["results"] if r.get("scenario") == "warm"]
        if warm_results:
            warm_p95 = warm_results[0].get("p95_ms", 0)
            status = format_gate_status(warm_p95, (3000, 5000), higher_is_better=False)
            gate_statuses.append(status)
            lines.append(f"| G4 | Warm-Start Latency (p95) | {warm_p95:.0f}ms | <3s | {status} |")
        else:
            gate_statuses.append("SKIP")
            lines.append("| G4 | Warm-Start Latency (p95) | N/A | <3s | SKIP |")

        cold_results = [r for r in latency["results"] if r.get("scenario") == "cold"]
        if cold_results:
            cold_p95 = cold_results[0].get("p95_ms", 0)
            status = format_gate_status(cold_p95, (15000, 20000), higher_is_better=False)
            gate_statuses.append(status)
            lines.append(f"| G5 | Cold-Start Latency (p95) | {cold_p95:.0f}ms | <15s | {status} |")
        else:
            gate_statuses.append("SKIP")
            lines.append("| G5 | Cold-Start Latency (p95) | N/A | <15s | SKIP |")
    else:
        gate_statuses.extend(["SKIP", "SKIP"])
        lines.append("| G4 | Warm-Start Latency (p95) | N/A | <3s | SKIP |")
        lines.append("| G5 | Cold-Start Latency (p95) | N/A | <15s | SKIP |")

    lines.append("")

    # Overall recommendation
    fail_count = gate_statuses.count("FAIL")
    if fail_count >= 2:
        recommendation = "Consider project cancellation (2+ failures)"
    elif fail_count == 1:
        recommendation = "Stop and reassess (1 failure)"
    else:
        recommendation = "Proceed with development"

    lines.append(f"**Recommendation**: {recommendation}")
    lines.append("")

    # Detailed sections
    lines.append("---")
    lines.append("")

    # Coverage Details
    lines.append("## Template Coverage (G1)")
    lines.append("")
    if coverage:
        lines.append(f"- **Total Queries**: {coverage.get('total_queries', 'N/A')}")
        lines.append(f"- **Coverage @0.5**: {coverage.get('coverage_at_50', 0):.1%}")
        lines.append(f"- **Coverage @0.7**: {coverage.get('coverage_at_70', 0):.1%}")
        lines.append(f"- **Coverage @0.9**: {coverage.get('coverage_at_90', 0):.1%}")

        if coverage.get("template_usage"):
            lines.append("")
            lines.append("### Template Usage")
            lines.append("")
            lines.append("| Template | Matches |")
            lines.append("|----------|---------|")
            # Sort by usage, show top 10
            usage = sorted(coverage["template_usage"].items(), key=lambda x: x[1], reverse=True)
            for template, count in usage[:10]:
                lines.append(f"| {template[:50]}{'...' if len(template) > 50 else ''} | {count} |")

        if coverage.get("unmatched_examples"):
            lines.append("")
            lines.append("### Unmatched Query Examples")
            lines.append("")
            for example in coverage["unmatched_examples"][:5]:
                lines.append(f"- `{example[:80]}{'...' if len(example) > 80 else ''}`")
    else:
        lines.append("*No coverage results available.*")
    lines.append("")

    # Memory Details
    lines.append("## Memory Profile (G2)")
    lines.append("")
    if memory and memory.get("profiles"):
        lines.append("| Component | RSS (MB) | Peak (MB) |")
        lines.append("|-----------|----------|-----------|")
        for profile in memory["profiles"]:
            name = profile.get("name", "Unknown")
            rss = profile.get("rss_mb", 0)
            peak = profile.get("peak_mb", rss)
            lines.append(f"| {name} | {rss:.0f} | {peak:.0f} |")

        total_rss = sum(p.get("rss_mb", 0) for p in memory["profiles"])
        lines.append(f"| **Total** | **{total_rss:.0f}** | - |")
    else:
        lines.append("*No memory profiling results available.*")
    lines.append("")

    # HHEM Details
    lines.append("## Hallucination Evaluation (G3)")
    lines.append("")
    if hhem:
        lines.append(f"- **Mean Score**: {hhem.get('mean_score', 0):.3f}")
        lines.append(f"- **Total Samples**: {hhem.get('total_samples', 'N/A')}")

        if hhem.get("category_scores"):
            lines.append("")
            lines.append("### Scores by Category")
            lines.append("")
            lines.append("| Category | Score |")
            lines.append("|----------|-------|")
            for category, score in hhem["category_scores"].items():
                lines.append(f"| {category} | {score:.3f} |")
    else:
        lines.append("*No hallucination evaluation results available.*")
    lines.append("")

    # Latency Details
    lines.append("## Latency Benchmarks (G4, G5)")
    lines.append("")
    if latency and latency.get("results"):
        lines.append("| Scenario | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |")
        lines.append("|----------|-----------|----------|----------|----------|")
        for result in latency["results"]:
            scenario = result.get("scenario", "Unknown")
            mean = result.get("mean_ms", 0)
            p50 = result.get("p50_ms", 0)
            p95 = result.get("p95_ms", 0)
            p99 = result.get("p99_ms", 0)
            lines.append(f"| {scenario} | {mean:.0f} | {p50:.0f} | {p95:.0f} | {p99:.0f} |")
    else:
        lines.append("*No latency benchmark results available.*")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `scripts/generate_report.py`*")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate BENCHMARKS.md from benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark result JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/BENCHMARKS.md"),
        help="Output Markdown file (default: docs/BENCHMARKS.md)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    # Generate report
    report = generate_report(args.results_dir)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    args.output.write_text(report)
    print(f"Report generated: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
