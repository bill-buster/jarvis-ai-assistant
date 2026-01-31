#!/usr/bin/env python3
"""Validate routing metrics accuracy by cross-referencing audit log with database.

Usage:
    python -m scripts.validate_metrics --check-completeness
    python -m scripts.validate_metrics --audit-stats
    python -m scripts.validate_metrics --cross-reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.metrics_validation import get_audit_logger


def check_completeness() -> int:
    """Check if all audit log entries have corresponding DB records."""
    logger = get_audit_logger()
    result = logger.cross_reference()

    print("=" * 60)
    print("METRICS COMPLETENESS CHECK")
    print("=" * 60)
    print(f"Total audit entries: {result['total']}")
    print(f"Missing from DB: {result['missing_count']}")
    print(f"Match rate: {result['match_rate']:.2%}")

    if result["missing_count"] > 0:
        print("\n⚠️  WARNING: Some audit entries missing from metrics DB!")
        print("\nFirst 5 missing entries:")
        for entry in result["missing_details"]:
            print(f"  - {entry['query_hash']} at {entry['timestamp']}")
        return 1
    else:
        print("\n✅ All audit entries found in metrics DB")
        return 0


def show_audit_stats() -> int:
    """Show statistics about the audit log."""
    logger = get_audit_logger()
    entries = logger._load_audit_entries()

    print("=" * 60)
    print("AUDIT LOG STATISTICS")
    print("=" * 60)
    print(f"Total entries: {len(entries)}")

    if not entries:
        print("No audit entries found")
        return 0

    # Count by decision type
    decisions = {}
    for entry in entries:
        decision = entry.get("routing_decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1

    print("\nBy routing decision:")
    for decision, count in sorted(decisions.items()):
        pct = count / len(entries) * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")

    # Check similarity score distribution
    scores = [e.get("similarity_score", 0) for e in entries]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage similarity score: {avg_score:.3f}")

    return 0


def cross_reference_detailed() -> int:
    """Detailed cross-reference analysis."""
    logger = get_audit_logger()
    result = logger.cross_reference()

    print("=" * 60)
    print("DETAILED CROSS-REFERENCE ANALYSIS")
    print("=" * 60)
    print(f"Audit entries: {result['total']}")
    print(f"Matched in DB: {result['total'] - result['missing_count']}")
    print(f"Missing: {result['missing_count']}")
    print(f"Match rate: {result['match_rate']:.2%}")

    if result["match_rate"] < 0.95:
        print("\n⚠️  Low match rate detected!")
        print("Possible causes:")
        print("  - Exceptions during metrics recording")
        print("  - SQLite write failures")
        print("  - Race conditions")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate routing metrics accuracy")
    parser.add_argument(
        "--check-completeness",
        action="store_true",
        help="Check if all audit entries exist in DB",
    )
    parser.add_argument(
        "--audit-stats",
        action="store_true",
        help="Show audit log statistics",
    )
    parser.add_argument(
        "--cross-reference",
        action="store_true",
        help="Detailed cross-reference analysis",
    )

    args = parser.parse_args()

    if not any([args.check_completeness, args.audit_stats, args.cross_reference]):
        # Default: run all checks
        ret = show_audit_stats()
        print()
        ret |= check_completeness()
        return ret

    ret = 0
    if args.audit_stats:
        ret |= show_audit_stats()
    if args.check_completeness:
        print()
        ret |= check_completeness()
    if args.cross_reference:
        print()
        ret |= cross_reference_detailed()

    return ret


if __name__ == "__main__":
    raise SystemExit(main())
