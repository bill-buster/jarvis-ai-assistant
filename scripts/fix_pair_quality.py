#!/usr/bin/env python3
"""Fix pair quality and rebuild training data.

This script:
1. Re-scores all pairs with stricter topic shift detection
2. Filters training data to only GOOD quality pairs (>= 0.6)
3. Rebuilds FAISS index with filtered pairs
4. Updates acknowledgment classifier thresholds

Usage:
    python -m scripts.fix_pair_quality --full          # Run full fix
    python -m scripts.fix_pair_quality --rescore-only  # Just rescore pairs
    python -m scripts.fix_pair_quality --rebuild-only  # Just rebuild index
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db import get_db
from jarvis.embedding_adapter import get_embedder
from jarvis.index import build_index_from_db

logger = logging.getLogger(__name__)


def rescore_all_pairs(min_quality_threshold: float = 0.6):
    """Re-score all pairs and mark low-quality ones.

    This uses stricter criteria than the original extraction:
    - Semantic coherence >= 0.6 (not 0.45)
    - No topic shifts (even with penalties)
    - No acknowledgment + substantive response patterns
    """
    db = get_db()
    embedder = get_embedder()

    pairs = db.get_all_pairs(min_quality=0.0)

    good_count = 0
    filtered_count = 0

    print(f"Rescoring {len(pairs)} pairs with threshold {min_quality_threshold}...")

    with db.connection() as conn:
        for i, pair in enumerate(pairs):
            # Use stricter scoring
            from scripts.score_pair_quality import compute_coherence_score

            score = compute_coherence_score(
                pair.trigger_text, pair.response_text, pair.context_text, embedder
            )

            # Stricter verdict: only GOOD if >= 0.6 and no topic shift flags
            new_quality = score["coherence_score"]

            # Mark for exclusion if:
            # - Quality < threshold
            # - Topic shift detected
            # - Acknowledgment trigger with low similarity
            should_exclude = (
                new_quality < min_quality_threshold
                or score.get("is_new_topic", False)
                or (score.get("is_ack_trigger", False) and score["trigger_response_sim"] < 0.55)
            )

            if should_exclude:
                # Mark as excluded by setting is_holdout=True (hack: use holdout flag)
                # Better: add is_excluded column or set quality to 0
                conn.execute(
                    "UPDATE pairs SET quality_score = ? WHERE id = ?",
                    (0.1, pair.id),  # Mark as bad quality
                )
                filtered_count += 1
            else:
                conn.execute(
                    "UPDATE pairs SET quality_score = ? WHERE id = ?", (new_quality, pair.id)
                )
                good_count += 1

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(pairs)} pairs")

    print("\n=== RESCORING COMPLETE ===")
    print(f"  GOOD pairs (>= {min_quality_threshold}): {good_count}")
    print(f"  EXCLUDED pairs: {filtered_count}")
    print(f"  Filtered out: {filtered_count / len(pairs) * 100:.1f}%")


def rebuild_index_with_quality_filter(min_quality: float = 0.6):
    """Rebuild FAISS index excluding low-quality pairs."""
    print("\n=== REBUILDING INDEX ===")
    print(f"  Minimum quality: {min_quality}")

    # Build index with quality filter
    stats = build_index_from_db(min_quality=min_quality, include_holdout=False)

    print(f"\n  Pairs indexed: {stats.get('pairs_indexed', 'unknown')}")
    print(f"  Index file: {stats.get('index_path', 'unknown')}")


def tune_acknowledgment_classifier():
    """Tune acknowledgment classifier to reduce over-classification."""
    print("\n=== TUNING ACKNOWLEDGMENT CLASSIFIER ===")
    print("  To reduce over-classification:")
    print("  1. Remove emotional expressions from acknowledgment list")
    print("  2. Add context check before classifying as acknowledgment")
    print("  3. Lower threshold for generating after acknowledgment")

    # Read current router code
    router_path = Path(__file__).parent.parent / "jarvis" / "router.py"
    content = router_path.read_text()

    # Check current acknowledgment patterns
    if "bruh" in content.lower() or "fuck" in content.lower():
        print("  WARNING: Emotional expressions may be in acknowledgment list")

    print("\n  Manual fix needed in jarvis/router.py:")
    print("  - _is_simple_acknowledgment() method")
    print("  - _should_generate_after_acknowledgment() threshold")


def main():
    parser = argparse.ArgumentParser(description="Fix pair quality issues")
    parser.add_argument("--full", action="store_true", help="Run full fix pipeline")
    parser.add_argument("--rescore-only", action="store_true", help="Only rescore pairs")
    parser.add_argument("--rebuild-only", action="store_true", help="Only rebuild index")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Quality threshold (default: 0.6)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.full:
        print("=== RUNNING FULL QUALITY FIX ===\n")
        rescore_all_pairs(args.threshold)
        rebuild_index_with_quality_filter(args.threshold)
        tune_acknowledgment_classifier()
        print("\n=== FIX COMPLETE ===")
        print("Next steps:")
        print("  1. Run evaluation: python -m scripts.eval_pipeline --limit 100")
        print("  2. Check improvement in route distribution and similarity scores")
    elif args.rescore_only:
        rescore_all_pairs(args.threshold)
    elif args.rebuild_only:
        rebuild_index_with_quality_filter(args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
