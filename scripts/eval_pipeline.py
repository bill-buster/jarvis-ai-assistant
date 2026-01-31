#!/usr/bin/env python3
"""Evaluation Pipeline - Test reply generation on held-out data.

Evaluates the router's performance on unseen conversation pairs by:
1. Loading holdout pairs (not used in FAISS index training)
2. Running each trigger through the router
3. Comparing generated response to actual response
4. Computing semantic similarity, length ratio, and type metrics

Usage:
    python -m scripts.eval_pipeline                    # Run evaluation
    python -m scripts.eval_pipeline --setup            # Create train/test split first
    python -m scripts.eval_pipeline --limit 50         # Evaluate first 50 pairs
    python -m scripts.eval_pipeline --verbose          # Show each pair's results
    python -m scripts.eval_pipeline --output results.json  # Save detailed results
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db import Pair, get_db
from jarvis.embedding_adapter import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result for a single evaluation pair."""

    pair_id: int
    contact_id: int | None
    trigger_text: str
    actual_response: str
    generated_response: str
    route_type: str  # template, generated, clarify, acknowledgment
    confidence: str
    similarity_score: float  # FAISS similarity to training data
    semantic_similarity: float  # similarity between actual and generated
    length_ratio: float  # len(generated) / len(actual)
    latency_ms: float


@dataclass
class EvalSummary:
    """Summary statistics for the evaluation run."""

    total_pairs: int
    route_distribution: dict[str, int]  # type -> count
    avg_semantic_similarity: float
    avg_length_ratio: float
    avg_latency_ms: float
    similarity_by_type: dict[str, float]  # type -> avg similarity
    pairs_above_threshold: int  # pairs with semantic_similarity >= 0.5
    threshold_rate: float  # pairs_above_threshold / total


def compute_semantic_similarity(text1: str, text2: str, embedder: Any) -> float:
    """Compute semantic similarity between two texts.

    Args:
        text1: First text.
        text2: Second text.
        embedder: Embedding model.

    Returns:
        Cosine similarity between embeddings (0 to 1).
    """
    if not text1 or not text2:
        return 0.0

    try:
        embeddings = embedder.encode([text1, text2], normalize=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    except Exception as e:
        logger.warning("Failed to compute similarity: %s", e)
        return 0.0


def evaluate_pair(
    pair: Pair,
    router: Any,
    embedder: Any,
    context: list[str] | None = None,
    use_live_context: bool = False,
) -> EvalResult:
    """Evaluate a single pair through the router.

    Args:
        pair: The holdout pair to evaluate.
        router: The ReplyRouter instance.
        embedder: Embedding model for similarity computation.
        context: Optional conversation context.
        use_live_context: If False (default), don't fetch live iMessage context.
            This prevents context contamination from current messages.

    Returns:
        EvalResult with metrics.
    """
    import time

    # Use stored context if available, otherwise empty (don't fetch live)
    eval_context = context
    if eval_context is None and pair.context_text:
        # Use the stored context from when the pair was extracted
        eval_context = [pair.context_text]
    elif eval_context is None and not use_live_context:
        # Don't pass chat_id to prevent live context fetching
        eval_context = []

    # Run through router
    start = time.perf_counter()
    try:
        result = router.route(
            incoming=pair.trigger_text,
            contact_id=pair.contact_id,
            thread=eval_context if eval_context else None,
            chat_id=pair.chat_id if use_live_context else None,  # Don't fetch live context
        )
        generated = result.get("response", "")
        route_type = result.get("type", "unknown")
        confidence = result.get("confidence", "unknown")
        faiss_similarity = result.get("similarity_score", 0.0)
    except Exception as e:
        logger.warning("Router failed for pair %d: %s", pair.id, e)
        generated = ""
        route_type = "error"
        confidence = "none"
        faiss_similarity = 0.0

    latency_ms = (time.perf_counter() - start) * 1000

    # Compute semantic similarity between generated and actual
    semantic_sim = compute_semantic_similarity(pair.response_text, generated, embedder)

    # Compute length ratio
    actual_len = len(pair.response_text)
    generated_len = len(generated) if generated else 0
    length_ratio = generated_len / actual_len if actual_len > 0 else 0.0

    return EvalResult(
        pair_id=pair.id or 0,
        contact_id=pair.contact_id,
        trigger_text=pair.trigger_text,
        actual_response=pair.response_text,
        generated_response=generated,
        route_type=route_type,
        confidence=confidence,
        similarity_score=faiss_similarity,
        semantic_similarity=semantic_sim,
        length_ratio=length_ratio,
        latency_ms=latency_ms,
    )


def summarize_results(results: list[EvalResult]) -> EvalSummary:
    """Compute summary statistics from evaluation results.

    Args:
        results: List of individual evaluation results.

    Returns:
        Summary statistics.
    """
    if not results:
        return EvalSummary(
            total_pairs=0,
            route_distribution={},
            avg_semantic_similarity=0.0,
            avg_length_ratio=0.0,
            avg_latency_ms=0.0,
            similarity_by_type={},
            pairs_above_threshold=0,
            threshold_rate=0.0,
        )

    # Route distribution
    route_dist: dict[str, int] = {}
    for r in results:
        route_dist[r.route_type] = route_dist.get(r.route_type, 0) + 1

    # Averages
    avg_sim = sum(r.semantic_similarity for r in results) / len(results)
    avg_len = sum(r.length_ratio for r in results) / len(results)
    avg_lat = sum(r.latency_ms for r in results) / len(results)

    # Similarity by type
    sim_by_type: dict[str, list[float]] = {}
    for r in results:
        if r.route_type not in sim_by_type:
            sim_by_type[r.route_type] = []
        sim_by_type[r.route_type].append(r.semantic_similarity)

    avg_sim_by_type = {t: sum(sims) / len(sims) for t, sims in sim_by_type.items()}

    # Threshold rate (pairs with >= 0.5 similarity)
    above_threshold = sum(1 for r in results if r.semantic_similarity >= 0.5)

    return EvalSummary(
        total_pairs=len(results),
        route_distribution=route_dist,
        avg_semantic_similarity=avg_sim,
        avg_length_ratio=avg_len,
        avg_latency_ms=avg_lat,
        similarity_by_type=avg_sim_by_type,
        pairs_above_threshold=above_threshold,
        threshold_rate=above_threshold / len(results),
    )


def run_evaluation(
    limit: int | None = None,
    verbose: bool = False,
    min_quality: float = 0.5,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the full evaluation pipeline.

    Args:
        limit: Maximum number of pairs to evaluate.
        verbose: Print results for each pair.
        min_quality: Minimum quality score for pairs.

    Returns:
        Tuple of (individual results, summary).
    """
    from jarvis.router import ReplyRouter

    # Initialize components
    db = get_db()
    db.init_schema()

    # Check split status
    split_stats = db.get_split_stats()
    if split_stats["holdout_pairs"] == 0:
        logger.error("No holdout pairs found. Run with --setup first.")
        return [], EvalSummary(
            total_pairs=0,
            route_distribution={},
            avg_semantic_similarity=0.0,
            avg_length_ratio=0.0,
            avg_latency_ms=0.0,
            similarity_by_type={},
            pairs_above_threshold=0,
            threshold_rate=0.0,
        )

    logger.info(
        "Split stats: %d training, %d holdout (%.1f%%)",
        split_stats["training_pairs"],
        split_stats["holdout_pairs"],
        split_stats["holdout_ratio"] * 100,
    )

    # Get holdout pairs
    holdout_pairs = db.get_holdout_pairs(min_quality=min_quality)
    if limit:
        holdout_pairs = holdout_pairs[:limit]

    logger.info("Evaluating %d holdout pairs", len(holdout_pairs))

    # Initialize router (will use training-only index)
    router = ReplyRouter(db=db)

    # Initialize embedder for similarity computation
    embedder = get_embedder()

    # Evaluate each pair
    results = []
    for i, pair in enumerate(holdout_pairs):
        if verbose:
            print(f"\n--- Pair {i + 1}/{len(holdout_pairs)} ---")
            print(f"Trigger: {pair.trigger_text[:100]}...")

        result = evaluate_pair(pair, router, embedder)
        results.append(result)

        if verbose:
            print(f"Actual: {pair.response_text[:100]}...")
            print(f"Generated ({result.route_type}): {result.generated_response[:100]}...")
            print(f"Semantic similarity: {result.semantic_similarity:.3f}")
            print(f"Latency: {result.latency_ms:.1f}ms")

        # Progress update every 10 pairs
        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d pairs", i + 1, len(holdout_pairs))

    # Compute summary
    summary = summarize_results(results)

    return results, summary


def setup_split(
    holdout_ratio: float = 0.2,
    min_pairs: int = 5,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Create train/test split in the database.

    Args:
        holdout_ratio: Fraction of contacts to hold out.
        min_pairs: Minimum pairs per contact to be eligible.
        seed: Random seed for reproducibility.

    Returns:
        Split statistics.
    """
    db = get_db()
    db.init_schema()

    result = db.split_train_test(
        holdout_ratio=holdout_ratio,
        min_pairs_per_contact=min_pairs,
        seed=seed,
    )

    if result["success"]:
        logger.info(
            "Created split: %d training / %d holdout contacts",
            result["contacts_training"],
            result["contacts_holdout"],
        )
        logger.info(
            "Pairs: %d training / %d holdout (%.1f%%)",
            result["pairs_training"],
            result["pairs_holdout"],
            result["holdout_ratio_actual"] * 100,
        )
    else:
        logger.error("Split failed: %s", result.get("error", "unknown"))

    return result


def rebuild_index() -> dict[str, Any]:
    """Rebuild FAISS index excluding holdout pairs.

    Returns:
        Index build statistics.
    """
    from jarvis.index import build_index_from_db

    db = get_db()
    db.init_schema()

    logger.info("Rebuilding index with training pairs only...")
    result = build_index_from_db(
        jarvis_db=db,
        min_quality=0.5,
        include_holdout=False,  # Exclude holdout pairs
    )

    if result["success"]:
        logger.info(
            "Index built: %d pairs indexed (version %s)",
            result["pairs_indexed"],
            result["version_id"],
        )
    else:
        logger.error("Index build failed: %s", result.get("error", "unknown"))

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate reply generation on held-out data")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create train/test split (run this first)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FAISS index excluding holdout pairs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show results for each pair",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of contacts to hold out (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Handle setup
    if args.setup:
        result = setup_split(
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
        )
        if result["success"]:
            print("\nSplit created successfully!")
            print(
                f"Training: {result['pairs_training']} pairs from "
                f"{result['contacts_training']} contacts"
            )
            print(
                f"Holdout:  {result['pairs_holdout']} pairs from "
                f"{result['contacts_holdout']} contacts"
            )
            print("\nNext steps:")
            print("  1. Rebuild the index: python -m scripts.eval_pipeline --rebuild-index")
            print("  2. Run evaluation:    python -m scripts.eval_pipeline")
        else:
            print(f"\nSetup failed: {result.get('error', 'unknown')}")
            sys.exit(1)
        return

    # Handle index rebuild
    if args.rebuild_index:
        result = rebuild_index()
        if result["success"]:
            print(f"\nIndex rebuilt with {result['pairs_indexed']} training pairs")
            print(f"Version: {result['version_id']}")
            print("\nNow run evaluation: python -m scripts.eval_pipeline")
        else:
            print(f"\nIndex build failed: {result.get('error', 'unknown')}")
            sys.exit(1)
        return

    # Run evaluation
    print("Running evaluation pipeline...")
    results, summary = run_evaluation(
        limit=args.limit,
        verbose=args.verbose,
    )

    if not results:
        print("\nNo results. Make sure to run --setup first.")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs evaluated: {summary.total_pairs}")
    print("\nRoute distribution:")
    for route_type, count in sorted(summary.route_distribution.items()):
        pct = count / summary.total_pairs * 100
        print(f"  {route_type:15s}: {count:4d} ({pct:5.1f}%)")

    print("\nOverall metrics:")
    print(f"  Avg semantic similarity: {summary.avg_semantic_similarity:.3f}")
    print(f"  Avg length ratio:        {summary.avg_length_ratio:.3f}")
    print(f"  Avg latency:             {summary.avg_latency_ms:.1f}ms")

    print("\nSimilarity by route type:")
    for route_type, sim in sorted(summary.similarity_by_type.items()):
        print(f"  {route_type:15s}: {sim:.3f}")

    print("\nQuality threshold (>= 0.5 similarity):")
    print(
        f"  {summary.pairs_above_threshold}/{summary.total_pairs} pairs "
        f"({summary.threshold_rate * 100:.1f}%)"
    )

    # Always save detailed results (auto-generate filename if not provided)
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-save to results/eval_pipeline/ with timestamp
        results_dir = Path("results/eval_pipeline")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"eval_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
