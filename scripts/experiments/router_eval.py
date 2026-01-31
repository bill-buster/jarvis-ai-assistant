#!/usr/bin/env python3
"""Router Pipeline Evaluation - 200 samples through full routing logic.

Tests the COMPLETE reply pipeline including:
- Clarify detection (context-dependent messages)
- Template matching (high similarity to past responses)
- Acknowledgment handling (ok, thanks, etc.)
- LLM generation (when needed)

Run: uv run python scripts/experiments/router_eval.py
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLES = 200
RESULTS_DIR = Path("results/router_eval")


@dataclass
class RouterEvalResult:
    """Result for a single router evaluation."""

    trigger: str
    actual_response: str
    router_response: str
    route_type: str  # 'template', 'generated', 'clarify', 'acknowledgment'
    similarity_to_actual: float
    route_time_ms: float
    confidence: str
    reason: str | None = None


def run_evaluation() -> None:
    """Run the router pipeline evaluation."""
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder
    from jarvis.router import ReplyRouter

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Router Pipeline Evaluation - 200 Samples")
    logger.info("=" * 60)

    # Load database
    logger.info("\n📂 Loading database...")
    db = get_db()
    all_pairs = db.get_all_pairs(min_quality=0.5)
    logger.info(f"   Total pairs: {len(all_pairs)}")

    if len(all_pairs) < SAMPLES:
        logger.error(f"Not enough pairs (need {SAMPLES})")
        return

    # Sample randomly
    random.seed(42)
    test_pairs = random.sample(all_pairs, SAMPLES)
    logger.info(f"   Test samples: {len(test_pairs)}")

    # Initialize components
    logger.info("\n🔗 Initializing embedder...")
    embedder = get_embedder()

    logger.info("\n🚦 Initializing router...")
    router = ReplyRouter()

    # Run evaluation
    logger.info("\n🏃 Running evaluation...")
    results: list[RouterEvalResult] = []
    total_time = 0
    route_counts: Counter = Counter()

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 20 == 0:
            avg_time = total_time / max(i, 1)
            logger.info(f"   Progress: {i + 1}/{SAMPLES} ({avg_time:.0f}ms avg)")

        trigger = pair.trigger_text
        actual = pair.response_text

        try:
            # Route through full pipeline
            start = time.time()
            route_result = router.route(trigger)
            route_time = (time.time() - start) * 1000
            total_time += route_time

            response = route_result.get("response", "")
            route_type = route_result.get("type", "unknown")
            confidence = route_result.get("confidence", "unknown")
            reason = route_result.get("reason")

            route_counts[route_type] += 1

            # Compute similarity to actual response
            if response and actual:
                embeddings = embedder.encode([actual, response], normalize=True)
                similarity = float(np.dot(embeddings[0], embeddings[1]))
            else:
                similarity = 0.0

            results.append(
                RouterEvalResult(
                    trigger=trigger[:100],
                    actual_response=actual[:100],
                    router_response=response[:100] if response else "",
                    route_type=route_type,
                    similarity_to_actual=similarity,
                    route_time_ms=route_time,
                    confidence=confidence,
                    reason=reason,
                )
            )

        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            results.append(
                RouterEvalResult(
                    trigger=trigger[:100],
                    actual_response=actual[:100],
                    router_response="",
                    route_type="error",
                    similarity_to_actual=0.0,
                    route_time_ms=0,
                    confidence="none",
                    reason=str(e),
                )
            )
            route_counts["error"] += 1

    # Compute statistics by route type
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS BY ROUTE TYPE")
    logger.info("=" * 60)

    for route_type in ["template", "generated", "clarify", "acknowledgment", "error"]:
        type_results = [r for r in results if r.route_type == route_type]
        if not type_results:
            continue

        count = len(type_results)
        pct = count / len(results) * 100
        avg_sim = np.mean([r.similarity_to_actual for r in type_results])
        avg_time = np.mean([r.route_time_ms for r in type_results])

        logger.info(f"\n{route_type.upper()} ({count} samples, {pct:.0f}%)")
        logger.info(f"  Avg similarity to actual: {avg_sim:.3f}")
        logger.info(f"  Avg response time:        {avg_time:.0f}ms")

        # Show examples
        logger.info("  Examples:")
        for r in type_results[:3]:
            logger.info(f"    Trigger: {r.trigger[:50]}...")
            logger.info(f"    Router:  {r.router_response[:50]}...")
            logger.info(f"    Actual:  {r.actual_response[:50]}...")
            logger.info(f"    Sim: {r.similarity_to_actual:.2f}")
            logger.info("")

    # Overall statistics
    logger.info("\n" + "=" * 60)
    logger.info("📈 OVERALL STATISTICS")
    logger.info("=" * 60)

    valid_results = [r for r in results if r.route_type != "error"]
    if valid_results:
        overall_sim = np.mean([r.similarity_to_actual for r in valid_results])
        overall_time = np.mean([r.route_time_ms for r in valid_results])

        logger.info("\nRoute Distribution:")
        for route_type, count in route_counts.most_common():
            pct = count / len(results) * 100
            logger.info(f"  {route_type:<15} {count:>4} ({pct:>5.1f}%)")

        logger.info("\nOverall Metrics:")
        logger.info(f"  Avg similarity to actual: {overall_sim:.3f}")
        logger.info(f"  Avg response time:        {overall_time:.0f}ms")
        logger.info(f"  Total time:               {total_time / 1000:.1f}s")

        # Breakdown: How good is each route type at matching actual?
        logger.info("\nSimilarity by Route Type:")
        for route_type in ["template", "generated", "clarify", "acknowledgment"]:
            type_results = [r for r in results if r.route_type == route_type]
            if type_results:
                avg = np.mean([r.similarity_to_actual for r in type_results])
                logger.info(f"  {route_type:<15} {avg:.3f}")

    # Save results
    output_file = RESULTS_DIR / f"router_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "samples": SAMPLES,
                "route_distribution": dict(route_counts),
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    logger.info(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
