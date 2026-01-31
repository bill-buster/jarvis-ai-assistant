#!/usr/bin/env python3
"""Full Pipeline Evaluation - Tests with relationship context.

Tests the COMPLETE pipeline:
1. Get trigger from pairs
2. Look up contact and relationship
3. Use relationship context in generation
4. Compare to actual response

Run: uv run python scripts/experiments/full_pipeline_eval.py
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLES = 200
RESULTS_DIR = Path("results/full_pipeline_eval")


@dataclass
class PipelineEvalResult:
    """Result for a single evaluation."""

    trigger: str
    actual_response: str
    generated_response: str
    relationship: str
    contact_name: str
    similarity_to_actual: float
    generation_time_ms: float
    used_relationship_context: bool


def run_evaluation() -> None:
    """Run the full pipeline evaluation."""
    from contracts.models import GenerationRequest
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder
    from models import get_generator

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Full Pipeline Evaluation - With Relationship Context")
    logger.info("=" * 60)

    # Load components
    logger.info("\n📂 Loading database...")
    db = get_db()

    # Get pairs that have contacts linked
    logger.info("\n🔗 Getting pairs with contact info...")
    all_pairs = db.get_all_pairs(min_quality=0.5)
    pairs_with_contacts = [p for p in all_pairs if p.contact_id]
    logger.info(f"   Total pairs: {len(all_pairs)}")
    logger.info(f"   Pairs with contacts: {len(pairs_with_contacts)}")

    if len(pairs_with_contacts) < SAMPLES:
        logger.warning(f"Only {len(pairs_with_contacts)} pairs with contacts, using all")
        test_pairs = pairs_with_contacts
    else:
        random.seed(42)
        test_pairs = random.sample(pairs_with_contacts, SAMPLES)

    logger.info(f"   Test samples: {len(test_pairs)}")

    # Initialize components
    logger.info("\n🔗 Initializing embedder...")
    embedder = get_embedder()

    logger.info("\n🤖 Initializing generator...")
    generator = get_generator()

    # Build contact lookup
    contact_cache = {}

    # Run evaluation
    logger.info("\n🏃 Running evaluation...")
    results: list[PipelineEvalResult] = []
    results_by_relationship: dict[str, list[PipelineEvalResult]] = {}
    total_time = 0

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 20 == 0:
            avg_time = total_time / max(i, 1)
            logger.info(f"   Progress: {i + 1}/{len(test_pairs)} ({avg_time:.0f}ms avg)")

        trigger = pair.trigger_text
        actual = pair.response_text

        # Get contact info
        contact_id = pair.contact_id
        if contact_id not in contact_cache:
            contact = db.get_contact(contact_id)
            contact_cache[contact_id] = contact
        else:
            contact = contact_cache[contact_id]

        relationship = contact.relationship if contact else "unknown"
        contact_name = contact.display_name if contact else "Unknown"

        # Build prompt with relationship context
        if relationship and relationship != "unknown":
            prompt = f"You are texting a {relationship}. Reply naturally to this message: {trigger}"
            used_context = True
        else:
            prompt = f"Reply naturally to this message: {trigger}"
            used_context = False

        # Generate response
        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=[],
                max_tokens=100,
                temperature=0.7,
            )

            start = time.time()
            response = generator.generate(request)
            gen_time = (time.time() - start) * 1000
            total_time += gen_time

            generated = response.text

            # Compute similarity
            embeddings = embedder.encode([actual, generated], normalize=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))

        except Exception as e:
            logger.error(f"Error: {e}")
            generated = ""
            similarity = 0
            gen_time = 0

        result = PipelineEvalResult(
            trigger=trigger[:100],
            actual_response=actual[:100],
            generated_response=generated[:100] if generated else "",
            relationship=relationship,
            contact_name=contact_name[:30] if contact_name else "",
            similarity_to_actual=similarity,
            generation_time_ms=gen_time,
            used_relationship_context=used_context,
        )
        results.append(result)

        # Group by relationship
        if relationship not in results_by_relationship:
            results_by_relationship[relationship] = []
        results_by_relationship[relationship].append(result)

    # Print results by relationship
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS BY RELATIONSHIP")
    logger.info("=" * 60)

    logger.info(f"\n{'Relationship':<18} {'Count':>6} {'Similarity':>12} {'Time':>10}")
    logger.info("-" * 50)

    for rel in sorted(results_by_relationship.keys()):
        rel_results = results_by_relationship[rel]
        count = len(rel_results)
        avg_sim = np.mean([r.similarity_to_actual for r in rel_results])
        avg_time = np.mean([r.generation_time_ms for r in rel_results])
        logger.info(f"{rel:<18} {count:>6} {avg_sim:>12.3f} {avg_time:>10.0f}ms")

    # Show examples for each relationship
    logger.info("\n" + "=" * 60)
    logger.info("📝 EXAMPLES BY RELATIONSHIP")
    logger.info("=" * 60)

    for rel in ["family", "close friend", "coworker", "romantic partner"]:
        if rel not in results_by_relationship:
            continue

        rel_results = results_by_relationship[rel]
        logger.info(f"\n{rel.upper()} ({len(rel_results)} samples)")

        # Show best example
        best = max(rel_results, key=lambda r: r.similarity_to_actual)
        logger.info(f"  Best (sim={best.similarity_to_actual:.2f}):")
        logger.info(f"    Trigger:   {best.trigger[:60]}...")
        logger.info(f"    Generated: {best.generated_response[:60]}...")
        logger.info(f"    Actual:    {best.actual_response[:60]}...")

        # Show worst example
        worst = min(rel_results, key=lambda r: r.similarity_to_actual)
        logger.info(f"  Worst (sim={worst.similarity_to_actual:.2f}):")
        logger.info(f"    Trigger:   {worst.trigger[:60]}...")
        logger.info(f"    Generated: {worst.generated_response[:60]}...")
        logger.info(f"    Actual:    {worst.actual_response[:60]}...")

    # Overall stats
    logger.info("\n" + "=" * 60)
    logger.info("📈 OVERALL")
    logger.info("=" * 60)

    overall_sim = np.mean([r.similarity_to_actual for r in results])
    overall_time = np.mean([r.generation_time_ms for r in results])

    logger.info(f"\nOverall similarity: {overall_sim:.3f}")
    logger.info(f"Avg generation time: {overall_time:.0f}ms")
    logger.info(f"Total time: {total_time / 1000:.1f}s")

    # Relationship distribution
    logger.info("\nRelationship distribution:")
    for rel, rel_results in sorted(
        results_by_relationship.items(), key=lambda x: len(x[1]), reverse=True
    ):
        pct = len(rel_results) / len(results) * 100
        logger.info(f"  {rel:<18} {len(rel_results):>4} ({pct:>5.1f}%)")

    # Save results
    output_file = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "samples": len(results),
                "overall_similarity": overall_sim,
                "overall_time_ms": overall_time,
                "by_relationship": {
                    rel: {
                        "count": len(rel_results),
                        "avg_similarity": np.mean([r.similarity_to_actual for r in rel_results]),
                        "avg_time_ms": np.mean([r.generation_time_ms for r in rel_results]),
                    }
                    for rel, rel_results in results_by_relationship.items()
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    logger.info(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
