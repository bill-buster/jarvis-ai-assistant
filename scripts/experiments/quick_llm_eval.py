#!/usr/bin/env python3
"""Quick 200-sample LLM evaluation.

Tests the reply generation pipeline on 200 random samples from your
extracted pairs, measuring:
- Response quality (brevity, naturalness)
- Semantic similarity to actual responses
- Time per generation

Run: uv run python scripts/experiments/quick_llm_eval.py
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
RESULTS_DIR = Path("results/quick_eval")


@dataclass
class EvalResult:
    """Result for a single evaluation."""

    trigger: str
    actual_response: str
    generated_response: str | None
    generation_time_ms: float
    similarity_to_actual: float
    brevity_score: float
    naturalness_score: float
    error: str | None = None


def score_brevity(response: str) -> float:
    """Score response brevity (5-30 words ideal)."""
    words = len(response.split())
    if words == 0:
        return 0.0
    if 5 <= words <= 30:
        return 1.0
    elif words < 5:
        return 0.5 + (words / 10)
    else:
        penalty = min(0.5, (words - 30) / 100)
        return max(0.3, 1.0 - penalty)


def score_naturalness(response: str) -> float:
    """Score naturalness (penalize assistant-like language)."""
    score = 1.0
    response_lower = response.lower()

    # Penalize assistant phrases
    assistant_phrases = [
        "i'm here to help",
        "let me know",
        "if you need anything",
        "how can i assist",
        "is there anything",
        "happy to help",
        "i understand",
        "that makes sense",
        "certainly",
        "absolutely",
    ]
    for phrase in assistant_phrases:
        if phrase in response_lower:
            score *= 0.8

    # Penalize formal phrases
    formal_phrases = ["furthermore", "additionally", "consequently", "therefore"]
    for phrase in formal_phrases:
        if phrase in response_lower:
            score *= 0.7

    return round(score, 3)


def run_evaluation() -> None:
    """Run the quick evaluation."""
    # Lazy imports
    from contracts.models import GenerationRequest
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder
    from models import get_generator

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Quick LLM Evaluation - 200 Samples")
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

    # Initialize embedder for similarity
    logger.info("\n🔗 Initializing embedder...")
    embedder = get_embedder()

    # Initialize generator
    logger.info("\n🤖 Initializing generator...")
    generator = get_generator()
    model_name = getattr(generator, "model_id", None) or getattr(generator, "_config", {})
    if hasattr(generator, "_loader") and hasattr(generator._loader, "config"):
        model_name = generator._loader.config.display_name
    logger.info(f"   Model: {model_name}")

    # Run evaluation
    logger.info("\n🏃 Running evaluation...")
    results: list[EvalResult] = []
    total_time = 0
    errors = 0

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 20 == 0:
            avg_time = total_time / max(i, 1)
            eta = avg_time * (SAMPLES - i - 1) / 1000
            logger.info(f"   Progress: {i + 1}/{SAMPLES} ({avg_time:.0f}ms avg, ETA: {eta:.0f}s)")

        trigger = pair.trigger_text
        actual = pair.response_text

        try:
            # Build generation request
            request = GenerationRequest(
                prompt=f"Reply to this message naturally: {trigger}",
                context_documents=[],
                few_shot_examples=[],
                max_tokens=100,
                temperature=0.7,
            )

            # Generate response
            start = time.time()
            response = generator.generate(request)
            gen_time = (time.time() - start) * 1000
            total_time += gen_time

            generated = response.text

            # Compute similarity
            embeddings = embedder.encode([actual, generated], normalize=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))

            # Score quality
            brevity = score_brevity(generated)
            naturalness = score_naturalness(generated)

            results.append(
                EvalResult(
                    trigger=trigger[:100],
                    actual_response=actual[:100],
                    generated_response=generated[:100],
                    generation_time_ms=gen_time,
                    similarity_to_actual=similarity,
                    brevity_score=brevity,
                    naturalness_score=naturalness,
                )
            )

        except Exception as e:
            errors += 1
            results.append(
                EvalResult(
                    trigger=trigger[:100],
                    actual_response=actual[:100],
                    generated_response=None,
                    generation_time_ms=0,
                    similarity_to_actual=0,
                    brevity_score=0,
                    naturalness_score=0,
                    error=str(e),
                )
            )

    # Compute statistics
    valid_results = [r for r in results if r.error is None]

    if not valid_results:
        logger.error("No valid results!")
        return

    avg_similarity = np.mean([r.similarity_to_actual for r in valid_results])
    avg_brevity = np.mean([r.brevity_score for r in valid_results])
    avg_naturalness = np.mean([r.naturalness_score for r in valid_results])
    avg_time = np.mean([r.generation_time_ms for r in valid_results])

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Samples evaluated: {len(valid_results)}/{SAMPLES}")
    logger.info(f"Errors: {errors}")
    logger.info("")
    logger.info("Quality Metrics:")
    logger.info(f"  Similarity to actual: {avg_similarity:.3f}")
    logger.info(f"  Brevity score:        {avg_brevity:.3f}")
    logger.info(f"  Naturalness score:    {avg_naturalness:.3f}")
    overall = (avg_similarity + avg_brevity + avg_naturalness) / 3
    logger.info(f"  Overall:              {overall:.3f}")
    logger.info("")
    logger.info("Performance:")
    logger.info(f"  Avg generation time:  {avg_time:.0f}ms")
    logger.info(f"  Total time:           {total_time / 1000:.1f}s")

    # Show some examples
    logger.info("\n" + "=" * 60)
    logger.info("📝 SAMPLE OUTPUTS")
    logger.info("=" * 60)

    # Best examples
    best = sorted(valid_results, key=lambda r: r.similarity_to_actual, reverse=True)[:5]
    logger.info("\n🟢 Best (highest similarity):")
    for r in best:
        logger.info(f"  Trigger: {r.trigger[:60]}...")
        logger.info(f"  Actual:  {r.actual_response[:60]}...")
        logger.info(f"  LLM:     {r.generated_response[:60]}...")
        logger.info(f"  Sim: {r.similarity_to_actual:.2f}")
        logger.info("")

    # Worst examples
    worst = sorted(valid_results, key=lambda r: r.similarity_to_actual)[:5]
    logger.info("🔴 Worst (lowest similarity):")
    for r in worst:
        logger.info(f"  Trigger: {r.trigger[:60]}...")
        logger.info(f"  Actual:  {r.actual_response[:60]}...")
        logger.info(f"  LLM:     {r.generated_response[:60]}...")
        logger.info(f"  Sim: {r.similarity_to_actual:.2f}")
        logger.info("")

    # Save results
    output_file = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "samples": SAMPLES,
                "valid": len(valid_results),
                "errors": errors,
                "metrics": {
                    "avg_similarity": avg_similarity,
                    "avg_brevity": avg_brevity,
                    "avg_naturalness": avg_naturalness,
                    "avg_time_ms": avg_time,
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    logger.info(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
