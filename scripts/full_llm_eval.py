#!/usr/bin/env python3
"""Full LLM Generation Evaluation.

Evaluates all queries through LLM generation (bypassing templates) to measure
the baseline quality of LLM responses against actual conversation history.

This evaluation helps understand:
1. How well the LLM captures the user's communication style
2. Semantic similarity between LLM responses and actual responses
3. Coherence scores for LLM-generated content
4. Areas where LLM generation excels or struggles

Usage:
    python scripts/full_llm_eval.py                    # Run with defaults
    python scripts/full_llm_eval.py --samples 500      # Custom sample size
    python scripts/full_llm_eval.py --output results/  # Custom output dir
    python scripts/full_llm_eval.py --verbose          # Detailed logging
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.db import JarvisDB, get_db
from jarvis.quality_metrics import score_response_coherence

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLES = 500
DEFAULT_OUTPUT_DIR = Path("results/full_llm_eval")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GenerationResult:
    """Result of generating an LLM response for a single trigger."""

    trigger: str
    actual_response: str
    llm_response: str | None
    similarity: float  # Semantic similarity between LLM and actual response
    coherence: float  # Coherence score for LLM response
    actual_coherence: float  # Coherence score for actual response
    generation_time_ms: float
    contact_id: int | None = None
    quality_score: float = 0.0  # Original pair quality score


@dataclass
class EvaluationSummary:
    """Summary statistics from the evaluation."""

    total_samples: int
    successful_generations: int
    failed_generations: int
    avg_similarity: float
    median_similarity: float
    std_similarity: float
    avg_coherence: float
    avg_actual_coherence: float
    avg_generation_time_ms: float
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Similarity distribution
    similarity_p10: float = 0.0
    similarity_p25: float = 0.0
    similarity_p50: float = 0.0
    similarity_p75: float = 0.0
    similarity_p90: float = 0.0

    # Coherence comparison
    llm_more_coherent_count: int = 0
    actual_more_coherent_count: int = 0
    equal_coherence_count: int = 0


@dataclass
class QualityBucket:
    """Statistics for a quality score bucket."""

    min_quality: float
    max_quality: float
    count: int
    avg_similarity: float
    avg_coherence: float


# =============================================================================
# Evaluation Logic
# =============================================================================


class FullLLMEvaluator:
    """Evaluator for measuring LLM generation quality against actual responses."""

    def __init__(
        self,
        db: JarvisDB | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        verbose: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            db: JarvisDB instance. Uses default if None.
            output_dir: Directory to write results.
            verbose: Enable detailed logging.
        """
        self.db = db or get_db()
        self.db.init_schema()
        self.output_dir = output_dir
        self.verbose = verbose

        self._generator = None
        self._sentence_model = None

    @property
    def generator(self) -> Any:
        """Get or create the MLX generator."""
        if self._generator is None:
            from models import get_generator

            self._generator = get_generator(skip_templates=True)
        return self._generator

    @property
    def sentence_model(self) -> Any:
        """Get or create the sentence transformer model."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._sentence_model

    def unload_models(self) -> None:
        """Unload models to free memory."""
        self._generator = None
        self._sentence_model = None
        gc.collect()

        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except ImportError:
            pass

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score (0-1).
        """
        embeddings = self.sentence_model.encode(
            [text1, text2],
            normalize_embeddings=True,
        )
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    def generate_response(
        self,
        trigger: str,
        contact_name: str = "them",
    ) -> str | None:
        """Generate LLM response for a trigger.

        Args:
            trigger: The incoming message.
            contact_name: Name of the contact.

        Returns:
            Generated response text or None.
        """
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        try:
            prompt = build_rag_reply_prompt(
                context=f"[Incoming]: {trigger}",
                last_message=trigger,
                contact_name=contact_name,
            )

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=[],
                max_tokens=150,
                temperature=0.7,
            )

            response = self.generator.generate(request)
            return response.text.strip()

        except Exception as e:
            logger.warning("Error generating LLM response: %s", e)
            return None

    def evaluate_single(
        self,
        trigger: str,
        actual_response: str,
        contact_id: int | None = None,
        quality_score: float = 0.0,
    ) -> GenerationResult:
        """Evaluate LLM generation for a single trigger.

        Args:
            trigger: The incoming message.
            actual_response: The actual response from the dataset.
            contact_id: Optional contact ID.
            quality_score: Original quality score of the pair.

        Returns:
            GenerationResult with evaluation details.
        """
        start_time = time.time()

        # Generate LLM response
        llm_response = self.generate_response(trigger)

        generation_time_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        if llm_response:
            similarity = self.compute_similarity(llm_response, actual_response)
            coherence = score_response_coherence(trigger, llm_response)
        else:
            similarity = 0.0
            coherence = 0.0

        actual_coherence = score_response_coherence(trigger, actual_response)

        return GenerationResult(
            trigger=trigger,
            actual_response=actual_response,
            llm_response=llm_response,
            similarity=similarity,
            coherence=coherence,
            actual_coherence=actual_coherence,
            generation_time_ms=generation_time_ms,
            contact_id=contact_id,
            quality_score=quality_score,
        )

    def run_evaluation(
        self,
        num_samples: int = DEFAULT_SAMPLES,
        min_quality: float = 0.5,
    ) -> tuple[EvaluationSummary | None, list[GenerationResult]]:
        """Run the full evaluation.

        Args:
            num_samples: Number of pairs to evaluate.
            min_quality: Minimum quality score for pairs to include.

        Returns:
            Tuple of (summary statistics, list of detailed results).
        """
        start_time = time.time()

        # Get pairs from database
        pairs = self.db.get_all_pairs(min_quality=min_quality)
        if not pairs:
            logger.error("No pairs found in database")
            return None, []

        # Sample pairs
        sample_size = min(num_samples, len(pairs))
        sampled_pairs = random.sample(pairs, sample_size)

        logger.info(
            "Evaluating %d pairs (from %d total, min_quality=%.2f)",
            sample_size,
            len(pairs),
            min_quality,
        )

        # Run evaluations
        results: list[GenerationResult] = []
        for i, pair in enumerate(sampled_pairs):
            if (i + 1) % 20 == 0 or self.verbose:
                logger.info("Progress: %d/%d", i + 1, sample_size)

            result = self.evaluate_single(
                trigger=pair.trigger_text,
                actual_response=pair.response_text,
                contact_id=pair.contact_id,
                quality_score=pair.quality_score,
            )
            results.append(result)

            # Log interesting cases
            if self.verbose:
                logger.info(
                    "  Trigger: %s... -> sim=%.2f, coh=%.2f",
                    pair.trigger_text[:40],
                    result.similarity,
                    result.coherence,
                )

        # Calculate summary statistics
        successful = [r for r in results if r.llm_response is not None]
        failed = [r for r in results if r.llm_response is None]

        similarities = [r.similarity for r in successful]
        coherences = [r.coherence for r in successful]
        actual_coherences = [r.actual_coherence for r in results]
        generation_times = [r.generation_time_ms for r in successful]

        total_time = time.time() - start_time

        # Count coherence comparisons
        llm_more_coherent = sum(
            1 for r in successful if r.coherence > r.actual_coherence
        )
        actual_more_coherent = sum(
            1 for r in successful if r.actual_coherence > r.coherence
        )
        equal_coherence = sum(
            1 for r in successful if abs(r.coherence - r.actual_coherence) < 0.01
        )

        if similarities:
            summary = EvaluationSummary(
                total_samples=len(results),
                successful_generations=len(successful),
                failed_generations=len(failed),
                avg_similarity=float(np.mean(similarities)),
                median_similarity=float(np.median(similarities)),
                std_similarity=float(np.std(similarities)),
                avg_coherence=float(np.mean(coherences)) if coherences else 0.0,
                avg_actual_coherence=float(np.mean(actual_coherences)),
                avg_generation_time_ms=(
                    float(np.mean(generation_times)) if generation_times else 0.0
                ),
                total_time_seconds=total_time,
                similarity_p10=float(np.percentile(similarities, 10)),
                similarity_p25=float(np.percentile(similarities, 25)),
                similarity_p50=float(np.percentile(similarities, 50)),
                similarity_p75=float(np.percentile(similarities, 75)),
                similarity_p90=float(np.percentile(similarities, 90)),
                llm_more_coherent_count=llm_more_coherent,
                actual_more_coherent_count=actual_more_coherent,
                equal_coherence_count=equal_coherence,
            )
        else:
            summary = EvaluationSummary(
                total_samples=len(results),
                successful_generations=0,
                failed_generations=len(results),
                avg_similarity=0.0,
                median_similarity=0.0,
                std_similarity=0.0,
                avg_coherence=0.0,
                avg_actual_coherence=(
                    float(np.mean(actual_coherences)) if actual_coherences else 0.0
                ),
                avg_generation_time_ms=0.0,
                total_time_seconds=total_time,
            )

        return summary, results

    def analyze_by_quality_bucket(
        self,
        results: list[GenerationResult],
    ) -> list[QualityBucket]:
        """Analyze results grouped by original quality score.

        Args:
            results: List of generation results.

        Returns:
            List of quality bucket statistics.
        """
        buckets = [
            (0.0, 0.3),
            (0.3, 0.5),
            (0.5, 0.7),
            (0.7, 0.9),
            (0.9, 1.0),
        ]

        bucket_stats = []
        for min_q, max_q in buckets:
            bucket_results = [
                r for r in results
                if min_q <= r.quality_score < max_q and r.llm_response is not None
            ]

            if bucket_results:
                bucket_stats.append(
                    QualityBucket(
                        min_quality=min_q,
                        max_quality=max_q,
                        count=len(bucket_results),
                        avg_similarity=float(
                            np.mean([r.similarity for r in bucket_results])
                        ),
                        avg_coherence=float(
                            np.mean([r.coherence for r in bucket_results])
                        ),
                    )
                )
            else:
                bucket_stats.append(
                    QualityBucket(
                        min_quality=min_q,
                        max_quality=max_q,
                        count=0,
                        avg_similarity=0.0,
                        avg_coherence=0.0,
                    )
                )

        return bucket_stats

    def save_results(
        self,
        summary: EvaluationSummary,
        results: list[GenerationResult],
        quality_buckets: list[QualityBucket],
    ) -> Path:
        """Save evaluation results to disk.

        Args:
            summary: Summary statistics.
            results: Detailed generation results.
            quality_buckets: Analysis by quality bucket.

        Returns:
            Path to the output directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save detailed results
        results_path = self.output_dir / f"results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save quality bucket analysis
        buckets_path = self.output_dir / f"quality_buckets_{timestamp}.json"
        with open(buckets_path, "w") as f:
            json.dump([asdict(b) for b in quality_buckets], f, indent=2)

        # Save human-readable report
        report_path = self.output_dir / f"report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Full LLM Generation Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Timestamp: {summary.timestamp}\n")
            f.write(f"Total Samples: {summary.total_samples}\n")
            f.write(f"Successful Generations: {summary.successful_generations}\n")
            f.write(f"Failed Generations: {summary.failed_generations}\n")
            f.write(f"Total Time: {summary.total_time_seconds:.1f}s\n\n")

            f.write("SIMILARITY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Similarity: {summary.avg_similarity:.3f}\n")
            f.write(f"Median Similarity: {summary.median_similarity:.3f}\n")
            f.write(f"Std Dev: {summary.std_similarity:.3f}\n\n")

            f.write("Percentiles:\n")
            f.write(f"  10th: {summary.similarity_p10:.3f}\n")
            f.write(f"  25th: {summary.similarity_p25:.3f}\n")
            f.write(f"  50th: {summary.similarity_p50:.3f}\n")
            f.write(f"  75th: {summary.similarity_p75:.3f}\n")
            f.write(f"  90th: {summary.similarity_p90:.3f}\n\n")

            f.write("COHERENCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Avg LLM Coherence: {summary.avg_coherence:.3f}\n")
            f.write(f"Avg Actual Coherence: {summary.avg_actual_coherence:.3f}\n")
            f.write(f"LLM More Coherent: {summary.llm_more_coherent_count}\n")
            f.write(f"Actual More Coherent: {summary.actual_more_coherent_count}\n")
            f.write(f"Equal Coherence: {summary.equal_coherence_count}\n\n")

            f.write("BY QUALITY BUCKET\n")
            f.write("-" * 40 + "\n")
            for bucket in quality_buckets:
                f.write(
                    f"  [{bucket.min_quality:.1f}-{bucket.max_quality:.1f}]: "
                    f"n={bucket.count}, sim={bucket.avg_similarity:.3f}, "
                    f"coh={bucket.avg_coherence:.3f}\n"
                )

            f.write("\n\nSAMPLE RESULTS\n")
            f.write("-" * 40 + "\n")

            # Show some high-similarity examples
            sorted_results = sorted(
                [r for r in results if r.llm_response],
                key=lambda x: x.similarity,
                reverse=True,
            )

            f.write("\nHigh Similarity Examples:\n")
            for r in sorted_results[:5]:
                f.write(f"\nTrigger: {r.trigger[:60]}...\n")
                f.write(f"Actual: {r.actual_response[:60]}...\n")
                f.write(f"LLM: {r.llm_response[:60] if r.llm_response else 'N/A'}...\n")
                f.write(f"Similarity: {r.similarity:.3f}, Coherence: {r.coherence:.3f}\n")

            f.write("\n\nLow Similarity Examples:\n")
            for r in sorted_results[-5:]:
                f.write(f"\nTrigger: {r.trigger[:60]}...\n")
                f.write(f"Actual: {r.actual_response[:60]}...\n")
                f.write(f"LLM: {r.llm_response[:60] if r.llm_response else 'N/A'}...\n")
                f.write(f"Similarity: {r.similarity:.3f}, Coherence: {r.coherence:.3f}\n")

        logger.info("Results saved to %s", self.output_dir)
        return self.output_dir


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the full LLM generation evaluation."""
    parser = argparse.ArgumentParser(
        description="Full LLM Generation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of samples to evaluate (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score for pairs (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run evaluation
    evaluator = FullLLMEvaluator(
        output_dir=args.output,
        verbose=args.verbose,
    )

    try:
        summary, results = evaluator.run_evaluation(
            num_samples=args.samples,
            min_quality=args.min_quality,
        )

        if summary:
            quality_buckets = evaluator.analyze_by_quality_bucket(results)
            evaluator.save_results(summary, results, quality_buckets)

            # Print summary to console
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            print(f"Total Samples: {summary.total_samples}")
            print(f"Successful: {summary.successful_generations}")
            print(f"Average Similarity: {summary.avg_similarity:.3f}")
            print(f"Median Similarity: {summary.median_similarity:.3f}")
            print(f"Avg LLM Coherence: {summary.avg_coherence:.3f}")
            print(f"Avg Actual Coherence: {summary.avg_actual_coherence:.3f}")
            print(f"Total Time: {summary.total_time_seconds:.1f}s")

            print("\nQuality Bucket Analysis:")
            for bucket in quality_buckets:
                print(
                    f"  [{bucket.min_quality:.1f}-{bucket.max_quality:.1f}]: "
                    f"n={bucket.count}, sim={bucket.avg_similarity:.3f}"
                )
        else:
            print("Evaluation failed - no results generated")
            sys.exit(1)

    finally:
        evaluator.unload_models()


if __name__ == "__main__":
    main()
