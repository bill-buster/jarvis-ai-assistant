#!/usr/bin/env python3
"""LLM-as-Judge Quality Evaluation with Holdout Validation.

Evaluates response quality by comparing template responses vs LLM-generated
responses using a proper train/test split to avoid data leakage.

Key insight: We split pairs into train/test sets, build a temporary index
from ONLY the train set, then evaluate on test triggers. This ensures
we're measuring real-world performance on unseen messages.

Usage:
    python scripts/experiments/llm_quality_eval.py                    # Run with defaults
    python scripts/experiments/llm_quality_eval.py --samples 100      # Custom sample size
    python scripts/experiments/llm_quality_eval.py --test-ratio 0.3   # 30% test set
    python scripts/experiments/llm_quality_eval.py --verbose          # Detailed logging
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
from jarvis.router import (
    COHERENCE_THRESHOLD,
    CONTEXT_THRESHOLD,
    GENERATE_THRESHOLD,
    TEMPLATE_THRESHOLD,
)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLES = 100
DEFAULT_TEST_RATIO = 0.2  # 20% for testing
DEFAULT_OUTPUT_DIR = Path("results/llm_eval")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RANDOM_SEED = 42  # For reproducibility

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
class ComparisonResult:
    """Result of comparing template vs LLM response for a single trigger."""

    trigger: str
    actual_response: str  # The real response from history (ground truth)
    template_response: str | None  # Best match from train set
    llm_response: str | None
    template_similarity: float  # How similar the best train match is
    template_coherence: float
    llm_coherence: float | None
    winner: str  # "template", "llm", "tie", "both_failed"
    reason: str
    comparison_time_ms: float


@dataclass
class EvaluationSummary:
    """Summary statistics from the evaluation."""

    # Dataset info
    total_pairs: int
    train_size: int
    test_size: int
    test_ratio: float

    # Results
    total_comparisons: int
    template_wins: int
    llm_wins: int
    ties: int
    both_failed: int
    template_win_rate: float
    llm_win_rate: float

    # Quality metrics
    avg_template_similarity: float  # How similar train matches are to test
    avg_template_coherence: float
    avg_llm_coherence: float

    # Timing
    avg_comparison_time_ms: float
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Breakdown by similarity buckets
    high_sim_results: dict = field(default_factory=dict)  # >= 0.80
    medium_sim_results: dict = field(default_factory=dict)  # 0.60 - 0.80
    low_sim_results: dict = field(default_factory=dict)  # < 0.60


# =============================================================================
# In-Memory Index for Holdout Evaluation
# =============================================================================


class HoldoutIndex:
    """Temporary in-memory FAISS index for holdout evaluation.

    Does NOT modify the production index or database.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """Initialize with embedding model."""
        self.embedding_model = embedding_model
        self._model = None
        self._index = None
        self._train_pairs: list[Any] = []

    @property
    def model(self) -> Any:
        """Lazy load sentence transformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def build_from_pairs(self, train_pairs: list[Any]) -> None:
        """Build in-memory index from training pairs.

        Args:
            train_pairs: List of Pair objects to index.
        """
        import faiss

        self._train_pairs = train_pairs
        triggers = [p.trigger_text for p in train_pairs]

        logger.info("Building holdout index from %d train pairs...", len(triggers))

        # Encode all triggers
        embeddings = self.model.encode(
            triggers,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)

        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        logger.info("Holdout index built: %d vectors, %d dimensions", len(triggers), dimension)

    def search(self, query: str, k: int = 1, threshold: float = 0.0) -> list[dict]:
        """Search for similar triggers in the train set.

        Args:
            query: The test trigger to match.
            k: Number of results.
            threshold: Minimum similarity.

        Returns:
            List of dicts with trigger, response, and similarity.
        """
        if self._index is None:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                pair = self._train_pairs[idx]
                results.append(
                    {
                        "similarity": float(score),
                        "trigger_text": pair.trigger_text,
                        "response_text": pair.response_text,
                    }
                )

        return results


# =============================================================================
# Evaluation Logic
# =============================================================================


class HoldoutEvaluator:
    """Evaluator using proper train/test split."""

    def __init__(
        self,
        db: JarvisDB | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        test_ratio: float = DEFAULT_TEST_RATIO,
        verbose: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            db: JarvisDB instance. Uses default if None.
            output_dir: Directory to write results.
            test_ratio: Fraction of data to use for testing (0.0-1.0).
            verbose: Enable detailed logging.
        """
        self.db = db or get_db()
        self.db.init_schema()
        self.output_dir = output_dir
        self.test_ratio = test_ratio
        self.verbose = verbose

        self._holdout_index: HoldoutIndex | None = None
        self._generator = None

    @property
    def generator(self) -> Any:
        """Get or create the MLX generator."""
        if self._generator is None:
            from models import get_generator

            self._generator = get_generator(skip_templates=True)
        return self._generator

    def unload_models(self) -> None:
        """Unload models to free memory."""
        self._generator = None
        self._holdout_index = None
        gc.collect()

        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except (ImportError, AttributeError):
            try:
                import mlx.core as mx

                mx.clear_cache()
            except (ImportError, AttributeError):
                pass

    def split_data(self, pairs: list[Any], seed: int = RANDOM_SEED) -> tuple[list[Any], list[Any]]:
        """Split pairs into train/test sets.

        Args:
            pairs: All pairs from database.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_pairs, test_pairs).
        """
        random.seed(seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - self.test_ratio))
        train_pairs = shuffled[:split_idx]
        test_pairs = shuffled[split_idx:]

        logger.info(
            "Data split: %d train, %d test (%.1f%% test ratio)",
            len(train_pairs),
            len(test_pairs),
            self.test_ratio * 100,
        )

        return train_pairs, test_pairs

    def get_template_response(self, trigger: str) -> tuple[str | None, float]:
        """Get best matching template from train set.

        Args:
            trigger: The test trigger to match.

        Returns:
            Tuple of (response text or None, similarity score).
        """
        if self._holdout_index is None:
            return None, 0.0

        results = self._holdout_index.search(
            query=trigger,
            k=1,
            threshold=GENERATE_THRESHOLD,
        )

        if results:
            return results[0]["response_text"], results[0]["similarity"]
        return None, 0.0

    def get_similar_from_train(
        self, trigger: str, k: int = 3, threshold: float = 0.3
    ) -> list[tuple[str, str]]:
        """Get similar exchanges from the train set for few-shot context.

        Args:
            trigger: The test trigger to find similar examples for.
            k: Number of similar examples to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (trigger, response) tuples from train set.
        """
        if self._holdout_index is None:
            return []

        # Search for similar triggers in train set
        results = self._holdout_index.search(
            query=trigger,
            k=k,
            threshold=threshold,
        )

        return [(r["trigger_text"], r["response_text"]) for r in results]

    def get_llm_response(
        self,
        trigger: str,
        contact_name: str = "them",
        similar_exchanges: list[tuple[str, str]] | None = None,
        conversation_context: str | None = None,
    ) -> str | None:
        """Generate LLM response for a trigger with full context.

        Args:
            trigger: The incoming message.
            contact_name: Name of the contact.
            similar_exchanges: Similar (trigger, response) pairs from train set.
            conversation_context: Previous messages from the conversation (stored in pair).

        Returns:
            Generated response text or None.
        """
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        try:
            # Format similar exchanges for the prompt
            # Convert (trigger, response) to (context, response) format
            formatted_exchanges = []
            if similar_exchanges:
                for trig, resp in similar_exchanges:
                    formatted_exchanges.append((f"[Incoming]: {trig}", resp))

            # Build context: use stored conversation context if available
            if conversation_context:
                # Conversation context is already formatted as "[Speaker]: message"
                context = f"{conversation_context}\n[Incoming]: {trigger}"
            else:
                context = f"[Incoming]: {trigger}"

            prompt = build_rag_reply_prompt(
                context=context,
                last_message=trigger,
                contact_name=contact_name,
                similar_exchanges=formatted_exchanges,
                relationship_profile={
                    "tone": "casual",  # Default to casual for evaluation
                    "avg_message_length": 50,
                },
            )

            # GenerationRequest now has LFM-optimal defaults built-in
            # Use shorter max_tokens for text messages (they should be concise)
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context] if context else [],
                few_shot_examples=formatted_exchanges,
                max_tokens=80,  # Text messages should be short
                # Using defaults: temp=0.1, top_p=0.1, top_k=50, repetition_penalty=1.05
            )

            response = self.generator.generate(request)
            return response.text.strip()

        except Exception as e:
            logger.warning("Error generating LLM response: %s", e)
            return None

    def compare_single(
        self,
        test_pair: Any,
    ) -> ComparisonResult:
        """Compare template vs LLM response for a single test trigger.

        Args:
            test_pair: A Pair object from the test set.

        Returns:
            ComparisonResult with comparison details.
        """
        start_time = time.time()

        trigger = test_pair.trigger_text
        actual_response = test_pair.response_text

        # Get conversation context from the pair (if available)
        # This is the context that was stored during extraction
        pair_context = getattr(test_pair, "context_text", None)

        # Get template response from train set (best single match)
        template_response, template_similarity = self.get_template_response(trigger)

        # Get similar exchanges from train set for LLM context
        # Use more examples and lower threshold to give LLM good context
        similar_exchanges = self.get_similar_from_train(trigger, k=3, threshold=0.3)

        # Get LLM response with full context
        llm_response = self.get_llm_response(
            trigger,
            similar_exchanges=similar_exchanges,
            conversation_context=pair_context,
        )

        # Calculate coherence scores
        template_coherence = (
            score_response_coherence(trigger, template_response) if template_response else 0.0
        )
        llm_coherence = score_response_coherence(trigger, llm_response) if llm_response else None

        # Determine winner based on coherence and availability
        if template_response and llm_response:
            t_score = template_coherence
            l_score = llm_coherence or 0.0

            # Factor in similarity - lower similarity means less reliable template
            # Penalize template if similarity is low
            if template_similarity < 0.70:
                t_score *= template_similarity

            if abs(t_score - l_score) < 0.1:
                winner = "tie"
                reason = f"Similar coherence (template={t_score:.2f}, llm={l_score:.2f})"
            elif t_score > l_score:
                winner = "template"
                reason = f"Higher coherence ({t_score:.2f} vs {l_score:.2f})"
            else:
                winner = "llm"
                reason = f"Higher coherence ({l_score:.2f} vs {t_score:.2f})"
        elif template_response:
            winner = "template"
            reason = "LLM generation failed"
        elif llm_response:
            winner = "llm"
            reason = f"No template match above threshold (best={template_similarity:.2f})"
        else:
            winner = "both_failed"
            reason = "Both template and LLM failed"

        elapsed_ms = (time.time() - start_time) * 1000

        return ComparisonResult(
            trigger=trigger,
            actual_response=actual_response,
            template_response=template_response,
            llm_response=llm_response,
            template_similarity=template_similarity,
            template_coherence=template_coherence,
            llm_coherence=llm_coherence,
            winner=winner,
            reason=reason,
            comparison_time_ms=elapsed_ms,
        )

    def run_evaluation(
        self,
        num_samples: int = DEFAULT_SAMPLES,
        min_quality: float = 0.5,
    ) -> tuple[EvaluationSummary | None, list[ComparisonResult]]:
        """Run the full holdout evaluation.

        Args:
            num_samples: Maximum number of test pairs to evaluate.
            min_quality: Minimum quality score for pairs to include.

        Returns:
            Tuple of (summary statistics, list of detailed results).
        """
        start_time = time.time()

        # Get all pairs from database
        all_pairs = self.db.get_all_pairs(min_quality=min_quality)
        if not all_pairs:
            logger.error("No pairs found in database")
            return None, []

        # Split into train/test
        train_pairs, test_pairs = self.split_data(all_pairs)

        if not test_pairs:
            logger.error("Test set is empty after split")
            return None, []

        # Build holdout index from train set only
        self._holdout_index = HoldoutIndex()
        self._holdout_index.build_from_pairs(train_pairs)

        # Sample from test set if needed
        sample_size = min(num_samples, len(test_pairs))
        sampled_test = random.sample(test_pairs, sample_size)

        logger.info("Evaluating %d test pairs (from %d total test)", sample_size, len(test_pairs))

        # Run comparisons
        results: list[ComparisonResult] = []
        for i, test_pair in enumerate(sampled_test):
            if self.verbose or (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d", i + 1, sample_size)

            result = self.compare_single(test_pair)
            results.append(result)

            if self.verbose:
                logger.info(
                    "  Trigger: %s... -> Winner: %s (sim=%.2f)",
                    test_pair.trigger_text[:50],
                    result.winner,
                    result.template_similarity,
                )

        # Calculate summary statistics
        template_wins = sum(1 for r in results if r.winner == "template")
        llm_wins = sum(1 for r in results if r.winner == "llm")
        ties = sum(1 for r in results if r.winner == "tie")
        both_failed = sum(1 for r in results if r.winner == "both_failed")

        valid_comparisons = len(results) - both_failed

        similarities = [r.template_similarity for r in results]
        template_coherences = [r.template_coherence for r in results if r.template_coherence > 0]
        llm_coherences = [r.llm_coherence for r in results if r.llm_coherence is not None]

        # Breakdown by similarity buckets
        high_sim = [r for r in results if r.template_similarity >= 0.80]
        medium_sim = [r for r in results if 0.60 <= r.template_similarity < 0.80]
        low_sim = [r for r in results if r.template_similarity < 0.60]

        def bucket_stats(bucket: list) -> dict:
            if not bucket:
                return {"count": 0, "template_wins": 0, "llm_wins": 0, "ties": 0}
            return {
                "count": len(bucket),
                "template_wins": sum(1 for r in bucket if r.winner == "template"),
                "llm_wins": sum(1 for r in bucket if r.winner == "llm"),
                "ties": sum(1 for r in bucket if r.winner == "tie"),
                "avg_similarity": sum(r.template_similarity for r in bucket) / len(bucket),
            }

        total_time = time.time() - start_time

        summary = EvaluationSummary(
            total_pairs=len(all_pairs),
            train_size=len(train_pairs),
            test_size=len(test_pairs),
            test_ratio=self.test_ratio,
            total_comparisons=len(results),
            template_wins=template_wins,
            llm_wins=llm_wins,
            ties=ties,
            both_failed=both_failed,
            template_win_rate=(
                template_wins / valid_comparisons * 100 if valid_comparisons > 0 else 0
            ),
            llm_win_rate=llm_wins / valid_comparisons * 100 if valid_comparisons > 0 else 0,
            avg_template_similarity=(sum(similarities) / len(similarities) if similarities else 0),
            avg_template_coherence=(
                sum(template_coherences) / len(template_coherences) if template_coherences else 0
            ),
            avg_llm_coherence=(sum(llm_coherences) / len(llm_coherences) if llm_coherences else 0),
            avg_comparison_time_ms=(
                sum(r.comparison_time_ms for r in results) / len(results) if results else 0
            ),
            total_time_seconds=total_time,
            high_sim_results=bucket_stats(high_sim),
            medium_sim_results=bucket_stats(medium_sim),
            low_sim_results=bucket_stats(low_sim),
        )

        return summary, results

    def save_results(
        self,
        summary: EvaluationSummary,
        results: list[ComparisonResult],
    ) -> Path:
        """Save evaluation results to disk.

        Args:
            summary: Summary statistics.
            results: Detailed comparison results.

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

        # Save human-readable report
        report_path = self.output_dir / f"report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("LLM vs Template Quality Evaluation (Holdout Validation)\n")
            f.write("=" * 70 + "\n\n")

            f.write("DATASET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Pairs: {summary.total_pairs}\n")
            f.write(f"Train Set: {summary.train_size} ({100 - summary.test_ratio * 100:.0f}%)\n")
            f.write(f"Test Set: {summary.test_size} ({summary.test_ratio * 100:.0f}%)\n")
            f.write(f"Samples Evaluated: {summary.total_comparisons}\n\n")

            f.write("OVERALL RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Template Wins: {summary.template_wins} ({summary.template_win_rate:.1f}%)\n")
            f.write(f"LLM Wins: {summary.llm_wins} ({summary.llm_win_rate:.1f}%)\n")
            f.write(f"Ties: {summary.ties}\n")
            f.write(f"Both Failed: {summary.both_failed}\n\n")

            f.write("QUALITY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Avg Template Similarity: {summary.avg_template_similarity:.3f}\n")
            f.write(f"Avg Template Coherence: {summary.avg_template_coherence:.3f}\n")
            f.write(f"Avg LLM Coherence: {summary.avg_llm_coherence:.3f}\n\n")

            f.write("RESULTS BY SIMILARITY BUCKET\n")
            f.write("-" * 40 + "\n")
            for name, bucket in [
                ("High (>=0.80)", summary.high_sim_results),
                ("Medium (0.60-0.80)", summary.medium_sim_results),
                ("Low (<0.60)", summary.low_sim_results),
            ]:
                if bucket.get("count", 0) > 0:
                    f.write(f"{name}: {bucket['count']} samples\n")
                    f.write(f"  Template: {bucket['template_wins']}, ")
                    f.write(f"LLM: {bucket['llm_wins']}, ")
                    f.write(f"Tie: {bucket['ties']}\n")
                    f.write(f"  Avg Similarity: {bucket.get('avg_similarity', 0):.3f}\n")
                else:
                    f.write(f"{name}: 0 samples\n")
            f.write("\n")

            f.write("SAMPLE COMPARISONS\n")
            f.write("-" * 40 + "\n")
            for result in results[:10]:
                f.write(f"\nTrigger: {result.trigger[:80]}...\n")
                f.write(f"Template (sim={result.template_similarity:.2f}): ")
                f.write(f"{(result.template_response or 'None')[:60]}...\n")
                f.write(f"LLM: {(result.llm_response or 'None')[:60]}...\n")
                f.write(f"Winner: {result.winner} - {result.reason}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Total Time: {summary.total_time_seconds:.1f}s\n")
            f.write(f"Timestamp: {summary.timestamp}\n")

        logger.info("Results saved to %s", self.output_dir)
        return self.output_dir


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the holdout evaluation."""
    parser = argparse.ArgumentParser(
        description="LLM vs Template Quality Evaluation (Holdout Validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Max test samples to evaluate (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of data for testing (default: {DEFAULT_TEST_RATIO})",
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
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
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

    # Print configuration
    logger.info("Holdout Evaluation Configuration:")
    logger.info("  Test ratio: %.1f%%", args.test_ratio * 100)
    logger.info("  Max samples: %d", args.samples)
    logger.info("  Random seed: %d", args.seed)
    logger.info("  Router thresholds:")
    logger.info("    TEMPLATE_THRESHOLD: %.2f", TEMPLATE_THRESHOLD)
    logger.info("    CONTEXT_THRESHOLD: %.2f", CONTEXT_THRESHOLD)
    logger.info("    GENERATE_THRESHOLD: %.2f", GENERATE_THRESHOLD)
    logger.info("    COHERENCE_THRESHOLD: %.2f", COHERENCE_THRESHOLD)

    # Set random seed
    random.seed(args.seed)

    # Run evaluation
    evaluator = HoldoutEvaluator(
        output_dir=args.output,
        test_ratio=args.test_ratio,
        verbose=args.verbose,
    )

    try:
        summary, results = evaluator.run_evaluation(
            num_samples=args.samples,
            min_quality=args.min_quality,
        )

        if summary:
            evaluator.save_results(summary, results)

            # Print summary to console
            print("\n" + "=" * 60)
            print("HOLDOUT EVALUATION COMPLETE")
            print("=" * 60)
            print(f"Dataset: {summary.train_size} train / {summary.test_size} test")
            print(f"Evaluated: {summary.total_comparisons} test samples")
            print(f"\nTemplate Win Rate: {summary.template_win_rate:.1f}%")
            print(f"LLM Win Rate: {summary.llm_win_rate:.1f}%")
            print(f"Ties: {summary.ties} ({summary.ties / summary.total_comparisons * 100:.1f}%)")
            print(f"\nAvg Template Similarity: {summary.avg_template_similarity:.3f}")
            print(f"Avg Template Coherence: {summary.avg_template_coherence:.3f}")
            print(f"Avg LLM Coherence: {summary.avg_llm_coherence:.3f}")
            print(f"\nTotal Time: {summary.total_time_seconds:.1f}s")
        else:
            print("Evaluation failed - no results generated")
            sys.exit(1)

    finally:
        evaluator.unload_models()


if __name__ == "__main__":
    main()
