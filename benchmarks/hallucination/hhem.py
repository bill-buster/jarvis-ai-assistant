"""HHEM hallucination evaluator implementation.

Workstream 2: HHEM Hallucination Benchmark

Implements the HallucinationEvaluator protocol from contracts/hallucination.py
using the vectara/hallucination_evaluation_model from HuggingFace.
"""

import logging
from datetime import UTC, datetime
from statistics import mean, median, stdev
from typing import Any, Protocol, runtime_checkable

from contracts.hallucination import HHEMBenchmarkResult, HHEMResult

logger = logging.getLogger(__name__)

# Batch size for efficient HHEM evaluation
_BATCH_SIZE = 16


@runtime_checkable
class HHEMModelProtocol(Protocol):
    """Protocol for HHEM model to enable mocking in tests."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Predict hallucination scores for source/summary pairs.

        Args:
            pairs: List of [source, summary] pairs

        Returns:
            List of scores from 0 (hallucinated) to 1 (grounded)
        """
        ...


class VectaraHHEMModel:
    """Wrapper for the Vectara HHEM model from HuggingFace.

    Uses vectara/hallucination_evaluation_model via sentence-transformers.
    """

    def __init__(self) -> None:
        """Initialize the HHEM model wrapper."""
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazy load the HHEM model.

        Returns:
            The loaded CrossEncoder model
        """
        if self._model is None:
            logger.info("Loading HHEM model: vectara/hallucination_evaluation_model")
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder("vectara/hallucination_evaluation_model")
            except Exception as e:
                logger.error("Failed to load HHEM model: %s", e)
                raise
        return self._model

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Predict hallucination scores for source/summary pairs.

        The model scores text from 0 (hallucinated) to 1 (grounded).

        Args:
            pairs: List of [source, summary] pairs where source is the
                   original text and summary is the generated summary.

        Returns:
            List of scores from 0 (hallucinated) to 1 (grounded)
        """
        if not pairs:
            return []

        model = self._ensure_model()
        # CrossEncoder.predict returns numpy array of scores
        scores = model.predict(pairs)

        # Ensure we return a list of floats
        return [float(s) for s in scores]


class HHEMEvaluator:
    """HHEM-based hallucination evaluator.

    Implements HallucinationEvaluator protocol from contracts/hallucination.py.

    Uses the vectara/hallucination_evaluation_model to score text on a scale
    from 0 (hallucinated) to 1 (grounded).
    """

    def __init__(self, model: HHEMModelProtocol | None = None) -> None:
        """Initialize the evaluator.

        Args:
            model: Optional HHEM model instance for dependency injection.
                   If not provided, uses the real Vectara model.
        """
        self._model: HHEMModelProtocol = model if model is not None else VectaraHHEMModel()

    def evaluate_single(self, source: str, summary: str) -> float:
        """Return HHEM score for a source/summary pair.

        Args:
            source: Original source text
            summary: Generated summary to evaluate

        Returns:
            Score from 0 (hallucinated) to 1 (grounded)
        """
        scores = self._model.predict([[source, summary]])
        return scores[0] if scores else 0.0

    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch evaluate multiple pairs. More efficient than single calls.

        Processes pairs in batches of 16-32 for optimal performance.

        Args:
            pairs: List of (source, summary) tuples

        Returns:
            List of scores from 0 (hallucinated) to 1 (grounded)
        """
        if not pairs:
            return []

        # Convert tuples to list format expected by model
        all_pairs = [[source, summary] for source, summary in pairs]
        all_scores: list[float] = []

        # Process in batches for efficiency
        for i in range(0, len(all_pairs), _BATCH_SIZE):
            batch = all_pairs[i : i + _BATCH_SIZE]
            batch_scores = self._model.predict(batch)
            all_scores.extend(batch_scores)

        return all_scores

    def run_benchmark(
        self,
        model_name: str,
        dataset: list[tuple[str, str, str]],
        prompt_templates: list[str],
    ) -> HHEMBenchmarkResult:
        """Run full benchmark and return aggregate results.

        Args:
            model_name: Name of the model being evaluated
            dataset: List of (source_text, generated_summary, template_name) tuples
            prompt_templates: List of prompt template names to filter by (or all if empty)

        Returns:
            HHEMBenchmarkResult with aggregate statistics
        """
        if not dataset:
            return HHEMBenchmarkResult(
                model_name=model_name,
                num_samples=0,
                mean_score=0.0,
                median_score=0.0,
                std_score=0.0,
                pass_rate_at_05=0.0,
                pass_rate_at_07=0.0,
                results=[],
                timestamp=datetime.now(UTC).isoformat(),
            )

        # Filter by templates if specified
        if prompt_templates:
            filtered_dataset = [
                (src, summ, tpl) for src, summ, tpl in dataset if tpl in prompt_templates
            ]
        else:
            filtered_dataset = list(dataset)

        if not filtered_dataset:
            return HHEMBenchmarkResult(
                model_name=model_name,
                num_samples=0,
                mean_score=0.0,
                median_score=0.0,
                std_score=0.0,
                pass_rate_at_05=0.0,
                pass_rate_at_07=0.0,
                results=[],
                timestamp=datetime.now(UTC).isoformat(),
            )

        # Prepare pairs for batch evaluation
        pairs = [(src, summ) for src, summ, _ in filtered_dataset]
        logger.info("Evaluating %d source/summary pairs", len(pairs))

        # Evaluate all pairs
        scores = self.evaluate_batch(pairs)

        # Create individual results
        timestamp = datetime.now(UTC).isoformat()
        results: list[HHEMResult] = []
        for (source, summary, template), score in zip(filtered_dataset, scores, strict=True):
            results.append(
                HHEMResult(
                    model_name=model_name,
                    prompt_template=template,
                    source_text=source,
                    generated_summary=summary,
                    hhem_score=score,
                    timestamp=timestamp,
                )
            )

        # Compute aggregate statistics
        mean_score = mean(scores)
        median_score = median(scores)
        std_score = stdev(scores) if len(scores) > 1 else 0.0
        pass_05 = sum(1 for s in scores if s >= 0.5) / len(scores)
        pass_07 = sum(1 for s in scores if s >= 0.7) / len(scores)

        return HHEMBenchmarkResult(
            model_name=model_name,
            num_samples=len(results),
            mean_score=mean_score,
            median_score=median_score,
            std_score=std_score,
            pass_rate_at_05=pass_05,
            pass_rate_at_07=pass_07,
            results=results,
            timestamp=timestamp,
        )


def get_evaluator(model: HHEMModelProtocol | None = None) -> HHEMEvaluator:
    """Get an HHEM evaluator instance.

    Args:
        model: Optional model for dependency injection (testing)

    Returns:
        Configured HHEMEvaluator instance
    """
    return HHEMEvaluator(model=model)
