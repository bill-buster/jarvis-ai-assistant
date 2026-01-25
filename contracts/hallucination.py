"""Hallucination evaluation interfaces.

Workstream 2 implements against these contracts.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class HHEMResult:
    """Result of evaluating a single source/summary pair."""

    model_name: str
    prompt_template: str
    source_text: str
    generated_summary: str
    hhem_score: float  # 0.0 = hallucinated, 1.0 = grounded
    timestamp: str


@dataclass
class HHEMBenchmarkResult:
    """Aggregate results from a benchmark run."""

    model_name: str
    num_samples: int
    mean_score: float
    median_score: float
    std_score: float
    pass_rate_at_05: float  # % of samples with score >= 0.5
    pass_rate_at_07: float  # % of samples with score >= 0.7
    results: list[HHEMResult]
    timestamp: str


class HallucinationEvaluator(Protocol):
    """Interface for hallucination evaluation (Workstream 2)."""

    def evaluate_single(self, source: str, summary: str) -> float:
        """Return HHEM score for a source/summary pair."""
        ...

    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch evaluate multiple pairs. More efficient than single calls."""
        ...

    def run_benchmark(
        self, model_name: str, dataset_path: str, prompt_templates: list[str]
    ) -> HHEMBenchmarkResult:
        """Run full benchmark and return aggregate results."""
        ...
