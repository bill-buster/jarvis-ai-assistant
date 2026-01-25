"""Latency benchmarking interfaces.

Workstream 4 implements against these contracts.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

Scenario = Literal["cold", "warm", "hot"]


@dataclass
class LatencyResult:
    """Result of a single latency measurement."""

    scenario: Scenario
    model_name: str
    context_length: int
    output_tokens: int
    load_time_ms: float
    prefill_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    timestamp: str


@dataclass
class LatencyBenchmarkResult:
    """Aggregate latency benchmark results."""

    scenario: Scenario
    model_name: str
    num_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    results: list[LatencyResult]
    timestamp: str


class LatencyBenchmarker(Protocol):
    """Interface for latency benchmarking (Workstream 4)."""

    def measure_single(
        self, model_path: str, scenario: Scenario, prompt: str, max_tokens: int
    ) -> LatencyResult:
        """Measure latency for a single generation."""
        ...

    def run_benchmark(
        self, model_path: str, scenario: Scenario, num_runs: int = 10
    ) -> LatencyBenchmarkResult:
        """Run full benchmark with statistical analysis."""
        ...
