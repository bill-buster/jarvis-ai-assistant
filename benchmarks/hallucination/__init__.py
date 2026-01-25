"""HHEM hallucination evaluation benchmark (Workstream 2).

This module implements the HallucinationEvaluator protocol using the
vectara/hallucination_evaluation_model from HuggingFace.

Usage:
    python -m benchmarks.hallucination.run --output results/hhem.json

Components:
    - hhem: HHEMEvaluator class implementing the protocol
    - datasets: Test case generation for email summarization
    - run: CLI entrypoint for running benchmarks
"""

from benchmarks.hallucination.datasets import (
    EmailTestCase,
    generate_grounded_pairs,
    generate_hallucinated_pairs,
    generate_mixed_dataset,
    generate_test_cases,
    get_dataset_metadata,
)
from benchmarks.hallucination.hhem import (
    HHEMEvaluator,
    HHEMModelProtocol,
    VectaraHHEMModel,
    get_evaluator,
)

__all__ = [
    # Evaluator
    "HHEMEvaluator",
    "HHEMModelProtocol",
    "VectaraHHEMModel",
    "get_evaluator",
    # Datasets
    "EmailTestCase",
    "generate_grounded_pairs",
    "generate_hallucinated_pairs",
    "generate_mixed_dataset",
    "generate_test_cases",
    "get_dataset_metadata",
]
