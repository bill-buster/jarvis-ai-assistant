"""Memory profiling benchmark (Workstream 1).

This module provides memory profiling for MLX models to validate
that the model stack fits within memory constraints.

Example usage:
    from benchmarks.memory import MLXMemoryProfiler, get_default_model

    profiler = MLXMemoryProfiler()
    model = get_default_model()
    profile = profiler.profile_model(model.path, context_length=512)
    print(f"RSS: {profile.rss_mb} MB")

CLI usage:
    python -m benchmarks.memory.run --output results/memory.json
"""

from benchmarks.memory.models import (
    CONTEXT_LENGTHS,
    ModelSpec,
    get_context_lengths,
    get_default_model,
    get_models_for_profiling,
)
from benchmarks.memory.profiler import MLXMemoryProfiler

__all__ = [
    "MLXMemoryProfiler",
    "ModelSpec",
    "CONTEXT_LENGTHS",
    "get_default_model",
    "get_models_for_profiling",
    "get_context_lengths",
]
