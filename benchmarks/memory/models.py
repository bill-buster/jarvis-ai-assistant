"""Model configurations for memory profiling.

Workstream 1: Memory Profiler

Defines the models and context lengths to profile for memory benchmarking.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model to profile.

    Attributes:
        path: HuggingFace model path or local path
        name: Human-readable name
        estimated_memory_mb: Estimated memory usage for planning
        description: Brief description of the model
    """

    path: str
    name: str
    estimated_memory_mb: float
    description: str


# Context lengths to test (as specified in requirements)
CONTEXT_LENGTHS: list[int] = [512, 1024, 2048, 4096]


# Default model (from project configuration)
DEFAULT_MODEL = ModelSpec(
    path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    name="Qwen2.5-0.5B-4bit",
    estimated_memory_mb=800,
    description="Default JARVIS model - Qwen 0.5B with 4-bit quantization",
)


# Models to profile for benchmarking
# Start with smaller models to validate profiler, then expand
BENCHMARK_MODELS: list[ModelSpec] = [
    DEFAULT_MODEL,
    # Additional models can be added here as needed
    # ModelSpec(
    #     path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    #     name="Qwen2.5-1.5B-4bit",
    #     estimated_memory_mb=1500,
    #     description="Qwen 1.5B with 4-bit quantization",
    # ),
    # ModelSpec(
    #     path="mlx-community/Qwen2.5-3B-Instruct-4bit",
    #     name="Qwen2.5-3B-4bit",
    #     estimated_memory_mb=2500,
    #     description="Qwen 3B with 4-bit quantization",
    # ),
]


def get_models_for_profiling() -> list[ModelSpec]:
    """Return list of models to profile.

    Returns:
        List of ModelSpec instances
    """
    return list(BENCHMARK_MODELS)


def get_context_lengths() -> list[int]:
    """Return list of context lengths to test.

    Returns:
        List of context length values
    """
    return list(CONTEXT_LENGTHS)


def get_default_model() -> ModelSpec:
    """Return the default model for quick profiling.

    Returns:
        Default ModelSpec
    """
    return DEFAULT_MODEL
