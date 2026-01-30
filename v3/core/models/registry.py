"""Model registry for JARVIS v3.

Single model: LFM2.5-1.2B - optimized for natural conversation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an MLX model."""

    id: str
    path: str  # HuggingFace path
    display_name: str
    size_gb: float
    description: str


# Single model - LFM2.5 optimized for natural conversation
MODELS: dict[str, ModelSpec] = {
    "lfm2.5-1.2b": ModelSpec(
        id="lfm2.5-1.2b",
        path="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        display_name="LFM2.5 1.2B",
        size_gb=0.5,
        description="Fast, natural conversation style (0.5GB)",
    ),
}

DEFAULT_MODEL = "lfm2.5-1.2b"


def get_model_spec(model_id: str = DEFAULT_MODEL) -> ModelSpec:
    """Get model specification.

    Args:
        model_id: Model identifier (ignored, always returns LFM2.5-1.2B)

    Returns:
        ModelSpec for LFM2.5-1.2B
    """
    return MODELS[DEFAULT_MODEL]
