"""Model registry for JARVIS v2.

Defines available MLX models for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an MLX model."""

    id: str
    path: str  # HuggingFace path
    display_name: str
    size_gb: float
    quality: Literal["basic", "good", "excellent"]
    description: str


# Available models for testing
MODELS: dict[str, ModelSpec] = {
    "qwen-0.5b": ModelSpec(
        id="qwen-0.5b",
        path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        display_name="Qwen 2.5 0.5B",
        size_gb=0.8,
        quality="basic",
        description="Fastest, basic quality",
    ),
    "qwen-1.5b": ModelSpec(
        id="qwen-1.5b",
        path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        display_name="Qwen 2.5 1.5B",
        size_gb=1.5,
        quality="good",
        description="Balanced speed/quality",
    ),
    "qwen-3b": ModelSpec(
        id="qwen-3b",
        path="mlx-community/Qwen2.5-3B-Instruct-4bit",
        display_name="Qwen 2.5 3B",
        size_gb=2.5,
        quality="excellent",
        description="Best quality for 4-bit",
    ),
    "phi3-mini": ModelSpec(
        id="phi3-mini",
        path="mlx-community/Phi-3-mini-4k-instruct-4bit",
        display_name="Phi-3 Mini",
        size_gb=2.5,
        quality="good",
        description="Fast, good for conversations",
    ),
    "gemma3-4b": ModelSpec(
        id="gemma3-4b",
        path="mlx-community/gemma-3-4b-it-4bit",
        display_name="Gemma 3 4B",
        size_gb=2.8,
        quality="excellent",
        description="High quality, slightly slower",
    ),
}

# Default model
DEFAULT_MODEL = "qwen-1.5b"


def get_model_spec(model_id: str) -> ModelSpec:
    """Get model specification by ID.

    Args:
        model_id: Model identifier

    Returns:
        ModelSpec for the model

    Raises:
        KeyError: If model not found
    """
    if model_id not in MODELS:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")
    return MODELS[model_id]


def get_recommended_model(available_ram_gb: float = 8.0) -> ModelSpec:
    """Get recommended model based on available RAM.

    Args:
        available_ram_gb: Available system RAM in GB

    Returns:
        Best ModelSpec for the RAM constraint
    """
    # Leave ~4GB for OS and other apps
    usable_ram = available_ram_gb - 4.0

    # Find best quality model that fits
    candidates = [
        spec for spec in MODELS.values()
        if spec.size_gb <= usable_ram
    ]

    if not candidates:
        # Fall back to smallest model
        return MODELS["qwen-0.5b"]

    # Sort by quality (excellent > good > basic), then by size (larger is better)
    quality_order = {"excellent": 3, "good": 2, "basic": 1}
    candidates.sort(key=lambda s: (quality_order[s.quality], s.size_gb), reverse=True)

    return candidates[0]
