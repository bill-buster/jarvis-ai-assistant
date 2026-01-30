"""Model Registry for JARVIS.

Provides model specifications and functions for selecting the best model
based on system capabilities. Supports multiple model tiers for different
RAM configurations.

Usage:
    from models.registry import get_recommended_model, get_model_spec, MODEL_REGISTRY

    # Get the best model for the user's system
    spec = get_recommended_model(available_ram_gb=16)

    # Get a specific model by ID
    spec = get_model_spec("qwen-1.5b")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported MLX model.

    Attributes:
        id: Unique identifier for the model (e.g., "qwen-0.5b").
        path: HuggingFace model path for MLX.
        display_name: Human-readable name for display in UI.
        size_gb: Approximate GPU memory usage in GB.
        min_ram_gb: Minimum system RAM required.
        quality_tier: Quality classification ("basic", "good", "excellent").
        description: User-facing description of the model.
        recommended_for: List of use cases this model excels at.
    """

    id: str
    path: str
    display_name: str
    size_gb: float
    min_ram_gb: int
    quality_tier: Literal["basic", "good", "excellent"]
    description: str
    recommended_for: list[str] = field(default_factory=list)

    @property
    def estimated_memory_mb(self) -> float:
        """Return estimated memory usage in MB."""
        return self.size_gb * 1024


# Registry of supported models, ordered by quality tier (ascending)
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen-0.5b": ModelSpec(
        id="qwen-0.5b",
        path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        display_name="Qwen 2.5 0.5B (Fast)",
        size_gb=0.8,
        min_ram_gb=8,
        quality_tier="basic",
        description="Fastest responses, basic quality. Good for quick replies.",
        recommended_for=["quick_replies"],
    ),
    "qwen-1.5b": ModelSpec(
        id="qwen-1.5b",
        path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        display_name="Qwen 2.5 1.5B (Balanced)",
        size_gb=1.5,
        min_ram_gb=8,
        quality_tier="good",
        description="Good balance of speed and quality. Recommended for most users.",
        recommended_for=["quick_replies", "summarization", "drafting"],
    ),
    "qwen-3b": ModelSpec(
        id="qwen-3b",
        path="mlx-community/Qwen2.5-3B-Instruct-4bit",
        display_name="Qwen 2.5 3B (Quality)",
        size_gb=2.5,
        min_ram_gb=8,
        quality_tier="excellent",
        description="Best quality responses for 4-bit quantized model.",
        recommended_for=["summarization", "drafting", "complex_replies"],
    ),
    "phi3-mini": ModelSpec(
        id="phi3-mini",
        path="mlx-community/Phi-3-mini-4k-instruct-4bit",
        display_name="Phi-3 Mini 4K",
        size_gb=2.5,
        min_ram_gb=8,
        quality_tier="good",
        description="Fast generation (28 tok/s), excellent for coding. Can be verbose.",
        recommended_for=["quick_replies", "coding_assistance"],
    ),
    "gemma3-4b": ModelSpec(
        id="gemma3-4b",
        path="mlx-community/gemma-3-4b-it-4bit",
        display_name="Gemma 3 4B Instruct",
        size_gb=2.75,
        min_ram_gb=8,
        quality_tier="excellent",
        description="Best instruction following, natural tone, concise responses. Recommended.",
        recommended_for=["quick_replies", "summarization", "drafting", "natural_conversation"],
    ),
    "bitnet-2b": ModelSpec(
        id="bitnet-2b",
        path="microsoft/bitnet-b1.58-2B-4T",
        display_name="BitNet b1.58 2B (Experimental)",
        size_gb=0.4,
        min_ram_gb=8,
        quality_tier="good",
        description="1.58-bit quantized model. 10x memory efficient, 2x faster on CPU. Experimental.",
        recommended_for=["quick_replies", "cpu_inference"],
    ),
    "lfm-1.2b": ModelSpec(
        id="lfm-1.2b",
        path="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        display_name="LFM 2.5 1.2B (Conversational)",
        size_gb=1.2,
        min_ram_gb=8,
        quality_tier="excellent",
        description="LFM 2.5 1.2B optimized for conversation. Best for natural chat.",
        recommended_for=["quick_replies", "natural_conversation", "drafting", "iMessage"],
    ),
}

# Default model ID when none specified
# LFM 2.5 is recommended as the default for conversational use cases
DEFAULT_MODEL_ID = "lfm-1.2b"


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Get model specification by ID.

    Args:
        model_id: The model identifier (e.g., "qwen-1.5b").

    Returns:
        ModelSpec if found, None otherwise.
    """
    return MODEL_REGISTRY.get(model_id)


def get_model_spec_by_path(model_path: str) -> ModelSpec | None:
    """Get model specification by HuggingFace path.

    Args:
        model_path: The HuggingFace model path (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit").

    Returns:
        ModelSpec if found, None otherwise.
    """
    for spec in MODEL_REGISTRY.values():
        if spec.path == model_path:
            return spec
    return None


def get_recommended_model(available_ram_gb: float) -> ModelSpec:
    """Return the best model for the user's available system RAM.

    Selects the highest quality model that fits within available RAM,
    with a safety buffer of 2GB for OS and other applications.

    Args:
        available_ram_gb: Available system RAM in GB.

    Returns:
        The recommended ModelSpec for the system.
    """
    # Sort models by quality tier (descending: excellent -> good -> basic)
    tier_order = {"excellent": 3, "good": 2, "basic": 1}
    sorted_models = sorted(
        MODEL_REGISTRY.values(),
        key=lambda m: tier_order[m.quality_tier],
        reverse=True,
    )

    # Find the best model that fits
    for model in sorted_models:
        # Add 2GB buffer for OS and other apps
        if available_ram_gb >= model.min_ram_gb:
            logger.debug(
                "Recommended model %s for %.1fGB RAM (min: %dGB)",
                model.id,
                available_ram_gb,
                model.min_ram_gb,
            )
            return model

    # Fallback to smallest model if somehow none fit
    fallback = MODEL_REGISTRY[DEFAULT_MODEL_ID]
    logger.warning(
        "No model fits %.1fGB RAM, falling back to %s",
        available_ram_gb,
        fallback.id,
    )
    return fallback


def get_all_models() -> list[ModelSpec]:
    """Return all available models sorted by quality tier.

    Returns:
        List of ModelSpec objects, sorted from basic to excellent.
    """
    tier_order = {"basic": 1, "good": 2, "excellent": 3}
    return sorted(
        MODEL_REGISTRY.values(),
        key=lambda m: tier_order[m.quality_tier],
    )


def is_model_available(model_id: str) -> bool:
    """Check if a model is downloaded and cached locally.

    Checks the HuggingFace cache directory for the model files.

    Args:
        model_id: The model identifier to check.

    Returns:
        True if model is cached locally, False otherwise.
    """
    spec = get_model_spec(model_id)
    if spec is None:
        return False

    # Check HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False

    # HuggingFace cache uses a specific naming pattern
    # e.g., models--mlx-community--Qwen2.5-0.5B-Instruct-4bit
    model_cache_name = f"models--{spec.path.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name

    if model_cache_path.exists():
        # Check if there are actual model files (snapshots directory with content)
        snapshots_dir = model_cache_path / "snapshots"
        if snapshots_dir.exists():
            # Check for at least one snapshot with model files
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir() and any(snapshot.iterdir()):
                    return True

    return False


def ensure_model_available(model_id: str) -> bool:
    """Download model if not available.

    Uses huggingface_hub to download the model to the local cache.

    Args:
        model_id: The model identifier to ensure is available.

    Returns:
        True if model is available (was cached or downloaded successfully),
        False if model_id is invalid or download failed.
    """
    spec = get_model_spec(model_id)
    if spec is None:
        logger.error("Unknown model ID: %s", model_id)
        return False

    # Check if already available
    if is_model_available(model_id):
        logger.debug("Model %s already available in cache", model_id)
        return True

    # Try to download
    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading model %s (%s)...", model_id, spec.path)
        snapshot_download(repo_id=spec.path)
        logger.info("Model %s downloaded successfully", model_id)
        return True

    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error("Failed to download model %s: %s", model_id, e)
        return False
