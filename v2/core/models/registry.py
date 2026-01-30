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
    # Prompt format: "chatml", "llama3", "gemma", "mistral", "raw"
    prompt_format: str = "chatml"


# Available models
# prompt_format options:
#   - "chatml": <|im_start|>user\n...<|im_end|> (Qwen, LFM)
#   - "llama3": <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
#   - "gemma": <start_of_turn>user\n...<end_of_turn>
#   - "mistral": [INST]...[/INST]
#   - "raw": No template, just text completion
MODELS: dict[str, ModelSpec] = {
    # === Small models (< 1.5GB) - Fast, good for casual chat ===

    # LFM2.5 - Liquid Foundation Model, optimized for natural conversation
    "lfm2.5-1.2b": ModelSpec(
        id="lfm2.5-1.2b",
        path="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        display_name="LFM2.5 1.2B",
        size_gb=0.5,
        quality="excellent",
        description="Fast, natural conversation style",
        prompt_format="chatml",
    ),
    "lfm2-2.6b": ModelSpec(
        id="lfm2-2.6b",
        path="mlx-community/LFM2-2.6B-4bit",
        display_name="LFM2 2.6B",
        size_gb=1.5,
        quality="excellent",
        description="Larger LFM2, better quality",
        prompt_format="chatml",
    ),
    "lfm2-2.6b-exp": ModelSpec(
        id="lfm2-2.6b-exp",
        path="mlx-community/LFM2-2.6B-Exp-4bit",
        display_name="LFM2 2.6B Exp",
        size_gb=1.5,
        quality="excellent",
        description="RL-tuned LFM2, beats larger models on benchmarks",
        prompt_format="chatml",
    ),

    # Llama 3.2 - Meta's lightweight model, good at following examples
    "llama-3.2-1b": ModelSpec(
        id="llama-3.2-1b",
        path="mlx-community/Llama-3.2-1B-Instruct-4bit",
        display_name="Llama 3.2 1B",
        size_gb=0.7,
        quality="good",
        description="Follows few-shot examples well",
        prompt_format="llama3",
    ),
    "llama-3.2-3b": ModelSpec(
        id="llama-3.2-3b",
        path="mlx-community/Llama-3.2-3B-Instruct-4bit",
        display_name="Llama 3.2 3B",
        size_gb=1.8,
        quality="excellent",
        description="Better reasoning than 1B, still fast",
        prompt_format="llama3",
    ),

    # Gemma 3 - Google's model, good at multi-turn conversation
    "gemma3-1b": ModelSpec(
        id="gemma3-1b",
        path="mlx-community/gemma-3-1b-it-qat-4bit",
        display_name="Gemma 3 1B",
        size_gb=0.6,
        quality="good",
        description="Google's efficient small model",
        prompt_format="gemma",
    ),
    "gemma3-4b": ModelSpec(
        id="gemma3-4b",
        path="mlx-community/gemma-3-4b-it-qat-4bit",
        display_name="Gemma 3 4B",
        size_gb=2.2,
        quality="excellent",
        description="Best balance of speed and quality",
        prompt_format="gemma",
    ),

    # Ministral 3 - Mistral's edge-optimized model
    "ministral-3b": ModelSpec(
        id="ministral-3b",
        path="mlx-community/Ministral-3b-Instruct-4bit",
        display_name="Ministral 3B",
        size_gb=1.8,
        quality="excellent",
        description="Mistral's edge model, 256k context",
        prompt_format="mistral",
    ),

    # SmolLM2 - HuggingFace's best small model
    "smollm2-1.7b": ModelSpec(
        id="smollm2-1.7b",
        path="mlx-community/SmolLM2-1.7B-Instruct",
        display_name="SmolLM2 1.7B",
        size_gb=1.0,
        quality="excellent",
        description="HF's best small model, beats Llama 3.2 1B",
        prompt_format="chatml",
    ),

    # Phi-3.5 - Microsoft's instruction-tuned model
    "phi-3.5-mini": ModelSpec(
        id="phi-3.5-mini",
        path="mlx-community/Phi-3.5-mini-instruct-4bit",
        display_name="Phi-3.5 Mini",
        size_gb=2.0,
        quality="excellent",
        description="Microsoft, great at following instructions",
        prompt_format="chatml",  # Phi uses ChatML-style
    ),

    # StableLM Zephyr - Stability AI's chat model
    "stablelm-zephyr-3b": ModelSpec(
        id="stablelm-zephyr-3b",
        path="mlx-community/stablelm-zephyr-3b-4bit",
        display_name="StableLM Zephyr 3B",
        size_gb=1.8,
        quality="good",
        description="Stability AI, optimized for chat",
        prompt_format="chatml",
    ),

    # === Medium models (1.5-3GB) - Better quality ===

    # Qwen2.5 - Often better than Qwen3 for casual text
    "qwen2.5-1.5b": ModelSpec(
        id="qwen2.5-1.5b",
        path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        display_name="Qwen2.5 1.5B",
        size_gb=1.0,
        quality="good",
        description="May be better than Qwen3 for casual text",
        prompt_format="chatml",
    ),

    # Qwen3 - Strong reasoning, tends to be verbose
    "qwen3-1.7b": ModelSpec(
        id="qwen3-1.7b",
        path="mlx-community/Qwen3-1.7B-4bit",
        display_name="Qwen3 1.7B",
        size_gb=1.2,
        quality="good",
        description="Good reasoning, can be verbose",
        prompt_format="chatml",
    ),
    "qwen3-4b": ModelSpec(
        id="qwen3-4b",
        path="Qwen/Qwen3-4B-MLX-4bit",
        display_name="Qwen3 4B",
        size_gb=2.1,
        quality="excellent",
        description="Strong reasoning and instruction following",
        prompt_format="chatml",
    ),

    # === New models to evaluate ===

    # SmolLM3 - HuggingFace's newest, beats Llama-3.2-3B
    "smollm3-3b": ModelSpec(
        id="smollm3-3b",
        path="mlx-community/SmolLM3-3B-4bit",
        display_name="SmolLM3 3B",
        size_gb=1.73,
        quality="excellent",
        description="HF's newest, beats Llama-3.2-3B on benchmarks",
        prompt_format="chatml",
    ),

    # Jan-v3 - Chat-optimized Qwen3-4B variant
    "jan-v3-4b": ModelSpec(
        id="jan-v3-4b",
        path="mlx-community/Jan-v3-4B-base-instruct-8bit",
        display_name="Jan v3 4B",
        size_gb=4.27,
        quality="excellent",
        description="Chat-tuned Qwen3-4B by janhq",
        prompt_format="chatml",
    ),

    # Qwen3-4B July 2025 update
    "qwen3-4b-2507": ModelSpec(
        id="qwen3-4b-2507",
        path="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        display_name="Qwen3 4B (July 2025)",
        size_gb=2.5,
        quality="excellent",
        description="Latest Qwen3-4B update",
        prompt_format="chatml",
    ),

    # Trinity Nano - Experimental MoE with 1B active params
    "trinity-nano": ModelSpec(
        id="trinity-nano",
        path="arcee-ai/Trinity-Nano-Preview-MLX-5bit",
        display_name="Trinity Nano",
        size_gb=4.21,
        quality="good",
        description="MoE: 6B total, 1B active, 128k context",
        prompt_format="chatml",
    ),

    # Qwen3-0.6B - Tiny but fast
    "qwen3-0.6b": ModelSpec(
        id="qwen3-0.6b",
        path="mlx-community/Qwen3-0.6B-4bit",
        display_name="Qwen3 0.6B",
        size_gb=0.5,
        quality="basic",
        description="Tiny, 80-120 tok/s, good for simple replies",
        prompt_format="chatml",
    ),
}

# Default model - Llama 3.2 follows few-shot examples well for casual texts
DEFAULT_MODEL = "llama-3.2-1b"


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
        return MODELS["lfm2.5-1.2b"]

    # Sort by quality (excellent > good > basic), then by size (larger is better)
    quality_order = {"excellent": 3, "good": 2, "basic": 1}
    candidates.sort(key=lambda s: (quality_order[s.quality], s.size_gb), reverse=True)

    return candidates[0]
