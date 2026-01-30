"""Model loading and generation interfaces.

Workstream 8 implements against these contracts.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationRequest:
    """Request for text generation.

    Default parameters are optimized for LFM2.5-1.2B-Instruct.
    Source: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
    """

    prompt: str
    context_documents: list[str]  # RAG context to inject
    few_shot_examples: list[tuple[str, str]]  # (input, output) pairs
    max_tokens: int = 100
    # LFM2.5-1.2B-Instruct optimal parameters
    temperature: float = 0.1  # Low temp for focused, consistent output
    top_p: float = 0.1  # Nucleus sampling threshold
    top_k: int = 50  # Limit vocabulary to top-k tokens
    repetition_penalty: float = 1.05  # Slight penalty to avoid repetition
    stop_sequences: list[str] | None = None


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    tokens_used: int
    generation_time_ms: float
    model_name: str
    used_template: bool
    template_name: str | None
    finish_reason: str  # "stop", "length", "template", "fallback", "error"
    error: str | None = None  # Error message if finish_reason is "error" or "fallback"


class Generator(Protocol):
    """Interface for text generation (Workstream 8)."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response. May use template or model."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        ...

    def load(self) -> bool:
        """Load model into memory. Returns success."""
        ...

    def unload(self) -> None:
        """Unload model to free memory."""
        ...

    def get_memory_usage_mb(self) -> float:
        """Return current memory usage of the model."""
        ...
