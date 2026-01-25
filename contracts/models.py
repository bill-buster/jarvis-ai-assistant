"""Model loading and generation interfaces.

Workstream 8 implements against these contracts.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    context_documents: list[str]  # RAG context to inject
    few_shot_examples: list[tuple[str, str]]  # (input, output) pairs
    max_tokens: int = 100
    temperature: float = 0.7
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
    finish_reason: str  # "stop", "length", "template"


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
