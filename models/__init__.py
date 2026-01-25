"""Model loading and generation (Workstream 8).

Provides MLX-based text generation with template fallback,
RAG context injection, and few-shot prompt formatting.
"""

import threading

from models.generator import MLXGenerator
from models.loader import MLXModelLoader, ModelConfig
from models.prompt_builder import PromptBuilder
from models.templates import (
    ResponseTemplate,
    SentenceModelError,
    TemplateMatch,
    TemplateMatcher,
)

__all__ = [
    "MLXGenerator",
    "MLXModelLoader",
    "ModelConfig",
    "PromptBuilder",
    "ResponseTemplate",
    "SentenceModelError",
    "TemplateMatcher",
    "TemplateMatch",
    "get_generator",
]

# Singleton generator instance with thread-safe initialization
_generator: MLXGenerator | None = None
_generator_lock = threading.Lock()


def get_generator() -> MLXGenerator:
    """Get or create singleton generator instance.

    Thread-safe using double-check locking pattern.

    Returns:
        The shared MLXGenerator instance
    """
    global _generator
    if _generator is None:
        with _generator_lock:
            # Double-check after acquiring lock
            if _generator is None:
                _generator = MLXGenerator()
    return _generator
