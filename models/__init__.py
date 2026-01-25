"""Model loading and generation (Workstream 8).

Provides MLX-based text generation with template fallback,
RAG context injection, and few-shot prompt formatting.
"""

from models.generator import MLXGenerator
from models.loader import MLXModelLoader, ModelConfig
from models.prompt_builder import PromptBuilder
from models.templates import ResponseTemplate, TemplateMatcher, TemplateMatch, SentenceModelError

__all__ = [
    "MLXGenerator",
    "MLXModelLoader",
    "ModelConfig",
    "PromptBuilder",
    "ResponseTemplate",
    "SentenceModelError",
    "TemplateMatcher",
    "TemplateMatch",
]

# Singleton generator instance
_generator: MLXGenerator | None = None


def get_generator() -> MLXGenerator:
    """Get or create singleton generator instance.

    Returns:
        The shared MLXGenerator instance
    """
    global _generator
    if _generator is None:
        _generator = MLXGenerator()
    return _generator
