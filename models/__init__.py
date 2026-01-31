"""Model loading and generation (Workstream 8).

Provides MLX-based text generation with template fallback,
RAG context injection, and few-shot prompt formatting.

Model Registry:
    from models import get_model_spec, get_recommended_model, MODEL_REGISTRY

    # Get best model for available RAM
    spec = get_recommended_model(16.0)  # 16GB RAM

    # Get specific model
    spec = get_model_spec("qwen-1.5b")

    # List all models
    for model_id, spec in MODEL_REGISTRY.items():
        print(f"{model_id}: {spec.display_name}")
"""

import threading

from models.embeddings import (
    DEFAULT_MLX_EMBEDDING_MODEL,
    MLX_EMBEDDING_DIM,
    MLXEmbedder,
    MLXEmbeddingError,
    MLXModelLoadError,
    MLXModelNotAvailableError,
    get_mlx_embedder,
    is_mlx_available,
    reset_mlx_embedder,
)
from models.generator import MLXGenerator
from models.loader import MLXModelLoader, ModelConfig
from models.prompt_builder import PromptBuilder
from models.registry import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    ModelSpec,
    ensure_model_available,
    get_all_models,
    get_model_spec,
    get_model_spec_by_path,
    get_recommended_model,
    is_model_available,
)
from models.templates import (
    ResponseTemplate,
    SentenceModelError,
    TemplateMatch,
    TemplateMatcher,
    is_sentence_model_loaded,
    unload_sentence_model,
)

__all__ = [
    # Generator
    "MLXGenerator",
    "MLXModelLoader",
    "ModelConfig",
    "PromptBuilder",
    # Registry
    "DEFAULT_MODEL_ID",
    "MODEL_REGISTRY",
    "ModelSpec",
    "ensure_model_available",
    "get_all_models",
    "get_model_spec",
    "get_model_spec_by_path",
    "get_recommended_model",
    "is_model_available",
    # MLX Embeddings
    "DEFAULT_MLX_EMBEDDING_MODEL",
    "MLX_EMBEDDING_DIM",
    "MLXEmbedder",
    "MLXEmbeddingError",
    "MLXModelLoadError",
    "MLXModelNotAvailableError",
    "get_mlx_embedder",
    "is_mlx_available",
    "reset_mlx_embedder",
    # Templates
    "ResponseTemplate",
    "SentenceModelError",
    "TemplateMatcher",
    "TemplateMatch",
    # Singleton functions
    "get_generator",
    "reset_generator",
    "unload_generator",
    "is_sentence_model_loaded",
    "unload_sentence_model",
]

# Singleton generator instance with thread-safe initialization
_generator: MLXGenerator | None = None
_generator_lock = threading.Lock()
_current_model_id: str | None = None


def get_generator(
    skip_templates: bool = True,
    model_id: str | None = None,
) -> MLXGenerator:
    """Get or create singleton generator instance.

    Thread-safe using double-check locking pattern.

    Args:
        skip_templates: If True (default), skip template matching to save memory.
                       Templates are disabled by default because they load a
                       separate model that consumes memory needed for the LLM.
        model_id: Optional model ID from registry. If different from current,
                 resets the generator to load the new model.

    Returns:
        The shared MLXGenerator instance
    """
    global _generator, _current_model_id

    # If requesting a different model, reset first
    if model_id is not None and _current_model_id != model_id:
        reset_generator()
        _current_model_id = model_id

    if _generator is None:
        with _generator_lock:
            # Double-check after acquiring lock
            if _generator is None:
                config = ModelConfig(model_id=model_id) if model_id else None
                _generator = MLXGenerator(config=config, skip_templates=skip_templates)
                _current_model_id = model_id
    return _generator


def unload_generator() -> None:
    """Unload the model from the current generator without resetting the singleton.

    Use this to free memory while keeping the generator instance.
    The model will be reloaded on the next generation request.
    """
    global _generator
    with _generator_lock:
        if _generator is not None:
            _generator.unload()


def reset_generator() -> None:
    """Reset the singleton generator instance and unload any loaded model.

    Use this to:
    - Clear state between tests
    - Switch to a different model configuration
    - Force complete reinitialization of the generator

    This function DOES unload any loaded models to prevent memory leaks.
    A new generator instance will be created on the next get_generator() call.
    """
    global _generator, _current_model_id
    with _generator_lock:
        if _generator is not None:
            # Unload model if loaded to prevent memory leaks
            _generator.unload()
        _generator = None
        _current_model_id = None
