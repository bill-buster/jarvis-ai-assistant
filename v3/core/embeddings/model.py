"""Embedding model loader for JARVIS v2.

Uses sentence-transformers for efficient text embeddings.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Singleton
_embedding_model: EmbeddingModel | None = None
_model_lock = threading.Lock()


@dataclass
class EmbeddingResult:
    """Result from embedding computation."""

    embedding: np.ndarray
    text: str
    model_id: str


class EmbeddingModel:
    """Lazy-loading embedding model wrapper."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        """Initialize embedding model.

        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return EMBEDDING_DIM

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded (thread-safe)."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            import time
            start = time.time()
            logger.info(f"Loading embedding model: {self.model_id}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_id)
                elapsed = time.time() - start
                logger.info(f"Embedding model loaded in {elapsed:.1f}s")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def preload(self) -> None:
        """Explicitly load the model (for eager initialization)."""
        self._ensure_loaded()

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (np.ndarray)
        """
        self._ensure_loaded()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Compute embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._ensure_loaded()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.astype(np.float32) for e in embeddings]

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._load_lock:
            if self._model is not None:
                logger.info("Unloading embedding model")
                self._model = None


def get_embedding_model(model_id: str | None = None) -> EmbeddingModel:
    """Get singleton embedding model.

    Args:
        model_id: Model to use (defaults to all-MiniLM-L6-v2)

    Returns:
        EmbeddingModel instance
    """
    global _embedding_model

    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = EmbeddingModel(model_id or DEFAULT_MODEL)

    return _embedding_model


def reset_embedding_model() -> None:
    """Reset the embedding model singleton."""
    global _embedding_model
    with _model_lock:
        if _embedding_model is not None:
            _embedding_model.unload()
            _embedding_model = None
