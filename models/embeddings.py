"""MLX-based embeddings for local inference on Apple Silicon.

Provides fast, memory-efficient embedding computation using MLX models.
This is an alternative to sentence-transformers for environments where
MLX acceleration is preferred.

Key Features:
- Native MLX acceleration on Apple Silicon
- Thread-safe singleton pattern
- Normalized embeddings by default
- Compatible with bge, e5, and other MLX embedding models

Usage:
    from models.embeddings import get_mlx_embedder, MLXEmbedder

    # Get singleton embedder
    embedder = get_mlx_embedder()

    # Encode texts
    embeddings = embedder.encode(["Hello, world!", "How are you?"])

    # Encode without normalization
    embeddings = embedder.encode(texts, normalize=False)

Note:
    Requires mlx-embeddings package: pip install mlx-embeddings
    Only available on Apple Silicon Macs.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from jarvis.errors import ErrorCode, JarvisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MLX_EMBEDDING_MODEL = "mlx-community/bge-small-en-v1.5"
MLX_EMBEDDING_DIM = 384  # Dimension for bge-small-en-v1.5


# =============================================================================
# Exceptions
# =============================================================================


class MLXEmbeddingError(JarvisError):
    """Raised when MLX embedding operations fail."""

    default_message = "MLX embedding operation failed"
    default_code = ErrorCode.MDL_LOAD_FAILED


class MLXModelNotAvailableError(MLXEmbeddingError):
    """Raised when MLX is not available (non-Apple Silicon)."""

    default_message = "MLX is not available on this platform"


class MLXModelLoadError(MLXEmbeddingError):
    """Raised when MLX embedding model fails to load."""

    default_message = "Failed to load MLX embedding model"


# =============================================================================
# MLX Embedder
# =============================================================================


class MLXEmbedder:
    """MLX-based embedding model for Apple Silicon.

    Provides text embedding using MLX-optimized models for fast
    inference on Apple Silicon. Uses lazy loading to defer model
    initialization until first use.

    Thread Safety:
        This class is thread-safe. The model is loaded using
        double-check locking to ensure safe initialization.

    Example:
        >>> embedder = MLXEmbedder()
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(embeddings.shape)
        (2, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MLX_EMBEDDING_MODEL) -> None:
        """Initialize the MLX embedder.

        Args:
            model_name: Name/path of the MLX embedding model to use.
                       Defaults to "mlx-community/bge-small-en-v1.5".

        Note:
            The model is not loaded until the first call to encode().
            This allows for fast initialization and deferred resource usage.
        """
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._is_available: bool | None = None

    def _check_mlx_available(self) -> bool:
        """Check if MLX is available on this system.

        Returns:
            True if MLX is available, False otherwise.
        """
        if self._is_available is not None:
            return self._is_available

        try:
            import mlx.core  # noqa: F401

            self._is_available = True
        except ImportError:
            self._is_available = False
            logger.warning("MLX is not available on this system")

        return self._is_available

    def _load_model(self) -> None:
        """Load the MLX embedding model.

        Uses double-check locking for thread-safe initialization.

        Raises:
            MLXModelNotAvailableError: If MLX is not available.
            MLXModelLoadError: If model fails to load.
        """
        # Fast path: already loaded
        if self._model is not None:
            return

        # Slow path: acquire lock and load
        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            if not self._check_mlx_available():
                raise MLXModelNotAvailableError(
                    "MLX is not available. This feature requires Apple Silicon."
                )

            try:
                from mlx_embeddings import load_model

                logger.info("Loading MLX embedding model: %s", self.model_name)
                self._model = load_model(self.model_name)
                logger.debug("MLX embedding model loaded successfully")

            except ImportError as e:
                raise MLXModelLoadError(
                    "mlx-embeddings package not installed. "
                    "Install with: pip install mlx-embeddings",
                    cause=e,
                ) from e

            except Exception as e:
                logger.exception("Failed to load MLX embedding model: %s", self.model_name)
                raise MLXModelLoadError(
                    f"Failed to load MLX embedding model '{self.model_name}': {e}",
                    cause=e,
                ) from e

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts into embedding vectors.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.
                      Normalized embeddings enable cosine similarity via dot product.

        Returns:
            NumPy array of shape (n_texts, embedding_dim) containing the embeddings.
            If a single string was provided, returns shape (1, embedding_dim).

        Raises:
            MLXModelNotAvailableError: If MLX is not available.
            MLXModelLoadError: If model fails to load.
            MLXEmbeddingError: If encoding fails.

        Example:
            >>> embedder = MLXEmbedder()
            >>> vecs = embedder.encode(["Hello", "World"])
            >>> similarity = np.dot(vecs[0], vecs[1])  # Cosine similarity
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, MLX_EMBEDDING_DIM)

        # Ensure model is loaded
        self._load_model()

        try:
            # Generate embeddings using the model
            embeddings = self._model.encode(texts)

            # Convert to numpy if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)

            # Ensure float32 dtype
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.maximum(norms, 1e-12)
                embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            logger.exception("Failed to encode texts with MLX")
            raise MLXEmbeddingError(
                f"Failed to encode texts: {e}",
                cause=e,
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded.

        Returns:
            True if the model is currently loaded in memory.
        """
        return self._model is not None

    def unload(self) -> None:
        """Unload the model to free memory.

        Safe to call even if model is not loaded.
        The model will be reloaded on the next encode() call.
        """
        with self._lock:
            if self._model is not None:
                logger.info("Unloading MLX embedding model")
                self._model = None

                # Attempt to free MLX memory
                try:
                    import gc

                    import mlx.core as mx

                    gc.collect()
                    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                        mx.metal.clear_cache()
                except Exception:
                    pass  # Best effort cleanup

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of the embedding vectors.
        """
        return MLX_EMBEDDING_DIM


# =============================================================================
# Singleton Access
# =============================================================================

_mlx_embedder: MLXEmbedder | None = None
_mlx_embedder_lock = threading.Lock()


def get_mlx_embedder(model_name: str | None = None) -> MLXEmbedder:
    """Get or create the singleton MLX embedder instance.

    Args:
        model_name: Optional model name. If provided and different from
                   the current model, a new embedder will be created.

    Returns:
        The shared MLXEmbedder instance.

    Example:
        >>> embedder = get_mlx_embedder()
        >>> embeddings = embedder.encode(["Hello, world!"])
    """
    global _mlx_embedder

    # Fast path: singleton exists and model matches
    if _mlx_embedder is not None:
        if model_name is None or model_name == _mlx_embedder.model_name:
            return _mlx_embedder

    with _mlx_embedder_lock:
        # Double-check after acquiring lock
        if _mlx_embedder is not None:
            if model_name is None or model_name == _mlx_embedder.model_name:
                return _mlx_embedder
            # Different model requested, unload old one
            _mlx_embedder.unload()

        # Create new embedder
        _mlx_embedder = MLXEmbedder(model_name or DEFAULT_MLX_EMBEDDING_MODEL)
        return _mlx_embedder


def reset_mlx_embedder() -> None:
    """Reset the singleton MLX embedder.

    Unloads the model and clears the singleton. A new instance
    will be created on the next get_mlx_embedder() call.
    """
    global _mlx_embedder

    with _mlx_embedder_lock:
        if _mlx_embedder is not None:
            _mlx_embedder.unload()
        _mlx_embedder = None


def is_mlx_available() -> bool:
    """Check if MLX is available on this system.

    Returns:
        True if MLX can be imported, False otherwise.
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MLX_EMBEDDING_MODEL",
    "MLX_EMBEDDING_DIM",
    # Exceptions
    "MLXEmbeddingError",
    "MLXModelNotAvailableError",
    "MLXModelLoadError",
    # Class
    "MLXEmbedder",
    # Singleton functions
    "get_mlx_embedder",
    "reset_mlx_embedder",
    "is_mlx_available",
]
