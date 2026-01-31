"""Unified Embedding Adapter - THE CANONICAL INTERFACE for embeddings.

This is the SINGLE SOURCE OF TRUTH for embedding computation in JARVIS.
All modules that need to compute embeddings should import from here.

Provides a unified interface that:
- Tries MLX first (GPU-accelerated on Apple Silicon)
- Falls back to SentenceTransformer (CPU) if MLX service unavailable
- Uses consistent model (bge-small-en-v1.5) regardless of backend
- Ensures compatible similarity scores and index compatibility

Architecture (3-layer embedding stack):
    1. jarvis/embeddings.py       - Embedding STORAGE (SQLite-backed message search)
           Uses get_embedder() from this module
    2. jarvis/embedding_adapter.py - UNIFIED INTERFACE (this file, use this!)
           Wraps MLX service client with CPU fallback
    3. models/embeddings.py       - MLX SERVICE CLIENT (low-level, internal)
           Direct HTTP client to MLX microservice

Usage:
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    print(f"Using backend: {embedder.backend}")  # 'mlx' or 'cpu'

    # Encode texts
    embeddings = embedder.encode(["Hello, world!", "How are you?"])

    # Single text
    embedding = embedder.encode("Hello, world!")  # Returns (1, 384) array

    # With caching (for repeated queries in a request)
    from jarvis.embedding_adapter import CachedEmbedder
    cached = CachedEmbedder(embedder, maxsize=256)
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

# Disable HuggingFace hub network checks after initial download
# This prevents slow version checks on every embedding computation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Use bge-small-en-v1.5 for consistency across all modules
# This was already used by cluster.py and index.py
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Batch size for MLX service requests (avoid request size limits)
MLX_BATCH_SIZE = 100


# =============================================================================
# Unified Embedder
# =============================================================================


class UnifiedEmbedder:
    """MLX-first embedder with SentenceTransformer fallback.

    Provides a consistent embedding interface regardless of whether
    the MLX embedding service is available.

    Thread Safety:
        This class is thread-safe for concurrent encode() calls.

    Example:
        >>> embedder = UnifiedEmbedder()
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(f"Backend: {embedder.backend}, Shape: {embeddings.shape}")
        Backend: mlx, Shape: (2, 384)
    """

    def __init__(self) -> None:
        """Initialize the unified embedder."""
        self._mlx_embedder: Any = None
        self._sentence_model: Any = None
        self._backend: str | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def _initialize_backend(self) -> None:
        """Initialize the embedding backend (MLX or SentenceTransformer).

        Tries MLX first, falls back to SentenceTransformer if unavailable.
        """
        if self._init_attempted:
            return

        with self._lock:
            if self._init_attempted:
                return

            self._init_attempted = True

            # Try MLX first (GPU-accelerated on Apple Silicon)
            try:
                from models.embeddings import get_mlx_embedder

                mlx_embedder = get_mlx_embedder()
                if mlx_embedder.is_available():
                    self._mlx_embedder = mlx_embedder
                    self._backend = "mlx"
                    logger.info("Using MLX embedding backend (GPU-accelerated)")
                    return
                else:
                    logger.debug("MLX embedding service not available")
            except Exception as e:
                logger.debug("Could not initialize MLX embedder: %s", e)

            # Fall back to SentenceTransformer (CPU)
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading SentenceTransformer: %s (CPU fallback)", EMBEDDING_MODEL)
                # Use local_files_only to skip network checks (model must be pre-downloaded)
                try:
                    self._sentence_model = SentenceTransformer(
                        EMBEDDING_MODEL, local_files_only=True
                    )
                except Exception:
                    # First time download - allow network access
                    logger.info("Model not cached, downloading %s...", EMBEDDING_MODEL)
                    self._sentence_model = SentenceTransformer(EMBEDDING_MODEL)
                self._backend = "cpu"
                logger.info("Using SentenceTransformer backend (CPU)")
            except Exception as e:
                logger.exception("Failed to initialize any embedding backend: %s", e)
                raise RuntimeError(
                    f"Could not initialize embedding backend: {e}. "
                    "Ensure either the MLX embedding service is running or "
                    "sentence-transformers is installed."
                ) from e

    def is_available(self) -> bool:
        """Check if an embedding backend is available.

        Returns:
            True if either MLX or SentenceTransformer is available.
        """
        try:
            self._initialize_backend()
            return self._backend is not None
        except RuntimeError:
            return False

    @property
    def backend(self) -> str:
        """Get the current backend type.

        Returns:
            'mlx' if using MLX service, 'cpu' if using SentenceTransformer.
        """
        self._initialize_backend()
        return self._backend or "none"

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            The dimensionality of the embedding vectors (384 for bge-small-en-v1.5).
        """
        return EMBEDDING_DIM

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,  # Ignored, always numpy (compat with SentenceTransformer)
        normalize_embeddings: bool | None = None,  # Alias for normalize (compat)
    ) -> NDArray[np.float32]:
        """Encode texts into embedding vectors.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.
                      Normalized embeddings allow cosine similarity via dot product.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
            For single string input, still returns shape (1, 384).

        Raises:
            RuntimeError: If no embedding backend is available.

        Example:
            >>> embedder = UnifiedEmbedder()
            >>> vecs = embedder.encode(["Hello", "World"])
            >>> similarity = np.dot(vecs[0], vecs[1])  # Cosine similarity
        """
        self._initialize_backend()

        # Handle normalize_embeddings alias (SentenceTransformer compatibility)
        if normalize_embeddings is not None:
            normalize = normalize_embeddings

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        # Use appropriate backend
        if self._backend == "mlx" and self._mlx_embedder is not None:
            return self._encode_batched_mlx(texts, normalize=normalize)
        elif self._backend == "cpu" and self._sentence_model is not None:
            embeddings = self._sentence_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            return embeddings.astype(np.float32)
        else:
            raise RuntimeError("No embedding backend available")

    def _encode_batched_mlx(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts using MLX service with batching.

        Splits large requests into smaller batches to avoid service
        request size limits.

        Args:
            texts: List of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        if len(texts) <= MLX_BATCH_SIZE:
            # Small enough for single request
            return self._mlx_embedder.encode(texts, normalize=normalize)

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), MLX_BATCH_SIZE):
            batch = texts[i : i + MLX_BATCH_SIZE]
            batch_embeddings = self._mlx_embedder.encode(batch, normalize=normalize)
            all_embeddings.append(batch_embeddings)

            # Log progress for large batches
            if len(texts) > 1000 and (i + MLX_BATCH_SIZE) % 1000 == 0:
                logger.debug(
                    "Encoded %d/%d texts (%.1f%%)",
                    min(i + MLX_BATCH_SIZE, len(texts)),
                    len(texts),
                    100 * min(i + MLX_BATCH_SIZE, len(texts)) / len(texts),
                )

        return np.vstack(all_embeddings)

    def unload(self) -> None:
        """Unload the embedding model to free memory.

        For MLX, sends unload request to service.
        For SentenceTransformer, clears the model reference.
        """
        with self._lock:
            if self._mlx_embedder is not None:
                self._mlx_embedder.unload()

            if self._sentence_model is not None:
                self._sentence_model = None
                gc.collect()
                logger.info("Unloaded SentenceTransformer model")

            self._backend = None
            self._init_attempted = False


class CachedEmbedder:
    """Per-request embedding cache wrapper."""

    def __init__(self, base_embedder: UnifiedEmbedder, maxsize: int = 256) -> None:
        self.base = base_embedder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._computations = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _make_key(self, text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

    def _get(self, key: str) -> np.ndarray | None:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return value

    def _set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @property
    def embedding_computations(self) -> int:
        return self._computations

    @property
    def cache_hit(self) -> bool:
        return self._cache_hits > 0

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool | None = None,
    ) -> NDArray[np.float32]:
        if normalize_embeddings is not None:
            normalize = normalize_embeddings

        if isinstance(texts, str):
            key = self._make_key(texts)
            cached = self._get(key)
            if cached is not None:
                return cached

            result = self.base.encode(
                [texts],
                normalize=normalize,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            vector = result.reshape(1, -1)
            self._set(key, vector)
            self._computations += 1
            return vector

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        cached_vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for i, text in enumerate(texts):
            key = self._make_key(text)
            cached = self._get(key)
            if cached is not None:
                cached_vectors[i] = cached
            else:
                missing_texts.append(text)
                missing_indices.append(i)

        if missing_texts:
            embeddings = self.base.encode(
                missing_texts,
                normalize=normalize,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            for idx, emb in zip(missing_indices, embeddings):
                vector = np.asarray(emb, dtype=np.float32).reshape(1, -1)
                cached_vectors[idx] = vector
                self._set(self._make_key(texts[idx]), vector)
            self._computations += len(missing_texts)

        if any(v is None for v in cached_vectors):
            raise RuntimeError("Failed to compute embeddings for all inputs")

        return np.vstack(cached_vectors)


# =============================================================================
# Singleton Access
# =============================================================================

_embedder: UnifiedEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> UnifiedEmbedder:
    """Get or create the singleton UnifiedEmbedder instance.

    Returns:
        The shared UnifiedEmbedder instance.

    Example:
        >>> embedder = get_embedder()
        >>> embeddings = embedder.encode(["Hello, world!"])
    """
    global _embedder

    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = UnifiedEmbedder()

    return _embedder


def reset_embedder() -> None:
    """Reset the singleton UnifiedEmbedder.

    Unloads the model and clears the singleton.
    A new instance will be created on the next get_embedder() call.
    """
    global _embedder

    with _embedder_lock:
        if _embedder is not None:
            _embedder.unload()
        _embedder = None


def is_embedder_available() -> bool:
    """Check if an embedding backend is available.

    Returns:
        True if embeddings can be computed.
    """
    embedder = get_embedder()
    return embedder.is_available()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    # Class
    "UnifiedEmbedder",
    "CachedEmbedder",
    # Singleton functions
    "get_embedder",
    "reset_embedder",
    "is_embedder_available",
]
