"""MLX Embedding Service Client - Low-level GPU-accelerated embeddings.

This is the LOW-LEVEL MLX embedding service client. It communicates with
a separate MLX microservice via HTTP for GPU-accelerated embeddings.

IMPORTANT: Most code should NOT import from this module directly.
Instead, use the unified interface in jarvis/embedding_adapter.py:

    from jarvis.embedding_adapter import get_embedder
    embedder = get_embedder()  # Auto-selects MLX or CPU fallback

Architecture (3-layer embedding stack):
    1. jarvis/embeddings.py       - Embedding STORAGE (SQLite-backed message search)
    2. jarvis/embedding_adapter.py - UNIFIED INTERFACE (MLX-first with CPU fallback)
    3. models/embeddings.py       - MLX SERVICE CLIENT (this file, low-level)

This module is appropriate for:
- Direct MLX service health checks
- Starting/managing the MLX embedding service
- Low-level MLX-specific operations

Key Features:
- GPU acceleration on Apple Silicon via MLX
- Thread-safe singleton pattern
- Automatic service health checking

Service Setup:
    The MLX embedding service runs separately at ~/.jarvis/mlx-embed-service/
    Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jarvis.errors import ErrorCode, JarvisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MLX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MLX_EMBEDDING_DIM = 384  # Dimension for bge-small-en-v1.5
MLX_EMBED_SERVICE_URL = "http://127.0.0.1:8766"
MLX_EMBED_SERVICE_DIR = Path.home() / ".jarvis" / "mlx-embed-service"


# =============================================================================
# Exceptions
# =============================================================================


class MLXEmbeddingError(JarvisError):
    """Raised when MLX embedding operations fail."""

    default_message = "MLX embedding operation failed"
    default_code = ErrorCode.MDL_LOAD_FAILED


class MLXServiceNotAvailableError(MLXEmbeddingError):
    """Raised when MLX embedding service is not running."""

    default_message = "MLX embedding service is not available"


class MLXModelLoadError(MLXEmbeddingError):
    """Raised when MLX embedding model fails to load."""

    default_message = "Failed to load MLX embedding model"


# For backwards compatibility
MLXModelNotAvailableError = MLXServiceNotAvailableError


# =============================================================================
# MLX Embedder (Service Client)
# =============================================================================


class MLXEmbedder:
    """Client for MLX embedding microservice.

    Communicates with a separate MLX embedding service to generate
    embeddings. The service runs in its own environment to avoid
    dependency conflicts.

    Thread Safety:
        This class is thread-safe for concurrent encode() calls.

    Example:
        >>> embedder = MLXEmbedder()
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(embeddings.shape)
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_EMBEDDING_MODEL,
        service_url: str = MLX_EMBED_SERVICE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the MLX embedder client.

        Args:
            model_name: Name/path of the embedding model to use.
            service_url: URL of the MLX embedding service.
            timeout: Request timeout in seconds.
        """
        self.model_name = model_name
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._service_available: bool | None = None
        self._last_health_check: float = 0

    def _check_service_health(self, force: bool = False) -> bool:
        """Check if the embedding service is healthy.

        Args:
            force: Force a fresh health check even if cached.

        Returns:
            True if service is healthy, False otherwise.
        """
        # Cache health check for 30 seconds
        now = time.time()
        if not force and self._service_available is not None:
            if now - self._last_health_check < 30:
                return self._service_available

        try:
            import urllib.request

            url = f"{self.service_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    self._service_available = True
                    self._last_health_check = now
                    return True
        except Exception as e:
            logger.debug("Health check failed: %s", e)

        self._service_available = False
        self._last_health_check = now
        return False

    def is_available(self) -> bool:
        """Check if the MLX embedding service is available.

        Returns:
            True if service is running and healthy.
        """
        return self._check_service_health()

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts into embedding vectors.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).

        Raises:
            MLXServiceNotAvailableError: If service is not running.
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

        # Check service availability
        if not self._check_service_health():
            raise MLXServiceNotAvailableError(
                "MLX embedding service is not running. "
                "Start it with: cd ~/.jarvis/mlx-embed-service && uv run python server.py"
            )

        try:
            import json
            import urllib.request

            # Prepare request
            url = f"{self.service_url}/embed"
            data = json.dumps(
                {
                    "texts": texts,
                    "normalize": normalize,
                    "model": self.model_name,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Make request
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Convert to numpy array
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            return embeddings

        except MLXServiceNotAvailableError:
            raise
        except Exception as e:
            logger.exception("Failed to encode texts via MLX service")
            raise MLXEmbeddingError(
                f"Failed to encode texts: {e}",
                cause=e,
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded in the service.

        Returns:
            True if model is loaded in the service.
        """
        if not self._check_service_health():
            return False

        try:
            import json
            import urllib.request

            url = f"{self.service_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                return bool(result.get("model_loaded", False))
        except Exception:
            return False

    def unload(self) -> None:
        """Request the service to unload the model.

        Sends unload request to free GPU memory.
        """
        if not self._check_service_health():
            return

        try:
            import urllib.request

            url = f"{self.service_url}/unload"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5):
                pass
            logger.info("Requested model unload from MLX service")
        except Exception as e:
            logger.warning("Failed to unload model: %s", e)

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

        # Create new embedder
        _mlx_embedder = MLXEmbedder(model_name or DEFAULT_MLX_EMBEDDING_MODEL)
        return _mlx_embedder


def reset_mlx_embedder() -> None:
    """Reset the singleton MLX embedder.

    Clears the singleton. A new instance will be created on
    the next get_mlx_embedder() call.
    """
    global _mlx_embedder

    with _mlx_embedder_lock:
        if _mlx_embedder is not None:
            _mlx_embedder.unload()
        _mlx_embedder = None


def is_mlx_available() -> bool:
    """Check if MLX embedding service is available.

    Returns:
        True if the MLX embedding service is running.
    """
    embedder = get_mlx_embedder()
    return embedder.is_available()


def start_mlx_service() -> subprocess.Popen[bytes] | None:
    """Start the MLX embedding service if not running.

    Returns:
        Popen object if service was started, None if already running
        or if service directory doesn't exist.
    """
    # Check if already running
    embedder = get_mlx_embedder()
    if embedder.is_available():
        logger.info("MLX embedding service already running")
        return None

    # Check if service directory exists
    if not MLX_EMBED_SERVICE_DIR.exists():
        logger.warning(
            "MLX embedding service not installed at %s",
            MLX_EMBED_SERVICE_DIR,
        )
        return None

    # Start the service
    logger.info("Starting MLX embedding service...")
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "server.py"],
            cwd=MLX_EMBED_SERVICE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a bit for startup
        time.sleep(2)

        # Check if it started successfully
        if embedder._check_service_health(force=True):
            logger.info("MLX embedding service started successfully")
            return process
        else:
            logger.warning("MLX embedding service failed to start")
            process.terminate()
            return None

    except Exception as e:
        logger.exception("Failed to start MLX embedding service: %s", e)
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MLX_EMBEDDING_MODEL",
    "MLX_EMBEDDING_DIM",
    "MLX_EMBED_SERVICE_URL",
    # Exceptions
    "MLXEmbeddingError",
    "MLXServiceNotAvailableError",
    "MLXModelNotAvailableError",  # backwards compat
    "MLXModelLoadError",
    # Class
    "MLXEmbedder",
    # Singleton functions
    "get_mlx_embedder",
    "reset_mlx_embedder",
    "is_mlx_available",
    "start_mlx_service",
]
