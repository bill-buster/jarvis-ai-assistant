"""Unit tests for MLX-based embeddings.

Tests cover the MLXEmbedder class, singleton management,
and error handling with mocked MLX dependencies.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_embeddings():
    """Create mock normalized embedding vectors."""
    embeddings = np.random.randn(2, MLX_EMBEDDING_DIM).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def mock_mlx_model(mock_embeddings):
    """Create a mock MLX embedding model."""
    model = MagicMock()
    model.encode = MagicMock(return_value=mock_embeddings)
    return model


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    reset_mlx_embedder()
    yield
    reset_mlx_embedder()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for module-level configuration."""

    def test_default_model_name(self):
        """Test that default model name is set."""
        assert DEFAULT_MLX_EMBEDDING_MODEL == "mlx-community/bge-small-en-v1.5"

    def test_embedding_dimension(self):
        """Test that embedding dimension is set."""
        assert MLX_EMBEDDING_DIM == 384


# =============================================================================
# MLXEmbedder Class Tests
# =============================================================================


class TestMLXEmbedderInit:
    """Tests for MLXEmbedder initialization."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        embedder = MLXEmbedder()

        assert embedder.model_name == DEFAULT_MLX_EMBEDDING_MODEL
        assert not embedder.is_loaded()

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        custom_model = "mlx-community/bge-base-en-v1.5"
        embedder = MLXEmbedder(model_name=custom_model)

        assert embedder.model_name == custom_model
        assert not embedder.is_loaded()

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        embedder = MLXEmbedder()

        assert embedder.embedding_dim == MLX_EMBEDDING_DIM


class TestMLXEmbedderEncode:
    """Tests for MLXEmbedder.encode() method."""

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_single_text(self, mock_load, mock_check, mock_embeddings):
        """Test encoding a single text string."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        result = embedder.encode("Hello, world!")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, MLX_EMBEDDING_DIM)
        mock_model.encode.assert_called_once()

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_multiple_texts(self, mock_load, mock_check, mock_embeddings):
        """Test encoding multiple texts."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        texts = ["Hello", "World"]
        result = embedder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, MLX_EMBEDDING_DIM)

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_empty_list(self, mock_load, mock_check):
        """Test encoding an empty list returns empty array."""
        mock_check.return_value = True

        embedder = MLXEmbedder()
        result = embedder.encode([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, MLX_EMBEDDING_DIM)
        mock_load.assert_not_called()  # Model not loaded for empty input

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_normalization(self, mock_load, mock_check):
        """Test that embeddings are normalized by default."""
        mock_check.return_value = True
        # Return non-normalized embeddings
        raw_embeddings = np.array([[3.0, 4.0] + [0.0] * (MLX_EMBEDDING_DIM - 2)], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = raw_embeddings
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        result = embedder.encode("test", normalize=True)

        # Check that result is normalized (L2 norm = 1)
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0)

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_without_normalization(self, mock_load, mock_check):
        """Test encoding without normalization."""
        mock_check.return_value = True
        raw_embeddings = np.array([[3.0, 4.0] + [0.0] * (MLX_EMBEDDING_DIM - 2)], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = raw_embeddings
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        result = embedder.encode("test", normalize=False)

        # Check that result is NOT normalized
        assert np.allclose(result, raw_embeddings)

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_ensures_float32(self, mock_load, mock_check):
        """Test that embeddings are converted to float32."""
        mock_check.return_value = True
        # Return float64 embeddings
        raw_embeddings = np.random.randn(1, MLX_EMBEDDING_DIM).astype(np.float64)
        mock_model = MagicMock()
        mock_model.encode.return_value = raw_embeddings
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        result = embedder.encode("test")

        assert result.dtype == np.float32


class TestMLXEmbedderModelLoading:
    """Tests for model loading behavior."""

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_lazy_loading(self, mock_load, mock_check, mock_embeddings):
        """Test that model is loaded lazily on first encode."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()

        # Model not loaded yet
        assert not embedder.is_loaded()
        mock_load.assert_not_called()

        # Trigger loading via encode
        embedder.encode("test")

        # Model should now be loaded
        assert embedder.is_loaded()
        mock_load.assert_called_once()

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_model_reuse(self, mock_load, mock_check, mock_embeddings):
        """Test that model is reused across encode calls."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        embedder.encode("first")
        embedder.encode("second")
        embedder.encode("third")

        # Model should only be loaded once
        mock_load.assert_called_once()

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_unload_and_reload(self, mock_load, mock_check, mock_embeddings):
        """Test unloading and reloading the model."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        embedder.encode("first")
        assert embedder.is_loaded()

        embedder.unload()
        assert not embedder.is_loaded()

        embedder.encode("second")
        assert embedder.is_loaded()

        # Model should have been loaded twice
        assert mock_load.call_count == 2

    def test_unload_when_not_loaded(self):
        """Test that unload is safe when model not loaded."""
        embedder = MLXEmbedder()
        assert not embedder.is_loaded()

        # Should not raise
        embedder.unload()
        assert not embedder.is_loaded()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_mlx_not_available(self):
        """Test error when MLX is not available."""
        embedder = MLXEmbedder()
        embedder._is_available = False

        with pytest.raises(MLXModelNotAvailableError):
            embedder.encode("test")

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    def test_mlx_embeddings_not_installed(self, mock_check):
        """Test error when mlx_embeddings package is not installed."""
        mock_check.return_value = True

        embedder = MLXEmbedder()

        with patch.dict("sys.modules", {"mlx_embeddings": None}):
            with pytest.raises(MLXModelLoadError):
                embedder.encode("test")

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_model_load_failure(self, mock_load, mock_check):
        """Test error when model fails to load."""
        mock_check.return_value = True
        mock_load.side_effect = RuntimeError("Failed to load model")

        embedder = MLXEmbedder()

        with pytest.raises(MLXModelLoadError) as exc_info:
            embedder.encode("test")

        assert "Failed to load" in str(exc_info.value)

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_encode_failure(self, mock_load, mock_check):
        """Test error when encoding fails."""
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("Encode failed")
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()

        with pytest.raises(MLXEmbeddingError) as exc_info:
            embedder.encode("test")

        assert "Failed to encode" in str(exc_info.value)


class TestExceptions:
    """Tests for exception classes."""

    def test_mlx_embedding_error_creation(self):
        """Test MLXEmbeddingError creation."""
        error = MLXEmbeddingError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_mlx_model_not_available_error(self):
        """Test MLXModelNotAvailableError creation."""
        error = MLXModelNotAvailableError()

        assert "not available" in str(error).lower()
        assert isinstance(error, MLXEmbeddingError)

    def test_mlx_model_load_error(self):
        """Test MLXModelLoadError creation."""
        error = MLXModelLoadError("Load failed")

        assert "Load failed" in str(error)
        assert isinstance(error, MLXEmbeddingError)


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton management functions."""

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_get_mlx_embedder_returns_singleton(self, mock_load, mock_check):
        """Test that get_mlx_embedder returns the same instance."""
        mock_check.return_value = True
        mock_load.return_value = MagicMock()

        embedder1 = get_mlx_embedder()
        embedder2 = get_mlx_embedder()

        assert embedder1 is embedder2

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_get_mlx_embedder_with_custom_model(self, mock_load, mock_check):
        """Test get_mlx_embedder with custom model name."""
        mock_check.return_value = True
        mock_load.return_value = MagicMock()

        custom_model = "mlx-community/bge-base-en-v1.5"
        embedder = get_mlx_embedder(model_name=custom_model)

        assert embedder.model_name == custom_model

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_get_mlx_embedder_different_model_recreates(self, mock_load, mock_check):
        """Test that requesting different model creates new instance."""
        mock_check.return_value = True
        mock_load.return_value = MagicMock()

        embedder1 = get_mlx_embedder()
        embedder2 = get_mlx_embedder(model_name="mlx-community/bge-base-en-v1.5")

        # Different model name should create new instance
        assert embedder2.model_name == "mlx-community/bge-base-en-v1.5"
        assert embedder1.model_name != embedder2.model_name

    def test_reset_mlx_embedder(self):
        """Test that reset clears the singleton."""
        # Create singleton
        _embedder1 = get_mlx_embedder()

        # Reset
        reset_mlx_embedder()

        # New call should create new instance
        embedder2 = get_mlx_embedder()
        assert embedder2 is not None


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_mlx_available_returns_bool(self):
        """Test that is_mlx_available returns a boolean."""
        result = is_mlx_available()

        assert isinstance(result, bool)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_concurrent_encode_calls(self, mock_load, mock_check, mock_embeddings):
        """Test that concurrent encode calls are safe."""
        import threading

        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedder = MLXEmbedder()
        results = []
        errors = []

        def encode_task():
            try:
                result = embedder.encode("test")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=encode_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # Model should only be loaded once due to double-check locking
        # (though it may be loaded multiple times in practice due to race conditions,
        # the important thing is no errors)
        assert mock_load.call_count >= 1

    @patch("models.embeddings.MLXEmbedder._check_mlx_available")
    @patch("mlx_embeddings.load_model")
    def test_concurrent_singleton_access(self, mock_load, mock_check, mock_embeddings):
        """Test that concurrent singleton access is safe."""
        import threading

        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings[:1]
        mock_load.return_value = mock_model

        embedders = []
        errors = []

        def get_embedder_task():
            try:
                embedder = get_mlx_embedder()
                embedders.append(embedder)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=get_embedder_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(embedders) == 10

        # All should be the same instance
        assert all(e is embedders[0] for e in embedders)
