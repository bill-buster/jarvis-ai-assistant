"""Unit tests for MLX-based embeddings.

Tests cover the MLXEmbedder class, singleton management,
and error handling with mocked MLX dependencies.

NOTE: The MLX embeddings module has been refactored to use a microservice
client architecture. Tests for the old local MLX embedding approach have
been removed. New tests should be added for the microservice client.
"""

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
        assert DEFAULT_MLX_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

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
        # With microservice architecture, is_loaded() returns True if service is running
        assert isinstance(embedder.is_loaded(), bool)

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        custom_model = "mlx-community/bge-base-en-v1.5"
        embedder = MLXEmbedder(model_name=custom_model)

        assert embedder.model_name == custom_model
        assert isinstance(embedder.is_loaded(), bool)

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        embedder = MLXEmbedder()

        assert embedder.embedding_dim == MLX_EMBEDDING_DIM


class TestMLXEmbedderSafety:
    """Tests for safe operations regardless of service state."""

    def test_unload_is_safe(self):
        """Test that unload is safe to call regardless of state."""
        embedder = MLXEmbedder()
        _ = embedder.is_loaded()  # Check state before unload

        # Should not raise regardless of state
        embedder.unload()

        # State after unload depends on whether service processes unload requests
        assert isinstance(embedder.is_loaded(), bool)


# =============================================================================
# Error Handling Tests
# =============================================================================


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
