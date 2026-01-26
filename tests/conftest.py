"""Pytest configuration for JARVIS tests.

Handles mocking of platform-specific dependencies (MLX) to allow tests
to run on non-macOS platforms.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


def _mock_mlx_modules():
    """Mock MLX modules before they are imported by the tests.

    MLX is only available on macOS Apple Silicon. These mocks allow
    basic tests to run on other platforms (Linux, Intel Mac, Windows).
    Note: Tests that require sentence_transformers will be skipped.
    """
    # Create mock for top-level mlx package (required for mlx.core to work)
    mock_mlx = types.ModuleType("mlx")
    mock_mlx.__path__ = []  # Make it a package

    # Create mock for mlx.core
    mock_mx = MagicMock()
    mock_mx.metal = MagicMock()
    mock_mx.metal.clear_cache = MagicMock()

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_mlx_lm.generate = MagicMock(return_value="Generated text")

    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler = MagicMock(return_value=MagicMock())

    # Install mocks in sys.modules before any imports
    # Need to mock 'mlx' top-level for 'mlx.core' to be importable
    sys.modules["mlx"] = mock_mlx
    sys.modules["mlx.core"] = mock_mx
    sys.modules["mlx_lm"] = mock_mlx_lm
    sys.modules["mlx_lm.sample_utils"] = mock_sample_utils


# Check if MLX is actually usable (not just installed)
# The package may be installed but fail to load on non-macOS/non-Apple-Silicon
try:
    import mlx.core  # noqa: F401

    MLX_AVAILABLE = True
except (ImportError, OSError):
    MLX_AVAILABLE = False

# Mock MLX modules if not usable - MUST happen before any imports
if not MLX_AVAILABLE:
    _mock_mlx_modules()

# Check if sentence_transformers can load properly (AFTER mocking MLX)
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ValueError):
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Pytest marker for tests that require sentence_transformers
requires_sentence_transformers = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available (requires macOS Apple Silicon with MLX)",
)
