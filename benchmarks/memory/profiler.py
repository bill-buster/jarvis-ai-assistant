"""Memory profiler implementation for MLX models.

Workstream 1: Memory Profiler

Implements the MemoryProfiler protocol from contracts/memory.py.
Measures RSS, virtual memory, and Metal GPU memory during model loading.
"""

import gc
import logging
import time
from datetime import UTC, datetime
from typing import Any

import psutil

from contracts.memory import MemoryProfile

logger = logging.getLogger(__name__)

# MLX is only available on Apple Silicon
# Use lazy imports to allow testing on other platforms
_mlx_available = False
_mx: Any = None
_load: Any = None

try:
    import mlx.core as _mx_module
    from mlx_lm import load as _load_module

    _mx = _mx_module
    _load = _load_module
    _mlx_available = True
except ImportError:
    logger.debug("MLX not available - memory profiler will use fallback mode")

# Constants
BYTES_PER_MB = 1024 * 1024


def _extract_model_info(model_path: str) -> tuple[str, str]:
    """Extract model name and quantization from path.

    Args:
        model_path: HuggingFace model path like "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    Returns:
        Tuple of (model_name, quantization)
    """
    # Extract the model identifier from the path
    parts = model_path.split("/")
    model_id = parts[-1] if parts else model_path

    # Try to extract quantization from common patterns
    quantization = "unknown"
    for quant in ["4bit", "8bit", "fp16", "bf16", "fp32"]:
        if quant in model_id.lower():
            quantization = quant
            break

    return model_id, quantization


def _get_metal_memory_mb() -> float:
    """Get current Metal GPU memory usage in MB.

    Returns:
        Metal memory in MB, or 0.0 if unavailable
    """
    if not _mlx_available or _mx is None:
        return 0.0
    try:
        # Get Metal memory stats
        # Note: MLX exposes peak_memory and active_memory
        peak_memory = _mx.metal.get_peak_memory()
        return float(peak_memory) / BYTES_PER_MB
    except (AttributeError, RuntimeError):
        logger.debug("Metal memory stats not available")
        return 0.0


def _get_process_memory() -> tuple[float, float]:
    """Get current process memory usage.

    Returns:
        Tuple of (rss_mb, virtual_mb)
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / BYTES_PER_MB
    vms_mb = mem_info.vms / BYTES_PER_MB
    return rss_mb, vms_mb


def _unload_model() -> None:
    """Unload model and free all memory including GPU cache.

    Following the pattern from models/loader.py.
    """
    # Clear Metal GPU memory
    if _mlx_available and _mx is not None:
        try:
            _mx.metal.clear_cache()
        except (AttributeError, RuntimeError):
            logger.debug("Metal cache clear not available")

    # Force garbage collection
    gc.collect()


class MLXMemoryProfiler:
    """Memory profiler for MLX models.

    Implements the MemoryProfiler protocol from contracts/memory.py.

    Measures actual memory usage by loading the model, capturing metrics,
    and then unloading to stay within memory budget.
    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._baseline_rss: float = 0.0
        self._baseline_vms: float = 0.0

    def _capture_baseline(self) -> None:
        """Capture baseline memory before loading model."""
        _unload_model()  # Ensure clean state
        self._baseline_rss, self._baseline_vms = _get_process_memory()
        logger.info(
            "Baseline memory - RSS: %.1fMB, VMS: %.1fMB",
            self._baseline_rss,
            self._baseline_vms,
        )

    def profile_model(self, model_path: str, context_length: int) -> MemoryProfile:
        """Profile a model's memory usage.

        Loads the model, measures memory, then unloads to free memory.
        CRITICAL: Always unloads model after profiling for 8GB safety.

        Args:
            model_path: HuggingFace model path or local path
            context_length: Context window size (affects KV cache allocation)

        Returns:
            MemoryProfile with measured memory usage
        """
        logger.info("Profiling model: %s at context_length: %d", model_path, context_length)

        # Extract model info
        model_name, quantization = _extract_model_info(model_path)

        # Capture baseline memory
        self._capture_baseline()

        # Reset peak memory counter before loading
        if _mlx_available and _mx is not None:
            try:
                _mx.metal.reset_peak_memory()
            except (AttributeError, RuntimeError):
                logger.debug("Metal reset_peak_memory not available")

        if not _mlx_available or _load is None:
            msg = "MLX is not available. Memory profiler requires Apple Silicon with MLX installed."
            raise RuntimeError(msg)

        # Load the model and measure time
        start_time = time.perf_counter()
        try:
            model, tokenizer = _load(model_path)
            load_time = time.perf_counter() - start_time

            # Force evaluation to ensure model is fully loaded
            _mx.eval(model.parameters())

            # Capture memory after load
            rss_mb, vms_mb = _get_process_memory()
            metal_mb = _get_metal_memory_mb()

            # Calculate delta from baseline
            rss_delta = rss_mb - self._baseline_rss
            vms_delta = vms_mb - self._baseline_vms

            logger.info(
                "Model loaded in %.2fs - RSS: %.1fMB (+%.1fMB), Metal: %.1fMB",
                load_time,
                rss_mb,
                rss_delta,
                metal_mb,
            )

            profile = MemoryProfile(
                model_name=model_name,
                quantization=quantization,
                context_length=context_length,
                rss_mb=rss_delta,  # Report delta, not absolute
                virtual_mb=vms_delta,
                metal_mb=metal_mb,
                load_time_seconds=load_time,
                timestamp=datetime.now(UTC).isoformat(),
            )

        except FileNotFoundError:
            logger.error("Model not found: %s", model_path)
            raise
        except Exception:
            logger.exception("Error profiling model: %s", model_path)
            raise
        finally:
            # CRITICAL: Always unload model to stay within memory budget
            logger.info("Unloading model for memory safety")
            model = None  # noqa: F841
            tokenizer = None  # noqa: F841
            _unload_model()

        return profile

    def profile_with_generation(
        self,
        model_path: str,
        context_length: int,
        prompt: str = "Hello",
        max_tokens: int = 10,
    ) -> MemoryProfile:
        """Profile memory during model loading and generation.

        This measures peak memory including KV cache allocation during inference.

        Args:
            model_path: HuggingFace model path or local path
            context_length: Context window size
            prompt: Test prompt for generation
            max_tokens: Maximum tokens to generate

        Returns:
            MemoryProfile with measured memory usage including generation
        """
        logger.info(
            "Profiling model with generation: %s at context_length: %d",
            model_path,
            context_length,
        )

        # Extract model info
        model_name, quantization = _extract_model_info(model_path)

        # Capture baseline memory
        self._capture_baseline()

        # Reset peak memory counter
        if _mlx_available and _mx is not None:
            try:
                _mx.metal.reset_peak_memory()
            except (AttributeError, RuntimeError):
                pass

        if not _mlx_available or _load is None:
            msg = "MLX is not available. Memory profiler requires Apple Silicon with MLX installed."
            raise RuntimeError(msg)

        start_time = time.perf_counter()
        try:
            model, tokenizer = _load(model_path)
            load_time = time.perf_counter() - start_time

            # Run a generation to allocate KV cache
            from mlx_lm import generate

            _ = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Capture peak memory after generation
            rss_mb, vms_mb = _get_process_memory()
            metal_mb = _get_metal_memory_mb()

            rss_delta = rss_mb - self._baseline_rss
            vms_delta = vms_mb - self._baseline_vms

            profile = MemoryProfile(
                model_name=model_name,
                quantization=quantization,
                context_length=context_length,
                rss_mb=rss_delta,
                virtual_mb=vms_delta,
                metal_mb=metal_mb,
                load_time_seconds=load_time,
                timestamp=datetime.now(UTC).isoformat(),
            )

        finally:
            # CRITICAL: Always unload
            model = None  # noqa: F841
            tokenizer = None  # noqa: F841
            _unload_model()

        return profile
