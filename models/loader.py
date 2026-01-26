"""MLX Model Loader with memory-safe lifecycle management.

Handles model loading/unloading with thread-safety, memory tracking,
and double-check locking patterns for lazy initialization.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import psutil
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024


@dataclass
class ModelConfig:
    """Configuration for MLX model loading."""

    model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    estimated_memory_mb: float = 800
    memory_buffer_multiplier: float = 1.5
    default_max_tokens: int = 100
    default_temperature: float = 0.7


@dataclass
class GenerationResult:
    """Result from synchronous generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float


class MLXModelLoader:
    """MLX model lifecycle manager with thread-safe loading and memory tracking.

    Implements lazy loading with double-check locking, memory pressure checks,
    and complete unloading including Metal GPU cache cleanup.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the loader.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()
        self._loaded_at: float | None = None

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None and self._tokenizer is not None

    def _can_load_model(self) -> bool:
        """Check if sufficient memory is available for loading.

        Requires available memory to exceed estimated model size
        multiplied by buffer factor to avoid memory pressure.
        """
        mem = psutil.virtual_memory()
        available_mb = mem.available / BYTES_PER_MB
        required_mb = self.config.estimated_memory_mb * self.config.memory_buffer_multiplier

        if available_mb < required_mb:
            logger.warning(
                "Insufficient memory for model load: %.0fMB available, %.0fMB required",
                available_mb,
                required_mb,
            )
            return False
        return True

    def load(self) -> bool:
        """Load model into memory with thread-safe double-check locking.

        Returns:
            True if model loaded successfully or was already loaded,
            False if loading failed or insufficient memory.
        """
        # Fast path: already loaded
        if self.is_loaded():
            return True

        # Check memory before acquiring lock
        if not self._can_load_model():
            return False

        with self._load_lock:
            # Double-check after acquiring lock
            if self.is_loaded():
                return True

            try:
                logger.info("Loading model: %s", self.config.model_path)
                start_time = time.perf_counter()

                # mlx_lm.load returns (model, tokenizer) tuple
                result = load(self.config.model_path)
                self._model = result[0]
                self._tokenizer = result[1]
                self._loaded_at = time.perf_counter()

                load_time = (self._loaded_at - start_time) * 1000
                logger.info("Model loaded in %.0fms", load_time)
                return True

            except FileNotFoundError:
                logger.error(
                    "Model not found: %s. Run `huggingface-cli download %s` first.",
                    self.config.model_path,
                    self.config.model_path,
                )
                self.unload()
                return False
            except MemoryError:
                logger.error("Out of memory loading model. Free up memory or use a smaller model.")
                self.unload()
                return False
            except OSError as e:
                # Covers network errors, disk errors, permission issues
                logger.error("OS error loading model: %s", e)
                self.unload()
                return False
            except Exception:
                logger.exception("Failed to load model")
                self.unload()
                return False

    def unload(self) -> None:
        """Unload model and free all memory including GPU cache.

        Call order: Set references to None -> Clear Metal cache -> Force GC
        """
        logger.info("Unloading model")

        # Clear references
        self._model = None
        self._tokenizer = None
        self._loaded_at = None

        # Clear Metal GPU memory
        try:
            mx.metal.clear_cache()
        except Exception:
            logger.debug("Metal cache clear not available")

        # Force garbage collection
        gc.collect()
        logger.info("Model unloaded")

    def get_memory_usage_mb(self) -> float:
        """Return estimated memory usage of loaded model.

        Returns actual process memory delta if loaded, 0 otherwise.
        """
        if not self.is_loaded():
            return 0.0

        # Use estimated size since MLX doesn't expose exact GPU memory
        return self.config.estimated_memory_mb

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text synchronously.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Strings that stop generation

        Returns:
            GenerationResult with text, token count, and timing

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            msg = "Prompt cannot be empty"
            raise ValueError(msg)

        if not self.is_loaded():
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature

        try:
            # Format prompt for Qwen2.5-Instruct chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Create sampler with temperature
            sampler = make_sampler(temp=temperature)

            # Generate
            start_time = time.perf_counter()
            response = generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            generation_time = (time.perf_counter() - start_time) * 1000

            # Strip the prompt from response
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt) :].strip()

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in response:
                        response = response[: response.index(stop_seq)]

            response = response.strip()

            # Count tokens using tokenizer for accuracy
            try:
                tokens_generated = len(self._tokenizer.encode(response))
            except Exception:
                # Fallback to word count if tokenizer fails
                tokens_generated = len(response.split())

            return GenerationResult(
                text=response,
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time,
            )

        except Exception as e:
            logger.exception("Generation failed")
            msg = f"Generation failed: {e}"
            raise RuntimeError(msg) from e
