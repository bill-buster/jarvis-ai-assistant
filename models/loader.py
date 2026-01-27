"""MLX Model Loader with memory-safe lifecycle management.

Handles model loading/unloading with thread-safety, memory tracking,
and double-check locking patterns for lazy initialization.

Supports model selection via the registry system:
    from models.loader import MLXModelLoader, ModelConfig

    # Use default model from registry
    loader = MLXModelLoader()

    # Use specific model from registry
    config = ModelConfig(model_id="qwen-3b")
    loader = MLXModelLoader(config)

    # Use custom model path (not in registry)
    config = ModelConfig(model_path="custom/model-path")
    loader = MLXModelLoader(config)
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import psutil
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from jarvis.errors import (
    ErrorCode,
    ModelGenerationError,
    ModelLoadError,
    model_generation_timeout,
    model_not_found,
    model_out_of_memory,
)
from models.registry import DEFAULT_MODEL_ID, ModelSpec, get_model_spec

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024


def _get_default_generation_timeout() -> float:
    """Get default generation timeout from config.

    Returns:
        Timeout in seconds from config, or 60.0 as fallback.
    """
    try:
        from jarvis.config import get_config

        return get_config().model.generation_timeout_seconds
    except Exception:
        return 60.0  # Fallback default


@dataclass
class ModelConfig:
    """Configuration for MLX model loading.

    Can be initialized with either model_id (from registry) or model_path.
    If both are provided, model_id takes precedence. If neither is provided,
    uses the default model from the registry.

    Attributes:
        model_id: Model identifier from registry (e.g., "qwen-1.5b").
        model_path: Direct HuggingFace path (overridden by model_id if set).
        estimated_memory_mb: Estimated memory usage (auto-set from registry if model_id used).
        memory_buffer_multiplier: Safety buffer for memory checks.
        default_max_tokens: Default max tokens for generation.
        default_temperature: Default sampling temperature.
        generation_timeout_seconds: Timeout for generation in seconds (None = no timeout).
            If not explicitly set, uses the value from config.model.generation_timeout_seconds.
    """

    model_id: str | None = None
    model_path: str = ""
    estimated_memory_mb: float = 800
    memory_buffer_multiplier: float = 1.1  # Reduced from 1.5 to 1.1 (10% safety buffer)
    default_max_tokens: int = 100
    default_temperature: float = 0.7
    generation_timeout_seconds: float | None = field(
        default_factory=_get_default_generation_timeout
    )
    _resolved_spec: ModelSpec | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Resolve model_id to model_path and estimated_memory_mb."""
        if self.model_id:
            spec = get_model_spec(self.model_id)
            if spec:
                self._resolved_spec = spec
                self.model_path = spec.path
                self.estimated_memory_mb = spec.estimated_memory_mb
            else:
                logger.warning("Unknown model_id '%s', using as-is", self.model_id)
                self.model_path = self.model_id
        elif not self.model_path:
            # Neither model_id nor model_path specified, use default
            spec = get_model_spec(DEFAULT_MODEL_ID)
            if spec:
                self._resolved_spec = spec
                self.model_id = spec.id
                self.model_path = spec.path
                self.estimated_memory_mb = spec.estimated_memory_mb
            else:
                # Fallback to hardcoded default
                self.model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    @property
    def display_name(self) -> str:
        """Return a human-readable name for the model."""
        if self._resolved_spec:
            return self._resolved_spec.display_name
        return self.model_path.split("/")[-1]

    @property
    def spec(self) -> ModelSpec | None:
        """Return the resolved ModelSpec if available."""
        return self._resolved_spec


@dataclass
class GenerationResult:
    """Result from synchronous generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float


@dataclass
class StreamToken:
    """A single token from streaming generation."""

    token: str
    token_index: int
    is_final: bool = False


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

    def _can_load_model(self) -> tuple[bool, int, int]:
        """Check if sufficient memory is available for loading.

        Requires available memory to exceed estimated model size
        multiplied by buffer factor to avoid memory pressure.

        Returns:
            Tuple of (can_load, available_mb, required_mb).
        """
        mem = psutil.virtual_memory()
        available_mb = int(mem.available / BYTES_PER_MB)
        required_mb = int(self.config.estimated_memory_mb * self.config.memory_buffer_multiplier)

        if available_mb < required_mb:
            logger.warning(
                "Insufficient memory for model load: %dMB available, %dMB required",
                available_mb,
                required_mb,
            )
            return False, available_mb, required_mb
        return True, available_mb, required_mb

    def load(self) -> bool:
        """Load model into memory with thread-safe double-check locking.

        Returns:
            True if model loaded successfully or was already loaded.

        Raises:
            ModelLoadError: If loading fails due to memory, missing files, or other issues.
        """
        # Fast path: already loaded
        if self.is_loaded():
            return True

        # Check memory before acquiring lock
        can_load, available_mb, required_mb = self._can_load_model()
        if not can_load:
            raise model_out_of_memory(
                self.config.display_name,
                available_mb=available_mb,
                required_mb=required_mb,
            )

        with self._load_lock:
            # Double-check after acquiring lock
            if self.is_loaded():
                return True

            try:
                logger.info(
                    "Loading model: %s (%s)",
                    self.config.display_name,
                    self.config.model_path,
                )
                start_time = time.perf_counter()

                # mlx_lm.load returns (model, tokenizer) tuple
                result = load(self.config.model_path)
                self._model = result[0]
                self._tokenizer = result[1]
                self._loaded_at = time.perf_counter()

                load_time = (self._loaded_at - start_time) * 1000
                logger.info("Model loaded in %.0fms", load_time)
                return True

            except FileNotFoundError as e:
                logger.error(
                    "Model not found: %s. Run `huggingface-cli download %s` first.",
                    self.config.model_path,
                    self.config.model_path,
                )
                self.unload()
                raise model_not_found(self.config.model_path) from e
            except MemoryError as e:
                logger.error("Out of memory loading model. Free up memory or use a smaller model.")
                self.unload()
                raise model_out_of_memory(self.config.display_name) from e
            except OSError as e:
                # Covers network errors, disk errors, permission issues
                logger.error("OS error loading model: %s", e)
                self.unload()
                raise ModelLoadError(
                    f"OS error loading model: {e}",
                    model_name=self.config.display_name,
                    model_path=self.config.model_path,
                    cause=e,
                ) from e
            except Exception as e:
                logger.exception("Failed to load model")
                self.unload()
                raise ModelLoadError(
                    f"Failed to load model: {e}",
                    model_name=self.config.display_name,
                    model_path=self.config.model_path,
                    cause=e,
                ) from e

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
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        """Generate text synchronously.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Strings that stop generation
            timeout_seconds: Timeout for generation (overrides config if provided)

        Returns:
            GenerationResult with text, token count, and timing

        Raises:
            ModelGenerationError: If model is not loaded, prompt is invalid,
                generation fails, or timeout is exceeded.
        """
        if not prompt or not prompt.strip():
            raise ModelGenerationError(
                "Prompt cannot be empty",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_INVALID_REQUEST,
            )

        if not self.is_loaded():
            raise ModelGenerationError(
                "Model not loaded. Call load() first.",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_LOAD_FAILED,
            )

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        # Use provided timeout, fall back to config, then None (no timeout)
        effective_timeout: float | None
        if timeout_seconds is not None:
            effective_timeout = timeout_seconds
        else:
            effective_timeout = self.config.generation_timeout_seconds

        try:
            # Format prompt for Qwen2.5-Instruct chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Create sampler with temperature
            sampler = make_sampler(temp=temperature)

            # Generate with timeout handling
            start_time = time.perf_counter()

            def _do_generate() -> str:
                """Inner function for generation to run in executor."""
                return generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )

            if effective_timeout is not None and effective_timeout > 0:
                # Use ThreadPoolExecutor for timeout handling
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_generate)
                    try:
                        response = future.result(timeout=effective_timeout)
                    except FuturesTimeoutError:
                        # Cancel the future (best effort, may not stop MLX generation)
                        future.cancel()
                        logger.warning(
                            "Generation timed out after %.1f seconds for model %s",
                            effective_timeout,
                            self.config.display_name,
                        )
                        raise model_generation_timeout(
                            self.config.display_name,
                            effective_timeout,
                            prompt=prompt,
                        )
            else:
                # No timeout - run directly
                response = _do_generate()

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

        except ModelGenerationError:
            # Re-raise JARVIS errors as-is
            raise
        except Exception as e:
            logger.exception("Generation failed")
            raise ModelGenerationError(
                f"Generation failed: {e}",
                prompt=prompt,
                model_name=self.config.display_name,
                cause=e,
            ) from e

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Generate text with streaming output (yields tokens).

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Strings that stop generation
            timeout_seconds: Timeout for generation (overrides config if provided)

        Yields:
            StreamToken objects with individual tokens

        Raises:
            ModelGenerationError: If model is not loaded, prompt is invalid,
                generation fails, or timeout is exceeded.
        """
        if not prompt or not prompt.strip():
            raise ModelGenerationError(
                "Prompt cannot be empty",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_INVALID_REQUEST,
            )

        if not self.is_loaded():
            raise ModelGenerationError(
                "Model not loaded. Call load() first.",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_LOAD_FAILED,
            )

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        effective_timeout: float | None
        if timeout_seconds is not None:
            effective_timeout = timeout_seconds
        else:
            effective_timeout = self.config.generation_timeout_seconds

        try:
            # Format prompt for Qwen2.5-Instruct chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Create sampler with temperature
            sampler = make_sampler(temp=temperature)

            # For streaming, we generate one token at a time
            # MLX generate doesn't have native streaming, so we simulate it
            # by generating incrementally
            token_index = 0

            def _do_generate() -> str:
                """Inner function for generation to run with timeout."""
                return generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )

            # Generate full response first (MLX limitation) with timeout
            if effective_timeout is not None and effective_timeout > 0:
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(_do_generate)
                    try:
                        response = future.result(timeout=effective_timeout)
                    except FuturesTimeoutError:
                        future.cancel()
                        # Don't wait for potentially stuck thread
                        executor.shutdown(wait=False, cancel_futures=True)
                        logger.warning(
                            "Streaming generation timed out after %.1f seconds for model %s",
                            effective_timeout,
                            self.config.display_name,
                        )
                        raise model_generation_timeout(
                            self.config.display_name,
                            effective_timeout,
                            prompt=prompt,
                        )
                    # Success - normal cleanup
                    executor.shutdown(wait=True)
                except Exception:
                    # Ensure cleanup on any exception
                    if not executor._shutdown:
                        executor.shutdown(wait=False, cancel_futures=True)
                    raise
            else:
                response = _do_generate()

            # Strip the prompt from response
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt) :].strip()

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in response:
                        response = response[: response.index(stop_seq)]

            response = response.strip()

            # Tokenize and yield each token
            # Since MLX doesn't support true streaming, we simulate it
            # by yielding words/tokens from the complete response
            tokens = self._tokenizer.encode(response)
            decoded_so_far = ""

            for i, _token in enumerate(tokens):
                # Decode up to current token
                partial_tokens = tokens[: i + 1]
                try:
                    current_decoded = self._tokenizer.decode(partial_tokens)
                except Exception:
                    continue

                # Get the new part
                new_text = current_decoded[len(decoded_so_far) :]
                decoded_so_far = current_decoded

                if new_text:
                    yield StreamToken(
                        token=new_text,
                        token_index=token_index,
                        is_final=(i == len(tokens) - 1),
                    )
                    token_index += 1

        except ModelGenerationError:
            raise
        except Exception as e:
            logger.exception("Streaming generation failed")
            raise ModelGenerationError(
                f"Streaming generation failed: {e}",
                prompt=prompt,
                model_name=self.config.display_name,
                cause=e,
            ) from e

    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model.

        Unloads the current model (if any) and updates configuration to use
        the new model. The new model will be loaded on the next generation request.

        Args:
            model_id: Model identifier from registry (e.g., "qwen-3b").

        Returns:
            True if configuration was updated successfully, False if model_id unknown.
        """
        spec = get_model_spec(model_id)
        if spec is None:
            logger.error("Unknown model_id: %s", model_id)
            return False

        # Unload current model if loaded
        if self.is_loaded():
            logger.info("Switching from %s to %s", self.config.display_name, spec.display_name)
            self.unload()

        # Update configuration
        self.config = ModelConfig(model_id=model_id)
        logger.info("Model configuration updated to %s", spec.display_name)
        return True

    def get_current_model_info(self) -> dict[str, Any]:
        """Return information about the current model configuration.

        Returns:
            Dictionary with model information including id, display_name,
            loaded status, and memory usage.
        """
        return {
            "id": self.config.model_id,
            "path": self.config.model_path,
            "display_name": self.config.display_name,
            "loaded": self.is_loaded(),
            "memory_usage_mb": self.get_memory_usage_mb(),
            "quality_tier": getattr(self.config.spec, "quality_tier", None),
        }
