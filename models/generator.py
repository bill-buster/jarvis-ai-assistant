"""MLX Generator implementing the Generator protocol.

Orchestrates template matching and model-based generation with
memory-safe patterns and performance tracking.
"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from contracts.models import GenerationRequest, GenerationResponse
from models.loader import MLXModelLoader, ModelConfig
from models.prompt_builder import PromptBuilder
from models.templates import TemplateMatcher

logger = logging.getLogger(__name__)


class MLXGenerator:
    """Generator implementation using MLX with template fallback.

    Generation flow:
    1. Try template match (similarity >= 0.7)
       - If match: return immediately with finish_reason="template"
    2. Check memory, load model if needed
    3. Build prompt with RAG context + few-shot examples
    4. Generate with MLX
    5. Return response with metadata
    """

    def __init__(
        self,
        loader: MLXModelLoader | None = None,
        template_matcher: TemplateMatcher | None = None,
        prompt_builder: PromptBuilder | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            loader: Model loader instance. Creates default if not provided.
            template_matcher: Template matcher instance. Creates default if not provided.
            prompt_builder: Prompt builder instance. Creates default if not provided.
            config: Model configuration for the loader.
        """
        self.config = config or ModelConfig()
        self._loader = loader or MLXModelLoader(self.config)
        self._template_matcher = template_matcher or TemplateMatcher()
        self._prompt_builder = prompt_builder or PromptBuilder()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using template matching or model.

        Args:
            request: Generation request with prompt, context, and examples

        Returns:
            GenerationResponse with text and metadata
        """
        start_time = time.perf_counter()

        # Try template match first
        template_response = self._try_template_match(request)
        if template_response is not None:
            return template_response

        # Fall back to model generation
        return self._generate_with_model(request, start_time)

    def _try_template_match(self, request: GenerationRequest) -> GenerationResponse | None:
        """Attempt to match request to a template.

        Args:
            request: Generation request

        Returns:
            GenerationResponse if template matched, None otherwise
        """
        match = self._template_matcher.match(request.prompt)
        if match is None:
            return None

        return GenerationResponse(
            text=match.template.response,
            tokens_used=0,
            generation_time_ms=0.0,
            model_name="template",
            used_template=True,
            template_name=match.template.name,
            finish_reason="template",
        )

    def _generate_with_model(
        self, request: GenerationRequest, start_time: float
    ) -> GenerationResponse:
        """Generate response using the MLX model.

        Args:
            request: Generation request
            start_time: When generation started (for timing)

        Returns:
            GenerationResponse with generated text

        Raises:
            RuntimeError: If model cannot be loaded or generation fails
        """
        # Track if we loaded the model for this call (for cleanup on error)
        loaded_for_this_call = False

        try:
            # Ensure model is loaded
            if not self._loader.is_loaded():
                if not self._loader.load():
                    msg = "Failed to load model"
                    raise RuntimeError(msg)
                loaded_for_this_call = True

            # Build formatted prompt
            formatted_prompt = self._prompt_builder.build(request)

            # Generate with model
            result = self._loader.generate_sync(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop_sequences,
            )

            total_time = (time.perf_counter() - start_time) * 1000

            return GenerationResponse(
                text=result.text,
                tokens_used=result.tokens_generated,
                generation_time_ms=total_time,
                model_name=self.config.model_path,
                used_template=False,
                template_name=None,
                finish_reason="stop",
            )

        except Exception:
            # If we loaded the model for this call and generation failed,
            # unload to free memory and prevent inconsistent state
            if loaded_for_this_call:
                logger.warning("Generation failed, unloading model loaded for this request")
                self._loader.unload()
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._loader.is_loaded()

    def load(self) -> bool:
        """Load model into memory.

        Returns:
            True if loaded successfully, False otherwise
        """
        return self._loader.load()

    def unload(self) -> None:
        """Unload model to free memory."""
        self._loader.unload()

    def get_memory_usage_mb(self) -> float:
        """Return current memory usage of the model."""
        return self._loader.get_memory_usage_mb()

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate a response with streaming output (yields tokens).

        Yields tokens as they're generated for real-time display.
        Falls back to template matching first (returns complete response).

        Args:
            request: Generation request with prompt, context, and examples

        Yields:
            Dictionary with token information:
                - token: The generated token text
                - token_index: Index of this token in the sequence
                - is_final: Whether this is the last token
        """
        # Try template match first - if matched, yield complete response
        template_response = self._try_template_match(request)
        if template_response is not None:
            # Templates return complete responses, yield as single token
            yield {
                "token": template_response.text,
                "token_index": 0,
                "is_final": True,
                "used_template": True,
                "template_name": template_response.template_name,
            }
            return

        # Track if we loaded the model for this call (for cleanup on error)
        loaded_for_this_call = False

        try:
            # Ensure model is loaded
            if not self._loader.is_loaded():
                if not self._loader.load():
                    msg = "Failed to load model"
                    raise RuntimeError(msg)
                loaded_for_this_call = True

            # Build formatted prompt
            formatted_prompt = self._prompt_builder.build(request)

            # Stream tokens from the loader
            async for stream_token in self._loader.generate_stream(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop_sequences,
            ):
                yield {
                    "token": stream_token.token,
                    "token_index": stream_token.token_index,
                    "is_final": stream_token.is_final,
                }

        except Exception:
            # If we loaded the model for this call and generation failed,
            # unload to free memory and prevent inconsistent state
            if loaded_for_this_call:
                logger.warning("Streaming generation failed, unloading model")
                self._loader.unload()
            raise
