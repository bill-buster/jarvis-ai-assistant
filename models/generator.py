"""MLX Generator implementing the Generator protocol.

Orchestrates template matching and model-based generation with
memory-safe patterns and performance tracking.

Also provides ThreadAwareGenerator for thread-context-aware reply generation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from contracts.models import GenerationRequest, GenerationResponse
from models.loader import MLXModelLoader, ModelConfig
from models.prompt_builder import PromptBuilder
from models.templates import TemplateMatcher

if TYPE_CHECKING:
    from jarvis.threading import ThreadContext, ThreadTopic

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
        skip_templates: bool = False,
    ) -> None:
        """Initialize the generator.

        Args:
            loader: Model loader instance. Creates default if not provided.
            template_matcher: Template matcher instance. Creates default if not provided.
            prompt_builder: Prompt builder instance. Creates default if not provided.
            config: Model configuration for the loader.
            skip_templates: If True, skip template matching entirely (saves memory).
        """
        self.config = config or ModelConfig()
        self._loader = loader or MLXModelLoader(self.config)
        self._skip_templates = skip_templates
        self._template_matcher = None if skip_templates else (template_matcher or TemplateMatcher())
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
        if self._template_matcher is None:
            return None

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

            # Generate with model using LFM-optimal parameters
            result = self._loader.generate_sync(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
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

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[dict[str, Any]]:
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


class ThreadAwareGenerator:
    """Thread-context-aware generator for improved reply generation.

    Uses thread analysis to:
    - Select topic-specific few-shot examples
    - Adjust response length based on thread type
    - Provide context-appropriate prompts for group chats
    - Include action items for planning/logistics threads

    Wraps MLXGenerator and adds thread-awareness to generation.

    Example:
        >>> from jarvis.threading import get_thread_analyzer
        >>> analyzer = get_thread_analyzer()
        >>> thread_context = analyzer.analyze(messages)
        >>> generator = ThreadAwareGenerator()
        >>> response = generator.generate_threaded(
        ...     thread_context=thread_context,
        ...     instruction="be friendly",
        ... )
    """

    def __init__(
        self,
        base_generator: MLXGenerator | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize the thread-aware generator.

        Args:
            base_generator: Optional base MLXGenerator. Creates one if not provided.
            config: Optional model config for creating base generator.
        """
        self._generator = base_generator or MLXGenerator(config=config)

    def generate_threaded(
        self,
        thread_context: ThreadContext,
        instruction: str | None = None,
        temperature: float | None = None,
    ) -> GenerationResponse:
        """Generate a thread-aware reply.

        Analyzes the thread context to build an appropriate prompt and
        adjust generation parameters for the thread type.

        Args:
            thread_context: Analyzed thread context from ThreadAnalyzer
            instruction: Optional custom instruction for the reply
            temperature: Optional temperature override (auto-selected if not provided)

        Returns:
            GenerationResponse with generated reply and metadata
        """
        from jarvis.prompts import build_threaded_reply_prompt, get_thread_max_tokens
        from jarvis.threading import ThreadTopic, get_thread_analyzer

        start_time = time.perf_counter()

        # Get response config for this thread type
        analyzer = get_thread_analyzer()
        config = analyzer.get_response_config(thread_context)

        # Build thread-aware prompt
        prompt = build_threaded_reply_prompt(
            thread_context=thread_context,
            config=config,
            instruction=instruction,
        )

        # Determine max tokens based on thread type
        max_tokens = get_thread_max_tokens(config)

        # Determine temperature based on thread type
        if temperature is None:
            temperature = self._get_temperature_for_topic(thread_context.topic)

        # Try template match first for quick exchanges
        if thread_context.topic == ThreadTopic.QUICK_EXCHANGE:
            template_response = self._try_quick_template(thread_context)
            if template_response is not None:
                return template_response

        # Build generation request
        # Include thread-specific examples as few-shot
        examples = self._get_thread_examples(thread_context.topic)

        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],  # Context is already in the prompt
            few_shot_examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Generate with base generator
        response = self._generator.generate(request)

        # Post-process for thread type
        processed_text = self._post_process_response(response.text, thread_context, config)

        total_time = (time.perf_counter() - start_time) * 1000

        return GenerationResponse(
            text=processed_text,
            tokens_used=response.tokens_used,
            generation_time_ms=total_time,
            model_name=response.model_name,
            used_template=response.used_template,
            template_name=response.template_name,
            finish_reason=response.finish_reason,
        )

    def _get_temperature_for_topic(self, topic: ThreadTopic) -> float:
        """Get appropriate temperature for thread topic.

        Args:
            topic: The thread topic

        Returns:
            Temperature value (lower = more deterministic)
        """
        from jarvis.threading import ThreadTopic

        # Lower temperature for logistics (need precision)
        # Higher for emotional support (need warmth/variety)
        temperature_map = {
            ThreadTopic.LOGISTICS: 0.3,
            ThreadTopic.QUICK_EXCHANGE: 0.2,
            ThreadTopic.INFORMATION: 0.3,
            ThreadTopic.PLANNING: 0.5,
            ThreadTopic.DECISION_MAKING: 0.5,
            ThreadTopic.EMOTIONAL_SUPPORT: 0.7,
            ThreadTopic.CATCHING_UP: 0.7,
            ThreadTopic.CELEBRATION: 0.7,
            ThreadTopic.UNKNOWN: 0.6,
        }

        return temperature_map.get(topic, 0.6)

    def _try_quick_template(self, thread_context: ThreadContext) -> GenerationResponse | None:
        """Try to match quick exchange to a template.

        Args:
            thread_context: The thread context

        Returns:
            GenerationResponse if matched, None otherwise
        """
        if not thread_context.messages:
            return None

        # Get last message
        last_msg = thread_context.messages[-1]
        last_text = getattr(last_msg, "text", None)

        # If text attribute is None or doesn't exist, try string conversion
        if last_text is None:
            last_text = str(last_msg)

        # Check for empty or whitespace-only text
        if not last_text or not last_text.strip():
            return None

        # Try template matching
        if self._generator._template_matcher is None:
            return None
        match = self._generator._template_matcher.match(last_text)
        if match is not None:
            return GenerationResponse(
                text=match.template.response,
                tokens_used=0,
                generation_time_ms=0.0,
                model_name="template",
                used_template=True,
                template_name=match.template.name,
                finish_reason="template",
            )

        return None

    def _get_thread_examples(self, topic: ThreadTopic) -> list[tuple[str, str]]:
        """Get few-shot examples for thread topic.

        Args:
            topic: The thread topic

        Returns:
            List of (input, output) example tuples
        """
        from jarvis.prompts import THREAD_EXAMPLES

        topic_to_key = {
            "logistics": "logistics",
            "planning": "planning",
            "catching_up": "catching_up",
            "emotional_support": "emotional_support",
            "quick_exchange": "quick_exchange",
            "information": "catching_up",
            "decision_making": "planning",
            "celebration": "catching_up",
            "unknown": "catching_up",
        }

        key = topic_to_key.get(topic.value, "catching_up")
        examples = THREAD_EXAMPLES.get(key, [])

        return [(ex.context, ex.output) for ex in examples[:2]]

    def _post_process_response(
        self,
        text: str,
        thread_context: ThreadContext,
        config: object,
    ) -> str:
        """Post-process generated response for thread type.

        Args:
            text: Generated text
            thread_context: Thread context
            config: Response configuration

        Returns:
            Processed text
        """
        from jarvis.threading import ThreadTopic

        # Clean up common issues
        text = text.strip()

        # Remove any prompt artifacts
        if "###" in text:
            text = text.split("###")[0].strip()

        # For quick exchanges, ensure brevity
        if thread_context.topic == ThreadTopic.QUICK_EXCHANGE:
            # Take only first sentence/line
            lines = text.split("\n")
            text = lines[0].strip()
            # Limit length
            if len(text) > 50:
                # Try to cut at a natural break within first 60 chars
                found_break = False
                for sep in [". ", "! ", "? "]:
                    idx = text[:60].find(sep)
                    if idx != -1:
                        text = text[: idx + 1]
                        found_break = True
                        break
                # If no natural break found, hard truncate
                if not found_break:
                    text = text[:50].rstrip() + "..."

        # For logistics, ensure we're specific
        # (no post-processing needed, prompt handles this)

        # For emotional support, ensure warmth
        # (no post-processing needed, prompt handles this)

        return text

    # Delegate to base generator for standard operations
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._generator.is_loaded()

    def load(self) -> bool:
        """Load model into memory."""
        return self._generator.load()

    def unload(self) -> None:
        """Unload model to free memory."""
        self._generator.unload()

    def get_memory_usage_mb(self) -> float:
        """Return current memory usage of the model."""
        return self._generator.get_memory_usage_mb()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Standard generation (delegates to base generator).

        Args:
            request: Generation request

        Returns:
            GenerationResponse
        """
        return self._generator.generate(request)
