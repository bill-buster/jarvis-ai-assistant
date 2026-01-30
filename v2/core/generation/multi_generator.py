"""Multi-model parallel reply generator for JARVIS v2.

Generates replies from multiple models concurrently, giving users options
with different speed/quality tradeoffs.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from core.models.loader import ModelLoader
from core.models.registry import get_model_spec


@dataclass
class GeneratedReply:
    """A single generated reply from one model."""

    model_id: str
    model_name: str
    text: str
    generation_time_ms: float
    tokens_generated: int
    quality_tier: str  # "fast", "balanced", "best"
    quality_score: float = 0.0  # 0-1 score based on heuristics

    def __str__(self) -> str:
        return f"[{self.quality_tier}] {self.text} ({self.generation_time_ms:.0f}ms)"


def score_reply(text: str, context: str = "") -> float:
    """Score a reply based on quality heuristics.

    Returns a score from 0 (bad) to 1 (good).

    Checks for:
    - Appropriate length (not too short, not too long)
    - Doesn't repeat context/prompt
    - No meta-commentary (I'm a chatbot, I can't, etc.)
    - Coherent (no random tangents)
    - Natural texting style
    """
    if not text or len(text.strip()) < 2:
        return 0.0

    score = 0.5  # Start neutral
    text_lower = text.lower().strip()

    # Length scoring (ideal: 5-50 chars for texting)
    length = len(text)
    if 5 <= length <= 50:
        score += 0.15  # Ideal length
    elif length < 5:
        score -= 0.2  # Too short
    elif length > 100:
        score -= 0.15  # Too long for texting

    # Penalize repetition of context
    if context:
        context_lower = context.lower()
        # Check if reply copies significant part of context
        for phrase in text_lower.split():
            if len(phrase) > 4 and phrase in context_lower:
                score -= 0.05

    # Penalize meta-commentary / breaking character
    bad_patterns = [
        "i'm a chatbot", "i can't", "i cannot", "as an ai",
        "i'm not sure what", "i don't understand",
        "i'm sorry, but", "i apologize",
        "[casual", "[brief", # Prompt leakage
    ]
    for pattern in bad_patterns:
        if pattern in text_lower:
            score -= 0.3
            break

    # Penalize nonsensical patterns
    if text_lower.count("wow") > 2:  # Repetitive
        score -= 0.2
    if text.count("!") > 3:  # Over-enthusiastic
        score -= 0.1

    # Bonus for natural texting patterns
    natural_patterns = [
        "yeah", "yea", "ok", "sure", "sounds good", "nice",
        "lol", "haha", "omg", "def", "prob",
    ]
    for pattern in natural_patterns:
        if pattern in text_lower:
            score += 0.05
            break

    # Bonus for ending appropriately (no trailing weirdness)
    if text.rstrip()[-1] in ".!?😊👍🎉":
        score += 0.05

    # Clamp to 0-1
    return max(0.0, min(1.0, score))


@dataclass
class MultiReplyResult:
    """Results from parallel multi-model generation."""

    replies: list[GeneratedReply]
    total_time_ms: float
    prompt: str

    @property
    def fastest(self) -> GeneratedReply | None:
        """Get the fastest reply."""
        if not self.replies:
            return None
        return min(self.replies, key=lambda r: r.generation_time_ms)

    @property
    def best_quality(self) -> GeneratedReply | None:
        """Get the highest quality reply based on score."""
        if not self.replies:
            return None
        return max(self.replies, key=lambda r: r.quality_score)

    @property
    def ranked(self) -> list[GeneratedReply]:
        """Get replies sorted by quality score (best first)."""
        return sorted(self.replies, key=lambda r: r.quality_score, reverse=True)

    def top_n(self, n: int = 2) -> list[GeneratedReply]:
        """Get top N replies by quality score."""
        return self.ranked[:n]


# Model configurations with quality tiers
DEFAULT_MODELS = [
    ("qwen3-0.6b", "fast"),
    ("lfm2.5-1.2b", "balanced"),
    ("lfm2-2.6b-exp", "best"),
]


class MultiModelGenerator:
    """Generates replies from multiple models in parallel."""

    def __init__(
        self,
        models: list[tuple[str, str]] | None = None,
        preload: bool = True,
    ):
        """Initialize multi-model generator.

        Args:
            models: List of (model_id, quality_tier) tuples
            preload: Whether to preload all models on init
        """
        self.models = models or DEFAULT_MODELS
        self._loaders: dict[str, ModelLoader] = {}
        self._lock = Lock()

        if preload:
            self._preload_models()

    def _preload_models(self):
        """Preload all models into memory."""
        print(f"Preloading {len(self.models)} models...")
        total_start = time.time()

        for model_id, tier in self.models:
            try:
                spec = get_model_spec(model_id)
                print(f"  Loading {spec.display_name}...", end=" ", flush=True)
                start = time.time()

                loader = ModelLoader(model_id)
                loader.preload()
                self._loaders[model_id] = loader

                print(f"done ({time.time() - start:.1f}s)")
            except Exception as e:
                print(f"failed: {e}")

        print(f"All models loaded in {time.time() - total_start:.1f}s")
        self._report_memory()

    def _report_memory(self):
        """Report current memory usage."""
        try:
            import mlx.core as mx
            mem_gb = mx.metal.get_active_memory() / (1024 ** 3)
            print(f"Total memory: {mem_gb:.2f} GB")
        except Exception:
            pass

    def _generate_single(
        self,
        model_id: str,
        tier: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> GeneratedReply | None:
        """Generate from a single model (called in thread)."""
        loader = self._loaders.get(model_id)
        if not loader:
            return None

        try:
            spec = get_model_spec(model_id)

            result = loader.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                use_chat_template=False,
            )

            text = result.text.strip()
            quality_score = score_reply(text, prompt)

            return GeneratedReply(
                model_id=model_id,
                model_name=spec.display_name,
                text=text,
                generation_time_ms=result.generation_time_ms,
                tokens_generated=result.tokens_generated,
                quality_tier=tier,
                quality_score=quality_score,
            )
        except Exception as e:
            print(f"Error generating with {model_id}: {e}")
            return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 30,
        temperature: float = 0.3,
        stop: list[str] | None = None,
        timeout: float = 5.0,
        on_reply: Callable[[GeneratedReply], None] | None = None,
    ) -> MultiReplyResult:
        """Generate replies from all models sequentially.

        MLX doesn't support true parallel generation, so we generate
        sequentially but stream results via callback as they complete.

        Args:
            prompt: The prompt to send to all models
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            timeout: Max time to wait for all models (seconds)
            on_reply: Optional callback called as each reply completes

        Returns:
            MultiReplyResult with all generated replies
        """
        if stop is None:
            stop = ["\n", "them:", "<|im_end|>", "<|eot_id|>", "<end_of_turn>"]

        start_time = time.time()
        replies: list[GeneratedReply] = []

        # Generate sequentially (MLX isn't thread-safe for parallel gen)
        for model_id, tier in self.models:
            result = self._generate_single(
                model_id,
                tier,
                prompt,
                max_tokens,
                temperature,
                stop,
            )
            if result:
                replies.append(result)
                # Stream result via callback
                if on_reply:
                    on_reply(result)

        total_time = (time.time() - start_time) * 1000

        return MultiReplyResult(
            replies=replies,
            total_time_ms=total_time,
            prompt=prompt,
        )

    def unload(self):
        """Unload all models and free memory."""
        for model_id, loader in self._loaders.items():
            try:
                loader.unload()
            except Exception:
                pass
        self._loaders.clear()
        gc.collect()

        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass


# Singleton instance
_multi_generator: MultiModelGenerator | None = None
_multi_generator_lock = Lock()


def get_multi_generator(preload: bool = True) -> MultiModelGenerator:
    """Get or create the singleton multi-model generator.

    Args:
        preload: Whether to preload models if creating new instance

    Returns:
        MultiModelGenerator instance
    """
    global _multi_generator

    with _multi_generator_lock:
        if _multi_generator is None:
            _multi_generator = MultiModelGenerator(preload=preload)
        return _multi_generator


def reset_multi_generator():
    """Reset the multi-model generator and free memory."""
    global _multi_generator

    with _multi_generator_lock:
        if _multi_generator is not None:
            _multi_generator.unload()
            _multi_generator = None


def generate_multi_replies(
    prompt: str,
    max_tokens: int = 30,
    temperature: float = 0.3,
) -> MultiReplyResult:
    """Convenience function to generate replies from all models.

    Args:
        prompt: The prompt to generate from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        MultiReplyResult with replies from all models
    """
    generator = get_multi_generator()
    return generator.generate(prompt, max_tokens, temperature)


# Quick test
if __name__ == "__main__":
    print("Testing multi-model generator...")

    # Test prompt
    prompt = """[brief, casual]

them: hey whats up
them: wanna grab dinner tonight?
me:"""

    print(f"\nPrompt:\n{prompt}\n")

    result = generate_multi_replies(prompt)

    print(f"\nResults (total: {result.total_time_ms:.0f}ms):")
    print("-" * 50)
    for reply in result.replies:
        print(f"  [{reply.quality_tier:8}] {reply.model_name:15} ({reply.generation_time_ms:4.0f}ms): {reply.text}")

    print(f"\nFastest: {result.fastest}")
    print(f"Best quality: {result.best_quality}")

    # Cleanup
    reset_multi_generator()
