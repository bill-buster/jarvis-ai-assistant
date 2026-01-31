#!/usr/bin/env python3
"""Experiment with different prompting strategies for better LLM responses.

This script tests multiple prompting approaches and compares their effectiveness:
1. Original prompt (baseline)
2. Style-explicit prompt (explicit rules about length, no filler)
3. Negative examples prompt (shows what NOT to do)
4. Context-aware prompt (teaches when to ask for clarification)

Run: uv run python scripts/experiments/prompt_experiment.py --samples 200 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 100
RESULTS_DIR = Path("results/prompt_experiment")


# =============================================================================
# PROMPTING STRATEGIES
# =============================================================================

# Strategy 1: Original (baseline)
PROMPT_ORIGINAL = """### Communication Style with {contact_name}:
{relationship_context}

### Similar Past Exchanges:
{similar_exchanges}

### Current Conversation:
{context}

### Instructions:
Generate a natural reply to the last message that:
- Matches how you typically communicate with {contact_name} ({tone})
- Is consistent with your past response patterns
- Sounds authentic to your voice

### Last message to reply to:
{last_message}

### Your reply:"""


# Strategy 2: Style-Explicit (explicit rules)
PROMPT_STYLE_EXPLICIT = """You are generating a text message reply AS the user, not as an assistant.

### CRITICAL STYLE RULES:
- Keep responses SHORT: 1-15 words maximum
- NO greetings like "Hey!", "Hi!", "Hello!"
- NO filler phrases like "Let me know if you need anything"
- NO assistant language like "I understand", "I see", "That makes sense"
- Use casual abbreviations: "u", "ur", "rn", "ngl", "fs", "bet"
- Match the energy of the incoming message
- It's OK to be blunt or brief

### Examples of YOUR past responses:
{similar_exchanges}

### Incoming message from {contact_name}:
{last_message}

### Your reply (1-15 words, casual, direct):"""


# Strategy 3: Negative Examples (what NOT to do)
PROMPT_NEGATIVE_EXAMPLES = """Generate a text message reply as if you are the user texting a friend.

### DO NOT write responses like these (too formal/assistant-like):
BAD: "Hey! So, how's it going? Let me know if you need anything! 😊"
BAD: "I totally understand what you mean. That sounds really interesting!"
BAD: "Awesome! I'm here to help with whatever you need."
BAD: "Hey there! Just checking in to see how things are going."

### GOOD responses look like these (from your actual history):
{similar_exchanges}

### Style notes:
- Short (1-10 words typical)
- No "Hey!" greetings
- Direct, sometimes blunt
- Use slang: "bet", "fs", "ngl", "lowkey"

### Message from {contact_name}:
{last_message}

### Your reply:"""


# Strategy 4: Context-Aware (teaches when to clarify)
PROMPT_CONTEXT_AWARE = """You're texting as the user. Generate a SHORT reply (1-15 words).

### When to ask for context:
- If the message is vague ("do it", "ok", "yeah"), ask what they mean
- If you'd need info you don't have, ask for it
- Keep clarifying questions short: "wdym?", "which one?", "wait what"

### When to respond directly:
- Clear questions get direct answers
- Statements can be acknowledged briefly
- Plans get confirmed or declined

### Your texting style (from history):
{similar_exchanges}

### If context is unclear, ask briefly. If clear, respond directly.

### Message from {contact_name}:
{last_message}

### Your reply (SHORT - max 15 words):"""


# Strategy 5: Few-shot Only (minimal instructions, heavy examples)
PROMPT_FEW_SHOT_HEAVY = """Text message reply. Match the style of these examples exactly:

{similar_exchanges}

---
Incoming: {last_message}
Reply:"""


STRATEGIES = {
    "original": PROMPT_ORIGINAL,
    "style_explicit": PROMPT_STYLE_EXPLICIT,
    "negative_examples": PROMPT_NEGATIVE_EXAMPLES,
    "context_aware": PROMPT_CONTEXT_AWARE,
    "few_shot_heavy": PROMPT_FEW_SHOT_HEAVY,
}


@dataclass
class StrategyResult:
    """Result for a single strategy on a single test case."""

    strategy: str
    trigger: str
    actual_response: str
    llm_response: str | None
    similarity_to_actual: float
    word_count: int
    has_generic_filler: bool
    has_greeting: bool
    generation_time_ms: float


@dataclass
class StrategyStats:
    """Aggregated stats for a strategy."""

    strategy: str
    num_samples: int
    avg_similarity_to_actual: float
    avg_word_count: float
    pct_with_filler: float
    pct_with_greeting: float
    avg_generation_time_ms: float


def detect_generic_filler(response: str) -> bool:
    """Detect generic assistant-like filler phrases."""
    filler_phrases = [
        "let me know",
        "if you need anything",
        "how's it going",
        "just checking",
        "i'm here",
        "sounds good",
        "that's great",
        "awesome!",
        "i understand",
        "i see",
        "that makes sense",
        "i get it",
        "what's on your mind",
        "we've got this",
        "let's keep",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in filler_phrases)


def detect_greeting(response: str) -> bool:
    """Detect assistant-style greetings."""
    greetings = ["hey!", "hey there", "hi!", "hello!", "hi there"]
    response_lower = response.lower().strip()
    return any(response_lower.startswith(g) for g in greetings)


class PromptExperimenter:
    """Experiments with different prompting strategies."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.embedder = None
        self.generator = None
        self.db = None
        self.train_pairs = []
        self.test_pairs = []
        self.train_embeddings = None

    def initialize(self) -> bool:
        """Initialize components."""
        try:
            from sentence_transformers import SentenceTransformer

            from jarvis.db import get_db
            from models import get_generator

            logger.info("Initializing components...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.generator = get_generator()
            self.db = get_db()

            all_pairs = self.db.get_all_pairs(min_quality=0.5)
            if len(all_pairs) < 100:
                logger.error("Not enough pairs")
                return False

            random.seed(42)
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * 0.8)
            self.train_pairs = all_pairs[:split_idx]
            self.test_pairs = all_pairs[split_idx:]

            logger.info(f"Dataset: {len(self.train_pairs)} train / {len(self.test_pairs)} test")

            # Build embeddings
            train_triggers = [p.trigger_text for p in self.train_pairs]
            self.train_embeddings = self.embedder.encode(train_triggers, normalize_embeddings=True)

            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False

    def get_similar_examples(self, trigger: str, k: int = 5) -> list[tuple[str, str]]:
        """Get similar examples from train set."""
        trigger_emb = self.embedder.encode([trigger], normalize_embeddings=True)[0]
        similarities = np.dot(self.train_embeddings, trigger_emb)
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= 0.3:
                pair = self.train_pairs[idx]
                results.append((pair.trigger_text, pair.response_text))
        return results

    def format_examples(self, examples: list[tuple[str, str]], strategy: str) -> str:
        """Format examples based on strategy."""
        if not examples:
            return "(No examples)"

        if strategy == "few_shot_heavy":
            # Minimal format for few-shot heavy
            lines = []
            for trigger, response in examples[:5]:
                lines.append(f"Incoming: {trigger[:100]}")
                lines.append(f"Reply: {response}")
                lines.append("")
            return "\n".join(lines)
        else:
            # Standard format
            lines = []
            for i, (trigger, response) in enumerate(examples[:3], 1):
                trigger_short = trigger[:80] + "..." if len(trigger) > 80 else trigger
                lines.append(f"Them: {trigger_short}")
                lines.append(f"You: {response}")
                lines.append("")
            return "\n".join(lines)

    def generate_with_strategy(
        self,
        trigger: str,
        strategy: str,
        examples: list[tuple[str, str]],
        contact_name: str = "Friend",
    ) -> tuple[str | None, float]:
        """Generate response with a specific strategy."""
        from contracts.models import GenerationRequest

        start = time.time()

        template = STRATEGIES[strategy]
        formatted_examples = self.format_examples(examples, strategy)

        # Build prompt based on strategy
        if strategy == "original":
            prompt = template.format(
                contact_name=contact_name,
                relationship_context="- Casual tone\n- Average message length: ~10 words",
                similar_exchanges=formatted_examples,
                context=f"[Incoming]: {trigger}",
                tone="casual",
                last_message=trigger,
            )
        elif strategy in ["style_explicit", "negative_examples", "context_aware"]:
            prompt = template.format(
                contact_name=contact_name,
                similar_exchanges=formatted_examples,
                last_message=trigger,
            )
        else:  # few_shot_heavy
            prompt = template.format(
                similar_exchanges=formatted_examples,
                last_message=trigger,
            )

        try:
            request = GenerationRequest(
                prompt=prompt,
                max_tokens=60,  # Shorter for text messages
            )
            response = self.generator.generate(request)
            elapsed = (time.time() - start) * 1000
            return response.text.strip(), elapsed
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            elapsed = (time.time() - start) * 1000
            return None, elapsed

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity."""
        if not text1 or not text2:
            return 0.0
        embeddings = self.embedder.encode([text1, text2], normalize_embeddings=True)
        # With normalized embeddings, dot product = cosine similarity
        return float(np.dot(embeddings[0], embeddings[1]))

    def evaluate_single(
        self,
        test_pair: Any,
        strategy: str,
    ) -> StrategyResult:
        """Evaluate a single test case with a strategy."""
        trigger = test_pair.trigger_text
        actual = test_pair.response_text

        examples = self.get_similar_examples(trigger, k=5)
        llm_response, gen_time = self.generate_with_strategy(trigger, strategy, examples)

        if llm_response:
            similarity = self.compute_similarity(llm_response, actual)
            word_count = len(llm_response.split())
            has_filler = detect_generic_filler(llm_response)
            has_greeting = detect_greeting(llm_response)
        else:
            similarity = 0.0
            word_count = 0
            has_filler = False
            has_greeting = False

        return StrategyResult(
            strategy=strategy,
            trigger=trigger,
            actual_response=actual,
            llm_response=llm_response,
            similarity_to_actual=similarity,
            word_count=word_count,
            has_generic_filler=has_filler,
            has_greeting=has_greeting,
            generation_time_ms=gen_time,
        )

    def run_experiment(
        self,
        num_samples: int = DEFAULT_SAMPLES,
        strategies: list[str] | None = None,
    ) -> dict[str, tuple[StrategyStats, list[StrategyResult]]]:
        """Run experiment comparing strategies."""
        strategies = strategies or list(STRATEGIES.keys())

        # Sample test cases
        if num_samples < len(self.test_pairs):
            samples = random.sample(self.test_pairs, num_samples)
        else:
            samples = self.test_pairs

        logger.info(f"Testing {len(strategies)} strategies on {len(samples)} samples")

        all_results: dict[str, list[StrategyResult]] = {s: [] for s in strategies}

        for i, pair in enumerate(samples):
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i + 1}/{len(samples)}")

            for strategy in strategies:
                result = self.evaluate_single(pair, strategy)
                all_results[strategy].append(result)

                if self.verbose and result.llm_response:
                    logger.debug(
                        f"[{strategy}] {result.trigger[:30]}... -> "
                        f"{result.llm_response[:50]}... (sim={result.similarity_to_actual:.2f})"
                    )

        # Compute stats
        final_results = {}
        for strategy, results in all_results.items():
            valid = [r for r in results if r.llm_response]
            if not valid:
                continue

            stats = StrategyStats(
                strategy=strategy,
                num_samples=len(valid),
                avg_similarity_to_actual=sum(r.similarity_to_actual for r in valid) / len(valid),
                avg_word_count=sum(r.word_count for r in valid) / len(valid),
                pct_with_filler=100 * sum(1 for r in valid if r.has_generic_filler) / len(valid),
                pct_with_greeting=100 * sum(1 for r in valid if r.has_greeting) / len(valid),
                avg_generation_time_ms=sum(r.generation_time_ms for r in valid) / len(valid),
            )
            final_results[strategy] = (stats, results)

        return final_results


def save_results(
    results: dict[str, tuple[StrategyStats, list[StrategyResult]]],
    output_dir: Path,
) -> None:
    """Save experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {}
    for strategy, (stats, _) in results.items():
        summary[strategy] = {
            "num_samples": stats.num_samples,
            "avg_similarity_to_actual": stats.avg_similarity_to_actual,
            "avg_word_count": stats.avg_word_count,
            "pct_with_filler": stats.pct_with_filler,
            "pct_with_greeting": stats.pct_with_greeting,
            "avg_generation_time_ms": stats.avg_generation_time_ms,
        }

    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    detailed = {}
    for strategy, (_, result_list) in results.items():
        detailed[strategy] = [
            {
                "trigger": r.trigger,
                "actual": r.actual_response,
                "llm": r.llm_response,
                "similarity": r.similarity_to_actual,
                "word_count": r.word_count,
                "has_filler": r.has_generic_filler,
                "has_greeting": r.has_greeting,
            }
            for r in result_list
        ]

    results_path = output_dir / f"detailed_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(detailed, f, indent=2)

    logger.info(f"Saved to {output_dir}")


def print_results(results: dict[str, tuple[StrategyStats, list[StrategyResult]]]) -> None:
    """Print comparison of strategies."""
    print("\n" + "=" * 70)
    print("PROMPT STRATEGY EXPERIMENT RESULTS")
    print("=" * 70)

    # Sort by similarity to actual (descending)
    sorted_strategies = sorted(
        results.items(),
        key=lambda x: x[1][0].avg_similarity_to_actual,
        reverse=True,
    )

    print(f"\n{'Strategy':<20} {'Sim→Actual':>10} {'Words':>8} {'Filler%':>8} {'Greet%':>8}")
    print("-" * 70)

    for strategy, (stats, _) in sorted_strategies:
        print(
            f"{strategy:<20} "
            f"{stats.avg_similarity_to_actual:>10.3f} "
            f"{stats.avg_word_count:>8.1f} "
            f"{stats.pct_with_filler:>8.1f} "
            f"{stats.pct_with_greeting:>8.1f}"
        )

    print("\n" + "-" * 70)
    print("BEST STRATEGY: ", sorted_strategies[0][0])
    print("=" * 70)

    # Show example outputs
    print("\n--- EXAMPLE OUTPUTS (same trigger, different strategies) ---\n")

    # Get first result's trigger
    first_strategy = list(results.keys())[0]
    first_result = results[first_strategy][1][0]
    trigger = first_result.trigger
    actual = first_result.actual_response

    print(f"TRIGGER: {trigger[:80]}")
    print(f"ACTUAL:  {actual}")
    print()

    for strategy, (_, result_list) in results.items():
        for r in result_list:
            if r.trigger == trigger:
                resp = r.llm_response or "(failed)"
                print(f"[{strategy}]: {resp[:80]}")
                break


def main():
    parser = argparse.ArgumentParser(description="Test prompting strategies")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES.keys()))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    experimenter = PromptExperimenter(verbose=args.verbose)

    if not experimenter.initialize():
        return 1

    results = experimenter.run_experiment(
        num_samples=args.samples,
        strategies=args.strategies,
    )

    save_results(results, Path(args.output_dir))
    print_results(results)

    return 0


if __name__ == "__main__":
    exit(main())
