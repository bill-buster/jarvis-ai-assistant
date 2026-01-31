#!/usr/bin/env python3
"""Improved LLM Evaluation with Semantic Similarity and Fair Coherence Scoring.

This script addresses the biases in the original evaluation:
1. Uses semantic similarity to compare LLM output vs actual human response
2. Has fairer coherence scoring (reduced penalties for questions/proper nouns)
3. Adds response quality metrics (brevity, directness, naturalness)

Run: uv run python scripts/experiments/improved_llm_eval.py --samples 2000 --verbose
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLES = 500
RESULTS_DIR = Path("results/improved_eval")


@dataclass
class ImprovedComparisonResult:
    """Result of comparing template vs LLM response with improved metrics."""

    trigger: str
    actual_response: str
    template_response: str | None
    llm_response: str | None

    # Similarity scores (embedding-based)
    template_to_actual_similarity: float  # How similar is template to actual?
    llm_to_actual_similarity: float | None  # How similar is LLM to actual?

    # Coherence scores (original, for comparison)
    template_coherence_original: float
    llm_coherence_original: float | None

    # Coherence scores (improved, fairer)
    template_coherence_fair: float
    llm_coherence_fair: float | None

    # Quality metrics
    template_brevity: float  # Shorter = better for texts
    llm_brevity: float | None
    template_directness: float  # Does it answer/respond?
    llm_directness: float | None
    template_naturalness: float  # Does it sound human?
    llm_naturalness: float | None

    # Aggregated scores
    template_overall: float
    llm_overall: float | None

    winner: str  # "template", "llm", "tie", or "both_failed"
    reason: str
    comparison_time_ms: float


@dataclass
class ImprovedEvaluationSummary:
    """Summary statistics for the improved evaluation."""

    total_samples: int
    template_wins: int
    llm_wins: int
    ties: int
    both_failed: int

    # Similarity to actual response (key metric!)
    avg_template_to_actual_sim: float
    avg_llm_to_actual_sim: float

    # Coherence (original vs fair)
    avg_template_coherence_original: float
    avg_llm_coherence_original: float
    avg_template_coherence_fair: float
    avg_llm_coherence_fair: float

    # Quality metrics
    avg_template_brevity: float
    avg_llm_brevity: float
    avg_template_directness: float
    avg_llm_directness: float
    avg_template_naturalness: float
    avg_llm_naturalness: float

    # Overall
    avg_template_overall: float
    avg_llm_overall: float

    total_time_seconds: float


def score_coherence_original(trigger: str, response: str) -> float:
    """Original coherence scoring (for comparison)."""
    import re

    score = 1.0
    trigger_stripped = trigger.strip()
    response_stripped = response.strip()

    # 1. Penalize if response is a question to a non-question
    if response_stripped.endswith("?") and not trigger_stripped.endswith("?"):
        score *= 0.5

    # 2. Penalize proper nouns not in trigger
    def extract_proper_nouns(text: str) -> set[str]:
        words = text.split()
        proper_nouns = set()
        for i, word in enumerate(words):
            if i == 0:
                continue
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                proper_nouns.add(clean_word.lower())
        return proper_nouns

    trigger_nouns = extract_proper_nouns(trigger)
    response_nouns = extract_proper_nouns(response)
    unrelated_nouns = response_nouns - trigger_nouns

    if unrelated_nouns:
        penalty = 0.6 ** len(unrelated_nouns)
        score *= max(0.3, penalty)

    # 3. Penalize very short responses
    trigger_words = len(trigger.split())
    response_words = len(response.split())

    if response_words < 3 and trigger_words > 10:
        score *= 0.7

    # 4. Penalize verbose responses
    if response_words > trigger_words * 5 and response_words > 20:
        score *= 0.8

    # 5. Bonus for length matching
    if 0.3 <= response_words / max(trigger_words, 1) <= 3.0:
        score = min(1.0, score * 1.1)

    return round(score, 3)


def score_coherence_fair(trigger: str, response: str) -> float:
    """Fair coherence scoring - reduced penalties for LLM-typical behaviors."""

    score = 1.0
    trigger_stripped = trigger.strip()
    response_stripped = response.strip()

    # 1. REDUCED penalty for questions (0.8 instead of 0.5)
    # Clarifying questions can be appropriate
    if response_stripped.endswith("?") and not trigger_stripped.endswith("?"):
        score *= 0.8

    # 2. REMOVED proper noun penalty entirely
    # LLMs often add relevant context, this shouldn't be penalized

    # 3. Penalize very short responses (keep this)
    trigger_words = len(trigger.split())
    response_words = len(response.split())

    if response_words < 3 and trigger_words > 10:
        score *= 0.7

    # 4. REDUCED verbose penalty (0.9 instead of 0.8)
    if response_words > trigger_words * 5 and response_words > 20:
        score *= 0.9

    # 5. Keep length matching bonus
    if 0.3 <= response_words / max(trigger_words, 1) <= 3.0:
        score = min(1.0, score * 1.1)

    return round(score, 3)


def score_brevity(response: str, target_length: int = 20) -> float:
    """Score brevity - text messages should be concise.

    Score is higher for responses closer to target length.
    Very short (1-5 words) or very long (50+ words) get penalized.
    """
    words = len(response.split())

    if words == 0:
        return 0.0

    # Ideal range is 5-30 words for a text message
    if 5 <= words <= 30:
        return 1.0
    elif words < 5:
        return 0.5 + (words / 10)  # 1 word = 0.6, 4 words = 0.9
    else:
        # Penalty increases with length beyond 30 words
        penalty = min(0.5, (words - 30) / 100)
        return max(0.3, 1.0 - penalty)


def score_directness(trigger: str, response: str) -> float:
    """Score directness - does the response address the trigger?

    Penalizes:
    - Generic filler phrases ("Hey!", "Let me know if you need anything")
    - Off-topic responses
    - Not answering questions
    """
    score = 1.0
    response_lower = response.lower()
    _ = trigger.lower()  # Available for future use

    # Penalize generic greetings/filler
    generic_phrases = [
        "hey!",
        "hey there!",
        "awesome!",
        "let me know if you need anything",
        "what's on your mind",
        "how's it going",
        "just checking in",
        "let's keep the vibe",
        "we've got this",
        "i'm here for it",
        "sounds good!",
        "i get it",
    ]

    filler_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
    if filler_count > 0:
        score *= max(0.4, 1.0 - (filler_count * 0.15))

    # If trigger is a question, response should not just acknowledge
    if trigger.strip().endswith("?"):
        # Check if response is just acknowledgment without substance
        ack_phrases = ["yeah", "yes", "no", "sure", "ok", "okay", "yep", "nope"]
        response_words = response_lower.split()
        if len(response_words) <= 3:
            is_just_ack = any(w.strip(",.!?") in ack_phrases for w in response_words)
            if not is_just_ack:
                # Short non-ack response to a question is probably not direct
                score *= 0.7

    return round(score, 3)


def score_naturalness(response: str) -> float:
    """Score naturalness - does it sound like a human text message?

    Penalizes:
    - Excessive punctuation/emojis
    - Formal language
    - Assistant-like phrasing
    """
    score = 1.0
    response_lower = response.lower()

    # Penalize assistant-like phrases
    assistant_phrases = [
        "i'm here to help",
        "let me know",
        "if you need anything",
        "how can i assist",
        "is there anything",
        "just let me know",
        "happy to help",
        "i understand",
        "i see what you mean",
        "that makes sense",
    ]

    assistant_count = sum(1 for phrase in assistant_phrases if phrase in response_lower)
    if assistant_count > 0:
        score *= max(0.3, 1.0 - (assistant_count * 0.2))

    # Penalize excessive emojis (more than 2)
    emoji_count = sum(1 for c in response if ord(c) > 127000)  # rough emoji detection
    if emoji_count > 2:
        score *= 0.8

    # Penalize formal phrases
    formal_phrases = [
        "furthermore",
        "additionally",
        "in conclusion",
        "as such",
        "therefore",
        "consequently",
    ]

    if any(phrase in response_lower for phrase in formal_phrases):
        score *= 0.7

    return round(score, 3)


class ImprovedEvaluator:
    """Evaluator with semantic similarity and improved metrics."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.embedder = None
        self.generator = None
        self.db = None
        self.train_pairs = []
        self.test_pairs = []
        self.train_index = None
        self.train_embeddings = None

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Import here to avoid slow startup
            from sentence_transformers import SentenceTransformer

            from jarvis.db import get_db
            from models import get_generator

            logger.info("Initializing embedder...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info("Initializing generator...")
            self.generator = get_generator()

            logger.info("Loading database...")
            self.db = get_db()

            # Load all pairs
            all_pairs = self.db.get_all_pairs(min_quality=0.5)
            if len(all_pairs) < 100:
                logger.error("Not enough pairs in database (need at least 100)")
                return False

            # 80/20 split
            random.seed(42)
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * 0.8)
            self.train_pairs = all_pairs[:split_idx]
            self.test_pairs = all_pairs[split_idx:]

            logger.info(f"Dataset: {len(self.train_pairs)} train / {len(self.test_pairs)} test")

            # Build embeddings for train set triggers
            logger.info("Building train set embeddings...")
            train_triggers = [p.trigger_text for p in self.train_pairs]
            self.train_embeddings = self.embedder.encode(train_triggers, normalize_embeddings=True)

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        embeddings = self.embedder.encode([text1, text2], normalize_embeddings=True)
        # With normalized embeddings, dot product = cosine similarity
        return float(np.dot(embeddings[0], embeddings[1]))

    def get_template_response(self, trigger: str) -> tuple[str | None, float]:
        """Get best template match from train set."""
        trigger_emb = self.embedder.encode([trigger], normalize_embeddings=True)[0]

        # Compute similarities
        similarities = np.dot(self.train_embeddings, trigger_emb)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= 0.70:
            return self.train_pairs[best_idx].response_text, best_sim
        return None, best_sim

    def get_similar_from_train(
        self, trigger: str, k: int = 3, threshold: float = 0.3
    ) -> list[tuple[str, str]]:
        """Get similar examples from train set for few-shot."""
        trigger_emb = self.embedder.encode([trigger], normalize_embeddings=True)[0]

        similarities = np.dot(self.train_embeddings, trigger_emb)
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                pair = self.train_pairs[idx]
                results.append((pair.trigger_text, pair.response_text))

        return results

    def get_llm_response(
        self,
        trigger: str,
        contact_name: str = "Friend",
        similar_exchanges: list[tuple[str, str]] | None = None,
        conversation_context: str | None = None,
    ) -> str | None:
        """Generate LLM response."""
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        try:
            formatted_exchanges = []
            if similar_exchanges:
                for trig, resp in similar_exchanges:
                    formatted_exchanges.append((f"[Incoming]: {trig}", resp))

            if conversation_context:
                context = f"{conversation_context}\n[Incoming]: {trigger}"
            else:
                context = f"[Incoming]: {trigger}"

            prompt = build_rag_reply_prompt(
                context=context,
                last_message=trigger,
                contact_name=contact_name,
                similar_exchanges=formatted_exchanges,
                relationship_profile={
                    "tone": "casual",
                    "avg_message_length": 50,
                },
            )

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context] if context else [],
                few_shot_examples=formatted_exchanges,
                max_tokens=80,
            )

            response = self.generator.generate(request)
            return response.text.strip()

        except Exception as e:
            logger.warning(f"Error generating LLM response: {e}")
            return None

    def compare_single(self, test_pair: Any) -> ImprovedComparisonResult:
        """Compare template vs LLM with improved metrics."""
        start_time = time.time()

        trigger = test_pair.trigger_text
        actual_response = test_pair.response_text
        pair_context = getattr(test_pair, "context_text", None)

        # Get template response
        template_response, template_trigger_sim = self.get_template_response(trigger)

        # Get similar exchanges for LLM
        similar_exchanges = self.get_similar_from_train(trigger, k=3, threshold=0.3)

        # Get LLM response
        llm_response = self.get_llm_response(
            trigger,
            similar_exchanges=similar_exchanges,
            conversation_context=pair_context,
        )

        # Compute similarity to actual response (KEY METRIC)
        template_to_actual = (
            self.compute_similarity(template_response, actual_response)
            if template_response
            else 0.0
        )
        llm_to_actual = (
            self.compute_similarity(llm_response, actual_response) if llm_response else None
        )

        # Compute coherence (original and fair)
        template_coherence_orig = (
            score_coherence_original(trigger, template_response) if template_response else 0.0
        )
        llm_coherence_orig = (
            score_coherence_original(trigger, llm_response) if llm_response else None
        )

        template_coherence_fair = (
            score_coherence_fair(trigger, template_response) if template_response else 0.0
        )
        llm_coherence_fair = score_coherence_fair(trigger, llm_response) if llm_response else None

        # Compute quality metrics
        template_brevity = score_brevity(template_response) if template_response else 0.0
        llm_brevity = score_brevity(llm_response) if llm_response else None

        template_directness = (
            score_directness(trigger, template_response) if template_response else 0.0
        )
        llm_directness = score_directness(trigger, llm_response) if llm_response else None

        template_naturalness = score_naturalness(template_response) if template_response else 0.0
        llm_naturalness = score_naturalness(llm_response) if llm_response else None

        # Compute overall scores (weighted average)
        # Weights: similarity to actual (0.4), coherence (0.2), brevity (0.15),
        #          directness (0.15), naturalness (0.1)
        def compute_overall(
            sim: float,
            coh: float,
            brev: float,
            direct: float,
            natural: float,
        ) -> float:
            return 0.40 * sim + 0.20 * coh + 0.15 * brev + 0.15 * direct + 0.10 * natural

        template_overall = compute_overall(
            template_to_actual,
            template_coherence_fair,
            template_brevity,
            template_directness,
            template_naturalness,
        )

        llm_overall = None
        if llm_response and llm_to_actual is not None:
            llm_overall = compute_overall(
                llm_to_actual,
                llm_coherence_fair or 0.0,
                llm_brevity or 0.0,
                llm_directness or 0.0,
                llm_naturalness or 0.0,
            )

        # Determine winner based on overall score
        if template_response and llm_response and llm_overall is not None:
            if abs(template_overall - llm_overall) < 0.05:
                winner = "tie"
                reason = f"Similar overall ({template_overall:.2f} vs {llm_overall:.2f})"
            elif template_overall > llm_overall:
                winner = "template"
                reason = f"Higher overall ({template_overall:.2f} vs {llm_overall:.2f})"
            else:
                winner = "llm"
                reason = f"Higher overall ({llm_overall:.2f} vs {template_overall:.2f})"
        elif template_response:
            winner = "template"
            reason = "LLM generation failed"
        elif llm_response:
            winner = "llm"
            reason = "No template match"
        else:
            winner = "both_failed"
            reason = "Both failed"

        elapsed_ms = (time.time() - start_time) * 1000

        return ImprovedComparisonResult(
            trigger=trigger,
            actual_response=actual_response,
            template_response=template_response,
            llm_response=llm_response,
            template_to_actual_similarity=template_to_actual,
            llm_to_actual_similarity=llm_to_actual,
            template_coherence_original=template_coherence_orig,
            llm_coherence_original=llm_coherence_orig,
            template_coherence_fair=template_coherence_fair,
            llm_coherence_fair=llm_coherence_fair,
            template_brevity=template_brevity,
            llm_brevity=llm_brevity,
            template_directness=template_directness,
            llm_directness=llm_directness,
            template_naturalness=template_naturalness,
            llm_naturalness=llm_naturalness,
            template_overall=template_overall,
            llm_overall=llm_overall,
            winner=winner,
            reason=reason,
            comparison_time_ms=elapsed_ms,
        )

    def run_evaluation(
        self,
        num_samples: int = DEFAULT_SAMPLES,
    ) -> tuple[ImprovedEvaluationSummary | None, list[ImprovedComparisonResult]]:
        """Run the full improved evaluation."""
        start_time = time.time()

        # Sample test pairs
        if num_samples < len(self.test_pairs):
            samples = random.sample(self.test_pairs, num_samples)
        else:
            samples = self.test_pairs

        logger.info(f"Evaluating {len(samples)} samples...")

        results: list[ImprovedComparisonResult] = []

        for i, pair in enumerate(samples):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(samples) - i - 1) / rate
                logger.info(
                    f"Progress: {i + 1}/{len(samples)} "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

            result = self.compare_single(pair)
            results.append(result)

            if self.verbose and (i + 1) % 100 == 0:
                # Print interim stats
                wins = {"template": 0, "llm": 0, "tie": 0, "both_failed": 0}
                for r in results:
                    wins[r.winner] += 1
                logger.info(f"Interim: T={wins['template']} L={wins['llm']} Tie={wins['tie']}")

        # Compute summary
        total_time = time.time() - start_time

        template_wins = sum(1 for r in results if r.winner == "template")
        llm_wins = sum(1 for r in results if r.winner == "llm")
        ties = sum(1 for r in results if r.winner == "tie")
        both_failed = sum(1 for r in results if r.winner == "both_failed")

        # Compute averages
        def safe_avg(values: list[float | None]) -> float:
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else 0.0

        summary = ImprovedEvaluationSummary(
            total_samples=len(results),
            template_wins=template_wins,
            llm_wins=llm_wins,
            ties=ties,
            both_failed=both_failed,
            avg_template_to_actual_sim=safe_avg([r.template_to_actual_similarity for r in results]),
            avg_llm_to_actual_sim=safe_avg([r.llm_to_actual_similarity for r in results]),
            avg_template_coherence_original=safe_avg(
                [r.template_coherence_original for r in results]
            ),
            avg_llm_coherence_original=safe_avg([r.llm_coherence_original for r in results]),
            avg_template_coherence_fair=safe_avg([r.template_coherence_fair for r in results]),
            avg_llm_coherence_fair=safe_avg([r.llm_coherence_fair for r in results]),
            avg_template_brevity=safe_avg([r.template_brevity for r in results]),
            avg_llm_brevity=safe_avg([r.llm_brevity for r in results]),
            avg_template_directness=safe_avg([r.template_directness for r in results]),
            avg_llm_directness=safe_avg([r.llm_directness for r in results]),
            avg_template_naturalness=safe_avg([r.template_naturalness for r in results]),
            avg_llm_naturalness=safe_avg([r.llm_naturalness for r in results]),
            avg_template_overall=safe_avg([r.template_overall for r in results]),
            avg_llm_overall=safe_avg([r.llm_overall for r in results]),
            total_time_seconds=total_time,
        )

        return summary, results


def save_results(
    summary: ImprovedEvaluationSummary,
    results: list[ImprovedComparisonResult],
    output_dir: Path,
) -> None:
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary_dict = {
        "timestamp": timestamp,
        "total_samples": summary.total_samples,
        "template_wins": summary.template_wins,
        "llm_wins": summary.llm_wins,
        "ties": summary.ties,
        "both_failed": summary.both_failed,
        "template_win_rate": summary.template_wins / summary.total_samples,
        "llm_win_rate": summary.llm_wins / summary.total_samples,
        "metrics": {
            "similarity_to_actual": {
                "template": summary.avg_template_to_actual_sim,
                "llm": summary.avg_llm_to_actual_sim,
            },
            "coherence_original": {
                "template": summary.avg_template_coherence_original,
                "llm": summary.avg_llm_coherence_original,
            },
            "coherence_fair": {
                "template": summary.avg_template_coherence_fair,
                "llm": summary.avg_llm_coherence_fair,
            },
            "brevity": {
                "template": summary.avg_template_brevity,
                "llm": summary.avg_llm_brevity,
            },
            "directness": {
                "template": summary.avg_template_directness,
                "llm": summary.avg_llm_directness,
            },
            "naturalness": {
                "template": summary.avg_template_naturalness,
                "llm": summary.avg_llm_naturalness,
            },
            "overall": {
                "template": summary.avg_template_overall,
                "llm": summary.avg_llm_overall,
            },
        },
        "total_time_seconds": summary.total_time_seconds,
    }

    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Save detailed results
    results_list = []
    for r in results:
        results_list.append(
            {
                "trigger": r.trigger,
                "actual_response": r.actual_response,
                "template_response": r.template_response,
                "llm_response": r.llm_response,
                "template_to_actual_similarity": r.template_to_actual_similarity,
                "llm_to_actual_similarity": r.llm_to_actual_similarity,
                "template_coherence_original": r.template_coherence_original,
                "llm_coherence_original": r.llm_coherence_original,
                "template_coherence_fair": r.template_coherence_fair,
                "llm_coherence_fair": r.llm_coherence_fair,
                "template_brevity": r.template_brevity,
                "llm_brevity": r.llm_brevity,
                "template_directness": r.template_directness,
                "llm_directness": r.llm_directness,
                "template_naturalness": r.template_naturalness,
                "llm_naturalness": r.llm_naturalness,
                "template_overall": r.template_overall,
                "llm_overall": r.llm_overall,
                "winner": r.winner,
                "reason": r.reason,
                "comparison_time_ms": r.comparison_time_ms,
            }
        )

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_list, f, indent=2)

    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Saved results to {results_path}")


def print_summary(summary: ImprovedEvaluationSummary) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("IMPROVED EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\nDataset: {summary.total_samples} samples evaluated")
    print(f"Total Time: {summary.total_time_seconds:.1f}s")

    print("\n--- WINNER BREAKDOWN ---")
    template_win_pct = 100 * summary.template_wins / summary.total_samples
    llm_win_pct = 100 * summary.llm_wins / summary.total_samples
    print(f"Template Wins: {summary.template_wins} ({template_win_pct:.1f}%)")
    print(f"LLM Wins: {summary.llm_wins} ({llm_win_pct:.1f}%)")
    print(f"Ties: {summary.ties} ({100 * summary.ties / summary.total_samples:.1f}%)")

    print("\n--- KEY METRIC: SIMILARITY TO ACTUAL RESPONSE ---")
    print(f"Template: {summary.avg_template_to_actual_sim:.3f}")
    print(f"LLM:      {summary.avg_llm_to_actual_sim:.3f}")

    print("\n--- COHERENCE (Original vs Fair) ---")
    print(f"Template (orig): {summary.avg_template_coherence_original:.3f}")
    print(f"LLM (orig):      {summary.avg_llm_coherence_original:.3f}")
    print(f"Template (fair): {summary.avg_template_coherence_fair:.3f}")
    print(f"LLM (fair):      {summary.avg_llm_coherence_fair:.3f}")

    print("\n--- QUALITY METRICS ---")
    print(f"{'Metric':<15} {'Template':>10} {'LLM':>10}")
    print("-" * 37)
    print(f"{'Brevity':<15} {summary.avg_template_brevity:>10.3f} {summary.avg_llm_brevity:>10.3f}")
    print(
        f"{'Directness':<15} {summary.avg_template_directness:>10.3f} "
        f"{summary.avg_llm_directness:>10.3f}"
    )
    print(
        f"{'Naturalness':<15} {summary.avg_template_naturalness:>10.3f} "
        f"{summary.avg_llm_naturalness:>10.3f}"
    )

    print("\n--- OVERALL SCORE ---")
    print(f"Template: {summary.avg_template_overall:.3f}")
    print(f"LLM:      {summary.avg_llm_overall:.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Improved LLM evaluation with semantic similarity")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of test samples (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Output directory (default: {RESULTS_DIR})",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    evaluator = ImprovedEvaluator(verbose=args.verbose)

    if not evaluator.initialize():
        logger.error("Failed to initialize evaluator")
        return 1

    summary, results = evaluator.run_evaluation(num_samples=args.samples)

    if summary is None:
        logger.error("Evaluation failed")
        return 1

    # Save results
    save_results(summary, results, Path(args.output_dir))

    # Print summary
    print_summary(summary)

    return 0


if __name__ == "__main__":
    exit(main())
