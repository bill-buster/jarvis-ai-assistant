#!/usr/bin/env python3
"""Proper Evaluation with Train/Test Split and Few-Shot Examples.

This is a REAL evaluation:
1. Split pairs into train (80%) and test (20%) PER CONTACT
2. Pre-compute embeddings for all training pairs (fast!)
3. Find similar examples via vector lookup
4. Test on held-out data the model has NEVER seen

Run: uv run python scripts/experiments/proper_eval.py
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TEST_SAMPLES = 200
RESULTS_DIR = Path("results/proper_eval")


@dataclass
class EvalResult:
    trigger: str
    actual_response: str
    generated_response: str
    relationship: str
    similarity_to_actual: float
    num_few_shot_examples: int
    generation_time_ms: float


def run_evaluation() -> None:
    """Run proper evaluation with pre-computed embeddings."""
    from contracts.models import GenerationRequest
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder
    from models import get_generator

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Proper Evaluation - Train/Test Split + Few-Shot")
    logger.info("=" * 60)

    # Load data
    logger.info("\n📂 Loading data...")
    db = get_db()
    all_pairs = db.get_all_pairs(min_quality=0.5)
    pairs_with_contacts = [p for p in all_pairs if p.contact_id]
    logger.info(f"   Pairs with contacts: {len(pairs_with_contacts)}")

    # Group by contact
    pairs_by_contact: dict[int, list] = defaultdict(list)
    for p in pairs_with_contacts:
        pairs_by_contact[p.contact_id].append(p)
    logger.info(f"   Unique contacts: {len(pairs_by_contact)}")

    # Split train/test PER CONTACT (80/20)
    logger.info("\n📊 Splitting train/test by contact...")
    train_by_contact: dict[int, list] = {}
    test_pairs: list = []

    random.seed(42)
    for contact_id, contact_pairs in pairs_by_contact.items():
        random.shuffle(contact_pairs)
        split_idx = int(len(contact_pairs) * 0.8)
        train_by_contact[contact_id] = contact_pairs[:split_idx]
        test_pairs.extend(contact_pairs[split_idx:])

    total_train = sum(len(v) for v in train_by_contact.values())
    logger.info(f"   Train: {total_train} pairs")
    logger.info(f"   Test: {len(test_pairs)} pairs (held out)")

    # Sample test pairs
    if len(test_pairs) > TEST_SAMPLES:
        test_pairs = random.sample(test_pairs, TEST_SAMPLES)
    logger.info(f"   Eval samples: {len(test_pairs)}")

    # Initialize components
    logger.info("\n🔗 Initializing...")
    embedder = get_embedder()
    generator = get_generator()

    # PRE-COMPUTE all training embeddings (do this ONCE - fast!)
    logger.info("\n⚡ Pre-computing training embeddings...")
    train_embeddings: dict[int, tuple[np.ndarray, list]] = {}

    start_precompute = time.time()
    total_texts = 0
    for contact_id, train_pairs in train_by_contact.items():
        if not train_pairs:
            continue
        triggers = [p.trigger_text for p in train_pairs]
        embs = embedder.encode(triggers, normalize=True)
        train_embeddings[contact_id] = (embs, train_pairs)
        total_texts += len(triggers)

    precompute_time = time.time() - start_precompute
    logger.info(f"   Embedded {total_texts} texts in {precompute_time:.1f}s")
    logger.info(f"   ({total_texts / precompute_time:.0f} texts/sec)")

    # Build contact lookup
    contact_cache = {cid: db.get_contact(cid) for cid in pairs_by_contact.keys()}

    # Run evaluation
    logger.info("\n🏃 Running evaluation...")
    results: list[EvalResult] = []
    results_by_relationship: dict[str, list[EvalResult]] = defaultdict(list)
    total_time = 0

    for i, pair in enumerate(test_pairs):
        if (i + 1) % 25 == 0:
            avg_time = total_time / max(i, 1)
            logger.info(f"   Progress: {i + 1}/{len(test_pairs)} ({avg_time:.0f}ms avg)")

        trigger = pair.trigger_text
        actual = pair.response_text
        contact_id = pair.contact_id

        # Get contact info
        contact = contact_cache.get(contact_id)
        relationship = contact.relationship if contact else "unknown"

        # FAST few-shot lookup using pre-computed embeddings
        few_shot = []
        if contact_id in train_embeddings:
            train_embs, train_pairs = train_embeddings[contact_id]

            # Encode just the test trigger (fast - single text)
            trigger_emb = embedder.encode([trigger], normalize=True)[0]

            # Fast vector similarity lookup
            similarities = np.dot(train_embs, trigger_emb)
            top_indices = np.argsort(similarities)[-3:][::-1]

            for idx in top_indices:
                if similarities[idx] > 0.3:  # Minimum threshold
                    p = train_pairs[idx]
                    # Truncate examples to avoid huge prompts
                    few_shot.append((p.trigger_text[:200], p.response_text[:200]))

        # Build prompt with few-shot examples
        if few_shot:
            examples_text = "\n".join([f"Message: {t}\nYour reply: {r}" for t, r in few_shot])
            prompt = (
                f"You text like this with your {relationship}:\n\n"
                f"{examples_text}\n\n"
                f"Reply in the same style to: {trigger}"
            )
        else:
            prompt = f"Reply briefly as a {relationship}: {trigger}"

        # Generate response
        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=few_shot,
                max_tokens=80,
                temperature=0.7,
            )

            start = time.time()
            response = generator.generate(request)
            gen_time = (time.time() - start) * 1000
            total_time += gen_time

            generated = response.text

            # Compute similarity to actual response
            embeddings = embedder.encode([actual, generated], normalize=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))

        except Exception as e:
            logger.error(f"Error: {e}")
            generated = ""
            similarity = 0
            gen_time = 0

        result = EvalResult(
            trigger=trigger[:100],
            actual_response=actual[:100],
            generated_response=generated[:100] if generated else "",
            relationship=relationship or "unknown",
            similarity_to_actual=similarity,
            num_few_shot_examples=len(few_shot),
            generation_time_ms=gen_time,
        )
        results.append(result)
        results_by_relationship[relationship or "unknown"].append(result)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS BY RELATIONSHIP")
    logger.info("=" * 60)

    logger.info(f"\n{'Relationship':<18} {'Count':>6} {'Similarity':>12} {'Avg Examples':>14}")
    logger.info("-" * 55)

    for rel in sorted(results_by_relationship.keys()):
        rel_results = results_by_relationship[rel]
        count = len(rel_results)
        avg_sim = np.mean([r.similarity_to_actual for r in rel_results])
        avg_examples = np.mean([r.num_few_shot_examples for r in rel_results])
        logger.info(f"{rel:<18} {count:>6} {avg_sim:>12.3f} {avg_examples:>14.1f}")

    # Impact of few-shot examples
    logger.info("\n📈 IMPACT OF FEW-SHOT EXAMPLES:")
    for n_examples in [0, 1, 2, 3]:
        subset = [r for r in results if r.num_few_shot_examples == n_examples]
        if subset:
            avg_sim = np.mean([r.similarity_to_actual for r in subset])
            logger.info(f"   {n_examples} examples: {avg_sim:.3f} ({len(subset)} samples)")

    # Show examples
    logger.info("\n" + "=" * 60)
    logger.info("📝 SAMPLE OUTPUTS")
    logger.info("=" * 60)

    # Best example
    best = max(results, key=lambda r: r.similarity_to_actual)
    logger.info(
        f"\n🟢 Best (sim={best.similarity_to_actual:.2f}, {best.num_few_shot_examples} examples):"
    )
    logger.info(f"   Relationship: {best.relationship}")
    logger.info(f"   Trigger:   {best.trigger}")
    logger.info(f"   Generated: {best.generated_response}")
    logger.info(f"   Actual:    {best.actual_response}")

    # Worst example
    worst = min(results, key=lambda r: r.similarity_to_actual)
    logger.info(
        f"\n🔴 Worst (sim={worst.similarity_to_actual:.2f}, "
        f"{worst.num_few_shot_examples} examples):"
    )
    logger.info(f"   Relationship: {worst.relationship}")
    logger.info(f"   Trigger:   {worst.trigger}")
    logger.info(f"   Generated: {worst.generated_response}")
    logger.info(f"   Actual:    {worst.actual_response}")

    # Overall statistics
    logger.info("\n" + "=" * 60)
    logger.info("📈 OVERALL SUMMARY")
    logger.info("=" * 60)

    overall_sim = np.mean([r.similarity_to_actual for r in results])
    overall_time = np.mean([r.generation_time_ms for r in results])
    avg_examples = np.mean([r.num_few_shot_examples for r in results])

    logger.info(f"\nOverall similarity: {overall_sim:.3f}")
    logger.info(f"Avg few-shot examples: {avg_examples:.1f}")
    logger.info(f"Avg generation time: {overall_time:.0f}ms")
    logger.info(f"Total eval time: {total_time / 1000:.1f}s")

    # Comparison with baselines
    logger.info("\n📊 COMPARISON:")
    logger.info("   Raw LLM (no context):        0.598")
    logger.info("   + Relationship context:      0.608 (+1.7%)")
    improvement = (overall_sim - 0.598) / 0.598 * 100
    logger.info(f"   + Few-shot from history:     {overall_sim:.3f} ({improvement:+.1f}%)")

    # Save results
    output_file = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "test_samples": len(test_pairs),
                    "train_pairs": total_train,
                },
                "overall_similarity": overall_sim,
                "avg_few_shot_examples": avg_examples,
                "by_relationship": {
                    rel: {
                        "count": len(rel_results),
                        "avg_similarity": float(
                            np.mean([r.similarity_to_actual for r in rel_results])
                        ),
                    }
                    for rel, rel_results in results_by_relationship.items()
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    logger.info(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
