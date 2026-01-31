#!/usr/bin/env python3
"""Score pair quality using multiple signals.

Identifies which pairs are true Q&A vs topic shifts by checking:
1. Trigger-Response semantic similarity
2. Response addresses the trigger (coherence)
3. Not a reaction (Liked, Loved, etc.)
4. Response isn't a new topic introduction

Usage:
    python -m scripts.score_pair_quality --analyze      # Analyze current pairs
    python -m scripts.score_pair_quality --update       # Update quality scores
    python -m scripts.score_pair_quality --threshold 0.6  # Custom threshold
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db import get_db
from jarvis.embedding_adapter import get_embedder

logger = logging.getLogger(__name__)

# Reactions that aren't real responses
REACTION_PATTERNS = [
    r'^Liked ".*"$',
    r'^Loved ".*"$',
    r'^Laughed at ".*"$',
    r'^Emphasized ".*"$',
    r'^Disliked ".*"$',
    r'^Questioned ".*"$',
]

# New topic indicators (response introduces new subject)
NEW_TOPIC_STARTERS = [
    "btw ",
    "by the way",
    "anyway ",
    "oh also",
    "also ",
    "speaking of",
    "unrelated but",
    "random but",
    "hey so ",
    "so anyway",
]

# Acknowledgment-only triggers that need special handling
ACKNOWLEDGMENT_TRIGGERS = {
    "ok",
    "okay",
    "k",
    "kk",
    "yes",
    "yeah",
    "yep",
    "sure",
    "cool",
    "nice",
    "got it",
    "sounds good",
    "alright",
    "np",
    "no problem",
    "thanks",
    "ty",
}


def is_reaction(text: str) -> bool:
    """Check if text is a tapback reaction."""
    for pattern in REACTION_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


def is_new_topic(text: str) -> bool:
    """Check if text starts a new topic."""
    text_lower = text.lower().strip()
    return any(text_lower.startswith(starter) for starter in NEW_TOPIC_STARTERS)


def is_acknowledgment_trigger(text: str) -> bool:
    """Check if trigger is just an acknowledgment."""
    text_lower = text.lower().strip().rstrip("!.?")
    return text_lower in ACKNOWLEDGMENT_TRIGGERS


def compute_coherence_score(
    trigger: str,
    response: str,
    context: str | None,
    embedder,
) -> dict:
    """Compute coherence score between trigger and response.

    Returns dict with:
        - trigger_response_sim: Direct similarity
        - is_reaction: True if response is a tapback
        - is_new_topic: True if response starts new topic
        - is_ack_trigger: True if trigger is just acknowledgment
        - coherence_score: Final 0-1 score
        - verdict: 'good', 'mediocre', 'bad'
    """
    result = {
        "trigger_response_sim": 0.0,
        "is_reaction": is_reaction(response),
        "is_new_topic": is_new_topic(response),
        "is_ack_trigger": is_acknowledgment_trigger(trigger),
        "coherence_score": 0.0,
        "verdict": "unknown",
    }

    # Reactions are never good pairs
    if result["is_reaction"]:
        result["coherence_score"] = 0.1
        result["verdict"] = "bad"
        return result

    # Compute semantic similarity
    embs = embedder.encode([trigger, response], normalize=True)
    sim = float(np.dot(embs[0], embs[1]))
    result["trigger_response_sim"] = round(sim, 3)

    # Acknowledgment triggers with low similarity = likely topic shift
    if result["is_ack_trigger"] and sim < 0.55:
        result["coherence_score"] = 0.2
        result["verdict"] = "bad"
        return result

    # New topic starters = not a true reply
    if result["is_new_topic"]:
        result["coherence_score"] = 0.3
        result["verdict"] = "mediocre"
        return result

    # Score based on similarity with bonuses/penalties
    score = sim

    # Bonus for high similarity
    if sim > 0.7:
        score = min(1.0, score + 0.1)

    # Penalty for very short responses to questions
    if trigger.strip().endswith("?") and len(response.split()) < 3:
        score *= 0.8

    result["coherence_score"] = round(score, 3)

    if score >= 0.6:
        result["verdict"] = "good"
    elif score >= 0.45:
        result["verdict"] = "mediocre"
    else:
        result["verdict"] = "bad"

    return result


def analyze_pairs(limit: int = 100):
    """Analyze pair quality distribution."""
    db = get_db()
    embedder = get_embedder()

    pairs = db.get_all_pairs(min_quality=0.0)[:limit]

    results = {"good": 0, "mediocre": 0, "bad": 0}
    examples = {"good": [], "mediocre": [], "bad": []}

    print(f"Analyzing {len(pairs)} pairs...")

    for p in pairs:
        score = compute_coherence_score(p.trigger_text, p.response_text, p.context_text, embedder)
        verdict = score["verdict"]
        results[verdict] += 1

        if len(examples[verdict]) < 3:
            examples[verdict].append(
                {
                    "trigger": p.trigger_text[:50],
                    "response": p.response_text[:50],
                    "sim": score["trigger_response_sim"],
                    "coherence": score["coherence_score"],
                }
            )

    print("\n=== PAIR QUALITY DISTRIBUTION ===")
    total = len(pairs)
    for verdict, count in results.items():
        pct = count / total * 100
        print(f"  {verdict.upper():10s}: {count:5d} ({pct:5.1f}%)")

    print("\n=== EXAMPLES ===")
    for verdict in ["good", "mediocre", "bad"]:
        print(f"\n{verdict.upper()}:")
        for ex in examples[verdict]:
            print(f"  T: {ex['trigger']}...")
            print(f"  R: {ex['response']}...")
            print(f"  sim={ex['sim']:.2f}, coherence={ex['coherence']:.2f}")
            print()


def update_quality_scores(threshold: float = 0.5, dry_run: bool = True):
    """Update pair quality scores based on coherence analysis."""
    db = get_db()
    embedder = get_embedder()

    pairs = db.get_all_pairs(min_quality=0.0)

    updates = {"upgraded": 0, "downgraded": 0, "unchanged": 0}

    print(f"Scoring {len(pairs)} pairs...")

    with db.connection() as conn:
        for i, p in enumerate(pairs):
            score = compute_coherence_score(
                p.trigger_text, p.response_text, p.context_text, embedder
            )

            new_quality = score["coherence_score"]
            old_quality = p.quality_score

            # Blend with existing quality (don't completely override)
            blended = old_quality * 0.3 + new_quality * 0.7

            if abs(blended - old_quality) < 0.05:
                updates["unchanged"] += 1
            elif blended > old_quality:
                updates["upgraded"] += 1
            else:
                updates["downgraded"] += 1

            if not dry_run:
                conn.execute(
                    "UPDATE pairs SET quality_score = ? WHERE id = ?", (round(blended, 2), p.id)
                )

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(pairs)}")

    print("\n=== QUALITY SCORE UPDATES ===")
    print(f"  Upgraded:   {updates['upgraded']}")
    print(f"  Downgraded: {updates['downgraded']}")
    print(f"  Unchanged:  {updates['unchanged']}")

    if dry_run:
        print("\n(Dry run - no changes made. Use --commit to apply.)")


def main():
    parser = argparse.ArgumentParser(description="Score pair quality")
    parser.add_argument("--analyze", action="store_true", help="Analyze pair quality distribution")
    parser.add_argument("--update", action="store_true", help="Update quality scores")
    parser.add_argument(
        "--commit", action="store_true", help="Actually commit updates (not dry run)"
    )
    parser.add_argument("--limit", type=int, default=500, help="Max pairs to analyze")
    parser.add_argument("--threshold", type=float, default=0.5, help="Quality threshold")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.analyze:
        analyze_pairs(limit=args.limit)
    elif args.update:
        update_quality_scores(threshold=args.threshold, dry_run=not args.commit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
