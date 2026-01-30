"""
Semantic Coherence Checker

Validates that multi-message responses are coherent and not contradictory.
Uses both rule-based checks and semantic embeddings.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Contradictory phrase pairs (rule-based)
CONTRADICTORY_PHRASES = [
    ("yes", "no"),
    ("yeah", "can't"),
    ("sure", "actually"),
    ("definitely", "wait"),
    ("ok", "nevermind"),
    ("fine", "actually"),
    ("sounds good", "wait"),
    ("i'm in", "can't make it"),
    ("count me in", "count me out"),
    ("i'll be there", "can't make it"),
    ("love to", "can't"),
    ("for sure", "maybe not"),
]


def check_phrase_contradictions(response_texts: list[str]) -> bool:
    """Check for hardcoded contradictory phrase pairs.

    Args:
        response_texts: List of response message texts

    Returns:
        True if coherent, False if contradictory
    """
    if len(response_texts) <= 1:
        return True

    combined = " ".join(response_texts).lower()

    for phrase1, phrase2 in CONTRADICTORY_PHRASES:
        if phrase1 in combined and phrase2 in combined:
            logger.debug(
                "Found phrase contradiction: '%s' and '%s' in: %s", phrase1, phrase2, combined[:60]
            )
            return False

    return True


def check_semantic_coherence(
    response_texts: list[str], model: Any, similarity_threshold: float = 0.3
) -> bool:
    """Check semantic coherence using embeddings.

    Flags responses where consecutive messages have very low similarity,
    indicating a contradiction or topic shift.

    Args:
        response_texts: List of response message texts
        model: SentenceTransformer model
        similarity_threshold: Minimum similarity between consecutive messages

    Returns:
        True if coherent, False if incoherent
    """
    if len(response_texts) <= 1:
        return True

    try:
        # Generate embeddings
        embeddings = model.encode(response_texts, convert_to_numpy=True)

        # Check pairwise similarity
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if similarity < similarity_threshold:
                logger.debug(
                    "Low semantic similarity (%.2f) between messages: '%s' and '%s'",
                    similarity,
                    response_texts[i][:40],
                    response_texts[i + 1][:40],
                )
                return False

        return True

    except Exception as e:
        logger.warning("Failed to check semantic coherence: %s", e)
        return True  # Default to accepting if check fails


def check_temporal_contradictions(response_texts: list[str]) -> bool:
    """Check for temporal contradictions (conflicting times/dates).

    Args:
        response_texts: List of response message texts

    Returns:
        True if coherent, False if contradictory
    """
    if len(response_texts) <= 1:
        return True

    combined = " ".join(response_texts).lower()

    # Time patterns
    import re

    times = re.findall(r"\b(\d{1,2})\s*(am|pm|:00|:30)?\b", combined)

    # If multiple different times mentioned, flag as suspicious
    if len(set(times)) > 1:
        # Check if they're actually different
        unique_hours = set(t[0] for t in times)
        if len(unique_hours) > 1:
            logger.debug("Multiple different times mentioned: %s in: %s", times, combined[:60])
            return False

    return True


def is_coherent_response(
    response_texts: list[str], model: Any | None = None, use_semantic_check: bool = True
) -> bool:
    """Check if multi-message response is coherent.

    Combines multiple coherence checks:
    1. Phrase-based contradiction detection
    2. Semantic similarity check (if model provided)
    3. Temporal contradiction detection

    Args:
        response_texts: List of response message texts
        model: Optional SentenceTransformer model for semantic check
        use_semantic_check: Whether to use semantic checking

    Returns:
        True if coherent, False otherwise
    """
    if len(response_texts) <= 1:
        return True

    # Rule-based checks
    if not check_phrase_contradictions(response_texts):
        return False

    if not check_temporal_contradictions(response_texts):
        return False

    # Semantic check (if enabled and model available)
    if use_semantic_check and model is not None:
        if not check_semantic_coherence(response_texts, model):
            return False

    return True


def calculate_coherence_score(response_texts: list[str], model: Any | None = None) -> float:
    """Calculate a coherence score (0-1).

    Args:
        response_texts: List of response message texts
        model: Optional SentenceTransformer model

    Returns:
        Coherence score from 0 (incoherent) to 1 (perfectly coherent)
    """
    if len(response_texts) <= 1:
        return 1.0

    score = 1.0

    # Phrase contradiction check
    if not check_phrase_contradictions(response_texts):
        score -= 0.5

    # Temporal contradiction check
    if not check_temporal_contradictions(response_texts):
        score -= 0.3

    # Semantic coherence (if model available)
    if model is not None:
        try:
            embeddings = model.encode(response_texts, convert_to_numpy=True)
            similarities = []

            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)

            # Average similarity
            avg_similarity = np.mean(similarities)

            # Penalize low similarity
            if avg_similarity < 0.3:
                score -= 0.3
            elif avg_similarity < 0.5:
                score -= 0.2

        except Exception as e:
            logger.warning("Failed to calculate semantic score: %s", e)

    return max(0.0, min(1.0, score))
