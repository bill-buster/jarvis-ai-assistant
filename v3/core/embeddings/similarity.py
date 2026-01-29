"""Similarity utilities for JARVIS v2.

Provides cosine similarity and related functions for semantic matching.
"""

from __future__ import annotations

import numpy as np

from .cache import EmbeddingCache, get_embedding_cache


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (0-1 range, higher = more similar)
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def find_most_similar(
    query: str,
    candidates: list[str],
    cache: EmbeddingCache | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find most similar candidates to query.

    Args:
        query: Query text
        candidates: List of candidate texts
        cache: Embedding cache (uses singleton if not provided)
        top_k: Number of top results to return

    Returns:
        List of (candidate, similarity) tuples, sorted by similarity descending
    """
    if cache is None:
        cache = get_embedding_cache()

    # Get query embedding
    query_embedding = cache.get_or_compute(query)

    # Get candidate embeddings (batch for efficiency)
    candidate_embeddings = cache.get_or_compute_batch(candidates)

    # Compute similarities
    similarities = []
    for candidate, embedding in zip(candidates, candidate_embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((candidate, sim))

    # Sort by similarity (descending) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def semantic_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.7,
    cache: EmbeddingCache | None = None,
) -> str | None:
    """Find best semantic match above threshold.

    Args:
        query: Query text
        candidates: List of candidate texts
        threshold: Minimum similarity threshold
        cache: Embedding cache (uses singleton if not provided)

    Returns:
        Best matching candidate, or None if no match above threshold
    """
    results = find_most_similar(query, candidates, cache, top_k=1)

    if results and results[0][1] >= threshold:
        return results[0][0]

    return None


def batch_cosine_similarity(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities between query and multiple embeddings.

    Optimized for batch processing.

    Args:
        query_embedding: Query vector (1D)
        embeddings: Matrix of embeddings (2D, each row is an embedding)

    Returns:
        Array of similarity scores
    """
    # Normalize vectors
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    embedding_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Dot product for cosine similarity
    similarities = np.dot(embedding_norms, query_norm)
    return similarities
