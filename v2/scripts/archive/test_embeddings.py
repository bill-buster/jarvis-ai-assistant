#!/usr/bin/env python3
"""Test embedding cache functionality.

Run with: python -m v2.scripts.test_embeddings
"""

from __future__ import annotations

import time


def main():
    print("=" * 60)
    print("JARVIS v2 Embedding Cache Test")
    print("=" * 60)

    from core.embeddings import (
        get_embedding_cache,
        find_most_similar,
        cosine_similarity,
    )

    cache = get_embedding_cache()
    print(f"\nCache location: {cache.db_path}")

    # Test 1: Basic embedding
    print("\n--- Test 1: Basic Embedding ---")
    text = "Can we meet tomorrow at 3pm?"

    start = time.time()
    embedding1 = cache.get_or_compute(text)
    first_time = (time.time() - start) * 1000
    print(f"First embedding: {first_time:.1f}ms (cache miss expected)")
    print(f"Embedding shape: {embedding1.shape}")

    # Test 2: Cached retrieval
    print("\n--- Test 2: Cached Retrieval ---")
    start = time.time()
    embedding2 = cache.get_or_compute(text)
    second_time = (time.time() - start) * 1000
    print(f"Second embedding: {second_time:.1f}ms (cache hit expected)")
    print(f"Speedup: {first_time / second_time:.1f}x")

    # Verify same embedding
    sim = cosine_similarity(embedding1, embedding2)
    print(f"Similarity (should be 1.0): {sim:.4f}")

    # Test 3: Similar text detection
    print("\n--- Test 3: Semantic Similarity ---")
    candidates = [
        "Yes, 3pm works for me!",
        "I'm busy tomorrow, sorry.",
        "Let me check my calendar.",
        "What about 4pm instead?",
        "The weather is nice today.",
    ]

    results = find_most_similar(text, candidates, cache)
    print(f"Query: {text}")
    print("Most similar responses:")
    for candidate, score in results:
        print(f"  [{score:.3f}] {candidate}")

    # Test 4: Batch embedding
    print("\n--- Test 4: Batch Embedding ---")
    texts = [
        "Good morning!",
        "How are you?",
        "Let's grab lunch",
        "See you later!",
    ]

    start = time.time()
    embeddings = cache.get_or_compute_batch(texts)
    batch_time = (time.time() - start) * 1000
    print(f"Batch embedding ({len(texts)} texts): {batch_time:.1f}ms")
    print(f"Average per text: {batch_time / len(texts):.1f}ms")

    # Test 5: Cache stats
    print("\n--- Test 5: Cache Statistics ---")
    stats = cache.stats()
    print(f"Total entries: {stats.total_entries}")
    print(f"Hits: {stats.hits}")
    print(f"Misses: {stats.misses}")
    print(f"Hit rate: {stats.hit_rate:.1%}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
