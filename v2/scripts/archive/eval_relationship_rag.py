#!/usr/bin/env python3
"""Evaluate the relationship-aware RAG pipeline.

Measures:
1. Coverage: % of queries that find past reply examples
2. Relevance: Average similarity score of found examples
3. Speed: Search latency

Run: uv run python scripts/eval_relationship_rag.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Setup path
script_dir = Path(__file__).parent.resolve()
v2_dir = script_dir.parent
sys.path.insert(0, str(v2_dir))
os.chdir(v2_dir)


# Test queries representing different intents
TEST_QUERIES = [
    # Social/hangout
    "wanna hang out?",
    "you free tonight?",
    "what are you doing this weekend?",
    "let's grab dinner",
    "wanna get food?",
    # Logistics
    "what time works?",
    "where should we meet?",
    "are you coming?",
    "when do you get here?",
    "running late",
    # Casual chat
    "what's up?",
    "how's it going?",
    "lol that's hilarious",
    "haha nice",
    "omg really?",
    # Questions
    "did you see that?",
    "have you tried it?",
    "what do you think?",
    "should I do it?",
    "which one?",
]


def main():
    print("\n" + "=" * 60)
    print("RELATIONSHIP-AWARE RAG EVALUATION")
    print("=" * 60)

    from core.embeddings import get_embedding_store
    from core.embeddings.relationship_registry import get_relationship_registry

    store = get_embedding_store()
    registry = get_relationship_registry()

    # Warm up the index
    print("\nWarming up indexes...")
    _ = store._get_or_build_reply_pairs_index()
    _ = store._get_or_build_phone_to_chatids()

    # Get friend chat_ids for filtered search
    similar = registry.get_similar_contacts("Ishani Desai")[:50]
    phones = [p for phones in registry.get_phones_for_contacts(similar).values() for p in phones]
    friend_chat_ids = store.resolve_phones_to_chatids(phones)

    print(f"Testing {len(TEST_QUERIES)} queries...")
    print(f"Friend filter: {len(friend_chat_ids)} chat_ids")
    print()

    # Evaluate global search
    print("-" * 60)
    print("GLOBAL SEARCH (all conversations)")
    print("-" * 60)
    global_results = evaluate_queries(store, TEST_QUERIES, target_chat_ids=None)
    print_results(global_results)

    # Evaluate filtered search (friends only)
    print("-" * 60)
    print("FILTERED SEARCH (friends only)")
    print("-" * 60)
    filtered_results = evaluate_queries(store, TEST_QUERIES, target_chat_ids=friend_chat_ids)
    print_results(filtered_results)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nGlobal search:")
    print(f"  Coverage: {global_results['coverage']*100:.0f}%")
    print(f"  Avg similarity: {global_results['avg_similarity']:.2f}")
    print(f"  Avg latency: {global_results['avg_latency_ms']:.0f}ms")

    print(f"\nFiltered search (friends):")
    print(f"  Coverage: {filtered_results['coverage']*100:.0f}%")
    print(f"  Avg similarity: {filtered_results['avg_similarity']:.2f}")
    print(f"  Avg latency: {filtered_results['avg_latency_ms']:.0f}ms")

    print("\n✅ Evaluation complete!")


def evaluate_queries(store, queries, target_chat_ids):
    """Evaluate search for a set of queries."""
    results = {
        "queries_with_results": 0,
        "total_results": 0,
        "total_similarity": 0.0,
        "total_latency_ms": 0.0,
        "details": [],
    }

    for query in queries:
        start = time.time()
        search_results = store.find_your_past_replies_cross_conversation(
            incoming_message=query,
            target_chat_ids=target_chat_ids,
            limit=5,
            min_similarity=0.45,
        )
        latency = (time.time() - start) * 1000

        results["total_latency_ms"] += latency

        if search_results:
            results["queries_with_results"] += 1
            results["total_results"] += len(search_results)

            # Sum similarities
            for _, _, score, _ in search_results:
                results["total_similarity"] += score

            results["details"].append({
                "query": query,
                "count": len(search_results),
                "top_score": search_results[0][2],
                "latency_ms": latency,
            })

    n = len(queries)
    results["coverage"] = results["queries_with_results"] / n if n > 0 else 0
    results["avg_latency_ms"] = results["total_latency_ms"] / n if n > 0 else 0
    results["avg_similarity"] = (
        results["total_similarity"] / results["total_results"]
        if results["total_results"] > 0
        else 0
    )

    return results


def print_results(results):
    print(f"\nCoverage: {results['coverage']*100:.0f}% ({results['queries_with_results']}/{len(TEST_QUERIES)} queries)")
    print(f"Total results: {results['total_results']}")
    print(f"Avg similarity: {results['avg_similarity']:.2f}")
    print(f"Avg latency: {results['avg_latency_ms']:.0f}ms")

    if results["details"]:
        print("\nTop queries with results:")
        sorted_details = sorted(results["details"], key=lambda x: x["top_score"], reverse=True)
        for d in sorted_details[:5]:
            print(f"  [{d['top_score']:.2f}] \"{d['query']}\" → {d['count']} results ({d['latency_ms']:.0f}ms)")
    print()


if __name__ == "__main__":
    main()
