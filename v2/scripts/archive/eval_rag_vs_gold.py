#!/usr/bin/env python3
"""Compare RAG-enhanced generation vs gold responses.

Measures: Does RAG make generated replies more similar to what you actually said?

Run: uv run python scripts/eval_rag_vs_gold.py --samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Setup path
script_dir = Path(__file__).parent.resolve()
v2_dir = script_dir.parent
sys.path.insert(0, str(v2_dir))

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG vs gold responses")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--verbose", action="store_true", help="Show individual results")
    args = parser.parse_args()

    print("=" * 70)
    print("RAG vs GOLD RESPONSE EVALUATION")
    print("=" * 70)

    # Load test data
    test_path = v2_dir / "results" / "test_set" / "clean_test_data.jsonl"
    with open(test_path) as f:
        all_samples = [json.loads(line) for line in f]

    # Filter to non-group chats with reasonable length responses
    samples = [
        s for s in all_samples
        if not s.get("is_group", False)
        and len(s.get("gold_response", "")) > 5
        and len(s.get("gold_response", "")) < 200
    ][:args.samples]

    print(f"Testing {len(samples)} samples\n")

    # Load embedding model for similarity
    from core.embeddings.model import get_embedding_model
    embed_model = get_embedding_model()

    # Load relationship registry to resolve contact names to chat_ids
    from core.embeddings.relationship_registry import get_relationship_registry
    from core.embeddings import get_embedding_store
    registry = get_relationship_registry()
    store = get_embedding_store()

    # Load generator
    from core.models.loader import ModelLoader
    from core.generation.reply_generator import ReplyGenerator

    loader = ModelLoader()
    gen = ReplyGenerator(loader)

    def resolve_contact_to_chat_id(contact_name: str) -> str | None:
        """Resolve a contact name to a chat_id."""
        # Try direct lookup
        info = registry.get_relationship(contact_name)
        if info and info.phones:
            chat_ids = store.resolve_phones_to_chatids(info.phones)
            if chat_ids:
                return list(chat_ids)[0]
        return None

    # Results
    results = {
        "with_rag": [],
        "without_rag": [],
    }

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample['contact'][:30]}...")
        print(f"  Their msg: \"{sample['last_message'][:50]}...\"")
        print(f"  Gold: \"{sample['gold_response'][:50]}\"")

        # Parse conversation into messages
        messages = []
        for line in sample.get("conversation", "").split("\n"):
            line = line.strip()
            if line.startswith("them:"):
                messages.append({"sender": "them", "text": line[5:].strip()})
            elif line.startswith("me:"):
                messages.append({"sender": "me", "text": line[3:].strip()})

        if not messages:
            messages = [{"sender": "them", "text": sample["last_message"]}]

        # Embed gold response
        gold_emb = embed_model.embed(sample["gold_response"])

        # Resolve contact name to chat_id for RAG
        contact_name = sample.get("contact", "")
        chat_id = resolve_contact_to_chat_id(contact_name)

        # Generate WITH RAG
        try:
            result_with = gen.generate_replies(
                messages=messages,
                chat_id=chat_id,  # Real chat_id for cross-conversation RAG
                num_replies=1,
                contact_name=contact_name,  # Fallback for group chats
            )
            gen_with_rag = result_with.replies[0].text if result_with.replies else ""
            rag_examples = len(result_with.past_replies)
        except Exception as e:
            print(f"  Error with RAG: {e}")
            gen_with_rag = ""
            rag_examples = 0

        # Generate WITHOUT RAG by temporarily disabling it
        # We'll simulate this by using an empty chat_id that won't match
        try:
            # Temporarily disable RAG by setting a flag or using minimal context
            old_method = gen._find_past_replies
            gen._find_past_replies = lambda *args, **kwargs: []

            result_without = gen.generate_replies(
                messages=messages,
                chat_id=None,
                num_replies=1,
            )
            gen_without_rag = result_without.replies[0].text if result_without.replies else ""

            # Restore
            gen._find_past_replies = old_method
        except Exception as e:
            print(f"  Error without RAG: {e}")
            gen_without_rag = ""

        # Calculate similarities to gold
        if gen_with_rag:
            with_emb = embed_model.embed(gen_with_rag)
            sim_with = cosine_similarity(gold_emb, with_emb)
        else:
            sim_with = 0.0

        if gen_without_rag:
            without_emb = embed_model.embed(gen_without_rag)
            sim_without = cosine_similarity(gold_emb, without_emb)
        else:
            sim_without = 0.0

        results["with_rag"].append(sim_with)
        results["without_rag"].append(sim_without)

        if args.verbose or True:  # Always show for now
            print(f"  WITH RAG ({rag_examples} examples): \"{gen_with_rag[:50]}\" [sim={sim_with:.3f}]")
            print(f"  WITHOUT RAG: \"{gen_without_rag[:50]}\" [sim={sim_without:.3f}]")
            winner = "WITH RAG" if sim_with > sim_without else "WITHOUT RAG" if sim_without > sim_with else "TIE"
            print(f"  Winner: {winner}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    avg_with = np.mean(results["with_rag"])
    avg_without = np.mean(results["without_rag"])

    print(f"\nAverage similarity to GOLD response:")
    print(f"  WITH RAG:    {avg_with:.3f}")
    print(f"  WITHOUT RAG: {avg_without:.3f}")
    print(f"  Difference:  {avg_with - avg_without:+.3f}")

    wins_with = sum(1 for w, wo in zip(results["with_rag"], results["without_rag"]) if w > wo)
    wins_without = sum(1 for w, wo in zip(results["with_rag"], results["without_rag"]) if wo > w)
    ties = len(samples) - wins_with - wins_without

    print(f"\nHead-to-head:")
    print(f"  RAG wins:     {wins_with}/{len(samples)} ({wins_with/len(samples)*100:.0f}%)")
    print(f"  No-RAG wins:  {wins_without}/{len(samples)} ({wins_without/len(samples)*100:.0f}%)")
    print(f"  Ties:         {ties}/{len(samples)} ({ties/len(samples)*100:.0f}%)")

    if avg_with > avg_without:
        print(f"\n✅ RAG improves similarity to your actual responses by {(avg_with-avg_without)*100:.1f}%")
    else:
        print(f"\n❌ RAG does NOT improve similarity (diff: {(avg_with-avg_without)*100:.1f}%)")


if __name__ == "__main__":
    main()
