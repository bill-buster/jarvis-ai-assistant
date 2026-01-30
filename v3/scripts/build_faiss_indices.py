#!/usr/bin/env python3
"""Build FAISS indices for fast RAG lookup.

This script pre-builds the following indices:
1. Reply-pairs index - for cross-conversation search (deduplicated unique pairs)
2. Global index - for all messages (optional, ~336K messages)

The reply-pairs index is SMART:
- Deduplicates by (their_text, your_reply) to avoid indexing "yeah" 1000 times
- Tracks last_message_id for incremental updates
- Only rebuilds when new messages are added

Usage:
    uv run python scripts/build_faiss_indices.py           # Build if needed
    uv run python scripts/build_faiss_indices.py --force   # Force rebuild
    uv run python scripts/build_faiss_indices.py --global  # Also build global index
"""

import argparse
import sys
import time
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indices for fast RAG lookup")
    parser.add_argument(
        "--global", "-g", dest="build_global", action="store_true",
        help="Also build global index (slower, ~336K vectors)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force rebuild even if index is up-to-date"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)

    from core.embeddings import get_embedding_store

    store = get_embedding_store()
    stats = store.get_stats()

    print(f"\nDatabase: {stats['total_messages']:,} messages in {stats['unique_conversations']} conversations")
    print(f"Database size: {stats['db_size_mb']:.1f} MB")

    # Check current state
    is_stale = store._is_reply_pairs_index_stale()
    print(f"\nReply-pairs index status: {'STALE (needs rebuild)' if is_stale else 'UP-TO-DATE'}")

    if not is_stale and not args.force:
        print("  Index is current. Use --force to rebuild anyway.")
        # Load and show stats
        result = store._get_or_build_reply_pairs_index()
        if result:
            _, metadata = result
            print(f"  Current index has {len(metadata):,} unique reply pairs")
    else:
        # Build reply-pairs index
        print("\nBuilding reply-pairs index...")
        print("  (Deduplicated: keeps one instance per unique their_text + your_reply)")
        start = time.time()

        result = store._get_or_build_reply_pairs_index(force_rebuild=args.force or is_stale)

        if result:
            index, metadata = result
            elapsed = time.time() - start
            print(f"  Done! {len(metadata):,} unique reply pairs indexed in {elapsed:.1f}s")
        else:
            print("  Failed to build reply-pairs index")

    # Optionally build global index
    if args.build_global:
        print("\nBuilding global index...")
        print("  (This indexes ALL ~336K messages - slower but enables global search)")
        start = time.time()

        result = store._get_or_build_global_faiss_index()

        if result:
            index, metadata = result
            elapsed = time.time() - start
            print(f"  Done! {len(metadata):,} messages indexed in {elapsed:.1f}s")
        else:
            print("  Failed to build global index")

    print("\n" + "=" * 60)
    print("Index Status:")
    print(f"  Reply-pairs index ready: {store.is_reply_pairs_index_ready()}")
    print(f"  Global index ready: {store.is_global_index_ready()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
