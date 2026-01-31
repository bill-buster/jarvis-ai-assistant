#!/usr/bin/env python3
"""Build embedding-based relationship profiles for all contacts.

Analyzes message history using embeddings to extract:
- Topic clusters (semantically grouped conversation themes)
- Communication dynamics (style similarity, initiation patterns)
- Response patterns (semantic shift in conversations)

Usage:
    python -m scripts.build_embedding_profiles              # Build for all contacts
    python -m scripts.build_embedding_profiles --limit 10   # First 10 contacts only
    python -m scripts.build_embedding_profiles --contact "Sarah"  # Specific contact
    python -m scripts.build_embedding_profiles --verbose    # Show detailed progress
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db import get_db
from jarvis.embedding_adapter import get_embedder
from jarvis.embedding_profile import (
    build_embedding_profile,
    build_profiles_for_all_contacts,
    generate_embedding_style_guide,
    save_embedding_profile,
)

logger = logging.getLogger(__name__)


def build_single_contact_profile(
    contact_name: str,
    verbose: bool = False,
) -> dict:
    """Build embedding profile for a single contact.

    Args:
        contact_name: Name to search for.
        verbose: Print detailed info.

    Returns:
        Profile building result.
    """
    db = get_db()
    db.init_schema()

    # Find contact
    contact = db.get_contact_by_name(contact_name)
    if not contact:
        return {"success": False, "error": f"Contact '{contact_name}' not found"}

    if not contact.chat_id:
        return {"success": False, "error": f"Contact '{contact_name}' has no chat_id"}

    print(f"Building profile for {contact.display_name}...")

    # Get embedder
    embedder = get_embedder()

    # Try to get messages from iMessage first, then fall back to pairs
    messages = []

    try:
        from integrations.imessage.reader import ChatDBReader

        reader = ChatDBReader()
        messages = reader.get_messages(contact.chat_id, limit=5000)
        if messages:
            print(f"Found {len(messages)} messages from iMessage")
    except Exception as e:
        logger.debug("iMessage reader failed: %s", e)

    # Fallback to pairs from database if iMessage didn't work
    if len(messages) < 30:
        pairs = db.get_pairs(contact_id=contact.id, limit=5000)
        from dataclasses import dataclass
        from datetime import datetime

        @dataclass
        class MockMessage:
            text: str
            is_from_me: bool
            date: datetime

        messages = []
        for p in pairs:
            if p.trigger_timestamp and p.response_timestamp:
                messages.append(MockMessage(p.trigger_text, False, p.trigger_timestamp))
                messages.append(MockMessage(p.response_text, True, p.response_timestamp))
        print(f"Using {len(messages)} messages from pairs database")

    if len(messages) < 30:
        return {
            "success": False,
            "error": f"Not enough messages ({len(messages)}). Need at least 30.",
        }

    # Build profile
    profile = build_embedding_profile(
        contact_id=str(contact.id),
        messages=messages,
        embedder=embedder,
        contact_name=contact.display_name,
    )

    # Save profile
    if save_embedding_profile(profile):
        result = {
            "success": True,
            "contact": contact.display_name,
            "messages_analyzed": profile.message_count,
            "topic_clusters": len(profile.topic_clusters),
        }

        if verbose and profile.dynamics:
            result["dynamics"] = {
                "style_similarity": profile.dynamics.style_similarity,
                "initiation_ratio": profile.dynamics.initiation_ratio,
                "topic_diversity": profile.dynamics.topic_diversity,
                "response_semantic_shift": profile.dynamics.response_semantic_shift,
            }
            result["style_guide"] = generate_embedding_style_guide(profile)

        return result

    return {"success": False, "error": "Failed to save profile"}


def main():
    parser = argparse.ArgumentParser(description="Build embedding-based relationship profiles")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of contacts to process",
    )
    parser.add_argument(
        "--contact",
        "-c",
        type=str,
        default=None,
        help="Build profile for specific contact by name",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=30,
        help="Minimum messages required for a profile (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Single contact mode
    if args.contact:
        result = build_single_contact_profile(args.contact, args.verbose)
        if result["success"]:
            print(f"\nProfile created for {result['contact']}:")
            print(f"  Messages analyzed: {result['messages_analyzed']}")
            print(f"  Topic clusters: {result['topic_clusters']}")
            if "dynamics" in result:
                d = result["dynamics"]
                print("\n  Communication dynamics:")
                print(f"    Style similarity: {d['style_similarity']:.1%}")
                print(f"    Initiation ratio: {d['initiation_ratio']:.1%}")
                print(f"    Topic diversity: {d['topic_diversity']:.2f}")
                print(f"    Response shift: {d['response_semantic_shift']:.2f}")
            if "style_guide" in result:
                print(f"\n  Style guide: {result['style_guide']}")
        else:
            print(f"\nFailed: {result['error']}")
            sys.exit(1)
        return

    # Full processing mode
    print("Building embedding profiles for all contacts...")

    db = get_db()
    db.init_schema()

    embedder = get_embedder()

    # Try to get iMessage reader
    try:
        from integrations.imessage.reader import ChatDBReader

        reader = ChatDBReader()
        print("Using iMessage database for message history")
    except Exception as e:
        logger.warning("iMessage reader not available: %s", e)
        reader = None

    stats = build_profiles_for_all_contacts(
        db=db,
        embedder=embedder,
        imessage_reader=reader,
        min_messages=args.min_messages,
        limit=args.limit,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING PROFILE BUILD SUMMARY")
    print("=" * 60)
    print(f"Contacts processed: {stats['contacts_processed']}")
    print(f"Profiles created:   {stats['profiles_created']}")
    print(f"Profiles skipped:   {stats['profiles_skipped']}")
    print(f"Messages analyzed:  {stats['total_messages_analyzed']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    print("\nProfiles saved to: ~/.jarvis/embedding_profiles/")


if __name__ == "__main__":
    main()
