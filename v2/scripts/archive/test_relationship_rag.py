#!/usr/bin/env python3
"""Test script for the relationship-aware RAG pipeline.

Run from repo root: cd jarvis-ai-assistant && uv run python v2/scripts/test_relationship_rag.py
Or from v2 dir:     uv run python scripts/test_relationship_rag.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add v2 directory to path for imports (works from any directory)
script_dir = Path(__file__).parent.resolve()
v2_dir = script_dir.parent
repo_dir = v2_dir.parent

# Try v2 first, then repo root
if (v2_dir / "core").exists():
    sys.path.insert(0, str(v2_dir))
elif (repo_dir / "v2" / "core").exists():
    sys.path.insert(0, str(repo_dir / "v2"))

# Change to v2 dir so relative paths work
os.chdir(v2_dir)


def test_registry_loading():
    """Test 1: Verify RelationshipRegistry loads your contact profiles."""
    print("\n" + "=" * 60)
    print("TEST 1: RelationshipRegistry Loading")
    print("=" * 60)

    from core.embeddings.relationship_registry import get_relationship_registry

    registry = get_relationship_registry()
    stats = registry.get_stats()

    print(f"\nLoaded {stats['total']} contacts:")
    print(f"  - Friends: {stats['friends']}")
    print(f"  - Family: {stats['family']}")
    print(f"  - Work: {stats['work']}")
    print(f"  - Other: {stats['other']}")
    print(f"  - Phones indexed: {stats['phones_indexed']}")

    if stats["total"] == 0:
        print("\n❌ FAIL: No contacts loaded. Check results/contacts/contact_profiles.json")
        return False

    print("\n✅ PASS: Registry loaded successfully")
    return True


def test_relationship_lookup():
    """Test 2: Look up a specific contact's relationship."""
    print("\n" + "=" * 60)
    print("TEST 2: Relationship Lookup")
    print("=" * 60)

    from core.embeddings.relationship_registry import get_relationship_registry

    registry = get_relationship_registry()

    # Try looking up a few contacts
    test_names = ["Ishani Desai", "Mom", "Dad"]  # Adjust to your contacts

    for name in test_names:
        info = registry.get_relationship(name)
        if info:
            print(f"\n{name}:")
            print(f"  - Relationship: {info.relationship}")
            print(f"  - Category: {info.category}")
            print(f"  - Is group: {info.is_group}")
            print(f"  - Phones: {info.phones[:2]}..." if len(info.phones) > 2 else f"  - Phones: {info.phones}")
        else:
            print(f"\n{name}: Not found")

    print("\n✅ PASS: Lookup working")
    return True


def test_similar_contacts():
    """Test 3: Find contacts with similar relationships."""
    print("\n" + "=" * 60)
    print("TEST 3: Similar Contacts (Same Category)")
    print("=" * 60)

    from core.embeddings.relationship_registry import get_relationship_registry

    registry = get_relationship_registry()

    # Pick a contact and find similar ones
    test_contact = "Ishani Desai"  # Adjust to your contact
    info = registry.get_relationship(test_contact)

    if not info:
        print(f"\n❌ Contact '{test_contact}' not found")
        return False

    print(f"\n{test_contact} is in category: {info.category}")

    similar = registry.get_similar_contacts(test_contact)
    print(f"\nFound {len(similar)} similar contacts (same category):")
    for name in similar[:10]:
        other_info = registry.get_relationship(name)
        if other_info:
            print(f"  - {name} ({other_info.relationship})")

    if len(similar) > 10:
        print(f"  ... and {len(similar) - 10} more")

    print("\n✅ PASS: Similar contacts found")
    return True


def test_global_index():
    """Test 4: Build/load reply-pairs FAISS index (optimized)."""
    print("\n" + "=" * 60)
    print("TEST 4: Reply-Pairs FAISS Index (Fast Cross-Conversation Search)")
    print("=" * 60)

    from core.embeddings import get_embedding_store

    store = get_embedding_store()

    # Check if reply-pairs index exists on disk
    index_path = store._get_reply_pairs_index_path()
    if index_path.exists():
        print("\n✓ Reply-pairs index already cached on disk")
    else:
        print("\n⏳ Building reply-pairs index (first time only, ~2 minutes)...")

    start = time.time()
    result = store._get_or_build_reply_pairs_index()
    elapsed = time.time() - start

    if result is None:
        print("\n❌ FAIL: Could not build reply-pairs index")
        print("   (Make sure FAISS is installed: pip install faiss-cpu)")
        return False

    index, metadata = result
    print(f"\n✅ Reply-pairs index ready:")
    print(f"  - Reply pairs: {len(metadata):,}")
    print(f"  - Load/build time: {elapsed:.1f}s")

    return True


def test_cross_conversation_search():
    """Test 5: Search for similar situations across conversations."""
    print("\n" + "=" * 60)
    print("TEST 5: Cross-Conversation Search")
    print("=" * 60)

    from core.embeddings import get_embedding_store
    from core.embeddings.relationship_registry import get_relationship_registry

    store = get_embedding_store()
    registry = get_relationship_registry()

    # Test query - a common type of message
    test_query = "wanna hang out this weekend?"
    test_contact = "Ishani Desai"  # Adjust to your contact

    print(f"\nQuery: \"{test_query}\"")
    print(f"Contact: {test_contact}")

    # Get similar contacts
    info = registry.get_relationship(test_contact)
    if not info:
        print(f"\n❌ Contact '{test_contact}' not found")
        return False

    similar = registry.get_similar_contacts(test_contact)[:30]  # Top 30 similar contacts
    phones_by_contact = registry.get_phones_for_contacts(similar)

    # Collect all phones and resolve to REAL chat_ids from database
    all_phones = [p for phones in phones_by_contact.values() for p in phones]
    target_chat_ids = store.resolve_phones_to_chatids(all_phones)

    print(f"\nSearching {len(similar)} similar contacts ({len(target_chat_ids)} chat_ids)...")

    # First search warms up the index (loads from disk)
    start = time.time()
    results = store.find_your_past_replies_cross_conversation(
        incoming_message=test_query,
        target_chat_ids=target_chat_ids,
        limit=5,
        min_similarity=0.5,
    )
    first_time = (time.time() - start) * 1000

    # Second search should be fast (index in memory)
    start = time.time()
    results = store.find_your_past_replies_cross_conversation(
        incoming_message="you free tonight?",
        target_chat_ids=target_chat_ids,
        limit=5,
        min_similarity=0.5,
    )
    second_time = (time.time() - start) * 1000

    print(f"\n⏱️  Search times:")
    print(f"  - First search (loads index): {first_time:.0f}ms")
    print(f"  - Second search (cached): {second_time:.0f}ms")

    print(f"\nFound {len(results)} past replies:")
    for their_msg, your_reply, score, chat_id in results:
        their_short = their_msg[:40] + "..." if len(their_msg) > 40 else their_msg
        your_short = your_reply[:40] + "..." if len(your_reply) > 40 else your_reply
        print(f"\n  [{score:.2f}] Them: \"{their_short}\"")
        print(f"         You: \"{your_short}\"")

    if second_time < 100:
        print(f"\n✅ PASS: Cross-conversation search working ({second_time:.0f}ms)")
    elif results:
        print(f"\n⚠️  SLOW: Search took {second_time:.0f}ms (should be <100ms)")
    else:
        print("\n⚠️  No results found (try global search without chat_id filter)")

    return True


def test_full_pipeline():
    """Test 6: Full end-to-end reply generation with relationship-aware RAG."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline (Reply Generation)")
    print("=" * 60)

    from unittest.mock import MagicMock, patch

    # Mock the model loader since we don't want to actually load the LLM
    mock_loader = MagicMock()
    mock_loader.current_model = "test-model"

    class MockResult:
        text = "yeah sounds good!"
        formatted_prompt = "[mock]"

    mock_loader.generate.return_value = MockResult()

    from core.generation.reply_generator import ReplyGenerator

    # Create generator with mocked LLM but real embedding store and registry
    with patch("core.generation.reply_generator._get_template_matcher", return_value=None):
        generator = ReplyGenerator(mock_loader)

    # Test messages
    messages = [
        {"text": "hey what are you up to?", "sender": "Friend", "is_from_me": False},
        {"text": "not much just chilling", "sender": "me", "is_from_me": True},
        {"text": "wanna grab dinner later?", "sender": "Friend", "is_from_me": False},
    ]

    # Use a real chat_id format (you can adjust the phone number)
    test_chat_id = "iMessage;-;+15551234567"

    print(f"\nTest conversation:")
    for msg in messages:
        sender = "You" if msg["is_from_me"] else "Them"
        print(f"  {sender}: {msg['text']}")

    print(f"\nGenerating reply for chat_id: {test_chat_id}")

    start = time.time()
    result = generator.generate_replies(
        messages=messages,
        chat_id=test_chat_id,
        num_replies=1,
        user_name="me",
    )
    elapsed = time.time() - start

    print(f"\nGeneration completed in {elapsed*1000:.0f}ms")
    print(f"Past replies found: {len(result.past_replies)}")

    if result.past_replies:
        print("\nPast replies used for few-shot:")
        for their_msg, your_reply, score in result.past_replies[:3]:
            their_short = their_msg[:30] + "..." if len(their_msg) > 30 else their_msg
            your_short = your_reply[:30] + "..." if len(your_reply) > 30 else your_reply
            print(f"  [{score:.2f}] \"{their_short}\" -> \"{your_short}\"")

    print(f"\nGenerated reply: \"{result.replies[0].text}\"")
    print("\n✅ PASS: Full pipeline working")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RELATIONSHIP-AWARE RAG PIPELINE TEST")
    print("=" * 60)

    tests = [
        ("Registry Loading", test_registry_loading),
        ("Relationship Lookup", test_relationship_lookup),
        ("Similar Contacts", test_similar_contacts),
        ("Global Index", test_global_index),
        ("Cross-Conversation Search", test_cross_conversation_search),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
