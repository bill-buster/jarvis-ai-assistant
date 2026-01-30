#!/usr/bin/env python3
"""Test reply generation with REAL iMessage data.

This is the actual intended flow:
1. Load real conversations from your iMessage
2. RAG finds YOUR past replies to similar messages
3. Those become few-shot examples
4. Model generates replies that sound like YOU

Usage:
    uv run python scripts/test_real_replies.py
    uv run python scripts/test_real_replies.py --contact "Mom"
    uv run python scripts/test_real_replies.py --list  # List recent conversations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_conversations(limit: int = 20):
    """List recent conversations."""
    from core.imessage.reader import MessageReader

    print("Loading conversations from iMessage...")
    reader = MessageReader()
    convos = reader.get_conversations(limit=limit)

    print(f"\nFound {len(convos)} recent conversations:\n")
    print(f"{'#':<4} {'Contact':<25} {'Last Message':<40}")
    print("-" * 70)

    for i, convo in enumerate(convos, 1):
        name = convo.display_name or convo.chat_id[:25]
        last_msg = (convo.last_message_text or "")[:37]
        if len(convo.last_message_text or "") > 37:
            last_msg += "..."
        print(f"{i:<4} {name:<25} {last_msg:<40}")

    return convos


def test_with_conversation(chat_id: str, contact_name: str | None = None):
    """Test reply generation with a real conversation."""
    from core.generation import ReplyGenerator
    from core.imessage.reader import MessageReader
    from core.models.loader import ModelLoader

    print(f"\n{'='*70}")
    print("REAL REPLY GENERATION TEST")
    print(f"{'='*70}\n")

    # 1. Load real messages
    print("1. Loading messages from iMessage...")
    reader = MessageReader()
    messages = reader.get_messages(chat_id, limit=30)

    if not messages:
        print(f"   ERROR: No messages found for chat_id: {chat_id}")
        return

    print(f"   Loaded {len(messages)} messages")

    # Show recent messages
    print("\n   Recent messages:")
    for msg in messages[-5:]:
        sender = "You" if msg.is_from_me else (msg.sender_name or "Them")
        text = (msg.text or "[attachment]")[:50]
        print(f"   {'→' if msg.is_from_me else '←'} {sender}: {text}")

    # 2. Load model
    print("\n2. Loading model...")
    loader = ModelLoader()
    print(f"   Model: {loader.current_model}")

    # 3. Generate replies
    print("\n3. Generating replies...")
    generator = ReplyGenerator(loader)
    # Convert Message dataclasses to dicts for the generator
    messages_dicts = [msg.to_dict() for msg in messages]
    result = generator.generate_replies(
        messages=messages_dicts,
        chat_id=chat_id,
        contact_name=contact_name,
    )

    # 4. Show results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nGeneration time: {result.generation_time_ms:.0f}ms")
    print(f"Model used: {result.model_used}")

    # Show what we're replying to
    print(f"\nReplying to: \"{result.context.last_message}\"")
    print(f"Intent detected: {result.context.intent.value}")

    # Show past replies found (THE KEY PART!)
    print("\n--- YOUR PAST REPLIES FOUND (RAG) ---")
    if result.past_replies:
        print(f"Found {len(result.past_replies)} similar situations from your history:\n")
        for i, (their_msg, your_reply, score) in enumerate(result.past_replies, 1):
            print(f"  {i}. They said: \"{their_msg[:50]}\"")
            print(f"     You replied: \"{your_reply}\"")
            print(f"     Similarity: {score:.2f}")
            print()
    else:
        print("  No past replies found - using generic fallback examples")
        print("  (Have you run 'make index' to build the embedding index?)")

    # Show the actual prompt
    print("\n--- PROMPT SENT TO MODEL ---")
    print("-" * 40)
    prompt = result.prompt_used
    if len(prompt) > 800:
        print(prompt[:800] + "\n... [truncated]")
    else:
        print(prompt)
    print("-" * 40)

    # Show generated replies
    print("\n--- GENERATED REPLIES ---")
    for i, reply in enumerate(result.replies, 1):
        print(
            f"  {i}. \"{reply.text}\" "
            f"(confidence: {reply.confidence:.2f}, type: {reply.reply_type})"
        )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if result.past_replies:
        print("✓ RAG is working - found your past replies")
        print("✓ Model is learning from YOUR texting style")
    else:
        print("✗ RAG returned nothing - model using generic examples")
        print("  To fix: Run 'uv run python scripts/index_messages.py'")

    print("\nTo test with a different contact:")
    print("  uv run python scripts/test_real_replies.py --list")
    print("  uv run python scripts/test_real_replies.py --contact \"Contact Name\"")


def find_chat_by_contact(contact_name: str) -> tuple[str, str] | None:
    """Find chat_id by contact name."""
    from core.imessage.reader import MessageReader

    reader = MessageReader()
    convos = reader.get_conversations(limit=100)

    # Exact match first
    for convo in convos:
        name = convo.display_name or ""
        if name.lower() == contact_name.lower():
            return convo.chat_id, name

    # Partial match
    for convo in convos:
        name = convo.display_name or ""
        if contact_name.lower() in name.lower():
            return convo.chat_id, name

    return None


def main():
    parser = argparse.ArgumentParser(description="Test reply generation with real iMessage data")
    parser.add_argument("--list", "-l", action="store_true", help="List recent conversations")
    parser.add_argument("--contact", "-c", type=str, help="Contact name to test with")
    parser.add_argument("--chat-id", type=str, help="Specific chat_id to test with")
    parser.add_argument("--index", "-i", type=int, help="Use conversation at index from --list")
    args = parser.parse_args()

    if args.list:
        list_conversations()
        return

    if args.contact:
        result = find_chat_by_contact(args.contact)
        if not result:
            print(f"No conversation found for contact: {args.contact}")
            print("Use --list to see available conversations")
            return
        chat_id, contact_name = result
        print(f"Found: {contact_name} (chat_id: {chat_id[:30]}...)")
        test_with_conversation(chat_id, contact_name)
        return

    if args.chat_id:
        test_with_conversation(args.chat_id)
        return

    if args.index is not None:
        convos = list_conversations()
        if args.index < 1 or args.index > len(convos):
            print(f"\nInvalid index. Choose 1-{len(convos)}")
            return
        convo = convos[args.index - 1]
        chat_id = convo.chat_id
        contact_name = convo.display_name
        test_with_conversation(chat_id, contact_name)
        return

    # Default: list conversations and prompt
    convos = list_conversations()
    print("\nTo test reply generation:")
    print("  uv run python scripts/test_real_replies.py --index 1")
    print("  uv run python scripts/test_real_replies.py --contact \"Mom\"")


if __name__ == "__main__":
    main()
