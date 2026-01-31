#!/usr/bin/env python3
"""Test simple reply with real iMessage conversations - last 20 messages.

Just pull recent messages, show them to LLM, get a reply.

Run: uv run python scripts/experiments/test_simple_real.py
"""

from jarvis.simple_reply import generate_reply


def get_real_conversations(limit: int = 5):
    """Get real conversations from iMessage."""
    from integrations.imessage.reader import ChatDBReader

    reader = ChatDBReader()
    conversations = reader.get_conversations(limit=50)

    # Filter to ones with recent activity and actual back-and-forth
    good_convos = []
    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=20)
        if not messages:
            continue

        # Need at least 5 messages with some from each side
        my_msgs = sum(1 for m in messages if m.is_from_me and m.text)
        their_msgs = sum(1 for m in messages if not m.is_from_me and m.text)

        if my_msgs >= 2 and their_msgs >= 2:
            good_convos.append((conv, messages))

        if len(good_convos) >= limit:
            break

    return good_convos


def test_real_conversations():
    """Test with real iMessage data."""
    print("Loading real conversations from iMessage...")

    try:
        convos = get_real_conversations(limit=5)
    except Exception as e:
        print(f"Error loading iMessage: {e}")
        return

    if not convos:
        print("No suitable conversations found")
        return

    print(f"\nFound {len(convos)} conversations with good back-and-forth\n")
    print("=" * 70)

    for conv, messages in convos:
        name = conv.display_name or conv.chat_id[:30]
        print(f"\n{'=' * 70}")
        print(f"CONVERSATION: {name}")
        print("=" * 70)

        # Build conversation list (oldest first)
        conversation = []
        for msg in reversed(messages):
            speaker = "me" if msg.is_from_me else "them"
            if msg.text:
                conversation.append((speaker, msg.text))

        # Show last 10 messages
        print("\nLast 10 messages:")
        for speaker, text in conversation[-10:]:
            prefix = "  [Me]:" if speaker == "me" else "  [Them]:"
            # Truncate long messages
            display = text[:60] + "..." if len(text) > 60 else text
            # Handle newlines in display
            display = display.replace("\n", " | ")
            print(f"{prefix} {display}")

        # Only suggest if last message is from them
        if conversation[-1][0] == "them":
            print("\n→ Generating reply...")
            result = generate_reply(conversation)

            print(f"\n  SUGGESTED: {result['response'] or result['question']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Raw: {result['raw'][:100]}...")
        else:
            print("\n  (Last message is from me, skipping)")


def interactive_mode():
    """Interactive mode - pick a conversation and iterate."""
    from integrations.imessage.reader import ChatDBReader

    reader = ChatDBReader()
    conversations = reader.get_conversations(limit=20)

    print("\nAvailable conversations:")
    for i, conv in enumerate(conversations[:15]):
        name = conv.display_name or conv.chat_id[:30]
        print(f"  {i + 1}. {name}")

    try:
        choice = input("\nPick a number (or 'q' to quit): ")
        if choice.lower() == "q":
            return

        idx = int(choice) - 1
        conv = conversations[idx]
    except (ValueError, IndexError):
        print("Invalid choice")
        return

    # Get messages
    messages = reader.get_messages(conv.chat_id, limit=20)
    conversation = []
    for msg in reversed(messages):
        speaker = "me" if msg.is_from_me else "them"
        if msg.text:
            conversation.append((speaker, msg.text))

    print(f"\n{'=' * 70}")
    print(f"CONVERSATION: {conv.display_name or conv.chat_id}")
    print("=" * 70)

    for speaker, text in conversation:
        prefix = "[Me]:" if speaker == "me" else "[Them]:"
        print(f"{prefix} {text}")

    if conversation[-1][0] == "them":
        print("\n→ Generating reply...")
        result = generate_reply(conversation)
        print(f"\nSUGGESTED: {result['response'] or result['question']}")
        print(f"Raw output: {result['raw']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        test_real_conversations()
