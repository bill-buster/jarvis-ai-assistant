#!/usr/bin/env python3
"""Manual test script for reply generation.

Run with: python -m v2.scripts.test_generation
"""

from __future__ import annotations

import sys


def test_imessage_reader():
    """Test iMessage reader."""
    print("=" * 60)
    print("Testing iMessage Reader")
    print("=" * 60)

    from core.imessage import MessageReader

    reader = MessageReader()

    if not reader.check_access():
        print("ERROR: Cannot access iMessage database")
        print("Grant Full Disk Access in System Settings")
        return False

    print("iMessage database accessible")

    # Get conversations
    conversations = reader.get_conversations(limit=5)
    print(f"\nFound {len(conversations)} recent conversations:")

    for i, conv in enumerate(conversations, 1):
        name = conv.display_name or ", ".join(conv.participants[:2]) or "Unknown"
        print(f"  {i}. {name}")
        print(f"     Last: {conv.last_message_text[:50] if conv.last_message_text else 'N/A'}...")

    if conversations:
        # Get messages from first conversation
        chat_id = conversations[0].chat_id
        messages = reader.get_messages(chat_id, limit=10)
        print(f"\nLast 10 messages from first conversation:")
        for msg in reversed(messages):
            sender = "You" if msg.is_from_me else msg.sender[:15]
            text = (msg.text[:40] + "...") if msg.text and len(msg.text) > 40 else (msg.text or "")
            print(f"  {sender}: {text}")

    reader.close()
    return True


def test_style_analyzer():
    """Test style analyzer."""
    print("\n" + "=" * 60)
    print("Testing Style Analyzer")
    print("=" * 60)

    from core.generation import StyleAnalyzer

    analyzer = StyleAnalyzer()

    # Simulate user messages
    test_messages = [
        {"text": "hey! yeah that works for me"},
        {"text": "lol same"},
        {"text": "sounds good!"},
        {"text": "omg yes please"},
        {"text": "ok cool"},
        {"text": "haha definitely"},
        {"text": "thanks!"},
        {"text": "np!"},
    ]

    style = analyzer.analyze(test_messages)

    print(f"Avg word count: {style.avg_word_count:.1f}")
    print(f"Uses emoji: {style.uses_emoji}")
    print(f"Capitalization: {style.capitalization}")
    print(f"Uses abbreviations: {style.uses_abbreviations}")
    print(f"Punctuation style: {style.punctuation_style}")
    print(f"Enthusiasm: {style.enthusiasm_level}")

    print("\nGenerated style instructions:")
    print(analyzer.to_prompt_instructions(style))

    return True


def test_context_analyzer():
    """Test context analyzer."""
    print("\n" + "=" * 60)
    print("Testing Context Analyzer")
    print("=" * 60)

    from core.generation import ContextAnalyzer

    analyzer = ContextAnalyzer()

    # Simulate conversation
    test_messages = [
        {"text": "Hey! How's it going?", "sender": "Sarah", "is_from_me": False},
        {"text": "pretty good! just got back from the gym", "sender": "me", "is_from_me": True},
        {"text": "Nice! Want to grab dinner tonight?", "sender": "Sarah", "is_from_me": False},
    ]

    context = analyzer.analyze(test_messages)

    print(f"Last message: {context.last_message}")
    print(f"Intent: {context.intent.value}")
    print(f"Relationship: {context.relationship.value}")
    print(f"Topic: {context.topic}")
    print(f"Mood: {context.mood}")
    print(f"Needs response: {context.needs_response}")

    return True


def test_full_generation():
    """Test full generation pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Generation Pipeline")
    print("=" * 60)

    # Check if we can load model
    try:
        from core.models import get_model_loader

        print("Loading model (this may take a minute)...")
        loader = get_model_loader()

        # Get real conversation
        from core.imessage import MessageReader

        reader = MessageReader()
        if not reader.check_access():
            print("Cannot access iMessage, using mock data")
            messages = [
                {"text": "Hey!", "sender": "Friend", "is_from_me": False},
                {"text": "hey whats up", "sender": "me", "is_from_me": True},
                {"text": "Want to grab coffee tomorrow?", "sender": "Friend", "is_from_me": False},
            ]
        else:
            conversations = reader.get_conversations(limit=1)
            if conversations:
                chat_id = conversations[0].chat_id
                raw_messages = reader.get_messages(chat_id, limit=15)
                messages = [
                    {
                        "text": m.text,
                        "sender": m.sender,
                        "is_from_me": m.is_from_me,
                    }
                    for m in reversed(raw_messages)
                ]
            else:
                messages = []
            reader.close()

        if not messages:
            print("No messages found")
            return False

        print(f"\nConversation ({len(messages)} messages):")
        for msg in messages[-5:]:
            sender = "You" if msg.get("is_from_me") else msg.get("sender", "Them")[:10]
            text = msg.get("text", "")[:50]
            print(f"  {sender}: {text}")

        # Generate replies
        from core.generation import ReplyGenerator

        generator = ReplyGenerator(loader)

        print("\nGenerating replies...")
        result = generator.generate_replies(messages)

        print(f"\nGenerated in {result.generation_time_ms:.0f}ms using {result.model_used}")
        print(f"Context: {result.context.intent.value}, {result.context.topic}")
        print("\nSuggested replies:")
        for i, reply in enumerate(result.replies, 1):
            print(f"  {i}. {reply.text}")
            print(f"     (type: {reply.reply_type}, confidence: {reply.confidence:.2f})")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("JARVIS v2 - Manual Testing")
    print("=" * 60)

    results = []

    # Test components
    results.append(("iMessage Reader", test_imessage_reader()))
    results.append(("Style Analyzer", test_style_analyzer()))
    results.append(("Context Analyzer", test_context_analyzer()))

    # Ask before full generation (loads model)
    print("\n" + "=" * 60)
    response = input("Run full generation test? (loads LLM, ~30s) [y/N]: ")
    if response.lower() == "y":
        results.append(("Full Generation", test_full_generation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
