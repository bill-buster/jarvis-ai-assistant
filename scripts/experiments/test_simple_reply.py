#!/usr/bin/env python3
"""Test the simple reply generator with examples from the eval data.

Run: uv run python scripts/experiments/test_simple_reply.py
"""

import json
from pathlib import Path


def test_with_eval_samples():
    """Test simple reply against samples from the improved eval."""
    from jarvis.simple_reply import generate_reply

    # Load some samples from the eval
    eval_file = Path(
        "results/experiment_20260130_114216/improved_eval/results_20260130_115414.json"
    )

    if not eval_file.exists():
        print("Eval results not found, using hardcoded examples")
        samples = None
    else:
        with open(eval_file) as f:
            all_samples = json.load(f)
        # Pick 10 random samples
        import random

        random.seed(42)
        samples = random.sample(all_samples, 10)

    print("=" * 70)
    print("SIMPLE REPLY GENERATOR TEST")
    print("=" * 70)

    if samples:
        for i, sample in enumerate(samples):
            trigger = sample["trigger"]
            actual = sample["actual_response"]
            old_llm = sample["llm_response"]

            # Build minimal conversation (just the trigger as "them")
            # In real usage, we'd have more context
            conversation = [("them", trigger)]

            print(f"\n--- Sample {i + 1} ---")
            print(f"TRIGGER: {trigger[:80]}{'...' if len(trigger) > 80 else ''}")
            print(f"ACTUAL:  {actual[:60]}{'...' if len(actual) > 60 else ''}")
            print(f"OLD LLM: {old_llm[:60]}{'...' if len(old_llm) > 60 else ''}")

            result = generate_reply(conversation)

            if result["response"]:
                print(f"SIMPLE:  {result['response']}")
            else:
                print(f"ASKS:    {result['question']}")
            print(f"Confidence: {result['confidence']}")
    else:
        # Hardcoded test cases - MUST include user's own messages to show style
        test_cases = [
            {
                "conversation": [
                    ("me", "yo"),
                    ("them", "sup"),
                    ("me", "nm wbu"),
                    ("them", "you tryna pull up to hets"),
                    ("them", "bouta head there rn"),
                    ("them", "just grabbing dillas and maybe watch movie"),
                ],
                "actual": "Nah im studying\nI watched football all day",
                "note": "User style: lowercase, no punctuation, terse",
            },
            {
                "conversation": [
                    ("me", "yo"),
                    ("them", "sup"),
                    ("me", "not much u"),
                    ("them", "same"),
                    ("me", "down for food later"),
                    ("them", "yea for sure"),
                    ("them", "what time works"),
                ],
                "actual": "like 2?",
                "note": "User style: very short, question without question mark",
            },
            {
                "conversation": [
                    ("them", "just got back"),
                    ("me", "nice how was it"),
                    ("them", "pretty good"),
                    ("them", "weather was nice"),
                    ("me", "thats sick"),
                    ("them", "No worries lol"),
                    ("them", "I ubered"),
                ],
                "actual": "u get home safe?",
                "note": "User style: caring but terse",
            },
            {
                "conversation": [
                    ("me", "that game was crazy"),
                    ("them", "fr"),
                    ("them", "that ending tho"),
                    ("me", "I know right"),
                    ("me", "clutch af"),
                    ("them", "we should watch the next one together"),
                ],
                "actual": "bet lmk when",
                "note": "User style: slang (bet, lmk), no caps",
            },
            {
                "conversation": [
                    ("them", "yo im outside"),
                    ("me", "k coming"),
                    ("them", "can you meet me down or u want me to come up"),
                ],
                "actual": "i can come down\nare u here?",
                "note": "User style: lowercase, uses newlines",
            },
            {
                "conversation": [
                    ("me", "hows the new job"),
                    ("them", "its good"),
                    ("them", "busy tho"),
                    ("me", "ya i bet"),
                    ("them", "wanna grab dinner this week"),
                ],
                "actual": "yea im down",
                "note": "User style: 'yea' not 'yeah', casual",
            },
        ]

        for i, case in enumerate(test_cases):
            print(f"\n--- Test {i + 1} ---")
            conv = case["conversation"]
            actual = case["actual"]

            # Show conversation
            print("CONVERSATION:")
            for speaker, text in conv:
                prefix = "  Me:" if speaker == "me" else "  Them:"
                print(f"{prefix} {text}")

            print(f"\nACTUAL RESPONSE: {actual}")

            result = generate_reply(conv)

            if result["response"]:
                print(f"SIMPLE REPLY:   {result['response']}")
            else:
                print(f"NEEDS INFO:     {result['question']}")
            print(f"Confidence: {result['confidence']}")


def test_with_real_imessage():
    """Test with actual iMessage data if available."""
    try:
        from integrations.imessage.reader import ChatDBReader

        reader = ChatDBReader()
        conversations = reader.get_conversations(limit=5)

        if not conversations:
            print("\nNo iMessage conversations found")
            return

        print("\n" + "=" * 70)
        print("TESTING WITH REAL IMESSAGE DATA")
        print("=" * 70)

        for conv in conversations[:3]:
            messages = reader.get_messages(conv.chat_id, limit=8)
            if not messages or len(messages) < 3:
                continue

            # Build conversation
            conversation = []
            for msg in reversed(messages):
                speaker = "me" if msg.is_from_me else "them"
                if msg.text:
                    conversation.append((speaker, msg.text))

            if len(conversation) < 2:
                continue

            print(f"\n--- Chat: {conv.display_name or conv.chat_id[:20]} ---")
            print("Recent messages:")
            for speaker, text in conversation[-5:]:
                prefix = "  Me:" if speaker == "me" else "  Them:"
                print(f"{prefix} {text[:50]}{'...' if len(text) > 50 else ''}")

            # Only generate if last message is from them
            if conversation[-1][0] == "them":
                from jarvis.simple_reply import generate_reply

                result = generate_reply(conversation)

                print("\nSuggested reply:")
                if result["response"]:
                    print(f"  {result['response']}")
                else:
                    print(f"  [Needs info]: {result['question']}")

    except Exception as e:
        print(f"\nCouldn't test with iMessage: {e}")


if __name__ == "__main__":
    test_with_eval_samples()
    test_with_real_imessage()
