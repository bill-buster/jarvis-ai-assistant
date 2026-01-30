"""Validate your labeled contact relationships.

Run this to check if your contact_profiles.json is properly set up.
"""

import json
import sys
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_profiles():
    """Check if contact profiles exist and are valid."""
    profiles_path = Path("results/contacts/contact_profiles.json")

    if not profiles_path.exists():
        print("❌ No contact_profiles.json found!")
        print(f"   Expected at: {profiles_path.absolute()}")
        print("\nRun: python scripts/profile_contacts.py")
        return False

    with open(profiles_path) as f:
        profiles = json.load(f)

    print(f"✅ Found {len(profiles)} labeled contacts\n")

    # Count by category
    categories = {}
    relationships = {}

    for name, data in profiles.items():
        cat = data.get("category", "unknown")
        rel = data.get("relationship", "unknown")

        categories[cat] = categories.get(cat, 0) + 1
        relationships[rel] = relationships.get(rel, 0) + 1

    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count}")

    print("\nTop Relationships:")
    for rel, count in sorted(relationships.items(), key=lambda x: -x[1])[:10]:
        print(f"  • {rel}: {count}")

    # Test loading into registry
    print("\n🔄 Testing relationship registry...")
    try:
        from core.embeddings.relationship_registry import get_relationship_registry

        registry = get_relationship_registry()
        # Load the profiles first
        registry._load()
        print(f"✅ Registry loaded with {len(registry._contacts)} contacts")

        # Test lookup
        test_name = list(profiles.keys())[0]
        rel = registry.get_relationship(test_name)
        if rel:
            print(f"✅ Lookup test: '{test_name}' → '{rel.relationship}' ({rel.category})")
        else:
            print(f"⚠️  Lookup test: '{test_name}' not found (might need phone number)")

        # Test category search
        if "friend" in categories:
            friends = registry.get_contacts_by_category("friend")
            print(f"✅ Found {len(friends)} friends in registry")

        return True

    except Exception as e:
        print(f"❌ Registry error: {e}")
        return False


def test_context_analysis():
    """Test context analyzer with sample messages."""
    print("\n" + "=" * 50)
    print("Testing Context Analysis")
    print("=" * 50)

    from core.generation.context_analyzer import ContextAnalyzer, ConversationContext

    analyzer = ContextAnalyzer()

    # Test cases with different intents
    test_cases = [
        {
            "name": "Question",
            "messages": [
                {"text": "Hey what time works for dinner?", "sender": "Friend", "is_from_me": False}
            ],
        },
        {
            "name": "Invitation",
            "messages": [
                {"text": "Want to grab drinks tonight?", "sender": "Friend", "is_from_me": False}
            ],
        },
        {
            "name": "Update",
            "messages": [
                {"text": "Just got to the restaurant", "sender": "Friend", "is_from_me": False}
            ],
        },
        {
            "name": "Multi-turn",
            "messages": [
                {"text": "Are you free tomorrow?", "sender": "Friend", "is_from_me": False},
                {"text": "Yeah should be", "sender": "me", "is_from_me": True},
                {"text": "Want to grab lunch?", "sender": "Friend", "is_from_me": False},
            ],
        },
    ]

    for case in test_cases:
        print(f"\n📝 Test: {case['name']}")
        print(f"   Last message: '{case['messages'][-1]['text']}'")

        try:
            context = analyzer.analyze(case["messages"])
            print(f"   ✓ Intent: {context.intent.value}")
            print(f"   ✓ Summary: {context.summary}")

            if hasattr(context, "relationship") and context.relationship:
                print(f"   ✓ Relationship: {context.relationship}")

        except Exception as e:
            print(f"   ✗ Error: {e}")


def test_style_analysis():
    """Test style analyzer."""
    print("\n" + "=" * 50)
    print("Testing Style Analysis")
    print("=" * 50)

    from core.generation.style_analyzer import StyleAnalyzer, UserStyle

    analyzer = StyleAnalyzer()

    # Sample your messages (simulated)
    your_messages = [
        "yeah sounds good",
        "sure!",
        "lol nice",
        "haha exactly",
        "ok cool",
        "nah i'm good",
        "maybe tomorrow?",
        "👍",
        "perfect 👌",
    ]

    print(f"\nAnalyzing {len(your_messages)} sample messages...")

    try:
        # Create sample message dicts
        sample_messages = [
            {"text": msg, "sender": "me", "is_from_me": True} for msg in your_messages
        ]

        style = analyzer.analyze(sample_messages)

        print(f"\n✅ Style detected:")
        print(f"   • Avg words: {style.avg_word_count:.1f}")
        print(f"   • Avg chars: {style.avg_char_count:.0f}")
        print(f"   • Uses emojis: {style.uses_emoji} ({style.emoji_frequency:.0%})")
        print(f"   • Punctuation: {style.punctuation_style}")
        print(f"   • Formality: {style.formality_score:.1f}")
        print(f"   • Enthusiasm: {style.enthusiasm_level}")

        if hasattr(style, "common_phrases") and style.common_phrases:
            print(f"   • Common phrases: {', '.join(style.common_phrases[:3])}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("JARVIS v3 - Validation Suite")
    print("=" * 50)

    # Check profiles
    if check_profiles():
        print("\n✅ Profiles validated successfully!")
    else:
        print("\n⚠️  Please set up profiles first")
        sys.exit(1)

    # Test analysis
    test_context_analysis()
    test_style_analysis()

    print("\n" + "=" * 50)
    print("Next: Run evaluation with real conversations")
    print("=" * 50)
    print("\nRun: python scripts/evaluate_replies.py")
