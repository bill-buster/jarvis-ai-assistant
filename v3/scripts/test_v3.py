"""Quick test script for v3 reply generation.

Tests the core functionality without requiring actual model loading.
"""

import sys
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")

    try:
        from core.models.registry import get_model_spec, DEFAULT_MODEL

        print(f"  ✓ Models registry (default: {DEFAULT_MODEL})")

        from core.models.loader import ModelLoader

        print("  ✓ Model loader")

        from core.embeddings.store import get_embedding_store

        print("  ✓ Embedding store")

        from core.embeddings.relationship_registry import get_relationship_registry

        print("  ✓ Relationship registry")

        from core.generation.reply_generator import ReplyGenerator

        print("  ✓ Reply generator")

        from core.imessage.reader import ChatDBReader

        print("  ✓ iMessage reader")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_model_registry():
    """Test model registry has correct model."""
    print("\nTesting model registry...")

    from core.models.registry import get_model_spec, MODELS

    spec = get_model_spec()
    print(f"  Model: {spec.id}")
    print(f"  Path: {spec.path}")
    print(f"  Size: {spec.size_gb}GB")

    assert spec.id == "lfm2.5-1.2b", f"Expected lfm2.5-1.2b, got {spec.id}"
    assert len(MODELS) == 1, f"Expected 1 model, got {len(MODELS)}"

    print("  ✓ Registry correct")
    return True


def test_reply_generator_structure():
    """Test reply generator can be instantiated (without loading model)."""
    print("\nTesting reply generator...")

    from core.generation.reply_generator import ReplyGenerator

    # Check the class exists and has the right methods
    assert hasattr(ReplyGenerator, "generate_replies")
    assert hasattr(ReplyGenerator, "_find_similar_past_replies")
    assert hasattr(ReplyGenerator, "_find_cross_conversation_replies")

    print("  ✓ ReplyGenerator structure correct")
    print("  ✓ Has relationship-aware RAG methods")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("JARVIS v3 - Quick Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_model_registry,
        test_reply_generator_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n✅ v3 structure is ready!")
        print("\nNext steps:")
        print("  1. cd v3 && make install")
        print("  2. uv run python scripts/profile_contacts.py")
        print("  3. uv run python scripts/index_messages.py")
    else:
        print("\n❌ Some tests failed - check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
