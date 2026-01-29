"""Minimal fast tests for JARVIS v3.

Tests run in <5 seconds without loading real models.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports_work():
    """Test that all core modules can be imported."""
    # Core generation modules
    from core.generation import ReplyGenerator
    from core.generation.context_analyzer import ContextAnalyzer, MessageIntent
    from core.generation.style_analyzer import StyleAnalyzer, UserStyle
    from core.generation.prompts import build_reply_prompt, build_conversation_prompt

    # Core models
    from core.models.registry import get_model_spec, MODELS, DEFAULT_MODEL

    # Core embeddings
    from core.embeddings.relationship_registry import RelationshipRegistry

    # API routes
    from api.routes.conversations import list_conversations, get_messages
    from api.routes.generate import generate_replies

    # Assert key classes exist
    assert ReplyGenerator is not None
    assert ContextAnalyzer is not None
    assert StyleAnalyzer is not None
    assert MessageIntent is not None
    assert UserStyle is not None


def test_model_registry_has_correct_model():
    """Test that model registry has LFM2.5-1.2B as default."""
    from core.models.registry import get_model_spec, MODELS, DEFAULT_MODEL

    # Check default model
    assert DEFAULT_MODEL == "lfm2.5-1.2b"

    # Check model exists
    assert "lfm2.5-1.2b" in MODELS

    # Check model spec
    spec = get_model_spec()
    assert spec.id == "lfm2.5-1.2b"
    assert "LFM2.5" in spec.display_name
    assert spec.size_gb < 1.0  # Should be lightweight


def test_can_create_reply_generator_mocked():
    """Test that ReplyGenerator can be created with mocked model loader."""
    from unittest.mock import MagicMock

    from core.generation import ReplyGenerator

    # Create mock model loader
    mock_loader = MagicMock()
    mock_loader.is_loaded = True
    mock_loader.current_model = "lfm2.5-1.2b"

    # Create generator
    generator = ReplyGenerator(mock_loader)

    # Assert generator was created
    assert generator is not None
    assert generator.model_loader == mock_loader


def test_context_analyzer_detects_intent():
    """Test that ContextAnalyzer can detect message intent."""
    from core.generation.context_analyzer import ContextAnalyzer, MessageIntent

    analyzer = ContextAnalyzer()

    # Test greeting detection
    messages = [{"text": "Hey! How are you?", "is_from_me": False, "sender": "Alice"}]
    result = analyzer.analyze(messages)
    assert result.intent == MessageIntent.GREETING

    # Test question detection
    messages = [{"text": "Do you want to grab dinner?", "is_from_me": False, "sender": "Bob"}]
    result = analyzer.analyze(messages)
    assert result.intent == MessageIntent.YES_NO_QUESTION


def test_style_analyzer_analyzes_messages():
    """Test that StyleAnalyzer can analyze message style."""
    from core.generation.style_analyzer import StyleAnalyzer, UserStyle

    analyzer = StyleAnalyzer()

    # Test with casual messages
    messages = [
        {"text": "lol that's hilarious", "is_from_me": True},
        {"text": "haha ikr", "is_from_me": True},
        {"text": "wanna hang tmrw?", "is_from_me": True},
    ]

    style = analyzer.analyze(messages)

    # Should return a UserStyle object
    assert isinstance(style, UserStyle)
    assert style.avg_word_count > 0
    assert style.uses_abbreviations is True  # Should detect "lol", "tmrw"


def test_prompt_building():
    """Test that prompts can be built."""
    from core.generation.prompts import build_conversation_prompt, build_reply_prompt

    # Test conversation prompt
    messages = [
        {"text": "Hey!", "is_from_me": False, "sender": "Alice"},
        {"text": "Hi there!", "is_from_me": True},
    ]

    prompt = build_conversation_prompt(messages)
    assert "Hey!" in prompt
    assert "Hi there!" in prompt

    # Test reply prompt
    prompt = build_reply_prompt(
        messages=messages,
        last_message="Hey!",
        last_sender="Alice",
        style_instructions="casual and brief",
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_relationship_registry_basic():
    """Test basic RelationshipRegistry functionality."""
    import json
    import tempfile
    from pathlib import Path

    from core.embeddings.relationship_registry import RelationshipRegistry

    # Create temp profiles file
    profiles = {
        "Alice": {
            "relationship": "friend",
            "category": "friend",
            "is_group": False,
            "phones": ["+1234567890"],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profiles, f)
        temp_path = Path(f.name)

    try:
        # Create registry
        registry = RelationshipRegistry(temp_path)

        # Test lookup
        info = registry.get_relationship("Alice")
        assert info is not None
        assert info.contact_name == "Alice"
        assert info.category == "friend"

        # Test phone lookup
        info = registry.get_relationship("+1234567890")
        assert info is not None
        assert info.contact_name == "Alice"
    finally:
        # Cleanup
        temp_path.unlink()
