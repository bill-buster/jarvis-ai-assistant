"""Essential tests for RelationshipRegistry.

Minimal test coverage for core functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.embeddings.relationship_registry import RelationshipRegistry

# Sample test data
SAMPLE_PROFILES = {
    "Alice": {
        "relationship": "friend",
        "category": "friend",
        "is_group": False,
        "phones": ["+15551234567"],
    },
    "Bob": {
        "relationship": "close friend",
        "category": "friend",
        "is_group": False,
        "phones": ["+15552345678"],
    },
    "Mom": {
        "relationship": "mother",
        "category": "family",
        "is_group": False,
        "phones": ["+15553456789"],
    },
}


@pytest.fixture
def temp_profiles_file():
    """Create a temporary profiles file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_PROFILES, f)
        return Path(f.name)


@pytest.fixture
def registry(temp_profiles_file):
    """Create a RelationshipRegistry with test data."""
    return RelationshipRegistry(temp_profiles_file)


class TestRelationshipRegistryBasics:
    """Test basic registry functionality."""

    def test_load_profiles(self, registry):
        """Test that profiles are loaded correctly."""
        stats = registry.get_stats()
        assert stats["total"] == 3
        assert stats["friends"] == 2
        assert stats["family"] == 1

    def test_get_relationship_by_name(self, registry):
        """Test lookup by contact name."""
        info = registry.get_relationship("Alice")
        assert info is not None
        assert info.contact_name == "Alice"
        assert info.relationship == "friend"
        assert info.category == "friend"

    def test_get_relationship_by_phone(self, registry):
        """Test lookup by phone number."""
        info = registry.get_relationship("+15551234567")
        assert info is not None
        assert info.contact_name == "Alice"

    def test_get_relationship_not_found(self, registry):
        """Test lookup for non-existent contact."""
        info = registry.get_relationship("Unknown Person")
        assert info is None


class TestRelationshipFromChatId:
    """Test chat_id resolution."""

    def test_get_relationship_from_chat_id_imessage(self, registry):
        """Test resolving iMessage chat_id format."""
        info = registry.get_relationship_from_chat_id("iMessage;-;+15551234567")
        assert info is not None
        assert info.contact_name == "Alice"

    def test_get_relationship_from_chat_id_unknown(self, registry):
        """Test unknown phone in chat_id."""
        info = registry.get_relationship_from_chat_id("iMessage;-;+15559999999")
        assert info is None


class TestCategoryLookup:
    """Test category-based lookups."""

    def test_get_contacts_by_category_friend(self, registry):
        """Test getting all friends."""
        friends = registry.get_contacts_by_category("friend")
        assert len(friends) == 2
        assert "Alice" in friends
        assert "Bob" in friends

    def test_get_contacts_by_category_empty(self, registry):
        """Test getting non-existent category."""
        result = registry.get_contacts_by_category("invalid")
        assert result == []


class TestEdgeCases:
    """Test edge cases."""

    def test_missing_profiles_file(self):
        """Test handling of missing profiles file."""
        registry = RelationshipRegistry(Path("/nonexistent/path.json"))
        stats = registry.get_stats()
        assert stats["total"] == 0

    def test_phone_normalization(self, registry):
        """Test that phone numbers are normalized correctly."""
        # Test with different formats
        info = registry.get_relationship("+1 555 123 4567")
        if info:
            assert info.contact_name == "Alice"
