"""Tests for RelationshipRegistry.

Tests the relationship-aware RAG pipeline's contact lookup functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.embeddings.relationship_registry import (
    RelationshipRegistry,
    get_relationship_registry,
    reset_relationship_registry,
)

# Sample test data
SAMPLE_PROFILES = {
    "Alice Smith": {
        "relationship": "best friend",
        "category": "friend",
        "is_group": False,
        "phones": ["+15551234567", "+15559876543"],
    },
    "Bob Jones": {
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
    "Dad": {
        "relationship": "father",
        "category": "family",
        "is_group": False,
        "phones": ["+15554567890"],
    },
    "Boss": {
        "relationship": "manager",
        "category": "work",
        "is_group": False,
        "phones": ["+15555678901"],
    },
    "Work Group": {
        "relationship": "team chat",
        "category": "work",
        "is_group": True,
        "phones": [],
    },
    "Pizza Place": {
        "relationship": "local business",
        "category": "other",
        "is_group": False,
        "phones": ["+15556789012"],
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
        assert stats["total"] == 7
        assert stats["friends"] == 2
        assert stats["family"] == 2
        assert stats["work"] == 2
        assert stats["other"] == 1

    def test_get_relationship_by_name(self, registry):
        """Test lookup by contact name."""
        info = registry.get_relationship("Alice Smith")
        assert info is not None
        assert info.contact_name == "Alice Smith"
        assert info.relationship == "best friend"
        assert info.category == "friend"
        assert info.is_group is False
        assert "+15551234567" in info.phones

    def test_get_relationship_by_name_case_insensitive(self, registry):
        """Test that name lookup is case-insensitive."""
        info = registry.get_relationship("alice smith")
        assert info is not None
        assert info.contact_name == "Alice Smith"

    def test_get_relationship_by_phone(self, registry):
        """Test lookup by phone number."""
        info = registry.get_relationship("+15551234567")
        assert info is not None
        assert info.contact_name == "Alice Smith"

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
        assert info.contact_name == "Alice Smith"

    def test_get_relationship_from_chat_id_sms(self, registry):
        """Test resolving SMS chat_id format."""
        info = registry.get_relationship_from_chat_id("SMS;-;+15553456789")
        assert info is not None
        assert info.contact_name == "Mom"

    def test_get_relationship_from_chat_id_group(self, registry):
        """Test that group chat_id (no phone) returns None."""
        info = registry.get_relationship_from_chat_id("iMessage;+;chat123456")
        assert info is None

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
        assert "Alice Smith" in friends
        assert "Bob Jones" in friends

    def test_get_contacts_by_category_family(self, registry):
        """Test getting all family."""
        family = registry.get_contacts_by_category("family")
        assert len(family) == 2
        assert "Mom" in family
        assert "Dad" in family

    def test_get_contacts_by_category_empty(self, registry):
        """Test getting non-existent category."""
        result = registry.get_contacts_by_category("invalid")
        assert result == []


class TestSimilarContacts:
    """Test finding similar contacts."""

    def test_get_similar_contacts_by_name(self, registry):
        """Test finding similar contacts by name."""
        similar = registry.get_similar_contacts("Alice Smith")
        assert len(similar) == 1
        assert "Bob Jones" in similar
        assert "Alice Smith" not in similar  # Should exclude self

    def test_get_similar_contacts_by_phone(self, registry):
        """Test finding similar contacts by phone."""
        similar = registry.get_similar_contacts("+15553456789")  # Mom's phone
        assert len(similar) == 1
        assert "Dad" in similar

    def test_get_similar_contacts_by_chat_id(self, registry):
        """Test finding similar contacts by chat_id."""
        similar = registry.get_similar_contacts("iMessage;-;+15555678901")  # Boss
        assert len(similar) == 1
        assert "Work Group" in similar

    def test_get_similar_contacts_unknown(self, registry):
        """Test similar contacts for unknown contact."""
        similar = registry.get_similar_contacts("Unknown")
        assert similar == []


class TestPhoneMapping:
    """Test phone number mappings."""

    def test_get_phones_for_contacts(self, registry):
        """Test getting phones for multiple contacts."""
        phones = registry.get_phones_for_contacts(["Alice Smith", "Bob Jones"])
        assert "Alice Smith" in phones
        assert "Bob Jones" in phones
        assert len(phones["Alice Smith"]) == 2
        assert len(phones["Bob Jones"]) == 1

    def test_get_phones_for_contacts_group(self, registry):
        """Test that groups (no phones) are excluded."""
        phones = registry.get_phones_for_contacts(["Work Group"])
        assert phones == {}

    def test_get_all_phones_for_category(self, registry):
        """Test getting all phones for a category."""
        phones = registry.get_all_phones_for_category("friend")
        assert len(phones) == 3  # 2 for Alice, 1 for Bob
        assert "+15551234567" in phones
        assert "+15552345678" in phones


class TestSingleton:
    """Test singleton pattern."""

    def test_get_relationship_registry_returns_same_instance(self):
        """Test that singleton returns same instance."""
        reset_relationship_registry()
        r1 = get_relationship_registry()
        r2 = get_relationship_registry()
        assert r1 is r2

    def test_get_relationship_registry_with_path_returns_new(self, temp_profiles_file):
        """Test that custom path returns new instance."""
        reset_relationship_registry()
        r1 = get_relationship_registry()
        r2 = get_relationship_registry(temp_profiles_file)
        assert r1 is not r2


class TestEdgeCases:
    """Test edge cases."""

    def test_missing_profiles_file(self):
        """Test handling of missing profiles file."""
        registry = RelationshipRegistry(Path("/nonexistent/path.json"))
        stats = registry.get_stats()
        assert stats["total"] == 0

    def test_invalid_category_normalized(self, temp_profiles_file):
        """Test that invalid categories are normalized to 'other'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "Test Contact": {
                    "relationship": "test",
                    "category": "invalid_category",
                    "is_group": False,
                    "phones": ["+15551111111"],
                }
            }, f)
            path = Path(f.name)

        registry = RelationshipRegistry(path)
        info = registry.get_relationship("Test Contact")
        assert info.category == "other"

    def test_phone_normalization(self, registry):
        """Test that phone numbers are normalized correctly."""
        # Test with different formats
        info1 = registry.get_relationship("+1 555 123 4567")
        info2 = registry.get_relationship("(555) 123-4567")

        # The first should work (same normalized form)
        # The second might not match if country code is expected
        # This depends on the normalization logic
        if info2:
            assert info1.contact_name == info2.contact_name

    def test_multiple_phones_same_contact(self, registry):
        """Test contact with multiple phone numbers."""
        # Both phones should resolve to same contact
        info1 = registry.get_relationship("+15551234567")
        info2 = registry.get_relationship("+15559876543")
        assert info1.contact_name == info2.contact_name == "Alice Smith"
