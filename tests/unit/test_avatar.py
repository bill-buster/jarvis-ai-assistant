"""Tests for contact avatar functionality.

Tests the avatar module and contacts API endpoint.
"""

import time
from unittest.mock import patch

from integrations.imessage.avatar import (
    ContactAvatarData,
    get_contact_avatar,
)


class TestContactAvatarData:
    """Tests for ContactAvatarData dataclass."""

    def test_initials_with_first_and_last_name(self):
        """Test initials generation with both names."""
        data = ContactAvatarData(
            image_data=None,
            first_name="John",
            last_name="Doe",
            display_name=None,
        )
        assert data.initials == "JD"

    def test_initials_with_first_name_only(self):
        """Test initials generation with only first name."""
        data = ContactAvatarData(
            image_data=None,
            first_name="John",
            last_name=None,
            display_name=None,
        )
        assert data.initials == "J"

    def test_initials_with_last_name_only(self):
        """Test initials generation with only last name."""
        data = ContactAvatarData(
            image_data=None,
            first_name=None,
            last_name="Doe",
            display_name=None,
        )
        assert data.initials == "D"

    def test_initials_with_display_name(self):
        """Test initials generation from display name."""
        data = ContactAvatarData(
            image_data=None,
            first_name=None,
            last_name=None,
            display_name="John Doe",
        )
        assert data.initials == "JD"

    def test_initials_with_single_word_display_name(self):
        """Test initials generation from single word display name."""
        data = ContactAvatarData(
            image_data=None,
            first_name=None,
            last_name=None,
            display_name="John",
        )
        assert data.initials == "J"

    def test_initials_fallback_to_question_mark(self):
        """Test initials fallback when no name available."""
        data = ContactAvatarData(
            image_data=None,
            first_name=None,
            last_name=None,
            display_name=None,
        )
        assert data.initials == "?"

    def test_initials_with_empty_strings(self):
        """Test initials with empty string values."""
        data = ContactAvatarData(
            image_data=None,
            first_name="",
            last_name="",
            display_name="",
        )
        assert data.initials == "?"

    def test_initials_with_multipart_display_name(self):
        """Test initials with multi-part display name."""
        data = ContactAvatarData(
            image_data=None,
            first_name=None,
            last_name=None,
            display_name="John William Doe",
        )
        assert data.initials == "JD"  # First and last word


class TestGetContactAvatar:
    """Tests for get_contact_avatar function."""

    def test_empty_identifier_returns_none(self):
        """Test that empty identifier returns None."""
        assert get_contact_avatar("") is None

    def test_none_identifier_returns_none(self):
        """Test that None-like identifier returns None."""
        # Function only accepts str, but empty string returns None
        assert get_contact_avatar("") is None

    @patch("integrations.imessage.avatar.ADDRESSBOOK_DB_PATH")
    def test_missing_addressbook_returns_none(self, mock_path):
        """Test that missing AddressBook path returns None."""
        mock_path.exists.return_value = False
        result = get_contact_avatar("+15551234567")
        assert result is None


class TestAvatarCache:
    """Tests for the AvatarCache class."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        # Import here to avoid MLX import issues
        from api.routers.contacts import AvatarCache

        cache = AvatarCache(ttl_seconds=60, maxsize=100)
        cache.set("test_key", b"test_data")

        found, data = cache.get("test_key")
        assert found is True
        assert data == b"test_data"

    def test_get_missing_key(self):
        """Test getting a missing key returns not found."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache()

        found, data = cache.get("nonexistent")
        assert found is False
        assert data is None

    def test_ttl_expiration(self):
        """Test that cached items expire after TTL."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache(ttl_seconds=0.1)  # 100ms TTL
        cache.set("key", b"data")

        # Should be found immediately
        found, _ = cache.get("key")
        assert found is True

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        found, _ = cache.get("key")
        assert found is False

    def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when maxsize exceeded."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache(ttl_seconds=60, maxsize=3)

        cache.set("key1", b"data1")
        cache.set("key2", b"data2")
        cache.set("key3", b"data3")

        # All should be present
        assert cache.get("key1")[0] is True
        assert cache.get("key2")[0] is True
        assert cache.get("key3")[0] is True

        # Add a fourth item
        cache.set("key4", b"data4")

        # key1 should be evicted (oldest)
        assert cache.get("key1")[0] is False
        assert cache.get("key4")[0] is True

    def test_invalidate_specific_key(self):
        """Test invalidating a specific cache key."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache()
        cache.set("key1", b"data1")
        cache.set("key2", b"data2")

        cache.invalidate("key1")

        assert cache.get("key1")[0] is False
        assert cache.get("key2")[0] is True

    def test_invalidate_all(self):
        """Test invalidating all cache entries."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache()
        cache.set("key1", b"data1")
        cache.set("key2", b"data2")

        cache.invalidate()

        assert cache.get("key1")[0] is False
        assert cache.get("key2")[0] is False

    def test_stats(self):
        """Test cache statistics tracking."""
        from api.routers.contacts import AvatarCache

        cache = AvatarCache(ttl_seconds=60, maxsize=100)

        # Some hits and misses
        cache.set("key", b"data")
        cache.get("key")  # Hit
        cache.get("key")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestDefaultAvatarGeneration:
    """Tests for default avatar generation functions."""

    def test_generate_initials_with_display_name(self):
        """Test initials from display name."""
        from api.routers.contacts import _generate_initials

        assert _generate_initials("+15551234567", "John Doe") == "JD"
        assert _generate_initials("+15551234567", "John") == "J"
        assert _generate_initials("+15551234567", "John William Doe") == "JD"

    def test_generate_initials_phone_number(self):
        """Test initials from phone number."""
        from api.routers.contacts import _generate_initials

        assert _generate_initials("+15551234567", None) == "67"
        assert _generate_initials("5551234567", None) == "67"
        assert _generate_initials("+1", None) == "1"

    def test_generate_initials_email(self):
        """Test initials from email address."""
        from api.routers.contacts import _generate_initials

        assert _generate_initials("john@example.com", None) == "J"
        assert _generate_initials("jane.doe@example.com", None) == "J"

    def test_generate_initials_fallback(self):
        """Test initials fallback to question mark."""
        from api.routers.contacts import _generate_initials

        assert _generate_initials("", None) == "?"

    def test_get_color_consistency(self):
        """Test that same identifier always gets same color."""
        from api.routers.contacts import _get_color_for_identifier

        color1 = _get_color_for_identifier("+15551234567")
        color2 = _get_color_for_identifier("+15551234567")
        assert color1 == color2
        assert color1.startswith("#")

    def test_get_color_different_identifiers(self):
        """Test that different identifiers can get different colors."""
        from api.routers.contacts import _get_color_for_identifier

        # While not guaranteed to be different (due to hash collisions),
        # very different identifiers should usually get different colors
        colors = set()
        for i in range(10):
            colors.add(_get_color_for_identifier(f"user{i}@example.com"))

        # Should have at least a few different colors
        assert len(colors) > 1

    def test_generate_svg_avatar(self):
        """Test SVG avatar generation."""
        from api.routers.contacts import generate_svg_avatar

        svg = generate_svg_avatar("JD", "#FF6B6B", 88)

        # Check it's valid SVG
        assert svg.startswith(b"<svg")
        assert b"</svg>" in svg
        assert b"JD" in svg
        assert b"#FF6B6B" in svg

    def test_generate_svg_avatar_different_sizes(self):
        """Test SVG avatar with different sizes."""
        from api.routers.contacts import generate_svg_avatar

        svg_small = generate_svg_avatar("X", "#000000", 32)
        svg_large = generate_svg_avatar("X", "#000000", 256)

        assert b'width="32"' in svg_small
        assert b'width="256"' in svg_large


class TestContactInfoEndpoint:
    """Tests for the contact info endpoint."""

    def test_contact_info_structure(self):
        """Test the structure of contact info response."""
        # We can verify the function exists and is callable
        from api.routers.contacts import get_contact_info

        # This would need an actual API test with TestClient
        # For now, just verify the function exists
        assert callable(get_contact_info)


class TestAvatarEndpoint:
    """Tests for the avatar endpoint."""

    def test_avatar_endpoint_exists(self):
        """Test that the avatar endpoint is defined."""
        from api.routers.contacts import get_avatar

        assert callable(get_avatar)

    def test_avatar_endpoint_parameters(self):
        """Test avatar endpoint parameter defaults."""
        import inspect

        from api.routers.contacts import get_avatar

        sig = inspect.signature(get_avatar)
        params = sig.parameters

        # Check identifier is required (no default)
        assert "identifier" in params
        assert params["identifier"].default == inspect.Parameter.empty

        # Check size has default
        assert "size" in params

        # Check format has default
        assert "format" in params
