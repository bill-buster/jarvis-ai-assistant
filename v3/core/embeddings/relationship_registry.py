"""Relationship registry for JARVIS v2.

Loads labeled contact profiles and provides relationship-based lookups
for cross-conversation RAG (finding examples from similar relationships).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path to contact profiles
DEFAULT_PROFILES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "contacts" / "contact_profiles.json"
)

# Categories for relationship grouping
VALID_CATEGORIES = {"friend", "family", "work", "other"}


@dataclass
class RelationshipInfo:
    """Information about a contact's relationship."""

    contact_name: str
    relationship: str  # e.g., "best friend", "close friend", "coworker"
    category: str  # One of: friend, family, work, other
    is_group: bool
    phones: list[str] = field(default_factory=list)


class RelationshipRegistry:
    """Registry for contact relationships.

    Loads contact_profiles.json and provides:
    - Lookup by contact name or phone number
    - Category-based grouping for cross-conversation RAG
    - Phone number → contact name resolution
    """

    def __init__(self, profiles_path: Path | str | None = None):
        """Initialize registry.

        Args:
            profiles_path: Path to contact_profiles.json. If None, uses default.
        """
        self.profiles_path = Path(profiles_path) if profiles_path else DEFAULT_PROFILES_PATH

        # Contact name → RelationshipInfo
        self._contacts: dict[str, RelationshipInfo] = {}

        # Category → list of contact names
        self._by_category: dict[str, list[str]] = {cat: [] for cat in VALID_CATEGORIES}

        # Phone number → contact name (for chat_id resolution)
        self._phone_to_name: dict[str, str] = {}

        # Normalized name → original name (for fuzzy matching)
        self._normalized_names: dict[str, str] = {}

        self._loaded = False

    def _load(self) -> None:
        """Load profiles from disk."""
        if self._loaded:
            return

        if not self.profiles_path.exists():
            logger.warning(f"Contact profiles not found at {self.profiles_path}")
            self._loaded = True
            return

        try:
            with open(self.profiles_path) as f:
                profiles = json.load(f)

            for name, data in profiles.items():
                category = data.get("category", "other")
                if category not in VALID_CATEGORIES:
                    category = "other"

                info = RelationshipInfo(
                    contact_name=name,
                    relationship=data.get("relationship", "unknown"),
                    category=category,
                    is_group=data.get("is_group", False),
                    phones=data.get("phones", []),
                )

                self._contacts[name] = info
                self._by_category[category].append(name)

                # Index by phone numbers
                for phone in info.phones:
                    # Normalize phone: remove all non-digit except leading +
                    normalized = self._normalize_phone(phone)
                    if normalized:
                        self._phone_to_name[normalized] = name

                # Index by normalized name for fuzzy matching
                normalized_name = self._normalize_name(name)
                self._normalized_names[normalized_name] = name

            logger.info(
                f"Loaded {len(self._contacts)} contacts: "
                f"{len(self._by_category['friend'])} friends, "
                f"{len(self._by_category['family'])} family, "
                f"{len(self._by_category['work'])} work, "
                f"{len(self._by_category['other'])} other"
            )

            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load contact profiles: {e}")
            self._loaded = True

    def _normalize_phone(self, phone: str) -> str | None:
        """Normalize phone number for matching.

        Keeps only digits (and leading +). Returns None for invalid phones.
        """
        if not phone:
            return None

        # Keep leading + if present, then only digits
        if phone.startswith("+"):
            digits = "+" + re.sub(r"\D", "", phone[1:])
        else:
            digits = re.sub(r"\D", "", phone)

        # Must have at least 10 digits
        digit_count = len(re.sub(r"\D", "", digits))
        if digit_count < 10:
            return None

        return digits

    def _normalize_name(self, name: str) -> str:
        """Normalize name for fuzzy matching.

        Lowercases, removes extra whitespace, removes common suffixes.
        """
        name = name.lower().strip()
        name = re.sub(r"\s+", " ", name)  # Collapse whitespace
        return name

    def get_relationship(self, identifier: str) -> RelationshipInfo | None:
        """Get relationship info by contact name or phone number.

        Args:
            identifier: Contact name or phone number

        Returns:
            RelationshipInfo if found, None otherwise
        """
        self._load()

        # Try direct name lookup
        if identifier in self._contacts:
            return self._contacts[identifier]

        # Try normalized name lookup
        normalized = self._normalize_name(identifier)
        if normalized in self._normalized_names:
            name = self._normalized_names[normalized]
            return self._contacts.get(name)

        # Try phone lookup
        phone = self._normalize_phone(identifier)
        if phone and phone in self._phone_to_name:
            name = self._phone_to_name[phone]
            return self._contacts.get(name)

        return None

    def get_relationship_from_chat_id(self, chat_id: str) -> RelationshipInfo | None:
        """Get relationship info from an iMessage chat_id.

        iMessage chat_ids typically look like:
        - "iMessage;-;+15551234567" (1:1 chat)
        - "iMessage;+;chat123456" (group chat)

        Args:
            chat_id: iMessage chat identifier

        Returns:
            RelationshipInfo if found, None otherwise
        """
        self._load()

        # Extract phone from 1:1 chat_id
        phone_match = re.search(r"\+\d{10,}", chat_id)
        if phone_match:
            phone = phone_match.group()
            normalized = self._normalize_phone(phone)
            if normalized and normalized in self._phone_to_name:
                name = self._phone_to_name[normalized]
                return self._contacts.get(name)

        return None

    # TODO: Remove if unused - only used in tests
    def get_contacts_by_category(self, category: str) -> list[str]:
        """Get all contact names in a category.

        Args:
            category: One of: friend, family, work, other

        Returns:
            List of contact names in that category
        """
        self._load()
        return self._by_category.get(category, [])

    def get_similar_contacts(self, identifier: str) -> list[str]:
        """Get contacts with similar relationships.

        For RAG: when generating reply for "best friend",
        return all contacts in the "friend" category.

        Args:
            identifier: Contact name, phone, or chat_id

        Returns:
            List of contact names with similar relationships
        """
        self._load()

        # Try to get relationship info
        info = self.get_relationship(identifier) or self.get_relationship_from_chat_id(identifier)

        if not info:
            return []

        # Get all contacts in same category
        similar = self._by_category.get(info.category, [])

        # Exclude the contact itself
        return [name for name in similar if name != info.contact_name]

    def get_phones_for_contacts(self, contact_names: list[str]) -> dict[str, list[str]]:
        """Get phone numbers for a list of contacts.

        Used to resolve contact names to potential chat_ids.

        Args:
            contact_names: List of contact names

        Returns:
            Dict mapping contact name → list of phone numbers
        """
        self._load()

        result = {}
        for name in contact_names:
            if name in self._contacts:
                phones = self._contacts[name].phones
                if phones:
                    result[name] = phones
        return result

    # TODO: Remove if unused - only used in tests
    def get_all_phones_for_category(self, category: str) -> list[str]:
        """Get all phone numbers for contacts in a category.

        Args:
            category: One of: friend, family, work, other

        Returns:
            List of phone numbers (normalized)
        """
        self._load()

        phones = []
        for name in self._by_category.get(category, []):
            info = self._contacts.get(name)
            if info:
                for phone in info.phones:
                    normalized = self._normalize_phone(phone)
                    if normalized:
                        phones.append(normalized)
        return phones

    # TODO: Remove if unused - only used in tests
    def get_stats(self) -> dict[str, int]:
        """Get registry statistics.

        Returns:
            Dict with counts per category
        """
        self._load()
        return {
            "total": len(self._contacts),
            "friends": len(self._by_category["friend"]),
            "family": len(self._by_category["family"]),
            "work": len(self._by_category["work"]),
            "other": len(self._by_category["other"]),
            "phones_indexed": len(self._phone_to_name),
        }


# Singleton
_registry: RelationshipRegistry | None = None


def get_relationship_registry(profiles_path: Path | str | None = None) -> RelationshipRegistry:
    """Get singleton relationship registry.

    Args:
        profiles_path: Optional path override for testing

    Returns:
        RelationshipRegistry instance
    """
    global _registry

    if profiles_path is not None:
        # Custom path - create new instance (for testing)
        return RelationshipRegistry(profiles_path)

    if _registry is None:
        _registry = RelationshipRegistry()

    return _registry


def reset_relationship_registry() -> None:
    """Reset the singleton registry."""
    global _registry
    _registry = None
