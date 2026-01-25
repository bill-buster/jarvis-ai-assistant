"""Message parsing utilities for iMessage chat.db.

Handles:
- attributedBody NSKeyedArchive parsing
- Apple Core Data timestamp conversion
- Phone number normalization
- Attachment and reaction parsing (stubs for v1)
"""

import logging
import plistlib
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Apple's Core Data epoch: 2001-01-01 00:00:00 UTC
# Timestamps are in nanoseconds since this epoch
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
APPLE_EPOCH_UNIX = 978307200  # Unix timestamp for 2001-01-01 00:00:00 UTC
NANOSECONDS_PER_SECOND = 1_000_000_000


def datetime_to_apple_timestamp(dt: datetime) -> int:
    """Convert datetime to Apple Core Data timestamp (nanoseconds since 2001-01-01).

    Args:
        dt: datetime object to convert

    Returns:
        Integer nanoseconds since Apple epoch
    """
    return int((dt.timestamp() - APPLE_EPOCH_UNIX) * NANOSECONDS_PER_SECOND)


def parse_attributed_body(data: bytes | None) -> str | None:
    """Extract plain text from NSKeyedArchive attributedBody.

    The attributedBody column in chat.db contains an NSKeyedArchive-encoded
    NSAttributedString. This function attempts to extract the plain text
    content using plistlib.

    Args:
        data: Raw bytes from attributedBody column, or None

    Returns:
        Extracted text string, or None if parsing fails
    """
    if data is None:
        return None

    try:
        # Try to load as binary plist
        plist = plistlib.loads(data)

        # The structure varies, but text is usually under NS.string or NSString
        # Try common locations in the plist structure
        if isinstance(plist, dict):
            # Check for $objects array (NSKeyedArchiver format)
            objects = plist.get("$objects", [])
            for obj in objects:
                if isinstance(obj, str) and len(obj) > 0:
                    # Skip metadata strings
                    if obj.startswith("$") or obj in (
                        "NSMutableAttributedString",
                        "NSAttributedString",
                        "NSMutableString",
                        "NSString",
                        "NSDictionary",
                        "NSArray",
                    ):
                        continue
                    return obj

                # Check for NSString or NS.string in dict objects
                if isinstance(obj, dict):
                    for key in ("NS.string", "NSString"):
                        if key in obj:
                            value = obj[key]
                            if isinstance(value, str):
                                return value

        return None

    except plistlib.InvalidFileException:
        logger.debug("Failed to parse attributedBody as plist")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error parsing attributedBody: {e}")
        return None


def parse_apple_timestamp(timestamp: int | float | None) -> datetime:
    """Convert Apple Core Data timestamp to datetime.

    Apple stores timestamps as nanoseconds since 2001-01-01 00:00:00 UTC.

    Args:
        timestamp: Nanoseconds since Apple epoch, or None

    Returns:
        datetime object in UTC, or APPLE_EPOCH if timestamp is invalid
    """
    if timestamp is None or timestamp == 0:
        return APPLE_EPOCH

    try:
        # Convert nanoseconds to seconds
        seconds = timestamp / NANOSECONDS_PER_SECOND
        return datetime.fromtimestamp(
            APPLE_EPOCH.timestamp() + seconds,
            tz=UTC,
        )
    except (ValueError, OSError, OverflowError) as e:
        logger.debug(f"Failed to parse timestamp {timestamp}: {e}")
        return APPLE_EPOCH


def parse_attachments(attachment_data: str | None) -> list[str]:
    """Parse attachment filenames from message.

    Args:
        attachment_data: Raw attachment data from database

    Returns:
        List of attachment filenames (empty list for v1)

    Note:
        This is a stub for v1. Full implementation in WS10.1 will query
        the attachment table via message_attachment_join.
    """
    # TODO: Implement attachment parsing (WS10.1)
    # Requires JOIN with attachment table
    return []


def parse_reactions(associated_messages: list[dict[str, Any]] | None) -> list[str]:
    """Extract tapback reactions from associated messages.

    Args:
        associated_messages: List of associated message dicts

    Returns:
        List of reaction strings (empty list for v1)

    Note:
        This is a stub for v1. Full implementation in WS10.1 will query
        messages with associated_message_guid pointing to the target.
    """
    # TODO: Implement tapback parsing (WS10.1)
    # Requires querying messages by associated_message_guid
    return []


def normalize_phone_number(phone: str | None) -> str:
    """Normalize phone number format.

    Strips formatting characters and ensures consistent format.

    Args:
        phone: Raw phone number string

    Returns:
        Normalized phone number, or original string if not a phone number
    """
    if phone is None:
        return ""

    # If it's an email, return as-is
    if "@" in phone:
        return phone.strip()

    # Remove common formatting characters
    cleaned = re.sub(r"[\s\-\(\)\.]", "", phone)

    # Ensure + prefix for international numbers
    if cleaned.startswith("1") and len(cleaned) == 11:
        cleaned = "+" + cleaned
    elif len(cleaned) == 10:
        # Assume US number
        cleaned = "+1" + cleaned

    return cleaned


def extract_text_from_row(row: dict[str, Any]) -> str:
    """Extract message text from database row.

    Tries text column first, falls back to attributedBody parsing.

    Args:
        row: Database row as dict with 'text' and 'attributedBody' keys

    Returns:
        Message text string, or empty string if no text available
    """
    # Try text column first
    text = row.get("text")
    if text and isinstance(text, str) and text.strip():
        return str(text.strip())

    # Fall back to attributedBody
    attributed_body = row.get("attributedBody")
    if attributed_body:
        parsed = parse_attributed_body(attributed_body)
        if parsed:
            return parsed.strip()

    return ""
