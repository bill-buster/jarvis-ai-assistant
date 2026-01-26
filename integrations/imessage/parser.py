"""Message parsing utilities for iMessage chat.db.

Handles:
- attributedBody NSKeyedArchive parsing
- Apple Core Data timestamp conversion
- Phone number normalization
- Attachment and reaction parsing
"""

import logging
import plistlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contracts.imessage import Attachment, Reaction

logger = logging.getLogger(__name__)

# Apple's Core Data epoch: 2001-01-01 00:00:00 UTC
# Timestamps are in nanoseconds since this epoch
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
APPLE_EPOCH_UNIX = int(APPLE_EPOCH.timestamp())  # 978307200
NANOSECONDS_PER_SECOND = 1_000_000_000


def datetime_to_apple_timestamp(dt: datetime) -> int:
    """Convert datetime to Apple Core Data timestamp (nanoseconds since 2001-01-01).

    Args:
        dt: datetime object to convert

    Returns:
        Integer nanoseconds since Apple epoch
    """
    return int((dt.timestamp() - APPLE_EPOCH_UNIX) * NANOSECONDS_PER_SECOND)


def _extract_from_typedstream(data: bytes) -> str | None:
    """Extract text from typedstream (legacy NSArchiver) format.

    Typedstream is Apple's older serialization format, used in some macOS versions
    for attributedBody. The format starts with 'streamtyped' magic bytes.

    Args:
        data: Raw bytes in typedstream format

    Returns:
        Extracted text string, or None if extraction fails
    """
    try:
        # Typedstream contains the text after NSString class marker
        # Look for the pattern: NSString marker followed by length-prefixed string
        nsstring_marker = b"NSString"
        idx = data.find(nsstring_marker)
        if idx == -1:
            return None

        # Skip past NSString and look for the actual string content
        # The format is: NSString + some bytes + length byte + string data
        search_start = idx + len(nsstring_marker)
        remaining = data[search_start:]

        # Skip metadata bytes until we find printable content
        # Look for a length-prefixed string (common pattern: \x01\x94\x84\x01+\x04Text)
        # The + character (0x2b) often precedes the length byte
        plus_idx = remaining.find(b"+")
        if plus_idx != -1 and plus_idx < 20:
            # Length byte follows the +
            length_pos = plus_idx + 1
            if length_pos < len(remaining):
                length = remaining[length_pos]
                text_start = length_pos + 1
                text_end = text_start + length
                if text_end <= len(remaining):
                    text_bytes = remaining[text_start:text_end]
                    try:
                        return text_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        pass

        # Fallback: scan for readable UTF-8 sequences
        # Find the longest printable sequence that isn't a class name
        skip_strings = {
            "streamtyped",
            "NSAttributedString",
            "NSObject",
            "NSString",
            "NSDictionary",
            "NSNumber",
            "NSValue",
            "NSArray",
            "NSMutableAttributedString",
            "NSMutableString",
            "__kIMMessagePartAttributeName",
            "__kIMFileTransferGUIDAttributeName",
            "__kIMDataDetectedAttributeName",
        }

        decoded = data.decode("utf-8", errors="ignore")
        # Find sequences of printable characters (at least 1 char)
        matches: list[str] = re.findall(r"[\x20-\x7e\u00a0-\uffff]+", decoded)
        for match in matches:
            clean = match.strip()
            if clean and clean not in skip_strings and not clean.startswith("$"):
                # Skip if it looks like a class name or metadata
                if not any(skip in clean for skip in ["NS", "kIM", "Attribute"]):
                    return clean

        return None
    except Exception as e:
        logger.debug(f"Failed to parse typedstream: {e}")
        return None


def parse_attributed_body(data: bytes | None) -> str | None:
    """Extract plain text from attributedBody column.

    The attributedBody column in chat.db contains serialized NSAttributedString
    in one of two formats:
    1. NSKeyedArchive (binary plist) - newer format
    2. Typedstream (legacy NSArchiver) - older format, starts with 'streamtyped'

    Args:
        data: Raw bytes from attributedBody column, or None

    Returns:
        Extracted text string, or None if parsing fails
    """
    if data is None:
        return None

    # Check for typedstream format (starts with \x04\x0bstreamtyped or similar)
    if b"streamtyped" in data[:20]:
        result = _extract_from_typedstream(data)
        if result:
            return result

    try:
        # Try to load as binary plist (NSKeyedArchive format)
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
        # Plist parsing failed, try typedstream as fallback
        logger.debug("Failed to parse attributedBody as plist, trying typedstream")
        return _extract_from_typedstream(data)
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


def parse_attachments(attachment_rows: list[dict[str, Any]] | None) -> list[Attachment]:
    """Parse attachment metadata from database rows.

    Args:
        attachment_rows: List of attachment rows from database query.
            Each row should have: filename, mime_type, file_size, transfer_name

    Returns:
        List of Attachment objects with metadata
    """
    if not attachment_rows:
        return []

    attachments = []
    for row in attachment_rows:
        # filename is the full path (e.g., ~/Library/Messages/Attachments/...)
        filename_raw = row.get("filename") or row.get("transfer_name") or ""

        # Extract just the filename for display
        if filename_raw:
            # Handle ~ in path
            if filename_raw.startswith("~"):
                full_path = str(Path(filename_raw).expanduser())
            else:
                full_path = filename_raw
            display_name = Path(filename_raw).name
        else:
            full_path = None
            display_name = "unknown"

        attachment = Attachment(
            filename=display_name,
            file_path=full_path,
            mime_type=row.get("mime_type"),
            file_size=row.get("file_size"),
        )
        attachments.append(attachment)

    return attachments


def parse_reactions(reaction_rows: list[dict[str, Any]] | None) -> list[Reaction]:
    """Extract tapback reactions from associated message rows.

    iMessage stores reactions as messages with associated_message_type:
    - 2000: Love
    - 2001: Like (thumbs up)
    - 2002: Dislike (thumbs down)
    - 2003: Laugh
    - 2004: Emphasize (exclamation marks)
    - 2005: Question
    - 3000-3005: Removed versions of the above reactions

    Args:
        reaction_rows: List of reaction message rows from database.
            Each row should have: associated_message_type, date, is_from_me, sender

    Returns:
        List of Reaction objects
    """
    if not reaction_rows:
        return []

    reactions = []
    for row in reaction_rows:
        reaction_type = _get_reaction_type(row.get("associated_message_type", 0))
        if reaction_type is None:
            continue

        # Parse the timestamp
        date = parse_apple_timestamp(row.get("date"))

        # Determine sender
        if row.get("is_from_me"):
            sender = "me"
        else:
            sender = normalize_phone_number(row.get("sender")) or row.get("sender") or "unknown"

        reaction = Reaction(
            type=reaction_type,
            sender=sender,
            sender_name=None,  # Will be resolved by reader if contacts available
            date=date,
        )
        reactions.append(reaction)

    return reactions


def _get_reaction_type(associated_message_type: int) -> str | None:
    """Convert associated_message_type to reaction type string.

    Args:
        associated_message_type: The numeric type from the message table

    Returns:
        Reaction type string, or None if not a valid reaction type
    """
    # Standard tapback types (added reactions)
    tapback_types = {
        2000: "love",
        2001: "like",
        2002: "dislike",
        2003: "laugh",
        2004: "emphasize",
        2005: "question",
    }

    # Removed tapback types (3000 series = removed reactions)
    removed_tapback_types = {
        3000: "removed_love",
        3001: "removed_like",
        3002: "removed_dislike",
        3003: "removed_laugh",
        3004: "removed_emphasize",
        3005: "removed_question",
    }

    if associated_message_type in tapback_types:
        return tapback_types[associated_message_type]
    elif associated_message_type in removed_tapback_types:
        return removed_tapback_types[associated_message_type]
    else:
        return None


def normalize_phone_number(phone: str | None) -> str | None:
    """Normalize phone number format.

    Strips formatting characters and ensures consistent format.

    Args:
        phone: Raw phone number string

    Returns:
        Normalized phone number, None if input is None, or original string if not a phone number
    """
    if phone is None:
        return None

    phone = phone.strip()

    # If it's an email, return as-is
    if "@" in phone:
        return phone

    # Check if number already has + prefix before stripping
    has_plus = phone.startswith("+")

    # Remove common formatting characters (but not +)
    cleaned = re.sub(r"[\s\-\(\)\.]", "", phone)

    # If already has + prefix, return cleaned number
    if has_plus or cleaned.startswith("+"):
        return cleaned

    # Add appropriate prefix for numbers without +
    if cleaned.startswith("1") and len(cleaned) == 11:
        # US number with country code
        return "+" + cleaned
    elif len(cleaned) == 10:
        # Assume US number without country code
        return "+1" + cleaned

    # For other formats (international without +), return as-is
    # since we can't reliably determine the country code
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
