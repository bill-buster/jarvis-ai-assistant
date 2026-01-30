"""Simplified iMessage reader for JARVIS v2.

Read-only access to macOS iMessage chat.db database with contact resolution.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default iMessage database path
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Path to macOS AddressBook database for contact name resolution
ADDRESSBOOK_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"

# Apple's timestamp epoch (2001-01-01)
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

# Database connection timeout
DB_TIMEOUT_SECONDS = 5.0


@dataclass
class Message:
    """A single iMessage."""

    id: int
    text: str
    sender: str
    sender_name: str | None  # Resolved contact name
    is_from_me: bool
    timestamp: datetime | None
    chat_id: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "is_from_me": self.is_from_me,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "chat_id": self.chat_id,
        }


@dataclass
class Conversation:
    """An iMessage conversation (chat)."""

    chat_id: str
    display_name: str | None
    participants: list[str] = field(default_factory=list)
    last_message_date: datetime | None = None
    last_message_text: str | None = None
    last_message_is_from_me: bool = False
    message_count: int = 0
    is_group: bool = False

    def to_dict(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "display_name": self.display_name,
            "participants": self.participants,
            "last_message_date": self.last_message_date.isoformat() if self.last_message_date else None,
            "last_message_text": self.last_message_text,
            "last_message_is_from_me": self.last_message_is_from_me,
            "message_count": self.message_count,
            "is_group": self.is_group,
        }


def _parse_apple_timestamp(timestamp: int | None) -> datetime | None:
    """Convert Apple timestamp (nanoseconds since 2001-01-01) to datetime."""
    if timestamp is None or timestamp == 0:
        return None
    # Apple timestamps are in nanoseconds since 2001-01-01
    seconds = timestamp / 1_000_000_000
    return datetime.fromtimestamp(APPLE_EPOCH.timestamp() + seconds, tz=timezone.utc)


def _datetime_to_apple_timestamp(dt: datetime) -> int:
    """Convert datetime to Apple timestamp (nanoseconds since 2001-01-01)."""
    seconds = dt.timestamp() - APPLE_EPOCH.timestamp()
    return int(seconds * 1_000_000_000)


def _parse_attributed_body(data: bytes | None) -> str | None:
    """Extract plain text from attributedBody blob.

    Handles two formats:
    1. Typedstream (legacy) - starts with 'streamtyped'
    2. NSKeyedArchive (binary plist) - newer format
    """
    if not data:
        return None

    # Strings to skip (metadata, not actual content)
    skip_strings = {
        "streamtyped", "NSAttributedString", "NSObject", "NSString",
        "NSDictionary", "NSNumber", "NSValue", "NSArray",
        "NSMutableAttributedString", "NSMutableString",
        "__kIMMessagePartAttributeName", "__kIMFileTransferGUIDAttributeName",
    }

    try:
        # Check for typedstream format
        if b"streamtyped" in data[:20]:
            return _extract_from_typedstream(data, skip_strings)

        # Try NSKeyedArchive (binary plist) format
        import plistlib
        try:
            plist = plistlib.loads(data)
            if isinstance(plist, dict):
                objects = plist.get("$objects", [])
                for obj in objects:
                    if isinstance(obj, str) and len(obj) > 0:
                        if obj.startswith("$") or obj in skip_strings:
                            continue
                        return obj
        except Exception:
            pass

        return None
    except Exception:
        return None


def _extract_from_typedstream(data: bytes, skip_strings: set) -> str | None:
    """Extract text from typedstream (legacy NSArchiver) format."""
    try:
        # Look for NSString marker and extract the text that follows
        nsstring_marker = b"NSString"
        idx = data.find(nsstring_marker)
        if idx == -1:
            return None

        # Skip past NSString and find the actual string content
        search_start = idx + len(nsstring_marker)
        remaining = data[search_start:]

        # Look for length-prefixed string (pattern: + followed by length byte)
        plus_idx = remaining.find(b"+")
        if plus_idx != -1 and plus_idx < 20:
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

        # Fallback: find longest printable sequence that isn't metadata
        decoded = data.decode("utf-8", errors="ignore")
        matches = re.findall(r'[\x20-\x7e\u00a0-\uffff]+', decoded)

        best_match = None
        best_length = 0

        for match in matches:
            clean = match.strip()
            if clean and clean not in skip_strings and not clean.startswith("$"):
                if len(clean) > best_length:
                    best_match = clean
                    best_length = len(clean)

        return best_match

    except Exception:
        return None


def _normalize_phone_number(phone: str | None) -> str | None:
    """Normalize a phone number to E.164-ish format for consistent lookups.

    Args:
        phone: Raw phone number string

    Returns:
        Normalized phone number or None if invalid
    """
    if not phone:
        return None

    # If it's an email, return lowercase
    if "@" in phone:
        return phone.lower().strip()

    # Remove all non-digit characters except leading +
    has_plus = phone.startswith("+")
    digits = re.sub(r'\D', '', phone)

    if not digits:
        return None

    # Add country code if missing (assume US +1)
    if len(digits) == 10:
        digits = "1" + digits

    return f"+{digits}" if has_plus or len(digits) > 10 else digits


class MessageReader:
    """Read-only access to iMessage database with contact resolution."""

    def __init__(self, db_path: Path | None = None):
        """Initialize reader.

        Args:
            db_path: Path to chat.db, defaults to ~/Library/Messages/chat.db
        """
        self.db_path = db_path or CHAT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._schema_version: str = "v14"
        self._contacts_cache: dict[str, str] | None = None

    def __enter__(self) -> MessageReader:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (lazy, cached)."""
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"iMessage database not found: {self.db_path}")

            try:
                uri = f"file:{self.db_path}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
                self._conn.row_factory = sqlite3.Row
                self._detect_schema()
            except sqlite3.OperationalError as e:
                if "unable to open" in str(e).lower():
                    raise PermissionError(
                        f"Cannot access iMessage database. Grant Full Disk Access in "
                        f"System Settings > Privacy & Security > Full Disk Access"
                    ) from e
                raise

        return self._conn

    def _detect_schema(self) -> None:
        """Detect database schema version."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(message)")
            columns = {row[1] for row in cursor.fetchall()}

            if "thread_originator_guid" in columns:
                cursor.execute("PRAGMA table_info(chat)")
                chat_columns = {row[1] for row in cursor.fetchall()}
                self._schema_version = "v15" if "service_name" in chat_columns else "v14"
            else:
                self._schema_version = "v14"

            logger.debug(f"Detected schema version: {self._schema_version}")
        except sqlite3.Error:
            self._schema_version = "v14"

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._contacts_cache = None

    def check_access(self) -> bool:
        """Check if database is accessible."""
        try:
            self._get_connection()
            return True
        except (FileNotFoundError, PermissionError, sqlite3.Error):
            return False

    # =========================================================================
    # Contact Resolution
    # =========================================================================

    def _resolve_contact_name(self, identifier: str) -> str | None:
        """Resolve a phone number or email to a contact name.

        Args:
            identifier: Phone number or email address

        Returns:
            Contact name if found, None otherwise
        """
        if not identifier or identifier == "me":
            return None

        # Lazily load contacts cache
        if self._contacts_cache is None:
            self._load_contacts_cache()

        # Normalize the identifier for lookup
        normalized = _normalize_phone_number(identifier)
        if normalized is None:
            return None

        return self._contacts_cache.get(normalized)

    def _load_contacts_cache(self) -> None:
        """Load contacts from ALL AddressBook databases into cache.

        Reads from all source databases (iCloud, Google, On My Mac, etc.)
        in ~/Library/Application Support/AddressBook/Sources/
        """
        self._contacts_cache = {}

        if not ADDRESSBOOK_PATH.exists():
            logger.debug("AddressBook path not found, contact resolution disabled")
            return

        try:
            loaded_count = 0
            for source_dir in ADDRESSBOOK_PATH.iterdir():
                if not source_dir.is_dir():
                    continue
                ab_db = source_dir / "AddressBook-v22.abcddb"
                if ab_db.exists():
                    self._load_contacts_from_db(ab_db)
                    loaded_count += 1

            logger.debug(f"Loaded {len(self._contacts_cache)} contacts from {loaded_count} AddressBook sources")
        except PermissionError:
            logger.debug("Permission denied accessing AddressBook directory")
        except OSError as e:
            logger.debug(f"I/O error accessing AddressBook: {e}")

    def _load_contacts_from_db(self, db_path: Path) -> None:
        """Load contacts from a specific AddressBook database.

        Args:
            db_path: Path to the AddressBook database
        """
        if self._contacts_cache is None:
            self._contacts_cache = {}

        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
            try:
                conn.row_factory = sqlite3.Row

                cursor = conn.cursor()

                # Load phone numbers with names
                try:
                    cursor.execute("""
                        SELECT
                            ZABCDPHONENUMBER.ZFULLNUMBER as identifier,
                            ZABCDRECORD.ZFIRSTNAME as first_name,
                            ZABCDRECORD.ZLASTNAME as last_name
                        FROM ZABCDPHONENUMBER
                        JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK
                        WHERE ZABCDPHONENUMBER.ZFULLNUMBER IS NOT NULL
                    """)
                    for row in cursor.fetchall():
                        identifier = _normalize_phone_number(row["identifier"])
                        name = self._format_name(row["first_name"], row["last_name"])
                        if identifier and name:
                            self._contacts_cache[identifier] = name
                except sqlite3.Error:
                    pass  # Table structure may differ

                # Load email addresses with names
                try:
                    cursor.execute("""
                        SELECT
                            ZABCDEMAILADDRESS.ZADDRESS as identifier,
                            ZABCDRECORD.ZFIRSTNAME as first_name,
                            ZABCDRECORD.ZLASTNAME as last_name
                        FROM ZABCDEMAILADDRESS
                        JOIN ZABCDRECORD ON ZABCDEMAILADDRESS.ZOWNER = ZABCDRECORD.Z_PK
                        WHERE ZABCDEMAILADDRESS.ZADDRESS IS NOT NULL
                    """)
                    for row in cursor.fetchall():
                        identifier = row["identifier"]
                        name = self._format_name(row["first_name"], row["last_name"])
                        if identifier and name:
                            self._contacts_cache[identifier.lower()] = name
                except sqlite3.Error:
                    pass  # Table structure may differ

            finally:
                conn.close()

        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error loading contacts from {db_path}: {e}")

    @staticmethod
    def _format_name(first_name: str | None, last_name: str | None) -> str | None:
        """Format first and last name into a display name."""
        parts = []
        if first_name:
            parts.append(first_name)
        if last_name:
            parts.append(last_name)
        return " ".join(parts) if parts else None

    # =========================================================================
    # Conversations
    # =========================================================================

    def get_conversations(self, limit: int | None = 50) -> list[Conversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return (None for all)

        Returns:
            List of conversations, newest first
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        limit_clause = "LIMIT ?" if limit is not None else ""

        # Optimized query using CTEs to avoid N+1 correlated subqueries
        # - chat_stats: single pass for message_count per chat
        # - last_messages: single pass with ROW_NUMBER to get last message per chat
        query = f"""
            WITH chat_stats AS (
                SELECT
                    chat_id,
                    COUNT(*) as message_count
                FROM chat_message_join
                GROUP BY chat_id
            ),
            last_messages AS (
                SELECT
                    cmj.chat_id,
                    m.text,
                    m.attributedBody,
                    m.is_from_me,
                    m.date,
                    ROW_NUMBER() OVER (PARTITION BY cmj.chat_id ORDER BY m.date DESC) as rn
                FROM chat_message_join cmj
                JOIN message m ON cmj.message_id = m.ROWID
            )
            SELECT
                c.guid as chat_id,
                c.display_name,
                (
                    SELECT GROUP_CONCAT(h.id, ', ')
                    FROM chat_handle_join chj
                    JOIN handle h ON chj.handle_id = h.ROWID
                    WHERE chj.chat_id = c.ROWID
                ) as participants,
                cs.message_count,
                lm.date as last_message_date,
                lm.text as last_message_text,
                lm.attributedBody as last_message_attributed_body,
                lm.is_from_me as last_message_is_from_me
            FROM chat c
            JOIN chat_stats cs ON cs.chat_id = c.ROWID
            LEFT JOIN last_messages lm ON lm.chat_id = c.ROWID AND lm.rn = 1
            WHERE cs.message_count > 0
            ORDER BY lm.date DESC
            {limit_clause}
        """

        try:
            params = (limit,) if limit is not None else ()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching conversations: {e}")
            return []

        # Ensure contacts cache is loaded once before processing
        if self._contacts_cache is None:
            self._load_contacts_cache()

        conversations = []
        for row in rows:
            participants_str = row["participants"] or ""
            raw_participants = [p.strip() for p in participants_str.split(",") if p.strip()]

            # Single pass: normalize and resolve contact names together
            participants = []
            resolved_participants = []
            for p in raw_participants:
                normalized = _normalize_phone_number(p)
                if normalized:
                    participants.append(normalized)
                    # Direct cache lookup (skip re-normalizing in _resolve_contact_name)
                    name = self._contacts_cache.get(normalized) if self._contacts_cache else None
                    resolved_participants.append(name if name else normalized)

            is_group = len(participants) > 1

            # Get display name - from DB or build from resolved participants
            display_name = row["display_name"] or None
            if not display_name:
                if is_group and resolved_participants:
                    # For groups, join first 3 names
                    names = resolved_participants[:3]
                    more = f" +{len(resolved_participants) - 3}" if len(resolved_participants) > 3 else ""
                    display_name = ", ".join(names) + more
                elif len(resolved_participants) == 1:
                    display_name = resolved_participants[0]

            last_message_text = row["last_message_text"]
            if not last_message_text:
                last_message_text = _parse_attributed_body(row["last_message_attributed_body"])

            conversations.append(Conversation(
                chat_id=row["chat_id"],
                display_name=display_name,
                participants=participants,
                last_message_date=_parse_apple_timestamp(row["last_message_date"]),
                last_message_text=last_message_text,
                last_message_is_from_me=bool(row["last_message_is_from_me"]),
                message_count=row["message_count"],
                is_group=is_group,
            ))

        return conversations

    # =========================================================================
    # Messages
    # =========================================================================

    def get_messages(
        self,
        chat_id: str,
        limit: int | None = 50,
        before: datetime | None = None,
        is_from_me: bool | None = None,
    ) -> list[Message]:
        """Get messages from a conversation.

        Args:
            chat_id: Conversation ID
            limit: Maximum number of messages (None for all)
            before: Only return messages before this datetime (for pagination)
            is_from_me: If True, only user's messages. If False, only others'. If None, all.

        Returns:
            List of messages, newest first
        """
        if not chat_id:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query with optional filters
        params: list = [chat_id]

        before_clause = ""
        if before is not None:
            before_clause = "AND message.date < ?"
            params.append(_datetime_to_apple_timestamp(before))

        from_me_clause = ""
        if is_from_me is not None:
            from_me_clause = "AND message.is_from_me = ?"
            params.append(1 if is_from_me else 0)

        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            SELECT
                message.ROWID as id,
                message.text,
                message.attributedBody,
                COALESCE(handle.id, 'me') as sender,
                message.is_from_me,
                message.date as timestamp,
                chat.guid as chat_id
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat.guid = ?
            {before_clause}
            {from_me_clause}
            ORDER BY message.date DESC
            {limit_clause}
        """

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching messages: {e}")
            return []

        messages = []
        for row in rows:
            text = row["text"]
            if not text:
                text = _parse_attributed_body(row["attributedBody"])

            # Skip empty messages
            if not text or not text.strip():
                continue

            # Skip reaction messages (Loved, Liked, Disliked, Laughed at, etc.)
            # Handle both regular quotes (") and smart/curly quotes (" ")
            text_lower = text.lower().strip()
            reaction_prefixes = [
                'loved "', 'liked "', 'disliked "', 'laughed at "',
                'emphasized "', 'questioned "',
                'loved \u201c', 'liked \u201c', 'disliked \u201c', 'laughed at \u201c',  # Smart quotes
                'emphasized \u201c', 'questioned \u201c',
            ]
            if any(text_lower.startswith(prefix) for prefix in reaction_prefixes):
                continue

            # Skip the Unicode object replacement character (attachment placeholders)
            if text.strip() == '\ufffc':
                continue

            # Resolve sender name from contacts
            sender = row["sender"]
            sender_name = None
            if not row["is_from_me"] and sender and sender != "me":
                normalized_sender = _normalize_phone_number(sender)
                if normalized_sender:
                    sender_name = self._resolve_contact_name(normalized_sender)
                    sender = normalized_sender

            messages.append(Message(
                id=row["id"],
                text=text,
                sender=sender,
                sender_name=sender_name,
                is_from_me=bool(row["is_from_me"]),
                timestamp=_parse_apple_timestamp(row["timestamp"]),
                chat_id=row["chat_id"],
            ))

        return messages

    def get_user_messages(self, chat_id: str, limit: int = 100) -> list[Message]:
        """Get only the user's sent messages from a conversation.

        Useful for analyzing user's texting style.

        Args:
            chat_id: Conversation ID
            limit: Maximum number of messages

        Returns:
            List of user's messages, newest first
        """
        return self.get_messages(chat_id, limit=limit, is_from_me=True)
