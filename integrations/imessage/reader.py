"""Read-only iMessage chat.db access.

Implements the iMessageReader protocol from contracts/imessage.py.
"""

import logging
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from contracts.imessage import Attachment, Conversation, Message, Reaction

from .parser import (
    datetime_to_apple_timestamp,
    extract_text_from_row,
    normalize_phone_number,
    parse_apple_timestamp,
    parse_attachments,
    parse_reactions,
)
from .queries import detect_schema_version, get_query

logger = logging.getLogger(__name__)

# Path to macOS AddressBook database for contact name resolution
ADDRESSBOOK_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"

# Default path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"


class ChatDBReader:
    """Read-only access to iMessage chat.db.

    Implements iMessageReader protocol from contracts/imessage.py.

    Thread Safety:
        This class is NOT thread-safe. Each thread should create its own
        ChatDBReader instance. The check_same_thread=False setting only
        allows the connection to be used from async contexts on the same
        thread, but does not provide synchronization for concurrent access.

    Example:
        # Preferred: use as context manager for automatic cleanup
        with ChatDBReader() as reader:
            if reader.check_access():
                conversations = reader.get_conversations(limit=10)
                for conv in conversations:
                    messages = reader.get_messages(conv.chat_id, limit=50)

        # Alternative: manual lifecycle management
        reader = ChatDBReader()
        try:
            if reader.check_access():
                conversations = reader.get_conversations(limit=10)
        finally:
            reader.close()
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the reader.

        Args:
            db_path: Path to chat.db. Defaults to ~/Library/Messages/chat.db
        """
        self.db_path = db_path or CHAT_DB_PATH
        self._connection: sqlite3.Connection | None = None
        self._schema_version: str | None = None
        # Cache for contact name lookups (phone/email -> name)
        self._contacts_cache: dict[str, str] | None = None
        # Cache for GUID to ROWID mappings
        self._guid_to_rowid_cache: dict[str, int] = {}

    def _get_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection.

        Uses connection pooling - same connection is reused across calls.
        Connection is opened in read-only mode for safety.

        Returns:
            SQLite connection with Row factory

        Raises:
            sqlite3.Error: If connection fails
        """
        if self._connection is None:
            # Open in read-only mode using URI
            uri = f"file:{self.db_path}?mode=ro"
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=5.0,  # Handle SQLITE_BUSY from iMessage app
                check_same_thread=False,  # Safe for async contexts
            )
            self._connection.row_factory = sqlite3.Row

            # Detect schema version
            self._schema_version = detect_schema_version(self._connection)
            logger.debug(f"Detected chat.db schema version: {self._schema_version}")

        return self._connection

    def close(self) -> None:
        """Close the database connection.

        Call this when done to release resources.
        Connection will be recreated on next use if needed.
        """
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass  # Suppress errors during cleanup
            self._connection = None
            self._schema_version = None
            self._contacts_cache = None
            self._guid_to_rowid_cache = {}

    def _get_attachments_for_message(self, message_id: int) -> list[Attachment]:
        """Fetch attachments for a specific message.

        Args:
            message_id: The message ROWID

        Returns:
            List of Attachment objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = get_query("attachments", self._schema_version or "v14")

        try:
            cursor.execute(query, (message_id,))
            rows = cursor.fetchall()
            return parse_attachments([dict(row) for row in rows])
        except sqlite3.OperationalError as e:
            logger.debug(f"Error fetching attachments for message {message_id}: {e}")
            return []

    def _get_reactions_for_message(self, message_guid: str) -> list[Reaction]:
        """Fetch reactions for a specific message.

        Args:
            message_guid: The message GUID (used for associated_message_guid lookups)

        Returns:
            List of Reaction objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = get_query("reactions", self._schema_version or "v14")

        try:
            cursor.execute(query, (message_guid,))
            rows = cursor.fetchall()
            reactions = parse_reactions([dict(row) for row in rows])

            # Resolve sender names for reactions
            for reaction in reactions:
                if reaction.sender != "me":
                    reaction.sender_name = self._resolve_contact_name(reaction.sender)

            return reactions
        except sqlite3.OperationalError as e:
            logger.debug(f"Error fetching reactions for message {message_guid}: {e}")
            return []

    def _get_message_rowid_by_guid(self, guid: str) -> int | None:
        """Get message ROWID from GUID.

        Args:
            guid: The message GUID

        Returns:
            Message ROWID, or None if not found
        """
        # Check cache first
        if guid in self._guid_to_rowid_cache:
            return self._guid_to_rowid_cache[guid]

        conn = self._get_connection()
        cursor = conn.cursor()

        query = get_query("message_by_guid", self._schema_version or "v14")

        try:
            cursor.execute(query, (guid,))
            row = cursor.fetchone()
            if row:
                rowid: int = int(row["id"])
                self._guid_to_rowid_cache[guid] = rowid
                return rowid
            return None
        except sqlite3.OperationalError as e:
            logger.debug(f"Error fetching message by GUID {guid}: {e}")
            return None

    def _resolve_contact_name(self, identifier: str) -> str | None:
        """Resolve a phone number or email to a contact name.

        Queries the macOS AddressBook database if available.

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
        normalized = normalize_phone_number(identifier)
        # Cache is guaranteed to be initialized after _load_contacts_cache
        if self._contacts_cache is None:
            return None
        return self._contacts_cache.get(normalized)

    def _load_contacts_cache(self) -> None:
        """Load contacts from AddressBook database into cache.

        This attempts to read from the macOS AddressBook SQLite databases.
        If unavailable (no Full Disk Access or different OS), cache remains empty.
        """
        self._contacts_cache = {}

        # Try to find AddressBook database
        if not ADDRESSBOOK_DB_PATH.exists():
            logger.debug("AddressBook path not found, contacts resolution disabled")
            return

        try:
            # Find the first AddressBook source database
            for source_dir in ADDRESSBOOK_DB_PATH.iterdir():
                ab_db = source_dir / "AddressBook-v22.abcddb"
                if ab_db.exists():
                    self._load_contacts_from_db(ab_db)
                    return
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access AddressBook: {e}")

    def _load_contacts_from_db(self, db_path: Path) -> None:
        """Load contacts from a specific AddressBook database.

        Args:
            db_path: Path to the AddressBook database
        """
        # Ensure cache is initialized
        if self._contacts_cache is None:
            self._contacts_cache = {}

        cache = self._contacts_cache

        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row

            # Query for phone numbers and emails with associated names
            # AddressBook schema: ZABCDRECORD has names, ZABCDPHONENUMBER has phones,
            # ZABCDEMAILADDRESS has emails
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
                    identifier = normalize_phone_number(row["identifier"])
                    name = self._format_name(row["first_name"], row["last_name"])
                    if identifier and name:
                        cache[identifier] = name
            except sqlite3.OperationalError:
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
                        cache[identifier.lower()] = name
            except sqlite3.OperationalError:
                pass  # Table structure may differ

            conn.close()
            logger.debug(f"Loaded {len(cache)} contacts from AddressBook")

        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error loading contacts: {e}")

    @staticmethod
    def _format_name(first_name: str | None, last_name: str | None) -> str | None:
        """Format first and last name into a display name.

        Args:
            first_name: First name or None
            last_name: Last name or None

        Returns:
            Formatted name, or None if both are empty
        """
        parts = []
        if first_name:
            parts.append(first_name)
        if last_name:
            parts.append(last_name)
        return " ".join(parts) if parts else None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and close connection."""
        self.close()

    def check_access(self) -> bool:
        """Check if we have permission to read chat.db.

        Tests Full Disk Access by attempting to open the database file.

        Returns:
            True if access is granted, False otherwise
        """
        if not self.db_path.exists():
            logger.warning(f"chat.db not found at {self.db_path}")
            return False

        try:
            # Try to open the database
            conn = self._get_connection()
            # Execute a simple query to verify read access
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM chat LIMIT 1")
            cursor.fetchone()
            return True
        except PermissionError:
            logger.warning(
                f"Permission denied for {self.db_path}. "
                "Grant Full Disk Access to Terminal/IDE in System Settings."
            )
            return False
        except sqlite3.OperationalError as e:
            if "unable to open database" in str(e).lower():
                logger.warning(f"Cannot open {self.db_path}. Ensure Full Disk Access is granted.")
            else:
                logger.warning(f"Database error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking access: {e}")
            return False

    def get_conversations(
        self,
        limit: int = 50,
        since: datetime | None = None,
    ) -> list[Conversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return
            since: Only return conversations with messages after this date

        Returns:
            List of Conversation objects, sorted by last message date (newest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build params list based on filters
        params: list[Any] = []
        if since is not None:
            params.append(datetime_to_apple_timestamp(since))
        params.append(limit)

        query = get_query(
            "conversations",
            self._schema_version or "v14",
            with_since_filter=since is not None,
        )

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Query error in get_conversations: {e}")
            return []

        conversations = []
        for row in rows:
            # Parse participants
            participants_str = row["participants"] or ""
            participants = [
                normalize_phone_number(p.strip()) for p in participants_str.split(",") if p.strip()
            ]

            # Determine if group chat
            is_group = len(participants) > 1

            # Parse last message date
            last_message_date = parse_apple_timestamp(row["last_message_date"])

            conversations.append(
                Conversation(
                    chat_id=row["chat_id"],
                    participants=participants,
                    display_name=row["display_name"] or None,
                    last_message_date=last_message_date,
                    message_count=row["message_count"],
                    is_group=is_group,
                )
            )

        return conversations

    def get_messages(
        self,
        chat_id: str,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[Message]:
        """Get messages from a conversation.

        Args:
            chat_id: The conversation ID (chat.guid)
            limit: Maximum number of messages to return
            before: Only return messages before this date

        Returns:
            List of Message objects, sorted by date (newest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build params list based on filters
        params: list[Any] = [chat_id]
        if before is not None:
            params.append(datetime_to_apple_timestamp(before))
        params.append(limit)

        query = get_query(
            "messages",
            self._schema_version or "v14",
            with_before_filter=before is not None,
        )

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Query error in get_messages: {e}")
            return []

        return self._rows_to_messages(rows, chat_id)

    def search(
        self,
        query: str,
        limit: int = 50,
        sender: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        chat_id: str | None = None,
        has_attachments: bool | None = None,
    ) -> list[Message]:
        """Full-text search across messages with optional filters.

        Args:
            query: Search query string
            limit: Maximum number of results
            sender: Filter by sender phone number or email (use "me" for own messages)
            after: Filter for messages after this datetime
            before: Filter for messages before this datetime
            chat_id: Filter by conversation ID (chat.guid)
            has_attachments: Filter for messages with (True) or without (False) attachments

        Returns:
            List of Message objects matching the query and filters
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Escape LIKE wildcards (query uses ESCAPE '\\' clause)
        escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"%{escaped_query}%"

        # Build params list based on filters
        # Order: like_pattern, [sender, sender], [after], [before], [chat_id], limit
        params: list[Any] = [like_pattern]

        # Normalize sender if provided
        normalized_sender: str | None = None
        if sender is not None:
            normalized_sender = normalize_phone_number(sender) if sender != "me" else "me"
            # Add sender param twice for the OR condition in the query
            params.append(normalized_sender)
            params.append(normalized_sender)

        if after is not None:
            params.append(datetime_to_apple_timestamp(after))

        if before is not None:
            params.append(datetime_to_apple_timestamp(before))

        if chat_id is not None:
            params.append(chat_id)

        params.append(limit)

        sql = get_query(
            "search",
            self._schema_version or "v14",
            with_sender_filter=sender is not None,
            with_after_filter=after is not None,
            with_search_before_filter=before is not None,
            with_chat_id_filter=chat_id is not None,
            with_has_attachments_filter=has_attachments,
        )

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Query error in search: {e}")
            return []

        messages = []
        for row in rows:
            msg = self._row_to_message(row, row["chat_id"])
            if msg:
                messages.append(msg)

        return messages

    def get_conversation_context(
        self,
        chat_id: str,
        around_message_id: int,
        context_messages: int = 5,
    ) -> list[Message]:
        """Get messages around a specific message for context.

        Args:
            chat_id: The conversation ID
            around_message_id: The message ROWID to center around
            context_messages: Number of messages before and after

        Returns:
            List of Message objects around the target message
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get 2*context + 1 messages centered on the target
        total_limit = context_messages * 2 + 1

        query = get_query("context", self._schema_version or "v14")

        try:
            cursor.execute(query, (chat_id, around_message_id, total_limit))
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Query error in get_conversation_context: {e}")
            return []

        messages = self._rows_to_messages(rows, chat_id)

        # Sort by date for proper ordering
        messages.sort(key=lambda m: m.date)

        return messages

    def _rows_to_messages(self, rows: Sequence[sqlite3.Row], chat_id: str) -> list[Message]:
        """Convert database rows to Message objects.

        Args:
            rows: Sequence of sqlite3.Row objects
            chat_id: The conversation ID

        Returns:
            List of Message objects
        """
        messages = []
        for row in rows:
            msg = self._row_to_message(row, chat_id)
            if msg:
                messages.append(msg)
        return messages

    def _row_to_message(self, row: sqlite3.Row, chat_id: str) -> Message | None:
        """Convert a database row to a Message object.

        Args:
            row: sqlite3.Row object
            chat_id: The conversation ID

        Returns:
            Message object, or None if text extraction fails
        """
        # Extract text (tries text column, falls back to attributedBody)
        text = extract_text_from_row(dict(row))
        if not text:
            return None

        message_id = row["id"]

        # Get sender and resolve name from contacts
        sender = normalize_phone_number(row["sender"])
        sender_name = None
        if not row["is_from_me"]:
            sender_name = self._resolve_contact_name(sender)

        # Parse reply_to_id from thread_originator_guid
        reply_to_id = None
        row_dict = dict(row)
        reply_to_guid = row_dict.get("reply_to_guid")
        if reply_to_guid:
            reply_to_id = self._get_message_rowid_by_guid(reply_to_guid)

        # Fetch attachments for this message
        attachments = self._get_attachments_for_message(message_id)

        # Fetch reactions for this message
        # Build the message GUID for reaction lookup (format: p:N/GUID or similar)
        # We need to query the actual GUID from the database for this message
        reactions = self._get_reactions_for_message_id(message_id)

        return Message(
            id=message_id,
            chat_id=chat_id,
            sender=sender,
            sender_name=sender_name,
            text=text,
            date=parse_apple_timestamp(row["date"]),
            is_from_me=bool(row["is_from_me"]),
            attachments=attachments,
            reply_to_id=reply_to_id,
            reactions=reactions,
        )

    def _get_reactions_for_message_id(self, message_id: int) -> list[Reaction]:
        """Fetch reactions for a message by its ROWID.

        First looks up the message GUID, then queries for associated reactions.

        Args:
            message_id: The message ROWID

        Returns:
            List of Reaction objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # First get the message GUID
        try:
            cursor.execute("SELECT guid FROM message WHERE ROWID = ?", (message_id,))
            row = cursor.fetchone()
            if not row or not row["guid"]:
                return []

            message_guid = row["guid"]
            return self._get_reactions_for_message(message_guid)
        except sqlite3.OperationalError as e:
            logger.debug(f"Error fetching GUID for message {message_id}: {e}")
            return []
