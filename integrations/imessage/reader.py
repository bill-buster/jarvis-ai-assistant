"""Read-only iMessage chat.db access.

Implements the iMessageReader protocol from contracts/imessage.py.
"""

import logging
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Self, TypeVar

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

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple LRU cache with bounded size.

    Uses OrderedDict to maintain insertion/access order.
    Oldest entries are evicted when maxsize is exceeded.
    Thread-safe via internal lock.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        """Initialize the LRU cache.

        Args:
            maxsize: Maximum number of items to store (default 1000)
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Get a value from the cache, returning None if not found.

        Moves the accessed item to the end (most recently used).
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: K, value: V) -> None:
        """Set a value in the cache, evicting oldest if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)


# Path to macOS AddressBook database for contact name resolution
ADDRESSBOOK_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"

# Default path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Database connection timeout (handles SQLITE_BUSY from concurrent iMessage app access)
DB_TIMEOUT_SECONDS = 5.0


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
        # LRU cache for GUID to ROWID mappings (bounded to prevent memory leaks)
        self._guid_to_rowid_cache: LRUCache[str, int] = LRUCache(maxsize=10000)

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
                timeout=DB_TIMEOUT_SECONDS,
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
            self._guid_to_rowid_cache.clear()

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
        # Check cache first (LRU cache with bounded size)
        cached = self._guid_to_rowid_cache.get(guid)
        if cached is not None:
            return cached

        conn = self._get_connection()
        cursor = conn.cursor()

        query = get_query("message_by_guid", self._schema_version or "v14")

        try:
            cursor.execute(query, (guid,))
            row = cursor.fetchone()
            if row:
                rowid: int = int(row["id"])
                self._guid_to_rowid_cache.set(guid, rowid)
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
        if normalized is None:
            return None
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
        except Exception as e:
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
            conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
            try:
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

                logger.debug(f"Loaded {len(cache)} contacts from AddressBook")
            finally:
                conn.close()

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
        before: datetime | None = None,
    ) -> list[Conversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return
            since: Only return conversations with messages after this date
            before: Only return conversations with last message before this date (for pagination)

        Returns:
            List of Conversation objects, sorted by last message date (newest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build params list based on filters
        params: list[Any] = []
        if since is not None:
            params.append(datetime_to_apple_timestamp(since))
        if before is not None:
            params.append(datetime_to_apple_timestamp(before))
        params.append(limit)

        query = get_query(
            "conversations",
            self._schema_version or "v14",
            with_since_filter=since is not None,
            with_conversations_before_filter=before is not None,
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
                normalized
                for p in participants_str.split(",")
                if p.strip() and (normalized := normalize_phone_number(p.strip())) is not None
            ]

            # Determine if group chat
            is_group = len(participants) > 1

            # Parse last message date
            last_message_date = parse_apple_timestamp(row["last_message_date"])

            # Resolve display name from database or contacts
            display_name = row["display_name"] or None

            # For individual chats without a display name, try to resolve from contacts
            if not display_name and not is_group and len(participants) == 1:
                display_name = self._resolve_contact_name(participants[0])

            # Get last message text (may be None if no text messages)
            last_message_text = (
                row["last_message_text"] if "last_message_text" in row.keys() else None
            )

            conversations.append(
                Conversation(
                    chat_id=row["chat_id"],
                    participants=participants,
                    display_name=display_name,
                    last_message_date=last_message_date,
                    message_count=row["message_count"],
                    is_group=is_group,
                    last_message_text=last_message_text,
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
            # If query fails due to missing columns, create a minimal fallback query
            # that replaces all optional columns with NULL
            logger.debug(f"Query failed ({e}), trying fallback with NULL for optional columns")
            fallback_query = query

            # Replace all optional columns with NULL placeholders
            # Group event columns
            fallback_query = (
                fallback_query.replace(
                    "message.group_action_type,",
                    "NULL as group_action_type,",
                )
                .replace(
                    "affected_handle.id as affected_handle_id",
                    "NULL as affected_handle_id",
                )
                .replace(
                    "LEFT JOIN handle AS affected_handle "
                    "ON message.other_handle = affected_handle.ROWID",
                    "",
                )
            )

            # Read receipt columns
            fallback_query = fallback_query.replace(
                "message.date_delivered,",
                "NULL as date_delivered,",
            ).replace(
                "message.date_read,",
                "NULL as date_read,",
            )

            try:
                cursor.execute(fallback_query, params)
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e2:
                logger.warning(f"Query error in get_messages: {e2}")
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
            normalized_sender = sender if sender == "me" else normalize_phone_number(sender)
            # If normalization returned None (invalid input), skip the filter
            if normalized_sender is not None:
                # Add sender param twice for the OR condition in the query
                params.append(normalized_sender)
                params.append(normalized_sender)
            else:
                # Treat as no sender filter if normalization failed
                sender = None

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
            # If query fails due to missing columns, create a minimal fallback query
            logger.debug(f"Context query failed ({e}), trying fallback")
            fallback_query = query

            # Replace all optional columns with NULL placeholders
            fallback_query = (
                fallback_query.replace(
                    "message.group_action_type,",
                    "NULL as group_action_type,",
                )
                .replace(
                    "affected_handle.id as affected_handle_id",
                    "NULL as affected_handle_id",
                )
                .replace(
                    "LEFT JOIN handle AS affected_handle "
                    "ON message.other_handle = affected_handle.ROWID",
                    "",
                )
            )

            try:
                cursor.execute(fallback_query, (chat_id, around_message_id, total_limit))
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e2:
                logger.warning(f"Query error in get_conversation_context: {e2}")
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
            Message object, or None if message has no text and no attachments
        """
        message_id = row["id"]
        row_dict = dict(row)

        # Get sender and resolve name from contacts
        sender = normalize_phone_number(row["sender"]) or row["sender"]
        sender_name = None
        if not row["is_from_me"] and sender:
            sender_name = self._resolve_contact_name(sender)

        # Check for group events (system messages)
        group_action_type = row_dict.get("group_action_type", 0) or 0
        is_system_message = group_action_type != 0

        if is_system_message:
            text = self._generate_group_event_text(
                group_action_type,
                sender,
                sender_name,
                row_dict.get("affected_handle_id"),
                bool(row["is_from_me"]),
            )
            # System messages have no attachments or reactions
            return Message(
                id=message_id,
                chat_id=chat_id,
                sender=sender,
                sender_name=sender_name,
                text=text,
                date=parse_apple_timestamp(row["date"]),
                is_from_me=bool(row["is_from_me"]),
                attachments=[],
                reply_to_id=None,
                reactions=[],
                is_system_message=True,
            )

        # Extract text (tries text column, falls back to attributedBody)
        text = extract_text_from_row(row_dict) or ""

        # Fetch attachments for this message
        attachments = self._get_attachments_for_message(message_id)

        # Skip messages with no text AND no attachments (and not a system message)
        if not text and not attachments:
            return None

        # Parse reply_to_id from thread_originator_guid
        reply_to_id = None
        reply_to_guid = row_dict.get("reply_to_guid")
        if reply_to_guid:
            reply_to_id = self._get_message_rowid_by_guid(reply_to_guid)

        # Fetch reactions for this message
        # Build the message GUID for reaction lookup (format: p:N/GUID or similar)
        # We need to query the actual GUID from the database for this message
        reactions = self._get_reactions_for_message_id(message_id)

        # Parse delivery/read receipts (only meaningful for messages you sent)
        # These columns may not exist in all databases/test fixtures
        date_delivered = None
        date_read = None
        if row["is_from_me"]:
            if row_dict.get("date_delivered"):
                date_delivered = parse_apple_timestamp(row_dict["date_delivered"])
            if row_dict.get("date_read"):
                date_read = parse_apple_timestamp(row_dict["date_read"])

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
            date_delivered=date_delivered,
            date_read=date_read,
            is_system_message=False,
        )

    def _generate_group_event_text(
        self,
        action_type: int,
        actor: str,
        actor_name: str | None,
        affected_handle_id: str | None,
        is_from_me: bool,
    ) -> str:
        """Generate human-readable text for group events.

        Args:
            action_type: The group_action_type from the database
                1 = left/removed, 2 = name changed, 3 = joined/added
            actor: The handle ID of who performed the action
            actor_name: Resolved contact name for actor
            affected_handle_id: The handle ID of who was affected (if different)
            is_from_me: Whether the current user performed the action

        Returns:
            Human-readable description of the group event
        """
        # Get display names
        if is_from_me:
            actor_display = "You"
        else:
            actor_display = actor_name or actor

        affected_display = None
        if affected_handle_id:
            normalized = normalize_phone_number(affected_handle_id) or affected_handle_id
            affected_display = self._resolve_contact_name(normalized) or normalized

        # Generate text based on action type
        if action_type == 1:  # Left or removed
            if affected_display and affected_display != actor_display:
                return f"{actor_display} removed {affected_display} from the group"
            else:
                return f"{actor_display} left the group"
        elif action_type == 2:  # Name changed
            return f"{actor_display} changed the group name"
        elif action_type == 3:  # Joined or added
            if affected_display and affected_display != actor_display:
                return f"{actor_display} added {affected_display} to the group"
            else:
                return f"{actor_display} joined the group"
        else:
            return f"Group event (type {action_type})"

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
