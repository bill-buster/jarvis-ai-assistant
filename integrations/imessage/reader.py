"""Read-only iMessage chat.db access.

Implements the iMessageReader protocol from contracts/imessage.py.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.imessage import Conversation, Message

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

# Default path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"


class ChatDBReader:
    """Read-only access to iMessage chat.db.

    Implements iMessageReader protocol from contracts/imessage.py.

    Example:
        reader = ChatDBReader()
        if reader.check_access():
            conversations = reader.get_conversations(limit=10)
            for conv in conversations:
                messages = reader.get_messages(conv.chat_id, limit=50)
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the reader.

        Args:
            db_path: Path to chat.db. Defaults to ~/Library/Messages/chat.db
        """
        self.db_path = db_path or CHAT_DB_PATH
        self._connection: sqlite3.Connection | None = None
        self._schema_version: str | None = None

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
            self._connection.close()
            self._connection = None
            self._schema_version = None

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection."""
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
                logger.warning(
                    f"Cannot open {self.db_path}. "
                    "Ensure Full Disk Access is granted."
                )
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

        # Build since filter if needed
        since_filter = ""
        params: list[Any] = []
        if since is not None:
            # Convert to Apple timestamp format
            since_filter = "AND last_message_date > ?"
            params.append(datetime_to_apple_timestamp(since))

        params.append(limit)

        query = get_query(
            "conversations",
            self._schema_version or "v14",
            since_filter=since_filter,
        )

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.error(f"Query error in get_conversations: {e}")
            return []

        conversations = []
        for row in rows:
            # Parse participants
            participants_str = row["participants"] or ""
            participants = [
                normalize_phone_number(p.strip())
                for p in participants_str.split(",")
                if p.strip()
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

        # Build before filter if needed
        before_filter = ""
        params: list[Any] = [chat_id]
        if before is not None:
            before_filter = "AND message.date < ?"
            params.append(datetime_to_apple_timestamp(before))

        params.append(limit)

        query = get_query(
            "messages",
            self._schema_version or "v14",
            before_filter=before_filter,
        )

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.error(f"Query error in get_messages: {e}")
            return []

        return self._rows_to_messages(rows, chat_id)

    def search(self, query: str, limit: int = 50) -> list[Message]:
        """Full-text search across messages.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of Message objects matching the query
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Escape and prepare LIKE pattern
        escaped_query = query.replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"%{escaped_query}%"

        sql = get_query("search", self._schema_version or "v14")

        try:
            cursor.execute(sql, (like_pattern, limit))
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.error(f"Query error in search: {e}")
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
            logger.error(f"Query error in get_conversation_context: {e}")
            return []

        messages = self._rows_to_messages(rows, chat_id)

        # Sort by date for proper ordering
        messages.sort(key=lambda m: m.date)

        return messages

    def _rows_to_messages(self, rows: list[sqlite3.Row], chat_id: str) -> list[Message]:
        """Convert database rows to Message objects.

        Args:
            rows: List of sqlite3.Row objects
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

        # Parse reply_to_id from thread_originator_guid
        # TODO: Map GUID to message ROWID for proper reply_to_id
        reply_to_id = None

        return Message(
            id=row["id"],
            chat_id=chat_id,
            sender=normalize_phone_number(row["sender"]),
            sender_name=None,  # TODO: Wire up Contacts resolution (WS10.1)
            text=text,
            date=parse_apple_timestamp(row["date"]),
            is_from_me=bool(row["is_from_me"]),
            attachments=parse_attachments(None),  # TODO: Implement (WS10.1)
            reply_to_id=reply_to_id,
            reactions=parse_reactions(None),  # TODO: Implement (WS10.1)
        )
