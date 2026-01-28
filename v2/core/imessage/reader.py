"""Simplified iMessage reader for JARVIS v2.

Read-only access to macOS iMessage chat.db database.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default iMessage database path
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Apple's timestamp epoch (2001-01-01)
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


@dataclass
class Message:
    """A single iMessage."""

    id: int
    text: str
    sender: str
    is_from_me: bool
    timestamp: datetime
    chat_id: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "sender": self.sender,
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
    message_count: int = 0
    is_group: bool = False

    def to_dict(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "display_name": self.display_name,
            "participants": self.participants,
            "last_message_date": self.last_message_date.isoformat() if self.last_message_date else None,
            "last_message_text": self.last_message_text,
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


def _parse_attributed_body(data: bytes | None) -> str | None:
    """Extract plain text from attributedBody blob."""
    if not data:
        return None
    try:
        # The text is usually between "NSString" markers
        # This is a simplified parser - the full one uses plist
        text = data.decode("utf-8", errors="ignore")
        # Find text between common markers
        if "NSString" in text:
            # Try to extract readable text
            import re
            matches = re.findall(r'[\x20-\x7E]{2,}', text)
            if matches:
                # Filter out common metadata strings
                filtered = [m for m in matches if not m.startswith("NS") and len(m) > 3]
                if filtered:
                    return filtered[0]
        return None
    except Exception:
        return None


class MessageReader:
    """Read-only access to iMessage database."""

    def __init__(self, db_path: Path | None = None):
        """Initialize reader.

        Args:
            db_path: Path to chat.db, defaults to ~/Library/Messages/chat.db
        """
        self.db_path = db_path or CHAT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._schema_version: str = "v14"

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
                self._conn = sqlite3.connect(uri, uri=True, timeout=5.0)
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

    def check_access(self) -> bool:
        """Check if database is accessible."""
        try:
            self._get_connection()
            return True
        except (FileNotFoundError, PermissionError, sqlite3.Error):
            return False

    def get_conversations(self, limit: int = 50) -> list[Conversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversations, newest first
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                chat.guid as chat_id,
                chat.display_name,
                (
                    SELECT GROUP_CONCAT(handle.id, ', ')
                    FROM chat_handle_join
                    JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                    WHERE chat_handle_join.chat_id = chat.ROWID
                ) as participants,
                (
                    SELECT COUNT(*)
                    FROM chat_message_join
                    WHERE chat_message_join.chat_id = chat.ROWID
                ) as message_count,
                (
                    SELECT MAX(message.date)
                    FROM chat_message_join
                    JOIN message ON chat_message_join.message_id = message.ROWID
                    WHERE chat_message_join.chat_id = chat.ROWID
                ) as last_message_date,
                (
                    SELECT message.text
                    FROM chat_message_join
                    JOIN message ON chat_message_join.message_id = message.ROWID
                    WHERE chat_message_join.chat_id = chat.ROWID
                    ORDER BY message.date DESC
                    LIMIT 1
                ) as last_message_text,
                (
                    SELECT message.attributedBody
                    FROM chat_message_join
                    JOIN message ON chat_message_join.message_id = message.ROWID
                    WHERE chat_message_join.chat_id = chat.ROWID
                    ORDER BY message.date DESC
                    LIMIT 1
                ) as last_message_attributed_body
            FROM chat
            WHERE message_count > 0
            ORDER BY last_message_date DESC
            LIMIT ?
        """

        try:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching conversations: {e}")
            return []

        conversations = []
        for row in rows:
            participants_str = row["participants"] or ""
            participants = [p.strip() for p in participants_str.split(",") if p.strip()]
            is_group = len(participants) > 1

            last_message_text = row["last_message_text"]
            if not last_message_text:
                last_message_text = _parse_attributed_body(row["last_message_attributed_body"])

            conversations.append(Conversation(
                chat_id=row["chat_id"],
                display_name=row["display_name"],
                participants=participants,
                last_message_date=_parse_apple_timestamp(row["last_message_date"]),
                last_message_text=last_message_text,
                message_count=row["message_count"],
                is_group=is_group,
            ))

        return conversations

    def get_messages(self, chat_id: str, limit: int = 50) -> list[Message]:
        """Get messages from a conversation.

        Args:
            chat_id: Conversation ID
            limit: Maximum number of messages

        Returns:
            List of messages, newest first
        """
        if not chat_id:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
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
            ORDER BY message.date DESC
            LIMIT ?
        """

        try:
            cursor.execute(query, (chat_id, limit))
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching messages: {e}")
            return []

        messages = []
        for row in rows:
            text = row["text"]
            if not text:
                text = _parse_attributed_body(row["attributedBody"])

            messages.append(Message(
                id=row["id"],
                text=text or "",
                sender=row["sender"],
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
        messages = self.get_messages(chat_id, limit=limit * 2)  # Fetch more to filter
        user_messages = [m for m in messages if m.is_from_me]
        return user_messages[:limit]
