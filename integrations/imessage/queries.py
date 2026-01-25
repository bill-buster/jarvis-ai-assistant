"""SQL queries for iMessage chat.db access.

Handles schema version differences between macOS versions.
"""

import sqlite3


def detect_schema_version(conn: sqlite3.Connection) -> str:
    """Detect chat.db schema version based on available columns.

    Args:
        conn: SQLite connection to chat.db

    Returns:
        Schema version string ("v14", "v15", or "unknown")
    """
    cursor = conn.cursor()

    # Check for columns that indicate specific macOS versions
    try:
        cursor.execute("PRAGMA table_info(message)")
        columns = {row[1] for row in cursor.fetchall()}

        # macOS 15+ has thread_originator_guid
        if "thread_originator_guid" in columns:
            # Check for newer macOS 15 specific columns
            cursor.execute("PRAGMA table_info(chat)")
            chat_columns = {row[1] for row in cursor.fetchall()}
            if "service_name" in chat_columns:
                return "v15"
            return "v14"

        # Older schema without thread_originator_guid
        return "v14"

    except sqlite3.Error:
        return "unknown"


# SQL query templates for different schema versions
QUERIES = {
    "v14": {
        "conversations": """
            SELECT
                chat.ROWID as chat_rowid,
                chat.guid as chat_id,
                chat.display_name,
                chat.chat_identifier,
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
                ) as last_message_date
            FROM chat
            WHERE message_count > 0
            {since_filter}
            ORDER BY last_message_date DESC
            LIMIT ?
        """,
        "messages": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                CASE
                    WHEN message.text IS NOT NULL AND message.text != ''
                    THEN message.text
                    ELSE NULL
                END as text,
                message.attributedBody,
                message.date as date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat.guid = ?
            {before_filter}
            ORDER BY message.date DESC
            LIMIT ?
        """,
        "search": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                message.text,
                message.attributedBody,
                message.date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE message.text LIKE ?
            ORDER BY message.date DESC
            LIMIT ?
        """,
        "context": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                message.text,
                message.attributedBody,
                message.date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat.guid = ?
            ORDER BY ABS(message.ROWID - ?)
            LIMIT ?
        """,
    },
    "v15": {
        # macOS 15 queries - same structure for now, can be updated
        # when specific v15 differences are identified
        "conversations": """
            SELECT
                chat.ROWID as chat_rowid,
                chat.guid as chat_id,
                chat.display_name,
                chat.chat_identifier,
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
                ) as last_message_date
            FROM chat
            WHERE message_count > 0
            {since_filter}
            ORDER BY last_message_date DESC
            LIMIT ?
        """,
        "messages": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                CASE
                    WHEN message.text IS NOT NULL AND message.text != ''
                    THEN message.text
                    ELSE NULL
                END as text,
                message.attributedBody,
                message.date as date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat.guid = ?
            {before_filter}
            ORDER BY message.date DESC
            LIMIT ?
        """,
        "search": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                message.text,
                message.attributedBody,
                message.date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE message.text LIKE ?
            ORDER BY message.date DESC
            LIMIT ?
        """,
        "context": """
            SELECT
                message.ROWID as id,
                chat.guid as chat_id,
                COALESCE(handle.id, 'me') as sender,
                message.text,
                message.attributedBody,
                message.date,
                message.is_from_me,
                message.thread_originator_guid as reply_to_guid
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat.guid = ?
            ORDER BY ABS(message.ROWID - ?)
            LIMIT ?
        """,
    },
}


def get_query(name: str, version: str, **kwargs: str) -> str:
    """Get SQL query for the specified schema version.

    Args:
        name: Query name (conversations, messages, search, context)
        version: Schema version (v14, v15)
        **kwargs: Format parameters (since_filter, before_filter)

    Returns:
        SQL query string with filters applied

    Raises:
        KeyError: If query name or version not found
    """
    # Fall back to v14 if version unknown
    if version not in QUERIES:
        version = "v14"

    query = QUERIES[version][name]

    # Apply default empty filters
    defaults = {
        "since_filter": "",
        "before_filter": "",
    }
    defaults.update(kwargs)

    return query.format(**defaults)
