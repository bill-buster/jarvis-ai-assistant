"""SQL queries for iMessage chat.db access.

Handles schema version differences between macOS versions.

Note: All filter parameters (with_since_filter, with_before_filter) are boolean
flags that control query construction. User input is NEVER interpolated into
query strings - all user values are passed as parameterized query arguments.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)


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


# Base SQL query templates (shared between v14 and v15 until divergence needed)
_BASE_QUERIES = {
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
        {since_filter}
        {before_filter}
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
            message.thread_originator_guid as reply_to_guid,
            message.date_delivered,
            message.date_read,
            message.group_action_type,
            affected_handle.id as affected_handle_id
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID
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
        WHERE message.text LIKE ? ESCAPE '\\'
        {sender_filter}
        {after_filter}
        {before_filter}
        {chat_id_filter}
        {has_attachments_filter}
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
            message.thread_originator_guid as reply_to_guid,
            message.group_action_type,
            affected_handle.id as affected_handle_id
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        LEFT JOIN handle AS affected_handle ON message.other_handle = affected_handle.ROWID
        WHERE chat.guid = ?
        ORDER BY ABS(message.ROWID - ?)
        LIMIT ?
    """,
    "attachments": """
        SELECT
            attachment.ROWID as attachment_id,
            attachment.filename,
            attachment.mime_type,
            attachment.total_bytes as file_size,
            attachment.transfer_name
        FROM attachment
        JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
        WHERE message_attachment_join.message_id = ?
    """,
    "attachments_extended": """
        SELECT
            attachment.ROWID as attachment_id,
            attachment.filename,
            attachment.mime_type,
            attachment.total_bytes as file_size,
            attachment.transfer_name,
            attachment.width,
            attachment.height,
            attachment.uti,
            attachment.is_sticker,
            attachment.created_date
        FROM attachment
        JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
        WHERE message_attachment_join.message_id = ?
    """,
    "all_attachments": """
        SELECT
            attachment.ROWID as attachment_id,
            attachment.filename,
            attachment.mime_type,
            attachment.total_bytes as file_size,
            attachment.transfer_name,
            attachment.width,
            attachment.height,
            attachment.uti,
            attachment.is_sticker,
            attachment.created_date,
            message.ROWID as message_id,
            message.date as message_date,
            chat.guid as chat_id,
            COALESCE(handle.id, 'me') as sender,
            message.is_from_me
        FROM attachment
        JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
        JOIN message ON message_attachment_join.message_id = message.ROWID
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        {chat_filter}
        {type_filter}
        {date_after_filter}
        {date_before_filter}
        ORDER BY message.date DESC
        LIMIT ?
    """,
    "attachment_stats": """
        SELECT
            COUNT(*) as total_count,
            COALESCE(SUM(attachment.total_bytes), 0) as total_size,
            attachment.mime_type
        FROM attachment
        JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
        JOIN message ON message_attachment_join.message_id = message.ROWID
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        WHERE chat.guid = ?
        GROUP BY
            CASE
                WHEN attachment.mime_type LIKE 'image/%' THEN 'images'
                WHEN attachment.mime_type LIKE 'video/%' THEN 'videos'
                WHEN attachment.mime_type LIKE 'audio/%' THEN 'audio'
                WHEN attachment.mime_type IN ('application/pdf', 'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'text/plain', 'application/rtf') THEN 'documents'
                ELSE 'other'
            END
    """,
    "storage_by_conversation": """
        SELECT
            chat.guid as chat_id,
            chat.display_name,
            COUNT(DISTINCT attachment.ROWID) as attachment_count,
            COALESCE(SUM(attachment.total_bytes), 0) as total_size
        FROM chat
        JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
        JOIN message ON chat_message_join.message_id = message.ROWID
        JOIN message_attachment_join ON message.ROWID = message_attachment_join.message_id
        JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID
        GROUP BY chat.guid
        ORDER BY total_size DESC
        LIMIT ?
    """,
    "reactions": """
        SELECT
            message.ROWID as id,
            message.associated_message_type,
            message.date,
            message.is_from_me,
            COALESCE(handle.id, 'me') as sender
        FROM message
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE message.associated_message_guid = ?
          AND message.associated_message_type != 0
    """,
    "message_by_guid": """
        SELECT message.ROWID as id
        FROM message
        WHERE message.guid = ?
        LIMIT 1
    """,
}

# Version-specific queries (currently identical, will diverge as needed)
# Reference base queries to avoid duplication
QUERIES = {
    "v14": _BASE_QUERIES,
    "v15": _BASE_QUERIES,  # Will be replaced with v15-specific dict when needed
}


def get_query(
    name: str,
    version: str,
    *,
    with_since_filter: bool = False,
    with_before_filter: bool = False,
    with_conversations_before_filter: bool = False,
    with_sender_filter: bool = False,
    with_after_filter: bool = False,
    with_search_before_filter: bool = False,
    with_chat_id_filter: bool = False,
    with_has_attachments_filter: bool | None = None,
    # Attachment query filters
    with_attachment_chat_filter: bool = False,
    with_attachment_type_filter: str | None = None,
    with_attachment_date_after_filter: bool = False,
    with_attachment_date_before_filter: bool = False,
) -> str:
    """Get SQL query for the specified schema version.

    Args:
        name: Query name (conversations, messages, search, context)
        version: Schema version (v14, v15)
        with_since_filter: If True, include AND last_message_date > ? clause (conversations)
        with_before_filter: If True, include AND message.date < ? clause (messages)
        with_conversations_before_filter: If True, include AND last_message_date < ? (conversations)
        with_sender_filter: If True, include sender filter clause (search)
        with_after_filter: If True, include AND message.date > ? clause (search)
        with_search_before_filter: If True, include AND message.date < ? clause (search)
        with_chat_id_filter: If True, include AND chat.guid = ? clause (search)
        with_has_attachments_filter: If True/False, filter by attachment presence (search)

    Returns:
        SQL query string with filters applied

    Raises:
        KeyError: If query name not found
    """
    # Fall back to v14 if version unknown
    if version not in QUERIES:
        logger.warning(
            "Unknown schema version '%s', falling back to v14 queries. "
            "Some queries may fail or return incorrect results.",
            version,
        )
        version = "v14"

    query = QUERIES[version][name]

    # Build filter clauses from boolean flags (never from user input)
    since_filter = "AND last_message_date > ?" if with_since_filter else ""
    conversations_before_filter = (
        "AND last_message_date < ?" if with_conversations_before_filter else ""
    )

    # For messages query, use with_before_filter
    # For search query, use with_search_before_filter
    # For conversations query, use with_conversations_before_filter
    if name == "search":
        before_filter = "AND message.date < ?" if with_search_before_filter else ""
    elif name == "conversations":
        before_filter = conversations_before_filter
    else:
        before_filter = "AND message.date < ?" if with_before_filter else ""

    # Search-specific filters
    if with_sender_filter:
        sender_filter = "AND (handle.id = ? OR (message.is_from_me = 1 AND ? = 'me'))"
    else:
        sender_filter = ""
    after_filter = "AND message.date > ?" if with_after_filter else ""
    chat_id_filter = "AND chat.guid = ?" if with_chat_id_filter else ""

    # Attachment filter: True = has attachments, False = no attachments, None = no filter
    if with_has_attachments_filter is True:
        has_attachments_filter = """AND EXISTS (
            SELECT 1 FROM message_attachment_join
            WHERE message_attachment_join.message_id = message.ROWID
        )"""
    elif with_has_attachments_filter is False:
        has_attachments_filter = """AND NOT EXISTS (
            SELECT 1 FROM message_attachment_join
            WHERE message_attachment_join.message_id = message.ROWID
        )"""
    else:
        has_attachments_filter = ""

    # Attachment query-specific filters
    attachment_chat_filter = "WHERE chat.guid = ?" if with_attachment_chat_filter else "WHERE 1=1"

    # Type filter for attachments (images, videos, audio, documents)
    if with_attachment_type_filter == "images":
        attachment_type_filter = "AND attachment.mime_type LIKE 'image/%'"
    elif with_attachment_type_filter == "videos":
        attachment_type_filter = "AND attachment.mime_type LIKE 'video/%'"
    elif with_attachment_type_filter == "audio":
        attachment_type_filter = "AND attachment.mime_type LIKE 'audio/%'"
    elif with_attachment_type_filter == "documents":
        attachment_type_filter = """AND attachment.mime_type IN (
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain', 'application/rtf',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )"""
    else:
        attachment_type_filter = ""

    attachment_date_after_filter = (
        "AND message.date > ?" if with_attachment_date_after_filter else ""
    )
    attachment_date_before_filter = (
        "AND message.date < ?" if with_attachment_date_before_filter else ""
    )

    return query.format(
        since_filter=since_filter,
        before_filter=before_filter,
        sender_filter=sender_filter,
        after_filter=after_filter,
        chat_id_filter=chat_id_filter,
        has_attachments_filter=has_attachments_filter,
        chat_filter=attachment_chat_filter,
        type_filter=attachment_type_filter,
        date_after_filter=attachment_date_after_filter,
        date_before_filter=attachment_date_before_filter,
    )
