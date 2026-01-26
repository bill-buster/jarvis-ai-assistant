"""Conversation export functionality.

Provides export capabilities for iMessage conversations in various formats:
- JSON: Full message data with metadata
- CSV: Flattened fields for spreadsheet analysis
- TXT: Human-readable format for reading
"""

import csv
import io
import json
from datetime import datetime
from enum import Enum
from typing import Any

from contracts.imessage import Conversation, Message


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    TXT = "txt"


def _serialize_datetime(dt: datetime | None) -> str | None:
    """Serialize datetime to ISO format string."""
    if dt is None:
        return None
    return dt.isoformat()


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Convert a Message dataclass to a serializable dictionary.

    Args:
        message: Message to convert.

    Returns:
        Dictionary with serializable values.
    """
    return {
        "id": message.id,
        "chat_id": message.chat_id,
        "sender": message.sender,
        "sender_name": message.sender_name,
        "text": message.text,
        "date": _serialize_datetime(message.date),
        "is_from_me": message.is_from_me,
        "attachments": [
            {
                "filename": a.filename,
                "file_path": a.file_path,
                "mime_type": a.mime_type,
                "file_size": a.file_size,
            }
            for a in message.attachments
        ],
        "reply_to_id": message.reply_to_id,
        "reactions": [
            {
                "type": r.type,
                "sender": r.sender,
                "sender_name": r.sender_name,
                "date": _serialize_datetime(r.date),
            }
            for r in message.reactions
        ],
        "date_delivered": _serialize_datetime(message.date_delivered),
        "date_read": _serialize_datetime(message.date_read),
        "is_system_message": message.is_system_message,
    }


def _conversation_to_dict(conversation: Conversation) -> dict[str, Any]:
    """Convert a Conversation dataclass to a serializable dictionary.

    Args:
        conversation: Conversation to convert.

    Returns:
        Dictionary with serializable values.
    """
    return {
        "chat_id": conversation.chat_id,
        "participants": conversation.participants,
        "display_name": conversation.display_name,
        "last_message_date": _serialize_datetime(conversation.last_message_date),
        "message_count": conversation.message_count,
        "is_group": conversation.is_group,
        "last_message_text": conversation.last_message_text,
    }


def export_messages_json(
    messages: list[Message],
    conversation: Conversation | None = None,
    include_metadata: bool = True,
) -> str:
    """Export messages to JSON format.

    Args:
        messages: List of messages to export.
        conversation: Optional conversation metadata.
        include_metadata: Whether to include export metadata.

    Returns:
        JSON string of exported data.
    """
    data: dict[str, Any] = {}

    if include_metadata:
        data["export_metadata"] = {
            "format": "json",
            "exported_at": _serialize_datetime(datetime.now()),
            "message_count": len(messages),
        }

    if conversation:
        data["conversation"] = _conversation_to_dict(conversation)

    data["messages"] = [_message_to_dict(m) for m in messages]

    return json.dumps(data, indent=2, ensure_ascii=False)


def export_messages_csv(
    messages: list[Message],
    include_attachments: bool = False,
) -> str:
    """Export messages to CSV format.

    Args:
        messages: List of messages to export.
        include_attachments: Whether to include attachment info columns.

    Returns:
        CSV string of exported data.
    """
    output = io.StringIO()

    # Define CSV columns
    base_columns = [
        "id",
        "chat_id",
        "sender",
        "sender_name",
        "text",
        "date",
        "is_from_me",
        "reply_to_id",
        "is_system_message",
        "reaction_count",
    ]

    attachment_columns = [
        "attachment_count",
        "attachment_filenames",
    ]

    columns = base_columns + (attachment_columns if include_attachments else [])

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for message in messages:
        row = {
            "id": message.id,
            "chat_id": message.chat_id,
            "sender": message.sender,
            "sender_name": message.sender_name or "",
            "text": message.text.replace("\n", "\\n"),  # Escape newlines
            "date": _serialize_datetime(message.date) or "",
            "is_from_me": message.is_from_me,
            "reply_to_id": message.reply_to_id or "",
            "is_system_message": message.is_system_message,
            "reaction_count": len(message.reactions),
        }

        if include_attachments:
            row["attachment_count"] = len(message.attachments)
            row["attachment_filenames"] = "; ".join(a.filename for a in message.attachments)

        writer.writerow(row)

    return output.getvalue()


def export_messages_txt(
    messages: list[Message],
    conversation: Conversation | None = None,
    include_metadata: bool = True,
) -> str:
    """Export messages to human-readable TXT format.

    Args:
        messages: List of messages to export.
        conversation: Optional conversation metadata.
        include_metadata: Whether to include header metadata.

    Returns:
        Plain text string of exported data.
    """
    lines: list[str] = []

    # Header section
    if include_metadata:
        lines.append("=" * 60)
        lines.append("CONVERSATION EXPORT")
        lines.append("=" * 60)

        if conversation:
            display = conversation.display_name or ", ".join(conversation.participants)
            lines.append(f"Conversation: {display}")
            lines.append(f"Type: {'Group' if conversation.is_group else 'Individual'}")
            lines.append(f"Total Messages: {conversation.message_count}")

        lines.append(f"Exported Messages: {len(messages)}")
        lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if messages:
            # Find date range
            dates = [m.date for m in messages if m.date]
            if dates:
                oldest = min(dates).strftime("%Y-%m-%d")
                newest = max(dates).strftime("%Y-%m-%d")
                lines.append(f"Date Range: {oldest} to {newest}")

        lines.append("=" * 60)
        lines.append("")

    # Messages section
    for message in messages:
        # Format sender name
        if message.is_from_me:
            sender = "Me"
        else:
            sender = message.sender_name or message.sender

        # Format date
        if message.date:
            date_str = message.date.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = "Unknown date"

        # Format message line
        lines.append(f"[{date_str}] {sender}:")
        lines.append(f"  {message.text}")

        # Include attachment info
        if message.attachments:
            attachment_names = [a.filename for a in message.attachments]
            lines.append(f"  📎 Attachments: {', '.join(attachment_names)}")

        # Include reactions
        if message.reactions:
            reaction_strs = []
            for r in message.reactions:
                reactor = r.sender_name or r.sender
                reaction_strs.append(f"{reactor}: {r.type}")
            lines.append(f"  ❤️ Reactions: {', '.join(reaction_strs)}")

        lines.append("")  # Empty line between messages

    return "\n".join(lines)


def export_messages(
    messages: list[Message],
    format: ExportFormat,
    conversation: Conversation | None = None,
    include_attachments: bool = False,
) -> str:
    """Export messages in the specified format.

    Args:
        messages: List of messages to export.
        format: Export format (JSON, CSV, or TXT).
        conversation: Optional conversation metadata.
        include_attachments: Whether to include attachment details (CSV only).

    Returns:
        Exported data as string.

    Raises:
        ValueError: If format is not supported.
    """
    if format == ExportFormat.JSON:
        return export_messages_json(messages, conversation)
    elif format == ExportFormat.CSV:
        return export_messages_csv(messages, include_attachments)
    elif format == ExportFormat.TXT:
        return export_messages_txt(messages, conversation)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_search_results(
    messages: list[Message],
    query: str,
    format: ExportFormat,
) -> str:
    """Export search results in the specified format.

    Args:
        messages: List of search result messages.
        query: The search query that was used.
        format: Export format (JSON, CSV, or TXT).

    Returns:
        Exported data as string.
    """
    if format == ExportFormat.JSON:
        data = {
            "export_metadata": {
                "format": "json",
                "type": "search_results",
                "query": query,
                "exported_at": _serialize_datetime(datetime.now()),
                "result_count": len(messages),
            },
            "messages": [_message_to_dict(m) for m in messages],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    elif format == ExportFormat.CSV:
        return export_messages_csv(messages, include_attachments=True)
    elif format == ExportFormat.TXT:
        lines = [
            "=" * 60,
            "SEARCH RESULTS EXPORT",
            "=" * 60,
            f"Search Query: {query}",
            f"Results Found: {len(messages)}",
            f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]
        # Reuse TXT export logic for messages
        txt_body = export_messages_txt(messages, include_metadata=False)
        return "\n".join(lines) + txt_body
    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_backup(
    conversations: list[tuple[Conversation, list[Message]]],
    format: ExportFormat = ExportFormat.JSON,
) -> str:
    """Export a full backup of multiple conversations.

    Args:
        conversations: List of (conversation, messages) tuples.
        format: Export format (only JSON supported for full backup).

    Returns:
        Exported data as string.

    Raises:
        ValueError: If format is not JSON (other formats not supported for backup).
    """
    if format != ExportFormat.JSON:
        raise ValueError("Full backup only supports JSON format")

    data = {
        "export_metadata": {
            "format": "json",
            "type": "full_backup",
            "exported_at": _serialize_datetime(datetime.now()),
            "conversation_count": len(conversations),
            "total_message_count": sum(len(msgs) for _, msgs in conversations),
        },
        "conversations": [
            {
                "metadata": _conversation_to_dict(conv),
                "messages": [_message_to_dict(m) for m in msgs],
            }
            for conv, msgs in conversations
        ],
    }

    return json.dumps(data, indent=2, ensure_ascii=False)


def get_export_filename(
    format: ExportFormat,
    prefix: str = "export",
    chat_id: str | None = None,
) -> str:
    """Generate a filename for an export.

    Args:
        format: Export format.
        prefix: Filename prefix.
        chat_id: Optional chat ID to include in filename.

    Returns:
        Generated filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if chat_id:
        # Sanitize chat_id for filename use
        safe_chat_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)
        safe_chat_id = safe_chat_id[:30]  # Truncate if too long
        return f"{prefix}_{safe_chat_id}_{timestamp}.{format.value}"
    else:
        return f"{prefix}_{timestamp}.{format.value}"
