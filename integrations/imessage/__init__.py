"""iMessage chat.db integration (Workstream 10).

Provides read-only access to the macOS iMessage database.

Example:
    from integrations.imessage import ChatDBReader

    reader = ChatDBReader()
    if reader.check_access():
        conversations = reader.get_conversations(limit=10)
        for conv in conversations:
            messages = reader.get_messages(conv.chat_id)
"""

from .reader import CHAT_DB_PATH, ChatDBReader

__all__ = ["ChatDBReader", "CHAT_DB_PATH"]
