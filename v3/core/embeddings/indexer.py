"""Message indexer for JARVIS v2.

Indexes all iMessage history for similarity search and style learning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .store import get_embedding_store

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Statistics from indexing run."""

    conversations_processed: int
    messages_indexed: int
    messages_skipped: int
    duplicates: int
    time_seconds: float


class MessageIndexer:
    """Index all iMessage history for style learning."""

    def __init__(self):
        self.store = get_embedding_store()

    def index_all(
        self,
        max_conversations: int = 500,
        max_messages_per_convo: int = 1000,
        progress_callback: callable | None = None,
    ) -> IndexingStats:
        """Index all messages from all conversations.

        Args:
            max_conversations: Maximum conversations to process
            max_messages_per_convo: Maximum messages per conversation
            progress_callback: Optional callback(current, total, message)

        Returns:
            IndexingStats with counts and timing
        """
        from core.imessage import MessageReader

        reader = MessageReader()
        start_time = time.time()

        conversations = reader.get_conversations(limit=max_conversations)
        total_convos = len(conversations)

        if progress_callback:
            progress_callback(0, total_convos, f"Found {total_convos} conversations")

        total_indexed = 0
        total_skipped = 0
        total_duplicates = 0

        for i, conv in enumerate(conversations):
            messages = reader.get_messages(conv.chat_id, limit=max_messages_per_convo)

            # Convert to dict format for store
            msg_dicts = [
                {
                    "id": m.id,
                    "text": m.text,
                    "chat_id": m.chat_id,
                    "sender": m.sender,
                    "sender_name": m.sender_name,
                    "timestamp": m.timestamp,
                    "is_from_me": m.is_from_me,
                }
                for m in messages
            ]

            stats = self.store.index_messages(msg_dicts)
            total_indexed += stats["indexed"]
            total_skipped += stats["skipped"]
            total_duplicates += stats["duplicates"]

            if progress_callback:
                progress_callback(
                    i + 1,
                    total_convos,
                    f"Indexed {total_indexed} messages ({i + 1}/{total_convos} conversations)",
                )

        elapsed = time.time() - start_time

        return IndexingStats(
            conversations_processed=total_convos,
            messages_indexed=total_indexed,
            messages_skipped=total_skipped,
            duplicates=total_duplicates,
            time_seconds=elapsed,
        )

    def index_conversation(self, chat_id: str, max_messages: int = 1000) -> dict[str, int]:
        """Index a single conversation.

        Args:
            chat_id: Conversation to index
            max_messages: Maximum messages to index

        Returns:
            Stats dict with indexed, skipped, duplicates
        """
        from core.imessage import MessageReader

        reader = MessageReader()
        messages = reader.get_messages(chat_id, limit=max_messages)

        msg_dicts = [
            {
                "id": m.id,
                "text": m.text,
                "chat_id": m.chat_id,
                "sender": m.sender,
                "sender_name": m.sender_name,
                "timestamp": m.timestamp,
                "is_from_me": m.is_from_me,
            }
            for m in messages
        ]

        return self.store.index_messages(msg_dicts)


def run_indexing(verbose: bool = True) -> IndexingStats:
    """Run full message indexing with progress output.

    Args:
        verbose: Print progress to stdout

    Returns:
        IndexingStats with results
    """
    def progress(current: int, total: int, message: str):
        if verbose:
            pct = (current / total * 100) if total > 0 else 0
            print(f"[{pct:5.1f}%] {message}")

    if verbose:
        print("=" * 60)
        print("JARVIS v2 Message Indexer")
        print("=" * 60)
        print("This indexes your iMessage history for style learning.")
        print("Your past replies will be used to match your texting style.")
        print()

    indexer = MessageIndexer()
    stats = indexer.index_all(progress_callback=progress if verbose else None)

    if verbose:
        print()
        print("=" * 60)
        print("Indexing Complete!")
        print("=" * 60)
        print(f"Conversations processed: {stats.conversations_processed}")
        print(f"Messages indexed: {stats.messages_indexed}")
        print(f"Messages skipped (too short): {stats.messages_skipped}")
        print(f"Already indexed (skipped): {stats.duplicates}")
        print(f"Time: {stats.time_seconds:.1f} seconds")
        print()

        store_stats = get_embedding_store().get_stats()
        print(f"Total in database: {store_stats['total_messages']} messages")
        print(f"Database size: {store_stats['db_size_mb']:.1f} MB")
        print()

    return stats
