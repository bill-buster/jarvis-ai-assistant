"""Context fetcher for iMessage conversations.

Provides structured context for RAG-based reply and summary generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from contracts.imessage import Conversation, Message

if TYPE_CHECKING:
    from contracts.imessage import iMessageReader

logger = logging.getLogger(__name__)


@dataclass
class ReplyContext:
    """Context for generating a reply to a conversation."""

    messages: list[Message]
    last_received_message: Message | None
    formatted_context: str
    conversation: Conversation | None = None


@dataclass
class SummaryContext:
    """Context for generating a conversation summary."""

    messages: list[Message]
    formatted_context: str
    date_range: tuple[datetime, datetime]
    participant_names: list[str]
    conversation: Conversation | None = None


class ContextFetcher:
    """Fetches and formats iMessage context for LLM prompts.

    Works with the iMessageReader protocol to retrieve messages
    and format them as context for generation.
    """

    def __init__(self, reader: iMessageReader) -> None:
        """Initialize the context fetcher.

        Args:
            reader: An iMessageReader implementation (e.g., ChatDBReader).
        """
        self.reader = reader

    def find_conversation_by_name(self, name: str) -> str | None:
        """Find a conversation by participant name.

        Searches through recent conversations for a matching name.

        Args:
            name: Name or partial name to search for.

        Returns:
            chat_id if found, None otherwise.
        """
        name_lower = name.lower()

        # Get recent conversations to search through
        conversations = self.reader.get_conversations(limit=100)

        for conv in conversations:
            # Check display name
            if conv.display_name and name_lower in conv.display_name.lower():
                logger.debug("Found conversation by display_name: %s", conv.chat_id)
                chat_id: str = conv.chat_id
                return chat_id

            # Check participants (for individual chats)
            for participant in conv.participants:
                if name_lower in participant.lower():
                    logger.debug("Found conversation by participant: %s", conv.chat_id)
                    chat_id = conv.chat_id
                    return chat_id

        logger.debug("No conversation found for name: %s", name)
        return None

    def get_reply_context(
        self,
        chat_id: str,
        num_messages: int = 20,
    ) -> ReplyContext:
        """Get context for generating a reply.

        Fetches recent messages and identifies the last received message
        to reply to.

        Args:
            chat_id: The conversation ID.
            num_messages: Number of recent messages to include.

        Returns:
            ReplyContext with messages and formatted context.
        """
        # Get recent messages (newest first from API)
        messages = self.reader.get_messages(chat_id, limit=num_messages)

        # Reverse to chronological order for context
        messages = list(reversed(messages))

        # Find the last received message (not from me)
        last_received = None
        for msg in reversed(messages):
            if not msg.is_from_me:
                last_received = msg
                break

        # Format context for the prompt
        formatted_context = self._format_messages_for_prompt(messages)

        # Try to get conversation info
        conversation = None
        try:
            convos = self.reader.get_conversations(limit=100)
            for conv in convos:
                if conv.chat_id == chat_id:
                    conversation = conv
                    break
        except Exception as e:
            logger.debug("Could not get conversation info: %s", e)

        return ReplyContext(
            messages=messages,
            last_received_message=last_received,
            formatted_context=formatted_context,
            conversation=conversation,
        )

    def get_summary_context(
        self,
        chat_id: str,
        num_messages: int = 50,
    ) -> SummaryContext:
        """Get context for generating a summary.

        Fetches messages and extracts metadata for summary generation.

        Args:
            chat_id: The conversation ID.
            num_messages: Number of messages to summarize.

        Returns:
            SummaryContext with messages and metadata.
        """
        # Get messages (newest first from API)
        messages = self.reader.get_messages(chat_id, limit=num_messages)

        # Reverse to chronological order
        messages = list(reversed(messages))

        # Extract date range
        if messages:
            start_date = messages[0].date
            end_date = messages[-1].date
        else:
            now = datetime.now()
            start_date = now
            end_date = now

        # Extract unique participant names
        participants: set[str] = set()
        for msg in messages:
            if msg.is_from_me:
                participants.add("You")
            elif msg.sender_name:
                participants.add(msg.sender_name)
            elif msg.sender:
                participants.add(msg.sender)

        # Format context for the prompt
        formatted_context = self._format_messages_for_prompt(messages)

        # Try to get conversation info
        conversation = None
        try:
            convos = self.reader.get_conversations(limit=100)
            for conv in convos:
                if conv.chat_id == chat_id:
                    conversation = conv
                    break
        except Exception as e:
            logger.debug("Could not get conversation info: %s", e)

        return SummaryContext(
            messages=messages,
            formatted_context=formatted_context,
            date_range=(start_date, end_date),
            participant_names=sorted(participants),
            conversation=conversation,
        )

    def _format_messages_for_prompt(self, messages: list[Message]) -> str:
        """Format messages as a string for LLM context.

        Args:
            messages: List of messages in chronological order.

        Returns:
            Formatted string with message history.
        """
        lines = []

        for msg in messages:
            # Skip system messages in context
            if msg.is_system_message:
                continue

            # Determine sender name
            if msg.is_from_me:
                sender = "You"
            elif msg.sender_name:
                sender = msg.sender_name
            else:
                sender = msg.sender

            # Format timestamp
            timestamp = msg.date.strftime("%b %d, %H:%M")

            # Format message
            text = msg.text or ""

            # Note attachments
            attachment_note = ""
            if msg.attachments:
                attachment_types = [a.mime_type or "file" for a in msg.attachments]
                attachment_note = f" [Attachments: {', '.join(attachment_types)}]"

            lines.append(f"[{timestamp}] {sender}: {text}{attachment_note}")

        return "\n".join(lines)

    def get_search_context(
        self,
        query: str,
        limit: int = 10,
        person_name: str | None = None,
    ) -> list[Message]:
        """Search for messages matching a query.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            person_name: Optional name filter.

        Returns:
            List of matching messages.
        """
        # If person name is provided, first find the chat_id
        chat_id = None
        if person_name:
            chat_id = self.find_conversation_by_name(person_name)

        results: list[Message] = self.reader.search(
            query=query,
            limit=limit,
            chat_id=chat_id,
        )
        return results
