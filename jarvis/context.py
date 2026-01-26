"""Context fetcher service for RAG-based iMessage retrieval.

Retrieves and formats iMessage data for use in LLM prompts.
"""

from dataclasses import dataclass
from datetime import datetime

from contracts.imessage import Conversation, Message, iMessageReader


@dataclass
class ReplyContext:
    """Context for generating a reply to a conversation."""

    chat_id: str
    participant_names: list[str]
    messages: list[Message]
    formatted_context: str  # Ready for LLM prompt
    last_received_message: Message | None


@dataclass
class SummaryContext:
    """Context for summarizing a conversation."""

    chat_id: str
    participant_names: list[str]
    messages: list[Message]
    formatted_context: str
    date_range: tuple[datetime, datetime]


@dataclass
class SearchContext:
    """Context for search results."""

    query: str
    results: list[Message]
    formatted_context: str


class ContextFetcher:
    """Fetches and formats iMessage context for RAG.

    Uses dependency injection for the iMessage reader to enable
    testing with mock implementations.

    Example:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            context = fetcher.get_reply_context("iMessage;-;+15551234567")
            print(context.formatted_context)
    """

    def __init__(self, reader: iMessageReader, max_cached_conversations: int = 500) -> None:
        """Initialize the context fetcher.

        Args:
            reader: An iMessageReader implementation (e.g., ChatDBReader)
            max_cached_conversations: Maximum number of conversations to cache
                for name resolution (default 500)
        """
        self._reader = reader
        self._max_cached_conversations = max_cached_conversations
        # Cache conversations for name resolution
        self._conversations_cache: list[Conversation] | None = None

    def clear_cache(self) -> None:
        """Clear the conversation cache.

        Call this method if you need to refresh conversation data,
        for example after new messages arrive or conversations change.
        """
        self._conversations_cache = None

    def _get_conversations(self) -> list[Conversation]:
        """Get cached conversations, loading if necessary."""
        if self._conversations_cache is None:
            self._conversations_cache = self._reader.get_conversations(
                limit=self._max_cached_conversations
            )
        return self._conversations_cache

    def _get_participant_names(self, chat_id: str) -> list[str]:
        """Get participant names for a conversation.

        Args:
            chat_id: The conversation ID

        Returns:
            List of participant display names
        """
        conversations = self._get_conversations()
        for conv in conversations:
            if conv.chat_id == chat_id:
                # Use display_name if available, otherwise use participants
                if conv.display_name:
                    return [conv.display_name]
                return conv.participants
        return []

    def _format_sender_name(self, message: Message) -> str:
        """Format the sender name for display.

        Args:
            message: The message to format sender for

        Returns:
            Display name for the sender
        """
        if message.is_from_me:
            return "Me"
        return message.sender_name or message.sender

    def _format_messages(self, messages: list[Message], participant_names: list[str]) -> str:
        """Format messages as context for LLM.

        Args:
            messages: List of messages to format
            participant_names: List of participant names for header

        Returns:
            Formatted string ready for LLM prompt
        """
        if not messages:
            return ""

        lines = []

        # Header with participants
        if participant_names:
            names_str = ", ".join(participant_names)
            lines.append(f"Conversation with: {names_str}")
        lines.append("---")

        # Format each message
        for msg in messages:
            # Skip system messages from formatting (or format them specially)
            if msg.is_system_message:
                timestamp = msg.date.strftime("%Y-%m-%d %H:%M")
                lines.append(f"[{timestamp}] [System] {msg.text}")
                continue

            timestamp = msg.date.strftime("%Y-%m-%d %H:%M")
            sender = self._format_sender_name(msg)
            text = msg.text

            # Include attachment info if present
            if msg.attachments:
                attachment_info = []
                for att in msg.attachments:
                    if att.mime_type and att.mime_type.startswith("image/"):
                        attachment_info.append("[Image]")
                    elif att.mime_type and att.mime_type.startswith("video/"):
                        attachment_info.append("[Video]")
                    elif att.mime_type and att.mime_type.startswith("audio/"):
                        attachment_info.append("[Audio]")
                    else:
                        attachment_info.append(f"[Attachment: {att.filename}]")

                if text:
                    text = f"{text} {' '.join(attachment_info)}"
                else:
                    text = " ".join(attachment_info)

            lines.append(f"[{timestamp}] {sender}: {text}")

        lines.append("---")

        return "\n".join(lines)

    def _format_search_results(self, query: str, messages: list[Message]) -> str:
        """Format search results as context for LLM.

        Args:
            query: The search query
            messages: List of matching messages

        Returns:
            Formatted string ready for LLM prompt
        """
        if not messages:
            return f'No messages found matching "{query}"'

        lines = [f'Search results for "{query}":']
        lines.append("---")

        for msg in messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M")
            sender = self._format_sender_name(msg)
            text = msg.text

            # Include attachment info if present
            if msg.attachments and not text:
                text = f"[{len(msg.attachments)} attachment(s)]"

            lines.append(f"[{timestamp}] {sender}: {text}")

        lines.append("---")

        return "\n".join(lines)

    def get_reply_context(self, chat_id: str, num_messages: int = 20) -> ReplyContext:
        """Get context for replying to a conversation.

        Retrieves recent messages from a conversation, formatted for LLM use.

        Args:
            chat_id: The conversation ID
            num_messages: Number of recent messages to retrieve (default 20)

        Returns:
            ReplyContext with formatted messages and metadata
        """
        messages = self._reader.get_messages(chat_id, limit=num_messages)

        # Messages come newest-first, reverse for chronological order
        messages = list(reversed(messages))

        participant_names = self._get_participant_names(chat_id)
        formatted = self._format_messages(messages, participant_names)

        # Find last received message (not from me)
        last_received: Message | None = None
        for msg in reversed(messages):
            if not msg.is_from_me:
                last_received = msg
                break

        return ReplyContext(
            chat_id=chat_id,
            participant_names=participant_names,
            messages=messages,
            formatted_context=formatted,
            last_received_message=last_received,
        )

    def get_summary_context(self, chat_id: str, num_messages: int = 50) -> SummaryContext:
        """Get context for summarizing a conversation.

        Retrieves messages from a conversation for summarization.

        Args:
            chat_id: The conversation ID
            num_messages: Number of messages to retrieve (default 50)

        Returns:
            SummaryContext with formatted messages and date range
        """
        messages = self._reader.get_messages(chat_id, limit=num_messages)

        # Messages come newest-first, reverse for chronological order
        messages = list(reversed(messages))

        participant_names = self._get_participant_names(chat_id)
        formatted = self._format_messages(messages, participant_names)

        # Calculate date range
        if messages:
            date_range = (messages[0].date, messages[-1].date)
        else:
            now = datetime.now()
            date_range = (now, now)

        return SummaryContext(
            chat_id=chat_id,
            participant_names=participant_names,
            messages=messages,
            formatted_context=formatted,
            date_range=date_range,
        )

    def get_search_context(self, query: str, limit: int = 20) -> SearchContext:
        """Get context from search results.

        Searches messages and formats results for LLM use.

        Args:
            query: Search query string
            limit: Maximum number of results (default 20)

        Returns:
            SearchContext with formatted search results
        """
        results = self._reader.search(query, limit=limit)
        formatted = self._format_search_results(query, results)

        return SearchContext(
            query=query,
            results=results,
            formatted_context=formatted,
        )

    def find_conversation_by_name(self, name: str) -> str | None:
        """Find a conversation by participant name.

        Performs fuzzy matching against conversation display names
        and participant identifiers.

        Args:
            name: Name to search for (case insensitive)

        Returns:
            chat_id if found, None otherwise
        """
        if not name:
            return None

        name_lower = name.lower().strip()
        conversations = self._get_conversations()

        # First pass: exact match on display name
        for conv in conversations:
            if conv.display_name and conv.display_name.lower() == name_lower:
                return conv.chat_id

        # Second pass: partial match on display name
        for conv in conversations:
            if conv.display_name and name_lower in conv.display_name.lower():
                return conv.chat_id

        # Third pass: match on participant identifiers (phone/email)
        for conv in conversations:
            for participant in conv.participants:
                # Check if name matches participant (case insensitive)
                if name_lower in participant.lower():
                    return conv.chat_id

        # Fourth pass: check if it's a phone number pattern
        # Remove common formatting and check against normalized participants
        name_digits = "".join(c for c in name if c.isdigit())
        if len(name_digits) >= 7:  # Minimum for a partial phone match
            for conv in conversations:
                for participant in conv.participants:
                    participant_digits = "".join(c for c in participant if c.isdigit())
                    if name_digits in participant_digits or participant_digits.endswith(
                        name_digits
                    ):
                        return conv.chat_id

        return None
