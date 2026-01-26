"""Unit tests for context fetcher service."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from contracts.imessage import Attachment, Conversation, Message
from jarvis.context import (
    ContextFetcher,
    ReplyContext,
    SearchContext,
    SummaryContext,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_reader() -> MagicMock:
    """Create a mock iMessageReader."""
    return MagicMock()


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(
            id=3,
            chat_id="iMessage;-;+15551234567",
            sender="+15551234567",
            sender_name="John Smith",
            text="Perfect, see you there!",
            date=datetime(2024, 1, 25, 10, 35, tzinfo=UTC),
            is_from_me=False,
            attachments=[],
            reply_to_id=None,
            reactions=[],
        ),
        Message(
            id=2,
            chat_id="iMessage;-;+15551234567",
            sender="me",
            sender_name=None,
            text="How about 7pm at the Italian place?",
            date=datetime(2024, 1, 25, 10, 33, tzinfo=UTC),
            is_from_me=True,
            attachments=[],
            reply_to_id=None,
            reactions=[],
        ),
        Message(
            id=1,
            chat_id="iMessage;-;+15551234567",
            sender="+15551234567",
            sender_name="John Smith",
            text="Hey, are we still on for dinner tomorrow?",
            date=datetime(2024, 1, 25, 10, 30, tzinfo=UTC),
            is_from_me=False,
            attachments=[],
            reply_to_id=None,
            reactions=[],
        ),
    ]


@pytest.fixture
def sample_conversations() -> list[Conversation]:
    """Create sample conversations for testing."""
    return [
        Conversation(
            chat_id="iMessage;-;+15551234567",
            participants=["+15551234567"],
            display_name="John Smith",
            last_message_date=datetime(2024, 1, 25, 10, 35, tzinfo=UTC),
            message_count=100,
            is_group=False,
        ),
        Conversation(
            chat_id="iMessage;-;+15559876543",
            participants=["+15559876543"],
            display_name="Jane Doe",
            last_message_date=datetime(2024, 1, 24, 15, 20, tzinfo=UTC),
            message_count=50,
            is_group=False,
        ),
        Conversation(
            chat_id="chat123456",
            participants=["+15551111111", "+15552222222", "+15553333333"],
            display_name="Family Group",
            last_message_date=datetime(2024, 1, 25, 8, 0, tzinfo=UTC),
            message_count=500,
            is_group=True,
        ),
    ]


# =============================================================================
# ContextFetcher Tests
# =============================================================================


class TestContextFetcherInit:
    """Tests for ContextFetcher initialization."""

    def test_init_with_reader(self, mock_reader: MagicMock):
        """Initialize with a reader."""
        fetcher = ContextFetcher(mock_reader)
        assert fetcher._reader is mock_reader
        assert fetcher._conversations_cache is None


class TestGetReplyContext:
    """Tests for get_reply_context method."""

    def test_basic_reply_context(
        self,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        sample_conversations: list[Conversation],
    ):
        """Get basic reply context."""
        mock_reader.get_messages.return_value = sample_messages
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("iMessage;-;+15551234567")

        assert isinstance(result, ReplyContext)
        assert result.chat_id == "iMessage;-;+15551234567"
        assert result.participant_names == ["John Smith"]
        assert len(result.messages) == 3
        # Messages should be in chronological order (reversed from input)
        assert result.messages[0].id == 1
        assert result.messages[-1].id == 3

    def test_reply_context_formatted_output(
        self,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        sample_conversations: list[Conversation],
    ):
        """Verify formatted context output."""
        mock_reader.get_messages.return_value = sample_messages
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("iMessage;-;+15551234567")

        # Check formatted context structure
        assert "Conversation with: John Smith" in result.formatted_context
        assert "---" in result.formatted_context
        assert "[2024-01-25 10:30] John Smith:" in result.formatted_context
        assert "[2024-01-25 10:33] Me:" in result.formatted_context
        assert "dinner tomorrow" in result.formatted_context

    def test_last_received_message(
        self,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        sample_conversations: list[Conversation],
    ):
        """Find last received message correctly."""
        mock_reader.get_messages.return_value = sample_messages
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("iMessage;-;+15551234567")

        assert result.last_received_message is not None
        assert result.last_received_message.id == 3
        assert result.last_received_message.text == "Perfect, see you there!"

    def test_no_received_messages(self, mock_reader: MagicMock):
        """Handle case where all messages are from me."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="me",
                sender_name=None,
                text="Hello",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=True,
                attachments=[],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert result.last_received_message is None

    def test_custom_num_messages(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Pass custom limit to get_messages."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        fetcher.get_reply_context("chat1", num_messages=50)

        mock_reader.get_messages.assert_called_once_with("chat1", limit=50)


class TestGetSummaryContext:
    """Tests for get_summary_context method."""

    def test_basic_summary_context(
        self,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        sample_conversations: list[Conversation],
    ):
        """Get basic summary context."""
        mock_reader.get_messages.return_value = sample_messages
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_summary_context("iMessage;-;+15551234567")

        assert isinstance(result, SummaryContext)
        assert result.chat_id == "iMessage;-;+15551234567"
        assert result.participant_names == ["John Smith"]
        assert len(result.messages) == 3

    def test_date_range_calculation(
        self,
        mock_reader: MagicMock,
        sample_messages: list[Message],
        sample_conversations: list[Conversation],
    ):
        """Calculate date range correctly."""
        mock_reader.get_messages.return_value = sample_messages
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_summary_context("iMessage;-;+15551234567")

        # After reversal: first message is oldest, last is newest
        start_date, end_date = result.date_range
        assert start_date.hour == 10
        assert start_date.minute == 30
        assert end_date.hour == 10
        assert end_date.minute == 35

    def test_empty_conversation_date_range(self, mock_reader: MagicMock):
        """Handle empty conversation date range."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_summary_context("chat1")

        # Should return same datetime for both start and end
        start_date, end_date = result.date_range
        assert start_date == end_date

    def test_custom_num_messages(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Pass custom limit for summary."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        fetcher.get_summary_context("chat1", num_messages=100)

        mock_reader.get_messages.assert_called_once_with("chat1", limit=100)


class TestGetSearchContext:
    """Tests for get_search_context method."""

    def test_basic_search_context(self, mock_reader: MagicMock):
        """Get basic search context."""
        mock_reader.search.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="Let's meet for dinner tomorrow",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[],
            ),
        ]

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_search_context("dinner")

        assert isinstance(result, SearchContext)
        assert result.query == "dinner"
        assert len(result.results) == 1

    def test_search_formatted_output(self, mock_reader: MagicMock):
        """Verify search results formatting."""
        mock_reader.search.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="Let's meet for dinner tomorrow",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[],
            ),
        ]

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_search_context("dinner")

        assert 'Search results for "dinner":' in result.formatted_context
        assert "---" in result.formatted_context
        assert "John:" in result.formatted_context
        assert "dinner tomorrow" in result.formatted_context

    def test_empty_search_results(self, mock_reader: MagicMock):
        """Handle empty search results."""
        mock_reader.search.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_search_context("nonexistent")

        assert result.query == "nonexistent"
        assert len(result.results) == 0
        assert 'No messages found matching "nonexistent"' in result.formatted_context

    def test_custom_limit(self, mock_reader: MagicMock):
        """Pass custom limit to search."""
        mock_reader.search.return_value = []

        fetcher = ContextFetcher(mock_reader)
        fetcher.get_search_context("query", limit=50)

        mock_reader.search.assert_called_once_with("query", limit=50)


class TestFindConversationByName:
    """Tests for find_conversation_by_name method."""

    def test_exact_match_display_name(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find by exact display name match."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("John Smith")

        assert result == "iMessage;-;+15551234567"

    def test_case_insensitive_match(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Match is case insensitive."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("JOHN SMITH")

        assert result == "iMessage;-;+15551234567"

    def test_partial_match_display_name(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find by partial display name match."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("John")

        assert result == "iMessage;-;+15551234567"

    def test_match_participant_identifier(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find by participant phone number."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("+15551234567")

        assert result == "iMessage;-;+15551234567"

    def test_match_phone_number_digits(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find by phone number with different formatting."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("555-123-4567")

        assert result == "iMessage;-;+15551234567"

    def test_match_partial_phone_number(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find by partial phone number."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("1234567")

        assert result == "iMessage;-;+15551234567"

    def test_no_match_returns_none(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Return None when no match found."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("Unknown Person")

        assert result is None

    def test_empty_name_returns_none(self, mock_reader: MagicMock):
        """Return None for empty name."""
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("")

        assert result is None

    def test_group_name_match(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Find group chat by name."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("Family Group")

        assert result == "chat123456"

    def test_whitespace_trimmed(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Trim whitespace from search name."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("  John Smith  ")

        assert result == "iMessage;-;+15551234567"


class TestContextFormatting:
    """Tests for message formatting."""

    def test_format_with_attachments(self, mock_reader: MagicMock):
        """Format messages with attachments."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="Check this out",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[
                    Attachment(
                        filename="photo.jpg",
                        file_path="/path/to/photo.jpg",
                        mime_type="image/jpeg",
                        file_size=1024,
                    ),
                ],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert "[Image]" in result.formatted_context
        assert "Check this out" in result.formatted_context

    def test_format_attachment_only_message(self, mock_reader: MagicMock):
        """Format message with only attachment (no text)."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[
                    Attachment(
                        filename="video.mp4",
                        file_path="/path/to/video.mp4",
                        mime_type="video/mp4",
                        file_size=10240,
                    ),
                ],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert "[Video]" in result.formatted_context

    def test_format_audio_attachment(self, mock_reader: MagicMock):
        """Format message with audio attachment."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="Voice memo",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[
                    Attachment(
                        filename="audio.m4a",
                        file_path="/path/to/audio.m4a",
                        mime_type="audio/m4a",
                        file_size=5000,
                    ),
                ],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert "[Audio]" in result.formatted_context

    def test_format_generic_attachment(self, mock_reader: MagicMock):
        """Format message with generic attachment type."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="Here's the document",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[
                    Attachment(
                        filename="document.pdf",
                        file_path="/path/to/document.pdf",
                        mime_type="application/pdf",
                        file_size=50000,
                    ),
                ],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert "[Attachment: document.pdf]" in result.formatted_context

    def test_format_system_message(self, mock_reader: MagicMock):
        """Format system messages (group events)."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="John left the group",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[],
                is_system_message=True,
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert "[System]" in result.formatted_context
        assert "left the group" in result.formatted_context

    def test_format_sender_without_name(self, mock_reader: MagicMock):
        """Format sender when no contact name available."""
        mock_reader.get_messages.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name=None,  # No contact name
                text="Hello",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[],
            ),
        ]
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        # Should fall back to phone number
        assert "+15551234567:" in result.formatted_context

    def test_search_attachment_only_formatting(self, mock_reader: MagicMock):
        """Format search result with attachment but no text."""
        mock_reader.search.return_value = [
            Message(
                id=1,
                chat_id="chat1",
                sender="+15551234567",
                sender_name="John",
                text="",
                date=datetime(2024, 1, 25, 10, 0, tzinfo=UTC),
                is_from_me=False,
                attachments=[
                    Attachment(
                        filename="photo1.jpg",
                        file_path=None,
                        mime_type="image/jpeg",
                        file_size=1024,
                    ),
                    Attachment(
                        filename="photo2.jpg",
                        file_path=None,
                        mime_type="image/jpeg",
                        file_size=1024,
                    ),
                ],
            ),
        ]

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_search_context("photos")

        assert "[2 attachment(s)]" in result.formatted_context


class TestEmptyConversationHandling:
    """Tests for handling empty conversations."""

    def test_empty_messages_reply_context(self, mock_reader: MagicMock):
        """Handle empty messages in reply context."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("chat1")

        assert result.messages == []
        assert result.last_received_message is None
        assert result.formatted_context == ""

    def test_empty_messages_summary_context(self, mock_reader: MagicMock):
        """Handle empty messages in summary context."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_summary_context("chat1")

        assert result.messages == []
        assert result.formatted_context == ""

    def test_unknown_chat_id_participant_names(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Handle unknown chat_id for participant names."""
        mock_reader.get_messages.return_value = []
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.get_reply_context("unknown_chat_id")

        assert result.participant_names == []


class TestConversationsCache:
    """Tests for conversations caching."""

    def test_conversations_cached(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """Conversations are cached after first fetch."""
        mock_reader.get_conversations.return_value = sample_conversations
        mock_reader.get_messages.return_value = []

        fetcher = ContextFetcher(mock_reader)

        # Call twice
        fetcher.get_reply_context("chat1")
        fetcher.get_reply_context("chat2")

        # get_conversations should only be called once
        mock_reader.get_conversations.assert_called_once()

    def test_find_conversation_uses_cache(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """find_conversation_by_name uses cached conversations."""
        mock_reader.get_conversations.return_value = sample_conversations

        fetcher = ContextFetcher(mock_reader)

        # Call multiple times
        fetcher.find_conversation_by_name("John")
        fetcher.find_conversation_by_name("Jane")
        fetcher.find_conversation_by_name("Family")

        # Should only fetch conversations once
        mock_reader.get_conversations.assert_called_once()

    def test_clear_cache_forces_refetch(
        self, mock_reader: MagicMock, sample_conversations: list[Conversation]
    ):
        """clear_cache() forces conversations to be refetched."""
        mock_reader.get_conversations.return_value = sample_conversations
        mock_reader.get_messages.return_value = []

        fetcher = ContextFetcher(mock_reader)

        # First call populates cache
        fetcher.get_reply_context("chat1")
        assert mock_reader.get_conversations.call_count == 1

        # Clear the cache
        fetcher.clear_cache()

        # Next call should refetch
        fetcher.get_reply_context("chat2")
        assert mock_reader.get_conversations.call_count == 2

    def test_custom_max_cached_conversations(self, mock_reader: MagicMock):
        """Custom max_cached_conversations is passed to reader."""
        mock_reader.get_conversations.return_value = []
        mock_reader.get_messages.return_value = []

        fetcher = ContextFetcher(mock_reader, max_cached_conversations=100)
        fetcher.get_reply_context("chat1")

        mock_reader.get_conversations.assert_called_once_with(limit=100)
