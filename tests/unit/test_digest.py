"""Unit tests for the digest module.

Tests digest generation functionality including:
- Unanswered conversation detection
- Group highlight extraction
- Action item detection
- Statistics calculation
- Export to markdown/HTML
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from contracts.imessage import Conversation, Message
from jarvis.digest import (
    ActionItem,
    Digest,
    DigestFormat,
    DigestGenerator,
    DigestPeriod,
    GroupHighlight,
    MessageStats,
    UnansweredConversation,
    export_digest_html,
    export_digest_markdown,
    generate_digest,
    get_digest_filename,
)


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    now = datetime.now()
    return [
        Message(
            id=1,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Hey, can you send me the report?",
            date=now - timedelta(hours=2),
            is_from_me=False,
            attachments=[],
            reactions=[],
        ),
        Message(
            id=2,
            chat_id="iMessage;-;+1234567890",
            sender="me",
            sender_name=None,
            text="Sure, I'll do that.",
            date=now - timedelta(hours=1, minutes=55),
            is_from_me=True,
            attachments=[],
            reactions=[],
        ),
        Message(
            id=3,
            chat_id="iMessage;-;+1234567890",
            sender="+1234567890",
            sender_name="John Smith",
            text="Thanks! Also, meeting tomorrow at 3:00pm?",
            date=now - timedelta(hours=1),
            is_from_me=False,
            attachments=[],
            reactions=[],
        ),
    ]


@pytest.fixture
def sample_conversations() -> list[Conversation]:
    """Create sample conversations for testing."""
    now = datetime.now()
    return [
        Conversation(
            chat_id="iMessage;-;+1234567890",
            participants=["+1234567890"],
            display_name="John Smith",
            last_message_date=now - timedelta(hours=1),
            message_count=10,
            is_group=False,
            last_message_text="Thanks! Also, meeting tomorrow at 3:00pm?",
        ),
        Conversation(
            chat_id="chat12345",
            participants=["+1111111111", "+2222222222", "+3333333333"],
            display_name="Family Group",
            last_message_date=now - timedelta(minutes=30),
            message_count=50,
            is_group=True,
            last_message_text="See you all tonight!",
        ),
    ]


@pytest.fixture
def group_messages() -> list[Message]:
    """Create sample group messages for testing."""
    now = datetime.now()
    return [
        Message(
            id=10,
            chat_id="chat12345",
            sender="+1111111111",
            sender_name="Alice",
            text="Let's plan dinner for tonight",
            date=now - timedelta(hours=3),
            is_from_me=False,
        ),
        Message(
            id=11,
            chat_id="chat12345",
            sender="+2222222222",
            sender_name="Bob",
            text="Sounds good! What time?",
            date=now - timedelta(hours=2, minutes=30),
            is_from_me=False,
        ),
        Message(
            id=12,
            chat_id="chat12345",
            sender="+1111111111",
            sender_name="Alice",
            text="How about 7pm?",
            date=now - timedelta(hours=2),
            is_from_me=False,
        ),
        Message(
            id=13,
            chat_id="chat12345",
            sender="me",
            sender_name=None,
            text="Works for me!",
            date=now - timedelta(hours=1, minutes=30),
            is_from_me=True,
        ),
        Message(
            id=14,
            chat_id="chat12345",
            sender="+3333333333",
            sender_name="Charlie",
            text="See you all tonight!",
            date=now - timedelta(minutes=30),
            is_from_me=False,
        ),
    ]


@pytest.fixture
def mock_reader(sample_conversations, sample_messages, group_messages):
    """Create a mock ChatDBReader."""
    reader = MagicMock()
    reader.get_conversations.return_value = sample_conversations

    def get_messages_side_effect(chat_id, limit=100):
        if chat_id == "iMessage;-;+1234567890":
            return sample_messages
        elif chat_id == "chat12345":
            return group_messages
        return []

    reader.get_messages.side_effect = get_messages_side_effect
    return reader


class TestDigestPeriod:
    """Tests for DigestPeriod enum."""

    def test_daily_period(self):
        """Daily period value is correct."""
        assert DigestPeriod.DAILY.value == "daily"

    def test_weekly_period(self):
        """Weekly period value is correct."""
        assert DigestPeriod.WEEKLY.value == "weekly"


class TestDigestFormat:
    """Tests for DigestFormat enum."""

    def test_markdown_format(self):
        """Markdown format value is correct."""
        assert DigestFormat.MARKDOWN.value == "markdown"

    def test_html_format(self):
        """HTML format value is correct."""
        assert DigestFormat.HTML.value == "html"


class TestDigestGenerator:
    """Tests for DigestGenerator class."""

    def test_generate_daily_digest(self, mock_reader):
        """Generates a daily digest."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate(period=DigestPeriod.DAILY)

        assert digest.period == DigestPeriod.DAILY
        assert digest.generated_at is not None
        assert digest.start_date < digest.end_date

    def test_generate_weekly_digest(self, mock_reader):
        """Generates a weekly digest."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate(period=DigestPeriod.WEEKLY)

        assert digest.period == DigestPeriod.WEEKLY
        # Weekly should be 7 days
        delta = digest.end_date - digest.start_date
        assert delta.days == 7

    def test_period_from_string(self, mock_reader):
        """Accepts period as string."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate(period="daily")

        assert digest.period == DigestPeriod.DAILY

    def test_custom_end_date(self, mock_reader):
        """Accepts custom end date."""
        end_date = datetime(2024, 6, 15, 12, 0, 0)
        generator = DigestGenerator(mock_reader)
        digest = generator.generate(end_date=end_date)

        assert digest.end_date == end_date

    def test_finds_unanswered_conversations(self, mock_reader):
        """Finds conversations with unanswered messages."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()

        # Should find the individual conversation (John Smith) as unanswered
        unanswered_names = [c.display_name for c in digest.needs_attention]
        assert "John Smith" in unanswered_names

    def test_finds_group_highlights(self, mock_reader):
        """Finds active group highlights."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()

        # Should find the Family Group as a highlight
        highlight_names = [h.display_name for h in digest.highlights]
        assert "Family Group" in highlight_names

    def test_detects_action_items(self, mock_reader):
        """Detects action items from messages."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()

        # Should detect the "can you send me the report" as a task
        task_texts = [a.text for a in digest.action_items if a.item_type == "task"]
        assert any("report" in text.lower() for text in task_texts)

    def test_calculates_stats(self, mock_reader):
        """Calculates message statistics."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()

        assert digest.stats.total_messages >= 0
        assert digest.stats.active_conversations >= 0

    def test_to_dict(self, mock_reader):
        """Digest can be converted to dictionary."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        data = digest.to_dict()

        assert "period" in data
        assert "generated_at" in data
        assert "needs_attention" in data
        assert "highlights" in data
        assert "action_items" in data
        assert "stats" in data


class TestActionItemDetection:
    """Tests for action item detection patterns."""

    def test_detects_task_requests(self):
        """Detects task requests like 'can you...'."""
        now = datetime.now()
        task_message = Message(
            id=100,
            chat_id="test-chat",
            sender="+1234567890",
            sender_name="John",
            text="Can you please review the document and send feedback?",
            date=now - timedelta(minutes=30),
            is_from_me=False,
        )
        conversation = Conversation(
            chat_id="test-chat",
            participants=["+1234567890"],
            display_name="John",
            last_message_date=now,
            message_count=1,
            is_group=False,
        )

        reader = MagicMock()
        reader.get_conversations.return_value = [conversation]
        reader.get_messages.return_value = [task_message]

        generator = DigestGenerator(reader)
        digest = generator.generate()

        task_types = [a.item_type for a in digest.action_items]
        assert "task" in task_types

    def test_detects_questions(self):
        """Detects questions."""
        now = datetime.now()
        question_message = Message(
            id=100,
            chat_id="test-chat",
            sender="+1234567890",
            sender_name="John",
            text="What time is the meeting tomorrow afternoon?",
            date=now - timedelta(minutes=30),
            is_from_me=False,
        )
        conversation = Conversation(
            chat_id="test-chat",
            participants=["+1234567890"],
            display_name="John",
            last_message_date=now,
            message_count=1,
            is_group=False,
        )

        reader = MagicMock()
        reader.get_conversations.return_value = [conversation]
        reader.get_messages.return_value = [question_message]

        generator = DigestGenerator(reader)
        digest = generator.generate()

        question_types = [a.item_type for a in digest.action_items]
        assert "question" in question_types

    def test_detects_events(self):
        """Detects event mentions."""
        now = datetime.now()
        event_message = Message(
            id=100,
            chat_id="test-chat",
            sender="+1234567890",
            sender_name="John",
            text="Meeting tomorrow at 3:00pm in the conference room",
            date=now - timedelta(minutes=30),
            is_from_me=False,
        )
        conversation = Conversation(
            chat_id="test-chat",
            participants=["+1234567890"],
            display_name="John",
            last_message_date=now,
            message_count=1,
            is_group=False,
        )

        reader = MagicMock()
        reader.get_conversations.return_value = [conversation]
        reader.get_messages.return_value = [event_message]

        generator = DigestGenerator(reader)
        digest = generator.generate()

        event_types = [a.item_type for a in digest.action_items]
        assert "event" in event_types

    def test_detects_reminders(self):
        """Detects reminders."""
        now = datetime.now()
        reminder_message = Message(
            id=100,
            chat_id="test-chat",
            sender="+1234567890",
            sender_name="John",
            text="Hey, remind me to call the doctor tomorrow",
            date=now - timedelta(minutes=30),
            is_from_me=False,
        )
        conversation = Conversation(
            chat_id="test-chat",
            participants=["+1234567890"],
            display_name="John",
            last_message_date=now,
            message_count=1,
            is_group=False,
        )

        reader = MagicMock()
        reader.get_conversations.return_value = [conversation]
        reader.get_messages.return_value = [reminder_message]

        generator = DigestGenerator(reader)
        digest = generator.generate()

        reminder_types = [a.item_type for a in digest.action_items]
        assert "reminder" in reminder_types


class TestGenerateDigestFunction:
    """Tests for the convenience generate_digest function."""

    def test_convenience_function(self, mock_reader):
        """Convenience function works correctly."""
        digest = generate_digest(mock_reader, period="daily")

        assert digest.period == DigestPeriod.DAILY
        assert isinstance(digest, Digest)


class TestExportDigestMarkdown:
    """Tests for Markdown export."""

    def test_exports_to_markdown(self, mock_reader):
        """Exports digest to Markdown format."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        md = export_digest_markdown(digest)

        assert "# JARVIS" in md
        assert "Digest" in md

    def test_includes_activity_summary(self, mock_reader):
        """Markdown includes activity summary."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        md = export_digest_markdown(digest)

        assert "Activity Summary" in md
        assert "Total Messages" in md

    def test_includes_needs_attention(self, mock_reader):
        """Markdown includes needs attention section."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        md = export_digest_markdown(digest)

        if digest.needs_attention:
            assert "Needs Attention" in md

    def test_includes_highlights(self, mock_reader):
        """Markdown includes highlights section."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        md = export_digest_markdown(digest)

        if digest.highlights:
            assert "Highlights" in md

    def test_includes_action_items(self, mock_reader):
        """Markdown includes action items section."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        md = export_digest_markdown(digest)

        if digest.action_items:
            assert "Action Items" in md


class TestExportDigestHtml:
    """Tests for HTML export."""

    def test_exports_to_html(self, mock_reader):
        """Exports digest to HTML format."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        html = export_digest_html(digest)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html

    def test_includes_styles(self, mock_reader):
        """HTML includes CSS styles."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        html = export_digest_html(digest)

        assert "<style>" in html
        assert "font-family" in html

    def test_includes_title(self, mock_reader):
        """HTML includes proper title."""
        generator = DigestGenerator(mock_reader)
        digest = generator.generate()
        html = export_digest_html(digest)

        assert "<title>" in html
        assert "Digest" in html


class TestGetDigestFilename:
    """Tests for filename generation."""

    def test_generates_markdown_filename(self):
        """Generates Markdown filename."""
        filename = get_digest_filename(DigestPeriod.DAILY, DigestFormat.MARKDOWN)

        assert filename.endswith(".md")
        assert "jarvis_digest" in filename
        assert "daily" in filename

    def test_generates_html_filename(self):
        """Generates HTML filename."""
        filename = get_digest_filename(DigestPeriod.DAILY, DigestFormat.HTML)

        assert filename.endswith(".html")

    def test_includes_period(self):
        """Filename includes period."""
        daily_filename = get_digest_filename(DigestPeriod.DAILY, DigestFormat.MARKDOWN)
        weekly_filename = get_digest_filename(DigestPeriod.WEEKLY, DigestFormat.MARKDOWN)

        assert "daily" in daily_filename
        assert "weekly" in weekly_filename

    def test_includes_date(self):
        """Filename includes date."""
        date = datetime(2024, 6, 15)
        filename = get_digest_filename(DigestPeriod.DAILY, DigestFormat.MARKDOWN, date)

        assert "20240615" in filename


class TestDigestDataclasses:
    """Tests for digest dataclasses."""

    def test_unanswered_conversation(self):
        """UnansweredConversation dataclass works correctly."""
        conv = UnansweredConversation(
            chat_id="test",
            display_name="Test User",
            participants=["+1234567890"],
            unanswered_count=3,
            last_message_date=datetime.now(),
            last_message_preview="Hello",
            is_group=False,
        )
        assert conv.chat_id == "test"
        assert conv.unanswered_count == 3

    def test_group_highlight(self):
        """GroupHighlight dataclass works correctly."""
        highlight = GroupHighlight(
            chat_id="test",
            display_name="Test Group",
            participants=["+1111", "+2222"],
            message_count=50,
            active_participants=["Alice", "Bob"],
            top_topics=["meeting", "plans"],
            last_activity=datetime.now(),
        )
        assert highlight.message_count == 50
        assert len(highlight.top_topics) == 2

    def test_action_item(self):
        """ActionItem dataclass works correctly."""
        item = ActionItem(
            text="Can you send the report?",
            chat_id="test",
            conversation_name="Test User",
            sender="Test User",
            date=datetime.now(),
            message_id=123,
            item_type="task",
        )
        assert item.item_type == "task"
        assert item.message_id == 123

    def test_message_stats(self):
        """MessageStats dataclass works correctly."""
        stats = MessageStats(
            total_sent=50,
            total_received=100,
            total_messages=150,
            active_conversations=10,
            most_active_conversation="Family",
            most_active_count=30,
            avg_messages_per_day=21.4,
            busiest_hour=14,
            hourly_distribution={14: 20, 15: 15},
        )
        assert stats.total_messages == 150
        assert stats.busiest_hour == 14


class TestDigestSchemas:
    """Tests for digest-related Pydantic schemas."""

    def test_digest_period_enum(self):
        """DigestPeriodEnum has correct values."""
        from api.schemas import DigestPeriodEnum

        assert DigestPeriodEnum.DAILY.value == "daily"
        assert DigestPeriodEnum.WEEKLY.value == "weekly"

    def test_digest_format_enum(self):
        """DigestFormatEnum has correct values."""
        from api.schemas import DigestFormatEnum

        assert DigestFormatEnum.MARKDOWN.value == "markdown"
        assert DigestFormatEnum.HTML.value == "html"

    def test_digest_generate_request_defaults(self):
        """DigestGenerateRequest has correct defaults."""
        from api.schemas import DigestGenerateRequest, DigestPeriodEnum

        request = DigestGenerateRequest()
        assert request.period == DigestPeriodEnum.DAILY
        assert request.end_date is None

    def test_digest_export_request_defaults(self):
        """DigestExportRequest has correct defaults."""
        from api.schemas import DigestExportRequest, DigestFormatEnum, DigestPeriodEnum

        request = DigestExportRequest()
        assert request.period == DigestPeriodEnum.DAILY
        assert request.format == DigestFormatEnum.MARKDOWN

    def test_digest_preferences_response(self):
        """DigestPreferencesResponse model works correctly."""
        from api.schemas import DigestPreferencesResponse

        response = DigestPreferencesResponse(
            enabled=True,
            schedule="daily",
            preferred_time="08:00",
            include_action_items=True,
            include_stats=True,
            max_conversations=50,
            export_format="markdown",
        )
        assert response.enabled is True
        assert response.schedule == "daily"
