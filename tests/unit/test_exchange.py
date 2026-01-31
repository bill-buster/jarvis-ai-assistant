"""Tests for jarvis/exchange.py - Exchange data structures."""

from datetime import datetime

from jarvis.exchange import CandidateExchange, ContextMessage, ExchangeConfig


class TestContextMessage:
    """Tests for ContextMessage dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a ContextMessage."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        msg = ContextMessage(
            speaker="them",
            timestamp=dt,
            text="Hello there!",
        )
        assert msg.speaker == "them"
        assert msg.timestamp == dt
        assert msg.text == "Hello there!"
        assert msg.flags == set()
        assert msg.raw_text is None

    def test_with_flags(self) -> None:
        """Test ContextMessage with flags."""
        msg = ContextMessage(
            speaker="me",
            timestamp=datetime.now(),
            text="<ATTACHMENT:image>",
            flags={"attachment", "emoji_only"},
            raw_text="",
        )
        assert "attachment" in msg.flags
        assert "emoji_only" in msg.flags

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        msg = ContextMessage(
            speaker="them",
            timestamp=dt,
            text="Hi",
            flags={"reaction"},
            raw_text='Liked "Hi"',
        )
        d = msg.to_dict()
        assert d["speaker"] == "them"
        assert d["text"] == "Hi"
        assert "reaction" in d["flags"]
        assert d["raw_text"] == 'Liked "Hi"'

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "speaker": "me",
            "timestamp": "2024-01-15T10:30:00",
            "text": "Hello",
            "flags": ["emoji_only"],
            "raw_text": None,
        }
        msg = ContextMessage.from_dict(d)
        assert msg.speaker == "me"
        assert msg.text == "Hello"
        assert "emoji_only" in msg.flags


class TestCandidateExchange:
    """Tests for CandidateExchange dataclass."""

    def test_basic_exchange(self) -> None:
        """Test creating a basic exchange."""
        t1 = datetime(2024, 1, 15, 10, 0, 0)
        t2 = datetime(2024, 1, 15, 10, 1, 0)

        trigger = ContextMessage(speaker="them", timestamp=t1, text="Hey, how are you?")
        response = ContextMessage(speaker="me", timestamp=t2, text="Doing great!")

        exchange = CandidateExchange(
            trigger_span=[trigger],
            response_span=[response],
            context_window=[],
            chat_id="chat123",
        )

        assert exchange.trigger_text == "Hey, how are you?"
        assert exchange.response_text == "Doing great!"
        assert exchange.chat_id == "chat123"
        assert exchange.contact_id is None

    def test_multi_message_spans(self) -> None:
        """Test exchange with multiple messages in spans."""
        t1 = datetime(2024, 1, 15, 10, 0, 0)
        t2 = datetime(2024, 1, 15, 10, 0, 30)
        t3 = datetime(2024, 1, 15, 10, 1, 0)
        t4 = datetime(2024, 1, 15, 10, 1, 15)

        trigger_span = [
            ContextMessage(speaker="them", timestamp=t1, text="Hey"),
            ContextMessage(speaker="them", timestamp=t2, text="Are you free tonight?"),
        ]
        response_span = [
            ContextMessage(speaker="me", timestamp=t3, text="Yeah!"),
            ContextMessage(speaker="me", timestamp=t4, text="What time?"),
        ]

        exchange = CandidateExchange(
            trigger_span=trigger_span,
            response_span=response_span,
            context_window=[],
            chat_id="chat123",
            trigger_msg_ids=[1, 2],
            response_msg_ids=[3, 4],
        )

        assert exchange.trigger_text == "Hey\nAre you free tonight?"
        assert exchange.response_text == "Yeah!\nWhat time?"
        assert exchange.trigger_start_time == t1
        assert exchange.trigger_end_time == t2
        assert exchange.response_start_time == t3
        assert exchange.response_end_time == t4

    def test_time_gap_calculation(self) -> None:
        """Test time_gap_minutes calculation."""
        t1 = datetime(2024, 1, 15, 10, 0, 0)
        t2 = datetime(2024, 1, 15, 10, 5, 30)  # 5.5 minutes later

        trigger = ContextMessage(speaker="them", timestamp=t1, text="Hi")
        response = ContextMessage(speaker="me", timestamp=t2, text="Hello")

        exchange = CandidateExchange(
            trigger_span=[trigger],
            response_span=[response],
            context_window=[],
            chat_id="chat123",
        )

        assert exchange.time_gap_minutes == 5.5

    def test_context_window(self) -> None:
        """Test context window serialization."""
        t0 = datetime(2024, 1, 15, 9, 55, 0)
        t1 = datetime(2024, 1, 15, 10, 0, 0)
        t2 = datetime(2024, 1, 15, 10, 1, 0)

        context = [
            ContextMessage(speaker="me", timestamp=t0, text="Previous message"),
        ]
        trigger = ContextMessage(speaker="them", timestamp=t1, text="Hi")
        response = ContextMessage(speaker="me", timestamp=t2, text="Hello")

        exchange = CandidateExchange(
            trigger_span=[trigger],
            response_span=[response],
            context_window=context,
            chat_id="chat123",
        )

        json_context = exchange.context_to_json()
        assert len(json_context) == 1
        assert json_context[0]["text"] == "Previous message"

    def test_flags_aggregation(self) -> None:
        """Test aggregating flags from spans."""
        trigger1 = ContextMessage(
            speaker="them",
            timestamp=datetime.now(),
            text="Hi",
            flags={"emoji_only"},
        )
        trigger2 = ContextMessage(
            speaker="them",
            timestamp=datetime.now(),
            text="<ATTACHMENT:image>",
            flags={"attachment"},
        )
        response = ContextMessage(
            speaker="me",
            timestamp=datetime.now(),
            text="Nice!",
            flags=set(),
        )

        exchange = CandidateExchange(
            trigger_span=[trigger1, trigger2],
            response_span=[response],
            context_window=[],
            chat_id="chat123",
        )

        assert "emoji_only" in exchange.has_trigger_flags
        assert "attachment" in exchange.has_trigger_flags
        assert len(exchange.has_response_flags) == 0

    def test_primary_msg_ids(self) -> None:
        """Test primary message ID accessors."""
        exchange = CandidateExchange(
            trigger_span=[],
            response_span=[],
            context_window=[],
            chat_id="chat123",
            trigger_msg_ids=[10, 11, 12],
            response_msg_ids=[20, 21],
        )

        assert exchange.primary_trigger_msg_id == 10
        assert exchange.primary_response_msg_id == 20

    def test_empty_spans(self) -> None:
        """Test handling of empty spans."""
        exchange = CandidateExchange(
            trigger_span=[],
            response_span=[],
            context_window=[],
            chat_id="chat123",
        )

        assert exchange.trigger_text == ""
        assert exchange.response_text == ""
        assert exchange.trigger_start_time == datetime.min
        assert exchange.time_gap_minutes == 0.0
        assert exchange.primary_trigger_msg_id is None


class TestExchangeConfig:
    """Tests for ExchangeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ExchangeConfig()
        assert config.time_gap_boundary_minutes == 30.0
        assert config.response_window_minutes == 5.0
        assert config.trigger_window_minutes == 5.0
        assert config.context_window_size == 20
        assert config.max_response_delay_hours == 168.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ExchangeConfig(
            time_gap_boundary_minutes=60.0,
            response_window_minutes=3.0,
            context_window_size=10,
        )
        assert config.time_gap_boundary_minutes == 60.0
        assert config.response_window_minutes == 3.0
        assert config.context_window_size == 10
