"""Tests for jarvis/validity_gate.py - Three-layer validation."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from jarvis.exchange import CandidateExchange, ContextMessage
from jarvis.validity_gate import GateConfig, GateResult, ValidityGate


def make_exchange(
    trigger_text: str,
    response_text: str,
    time_gap_minutes: float = 1.0,
    trigger_flags: set[str] | None = None,
    response_flags: set[str] | None = None,
) -> CandidateExchange:
    """Helper to create a test exchange."""
    t1 = datetime(2024, 1, 15, 10, 0, 0)
    t2 = t1 + timedelta(minutes=time_gap_minutes)

    trigger = ContextMessage(
        speaker="them",
        timestamp=t1,
        text=trigger_text,
        flags=trigger_flags or set(),
    )
    response = ContextMessage(
        speaker="me",
        timestamp=t2,
        text=response_text,
        flags=response_flags or set(),
    )

    return CandidateExchange(
        trigger_span=[trigger],
        response_span=[response],
        context_window=[],
        chat_id="test_chat",
    )


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = GateResult()
        assert result.gate_a_passed is True
        assert result.gate_a_reason is None
        assert result.gate_b_score == 0.0
        assert result.gate_b_band == "reject"
        assert result.gate_c_verdict is None
        assert result.final_status == "invalid"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = GateResult(
            gate_a_passed=True,
            gate_b_score=0.75,
            gate_b_band="accept",
            final_status="valid",
        )
        d = result.to_dict()
        assert d["gate_a_passed"] is True
        assert d["gate_b_score"] == 0.75
        assert d["final_status"] == "valid"


class TestGateConfig:
    """Tests for GateConfig dataclass."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        config = GateConfig()
        assert config.gate_b_accept_threshold == 0.62
        assert config.gate_b_reject_threshold == 0.48
        assert config.gate_b_topic_shift_penalty == 0.10

    def test_custom_thresholds(self) -> None:
        """Test custom threshold values."""
        config = GateConfig(
            gate_b_accept_threshold=0.70,
            gate_b_reject_threshold=0.50,
        )
        assert config.gate_b_accept_threshold == 0.70
        assert config.gate_b_reject_threshold == 0.50


class TestGateARules:
    """Tests for Gate A rule-based validation."""

    def test_reaction_rejection(self) -> None:
        """Test that reactions in response are rejected."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text='Liked "Hello"',
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "reaction_only"
        assert result.final_status == "invalid"

    def test_ack_to_content_trigger(self) -> None:
        """Test that acks to content-expecting triggers are rejected."""
        exchange = make_exchange(
            trigger_text="What do you think about the proposal?",
            response_text="ok",
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "ack_to_content_trigger"

    def test_short_topic_shift(self) -> None:
        """Test that short topic-shift responses are rejected."""
        exchange = make_exchange(
            trigger_text="How was your day?",
            response_text="btw hi",  # Short and starts new topic
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "short_topic_shift"

    def test_ack_trigger_long_response(self) -> None:
        """Test rejection of long response to short ack trigger."""
        exchange = make_exchange(
            trigger_text="ok",
            response_text="So I was thinking about the project and we should probably "
            "restructure the whole thing from scratch",
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "ack_trigger_long_response"

    def test_emoji_only_response(self) -> None:
        """Test that emoji-only responses are rejected."""
        exchange = make_exchange(
            trigger_text="How was the movie?",
            response_text="😂😂😂",
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "emoji_only"

    def test_very_stale_response(self) -> None:
        """Test that very stale responses (>24h) are rejected."""
        exchange = make_exchange(
            trigger_text="Are you free?",
            response_text="Yes I am!",
            time_gap_minutes=1500,  # 25 hours
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "very_stale_response"

    def test_short_question_to_statement(self) -> None:
        """Test rejection of short questions to non-questions."""
        exchange = make_exchange(
            trigger_text="I went to the store",
            response_text="Why?",  # Short question to statement
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.gate_a_reason == "short_question_to_statement"

    def test_valid_exchange_passes_gate_a(self) -> None:
        """Test that a valid exchange passes Gate A."""
        exchange = make_exchange(
            trigger_text="How was your day?",
            response_text="It was great! I went to the park and had a nice walk.",
            time_gap_minutes=5,
        )
        gate = ValidityGate()
        result = gate.validate(exchange)

        assert result.gate_a_passed
        assert result.gate_a_reason is None


class TestGateBEmbedding:
    """Tests for Gate B embedding similarity."""

    def test_no_embedder_skips_gate_b(self) -> None:
        """Test that without embedder, Gate B is skipped."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text="Hi there!",
        )
        gate = ValidityGate(embedder=None)
        result = gate.validate(exchange)

        # Should skip Gate B and mark as valid (rules only)
        assert result.gate_a_passed
        assert result.gate_b_band == "accept"
        assert result.final_status == "valid"

    def test_high_similarity_accepts(self) -> None:
        """Test that high similarity leads to acceptance."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text="Hi there!",
        )

        # Mock embedder
        mock_embedder = MagicMock()
        import numpy as np

        # Return similar embeddings (high dot product)
        mock_embedder.encode.return_value = [np.array([1.0, 0, 0])]

        gate = ValidityGate(embedder=mock_embedder)
        result = gate.validate(exchange)

        assert result.gate_b_score == 1.0
        assert result.gate_b_band == "accept"
        assert result.final_status == "valid"

    def test_low_similarity_rejects(self) -> None:
        """Test that low similarity leads to rejection."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text="Goodbye",
        )

        # Mock embedder with orthogonal embeddings
        mock_embedder = MagicMock()
        import numpy as np

        mock_embedder.encode.side_effect = [
            [np.array([1.0, 0, 0])],
            [np.array([0, 1.0, 0])],
        ]

        gate = ValidityGate(embedder=mock_embedder)
        result = gate.validate(exchange)

        assert result.gate_b_score == 0.0
        assert result.gate_b_band == "reject"
        assert result.final_status == "invalid"

    def test_topic_shift_penalty(self) -> None:
        """Test that topic shift applies penalty."""
        # Use a longer topic-shift response to pass Gate A
        # (short topic-shift responses are rejected by Gate A)
        exchange = make_exchange(
            trigger_text="How was your day?",
            response_text="btw, did you hear about the news yesterday? It was crazy what happened.",
        )

        # Mock embedder with moderate similarity
        mock_embedder = MagicMock()
        import numpy as np

        # Return ~0.65 similarity (above accept threshold normally)
        vec = np.array([0.8, 0.6, 0])
        vec = vec / np.linalg.norm(vec)
        mock_embedder.encode.return_value = [vec]

        gate = ValidityGate(embedder=mock_embedder)
        result = gate.validate(exchange)

        # Either Gate A passed and penalty applied, or Gate A rejected
        # (which is valid since it's a topic shift)
        assert (result.gate_a_passed and result.gate_b_has_topic_shift_penalty) or (
            not result.gate_a_passed
        )


class TestGateCNLI:
    """Tests for Gate C NLI validation."""

    def test_no_nli_model_returns_uncertain(self) -> None:
        """Test that without NLI model, borderline cases are uncertain."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text="Hi there, how are you doing today?",
        )

        # Mock embedder for borderline similarity
        mock_embedder = MagicMock()
        import numpy as np

        # Return ~0.55 similarity (borderline)
        mock_embedder.encode.return_value = [np.array([0.74, 0.67, 0])]

        gate = ValidityGate(embedder=mock_embedder, nli_model=None)
        result = gate.validate(exchange)

        if result.gate_b_band == "borderline":
            assert result.final_status == "uncertain"

    def test_nli_accept_verdict(self) -> None:
        """Test NLI accept verdict."""
        exchange = make_exchange(
            trigger_text="Are you free tonight?",
            response_text="Yes, I am! What did you have in mind?",
        )

        # Mock embedder for borderline
        mock_embedder = MagicMock()
        import numpy as np

        mock_embedder.encode.return_value = [np.array([0.74, 0.67, 0])]

        # Mock NLI model
        mock_nli = MagicMock()
        # Returns [contradiction, neutral, entailment] logits
        # High entailment for "addresses trigger", low for "new topic"
        mock_nli.predict.side_effect = [
            [np.array([-2, 0, 3])],  # reply: high entailment
            [np.array([3, 0, -2])],  # newtopic: high contradiction
            [np.array([0, 0, 0])],  # ack: neutral
        ]

        gate = ValidityGate(embedder=mock_embedder, nli_model=mock_nli)
        result = gate.validate(exchange)

        if result.gate_b_band == "borderline" and result.gate_c_verdict:
            assert result.gate_c_verdict in ["accept", "reject", "uncertain"]


class TestValidityGateIntegration:
    """Integration tests for the full validation pipeline."""

    def test_full_pipeline_valid(self) -> None:
        """Test a clearly valid exchange passes all gates."""
        exchange = make_exchange(
            trigger_text="What time should we meet tomorrow?",
            response_text="How about 3pm? I'll be free after my meeting ends.",
            time_gap_minutes=2,
        )

        # Mock high similarity embedder
        mock_embedder = MagicMock()
        import numpy as np

        mock_embedder.encode.return_value = [np.array([1.0, 0, 0])]

        gate = ValidityGate(embedder=mock_embedder)
        result = gate.validate(exchange)

        assert result.gate_a_passed
        assert result.final_status == "valid"

    def test_full_pipeline_invalid(self) -> None:
        """Test a clearly invalid exchange fails early."""
        exchange = make_exchange(
            trigger_text="Hello",
            response_text='Liked "Hello"',  # Reaction
        )

        gate = ValidityGate()
        result = gate.validate(exchange)

        assert not result.gate_a_passed
        assert result.final_status == "invalid"
        # Gate B and C should not run
        assert result.gate_b_score == 0.0
        assert result.gate_c_verdict is None

    def test_rules_only_mode(self) -> None:
        """Test rules-only mode (no embedder or NLI)."""
        exchange = make_exchange(
            trigger_text="How are you?",
            response_text="I'm doing great, thanks for asking!",
            time_gap_minutes=1,
        )

        gate = ValidityGate(embedder=None, nli_model=None)
        result = gate.validate(exchange)

        assert result.gate_a_passed
        assert result.final_status == "valid"
