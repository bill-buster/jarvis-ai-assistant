"""Validity Gate - Three-layer validation for candidate exchanges.

Gate A: Fast rule-based rejection (lexical features, message properties)
Gate B: Embedding similarity with topic-shift penalty
Gate C: NLI cross-encoder for borderline cases

Usage:
    from jarvis.validity_gate import ValidityGate, GateResult

    gate = ValidityGate(embedder)  # NLI model optional
    result = gate.validate(exchange)
    if result.final_status == "valid":
        # Store the pair
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from jarvis.exchange import CandidateExchange
from jarvis.text_normalizer import (
    extract_text_features,
    is_acknowledgment_only,
    is_emoji_only,
    is_question,
    is_reaction,
    starts_new_topic,
    trigger_expects_content,
)

if TYPE_CHECKING:
    from jarvis.embedding_adapter import UnifiedEmbedder

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result from running all validity gates.

    Attributes:
        gate_a_passed: Whether Gate A (rules) passed.
        gate_a_reason: Why Gate A rejected (if rejected).
        gate_b_score: Embedding similarity score (0.0-1.0).
        gate_b_band: Gate B decision band.
        gate_b_has_topic_shift_penalty: Whether topic-shift penalty was applied.
        gate_c_verdict: NLI verdict (if Gate C was run).
        gate_c_scores: Raw NLI scores (if Gate C was run).
        final_status: Final validity status.
    """

    gate_a_passed: bool = True
    gate_a_reason: str | None = None
    gate_b_score: float = 0.0
    gate_b_band: Literal["accept", "borderline", "reject"] = "reject"
    gate_b_has_topic_shift_penalty: bool = False
    gate_c_verdict: Literal["accept", "reject", "uncertain"] | None = None
    gate_c_scores: dict[str, float] = field(default_factory=dict)
    final_status: Literal["valid", "invalid", "uncertain"] = "invalid"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "gate_a_passed": self.gate_a_passed,
            "gate_a_reason": self.gate_a_reason,
            "gate_b_score": self.gate_b_score,
            "gate_b_band": self.gate_b_band,
            "gate_b_has_topic_shift_penalty": self.gate_b_has_topic_shift_penalty,
            "gate_c_verdict": self.gate_c_verdict,
            "gate_c_scores": self.gate_c_scores,
            "final_status": self.final_status,
        }


@dataclass
class GateConfig:
    """Configuration for validity gates.

    Attributes:
        gate_b_accept_threshold: Similarity >= this is accepted.
        gate_b_reject_threshold: Similarity < this is rejected.
        gate_b_topic_shift_penalty: Penalty applied when topic-shift detected.
        gate_c_reply_accept: p_reply >= this to accept in Gate C.
        gate_c_newtopic_reject: p_newtopic <= this to accept in Gate C.
        gate_c_newtopic_accept_reject: p_newtopic >= this to reject in Gate C.
        gate_c_reply_accept_reject: p_reply <= this to reject in Gate C.
        short_text_threshold: Char count below this adjusts thresholds down.
        short_text_adjustment: Amount to lower thresholds for short text.
    """

    gate_b_accept_threshold: float = 0.62
    gate_b_reject_threshold: float = 0.48
    gate_b_topic_shift_penalty: float = 0.10
    gate_c_reply_accept: float = 0.60
    gate_c_newtopic_reject: float = 0.35
    gate_c_newtopic_accept_reject: float = 0.60
    gate_c_reply_accept_reject: float = 0.40
    short_text_threshold: int = 30
    short_text_adjustment: float = 0.05


class ValidityGate:
    """Three-layer validation for candidate exchanges.

    Gate A: Pure rule-based rejection (no embeddings)
    Gate B: Embedding similarity with topic-shift penalty
    Gate C: NLI cross-encoder for borderline cases (optional)
    """

    def __init__(
        self,
        embedder: "UnifiedEmbedder | None" = None,
        nli_model: Any = None,
        config: GateConfig | None = None,
    ) -> None:
        """Initialize validity gate.

        Args:
            embedder: Embedder for Gate B similarity. If None, Gate B is skipped.
            nli_model: Cross-encoder NLI model for Gate C. If None, Gate C is skipped.
            config: Configuration for gate thresholds.
        """
        self.embedder = embedder
        self.nli_model = nli_model
        self.config = config or GateConfig()

    def validate(self, exchange: CandidateExchange) -> GateResult:
        """Run all gates and return combined result.

        Args:
            exchange: Candidate exchange to validate.

        Returns:
            GateResult with all gate outcomes and final status.
        """
        result = GateResult()

        # Gate A: Rules (always runs)
        gate_a_passed, gate_a_reason = self._gate_a_rules(exchange)
        result.gate_a_passed = gate_a_passed
        result.gate_a_reason = gate_a_reason

        if not gate_a_passed:
            result.final_status = "invalid"
            return result

        # Gate B: Embedding similarity (requires embedder)
        if self.embedder is not None:
            gate_b_score, gate_b_band, has_penalty = self._gate_b_embedding(exchange)
            result.gate_b_score = gate_b_score
            result.gate_b_band = gate_b_band
            result.gate_b_has_topic_shift_penalty = has_penalty

            if gate_b_band == "reject":
                result.final_status = "invalid"
                return result
            elif gate_b_band == "accept":
                result.final_status = "valid"
                return result
            # gate_b_band == "borderline" -> continue to Gate C
        else:
            # No embedder, skip to valid (rely on rules only)
            result.gate_b_band = "accept"
            result.final_status = "valid"
            return result

        # Gate C: NLI (requires nli_model, only for borderline)
        if self.nli_model is not None:
            gate_c_verdict, gate_c_scores = self._gate_c_nli(exchange)
            result.gate_c_verdict = gate_c_verdict
            result.gate_c_scores = gate_c_scores

            if gate_c_verdict == "accept":
                result.final_status = "valid"
            elif gate_c_verdict == "reject":
                result.final_status = "invalid"
            else:
                result.final_status = "uncertain"
        else:
            # No NLI model, borderline -> uncertain
            result.final_status = "uncertain"

        return result

    def _gate_a_rules(self, exchange: CandidateExchange) -> tuple[bool, str | None]:
        """Gate A: Pure rule-based rejection.

        Uses only lexical features, message properties, and structural checks.
        No embeddings involved to keep this gate deterministic and fast.

        Rules:
        1. Response is reaction-only
        2. Response is ack-only AND trigger expects content
        3. Response has topic-shift markers (btw, anyway) AND is short
        4. Trigger is very short ack AND response is long content-heavy
        5. Response is emoji-only
        6. Time gap > 24 hours (very stale response)
        """
        trigger_text = exchange.trigger_text
        response_text = exchange.response_text
        time_gap = exchange.time_gap_minutes

        # 1. Response is reaction-only (tapback)
        if is_reaction(response_text):
            return False, "reaction_only"

        # Also check individual response messages
        for msg in exchange.response_span:
            if "reaction" in msg.flags:
                return False, "reaction_only"

        # 2. Response is ack-only AND trigger expects content
        if is_acknowledgment_only(response_text) and trigger_expects_content(trigger_text):
            return False, "ack_to_content_trigger"

        # 3. Response starts with topic-shift marker AND is short
        # (Short topic-shift responses are likely unrelated to trigger)
        if starts_new_topic(response_text):
            response_words = len(response_text.split())
            if response_words <= 10:
                return False, "short_topic_shift"

        # 4. Trigger is very short ack AND response is long content-heavy
        # (The long response is probably to something else not captured)
        trigger_words = len(trigger_text.split())
        response_words = len(response_text.split())
        if trigger_words <= 2 and is_acknowledgment_only(trigger_text):
            if response_words >= 15:
                return False, "ack_trigger_long_response"

        # 5. Response is emoji-only
        if is_emoji_only(response_text):
            return False, "emoji_only"

        # 6. Time gap > 24 hours (1440 minutes) - very stale
        if time_gap > 1440:
            return False, "very_stale_response"

        # 7. Check for question response to non-question statement
        # (Response asking question when trigger didn't - may be unrelated)
        if is_question(response_text) and not is_question(trigger_text):
            # Only reject if response is short (short questions often unrelated)
            if response_words <= 5:
                return False, "short_question_to_statement"

        return True, None

    def _gate_b_embedding(
        self, exchange: CandidateExchange
    ) -> tuple[float, Literal["accept", "borderline", "reject"], bool]:
        """Gate B: Embedding similarity with topic-shift penalty.

        Computes cosine similarity between trigger and response embeddings.
        Applies penalty if response has topic-shift markers.

        Returns:
            Tuple of (score, band, has_topic_shift_penalty).
        """
        trigger_text = exchange.trigger_text
        response_text = exchange.response_text

        if not self.embedder or not trigger_text or not response_text:
            return 0.0, "reject", False

        # Get embeddings
        try:
            trigger_emb = self.embedder.encode([trigger_text])[0]
            response_emb = self.embedder.encode([response_text])[0]

            # Compute cosine similarity (embeddings are L2-normalized)
            import numpy as np

            similarity = float(np.dot(trigger_emb, response_emb))
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return 0.0, "reject", False

        # Check for topic-shift and apply penalty
        has_penalty = False
        if starts_new_topic(response_text):
            similarity -= self.config.gate_b_topic_shift_penalty
            has_penalty = True

        # Adjust thresholds for short text (short pairs naturally have lower similarity)
        accept_threshold = self.config.gate_b_accept_threshold
        reject_threshold = self.config.gate_b_reject_threshold

        trigger_features = extract_text_features(trigger_text)
        response_features = extract_text_features(response_text)

        if (
            trigger_features.char_count < self.config.short_text_threshold
            or response_features.char_count < self.config.short_text_threshold
        ):
            accept_threshold -= self.config.short_text_adjustment
            reject_threshold -= self.config.short_text_adjustment

        # Determine band
        if similarity >= accept_threshold:
            return similarity, "accept", has_penalty
        elif similarity < reject_threshold:
            return similarity, "reject", has_penalty
        else:
            return similarity, "borderline", has_penalty

    def _gate_c_nli(
        self, exchange: CandidateExchange
    ) -> tuple[Literal["accept", "reject", "uncertain"], dict[str, float]]:
        """Gate C: NLI cross-encoder for borderline cases.

        Uses 3-way scoring with hypotheses:
        - "The response directly addresses the trigger message."
        - "The response starts an unrelated topic."
        - "The response is just an acknowledgment."

        Decision rule (robust to NLI model quirks):
        - accept if p_reply >= 0.60 AND p_newtopic <= 0.35
        - reject if p_newtopic >= 0.60 AND p_reply <= 0.40
        - otherwise uncertain
        """
        trigger_text = exchange.trigger_text
        response_text = exchange.response_text

        if not self.nli_model:
            return "uncertain", {}

        # Construct hypotheses
        hypotheses = {
            "reply": "The response directly addresses the trigger message.",
            "newtopic": "The response starts an unrelated topic.",
            "ack": "The response is just an acknowledgment.",
        }

        # Format for NLI: premise is trigger+response, hypothesis varies
        premise = f"Trigger: {trigger_text}\nResponse: {response_text}"

        scores: dict[str, float] = {}
        try:
            for key, hypothesis in hypotheses.items():
                # Cross-encoder expects (premise, hypothesis) pairs
                # Returns logits or probabilities depending on model
                score = self._get_nli_entailment_score(premise, hypothesis)
                scores[key] = score
        except Exception as e:
            logger.warning("NLI inference failed: %s", e)
            return "uncertain", {}

        p_reply = scores.get("reply", 0.0)
        p_newtopic = scores.get("newtopic", 0.0)

        # Decision rule (robust version)
        accept_threshold = self.config.gate_c_reply_accept
        newtopic_threshold = self.config.gate_c_newtopic_reject
        if p_reply >= accept_threshold and p_newtopic <= newtopic_threshold:
            return "accept", scores
        elif (
            p_newtopic >= self.config.gate_c_newtopic_accept_reject
            and p_reply <= self.config.gate_c_reply_accept_reject
        ):
            return "reject", scores
        else:
            return "uncertain", scores

    def _get_nli_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Get entailment probability from NLI model.

        This is a placeholder that works with cross-encoder/nli-deberta-v3-small.
        Override or adapt for other NLI models.
        """
        if self.nli_model is None:
            return 0.0

        try:
            # cross-encoder models return [contradiction, neutral, entailment] logits
            # We want the entailment probability
            import numpy as np

            logits = self.nli_model.predict([(premise, hypothesis)])[0]

            # Check if logits is a single value (some models) or array
            if isinstance(logits, (float, int)):
                return float(logits)

            # Convert logits to probabilities via softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Return entailment probability (index 2 for contradiction/neutral/entailment)
            return float(probs[2])
        except Exception as e:
            logger.warning("NLI score computation failed: %s", e)
            return 0.0


def load_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small") -> Any:
    """Load a cross-encoder NLI model for Gate C.

    Args:
        model_name: HuggingFace model name.

    Returns:
        CrossEncoder model or None if loading fails.
    """
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        logger.info("Loaded NLI model: %s", model_name)
        return model
    except ImportError:
        logger.warning("sentence-transformers not installed, Gate C disabled")
        return None
    except Exception as e:
        logger.warning("Failed to load NLI model %s: %s", model_name, e)
        return None
