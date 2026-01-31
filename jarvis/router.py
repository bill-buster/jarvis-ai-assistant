"""Reply Router - Decision logic for template, generation, or clarification routing.

Routes incoming messages to the appropriate response strategy:
- Template (high confidence): Direct template response when similarity >= TEMPLATE_THRESHOLD
- Generate (medium confidence): LLM generation with context and few-shot examples
- Clarify (low confidence): Request more information when context is insufficient

The router uses FAISS vector similarity to match incoming messages against
historical trigger patterns, enabling fast, personalized response selection.

Usage:
    from jarvis.router import get_reply_router, ReplyRouter

    router = get_reply_router()
    result = router.route(
        incoming="Want to grab lunch?",
        contact_id=1,
        thread=["Hey!", "What's up?"],
    )

    if result['type'] == 'template':
        print(f"Quick response: {result['response']}")
    elif result['type'] == 'generated':
        print(f"AI response: {result['response']}")
    else:  # clarify
        print(f"Need more info: {result['response']}")
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.config import get_config
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.embedding_adapter import CachedEmbedder, get_embedder
from jarvis.errors import ErrorCode, JarvisError
from jarvis.intent import IntentClassifier, IntentType, get_intent_classifier
from jarvis.message_classifier import (
    ContextRequirement,
    MessageClassifier,
    MessageType,
    get_message_classifier,
)
from jarvis.metrics_router import RoutingMetrics, get_routing_metrics_store, hash_query
from jarvis.metrics_validation import get_audit_logger, get_sampling_validator
from jarvis.quality_metrics import score_response_coherence
from jarvis.relationships import (
    MIN_MESSAGES_FOR_PROFILE,
    generate_style_guide,
    load_profile,
)

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.index import TriggerIndexSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Similarity thresholds for routing decisions
# Updated thresholds to force more LLM generation for better quality
TEMPLATE_THRESHOLD = 0.90  # Very confident -> use template directly
CONTEXT_THRESHOLD = 0.70  # Below this -> ask for clarification
GENERATE_THRESHOLD = 0.50  # Minimum for attempting generation

# Coherence threshold for filtering responses
COHERENCE_THRESHOLD = 0.5  # Minimum coherence score to use a template response

# Response variety
MAX_TEMPLATE_RESPONSES = 5  # Max responses to choose from for variety
MIN_RESPONSE_SIMILARITY = 0.6  # Responses must be somewhat similar to pick randomly

# Simple acknowledgments that should be handled directly
SIMPLE_ACKNOWLEDGMENTS = frozenset(
    {
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
        "sure",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "np",
        "cool",
        "nice",
        "good",
        "great",
        "awesome",
        "alright",
        "got it",
        "lol",
        "haha",
        "bye",
        "later",
    }
)

# Context-dependent patterns that ALWAYS need clarification
# These should NEVER use template responses even with high similarity
# because the answer depends on current context, not past responses
CONTEXT_DEPENDENT_PATTERNS = frozenset(
    {
        # Time questions - answer depends on WHAT event
        "what time",
        "what time?",
        "when?",
        "when",
        "how long",
        # Location questions - answer depends on WHAT place
        "where?",
        "where",
        "which place",
        # Reference questions - answer depends on WHAT thing
        "which one",
        "which one?",
        "what?",
        "what",
        # Vague confirmations that need context
        "you sure",
        "you sure?",
        "really",
        "really?",
    }
)

# Questions that require USER's personal input - system cannot answer these
# Even with high similarity, system doesn't know user's current schedule/plans/opinions
USER_INPUT_REQUIRED_STARTERS = (
    # Commitment questions - system can't commit for user
    "are you coming",
    "are you going",
    "are you free",
    "can you come",
    "can you make it",
    "will you be",
    "will you come",
    "are you available",
    "can you do",
    "could you",
    "would you",
    "do you want",
    # Personal fact questions - system doesn't know user's current state
    "where are you",
    "what are you doing",
    "how are you",
    "what's your",
    # Opinion questions - system doesn't know user's opinion
    "what do you think",
    "do you like",
    "how do you feel",
    "what's your opinion",
    # Yes/no about user's actions/plans
    "did you",
    "have you",
    "do you have",
)


# =============================================================================
# Exceptions
# =============================================================================


class RouterError(JarvisError):
    """Raised when routing operations fail."""

    default_message = "Router operation failed"
    default_code = ErrorCode.UNKNOWN


class IndexNotAvailableError(RouterError):
    """Raised when FAISS index is not available."""

    default_message = "FAISS index not available. Run 'jarvis db build-index' first."


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RouteResult:
    """Result of routing an incoming message.

    Attributes:
        response: The response text.
        type: Response type ('template', 'generated', 'clarify').
        confidence: Confidence level ('high', 'medium', 'low').
        similarity_score: Best similarity score from FAISS search.
        cluster_name: Name of matched cluster (for templates).
        contact_style: Style notes for the contact.
        similar_triggers: List of similar past triggers found.
    """

    response: str
    type: str  # 'template', 'generated', 'clarify'
    confidence: str  # 'high', 'medium', 'low'
    similarity_score: float = 0.0
    cluster_name: str | None = None
    contact_style: str | None = None
    similar_triggers: list[str] | None = None


# =============================================================================
# Reply Router
# =============================================================================


class ReplyRouter:
    """Routes incoming messages to template, LLM generation, or clarification.

    Implements a three-tier routing strategy:
    1. Template (similarity >= template threshold): Return cached response instantly
    2. Generate (similarity >= generate threshold): Use LLM with similar examples
    3. Clarify (below thresholds): Ask user for more context

    Thread Safety:
        This class is thread-safe for routing operations.
        Index and generator initialization uses lazy loading.
    """

    def __init__(
        self,
        db: JarvisDB | None = None,
        index_searcher: TriggerIndexSearcher | None = None,
        generator: MLXGenerator | None = None,
        intent_classifier: IntentClassifier | None = None,
        message_classifier: MessageClassifier | None = None,
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            db: Database instance for contacts and pairs. Uses default if None.
            index_searcher: FAISS index searcher. Created lazily if None.
            generator: MLX generator for LLM responses. Created lazily if None.
            intent_classifier: Intent classifier for routing decisions. Created lazily if None.
            message_classifier: Message classifier for type/context analysis.
                Created lazily if None.
            imessage_reader: iMessage reader for fetching conversation history.
                Created lazily if None.
        """
        self._db = db
        self._index_searcher = index_searcher
        self._generator = generator
        self._intent_classifier = intent_classifier
        self._message_classifier = message_classifier
        self._imessage_reader = imessage_reader
        self._lock = threading.Lock()

    @property
    def db(self) -> JarvisDB:
        """Get or create the database instance."""
        if self._db is None:
            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def index_searcher(self) -> TriggerIndexSearcher:
        """Get or create the FAISS index searcher."""
        if self._index_searcher is None:
            with self._lock:
                if self._index_searcher is None:
                    from jarvis.index import TriggerIndexSearcher

                    self._index_searcher = TriggerIndexSearcher(self.db)
        return self._index_searcher

    @property
    def generator(self) -> MLXGenerator:
        """Get or create the MLX generator."""
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from models import get_generator

                    self._generator = get_generator(skip_templates=True)
        return self._generator

    @property
    def intent_classifier(self) -> IntentClassifier:
        """Get or create the intent classifier."""
        if self._intent_classifier is None:
            with self._lock:
                if self._intent_classifier is None:
                    self._intent_classifier = get_intent_classifier()
        return self._intent_classifier

    @property
    def message_classifier(self) -> MessageClassifier:
        """Get or create the message classifier."""
        if self._message_classifier is None:
            with self._lock:
                if self._message_classifier is None:
                    self._message_classifier = get_message_classifier()
        return self._message_classifier

    @property
    def imessage_reader(self) -> ChatDBReader | None:
        """Get or create the iMessage reader for fetching conversation history."""
        if self._imessage_reader is None:
            with self._lock:
                if self._imessage_reader is None:
                    try:
                        from integrations.imessage.reader import ChatDBReader

                        self._imessage_reader = ChatDBReader()
                    except Exception as e:
                        logger.warning("Could not initialize iMessage reader: %s", e)
                        return None
        return self._imessage_reader

    def _fetch_conversation_context(self, chat_id: str, limit: int = 10) -> list[str]:
        """Fetch recent conversation history from iMessage.

        Args:
            chat_id: The chat/conversation ID.
            limit: Maximum number of messages to fetch.

        Returns:
            List of formatted message strings for context.
        """
        if not self.imessage_reader:
            return []

        try:
            messages = self.imessage_reader.get_messages(chat_id, limit=limit)
            if not messages:
                return []

            # Format messages for context (newest first, so reverse for chronological)
            context_messages = []
            for msg in reversed(messages):
                sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender or "Them")
                text = msg.text or ""
                if text:
                    context_messages.append(f"[{sender}]: {text}")

            return context_messages

        except Exception as e:
            logger.warning("Failed to fetch conversation context: %s", e)
            return []

    def _get_thresholds(self) -> dict[str, float]:
        """Get routing thresholds with optional A/B overrides."""
        config = get_config()
        routing = config.routing

        if routing.ab_test_group in routing.ab_test_thresholds:
            group_thresholds = routing.ab_test_thresholds[routing.ab_test_group]
            return {
                "template": group_thresholds.get("template", routing.template_threshold),
                "context": group_thresholds.get("context", routing.context_threshold),
                "generate": group_thresholds.get("generate", routing.generate_threshold),
            }

        return {
            "template": routing.template_threshold,
            "context": routing.context_threshold,
            "generate": routing.generate_threshold,
        }

    def _normalize_routing_decision(self, result_type: str) -> str:
        if result_type == "generated":
            return "generate"
        if result_type == "template":
            return "template"
        return "clarify"

    def _record_routing_metrics(
        self,
        incoming: str,
        decision: str,
        similarity_score: float,
        latency_ms: dict[str, float],
        cached_embedder: CachedEmbedder,
        faiss_candidates: int,
        model_loaded: bool,
    ) -> None:
        try:
            metrics = RoutingMetrics(
                timestamp=time.time(),
                query_hash=hash_query(incoming),
                latency_ms=latency_ms,
                embedding_computations=cached_embedder.embedding_computations,
                faiss_candidates=faiss_candidates,
                routing_decision=decision,
                similarity_score=similarity_score,
                cache_hit=cached_embedder.cache_hit,
                model_loaded=model_loaded,
            )
            get_routing_metrics_store().record(metrics)

            # Validation layer: audit logging
            try:
                get_audit_logger().log_request(metrics)
            except Exception as e:
                logger.debug("Audit logging failed: %s", e)

            # Validation layer: sampling validator
            try:
                validator = get_sampling_validator()
                if validator.should_sample():
                    validator.validate_sample(
                        query=incoming,
                        decision=decision,
                        similarity=similarity_score,
                        latency_ms=latency_ms,
                        computations=cached_embedder.embedding_computations,
                        cache_hit=cached_embedder.cache_hit,
                        model_loaded=model_loaded,
                    )
            except Exception as e:
                logger.debug("Sampling validation failed: %s", e)

        except Exception as e:
            logger.debug("Routing metrics write failed: %s", e)

    def _is_simple_acknowledgment(self, text: str) -> bool:
        """Check if the message is a simple acknowledgment.

        Simple acknowledgments like "ok", "thanks", "yes" are too
        context-dependent for template matching and should be handled
        directly with generic responses.

        Args:
            text: The incoming message text.

        Returns:
            True if the message is a simple acknowledgment.
        """
        normalized = text.lower().strip()
        # Remove punctuation for matching
        normalized = normalized.rstrip("!.?")
        return normalized in SIMPLE_ACKNOWLEDGMENTS

    def _is_context_dependent(self, text: str) -> bool:
        """Check if the message is context-dependent or requires user input.

        Two categories that should NOT use template responses:
        1. Context-dependent: "what time?", "where?" - need to know WHAT we're discussing
        2. User-input-required: "are you coming?", "can you?" - system can't answer for user

        Args:
            text: The incoming message text.

        Returns:
            True if the message requires current context or user's personal input.
        """
        normalized = text.lower().strip()
        # Remove punctuation for matching
        normalized_no_punct = normalized.rstrip("!.?")

        # Check exact matches for context-dependent patterns
        if normalized_no_punct in CONTEXT_DEPENDENT_PATTERNS:
            return True

        # Check if message starts with context-dependent patterns
        context_starters = ("what time", "when ", "where ", "which ")
        if any(normalized.startswith(s) for s in context_starters):
            return True

        # Check if message requires user's personal input
        # These are questions only the USER can answer (commitments, opinions, facts)
        if any(normalized.startswith(s) for s in USER_INPUT_REQUIRED_STARTERS):
            return True

        return False

    def _should_generate_after_acknowledgment(
        self,
        incoming: str,
        contact: Contact | None,
        thread: list[str] | None,
    ) -> bool:
        """Check if user typically provides substantive info after acknowledgments.

        Returns True if we should generate instead of using canned acknowledgment.
        This helps avoid responding with just "Got it!" when the user typically
        follows acknowledgments with substantive information.

        Args:
            incoming: The incoming acknowledgment message.
            contact: Optional contact for pattern lookup.
            thread: Optional conversation thread context.

        Returns:
            True if generation is recommended over canned acknowledgment.
        """
        # If we have active thread context, likely mid-conversation
        if thread and len(thread) >= 2:
            # Check if previous message was a question - if so, definitely generate
            if thread:
                last_thread_msg = thread[-1] if thread else ""
                if "?" in last_thread_msg:
                    logger.debug("Acknowledgment follows question, generating substantive response")
                    return True
            return True

        # Check contact's historical pattern
        if contact and contact.id:
            try:
                ack_pairs = self.db.get_pairs_by_trigger_pattern(
                    contact_id=contact.id,
                    pattern_type="acknowledgment",
                    limit=10,
                )
                if ack_pairs:
                    # If user's actual responses were substantive (>15 chars avg, lowered from 30),
                    # we should generate rather than use canned response
                    avg_response_len = sum(len(p.response_text) for p in ack_pairs) / len(ack_pairs)
                    if avg_response_len > 15:
                        logger.debug(
                            "Contact %s has avg ack response length %.1f, using generation",
                            contact.display_name,
                            avg_response_len,
                        )
                        return True
            except Exception as e:
                logger.debug("Could not check ack patterns: %s", e)

        return False

    def _generic_acknowledgment_response(
        self,
        incoming: str,
        contact: Contact | None,
    ) -> dict[str, Any]:
        """Generate a generic acknowledgment response.

        For simple acknowledgments (ok, thanks, yes), we return
        a simple acknowledgment rather than searching the index.

        Args:
            incoming: The incoming acknowledgment.
            contact: Optional contact for personalization.

        Returns:
            Routing result dict with acknowledgment response.
        """
        incoming_lower = incoming.lower().strip()

        # Map acknowledgments to appropriate responses
        if incoming_lower in ("thanks", "thank you", "thx", "ty"):
            responses = ["You're welcome!", "No problem!", "Anytime!", "Of course!"]
        elif incoming_lower in ("ok", "okay", "k", "kk", "got it", "alright"):
            responses = ["👍", "Sounds good!", "Great!", "Perfect!"]
        elif incoming_lower in ("yes", "yeah", "yep", "yup", "sure"):
            responses = ["Great!", "Awesome!", "Perfect!", "👍"]
        elif incoming_lower in ("no", "nope", "nah"):
            responses = ["No worries!", "Okay!", "Got it!", "Understood!"]
        elif incoming_lower in ("bye", "later"):
            responses = ["Bye!", "Talk later!", "See you!", "👋"]
        else:
            responses = ["👍", "Got it!", "Okay!"]

        selected = random.choice(responses)

        return {
            "type": "acknowledgment",
            "response": selected,
            "confidence": "high",
            "similarity_score": 1.0,
            "contact_style": contact.style_notes if contact else None,
            "reason": "simple_acknowledgment",
        }

    def _is_professional_response(self, response: str) -> bool:
        """Check if a response is professional in tone.

        Args:
            response: The response text to check.

        Returns:
            True if the response appears professional.
        """
        response_lower = response.lower()

        # Unprofessional indicators
        unprofessional = [
            "lol",
            "haha",
            "lmao",
            "omg",
            "wtf",
            "bruh",
            "dude",
            "bro",
            "🤣",
            "😂",
            "😝",
            "🙄",
            "💀",
        ]

        return not any(term in response_lower for term in unprofessional)

    def route(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        """Route an incoming message to the appropriate response strategy.

        Uses both intent classification and message classification:
        - Message classifier determines type (question, statement, acknowledgment, etc.)
        - Intent classifier determines user intent (reply, summarize, search, etc.)
        - Simple acknowledgments/reactions are handled directly
        - Messages needing clarification are flagged appropriately
        - Other messages go through the template/generate/clarify pipeline

        Args:
            incoming: The incoming message text to respond to.
            contact_id: Optional contact ID for personalization.
            thread: Optional list of recent messages for context.
            chat_id: Optional chat ID for context lookup.

        Returns:
            Dict with routing result containing:
            - type: 'template', 'generated', 'clarify', or 'acknowledgment'
            - response: The response text
            - confidence: 'high', 'medium', or 'low'
            - Additional metadata (cluster_name, similarity_score, message_type, etc.)
        """
        routing_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
        cached_embedder = CachedEmbedder(get_embedder())

        def record_and_return(
            result: dict[str, Any],
            similarity_score: float,
            faiss_candidates: int = 0,
            model_loaded: bool = False,
            decision: str | None = None,
        ) -> dict[str, Any]:
            latency_ms["total"] = (time.perf_counter() - routing_start) * 1000
            routing_decision = decision or self._normalize_routing_decision(
                result.get("type", "clarify")
            )
            self._record_routing_metrics(
                incoming=incoming,
                decision=routing_decision,
                similarity_score=similarity_score,
                latency_ms=latency_ms,
                cached_embedder=cached_embedder,
                faiss_candidates=faiss_candidates,
                model_loaded=model_loaded,
            )
            return result

        if not incoming or not incoming.strip():
            result = self._clarify_response(
                "I received an empty message. Could you tell me what you need?",
                reason="empty_message",
            )
            return record_and_return(result, similarity_score=0.0, decision="clarify")

        # Step 0: Get contact info for personalization (needed early for acknowledgments)
        contact = None
        if contact_id:
            contact = self.db.get_contact(contact_id)
        elif chat_id:
            contact = self.db.get_contact_by_chat_id(chat_id)

        # Step 1: Classify message type using the new MessageClassifier
        try:
            msg_start = time.perf_counter()
            msg_classification = self.message_classifier.classify(
                incoming,
                embedder=cached_embedder,
            )
            latency_ms["message_classify"] = (time.perf_counter() - msg_start) * 1000
            logger.debug(
                "Message classified as %s (confidence: %.3f, method: %s)",
                msg_classification.message_type.value,
                msg_classification.type_confidence,
                msg_classification.classification_method,
            )
        except Exception as e:
            logger.warning("Message classification failed: %s", e)
            msg_classification = None

        # Step 2: Classify intent (for routing decisions)
        try:
            intent_start = time.perf_counter()
            intent_result = self.intent_classifier.classify(
                incoming,
                embedder=cached_embedder,
            )
            latency_ms["intent_classify"] = (time.perf_counter() - intent_start) * 1000
            logger.debug(
                "Intent classified as %s (confidence: %.3f)",
                intent_result.intent.value,
                intent_result.confidence,
            )
        except Exception as e:
            logger.warning("Intent classification failed: %s", e)
            intent_result = None

        # Step 3: Handle acknowledgments and reactions directly (no FAISS search needed)
        if msg_classification:
            # Acknowledgments - check if we should generate instead
            if msg_classification.message_type == MessageType.ACKNOWLEDGMENT:
                # Check if user typically follows acks with substantive info
                if self._should_generate_after_acknowledgment(incoming, contact, thread):
                    logger.debug("Acknowledgment but context suggests generation needed")
                    # Fall through to FAISS search and generation
                else:
                    result = self._generic_acknowledgment_response(incoming, contact)
                    return record_and_return(
                        result,
                        similarity_score=result.get("similarity_score", 0.0),
                        decision="clarify",
                    )

            # Reactions typically don't need responses
            if msg_classification.message_type == MessageType.REACTION:
                result = self._generic_acknowledgment_response(incoming, contact)
                return record_and_return(
                    result,
                    similarity_score=result.get("similarity_score", 0.0),
                    decision="clarify",
                )

            # Greetings get quick responses
            if msg_classification.message_type == MessageType.GREETING:
                result = self._generic_acknowledgment_response(incoming, contact)
                return record_and_return(
                    result,
                    similarity_score=result.get("similarity_score", 0.0),
                    decision="clarify",
                )

            # Farewells get quick responses
            if msg_classification.message_type == MessageType.FAREWELL:
                result = self._generic_acknowledgment_response(incoming, contact)
                return record_and_return(
                    result,
                    similarity_score=result.get("similarity_score", 0.0),
                    decision="clarify",
                )

            # If context is vague and we need clarification, ask for it
            if msg_classification.context_requirement == ContextRequirement.VAGUE:
                if not thread or len(thread) < 2:
                    result = self._ask_for_clarification(incoming, thread)
                    return record_and_return(
                        result,
                        similarity_score=result.get("similarity_score", 0.0),
                        decision="clarify",
                    )

        # Legacy fallback: Handle simple acknowledgments using old method
        elif self._is_simple_acknowledgment(incoming):
            result = self._generic_acknowledgment_response(incoming, contact)
            return record_and_return(
                result,
                similarity_score=result.get("similarity_score", 0.0),
                decision="clarify",
            )

        # Step 4: For QUICK_REPLY intents with high confidence, handle as acknowledgment
        if (
            intent_result
            and intent_result.intent == IntentType.QUICK_REPLY
            and intent_result.confidence >= 0.8
        ):
            # Even if not in our exact list, high-confidence quick replies
            # should be handled with generic responses
            result = self._generic_acknowledgment_response(incoming, contact)
            return record_and_return(
                result,
                similarity_score=result.get("similarity_score", 0.0),
                decision="clarify",
            )

        thresholds = self._get_thresholds()
        template_threshold = thresholds["template"]
        context_threshold = thresholds["context"]
        generate_threshold = thresholds["generate"]

        # Step 5: Search FAISS index for similar triggers
        try:
            search_start = time.perf_counter()
            search_results = self.index_searcher.search_with_pairs(
                query=incoming,
                k=5,
                threshold=generate_threshold,
                prefer_recent=True,
                embedder=cached_embedder,
            )
            latency_ms["faiss_search"] = (time.perf_counter() - search_start) * 1000
        except FileNotFoundError:
            # Index not built yet - fall back to generation
            logger.warning("FAISS index not found, falling back to generation")
            search_results = []
        except Exception as e:
            logger.exception("Error searching index: %s", e)
            search_results = []

        # Step 6: Check if message is context-dependent (needs current info)
        is_context_dependent = self._is_context_dependent(incoming)
        if is_context_dependent:
            logger.debug("Message is context-dependent, skipping template matching")
            # Context-dependent questions need clarification or current context
            if not thread or len(thread) < 2:
                result = self._ask_for_clarification(incoming, thread)
                return record_and_return(
                    result,
                    similarity_score=0.0,
                    faiss_candidates=len(search_results),
                    decision="clarify",
                )
            # If we have thread context, generate with it
            model_loaded = self.generator.is_loaded()
            generate_start = time.perf_counter()
            result = self._generate_response(
                incoming,
                search_results,
                contact,
                thread,
                chat_id=chat_id,
                reason="context_dependent",
            )
            latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000
            similarity = search_results[0]["similarity"] if search_results else 0.0
            return record_and_return(
                result,
                similarity_score=similarity,
                faiss_candidates=len(search_results),
                model_loaded=model_loaded,
                decision="generate",
            )

        # Step 7: Route based on best similarity score
        if search_results:
            best_result = search_results[0]
            best_score = best_result["similarity"]

            # High confidence -> template response with top-K variety
            if best_score >= template_threshold:
                # Get all high-confidence matches for variety
                high_confidence_matches = [
                    r for r in search_results if r["similarity"] >= template_threshold
                ]
                template_start = time.perf_counter()
                result = self._template_response(high_confidence_matches, contact, incoming)
                latency_ms["template_select"] = (time.perf_counter() - template_start) * 1000

                # Handle fallback if no coherent templates found
                if result.get("type") == "fallback_to_generation":
                    logger.debug("Falling back to generation due to no coherent matches")
                    model_loaded = self.generator.is_loaded()
                    generate_start = time.perf_counter()
                    result = self._generate_response(
                        incoming,
                        search_results,
                        contact,
                        thread,
                        chat_id=chat_id,
                        fallback=True,
                        reason="no_coherent_templates",
                    )
                    latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000
                    return record_and_return(
                        result,
                        similarity_score=best_score,
                        faiss_candidates=len(search_results),
                        model_loaded=model_loaded,
                        decision="generate",
                    )
                return record_and_return(
                    result,
                    similarity_score=best_score,
                    faiss_candidates=len(search_results),
                    decision="template",
                )

            # Medium confidence -> LLM generation with context
            if best_score >= context_threshold:
                model_loaded = self.generator.is_loaded()
                generate_start = time.perf_counter()
                result = self._generate_response(
                    incoming,
                    search_results,
                    contact,
                    thread,
                    chat_id=chat_id,
                )
                latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000
                return record_and_return(
                    result,
                    similarity_score=best_score,
                    faiss_candidates=len(search_results),
                    model_loaded=model_loaded,
                    decision="generate",
                )

            # Low but above minimum -> try generation with caution
            if best_score >= generate_threshold:
                model_loaded = self.generator.is_loaded()
                generate_start = time.perf_counter()
                result = self._generate_response(
                    incoming,
                    search_results,
                    contact,
                    thread,
                    chat_id=chat_id,
                    cautious=True,
                )
                latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000
                return record_and_return(
                    result,
                    similarity_score=best_score,
                    faiss_candidates=len(search_results),
                    model_loaded=model_loaded,
                    decision="generate",
                )

        # Very low confidence or no results -> check for vague references
        if self._needs_clarification(incoming, thread):
            result = self._ask_for_clarification(incoming, thread)
            return record_and_return(
                result,
                similarity_score=0.0,
                faiss_candidates=len(search_results),
                decision="clarify",
            )

        # No similar patterns found - generate with general context
        model_loaded = self.generator.is_loaded()
        generate_start = time.perf_counter()
        result = self._generate_response(
            incoming,
            search_results,
            contact,
            thread,
            chat_id=chat_id,
            fallback=True,
        )
        latency_ms["generate"] = (time.perf_counter() - generate_start) * 1000
        similarity = search_results[0]["similarity"] if search_results else 0.0
        return record_and_return(
            result,
            similarity_score=similarity,
            faiss_candidates=len(search_results),
            model_loaded=model_loaded,
            decision="generate",
        )

    def _template_response(
        self,
        matches: list[dict[str, Any]],
        contact: Contact | None,
        incoming: str = "",
    ) -> dict[str, Any]:
        """Return a template response with variety from top-K matches.

        Uses quality filtering and contact-aware selection:
        - Filter by coherence score (response appropriateness)
        - Filter by contact style (professional for boss, casual for friends)
        - Pick randomly from remaining high-quality responses

        Args:
            matches: List of high-confidence matches from FAISS search.
            contact: Optional contact for style adaptation.
            incoming: The incoming message (for coherence scoring).

        Returns:
            Routing result dict with template response.
        """
        if not matches:
            # Shouldn't happen, but handle gracefully
            return self._clarify_response(
                "I couldn't find a good match for that message.",
                reason="no_matches",
            )

        best_match = matches[0]

        # Step 1: Apply coherence scoring to filter inappropriate responses
        scored_matches = []
        for m in matches:
            coherence = score_response_coherence(
                m.get("trigger_text", incoming),
                m["response_text"],
            )
            if coherence >= COHERENCE_THRESHOLD:
                scored_matches.append((m, coherence))

        # Step 2: Filter by contact style if needed
        if contact and contact.relationship:
            relationship = contact.relationship.lower()

            # For professional relationships, filter out casual responses
            if relationship in ("boss", "manager", "coworker", "colleague", "client"):
                scored_matches = [
                    (m, c)
                    for m, c in scored_matches
                    if self._is_professional_response(m["response_text"])
                ]

        # Step 3: If no coherent/appropriate responses remain, fall back to generation
        if not scored_matches:
            logger.debug(
                "No coherent template matches (original: %d), falling back to generation",
                len(matches),
            )
            # Return None to signal caller to use generation instead
            # But we're already in _template_response, so return a clarify response
            # Actually, we should fall back to generation - let's return a special marker
            return {
                "type": "fallback_to_generation",
                "matches": matches,
                "reason": "no_coherent_matches",
            }

        # Sort by coherence score (highest first)
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        # If only one coherent match, use it directly
        if len(scored_matches) == 1:
            match, coherence = scored_matches[0]
            return {
                "type": "template",
                "response": match["response_text"],
                "confidence": "high",
                "similarity_score": match["similarity"],
                "coherence_score": coherence,
                "cluster_name": match.get("cluster_name"),
                "contact_style": contact.style_notes if contact else None,
                "trigger_matched": match["trigger_text"],
            }

        # Multiple coherent matches: pick randomly from top responses for variety
        top_matches = scored_matches[:MAX_TEMPLATE_RESPONSES]
        candidate_responses = [m["response_text"] for m, _ in top_matches]

        # Deduplicate similar responses (exact matches)
        unique_responses = list(dict.fromkeys(candidate_responses))

        # Pick randomly from unique responses
        selected_response = random.choice(unique_responses)

        # Find which match provided the selected response (for metadata)
        selected_match = scored_matches[0][0]  # Default to best
        selected_coherence = scored_matches[0][1]
        for m, c in scored_matches:
            if m["response_text"] == selected_response:
                selected_match = m
                selected_coherence = c
                break

        return {
            "type": "template",
            "response": selected_response,
            "confidence": "high",
            "similarity_score": best_match["similarity"],  # Report best score
            "coherence_score": selected_coherence,
            "cluster_name": selected_match.get("cluster_name"),
            "contact_style": contact.style_notes if contact else None,
            "trigger_matched": selected_match["trigger_text"],
            "candidates_considered": len(unique_responses),
            "coherent_matches": len(scored_matches),
        }

    def _generate_response(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        chat_id: str | None = None,
        cautious: bool = False,
        fallback: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Generate an LLM response with context from similar patterns.

        Args:
            incoming: The incoming message.
            search_results: Similar patterns from FAISS.
            contact: Contact for personalization.
            thread: Recent conversation context (if already available).
            chat_id: Chat ID for fetching conversation history from iMessage.
            cautious: If True, add hedge language to response.
            fallback: If True, this is a fallback when no patterns matched.
            reason: Optional reason why generation was chosen over template.

        Returns:
            Routing result dict with generated response.
        """
        from contracts.models import GenerationRequest
        from jarvis.prompts import build_rag_reply_prompt

        # Build few-shot examples from similar patterns
        similar_exchanges = []
        similar_triggers = []
        for result in search_results[:3]:
            similar_exchanges.append((result["trigger_text"], result["response_text"]))
            similar_triggers.append(result["trigger_text"])

        # Build conversation context - prefer passed thread, fetch from iMessage if not available
        context_messages = []
        if thread:
            context_messages = thread[-10:]  # Use provided thread context
        elif chat_id:
            # Fetch conversation history from iMessage database
            context_messages = self._fetch_conversation_context(chat_id, limit=10)
            if context_messages:
                logger.debug("Fetched %d messages from iMessage for context", len(context_messages))

        # Format context for prompt
        context = ""
        if context_messages:
            context = "\n".join(context_messages)
        context += f"\n[Incoming]: {incoming}"

        # Get relationship profile for the contact
        relationship_profile = None
        style_guide = None
        if contact:
            # Try to load the full learned relationship profile
            full_profile = load_profile(str(contact.id)) if contact.id else None

            if full_profile and full_profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
                # Use learned patterns from message history
                style_guide = generate_style_guide(full_profile)
                relationship_profile = {
                    "tone": (
                        "professional"
                        if full_profile.tone_profile.formality_score >= 0.7
                        else "casual"
                        if full_profile.tone_profile.formality_score < 0.4
                        else "balanced"
                    ),
                    "avg_message_length": full_profile.tone_profile.avg_message_length,
                    "emoji_frequency": full_profile.tone_profile.emoji_frequency,
                    "relationship": contact.relationship or "friend",
                    "style_guide": style_guide,
                }
                logger.debug(
                    "Using learned relationship profile for %s (formality=%.2f, %d messages)",
                    contact.display_name,
                    full_profile.tone_profile.formality_score,
                    full_profile.message_count,
                )
            else:
                # Fall back to manual style notes
                relationship_profile = {
                    "tone": contact.style_notes or "casual",
                    "relationship": contact.relationship or "friend",
                }

        # Build the prompt
        prompt = build_rag_reply_prompt(
            context=context,
            last_message=incoming,
            contact_name=contact.display_name if contact else "them",
            similar_exchanges=similar_exchanges if similar_exchanges else None,
            relationship_profile=relationship_profile,
        )

        # Generate with the model using LFM-optimal parameters
        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context] if context else [],
                few_shot_examples=similar_exchanges,
                max_tokens=100,  # Text messages should be short
                # Using LFM defaults: temp=0.1, top_p=0.1, top_k=50, repetition_penalty=1.05
            )

            response = self.generator.generate(request)
            generated_text = response.text.strip()

            # Remove common formal greetings that don't match texting style
            formal_greetings = (
                "hey!",
                "hi!",
                "hello!",
                "hey there!",
                "hi there!",
                "hello there!",
            )
            for greeting in formal_greetings:
                if generated_text.lower().startswith(greeting):
                    generated_text = generated_text[len(greeting) :].strip()
                    # Re-capitalize first letter
                    if generated_text:
                        generated_text = generated_text[0].upper() + generated_text[1:]
                    break

            # Trim overly long responses (>2x expected length)
            avg_msg_len = 50
            if relationship_profile:
                avg_msg_len = relationship_profile.get("avg_message_length", 50)
            expected_length = int(avg_msg_len) * 2
            if len(generated_text) > max(80, expected_length) and ". " in generated_text:
                # Keep only first sentence(s) up to limit
                sentences = generated_text.split(". ")
                trimmed = []
                current_len = 0
                for s in sentences:
                    if current_len + len(s) > expected_length:
                        break
                    trimmed.append(s)
                    current_len += len(s) + 2
                if trimmed:
                    generated_text = ". ".join(trimmed)
                    if not generated_text.endswith((".", "!", "?")):
                        generated_text += "."

            confidence = "medium" if not cautious else "low"
            similarity = search_results[0]["similarity"] if search_results else 0.0

            result = {
                "type": "generated",
                "response": generated_text,
                "confidence": confidence,
                "similarity_score": similarity,
                "contact_style": contact.style_notes if contact else None,
                "similar_triggers": similar_triggers if similar_triggers else None,
                "is_fallback": fallback,
            }
            if reason:
                result["generation_reason"] = reason
            return result

        except Exception as e:
            logger.exception("Generation failed: %s", e)
            return self._clarify_response(
                "I'm having trouble generating a response. Could you give me more details?",
                reason="generation_error",
            )

    def _needs_clarification(
        self,
        incoming: str,
        thread: list[str] | None,
    ) -> bool:
        """Check if the message needs clarification due to vague references.

        Args:
            incoming: The incoming message.
            thread: Previous conversation context.

        Returns:
            True if clarification is needed, False otherwise.
        """
        incoming_lower = incoming.lower()

        # Vague reference patterns
        vague_references = [
            "that",
            "it",
            "the thing",
            "this",
            "those",
            "these",
            "what you said",
            "what we discussed",
            "the other",
        ]

        # Check for vague references without clear context
        has_vague_ref = any(ref in incoming_lower for ref in vague_references)

        if has_vague_ref:
            # If we have thread context, we might be able to resolve the reference
            if thread and len(thread) >= 2:
                return False  # Assume context provides enough info
            return True

        # Very short messages with no context
        if len(incoming.split()) <= 2 and not thread:
            return True

        return False

    def _ask_for_clarification(
        self,
        incoming: str,
        thread: list[str] | None,
    ) -> dict[str, Any]:
        """Generate a clarification request.

        Args:
            incoming: The incoming message.
            thread: Previous conversation context.

        Returns:
            Routing result dict with clarification request.
        """
        incoming_lower = incoming.lower()

        # Detect what kind of clarification is needed
        if any(word in incoming_lower for word in ["that", "it", "the thing", "this"]):
            clarification = "What are you referring to? I want to make sure I understand."
        elif any(word in incoming_lower for word in ["when", "what time"]):
            clarification = "Could you be more specific about the timing?"
        elif any(word in incoming_lower for word in ["where", "location"]):
            clarification = "Which location are you asking about?"
        else:
            clarification = (
                "Could you give me a bit more context? "
                "I want to make sure I give you the right response."
            )

        return self._clarify_response(clarification, reason="vague_reference")

    def _clarify_response(
        self,
        message: str,
        reason: str = "unknown",
    ) -> dict[str, Any]:
        """Create a clarification response.

        Args:
            message: The clarification message.
            reason: Why clarification was needed.

        Returns:
            Routing result dict.
        """
        return {
            "type": "clarify",
            "response": message,
            "confidence": "low",
            "similarity_score": 0.0,
            "reason": reason,
        }

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics about the router's index and database.

        Returns:
            Dict with index and database statistics.
        """
        stats = {
            "db_stats": self.db.get_stats(),
            "index_available": False,
        }

        try:
            active_index = self.db.get_active_index()
            if active_index:
                stats["index_available"] = True
                stats["index_version"] = active_index.version_id
                stats["index_vectors"] = active_index.num_vectors
                stats["index_model"] = active_index.model_name
        except Exception as e:
            logger.debug("Failed to get index stats: %s", e)

        return stats


# =============================================================================
# Singleton Access
# =============================================================================

_router: ReplyRouter | None = None
_router_lock = threading.Lock()


def get_reply_router() -> ReplyRouter:
    """Get or create the singleton ReplyRouter instance.

    Returns:
        The shared ReplyRouter instance.
    """
    global _router

    if _router is None:
        with _router_lock:
            if _router is None:
                _router = ReplyRouter()

    return _router


def reset_reply_router() -> None:
    """Reset the singleton ReplyRouter.

    Useful for testing or when the index needs to be reloaded.
    """
    global _router

    with _router_lock:
        _router = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "TEMPLATE_THRESHOLD",
    "CONTEXT_THRESHOLD",
    "GENERATE_THRESHOLD",
    # Exceptions
    "RouterError",
    "IndexNotAvailableError",
    # Classes
    "RouteResult",
    "ReplyRouter",
    # Singleton functions
    "get_reply_router",
    "reset_reply_router",
]
