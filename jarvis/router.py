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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.db import Contact, JarvisDB, get_db
from jarvis.errors import ErrorCode, JarvisError
from jarvis.intent import IntentClassifier, IntentType, get_intent_classifier
from jarvis.quality_metrics import score_response_coherence
from jarvis.relationships import (
    MIN_MESSAGES_FOR_PROFILE,
    RelationshipProfile,
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
SIMPLE_ACKNOWLEDGMENTS = frozenset({
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "no", "nope", "nah",
    "sure", "thanks", "thank you", "thx", "ty", "np", "cool", "nice", "good",
    "great", "awesome", "alright", "got it", "lol", "haha", "bye", "later",
})


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
    1. Template (similarity >= 0.85): Return cached response instantly
    2. Generate (0.40 <= similarity < 0.85): Use LLM with similar examples
    3. Clarify (similarity < 0.40): Ask user for more context

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
        imessage_reader: ChatDBReader | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            db: Database instance for contacts and pairs. Uses default if None.
            index_searcher: FAISS index searcher. Created lazily if None.
            generator: MLX generator for LLM responses. Created lazily if None.
            intent_classifier: Intent classifier for routing decisions. Created lazily if None.
            imessage_reader: iMessage reader for fetching conversation history. Created lazily if None.
        """
        self._db = db
        self._index_searcher = index_searcher
        self._generator = generator
        self._intent_classifier = intent_classifier
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

    def _fetch_conversation_context(
        self, chat_id: str, limit: int = 10
    ) -> list[str]:
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
            "lol", "haha", "lmao", "omg", "wtf", "bruh", "dude", "bro",
            "🤣", "😂", "😝", "🙄", "💀",
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

        Uses intent classification to handle different message types appropriately:
        - Simple acknowledgments are handled directly without FAISS search
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
            - Additional metadata (cluster_name, similarity_score, etc.)
        """
        if not incoming or not incoming.strip():
            return self._clarify_response(
                "I received an empty message. Could you tell me what you need?",
                reason="empty_message",
            )

        # Step 0: Get contact info for personalization (needed early for acknowledgments)
        contact = None
        if contact_id:
            contact = self.db.get_contact(contact_id)
        elif chat_id:
            contact = self.db.get_contact_by_chat_id(chat_id)

        # Step 1: Classify intent
        try:
            intent_result = self.intent_classifier.classify(incoming)
            logger.debug(
                "Intent classified as %s (confidence: %.3f)",
                intent_result.intent.value,
                intent_result.confidence,
            )
        except Exception as e:
            logger.warning("Intent classification failed: %s", e)
            intent_result = None

        # Step 2: Handle simple acknowledgments directly (no FAISS search needed)
        if self._is_simple_acknowledgment(incoming):
            return self._generic_acknowledgment_response(incoming, contact)

        # Step 3: For QUICK_REPLY intents with high confidence, check if it's a simple ack
        if (
            intent_result
            and intent_result.intent == IntentType.QUICK_REPLY
            and intent_result.confidence >= 0.8
        ):
            # Even if not in our exact list, high-confidence quick replies
            # should be handled with generic responses
            return self._generic_acknowledgment_response(incoming, contact)

        # Step 4: Search FAISS index for similar triggers
        try:
            search_results = self.index_searcher.search_with_pairs(
                query=incoming,
                k=5,
                threshold=GENERATE_THRESHOLD,
                prefer_recent=True,
            )
        except FileNotFoundError:
            # Index not built yet - fall back to generation
            logger.warning("FAISS index not found, falling back to generation")
            search_results = []
        except Exception as e:
            logger.exception("Error searching index: %s", e)
            search_results = []

        # Step 5: Route based on best similarity score
        if search_results:
            best_result = search_results[0]
            best_score = best_result["similarity"]

            # High confidence -> template response with top-K variety
            if best_score >= TEMPLATE_THRESHOLD:
                # Get all high-confidence matches for variety
                high_confidence_matches = [
                    r for r in search_results if r["similarity"] >= TEMPLATE_THRESHOLD
                ]
                result = self._template_response(high_confidence_matches, contact, incoming)

                # Handle fallback if no coherent templates found
                if result.get("type") == "fallback_to_generation":
                    logger.debug("Falling back to generation due to no coherent matches")
                    return self._generate_response(
                        incoming, search_results, contact, thread,
                        chat_id=chat_id, fallback=True, reason="no_coherent_templates"
                    )
                return result

            # Medium confidence -> LLM generation with context
            if best_score >= CONTEXT_THRESHOLD:
                return self._generate_response(
                    incoming, search_results, contact, thread, chat_id=chat_id
                )

            # Low but above minimum -> try generation with caution
            if best_score >= GENERATE_THRESHOLD:
                return self._generate_response(
                    incoming, search_results, contact, thread, chat_id=chat_id, cautious=True
                )

        # Very low confidence or no results -> check for vague references
        if self._needs_clarification(incoming, thread):
            return self._ask_for_clarification(incoming, thread)

        # No similar patterns found - generate with general context
        return self._generate_response(
            incoming, search_results, contact, thread, chat_id=chat_id, fallback=True
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
                    (m, c) for m, c in scored_matches
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
                logger.debug(
                    "Fetched %d messages from iMessage for context", len(context_messages)
                )

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

            # Add hedge for cautious mode
            if cautious and generated_text:
                # Don't modify the response, just mark it as lower confidence
                pass

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
        except Exception:
            pass

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
