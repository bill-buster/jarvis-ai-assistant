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

if TYPE_CHECKING:
    from jarvis.index import TriggerIndexSearcher
    from models import MLXGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Similarity thresholds for routing decisions
TEMPLATE_THRESHOLD = 0.85  # Very confident -> use template directly
CONTEXT_THRESHOLD = 0.60  # Below this -> ask for clarification
GENERATE_THRESHOLD = 0.40  # Minimum for attempting generation

# Response variety
MAX_TEMPLATE_RESPONSES = 5  # Max responses to choose from per cluster


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
    ) -> None:
        """Initialize the router.

        Args:
            db: Database instance for contacts and pairs. Uses default if None.
            index_searcher: FAISS index searcher. Created lazily if None.
            generator: MLX generator for LLM responses. Created lazily if None.
        """
        self._db = db
        self._index_searcher = index_searcher
        self._generator = generator
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

    def route(
        self,
        incoming: str,
        contact_id: int | None = None,
        thread: list[str] | None = None,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        """Route an incoming message to the appropriate response strategy.

        Args:
            incoming: The incoming message text to respond to.
            contact_id: Optional contact ID for personalization.
            thread: Optional list of recent messages for context.
            chat_id: Optional chat ID for context lookup.

        Returns:
            Dict with routing result containing:
            - type: 'template', 'generated', or 'clarify'
            - response: The response text
            - confidence: 'high', 'medium', or 'low'
            - Additional metadata (cluster_name, similarity_score, etc.)
        """
        if not incoming or not incoming.strip():
            return self._clarify_response(
                "I received an empty message. Could you tell me what you need?",
                reason="empty_message",
            )

        # Step 1: Search FAISS index for similar triggers
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

        # Step 2: Get contact info for personalization
        contact = None
        if contact_id:
            contact = self.db.get_contact(contact_id)
        elif chat_id:
            contact = self.db.get_contact_by_chat_id(chat_id)

        # Step 3: Route based on best similarity score
        if search_results:
            best_result = search_results[0]
            best_score = best_result["similarity"]

            # High confidence -> template response
            if best_score >= TEMPLATE_THRESHOLD:
                return self._template_response(best_result, contact)

            # Medium confidence -> LLM generation with context
            if best_score >= CONTEXT_THRESHOLD:
                return self._generate_response(incoming, search_results, contact, thread)

            # Low but above minimum -> try generation with caution
            if best_score >= GENERATE_THRESHOLD:
                return self._generate_response(
                    incoming, search_results, contact, thread, cautious=True
                )

        # Very low confidence or no results -> check for vague references
        if self._needs_clarification(incoming, thread):
            return self._ask_for_clarification(incoming, thread)

        # No similar patterns found - generate with general context
        return self._generate_response(incoming, search_results, contact, thread, fallback=True)

    def _template_response(
        self,
        match: dict[str, Any],
        contact: Contact | None,
    ) -> dict[str, Any]:
        """Return a template response for high-confidence matches.

        Args:
            match: Best match from FAISS search.
            contact: Optional contact for style adaptation.

        Returns:
            Routing result dict with template response.
        """
        response = match["response_text"]

        # Get cluster info if available
        cluster_name = match.get("cluster_name")

        # If we have a cluster with multiple example responses, add variety
        if match.get("cluster_id"):
            cluster = self.db.get_cluster(match["cluster_id"])
            if cluster and cluster.example_responses:
                # Pick a random response from the cluster
                responses = cluster.example_responses[:MAX_TEMPLATE_RESPONSES]
                response = random.choice(responses)

        return {
            "type": "template",
            "response": response,
            "confidence": "high",
            "similarity_score": match["similarity"],
            "cluster_name": cluster_name,
            "contact_style": contact.style_notes if contact else None,
            "trigger_matched": match["trigger_text"],
        }

    def _generate_response(
        self,
        incoming: str,
        search_results: list[dict[str, Any]],
        contact: Contact | None,
        thread: list[str] | None,
        cautious: bool = False,
        fallback: bool = False,
    ) -> dict[str, Any]:
        """Generate an LLM response with context from similar patterns.

        Args:
            incoming: The incoming message.
            search_results: Similar patterns from FAISS.
            contact: Contact for personalization.
            thread: Recent conversation context.
            cautious: If True, add hedge language to response.
            fallback: If True, this is a fallback when no patterns matched.

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

        # Build conversation context
        context = ""
        if thread:
            context = "\n".join(f"[Previous]: {msg}" for msg in thread[-5:])
        context += f"\n[Incoming]: {incoming}"

        # Get relationship profile for the contact
        relationship_profile = None
        if contact:
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

        # Generate with the model
        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context] if context else [],
                few_shot_examples=similar_exchanges,
                max_tokens=150,
                temperature=0.7 if not cautious else 0.5,
            )

            response = self.generator.generate(request)
            generated_text = response.text.strip()

            # Add hedge for cautious mode
            if cautious and generated_text:
                # Don't modify the response, just mark it as lower confidence
                pass

            confidence = "medium" if not cautious else "low"
            similarity = search_results[0]["similarity"] if search_results else 0.0

            return {
                "type": "generated",
                "response": generated_text,
                "confidence": confidence,
                "similarity_score": similarity,
                "contact_style": contact.style_notes if contact else None,
                "similar_triggers": similar_triggers if similar_triggers else None,
                "is_fallback": fallback,
            }

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
