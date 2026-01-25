"""Template Matcher for semantic template matching.

Bypasses model generation for common request patterns using
semantic similarity with sentence embeddings.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded sentence transformer
_sentence_model = None


class SentenceModelError(Exception):
    """Raised when sentence transformer model cannot be loaded."""


def _get_sentence_model():
    """Lazy-load the sentence transformer model.

    Returns:
        The loaded SentenceTransformer model

    Raises:
        SentenceModelError: If model cannot be loaded (network issues, etc.)
    """
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence transformer: all-MiniLM-L6-v2")
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.exception("Failed to load sentence transformer")
            msg = f"Failed to load sentence transformer: {e}"
            raise SentenceModelError(msg) from e
    return _sentence_model


@dataclass
class ResponseTemplate:
    """A template for common response patterns."""

    name: str
    patterns: list[str]  # Example prompts that match this template
    response: str  # The response to return


@dataclass
class TemplateMatch:
    """Result of template matching."""

    template: ResponseTemplate
    similarity: float
    matched_pattern: str


def _get_minimal_fallback_templates() -> list[ResponseTemplate]:
    """Minimal templates for development when WS3 not available."""
    return [
        ResponseTemplate(
            name="thank_you_acknowledgment",
            patterns=[
                "Thanks for sending the report",
                "Thank you for the update",
                "Thanks for letting me know",
                "Thank you for your email",
                "Thanks for the information",
            ],
            response="You're welcome! Let me know if you need anything else.",
        ),
        ResponseTemplate(
            name="meeting_confirmation",
            patterns=[
                "Confirming our meeting tomorrow",
                "Just confirming our call",
                "Confirming the meeting time",
                "See you at the meeting",
                "Looking forward to our meeting",
            ],
            response="Confirmed! Looking forward to it.",
        ),
        ResponseTemplate(
            name="schedule_request",
            patterns=[
                "Can we schedule a meeting",
                "When are you free to meet",
                "Let's set up a call",
                "What times work for you",
                "Can we find a time to talk",
            ],
            response="I'd be happy to meet. Could you share a few time options that work for you?",
        ),
        ResponseTemplate(
            name="acknowledgment",
            patterns=[
                "Got it",
                "Understood",
                "Makes sense",
                "Sounds good",
                "Perfect",
            ],
            response="Great, thanks for confirming!",
        ),
        ResponseTemplate(
            name="file_receipt",
            patterns=[
                "I've attached the file",
                "Please find attached",
                "Here's the document",
                "Attached is the file you requested",
                "I'm sending over the file",
            ],
            response="Thanks for sending this over! I'll review it shortly.",
        ),
        ResponseTemplate(
            name="deadline_reminder",
            patterns=[
                "Just a reminder about the deadline",
                "Don't forget the deadline",
                "Reminder: deadline approaching",
                "The deadline is coming up",
                "Final reminder about the due date",
            ],
            response="Thanks for the reminder! I'm on track to complete this by the deadline.",
        ),
        ResponseTemplate(
            name="greeting",
            patterns=[
                "Hi, how are you",
                "Hello, hope you're doing well",
                "Good morning",
                "Hey, hope all is well",
                "Hi there",
            ],
            response="Hi! I'm doing well, thanks for asking. How can I help you today?",
        ),
        ResponseTemplate(
            name="out_of_office",
            patterns=[
                "I'll be out of office",
                "I'm on vacation",
                "I'll be unavailable",
                "Out of the office until",
                "Taking some time off",
            ],
            response="Thanks for letting me know! Enjoy your time off.",
        ),
        ResponseTemplate(
            name="follow_up",
            patterns=[
                "Just following up",
                "Wanted to check in",
                "Any updates on this",
                "Circling back on this",
                "Following up on my previous email",
            ],
            response="Thanks for following up! Let me check on this and get back to you shortly.",
        ),
        ResponseTemplate(
            name="apology",
            patterns=[
                "Sorry for the delay",
                "Apologies for the late response",
                "Sorry I missed your message",
                "My apologies for not responding sooner",
                "Sorry for the wait",
            ],
            response="No worries at all! I appreciate you getting back to me.",
        ),
    ]


def _load_templates() -> list[ResponseTemplate]:
    """Load templates from WS3, fallback to minimal set.

    WS3 templates are organized by category with response strings.
    We convert them to ResponseTemplate objects, using the response
    as both pattern and response for simple matching.
    """
    try:
        from benchmarks.coverage.templates import get_templates_by_category

        category_templates = get_templates_by_category()
        templates = []

        for category, responses in category_templates.items():
            # Convert WS3 format: use responses as both patterns and responses
            templates.append(
                ResponseTemplate(
                    name=category,
                    patterns=responses,  # Use responses as matching patterns
                    response=responses[0],  # Use first response as default
                )
            )

        logger.info("Loaded %d template categories from WS3", len(templates))
        return templates
    except ImportError:
        logger.warning("WS3 templates not available, using minimal fallback set")
        return _get_minimal_fallback_templates()


class TemplateMatcher:
    """Semantic template matcher using sentence embeddings.

    Computes cosine similarity between input prompt and template patterns.
    Returns best matching template if similarity exceeds threshold.
    """

    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, templates: list[ResponseTemplate] | None = None) -> None:
        """Initialize the template matcher.

        Args:
            templates: List of templates to use. Loads defaults if not provided.
        """
        self.templates = templates or _load_templates()
        self._pattern_embeddings: np.ndarray | None = None
        self._pattern_to_template: list[tuple[str, ResponseTemplate]] = []

    def _ensure_embeddings(self) -> None:
        """Compute and cache embeddings for all template patterns."""
        if self._pattern_embeddings is not None:
            return

        model = _get_sentence_model()

        # Collect all patterns with their templates
        all_patterns = []
        for template in self.templates:
            for pattern in template.patterns:
                all_patterns.append(pattern)
                self._pattern_to_template.append((pattern, template))

        # Compute embeddings in batch
        self._pattern_embeddings = model.encode(all_patterns, convert_to_numpy=True)
        logger.info("Computed embeddings for %d patterns", len(all_patterns))

    def match(self, query: str) -> TemplateMatch | None:
        """Find best matching template for a query.

        Args:
            query: Input prompt to match against templates

        Returns:
            TemplateMatch if similarity >= threshold, None otherwise.
            Returns None if sentence model fails to load (falls back to model generation).
        """
        try:
            self._ensure_embeddings()

            model = _get_sentence_model()
            query_embedding = model.encode([query], convert_to_numpy=True)[0]

            # Compute cosine similarities
            similarities = np.dot(self._pattern_embeddings, query_embedding) / (
                np.linalg.norm(self._pattern_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            if best_similarity >= self.SIMILARITY_THRESHOLD:
                matched_pattern, template = self._pattern_to_template[best_idx]
                logger.info(
                    "Template match: '%s' -> %s (similarity: %.3f)",
                    query[:50],
                    template.name,
                    best_similarity,
                )
                return TemplateMatch(
                    template=template,
                    similarity=best_similarity,
                    matched_pattern=matched_pattern,
                )

            logger.debug(
                "No template match for '%s' (best similarity: %.3f)",
                query[:50],
                best_similarity,
            )
            return None

        except SentenceModelError:
            # Fall back to model generation if sentence model unavailable
            logger.warning("Template matching unavailable, falling back to model generation")
            return None
