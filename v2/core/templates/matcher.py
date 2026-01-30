"""Response template matcher for instant replies.

Loads learned response patterns and matches incoming messages
to skip LLM generation for common responses.

Based on QMD research: template matching before model generation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default template file location (relative to project root)
DEFAULT_TEMPLATES_PATH = Path(__file__).parent.parent.parent.parent / "response_templates.json"

# Minimum confidence for template match
MIN_CONFIDENCE = 0.75

# Minimum count to consider a template reliable
MIN_COUNT = 10


@dataclass
class TemplateMatch:
    """Result of template matching."""

    response: str  # The response to use
    actual: str  # The actual text you typed (preserves casing)
    confidence: float  # Similarity score
    trigger: str  # The trigger that matched
    count: int  # How many times you've used this response


@dataclass
class ResponseTemplate:
    """A learned response template."""

    response: str  # Normalized response (lowercase)
    actual: str  # Actual text with original casing
    count: int  # Usage count
    triggers: list[str]  # Sample trigger messages
    trigger_embeddings: np.ndarray | None = None  # Pre-computed embeddings


class TemplateMatcher:
    """Matches incoming messages to learned response templates."""

    def __init__(
        self,
        templates_path: Path | str | None = None,
        min_confidence: float = MIN_CONFIDENCE,
        min_count: int = MIN_COUNT,
    ):
        """Initialize template matcher.

        Args:
            templates_path: Path to response_templates.json
            min_confidence: Minimum similarity for a match (0-1)
            min_count: Minimum usage count for template to be used
        """
        self.templates_path = Path(templates_path) if templates_path else DEFAULT_TEMPLATES_PATH
        self.min_confidence = min_confidence
        self.min_count = min_count

        self._templates: list[ResponseTemplate] = []
        self._all_triggers: list[str] = []
        self._trigger_to_template: dict[int, int] = {}  # trigger_idx -> template_idx
        self._trigger_embeddings: np.ndarray | None = None
        self._embedding_model = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from core.embeddings.model import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def load(self) -> bool:
        """Load templates from file.

        Returns:
            True if loaded successfully
        """
        with self._load_lock:
            if self._loaded:
                return True

            if not self.templates_path.exists():
                logger.warning(f"Templates file not found: {self.templates_path}")
                return False

            try:
                start = time.time()

                with open(self.templates_path) as f:
                    data = json.load(f)

                # Filter to reliable templates (enough usage count)
                self._templates = []
                self._all_triggers = []
                self._trigger_to_template = {}

                for item in data:
                    if item.get("count", 0) < self.min_count:
                        continue

                    template = ResponseTemplate(
                        response=item["response"],
                        actual=item["actual"],
                        count=item["count"],
                        triggers=item.get("sample_triggers", [])[:20],  # Limit triggers
                    )
                    template_idx = len(self._templates)
                    self._templates.append(template)

                    # Index triggers
                    for trigger in template.triggers:
                        trigger_idx = len(self._all_triggers)
                        self._all_triggers.append(trigger)
                        self._trigger_to_template[trigger_idx] = template_idx

                # Pre-compute embeddings for all triggers
                if self._all_triggers:
                    model = self._get_embedding_model()
                    self._trigger_embeddings = model.embed_batch(self._all_triggers)
                    # Normalize for cosine similarity
                    norms = np.linalg.norm(self._trigger_embeddings, axis=1, keepdims=True)
                    self._trigger_embeddings = self._trigger_embeddings / (norms + 1e-8)

                elapsed = (time.time() - start) * 1000
                logger.info(
                    f"Loaded {len(self._templates)} templates with "
                    f"{len(self._all_triggers)} triggers in {elapsed:.0f}ms"
                )

                self._loaded = True
                return True

            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                return False

    def match(self, message: str) -> TemplateMatch | None:
        """Match a message to a response template.

        Args:
            message: Incoming message to match

        Returns:
            TemplateMatch if confident match found, None otherwise
        """
        if not self._loaded:
            self.load()

        if not self._templates or self._trigger_embeddings is None:
            return None

        try:
            # Embed the incoming message
            model = self._get_embedding_model()
            message_emb = model.embed(message)

            # Normalize
            message_emb = message_emb / (np.linalg.norm(message_emb) + 1e-8)

            # Compute similarities with all triggers
            similarities = self._trigger_embeddings @ message_emb

            # Find best match
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim < self.min_confidence:
                return None

            # Get the template
            template_idx = self._trigger_to_template[best_idx]
            template = self._templates[template_idx]
            trigger = self._all_triggers[best_idx]

            logger.info(
                f"Template match: '{message[:50]}...' -> '{template.actual}' "
                f"(confidence={best_sim:.2f}, count={template.count})"
            )

            return TemplateMatch(
                response=template.response,
                actual=template.actual,
                confidence=best_sim,
                trigger=trigger,
                count=template.count,
            )

        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Get template statistics."""
        return {
            "loaded": self._loaded,
            "template_count": len(self._templates),
            "trigger_count": len(self._all_triggers),
            "min_confidence": self.min_confidence,
            "min_count": self.min_count,
        }


# Singleton
_matcher: TemplateMatcher | None = None
_matcher_lock = threading.Lock()


def get_template_matcher(
    templates_path: Path | str | None = None,
    preload: bool = True,
) -> TemplateMatcher:
    """Get singleton template matcher.

    Args:
        templates_path: Path to templates file (uses default if None)
        preload: If True, load templates immediately

    Returns:
        TemplateMatcher instance
    """
    global _matcher

    if _matcher is None:
        with _matcher_lock:
            if _matcher is None:
                _matcher = TemplateMatcher(templates_path)
                if preload:
                    _matcher.load()

    return _matcher


def reset_template_matcher() -> None:
    """Reset the template matcher singleton."""
    global _matcher
    with _matcher_lock:
        _matcher = None
