"""Template Matcher for semantic template matching.

Bypasses model generation for common request patterns using
semantic similarity with sentence embeddings.

Performance optimizations:
- Pre-normalized pattern embeddings for O(1) cosine similarity
- LRU cache for query embeddings (repeated queries)
- Batch encoding for initial setup

Analytics:
- Tracks template hit/miss rates
- Records queries that miss templates for optimization
- Monitors cache efficiency

Custom Templates:
- User-defined templates stored in ~/.jarvis/custom_templates.json
- Support for trigger phrases, category tags, and group size constraints
- Import/export functionality for sharing template packs
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from jarvis.embedding_adapter import get_embedder, reset_embedder
from jarvis.metrics import get_template_analytics

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class EmbeddingCache(Generic[K, V]):
    """LRU cache for query embeddings.

    Thread-safe implementation with bounded size.
    """

    def __init__(self, maxsize: int = 500) -> None:
        """Initialize the cache.

        Args:
            maxsize: Maximum number of embeddings to cache
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K, track_analytics: bool = True) -> V | None:
        """Get embedding from cache.

        Args:
            key: Cache key to look up
            track_analytics: Whether to record this access in template analytics
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                if track_analytics:
                    get_template_analytics().record_cache_access(hit=True)
                return self._cache[key]
            self._misses += 1
            if track_analytics:
                get_template_analytics().record_cache_access(hit=False)
            return None

    def set(self, key: K, value: V) -> None:
        """Store embedding in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class SentenceModelError(Exception):
    """Raised when sentence transformer model cannot be loaded."""


def _get_sentence_model() -> Any:
    """Get the embedder for template matching.

    Returns the unified embedder instance. For backward compatibility,
    this function name is preserved but now delegates to the unified adapter.

    Returns:
        The UnifiedEmbedder instance (not SentenceTransformer directly)

    Raises:
        SentenceModelError: If no embedding backend is available
    """
    try:
        embedder = get_embedder()
        if not embedder.is_available():
            raise SentenceModelError("No embedding backend available")
        return embedder
    except Exception as e:
        logger.exception("Failed to initialize embedding backend")
        msg = f"Failed to initialize embedding backend: {e}"
        raise SentenceModelError(msg) from e


def unload_sentence_model() -> None:
    """Unload the embedding model to free memory.

    Call this when template matching is no longer needed and you want
    to reclaim memory for other operations (e.g., loading the MLX model).
    """
    logger.info("Unloading embedding model")
    reset_embedder()
    gc.collect()


def is_sentence_model_loaded() -> bool:
    """Check if an embedding model is currently available.

    Returns:
        True if an embedding backend is available, False otherwise
    """
    try:
        embedder = get_embedder()
        return embedder.backend != "none"
    except Exception:
        return False


@dataclass
class ResponseTemplate:
    """A template for common response patterns.

    Attributes:
        name: Unique identifier for the template
        patterns: Example prompts that match this template
        response: The response to return
        is_group_template: Whether this is a group chat specific template
        min_group_size: Minimum group size for this template (None = any size)
        max_group_size: Maximum group size for this template (None = any size)
    """

    name: str
    patterns: list[str]  # Example prompts that match this template
    response: str  # The response to return
    is_group_template: bool = False
    min_group_size: int | None = None  # None means no minimum
    max_group_size: int | None = None  # None means no maximum


@dataclass
class TemplateMatch:
    """Result of template matching."""

    template: ResponseTemplate
    similarity: float
    matched_pattern: str


# =============================================================================
# Custom Template Support
# =============================================================================


@dataclass
class CustomTemplate:
    """User-defined template for custom response patterns.

    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable template name
        template_text: The response text to return when matched
        trigger_phrases: List of phrases that should trigger this template
        category: Category for organization (e.g., "work", "personal", "casual")
        tags: Additional tags for filtering and organization
        min_group_size: Minimum group size to apply this template (None = any)
        max_group_size: Maximum group size to apply this template (None = any)
        enabled: Whether this template is active
        created_at: When the template was created
        updated_at: When the template was last modified
        usage_count: Number of times this template has been matched
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    template_text: str = ""
    trigger_phrases: list[str] = field(default_factory=list)
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    min_group_size: int | None = None
    max_group_size: int | None = None
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "template_text": self.template_text,
            "trigger_phrases": self.trigger_phrases,
            "category": self.category,
            "tags": self.tags,
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomTemplate:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            template_text=data.get("template_text", ""),
            trigger_phrases=data.get("trigger_phrases", []),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            min_group_size=data.get("min_group_size"),
            max_group_size=data.get("max_group_size"),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            usage_count=data.get("usage_count", 0),
        )

    def to_response_template(self) -> ResponseTemplate:
        """Convert to ResponseTemplate for matching."""
        return ResponseTemplate(
            name=f"custom_{self.id}",
            patterns=self.trigger_phrases,
            response=self.template_text,
        )


# Custom templates storage path
CUSTOM_TEMPLATES_PATH = Path.home() / ".jarvis" / "custom_templates.json"

# Thread-safe singleton for custom template store
_custom_template_store: CustomTemplateStore | None = None
_custom_template_store_lock = threading.Lock()


class CustomTemplateStore:
    """Thread-safe storage for custom templates.

    Manages loading, saving, and querying custom templates from
    ~/.jarvis/custom_templates.json
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize the store.

        Args:
            storage_path: Override the default storage path (for testing)
        """
        self._storage_path = storage_path or CUSTOM_TEMPLATES_PATH
        self._templates: dict[str, CustomTemplate] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load templates from storage file."""
        with self._lock:
            if self._storage_path.exists():
                try:
                    with self._storage_path.open() as f:
                        data = json.load(f)
                        templates = data.get("templates", [])
                        self._templates = {t["id"]: CustomTemplate.from_dict(t) for t in templates}
                    logger.info("Loaded %d custom templates", len(self._templates))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load custom templates: %s", e)
                    self._templates = {}
            else:
                self._templates = {}

    def _save(self) -> bool:
        """Save templates to storage file.

        MUST be called while holding self._lock.

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("w") as f:
                data = {
                    "version": 1,
                    "templates": [t.to_dict() for t in self._templates.values()],
                }
                json.dump(data, f, indent=2)
            return True
        except OSError as e:
            logger.error("Failed to save custom templates: %s", e)
            return False

    def get(self, template_id: str) -> CustomTemplate | None:
        """Get a template by ID."""
        with self._lock:
            return self._templates.get(template_id)

    def list_all(self) -> list[CustomTemplate]:
        """List all templates."""
        with self._lock:
            return list(self._templates.values())

    def list_enabled(self) -> list[CustomTemplate]:
        """List only enabled templates."""
        with self._lock:
            return [t for t in self._templates.values() if t.enabled]

    def list_by_category(self, category: str) -> list[CustomTemplate]:
        """List templates by category."""
        with self._lock:
            return [t for t in self._templates.values() if t.category == category]

    def list_by_tag(self, tag: str) -> list[CustomTemplate]:
        """List templates by tag."""
        with self._lock:
            return [t for t in self._templates.values() if tag in t.tags]

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        with self._lock:
            return sorted(set(t.category for t in self._templates.values()))

    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        with self._lock:
            all_tags: set[str] = set()
            for t in self._templates.values():
                all_tags.update(t.tags)
            return sorted(all_tags)

    def create(self, template: CustomTemplate) -> CustomTemplate:
        """Create a new template.

        Args:
            template: The template to create

        Returns:
            The created template with ID assigned
        """
        with self._lock:
            if not template.id:
                template.id = str(uuid.uuid4())
            template.created_at = datetime.now().isoformat()
            template.updated_at = template.created_at
            self._templates[template.id] = template
            self._save()
            logger.info("Created custom template: %s", template.name)
            return template

    def update(self, template_id: str, updates: dict[str, Any]) -> CustomTemplate | None:
        """Update an existing template.

        Args:
            template_id: The template ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated template or None if not found
        """
        with self._lock:
            if template_id not in self._templates:
                return None

            template = self._templates[template_id]

            # Apply updates (excluding id and timestamps)
            for key, value in updates.items():
                if key not in ("id", "created_at") and hasattr(template, key):
                    setattr(template, key, value)

            template.updated_at = datetime.now().isoformat()
            self._save()
            logger.info("Updated custom template: %s", template.name)
            return template

    def delete(self, template_id: str) -> bool:
        """Delete a template.

        Args:
            template_id: The template ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if template_id not in self._templates:
                return False

            template = self._templates.pop(template_id)
            self._save()
            logger.info("Deleted custom template: %s", template.name)
            return True

    def increment_usage(self, template_id: str) -> None:
        """Increment usage count for a template."""
        with self._lock:
            if template_id in self._templates:
                self._templates[template_id].usage_count += 1
                self._save()

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for all templates.

        Returns:
            Dictionary with usage stats
        """
        with self._lock:
            total_usage = sum(t.usage_count for t in self._templates.values())
            by_category: dict[str, int] = {}
            for t in self._templates.values():
                by_category[t.category] = by_category.get(t.category, 0) + t.usage_count

            top_templates = sorted(
                self._templates.values(),
                key=lambda x: x.usage_count,
                reverse=True,
            )[:10]

            return {
                "total_templates": len(self._templates),
                "enabled_templates": len([t for t in self._templates.values() if t.enabled]),
                "total_usage": total_usage,
                "usage_by_category": by_category,
                "top_templates": [
                    {"id": t.id, "name": t.name, "usage_count": t.usage_count}
                    for t in top_templates
                ],
            }

    def export_templates(self, template_ids: list[str] | None = None) -> dict[str, Any]:
        """Export templates for sharing.

        Args:
            template_ids: Specific templates to export, or None for all

        Returns:
            Export data dictionary
        """
        with self._lock:
            if template_ids:
                templates = [
                    self._templates[tid].to_dict() for tid in template_ids if tid in self._templates
                ]
            else:
                templates = [t.to_dict() for t in self._templates.values()]

            return {
                "version": 1,
                "export_date": datetime.now().isoformat(),
                "template_count": len(templates),
                "templates": templates,
            }

    def import_templates(self, data: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
        """Import templates from exported data.

        Args:
            data: Export data dictionary
            overwrite: If True, overwrite existing templates with same ID

        Returns:
            Import results with counts
        """
        imported = 0
        skipped = 0
        errors = 0

        templates = data.get("templates", [])

        with self._lock:
            for template_data in templates:
                try:
                    template = CustomTemplate.from_dict(template_data)

                    if template.id in self._templates and not overwrite:
                        # Generate new ID to avoid conflict
                        template.id = str(uuid.uuid4())

                    template.created_at = datetime.now().isoformat()
                    template.updated_at = template.created_at
                    template.usage_count = 0  # Reset usage on import

                    self._templates[template.id] = template
                    imported += 1
                except Exception as e:
                    logger.warning("Failed to import template: %s", e)
                    errors += 1

            self._save()

        logger.info("Imported %d templates (%d skipped, %d errors)", imported, skipped, errors)
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "total_templates": len(self._templates),
        }

    def reload(self) -> None:
        """Reload templates from storage file."""
        self._load()


def get_custom_template_store() -> CustomTemplateStore:
    """Get the singleton custom template store instance.

    Returns:
        The CustomTemplateStore singleton
    """
    global _custom_template_store
    if _custom_template_store is None:
        with _custom_template_store_lock:
            if _custom_template_store is None:
                _custom_template_store = CustomTemplateStore()
    return _custom_template_store


def reset_custom_template_store() -> None:
    """Reset the custom template store singleton (for testing)."""
    global _custom_template_store
    with _custom_template_store_lock:
        _custom_template_store = None


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
                "thanks for the help",  # For test compatibility
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
        # iMessage-specific templates for quick text patterns
        ResponseTemplate(
            name="quick_ok",
            patterns=[
                "ok",
                "k",
                "kk",
                "okay",
                "okie",
                "okey",
                "alright",
            ],
            response="Got it!",
        ),
        ResponseTemplate(
            name="quick_affirmative",
            patterns=[
                "sure",
                "yep",
                "yup",
                "yeah",
                "ya",
                "yes",
                "definitely",
                "for sure",
            ],
            response="Sounds good!",
        ),
        ResponseTemplate(
            name="quick_thanks",
            patterns=[
                "thx",
                "ty",
                "tysm",
                "thanks",
                "thank u",
                "thank you",
                "thanks!",
            ],
            response="You're welcome!",
        ),
        ResponseTemplate(
            name="quick_no_problem",
            patterns=[
                "np",
                "no prob",
                "no problem",
                "no worries",
                "nw",
                "all good",
            ],
            response="Glad I could help!",
        ),
        ResponseTemplate(
            name="on_my_way",
            patterns=[
                "omw",
                "on my way",
                "leaving now",
                "heading out",
                "be there in 10",
                "coming now",
                "just left",
            ],
            response="See you soon!",
        ),
        ResponseTemplate(
            name="be_there_soon",
            patterns=[
                "be there soon",
                "almost there",
                "5 mins away",
                "pulling up",
                "around the corner",
                "just about there",
            ],
            response="Great, see you in a bit!",
        ),
        ResponseTemplate(
            name="running_late",
            patterns=[
                "running late",
                "gonna be late",
                "running behind",
                "stuck in traffic",
                "be there in a bit",
                "sorry running late",
            ],
            response="No worries, take your time!",
        ),
        ResponseTemplate(
            name="where_are_you",
            patterns=[
                "where are you",
                "where r u",
                "where you at",
                "wya",
                "where u at",
                "you here yet",
            ],
            response="On my way! Be there soon.",
        ),
        ResponseTemplate(
            name="what_time",
            patterns=[
                "what time",
                "when",
                "what time works",
                "when should we meet",
                "what time is good",
                "when are you free",
            ],
            response="Let me check my schedule and get back to you!",
        ),
        ResponseTemplate(
            name="time_proposal",
            patterns=[
                "how about 3",
                "does 5 work",
                "is 7 ok",
                "maybe around 2",
                "let's say noon",
                "how's 6pm",
                "works for me at 4",
            ],
            response="That time works for me!",
        ),
        ResponseTemplate(
            name="hang_out_invite",
            patterns=[
                "wanna hang",
                "want to hang out",
                "down to hang",
                "u free",
                "you free",
                "wanna chill",
                "want to chill",
            ],
            response="Yeah, I'm down! What did you have in mind?",
        ),
        ResponseTemplate(
            name="dinner_plans",
            patterns=[
                "down for dinner",
                "wanna grab dinner",
                "want to get food",
                "hungry?",
                "wanna eat",
                "let's get food",
                "dinner tonight?",
            ],
            response="I'm in! Where were you thinking?",
        ),
        ResponseTemplate(
            name="free_tonight",
            patterns=[
                "free tonight",
                "doing anything tonight",
                "busy tonight",
                "plans tonight",
                "what are you doing tonight",
                "got plans",
            ],
            response="Let me check! What's up?",
        ),
        ResponseTemplate(
            name="coffee_drinks",
            patterns=[
                "let's grab coffee",
                "wanna get coffee",
                "coffee sometime",
                "grab a drink",
                "wanna get drinks",
                "drinks later",
            ],
            response="Sounds great! When works for you?",
        ),
        ResponseTemplate(
            name="laughter",
            patterns=[
                "lol",
                "lmao",
                "haha",
                "hahaha",
                "lolol",
                "rofl",
                "dying",
            ],
            response="Haha right?!",
        ),
        ResponseTemplate(
            name="emoji_reaction",
            patterns=[
                "😂",
                "🤣",
                "😭",
                "💀",
                "😆",
                "🙃",
            ],
            response="😊",
        ),
        ResponseTemplate(
            name="positive_reaction",
            patterns=[
                "nice",
                "nice!",
                "awesome",
                "amazing",
                "love it",
                "so cool",
                "that's great",
                "dope",
                "sick",
            ],
            response="Thanks! 😊",
        ),
        ResponseTemplate(
            name="check_in",
            patterns=[
                "you there",
                "you there?",
                "u there",
                "hello?",
                "hey?",
                "you alive",
                "earth to you",
            ],
            response="I'm here! What's up?",
        ),
        ResponseTemplate(
            name="did_you_see",
            patterns=[
                "did you see my text",
                "did you get my message",
                "see my last message",
                "did you read that",
                "u see that",
                "did u see",
            ],
            response="Just saw it! Let me respond.",
        ),
        ResponseTemplate(
            name="talk_later",
            patterns=[
                "ttyl",
                "talk later",
                "talk to you later",
                "catch you later",
                "later",
                "laters",
                "chat soon",
            ],
            response="Talk soon!",
        ),
        ResponseTemplate(
            name="goodnight",
            patterns=[
                "gn",
                "goodnight",
                "good night",
                "night",
                "nite",
                "sleep well",
                "sweet dreams",
            ],
            response="Goodnight! Sleep well!",
        ),
        ResponseTemplate(
            name="goodbye",
            patterns=[
                "bye",
                "bye!",
                "cya",
                "see ya",
                "see you",
                "peace",
                "take care",
            ],
            response="Bye! Take care!",
        ),
        ResponseTemplate(
            name="appreciation",
            patterns=[
                "you're the best",
                "ur the best",
                "appreciate it",
                "thanks so much",
                "really appreciate it",
                "you rock",
                "ily",
                "love you",
            ],
            response="Aw, thanks! You're the best too!",
        ),
        ResponseTemplate(
            name="question_response",
            patterns=[
                "idk",
                "i don't know",
                "dunno",
                "not sure",
                "no idea",
                "beats me",
            ],
            response="No worries, we can figure it out!",
        ),
        ResponseTemplate(
            name="agreement",
            patterns=[
                "same",
                "same here",
                "me too",
                "i agree",
                "totally",
                "exactly",
                "fr",
                "for real",
            ],
            response="Right?!",
        ),
        ResponseTemplate(
            name="brb",
            patterns=[
                "brb",
                "be right back",
                "one sec",
                "gimme a sec",
                "hold on",
                "one minute",
                "give me a minute",
            ],
            response="No rush, take your time!",
        ),
        # iMessage Assistant Scenarios - queries to the AI assistant about messages
        ResponseTemplate(
            name="summarize_conversation",
            patterns=[
                "summarize my conversation with",
                "give me a summary of my chat with",
                "what did I talk about with",
                "summarize the messages from",
                "recap my conversation with",
                "what's the summary of my texts with",
                "sum up my chat with",
            ],
            response=(
                "I'll analyze your conversation and provide a summary of the key points, "
                "topics discussed, and any action items mentioned."
            ),
        ),
        ResponseTemplate(
            name="summarize_recent_messages",
            patterns=[
                "summarize my recent messages",
                "what have I been texting about",
                "recap my recent conversations",
                "summarize my texts from today",
                "what's been happening in my messages",
                "give me a summary of recent chats",
                "summarize today's messages",
            ],
            response=(
                "I'll review your recent messages and provide a summary of conversations, "
                "key topics, and any items that may need your attention."
            ),
        ),
        ResponseTemplate(
            name="find_messages_from_person",
            patterns=[
                "find messages from",
                "show me texts from",
                "what did say",
                "messages from",
                "show messages from",
                "get messages from",
                "search messages from",
            ],
            response=(
                "I'll search your messages and show you the conversations "
                "from that person. You can specify a time range if needed."
            ),
        ),
        ResponseTemplate(
            name="find_unread_messages",
            patterns=[
                "show me unread messages",
                "what messages haven't I read",
                "do I have unread texts",
                "any unread messages",
                "show unread",
                "unread messages",
                "messages I haven't seen",
            ],
            response=(
                "I'll check for any messages you haven't read yet "
                "and show you a summary of who they're from."
            ),
        ),
        ResponseTemplate(
            name="unread_message_recap",
            patterns=[
                "recap my unread messages",
                "summarize unread texts",
                "what did I miss",
                "catch me up on messages",
                "what messages did I miss",
                "summarize what I haven't read",
                "what's new in my messages",
            ],
            response=(
                "I'll provide a recap of your unread messages, "
                "highlighting important ones and summarizing the rest."
            ),
        ),
        ResponseTemplate(
            name="find_dates_times",
            patterns=[
                "find messages about dates",
                "when did we plan to meet",
                "search for times mentioned",
                "find messages with dates",
                "what dates were mentioned",
                "find scheduled times",
                "search for meeting times",
            ],
            response=(
                "I'll search your messages for mentions of dates, times, "
                "and scheduled events to help you find what you're looking for."
            ),
        ),
        ResponseTemplate(
            name="find_shared_links",
            patterns=[
                "find links in messages",
                "show shared links",
                "what links did they send",
                "find urls in my texts",
                "search for shared links",
                "show me links from",
                "find websites shared",
            ],
            response=(
                "I'll search your messages for shared links and URLs, "
                "and show you when and who shared them."
            ),
        ),
        ResponseTemplate(
            name="find_shared_photos",
            patterns=[
                "find photos in messages",
                "show shared photos",
                "what pictures did they send",
                "find images in my texts",
                "search for photos from",
                "show me pictures from",
                "find shared images",
            ],
            response=(
                "I'll search your messages for shared photos and images, "
                "showing you who sent them and when."
            ),
        ),
        ResponseTemplate(
            name="find_attachments",
            patterns=[
                "find attachments in messages",
                "show shared files",
                "what files did they send",
                "find documents in my texts",
                "search for attachments from",
                "show me files from",
                "find shared documents",
            ],
            response=(
                "I'll search your messages for attachments and files, "
                "showing you the type, sender, and date for each."
            ),
        ),
        ResponseTemplate(
            name="search_topic",
            patterns=[
                "find messages about",
                "search for texts about",
                "show messages mentioning",
                "find conversations about",
                "search my messages for",
                "find texts mentioning",
                "look for messages about",
            ],
            response=(
                "I'll search your messages for that topic and show you relevant conversations."
            ),
        ),
        ResponseTemplate(
            name="search_keyword",
            patterns=[
                "search for keyword",
                "find texts containing",
                "search messages for word",
                "find messages with word",
                "look for keyword in messages",
                "search for specific word",
                "find word in my texts",
            ],
            response=(
                "I'll search your messages for that keyword and show you all matches with context."
            ),
        ),
        ResponseTemplate(
            name="recent_conversations",
            patterns=[
                "who have I texted recently",
                "show recent conversations",
                "who messaged me lately",
                "my recent chats",
                "show my latest conversations",
                "who have I been talking to",
                "list recent contacts",
            ],
            response=(
                "I'll show you a list of your most recent conversations, "
                "sorted by the last message time."
            ),
        ),
        ResponseTemplate(
            name="messages_from_today",
            patterns=[
                "show today's messages",
                "what messages did I get today",
                "today's texts",
                "messages from today",
                "show me today's conversations",
                "who texted me today",
                "today's chats",
            ],
            response=(
                "I'll show you all the messages you've received today, organized by conversation."
            ),
        ),
        ResponseTemplate(
            name="messages_from_yesterday",
            patterns=[
                "show yesterday's messages",
                "what messages did I get yesterday",
                "yesterday's texts",
                "messages from yesterday",
                "show me yesterday's conversations",
                "who texted me yesterday",
                "yesterday's chats",
            ],
            response=("I'll show you all the messages from yesterday, organized by conversation."),
        ),
        ResponseTemplate(
            name="messages_this_week",
            patterns=[
                "show this week's messages",
                "messages from this week",
                "what texts did I get this week",
                "this week's conversations",
                "show me messages since monday",
                "weekly message summary",
                "recap of this week's texts",
            ],
            response=(
                "I'll provide a summary of your messages from this week, "
                "highlighting key conversations and topics."
            ),
        ),
        ResponseTemplate(
            name="find_address_location",
            patterns=[
                "find addresses in messages",
                "search for locations shared",
                "what addresses were sent",
                "find location in texts",
                "search for places mentioned",
                "find shared locations",
                "where did they say to meet",
            ],
            response=(
                "I'll search your messages for addresses and locations "
                "that were shared or mentioned."
            ),
        ),
        ResponseTemplate(
            name="find_phone_numbers",
            patterns=[
                "find phone numbers in messages",
                "search for numbers shared",
                "what phone numbers were sent",
                "find contact numbers in texts",
                "search for phone numbers",
                "find shared phone numbers",
                "numbers mentioned in messages",
            ],
            response=(
                "I'll search your messages for phone numbers and show you who shared them and when."
            ),
        ),
        ResponseTemplate(
            name="message_count",
            patterns=[
                "how many messages from",
                "count messages from",
                "how many texts did I get",
                "message count with",
                "how many times did they text",
                "count my messages",
                "how many texts today",
            ],
            response=(
                "I'll count the messages matching your criteria "
                "and provide you with the statistics."
            ),
        ),
        ResponseTemplate(
            name="last_message_from",
            patterns=[
                "when did I last hear from",
                "last message from",
                "when did they last text",
                "last time I heard from",
                "most recent message from",
                "when was the last text from",
                "how long since I heard from",
            ],
            response=(
                "I'll find the most recent message from that person and tell you when it was sent."
            ),
        ),
        ResponseTemplate(
            name="find_plans_events",
            patterns=[
                "find plans in messages",
                "what events are mentioned",
                "search for plans we made",
                "find scheduled events",
                "what did we plan",
                "search for upcoming plans",
                "find events in my texts",
            ],
            response=(
                "I'll search your messages for mentions of plans, events, and scheduled activities."
            ),
        ),
        ResponseTemplate(
            name="find_recommendations",
            patterns=[
                "find recommendations in messages",
                "what did they recommend",
                "search for suggestions",
                "find recommended places",
                "what restaurants were suggested",
                "find movie recommendations",
                "search for recommendations",
            ],
            response=(
                "I'll search your messages for recommendations and suggestions "
                "that were shared with you."
            ),
        ),
        ResponseTemplate(
            name="group_chat_summary",
            patterns=[
                "summarize the group chat",
                "what happened in the group",
                "recap group conversation",
                "group chat summary",
                "what did I miss in group",
                "summarize group messages",
                "catch me up on group chat",
            ],
            response=(
                "I'll provide a summary of the group chat, including key discussions, "
                "decisions made, and any action items."
            ),
        ),
        ResponseTemplate(
            name="who_mentioned_me",
            patterns=[
                "who mentioned me",
                "find messages mentioning my name",
                "was I mentioned in any chats",
                "search for mentions of me",
                "who talked about me",
                "find where I was mentioned",
                "any messages about me",
            ],
            response=(
                "I'll search your messages for mentions of your name "
                "and show you the relevant conversations."
            ),
        ),
        ResponseTemplate(
            name="important_messages",
            patterns=[
                "show important messages",
                "find urgent texts",
                "what messages need attention",
                "priority messages",
                "find important conversations",
                "urgent messages",
                "messages that need reply",
            ],
            response=(
                "I'll identify messages that may need your attention based on "
                "content, sender, and conversation context."
            ),
        ),
        ResponseTemplate(
            name="conversation_history",
            patterns=[
                "show full conversation with",
                "entire chat history with",
                "all messages with",
                "complete conversation with",
                "full message history with",
                "show all texts with",
                "entire chat with",
            ],
            response=(
                "I'll show you the complete conversation history with that person, "
                "starting from the earliest message."
            ),
        ),
        # ============================================================================
        # GROUP CHAT TEMPLATES
        # ============================================================================
        # --- Event Planning ---
        ResponseTemplate(
            name="group_event_when_works",
            patterns=[
                "when works for everyone",
                "what time works for everyone",
                "when is everyone free",
                "when are you all available",
                "what works for the group",
                "when can everyone make it",
                "what day works for all",
            ],
            response="I'm flexible! What times are you all thinking?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_day_proposal",
            patterns=[
                "I can do Saturday",
                "Saturday works for me",
                "I'm free on Sunday",
                "Friday works",
                "I can make it on",
                "that day works for me",
                "I'm available then",
            ],
            response="That works for me too!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_conflict",
            patterns=[
                "that doesn't work for me",
                "I can't do that day",
                "I have a conflict",
                "that time doesn't work",
                "I'm busy then",
                "can we do a different day",
                "any other options",
            ],
            response="No worries! What other times work for you?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_locked_in",
            patterns=[
                "let's do Saturday then",
                "Saturday it is",
                "let's lock that in",
                "sounds like a plan",
                "we're all set then",
                "it's a date",
                "perfect, see everyone then",
            ],
            response="Sounds good! See everyone there!",
            is_group_template=True,
        ),
        # --- RSVP Coordination ---
        ResponseTemplate(
            name="group_rsvp_yes",
            patterns=[
                "count me in",
                "I'm in",
                "I'll be there",
                "yes I'm coming",
                "definitely coming",
                "I'm down",
                "sign me up",
                "add me to the list",
            ],
            response="Awesome, see you there!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_plus_one",
            patterns=[
                "I'll be there +1",
                "count me plus one",
                "I'm bringing someone",
                "can I bring a friend",
                "I'll be there with my partner",
                "plus one for me",
                "bringing my +1",
            ],
            response="Great, the more the merrier!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_no",
            patterns=[
                "can't make it",
                "I won't be able to come",
                "count me out",
                "I have to skip this one",
                "I can't come",
                "unfortunately I can't make it",
                "sorry I can't be there",
            ],
            response="No worries, we'll miss you! Maybe next time.",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_maybe",
            patterns=[
                "I might be able to come",
                "I'll try to make it",
                "tentative yes",
                "put me down as a maybe",
                "I'll let you know",
                "not sure yet",
                "I'll confirm later",
            ],
            response="Sounds good, just let us know when you can!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_headcount",
            patterns=[
                "who's coming",
                "how many people so far",
                "what's the headcount",
                "who's confirmed",
                "how many are coming",
                "who all is going",
                "what's the count",
            ],
            response="Let me check - I think we have a few confirmed so far!",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Poll Responses ---
        ResponseTemplate(
            name="group_poll_vote_a",
            patterns=[
                "I vote for option A",
                "option A",
                "I prefer A",
                "A for me",
                "going with A",
                "my vote is A",
                "definitely A",
            ],
            response="Got it, A it is for me too!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_vote_b",
            patterns=[
                "I vote for option B",
                "option B",
                "I prefer B",
                "B for me",
                "going with B",
                "my vote is B",
                "definitely B",
            ],
            response="B sounds good!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_either",
            patterns=[
                "either works for me",
                "I'm fine with both",
                "no preference",
                "both options work",
                "I can go either way",
                "happy with whatever",
                "any option is fine",
            ],
            response="Same here, flexible on this one!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_create",
            patterns=[
                "let's do a poll",
                "let's vote on it",
                "should we vote",
                "let's put it to a vote",
                "what does everyone think",
                "can we get everyone's input",
                "let's see what everyone wants",
            ],
            response="Good idea! What are the options?",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Group Logistics ---
        ResponseTemplate(
            name="group_logistics_who_bringing",
            patterns=[
                "who's bringing what",
                "what should I bring",
                "who's bringing food",
                "should I bring anything",
                "what do we need",
                "who's handling what",
                "what's everyone bringing",
            ],
            response="I can bring drinks! What else do we need?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_ill_handle",
            patterns=[
                "I'll handle the reservation",
                "I'll book it",
                "I can make the reservation",
                "I'll take care of it",
                "leave it to me",
                "I got this",
                "I'll set it up",
            ],
            response="You're the best! Thanks for handling that!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_location",
            patterns=[
                "where are we meeting",
                "what's the address",
                "where should we go",
                "any suggestions for a place",
                "where's the spot",
                "what venue",
                "location suggestions",
            ],
            response="Good question! Anyone have ideas?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_carpooling",
            patterns=[
                "anyone need a ride",
                "I can drive",
                "can someone pick me up",
                "let's carpool",
                "who's driving",
                "I need a ride",
                "anyone driving from downtown",
            ],
            response="I might need a ride! Where are you coming from?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_splitting_bill",
            patterns=[
                "let's split the bill",
                "how should we split it",
                "I'll venmo everyone",
                "everyone pay their share",
                "let's split evenly",
                "how much do I owe",
                "I'll send the payment request",
            ],
            response="Sounds fair! Just let me know the amount.",
            is_group_template=True,
        ),
        # --- Celebratory Messages ---
        ResponseTemplate(
            name="group_celebration_birthday",
            patterns=[
                "happy birthday",
                "happy bday",
                "hbd",
                "hope you have a great birthday",
                "wishing you a happy birthday",
                "birthday wishes",
                "have an amazing birthday",
            ],
            response="Happy birthday! Hope it's amazing! 🎉",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_congrats",
            patterns=[
                "congrats everyone",
                "congratulations to all",
                "way to go team",
                "we did it",
                "great job everyone",
                "congrats all around",
                "proud of everyone",
            ],
            response="Congrats all! We crushed it! 🎉",
            is_group_template=True,
            min_group_size=3,
        ),
        ResponseTemplate(
            name="group_celebration_individual",
            patterns=[
                "congrats",
                "congratulations",
                "so proud of you",
                "well done",
                "amazing job",
                "you did it",
                "so happy for you",
            ],
            response="Congrats! That's awesome! 🎉",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_holiday",
            patterns=[
                "happy holidays",
                "happy new year",
                "merry christmas",
                "happy thanksgiving",
                "happy easter",
                "have a great holiday",
                "enjoy the holidays",
            ],
            response="Happy holidays to everyone! 🎊",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_thanks",
            patterns=[
                "thanks everyone",
                "thank you all",
                "appreciate everyone",
                "thanks to all of you",
                "grateful for this group",
                "thanks for everything",
                "you all are the best",
            ],
            response="Aw, this group is the best! ❤️",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Information Sharing ---
        ResponseTemplate(
            name="group_info_fyi",
            patterns=[
                "fyi",
                "for your information",
                "just so everyone knows",
                "heads up",
                "just a heads up",
                "wanted to let you all know",
                "quick update",
            ],
            response="Thanks for the heads up!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_sharing",
            patterns=[
                "sharing with the group",
                "thought you'd all want to see this",
                "check this out everyone",
                "sharing this with everyone",
                "wanted to share this",
                "look what I found",
                "you all need to see this",
            ],
            response="Thanks for sharing! This is great!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_update",
            patterns=[
                "update for everyone",
                "quick update for the group",
                "here's what's happening",
                "status update",
                "letting everyone know",
                "keeping everyone posted",
                "just an update",
            ],
            response="Thanks for the update! Good to know.",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_reminder",
            patterns=[
                "reminder for everyone",
                "don't forget",
                "just a reminder",
                "quick reminder",
                "remember that",
                "reminding everyone",
                "friendly reminder",
            ],
            response="Thanks for the reminder!",
            is_group_template=True,
        ),
        # --- Large Group Specific (10+ people) ---
        ResponseTemplate(
            name="group_large_lost_track",
            patterns=[
                "sorry catching up on messages",
                "catching up on the chat",
                "so many messages",
                "what did I miss",
                "can someone summarize",
                "tldr of the chat",
                "filling in on missed messages",
            ],
            response="No worries! Here's the quick version...",
            is_group_template=True,
            min_group_size=10,
        ),
        ResponseTemplate(
            name="group_large_quiet_down",
            patterns=[
                "so many notifications",
                "my phone is blowing up",
                "this chat is active",
                "loving the energy",
                "lots of messages",
                "active chat today",
                "the group is popping",
            ],
            response="Haha right? Love this group's energy!",
            is_group_template=True,
            min_group_size=10,
        ),
        # --- Small Group Specific (3-5 people) ---
        ResponseTemplate(
            name="group_small_intimate",
            patterns=[
                "just us three",
                "just the four of us",
                "our little group",
                "the gang",
                "the crew",
                "the squad",
                "just us",
            ],
            response="Love our little crew! 💯",
            is_group_template=True,
            min_group_size=3,
            max_group_size=5,
        ),
    ]


def _load_templates() -> list[ResponseTemplate]:
    """Load response templates.

    Returns the built-in template set for template matching.
    """
    return _get_minimal_fallback_templates()


class TemplateMatcher:
    """Semantic template matcher using sentence embeddings.

    Computes cosine similarity between input prompt and template patterns.
    Returns best matching template if similarity exceeds threshold.

    Performance optimizations:
    - Pre-normalized pattern embeddings (computed once at init)
    - LRU cache for query embeddings (avoids re-encoding repeated queries)
    - Optimized dot product for cosine similarity (O(n) instead of O(n*d))
    """

    SIMILARITY_THRESHOLD = 0.7
    QUERY_CACHE_SIZE = 500

    def __init__(self, templates: list[ResponseTemplate] | None = None) -> None:
        """Initialize the template matcher.

        Args:
            templates: List of templates to use. Loads defaults if not provided.
        """
        self.templates = templates or _load_templates()
        self._pattern_embeddings: np.ndarray | None = None
        self._pattern_norms: np.ndarray | None = None  # Pre-computed norms
        self._pattern_to_template: list[tuple[str, ResponseTemplate]] = []
        self._embeddings_lock = threading.Lock()
        self._query_cache: EmbeddingCache[str, np.ndarray] = EmbeddingCache(
            maxsize=self.QUERY_CACHE_SIZE
        )

    def _ensure_embeddings(self) -> None:
        """Compute and cache embeddings for all template patterns.

        Uses double-check locking for thread-safe lazy initialization.
        Embeddings are normalized by the embedder for direct cosine similarity.
        """
        # Fast path: embeddings already computed
        if self._pattern_embeddings is not None:
            return

        # Slow path: acquire lock and double-check
        with self._embeddings_lock:
            # Double-check after acquiring lock
            if self._pattern_embeddings is not None:
                return

            embedder = _get_sentence_model()

            # Collect all patterns with their templates
            all_patterns = []
            pattern_to_template: list[tuple[str, ResponseTemplate]] = []
            for template in self.templates:
                for pattern in template.patterns:
                    all_patterns.append(pattern)
                    pattern_to_template.append((pattern, template))

            embeddings = None
            norms = None
            try:
                # Compute embeddings in batch (normalized by default)
                embeddings = embedder.encode(all_patterns, normalize=True)

                # Pre-compute norms for faster cosine similarity
                # Even though embeddings are normalized, we keep this for consistency
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.where(norms == 0, 1, norms)

                # Assign atomically
                self._pattern_to_template = pattern_to_template
                self._pattern_norms = norms.flatten()
                self._pattern_embeddings = embeddings
                logger.info("Computed embeddings for %d patterns", len(all_patterns))
            except Exception as e:
                # Clean up partial results to avoid memory leak
                embeddings = None
                norms = None
                logger.error("Failed to compute pattern embeddings: %s", e)
                raise

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query, using cache if available.

        Args:
            query: Query string to encode

        Returns:
            Query embedding as numpy array (normalized)
        """
        # Create cache key from query hash
        cache_key = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()

        # Check cache first
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Encode and cache (normalized by default)
        embedder = _get_sentence_model()
        embedding_result = embedder.encode([query], normalize=True)[0]
        # Cast to ndarray to satisfy mypy (encode returns Any)
        embedding: np.ndarray = np.asarray(embedding_result)
        self._query_cache.set(cache_key, embedding)
        return embedding

    def match(self, query: str, track_analytics: bool = True) -> TemplateMatch | None:
        """Find best matching template for a query.

        Args:
            query: Input prompt to match against templates
            track_analytics: Whether to record this query in template analytics

        Returns:
            TemplateMatch if similarity >= threshold, None otherwise.
            Returns None if sentence model fails to load (falls back to model generation).
        """
        analytics = get_template_analytics() if track_analytics else None

        try:
            self._ensure_embeddings()

            # Type guard: _ensure_embeddings guarantees this is not None
            pattern_embeddings = self._pattern_embeddings
            if pattern_embeddings is None:
                return None

            # Compute norms on-the-fly if not pre-computed (for backward compat with tests)
            pattern_norms = self._pattern_norms
            if pattern_norms is None:
                pattern_norms = np.linalg.norm(pattern_embeddings, axis=1)
                pattern_norms = np.where(pattern_norms == 0, 1, pattern_norms)

            # Get query embedding (cached if previously seen)
            query_embedding = self._get_query_embedding(query)
            query_norm = np.linalg.norm(query_embedding)

            if query_norm == 0:
                return None

            # Compute similarities in batch (optimized dot product)
            # similarities.shape = (n_patterns,)
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                pattern_norms * query_norm
            )

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])

            if best_similarity >= self.SIMILARITY_THRESHOLD:
                matched_pattern, template = self._pattern_to_template[best_idx]

                # Update template usage if matched
                if template.name.startswith("custom_"):
                    store = get_custom_template_store()
                    store.increment_usage(template.name.replace("custom_", ""))

                # Record hit in analytics
                if analytics:
                    analytics.record_hit(template.name, best_similarity)

                logger.debug(
                    "Template match: %s (similarity: %.3f)",
                    template.name,
                    best_similarity,
                )
                return TemplateMatch(
                    template=template,
                    similarity=best_similarity,
                    matched_pattern=matched_pattern,
                )

            # Record miss in analytics
            if analytics:
                analytics.record_miss(query, best_similarity)

            return None

        except SentenceModelError:
            logger.warning("Template matching unavailable, falling back to model generation")
            return None

    def _template_matches_group_size(
        self, template: ResponseTemplate, group_size: int | None
    ) -> bool:
        """Check if a template is appropriate for a given group size.

        Args:
            template: The template to check
            group_size: Number of participants in the chat (None means unknown)

        Returns:
            True if template is appropriate, False otherwise
        """
        # If group_size is None, only non-group templates are appropriate
        if group_size is None:
            return not template.is_group_template

        # If no constraints, it's appropriate
        if template.min_group_size is None and template.max_group_size is None:
            # If specifically marked as group template, needs group_size >= 3
            if template.is_group_template:
                return group_size >= 3
            return True

        # Check min constraint
        if template.min_group_size is not None and group_size < template.min_group_size:
            return False

        # Check max constraint
        if template.max_group_size is not None and group_size > template.max_group_size:
            return False

        return True

    def match_with_context(
        self, query: str, group_size: int | None = None, track_analytics: bool = True
    ) -> TemplateMatch | None:
        """Find best matching template considering conversation context.

        Filters templates based on group size and other context before matching.

        Args:
            query: Input prompt to match
            group_size: Number of participants in the chat
            track_analytics: Whether to record in analytics

        Returns:
            TemplateMatch or None
        """
        # If no group size provided, use standard match
        if group_size is None:
            return self.match(query, track_analytics)

        try:
            self._ensure_embeddings()
            pattern_embeddings = self._pattern_embeddings
            if pattern_embeddings is None:
                return None

            # Compute norms on-the-fly if not pre-computed
            pattern_norms = self._pattern_norms
            if pattern_norms is None:
                pattern_norms = np.linalg.norm(pattern_embeddings, axis=1)
                pattern_norms = np.where(pattern_norms == 0, 1, pattern_norms)

            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return None

            # Score ALL templates first
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                pattern_norms * query_norm
            )

            # Now find the best match that also satisfies group size constraints
            best_match = None
            best_similarity = -1.0

            for i, (matched_pattern, template) in enumerate(self._pattern_to_template):
                similarity = float(similarities[i])

                # Must meet base threshold
                if similarity < self.SIMILARITY_THRESHOLD:
                    continue

                # Check group size constraints
                if not self._template_matches_group_size(template, group_size):
                    continue

                # Boost similarity for specific group templates if we're in a group
                effective_similarity = similarity
                if group_size >= 3 and template.is_group_template:
                    # Give preference to group-specific templates in group chats
                    effective_similarity = min(1.0, similarity + 0.05)

                if effective_similarity > best_similarity:
                    best_similarity = effective_similarity
                    best_match = TemplateMatch(
                        template=template,
                        similarity=similarity,  # Store actual similarity, not boosted
                        matched_pattern=matched_pattern,
                    )

            if best_match is not None:
                # Record hit in analytics
                if track_analytics:
                    get_template_analytics().record_hit(
                        best_match.template.name, best_match.similarity
                    )

                logger.debug(
                    "Template match with context: %s (similarity: %.3f, group_size: %s)",
                    best_match.template.name,
                    best_match.similarity,
                    group_size,
                )

            return best_match

        except SentenceModelError:
            logger.warning("Template matching unavailable, falling back to model generation")
            return None

    def get_group_templates(self) -> list[ResponseTemplate]:
        """Get all group-specific templates."""
        return [t for t in self.templates if t.is_group_template]

    def get_templates_for_group_size(self, group_size: int) -> list[ResponseTemplate]:
        """Get templates appropriate for a specific group size."""
        return [t for t in self.templates if self._template_matches_group_size(t, group_size)]

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        self._pattern_embeddings = None
        self._pattern_norms = None
        self._pattern_to_template = []
        self._query_cache.clear()
        logger.debug("Template matcher cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get query cache statistics."""
        return self._query_cache.stats()
