"""Topics API endpoints.

Provides automatic topic detection for conversations based on
message content analysis using semantic similarity.

Topics are cached with a 30-minute TTL and refreshed on-demand.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_imessage_reader
from integrations.imessage import ChatDBReader
from jarvis.metrics import TTLCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["topics"])


class TopicType(str, Enum):
    """Types of conversation topics."""

    SCHEDULING = "scheduling"
    QUESTIONS = "questions"
    LINKS = "links"
    PHOTOS = "photos"
    EVENTS = "events"
    SMALL_TALK = "small_talk"
    WORK = "work"
    FOOD = "food"
    TRAVEL = "travel"
    FAMILY = "family"


# Topic detection examples for semantic similarity matching
TOPIC_EXAMPLES: dict[TopicType, list[str]] = {
    TopicType.SCHEDULING: [
        "what time works for you",
        "are you free tomorrow",
        "let's meet at 3pm",
        "can we reschedule",
        "when are you available",
        "I'm busy on Monday",
        "how about next week",
        "does Saturday work",
        "let me check my calendar",
        "set a reminder",
        "appointment confirmed",
        "running late",
        "on my way",
        "be there in 10",
    ],
    TopicType.QUESTIONS: [
        "what do you think",
        "can you help me with",
        "do you know how to",
        "where is the",
        "why did you",
        "how does this work",
        "what should I do",
        "any ideas",
        "have you ever",
        "did you hear about",
        "what happened",
        "is it true that",
    ],
    TopicType.LINKS: [
        "check out this link",
        "here's the url",
        "https://",
        "http://",
        "www.",
        "sent you a link",
        "click here",
        "found this article",
        "watch this video",
        "look at this website",
        "youtube.com",
        "shared a link",
    ],
    TopicType.PHOTOS: [
        "sent you a photo",
        "here's a picture",
        "check out this pic",
        "look at this image",
        "attached a photo",
        "screenshot",
        "took this picture",
        "selfie",
        "here's what it looks like",
        "see the attachment",
        "photo of",
        "image attached",
    ],
    TopicType.EVENTS: [
        "party this weekend",
        "birthday celebration",
        "wedding invitation",
        "concert tickets",
        "game tonight",
        "festival",
        "holiday plans",
        "graduation ceremony",
        "anniversary dinner",
        "baby shower",
        "reunion",
        "gathering at",
    ],
    TopicType.SMALL_TALK: [
        "how are you",
        "what's up",
        "hey there",
        "good morning",
        "have a nice day",
        "how's it going",
        "long time no see",
        "nice weather today",
        "how was your weekend",
        "take care",
        "talk to you later",
        "miss you",
        "thinking of you",
        "haha",
        "lol",
    ],
    TopicType.WORK: [
        "meeting tomorrow",
        "project deadline",
        "boss wants",
        "coworker",
        "office",
        "work from home",
        "presentation",
        "client meeting",
        "quarterly report",
        "promotion",
        "job interview",
        "salary",
        "overtime",
    ],
    TopicType.FOOD: [
        "dinner plans",
        "lunch break",
        "restaurant recommendation",
        "what should we eat",
        "cooking tonight",
        "recipe for",
        "hungry",
        "coffee",
        "brunch",
        "takeout",
        "delivery",
        "reservation at",
    ],
    TopicType.TRAVEL: [
        "flight booked",
        "hotel reservation",
        "road trip",
        "vacation plans",
        "packing for",
        "airport",
        "train tickets",
        "travel itinerary",
        "visiting",
        "tourism",
        "sightseeing",
        "passport",
    ],
    TopicType.FAMILY: [
        "mom said",
        "dad called",
        "kids are",
        "sister's birthday",
        "brother wants",
        "family dinner",
        "grandparents visiting",
        "cousin's wedding",
        "family reunion",
        "parents anniversary",
        "nephew",
        "niece",
    ],
}

# Color mapping for topic types (used by frontend)
TOPIC_COLORS: dict[TopicType, str] = {
    TopicType.SCHEDULING: "blue",
    TopicType.QUESTIONS: "green",
    TopicType.LINKS: "purple",
    TopicType.PHOTOS: "pink",
    TopicType.EVENTS: "orange",
    TopicType.SMALL_TALK: "gray",
    TopicType.WORK: "indigo",
    TopicType.FOOD: "amber",
    TopicType.TRAVEL: "cyan",
    TopicType.FAMILY: "rose",
}


@dataclass
class DetectedTopic:
    """A detected topic with confidence score."""

    topic: TopicType
    confidence: float
    color: str


# Response schemas
class TopicResponse(BaseModel):
    """A detected topic for a conversation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "scheduling",
                "confidence": 0.85,
                "color": "blue",
                "display_name": "Scheduling",
            }
        }
    )

    topic: str = Field(
        ...,
        description="Topic type identifier",
        examples=["scheduling", "questions", "photos"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this topic (0.0 to 1.0)",
        examples=[0.85, 0.72],
    )
    color: str = Field(
        ...,
        description="Color for displaying this topic tag",
        examples=["blue", "green", "purple"],
    )
    display_name: str = Field(
        ...,
        description="Human-readable topic name",
        examples=["Scheduling", "Questions", "Photos"],
    )


class TopicsResponse(BaseModel):
    """Response containing detected topics for a conversation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "topics": [
                    {
                        "topic": "scheduling",
                        "confidence": 0.85,
                        "color": "blue",
                        "display_name": "Scheduling",
                    },
                    {
                        "topic": "food",
                        "confidence": 0.72,
                        "color": "amber",
                        "display_name": "Food",
                    },
                ],
                "all_topics": [
                    {
                        "topic": "scheduling",
                        "confidence": 0.85,
                        "color": "blue",
                        "display_name": "Scheduling",
                    },
                    {
                        "topic": "food",
                        "confidence": 0.72,
                        "color": "amber",
                        "display_name": "Food",
                    },
                    {
                        "topic": "small_talk",
                        "confidence": 0.45,
                        "color": "gray",
                        "display_name": "Small Talk",
                    },
                ],
                "cached": False,
                "message_count_analyzed": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID",
        examples=["chat123456789"],
    )
    topics: list[TopicResponse] = Field(
        ...,
        description="Top 2 detected topics for display",
    )
    all_topics: list[TopicResponse] = Field(
        ...,
        description="All detected topics with confidence scores (for tooltip)",
    )
    cached: bool = Field(
        ...,
        description="Whether this result was served from cache",
    )
    message_count_analyzed: int = Field(
        ...,
        description="Number of messages analyzed for topic detection",
        examples=[50, 100],
        ge=0,
    )


class TopicDetector:
    """Detects conversation topics using semantic similarity.

    Uses sentence embeddings to match messages against topic examples.
    Thread-safe with lazy initialization of embeddings.
    """

    CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence to include a topic
    TOP_TOPICS_COUNT = 2  # Number of topics to return for display

    def __init__(self) -> None:
        """Initialize the topic detector."""
        self._topic_centroids: dict[TopicType, np.ndarray] | None = None
        self._lock = threading.Lock()

    def _get_sentence_model(self):  # type: ignore[no-untyped-def]
        """Get the sentence transformer model from templates module."""
        from models.templates import SentenceModelError, _get_sentence_model

        try:
            return _get_sentence_model()
        except SentenceModelError:
            logger.warning("Failed to load sentence model for topic detection")
            raise

    def _ensure_centroids_computed(self) -> None:
        """Compute and cache centroids for all topic examples."""
        if self._topic_centroids is not None:
            return

        with self._lock:
            if self._topic_centroids is not None:
                return

            model = self._get_sentence_model()
            topic_centroids: dict[TopicType, np.ndarray] = {}

            for topic_type, examples in TOPIC_EXAMPLES.items():
                embeddings = model.encode(examples, convert_to_numpy=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                topic_centroids[topic_type] = centroid

            self._topic_centroids = topic_centroids
            logger.info("Computed topic centroids for %d topics", len(TOPIC_EXAMPLES))

    def detect_topics(self, messages: list[str]) -> list[DetectedTopic]:
        """Detect topics from a list of message texts.

        Args:
            messages: List of message text contents

        Returns:
            List of detected topics sorted by confidence (highest first)
        """
        if not messages:
            return []

        try:
            self._ensure_centroids_computed()
        except Exception:
            logger.warning("Topic detection unavailable")
            return []

        if self._topic_centroids is None:
            return []

        try:
            model = self._get_sentence_model()

            # Combine messages into chunks for more efficient processing
            # Use last N messages for recency bias
            recent_messages = messages[-100:] if len(messages) > 100 else messages
            combined_text = " ".join(recent_messages)

            # Also check for URL patterns directly
            has_links = bool(re.search(r"https?://|www\.", combined_text, re.IGNORECASE))

            # Encode the combined text
            text_embedding = model.encode([combined_text], convert_to_numpy=True)[0]
            text_norm = text_embedding / np.linalg.norm(text_embedding)

            # Compute similarity to each topic
            topic_scores: dict[TopicType, float] = {}
            for topic_type, centroid in self._topic_centroids.items():
                similarity = float(np.dot(text_norm, centroid))
                topic_scores[topic_type] = similarity

            # Boost links topic if URLs are detected
            if has_links and TopicType.LINKS in topic_scores:
                topic_scores[TopicType.LINKS] = min(1.0, topic_scores[TopicType.LINKS] + 0.3)

            # Filter and sort by confidence
            detected = []
            for topic_type, confidence in topic_scores.items():
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    detected.append(
                        DetectedTopic(
                            topic=topic_type,
                            confidence=confidence,
                            color=TOPIC_COLORS[topic_type],
                        )
                    )

            detected.sort(key=lambda t: t.confidence, reverse=True)
            return detected

        except Exception:
            logger.exception("Error during topic detection")
            return []


# Global topic detector instance
_topic_detector: TopicDetector | None = None
_detector_lock = threading.Lock()


def get_topic_detector() -> TopicDetector:
    """Get the singleton TopicDetector instance."""
    global _topic_detector
    if _topic_detector is None:
        with _detector_lock:
            if _topic_detector is None:
                _topic_detector = TopicDetector()
    return _topic_detector


# Topic cache with 30-minute TTL
_topic_cache: TTLCache | None = None


def get_topic_cache() -> TTLCache:
    """Get the topic cache (TTL: 30 minutes)."""
    global _topic_cache
    if _topic_cache is None:
        _topic_cache = TTLCache(ttl_seconds=1800.0, maxsize=200)  # 30 minutes
    return _topic_cache


def _format_display_name(topic: TopicType) -> str:
    """Format topic type as display name."""
    return topic.value.replace("_", " ").title()


@router.post(
    "/{chat_id}/topics",
    response_model=TopicsResponse,
    response_model_exclude_unset=True,
    response_description="Detected topics for the conversation",
    summary="Analyze conversation topics",
    responses={
        200: {
            "description": "Topics detected successfully",
            "content": {
                "application/json": {
                    "example": {
                        "chat_id": "chat123456789",
                        "topics": [
                            {
                                "topic": "scheduling",
                                "confidence": 0.85,
                                "color": "blue",
                                "display_name": "Scheduling",
                            }
                        ],
                        "all_topics": [],
                        "cached": False,
                        "message_count_analyzed": 50,
                    }
                }
            },
        },
    },
)
def analyze_topics(
    chat_id: str,
    limit: int = Query(
        default=50,
        ge=10,
        le=200,
        description="Number of recent messages to analyze",
        examples=[50, 100],
    ),
    refresh: bool = Query(
        default=False,
        description="Force refresh cache and re-analyze",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> TopicsResponse:
    """Analyze conversation to detect topics.

    Uses semantic similarity to identify common topics like scheduling,
    questions, sharing links/photos, events, and small talk.

    Results are cached for 30 minutes. Use `refresh=true` to force re-analysis.

    **Topic Types:**
    - `scheduling`: Time/meeting coordination (blue)
    - `questions`: Questions and help requests (green)
    - `links`: Shared URLs and websites (purple)
    - `photos`: Photo/image sharing (pink)
    - `events`: Events and celebrations (orange)
    - `small_talk`: Casual conversation (gray)
    - `work`: Work-related topics (indigo)
    - `food`: Food and dining (amber)
    - `travel`: Travel and trips (cyan)
    - `family`: Family-related (rose)

    **Response:**
    - `topics`: Top 2 topics for sidebar display
    - `all_topics`: All detected topics (for hover tooltip)
    - `confidence`: Score from 0.0 to 1.0 indicating match strength

    Args:
        chat_id: The conversation ID to analyze
        limit: Number of recent messages to analyze (10-200, default 50)
        refresh: Force refresh the cached topics

    Returns:
        TopicsResponse with detected topics and confidence scores
    """
    cache = get_topic_cache()
    cache_key = f"topics:{chat_id}:{limit}"

    # Check cache unless refresh requested
    if not refresh:
        found, cached_result = cache.get(cache_key)
        if found:
            result = cached_result
            result["cached"] = True
            return TopicsResponse(**result)

    # Fetch recent messages
    messages = reader.get_messages(chat_id=chat_id, limit=limit)
    message_texts = [m.text for m in messages if m.text and m.text.strip()]

    # Detect topics
    detector = get_topic_detector()
    detected_topics = detector.detect_topics(message_texts)

    # Format response
    all_topic_responses = [
        TopicResponse(
            topic=t.topic.value,
            confidence=round(t.confidence, 3),
            color=t.color,
            display_name=_format_display_name(t.topic),
        )
        for t in detected_topics
    ]

    # Top 2 for display
    top_topics = all_topic_responses[: TopicDetector.TOP_TOPICS_COUNT]

    result = {
        "chat_id": chat_id,
        "topics": [t.model_dump() for t in top_topics],
        "all_topics": [t.model_dump() for t in all_topic_responses],
        "cached": False,
        "message_count_analyzed": len(message_texts),
    }

    # Cache the result
    cache.set(cache_key, result)

    return TopicsResponse(**result)
