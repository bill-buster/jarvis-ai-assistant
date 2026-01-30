"""Conversation coherence detection for JARVIS v2.

Detects topic shifts in conversations using embedding similarity.
Uses TextTiling-inspired approach with our existing embedding model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConversationSegment:
    """A coherent segment of conversation."""

    messages: list[dict]
    start_idx: int
    end_idx: int
    avg_coherence: float


def get_embedding_model():
    """Lazy import embedding model."""
    from core.embeddings.model import get_embedding_model
    return get_embedding_model()


def detect_topic_breaks(
    messages: list[dict],
    coherence_threshold: float = 0.35,
    time_gap_hours: float = 4.0,
) -> list[int]:
    """Detect topic breaks in a conversation.

    Uses embedding similarity between consecutive messages.
    A break is detected when:
    1. Coherence drops below threshold AND time gap > 5 min (avoid splitting rapid-fire messages)
    2. OR time gap exceeds threshold (different conversation session)

    Args:
        messages: List of messages [{"text": ..., "timestamp": ..., "is_from_me": bool}]
        coherence_threshold: Min similarity to consider same topic (0-1)
        time_gap_hours: Hours gap that always indicates new conversation

    Returns:
        List of indices where topic breaks occur
    """
    if len(messages) < 3:
        return []

    breaks = []
    model = get_embedding_model()

    # Get texts and timestamps
    texts = []
    timestamps = []
    for msg in messages:
        text = msg.get("text", "")
        if text and len(text) > 2:
            texts.append(text)
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = None
            timestamps.append(ts)

    if len(texts) < 3:
        return []

    # Compute embeddings
    embeddings = model.embed_batch(texts)

    # Compute pairwise coherence (cosine similarity between consecutive messages)
    coherence_scores = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        coherence_scores.append(sim)

    # Detect breaks based on coherence + time
    for i, score in enumerate(coherence_scores):
        time_gap = None
        if timestamps[i] and timestamps[i + 1]:
            time_gap = timestamps[i + 1] - timestamps[i]

        # Long time gap (hours) = always a break (different session)
        if time_gap and time_gap > timedelta(hours=time_gap_hours):
            logger.debug(f"Topic break at {i+1}: time gap {time_gap}")
            breaks.append(i + 1)
            continue

        # Low coherence + some time gap (>5 min) = topic shift within session
        if score < coherence_threshold:
            if time_gap and time_gap > timedelta(minutes=5):
                logger.debug(f"Topic break at {i+1}: coherence {score:.2f} + gap {time_gap}")
                breaks.append(i + 1)

    if breaks:
        logger.info(f"Detected {len(breaks)} topic breaks in {len(messages)} messages")

    return breaks


def get_recent_coherent_segment(
    messages: list[dict],
    coherence_threshold: float = 0.35,
    time_gap_hours: float = 4.0,
    max_messages: int = 10,
) -> list[dict]:
    """Get the most recent coherent segment of conversation.

    Finds topic breaks and returns messages from the last break onwards.
    This gives the LLM only contextually relevant messages.

    Args:
        messages: All messages (oldest first)
        coherence_threshold: Min similarity for same topic
        time_gap_minutes: Max gap for same topic
        max_messages: Max messages to return

    Returns:
        Recent coherent messages (oldest first)
    """
    if len(messages) <= max_messages:
        return messages

    # Find breaks in recent messages
    recent = messages[-max_messages * 2:]  # Look at more to find breaks
    breaks = detect_topic_breaks(recent, coherence_threshold, time_gap_hours)

    if not breaks:
        # No breaks found, return last max_messages
        return messages[-max_messages:]

    # Get messages from last break onwards
    last_break = breaks[-1]
    segment_start = len(messages) - len(recent) + last_break
    segment = messages[segment_start:]

    # Limit to max_messages
    if len(segment) > max_messages:
        segment = segment[-max_messages:]

    return segment


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
