"""Embedding-based Relationship Profiling.

Uses vector embeddings to analyze communication patterns:
- Topic clusters from message embeddings
- Semantic similarity between your style and theirs
- Communication dynamics (who initiates, response patterns)
- Topic evolution over time

Complements the keyword-based profiling in relationships.py with
learned representations that capture semantic meaning.

Profile storage: ~/.jarvis/embedding_profiles/{contact_hash}.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

EMBEDDING_PROFILES_DIR = Path.home() / ".jarvis" / "embedding_profiles"
MIN_MESSAGES_FOR_EMBEDDING_PROFILE = 30  # Need enough data for meaningful clusters
PROFILE_VERSION = "1.0.0"

# Clustering config
DEFAULT_NUM_CLUSTERS = 5  # Number of topic clusters to extract
MIN_CLUSTER_SIZE = 3  # Minimum messages per cluster


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TopicCluster:
    """A cluster of semantically similar messages representing a topic.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        centroid: The average embedding vector (serialized as list).
        sample_messages: Representative messages from this cluster.
        message_count: Number of messages in the cluster.
        from_me_ratio: Ratio of messages from you vs them (0-1).
        label: Optional human-readable label.
    """

    cluster_id: int
    centroid: list[float]
    sample_messages: list[str]
    message_count: int
    from_me_ratio: float
    label: str | None = None


@dataclass
class CommunicationDynamics:
    """Semantic analysis of communication patterns.

    Attributes:
        your_avg_embedding: Average embedding of your messages.
        their_avg_embedding: Average embedding of their messages.
        style_similarity: Cosine similarity between your and their avg embeddings.
        initiation_ratio: Ratio of conversations you initiate (0-1).
        topic_diversity: Entropy of topic cluster distribution.
        response_semantic_shift: How much your responses differ from triggers.
    """

    your_avg_embedding: list[float]
    their_avg_embedding: list[float]
    style_similarity: float
    initiation_ratio: float
    topic_diversity: float
    response_semantic_shift: float


@dataclass
class EmbeddingProfile:
    """Complete embedding-based relationship profile.

    Attributes:
        contact_id: Hashed contact identifier.
        contact_name: Display name if available.
        topic_clusters: List of discovered topic clusters.
        dynamics: Communication dynamics analysis.
        message_count: Total messages analyzed.
        embedding_model: Model used for embeddings.
        last_updated: ISO timestamp of last update.
        version: Profile format version.
    """

    contact_id: str
    contact_name: str | None = None
    topic_clusters: list[TopicCluster] = field(default_factory=list)
    dynamics: CommunicationDynamics | None = None
    message_count: int = 0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    last_updated: str = ""
    version: str = PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "topic_clusters": [asdict(tc) for tc in self.topic_clusters],
            "dynamics": asdict(self.dynamics) if self.dynamics else None,
            "message_count": self.message_count,
            "embedding_model": self.embedding_model,
            "last_updated": self.last_updated,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingProfile:
        """Create profile from dictionary."""
        topic_clusters = [TopicCluster(**tc) for tc in data.get("topic_clusters", [])]
        dynamics_data = data.get("dynamics")
        dynamics = CommunicationDynamics(**dynamics_data) if dynamics_data else None

        return cls(
            contact_id=data["contact_id"],
            contact_name=data.get("contact_name"),
            topic_clusters=topic_clusters,
            dynamics=dynamics,
            message_count=data.get("message_count", 0),
            embedding_model=data.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            last_updated=data.get("last_updated", ""),
            version=data.get("version", PROFILE_VERSION),
        )


# =============================================================================
# Embedding Analysis Functions
# =============================================================================


def _hash_contact_id(contact_id: str) -> str:
    """Create a stable hash for contact ID storage."""
    return hashlib.sha256(contact_id.encode("utf-8")).hexdigest()[:16]


def _compute_embeddings(
    texts: list[str],
    embedder: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute normalized embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        embedder: Embedding model with encode() method.
        batch_size: Batch size for encoding.

    Returns:
        Normalized embedding matrix (n_texts x embedding_dim).
    """
    if not texts:
        return np.array([])

    embeddings = embedder.encode(texts, normalize=True)
    return embeddings.astype(np.float32)


def _cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings using K-means.

    Args:
        embeddings: Embedding matrix (n_samples x embedding_dim).
        n_clusters: Number of clusters to create.
        min_cluster_size: Minimum samples per cluster.

    Returns:
        Tuple of (cluster_labels, cluster_centroids).
    """
    from sklearn.cluster import KMeans

    # Adjust clusters if not enough data
    n_samples = len(embeddings)
    effective_clusters = min(n_clusters, n_samples // min_cluster_size)
    effective_clusters = max(1, effective_clusters)

    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    return labels, centroids


def _compute_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    distribution = np.array(distribution)
    distribution = distribution / distribution.sum()  # Normalize
    # Filter out zeros to avoid log(0)
    distribution = distribution[distribution > 0]
    return float(-np.sum(distribution * np.log2(distribution)))


def _extract_sample_messages(
    messages: list[str],
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int = 3,
) -> list[str]:
    """Extract representative sample messages from a cluster.

    Args:
        messages: All message texts.
        labels: Cluster labels for each message.
        cluster_id: Which cluster to sample from.
        n_samples: Number of samples to extract.

    Returns:
        List of sample message texts.
    """
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return []

    # Take evenly spaced samples
    step = max(1, len(cluster_indices) // n_samples)
    sample_indices = cluster_indices[::step][:n_samples]

    return [messages[i][:200] for i in sample_indices]  # Truncate long messages


def build_embedding_profile(
    contact_id: str,
    messages: list[Any],
    embedder: Any,
    contact_name: str | None = None,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
) -> EmbeddingProfile:
    """Build an embedding-based relationship profile.

    Args:
        contact_id: Unique identifier for the contact.
        messages: List of Message objects with .text and .is_from_me attributes.
        embedder: Embedding model with encode() method.
        contact_name: Optional display name for the contact.
        n_clusters: Number of topic clusters to extract.

    Returns:
        EmbeddingProfile with semantic analysis.
    """
    hashed_id = _hash_contact_id(contact_id)

    # Filter messages with text
    valid_messages = [m for m in messages if m.text and len(m.text.strip()) > 2]

    if len(valid_messages) < MIN_MESSAGES_FOR_EMBEDDING_PROFILE:
        logger.info(
            "Not enough messages for embedding profile (%d < %d)",
            len(valid_messages),
            MIN_MESSAGES_FOR_EMBEDDING_PROFILE,
        )
        return EmbeddingProfile(
            contact_id=hashed_id,
            contact_name=contact_name,
            message_count=len(valid_messages),
            last_updated=datetime.now().isoformat(),
        )

    # Extract texts and metadata
    texts = [m.text for m in valid_messages]
    is_from_me = [m.is_from_me for m in valid_messages]

    # Compute embeddings
    logger.info("Computing embeddings for %d messages", len(texts))
    embeddings = _compute_embeddings(texts, embedder)

    # Cluster messages by topic
    logger.info("Clustering into %d topic clusters", n_clusters)
    labels, centroids = _cluster_embeddings(embeddings, n_clusters)

    # Build topic clusters
    topic_clusters = []
    for cluster_id in range(len(centroids)):
        cluster_mask = labels == cluster_id
        cluster_count = cluster_mask.sum()

        if cluster_count < MIN_CLUSTER_SIZE:
            continue

        # Compute ratio of "from me" messages in this cluster
        cluster_from_me = np.array(is_from_me)[cluster_mask]
        from_me_ratio = cluster_from_me.sum() / len(cluster_from_me)

        # Get sample messages
        samples = _extract_sample_messages(texts, labels, cluster_id)

        topic_clusters.append(
            TopicCluster(
                cluster_id=cluster_id,
                centroid=centroids[cluster_id].tolist(),
                sample_messages=samples,
                message_count=int(cluster_count),
                from_me_ratio=round(float(from_me_ratio), 3),
            )
        )

    # Compute communication dynamics
    my_mask = np.array(is_from_me)
    their_mask = ~my_mask

    if my_mask.sum() > 0 and their_mask.sum() > 0:
        my_embeddings = embeddings[my_mask]
        their_embeddings = embeddings[their_mask]

        my_avg = my_embeddings.mean(axis=0)
        their_avg = their_embeddings.mean(axis=0)

        # Style similarity (cosine since embeddings are normalized)
        style_sim = float(np.dot(my_avg, their_avg))

        # Topic diversity (entropy of cluster distribution)
        cluster_counts = np.bincount(labels, minlength=len(centroids))
        topic_diversity = _compute_entropy(cluster_counts)

        # Response semantic shift (avg distance between consecutive messages)
        response_shifts = []
        for i in range(1, len(embeddings)):
            if is_from_me[i] != is_from_me[i - 1]:
                # This is a response - compute semantic distance
                dist = 1 - float(np.dot(embeddings[i], embeddings[i - 1]))
                response_shifts.append(dist)

        avg_shift = sum(response_shifts) / len(response_shifts) if response_shifts else 0.0

        # Initiation ratio (who starts conversations)
        # Simple heuristic: first message in each "session" (gap > 1 hour)
        initiations_me = 0
        initiations_them = 0
        prev_time = None

        for m in valid_messages:
            if prev_time is None or (m.date - prev_time).total_seconds() > 3600:
                # New conversation session
                if m.is_from_me:
                    initiations_me += 1
                else:
                    initiations_them += 1
            prev_time = m.date

        total_initiations = initiations_me + initiations_them
        initiation_ratio = initiations_me / total_initiations if total_initiations > 0 else 0.5

        dynamics = CommunicationDynamics(
            your_avg_embedding=my_avg.tolist(),
            their_avg_embedding=their_avg.tolist(),
            style_similarity=round(style_sim, 3),
            initiation_ratio=round(float(initiation_ratio), 3),
            topic_diversity=round(topic_diversity, 3),
            response_semantic_shift=round(avg_shift, 3),
        )
    else:
        dynamics = None

    return EmbeddingProfile(
        contact_id=hashed_id,
        contact_name=contact_name,
        topic_clusters=topic_clusters,
        dynamics=dynamics,
        message_count=len(valid_messages),
        last_updated=datetime.now().isoformat(),
    )


# =============================================================================
# Profile Storage Functions
# =============================================================================


def _ensure_profiles_dir() -> Path:
    """Ensure the embedding profiles directory exists."""
    EMBEDDING_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDING_PROFILES_DIR


def _get_profile_path(contact_id: str) -> Path:
    """Get the file path for a contact's embedding profile."""
    hashed_id = _hash_contact_id(contact_id)
    return _ensure_profiles_dir() / f"{hashed_id}.json"


def save_embedding_profile(profile: EmbeddingProfile) -> bool:
    """Save an embedding profile to disk.

    Args:
        profile: The EmbeddingProfile to save.

    Returns:
        True if saved successfully.
    """
    try:
        profile_path = _ensure_profiles_dir() / f"{profile.contact_id}.json"
        with profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return True
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to save embedding profile: %s", e)
        return False


def load_embedding_profile(contact_id: str) -> EmbeddingProfile | None:
    """Load an embedding profile from disk.

    Args:
        contact_id: The contact identifier (will be hashed).

    Returns:
        EmbeddingProfile if found, None otherwise.
    """
    profile_path = _get_profile_path(contact_id)

    if not profile_path.exists():
        return None

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return EmbeddingProfile.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load embedding profile: %s", e)
        return None


def delete_embedding_profile(contact_id: str) -> bool:
    """Delete an embedding profile from disk."""
    profile_path = _get_profile_path(contact_id)
    try:
        if profile_path.exists():
            profile_path.unlink()
        return True
    except OSError:
        return False


def list_embedding_profiles() -> list[str]:
    """List all saved embedding profile IDs."""
    try:
        _ensure_profiles_dir()
        return [p.stem for p in EMBEDDING_PROFILES_DIR.glob("*.json")]
    except OSError:
        return []


# =============================================================================
# Profile Generation from Database
# =============================================================================


def build_profiles_for_all_contacts(
    db: Any,
    embedder: Any,
    imessage_reader: Any | None = None,
    min_messages: int = MIN_MESSAGES_FOR_EMBEDDING_PROFILE,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build embedding profiles for all contacts in the database.

    Args:
        db: JarvisDB instance.
        embedder: Embedding model with encode() method.
        imessage_reader: Optional ChatDBReader for fetching message history.
        min_messages: Minimum messages required for a profile.
        limit: Maximum number of contacts to process.

    Returns:
        Statistics about the profile building.
    """
    contacts = db.list_contacts(limit=limit or 1000)
    logger.info("Building embedding profiles for %d contacts", len(contacts))

    stats = {
        "contacts_processed": 0,
        "profiles_created": 0,
        "profiles_skipped": 0,
        "total_messages_analyzed": 0,
        "errors": [],
    }

    for contact in contacts:
        if not contact.chat_id and not contact.id:
            stats["profiles_skipped"] += 1
            continue

        try:
            messages = []

            # Try iMessage reader first
            if imessage_reader and contact.chat_id:
                try:
                    messages = imessage_reader.get_messages(
                        contact.chat_id,
                        limit=5000,
                    )
                except Exception as e:
                    logger.debug("iMessage failed for %s: %s", contact.display_name, e)

            # Fallback: use pairs from database if iMessage didn't work
            if len(messages) < min_messages and contact.id:
                pairs = db.get_pairs(contact_id=contact.id, limit=5000)
                messages = []
                for p in pairs:
                    if p.trigger_timestamp and p.response_timestamp:
                        messages.append(_MockMessage(p.trigger_text, False, p.trigger_timestamp))
                        messages.append(_MockMessage(p.response_text, True, p.response_timestamp))

            if len(messages) < min_messages:
                logger.debug(
                    "Skipping %s: only %d messages",
                    contact.display_name,
                    len(messages),
                )
                stats["profiles_skipped"] += 1
                continue

            # Build profile
            profile = build_embedding_profile(
                contact_id=str(contact.id),
                messages=messages,
                embedder=embedder,
                contact_name=contact.display_name,
            )

            # Save profile
            if save_embedding_profile(profile):
                stats["profiles_created"] += 1
                stats["total_messages_analyzed"] += profile.message_count
                logger.info(
                    "Created profile for %s (%d messages, %d clusters)",
                    contact.display_name,
                    profile.message_count,
                    len(profile.topic_clusters),
                )
            else:
                stats["errors"].append(f"Failed to save profile for {contact.display_name}")

            stats["contacts_processed"] += 1

        except Exception as e:
            logger.warning("Error processing %s: %s", contact.display_name, e)
            stats["errors"].append(f"{contact.display_name}: {str(e)}")
            stats["contacts_processed"] += 1

    return stats


@dataclass
class _MockMessage:
    """Mock message object for when we only have pairs."""

    text: str
    is_from_me: bool
    date: datetime


# =============================================================================
# Profile Usage for Reply Generation
# =============================================================================


def get_relevant_topic_cluster(
    incoming_message: str,
    profile: EmbeddingProfile,
    embedder: Any,
    threshold: float = 0.5,
) -> TopicCluster | None:
    """Find the most relevant topic cluster for an incoming message.

    Args:
        incoming_message: The message to match.
        profile: The contact's embedding profile.
        embedder: Embedding model.
        threshold: Minimum similarity to consider a match.

    Returns:
        The most similar TopicCluster, or None if below threshold.
    """
    if not profile.topic_clusters:
        return None

    # Embed the incoming message
    msg_embedding = embedder.encode([incoming_message], normalize=True)[0]

    # Find most similar cluster
    best_cluster = None
    best_similarity = threshold

    for cluster in profile.topic_clusters:
        centroid = np.array(cluster.centroid)
        similarity = float(np.dot(msg_embedding, centroid))

        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster

    return best_cluster


def generate_embedding_style_guide(profile: EmbeddingProfile) -> str:
    """Generate style guidance based on embedding analysis.

    Args:
        profile: The embedding profile.

    Returns:
        Natural language description of communication patterns.
    """
    if not profile.dynamics:
        return "Limited embedding data. Using default patterns."

    parts = []
    dynamics = profile.dynamics

    # Style similarity guidance
    if dynamics.style_similarity > 0.8:
        parts.append("You and this person communicate very similarly")
    elif dynamics.style_similarity > 0.6:
        parts.append("Your communication styles are moderately aligned")
    else:
        parts.append("You and this person have different communication styles")

    # Initiation guidance
    if dynamics.initiation_ratio > 0.7:
        parts.append("you typically start conversations")
    elif dynamics.initiation_ratio < 0.3:
        parts.append("they usually initiate")
    else:
        parts.append("conversation initiation is balanced")

    # Topic diversity
    if dynamics.topic_diversity > 2.0:
        parts.append("you discuss a wide variety of topics")
    elif dynamics.topic_diversity < 1.0:
        parts.append("conversations tend to focus on specific topics")

    # Response shift
    if dynamics.response_semantic_shift > 0.4:
        parts.append("your responses often shift the topic")
    elif dynamics.response_semantic_shift < 0.2:
        parts.append("you tend to stay on topic in responses")

    # Topic clusters
    if profile.topic_clusters:
        top_clusters = sorted(
            profile.topic_clusters,
            key=lambda c: c.message_count,
            reverse=True,
        )[:3]

        for cluster in top_clusters:
            if cluster.sample_messages:
                sample = cluster.sample_messages[0][:50]
                parts.append(f'common topic example: "{sample}..."')

    return ". ".join(parts) + "." if parts else "No patterns detected."


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "TopicCluster",
    "CommunicationDynamics",
    "EmbeddingProfile",
    # Profile building
    "build_embedding_profile",
    "build_profiles_for_all_contacts",
    # Storage
    "save_embedding_profile",
    "load_embedding_profile",
    "delete_embedding_profile",
    "list_embedding_profiles",
    # Usage
    "get_relevant_topic_cluster",
    "generate_embedding_style_guide",
    # Constants
    "MIN_MESSAGES_FOR_EMBEDDING_PROFILE",
]
