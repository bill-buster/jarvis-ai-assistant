"""FAISS Index Building - Create versioned, searchable vector index of triggers.

Key insight: Match incoming messages to past TRIGGERS (not responses).
Similar questions/requests lead to similar response patterns.

Indexes are versioned for safe rebuilds:
    ~/.jarvis/indexes/triggers/<model_name>/<YYYYMMDD-HHMM>/index.faiss

Supports incremental updates:
    - Add new pairs without full rebuild
    - Mark pairs as deleted (soft delete)
    - Compact index when deletion ratio exceeds threshold

Usage:
    jarvis db build-index              # Build new FAISS index version
    jarvis db stats                    # Show active index info

    # Incremental updates
    from jarvis.index import get_incremental_index
    index = get_incremental_index(jarvis_db)
    index.add_pairs(new_pairs)         # Add new pairs
    index.remove_pairs([pair_id1])     # Mark pairs as deleted
    index.compact()                    # Rebuild if needed
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default paths
JARVIS_DIR = Path.home() / ".jarvis"
INDEXES_DIR = JARVIS_DIR / "indexes" / "triggers"


@dataclass
class IndexConfig:
    """Configuration for FAISS index building."""

    # Base directory for indexes
    indexes_dir: Path = INDEXES_DIR

    # Batch size for encoding
    batch_size: int = 32

    # Whether to normalize embeddings (required for cosine similarity via IP)
    normalize: bool = True


@dataclass
class IndexStats:
    """Statistics from index building."""

    pairs_indexed: int
    dimension: int
    index_size_bytes: int
    embeddings_stored: int
    version_id: str
    index_path: str


class TriggerIndexBuilder:
    """Builds versioned FAISS indexes of trigger embeddings.

    Each build creates a new version in:
    ~/.jarvis/indexes/triggers/<model_name>/<YYYYMMDD-HHMM>/index.faiss

    This allows safe rebuilds and rollback.
    """

    def __init__(self, config: IndexConfig | None = None) -> None:
        """Initialize index builder."""
        self.config = config or IndexConfig()
        self._embedder = None

    @property
    def embedder(self) -> Any:
        """Get the unified embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _get_model_name(self) -> str:
        """Get the model name from the embedder."""
        from jarvis.embedding_adapter import EMBEDDING_MODEL

        return EMBEDDING_MODEL

    def _get_model_dir_name(self) -> str:
        """Get safe directory name for model."""
        # Convert "BAAI/bge-small-en-v1.5" to "bge-small-en-v1.5"
        return self._get_model_name().split("/")[-1]

    def _generate_version_id(self) -> str:
        """Generate a version ID based on current timestamp."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _get_index_path(self, version_id: str) -> Path:
        """Get the path for an index version."""
        model_dir = self._get_model_dir_name()
        return self.config.indexes_dir / model_dir / version_id / "index.faiss"

    def build_index(
        self,
        pairs: list[Any],
        jarvis_db: Any,
        progress_callback: Any | None = None,
    ) -> IndexStats:
        """Build a new versioned FAISS index from trigger texts.

        Args:
            pairs: List of Pair objects from the database.
            jarvis_db: JarvisDB instance for storing embeddings.
            progress_callback: Optional callback(stage, progress, message).

        Returns:
            IndexStats with build information.
        """
        import faiss

        if not pairs:
            raise ValueError("No pairs provided for indexing")

        # Generate version ID
        version_id = self._generate_version_id()
        index_path = self._get_index_path(version_id)

        # Stage 1: Extract triggers
        if progress_callback:
            progress_callback("extracting", 0.0, "Extracting triggers...")

        triggers = [p.trigger_text for p in pairs]
        pair_ids = [p.id for p in pairs]

        logger.info("Building index version %s for %d triggers", version_id, len(triggers))

        # Stage 2: Compute embeddings
        if progress_callback:
            progress_callback("encoding", 0.2, f"Encoding {len(triggers)} triggers...")

        embeddings = self.embedder.encode(triggers, normalize=self.config.normalize)

        embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]

        logger.info("Embeddings shape: %s (dimension=%d)", embeddings.shape, dimension)

        # Stage 3: Create FAISS index
        if progress_callback:
            progress_callback("indexing", 0.6, "Creating FAISS index...")

        # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Stage 4: Save index to disk
        if progress_callback:
            progress_callback("saving", 0.8, "Saving index to disk...")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        index_size = index_path.stat().st_size

        logger.info("Saved FAISS index to %s (%d bytes)", index_path, index_size)

        # Stage 5: Store embeddings mapping in database
        if progress_callback:
            progress_callback("storing", 0.9, "Storing embedding mappings...")

        # Clear embeddings for this version (in case of rebuild)
        jarvis_db.clear_embeddings(index_version=version_id)

        # Add new embeddings (FAISS ID = position in index, pair_id = stable key)
        embedding_dicts = [
            {
                "pair_id": pair_ids[i],
                "faiss_id": i,
                "cluster_id": None,
                "index_version": version_id,
            }
            for i in range(len(pairs))
        ]
        jarvis_db.add_embeddings_bulk(embedding_dicts)

        # Stage 6: Register index version and set as active
        jarvis_db.add_index_version(
            version_id=version_id,
            model_name=self._get_model_name(),
            embedding_dim=dimension,
            num_vectors=len(pairs),
            index_path=str(index_path.relative_to(JARVIS_DIR)),
            is_active=True,
        )

        if progress_callback:
            progress_callback("done", 1.0, f"Indexed {len(pairs)} triggers (version {version_id})")

        return IndexStats(
            pairs_indexed=len(pairs),
            dimension=dimension,
            index_size_bytes=index_size,
            embeddings_stored=len(embedding_dicts),
            version_id=version_id,
            index_path=str(index_path),
        )


class TriggerIndexSearcher:
    """Search the FAISS trigger index for similar messages.

    Automatically uses the active index version from the database.
    """

    def __init__(
        self,
        jarvis_db: Any,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize searcher.

        Args:
            jarvis_db: JarvisDB instance for index metadata.
            embedding_model: Deprecated - model is now determined by unified adapter.
        """
        self.jarvis_db = jarvis_db
        self._index = None
        self._embedder = None
        self._active_version: str | None = None

    @property
    def embedder(self) -> Any:
        """Get the unified embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def index(self) -> Any:
        """Get or load the active FAISS index."""
        import faiss

        # Check if we need to reload (new active version)
        active_index = self.jarvis_db.get_active_index()
        if active_index is None:
            raise FileNotFoundError("No active FAISS index. Run 'jarvis db build-index' first.")

        if self._index is None or self._active_version != active_index.version_id:
            index_path = JARVIS_DIR / active_index.index_path
            if not index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {index_path}. "
                    "Run 'jarvis db build-index' to rebuild."
                )
            self._index = faiss.read_index(str(index_path))
            self._active_version = active_index.version_id
            logger.info("Loaded index version %s", self._active_version)

        return self._index

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        embedder: Any | None = None,
    ) -> list[tuple[int, float]]:
        """Search for similar triggers.

        Args:
            query: The incoming message to match.
            k: Number of results to return.
            threshold: Minimum similarity score (0-1).
            embedder: Optional embedder override (for per-request caching).

        Returns:
            List of (faiss_id, similarity_score) tuples, sorted by score descending.
        """
        # Encode query
        query_embedder = embedder or self.embedder
        query_embedding = query_embedder.encode([query], normalize=True).astype(np.float32)

        # Search index
        scores, indices = self.index.search(query_embedding, k)

        # Filter by threshold and format results
        results = []
        for score, faiss_id in zip(scores[0], indices[0]):
            if faiss_id >= 0 and score >= threshold:
                results.append((int(faiss_id), float(score)))

        return results

    def search_with_pairs(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        prefer_recent: bool = True,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Search and return full pair information with freshness weighting.

        Args:
            query: The incoming message to match.
            k: Number of results to return.
            threshold: Minimum similarity score.
            prefer_recent: Weight results by recency (source_timestamp).
            embedder: Optional embedder override (for per-request caching).

        Returns:
            List of dicts with pair info and similarity score.
        """
        # Get active index version
        active_index = self.jarvis_db.get_active_index()
        if not active_index:
            return []

        matches = self.search(
            query,
            k * 2 if prefer_recent else k,
            threshold,
            embedder=embedder,
        )

        results = []
        for faiss_id, score in matches:
            pair = self.jarvis_db.get_pair_by_faiss_id(faiss_id, active_index.version_id)
            if pair:
                # Apply freshness weighting if requested
                final_score = score
                if prefer_recent and pair.source_timestamp:
                    # Decay factor: lose 10% per year of age
                    age_days = (datetime.now() - pair.source_timestamp).days
                    decay = max(0.5, 1.0 - (age_days / 365) * 0.1)
                    final_score = score * decay

                embedding = self.jarvis_db.get_embedding_by_pair(pair.id)
                cluster = None
                if embedding and embedding.cluster_id:
                    cluster = self.jarvis_db.get_cluster(embedding.cluster_id)

                results.append(
                    {
                        "similarity": round(score, 3),
                        "weighted_score": round(final_score, 3),
                        "trigger_text": pair.trigger_text,
                        "response_text": pair.response_text,
                        "chat_id": pair.chat_id,
                        "faiss_id": faiss_id,
                        "pair_id": pair.id,
                        "source_timestamp": pair.source_timestamp,
                        "quality_score": pair.quality_score,
                        "cluster_id": embedding.cluster_id if embedding else None,
                        "cluster_name": cluster.name if cluster else None,
                    }
                )

        # Sort by weighted score and limit
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results[:k]


@dataclass
class IncrementalIndexConfig:
    """Configuration for incremental FAISS index updates.

    Attributes:
        indexes_dir: Base directory for indexes.
        compact_threshold: Rebuild index when deleted ratio exceeds this (0.0-1.0).
        auto_save: Automatically save after modifications.
        normalize: Normalize embeddings for cosine similarity.
    """

    indexes_dir: Path = field(default_factory=lambda: INDEXES_DIR)
    compact_threshold: float = 0.2  # Rebuild when 20% deleted
    auto_save: bool = True
    normalize: bool = True


@dataclass
class IncrementalIndexStats:
    """Statistics for an incremental index.

    Attributes:
        total_vectors: Total vectors in FAISS index (including deleted).
        active_vectors: Vectors not marked as deleted.
        deleted_vectors: Number of soft-deleted vectors.
        deletion_ratio: Ratio of deleted to total vectors.
        needs_compact: Whether index should be compacted.
        version_id: Current index version.
        last_modified: Timestamp of last modification.
    """

    total_vectors: int
    active_vectors: int
    deleted_vectors: int
    deletion_ratio: float
    needs_compact: bool
    version_id: str
    last_modified: datetime | None


class IncrementalTriggerIndex:
    """Incremental FAISS index that supports add/remove without full rebuilds.

    Maintains a soft-delete set to mark removed vectors. During search,
    deleted vectors are filtered out. When the deletion ratio exceeds
    the threshold, call compact() to rebuild the index.

    Thread-safe for concurrent access.

    Example:
        >>> from jarvis.index import get_incremental_index
        >>> from jarvis.db import get_db
        >>> db = get_db()
        >>> index = get_incremental_index(db)
        >>>
        >>> # Add new pairs
        >>> new_pairs = db.get_pairs_since(last_indexed_at)
        >>> index.add_pairs(new_pairs)
        >>>
        >>> # Remove deleted pairs
        >>> index.remove_pairs([123, 456])
        >>>
        >>> # Search (automatically skips deleted)
        >>> results = index.search("hello", k=5)
        >>>
        >>> # Compact when needed
        >>> if index.needs_compact():
        ...     index.compact()
    """

    def __init__(
        self,
        jarvis_db: Any,
        config: IncrementalIndexConfig | None = None,
    ) -> None:
        """Initialize incremental index.

        Args:
            jarvis_db: JarvisDB instance for metadata storage.
            config: Index configuration.
        """
        self.jarvis_db = jarvis_db
        self.config = config or IncrementalIndexConfig()
        self._lock = threading.RLock()
        self._index: Any = None
        self._embedder: Any = None
        self._version_id: str | None = None
        self._deleted_faiss_ids: set[int] = set()
        self._pair_to_faiss: dict[int, int] = {}  # pair_id -> faiss_id
        self._faiss_to_pair: dict[int, int] = {}  # faiss_id -> pair_id
        self._last_modified: datetime | None = None
        self._metadata_path: Path | None = None
        self._loaded = False

    @property
    def embedder(self) -> Any:
        """Get the unified embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _get_model_dir_name(self) -> str:
        """Get safe directory name for model."""
        from jarvis.embedding_adapter import EMBEDDING_MODEL

        return EMBEDDING_MODEL.split("/")[-1]

    def _get_index_dir(self) -> Path:
        """Get the directory for incremental index."""
        model_dir = self._get_model_dir_name()
        return self.config.indexes_dir / model_dir / "incremental"

    def _get_metadata_path(self) -> Path:
        """Get path to metadata JSON file."""
        return self._get_index_dir() / "metadata.json"

    def _get_index_path(self) -> Path:
        """Get path to FAISS index file."""
        return self._get_index_dir() / "index.faiss"

    def _ensure_loaded(self) -> None:
        """Ensure index is loaded, loading from disk if needed."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            self._load_or_initialize()
            self._loaded = True

    def _load_or_initialize(self) -> None:
        """Load existing index or initialize empty one."""
        import faiss

        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()

        if index_path.exists() and metadata_path.exists():
            # Load existing index
            self._index = faiss.read_index(str(index_path))
            self._load_metadata()
            logger.info(
                "Loaded incremental index: %d vectors (%d deleted)",
                self._index.ntotal,
                len(self._deleted_faiss_ids),
            )
        else:
            # Initialize empty index
            # We'll create the actual index on first add
            self._index = None
            self._version_id = self._generate_version_id()
            self._deleted_faiss_ids = set()
            self._pair_to_faiss = {}
            self._faiss_to_pair = {}
            self._last_modified = datetime.now()
            logger.info("Initialized new incremental index (version %s)", self._version_id)

    def _generate_version_id(self) -> str:
        """Generate version ID from timestamp."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        metadata_path = self._get_metadata_path()
        try:
            with open(metadata_path) as f:
                data = json.load(f)

            self._version_id = data.get("version_id")
            self._deleted_faiss_ids = set(data.get("deleted_faiss_ids", []))
            self._pair_to_faiss = {int(k): v for k, v in data.get("pair_to_faiss", {}).items()}
            self._faiss_to_pair = {int(k): v for k, v in data.get("faiss_to_pair", {}).items()}
            last_mod = data.get("last_modified")
            self._last_modified = datetime.fromisoformat(last_mod) if last_mod else None

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load metadata, reinitializing: %s", e)
            self._version_id = self._generate_version_id()
            self._deleted_faiss_ids = set()
            self._pair_to_faiss = {}
            self._faiss_to_pair = {}
            self._last_modified = datetime.now()

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata_path = self._get_metadata_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version_id": self._version_id,
            "deleted_faiss_ids": list(self._deleted_faiss_ids),
            "pair_to_faiss": self._pair_to_faiss,
            "faiss_to_pair": self._faiss_to_pair,
            "last_modified": self._last_modified.isoformat() if self._last_modified else None,
        }

        with open(metadata_path, "w") as f:
            json.dump(data, f)

    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        import faiss

        if self._index is None:
            return

        index_path = self._get_index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

    def save(self) -> None:
        """Save index and metadata to disk."""
        with self._lock:
            self._save_index()
            self._save_metadata()
            logger.debug("Saved incremental index")

    def add_pairs(
        self,
        pairs: list[Any],
        progress_callback: Any | None = None,
    ) -> int:
        """Add new pairs to the index incrementally.

        Only adds pairs that aren't already indexed.

        Args:
            pairs: List of Pair objects to add.
            progress_callback: Optional callback(stage, progress, message).

        Returns:
            Number of pairs actually added.
        """
        import faiss

        self._ensure_loaded()

        with self._lock:
            # Filter out pairs already in index
            new_pairs = [p for p in pairs if p.id not in self._pair_to_faiss]
            if not new_pairs:
                logger.debug("No new pairs to add")
                return 0

            if progress_callback:
                progress_callback("encoding", 0.2, f"Encoding {len(new_pairs)} triggers...")

            # Encode triggers
            triggers = [p.trigger_text for p in new_pairs]
            embeddings = self.embedder.encode(triggers, normalize=self.config.normalize)
            embeddings = embeddings.astype(np.float32)
            dimension = embeddings.shape[1]

            # Initialize index if needed
            if self._index is None:
                self._index = faiss.IndexFlatIP(dimension)
                logger.info("Created new FAISS index (dim=%d)", dimension)

            # Get starting faiss_id
            start_faiss_id = self._index.ntotal

            # Add to FAISS index
            if progress_callback:
                progress_callback("indexing", 0.6, "Adding to FAISS index...")

            self._index.add(embeddings)

            # Update mappings
            for i, pair in enumerate(new_pairs):
                faiss_id = start_faiss_id + i
                self._pair_to_faiss[pair.id] = faiss_id
                self._faiss_to_pair[faiss_id] = pair.id

            self._last_modified = datetime.now()

            # Update database embeddings
            if progress_callback:
                progress_callback("storing", 0.8, "Storing embedding mappings...")

            embedding_dicts = [
                {
                    "pair_id": new_pairs[i].id,
                    "faiss_id": start_faiss_id + i,
                    "cluster_id": None,
                    "index_version": self._version_id,
                }
                for i in range(len(new_pairs))
            ]
            self.jarvis_db.add_embeddings_bulk(embedding_dicts)

            # Auto-save if enabled
            if self.config.auto_save:
                self.save()

            if progress_callback:
                progress_callback("done", 1.0, f"Added {len(new_pairs)} pairs")

            logger.info("Added %d pairs to incremental index", len(new_pairs))
            return len(new_pairs)

    def remove_pairs(self, pair_ids: list[int]) -> int:
        """Mark pairs as deleted (soft delete).

        Deleted pairs are skipped during search. Call compact() to
        physically remove them and reclaim space.

        Args:
            pair_ids: List of pair IDs to remove.

        Returns:
            Number of pairs actually removed.
        """
        self._ensure_loaded()

        with self._lock:
            removed = 0
            for pair_id in pair_ids:
                if pair_id in self._pair_to_faiss:
                    faiss_id = self._pair_to_faiss[pair_id]
                    self._deleted_faiss_ids.add(faiss_id)
                    removed += 1

            if removed > 0:
                self._last_modified = datetime.now()

                # Auto-save if enabled
                if self.config.auto_save:
                    self._save_metadata()

                logger.info("Marked %d pairs as deleted", removed)

            return removed

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        embedder: Any | None = None,
    ) -> list[tuple[int, float]]:
        """Search for similar triggers, skipping deleted vectors.

        Args:
            query: The query text to search for.
            k: Number of results to return.
            threshold: Minimum similarity score.
            embedder: Optional embedder override (for caching).

        Returns:
            List of (faiss_id, similarity_score) tuples.
        """
        self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            # Encode query
            query_embedder = embedder or self.embedder
            query_embedding = query_embedder.encode([query], normalize=True).astype(np.float32)

            # Search for more results to account for deleted vectors
            deleted_count = len(self._deleted_faiss_ids)
            search_k = min(k + deleted_count + 5, self._index.ntotal)

            scores, indices = self._index.search(query_embedding, search_k)

            # Filter deleted and below threshold
            results = []
            for score, faiss_id in zip(scores[0], indices[0]):
                if faiss_id < 0:
                    continue
                if faiss_id in self._deleted_faiss_ids:
                    continue
                if score < threshold:
                    continue
                results.append((int(faiss_id), float(score)))
                if len(results) >= k:
                    break

            return results

    def search_with_pairs(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        prefer_recent: bool = True,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Search and return full pair information.

        Args:
            query: The query text to search for.
            k: Number of results to return.
            threshold: Minimum similarity score.
            prefer_recent: Weight results by recency.
            embedder: Optional embedder override.

        Returns:
            List of dicts with pair info and similarity score.
        """
        self._ensure_loaded()

        matches = self.search(
            query,
            k * 2 if prefer_recent else k,
            threshold,
            embedder=embedder,
        )

        results = []
        for faiss_id, score in matches:
            pair_id = self._faiss_to_pair.get(faiss_id)
            if pair_id is None:
                continue

            pair = self.jarvis_db.get_pair(pair_id)
            if pair is None:
                continue

            # Apply freshness weighting
            final_score = score
            if prefer_recent and pair.source_timestamp:
                age_days = (datetime.now() - pair.source_timestamp).days
                decay = max(0.5, 1.0 - (age_days / 365) * 0.1)
                final_score = score * decay

            results.append(
                {
                    "similarity": round(score, 3),
                    "weighted_score": round(final_score, 3),
                    "trigger_text": pair.trigger_text,
                    "response_text": pair.response_text,
                    "chat_id": pair.chat_id,
                    "faiss_id": faiss_id,
                    "pair_id": pair.id,
                    "source_timestamp": pair.source_timestamp,
                    "quality_score": pair.quality_score,
                }
            )

        # Sort by weighted score and limit
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results[:k]

    def needs_compact(self) -> bool:
        """Check if index should be compacted.

        Returns:
            True if deletion ratio exceeds threshold.
        """
        self._ensure_loaded()

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return False

            deletion_ratio = len(self._deleted_faiss_ids) / self._index.ntotal
            return deletion_ratio >= self.config.compact_threshold

    def get_stats(self) -> IncrementalIndexStats:
        """Get statistics about the incremental index.

        Returns:
            IncrementalIndexStats with current state.
        """
        self._ensure_loaded()

        with self._lock:
            total = self._index.ntotal if self._index else 0
            deleted = len(self._deleted_faiss_ids)
            active = total - deleted
            ratio = deleted / total if total > 0 else 0.0

            return IncrementalIndexStats(
                total_vectors=total,
                active_vectors=active,
                deleted_vectors=deleted,
                deletion_ratio=ratio,
                needs_compact=ratio >= self.config.compact_threshold,
                version_id=self._version_id or "none",
                last_modified=self._last_modified,
            )

    def compact(
        self,
        progress_callback: Any | None = None,
        min_quality: float = 0.5,
    ) -> IncrementalIndexStats:
        """Rebuild index to remove deleted vectors and reclaim space.

        Creates a new index with only active pairs.

        Args:
            progress_callback: Optional callback(stage, progress, message).
            min_quality: Minimum quality score for pairs to include.

        Returns:
            IncrementalIndexStats after compaction.
        """
        import faiss

        self._ensure_loaded()

        with self._lock:
            if progress_callback:
                progress_callback("collecting", 0.1, "Collecting active pairs...")

            # Get active pair IDs (not deleted)
            active_pair_ids = [
                pair_id
                for pair_id, faiss_id in self._pair_to_faiss.items()
                if faiss_id not in self._deleted_faiss_ids
            ]

            # Fetch pairs from database
            active_pairs = []
            for pair_id in active_pair_ids:
                pair = self.jarvis_db.get_pair(pair_id)
                if pair and (pair.quality_score or 0) >= min_quality:
                    active_pairs.append(pair)

            if not active_pairs:
                logger.warning("No active pairs to compact")
                return self.get_stats()

            if progress_callback:
                progress_callback("encoding", 0.3, f"Re-encoding {len(active_pairs)} triggers...")

            # Re-encode all active triggers
            triggers = [p.trigger_text for p in active_pairs]
            embeddings = self.embedder.encode(triggers, normalize=self.config.normalize)
            embeddings = embeddings.astype(np.float32)
            dimension = embeddings.shape[1]

            if progress_callback:
                progress_callback("rebuilding", 0.6, "Rebuilding FAISS index...")

            # Create new index
            new_index = faiss.IndexFlatIP(dimension)
            new_index.add(embeddings)

            # Update version and mappings
            self._version_id = self._generate_version_id()
            self._index = new_index
            self._deleted_faiss_ids = set()
            self._pair_to_faiss = {pair.id: i for i, pair in enumerate(active_pairs)}
            self._faiss_to_pair = {i: pair.id for i, pair in enumerate(active_pairs)}
            self._last_modified = datetime.now()

            if progress_callback:
                progress_callback("storing", 0.8, "Updating database...")

            # Update database embeddings
            self.jarvis_db.clear_embeddings(index_version=None)  # Clear all
            embedding_dicts = [
                {
                    "pair_id": active_pairs[i].id,
                    "faiss_id": i,
                    "cluster_id": None,
                    "index_version": self._version_id,
                }
                for i in range(len(active_pairs))
            ]
            self.jarvis_db.add_embeddings_bulk(embedding_dicts)

            # Save to disk
            self.save()

            if progress_callback:
                progress_callback("done", 1.0, f"Compacted to {len(active_pairs)} pairs")

            logger.info(
                "Compacted index: %d pairs (version %s)",
                len(active_pairs),
                self._version_id,
            )

            return self.get_stats()

    def sync_with_db(
        self,
        min_quality: float = 0.5,
        include_holdout: bool = False,
    ) -> tuple[int, int]:
        """Sync index with database, adding new pairs and removing deleted ones.

        Args:
            min_quality: Minimum quality score for pairs.
            include_holdout: Include holdout pairs.

        Returns:
            Tuple of (added_count, removed_count).
        """
        self._ensure_loaded()

        with self._lock:
            # Get all qualifying pairs from database
            if include_holdout:
                db_pairs = self.jarvis_db.get_all_pairs(min_quality=min_quality)
            else:
                db_pairs = self.jarvis_db.get_training_pairs(min_quality=min_quality)

            db_pair_ids = {p.id for p in db_pairs}
            indexed_pair_ids = set(self._pair_to_faiss.keys())

            # Find pairs to add (in DB but not indexed)
            to_add_ids = db_pair_ids - indexed_pair_ids
            pairs_to_add = [p for p in db_pairs if p.id in to_add_ids]

            # Find pairs to remove (indexed but not in DB or below quality)
            to_remove_ids = indexed_pair_ids - db_pair_ids

            # Add new pairs
            added = self.add_pairs(pairs_to_add) if pairs_to_add else 0

            # Remove deleted pairs
            removed = self.remove_pairs(list(to_remove_ids)) if to_remove_ids else 0

            logger.info("Synced index: added %d, removed %d", added, removed)
            return added, removed


# Singleton incremental index
_incremental_index: IncrementalTriggerIndex | None = None
_incremental_index_lock = threading.Lock()


def get_incremental_index(jarvis_db: Any) -> IncrementalTriggerIndex:
    """Get or create singleton incremental index.

    Args:
        jarvis_db: JarvisDB instance.

    Returns:
        The shared IncrementalTriggerIndex instance.
    """
    global _incremental_index
    if _incremental_index is None:
        with _incremental_index_lock:
            if _incremental_index is None:
                _incremental_index = IncrementalTriggerIndex(jarvis_db)
    return _incremental_index


def reset_incremental_index() -> None:
    """Reset the singleton incremental index.

    Use for testing or to force reinitialization.
    """
    global _incremental_index
    with _incremental_index_lock:
        _incremental_index = None


def build_index_from_db(
    jarvis_db: Any,
    config: IndexConfig | None = None,
    progress_callback: Any | None = None,
    min_quality: float = 0.5,
    include_holdout: bool = False,
) -> dict[str, Any]:
    """Build a new versioned FAISS index from training pairs in the database.

    Args:
        jarvis_db: JarvisDB instance.
        config: Index configuration.
        progress_callback: Optional progress callback.
        min_quality: Minimum quality score for pairs to include.
        include_holdout: If False (default), exclude holdout pairs from index.
            Set to True only for full index rebuild (not recommended for eval).

    Returns:
        Statistics about the index building.
    """
    # Get pairs meeting quality threshold, excluding holdout by default
    if include_holdout:
        pairs = jarvis_db.get_all_pairs(min_quality=min_quality)
    else:
        pairs = jarvis_db.get_training_pairs(min_quality=min_quality)

    if not pairs:
        return {
            "success": False,
            "error": "No pairs found in database. Run 'jarvis db extract' first.",
            "pairs_indexed": 0,
        }

    # Build index
    builder = TriggerIndexBuilder(config)
    stats = builder.build_index(pairs, jarvis_db, progress_callback)

    return {
        "success": True,
        "pairs_indexed": stats.pairs_indexed,
        "dimension": stats.dimension,
        "index_size_bytes": stats.index_size_bytes,
        "version_id": stats.version_id,
        "index_path": stats.index_path,
        "model_name": builder._get_model_name(),
    }


def get_index_stats(jarvis_db: Any = None) -> dict[str, Any] | None:
    """Get statistics about the active FAISS index.

    Args:
        jarvis_db: Optional JarvisDB instance. If None, only checks file existence.

    Returns:
        Statistics dict or None if no active index.
    """
    if jarvis_db is None:
        # Legacy path check
        legacy_path = JARVIS_DIR / "triggers.index"
        if legacy_path.exists():
            import faiss

            index = faiss.read_index(str(legacy_path))
            return {
                "exists": True,
                "path": str(legacy_path),
                "size_bytes": legacy_path.stat().st_size,
                "num_vectors": index.ntotal,
                "dimension": index.d,
                "is_trained": index.is_trained,
                "version_id": "legacy",
            }
        return None

    # Get active index from database
    active_index = jarvis_db.get_active_index()
    if not active_index:
        return None

    index_path = JARVIS_DIR / active_index.index_path
    if not index_path.exists():
        return {
            "exists": False,
            "version_id": active_index.version_id,
            "error": f"Index file missing: {index_path}",
        }

    return {
        "exists": True,
        "path": str(index_path),
        "size_bytes": index_path.stat().st_size,
        "num_vectors": active_index.num_vectors,
        "dimension": active_index.embedding_dim,
        "model_name": active_index.model_name,
        "version_id": active_index.version_id,
        "is_active": active_index.is_active,
        "created_at": active_index.created_at,
    }


def list_index_versions(jarvis_db: Any) -> list[dict[str, Any]]:
    """List all available index versions.

    Args:
        jarvis_db: JarvisDB instance.

    Returns:
        List of version info dicts.
    """
    versions = jarvis_db.list_index_versions()
    return [
        {
            "version_id": v.version_id,
            "model_name": v.model_name,
            "num_vectors": v.num_vectors,
            "dimension": v.embedding_dim,
            "is_active": v.is_active,
            "created_at": v.created_at,
            "index_path": v.index_path,
        }
        for v in versions
    ]
