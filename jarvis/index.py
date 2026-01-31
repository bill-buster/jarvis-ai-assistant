"""FAISS Index Building - Create versioned, searchable vector index of triggers.

Key insight: Match incoming messages to past TRIGGERS (not responses).
Similar questions/requests lead to similar response patterns.

Indexes are versioned for safe rebuilds:
    ~/.jarvis/indexes/triggers/<model_name>/<YYYYMMDD-HHMM>/index.faiss

Usage:
    jarvis db build-index              # Build new FAISS index version
    jarvis db stats                    # Show active index info
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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
