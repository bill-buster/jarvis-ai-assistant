"""Tests for FAISS trigger index (jarvis/index.py).

Tests cover:
- TriggerIndexBuilder: index building, versioning, empty pairs handling
- TriggerIndexSearcher: search, search_with_pairs, similarity thresholds
- Index persistence (save/load)
- Edge cases (empty index, no matches, duplicate triggers)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import sys
import numpy as np
import pytest

from jarvis.db import Cluster, IndexVersion, Pair, PairEmbedding
from jarvis.index import (
    IndexConfig,
    IndexStats,
    TriggerIndexBuilder,
    TriggerIndexSearcher,
    build_index_from_db,
    get_index_stats,
    list_index_versions,
)

# Create a mock faiss module for testing
mock_faiss_module = MagicMock()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_jarvis_dir(tmp_path: Path) -> Path:
    """Create a temporary JARVIS directory structure."""
    jarvis_dir = tmp_path / ".jarvis"
    jarvis_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir = jarvis_dir / "indexes" / "triggers"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    return jarvis_dir


@pytest.fixture
def index_config(temp_jarvis_dir: Path) -> IndexConfig:
    """Create an IndexConfig with temporary directory."""
    return IndexConfig(indexes_dir=temp_jarvis_dir / "indexes" / "triggers")


@pytest.fixture
def mock_embedding() -> np.ndarray:
    """Create a mock normalized embedding vector."""
    embedding = np.random.randn(384).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


@pytest.fixture
def sample_pairs() -> list[Pair]:
    """Create sample pairs for testing."""
    now = datetime.now()
    return [
        Pair(
            id=1,
            contact_id=1,
            trigger_text="Hey, want to grab dinner tonight?",
            response_text="Sure, sounds great! What time?",
            trigger_timestamp=now - timedelta(hours=2),
            response_timestamp=now - timedelta(hours=1, minutes=55),
            chat_id="chat123",
            quality_score=0.9,
            source_timestamp=now - timedelta(hours=2),
        ),
        Pair(
            id=2,
            contact_id=1,
            trigger_text="Can you help me with this project?",
            response_text="Of course! Let me take a look.",
            trigger_timestamp=now - timedelta(hours=5),
            response_timestamp=now - timedelta(hours=4, minutes=50),
            chat_id="chat123",
            quality_score=0.85,
            source_timestamp=now - timedelta(hours=5),
        ),
        Pair(
            id=3,
            contact_id=2,
            trigger_text="Are you free this weekend?",
            response_text="Yeah, I'm available Saturday!",
            trigger_timestamp=now - timedelta(days=1),
            response_timestamp=now - timedelta(days=1) + timedelta(minutes=5),
            chat_id="chat456",
            quality_score=0.8,
            source_timestamp=now - timedelta(days=1),
        ),
    ]


@pytest.fixture
def mock_jarvis_db() -> MagicMock:
    """Create a mock JarvisDB instance."""
    db = MagicMock()
    db.clear_embeddings = MagicMock()
    db.add_embeddings_bulk = MagicMock()
    db.add_index_version = MagicMock()
    db.get_active_index = MagicMock(return_value=None)
    db.get_pair_by_faiss_id = MagicMock(return_value=None)
    db.get_embedding_by_pair = MagicMock(return_value=None)
    db.get_cluster = MagicMock(return_value=None)
    return db


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder."""
    embedder = MagicMock()

    def encode_side_effect(texts: list[str], normalize: bool = True) -> np.ndarray:
        """Generate deterministic embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384).astype(np.float32)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    embedder.encode = MagicMock(side_effect=encode_side_effect)
    return embedder


# =============================================================================
# IndexConfig Tests
# =============================================================================


class TestIndexConfig:
    """Tests for IndexConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = IndexConfig()
        assert config.batch_size == 32
        assert config.normalize is True
        assert "indexes" in str(config.indexes_dir)

    def test_custom_values(self, temp_jarvis_dir: Path) -> None:
        """Test custom configuration values."""
        config = IndexConfig(
            indexes_dir=temp_jarvis_dir / "indexes" / "triggers",
            batch_size=64,
            normalize=False,
        )
        assert config.indexes_dir == temp_jarvis_dir / "indexes" / "triggers"
        assert config.batch_size == 64
        assert config.normalize is False


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_creation(self) -> None:
        """Test creating IndexStats."""
        stats = IndexStats(
            pairs_indexed=100,
            dimension=384,
            index_size_bytes=50000,
            embeddings_stored=100,
            version_id="20240115-143022",
            index_path="/path/to/index.faiss",
        )
        assert stats.pairs_indexed == 100
        assert stats.dimension == 384
        assert stats.index_size_bytes == 50000
        assert stats.version_id == "20240115-143022"


# =============================================================================
# TriggerIndexBuilder Tests
# =============================================================================


class TestTriggerIndexBuilder:
    """Tests for TriggerIndexBuilder class."""

    def test_initialization(self, index_config: IndexConfig) -> None:
        """Test builder initialization."""
        builder = TriggerIndexBuilder(config=index_config)
        assert builder.config == index_config

    def test_initialization_default_config(self) -> None:
        """Test builder with default config."""
        builder = TriggerIndexBuilder()
        assert builder.config is not None
        assert builder.config.batch_size == 32

    def test_generate_version_id(self, index_config: IndexConfig) -> None:
        """Test version ID generation."""
        builder = TriggerIndexBuilder(config=index_config)
        version_id = builder._generate_version_id()

        # Should be in format YYYYMMDD-HHMMSS
        assert len(version_id) == 15
        assert version_id[8] == "-"

    def test_get_index_path(self, index_config: IndexConfig) -> None:
        """Test index path generation."""
        builder = TriggerIndexBuilder(config=index_config)
        path = builder._get_index_path("20240115-143022")

        assert "20240115-143022" in str(path)
        assert str(path).endswith("index.faiss")

    def test_build_index_success(
        self,
        index_config: IndexConfig,
        sample_pairs: list[Pair],
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test successful index building."""
        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            builder = TriggerIndexBuilder(config=index_config)
            builder._embedder = mock_embedder

            stats = builder.build_index(sample_pairs, mock_jarvis_db)

        assert stats.pairs_indexed == 3
        assert stats.dimension == 384
        assert stats.embeddings_stored == 3
        assert stats.index_size_bytes > 0

        # Verify database calls
        mock_jarvis_db.clear_embeddings.assert_called_once()
        mock_jarvis_db.add_embeddings_bulk.assert_called_once()
        mock_jarvis_db.add_index_version.assert_called_once()

        # Verify index file was created
        index_path = Path(stats.index_path)
        assert index_path.exists()

    def test_build_index_empty_pairs_raises(
        self,
        index_config: IndexConfig,
        mock_jarvis_db: MagicMock,
    ) -> None:
        """Test that empty pairs list raises ValueError."""
        builder = TriggerIndexBuilder(config=index_config)

        with pytest.raises(ValueError, match="No pairs provided"):
            builder.build_index([], mock_jarvis_db)

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_build_index_with_progress_callback(
        self,
        mock_get_embedder: MagicMock,
        index_config: IndexConfig,
        sample_pairs: list[Pair],
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test index building with progress callback."""
        mock_get_embedder.return_value = mock_embedder

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            builder = TriggerIndexBuilder(config=index_config)
            builder._embedder = mock_embedder

            progress_updates: list[tuple[str, float, str]] = []

            def progress_callback(stage: str, progress: float, message: str) -> None:
                progress_updates.append((stage, progress, message))

            stats = builder.build_index(sample_pairs, mock_jarvis_db, progress_callback)

        assert len(progress_updates) > 0
        # Should have stages: extracting, encoding, indexing, saving, storing, done
        stages = [update[0] for update in progress_updates]
        assert "extracting" in stages
        assert "encoding" in stages
        assert "indexing" in stages
        assert "saving" in stages
        assert "storing" in stages
        assert "done" in stages

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_build_index_versioning(
        self,
        mock_get_embedder: MagicMock,
        index_config: IndexConfig,
        sample_pairs: list[Pair],
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that each build creates a new version."""
        mock_get_embedder.return_value = mock_embedder

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            builder = TriggerIndexBuilder(config=index_config)
            builder._embedder = mock_embedder

            # Build first index
            stats1 = builder.build_index(sample_pairs, mock_jarvis_db)

            # Reset mocks
            mock_jarvis_db.reset_mock()

            # Build second index (should have different version)
            import time

            time.sleep(1.1)  # Ensure different timestamp
            stats2 = builder.build_index(sample_pairs, mock_jarvis_db)

        assert stats1.version_id != stats2.version_id
        assert stats1.index_path != stats2.index_path


# =============================================================================
# TriggerIndexSearcher Tests
# =============================================================================


class TestTriggerIndexSearcher:
    """Tests for TriggerIndexSearcher class."""

    def test_initialization(self, mock_jarvis_db: MagicMock) -> None:
        """Test searcher initialization."""
        searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
        assert searcher.jarvis_db == mock_jarvis_db
        assert searcher._index is None
        assert searcher._active_version is None

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_no_active_index(
        self,
        mock_get_embedder: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        """Test search when no active index exists."""
        mock_get_embedder.return_value = mock_embedder
        mock_jarvis_db.get_active_index.return_value = None
        searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)

        with pytest.raises(FileNotFoundError, match="No active FAISS index"):
            searcher.search("test query")

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_returns_results(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that search returns matching results."""
        mock_get_embedder.return_value = mock_embedder

        # Setup mock active index
        mock_active_index = IndexVersion(
            id=1,
            version_id="20240115-143022",
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
            num_vectors=5,
            index_path="indexes/triggers/bge-small-en-v1.5/20240115-143022/index.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        # Create the index file path
        index_file = (
            temp_jarvis_dir / "indexes" / "triggers" / "bge-small-en-v1.5" / "20240115-143022"
        )
        index_file.mkdir(parents=True, exist_ok=True)
        (index_file / "index.faiss").write_bytes(b"fake")

        # Setup mock embedder
        query_embedding = np.random.randn(1, 384).astype(np.float32)
        mock_embedder.encode.return_value = query_embedding

        # Setup mock FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.3]]),  # scores
            np.array([[0, 1, 2, 3, 4]]),  # indices
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            results = searcher.search("test query", k=5, threshold=0.5)

        assert len(results) == 4  # Only scores >= 0.5
        assert all(score >= 0.5 for _, score in results)

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_respects_threshold(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that search respects similarity threshold."""
        mock_get_embedder.return_value = mock_embedder

        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=5,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        # Create the index file
        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95, 0.85, 0.75, 0.65, 0.45]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder

            # High threshold
            results_high = searcher.search("test", k=5, threshold=0.8)
            assert len(results_high) == 2  # Only 0.95 and 0.85

            # Low threshold
            mock_index.search.return_value = (
                np.array([[0.95, 0.85, 0.75, 0.65, 0.45]]),
                np.array([[0, 1, 2, 3, 4]]),
            )
            results_low = searcher.search("test", k=5, threshold=0.5)
            assert len(results_low) == 4  # All except 0.45

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_handles_invalid_indices(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that search handles invalid (negative) indices from FAISS."""
        mock_get_embedder.return_value = mock_embedder

        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=3,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        mock_index = MagicMock()
        # FAISS returns -1 for positions that don't have valid results
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.0, 0.0, 0.0]]),
            np.array([[0, 1, -1, -1, -1]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            results = searcher.search("test", k=5, threshold=0.5)

        assert len(results) == 2  # Only valid indices (0, 1)

    def test_search_with_pairs_no_active_index(self, mock_jarvis_db: MagicMock) -> None:
        """Test search_with_pairs when no active index exists."""
        mock_jarvis_db.get_active_index.return_value = None
        searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)

        results = searcher.search_with_pairs("test query")
        assert results == []

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_with_pairs_returns_full_info(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        sample_pairs: list[Pair],
        temp_jarvis_dir: Path,
    ) -> None:
        """Test search_with_pairs returns full pair information."""
        mock_get_embedder.return_value = mock_embedder

        mock_active_index = IndexVersion(
            id=1,
            version_id="test-version",
            model_name="test",
            embedding_dim=384,
            num_vectors=3,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index
        mock_jarvis_db.get_pair_by_faiss_id.side_effect = (
            lambda fid, _: sample_pairs[fid] if fid < len(sample_pairs) else None
        )
        mock_jarvis_db.get_embedding_by_pair.return_value = PairEmbedding(
            pair_id=1, faiss_id=0, cluster_id=1, index_version="test-version"
        )
        mock_jarvis_db.get_cluster.return_value = Cluster(
            id=1, name="DINNER", description="Dinner plans"
        )

        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.85]]),
            np.array([[0, 1]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            results = searcher.search_with_pairs("dinner tonight", k=2, threshold=0.5)

        assert len(results) == 2
        assert "similarity" in results[0]
        assert "weighted_score" in results[0]
        assert "trigger_text" in results[0]
        assert "response_text" in results[0]
        assert "cluster_name" in results[0]

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_with_pairs_freshness_weighting(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that freshness weighting affects scores."""
        mock_get_embedder.return_value = mock_embedder
        now = datetime.now()

        # Create pairs with different ages
        old_pair = Pair(
            id=1,
            contact_id=1,
            trigger_text="Old trigger",
            response_text="Old response",
            trigger_timestamp=now - timedelta(days=365 * 2),  # 2 years old
            response_timestamp=now - timedelta(days=365 * 2),
            chat_id="chat123",
            source_timestamp=now - timedelta(days=365 * 2),
        )
        new_pair = Pair(
            id=2,
            contact_id=1,
            trigger_text="New trigger",
            response_text="New response",
            trigger_timestamp=now - timedelta(days=1),  # 1 day old
            response_timestamp=now - timedelta(days=1),
            chat_id="chat123",
            source_timestamp=now - timedelta(days=1),
        )

        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=2,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index
        mock_jarvis_db.get_pair_by_faiss_id.side_effect = (
            lambda fid, _: [old_pair, new_pair][fid] if fid < 2 else None
        )
        mock_jarvis_db.get_embedding_by_pair.return_value = None

        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        mock_index = MagicMock()
        # Both have same raw similarity
        mock_index.search.return_value = (
            np.array([[0.9, 0.9]]),
            np.array([[0, 1]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            results = searcher.search_with_pairs("test", k=2, threshold=0.5, prefer_recent=True)

        # New pair should have higher weighted score
        assert len(results) == 2
        new_result = next(r for r in results if "New" in r["trigger_text"])
        old_result = next(r for r in results if "Old" in r["trigger_text"])
        assert new_result["weighted_score"] > old_result["weighted_score"]


# =============================================================================
# build_index_from_db Tests
# =============================================================================


class TestBuildIndexFromDb:
    """Tests for build_index_from_db function."""

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_build_index_from_db_success(
        self,
        mock_get_embedder: MagicMock,
        mock_jarvis_db: MagicMock,
        sample_pairs: list[Pair],
        index_config: IndexConfig,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test successful index building from database."""
        mock_get_embedder.return_value = mock_embedder
        mock_jarvis_db.get_training_pairs.return_value = sample_pairs

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            result = build_index_from_db(
                mock_jarvis_db,
                config=index_config,
                min_quality=0.5,
            )

        assert result["success"] is True
        assert result["pairs_indexed"] == 3
        assert "version_id" in result
        assert "index_path" in result

    def test_build_index_from_db_no_pairs(
        self,
        mock_jarvis_db: MagicMock,
        index_config: IndexConfig,
    ) -> None:
        """Test build_index_from_db when no pairs exist."""
        mock_jarvis_db.get_training_pairs.return_value = []

        result = build_index_from_db(
            mock_jarvis_db,
            config=index_config,
            min_quality=0.5,
        )

        assert result["success"] is False
        assert "No pairs found" in result["error"]
        assert result["pairs_indexed"] == 0

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_build_index_from_db_excludes_holdout_by_default(
        self,
        mock_get_embedder: MagicMock,
        mock_jarvis_db: MagicMock,
        sample_pairs: list[Pair],
        index_config: IndexConfig,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that holdout pairs are excluded by default."""
        mock_get_embedder.return_value = mock_embedder
        mock_jarvis_db.get_training_pairs.return_value = sample_pairs

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            build_index_from_db(mock_jarvis_db, config=index_config)

        # Should call get_training_pairs, not get_all_pairs
        mock_jarvis_db.get_training_pairs.assert_called_once()
        mock_jarvis_db.get_all_pairs.assert_not_called()

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_build_index_from_db_includes_holdout_when_requested(
        self,
        mock_get_embedder: MagicMock,
        mock_jarvis_db: MagicMock,
        sample_pairs: list[Pair],
        index_config: IndexConfig,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test that holdout pairs can be included."""
        mock_get_embedder.return_value = mock_embedder
        mock_jarvis_db.get_all_pairs.return_value = sample_pairs

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            build_index_from_db(
                mock_jarvis_db,
                config=index_config,
                include_holdout=True,
            )

        mock_jarvis_db.get_all_pairs.assert_called_once()


# =============================================================================
# get_index_stats Tests
# =============================================================================


class TestGetIndexStats:
    """Tests for get_index_stats function."""

    def test_get_index_stats_no_db_no_index(self, temp_jarvis_dir: Path) -> None:
        """Test get_index_stats with no database and no legacy index."""
        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            result = get_index_stats(jarvis_db=None)

        assert result is None

    def test_get_index_stats_no_active_index(self, mock_jarvis_db: MagicMock) -> None:
        """Test get_index_stats when no active index exists."""
        mock_jarvis_db.get_active_index.return_value = None

        result = get_index_stats(jarvis_db=mock_jarvis_db)
        assert result is None

    def test_get_index_stats_active_index(
        self,
        mock_jarvis_db: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test get_index_stats with active index."""
        # Create a mock index file
        index_file = temp_jarvis_dir / "test.faiss"
        index_file.write_bytes(b"fake index data")

        mock_active_index = IndexVersion(
            id=1,
            version_id="20240115-143022",
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
            num_vectors=100,
            index_path="test.faiss",
            is_active=True,
            created_at=datetime.now(),
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            result = get_index_stats(jarvis_db=mock_jarvis_db)

        assert result is not None
        assert result["exists"] is True
        assert result["version_id"] == "20240115-143022"
        assert result["num_vectors"] == 100
        assert result["dimension"] == 384

    def test_get_index_stats_missing_file(
        self,
        mock_jarvis_db: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test get_index_stats when index file is missing."""
        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=100,
            index_path="nonexistent.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            result = get_index_stats(jarvis_db=mock_jarvis_db)

        assert result is not None
        assert result["exists"] is False
        assert "error" in result


# =============================================================================
# list_index_versions Tests
# =============================================================================


class TestListIndexVersions:
    """Tests for list_index_versions function."""

    def test_list_index_versions_empty(self, mock_jarvis_db: MagicMock) -> None:
        """Test listing when no versions exist."""
        mock_jarvis_db.list_index_versions.return_value = []

        result = list_index_versions(mock_jarvis_db)
        assert result == []

    def test_list_index_versions_multiple(self, mock_jarvis_db: MagicMock) -> None:
        """Test listing multiple versions."""
        now = datetime.now()
        mock_versions = [
            IndexVersion(
                id=1,
                version_id="20240115-143022",
                model_name="BAAI/bge-small-en-v1.5",
                embedding_dim=384,
                num_vectors=100,
                index_path="v1/index.faiss",
                is_active=False,
                created_at=now - timedelta(days=1),
            ),
            IndexVersion(
                id=2,
                version_id="20240116-093045",
                model_name="BAAI/bge-small-en-v1.5",
                embedding_dim=384,
                num_vectors=150,
                index_path="v2/index.faiss",
                is_active=True,
                created_at=now,
            ),
        ]
        mock_jarvis_db.list_index_versions.return_value = mock_versions

        result = list_index_versions(mock_jarvis_db)

        assert len(result) == 2
        assert result[0]["version_id"] == "20240115-143022"
        assert result[0]["is_active"] is False
        assert result[1]["version_id"] == "20240116-093045"
        assert result[1]["is_active"] is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_duplicate_triggers(
        self,
        mock_get_embedder: MagicMock,
        index_config: IndexConfig,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test handling of duplicate trigger texts."""
        mock_get_embedder.return_value = mock_embedder
        now = datetime.now()
        # Create pairs with duplicate trigger texts
        pairs = [
            Pair(
                id=1,
                contact_id=1,
                trigger_text="Hello there!",
                response_text="Hi!",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="chat1",
            ),
            Pair(
                id=2,
                contact_id=1,
                trigger_text="Hello there!",  # Same trigger
                response_text="Hey!",  # Different response
                trigger_timestamp=now - timedelta(hours=1),
                response_timestamp=now - timedelta(hours=1),
                chat_id="chat1",
            ),
        ]

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            builder = TriggerIndexBuilder(config=index_config)
            builder._embedder = mock_embedder

            # Should not raise, should index both
            stats = builder.build_index(pairs, mock_jarvis_db)

        assert stats.pairs_indexed == 2

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_empty_query(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test search with empty query string."""
        mock_get_embedder.return_value = mock_embedder

        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=5,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        # Embedder returns embedding for empty string
        mock_embedder.encode.return_value = np.zeros((1, 384), dtype=np.float32)

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            # Low threshold to potentially get results
            results = searcher.search("", k=5, threshold=0.0)

        # Empty query should still work (embedder handles it)
        assert isinstance(results, list)

    @patch.dict("sys.modules", {"faiss": mock_faiss_module})
    @patch("jarvis.embedding_adapter.get_embedder")
    def test_search_no_matches_above_threshold(
        self,
        mock_get_embedder: MagicMock,
        mock_faiss: MagicMock,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test search when no results meet the threshold."""
        mock_get_embedder.return_value = mock_embedder

        mock_active_index = IndexVersion(
            id=1,
            version_id="test",
            model_name="test",
            embedding_dim=384,
            num_vectors=5,
            index_path="test.faiss",
            is_active=True,
        )
        mock_jarvis_db.get_active_index.return_value = mock_active_index

        (temp_jarvis_dir / "test.faiss").write_bytes(b"fake")

        mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        mock_index = MagicMock()
        # All scores below threshold
        mock_index.search.return_value = (
            np.array([[0.3, 0.2, 0.1, 0.05, 0.01]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        mock_faiss.read_index.return_value = mock_index

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            searcher = TriggerIndexSearcher(jarvis_db=mock_jarvis_db)
            searcher._embedder = mock_embedder
            results = searcher.search("test query", k=5, threshold=0.5)

        assert results == []

    @patch("jarvis.embedding_adapter.get_embedder")
    def test_single_pair(
        self,
        mock_get_embedder: MagicMock,
        index_config: IndexConfig,
        mock_jarvis_db: MagicMock,
        mock_embedder: MagicMock,
        temp_jarvis_dir: Path,
    ) -> None:
        """Test building index with a single pair."""
        mock_get_embedder.return_value = mock_embedder
        single_pair = [
            Pair(
                id=1,
                contact_id=1,
                trigger_text="Single trigger",
                response_text="Single response",
                trigger_timestamp=datetime.now(),
                response_timestamp=datetime.now(),
                chat_id="chat1",
            )
        ]

        with patch("jarvis.index.JARVIS_DIR", temp_jarvis_dir):
            builder = TriggerIndexBuilder(config=index_config)
            builder._embedder = mock_embedder
            stats = builder.build_index(single_pair, mock_jarvis_db)

        assert stats.pairs_indexed == 1
        assert stats.embeddings_stored == 1
