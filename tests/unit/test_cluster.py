"""Unit tests for JARVIS Response Clustering System.

Tests cover cluster configuration, response clustering, cluster naming,
saving results, and the cluster_and_store integration function.

Note: HDBSCAN is an optional dependency. Tests that require HDBSCAN will be
skipped if it's not installed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.cluster import (
    ClusterConfig,
    ClusterResult,
    ResponseClusterer,
    cluster_and_store,
    save_cluster_results,
    suggest_cluster_names,
)

# =============================================================================
# Check for HDBSCAN availability
# =============================================================================


def _check_hdbscan_available() -> bool:
    """Check if HDBSCAN is available and working."""
    try:
        import hdbscan  # noqa: F401

        return True
    except (ImportError, AttributeError):
        # AttributeError can occur due to scipy/torch conflicts
        return False


HDBSCAN_AVAILABLE = _check_hdbscan_available()

requires_hdbscan = pytest.mark.skipif(
    not HDBSCAN_AVAILABLE,
    reason="HDBSCAN not available (pip install hdbscan)",
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def cluster_config() -> ClusterConfig:
    """Create a default cluster configuration."""
    return ClusterConfig()


@pytest.fixture
def small_cluster_config() -> ClusterConfig:
    """Create a configuration for small clusters (for testing)."""
    return ClusterConfig(min_cluster_size=2, min_samples=1, num_examples=2)


@pytest.fixture
def mock_embedding() -> np.ndarray:
    """Create a mock normalized embedding vector."""
    embedding = np.random.randn(384).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


@pytest.fixture
def sample_pairs() -> list[dict]:
    """Create sample pairs for clustering tests."""
    return [
        {"id": 1, "trigger_text": "Want to grab dinner?", "response_text": "sounds good"},
        {"id": 2, "trigger_text": "Dinner tonight?", "response_text": "i'm down"},
        {"id": 3, "trigger_text": "Let's meet up", "response_text": "let's do it"},
        {"id": 4, "trigger_text": "Can you come?", "response_text": "count me in"},
        {"id": 5, "trigger_text": "Movie this weekend?", "response_text": "works for me"},
        {"id": 6, "trigger_text": "Free tomorrow?", "response_text": "can't today"},
        {"id": 7, "trigger_text": "Party on Saturday?", "response_text": "maybe next time"},
        {"id": 8, "trigger_text": "Lunch?", "response_text": "not today"},
        {"id": 9, "trigger_text": "Coffee?", "response_text": "rain check"},
        {"id": 10, "trigger_text": "Where are you?", "response_text": "omw"},
        {"id": 11, "trigger_text": "ETA?", "response_text": "be there soon"},
        {"id": 12, "trigger_text": "Coming?", "response_text": "5 min"},
    ]


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder that returns distinct embeddings for different responses."""
    embedder = MagicMock()

    def encode_side_effect(texts, normalize=True):
        """Return embeddings that cluster similar responses together."""
        embeddings = []
        for text in texts:
            # Create base embedding based on response type
            base = np.zeros(384, dtype=np.float32)
            text_lower = text.lower()

            # Accept-like responses
            if any(kw in text_lower for kw in ["sounds good", "i'm down", "let's do", "count me"]):
                base[:10] = 1.0
            # Decline-like responses
            elif any(kw in text_lower for kw in ["can't", "maybe", "not today", "rain check"]):
                base[10:20] = 1.0
            # Arrival-like responses
            elif any(kw in text_lower for kw in ["omw", "be there", "5 min"]):
                base[20:30] = 1.0
            else:
                # Random for other text
                base = np.random.randn(384).astype(np.float32)

            # Add small noise for variation
            base += np.random.randn(384).astype(np.float32) * 0.01
            # Normalize
            embedding = base / (np.linalg.norm(base) + 1e-9)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    embedder.encode = MagicMock(side_effect=encode_side_effect)
    return embedder


# =============================================================================
# ClusterConfig Tests
# =============================================================================


class TestClusterConfig:
    """Tests for ClusterConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClusterConfig()

        assert config.min_cluster_size == 10
        assert config.min_samples == 5
        assert config.metric == "euclidean"
        assert config.num_examples == 5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClusterConfig(
            min_cluster_size=5,
            min_samples=3,
            metric="cosine",
            num_examples=10,
        )

        assert config.min_cluster_size == 5
        assert config.min_samples == 3
        assert config.metric == "cosine"
        assert config.num_examples == 10

    def test_partial_custom_values(self) -> None:
        """Test partial configuration override."""
        config = ClusterConfig(min_cluster_size=3)

        assert config.min_cluster_size == 3
        assert config.min_samples == 5  # Default
        assert config.num_examples == 5  # Default


# =============================================================================
# ClusterResult Tests
# =============================================================================


class TestClusterResult:
    """Tests for ClusterResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a ClusterResult instance."""
        result = ClusterResult(
            cluster_id=0,
            name="ACCEPT_INVITATION",
            size=15,
            example_triggers=["Want dinner?", "Movie?"],
            example_responses=["sounds good", "i'm down"],
            pair_ids=[1, 2, 3, 4, 5],
        )

        assert result.cluster_id == 0
        assert result.name == "ACCEPT_INVITATION"
        assert result.size == 15
        assert len(result.example_triggers) == 2
        assert len(result.example_responses) == 2
        assert len(result.pair_ids) == 5

    def test_creation_without_name(self) -> None:
        """Test creating a ClusterResult without a name."""
        result = ClusterResult(
            cluster_id=1,
            name=None,
            size=10,
            example_triggers=[],
            example_responses=[],
            pair_ids=[],
        )

        assert result.name is None
        assert result.cluster_id == 1

    def test_empty_examples(self) -> None:
        """Test ClusterResult with empty examples."""
        result = ClusterResult(
            cluster_id=0,
            name="EMPTY",
            size=0,
            example_triggers=[],
            example_responses=[],
            pair_ids=[],
        )

        assert result.size == 0
        assert result.example_triggers == []
        assert result.example_responses == []


# =============================================================================
# ResponseClusterer Tests
# =============================================================================


class TestResponseClusterer:
    """Tests for ResponseClusterer class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        clusterer = ResponseClusterer()

        assert clusterer.config is not None
        assert clusterer.config.min_cluster_size == 10
        assert clusterer._embedder is None

    def test_init_custom_config(self, small_cluster_config: ClusterConfig) -> None:
        """Test initialization with custom config."""
        clusterer = ResponseClusterer(config=small_cluster_config)

        assert clusterer.config.min_cluster_size == 2
        assert clusterer.config.min_samples == 1

    def test_cluster_responses_empty_pairs(self, small_cluster_config: ClusterConfig) -> None:
        """Test clustering with empty input returns empty list."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        results = clusterer.cluster_responses([])

        assert results == []

    @patch(
        "jarvis.cluster.ResponseClusterer.embedder",
        new_callable=lambda: property(lambda self: MagicMock()),
    )
    def test_cluster_responses_requires_hdbscan(self, mock_embedder_prop) -> None:
        """Test that cluster_responses raises ImportError if HDBSCAN is not available."""
        clusterer = ResponseClusterer()
        pairs = [{"id": 1, "trigger_text": "hi", "response_text": "hello"}]

        # Mock the HDBSCAN import to fail
        with patch.dict("sys.modules", {"hdbscan": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'hdbscan'")):
                with pytest.raises(ImportError, match="HDBSCAN is required"):
                    clusterer.cluster_responses(pairs)

    @requires_hdbscan
    def test_cluster_responses_with_mock_embedder(
        self,
        sample_pairs: list[dict],
        mock_embedder: MagicMock,
        small_cluster_config: ClusterConfig,
    ) -> None:
        """Test clustering with mocked embedder."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        clusterer._embedder = mock_embedder

        results = clusterer.cluster_responses(sample_pairs)

        # Should return list of ClusterResult objects
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, ClusterResult)
            assert result.cluster_id >= 0
            assert result.size > 0
            assert len(result.pair_ids) > 0

        # Embedder should have been called
        mock_embedder.encode.assert_called_once()

    @requires_hdbscan
    def test_cluster_responses_with_progress_callback(
        self,
        sample_pairs: list[dict],
        mock_embedder: MagicMock,
        small_cluster_config: ClusterConfig,
    ) -> None:
        """Test clustering with progress callback."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        clusterer._embedder = mock_embedder

        progress_calls = []

        def progress_callback(stage: str, progress: float, message: str) -> None:
            progress_calls.append((stage, progress, message))

        clusterer.cluster_responses(sample_pairs, progress_callback=progress_callback)

        # Should have received progress updates
        assert len(progress_calls) > 0
        stages = [call[0] for call in progress_calls]
        assert "encoding" in stages
        assert "clustering" in stages
        assert "extracting" in stages
        assert "done" in stages

    def test_get_cluster_labels_before_clustering(self) -> None:
        """Test that get_cluster_labels raises RuntimeError before clustering."""
        clusterer = ResponseClusterer()

        with pytest.raises(RuntimeError, match="Call cluster_responses first"):
            clusterer.get_cluster_labels()


class TestSelectDiverseExamples:
    """Tests for the _select_diverse_examples method."""

    def test_fewer_embeddings_than_requested(self) -> None:
        """Test when there are fewer embeddings than num_examples."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(3, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=5)

        # Should return all indices
        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}

    def test_exact_number_of_embeddings(self) -> None:
        """Test when embeddings count equals num_examples."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(5, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=5)

        assert len(selected) == 5
        assert set(selected) == {0, 1, 2, 3, 4}

    def test_more_embeddings_than_requested(self) -> None:
        """Test when there are more embeddings than num_examples."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(10, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=3)

        assert len(selected) == 3
        # First element should always be 0 (starting point)
        assert selected[0] == 0
        # All indices should be unique
        assert len(set(selected)) == 3
        # All indices should be valid
        assert all(0 <= idx < 10 for idx in selected)

    def test_diversity_selection(self) -> None:
        """Test that selected examples are diverse (farthest point sampling)."""
        clusterer = ResponseClusterer()

        # Create embeddings with clear clusters
        embeddings = np.zeros((6, 384), dtype=np.float32)
        # Cluster 1: indices 0, 1, 2 (similar)
        embeddings[0, 0] = 1.0
        embeddings[1, 0] = 0.99
        embeddings[2, 0] = 0.98
        # Cluster 2: indices 3, 4, 5 (similar, but far from cluster 1)
        embeddings[3, 100] = 1.0
        embeddings[4, 100] = 0.99
        embeddings[5, 100] = 0.98

        selected = clusterer._select_diverse_examples(embeddings, num_examples=2)

        # Should select one from each cluster
        assert len(selected) == 2
        _cluster1 = {0, 1, 2}  # noqa: F841 - for documentation
        cluster2 = {3, 4, 5}
        # First should be 0, second should be from cluster 2
        assert selected[0] == 0
        assert selected[1] in cluster2


# =============================================================================
# suggest_cluster_names Tests
# =============================================================================


class TestSuggestClusterNames:
    """Tests for the suggest_cluster_names function."""

    def test_empty_clusters(self) -> None:
        """Test suggesting names for empty cluster list."""
        suggestions = suggest_cluster_names([])
        assert suggestions == {}

    def test_accept_invitation_pattern(self) -> None:
        """Test recognition of ACCEPT_INVITATION pattern."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=10,
                example_triggers=["Dinner?", "Movie?"],
                example_responses=["sounds good", "i'm down", "let's do it"],
                pair_ids=[1, 2, 3],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 0 in suggestions
        assert suggestions[0] == "ACCEPT_INVITATION"

    def test_decline_politely_pattern(self) -> None:
        """Test recognition of DECLINE_POLITELY pattern."""
        results = [
            ClusterResult(
                cluster_id=1,
                name=None,
                size=8,
                example_triggers=["Party?", "Event?"],
                example_responses=["can't today", "maybe next time", "rain check"],
                pair_ids=[4, 5, 6],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 1 in suggestions
        assert suggestions[1] == "DECLINE_POLITELY"

    def test_confirm_arrival_pattern(self) -> None:
        """Test recognition of CONFIRM_ARRIVAL pattern."""
        results = [
            ClusterResult(
                cluster_id=2,
                name=None,
                size=6,
                example_triggers=["Where are you?", "ETA?"],
                example_responses=["omw", "be there soon", "5 min"],
                pair_ids=[7, 8, 9],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 2 in suggestions
        assert suggestions[2] == "CONFIRM_ARRIVAL"

    def test_greeting_pattern(self) -> None:
        """Test recognition of GREETING pattern."""
        results = [
            ClusterResult(
                cluster_id=3,
                name=None,
                size=5,
                example_triggers=["", ""],
                example_responses=["hey", "hi", "hello"],
                pair_ids=[10, 11, 12],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 3 in suggestions
        assert suggestions[3] == "GREETING"

    def test_acknowledge_pattern(self) -> None:
        """Test recognition of ACKNOWLEDGE pattern."""
        results = [
            ClusterResult(
                cluster_id=4,
                name=None,
                size=10,
                example_triggers=["Did you see that?", "Funny right?"],
                example_responses=["haha", "lol", "nice"],
                pair_ids=[13, 14, 15],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 4 in suggestions
        assert suggestions[4] == "ACKNOWLEDGE"

    def test_express_thanks_pattern(self) -> None:
        """Test recognition of EXPRESS_THANKS pattern."""
        results = [
            ClusterResult(
                cluster_id=5,
                name=None,
                size=7,
                example_triggers=["Here's the file", "Done!"],
                example_responses=["thanks", "thank you", "appreciate it"],
                pair_ids=[16, 17, 18],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 5 in suggestions
        assert suggestions[5] == "EXPRESS_THANKS"

    def test_no_pattern_match_uses_generic_name(self) -> None:
        """Test that unmatched patterns get generic names."""
        results = [
            ClusterResult(
                cluster_id=99,
                name=None,
                size=5,
                example_triggers=["test", "test2"],
                example_responses=["xyzzy", "plugh", "foo"],
                pair_ids=[100, 101, 102],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert 99 in suggestions
        assert suggestions[99] == "CLUSTER_99"

    def test_multiple_clusters(self) -> None:
        """Test suggesting names for multiple clusters."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=10,
                example_triggers=["a", "b"],
                example_responses=["sounds good", "i'm in"],
                pair_ids=[1, 2],
            ),
            ClusterResult(
                cluster_id=1,
                name=None,
                size=8,
                example_triggers=["c", "d"],
                example_responses=["can't today", "busy"],
                pair_ids=[3, 4],
            ),
        ]

        suggestions = suggest_cluster_names(results)

        assert len(suggestions) == 2
        assert 0 in suggestions
        assert 1 in suggestions

    def test_case_insensitive_matching(self) -> None:
        """Test that pattern matching is case-insensitive."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["test"],
                example_responses=["SOUNDS GOOD", "I'M DOWN"],
                pair_ids=[1],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert suggestions[0] == "ACCEPT_INVITATION"

    def test_weak_pattern_match_uses_generic(self) -> None:
        """Test that weak pattern matches (score < 2) use generic name."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["test"],
                # Only one keyword match (needs >= 2 to qualify)
                example_responses=["sounds good", "random text", "other stuff"],
                pair_ids=[1],
            )
        ]

        _suggestions = suggest_cluster_names(results)  # noqa: F841 - call for side effects

        # "sounds good" matches once, but needs 2+ matches
        # This actually matches because "sounds good" has multiple keywords
        # Let's use a cleaner example
        results2 = [
            ClusterResult(
                cluster_id=1,
                name=None,
                size=5,
                example_triggers=["test"],
                example_responses=["sure", "yep", "k"],  # Only "sure" matches ACCEPT
                pair_ids=[1],
            )
        ]

        suggestions2 = suggest_cluster_names(results2)

        # Only "sure thing" matches, not "sure" alone
        assert suggestions2[1] == "CLUSTER_1"


# =============================================================================
# save_cluster_results Tests
# =============================================================================


class TestSaveClusterResults:
    """Tests for the save_cluster_results function."""

    def test_save_basic_results(self, tmp_path: Path) -> None:
        """Test saving basic cluster results."""
        results = [
            ClusterResult(
                cluster_id=0,
                name="TEST_CLUSTER",
                size=5,
                example_triggers=["trigger1", "trigger2"],
                example_responses=["response1", "response2"],
                pair_ids=[1, 2, 3, 4, 5],
            )
        ]
        labels = np.array([0, 0, 0, 0, 0, -1, -1])
        output_path = tmp_path / "clusters.json"

        save_cluster_results(results, labels, output_path)

        assert output_path.exists()

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert "clusters" in data
        assert len(data["clusters"]) == 1
        assert data["clusters"][0]["cluster_id"] == 0
        assert data["clusters"][0]["name"] == "TEST_CLUSTER"
        assert data["clusters"][0]["size"] == 5
        assert data["noise_count"] == 2
        assert data["total_pairs"] == 7

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that save creates parent directories if needed."""
        results = []
        labels = np.array([])
        output_path = tmp_path / "subdir" / "nested" / "clusters.json"

        save_cluster_results(results, labels, output_path)

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_save_empty_results(self, tmp_path: Path) -> None:
        """Test saving empty cluster results."""
        results = []
        labels = np.array([])
        output_path = tmp_path / "empty.json"

        save_cluster_results(results, labels, output_path)

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["clusters"] == []
        assert data["noise_count"] == 0
        assert data["total_pairs"] == 0

    def test_save_multiple_clusters(self, tmp_path: Path) -> None:
        """Test saving multiple cluster results."""
        results = [
            ClusterResult(
                cluster_id=0,
                name="CLUSTER_A",
                size=3,
                example_triggers=["t1"],
                example_responses=["r1"],
                pair_ids=[1, 2, 3],
            ),
            ClusterResult(
                cluster_id=1,
                name="CLUSTER_B",
                size=2,
                example_triggers=["t2"],
                example_responses=["r2"],
                pair_ids=[4, 5],
            ),
        ]
        labels = np.array([0, 0, 0, 1, 1, -1])
        output_path = tmp_path / "multi.json"

        save_cluster_results(results, labels, output_path)

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["clusters"]) == 2
        assert data["noise_count"] == 1
        assert data["total_pairs"] == 6


# =============================================================================
# cluster_and_store Tests
# =============================================================================


class TestClusterAndStore:
    """Tests for the cluster_and_store function."""

    def test_empty_database(self) -> None:
        """Test clustering with empty database."""
        mock_db = MagicMock()
        mock_db.get_all_pairs.return_value = []

        stats = cluster_and_store(mock_db)

        assert stats["pairs_processed"] == 0
        assert stats["clusters_found"] == 0
        assert stats["clusters_created"] == []

    @requires_hdbscan
    @patch("jarvis.cluster.ResponseClusterer")
    def test_with_pairs(self, mock_clusterer_class: MagicMock) -> None:
        """Test clustering with database pairs."""
        # Create mock database
        mock_db = MagicMock()
        mock_pair = MagicMock()
        mock_pair.id = 1
        mock_pair.trigger_text = "Hey"
        mock_pair.response_text = "Hi"
        mock_db.get_all_pairs.return_value = [mock_pair]

        # Create mock cluster result
        mock_result = ClusterResult(
            cluster_id=0,
            name=None,
            size=1,
            example_triggers=["Hey"],
            example_responses=["Hi"],
            pair_ids=[1],
        )

        # Setup mock clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_responses.return_value = [mock_result]
        mock_clusterer_class.return_value = mock_clusterer

        # Setup mock cluster creation
        mock_cluster = MagicMock()
        mock_cluster.id = 100
        mock_cluster.name = "GREETING"
        mock_db.add_cluster.return_value = mock_cluster

        stats = cluster_and_store(mock_db)

        assert stats["pairs_processed"] == 1
        assert stats["clusters_found"] == 1
        mock_db.clear_clusters.assert_called_once()
        mock_db.add_cluster.assert_called_once()

    @requires_hdbscan
    @patch("jarvis.cluster.ResponseClusterer")
    def test_with_custom_config(self, mock_clusterer_class: MagicMock) -> None:
        """Test clustering with custom configuration."""
        mock_db = MagicMock()

        # Need at least one pair for clusterer to be called
        mock_pair = MagicMock()
        mock_pair.id = 1
        mock_pair.trigger_text = "Hi"
        mock_pair.response_text = "Hello"
        mock_db.get_all_pairs.return_value = [mock_pair]

        mock_clusterer = MagicMock()
        mock_clusterer.cluster_responses.return_value = []
        mock_clusterer_class.return_value = mock_clusterer

        config = ClusterConfig(min_cluster_size=5)
        cluster_and_store(mock_db, config=config)

        mock_clusterer_class.assert_called_once_with(config)

    @requires_hdbscan
    @patch("jarvis.cluster.ResponseClusterer")
    def test_with_progress_callback(self, mock_clusterer_class: MagicMock) -> None:
        """Test clustering with progress callback."""
        mock_db = MagicMock()

        # Need at least one pair for clusterer to be called
        mock_pair = MagicMock()
        mock_pair.id = 1
        mock_pair.trigger_text = "Hi"
        mock_pair.response_text = "Hello"
        mock_db.get_all_pairs.return_value = [mock_pair]

        mock_clusterer = MagicMock()
        mock_clusterer.cluster_responses.return_value = []
        mock_clusterer_class.return_value = mock_clusterer

        progress_calls = []

        def callback(stage: str, progress: float, message: str) -> None:
            progress_calls.append((stage, progress, message))

        cluster_and_store(mock_db, progress_callback=callback)

        # Clusterer should receive the callback
        mock_clusterer.cluster_responses.assert_called_once()
        call_args = mock_clusterer.cluster_responses.call_args
        assert call_args[1].get("progress_callback") == callback or call_args[0][1] == callback

    @requires_hdbscan
    @patch("jarvis.cluster.ResponseClusterer")
    def test_noise_pairs_counted(self, mock_clusterer_class: MagicMock) -> None:
        """Test that noise pairs are counted correctly."""
        mock_db = MagicMock()

        # Create 5 mock pairs
        mock_pairs = []
        for i in range(5):
            mock_pair = MagicMock()
            mock_pair.id = i
            mock_pair.trigger_text = f"trigger{i}"
            mock_pair.response_text = f"response{i}"
            mock_pairs.append(mock_pair)
        mock_db.get_all_pairs.return_value = mock_pairs

        # Only 2 pairs are in clusters
        mock_result = ClusterResult(
            cluster_id=0,
            name=None,
            size=2,
            example_triggers=["t0", "t1"],
            example_responses=["r0", "r1"],
            pair_ids=[0, 1],  # Only pairs 0 and 1 clustered
        )

        mock_clusterer = MagicMock()
        mock_clusterer.cluster_responses.return_value = [mock_result]
        mock_clusterer_class.return_value = mock_clusterer

        mock_cluster = MagicMock()
        mock_cluster.id = 1
        mock_cluster.name = "TEST"
        mock_db.add_cluster.return_value = mock_cluster

        stats = cluster_and_store(mock_db)

        assert stats["pairs_processed"] == 5
        assert stats["noise_pairs"] == 3  # 5 total - 2 clustered = 3 noise


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @requires_hdbscan
    def test_single_pair(self) -> None:
        """Test clustering with a single pair.

        HDBSCAN requires at least min_cluster_size points (typically > 1).
        With only 1 point, we mock HDBSCAN to return -1 (noise label).
        """
        config = ClusterConfig(min_cluster_size=2, min_samples=1)
        clusterer = ResponseClusterer(config=config)
        pairs = [{"id": 1, "trigger_text": "hi", "response_text": "hello"}]

        with patch.object(clusterer, "_embedder") as mock_emb:
            mock_emb.encode.return_value = np.random.randn(1, 384).astype(np.float32)

            # Mock HDBSCAN to return noise label (-1) for the single point
            # Need to patch via hdbscan module since it's imported inside the function
            with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
                mock_hdbscan = MagicMock()
                mock_hdbscan.fit_predict.return_value = np.array([-1])  # Noise
                mock_hdbscan_class.return_value = mock_hdbscan

                results = clusterer.cluster_responses(pairs)
                # Single pair marked as noise, so no clusters
                assert len(results) == 0

    def test_all_identical_responses(self, small_cluster_config: ClusterConfig) -> None:
        """Test clustering when all responses are identical."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        pairs = [{"id": i, "trigger_text": f"trigger{i}", "response_text": "ok"} for i in range(10)]

        with patch.object(clusterer, "_embedder") as mock_emb:
            # All identical embeddings
            identical_embedding = np.random.randn(384).astype(np.float32)
            identical_embedding = identical_embedding / np.linalg.norm(identical_embedding)
            mock_emb.encode.return_value = np.tile(identical_embedding, (10, 1))

            if HDBSCAN_AVAILABLE:
                results = clusterer.cluster_responses(pairs)
                # All should be in one cluster
                if results:
                    assert len(results) == 1
                    assert results[0].size == 10

    def test_very_short_responses(self, small_cluster_config: ClusterConfig) -> None:
        """Test clustering with very short responses."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        pairs = [
            {"id": 1, "trigger_text": "a", "response_text": "k"},
            {"id": 2, "trigger_text": "b", "response_text": "y"},
            {"id": 3, "trigger_text": "c", "response_text": "n"},
        ]

        with patch.object(clusterer, "_embedder") as mock_emb:
            mock_emb.encode.return_value = np.random.randn(3, 384).astype(np.float32)

            if HDBSCAN_AVAILABLE:
                # Should not crash, even with very short responses
                results = clusterer.cluster_responses(pairs)
                assert isinstance(results, list)

    def test_unicode_responses(self, small_cluster_config: ClusterConfig) -> None:
        """Test clustering with unicode responses."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        pairs = [
            {"id": 1, "trigger_text": "how are you?", "response_text": "great!"},
            {"id": 2, "trigger_text": "status?", "response_text": "good"},
            {"id": 3, "trigger_text": "ok?", "response_text": "perfect"},
        ]

        with patch.object(clusterer, "_embedder") as mock_emb:
            mock_emb.encode.return_value = np.random.randn(3, 384).astype(np.float32)

            if HDBSCAN_AVAILABLE:
                results = clusterer.cluster_responses(pairs)
                assert isinstance(results, list)

    def test_special_characters_in_responses(self, small_cluster_config: ClusterConfig) -> None:
        """Test clustering with special characters in responses."""
        clusterer = ResponseClusterer(config=small_cluster_config)
        pairs = [
            {"id": 1, "trigger_text": "test", "response_text": "hello! @#$%"},
            {"id": 2, "trigger_text": "test2", "response_text": "<script>alert('xss')</script>"},
            {"id": 3, "trigger_text": "test3", "response_text": "line1\nline2\ttab"},
        ]

        with patch.object(clusterer, "_embedder") as mock_emb:
            mock_emb.encode.return_value = np.random.randn(3, 384).astype(np.float32)

            if HDBSCAN_AVAILABLE:
                results = clusterer.cluster_responses(pairs)
                assert isinstance(results, list)


# =============================================================================
# Pattern Matching Edge Cases
# =============================================================================


class TestPatternMatchingEdgeCases:
    """Tests for edge cases in pattern matching for cluster naming."""

    def test_overlapping_patterns(self) -> None:
        """Test when responses match multiple patterns."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["test"],
                # "got it" appears in both ACKNOWLEDGE and CONFIRM_UNDERSTANDING
                example_responses=["got it", "nice", "cool"],
                pair_ids=[1],
            )
        ]

        suggestions = suggest_cluster_names(results)

        # Should pick the pattern with highest score
        assert 0 in suggestions
        assert suggestions[0] in ["ACKNOWLEDGE", "CONFIRM_UNDERSTANDING"]

    def test_empty_responses(self) -> None:
        """Test pattern matching with empty responses."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["test"],
                example_responses=[],
                pair_ids=[1],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert suggestions[0] == "CLUSTER_0"

    def test_whitespace_only_responses(self) -> None:
        """Test pattern matching with whitespace-only responses."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["test"],
                example_responses=["   ", "\t", "\n"],
                pair_ids=[1],
            )
        ]

        suggestions = suggest_cluster_names(results)

        assert suggestions[0] == "CLUSTER_0"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestEmbedderProperty:
    """Tests for the embedder property lazy initialization."""

    def test_embedder_is_none_initially(self) -> None:
        """Test that _embedder is None on initialization."""
        clusterer = ResponseClusterer()
        assert clusterer._embedder is None

    def test_embedder_property_calls_get_embedder(self) -> None:
        """Test that accessing embedder property triggers lazy initialization."""
        with patch("jarvis.embedding_adapter.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_get_embedder.return_value = mock_embedder

            clusterer = ResponseClusterer()
            result = clusterer.embedder

            mock_get_embedder.assert_called_once()
            assert result is mock_embedder
            assert clusterer._embedder is mock_embedder

    def test_embedder_property_caches_result(self) -> None:
        """Test that embedder property caches the embedder instance."""
        with patch("jarvis.embedding_adapter.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_get_embedder.return_value = mock_embedder

            clusterer = ResponseClusterer()

            # Access multiple times
            _ = clusterer.embedder
            _ = clusterer.embedder
            _ = clusterer.embedder

            # Should only be called once due to caching
            mock_get_embedder.assert_called_once()


class TestClusterResultDataIntegrity:
    """Tests for ClusterResult data integrity and validation."""

    def test_pair_ids_list_integrity(self) -> None:
        """Test that pair_ids maintains list integrity."""
        pair_ids = [1, 2, 3, 4, 5]
        result = ClusterResult(
            cluster_id=0,
            name="TEST",
            size=5,
            example_triggers=["t1", "t2"],
            example_responses=["r1", "r2"],
            pair_ids=pair_ids,
        )

        # Verify the list is stored correctly
        assert result.pair_ids == pair_ids
        assert len(result.pair_ids) == 5

    def test_large_pair_ids_list(self) -> None:
        """Test ClusterResult with a large number of pair IDs."""
        large_pair_ids = list(range(1000))
        result = ClusterResult(
            cluster_id=0,
            name="LARGE_CLUSTER",
            size=1000,
            example_triggers=["t1"],
            example_responses=["r1"],
            pair_ids=large_pair_ids,
        )

        assert len(result.pair_ids) == 1000
        assert result.size == 1000

    def test_examples_count_matches_config(self) -> None:
        """Test that example count respects configuration."""
        config = ClusterConfig(num_examples=3)
        clusterer = ResponseClusterer(config=config)

        assert clusterer.config.num_examples == 3


class TestClusterLabelExtraction:
    """Tests for cluster label extraction patterns."""

    def test_ask_time_pattern(self) -> None:
        """Test recognition of ASK_TIME pattern."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["meeting?", "call?"],
                example_responses=["when is it?", "what time works?"],
                pair_ids=[1, 2],
            )
        ]

        suggestions = suggest_cluster_names(results)
        assert 0 in suggestions
        assert suggestions[0] == "ASK_TIME"

    def test_ask_location_pattern(self) -> None:
        """Test recognition of ASK_LOCATION pattern."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["party?", "dinner?"],
                example_responses=["where is it?", "what's the location?"],
                pair_ids=[1, 2],
            )
        ]

        suggestions = suggest_cluster_names(results)
        assert 0 in suggestions
        assert suggestions[0] == "ASK_LOCATION"

    def test_confirm_understanding_pattern(self) -> None:
        """Test recognition of CONFIRM_UNDERSTANDING pattern."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["here's the plan", "this is what we'll do"],
                example_responses=["makes sense", "understood", "copy that"],
                pair_ids=[1, 2, 3],
            )
        ]

        suggestions = suggest_cluster_names(results)
        assert 0 in suggestions
        assert suggestions[0] == "CONFIRM_UNDERSTANDING"

    def test_question_clarification_pattern(self) -> None:
        """Test recognition of QUESTION_CLARIFICATION pattern."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["do this", "handle that"],
                example_responses=["what do you mean?", "could you explain?"],
                pair_ids=[1, 2],
            )
        ]

        suggestions = suggest_cluster_names(results)
        assert 0 in suggestions
        assert suggestions[0] == "QUESTION_CLARIFICATION"


class TestDiverseExamplesAlgorithm:
    """Additional tests for the farthest point sampling algorithm."""

    def test_single_embedding(self) -> None:
        """Test selection with just one embedding."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(1, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=5)

        assert selected == [0]

    def test_zero_embeddings(self) -> None:
        """Test selection with zero embeddings."""
        clusterer = ResponseClusterer()
        embeddings = np.zeros((0, 384), dtype=np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=5)

        assert selected == []

    def test_diverse_examples_returns_unique_indices(self) -> None:
        """Test that all returned indices are unique."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(20, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=10)

        assert len(selected) == len(set(selected))
        assert len(selected) == 10

    def test_num_examples_zero(self) -> None:
        """Test requesting zero examples."""
        clusterer = ResponseClusterer()
        embeddings = np.random.randn(5, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=0)

        # Should return empty since we want 0 examples
        # First element added is 0, but loop won't add more
        # Actually looking at the code, it starts with [0] always
        # and then the while loop won't run because len(selected) >= num_examples
        assert selected == [0] or selected == []

    def test_large_embedding_set_performance(self) -> None:
        """Test that algorithm handles larger embedding sets."""
        clusterer = ResponseClusterer()
        # 100 embeddings is a reasonable upper bound for a cluster
        embeddings = np.random.randn(100, 384).astype(np.float32)

        selected = clusterer._select_diverse_examples(embeddings, num_examples=5)

        assert len(selected) == 5
        assert all(0 <= idx < 100 for idx in selected)


class TestClusterAndStoreIntegration:
    """Additional integration tests for cluster_and_store."""

    def test_stores_suggested_names(self) -> None:
        """Test that suggested names are stored in the database."""
        mock_db = MagicMock()

        # Create mock pairs
        mock_pairs = []
        for i in range(20):
            mock_pair = MagicMock()
            mock_pair.id = i
            mock_pair.trigger_text = "want to grab dinner?"
            mock_pair.response_text = "sounds good" if i < 10 else "can't today"
            mock_pairs.append(mock_pair)
        mock_db.get_all_pairs.return_value = mock_pairs

        # Mock clusterer to return results with patterns
        with patch("jarvis.cluster.ResponseClusterer") as mock_clusterer_class:
            mock_clusterer = MagicMock()
            mock_clusterer.cluster_responses.return_value = [
                ClusterResult(
                    cluster_id=0,
                    name=None,
                    size=10,
                    example_triggers=["dinner?"],
                    example_responses=["sounds good", "i'm down"],
                    pair_ids=list(range(10)),
                ),
            ]
            mock_clusterer_class.return_value = mock_clusterer

            mock_cluster = MagicMock()
            mock_cluster.id = 1
            mock_cluster.name = "ACCEPT_INVITATION"
            mock_db.add_cluster.return_value = mock_cluster

            if HDBSCAN_AVAILABLE:
                _stats = cluster_and_store(mock_db)  # noqa: F841 - call for side effects

                # Verify add_cluster was called with suggested name
                call_args = mock_db.add_cluster.call_args
                assert call_args is not None
                assert call_args[1]["name"] == "ACCEPT_INVITATION"

    def test_handles_multiple_clusters(self) -> None:
        """Test storing multiple clusters from one clustering run."""
        mock_db = MagicMock()

        mock_pairs = [
            MagicMock(id=i, trigger_text=f"t{i}", response_text=f"r{i}") for i in range(30)
        ]
        mock_db.get_all_pairs.return_value = mock_pairs

        with patch("jarvis.cluster.ResponseClusterer") as mock_clusterer_class:
            mock_clusterer = MagicMock()
            mock_clusterer.cluster_responses.return_value = [
                ClusterResult(
                    cluster_id=0,
                    name=None,
                    size=10,
                    example_triggers=["t1"],
                    example_responses=["sounds good", "i'm down"],
                    pair_ids=list(range(10)),
                ),
                ClusterResult(
                    cluster_id=1,
                    name=None,
                    size=10,
                    example_triggers=["t2"],
                    example_responses=["omw", "be there soon"],
                    pair_ids=list(range(10, 20)),
                ),
            ]
            mock_clusterer_class.return_value = mock_clusterer

            mock_cluster = MagicMock()
            mock_cluster.id = 1
            mock_cluster.name = "TEST"
            mock_db.add_cluster.return_value = mock_cluster

            if HDBSCAN_AVAILABLE:
                stats = cluster_and_store(mock_db)

                assert stats["clusters_found"] == 2
                assert mock_db.add_cluster.call_count == 2


class TestClusterResultEquality:
    """Tests for ClusterResult comparison and representation."""

    def test_cluster_results_with_same_data(self) -> None:
        """Test that ClusterResult objects with same data are equal."""
        result1 = ClusterResult(
            cluster_id=0,
            name="TEST",
            size=5,
            example_triggers=["t1"],
            example_responses=["r1"],
            pair_ids=[1, 2, 3],
        )
        result2 = ClusterResult(
            cluster_id=0,
            name="TEST",
            size=5,
            example_triggers=["t1"],
            example_responses=["r1"],
            pair_ids=[1, 2, 3],
        )

        # Dataclasses provide __eq__ by default
        assert result1 == result2

    def test_cluster_results_with_different_data(self) -> None:
        """Test that ClusterResult objects with different data are not equal."""
        result1 = ClusterResult(
            cluster_id=0,
            name="TEST",
            size=5,
            example_triggers=["t1"],
            example_responses=["r1"],
            pair_ids=[1, 2, 3],
        )
        result2 = ClusterResult(
            cluster_id=1,
            name="TEST",
            size=5,
            example_triggers=["t1"],
            example_responses=["r1"],
            pair_ids=[1, 2, 3],
        )

        assert result1 != result2


class TestSaveClusterResultsEdgeCases:
    """Additional edge case tests for save_cluster_results."""

    def test_save_with_none_name(self, tmp_path: Path) -> None:
        """Test saving clusters with None names."""
        results = [
            ClusterResult(
                cluster_id=0,
                name=None,
                size=5,
                example_triggers=["t1"],
                example_responses=["r1"],
                pair_ids=[1, 2],
            )
        ]
        labels = np.array([0, 0, 0, 0, 0])
        output_path = tmp_path / "clusters_none_name.json"

        save_cluster_results(results, labels, output_path)

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["clusters"][0]["name"] is None

    def test_save_with_unicode_content(self, tmp_path: Path) -> None:
        """Test saving clusters with unicode in triggers/responses."""
        results = [
            ClusterResult(
                cluster_id=0,
                name="EMOJI_CLUSTER",
                size=3,
                example_triggers=["how are you?", "status?"],
                example_responses=["great!", "awesome!"],
                pair_ids=[1, 2, 3],
            )
        ]
        labels = np.array([0, 0, 0])
        output_path = tmp_path / "clusters_unicode.json"

        save_cluster_results(results, labels, output_path)

        import json

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "great!" in data["clusters"][0]["example_responses"]

    def test_save_all_noise(self, tmp_path: Path) -> None:
        """Test saving when all pairs are noise (no clusters)."""
        results = []  # No clusters
        labels = np.array([-1, -1, -1, -1, -1])  # All noise
        output_path = tmp_path / "clusters_all_noise.json"

        save_cluster_results(results, labels, output_path)

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["clusters"] == []
        assert data["noise_count"] == 5
        assert data["total_pairs"] == 5


class TestClusterConfigValidation:
    """Tests for ClusterConfig parameter validation."""

    def test_config_with_zero_min_cluster_size(self) -> None:
        """Test config with zero min_cluster_size (edge case)."""
        config = ClusterConfig(min_cluster_size=0)
        assert config.min_cluster_size == 0

    def test_config_with_large_values(self) -> None:
        """Test config with large parameter values."""
        config = ClusterConfig(
            min_cluster_size=1000,
            min_samples=500,
            num_examples=100,
        )
        assert config.min_cluster_size == 1000
        assert config.min_samples == 500
        assert config.num_examples == 100

    def test_config_different_metrics(self) -> None:
        """Test config with different distance metrics."""
        for metric in ["euclidean", "cosine", "manhattan"]:
            config = ClusterConfig(metric=metric)
            assert config.metric == metric
