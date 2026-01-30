#!/usr/bin/env python3
"""Template mining from iMessage conversations using clustering and semantic similarity.

This module discovers common message patterns by:
1. Loading messages from the iMessage database
2. Generating embeddings using sentence transformers
3. Clustering similar messages using DBSCAN
4. Extracting representative templates from clusters
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Conditional imports for sentence transformers
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """A discovered message template."""

    representative: str  # Most central example from cluster
    frequency: int  # Number of messages in cluster
    cluster_id: int  # Cluster identifier
    examples: list[str]  # Sample messages from cluster


@dataclass
class MiningStats:
    """Statistics from template mining run."""

    total_messages: int
    messages_sampled: int
    templates_extracted: int
    coverage: float  # Fraction of sampled messages covered by templates
    largest_cluster_size: int
    embedding_time_seconds: float
    clustering_time_seconds: float


@dataclass
class MiningResult:
    """Result from template mining."""

    templates: list[Template]
    stats: MiningStats
    config: dict[str, Any]


class TemplateMiner:
    """Discovers common message templates using clustering."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        eps: float = 0.35,
        min_samples: int = 5,
        min_frequency: int = 8,
        db_path: str | None = None,
    ):
        """Initialize template miner.

        Args:
            model_name: Sentence transformer model for embeddings
            eps: DBSCAN epsilon - max distance for neighbors (lower = tighter clusters)
            min_samples: DBSCAN min samples to form a core point
            min_frequency: Minimum cluster size to extract as template
            db_path: Path to iMessage database (uses default if None)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for template mining. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.eps = eps
        self.min_samples = min_samples
        self.min_frequency = min_frequency
        self.db_path = db_path or str(Path.home() / "Library/Messages/chat.db")

        logger.info("Initializing template miner with model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def load_messages(self, sample_size: int | None = None) -> list[str]:
        """Load messages from iMessage database.

        Args:
            sample_size: Maximum number of messages to load (None = all)

        Returns:
            List of message texts
        """
        logger.info("Loading messages from: %s", self.db_path)

        try:
            # Open read-only with URI
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()

            # Query sent messages (is_from_me=1) that are text (not attachments)
            query = """
                SELECT text
                FROM message
                WHERE is_from_me = 1
                  AND text IS NOT NULL
                  AND text != ''
                  AND length(text) > 0
                ORDER BY RANDOM()
            """

            if sample_size:
                query += f" LIMIT {sample_size}"

            cursor.execute(query)
            messages = [row[0] for row in cursor.fetchall()]
            conn.close()

            logger.info("Loaded %d messages", len(messages))
            return messages

        except sqlite3.Error as e:
            logger.error("Failed to load messages: %s", e)
            raise

    def generate_embeddings(self, messages: list[str]) -> np.ndarray:
        """Generate embeddings for messages.

        Args:
            messages: List of message texts

        Returns:
            Numpy array of embeddings (n_messages, embedding_dim)
        """
        logger.info("Generating embeddings for %d messages", len(messages))
        start = time.time()

        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            messages,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

        elapsed = time.time() - start
        logger.info("Generated embeddings in %.1fs", elapsed)

        return embeddings

    def cluster_messages(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]]]:
        """Cluster messages using DBSCAN.

        Args:
            embeddings: Message embeddings

        Returns:
            Tuple of (cluster_labels, cluster_to_indices)
        """
        logger.info("Clustering with DBSCAN (eps=%.2f, min_samples=%d)", self.eps, self.min_samples)
        start = time.time()

        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine",
            n_jobs=-1,
        )

        labels = clusterer.fit_predict(embeddings)
        elapsed = time.time() - start

        # Group indices by cluster
        cluster_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise (-1)
                if label not in cluster_to_indices:
                    cluster_to_indices[label] = []
                cluster_to_indices[label].append(idx)

        n_clusters = len(cluster_to_indices)
        n_noise = np.sum(labels == -1)

        logger.info(
            "Clustering complete in %.1fs: %d clusters, %d noise points",
            elapsed,
            n_clusters,
            n_noise,
        )

        return labels, cluster_to_indices

    def extract_templates(
        self,
        messages: list[str],
        embeddings: np.ndarray,
        cluster_to_indices: dict[int, list[int]],
    ) -> list[Template]:
        """Extract templates from clusters.

        Args:
            messages: Original message texts
            embeddings: Message embeddings
            cluster_to_indices: Mapping from cluster ID to message indices

        Returns:
            List of extracted templates
        """
        templates: list[Template] = []

        for cluster_id, indices in tqdm(
            cluster_to_indices.items(),
            desc="Extracting templates",
        ):
            # Filter by minimum frequency
            if len(indices) < self.min_frequency:
                continue

            # Get cluster messages and embeddings
            cluster_messages = [messages[i] for i in indices]
            cluster_embeddings = embeddings[indices]

            # Find most central message (closest to cluster centroid)
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            most_central_idx = np.argmin(distances)
            representative = cluster_messages[most_central_idx]

            # Sample examples (up to 5)
            example_indices = np.random.choice(
                len(cluster_messages),
                size=min(5, len(cluster_messages)),
                replace=False,
            )
            examples = [cluster_messages[i] for i in example_indices]

            templates.append(
                Template(
                    representative=representative,
                    frequency=len(indices),
                    cluster_id=cluster_id,
                    examples=examples,
                )
            )

        # Sort by frequency (descending)
        templates.sort(key=lambda t: t.frequency, reverse=True)

        logger.info("Extracted %d templates", len(templates))
        return templates

    def mine(self, sample_size: int = 100000) -> MiningResult:
        """Run template mining pipeline.

        Args:
            sample_size: Number of messages to sample

        Returns:
            MiningResult with templates and statistics
        """
        # Load messages
        messages = self.load_messages(sample_size)
        total_messages = len(messages)

        if total_messages == 0:
            logger.warning("No messages found")
            return MiningResult(
                templates=[],
                stats=MiningStats(
                    total_messages=0,
                    messages_sampled=0,
                    templates_extracted=0,
                    coverage=0.0,
                    largest_cluster_size=0,
                    embedding_time_seconds=0.0,
                    clustering_time_seconds=0.0,
                ),
                config={
                    "model_name": self.model_name,
                    "eps": self.eps,
                    "min_samples": self.min_samples,
                    "min_frequency": self.min_frequency,
                    "sample_size": sample_size,
                },
            )

        # Generate embeddings
        embed_start = time.time()
        embeddings = self.generate_embeddings(messages)
        embed_time = time.time() - embed_start

        # Cluster messages
        cluster_start = time.time()
        labels, cluster_to_indices = self.cluster_messages(embeddings)
        cluster_time = time.time() - cluster_start

        # Extract templates
        templates = self.extract_templates(messages, embeddings, cluster_to_indices)

        # Calculate statistics
        messages_in_clusters = sum(len(indices) for indices in cluster_to_indices.values())
        coverage = messages_in_clusters / total_messages if total_messages > 0 else 0.0
        largest_cluster = max((len(indices) for indices in cluster_to_indices.values()), default=0)

        stats = MiningStats(
            total_messages=int(total_messages),
            messages_sampled=int(total_messages),
            templates_extracted=int(len(templates)),
            coverage=float(coverage),
            largest_cluster_size=int(largest_cluster),
            embedding_time_seconds=float(embed_time),
            clustering_time_seconds=float(cluster_time),
        )

        return MiningResult(
            templates=templates,
            stats=stats,
            config={
                "model_name": self.model_name,
                "eps": self.eps,
                "min_samples": self.min_samples,
                "min_frequency": self.min_frequency,
                "sample_size": sample_size,
            },
        )


def main() -> int:
    """CLI entry point for template mining."""
    parser = argparse.ArgumentParser(description="Mine templates from iMessage conversations")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of messages to sample (default: 100000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence transformer model (default: all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.35,
        help="DBSCAN epsilon - max distance for neighbors (default: 0.35)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min samples to form cluster (default: 5)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=8,
        help="Minimum cluster size to extract as template (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to iMessage database (default: ~/Library/Messages/chat.db)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        # Create miner
        miner = TemplateMiner(
            model_name=args.model,
            eps=args.eps,
            min_samples=args.min_samples,
            min_frequency=args.min_frequency,
            db_path=args.db_path,
        )

        # Run mining
        result = miner.mine(sample_size=args.sample_size)

        # Convert to JSON-serializable format
        output_data = {
            "templates": [
                {
                    "representative": t.representative,
                    "frequency": int(t.frequency),  # Convert numpy int64 to Python int
                    "cluster_id": int(t.cluster_id),  # Convert numpy int64 to Python int
                    "examples": t.examples,
                }
                for t in result.templates
            ],
            "stats": asdict(result.stats),
            "config": result.config,
        }

        # Write output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("Results written to: %s", output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("TEMPLATE MINING COMPLETE")
        print("=" * 60)
        print(f"Messages analyzed:     {result.stats.total_messages:,}")
        print(f"Templates extracted:   {result.stats.templates_extracted}")
        print(f"Coverage:              {result.stats.coverage * 100:.1f}%")
        print(f"Largest cluster:       {result.stats.largest_cluster_size}")
        print(f"Embedding time:        {result.stats.embedding_time_seconds:.1f}s")
        print(f"Clustering time:       {result.stats.clustering_time_seconds:.1f}s")
        print("\nTop 10 templates:")
        for i, template in enumerate(result.templates[:10], 1):
            print(f"  {i}. [{template.frequency:3d} uses] {template.representative[:70]}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.exception("Template mining failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
