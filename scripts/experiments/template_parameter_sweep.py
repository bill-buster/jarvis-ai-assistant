#!/usr/bin/env python3
"""
Template Mining Parameter Sweep

Tests multiple eps and min_samples values using cached embeddings.
Much faster than re-running the full mining pipeline each time.

Usage:
    # First run: generates and saves embeddings
    python scripts/experiments/template_parameter_sweep.py --generate-embeddings

    # Subsequent runs: use cached embeddings
    python scripts/experiments/template_parameter_sweep.py --sweep
"""

import argparse
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default paths
CACHE_DIR = Path("results/embedding_cache")
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.pkl"
MESSAGES_FILE = CACHE_DIR / "messages.pkl"


def load_messages(db_path: Path, sample_size: int | None = None) -> list[str]:
    """Load messages from iMessage database."""

    logger.info("Loading messages from: %s", db_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    query = """
        SELECT text
        FROM message
        WHERE is_from_me = 0
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


def generate_embeddings(
    messages: list[str], model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> np.ndarray:
    """Generate embeddings for messages."""

    logger.info("Loading sentence transformer: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Generating embeddings for %d messages...", len(messages))
    start = time.time()

    embeddings = model.encode(
        messages, show_progress_bar=True, batch_size=32, convert_to_numpy=True
    )

    elapsed = time.time() - start
    logger.info("Generated embeddings in %.1fs", elapsed)

    return embeddings


def save_cache(messages: list[str], embeddings: np.ndarray):
    """Save messages and embeddings to disk."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving embeddings to %s", EMBEDDINGS_FILE)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    logger.info("Saving messages to %s", MESSAGES_FILE)
    with open(MESSAGES_FILE, "wb") as f:
        pickle.dump(messages, f)

    logger.info("Cache saved successfully")


def load_cache() -> tuple[list[str], np.ndarray]:
    """Load cached messages and embeddings."""

    if not EMBEDDINGS_FILE.exists() or not MESSAGES_FILE.exists():
        raise FileNotFoundError("Cache not found. Run with --generate-embeddings first.")

    logger.info("Loading embeddings from %s", EMBEDDINGS_FILE)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    logger.info("Loading messages from %s", MESSAGES_FILE)
    with open(MESSAGES_FILE, "rb") as f:
        messages = pickle.load(f)

    logger.info("Loaded %d messages and embeddings", len(messages))
    return messages, embeddings


def cluster_and_extract(
    messages: list[str],
    embeddings: np.ndarray,
    eps: float,
    min_samples: int,
    min_frequency: int = 8,
) -> dict:
    """Cluster embeddings and extract templates."""

    logger.info("Clustering with DBSCAN (eps=%.2f, min_samples=%d)", eps, min_samples)
    start = time.time()

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = clusterer.fit_predict(embeddings)

    cluster_time = time.time() - start

    # Count clusters and noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    logger.info("Found %d clusters, %d noise points in %.1fs", n_clusters, n_noise, cluster_time)

    # Extract templates from clusters
    templates = []
    covered_messages = 0

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_messages = [msg for msg, is_in in zip(messages, cluster_mask) if is_in]

        if len(cluster_messages) < min_frequency:
            continue

        # Use most common message as representative
        from collections import Counter

        msg_counts = Counter(cluster_messages)
        representative, frequency = msg_counts.most_common(1)[0]

        templates.append(
            {
                "representative": representative,
                "frequency": frequency,
                "cluster_size": len(cluster_messages),
            }
        )

        covered_messages += len(cluster_messages)

    # Sort by frequency
    templates.sort(key=lambda t: t["frequency"], reverse=True)

    coverage = covered_messages / len(messages) if messages else 0

    return {
        "eps": eps,
        "min_samples": min_samples,
        "min_frequency": min_frequency,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "templates_extracted": len(templates),
        "coverage": coverage,
        "largest_cluster": templates[0]["cluster_size"] if templates else 0,
        "clustering_time_s": cluster_time,
        "top_10_templates": templates[:10],
    }


def parameter_sweep(messages: list[str], embeddings: np.ndarray) -> list[dict]:
    """Test multiple parameter combinations."""

    # Parameter grid
    eps_values = [0.25, 0.30, 0.35, 0.40, 0.45]
    min_samples_values = [3, 5, 8, 10]

    results = []
    total_combos = len(eps_values) * len(min_samples_values)

    logger.info("Testing %d parameter combinations...", total_combos)

    for eps in eps_values:
        for min_samples in min_samples_values:
            result = cluster_and_extract(messages, embeddings, eps, min_samples)
            results.append(result)

            logger.info(
                "  eps=%.2f, min_samples=%d: %d templates, %.1f%% coverage",
                eps,
                min_samples,
                result["templates_extracted"],
                result["coverage"] * 100,
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Template mining parameter sweep")
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate and cache embeddings (slow, run once)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run parameter sweep using cached embeddings (fast)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of messages to sample (default: all received messages)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/parameter_sweep_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    if not (args.generate_embeddings or args.sweep):
        parser.print_help()
        return

    # Database path
    db_path = Path.home() / "Library" / "Messages" / "chat.db"

    if args.generate_embeddings:
        # Generate and cache embeddings
        messages = load_messages(db_path, args.sample_size)
        embeddings = generate_embeddings(messages)
        save_cache(messages, embeddings)
        logger.info("✓ Embeddings cached. Now run with --sweep to test parameters.")

    if args.sweep:
        # Load cached embeddings and run sweep
        messages, embeddings = load_cache()
        results = parameter_sweep(messages, embeddings)

        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_messages": len(messages),
                    "results": results,
                },
                f,
                indent=2,
            )

        logger.info("✓ Results saved to: %s", output_file)

        # Show best result
        best = max(results, key=lambda r: r["coverage"])
        logger.info("")
        logger.info("Best result:")
        logger.info("  eps=%.2f, min_samples=%d", best["eps"], best["min_samples"])
        logger.info("  Templates: %d", best["templates_extracted"])
        logger.info("  Coverage: %.1f%%", best["coverage"] * 100)


if __name__ == "__main__":
    main()
