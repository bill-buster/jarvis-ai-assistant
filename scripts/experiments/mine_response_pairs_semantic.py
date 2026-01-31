#!/usr/bin/env python3
"""
Semantic Response Pair Mining with Clustering

Adds semantic clustering on top of optimized temporal scoring to:
1. Group similar patterns together (e.g., "yeah" → "bet" and "yea" → "bet")
2. Find many more patterns by clustering variations
3. Extract representative patterns from each semantic cluster
4. Score clusters by combined frequency and temporal factors

This creates high-quality, semantically-aware templates for reply generation.
"""

import json
import logging
import pickle
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# iMessage epoch: 2001-01-01 00:00:00 UTC
IMESSAGE_EPOCH = datetime(2001, 1, 1)

# System messages and game invites to filter out
SYSTEM_MESSAGE_PATTERNS = [
    "Loved ",
    "Laughed at ",
    "Emphasized ",
    "Questioned ",
    "Liked ",
    "Disliked ",
    "Loved an image",
    "Laughed at an image",
    "￼",  # Attachment placeholder
]

# iMessage game/app invitations (not conversational)
IMESSAGE_GAME_PATTERNS = [
    "cup pong",
    "8 ball",
    "chess",
    "game pigeon",
    "filler",  # Test/filler data
]

# Cache directory for embeddings
CACHE_DIR = Path("results/embedding_cache")
EMBEDDINGS_FILE = CACHE_DIR / "response_pair_embeddings.pkl"
PAIRS_FILE = CACHE_DIR / "response_pairs_data.pkl"


def imessage_timestamp_to_datetime(timestamp_ns: int) -> datetime:
    """Convert iMessage nanosecond timestamp to datetime."""
    return IMESSAGE_EPOCH + timedelta(microseconds=timestamp_ns / 1000)


def is_system_message(text: str) -> bool:
    """Check if message is a system message/reaction/game invite."""
    if not text or text.strip() == "":
        return True

    text_lower = text.lower()

    # Filter system messages
    for pattern in SYSTEM_MESSAGE_PATTERNS:
        if pattern in text:
            return True

    # Filter iMessage games
    for game in IMESSAGE_GAME_PATTERNS:
        if game in text_lower:
            return True

    return False


def get_response_groups(
    db_path: Path,
    max_time_gap_seconds: int = 300,
    max_burst_gap_seconds: int = 30,
) -> list[dict]:
    """Get (incoming message → your multi-message response) pairs.

    Handles cases where you send multiple short messages in a row.
    """

    logger.info("Extracting message pairs with multi-message grouping...")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all messages in chronological order
    query = """
        SELECT
            m.ROWID,
            m.text,
            m.is_from_me,
            m.date,
            cmj.chat_id
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE m.text IS NOT NULL
          AND m.text != ''
          AND length(m.text) > 0
        ORDER BY cmj.chat_id, m.date
    """

    cursor.execute(query)

    # Group messages by chat and build response pairs
    current_chat = None
    chat_messages = []
    response_groups = []

    for row in cursor.fetchall():
        rowid, text, is_from_me, date_ns, chat_id = row

        # New chat - process accumulated messages
        if chat_id != current_chat:
            if chat_messages:
                pairs = extract_pairs_from_chat(
                    chat_messages, max_time_gap_seconds, max_burst_gap_seconds
                )
                response_groups.extend(pairs)

            current_chat = chat_id
            chat_messages = []

        # Skip system messages
        if is_system_message(text):
            continue

        chat_messages.append(
            {
                "rowid": rowid,
                "text": text,
                "is_from_me": is_from_me,
                "date_ns": date_ns,
            }
        )

    # Process last chat
    if chat_messages:
        pairs = extract_pairs_from_chat(chat_messages, max_time_gap_seconds, max_burst_gap_seconds)
        response_groups.extend(pairs)

    conn.close()

    logger.info("Extracted %d response groups", len(response_groups))
    return response_groups


def extract_pairs_from_chat(
    messages: list[dict],
    max_time_gap_seconds: int,
    max_burst_gap_seconds: int,
) -> list[dict]:
    """Extract (incoming → response) pairs from a single chat."""

    pairs = []
    time_gap_ns = max_time_gap_seconds * 1_000_000_000
    burst_gap_ns = max_burst_gap_seconds * 1_000_000_000

    i = 0
    while i < len(messages) - 1:
        msg = messages[i]

        # Skip if this is your message
        if msg["is_from_me"] == 1:
            i += 1
            continue

        # This is an incoming message - look for your response(s)
        incoming_text = msg["text"]
        incoming_date = msg["date_ns"]

        # Collect your consecutive response messages
        response_texts = []
        response_dates = []

        j = i + 1
        while j < len(messages):
            next_msg = messages[j]

            # Not your message - stop
            if next_msg["is_from_me"] == 0:
                break

            # Too much time passed since incoming - stop
            if next_msg["date_ns"] - incoming_date > time_gap_ns:
                break

            # Check if this is part of a burst
            if response_dates and next_msg["date_ns"] - response_dates[-1] > burst_gap_ns:
                break

            response_texts.append(next_msg["text"])
            response_dates.append(next_msg["date_ns"])
            j += 1

        # If we found a response, add the pair
        if response_texts:
            combined_response = " ".join(response_texts)

            pairs.append(
                {
                    "incoming": incoming_text,
                    "response": combined_response,
                    "response_dates": response_dates,
                }
            )

        i = j if j > i + 1 else i + 1

    return pairs


def generate_embeddings(
    response_groups: list[dict],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    use_cache: bool = True,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Generate embeddings for response pairs.

    Embeds each pair as: "incoming [SEP] response"
    Caches embeddings to disk for faster re-runs.
    """

    # Check cache
    if use_cache and EMBEDDINGS_FILE.exists() and PAIRS_FILE.exists():
        logger.info("Loading cached embeddings from %s", EMBEDDINGS_FILE)
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
        with open(PAIRS_FILE, "rb") as f:
            cached_groups = pickle.load(f)

        # Verify cache matches
        if len(cached_groups) == len(response_groups):
            logger.info("✓ Using cached embeddings for %d pairs", len(embeddings))
            model = SentenceTransformer(model_name)
            return embeddings, model
        else:
            logger.warning("Cache size mismatch, regenerating embeddings")

    # Generate embeddings
    logger.info("Loading sentence transformer: %s", model_name)
    model = SentenceTransformer(model_name)

    # Create text representations for embedding
    # Format: "incoming [SEP] response"
    texts = []
    for group in response_groups:
        text = f"{group['incoming']} [SEP] {group['response']}"
        texts.append(text)

    logger.info("Generating embeddings for %d response pairs...", len(texts))
    start = time.time()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )

    elapsed = time.time() - start
    logger.info("Generated embeddings in %.1fs (%.1f pairs/sec)", elapsed, len(texts) / elapsed)

    # Cache embeddings
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    with open(PAIRS_FILE, "wb") as f:
        pickle.dump(response_groups, f)
    logger.info("✓ Cached embeddings to %s", EMBEDDINGS_FILE)

    return embeddings, model


def semantic_clustering(
    response_groups: list[dict], embeddings: np.ndarray, eps: float = 0.30, min_samples: int = 2
) -> list[dict]:
    """Cluster semantically similar response pairs.

    Args:
        response_groups: List of (incoming, response, dates) dicts
        embeddings: Normalized embeddings for each pair
        eps: DBSCAN epsilon (similarity threshold)
        min_samples: Minimum cluster size

    Returns:
        List of cluster dicts with patterns and scores
    """

    logger.info("Clustering with DBSCAN (eps=%.2f, min_samples=%d)", eps, min_samples)
    start = time.time()

    # Use cosine distance (1 - cosine similarity)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = clusterer.fit_predict(embeddings)

    elapsed = time.time() - start

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    logger.info("Found %d clusters, %d noise points in %.1fs", n_clusters, n_noise, elapsed)

    # Group pairs by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        clusters[label].append(response_groups[i])

    # Extract patterns from each cluster
    cluster_patterns = []

    for cluster_id, pairs in clusters.items():
        # Get all variations in this cluster
        incoming_texts = [p["incoming"].lower().strip() for p in pairs]
        response_texts = [p["response"].lower().strip() for p in pairs]

        # Count frequencies
        incoming_counter = Counter(incoming_texts)
        response_counter = Counter(response_texts)

        # Get representative (most common)
        rep_incoming = incoming_counter.most_common(1)[0][0]
        rep_response = response_counter.most_common(1)[0][0]

        # Collect all dates for temporal scoring
        all_dates = []
        for p in pairs:
            all_dates.extend(p["response_dates"])

        # Calculate temporal scores
        temporal_scores = calculate_temporal_scores_for_cluster(all_dates)

        cluster_patterns.append(
            {
                "cluster_id": cluster_id,
                "representative_incoming": rep_incoming,
                "representative_response": rep_response,
                "total_frequency": len(pairs),
                "incoming_variations": list(incoming_counter.keys())[:5],  # Top 5
                "response_variations": list(response_counter.keys())[:5],  # Top 5
                **temporal_scores,
            }
        )

    # Sort by combined score
    cluster_patterns.sort(key=lambda x: x["combined_score"], reverse=True)

    logger.info("Extracted %d cluster patterns", len(cluster_patterns))
    return cluster_patterns


def calculate_temporal_scores_for_cluster(dates: list[int]) -> dict:
    """Calculate consistency and recency scores for a cluster."""

    current_time_ns = int((datetime.now() - IMESSAGE_EPOCH).total_seconds() * 1_000_000_000)

    frequency = len(dates)

    # Get years
    years = [imessage_timestamp_to_datetime(d).year for d in dates]
    year_counts = Counter(years)

    # Consistency score
    if len(year_counts) == 1:
        consistency_score = 0.5
    else:
        mean_per_year = np.mean(list(year_counts.values()))
        std_per_year = np.std(list(year_counts.values()))
        cv = std_per_year / mean_per_year if mean_per_year > 0 else 1.0
        consistency_score = 1.0 / (1.0 + cv)

    # Recency weight (most recent occurrence)
    most_recent_date_ns = max(dates)
    age_ns = current_time_ns - most_recent_date_ns
    age_days = age_ns / (1_000_000_000 * 86400)

    decay_constant = 730  # 2 year half-life
    recency_weight = np.exp(-age_days / decay_constant)

    # Consistency bonus (patterns spanning many years)
    year_span = max(years) - min(years) + 1
    consistency_bonus = 1.0 + (len(year_counts) / max(3, year_span))

    # Combined score
    combined_score = frequency * consistency_score * recency_weight * consistency_bonus

    return {
        "frequency": int(frequency),
        "consistency_score": float(consistency_score),
        "recency_weight": float(recency_weight),
        "consistency_bonus": float(consistency_bonus),
        "combined_score": float(combined_score),
        "years_active": [int(y) for y in sorted(year_counts.keys())],
        "age_days": int(age_days),
        "most_recent": imessage_timestamp_to_datetime(most_recent_date_ns).strftime("%Y-%m-%d"),
    }


def test_multiple_eps(
    response_groups: list[dict],
    embeddings: np.ndarray,
    eps_values: list[float] = [0.20, 0.25, 0.30, 0.35, 0.40],
) -> dict:
    """Test multiple eps values to find optimal clustering."""

    logger.info("Testing %d different eps values...", len(eps_values))

    results = {}

    for eps in eps_values:
        clusters = semantic_clustering(response_groups, embeddings, eps=eps, min_samples=2)

        # Calculate metrics
        total_patterns = len(clusters)
        total_coverage = sum(c["frequency"] for c in clusters)
        avg_cluster_size = total_coverage / total_patterns if total_patterns > 0 else 0

        results[eps] = {
            "eps": eps,
            "num_clusters": total_patterns,
            "total_coverage": total_coverage,
            "avg_cluster_size": avg_cluster_size,
            "top_10_patterns": clusters[:10],
        }

        logger.info(
            "  eps=%.2f: %d clusters, %.1f avg size, %d total pairs covered",
            eps,
            total_patterns,
            avg_cluster_size,
            total_coverage,
        )

    return results


def analyze_results(cluster_patterns: list[dict]):
    """Display analysis of clustered patterns."""

    print("\n" + "=" * 80)
    print("SEMANTIC RESPONSE PAIR CLUSTERING RESULTS")
    print("=" * 80)

    print(f"\nTotal semantic clusters: {len(cluster_patterns)}")

    # Top patterns by combined score
    print("\n--- Top 50 Semantic Patterns (by score: freq × consistency × recency) ---\n")
    for i, p in enumerate(cluster_patterns[:50], 1):
        print(
            f"{i:2}. Score: {p['combined_score']:>7.1f} | "
            f"Freq: {p['frequency']:>4} | "
            f"Age: {p['age_days']:>4}d"
        )
        print(f'    "{p["representative_incoming"][:60]}" → "{p["representative_response"][:60]}"')

        # Show variations if multiple
        if len(p["incoming_variations"]) > 1:
            incoming_vars = [v[:30] for v in p["incoming_variations"][:3]]
            print(f"    Incoming variations: {incoming_vars}")
        if len(p["response_variations"]) > 1:
            response_vars = [v[:30] for v in p["response_variations"][:3]]
            print(f"    Response variations: {response_vars}")

        print(
            f"    Years: {p['years_active']} | "
            f"Consistency: {p['consistency_score']:.2f} | "
            f"Recency: {p['recency_weight']:.2f}"
        )
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic response pair mining")
    parser.add_argument(
        "--eps", type=float, default=0.30, help="DBSCAN eps parameter (default: 0.30)"
    )
    parser.add_argument(
        "--test-eps", action="store_true", help="Test multiple eps values to find optimal"
    )
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached embeddings")
    parser.add_argument(
        "--output", type=str, default="results/response_pairs_semantic.json", help="Output file"
    )

    args = parser.parse_args()

    db_path = Path.home() / "Library" / "Messages" / "chat.db"

    # Extract response groups
    response_groups = get_response_groups(db_path)

    # Generate embeddings
    embeddings, model = generate_embeddings(response_groups, use_cache=not args.no_cache)

    if args.test_eps:
        # Test multiple eps values
        results = test_multiple_eps(response_groups, embeddings)

        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("\n✓ Parameter sweep results saved to: %s", output_file)

    else:
        # Single eps value
        cluster_patterns = semantic_clustering(response_groups, embeddings, eps=args.eps)

        # Analyze
        analyze_results(cluster_patterns)

        # Save
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(
                {
                    "total_clusters": len(cluster_patterns),
                    "eps": args.eps,
                    "patterns": cluster_patterns,
                },
                f,
                indent=2,
            )

        logger.info("\n✓ Semantic patterns saved to: %s", output_file)


if __name__ == "__main__":
    main()
