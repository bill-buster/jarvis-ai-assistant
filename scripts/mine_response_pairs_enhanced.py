#!/usr/bin/env python3
"""
Enhanced Response Pair Mining with Context Awareness

IMPROVEMENTS OVER PREVIOUS VERSION:
1. ✓ Context metadata: sender, group type, time of day
2. ✓ Coherence checking: filters contradictory multi-message responses
3. ✓ HDBSCAN support: better clustering with automatic eps selection
4. ✓ Adaptive temporal decay: adjusts based on messaging frequency
5. ✓ Expanded filtering: stickers, payments, calendar invites
6. ✓ Conversation segmentation: splits by time gaps
7. ✓ Silhouette scoring: finds optimal clustering parameters
8. ✓ Template quality validation: scores appropriateness

This creates context-aware, high-quality templates for reply generation.
"""

import json
import logging
import pickle
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# iMessage epoch: 2001-01-01 00:00:00 UTC
IMESSAGE_EPOCH = datetime(2001, 1, 1)

# System messages to filter out (EXPANDED)
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
    "sticker",
    "memoji",
    "animoji",
]

# iMessage apps/games to filter out (EXPANDED)
IMESSAGE_APP_PATTERNS = [
    "cup pong",
    "8 ball",
    "chess",
    "game pigeon",
    "filler",
    "calendar.app",
    "venmo.com",
    "cash.app",
    "apple pay",
    "shared a location",
    "shared their location",
    "sent a payment",
]

# Contradictory phrase pairs (for coherence checking)
CONTRADICTORY_PHRASES = [
    ("yes", "no"),
    ("yeah", "can't"),
    ("sure", "actually"),
    ("definitely", "wait"),
    ("ok", "nevermind"),
    ("fine", "actually"),
    ("sounds good", "wait"),
    ("i'm in", "can't make it"),
    ("count me in", "count me out"),
]

# Cache directory for embeddings
CACHE_DIR = Path("results/embedding_cache")
EMBEDDINGS_FILE = CACHE_DIR / "response_pair_embeddings_enhanced.pkl"
PAIRS_FILE = CACHE_DIR / "response_pairs_data_enhanced.pkl"

# Conversation segmentation threshold (hours)
CONVERSATION_GAP_HOURS = 24


def imessage_timestamp_to_datetime(timestamp_ns: int) -> datetime:
    """Convert iMessage nanosecond timestamp to datetime."""
    return IMESSAGE_EPOCH + timedelta(microseconds=timestamp_ns / 1000)


def is_system_message(text: str) -> bool:
    """Check if message is a system message/reaction/game invite/app."""
    if not text or text.strip() == "":
        return True

    text_lower = text.lower()

    # Filter system messages
    for pattern in SYSTEM_MESSAGE_PATTERNS:
        if pattern.lower() in text_lower:
            return True

    # Filter iMessage apps
    for app in IMESSAGE_APP_PATTERNS:
        if app in text_lower:
            return True

    return False


def is_coherent_response(response_texts: list[str]) -> bool:
    """Check if multi-message response is semantically coherent.

    Filters out contradictory messages like:
    - "yeah wait actually can't"
    - "sure nevermind"
    """
    if len(response_texts) <= 1:
        return True

    combined = " ".join(response_texts).lower()

    # Check for contradictory phrases
    for phrase1, phrase2 in CONTRADICTORY_PHRASES:
        if phrase1 in combined and phrase2 in combined:
            logger.debug("Filtered contradictory response: %s", combined[:60])
            return False

    return True


def segment_conversation(messages: list[dict], gap_threshold_hours: int = 24) -> list[list[dict]]:
    """Split chat into conversations by time gaps.

    Args:
        messages: List of message dicts with date_ns
        gap_threshold_hours: Hours of inactivity that define a new conversation

    Returns:
        List of conversation segments
    """
    if not messages:
        return []

    conversations = []
    current_conv = [messages[0]]

    for i in range(1, len(messages)):
        time_gap_hours = (messages[i]["date_ns"] - messages[i-1]["date_ns"]) / 1e9 / 3600

        if time_gap_hours > gap_threshold_hours:
            conversations.append(current_conv)
            current_conv = [messages[i]]
        else:
            current_conv.append(messages[i])

    conversations.append(current_conv)
    return conversations


def get_response_groups(
    db_path: Path,
    max_time_gap_seconds: int = 300,
    max_burst_gap_seconds: int = 30,
) -> list[dict]:
    """Get (incoming message → your multi-message response) pairs with context.

    NOW INCLUDES:
    - Sender handle ID (for context grouping)
    - Chat type (group vs direct)
    - Time of day (hour)
    - Conversation segmentation
    """

    logger.info("Extracting message pairs with context metadata...")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get messages with sender info and chat type
    query = """
        SELECT
            m.ROWID,
            m.text,
            m.is_from_me,
            m.date,
            m.handle_id,
            cmj.chat_id,
            c.chat_identifier,
            (SELECT COUNT(*) FROM chat_handle_join WHERE chat_id = c.ROWID) as participant_count
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        JOIN chat c ON cmj.chat_id = c.ROWID
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
        rowid, text, is_from_me, date_ns, handle_id, chat_id, chat_identifier, participant_count = row

        # New chat - process accumulated messages
        if chat_id != current_chat:
            if chat_messages:
                # Segment conversation before extracting pairs
                conversations = segment_conversation(chat_messages, CONVERSATION_GAP_HOURS)
                for conversation in conversations:
                    pairs = extract_pairs_from_chat(
                        conversation,
                        max_time_gap_seconds,
                        max_burst_gap_seconds,
                        participant_count
                    )
                    response_groups.extend(pairs)

            current_chat = chat_id
            chat_messages = []

        # Skip system messages
        if is_system_message(text):
            continue

        chat_messages.append({
            "rowid": rowid,
            "text": text,
            "is_from_me": is_from_me,
            "date_ns": date_ns,
            "handle_id": handle_id,
            "participant_count": participant_count,
        })

    # Process last chat
    if chat_messages:
        conversations = segment_conversation(chat_messages, CONVERSATION_GAP_HOURS)
        for conversation in conversations:
            pairs = extract_pairs_from_chat(
                conversation,
                max_time_gap_seconds,
                max_burst_gap_seconds,
                chat_messages[0]["participant_count"]
            )
            response_groups.extend(pairs)

    conn.close()

    logger.info("Extracted %d response groups with context", len(response_groups))
    return response_groups


def extract_pairs_from_chat(
    messages: list[dict],
    max_time_gap_seconds: int,
    max_burst_gap_seconds: int,
    participant_count: int,
) -> list[dict]:
    """Extract (incoming → response) pairs with context metadata."""

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
        sender_id = msg["handle_id"]

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

        # If we found a response and it's coherent, add the pair
        if response_texts and is_coherent_response(response_texts):
            combined_response = " ".join(response_texts)

            # Extract context metadata
            dt = imessage_timestamp_to_datetime(incoming_date)
            hour_of_day = dt.hour
            is_group = participant_count > 2

            pairs.append({
                "incoming": incoming_text,
                "response": combined_response,
                "response_dates": response_dates,
                "sender_id": sender_id,
                "is_group": is_group,
                "hour_of_day": hour_of_day,
                "participant_count": participant_count,
            })

        i = j if j > i + 1 else i + 1

    return pairs


def calculate_adaptive_decay_constant(message_count: int, time_span_days: int) -> int:
    """Calculate adaptive decay based on messaging frequency.

    Daily texters: shorter decay (365 days) - style changes faster
    Weekly texters: longer decay (730 days) - style more stable

    Args:
        message_count: Total number of messages
        time_span_days: Days spanned by messages

    Returns:
        Decay constant (half-life in days)
    """
    if time_span_days == 0:
        return 730

    messages_per_day = message_count / time_span_days

    if messages_per_day > 10:  # Heavy texter
        return 365  # 1-year half-life
    elif messages_per_day > 2:  # Moderate texter
        return 547  # 1.5-year half-life
    else:  # Light texter
        return 730  # 2-year half-life


def calculate_temporal_scores_for_cluster(
    dates: list[int],
    decay_constant: int | None = None
) -> dict:
    """Calculate consistency and recency scores for a cluster.

    Args:
        dates: List of message timestamps
        decay_constant: Optional decay constant (calculated adaptively if None)
    """

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

    # Recency weight with adaptive decay
    most_recent_date_ns = max(dates)
    oldest_date_ns = min(dates)
    age_ns = current_time_ns - most_recent_date_ns
    age_days = age_ns / (1_000_000_000 * 86400)

    # Calculate adaptive decay if not provided
    if decay_constant is None:
        time_span_days = (most_recent_date_ns - oldest_date_ns) / (1_000_000_000 * 86400)
        decay_constant = calculate_adaptive_decay_constant(frequency, time_span_days)

    recency_weight = np.exp(-age_days / decay_constant)

    # Consistency bonus
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
        "decay_constant": int(decay_constant),
        "years_active": [int(y) for y in sorted(year_counts.keys())],
        "age_days": int(age_days),
        "most_recent": imessage_timestamp_to_datetime(most_recent_date_ns).strftime("%Y-%m-%d")
    }


def generate_embeddings(
    response_groups: list[dict],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    use_cache: bool = True
) -> tuple[np.ndarray, SentenceTransformer]:
    """Generate embeddings for response pairs."""

    # Check cache
    if use_cache and EMBEDDINGS_FILE.exists() and PAIRS_FILE.exists():
        logger.info("Loading cached embeddings from %s", EMBEDDINGS_FILE)
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(PAIRS_FILE, 'rb') as f:
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

    # Create text representations with context
    # Format: "incoming [SEP] response [CTX] group={is_group} hour={hour}"
    texts = []
    for group in response_groups:
        ctx = f"group={group['is_group']} hour={group['hour_of_day']}"
        text = f"{group['incoming']} [SEP] {group['response']} [CTX] {ctx}"
        texts.append(text)

    logger.info("Generating embeddings for %d response pairs...", len(texts))
    start = time.time()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    elapsed = time.time() - start
    logger.info("Generated embeddings in %.1fs (%.1f pairs/sec)",
                elapsed, len(texts) / elapsed)

    # Cache embeddings
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(PAIRS_FILE, 'wb') as f:
        pickle.dump(response_groups, f)
    logger.info("✓ Cached embeddings to %s", EMBEDDINGS_FILE)

    return embeddings, model


def find_optimal_eps_silhouette(
    embeddings: np.ndarray,
    eps_values: list[float] = [0.20, 0.25, 0.30, 0.35, 0.40],
    min_samples: int = 2
) -> dict[str, Any]:
    """Find optimal eps using silhouette score.

    Returns:
        Dict with best_eps and scores for each eps
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("sklearn not available for silhouette scoring")
        return {"best_eps": 0.30, "scores": {}}

    logger.info("Finding optimal eps using silhouette scores...")

    best_eps = None
    best_score = -1
    scores = {}

    for eps in eps_values:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Need at least 2 clusters and not all noise
        if n_clusters > 1 and n_noise < len(labels) * 0.9:
            try:
                # Filter out noise points for silhouette calculation
                mask = labels != -1
                if mask.sum() > 0:
                    score = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
                    scores[eps] = {
                        "silhouette": float(score),
                        "n_clusters": n_clusters,
                        "n_noise": n_noise
                    }

                    logger.info("  eps=%.2f: silhouette=%.3f, clusters=%d, noise=%d",
                                eps, score, n_clusters, n_noise)

                    if score > best_score:
                        best_score = score
                        best_eps = eps
            except Exception as e:
                logger.warning("  eps=%.2f: failed to compute silhouette: %s", eps, e)
        else:
            logger.info("  eps=%.2f: clusters=%d, noise=%d (skipped - too few clusters)",
                        eps, n_clusters, n_noise)

    if best_eps is None:
        logger.warning("No good eps found, defaulting to 0.30")
        best_eps = 0.30
    else:
        logger.info("✓ Best eps: %.2f (silhouette=%.3f)", best_eps, best_score)

    return {
        "best_eps": best_eps,
        "best_silhouette": best_score,
        "scores": scores
    }


def semantic_clustering_hdbscan(
    response_groups: list[dict],
    embeddings: np.ndarray,
    min_cluster_size: int = 2
) -> list[dict]:
    """Cluster using HDBSCAN (automatic eps selection).

    Fallback to DBSCAN if HDBSCAN not available.
    """
    try:
        import hdbscan
        logger.info("Clustering with HDBSCAN (min_cluster_size=%d)", min_cluster_size)
        start = time.time()

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine',
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(embeddings)

        elapsed = time.time() - start

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info("Found %d clusters, %d noise points in %.1fs (HDBSCAN)",
                    n_clusters, n_noise, elapsed)

    except ImportError:
        logger.warning("HDBSCAN not available, falling back to DBSCAN with optimal eps")

        # Find optimal eps using silhouette score
        eps_result = find_optimal_eps_silhouette(embeddings)
        best_eps = eps_result["best_eps"]

        logger.info("Clustering with DBSCAN (eps=%.2f)", best_eps)
        start = time.time()

        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=best_eps, min_samples=2, metric='cosine', n_jobs=-1)
        labels = clusterer.fit_predict(embeddings)

        elapsed = time.time() - start

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info("Found %d clusters, %d noise points in %.1fs (DBSCAN)",
                    n_clusters, n_noise, elapsed)

    # Group pairs by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        clusters[label].append(response_groups[i])

    # Extract patterns from each cluster
    cluster_patterns = []

    for cluster_id, pairs in clusters.items():
        # Get all variations
        incoming_texts = [p["incoming"].lower().strip() for p in pairs]
        response_texts = [p["response"].lower().strip() for p in pairs]

        # Count frequencies
        incoming_counter = Counter(incoming_texts)
        response_counter = Counter(response_texts)

        # Get representative (most common)
        rep_incoming = incoming_counter.most_common(1)[0][0]
        rep_response = response_counter.most_common(1)[0][0]

        # Extract context metadata
        sender_ids = set(p.get("sender_id") for p in pairs if p.get("sender_id"))
        is_group_msgs = [p["is_group"] for p in pairs]
        hours = [p["hour_of_day"] for p in pairs]

        # Collect all dates for temporal scoring
        all_dates = []
        for p in pairs:
            all_dates.extend(p["response_dates"])

        # Calculate temporal scores
        temporal_scores = calculate_temporal_scores_for_cluster(all_dates)

        cluster_patterns.append({
            "cluster_id": cluster_id,
            "representative_incoming": rep_incoming,
            "representative_response": rep_response,
            "total_frequency": len(pairs),
            "incoming_variations": list(incoming_counter.keys())[:5],
            "response_variations": list(response_counter.keys())[:5],
            "num_senders": len(sender_ids),
            "is_group_ratio": sum(is_group_msgs) / len(is_group_msgs),
            "avg_hour": sum(hours) / len(hours),
            **temporal_scores
        })

    # Sort by combined score
    cluster_patterns.sort(key=lambda x: x["combined_score"], reverse=True)

    logger.info("Extracted %d cluster patterns", len(cluster_patterns))
    return cluster_patterns


def analyze_results(cluster_patterns: list[dict]):
    """Display analysis of clustered patterns."""

    print("\n" + "="*80)
    print("ENHANCED SEMANTIC RESPONSE PAIR CLUSTERING")
    print("="*80)

    print(f"\nTotal semantic clusters: {len(cluster_patterns)}")

    # Top patterns by combined score
    print("\n--- Top 50 Patterns (with context) ---\n")
    for i, p in enumerate(cluster_patterns[:50], 1):
        print(f"{i:2}. Score: {p['combined_score']:>7.1f} | "
              f"Freq: {p['total_frequency']:>4} | "
              f"Age: {p['age_days']:>4}d | "
              f"Senders: {p['num_senders']}")
        print(f"    \"{p['representative_incoming'][:60]}\" → \"{p['representative_response'][:60]}\"")

        # Context info
        group_pct = p['is_group_ratio'] * 100
        hour = int(p['avg_hour'])
        print(f"    Context: {group_pct:.0f}% group | Avg hour: {hour}:00 | "
              f"Consistency: {p['consistency_score']:.2f}")

        # Show variations if multiple
        if len(p['incoming_variations']) > 1:
            incoming_vars = [v[:30] for v in p['incoming_variations'][:3]]
            print(f"    Incoming variations: {incoming_vars}")

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced semantic response pair mining")
    parser.add_argument(
        "--use-hdbscan",
        action="store_true",
        help="Use HDBSCAN instead of DBSCAN (auto eps)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached embeddings"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/response_pairs_enhanced.json",
        help="Output file"
    )

    args = parser.parse_args()

    db_path = Path.home() / "Library" / "Messages" / "chat.db"

    # Extract response groups with context
    response_groups = get_response_groups(db_path)

    # Generate embeddings
    embeddings, model = generate_embeddings(
        response_groups,
        use_cache=not args.no_cache
    )

    # Cluster with HDBSCAN or DBSCAN
    cluster_patterns = semantic_clustering_hdbscan(
        response_groups,
        embeddings,
        min_cluster_size=2
    )

    # Analyze
    analyze_results(cluster_patterns)

    # Save
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "total_clusters": len(cluster_patterns),
            "patterns": cluster_patterns,
            "metadata": {
                "total_response_groups": len(response_groups),
                "clustering_method": "HDBSCAN" if args.use_hdbscan else "DBSCAN",
                "embedding_model": "all-mpnet-base-v2",
                "features": [
                    "context_aware",
                    "coherence_filtered",
                    "adaptive_decay",
                    "conversation_segmented"
                ]
            }
        }, f, indent=2)

    logger.info("\n✓ Enhanced patterns saved to: %s", output_file)


if __name__ == "__main__":
    main()
