#!/usr/bin/env python3
"""Analyze contacts by style features (no embeddings needed).

Extracts quantifiable style metrics from conversations and clusters
contacts based on communication patterns.

Usage:
    python scripts/analyze_style_features.py
    python scripts/analyze_style_features.py --visualize
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/style_analysis")


@dataclass
class StyleFeatures:
    """Extracted style features for a contact."""
    name: str
    message_count: int

    # Length features
    my_avg_len: float
    their_avg_len: float
    my_len_variance: float

    # Punctuation
    my_no_punct_rate: float  # Messages with no ending punctuation
    my_exclaim_rate: float
    my_question_rate: float

    # Casual markers
    my_lol_rate: float
    my_emoji_rate: float
    my_abbreviation_rate: float  # u, ur, rn, tmrw, etc.

    # Capitalization
    my_lowercase_start_rate: float
    my_all_lowercase_rate: float

    # Response patterns
    my_short_response_rate: float  # < 10 chars
    my_one_word_rate: float

    # Common starters
    top_starters: list  # Most common first words

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for clustering."""
        return np.array([
            self.my_avg_len,
            self.their_avg_len,
            self.my_len_variance,
            self.my_no_punct_rate,
            self.my_exclaim_rate,
            self.my_question_rate,
            self.my_lol_rate,
            self.my_emoji_rate,
            self.my_abbreviation_rate,
            self.my_lowercase_start_rate,
            self.my_all_lowercase_rate,
            self.my_short_response_rate,
            self.my_one_word_rate,
        ])


def extract_style_features(name: str, my_messages: list[str], their_messages: list[str]) -> StyleFeatures:
    """Extract style features from messages."""

    if not my_messages:
        my_messages = [""]
    if not their_messages:
        their_messages = [""]

    # Length features
    my_lens = [len(m) for m in my_messages]
    their_lens = [len(m) for m in their_messages]

    my_avg_len = np.mean(my_lens)
    their_avg_len = np.mean(their_lens)
    my_len_variance = np.std(my_lens) if len(my_lens) > 1 else 0

    # Punctuation
    def ends_with(msgs, chars):
        return sum(1 for m in msgs if m.rstrip() and m.rstrip()[-1] in chars) / max(len(msgs), 1)

    my_no_punct_rate = 1 - ends_with(my_messages, ".!?")
    my_exclaim_rate = ends_with(my_messages, "!")
    my_question_rate = ends_with(my_messages, "?")

    # Casual markers
    def contains_rate(msgs, patterns):
        count = 0
        for m in msgs:
            m_lower = m.lower()
            if any(re.search(p, m_lower) for p in patterns):
                count += 1
        return count / max(len(msgs), 1)

    my_lol_rate = contains_rate(my_messages, [r'\blol\b', r'\blmao\b', r'\bhaha', r'\brotfl'])

    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
    my_emoji_rate = contains_rate(my_messages, [emoji_pattern])

    abbrev_patterns = [r'\bu\b', r'\bur\b', r'\brn\b', r'\btmrw\b', r'\bidk\b', r'\bnvm\b', r'\btbh\b', r'\bimo\b', r'\bwyd\b', r'\bhbu\b']
    my_abbreviation_rate = contains_rate(my_messages, abbrev_patterns)

    # Capitalization
    my_lowercase_start_rate = sum(1 for m in my_messages if m and m[0].islower()) / max(len(my_messages), 1)
    my_all_lowercase_rate = sum(1 for m in my_messages if m and m == m.lower()) / max(len(my_messages), 1)

    # Response patterns
    my_short_response_rate = sum(1 for m in my_messages if len(m) < 10) / max(len(my_messages), 1)
    my_one_word_rate = sum(1 for m in my_messages if len(m.split()) == 1) / max(len(my_messages), 1)

    # Common starters
    starters = Counter()
    for m in my_messages:
        words = m.split()
        if words:
            starters[words[0].lower()] += 1
    top_starters = [w for w, _ in starters.most_common(5)]

    return StyleFeatures(
        name=name,
        message_count=len(my_messages) + len(their_messages),
        my_avg_len=my_avg_len,
        their_avg_len=their_avg_len,
        my_len_variance=my_len_variance,
        my_no_punct_rate=my_no_punct_rate,
        my_exclaim_rate=my_exclaim_rate,
        my_question_rate=my_question_rate,
        my_lol_rate=my_lol_rate,
        my_emoji_rate=my_emoji_rate,
        my_abbreviation_rate=my_abbreviation_rate,
        my_lowercase_start_rate=my_lowercase_start_rate,
        my_all_lowercase_rate=my_all_lowercase_rate,
        my_short_response_rate=my_short_response_rate,
        my_one_word_rate=my_one_word_rate,
        top_starters=top_starters,
    )


def load_contacts(min_messages: int = 20) -> list[tuple[str, list[str], list[str]]]:
    """Load contacts with their messages."""
    from core.imessage.reader import MessageReader

    reader = MessageReader()
    print("Loading conversations...")
    conversations = reader.get_conversations(limit=500)

    spam_keywords = ["verification code", "click here", "unsubscribe", "your order"]

    contacts = []
    seen = set()

    for conv in conversations:
        name = conv.display_name or ""
        participants = conv.participants or []
        contact_key = name or (participants[0] if participants else "")

        if contact_key in seen or not contact_key:
            continue

        # Skip short codes
        if contact_key.isdigit() and 5 <= len(contact_key) <= 6:
            continue

        try:
            messages = reader.get_messages(conv.chat_id, limit=300)
            if not messages or len(messages) < min_messages:
                continue

            # Check spam
            sample = " ".join((m.text or "").lower() for m in messages[:20])
            if sum(1 for kw in spam_keywords if kw in sample) >= 2:
                continue

            my_msgs = []
            their_msgs = []

            for m in messages:
                text = (m.text or "").strip()
                if not text or len(text) < 1:
                    continue
                # Skip reactions
                if any(r in text.lower() for r in ["loved", "liked", "emphasized", "laughed at"]):
                    continue

                if m.is_from_me:
                    my_msgs.append(text)
                else:
                    their_msgs.append(text)

            if len(my_msgs) >= 10 and len(their_msgs) >= 10:
                contacts.append((contact_key, my_msgs, their_msgs))
                seen.add(contact_key)

        except Exception:
            continue

    print(f"Loaded {len(contacts)} contacts")
    return contacts


def cluster_by_style(features: list[StyleFeatures], n_clusters: int = 5) -> tuple[np.ndarray, dict]:
    """Cluster contacts by style features."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Build feature matrix
    X = np.array([f.to_vector() for f in features])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Get cluster centers in original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, {"centers": centers, "scaler": scaler}


def analyze_clusters(features: list[StyleFeatures], labels: np.ndarray) -> dict:
    """Analyze what makes each cluster distinct."""

    feature_names = [
        "my_avg_len", "their_avg_len", "my_len_variance",
        "no_punct", "exclaim", "question",
        "lol", "emoji", "abbreviations",
        "lowercase_start", "all_lowercase",
        "short_response", "one_word"
    ]

    clusters = defaultdict(list)
    for f, label in zip(features, labels):
        clusters[int(label)].append(f)

    analysis = {}

    for label, members in sorted(clusters.items()):
        # Calculate average features for this cluster
        vectors = np.array([f.to_vector() for f in members])
        avg_features = vectors.mean(axis=0)

        # Find distinguishing features (compare to overall mean)
        all_vectors = np.array([f.to_vector() for f in features])
        overall_mean = all_vectors.mean(axis=0)
        overall_std = all_vectors.std(axis=0) + 1e-6

        # Z-score of cluster mean vs overall
        z_scores = (avg_features - overall_mean) / overall_std

        # Top distinguishing features
        distinguishing = []
        for i, (name, z) in enumerate(zip(feature_names, z_scores)):
            if abs(z) > 0.5:
                direction = "high" if z > 0 else "low"
                distinguishing.append(f"{name} ({direction})")

        # Describe the cluster
        if avg_features[0] < 15:  # short messages
            style_desc = "Very brief"
        elif avg_features[0] > 40:
            style_desc = "Longer messages"
        else:
            style_desc = "Medium length"

        if avg_features[6] > 0.1:  # lol rate
            style_desc += ", casual/playful"
        if avg_features[3] > 0.8:  # no punct
            style_desc += ", no punctuation"
        if avg_features[8] > 0.1:  # abbreviations
            style_desc += ", uses abbreviations"

        analysis[f"cluster_{label}"] = {
            "members": [f.name for f in members],
            "count": len(members),
            "style_description": style_desc,
            "distinguishing_features": distinguishing,
            "avg_my_msg_len": avg_features[0],
            "avg_their_msg_len": avg_features[1],
            "no_punct_rate": avg_features[3],
            "lol_rate": avg_features[6],
            "emoji_rate": avg_features[7],
            "abbreviation_rate": avg_features[8],
            "short_response_rate": avg_features[11],
        }

    return analysis


def visualize_clusters(features: list[StyleFeatures], labels: np.ndarray):
    """Create visualization of clusters."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    # Get feature vectors
    X = np.array([f.to_vector() for f in features])

    # PCA for 2D projection
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(14, 10))

    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[label]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=100,
        )

    # Add names
    for i, f in enumerate(features):
        plt.annotate(
            f.name[:12],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.8,
        )

    plt.legend()
    plt.title("Contact Clusters by Communication Style")
    plt.xlabel("Style Dimension 1")
    plt.ylabel("Style Dimension 2")
    plt.tight_layout()

    viz_file = RESULTS_DIR / "style_clusters.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {viz_file}")
    plt.close()


def run_analysis(n_clusters: int = 5, min_messages: int = 20):
    """Run the full style analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    contacts = load_contacts(min_messages=min_messages)

    if len(contacts) < n_clusters:
        print(f"Not enough contacts ({len(contacts)}) for {n_clusters} clusters")
        return

    # Extract features
    print("Extracting style features...")
    features = [extract_style_features(name, my, their) for name, my, their in contacts]

    # Cluster
    print(f"Clustering into {n_clusters} groups...")
    labels, cluster_data = cluster_by_style(features, n_clusters=n_clusters)

    # Analyze
    analysis = analyze_clusters(features, labels)

    # Print results
    print("\n" + "=" * 70)
    print("STYLE-BASED CLUSTERING RESULTS")
    print("=" * 70)
    print(f"Contacts analyzed: {len(features)}")
    print(f"Clusters: {n_clusters}")

    for cluster_name, data in sorted(analysis.items()):
        print(f"\n{'-' * 60}")
        print(f"{cluster_name.upper()} ({data['count']} contacts)")
        print(f"Style: {data['style_description']}")
        print(f"Distinguishing: {', '.join(data['distinguishing_features'][:5]) or 'average'}")
        print(f"Avg msg length: {data['avg_my_msg_len']:.0f} chars")
        print(f"No punctuation: {data['no_punct_rate']*100:.0f}%")
        print(f"Uses lol/haha: {data['lol_rate']*100:.0f}%")
        print(f"Uses abbreviations: {data['abbreviation_rate']*100:.0f}%")
        print(f"Members: {', '.join(data['members'][:8])}")
        if len(data['members']) > 8:
            print(f"         ... and {len(data['members']) - 8} more")

    # Save results
    results = {
        "n_contacts": len(features),
        "n_clusters": n_clusters,
        "contacts": [
            {
                "name": f.name,
                "cluster": int(labels[i]),
                "my_avg_len": f.my_avg_len,
                "no_punct_rate": f.my_no_punct_rate,
                "lol_rate": f.my_lol_rate,
                "abbreviation_rate": f.my_abbreviation_rate,
                "top_starters": f.top_starters,
            }
            for i, f in enumerate(features)
        ],
        "clusters": analysis,
    }

    results_file = RESULTS_DIR / "style_clusters.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_file}")

    # Visualize
    visualize_clusters(features, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--min-messages", type=int, default=20, help="Min messages per contact")
    args = parser.parse_args()

    run_analysis(n_clusters=args.clusters, min_messages=args.min_messages)


if __name__ == "__main__":
    main()
