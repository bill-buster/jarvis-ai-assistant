#!/usr/bin/env python3
"""Cluster contacts by CONTENT + STYLE using chunked embeddings.

Embeds conversations in chunks (to handle long context), extracts style
features, and combines both for clustering.

Usage:
    python scripts/cluster_contacts.py
    python scripts/cluster_contacts.py --clusters 6
"""

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/clustering")


@dataclass
class ContactData:
    """Data for a single contact."""
    name: str
    message_count: int
    my_messages: list[str]
    their_messages: list[str]
    chunks: list[str]  # Conversation split into chunks for embedding


def load_contacts(min_messages: int = 20, chunk_size: int = 1500) -> list[ContactData]:
    """Load contacts and split conversations into chunks."""
    from core.imessage.reader import MessageReader

    reader = MessageReader()
    print("Loading conversations...")
    conversations = reader.get_conversations(limit=500)

    spam_keywords = ["verification code", "click here", "unsubscribe", "your order", "tracking"]

    contacts = []
    seen = set()

    for conv in conversations:
        name = conv.display_name or ""
        participants = conv.participants or []
        contact_key = name or (participants[0] if participants else "")

        if contact_key in seen or not contact_key:
            continue
        if contact_key.isdigit() and 5 <= len(contact_key) <= 6:
            continue

        try:
            messages = reader.get_messages(conv.chat_id, limit=500)
            if not messages or len(messages) < min_messages:
                continue

            messages = list(reversed(messages))

            sample = " ".join((m.text or "").lower() for m in messages[:30])
            if sum(1 for kw in spam_keywords if kw in sample) >= 2:
                continue

            my_msgs = []
            their_msgs = []
            all_lines = []

            for m in messages:
                text = (m.text or "").strip()
                if not text:
                    continue
                if any(r in text.lower() for r in ["loved", "liked", "emphasized", "laughed at"]):
                    continue

                prefix = "me:" if m.is_from_me else "them:"
                all_lines.append(f"{prefix} {text}")

                if m.is_from_me:
                    my_msgs.append(text)
                else:
                    their_msgs.append(text)

            if len(my_msgs) < 10 or len(their_msgs) < 10:
                continue

            # Split into chunks
            full_text = "\n".join(all_lines)
            chunks = []
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                if len(chunk) > 200:  # Skip tiny chunks
                    chunks.append(chunk)

            if not chunks:
                chunks = [full_text[-chunk_size:]]

            contacts.append(ContactData(
                name=contact_key,
                message_count=len(messages),
                my_messages=my_msgs[-100:],
                their_messages=their_msgs[-100:],
                chunks=chunks,
            ))
            seen.add(contact_key)

        except Exception:
            continue

    print(f"Loaded {len(contacts)} contacts")
    return contacts


def extract_style_features(my_messages: list[str], their_messages: list[str]) -> np.ndarray:
    """Extract style features as a vector."""

    if not my_messages:
        my_messages = [""]

    # Length
    my_lens = [len(m) for m in my_messages]
    my_avg_len = np.mean(my_lens)
    my_len_std = np.std(my_lens) if len(my_lens) > 1 else 0

    # Punctuation
    def ends_rate(msgs, chars):
        return sum(1 for m in msgs if m.rstrip() and m.rstrip()[-1] in chars) / max(len(msgs), 1)

    no_punct = 1 - ends_rate(my_messages, ".!?")
    exclaim = ends_rate(my_messages, "!")
    question = ends_rate(my_messages, "?")

    # Casual markers
    def pattern_rate(msgs, patterns):
        count = sum(1 for m in msgs if any(re.search(p, m.lower()) for p in patterns))
        return count / max(len(msgs), 1)

    lol_rate = pattern_rate(my_messages, [r'\blol\b', r'\blmao\b', r'\bhaha'])
    emoji_rate = pattern_rate(my_messages, [r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]'])
    abbrev_rate = pattern_rate(my_messages, [r'\bu\b', r'\bur\b', r'\brn\b', r'\btmrw\b', r'\bidk\b'])

    # Capitalization
    lowercase_start = sum(1 for m in my_messages if m and m[0].islower()) / max(len(my_messages), 1)

    # Response patterns
    short_rate = sum(1 for m in my_messages if len(m) < 10) / max(len(my_messages), 1)
    one_word = sum(1 for m in my_messages if len(m.split()) == 1) / max(len(my_messages), 1)

    return np.array([
        my_avg_len / 50,  # Normalize
        my_len_std / 30,
        no_punct,
        exclaim,
        question,
        lol_rate,
        emoji_rate,
        abbrev_rate,
        lowercase_start,
        short_rate,
        one_word,
    ])


def embed_chunks_batched(contacts: list[ContactData], model_name: str = "BAAI/bge-base-en-v1.5") -> np.ndarray:
    """Embed all chunks and average per contact."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Collect all chunks with contact index
    all_chunks = []
    chunk_to_contact = []

    for i, contact in enumerate(contacts):
        for chunk in contact.chunks:
            all_chunks.append(chunk)
            chunk_to_contact.append(i)

    print(f"Embedding {len(all_chunks)} chunks from {len(contacts)} contacts...")

    # Embed in batches
    chunk_embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=16,
    )

    # Average embeddings per contact
    contact_embeddings = []
    for i in range(len(contacts)):
        mask = [j for j, ci in enumerate(chunk_to_contact) if ci == i]
        if mask:
            avg_emb = chunk_embeddings[mask].mean(axis=0)
        else:
            avg_emb = np.zeros(chunk_embeddings.shape[1])
        contact_embeddings.append(avg_emb)

    del model
    gc.collect()

    return np.array(contact_embeddings)


def combine_features(content_embeddings: np.ndarray, style_features: np.ndarray, content_weight: float = 0.7) -> np.ndarray:
    """Combine content embeddings with style features."""
    from sklearn.preprocessing import StandardScaler

    # Normalize both
    content_norm = content_embeddings / (np.linalg.norm(content_embeddings, axis=1, keepdims=True) + 1e-8)

    scaler = StandardScaler()
    style_norm = scaler.fit_transform(style_features)

    # Weight and concatenate
    # Scale style to have similar magnitude as content
    style_scaled = style_norm * 0.1  # Style features contribute less than content

    combined = np.concatenate([
        content_norm * content_weight,
        style_scaled * (1 - content_weight)
    ], axis=1)

    return combined


def cluster_contacts(features: np.ndarray, n_clusters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Cluster using KMeans."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # 2D projection for visualization
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(features)

    return labels, coords_2d


def analyze_clusters(contacts: list[ContactData], labels: np.ndarray, style_features: np.ndarray) -> dict:
    """Analyze clusters."""

    style_names = ["avg_len", "len_std", "no_punct", "exclaim", "question",
                   "lol", "emoji", "abbrev", "lowercase", "short", "one_word"]

    clusters = defaultdict(list)
    cluster_styles = defaultdict(list)

    for i, (contact, label) in enumerate(zip(contacts, labels)):
        clusters[int(label)].append(contact)
        cluster_styles[int(label)].append(style_features[i])

    analysis = {}

    for label in sorted(clusters.keys()):
        members = clusters[label]
        styles = np.array(cluster_styles[label])
        avg_style = styles.mean(axis=0)

        # Describe cluster
        desc_parts = []
        if avg_style[0] * 50 < 20:
            desc_parts.append("brief messages")
        elif avg_style[0] * 50 > 40:
            desc_parts.append("longer messages")

        if avg_style[2] > 0.7:
            desc_parts.append("no punctuation")
        if avg_style[5] > 0.05:
            desc_parts.append("uses lol/haha")
        if avg_style[7] > 0.05:
            desc_parts.append("uses abbreviations")
        if avg_style[9] > 0.3:
            desc_parts.append("many short responses")

        # Common topics from messages
        all_words = []
        for m in members:
            for msg in m.my_messages[-30:]:
                all_words.extend(msg.lower().split())

        word_counts = Counter(all_words)
        # Filter common words
        stop_words = {"i", "the", "a", "to", "and", "is", "it", "that", "you", "of", "in", "for", "on", "my", "me", "we", "be"}
        topic_words = [w for w, c in word_counts.most_common(30) if w not in stop_words and len(w) > 2][:10]

        analysis[f"cluster_{label}"] = {
            "count": len(members),
            "members": [m.name for m in members],
            "style_description": ", ".join(desc_parts) if desc_parts else "average style",
            "avg_msg_len": avg_style[0] * 50,
            "no_punct_rate": avg_style[2],
            "lol_rate": avg_style[5],
            "abbrev_rate": avg_style[7],
            "short_response_rate": avg_style[9],
            "topic_words": topic_words,
        }

    return analysis


def visualize(contacts: list[ContactData], labels: np.ndarray, coords_2d: np.ndarray):
    """Create visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(16, 12))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[label]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=120,
        )

    for i, contact in enumerate(contacts):
        plt.annotate(
            contact.name[:15],
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=7,
            alpha=0.8,
        )

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title("Contact Clusters (Content + Style)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    viz_file = RESULTS_DIR / "content_style_clusters.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {viz_file}")
    plt.close()


def run_clustering(n_clusters: int = 5, min_messages: int = 20):
    """Run full clustering pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load contacts
    contacts = load_contacts(min_messages=min_messages)

    if len(contacts) < n_clusters:
        print(f"Not enough contacts for {n_clusters} clusters")
        return

    # Extract style features
    print("Extracting style features...")
    style_features = np.array([
        extract_style_features(c.my_messages, c.their_messages)
        for c in contacts
    ])

    # Embed content (chunked)
    content_embeddings = embed_chunks_batched(contacts)

    # Combine
    print("Combining content + style features...")
    combined = combine_features(content_embeddings, style_features)

    # Cluster
    print(f"Clustering {len(contacts)} contacts into {n_clusters} groups...")
    labels, coords_2d = cluster_contacts(combined, n_clusters=n_clusters)

    # Analyze
    analysis = analyze_clusters(contacts, labels, style_features)

    # Print results
    print("\n" + "=" * 70)
    print("CONTENT + STYLE CLUSTERING RESULTS")
    print("=" * 70)

    for cluster_name, data in sorted(analysis.items()):
        print(f"\n{'-' * 60}")
        print(f"{cluster_name.upper()} ({data['count']} contacts)")
        print(f"Style: {data['style_description']}")
        print(f"Avg msg len: {data['avg_msg_len']:.0f} | No punct: {data['no_punct_rate']*100:.0f}% | Lol: {data['lol_rate']*100:.0f}%")
        print(f"Topics: {', '.join(data['topic_words'][:7])}")
        print(f"Members: {', '.join(data['members'][:10])}")
        if len(data['members']) > 10:
            print(f"         ... and {len(data['members']) - 10} more")

    # Save
    results = {
        "n_contacts": len(contacts),
        "n_clusters": n_clusters,
        "clusters": analysis,
        "contacts": [
            {"name": c.name, "cluster": int(labels[i]), "n_messages": c.message_count}
            for i, c in enumerate(contacts)
        ]
    }

    with open(RESULTS_DIR / "clusters.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualize
    visualize(contacts, labels, coords_2d)

    print(f"\nResults saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--min-messages", type=int, default=20, help="Min messages per contact")
    args = parser.parse_args()

    run_clustering(n_clusters=args.clusters, min_messages=args.min_messages)


if __name__ == "__main__":
    main()
