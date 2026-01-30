#!/usr/bin/env python3
"""Label your contacts with relationship type.

Instead of labeling 500 samples, label 193 contacts once.
Each contact's relationship applies to all their messages.

Usage:
    python scripts/label_contacts.py                # Interactive labeling
    python scripts/label_contacts.py --stats        # Show progress
    python scripts/label_contacts.py --export       # Export labeled contacts
    python scripts/label_contacts.py --apply        # Apply labels to test data
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_DATA = Path("results/test_set/clean_test_data.jsonl")
LABELED_DATA = Path("results/test_set/model_results.jsonl")
CONTACT_LABELS = Path("results/contacts/contact_labels.json")

# Relationship categories
RELATIONSHIPS = {
    "f": "family",
    "c": "close_friend",
    "a": "casual_friend",
    "w": "work",
    "r": "romantic",
    "q": "acquaintance",
    "g": "group_mixed",  # Group chat with mixed relationships
    "s": "skip",         # Skip (spam, unknown, etc.)
}


def load_contacts():
    """Extract unique contacts from test data."""
    contacts = {}

    # From clean data
    if CLEAN_DATA.exists():
        with open(CLEAN_DATA) as f:
            for line in f:
                d = json.loads(line)
                contact = d.get("contact", "unknown")
                is_group = d.get("is_group", False)

                if contact not in contacts:
                    contacts[contact] = {
                        "name": contact,
                        "is_group": is_group,
                        "sample_count": 0,
                        "sample_responses": [],
                    }

                contacts[contact]["sample_count"] += 1
                # Keep a few sample responses for context
                if len(contacts[contact]["sample_responses"]) < 3:
                    contacts[contact]["sample_responses"].append({
                        "last_msg": d.get("last_message", "")[:50],
                        "gold": d.get("gold_response", "")[:50],
                    })

    return contacts


def load_existing_labels():
    """Load already-labeled contacts."""
    if CONTACT_LABELS.exists():
        with open(CONTACT_LABELS) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    """Save contact labels."""
    CONTACT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTACT_LABELS, "w") as f:
        json.dump(labels, f, indent=2)


def run_labeling():
    """Interactive contact labeling."""
    print("\n" + "=" * 70)
    print("CONTACT LABELING")
    print("=" * 70)
    print("""
Label your contacts ONCE, and it applies to all their messages.
This is much faster than labeling 500 individual samples.

Relationship types:
  f = family (parents, siblings, relatives)
  c = close_friend (best friends, inner circle)
  a = casual_friend (acquaintances you text sometimes)
  w = work (colleagues, professional contacts)
  r = romantic (partner, dating)
  q = acquaintance (barely know them)
  g = group_mixed (group chat with mixed relationships)
  s = skip (spam, unknown numbers, etc.)

Commands: [f/c/a/w/r/q/g/s] to label, [enter] to skip, [quit] to exit
""")

    contacts = load_contacts()
    labels = load_existing_labels()

    # Sort by sample count (most messages first = most important)
    sorted_contacts = sorted(
        contacts.items(),
        key=lambda x: x[1]["sample_count"],
        reverse=True
    )

    unlabeled = [(name, info) for name, info in sorted_contacts if name not in labels]
    labeled_count = len(labels)
    total_count = len(contacts)

    print(f"Progress: {labeled_count}/{total_count} contacts labeled")
    print(f"Remaining: {len(unlabeled)} contacts\n")

    if not unlabeled:
        print("All contacts labeled!")
        show_stats()
        return

    for idx, (name, info) in enumerate(unlabeled):
        print("\n" + "-" * 70)
        print(f"[{labeled_count + idx + 1}/{total_count}] {name}")
        print(f"  Messages in dataset: {info['sample_count']}")
        print(f"  Is group chat: {info['is_group']}")

        if info["sample_responses"]:
            print("  Sample exchanges:")
            for sample in info["sample_responses"]:
                print(f"    them: \"{sample['last_msg']}\"")
                print(f"    you:  \"{sample['gold']}\"")

        print("\n  [f]amily [c]lose [a]casual [w]ork [r]omantic [q]acquaint [g]roup [s]kip")

        while True:
            choice = input("  Relationship: ").strip().lower()

            if choice == "quit":
                save_labels(labels)
                print(f"\nSaved {len(labels)} labels.")
                show_stats()
                return

            if choice == "" or choice not in RELATIONSHIPS:
                print("  Skipping...")
                break

            labels[name] = {
                "relationship": RELATIONSHIPS[choice],
                "is_group": info["is_group"],
            }
            print(f"  → Labeled as: {RELATIONSHIPS[choice]}")
            break

    save_labels(labels)
    print(f"\n✓ Saved {len(labels)} labels to {CONTACT_LABELS}")
    show_stats()


def show_stats():
    """Show labeling progress and distribution."""
    contacts = load_contacts()
    labels = load_existing_labels()

    print("\n" + "=" * 70)
    print("LABELING STATS")
    print("=" * 70)

    total = len(contacts)
    labeled = len(labels)

    print(f"\nProgress: {labeled}/{total} contacts ({labeled/total*100:.0f}%)")

    # Sample coverage
    total_samples = sum(c["sample_count"] for c in contacts.values())
    labeled_samples = sum(
        contacts[name]["sample_count"]
        for name in labels.keys()
        if name in contacts
    )
    pct = labeled_samples / total_samples * 100
    print(f"Sample coverage: {labeled_samples}/{total_samples} ({pct:.0f}%)")

    # Distribution
    if labels:
        dist = Counter(v["relationship"] for v in labels.values())
        print("\nRelationship distribution:")
        for rel, count in dist.most_common():
            print(f"  {rel}: {count}")

    # Top unlabeled by message count
    unlabeled = [
        (name, info) for name, info in contacts.items()
        if name not in labels
    ]
    unlabeled.sort(key=lambda x: x[1]["sample_count"], reverse=True)

    if unlabeled:
        print("\nTop unlabeled contacts (by message count):")
        for name, info in unlabeled[:10]:
            print(f"  {name}: {info['sample_count']} messages")


def apply_labels():
    """Apply contact labels to test data."""
    labels = load_existing_labels()

    if not labels:
        print("No labels yet. Run: python scripts/label_contacts.py")
        return

    # Load and update clean data
    updated = []
    labeled_count = 0

    with open(CLEAN_DATA) as f:
        for line in f:
            d = json.loads(line)
            contact = d.get("contact", "unknown")

            if contact in labels:
                d["relationship"] = labels[contact]["relationship"]
                labeled_count += 1
            else:
                d["relationship"] = "unknown"

            updated.append(d)

    # Save to new file
    output_path = Path("results/test_set/labeled_test_data.jsonl")
    with open(output_path, "w") as f:
        for d in updated:
            f.write(json.dumps(d) + "\n")

    print(f"✓ Applied labels to {labeled_count}/{len(updated)} samples")
    print(f"  Saved to: {output_path}")

    # Show distribution
    dist = Counter(d.get("relationship", "unknown") for d in updated)
    print("\nRelationship distribution in test data:")
    for rel, count in dist.most_common():
        pct = count / len(updated) * 100
        print(f"  {rel}: {count} ({pct:.0f}%)")


def export_labels():
    """Export labels in a simple format."""
    labels = load_existing_labels()

    if not labels:
        print("No labels yet.")
        return

    print("\n" + "=" * 70)
    print("CONTACT LABELS EXPORT")
    print("=" * 70)

    for name, info in sorted(labels.items()):
        print(f"{info['relationship']:15} | {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Show labeling stats")
    parser.add_argument("--export", action="store_true", help="Export labels")
    parser.add_argument("--apply", action="store_true", help="Apply labels to test data")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.export:
        export_labels()
    elif args.apply:
        apply_labels()
    else:
        run_labeling()


if __name__ == "__main__":
    main()
