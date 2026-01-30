#!/usr/bin/env python3
"""Extract ALL contacts from iMessage with resolved names from AddressBook.

Uses the ChatDBReader which automatically resolves phone numbers to contact names.

Usage:
    python scripts/extract_contacts_with_names.py           # Extract all
    python scripts/extract_contacts_with_names.py --stats   # Show stats
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add parent jarvis repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "jarvis-ai-assistant"))
sys.path.insert(0, str(Path(__file__).parent.parent))

CONTACTS_OUTPUT = Path("results/contacts/all_contacts_with_names.json")


def extract_contacts():
    """Extract all contacts with resolved names."""
    print("\n" + "=" * 70)
    print("EXTRACTING CONTACTS WITH RESOLVED NAMES")
    print("=" * 70)

    from integrations.imessage.reader import ChatDBReader

    reader = ChatDBReader()

    print("\nLoading conversations (this resolves names from AddressBook)...")
    convos = reader.get_conversations(limit=2000)  # Get lots

    print(f"Found {len(convos)} conversations")

    contacts = {}
    no_name_count = 0

    for conv in convos:
        display_name = conv.display_name
        participants = conv.participants or []
        participant = participants[0] if participants else ""

        # Use display_name if available, otherwise participant
        if display_name and display_name != "NO_NAME":
            name = display_name
        elif participant and not participant.startswith("+"):
            name = participant
        else:
            no_name_count += 1
            continue  # Skip contacts without names

        # Skip numeric-only names (short codes, spam)
        if name.replace(" ", "").isdigit():
            continue

        # Skip business URNs
        if "urn:biz" in str(participant):
            continue

        is_group = len(participants) > 1 if participants else False

        if name not in contacts:
            contacts[name] = {
                "name": name,
                "identifier": participant,
                "is_group": is_group,
                "chat_id": conv.chat_id,
            }

    # Save
    CONTACTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTACTS_OUTPUT, "w") as f:
        json.dump(contacts, f, indent=2)

    print(f"\n✓ Extracted {len(contacts)} contacts with names")
    print(f"  Skipped {no_name_count} without names")
    print(f"  Saved to: {CONTACTS_OUTPUT}")

    # Show breakdown
    groups = sum(1 for c in contacts.values() if c.get("is_group"))
    individuals = len(contacts) - groups

    print(f"\n  Individuals: {individuals}")
    print(f"  Groups: {groups}")

    # Show sample
    print(f"\nSample contacts:")
    for name in list(contacts.keys())[:20]:
        group_marker = " (group)" if contacts[name].get("is_group") else ""
        print(f"  - {name}{group_marker}")

    return contacts


def show_stats():
    """Show extraction stats."""
    if not CONTACTS_OUTPUT.exists():
        print("No contacts extracted yet. Run: python scripts/extract_contacts_with_names.py")
        return

    with open(CONTACTS_OUTPUT) as f:
        contacts = json.load(f)

    # Load profiles to see what's done
    profiles_file = Path("results/contacts/contact_profiles.json")
    profiles = {}
    if profiles_file.exists():
        with open(profiles_file) as f:
            profiles = json.load(f).get("individuals", {})

    print("\n" + "=" * 70)
    print("CONTACT STATS")
    print("=" * 70)

    groups = sum(1 for c in contacts.values() if c.get("is_group"))
    individuals = len(contacts) - groups

    print(f"\nTotal contacts with names: {len(contacts)}")
    print(f"  Individuals: {individuals}")
    print(f"  Groups: {groups}")

    profiled = sum(1 for name in contacts if name in profiles)
    print(f"\nProfiled: {profiled}/{len(contacts)} ({profiled/len(contacts)*100:.0f}%)")

    # Show unprofiled
    unprofiled = [name for name in contacts if name not in profiles]
    if unprofiled:
        print(f"\nSample unprofiled:")
        for name in unprofiled[:15]:
            print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        extract_contacts()


if __name__ == "__main__":
    main()
