#!/usr/bin/env python3
"""Import ALL contacts from iMessage database for profiling.

Pulls every contact you've ever texted, with:
- Display name (if available)
- Phone number / email
- Message count (how much you text them)
- Last message date

Then you can profile them systematically.

Usage:
    python scripts/import_all_contacts.py              # Import contacts
    python scripts/import_all_contacts.py --stats      # Show stats
    python scripts/import_all_contacts.py --top 100    # Show top 100 by message count
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

IMESSAGE_DB = os.path.expanduser("~/Library/Messages/chat.db")
CONTACTS_FILE = Path("results/contacts/all_contacts.json")
PROFILES_FILE = Path("results/contacts/contact_profiles.json")

# Relationship types for quick reference
RELATIONSHIPS = {
    # Family
    "dad": "family", "mom": "family", "brother": "family", "sister": "family",
    "cousin": "family", "uncle": "family", "aunt": "family", "grandparent": "family",
    # Friends
    "best_friend": "friend", "close_friend": "friend", "friend": "friend",
    "acquaintance": "acquaintance",
    # Romantic
    "partner": "romantic", "dating": "romantic", "ex": "romantic",
    # Work/School
    "coworker": "work", "boss": "work", "classmate": "school", "professor": "school",
    # Other
    "service": "other", "spam": "spam", "unknown": "unknown",
}


def import_contacts():
    """Import all contacts from iMessage database."""
    print("\n" + "=" * 70)
    print("IMPORTING CONTACTS FROM IMESSAGE")
    print("=" * 70)

    if not os.path.exists(IMESSAGE_DB):
        print(f"Error: iMessage database not found at {IMESSAGE_DB}")
        print("Make sure you have Full Disk Access enabled.")
        return

    conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all handles with message counts
    query = """
    SELECT
        h.id as handle_id,
        h.service,
        COUNT(m.rowid) as message_count,
        MAX(m.date) as last_message_date
    FROM handle h
    LEFT JOIN message m ON m.handle_id = h.rowid
    GROUP BY h.id
    ORDER BY message_count DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Get display names from chat table where available
    chat_names = {}
    cursor.execute("""
        SELECT
            c.chat_identifier,
            c.display_name
        FROM chat c
        WHERE c.display_name IS NOT NULL AND c.display_name != ''
    """)
    for chat_id, display_name in cursor.fetchall():
        chat_names[chat_id] = display_name

    conn.close()

    # Process contacts
    contacts = {}
    for handle_id, service, msg_count, last_date in rows:
        # Skip if no messages
        if msg_count == 0:
            continue

        # Clean up handle
        clean_id = handle_id.strip()

        # Get display name if available
        display_name = chat_names.get(clean_id, "")

        # Determine contact type
        if clean_id.startswith("+"):
            contact_type = "phone"
        elif "@" in clean_id:
            contact_type = "email"
        elif clean_id.startswith("urn:"):
            contact_type = "business"
            continue  # Skip business messages
        elif clean_id.isdigit() and len(clean_id) <= 6:
            contact_type = "shortcode"
            continue  # Skip short codes (spam)
        else:
            contact_type = "other"

        # Convert Apple timestamp to datetime
        if last_date:
            # Apple uses nanoseconds since 2001-01-01
            try:
                last_dt = datetime(2001, 1, 1) + timedelta(seconds=last_date / 1e9)
                last_date_str = last_dt.strftime("%Y-%m-%d")
            except:
                last_date_str = None
        else:
            last_date_str = None

        contacts[clean_id] = {
            "handle": clean_id,
            "display_name": display_name,
            "contact_type": contact_type,
            "service": service,
            "message_count": msg_count,
            "last_message": last_date_str,
        }

    # Save
    CONTACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTACTS_FILE, "w") as f:
        json.dump(contacts, f, indent=2)

    print(f"\n✓ Imported {len(contacts)} contacts")
    print(f"  Saved to: {CONTACTS_FILE}")

    # Show stats
    show_stats()


def show_stats():
    """Show contact statistics."""
    if not CONTACTS_FILE.exists():
        print("No contacts imported yet. Run: python scripts/import_all_contacts.py")
        return

    with open(CONTACTS_FILE) as f:
        contacts = json.load(f)

    # Load existing profiles
    profiles = {}
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            profiles = json.load(f).get("individuals", {})

    print("\n" + "=" * 70)
    print("CONTACT STATS")
    print("=" * 70)

    print(f"\nTotal contacts: {len(contacts)}")

    # By type
    by_type = {}
    for c in contacts.values():
        t = c.get("contact_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # With display names
    with_names = sum(1 for c in contacts.values() if c.get("display_name"))
    print(f"\nWith display names: {with_names}/{len(contacts)} ({with_names/len(contacts)*100:.0f}%)")

    # Profiled
    profiled = sum(1 for handle in contacts if handle in profiles or
                   any(c.get("display_name", "") == p for c in contacts.values() for p in profiles))
    print(f"Already profiled: {profiled}/{len(contacts)} ({profiled/len(contacts)*100:.0f}%)")

    # Message distribution
    counts = sorted([c["message_count"] for c in contacts.values()], reverse=True)
    print(f"\nMessage distribution:")
    print(f"  Top 10 contacts: {sum(counts[:10])} messages")
    print(f"  Top 50 contacts: {sum(counts[:50])} messages")
    print(f"  Top 100 contacts: {sum(counts[:100])} messages")
    print(f"  Total messages: {sum(counts)}")


def show_top(n: int = 100):
    """Show top N contacts by message count."""
    if not CONTACTS_FILE.exists():
        print("No contacts imported yet. Run: python scripts/import_all_contacts.py")
        return

    with open(CONTACTS_FILE) as f:
        contacts = json.load(f)

    # Load existing profiles
    profiles = {}
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            profiles = json.load(f).get("individuals", {})

    # Sort by message count
    sorted_contacts = sorted(
        contacts.items(),
        key=lambda x: x[1]["message_count"],
        reverse=True
    )[:n]

    print("\n" + "=" * 70)
    print(f"TOP {n} CONTACTS BY MESSAGE COUNT")
    print("=" * 70)
    print(f"\n{'#':>4} {'Messages':>8} {'Name/Handle':<35} {'Profiled'}")
    print("-" * 70)

    for i, (handle, info) in enumerate(sorted_contacts, 1):
        name = info.get("display_name") or handle
        if len(name) > 35:
            name = name[:32] + "..."

        # Check if profiled
        is_profiled = handle in profiles or info.get("display_name", "") in profiles
        profiled_str = "✓" if is_profiled else ""

        print(f"{i:>4} {info['message_count']:>8} {name:<35} {profiled_str}")


def run_profile_import():
    """Import contacts and start profiling flow."""
    import_contacts()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
Your contacts are imported. Now profile them:

1. Quick profile top contacts (most messages first):
   python scripts/profile_my_contacts.py

2. Or view top contacts first:
   python scripts/import_all_contacts.py --top 50

3. Check progress anytime:
   python scripts/profile_my_contacts.py --stats
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Show contact stats")
    parser.add_argument("--top", type=int, metavar="N", help="Show top N contacts")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.top:
        show_top(args.top)
    else:
        run_profile_import()


if __name__ == "__main__":
    main()
