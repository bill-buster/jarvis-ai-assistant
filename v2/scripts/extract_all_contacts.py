#!/usr/bin/env python3
"""Extract ALL contacts from macOS Contacts/AddressBook with phone numbers.

This accesses the AddressBook SQLite databases directly to get complete
contact information including phone numbers, emails, and names.

Usage:
    python scripts/extract_all_contacts.py              # Extract all contacts
    python scripts/extract_all_contacts.py --stats      # Show stats
    python scripts/extract_all_contacts.py --merge      # Merge with existing
"""

import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

ADDRESSBOOK_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"
OUTPUT_FILE = Path("results/contacts/all_contacts_with_phones.json")
EXISTING_FILE = Path("results/contacts/all_contacts_with_names.json")

DB_TIMEOUT = 5.0


def normalize_phone(phone: str | None) -> str | None:
    """Normalize phone number to consistent format."""
    if not phone:
        return None
    # Remove all non-digit characters except leading +
    digits = "".join(c for c in phone if c.isdigit() or c == "+")
    if not digits:
        return None
    # Ensure US numbers have +1 prefix
    if len(digits) == 10 and not digits.startswith("+"):
        digits = "+1" + digits
    elif len(digits) == 11 and digits.startswith("1"):
        digits = "+" + digits
    return digits


def format_name(first: str | None, last: str | None) -> str | None:
    """Format first and last name into display name."""
    parts = []
    if first:
        parts.append(first.strip())
    if last:
        parts.append(last.strip())
    return " ".join(parts) if parts else None


def extract_from_addressbook():
    """Extract all contacts from macOS AddressBook databases."""
    if not ADDRESSBOOK_PATH.exists():
        print(f"AddressBook path not found: {ADDRESSBOOK_PATH}")
        print("Make sure you have Full Disk Access enabled.")
        return {}

    contacts = {}  # name -> {phones: [], emails: [], ...}

    # Find all AddressBook databases
    db_files = []
    for source_dir in ADDRESSBOOK_PATH.iterdir():
        if source_dir.is_dir():
            db_path = source_dir / "AddressBook-v22.abcddb"
            if db_path.exists():
                db_files.append(db_path)

    if not db_files:
        print("No AddressBook databases found.")
        return {}

    print(f"Found {len(db_files)} AddressBook database(s)")

    for db_path in db_files:
        try:
            extract_from_db(db_path, contacts)
        except Exception as e:
            print(f"  Error reading {db_path.parent.name}: {e}")

    return contacts


def extract_from_db(db_path: Path, contacts: dict):
    """Extract contacts from a specific AddressBook database."""
    source_name = db_path.parent.name

    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all contacts with their names
        cursor.execute("""
            SELECT
                Z_PK as record_id,
                ZFIRSTNAME as first_name,
                ZLASTNAME as last_name,
                ZNICKNAME as nickname,
                ZORGANIZATION as organization
            FROM ZABCDRECORD
            WHERE ZFIRSTNAME IS NOT NULL OR ZLASTNAME IS NOT NULL OR ZORGANIZATION IS NOT NULL
        """)

        records = {}
        for row in cursor.fetchall():
            name = format_name(row["first_name"], row["last_name"])
            if not name and row["organization"]:
                name = row["organization"]
            if not name and row["nickname"]:
                name = row["nickname"]
            if name:
                records[row["record_id"]] = {
                    "name": name,
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "nickname": row["nickname"],
                    "organization": row["organization"],
                }

        # Get phone numbers
        cursor.execute("""
            SELECT
                ZOWNER as record_id,
                ZFULLNUMBER as phone,
                ZLABEL as label
            FROM ZABCDPHONENUMBER
            WHERE ZFULLNUMBER IS NOT NULL
        """)

        for row in cursor.fetchall():
            record_id = row["record_id"]
            if record_id not in records:
                continue

            phone = normalize_phone(row["phone"])
            if phone:
                name = records[record_id]["name"]
                if name not in contacts:
                    contacts[name] = {
                        "name": name,
                        "phones": [],
                        "emails": [],
                        "is_group": False,
                        "source": source_name,
                    }
                if phone not in contacts[name]["phones"]:
                    contacts[name]["phones"].append(phone)

        # Get email addresses
        cursor.execute("""
            SELECT
                ZOWNER as record_id,
                ZADDRESS as email,
                ZLABEL as label
            FROM ZABCDEMAILADDRESS
            WHERE ZADDRESS IS NOT NULL
        """)

        for row in cursor.fetchall():
            record_id = row["record_id"]
            if record_id not in records:
                continue

            email = row["email"]
            if email:
                name = records[record_id]["name"]
                if name not in contacts:
                    contacts[name] = {
                        "name": name,
                        "phones": [],
                        "emails": [],
                        "is_group": False,
                        "source": source_name,
                    }
                email_lower = email.lower()
                if email_lower not in contacts[name]["emails"]:
                    contacts[name]["emails"].append(email_lower)

        conn.close()
        print(f"  Loaded from {source_name}: {len(records)} records")

    except sqlite3.Error as e:
        print(f"  SQLite error for {source_name}: {e}")
    except PermissionError:
        print(f"  Permission denied for {source_name}. Enable Full Disk Access.")


def merge_with_existing(contacts: dict) -> dict:
    """Merge extracted contacts with existing contacts file."""
    if not EXISTING_FILE.exists():
        return contacts

    with open(EXISTING_FILE) as f:
        existing = json.load(f)

    # Add any contacts from existing file that aren't in AddressBook
    for name, info in existing.items():
        if name not in contacts:
            # Keep existing contact without phone info
            contacts[name] = {
                "name": name,
                "phones": [],
                "emails": [],
                "is_group": info.get("is_group", False),
                "from_imessage": True,
            }

    return contacts


def show_stats():
    """Show statistics about extracted contacts."""
    if not OUTPUT_FILE.exists():
        print("No contacts file found. Run: python scripts/extract_all_contacts.py")
        return

    with open(OUTPUT_FILE) as f:
        contacts = json.load(f)

    print("\n" + "=" * 60)
    print("CONTACT STATISTICS")
    print("=" * 60)

    total = len(contacts)
    with_phones = sum(1 for c in contacts.values() if c.get("phones"))
    with_emails = sum(1 for c in contacts.values() if c.get("emails"))
    groups = sum(1 for c in contacts.values() if c.get("is_group"))

    print(f"\nTotal contacts: {total}")
    print(f"  With phone numbers: {with_phones}")
    print(f"  With emails: {with_emails}")
    print(f"  Groups: {groups}")
    print(f"  Individuals: {total - groups}")

    # Sample contacts with phones
    print("\nSample contacts with phone numbers:")
    shown = 0
    for name, info in contacts.items():
        if info.get("phones") and shown < 15:
            phones = ", ".join(info["phones"][:2])
            print(f"  {name}: {phones}")
            shown += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--merge", action="store_true", help="Merge with existing contacts")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    print("\n" + "=" * 60)
    print("EXTRACTING CONTACTS FROM macOS ADDRESSBOOK")
    print("=" * 60)

    contacts = extract_from_addressbook()

    if args.merge:
        print("\nMerging with existing contacts...")
        contacts = merge_with_existing(contacts)

    if not contacts:
        print("No contacts extracted.")
        return

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(contacts, f, indent=2)

    # Stats
    with_phones = sum(1 for c in contacts.values() if c.get("phones"))
    with_emails = sum(1 for c in contacts.values() if c.get("emails"))

    print(f"\n{'-' * 60}")
    print(f"Extracted {len(contacts)} contacts")
    print(f"  With phone numbers: {with_phones}")
    print(f"  With emails: {with_emails}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
