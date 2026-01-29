#!/usr/bin/env python3
"""Fix/edit existing contact profiles.

Use this to correct mistakes from initial profiling.

Usage:
    python scripts/fix_profiles.py                    # List all profiles
    python scripts/fix_profiles.py --search "Shah"    # Find profiles by name
    python scripts/fix_profiles.py --edit "Mom"       # Edit specific profile
    python scripts/fix_profiles.py --category family  # List all family
    python scripts/fix_profiles.py --fix-groups       # Re-infer group relationships
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROFILES_FILE = Path("results/contacts/contact_profiles.json")


def load_profiles():
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {"individuals": {}, "groups": {}}


def save_profiles(profiles):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def list_profiles(category_filter=None):
    """List all profiles, optionally filtered by category."""
    profiles = load_profiles()
    individuals = profiles.get("individuals", {})
    groups = profiles.get("groups", {})

    print("\n" + "=" * 70)
    print("INDIVIDUAL PROFILES")
    print("=" * 70)

    # Group by category
    by_category = {}
    for name, info in individuals.items():
        cat = info.get("category", "other")
        if category_filter and cat != category_filter:
            continue
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((name, info))

    for cat in sorted(by_category.keys()):
        print(f"\n--- {cat.upper()} ---")
        for name, info in sorted(by_category[cat]):
            rel = info.get("relationship", "?")
            notes = info.get("notes", "")
            notes_str = f" ({notes})" if notes else ""
            print(f"  {name}: {rel}{notes_str}")

    print(f"\n" + "=" * 70)
    print(f"GROUP PROFILES ({len(groups)})")
    print("=" * 70)

    for name, info in sorted(groups.items()):
        rel = info.get("relationship", "?")
        members = info.get("inferred_from", [])
        print(f"  {name[:40]}: {rel}")
        if members:
            print(f"    from: {', '.join(members[:3])}")


def search_profiles(query):
    """Search profiles by name."""
    profiles = load_profiles()
    query_lower = query.lower()

    print(f"\nSearching for '{query}'...")

    found = []
    for name, info in profiles.get("individuals", {}).items():
        if query_lower in name.lower():
            found.append(("individual", name, info))

    for name, info in profiles.get("groups", {}).items():
        if query_lower in name.lower():
            found.append(("group", name, info))

    if not found:
        print("No matches found.")
        return

    print(f"\nFound {len(found)} matches:")
    for ptype, name, info in found:
        rel = info.get("relationship", "?")
        cat = info.get("category", "?")
        print(f"  [{ptype}] {name}: {rel} ({cat})")


def edit_profile(name):
    """Edit a specific profile."""
    profiles = load_profiles()

    # Find the profile
    if name in profiles.get("individuals", {}):
        profile = profiles["individuals"][name]
        ptype = "individual"
    elif name in profiles.get("groups", {}):
        profile = profiles["groups"][name]
        ptype = "group"
    else:
        # Try fuzzy match
        matches = [n for n in profiles.get("individuals", {}).keys() if name.lower() in n.lower()]
        matches += [n for n in profiles.get("groups", {}).keys() if name.lower() in n.lower()]

        if not matches:
            print(f"Profile '{name}' not found.")
            return

        print(f"Did you mean one of these?")
        for i, m in enumerate(matches[:10]):
            print(f"  {i+1}. {m}")

        choice = input("Enter number or 'n' to cancel: ").strip()
        if choice == 'n' or not choice.isdigit():
            return

        name = matches[int(choice) - 1]
        if name in profiles.get("individuals", {}):
            profile = profiles["individuals"][name]
            ptype = "individual"
        else:
            profile = profiles["groups"][name]
            ptype = "group"

    print(f"\n--- EDITING: {name} ({ptype}) ---")
    print(f"Current relationship: {profile.get('relationship', '?')}")
    print(f"Current category: {profile.get('category', '?')}")
    print(f"Current notes: {profile.get('notes', '')}")

    print(f"\nEnter new values (leave blank to keep current):")

    new_rel = input(f"  Relationship [{profile.get('relationship', '')}]: ").strip()
    if new_rel:
        profile["relationship"] = new_rel

    new_cat = input(f"  Category [{profile.get('category', '')}]: ").strip()
    if new_cat:
        profile["category"] = new_cat

    new_notes = input(f"  Notes [{profile.get('notes', '')}]: ").strip()
    if new_notes:
        profile["notes"] = new_notes

    # Save
    if ptype == "individual":
        profiles["individuals"][name] = profile
    else:
        profiles["groups"][name] = profile

    save_profiles(profiles)
    print(f"\n✓ Updated {name}")


def fix_groups():
    """Re-infer group relationships based on current individual profiles."""
    profiles = load_profiles()
    individuals = profiles.get("individuals", {})

    print("\n" + "=" * 70)
    print("RE-INFERRING GROUP RELATIONSHIPS")
    print("=" * 70)

    # Get all groups
    from scripts.profile_my_contacts import infer_group_relationship, load_contacts

    contacts = load_contacts()
    groups = {k: v for k, v in contacts.items() if v.get("is_group")}

    profiles["groups"] = {}

    for group_name in groups:
        result = infer_group_relationship(group_name, profiles)

        if result["relationship"] != "unknown":
            profiles["groups"][group_name] = result
            members = result.get("inferred_from", [])
            member_rels = result.get("member_relationships", [])
            print(f"\n  {group_name}")
            print(f"    → {result['relationship']}")
            print(f"    members: {', '.join(f'{m}({r})' for m, r in zip(members, member_rels))}")

    save_profiles(profiles)
    print(f"\n✓ Updated {len(profiles['groups'])} group profiles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=str, help="Search profiles by name")
    parser.add_argument("--edit", type=str, help="Edit a specific profile")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--fix-groups", action="store_true", help="Re-infer groups")
    args = parser.parse_args()

    if args.search:
        search_profiles(args.search)
    elif args.edit:
        edit_profile(args.edit)
    elif args.fix_groups:
        fix_groups()
    else:
        list_profiles(args.category)


if __name__ == "__main__":
    main()
