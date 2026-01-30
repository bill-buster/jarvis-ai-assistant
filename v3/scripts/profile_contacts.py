#!/usr/bin/env python3
"""Simple manual contact profiler with back button.

Just label who each person is to you - no automatic classification.
Tone/style will be inferred from actual conversations.

Usage:
    python scripts/profile_contacts.py              # Start profiling
    python scripts/profile_contacts.py --stats      # Show progress
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ALL_CONTACTS = Path("results/contacts/all_contacts_with_phones.json")
PROFILES_FILE = Path("results/contacts/contact_profiles.json")
CLEAN_DATA = Path("results/test_set/clean_test_data.jsonl")

# Shortcuts - but you can type ANYTHING
SHORTCUTS = {
    # Family
    "d": "dad", "m": "mom", "b": "brother", "s": "sister",
    "c": "cousin", "u": "uncle", "a": "aunt", "g": "grandparent",

    # Friends
    "bf": "best_friend", "cf": "close_friend", "f": "friend",
    "ff": "family_friend",  # Not actual family but close

    # Work/School
    "cw": "coworker", "cl": "classmate",

    # Other
    "x": "unknown", "sk": "skip",
}


def load_contacts():
    """Load all contacts with names and phone numbers."""
    contacts = []

    # Load extracted contacts
    if ALL_CONTACTS.exists():
        with open(ALL_CONTACTS) as f:
            all_contacts = json.load(f)

        for name, info in all_contacts.items():
            if "@" in name:  # Skip emails
                continue
            contacts.append({
                "name": name,
                "is_group": info.get("is_group", False),
                "phones": info.get("phones", []),
                "samples": [],
            })

    # Enrich with sample messages
    if CLEAN_DATA.exists():
        samples_by_contact = {}
        with open(CLEAN_DATA) as f:
            for line in f:
                d = json.loads(line)
                contact = d.get("contact", "")
                if contact not in samples_by_contact:
                    samples_by_contact[contact] = []
                if len(samples_by_contact[contact]) < 3:
                    samples_by_contact[contact].append({
                        "them": d.get("last_message", "")[:80],
                        "you": d.get("gold_response", "")[:80],
                    })

        for c in contacts:
            if c["name"] in samples_by_contact:
                c["samples"] = samples_by_contact[c["name"]]

    # Sort: contacts with samples first, then alphabetically
    contacts.sort(key=lambda x: (len(x["samples"]) == 0, x["name"].lower()))

    return contacts


def load_profiles():
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}


def save_profiles(profiles):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def get_category(relationship):
    """Infer category from relationship."""
    rel = relationship.lower()

    if rel in ["dad", "mom", "brother", "sister", "cousin", "uncle", "aunt",
               "grandparent", "grandma", "grandpa"]:
        return "family"
    elif rel in ["best_friend", "close_friend", "friend", "childhood_friend",
                 "college_friend", "roommate"]:
        return "friend"
    elif rel in ["family_friend", "parents_friend"]:
        return "family_friend"
    elif rel in ["coworker", "boss", "classmate", "professor", "colleague"]:
        return "work"
    elif rel in ["partner", "girlfriend", "boyfriend", "wife", "husband", "ex"]:
        return "romantic"
    else:
        return "other"


def run_profiling():
    """Interactive profiling with back button."""
    print("\n" + "=" * 70)
    print("CONTACT PROFILER")
    print("=" * 70)
    print("""
Label who each contact is to you.

SHORTCUTS:
  d=dad  m=mom  b=brother  s=sister  c=cousin  u=uncle  a=aunt
  bf=best_friend  cf=close_friend  f=friend  ff=family_friend
  cw=coworker  cl=classmate  x=unknown  sk=skip

OR just type anything: "childhood_friend", "gym_buddy", "roommate", etc.

COMMANDS:
  <enter> = skip this one
  back    = go back and redo previous
  quit    = save and exit
""")

    contacts = load_contacts()
    profiles = load_profiles()

    # Filter out already profiled
    unprofiled_indices = [i for i, c in enumerate(contacts) if c["name"] not in profiles]

    print(f"Total contacts: {len(contacts)}")
    print(f"Already profiled: {len(profiles)}")
    print(f"Remaining: {len(unprofiled_indices)}")

    if not unprofiled_indices:
        print("\nAll contacts profiled!")
        return

    # Track position and history for back button
    pos = 0  # Position in unprofiled_indices
    history = []  # Names we've profiled this session

    while pos < len(unprofiled_indices):
        idx = unprofiled_indices[pos]
        contact = contacts[idx]
        name = contact["name"]
        is_group = contact.get("is_group", False)
        samples = contact.get("samples", [])

        print("\n" + "-" * 70)
        group_marker = " (GROUP)" if is_group else ""
        phones = contact.get("phones", [])
        phone_str = f" | {phones[0]}" if phones else ""
        print(f"[{len(profiles) + 1}/{len(contacts)}] {name}{group_marker}{phone_str}")

        if samples:
            print("\n  Sample conversations:")
            for s in samples:
                print(f"    them: \"{s['them']}\"")
                print(f"    you:  \"{s['you']}\"")
        else:
            print("  (no sample messages)")

        print()
        rel_input = input("  Who is this? ").strip()

        # Handle commands
        if rel_input.lower() == "quit":
            save_profiles(profiles)
            print(f"\n✓ Saved {len(profiles)} profiles")
            return

        if rel_input.lower() == "back":
            if history:
                last_name = history.pop()
                if last_name in profiles:
                    del profiles[last_name]
                    print(f"  ← Removed {last_name}, going back")
                pos = max(0, pos - 1)
            else:
                print("  (nothing to go back to)")
            continue

        if rel_input == "" or rel_input.lower() == "sk":
            pos += 1
            continue

        # Resolve shortcut
        relationship = SHORTCUTS.get(rel_input.lower(), rel_input)
        category = get_category(relationship)

        # Save profile (include phones for matching)
        phones = contact.get("phones", [])
        profiles[name] = {
            "relationship": relationship,
            "category": category,
            "is_group": is_group,
            "phones": phones,
        }

        history.append(name)
        print(f"  → {name}: {relationship} ({category})")
        pos += 1

        # Auto-save every 10
        if len(history) % 10 == 0:
            save_profiles(profiles)

    save_profiles(profiles)
    print(f"\n✓ Done! Saved {len(profiles)} profiles")

    # Offer to infer groups
    groups = [c for c in contacts if c.get("is_group") and c["name"] not in profiles]
    if groups:
        print(f"\n{len(groups)} groups not profiled. Infer from members? (y/n)")
        if input().strip().lower() == 'y':
            infer_groups(profiles, contacts)


def infer_groups(profiles, contacts):
    """Infer group relationships from member profiles."""
    print("\n--- INFERRING GROUPS ---")

    for contact in contacts:
        if not contact.get("is_group"):
            continue
        if contact["name"] in profiles:
            continue

        name = contact["name"]

        # Parse members from group name
        import re
        members = re.split(r'[,+]', name)
        members = [m.strip() for m in members if m.strip() and not m.strip().isdigit()]

        # Find member profiles
        member_cats = []
        member_info = []
        for member in members[:5]:
            for profile_name, profile in profiles.items():
                if member.lower() in profile_name.lower() or profile_name.lower() in member.lower():
                    member_cats.append(profile.get("category", "other"))
                    member_info.append(f"{member}={profile.get('relationship')}")
                    break

        if not member_cats:
            continue

        # Determine group type
        from collections import Counter
        cat_counts = Counter(member_cats)
        dominant = cat_counts.most_common(1)[0][0]

        profiles[name] = {
            "relationship": f"{dominant}_group",
            "category": dominant,
            "is_group": True,
            "members": member_info,
        }

        print(f"  {name[:45]}")
        print(f"    → {dominant}_group ({', '.join(member_info[:3])})")

    save_profiles(profiles)
    print("\n✓ Saved group profiles")


def show_stats():
    """Show profiling stats."""
    contacts = load_contacts()
    profiles = load_profiles()

    print("\n" + "=" * 70)
    print("PROFILE STATS")
    print("=" * 70)

    print(f"\nTotal contacts: {len(contacts)}")
    print(f"Profiled: {len(profiles)} ({len(profiles)/max(len(contacts),1)*100:.0f}%)")

    from collections import Counter

    # By category
    cats = Counter(p.get("category", "other") for p in profiles.values())
    print("\nBy category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # By relationship
    rels = Counter(p.get("relationship", "?") for p in profiles.values())
    print("\nBy relationship:")
    for rel, count in rels.most_common(15):
        print(f"  {rel}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        run_profiling()


if __name__ == "__main__":
    main()
