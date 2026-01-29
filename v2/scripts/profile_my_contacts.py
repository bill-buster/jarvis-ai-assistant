#!/usr/bin/env python3
"""Create detailed contact profiles for personalized reply generation.

Instead of broad categories like "family", capture specific relationships:
- "Dad" → dad, formal-casual, respectful
- "Mihir Shah" → brother, very casual, jokes around
- "Mihir" → close friend, casual

This enables:
1. Accurate relationship context in prompts
2. Group chat inference from members
3. Personalized tone/style per contact

Usage:
    python scripts/profile_my_contacts.py                # Interactive profiling
    python scripts/profile_my_contacts.py --stats        # Show progress
    python scripts/profile_my_contacts.py --infer-groups # Infer group relationships
    python scripts/profile_my_contacts.py --export       # Export for use in generation
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_DATA = Path("results/test_set/clean_test_data.jsonl")
ALL_CONTACTS = Path("results/contacts/all_contacts_with_names.json")
PROFILES_FILE = Path("results/contacts/contact_profiles.json")

# Specific relationships (more granular than categories)
# You can also type ANY custom relationship - these are just shortcuts
RELATIONSHIPS = {
    # Family
    "dad": {"category": "family", "typical_tone": "respectful but casual"},
    "mom": {"category": "family", "typical_tone": "warm, caring"},
    "brother": {"category": "family", "typical_tone": "casual, joking"},
    "sister": {"category": "family", "typical_tone": "casual, supportive"},
    "cousin": {"category": "family", "typical_tone": "friendly casual"},
    "uncle": {"category": "family", "typical_tone": "respectful"},
    "aunt": {"category": "family", "typical_tone": "respectful warm"},
    "grandparent": {"category": "family", "typical_tone": "respectful loving"},
    "family_other": {"category": "family", "typical_tone": "respectful"},

    # Friends
    "best_friend": {"category": "friend", "typical_tone": "very casual, inside jokes"},
    "close_friend": {"category": "friend", "typical_tone": "casual, comfortable"},
    "friend": {"category": "friend", "typical_tone": "friendly casual"},
    "acquaintance": {"category": "acquaintance", "typical_tone": "polite casual"},

    # Family friends (not actual family but close like family)
    "family_friend": {"category": "family_friend", "typical_tone": "warm, respectful casual"},
    "parents_friend": {"category": "family_friend", "typical_tone": "polite, respectful"},
    "uncle_friend": {"category": "family_friend", "typical_tone": "respectful casual"},  # Not real uncle
    "auntie_friend": {"category": "family_friend", "typical_tone": "respectful casual"},  # Not real aunt

    # Romantic
    "partner": {"category": "romantic", "typical_tone": "intimate, affectionate"},
    "dating": {"category": "romantic", "typical_tone": "flirty, interested"},
    "ex": {"category": "romantic", "typical_tone": "varies"},

    # Work/School
    "coworker": {"category": "work", "typical_tone": "professional friendly"},
    "boss": {"category": "work", "typical_tone": "professional respectful"},
    "classmate": {"category": "school", "typical_tone": "casual friendly"},
    "professor": {"category": "school", "typical_tone": "formal respectful"},

    # Other
    "service": {"category": "other", "typical_tone": "transactional"},
    "unknown": {"category": "other", "typical_tone": "neutral"},
}

RELATIONSHIP_SHORTCUTS = {
    "d": "dad", "m": "mom", "b": "brother", "s": "sister",
    "c": "cousin", "u": "uncle", "a": "aunt", "g": "grandparent",
    "bf": "best_friend", "cf": "close_friend", "f": "friend",
    "aq": "acquaintance", "p": "partner", "dt": "dating",
    "cw": "coworker", "bo": "boss", "cl": "classmate",
    "ff": "family_friend", "pf": "parents_friend",  # Family friends
    "uf": "uncle_friend", "af": "auntie_friend",    # Not real uncle/aunt
    "x": "unknown", "sk": "skip",
}


def load_contacts():
    """Load contacts from extracted contacts file (with resolved names)."""
    contacts = {}

    # First, load all contacts with resolved names
    if ALL_CONTACTS.exists():
        with open(ALL_CONTACTS) as f:
            all_contacts = json.load(f)

        for name, info in all_contacts.items():
            # Skip emails and weird names
            if "@" in name or name.startswith("urn:"):
                continue

            contacts[name] = {
                "name": name,
                "is_group": info.get("is_group", False),
                "sample_count": 0,
                "samples": [],
            }

    # Then enrich with sample messages from test data
    if CLEAN_DATA.exists():
        with open(CLEAN_DATA) as f:
            for line in f:
                d = json.loads(line)
                contact = d.get("contact", "unknown")

                if contact in contacts:
                    contacts[contact]["sample_count"] += 1
                    if len(contacts[contact]["samples"]) < 3:
                        contacts[contact]["samples"].append({
                            "last_msg": d.get("last_message", "")[:60],
                            "your_reply": d.get("gold_response", "")[:60],
                            "conversation": d.get("conversation", "")[-200:],
                        })

    # If no extracted contacts, fall back to test data only
    if not contacts:
        print("No extracted contacts found. Run: python scripts/extract_contacts_with_names.py")
        return {}

    return contacts


def load_profiles():
    """Load existing profiles."""
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {"individuals": {}, "groups": {}}


def save_profiles(profiles):
    """Save profiles."""
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def parse_group_members(group_name: str) -> list[str]:
    """Extract individual names from group chat name."""
    # Handle "+N" suffix (e.g., "Het Patel, Meethre Bharot +3")
    name = re.sub(r'\s*\+\d+$', '', group_name)
    # Split by comma
    members = [m.strip() for m in name.split(',')]
    return [m for m in members if m]


def find_matching_profile(name: str, profiles: dict) -> tuple[str, float]:
    """Find best matching profile for a name using fuzzy matching."""
    best_match = None
    best_score = 0

    name_lower = name.lower()

    for profile_name in profiles.get("individuals", {}).keys():
        profile_lower = profile_name.lower()

        # Exact match
        if name_lower == profile_lower:
            return profile_name, 1.0

        # Check if one contains the other
        if name_lower in profile_lower or profile_lower in name_lower:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = profile_name

        # Fuzzy match
        score = SequenceMatcher(None, name_lower, profile_lower).ratio()
        if score > best_score and score > 0.6:
            best_score = score
            best_match = profile_name

    return best_match, best_score


def infer_group_relationship(group_name: str, profiles: dict) -> dict:
    """Infer group relationship from member profiles."""
    members = parse_group_members(group_name)
    member_profiles = []

    for member in members:
        match, score = find_matching_profile(member, profiles)
        if match and score > 0.6:
            member_profiles.append(profiles["individuals"][match])

    if not member_profiles:
        return {"relationship": "unknown", "category": "unknown", "inferred_from": []}

    # Determine dominant category
    categories = [p.get("category", "unknown") for p in member_profiles]
    relationships = [p.get("relationship", "unknown") for p in member_profiles]

    category_counts = Counter(categories)
    dominant_category = category_counts.most_common(1)[0][0]

    # Infer group type
    if dominant_category == "family":
        group_rel = "family_group"
    elif dominant_category == "friend":
        group_rel = "friend_group"
    elif dominant_category == "work":
        group_rel = "work_group"
    elif len(set(categories)) > 1:
        group_rel = "mixed_group"
    else:
        group_rel = f"{dominant_category}_group"

    return {
        "relationship": group_rel,
        "category": dominant_category,
        "member_relationships": relationships,
        "inferred_from": members,
    }


def run_profiling():
    """Interactive contact profiling."""
    print("\n" + "=" * 70)
    print("CONTACT PROFILING")
    print("=" * 70)
    print("""
Create detailed profiles for your contacts.

Relationship types (shortcuts in parentheses):
  FAMILY:  dad(d), mom(m), brother(b), sister(s), cousin(c),
           uncle(u), aunt(a), grandparent(g), family_other
  FRIENDS: best_friend(bf), close_friend(cf), friend(f), acquaintance(aq)
  ROMANTIC: partner(p), dating(dt), ex
  WORK:    coworker(cw), boss(bo), classmate(cl), professor
  OTHER:   unknown(x), skip(sk)

You can also add custom notes about how you text this person.
""")

    contacts = load_contacts()
    profiles = load_profiles()

    # Separate individuals and groups
    individuals = {k: v for k, v in contacts.items() if not v["is_group"]}
    groups = {k: v for k, v in contacts.items() if v["is_group"]}

    # Sort by sample count
    sorted_individuals = sorted(individuals.items(), key=lambda x: -x[1]["sample_count"])

    profiled = set(profiles.get("individuals", {}).keys())
    unprofiled = [(n, i) for n, i in sorted_individuals if n not in profiled]

    print(f"\nIndividuals: {len(profiled)}/{len(individuals)} profiled")
    print(f"Groups: {len(groups)} (will be inferred from members)")
    print(f"\nCommands: [relationship shortcut], [enter]=skip, [quit]=exit\n")

    for idx, (name, info) in enumerate(unprofiled):
        print("\n" + "-" * 70)
        print(f"[{len(profiled) + idx + 1}/{len(individuals)}] {name}")
        print(f"  Messages: {info['sample_count']}")

        # Show sample conversation
        if info["samples"]:
            print(f"\n  Sample exchanges:")
            for s in info["samples"][:2]:
                print(f"    them: \"{s['last_msg']}\"")
                print(f"    you:  \"{s['your_reply']}\"")

        # Check for fuzzy match to existing profile
        match, score = find_matching_profile(name, profiles)
        if match and score > 0.7:
            print(f"\n  ⚠️  Similar to existing profile: {match}")
            use_existing = input(f"  Use same profile? (y/n): ").strip().lower()
            if use_existing == 'y':
                profiles["individuals"][name] = profiles["individuals"][match].copy()
                profiles["individuals"][name]["alias_of"] = match
                print(f"  → Linked to {match}")
                continue

        print(f"\n  SHORTCUTS: d=dad m=mom b=brother s=sister c=cousin u=uncle a=aunt")
        print(f"  bf=best_friend cf=close_friend f=friend ff=family_friend")
        print(f"  cw=coworker cl=classmate x=unknown sk=skip")
        print(f"  OR type any custom relationship (e.g., 'childhood_friend', 'gym_buddy')")

        while True:
            rel_input = input("  Relationship: ").strip()

            if rel_input.lower() == "quit":
                save_profiles(profiles)
                print(f"\nSaved {len(profiles['individuals'])} profiles.")
                return

            if rel_input == "" or rel_input.lower() == "sk":
                break

            # Resolve shortcut (case-insensitive for shortcuts)
            relationship = RELATIONSHIP_SHORTCUTS.get(rel_input.lower(), rel_input)

            # Accept ANY input - custom relationships are allowed
            if relationship not in RELATIONSHIPS:
                # Create a custom relationship entry
                print(f"  Custom relationship: '{relationship}'")
                category = input("  Category (family/friend/work/other) [friend]: ").strip().lower() or "friend"
                RELATIONSHIPS[relationship] = {"category": category, "typical_tone": "custom"}

            # Get optional notes
            notes = input("  Notes (how you text them, optional): ").strip()

            rel_info = RELATIONSHIPS[relationship]
            profiles["individuals"][name] = {
                "relationship": relationship,
                "category": rel_info["category"],
                "typical_tone": rel_info["typical_tone"],
                "notes": notes if notes else None,
            }

            print(f"  → {name}: {relationship} ({rel_info['category']})")
            break

    save_profiles(profiles)
    print(f"\n✓ Saved {len(profiles['individuals'])} profiles")

    # Offer to infer groups
    if groups:
        print(f"\nFound {len(groups)} group chats. Infer relationships from members?")
        infer = input("Infer group relationships? (y/n): ").strip().lower()
        if infer == 'y':
            infer_groups(profiles, groups)


def infer_groups(profiles: dict = None, groups: dict = None):
    """Infer group chat relationships from member profiles."""
    if profiles is None:
        profiles = load_profiles()
    if groups is None:
        contacts = load_contacts()
        groups = {k: v for k, v in contacts.items() if v["is_group"]}

    print("\n" + "=" * 70)
    print("GROUP INFERENCE")
    print("=" * 70)

    if "groups" not in profiles:
        profiles["groups"] = {}

    inferred = 0
    for group_name, info in groups.items():
        result = infer_group_relationship(group_name, profiles)

        if result["relationship"] != "unknown":
            profiles["groups"][group_name] = result
            inferred += 1
            print(f"\n  {group_name}")
            print(f"    → {result['relationship']} (from: {result['inferred_from']})")

    save_profiles(profiles)
    print(f"\n✓ Inferred {inferred}/{len(groups)} group relationships")


def show_stats():
    """Show profiling progress."""
    contacts = load_contacts()
    profiles = load_profiles()

    individuals = {k: v for k, v in contacts.items() if not v["is_group"]}
    groups = {k: v for k, v in contacts.items() if v["is_group"]}

    print("\n" + "=" * 70)
    print("PROFILE STATS")
    print("=" * 70)

    ind_profiled = len(profiles.get("individuals", {}))
    grp_profiled = len(profiles.get("groups", {}))

    print(f"\nIndividuals: {ind_profiled}/{len(individuals)} profiled")
    print(f"Groups: {grp_profiled}/{len(groups)} inferred")

    # Coverage
    total_samples = sum(c["sample_count"] for c in contacts.values())
    covered_samples = sum(
        contacts[name]["sample_count"]
        for name in profiles.get("individuals", {}).keys()
        if name in contacts
    )
    covered_samples += sum(
        contacts[name]["sample_count"]
        for name in profiles.get("groups", {}).keys()
        if name in contacts
    )

    print(f"\nSample coverage: {covered_samples}/{total_samples} ({covered_samples/total_samples*100:.0f}%)")

    # Relationship distribution
    if profiles.get("individuals"):
        print(f"\nRelationship distribution:")
        rels = Counter(p.get("relationship") for p in profiles["individuals"].values())
        for rel, count in rels.most_common():
            print(f"  {rel}: {count}")

    # Show some profiles
    if profiles.get("individuals"):
        print(f"\nSample profiles:")
        for name, p in list(profiles["individuals"].items())[:5]:
            notes = f" - {p['notes']}" if p.get('notes') else ""
            print(f"  {name}: {p['relationship']}{notes}")


def export_for_generation():
    """Export profiles in format ready for prompt generation."""
    profiles = load_profiles()

    print("\n" + "=" * 70)
    print("PROFILE EXPORT (for prompts)")
    print("=" * 70)

    for name, p in profiles.get("individuals", {}).items():
        relationship = p.get("relationship", "unknown")
        notes = p.get("notes", "")

        # Generate prompt snippet
        if relationship in ["dad", "mom"]:
            prompt = f"You are texting your {relationship}."
        elif relationship == "brother":
            prompt = f"You are texting your brother {name.split()[0]}."
        elif relationship == "sister":
            prompt = f"You are texting your sister {name.split()[0]}."
        elif relationship in ["best_friend", "close_friend"]:
            prompt = f"You are texting your close friend {name.split()[0]}."
        elif relationship == "partner":
            prompt = f"You are texting your partner."
        else:
            prompt = f"You are texting {name.split()[0]} ({relationship})."

        if notes:
            prompt += f" {notes}"

        print(f"\n{name}:")
        print(f"  Prompt: \"{prompt}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--infer-groups", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.infer_groups:
        infer_groups()
    elif args.export:
        export_for_generation()
    else:
        run_profiling()


if __name__ == "__main__":
    main()
