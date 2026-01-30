"""Style-aware prompt generation based on contact clusters and user style.

Uses clustering data + style analysis to create personalized prompts.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CLUSTERS_FILE = Path("results/clustering/clusters.json")

# Your personal style profile (from analysis)
USER_STYLE = {
    "no_punctuation": 0.95,  # 95% of messages have no ending punctuation
    "uses_abbreviations": True,  # u, ur, rn, tmrw, idk
    "common_starters": ["i", "no", "oh", "ok", "what", "yea", "lol"],
    "avg_length": 28,
    "emoji_rate": 0.01,  # Rarely uses emoji
    "lol_rate": 0.03,
}

# Cluster-specific style overrides
CLUSTER_STYLES = {
    0: {  # Casual friends (Fantasy Ball, etc.)
        "name": "casual_friends",
        "description": "casual friend group chat",
        "style_notes": "brief, casual, uses abbreviations",
        "avg_length": 29,
        "lol_rate": 0.02,
    },
    1: {  # More formal (Dad, Soham, etc.)
        "name": "formal_contacts",
        "description": "family or formal contact",
        "style_notes": "slightly longer, more complete thoughts",
        "avg_length": 38,
        "lol_rate": 0.01,
    },
    2: {  # Family chats (longest messages)
        "name": "family",
        "description": "family group chat",
        "style_notes": "longer messages, complete sentences, no lol",
        "avg_length": 45,
        "lol_rate": 0.0,
    },
    3: {  # Casual friends (uses lol)
        "name": "playful_friends",
        "description": "close casual friend",
        "style_notes": "brief, playful, uses lol/haha, abbreviations",
        "avg_length": 26,
        "lol_rate": 0.04,
    },
    4: {  # Close friends
        "name": "close_friends",
        "description": "close friend",
        "style_notes": "medium length, casual, uses abbreviations",
        "avg_length": 31,
        "lol_rate": 0.03,
    },
    5: {  # Group chats (shortest)
        "name": "group_chats",
        "description": "friend group chat",
        "style_notes": "very brief, quick responses, minimal",
        "avg_length": 24,
        "lol_rate": 0.02,
    },
}


@dataclass
class StyleContext:
    """Style context for generating a reply."""
    contact_name: str
    cluster_id: int
    cluster_name: str
    style_notes: str
    target_length: str  # "brief", "medium", "longer"
    use_lol: bool
    use_abbreviations: bool


def load_contact_clusters() -> dict[str, int]:
    """Load contact -> cluster mapping."""
    if not CLUSTERS_FILE.exists():
        return {}

    with open(CLUSTERS_FILE) as f:
        data = json.load(f)

    return {c["name"]: c["cluster"] for c in data.get("contacts", [])}


# Cache the mapping
_contact_clusters: dict[str, int] | None = None


def get_contact_cluster(contact_name: str) -> int | None:
    """Get cluster ID for a contact."""
    global _contact_clusters

    if _contact_clusters is None:
        _contact_clusters = load_contact_clusters()

    # Try exact match
    if contact_name in _contact_clusters:
        return _contact_clusters[contact_name]

    # Try partial match
    for name, cluster in _contact_clusters.items():
        if contact_name.lower() in name.lower() or name.lower() in contact_name.lower():
            return cluster

    return None


def get_style_context(contact_name: str) -> StyleContext:
    """Get style context for a contact."""
    cluster_id = get_contact_cluster(contact_name)

    if cluster_id is not None and cluster_id in CLUSTER_STYLES:
        cluster = CLUSTER_STYLES[cluster_id]
        avg_len = cluster["avg_length"]

        if avg_len < 27:
            target_length = "brief"
        elif avg_len > 40:
            target_length = "longer"
        else:
            target_length = "medium"

        return StyleContext(
            contact_name=contact_name,
            cluster_id=cluster_id,
            cluster_name=cluster["name"],
            style_notes=cluster["style_notes"],
            target_length=target_length,
            use_lol=cluster["lol_rate"] > 0.02,
            use_abbreviations=True,  # You always use abbreviations
        )

    # Default style
    return StyleContext(
        contact_name=contact_name,
        cluster_id=-1,
        cluster_name="unknown",
        style_notes="brief, casual",
        target_length="brief",
        use_lol=False,
        use_abbreviations=True,
    )


def build_style_prompt(contact_name: str, conversation: list[dict]) -> str:
    """Build a style-aware prompt for reply generation.

    Args:
        contact_name: Name of the contact
        conversation: List of {"text": "...", "is_from_me": bool}

    Returns:
        Formatted prompt with style instructions
    """
    style = get_style_context(contact_name)

    # Build style instruction
    style_parts = []

    # Length
    if style.target_length == "brief":
        style_parts.append("very brief (under 30 chars)")
    elif style.target_length == "longer":
        style_parts.append("complete sentence")
    else:
        style_parts.append("brief")

    # Tone
    if style.cluster_name == "family":
        style_parts.append("warm")
    elif style.cluster_name in ["playful_friends", "casual_friends"]:
        style_parts.append("casual")
    else:
        style_parts.append("friendly")

    # Specific style markers
    style_parts.append("no ending punctuation")

    if style.use_abbreviations:
        style_parts.append("use u/ur not you/your")

    if style.use_lol:
        style_parts.append("can use lol/haha")

    style_instruction = f"[{', '.join(style_parts)}]"

    # Format conversation
    lines = []
    for msg in conversation[-12:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            lines.append(f"{prefix} {text}")

    conversation_text = "\n".join(lines)

    return f"{style_instruction}\n\n{conversation_text}\nme:"


def build_detailed_prompt(contact_name: str, conversation: list[dict], include_examples: bool = False) -> str:
    """Build a more detailed prompt with explicit style instructions.

    This version is more explicit about matching the user's texting style.
    """
    style = get_style_context(contact_name)

    # System instruction
    system = f"""You are texting as the user. Match their EXACT style:
- NO ending punctuation (95% of their messages have none)
- Use abbreviations: u, ur, rn, tmrw, idk, nvm
- Keep it {style.target_length} (~{CLUSTER_STYLES.get(style.cluster_id, {}).get('avg_length', 28)} chars)
- Start messages naturally with: I, No, Oh, Ok, What, Yea
- {"Can use lol/haha occasionally" if style.use_lol else "Don't use lol/haha"}
- This is a {style.cluster_name.replace('_', ' ')} - {style.style_notes}

Reply naturally as them. Just the reply text, nothing else."""

    # Format conversation
    lines = []
    for msg in conversation[-12:]:
        text = msg.get("text", "").strip()
        if text:
            prefix = "me:" if msg.get("is_from_me") else "them:"
            lines.append(f"{prefix} {text}")

    conversation_text = "\n".join(lines)

    return f"{system}\n\nConversation:\n{conversation_text}\n\nReply:"


# Quick test
if __name__ == "__main__":
    # Test with a contact
    test_conv = [
        {"text": "hey what are you up to", "is_from_me": False},
        {"text": "nm just working", "is_from_me": True},
        {"text": "wanna grab dinner later", "is_from_me": False},
    ]

    contacts = ["Mihir Shah", "Mom", "Faith", "Unknown Person"]

    for contact in contacts:
        style = get_style_context(contact)
        print(f"\n{contact}: cluster={style.cluster_id} ({style.cluster_name})")
        print(f"  Style: {style.style_notes}")
        print(f"  Length: {style.target_length}, lol: {style.use_lol}")

        prompt = build_style_prompt(contact, test_conv)
        print(f"  Prompt preview: {prompt[:80]}...")
