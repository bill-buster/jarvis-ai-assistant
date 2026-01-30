"""Build response templates from training pairs.

Analyzes your message patterns to find consistent responses
that can be used without calling the LLM.

Usage:
    python -m v2.scripts.build_response_templates
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def cluster_by_response(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group pairs by similar responses."""
    # Normalize responses
    response_groups = defaultdict(list)

    for pair in pairs:
        response = pair["output"].lower().strip()
        # Normalize common variants
        if response in {"yes", "yea", "yeah", "yep", "ya", "ye", "yup"}:
            response_groups["yes"].append(pair)
        elif response in {"no", "nah", "nope"}:
            response_groups["no"].append(pair)
        elif response in {"ok", "okay", "k", "kk"}:
            response_groups["ok"].append(pair)
        elif response in {"lol", "lmao", "haha", "hahaha", "😂"}:
            response_groups["lol"].append(pair)
        elif response in {"thanks", "thank you", "thx", "ty"}:
            response_groups["thanks"].append(pair)
        elif response in {"idk", "i don't know", "i dont know", "not sure"}:
            response_groups["idk"].append(pair)
        elif response in {"nice", "cool", "dope", "sick", "awesome"}:
            response_groups["nice"].append(pair)
        elif response in {"bet", "sounds good", "down", "im down", "i'm down"}:
            response_groups["bet"].append(pair)
        else:
            # Keep other responses as-is if they appear multiple times
            response_groups[response].append(pair)

    return response_groups


def build_templates(
    pairs_file: Path = Path("training_pairs.jsonl"),
    min_occurrences: int = 5,
) -> list[dict]:
    """Build response templates from training pairs.

    Args:
        pairs_file: Path to training pairs JSONL
        min_occurrences: Minimum times a response pattern must appear

    Returns:
        List of template dicts
    """
    # Load pairs
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    console.print(f"[blue]Loaded {len(pairs)} pairs[/blue]")

    # Group by response
    response_groups = cluster_by_response(pairs)

    # Filter to common responses
    common_responses = {
        k: v for k, v in response_groups.items()
        if len(v) >= min_occurrences
    }

    console.print(f"[green]Found {len(common_responses)} common response patterns[/green]")

    # Build templates
    templates = []

    for response, examples in sorted(common_responses.items(), key=lambda x: -len(x[1])):
        # Get sample inputs that triggered this response
        sample_inputs = [p["input"] for p in examples[:10]]

        # Find the most common actual response form
        actual_responses = [p["output"] for p in examples]
        most_common = max(set(actual_responses), key=actual_responses.count)

        templates.append({
            "response": response,  # Normalized form
            "actual": most_common,  # Your actual typical response
            "count": len(examples),
            "sample_triggers": sample_inputs,
        })

    return templates


def main():
    console.print("[bold]Building Response Templates[/bold]\n")

    pairs_file = Path("training_pairs.jsonl")
    if not pairs_file.exists():
        console.print("[red]training_pairs.jsonl not found. Run extract_training_pairs.py first.[/red]")
        return

    templates = build_templates(pairs_file, min_occurrences=10)

    # Display top templates
    table = Table(title="Your Most Common Responses")
    table.add_column("Response", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Sample Triggers", style="dim")

    for t in templates[:20]:
        triggers = ", ".join(t["sample_triggers"][:3])
        if len(triggers) > 50:
            triggers = triggers[:50] + "..."
        table.add_row(t["actual"], str(t["count"]), triggers)

    console.print(table)

    # Save templates
    output_file = Path("response_templates.json")
    with open(output_file, "w") as f:
        json.dump(templates, f, indent=2)

    console.print(f"\n[green]Saved {len(templates)} templates to {output_file}[/green]")

    # Show some interesting patterns
    console.print("\n[bold]Sample Patterns:[/bold]")
    for t in templates[:10]:
        console.print(f"\n[cyan]'{t['actual']}'[/cyan] ({t['count']} times)")
        console.print("  Triggered by:")
        for trigger in t["sample_triggers"][:5]:
            console.print(f"    - \"{trigger}\"")


if __name__ == "__main__":
    main()
