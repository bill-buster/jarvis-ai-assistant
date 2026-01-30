"""Extract message pairs for fine-tuning.

Creates training data from your actual conversations:
- Input: message you received
- Output: your actual reply

Usage:
    python -m v2.scripts.extract_training_pairs
    python -m v2.scripts.extract_training_pairs --output training_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()


def extract_pairs(
    output_path: Path,
    min_reply_length: int = 2,
    max_reply_length: int = 100,
    min_gap_seconds: int = 1,
    max_gap_seconds: int = 3600,  # 1 hour
) -> dict:
    """Extract message pairs from embedding store.

    Args:
        output_path: Where to save JSONL file
        min_reply_length: Minimum reply length to include
        max_reply_length: Maximum reply length (skip long messages)
        min_gap_seconds: Minimum time between their message and your reply
        max_gap_seconds: Maximum time gap (to ensure reply is related)

    Returns:
        Stats dict
    """
    from core.embeddings import get_embedding_store

    store = get_embedding_store()
    stats = {"total_pairs": 0, "skipped_short": 0, "skipped_long": 0, "by_chat": {}}

    # Get all messages ordered by chat and time
    with store._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT chat_id, text_preview, is_from_me, timestamp, sender_name
            FROM message_embeddings
            ORDER BY chat_id, timestamp
            """
        ).fetchall()

    console.print(f"[blue]Processing {len(rows)} messages...")

    pairs = []
    prev_row = None

    with Progress() as progress:
        task = progress.add_task("Extracting pairs...", total=len(rows))

        for row in rows:
            progress.update(task, advance=1)

            if prev_row is None:
                prev_row = row
                continue

            # Check if this is a reply to previous message
            same_chat = row["chat_id"] == prev_row["chat_id"]
            prev_from_them = prev_row["is_from_me"] == 0
            this_from_me = row["is_from_me"] == 1

            if same_chat and prev_from_them and this_from_me:
                gap = row["timestamp"] - prev_row["timestamp"]

                if min_gap_seconds <= gap <= max_gap_seconds:
                    their_msg = prev_row["text_preview"] or ""
                    my_reply = row["text_preview"] or ""

                    # Filter by length
                    if len(my_reply) < min_reply_length:
                        stats["skipped_short"] += 1
                    elif len(my_reply) > max_reply_length:
                        stats["skipped_long"] += 1
                    else:
                        # Skip iMessage reactions
                        if any(r in their_msg.lower() for r in ["loved ", "liked ", "emphasized ", "laughed at"]):
                            prev_row = row
                            continue
                        if any(r in my_reply.lower() for r in ["loved ", "liked ", "emphasized ", "laughed at"]):
                            prev_row = row
                            continue

                        pair = {
                            "input": their_msg,
                            "output": my_reply,
                            "chat_id": row["chat_id"],
                            "contact": prev_row["sender_name"] or "Unknown",
                            "gap_seconds": gap,
                        }
                        pairs.append(pair)
                        stats["total_pairs"] += 1

                        chat_id = row["chat_id"]
                        stats["by_chat"][chat_id] = stats["by_chat"].get(chat_id, 0) + 1

            prev_row = row

    # Write JSONL
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return stats, pairs


def main():
    parser = argparse.ArgumentParser(description="Extract message pairs for fine-tuning")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("training_pairs.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum reply length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum reply length",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="Number of sample pairs to display",
    )

    args = parser.parse_args()

    console.print("[bold]Extracting training pairs from your messages...[/bold]\n")

    stats, pairs = extract_pairs(
        output_path=args.output,
        min_reply_length=args.min_length,
        max_reply_length=args.max_length,
    )

    console.print(f"\n[green]Extracted {stats['total_pairs']} message pairs[/green]")
    console.print(f"[yellow]Skipped {stats['skipped_short']} short replies[/yellow]")
    console.print(f"[yellow]Skipped {stats['skipped_long']} long replies[/yellow]")
    console.print(f"\n[blue]Saved to: {args.output}[/blue]")

    # Show top chats
    console.print("\n[bold]Pairs by conversation:[/bold]")
    sorted_chats = sorted(stats["by_chat"].items(), key=lambda x: x[1], reverse=True)[:10]
    for chat_id, count in sorted_chats:
        console.print(f"  {chat_id[:30]}... : {count} pairs")

    # Show samples
    if pairs and args.show_samples > 0:
        console.print(f"\n[bold]Sample pairs:[/bold]")
        import random
        samples = random.sample(pairs, min(args.show_samples, len(pairs)))
        for i, pair in enumerate(samples, 1):
            console.print(f"\n[cyan]{i}. From {pair['contact']}:[/cyan]")
            console.print(f"   They: \"{pair['input']}\"")
            console.print(f"   You:  \"{pair['output']}\"")


if __name__ == "__main__":
    main()
