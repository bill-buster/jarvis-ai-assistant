#!/usr/bin/env python3
"""Index all iMessage history for style learning.

This is a one-time setup that indexes your message history so JARVIS can:
1. Learn your texting style (lowercase, emojis, vocabulary)
2. Find your past replies to similar messages
3. Generate responses that match how YOU actually text

Run with: python -m v2.scripts.index_messages
"""

from __future__ import annotations

import sys
import time


def main():
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold blue]JARVIS v2 - Message Indexer[/bold blue]\n\n"
        "This indexes your iMessage history to learn your texting style.\n"
        "Your data stays 100% local - nothing is sent to the cloud.\n\n"
        "[dim]Embedding model: all-MiniLM-L6-v2 (~90MB, 384 dimensions)[/dim]",
        title="Style Learning",
    ))
    console.print()

    # Check for --yes flag to skip confirmation
    if "--yes" not in sys.argv and "-y" not in sys.argv:
        if not console.input("[yellow]Continue? [y/N][/yellow] ").strip().lower() == "y":
            console.print("[dim]Cancelled.[/dim]")
            return

    console.print()

    # First, count messages to give accurate estimate
    console.print("[dim]Scanning conversations...[/dim]")

    from core.imessage import MessageReader
    from core.embeddings import get_embedding_store, get_embedding_model

    reader = MessageReader()
    conversations = reader.get_conversations(limit=None)  # All conversations

    # Count total messages
    total_messages = 0
    conv_message_counts = []
    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=None)  # All messages
        valid = [m for m in messages if m.text and len(m.text.strip()) >= 3]
        conv_message_counts.append((conv, valid))
        total_messages += len(valid)

    # Estimate time (~1ms per message in batches, plus model load)
    est_seconds = 5 + (total_messages * 0.001)  # 5s model load + ~1ms per msg
    est_minutes = est_seconds / 60

    console.print()
    table = Table(show_header=False, box=None)
    table.add_row("Conversations:", f"[cyan]{len(conversations)}[/cyan]")
    table.add_row("Total messages:", f"[cyan]{total_messages:,}[/cyan]")
    table.add_row("Estimated time:", f"[cyan]{est_minutes:.1f} minutes[/cyan]")
    console.print(table)
    console.print()

    # Load model first (show spinner)
    with console.status("[bold green]Loading embedding model...[/bold green]"):
        model = get_embedding_model()
        model._ensure_loaded()
    console.print("[green]✓[/green] Model loaded")

    # Index with progress bar
    store = get_embedding_store()
    indexed = 0
    skipped = 0
    duplicates = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Indexing messages...",
            total=total_messages,
        )

        for conv, messages in conv_message_counts:
            if not messages:
                continue

            # Convert to dict format
            msg_dicts = [
                {
                    "id": m.id,
                    "text": m.text,
                    "chat_id": m.chat_id,
                    "sender": m.sender,
                    "sender_name": m.sender_name,
                    "timestamp": m.timestamp,
                    "is_from_me": m.is_from_me,
                }
                for m in messages
            ]

            stats = store.index_messages(msg_dicts)
            indexed += stats["indexed"]
            skipped += stats["skipped"]
            duplicates += stats["duplicates"]

            progress.update(task, advance=len(messages))

    elapsed = time.time() - start_time

    # Results
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Indexing Complete![/green]\n\n"
        f"Messages indexed: [cyan]{indexed:,}[/cyan]\n"
        f"Already indexed: [dim]{duplicates:,}[/dim]\n"
        f"Skipped (too short): [dim]{skipped:,}[/dim]\n"
        f"Time: [cyan]{elapsed:.1f} seconds[/cyan]\n\n"
        f"Database: [dim]~/.jarvis/embeddings.db[/dim]",
        title="Results",
    ))

    if indexed > 0:
        console.print()
        console.print("[green]Your message history is now indexed![/green]")
        console.print("JARVIS will use your past replies to match your texting style.")
    console.print()


if __name__ == "__main__":
    main()
