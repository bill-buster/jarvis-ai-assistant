#!/usr/bin/env python3
"""Re-index all messages with full text (no truncation).

This script:
1. Backs up the existing embeddings database
2. Clears and rebuilds with full message text
3. Shows progress

Run with: uv run python scripts/reindex_full_text.py
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings


def main():
    print("=" * 60)
    print("JARVIS v3 - Full Text Re-indexer")
    print("=" * 60)
    print()
    print("This will re-index all messages with FULL text (no 200 char truncation)")
    print()

    db_path = settings.embeddings.db_path

    if db_path.exists():
        # Backup existing
        backup_path = db_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        print(f"📦 Backing up existing database to: {backup_path.name}")
        shutil.copy(db_path, backup_path)

        # Delete existing
        print(f"🗑️  Removing old database...")
        db_path.unlink()

        # Also remove FAISS indices
        faiss_dir = settings.embeddings.faiss_cache_dir
        if faiss_dir.exists():
            print(f"🗑️  Removing old FAISS indices...")
            shutil.rmtree(faiss_dir)

    print()
    print("🔄 Starting fresh index with full text...")
    print()

    # Now run the regular indexer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    console = Console()

    # Count messages
    console.print("[dim]Scanning conversations...[/dim]")

    from core.embeddings import get_embedding_store
    from core.imessage import MessageReader

    reader = MessageReader()
    conversations = reader.get_conversations(limit=None)

    total_messages = 0
    conv_message_counts = []
    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=None)
        valid = [m for m in messages if m.text and len(m.text.strip()) >= 3]
        conv_message_counts.append((conv, valid))
        total_messages += len(valid)

    console.print()
    table = Table(show_header=False, box=None)
    table.add_row("Conversations:", f"[cyan]{len(conversations)}[/cyan]")
    table.add_row("Total messages:", f"[cyan]{total_messages:,}[/cyan]")
    table.add_row("Estimated time:", f"[cyan]{total_messages * 0.001 / 60:.1f} minutes[/cyan]")
    console.print(table)
    console.print()

    # Index with progress
    store = get_embedding_store()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing messages...", total=len(conv_message_counts))

        total_indexed = 0
        for conv, messages in conv_message_counts:
            if messages:
                # Convert to dict format for store
                msg_dicts = [
                    {
                        "id": m.id,
                        "chat_id": m.chat_id,
                        "text": m.text,  # FULL TEXT - no truncation!
                        "sender": m.sender,
                        "sender_name": m.sender_name,
                        "timestamp": m.timestamp,
                        "is_from_me": m.is_from_me,
                    }
                    for m in messages
                ]

                stats = store.index_messages(msg_dicts)
                total_indexed += stats.get("indexed", 0)

            progress.update(task, advance=1)

    console.print()
    console.print(f"[green]✅ Indexed {total_indexed:,} messages with full text![/green]")
    console.print()
    console.print("[dim]You can now run evaluate_replies.py to test the improvements.[/dim]")

    reader.close()


if __name__ == "__main__":
    main()
