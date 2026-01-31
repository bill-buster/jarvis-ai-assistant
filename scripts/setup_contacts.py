#!/usr/bin/env python3
"""Setup contacts with auto-classification and style embeddings.

This script:
1. Gets all 1:1 contacts from iMessage chat.db
2. Runs the relationship classifier on each
3. Computes a "style embedding" for each contact
4. Stores everything in the contacts table
5. Links existing pairs to their contacts

Run: uv run python scripts/setup_contacts.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHAT_DB = Path.home() / "Library/Messages/chat.db"


def get_all_1to1_chats(min_messages: int = 20) -> list[dict]:
    """Get all 1:1 chats from iMessage database.

    Args:
        min_messages: Minimum messages to include a chat.

    Returns:
        List of chat info dicts with chat_id, display_name, message_count.
    """
    if not CHAT_DB.exists():
        logger.error(f"Chat database not found: {CHAT_DB}")
        return []

    chats = []
    try:
        conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row

        # Get 1:1 chats with message counts
        query = """
            SELECT
                c.chat_identifier,
                c.display_name,
                (SELECT COUNT(*) FROM chat_handle_join WHERE chat_id = c.ROWID) as handle_count,
                (SELECT COUNT(*) FROM chat_message_join WHERE chat_id = c.ROWID) as msg_count
            FROM chat c
            WHERE handle_count = 1
            ORDER BY msg_count DESC
        """

        for row in conn.execute(query):
            if row["msg_count"] >= min_messages:
                chats.append(
                    {
                        "chat_id": row["chat_identifier"],
                        "display_name": row["display_name"] or row["chat_identifier"],
                        "message_count": row["msg_count"],
                    }
                )

        conn.close()

    except Exception as e:
        logger.error(f"Failed to get chats: {e}")

    return chats


def get_sample_messages(chat_id: str, limit: int = 100) -> list[str]:
    """Get sample messages FROM the contact (not from me).

    These are used to compute their "style" - how THEY text.
    """
    if not CHAT_DB.exists():
        return []

    messages = []
    try:
        conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT m.text
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE c.chat_identifier = ?
                AND m.is_from_me = 0
                AND m.text IS NOT NULL
                AND m.text != ''
                AND LENGTH(m.text) > 5
            ORDER BY m.date DESC
            LIMIT ?
        """

        for row in conn.execute(query, (chat_id, limit)):
            messages.append(row["text"])

        conn.close()

    except Exception as e:
        logger.error(f"Failed to get messages for {chat_id}: {e}")

    return messages


def compute_style_embedding(messages: list[str], embedder) -> np.ndarray | None:
    """Compute an average embedding representing the contact's texting style."""
    if not messages:
        return None

    # Sample up to 50 messages
    sample = messages[:50]

    try:
        embeddings = embedder.encode(sample, normalize=True)
        # Average embedding
        avg = np.mean(embeddings, axis=0)
        # Normalize
        avg = avg / np.linalg.norm(avg)
        return avg.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to compute embedding: {e}")
        return None


def setup_contacts(limit: int = 100, min_messages: int = 20) -> None:
    """Setup contacts with auto-classification and style embeddings.

    Args:
        limit: Maximum number of contacts to process.
        min_messages: Minimum messages to include a contact.
    """
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder
    from jarvis.relationship_classifier import RelationshipClassifier

    logger.info("=" * 60)
    logger.info("Setting Up Contacts with Auto-Classification")
    logger.info("=" * 60)

    # Get all 1:1 chats
    logger.info("\n📂 Getting 1:1 chats from iMessage...")
    all_chats = get_all_1to1_chats(min_messages=min_messages)
    logger.info(f"   Found {len(all_chats)} chats with >= {min_messages} messages")

    # Limit
    chats_to_process = all_chats[:limit]
    logger.info(f"   Processing top {len(chats_to_process)} chats")

    # Initialize components
    logger.info("\n🔍 Initializing classifier and embedder...")
    classifier = RelationshipClassifier(min_messages=min_messages)
    embedder = get_embedder()

    # Get database
    db = get_db()

    # Process each chat
    logger.info("\n🏃 Processing contacts...")
    results = []

    for i, chat_info in enumerate(chats_to_process):
        if (i + 1) % 10 == 0:
            logger.info(f"   Progress: {i + 1}/{len(chats_to_process)}")

        chat_id = chat_info["chat_id"]
        display_name = chat_info["display_name"]

        # Check if already exists
        existing = db.get_contact_by_chat_id(chat_id)
        if existing:
            results.append(
                {
                    "chat_id": chat_id,
                    "display_name": display_name,
                    "relationship": existing.relationship or "unknown",
                    "confidence": 1.0,
                    "message_count": chat_info["message_count"],
                    "status": "exists",
                }
            )
            continue

        # Classify relationship
        classification = classifier.classify_contact(chat_id, display_name)

        # Get sample messages and compute style embedding
        sample_messages = get_sample_messages(chat_id, limit=100)
        style_embedding = compute_style_embedding(sample_messages, embedder)

        # Create style notes
        style_notes = (
            f"Auto-classified: {classification.relationship} "
            f"({classification.confidence:.0%} confidence)"
        )
        if classification.features:
            style_notes += f"\nFeatures: {json.dumps(classification.features)}"

        # Add to database
        try:
            db.add_contact(
                display_name=display_name,
                chat_id=chat_id,
                phone_or_email=chat_id if chat_id.startswith("+") else None,
                relationship=classification.relationship,
                style_notes=style_notes,
            )
            status = "added"
        except Exception as e:
            logger.error(f"Failed to add contact {display_name}: {e}")
            status = "error"

        results.append(
            {
                "chat_id": chat_id,
                "display_name": display_name,
                "relationship": classification.relationship,
                "confidence": classification.confidence,
                "message_count": chat_info["message_count"],
                "has_style_embedding": style_embedding is not None,
                "status": status,
            }
        )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS")
    logger.info("=" * 60)

    # Count by status
    status_counts = Counter(r["status"] for r in results)
    logger.info(f"\nStatus: {dict(status_counts)}")

    # Distribution
    rel_counts: Counter = Counter(r["relationship"] for r in results)
    logger.info("\nRelationship Distribution:")
    for rel, count in rel_counts.most_common():
        pct = count / len(results) * 100 if results else 0
        logger.info(f"  {rel:<18} {count:>4} ({pct:>5.1f}%)")

    # Show top contacts
    logger.info(f"\nTop {min(25, len(results))} Contacts:")
    logger.info(f"{'Name':<25} {'Relationship':<15} {'Conf':<6} {'Msgs':<8} {'Status':<8}")
    logger.info("-" * 70)

    for r in results[:25]:
        name = r["display_name"][:24]
        logger.info(
            f"{name:<25} {r['relationship']:<15} {r['confidence']:<6.2f} "
            f"{r['message_count']:<8} {r['status']:<8}"
        )

    # Link pairs to contacts
    logger.info("\n🔗 Linking pairs to contacts...")

    with db.connection() as conn:
        # Get all contacts with chat_ids
        contacts_with_ids = conn.execute(
            "SELECT id, chat_id FROM contacts WHERE chat_id IS NOT NULL"
        ).fetchall()

        total_linked = 0
        for contact_id, chat_id in contacts_with_ids:
            cursor = conn.execute(
                "UPDATE pairs SET contact_id = ? WHERE chat_id = ? AND contact_id IS NULL",
                (contact_id, chat_id),
            )
            total_linked += cursor.rowcount

        conn.commit()
        logger.info(f"   Linked {total_linked} pairs to contacts")

    # Final stats
    logger.info("\n" + "=" * 60)
    logger.info("✅ DONE")
    logger.info("=" * 60)

    contacts = db.list_contacts()
    logger.info(f"Total contacts in DB: {len(contacts)}")

    with db.connection() as conn:
        pairs_with_contact = conn.execute(
            "SELECT COUNT(*) FROM pairs WHERE contact_id IS NOT NULL"
        ).fetchone()[0]
        total_pairs = conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]

    logger.info(f"Pairs linked to contacts: {pairs_with_contact}/{total_pairs}")


if __name__ == "__main__":
    setup_contacts(limit=100, min_messages=20)
