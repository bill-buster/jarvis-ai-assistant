"""Daily/Weekly digest generation for JARVIS.

Generates summaries including:
- Conversations with unread/unanswered messages
- Highlights from active group chats
- Detected action items and upcoming events
- Message volume statistics

Usage:
    from jarvis.digest import DigestGenerator, generate_digest

    generator = DigestGenerator(reader)
    digest = generator.generate(period="daily")

    # Or use convenience function
    digest = generate_digest(reader, period="weekly")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from contracts.imessage import Conversation, Message
from integrations.imessage import ChatDBReader

logger = logging.getLogger(__name__)


class DigestPeriod(str, Enum):
    """Time period for digest generation."""

    DAILY = "daily"
    WEEKLY = "weekly"


class DigestFormat(str, Enum):
    """Supported digest export formats."""

    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class UnansweredConversation:
    """A conversation with unanswered messages."""

    chat_id: str
    display_name: str
    participants: list[str]
    unanswered_count: int
    last_message_date: datetime | None
    last_message_preview: str | None
    is_group: bool


@dataclass
class GroupHighlight:
    """Highlight from an active group chat."""

    chat_id: str
    display_name: str
    participants: list[str]
    message_count: int
    active_participants: list[str]
    top_topics: list[str]
    last_activity: datetime | None


@dataclass
class ActionItem:
    """Detected action item from messages."""

    text: str
    chat_id: str
    conversation_name: str
    sender: str
    date: datetime
    message_id: int
    item_type: str  # "task", "question", "event", "reminder"


@dataclass
class MessageStats:
    """Message volume statistics."""

    total_sent: int
    total_received: int
    total_messages: int
    active_conversations: int
    most_active_conversation: str | None
    most_active_count: int
    avg_messages_per_day: float
    busiest_hour: int | None
    hourly_distribution: dict[int, int] = field(default_factory=dict)


@dataclass
class Digest:
    """Complete digest for a time period."""

    period: DigestPeriod
    generated_at: datetime
    start_date: datetime
    end_date: datetime
    needs_attention: list[UnansweredConversation]
    highlights: list[GroupHighlight]
    action_items: list[ActionItem]
    stats: MessageStats

    def to_dict(self) -> dict[str, Any]:
        """Convert digest to serializable dictionary."""
        return {
            "period": self.period.value,
            "generated_at": self.generated_at.isoformat(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "needs_attention": [
                {
                    "chat_id": c.chat_id,
                    "display_name": c.display_name,
                    "participants": c.participants,
                    "unanswered_count": c.unanswered_count,
                    "last_message_date": c.last_message_date.isoformat()
                    if c.last_message_date
                    else None,
                    "last_message_preview": c.last_message_preview,
                    "is_group": c.is_group,
                }
                for c in self.needs_attention
            ],
            "highlights": [
                {
                    "chat_id": h.chat_id,
                    "display_name": h.display_name,
                    "participants": h.participants,
                    "message_count": h.message_count,
                    "active_participants": h.active_participants,
                    "top_topics": h.top_topics,
                    "last_activity": h.last_activity.isoformat() if h.last_activity else None,
                }
                for h in self.highlights
            ],
            "action_items": [
                {
                    "text": a.text,
                    "chat_id": a.chat_id,
                    "conversation_name": a.conversation_name,
                    "sender": a.sender,
                    "date": a.date.isoformat(),
                    "message_id": a.message_id,
                    "item_type": a.item_type,
                }
                for a in self.action_items
            ],
            "stats": {
                "total_sent": self.stats.total_sent,
                "total_received": self.stats.total_received,
                "total_messages": self.stats.total_messages,
                "active_conversations": self.stats.active_conversations,
                "most_active_conversation": self.stats.most_active_conversation,
                "most_active_count": self.stats.most_active_count,
                "avg_messages_per_day": round(self.stats.avg_messages_per_day, 1),
                "busiest_hour": self.stats.busiest_hour,
                "hourly_distribution": self.stats.hourly_distribution,
            },
        }


# Action item detection patterns
ACTION_PATTERNS = {
    "task": [
        r"(?:can you|could you|please|pls|plz)\s+(.{10,60})\??",
        r"(?:need to|have to|must|should)\s+(.{10,60})",
        r"(?:don't forget|remember to)\s+(.{10,60})",
        r"(?:todo|to-do|to do)[:.]?\s*(.{5,60})",
    ],
    "question": [
        r"(?:^|\s)(?:what|when|where|why|how|who|which|can|could|would|will|do|does|is|are)\s+.{10,80}\?",
    ],
    "event": [
        r"(?:meeting|call|appointment|dinner|lunch|breakfast|party)\s+(?:at|on|tomorrow|tonight|today)",
        r"(?:tomorrow|tonight|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:at|@)\s*\d",
        r"\d{1,2}[:.]\d{2}\s*(?:am|pm|AM|PM)?",
    ],
    "reminder": [
        r"(?:remind me|reminder|don't forget|remember)\s+(.{10,60})",
        r"(?:by|before|deadline)\s+(?:tomorrow|tonight|today|monday|tuesday|wednesday|thursday|friday)",
    ],
}


class DigestGenerator:
    """Generates daily/weekly digests from iMessage data."""

    def __init__(self, reader: ChatDBReader):
        """Initialize the digest generator.

        Args:
            reader: ChatDBReader instance for accessing iMessage data.
        """
        self.reader = reader

    def generate(
        self,
        period: DigestPeriod | str = DigestPeriod.DAILY,
        end_date: datetime | None = None,
    ) -> Digest:
        """Generate a digest for the specified period.

        Args:
            period: Time period ("daily" or "weekly").
            end_date: End date for the digest (defaults to now).

        Returns:
            Digest containing all sections.
        """
        if isinstance(period, str):
            period = DigestPeriod(period)

        end = end_date or datetime.now()
        if period == DigestPeriod.DAILY:
            start = end - timedelta(days=1)
        else:
            start = end - timedelta(weeks=1)

        # Fetch conversations with activity in period
        conversations = self.reader.get_conversations(limit=200, since=start)

        # Generate each section
        needs_attention = self._find_unanswered(conversations, start, end)
        highlights = self._find_group_highlights(conversations, start, end)
        action_items = self._detect_action_items(conversations, start, end)
        stats = self._calculate_stats(conversations, start, end)

        return Digest(
            period=period,
            generated_at=datetime.now(),
            start_date=start,
            end_date=end,
            needs_attention=needs_attention,
            highlights=highlights,
            action_items=action_items,
            stats=stats,
        )

    def _find_unanswered(
        self,
        conversations: list[Conversation],
        start: datetime,
        end: datetime,
    ) -> list[UnansweredConversation]:
        """Find conversations with unanswered messages.

        Args:
            conversations: List of conversations to check.
            start: Start of the digest period.
            end: End of the digest period.

        Returns:
            List of conversations needing attention.
        """
        unanswered = []

        for conv in conversations:
            # Get recent messages
            messages = self.reader.get_messages(conv.chat_id, limit=20)
            if not messages:
                continue

            # Filter to messages in period
            period_messages = [m for m in messages if m.date and start <= m.date <= end]
            if not period_messages:
                continue

            # Check if last message is from someone else (not me)
            # and we haven't replied
            last_from_others = [m for m in period_messages if not m.is_from_me]
            if not last_from_others:
                continue

            # Find the most recent message from others
            last_other = max(last_from_others, key=lambda m: m.date or start)

            # Check if we've replied after their last message
            my_replies_after = [
                m
                for m in period_messages
                if m.is_from_me and m.date and last_other.date and m.date > last_other.date
            ]

            if not my_replies_after:
                # Count unanswered messages from others after our last reply
                my_last_reply = None
                for m in reversed(messages):
                    if m.is_from_me and m.date:
                        my_last_reply = m.date
                        break

                unanswered_count = len(
                    [
                        m
                        for m in last_from_others
                        if my_last_reply is None or (m.date and m.date > my_last_reply)
                    ]
                )

                if unanswered_count > 0:
                    unanswered.append(
                        UnansweredConversation(
                            chat_id=conv.chat_id,
                            display_name=conv.display_name or ", ".join(conv.participants[:2]),
                            participants=conv.participants,
                            unanswered_count=unanswered_count,
                            last_message_date=last_other.date,
                            last_message_preview=last_other.text[:100] if last_other.text else None,
                            is_group=conv.is_group,
                        )
                    )

        # Sort by unanswered count and recency
        unanswered.sort(
            key=lambda x: (x.unanswered_count, x.last_message_date or start),
            reverse=True,
        )

        return unanswered[:10]  # Limit to top 10

    def _find_group_highlights(
        self,
        conversations: list[Conversation],
        start: datetime,
        end: datetime,
    ) -> list[GroupHighlight]:
        """Find highlights from active group chats.

        Args:
            conversations: List of conversations to check.
            start: Start of the digest period.
            end: End of the digest period.

        Returns:
            List of group highlights.
        """
        highlights = []

        # Filter to group conversations with significant activity
        groups = [c for c in conversations if c.is_group]

        for conv in groups:
            messages = self.reader.get_messages(conv.chat_id, limit=100)
            period_messages = [m for m in messages if m.date and start <= m.date <= end]

            if len(period_messages) < 5:  # Minimum activity threshold
                continue

            # Find active participants
            participants_counts: dict[str, int] = {}
            for m in period_messages:
                sender = m.sender_name or m.sender
                participants_counts[sender] = participants_counts.get(sender, 0) + 1

            active_participants = sorted(
                participants_counts.keys(),
                key=lambda x: participants_counts[x],
                reverse=True,
            )[:5]

            # Extract simple topics from messages (keyword extraction)
            topics = self._extract_topics(period_messages)

            last_activity = max((m.date for m in period_messages if m.date), default=None)

            highlights.append(
                GroupHighlight(
                    chat_id=conv.chat_id,
                    display_name=conv.display_name or ", ".join(conv.participants[:3]),
                    participants=conv.participants,
                    message_count=len(period_messages),
                    active_participants=active_participants,
                    top_topics=topics[:5],
                    last_activity=last_activity,
                )
            )

        # Sort by message count
        highlights.sort(key=lambda x: x.message_count, reverse=True)
        return highlights[:5]  # Top 5 groups

    def _extract_topics(self, messages: list[Message]) -> list[str]:
        """Extract simple topic keywords from messages.

        Args:
            messages: Messages to analyze.

        Returns:
            List of topic keywords.
        """
        # Simple keyword extraction - common discussion topics
        topic_keywords = {
            "meeting": ["meeting", "call", "zoom", "teams"],
            "plans": ["plan", "plans", "planning", "schedule"],
            "travel": ["trip", "travel", "flight", "hotel", "vacation"],
            "food": ["lunch", "dinner", "restaurant", "food", "eat"],
            "work": ["work", "project", "deadline", "task"],
            "social": ["party", "hangout", "drinks", "event"],
            "family": ["family", "kids", "mom", "dad", "parents"],
            "health": ["doctor", "appointment", "health", "sick"],
        }

        text_combined = " ".join(m.text.lower() for m in messages if m.text)

        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text_combined for kw in keywords):
                found_topics.append(topic)

        return found_topics

    def _detect_action_items(
        self,
        conversations: list[Conversation],
        start: datetime,
        end: datetime,
    ) -> list[ActionItem]:
        """Detect action items from messages.

        Args:
            conversations: Conversations to scan.
            start: Start of the digest period.
            end: End of the digest period.

        Returns:
            List of detected action items.
        """
        action_items = []

        for conv in conversations[:20]:  # Limit conversations scanned
            messages = self.reader.get_messages(conv.chat_id, limit=50)
            period_messages = [m for m in messages if m.date and start <= m.date <= end]

            for message in period_messages:
                if not message.text:
                    continue

                text = message.text.strip()
                if len(text) < 10:
                    continue

                for item_type, patterns in ACTION_PATTERNS.items():
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            conv_name = conv.display_name or ", ".join(conv.participants[:2])
                            sender = message.sender_name or message.sender
                            if message.is_from_me:
                                sender = "You"

                            action_items.append(
                                ActionItem(
                                    text=text[:150],
                                    chat_id=conv.chat_id,
                                    conversation_name=conv_name,
                                    sender=sender,
                                    date=message.date or datetime.now(),
                                    message_id=message.id,
                                    item_type=item_type,
                                )
                            )
                            break  # Only count once per message
                    else:
                        continue
                    break

        # Sort by date, most recent first
        action_items.sort(key=lambda x: x.date, reverse=True)
        return action_items[:20]  # Limit to 20 items

    def _calculate_stats(
        self,
        conversations: list[Conversation],
        start: datetime,
        end: datetime,
    ) -> MessageStats:
        """Calculate message volume statistics.

        Args:
            conversations: Conversations to analyze.
            start: Start of the digest period.
            end: End of the digest period.

        Returns:
            Message statistics.
        """
        total_sent = 0
        total_received = 0
        active_conversations = 0
        most_active_conversation = None
        most_active_count = 0
        hourly_counts: dict[int, int] = {h: 0 for h in range(24)}
        conversation_counts: dict[str, int] = {}

        for conv in conversations:
            messages = self.reader.get_messages(conv.chat_id, limit=100)
            period_messages = [m for m in messages if m.date and start <= m.date <= end]

            if not period_messages:
                continue

            active_conversations += 1
            conv_count = len(period_messages)
            conv_name = conv.display_name or ", ".join(conv.participants[:2])
            conversation_counts[conv_name] = conv_count

            if conv_count > most_active_count:
                most_active_count = conv_count
                most_active_conversation = conv_name

            for m in period_messages:
                if m.is_from_me:
                    total_sent += 1
                else:
                    total_received += 1

                if m.date:
                    hourly_counts[m.date.hour] += 1

        total_messages = total_sent + total_received
        days = max((end - start).days, 1)
        avg_messages_per_day = total_messages / days

        # Find busiest hour
        busiest_hour_candidate = max(hourly_counts.keys(), key=lambda h: hourly_counts[h])
        busiest_hour: int | None = (
            None if hourly_counts[busiest_hour_candidate] == 0 else busiest_hour_candidate
        )

        return MessageStats(
            total_sent=total_sent,
            total_received=total_received,
            total_messages=total_messages,
            active_conversations=active_conversations,
            most_active_conversation=most_active_conversation,
            most_active_count=most_active_count,
            avg_messages_per_day=avg_messages_per_day,
            busiest_hour=busiest_hour,
            hourly_distribution=hourly_counts,
        )


def generate_digest(
    reader: ChatDBReader,
    period: DigestPeriod | str = DigestPeriod.DAILY,
    end_date: datetime | None = None,
) -> Digest:
    """Convenience function to generate a digest.

    Args:
        reader: ChatDBReader instance.
        period: Time period for the digest.
        end_date: End date for the digest.

    Returns:
        Generated Digest.
    """
    generator = DigestGenerator(reader)
    return generator.generate(period, end_date)


def export_digest_markdown(digest: Digest) -> str:
    """Export digest to Markdown format.

    Args:
        digest: Digest to export.

    Returns:
        Markdown string.
    """
    lines = [
        f"# JARVIS {digest.period.value.title()} Digest",
        "",
        f"**Generated:** {digest.generated_at.strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** {digest.start_date.strftime('%Y-%m-%d')} to "
        f"{digest.end_date.strftime('%Y-%m-%d')}",
        "",
    ]

    # Stats summary
    lines.extend(
        [
            "## Activity Summary",
            "",
            f"- **Total Messages:** {digest.stats.total_messages}",
            f"- **Sent:** {digest.stats.total_sent} | **Received:** {digest.stats.total_received}",
            f"- **Active Conversations:** {digest.stats.active_conversations}",
            f"- **Avg Messages/Day:** {digest.stats.avg_messages_per_day:.1f}",
        ]
    )
    if digest.stats.busiest_hour is not None:
        lines.append(f"- **Busiest Hour:** {digest.stats.busiest_hour}:00")
    if digest.stats.most_active_conversation:
        lines.append(
            f"- **Most Active:** {digest.stats.most_active_conversation} "
            f"({digest.stats.most_active_count} messages)"
        )
    lines.append("")

    # Needs attention
    if digest.needs_attention:
        lines.extend(
            [
                "## Needs Attention",
                "",
                "Conversations with unanswered messages:",
                "",
            ]
        )
        for conv in digest.needs_attention:
            date_str = (
                conv.last_message_date.strftime("%m/%d %H:%M")
                if conv.last_message_date
                else "Unknown"
            )
            preview = conv.last_message_preview or "(no preview)"
            lines.append(
                f"- **{conv.display_name}** ({conv.unanswered_count} unanswered) - {date_str}"
            )
            lines.append(f"  > {preview[:80]}...")
        lines.append("")

    # Highlights
    if digest.highlights:
        lines.extend(
            [
                "## Group Highlights",
                "",
            ]
        )
        for highlight in digest.highlights:
            lines.append(f"### {highlight.display_name}")
            lines.append(f"- **Messages:** {highlight.message_count}")
            lines.append(f"- **Active:** {', '.join(highlight.active_participants[:3])}")
            if highlight.top_topics:
                lines.append(f"- **Topics:** {', '.join(highlight.top_topics)}")
            lines.append("")

    # Action items
    if digest.action_items:
        lines.extend(
            [
                "## Action Items",
                "",
            ]
        )
        items_by_type: dict[str, list[ActionItem]] = {}
        for item in digest.action_items:
            items_by_type.setdefault(item.item_type, []).append(item)

        type_labels = {
            "task": "Tasks",
            "question": "Questions",
            "event": "Events",
            "reminder": "Reminders",
        }
        for item_type, items in items_by_type.items():
            lines.append(f"### {type_labels.get(item_type, item_type.title())}")
            for item in items[:5]:
                date_str = item.date.strftime("%m/%d")
                lines.append(f"- [{date_str}] **{item.conversation_name}** - {item.sender}")
                lines.append(f"  > {item.text[:100]}...")
            lines.append("")

    return "\n".join(lines)


def export_digest_html(digest: Digest) -> str:
    """Export digest to HTML format.

    Args:
        digest: Digest to export.

    Returns:
        HTML string.
    """
    md_content = export_digest_markdown(digest)

    # Simple Markdown to HTML conversion (basic)
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>JARVIS {digest.period.value.title()} Digest</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; "
        "max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }",
        "h1 { color: #1a73e8; }",
        "h2 { color: #202124; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }",
        "h3 { color: #5f6368; }",
        "blockquote { background: #f8f9fa; border-left: 3px solid #1a73e8; "
        "padding: 8px 16px; margin: 8px 0; color: #5f6368; }",
        "ul { padding-left: 20px; }",
        "li { margin: 4px 0; }",
        ".stat { display: inline-block; background: #e8f0fe; "
        "padding: 4px 12px; border-radius: 4px; margin: 2px; }",
        ".attention { background: #fce8e6; border-left: 3px solid #ea4335; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    # Convert markdown to basic HTML
    for line in md_content.split("\n"):
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{_convert_bold(line[2:])}</li>")
        elif line.startswith("  > "):
            html_lines.append(f"<blockquote>{line[4:]}</blockquote>")
        elif line.startswith("**"):
            html_lines.append(f"<p>{_convert_bold(line)}</p>")
        elif line.strip():
            html_lines.append(f"<p>{_convert_bold(line)}</p>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def _convert_bold(text: str) -> str:
    """Convert Markdown bold to HTML."""
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def get_digest_filename(
    period: DigestPeriod,
    format: DigestFormat,
    date: datetime | None = None,
) -> str:
    """Generate a filename for digest export.

    Args:
        period: Digest period type.
        format: Export format.
        date: Date for the digest (defaults to now).

    Returns:
        Generated filename.
    """
    date = date or datetime.now()
    date_str = date.strftime("%Y%m%d")
    ext = "md" if format == DigestFormat.MARKDOWN else "html"
    return f"jarvis_digest_{period.value}_{date_str}.{ext}"
