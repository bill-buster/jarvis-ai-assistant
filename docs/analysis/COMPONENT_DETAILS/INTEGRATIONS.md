# Integrations Subsystem Deep Dive

**Last Updated**: 2026-01-27

---

## Overview

The Integrations subsystem provides read-only access to macOS iMessage (chat.db) and Calendar databases.

---

## iMessage Integration

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     ChatDBReader                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Core Methods                                        │  │
│  │  - check_access() / require_access()                │  │
│  │  - get_conversations(limit, since, before)          │  │
│  │  - get_messages(chat_id, limit, before)             │  │
│  │  - search(query, sender, after, before, chat_id)    │  │
│  │  - get_conversation_context(around_message_id)      │  │
│  │  - get_attachments(chat_id, type, date_range)       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Parser       │  │ Queries      │  │ Avatar       │     │
│  │ (LRU cache)  │  │ (Schema v14/ │  │ (Contact     │     │
│  │              │  │  v15)        │  │  resolution) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Key Components

#### ChatDBReader (`integrations/imessage/reader.py`)

**Purpose**: Read-only SQLite access to chat.db

**Key Features**:
- Read-only mode via URI: `file:...?mode=ro`
- Schema version detection (v14 Sonoma, v15 Sequoia)
- Contact name resolution from AddressBook database
- LRU cache for GUID-to-ROWID mappings (10,000 max)
- Context manager support for automatic cleanup
- Timeout handling for SQLITE_BUSY (5s)

**Thread Safety**:
> "This class is NOT thread-safe. Each thread should create its own ChatDBReader instance."

#### Parser (`integrations/imessage/parser.py`)

**Purpose**: Message text extraction and normalization

**Key Features**:
- Handles `attributedBody` parsing (NSArchiver plist format)
- LRU cache for parsed attributedBody (MD5 keys)
- Phone number normalization
- Attachment categorization
- Reaction parsing

#### Queries (`integrations/imessage/queries.py`)

**Purpose**: Schema-aware SQL query generation

**Supported Queries**:
- `conversations` - List conversations with last message
- `messages` - Get messages in a conversation
- `search` - Full-text search with filters
- `context` - Messages around a specific message
- `attachments` - Attachment metadata
- `reactions` - Message reactions

**Schema Versions**:
- v14: macOS Sonoma (14.x)
- v15: macOS Sequoia (15.x)

### Data Structures

```python
@dataclass
class Message:
    id: int
    chat_id: str
    sender: str  # Phone number or email
    sender_name: str | None  # From contacts
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[Attachment]
    reply_to_id: int | None
    reactions: list[Reaction]
    date_delivered: datetime | None
    date_read: datetime | None
    is_system_message: bool = False

@dataclass
class Conversation:
    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool
    last_message_text: str | None
```

### Permissions Required

- **Full Disk Access** - Required for chat.db access
- **Contacts** - Optional, for name resolution

---

## Calendar Integration

### Components

| File | Purpose |
|------|---------|
| `detector.py` | NLP-based event detection from messages |
| `reader.py` | Read events from macOS Calendar |
| `writer.py` | Create events in macOS Calendar |

### Event Detection

The `EventDetector` uses:
- Date parsing with python-dateutil
- Pattern matching for event keywords
- Confidence scoring

```python
@dataclass
class DetectedEvent:
    title: str
    start: datetime
    end: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False
    confidence: float = 0.0
    source_text: str = ""
    message_id: int | None = None
```

---

## iMessage Sender (Deprecated)

**Status**: EXPERIMENTAL / UNRELIABLE

From CLAUDE.md:
> "IMessageSender in integrations/imessage/sender.py is deprecated. Apple's AppleScript automation has significant restrictions: requires Automation permission, may be blocked by SIP, requires Messages.app running, and may break in future macOS versions."

---

## Test Coverage

| File | Coverage | Notes |
|------|----------|-------|
| `test_imessage.py` | 100% | 2,747 lines, comprehensive |
| `test_calendar.py` | 100% | Event detection and reading |
| `test_avatar.py` | 100% | Contact avatar resolution |

---

## Security Considerations

1. **Read-only access**: All queries use `?mode=ro`
2. **Parameterized SQL**: No SQL injection risk
3. **Permission validation**: Check before access

---

## Key Files

- `integrations/imessage/reader.py` (1,312 lines)
- `integrations/imessage/parser.py` (543 lines)
- `integrations/imessage/queries.py` (416 lines)
- `integrations/imessage/avatar.py` (215 lines)
- `integrations/calendar/detector.py` (496 lines)
- `integrations/calendar/reader.py` (428 lines)
- `integrations/calendar/writer.py` (248 lines)
