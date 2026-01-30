# JARVIS v3 - Data Storage Guide

## Overview

All JARVIS data is now stored in **one place**: `v3/data/`

No more confusion between v2 and v3. Everything is self-contained.

## Directory Structure

```
v3/data/
├── contacts/              # Contact profiles and relationships
│   ├── contact_profiles.json      # Your 562 labeled contacts ⭐
│   ├── all_contacts.json          # Raw contact list
│   ├── all_contacts_with_names.json
│   └── all_contacts_with_phones.json
│
├── embeddings/            # RAG (Retrieval-Augmented Generation)
│   ├── embeddings.db              # SQLite with message embeddings ⭐
│   └── faiss_indices/             # FAISS search indices
│       └── {chat_id}.faiss        # Per-conversation indices
│
└── cache/                 # Temporary caches
    └── embedding_cache.pkl        # Embedding model cache
```

## Data Sources

### 1. iMessage Database (Read-Only, System)

**Location**: `~/Library/Messages/chat.db`

**What it contains**:
- All your iMessage conversations
- Messages, timestamps, senders
- Chat IDs (unique identifiers)
- Attachments metadata

**How we use it**:
```python
from core.imessage import MessageReader

reader = MessageReader()
messages = reader.get_messages(chat_id="chat123", limit=10)
# Returns: [{text, sender, is_from_me, timestamp}, ...]
```

**Important**: We only READ from this. Never write.

---

### 2. Contact Profiles (Your Labels)

**Location**: `v3/data/contacts/contact_profiles.json`

**What it contains**:
```json
{
  "John Doe": {
    "relationship": "close_friend",
    "category": "friend",
    "identifiers": ["+1234567890", "john@email.com"]
  },
  "Mom": {
    "relationship": "mom",
    "category": "family"
  }
}
```

**How we use it**:
```python
from core.embeddings import get_relationship_registry

registry = get_relationship_registry()

# Get relationship type
info = registry.get_relationship("John Doe")
print(info.relationship)  # "close_friend"
print(info.category)      # "friend"

# Find similar contacts (for cross-conversation RAG)
similar = registry.get_similar_contacts("John Doe")
# Returns: ["Jane Doe", "Bob Smith"] (other close_friend contacts)
```

**How to update**:
```bash
# Edit profiles manually
python scripts/profile_contacts.py

# Or edit the JSON directly
vim v3/data/contacts/contact_profiles.json
```

---

### 3. Message Embeddings (RAG)

**Location**: `v3/data/embeddings/embeddings.db`

**What it contains**:
- Message text (truncated)
- 384-dimensional embeddings (all-MiniLM-L6-v2)
- Chat IDs and sender info
- Timestamps

**How we use it**:
```python
from core.embeddings import get_embedding_store

store = get_embedding_store()

# Search for similar messages
results = store.find_similar_messages(
    query="Want to grab dinner?",
    chat_id="chat123",
    limit=5
)

# Find your past replies to similar messages
replies = store.find_your_past_replies(
    incoming_message="Want to hang?",
    chat_id="chat123",
    limit=3
)
```

**How to build**:
```bash
# Index all messages (one-time setup)
python scripts/index_messages.py

# Or use make
make index
```

**Size**: ~500MB for 100K messages (depends on your history)

---

### 4. FAISS Indices (Fast Search)

**Location**: `v3/data/embeddings/faiss_indices/`

**What it contains**:
- Binary FAISS index files (one per conversation)
- Enables fast similarity search (milliseconds)
- Auto-generated from embeddings.db

**How it works**:
1. First search for a chat → builds index (slow, ~1-2s)
2. Subsequent searches → uses cached index (fast, ~10-50ms)
3. Indices auto-update when new messages added

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    iMessage (System)                        │
│              ~/Library/Messages/chat.db                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  chat_id: "+1234567890"                             │   │
│  │  messages: ["Hey!", "Want to hang?"]                │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ Read messages
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    v3/data/                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  contacts/contact_profiles.json                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "Friend": {                                        │   │
│  │    "relationship": "close_friend",                  │   │
│  │    "identifiers": ["+1234567890"]                   │   │
│  │  }                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                       │
│                     │ Map chat_id → contact                 │
│                     ▼                                       │
│  embeddings/embeddings.db                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  message: "Want to hang?"                           │   │
│  │  embedding: [0.23, -0.45, ...] (384-dim)            │   │
│  │  chat_id: "+1234567890"                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└────────────────────┬────────────────────────────────────────┘
                     │ RAG Search
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Reply Generator                             │
│                                                             │
│  1. Get conversation from iMessage                          │
│  2. Look up contact in profiles                             │
│  3. Search embeddings for similar messages                  │
│  4. Generate reply using context + style                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Chat ID Resolution

**The Problem**: iMessage uses chat IDs like `+1234567890` or `chat1234567890`, but you label contacts by name like "John Doe".

**The Solution**: Multiple resolution strategies

```python
from core.embeddings import get_relationship_registry

registry = get_relationship_registry()

# Strategy 1: Direct phone number match
chat_id = "+1234567890"
info = registry.get_relationship(chat_id)
# Returns: RelationshipInfo for "John Doe"

# Strategy 2: Chat ID contains phone
chat_id = "chat1234567890"
info = registry.get_relationship_from_chat_id(chat_id)
# Extracts phone from chat_id, then looks up

# Strategy 3: Use contact name directly
info = registry.get_relationship("John Doe")

# All return:
# RelationshipInfo(
#   contact_name="John Doe",
#   relationship="close_friend",
#   category="friend",
#   ...
# )
```

---

## Common Operations

### Check Your Data

```bash
# Count labeled contacts
cat v3/data/contacts/contact_profiles.json | python -m json.tool | grep -c '"relationship"'

# Check embedding database size
ls -lh v3/data/embeddings/embeddings.db

# Count indexed messages
sqlite3 v3/data/embeddings/embeddings.db "SELECT COUNT(*) FROM message_embeddings;"
```

### Backup Your Data

```bash
# Backup everything
cp -r v3/data v3/data_backup_$(date +%Y%m%d)

# Or just contacts
cp v3/data/contacts/contact_profiles.json v3/data/contacts/contact_profiles_backup.json
```

### Reset Embeddings (Start Fresh)

```bash
# Delete and rebuild
rm -rf v3/data/embeddings/*
python scripts/index_messages.py
```

---

## Troubleshooting

### "No contact_profiles.json found"

```bash
# Copy from v2 (if you have it)
cp v2/results/contacts/contact_profiles.json v3/data/contacts/

# Or create new
python scripts/profile_contacts.py
```

### "Embeddings not found"

```bash
# Build the index
python scripts/index_messages.py

# Check if it worked
ls -lh v3/data/embeddings/embeddings.db
```

### "Chat ID not resolving to contact"

```bash
# Check what chat IDs look like
python -c "
from core.imessage import MessageReader
reader = MessageReader()
convs = reader.get_conversations(limit=5)
for c in convs:
    print(f'{c.chat_id}: {c.display_name}')
reader.close()
"

# Check if your profiles have matching identifiers
python -c "
import json
with open('v3/data/contacts/contact_profiles.json') as f:
    profiles = json.load(f)
for name, data in profiles.items():
    if 'identifiers' in data:
        print(f'{name}: {data[\"identifiers\"]}')
"
```

---

## Quick Reference

| Data | Location | Size | How to Update |
|------|----------|------|---------------|
| iMessage | `~/Library/Messages/chat.db` | 100MB-1GB | Auto (system) |
| Contacts | `v3/data/contacts/contact_profiles.json` | ~100KB | `profile_contacts.py` |
| Embeddings | `v3/data/embeddings/embeddings.db` | ~500MB | `index_messages.py` |
| FAISS | `v3/data/embeddings/faiss_indices/` | ~200MB | Auto-generated |

---

## Summary

**Everything is in `v3/data/` now.**

- **contacts/**: Your labeled relationships
- **embeddings/**: RAG search data
- **cache/**: Temporary files

**iMessage stays in system** (read-only at `~/Library/Messages/chat.db`)

No more confusion. No more v2 vs v3. Just `v3/data/`.
