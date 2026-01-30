# Embeddings & Semantic Search

This document covers the embedding system used for semantic search and style learning.

## Overview

JARVIS uses embeddings (vector representations of text) for:
- **Semantic search**: Find messages by meaning, not just keywords
- **Past reply lookup**: Find how you responded to similar messages
- **Contact profiling**: Analyze communication patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingStore                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   SQLite DB     │  │  FAISS Indices  │  │  Embedding  │ │
│  │ (embeddings.db) │  │   (per-chat)    │  │    Model    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           │                    │                    │
    Storage (SQLite)      Search (FAISS)       Compute (CPU)
```

## Components

### Embedding Model (`core/embeddings/model.py`)

**Model**: `all-MiniLM-L6-v2` (sentence-transformers)
**Dimensions**: 384
**Size**: ~90MB

```python
from core.embeddings import get_embedding_model

model = get_embedding_model()

# Single embedding
embedding = model.embed("Hello, how are you?")
# → numpy array of shape (384,)

# Batch embedding
embeddings = model.embed_batch(["Hello", "Goodbye", "Thanks"])
# → list of numpy arrays
```

### Embedding Store (`core/embeddings/store.py`)

Hybrid SQLite + FAISS storage for embeddings.

**Database Schema**:
```sql
CREATE TABLE message_embeddings (
    message_id INTEGER PRIMARY KEY,
    chat_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    text_hash TEXT NOT NULL,
    sender TEXT,
    sender_name TEXT,
    timestamp INTEGER NOT NULL,
    is_from_me INTEGER NOT NULL,
    text_preview TEXT
);

-- Indices for fast queries
CREATE INDEX idx_chat_id ON message_embeddings(chat_id);
CREATE INDEX idx_timestamp ON message_embeddings(timestamp);
CREATE INDEX idx_is_from_me ON message_embeddings(is_from_me);
CREATE INDEX idx_chat_reply_lookup ON message_embeddings(chat_id, is_from_me, timestamp);

-- Full-text search (BM25)
CREATE VIRTUAL TABLE messages_fts USING fts5(...);
```

**FAISS Indexing**:
- Per-chat FAISS indices
- HNSW algorithm for O(log n) search
- Indices cached in memory and on disk

### Embedding Cache (`core/embeddings/cache.py`)

In-memory LRU cache for computed embeddings.

```python
from core.embeddings import get_embedding_cache

cache = get_embedding_cache()
embedding = cache.get_or_compute("Hello world")
# → Returns cached if exists, computes otherwise

stats = cache.get_stats()
# → {"hits": 1250, "misses": 200, "hit_rate": 0.862}
```

## Semantic Search

### Basic Search

```python
from core.embeddings import get_embedding_store

store = get_embedding_store()

# Find similar messages
results = store.find_similar(
    query="dinner plans",
    chat_id="iMessage;+;chat123",  # Optional filter
    limit=10,
    min_similarity=0.4
)

for msg in results:
    print(f"{msg.text} (sim: {msg.similarity:.2f})")
```

### Hybrid Search (Vector + BM25)

Combines semantic similarity with keyword matching using Reciprocal Rank Fusion.

```python
results = store.find_similar_hybrid(
    query="dinner tomorrow",
    chat_id="iMessage;+;chat123",
    limit=10,
    vector_weight=0.7,  # Weight for vector results
    bm25_weight=0.3     # Weight for BM25 results
)
```

### Past Replies Lookup

Find how YOU responded to similar messages:

```python
past_replies = store.find_your_past_replies(
    incoming_message="Want to grab dinner?",
    chat_id="iMessage;+;chat123",
    limit=5,
    min_similarity=0.6,
    use_time_weighting=True  # Boost recent replies
)

for their_msg, your_reply, score in past_replies:
    print(f"They: {their_msg}")
    print(f"You: {your_reply}")
    print(f"Score: {score:.2f}")
```

## Time-Weighted Scoring

Recent replies are more relevant than old ones. The scoring formula:

```python
final_score = (
    semantic_similarity * 0.85 +
    recency_factor * 0.15 +
    time_window_boost +
    day_type_boost
)
```

**Parameters**:
- `recency_factor`: 1.0 for today → 0.0 for max_age_days ago
- `time_window_boost`: +0.1 if message from same time of day (±3 hours)
- `day_type_boost`: +0.05 if same day type (weekday/weekend)

## Indexing Messages

### Full Indexing

```bash
python scripts/index_messages.py
```

This:
1. Reads all messages from iMessage
2. Computes embeddings in batches
3. Stores in SQLite
4. Builds FAISS indices

### Incremental Indexing

```python
from core.embeddings import get_embedding_store

store = get_embedding_store()
stats = store.index_messages(messages)
# → {"indexed": 150, "skipped": 10, "duplicates": 5}
```

### Preloading Indices

For faster first search, preload indices on app start:

```python
store.preload_index("iMessage;+;chat123")
# Builds FAISS index in background thread
```

## Contact Profiling

Rich analysis of communication patterns:

```python
from core.embeddings import get_contact_profile

profile = get_contact_profile(
    chat_id="iMessage;+;chat123",
    include_topics=True  # Set False for faster, style-only profile
)

print(f"Name: {profile.display_name}")
print(f"Relationship: {profile.relationship_type}")
print(f"Total messages: {profile.total_messages}")
print(f"Tone: {profile.tone}")
print(f"Uses emoji: {profile.uses_emoji}")
print(f"Topics: {[t.name for t in profile.topics]}")
```

### Profile Caching

Profiles are cached in SQLite (`~/.jarvis/profile_cache.db`):
- Cache invalidates when message count changes
- Or when profile is older than 24 hours

## File Locations

| File | Purpose |
|------|---------|
| `~/.jarvis/embeddings.db` | SQLite database with embeddings |
| `~/.jarvis/faiss_indices/` | Cached FAISS indices (per-chat) |
| `~/.jarvis/profile_cache.db` | Cached contact profiles |

## Performance

### Indexing Speed
- ~100 messages/second (batch embedding)
- ~1000 messages/second (storage)

### Search Latency
- FAISS search: <5ms for 10K vectors
- Hybrid search: ~50ms (includes BM25)
- Full profile build: ~500ms (first time)

### Memory Usage
- Model: ~90MB
- FAISS index: ~100MB per 100K messages
- Cache: Configurable (default 10K entries)

## Configuration

### FAISS Parameters

```python
# In store.py
HNSW_M = 32              # Neighbors per node
HNSW_EF_CONSTRUCTION = 200  # Build quality
HNSW_EF_SEARCH = 64      # Search quality
```

Higher values = better recall, more memory.

### Time-Weighting Parameters

```python
# In store.py
RECENCY_WEIGHT = 0.15
TIME_WINDOW_BOOST = 0.1
DAY_TYPE_BOOST = 0.05
MAX_AGE_DAYS = 365
```

### Cache Configuration

```python
# In cache.py
MAX_CACHE_SIZE = 10000  # Max entries
```

## Debugging

### Check Index Status

```python
store = get_embedding_store()
print(store.get_stats())
# {
#   "total_messages": 50000,
#   "unique_conversations": 150,
#   "db_size_mb": 245.3
# }
```

### Check Cache Stats

```python
cache = get_embedding_cache()
print(cache.get_stats())
# {"hits": 1250, "misses": 200, "hit_rate": 0.862}
```

### Check FAISS Index

```python
store = get_embedding_store()
if store.is_index_ready("iMessage;+;chat123", only_from_me=False):
    print("Index is cached")
else:
    print("Index needs building")
```

## Best Practices

1. **Preload indices** for frequently-accessed chats on app start
2. **Use hybrid search** for user-facing queries (better recall)
3. **Use vector-only search** for internal lookups (faster)
4. **Set appropriate min_similarity** (0.4 for broad, 0.7 for precise)
5. **Enable time-weighting** for past reply lookups
6. **Cache profiles** with `use_cache=True` (default)

## Troubleshooting

**"FAISS not available"**
```bash
pip install faiss-cpu
```
Falls back to brute-force search if unavailable.

**"Slow first search"**
- FAISS index builds on first access (~2s for 1000 vectors)
- Use `preload_index()` on app start

**"Out of memory"**
- Reduce HNSW_M parameter
- Increase cache eviction frequency
- Use smaller embedding model

**"Stale search results"**
- Re-index after new messages: `store.index_messages(new_messages)`
- Clear FAISS cache: `store._faiss_indices.clear()`
