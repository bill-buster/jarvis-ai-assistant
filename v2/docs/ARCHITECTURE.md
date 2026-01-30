# JARVIS v2 Architecture

## Overview

JARVIS v2 follows a modular architecture with clear separation between:
- **API Layer**: FastAPI REST endpoints for frontend communication
- **Core Layer**: Business logic for generation, embeddings, and iMessage access
- **Frontend**: Tauri + Svelte desktop application

```
┌─────────────────────────────────────────────────────────────┐
│                    Desktop App (Tauri)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Svelte Components                      │   │
│  │  ConversationList │ MessageList │ ReplySuggestions  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                    HTTP/WebSocket                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ /health  │ │/convers- │ │/generate │ │   /ws    │      │
│  │          │ │ ations   │ │          │ │          │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │   Generation   │  │   Embeddings  │  │   iMessage    │  │
│  │    Pipeline    │  │    & Search   │  │    Reader     │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│           │                  │                  │           │
│           ▼                  ▼                  ▼           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  MLX Models   │  │    FAISS      │  │   chat.db     │  │
│  │  (LFM2.5)     │  │   Indices     │  │  (read-only)  │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Local-First
All processing happens on-device. No external API calls for core functionality.

### 2. Privacy-Preserving
- iMessage database accessed read-only
- No data leaves the device
- Embeddings stored locally in `~/.jarvis/`

### 3. Performance-Optimized
- Fast-path template matching skips LLM for common responses
- Lazy model loading to reduce startup time
- Per-chat FAISS indices for O(log n) search
- Aggressive caching (styles, profiles, embeddings)

### 4. Graceful Degradation
- Fallback replies when LLM fails
- Template responses when embeddings unavailable
- Clear error messages for permission issues

## Key Architectural Patterns

### Singleton Pattern
Single instances for expensive resources:

```python
# Model loader singleton
_model_loader: ModelLoader | None = None

def get_model_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        with _loader_lock:
            if _model_loader is None:
                _model_loader = ModelLoader()
    return _model_loader
```

Used for: `ModelLoader`, `EmbeddingStore`, `EmbeddingModel`, `ContactProfiler`

### Thread-Safe Lazy Loading
Double-check locking prevents race conditions:

```python
def _ensure_loaded(self):
    if self._model is not None:
        return
    with self._load_lock:
        if self._model is not None:
            return
        self._model = load_model()  # Expensive operation
```

### Fast-Path Optimization
Three-tier response strategy:

1. **Template Match** (~2ms): Check if incoming message matches learned patterns
2. **Past Reply Match** (~100ms): Find your similar past responses via embedding search
3. **LLM Generation** (~1000ms): Full generation pipeline

```python
# Fast path check
template_match = self._template_matcher.match(message)
if template_match and template_match.confidence > 0.75:
    return template_match.response  # Skip LLM entirely!

# Medium path - check past replies
past_replies = self._find_past_replies(message, chat_id)
template_reply = self._try_template_match(past_replies)
if template_reply:
    return template_reply  # Skip LLM!

# Slow path - full LLM generation
reply = self.model_loader.generate(prompt)
```

### Caching Strategy

| Cache | Type | TTL | Invalidation |
|-------|------|-----|--------------|
| Style analysis | In-memory | Session | Chat change |
| Contact profiles | SQLite | 24h | Message count change |
| Embeddings | In-memory LRU | 10K entries | LRU eviction |
| FAISS indices | In-memory | Session | Manual rebuild |

## Module Architecture

### API Layer (`api/`)

```
api/
├── main.py          # FastAPI app, lifespan, middleware
├── schemas.py       # Pydantic request/response models
└── routes/
    ├── conversations.py  # Chat listing, messages, profiles
    ├── generate.py       # Reply generation endpoint
    ├── websocket.py      # Real-time updates
    ├── search.py         # Semantic search
    ├── settings.py       # App configuration
    └── health.py         # System health checks
```

**Key Design Decisions**:
- CORS permissive for local development
- Background model preloading via lifespan handler
- WebSocket for streaming generation results

### Core Layer (`core/`)

```
core/
├── models/
│   ├── loader.py     # MLX model loading & generation
│   └── registry.py   # Model specifications
│
├── generation/
│   ├── reply_generator.py   # Main orchestrator (980 lines)
│   ├── context_analyzer.py  # Intent/mood detection
│   ├── style_analyzer.py    # User style detection
│   ├── coherence.py         # Topic coherence
│   └── prompts.py           # Prompt templates
│
├── imessage/
│   ├── reader.py     # Read-only chat.db access
│   └── sender.py     # Message sending (experimental)
│
├── embeddings/
│   ├── store.py           # SQLite + FAISS hybrid
│   ├── model.py           # all-MiniLM-L6-v2
│   ├── cache.py           # LRU embedding cache
│   ├── similarity.py      # Vector operations
│   ├── indexer.py         # Indexing pipeline
│   └── contact_profiler.py # Rich contact analysis
│
└── templates/
    └── matcher.py    # Response template matching
```

### Frontend (`desktop/`)

```
desktop/
├── src/
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts     # HTTP client
│   │   │   ├── websocket.ts  # WS client
│   │   │   └── types.ts      # TypeScript interfaces
│   │   │
│   │   ├── components/
│   │   │   ├── ConversationList.svelte
│   │   │   ├── MessageList.svelte
│   │   │   ├── ReplySuggestions.svelte
│   │   │   ├── ContactProfilePanel.svelte
│   │   │   └── GenerationDebugPanel.svelte
│   │   │
│   │   └── stores/
│   │       └── app.ts  # Centralized Svelte store
│   │
│   └── routes/
│       ├── +layout.svelte  # App layout, polling lifecycle
│       └── +page.svelte    # Main chat UI
│
├── src-tauri/        # Tauri native config
├── package.json
└── vite.config.ts
```

**State Management**: Centralized Svelte store (`app.ts`) with:
- Conversations, messages, replies state
- Loading states per operation
- WebSocket connection status
- Unread tracking via `lastSeenDate`

## Data Flow

### Reply Generation Flow

```
1. User clicks "Generate" in frontend
                    │
                    ▼
2. POST /generate/replies { chat_id, num_replies }
                    │
                    ▼
3. MessageReader.get_messages(chat_id, limit=30)
                    │
                    ▼
4. ReplyGenerator.generate_replies()
   │
   ├─ 4a. Coherence filter (extract relevant messages)
   ├─ 4b. Template match check (fast-path)
   ├─ 4c. Style analysis (your texting patterns)
   ├─ 4d. Context analysis (intent, mood, urgency)
   ├─ 4e. Reply strategy determination
   ├─ 4f. Past replies lookup (FAISS + time-weighting)
   ├─ 4g. Availability signal detection
   ├─ 4h. Contact profile loading
   ├─ 4i. Style instructions building
   ├─ 4j. Context refresh (for long conversations)
   ├─ 4k. Prompt building
   ├─ 4l. LLM generation (MLX)
   ├─ 4m. Reply parsing & filtering
   └─ 4n. Fallback handling if needed
                    │
                    ▼
5. GenerateRepliesResponse { replies, debug_info, timing }
                    │
                    ▼
6. Frontend displays replies with timing breakdown
```

### Semantic Search Flow

```
1. User types query in search box
                    │
                    ▼
2. GET /search?q={query}&limit=20
                    │
                    ▼
3. EmbeddingModel.embed(query)
                    │
                    ▼
4. EmbeddingStore.find_similar(query_embedding)
   │
   ├─ Check FAISS index (if chat_id specified)
   └─ Or brute-force search across all messages
                    │
                    ▼
5. Return top-k messages with similarity scores
```

## File Locations

| Data | Path |
|------|------|
| iMessage database | `~/Library/Messages/chat.db` |
| AddressBook | `~/Library/Application Support/AddressBook/Sources/` |
| Embeddings DB | `~/.jarvis/embeddings.db` |
| FAISS indices | `~/.jarvis/faiss_indices/` |
| Profile cache | `~/.jarvis/profile_cache.db` |
| MLX models | `~/.cache/mlx-community/` |

## Performance Characteristics

### Latency Targets

| Operation | Target | Typical |
|-----------|--------|---------|
| Template match | <10ms | ~2ms |
| Past replies lookup | <200ms | ~100ms |
| Full LLM generation | <3s | ~1-2s |
| Cold start | <15s | ~10s |

### Memory Profile

| Component | Memory |
|-----------|--------|
| LFM2.5 1.2B model | ~0.5GB |
| FAISS indices | ~100MB per 100K messages |
| SQLite databases | <100MB combined |
| **Total process** | ~1-2GB typical |

## Error Handling

### API Layer
- HTTP 503: Permission issues (Full Disk Access)
- HTTP 404: Resource not found
- HTTP 500: Server errors with stack trace in dev

### Core Layer
- Graceful degradation with fallback replies
- Logged errors with timing context
- Clear exception hierarchy

### Frontend
- Connection status indicator
- Retry on WebSocket disconnect
- Loading states for all async operations
