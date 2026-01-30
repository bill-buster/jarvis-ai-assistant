# JARVIS v3 - Minimal Reply Generation

A clean, minimal version of JARVIS focused on one thing: generating natural reply suggestions for iMessage conversations.

## What This Is

Click on a conversation → Get 3 smart, natural reply suggestions that sound like you.

## Quick Start

```bash
# 1. Setup
make install

# 2. Profile your contacts (one-time, ~30 min)
uv run python scripts/profile_contacts.py

# 3. Index your messages for RAG
uv run python scripts/index_messages.py

# 4. Test reply generation
uv run python -c "
from core.generation.reply_generator import ReplyGenerator
from core.models.loader import ModelLoader

loader = ModelLoader('lfm2.5-1.2b')
generator = ReplyGenerator(loader)

result = generator.generate_replies(
    messages=['Hey want to grab dinner tonight?'],
    chat_id='test-chat',
    contact_name='Friend'
)

for reply in result.replies:
    print(f'→ {reply.text}')
"

# 5. Start API server
uv run python -m api.main
```

## Architecture

```
v3/
├── core/
│   ├── generation/          # Reply generation pipeline
│   │   ├── reply_generator.py    # Main orchestrator
│   │   ├── prompts.py            # Prompt templates
│   │   ├── style_analyzer.py     # Analyze your texting style
│   │   └── context_analyzer.py   # Analyze conversation context
│   ├── embeddings/          # RAG (Retrieval-Augmented Generation)
│   │   ├── store.py              # Message embedding store
│   │   ├── model.py              # Embedding model (all-MiniLM-L6-v2)
│   │   ├── relationship_registry.py  # Contact relationships
│   │   └── contact_profiler.py   # Profile contacts
│   ├── models/              # MLX model loading
│   │   ├── registry.py           # Available models
│   │   └── loader.py             # MLX loader
│   ├── imessage/            # iMessage database reader
│   │   └── reader.py
│   └── utils/               # Text/emoji utilities
├── api/                     # FastAPI server
│   ├── main.py
│   └── routes/
│       ├── health.py
│       ├── conversations.py
│       ├── generate.py      # POST /generate/replies
├── scripts/                 # Essential scripts only
│   ├── profile_contacts.py  # Label contact relationships
│   ├── extract_all_contacts.py
│   ├── label_contacts.py
│   └── index_messages.py    # Build embedding index
└── tests/                   # Unit tests
```

## How It Works

1. **Contact Profiling** → You label contacts (dad, close_friend, coworker)
2. **Message Indexing** → Your past messages are embedded and stored
3. **Reply Generation**:
   - Find similar past conversations (same + cross-conversation)
   - Retrieve your actual replies as examples
   - Generate 3 suggestions using MLX model

## Data Storage

All data lives in **`v3/data/`** (self-contained, no v2 dependencies):

```
v3/data/
├── contacts/contact_profiles.json     # Your 562 labeled relationships ⭐
├── embeddings/embeddings.db           # RAG message embeddings ⭐
└── embeddings/faiss_indices/          # Fast search indices
```

**iMessage** (read-only, system): `~/Library/Messages/chat.db`

See [docs/DATA_STORAGE.md](docs/DATA_STORAGE.md) for complete details on:
- How chat IDs map to contacts
- Where embeddings are stored
- How to backup/reset data
- Troubleshooting missing data

## Key Features

- **Relationship-Aware RAG**: Searches your past replies across similar relationships
- **Cross-Conversation Learning**: Learns from all your conversations, not just current one
- **Style Matching**: Matches your texting style (casual, emoji usage, punctuation)
- **Fast**: Uses small MLX models (0.5-1.5GB) for speed

## Model

**LFM2.5-1.2B** (0.5GB) - Liquid Foundation Model optimized for natural conversation.

- Fast loading (~2-3s on Apple Silicon)
- Natural, casual response style
- Proven baseline: 28% intent match

## API Endpoints

```
GET  /health                          # Health check
GET  /conversations                   # List conversations
GET  /conversations/{id}/messages     # Get messages
POST /generate/replies                # Generate reply suggestions
```

## Configuration

Runtime settings live in `core/config.py` and are loaded via Pydantic Settings.
Override values with environment variables using the `JARVIS_` prefix and `__`
as a nested delimiter.

Examples:

```bash
# Model + generation
export JARVIS_GENERATION__MODEL_NAME="lfm2.5-1.2b"
export JARVIS_GENERATION__MAX_TOKENS="50"
export JARVIS_GENERATION__TEMPERATURE_SCALE='[0.2,0.4,0.6,0.8,0.9]'

# API
export JARVIS_API__PORT="8000"
export JARVIS_API__DEBUG="false"
export JARVIS_API__ALLOW_ORIGINS='["http://localhost:1420"]'
```

Key sections:
- `generation`: model, max tokens, temperature scale, RAG weights
- `embeddings`: data paths, FAISS settings, time-weighting
- `api`: host/port/debug, CORS, pagination limits

## Development

```bash
make test          # Run tests
make lint          # Check code style
make format        # Auto-format code
```

## What's Different from v2

- **Removed**: All experiment scripts, archived code, multiple eval frameworks
- **Kept**: Only the working parts (RAG, relationship registry, core generation)
- **Simplified**: Single consolidated documentation
- **Focused**: Just reply generation, nothing else

## Current Performance (January 2026)

### Quality Metrics

| Metric | Baseline | After Improvements |
|--------|----------|-------------------|
| Fallback usage | 63% | **12%** (-81%) |
| RAG suggestions | 0% | **58%** (new!) |
| Generation time | 2,038ms | **1,395ms** (-31%) |
| Intent match | 28% | 28% (no change) |

### Startup Performance

| Metric | Before Preload | After Preload |
|--------|---------------|---------------|
| First query delay | ~15s | Instant |
| App startup time | ~1s | ~15s |
| RAG lookup | 10-30ms | 10-30ms |

The ~15s startup delay comes from PyTorch/sentence-transformers initialization, not the embedding model itself. See [docs/EMBEDDING_PERFORMANCE.md](docs/EMBEDDING_PERFORMANCE.md) for optimization strategies.

### Success Metrics

Current baseline: 28% intent match (lfm2.5-1.2b with roleplay prompt)

Target improvements:
- 50%: Usable with user review
- 65%: Reliable suggestions
- 80%: Can auto-send

## Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design, components, data flow, and roadmap
- **[Embedding Performance](docs/EMBEDDING_PERFORMANCE.md)** - Startup timing, optimization strategies, MLX options
- **[Visual Flowcharts](docs/FLOWCHARTS.md)** - Mermaid diagrams showing how everything connects
- **[Testing Guide](docs/TESTING.md)** - How to test, evaluate quality, and debug
- **[Data Storage Guide](docs/DATA_STORAGE.md)** - Where everything lives (contacts, embeddings, chat IDs)

**Quick Navigation:**
- [System Overview](#system-overview) - High-level architecture
- [Quick Start](#quick-start) - Get running in 5 minutes
- [How It Works](#how-it-works) - 30-second explanation
- [Data Storage](#data-storage) - Where contacts, embeddings, and chat IDs live
- [Testing Strategy](#testing-strategy) - Validate everything works
- [Roadmap](#next-steps) - Where we're going

## Understanding the System

### Quick Start for Developers

```bash
# 1. Read the architecture doc
cat docs/ARCHITECTURE.md

# 2. Run tests to see components in action
make test-v3

# 3. Explore the code structure
find v3/core -name "*.py" | head -20

# 4. Key files to understand:
#    - core/generation/reply_generator.py (main orchestrator)
#    - core/embeddings/store.py (RAG system)
#    - core/embeddings/relationship_registry.py (cross-convo learning)
```

### How It Works (30-second version)

1. **User clicks conversation** → API receives chat_id
2. **Fetch messages** → Analyze context & style
3. **RAG retrieval** → Find similar past situations
   - Same-conversation search (high priority)
   - Cross-conversation search (similar relationships)
4. **Build prompt** → Use past replies as examples
5. **Generate** → LFM2.5-1.2B creates 3 reply options

### Testing Strategy

```bash
# Level 1: Unit tests (fast, no model loading)
make test-v3              # 17 tests, 0.36s

# Level 2: Integration tests (API + mocked services)
make test                 # Full suite

# Level 3: Manual testing (with real model)
uv run python -c "
from core.generation.reply_generator import ReplyGenerator
from core.models.loader import ModelLoader
loader = ModelLoader()
gen = ReplyGenerator(loader)
result = gen.generate_replies(['Hey want to grab dinner?'], 'test-chat')
for r in result.replies:
    print(f'→ {r.text}')
"
```

## Next Steps

### Phase 1: Validate (This Week)
1. ✅ Run `make test-v3` - verify everything works
2. ⏳ Run `make profile` - label your contact relationships
3. ⏳ Run `make index` - build the RAG embedding index
4. ⏳ Generate 30 test samples and rate them 1-5
5. ⏳ Establish baseline: current ~28% intent match

### Phase 2: Optimize (Week 2)
- Tune RAG weights (same-chat vs cross-chat)
- Test similarity thresholds
- Measure improvement with human ratings

### Phase 3: Prompt Engineering (Week 3)
- Test different prompt styles
- Try relationship-specific prompts
- Target: 40%+ intent match or 60%+ human approval

### Phase 4: Evaluation Framework (Week 4)
- Build reliable automated evaluation
- Hybrid: intent match + embedding similarity
- Calibrate against human ratings

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed roadmap.

## Troubleshooting

**No messages found?** → Grant Full Disk Access in System Settings
**Model won't load?** → Check you have MLX installed (Apple Silicon only)
**Embeddings slow?** → First run builds index, subsequent runs are fast

## License

MIT
