# JARVIS v3 Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  (Tauri Desktop App or Direct API Calls)                        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI LAYER (api/)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ GET /health │  │ GET /convos │  │ POST /generate/replies  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REPLY GENERATION PIPELINE                     │
│                     (core/generation/)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CONTEXT ANALYSIS          2. STYLE ANALYSIS                │
│     ↓ intent detection          ↓ your texting patterns        │
│     ↓ relationship type         ↓ emoji usage                  │
│     ↓ conversation summary      ↓ punctuation style            │
│                                                                 │
│  3. RAG RETRIEVAL (EMBEDDINGS)                                 │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  Same-Conversation Search    Cross-Conv Search     │    │
│     │  ↓ find similar messages     ↓ similar contacts    │    │
│     │  ↓ get your past replies     ↓ their relationships │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. PROMPT BUILDING                                            │
│     ↓ conversation context                                     │
│     ↓ style instructions                                       │
│     ↓ few-shot examples (from RAG)                             │
│                                                                 │
│  5. MLX GENERATION (LFM2.5-1.2B)                               │
│     ↓ generate 3 reply options                                 │
│     ↓ temperature scaling                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  iMessage DB (read-only)  │  Embeddings (SQLite + FAISS)       │
│  Contact Profiles (JSON)  │  Relationship Registry (JSON)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Configuration

Runtime configuration is centralized in `core/config.py` using Pydantic Settings.
Override values with environment variables using the `JARVIS_` prefix and `__`
for nested fields. Example:

```bash
export JARVIS_GENERATION__MAX_TOKENS="50"
export JARVIS_API__ALLOW_ORIGINS='["http://localhost:1420"]'
```

### 1. Reply Generation Flow

```
User clicks conversation
         │
         ▼
┌─────────────────┐
│  API receives   │
│  chat_id        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Fetch Messages  │────▶│  ContextAnalyzer │
│ from iMessage   │     │  • Detect intent │
└─────────────────┘     │  • Relationship  │
         │              │  • Summary       │
         │              └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         │              │  StyleAnalyzer  │
         │              │  • Your patterns │
         │              │  • Emoji usage   │
         │              └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         RAG: Find Similar Replies       │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────┐  ┌──────────────┐ │
│  │ Same-Chat Search│  │ Cross-Chat   │ │
│  │ • Similar msgs  │  │ • Same rel   │ │
│  │ • Your replies  │  │ • Their reps │ │
│  └────────┬────────┘  └──────┬───────┘ │
│           │                  │         │
│           └────────┬─────────┘         │
│                    ▼                   │
│         ┌─────────────────┐            │
│         │ Merge & Rank    │            │
│         │ (weighted)      │            │
│         └────────┬────────┘            │
│                  ▼                     │
│         Past replies as examples       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Build Prompt    │
│ • Context       │
│ • Style         │
│ • Examples      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LFM2.5-1.2B     │
│ Generation      │
│ (3 replies)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return to User  │
└─────────────────┘
```

### 2. RAG System Architecture

```
                    RAG (Retrieval-Augmented Generation)
                    
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING STORE                              │
│                    (core/embeddings/store.py)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Messages    │    │  Embeddings  │    │    FAISS     │      │
│  │  (SQLite)    │◄───│  (all-MiniLM)│◄───│   Index      │      │
│  │              │    │              │    │              │      │
│  │ • text       │    │ • 384-dim    │    │ • vectors    │      │
│  │ • sender     │    │ • normalized │    │ • metadata   │      │
│  │ • timestamp  │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│  SAME-CONVERSATION      │      │  CROSS-CONVERSATION     │
│  SEARCH                 │      │  SEARCH                 │
├─────────────────────────┤      ├─────────────────────────┤
│                         │      │                         │
│ Input: incoming message │      │ Input: incoming message │
│         + chat_id       │      │         + relationship  │
│                         │      │                         │
│ 1. Embed query          │      │ 1. Find similar         │
│ 2. FAISS search         │      │    contacts (same rel)  │
│ 3. Filter by chat_id    │      │ 2. Search their chats   │
│ 4. Get your replies     │      │ 3. Get their replies    │
│                         │      │                         │
│ Returns:                │      │ Returns:                │
│ [(msg, reply, sim)]     │      │ [(msg, reply, sim)]     │
└───────────┬─────────────┘      └───────────┬─────────────┘
            │                                │
            └────────────┬───────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    MERGE & RANK         │
            ├─────────────────────────┤
            │                         │
            │ • Deduplicate           │
            │ • Weight: same (0.6)    │
            │ • Weight: cross (0.4)   │
            │ • Sort by similarity    │
            │ • Take top 3-5          │
            │                         │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │   FEW-SHOT EXAMPLES     │
            │   for LLM prompt        │
            └─────────────────────────┘
```

### 3. Relationship Registry

```
┌─────────────────────────────────────────────────────────────────┐
│              RELATIONSHIP REGISTRY                              │
│         (core/embeddings/relationship_registry.py)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROFILES (contact_profiles.json)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ {                                                       │   │
│  │   "John Doe": {                                         │   │
│  │     "relationship": "close_friend",                     │   │
│  │     "category": "friend",                               │   │
│  │     "identifiers": ["+1234567890", "john@email.com"]    │   │
│  │   },                                                    │   │
│  │   "Mom": {                                              │   │
│  │     "relationship": "mom",                              │   │
│  │     "category": "family"                                │   │
│  │   }                                                     │   │
│  │ }                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  HIERARCHY                                                      │
│  ┌─────────────┐                                                │
│  │   family    │──┬── dad                                       │
│  │             │  ├── mom                                       │
│  │             │  ├── brother                                   │
│  │             │  └── sister                                    │
│  ├─────────────┤                                                │
│  │   friend    │──┬── best_friend                              │
│  │             │  ├── close_friend                              │
│  │             │  └── friend                                    │
│  ├─────────────┤                                                │
│  │   work      │──┬── coworker                                  │
│  │             │  └── boss                                      │
│  └─────────────┘                                                │
│                                                                 │
│  USAGE FLOW                                                     │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │ chat_id │────▶│  Resolve    │────▶│  Get Relationship   │   │
│  │         │     │  to contact │     │  Type (e.g., "dad") │   │
│  └─────────┘     └─────────────┘     └─────────────────────┘   │
│                                               │                 │
│                                               ▼                 │
│                              ┌─────────────────────────────┐   │
│                              │  Find Similar Contacts      │   │
│                              │  (same relationship type)   │   │
│                              └─────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   iMessage  │────▶│  Indexer    │────▶│ Embeddings  │
│   Database  │     │  (1-time)   │     │   Store     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┘
                         │
                         ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │◄────│   Reply     │◄────│    RAG      │
│  Interface  │     │  Generator  │     │   Search    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       │ Profile contacts (1-time setup)
       ▼
┌─────────────┐
│ Relationship│
│  Registry   │
└─────────────┘
```

---

## Component Details

### ReplyGenerator (core/generation/reply_generator.py)

**Purpose**: Orchestrates the entire reply generation process

```python
class ReplyGenerator:
    def generate_replies(messages, chat_id) -> ReplyGenerationResult:
        # 1. Analyze context
        context = self.context_analyzer.analyze(messages)
        
        # 2. Analyze style  
        style = self.style_analyzer.analyze(messages, chat_id)
        
        # 3. RAG: Find past replies
        past_replies = self._find_past_replies(
            incoming_message, 
            chat_id,
            cross_conversation=True  # NEW!
        )
        
        # 4. Build prompt with examples
        prompt = build_reply_prompt(
            messages=messages,
            style=style,
            examples=past_replies  # Your actual past replies
        )
        
        # 5. Generate with MLX
        result = self.model_loader.generate(prompt)
        
        return ReplyGenerationResult(
            replies=result.texts,
            context=context,
            past_replies_used=past_replies
        )
```

### EmbeddingStore (core/embeddings/store.py)

**Purpose**: Stores and searches message embeddings for RAG

Key Methods:
- `add_messages()`: Index new messages with embeddings
- `find_similar_messages()`: Semantic search within a chat
- `find_your_past_replies()`: Find your replies to similar messages
- `search_cross_conversation()`: NEW - Search across similar relationships

### RelationshipRegistry (core/embeddings/relationship_registry.py)

**Purpose**: Tracks contact relationships for cross-conversation learning

Key Methods:
- `load_profiles()`: Load from contact_profiles.json
- `get_relationship_by_name()`: Get relationship type for a contact
- `get_similar_contacts()`: Find contacts with same relationship type
- `resolve_chat_id()`: Map iMessage chat ID to contact

---

## Testing Strategy

### Test Pyramid

```
                    ┌─────────┐
                    │  E2E    │  ← Full flow with real model
                    │  (1%)   │    (slow, validates everything)
                    └────┬────┘
                         │
                   ┌─────┴─────┐
                   │ Integration│  ← API + mocked services
                   │   (10%)   │    (medium speed)
                   └─────┬─────┘
                         │
              ┌──────────┴──────────┐
              │       Unit Tests     │  ← Fast, isolated
              │        (89%)         │    (no model loading)
              └──────────────────────┘
```

### Current Test Coverage

```
v3/tests/
├── test_basic.py (7 tests)           ← Core functionality
│   ├── test_imports_work
│   ├── test_model_registry_has_correct_model
│   ├── test_can_create_reply_generator_mocked
│   ├── test_context_analyzer_detects_intent
│   ├── test_style_analyzer_analyzes_messages
│   ├── test_prompt_building
│   └── test_relationship_registry_basic
│
└── test_relationship_registry.py (10 tests)  ← RAG system
    ├── test_load_profiles
    ├── test_get_relationship_by_name
    ├── test_get_relationship_by_phone
    ├── test_get_relationship_not_found
    ├── test_get_relationship_from_chat_id_imessage
    ├── test_get_relationship_from_chat_id_unknown
    ├── test_get_contacts_by_category_friend
    ├── test_get_contacts_by_category_empty
    ├── test_missing_profiles_file
    └── test_phone_normalization

Total: 17 tests, all pass in 0.36s
```

### Testing Best Practices

1. **Mock expensive operations**:
   ```python
   @pytest.fixture
   def mock_model_loader():
       mock = MagicMock()
       mock.generate.return_value = "sounds good"
       return mock
   ```

2. **Test behavior, not implementation**:
   - ✓ Test: "generator returns 3 replies"
   - ✗ Test: "generator calls _parse_replies with correct args"

3. **Fast feedback loop**:
   - Unit tests: <1 second
   - Integration tests: <10 seconds  
   - E2E tests: <60 seconds (with real model)

---

## Roadmap: Moving Forward

### Phase 1: Validation (This Week)

```
┌─────────────────────────────────────────────────────────────┐
│  GOAL: Verify RAG improves reply quality                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Profile your contacts                                   │
│     └─► python scripts/profile_contacts.py                 │
│                                                             │
│  2. Index your messages                                     │
│     └─► python scripts/index_messages.py                   │
│                                                             │
│  3. Generate test samples                                   │
│     └─► Create 30-50 reply examples                        │
│                                                             │
│  4. Human evaluation                                        │
│     └─► Rate each 1-5 on "would I send this?"              │
│                                                             │
│  5. Establish baseline metrics                              │
│     └─► Current: 28% intent match                          │
│     └─► Target: 50%+ human approval                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: RAG Optimization (Week 2)

```
┌─────────────────────────────────────────────────────────────┐
│  GOAL: Tune RAG weights for best retrieval                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Experiments to run:                                        │
│                                                             │
│  1. Same-conversation weight                                │
│     └─► Test: 0.5, 0.6, 0.7, 0.8                          │
│                                                             │
│  2. Cross-conversation weight                               │
│     └─► Test: 0.2, 0.3, 0.4, 0.5                          │
│                                                             │
│  3. Similarity threshold                                    │
│     └─► Test: 0.5, 0.55, 0.6, 0.65                        │
│                                                             │
│  4. Number of examples                                      │
│     └─► Test: 2, 3, 4, 5 past replies                      │
│                                                             │
│  Success: Higher human approval rating                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Prompt Engineering (Week 3)

```
┌─────────────────────────────────────────────────────────────┐
│  GOAL: Optimize prompts for LFM2.5-1.2B                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Test prompt styles:                                     │
│     • Roleplay framing (current best: 28%)                 │
│     • Pure completion                                       │
│     • Few-shot heavy                                        │
│     • Anti-assistant                                        │
│                                                             │
│  2. Test relationship-specific prompts                      │
│     • Different prompts for dad vs friend                  │
│                                                             │
│  3. Test balanced few-shot                                  │
│     • Ensure all intent types represented                  │
│                                                             │
│  Success: 40%+ intent match OR 60%+ human approval          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Evaluation Framework (Week 4)

```
┌─────────────────────────────────────────────────────────────┐
│  GOAL: Reliable automated evaluation                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Options:                                                   │
│                                                             │
│  1. Intent match (current)                                  │
│     └─► Fast, automated                                    │
│     └─► Problem: rigid, doesn't capture quality            │
│                                                             │
│  2. LLM-as-judge (GPT-4)                                    │
│     └─► "Rate 1-5: would I send this?"                     │
│     └─► More nuanced, but expensive                        │
│                                                             │
│  3. Embedding similarity                                    │
│     └─► Compare to gold response embedding                 │
│     └─► Captures semantic similarity                       │
│                                                             │
│  4. Hybrid approach                                         │
│     └─► Intent match + embedding similarity                │
│     └─► Calibrate against human ratings                    │
│                                                             │
│  Recommendation: Start with #4, validate with human eval   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Understanding the System

### Key Insight: Why RAG Works

```
Traditional Approach (Failed):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Incoming   │───▶│  Classify   │───▶│  Generate   │
│  Message    │    │  Intent     │    │  w/ intent  │
└─────────────┘    └─────────────┘    └─────────────┘

Problem: Intent is unpredictable!
"wanna hang?" → accept? decline? question? reaction?
All are valid - can't predict which.

RAG Approach (Working):
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│  Incoming   │───▶│  Find Similar    │───▶│  Generate   │
│  Message    │    │  Past Situations │    │  w/ examples│
└─────────────┘    └──────────────────┘    └─────────────┘

Solution: Show similar situations, let model decide!
"wanna hang?" → find past "wanna hang?" → show your replies
```

### Cross-Conversation Learning

```
Before (Single-Conversation RAG):
┌─────────────────┐
│ Chat with Dad   │
│                 │
│ Dad: "Dinner?"  │
│ You: "sure"     │
│                 │
│ Dad: "When?"    │
│ You: "7pm"      │
│                 │
│ Dad: "Dinner?"  │◄── Query
│                 │
│ RAG finds:      │
│ • "sure" (sim 0.9)
│ • "7pm" (sim 0.6)
└─────────────────┘

After (Cross-Conversation RAG):
┌─────────────────┐    ┌─────────────────┐
│ Chat with Dad   │    │ Chat with Mom   │
│                 │    │                 │
│ Dad: "Dinner?"  │    │ Mom: "Lunch?"   │
│ You: "sure"     │    │ You: "yeah"     │
│                 │    │                 │
│ Dad: "When?"    │    │ Mom: "When?"    │
│ You: "7pm"      │    │ You: "12pm"     │
│                 │    │                 │
│ Dad: "Dinner?"  │◄───┤ Mom: "Dinner?"  │
└─────────────────┘    └─────────────────┘
        │                       │
        │    Same relationship  │
        │    (family/parent)    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ RAG finds:          │
        │ • "sure" (sim 0.9)  │
        │ • "yeah" (sim 0.85) │◄── Cross-convo!
        │ • "7pm" (sim 0.6)   │
        └─────────────────────┘

Result: More examples, better variety!
```

---

## Quick Reference

### File Organization

```
v3/
├── core/
│   ├── generation/
│   │   ├── reply_generator.py    ← Main orchestrator (900 lines)
│   │   ├── prompts.py            ← Prompt templates
│   │   ├── style_analyzer.py     ← Your texting style
│   │   └── context_analyzer.py   ← Intent & relationship
│   ├── embeddings/
│   │   ├── store.py              ← RAG search & storage
│   │   ├── relationship_registry.py  ← Cross-convo learning
│   │   └── model.py              ← Embedding model
│   ├── models/
│   │   ├── registry.py           ← LFM2.5-1.2B config
│   │   └── loader.py             ← MLX model loading
│   └── imessage/
│       └── reader.py             ← iMessage DB access
├── api/
│   ├── main.py                   ← FastAPI app
│   └── routes/
│       ├── health.py             ← GET /health
│       ├── conversations.py      ← GET /conversations
│       └── generate.py           ← POST /generate/replies
└── scripts/
    ├── profile_contacts.py       ← Label relationships
    └── index_messages.py         ← Build RAG index
```

### Essential Commands

```bash
# Setup
cd v3 && make install

# Test
make test-v3              # Quick test (0.36s)
make test                 # All tests

# Code quality  
make lint                 # Check style
make format               # Auto-format
make check                # Lint + test

# Run
make api                  # Start API server
make profile              # Profile contacts
make index                # Index messages

# Development
uv run python scripts/test_v3.py
uv run pytest tests/ -v
```

---

## Success Metrics

| Metric | Current | Target | Excellent |
|--------|---------|--------|-----------|
| Intent Match | 28% | 50% | 65% |
| Human Approval | ? | 60% | 80% |
| Generation Time | ~2s | <3s | <1s |
| Test Coverage | 17 tests | 30+ tests | 50+ tests |

---

## Next Steps

1. **Run the tests**: `make test-v3`
2. **Profile contacts**: `make profile`
3. **Index messages**: `make index`
4. **Generate samples**: Create 30 test cases
5. **Human evaluation**: Rate each 1-5
6. **Tune RAG**: Adjust weights based on results

The foundation is solid. Now we iterate on quality!
