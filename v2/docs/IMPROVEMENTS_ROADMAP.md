# JARVIS v2 Improvements Roadmap

**Created**: 2026-01-28
**Vision**: Local-first AI assistant with Clawdbot-level capabilities, understanding your schedule, commitments, and communication patterns.

---

## Table of Contents

1. [Vision & Goals](#vision--goals)
2. [Research Findings](#research-findings)
3. [Current State Analysis](#current-state-analysis)
4. [Architecture Decisions](#architecture-decisions)
5. [Model Strategy](#model-strategy)
6. [Implementation Phases](#implementation-phases)
7. [V1 Assets to Port](#v1-assets-to-port)
8. [Technical Specifications](#technical-specifications)

---

## Vision & Goals

### What JARVIS Should Understand

| Capability | Current | Target |
|------------|---------|--------|
| Message classification | Basic rule-based | ML-based with context |
| Your schedule | None | Calendar + commitment tracking |
| What you're doing | None | Context from recent messages |
| What you need to do | None | Action items + follow-ups |
| Cross-conversation awareness | None | "You told Sarah you'd be there" |
| Proactive reminders | None | Push notifications for commitments |

### Differentiation from Clawdbot

| Aspect | Clawdbot | JARVIS |
|--------|----------|--------|
| Focus | General assistant (flights, emails, etc.) | iMessage-specific, deep specialization |
| Data scope | All your files, calendar, email | Just iMessage + Calendar (privacy) |
| Style | Generic Claude responses | Learns YOUR texting style |
| Model | Cloud API (Anthropic) | 100% local MLX (no data leaves device) |
| Memory | Cloud-synced | Local SQLite + knowledge graph |

---

## Research Findings

### Clawdbot/Moltbot
- **Source**: https://github.com/clawdbot/clawdbot
- **Key insight**: "Claude with hands" - does things, not just chats
- **Features**: Long-term memory, cross-platform, browser automation
- **Concern**: Stores secrets in plaintext (we'll do better)

### QMD (by Tobi Lütke)
- **Source**: https://github.com/tobi/qmd
- **Key insight**: Hybrid search = BM25 + vector + LLM re-ranking
- **Already implemented**: `find_similar_hybrid()` in `core/embeddings/store.py` uses RRF
- **To add**: Query expansion, position-aware blending

### Supermemory
- **Source**: https://supermemory.ai/
- **Key insight**: Human-like memory with decay, recency bias, smart forgetting
- **Features**: Multi-sector memory (episodic, semantic, procedural, emotional)
- **To adopt**: Salience scoring, knowledge graphs, memory decay

---

## Current State Analysis

### Intent Classification (Weak Point)

Current implementation in `core/generation/context_analyzer.py`:

```python
class MessageIntent(Enum):
    YES_NO_QUESTION = "yes_no_question"
    OPEN_QUESTION = "open_question"
    CHOICE_QUESTION = "choice_question"
    STATEMENT = "statement"
    EMOTIONAL = "emotional"
    GREETING = "greeting"
    LOGISTICS = "logistics"
    SHARING = "sharing"
    THANKS = "thanks"
    FAREWELL = "farewell"
```

**Problems**:
1. Pure rule-based keyword matching
2. No learning from patterns
3. No temporal awareness ("tomorrow" isn't resolved)
4. No commitment tracking
5. No cross-conversation context

### What Works Well

- **Embedding store**: FAISS indexing, hybrid BM25+vector search
- **Style learning**: Finds past replies to similar messages
- **Template matching**: Fast path for common responses
- **LFM2.5 generation**: Fast, natural text replies

---

## Code Quality Improvements

Technical debt and refactoring opportunities identified via code review (2026-01-28).

### Completed ✅

| Issue | File(s) | Fix |
|-------|---------|-----|
| Duplicate emoji regex (3 places) | `reply_generator.py`, `style_analyzer.py`, `contact_profiler.py` | Created `core/utils/emoji.py` shared utility |
| FAISS cache grows unbounded | `core/embeddings/store.py` | Added LRU eviction (max 50 indices) |
| Fragile hash-based message tracking | `core/generation/reply_generator.py:120` | Replaced `hash()` with MD5 for stability |
| Silent exception catching masks errors | `reply_generator.py:22-49` | Catch `ImportError` separately, log unexpected errors at warning level |
| Dead code in `get_reply_strategy()` | `context_analyzer.py` | Removed unused function |
| Thread-safe singleton race condition | `store.py`, `loader.py`, `contact_profiler.py` | All now use `@lru_cache(maxsize=1)` |
| ReplyGenerator has 4 separate state caches | `reply_generator.py` | Consolidated into `ChatState` dataclass |
| Duplicate style instruction builders | `reply_generator.py`, `StyleAnalyzer` | Removed wrapper, use `StyleAnalyzer.build_style_instructions()` directly |
| Inefficient `get_user_messages` query | `core/imessage/reader.py` | Now uses `get_messages(..., is_from_me=True)` which filters in SQL |
| Contact cache reloads from disk | `core/imessage/reader.py` | Cache is instance variable, persists for reader lifetime |
| Stop words defined in 2 places | `store.py`, `contact_profiler.py` | Created `core/utils/text.py` with shared `STOP_WORDS` |
| Magic numbers without constants | `reply_generator.py` | Added constants: `TEMPERATURE_SCALE`, `MAX_REPLY_TOKENS`, `TEMPLATE_CONFIDENCE`, `PAST_REPLY_CONFIDENCE` |
| Inconsistent error handling in routes | `api/routes/settings.py` | Moved `HTTPException` import to top of file |
| No type hints on `messages` param | `store.py:369` | Added `MessageDict` TypedDict in `core/utils/text.py` |

### Deferred / Won't Fix

| Issue | File(s) | Decision |
|-------|---------|----------|
| ConversationContext fields (`relationship`, `mood`, `summary`) | `context_analyzer.py:38-51` | **Keeping** - may be used for future features |
| `_get_coherent_messages` is complex | `reply_generator.py` | **Deferred** - may refactor or delete later |

---

## Architecture Decisions

### Knowledge Graph vs Embeddings: Use Both

| Use Case | Solution |
|----------|----------|
| "Find messages similar to X" | Embeddings (semantic similarity) |
| "What did I commit to?" | Knowledge Graph (explicit relationships) |
| "Am I free Tuesday?" | Knowledge Graph (multi-hop reasoning) |
| "How do I reply to Sarah?" | Embeddings (pattern matching) |

**Implementation**: HybridRAG with SQLite for both:
- Extend existing `embeddings.db` with commitment tables
- Simple adjacency list for relationships
- Migrate to DuckDB/LanceDB if scale requires

### Storage: SQLite (Not In-Memory)

In-memory doesn't scale. Continue using SQLite with:
- WAL mode for concurrent access
- FAISS indices persisted to disk
- New tables for commitments and entities

---

## Model Strategy

### Multi-Model Pipeline

Don't use 1.2B LFM for everything. Use specialists:

| Task | Model | Size | Speed |
|------|-------|------|-------|
| Embeddings | all-MiniLM-L6-v2 | **22M** | <10ms |
| Intent Classification | Fine-tuned DistilBERT | **66M** | <20ms |
| NER/Slot Filling | ModernBERT-base | **149M** | <30ms |
| Date/Time Parsing | dateparser (rules) | **0** | <1ms |
| Commitment Extraction | LFM2-350M (fine-tuned) | **350M** | ~100ms |
| Reply Generation | LFM2.5-1.2B | **1.2B** | ~500ms |

### Fine-Tuning LFM

Liquid AI recommends fine-tuning for narrow use cases:

1. **LFM2.5-1.2B-Base** for heavy fine-tuning (not Instruct)
2. **DPO (Direct Preference Optimization)** with LoRA
3. Train on: `(their_message, your_actual_reply)` pairs

**Fine-tuning targets**:
- Your texting style (capitalization, emoji, abbreviations)
- Commitment extraction (structured JSON output)
- Calendar-aware response generation

### Specialist Models to Evaluate

| Model | Params | Use Case | Source |
|-------|--------|----------|--------|
| ModernBERT-base | 149M | NER, classification | https://huggingface.co/answerdotai/ModernBERT-base |
| DistilBERT | 66M | Intent classification | Fine-tune on dialog acts |
| TinyBERT | 14M | Sentiment | https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D |
| LFM2-350M | 350M | Structured extraction | Fine-tune for commitments |
| LFM2-700M | 700M | Complex reasoning | Backup for hard cases |

---

## Implementation Phases

### Phase 1: Foundation (Start Here)

**Goal**: Storage + basic commitment extraction

#### 1a. Knowledge Graph Schema

```sql
-- Commitments (what you said you'd do)
CREATE TABLE commitments (
    id TEXT PRIMARY KEY,
    what TEXT NOT NULL,              -- "dinner", "pick up groceries"
    when_time INTEGER,               -- Unix timestamp (nullable)
    when_text TEXT,                  -- Original: "tomorrow at 7"
    who TEXT,                        -- Contact name
    chat_id TEXT NOT NULL,
    source_message_id INTEGER,
    your_message_id INTEGER,         -- Your reply confirming
    status TEXT DEFAULT 'pending',   -- pending, confirmed, cancelled, completed
    confidence REAL,
    created_at INTEGER,
    reminder_sent INTEGER DEFAULT 0
);

-- Named entities extracted from messages
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,              -- person, place, event, time
    chat_id TEXT,
    first_seen INTEGER,
    last_seen INTEGER
);

-- Relationships between entities
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,        -- entity or "user"
    predicate TEXT NOT NULL,         -- "committed_to", "attending", "knows"
    object_id TEXT NOT NULL,
    chat_id TEXT,
    created_at INTEGER,
    valid_until INTEGER              -- For time-bound relationships
);

-- Unanswered questions (need response tracking)
CREATE TABLE pending_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    question_text TEXT,
    asked_at INTEGER,
    answered_at INTEGER,
    answer_message_id INTEGER
);

CREATE INDEX idx_commitments_status ON commitments(status);
CREATE INDEX idx_commitments_when ON commitments(when_time);
CREATE INDEX idx_pending_answered ON pending_responses(answered_at);
```

#### 1b. Basic Commitment Extraction

Start with structured prompting (no fine-tuning):

```python
def extract_commitment(message: str, sender: str, your_reply: str) -> Commitment | None:
    """Extract commitment from a message exchange."""
    prompt = f"""Extract any commitment from this conversation.

They said: "{message}"
You replied: "{your_reply}"

If you committed to something, return JSON:
{{"what": "description", "when": "time expression or null", "confidence": 0.0-1.0}}

If no commitment, return: null
"""
    result = model.generate(prompt, max_tokens=100)
    return parse_commitment(result)
```

---

### Phase 2: Context (Calendar + Cross-Conversation)

**Goal**: Know your schedule, detect conflicts

#### 2a. Calendar Integration

**DONE**: Ported to v2 in `core/calendar/`:
- `EventDetector` - Detect events in message text (works without permissions)
- `CalendarReader` - Read events via AppleScript (requires Automation permission)

```python
from core.calendar import get_event_detector, get_calendar_reader, check_for_conflicts

# Event detection (always works, no permissions needed)
detector = get_event_detector()
events = detector.detect_events("dinner tomorrow at 7pm")
# -> DetectedEvent(title="Dinner", start=<tomorrow 7pm>, confidence=0.8)

# Calendar reading (requires permission)
reader = get_calendar_reader()
if reader.check_access():
    events = reader.get_events()
    is_busy, conflict = reader.is_busy_at(proposed_time)

# Conflict checking helper
conflicts = check_for_conflicts("dinner tomorrow at 7pm")
```

**Permissions**: Calendar reading requires **Automation** permission:
- System Settings > Privacy & Security > Automation
- Enable Calendar for your terminal app

#### 2b. Cross-Conversation Awareness

Query commitments when generating replies:

```python
def get_relevant_commitments(when: datetime, who: str | None = None) -> list[Commitment]:
    """Find commitments that might conflict."""
    # Check within 2 hours of proposed time
    window_start = when - timedelta(hours=2)
    window_end = when + timedelta(hours=2)

    return db.execute("""
        SELECT * FROM commitments
        WHERE status = 'pending'
          AND when_time BETWEEN ? AND ?
        ORDER BY when_time
    """, (window_start.timestamp(), window_end.timestamp())).fetchall()
```

---

### Phase 3: Intelligence (Specialist Models)

**Goal**: Better accuracy, faster classification

#### 3a. ModernBERT for NER

Replace rule-based entity extraction:

```python
from transformers import pipeline

ner = pipeline("ner", model="answerdotai/ModernBERT-base")

def extract_entities(text: str) -> list[Entity]:
    results = ner(text)
    return [
        Entity(name=r["word"], type=r["entity_group"])
        for r in results
    ]
```

#### 3b. Fine-tune Intent Classifier

Train DistilBERT on dialog act corpus + your messages:

```python
# Training data format
training_examples = [
    ("want to grab dinner tomorrow?", "scheduling_request"),
    ("sounds good!", "affirmative"),
    ("I'll be there at 7", "commitment"),
    ("running late", "logistics_update"),
]
```

#### 3c. Fine-tune LFM2-350M for Extraction

Use Liquid AI's cookbook:
- LoRA fine-tuning for memory efficiency
- Train on commitment extraction pairs
- Output: structured JSON

---

### Phase 4: Proactive (Reminders + Style)

**Goal**: JARVIS that reminds you, sounds like you

#### 4a. Reminder Daemon

Background process (LaunchAgent on macOS):

```python
# ~/.jarvis/daemon.py
import schedule
import time
from jarvis.notifications import send_notification
from jarvis.commitments import get_upcoming_commitments

def check_reminders():
    upcoming = get_upcoming_commitments(within_minutes=30)
    for commitment in upcoming:
        if not commitment.reminder_sent:
            send_notification(
                title=f"Reminder: {commitment.what}",
                body=f"You told {commitment.who} you'd be there",
                chat_id=commitment.chat_id
            )
            mark_reminder_sent(commitment.id)

schedule.every(5).minutes.do(check_reminders)

while True:
    schedule.run_pending()
    time.sleep(60)
```

#### 4b. Fine-tune LFM2.5 on Your Style

DPO training with your message history:

```python
# Generate preference pairs
training_pairs = [
    {
        "prompt": "They said: 'want to hang tomorrow?'",
        "chosen": "yeah down",      # Your actual reply
        "rejected": "Yes, I would like that!"  # Generic
    },
]
```

---

## V1 Assets to Port

### High Priority (Ready to Use)

| Module | Location | What It Does | Port Effort |
|--------|----------|--------------|-------------|
| **Calendar Integration** | `integrations/calendar/` | Read/write macOS calendar | Low - copy + adapt imports |
| **Relationships** | `jarvis/relationships.py` | Communication profile per contact | Medium - integrate with embeddings |
| **Priority Scoring** | `jarvis/priority.py` | Score message importance | Low - standalone |
| **Action Item Detection** | `jarvis/digest.py` | Extract TODOs from messages | Low - extract patterns |

### Medium Priority (Useful Features)

| Module | Location | What It Does | Port Effort |
|--------|----------|--------------|-------------|
| **Insights** | `jarvis/insights.py` | Sentiment analysis, response patterns | Medium |
| **Digest Generator** | `jarvis/digest.py` | Daily/weekly summaries | Medium |
| **Context Manager** | `jarvis/context.py` | Conversation context handling | Low |

### Research to Reference

| Document | Location | Contents |
|----------|----------|----------|
| **QMD Research** | `docs/QMD_RESEARCH.md` | Detailed hybrid search patterns |
| **Memory Systems** | `docs/MEMORY_SYSTEMS_RESEARCH.md` | Clawdbot + QMD analysis |
| **Model Selection** | `docs/MODEL_SELECTION_STRATEGY.md` | Model evaluation criteria |

---

## Technical Specifications

### New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INCOMING MESSAGE                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Classification (<50ms total)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Intent       │  │ NER/Entities │  │ Date/Time    │          │
│  │ DistilBERT   │  │ ModernBERT   │  │ dateparser   │          │
│  │ (66M)        │  │ (149M)       │  │ (rules)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────┐
│ SCHEDULING?  │    │ COMMITMENT?  │      │ JUST CHAT?   │
│              │    │              │      │              │
│ Check calendar│    │ Extract &    │      │ Skip to      │
│ via EventKit │    │ store in KG  │      │ generation   │
└──────────────┘    └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Context Retrieval (<100ms)                            │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ Embedding Search     │  │ Knowledge Graph      │            │
│  │ (past similar msgs)  │  │ (commitments, cal)   │            │
│  │ FAISS + BM25 hybrid  │  │ SQLite queries       │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Generation (LFM2.5-1.2B, ~500ms)                     │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Prompt includes:                                  │          │
│  │ - Your style (from embeddings)                   │          │
│  │ - Your commitments (from KG)                     │          │
│  │ - Calendar conflicts detected                    │          │
│  │ - Past similar replies                           │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Post-processing                                       │
│  - Extract NEW commitments from your reply                      │
│  - Update knowledge graph                                       │
│  - Schedule proactive reminder if needed                        │
└─────────────────────────────────────────────────────────────────┘
```

### Commitment Data Model

```python
@dataclass
class Commitment:
    id: str
    what: str                           # "dinner at Italian place"
    when_time: datetime | None          # Resolved timestamp
    when_text: str | None               # Original: "tomorrow at 7"
    who: str                            # Contact name
    chat_id: str
    source_message_id: int              # Their message
    your_message_id: int | None         # Your confirming reply
    status: Literal["pending", "confirmed", "cancelled", "completed"]
    confidence: float                   # Extraction confidence
    reminder_sent: bool
    created_at: datetime

@dataclass
class CalendarConflict:
    commitment: Commitment
    conflicting_event: CalendarEvent
    overlap_minutes: int
    suggestion: str                     # "You have a meeting at 6:30"
```

### Memory Budget

Target: Run everything within 8GB RAM

| Component | Memory | When Loaded |
|-----------|--------|-------------|
| all-MiniLM-L6-v2 | ~100MB | Always (embeddings) |
| DistilBERT (intent) | ~250MB | Always |
| ModernBERT (NER) | ~500MB | On demand |
| LFM2.5-1.2B-4bit | ~800MB | On demand |
| SQLite + FAISS | ~200MB | Always |
| **Total active** | ~1.3GB | Typical |
| **Peak (generation)** | ~2.1GB | During reply |

---

## Success Metrics

### Phase 1 Complete When:
- [ ] Commitments table exists and is populated
- [ ] Basic extraction works for "yes I'll be there at 7" type messages
- [ ] Can query "what did I commit to this week?"

### Phase 2 Complete When:
- [ ] Calendar events are readable
- [ ] Conflict detection works: "You have dinner with Mom at 6"
- [ ] Cross-conversation: "You already told Sarah you'd be busy"

### Phase 3 Complete When:
- [ ] Intent classification accuracy >85% on test set
- [ ] NER extracts people, places, times correctly
- [ ] Generation latency <600ms end-to-end

### Phase 4 Complete When:
- [ ] Daemon runs in background, sends notifications
- [ ] Fine-tuned model sounds like you (blind test)
- [ ] Zero-shot: "What's on my plate this week?" works

---

## Open Questions

1. **Calendar write access**: Should JARVIS create calendar events, or just read?
2. **Notification UX**: macOS native vs in-app notifications?
3. **Privacy**: How to handle commitments with sensitive content?
4. **Sync**: Should knowledge graph sync across devices?
5. **Fallback**: What happens when extraction confidence is low?

---

## References

- [Clawdbot GitHub](https://github.com/clawdbot/clawdbot)
- [QMD GitHub](https://github.com/tobi/qmd)
- [Supermemory](https://supermemory.ai/)
- [ModernBERT](https://huggingface.co/blog/modernbert)
- [LFM2.5 Fine-tuning](https://github.com/Liquid4All/cookbook)
- [DPO Fine-tuning Guide](https://www.analyticsvidhya.com/blog/2026/01/lfm-2-preference-fine-tuning-using-dpo/)
- v1 docs: `../docs/QMD_RESEARCH.md`, `../docs/MEMORY_SYSTEMS_RESEARCH.md`
