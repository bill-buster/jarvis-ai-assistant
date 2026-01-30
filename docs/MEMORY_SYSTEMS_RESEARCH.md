# Memory Systems Research: Clawdbot + QMD Analysis for JARVIS

**Date**: 2026-01-27
**Purpose**: Evaluate Clawdbot's conversational memory and QMD's hybrid search for integration into JARVIS

---

## Executive Summary

This document analyzes two complementary open-source memory systems to identify patterns applicable to JARVIS:

- **Clawdbot**: Conversational agent memory (persistent context, user preferences, session history)
- **QMD**: Knowledge base search engine (hybrid retrieval, query expansion, re-ranking)

**Key Finding**: JARVIS could benefit from **QMD's retrieval techniques** for iMessage search and template matching, plus **Clawdbot's persistence layer** for user preferences and conversation history.

**Estimated Memory Impact**: QMD uses ~3.1GB for 3 models vs JARVIS's current ~5.5GB for 1 model. A combined system could fit within the 8GB budget by selectively loading models.

---

## System Comparison Matrix

| Aspect | Clawdbot | QMD | JARVIS (Current) |
|--------|----------|-----|------------------|
| **Primary Use Case** | Multi-turn conversations | Knowledge retrieval | iMessage Q&A |
| **Storage Model** | Markdown + sqlite-vec | Pure SQLite | In-memory templates |
| **Search Strategy** | 70% vector + 30% BM25 | Query expansion → RRF → Re-rank | 100% vector (0.7 threshold) |
| **Chunking** | 400 tokens, 80 overlap | 800 tokens, 15% overlap | N/A (static templates) |
| **Persistence** | Two-layer (daily + long-term) | Single indexed documents | None |
| **Memory Models** | 1 embedding model | 3 specialized models | 1 generation model |
| **Integration** | MCP server | MCP server | FastAPI only |
| **Code Language** | TypeScript | TypeScript (Bun) | Python |

---

## Part 1: QMD Deep Dive

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    QMD Search Pipeline                      │
└─────────────────────────────────────────────────────────────┘

User Query
    │
    ├──► Query Expansion (Qwen3-1.7B)
    │    └──► [Original Query ×2, Variant 1, Variant 2]
    │
    ├──► Parallel Retrieval (for each query)
    │    ├──► BM25 (FTS5) → Ranked list
    │    └──► Vector Search → Ranked list
    │
    ├──► RRF Fusion
    │    ├──► Combine all lists (k=60)
    │    ├──► Apply top-rank bonus (+0.05 for #1, +0.02 for #2-3)
    │    └──► Keep top 30 candidates
    │
    ├──► LLM Re-ranking (Qwen3-Reranker 0.6B)
    │    └──► Yes/No + logprob confidence
    │
    └──► Position-Aware Blending
         ├──► Rank 1-3:  75% retrieval, 25% reranker
         ├──► Rank 4-10: 60% retrieval, 40% reranker
         └──► Rank 11+:  40% retrieval, 60% reranker
```

### 1. Query Expansion

**Problem**: Single query may miss relevant results due to vocabulary mismatch.

**Solution**: Generate 2 alternative phrasings using Qwen3-1.7B:
1. **Original query** (weighted ×2 in fusion)
2. **Lexical variant** (keyword-focused rephrasing)
3. **Semantic variant** (conceptual equivalent)

**Example**:
```
Original: "authentication flow"
Variant 1: "login process steps"
Variant 2: "user credential verification"
```

**JARVIS Application**:
- Template matching: Expand user query before semantic search
- iMessage search: Generate alternative search terms for better recall

### 2. Reciprocal Rank Fusion (RRF)

**Formula**:
```python
RRF_score = Σ(weight / (k + rank))
where k = 60 (constant), weight = 2 for original query, 1 for variants
```

**Top-Rank Bonus**:
- Rank #1 in any list: +0.05 bonus
- Rank #2-3: +0.02 bonus

**Why**: Preserves documents that are perfect matches for the original query, even if expanded variants don't match.

**JARVIS Application**:
- Combine BM25 (exact keyword) + vector (semantic) searches for iMessage
- Merge template match + model generation scores

### 3. Position-Aware Blending

**The Problem**: Pure reranker confidence can override obviously correct exact matches.

**The Solution**: Trust retrieval more for top-ranked results, reranker more for lower ranks.

```python
def blend_score(retrieval_rank, retrieval_score, reranker_score):
    if retrieval_rank <= 3:
        return 0.75 * retrieval_score + 0.25 * reranker_score
    elif retrieval_rank <= 10:
        return 0.60 * retrieval_score + 0.40 * reranker_score
    else:
        return 0.40 * retrieval_score + 0.60 * reranker_score
```

**JARVIS Application**:
- Template matching: If template score ≥0.9, trust it more than model generation
- Reply suggestions: Blend template confidence + model quality score

### 4. Score Normalization

**Challenge**: Different search backends produce incomparable scores.

| Backend | Raw Score Range | Normalization |
|---------|----------------|---------------|
| BM25 (FTS5) | 0 to ~25+ | `abs(score)` → sigmoid |
| Vector | 0.0 to 1.0 | `1 / (1 + distance)` |
| Reranker | 0 to 10 | `score / 10` |

**Normalization Code** (from QMD):
```typescript
// BM25 sigmoid normalization
const normalizeBM25 = (score: number) => {
  return 1 / (1 + Math.exp(-score / 5));
};

// Min-max scaling for vectors
const normalizeVector = (distance: number) => {
  return 1 / (1 + distance);
};
```

**JARVIS Application**:
- Normalize template similarity (0.0-1.0) with BM25 keyword scores before RRF

### 5. Chunking Strategy

**QMD**: 800 tokens with 15% overlap
**Clawdbot**: 400 tokens with 80 token overlap

**Why the difference?**
- QMD: Optimized for retrieval (larger chunks = more context per result)
- Clawdbot: Optimized for memory indexing (smaller chunks = more granular recall)

**JARVIS Application**:
- iMessage conversations: 800-token chunks (like QMD) for better context
- User preferences: 200-400 token chunks (like Clawdbot) for precise recall

### 6. sqlite-vec Integration

**Schema** (from QMD):
```sql
-- Documents table (relational metadata)
CREATE TABLE documents (
  docid TEXT PRIMARY KEY,  -- 6-char content hash
  title TEXT,
  path TEXT,
  content TEXT
);

-- Vector table (virtual, using sqlite-vec extension)
CREATE VIRTUAL TABLE vectors_vec USING vec0(
  hash_seq TEXT PRIMARY KEY,  -- "docid_sequence"
  embedding FLOAT[768]
);

-- Chunking table
CREATE TABLE content_vectors (
  hash TEXT,      -- document hash
  seq INTEGER,    -- chunk sequence (0, 1, 2...)
  pos INTEGER,    -- character position in document
  text TEXT,      -- chunk content
  PRIMARY KEY (hash, seq)
);
```

**Two-Step Query Pattern** (avoids sqlite-vec JOIN hang):
```sql
-- Step 1: Find similar vectors
SELECT hash_seq, distance
FROM vectors_vec
WHERE embedding MATCH ?
ORDER BY distance
LIMIT 50;

-- Step 2: Lookup document metadata
SELECT d.*, cv.text
FROM content_vectors cv
JOIN documents d ON cv.hash = d.docid
WHERE cv.hash || '_' || cv.seq IN (?, ?, ...);
```

**JARVIS Application**:
- Pre-compute embeddings for all iMessage messages
- Use sqlite-vec for fast semantic search (no external vector DB needed)

### 7. Content Hashing for Deduplication

**QMD's Approach**:
```typescript
function generateDocid(content: string): string {
  const hash = crypto.createHash('sha256')
    .update(content)
    .digest('hex');
  return hash.slice(0, 6);  // 6-char prefix
}
```

**Benefits**:
- Identical content = identical docid (automatic deduplication)
- Reproducible across systems
- Stable references (docid doesn't change unless content changes)

**JARVIS Application**:
- Hash iMessage messages to detect duplicates/edits
- Cache embeddings by content hash (avoid recomputing)

### 8. MCP Server Implementation

**QMD exposes 6 tools**:
1. `qmd_search` - Fast BM25 keyword search
2. `qmd_vsearch` - Semantic vector search
3. `qmd_query` - Hybrid search with re-ranking
4. `qmd_get` - Retrieve single document
5. `qmd_multi_get` - Batch retrieve by pattern
6. `qmd_status` - Index health check

**Tool Schema Example**:
```typescript
{
  name: "qmd_query",
  description: "Hybrid search with re-ranking for best quality",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string" },
      collection: { type: "string", optional: true },
      limit: { type: "integer", default: 5 },
      minScore: { type: "number", default: 0.0 }
    }
  }
}
```

**JARVIS Application**:
- Expose `jarvis_search_messages`, `jarvis_reply`, `jarvis_summarize` tools
- Enable Claude Code to use JARVIS directly via MCP

---

## Part 2: Clawdbot Deep Dive

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Clawdbot Memory System                     │
└─────────────────────────────────────────────────────────────┘

Workspace: ~/clawd/
├── MEMORY.md              ← Layer 2: Long-term curated knowledge
└── memory/
    ├── 2026-01-26.md      ← Layer 1: Today's raw notes
    ├── 2026-01-25.md
    └── ...

Index: ~/.clawdbot/memory/
└── main.sqlite            ← sqlite-vec index (chunks + embeddings)

Every session start:
1. Read SOUL.md (agent identity)
2. Read USER.md (user preferences)
3. Read memory/YYYY-MM-DD.md (today + yesterday)
4. Read MEMORY.md (long-term knowledge)

Every conversation:
- memory_search: Find relevant context from all files
- memory_get: Retrieve specific content
- Writes to daily logs or MEMORY.md as needed
```

### 1. Two-Layer Memory

**Layer 1: Daily Logs** (`memory/YYYY-MM-DD.md`)
- Append-only notes throughout the day
- Agent writes here when it wants to remember something
- Raw, unfiltered stream of consciousness

**Example**:
```markdown
# 2026-01-26

## 10:30 AM - API Discussion
Discussed REST vs GraphQL with user. Decision: use REST for simplicity.
Key endpoints: /users, /auth, /projects.

## 2:15 PM - Deployment
Deployed v2.3.0 to production. No issues.

## 4:00 PM - User Preference
User mentioned they prefer TypeScript over JavaScript.
```

**Layer 2: Long-term Memory** (`MEMORY.md`)
- Curated, persistent knowledge
- Significant events, decisions, opinions, lessons learned
- Manually or automatically promoted from daily logs

**Example**:
```markdown
# Long-term Memory

## User Preferences
- Prefers TypeScript over JavaScript
- Likes concise explanations
- Working on project "Acme Dashboard"

## Important Decisions
- 2026-01-15: Chose PostgreSQL for database
- 2026-01-20: Adopted REST over GraphQL
- 2026-01-26: Using Tailwind CSS for styling

## Key Contacts
- Alice (alice@acme.com) - Design lead
- Bob (bob@acme.com) - Backend engineer
```

**JARVIS Application**:
- **Layer 1**: Daily iMessage interaction logs (queries, replies, preferences discovered)
- **Layer 2**: User profile (preferred tone, common contacts, relationship context)

### 2. Memory Search Tools

**Tool 1: memory_search**
```typescript
{
  name: "memory_search",
  description: "Semantically search MEMORY.md + memory/*.md",
  parameters: {
    query: string,
    maxResults: number,  // default: 6
    minScore: number     // default: 0.35
  }
}
```

**Returns**:
```json
{
  "results": [
    {
      "path": "memory/2026-01-20.md",
      "startLine": 45,
      "endLine": 52,
      "score": 0.87,
      "snippet": "## API Discussion\nDecided to use REST...",
      "source": "memory"
    }
  ]
}
```

**Tool 2: memory_get**
```typescript
{
  name: "memory_get",
  description: "Read specific lines from a memory file",
  parameters: {
    path: string,
    from: number,     // start line
    lines: number     // how many lines to read
  }
}
```

**JARVIS Application**:
- `jarvis_recall`: Search user preference and conversation history
- `jarvis_get_context`: Retrieve specific conversation by date or contact

### 3. Automatic Memory Indexing

**File Watcher Flow**:
```
File Saved (MEMORY.md or memory/*.md)
    │
    ▼
Chokidar detects change (debounced 1.5s)
    │
    ▼
Chunk into ~400 tokens with 80 token overlap
    │
    ▼
Generate embeddings (OpenAI/Gemini/Local)
    │
    ▼
Store in ~/.clawdbot/memory/<agentId>.sqlite
    ├── chunks (id, path, start_line, end_line, text, hash)
    ├── chunks_vec (id, embedding)  ← sqlite-vec
    ├── chunks_fts (text)           ← FTS5 full-text
    └── embedding_cache (hash, vector)
```

**Why 400 tokens with 80 overlap?**
- Balances semantic coherence vs granularity
- Overlap ensures facts spanning boundaries are captured in both chunks
- Both values are configurable

**JARVIS Application**:
- Watch `~/.jarvis/memory/` directory for user preference files
- Auto-index when files change
- Use same chunking strategy for conversation history

### 4. Hybrid Search (70/30 Split)

**Scoring**:
```python
final_score = (0.7 * vector_score) + (0.3 * text_score)
```

**Why 70/30?**
- Semantic similarity is primary signal for memory recall
- BM25 keyword matching catches exact terms that vectors might miss (names, IDs, dates)

**Results Filtering**:
- Filter out results below `minScore` threshold (default: 0.35)
- Return top `maxResults` (default: 6)

**JARVIS Application**:
- Same 70/30 split for template matching
- Adjust weights based on query type (casual chat: 80/20 semantic, specific search: 50/50)

### 5. Pre-Compaction Memory Flush

**The Problem**: LLM compaction is lossy - important info may be summarized away.

**The Solution**: Silent memory flush before compaction.

**Flow**:
```
Context usage: 75% of limit
    │
    ▼
Trigger silent memory flush turn
    │
    ├──► System: "Pre-compaction memory flush. Store durable
    │            memories now (use memory/YYYY-MM-DD.md).
    │            If nothing to store, reply with NO_REPLY."
    │
    ├──► Agent: Reviews conversation for important info
    │           Writes key decisions/facts to memory files
    │           Replies: NO_REPLY (user sees nothing)
    │
    └──► Compaction proceeds safely
         (Important information is now on disk)
```

**Configuration**:
```json
{
  "compaction": {
    "reserveTokensFloor": 20000,
    "memoryFlush": {
      "enabled": true,
      "softThresholdTokens": 4000,
      "systemPrompt": "Session nearing compaction. Store durable memories now.",
      "prompt": "Write lasting notes to memory/YYYY-MM-DD.md; reply NO_REPLY if nothing to store."
    }
  }
}
```

**JARVIS Application**:
- Before compacting chat history, flush important context to user profile
- Example: "User mentioned they prefer casual tone" → write to MEMORY.md

### 6. Session Lifecycle Hooks

**Session Memory Hook**:
```
User runs /new command
    │
    ▼
Extract last 15 messages from ending session
    │
    ▼
Generate descriptive slug via LLM
    │
    ▼
Save to memory/2026-01-26-api-design.md
    │
    ▼
New session starts (previous context now searchable)
```

**JARVIS Application**:
- When user switches contexts (e.g., from "Mom" to "Work group"), save previous context
- Auto-generate conversation summaries for later retrieval

### 7. Multi-Agent Memory Isolation

**Structure**:
```
~/.clawdbot/memory/          ← State directory (indexes)
├── main.sqlite              ← "main" agent vector index
└── work.sqlite              ← "work" agent vector index

~/clawd/                     ← "main" agent workspace
├── MEMORY.md
└── memory/

~/clawd-work/                ← "work" agent workspace
├── MEMORY.md
└── memory/
```

**Key Points**:
- Each agent gets own workspace and index
- Memory manager keyed by `agentId + workspaceDir`
- No cross-agent memory search by default
- Soft sandbox (default working directory, not hard boundary)

**JARVIS Application**:
- Personal agent: iMessage conversations, casual tone
- Work agent: Professional context, different prompts
- Each with isolated memory and preferences

---

## Part 3: Synthesis for JARVIS

### What JARVIS Needs

| Need | Current State | QMD Solution | Clawdbot Solution |
|------|--------------|--------------|-------------------|
| **Better template matching** | 100% vector, 0.7 threshold | Query expansion + RRF | 70/30 hybrid search |
| **iMessage search quality** | Basic text search | Hybrid BM25 + vector + re-rank | N/A (not search-focused) |
| **User preferences** | None | N/A | Two-layer memory (daily + long-term) |
| **Conversation history** | Session-only | N/A | Daily logs + compaction |
| **Claude integration** | None | MCP server | MCP server |
| **Embedding cache** | Recompute every time | Content hashing + sqlite cache | Embedding cache table |

### Recommended Hybrid Approach

**Phase 1: Enhanced Retrieval (QMD-inspired)**
1. Add sqlite-vec for iMessage embeddings
2. Implement hybrid search (BM25 + vector) for `search-messages` command
3. Add RRF fusion for template matching
4. Implement content hashing for embedding cache

**Phase 2: User Preferences (Clawdbot-inspired)**
1. Create `~/.jarvis/memory/` directory
2. Implement two-layer memory:
   - `~/.jarvis/memory/YYYY-MM-DD.md` (daily interaction logs)
   - `~/.jarvis/memory/USER.md` (long-term preferences)
3. Add memory indexing with file watcher
4. Expose `jarvis_recall` tool for memory search

**Phase 3: MCP Integration**
1. Implement MCP server in `jarvis/mcp.py`
2. Expose tools:
   - `jarvis_search_messages` (hybrid search)
   - `jarvis_reply` (generate reply with context)
   - `jarvis_summarize` (conversation summary)
   - `jarvis_recall` (search user preferences)
3. Add to Claude Code config

**Phase 4: Conversation Context**
1. Implement session history persistence
2. Add pre-compaction memory flush
3. Implement conversation summarization
4. Add context switching between contacts

---

## Implementation Patterns

### Pattern 1: Hybrid Template Matching

**Current** (templates.py):
```python
def match_template(query: str, group_size: int = 1) -> Optional[Template]:
    query_embedding = model.encode(query)

    for template in templates:
        if not template.is_valid_for_group(group_size):
            continue

        template_embedding = model.encode(template.text)
        similarity = cosine_similarity(query_embedding, template_embedding)

        if similarity >= 0.7:
            return template

    return None
```

**Proposed** (QMD + Clawdbot hybrid):
```python
def match_template(query: str, group_size: int = 1) -> Optional[Template]:
    # Step 1: Query expansion
    expanded_queries = expand_query(query)  # [original ×2, variant1, variant2]

    # Step 2: Parallel retrieval for each query
    all_results = []
    for q, weight in expanded_queries:
        # BM25 keyword search
        bm25_results = search_templates_bm25(q)

        # Vector semantic search
        vector_results = search_templates_vector(q)

        all_results.extend([(r, weight, 'bm25') for r in bm25_results])
        all_results.extend([(r, weight, 'vector') for r in vector_results])

    # Step 3: RRF fusion
    fused_results = rrf_fusion(all_results, k=60)

    # Step 4: Position-aware blending
    top_template = fused_results[0]
    if top_template.rank <= 3:
        # Trust retrieval heavily for top matches
        confidence = 0.75 * top_template.retrieval_score + 0.25 * top_template.rerank_score
    else:
        confidence = 0.60 * top_template.retrieval_score + 0.40 * top_template.rerank_score

    # Step 5: Apply threshold
    if confidence >= 0.7:
        return top_template.template

    return None
```

### Pattern 2: iMessage Embedding Cache

**Current**: Recompute embeddings every time.

**Proposed** (QMD content hashing):
```python
import hashlib
import sqlite3

def get_message_embedding(message_text: str) -> np.ndarray:
    # Generate content hash
    content_hash = hashlib.sha256(message_text.encode()).hexdigest()[:12]

    # Check cache
    conn = sqlite3.connect("~/.jarvis/cache/embeddings.db")
    cursor = conn.execute(
        "SELECT embedding FROM embedding_cache WHERE hash = ?",
        (content_hash,)
    )
    row = cursor.fetchone()

    if row:
        # Cache hit
        return np.frombuffer(row[0], dtype=np.float32)

    # Cache miss - compute and store
    embedding = model.encode(message_text)
    conn.execute(
        "INSERT INTO embedding_cache (hash, embedding) VALUES (?, ?)",
        (content_hash, embedding.tobytes())
    )
    conn.commit()

    return embedding
```

### Pattern 3: User Preference Learning

**Proposed** (Clawdbot two-layer memory):
```python
class UserPreferenceManager:
    def __init__(self):
        self.memory_dir = Path.home() / ".jarvis" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.daily_log = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        self.long_term = self.memory_dir / "USER.md"

    def record_interaction(self, event: str, details: str):
        """Append to today's log."""
        timestamp = datetime.now().strftime("%H:%M")
        with open(self.daily_log, "a") as f:
            f.write(f"\n## {timestamp} - {event}\n{details}\n")

    def update_preference(self, category: str, preference: str):
        """Update long-term memory."""
        # Read existing content
        content = self.long_term.read_text() if self.long_term.exists() else "# User Preferences\n"

        # Update category
        if f"## {category}" not in content:
            content += f"\n## {category}\n- {preference}\n"
        else:
            # Append to existing category
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line == f"## {category}":
                    lines.insert(i + 1, f"- {preference}")
                    break
            content = "\n".join(lines)

        self.long_term.write_text(content)

    def recall(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search memory (hybrid vector + BM25)."""
        # Use memory_search implementation (see Pattern 1)
        return memory_search(query, max_results=max_results, min_score=0.35)
```

### Pattern 4: RRF Fusion

**Implementation** (from QMD):
```python
def rrf_fusion(
    results: List[Tuple[Template, float, str]],  # (template, score, source)
    k: int = 60
) -> List[Tuple[Template, float]]:
    """
    Reciprocal Rank Fusion with top-rank bonus.
    """
    # Group results by source and query
    grouped = defaultdict(list)
    for template, score, source in results:
        grouped[source].append((template, score))

    # Compute RRF scores
    rrf_scores = defaultdict(float)
    for source, source_results in grouped.items():
        # Sort by score (descending)
        ranked = sorted(source_results, key=lambda x: x[1], reverse=True)

        for rank, (template, score) in enumerate(ranked, start=1):
            weight = 2.0 if source == 'original' else 1.0
            rrf_scores[template] += weight / (k + rank)

            # Top-rank bonus
            if rank == 1:
                rrf_scores[template] += 0.05
            elif rank <= 3:
                rrf_scores[template] += 0.02

    # Sort by RRF score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### Pattern 5: MCP Server for JARVIS

**Proposed** (jarvis/mcp.py):
```python
from mcp.server.stdio import serve_mcp
from mcp.types import Tool, TextContent

class JarvisMCPServer:
    def __init__(self):
        self.tools = [
            Tool(
                name="jarvis_search_messages",
                description="Search iMessage conversations with hybrid BM25 + vector search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                        "min_score": {"type": "number", "default": 0.3}
                    }
                }
            ),
            Tool(
                name="jarvis_reply",
                description="Generate reply suggestion for an iMessage conversation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "contact": {"type": "string"},
                        "instruction": {"type": "string", "optional": True}
                    }
                }
            ),
            Tool(
                name="jarvis_recall",
                description="Search user preferences and conversation history",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5}
                    }
                }
            )
        ]

    async def handle_tool_call(self, name: str, arguments: dict) -> TextContent:
        if name == "jarvis_search_messages":
            # Use hybrid search implementation
            results = hybrid_search_messages(**arguments)
            return TextContent(
                type="text",
                text=format_search_results(results)
            )

        elif name == "jarvis_reply":
            # Generate reply
            reply = generate_reply(**arguments)
            return TextContent(type="text", text=reply)

        elif name == "jarvis_recall":
            # Search memory
            memories = memory_search(**arguments)
            return TextContent(
                type="text",
                text=format_memories(memories)
            )

    def run(self):
        serve_mcp(
            server_name="jarvis",
            tools=self.tools,
            tool_handler=self.handle_tool_call
        )

if __name__ == "__main__":
    server = JarvisMCPServer()
    server.run()
```

**Claude Code Config** (~/.claude/settings.json):
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "python",
      "args": ["-m", "jarvis.mcp"]
    }
  }
}
```

### Pattern 6: Position-Aware Score Blending

**Implementation**:
```python
def blend_scores(
    retrieval_rank: int,
    retrieval_score: float,
    reranker_score: float
) -> float:
    """
    Position-aware blending: trust retrieval more for top ranks.
    """
    if retrieval_rank <= 3:
        # Top 3: Trust retrieval heavily (preserve exact matches)
        return 0.75 * retrieval_score + 0.25 * reranker_score
    elif retrieval_rank <= 10:
        # Top 10: Balanced
        return 0.60 * retrieval_score + 0.40 * reranker_score
    else:
        # Lower ranks: Trust semantic understanding
        return 0.40 * retrieval_score + 0.60 * reranker_score
```

---

## Memory Budget Analysis

### Current JARVIS

| Component | Model | Size | Usage |
|-----------|-------|------|-------|
| Generation | Qwen2.5-1.5B-Instruct-4bit | ~1.5GB | Always loaded |
| Embedding | all-MiniLM-L6-v2 | ~90MB | Always loaded |
| **Total** | | **~1.6GB** | **Base footprint** |

*Note: Current memory profiling shows ~5.5GB peak usage due to MLX allocations and context*

### QMD Models

| Component | Model | Size | Usage |
|-----------|-------|------|-------|
| Embedding | EmbeddingGemma-300M-Q8_0 | ~300MB | Embedding generation |
| Reranking | Qwen3-Reranker-0.6B-Q8_0 | ~640MB | Result re-ranking |
| Expansion | Qwen3-1.7B-Q8_0 | ~2.2GB | Query expansion |
| **Total** | | **~3.1GB** | **All loaded** |

### Proposed JARVIS + Memory System

**Option A: Keep Current Models + Add sqlite-vec**
```
Qwen2.5-1.5B-Instruct-4bit    ~1.5GB  (generation)
all-MiniLM-L6-v2              ~90MB   (embedding)
sqlite-vec index              ~50MB   (10K messages @ 768 dims)
----------------------------------------
Total:                        ~1.6GB  (well under 8GB budget)
```

**Option B: Replace with QMD Models**
```
EmbeddingGemma-300M           ~300MB  (embedding)
Qwen3-1.7B-Q8_0              ~2.2GB  (generation + expansion)
Qwen3-Reranker-0.6B          ~640MB  (re-ranking, loaded on-demand)
sqlite-vec index             ~50MB   (10K messages)
----------------------------------------
Total:                       ~3.2GB  (fits comfortably in 8GB)
```

**Option C: Selective Loading (Recommended)**
```
Base (always loaded):
- Qwen2.5-1.5B-Instruct-4bit  ~1.5GB  (generation)
- all-MiniLM-L6-v2            ~90MB   (embedding)

On-demand (load when needed):
- Qwen3-Reranker-0.6B         ~640MB  (only for complex searches)

Cached (pre-computed):
- sqlite-vec index            ~50MB   (all iMessage embeddings)
----------------------------------------
Total (base):                 ~1.6GB
Total (peak):                 ~2.2GB  (when reranker loaded)
```

**Recommendation**: **Option C** - Keeps current models, adds sqlite-vec for caching, loads reranker only for complex hybrid searches. Stays well within 8GB budget.

---

## Code Exploration Checklist

Before implementing, explore these repositories:

### Clawdbot
- [ ] Memory indexing pipeline (`src/memory/indexer.ts`)
- [ ] Hybrid search implementation (`src/memory/search.ts`)
- [ ] Memory tools (memory_search, memory_get)
- [ ] File watcher (Chokidar integration)
- [ ] Compaction + memory flush logic
- [ ] Multi-agent isolation

### QMD
- [ ] Query expansion prompts (`src/llm.ts`)
- [ ] RRF fusion algorithm (`src/search.ts`)
- [ ] Position-aware blending (`src/search.ts`)
- [ ] sqlite-vec schema and queries (`src/db.ts`)
- [ ] Content hashing for deduplication
- [ ] MCP server implementation (`src/mcp.ts`)

---

## Phased Implementation Roadmap

### Phase 1: Foundation (1 week)
**Goal**: Add sqlite-vec + basic caching

- [ ] Install sqlite-vec extension
- [ ] Create `~/.jarvis/cache/embeddings.db`
- [ ] Implement content hashing for messages
- [ ] Add embedding cache (hash → vector)
- [ ] Write tests for cache hit/miss

**Deliverable**: iMessage embeddings are cached, reducing recomputation

### Phase 2: Hybrid Search (1 week)
**Goal**: Improve iMessage search quality

- [ ] Implement BM25 search using FTS5
- [ ] Add RRF fusion for combining BM25 + vector
- [ ] Add score normalization (BM25 sigmoid, vector cosine)
- [ ] Update `search-messages` command to use hybrid search
- [ ] Write tests comparing old vs new search quality

**Deliverable**: `jarvis search-messages` uses hybrid retrieval

### Phase 3: User Preferences (1 week)
**Goal**: Persistent memory for user context

- [ ] Create `~/.jarvis/memory/` directory structure
- [ ] Implement `UserPreferenceManager` class
- [ ] Add file watcher for auto-indexing
- [ ] Create `jarvis_recall` function
- [ ] Write tests for preference storage/retrieval

**Deliverable**: JARVIS remembers user preferences across sessions

### Phase 4: Template Enhancement (3 days)
**Goal**: Better template matching

- [ ] Add query expansion (using Qwen2.5-1.5B)
- [ ] Implement RRF for template matching
- [ ] Add position-aware blending
- [ ] Update `match_template()` function
- [ ] Write tests with edge cases

**Deliverable**: Template matching is more accurate

### Phase 5: MCP Integration (3 days)
**Goal**: Claude Code can use JARVIS

- [ ] Implement `jarvis/mcp.py` server
- [ ] Expose 3 tools (search, reply, recall)
- [ ] Add to Claude Code config
- [ ] Test with Claude Code
- [ ] Document MCP usage

**Deliverable**: `jarvis mcp-serve` command works with Claude Code

### Phase 6: Advanced Features (1 week)
**Goal**: Conversation history + compaction

- [ ] Implement session persistence
- [ ] Add pre-compaction memory flush
- [ ] Add conversation summarization
- [ ] Implement context switching
- [ ] Write integration tests

**Deliverable**: JARVIS maintains conversation context across sessions

---

## Open Questions

1. **Model Selection**: Keep current Qwen2.5-1.5B or switch to QMD's Qwen3-1.7B?
   - **Consideration**: Qwen3-1.7B is designed for query expansion, may be better suited
   - **Trade-off**: Slightly larger (+200MB), but still fits in budget

2. **Chunking Strategy**: 800 tokens (QMD) or 400 tokens (Clawdbot)?
   - **For iMessage**: 800 tokens (conversations need more context)
   - **For Preferences**: 200-400 tokens (precise recall)

3. **Memory Flush Timing**: When to trigger?
   - **Option A**: Time-based (every 30 minutes)
   - **Option B**: Message count (every 100 messages)
   - **Option C**: Context usage (at 75% of limit, like Clawdbot)

4. **MCP vs FastAPI**: Replace FastAPI or run both?
   - **Recommendation**: Keep FastAPI for Tauri, add MCP for Claude Code
   - Both can coexist - different use cases

5. **Reranker Model**: Worth the 640MB?
   - **For quick replies**: No, template matching is sufficient
   - **For complex searches**: Yes, improves precision
   - **Recommendation**: Load on-demand only for `query` mode

---

## Success Metrics

After implementation, measure:

1. **Template Match Accuracy**: % of queries where hybrid > vector-only
2. **Search Quality**: User satisfaction with `search-messages` results
3. **Cache Hit Rate**: % of embeddings served from cache
4. **Memory Usage**: Peak RAM (should stay under 8GB)
5. **Latency Impact**: Query expansion + RRF overhead (target: <500ms)

---

## References

- **Clawdbot**: https://github.com/psteinbachs/clawdbot (find correct repo)
- **QMD**: https://github.com/tobi/qmd
- **sqlite-vec**: https://github.com/asg017/sqlite-vec
- **node-llama-cpp**: https://github.com/withcatai/node-llama.cpp
- **MCP Specification**: https://modelcontextprotocol.io/

---

## Next Steps

1. **Validate Assumptions**: Run benchmarks to confirm memory budgets
2. **Prototype RRF**: Test RRF fusion with current templates (1 day)
3. **Test sqlite-vec**: Benchmark query performance with 10K messages (1 day)
4. **Explore Codebases**: Deep dive into Clawdbot + QMD implementations (2 days)
5. **Write Design Doc**: Detailed implementation plan for Phase 1 (1 day)

**Estimated Total Time**: 6-8 weeks for full implementation (all phases)
**Quick Win**: Phase 1 + 2 (hybrid search) can be done in 2 weeks

---

*Document created: 2026-01-27*
*Status: Research complete, awaiting implementation decision*
