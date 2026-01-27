# QMD Search Engine Research

**Research Date**: 2026-01-27
**Repository**: https://github.com/tobi/qmd
**Purpose**: Extract implementation patterns for hybrid search system applicable to JARVIS

---

## Executive Summary

QMD (Quick Markdown Search) is a local-first hybrid search engine combining BM25 full-text search, vector semantic search, and LLM re-ranking. It implements a sophisticated three-stage pipeline with position-aware blending that preserves high-confidence exact matches while allowing semantic understanding to influence lower-ranked results.

**Key Technologies:**
- **Runtime**: Bun + TypeScript
- **Vector Store**: sqlite-vec with cosine similarity
- **Embeddings**: EmbeddingGemma-300M (Q8_0, ~300MB)
- **Re-ranker**: Qwen3-Reranker-0.6B (~640MB)
- **Query Expansion**: Qwen3-1.7B (~2.2GB)
- **Full-Text**: SQLite FTS5 with BM25

**Memory Footprint**: ~3.1GB total for all models

---

## 1. Three-Stage Pipeline Architecture

### Stage 1: Query Expansion

The system generates alternative query phrasings to improve recall before retrieval.

**Prompt Template Structure:**

```
Role: "You are a search query optimization expert. Your task is to improve
retrieval by rewriting queries and generating hypothetical documents."

Analysis Steps:
1. Analyze the query intent
2. Generate hypothetical document that would answer the query
3. Rewrite query in multiple ways
4. Produce final retrieval text

Output Format:
  lex: {single search term}      # Lexical keywords
  vec: {single vector query}      # Semantic queries
  hyde: {hypothetical document}   # HyDE (Hypothetical Document Embeddings)

Rules:
- 1-3 'lex' lines
- 1-3 'vec' lines
- MAX ONE 'hyde' line
- No duplicates
- Each variation must differ semantically
```

**Grammar Constraints:**

```typescript
root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+
```

**Implementation:**

```typescript
async function expandQueryStructured(
  query: string,
  includeLexical: boolean = true,
  context?: string
): Promise<Queryable[]> {
  process.stderr.write(`${c.dim}Expanding query...${c.reset}\n`);

  const llm = getDefaultLlamaCpp();
  const queryables = await llm.expandQuery(query, { includeLexical, context });

  // Original query is always included
  // Returns array of Queryable objects with:
  //   - text: the query text
  //   - type: 'lex' | 'vec' | 'hyde'
  //   - search method: lexical + vector for original/lex, vector-only for vec/hyde

  return queryables;
}
```

**Key Insight**: Original query receives special treatment (2x weight in RRF) to ensure it's not undermined by expansions.

---

### Stage 2: Parallel Retrieval

Each query variant (original + expansions) searches both FTS5 and vector indexes simultaneously.

**Vector Index Creation:**

```typescript
function ensureVecTableInternal(db: Database, dimensions: number): void {
  // Creates virtual table using vec0 with cosine distance
  db.exec(`
    CREATE VIRTUAL TABLE vectors_vec USING vec0(
      hash_seq TEXT PRIMARY KEY,
      embedding float[${dimensions}] distance_metric=cosine
    )
  `);
}
```

**Dual-Table Storage:**
1. **content_vectors** (relational): Metadata (hash, sequence, position, model, timestamp)
2. **vectors_vec** (virtual): Actual embedding vectors for similarity search

**Two-Step Query Pattern** (avoids sqlite-vec hanging with JOINs):

```typescript
// Step 1: Get vector matches without joins
const vectorMatches = db.prepare(`
  SELECT hash_seq, distance
  FROM vectors_vec
  WHERE embedding MATCH ?
  ORDER BY distance
  LIMIT ?
`).all(queryEmbedding, limit);

// Step 2: Lookup document metadata using hashes
const results = db.prepare(`
  SELECT d.path, d.title, cv.body
  FROM documents d
  JOIN content_vectors cv ON d.hash = cv.hash
  WHERE cv.hash_seq IN (${vectorMatches.map(m => m.hash_seq).join(',')})
`).all();
```

**Chunking Strategy:**
- 800 tokens per chunk
- 15% overlap between chunks
- Each chunk gets embedded independently
- `hash_seq` key: `{content_hash}_{sequence_number}`

---

### Stage 3: Fusion & Re-ranking

**Reciprocal Rank Fusion (RRF) Implementation:**

```typescript
function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, {
    score: number;
    displayPath: string;
    title: string;
    body: string;
    bestRank: number
  }>();

  // Compute RRF scores with per-list weighting
  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const results = resultLists[listIdx];
    if (!results) continue;

    const weight = weights[listIdx] ?? 1.0;  // Original query gets 2x weight

    for (let rank = 0; rank < results.length; rank++) {
      const doc = results[rank];
      if (!doc) continue;

      const rrfScore = weight / (k + rank + 1);  // Classic RRF formula

      const existing = scores.get(doc.file);
      if (existing) {
        existing.score += rrfScore;
        existing.bestRank = Math.min(existing.bestRank, rank);
      } else {
        scores.set(doc.file, {
          score: rrfScore,
          displayPath: doc.displayPath,
          title: doc.title,
          body: doc.body,
          bestRank: rank
        });
      }
    }
  }

  // Add top-rank bonus
  return Array.from(scores.entries())
    .map(([file, { score, displayPath, title, body, bestRank }]) => {
      let bonus = 0;
      if (bestRank === 0) bonus = 0.05;           // 5% boost for #1 rank
      else if (bestRank <= 2) bonus = 0.02;       // 2% boost for top-3

      return {
        file,
        displayPath,
        title,
        body,
        score: score + bonus
      };
    })
    .sort((a, b) => b.score - a.score);
}
```

**Weighting Strategy:**
- Original query: 2.0x weight
- Expanded queries: 1.0x weight (default)
- k = 60 constant (standard RRF parameter)

**Top-Rank Bonus:**
- Rank #1 in any list: +0.05 (5%)
- Ranks #2-3: +0.02 (2%)
- Ranks #4+: no bonus

**Rationale**: Documents appearing at the top of any result list likely have strong relevance signals worth preserving.

---

## 2. Position-Aware Blending

After RRF fusion, the top 30 candidates proceed to LLM re-ranking. Final scores blend retrieval confidence with re-ranker scores based on RRF position.

**Implementation:**

```typescript
const finalResults = Array.from(aggregatedScores.entries()).map(
  ([file, { score: rerankScore, bestChunkIdx }]) => {
    const rrfRank = rrfRankMap.get(file) || 30;

    // Position-aware blending weights
    let rrfWeight: number;
    if (rrfRank <= 3) {
      rrfWeight = 0.75;  // Top-3: 75% RRF, 25% reranker
    } else if (rrfRank <= 10) {
      rrfWeight = 0.60;  // Rank 4-10: 60% RRF, 40% reranker
    } else {
      rrfWeight = 0.40;  // Rank 11+: 40% RRF, 60% reranker
    }

    const rrfScore = 1 / rrfRank;  // Simple rank reciprocal
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * rerankScore;

    return {
      file,
      score: blendedScore,
      bestChunkIdx,
      // ... other fields
    };
  }
).sort((a, b) => b.score - a.score);
```

**Blending Formula Breakdown:**

| RRF Rank | RRF Weight | Reranker Weight | Rationale |
|----------|------------|-----------------|-----------|
| 1-3 | 75% | 25% | High-confidence exact matches - preserve retrieval signal |
| 4-10 | 60% | 40% | Balanced weighting for middle ranks |
| 11+ | 40% | 60% | Trust reranker for marginal results |

**Design Philosophy:**
- Prevents re-ranking from eliminating documents with strong retrieval signals
- Allows semantic understanding to influence borderline cases
- Progressively increases reranker influence for lower-ranked candidates

---

## 3. Score Normalization

Different backends produce incompatible score ranges. Normalization ensures fair comparison in RRF fusion.

**BM25 Score Normalization:**

```typescript
function normalizeBM25(score: number): number {
  // BM25 scores are negative in SQLite FTS5 (lower = better)
  // Typical range: -15 (excellent) to -2 (weak match)
  // Map to 0-1 range where higher is better
  const absScore = Math.abs(score);

  // Sigmoid-ish normalization: maps ~2-15 range to ~0.1-0.95
  return 1 / (1 + Math.exp(-(absScore - 5) / 3));
}
```

**Vector Distance Normalization:**

```typescript
// Cosine distance: 0.0 (identical) to 2.0 (opposite)
// Convert to similarity score: 0-1 range
const similarity = 1 / (1 + distance);
```

**Min-Max Scaling (Post-Retrieval):**

```typescript
function normalizeScores(results: SearchResult[]): SearchResult[] {
  if (results.length === 0) return results;

  const maxScore = Math.max(...results.map(r => r.score));
  const minScore = Math.min(...results.map(r => r.score));
  const range = maxScore - minScore || 1;

  return results.map(r => ({
    ...r,
    score: (r.score - minScore) / range
  }));
}
```

**Reranker Score Normalization:**

```typescript
// Qwen3-Reranker outputs 0-10 rating
// Normalize to 0-1 range
const normalizedScore = score / 10;
```

**Score Range Summary:**

| Backend | Raw Range | Normalized Range | Method |
|---------|-----------|------------------|--------|
| FTS5 (BM25) | -15 to -2 (negative) | 0.0-1.0 | Sigmoid transform |
| Vector | 0.0-2.0 (distance) | 0.0-1.0 | `1 / (1 + distance)` |
| Reranker | 0-10 (rating) | 0.0-1.0 | Division by 10 |
| RRF Output | Variable | 0.0-1.0 | Min-max scaling |

---

## 4. Embedding Caching

QMD uses content-based deduplication rather than explicit caching.

**Content Hash Strategy:**

```typescript
// Embeddings are stored with hash_seq key: {content_hash}_{sequence_number}
// Identical content produces identical hash
// Database lookup identifies which hashes already have embeddings

function getHashesForEmbedding(db: Database): string[] {
  // Returns only hashes that lack embeddings
  const result = db.prepare(`
    SELECT DISTINCT d.hash
    FROM documents d
    LEFT JOIN content_vectors cv ON d.hash = cv.hash
    WHERE cv.hash IS NULL
  `).all();

  return result.map(row => row.hash);
}
```

**Benefits:**
- Automatic deduplication for repeated content
- No explicit cache expiration needed
- Database serves as persistent embedding store

**Inactivity Management:**

```typescript
// LLM contexts (not embeddings) are disposed after inactivity
// Default: 2 minutes timeout
// Models remain loaded in VRAM to avoid reallocation cost

const llm = new LlamaCpp({
  disposeModelsOnInactivity: true,  // Context disposal
  inactivityTimeout: 120000,        // 2 minutes
  // Models themselves stay loaded until process exit
});
```

**Key Insight**: Separate model loading (expensive, persistent) from context creation (cheap, disposable).

---

## 5. MCP Server Implementation

QMD exposes its search capabilities through the Model Context Protocol for integration with Claude Desktop and Claude Code.

**Server Initialization:**

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "qmd",
  version: "1.0.0",
});

// Tool registration
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "qmd_search",
      description: "Fast keyword-based full-text search using BM25",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", description: "Max results", default: 10 },
          collection: { type: "string", description: "Collection name (optional)" },
        },
        required: ["query"],
      },
    },
    {
      name: "qmd_vsearch",
      description: "Semantic search using vector embeddings",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", default: 10 },
          collection: { type: "string" },
        },
        required: ["query"],
      },
    },
    {
      name: "qmd_query",
      description: "Hybrid search combining BM25, vectors, query expansion, and LLM reranking",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", default: 10 },
          collection: { type: "string" },
          context: { type: "string", description: "Additional context for query expansion" },
        },
        required: ["query"],
      },
    },
    {
      name: "qmd_get",
      description: "Retrieve single document by path or docid",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path or docid" },
          lines: { type: "string", description: "Line range (e.g., '10-20')" },
        },
        required: ["path"],
      },
    },
    {
      name: "qmd_multi_get",
      description: "Batch document retrieval (supports glob patterns and comma-separated lists)",
      inputSchema: {
        type: "object",
        properties: {
          paths: { type: "string", description: "Comma-separated paths or glob pattern" },
        },
        required: ["paths"],
      },
    },
    {
      name: "qmd_status",
      description: "Report index health, document counts, and collection metadata",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

// Request handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "qmd_search":
      // Execute BM25 search
      const results = await db.ftsSearch(args.query, args.limit);
      return formatResults(results);

    case "qmd_vsearch":
      // Execute vector search
      const vecResults = await db.vectorSearch(args.query, args.limit);
      return formatResults(vecResults);

    case "qmd_query":
      // Execute full hybrid pipeline
      const hybridResults = await performHybridSearch(
        args.query,
        args.limit,
        args.collection,
        args.context
      );
      return formatResults(hybridResults);

    case "qmd_get":
      // Single document retrieval
      const doc = await db.getDocument(args.path, args.lines);
      return formatDocument(doc);

    case "qmd_multi_get":
      // Batch retrieval
      const docs = await db.getMultipleDocuments(args.paths);
      return formatDocuments(docs);

    case "qmd_status":
      // System status
      const status = await db.getIndexStatus();
      return formatStatus(status);

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

**Response Format:**

```typescript
// MCP response structure (spec: 2025-06-18)
{
  content: [
    {
      type: "text",
      text: "Human-readable summary of results"
    },
    {
      type: "resource",
      resource: {
        uri: "qmd://path/to/document.md",
        mimeType: "text/markdown",
        text: "Document content..."
      }
    }
  ],
  structuredContent: {
    results: [
      {
        docid: 123,
        file: "/path/to/document.md",
        score: 0.85,
        snippet: "Relevant excerpt...",
        // ... additional metadata
      }
    ]
  }
}
```

**Resource URIs:**

- Documents accessible via `qmd://` URIs
- No explicit list endpoint (discovery through search)
- Support for line ranges: `qmd://path/to/file.md#L10-20`

**Protocol Features:**
- Structured input validation using Zod schemas
- Both human-readable and machine-parseable outputs
- Collection filtering (applied post-search)
- Persistent database connection for performance

---

## 6. Adaptable Patterns for JARVIS

### Pattern 1: Template-First with Semantic Fallback

**Current JARVIS**: Template matching with 0.7 threshold → model generation

**QMD Enhancement**: Add query expansion layer before model generation:

```python
# jarvis/intent.py enhancement
class IntentClassifier:
    def classify_with_expansion(self, query: str, context: Optional[str] = None):
        # Try template match first (fast path)
        template_match = self.match_template(query)
        if template_match.similarity >= 0.7:
            return template_match

        # Expand query for better recall
        expanded_queries = self.expand_query(query, context)

        # Try template matching on expansions
        for exp_query in expanded_queries:
            match = self.match_template(exp_query.text)
            if match.similarity >= 0.65:  # Lower threshold for expansions
                return match

        # Fall through to model generation
        return None
```

**Benefits:**
- Improved recall without always invoking large model
- Expansion can happen with smaller model (e.g., Qwen2.5-0.5B)
- Lower latency than full generation for near-matches

---

### Pattern 2: Reciprocal Rank Fusion for iMessage Search

**Current JARVIS**: Single-method search (keyword or semantic)

**QMD Enhancement**: Combine multiple signals with RRF:

```python
# integrations/imessage/search.py
def hybrid_imessage_search(
    query: str,
    limit: int = 50,
    filters: Optional[MessageFilters] = None
) -> List[Message]:
    """
    Hybrid search combining:
    1. Full-text search on message text (BM25-like via SQLite)
    2. Semantic search on embedded messages
    3. Metadata filtering (sender, date range, attachments)
    """

    # Parallel retrieval
    fts_results = search_messages_fts(query, limit=100, filters=filters)
    vec_results = search_messages_vector(query, limit=100, filters=filters)

    # RRF fusion with equal weights
    merged = reciprocal_rank_fusion(
        result_lists=[fts_results, vec_results],
        weights=[1.0, 1.0],
        k=60
    )

    return merged[:limit]

def reciprocal_rank_fusion(
    result_lists: List[List[Message]],
    weights: List[float],
    k: int = 60
) -> List[Message]:
    """Merge ranked lists using RRF algorithm."""
    scores = {}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, msg in enumerate(results):
            rrf_score = weight / (k + rank + 1)

            if msg.guid in scores:
                scores[msg.guid]["score"] += rrf_score
                scores[msg.guid]["best_rank"] = min(scores[msg.guid]["best_rank"], rank)
            else:
                scores[msg.guid] = {
                    "message": msg,
                    "score": rrf_score,
                    "best_rank": rank
                }

    # Add top-rank bonus
    for guid, data in scores.items():
        if data["best_rank"] == 0:
            data["score"] += 0.05
        elif data["best_rank"] <= 2:
            data["score"] += 0.02

    # Sort by score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["message"] for item in ranked]
```

**Benefits:**
- Better recall by combining complementary search methods
- Automatic deduplication across result lists
- Preserves high-confidence exact matches via top-rank bonus

---

### Pattern 3: Message Embedding Cache

**Current JARVIS**: No embedding caching (templates matched at runtime)

**QMD Enhancement**: Pre-compute and cache message embeddings:

```python
# integrations/imessage/embeddings.py
from hashlib import sha256
import sqlite3

def get_or_create_message_embedding(
    db: sqlite3.Connection,
    message: Message,
    embedding_model: EmbeddingModel
) -> np.ndarray:
    """
    Get cached embedding or compute if missing.
    Uses content hash to avoid recomputation.
    """
    # Hash message content
    content = f"{message.text}_{message.sender}_{message.chat_id}"
    content_hash = sha256(content.encode()).hexdigest()

    # Check cache
    cached = db.execute(
        "SELECT embedding FROM message_embeddings WHERE hash = ?",
        (content_hash,)
    ).fetchone()

    if cached:
        return np.frombuffer(cached[0], dtype=np.float32)

    # Compute and cache
    embedding = embedding_model.embed(content)
    db.execute(
        """
        INSERT INTO message_embeddings (hash, message_guid, embedding, created_at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (content_hash, message.guid, embedding.tobytes())
    )
    db.commit()

    return embedding

# Schema addition to chat.db (or separate JARVIS db)
def init_embedding_cache(db: sqlite3.Connection):
    db.execute("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            hash TEXT PRIMARY KEY,
            message_guid TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model_name TEXT DEFAULT 'all-MiniLM-L6-v2',
            created_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_message_guid
        ON message_embeddings(message_guid)
    """)
```

**Benefits:**
- Avoid recomputing embeddings for every search
- Content-based deduplication (identical messages share embeddings)
- Persistent cache across JARVIS sessions

---

### Pattern 4: Position-Aware Response Ranking

**Current JARVIS**: Template match or model generation (binary choice)

**QMD Enhancement**: Blend template confidence with model quality scores:

```python
# models/generator.py
def generate_with_confidence_blending(
    query: str,
    context: str,
    template_match: Optional[TemplateMatch] = None
) -> Response:
    """
    Generate response with position-aware blending of template and model scores.
    """
    if template_match and template_match.similarity >= 0.85:
        # Very high template confidence - trust it completely
        return Response(
            text=template_match.response,
            confidence=template_match.similarity,
            source="template"
        )

    # Generate with model
    model_response = model.generate(query, context)
    model_confidence = estimate_confidence(model_response)  # HHEM or similar

    if template_match:
        # Blend template and model based on template rank
        if template_match.similarity >= 0.7:
            # High template confidence - blend 70/30
            final_confidence = 0.7 * template_match.similarity + 0.3 * model_confidence
            # Could also blend response texts or use template as fallback
        else:
            # Lower template confidence - trust model more (40/60)
            final_confidence = 0.4 * template_match.similarity + 0.6 * model_confidence
    else:
        final_confidence = model_confidence

    return Response(
        text=model_response,
        confidence=final_confidence,
        source="blended" if template_match else "model"
    )
```

**Benefits:**
- Graceful degradation from templates to model generation
- More nuanced confidence estimates
- Could use HHEM to validate both template and model responses

---

### Pattern 5: Chunked Message History

**Current JARVIS**: Summarize or load full conversation history

**QMD Enhancement**: Chunk message history with overlap for context preservation:

```python
# integrations/imessage/chunking.py
def chunk_conversation(
    messages: List[Message],
    chunk_size: int = 50,  # messages per chunk
    overlap: int = 5       # overlapping messages
) -> List[ConversationChunk]:
    """
    Split conversation into overlapping chunks for better context.
    Similar to QMD's 800 token chunks with 15% overlap.
    """
    chunks = []

    for i in range(0, len(messages), chunk_size - overlap):
        chunk_messages = messages[i:i + chunk_size]

        chunks.append(ConversationChunk(
            messages=chunk_messages,
            start_idx=i,
            end_idx=min(i + chunk_size, len(messages)),
            summary=None  # Can be computed lazily
        ))

    return chunks

def search_chunked_conversation(
    query: str,
    conversation: Conversation,
    limit: int = 3
) -> List[ConversationChunk]:
    """
    Search for relevant chunks instead of individual messages.
    Provides better context for response generation.
    """
    chunks = chunk_conversation(conversation.messages)

    # Embed each chunk (cache by hash of message GUIDs)
    chunk_embeddings = [embed_chunk(chunk) for chunk in chunks]
    query_embedding = embed_query(query)

    # Rank by similarity
    similarities = [
        cosine_similarity(query_embedding, chunk_emb)
        for chunk_emb in chunk_embeddings
    ]

    # Return top-k chunks
    ranked_indices = np.argsort(similarities)[::-1][:limit]
    return [chunks[i] for i in ranked_indices]
```

**Benefits:**
- Better context preservation than single messages
- Overlap prevents information loss at boundaries
- More efficient than embedding every message individually

---

### Pattern 6: MCP Server for JARVIS

**Current JARVIS**: CLI + FastAPI (for Tauri frontend)

**QMD Enhancement**: Add MCP server for Claude Code integration:

```python
# jarvis/mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource

class JarvisMCPServer:
    def __init__(self):
        self.server = Server("jarvis")
        self._register_tools()

    def _register_tools(self):
        """Register JARVIS capabilities as MCP tools."""
        self.server.add_tool(Tool(
            name="jarvis_search",
            description="Search iMessage conversations with hybrid BM25+vector search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "number", "default": 10},
                    "sender": {"type": "string"},
                    "start_date": {"type": "string"},
                    "has_attachment": {"type": "boolean"}
                },
                "required": ["query"]
            }
        ))

        self.server.add_tool(Tool(
            name="jarvis_reply",
            description="Generate reply suggestion for iMessage conversation",
            inputSchema={
                "type": "object",
                "properties": {
                    "contact": {"type": "string"},
                    "instruction": {"type": "string"},
                    "tone": {"type": "string", "enum": ["casual", "professional"]}
                },
                "required": ["contact"]
            }
        ))

        self.server.add_tool(Tool(
            name="jarvis_summarize",
            description="Summarize iMessage conversation",
            inputSchema={
                "type": "object",
                "properties": {
                    "contact": {"type": "string"},
                    "num_messages": {"type": "number", "default": 100}
                },
                "required": ["contact"]
            }
        ))

        self.server.add_tool(Tool(
            name="jarvis_status",
            description="Get JARVIS system health and model status",
            inputSchema={"type": "object", "properties": {}}
        ))

    async def handle_request(self, tool_name: str, args: dict):
        """Route tool requests to JARVIS capabilities."""
        if tool_name == "jarvis_search":
            results = search_messages(
                query=args["query"],
                limit=args.get("limit", 10),
                sender=args.get("sender"),
                start_date=args.get("start_date"),
                has_attachment=args.get("has_attachment")
            )
            return self._format_search_results(results)

        elif tool_name == "jarvis_reply":
            suggestion = generate_reply(
                contact=args["contact"],
                instruction=args.get("instruction"),
                tone=args.get("tone", "casual")
            )
            return self._format_reply(suggestion)

        elif tool_name == "jarvis_summarize":
            summary = summarize_conversation(
                contact=args["contact"],
                num_messages=args.get("num_messages", 100)
            )
            return self._format_summary(summary)

        elif tool_name == "jarvis_status":
            status = get_system_health()
            return self._format_status(status)

    def _format_search_results(self, results: List[Message]) -> dict:
        """Format search results for MCP response."""
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Found {len(results)} messages"
                )
            ],
            "structuredContent": {
                "messages": [
                    {
                        "guid": msg.guid,
                        "text": msg.text,
                        "sender": msg.sender,
                        "timestamp": msg.timestamp.isoformat(),
                        "score": msg.relevance_score
                    }
                    for msg in results
                ]
            }
        }

    async def run_stdio(self):
        """Run MCP server on stdio transport."""
        from mcp.server.stdio import stdio_server
        async with stdio_server() as streams:
            await self.server.run(streams)
```

**CLI Integration:**

```python
# jarvis/cli.py
@click.command()
@click.option("--transport", type=click.Choice(["stdio", "http"]), default="stdio")
@click.option("--port", type=int, default=8765)
def mcp_serve(transport: str, port: int):
    """Start JARVIS MCP server for Claude Code integration."""
    server = JarvisMCPServer()

    if transport == "stdio":
        asyncio.run(server.run_stdio())
    else:
        asyncio.run(server.run_http(port=port))
```

**Benefits:**
- Native integration with Claude Code
- Expose JARVIS capabilities as tools
- Maintain existing CLI/API alongside MCP

---

## 7. Implementation Recommendations

### High Priority (Immediate Impact)

1. **RRF for iMessage Search** (Pattern 2)
   - Combine FTS and vector search with RRF
   - Estimated effort: 2-3 days
   - Benefit: Significantly better search recall and precision

2. **Message Embedding Cache** (Pattern 3)
   - Add embedding cache table to avoid recomputation
   - Estimated effort: 1-2 days
   - Benefit: Faster search, lower latency

3. **MCP Server** (Pattern 6)
   - Expose JARVIS as MCP server for Claude Code
   - Estimated effort: 2-3 days
   - Benefit: Better integration with LLM workflows

### Medium Priority (Quality Improvements)

4. **Query Expansion** (Pattern 1)
   - Add lightweight query expansion before model generation
   - Estimated effort: 3-4 days
   - Benefit: Better template recall, fewer model invocations

5. **Chunked History** (Pattern 5)
   - Implement conversation chunking for better context
   - Estimated effort: 2-3 days
   - Benefit: More relevant context for reply/summary generation

### Lower Priority (Advanced Features)

6. **Position-Aware Blending** (Pattern 4)
   - Blend template and model confidence scores
   - Estimated effort: 2-3 days
   - Benefit: More nuanced response selection

7. **LLM Re-ranking**
   - Add re-ranking stage after RRF fusion
   - Estimated effort: 4-5 days
   - Benefit: Better final ranking, but highest memory cost

---

## 8. Key Takeaways

### What QMD Does Well

1. **Hybrid Search Excellence**: Three-stage pipeline balances precision and recall
2. **Smart Fusion**: RRF with position-aware blending preserves high-confidence matches
3. **Efficient Caching**: Content-based deduplication eliminates redundant computation
4. **Graceful Degradation**: Progressive trust in different signals based on position
5. **Protocol Integration**: MCP server exposes capabilities to AI assistants

### Lessons for JARVIS

1. **Don't Force Re-ranking**: Position-aware blending lets retrieval signals shine
2. **Multi-Signal Fusion**: Combine complementary methods (BM25 + vector + metadata)
3. **Cache Everything**: Pre-compute embeddings, cache expansions, store contexts
4. **Chunking Matters**: Overlap prevents information loss at boundaries
5. **Model Separation**: Keep embedding, expansion, and re-ranking models separate

### Architectural Differences

| Aspect | QMD | JARVIS (Current) | JARVIS (Enhanced) |
|--------|-----|------------------|-------------------|
| Search Method | BM25 + Vector + LLM | Template matching + Model | RRF fusion of FTS + Vector |
| Response Gen | LLM re-ranking | Template-first with fallback | Position-aware blending |
| Caching | Content hash embeddings | Runtime template matching | Pre-computed message embeddings |
| Context | 800-token chunks with overlap | Full conversation or summary | Chunked history with overlap |
| Integration | MCP server | CLI + FastAPI | CLI + FastAPI + MCP |
| Memory | ~3.1GB (3 models) | ~5.5GB (1 model + full stack) | ~6GB (add embedding model) |

---

## 9. Code References

### TypeScript Implementation Files

- **Query Expansion**: `src/llm.ts` - `expandQuery()` method
- **RRF Fusion**: `src/qmd.ts` - `reciprocalRankFusion()` function
- **Position-Aware Blending**: `src/qmd.ts` - Final score computation
- **Vector Index**: `src/store.ts` - `ensureVecTableInternal()`
- **MCP Server**: `src/mcp.ts` - Tool registration and request handling
- **Chunking**: `src/store.ts` - Document processing with overlap
- **Score Normalization**: `src/qmd.ts` - `normalizeBM25()`, `normalizeScores()`

### Model Configuration

```typescript
// Default models (from src/llm.ts)
const DEFAULT_MODELS = {
  embedding: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
  generation: "hf:ggml-org/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf",
  reranking: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
};

// Memory footprint (approximate)
// - EmbeddingGemma: ~300MB
// - Qwen3-1.7B: ~2.2GB
// - Qwen3-Reranker: ~640MB
// Total: ~3.1GB
```

---

## 10. Next Steps

1. **Prototype RRF Search** (1 week)
   - Implement RRF fusion for iMessage search
   - Benchmark against current search (precision/recall)
   - Measure latency impact

2. **Add Embedding Cache** (3 days)
   - Create message_embeddings table
   - Pre-compute embeddings for recent messages
   - Update search to use cached embeddings

3. **Evaluate Memory Budget** (2 days)
   - Measure memory impact of additional embedding model
   - Determine if 8GB target is feasible with RRF
   - Consider model size tradeoffs (e.g., smaller embedding model)

4. **Design MCP Integration** (1 week)
   - Define JARVIS MCP tools
   - Implement server with stdio transport
   - Test with Claude Code

5. **Query Expansion Experiment** (1 week)
   - Implement lightweight expansion with Qwen2.5-0.5B
   - Measure impact on template recall
   - Compare latency vs. full model generation

---

## Appendix: QMD Repository Structure

```
qmd/
├── src/
│   ├── qmd.ts           # Main search pipeline (RRF, blending, normalization)
│   ├── llm.ts           # LLM initialization, query expansion, caching
│   ├── store.ts         # sqlite-vec integration, embedding storage
│   ├── mcp.ts           # MCP server implementation
│   ├── collections.ts   # Collection management
│   └── formatter.ts     # Output formatting
├── test/
│   ├── qmd.test.ts
│   ├── llm.test.ts
│   ├── store.test.ts
│   └── mcp.test.ts
├── package.json
├── tsconfig.json
├── README.md
├── CLAUDE.md            # Architecture guidelines
└── example-index.yml
```

**Lines of Code**: ~3000 TypeScript (estimated from file sizes)

---

## References

- **QMD Repository**: https://github.com/tobi/qmd
- **sqlite-vec**: https://github.com/asg017/sqlite-vec
- **Model Context Protocol**: https://github.com/modelcontextprotocol/protocol
- **Reciprocal Rank Fusion Paper**: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **HyDE Paper**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"

---

**Document Version**: 1.0
**Author**: JARVIS Research (via Claude Code)
**Last Updated**: 2026-01-27
