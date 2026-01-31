# JARVIS Codebase Review for Strategic Planning

**Generated:** 2026-01-30
**Version:** 1.0.0
**Status:** Beta

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Implementation Status](#2-implementation-status)
3. [Current Performance](#3-current-performance)
4. [Technical Debt & Blockers](#4-technical-debt--blockers)
5. [Test Coverage](#5-test-coverage)
6. [Dependencies & Constraints](#6-dependencies--constraints)
7. [Open Questions](#7-open-questions)

---

## 1. Architecture Overview

### Directory Structure

```
jarvis-ai-assistant/
├── contracts/           # 9 Protocol interfaces (interface layer)
├── core/               # Health monitoring and memory controllers
│   ├── health/         # Circuit breaker, permissions, schema detection
│   └── memory/         # Three-tier memory controller
├── models/             # MLX model loading, generation, embeddings
├── integrations/       # External integrations
│   ├── imessage/       # iMessage reader (50KB main file)
│   └── calendar/       # Calendar integration
├── jarvis/             # Main CLI module (39 files, ~1.7MB)
├── api/                # FastAPI REST backend (28 routers)
├── mcp_server/         # Model Context Protocol for Claude Code
├── desktop/            # Tauri + Svelte desktop UI
├── benchmarks/         # Performance validation gates
├── tests/              # Unit (45) + Integration (7) tests
├── scripts/            # Utility scripts (40+)
├── docs/               # Design docs, audit reports
└── results/            # Benchmark outputs
```

### Key Components and Responsibilities

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **CLI Entry Point** | `jarvis/cli.py` | 40+ commands: chat, reply, search, summarize, serve |
| **Intent Classifier** | `jarvis/intent.py` | Semantic routing (REPLY, SUMMARIZE, SEARCH, GROUP_*) |
| **Reply Router** | `jarvis/router.py` | Three-tier routing: template/generate/clarify |
| **MLX Generator** | `models/generator.py` | LLM text generation with template fallback |
| **Template Matcher** | `models/templates.py` | 25+ templates with semantic similarity |
| **MLX Embedder** | `models/embeddings.py` | GPU-accelerated embeddings via microservice |
| **iMessage Reader** | `integrations/imessage/reader.py` | Read-only chat.db access, schema detection |
| **JARVIS Database** | `jarvis/db.py` | Contacts, pairs, clusters, FAISS metadata |
| **FAISS Index** | `jarvis/index.py` | Trigger similarity search |
| **Pair Extractor** | `jarvis/extract.py` | Mining (trigger, response) pairs from messages |
| **Memory Controller** | `core/memory/controller.py` | FULL/LITE/MINIMAL modes based on RAM |
| **Circuit Breaker** | `core/health/degradation.py` | Graceful degradation (CLOSED/OPEN/HALF_OPEN) |
| **FastAPI Backend** | `api/main.py` | REST API for Tauri desktop app |

### Data Flow

```
User Input → Intent Classification → Context Fetching → Reply Router
                                                              │
                    ┌────────────────────────────────────────┼────────────────┐
                    │                                        │                │
                    ▼                                        ▼                ▼
            [score ≥ 0.90]                          [0.50-0.90]          [< 0.50]
            Template Match                          LLM Generate         Clarify
            (FAISS lookup)                          (MLX model)          (Ask user)
                    │                                        │
                    └────────────────────────────────────────┘
                                        │
                                        ▼
                               Quality Scoring → Response
```

---

## 2. Implementation Status

### Fully Implemented and Working (100%)

| Component | Status | Notes |
|-----------|--------|-------|
| **Contracts/Interfaces** | ✅ COMPLETE | 9 protocols in `contracts/` |
| **CLI Entry Point** | ✅ COMPLETE | 40+ commands working |
| **iMessage Reader (WS10)** | ✅ COMPLETE | Schema detection, attachments, reactions |
| **Model Generator (WS8)** | ✅ COMPLETE | MLX loader, template fallback, RAG |
| **Model Registry** | ✅ COMPLETE | 0.5B/1.5B/3B/LFM tiers, auto-selection |
| **Memory Controller (WS5)** | ✅ COMPLETE | FULL/LITE/MINIMAL modes |
| **Degradation Controller (WS6)** | ✅ COMPLETE | Circuit breaker pattern |
| **Memory Profiler (WS1)** | ✅ COMPLETE | MLX memory profiling |
| **HHEM Benchmark (WS2)** | ✅ COMPLETE | Vectara HHEM model evaluation |
| **Latency Benchmark (WS4)** | ✅ COMPLETE | Cold/warm/hot scenarios |
| **Setup Wizard** | ✅ COMPLETE | Environment validation, config init |
| **Intent Classification** | ✅ COMPLETE | Semantic similarity routing |
| **Prompts Registry** | ✅ COMPLETE | Centralized prompt templates |
| **Error Handling** | ✅ COMPLETE | Unified exception hierarchy |
| **Metrics System** | ✅ COMPLETE | Prometheus-compatible metrics |
| **Export System** | ✅ COMPLETE | JSON/CSV/TXT export |
| **MLX Embeddings** | ✅ COMPLETE | GPU-accelerated on Apple Silicon |
| **FastAPI Backend** | ✅ COMPLETE | 28 routers for Tauri integration |
| **FAISS Trigger Index** | ✅ COMPLETE | Versioned vector search |
| **JARVIS Database** | ✅ COMPLETE | 40,061 pairs indexed |
| **Reply Router** | ✅ COMPLETE | Template/generate/clarify routing |
| **Pair Extraction** | ✅ COMPLETE | Mining from chat.db |

### Partially Implemented (~75%)

| Component | Status | Remaining Work |
|-----------|--------|----------------|
| **Cluster Analysis** | 🟡 75% | 0 clusters in DB despite 40K pairs |
| **Continuous Learning** | 🟡 50% | TODO in `scripts/utils/continuous_learning.py` |
| **Permission Checking** | 🟡 80% | Non-FDA permissions stub to `True` |
| **Calendar Integration** | 🟡 70% | Reader/detector done, writer untested |

### Stubbed/TODO Items

| Item | Location | Description |
|------|----------|-------------|
| Continuous learning update | `scripts/utils/continuous_learning.py:299` | Pattern mining with new messages not implemented |
| Search evaluation benchmark | `scripts/overnight_embedding_comparison.sh:83` | Python script not created |
| Intent evaluation benchmark | `scripts/overnight_embedding_comparison.sh:103` | Python script not created |
| Comparison scripts | Multiple shell scripts | Baseline template comparison not implemented |

### Deprecated/Experimental

| Component | Status | Notes |
|-----------|--------|-------|
| **IMessageSender** | ⚠️ DEPRECATED | AppleScript automation unreliable |
| **BitNet loader** | 🗑️ REMOVED | Files deleted (see git status) |
| **Template Coverage (WS3)** | 🔀 MOVED | Merged into `models/templates.py` |

---

## 3. Current Performance

### Latest Benchmark Results (2026-01-29/30)

#### Database & Index Metrics
| Metric | Value |
|--------|-------|
| Total Message Pairs Extracted | 40,061 |
| iMessages Scanned | 309,000 |
| Extraction Time | ~8 seconds |
| FAISS Index Vectors | 40,061 |
| Index Dimension | 384 |
| Index Size | 60MB |
| Embedding Model | BAAI/bge-small-en-v1.5 |

#### Overnight Evaluation (8 hours, 88 experiments)
| Metric | Value |
|--------|-------|
| Phases Completed | 18 |
| Total Experiments | 88 |
| **Recommended TEMPLATE_THRESHOLD** | 0.650 |
| **Recommended CLARIFY_THRESHOLD** | 0.450 |
| Routing: Template path | 100% |
| Routing: Generate path | 0% |
| Routing: Clarify path | 0% |
| Avg Response Similarity | 0.565 |
| Response Diversity Grade | **A** |
| Unique Responses | 18,106 |
| Normalized Entropy | 0.956 |
| Peak Memory | 1.54 GB |
| Average Memory | 1.05 GB |

#### Template vs LLM Evaluation (2000 samples)
| Metric | Template | LLM |
|--------|----------|-----|
| Win Rate | **81.1%** | 1.7% |
| Ties | 17.2% | - |
| Avg Similarity | 0.8269 | 0.5845 |
| Avg Coherence | 0.7498 | 0.2843 |
| Comparison Time | - | 672.9 ms/gen |

#### Improved Evaluation (500 samples, multi-metric)
| Metric | Template | LLM |
|--------|----------|-----|
| Win Rate | 14.6% | **76.6%** |
| Overall Score | 0.1726 | **0.6175** |
| Coherence (Fair) | 0.2595 | **0.9296** |
| Brevity Score | 0.2345 | **0.9143** |
| Naturalness Score | 0.264 | **0.9396** |

**Key Insight:** Template matching wins on similarity to actual responses, but LLM wins on coherence, brevity, and naturalness when evaluated fairly.

#### Classifier Pipeline Results (17 test cases)
| Category | Accuracy |
|----------|----------|
| Vague time tests | 100% (3/3) |
| Intent vague tests | 100% (3/3) |
| Ambiguous tests | 100% (2/2) |
| Clear intent tests | **0% (0/4)** |
| No-reply tests | **0% (0/3)** |
| Context tests | 50% (1/2) |
| **Overall** | **52.94%** |

**Critical Issue:** The classifier correctly asks for clarification when needed but fails to respond when intent is clear (100% asks-when-should-ask, 0% responds-when-should-respond).

### Validation Gates Status

| Gate | Metric | Target | Status |
|------|--------|--------|--------|
| G1 | Model stack memory | <5.5GB | ✅ (1.54GB peak) |
| G2 | Mean HHEM score | ≥0.5 | ⚠️ Not measured recently |
| G3 | Warm-start latency | <3s | ⚠️ Not measured recently |
| G4 | Cold-start latency | <15s | ⚠️ Not measured recently |

---

## 4. Technical Debt & Blockers

### Hardcoded Values Needing Configuration

#### Similarity Thresholds (40+ magic numbers)
| File | Constant | Value | Should Be Config |
|------|----------|-------|------------------|
| `jarvis/router.py` | TEMPLATE_THRESHOLD | 0.90 | Yes |
| `jarvis/router.py` | GENERATE_THRESHOLD | 0.50 | Yes |
| `jarvis/router.py` | CLARIFY_THRESHOLD | 0.70 | Yes |
| `jarvis/intent.py` | CONFIDENCE_THRESHOLD | 0.6 | Yes |
| `models/templates.py` | SIMILARITY_THRESHOLD | 0.7 | Yes |

**Recommendation:** Consolidate all thresholds into `jarvis/config.py` instead of scattered module-level constants. Overnight evaluation suggests optimal values: TEMPLATE=0.65, CLARIFY=0.45.

#### Database/Memory Limits
| File | Constant | Value |
|------|----------|-------|
| `jarvis/evaluation.py` | MAX_FEEDBACK_ENTRIES | 10000 |
| `jarvis/quality_metrics.py` | MAX_EVENTS | 10000 |
| `jarvis/metrics.py` | MAX_SIMILARITY_SAMPLES | 1000 |
| `models/templates.py` | QUERY_CACHE_SIZE | 500 |

### Missing Error Handling / Edge Cases

1. **Permission checking stub** (`jarvis/setup.py:117-119`): Non-FDA permissions default to `True`
2. **Calendar writer**: Untested in production scenarios
3. **Classifier pipeline**: Fails on clear intents (0% accuracy)

### Integration Points Not Working

1. **IMessageSender**: Deprecated due to Apple restrictions
   - Requires Automation permission
   - May be blocked by SIP
   - Requires Messages.app running
   - API instability across macOS versions

2. **Cluster Analysis**: 0 clusters despite 40,061 pairs
   - HDBSCAN clustering appears to not be running or failing silently

### Template Matching Coverage Issue

**Current State:**
- Overnight evaluation shows 100% template routing (no LLM generation triggered)
- This suggests thresholds are too permissive OR the FAISS index is matching everything
- The "improved eval" shows LLM actually produces better quality responses

**The 5% Coverage Issue:**
- Not explicitly documented in codebase
- May refer to coverage of unique response patterns vs total queries
- Needs investigation into `jarvis/router.py` routing logic

---

## 5. Test Coverage

### Summary Statistics

| Metric | Count |
|--------|-------|
| Total Test Files | 52 |
| Unit Test Files | 45 |
| Integration Test Files | 7 |
| Total Test Functions | 2,303 |
| Test Classes | 466 |
| Lines of Test Code | ~32,000 |
| Pytest Fixtures | 97 |

### Modules WITH Tests (36+)

All core contracts and their implementations have tests:
- `jarvis/config.py`, `intent.py`, `prompts.py`, `errors.py`, `metrics.py`
- `models/generator.py`, `registry.py`, `embeddings.py`, `templates.py`
- `integrations/imessage/*`
- `core/health/*`, `core/memory/*`
- `benchmarks/memory/*`, `hallucination/*`, `latency/*`

### Modules WITHOUT Dedicated Tests (13)

| Module | Risk | Notes |
|--------|------|-------|
| `jarvis/db.py` | HIGH | Database abstraction (43KB) |
| `jarvis/router.py` | HIGH | Reply routing logic (33KB) |
| `jarvis/index.py` | HIGH | FAISS trigger indexing (15KB) |
| `jarvis/extract.py` | MEDIUM | Pair extraction (24KB) |
| `jarvis/cluster.py` | MEDIUM | Clustering logic (14KB) |
| `models/loader.py` | MEDIUM | MLX model loading (24KB) |
| `jarvis/simple_reply.py` | LOW | New file (8KB) |
| `jarvis/api.py` | LOW | Tested via integration |
| `jarvis/cli.py` | LOW | Tested via integration |

### Failing/Skipped Tests

- **12 skipif markers**: MLX-dependent tests skip on non-Apple Silicon
- **1 xfail marker**: `model_dependent` in `test_cli.py` (output varies)
- **No known failing tests** in CI

### Golden Examples / Evaluation Datasets

- **No golden datasets committed** to repository
- Evaluation data stored in `results/` (not version controlled)
- Test fixtures create synthetic data inline

---

## 6. Dependencies & Constraints

### Core Dependencies

```toml
dependencies = [
    "mlx>=0.22.0",              # Apple Silicon ML framework
    "mlx-lm>=0.22.0",           # MLX language model utils
    "sentence-transformers>=5.0.0",  # Embedding models
    "psutil>=7.0.0",            # System monitoring
    "pydantic>=2.10.0",         # Data validation
    "rich>=14.0.0",             # CLI formatting
    "fastapi>=0.125.0",         # REST API
    "uvicorn[standard]>=0.35.0", # ASGI server
    "faiss-cpu>=1.9.0",         # Vector search
    "pyarrow>=18.0.0",          # Data serialization
]
```

### Development Dependencies

```toml
dev = [
    "pytest>=9.0.0",
    "pytest-cov>=7.0.0",
    "ruff>=0.14.0",
    "mypy>=1.15.0",
    "httpx>=0.28.0",
    "pytest-asyncio>=1.0.0",
]
```

### Benchmark Dependencies

```toml
benchmarks = [
    "transformers>=4.45.0",     # HuggingFace models
    "torch>=2.5.0",             # PyTorch (for HHEM)
    "hdbscan>=0.8.38",          # Clustering
    "scikit-learn>=1.6.1",
    "pandas>=2.3.0",
    "matplotlib>=3.10.0",
]
```

### Memory/Performance Constraints

| Constraint | Target | Current |
|------------|--------|---------|
| Memory budget | 8GB min | 1.54GB peak observed |
| Model stack | <5.5GB | ~1GB (LFM-2.5-1.2B) |
| Warm start | <3s | Not measured recently |
| Cold start | <15s | Not measured recently |

### Required Files/Configs

| File | Purpose | Location |
|------|---------|----------|
| iMessage database | Source data | `~/Library/Messages/chat.db` |
| JARVIS config | Settings | `~/.jarvis/config.json` |
| JARVIS database | Pairs, contacts | `~/.jarvis/jarvis.db` |
| FAISS index | Vector search | `~/.jarvis/indexes/` |
| HuggingFace cache | Models | `~/.cache/huggingface/hub` |

### Platform Requirements

- **macOS only** (iMessage integration)
- **Apple Silicon recommended** (MLX acceleration)
- **Full Disk Access** permission required
- Python 3.11+

---

## 7. Open Questions

### Architectural Decisions Unresolved

1. **Template vs LLM Trade-off**
   - Template wins on similarity (81%), LLM wins on quality metrics (77%)
   - Current routing sends 100% to template path
   - **Question:** Should we prefer LLM generation for better quality at cost of latency?

2. **Threshold Tuning**
   - Overnight eval recommends: TEMPLATE=0.65, CLARIFY=0.45
   - Code uses: TEMPLATE=0.90, CLARIFY=0.70
   - **Question:** Should thresholds be updated to recommended values?

3. **Cluster Analysis Not Running**
   - 40,061 pairs extracted but 0 clusters
   - **Question:** Is HDBSCAN being run? What are the clustering parameters?

4. **Embedding Service Architecture**
   - MLX and transformers have conflicting dependencies
   - Currently uses HTTP microservice on port 8766
   - **Question:** Is this architecture sustainable for desktop app?

### Comments/TODOs Indicating Uncertainty

1. **Continuous learning** (`scripts/utils/continuous_learning.py:299`)
   ```python
   # TODO: Remine patterns with new messages
   # This is a simplified version - full implementation would...
   ```

2. **Permission checking** (`jarvis/setup.py:117-119`)
   ```python
   # For other permissions, we'd need to attempt access
   # For now, mark as granted (placeholder for future implementation)
   granted = True
   ```

3. **Search/Intent evaluation** (`scripts/overnight_embedding_comparison.sh`)
   ```bash
   # TODO: Create benchmarks/search/evaluate.py script
   # TODO: Create benchmarks/intent/evaluate.py script
   ```

### Code Diverges from Documented Plans

1. **CLAUDE.md states:**
   - "Template matching coverage issue (currently ~5%)"
   - But overnight eval shows 100% template routing
   - **Discrepancy needs investigation**

2. **Router thresholds in docs vs code:**
   - Docs: "Template (similarity >= 0.85)"
   - `jarvis/router.py`: TEMPLATE_THRESHOLD = 0.90
   - **Values don't match**

3. **Default model:**
   - Registry shows LFM-2.5-1.2B as default
   - Some scripts still reference Qwen models
   - **Potential inconsistency in model selection**

### Critical Blockers for Production

1. **Classifier fails on clear intents** (0% accuracy)
   - System over-clarifies, never generates direct responses
   - Blocks real-world usability

2. **No clustering** despite pairs
   - Intent clustering not producing results
   - Affects response diversity analysis

3. **iMessage sender deprecated**
   - No reliable way to send messages programmatically
   - Limits to read-only assistant

---

## Recommendations for Next Steps

### Immediate (P0)

1. **Fix classifier pipeline** - Investigate why clear intents fail (0%)
2. **Run clustering** - Debug why HDBSCAN produces 0 clusters
3. **Update thresholds** - Apply overnight eval recommendations (0.65/0.45)

### Short-term (P1)

4. **Consolidate magic numbers** - Move 40+ thresholds to config
5. **Add missing tests** - Prioritize `db.py`, `router.py`, `index.py`
6. **Run full gate validation** - G2-G4 not measured recently

### Medium-term (P2)

7. **Implement continuous learning** - Complete the TODO
8. **Create evaluation benchmarks** - `search/evaluate.py`, `intent/evaluate.py`
9. **Fix permission checking** - Replace placeholder with real detection

### Long-term (P3)

10. **Resolve template vs LLM** - Determine optimal routing strategy
11. **Add golden datasets** - Commit evaluation data for reproducibility
12. **Document architectural decisions** - ADR for embedding microservice, etc.

---

*Report generated by Claude Code analysis. See `CLAUDE.md` for development guidelines.*
