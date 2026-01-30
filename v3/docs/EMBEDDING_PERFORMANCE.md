# Embedding Performance: Analysis & Optimization Strategies

## Current State (January 2026)

### Performance Numbers

| Metric | Before Preload | After Preload |
|--------|---------------|---------------|
| First query delay | ~15s | Instant |
| App startup time | ~1s | ~15s |
| Warm generation | 2,038ms | 1,395ms |
| RAG lookup | 10-30ms | 10-30ms |
| Fallback usage | 63% | 12% |
| RAG suggestions | 0% | 58% |

### Timing Breakdown

```
Cold Start (first request):
├── Embedding model (sentence-transformers): ~8-10s
├── FAISS index load/build: ~5-7s
└── Total: ~15s

Warm Start (subsequent requests):
├── Style analysis: 15ms
├── Context analysis: 8ms
├── RAG lookup (FAISS): 40-50ms
├── LLM generation: 400-500ms
└── Total: ~500-600ms
```

### Key Finding

**The embedding model itself is fast (~10-50ms)**. The slowness comes from:
1. **PyTorch/sentence-transformers initialization** (~8-10s)
2. **FAISS index loading from disk** (~5-7s)

## What Changed (Preloading Fix)

In v1/v2, the delay happened on first query but was masked by a "Computing semantic embeddings..." message. Users expected it as part of normal operation.

v3 now preloads at startup:

```python
# ReplyGenerator.__init__()
def __init__(self, model_loader, preload_embeddings: bool = True):
    if preload_embeddings:
        self._preload_embeddings()

def _preload_embeddings(self) -> None:
    # 1. Preload embedding model (~10s cold)
    model = get_embedding_model()
    if not model.is_loaded:
        model.preload()

    # 2. Preload reply-pairs FAISS index (~5-7s from disk)
    store = get_embedding_store()
    if store and store.is_reply_pairs_index_ready():
        store._get_or_build_reply_pairs_index()
```

**Trade-off**: Delay moved from first query to app startup. User sees loading at startup instead of mid-operation.

---

## Optimization Strategies (From Research)

### Strategy 1: MLX-Based Embedding Models

**Current**: sentence-transformers (PyTorch-based, slow init)
**Alternative**: MLX embedding models (memory-mapped, fast init)

Available MLX embedding models:
- `mlx-community/bge-small-en-v1.5` (~33M params, 128MB)
- `mlx-community/bge-base-en-v1.5` (~109M params, 420MB)
- `mlx-community/bge-m3` (multilingual, larger)

**Expected improvement**: Startup from ~8-10s → ~1-2s

**Considerations**:
- Need to re-embed all messages (one-time migration)
- BGE models produce 768-dim vectors vs MiniLM's 384-dim
- Quality should be similar or better (BGE is newer)

### Strategy 2: Content-Based Deduplication (Already Implemented)

```python
# EmbeddingCache uses SHA256 hashing
# Identical content = identical hash = single embedding
# Already in core/embeddings/cache.py
```

**Status**: ✅ Implemented - avoids re-computing embeddings for identical messages.

### Strategy 3: Hybrid Search with RRF (Already Implemented)

```python
# Combines BM25 (FTS5) + Vector (FAISS) search
# Reciprocal Rank Fusion merges results
# Already in core/embeddings/store.py
```

**Status**: ✅ Implemented - better recall without extra latency.

### Strategy 4: Two-Model Strategy (Recommended for Future)

**Mining (offline, one-time)**:
- Use high-quality model: `sentence-t5-large` or `BAAI/bge-large-en-v1.5`
- Takes ~4 hours to re-embed all messages
- Better template discovery, 95%+ clustering quality

**Production (real-time)**:
- Use fast model: `all-MiniLM-L6-v2` (current) or MLX BGE
- Templates stored as text, re-embedded at deploy
- No compatibility issues

### Strategy 5: Lazy Loading with Background Init

```python
# API startup:
# 1. Start server immediately (respond to health checks)
# 2. Background thread loads embedding model
# 3. First real query waits if needed, or uses cached

import threading

class LazyEmbeddings:
    def __init__(self):
        self._model = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._load)
        self._thread.start()

    def _load(self):
        self._model = load_embedding_model()
        self._ready.set()

    def embed(self, text):
        self._ready.wait()  # Block until ready
        return self._model.encode(text)
```

**Benefit**: Server starts instantly, loading happens in background.

---

## Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| LFM2.5-1.2B (LLM) | ~1.5GB | MLX, memory-mapped |
| Embedding model | ~90MB | all-MiniLM-L6-v2 |
| FAISS indices | ~50-100MB | Per-conversation cache |
| Working memory | ~500MB | Peak during generation |
| **Total** | ~2.2GB | Well within 8GB budget |

With MLX embedding model:
- Same ~90MB footprint
- Faster loading (memory-mapped vs PyTorch init)

---

## Implementation Priority

### Short-Term (Quick Wins)

1. **Background preloading** - Server starts instantly, loads in background
   - Effort: Small (threading wrapper)
   - Impact: Perceived instant startup

2. **Progress indication** - Show "Loading..." during startup
   - Effort: Tiny (API endpoint)
   - Impact: Better UX, matches v1/v2 behavior

### Medium-Term (Performance Gains)

3. **MLX embedding model** - Replace sentence-transformers
   - Effort: Medium (model swap, re-index)
   - Impact: ~6-8s faster startup

4. **Incremental FAISS updates** - Don't rebuild full index
   - Effort: Medium (FAISS add() instead of rebuild)
   - Impact: Faster subsequent loads

### Long-Term (Quality Improvements)

5. **Two-model strategy** - High-quality mining, fast production
   - Effort: Large (infrastructure change)
   - Impact: Better template discovery

---

## Benchmarks to Run

```bash
# Time embedding model load
uv run python -c "
import time
from core.embeddings.model import get_embedding_model
start = time.time()
model = get_embedding_model()
model.preload()
print(f'Embedding model load: {time.time() - start:.2f}s')
"

# Time FAISS index load
uv run python -c "
import time
from core.embeddings.store import get_embedding_store
start = time.time()
store = get_embedding_store()
# Force index load
store._get_or_build_reply_pairs_index()
print(f'FAISS index load: {time.time() - start:.2f}s')
"

# Full cold start
uv run python -c "
import time
from core.generation.reply_generator import ReplyGenerator
from core.models.loader import ModelLoader
start = time.time()
loader = ModelLoader('lfm2.5-1.2b')
gen = ReplyGenerator(loader)
print(f'Full cold start: {time.time() - start:.2f}s')
"
```

---

## Conclusion

The ~15s startup delay is primarily from PyTorch/sentence-transformers initialization, not the embedding model itself. The best path forward is:

1. **Immediate**: Background loading with progress indicator
2. **Near-term**: Evaluate MLX embedding models (bge-small-en-v1.5)
3. **Long-term**: Two-model strategy for quality + speed

Current implementation is functional with preloading. The delay is acceptable for app startup (users expect apps to load) vs mid-operation (feels broken).
