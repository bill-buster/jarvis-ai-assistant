# Unified Implementation Plan

This plan tracks multi-phase routing/optimization work across chats.

## Phase 0: Metrics & Observability (Foundation)

Status: Done

Deliverables:
- `jarvis/metrics_router.py` routing metrics store in SQLite
- Router emits routing metrics with latency breakdowns
- `scripts/analyze_routing_metrics.py` for analysis

Success criteria:
- Metrics DB created at `~/.jarvis/metrics.db`
- Per-request routing decisions and latencies recorded

## Phase 1: Query Embedding Cache (Quick Win)

Status: Done

Deliverables:
- `CachedEmbedder` per-request cache in `jarvis/embedding_adapter.py`
- Router uses cached embedder across intent, message classification, and FAISS

Success criteria:
- Multiple embedding calls per request collapse to one compute per unique text

## Phase 2: Configurable Thresholds + A/B Testing

Status: Done

Deliverables:
- `routing` config section in `jarvis/config.py`
- A/B thresholds lookup in router
- `scripts/eval_thresholds.py` grid search on holdout pairs

Success criteria:
- Thresholds configurable via `routing.*` config
- A/B group thresholds override defaults

## Phase 2.5: Data Quality Improvements

Status: Done

Deliverables:
- Topic shift detection in `jarvis/extract.py` (filters "btw", "anyway", etc.)
- Response length constraints in `jarvis/prompts.py` (avg_length in template)
- Formal greeting removal in `jarvis/router.py` (strips "Hey!", "Hi there!", etc.)
- Response trimming for overly long generations (2x expected length cap)
- Context-aware acknowledgment routing via `_should_generate_after_acknowledgment()`
- `get_pairs_by_trigger_pattern()` helper in `jarvis/db.py`

Success criteria:
- GOOD pair quality >= 70% (was 49.8%)
- Length ratio <= 1.5x (was 3.6x)
- Semantic similarity >= 0.65 (was 0.576)
- Acknowledgment route <= 20% (was 31%)

Verification:
```bash
uv run python -m scripts.score_pair_quality --analyze --limit 500
uv run python -m scripts.eval_pipeline --limit 100
```

## Phase 3: CrossEncoder Re-Ranking (Optional)

Status: Pending

Prerequisite:
- Metrics show >15% template false positives

Planned work:
- `jarvis/reranker.py` with lazy-loaded cross-encoder
- Router uses reranking for borderline similarities

## Phase 4: Smart Model Loading

Status: Done

Deliverables:
- `jarvis/model_warmer.py` with `ModelWarmer` class
- `get_model_warmer()` singleton accessor
- `get_warm_generator()` convenience function
- Config options: `idle_timeout_seconds`, `warm_on_startup` in `jarvis/config.py`
- Integration with API lifecycle (start on startup, stop on shutdown)
- Memory pressure callback integration for emergency unloads

Success criteria:
- Model unloads after configurable idle timeout (default 5 minutes)
- Model warmer respects memory controller modes (FULL/LITE/MINIMAL)
- All API generation endpoints use `get_warm_generator()` to touch warmer

## Phase 5: Incremental FAISS Updates

Status: Done

Deliverables:
- `IncrementalTriggerIndex` class in `jarvis/index.py`
- `IncrementalIndexConfig` and `IncrementalIndexStats` dataclasses
- `get_incremental_index()` singleton accessor
- Incremental add/remove without full rebuilds
- Soft-delete with automatic skip during search
- `compact()` method to rebuild when deletion ratio exceeds threshold
- `sync_with_db()` to sync index with database state
- Persistent storage of metadata (deleted IDs, mappings)

Success criteria:
- New pairs can be added without full index rebuild
- Deleted pairs are soft-deleted and skipped during search
- `needs_compact()` returns True when deletion ratio >= 20%
- Index state persists across restarts

## Verification

Current baseline checks:
- `make verify`
- `uv run python -m scripts.analyze_routing_metrics`
- `uv run python -m scripts.eval_thresholds`

## Refactor Backlog

Candidates to reduce drift and tech debt:

1. ~~Config alignment~~ (DONE)
   - Problem: `template_similarity_threshold` and `routing.template_threshold` can diverge.
   - Solution: Deprecated top-level field. Non-default values are migrated to `routing.template_threshold` during config load (v8 migration).

2. ~~Routing metrics write strategy~~ (DONE)
   - Problem: Per-request SQLite connection may add overhead/locks.
   - Solution: Added `RoutingMetricsStore` with buffered writes. Metrics are queued
     in memory and flushed in batches (by count or time interval) via background thread.
     Uses `executemany` for efficient batch inserts.

3. ~~Embedder cache cohesion~~ (DONE)
   - Problem: Template matcher caches query embeddings separately from per-request cache.
   - Solution: Added `embedder` parameter to `TemplateMatcher.match()`, `match_with_context()`,
     and generator methods. When a `CachedEmbedder` is passed, it's used directly instead of
     internal cache, enabling cache sharing across the request pipeline.

4. ~~Generator test alignment~~ (DONE)
   - Problem: mocks do not accept `top_p`/`top_k`, causing failures.
   - Solution: Updated 4 test mocks in `test_generator.py` to use `**kwargs` instead
     of hardcoded positional defaults.

5. ~~Known issues doc consolidation~~ (DONE)
   - Problem: `docs/known_issues.md` duplicates `docs/EVALUATION_AND_KNOWN_ISSUES.md`.
   - Solution: Merged all content into `docs/EVALUATION_AND_KNOWN_ISSUES.md` with
     unified structure (platform reqs, known issues by priority, feature limitations,
     performance, pair quality, workarounds). Deleted `docs/known_issues.md`.
