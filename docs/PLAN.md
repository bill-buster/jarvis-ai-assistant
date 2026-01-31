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

Status: Pending

Planned work:
- `jarvis/model_warmer.py` to keep model warm for recent traffic
- Integrate with model loader unload policy

## Phase 5: Incremental FAISS Updates

Status: Pending

Planned work:
- `IncrementalTriggerIndex` in `jarvis/index.py`
- Persist incremental updates without full rebuilds

## Verification

Current baseline checks:
- `make verify`
- `uv run python -m scripts.analyze_routing_metrics`
- `uv run python -m scripts.eval_thresholds`

## Refactor Backlog

Candidates to reduce drift and tech debt:

1. Config alignment
   - Problem: `template_similarity_threshold` and `routing.template_threshold` can diverge.
   - Plan: Deprecate top-level field and map into `routing.template_threshold` during migration.

2. Routing metrics write strategy
   - Problem: Per-request SQLite connection may add overhead/locks.
   - Plan: Add buffered writer or a single queued writer thread.

3. Embedder cache cohesion
   - Problem: Template matcher caches query embeddings separately from per-request cache.
   - Plan: Allow template matcher to accept an embedder override to reuse per-request cache.

4. Generator test alignment
   - Problem: mocks do not accept `top_p`/`top_k`, causing failures.
   - Plan: Update test mocks to match sampler signature.

5. Known issues doc consolidation
   - Problem: `docs/known_issues.md` duplicates `docs/EVALUATION_AND_KNOWN_ISSUES.md`.
   - Plan: Merge and delete legacy file.
