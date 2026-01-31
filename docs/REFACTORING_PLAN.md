# JARVIS Codebase Refactoring Plan

**Version:** 1.0  
**Date:** 2026-01-30  
**Author:** AI Assistant Analysis  

---

## Executive Summary

This document outlines a comprehensive refactoring plan for the JARVIS codebase to improve modularity, maintainability, and code quality.

**Current State:**
- Total Python files: 180+
- Lines of code: ~45,000
- Test coverage: 72%
- Tests passing: 3,000/3,005 (5 pre-existing failures)

---

## Phase 1: Quick Wins (1-2 weeks)

### 1.1 Clean Up Experiment Scripts
**Priority:** HIGH  
**Effort:** 4-6 hours  
**Impact:** Remove 6,000+ lines of duplicate code

**Current State:**
scripts/experiments/ has 20 files totaling 9,193 lines with massive duplication:
- 5 different mine_response_pairs variants
- 3 overlapping reply test scripts
- No shared framework

**Target:** Consolidate to 3-4 scripts with shared framework

**Migration Steps:**
1. Create scripts/experiments/framework/ with base classes
2. Extract common utilities
3. Consolidate duplicate scripts
4. Delete obsolete variants
5. Update documentation

---

### 1.2 Simplify API Router Structure
**Priority:** HIGH  
**Effort:** 6-8 hours  
**Impact:** Reduce 35 routers to ~8 domain groups

**Current State:** 35 router files (20,473 lines)
**Target:** 8 domain-based routers

Group by:
- core: health, metrics, websocket
- messaging: conversations, threads, attachments, search  
- generation: drafts, suggestions
- templates: custom_templates, template_analytics
- insights: insights, stats, digest
- data: contacts, embeddings, topics
- tasks: tasks, batch
- config: settings, experiments, feedback

---

## Phase 2: Structural Improvements (2-4 weeks)

### 2.1 Refactor CLI Module
**Priority:** HIGH  
**Effort:** 12-16 hours  
**Current:** 3,151 lines in single file
**Target:** jarvis/cli/ package with separate command modules

Structure:
```
jarvis/cli/
├── __init__.py
├── parser.py              # Argument definitions
├── commands/
│   ├── chat.py
│   ├── reply.py
│   ├── summarize.py
│   ├── search.py
│   ├── health.py
│   ├── benchmark.py
│   ├── export.py
│   ├── batch.py
│   ├── tasks.py
│   ├── serve.py
│   └── db.py
└── utils.py               # Shared utilities
```

---

### 2.2 Refactor Prompts Module  
**Priority:** HIGH  
**Effort:** 8-12 hours  
**Current:** 1,940 lines mixing data and logic
**Target:** Separate data from logic

Structure:
```
jarvis/prompts/
├── __init__.py
├── models.py              # Dataclasses
├── data/                  # Static examples
│   ├── reply_examples.py
│   ├── summary_examples.py
│   └── search_examples.py
├── builders/              # Prompt building functions
│   ├── reply.py
│   ├── summary.py
│   └── search.py
├── style.py               # User style analysis
├── utils.py               # Token estimation
└── registry.py            # PromptRegistry class
```

---

## Phase 3: Database Refactoring (3-4 weeks)

### 3.1 Implement Repository Pattern
**Priority:** MEDIUM  
**Effort:** 20-24 hours  
**Current:** 1,800-line god class
**Target:** Repository pattern with clean separation

Structure:
```
jarvis/db/
├── __init__.py
├── models.py              # All dataclasses
├── schema.py              # SQL schema
├── base.py                # DatabaseConnection class
├── manager.py             # High-level coordinator
└── repositories/
    ├── contacts.py        # ContactRepository
    ├── pairs.py           # PairRepository
    ├── clusters.py        # ClusterRepository
    ├── embeddings.py      # EmbeddingRepository
    ├── indexes.py         # IndexRepository
    ├── artifacts.py       # ArtifactRepository
    └── splits.py          # SplitRepository
```

**Key Changes:**
- Move dataclasses to models.py
- Create separate Repository classes
- JarvisDB becomes a coordinator using repositories
- Maintain backward compatibility

---

## Phase 4: Advanced Refactoring (4-6 weeks)

### 4.1 Refactor Router Module
**Priority:** MEDIUM  
**Effort:** 16-20 hours  
**Current:** 1,415 lines
**Target:** Split routing, matching, and generation

Split ReplyRouter into:
- Router: Decision logic only
- TemplateMatcher: FAISS-based matching
- GenerationOrchestrator: LLM coordination
- ResponseSelector: Final response selection

---

### 4.2 Consolidate Utility Scripts
**Priority:** LOW  
**Effort:** 8-12 hours

Move from scripts/utils/ to proper packages:
- scripts/utils/coherence_checker.py -> jarvis/quality/
- scripts/utils/context_analysis.py -> jarvis/analysis/
- scripts/utils/continuous_learning.py -> jarvis/learning/

---

## Implementation Guidelines

### Testing Requirements
1. All changes must maintain test coverage
2. Run full test suite after each refactoring
3. Add tests for new modules
4. 3,000+ tests currently passing - maintain this

### Backward Compatibility
1. Keep __init__.py re-exports during transition
2. Deprecate old imports with warnings
3. Update documentation
4. Remove deprecated code in next major version

### Code Style
1. Follow existing patterns
2. Use type hints consistently
3. Document all public APIs
4. Keep line length <= 100 characters

---

## Estimated Timeline

**Phase 1 (Quick Wins):** 1-2 weeks
- Experiments cleanup: 4-6 hours
- API router consolidation: 6-8 hours

**Phase 2 (Structural):** 2-4 weeks  
- CLI refactoring: 12-16 hours
- Prompts refactoring: 8-12 hours

**Phase 3 (Database):** 3-4 weeks
- Repository pattern: 20-24 hours

**Phase 4 (Advanced):** 4-6 weeks
- Router refactoring: 16-20 hours
- Utilities consolidation: 8-12 hours

**Total:** 10-16 weeks (with testing and reviews)

---

## Success Metrics

1. **File Size Reduction**
   - cli.py: 3,151 -> ~500 lines (main entry)
   - prompts.py: 1,940 -> ~300 lines (exports only)
   - db.py: 1,800 -> ~400 lines (coordinator)

2. **Code Organization**
   - Average file size: <500 lines
   - Clear separation of concerns
   - Repository pattern for DB

3. **Maintainability**
   - Reduced duplication
   - Better testability
   - Clear module boundaries

4. **Quality Gates**
   - All tests passing
   - Coverage maintained at 72%+
   - No breaking changes

---

## Appendix: File Size Analysis

### God Files (>1000 lines)
- jarvis/cli.py: 3,151
- jarvis/prompts.py: 1,940
- jarvis/db.py: 1,800
- jarvis/extract.py: 1,447
- jarvis/relationships.py: 1,314
- jarvis/router.py: 1,415
- jarvis/errors.py: 1,113

### Medium Files (500-1000 lines)
- jarvis/metrics.py: 1,004
- jarvis/embeddings.py: 1,013
- jarvis/evaluation.py: 1,028
- jarvis/quality_metrics.py: 1,057
- jarvis/index.py: 1,162

### Target After Refactoring
- Maximum file size: 500 lines
- Average file size: ~200 lines
- Total files: ~250 (from ~180)
