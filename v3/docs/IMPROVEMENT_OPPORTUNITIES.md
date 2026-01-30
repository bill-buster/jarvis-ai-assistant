# V3 Improvement Opportunities

Analysis of features from root/v2 that could improve v3 reply generation.

## Current V3 Output Issues

Looking at the test output:
- RAG found 5 past replies (working!)
- Style analysis ran (0ms - cached)
- Global style ran (44ms)
- **BUT the prompt doesn't include style instructions!**

The prompt sent was just:
```
casual texts:

them: my head still hurts
me: Oh ok well that's tough
...
(free) them: And my head kinda hurts a bit actually now
me:
```

No style, no relationship context, no tone guidance.

---

## Top 5 Features to Add (Priority Order)

### 1. FIX PROMPT TO USE STYLE (Immediate - Already Computed!)
- **Location**: `v3/core/generation/prompts.py` and `reply_generator.py`
- **Issue**: Style analysis runs but isn't injected into prompt
- **Effort**: Low (1-2 hours)
- **Impact**: High - style data is computed but wasted

### 2. Coherence Detection (Filter Topic Drift)
- **Location**: `/Users/jwalinshah/coding/jarvis-ai-assistant/v2/core/generation/coherence.py`
- **What it does**: Detects topic changes using embedding similarity + time gaps
- **Effort**: Low (~150 lines to adapt)
- **Impact**: High - prevents mixing unrelated context

### 3. Reply Quality Scoring
- **Location**: `/Users/jwalinshah/coding/jarvis-ai-assistant/v2/core/generation/multi_generator.py`
- **What it does**: `score_reply()` checks length, repetition, meta-commentary
- **Effort**: Low (~50 lines of heuristics)
- **Impact**: Medium - filters bad candidates before showing

### 4. Intent Classification (Semantic)
- **Location**: `/Users/jwalinshah/coding/jarvis-ai-assistant/jarvis/intent.py`
- **What it does**: 7 intents + group-specific, semantic matching, parameter extraction
- **Effort**: Medium (820 lines, needs integration)
- **Impact**: High - enables intent-specific prompts

### 5. Template Fast Path
- **Location**: `/Users/jwalinshah/coding/jarvis-ai-assistant/models/templates.py`
- **What it does**: Semantic template matching, skips model for common patterns
- **Effort**: Medium (2196 lines, mature system)
- **Impact**: High for latency - 50ms vs 2-3s for matches

---

## What V3 Already Has (Working)

| Component | Status | Notes |
|-----------|--------|-------|
| RAG/Embeddings | ✅ Working | 336k messages indexed |
| Style Analyzer | ✅ Computed | But not in prompt! |
| Global Styler | ✅ Computed | But not in prompt! |
| Contact Profiler | ✅ Working | Relationship detection |
| Context Analyzer | ✅ Basic | Heuristic intent detection |

---

## Recommended Action

**FIRST**: Fix the prompt to include style instructions (already computed, just not used)

**THEN**: Add coherence detection to filter irrelevant context

**LATER**: Add quality scoring and better intent classification

---

## Files to Review

1. `v3/core/generation/prompts.py` - Current prompt building (needs style injection)
2. `v3/core/generation/reply_generator.py` - See where style is computed but not passed
3. `v2/core/generation/coherence.py` - Topic drift detection (copy this)
4. `v2/core/generation/multi_generator.py` - Quality scoring heuristics
